"""某个球员的个人视频：把他在场上的每一段剪出来，镜头跟着他（从 4K 原画裁切放大到 1080p）。

用法（先跑完 track、identify，最好也跑 ball）：
    .venv/bin/python -m bballgoal.reel videos/比赛.mp4 A#18
输出 output/track_<视频名>/reel_A18.mp4

画面上：他脚下一个圈和号码；看得到球时球上一个圈；他持球时圈变黄并显示 HAS BALL。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.ball import people_at
from bballgoal.frames import read_frames

OUT_W, OUT_H = 1920, 1080
MERGE_GAP = 2.0  # 秒：同一人两次出现间隔小于这个，剪成一段（中间短暂被挡住）
MIN_CLIP = 1.5  # 秒：短于这个的出现不剪（多半是一闪而过的误认）
PAD = 0.5  # 秒：每段前后各多留一点
VIEW_HEIGHTS = 5.0  # 画面高度 = 他身高的这么多倍（看得到身边的人和球）
SMOOTH = 0.12  # 镜头跟随的平滑系数（每帧向目标移动的比例），越小越稳
HOLD_COLOR = (0, 215, 255)
MARK_COLOR = (255, 255, 255)


def clips_for(frames: list[dict], player: str) -> list[tuple[float, float]]:
    """他出现的时段，合并短暂中断，去掉太短的，前后留余量。"""
    times = [f["time"] for f in frames if any(p.get("player") == player for p in f["people"])]
    clips: list[list[float]] = []
    for t in times:
        if clips and t - clips[-1][1] <= MERGE_GAP:
            clips[-1][1] = t
        else:
            clips.append([t, t])
    start, end = frames[0]["time"], frames[-1]["time"]
    return [(max(start, a - PAD), min(end, b + PAD)) for a, b in clips if b - a >= MIN_CLIP]


def find_player(entry: dict, player: str) -> list[float] | None:
    for p in entry["people"]:
        if p.get("player") == player:
            return p["box"]
    return None


class Camera:
    """平滑跟随：中心和画面大小都慢慢趋向目标，避免抖动。"""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.state: np.ndarray | None = None  # [中心 x, 中心 y, 画面高度]

    def update(self, box: list[float] | None) -> tuple[int, int, int, int]:
        if box is not None:
            x1, y1, x2, y2 = box
            view_h = min(self.height, max(OUT_H * 0.6, VIEW_HEIGHTS * (y2 - y1)))
            target = np.array([(x1 + x2) / 2, (y1 + y2) / 2, view_h])
            self.state = target if self.state is None else self.state + SMOOTH * (target - self.state)
        if self.state is None:
            return 0, 0, self.width, self.height
        cx, cy, vh = self.state
        vw = vh * OUT_W / OUT_H
        x0 = int(np.clip(cx - vw / 2, 0, self.width - vw))
        y0 = int(np.clip(cy - vh / 2, 0, self.height - vh))
        return x0, y0, int(vw), int(vh)


def render_reel(result: dict, ball: dict | None, video: Path, player: str, out_path: Path) -> list[tuple[float, float]]:
    info = result["info"]
    frames = result["frames"]
    clips = clips_for(frames, player)
    fps = info["fps"]
    ball_frames = ball["frames"] if ball else []
    ball_fps = ball["fps"] if ball else 1.0
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "h264_videotoolbox", "-b:v", "12M", "-pix_fmt", "yuv420p", str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    for k, (a, b) in enumerate(clips):
        camera = Camera(info["width"], info["height"])
        for t, frame in read_frames(video, info["width"], info["height"], a, b, fps):
            box = find_player(people_at(frames, result["fps"], t), player)
            x0, y0, vw, vh = camera.update(box)
            scale = OUT_W / vw
            view = cv2.resize(frame[y0 : y0 + vh, x0 : x0 + vw], (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

            def to_view(x: float, y: float) -> tuple[int, int]:
                return int((x - x0) * scale), int((y - y0) * scale)

            i = int(round((t - (ball["frames"][0]["time"] if ball_frames else 0)) * ball_fps))
            bf = ball_frames[i] if 0 <= i < len(ball_frames) else None
            holding = bf is not None and bf["holder"] == player
            if bf and bf["ball"]:
                bx, by, br, _c = bf["ball"]
                cv2.circle(view, to_view(bx, by), int(br * scale) + 10, HOLD_COLOR if holding else (0, 140, 255), 3, cv2.LINE_AA)
            if box is not None:
                x1, y1, x2, y2 = box
                color = HOLD_COLOR if holding else MARK_COLOR
                fx, fy = to_view((x1 + x2) / 2, y2)
                w = int((x2 - x1) * scale * 0.7)
                cv2.ellipse(view, (fx, fy), (w, max(6, w // 3)), 0, 0, 360, color, 3, cv2.LINE_AA)
                label = player.split("#")[-1] if "#" in player else player
                lx, ly = to_view((x1 + x2) / 2, y1)
                _text(view, f"#{label}" + ("  HAS BALL" if holding else ""), (lx - 30, ly - 16), color, 1.1)
            _text(view, f"{player}   clip {k + 1}/{len(clips)}   {int(t // 60)}:{t % 60:04.1f}", (24, OUT_H - 28), (255, 255, 255), 0.9)
            ffmpeg.stdin.write(view.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return clips


def _text(img: np.ndarray, text: str, org: tuple[int, int], color, size: float) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("player", help="固定编号，如 A#18")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or out_dir_for(args.video)
    result = json.loads((out_dir / "identified.json").read_text())
    known = {p.get("player") for f in result["frames"] for p in f["people"]}
    if args.player not in known:
        print(f"没有 {args.player}。已有：{' '.join(sorted(k for k in known if k))}", file=sys.stderr)
        return 1
    ball_file = out_dir / "ball.json"
    ball = json.loads(ball_file.read_text()) if ball_file.exists() else None
    out_path = out_dir / f"reel_{args.player.replace('#', '').replace('?', 'q')}.mp4"
    clips = render_reel(result, ball, args.video, args.player, out_path)
    total = sum(b - a for a, b in clips)
    print(f"{args.player}：{len(clips)} 段，共 {total:.0f} 秒 → {out_path}")
    for a, b in clips:
        print(f"  {int(a // 60)}:{a % 60:04.1f} – {int(b // 60)}:{b % 60:04.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
