"""第一步“先认人”：从整场里均匀抽几小段，接成一个短视频，用来让你确认名单。

不只看开头：替补后面才上、下半场两队换边，均匀抽能让每个孩子都出现，正面背面也都有机会拍到。
段与段之间插几秒黑屏，跟踪器不会把前一段的人和后一段的人接起来。

用法：
    .venv/bin/python -m bballgoal.enroll videos/raw/..._016.insv videos/raw/..._017.insv -o videos/比赛_enroll.mp4
然后照常跑 track → identify → cards → review。
输出旁边的 .enroll.json 记录短视频里每一段对应整场的哪个时间。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bballgoal.fisheye import BITRATE, LENS_FOV, choose_view, convert, probe

SNIPPETS = 8
SNIPPET_SECONDS = 40.0
GAP_SECONDS = 4.0  # 黑屏长度：比跟踪器的轨迹保留时间（3 秒）长
EDGE = 0.03  # 不从最开头和最末尾取（热身、散场）


def plan(durations: list[float], n: int, seconds: float) -> list[tuple[int, float, float]]:
    """在整场时间轴上均匀取 n 段。返回 (文件序号, 文件内起点, 整场时间起点)。"""
    total = sum(durations)
    starts = np.linspace(total * EDGE, total * (1 - EDGE) - seconds, n)
    out = []
    for g in starts:
        offset = 0.0
        for k, d in enumerate(durations):
            if g < offset + d:
                local = min(g - offset, d - seconds)  # 不跨文件
                out.append((k, max(0.0, local), offset + max(0.0, local)))
                break
            offset += d
    return out


def black_clip(path: Path, width: int, height: int, fps: str, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={width}x{height}:r={fps}:d={seconds}",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(seconds),
         "-c:v", "hevc_videotoolbox", "-b:v", BITRATE, "-tag:v", "hvc1", "-c:a", "aac", str(path)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", type=Path, nargs="+", help="同一次录像的各段 .insv（按时间顺序），或一个平面视频")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--view", type=Path, default=None, help="沿用之前定好的镜头和画面范围（.view.json）")
    parser.add_argument("--snippets", type=int, default=SNIPPETS)
    parser.add_argument("--seconds", type=float, default=SNIPPET_SECONDS)
    args = parser.parse_args()

    infos = [probe(p) for p in args.parts]
    durations = [float(i["format"]["duration"]) for i in infos]
    fps = next(s["r_frame_rate"] for s in infos[0]["streams"] if s["codec_type"] == "video")
    flat = args.parts[0].suffix.lower() != ".insv"
    if flat:
        stream = next(s for s in infos[0]["streams"] if s["codec_type"] == "video")
        view = None
        size = (stream["width"], stream["height"])
    else:
        if args.view:
            view = json.loads(args.view.read_text())
        else:
            n_lenses = sum(s["codec_type"] == "video" for s in infos[0]["streams"])
            view = choose_view(args.parts[0], durations[0], n_lenses, LENS_FOV)
        size = (view["w"], view["h"])
    lens_fov = view.get("lens_fov", LENS_FOV) if view else LENS_FOV

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        black = tmp / "black.mp4"
        black_clip(black, size[0], size[1], fps, GAP_SECONDS)
        pieces = []
        position = 0.0
        for k, (part, local, game) in enumerate(plan(durations, args.snippets, args.seconds)):
            raw = tmp / f"raw{k}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{local:.3f}", "-t", str(args.seconds), "-i", str(args.parts[part]),
                            "-map", "0", "-c", "copy", str(raw)], check=True)
            piece = tmp / f"flat{k}.mp4"
            convert([raw], piece, view, lens_fov, quiet=True)
            print(f"  第 {k + 1}/{args.snippets} 段：整场 {int(game // 60)}:{game % 60:04.1f}", flush=True)
            pieces += [piece, black]
            segments.append({"start": round(position, 3), "end": round(position + args.seconds, 3), "game_start": round(game, 3)})
            position += args.seconds + GAP_SECONDS
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in pieces))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(args.out)], check=True)
    args.out.with_suffix(".enroll.json").write_text(json.dumps({"segments": segments, "parts": [str(p) for p in args.parts]}, indent=2))
    if view:
        args.out.with_suffix(".view.json").write_text(json.dumps({**view, "lens_fov": lens_fov}, indent=2))
    print(f"输出：{args.out}（{len(segments)} 段，共 {position:.0f} 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
