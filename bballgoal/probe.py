"""识别效果探测：从视频里均匀抽几帧，画出篮球 / 篮筐 / 人的检测框。

用法：
    .venv/bin/python -m bballgoal.probe videos/比赛.mp4
    .venv/bin/python -m bballgoal.probe videos/比赛.mp4 --frames 12 --start 60 --end 240
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from bballgoal.detect import Detection, TiledDetector

COLORS = {"ball": (0, 140, 255), "hoop": (0, 0, 255), "person": (80, 200, 80)}


def video_info(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": float(num) / float(den),
        "codec": stream["codec_name"],
        "duration": float(data["format"]["duration"]),
    }


def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = frame.copy()
    scale = max(1.0, frame.shape[1] / 1920)
    thickness = max(2, int(2 * scale))
    for det in detections:
        color = COLORS[det.label]
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        cv2.putText(
            canvas, f"{det.label} {det.conf:.2f}", (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, thickness,
        )
        if det.label == "ball":
            # 球很小，额外画个大圈方便在缩略图里找到
            cx, cy = map(int, det.center)
            cv2.circle(canvas, (cx, cy), int(30 * scale), color, thickness)
    return canvas


def read_frame(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=8, help="抽多少帧（默认 8）")
    parser.add_argument("--start", type=float, default=0, help="从第几秒开始")
    parser.add_argument("--end", type=float, default=None, help="到第几秒结束")
    parser.add_argument("--tile", type=int, default=1280, help="分块边长（像素）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认 output/probe_<视频名>）")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"找不到视频：{args.video}", file=sys.stderr)
        return 1

    info = video_info(args.video)
    print(
        f"视频：{info['width']}x{info['height']}  {info['fps']:.2f}fps  "
        f"{info['codec']}  {info['duration']:.1f}s"
    )
    out_dir = args.out or Path("output") / f"probe_{args.video.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    end = min(args.end or info["duration"], info["duration"] - 0.5)
    times = np.linspace(args.start, end, args.frames)

    detector = TiledDetector(tile=args.tile)
    print(f"推理设备：{detector.device}")
    cap = cv2.VideoCapture(str(args.video))
    summary = []
    thumbs = []
    for i, t in enumerate(times):
        frame = read_frame(cap, float(t))
        if frame is None:
            print(f"  {t:7.1f}s  读帧失败")
            continue
        started = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = time.perf_counter() - started
        counts = {k: sum(d.label == k for d in detections) for k in COLORS}
        print(
            f"  {t:7.1f}s  人 {counts['person']:2d}  篮筐 {counts['hoop']}  "
            f"篮球 {counts['ball']}  ({elapsed:.1f}s)"
        )
        annotated = draw(frame, detections)
        cv2.imwrite(str(out_dir / f"frame_{i:02d}_{t:07.1f}s.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        thumbs.append(cv2.resize(annotated, (960, int(960 * frame.shape[0] / frame.shape[1]))))
        summary.append({
            "time": round(float(t), 2),
            "counts": counts,
            "seconds": round(elapsed, 2),
            "detections": [
                {"label": d.label, "conf": round(d.conf, 3),
                 "box": [round(v, 1) for v in (d.x1, d.y1, d.x2, d.y2)]}
                for d in detections
            ],
        })
    cap.release()

    if thumbs:
        rows = [np.hstack(thumbs[i : i + 2]) if i + 1 < len(thumbs) else np.hstack([thumbs[i], np.zeros_like(thumbs[i])])
                for i in range(0, len(thumbs), 2)]
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

    (out_dir / "summary.json").write_text(
        json.dumps({"video": str(args.video), "info": info, "frames": summary}, ensure_ascii=False, indent=2)
    )
    print(f"\n输出：{out_dir}/  （contact_sheet.jpg 是总览图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
