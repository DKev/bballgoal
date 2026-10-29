"""把 360 相机的原始文件（.insv，两路鱼眼）转成后续流程用的平面视频。

不用拼接：相机侧对球场时，整个球场落在一个镜头里。步骤：
1. 自动选镜头：两路各抽几帧找人，场上人多的那路就是对着球场的；
2. 鱼眼展开成等距柱状投影（每度像素数处处相同，竖直的人还是竖直的）；
3. 自动定画面范围：抽帧找人，取所有人所在的横带，往上留出篮筐和投篮弧线的空间；
4. 多段文件（相机按大小自动切分的同一次录像）按顺序接成一个视频，保留原始帧率和声音。

用法：
    .venv/bin/python -m bballgoal.fisheye videos/raw/VID_..._016.insv videos/raw/VID_..._017.insv -o videos/比赛.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bballgoal.detect import default_device

LENS_FOV = 200.0  # 鱼眼镜头视场角（度）。X 系列大约 190–200，偏差只让边缘略微拉伸，不影响识别
EQ_WIDTH = 5760  # 展开后整圈 360° 的宽度：和镜头原生分辨率（2880 / 180°）一致，不放大不缩小
SAMPLES = 12  # 抽多少帧来选镜头、定范围
HEAD_ROOM = 2.0  # 人群横带上方再留多少倍人身高（篮筐、投篮弧线）
FOOT_ROOM = 0.6  # 下方留多少倍人身高
BITRATE = "60M"  # 中间视频码率：要高，避免二次压缩把号码压糊


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def grab(path: Path, t: float, stream: int, vf: str | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-map", f"0:v:{stream}", "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def v360(lens_fov: float, width: int) -> str:
    return f"v360=input=fisheye:ih_fov={lens_fov}:iv_fov={lens_fov}:output=equirect:w={width}:h={width // 2}:interp=lanczos"


def detect_people(model: YOLO, image: np.ndarray, tile: int = 1280) -> np.ndarray:
    """分块找人，返回 N×4 框。"""
    h, w = image.shape[:2]
    boxes = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            crop = image[y : y + tile, x : x + tile]
            r = model.predict(crop, imgsz=tile, conf=0.4, classes=[0], device=default_device(), verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                boxes.append(b + [x, y, x, y])
    return np.array(boxes).reshape(-1, 4)


def choose_view(path: Path, duration: float, n_lenses: int, lens_fov: float) -> dict:
    """选镜头、定裁切范围（展开后画面里的 x, y, w, h）。"""
    model = YOLO("models/yolo11m.pt")
    times = np.linspace(duration * 0.1, duration * 0.9, SAMPLES)
    per_lens = {}
    for lens in range(n_lenses):
        boxes = [detect_people(model, grab(path, t, lens, v360(lens_fov, EQ_WIDTH))) for t in times]
        per_lens[lens] = np.concatenate(boxes)
    lens = max(per_lens, key=lambda k: len(per_lens[k]))
    boxes = per_lens[lens]
    heights = boxes[:, 3] - boxes[:, 1]
    # 只看“像球员”的框：去掉特别大的（离镜头很近的观众）
    keep = heights <= np.percentile(heights, 90)
    boxes, heights = boxes[keep], heights[keep]
    h = float(np.median(heights))
    feet = boxes[:, 3]
    y0 = max(0, int(np.percentile(boxes[:, 1], 5) - HEAD_ROOM * h))
    y1 = min(EQ_WIDTH // 2, int(np.percentile(feet, 98) + FOOT_ROOM * h))
    # 横向：镜头能看到的全部（展开后中间 lens_fov 那一段），去掉最边缘 5°
    span = EQ_WIDTH * (lens_fov - 10) / 360
    x0 = int(EQ_WIDTH / 2 - span / 2)
    w = int(span) // 2 * 2
    return {"lens": lens, "x": x0, "y": y0 // 2 * 2, "w": w, "h": (y1 - y0) // 2 * 2, "people_per_lens": {k: len(v) for k, v in per_lens.items()}}


def convert(parts: list[Path], out: Path, view: dict | None, lens_fov: float = LENS_FOV, quiet: bool = False) -> None:
    """parts 按顺序接起来转码。view 为 None 时不展开（平面视频，只重新编码）。打印进度百分比。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{Path(p).resolve()}'\n")
        listing = f.name
    total = sum(float(probe(Path(p))["format"]["duration"]) for p in parts)
    cmd = ["ffmpeg", "-v", "error", "-nostats", "-progress", "pipe:1", "-y", "-f", "concat", "-safe", "0", "-i", listing]
    if view is not None:
        vf = f"{v360(lens_fov, EQ_WIDTH)},crop={view['w']}:{view['h']}:{view['x']}:{view['y']}"
        cmd += ["-map", f"0:v:{view['lens']}", "-map", "0:a:0?", "-vf", vf]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    partial = Path(out).with_name(Path(out).stem + ".partial" + Path(out).suffix)  # 转完才改名：中途停了不会被当成转好了
    cmd += ["-c:v", "hevc_videotoolbox", "-b:v", BITRATE, "-tag:v", "hvc1", "-c:a", "aac", str(partial)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    last = -1
    for line in proc.stdout:
        if line.startswith("out_time_us=") and not quiet:
            try:
                pct = int(100 * int(line.split("=")[1]) / 1e6 / max(1.0, total))
            except ValueError:
                continue
            if pct > last:
                print(f"  转换 {pct}%", flush=True)
                last = pct
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    partial.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", type=Path, nargs="+", help="同一次录像的各段 .insv，按时间顺序")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--lens-fov", type=float, default=LENS_FOV)
    parser.add_argument("--view", type=Path, default=None, help="沿用之前定好的镜头和画面范围（.view.json）")
    parser.add_argument("--view-only", action="store_true", help="只判断镜头和画面范围（写 .view.json），不转换")
    parser.add_argument("--start", type=float, default=0, help="从第一段的第几秒开始（试效果用）")
    parser.add_argument("--duration", type=float, default=None, help="只转多少秒（试效果用）")
    args = parser.parse_args()

    info = probe(args.parts[0])
    n_lenses = sum(s["codec_type"] == "video" for s in info["streams"])
    duration = float(info["format"]["duration"])
    if args.view:
        view = json.loads(args.view.read_text())
    else:
        view = choose_view(args.parts[0], duration, n_lenses, args.lens_fov)
    print(f"对着球场的镜头：{view['lens']}（各镜头找到的人数 {view['people_per_lens']}）")
    print(f"画面范围：{view['w']}×{view['h']}，从 ({view['x']}, {view['y']}) 开始")
    args.out.with_suffix(".view.json").write_text(json.dumps({**view, "lens_fov": args.lens_fov, "parts": [str(p) for p in args.parts]}, indent=2))
    if args.view_only:
        return 0

    parts = args.parts
    if args.duration:
        # 试效果：只取第一段里的一小段
        tmp = args.out.with_suffix(".part.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(args.start), "-t", str(args.duration), "-i", str(parts[0]),
                        "-map", "0", "-c", "copy", str(tmp)], check=True)
        parts = [tmp]
    convert(parts, args.out, view, args.lens_fov)
    if args.duration:
        parts[0].unlink()
    print(f"输出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
