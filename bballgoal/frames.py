"""用 ffmpeg（Mac 上走 VideoToolbox 硬件解码）按固定帧率读视频帧。"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def read_frames(
    video: Path, width: int, height: int, start: float, end: float, fps: float
) -> Iterator[tuple[float, np.ndarray]]:
    """按 fps 均匀取帧，逐帧返回 (时间秒, BGR 原分辨率画面)。"""
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{start:.3f}"]
    if end != float("inf"):
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-i", str(video), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = width * height * 3
    # 读够了会提前关掉 ffmpeg，它会报一串 Broken pipe，不是错误，别让它刷到日志里
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes)
    try:
        index = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield start + index / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
