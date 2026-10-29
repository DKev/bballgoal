"""虚拟镜头：从 360 原始素材（鱼眼）里渲染出“普通镜头”的画面，直线保持笔直，镜头平滑跟随。

给人看的视频（事件片段、集锦、回放）都用这个；分析仍用等距柱状展开的 full.mp4。
两者是同一个球面的不同投影，坐标可以精确换算：分析画面里的框、号码都能画到虚拟镜头画面上。

坐标约定（与 ffmpeg v360 一致）：x 向右、y 向上、z 向前（镜头光轴）。
  方向 ↔ 经纬度：经度 lon = atan2(x, z)，纬度 lat = asin(y)
  等距柱状图（宽 W = 2H）：u = (lon / 2π + 0.5) W，v = (0.5 - lat / π) H
  鱼眼（等距模型）：离光轴角 α 映射到半径 r = α / (fov/2) · R
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bballgoal.fisheye import EQ_WIDTH, LENS_FOV, probe

OUT_W, OUT_H = 1920, 1080
ENHANCE = "hqdn3d=2:1.5:4:3,unsharp=5:5:0.6:3:3:0,eq=contrast=1.04:saturation=1.08"  # 去噪 + 锐化 + 轻微提色


# ---------- 几何 ----------

def analysis_to_dir(px: np.ndarray, py: np.ndarray, view: dict) -> np.ndarray:
    """分析画面（full.mp4，等距柱状图裁出来的一块）上的点 → 单位方向向量（…×3）。"""
    width = EQ_WIDTH
    height = width // 2
    u = np.asarray(px, np.float64) + view["x"]
    v = np.asarray(py, np.float64) + view["y"]
    lon = (u / width - 0.5) * 2 * np.pi
    lat = (0.5 - v / height) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)


def dir_to_lonlat(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arctan2(d[..., 0], d[..., 2]), np.arcsin(np.clip(d[..., 1], -1, 1))


def dir_to_fisheye(d: np.ndarray, size: int, fov: float = LENS_FOV) -> tuple[np.ndarray, np.ndarray]:
    """方向 → 鱼眼原片像素（等距模型，成像圆铺满画面）。"""
    alpha = np.arccos(np.clip(d[..., 2], -1, 1))
    beta = np.arctan2(d[..., 1], d[..., 0])
    r = alpha / np.radians(fov / 2) * (size / 2)
    # 返回像素下标坐标（cv2.remap 约定：第 i 个像素的中心是 i），所以减半个像素；实测与 ffmpeg 逐像素一致
    return size / 2 + r * np.cos(beta) - 0.5, size / 2 - r * np.sin(beta) - 0.5


def rotation(yaw: float, pitch: float) -> np.ndarray:
    """先绕 x 轴抬头 pitch，再绕 y 轴转 yaw（弧度）。"""
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx


def view_rays(hfov: float, width: int = OUT_W, height: int = OUT_H) -> np.ndarray:
    """普通镜头（直线投影）每个像素的方向（镜头坐标系），hfov 为弧度。"""
    f = (width / 2) / np.tan(hfov / 2)
    xs = (np.arange(width) + 0.5 - width / 2) / f
    ys = -(np.arange(height) + 0.5 - height / 2) / f
    x, y = np.meshgrid(xs, ys)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project_to_view(d: np.ndarray, yaw: float, pitch: float, hfov: float, width: int = OUT_W, height: int = OUT_H):
    """世界方向 → 虚拟镜头画面像素（在镜头背后的点返回 nan）。"""
    local = d @ rotation(yaw, pitch)  # 旋转矩阵的转置 = 逆
    f = (width / 2) / np.tan(hfov / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + f * local[..., 0] / local[..., 2]
        v = height / 2 - f * local[..., 1] / local[..., 2]
    behind = local[..., 2] <= 0
    return np.where(behind, np.nan, u), np.where(behind, np.nan, v)


class Renderer:
    """把鱼眼原片按镜头姿态重采样成普通镜头画面。方向网格只算一次（同一视场角时）。"""

    def __init__(self, fisheye_size: int, lens_fov: float = LENS_FOV, width: int = OUT_W, height: int = OUT_H):
        self.size, self.fov, self.width, self.height = fisheye_size, lens_fov, width, height
        self._rays: dict[int, np.ndarray] = {}

    def maps(self, yaw: float, pitch: float, hfov: float) -> tuple[np.ndarray, np.ndarray]:
        key = int(round(np.degrees(hfov) * 4))  # 视场角按 0.25° 取整缓存
        if key not in self._rays:
            if len(self._rays) > 64:
                self._rays.clear()
            self._rays[key] = view_rays(np.radians(key / 4), self.width, self.height).astype(np.float32)
        d = self._rays[key] @ rotation(yaw, pitch).T.astype(np.float32)
        mx, my = dir_to_fisheye(d, self.size, self.fov)
        return mx.astype(np.float32), my.astype(np.float32)

    def render(self, fisheye: np.ndarray, yaw: float, pitch: float, hfov: float) -> np.ndarray:
        mx, my = self.maps(yaw, pitch, hfov)
        return cv2.remap(fisheye, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------- 镜头路线 ----------

@dataclass
class Shot:
    times: np.ndarray  # 秒（整场时间）
    yaw: np.ndarray  # 弧度
    pitch: np.ndarray
    hfov: np.ndarray

    def at(self, t: float) -> tuple[float, float, float]:
        return (float(np.interp(t, self.times, self.yaw)), float(np.interp(t, self.times, self.pitch)),
                float(np.interp(t, self.times, self.hfov)))


def plan_shot(times: np.ndarray, targets: list[np.ndarray | None], view: dict, aspect: float = OUT_W / OUT_H,
              min_fov: float = 30.0, max_fov: float = 80.0, margin: float = 1.6, smooth: float = 0.8,
              lift: float = 0.15, lead: float = 0.4) -> Shot:
    """targets[i]：第 i 个时刻要拍进画面的框（N×4，分析画面坐标），None 表示没有（沿用前后）。
    镜头对准这些框的中心，视场角刚好装下它们（再留 margin 倍余量），然后按 smooth 秒做平滑。
    录好的比赛可以“看到未来”：平滑用的是前后对称的窗口；再整体提前 lead 秒，镜头会预判转向，
    人不会跑到画面边上镜头才追过去。"""
    yaw, pitch, fov = [], [], []
    for boxes in targets:
        if boxes is None or len(boxes) == 0:
            yaw.append(np.nan), pitch.append(np.nan), fov.append(np.nan)
            continue
        b = np.asarray(boxes, np.float64)
        corners = np.concatenate([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]])
        d = analysis_to_dir(corners[:, 0], corners[:, 1], view)
        center = d.mean(axis=0)
        lon, lat = dir_to_lonlat(center / np.linalg.norm(center))
        # 构图：人放在画面中间偏下（上方留给篮筐和球的弧线，但别空出一大片天花板）
        lat = lat + lift * (np.max(dir_to_lonlat(d)[1]) - np.min(dir_to_lonlat(d)[1]) + 1e-3)
        # 视场角：各角点离中心的最大夹角（横向、纵向分别按画面比例折算）
        clon, clat = dir_to_lonlat(d)
        half_w = np.max(np.abs(np.angle(np.exp(1j * (clon - lon))))) * margin
        half_h = np.max(np.abs(clat - lat)) * margin
        need = 2 * max(half_w, np.arctan(np.tan(half_h) * aspect))
        yaw.append(float(lon)), pitch.append(float(lat))
        fov.append(float(np.clip(np.degrees(need), min_fov, max_fov)))
    # 整段都没有目标（当事人一直没被认出来）：退回广角对准画面中央，而不是视场角 0
    center = analysis_to_dir(np.array([view.get("w", 0) / 2]), np.array([view.get("h", 0) / 2]), view)[0]
    home = (*dir_to_lonlat(center / np.linalg.norm(center)), max_fov)
    arrays = [_fill(np.array(a, np.float64), float(d)) for a, d in zip((yaw, pitch, fov), home)]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    sigma = max(1.0, smooth / dt)
    yaw_s, pitch_s, fov_s = (_gauss(a, sigma) for a in arrays)
    times = np.asarray(times, np.float64)
    # 提前 lead 秒：t 时刻的镜头用 t+lead 时刻的目标
    yaw_s, pitch_s, fov_s = (np.interp(times + lead, times, a) for a in (yaw_s, pitch_s, fov_s))
    return Shot(times, yaw_s, pitch_s, np.radians(fov_s))


def _fill(a: np.ndarray, default: float = 0.0) -> np.ndarray:
    ok = ~np.isnan(a)
    if not ok.any():
        return np.full_like(a, default)
    idx = np.arange(len(a))
    return np.interp(idx, idx[ok], a[ok])


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(a, radius, mode="edge")
    return np.convolve(padded, k, mode="valid")


# ---------- 读原片、渲染 ----------

def part_offsets(parts: list[str]) -> list[tuple[str, float, float]]:
    """[(文件, 在整场里的起点, 时长)]。整场时间 = 各段按顺序接起来的时间（full.mp4 就是这么接的）。"""
    out, start = [], 0.0
    for p in parts:
        d = float(probe(Path(p))["format"]["duration"])
        out.append((p, start, d))
        start += d
    return out


def read_fisheye(parts: list[str], lens: int, start: float, end: float, fps: float):
    """按整场时间读鱼眼原片的某个镜头，逐帧返回 (整场时间, BGR 图)。可能跨文件。"""
    for path, offset, duration in part_offsets(parts):
        a, b = max(start, offset), min(end, offset + duration)
        if a >= b:
            continue
        info = probe(Path(path))
        stream = [s for s in info["streams"] if s["codec_type"] == "video"][lens]
        w, h = stream["width"], stream["height"]
        cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{a - offset:.3f}", "-t", f"{b - a:.3f}",
               "-i", path, "-map", f"0:v:{lens}", "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3)
        i = 0
        try:
            while True:
                buf = proc.stdout.read(w * h * 3)
                if len(buf) < w * h * 3:
                    break
                yield a + i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                i += 1
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()


class VideoWriter:
    """把渲染好的帧交给 ffmpeg：去噪、锐化、调色，硬件编码。"""

    def __init__(self, path: Path, fps: float, width: int = OUT_W, height: int = OUT_H, enhance: bool = True, bitrate: str = "16M"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", f"{fps:.3f}", "-i", "-"]
        if enhance:
            cmd += ["-vf", ENHANCE]
        cmd += ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def load_view(game_dir: Path) -> dict:
    return json.loads((game_dir / "full.view.json").read_text())
