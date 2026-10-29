"""按球衣颜色自动分队。

每个人取躯干中间一块（避开头、腿和背景），按 HSV 把像素分成白色、黑色、彩色三类：
- 裁判：黑白竖条纹，躯干横向明暗来回交替多次；
- 球员：看彩色像素的主色调，加上黑 / 白占比（应对白色或黑色球衣），聚成两队。

只看颜色中位数行不通：深色球衣上的白色号码和褶皱阴影，会让它的中位数和裁判条纹很像。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

REFEREE = "ref"
UNKNOWN = "?"

# 特征：[白色占比, 黑色占比, 彩色占比, 彩色主色调 cos × 彩色占比, sin × 彩色占比, 竖条纹明暗交替次数,
#        彩色像素的平均亮度 × 彩色占比]
WHITE, BLACK, CHROMA, HUE_X, HUE_Y, STRIPES, BRIGHT = range(7)


def stripe_count(patch: np.ndarray, amplitude: float = 20) -> int:
    """躯干横向亮度曲线的明暗交替次数。裁判条纹约 6 次，球衣号码只有 1–3 次。"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = cv2.resize(gray.mean(axis=0)[None], (40, 1))[0]
    # 去掉灯光造成的左右明暗渐变，只留条纹本身的起伏
    x = np.arange(len(profile), dtype=np.float32)
    profile -= np.polyval(np.polyfit(x, profile, 1), x)
    states = [1 if v > amplitude else -1 for v in profile if abs(v) > amplitude]
    return sum(1 for a, b in zip(states, states[1:]) if a != b)


def torso_feature(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 30:
        return None
    tx1, tx2 = int(x1 + 0.3 * w), int(x1 + 0.7 * w)
    ty1, ty2 = int(y1 + 0.18 * h), int(y1 + 0.45 * h)
    patch = frame[max(0, ty1) : ty2, max(0, tx1) : tx2]
    if patch.size == 0:
        return None
    # 用 HSV 饱和度判断“有颜色”：暗红这类深色球衣亮度低，但饱和度仍然高；
    # Lab 色度会随亮度一起变小，深色球衣会被误当成黑色，和裁判条纹混在一起。
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]  # OpenCV 的 H 是 0–180
    colorful = (sat >= 70) & (val >= 45)
    white = ~colorful & (sat < 50) & (val > 150)
    black = ~colorful & (val < 60)
    frac = colorful.mean()
    if colorful.any():
        angles = np.deg2rad(hue[colorful])
        hue_x, hue_y = np.cos(angles).mean(), np.sin(angles).mean()
        norm = max(1e-6, float(np.hypot(hue_x, hue_y)))
        hue_x, hue_y = hue_x / norm * frac, hue_y / norm * frac
        # 亮度：酒红和黄色在暖光 / 红墙前色调会靠近（14° 对 42°），但一个很暗一个很亮
        bright = float(val[colorful].mean()) / 255 * frac
    else:
        hue_x = hue_y = bright = 0.0
    stripe_patch = frame[max(0, int(y1 + 0.15 * h)) : int(y1 + 0.5 * h), max(0, int(x1 + 0.15 * w)) : int(x1 + 0.85 * w)]
    stripes = stripe_count(stripe_patch) if stripe_patch.shape[1] >= 12 else 0
    return np.array([white.mean(), black.mean(), frac, hue_x, hue_y, stripes, bright], dtype=np.float32)


def is_referee(feature: np.ndarray) -> bool:
    # 颜色上，深色球衣配大号白字和条纹很像，主要靠条纹交替次数区分
    return feature[STRIPES] >= 5 and feature[WHITE] > 0.08 and feature[BLACK] > 0.08


def _team_space(features: np.ndarray) -> np.ndarray:
    return features[:, [WHITE, BLACK, HUE_X, HUE_Y, BRIGHT]]


@dataclass
class TeamModel:
    centers: np.ndarray  # (2, 5) 两队在 _team_space 里的中心，A 队在前
    colors_bgr: dict[str, tuple[int, int, int]]
    max_distance: float

    def assign(self, feature: np.ndarray) -> str:
        if is_referee(feature):
            return REFEREE
        dists = np.linalg.norm(self.centers - _team_space(feature[None]), axis=1)
        best = int(np.argmin(dists))
        return ("A", "B")[best] if dists[best] <= self.max_distance else UNKNOWN


def _display_color(center: np.ndarray) -> tuple[int, int, int]:
    """把队伍中心特征换成一个画框用的颜色。"""
    if center[2] ** 2 + center[3] ** 2 < 0.05:  # 主要是黑或白
        return (255, 255, 255) if center[0] > center[1] else (40, 40, 40)
    hue = np.rad2deg(np.arctan2(center[3], center[2])) % 360
    hsv = np.uint8([[[hue / 2, 200, 230]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def fit_teams(track_features: dict[int, np.ndarray]) -> TeamModel:
    players = np.stack([f for f in track_features.values() if not is_referee(f)])
    space = _team_space(players)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, assignment, centers = cv2.kmeans(space, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # 固定顺序：色调角度小的为 A，保证同一场比赛重复运行时队名不变
    order = np.argsort(np.arctan2(centers[:, 3], centers[:, 2]))
    centers = centers[order]
    assignment = np.argsort(order)[assignment.ravel()]

    dists = np.linalg.norm(space - centers[assignment], axis=1)
    max_distance = max(0.15, float(np.percentile(dists, 90) * 2)) if len(dists) >= 5 else float("inf")
    colors = {
        "A": _display_color(centers[0]),
        "B": _display_color(centers[1]),
        REFEREE: (255, 255, 255),
        UNKNOWN: (160, 160, 160),
    }
    return TeamModel(centers=centers, colors_bgr=colors, max_distance=max_distance)
