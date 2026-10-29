"""头像素材：给每段轨迹挑脸最清楚的一帧，取头肩截图和外貌颜色（肤色、发色）。

远处小孩的脸只有十几个像素，照片转手绘只会糊成色块。所以网页上按这些颜色加上号码、球衣颜色
“画”一个卡通头像（发色、肤色、队服、号码都对得上，大概像），鼠标停上去再看真实截图。

脸清不清楚用姿态关键点判断：鼻子和两只眼睛都看得到（正脸）、头越大越好。
"""

from __future__ import annotations

import cv2
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = range(5)
KP_CONF = 0.3
SIZE = 192


def face_score(kp: np.ndarray | None, box) -> float:
    """越大越正的脸分越高；看不到脸（背对）接近 0。"""
    if kp is None:
        return 0.0
    front = float(np.mean(kp[[NOSE, L_EYE, R_EYE], 2]))
    return front * (box[3] - box[1])


def head_square(kp: np.ndarray | None, box) -> tuple[float, float, float]:
    """头肩方框：(中心 x, 中心 y, 边长)。没有关键点时按人框比例估计。"""
    x1, y1, x2, y2 = box
    h = y2 - y1
    if kp is not None:
        head = kp[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR]]
        head = head[head[:, 2] >= KP_CONF]
        if len(head):
            side = 0.5 * h
            return float(head[:, 0].mean()), float(head[:, 1].mean()) + 0.18 * side, side
    return (x1 + x2) / 2, y1 + 0.16 * h, 0.42 * h


def head_photo(frame: np.ndarray, kp: np.ndarray | None, box) -> np.ndarray:
    """头肩截图，放大并轻微锐化。"""
    cx, cy, side = head_square(kp, box)
    m = np.float32([[SIZE / side, 0, SIZE / 2 - cx * SIZE / side], [0, SIZE / side, SIZE / 2 - cy * SIZE / side]])
    img = cv2.warpAffine(frame, m, (SIZE, SIZE), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1.4, blur, -0.4, 0)


def _patch_color(frame: np.ndarray, x: float, y: float, r: float) -> np.ndarray | None:
    r = max(1, int(round(r)))
    x, y = int(round(x)), int(round(y))
    patch = frame[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1].reshape(-1, 3)
    return np.median(patch, axis=0) if len(patch) else None


def look(frame: np.ndarray, kp: np.ndarray | None, box) -> dict | None:
    """肤色（鼻子和眼睛之间）和发色（眼睛上方的头顶）。必须看得到正脸。"""
    if kp is None or (kp[[NOSE, L_EYE, R_EYE], 2] < KP_CONF).any():
        return None
    h = box[3] - box[1]
    eyes = kp[[L_EYE, R_EYE], :2].mean(axis=0)
    nose = kp[NOSE, :2]
    skin = _patch_color(frame, (eyes[0] + nose[0]) / 2, (eyes[1] + nose[1]) / 2 + 0.01 * h, 0.012 * h)
    hair = _patch_color(frame, eyes[0], eyes[1] - 0.055 * h, 0.012 * h)
    if skin is None or hair is None:
        return None
    return {"skin": _hex(skin), "hair": _hex(hair)}


def _hex(bgr: np.ndarray) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"
