"""球衣号码识别：用 macOS 自带的 Vision 文字识别，完全在本地运行。

单帧识别不可靠（背对 / 侧身 / 被挡 / 太小），所以每条轨迹取多帧，投票决定号码。
"""

from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np
import Vision
from Foundation import NSData

DIGITS = re.compile(r"^\d{1,2}$")
MIN_HEIGHT = 110  # 人框高度低于这个像素，号码基本看不清，不识别


def number_crop(frame: np.ndarray, box: tuple[float, float, float, float], torso=None) -> np.ndarray | None:
    """取躯干（号码所在的胸 / 背部），放大到识别器容易处理的尺寸。

    torso 是姿态关键点给出的肩到髋的框，有就用它（弯腰、跳起时也准）；没有就按人框比例估计。
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h < MIN_HEIGHT:
        return None
    if torso is not None:
        cx1, cy1, cx2, cy2 = (int(v) for v in torso)
        cx1, cy1 = max(0, cx1), max(0, cy1)
    else:
        cx1, cx2 = int(max(0, x1 + 0.1 * w)), int(x1 + 0.9 * w)
        cy1, cy2 = int(max(0, y1 + 0.15 * h)), int(y1 + 0.55 * h)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    scale = 160 / crop.shape[0]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _loop_position(binary: np.ndarray) -> str | None:
    """binary 里白色是数字：找最大且靠中间的连通块，看它唯一的内孔在上半还是下半。"""
    h, w = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    best, best_score = None, 0.0
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if bh < h * 0.3 or bw > w * 0.8 or bh > h * 0.95 or area < 50:
            continue
        score = area / (1 + abs(x + bw / 2 - w / 2))  # 偏好又大又靠中间的
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None

    digit = np.uint8(labels == best) * 255
    contours, hierarchy = cv2.findContours(digit, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    x, y, bw, bh, _ = stats[best]
    holes = [c for c, info in zip(contours, hierarchy[0]) if info[3] != -1 and cv2.contourArea(c) > 0.02 * bw * bh]
    if len(holes) != 1:
        return None
    m = cv2.moments(holes[0])
    if m["m00"] == 0:
        return None
    hole_y = (m["m01"] / m["m00"] - y) / bh
    if hole_y > 0.55:
        return "6"
    if hole_y < 0.45:
        return "9"
    return None


def six_or_nine(image: np.ndarray) -> str | None:
    """Vision 经常把球衣字体的 6 读成 9。看数字里封闭的圈在上半还是下半来纠正。

    不知道号码是深色还是浅色，两种极性都试；只有一种给出结论、或两种结论一致时才采用。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    answers = {_loop_position(binary), _loop_position(255 - binary)} - {None}
    return answers.pop() if len(answers) == 1 else None


def read_numbers(image: np.ndarray) -> list[tuple[str, float]]:
    """返回识别到的 1–2 位数字及置信度。"""
    ok, png = cv2.imencode(".png", image)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setMinimumTextHeight_(0.15)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    success, _err = handler.performRequests_error_([request], None)
    if not success:
        return []
    found = []
    for observation in request.results() or []:
        for candidate in observation.topCandidates_(3):
            text = str(candidate.string()).strip().replace("O", "0").replace("o", "0")
            if DIGITS.match(text):
                number = str(int(text))
                if number in ("6", "9"):
                    number = six_or_nine(image) or number
                found.append((number, float(candidate.confidence())))
                break
    return found


def vote(readings: list[tuple[str, float]], min_votes: int = 2, min_share: float = 0.6) -> tuple[str | None, float]:
    """多帧投票：至少 min_votes 次、且占全部读数 min_share 以上才采信。返回 (号码, 占比)。"""
    if not readings:
        return None, 0.0
    weights: Counter[str] = Counter()
    for number, conf in readings:
        weights[number] += conf
    number, weight = weights.most_common(1)[0]
    count = sum(1 for n, _ in readings if n == number)
    share = weight / sum(weights.values())
    if count >= min_votes and share >= min_share:
        return number, share
    return None, share
