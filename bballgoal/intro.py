"""集锦里每段的游戏式开场：画面定格，背景压暗、当事人打光，脚下一个圈落下来，
头顶拉出一根折线，线头打字机似地打出名字和这一球（“投篮 · 命中 · 2 分”），然后恢复播放。

    make_segment(clip, marks, title, team, sub, out)
clip 是渲染好的事件片段；marks[i] 是当事人在第 i 帧画面里的位置（clips.mark_at 给的 foot / w / head）。
在他第一次出现的那一帧定格（通常是第一帧），之前的画面照常播放，不跳过。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bballgoal.clips import FPS, MARK, font
from bballgoal.vcam import OUT_H, OUT_W, VideoWriter

FRAMES = 90  # 开场一共多少帧（3.6 秒）：名字打完后停 1.6 秒左右，看得清
RING_IN = (0, 10)  # 圈从大缩到脚下
LINE_IN = (8, 18)  # 线从头顶往上画出来
TYPE_IN = (18, 38)  # 名字一个字一个字打出来
OUT = (78, 90)  # 特效淡出、画面恢复亮度，接着播放
RING_HOLD = 8  # 帧：开场之后跟拍的光圈，短暂跟丢这么多帧以内停在原位
DIM = 0.55  # 背景压暗到多少
ACCENT = MARK  # BGR 金黄
WHITE = (255, 255, 255)


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _phase(i: int, span: tuple[int, int]) -> float:
    return _ease((i - span[0]) / max(1, span[1] - span[0]))


def _glow(layer: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    return cv2.GaussianBlur(layer, (0, 0), sigma)


def ring_layer(shape: tuple, mark: dict, scale: float = 1.0) -> np.ndarray:
    """脚下的金色光圈（外圈金、内圈白）。"""
    layer = np.zeros(shape, np.uint8)
    fx, fy = mark["foot"]
    rw = int(mark["w"] * scale)
    cv2.ellipse(layer, (int(fx), int(fy)), (rw, max(6, rw // 3)), 0, 0, 360, ACCENT, 6, cv2.LINE_AA)
    cv2.ellipse(layer, (int(fx), int(fy)), (int(rw * 0.72), max(4, int(rw * 0.72) // 3)), 0, 0, 360, WHITE, 2, cv2.LINE_AA)
    return layer


def _add_glow(img: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return cv2.add(img, cv2.add(layer, _glow(layer)))


def _geometry(mark: dict | None) -> dict:
    """线怎么走、名牌放哪：从头顶往上，再折向画面中间那一侧。"""
    if not mark:
        return {"anchor": (OUT_W / 2 - 300, OUT_H * 0.32), "points": [], "side": 1}
    fx, fy = mark["foot"]
    hx, hy = mark["head"]
    height = max(40.0, fy - hy)
    up = min(max(90.0, 0.9 * height), hy - 120) if hy > 220 else max(60.0, hy - 60)
    side = 1 if hx < OUT_W * 0.6 else -1
    elbow = (hx, hy - 12 - up)
    end = (hx + side * 70, elbow[1] - 40)
    return {"anchor": end, "points": [(hx, hy - 12), elbow, end], "side": side}


def _polyline_part(points: list, p: float) -> list:
    """折线画到 p（0~1）的那一截。"""
    if len(points) < 2 or p <= 0:
        return points[:1]
    seg = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    left = p * sum(seg)
    out = [points[0]]
    for (a, b), d in zip(zip(points, points[1:]), seg):
        if left >= d:
            out.append(b)
            left -= d
        else:
            r = left / max(d, 1e-6)
            out.append((a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r))
            break
    return out


def intro_frames(frame: np.ndarray, mark: dict | None, title: str, team: str, sub: str, frames: int = FRAMES):
    """逐帧生成开场动画（BGR）。"""
    geo = _geometry(mark)
    h, w = frame.shape[:2]
    # 打光：当事人身边亮，其余压暗
    light = np.zeros((h, w), np.float32)
    if mark:
        fx, fy = mark["foot"]
        hx, hy = mark["head"]
        r = max(120.0, 1.3 * (fy - hy))
        cv2.ellipse(light, (int(fx), int((fy + hy) / 2)), (int(r * 0.8), int(r)), 0, 0, 360, 1.0, -1)
        light = cv2.GaussianBlur(light, (0, 0), r * 0.35)
    big, mid, small = font(88), font(36), font(46)
    for i in range(frames):
        fade = 1 - _phase(i, OUT)
        dim = 1 - (1 - DIM) * _phase(i, (0, 6)) * fade
        gain = dim + (1 - dim) * light
        img = np.clip(frame.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
        fx_layer = np.zeros_like(img)
        if mark:
            fx, fy = mark["foot"]
            # 圈不淡出：开场之后接着跟在他脚下（make_segment 里逐帧画）
            img = _add_glow(img, ring_layer(img.shape, mark, 2.4 - 1.4 * _phase(i, RING_IN)))
            # 落地后的一圈冲击波
            wave = (i - RING_IN[1]) / 12
            if 0 <= wave <= 1:
                ww = int(mark["w"] * (1 + 1.6 * wave))
                cv2.ellipse(fx_layer, (int(fx), int(fy)), (ww, max(6, ww // 3)), 0, 0, 360,
                            tuple(int(c * (1 - wave)) for c in ACCENT), 3, cv2.LINE_AA)
            pts = _polyline_part(geo["points"], _phase(i, LINE_IN))
            if len(pts) >= 2:
                cv2.polylines(fx_layer, [np.array(pts, np.int32)], False, WHITE, 3, cv2.LINE_AA)
            if i >= LINE_IN[0]:
                hx, hy = geo["points"][0]
                cv2.circle(fx_layer, (int(hx), int(hy)), 7, ACCENT, -1, cv2.LINE_AA)
        if fx_layer.any():
            fx_layer = (fx_layer.astype(np.float32) * fade).astype(np.uint8)
            img = cv2.add(img, cv2.add(fx_layer, _glow(fx_layer)))
        # 名牌：打字机
        typed = _phase(i, TYPE_IN) if i < TYPE_IN[1] else 1.0
        shown = int(round(typed * (len(title) + len(sub))))  # 先打名字，再打下一行
        n_title, n_sub = min(len(title), shown), max(0, shown - len(title))
        if i >= TYPE_IN[0] and fade > 0:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
            over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(over)
            ax, ay = geo["anchor"]
            text_w = max(d.textlength(title, font=big), d.textlength(sub, font=small), d.textlength(team, font=mid)) + 40
            x0 = ax if geo["side"] > 0 else ax - text_w - 24
            x0 = min(max(16, x0), OUT_W - text_w - 40)
            y0 = max(16, ay - 200)
            alpha = int(255 * fade)
            box_w = (text_w + 24) * _ease((i - TYPE_IN[0]) / 5)
            d.rectangle([x0, y0, x0 + box_w, y0 + 205], fill=(10, 14, 22, int(190 * fade)))
            d.rectangle([x0, y0, x0 + 10, y0 + 205], fill=(255, 215, 0, alpha))
            cursor = "▍" if (i // 6) % 2 == 0 and i < TYPE_IN[1] + 8 else ""
            d.text((x0 + 26, y0 + 10), team, font=mid, fill=(255, 215, 0, alpha))
            d.text((x0 + 24, y0 + 50), title[:n_title] + (cursor if n_sub == 0 else ""), font=big, fill=(255, 255, 255, alpha))
            if n_sub:
                d.text((x0 + 28, y0 + 148), sub[:n_sub] + cursor, font=small, fill=(220, 226, 235, alpha))
            img = cv2.cvtColor(np.asarray(Image.alpha_composite(pil, over).convert("RGB")), cv2.COLOR_RGB2BGR)
        yield img


BAR_H, BAR_W = 92, OUT_W - 240  # 片段底部字幕条（右边留着比赛时间）


def caption_strip(text: str) -> np.ndarray:
    """片段底部字幕条的左边部分（片段渲染时写的是中文；集锦按所选语言重画，每段画一次逐帧贴上）。"""
    pil = Image.new("RGB", (BAR_W, BAR_H), (0, 0, 0))
    ImageDraw.Draw(pil).text((36, 14), text, font=font(44), fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def make_segment(clip: Path, marks: list[dict | None], title: str, team: str, sub: str, out: Path, intro: bool = True,
                 caption: str | None = None) -> None:
    """在 marks 里第一个找到当事人的那一帧 k 定格介绍（通常就是第一帧；他一开始不在画面里就先照常播放，
    等他出现再定格，不跳过画面；都没找到就定格第一帧、名字打在画面中间），然后接着放。
    intro=False 不定格，只画光圈；caption 给了就重画底部字幕条。开场之后脚下的光圈一直跟着他（marks 给每一帧的位置；短暂跟丢 RING_HOLD 帧以内停在原位，不闪）。边读边写，不占内存。"""
    strip = caption_strip(caption) if caption else None
    k = next((i for i, m in enumerate(marks) if m), None)
    mark = marks[k] if k is not None else None
    k = k or 0
    cap = cv2.VideoCapture(str(clip))
    writer = VideoWriter(out, FPS, enhance=False)
    last, gap = mark, 0
    try:
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if strip is not None:
                f = f.copy()
                f[OUT_H - BAR_H:, :BAR_W] = strip
            if i == k and intro:
                for img in intro_frames(f, mark, title, team, sub):
                    writer.write(img)
            if i >= k and mark:
                m = marks[i] if i < len(marks) else None
                if m:
                    last, gap = m, 0
                else:
                    gap += 1
                if last and gap <= RING_HOLD:
                    f = _add_glow(f, ring_layer(f.shape, last))
            writer.write(f)
            i += 1
    finally:
        cap.release()
        writer.close()
    if i <= k:
        raise RuntimeError(f"读不出片段 {clip}")


OUTRO_FRAMES = 210  # 结尾一共多少帧（8.4 秒），之后再留 OUTRO_BLACK 帧黑场
OUTRO_BLUR = (0, 18)  # 最后一帧定格、变暗变糊
OUTRO_TYPE = (14, 60)  # 名字、数据、比分一行行打出来
OUTRO_STATS_OUT = (120, 138)  # 数据淡出
OUTRO_CLOSE_IN = (138, 162)  # 收尾语（“队名 加油！”）淡入、慢慢放大
OUTRO_FADE = (180, 210)  # 慢慢淡到黑
OUTRO_BLACK = 12
OUTRO_ZOOM = 0.10  # 定格画面在整个结尾里缓慢推近这么多，不是一张死图


def _zoom(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def outro_frames(frame: np.ndarray, title: str, team: str, lines: list[str], closing: str = "", frames: int = OUTRO_FRAMES):
    """集锦结尾：最后一帧定格、压暗虚化并缓慢推近；打字机打出名字和几行字（本场数据、全场比分），
    停一会儿淡出，换上一句收尾语，最后慢慢淡到黑。"""
    blurred = cv2.GaussianBlur(frame, (0, 0), 14)
    big, mid, small, close_font = font(120), font(40), font(52), font(96)
    rows = [(team, mid, (255, 215, 0)), (title, big, (255, 255, 255))] + [(t, small, (225, 230, 238)) for t in lines]
    total = sum(len(t) for t, *_ in rows)
    heights = [44, 140] + [72] * len(lines)
    top = OUT_H / 2 - sum(heights) / 2 - 20
    for i in range(frames):
        b = _phase(i, OUTRO_BLUR)
        img = cv2.addWeighted(frame, 1 - b, blurred, b, 0)
        img = _zoom(img, 1 + OUTRO_ZOOM * i / frames)
        img = (img.astype(np.float32) * (1 - 0.6 * b)).astype(np.uint8)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        alpha = int(255 * (1 - _phase(i, OUTRO_STATS_OUT)))
        if i >= OUTRO_TYPE[0] and alpha > 0:
            shown = int(round(_phase(i, OUTRO_TYPE) * total))
            line_w = int(560 * _ease((i - OUTRO_TYPE[0]) / 8))  # 名字下面一道金线从中间展开
            d.rectangle([OUT_W / 2 - line_w / 2, top + 44 + 140 + 6, OUT_W / 2 + line_w / 2, top + 44 + 140 + 12],
                        fill=(255, 215, 0, alpha))
            y, cursor_done = top, False
            for (text, f, color), hgt in zip(rows, heights):
                n = min(len(text), shown)
                shown -= n
                part = text[:n]
                if not cursor_done and n < len(text) and i < OUTRO_TYPE[1]:
                    part += "▍" if (i // 6) % 2 == 0 else ""
                    cursor_done = True
                if part:
                    d.text((OUT_W / 2, y + hgt / 2), part, font=f, fill=(*color, alpha), anchor="mm")
                y += hgt + (24 if hgt == 140 else 0)
        if closing and i >= OUTRO_CLOSE_IN[0]:
            c = _phase(i, OUTRO_CLOSE_IN)
            f = font(int(96 * (0.85 + 0.15 * c)))
            d.text((OUT_W / 2, OUT_H / 2), closing, font=f, fill=(255, 215, 0, int(255 * c)), anchor="mm")
        img = cv2.cvtColor(np.asarray(Image.alpha_composite(pil, over).convert("RGB")), cv2.COLOR_RGB2BGR)
        img = (img.astype(np.float32) * (1 - _phase(i, OUTRO_FADE))).astype(np.uint8)
        yield img
    black = np.zeros_like(frame)
    for _ in range(OUTRO_BLACK):
        yield black


def make_outro(clip: Path, title: str, team: str, lines: list[str], out: Path, closing: str = "") -> None:
    """用 clip 的最后一帧做结尾。"""
    cap = cv2.VideoCapture(str(clip))
    last = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        last = f
    cap.release()
    if last is None:
        last = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    writer = VideoWriter(out, FPS, enhance=False)
    try:
        for img in outro_frames(last, title, team, lines, closing):
            writer.write(img)
    finally:
        writer.close()
