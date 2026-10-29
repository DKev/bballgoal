"""给集锦现场合成一段背景音乐（不用任何外部素材）：轻快的游戏感配乐——鼓、贝斯、铺底和弦、8-bit 琶音。

长度和集锦对齐：片头只有和弦和琶音（前奏），正片鼓和贝斯进来，结尾数据淡出时鼓停，
最后一个和弦拖长、跟着画面淡到无声。每个孩子用自己的 id 做随机种子，调子、和弦走向、速度都不一样。

    write_music(path, total, intro, outro, seed)
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

SR = 44100
KEYS = [57, 60, 62, 64, 55, 53]  # 主音（MIDI）：A3 C4 D4 E4 G3 F3
PROGRESSIONS = [  # (相对主音的半音, 大三 / 小三)
    [(0, "M"), (7, "M"), (9, "m"), (5, "M")],  # I V vi IV
    [(9, "m"), (5, "M"), (0, "M"), (7, "M")],  # vi IV I V
    [(0, "M"), (9, "m"), (5, "M"), (7, "M")],  # I vi IV V
    [(0, "M"), (5, "M"), (9, "m"), (7, "M")],  # I IV vi V
]
ARP_PATTERNS = [[0, 1, 2, 1, 0, 1, 2, 3], [0, 2, 1, 2, 0, 2, 3, 2], [0, 1, 2, 3, 2, 1, 2, 1]]


def _freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """滑动平均低通（快，够用）。"""
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _env(n: int, attack: float, release: float) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(attack, 1e-4))
    tail = np.clip((n / SR - t) / max(release, 1e-4), 0, 1)
    return e * tail


def _add(buf: np.ndarray, start: float, sig: np.ndarray) -> None:
    i = int(start * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def _saw(f: float, n: int, harmonics: int = 10) -> np.ndarray:
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, harmonics + 1) if k * f < SR / 2) * 0.6


def _pulse(f: float, n: int, duty: float = 0.25) -> np.ndarray:
    t = np.arange(n) / SR
    return np.where((t * f) % 1.0 < duty, 1.0, -1.0)


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = 48 + 90 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)


def _snare(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - _smooth(noise, 6)  # 去掉低频，像沙沙的响弦
    return 0.55 * noise * np.exp(-t * 18) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22)


def _hat(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - _smooth(noise, 3)
    return noise * np.exp(-t * 70)


def _gain_curve(n: int, points: list[tuple[float, float]]) -> np.ndarray:
    """按 (秒, 音量) 折线插值的音量曲线。"""
    t = np.arange(n) / SR
    xs, ys = zip(*sorted(points))
    return np.interp(t, xs, ys)


def compose(total: float, intro: float, outro: float, seed: str) -> np.ndarray:
    """返回立体声浮点音频（n×2）。intro：片头秒数（只有前奏）；outro：结尾秒数（鼓停、和弦淡出）。"""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % 2**32)
    key = KEYS[h % len(KEYS)]
    prog = PROGRESSIONS[(h // 7) % len(PROGRESSIONS)]
    arp = ARP_PATTERNS[(h // 31) % len(ARP_PATTERNS)]
    bpm = 108 + (h // 97) % 17  # 108–124
    beat = 60.0 / bpm
    bar = 4 * beat
    n = int((total + 0.5) * SR)
    drums, bass, pad, lead = (np.zeros(n) for _ in range(4))
    kick_env = np.zeros(n)

    bars = int(np.ceil(total / bar)) + 1
    for b in range(bars):
        t0 = b * bar
        if t0 > total:
            break
        root, quality = prog[b % len(prog)]
        third = 4 if quality == "M" else 3
        chord = [key + root, key + root + third, key + root + 7, key + root + 12]
        # 铺底和弦：两个微微失谐的锯齿波，慢起慢收
        m = int(bar * SR)
        for note in chord[:3]:
            for detune in (-0.06, 0.06):
                pad_note = _saw(_freq(note + detune), m, harmonics=6) * _env(m, 0.25, 0.3)
                _add(pad, t0, 0.07 * pad_note)
        # 贝斯：根音八分音符，低八度
        for k in range(8):
            m = int(beat / 2 * 0.9 * SR)
            note = key + root - 24 + (12 if k % 4 == 3 else 0)
            _add(bass, t0 + k * beat / 2, 0.32 * _saw(_freq(note), m, harmonics=8) * _env(m, 0.005, 0.08))
        # 8-bit 琶音：十六分音符
        for k in range(16):
            m = int(beat / 4 * 0.85 * SR)
            note = chord[arp[k % len(arp)]] + 12
            _add(lead, t0 + k * beat / 4, 0.10 * _pulse(_freq(note), m) * _env(m, 0.003, 0.05))
        # 鼓：1、3 拍底鼓，2、4 拍军鼓，八分音符踩镲（每两小节最后加一个底鼓）
        for k in range(4):
            m = int(0.35 * SR)
            if k in (0, 2) or (b % 2 == 1 and k == 3):
                kk = _kick(m)
                _add(drums, t0 + k * beat + (beat / 2 if k == 3 else 0), 0.9 * kk)
                _add(kick_env, t0 + k * beat, np.exp(-np.arange(m) / SR * 10))
            if k in (1, 3):
                _add(drums, t0 + k * beat, 0.45 * _snare(int(0.25 * SR), rng))
        for k in range(8):
            _add(drums, t0 + k * beat / 2, (0.12 if k % 2 else 0.07) * _hat(int(0.08 * SR), rng))

    lead = _smooth(lead, 3)  # 方波稍微磨圆
    pad *= 1 - 0.35 * np.clip(kick_env, 0, 1)  # 底鼓一响和弦让一让，律动感

    end = total
    out_start = total - outro
    stats_out = out_start + 0.55 * outro  # 结尾里数据淡出、收尾语出现的时候
    g_drums = _gain_curve(n, [(0, 0), (max(0, intro - 0.05), 0), (intro, 1), (stats_out - 0.4, 1), (stats_out, 0), (end + 1, 0)])
    g_bass = g_drums
    g_lead = _gain_curve(n, [(0, 0.5), (intro, 1), (out_start, 1), (stats_out, 0.3), (end - 1.2, 0), (end + 1, 0)])
    g_pad = _gain_curve(n, [(0, 0), (0.6, 1), (end - 2.2, 1), (end - 0.3, 0), (end + 1, 0)])
    left = drums * g_drums + bass * g_bass + pad * g_pad + lead * g_lead * 0.8
    right = drums * g_drums + bass * g_bass + pad * g_pad * 0.9 + lead * g_lead
    mix = np.stack([left, right], axis=1)
    mix = np.tanh(mix * 1.2)  # 软限幅
    peak = np.max(np.abs(mix)) or 1.0
    return mix / peak * 0.8


def write_music(path: Path, total: float, intro: float, outro: float, seed: str) -> None:
    audio = compose(total, intro, outro, seed)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
