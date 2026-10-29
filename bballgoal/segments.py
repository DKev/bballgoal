"""把串了人的轨迹切开。

跟踪器在两人交错时可能把一条轨迹从一个人接到另一个人身上。这里沿轨迹看每帧的
判断（队伍、号码），出现持续的变化就在那里切开，切出来的每段当作独立的轨迹。
单帧判断有噪声（被挡住时躯干会截到别人），所以先做时间平滑、忽略短暂跳变。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

ABSTAIN = None


def smooth_labels(labels: Sequence[Hashable | None], window: int) -> list[Hashable | None]:
    """滑动窗口多数投票，ABSTAIN（None）不参与投票。"""
    out = []
    for i in range(len(labels)):
        votes = Counter(l for l in labels[max(0, i - window) : i + window + 1] if l is not ABSTAIN)
        out.append(votes.most_common(1)[0][0] if votes else ABSTAIN)
    return out


def change_points(labels: Sequence[Hashable | None], window: int, min_run: int) -> list[int]:
    """返回切分位置（新段起点的下标）。短于 min_run 的段并入前一段，不切。"""
    smoothed = smooth_labels(labels, window)
    # 把 ABSTAIN 用前一个有效值填上，开头的用第一个有效值
    first = next((l for l in smoothed if l is not ABSTAIN), ABSTAIN)
    filled, last = [], first
    for l in smoothed:
        last = l if l is not ABSTAIN else last
        filled.append(last)

    runs: list[list] = []  # [label, start, length]
    for i, l in enumerate(filled):
        if runs and runs[-1][0] == l:
            runs[-1][2] += 1
        else:
            runs.append([l, i, 1])
    merged: list[list] = []
    for run in runs:
        if merged and (run[2] < min_run or merged[-1][0] == run[0]):
            merged[-1][2] += run[2]
        else:
            merged.append(run)
    # 第一段太短时并入第二段
    if len(merged) > 1 and merged[0][2] < min_run:
        merged[1][1], merged[1][2] = 0, merged[1][2] + merged[0][2]
        merged.pop(0)
    return [run[1] for run in merged[1:]]


def majority(labels: Sequence[Hashable | None]) -> Hashable | None:
    votes = Counter(l for l in labels if l is not ABSTAIN)
    return votes.most_common(1)[0][0] if votes else ABSTAIN


def split_by_readings(readings: Sequence[tuple[int, Hashable]], min_count: int = 2) -> list[int]:
    """按稀疏读数（如号码）切分。readings 是 (位置下标, 读数)，按位置排序。

    连续相同的读数算一段；少于 min_count 次的段当噪声丢掉；相邻同值段合并。
    剩下的段之间取中点作为切分位置。
    """
    runs: list[list] = []  # [读数, 次数, 首个位置, 末个位置]
    for index, value in readings:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
            runs[-1][3] = index
        else:
            runs.append([value, 1, index, index])
    kept: list[list] = []
    for run in runs:
        if run[1] < min_count:
            continue
        if kept and kept[-1][0] == run[0]:
            kept[-1][1] += run[1]
            kept[-1][3] = run[3]
        else:
            kept.append(run)
    return [(a[3] + b[2] + 1) // 2 for a, b in zip(kept, kept[1:])]
