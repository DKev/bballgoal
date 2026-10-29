"""把所有轨迹聚成固定的若干个人。

1. 每场比赛单独学一个投影（LDA）：同一条轨迹里的样本是同一个人，不同轨迹大多是不同的人。
   投影后去掉了朝向、远近、动作带来的变化，只留下区分人的部分。不需要任何人工标注。
2. 同队的轨迹按外观逐步合并（每次合并最像的两组），但：
   - 同一帧里同时出现过的两条轨迹一定是不同的人，永远不合并；
   - 号码认得很确定且不同的，不合并；号码相同的优先合并。
3. 合并后用这个人所有轨迹的号码读数一起投票，比单条轨迹可靠得多。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("body", "shoes")
MIN_SAMPLES = 2  # 一条轨迹某种特征至少这么多样本才参与比较
MERGE_THRESHOLD = 0.55  # 投影空间里的余弦距离；超过就认为是不同的人
MIN_COOCCUR = 3  # 两条轨迹在至少这么多帧里同时出现，才算“不同的人”（少量重叠是切分边界或重复框）
OUTSIDER_SIMILARITY = 0.75  # 和本队平均外观的余弦相似度低于这个算外人（实测中位数 0.91）
SPLIT_THRESHOLD = 0.55  # 轨迹前后两段外观的距离超过这个就切开（实测同一人前后 90% 在 0.42 以下）
SPLIT_MIN_SIDE = 4  # 每段至少这么多外观样本才考虑切


@dataclass
class Projection:
    mean: np.ndarray
    matrix: np.ndarray  # 原始维度 × 投影维度

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) @ self.matrix


def fit_projection(samples: dict[int, np.ndarray], dims: int = 16, pca: int = 64, min_per_track: int = 5) -> Projection | None:
    """samples: 轨迹号 → 该轨迹的样本矩阵。先 PCA 降维，再做 LDA（类 = 轨迹）。"""
    groups = [v for v in samples.values() if len(v) >= min_per_track]
    if len(groups) < 3:
        return None
    x = np.concatenate(groups).astype(np.float64)
    mean = x.mean(axis=0)
    _u, _s, vt = np.linalg.svd(x - mean, full_matrices=False)
    pca_m = vt[: min(pca, len(vt))].T
    z_groups = [(g - mean) @ pca_m for g in groups]
    within = sum((z - z.mean(axis=0)).T @ (z - z.mean(axis=0)) for z in z_groups) / len(x)
    means = np.stack([z.mean(axis=0) for z in z_groups])
    between = np.cov(means.T)
    reg = 1e-3 * np.trace(within) / len(within)
    values, vectors = np.linalg.eig(np.linalg.solve(within + reg * np.eye(len(within)), between))
    # LDA 最多只有“类数 - 1”个有意义的方向，多取的是噪声，会稀释距离
    order = np.argsort(-values.real)[: min(dims, len(groups) - 1)]
    return Projection(mean, pca_m @ vectors[:, order].real)


def track_vectors(
    samples: dict[str, dict[int, np.ndarray]], projections: dict[str, Projection | None]
) -> dict[int, dict[str, np.ndarray]]:
    """每条轨迹每种特征：投影后取平均，再归一化。"""
    out: dict[int, dict[str, np.ndarray]] = {}
    for name, per_track in samples.items():
        proj = projections.get(name)
        if proj is None:
            continue
        for tid, x in per_track.items():
            if len(x) < MIN_SAMPLES:
                continue
            v = proj(x).mean(axis=0)
            out.setdefault(tid, {})[name] = v / max(1e-9, float(np.linalg.norm(v)))
    return out


def cluster_team(
    tids: list[int],
    vectors: dict[int, dict[str, np.ndarray]],
    spans: dict[int, dict],
    numbers: dict[int, str | None],
    weights: dict[int, float],
    threshold: float = MERGE_THRESHOLD,
) -> list[list[int]]:
    """同队轨迹聚类（中心连接）。必须有整体外观特征的轨迹才参与。"""
    tids = [t for t in tids if "body" in vectors.get(t, {})]
    n = len(tids)
    if n == 0:
        return []
    members = [[t] for t in tids]
    # 每组的特征和（按样本量加权），合并时直接相加，归一化后就是中心方向
    sums = {}
    for name in FEATURES:
        dim = _dim(vectors, name)
        if dim:
            sums[name] = np.stack([vectors[t][name] * weights[t] if name in vectors[t] else np.zeros(dim) for t in tids])
    # 只看真正同框的帧：按首尾时间算会把轨迹中间消失的时段也当成“同时出现”
    cannot = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if len(spans[tids[i]]["seen"] & spans[tids[j]]["seen"]) >= MIN_COOCCUR:
                cannot[i, j] = cannot[j, i] = True
    labels = [numbers.get(t) for t in tids]
    for i in range(n):
        for j in range(n):
            if labels[i] and labels[j] and labels[i] != labels[j]:
                cannot[i, j] = True
    np.fill_diagonal(cannot, True)
    alive = np.ones(n, bool)

    while True:
        dist = _centroid_distances(sums)
        same_number = np.array([[bool(a) and a == b for b in labels] for a in labels])
        dist = np.where(same_number, dist - 1.0, dist)  # 同号优先合并
        dist[cannot | ~alive[:, None] | ~alive[None, :]] = np.inf
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if not np.isfinite(dist[i, j]) or dist[i, j] > threshold:
            break
        members[i] += members[j]
        members[j] = []
        for s in sums.values():
            s[i] += s[j]
            s[j] = 0
        cannot[i] |= cannot[j]
        cannot[:, i] |= cannot[:, j]
        labels[i] = labels[i] or labels[j]
        alive[j] = False
    return [m for m in members if m]


def _dim(vectors: dict[int, dict[str, np.ndarray]], name: str) -> int:
    for v in vectors.values():
        if name in v:
            return len(v[name])
    return 0


def _centroid_distances(sums: dict[str, np.ndarray]) -> np.ndarray:
    """各特征中心的余弦距离取平均；某组缺这种特征时不计这一项。必须有整体外观。"""
    total, count = 0.0, 0
    for name, s in sums.items():
        norm = np.linalg.norm(s, axis=1)
        has = norm > 1e-9
        unit = s / np.where(has, norm, 1)[:, None]
        d = 1 - unit @ unit.T
        both = has[:, None] & has[None, :]
        total = total + np.where(both, d, 0)
        count = count + both
    return np.where(count > 0, total / np.maximum(count, 1), np.inf)


def refine_teams(frames: list[dict], body: np.ndarray, rounds: int = 2, min_train: int = 6, min_prob: float = 0.6) -> int:
    """用整体外观重新判断每条轨迹属于哪队（含裁判）。返回改了几条。

    躯干颜色会被背景（红墙）和过曝带偏：酒红球衣在红墙前像黄色，黄色过曝像白字酒红。
    OSNet 外观对背景不敏感：用颜色判断的长轨迹当训练样本，每场训练一个分类器，
    再逐条轨迹平均预测概率；足够确定且和原来不同就改。改完再训练一轮。
    """
    from sklearn.linear_model import LogisticRegression

    rows: dict[int, list[int]] = {}
    entries: dict[int, list[dict]] = {}
    for f in frames:
        for p in f["people"]:
            entries.setdefault(p["id"], []).append(p)
            if "sample" in p:
                rows.setdefault(p["id"], []).append(p["sample"])
    team = {tid: items[0]["team"] for tid, items in entries.items()}
    changed: set[int] = set()
    for _ in range(rounds):
        train = [(tid, r) for tid, r in rows.items() if len(r) >= min_train and team[tid] in ("A", "B", "ref")]
        labels = {team[tid] for tid, _r in train}
        if not {"A", "B"} <= labels:
            break
        x = np.concatenate([body[r] for _tid, r in train]).astype(np.float32)
        y = [team[tid] for tid, r in train for _ in r]
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(x, y)
        for tid, r in rows.items():
            prob = clf.predict_proba(body[r].astype(np.float32)).mean(axis=0)
            best = str(clf.classes_[int(prob.argmax())])
            if best != team[tid] and prob.max() >= min_prob:
                team[tid] = best
                changed.add(tid)
    # 外人：和本队所有人的平均外观都差得远（场边教练、路过的别队孩子），不算任何一队
    centers = {}
    for name in ("A", "B"):
        member_rows = [i for tid, r in rows.items() if team[tid] == name for i in r]
        if member_rows:
            c = body[member_rows].astype(np.float32).mean(axis=0)
            centers[name] = c / max(1e-9, float(np.linalg.norm(c)))
    for tid, r in rows.items():
        if team[tid] in centers and len(r) >= 2:
            m = body[r].astype(np.float32).mean(axis=0)
            if float(m @ centers[team[tid]]) / max(1e-9, float(np.linalg.norm(m))) < OUTSIDER_SIMILARITY:
                team[tid] = "?"
                changed.add(tid)
    for tid in changed:
        for p in entries[tid]:
            p["team"] = team[tid]
    return len(changed)



def best_split(z: np.ndarray, min_side: int = SPLIT_MIN_SIDE) -> tuple[float, int | None]:
    """z 是按时间排列的投影后样本。返回 (前后两段平均方向的最大余弦距离, 切分位置)。"""
    best: tuple[float, int | None] = (0.0, None)
    if len(z) < 2 * min_side:
        return best
    csum = np.cumsum(z, axis=0)
    total = csum[-1]
    for s in range(min_side, len(z) - min_side + 1):
        a, b = csum[s - 1], total - csum[s - 1]
        d = 1 - float(a @ b) / max(1e-9, float(np.linalg.norm(a) * np.linalg.norm(b)))
        if d > best[0]:
            best = (d, s)
    return best


def split_tracks_by_appearance(frames: list[dict], body: np.ndarray, threshold: float = SPLIT_THRESHOLD) -> int:
    """跟踪器把两个人接成一条轨迹（同队、号码又没认出时，前面的切分都发现不了）：
    沿轨迹看外观样本，前后两段明显是两个人就在中间切开，切出来的段再继续检查。原地改写 frames。"""
    occurrences: dict[int, list[tuple[int, dict]]] = {}
    for i, f in enumerate(frames):
        for p in f["people"]:
            occurrences.setdefault(p["id"], []).append((i, p))
    rows = {
        tid: [p["sample"] for _i, p in items if "sample" in p] for tid, items in occurrences.items()
    }
    proj = fit_projection({tid: body[r].astype(np.float32) for tid, r in rows.items() if r})
    if proj is None:
        return 0
    next_id = max(occurrences, default=0) + 1
    splits = 0
    pending = list(occurrences)
    while pending:
        tid = pending.pop()
        items = occurrences[tid]
        sampled = [(i, p["sample"]) for i, p in items if "sample" in p]
        dist, s = best_split(proj(body[[r for _i, r in sampled]]))
        if s is None or dist <= threshold:
            continue
        cut = (sampled[s - 1][0] + sampled[s][0] + 1) // 2  # 两个样本帧的中点
        tail = [(i, p) for i, p in items if i >= cut]
        for _i, p in tail:
            p["id"] = next_id
        occurrences[tid] = [(i, p) for i, p in items if i < cut]
        occurrences[next_id] = tail
        pending += [tid, next_id]
        next_id += 1
        splits += 1
    return splits
