import numpy as np

from bballgoal.identify import player_sort_key
from bballgoal.identity import best_split, cluster_team, fit_projection, split_tracks_by_appearance, track_vectors


def _span(start, end, team="A"):
    seen = set(range(int(start * 10), int(end * 10) + 1))
    return {"team": team, "start": start, "end": end, "frames": len(seen), "seen": seen}


def _people(rng, n_people=3, tracks_per_person=3, dim=32):
    """每人一个外观中心，每条轨迹的样本 = 中心 + 噪声；同一人的轨迹时间上互不重叠。"""
    centers = rng.normal(size=(n_people, dim)) * 3
    samples, spans, owner = {}, {}, {}
    tid = 1
    for k in range(n_people):
        for j in range(tracks_per_person):
            samples[tid] = centers[k] + rng.normal(size=(8, dim))
            spans[tid] = _span(j * 10.0, j * 10.0 + 8)
            owner[tid] = k
            tid += 1
    return samples, spans, owner


def test_fit_projection_needs_enough_tracks():
    rng = np.random.default_rng(0)
    assert fit_projection({1: rng.normal(size=(8, 4)), 2: rng.normal(size=(8, 4))}) is None


def test_cluster_recovers_people():
    rng = np.random.default_rng(1)
    samples, spans, owner = _people(rng)
    proj = fit_projection(samples, dims=4, pca=16)
    vectors = track_vectors({"body": samples}, {"body": proj})
    clusters = cluster_team(list(spans), vectors, spans, {}, {t: 8.0 for t in spans})
    assert sorted(sorted(owner[t] for t in c) for c in clusters) == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]


def test_overlapping_tracks_never_merge():
    vectors = {1: {"body": np.array([1.0, 0.0])}, 2: {"body": np.array([1.0, 0.0])}}
    spans = {1: _span(0, 10), 2: _span(5, 15)}
    clusters = cluster_team([1, 2], vectors, spans, {}, {1: 1.0, 2: 1.0})
    assert sorted(map(sorted, clusters)) == [[1], [2]]


def test_different_numbers_never_merge_and_same_number_merges():
    same = np.array([1.0, 0.0])
    vectors = {1: {"body": same}, 2: {"body": same}, 3: {"body": np.array([0.0, 1.0])}}
    spans = {1: _span(0, 5), 2: _span(10, 15), 3: _span(20, 25)}
    weights = {1: 1.0, 2: 1.0, 3: 1.0}
    assert sorted(map(sorted, cluster_team([1, 2], vectors, spans, {1: "5", 2: "8"}, weights))) == [[1], [2]]
    # 外观不像但号码相同（例如换了个角度），也合并
    assert cluster_team([1, 3], vectors, spans, {1: "5", 3: "5"}, weights) == [[1, 3]]


def test_shoes_used_only_when_both_have_them():
    vectors = {
        1: {"body": np.array([1.0, 0.0]), "shoes": np.array([1.0, 0.0])},
        2: {"body": np.array([1.0, 0.0]), "shoes": np.array([-1.0, 0.0])},  # 鞋完全不同
        3: {"body": np.array([1.0, 0.0])},  # 没看到鞋
    }
    spans = {1: _span(0, 5), 2: _span(10, 15), 3: _span(20, 25)}
    weights = {t: 1.0 for t in spans}
    assert sorted(map(sorted, cluster_team([1, 2], vectors, spans, {}, weights))) == [[1], [2]]
    assert cluster_team([1, 3], vectors, spans, {}, weights) == [[1, 3]]


def test_player_sort_key():
    keys = ["B#5", "A?1", "A#12", "A#3", "B?2"]
    assert sorted(keys, key=player_sort_key) == ["A#3", "A#12", "A?1", "B#5", "B?2"]


def test_tracks_with_gap_overlap_can_merge():
    """首尾时间重叠但从没同框（中间消失了）的两条轨迹可以合并。"""
    v = {"body": np.array([1.0, 0.0])}
    a, b = _span(0, 10), _span(20, 25)
    a["end"] = 30  # a 的首尾跨过 b，但 b 出现时 a 并不在画面里
    clusters = cluster_team([1, 2], {1: v, 2: v}, {1: a, 2: b}, {}, {1: 1.0, 2: 1.0})
    assert clusters == [[1, 2]]


def test_best_split_finds_change():
    rng = np.random.default_rng(2)
    z = np.concatenate([rng.normal(size=(6, 4)) * 0.1 + [1, 0, 0, 0], rng.normal(size=(5, 4)) * 0.1 + [0, 1, 0, 0]])
    dist, s = best_split(z)
    assert s == 6 and dist > 0.8
    assert best_split(z[:6])[1] is None  # 太短不切


def test_split_tracks_by_appearance():
    """轨迹 1 前半是甲、后半是乙，切开；其他干净的轨迹（用来学投影，每人 3 条）不动。"""
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(4, 16)) * 3
    body, frames = [], []
    plan = {1: [0] * 8 + [1] * 8}
    for k in range(12):
        plan[k + 2] = [k % 4] * 16
    for i in range(16):
        people = []
        for tid, owners in plan.items():
            people.append({"id": tid, "sample": len(body)})
            body.append(centers[owners[i]] + rng.normal(size=16) * 0.3)
        frames.append({"people": people})
    assert split_tracks_by_appearance(frames, np.array(body), threshold=0.5) == 1
    ids = [f["people"][0]["id"] for f in frames]
    assert ids[:8] == [1] * 8 and len(set(ids[8:])) == 1 and ids[8] not in plan
