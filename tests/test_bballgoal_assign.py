import numpy as np

from bballgoal.assign import build_gallery, merge_samples, similarities


def _samples(rng, centers, owners, n=6):
    body = {t: centers[o] + rng.normal(size=(n, centers.shape[1])) * 0.3 for t, o in owners.items()}
    return {"body": body, "shoes": {}}


def test_answers_join_gallery_and_new_people_are_created():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16)) * 3
    gal = _samples(rng, centers, {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2})
    full = _samples(rng, centers, {1: 0, 2: 3, 3: 3})  # 全场：1 是甲，2、3 是名单里没有的丁
    gal_people = [{"id": "A#1", "tracks": [1, 2]}, {"id": "A#2", "tracks": [3, 4]}, {"id": "A#3", "tracks": [5, 6]}]
    people, combined = merge_samples(gal_people, gal, full, {"1": "A#1", "2": "new"}, {2: "A"})
    ids = [p["id"] for p in people]
    assert ids == ["A#1", "A#2", "A#3", "新1"]
    assert "f1" in people[0]["tracks"] and people[3]["tracks"] == ["f2"] and people[3]["team"] == "A"
    gallery = build_gallery(people, combined)
    # 第 3 段（丁）现在能认成“新1”
    scores = similarities(gallery, {"body": full["body"][3]}, None, None)
    assert max(scores, key=scores.get) == "新1"


def test_new_answers_with_number_group_into_one_person_or_join_roster():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(3, 16)) * 3
    gal = _samples(rng, centers, {1: 0, 2: 1})
    full = _samples(rng, centers, {1: 2, 2: 2, 3: 1})
    gal_people = [{"id": "c1", "kind": "player", "team": "A", "number": "7", "tracks": [1]},
                  {"id": "c2", "kind": "player", "team": "B", "number": "9", "tracks": [2]}]
    answers = {"1": "new:A:23", "2": "new:A:23", "3": "new:B:9"}  # 两段同一个新的 A#23；B#9 名单里已有
    people, _ = merge_samples(gal_people, gal, full, answers, {})
    by_id = {p["id"]: p for p in people}
    assert by_id["A#23"]["tracks"] == ["f1", "f2"] and by_id["A#23"]["from_answers"] == [1, 2]
    assert "f3" in by_id["c2"]["tracks"] and len(people) == 3


def _frames(spans):
    """spans: 轨迹号 → (开始帧, 结束帧, 起点 x, 终点 x)；人高 100。"""
    n = max(e for _s, e, *_ in spans.values()) + 1
    frames = [{"time": i / 10, "people": []} for i in range(n)]
    for tid, (s, e, x0, x1) in spans.items():
        for i in range(s, e + 1):
            x = x0 + (x1 - x0) * (i - s) / max(1, e - s)
            frames[i]["people"].append({"id": tid, "team": "A", "box": [x - 20, 100, x + 20, 200]})
    return {"frames": frames, "fps": 10}


def test_stitch_links_fragment_to_neighbour_only_when_unambiguous():
    from bballgoal.assign import stitch, track_info

    # 1 是 #18，结束后 0.2 秒、在原地冒出 2（没外观）；3 是 #5，离得远
    result = _frames({1: (0, 20, 100, 300), 2: (23, 40, 305, 500), 3: (0, 40, 900, 900), 4: (23, 40, 1500, 1600)})
    out = {t: {"person": p, "reason": None if p else "no_features", "guess": []}
           for t, p in {1: "A#18", 2: None, 3: "A#5", 4: None}.items()}
    people = {"A#18": {"team": "A", "kind": "player"}, "A#5": {"team": "A", "kind": "player"}}
    assert stitch(track_info(result), out, people) == 1
    assert out[2]["person"] == "A#18" and out[2]["via"] == "stitch"
    assert out[4]["person"] is None  # 附近没有刚结束的人


def test_backfill_plans_only_small_unsampled_people_of_thin_tracks():
    from bballgoal.backfill import plan

    result = _frames({1: (0, 9, 100, 100), 2: (0, 9, 500, 500)})
    for f in result["frames"]:
        for p in f["people"]:
            if p["id"] == 1:
                p["sample"] = 0  # 1 号已经有足够样本
    todo = plan(result, min_height=50, enough=4)
    assert todo and all(result["frames"][i]["people"][k]["id"] == 2 for i, ks in todo.items() for k in ks)


def test_track_chain_links_earlier_fragments_of_same_person():
    from bballgoal.clips import track_chain

    # 1 断成了 1→2→3；4 在远处
    result = _frames({1: (0, 10, 100, 200), 2: (13, 20, 205, 300), 3: (22, 40, 305, 500), 4: (0, 40, 900, 900)})
    assert track_chain(result, 3, t_min=0.0) == {1, 2, 3}
    assert track_chain(result, 3, t_min=1.5) == {2, 3}  # 接到片段开头之前就够了


def test_track_chain_links_later_fragments_too():
    from bballgoal.clips import track_chain

    result = _frames({1: (0, 10, 100, 200), 2: (13, 20, 205, 300), 3: (22, 40, 305, 500), 4: (0, 40, 900, 900)})
    assert track_chain(result, 1, t_min=0.0, t_max=4.0) == {1, 2, 3}
