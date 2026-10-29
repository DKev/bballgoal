from bballgoal.ball import holder_of, people_at, possessions, smooth_holders


def _person(pid, box, player, team="A"):
    return {"id": pid, "box": box, "team": team, "player": player}


def test_holder_is_nearest_hand_area():
    entry = {"people": [_person(1, [100, 100, 150, 250], "A#1"), _person(2, [300, 100, 350, 250], "B#2")]}
    assert holder_of([160, 200, 5, 0.9], entry) == "A#1"  # 身体右侧、腰部高度
    assert holder_of([325, 245, 5, 0.9], entry) == "B#2"  # 运球：脚边
    assert holder_of([230, 50, 5, 0.9], entry) is None  # 空中，谁都不挨着
    assert holder_of(None, entry) is None


def test_holder_ignores_unlabeled_people():
    entry = {"people": [_person(1, [100, 100, 150, 250], None)]}
    assert holder_of([125, 180, 5, 0.9], entry) is None


def test_smooth_fills_short_gaps_and_drops_blips():
    fps = 10  # FILL_GAP 0.5s = 5 帧，MIN_HOLD 0.2s = 2 帧
    h = ["A#1", None, None, "A#1", None, "B#2", None, None, None, None, None, None, None, "A#1", "A#1"]
    out = smooth_holders(h, fps)
    assert out[:4] == ["A#1"] * 4  # 中间丢 2 帧补上
    assert out[5] is None  # B#2 只出现 1 帧，去掉
    assert out[13:] == ["A#1", "A#1"]


def test_possessions_mark_receptions():
    h = ["A#1", "A#1", None, "A#2", "A#2", None, "A#2"]
    t = [i / 10 for i in range(len(h))]
    poss = possessions(h, t)
    assert [(p["player"], p["received"]) for p in poss] == [("A#1", False), ("A#2", True), ("A#2", False)]


def test_people_at_interpolates_boxes():
    frames = [
        {"time": 0.0, "people": [_person(1, [0, 0, 10, 20], "A#1")]},
        {"time": 0.1, "people": [_person(1, [10, 0, 20, 20], "A#1")]},
    ]
    box = people_at(frames, 10, 0.05)["people"][0]["box"]
    assert box == [5, 0, 15, 20]


def test_low_confidence_track_survives_when_miss_cost_follows_threshold():
    from bballgoal.ball import best_path

    # 训练过的模型：真球的置信度常在 0.2 左右（老模型的 0.3 代价之下），稳稳地每帧挪一点
    frames = [{"time": i / 25, "people": [{"id": 1, "team": "A", "box": [90, 100, 130, 250]}]} for i in range(50)]
    cands = [[[100 + i, 150, 12, 0.2]] for i in range(50)]
    assert sum(b is not None for b in best_path(cands, frames, 25)) == 0  # 固定 0.3：全当看不到
    assert sum(b is not None for b in best_path(cands, frames, 25, miss_cost=0.12 * 0.3 / 0.25)) > 40
