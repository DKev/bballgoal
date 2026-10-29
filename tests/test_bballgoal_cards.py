import numpy as np

from bballgoal.cards import card_key, crop, pick_frames


def test_pick_frames_spreads_over_time_and_prefers_big_boxes():
    items = [(i, i / 10, [0, 0, 10, 10 + (5 if i in (2, 8) else 0)]) for i in range(10)]
    picked = pick_frames(items, 2)
    assert [p[0] for p in picked] == [2, 8]
    assert len(pick_frames(items[:1], 3)) == 1
    assert pick_frames([], 3) == []


def test_card_key():
    assert card_key("A", "A#18") == "A#18"
    assert card_key("ref", None) == "裁判"
    assert card_key("?", None) == "未分队"
    assert card_key("B", None) == "B?"


def test_crop_clamps_to_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    out = crop(frame, [190, 80, 210, 120], 50, widen=4, taller=2)
    assert out.shape[0] == 50 and out.shape[1] >= 1


def test_game_clock_maps_enroll_time(tmp_path):
    import json

    from bballgoal.cards import game_clock

    video = tmp_path / "x_enroll.mp4"
    video.with_suffix(".enroll.json").write_text(json.dumps({"segments": [
        {"start": 0, "end": 40, "game_start": 100}, {"start": 44, "end": 84, "game_start": 900}]}))
    clock = game_clock(video)
    assert clock(10) == 110 and clock(50) == 906
    assert game_clock(tmp_path / "plain.mp4")(10) == 10
