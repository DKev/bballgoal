import cv2
import numpy as np

from bballgoal.identify import assign_players
from bballgoal.jersey import six_or_nine, vote


def _digit(text):
    img = np.full((160, 120, 3), (30, 200, 230), np.uint8)
    cv2.putText(img, text, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 4, (20, 20, 20), 12)
    return img


def test_six_or_nine_uses_loop_position():
    assert six_or_nine(_digit("6")) == "6"
    assert six_or_nine(cv2.rotate(_digit("6"), cv2.ROTATE_180)) == "9"


def test_vote_needs_agreement():
    assert vote([("8", 1.0), ("8", 1.0), ("3", 1.0)])[0] == "8"
    assert vote([("8", 1.0)])[0] is None  # 只读到一次不采信
    assert vote([("6", 1.0), ("9", 1.0), ("6", 1.0), ("9", 1.0)])[0] is None


def _span(team, start, end, first=(0, 0, 100), last=(0, 0, 100)):
    return {"team": team, "start": start, "end": end, "first": first, "last": last, "frames": 10}


def test_assign_players_merges_same_number_and_stitches_neighbours():
    spans = {
        1: _span("A", 0, 10, last=(500, 1500, 200)),
        2: _span("A", 11, 20, first=(520, 1500, 200)),  # 没认出号码，紧接着 1 出现在附近
        3: _span("A", 30, 40),
        4: _span("A", 5, 15),  # 和 1 时间重叠却也读成 8：冲突，放弃
    }
    readings = {1: [("8", 1.0)] * 5, 3: [("8", 1.0)] * 3, 4: [("8", 1.0)] * 2}
    labels, _ = assign_players(spans, readings)
    assert labels[1] == labels[2] == labels[3] == "A#8"
    assert 4 not in labels


def test_split_by_readings_ignores_single_misreads():
    from bballgoal.segments import split_by_readings

    assert split_by_readings([(0, "6"), (2, "6"), (4, "9"), (6, "6"), (8, "6")]) == []
    assert split_by_readings([(0, "6"), (2, "6"), (10, "29"), (12, "29")]) == [6]


def test_change_points_ignores_brief_flips():
    from bballgoal.segments import change_points

    flicker = ["A"] * 20 + ["B"] * 2 + ["A"] * 20
    assert change_points(flicker, window=3, min_run=8) == []
    swap = ["A"] * 20 + [None] * 3 + ["B"] * 20
    cuts = change_points(swap, window=3, min_run=8)
    assert len(cuts) == 1 and 19 <= cuts[0] <= 24
