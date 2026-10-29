import cv2
import numpy as np

from bballgoal.court import Court
from bballgoal.teams import REFEREE, fit_teams, stripe_count, torso_feature


def _jersey(color_bgr, number=False, stripes=False, size=(200, 80)):
    h, w = size
    img = np.full((h, w, 3), color_bgr, np.uint8)
    if stripes:
        for x in range(0, w, 10):
            img[:, x : x + 5] = (20, 20, 20)
            img[:, x + 5 : x + 10] = (235, 235, 235)
    if number:
        cv2.putText(img, "8", (w // 2 - 14, h // 3 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 5)
    return img


def test_stripe_count_separates_referee_from_number():
    referee = _jersey((0, 0, 0), stripes=True)
    maroon_with_number = _jersey((40, 30, 110), number=True)
    assert stripe_count(referee) >= 5
    assert stripe_count(maroon_with_number) <= 3


def test_fit_teams_splits_yellow_maroon_and_referee():
    box = (0, 0, 80, 200)
    features = {}
    for i in range(4):
        features[i] = torso_feature(_jersey((30, 200, 230)), box)  # 黄
        features[10 + i] = torso_feature(_jersey((40, 30, 110), number=True), box)  # 酒红带白号
    features[99] = torso_feature(_jersey((0, 0, 0), stripes=True), box)  # 裁判
    model = fit_teams(features)
    teams = {tid: model.assign(f) for tid, f in features.items()}
    assert teams[99] == REFEREE
    assert len({teams[i] for i in range(4)}) == 1
    assert len({teams[10 + i] for i in range(4)}) == 1
    assert teams[0] != teams[10]


def test_court_contains_and_y_range():
    court = Court(polygon=[(0, 1000), (100, 1000), (100, 2000), (0, 2000)], hoops=[])
    assert court.contains(50, 1500)
    assert not court.contains(50, 900)
    assert court.y_range(2160) == (550, 2000)
