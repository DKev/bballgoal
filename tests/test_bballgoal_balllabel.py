import json

import numpy as np


def _game(root, name, labels, model=False):
    import cv2

    balls = root / name / "balls"
    (balls / "imgs").mkdir(parents=True)
    items = []
    for k in labels:
        cv2.imwrite(str(balls / k), np.zeros((64, 64, 3), np.uint8))
        items.append({"file": k, "w": 64, "h": 64, "person_height": 30})
    (balls / "items.json").write_text(json.dumps({"items": items}))
    (balls / "labels.json").write_text(json.dumps({"labels": labels}))
    if model:
        (balls / "model.pt").write_bytes(b"x")
        (balls / "train.json").write_text(json.dumps({"recall_new": 0.5, "conf": 0.05}))
    return balls


def test_new_game_inherits_latest_model_and_trains_with_older_labels(tmp_path):
    from bballgoal.balllabel import build_dataset, inherit

    _game(tmp_path, "old", {f"imgs/o{i}.jpg": {"ball": [10, 10]} for i in range(5)}, model=True)
    new = _game(tmp_path, "new", {f"imgs/n{i}.jpg": {} for i in range(5)})
    inherit(new)
    assert (new / "model.pt").exists() and json.loads((new / "train.json").read_text())["inherited_from"] == "old"
    _, n, with_ball = build_dataset(new)
    assert n == 10 and with_ball == 5
    val = list((new / "dataset" / "images" / "val").glob("*.jpg"))
    assert val and all(p.name.startswith("n") for p in val)  # 验证集只用这一场的
