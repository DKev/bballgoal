import json


def _game(root, name, colors, people=None):
    d = root / name
    (d / "enroll" / "review").mkdir(parents=True)
    (d / "game.json").write_text(json.dumps({"name": name, "enroll_video": str(d / "enroll.mp4")}))
    (d / "enroll" / "tracks.json").write_text(json.dumps({"video": "x", "team_colors_bgr": colors, "tracks": {}, "frames": []}))
    if people:
        (d / "enroll" / "review" / "people.json").write_text(json.dumps(people))
    return json.loads((d / "game.json").read_text())


def test_roster_carries_over_by_jersey_color_even_when_team_letters_swap(tmp_path, monkeypatch):
    from bballgoal import app

    monkeypatch.setattr(app, "OUTPUT", tmp_path)
    _game(tmp_path, "g1", {"A": [40, 40, 40], "B": [50, 170, 230]},
          {"teams": {"A": {"name": "Hawks"}, "B": {"name": "Falcons"}},
           "people": [{"team": "A", "kind": "player", "number": "18", "name": "小明"},
                      {"team": "B", "kind": "player", "number": "5", "name": "X"}]})
    # 第二场：我们（深色）这回被分成了 B，对手是新的白色队
    g2 = _game(tmp_path, "g2", {"A": [235, 235, 235], "B": [45, 42, 38]})
    carry = app.carry_roster(g2)
    assert carry["from"] == "g1" and set(carry["teams"]) == {"B"}
    assert carry["teams"]["B"] == {"name": "Hawks", "roster": {"18": "小明"}}
