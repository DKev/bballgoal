from bballgoal.events import approach_windows, attribute
from bballgoal.reels import reviewed_events, stats

RIM = {"center_x": 100.0, "rim_y": 100.0, "width": 40.0}
RIMS = [{"center_x": 100.0, "rim_y": 100.0, "width": 20.0}]


def _poss(player, start, end, received=True):
    return {"player": player, "start": start, "end": end, "received": received}


def _result(boxes):
    """boxes: {(时间, 人): 框}"""
    frames = {}
    for (t, who), box in boxes.items():
        frames.setdefault(t, []).append({"id": hash(who) % 1000, "box": box, "team": who[0], "player": who})
    return {"fps": 10, "frames": [{"time": t, "people": ps} for t, ps in sorted(frames.items())]}


def test_made_shot_gets_shooter_as_two_pointer():
    poss = [_poss("A#3", 0.0, 2.0), _poss("A#18", 2.8, 4.0)]
    result = _result({(4.0, "A#18"): [90, 120, 110, 160]})  # 就在篮筐下面
    shots = [{"time": 5.0, "hoop": 0, "result": "made", "reason": None, "net_motion": 3.0}]
    ev = attribute(shots, poss, result, RIMS, {})
    shot = next(e for e in ev if e["type"] == "shot")
    assert shot["player"] == "A#18" and shot["team"] == "A" and shot["kind"] == "two"
    assert {e["type"] for e in ev} == {"shot"}  # 篮板、助攻、抢断不做了


def test_approach_windows_merge_and_pad():
    def ball(y):
        return [{"name": "ball", "confidence": 0.8, "xyxy": [95, y - 5, 105, y + 5]}]
    frames = [[] for _ in range(200)]
    for i in (50, 55, 60):  # 球在篮圈上方
        frames[i] = ball(100 - 40)
    frames[150] = ball(100 - 40)
    w = approach_windows(frames, RIM, 25)
    assert len(w) == 2 and w[0][0] < 50 < 60 < w[0][1]


def test_review_merge_and_stats():
    events = [{"id": 1, "type": "shot", "time": 1, "player": "A#18", "result": "missed"},
              {"id": 2, "type": "rebound", "time": 2, "player": "A#18"},
              {"id": 3, "type": "steal", "time": 3, "player": "B#5"}]
    review = {"events": {"1": {"result": "made", "points": 3, "status": "confirmed"}, "3": {"status": "deleted"}},
              "added": [{"id": 9, "type": "assist", "time": 4, "player": "A#18", "status": "confirmed"}]}
    merged = reviewed_events(events, review)
    assert [e["id"] for e in merged] == [1]  # 以前的篮板、抢断、助攻都不要了
    s = stats(merged)["A#18"]
    assert (s["points"], s["tpm"], s["tpa"], s["fga"]) == (3, 1, 1, 0)  # 改成 3 分的老记法算三分


def test_rerun_keeps_event_ids_and_reports_stale_clips():
    from bballgoal.events import carry_ids

    old = [{"id": 1, "type": "shot", "time": 10.0, "player": "A#1", "result": "made", "clip": [7.0, 12.0]},
           {"id": 2, "type": "rebound", "time": 20.0, "player": "B#2", "clip": [17.0, 22.0]},
           {"id": 3, "type": "shot", "time": 30.0, "player": None, "result": "missed", "clip": [27.0, 32.0]}]
    new = [{"type": "shot", "time": 10.2, "player": "A#1", "result": "made", "clip": [7.0, 12.0]},  # 同一个，片段没变
           {"type": "shot", "time": 30.0, "player": "A#5", "result": "missed", "clip": [27.0, 32.0]},  # 认出了投篮人
           {"type": "steal", "time": 40.0, "player": "A#1", "clip": [37.0, 42.0]}]  # 新事件
    stale = carry_ids(new, old)
    assert [e["id"] for e in new] == [1, 3, 4]
    assert stale == [2, 3]  # 2 消失了，3 换了人要重新渲染


def test_free_throws_counted_apart_and_team_guessed_from_hoop():
    from bballgoal.events import guess_teams

    events = [{"type": "shot", "time": t, "hoop": h, "team": team, "player": p, "result": "made"}
              for t, h, team, p in [(10, 0, "A", "A#1"), (20, 0, "A", "A#2"), (30, 1, "B", "B#3"), (40, 0, None, None)]]
    events.append({"type": "shot", "time": 50, "hoop": 0, "team": "A", "player": "A#1", "result": "made", "kind": "freethrow"})
    guess_teams(events)
    assert events[3]["team"] == "A" and events[3]["team_guess"]
    s = stats(events)["A#1"]
    assert (s["fgm"], s["fga"], s["ftm"], s["fta"], s["points"]) == (1, 1, 1, 1, 3)
    team = stats(events, key=lambda e: e.get("team"))
    assert team["A"]["points"] == 7 and team["B"]["points"] == 2


def test_clip_signature_follows_review_edits_and_ignores_empty_fields():
    from bballgoal.clips import signature, with_review

    events = [{"id": 2, "type": "shot", "time": 10.0, "player": "A#1", "result": "made", "clip": [7.0, 12.0]},
              {"id": 3, "type": "shot", "time": 30.0, "player": "B#4", "result": "missed", "clip": [27.0, 32.0]}]
    review = {"events": {"2": {"player": "A#7", "status": "confirmed", "edited": True}, "3": {"status": "deleted"}}}
    merged = {e["id"]: e for e in with_review(events, review)}
    assert set(merged) == {2} and signature(merged[2]) != signature(events[0])
    assert signature({**events[0], "kind": None}) == signature(events[0])


def test_shot_kinds_and_legacy_points():
    from bballgoal.clips import shot_kind, shot_points, shot_text

    assert [shot_kind(e) for e in ({"kind": "layup"}, {"kind": "jumper", "points": 1}, {"points": 3}, {"kind": "freethrow"})] \
        == ["two", "freethrow", "three", "freethrow"]
    assert shot_points({"kind": "three"}) == 3 and shot_text({"kind": "freethrow", "result": "made"}) == "罚篮 · 命中 · 1 分"
