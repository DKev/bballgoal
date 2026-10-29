from bballgoal.enroll import plan


def test_plan_spreads_over_all_parts_without_crossing_files():
    segs = plan([1800.0, 1000.0], 8, 40.0)
    assert len(segs) == 8
    assert {part for part, _l, _g in segs} == {0, 1}  # 上下两个文件都有
    games = [g for _p, _l, g in segs]
    assert games == sorted(games) and games[0] > 0
    durations = [1800.0, 1000.0]
    for part, local, _g in segs:
        assert 0 <= local and local + 40 <= durations[part]
