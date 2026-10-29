from bballgoal.detect import merge_tile_boxes, tile_origins


def test_tile_origins_cover_full_length():
    assert tile_origins(1000, 1280, 0.2) == [0]
    origins = tile_origins(3840, 1280, 0.2)
    assert origins[0] == 0
    assert origins[-1] == 3840 - 1280
    assert all(b - a <= 1280 for a, b in zip(origins, origins[1:]))


def test_merge_tile_boxes_joins_box_cut_at_tile_edge():
    full = (100.0, 100.0, 200.0, 400.0, 0.9)
    upper_half = (100.0, 100.0, 200.0, 250.0, 0.6)  # 相邻块里被切掉下半身
    merged = merge_tile_boxes([full, upper_half])
    assert len(merged) == 1
    assert merged[0][:4] == (100.0, 100.0, 200.0, 400.0)


def test_merge_tile_boxes_keeps_separate_people():
    a = (100.0, 100.0, 200.0, 400.0, 0.9)
    b = (180.0, 100.0, 280.0, 400.0, 0.8)  # 并排站着，只有一点重叠
    assert len(merge_tile_boxes([a, b])) == 2
