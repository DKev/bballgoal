from bballgoal.reel import Camera, clips_for


def _frames(present, fps=10):
    """present: 每帧这个人在不在。"""
    return [{"time": i / fps, "people": [{"player": "A#18", "box": [0, 0, 10, 20]}] if on else []} for i, on in enumerate(present)]


def test_clips_merge_short_gaps_and_drop_blips():
    present = [True] * 30 + [False] * 10 + [True] * 30 + [False] * 50 + [True] * 3 + [False] * 30
    clips = clips_for(_frames(present), "A#18")
    assert len(clips) == 1  # 中间断 1 秒合并；最后 0.3 秒一闪而过不要
    a, b = clips[0]
    assert a == 0.0 and abs(b - 7.4) < 1e-6  # 6.9 秒 + 0.5 秒余量


def test_camera_stays_inside_frame():
    cam = Camera(3840, 2160)
    x0, y0, w, h = cam.update([3800, 2000, 3840, 2160])
    assert x0 + w <= 3840 and y0 + h <= 2160 and x0 >= 0 and y0 >= 0
    assert abs(w / h - 16 / 9) < 0.01


def test_music_matches_length_and_fades_out():
    import numpy as np

    from bballgoal.music import SR, compose

    audio = compose(total=20.0, intro=2.5, outro=8.9, seed="A#18")
    assert audio.shape[1] == 2 and abs(len(audio) / SR - 20.5) < 0.1
    loud = np.abs(audio[int(8 * SR):int(10 * SR)]).mean()
    assert loud > 0.05 and np.abs(audio[int(19.8 * SR):]).max() < 0.02  # 正片有声音，结尾淡到没声
    assert not np.allclose(audio[:SR], compose(20.0, 2.5, 8.9, "B#5")[:SR])  # 每个孩子的音乐不一样


def test_top_plays_prefers_made_high_value_and_variety_counting_down():
    from bballgoal.reels import top_plays

    def shot(i, player, result, kind="two", team="A"):
        return {"id": i, "type": "shot", "time": float(i), "player": player, "team": team, "result": result, "kind": kind}

    events = [shot(1, "A#1", "made"), shot(2, "A#1", "made"), shot(3, "A#2", "made", "three"), shot(4, "A#3", "missed"),
              shot(5, "A#4", "made", "freethrow"), shot(6, "A#1", "unsure"), shot(7, "B#9", "made", team="B")]
    picks = [e["id"] for e in top_plays(events, "A", {})]
    assert picks[-1] == 3  # 三分进球是第一名，最后播
    assert 7 not in picks and len(picks) == 5
    assert set(picks) == {1, 2, 3, 5, 6}  # 进球都进来了，不够用拿不准的补，没进的排最后被挤掉
