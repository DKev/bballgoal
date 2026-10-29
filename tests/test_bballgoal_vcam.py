import numpy as np

from bballgoal.vcam import analysis_to_dir, dir_to_lonlat, plan_shot, project_to_view, rotation, view_rays

VIEW = {"x": 1360, "y": 1280, "w": 3040, "h": 1158}


def test_view_center_looks_where_rotated():
    yaw, pitch = np.radians(20), np.radians(-15)
    rays = view_rays(np.radians(60), 64, 36)
    center = (rays[17, 31] + rays[18, 32]) / 2 @ rotation(yaw, pitch).T
    lon, lat = dir_to_lonlat(center / np.linalg.norm(center))
    assert abs(lon - yaw) < 0.02 and abs(lat - pitch) < 0.02


def test_project_round_trip():
    d = analysis_to_dir(np.array([1500.0]), np.array([600.0]), VIEW)[0]
    lon, lat = dir_to_lonlat(d)
    u, v = project_to_view(d, float(lon), float(lat), np.radians(50))
    assert abs(u - 960) < 1e-6 and abs(v - 540) < 1e-6
    # 右边的点在画面右边，上面的点在画面上面
    right = analysis_to_dir(np.array([1600.0]), np.array([600.0]), VIEW)[0]
    up = analysis_to_dir(np.array([1500.0]), np.array([500.0]), VIEW)[0]
    assert project_to_view(right, float(lon), float(lat), np.radians(50))[0] > 960
    assert project_to_view(up, float(lon), float(lat), np.radians(50))[1] < 540


def test_plan_shot_frames_targets_and_is_smooth():
    times = np.arange(0, 10, 0.1)
    targets = [np.array([[500 + 20 * t, 500, 540 + 20 * t, 650]]) if t < 8 else None for t in times]
    shot = plan_shot(times, targets, VIEW, smooth=0.5)
    assert np.all(np.diff(shot.yaw[5:65]) > 0)  # 目标一直向右走，镜头也一直向右转（提前量让末尾提早变平）
    assert np.all(np.degrees(shot.hfov) >= 30) and np.all(np.degrees(shot.hfov) <= 80)
    assert not np.isnan(shot.yaw).any()  # 没目标的时刻沿用前面的
