import cv2
import numpy as np

from basketball_highlight.net_motion import motion_box, net_motion_series, prepare

# Motion box of this rim is the whole 300x300 frame.
RIM = {"center_x": 150.0, "rim_y": 100.0, "width": 100.0, "height": 30.0}


def scene(seed=0):
    noise = np.random.default_rng(seed).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (7, 7), 0)


def shot(world, dx=0, dy=0):
    """The camera view of ``world``, panned by (dx, dy) pixels."""
    return world[50 + dy:350 + dy, 50 + dx:350 + dx].copy()


def series(frames, detections=None):
    detections = detections or [[] for _ in frames]
    return net_motion_series([prepare(frame) for frame in frames], detections, RIM, (0, 0))


def test_motion_box_stays_inside_the_frame():
    assert motion_box(RIM, 300, 300) == [0, 0, 300, 300]
    assert motion_box(RIM, 250, 250) == [0, 0, 250, 250]


def test_camera_shake_is_not_net_motion():
    world = scene()
    values = series([shot(world), shot(world, 3, 0), shot(world, 3, 2), shot(world)])
    assert values[0] is None
    assert max(values[1:]) < 1.0


def test_a_moving_net_is_net_motion():
    world, other = scene(), scene(1)
    swung = shot(world)
    swung[110:195, 100:200] = shot(other)[110:195, 100:200]
    values = series([shot(world), swung])
    assert values[1] > 5


def test_the_ball_itself_is_masked_out():
    world = scene()
    frames = []
    detections = []
    for y in (120, 150):
        frame = shot(world)
        cv2.circle(frame, (150, y), 20, (40, 90, 200), -1)
        frames.append(frame)
        detections.append([{"name": "ball", "confidence": 0.9, "xyxy": [130, y - 20, 170, y + 20]}])
    assert series(frames, detections)[1] < 1.0
