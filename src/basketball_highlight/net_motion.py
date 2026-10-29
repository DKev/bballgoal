"""Frame-to-frame motion of the net below the rim, with the ball masked out.

A make pushes the net; a ball that falls in front of the rim leaves it still.
Camera shake is removed by aligning each frame pair first.
"""
import cv2
import numpy as np

MASK_BALL_CONF = 0.3
MASK_PAD = 0.2
MIN_NET_PIXELS = 50


def motion_box(rim, frame_width, frame_height):
    """Crop around the rim, 1.5 rim widths to each side, 1 above, 2 below."""
    width = float(rim["width"])
    center_x = float(rim["center_x"])
    rim_y = float(rim["rim_y"])
    return [
        max(0, int(center_x - 1.5 * width)),
        max(0, int(rim_y - 1.0 * width)),
        min(frame_width, int(center_x + 1.5 * width)),
        min(frame_height, int(rim_y + 2.0 * width)),
    ]


def prepare(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


def net_motion_series(grays, frames, rim, origin):
    """Mean change of the net region between frame k-1 and frame k.

    ``grays`` are ``prepare``d crops at ``origin`` (x, y) in the source frame;
    ``frames`` are the detections of the same frames. Item 0 is ``None``.
    """
    if not grays:
        return []
    width = float(rim["width"])
    center_x = float(rim["center_x"])
    rim_y = float(rim["rim_y"])
    x0, y0 = origin
    height, crop_width = grays[0].shape
    net = np.zeros((height, crop_width), bool)
    net[
        max(0, int(rim_y + 0.05 * width) - y0):max(0, int(rim_y + 1.0 * width) - y0),
        max(0, int(center_x - 0.55 * width) - x0):max(0, int(center_x + 0.55 * width) - x0),
    ] = True
    pad = MASK_PAD * width
    window = cv2.createHanningWindow((crop_width, height), cv2.CV_32F)
    series = [None]
    for index in range(1, len(grays)):
        mask = net.copy()
        for detections in frames[index - 1:index + 1]:
            for item in detections:
                if item["name"] != "ball" or item["confidence"] < MASK_BALL_CONF:
                    continue
                x1, y1, x2, y2 = item["xyxy"]
                mask[
                    max(0, int(y1 - pad) - y0):max(0, int(y2 + pad) - y0),
                    max(0, int(x1 - pad) - x0):max(0, int(x2 + pad) - x0),
                ] = False
        if mask.sum() <= MIN_NET_PIXELS:
            series.append(None)
            continue
        # phaseCorrelate windows its inputs in place (OpenCV 5.0).
        (dx, dy), _ = cv2.phaseCorrelate(grays[index - 1].copy(), grays[index].copy(), window)
        aligned = cv2.warpAffine(
            grays[index], np.float32([[1, 0, -dx], [0, 1, -dy]]), (crop_width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        series.append(round(float(np.abs(aligned - grays[index - 1])[mask].mean()), 3))
    return series
