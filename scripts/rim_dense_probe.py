"""Dense rim-crop ball detection around labeled candidates.

For every candidate, decode every source frame in [event - before, event + after]
and run the detector on two inputs:

* ``wide``  - the analysis ROI upscaled like the current refine stage;
* ``tight`` - a square of ``crop_rims`` rim widths around the rim.

Detections are stored in source-video coordinates so later steps can rebuild
tracks from either input.
"""
import argparse
import json
from pathlib import Path

import cv2


def tight_box(rim, crop_rims, min_side, width, height):
    side = max(min_side, crop_rims * float(rim["width"]))
    cx = float(rim["center_x"])
    cy = float(rim["rim_y"]) + 0.75 * float(rim["width"])
    x1 = int(round(min(max(0, cx - side / 2), width - side)))
    y1 = int(round(min(max(0, cy - side / 2), height - side)))
    return [x1, y1, x1 + int(side), y1 + int(side)]


def detect(model, crops, device):
    results = model.predict(
        [crop for crop, _, _ in crops], imgsz=640, conf=0.05, iou=0.7,
        rect=False, device=device, verbose=False,
    )
    output = []
    for (_, offset, scale), result in zip(crops, results):
        items = []
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            if name not in {"ball", "hoop"}:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            items.append({
                "name": name,
                "conf": round(float(box.conf[0]), 3),
                "xyxy": [
                    round(offset[0] + bx1 / scale, 1), round(offset[1] + by1 / scale, 1),
                    round(offset[0] + bx2 / scale, 1), round(offset[1] + by2 / scale, 1),
                ],
            })
        output.append(items)
    return output


def probe_case(model, case, args, device):
    cap = cv2.VideoCapture(case["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    event = case["event_ms"] / 1000.0
    first = max(0, int(round((event - args.before) * fps)))
    last = int(round((event + args.after) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rim = case["rim"]
    tight = tight_box(rim, args.crop_rims, args.min_side, width, height)
    wx1, wy1, wx2, wy2 = [int(round(value)) for value in case["analysis_roi"]]
    frames = []
    for index in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        wide_crop = cv2.resize(frame[wy1:wy2, wx1:wx2], None, fx=args.wide_scale,
                               fy=args.wide_scale, interpolation=cv2.INTER_CUBIC)
        tight_crop = frame[tight[1]:tight[3], tight[0]:tight[2]]
        frames.append({
            "frame": index,
            "time": round(index / fps, 4),
            "_wide": (wide_crop, (wx1, wy1), args.wide_scale),
            "_tight": (tight_crop, (tight[0], tight[1]), 1.0),
        })
    cap.release()
    for key in args.inputs:
        for start in range(0, len(frames), args.batch):
            chunk = frames[start:start + args.batch]
            for frame, items in zip(chunk, detect(model, [f["_" + key] for f in chunk], device)):
                frame[key] = items
    for frame in frames:
        frame.pop("_wide")
        frame.pop("_tight")
    return {**case, "source_fps": fps, "tight_box": tight, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", default="models/bball_model.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--before", type=float, default=1.0)
    parser.add_argument("--after", type=float, default=1.0)
    parser.add_argument("--crop-rims", type=float, default=5.0)
    parser.add_argument("--min-side", type=float, default=96.0)
    parser.add_argument("--wide-scale", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--inputs", nargs="+", default=["wide", "tight"],
                        choices=["wide", "tight"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    cases = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    results = []
    for number, case in enumerate(cases, 1):
        results.append(probe_case(model, case, args, args.device))
        print(f"[{number}/{len(cases)}] {case['project'][:6]} {case['event_ms'] / 1000:.2f}s",
              flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
