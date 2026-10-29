"""场地设置：比赛区域多边形 + 两个篮筐位置，每个机位只需设置一次。

设置文件与视频放在一起：videos/比赛.mp4 -> videos/比赛.court.json

用法（会弹出窗口，在画面上点选）：
    .venv/bin/python -m bballgoal.court videos/比赛.mp4
    .venv/bin/python -m bballgoal.court videos/比赛.mp4 --time 30   # 用第 30 秒的画面

操作：
    1. 沿比赛区域边缘依次点击（含边线外一点余量，不含板凳席和观众），回车结束
    2. 点击左边篮筐的篮圈中心
    3. 点击右边篮筐的篮圈中心
    按 Backspace 撤销上一个点，Esc 放弃。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Court:
    polygon: list[tuple[float, float]]  # 比赛区域，按脚底点判断是否在场内
    hoops: list[tuple[float, float]]  # [左篮筐, 右篮筐] 篮圈中心

    @classmethod
    def load(cls, path: Path) -> "Court":
        data = json.loads(path.read_text())
        return cls(
            polygon=[tuple(p) for p in data["polygon"]],
            hoops=[tuple(p) for p in data["hoops"]],
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"polygon": self.polygon, "hoops": self.hoops}, indent=2))

    def contains(self, x: float, y: float) -> bool:
        contour = np.array(self.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def y_range(self, frame_height: int, head_room: int = 450) -> tuple[int, int]:
        """需要检测的横带：比赛区域最高点再往上留出人身高的余量。"""
        ys = [p[1] for p in self.polygon]
        return max(0, int(min(ys)) - head_room), min(frame_height, int(max(ys)))


def court_path_for(video: Path) -> Path:
    return video.with_suffix(".court.json")


def calibrate(frame: np.ndarray) -> Court | None:
    h, w = frame.shape[:2]
    scale = min(1.0, 1600 / w)
    polygon: list[tuple[float, float]] = []
    hoops: list[tuple[float, float]] = []
    stage = ["polygon"]

    def redraw() -> None:
        view = cv2.resize(frame, (int(w * scale), int(h * scale)))
        pts = [(int(x * scale), int(y * scale)) for x, y in polygon]
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts)], stage[0] != "polygon", (0, 255, 255), 2)
        for p in pts:
            cv2.circle(view, p, 5, (0, 255, 255), -1)
        for x, y in hoops:
            cv2.circle(view, (int(x * scale), int(y * scale)), 10, (0, 0, 255), 2)
        hint = {
            "polygon": "点击比赛区域边缘，回车结束  (Click court area outline, Enter to finish)",
            "hoop_left": "点击左篮筐中心  (Click LEFT hoop)",
            "hoop_right": "点击右篮筐中心  (Click RIGHT hoop)",
        }[stage[0]]
        cv2.putText(view, hint.split("  ")[1], (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(view, hint.split("  ")[1], (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("bballgoal court", view)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = (round(x / scale, 1), round(y / scale, 1))
        if stage[0] == "polygon":
            polygon.append(point)
        else:
            hoops.append(point)
            stage[0] = "hoop_right" if len(hoops) == 1 else "done"
        redraw()

    cv2.namedWindow("bballgoal court")
    cv2.setMouseCallback("bballgoal court", on_mouse)
    redraw()
    while stage[0] != "done":
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key in (13, 10) and stage[0] == "polygon" and len(polygon) >= 3:
            stage[0] = "hoop_left"
            redraw()
        if key in (8, 127):
            if hoops:
                hoops.pop()
                stage[0] = "hoop_left" if not hoops else "hoop_right"
            elif stage[0] != "polygon":
                stage[0] = "polygon"
            elif polygon:
                polygon.pop()
            redraw()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return Court(polygon=polygon, hoops=hoops)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--time", type=float, default=0, help="用第几秒的画面来设置")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"读不到视频画面：{args.video}", file=sys.stderr)
        return 1

    court = calibrate(frame)
    if court is None:
        print("已取消")
        return 1
    out = court_path_for(args.video)
    court.save(out)
    print(f"已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
