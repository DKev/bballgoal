"""全场画面的分块检测。

全场视频里篮球只有十几个像素，整帧缩到 640 再检测会丢掉。这里把原始分辨率的画面
切成有重叠的方块，每块单独检测，再把框映射回整帧坐标并做 NMS 合并。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BALL_HOOP_MODEL = ROOT / "models" / "bball_model.pt"
PERSON_MODEL = ROOT / "models" / "yolo11m.pt"


@dataclass
class Detection:
    label: str  # "ball" | "hoop" | "person"
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """沿一条边给出各块的起点，最后一块贴齐边缘。"""
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return origins


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TiledDetector:
    def __init__(
        self,
        tile: int = 1280,
        overlap: float = 0.2,
        ball_conf: float = 0.25,
        hoop_conf: float = 0.35,
        person_conf: float = 0.35,
        device: str | None = None,
    ) -> None:
        self.tile = tile
        self.overlap = overlap
        self.device = device or default_device()
        self.ball_hoop = YOLO(str(BALL_HOOP_MODEL))
        self.person = YOLO(str(PERSON_MODEL) if PERSON_MODEL.exists() else "yolo11m.pt")
        self.thresholds = {"ball": ball_conf, "hoop": hoop_conf, "person": person_conf}

    def _run(self, model: YOLO, tiles: list[np.ndarray], classes: list[int] | None) -> list:
        return model.predict(
            tiles,
            imgsz=self.tile,
            conf=min(self.thresholds.values()),
            classes=classes,
            device=self.device,
            verbose=False,
        )

    def detect_people_scaled(
        self, frame: np.ndarray, y_range: tuple[int, int], scale: float = 0.5
    ) -> list[Detection]:
        """把横带缩小后整条检测人。人比篮球大得多，半分辨率就够，速度快 2–3 倍。"""
        top, bottom = y_range
        band = frame[top:bottom]
        small = cv2.resize(band, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = self.person.predict(
            small,
            imgsz=int(np.ceil(small.shape[1] / 32) * 32),
            conf=self.thresholds["person"],
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        for (x1, y1, x2, y2), conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            detections.append(
                Detection("person", float(conf), x1 / scale, y1 / scale + top, x2 / scale, y2 / scale + top)
            )
        return detections

    def detect(
        self,
        frame: np.ndarray,
        y_range: tuple[int, int] | None = None,
        people_only: bool = False,
    ) -> list[Detection]:
        """检测整帧，或只检测 y_range=(top, bottom) 这条横带（跳过天花板等无关区域）。"""
        h, w = frame.shape[:2]
        top, bottom = y_range or (0, h)
        top, bottom = max(0, top), min(h, bottom)
        boxes: dict[str, list[tuple[float, float, float, float, float]]] = {
            "ball": [],
            "hoop": [],
            "person": [],
        }
        origins = [
            (x, top + y)
            for y in tile_origins(bottom - top, self.tile, self.overlap)
            for x in tile_origins(w, self.tile, self.overlap)
        ]
        tiles = [frame[y : min(y + self.tile, bottom), x : x + self.tile] for x, y in origins]

        passes = [(self.person, [0], self.person.names)]
        if not people_only:
            passes.insert(0, (self.ball_hoop, None, self.ball_hoop.names))
        for model, classes, names in passes:
            for (ox, oy), result in zip(origins, self._run(model, tiles, classes)):
                for xyxy, conf, cls in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    label = names[cls]
                    if label not in boxes or conf < self.thresholds[label]:
                        continue
                    x1, y1, x2, y2 = xyxy
                    boxes[label].append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(conf)))

        detections: list[Detection] = []
        for label, items in boxes.items():
            for x1, y1, x2, y2, conf in merge_tile_boxes(items):
                detections.append(Detection(label, conf, x1, y1, x2, y2))
        return detections


def merge_tile_boxes(
    items: list[tuple[float, float, float, float, float]], threshold: float = 0.6
) -> list[tuple[float, float, float, float, float]]:
    """合并相邻块的重复框。

    在块边缘被切掉一半的人，会和相邻块里完整的框重叠，但 IoU 不高。这里按
    交集 / 较小框面积（IoS）判重：被大框基本包住的小框视为同一目标，保留并集。
    """
    if not items:
        return []
    arr = torch.tensor(items, dtype=torch.float32)
    keep = torchvision.ops.nms(arr[:, :4], arr[:, 4], 0.5)
    candidates = [list(row) for row in arr[keep].tolist()]  # 按置信度降序

    merged: list[list[float]] = []
    for box in candidates:
        for kept in merged:
            ix = max(0.0, min(box[2], kept[2]) - max(box[0], kept[0]))
            iy = max(0.0, min(box[3], kept[3]) - max(box[1], kept[1]))
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_kept = (kept[2] - kept[0]) * (kept[3] - kept[1])
            if ix * iy / max(1e-6, min(area_box, area_kept)) > threshold:
                kept[0], kept[1] = min(box[0], kept[0]), min(box[1], kept[1])
                kept[2], kept[3] = max(box[2], kept[2]), max(box[3], kept[3])
                break
        else:
            merged.append(box)
    return [tuple(b) for b in merged]
