"""外观特征：区分同队的不同人（同队球衣一样，光看球衣颜色分不开）。

两种特征，在同一场比赛里实测区分同队不同人效果最好的组合：
- 整体外观：OSNet 行人重识别模型（BoxMOT 提供，在 MSMT17 行人数据集上训练），
  看体型、发型、肤色、衣服细节；
- 鞋：先用姿态模型找到脚踝，在 ResNet50 特征图上取脚踝附近一块。小孩的鞋颜色款式
  各不相同，而且不随球衣变化，是最好认的部位之一。

头部在这个分辨率下只有十几个像素，单独取特征反而降低准确度，交给整体外观去覆盖。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from bballgoal.detect import default_device

POSE_MODEL = Path("models/yolo11m-pose.pt")
REID_MODEL = Path("models/osnet_ain_x1_0_msmt17.pt")
KP_CONF = 0.5
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
L_ANKLE, R_ANKLE = 15, 16
CROP_H, CROP_W, STRIDE = 256, 128, 16  # 人像统一缩放到 256×128，ResNet layer3 输出 16×8
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_reid(device: str | None = None):
    from boxmot.reid.core.runtime import ReID

    return ReID(path=REID_MODEL, device=device or default_device())


class PoseEstimator:
    """对每个人的框单独做姿态估计（自上而下），比整帧做快，小个子的关键点也更准。"""

    def __init__(self, device: str | None = None, imgsz: int = 320):
        self.model = YOLO(str(POSE_MODEL))
        self.device = device or default_device()
        self.imgsz = imgsz

    def keypoints(self, frame: np.ndarray, boxes: list) -> list[np.ndarray | None]:
        """返回每个框的 17×3 关键点（原图坐标），找不到的为 None。"""
        crops, origins = [], []
        for x1, y1, x2, y2 in boxes:
            mx, my = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            ox, oy = int(max(0, x1 - mx)), int(max(0, y1 - my))
            crops.append(frame[oy : int(y2 + my), ox : int(x2 + mx)])
            origins.append((ox, oy, x1, y1, x2, y2))
        if not crops:
            return []
        results = self.model.predict(crops, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)
        out: list[np.ndarray | None] = []
        for r, (ox, oy, x1, y1, x2, y2) in zip(results, origins):
            if r.keypoints is None or len(r.boxes) == 0:
                out.append(None)
                continue
            # 框里可能有别人，取和目标框重合最多的那个
            b = r.boxes.xyxy.cpu().numpy() + [ox, oy, ox, oy]
            ix = np.clip(np.minimum(b[:, 2], x2) - np.maximum(b[:, 0], x1), 0, None)
            iy = np.clip(np.minimum(b[:, 3], y2) - np.maximum(b[:, 1], y1), 0, None)
            inter = ix * iy
            union = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (x2 - x1) * (y2 - y1) - inter
            best = int(np.argmax(inter / union))
            if inter[best] / union[best] < 0.5:
                out.append(None)
                continue
            kp = r.keypoints.data[best].cpu().numpy().copy()
            kp[:, 0] += ox
            kp[:, 1] += oy
            out.append(kp)
        return out


def body_crop(frame: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """以人框为中心取 2:1 的竖条，缩放到 CROP_H×CROP_W。返回图块和原图→图块的仿射矩阵。"""
    x1, y1, x2, y2 = box
    cw = max(x2 - x1, (y2 - y1) / 2) * 1.1
    ch = cw * 2
    ox, oy = (x1 + x2) / 2 - cw / 2, (y1 + y2) / 2 - ch / 2
    m = np.float32([[CROP_W / cw, 0, -ox * CROP_W / cw], [0, CROP_H / ch, -oy * CROP_H / ch]])
    crop = cv2.warpAffine(frame, m, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return crop, m


def shoe_cells(kp: np.ndarray, m: np.ndarray) -> tuple[int, int, int, int] | None:
    """两只脚踝在特征图上的范围（行起止、列起止），往下多取一格包含鞋底。看不到脚踝时为 None。"""
    ankles = kp[[L_ANKLE, R_ANKLE]]
    ankles = ankles[ankles[:, 2] >= KP_CONF]
    if len(ankles) == 0:
        return None
    pts = ankles[:, :2] @ m[:, :2].T + m[:, 2]
    cols = (pts[:, 0] / STRIDE).astype(int).clip(0, CROP_W // STRIDE - 1)
    rows = (pts[:, 1] / STRIDE).astype(int).clip(0, CROP_H // STRIDE - 1)
    return int(rows.min()), int(min(CROP_H // STRIDE - 1, rows.max() + 1)), int(cols.min()), int(cols.max())


def torso_box(kp: np.ndarray) -> list[float] | None:
    """肩到髋的范围（号码所在位置），左右各留一点边。四个点都看得清才返回。"""
    pts = kp[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    if (pts[:, 2] < KP_CONF).any():
        return None
    x1, x2 = pts[:, 0].min(), pts[:, 0].max()
    y1, y2 = pts[:2, 1].mean(), pts[2:, 1].mean()
    if y2 - y1 < 10:
        return None
    pad = max(0.15 * (x2 - x1), 0.1 * (y2 - y1))
    return [round(float(v), 1) for v in (x1 - pad, y1, x2 + pad, y2)]


class AppearanceExtractor:
    """一次处理一帧里的多个人：返回每人的整体外观向量、鞋向量和躯干框（后两者可能为 None）。"""

    def __init__(self, reid=None, device: str | None = None):
        self.device = device or default_device()
        self.reid = reid or load_reid(self.device)
        self.pose = PoseEstimator(self.device)
        net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = torch.nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3
        ).to(self.device).eval()

    def extract(self, frame: np.ndarray, boxes: list) -> list[tuple[np.ndarray, np.ndarray | None, list | None]]:
        if not boxes:
            return []
        body = self.reid(frame, boxes=np.array(boxes, np.float32))
        crops, mats = zip(*(body_crop(frame, b) for b in boxes))
        batch = np.stack([(c[:, :, ::-1].astype(np.float32) / 255 - _MEAN) / _STD for c in crops])
        with torch.no_grad():
            maps = self.backbone(torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device))
        maps = maps.permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for i, kp in enumerate(self.pose.keypoints(frame, boxes)):
            shoes = None
            cells = shoe_cells(kp, mats[i]) if kp is not None else None
            if cells is not None:
                r0, r1, c0, c1 = cells
                v = maps[i, r0 : r1 + 1, c0 : c1 + 1].reshape(-1, maps.shape[-1]).mean(axis=0)
                shoes = v / max(1e-6, float(np.linalg.norm(v)))
            out.append((body[i], shoes, torso_box(kp) if kp is not None else None))
        return out
