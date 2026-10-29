"""专门为这个机位训练找球模型：先准备一批画面让你点球在哪，再用你的标注微调。

现成的球模型是在近景视频上训练的，全场远景里的球只有十几个像素，只能找到很少一部分。
用你在网页“点球”里点的 200 来张画面微调，模型就学会了这台相机、这个场馆、这种光线下的球长什么样。

    .venv/bin/python -m bballgoal.balllabel prep  output/<比赛>/enroll.mp4     # 准备要点的画面
    .venv/bin/python -m bballgoal.balllabel train output/<比赛>                 # 用标注训练
数据都在 output/<比赛>/balls/：imgs/（画面）、items.json（画面列表和模型猜测）、labels.json（你的标注）、model.pt。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bballgoal.court import Court
from bballgoal.detect import default_device

BASE_MODEL = Path("models/bball_model.pt")
PERSON_MODEL = Path("models/yolo11m.pt")
SAMPLES = 240
CROP_W, CROP_H = 1280, 720  # 原始分辨率裁一块：球员最集中的地方
HOOP_SHARE = 3  # 每 3 张里有 1 张以篮筐为中心（人在那个半场时）：投篮、球在篮圈附近的样子也要学
BALL_TO_PERSON = 0.17  # 篮球直径约为小孩身高的这个比例（用来从“点”生成框）
EPOCHS = 60


def video_times(video: Path) -> list[float]:
    """可取样的时间：抽段视频跳过段与段之间的黑屏。"""
    plan_file = video.with_suffix(".enroll.json")
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if plan_file.exists():
        segs = json.loads(plan_file.read_text())["segments"]
        return [t for s in segs for t in np.arange(s["start"] + 0.5, s["end"] - 0.5, 0.2)]
    return list(np.arange(1.0, duration - 1.0, 0.2))


def prep(video: Path, out: Path, n: int = SAMPLES) -> None:
    court_file = video.with_suffix(".court.json")
    court = Court.load(court_file) if court_file.exists() else None
    imgs = out / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)
    person = YOLO(str(PERSON_MODEL))
    ball = YOLO(str(BASE_MODEL))
    ball_class = [k for k, v in ball.names.items() if v.lower() == "ball"]
    random.seed(0)
    times = sorted(random.sample(video_times(video), min(n * 2, len(video_times(video)))))
    cap = cv2.VideoCapture(str(video))
    items = []
    for t in times:
        if len(items) >= n:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        r = person.predict(frame, imgsz=1920, conf=0.35, classes=[0], device=default_device(), verbose=False)[0]
        boxes = [b for b in r.boxes.xyxy.cpu().numpy() if court is None or court.contains((b[0] + b[2]) / 2, b[3])]
        if len(boxes) < 4:  # 场上人太少：多半是暂停、死球
            continue
        b = np.array(boxes)
        cx, cy = float(np.median((b[:, 0] + b[:, 2]) / 2)), float(np.median((b[:, 1] + b[:, 3]) / 2))
        hoop = None
        if court is not None and court.hoops and len(items) % HOOP_SHARE == HOOP_SHARE - 1:
            near = min(court.hoops, key=lambda h: abs(h[0] - cx))
            if abs(near[0] - cx) < CROP_W:  # 人在这个篮筐的半场
                hoop = near
        if hoop is not None:  # 篮筐放在画面上方三分之一，下面是篮下的人
            x0 = int(np.clip(hoop[0] - CROP_W / 2, 0, max(0, W - CROP_W)))
            y0 = int(np.clip(hoop[1] - CROP_H * 0.3, 0, max(0, H - CROP_H)))
        else:
            x0 = int(np.clip(cx - CROP_W / 2, 0, max(0, W - CROP_W)))
            y0 = int(np.clip(cy - CROP_H * 0.55, 0, max(0, H - CROP_H)))
        crop = frame[y0 : y0 + CROP_H, x0 : x0 + CROP_W]
        guesses = []
        res = ball.predict(crop, imgsz=1280, conf=0.15, classes=ball_class, device=default_device(), verbose=False)[0]
        for (a, c, d, e), conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            guesses.append({"x": round(float(a + d) / 2, 1), "y": round(float(c + e) / 2, 1), "conf": round(float(conf), 2)})
        heights = b[:, 3] - b[:, 1]
        name = f"{len(items):04d}.jpg"
        cv2.imwrite(str(imgs / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({"file": f"imgs/{name}", "time": round(t, 2), "origin": [x0, y0], "w": crop.shape[1], "h": crop.shape[0], "hoop": hoop is not None,
                      "person_height": round(float(np.median(heights)), 1), "guesses": sorted(guesses, key=lambda g: -g["conf"])[:3]})
        print(f"  {len(items)}/{n} 段", flush=True)
    cap.release()
    (out / "items.json").write_text(json.dumps({"video": str(video), "items": items}, ensure_ascii=False))


def other_games(out: Path) -> list[Path]:
    """同一个 output 目录下其他比赛的 balls/ 文件夹（标过球的）。"""
    return sorted(d for d in out.parent.parent.glob("*/balls") if d.resolve() != out.resolve() and (d / "labels.json").exists()
                  and (d / "items.json").exists())


def _labeled(out: Path) -> list[tuple[Path, str, dict, dict]]:
    items = {it["file"]: it for it in json.loads((out / "items.json").read_text())["items"]}
    labels = json.loads((out / "labels.json").read_text())["labels"]
    return [(out, k, items[k], labels[k]) for k in sorted(labels) if k in items and (out / k).exists()]


def build_dataset(out: Path, val_share: float = 0.2) -> tuple[Path, int, int]:
    """把标注变成 YOLO 训练集：点 → 以人身高估算大小的方框；没有球的画面当作负样本。
    以前比赛（同一个 output 目录下）标过的也一起放进训练集：新比赛只要再标几十张。验证集只用这一场的，门槛按这一场选。"""
    ds = out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    mine = _labeled(out)
    random.seed(1)
    random.shuffle(mine)
    n_val = max(1, int(len(mine) * val_share))
    older = [row for d in other_games(out) for row in _labeled(d)]
    if older:
        print(f"加上以前比赛标过的 {len(older)} 张一起训练", flush=True)
    with_ball = 0
    for split, chunk in (("val", mine[:n_val]), ("train", mine[n_val:] + older)):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for src, k, it, lab in chunk:
            name = Path(k).name if src == out else f"{src.parent.name}_{Path(k).name}"
            shutil.copy(src / k, ds / "images" / split / name)
            lines = []
            if lab.get("ball"):
                x, y = lab["ball"]
                size = max(8.0, BALL_TO_PERSON * it["person_height"])
                lines.append(f"0 {x / it['w']:.6f} {y / it['h']:.6f} {size / it['w']:.6f} {size / it['h']:.6f}")
                with_ball += 1
            (ds / "labels" / split / name).with_suffix(".txt").write_text("\n".join(lines))
    (ds / "data.yaml").write_text(f"path: {ds.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    return ds / "data.yaml", len(mine) + len(older), with_ball


def inherit(out: Path) -> None:
    """新比赛还没有自己的找球模型：沿用最近一场训练好的（同一个场馆、相机位置差不多时直接就能用）。"""
    if (out / "model.pt").exists():
        print("这场已经有自己的找球模型")
        return
    trained = [d for d in out.parent.parent.glob("*/balls") if d.resolve() != out.resolve() and (d / "model.pt").exists()]
    if not trained:
        print("以前没有训练过找球模型，先用原项目的")
        return
    src = max(trained, key=lambda d: (d / "model.pt").stat().st_mtime)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "model.pt", out / "model.pt")
    info = json.loads((src / "train.json").read_text()) if (src / "train.json").exists() else {}
    (out / "train.json").write_text(json.dumps({**info, "inherited_from": src.parent.name}))
    print(f"沿用 {src.parent.name} 的找球模型")


def _hits(model_path: Path, out: Path, keys: list[str]) -> list[tuple[bool, list[tuple[float, bool]]]]:
    """每张验证图：有没有球，以及每个检测（置信度, 是不是落在你点的球上）。"""
    items = {it["file"]: it for it in json.loads((out / "items.json").read_text())["items"]}
    labels = json.loads((out / "labels.json").read_text())["labels"]
    model = YOLO(str(model_path))
    cls = [k for k, v in model.names.items() if v.lower() == "ball"]
    rows = []
    for k in keys:
        lab = labels[k]
        res = model.predict(str(out / k), imgsz=1280, conf=0.01, classes=cls, device=default_device(), verbose=False)[0]
        size = max(10, BALL_TO_PERSON * items[k]["person_height"])
        dets = []
        for (a, b, c, d), conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            on_ball = bool(lab.get("ball")) and np.hypot((a + c) / 2 - lab["ball"][0], (b + d) / 2 - lab["ball"][1]) <= size
            dets.append((float(conf), on_ball))
        rows.append((bool(lab.get("ball")), dets))
    return rows


def recall_at(rows, conf: float) -> tuple[float, float]:
    """(有球的画面里找到球的比例, 每张图的误报数)。"""
    with_ball = [d for has, d in rows if has]
    found = sum(any(c >= conf and ok for c, ok in d) for d in with_ball)
    false = sum(c >= conf and not ok for _has, d in rows for c, ok in d)
    return found / max(1, len(with_ball)), false / max(1, len(rows))


def pick_conf(rows, max_false: float) -> float:
    """误报不超过 max_false（每张图）的前提下，找到球最多的门槛。训练过的模型置信度普遍偏低，固定门槛会把它的检测挡掉。"""
    best = (-1.0, 0.5)
    for conf in np.arange(0.02, 0.51, 0.01):
        recall, false = recall_at(rows, conf)
        if false <= max_false and recall > best[0]:
            best = (recall, round(float(conf), 2))
    return best[1]


def calibrate(model_path: Path, out: Path, val_keys: list[str]) -> dict:
    from bballgoal.ball import MIN_CONF

    old_rows, new_rows = _hits(BASE_MODEL, out, val_keys), _hits(model_path, out, val_keys)
    conf, strict = pick_conf(new_rows, 0.1), pick_conf(new_rows, 0.02)
    return {"conf": conf, "conf_strict": strict,
            "recall_old": recall_at(old_rows, MIN_CONF)[0], "false_old": recall_at(old_rows, MIN_CONF)[1],
            "recall_new": recall_at(new_rows, conf)[0], "false_new": recall_at(new_rows, conf)[1]}


def train(out: Path, epochs: int = EPOCHS) -> None:
    data, n, with_ball = build_dataset(out)
    print(f"训练集：{n} 张（{with_ball} 张有球）", flush=True)
    model = YOLO(str(BASE_MODEL))
    model.add_callback("on_train_epoch_end", lambda t: print(f"\n  训练 第 {t.epoch + 1}/{t.epochs} 轮", flush=True))
    model.train(data=str(data), epochs=epochs, imgsz=1280, batch=4, device=default_device(), project=str((out / "runs").resolve()),
                name="ball", exist_ok=True, verbose=False, plots=False, patience=20, fliplr=0.5, mosaic=0.5, degrees=0)
    best = out / "runs" / "ball" / "weights" / "best.pt"
    val_keys = [f"imgs/{p.name}" for p in (out / "dataset" / "images" / "val").glob("*.jpg")]
    cal = calibrate(best, out, val_keys)
    shutil.copy(best, out / "model.pt")
    (out / "train.json").write_text(json.dumps({"images": n, "with_ball": with_ball, "images_here": len(_labeled(out)), **cal}))
    print(f"验证集上找到球的比例：原模型 {cal['recall_old']:.0%}（每张误报 {cal['false_old']:.2f}）"
          f" → 新模型 {cal['recall_new']:.0%}（每张误报 {cal['false_new']:.2f}，门槛 {cal['conf']}）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("video", type=Path)
    p.add_argument("--n", type=int, default=SAMPLES)
    c = sub.add_parser("calibrate", help="只给已训练的模型重新选门槛")
    c.add_argument("game_dir", type=Path)
    i = sub.add_parser("inherit", help="沿用最近一场训练好的找球模型")
    i.add_argument("game_dir", type=Path)
    t = sub.add_parser("train")
    t.add_argument("game_dir", type=Path)
    t.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.cmd == "prep":
        prep(args.video, args.video.parent / "balls", args.n)
    elif args.cmd == "inherit":
        inherit(args.game_dir / "balls")
    elif args.cmd == "calibrate":
        out = args.game_dir / "balls"
        val_keys = [f"imgs/{p.name}" for p in (out / "dataset" / "images" / "val").glob("*.jpg")]
        info = json.loads((out / "train.json").read_text())
        info.update(calibrate(out / "model.pt", out, val_keys))
        (out / "train.json").write_text(json.dumps(info))
        print(info)
    else:
        if not (args.game_dir / "balls" / "labels.json").exists():
            print("还没有标注：先在网页“点球”里点一些画面", file=sys.stderr)
            return 1
        train(args.game_dir / "balls", args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
