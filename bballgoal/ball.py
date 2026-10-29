"""找球、判断球在谁手里。

1. 每帧把场地横带切成几块找球（比赛文件夹里有点球训练出的 balls/model.pt 就用它，否则用 BALL_MODEL），
   按原视频帧率逐帧找，所有候选都存下来（ball_raw.json），后面的过滤可以反复调，不用重跑检测；
2. 去掉误检：总在同一个位置出现的（场边静止的东西）、太靠场外的、裁判头上的（光头像球）；
3. 在所有帧的候选里挑一条最连贯的球轨迹（动态规划：置信度高、前后帧距离近）；
4. 每帧球离谁的手最近（相对人身高），就算谁持球；很短的持球段去掉，短暂丢失补上。

用法（先跑完 track 和 identify）：
    .venv/bin/python -m bballgoal.ball videos/比赛.mp4
输出 output/track_<视频名>/ball.json：每帧球的位置和持球人（固定编号，如 A#18）。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.court import Court
from bballgoal.detect import default_device, tile_origins
from bballgoal.frames import read_frames

# 原项目带的 球/篮筐 模型。basketball_analysis 的 ball_detector_model.pt（NBA 转播画面训练）
# 在全场远景上实测几乎找不到球，不用
BALL_MODEL = Path("models/bball_model.pt")
MIN_CONF = 0.15  # 候选都留下，交给后面的轨迹挑选（原项目模型；训练过的模型用验证集选的门槛）
STRICT_CONF = 0.25  # 几乎不会是误报的门槛
TILE_W = 1100
HEAD_ROOM = 900  # 场地最高点往上留多少像素（投篮的球会飞得很高）
STATIC_CELL = 24  # 像素：判断“总在同一位置”的格子大小
STATIC_SHARE = 0.05  # 同一格子出现在超过这个比例的帧里，算静止误检
DETECT_FPS = 25.0  # 找球用原视频帧率：球快，帧多一倍半，看到球的次数也多一倍半
MAX_SPEED = 6.0  # 每秒球最多移动多少倍“附近人身高”（快传也够用）
MISS_COST = 0.3  # 某帧不选任何候选的代价（相当于一个置信度 0.3 的候选）；原项目模型的值，训练过的模型按门槛等比换算
NEAR_PLAYER = 1.5  # 球必须在某个场上的人（人身高的这么多倍）范围内，否则是隔壁场地或网后的
NEAR_HOOP = 2.0  # 或者在篮筐附近（人身高的这么多倍）：投篮的球飞到篮圈时离所有人都远
HEAD_SHARE = 0.16  # 人框顶部这么多是头：落在这里的一律当作头（光头裁判的头置信度能到 0.8）
HOLD_DISTANCE = 0.45  # 球到手部区域的距离 / 人身高，小于这个算拿着球
MIN_HOLD = 0.2  # 秒：持球段至少这么长
FILL_GAP = 0.5  # 秒：同一人持球中间丢失不超过这么久，补上


def ball_model_for(video: Path) -> Path:
    """比赛文件夹里有用你的标注训练过的模型（balls/model.pt）就用它，否则用原项目的。"""
    trained = video.parent / "balls" / "model.pt"
    return trained if trained.exists() else BALL_MODEL


def ball_conf_for(video: Path, strict: bool = False) -> float:
    """检测门槛。训练过的模型给的置信度普遍偏低，用训练时在验证集上选出的门槛（balls/train.json）。
    strict：要求几乎没有误报时的门槛（判断投篮用）。"""
    model = ball_model_for(video)
    info = model.parent / "train.json"
    if model != BALL_MODEL and info.exists():
        cal = json.loads(info.read_text())
        if "conf" in cal:
            return cal["conf_strict" if strict else "conf"]
    return STRICT_CONF if strict else MIN_CONF


def model_id(path: Path) -> str:
    """模型的身份（路径 + 修改时间 + 门槛）：检测结果里记下来，模型或门槛换了就重新检测。"""
    info = path.parent / "train.json"
    conf = json.loads(info.read_text()).get("conf") if path != BALL_MODEL and info.exists() else None
    return f"{path}:{int(path.stat().st_mtime)}" + (f":{conf}" if conf is not None else "")


def detect_candidates(video: Path, info: dict, court: Court, start: float, end: float, fps: float) -> list[list]:
    """每帧的候选：[x, y, 半径, 置信度]。"""
    model = YOLO(str(ball_model_for(video)))
    ball_class = [k for k, name in model.names.items() if name.lower() == "ball"]
    device = default_device()
    y0, y1 = court.y_range(info["height"], HEAD_ROOM)
    xs = tile_origins(info["width"], TILE_W, 0.1)
    conf = ball_conf_for(video)
    out = []
    total = int((end - start) * fps)
    started = time.perf_counter()
    for i, (_t, frame) in enumerate(read_frames(video, info["width"], info["height"], start, end, fps)):
        tiles = [frame[y0:y1, x : x + TILE_W] for x in xs]
        results = model.predict(tiles, imgsz=1280, conf=conf, classes=ball_class, device=device, verbose=False)
        found = []
        for x, r in zip(xs, results):
            for (bx1, by1, bx2, by2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                found.append([round(float(bx1 + bx2) / 2 + x, 1), round(float(by1 + by2) / 2 + y0, 1),
                              round(float(max(bx2 - bx1, by2 - by1)) / 2, 1), round(float(c), 3)])
        out.append(_dedupe(found))
        if i and i % 200 == 0:
            rate = i / (time.perf_counter() - started)
            print(f"  {i}/{total} 帧  {rate:.1f} 帧/秒  剩余约 {max(0, total - i) / rate / 60:.1f} 分钟", flush=True)
    return out


def _dedupe(found: list[list]) -> list[list]:
    """块与块重叠处同一个球会被找到两次，留置信度高的。"""
    kept: list[list] = []
    for c in sorted(found, key=lambda c: -c[3]):
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > max(c[2], k[2]) for k in kept):
            kept.append(c)
    return kept


def people_at(frames: list[dict], fps: float, t: float) -> dict:
    """跟踪只有 10fps：取前后两帧，同一人的框按时间线性插值。"""
    k = min(len(frames) - 1, max(0, int((t - frames[0]["time"]) * fps)))
    a = frames[k]
    b = frames[min(len(frames) - 1, k + 1)]
    if b is a or b["time"] <= a["time"]:
        return a
    w = min(1.0, max(0.0, (t - a["time"]) / (b["time"] - a["time"])))
    later = {p["id"]: p for p in b["people"]}
    people = []
    for p in a["people"]:
        q = later.get(p["id"])
        box = [u + (v - u) * w for u, v in zip(p["box"], q["box"])] if q else p["box"]
        people.append({**p, "box": box})
    return {"time": t, "people": people}


def remove_false_positives(candidates: list[list], frames: list[dict], court: Court) -> list[list]:
    cells = Counter()
    for found in candidates:
        for cell in {(int(c[0] // STATIC_CELL), int(c[1] // STATIC_CELL)) for c in found}:
            cells[cell] += 1
    static = {cell for cell, n in cells.items() if n > STATIC_SHARE * len(candidates)}
    xs = [p[0] for p in court.polygon]
    bottom = max(p[1] for p in court.polygon) - 1
    out = []
    for found, entry in zip(candidates, frames):
        boxes = [p["box"] for p in entry["people"] if p["team"] in ("A", "B", "ref")]
        keep = []
        for c in found:
            x, y = c[0], c[1]
            cx, cy = int(x // STATIC_CELL), int(y // STATIC_CELL)
            if any((cx + dx, cy + dy) in static for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            if not min(xs) <= x <= max(xs):
                continue
            # 球不会在场地多边形下方（靠镜头一侧场外），上方可以（空中）
            # 在场地里，或者在场地正上方（空中）：往下逐段找，正下方有一段落在场地里就算
            if not any(court.contains(x, min(bottom, y + dy)) for dy in range(0, HEAD_ROOM + 1, 40)):
                continue
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[1] + HEAD_SHARE * (b[3] - b[1]) for b in boxes):
                continue
            near_player = any(_box_distance(b, x, y) <= NEAR_PLAYER for b in boxes)
            body = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 100.0
            near_hoop = any(np.hypot(x - hx, y - hy) <= NEAR_HOOP * body for hx, hy in court.hoops)
            if not (near_player or near_hoop):
                continue
            keep.append(c)
        out.append(keep)
    return out


def _box_distance(b: list, x: float, y: float) -> float:
    """点到人框的距离，以人身高为单位（在框内为 0）。"""
    dx = max(b[0] - x, 0, x - b[2])
    dy = max(b[1] - y, 0, y - b[3])
    return float(np.hypot(dx, dy)) / max(1.0, b[3] - b[1])


def _typical_height(entry: dict, x: float, y: float) -> float:
    """球附近的人身高（远处的人小，球移动的像素也小）。"""
    boxes = [p["box"] for p in entry["people"]]
    if not boxes:
        return 150.0
    b = min(boxes, key=lambda b: abs((b[0] + b[2]) / 2 - x) + abs((b[1] + b[3]) / 2 - y))
    return max(60.0, b[3] - b[1])


def best_path(candidates: list[list], frames: list[dict], fps: float, miss_cost: float = MISS_COST) -> list[list | None]:
    """动态规划挑一条球轨迹。状态 = 某帧选哪个候选，或“这帧看不到球”。
    看不到球之后重新出现时，和最后一次看到的位置比距离（允许按丢失帧数放宽）。"""
    n = len(candidates)
    # 每个状态：(累计得分, 上一状态, 最后看到的候选, 离最后看到过了几帧)
    layers: list[list[tuple]] = []
    prev = [(0.0, None, None, 0)]  # 起点：没有球
    for i in range(n):
        layer = []
        for c in candidates[i]:
            h = _typical_height(frames[i], c[0], c[1])
            best = None
            for k, (score, _p, last, gap) in enumerate(prev):
                if last is not None:
                    d = np.hypot(c[0] - last[0], c[1] - last[1]) / h
                    if d > MAX_SPEED / fps * (gap + 1):
                        continue
                    s = score + c[3] - 0.1 * d / (gap + 1)
                else:
                    s = score + c[3]
                if best is None or s > best[0]:
                    best = (s, k, c, 0)
            if best is None:  # 和之前的轨迹都接不上：从这里新开始
                best_k = int(np.argmax([p[0] for p in prev]))
                best = (prev[best_k][0] + c[3] - 1.0, best_k, c, 0)
            layer.append(best)
        # “看不到球”状态：从上一层每个状态各延续一个，只留最好的几个，免得状态爆炸
        missing = sorted(
            ((score + miss_cost, k, last, gap + 1) for k, (score, _p, last, gap) in enumerate(prev)),
            key=lambda s: -s[0],
        )[:4]
        missing = [m if m[3] <= 3 * fps else (m[0], m[1], None, 0) for m in missing]  # 丢失 3 秒以上就忘掉位置
        layer += missing
        layers.append(layer)
        prev = layer
    k = int(np.argmax([s[0] for s in prev]))
    path: list[list | None] = [None] * n
    for i in range(n - 1, -1, -1):
        state = layers[i][k]
        path[i] = state[2] if state[3] == 0 else None
        k = state[1]
    return path


def holder_of(ball: list | None, entry: dict) -> str | None:
    """球离谁的手最近。手部区域：人框中间偏上到脚，左右各放宽 1/4。"""
    if ball is None:
        return None
    x, y = ball[0], ball[1]
    best, best_d = None, HOLD_DISTANCE
    for p in entry["people"]:
        label = p.get("person") or p.get("player")  # 第二遍确认过的身份优先
        if not label:
            continue
        x1, y1, x2, y2 = p["box"]
        h = y2 - y1
        hx1, hx2 = x1 - 0.25 * (x2 - x1), x2 + 0.25 * (x2 - x1)
        hy1, hy2 = y1 + 0.2 * h, y2  # 运球时球在脚边，接球/持球在胸前
        dx = max(hx1 - x, 0, x - hx2)
        dy = max(hy1 - y, 0, y - hy2)
        d = np.hypot(dx, dy) / h + 0.15 * abs(x - (x1 + x2) / 2) / h  # 都碰到时选离身体中线近的
        if d < best_d:
            best, best_d = label, d
    return best


def smooth_holders(holders: list[str | None], fps: float) -> list[str | None]:
    out = list(holders)
    fill_gap, min_hold = round(FILL_GAP * fps), max(1, round(MIN_HOLD * fps))
    # 同一人中间短暂丢失：补上
    last, last_i = None, -1
    for i, h in enumerate(out):
        if h is None:
            continue
        if h == last and 1 < i - last_i <= fill_gap + 1:
            for j in range(last_i + 1, i):
                out[j] = h
        last, last_i = h, i
    # 太短的持球段：多半是球从旁边飞过，去掉
    i = 0
    while i < len(out):
        if out[i] is None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < min_hold:
            for k in range(i, j):
                out[k] = None
        i = j
    return out


def possessions(holders: list[str | None], times: list[float]) -> list[dict]:
    """连续持球段：{player, start, end, received}。received = 从别人那里接到球（不是自己丢了又捡回）。"""
    out = []
    i = 0
    prev_player = None
    while i < len(holders):
        if holders[i] is None:
            i += 1
            continue
        j = i
        while j < len(holders) and holders[j] == holders[i]:
            j += 1
        out.append({"player": holders[i], "start": times[i], "end": times[j - 1],
                    "received": prev_player is not None and prev_player != holders[i]})
        prev_player = holders[i]
        i = j
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--redetect", action="store_true", help="重新检测（默认复用 ball_raw.json）")
    parser.add_argument("--detect-only", action="store_true", help="只做逐帧找球（不需要认人结果，可以提前跑）")
    args = parser.parse_args()

    out_dir = args.out or out_dir_for(args.video)
    # 有第二遍（认整场）的结果就用它：持球人是确认过的人；只做检测时有跟踪结果就够了
    for name in ("final.json", "identified.json", "tracks.json"):
        if (out_dir / name).exists():
            result = json.loads((out_dir / name).read_text())
            break
    court = Court(polygon=[tuple(p) for p in result["court"]["polygon"]], hoops=[tuple(p) for p in result["court"]["hoops"]])
    frames = result["frames"]
    raw_file = out_dir / "ball_raw.json"
    raw_data = json.loads(raw_file.read_text()) if raw_file.exists() else {}
    current = model_id(ball_model_for(args.video))
    if args.redetect or raw_data.get("fps") != DETECT_FPS or raw_data.get("model") != current:
        end = frames[-1]["time"] + 0.5 / result["fps"]
        print(f"找球：{end - result['start']:.0f} 秒，每秒 {DETECT_FPS:.0f} 帧")
        raw = detect_candidates(args.video, result["info"], court, result["start"], end, DETECT_FPS)
        raw_data = {"fps": DETECT_FPS, "start": result["start"], "model": current, "frames": raw}
        raw_file.write_text(json.dumps(raw_data))
    if args.detect_only:
        print(f"逐帧找球完成：{sum(bool(f) for f in raw_data['frames'])} 帧有候选")
        return 0
    raw = raw_data["frames"]
    fps = raw_data["fps"]
    times = [raw_data["start"] + i / fps for i in range(len(raw))]
    entries = [people_at(frames, result["fps"], t) for t in times]

    candidates = remove_false_positives(raw, entries, court)
    # 训练过的模型置信度普遍低（0.05 就算球），固定 0.3 的代价会让“看不到球”永远划算
    path = best_path(candidates, entries, fps, MISS_COST * ball_conf_for(args.video, strict=True) / STRICT_CONF)
    holders = smooth_holders([holder_of(b, e) for b, e in zip(path, entries)], fps)
    poss = possessions(holders, times)
    (out_dir / "ball.json").write_text(json.dumps(
        {"fps": fps, "frames": [{"time": round(t, 3), "ball": b, "holder": h} for t, b, h in zip(times, path, holders)],
         "possessions": poss},
        ensure_ascii=False,
    ))
    seen = sum(b is not None for b in path)
    held = sum(h is not None for h in holders)
    print(f"看到球的帧：{seen}/{len(times)}（{seen / len(times):.0%}），能判断持球人的帧：{held}")
    per = Counter(p["player"] for p in poss)
    print("持球次数：" + "  ".join(f"{k} {v}" for k, v in per.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
