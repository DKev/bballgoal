"""找比赛事件：出手（进 / 没进）。都先当 2 分投篮，三分和罚篮在审核时标（自动认不准）。

1. 篮筐监控：两个篮筐周围各裁一块小画面，放大到篮圈够宽后逐帧找球（全场画面里球太小，
   篮筐附近放大看才准）。篮圈的确切位置和大小从画面里自动找（球/篮筐模型的篮筐类），不写死；
2. 进没进：直接用原项目（basketball-highlight-editor）成熟的判断：球从上方进入、在篮圈内穿过
   篮圈平面、从网底出来、没有弹起，再加上球网有没有被推动；看不清的标“拿不准”交给审核；
3. 记到人头上：用持球数据（bballgoal.ball）——出手前最后一个持球的人是投篮人；认不出人的按篮筐推断是哪队。
   （篮板、助攻、抢断试过，持球数据不够准，已经不做了。）

用法（先跑完 track、identify/assign、ball）：
    .venv/bin/python -m bballgoal.events output/<比赛>/full.mp4
输出 events.json。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bballgoal.ball import BALL_MODEL, ball_conf_for, ball_model_for, model_id
from bballgoal.court import Court
from bballgoal.detect import default_device
from bballgoal.frames import read_frames
from bballgoal.paths import out_dir as out_dir_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from basketball_highlight.net_motion import net_motion_series, prepare  # noqa: E402
from basketball_highlight.rim_check import UNSEEN_REASONS, assess_rim_crossing  # noqa: E402

RIM_PIXELS = 48  # 放大到篮圈至少这么宽（原项目要求 ≥30）
CROP_RIM_WIDTHS = 6.0  # 篮筐小画面边长 = 篮圈宽的这么多倍（够看到下落和出网）
SCAN_FPS = 25.0
APPROACH = (1.5, -2.5, -0.1)  # 篮圈坐标（以篮圈宽为单位）：|x|≤1.5、-2.5≤y≤-0.1 = 从上方接近
WINDOW_BEFORE, WINDOW_AFTER = 1.0, 1.6  # 秒：每次接近篮筐，前后各取多少交给进球判断
MERGE_GAP = 1.5  # 秒：两次接近间隔小于这个算同一次出手
SHOOTER_GAP = 2.5  # 秒：最后一次持球结束到球到篮筐，超过这个就不确定是谁投的
SHOOTER_GUESS_GAP = 6.0  # 秒：更早的最后持球人也先填上，但标成“拿不准”请你确认（突破上篮时球常常跟丢好几秒）
CLIP_BEFORE = 5.5  # 秒：片段从球到篮筐前这么久开始（出手前留几秒组织进攻，不会一拿球就投、反应不过来）
CLIP_AFTER = 2.0  # 秒：球到篮筐之后再拍这么久
TEAM_VOTES = 5  # 不知道是谁投的：看同一个篮筐、时间最近的这么多次已知投篮人是哪队（每半场每队固定攻一个篮筐）
CARRY_TIME = 0.5  # 秒：重新找事件时，同类型、时间差这么小的算同一个事件，沿用编号


# ---------- 篮圈 ----------

def find_rims(video: Path, hoops: list, info: dict, samples: int = 40) -> list[dict]:
    """在每个篮筐点附近找篮筐框，取中位数：篮圈中心、篮圈平面高度、篮圈宽。"""
    model = YOLO(str(BALL_MODEL))  # 篮筐类在原项目模型里
    hoop_cls = [k for k, v in model.names.items() if v.lower() == "hoop"]
    cap = cv2.VideoCapture(str(video))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    found: list[list[np.ndarray]] = [[] for _ in hoops]
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        for j, (hx, hy) in enumerate(hoops):
            side = 220
            x0, y0 = int(max(0, hx - side / 2)), int(max(0, hy - side / 2))
            crop = frame[y0 : y0 + side, x0 : x0 + side]
            if crop.size == 0:
                continue
            r = model.predict(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), imgsz=640, conf=0.25,
                              classes=hoop_cls, device=default_device(), verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy():
                b = b / 3 + [x0, y0, x0, y0]
                if np.hypot((b[0] + b[2]) / 2 - hx, (b[1] + b[3]) / 2 - hy) < side / 3:
                    found[j].append(b)
    cap.release()
    rims = []
    for (hx, hy), boxes in zip(hoops, found):
        if len(boxes) >= 3:
            b = np.median(np.array(boxes), axis=0)
            width = float(b[2] - b[0])
            rims.append({"center_x": float((b[0] + b[2]) / 2), "rim_y": float(b[1] + 0.1 * (b[3] - b[1])), "width": width, "found": len(boxes)})
        else:  # 没找到：用你点的位置，宽度给个保守值
            rims.append({"center_x": float(hx), "rim_y": float(hy), "width": 20.0, "found": 0})
    return rims


def crop_geometry(rim: dict, frame_w: int, frame_h: int) -> dict:
    """篮筐小画面：原画上的范围和放大倍数。返回放大后坐标系里的篮圈。"""
    side = CROP_RIM_WIDTHS * rim["width"]
    x0 = int(max(0, rim["center_x"] - side / 2))
    y0 = int(max(0, rim["rim_y"] - side * 0.45))
    x1, y1 = int(min(frame_w, x0 + side)), int(min(frame_h, y0 + side))
    scale = max(1.0, RIM_PIXELS / rim["width"])
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "scale": scale,
            "rim": {"center_x": (rim["center_x"] - x0) * scale, "rim_y": (rim["rim_y"] - y0) * scale, "width": rim["width"] * scale}}


def cut(frame: np.ndarray, g: dict) -> np.ndarray:
    crop = frame[g["y0"] : g["y1"], g["x0"] : g["x1"]]
    return cv2.resize(crop, None, fx=g["scale"], fy=g["scale"], interpolation=cv2.INTER_CUBIC)


# ---------- 逐帧看篮筐 ----------

def scan(video: Path, result: dict, geoms: list[dict], fps: float = SCAN_FPS) -> list[list[list[dict]]]:
    """每个篮筐每帧的球检测（放大后的小画面坐标）。dets[篮筐][帧] = [{name, confidence, xyxy}]。"""
    model = YOLO(str(ball_model_for(video)))
    cls = [k for k, v in model.names.items() if v.lower() == "ball"]
    info = result["info"]
    end = result["frames"][-1]["time"] + 1.0
    dets: list[list[list[dict]]] = [[] for _ in geoms]
    started = time.perf_counter()
    for i, (_t, frame) in enumerate(read_frames(video, info["width"], info["height"], result["start"], end, fps)):
        crops = [cut(frame, g) for g in geoms]
        size = int(np.ceil(max(c.shape[1] for c in crops) / 32) * 32)
        for j, r in enumerate(model.predict(crops, imgsz=size, conf=ball_conf_for(video), classes=cls,
                                            device=default_device(), verbose=False)):
            dets[j].append([{"name": "ball", "confidence": float(c), "xyxy": [float(v) for v in b]}
                            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())])
        if i and i % 1000 == 0:
            total = int((end - result['start']) * fps)
            rate = i / (time.perf_counter() - started)
            print(f"  {i}/{total} 帧  剩余约 {max(0, total - i) / rate / 60:.1f} 分钟", flush=True)
    return dets


def approach_windows(frames: list[list[dict]], rim: dict, fps: float, min_conf: float = 0.25) -> list[tuple[int, int]]:
    """球从上方接近篮圈的时刻，合并成一次次出手的窗口（帧号）。"""
    xs, top, bottom = APPROACH
    hits = []
    for i, items in enumerate(frames):
        for d in items:
            x1, y1, x2, y2 = d["xyxy"]
            x = ((x1 + x2) / 2 - rim["center_x"]) / rim["width"]
            y = ((y1 + y2) / 2 - rim["rim_y"]) / rim["width"]
            if abs(x) <= xs and top <= y <= bottom and d["confidence"] >= min_conf:
                hits.append(i)
                break
    windows: list[list[int]] = []
    for i in hits:
        if windows and i - windows[-1][1] <= MERGE_GAP * fps:
            windows[-1][1] = i
        else:
            windows.append([i, i])
    n = len(frames)
    return [(max(0, a - int(WINDOW_BEFORE * fps)), min(n, b + int(WINDOW_AFTER * fps))) for a, b in windows]


def path_windows(ball_frames: list[dict], rim: dict, start: float, n: int, fps: float) -> list[tuple[int, int]]:
    """第二个线索：全场球轨迹里，球出现在篮圈上方附近（篮筐小画面没认出球时也能抓到出手）。"""
    times = []
    for f in ball_frames:
        b = f.get("ball")
        if not b:
            continue
        x = (b[0] - rim["center_x"]) / rim["width"]
        y = (b[1] - rim["rim_y"]) / rim["width"]
        if abs(x) <= 3.0 and -6.0 <= y <= 0.5:
            times.append(f["time"])
    windows: list[list[float]] = []
    for t in times:
        if windows and t - windows[-1][1] <= MERGE_GAP:
            windows[-1][1] = t
        else:
            windows.append([t, t])
    return [(max(0, int((a - start - WINDOW_BEFORE) * fps)), min(n, int((b - start + WINDOW_AFTER) * fps))) for a, b in windows]


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def judge(video: Path, result: dict, g: dict, frames: list[list[dict]], window: tuple[int, int], fps: float) -> dict:
    """一次出手：原项目的篮圈穿越判断 + 球网晃动。"""
    a, b = window
    info = result["info"]
    grays = [prepare(cut(fr, g)) for _t, fr in read_frames(video, info["width"], info["height"],
                                                             result["start"] + a / fps, result["start"] + b / fps, fps)]
    window_dets = frames[a : a + len(grays)]
    motion = net_motion_series(grays, window_dets, g["rim"], (0, 0)) if len(grays) > 2 else None
    verdict = assess_rim_crossing(window_dets, g["rim"], motion)
    if verdict.get("status") == "made":
        result_ = "made"
    elif verdict.get("reason") in UNSEEN_REASONS or verdict.get("status") == "skipped":
        result_ = "unsure"
    else:
        result_ = "missed"
    if result_ == "missed" and (verdict.get("net_motion") or 0) > 2.0:  # 网明显被推动：可能是进了没看清
        result_ = "unsure"
    cross = verdict.get("cross_frame")
    t = result["start"] + (a + cross if cross is not None else (a + b) / 2) / fps
    return {"time": round(t, 2), "result": result_, "reason": verdict.get("reason"), "net_motion": verdict.get("net_motion")}


# ---------- 记到人头上 ----------

def team_of(label: str | None, people: dict) -> str | None:
    if not label:
        return None
    if label in people:
        return people[label].get("team") or None
    return label[0] if label[:1] in ("A", "B") else None


def attribute(shots: list[dict], possessions: list[dict], result: dict, rims: list[dict], people: dict) -> list[dict]:
    events = []
    for s in shots:
        before = [p for p in possessions if p["end"] <= s["time"] and s["time"] - p["end"] <= SHOOTER_GUESS_GAP]
        shooter = max(before, key=lambda p: p["end"]) if before else None
        if shooter and not team_of(shooter["player"], people):  # 最后碰球的是裁判/教练：死球（比如罚球递球），不猜
            shooter = None
        guessed = bool(shooter) and s["time"] - shooter["end"] > SHOOTER_GAP
        ev = {"type": "shot", "time": s["time"], "hoop": s["hoop"], "result": s["result"], "kind": "two",
              "player": shooter["player"] if shooter else None, "team": team_of(shooter["player"], people) if shooter else None,
              "release": shooter["end"] if shooter else None, "detail": {"reason": s["reason"], "net_motion": s["net_motion"]}, "guessed": guessed}
        events.append(ev)
    guess_teams(events)
    for k, ev in enumerate(sorted(events, key=lambda e: e["time"])):
        ev["id"] = k + 1
        ev["clip"] = [round(ev["time"] - CLIP_BEFORE, 2), round(ev["time"] + CLIP_AFTER, 2)]
        ev["sure"] = ev.get("player") is not None and ev.get("result", "made") != "unsure" and not ev.get("guessed")
    return sorted(events, key=lambda e: e["time"])


def guess_teams(events: list[dict]) -> None:
    """不知道投篮人的出手也要算进球队比分：每半场每队固定攻一个篮筐，
    用同一个篮筐、时间最近的几次已知投篮人的队伍投票（team_guess 标出来，审核时可以改）。"""
    known = [e for e in events if e["type"] == "shot" and e.get("team")]
    for e in events:
        if e["type"] != "shot" or e.get("team"):
            continue
        near = sorted((k for k in known if k["hoop"] == e["hoop"]), key=lambda k: abs(k["time"] - e["time"]))[:TEAM_VOTES]
        votes = Counter(k["team"] for k in near)
        if votes:
            team, n = votes.most_common(1)[0]
            if n > len(near) / 2:
                e["team"], e["team_guess"] = team, True


def carry_ids(events: list[dict], old: list[dict]) -> list[int]:
    """重新找事件（比如认人核对完再跑一遍）时，沿用上一次同一事件的编号：你的审核结果和渲染好的片段都按编号存。
    同类型、时间差不到 CARRY_TIME 的算同一事件。返回片段已经过时的编号（内容变了的、消失了的），由调用方删片段。"""
    free = {e["id"]: e for e in old}
    next_id = max((e["id"] for e in old), default=0) + 1
    for e in events:
        match = min((o for o in free.values() if o["type"] == e["type"] and abs(o["time"] - e["time"]) <= CARRY_TIME),
                    key=lambda o: abs(o["time"] - e["time"]), default=None)
        if match:
            e["id"] = match["id"]
            del free[match["id"]]
            if any(match.get(k) != e.get(k) for k in ("player", "result", "clip")):
                free[e["id"]] = match  # 编号沿用，但片段要重新渲染
        else:
            e["id"] = next_id
            next_id += 1
    return sorted(free)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scan-only", action="store_true", help="只做篮筐逐帧检测（不需要认人结果，可以提前跑）")
    args = parser.parse_args()
    out = args.out or out_dir_for(args.video)
    for name in ("final.json", "identified.json", "tracks.json"):
        if (out / name).exists():
            result = json.loads((out / name).read_text())
            break
    people = result.get("people", {})
    court = Court.load(args.video.with_suffix(".court.json"))
    info = result["info"]

    rims = find_rims(args.video, court.hoops, info)
    print("篮圈：" + "；".join(f"{'左右'[j]} 宽 {r['width']:.0f} 像素（找到 {r['found']} 次）" for j, r in enumerate(rims)))
    geoms = [crop_geometry(r, info["width"], info["height"]) for r in rims]
    scan_file = out / "rim_scan.json"
    current = model_id(ball_model_for(args.video))
    cached = json.loads(scan_file.read_text()) if scan_file.exists() else {}
    if isinstance(cached, dict) and cached.get("model") == current and cached.get("rims") == rims:
        dets = cached["dets"]
    else:  # 没算过，或者找球模型换了（点球训练之后）、篮圈变了：重新逐帧看
        print("逐帧看篮筐…", flush=True)
        dets = scan(args.video, result, geoms)
        scan_file.write_text(json.dumps({"model": current, "rims": rims, "dets": dets}))
    if args.scan_only:
        print("篮筐逐帧检测完成")
        return 0
    ball = json.loads((out / "ball.json").read_text())
    shots = []
    for j, g in enumerate(geoms):
        windows = approach_windows(dets[j], g["rim"], SCAN_FPS, ball_conf_for(args.video, strict=True)) + path_windows(ball["frames"], rims[j], result["start"], len(dets[j]), SCAN_FPS)
        for w in merge_windows(windows):
            shots.append({**judge(args.video, result, g, dets[j], w, SCAN_FPS), "hoop": j})
    events = attribute(shots, ball["possessions"], result, rims, people)
    events_file = out / "events.json"
    if events_file.exists():
        stale = carry_ids(events, json.loads(events_file.read_text()).get("events", []))
        for k in stale:
            (args.video.parent / "clips" / f"{k}.mp4").unlink(missing_ok=True)
        if stale:
            print(f"沿用了上次的事件编号，{len(stale)} 个片段过时已删除，渲染时重做")
    events.sort(key=lambda e: e["time"])
    events_file.write_text(json.dumps({"rims": rims, "events": events}, ensure_ascii=False, indent=1))
    count = defaultdict(int)
    for e in events:
        count[e["type"] + ("_" + e["result"] if e["type"] == "shot" else "")] += 1
    print("事件：" + "  ".join(f"{k} {v}" for k, v in sorted(count.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
