"""把事件渲染成小片段：虚拟镜头从 360 原片取“普通镜头”画面，对准当事人（出手时连篮筐一起；不知道是谁投的就跟着球），
脚下画圈，下方写中文说明（“红队 #18 小明 · 投篮 · 命中 · 2 分”）。事件审核和集锦都用它。

    .venv/bin/python -m bballgoal.clips output/<比赛>/full.mp4            # 渲染所有还没有片段的事件
    .venv/bin/python -m bballgoal.clips output/<比赛>/full.mp4 --event 12 # 只渲染一个
输出 output/<比赛>/clips/<事件号>.mp4。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bballgoal.ball import people_at
from bballgoal.cards import game_clock
from bballgoal.paths import out_dir as out_dir_for
from bballgoal.vcam import OUT_H, OUT_W, Renderer, VideoWriter, analysis_to_dir, plan_shot, project_to_view, read_fisheye

FPS = 25.0
CONTEXT_BODIES = 2.5  # 当事人身边这么多倍身高内的人也拍进来
HOLDER_WINDOW = (2.2, 0.6)  # 秒：认不出投篮人时，在球到篮筐前这段时间里找拿球的人
BALL_RECENT = 1.0  # 秒：当事人不在画面里时，镜头跟着球；球跟丢不超过这么久，用最近一次看到的位置
NEAR_RIM_PLAYERS = 6  # 出手片段把离篮筐最近的这几个球员都拍进来（场上 10 个人的大部分），看得清谁投的、谁在抢位
RIM_SHOW = (-1.5, 1.0)  # 秒：只在球到篮筐前后这段时间把篮筐带进画面（篮筐在高处，一直带着镜头就得拉得很远）
FONTS = ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
         "/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf")
KIND_NAMES = {"two": "投篮", "three": "三分", "freethrow": "罚篮"}
KIND_POINTS = {"two": 2, "three": 3, "freethrow": 1}
RESULT_NAMES = {"made": "命中", "missed": "没进", "unsure": "进没进拿不准"}
MARK = (0, 215, 255)
RENDER_VERSION = 3  # 镜头/画面的做法改了就加一：旧片段会自动重新渲染
SIGNATURE_KEYS = ("type", "player", "result", "kind", "points", "clip", "hoop")  # 审核里能改的、画面会变的


def font(size: int):
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def player_text(label: str | None, people: dict, teams: dict) -> str:
    if not label:
        return "（还不知道是谁）"
    p = people.get(label)
    if p:
        team = teams.get(p.get("team"), {}).get("name", p.get("team") or "")
        number = f"#{p['number']}" if p.get("number") else label
        return " ".join(x for x in (team, number, p.get("name") or "") if x)
    if label[:1] in teams and "#" in label:
        return f"{teams[label[0]]['name']} #{label.split('#')[1]}"
    return label


def shot_kind(event: dict) -> str:
    """投篮 two / 三分 three / 罚篮 freethrow。以前的数据里有上篮、跳投（都算投篮），或者只改过分数（1 分算罚篮、3 分算三分）。"""
    kind = event.get("kind")
    if kind in KIND_POINTS:
        return kind
    return {1: "freethrow", 3: "three"}.get(event.get("points"), "two")


def shot_points(event: dict) -> int:
    return KIND_POINTS[shot_kind(event)]


def shot_text(event: dict) -> str:
    """“投篮 · 命中 · 2 分”。"""
    parts = [KIND_NAMES[shot_kind(event)], RESULT_NAMES.get(event.get("result"), "")]
    if event.get("result") == "made":
        parts.append(f"{shot_points(event)} 分")
    return " · ".join(p for p in parts if p)


def caption(event: dict, people: dict, teams: dict) -> str:
    what = "拿球" if event["type"] == "touch" else shot_text(event)
    return f"{player_text(event.get('player'), people, teams)} · {what}"


def label_of(p: dict) -> str | None:
    return p.get("person") or p.get("player")


def _around(entry: dict, cx: float, cy: float, h: float) -> list:
    return [p["box"] for p in entry["people"] if p["team"] in ("A", "B")
            and np.hypot((p["box"][0] + p["box"][2]) / 2 - cx, (p["box"][1] + p["box"][3]) / 2 - cy) <= CONTEXT_BODIES * h]


def ball_near(ball: dict | None, t: float) -> list | None:
    """t 前后 BALL_RECENT 秒内最近一次看到的球（ball.json 的 [x, y, 大小, 置信度]）。"""
    if not ball or not ball["frames"]:
        return None
    frames, fps = ball["frames"], ball["fps"]
    i = int(round((t - frames[0]["time"]) * fps))
    for k in range(int(BALL_RECENT * fps) + 1):
        for j in (i - k, i + k):
            if 0 <= j < len(frames) and frames[j]["ball"]:
                return frames[j]["ball"]
    return None


def targets_for(result: dict, event: dict, times: np.ndarray, rims: list[dict], ball: dict | None = None,
                track: set[int] | None = None) -> list:
    """每个时刻要拍进画面的框：投篮人和他身边的人，再加上篮下离篮筐最近的几个球员（看清大部分人在干什么），
    球到篮筐前后带上篮筐，但不只盯着篮筐；投篮人不在画面里（没认出来、或者猜错了）就再加上球和球附近的人。"""
    rim = rims[event["hoop"]] if event.get("hoop") is not None and rims else None
    out = []
    for t in times:
        entry = people_at(result["frames"], result["fps"], t)
        me = _subject(entry, event, track)
        heights = [p["box"][3] - p["box"][1] for p in entry["people"] if p["team"] in ("A", "B")]
        typical = float(np.median(heights)) if heights else 100.0
        boxes = []
        if me:
            b = me[0]["box"]
            boxes = _around(entry, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]) or [b]
        if rim is not None:
            players = sorted((p["box"] for p in entry["people"] if p["team"] in ("A", "B")),
                             key=lambda b: abs((b[0] + b[2]) / 2 - rim["center_x"]))
            boxes += players[:NEAR_RIM_PLAYERS]
            pos = None if me else ball_near(ball, t)
            if pos:
                r = typical / 4
                boxes += _around(entry, pos[0], pos[1], typical) + [[pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r]]
            w = rim["width"]
            if RIM_SHOW[0] <= t - event["time"] <= RIM_SHOW[1]:
                boxes.append([rim["center_x"] - w, rim["rim_y"] - w, rim["center_x"] + w, rim["rim_y"] + w])
        out.append(np.array(boxes) if boxes else None)
    return out


_ball_cache: dict = {}


def load_ball(video: Path) -> dict | None:
    """ball.json 每个片段都要用，读一次共用（按修改时间失效）。"""
    f = out_dir_for(video) / "ball.json"
    if not f.exists():
        return None
    key = (f, f.stat().st_mtime)
    if key not in _ball_cache:
        _ball_cache.clear()
        _ball_cache[key] = json.loads(f.read_text())
    return _ball_cache[key]


def _subject(entry: dict, event: dict, track: set[int] | None) -> list:
    if track:
        return [p for p in entry["people"] if p["id"] in track]
    return [p for p in entry["people"] if label_of(p) == event.get("player")] if event.get("player") else []


def subject_tracks(result: dict, event: dict, ball: dict | None) -> set[int] | None:
    """投篮人在出手之前一直认不出来（比如你在审核里改了人，跟踪当时没认出他）：改跟“出手前拿球的那个人”，
    沿他的轨迹往前接到片段开头。认得出就返回 None（按名字跟）。"""
    t0 = event["clip"][0]
    if event.get("player") and any(label_of(p) == event["player"]
                                   for t in np.arange(t0, event["time"] - HOLDER_WINDOW[1], 0.2)
                                   for p in people_at(result["frames"], result["fps"], t)["people"]):
        return None
    for t in np.arange(event["time"] - HOLDER_WINDOW[0], event["time"] - HOLDER_WINDOW[1], 0.1):
        tid = holder_track(result, ball, t)
        if tid is not None:
            return track_chain(result, tid, t0, event["clip"][1])
    return None


def plan_for(game_dir: Path, video: Path, result: dict, event: dict, rims: list[dict]):
    """这个事件片段的虚拟镜头：(镜头, 画面参数, 开始秒, 结束秒, 跟拍的轨迹)。渲染片段和集锦的开场特效共用，位置才对得上。"""
    ball = load_ball(video)
    track = subject_tracks(result, event, ball)
    view = json.loads((game_dir / "full.view.json").read_text())
    t0, t1 = event["clip"]
    t0, t1 = max(result["frames"][0]["time"], t0), min(result["frames"][-1]["time"], t1)
    times = np.arange(t0, t1 + 1e-6, 0.1)
    shot = plan_shot(times, targets_for(result, event, times, rims, ball, track), view, min_fov=28, max_fov=70,
                     margin=1.35, smooth=0.6)
    return shot, view, t0, t1, track


def mark_at(result: dict, event: dict, shot, view: dict, t: float, track: int | set[int] | None = None) -> dict | None:
    """t 时刻当事人（给了 track 就是这条跟踪轨迹上的人）在片段画面里的位置：脚下中心 foot、圈的半宽 w、
    头顶 head（像素）。不在画面里返回 None。"""
    entry = people_at(result["frames"], result["fps"], t)
    me = _subject(entry, event, {track} if isinstance(track, int) else track)
    if not me:
        return None
    b = me[0]["box"]
    yaw, pitch, hfov = shot.at(t)
    pts = analysis_to_dir(np.array([b[0], b[2], (b[0] + b[2]) / 2]), np.array([b[3], b[3], b[1]]), view)
    u, v = project_to_view(pts, yaw, pitch, hfov)
    if np.isnan(u).any() or not (0 <= (u[0] + u[1]) / 2 < OUT_W and 0 <= v[0] < OUT_H):
        return None
    return {"foot": ((u[0] + u[1]) / 2, float(v[0])), "w": max(12, int(abs(u[1] - u[0]) * 0.7)), "head": (float(u[2]), float(v[2]))}


def holder_track(result: dict, ball: dict | None, t: float) -> int | None:
    """认不出投篮人时（比如你在审核里改了人，跟踪那几秒没认出他）：t 时刻离球最近、手够得着球的人的跟踪轨迹号，
    沿着这条轨迹往前就能在片段开头找到他。"""
    pos = ball_near(ball, t)
    if not pos:
        return None
    entry = people_at(result["frames"], result["fps"], t)
    near = [p for p in entry["people"] if p["team"] in ("A", "B")
            and np.hypot((p["box"][0] + p["box"][2]) / 2 - pos[0], (p["box"][1] + p["box"][3]) / 2 - pos[1]) <= 0.8 * (p["box"][3] - p["box"][1])]
    if not near:
        return None
    return min(near, key=lambda p: np.hypot((p["box"][0] + p["box"][2]) / 2 - pos[0], (p["box"][1] + p["box"][3]) / 2 - pos[1]))["id"]


CHAIN_GAP = 1.2  # 秒：接轨迹时，两截之间最多断这么久（钻进人堆被挡住，常常一秒左右才重新出现）
CHAIN_REACH = 0.6  # 两截衔接处的距离 < 这么多倍身高（再加上断开这段时间能走的距离），才算同一个人
CHAIN_SPEED = 1.0  # 每秒最多走这么多倍身高


def _next_piece(frames: list[dict], cur: int, chain: set[int], forward: bool) -> int | None:
    """cur 这一截结束后（forward）/ 开始前，紧挨着冒出来 / 刚消失的另一截。"""
    idx = [i for i, f in enumerate(frames) if any(p["id"] == cur for p in f["people"])]
    if not idx:
        return None
    end = idx[-1] if forward else idx[0]
    me = next(p for p in frames[end]["people"] if p["id"] == cur)
    box, team = me["box"], me["team"]
    h = box[3] - box[1]
    near = frames[max(0, end - 2):end + 1] if forward else frames[end:end + 3]
    busy = {p["id"] for f in near for p in f["people"]}  # 同时在场的不是他
    step = 1 if forward else -1
    j = end + step
    while 0 <= j < len(frames) and abs(frames[j]["time"] - frames[end]["time"]) <= CHAIN_GAP:
        cands = [(np.hypot((p["box"][0] + p["box"][2] - box[0] - box[2]) / 2, p["box"][3] - box[3]) / max(1.0, h), p["id"])
                 for p in frames[j]["people"] if p["id"] not in busy and p["id"] not in chain and p["team"] == team]
        reach = CHAIN_REACH + CHAIN_SPEED * abs(frames[j]["time"] - frames[end]["time"])
        cands = [c for c in cands if c[0] <= reach]
        if cands:
            return min(cands)[1]
        j += step
    return None


def _span(frames: list[dict], tid: int) -> tuple[float, float]:
    ts = [f["time"] for f in frames if any(p["id"] == tid for p in f["people"])]
    return ts[0], ts[-1]


def track_chain(result: dict, track: int, t_min: float, t_max: float = float("inf")) -> set[int]:
    """跟踪轨迹常常中途断成几截（换了号）。从 track 往前、往后接：紧挨着消失 / 冒出来、位置挨着的另一截，
    一直接到 t_min ~ t_max 都覆盖。认不出投篮人、只能跟着拿球的人时用：开场能从片段开头就找到他，光圈也一直跟着。"""
    frames = result["frames"]
    chain = {track}
    for forward in (False, True):
        cur = track
        while True:
            a, b = _span(frames, cur)
            if (b if forward else -a) >= (t_max if forward else -t_min):
                break
            cur = _next_piece(frames, cur, chain, forward)
            if cur is None:
                break
            chain.add(cur)
    return chain


def render_clip(game_dir: Path, video: Path, result: dict, event: dict, out_path: Path, people: dict, teams: dict,
                rims: list[dict]) -> None:
    game = json.loads((game_dir / "game.json").read_text())
    to_game = game_clock(video)
    shot, view, t0, t1, track = plan_for(game_dir, video, result, event, rims)
    renderer = Renderer(2880, view.get("lens_fov", 200))
    writer = VideoWriter(out_path, FPS)
    text = caption(event, people, teams)
    big, small = font(44), font(26)
    g0 = to_game(t0)
    for gt, fish in read_fisheye(game["sources"], view["lens"], g0, g0 + (t1 - t0), FPS):
        t = t0 + (gt - g0)
        yaw, pitch, hfov = shot.at(t)
        frame = renderer.render(fish, yaw, pitch, hfov)
        # 当事人脚下画圈
        m = mark_at(result, event, shot, view, t, track=track)
        if m:
            cv2.ellipse(frame, (int(m["foot"][0]), int(m["foot"][1])), (m["w"], max(5, m["w"] // 3)), 0, 0, 360, MARK, 4, cv2.LINE_AA)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, OUT_H - 92, OUT_W, OUT_H], fill=(0, 0, 0))
        draw.text((36, OUT_H - 78), text, font=big, fill=(255, 255, 255))
        minutes = int(gt // 60)
        draw.text((OUT_W - 220, OUT_H - 66), f"{minutes}:{gt % 60:04.1f}", font=small, fill=(200, 200, 200))
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    writer.close()


def ensure_clip(game_dir: Path, video: Path, result: dict, event: dict, people: dict, teams: dict, rims: list[dict],
                force: bool = False) -> Path:
    """渲染一个事件的片段到 clips/<事件号>.mp4（已有就跳过）。后台任务、网页的“现在渲染”、集锦可能同时渲染，
    各用自己的临时文件，渲染完再原子替换，不会互相写坏。"""
    clips = game_dir / "clips"
    clips.mkdir(exist_ok=True)
    target = clips / f"{event['id']}.mp4"
    sig_file = clips / f"{event['id']}.json"
    track = subject_tracks(result, event, load_ball(video))
    sig = signature(event, track)
    if not force and clip_fresh(clips, event, track):
        return target
    partial = clips / f"{event['id']}.{os.getpid()}-{threading.get_ident()}.partial.mp4"
    try:
        render_clip(game_dir, video, result, event, partial, people, teams, rims)
        partial.replace(target)
        sig_file.write_text(json.dumps(sig, ensure_ascii=False))
    finally:
        partial.unlink(missing_ok=True)
    return target


def signature(event: dict, track: set[int] | None = None) -> dict:
    """片段画的是什么：这些变了（你在审核里改了人、重新找事件、镜头做法升级）片段就要重新渲染。"""
    sig = {"v": RENDER_VERSION, **{k: event[k] for k in SIGNATURE_KEYS if event.get(k) is not None}}
    sig.pop("points", None)
    sig["kind"] = shot_kind(event)  # 新旧两种记法（kind / points）画出来一样，按归一后的比
    if track:
        sig["track"] = sorted(track)  # 认不出投篮人时跟的是哪条轨迹
    return sig


def clip_fresh(clips: Path, event: dict, track: set[int] | None = None) -> bool:
    """片段在，而且是按现在的内容渲染的（比较时不管空字段，签名里加减字段不会让旧片段全部重做）。"""
    sig = clips / f"{event['id']}.json"
    if not (clips / f"{event['id']}.mp4").exists() or not sig.exists():
        return False
    return {k: v for k, v in json.loads(sig.read_text()).items() if v is not None} == signature(event, track)


def with_review(events: list[dict], review: dict) -> list[dict]:
    """套上你在事件审核里的修改（删掉的不要）。"""
    edits = review.get("events", {})
    out = []
    for e in events:
        r = edits.get(str(e["id"]), {})
        if r.get("status") == "deleted":
            continue
        out.append({**e, **{k: v for k, v in r.items() if k in SIGNATURE_KEYS}})
    return out + [e for e in review.get("added", []) if e.get("status") != "deleted"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_context(video: Path) -> tuple[Path, dict, dict, dict, list]:
    out = out_dir_for(video)
    game_dir = video.parent
    final = out / "final.json"
    result = json.loads((final if final.exists() else out / "identified.json").read_text())
    people = result.get("people", {})
    people_file = game_dir / "enroll" / "review" / "people.json"
    teams = json.loads(people_file.read_text()).get("teams", {}) if people_file.exists() else {}
    events = json.loads((out / "events.json").read_text())
    return game_dir, result, people, teams, events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--event", type=int, default=None)
    args = parser.parse_args()
    game_dir, result, people, teams, data = load_context(args.video)
    clips = game_dir / "clips"
    clips.mkdir(exist_ok=True)
    for f in clips.glob("*.partial.mp4"):  # 上次被中途停掉留下的半成品（渲染它的进程已经不在了）
        if not _alive(int(f.name.split(".")[1].split("-")[0])):
            f.unlink(missing_ok=True)
    review_file = out_dir_for(args.video) / "review" / "events_review.json"
    review = json.loads(review_file.read_text()) if review_file.exists() else {}
    events = with_review(data["events"], review)

    events = [e for e in events if e["type"] == "shot"]
    todo = [e for e in events if e["id"] == args.event] if args.event is not None else [e for e in events if not clip_fresh(clips, e, subject_tracks(result, e, load_ball(args.video)))]
    for k, e in enumerate(todo):
        ensure_clip(game_dir, args.video, result, e, people, teams, data.get("rims", []), force=args.event is not None)
        print(f"  {k + 1}/{len(todo)} 段", flush=True)
    print(f"渲染了 {len(todo)} 个事件片段 → {clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
