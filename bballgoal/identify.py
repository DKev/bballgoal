"""按球衣号码识别球员，把断开的轨迹合并成“一个人”。

先运行 bballgoal.track 生成 tracks.json，再运行：
    .venv/bin/python -m bballgoal.identify videos/比赛.mp4

步骤：
1. 轨迹前后外观明显是两个人（跟踪器串号）就切开；用整体外观重新判断每条轨迹的队伍；
2. 每条轨迹多帧识别号码；同一轨迹前后读到不同号码（串人）也切开；
3. 按外观（整体 + 鞋，见 bballgoal.identity）把同队轨迹聚成固定的人：
   同时出现的不合并，号码确定且不同的不合并；
4. 每个人用他所有轨迹的号码读数一起投票定号码；认不出号码的显示 A?1、A?2…；
5. 太小没采到外观的短轨迹，如果紧接在某人轨迹的前后、位置也接得上，就归给那个人。

没有 appearance.npz（旧版 track 输出）时退回只按号码合并。

输出在 output/track_<视频名>/：
    players.json  每个球员（队 + 号码）对应的轨迹和出场时间
    players.mp4   标注视频（--debug-video 才出），框上显示 A#73 这样的号码
    identified.json  tracks.json 加上切分后的轨迹号和每帧的球员（后续统计用）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.frames import read_frames
from bballgoal.identity import (
    MIN_COOCCUR,
    cluster_team,
    fit_projection,
    refine_teams,
    split_tracks_by_appearance,
    track_vectors,
)
from bballgoal.jersey import number_crop, read_numbers, vote
from bballgoal.segments import split_by_readings
from bballgoal.teams import REFEREE, UNKNOWN
from bballgoal.track import render

OCR_EVERY = 2  # 每隔几帧识别一次号码
STITCH_GAP = 2.0  # 秒：轨迹断开后多久内重新出现还算同一人
STITCH_DISTANCE = 1.5  # 倍人身高：断开处和重新出现处的距离上限


def collect_readings(result: dict, video: Path) -> dict[int, list[tuple[int, str, float]]]:
    """每条轨迹的号码读数：(帧序号, 号码, 置信度)。"""
    info = result["info"]
    frames = result["frames"]
    end = frames[-1]["time"] + 0.5 / result["fps"]
    readings: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    started = time.perf_counter()
    reader = read_frames(video, info["width"], info["height"], result["start"], end, result["fps"])
    for i, (entry, (_t, frame)) in enumerate(zip(frames, reader)):
        for p in entry["people"]:
            # 每隔 OCR_EVERY 帧识别一次；有关键点躯干框的帧（裁得更准）也识别
            if p["team"] in (REFEREE, UNKNOWN) or (i % OCR_EVERY and "torso" not in p):
                continue
            crop = number_crop(frame, p["box"], p.get("torso"))
            if crop is not None:
                readings[p["id"]] += [(i, number, conf) for number, conf in read_numbers(crop)]
        if i and i % 200 == 0:
            rate = i / (time.perf_counter() - started)
            print(f"  {i}/{len(frames)} 帧  剩余约 {(len(frames) - i) / rate / 60:.1f} 分钟", flush=True)
    return readings


def split_tracks_by_number(
    result: dict, readings: dict[int, list[tuple[int, str, float]]]
) -> tuple[dict[int, list[tuple[str, float]]], int]:
    """同一条轨迹前后读到不同号码（同队串人），在中间切开。原地改写 result["frames"]。

    返回按新轨迹号整理的读数 和 切开的处数。
    """
    occurrences: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for i, f in enumerate(result["frames"]):
        for p in f["people"]:
            occurrences[p["id"]].append((i, p))
    next_id = max(occurrences, default=0) + 1
    new_readings: dict[int, list[tuple[str, float]]] = defaultdict(list)
    splits = 0
    for tid, items in occurrences.items():
        track_readings = sorted(readings.get(tid, []))
        cut_frames = split_by_readings([(i, number) for i, number, _c in track_readings])
        bounds = [-1, *cut_frames, float("inf")]
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            new_id = tid if k == 0 else next_id
            if k:
                next_id += 1
                splits += 1
            for i, p in items:
                if a <= i < b:
                    p["id"] = new_id
            new_readings[new_id] = [(n, c) for i, n, c in track_readings if a <= i < b]
    return new_readings, splits


def track_spans(result: dict) -> dict[int, dict]:
    """每条轨迹的起止时间、首尾位置（脚底点）和队伍。"""
    spans: dict[int, dict] = {}
    for i, f in enumerate(result["frames"]):
        for p in f["people"]:
            x1, y1, x2, y2 = p["box"]
            point = ((x1 + x2) / 2, y2, y2 - y1)
            span = spans.setdefault(
                p["id"], {"team": p["team"], "start": f["time"], "first": point, "frames": 0, "seen": set()}
            )
            span["end"], span["last"] = f["time"], point
            span["frames"] += 1
            span["seen"].add(i)
    return spans


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def assign_players(
    spans: dict[int, dict], readings: dict[int, list[tuple[str, float]]]
) -> tuple[dict[int, str], dict[int, tuple[str, float]]]:
    """返回 轨迹→球员键（如 "A#73"）以及每条轨迹的投票结果。"""
    votes = {tid: vote(r) for tid, r in readings.items()}
    labels: dict[int, str] = {}
    # 票数高的先占号码；与同号已有轨迹时间重叠的，说明读错了或是跟踪串号，放弃
    ranked = sorted(
        ((tid, number, share) for tid, (number, share) in votes.items() if number),
        key=lambda item: (-len(readings[item[0]]) * item[2]),
    )
    for tid, number, _share in ranked:
        key = f"{spans[tid]['team']}#{number}"
        if any(labels.get(other) == key and overlaps(spans[tid], spans[other]) for other in labels):
            continue
        labels[tid] = key

    stitch(spans, labels)
    return labels, votes


def stitch(spans: dict[int, dict], labels: dict[int, str]) -> None:
    """没归属的轨迹接到前后紧挨着的同队已知轨迹上，反复做直到没有新变化。原地改写 labels。"""
    changed = True
    while changed:
        changed = False
        for tid, span in spans.items():
            if tid in labels or span["team"] in (REFEREE, UNKNOWN):
                continue
            best_key, best_gap = None, STITCH_GAP
            for other, key in labels.items():
                o = spans[other]
                if o["team"] != span["team"]:
                    continue
                if 0 <= span["start"] - o["end"] < best_gap:
                    a, b, gap = o["last"], span["first"], span["start"] - o["end"]
                elif 0 <= o["start"] - span["end"] < best_gap:
                    a, b, gap = span["last"], o["first"], o["start"] - span["end"]
                else:
                    continue
                height = max(a[2], b[2])
                if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > STITCH_DISTANCE * height:
                    continue
                same_key = [t for t, k in labels.items() if k == key]
                if any(overlaps(span, spans[t]) for t in same_key):
                    continue
                best_key, best_gap = key, gap
            if best_key:
                labels[tid] = best_key
                changed = True


def load_track_samples(result: dict, samples_file: Path) -> dict[str, dict[int, np.ndarray]]:
    """按（切分后的）轨迹号整理外观样本。"""
    with np.load(samples_file) as z:  # npz 每次取键都会重新从磁盘读整个数组，先读一次
        data = {k: z[k] for k in z.files}
    per_track: dict[str, dict[int, list[int]]] = {"body": defaultdict(list), "shoes": defaultdict(list)}
    for f in result["frames"]:
        for p in f["people"]:
            row = p.get("sample")
            if row is None or p["team"] in (REFEREE, UNKNOWN):
                continue
            per_track["body"][p["id"]].append(row)
            if data["has_shoes"][row]:
                per_track["shoes"][p["id"]].append(row)
    return {
        name: {tid: data[name][rows].astype(np.float32) for tid, rows in tracks.items()}
        for name, tracks in per_track.items()
    }


def assign_by_appearance(
    spans: dict[int, dict],
    readings: dict[int, list[tuple[str, float]]],
    samples: dict[str, dict[int, np.ndarray]],
) -> tuple[dict[int, str], dict[int, tuple[str, float]]]:
    """按外观聚成固定的人，再按人投票定号码。返回 轨迹→球员键 以及每条轨迹的投票结果。"""
    votes = {tid: vote(r) for tid, r in readings.items()}
    # 只用来阻止合并的号码：读到一次就算（读错的代价只是一个人被拆成两段，不会把两人混在一起）
    numbers = {tid: n for tid, r in readings.items() if (n := vote(r, min_votes=1)[0])}
    vectors = track_vectors(samples, {name: fit_projection(per_track) for name, per_track in samples.items()})
    weights = {tid: float(len(samples["body"].get(tid, ()))) for tid in spans}

    people: list[tuple[str, list[int]]] = []
    for team in ("A", "B"):
        tids = [t for t, s in spans.items() if s["team"] == team]
        people += [(team, members) for members in cluster_team(tids, vectors, spans, numbers, weights)]

    # 每个人汇总所有轨迹的读数投票
    pooled_numbers = []
    for team, members in people:
        pooled = [r for t in members for r in readings.get(t, [])]
        pooled_numbers.append(vote(pooled, min_votes=2, min_share=0.5)[0])
    # 投出同一号码、又从没同时出现过的两个人，其实是同一个人（外观差异大，例如正面 / 背面），合并
    for a in range(len(people)):
        for b in range(a + 1, len(people)):
            if not pooled_numbers[a] or pooled_numbers[a] != pooled_numbers[b] or people[a][0] != people[b][0]:
                continue
            seen_a = set().union(*(spans[t]["seen"] for t in people[a][1]))
            seen_b = set().union(*(spans[t]["seen"] for t in people[b][1]))
            if len(seen_a & seen_b) < MIN_COOCCUR:
                people[a][1].extend(people[b][1])
                people[b] = (people[b][0], [])
    people = [(team, members) for team, members in people if members]

    # 同队两人投出同一个号码时（确实同时出现过），读数多的那个拿号码
    claims = []
    for k, (team, members) in enumerate(people):
        pooled = [r for t in members for r in readings.get(t, [])]
        number, share = vote(pooled, min_votes=2, min_share=0.5)
        if number:
            claims.append((sum(1 for n, _c in pooled if n == number) * share, k, f"{team}#{number}"))
    keys: dict[int, str] = {}
    for _strength, k, key in sorted(claims, reverse=True):
        if key not in keys.values():
            keys[k] = key
    # 认不出号码的人按出场时间从长到短编 A?1、A?2…
    seconds = {k: sum(spans[t]["frames"] for t in members) for k, (_team, members) in enumerate(people)}
    counters: dict[str, int] = defaultdict(int)
    for k in sorted(range(len(people)), key=lambda k: -seconds[k]):
        if k not in keys:
            team = people[k][0]
            counters[team] += 1
            keys[k] = f"{team}?{counters[team]}"

    labels = {t: keys[k] for k, (_team, members) in enumerate(people) for t in members}
    stitch(spans, labels)
    return labels, votes


HINT_WEIGHT = 20  # 一条人工指定相当于多少次读数（远多于 OCR，一定采信）


def apply_hints(result: dict, readings: dict[int, list[tuple[str, float]]], hints: list[dict]) -> int:
    """人工指定：{"time": 秒, "x", "y": 画面上这个人身上任一点, "player": "A#18"}。
    OCR 读不出的号码（数字细、总是侧身）用这个补。按位置找到那一刻的轨迹，
    给它加上大量该号码的读数并改成指定的队，后面外观聚类会把同一人的其他轨迹并进来。
    用时间和位置而不用轨迹号，重新跟踪后轨迹号会变。返回用上了几条。"""
    used = 0
    for h in hints:
        entry = min(result["frames"], key=lambda f: abs(f["time"] - h["time"]))
        inside = [p for p in entry["people"] if p["box"][0] <= h["x"] <= p["box"][2] and p["box"][1] <= h["y"] <= p["box"][3]]
        if not inside:
            print(f"  人工指定 {h['player']} @ {h['time']}s：那里没有人，跳过")
            continue
        target = min(inside, key=lambda p: abs((p["box"][0] + p["box"][2]) / 2 - h["x"]))
        team, number = h["player"].split("#")
        readings[target["id"]] = readings.get(target["id"], []) + [(number, 1.0)] * HINT_WEIGHT
        for f in result["frames"]:
            for p in f["people"]:
                if p["id"] == target["id"]:
                    p["team"] = team
        used += 1
    return used


def player_sort_key(key: str) -> tuple:
    """A#3 < A#12 < A?1 < B#5 …"""
    team, sep, rest = key[0], key[1], key[2:]
    return (team, sep != "#", int(rest) if rest.isdigit() else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--debug-video", action="store_true", help="另外渲染标注视频 players.mp4（检查认号码用；一场几个 GB，默认不出）")
    args = parser.parse_args()

    out_dir = args.out or out_dir_for(args.video)
    tracks_file = out_dir / "tracks.json"
    if not tracks_file.exists():
        print(f"缺少 {tracks_file}，先运行：.venv/bin/python -m bballgoal.track {args.video}", file=sys.stderr)
        return 1
    result = json.loads(tracks_file.read_text())

    samples_file = out_dir / "appearance.npz"
    if samples_file.exists():
        body = np.load(samples_file)["body"]
        splits = split_tracks_by_appearance(result["frames"], body)
        print(f"按外观前后变化切开了 {splits} 处串人")
        changed = refine_teams(result["frames"], body)
        print(f"按整体外观修正了 {changed} 条轨迹的队伍")

    print("识别球衣号码…")
    raw_readings = collect_readings(result, args.video)
    readings, splits = split_tracks_by_number(result, raw_readings)
    print(f"按号码变化切开了 {splits} 处串人")
    hints_file = args.video.with_suffix(".labels.json")
    if hints_file.exists():
        print(f"人工指定：用上 {apply_hints(result, readings, json.loads(hints_file.read_text()))} 条（{hints_file}）")
    spans = track_spans(result)
    if samples_file.exists():
        print("按外观固定每个人…")
        labels, votes = assign_by_appearance(spans, readings, load_track_samples(result, samples_file))
    else:
        print(f"没有 {samples_file}，只按号码合并（重新运行 bballgoal.track 可启用外观识别）")
        labels, votes = assign_players(spans, readings)

    players: dict[str, dict] = defaultdict(lambda: {"tracks": [], "seconds": 0.0})
    for tid, key in labels.items():
        players[key]["tracks"].append(tid)
        players[key]["seconds"] += spans[tid]["frames"] / result["fps"]
    players_out = {
        key: {"tracks": sorted(v["tracks"]), "seconds": round(v["seconds"], 1)}
        for key, v in sorted(players.items(), key=lambda kv: player_sort_key(kv[0]))
    }

    numbered = [t for t, k in labels.items() if "#" in k]
    player_frames = sum(spans[t]["frames"] for t in numbered)
    total_frames = sum(s["frames"] for s in spans.values() if s["team"] not in (REFEREE, UNKNOWN))
    print(
        f"\n固定为 {len(players_out)} 人，其中 {sum('#' in k for k in players_out)} 人认出号码；"
        f"球员画面中 {player_frames / max(1, total_frames):.0%} 已对应到号码"
    )
    for key, v in players_out.items():
        print(f"  {key:6s}  出现 {v['seconds']:6.1f} 秒  （{len(v['tracks'])} 段轨迹）")

    (out_dir / "players.json").write_text(
        json.dumps(
            {
                "players": players_out,
                "track_labels": {str(t): k for t, k in labels.items()},
                "track_votes": {str(t): {"number": n, "share": round(s, 2), "readings": len(readings[t])} for t, (n, s) in votes.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    display = {}
    for tid, span in spans.items():
        if tid in labels:
            display[tid] = labels[tid]
        else:
            display[tid] = {"A": "A?", "B": "B?", REFEREE: "ref"}.get(span["team"], "?")
    for f in result["frames"]:
        for p in f["people"]:
            p["player"] = labels.get(p["id"])
    (out_dir / "identified.json").write_text(json.dumps(result, ensure_ascii=False))

    if args.debug_video:
        print("生成标注视频…")
        render(result, args.video, out_dir / "players.mp4", labels=display)
        print(f"输出：{out_dir}/players.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
