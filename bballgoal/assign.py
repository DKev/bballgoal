"""第二遍：拿认人网页确认过的名单（特征库），去认整场的每一段轨迹。

1. 特征库：确认过的每个人的外观样本（OSNet 整体 + 鞋）。这次知道谁是谁，直接学“区分这些人”的投影
   （LDA，类 = 人），比第一遍无监督地猜准得多；
2. 每段轨迹和库里每个人比相似度（各特征投影后的余弦相似度取平均），号码读数是强证据：
   读到的号码和某人对得上加分，对不上减分；
3. 拿不准的放进问题队列，交给你：
   - 和谁都不像（可能是名单里没有的新的人）；
   - 第一名和第二名太接近；
   - 同一时刻两段轨迹被认成同一个人（其中把握小的那段）。
   门槛不写死：在特征库里做“留一段”自测，按正确匹配的分布定。

你回答过的问题（review/answers.json）：直接按答案定；答过的段落加进特征库（同一个孩子不同时间、
不同角度的样子都学进去），再把没回答的重新认一遍，所以每答一轮，剩下要问的会更少。
回答“新的人”的段落各自成为一个新的人（“新1”“新2”…），也进特征库，后面相似的段落会被认成他。

用法（全场先跑完 track，抽段视频已在认人网页确认）：
    .venv/bin/python -m bballgoal.assign videos/比赛.mp4 --gallery videos/比赛_enroll.mp4
输出 assign.json（每段轨迹认成谁、把握多大）、review/questions.json（还要问的），
以及 final.json（全场每个框标上是谁，给事件和集锦用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.identity import FEATURES, Projection, fit_projection

NUMBER_BONUS = 0.3  # 读到的号码和这个人对得上
NUMBER_PENALTY = 0.3  # 读到了号码但和这个人对不上
TEAM_PENALTY = 0.15  # 颜色判断的队和这个人的队不一样（颜色会被背景带偏，所以只是轻罚）
MIN_SAMPLES = 2
MIN_QUESTION_SECONDS = 1.0  # 短于这个的轨迹不问，直接按最像的（或不标）
MIN_COOCCUR = 3
STITCH_GAP = 0.5  # 秒：一段结束后这么久内、在旁边冒出来的新一段，可能是跟踪器断开的同一个人
STITCH_REACH = 0.3  # 接力时允许的位移：(这个数 + 每秒 1.5) × 身高
STITCH_CLEAR = 0.3  # 最近的候选要比第二近的（另一个人）近这么多倍才算数


def track_samples(result: dict, data: dict) -> dict[str, dict[int, np.ndarray]]:
    """每条轨迹的外观样本（不按队过滤：队可能判断错了，交给特征库来认）。"""
    data = {k: np.asarray(data[k]) for k in ("body", "shoes", "has_shoes")}  # npz 每次取键都会重新从磁盘读整个数组，先读一次
    rows: dict[str, dict[int, list[int]]] = {"body": defaultdict(list), "shoes": defaultdict(list)}
    for f in result["frames"]:
        for p in f["people"]:
            r = p.get("sample")
            if r is None:
                continue
            rows["body"][p["id"]].append(r)
            if data["has_shoes"][r]:
                rows["shoes"][p["id"]].append(r)
    return {name: {t: data[name][r].astype(np.float32) for t, r in per.items()} for name, per in rows.items()}


def track_info(result: dict) -> dict[int, dict]:
    info: dict[int, dict] = {}
    for i, f in enumerate(result["frames"]):
        for p in f["people"]:
            t = info.setdefault(p["id"], {"team": p["team"], "start": f["time"], "seen": set(), "frames": 0, "first_box": p["box"]})
            t["end"] = f["time"]
            t["last_box"] = p["box"]
            t["frames"] += 1
            t["seen"].add(i)
    return info


@dataclass
class Gallery:
    people: dict[str, dict]  # 人 id → {team, number, name, kind}
    projections: dict[str, Projection | None]
    centroids: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # 特征 → 人 → 单位向量
    samples: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # 特征 → 人 → 投影后样本（留一自测用）


def build_gallery(people: list[dict], samples: dict[str, dict[int, np.ndarray]]) -> Gallery:
    """people：认人网页确认的人（含 tracks）。只收球员和教练、裁判（都可能出现在场上）。"""
    members = {p["id"]: p for p in people if p.get("tracks") and p.get("kind", "player") != "ignore"}
    per_person = {
        name: {pid: np.concatenate([samples[name][t] for t in p["tracks"] if t in samples[name]])
               for pid, p in members.items() if any(t in samples[name] for t in p["tracks"])}
        for name in FEATURES
    }
    projections = {name: fit_projection(per_person[name], dims=32, min_per_track=3) for name in FEATURES}
    gallery = Gallery(people=members, projections=projections)
    for name in FEATURES:
        proj = projections[name]
        if proj is None:
            continue
        gallery.samples[name] = {pid: proj(x) for pid, x in per_person[name].items()}
        gallery.centroids[name] = {pid: _unit(z.mean(axis=0)) for pid, z in gallery.samples[name].items()}
    return gallery


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(1e-9, float(np.linalg.norm(v)))


def similarities(gallery: Gallery, track: dict[str, np.ndarray], number: str | None, team: str | None,
                 exclude: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, float]:
    """这段轨迹和库里每个人的相似度。exclude：从某人的质心里扣掉这些样本（留一自测）。"""
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for name, cents in gallery.centroids.items():
        x = track.get(name)
        if x is None or len(x) < MIN_SAMPLES:
            continue
        z = _unit(gallery.projections[name](x).mean(axis=0))
        for pid, c in cents.items():
            if exclude and pid in exclude.get(name, {}):
                rest = gallery.samples[name][pid]
                drop = exclude[name][pid]
                if len(rest) - len(drop) < MIN_SAMPLES:
                    continue
                c = _unit(rest.sum(axis=0) - drop.sum(axis=0))
            total[pid] += float(z @ c)
            count[pid] += 1
    scores = {pid: total[pid] / count[pid] for pid in total}
    for pid, s in scores.items():
        person = gallery.people[pid]
        if number and person.get("number"):
            scores[pid] = s + (NUMBER_BONUS if person["number"] == number else -NUMBER_PENALTY)
        if team in ("A", "B") and person.get("team") in ("A", "B") and person["team"] != team:
            scores[pid] -= TEAM_PENALTY
    return scores


def calibrate(gallery: Gallery, samples: dict[str, dict[int, np.ndarray]]) -> dict[str, float]:
    """留一段自测：每段确认过的轨迹，从它主人的质心里扣掉它自己，看能不能认回来。
    用正确匹配的相似度和领先幅度的低分位定门槛。"""
    correct, margins, right = [], [], 0
    tested = 0
    todo = [(pid, t) for pid, person in gallery.people.items() for t in person["tracks"]]
    for k, (pid, t) in enumerate(todo):
        if k % 50 == 0:
            print(f"  自测特征库 {k}/{len(todo)} 段", flush=True)
        track = {name: samples[name][t] for name in FEATURES if t in samples[name]}
        if len(track.get("body", ())) < MIN_SAMPLES:
            continue
        exclude = {name: {pid: gallery.projections[name](track[name])} for name in gallery.centroids if name in track}
        scores = similarities(gallery, track, None, None, exclude)
        if pid not in scores or len(scores) < 2:
            continue
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        tested += 1
        right += ranked[0][0] == pid
        correct.append(scores[pid])
        margins.append(scores[pid] - max(s for q, s in scores.items() if q != pid))
    if tested < 5:
        return {"unknown": 0.3, "margin": 0.1, "tested": tested, "accuracy": None}
    return {
        "unknown": float(np.percentile(correct, 5)),  # 比 95% 的正确匹配还不像：多半是新的人
        "margin": float(max(0.02, np.percentile(margins, 10))),
        "tested": tested,
        "accuracy": right / tested,
    }


def assign(result: dict, samples: dict[str, dict[int, np.ndarray]], numbers: dict[int, str], gallery: Gallery,
           thresholds: dict[str, float], fps: float) -> dict[int, dict]:
    info = track_info(result)
    out: dict[int, dict] = {}
    for k, (tid, t) in enumerate(info.items()):
        if k % 200 == 0:
            print(f"  认每一段 {k}/{len(info)} 段", flush=True)
        track = {name: samples[name][tid] for name in FEATURES if tid in samples[name]}
        scores = similarities(gallery, track, numbers.get(tid), t["team"]) if track else {}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        best = ranked[0] if ranked else (None, -1.0)
        margin = best[1] - ranked[1][1] if len(ranked) > 1 else 1.0
        reason = None
        if not ranked:
            reason = "no_features"
        elif best[1] < thresholds["unknown"]:
            reason = "unknown"
        elif margin < thresholds["margin"]:
            reason = "close"
        out[tid] = {
            "person": best[0] if reason is None else None,
            "guess": [{"person": pid, "score": round(s, 3)} for pid, s in ranked],
            "reason": reason,
            "seconds": round(t["frames"] / fps, 1),
            "team": t["team"],
            "start": t["start"],
            "end": t["end"],
        }
    # 同一时刻两段被认成同一个人：把握小的那段改为待问
    by_person: dict[str, list[int]] = defaultdict(list)
    for tid, a in out.items():
        if a["person"]:
            by_person[a["person"]].append(tid)
    for pid, tids in by_person.items():
        if gallery.people[pid].get("kind") in ("ref", "coach"):  # 裁判、教练可能有好几个，同时出现不算冲突
            continue
        tids.sort(key=lambda t: -out[t]["guess"][0]["score"])
        kept: list[int] = []
        for tid in tids:
            if any(len(info[tid]["seen"] & info[k]["seen"]) >= MIN_COOCCUR for k in kept):
                out[tid]["person"] = None
                out[tid]["reason"] = "conflict"
            else:
                kept.append(tid)
    return out


def _foot(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, box[3]])


def stitch(info: dict[int, dict], out: dict[int, dict], people: dict[str, dict], max_passes: int = 10) -> int:
    """按时间接力：认不出（没外观、撞人、不像谁）的一段，如果前面紧挨着刚结束一段、或者后面紧挨着接上一段，
    位置也对得上，而且那个人这段时间不在别处，就认成同一个人。反复做，让认出来的往两头延伸。返回接上的段数。"""
    linked = 0
    for _ in range(max_passes):
        mine: dict[str, list[int]] = defaultdict(list)
        for tid, a in out.items():
            if a["person"]:
                mine[a["person"]].append(tid)
        ends = sorted((info[t]["end"], t) for ts in mine.values() for t in ts)
        starts = sorted((info[t]["start"], t) for ts in mine.values() for t in ts)
        end_times = [e for e, _ in ends]
        start_times = [s for s, _ in starts]
        changed = 0
        for tid, a in out.items():
            if a["person"] or "answered" in a:
                continue
            t = info[tid]
            allowed = None if a["reason"] == "no_features" else {g["person"] for g in a["guess"]}
            votes = []
            for side in ("before", "after"):
                if side == "before":  # 前一段在我开始前不久结束
                    lo, hi = np.searchsorted(end_times, [t["start"] - STITCH_GAP, t["start"] + 0.2])
                    cands = [(ends[k][1], t["start"] - info[ends[k][1]]["end"], info[ends[k][1]]["last_box"], t["first_box"]) for k in range(lo, hi)]
                else:  # 后一段在我结束后不久开始
                    lo, hi = np.searchsorted(start_times, [t["end"] - 0.2, t["end"] + STITCH_GAP])
                    cands = [(starts[k][1], info[starts[k][1]]["start"] - t["end"], info[starts[k][1]]["first_box"], t["last_box"]) for k in range(lo, hi)]
                height = t["first_box"][3] - t["first_box"][1]
                scored = []
                for other, gap, b0, b1 in cands:
                    if other == tid:
                        continue
                    d = float(np.linalg.norm(_foot(b0) - _foot(b1))) / max(1.0, height)
                    if d <= STITCH_REACH + 1.5 * max(0.0, gap):
                        scored.append((d, out[other]["person"]))
                scored.sort()
                if not scored:
                    continue
                best = scored[0]
                rival = next((d for d, p in scored[1:] if p != best[1]), None)
                if rival is None or best[0] <= STITCH_CLEAR * rival:
                    votes.append(best[1])
            if not votes or len(set(votes)) > 1:
                continue
            pid = votes[0]
            p = people.get(pid, {})
            if allowed is not None and pid not in allowed:
                continue
            if t["team"] in ("A", "B") and p.get("kind", "player") == "player" and p.get("team") in ("A", "B") and p["team"] != t["team"]:
                continue
            if p.get("kind") not in ("ref", "coach") and any(len(t["seen"] & info[k]["seen"]) >= MIN_COOCCUR for k in mine[pid]):
                continue  # 那个人这段时间在别处
            a["person"], a["reason"], a["via"] = pid, None, "stitch"
            mine[pid].append(tid)
            changed += 1
        linked += changed
        if not changed:
            break
    return linked


def merge_samples(gal_people: list[dict], gal_samples: dict, full_samples: dict, answers: dict[str, str],
                  full_team: dict[int, str]) -> tuple[list[dict], dict]:
    """抽段视频的确认 + 全场里回答过的段落，合成一个特征库。两边的轨迹号会重复，所以加前缀区分。"""
    samples = {name: {} for name in FEATURES}
    for name in FEATURES:
        samples[name].update({f"e{t}": x for t, x in gal_samples[name].items()})
        samples[name].update({f"f{t}": x for t, x in full_samples[name].items()})
    people = [{**p, "tracks": [f"e{t}" for t in p.get("tracks", [])]} for p in gal_people]
    by_id = {p["id"]: p for p in people}
    new_count = 0
    for tid, ans in sorted(answers.items(), key=lambda kv: int(kv[0])):
        if ans in by_id:
            by_id[ans]["tracks"].append(f"f{tid}")
        elif ans == "new":
            new_count += 1
            person = {"id": f"新{new_count}", "kind": "player", "team": full_team.get(int(tid), ""), "number": "",
                      "name": "", "confirmed": True, "tracks": [], "from_answers": []}
            people.append(person)
            by_id[person["id"]] = person
            person["tracks"].append(f"f{tid}")
            person["from_answers"].append(int(tid))
        elif ans.startswith("new:"):  # 名单里没有、但你告诉了队和号码：同队同号的答案归成同一个人
            _, team, number = (ans.split(":") + ["", ""])[:3]
            person = next((p for p in people if p.get("kind", "player") == "player" and p.get("team") == team
                           and p.get("number") == number and number), None)
            if person is None:
                pid = f"{team}#{number}" if f"{team}#{number}" not in by_id else f"新{team}#{number}"
                person = {"id": pid, "kind": "player", "team": team, "number": number, "name": "", "confirmed": True,
                          "tracks": [], "from_answers": []}
                people.append(person)
                by_id[pid] = person
            person["tracks"].append(f"f{tid}")
            person.setdefault("from_answers", []).append(int(tid))
    return people, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--gallery", type=Path, required=True, help="已在认人网页确认过的视频（通常是抽段视频）")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or out_dir_for(args.video)
    gal_dir = out_dir_for(args.gallery)
    people_file = gal_dir / "review" / "people.json"
    if not people_file.exists():
        print(f"缺少 {people_file}：先在认人网页确认抽段视频里的人", file=sys.stderr)
        return 1
    print("  读取抽段的名单和特征 0/4 段", flush=True)
    gal_result = json.loads((gal_dir / "identified.json").read_text())
    gal_samples = track_samples(gal_result, np.load(gal_dir / "appearance.npz"))
    gal_people = json.loads(people_file.read_text())["people"]

    print("  读取整场的跟踪结果 1/4 段", flush=True)
    result_file = out_dir / ("identified.json" if (out_dir / "identified.json").exists() else "tracks.json")
    result = json.loads(result_file.read_text())
    samples = track_samples(result, np.load(out_dir / "appearance.npz"))
    answers_file = out_dir / "review" / "answers.json"
    answers = json.loads(answers_file.read_text()).get("answers", {}) if answers_file.exists() else {}
    full_team = {t: i["team"] for t, i in track_info(result).items()}
    print("  读取整场的外观特征 2/4 段", flush=True)
    people, combined = merge_samples(gal_people, gal_samples, samples, answers, full_team)
    print("  建特征库 3/4 段", flush=True)
    gallery = build_gallery(people, combined)
    thresholds = calibrate(gallery, combined)
    acc = thresholds["accuracy"]
    print(f"特征库：{len(gallery.people)} 人（含你回答的 {len(answers)} 段）；自测 {thresholds['tested']} 段，认对 {acc:.0%}"
          if acc is not None else "特征库样本太少，用默认门槛")
    players_file = out_dir / "players.json"
    numbers = {}
    if players_file.exists():
        numbers = {int(t): v["number"] for t, v in json.loads(players_file.read_text())["track_votes"].items() if v["number"]}
    assignments = assign(result, samples, numbers, gallery, thresholds, result["fps"])
    # 回答过的：按答案定（跳过的、无关的人都不再问）
    new_of = {tid: p["id"] for p in people for tid in p.get("from_answers", [])}
    for tid_s, ans in answers.items():
        a = assignments.get(int(tid_s))
        if a is None:
            continue
        a["answered"] = ans
        a["person"] = new_of.get(int(tid_s)) if ans.startswith("new") else (ans if ans in gallery.people else None)
        a["reason"] = None if a["person"] else "answered"
    linked = stitch(track_info(result), assignments, gallery.people)
    print(f"按时间接力又认出 {linked} 段", flush=True)

    total = sum(a["seconds"] for a in assignments.values())
    sure = sum(a["seconds"] for a in assignments.values() if a["person"])
    questions = sorted((t for t, a in assignments.items()
                        if not a["person"] and "answered" not in a and a["seconds"] >= MIN_QUESTION_SECONDS),
                       key=lambda t: -assignments[t]["seconds"])
    (out_dir / "assign.json").write_text(json.dumps(
        {"thresholds": thresholds, "tracks": {str(t): a for t, a in assignments.items()}}, ensure_ascii=False))
    (out_dir / "review").mkdir(exist_ok=True)
    (out_dir / "review" / "questions.json").write_text(json.dumps(
        {"questions": [{"track": t, **assignments[t]} for t in questions],
         "coverage": round(sure / max(1, total), 4),
         "people": {pid: {k: p.get(k) for k in ("team", "number", "name", "kind")} for pid, p in gallery.people.items()}},
        ensure_ascii=False))
    print("  写入结果 0/1 段", flush=True)
    # 全场每个框标上是谁（事件、集锦用）
    for f in result["frames"]:
        for p in f["people"]:
            p["person"] = assignments.get(p["id"], {}).get("person")
    (out_dir / "final.json").write_text(json.dumps(
        {**result, "people": {pid: {k: p.get(k) for k in ("team", "number", "name", "kind")} for pid, p in gallery.people.items()}},
        ensure_ascii=False))
    print(f"直接认出：{sure / max(1, total):.0%} 的出场时间；还需要问你：{len(questions)} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
