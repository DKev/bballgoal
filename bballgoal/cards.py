"""第一遍之后：给每个候选人做一张“卡片”，供认人网页（bballgoal.review）确认。

卡片 = 第一遍聚出来的一个人（A#12、A?3…），里面按轨迹（一段连续跟踪）列出截图。
确认的最小单位是轨迹：一张卡里混进了别人，就把那条轨迹点掉或拖到别的卡上。

用法（先跑完 track、identify）：
    .venv/bin/python -m bballgoal.cards videos/比赛.mp4
输出 output/track_<视频名>/review/cards.json 和截图 review/crops/。
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.appearance import PoseEstimator
from bballgoal.avatar import face_score, head_photo, look
from bballgoal.frames import read_frames
from bballgoal.identify import player_sort_key

CROPS_PER_TRACK = 3  # 每条轨迹最多几张截图（时间上分散，挑框最大的）
MIN_TRACK_SECONDS = 0.5  # 短于这个的轨迹不单独给截图（并在卡片里显示为“零碎片段”）
TILE_HEIGHT = 220
CONTEXT_HEIGHT = 420


def pick_frames(items: list[tuple[int, float, list]], n: int) -> list[tuple[int, float, list]]:
    """把轨迹按时间平分成 n 段，每段取框最高的那一帧。"""
    if not items:
        return []
    chunks = np.array_split(np.arange(len(items)), min(n, len(items)))
    return [max((items[i] for i in c), key=lambda it: it[2][3] - it[2][1]) for c in chunks if len(c)]


MARK_COLOR = (0, 215, 255)  # 黄色框：要认的是这个人
DIM = 0.45  # 框外调暗到这个亮度


def crop(frame: np.ndarray, box: list, height: int, widen: float = 1.0, taller: float = 1.0, mark: bool = True) -> np.ndarray:
    """以人框为中心裁一块缩放到指定高度。mark：画黄框并把框外调暗，旁边有别人时一眼看出说的是谁。"""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * widen, (y2 - y1) * taller
    w = max(w, h * 0.5)
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    c, d = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    patch = frame[c:d, a:b]
    scale = height / max(1, patch.shape[0])
    out = cv2.resize(patch, (max(1, int(patch.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)
    if mark:
        bx1, by1 = int((x1 - a) * scale), int((y1 - c) * scale)
        bx2, by2 = int((x2 - a) * scale), int((y2 - c) * scale)
        inside = out[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)].copy()
        out = (out * DIM).astype(np.uint8)
        out[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = inside
        cv2.rectangle(out, (bx1, by1), (bx2, by2), MARK_COLOR, max(2, height // 110))
    return out


def card_key(team: str, player: str | None) -> str:
    if player:
        return player
    return {"ref": "裁判", "?": "未分队"}.get(team, f"{team}?")


def game_clock(video: Path):
    """抽段短视频（bballgoal.enroll）的时间 → 整场比赛的时间；普通视频原样返回。"""
    plan_file = video.with_suffix(".enroll.json")
    if not plan_file.exists():
        return lambda t: t
    segments = json.loads(plan_file.read_text())["segments"]

    def to_game(t: float) -> float:
        for seg in segments:
            if t < seg["end"] + 0.5:
                return seg["game_start"] + max(0.0, t - seg["start"])
        return t

    return to_game


def build_cards(result: dict, players: dict, video: Path, out_dir: Path) -> dict:
    fps = result["fps"]
    to_game = game_clock(video)
    tracks: dict[int, list[tuple[int, float, list]]] = defaultdict(list)
    team_of, player_of = {}, {}
    for i, f in enumerate(result["frames"]):
        for p in f["people"]:
            tracks[p["id"]].append((i, f["time"], p["box"]))
            team_of[p["id"]] = p["team"]
            player_of[p["id"]] = p.get("player")
    votes = {int(t): v for t, v in players.get("track_votes", {}).items()}

    wanted: dict[int, list[tuple[int, list]]] = defaultdict(list)  # 帧号 → [(轨迹, 框)]
    for tid, items in tracks.items():
        n = CROPS_PER_TRACK if len(items) / fps >= MIN_TRACK_SECONDS else 1
        for i, _t, box in pick_frames(items, n):
            wanted[i].append((tid, box))

    crops_dir = out_dir / "review" / "crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True)
    files: dict[int, list[dict]] = defaultdict(list)
    avatars: dict[int, tuple[float, np.ndarray]] = {}  # 轨迹 → (脸的清楚程度, 头肩截图)
    looks: dict[int, list[tuple[float, dict]]] = defaultdict(list)  # 轨迹 → [(脸的清楚程度, 肤色发色)]
    pose = PoseEstimator()
    info = result["info"]
    end = result["frames"][-1]["time"] + 0.5 / fps
    last = max(wanted) if wanted else -1
    for i, (t, frame) in enumerate(read_frames(video, info["width"], info["height"], result["start"], end, fps)):
        if i > last:
            break
        if i in wanted:
            for (tid, box), kp in zip(wanted[i], pose.keypoints(frame, [b for _t, b in wanted[i]])):
                score = face_score(kp, box)
                if tid not in avatars or score > avatars[tid][0]:
                    avatars[tid] = (score, head_photo(frame, kp, box))
                if (colors := look(frame, kp, box)) is not None:
                    looks[tid].append((score, colors))
        for tid, box in wanted.get(i, []):
            name = f"{tid}_{i}"
            cv2.imwrite(str(crops_dir / f"{name}.jpg"), crop(frame, box, TILE_HEIGHT, 1.6, 1.12), [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(crops_dir / f"{name}_ctx.jpg"), crop(frame, box, CONTEXT_HEIGHT, 4.0, 2.2), [cv2.IMWRITE_JPEG_QUALITY, 85])
            files[tid].append({"tile": f"crops/{name}.jpg", "context": f"crops/{name}_ctx.jpg", "time": round(to_game(t), 1)})

    avatar_dir = out_dir / "review" / "avatars"
    if avatar_dir.exists():
        shutil.rmtree(avatar_dir)
    avatar_dir.mkdir(parents=True)
    for tid, (_score, image) in avatars.items():
        cv2.imwrite(str(avatar_dir / f"{tid}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cards: dict[str, dict] = {}
    for tid, items in tracks.items():
        key = card_key(team_of[tid], player_of[tid])
        card = cards.setdefault(key, {"id": key, "team": team_of[tid], "tracks": []})
        v = votes.get(tid, {})
        card["tracks"].append({
            "id": tid,
            "start": round(to_game(items[0][1]), 1),
            "end": round(to_game(items[-1][1]), 1),
            "seconds": round(len(items) / fps, 1),
            "number": v.get("number"),
            "readings": v.get("readings", 0),
            "crops": sorted(files.get(tid, []), key=lambda c: c["time"]),
            "avatar": f"avatars/{tid}.jpg" if tid in avatars else None,
            "face": round(avatars[tid][0], 1) if tid in avatars else 0,
            "looks": [{"weight": round(w, 1), **c} for w, c in looks.get(tid, [])],
        })
    for card in cards.values():
        card["tracks"].sort(key=lambda t: -t["seconds"])
        card["seconds"] = round(sum(t["seconds"] for t in card["tracks"]), 1)
        numbers: dict[str, int] = defaultdict(int)
        for t in card["tracks"]:
            if t["number"]:
                numbers[t["number"]] += t["readings"]
        card["ocr"] = dict(sorted(numbers.items(), key=lambda kv: -kv[1]))

    def order(key: str) -> tuple:
        return (key in ("裁判", "未分队"), player_sort_key(key) if key[:1] in "AB" and len(key) > 1 else (key,))

    return {
        "video": str(video),
        "team_colors_bgr": result.get("team_colors_bgr", {}),
        "cards": [cards[k] for k in sorted(cards, key=order)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or out_dir_for(args.video)
    result = json.loads((out_dir / "identified.json").read_text())
    players = json.loads((out_dir / "players.json").read_text())
    data = build_cards(result, players, args.video, out_dir)
    (out_dir / "review" / "cards.json").write_text(json.dumps(data, ensure_ascii=False))
    n_tracks = sum(len(c["tracks"]) for c in data["cards"])
    print(f"{len(data['cards'])} 张卡片，{n_tracks} 条轨迹 → {out_dir / 'review'}")
    print(f"打开认人网页：.venv/bin/python -m bballgoal.review {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
