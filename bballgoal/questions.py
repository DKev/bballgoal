"""给问题队列（bballgoal.assign 的 review/questions.json）配截图和名单头像，供网页第 ③ 步使用。

用法：
    .venv/bin/python -m bballgoal.questions videos/比赛.mp4 --gallery videos/比赛_enroll.mp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.cards import TILE_HEIGHT, crop, pick_frames

CROPS = 2


def person_looks(gallery_review: Path) -> dict[str, dict]:
    """名单里每个人的肤色、发色：取他各段轨迹里脸最清楚的几次。"""
    cards = json.loads((gallery_review / "cards.json").read_text())
    people = json.loads((gallery_review / "people.json").read_text())["people"]
    looks = {t["id"]: t.get("looks", []) for c in cards["cards"] for t in c["tracks"]}
    out = {}
    for p in people:
        items = sorted((lk for t in p["tracks"] for lk in looks.get(t, [])), key=lambda lk: -lk["weight"])[:5]
        if items:
            out[p["id"]] = items
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or out_dir_for(args.video)
    review = out_dir / "review"
    data = json.loads((review / "questions.json").read_text())
    result_file = out_dir / ("identified.json" if (out_dir / "identified.json").exists() else "tracks.json")
    result = json.loads(result_file.read_text())
    fps = result["fps"]

    wanted_tracks = {q["track"] for q in data["questions"]}
    items: dict[int, list] = defaultdict(list)
    for i, f in enumerate(result["frames"]):
        for p in f["people"]:
            if p["id"] in wanted_tracks:
                items[p["id"]].append((i, f["time"], p["box"]))
    wanted: dict[int, list] = defaultdict(list)
    for tid, its in items.items():
        for i, _t, box in pick_frames(its, CROPS):
            wanted[i].append((tid, box))

    crops_dir = review / "qcrops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    files: dict[int, list[dict]] = defaultdict(list)
    frame_times = [f["time"] for f in result["frames"]]
    todo: dict[int, list] = defaultdict(list)  # 上一轮已经截过的直接用：每答一轮重新认一遍时只截新的
    for i, entries in wanted.items():
        for tid, box in entries:
            name = f"{tid}_{i}"
            files[tid].append({"tile": f"qcrops/{name}.jpg", "context": f"qcrops/{name}_ctx.jpg", "time": frame_times[i]})
            if not (crops_dir / f"{name}_ctx.jpg").exists():
                todo[i].append((tid, box, name))
    if todo:
        cap = cv2.VideoCapture(str(args.video))  # 只需要几百帧：直接跳过去截，不用把整场读一遍
        order = sorted(todo)
        for k, i in enumerate(order):
            if k % 10 == 0:
                print(f"  截图 {k}/{len(order)} 帧", flush=True)
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_times[i] * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            for tid, box, name in todo[i]:
                cv2.imwrite(str(crops_dir / f"{name}.jpg"), crop(frame, box, TILE_HEIGHT + 40, 1.6, 1.12), [cv2.IMWRITE_JPEG_QUALITY, 88])
                cv2.imwrite(str(crops_dir / f"{name}_ctx.jpg"), crop(frame, box, TILE_HEIGHT + 40, 4.0, 2.2), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cap.release()
    for tid in files:
        files[tid] = [c for c in sorted(files[tid], key=lambda c: c["time"]) if (out_dir / "review" / c["context"]).exists()]
    for q in data["questions"]:
        q["crops"] = files.get(q["track"], [])
    gallery_review = out_dir_for(args.gallery) / "review"
    looks = person_looks(gallery_review)
    for pid, p in data["people"].items():
        p["looks"] = looks.get(pid, [])
    data["teams"] = json.loads((gallery_review / "people.json").read_text()).get("teams", {})
    data["team_colors_bgr"] = json.loads((gallery_review / "cards.json").read_text()).get("team_colors_bgr", {})
    (review / "questions.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"{len(data['questions'])} 个问题的截图已准备好")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
