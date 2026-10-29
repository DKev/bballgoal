"""给远处的人补外观样本。

跟踪时只给框高够大的人采外观（track.MIN_SAMPLE_HEIGHT），对面半场的孩子常常比这个小，整段轨迹
一个样本都没有，认整场时只能“看不清”留给你问。这里把视频再读一遍（不重新检测和跟踪，所以比跟踪快很多），
只给样本太少的轨迹、在框高至少 MIN_HEIGHT 的帧上补采样，追加到 appearance.npz，并把行号写回 identified.json。

    .venv/bin/python -m bballgoal.backfill output/<比赛>/full.mp4
做过一次会在 identified.json 里记下，再跑直接跳过（identify 重跑后会重新补）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bballgoal.frames import read_frames
from bballgoal.probe import video_info
from bballgoal.paths import out_dir as out_dir_for
from bballgoal.track import SAMPLE_EVERY

MIN_HEIGHT = 45  # 再小 OSNet 也看不出什么
ENOUGH = 4  # 已经有这么多样本的轨迹不用补


def plan(result: dict, min_height: float = MIN_HEIGHT, enough: int = ENOUGH) -> dict[int, list[int]]:
    """帧序号 → 这一帧里要补样本的人（在 frame["people"] 里的下标）。"""
    have: dict[int, int] = {}
    for f in result["frames"]:
        for p in f["people"]:
            if p.get("sample") is not None:
                have[p["id"]] = have.get(p["id"], 0) + 1
    todo: dict[int, list[int]] = {}
    for i, f in enumerate(result["frames"]):
        if i % SAMPLE_EVERY:
            continue
        idx = [k for k, p in enumerate(f["people"]) if p.get("sample") is None and have.get(p["id"], 0) < enough
               and p["box"][3] - p["box"][1] >= min_height]
        if idx:
            todo[i] = idx
    return todo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    out = out_dir_for(args.video)
    result_file = out / "identified.json"
    result = json.loads(result_file.read_text())
    if result.get("backfilled"):
        print("已经补过外观样本")
        return 0
    todo = plan(result)
    print(f"要补样本的帧：{len(todo)}，人次：{sum(len(v) for v in todo.values())}", flush=True)
    with np.load(out / "appearance.npz") as z:
        data = {k: z[k] for k in z.files}
    body, shoes, has = list(data["body"]), list(data["shoes"]), list(data["has_shoes"])
    if todo:
        from bballgoal.appearance import AppearanceExtractor

        extractor = AppearanceExtractor()
        info = video_info(args.video)
        fps = result["fps"]
        start = result.get("start", 0.0)
        end = start + len(result["frames"]) / fps
        last = max(todo)
        started = time.perf_counter()
        for i, (_t, frame) in enumerate(read_frames(args.video, info["width"], info["height"], start, end, fps)):
            if i > last:
                break
            if i in todo:
                people = [result["frames"][i]["people"][k] for k in todo[i]]
                for p, (b, s, _torso) in zip(people, extractor.extract(frame, [p["box"] for p in people])):
                    p["sample"] = len(body)
                    body.append(b.astype(np.float16))
                    shoes.append(s.astype(np.float16) if s is not None else np.zeros(data["shoes"].shape[1], np.float16))
                    has.append(s is not None)
            if i % 100 == 0 and i:
                rate = i / (time.perf_counter() - started)
                print(f"  {i}/{last + 1} 帧  剩余约 {(last - i) / rate / 60:.1f} 分钟", flush=True)
        np.savez_compressed(out / "appearance.partial.npz", body=np.array(body, np.float16), shoes=np.array(shoes, np.float16),
                            has_shoes=np.array(has, bool))
        (out / "appearance.partial.npz").replace(out / "appearance.npz")
    result["backfilled"] = True
    tmp = result_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    tmp.replace(result_file)
    print(f"补了 {len(body) - len(data['body'])} 个外观样本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
