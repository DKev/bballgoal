"""球员跟踪 + 自动分队，输出带标注的视频和轨迹数据。

需要先有场地设置文件（videos/比赛.court.json，用 bballgoal.court 生成）。

用法：
    .venv/bin/python -m bballgoal.track videos/比赛.mp4
    .venv/bin/python -m bballgoal.track videos/比赛.mp4 --start 0 --end 60 --fps 10

输出在 output/track_<视频名>/：
    tracks.mp4   标注视频（--debug-video 才出）：A 队 / B 队用球衣颜色画框，裁判白框，每人带轨迹编号
    tracks.json  每帧每人的框、轨迹编号和所属队伍（后续统计用）
    appearance.npz  每人每隔几帧的外观样本（整体外观 + 鞋），给 bballgoal.identify 固定身份用
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from boxmot import BotSort

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.appearance import AppearanceExtractor
from bballgoal.court import Court, court_path_for
from bballgoal.detect import TiledDetector
from bballgoal.frames import read_frames
from bballgoal.probe import video_info
from bballgoal.segments import change_points, majority
from bballgoal.teams import CHROMA, REFEREE, UNKNOWN, fit_teams, is_referee, torso_feature

OUTPUT_WIDTH = 1920
MIN_TRACK_SAMPLES = 10  # 少于这么多帧的轨迹不参与学习两队颜色
REFEREE_SHARE = 0.2  # 轨迹里至少这么多比例的帧看到裁判条纹，就整条算裁判
ADULT_HEIGHT_RATIO = 1.2  # 比同位置小孩的平均身高高出这么多，视为大人
SAMPLE_EVERY = 3  # 每隔几帧采一次外观样本
MIN_SAMPLE_HEIGHT = 90  # 人框低于这个像素太糊，不采外观


def make_tracker(fps: float, appearance: AppearanceExtractor) -> BotSort:
    """BoT-SORT：位置预测 + 外观（OSNet 重识别）一起匹配，两人交错时不容易串号。"""
    return BotSort(
        reid_model=appearance.reid.model,
        use_cmc=False,  # 三脚架固定机位，不需要补偿镜头移动
        frame_rate=int(round(fps)),
        track_high_thresh=0.45,
        track_low_thresh=0.15,
        new_track_thresh=0.5,
        track_buffer=int(fps * 3),  # 被挡住 3 秒内重新出现仍沿用原编号
        match_thresh=0.8,
    )


def run_tracking(video: Path, court: Court, start: float, end: float, fps: float) -> tuple[dict, dict]:
    """返回 (轨迹结果, 外观样本)。外观样本的行号记在每个人的 "sample" 字段里。"""
    info = video_info(video)
    end = min(end, info["duration"])
    detector = TiledDetector(person_conf=0.3)
    appearance = AppearanceExtractor()
    tracker = make_tracker(fps, appearance)
    body_samples: list[np.ndarray] = []
    shoe_samples: list[np.ndarray | None] = []
    y_range = court.y_range(info["height"])
    total = int((end - start) * fps)
    print(f"推理设备：{detector.device}；检测横带 y={y_range[0]}–{y_range[1]}；每秒 {fps:g} 帧")

    frames = []
    detection_features: dict[tuple[int, int], np.ndarray] = {}  # (帧序号, 轨迹号) -> 躯干特征
    started = time.perf_counter()
    for t, frame in read_frames(video, info["width"], info["height"], start, end, fps):
        people = [
            d for d in detector.detect_people_scaled(frame, y_range)
            if court.contains((d.x1 + d.x2) / 2, d.y2)  # 脚底点在比赛区域内
        ]
        if people:
            data = np.array([[d.x1, d.y1, d.x2, d.y2, d.conf, 0] for d in people], dtype=np.float32)
        else:
            data = np.zeros((0, 6), dtype=np.float32)
        tracked = tracker.update(data, frame)

        entries = []
        for x1, y1, x2, y2, track_id, score, _cls, _idx in tracked:
            tid = int(track_id) + 1  # BoxMOT 从 0 开始编号，这里统一从 1 开始
            entries.append({"id": tid, "box": [round(float(v), 1) for v in (x1, y1, x2, y2)], "conf": round(float(score), 3)})
            feat = torso_feature(frame, (x1, y1, x2, y2))
            if feat is not None:
                detection_features[(len(frames), tid)] = feat
        if len(frames) % SAMPLE_EVERY == 0:
            sampled = [p for p in entries if p["box"][3] - p["box"][1] >= MIN_SAMPLE_HEIGHT]
            for p, (body, shoes, torso) in zip(sampled, appearance.extract(frame, [p["box"] for p in sampled])):
                p["sample"] = len(body_samples)
                if torso:
                    p["torso"] = torso
                body_samples.append(body)
                shoe_samples.append(shoes)
        frames.append({"time": round(t, 3), "people": entries})

        done = len(frames)
        if done % 50 == 0 or done == total:
            rate = done / (time.perf_counter() - started)
            print(f"  {done}/{total} 帧  {rate:.1f} 帧/秒  剩余约 {max(0, total - done) / rate / 60:.1f} 分钟", flush=True)

    # 用采样充足的轨迹的中位特征学两队颜色（比单帧稳），再逐帧判断队伍
    per_track: dict[int, list[np.ndarray]] = defaultdict(list)
    for (_i, tid), feat in detection_features.items():
        per_track[tid].append(feat)
    fit_set = {
        tid: np.median(np.stack(f), axis=0) for tid, f in per_track.items() if len(f) >= MIN_TRACK_SAMPLES
    }
    if len(fit_set) < 3:
        print("有效轨迹太少，无法分队", file=sys.stderr)
        colors: dict = {}
        for f in frames:
            for p in f["people"]:
                p["team"] = UNKNOWN
        split_count = 0
    else:
        model = fit_teams(fit_set)
        colors = model.colors_bgr
        split_count = split_tracks_by_team(frames, detection_features, model, fps)

    teams_by_track: dict[int, str] = {}
    for f in frames:
        for p in f["people"]:
            teams_by_track[p["id"]] = p["team"]
    print(f"按队伍变化切开了 {split_count} 处串人")

    shoe_dim = next((len(v) for v in shoe_samples if v is not None), 1)
    samples = {
        "body": np.array(body_samples, np.float16).reshape(len(body_samples), -1),
        "shoes": np.array([v if v is not None else np.zeros(shoe_dim) for v in shoe_samples], np.float16),
        "has_shoes": np.array([v is not None for v in shoe_samples], bool),
    }
    return {
        "video": str(video),
        "info": info,
        "fps": fps,
        "start": start,
        "court": {"polygon": court.polygon, "hoops": court.hoops},
        "team_colors_bgr": {k: list(v) for k, v in colors.items()},
        "tracks": {str(tid): {"team": team} for tid, team in sorted(teams_by_track.items())},
        "frames": frames,
    }, samples


def fit_height_model(frames: list[dict]):
    """拟合“脚底 y 坐标 → 人框高度”的关系（离镜头越近越高），绝大多数检测是小孩，所以代表小孩身高。"""
    ys, hs = [], []
    for f in frames:
        for p in f["people"]:
            ys.append(p["box"][3])
            hs.append(p["box"][3] - p["box"][1])
    if len(ys) < 20:
        return lambda _y: 1e9
    ys_arr, hs_arr = np.array(ys), np.array(hs)
    coef = np.polyfit(ys_arr, hs_arr, 1)
    resid = hs_arr - np.polyval(coef, ys_arr)
    keep = np.abs(resid) < 2 * resid.std()  # 去掉离群点后再拟合一次
    coef = np.polyfit(ys_arr[keep], hs_arr[keep], 1)
    return lambda y: max(1.0, float(np.polyval(coef, y)))


def split_tracks_by_team(
    frames: list[dict], detection_features: dict[tuple[int, int], np.ndarray], model, fps: float
) -> int:
    """逐帧判断队伍，轨迹中途队伍持续变化（串到别队的人）就切成新轨迹。原地改写 frames。"""
    occurrences: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for i, f in enumerate(frames):
        for p in f["people"]:
            occurrences[p["id"]].append((i, p))
    next_id = max(occurrences, default=0) + 1
    window, min_run = max(1, int(fps * 0.5)), max(2, int(fps * 0.8))
    expected_height = fit_height_model(frames)
    splits = 0
    for tid, items in occurrences.items():
        feats = [detection_features[(i, tid)] for i, _p in items if (i, tid) in detection_features]
        # 裁判按整条轨迹判断：条纹只在部分帧看得清（侧身、远处时看不出），
        # 但小孩球衣几乎不会出现条纹。条纹帧比例够高，或偶尔有条纹且明显比同位置的小孩高（大人），就是裁判
        stripe_share = float(np.mean([is_referee(f) for f in feats])) if feats else 0.0
        tall = float(np.median([
            (p["box"][3] - p["box"][1]) / expected_height(p["box"][3]) for _i, p in items
        ]))
        # 另外：大人、衣服基本没有彩色（黑白条纹糊了以后就是黑灰白），也是裁判
        colorless = bool(feats) and float(np.median([f[CHROMA] for f in feats])) < 0.15
        if stripe_share >= REFEREE_SHARE or (tall >= ADULT_HEIGHT_RATIO and (stripe_share >= 0.05 or colorless)):
            for _i, p in items:
                p["team"] = REFEREE
            continue
        labels = []
        for i, _p in items:
            feat = detection_features.get((i, tid))
            team = model.assign(feat) if feat is not None else UNKNOWN
            labels.append(None if team in (UNKNOWN, REFEREE) else team)
        cuts = change_points(labels, window, min_run)
        bounds = [0, *cuts, len(items)]
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            new_id = tid if k == 0 else next_id
            if k:
                next_id += 1
                splits += 1
            team = majority(labels[a:b]) or UNKNOWN
            for _i, p in items[a:b]:
                p["id"], p["team"] = new_id, team
    return splits


def short_labels(result: dict) -> dict[int, str]:
    """把 ByteTrack 的大编号换成每队从 1 开始的短编号：A1、A2…、B1…、裁判 R1…"""
    counters: dict[str, int] = defaultdict(int)
    labels = {}
    for f in result["frames"]:
        for p in f["people"]:
            tid = p["id"]
            if tid in labels:
                continue
            team = p["team"]
            prefix = {"A": "A", "B": "B", REFEREE: "R"}.get(team, "?")
            counters[prefix] += 1
            labels[tid] = f"{prefix}{counters[prefix]}"
    return labels


def render(result: dict, video: Path, out_path: Path, labels: dict[int, str] | None = None) -> None:
    info = result["info"]
    scale = OUTPUT_WIDTH / info["width"]
    out_h = int(round(info["height"] * scale / 2) * 2)
    colors = {k: tuple(v) for k, v in result["team_colors_bgr"].items()}
    colors.setdefault(UNKNOWN, (160, 160, 160))
    labels = labels or short_labels(result)
    polygon = (np.array(result["court"]["polygon"]) * scale).astype(np.int32)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUTPUT_WIDTH}x{out_h}",
            "-r", f"{result['fps']:.3f}", "-i", "-",
            "-c:v", "h264_videotoolbox", "-b:v", "8M", "-pix_fmt", "yuv420p", str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    frames = result["frames"]
    end = frames[-1]["time"] + 0.5 / result["fps"]
    reader = read_frames(video, info["width"], info["height"], result["start"], end, result["fps"])
    for entry, (_t, frame) in zip(frames, reader):
        view = cv2.resize(frame, (OUTPUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
        cv2.polylines(view, [polygon], True, (0, 200, 255), 1)
        counts: dict[str, int] = defaultdict(int)
        for p in entry["people"]:
            x1, y1, x2, y2 = (int(v * scale) for v in p["box"])
            color = colors.get(p["team"], colors[UNKNOWN])
            counts[p["team"]] += 1
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            text = labels[p["id"]]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(view, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv2.putText(view, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        status = (
            f"{entry['time']:7.1f}s   A: {counts['A']}   B: {counts['B']}   "
            f"ref: {counts[REFEREE]}   ?: {counts[UNKNOWN]}"
        )
        cv2.rectangle(view, (0, out_h - 40), (OUTPUT_WIDTH, out_h), (0, 0, 0), -1)
        cv2.putText(view, status, (16, out_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        ffmpeg.stdin.write(view.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=float("inf"))
    parser.add_argument("--fps", type=float, default=10, help="每秒处理多少帧（默认 10）")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--debug-video", action="store_true", help="另外渲染带框的标注视频 tracks.mp4（检查跟踪用；一场几个 GB，默认不出）")
    args = parser.parse_args()

    court_file = court_path_for(args.video)
    if not court_file.exists():
        print(f"缺少场地设置：{court_file}\n先运行：.venv/bin/python -m bballgoal.court {args.video}", file=sys.stderr)
        return 1
    court = Court.load(court_file)
    out_dir = args.out or out_dir_for(args.video)
    out_dir.mkdir(parents=True, exist_ok=True)

    result, samples = run_tracking(args.video, court, args.start, args.end, args.fps)
    (out_dir / "tracks.json").write_text(json.dumps(result, ensure_ascii=False))
    np.savez_compressed(out_dir / "appearance.npz", **samples)

    teams = defaultdict(int)
    for t in result["tracks"].values():
        teams[t["team"]] += 1
    print(f"轨迹数：{len(result['tracks'])}  （A 队 {teams['A']}，B 队 {teams['B']}，裁判 {teams[REFEREE]}，未分队 {teams[UNKNOWN]}）")

    if args.debug_video:
        print("生成标注视频…")
        render(result, args.video, out_dir / "tracks.mp4")
        print(f"输出：{out_dir}/tracks.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
