"""bballgoal 本地网站：从导入视频到出集锦，一步一步在浏览器里完成。

    .venv/bin/python -m bballgoal.app          # 打开 http://localhost:8000
    .venv/bin/python -m bballgoal.app --lan    # 同一 Wi-Fi 的 iPad / 手机也能打开

只在本机运行，视频和结果都留在这台电脑上，不上传。

步骤：
  ① 导入视频：选相机里的一次录像（.insv，可能被相机切成好几个文件）或已有的平面视频，
     在画面上点出场地边界和两个篮筐；
  ② 认人：名单——从整场均匀抽几段，跟踪、聚类后做成卡片，你一张张确认队和号码（同队同号自动合并）；
     整场核对——拿名单认整场，按人批量核对，剩下的逐个问你。后台同时跑整场的转换、跟踪、补外观；
  ③ 点球：点 150–200 张画面里的球，为这个机位训练找球模型；
  ④ 事件审核：自动找投篮（进没进、谁投的），虚拟镜头片段逐个确认，标出三分和罚篮；
  ⑤ 集锦：每个孩子确认过的精彩片段接成个人集锦。

每场比赛一个文件夹 output/<名字>/：game.json、raw/（原始文件）、full.mp4、enroll.mp4、full/ 和 enroll/（各自的结果）、logs/。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from glob import glob

import numpy as np
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

WEB_DIR = Path(__file__).with_name("review_web")
OUTPUT = Path("output")  # 每场比赛一个文件夹：output/<比赛>/（game.json、raw/、full.mp4、enroll.mp4、full/、enroll/、logs/）
PY = sys.executable
CAMERA_GLOBS = ("/Volumes/*/DCIM/*/VID_*.insv", "/Volumes/*/DCIM/VID_*.insv")
RECORDING = re.compile(r"(VID_\d{8}_\d{6})_\d+_(\d+)\.insv$")


# ---------- 比赛 ----------

def game_dir(name: str) -> Path:
    return OUTPUT / name


def game_file(name: str) -> Path:
    return game_dir(name) / "game.json"


def load_game(name: str) -> dict:
    return json.loads(game_file(name).read_text())


def save_game(game: dict) -> None:
    game_dir(game["name"]).mkdir(parents=True, exist_ok=True)
    tmp = game_file(game["name"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(game, ensure_ascii=False, indent=1))
    tmp.replace(game_file(game["name"]))


def out_dir(video: str) -> Path:
    from bballgoal.paths import out_dir as video_out_dir

    return video_out_dir(video)


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name.strip())[:60]
    return name or time.strftime("game_%Y%m%d_%H%M")


def new_game(name: str, sources: list[str]) -> dict:
    name = safe_name(name)
    if game_file(name).exists():
        raise ValueError(f"已经有叫 {name} 的比赛了")
    raw = all(s.endswith(".insv") for s in sources)
    folder = game_dir(name)
    # 原始文件都会放进 <比赛>/raw/（导入时复制或移动过去）
    full = str(folder / "full.mp4") if raw else str(folder / "raw" / Path(sources[0]).name)
    game = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "sources": sources,
        "raw": raw,
        "full_video": full,
        "enroll_video": str(folder / "enroll.mp4"),
    }
    save_game(game)
    return game


def list_games() -> list[dict]:
    games = []
    for f in sorted(OUTPUT.glob("*/game.json"), key=lambda p: -p.stat().st_mtime):
        try:
            game = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        games.append({**game, "steps": steps(game)})
    return games


def court_file(video: str) -> Path:
    return Path(video).with_suffix(".court.json")


def steps(game: dict) -> dict:
    """每一步的状态：todo / running / ready（可以开始操作）/ done。从产出的文件推断，服务重启也不丢。"""
    jobs = {k: job_status(game["name"], k) for k in ("import", "enroll", "full", "assign", "balltrain", "events", "reel")}
    enroll_dir, full_dir = out_dir(game["enroll_video"]), out_dir(game["full_video"])
    people_file = enroll_dir / "review" / "people.json"
    people = json.loads(people_file.read_text())["people"] if people_file.exists() else []
    people = [p for p in people if p.get("tracks")]
    confirmed = sum(p.get("confirmed") for p in people)

    s1 = "done" if court_file(game["full_video"]).exists() else ("running" if jobs["import"]["running"] else
                                                                "ready" if Path(frame_file(game)).exists() else "todo")
    if (enroll_dir / "review" / "cards.json").exists():
        s2 = "done" if people and confirmed == len(people) else "ready"
    else:
        s2 = "running" if jobs["enroll"]["running"] else "todo"
    answers = full_dir / "review" / "answers.json"
    questions_file = full_dir / "review" / "questions.json"
    remaining = None
    if questions_file.exists():
        saved = json.loads(answers.read_text()) if answers.exists() else {}
        answered = saved.get("answers", {})
        remaining = 0 if saved.get("finished") else sum(str(q["track"]) not in answered for q in json.loads(questions_file.read_text())["questions"])
    if jobs["assign"]["running"]:
        s3 = "running"
    elif remaining is not None:
        s3 = "done" if remaining == 0 else "ready"
    elif jobs["assign"]["running"] or jobs["full"]["running"]:
        s3 = "running"
    else:
        s3 = "todo"
    bdir = balls_dir(game)
    labels_file = bdir / "labels.json"
    labeled = len(json.loads(labels_file.read_text())["labels"]) if labels_file.exists() else 0
    if jobs["balltrain"]["running"]:
        s_balls = "running"
    elif (bdir / "model.pt").exists():
        s_balls = "done"
    elif (bdir / "items.json").exists():
        s_balls = "ready"
    else:
        s_balls = "running" if jobs["enroll"]["running"] else "todo"
    train_info = json.loads((bdir / "train.json").read_text()) if (bdir / "train.json").exists() else None
    events_file = full_dir / "events.json"
    review_file = full_dir / "review" / "events_review.json"
    shot_ids = {e["id"] for e in json.loads(events_file.read_text())["events"] if e["type"] == "shot"} if events_file.exists() else set()
    events_total = len(shot_ids)
    reviewed = json.loads(review_file.read_text()) if review_file.exists() else {}
    events_done = sum(1 for k, v in reviewed.get("events", {}).items()
                      if int(k) in shot_ids and v.get("status") in ("confirmed", "deleted"))
    if jobs["events"]["running"]:
        s_events = "running"
    elif events_file.exists():
        s_events = "done" if events_total and events_done >= events_total else "ready"
    else:
        s_events = "todo"
    reels = sorted(p.stem for p in (game_dir(game["name"]) / "reels").glob("*.mp4")) if (game_dir(game["name"]) / "reels").exists() else []
    return {
        "import": s1, "people": s2, "questions": s3, "balls": s_balls, "events": s_events,
        "reels": "done" if reels else ("ready" if s_events in ("ready", "done") else "todo"),
        "events_total": events_total, "events_reviewed": events_done, "reel_files": reels,
        "balls_labeled": labeled, "balls_train": train_info,
        "jobs": jobs,
        "people_count": len(people), "people_confirmed": confirmed,
        "full_tracked": (full_dir / "identified.json").exists(),
        "questions_remaining": remaining,
    }


def frame_file(game: dict) -> str:
    return str(game_dir(game["name"]) / "frame.jpg")


# ---------- 后台任务 ----------
# 一个任务 = 依次执行的几条命令。输出写进日志文件，进度从日志里读（服务重启也能接着看）。



def log_file(game: str, job: str) -> Path:
    return game_dir(game) / "logs" / f"{job}.log"


def pid_file(game: str, job: str) -> Path:
    return game_dir(game) / "logs" / f"{job}.pid"


JOB_COMMANDS = {}  # 任务名 → 生成命令列表的函数（在下面定义后填入）


def start_job(game: str, job: str, commands=None) -> None:
    """在独立的后台进程里依次执行这个任务的各步（网站服务重启、浏览器关掉都不影响）。"""
    if job_status(game, job)["running"]:
        return
    log = log_file(game, job)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    proc = subprocess.Popen([PY, "-m", "bballgoal.app", "runjob", game, job], stdout=subprocess.DEVNULL,
                            stderr=log.open("a"), start_new_session=True)
    pid_file(game, job).write_text(str(proc.pid))
    # 进程结束后要有人收尸，否则它变成僵尸进程，pid 一直“活着”，网页永远显示在跑、也开不了新任务
    threading.Thread(target=proc.wait, daemon=True).start()
    time.sleep(0.3)


def run_job(game_name: str, job: str) -> int:
    """后台进程：按顺序执行各步，输出写进日志。日志开头写计划，每步写开始时间，供网页显示步骤清单和剩余时间。"""
    game = load_game(game_name)
    commands = JOB_COMMANDS[job](game)
    log = log_file(game_name, job)
    pid_file(game_name, job).write_text(str(os.getpid()))
    with log.open("a") as out:
        out.write("## " + json.dumps([t for t, _c in commands], ensure_ascii=False) + "\n")
        for title, cmd in commands:
            out.write(f"== {title} @{time.time():.0f}\n")
            out.flush()
            # 同一场比赛的两个任务可能走到同一个模块（整场任务和找事件任务都会逐帧找球、看篮筐）：
            # 后到的等先到的做完，再跑时直接用它的缓存，免得两份 GPU 检测抢资源、各做一遍
            with (log.parent / f"{cmd[2]}.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    out.write("等另一个任务里的同一步做完…\n")
                    out.flush()
                    fcntl.flock(lock, fcntl.LOCK_EX)
                code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
            if code != 0:
                out.write(f"!! 失败（退出码 {code}）\n")
                break
        else:
            out.write(f"== 完成 @{time.time():.0f}\n")
    pid_file(game_name, job).unlink(missing_ok=True)
    return 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


PROGRESS = re.compile(r"(\d+)/(\d+)\s*(?:帧|段|轮)")
EPOCH = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s.*?(\d+)%")  # ultralytics 训练进度：轮/总轮数 显存 … 这一轮的百分比
PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
STEP = re.compile(r"^== (.+?)(?: @(\d+))?$")
ETA = re.compile(r"剩余约\s*([\d.]+)\s*分钟")


def job_status(game: str, job: str) -> dict:
    log = log_file(game, job)
    pid = pid_file(game, job)
    try:
        running = pid.exists() and _alive(int(pid.read_text() or 0))
    except ValueError:
        running = False
    if not log.exists():
        return {"running": False, "state": "none", "step": None, "progress": None, "error": None, "steps": [], "eta": None}
    lines = [ln for ln in re.split(r"[\r\n]+", log.read_text(errors="ignore").replace("\x1b[K", "")) if ln.strip()]
    plan = next((json.loads(ln[3:]) for ln in lines if ln.startswith("## ")), None)
    started = [(m[1], int(m[2]) if m[2] else None) for ln in lines if (m := STEP.match(ln))]
    step, step_start = started[-1] if started else (None, None)
    failed = next((ln for ln in reversed(lines) if ln.startswith("!! ")), None)
    progress, eta, detail = None, None, None
    for ln in reversed(lines[-60:]):
        if ln.startswith("== "):
            break
        if m := EPOCH.search(ln):
            progress = (int(m[1]) - 1 + int(m[3]) / 100) / max(1, int(m[2]))
            detail = f"第 {m[1]}/{m[2]} 轮"
            break
        if m := PROGRESS.search(ln):
            progress = int(m[1]) / max(1, int(m[2]))
            detail = ln[: m.start()].strip() or None
            if e := ETA.search(ln):
                eta = float(e[1]) * 60
            break
        if (m := PERCENT.search(ln)) and "转换" in ln:
            progress = float(m[1]) / 100
            break
    if eta is None and progress and step_start and running:
        elapsed = time.time() - step_start
        if progress > 0.02 and elapsed > 20:
            eta = elapsed / progress * (1 - progress)
    state = "running" if running else "failed" if failed else "done" if step == "完成" else "stopped"
    done_titles = [t for t, _ in started[:-1]] if step != "完成" else [t for t, _ in started]
    titles = plan or [t for t, _ in started if t != "完成"]
    steps = [{"title": t, "state": "done" if t in done_titles or state == "done" else
              ("running" if t == step and running else "failed" if t == step and failed else "todo")} for t in titles]
    return {"running": running, "state": state, "step": step, "detail": detail if running else None, "progress": progress, "eta": eta, "steps": steps,
            "error": failed[3:] if failed and not running else None, "tail": lines[-3:]}


def stop_job(game: str, job: str) -> None:
    pid = pid_file(game, job)
    if pid.exists():
        try:
            os.killpg(int(pid.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            return
        for _ in range(50):  # 等它真的退出，否则紧接着的重新开始会以为还在跑
            if not job_status(game, job)["running"]:
                return
            time.sleep(0.1)
        try:
            os.killpg(int(pid.read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


# ---------- 各步骤要跑的命令 ----------

def import_commands(game: dict) -> list[tuple[str, list[str]]]:
    """导入：原始文件放进 <比赛>/raw/（相机存储卡上的复制过来，卡可能被拔掉；电脑上的直接移动，不占双份空间），
    再定画面、截一帧画场地。"""
    raw_dir = game_dir(game["name"]) / "raw"
    cmds, local = [], []
    for src in game["sources"]:
        dst = raw_dir / Path(src).name
        local.append(str(dst))
        if Path(src).resolve() == dst.resolve():
            continue
        if src.startswith("/Volumes/"):
            cmds.append((f"复制 {Path(src).name}", [PY, "-m", "bballgoal.app", "copy", src, str(dst)]))
        else:
            cmds.append((f"移动 {Path(src).name}", [PY, "-m", "bballgoal.app", "move", src, str(dst)]))
    game["sources"] = local
    if game["raw"]:
        game["view"] = str(Path(game["full_video"]).with_suffix(".view.json"))
        cmds.append(("判断哪个镜头对着球场", [PY, "-m", "bballgoal.fisheye", *local, "-o", game["full_video"], "--view-only"]))
    else:
        game["full_video"] = local[0]
    save_game(game)
    cmds.append(("截一帧画面", [PY, "-m", "bballgoal.app", "frame", game["name"]]))
    # 沿用最近一场训练好的找球模型：整场跟踪完就直接逐帧找球，点球这一步可以跳过
    cmds.append(("沿用上一场的找球模型", [PY, "-m", "bballgoal.balllabel", "inherit", str(game_dir(game["name"]))]))
    return cmds


def enroll_commands(game: dict) -> list[tuple[str, list[str]]]:
    """每一步的结果已经有了就跳过（重启任务不会白白重跑几个小时）。"""
    video = game["enroll_video"]
    out = out_dir(video)
    cmds = []
    if not Path(video).exists():
        src = game["sources"] if game["raw"] else [game["full_video"]]
        extra = ["--view", game["view"]] if game["raw"] else []
        cmds.append(("从整场均匀抽 8 段", [PY, "-m", "bballgoal.enroll", *src, "-o", video, *extra]))
    if not (out / "tracks.json").exists():
        cmds.append(("跟踪每个人", [PY, "-m", "bballgoal.track", video]))
    if not (out / "identified.json").exists():
        cmds.append(("认号码、按外观聚成人", [PY, "-m", "bballgoal.identify", video]))
    if not (out / "review" / "cards.json").exists():
        cmds.append(("做卡片和头像", [PY, "-m", "bballgoal.cards", video]))
    if not (balls_dir(game) / "items.json").exists():
        cmds.append(("准备点球画面", [PY, "-m", "bballgoal.balllabel", "prep", video]))
    return cmds


def balls_dir(game: dict) -> Path:
    return game_dir(game["name"]) / "balls"


def balltrain_commands(game: dict) -> list[tuple[str, list[str]]]:
    """训练完，如果整场已经跟踪好，就用新模型提前把最耗时的逐帧找球、篮筐检测跑掉（不依赖认人结果）。"""
    cmds = [("用你的标注训练找球模型", [PY, "-m", "bballgoal.balllabel", "train", str(game_dir(game["name"]))])]
    if (out_dir(game["full_video"]) / "tracks.json").exists():
        cmds += [("用新模型在整场逐帧找球", [PY, "-m", "bballgoal.ball", game["full_video"], "--detect-only"]),
                 ("用新模型逐帧看篮筐", [PY, "-m", "bballgoal.events", game["full_video"], "--scan-only"])]
    return cmds


def full_commands(game: dict) -> list[tuple[str, list[str]]]:
    video = game["full_video"]
    out = out_dir(video)
    cmds = []
    if game["raw"] and not Path(video).exists():
        cmds.append(("整场 360 转平面", [PY, "-m", "bballgoal.fisheye", *game["sources"], "-o", video, "--view", game["view"]]))
    if not (out / "tracks.json").exists():
        cmds.append(("跟踪整场", [PY, "-m", "bballgoal.track", video]))
    if not (out / "identified.json").exists():
        cmds.append(("认整场的号码", [PY, "-m", "bballgoal.identify", video]))
    if not (out / "identified.json").exists() or b'"backfilled"' not in (out / "identified.json").read_bytes()[-200:]:
        cmds.append(("给远处的人补外观", [PY, "-m", "bballgoal.backfill", video]))
    if (balls_dir(game) / "model.pt").exists():  # 已经训练好找球模型：顺手把逐帧找球、篮筐检测跑掉
        cmds += [("逐帧找球", [PY, "-m", "bballgoal.ball", video, "--detect-only"]),
                 ("逐帧看篮筐", [PY, "-m", "bballgoal.events", video, "--scan-only"])]
    return cmds


def events_commands(game: dict) -> list[tuple[str, list[str]]]:
    video = game["full_video"]
    out = out_dir(video)
    cmds = []
    if game["raw"] and not Path(video).exists():  # 处理完为了省空间删掉了：找篮圈、判断进没进还要用，从 360 原片再转一份
        cmds.append(("整场 360 转平面", [PY, "-m", "bballgoal.fisheye", *game["sources"], "-o", video, "--view", game["view"]]))
    cmds.append(("找球、判断谁持球", [PY, "-m", "bballgoal.ball", video]))  # 逐帧检测有缓存，这里主要是按最新的认人结果判断持球人
    cmds.append(("找投篮（进没进、谁投的）", [PY, "-m", "bballgoal.events", video]))
    cmds.append(("渲染事件片段（虚拟镜头）", [PY, "-m", "bballgoal.clips", video]))
    return cmds


def reel_commands(game: dict) -> list[tuple[str, list[str]]]:
    """一个或一批集锦（{"batch": [...]}：给所有人生成时），按顺序一个个剪。"""
    req = game.get("reel_request", {})
    cmds = []
    for r in req.get("batch") or [req]:
        cmd = [PY, "-m", "bballgoal.reels", game["full_video"], r.get("player", "")]
        if r.get("team"):
            cmd += ["--team", r["team"]]
        if r.get("misses"):
            cmd.append("--misses")
        if r.get("touches"):
            cmd.append("--touches")
        if r.get("lang") in ("zh", "en"):
            cmd += ["--lang", r["lang"]]
        if r.get("music") is False:
            cmd.append("--no-music")
        kind = "五佳球" if r.get("team") else "触球集锦" if r.get("touches") else "投篮集锦"
        cmds.append((f"剪{kind}：{r.get('label', r.get('player', ''))}", cmd))
    return cmds


def assign_commands(game: dict) -> list[tuple[str, list[str]]]:
    return [("拿名单认整场", [PY, "-m", "bballgoal.assign", game["full_video"], "--gallery", game["enroll_video"]]),
            ("准备问题的截图", [PY, "-m", "bballgoal.questions", game["full_video"], "--gallery", game["enroll_video"]])]


JOB_COMMANDS.update({"import": lambda g: import_commands(g), "enroll": lambda g: enroll_commands(g),
                     "full": lambda g: full_commands(g), "assign": lambda g: assign_commands(g),
                     "balltrain": lambda g: balltrain_commands(g), "events": lambda g: events_commands(g),
                     "reel": lambda g: reel_commands(g)})


# ---------- 发现视频 ----------

def find_sources() -> list[dict]:
    """相机存储卡和 videos/raw 里的 360 录像（按录像分组），以及 videos/ 里的平面视频。"""
    files = [f for pattern in CAMERA_GLOBS for f in glob(pattern)] + glob("videos/**/*.insv", recursive=True)
    groups: dict[str, list[str]] = {}
    for f in files:
        if m := RECORDING.search(f):
            groups.setdefault(f"{Path(f).parent}|{m[1]}", []).append(f)
    out = []
    for key, parts in groups.items():
        parts.sort(key=lambda f: int(RECORDING.search(f)[2]))
        size = sum(os.path.getsize(p) for p in parts)
        if size < 1e9:  # 太短的（几十秒的测试）不列
            continue
        stamp = RECORDING.search(parts[0])[1]
        out.append({
            "kind": "360", "parts": parts, "size_gb": round(size / 1e9, 1),
            "label": f"{stamp[4:8]}-{stamp[8:10]}-{stamp[10:12]} {stamp[13:15]}:{stamp[15:17]}",
            "where": "相机存储卡" if parts[0].startswith("/Volumes/") else "这台电脑",
            "minutes": round(sum(_duration(p) for p in parts) / 60),
        })
    for f in sorted(glob("videos/*.mp4")):
        if re.search(r"_(enroll|preview|360)\.mp4$", f) or "test" in f:
            continue
        out.append({"kind": "flat", "parts": [f], "size_gb": round(os.path.getsize(f) / 1e9, 1), "label": Path(f).stem,
                    "where": "这台电脑", "minutes": round(_duration(f) / 60)})
    return out


_durations: dict[str, float] = {}


def _duration(path: str) -> float:
    if path not in _durations:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True)
        try:
            _durations[path] = float(out.stdout.strip())
        except ValueError:
            _durations[path] = 0.0
    return _durations[path]


# ---------- 认人时的建议：用当前（网页上的）名单现算相似度 ----------

_samples_cache: dict[str, tuple] = {}


def suggest(game: dict, people: list[dict], track_ids: list[int], top: int = 3) -> dict[str, list[dict]]:
    import numpy as np

    from bballgoal.assign import build_gallery, similarities, track_info, track_samples

    out = out_dir(game["enroll_video"])
    key = str(out)
    if key not in _samples_cache:
        result = json.loads((out / "identified.json").read_text())
        samples = track_samples(result, np.load(out / "appearance.npz"))
        votes = json.loads((out / "players.json").read_text()).get("track_votes", {})
        numbers = {int(t): v["number"] for t, v in votes.items() if v["number"]}
        teams = {t: i["team"] for t, i in track_info(result).items()}
        _samples_cache[key] = (samples, numbers, teams)
    samples, numbers, teams = _samples_cache[key]
    named = [p for p in people if p.get("tracks") and p.get("kind", "player") != "ignore" and p.get("confirmed")]
    if len(named) < 3:
        named = [p for p in people if p.get("tracks") and p.get("kind", "player") != "ignore"]
    gallery = build_gallery(named, samples)
    result = {}
    for tid in track_ids:
        track = {name: samples[name][tid] for name in ("body", "shoes") if tid in samples[name]}
        scores = similarities(gallery, track, numbers.get(tid), teams.get(tid)) if track else {}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
        result[str(tid)] = [{"person": pid, "score": round(sc, 3)} for pid, sc in ranked]
    return result


# ---------- 名单沿用到下一场 ----------

SAME_JERSEY = 80.0  # 两场的球衣颜色（BGR）差多少以内算同一个队


def _team_colors(tracks_file: Path) -> dict:
    """tracks.json 里各队的球衣颜色。文件很大，颜色写在开头，只读前面一点。"""
    if not tracks_file.exists():
        return {}
    with tracks_file.open("rb") as f:
        head = f.read(65536).decode("utf-8", "ignore")
    m = re.search(r'"team_colors_bgr":\s*(\{[^{}]*\})', head)
    return {k: v for k, v in json.loads(m.group(1)).items() if k in ("A", "B")} if m else {}


def carry_roster(game: dict) -> dict | None:
    """新比赛第一次打开名单：找最近一场认过人的比赛，按球衣颜色把两场的队对上（A/B 是按颜色分的，
    明天你们队不一定还是 A），带过来队名和“号码 → 名字”。对方换了队、颜色对不上的就不带。"""
    here = _team_colors(out_dir(game["enroll_video"]) / "tracks.json")
    best = None
    for f in OUTPUT.glob("*/game.json"):
        g = json.loads(f.read_text())
        if g["name"] == game["name"]:
            continue
        people = out_dir(g["enroll_video"]) / "review" / "people.json"
        if people.exists() and (best is None or people.stat().st_mtime > best[1].stat().st_mtime):
            best = (g, people)
    if not best or not here:
        return None
    prev, people_file = best
    there = _team_colors(out_dir(prev["enroll_video"]) / "tracks.json")
    data = json.loads(people_file.read_text())
    teams = {}
    for t, color in here.items():
        match = min(there, key=lambda k: np.linalg.norm(np.subtract(there[k], color)), default=None)
        if match is None or np.linalg.norm(np.subtract(there[match], color)) > SAME_JERSEY:
            continue
        roster = {p["number"]: p.get("name", "") for p in data["people"]
                  if p.get("team") == match and p.get("kind") == "player" and p.get("number")}
        teams[t] = {"name": data.get("teams", {}).get(match, {}).get("name", ""), "roster": roster}
    return {"from": prev["name"], "teams": teams} if teams else None


# ---------- 拆开一段：跟踪器中途从一个人跳到了另一个人身上 ----------

_pose = None


def _frames_at(video: str, times: list[float]):
    import cv2

    cap = cv2.VideoCapture(video)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        yield frame if ok else None
    cap.release()


def _track_items(result: dict, tid: int) -> list[tuple[int, float, list]]:
    return [(i, f["time"], p["box"]) for i, f in enumerate(result["frames"]) for p in f["people"] if p["id"] == tid]


def track_strip(game: dict, tid: int, n: int = 10) -> list[dict]:
    """这一段从头到尾均匀取 n 张截图（帧号、整场时间），给“拆开”用。"""
    import cv2

    from bballgoal.cards import TILE_HEIGHT, crop, game_clock

    out = out_dir(game["enroll_video"])
    result = json.loads((out / "identified.json").read_text())
    items = _track_items(result, tid)
    if not items:
        return []
    pick = [items[int(k)] for k in np.linspace(0, len(items) - 1, min(n, len(items)))] if len(items) > 1 else items
    to_game = game_clock(Path(game["enroll_video"]))
    strip_dir = out / "review" / "strips"
    strip_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for (i, t, box), frame in zip(pick, _frames_at(game["enroll_video"], [t for _i, t, _b in pick])):
        if frame is None:
            continue
        name = f"strips/{tid}_{i}.jpg"
        cv2.imwrite(str(out / "review" / name), crop(frame, box, TILE_HEIGHT, 1.6, 1.12), [cv2.IMWRITE_JPEG_QUALITY, 85])
        shots.append({"frame": i, "time": round(to_game(t), 1), "tile": name})
    return shots


def split_track(game: dict, tid: int, frame: int) -> dict:
    """从第 frame 帧起（含）这一段改成新的一段。更新 identified.json 和 cards.json，返回两段的新信息。"""
    global _pose
    import cv2

    from bballgoal.appearance import PoseEstimator
    from bballgoal.avatar import face_score, head_photo, look
    from bballgoal.cards import CONTEXT_HEIGHT, CROPS_PER_TRACK, TILE_HEIGHT, crop, game_clock, pick_frames

    out = out_dir(game["enroll_video"])
    result = json.loads((out / "identified.json").read_text())
    new_id = max(p["id"] for f in result["frames"] for p in f["people"]) + 1
    moved = 0
    for i, f in enumerate(result["frames"][frame:], start=frame):
        for p in f["people"]:
            if p["id"] == tid:
                p["id"] = new_id
                moved += 1
    if not moved:
        raise ValueError("这一帧之后没有这段了")
    (out / "identified.json").write_text(json.dumps(result, ensure_ascii=False))
    _samples_cache.pop(str(out), None)

    if _pose is None:
        _pose = PoseEstimator()
    to_game = game_clock(Path(game["enroll_video"]))
    fps = result["fps"]
    cards_file = out / "review" / "cards.json"
    cards = json.loads(cards_file.read_text())
    owner = next(c for c in cards["cards"] for t in c["tracks"] if t["id"] == tid)
    old = next(t for t in owner["tracks"] if t["id"] == tid)
    entries = {}
    for piece in (tid, new_id):
        items = _track_items(result, piece)
        picks = pick_frames(items, CROPS_PER_TRACK)
        crops_, best, looks = [], None, []
        for (i, t, box), fr in zip(picks, _frames_at(game["enroll_video"], [t for _i, t, _b in picks])):
            if fr is None:
                continue
            name = f"crops/{piece}_{i}"
            cv2.imwrite(str(out / "review" / f"{name}.jpg"), crop(fr, box, TILE_HEIGHT, 1.6, 1.12), [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(out / "review" / f"{name}_ctx.jpg"), crop(fr, box, CONTEXT_HEIGHT, 4.0, 2.2), [cv2.IMWRITE_JPEG_QUALITY, 85])
            crops_.append({"tile": f"{name}.jpg", "context": f"{name}_ctx.jpg", "time": round(to_game(t), 1)})
            kp = _pose.keypoints(fr, [box])[0]
            score = face_score(kp, box)
            if best is None or score > best[0]:
                best = (score, head_photo(fr, kp, box))
            if (c := look(fr, kp, box)) is not None:
                looks.append({"weight": round(score, 1), **c})
        if best:
            cv2.imwrite(str(out / "review" / "avatars" / f"{piece}.jpg"), best[1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        entries[piece] = {
            "id": piece, "start": round(to_game(items[0][1]), 1), "end": round(to_game(items[-1][1]), 1),
            "seconds": round(len(items) / fps, 1), "number": old.get("number") if piece == tid else None,
            "readings": old.get("readings", 0) if piece == tid else 0, "crops": crops_,
            "avatar": f"avatars/{piece}.jpg" if best else None, "face": round(best[0], 1) if best else 0, "looks": looks,
        }
    owner["tracks"] = [entries[tid] if t["id"] == tid else t for t in owner["tracks"]] + [entries[new_id]]
    cards_file.write_text(json.dumps(cards, ensure_ascii=False))
    return {"tracks": list(entries.values()), "new_id": new_id}


# ---------- 网页服务 ----------

def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = HTTPStatus.OK) -> None:
            self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def _body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _file(self, path: Path, root: Path) -> None:
            path, root = path.resolve(), root.resolve()
            if root not in path.parents or not path.is_file():
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            size = path.stat().st_size
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):  # 视频拖动进度条要分段读取（Safari 必须）
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = min(size - 1, int(b)) if a and b else size - 1
                with path.open("rb") as f:
                    f.seek(start)
                    body = f.read(end - start + 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-cache")  # 片段会重新渲染（同名覆盖），别播缓存里的旧版
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(path.read_bytes(), ctype)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            parts = [unquote(p) for p in path.split("/") if p]  # 文件名里的 # 等字符浏览器会转成 %23
            try:
                if not parts or parts == ["index.html"]:
                    self._send((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts[0] == "static":
                    self._file(WEB_DIR / "/".join(parts[1:]), WEB_DIR)
                elif parts == ["api", "games"]:
                    self._json(list_games())
                elif parts == ["api", "sources"]:
                    self._json(find_sources())
                elif parts[:2] == ["api", "games"] and len(parts) == 3:
                    game = load_game(parts[2])
                    self._json({**game, "steps": steps(game)})
                elif parts[:2] == ["api", "games"] and parts[3:] == ["frame.jpg"]:
                    self._file(Path(frame_file(load_game(parts[2]))), OUTPUT)
                elif parts[:2] == ["api", "games"] and parts[3:] == ["court"]:
                    f = court_file(load_game(parts[2])["full_video"])
                    self._json(json.loads(f.read_text()) if f.exists() else None)
                elif parts[:2] == ["api", "games"] and parts[3:] == ["people"]:
                    review = out_dir(load_game(parts[2])["enroll_video"]) / "review"
                    people = review / "people.json"
                    self._json({"cards": json.loads((review / "cards.json").read_text()),
                                "people": json.loads(people.read_text()) if people.exists() else None,
                                "carry": None if people.exists() else carry_roster(load_game(parts[2]))})
                elif parts[:2] == ["api", "games"] and parts[3:] == ["questions"]:
                    review = out_dir(load_game(parts[2])["full_video"]) / "review"
                    answers = review / "answers.json"
                    self._json({"questions": json.loads((review / "questions.json").read_text()),
                                "answers": json.loads(answers.read_text()) if answers.exists() else None})
                elif parts[:2] == ["api", "games"] and parts[3:] == ["balls"]:
                    bdir = balls_dir(load_game(parts[2]))
                    labels = bdir / "labels.json"
                    self._json({"items": json.loads((bdir / "items.json").read_text())["items"],
                                "labels": json.loads(labels.read_text())["labels"] if labels.exists() else {}})
                elif parts[:2] == ["api", "games"] and parts[3:] == ["events"]:
                    game = load_game(parts[2])
                    out = out_dir(game["full_video"])
                    review = out / "review" / "events_review.json"
                    final = out / "final.json"
                    people = json.loads(final.read_text()).get("people", {}) if final.exists() else {}
                    pfile = out_dir(game["enroll_video"]) / "review" / "people.json"
                    teams = json.loads(pfile.read_text()).get("teams", {}) if pfile.exists() else {}
                    clips = game_dir(game["name"]) / "clips"
                    from bballgoal.reels import touch_segments

                    ball = out / "ball.json"
                    possessions = json.loads(ball.read_text())["possessions"] if ball.exists() else []
                    touches = {k: len(touch_segments(possessions, k)) for k in people}
                    self._json({"events": json.loads((out / "events.json").read_text())["events"],
                                "touches": touches, "final_score": game.get("final_score"),
                                "review": json.loads(review.read_text()) if review.exists() else {"events": {}, "added": []},
                                "people": people, "teams": teams,
                                "clips": sorted(int(p.stem) for p in clips.glob("*.mp4") if p.stem.isdigit()) if clips.exists() else []})
                elif parts[:2] == ["api", "games"] and len(parts) > 4 and parts[3] in ("clips", "reels"):
                    root = game_dir(load_game(parts[2])["name"])
                    self._file(root / parts[3] / "/".join(parts[4:]), root / parts[3])
                elif parts[:2] == ["api", "games"] and len(parts) > 4 and parts[3] == "ballfiles":
                    bdir = balls_dir(load_game(parts[2]))
                    self._file(bdir / "/".join(parts[4:]), bdir)
                elif parts[:2] == ["api", "games"] and len(parts) > 4 and parts[3] in ("enroll", "full"):
                    game = load_game(parts[2])
                    review = out_dir(game["enroll_video" if parts[3] == "enroll" else "full_video"]) / "review"
                    self._file(review / "/".join(parts[4:]), review)
                else:
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            except FileNotFoundError:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parts = [unquote(p) for p in self.path.split("?")[0].split("/") if p]
            try:
                body = self._body()
                if parts == ["api", "games"]:
                    game = new_game(body.get("name", ""), body["sources"])
                    start_job(game["name"], "import")
                    self._json(game)
                    return
                game = load_game(parts[2])
                action = parts[3]
                if action == "court":
                    for video in (game["full_video"], game["enroll_video"]):
                        court_file(video).write_text(json.dumps({"polygon": body["polygon"], "hoops": body["hoops"]}))
                    start_job(game["name"], "enroll")
                    start_job(game["name"], "full")
                elif action == "people":
                    review = out_dir(game["enroll_video"]) / "review"
                    body["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    tmp = review / "people.json.tmp"
                    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1))
                    tmp.replace(review / "people.json")
                    self._json({"ok": True, "saved_at": body["saved_at"]})
                    return
                elif action == "strip":
                    self._json(track_strip(game, int(body["track"])))
                    return
                elif action == "split":
                    self._json(split_track(game, int(body["track"]), int(body["frame"])))
                    return
                elif action == "suggest":
                    self._json(suggest(game, body["people"], [int(t) for t in body["tracks"]]))
                    return
                elif action == "answers":
                    review = out_dir(game["full_video"]) / "review"
                    body["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    tmp = review / "answers.json.tmp"
                    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1))
                    tmp.replace(review / "answers.json")
                    self._json({"ok": True, "saved_at": body["saved_at"]})
                    return
                elif action == "assign":
                    start_job(game["name"], "assign")
                elif action == "balls":
                    bdir = balls_dir(game)
                    tmp = bdir / "labels.json.tmp"
                    tmp.write_text(json.dumps({"labels": body["labels"], "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
                    tmp.replace(bdir / "labels.json")
                    self._json({"ok": True, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                    return
                elif action == "balltrain":
                    start_job(game["name"], "balltrain")
                elif action == "events":
                    start_job(game["name"], "events")
                elif action == "final_score":  # 真实比分（记分牌 / 记录台），集锦片头片尾用它
                    game["final_score"] = {k: int(v) for k, v in body.items() if k in ("A", "B") and str(v).strip() != ""} or None
                    save_game(game)
                elif action == "events_review":
                    out = out_dir(game["full_video"]) / "review"
                    out.mkdir(parents=True, exist_ok=True)
                    body["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    tmp = out / "events_review.json.tmp"
                    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1))
                    tmp.replace(out / "events_review.json")
                    self._json({"ok": True, "saved_at": body["saved_at"]})
                    return
                elif action == "clip":
                    from bballgoal.clips import ensure_clip, load_context
                    video = Path(game["full_video"])
                    gdir, result, people, teams, data = load_context(video)
                    event = body["event"]
                    clips = gdir / "clips"
                    clips.mkdir(exist_ok=True)
                    ensure_clip(gdir, video, result, event, people, teams, data.get("rims", []), force=True)
                    self._json({"ok": True})
                    return
                elif action == "reel":
                    def one(b: dict) -> dict:
                        return {"player": b.get("player", ""), "team": b.get("team"), "label": b.get("label", ""), "misses": bool(b.get("misses")),
                                "touches": bool(b.get("touches")), "lang": b.get("lang", "zh"), "music": b.get("music", True) is not False}
                    game["reel_request"] = {"batch": [one(b) for b in body["batch"]]} if body.get("batch") else one(body)
                    save_game(game)
                    stop_job(game["name"], "reel")
                    time.sleep(0.3)
                    start_job(game["name"], "reel")
                elif action == "restart":
                    job = body["job"]
                    stop_job(game["name"], job)
                    time.sleep(0.5)
                    start_job(game["name"], job)
                else:
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                self._json({**game, "steps": steps(game)})
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


# ---------- 后台任务用的小命令 ----------

def copy_with_progress(src: str, dst: str) -> None:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size == os.path.getsize(src):
        print("已经复制过了")
        return
    total = os.path.getsize(src)
    done, last = 0, -1
    tmp = dst_path.with_suffix(".part")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        while chunk := fin.read(64 << 20):
            fout.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total)
            if pct != last:
                print(f"  转换 {pct}%（复制）", flush=True)
                last = pct
    tmp.replace(dst_path)


def grab_frame(name: str) -> None:
    """导入后截一帧给你画场地：360 素材按定好的镜头展开；取比赛中段（人在场上）。"""
    game = load_game(name)
    out = frame_file(game)
    if game["raw"]:
        from bballgoal.fisheye import grab, probe, v360
        view = json.loads(Path(game["view"]).read_text())
        src = game["sources"][0]
        t = float(probe(Path(src))["format"]["duration"]) * 0.4
        vf = f"{v360(view.get('lens_fov', 200), 5760)},crop={view['w']}:{view['h']}:{view['x']}:{view['y']}"
        import cv2
        cv2.imwrite(out, grab(Path(src), t, view["lens"], vf), [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        t = _duration(game["full_video"]) * 0.4
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.1f}", "-i", game["full_video"], "-frames:v", "1", out], check=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "runjob":
        return run_job(sys.argv[2], sys.argv[3])
    if len(sys.argv) > 1 and sys.argv[1] == "copy":
        copy_with_progress(sys.argv[2], sys.argv[3])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "move":
        Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(sys.argv[2], sys.argv[3])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "frame":
        grab_frame(sys.argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true", help="允许同一局域网的其他设备打开")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), make_handler())
    url = f"http://localhost:{args.port}/"
    print(f"bballgoal：{url}")
    if args.lan:
        print(f"同一 Wi-Fi 的其他设备：http://{socket.gethostbyname(socket.gethostname())}:{args.port}/")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
