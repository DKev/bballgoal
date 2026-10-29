"""个人集锦：某个孩子确认过的投篮（进球和罚篮；没进的可选），片头写名字、数据和全场比分，每段开头有游戏式的定格介绍，结尾定格打出本场数据，
片段都是虚拟镜头（bballgoal.clips）。

    .venv/bin/python -m bballgoal.reels output/<比赛>/full.mp4 <人的 id> [--misses]
    .venv/bin/python -m bballgoal.reels output/<比赛>/full.mp4 <人的 id> --touches   # 触球集锦：每次拿到球
    .venv/bin/python -m bballgoal.reels output/<比赛>/full.mp4 --team A              # 球队五佳球（team_A_top5.mp4）
输出 output/<比赛>/reels/<人>.mp4（触球集锦是 <人>_touches.mp4）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bballgoal.clips import (FPS, ensure_clip, font, load_context, mark_at, plan_for, player_text, shot_kind, shot_points,
                             shot_text)
from bballgoal.intro import OUTRO_BLACK, OUTRO_FRAMES, make_outro, make_segment
from bballgoal.music import write_music
from bballgoal.paths import out_dir as out_dir_for
from bballgoal.vcam import OUT_H, OUT_W, VideoWriter

TITLE_SECONDS = 2.5
HIGHLIGHT_TYPES = ("shot",)
TOP_N = 5  # 五佳球
TOUCH_MIN = 0.4  # 秒：持球至少这么久才算拿到球（球从身边飞过不算）
TOUCH_MERGE = 4.0  # 秒：两次拿球相隔不到这么久，连成一段
TOUCH_BEFORE = 2.0  # 秒：触球片段从拿球前这么久开始
TOUCH_AFTER = 1.5  # 秒：拿完球之后再拍这么久


def reviewed_events(events: list[dict], review: dict) -> list[dict]:
    """自动找的事件 + 你的修改 / 删除 / 补充。"""
    edits = review.get("events", {})
    out = []
    for e in events:
        r = edits.get(str(e["id"]), {})
        if r.get("status") == "deleted":
            continue
        out.append({**e, **{k: v for k, v in r.items() if k not in ("status",)}, "status": r.get("status", "auto")})
    for e in review.get("added", []):
        if e.get("status") != "deleted":
            out.append(e)
    out = [e for e in out if e["type"] in HIGHLIGHT_TYPES]  # 以前找过的篮板、助攻、抢断不要了
    return sorted(out, key=lambda e: e["time"])


def stats(events: list[dict], key=lambda e: e.get("player")) -> dict[str, dict]:
    """每人（或者 key 给的分组，比如球队）的数据：投篮（两分，fga/fgm）、三分（tpa/tpm）、罚篮（fta/ftm）分开算。"""
    s: dict[str, dict] = defaultdict(lambda: {"points": 0, "fga": 0, "fgm": 0, "tpa": 0, "tpm": 0, "fta": 0, "ftm": 0})
    for e in events:
        p = key(e)
        if not p:
            continue
        if e["type"] == "shot":
            a, m = {"two": ("fga", "fgm"), "three": ("tpa", "tpm"), "freethrow": ("fta", "ftm")}[shot_kind(e)]
            s[p][a] += 1
            if e.get("result") == "made":
                s[p][m] += 1
                s[p]["points"] += shot_points(e)
    return dict(s)


def team_of_event(e: dict, people: dict) -> str | None:
    """事件算哪队：认出了人就按人；没认出来用事件自己的 team（自动按篮筐推断的，或者你审核时选的）。"""
    p = people.get(e.get("player") or "")
    return (p.get("team") if p else None) or e.get("team") or None


def score_line(events: list[dict], people: dict, teams: dict, final: dict | None = None, lang: str = "zh") -> str:
    """全场比分。你在事件审核里填了真实比分（final）就用它，否则按确认过的投篮算。"""
    t = {k: v["points"] for k, v in stats(events, key=lambda e: team_of_event(e, people)).items()}
    if final:
        t = {k: int(v) for k, v in final.items()}
    names = [k for k in ("A", "B") if k in teams or k in t]
    return "  :  ".join(f"{team_name(k, teams, lang)} {t.get(k, 0)}" for k in names)


def touch_segments(possessions: list[dict], player: str) -> list[tuple[float, float]]:
    """这个人每次拿球（持球至少 TOUCH_MIN 秒），相隔不到 TOUCH_MERGE 秒的连成一段。返回 [(开始, 结束)]。"""
    segs: list[list[float]] = []
    for p in sorted((p for p in possessions if p["player"] == player and p["end"] - p["start"] >= TOUCH_MIN),
                    key=lambda p: p["start"]):
        if segs and p["start"] - segs[-1][1] < TOUCH_MERGE:
            segs[-1][1] = max(segs[-1][1], p["end"])
        else:
            segs.append([p["start"], p["end"]])
    return [(a, b) for a, b in segs]


TEXT = {  # 集锦里的文字（--lang）。队名、孩子名字用你在名单里填的，不翻译
    "zh": {"points": "{} 分", "two": "投篮", "three": "三分", "freethrow": "罚篮", "made": "命中", "missed": "没进",
           "unsure": "进没进拿不准", "pts": "{} 分", "touch": "拿球", "touch_title": "触球集锦 · 拿球 {} 次",
           "touch_count": "拿球 {} 次", "final": "全场比分", "team": "{}队", "closing": "{} 加油！",
           "top_title": "本场五佳球", "rank": "第 {} 名", "team_points": "全队得分 {}"},
    "en": {"points": "{} PTS", "two": "FG", "three": "3PT", "freethrow": "FT", "made": "Made", "missed": "Missed",
           "unsure": "Unclear", "pts": "{} pts", "touch": "Touch", "touch_title": "Every Touch · {}",
           "touch_count": "{} touches", "final": "Final", "team": "Team {}", "closing": "Go {}!",
           "top_title": "Top {} Plays", "rank": "No.{}", "team_points": "Team points {}"},
}
KIND_LONG = {"zh": {"two": "投篮", "three": "三分", "freethrow": "罚篮"}, "en": {"two": "Field goal", "three": "3-pointer", "freethrow": "Free throw"}}


def stat_line(s: dict, lang: str = "zh") -> str:
    """“7 分 · 投篮 2/5 · 三分 1/2 · 罚篮 0/1”（没出过手的那项不写）。"""
    tx = TEXT[lang]
    parts = [tx["points"].format(s.get("points", 0)), f"{tx['two']} {s.get('fgm', 0)}/{s.get('fga', 0)}"]
    parts += [f"{tx['three']} {s['tpm']}/{s['tpa']}"] if s.get("tpa") else []
    parts += [f"{tx['freethrow']} {s['ftm']}/{s['fta']}"] if s.get("fta") else []
    return " · ".join(parts)


def shot_label(e: dict, lang: str = "zh") -> str:
    """“投篮 · 命中 · 2 分” / “Field goal · Made · 2 pts”。"""
    if e["type"] == "touch":
        return TEXT[lang]["touch"]
    kind = shot_kind(e)
    parts = [KIND_LONG[lang][kind], TEXT[lang].get(e.get("result") or "", "")]
    if e.get("result") == "made":
        parts.append(TEXT[lang]["pts"].format(shot_points(e)))
    return " · ".join(x for x in parts if x)


def team_name(code: str, teams: dict, lang: str = "zh") -> str:
    return teams.get(code, {}).get("name") or (TEXT[lang]["team"].format(code) if code else "")


def name_text(label: str, people: dict, teams: dict, lang: str = "zh") -> str:
    """“MSK Hawks #18 小明”；名单里没有的就用队名 + 号码。"""
    title, team = tag_text(label, people, teams, lang)
    return " ".join(x for x in (team, title) if x) or player_text(label, people, teams)


def tag_text(label: str, people: dict, teams: dict, lang: str = "zh") -> tuple[str, str]:
    """名牌上的两行：大字“#18 名字”，小字队名。"""
    p = people.get(label) or {}
    number = p.get("number") or (label.split("#")[1] if "#" in label else "")
    title = " ".join(x for x in ((f"#{number}" if number else ""), p.get("name") or "") if x) or label
    code = p.get("team") or (label[:1] if label[:1] in ("A", "B") else "")
    return title, team_name(code, teams, lang)


def title_card(path: Path, name: str, line: str, score: str = "") -> None:
    img = Image.new("RGB", (OUT_W, OUT_H), (16, 20, 28))
    draw = ImageDraw.Draw(img)
    big, mid = font(96), font(48)
    draw.text((OUT_W / 2, OUT_H / 2 - 70), name, font=big, fill=(255, 255, 255), anchor="mm")
    draw.text((OUT_W / 2, OUT_H / 2 + 60), line, font=mid, fill=(250, 204, 21), anchor="mm")
    if score:
        draw.text((OUT_W / 2, OUT_H / 2 + 150), score, font=font(36), fill=(170, 178, 190), anchor="mm")
    frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    writer = VideoWriter(path, FPS, enhance=False)
    for _ in range(int(TITLE_SECONDS * FPS)):
        writer.write(frame)
    writer.close()


def top_plays(events: list[dict], team: str, people: dict, n: int = TOP_N) -> list[dict]:
    """一个队本场最好的 n 个球，按倒计时顺序（第 n 名在前，第 1 名最后）。
    进球优先，分值高的优先；同样条件尽量换不同的孩子；进球不够用拿不准的、再用没进的补。"""
    rank = {"made": 2, "unsure": 1}
    shots = [e for e in events if e["type"] == "shot" and team_of_event(e, people) == team]
    shots.sort(key=lambda e: (rank.get(e.get("result"), 0), shot_points(e), e.get("player") is not None), reverse=True)
    picked, seen = [], set()
    for tier in (2, 1, 0):  # 进球 → 拿不准 → 没进；同一档里先每个孩子上一个，再补
        group = [e for e in shots if rank.get(e.get("result"), 0) == tier]
        for e in [e for e in group if e.get("player") is None or e["player"] not in seen] + group:
            if len(picked) < n and e not in picked:
                picked.append(e)
                seen.add(e.get("player"))
    picked.sort(key=lambda e: (rank.get(e.get("result"), 0), shot_points(e), e.get("player") is not None, -e["time"]), reverse=True)
    return picked[::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("player", nargs="?", default="", help="人的 id（--team 时不用）")
    parser.add_argument("--team", default=None, help="球队五佳球：A 或 B")
    parser.add_argument("--misses", action="store_true", help="也放没进的出手")
    parser.add_argument("--touches", action="store_true", help="触球集锦：他每次拿到球（整场没投篮的孩子也有）")
    parser.add_argument("--lang", choices=sorted(TEXT), default="zh", help="集锦里文字的语言")
    parser.add_argument("--no-music", action="store_true", help="不要背景音乐")
    args = parser.parse_args()
    game_dir, result, people, teams, data = load_context(args.video)
    game = json.loads((game_dir / "game.json").read_text())
    review_file = out_dir_for(args.video) / "review" / "events_review.json"
    review = json.loads(review_file.read_text()) if review_file.exists() else {}
    events = reviewed_events(data["events"], review)
    lang, tx = args.lang, TEXT[args.lang]
    score = f"{tx['final']}  {score_line(events, people, teams, game.get('final_score'), lang)}"
    safe = re.sub(r"[^\w#\-]+", "_", args.player)
    ranks: list[int] = []
    if args.team:
        mine = top_plays(events, args.team, people)
        ranks = list(range(len(mine), 0, -1))
        team_full = team_name(args.team, teams, lang)
        t = stats(events, key=lambda e: team_of_event(e, people)).get(args.team, {})
        points = (game.get("final_score") or {}).get(args.team, t.get("points", 0))
        name, card_line, out_name, seed = team_full, tx["top_title"].format(len(mine)), f"team_{args.team}_top5", f"team{args.team}"
        end_title, end_team, end_lines = team_full, "", [tx["team_points"].format(points), score]
    else:
        if args.touches:
            ball = json.loads((out_dir_for(args.video) / "ball.json").read_text())
            mine = [{"id": f"touch-{safe}-{a:.1f}", "type": "touch", "time": b, "player": args.player,
                     "clip": [round(a - TOUCH_BEFORE, 2), round(b + TOUCH_AFTER, 2)]}
                    for a, b in touch_segments(ball["possessions"], args.player)]
        else:
            mine = [e for e in events if e.get("player") == args.player and e["type"] in HIGHLIGHT_TYPES
                    and (e["type"] != "shot" or e.get("result") == "made" or args.misses)]
        s = stats(events).get(args.player, {})
        name = name_text(args.player, people, teams, lang)
        line = stat_line(s, lang)
        card_line = tx["touch_title"].format(len(mine)) if args.touches else line
        out_name, seed = f"{safe}{'_touches' if args.touches else ''}", f"{args.player}{args.touches}"
        end_title, end_team = tag_text(args.player, people, teams, lang)
        end_lines = [f"{tx['touch_count'].format(len(mine))} · {line}" if args.touches else line, score]
    if not mine:
        print("这个队还没有投篮" if args.team else "这个人没有拿到过球" if args.touches else "这个人还没有确认过的精彩事件")
        return 1
    reels = game_dir / "reels"
    reels.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "title.mp4"
        title_card(card, name, card_line, score)
        parts = [card]
        for k, e in enumerate(mine):
            clip = ensure_clip(game_dir, args.video, result, e, people, teams, data.get("rims", []))
            # 游戏式开场：定格、脚下画圈、头顶拉线、打字机打出名字，再接着放；之后光圈一直跟着他
            shot, view, t0, t1, track = plan_for(game_dir, args.video, result, e, data.get("rims", []))
            marks = [mark_at(result, e, shot, view, t0 + i / FPS, track=track) for i in range(int((t1 - t0) * FPS) + 1)]
            who = e.get("player")
            title, team = tag_text(who, people, teams, lang) if who else (team_name(team_of_event(e, people) or "", teams, lang), "")
            sub = shot_label(e, lang)
            if ranks:  # 五佳球：倒计时名次
                sub = f"{tx['rank'].format(ranks[k])} · {sub}"
            who_text = name_text(who, people, teams, lang) if who else title
            seg = Path(tmp) / f"{k}.mp4"
            # 触球片段多而短：只在第一段定格介绍，后面每段只有光圈跟着
            make_segment(clip, marks, title, team, sub, seg, intro=not args.touches or k == 0, caption=f"{who_text} · {sub}")
            parts.append(seg)
            print(f"  {k + 1}/{len(mine)} 段", flush=True)
        # 结尾：最后一个镜头定格、虚化，打出名字、本场数据和全场比分，淡出到黑
        ending = Path(tmp) / "ending.mp4"
        closing_team = team_name(args.team, teams, lang) if args.team else end_team
        make_outro(parts[-1], end_title, end_team, end_lines, ending, closing=tx["closing"].format(closing_team) if closing_team else "")
        parts.append(ending)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
        out = reels / f"{out_name}.mp4"
        silent = Path(tmp) / "silent.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                        str(silent)], check=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(silent)]
        if not args.no_music:  # 现场合成的背景音乐：片头前奏、正片鼓点、结尾跟着画面淡出
            total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(silent)],
                                         capture_output=True, text=True, check=True).stdout)
            music = Path(tmp) / "music.wav"
            write_music(music, total, TITLE_SECONDS, (OUTRO_FRAMES + OUTRO_BLACK) / FPS, seed)
            cmd += ["-i", str(music), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        subprocess.run(cmd + ["-c:v", "copy", "-movflags", "+faststart", str(out)], check=True)
    out.with_suffix(".json").write_text(json.dumps({"player": args.player, "team": args.team, "name": name, "touches": args.touches,
                                                    "lang": lang, "events": [e["id"] for e in mine]}, ensure_ascii=False))
    print(f"{'五佳球' if args.team else '触球集锦' if args.touches else '集锦'}：{name}，{len(mine)} 段 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
