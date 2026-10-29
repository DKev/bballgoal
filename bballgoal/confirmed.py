"""把认人网页里确认的结果（review/people.json）套到视频上：每个人显示确认的号码。

用法：
    .venv/bin/python -m bballgoal.confirmed videos/比赛.mp4
输出 output/track_<视频名>/confirmed.json 和 confirmed.mp4。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bballgoal.paths import out_dir as out_dir_for
from bballgoal.track import REFEREE, UNKNOWN, render


def person_label(person: dict) -> tuple[str | None, str]:
    """返回 (显示的编号, 队)。无关的人返回 (None, ...)，视频里不画。"""
    kind = person.get("kind", "player")
    if kind == "ignore":
        return None, UNKNOWN
    if kind == "ref":
        return "ref", REFEREE
    if kind == "coach":
        return f"{person.get('team') or ''} coach".strip(), UNKNOWN
    number = person.get("number") or ""
    name = person.get("name") or ""
    if not number and name.isdigit():  # 号码误填进名字栏
        number = name
    team = person.get("team") or UNKNOWN
    return f"{team}#{number}" if number else f"{team}?", team


def apply_people(result: dict, people: dict) -> dict[int, str]:
    labels: dict[int, str] = {}
    team_of: dict[int, str] = {}
    for person in people["people"]:
        label, team = person_label(person)
        for tid in person["tracks"]:
            if label:
                labels[tid] = label
            team_of[tid] = team
    for f in result["frames"]:
        f["people"] = [p for p in f["people"] if p["id"] in labels]
        for p in f["people"]:
            p["player"] = labels[p["id"]]
            p["team"] = team_of[p["id"]]
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or out_dir_for(args.video)
    result = json.loads((out_dir / "identified.json").read_text())
    people = json.loads((out_dir / "review" / "people.json").read_text())
    labels = apply_people(result, people)
    (out_dir / "confirmed.json").write_text(json.dumps(result, ensure_ascii=False))
    render(result, args.video, out_dir / "confirmed.mp4", labels=labels)
    names = sorted({v for v in labels.values()})
    print(f"已套用 {len(names)} 个确认的人：{' '.join(names)}")
    print(f"输出：{out_dir / 'confirmed.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
