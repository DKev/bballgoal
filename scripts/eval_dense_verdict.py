"""Score the production rim check on dense rim-crop detections.

Reads the output of ``rim_dense_probe.py`` and runs
``basketball_highlight.rim_check.assess_rim_crossing`` on every case, so the
evaluation always matches the code the engine runs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from basketball_highlight.rim_check import apply_video_occlusion, assess_rim_crossing  # noqa: E402
from eval_review_labels import report  # noqa: E402

RULES = {
    "rim_check_made": lambda case: case["rim_check"]["status"] == "made",
    "keep_unless_rejected": lambda case: case["rim_check"]["status"] != "rejected",
}


def probe_frames(case, source):
    return [
        [
            {"name": item["name"], "confidence": item["conf"], "xyxy": item["xyxy"]}
            for item in frame.get(source, [])
        ]
        for frame in case["frames"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--source", default="tight", choices=["tight", "wide"])
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    cases = json.loads(Path(args.dense).read_text(encoding="utf-8"))
    for case in cases:
        case["rim_check"] = assess_rim_crossing(probe_frames(case, args.source), case["rim"])
        case.pop("frames")
    for project in {case["project"] for case in cases}:
        group = [case for case in cases if case["project"] == project]
        for case, check in zip(group, apply_video_occlusion([case["rim_check"] for case in group])):
            case["rim_check"] = check
    for case in cases:
        if args.details:
            print(f"{'GOAL' if case['goal'] else 'miss'} {case['project'][:2]} "
                  f"{case['event_ms'] / 1000:7.2f} {case['verdict'][:5]:5s} {case['rim_check']}")
    print(json.dumps(report(cases, RULES), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
