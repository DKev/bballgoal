"""Build a labeled evaluation set from app review results and score verdicts.

Each project's review status is treated as ground truth: ``excluded`` means
"not a goal", every other status means "goal".
"""
import argparse
import json
import sqlite3
from pathlib import Path


def load_projects(roots):
    cases = []
    for db in sorted(path for root in roots for path in Path(root).glob("*/project.db")):
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        video = con.execute("select source_path, fps from videos limit 1").fetchone()
        roi = con.execute(
            "select x1, y1, x2, y2, calibration_json from rois where is_active=1"
        ).fetchone()
        calibration = json.loads(roi["calibration_json"] or "{}") if roi else {}
        # Automatic exclusions are the engine's own decision, not a label.
        rows = con.execute(
            "select c.id, c.event_time_ms, c.evidence_json, coalesce(r.status, 'pending') status "
            "from candidates c left join candidate_reviews r on r.candidate_id = c.id "
            "where coalesce(r.reason, '') != 'auto_rim_check' "
            "order by c.event_time_ms"
        ).fetchall()
        for row in rows:
            evidence = json.loads(row["evidence_json"] or "{}")
            overlay = evidence.get("overlay") or {}
            cases.append({
                "project": db.parent.name,
                "video": video["source_path"],
                "fps": video["fps"],
                "candidate_id": row["id"],
                "event_ms": row["event_time_ms"],
                "goal": row["status"] != "excluded",
                "status": row["status"],
                "verdict": evidence.get("verdict") or "coarse_fallback",
                "complete_crossing": evidence.get("complete_crossing"),
                "rim": overlay.get("rim"),
                "net_roi": calibration.get("net_roi"),
                "analysis_roi": [roi["x1"], roi["y1"], roi["x2"], roi["y2"]] if roi else None,
            })
    return cases


def score(cases, keep):
    tp = sum(1 for case in cases if keep(case) and case["goal"])
    fp = sum(1 for case in cases if keep(case) and not case["goal"])
    fn = sum(1 for case in cases if not keep(case) and case["goal"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"kept": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3)}


BASELINE_RULES = {
    "current_default(all kept)": lambda case: True,
    "made_only": lambda case: case["verdict"] == "made",
    "made_or_ambiguous": lambda case: case["verdict"] in {"made", "ambiguous"},
    "complete_crossing": lambda case: case["complete_crossing"] is True,
}


def report(cases, rules):
    groups = {"all": cases}
    for case in cases:
        groups.setdefault(case["project"], []).append(case)
    return {
        name: {rule: score(group, keep) for rule, keep in rules.items()}
        for name, group in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="+",
                        default=[str(Path.home() / "Movies/BasketballProjects")])
    parser.add_argument("--output", required=True, help="Write the labeled cases JSON here.")
    args = parser.parse_args()
    cases = load_projects(args.projects)
    Path(args.output).write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report(cases, BASELINE_RULES), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
