import json
from pathlib import Path

import pytest

from engine.python.basketball_engine.protocol import ProtocolError
from engine.python.basketball_engine.service import EngineService
from engine.python.basketball_engine.storage import ProjectStore


@pytest.fixture
def project(tmp_path: Path):
    service = EngineService()
    root = tmp_path / "project"
    service.handle("create_project", {"name": "导出复核回归", "root_path": str(root)})
    store = ProjectStore(root)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    video = store.link_video({
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "duration_ms": 20_000,
        "width": 960,
        "height": 720,
    })
    rows = [{
        "id": f"candidate-{index}", "video_id": video["id"],
        "event_time_ms": index * 3000 + 1000,
        "default_start_ms": index * 3000,
        "default_end_ms": index * 3000 + 2000,
        "review_start_ms": index * 3000,
        "review_end_ms": index * 3000 + 2000,
        "detector_version": "test", "evidence_json": "{}",
    } for index in range(5)]
    store.replace_candidates(video["id"], rows)
    try:
        yield service, store, video, rows
    finally:
        service.shutdown()


@pytest.mark.parametrize("filter_kind,include_unassigned,original_indices,expected_indices", [
    ("selected", False, [0, 1], [0, 2]),
    ("selected", True, [0, 1, 4], [0, 2, 4]),
    ("empty", True, [4], [4]),
    ("empty", False, [], []),
    ("legacy", None, [0, 1, 3, 4], [0, 2, 3, 4]),
])
def test_retry_export_preserves_player_filter_with_latest_reviews(
    project, monkeypatch, filter_kind, include_unassigned, original_indices, expected_indices,
):
    service, store, video, rows = project
    player = store.create_player("#1")
    other_player = store.create_player("#2")
    for row in rows[:3]:
        store.set_candidate_player(row["id"], player["id"])
    store.set_candidate_player(rows[3]["id"], other_player["id"])
    store.review_candidate(rows[2]["id"], "excluded")
    checkpoint = {
        "mode": "merge",
        "output_path": str(store.root / "highlights.mp4"),
        "candidate_snapshot": [dict(rows[i]) for i in original_indices],
    }
    if filter_kind != "legacy":
        checkpoint["player_ids"] = [player["id"]] if filter_kind == "selected" else []
        checkpoint["include_unassigned"] = include_unassigned
    old_job = store.create_job(store.project()["id"], video["id"], "export", checkpoint)
    store.update_job(old_job["id"], state="failed", error_code="EXPORT_FAILED")
    store.review_candidate(rows[1]["id"], "excluded")
    store.review_candidate(rows[2]["id"], "goal")
    store.update_clip_range(rows[0]["id"], 500, 1500)
    captured = []

    def execute_export(payload, **_kwargs):
        captured.append(payload)
        return {"export": {"id": "export-test"}, "files": []}

    monkeypatch.setattr(service, "_execute_export", execute_export)
    payload = {"project_root": str(store.root), "job_id": old_job["id"]}
    if not expected_indices:
        with pytest.raises(ProtocolError) as error:
            service.handle("retry_export", payload)
        assert error.value.code == "NO_EXPORT_CANDIDATES"
        assert captured == []
        return
    started = service.handle("retry_export", payload)
    job_id = started["job"]["id"]
    with service._job_lock:
        thread = service._job_threads.get(job_id)
    if thread is not None:
        thread.join(timeout=2)
        assert not thread.is_alive()
    job = store.get_job(job_id)
    assert job["state"] == "completed"
    retried = json.loads(job["checkpoint_json"])
    snapshot = retried["candidate_snapshot"]
    assert [row["id"] for row in snapshot] == [rows[i]["id"] for i in expected_indices]
    assert captured[0]["candidate_snapshot"] == snapshot
    assert retried["player_ids"] == checkpoint.get("player_ids")
    assert retried["include_unassigned"] == checkpoint.get("include_unassigned", True)
    assert retried["output_path"] == checkpoint["output_path"]
    if 0 in expected_indices:
        assert (snapshot[0]["review_start_ms"], snapshot[0]["review_end_ms"]) == (500, 1500)


@pytest.mark.parametrize("manual_range,event_time_ms,duration_ms,expected", [
    (None, 8500, 20_000, (7000, 10_000)),
    ((8000, 9900), 8500, 20_000, (8000, 9900)),
    ((8000, 9500), 8500, 20_000, (8000, 9500)),
    ((6000, 9900), 8500, 20_000, (6000, 9900)),
    ((8000, 9900), 8500, 9000, (8000, 9000)),
    ((8000, 9900), 9500, 10_000, (9000, 10_000)),
    ((6000, 9000), 500, 20_000, (0, 1000)),
])
def test_replace_candidates_preserves_stable_id_manual_range(
    project, manual_range, event_time_ms, duration_ms, expected,
):
    _, store, video, rows = project
    old = {
        **rows[0], "event_time_ms": 8500,
        "default_start_ms": 6000, "default_end_ms": 9500,
        "review_start_ms": 6000, "review_end_ms": 9500,
    }
    store.replace_candidates(video["id"], [old])
    store.review_candidate(old["id"], "goal")
    if manual_range is not None:
        store.update_clip_range(old["id"], *manual_range)
    with store.connect() as connection:
        connection.execute("UPDATE videos SET duration_ms = ? WHERE id = ?", (duration_ms, video["id"]))
    new = {
        **old, "event_time_ms": event_time_ms,
        "default_start_ms": 7000, "default_end_ms": min(10_000, duration_ms),
        "review_start_ms": 7000, "review_end_ms": min(10_000, duration_ms),
    }
    store.replace_candidates(video["id"], [new])
    candidate = store.list_candidates(video["id"])[0]
    assert candidate["review_status"] == "goal"
    assert (candidate["review_start_ms"], candidate["review_end_ms"]) == expected
    assert (candidate["default_start_ms"], candidate["default_end_ms"]) == (7000, min(10_000, duration_ms))


def test_failed_retry_preflight_keeps_the_interrupted_export_job(project):
    service, store, video, rows = project
    job = store.create_job(store.project()["id"], video["id"], "export", {"mode": "merge"})
    store.update_job(job["id"], state="running", stage="export_clips")
    for row in rows:
        store.review_candidate(row["id"], "excluded")

    with pytest.raises(ProtocolError) as error:
        service.handle("retry_export", {"project_root": str(store.root), "job_id": job["id"]})

    assert error.value.code == "NO_EXPORT_CANDIDATES"
    assert store.get_job(job["id"])["state"] == "running"
