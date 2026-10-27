"""
tests/test_routers_admin_index.py — admin Index router contract.

The router is a thin layer over `IndexerRunner` + `IndexDB`. We test
that it:
  - surfaces the runner state on /status and /log
  - starts a job via POST /
  - rejects concurrent starts with 409
  - cancels a running job
  - wipes the SQLite side store on rebuild mode
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from search.index_db import IndexDB
from search.indexer_runner import IndexerRunner
from search.routers.admin_index import build_admin_index_router


SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import json, sys, time, signal

    _stop = False
    def _on_term(sig, frame):
        global _stop
        _stop = True
    signal.signal(signal.SIGTERM, _on_term)

    sys.stdout.write(json.dumps({{"event": "start"}}) + "\\n")
    sys.stdout.flush()

    seconds = int({seconds})
    for i in range(seconds):
        if _stop:
            sys.stdout.write(json.dumps({{"event": "cancelled"}}) + "\\n")
            sys.stdout.flush()
            sys.exit(130)
        sys.stdout.write(json.dumps({{
            "event": "batch",
            "indexed": (i + 1) * 10,
            "reembedded": 0,
            "skipped": 0,
            "errors": 0,
        }}) + "\\n")
        sys.stdout.flush()
        time.sleep(1.0)

    sys.stdout.write(json.dumps({{"event": "done"}}) + "\\n")
    sys.stdout.flush()
    sys.exit(0)
    """
)


def _write_fake_indexer(tmp_path: Path, *, seconds: int) -> Path:
    script = tmp_path / "fake_indexer.py"
    script.write_text(SCRIPT_TEMPLATE.format(seconds=seconds))
    return script


@pytest.fixture
def client(tmp_path: Path):
    script = _write_fake_indexer(tmp_path, seconds=0)
    factory = lambda mode: [sys.executable, str(script)]
    runner = IndexerRunner(command_factory=factory)

    # In-memory IndexDB so reset_side_store has somewhere to wipe.
    # We don't exercise Qdrant paths in these tests, so the client
    # can be None (the router only calls reset_side_store, which
    # never touches Qdrant).
    db = IndexDB(":memory:", qdrant_client=None)  # type: ignore[arg-type]

    app = FastAPI()
    app.include_router(build_admin_index_router(
        indexer_runner=runner, index_db=db,
    ))
    return TestClient(app), runner, db


def test_initial_status_is_idle(client):
    http, _runner, _db = client
    r = http.get("/api/admin/index/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["job_id"] is None
    # Round-50: thumb_regens added to the progress model. Default 0
    # is the same as the rest of the progress counters on init.
    assert body["progress"] == {
        "indexed": 0,
        "reembedded": 0,
        "skipped": 0,
        "errors": 0,
        "thumb_regens": 0,
    }


def test_post_starts_job_and_returns_202(client):
    http, runner, _db = client
    r = http.post("/api/admin/index", json={"mode": "incremental"})
    assert r.status_code == 202
    body = r.json()
    assert body["mode"] == "incremental"
    assert body["state"] in ("running", "idle")  # very fast jobs may finish by then
    assert body["job_id"] is not None
    runner.wait_idle(timeout=10)


def test_post_rebuild_wipes_sqlite_side_store(client):
    """A rebuild clears the side store's user state.

    The side store holds only user state now — favourites, dislikes,
    albums, memberships, saved searches, feedback. A rebuild replaces
    the collection's point ids, so the rows that referenced the old
    ids are cleared first rather than left dangling.
    """
    http, _runner, db = client
    # Seed a favourite + a saved search so we can prove the wipe
    # cleared them. The row is written directly because the fixture
    # has no Qdrant to check liveness against.
    db._conn.execute(
        "INSERT INTO favorites (id, favorited_at) VALUES (?, ?)",
        ("deadbeef", "2026-01-01T00:00:00+00:00"),
    )
    db._conn.commit()
    db.create_saved_search(name="trip", positives=["beach"], negatives=[])
    assert db.count_favorites() >= 1

    r = http.post("/api/admin/index", json={"mode": "rebuild"})
    assert r.status_code == 202
    _runner.wait_idle(timeout=10)
    # After rebuild, the side store should be empty.
    assert db.count_favorites() == 0


def test_concurrent_start_returns_409(client):
    http, runner, _db = client
    # Easier: start with the short fixture, but the job finishes too
    # fast. We'll instead check the IndexConflictError by directly
    # faking the runner — the route-level test is below.
    runner.start("incremental")
    try:
        r = http.post("/api/admin/index", json={"mode": "incremental"})
        assert r.status_code == 409
        assert "already running" in r.json()["detail"].lower()
    finally:
        runner.cancel()
        runner.wait_idle(timeout=5)


def test_cancel_running_job(client):
    http, runner, _db = client
    long_script_path = Path("/tmp") / "admin_test_long_indexer.py"  # noqa: S108 — test scratch dir
    long_script_path.write_text(SCRIPT_TEMPLATE.format(seconds=30))
    runner_long = IndexerRunner(
        command_factory=lambda mode: [sys.executable, str(long_script_path)],
    )
    # Swap the runner the router uses by recreating the app with the
    # long runner. (Constructing a fresh router would require more
    # wiring; this is the smallest test.)
    from fastapi import FastAPI
    from search.routers.admin_index import build_admin_index_router
    app = FastAPI()
    app.include_router(build_admin_index_router(
        indexer_runner=runner_long, index_db=None,
    ))
    http2 = TestClient(app)
    http2.post("/api/admin/index", json={"mode": "incremental"})
    assert runner_long.status().state.value == "running"
    r = http2.post("/api/admin/index/cancel")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_cancel_when_idle_returns_400(client):
    http, _runner, _db = client
    r = http.post("/api/admin/index/cancel")
    assert r.status_code == 400


def test_log_endpoint_returns_lines(client):
    http, runner, _db = client
    runner.start("incremental")
    runner.wait_idle(timeout=10)
    r = http.get("/api/admin/index/log")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["lines"], list)
    assert body["total"] >= 1
    assert any('"event"' in ln for ln in body["lines"])


def test_log_since_line_resumes(client):
    http, runner, _db = client
    runner.start("incremental")
    runner.wait_idle(timeout=10)
    first = http.get("/api/admin/index/log").json()
    second = http.get(f"/api/admin/index/log?since_line={first['next_line']}").json()
    assert second["lines"] == []


def test_post_invalid_mode_returns_422(client):
    http, _runner, _db = client
    r = http.post("/api/admin/index", json={"mode": "wat"})
    assert r.status_code == 422
