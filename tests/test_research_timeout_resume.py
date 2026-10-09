"""Forced Pi-timeout closure: FAILED/TIMEOUT persisted, resume stable with no dup."""

import subprocess
from pathlib import Path

import pytest

from app.research.repository import ResearchRepository
from app.research.runner import LiveModelError, run_live


def test_timeout_closes_failed_and_resume_has_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forced Pi-timeout closure: FAILED/TIMEOUT persisted, resume stable with no dup.

    Re-pinned to the per-assignment scout failure contract (live-run defect: one
    scout model timeout aborted the whole run). A provider that times out every
    scout call now degrades those assignments - the wave still closes its source
    job and the run dies on the committee's model call, which stays session-fatal.
    The closure contract this test owns is unchanged: FAILED/TIMEOUT persisted,
    every job closed, and resume never reopens or duplicates work.
    """
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))

    def _timeout_model(_prompt: str) -> str:
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1)

    with pytest.raises(LiveModelError) as excinfo:
        run_live("timeout probe?", "probe", None, ["NVDA"], lambda n, a: {}, _timeout_model)
    repo = ResearchRepository()
    sid = excinfo.value.session_id
    session = repo.get_session(sid)
    assert session.status == "failed"
    assert session.failure is not None and session.failure.category == "timeout"
    jobs = repo.list_jobs(sid)
    by_type = {j.job_type: j for j in jobs}
    scouts = [j for j in jobs if j.job_type == "scout"]
    assert len(scouts) == 3 and all(j.status == "failed" for j in scouts)
    assert {j.failure.category for j in scouts if j.failure is not None} == {"timeout"}
    assert by_type["source_agent"].status == "completed"  # the wave still closed its source job
    assert [j for j in jobs if j.status in ("queued", "running")] == []
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert kinds.count("scout.degraded") == 3
    assert "job.failed" in kinds and "model.failed" in kinds
    assert "research.failed" in kinds and "wave.stopped" in kinds
    assert repo.resume(sid).session.status == "failed"
    assert repo.resume(sid).open_job_ids == []
    assert len(repo.list_jobs(sid)) == len(jobs)  # resume reopens and duplicates nothing


def test_failed_run_bundle_still_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session-fatal model failure persists state before the bundle write."""
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))

    def _timeout_model(_prompt: str) -> str:
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1)

    def _noop(_name: str, _args: dict[str, object]) -> dict[str, object]:
        return {}

    with pytest.raises(LiveModelError) as excinfo:
        run_live("timeout probe?", "probe", None, ["NVDA"], _noop, _timeout_model)
    repo = ResearchRepository()
    sid = excinfo.value.session_id
    session = repo.get_session(sid)
    assert session.status == "failed"
    run = _LiveRun(
        repo,
        session.query,
        session.objective,
        None,
        "",
        ["NVDA"],
        _noop,
        _timeout_model,
        DirectorBudgets(),
    )
    bundle = run.write_bundle(sid)
    assert bundle.name == sid
    assert (bundle / "request.json").exists()
