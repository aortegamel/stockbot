"""Thesis monitor API (api.theses_routes) over a temp data root."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.thesis.runner as runner_mod
from api import theses_routes
from app.thesis.models import Thesis
from app.thesis.monitor import CanonicalEvent, SourceService
from app.thesis.repository import ThesisRepository

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T00:00:00+00:00"


class _Filings:
    name = "sec_filings"

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
        return [
            CanonicalEvent(
                event_id="sec:1",
                canonical_ref="sec:1",
                source="sec_filings",
                known_at=T1,
                entity="NVDA",
                summary="10-K filed by NVDA (filed 2026-01-02)",
                cursor="2026-01-02",
            )
        ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, ThesisRepository, str]:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = ThesisRepository(tmp_path / "thesis")
    thesis = repo.create_thesis("NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    cid = repo.load_thesis(thesis.thesis_id).claims[0].claim_id
    rule = {"rule_id": "rule:1", "rule_type": "new_filing", "claim_ids": [cid]}
    repo.apply_research_result(thesis.thesis_id, {"watch_add": [rule]}, "", effective_at=T0)

    def _services(_thesis: Thesis) -> dict[str, SourceService]:
        return {"sec_filings": _Filings()}

    def _kernel(prompt: str, *, thesis_id: str, trigger_id: str, run_id: str) -> None:
        entry = {"title": "Filing reviewed", "body": "steady demand", "trigger_id": trigger_id, "run_id": run_id}
        repo.append_journal_entry(thesis_id, entry)

    monkeypatch.setattr(theses_routes, "_services", _services)
    monkeypatch.setattr(runner_mod, "_RUN_KERNEL", _kernel)
    app = FastAPI()
    app.include_router(theses_routes.router)
    return TestClient(app), repo, thesis.thesis_id


def test_empty_root_lists_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(theses_routes.router)
    assert TestClient(app).get("/api/theses").json() == []


def test_list_and_detail_before_any_tick(env: tuple[TestClient, ThesisRepository, str]) -> None:
    client, _repo, tid = env
    [row] = client.get("/api/theses").json()
    assert row["thesis_id"] == tid and row["status"] == "active" and row["scope"] == "NVDA"
    assert (row["alerts"], row["pending"]) == (0, 0)
    detail = client.get(f"/api/theses/{tid}").json()
    assert [c["statement"] for c in detail["claims"]] == ["NVDA demand grows"]
    assert [r["rule_type"] for r in detail["watch_rules"]] == ["new_filing"]
    assert detail["journal"] == [] and detail["triggers"] == [] and detail["last_tick"] is None


def test_check_runs_one_tick_and_detail_shows_its_effects(env: tuple[TestClient, ThesisRepository, str]) -> None:
    client, _repo, tid = env
    out = client.post(f"/api/theses/{tid}/check").json()
    assert len(out["triggers_created"]) == 1 and len(out["runs"]) == 1 and out["no_op"] is False
    detail = client.get(f"/api/theses/{tid}").json()
    assert (detail["alerts"], detail["pending"]) == (1, 0)
    assert detail["triggers"][0]["status"] == "processed"
    [entry] = detail["journal"]
    assert (entry["title"], entry["body"]) == ("Filing reviewed", "steady demand")
    assert entry["trigger_id"] == out["triggers_created"][0]
    assert detail["last_tick"] is not None
    again = client.post(f"/api/theses/{tid}/check").json()
    assert again["triggers_created"] == [] and again["no_op"] is True


def test_pause_resume_and_paused_check_is_a_no_op(env: tuple[TestClient, ThesisRepository, str]) -> None:
    client, _repo, tid = env
    assert client.post(f"/api/theses/{tid}/pause").json() == {"thesis_id": tid, "status": "paused"}
    out = client.post(f"/api/theses/{tid}/check").json()
    assert out["no_op"] is True and out["no_op_reason"] == "paused" and out["triggers_created"] == []
    assert client.post(f"/api/theses/{tid}/resume").json()["status"] == "active"


def test_unknown_is_404_and_closed_is_409(env: tuple[TestClient, ThesisRepository, str]) -> None:
    client, repo, tid = env
    for method, path in (("get", ""), ("post", "/check"), ("post", "/pause")):
        assert client.request(method, f"/api/theses/thesis:nope{path}").status_code == 404
    repo.close_thesis(tid)
    assert client.post(f"/api/theses/{tid}/check").status_code == 409
    assert client.post(f"/api/theses/{tid}/resume").status_code == 409
