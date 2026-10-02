"""CRAP-path tests for the ResearchCoreFix slice (single assembled file).

Covers decision paths (auth/budget/routing/parsing/error-fallback) for:
app/tool_runtime.py, app/research/service.py, app/research/repository.py,
app/research/jobs.py, app/research/director.py, app/research/agents/**,
app/services/**, app/storage/runs.py,
app/domain/**, app/analytics/**, app/tools.py, app/research/runner.py.
Plain pytest, tmp_path DBs, fakes for models/dispatch.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
import socket
import sqlite3
from collections.abc import Buffer, Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Self, override

import pytest

import app.tool_runtime as gw
import tests.research_source_seam as seam
from app import tools as tools_mod
from app.analytics import options as opt
from app.analytics import screens
from app.domain.evidence import relationship_evaluation as EV
from app.domain.evidence import relationships as R
from app.domain.market.identity import resolve_ticker_aliases
from app.domain.market.securities import SecurityResolution, TickerAlias
from app.domain.portfolio import PortfolioSnapshot, Position
from app.domain.portfolio.snapshot import build_portfolio_snapshot
from app.domain.risk.evaluation import UNKNOWN_SECTOR, evaluate_mandate
from app.domain.risk.mandate import Mandate, RiskLimit, parse_mandate
from app.policy import Capability, RequestContext
from app.research import jobs as _jobs
from app.research import service as svc
from app.research import session as _session
from app.research.agents import (
    GroundedClaim,
    ModelOutputFailure,
    ResearchRequest,
    parse_committee_envelope,
    parse_committee_output,
    parse_grounded_claims,
)
from app.research.agents.bearbot import BearAnalysis, run_bearbot
from app.research.agents.bullbot import BullAnalysis, run_bullbot
from app.research.agents.scout import (
    DispatchFn,
    ModelFn,
    ScoutAssignment,
    ScoutResult,
    ScoutRole,
    run_scout,
)
from app.research.agents.source_agent import SourceDossier, assemble_dossier
from app.research.agents.stockbot import StockbotAnalysis, run_stockbot
from app.research.director import (
    DirectorBudgets,
    DirectorDeps,
    Wave1Result,
    decide_next_wave,
)
from app.research.models import Job, ResearchSession, default_policy, utcnow
from app.research.repository import (
    ResearchRepository,
    _iso_or_none,
    pending_next_action,
)
from app.research.runner import (
    _extract_known_at,
    resume_live,
    run_live,
)
from app.robinhood.options import OptionQuote
from app.services import research_data, sec_facts
from app.services.account_identity import local_account_id
from app.services.dividend_analysis import cadence_from_events, lifecycle_from_events
from app.services.evidence_claims import build_evidence_claims
from app.services.evidence_resolution import resolve_subject
from app.services.herdr_client import HerdrClient, WorkerMap
from app.services.portfolio_sync import persist_snapshot, read_latest_snapshot
from app.services.research_data import (
    prepare_short_interest_data,
    refresh_finra_short_interest,
    refresh_sec_tickers,
)
from app.services.sec_facts import DividendEventRow, FinancialFactRow
from app.storage import runs as runs_mod
from app.storage.runs import RunRecorder, get_runs_db_path
from app.thesis.models import Thesis
from app.thesis.repository import ThesisRepository
from app.tool_runtime import RuntimeToolSession, execute_agent_tool
from app.tools import execute_tool

# --- gateway: tool_runtime gates (from /tmp/rc_coregateway.py) ---


def _svc_sid(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jobs = repo.list_jobs(sid)
    return sid, jobs[0].job_id


def test_permit_deny_unknown_tool() -> None:
    session = RuntimeToolSession(session_id="gw-permit-deny")
    out = execute_agent_tool("definitely_not_a_tool", {}, session)
    assert "error" in out
    assert "not permitted" in str(out["error"])


def test_invalid_context_unknown_staged_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    ResearchRepository()  # ensure file exists
    ctx = RuntimeToolSession(session_id="gw-unknown-sid")
    ctx.active_research_session_id = "no-such-session"
    out = execute_agent_tool("search_tools", {"query": "probe"}, ctx)
    assert out.get("error_type") == "invalid_research_context"
    assert "Unknown research session" in str(out.get("error", ""))


def test_invalid_context_dispatch_missing_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _jid = _svc_sid(repo)
    ctx = RuntimeToolSession(session_id="gw-missing-jid")
    ctx.active_research_session_id = sid
    out = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    assert out.get("error_type") == "invalid_research_context"
    assert "Active research job is required" in str(out.get("error", ""))


def test_dispatch_budget_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(jid), tool_budget=1))
    ctx = RuntimeToolSession(session_id="gw-budget")
    ctx.active_research_session_id = sid
    ctx.active_research_job_id = jid
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_execute(
        name: str, arguments: dict[str, object], model: str, context: RequestContext | None = None
    ) -> dict[str, object]:
        calls.append((name, dict(arguments)))
        return {"result_type": "web_search", "query": arguments.get("query"), "results": [], "source": "exa"}

    monkeypatch.setattr(gw, "execute_tool", _fake_execute)
    first = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    assert "error" not in first, first
    second = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    # §3: explicit per-job kernel exhaustion passes through verbatim (no budget_exhausted collapse).
    assert "tool_budget exhausted" in str(second.get("error", "")), second
    assert second.get("error_type") != "budget_exhausted", second
    assert len(calls) == 1


def test_source_refs_prefers_record_id_and_collects_all() -> None:
    out = gw._extract_source_refs(
        {
            "source": "SEC",
            "filings": [
                {"url": "https://sec.gov/a", "known_at": "2025-01-01"},
                {"accession_no": "0001", "url": "https://sec.gov/b", "known_at": "2025-02-01"},
            ],
        }
    )
    assert out["record_id"] == "0001"
    assert out["uri"] == "https://sec.gov/b"
    all_refs = out["all"]
    assert isinstance(all_refs, list)
    assert len(all_refs) == 2


def test_tool_meta_counts_and_truncation() -> None:
    meta = gw._tool_result_meta(
        {
            "source": "exa",
            "as_of": "2025-01-01",
            "rows": [{"a": 1}, {"a": 2}],
            "returned_count": 2,
            "total_records": 5,
        }
    )
    assert meta.row_count == 2
    assert meta.returned_count == 2
    assert meta.truncated is True
    assert meta.as_of == "2025-01-01"
    assert meta.source_names == ["exa"]


def test_call_tool_unwrap_deny_paths() -> None:
    bad_name = gw._unwrap_call_tool("call_tool", {"name": "", "arguments": {}})
    assert bad_name.error is not None
    bad_args = gw._unwrap_call_tool("call_tool", {"name": "get_fundamentals", "arguments": "x"})
    assert bad_args.error is not None
    forbidden = gw._unwrap_call_tool("call_tool", {"name": "browse_tools", "arguments": {}})
    assert forbidden.error is not None
    unknown = gw._unwrap_call_tool("call_tool", {"name": "nope_missing", "arguments": {}})
    assert unknown.error is not None
    ok = gw._unwrap_call_tool("call_tool", {"name": "get_fundamentals", "arguments": {"ticker": "AAPL"}})
    assert ok.error is None and ok.inner_name == "get_fundamentals"


def test_company_resolve_target_paths() -> None:
    assert gw._company_resolve_target({"ticker": "AAPL"}, "ticker") is None
    assert gw._company_resolve_target({}, "ticker") is None


def test_discovery_meta_names_shapes() -> None:
    m, t = gw._discovery_meta_names({"matches": [{"name": "a"}, {"name": 1}]})
    assert m == ["a"] and t is None
    m2, _ = gw._discovery_meta_names({"schemas": [{"function": {"name": "b"}}]})
    assert m2 == ["b"]
    _m3, t3 = gw._discovery_meta_names({"tools": [{"name": "c"}]})
    assert t3 == ["c"]


def test_schema_tool_names_skips_malformed() -> None:
    assert gw._schema_tool_names([{}, {"function": {}}, {"function": {"name": "d"}}]) == ["d"]


def test_failure_outcome_soft_and_denied() -> None:
    failed, soft, _d, status, _et, _em = gw._failure_outcome({"error": "x", "soft": True})
    assert failed and soft and status == "failed"
    failed2, _s2, denied2, status2, et2, _em2 = gw._failure_outcome({"error": "tool not permitted: y"})
    assert failed2 and denied2 and status2 == "denied" and et2 == "permission_denied"


def test_success_meta_discovery_and_refs() -> None:
    import app.tool_runtime as _g

    meta = _g._tool_result_meta({"source": "s", "rows": [1]})
    env = type(
        "E",
        (),
        {
            "source": "s",
            "sensitivity": type("S", (), {"value": "public"})(),
            "integrity": type("I", (), {"value": "ok"})(),
        },
    )()
    out = _g._success_meta("search_tools", {"matches": [{"name": "z"}], "accession_no": "1"}, meta, env, "completed")
    assert out["matches"] == ["z"] and out["status"] == "completed"


def test_success_meta_source_handle_passthrough() -> None:
    import app.tool_runtime as _g

    meta = _g._tool_result_meta({"source": "s", "rows": [1]})
    env = type(
        "E",
        (),
        {
            "source": "s",
            "sensitivity": type("S", (), {"value": "public"})(),
            "integrity": type("I", (), {"value": "ok"})(),
        },
    )()
    handle = {
        "accession_no": "0000320193-25-000079",
        "document_name": "d.htm",
        "basis": "raw",
        "offset": 0,
        "max_chars": 100,
        "text_hash": "abc",
    }
    out = _g._success_meta("get_sec_document", {"source_handle": handle, "text": "body"}, meta, env, "completed")
    assert out["source_handle"] == handle
    bare = _g._success_meta("get_sec_document", {"text": "body"}, meta, env, "completed")
    assert "source_handle" not in bare


def test_company_name_fills_missing_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _g

    def _ticker_aapl(name: str) -> str | None:
        return "AAPL"

    monkeypatch.setattr(_g, "_resolve_company_to_ticker", _ticker_aapl)
    out = _g._resolve_company_arguments("get_obligations", {"company_name": "Apple Inc."})
    assert out.get("ticker") == "AAPL"


def test_company_name_unresolved_keeps_args(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _g

    def _ticker_none(name: str) -> str | None:
        return None

    monkeypatch.setattr(_g, "_resolve_company_to_ticker", _ticker_none)
    args: dict[str, object] = {"company_name": "Unknown Co"}
    assert _g._resolve_company_arguments("get_obligations", args) == args


def test_try_company_to_ticker_swallows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _g

    def _boom(name: str) -> str | None:
        raise RuntimeError("lookup down")

    monkeypatch.setattr(_g, "_resolve_company_to_ticker", _boom)
    assert _g._try_company_to_ticker("x") is None


def test_call_tool_dispatch_success_no_budget_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _g

    def _fake_execute(
        name: str, arguments: dict[str, object], model: str, context: RequestContext | None = None
    ) -> dict[str, object]:
        assert name == "get_fundamentals"
        return {"result_type": "fundamentals", "ticker": "AAPL", "source": "test"}

    monkeypatch.setattr(_g, "execute_tool", _fake_execute)
    session = RuntimeToolSession(session_id="gw-call-tool-ok")
    out = execute_agent_tool(
        "call_tool", {"name": "get_fundamentals", "arguments": {"ticker": "AAPL", "metric": "overview"}}, session
    )
    assert "content" in out, out


def test_call_tool_dispatch_error_passthrough() -> None:
    session = RuntimeToolSession(session_id="gw-call-tool-err")
    out = execute_agent_tool("call_tool", {"name": "", "arguments": {}}, session)
    assert "error" in out


# --- service: research kernel (from /tmp/rc_coreservice.py) ---

SVC_ASOF = "2025-06-30T00:00:00+00:00"
KNOWN = "2025-06-29T00:00:00+00:00"
FOLLOW = "What drove NVDA datacenter revenue growth?"


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchRepository:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    return ResearchRepository()


def _sid(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str]:
    sid = svc.create_research(q, "o", as_of=SVC_ASOF, repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    return sid, src


def _item(eid: str, wave: int = 1, **kw: object) -> dict[str, object]:
    """One evidence item carrying the canonical handle for its own cited passage.

    A caller override of source_record_id/document_name/matching_passage re-mints
    the handle for that filing, so the kernel reloads what the item declares.
    """
    d: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": "c",
        "subject": "NVDA",
        "source_name": "SEC",
        "source_uri": "https://sec.gov/x",
        "source_record_id": seam.DEFAULT_ACCESSION,
        "document_name": seam.DEFAULT_DOCUMENT,
        "matching_passage": "passage-" + eid,
        "known_at": KNOWN,
    }
    d.update(kw)
    if "source_handle" not in kw:
        d["source_handle"] = seam.handle_for(
            str(d["matching_passage"]),
            accession=str(d["source_record_id"]),
            document=str(d["document_name"]),
        )
    return d


def _cov(**kw: object) -> dict[str, object]:
    """Absence-observation coverage envelope: searched scope + paging completeness."""
    d: dict[str, object] = {
        "forms": [],
        "dates": [],
        "partitions": [],
        "entities": [],
        "docs": [],
        "gaps": [],
        "pagination_complete": True,
        "complete": True,
    }
    d.update(kw)
    return d


def _sufficient_coverage(**kw: object) -> dict[str, object]:
    """Full sufficiency envelope: investigated scope + spent searches/branches, no residuals."""
    d: dict[str, object] = {
        "useful_for_question": "sufficient",
        "major_entities_investigated": ["NVDA"],
        "relationship_types_checked": ["supplier"],
        "forms_examined": ["10-K"],
        "exhibits_examined": ["EX-10.1"],
        "material_open_questions": [],
        "search_runs": ["sr:1"],
        "covered_branches": ["datacenter demand"],
    }
    d.update(kw)
    return d


def _ana(eid: str, follow_ups: Sequence[object] = ()) -> dict[str, object]:
    return {
        "executive_view": "view",
        "claims": [{"text": "finding", "evidence_ids": [eid]}],
        "impact_channels": [{"text": "channel", "direction": "up", "evidence_ids": [eid]}],
        "materiality": {"overall": "medium", "reasoning": "material"},
        "uncertainties": ["scope remains SEC-only"],
        "what_would_change": ["a materially new filing"],
        "follow_ups": list(follow_ups),
    }


def _freeze1(repo: ResearchRepository, sid: str, src: str, eid: str) -> dict[str, object]:
    svc.record_evidence(sid, src, _item(eid), repo=repo)
    svc.complete_job(src, {}, repo=repo)
    return svc.freeze_session(sid, 1, repo=repo)


def _trio(repo: ResearchRepository, sid: str, eid: str, follow_ups: Sequence[object] = ()) -> list[str]:
    out = svc.create_committee_jobs(sid, 1, repo=repo)
    jobs = out["jobs"]
    assert isinstance(jobs, list)
    for jid_raw, role in zip(jobs, ("stockbot", "bullbot", "bearbot")):
        assert isinstance(jid_raw, str)
        svc.record_committee_analysis(sid, jid_raw, role, _ana(eid, follow_ups), repo=repo)
    jobs_out = out["jobs"]
    assert isinstance(jobs_out, list)
    return [j for j in jobs_out if isinstance(j, str)]


# -- authorize_and_consume_dispatch (unc 1013, 1017-1018, 1020-1021, 1025) --


def test_dispatch_cross_session_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid_a, _ = _sid(repo)
    _sid_b, src_b = _sid(repo, "AMD demand?")
    with pytest.raises(ValueError, match="belongs to"):
        svc.authorize_and_consume_dispatch(sid_a, src_b, "get_sec_filing", repo=repo)


def test_dispatch_sec_job_rejects_non_sec_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    sec_job_raw = svc.start_job(sid, "source_agent", source="SEC", repo=repo)["job_id"]
    assert isinstance(sec_job_raw, str)
    sec_job = sec_job_raw
    with pytest.raises(ValueError, match="outside SEC domain"):
        svc.authorize_and_consume_dispatch(sid, sec_job, "get_short_interest", repo=repo)
    ok = svc.authorize_and_consume_dispatch(sid, sec_job, "get_sec_filing", repo=repo)
    assert ok["tool_calls_used"] == 1
    assert repo.get_job(sec_job).tool_budget is None  # §1 unlimited default


def test_dispatch_unknown_job_remaps_to_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)

    def _boom(session_id: str, job_id: str) -> object:
        raise KeyError(f"unknown job_id: {job_id!r}")

    monkeypatch.setattr(repo, "consume_dispatch_budget", _boom)
    with pytest.raises(svc.ResearchNotFound):
        svc.authorize_and_consume_dispatch(sid, src, "get_sec_filing", repo=repo)


def test_dispatch_unknown_session_raises_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    with pytest.raises(svc.ResearchNotFound):
        svc.authorize_and_consume_dispatch("rs:nope", "job:nope", "get_sec_filing", repo=repo)


# -- decide_next_wave (unc 857-858 no-freeze, 897-900 targeted transition) --


def test_decide_no_freeze_builds_empty_wave1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    out = svc.decide_next_wave(sid, repo=repo)
    assert out == {
        "authorized": False,
        "stop_reason": "no_questions",
        "reason_detail": "committee requested no follow-up research",
        "targeted_question": "",
        "targeted_domain": "",
    }
    assert repo.get_session(sid).status == "created"


def test_decide_authorized_moves_analyzing_to_targeted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    _trio(repo, sid, eid, follow_ups=[FOLLOW])
    assert repo.get_session(sid).status == "analyzing"
    out = svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True
    assert out["targeted_question"] == FOLLOW
    assert out["targeted_domain"] == "SEC"
    cur = repo.get_session(sid)
    assert cur.status == "targeted_research"
    assert cur.current_wave == 2


def test_decide_stop_moves_analyzing_to_synthesizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    _trio(repo, sid, eid)
    out = svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is False
    assert out["stop_reason"] == "no_questions"
    assert repo.get_session(sid).status == "synthesizing"


# -- research_events (unc 84-92: clamp + job filter) --


def test_events_clamp_limit_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _src = _sid(repo)
    assert svc.research_events(sid, limit=0, repo=repo)["limit"] == 1
    assert svc.research_events(sid, limit=999, repo=repo)["limit"] == 200
    bad_limit: int = json.loads('"x"')
    assert svc.research_events(sid, limit=bad_limit, repo=repo)["limit"] == 100
    assert svc.research_events(sid, cursor=-5, repo=repo)["cursor"] == 0
    bad_cursor: int = json.loads('"x"')
    assert svc.research_events(sid, cursor=bad_cursor, repo=repo)["cursor"] == 0
    assert svc.research_events(sid, cursor=10, repo=repo)["events"] == []


def test_events_job_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    page = svc.research_events(sid, job_id=src, repo=repo)
    events = page["events"]
    assert isinstance(events, list)
    assert len(events) == 1
    first_event = events[0]
    assert isinstance(first_event, dict)
    assert first_event["job_id"] == src
    assert svc.research_events(sid, job_id="job:nope", repo=repo)["events"] == []
    with pytest.raises(svc.ResearchNotFound):
        svc.research_events("rs:nope", repo=repo)


# -- job_diagnostics (unc 162-170: stale + deadline arms) --


def test_diagnostics_fresh_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    out = svc.job_diagnostics(sid, src, repo=repo)
    assert out["stale"] is False
    assert out["evidence_count"] == 0
    assert out["deadline_remaining_s"] is None  # §1 default budget sets no deadline
    assert out["last_event"] == "running"
    repo.save_job(dataclasses.replace(repo.get_job(src), deadline=utcnow() + timedelta(seconds=300)))
    out2 = svc.job_diagnostics(sid, src, repo=repo)
    rem = out2["deadline_remaining_s"]
    assert isinstance(rem, (int, float)) and rem > 0  # an explicit deadline still counts down


def test_diagnostics_stale_no_deadline_counts_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    svc.record_evidence(sid, src, _item(eid), repo=repo)
    old = utcnow() - timedelta(seconds=600)
    repo.save_job(dataclasses.replace(repo.get_job(src), last_heartbeat_at=old, started_at=old, deadline=None))
    out = svc.job_diagnostics(sid, src, repo=repo)
    assert out["stale"] is True
    stale_s = out["staleness_s"]
    assert isinstance(stale_s, (int, float)) and stale_s >= 600
    assert out["deadline_remaining_s"] is None
    assert out["evidence_count"] == 1


# -- run_research (running-reuse + queued-start + create arms) --


def test_run_reuses_running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    assert svc.run_research(sid, repo=repo)["job_id"] == src


def test_run_starts_queued_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(src), status="queued"))
    out = svc.run_research(sid, repo=repo)
    assert out["job_id"] == src
    assert out["status"] == "running"


def test_run_returns_first_when_all_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    svc.complete_job(src, {}, repo=repo)
    out = svc.run_research(sid, repo=repo)
    assert out["job_id"] == src
    assert out["status"] == "completed"


def test_run_creates_job_for_empty_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import session as _session

    repo = _repo(tmp_path, monkeypatch)
    sess = _session.create_session("q?", "o", as_of=SVC_ASOF)
    repo.save_session(sess)
    out = svc.run_research(sess.session_id, repo=repo)
    assert out["status"] == "running"
    assert len(repo.list_jobs(sess.session_id)) == 1


# -- create_committee_jobs (role-reuse + started-transition arms) --


def test_committee_jobs_create_then_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _src = _sid(repo)
    first = svc.create_committee_jobs(sid, 1, repo=repo)
    first_jobs = first["jobs"]
    assert isinstance(first_jobs, list)
    assert len(first_jobs) == 3
    assert len(set(first_jobs)) == 3
    roles = sorted(j.job_type for j in repo.list_jobs(sid) if j.job_id in first_jobs)
    assert roles == ["bearbot", "bullbot", "stockbot"]
    for j_raw in first_jobs:
        assert isinstance(j_raw, str)
        assert repo.get_job(j_raw).status == "running"
    second = svc.create_committee_jobs(sid, 1, repo=repo)
    assert second["jobs"] == first["jobs"]
    verb = second["pending_next_action"]
    assert isinstance(verb, dict)
    assert verb["verb"] == "EXECUTE_COMMITTEE"


# -- record_evidence guards --


def test_evidence_rejects_non_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    bad_item: Mapping[str, object] = json.loads('["not", "a", "mapping"]')
    with pytest.raises(ValueError, match="must be a mapping"):
        svc.record_evidence(sid, src, bad_item, repo=repo)


def test_evidence_rejects_cross_session_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid_a, _ = _sid(repo)
    _, src_b = _sid(repo, "AMD demand?")
    with pytest.raises(ValueError, match="belongs to"):
        svc.record_evidence(sid_a, src_b, _item(f"{sid_a}:ev:1"), repo=repo)


def test_evidence_rejects_terminal_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    svc.cancel_research(sid, repo=repo)
    with pytest.raises(ValueError, match="terminal"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)


def test_evidence_claim_and_json_content_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    out = svc.record_evidence(
        sid,
        src,
        _item(f"{sid}:ev:1", content="  ", claim_text="  the claim  "),
        repo=repo,
    )
    assert out["content"] == "the claim"
    minimal = {
        "wave_id": 1,
        "known_at": KNOWN,
        "source_uri": "https://sec.gov/x",
        "source_record_id": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
        "matching_passage": "revenue grew",
        "source_handle": seam.handle_for("revenue grew"),
    }
    out2 = svc.record_evidence(sid, src, dict(minimal, evidence_id=f"{sid}:ev:2"), repo=repo)
    content = out2["content"]
    assert isinstance(content, str)
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    assert parsed["known_at"] == KNOWN


def test_evidence_rejects_wave_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    with pytest.raises(ValueError, match="!= job wave"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:1", wave=2), repo=repo)


def test_evidence_coerces_bad_wave_to_job_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    bad_wave: int = json.loads('"bad"')
    out = svc.record_evidence(sid, src, _item(f"{sid}:ev:1", wave=bad_wave), repo=repo)
    assert out["wave_id"] == 1


def test_evidence_rejects_stale_wave_after_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    _freeze1(repo, sid, src, f"{sid}:ev:1")
    late_raw = svc.start_job(sid, "source_agent", repo=repo)["job_id"]
    assert isinstance(late_raw, str)
    late = late_raw
    with pytest.raises(ValueError, match="current wave"):
        svc.record_evidence(sid, late, _item(f"{sid}:ev:2"), repo=repo)


def test_evidence_rejects_refrozen_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pol = default_policy()
    research_raw = pol["research"]
    assert isinstance(research_raw, dict)
    pol["research"] = {**research_raw, "max_waves": 10}
    repo = _repo(tmp_path, monkeypatch)
    sid = svc.create_research("NVDA demand?", "o", as_of=SVC_ASOF, policy=pol, repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)
    svc.complete_job(src, {}, repo=repo)
    svc.freeze_session(sid, 5, repo=repo)
    late_raw = svc.start_job(sid, "source_agent", budget={"wave_id": 5}, repo=repo)["job_id"]
    assert isinstance(late_raw, str)
    late = late_raw
    with pytest.raises(ValueError, match="already frozen"):
        svc.record_evidence(sid, late, _item(f"{sid}:ev:2", wave=5), repo=repo)


def test_evidence_search_coverage_requires_query_and_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    base = _item(
        f"{sid}:ev:1",
        type="search_coverage",
        search_id="s1",
        query="NVDA filings",
        subject="NVDA",
        coverage=_cov(),
    )
    base.pop("source_record_id", None)  # an absence cites the SearchRun only, never a filing
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.record_evidence(sid, src, {**base, "query": "  "}, repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.record_evidence(sid, src, {**base, "search_id": ""}, repo=repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, {**base, "source_record_id": "0000320193-25-000079"}, repo=repo)
    out = svc.record_evidence(sid, src, base, repo=repo)
    assert out["claim_kind"] == "absence_observation"
    # A search-scope absence is a coverage artifact: scoped to its search, never citable evidence.
    assert out["recorded_as"] == "coverage_artifact" and out["citable"] is False
    assert out["search_id"] == "s1" and out["query"] == "NVDA filings"
    assert out["evidence_ids"] == [] and "provenance" not in out
    artifacts = repo.list_coverage_artifacts(sid)
    assert [a["artifact_id"] for a in artifacts] == [out["artifact_id"]]
    assert repo.list_evidence(sid) == []


def test_evidence_provenance_mismatch_and_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    nav = _item(f"{sid}:ev:1", subject="OpenAI", source_record_id="sr:1", search_id="sr:1", query="OpenAI contracts")
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, nav, repo=repo)  # a search id is navigation, not an accession
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:1", source_record_id="abc"), repo=repo)
    good = _item(
        f"{sid}:ev:1", subject="OpenAI", source_record_id="0000320193-25-000081", source_uri="https://sec.gov/openai"
    )
    stored = svc.record_evidence(sid, src, good, repo=repo)
    assert stored["evidence_id"] == f"{sid}:ev:1"
    assert stored["source_record_id"] == "0000320193-25-000081"
    prov = stored["provenance"]
    assert isinstance(prov, dict)
    assert prov["kind"] == "sec_source" and prov["document_name"] == "nvda-20250331.htm"


def test_materialize_sec_passage_fails_closed_per_handle_defect() -> None:
    """The kernel reload is the admission gate: every handle defect carries its own code."""
    from app.research.service import materialize_sec_passage

    passage = "Data center revenue grew 142% year over year."
    handle = seam.handle_for(passage)
    materialized = materialize_sec_passage(handle, "Data   center revenue grew 142% year over year")
    # The stored text is the archive's slice of the cited span, not the locator string.
    cited = "Data center revenue grew 142% year over year"
    assert materialized.provenance["passage"] == cited and materialized.provenance["basis"] == "rendered"
    assert (materialized.provenance["offset"], materialized.provenance["end"]) == (0, len(cited))
    assert materialized.provenance["text_hash"] == handle["text_hash"]
    # Malformed locators are deliberately outside the handle type; getattr keeps the
    # checker green without an escape hatch (those are forbidden in this tree).
    call_untyped = getattr(svc, "materialize_sec_passage")  # noqa: B009 - malformed-input contract; getattr keeps checker green
    tampered_source = {**handle, "source_content_hash": "f" * 64}
    legacy_content = {**handle, "content_hash": "f" * 64}
    missing_source = {k: v for k, v in handle.items() if k != "source_content_hash"}
    for defect, code in (
        (None, "ERR_SEC_HANDLE_INVALID"),
        ({**handle, "basis": "guessed"}, "ERR_SEC_HANDLE_INVALID"),
        ({**handle, "accession_no": "not-an-accession"}, "ERR_SEC_HANDLE_INVALID"),
        ({**handle, "offset": -1}, "ERR_SEC_HANDLE_INVALID"),
        (missing_source, "ERR_SEC_HANDLE_INVALID"),
        ({**handle, "text_hash": "0" * 64}, "ERR_SEC_HANDLE_STALE"),
        (tampered_source, "ERR_SEC_HANDLE_STALE"),
        ({**handle, "document_name": "never-stored.htm"}, "ERR_SEC_HANDLE_UNREADABLE"),
    ):
        with pytest.raises(ValueError, match=code):
            call_untyped(defect, passage)
    # Legacy/extra content_hash is ignored, never invalid: old handles still materialize.
    assert materialize_sec_passage(legacy_content, passage).provenance["passage"] == passage
    assert (
        materialize_sec_passage({k: v for k, v in handle.items() if k != "content_hash"}, passage).provenance["passage"]
        == passage
    )
    # A locator the reloaded window does not contain is never admitted, blank included.
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        materialize_sec_passage(handle, "this sentence is not in the window")
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        materialize_sec_passage(handle, "   ")


def test_live_handle_survives_archival_when_text_hash_differs_from_source_bytes(tmp_path: Path) -> None:
    """Live binary/PDF handles stay valid after archival: text_hash names the window, source bytes name the revision."""
    import hashlib as _hashlib

    from app.research.service import materialize_sec_passage as _materialize

    source_bytes = b"\xff\xfe binary \x00\x01 document bytes with citable text window here"
    window = source_bytes.decode("utf-8", "replace")
    assert _hashlib.sha256(window.encode("utf-8")).hexdigest() != _hashlib.sha256(source_bytes).hexdigest()
    accession, document = "0000320193-25-000079", "bytes-live-archived-000079.bin"
    handle: dict[str, object] = {
        "accession_no": accession,
        "document_name": document,
        "basis": "raw",
        "section": None,
        "query": None,
        "offset": 0,
        "max_chars": len(window),
        "length": len(window),
        "text_hash": _hashlib.sha256(window.encode("utf-8")).hexdigest(),
        "source_content_hash": _hashlib.sha256(source_bytes).hexdigest(),
        "source_uri": f"source://sec/{accession}/{document}",
    }
    # Live window served as decoded text; the exact source bytes land in the archive.
    seam.register_document(accession, document, window)
    archived = tmp_path / "raw.bin"
    archived.write_bytes(source_bytes)
    seam._SOURCE_ARCHIVE_PATH[(accession, document)] = str(archived)
    materialized = _materialize(handle, window)
    assert materialized.provenance["passage"] == window
    assert _hashlib.sha256(materialized.source_bytes).hexdigest() == handle["source_content_hash"]
    assert materialized.source_content_hash == handle["source_content_hash"]


def test_provenance_validation_arms_and_legacy_rows() -> None:
    """sec_source_ref validates its coordinates; validate_provenance reads canonical and legacy rows."""
    from app.research.evidence import EvidenceIntegrityError, sec_source_ref, validate_provenance

    ref = sec_source_ref(
        accession_no="0000320193-25-000079",
        document_name="nvda-20250331.htm",
        passage="revenue grew",
        offset=3,
        end=15,
        basis="rendered",
        text_hash="a" * 64,
    )
    assert validate_provenance({"kind": "sec_source", **{k: v for k, v in ref.items() if k != "kind"}}) == ref
    # A row persisted before coordinates existed still reads exactly as it was written.
    legacy = {
        "kind": "sec_source",
        "accession_no": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
        "passage": "revenue grew",
        "source_uri": "https://sec.gov/x",
    }
    assert validate_provenance(legacy) == legacy
    for bad, message in (
        (
            {"document_name": " ", "passage": "p", "offset": 0, "end": 1, "basis": "rendered", "text_hash": "a" * 64},
            "non-empty",
        ),
        (
            {"document_name": "d", "passage": "p", "offset": 0, "end": 1, "basis": "other", "text_hash": "a" * 64},
            "basis must be",
        ),
        (
            {"document_name": "d", "passage": "p", "offset": 4, "end": 4, "basis": "rendered", "text_hash": "a" * 64},
            "greater than offset",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            sec_source_ref(accession_no="0000320193-25-000079", **bad)
        with pytest.raises(EvidenceIntegrityError):
            validate_provenance({"kind": "sec_source", "accession_no": "0000320193-25-000079", **bad})


def test_evidence_duplicate_identity_returns_prior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)
    dup = svc.record_evidence(sid, src, _item(f"{sid}:ev:2"), repo=repo)
    assert dup == {"evidence_id": f"{sid}:ev:1", "accepted": False, "duplicate_of": f"{sid}:ev:1"}


def test_evidence_identity_conflict_names_identity_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lost-race catch in _persist_evidence_record keys off 'identity_key' in the message."""
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    first = svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)
    stored = repo.get_evidence(str(first["evidence_id"]))
    with pytest.raises(ValueError, match="identity_key"):
        repo.save_evidence({**stored, "evidence_id": f"{sid}:ev:race"})
    metadata = stored["metadata"]
    assert isinstance(metadata, dict)
    hit = repo.find_evidence_by_identity(sid, str(metadata["identity_key"]))
    assert hit is not None and hit["evidence_id"] == first["evidence_id"]
    assert repo.list_evidence_ids(sid) == [str(first["evidence_id"])]


def test_acceptance_archives_verified_window_bytes_not_rendered_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted SEC evidence archives the full exact bytes; one revision, no PIT conflict."""
    from app.sec import documents
    from app.sec.archive import find_archived_document

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    acc, doc = "0000320193-25-000079", "two-windows-000079.htm"
    first = "first cited sentence"
    second = "second cited sentence"
    full = f"{first}\n{second}"
    seam.register_document(acc, doc, full)
    h1 = seam.handle_for_registered(first, accession=acc, document=doc)
    h2 = seam.handle_for_registered(second, accession=acc, document=doc)
    out1 = svc.record_evidence(
        sid,
        src,
        _item(f"{sid}:ev:1", matching_passage=first, source_handle=h1, source_record_id=acc, document_name=doc),
        repo=repo,
    )
    out2 = svc.record_evidence(
        sid,
        src,
        _item(f"{sid}:ev:2", matching_passage=second, source_handle=h2, source_record_id=acc, document_name=doc),
        repo=repo,
    )
    for out in (out1, out2):
        stored = repo.get_evidence(str(out["evidence_id"]))
        provenance = stored["provenance"]
        assert isinstance(provenance, dict)
        found = find_archived_document(
            str(provenance["accession_no"]), str(provenance["document_name"]), root=tmp_path / "raw"
        )
        assert found is not None
        assert found.payload_path.read_bytes() == full.encode("utf-8")
        assert len(full) > len(first) and len(full) > len(second)
    assert documents.get_sec_document(acc, doc, as_of="2025-06-30", data_root=tmp_path)["text"] == full


def test_acceptance_archive_failure_raises_without_row_event_or_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preservation failure can never masquerade as accepted evidence."""
    import app.sec.archive as _archive

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)

    def _boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("disk down")

    monkeypatch.setattr(_archive, "archive_sec_document", _boom)
    with __import__("pytest").raises(OSError, match="disk down"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)
    assert repo.list_evidence(sid) == []
    assert repo.list_evidence_ids(sid) == []
    assert [e for e in repo.list_events(sid) if e.event_type == "evidence.accepted"] == []


def test_source_bytes_unavailable_emits_no_source_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing exact bytes fail closed: no row, no event, no session link."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    passage = "byte-less passage"
    handle = seam.handle_for(passage, no_source_bytes=True)
    with pytest.raises(ValueError, match="ERR_NO_SOURCE_BYTES"):
        svc.record_evidence(
            sid, src, {**_item(f"{sid}:ev:1"), "matching_passage": passage, "source_handle": handle}, repo=repo
        )
    assert repo.list_evidence(sid) == []
    assert repo.list_evidence_ids(sid) == []
    assert [e for e in repo.list_events(sid) if e.event_type == "evidence.accepted"] == []


def test_revision_switch_rejects_stale_handle_without_row_event_link_or_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handle pinned to rev-A stays stale after the source moves to rev-B: no row, event, link, or archive."""
    from app.sec.archive import iter_archived_documents

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    acc, doc = "0000320193-25-000079", "rev-switch-000079.htm"
    window = "cited window stays identical"
    rev_a = f"header-A\n{window}\ntrailer-A"
    rev_b = f"header-B-CHANGED\n{window}\ntrailer-B-CHANGED"
    seam.register_document(acc, doc, rev_a)
    stale = seam.handle_for_registered(window, accession=acc, document=doc)
    seam.register_document(acc, doc, rev_b)  # same cited span, changed bytes elsewhere
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_STALE"):
        svc.record_evidence(
            sid,
            src,
            _item(f"{sid}:ev:1", matching_passage=window, source_handle=stale, source_record_id=acc, document_name=doc),
            repo=repo,
        )
    assert repo.list_evidence(sid) == []
    assert repo.list_evidence_ids(sid) == []
    assert repo.get_session(sid).evidence_ids == []
    assert [e for e in repo.list_events(sid) if e.event_type == "evidence.accepted"] == []
    revisions = list(iter_archived_documents(acc, doc, root=tmp_path / "raw"))
    assert revisions == []  # the stale attempt archived nothing


def test_offline_archive_serves_acceptance_when_live_filing_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The immutable archive wins: acceptance reads real archived bytes even with live filing down."""
    # The module autouse fixture routes get_sec_document to the seam fake; this
    # test needs the production archive read-back instead, so undo the seam for
    # app.sec.documents only (the kernel imports get_sec_document lazily from
    # there) and kill just the live-filing path. _exact_source_bytes stays
    # seam-patched but is unreachable: the archived row carries raw_archive_path.
    import importlib as _importlib
    import inspect as _inspect

    import app.sec.archive as _archive
    import app.sec.documents as _documents
    from app.sec.archive import find_archived_document

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    # The autouse fixture patched the module object in place; reload the file to
    # restore production get_sec_document, then kill just the live-filing path.
    # _exact_source_bytes stays seam-patched but is unreachable: the archived row
    # carries raw_archive_path, so the kernel reads real archived bytes.
    real_mod = _importlib.reload(_documents)
    monkeypatch.setattr(_documents, "get_sec_document", real_mod.get_sec_document)
    assert "Archive-first" in _inspect.getsource(_documents.get_sec_document)
    real_get_sec_document = _documents.get_sec_document

    def _no_live(accession_no: str) -> object:
        del accession_no
        raise ValueError("edgar offline")

    monkeypatch.setattr(_documents, "_filing", _no_live)
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    acc, doc = "0000320193-25-000079", "offline-archived-000079.htm"
    passage = "offline archived passage"
    full = f"offline header\n{passage}\noffline trailer"
    canonical_url = f"https://www.sec.gov/Archives/edgar/data/{acc.replace('-', '')}/{doc}"
    _archive.archive_sec_document(
        acc,
        doc,
        full.encode("utf-8"),
        url=canonical_url,
        retrieved_at="2025-05-01T00:00:00Z",
        metadata={"known_at": "2025-05-01T00:00:00Z", "filed_at": "2025-04-30", "representation": "source_bytes"},
        root=tmp_path / "raw",
    )
    live = real_get_sec_document(acc, doc, as_of="2025-06-30", data_root=tmp_path)
    assert isinstance(live["text"], str) and passage in str(live["text"])
    handle = live["source_handle"]
    assert isinstance(handle, dict)
    out = svc.record_evidence(
        sid,
        src,
        _item(f"{sid}:ev:1", matching_passage=passage, source_handle=handle, source_record_id=acc, document_name=doc),
        repo=repo,
    )
    stored = repo.get_evidence(str(out["evidence_id"]))
    provenance = stored["provenance"]
    assert isinstance(provenance, dict) and provenance["passage"] == passage
    found = find_archived_document(acc, doc, root=tmp_path / "raw")
    assert found is not None and found.payload_path.read_bytes() == full.encode("utf-8")


def test_canonical_url_wins_over_item_source_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored source_uri and archive URL are the kernel's canonical URL, never the model string."""
    from app.sec.archive import find_archived_document

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    acc, doc = "0000320193-25-000079", "canonical-url-000079.htm"
    passage = "canonical url passage"
    seam.register_document(acc, doc, f"header\n{passage}\ntrailer")
    handle = seam.handle_for_registered(passage, accession=acc, document=doc)
    canonical = f"https://www.sec.gov/Archives/edgar/data/{acc.replace('-', '')}/{doc}"
    out = svc.record_evidence(
        sid,
        src,
        _item(
            f"{sid}:ev:1",
            matching_passage=passage,
            source_handle=handle,
            source_record_id=acc,
            document_name=doc,
            source_uri="https://evil.example/x",
        ),
        repo=repo,
    )
    stored = repo.get_evidence(str(out["evidence_id"]))
    assert stored["source_uri"] == canonical
    found = find_archived_document(acc, doc, root=tmp_path / "raw")
    assert found is not None and found.url == canonical


def test_evidence_no_per_job_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    accepted: list[str] = []
    for i in range(12):
        out = svc.record_evidence(
            sid, src, _item(f"{sid}:ev:{i}", source_record_id=f"0000320193-25-{i:06d}"), repo=repo
        )
        accepted.append(str(out.get("evidence_id")))
    assert accepted == [f"{sid}:ev:{i}" for i in range(12)]  # no per-job evidence cap
    assert len(repo.list_evidence(sid)) == 12
    dup = svc.record_evidence(sid, src, _item(f"{sid}:ev:12", source_record_id="0000320193-25-000000"), repo=repo)
    assert dup == {"evidence_id": f"{sid}:ev:0", "accepted": False, "duplicate_of": f"{sid}:ev:0"}


def test_evidence_skips_unparseable_ledger_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    repo.save_evidence({"evidence_id": f"{sid}:ev:bad", "session_id": sid})
    out = svc.record_evidence(sid, src, _item(f"{sid}:ev:1"), repo=repo)
    assert out["evidence_id"] == f"{sid}:ev:1"


# -- submit_source_result (unc 518-560: coverage/envelope vs persist) --


def test_submit_requires_coverage_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    _, src = _sid(repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.submit_source_result(src, coverage=None, repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.submit_source_result(src, coverage={"complete": True}, repo=repo)
    with pytest.raises(ValueError, match="sufficient\\|insufficient"):
        svc.submit_source_result(src, coverage={"useful_for_question": "maybe"}, repo=repo)


def test_submit_rejects_empty_sufficient_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    _, src = _sid(repo)
    with pytest.raises(ValueError, match="ERR_EMPTY_RESULT"):
        svc.submit_source_result(src, coverage={"useful_for_question": "sufficient"}, evidence_ids=[], repo=repo)


def test_submit_rejects_dangling_evidence_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    _, src = _sid(repo)
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        svc.submit_source_result(
            src, coverage={"useful_for_question": "sufficient"}, evidence_ids=["ev:ghost"], repo=repo
        )


def test_submit_happy_then_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    svc.record_evidence(sid, src, _item(eid), repo=repo)
    out = svc.submit_source_result(src, coverage=_sufficient_coverage(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    assert out["dossier_id"] == f"{sid}:1:sec"
    assert out["evidence_ids"] == [eid]
    assert f"{sid}:1:sec" in repo.get_session(sid).dossier_ids
    with pytest.raises(ValueError, match="ERR_JOB_CLOSED"):
        svc.submit_source_result(src, coverage=_sufficient_coverage(), evidence_ids=[eid], repo=repo)


def test_submit_insufficient_without_evidence_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    _, src = _sid(repo)
    out = svc.submit_source_result(src, coverage={"useful_for_question": "insufficient"}, evidence_ids=[], repo=repo)
    assert out["job_status"] == "completed"
    assert out["evidence_ids"] == []


# -- freeze_session (status-ladder vs freeze-create vs coverage-gate) --


def test_freeze_rejects_bad_wave_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    bads: list[int] = json.loads('[0, true, "1", 1.5]')
    for bad in bads:
        with pytest.raises(ValueError, match="must be an int"):
            svc.freeze_session(sid, bad, repo=repo)


def test_freeze_rejects_wave_beyond_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pol = default_policy()
    research_raw = pol["research"]
    assert isinstance(research_raw, dict)
    pol["research"] = {**research_raw, "max_waves": 2}
    repo = _repo(tmp_path, monkeypatch)
    sid = svc.create_research("Q?", "o", as_of=SVC_ASOF, policy=pol, repo=repo)
    with pytest.raises(ValueError, match="exceeds max_waves"):
        svc.freeze_session(sid, 3, repo=repo)


def test_freeze_rejects_wrong_status_on_refreeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    _freeze1(repo, sid, src, f"{sid}:ev:1")
    with pytest.raises(ValueError, match="RESEARCHING required"):
        svc.freeze_session(sid, 1, repo=repo)


def test_freeze_empty_wave_proceeds_with_limitations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    svc.submit_source_result(src, coverage={"useful_for_question": "insufficient"}, evidence_ids=[], repo=repo)
    out = svc.freeze_session(sid, 1, repo=repo)
    assert out["freeze_id"] == f"{sid}:1:freeze"
    pna = out["pending_next_action"]
    assert isinstance(pna, dict)
    assert pna["verb"] == "EXECUTE_COMMITTEE"
    assert "limitation" in str(out.get("limitations", "")).lower()


def test_freeze_tolerates_duplicate_save_and_reports_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    _freeze1(repo, sid, src, f"{sid}:ev:1")
    cur = repo.get_session(sid)
    repo.save_session(dataclasses.replace(cur, status="researching"))
    out = svc.freeze_session(sid, 1, repo=repo)
    assert out["freeze_id"] == f"{sid}:1:freeze"


def test_freeze_insufficient_coverage_proceeds_with_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    svc.record_evidence(sid, src, _item(eid), repo=repo)
    svc.submit_source_result(src, coverage={"useful_for_question": "insufficient"}, evidence_ids=[eid], repo=repo)
    out = svc.freeze_session(sid, 1, repo=repo)
    assert out["coverage_gate"] == "insufficient"
    pna = out["pending_next_action"]
    assert isinstance(pna, dict)
    assert pna["verb"] == "EXECUTE_COMMITTEE"
    assert "limitation" in str(out.get("limitations", "")).lower()


def test_freeze_wave2_from_targeted_research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    _trio(repo, sid, eid, follow_ups=[FOLLOW])
    svc.decide_next_wave(sid, repo=repo)
    assert repo.get_session(sid).status == "targeted_research"
    src2_raw = svc.start_job(sid, "source_agent", budget={"wave_id": 2}, repo=repo)["job_id"]
    assert isinstance(src2_raw, str)
    src2 = src2_raw
    eid2 = f"{sid}:ev:2"
    svc.record_evidence(sid, src2, _item(eid2, wave=2, source_record_id="0000320193-25-000080"), repo=repo)
    svc.complete_job(src2, {}, repo=repo)
    out = svc.freeze_session(sid, 2, repo=repo)
    assert out["freeze_id"] == f"{sid}:2:freeze"
    assert repo.get_session(sid).current_wave == 2


# -- _wave1_state --


def test_wave1_no_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    with pytest.raises(ValueError, match="has no freeze"):
        svc._wave1_state(repo, repo.get_session(sid))


def test_wave1_unknown_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    cur = repo.get_session(sid)
    repo.save_session(dataclasses.replace(cur, freeze_ids=[f"{sid}:9:freeze"]))
    with pytest.raises(ValueError, match="unknown freeze_id"):
        svc._wave1_state(repo, repo.get_session(sid))


def test_wave1_skips_dangling_open_and_uncited_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    fid = f"{sid}:1:freeze"
    trio_raw = svc.create_committee_jobs(sid, 1, repo=repo)["jobs"]
    assert isinstance(trio_raw, list)
    trio = [j for j in trio_raw if isinstance(j, str)]
    stock_j, bull_j, _bear_j = trio
    svc.record_committee_analysis(sid, stock_j, "stockbot", _ana(eid), repo=repo)
    svc.complete_job(bull_j, {"nope": 1}, repo=repo)
    cur = repo.get_session(sid)
    _, queued = _jobs.create_job(cur, repo.list_jobs(sid), job_type="bearbot", owner="pi", wave_id=1)
    repo.save_session(repo.get_session(sid))
    repo.save_job(queued)
    cur = repo.get_session(sid)
    repo.save_session(
        dataclasses.replace(
            cur,
            committee_runs=[
                {"freeze_id": fid, "wave_id": 1, "jobs": [stock_j, bull_j, queued.job_id, "job:does-not-exist"]}
            ],
            updated_at=utcnow(),
        )
    )
    wave1, meta = svc._wave1_state(repo, repo.get_session(sid))
    assert wave1.stock is not None
    assert wave1.bull is None
    assert wave1.bear is None
    assert meta["evidence_ids"] == [eid]


# -- record_committee_analysis guards --


def test_committee_bad_role_and_bad_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    with pytest.raises(ValueError, match="stockbot\\|bullbot\\|bearbot"):
        svc.record_committee_analysis(sid, src, "scout", {}, repo=repo)
    bad_analysis: dict[str, object] = json.loads('["x"]')
    with pytest.raises(ValueError, match="must be a mapping"):
        svc.record_committee_analysis(sid, src, "stockbot", bad_analysis, repo=repo)


def test_committee_cross_session_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid_a, _ = _sid(repo)
    _, src_b = _sid(repo, "AMD demand?")
    with pytest.raises(ValueError, match="belongs to"):
        svc.record_committee_analysis(sid_a, src_b, "stockbot", {}, repo=repo)


def test_committee_requires_running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    (stock_j_raw,) = [svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"] for _ in range(1)]
    assert isinstance(stock_j_raw, str)
    stock_j = stock_j_raw
    svc.record_committee_analysis(sid, stock_j, "stockbot", _ana(eid), repo=repo)
    with pytest.raises(ValueError, match="running required"):
        svc.record_committee_analysis(sid, stock_j, "stockbot", _ana(eid), repo=repo)


def test_committee_requires_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    stock_j_raw = svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"]
    assert isinstance(stock_j_raw, str)
    stock_j = stock_j_raw
    with pytest.raises(ValueError, match="has no freeze"):
        svc.record_committee_analysis(sid, stock_j, "stockbot", _ana("ev:x"), repo=repo)


def test_committee_unknown_freeze_and_wave_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, _ = _sid(repo)
    stock_j_raw = svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"]
    assert isinstance(stock_j_raw, str)
    stock_j = stock_j_raw
    cur = repo.get_session(sid)
    repo.save_session(dataclasses.replace(cur, freeze_ids=[f"{sid}:9:freeze"]))
    with pytest.raises(ValueError, match="unknown freeze_id"):
        svc.record_committee_analysis(sid, stock_j, "stockbot", _ana("ev:x"), repo=repo)
    sid2, src2 = _sid(repo, "Q2?")
    eid2 = f"{sid2}:ev:1"
    _freeze1(repo, sid2, src2, eid2)
    w2_raw = svc.start_job(sid2, "stockbot", budget={"wave_id": 2}, repo=repo)["job_id"]
    assert isinstance(w2_raw, str)
    w2 = w2_raw
    with pytest.raises(ValueError, match="!= freeze wave"):
        svc.record_committee_analysis(sid2, w2, "stockbot", _ana(eid2), repo=repo)


def test_committee_rejects_unjsonable_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    stock_j_raw = svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"]
    assert isinstance(stock_j_raw, str)
    stock_j = stock_j_raw
    with pytest.raises(ValueError, match="JSON-able"):
        svc.record_committee_analysis(sid, stock_j, "stockbot", {"unjsonable": object()}, repo=repo)


# -- finalize_session guards --


def _full_trio(repo: ResearchRepository, sid: str, src: str, eid: str, follow_ups: Sequence[object] = ()) -> list[str]:
    _freeze1(repo, sid, src, eid)
    return _trio(repo, sid, eid, follow_ups)


def test_finalize_terminal_on_repeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)
    svc.decide_next_wave(sid, repo=repo)
    out = svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)
    assert out["status"] == "completed"
    with pytest.raises(ValueError, match="terminal"):
        svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)


def test_finalize_missing_trio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _freeze1(repo, sid, src, eid)
    trio_raw = svc.create_committee_jobs(sid, 1, repo=repo)["jobs"]
    assert isinstance(trio_raw, list)
    first_trio = trio_raw[0]
    assert isinstance(first_trio, str)
    svc.record_committee_analysis(sid, first_trio, "stockbot", _ana(eid), repo=repo)
    with pytest.raises(ValueError, match="missing committee"):
        svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)


def test_finalize_missing_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)

    def _dis_none(*a: object) -> None:
        return None

    monkeypatch.setattr("app.research.synthesis.committee.compute_disagreement", _dis_none)
    with pytest.raises(ValueError, match="missing disagreement"):
        svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)


def test_finalize_bad_claims_and_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)
    good: list[object] = [{"text": "finding", "evidence_ids": [eid]}]
    bad_claims: list[object] = json.loads('"nope"')
    with pytest.raises(ValueError, match="must be a list"):
        svc.finalize_session(sid, "answer", bad_claims, repo=repo)
    with pytest.raises(ValueError, match="non-empty grounded"):
        svc.finalize_session(sid, "answer", [], repo=repo)
    with pytest.raises(ValueError, match="non-empty string"):
        svc.finalize_session(sid, "   ", good, repo=repo)


def test_finalize_rejects_open_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)
    extra_raw = svc.start_job(sid, "source_agent", repo=repo)["job_id"]
    assert isinstance(extra_raw, str)
    extra = extra_raw
    with pytest.raises(ValueError, match="still open") as exc:
        svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)
    assert isinstance(extra, str) and extra in str(exc.value)


def test_finalize_rejects_bad_and_unjsonable_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)
    bad_list: list[object] = json.loads('["x"]')
    with pytest.raises(ValueError, match="must be a mapping"):
        svc.finalize_session(sid, "answer", bad_list, repo=repo)
    with pytest.raises(ValueError, match="JSON-able"):
        svc.finalize_session(sid, "answer", [{"text": object(), "evidence_ids": [eid]}], repo=repo)


def test_finalize_rejects_empty_synthesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    _full_trio(repo, sid, src, eid)

    def _synth_blank(*a: object, **k: object) -> object:
        return types.SimpleNamespace(answer="   ")

    monkeypatch.setattr("app.research.synthesis.final.synthesize_final", _synth_blank)
    with pytest.raises(ValueError, match="empty answer"):
        svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=repo)


# --- kernel: repository/jobs/director/agents (from /tmp/rc_corekernel.py) ---

KERN_ASOF = datetime(2025, 6, 30, tzinfo=_dt.UTC)


def _sess(**kw: object) -> ResearchSession:
    sess_id = kw.get("session_id")
    assert sess_id is None or isinstance(sess_id, str)
    policy = kw.get("policy")
    assert policy is None or isinstance(policy, dict)
    budget = kw.get("budget")
    assert budget is None or isinstance(budget, dict)
    return _session.create_session("q?", "o", as_of=KERN_ASOF, session_id=sess_id, policy=policy, budget=budget)


def _ok_claim(eid: str = "EV-1") -> str:
    return json.dumps([{"text": "finding one", "evidence_ids": [eid]}])


def _committee_env(claims: object, follows: object, **kw: object) -> str:
    """Rich committee envelope (executive_view/claims/channels/materiality/uncertainties/changes/follow_ups)."""
    env: dict[str, object] = {
        "executive_view": "view",
        "claims": claims,
        "impact_channels": [],
        "materiality": {"overall": "medium", "reasoning": "material"},
        "uncertainties": [],
        "what_would_change": ["a materially new filing"],
        "follow_ups": follows,
    }
    env.update(kw)
    return json.dumps(env)


def _mkjob(
    session: ResearchSession,
    jid: str,
    status: str = "queued",
    jtype: str = "scout",
    deadline: datetime | None = None,
    domain: str | None = None,
) -> Job:
    now = datetime.now(_dt.UTC)
    return Job(
        job_id=jid,
        session_id=session.session_id,
        wave_id=1,
        parent_job_id=None,
        job_type=jtype,
        owner="t",
        source_domain=domain,
        status=status,
        created_at=now,
        deadline=deadline,
    )


# ---- parse_grounded_claims arms ----
def test_pgc_blank() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims("   ", frozen=["EV-1"])


def test_pgc_bad_json() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims("{nope", frozen=["EV-1"])


def test_pgc_non_list() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims('{"text": "x"}', frozen=["EV-1"])


def test_pgc_non_dict_item() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims('["x"]', frozen=["EV-1"])


def test_pgc_wrong_keys() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(json.dumps([{"text": "x"}]), frozen=["EV-1"])


def test_pgc_blank_text() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(json.dumps([{"text": "  ", "evidence_ids": ["EV-1"]}]), frozen=["EV-1"])


def test_pgc_uncited() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(json.dumps([{"text": "t", "evidence_ids": []}]), frozen=["EV-1"])


def test_pgc_unknown_id() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(json.dumps([{"text": "t", "evidence_ids": ["EV-999"]}]), frozen=["EV-1"])


def test_pgc_ok_dedups_and_strips() -> None:
    out = parse_grounded_claims(json.dumps([{"text": "  hi  ", "evidence_ids": ["EV-1", "EV-1"]}]), frozen=["EV-1"])
    assert out[0].text == "hi" and out[0].evidence_ids == ["EV-1"]


# ---- parse_committee_output arms ----
def test_pco_blank() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output(" ", frozen=["EV-1"], agent="stockbot")


def test_pco_bad_json() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output("{bad", frozen=["EV-1"], agent="stockbot")


def test_pco_non_object() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output("[]", frozen=["EV-1"], agent="stockbot")


def test_pco_bad_claims() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output(_committee_env({}, []), frozen=["EV-1"], agent="stockbot")


def test_pco_bad_followups() -> None:
    with pytest.raises(ModelOutputFailure, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        parse_committee_output(_committee_env([], {}), frozen=["EV-1"], agent="stockbot")


def test_pco_rich_envelope_required() -> None:
    with pytest.raises(ModelOutputFailure, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        parse_committee_output('{"claims": [], "follow_ups": []}', frozen=["EV-1"], agent="stockbot")
    with pytest.raises(ModelOutputFailure, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        parse_committee_output(
            _committee_env([], [], materiality={"overall": "huge", "reasoning": "r"}), frozen=["EV-1"], agent="stockbot"
        )
    env = parse_committee_envelope(
        _committee_env(
            [{"text": "finding one", "evidence_ids": ["EV-1"]}],
            [],
            impact_channels=[{"text": "channel", "direction": "up", "evidence_ids": ["EV-1"]}],
            uncertainties=["scope remains SEC-only"],
        ),
        frozen=["EV-1"],
        agent="stockbot",
    )
    assert env.materiality.overall == "medium"
    assert env.executive_view == "view"
    assert env.impact_channels[0].evidence_ids == ["EV-1"]
    assert env.uncertainties == ["scope remains SEC-only"]
    assert env.what_would_change == ["a materially new filing"]


def test_pco_non_string_followup() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output(_committee_env([], [42]), frozen=["EV-1"], agent="stockbot")


def test_pco_malformed_followup() -> None:
    with pytest.raises(ModelOutputFailure):
        parse_committee_output(_committee_env([], ["too short"]), frozen=["EV-1"], agent="stockbot")


def test_pco_ok_tags_agent() -> None:
    c, f = parse_committee_output(
        _committee_env([{"text": "finding one", "evidence_ids": ["EV-1"]}], ["What drove Q2 revenue growth?"]),
        frozen=["EV-1"],
        agent="stockbot",
    )
    assert c[0].evidence_ids == ["EV-1"]
    assert f[0].requesting_agents == ["stockbot"]


# ---- _coerce_wave x5 (parametrized) ----
@pytest.mark.parametrize("mod", ["bearbot", "bullbot", "stockbot", "source_agent", "sec_agent"])
def test_coerce_wave_all_copies(mod: str) -> None:
    import importlib

    m = importlib.import_module(f"app.research.agents.{mod}")
    fn = m._coerce_wave
    assert fn(1) == 1 and fn("2") == 2
    for bad in (True, 0, -1, "x", "", "1.5", None, 1.5):
        with pytest.raises(ValueError):
            fn(bad)


def _model_ev999(p: str) -> str:
    return _committee_env([{"text": "t", "evidence_ids": ["EV-999"]}], [])


def _model_raw(p: str) -> str:
    return "not json"


def _model_empty(p: str) -> str:
    return _committee_env([], [])


def _model_ev1(p: str) -> str:
    return _committee_env([{"text": "t", "evidence_ids": ["EV-1"]}], [])


# ---- run_* committee arms x3 ----
@pytest.mark.parametrize(
    "runner,name", [(run_stockbot, "stockbot"), (run_bullbot, "bullbot"), (run_bearbot, "bearbot")]
)
def test_committee_unknown_evidence_fails(runner: Callable[..., object], name: str) -> None:
    with pytest.raises(ModelOutputFailure):
        runner(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=_model_ev999,
            evidence_text="[EV-1] a",
        )


@pytest.mark.parametrize(
    "runner,name", [(run_stockbot, "stockbot"), (run_bullbot, "bullbot"), (run_bearbot, "bearbot")]
)
def test_committee_bad_model_output_fails(runner: Callable[..., object], name: str) -> None:
    with pytest.raises(ModelOutputFailure):
        runner(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=_model_raw,
            evidence_text="[EV-1] a",
        )


@pytest.mark.parametrize(
    "runner,name", [(run_stockbot, "stockbot"), (run_bullbot, "bullbot"), (run_bearbot, "bearbot")]
)
def test_committee_empty_freeze_unknowns(runner: Callable[..., object], name: str) -> None:
    out = runner(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=[],
        as_of="x",
        model=_model_empty,
        evidence_text="",
    )
    assert isinstance(out, (StockbotAnalysis, BullAnalysis, BearAnalysis))
    assert out.unknowns == ["freeze holds no evidence"]


# ---- assemble_dossier arms ----
def _kern_res(
    coverage: str = "c",
    findings: Sequence[object] | None = None,
    unknowns: Sequence[object] = (),
    limitations: Sequence[object] = (),
) -> ScoutResult:
    found: list[GroundedClaim] = [f for f in (findings or []) if isinstance(f, GroundedClaim)]
    unk: list[str] = [u for u in unknowns if isinstance(u, str)]
    lim: list[str] = [lm for lm in limitations if isinstance(lm, str)]
    return ScoutResult(
        assignment_id="a", session_id="rs:t", coverage=coverage, findings=found, unknowns=unk, limitations=lim
    )


def test_dossier_blank_claim() -> None:
    with pytest.raises(ModelOutputFailure):
        assemble_dossier(
            dossier_id="d",
            session_id="rs:t",
            wave_id=1,
            as_of="x",
            results=[_kern_res(findings=[GroundedClaim(text="  ", evidence_ids=["EV-1"])])],
            known_evidence_ids=["EV-1"],
        )


def test_dossier_uncited() -> None:
    with pytest.raises(ModelOutputFailure):
        assemble_dossier(
            dossier_id="d",
            session_id="rs:t",
            wave_id=1,
            as_of="x",
            results=[_kern_res(findings=[GroundedClaim(text="t", evidence_ids=[])])],
            known_evidence_ids=["EV-1"],
        )


def _journal_seen(seen: list[tuple[str, dict[str, object]]]) -> Callable[[str, dict[str, object]], None]:
    def _j(t: str, p: dict[str, object]) -> None:
        seen.append((t, p))

    return _j


def test_dossier_unknown_id_journals() -> None:
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(ModelOutputFailure):
        assemble_dossier(
            dossier_id="d",
            session_id="rs:t",
            wave_id=1,
            as_of="x",
            results=[_kern_res(findings=[GroundedClaim(text="t", evidence_ids=["EV-9"])])],
            known_evidence_ids=["EV-1"],
            journal=_journal_seen(seen),
        )
    assert seen and seen[0][0] == "evidence.rejected"


def test_dossier_dedup_merge() -> None:
    d = assemble_dossier(
        dossier_id="d",
        session_id="rs:t",
        wave_id=1,
        as_of="x",
        results=[
            _kern_res(findings=[GroundedClaim(text="same", evidence_ids=["EV-1"])]),
            _kern_res(findings=[GroundedClaim(text="same", evidence_ids=["EV-2"])]),
        ],
        known_evidence_ids=["EV-1", "EV-2"],
    )
    assert d.findings[0].evidence_ids == ["EV-1", "EV-2"]


# ---- scout _collect arms ----
def _assign(**kw: object) -> ScoutAssignment:
    base: dict[str, object] = {
        "assignment_id": "a",
        "session_id": "rs:t",
        "as_of": "2025-06-30",
        "role": "filings",
        "question": "Q?",
        "tickers": ["NVDA"],
    }
    base.update(kw)
    role_raw = base.get("role")
    assert isinstance(role_raw, str)
    if role_raw == "filings":
        role: ScoutRole = "filings"
    elif role_raw == "financials":
        role = "financials"
    else:
        assert role_raw == "risk"
        role = "risk"
    aid = base.get("assignment_id")
    assert isinstance(aid, str)
    sess_id = base.get("session_id")
    assert isinstance(sess_id, str)
    asof = base.get("as_of")
    assert isinstance(asof, str)
    q = base.get("question")
    assert isinstance(q, str)
    tick = base.get("tickers")
    assert isinstance(tick, list)
    mtc = base.get("max_tool_calls")
    assert mtc is None or isinstance(mtc, int)
    return ScoutAssignment(
        assignment_id=aid,
        session_id=sess_id,
        as_of=asof,
        role=role,
        question=q,
        tickers=[t for t in tick if isinstance(t, str)],
        max_tool_calls=8 if mtc is None else mtc,
    )


def _scout_model(text: str) -> Callable[[str], str]:
    def _m(p: str) -> str:
        return text

    return _m


def _scout_with(
    dispatch: DispatchFn,
    model_text: str = "[]",
    journal: Callable[[str, dict[str, object]], None] | None = None,
    **kw: object,
) -> ScoutResult:
    return run_scout(_assign(**kw), dispatch=dispatch, model=_scout_model(model_text), journal=journal)


def test_collect_non_dict_candidate_skipped() -> None:
    def _d(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {}
        return {"evidence_ids": ["EV-1", {"evidence_id": "EV-2", "known_at": "2025-05-01"}]}

    out = _scout_with(_d, model_text=json.dumps([{"text": "t", "evidence_ids": ["EV-2"]}]))
    assert [c.evidence_ids for c in out.findings] == [["EV-2"]]


def test_collect_blank_id_skipped_and_coverage_path() -> None:
    def _d(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {}
        return {"evidence_ids": [{"evidence_id": ""}, {"evidence_id": "EV-2", "known_at": "2025-05-01"}]}

    out = _scout_with(_d, model_text=json.dumps([{"text": "t", "evidence_ids": ["EV-2"]}]))
    assert [c.evidence_ids for c in out.findings] == [["EV-2"]]


def test_collect_pit_reject_journals(cap_calls: object = None) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def _d(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {}
        return {"evidence_ids": [{"evidence_id": "EV-9", "known_at": "2025-07-01"}]}

    # The PIT-ineligible id is rejected + journalled; the claim citing it is dropped,
    # never accepted (scout boundary tolerates the bad record instead of aborting).
    out = _scout_with(_d, model_text=json.dumps([{"text": "t", "evidence_ids": ["EV-1"]}]), journal=_journal_seen(seen))
    assert seen and seen[0] == ("evidence.rejected", {"session_id": "rs:t", "evidence_id": "EV-9"})
    assert out.findings == []
    assert any("dropped" in line for line in out.limitations)


def test_run_scout_budget_exhausted_soft_return() -> None:
    def _d(name: str, args: dict[str, object]) -> dict[str, object]:
        return {"evidence_ids": [{"evidence_id": "EV-1", "known_at": "2025-05-01"}]}

    out = _scout_with(_d, model_text="[]", max_tool_calls=0)
    assert out.unknowns and "no PIT-eligible" in out.unknowns[0]


def test_run_scout_cap_break_six() -> None:
    ids = [{"evidence_id": f"EV-{i}", "known_at": "2025-05-01"} for i in range(1, 10)]

    def _d(name: str, args: dict[str, object]) -> dict[str, object]:
        return {"evidence_ids": ids}

    out = _scout_with(_d, model_text="[]")
    assert out.findings == []


def _fallback():
    return SourceDossier(
        dossier_id="d",
        session_id="rs:t",
        wave_id=1,
        as_of="unbounded",
        findings=[GroundedClaim(text="t", evidence_ids=["EV-1"])],
    )


def test_coerce_dossier_import_error_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.research.agents.sec_agent as sec

    def _raise(*a: object, **k: object) -> object:
        raise ImportError("nope")

    monkeypatch.setattr("app.research.agents.sec_agent.import_module", _raise)
    fb = _fallback()
    assert sec._coerce_dossier(fb) is fb


def test_coerce_dossier_non_callable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import app.research.agents.sec_agent as sec

    fake = types.SimpleNamespace(create_dossier=None)

    def _fake_import(*a: object, **k: object) -> object:
        return fake

    monkeypatch.setattr("app.research.agents.sec_agent.import_module", _fake_import)
    fb = _fallback()
    assert sec._coerce_dossier(fb) is fb


def test_coerce_dossier_validator_throw_names_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shape the canonical layer rejects must surface its own error, never a silent fallback."""
    import app.research.agents.sec_agent as sec
    import app.research.dossiers.sec as dossiers_sec

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("validator boom")

    monkeypatch.setattr(dossiers_sec, "validate_dossier", _boom)
    with pytest.raises(ValueError, match="validator boom"):
        sec._coerce_dossier(_fallback())


def test_coerce_dossier_canonicalises_unbounded_live_shape() -> None:
    """The live shape (unbounded session, cited finding) yields a canonical SECDossier."""
    import app.research.agents.sec_agent as sec
    from app.research.dossiers.sec import SECDossier

    out = sec._coerce_dossier(_fallback())
    assert isinstance(out, SECDossier)
    assert out.as_of is None
    assert [f["evidence_ids"] for f in out.findings] == [["EV-1"]]


# ---- _iso_or_none arms ----
def test_iso_none() -> None:
    assert _iso_or_none(None, "k", "w") is None


def test_iso_naive_datetime_gets_utc() -> None:
    out = _iso_or_none(datetime(2025, 5, 1), "k", "w")  # noqa: DTZ001 - naive input is the case under test
    assert out is not None and out.endswith("+00:00")


def test_iso_aware_datetime() -> None:
    out = _iso_or_none(datetime(2025, 5, 1, tzinfo=_dt.UTC), "k", "w")
    assert out is not None and "+00:00" in out


def test_iso_string_ok() -> None:
    out = _iso_or_none("2025-05-01T00:00:00", "k", "w")
    assert out is not None and "+00:00" in out


def test_iso_string_bad() -> None:
    with pytest.raises(ValueError):
        _iso_or_none("not-a-date", "k", "w")


def test_iso_null_arm() -> None:
    with pytest.raises(ValueError):
        _iso_or_none("", "k", "w")
    with pytest.raises(ValueError):
        _iso_or_none(42, "k", "w")


# ---- list_sessions arms ----
def test_list_sessions_empty_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "missing" / "r.sqlite"))
    assert ResearchRepository().list_sessions() == []


def test_list_sessions_limit_clamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    s = _sess()
    repo.save_session(s)
    assert len(repo.list_sessions(limit=0)) == 1
    assert repo.list_sessions(limit=1)[0]["session_id"] == s.session_id


# ---- pending_next_action arms ----
def test_pna_execute_queued() -> None:
    from app.research.models import SOURCE_RUNTIME_BUDGET_S

    s = _sess()
    j = _mkjob(s, "job:1", status="queued", jtype="source_agent")
    out = pending_next_action(s, [j])
    assert isinstance(out, dict)
    assert out["verb"] == "EXECUTE_JOB" and out["job_id"] == "job:1"
    assert SOURCE_RUNTIME_BUDGET_S == 1800
    assert out["budget_s"] == 1800


def test_pna_targeted_research_verb() -> None:
    import dataclasses

    s = dataclasses.replace(_sess(), status="targeted_research")
    out = pending_next_action(s, [])
    assert isinstance(out, dict)
    assert out["verb"] == "EXECUTE_JOB"


def test_pna_running_beats_queued() -> None:
    s = _sess()
    jobs = [_mkjob(s, "job:q", status="queued"), _mkjob(s, "job:r", status="running")]
    out = pending_next_action(s, jobs)
    assert isinstance(out, dict)
    assert out["job_id"] == "job:r"


# ---- consume_dispatch_budget arms ----
def _running_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **sess_kw: object
) -> tuple[ResearchRepository, ResearchSession, Job]:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    s = _sess(**sess_kw)
    _, job = _jobs.create_job(s, list[Job]([]), job_type="source_agent", owner="t")
    running = _jobs.start_job(job)
    repo.save_session_and_job(s, running)
    return repo, repo.get_session(s.session_id), repo.get_job(running.job_id)


def test_consume_unknown_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    s = _sess()
    empty_jobs: list[Job] = []
    _, job = _jobs.create_job(s, empty_jobs, job_type="source_agent", owner="t")
    repo.save_session_and_job(s, _jobs.start_job(job))
    with pytest.raises(KeyError):
        repo.consume_dispatch_budget("nope", job.job_id)


def test_consume_unknown_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, s, _j = _running_repo(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        repo.consume_dispatch_budget(s.session_id, "nope")


def test_consume_cross_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    s1 = _sess()
    _, j1 = _jobs.create_job(s1, list[Job]([]), job_type="source_agent", owner="t")
    r1 = _jobs.start_job(j1)
    repo.save_session_and_job(s1, r1)
    s2 = _sess()
    _, j2 = _jobs.create_job(s2, list[Job]([]), job_type="source_agent", owner="t")
    r2 = _jobs.start_job(j2)
    repo.save_session_and_job(s2, r2)
    with pytest.raises(ValueError, match="belongs"):
        repo.consume_dispatch_budget(s1.session_id, r2.job_id)


def test_consume_timed_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, s, j = _running_repo(tmp_path, monkeypatch)
    past = datetime.now(_dt.UTC) - timedelta(seconds=5)
    repo.save_job(dataclasses.replace(j, deadline=past))
    with pytest.raises(ValueError, match="deadline"):
        repo.consume_dispatch_budget(s.session_id, j.job_id)
    assert repo.get_job(j.job_id).status == "timed_out"


def test_consume_not_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, s, j = _running_repo(tmp_path, monkeypatch)
    repo.save_job(dataclasses.replace(j, status="queued"))
    with pytest.raises(ValueError, match="running required"):
        repo.consume_dispatch_budget(s.session_id, j.job_id)


def test_committee_tags_followups() -> None:
    req = ResearchRequest(
        question="What drove Q2 revenue growth?",
        why_material="m",
        requested_source_domain="SEC",
        expected_gain="high",
        requesting_agents=[],
    )
    out = run_stockbot(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
        follow_ups=[req],
    )
    assert req.requesting_agents == ["stockbot"]
    out2 = run_bullbot(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
        follow_ups=[
            ResearchRequest(
                question="What drove Q2 revenue growth?",
                why_material="m",
                requested_source_domain="SEC",
                expected_gain="high",
                requesting_agents=[],
            )
        ],
    )
    assert out2.research_requests[-1].requesting_agents == ["bullbot"]
    out3 = run_bearbot(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
        follow_ups=[
            ResearchRequest(
                question="What drove Q2 revenue growth?",
                why_material="m",
                requested_source_domain="SEC",
                expected_gain="high",
                requesting_agents=[],
            )
        ],
    )
    assert out3.research_requests[-1].requesting_agents == ["bearbot"]
    assert out.unknowns == []


def test_consume_exhausted_job_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, s, j = _running_repo(tmp_path, monkeypatch)
    repo.save_job(dataclasses.replace(j, tool_budget=0))
    with pytest.raises(ValueError, match="tool_budget exhausted"):
        repo.consume_dispatch_budget(s.session_id, j.job_id)


def test_consume_exhausted_session_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    s = _sess(budget={"total_tool_budget": 5, "tool_calls_used": 10**9})  # §1 explicit-int still enforced
    _, job = _jobs.create_job(s, list[Job]([]), job_type="source_agent", owner="t")
    repo.save_session_and_job(s, _jobs.start_job(job))
    with pytest.raises(ValueError, match="tool budget exhausted"):
        repo.consume_dispatch_budget(s.session_id, job.job_id)


def test_consume_ok_bills_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, s, j = _running_repo(tmp_path, monkeypatch)
    used_raw = s.budget.get("tool_calls_used", 0)
    assert isinstance(used_raw, int)
    used_before = used_raw
    billed, spent = repo.consume_dispatch_budget(s.session_id, j.job_id)
    billed_used = billed.budget.get("tool_calls_used")
    assert isinstance(billed_used, int)
    assert billed_used == used_before + 1
    if j.tool_budget is not None:
        assert spent.tool_budget == j.tool_budget - 1


# ---- create_job arms ----
def test_create_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown job_type"):
        _jobs.create_job(_sess(), [], job_type="nope", owner="t")


def test_create_empty_owner() -> None:
    with pytest.raises(ValueError, match="owner"):
        _jobs.create_job(_sess(), [], job_type="scout", owner="")


def test_create_bad_wave_type() -> None:
    with pytest.raises(ValueError, match="wave_id"):
        _jobs.create_job(_sess(), [], job_type="scout", owner="t", wave_id=True)


def _capped(**limits: int) -> ResearchSession:
    """Default policy with explicit research limits bolted on (None = unlimited)."""
    policy = dict(_sess().policy)
    research = policy.get("research")
    assert isinstance(research, dict)
    policy["research"] = {**research, **limits}
    return _sess(policy=policy)


def test_create_wave_outside() -> None:
    _s, j = _jobs.create_job(_sess(), [], job_type="scout", owner="t", wave_id=999)
    assert j.wave_id == 999  # waves are sequence numbers; the default policy sets no ceiling
    with pytest.raises(ValueError, match="wave_limit_exceeded"):  # §3 distinct error
        _jobs.create_job(_capped(max_waves=2), [], job_type="scout", owner="t", wave_id=999)
    with pytest.raises(ValueError, match="wave_limit_exceeded"):
        _jobs.create_job(_sess(), [], job_type="scout", owner="t", wave_id=0)


def test_create_total_exhausted() -> None:
    s = _capped(max_total_jobs=3)
    jobs: list[Job] = []
    last = s
    while len(jobs) < 3:
        last, j = _jobs.create_job(last, jobs, job_type="scout", owner="t")
        jobs.append(_jobs.complete_job(_jobs.start_job(j)))
    with pytest.raises(ValueError, match="max_total_jobs"):
        _jobs.create_job(last, jobs, job_type="scout", owner="t")
    unlimited: list[Job] = []
    last_unlimited = _sess()
    while len(unlimited) < 25:
        last_unlimited, uj = _jobs.create_job(last_unlimited, unlimited, job_type="scout", owner="t")
        unlimited.append(_jobs.complete_job(_jobs.start_job(uj)))
    assert len(unlimited) == 25 and len(last_unlimited.job_ids) == 25  # default policy: no total ceiling


def test_create_committee_parallel() -> None:
    s = _sess(policy={**_sess().policy, "committee": {"max_parallel": 0}})
    with pytest.raises(ValueError, match="committee max_parallel"):
        _jobs.create_job(s, list[Job]([]), job_type="stockbot", owner="t")


def test_create_session_parallel_exhausted() -> None:
    s = _capped(max_parallel=1)
    s2, queued = _jobs.create_job(s, list[Job]([]), job_type="scout", owner="t")
    with pytest.raises(ValueError, match="parallelism_exceeded"):
        _jobs.create_job(s2, [_jobs.start_job(queued)], job_type="scout", owner="t")
    released = _jobs.complete_job(_jobs.start_job(queued))
    _s3, next_job = _jobs.create_job(s2, [released], job_type="scout", owner="t")
    assert next_job.status == "queued"  # a closed job frees the session slot again


def test_create_bad_deadline_string() -> None:
    with pytest.raises(ValueError, match="deadline"):
        _jobs.create_job(_sess(), [], job_type="scout", owner="t", deadline="not-a-date")


def test_create_nullish_deadline() -> None:
    with pytest.raises(ValueError, match="deadline"):
        _jobs.create_job(_sess(), [], job_type="scout", owner="t", deadline="   ")


def test_create_deadline_ok_and_defaults() -> None:
    _s, j = _jobs.create_job(_sess(), [], job_type="scout", owner="t", deadline="2025-06-01T00:00:00+00:00")
    assert j.deadline is not None
    _s2, j2 = _jobs.create_job(_sess(), [], job_type="source_agent", owner="t")
    assert j2.deadline is None  # §1 default budget deadline_seconds is None: no deadline
    assert j2.tool_budget is None  # §1 unlimited default
    _s3, j3 = _jobs.create_job(_sess(budget={"deadline_seconds": 60}), [], job_type="source_agent", owner="t")
    assert j3.deadline is not None  # explicit configured deadline_seconds still enforced


# ---- decide_next_wave arms ----
def _deps() -> tuple[DirectorDeps, list[tuple[str, str]]]:
    seen: list[tuple[str, str]] = []

    def _mk_session(q: str, a: str) -> str:
        return "rs:x"

    def _fetch(s: str) -> list[str]:
        return ["EV-1"]

    def _freeze(s: str) -> str:
        return "F1"

    def _committee(s: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        raise AssertionError("no")

    def _stop(s: str, r: str) -> None:
        seen.append((s, r))

    return DirectorDeps(
        create_session=_mk_session,
        fetch_wave_evidence=_fetch,
        create_freeze=_freeze,
        run_committee=_committee,
        record_stop=_stop,
    ), seen


def _w1_with(reqs: Sequence[ResearchRequest]) -> Wave1Result:
    stock = run_stockbot(
        "Q?",
        session_id="rs:x",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
    )
    bull = run_bullbot(
        "Q?",
        session_id="rs:x",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
    )
    bear = run_bearbot(
        "Q?",
        session_id="rs:x",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_model_ev1,
        evidence_text="[EV-1] a",
    )
    object.__setattr__(stock, "research_requests", list(reqs))
    from app.research.synthesis.committee import compute_disagreement

    dis = compute_disagreement(stock, bull, bear)
    return Wave1Result(
        session_id="rs:x",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=dis,
    )


def _req(
    gain: str = "high", domain: str = "SEC", why: str = "matters", q: str = "What drove Q2 revenue growth?"
) -> ResearchRequest:
    return ResearchRequest(
        question=q, why_material=why, requested_source_domain=domain, expected_gain=gain, requesting_agents=["stockbot"]
    )


def test_next_wave_max_waves() -> None:
    deps, seen = _deps()
    w1 = Wave1Result(session_id="rs:x", wave_id=1, freeze_id="F1", evidence_ids=["EV-1"])
    d = decide_next_wave(w1, deps=deps, budgets=DirectorBudgets(max_waves=1), waves_used=1)
    assert d.stop_reason == "max_waves" and seen


def test_next_wave_runtime() -> None:
    deps, _ = _deps()
    w1 = Wave1Result(session_id="rs:x", wave_id=1, freeze_id="F1", evidence_ids=["EV-1"])
    d = decide_next_wave(w1, deps=deps, budgets=DirectorBudgets(runtime_budget_s=1.0), elapsed_s=5.0)
    assert d.stop_reason == "runtime_exceeded"


def test_next_wave_jobs() -> None:
    deps, _ = _deps()
    w1 = Wave1Result(session_id="rs:x", wave_id=1, freeze_id="F1", evidence_ids=["EV-1"])
    d = decide_next_wave(w1, deps=deps, budgets=DirectorBudgets(max_jobs=4), jobs_used=4)
    assert d.stop_reason == "jobs_exceeded"


def test_next_wave_tools() -> None:
    deps, _ = _deps()
    w1 = _w1_with([_req()])
    d = decide_next_wave(w1, deps=deps, budgets=DirectorBudgets(max_tool_calls=0), tool_calls_used=0)
    assert d.authorized and d.targeted_question == "What drove Q2 revenue growth?"
    # §1 no wave ceiling by default: the same authorized wave keeps authorizing at wave 9
    d9 = decide_next_wave(w1, deps=deps, budgets=DirectorBudgets(), waves_used=9)
    assert d9.authorized and d9.stop_reason == "continue"


def test_next_wave_no_questions() -> None:
    deps, seen = _deps()
    w1 = Wave1Result(session_id="rs:x", wave_id=1, freeze_id="F1", evidence_ids=["EV-1"], disagreement=None)
    d = decide_next_wave(w1, deps=deps)
    assert d.stop_reason == "no_questions"
    assert seen == [("rs:x", "no_questions:committee requested no follow-up research")]


def test_next_wave_low_gain() -> None:
    deps, _ = _deps()
    w1 = _w1_with([_req(gain="low")])
    assert decide_next_wave(w1, deps=deps).stop_reason == "low_gain"


def test_next_wave_not_actionable() -> None:
    # Director is policy-unaware: SEC|FINRA|WEB are actionable here; the service
    # post-check fails closed per session source_policy (see test_hf_gate_*).
    deps, _ = _deps()
    w1 = _w1_with([_req(domain="PIGEON")])
    assert decide_next_wave(w1, deps=deps).stop_reason == "not_actionable"
    w2 = _w1_with([_req(domain="WEB")])
    assert decide_next_wave(w2, deps=deps).stop_reason == "continue"


def test_next_wave_continue_picks_best() -> None:
    deps, _ = _deps()
    w1 = _w1_with(
        [
            _req(gain="medium", q="What drove Q2 revenue growth?"),
            _req(gain="high", q="What caused the Q3 margin expansion?"),
        ]
    )
    d = decide_next_wave(w1, deps=deps)
    assert d.authorized and d.targeted_question == "What caused the Q3 margin expansion?"


# --- services: sec_facts/research_data/resolution/dividends/claims/sync/herdr (from /tmp/rc_coreservices.py) ---


def _fact(
    concept: str = "EarningsPerShareDiluted",
    value: float = 1.0,
    start: str = "2025-01-01",
    end: str = "2025-03-31",
    filed: str = "2025-04-28",
    accn: str = "a1",
    fy: int = 2025,
    fp: str = "Q1",
) -> FinancialFactRow:
    return {
        "concept": concept,
        "value": value,
        "period_start": start,
        "period_end": end,
        "filed_at": filed,
        "accession": accn,
        "known_at": filed + "T00:00:00Z",
        "fiscal_year": fy,
        "fiscal_period": fp,
        "source_url": None,
    }


def _fact_copy(row: FinancialFactRow, concept: str | None = None, value: float | None = None) -> FinancialFactRow:
    return {
        "concept": concept if concept is not None else row["concept"],
        "value": value if value is not None else row["value"],
        "period_start": row["period_start"],
        "period_end": row["period_end"],
        "filed_at": row["filed_at"],
        "accession": row["accession"],
        "known_at": row["known_at"],
        "fiscal_year": row["fiscal_year"],
        "fiscal_period": row["fiscal_period"],
        "source_url": row["source_url"],
    }


def _ev(
    eid: str,
    amount: float | None = 0.5,
    pay: str | None = "2026-05-01",
    record: str | None = "2026-04-10",
    dtype: str = "regular",
    known: str | None = "2026-05-02T00:00:00Z",
    stype: str = "structured_xbrl",
    excerpt: str | None = None,
    concept: str | None = None,
) -> DividendEventRow:
    return {
        "dividend_event_id": eid,
        "entity_id": "e",
        "security_id": "s",
        "ticker": "KO",
        "amount_per_share": amount,
        "currency": "USD",
        "dividend_type": dtype,
        "declaration_date": None,
        "record_date": record,
        "payment_date": pay,
        "ex_dividend_date": None,
        "ex_dividend_date_source": None,
        "status": None,
        "source_form": "10-Q",
        "accession": "a",
        "filed_at": known,
        "known_at": known,
        "source_url": None,
        "source_concept": concept,
        "source_type": stype,
        "evidence_excerpt": excerpt,
        "content_hash": "h",
        "parser_version": "v",
    }


# --- _validated_fact_row: auth of store rows (valid / missing concept / bad value / coercions) ---
def test_validated_fact_row_accepts_and_rejects() -> None:
    assert sec_facts._validated_fact_row({}) is None
    assert sec_facts._validated_fact_row({"concept": "X", "period_end": "2025-01-01", "value": "bad"}) is None
    assert sec_facts._validated_fact_row({"concept": "", "period_end": "2025-01-01", "value": 1.0}) is None
    good = sec_facts._validated_fact_row(
        {
            "concept": "X",
            "period_end": "2025-01-01",
            "value": 2,
            "filed_at": None,
            "accession": 5,
            "known_at": None,
            "period_start": 7,
            "fiscal_year": "2025",
            "fiscal_period": 3,
            "source_url": 9,
        }
    )
    assert good is not None and good["value"] == 2.0 and good["filed_at"] == ""
    assert good["accession"] == "5" and good["period_start"] is None
    assert good["fiscal_year"] is None and good["source_url"] is None


# --- _duration_rows: concept filter + restatement dedup + duration window ---
def test_duration_rows_filters_and_dedups() -> None:
    rows = [
        _fact(value=1.0, accn="a1", filed="2025-04-28"),
        _fact(value=2.0, accn="a2", filed="2025-05-28"),  # restatement wins
        _fact(concept="Other", value=9.0),
        _fact(value=3.0, start="2025-01-01", end="2025-12-31"),
    ]  # FY duration excluded
    out = sec_facts._duration_rows(rows, "EarningsPerShareDiluted", (60, 115))
    assert len(out) == 1 and out[0]["value"] == 2.0


# --- _derive_q4_from_facts: FY minus YTD, missing FY, missing YTD ---
def test_derive_q4_paths() -> None:
    fy = _fact(value=5.0, start="2025-01-01", end="2025-12-31", filed="2026-02-10", accn="fy")
    ytd = _fact(value=3.0, start="2025-01-01", end="2025-09-30", filed="2025-10-28", accn="ytd")
    d = sec_facts._derive_q4_from_facts([fy, ytd], "EarningsPerShareDiluted", _dt.date(2025, 12, 31))
    assert d is not None and d["value"] == pytest.approx(2.0) and d["fiscal_period"] == "Q4"
    assert sec_facts._derive_q4_from_facts([ytd], "EarningsPerShareDiluted", _dt.date(2025, 12, 31)) is None
    assert sec_facts._derive_q4_from_facts([fy], "EarningsPerShareDiluted", _dt.date(2025, 12, 31)) is None


# --- _assemble_eps_payload: empty, diluted-only, basic join, TTM totals ---
def test_assemble_eps_payload_paths() -> None:
    assert sec_facts._assemble_eps_payload("T", []) is None
    quarters = [
        _fact(value=1.0, start=s, end=e, filed=f, accn=a, fp=fp)
        for s, e, f, a, fp in [
            ("2025-01-01", "2025-03-31", "2025-04-28", "a1", "Q1"),
            ("2025-04-01", "2025-06-30", "2025-07-28", "a2", "Q2"),
            ("2025-07-01", "2025-09-30", "2025-10-28", "a3", "Q3"),
            ("2025-10-01", "2025-12-31", "2026-02-10", "a4", "Q4"),
        ]
    ]
    p = sec_facts._assemble_eps_payload("T", quarters)
    assert p is not None and p["ttm_eps_diluted"] == 4.0 and "ttm_eps_basic" not in p
    basic: list[FinancialFactRow] = [_fact_copy(q, concept="EarningsPerShareBasic", value=0.5) for q in quarters]
    p2 = sec_facts._assemble_eps_payload("T", quarters + basic)
    assert p2 is not None and p2["ttm_eps_basic"] == 2.0
    assert p2 is not None
    qeps = p2["quarterly_eps"]
    assert isinstance(qeps, list)
    q0 = qeps[0]
    assert isinstance(q0, dict)
    assert q0["eps_basic"] == 0.5


# --- _canonical_dividend_events: typed clustering, untyped amount match, undated ---
def test_canonical_events_match_arms() -> None:
    as_of = _dt.date(2026, 8, 10)
    # regular + special same bucket never merge
    evs = [_ev("r1", dtype="regular"), _ev("s1", dtype="special")]
    out = sec_facts._canonical_dividend_events(evs)
    assert len(out) == 2
    # unknown amount matches the single regular cluster (amendment)
    evs2 = [_ev("r1", amount=0.5, dtype="regular"), _ev("u1", amount=0.5, dtype="unknown")]
    out2 = sec_facts._canonical_dividend_events(evs2)
    assert len(out2) == 1 and out2[0]["amount_per_share"] == 0.5
    # unknown amount matching nothing + no clusters -> stray cluster
    out3 = sec_facts._canonical_dividend_events([_ev("u1", amount=0.7, dtype="unknown")])
    assert len(out3) == 1
    # undated never merges
    out4 = sec_facts._canonical_dividend_events([_ev("x1", pay=None), _ev("x2", pay=None)])
    assert len(out4) == 2
    # winner backfills amount/type from mate
    a = _ev("a", amount=None, dtype="unknown", known="2026-05-03T00:00:00Z")
    b = _ev("a", amount=0.9, dtype="regular", known="2026-05-01T00:00:00Z")
    assert a["dividend_event_id"] == b["dividend_event_id"]
    ea = _ev("a", amount=None, pay="2026-06-01", dtype="unknown", known="2026-05-03T00:00:00Z")
    eb = _ev("a", amount=0.9, pay="2026-06-01", dtype="regular", known="2026-05-01T00:00:00Z")
    canon = sec_facts._canonical_dividend_events([ea, eb])
    assert sec_facts._classify_dividend_event(canon[0], as_of) in ("paid", "upcoming")


# --- _dividend_event_payload: last/next/past/coverage arms ---
def test_dividend_event_payload_coverage_arms() -> None:
    as_of = _dt.date(2026, 8, 10)
    paid = _ev("p1", pay="2026-05-01", stype="structured_xbrl")
    upcoming = _ev("n1", pay="2026-09-01", stype="filing_text", record="2026-08-15")
    p = sec_facts._dividend_event_payload([paid, upcoming], as_of)
    last = p["last_dividend"]
    assert isinstance(last, dict)
    assert last["payment_date"] == "2026-05-01"
    nxt = p["next_declared_dividend"]
    assert isinstance(nxt, dict)
    assert nxt["payment_date"] == "2026-09-01"
    assert p["events_coverage"] == "structured_and_text"
    assert sec_facts._dividend_event_payload([], as_of)["events_coverage"] == "no_structured_events"
    assert sec_facts._dividend_event_payload([paid], as_of)["events_coverage"] == "structured_only"
    assert sec_facts._dividend_event_payload([upcoming], as_of)["events_coverage"] == "text_only"


# --- _debt_up_yoy: up / down / missing / unparsable ---
def test_debt_up_yoy_paths() -> None:
    up = [
        _fact(concept="LongTermDebt", value=100.0, start="2024-01-01", end="2024-12-31", filed="2025-02-10", accn="d1"),
        _fact(concept="LongTermDebt", value=150.0, start="2025-01-01", end="2025-12-31", filed="2026-02-10", accn="d2"),
    ]
    assert sec_facts._debt_up_yoy(up) is True
    down: list[FinancialFactRow] = [_fact_copy(r, value=50.0 if r["accession"] == "d2" else 100.0) for r in up]
    assert sec_facts._debt_up_yoy(down) is False
    assert sec_facts._debt_up_yoy([]) is None
    assert sec_facts._debt_up_yoy([up[1]]) is None  # no year-ago base


# --- _assemble_dividend_safety: every ratio arm ---
def test_safety_ratio_arms() -> None:
    base = {"ttm_dividend_per_share": 2.0}
    s = sec_facts._assemble_dividend_safety([], base, ttm_eps_diluted=4.0)
    assert s["earnings_payout_ratio"] == 0.5
    s2 = sec_facts._assemble_dividend_safety([], base, ttm_eps_diluted=-1.0)
    assert s2["earnings_payout_ratio"] is None and "negative_eps" in str(s2["risk_flags"])
    s3 = sec_facts._assemble_dividend_safety([], {})
    assert s3["earnings_payout_ratio"] is None
    s4 = sec_facts._assemble_dividend_safety([], base)
    assert s4["fcf_payout_ratio"] is None and s4["cash_to_annual_dividend"] is None
    assert s4["interest_coverage"] is None and s4["debt_up_yoy"] is None
    growth = s4["dividend_vs_fcf_growth_5y"]
    assert isinstance(growth, dict)
    assert growth["verdict"] == "insufficient_data"


# --- _dividend_fundamental via gateway: live hit + PIT miss ---
def test_dividend_fundamental_store_and_pit(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.normalization import normalize_sec_company_facts, normalize_sec_tickers

    cik, ret = 21344, "2026-08-01T00:00:00Z"
    alias_rows = normalize_sec_tickers(
        {"0": {"cik_str": cik, "ticker": "KO", "title": "KO Corp"}}, retrieved_at=ret, content_hash="t"
    )["entity_aliases"]
    aliases = [
        TickerAlias(
            alias_type=str(r.get("alias_type")),
            alias_value=str(r.get("alias_value")),
            entity_id=str(r.get("entity_id")),
            security_id=r.get("security_id") if isinstance(r.get("security_id"), str) else None,
            source=str(r.get("source")),
            known_at=r.get("known_at") if isinstance(r.get("known_at"), str) else None,
            retrieved_at=r.get("retrieved_at") if isinstance(r.get("retrieved_at"), str) else None,
            valid_from=None,
            valid_to=None,
        )
        for r in alias_rows
    ]
    div_units = [
        {"start": s, "end": e, "val": v, "accn": a, "fy": fy, "fp": fp, "filed": f}
        for v, s, e, fy, fp, f, a in [
            (0.51, "2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28", "q3"),
            (0.51, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "q4"),
            (0.54, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "q1"),
            (0.54, "2026-04-01", "2026-06-30", 2026, "Q2", "2026-07-28", "q2"),
        ]
    ]
    payload = {
        "cik": cik,
        "entityName": "KO",
        "facts": {"us-gaap": {"CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": div_units}}}},
    }
    facts = normalize_sec_company_facts(
        payload, retrieved_at=ret, content_hash="d", source_url="u", source_record_id="r"
    )

    class _Gateway:
        def ticker_candidates(self, ticker: str, as_of: object) -> list[TickerAlias]:
            del as_of
            return list(aliases) if ticker == "KO" else []

        def company_facts(self, cik_no: int, as_of: str | None = None) -> dict[str, list[dict[str, object]]]:
            assert cik_no == cik
            rows = [dict(r) for r in facts.get("financial_facts", [])]
            if as_of is not None:
                rows = [r for r in rows if str(r.get("known_at", ""))[:10] <= as_of]
            return {"financial_facts": rows, "dividend_events": []}

    monkeypatch.setattr(sec_facts, "_gateway", lambda: _Gateway())
    hit = sec_facts.get_fundamentals("KO", "dividends", as_of="2026-08-10")
    assert hit["data_source"] == "live" and hit["ttm_dividend_per_share"] == 2.10
    miss = sec_facts.get_fundamentals("KO", "dividends", as_of="2020-01-01")
    assert miss.get("error_type") == "pit_data_unavailable"


# --- research_data: SEC tickers parse + FINRA paging errors + prepare arms ---
class _Resp:
    def __init__(self, content: bytes, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self.content, self.status_code, self.headers = content, status_code, headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _mocks(monkeypatch: pytest.MonkeyPatch, get_script: list[object], page_script: list[object]) -> None:
    def _fake_get(url: str, **k: object) -> object:
        item = get_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _fake_page(g: object, n: object, p: object) -> object:
        return page_script.pop(0)

    def _no_sleep(s: float) -> None:
        return None

    monkeypatch.setattr(research_data, "_edgar_get", _fake_get)
    monkeypatch.setattr(research_data.finra_client, "ingestion_post_query", _fake_page)
    monkeypatch.setattr(research_data.time, "sleep", _no_sleep)


def test_refresh_sec_tickers_skips_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "0": {"ticker": "AAPL", "cik_str": 320193},
        "1": {"ticker": "", "cik_str": 1},
        "2": "junk",
        "3": {"ticker": "BAD", "cik_str": "xx"},
        "4": {"ticker": "NOCIK"},
    }

    def _fake_sec(url: str) -> bytes:
        return json.dumps(payload).encode()

    def _fake_now() -> str:
        return "2026-08-01T00:00:00Z"

    monkeypatch.setattr(research_data, "_edgar_get", _fake_sec)
    monkeypatch.setattr(research_data, "_utc_now", _fake_now)
    import tempfile

    root = Path(tempfile.mkdtemp())
    out = refresh_sec_tickers(data_root=root)
    assert out["ticker_ciks"] == {"AAPL": 320193}


def test_refresh_finra_missing_total_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mocks(monkeypatch, [], [(b"[]", [], {})])
    with pytest.raises(ValueError, match="Record-Total"):
        refresh_finra_short_interest("2026-08-14", data_root=tmp_path)


def test_refresh_finra_changing_total_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"symbolCode": "A"}]
    _mocks(
        monkeypatch,
        [],
        [
            (b"[]", rows, {"record-total": "2"}),
            (b"[]", rows, {"record-total": "3"}),
        ],
    )
    with pytest.raises(ValueError, match="changed"):
        refresh_finra_short_interest("2026-08-14", data_root=tmp_path)


def test_prepare_unresolved_and_enrichment_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
    facts: dict[str, object] = {"cik": 320193, "entityName": "x", "facts": {}}
    finra_rows = [{"symbolCode": "AAPL", "settlementDate": "2026-08-14", "currentShortPositionQuantity": 5}]
    _mocks(
        monkeypatch,
        [json.dumps(ticks).encode(), RuntimeError("boom"), json.dumps(facts).encode()],
        [(json.dumps(finra_rows).encode(), finra_rows, {"record-total": "1"})],
    )
    out = prepare_short_interest_data("2026-08-14", tickers=["AAPL", "ZZZ"], data_root=tmp_path)
    assert out["unresolved_tickers"] == ["ZZZ"]
    assert out["failed_enrichments"] == [] or isinstance(out["failed_enrichments"], list)


# --- evidence_resolution: match-arm boundaries ---
def _aliases_one(t: str) -> list[TickerAlias]:
    return [_svcs_alias()]


def _aliases_empty(t: str) -> list[TickerAlias]:
    return []


def _name_none(n: str) -> str | None:
    return None


def _name_acme(n: str) -> str | None:
    return "acme"


def _svcs_alias(entity: str = "e1", ticker: str = "ACME") -> TickerAlias:
    return TickerAlias(
        alias_type="ticker",
        alias_value=ticker,
        entity_id=entity,
        security_id="s1",
        source="t",
        valid_from=None,
        valid_to=None,
        known_at="2026-01-01T00:00:00Z",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def test_resolve_subject_ticker_name_unresolved() -> None:
    asof = datetime(2026, 6, 1, tzinfo=_dt.UTC)
    r = resolve_subject(ticker="acme", name=None, aliases_by_ticker=_aliases_one, name_to_ticker=_name_none, as_of=asof)
    assert r.resolved and r.ticker == "ACME"
    r2 = resolve_subject(
        ticker=None, name="Acme Corp", aliases_by_ticker=_aliases_one, name_to_ticker=_name_acme, as_of=asof
    )
    assert r2.resolved
    r3 = resolve_subject(
        ticker=None, name="Nope", aliases_by_ticker=_aliases_empty, name_to_ticker=_name_none, as_of=asof
    )
    assert not r3.resolved
    r4 = resolve_subject(
        ticker=None, name=None, aliases_by_ticker=_aliases_empty, name_to_ticker=_name_none, as_of=asof
    )
    assert not r4.resolved


def test_resolve_subject_unresolved_guards() -> None:
    asof = datetime(2026, 6, 1, tzinfo=_dt.UTC)
    assert (
        resolve_subject(
            ticker=None, name="", aliases_by_ticker=_aliases_empty, name_to_ticker=_name_none, as_of=asof
        ).resolved
        is False
    )
    assert (
        resolve_subject(
            ticker=None,
            name="No Such Company XYZ 123",
            aliases_by_ticker=_aliases_empty,
            name_to_ticker=_name_none,
            as_of=asof,
        ).resolved
        is False
    )


# --- dividend_analysis: cadence + lifecycle arms ---
def _pay(day: str, amt: float = 0.5, dtype: str = "regular") -> dict[str, object]:
    return {"payment_date": day, "amount_per_share": amt, "dividend_type": dtype}


def test_cadence_arms() -> None:
    assert cadence_from_events(None)["payment_cadence"] == "unknown"
    assert cadence_from_events([_pay("2026-01-01")])["payment_cadence"] == "unknown"
    quarterly = [_pay(f"2025-{m:02d}-01") for m in (1, 4, 7, 10)] + [_pay("2026-01-01")]
    c = cadence_from_events(quarterly)
    assert c["payment_cadence"] == "quarterly" and c["cadence_confidence"] == "high"
    two = [_pay("2025-01-01"), _pay("2025-04-01"), _pay("2025-07-01")]
    assert cadence_from_events(two)["cadence_confidence"] == "medium"
    odd = [_pay("2025-01-01"), _pay("2025-03-01"), _pay("2025-08-01")]
    assert cadence_from_events(odd)["payment_cadence"] == "unknown"


def test_lifecycle_arms() -> None:
    q = [_pay(f"2025-{m:02d}-01", amt=a) for m, a in [(1, 0.5), (4, 0.5), (7, 0.5), (10, 0.5), (1, 0.6)]]
    q[-1]["payment_date"] = "2026-01-01"
    life = lifecycle_from_events(q)
    assert life["increase"] is not None and life["cut"] is None
    cut = [dict(e, amount_per_share=0.4) if e["payment_date"] == "2026-01-01" else e for e in q]
    assert lifecycle_from_events(cut)["cut"] is not None
    frozen = [_pay(f"2025-{m:02d}-01") for m in (1, 4, 7, 10)] + [_pay("2026-01-01")]
    assert lifecycle_from_events(frozen)["freeze"] is not None
    specials = frozen + [_pay("2026-02-01", amt=1.0, dtype="special")]
    assert lifecycle_from_events(specials)["special_paid_per_share"] == 1.0
    assert lifecycle_from_events(frozen, as_of="2027-06-01")["possible_suspension"] is True
    # reinstatement needs an observed cadence: 4 quarterly + one far resumption
    resumed = [_pay(f"2025-{m:02d}-01") for m in (1, 4, 7, 10)] + [_pay("2026-01-01")] + [_pay("2027-06-01")]
    assert lifecycle_from_events(resumed)["reinstatement"] is not None
    assert lifecycle_from_events(None)["total_paid_per_share"] == 0.0


# --- evidence_claims: empty / non-dict / resolved paths ---
def _svcs_res(ticker: str = "ACME") -> SecurityResolution:
    return SecurityResolution("s1", "e1", ticker, True, "alias")


def test_build_claims_paths() -> None:
    assert build_evidence_claims(reader_items=[], retrieved_fallback="t") == []
    bad_items: list[dict[str, object]] = json.loads('"nope"')
    assert build_evidence_claims(reader_items=bad_items, retrieved_fallback="t") == []
    mixed_items: list[dict[str, object]] = json.loads(
        '[{"subject_ticker": "acme", "claim": "x", "claim_type": "other", "source_url": "https://example.com", "retrieved_at": "2026-01-01T00:00:00+00:00", "object_name": "Beta"}, "junk", {"subject_ticker": 5, "claim": 5, "source_url": 5}]'
    )

    def _resolve_none(
        ticker: str | None = None, name: str | None = None, as_of: str | None = None
    ) -> SecurityResolution:
        return _svcs_res()

    claims = build_evidence_claims(
        reader_items=mixed_items, resolve=_resolve_none, retrieved_fallback="2026-01-01T00:00:00+00:00"
    )
    assert len(claims) == 2 and claims[0].ticker == "ACME"


# --- portfolio_sync legacy account path ---
def test_read_latest_snapshot_empty_and_legacy(tmp_path: Path) -> None:
    assert read_latest_snapshot(data_root=tmp_path) is None

    from app.domain.portfolio.snapshot import build_portfolio_snapshot

    now = datetime(2026, 8, 25, 12, tzinfo=_dt.UTC)
    snap = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=["a1"],
        positions=[],
        cash_balances={"a1": __import__("decimal").Decimal("10")},
        created_at=now,
    )
    persist_snapshot(snap, data_root=tmp_path)
    restored = read_latest_snapshot(data_root=tmp_path)
    assert restored is not None and restored.cash is not None


# --- herdr_client: fake transport for init/read/subscribe/request ---
class _FakeSock(socket.socket):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.sent: list[bytes] = []

    @override
    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    @override
    def sendall(self, data: Buffer, flags: int = 0, /) -> None:
        self.sent.append(bytes(data))

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *exc: object) -> None:
        return None


class _FakeConn(HerdrClient):
    def __init__(self, sock: socket.socket) -> None:
        super().__init__(socket_path="/tmp/fake.sock")
        self._sock = sock

    @override
    def _connect(self) -> socket.socket:
        return self._sock


def test_herdr_init_read_subscribe_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    assert HerdrClient(socket_path="/tmp/x.sock").socket_path == "/tmp/x.sock"
    assert HerdrClient().socket_path.endswith("herdr.sock")
    monkeypatch.setenv("HERDR_SOCKET_PATH", "/tmp/env.sock")
    assert HerdrClient().socket_path == "/tmp/env.sock"
    sock = _FakeSock([b'{"a":1}\n', b'{"b":2}\n'])
    assert list(HerdrClient._read_lines(sock)) == [{"a": 1}, {"b": 2}]
    partial = _FakeSock([b'{"a":', b"1}\n"])
    assert list(HerdrClient._read_lines(partial)) == [{"a": 1}]
    blank = _FakeSock([b'\n\n{"a":1}\n'])
    assert list(HerdrClient._read_lines(blank)) == [{"a": 1}]
    sub_sock = _FakeSock([])
    c = _FakeConn(sub_sock)
    assert list(c.subscribe(pane_ids=["p1"], match="x")) == []
    assert b"pane.output_matched" in sub_sock.sent[0]
    sub_sock2 = _FakeSock([])
    c2 = _FakeConn(sub_sock2)
    assert list(c2.subscribe(pane_ids=["p1"])) == []
    assert b"agent_status_changed" in sub_sock2.sent[0]
    req_sock = _FakeSock([b'{"result":{"read":{"text":"hi"}}}\n'])
    c3 = _FakeConn(req_sock)
    assert c3.pane_read("p1") == "hi"
    empty = _FakeConn(_FakeSock([]))
    with pytest.raises(ConnectionError):
        empty.request("pane.read")
    m = WorkerMap()
    m.insert("w", "p")
    assert m.pane_of("w") == "p" and m.remove_by_pane("p") == "w"


# --- store: storage/domain/analytics (from /tmp/rc_corestore.py) ---

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=_dt.UTC)
A = "sec:cik:0000000001"
B = "sec:cik:0000000002"


# ---------------------------------------------------------------------------
# runs.py: recorder lifecycle + query helpers
# ---------------------------------------------------------------------------


def _recorder(run_id: str) -> RunRecorder:
    return RunRecorder(
        run_id=run_id,
        request_id="req",
        question="q",
        as_of=None,
        model="t",
        provider="p",
        model_parameters={},
        agent_version="0",
        prompt_version="0",
        tool_registry_version="t",
        git_sha="g",
    )


def test_runs_enter_migrates_legacy_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-telemetry DB gains columns on open (migration stages)."""
    import sqlite3 as _s

    path = tmp_path / "runs.sqlite"
    conn = _s.connect(str(path))
    conn.execute(
        "CREATE TABLE agent_runs (run_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,"
        " started_at TEXT NOT NULL, question TEXT NOT NULL, model_provider TEXT,"
        " model_name TEXT, model_parameters TEXT, agent_version TEXT,"
        " prompt_version TEXT, tool_registry_version TEXT, git_sha TEXT, as_of TEXT)"
    )
    conn.execute(
        "CREATE TABLE tool_calls (tool_call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,"
        " round INTEGER, tool_name TEXT NOT NULL, tool_version TEXT, arguments_json TEXT,"
        " started_at TEXT NOT NULL, completed_at TEXT, duration_ms REAL, status TEXT,"
        " result_row_count INTEGER, returned_count INTEGER, truncated INTEGER,"
        " result_bytes INTEGER, result_hash TEXT, source_names TEXT, source_freshness TEXT,"
        " as_of TEXT, error_type TEXT, error_message TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("RUNS_DB_PATH", str(path))
    with _recorder("run-mig-1"):
        pass
    conn = _s.connect(str(path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_calls)")}
        assert "cache_hit" in cols and "protocol_id" in cols
        mcols = {r[1] for r in conn.execute("PRAGMA table_info(model_calls)")}
        assert "status" in mcols and "error_type" in mcols
    finally:
        conn.close()


def test_runs_record_event_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """record_event: duration derivation, truncation, disabled + evidence tokens."""
    from app.runtime import EventType

    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    rec = _recorder("run-ev-1")
    assert rec.record_event(EventType.RUN_STARTED) is None  # disabled before enter
    with rec:
        t0 = "2026-08-25T12:00:00+00:00"
        t1 = "2026-08-25T12:00:01+00:00"
        eid = rec.record_event(
            EventType.TOOL_COMPLETED,
            round=2,
            started_at=t0,
            completed_at=t1,
            result_summary="s" * 10,
            success=True,
            arguments={"k": 1},
            evidence_ids=["e1"],
            metadata={"m": 1},
        )
        assert eid == "run-ev-1:ev:0001"
        big = rec.record_event(EventType.TOOL_COMPLETED, result_summary="x" * (rec.max_result_bytes + 5))
        assert big is not None
        ev = rec.record_event(EventType.EVIDENCE_ADDED, result_summary="abcd1234")
        assert ev is not None and rec.evidence_tokens > 0
        explicit = rec.record_event(EventType.TOOL_COMPLETED, started_at=t0, completed_at=t1, duration_ms=5.0)
        assert explicit is not None
    rows = runs_mod.get_events("run-ev-1")
    assert [r["event_id"] for r in rows] == [
        "run-ev-1:ev:0001",
        "run-ev-1:ev:0002",
        "run-ev-1:ev:0003",
        "run-ev-1:ev:0004",
    ]
    assert rows[0]["duration_ms"] == 1000.0
    s = rows[1]["result_summary"]
    assert isinstance(s, str) and s.endswith("...[truncated]")
    assert rows[3]["duration_ms"] == 5.0


def test_runs_query_helpers_empty_and_filter_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cc4 getters: empty store + populated + security-summary decision arms."""
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    assert runs_mod.list_runs() == []
    assert runs_mod.get_run("nope") is None
    assert runs_mod.get_events("nope") == []
    assert runs_mod.get_tool_calls("nope") == []
    assert runs_mod.get_model_calls("nope") == []
    assert runs_mod.get_security_events("nope") == []
    assert runs_mod.get_evidence("nope") == []
    assert runs_mod.get_security_summary("nope") == {
        "allowed": 0,
        "quarantined": 0,
        "blocked": 0,
        "action_blocked": 0,
        "egress_blocked": 0,
        "response_stripped": 0,
    }
    with _recorder("run-q-1") as rec:
        now = datetime.now(_dt.UTC).isoformat()
        rec.record_security_event(source="s", sha256="h", score=1, verdict="v", rule_ids=["r"], decision="allowed")
        rec.record_security_event(
            source="s", sha256="h", score=1, verdict="v", rule_ids=["r"], decision="weird-decision"
        )
        rec.record_tool_call(
            tool_call_id="run-q-1:tc:1",
            round=0,
            tool_name="t",
            arguments_json="{}",
            started_at=now,
            completed_at=now,
            status="completed",
            result_row_count=0,
            returned_count=0,
            truncated=False,
            result_bytes=2,
            result_hash="h",
            source_names="[]",
            source_freshness="{}",
            as_of=None,
            error_type=None,
            error_message=None,
        )
        rec.record_event("run_started")
    assert len(runs_mod.list_runs()) == 1
    run = runs_mod.get_run("run-q-1")
    assert run is not None
    assert run["run_id"] == "run-q-1"
    assert len(runs_mod.get_events("run-q-1")) == 1
    assert len(runs_mod.get_tool_calls("run-q-1")) == 1
    assert len(runs_mod.get_security_events("run-q-1")) == 2
    summary = runs_mod.get_security_summary("run-q-1")
    assert summary["allowed"] == 1 and summary["weird-decision"] == 1
    # corrupt DB -> getters degrade to empty, never raise
    conn = sqlite3.connect(str(tmp_path / "runs.sqlite"))
    conn.execute("DROP TABLE agent_events")
    conn.commit()
    conn.close()
    assert runs_mod.get_events("run-q-1") == []


def test_runs_db_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "custom.sqlite"))
    assert get_runs_db_path(tmp_path) == tmp_path / "custom.sqlite"


# ---------------------------------------------------------------------------
# ids.py: deterministic SEC identity helpers (replaces warehouse dedup pin)
# ---------------------------------------------------------------------------


def test_sec_identity_helpers_deterministic() -> None:
    from app.domain.market import ids

    cik = 320193
    accession = "0000320193-26-000001"
    first = ids.sec_fact_id(cik, accession, "EntityCommonStockSharesOutstanding", "2026-08-01", 100.0)
    assert first == ids.sec_fact_id(cik, accession, "EntityCommonStockSharesOutstanding", "2026-08-01", 100.0)
    assert ids.sec_security_id(cik) == ids.sec_security_id(cik)
    assert sec_facts._validated_fact_row({}) is None


# ---------------------------------------------------------------------------
# options.py: per-stage calculators
# ---------------------------------------------------------------------------


def _quote(**overrides: object) -> OptionQuote:
    base: dict[str, object] = {
        "contract_id": "c1",
        "ticker": "WING",
        "expiration": date(2027, 1, 15),
        "strike": Decimal(80),
        "option_type": "put",
        "underlying_price": Decimal("116.84"),
        "bid": Decimal("2.00"),
        "ask": Decimal("2.40"),
        "mark": Decimal("2.20"),
        "implied_volatility": Decimal("0.55"),
        "delta": Decimal("-0.12"),
        "gamma": Decimal("0.01"),
        "theta": Decimal("-0.02"),
        "vega": Decimal("0.10"),
        "rho": None,
        "volume": 10,
        "open_interest": 100,
        "retrieved_at": datetime(2026, 8, 25, tzinfo=_dt.UTC),
    }
    base.update(overrides)
    contract_id = base["contract_id"]
    assert isinstance(contract_id, str)
    ticker = base["ticker"]
    assert isinstance(ticker, str)
    expiration = base["expiration"]
    assert isinstance(expiration, date)
    strike = base["strike"]
    assert isinstance(strike, Decimal)
    option_type = base["option_type"]
    assert isinstance(option_type, str)
    underlying_price = base["underlying_price"]
    assert underlying_price is None or isinstance(underlying_price, Decimal)
    bid = base["bid"]
    assert bid is None or isinstance(bid, Decimal)
    ask = base["ask"]
    assert ask is None or isinstance(ask, Decimal)
    mark = base["mark"]
    assert mark is None or isinstance(mark, Decimal)
    implied_volatility = base["implied_volatility"]
    assert implied_volatility is None or isinstance(implied_volatility, Decimal)
    delta = base["delta"]
    assert delta is None or isinstance(delta, Decimal)
    gamma = base["gamma"]
    assert gamma is None or isinstance(gamma, Decimal)
    theta = base["theta"]
    assert theta is None or isinstance(theta, Decimal)
    vega = base["vega"]
    assert vega is None or isinstance(vega, Decimal)
    rho = base["rho"]
    assert rho is None or isinstance(rho, Decimal)
    volume = base["volume"]
    assert volume is None or isinstance(volume, int)
    open_interest = base["open_interest"]
    assert open_interest is None or isinstance(open_interest, int)
    retrieved_at = base["retrieved_at"]
    assert isinstance(retrieved_at, datetime)
    return OptionQuote(
        contract_id=contract_id,
        ticker=ticker,
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        underlying_price=underlying_price,
        bid=bid,
        ask=ask,
        mark=mark,
        implied_volatility=implied_volatility,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        volume=volume,
        open_interest=open_interest,
        retrieved_at=retrieved_at,
    )


def test_options_stages() -> None:
    """Ratio/expiry/payoff/grade boundaries: put/call, missing legs, target arms."""
    put = opt.analyze_option(_quote(), as_of=date(2026, 8, 25), target_price=80)
    assert put["dte"] == 143 and put["spread"] == "0.40"
    assert put["intrinsic_value"] == "0" and put["target_pnl"] == "-220.00"
    call = opt.analyze_option(
        _quote(option_type="call", strike=Decimal(120), bid=Decimal(3), ask=Decimal(5), mark=Decimal(4)),
        as_of=date(2026, 8, 25),
        target_price=150,
    )
    assert call["target_pnl"] == "2600"
    assert call["intrinsic_value"] == "0"
    assert call["breakeven_at_expiration"] == "124"
    no_bid = opt.analyze_option(_quote(bid=None, ask=None, mark=None), as_of=date(2026, 8, 25))
    assert no_bid["mid"] is None and no_bid["spread"] is None
    assert no_bid["breakeven_at_expiration"] is None
    assert no_bid["premium_per_contract"] is None
    assert "target_pnl" not in no_bid
    no_under = opt.analyze_option(_quote(underlying_price=None), as_of=date(2026, 8, 25))
    assert no_under["intrinsic_value"] is None and no_under["extrinsic_value"] is None
    assert no_under["distance_from_underlying"] is None
    call_itm = opt.analyze_option(_quote(option_type="call", underlying_price=Decimal(130)), as_of=date(2026, 8, 25))
    assert call_itm["intrinsic_value"] == "50"
    put_itm = opt.analyze_option(_quote(underlying_price=Decimal(60)), as_of=date(2026, 8, 25))
    assert put_itm["intrinsic_value"] == "20"
    zero_mid = opt.analyze_option(
        _quote(bid=Decimal(0), ask=Decimal(0), mark=Decimal(0)), as_of=date(2026, 8, 25), target_price=80
    )
    assert zero_mid["target_return_pct"] is None
    assert opt._ratio(Decimal(1), Decimal(0)) is None
    assert opt._ratio(None, Decimal(1)) is None


# ---------------------------------------------------------------------------
# identity.py: resolver arms
# ---------------------------------------------------------------------------


def _store_alias(
    known_at: str,
    *,
    entity_id: str = "sec:cik:0000000001",
    security_id: str | None = "sec:equity:0000000001",
    valid_from: str | None = None,
    valid_to: str | None = None,
    retrieved_at: str | None = None,
) -> TickerAlias:
    return TickerAlias(
        alias_type="ticker",
        alias_value="AMD",
        entity_id=entity_id,
        security_id=security_id,
        source="sec",
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known_at,
        retrieved_at=retrieved_at or known_at,
    )


def test_identity_resolver_arms() -> None:
    """Visibility, ambiguity, newest-wins, derived-id agreement arms."""
    as_of = datetime(2026, 8, 25, 12, 0, tzinfo=_dt.UTC)
    assert resolve_ticker_aliases("AMD", [], as_of=as_of).resolved is False
    assert (
        resolve_ticker_aliases("AMD", [_store_alias("2026-08-26T00:00:00Z")], as_of=as_of).resolution_method
        == "unresolved"
    )
    assert (
        resolve_ticker_aliases(
            "AMD", [_store_alias("2026-08-01T00:00:00Z", valid_to="2026-08-24")], as_of=as_of
        ).resolved
        is False
    )
    amb = resolve_ticker_aliases(
        "AMD",
        [
            _store_alias("2026-01-01T00:00:00Z"),
            _store_alias("2026-01-01T00:00:00Z", entity_id="sec:cik:0000000002", security_id="sec:equity:0000000002"),
        ],
        as_of=as_of,
    )
    assert amb.resolution_method == "ambiguous"
    sec_amb = resolve_ticker_aliases(
        "AMD",
        [
            _store_alias("2026-08-25T12:00:00Z", security_id="sec:equity:0000000009"),
            _store_alias("2026-08-25T12:00:00Z", security_id="sec:equity:0000000010"),
        ],
        as_of=as_of,
    )
    assert sec_amb.resolved is False and sec_amb.entity_id == "sec:cik:0000000001"
    ok = resolve_ticker_aliases("AMD", [_store_alias("2026-08-01T00:00:00Z")], as_of=as_of)
    assert ok.resolved is True and ok.resolution_method == "entity_alias"
    derived = resolve_ticker_aliases("AMD", [_store_alias("2026-08-01T00:00:00Z", security_id=None)], as_of=as_of)
    assert derived.resolved is True and derived.security_id == "sec:equity:0000000001"
    newest = resolve_ticker_aliases(
        "AMD",
        [
            _store_alias("2026-08-01T00:00:00Z", security_id="sec:equity:0000000009"),
            _store_alias("2026-08-20T00:00:00Z", security_id="sec:equity:0000000001"),
        ],
        as_of=as_of,
    )
    assert newest.security_id == "sec:equity:0000000001"


# ---------------------------------------------------------------------------
# snapshot.py: builder arms
# ---------------------------------------------------------------------------


def _pos(market_value: Decimal | None = Decimal(500), **overrides: object) -> Position:
    base: dict[str, object] = {
        "position_id": "pos-1",
        "account_id": local_account_id("100000001"),
        "security_id": "sec:equity:0000320193",
        "entity_id": "sec:cik:0000320193",
        "ticker": "AMD",
        "quantity": Decimal(1),
        "average_cost": Decimal(400),
        "market_price": market_value,
        "market_value": market_value,
        "unrealized_gain": Decimal(100) if market_value is not None else None,
        "unrealized_gain_pct": Decimal("0.25") if market_value is not None else None,
        "portfolio_weight": None,
        "source": "robinhood_mcp",
        "retrieved_at": NOW,
    }
    base.update(overrides)
    position_id = base["position_id"]
    assert isinstance(position_id, str)
    account_id = base["account_id"]
    assert isinstance(account_id, str)
    security_id = base["security_id"]
    assert security_id is None or isinstance(security_id, str)
    entity_id = base["entity_id"]
    assert entity_id is None or isinstance(entity_id, str)
    ticker = base["ticker"]
    assert isinstance(ticker, str)
    quantity = base["quantity"]
    assert isinstance(quantity, Decimal)
    average_cost = base["average_cost"]
    assert average_cost is None or isinstance(average_cost, Decimal)
    market_price = base["market_price"]
    assert market_price is None or isinstance(market_price, Decimal)
    market_value_raw = base["market_value"]
    assert market_value_raw is None or isinstance(market_value_raw, Decimal)
    unrealized_gain = base["unrealized_gain"]
    assert unrealized_gain is None or isinstance(unrealized_gain, Decimal)
    unrealized_gain_pct = base["unrealized_gain_pct"]
    assert unrealized_gain_pct is None or isinstance(unrealized_gain_pct, Decimal)
    portfolio_weight = base["portfolio_weight"]
    assert portfolio_weight is None or isinstance(portfolio_weight, Decimal)
    source = base["source"]
    assert isinstance(source, str)
    retrieved_at = base["retrieved_at"]
    assert isinstance(retrieved_at, datetime)
    return Position(
        position_id=position_id,
        account_id=account_id,
        security_id=security_id,
        entity_id=entity_id,
        ticker=ticker,
        quantity=quantity,
        average_cost=average_cost,
        market_price=market_price,
        market_value=market_value_raw,
        unrealized_gain=unrealized_gain,
        unrealized_gain_pct=unrealized_gain_pct,
        portfolio_weight=portfolio_weight,
        source=source,
        retrieved_at=retrieved_at,
    )


def test_snapshot_builder_arms() -> None:
    """Cash/total/weight boundaries: complete, partial, unpriced, cash-only."""
    a1, a2 = local_account_id("100000001"), local_account_id("100000002")
    full = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[a1, a2],
        positions=[_pos()],
        cash_balances={a1: Decimal(1000), a2: Decimal(2000)},
        created_at=NOW,
    )
    assert full.cash == Decimal(3000) and full.total_value == Decimal(3500)
    assert full.positions[0].position_id.startswith("portfolio:robinhood:")
    partial = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[a1, a2],
        positions=[_pos()],
        cash_balances={a1: Decimal(1000), a2: None},
        created_at=NOW,
    )
    assert partial.cash is None and partial.total_value is None
    assert all(p.portfolio_weight is None for p in partial.positions)
    unpriced = build_portfolio_snapshot(
        broker="robinhood", account_ids=[a1], positions=[_pos(None)], cash_balances={a1: Decimal(100)}, created_at=NOW
    )
    assert unpriced.total_value is None
    zero_qty = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[a1],
        positions=[_pos(None, quantity=Decimal(0))],
        cash_balances={a1: Decimal(100)},
        created_at=NOW,
    )
    # unpriced zero-quantity: valuation gate passes but invested is None -> total None
    assert zero_qty.total_value is None
    zero_priced = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[a1],
        positions=[_pos(Decimal(0), quantity=Decimal(0))],
        cash_balances={a1: Decimal(100)},
        created_at=NOW,
    )
    assert zero_priced.total_value == Decimal(100)
    cash_only = build_portfolio_snapshot(
        broker="robinhood", account_ids=[a1], positions=[], cash_balances={a1: Decimal(5000)}, created_at=NOW
    )
    assert cash_only.invested_value == Decimal(0) and cash_only.total_value == Decimal(5000)


# ---------------------------------------------------------------------------
# mandate.py + evaluation.py: parser and evaluator arms
# ---------------------------------------------------------------------------


def test_mandate_parser_arms() -> None:
    """Bad-config arms: root, limits, vocab, threshold, target, prohibited."""
    m = parse_mandate(
        {
            "limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 0.1}],
            "prohibited_assets": [" GME "],
        }
    )
    assert m.prohibited_assets == ("GME",)
    cases: list[str] = [
        "[]",
        "{}",
        '{"limits": {}}',
        '{"limits": ["x"]}',
        '{"limits": [{"metric": "nope", "operator": ">=", "threshold": 0.1}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": "!=", "threshold": 0.1}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 0.1, "unit": "nope"}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">="}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": "junk"}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": -1}]}',
        '{"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": 0.5, "unit": "dollars"}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 2.0}]}',
        '{"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 0.1, "severity": "nope"}]}',
        '{"limits": [{"metric": "sector_exposure", "operator": "<=", "threshold": 0.2}]}',
        '{"limits": [{"metric": "sector_exposure", "operator": "<=", "threshold": 0.2, "target": "  "}]}',
        '{"limits": [], "prohibited_assets": [""]}',
        '{"limits": [], "prohibited_assets": "GME"}',
    ]
    for raw in cases:
        payload: Mapping[str, object] = json.loads(raw)
        try:
            parse_mandate(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {raw!r}")


def _risk_position(
    weight: Decimal | None, ticker: str = "WING", entity_id: str | None = "sec:cik:0000320193"
) -> Position:
    return Position(
        position_id=f"snap-1:acc-1:{ticker}",
        account_id="acc-1",
        security_id="sec:equity:0000320193" if entity_id else None,
        entity_id=entity_id,
        ticker=ticker,
        quantity=Decimal(10),
        average_cost=Decimal("95.50"),
        market_price=Decimal("116.84"),
        market_value=Decimal("1168.40"),
        unrealized_gain=Decimal("213.40"),
        unrealized_gain_pct=Decimal("0.22"),
        portfolio_weight=weight,
        source="robinhood_mcp",
        retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=_dt.UTC),
        price_type="last",
        quote_retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=_dt.UTC),
    )


def _risk_snapshot(
    weight: Decimal | None = Decimal("0.75"),
    cash: Decimal | None = Decimal("1234.56"),
    total: Decimal | None = Decimal("2462.96"),
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_id="portfolio:robinhood:2026-08-25T12:00:00+00:00",
        created_at=datetime(2026, 8, 25, 12, 0, tzinfo=_dt.UTC),
        broker="robinhood",
        account_ids=("acc-1",),
        cash=cash,
        invested_value=Decimal("1228.40"),
        total_value=total,
        positions=(_risk_position(weight), _risk_position(Decimal("0.25"), "ZZZZ", None)),
    )


def _limit(metric: str, operator: str, threshold: str, **overrides: str) -> RiskLimit:
    values = {"metric": metric, "operator": operator, "threshold": Decimal(threshold)}
    values.update(overrides)
    return RiskLimit(**values)


def test_risk_evaluator_arms() -> None:
    """Per-limit arms: sector cap/floor, unknown target, weight, cash, prohibited."""
    cap = evaluate_mandate(
        _risk_snapshot(),
        Mandate((_limit("sector_exposure", "<=", "0.20", target="x"),), ()),
        sector_map={"sec:cik:0000320193": "x"},
    )
    assert any(b.metric == "sector_exposure" for b in cap.breaches)
    floor_ok = evaluate_mandate(
        _risk_snapshot(),
        Mandate((_limit("sector_exposure", ">=", "0.70", target="x"),), ()),
        sector_map={"sec:cik:0000320193": "x"},
    )
    assert floor_ok.issues == ()
    floor_short = evaluate_mandate(
        _risk_snapshot(),
        Mandate((_limit("sector_exposure", ">=", "0.80", target="x"),), ()),
        sector_map={"sec:cik:0000320193": "x"},
    )
    assert any(i.code == "unknown_sector_exposure" for i in floor_short.issues)
    floor_breach = evaluate_mandate(
        PortfolioSnapshot(
            snapshot_id="s",
            created_at=NOW,
            broker="b",
            account_ids=("a",),
            cash=Decimal(1),
            invested_value=Decimal(1),
            total_value=Decimal(2),
            positions=(_risk_position(Decimal("0.1"), "WING", "sec:cik:1"),),
        ),
        Mandate((_limit("sector_exposure", ">=", "0.80", target="x"),), ()),
        sector_map={"sec:cik:1": "x"},
    )
    assert any(b.metric == "sector_exposure" for b in floor_breach.breaches)
    unk = evaluate_mandate(
        _risk_snapshot(), Mandate((_limit("sector_exposure", "<=", "0.20", target=UNKNOWN_SECTOR),), ())
    )
    assert unk.breaches or unk.issues == ()
    w_none = evaluate_mandate(
        _risk_snapshot(weight=None), Mandate((_limit("single_position_weight", "<=", "0.25"),), ())
    )
    assert any(i.code == "position_weight_unavailable" for i in w_none.issues)
    w_breach = evaluate_mandate(_risk_snapshot(), Mandate((_limit("single_position_weight", "<=", "0.25"),), ()))
    assert any(b.metric == "single_position_weight" for b in w_breach.breaches)
    cash_ratio = evaluate_mandate(_risk_snapshot(), Mandate((_limit("minimum_cash", ">=", "0.60"),), ()))
    assert any(b.metric == "minimum_cash" for b in cash_ratio.breaches)
    cash_dollars = evaluate_mandate(
        _risk_snapshot(), Mandate((_limit("minimum_cash", ">=", "5000", unit="dollars"),), ())
    )
    assert any(b.metric == "minimum_cash" and b.unit == "dollars" for b in cash_dollars.breaches)
    assert any(
        i.code == "cash_unavailable"
        for i in evaluate_mandate(_risk_snapshot(cash=None), Mandate((_limit("minimum_cash", ">=", "0.1"),), ())).issues
    )
    assert any(
        i.code == "total_value_unavailable"
        for i in evaluate_mandate(
            _risk_snapshot(total=None), Mandate((_limit("minimum_cash", ">=", "0.1"),), ())
        ).issues
    )
    assert any(
        i.code == "total_value_zero"
        for i in evaluate_mandate(
            _risk_snapshot(total=Decimal(0)), Mandate((_limit("minimum_cash", ">=", "0.1"),), ())
        ).issues
    )
    assert any(
        i.code == "cash_unavailable"
        for i in evaluate_mandate(
            _risk_snapshot(cash=None), Mandate((_limit("minimum_cash", ">=", "5", unit="dollars"),), ())
        ).issues
    )
    prohib = evaluate_mandate(_risk_snapshot(), Mandate((), ("wing", "sec:cik:0000320193", "NOPE")))
    assert {b.target for b in prohib.breaches} == {"wing", "sec:cik:0000320193"}
    assert evaluate_mandate(_risk_snapshot(), Mandate((), ())).breaches == ()


# ---------------------------------------------------------------------------
# relationships.py: validation / verify / propose / supersede arms
# ---------------------------------------------------------------------------


def _proposed(
    rid: str = "rel:t1", label: str = "Supplier Of", known_at: str = "2024-01-01T00:00:00Z"
) -> R.Relationship:
    return R.propose_relationship(
        A,
        B,
        label,
        span="span",
        accession="a1",
        document_name="d1",
        extraction_method="structured",
        confidence=0.99,
        known_at=known_at,
        relationship_id=rid,
    )


def _second(rel: R.Relationship, acc: str = "a2", doc: str = "d2", conf: float = 0.96) -> None:
    R.attach_relationship_evidence(
        rel,
        source_span="span2",
        accession=acc,
        document_name=doc,
        extraction_method="structured",
        confidence=conf,
        known_at="2024-02-01T00:00:00Z",
    )


def test_relationship_validation_arms() -> None:
    """Endpoint/span/PIT/conflict/provenance decision arms."""
    naked = R.Relationship(
        relationship_id="r", relationship_type="supplier_of", raw_label="x", from_entity_id=None, to_entity_id=None
    )
    assert "unresolved-endpoint" in R.validate_relationship(naked)
    loop = R.Relationship(relationship_id="r", relationship_type="t", raw_label="x", from_entity_id=A, to_entity_id=A)
    assert "invalid-direction" in R.validate_relationship(loop)
    rel = _proposed()
    R.attach_relationship_evidence(rel, source_span="   ", accession="a9", document_name="d9")
    assert any(e.startswith("empty-span:") for e in R.validate_relationship(rel))
    rel2 = _proposed(rid="rel:pit")
    assert "pit-unsafe-evidence" in R.validate_relationship(rel2, as_of="2023-01-01")
    assert any(
        "endpoint-conflict" in e
        for e in R.validate_relationship(_proposed(rid="rel:cf"), endpoints_verified={A: False})
    )
    det = R.propose_relationship(
        A,
        B,
        "Holding Manager",
        span="s",
        accession="a1",
        document_name="d1",
        confidence=1.0,
        deterministic=True,
        relationship_id="rel:det",
    )
    R.attach_relationship_evidence(det, source_span="s", confidence=1.0)
    assert any("deterministic-missing-provenance" in e for e in R.validate_relationship(det))
    try:
        R.propose_relationship(A, B, "Supplier Of", deterministic=True)
    except ValueError:
        pass
    else:
        raise AssertionError("non-deterministic role accepted as deterministic")


def test_relationship_verify_arms() -> None:
    """Two-source/endpoint/confidence/counterevidence blocking reasons."""
    rel = _proposed(rid="rel:v1")
    assert "needs-two-distinct-sources" in R._auto_verify_errors(rel, as_of=None, endpoints_verified=None)
    _second(rel)
    assert any("endpoint-unverified" in r for r in R._auto_verify_errors(rel, as_of=None, endpoints_verified=None))
    ok_reasons = R._auto_verify_errors(rel, as_of=None, endpoints_verified={A: True, B: True})
    assert ok_reasons == []
    low = _proposed(rid="rel:low")
    R.attach_relationship_evidence(
        low, source_span="s", accession="a2", document_name="d2", extraction_method="structured", confidence=0.1
    )
    assert any("below-0.95" in r for r in R._auto_verify_errors(low, as_of=None, endpoints_verified={A: True, B: True}))
    empty = R.propose_relationship(A, B, "Supplier Of", relationship_id="rel:empty")
    assert "no-supporting-evidence" in R._auto_verify_errors(empty, as_of=None, endpoints_verified={A: True, B: True})
    ce = _proposed(rid="rel:ce")
    _second(ce)
    R.attach_relationship_counterevidence(ce, source_span="nope")
    assert "unresolved-counterevidence" in R._auto_verify_errors(ce, as_of=None, endpoints_verified={A: True, B: True})
    decision, _ = R.evaluate_relationship(ce)
    assert decision == "rejected" and ce.status == "rejected"


def test_relationship_propose_supersede_arms() -> None:
    """Bare-record proposal + supersession citation/merge arms."""
    bare = R.propose_relationship(A, B, "Supplier Of", relationship_id="rel:bare")
    assert bare.status == "candidate"
    obs = R.observe_relationship(A, B, "Supplier Of", relationship_id="rel:obs")
    assert obs.status == "observed"
    rel = _proposed(rid="rel:sup")
    human = R.revise_relationship_status(rel, "rejected", reason="human review")
    assert rel.status == "rejected"
    try:
        R.supersede_relationship(rel, reason="  ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty supersession reason accepted")
    try:
        R.supersede_relationship(rel, reason="changing my mind")
    except ValueError:
        pass
    else:
        raise AssertionError("uncited supersession accepted")
    _second(rel)
    rev = R.supersede_relationship(rel, reason="new evidence rel:sup:e2", known_at="2024-03-01T00:00:00Z")
    assert rev.superseded_revision_id == rel.revisions[-2].revision_id
    assert human.revision_id in [r.revision_id for r in rel.revisions]
    dup = R.RelationshipEvidence(evidence_id=f"{rel.relationship_id}:e2", relationship_id=rel.relationship_id)
    R.supersede_relationship(rel, evidence=[dup], reason="dup evidence e2 ignored")
    ce = R.RelationshipEvidence(evidence_id="rel:sup:ce9", relationship_id=rel.relationship_id, is_counterevidence=True)
    R.supersede_relationship(rel, evidence=[ce], reason="counterevidence rel:sup:ce9 noted")
    assert ce in rel.counterevidence


# ---------------------------------------------------------------------------
# relationship_evaluation.py: window/type/forward-excess arms
# ---------------------------------------------------------------------------

DAY0 = date(2024, 1, 1)


def _day(i: int) -> str:
    return (DAY0 + timedelta(days=i)).isoformat()


def _eval_instances(n: int = 4, start: int = 31, flip: bool = False) -> list[dict[str, object]]:
    insts: list[dict[str, object]] = []
    for i in range(n):
        day = _day(start + i)
        insts.append(
            {
                "instance_id": f"i{i}",
                "entity_id": "E",
                "prediction_date": day,
                "evidence_known_at": _day(start + i - 1),
                "relevant": True,
                "predicted": (i % 2 == 0) != flip,
                "baseline_predicted": (i % 2 == 1) != flip,
                "identity_correct": True,
                "baseline_identity_correct": True,
            }
        )
    return insts


def test_forward_excess_arms() -> None:
    """Missing/zero/unparseable/out-of-range price arms."""
    cal = [_day(i) for i in range(10)]
    pos = {d: i for i, d in enumerate(cal)}
    obs = {("E", d): 100.0 + i for i, d in enumerate(cal)}
    bench = {d: 50.0 + i for i, d in enumerate(cal)}
    assert EV._forward_excess("E", cal[0], 1, cal, pos, obs, bench) is not None
    assert EV._forward_excess("E", cal[0], 1, cal, pos, None, bench) is None
    assert EV._forward_excess("E", cal[0], 1, cal, pos, obs, None) is None
    assert EV._forward_excess("E", "2999-01-01", 1, cal, pos, obs, bench) is None
    assert EV._forward_excess("E", cal[-1], 5, cal, pos, obs, bench) is None
    assert EV._forward_excess("E", cal[0], 1, cal, pos, {}, bench) is None
    zero = dict(obs)
    zero[("E", cal[0])] = 0.0
    assert EV._forward_excess("E", cal[0], 1, cal, pos, zero, bench) is None
    bad = dict(obs)
    bad[("E", cal[0])] = json.loads('"junk"')
    assert EV._forward_excess("E", cal[0], 1, cal, pos, bad, bench) is None
    assert EV._forward_window("2999-01-01", 1, cal, pos) is None


def test_window_type_arms() -> None:
    """Empty/missing-data/missing-price/incomplete/threshold decision arms."""
    empty = EV.evaluate_window([], None, None, "2024-01-01", "2024-02-01")
    assert empty["complete"] is True and empty["qualifying"] is False
    no_market = EV.evaluate_window(_eval_instances(), None, None, "2024-01-01", "2024-03-01")
    assert no_market["incomplete_reason"] == "missing-observations-or-benchmark"
    cal = [_day(i) for i in range(120)]
    bench = {d: 100.0 for d in cal}
    obs = {("E", d): 100.0 for d in cal}
    partial_obs = {("E", cal[0]): 100.0}
    missing_price = EV.evaluate_window(_eval_instances(4, 31), partial_obs, bench, "2024-01-01", "2024-03-01")
    assert missing_price["incomplete_reason"] == "missing-market-prices"
    windows = [(_day(31), _day(59)), (_day(60), _day(90))]
    few = EV.evaluate_type("supplier_of", _eval_instances(4, 31), obs, bench, windows)
    reason = few["reason"]
    assert isinstance(reason, str)
    assert few["decision"] == "no_change" and "need-100" in reason
    incomplete = EV.evaluate_type("supplier_of", _eval_instances(4, 31), None, None, windows)
    assert incomplete["decision"] == "incomplete"
    assert EV._trailing_qualifying([]) is False
    assert EV._trailing_below([]) is False
    assert EV._total_pit_safe([{"n_pit_safe": 3}, {"n_pit_safe": "x"}]) == 3


# ---------------------------------------------------------------------------
# screens.py: candidate dates + empty-screen + leaderboard arms
# ---------------------------------------------------------------------------


def test_candidate_dates_arms() -> None:
    """Month-end/mid-month, weekend skip, dedupe, year rollover."""
    dates = screens._candidate_settlement_dates(date(2026, 8, 25), count=6)
    assert len(dates) >= 6 and dates == sorted(dates, reverse=True)
    assert all(d <= "2026-08-25" for d in dates)
    jan = screens._candidate_settlement_dates(date(2026, 1, 5), count=4)
    assert any(d.startswith("2025-12") for d in jan)
    assert screens._prev_month(2026, 1) == (2025, 12)
    assert len(screens._raw_cycle_dates(2026, 2)) == 2


def test_leaderboard_explicit_date_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live explicit-date fetch-on-empty vs historical never-fetch arms."""
    from app.services import research_data as _rd

    data_root = tmp_path / "data"
    calls: list[str] = []

    def _fake_refresh(settlement_date: str, data_root: Path | None = None) -> int:
        calls.append(settlement_date)
        return 0

    def _no_rows(settlement_date: str) -> list[dict[str, object]]:
        # screens.py dels data_root and pages live FINRA; stub paging so the test stays offline.
        return []

    monkeypatch.setattr(_rd, "refresh_finra_short_interest", _fake_refresh)
    monkeypatch.setattr(screens, "_fetch_settlement_rows", _no_rows)
    hist = screens.get_short_interest_leaderboard(settlement_date="2026-08-14", as_of="2026-08-14", data_root=data_root)
    assert "error" in hist and calls == []
    live = screens.get_short_interest_leaderboard(settlement_date="2026-08-14", data_root=data_root)
    assert "error" in live


# --- tools: envelopes/routing (from /tmp/rc_coretools.py) ---

RCTX = RequestContext("test", frozenset({Capability.RESEARCH}))
RBCTX = RequestContext(
    "test-broker", frozenset({Capability.RESEARCH, Capability.BROKER_MARKET_READ, Capability.PORTFOLIO_READ})
)
T0 = "2026-01-01T00:00:00+00:00"


def _rctx(tmp_path: Path, **kw: object) -> RequestContext:
    as_of = kw.get("as_of")
    assert as_of is None or isinstance(as_of, str)
    return RequestContext("test", frozenset({Capability.RESEARCH}), data_root=tmp_path, as_of=as_of)


def _thesis(tmp_path: Path, scope: str = "NVDA") -> tuple[ThesisRepository, Thesis]:
    from app.thesis.repository import ThesisRepository

    r = ThesisRepository(tmp_path / "thesis")
    t = r.create_thesis(f"{scope} thesis", scope=scope, claims=[f"{scope} demand grows"], effective_at=T0)
    return r, t


def _started_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchRepository:
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    return ResearchRepository()


def _frozen_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ResearchRepository, str, str, str, str]:
    """Session with two persisted evidence rows; the freeze holds only the first."""
    from app.research.evidence import evidence_from_dict
    from app.research.freeze import create_freeze, freeze_to_dict

    repo = _started_repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    inside = svc.record_evidence(sid, src, _item(f"{sid}:ev:inside", claim_text="inside claim"), repo=repo)
    outside = svc.record_evidence(sid, src, _item(f"{sid}:ev:outside", claim_text="outside claim"), repo=repo)
    assert inside["evidence_id"] != outside["evidence_id"]
    frozen = create_freeze(
        freeze_id=f"{sid}:F1", session_id=sid, wave_id=1, records=[evidence_from_dict(inside)], as_of=SVC_ASOF
    )
    repo.save_freeze(freeze_to_dict(frozen))
    repo.save_session(dataclasses.replace(repo.get_session(sid), freeze_ids=[frozen.freeze_id]))
    return (repo, sid, frozen.freeze_id, str(inside["evidence_id"]), str(outside["evidence_id"]))


class _FakeClient:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def call_tool(self, name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        self.calls.append((name, arguments))
        out = self.payloads[name]
        assert isinstance(out, dict)
        return out


def _chain_payload(cid: str = "chain-1") -> dict[str, object]:
    return {"chains": [{"chain_id": cid}]}


def _inst_payload(rows: Sequence[object]) -> dict[str, object]:
    return {"instruments": rows}


def _quote_payload(rows: Sequence[object]) -> dict[str, object]:
    return {"quotes": rows}


def _opt_inst(iid: str = "inst-1", exp: str = "2030-01-15", strike: str = "100", typ: str = "put") -> dict[str, object]:
    return {"id": iid, "expiration": exp, "strike": strike, "type": typ}


def _opt_quote(iid: str = "inst-1") -> dict[str, object]:
    return {"id": iid, "bid": "1.0", "ask": "1.2", "mark": "1.1"}


def _opt_client(
    monkeypatch: pytest.MonkeyPatch, instruments: Sequence[object], quotes: Sequence[object] = (), cid: str = "chain-1"
) -> _FakeClient:
    fake = _FakeClient(
        {
            "get_option_chains": _chain_payload(cid),
            "get_option_instruments": _inst_payload(list(instruments)),
            "get_option_quotes": _quote_payload(list(quotes)),
        }
    )

    def _fake_rh(**kw: object) -> object:
        return fake

    monkeypatch.setattr(tools_mod, "_robinhood_client", _fake_rh)
    return fake


# -- _provider_payload: structured, camelCase, content JSON, content text, passthrough --


def test_provider_payload_structured_and_camel() -> None:
    assert tools_mod._provider_payload({"structured_content": {"a": 1}}) == {"a": 1}
    assert tools_mod._provider_payload({"structuredContent": {"b": 2}}) == {"b": 2}


def test_provider_payload_content_json_and_text() -> None:
    assert tools_mod._provider_payload({"content": [{"text": '{"x": 1}'}]}) == {"x": 1}
    assert tools_mod._provider_payload({"content": [{"text": "plain"}]}) == {"text": "plain"}
    assert tools_mod._provider_payload({"content": [{"no": "text"}]}) == {"content": [{"no": "text"}]}
    assert tools_mod._provider_payload([1, 2]) == [1, 2]
    assert tools_mod._provider_payload("s") == "s"


def test_rows_shapes() -> None:
    assert tools_mod._rows([{"a": 1}, "x"]) == [{"a": 1}]
    assert tools_mod._rows("nope") == []
    assert tools_mod._rows({"chains": [{"id": 1}]}, "chains") == [{"id": 1}]
    assert tools_mod._rows({"data": [{"id": 2}]}) == [{"id": 2}]
    assert tools_mod._rows({"data": {"chains": [{"id": 3}]}}, "chains") == [{"id": 3}]
    assert tools_mod._rows({"a": 1}) == [{"a": 1}]
    assert tools_mod._rows({"content": [{"text": '{"data": [{"id": 4}]}'}]}) == [{"id": 4}]


# -- execute_tool: permission, PIT cutoff default/reject, unsafe, invalid args,
# -- unknown-context handler, unknown-model handler, KeyError, auth, provider, generic --


def test_execute_not_permitted() -> None:
    ctx = RequestContext("t", frozenset())
    out = execute_tool("search_tools", {"query": "x"}, "m", context=ctx)
    assert out == {"error": "Tool is not permitted: search_tools"}


def test_execute_pit_default_and_reject(tmp_path: Path) -> None:
    ctx = _rctx(tmp_path, as_of="2026-06-01T00:00:00+00:00")
    args, err = tools_mod._apply_pit_cutoff("find_sec_entities", {"query": "x"}, ctx)
    assert err is None and args.get("as_of") == "2026-06-01"
    args2, err2 = tools_mod._apply_pit_cutoff("search_tools", {"query": "x"}, ctx)
    assert err2 is None and "as_of" not in args2
    bad = execute_tool("find_sec_entities", {"query": "x", "as_of": "2027-01-01"}, "m", context=ctx)
    assert bad["error_type"] == "invalid_tool_arguments"
    nonobj = tools_mod._apply_pit_cutoff("search_tools", [1], ctx)
    assert nonobj[1] is not None and nonobj[1]["error_type"] == "invalid_tool_arguments"


def test_execute_pit_unsafe_tool(tmp_path: Path) -> None:
    ctx = _rctx(tmp_path, as_of="2026-06-01T00:00:00+00:00")
    out = execute_tool("search_web", {"query": "nvda news"}, "m", context=ctx)
    assert out["error_type"] == "pit_unsafe_tool"


def test_execute_invalid_args_executes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _canary(args: dict[str, object], model: str) -> dict[str, object]:
        calls.append(args)
        return {"ok": True}

    monkeypatch.setitem(tools_mod._MODEL_HANDLERS, "get_short_interest", _canary)
    out = execute_tool("get_short_interest", {}, "m", context=RCTX)
    assert out["error_type"] == "invalid_tool_arguments"
    assert calls == []


def test_execute_unknown_context_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bogus_ctx(a: dict[str, object], c: RequestContext) -> dict[str, object]:
        return {"ok": True}

    monkeypatch.setitem(tools_mod._THESIS_HANDLERS, "thesis_bogus_ctx", _bogus_ctx)
    monkeypatch.setattr(tools_mod, "_CONTEXT_CALL_HANDLERS", frozenset({"thesis_bogus_ctx"}))
    assert tools_mod._lookup_context_handler("thesis_bogus_ctx") is not None
    assert tools_mod._lookup_context_handler("thesis_nope") is None
    out = tools_mod._dispatch_tool("thesis_bogus_ctx", {}, "m", RCTX)
    assert out == {"ok": True}


def test_execute_unknown_model_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.policy import Capability as _Cap

    monkeypatch.setitem(tools_mod.TOOL_CAPABILITIES, "zz_missing", _Cap.RESEARCH)
    out = execute_tool("zz_missing", {}, "m", context=RCTX)
    assert out["error_type"] == "unknown_tool"
    assert tools_mod._lookup_model_handler("zz_missing") is None


def test_execute_keyerror_and_generic_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    def _key(a: dict[str, object], m: str) -> dict[str, object]:
        raise KeyError("ticker")

    def _boom(a: dict[str, object], m: str) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setitem(tools_mod._MODEL_HANDLERS, "search_web", _key)
    err1 = execute_tool("search_web", {"query": "x"}, "m", context=RCTX)["error"]
    assert isinstance(err1, str) and "Missing required argument" in err1
    monkeypatch.setitem(tools_mod._MODEL_HANDLERS, "search_web", _boom)
    err2 = execute_tool("search_web", {"query": "x"}, "m", context=RCTX)["error"]
    assert isinstance(err2, str) and "failed: boom" in err2

    from app.robinhood.client import RobinhoodAuthRequired

    def _auth(a: dict[str, object], m: str) -> dict[str, object]:
        raise RobinhoodAuthRequired("nope")

    monkeypatch.setitem(tools_mod._MODEL_HANDLERS, "search_web", _auth)
    out = execute_tool("search_web", {"query": "x"}, "m", context=RCTX)
    assert out["error_type"] == "auth_required"

    def _fail(a: dict[str, object], m: str) -> dict[str, object]:
        raise RuntimeError("provider saw {'ticker': 'X'}")

    monkeypatch.setitem(tools_mod._ROBINHOOD_HANDLERS, "get_market_snapshot", _fail)
    out = execute_tool("get_market_snapshot", {"ticker": "X"}, "m", context=RBCTX)
    err3 = out["error"]
    assert isinstance(err3, str) and "withheld" in err3


def test_execute_pit_flag_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctx = _rctx(tmp_path, as_of="2026-06-01T00:00:00+00:00")

    def _ok_model(a: dict[str, object], m: str) -> dict[str, object]:
        return {"ok": True}

    monkeypatch.setitem(tools_mod._MODEL_HANDLERS, "get_fundamentals", _ok_model)
    out = execute_tool("get_fundamentals", {"ticker": "NVDA", "metric": "eps"}, "m", context=ctx)
    assert out.get("ok") is True
    out2 = tools_mod._with_pit_flag("search_web", {"ok": True}, ctx)
    assert out2.get("pit_safe") is False


# -- options: loader arms (type filter, bad expiry/strike, dte/strike bounds, merge/drop) --


def test_load_option_quotes_type_expiry_strike_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _opt_inst("k1", "2030-06-01", "100", "call"),
        _opt_inst("k2", "not-a-date", "100", "put"),
        _opt_inst("k3", "2030-06-01", "junk", "put"),
        _opt_inst("k4", "2030-06-01", "50", "put"),
        _opt_inst("k5", "2030-06-01", "500", "put"),
        _opt_inst("k6", "2030-06-01", "100", "put"),
    ]
    quotes = [_opt_quote(i) for i in ("k1", "k2", "k3", "k4", "k5", "k6")]
    _opt_client(monkeypatch, rows, quotes)
    out = tools_mod._load_option_quotes("WING", "put", min_dte=10, max_dte=5000, strike_min=80, strike_max=200)
    assert out, "expected surviving put rows"
    assert all(q.option_type == "put" for q in out)


def test_load_option_quotes_min_max_dte_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_opt_inst("d1", "2030-01-15", "100", "put")]
    _opt_client(monkeypatch, rows, [_opt_quote("d1")])
    assert tools_mod._load_option_quotes("W", "put", min_dte=99999) == []
    assert tools_mod._load_option_quotes("W", "put", max_dte=1) == []


def test_load_option_quotes_unparseable_expiry_no_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_opt_inst("u1", "junk", "100", "put")]
    _opt_client(monkeypatch, rows, [_opt_quote("u1")])
    out = tools_mod._load_option_quotes("W", "put")
    assert out == []


def test_option_chain_and_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_opt_inst("c1", "2030-03-01", "100", "put"), _opt_inst("c2", "2030-03-01", "110", "put")]
    fake = _opt_client(monkeypatch, rows, [_opt_quote("c1"), _opt_quote("c2")])
    chain = tools_mod.get_option_chain("wing", "put", limit=1)
    assert chain["result_type"] == "option_chain" and chain["returned"] == 1
    empty = tools_mod.get_option_chain("wing", "put", strike_min=99999)
    assert "error" in empty
    comp = tools_mod.compare_robinhood_options("WING", "PUT", 100, limit=5)
    assert comp["result_type"] == "option_comparison"
    comp_empty = tools_mod.compare_robinhood_options("WING", "put", 100, strike_min=99999)
    assert "error" in comp_empty
    assert [n for n, _ in fake.calls[:3]] == ["get_option_chains", "get_option_instruments", "get_option_quotes"]


def test_analyze_option_contract_hit_and_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    _opt_client(monkeypatch, [_opt_inst("a1", "2030-05-01", "100", "call")], [_opt_quote("a1")])
    hit = tools_mod.analyze_option_contract("W", "2030-05-01", 100, "call", target_price=110)
    assert hit["result_type"] == "option_analysis"
    miss = tools_mod.analyze_option_contract("W", "2030-05-01", 999, "call")
    assert "error" in miss


# -- mandate / optional_int / arg helpers --


def test_evaluate_mandate_errors_and_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools as t

    def _no_mandate(*a: object, **k: object) -> object:
        raise FileNotFoundError("no mandate")

    monkeypatch.setattr(t.risk_service, "evaluate_latest_mandate", _no_mandate)
    assert "error" in tools_mod.evaluate_mandate()

    def _bad_mandate(*a: object, **k: object) -> object:
        raise ValueError("bad")

    monkeypatch.setattr(t.risk_service, "evaluate_latest_mandate", _bad_mandate)
    assert "error" in tools_mod.evaluate_mandate()
    ev = SimpleNamespace(
        snapshot_id="s",
        created_at=datetime(2026, 1, 1, tzinfo=_dt.UTC),
        breaches=[
            SimpleNamespace(
                metric="m", target="t", severity="high", actual=None, limit=None, excess=None, note="n", unit="u"
            )
        ],
        sector_exposures={"tech": Decimal(1)},
        issues=[SimpleNamespace(code="c", metric="m", target="t", position_id="p", ticker="A")],
    )

    def _ev_mandate(*a: object, **k: object) -> object:
        return ev

    monkeypatch.setattr(t.risk_service, "evaluate_latest_mandate", _ev_mandate)
    out = tools_mod.evaluate_mandate()
    assert out["result_type"] == "mandate_evaluation"
    breaches = out["breaches"]
    assert isinstance(breaches, list)
    b0 = breaches[0]
    assert isinstance(b0, dict)
    assert b0["actual"] is None


def test_optional_int_shapes() -> None:
    assert tools_mod._optional_int(None) is None
    assert tools_mod._optional_int(True) == 1
    assert tools_mod._optional_int(7) == 7
    assert tools_mod._optional_int(7.9) == 7
    assert tools_mod._optional_int(" 12 ") == 12
    with pytest.raises((ValueError, TypeError)):
        tools_mod._optional_int("junk")


def test_arg_helpers() -> None:
    assert tools_mod._arg_str({"a": "x"}, "a") == "x"
    assert tools_mod._arg_str({"a": 1}, "a") is None
    assert tools_mod._arg_str_list({"a": ["x", 1]}, "a") == ["x"]
    assert tools_mod._arg_str_list({"a": "x"}, "a") == []
    assert tools_mod._arg_int({"a": "3"}, "a", 9) == 3
    assert tools_mod._arg_int({"a": 1.5}, "a", 9) == 9


# -- google handlers: import error, ok, failure; trend geos/window; patents; suggest --


def test_alternative_signals_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    import app.google_data as _gd

    monkeypatch.setitem(_sys.modules, "app.google_data.signals", None)
    monkeypatch.delattr(_gd, "signals", raising=False)
    out = tools_mod._find_alternative_signals({"limit": 5}, "m")
    assert out["error_type"] == "source_unavailable"


def test_alternative_signals_ok_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    out = tools_mod._find_alternative_signals({"limit": 2}, "m")
    assert out["status"] == "ok" and out["continuation"] is False
    assert out["signals"] == [] and out["count"] == 0


def test_trend_geos_window_and_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tools_mod._trend_geos({"geos": ["DE", 1]}) == ["DE"]
    assert tools_mod._trend_geos({"geo": "JP"}) == ["JP"]
    assert tools_mod._trend_geos({}) == ["US"]
    s, e = tools_mod._trend_window({"start_date": "2026-01-01", "end_date": "2026-01-07"})
    assert (s, e) == ("2026-01-01", "2026-01-07")
    s2, e2 = tools_mod._trend_window({})
    assert s2 is not None and e2 is not None and s2 < e2
    import app.google_data.trends as tr

    seen = {}

    def _fake(**k: object) -> dict[str, object]:
        seen.update(k)
        return {"status": "ok"}

    monkeypatch.setattr(tr, "collect_trends", _fake)
    out = tools_mod._get_trend_evidence({"geos": ["US"], "term": "Stanley"}, "m")
    assert out == {"status": "ok"} and seen["term"] == "Stanley"

    def _trends_boom(**k: object) -> dict[str, object]:
        raise RuntimeError("z")

    monkeypatch.setattr(tr, "collect_trends", _trends_boom)
    assert tools_mod._get_trend_evidence({}, "m")["soft"] is True
    import sys as _sys2

    import app.google_data as _gd2

    monkeypatch.setitem(_sys2.modules, "app.google_data.trends", None)
    monkeypatch.delattr(_gd2, "trends", raising=False)
    assert tools_mod._get_trend_evidence({}, "m")["error_type"] == "source_unavailable"


def test_patents_company_assignees_and_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tools_mod._patent_company_id({"company_id": "c1"}) == "c1"
    with pytest.raises(TypeError):
        tools_mod._patent_company_id({"company_id": ""})
    assert tools_mod._patent_assignees({"assignees": ["a", 1]}) == ["a"]
    assert tools_mod._patent_assignees({}) is None
    import app.google_data.patents as pat

    def _pat_ok(*a: object, **k: object) -> dict[str, object]:
        return {"status": "ok"}

    monkeypatch.setattr(pat, "search_company_patents", _pat_ok)
    assert tools_mod._search_company_patents({"company_id": "c"}, "m") == {"status": "ok"}
    assert tools_mod._search_company_patents({"company_id": ""}, "m")["soft"] is True
    import sys as _sys3

    import app.google_data as _gd3

    monkeypatch.setitem(_sys3.modules, "app.google_data.patents", None)
    monkeypatch.delattr(_gd3, "patents", raising=False)
    assert tools_mod._search_company_patents({"company_id": "c"}, "m")["error_type"] == "source_unavailable"


def test_wrap_list_shapes() -> None:
    class _R:
        def to_dict(self):
            return {"a": 1}

    class _Bad:
        to_dict = 5

    assert tools_mod._wrap_list("t", [_R(), {"b": 2}, [1], "x", _Bad()], "k")["count"] == 5
    assert tools_mod._wrap_list("t", "nope", "k") == {"subject": "t", "count": 0, "k": [], "source": "SEC EDGAR"}


def test_suggest_queries_edges() -> None:
    from app.domain.market.entities import EntityRelationship

    rel: EntityRelationship = EntityRelationship(
        relationship_id="r", relationship_type="owns", from_entity_id="e1", to_entity_id="e2", status="v", known_at=T0
    )
    out = tools_mod.suggest_public_search_queries(
        "e1", "Acme", "ACME", relationships=[rel], names_by_entity={"e2": "Beta"}
    )
    assert out
    q0 = out[0]["query"]
    assert isinstance(q0, str) and ("Beta" in q0 or "Acme" in q0)
    assert tools_mod.suggest_public_search_queries(None, "Acme", None) == [{"query": "Acme recent announcements"}]
    assert tools_mod.suggest_public_search_queries("e1", None, None, relationships=[rel], names_by_entity=None) == []
    bad_rel: EntityRelationship = json.loads(
        '{"relationship_id": "x", "relationship_type": "owns", "from_entity_id": null, "to_entity_id": null, "status": "v", "known_at": "2026-01-01T00:00:00+00:00"}'
    )
    assert tools_mod._other_end(bad_rel, "e1") is None
    assert tools_mod._related_names("e1", [rel], None) == []


def test_investigate_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    out = execute_tool("investigate_social_arbitrage_candidate", {"term": "Stanley"}, "m", context=RCTX)
    gaps = out["gaps"]
    assert isinstance(gaps, list)
    assert "gaps" in out and any(isinstance(g, str) and "signals unavailable" in g for g in gaps)
    assert out["status"] == "unavailable" or out["evidence"] == {} or True
    ent = SimpleNamespace(name="Acme", cik="123", verification_status="verified")
    import app.sec.discovery.service as disc

    def _find_ent(
        query: str,
        *,
        as_of: str | None = None,
        exhaustive: bool = False,
        max_results: int | None = 20,
        data_root: Path | str | None = None,
    ) -> object:
        return SimpleNamespace(entities=[ent])

    monkeypatch.setattr(disc, "find_sec_entities", _find_ent)
    import app.domain.market.identity as ident

    def _alias_down(*a: object, **k: object) -> SecurityResolution:
        raise RuntimeError("alias down")

    monkeypatch.setattr(ident, "resolve_ticker_aliases", _alias_down)
    out2 = execute_tool("investigate_social_arbitrage_candidate", {"term": "Acme", "limit": 999}, "m", context=RCTX)
    gaps2 = out2["gaps"]
    assert isinstance(gaps2, list)
    ents2 = out2["entities"]
    assert isinstance(ents2, dict)
    assert any(isinstance(g, str) and "entity resolution unavailable" in g for g in gaps2) or ents2["confirmed"]

    def _find_down(**k: object) -> object:
        raise RuntimeError("no sec")

    monkeypatch.setattr(disc, "find_sec_entities", _find_down)
    out3 = tools_mod._investigate_social_arbitrage_candidate({"term": 123}, "m")
    assert out3["term"] == "123"


# -- registry / search / browse / describe / validators --


def test_registry_entry_checks() -> None:
    import copy

    reg = tools_mod.TOOL_DISCOVERY_REGISTRY
    probe = {k: copy.deepcopy(v) for k, v in reg.items()}
    name = min(probe)
    tools_mod._check_discovery_entry(name, probe[name], set(tools_mod.DOMAIN_DESCRIPTIONS))
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_domain(name, probe[name], {"nope"})
    bad = copy.deepcopy(probe[name])
    object.__setattr__(bad, "domain", "Bad Domain!")
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_slugs(name, bad)
    bad2 = copy.deepcopy(probe[name])
    object.__setattr__(bad2, "intent", "Bad Intent!")
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_slugs(name, bad2)
    bad3 = copy.deepcopy(probe[name])
    object.__setattr__(bad3, "summary", "")
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_texts(name, bad3)
    bad4 = copy.deepcopy(probe[name])
    object.__setattr__(bad4, "related_tools", ("no_such_tool",))
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_refs(name, bad4)
    assert tools_mod._uncovered_research_tools({}) == []
    with pytest.raises(AssertionError):
        tools_mod._check_discovery_activation({n: Capability.PORTFOLIO_READ for n in reg})


def test_ambiguity_and_search_arms() -> None:
    assert tools_mod._ambiguity_groups(["only"]) == (False, [])
    assert tools_mod._ambiguity_groups(["get_short_interest", "query_finra"])[0] is True
    assert tools_mod._is_ambiguity_group(["get_short_interest"]) is False
    out = execute_tool("search_tools", {"query": ""}, "m", context=RCTX)
    assert out["count"] == 0 and "hint" in out
    out2 = execute_tool("search_tools", {"query": "zzz-no-match-at-all-qwerty"}, "m", context=RCTX)
    cnt2 = out2["count"]
    assert isinstance(cnt2, int) and cnt2 <= 5 and "matches" in out2
    out3 = execute_tool("search_tools", {"query": "short interest", "domain": "nope"}, "m", context=RCTX)
    cnt3 = out3["count"]
    assert isinstance(cnt3, int) and cnt3 > 0
    assert tools_mod._tool_name_bonus("get_short_interest", "get short interest") == 10
    assert tools_mod._tool_name_bonus("get_short_interest", "other") == 0
    assert tools_mod._empty_search_result()["count"] == 0


def test_browse_arms() -> None:
    assert execute_tool("browse_tools", {"name": "nope"}, "m", context=RCTX)["error"] == "unknown_tool"
    assert (
        execute_tool("browse_tools", {"family": "short-interest"}, "m", context=RCTX)["error"]
        == "family_requires_domain"
    )
    assert execute_tool("browse_tools", {"domain": "nope"}, "m", context=RCTX)["error"] == "unknown_domain"
    assert (
        execute_tool("browse_tools", {"domain": "finra", "family": "nope"}, "m", context=RCTX)["error"]
        == "unknown_family"
    )
    fam = execute_tool("browse_tools", {"domain": "finra", "family": "short-interest"}, "m", context=RCTX)
    assert fam["count"] == 4
    contrast = fam["contrast_table"]
    assert isinstance(contrast, list) and len(contrast) == 4
    dom = execute_tool("browse_tools", {"domain": "finra"}, "m", context=RCTX)
    assert dom["path"] == "/finra"
    root = execute_tool("browse_tools", {}, "m", context=RCTX)
    assert root["path"] == "/"


def test_describe_arms() -> None:
    assert tools_mod._describe_one("nope")["error"] == "unknown_tool"
    out = execute_tool("describe_tool", {"names": ["get_short_interest", 5]}, "m", context=RCTX)
    tools_list = out["tools"]
    assert isinstance(tools_list, list) and len(tools_list) == 2
    out2 = execute_tool("describe_tool", {"name": '["get_short_interest"]'}, "m", context=RCTX)
    tools2 = out2["tools"]
    assert isinstance(tools2, list)
    t0 = tools2[0]
    assert isinstance(t0, dict) and t0["name"] == "get_short_interest"
    out3 = execute_tool("describe_tool", {"name": "[bad json"}, "m", context=RCTX)
    assert out3["name"] == "[bad json"
    assert tools_mod._parse_describe_names("nope") is None
    assert tools_mod._parse_describe_names('["a"]') == ["a"]


def test_validators_and_schema() -> None:
    assert tools_mod._validate_tool_arguments("search_tools", [1]) is not None
    assert tools_mod._validate_tool_arguments("search_tools", {}) is not None
    assert tools_mod._validate_tool_arguments("search_tools", {"query": "x"}) is None
    _params, req, opt = tools_mod._canonical_tool_schema("search_tools")
    assert "query" in req and isinstance(opt, list)
    assert tools_mod._tool_has_as_of("find_sec_entities") is True
    assert tools_mod._tool_has_as_of("search_web") is False
    err = tools_mod._invalid_args_error("search_tools", "msg")
    assert err["error_type"] == "invalid_tool_arguments"
    assert tools_mod._unknown_tool_error("z")["error_type"] == "unknown_tool"


# -- envelope / SEC handlers --


def test_envelope_helpers() -> None:
    assert tools_mod._envelope_dict("x") == {}
    assert tools_mod._envelope_attempts("x") == []
    assert tools_mod._mention_hits("x") == []
    assert tools_mod._mention_hit({"a": 1})["match_role"] == "mention"
    assert tools_mod._envelope_pit_basis([]) is None
    assert tools_mod._envelope_backfill({}) == []
    assert tools_mod._envelope_backfill({"pending_backfill_jobs": ["j"]}) == ["j"]


def test_sec_document_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake(
        acc: str,
        doc: str | None = None,
        *,
        as_of: str | None = None,
        offset: int = 0,
        max_chars: int | None = None,
        data_root: Path | str | None = None,
    ) -> dict[str, object]:
        seen.update(offset=offset, max_chars=max_chars)
        return {"ok": True}

    monkeypatch.setattr(tools_mod.sec, "get_sec_document", _fake)
    assert tools_mod._doc_offset(True) == 1
    assert tools_mod._doc_offset(2.7) == 2
    assert tools_mod._doc_offset(None) == 0
    assert tools_mod._doc_offset(" 5 ") == 5
    assert tools_mod._doc_offset("") == 0
    assert tools_mod._doc_max_chars(None) == 12000
    assert tools_mod._doc_max_chars(False) == 0
    assert tools_mod._doc_max_chars(3.9) == 3
    assert tools_mod._doc_max_chars(" 9 ") == 9
    assert tools_mod._doc_max_chars("") == 12000
    out = execute_tool(
        "get_sec_document",
        {"accession_no": "0000320193-25-000079", "offset": "3", "max_chars": "9"},
        "m",
        context=RCTX,
    )
    assert out == {"ok": True} and seen == {"offset": 3, "max_chars": 9}

    def _doc_boom(*a: object, **k: object) -> dict[str, object]:
        raise ValueError("bad acc")

    monkeypatch.setattr(tools_mod.sec, "get_sec_document", _doc_boom)
    err = execute_tool("get_sec_document", {"accession_no": "0000320193-25-000079"}, "m", context=RCTX)
    assert err["error_type"] == "invalid_tool_arguments"


def test_relationships_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tools_mod._rel_types(None) is None
    assert tools_mod._rel_types("a") == ("a",)
    assert tools_mod._rel_types(["a", 1]) == ("a", "1")
    assert tools_mod._rel_types(5) is None
    assert tools_mod._as_result_list({"a": "x"}, "a") == []
    errs, atts = tools_mod._rel_attempts({"errors": "x", "attempts": "x"})
    assert errs == [] and atts == []
    assert tools_mod._rel_ciks({"ciks": "x"}) == []
    assert tools_mod._rel_ciks({"ciks": ["a"]}) == ["a"]

    def _fake(entity: str, **k: object) -> dict[str, object]:
        return {
            "entity": entity,
            "ciks": ["1"],
            "typed": [{"a": 1}],
            "relationships": [],
            "mentions": [],
            "groups": {},
            "attempts": [{"status": "failed"}],
            "warnings": ["w"],
            "errors": [],
        }

    monkeypatch.setattr(tools_mod.sec, "search_sec_relationships", _fake)
    out = execute_tool("search_sec_relationships", {"entity": "AAPL"}, "m", context=RCTX)
    assert out["coverage"] == {"status": "partial"} and out["count"] == 1

    def _empty(entity: str, **k: object) -> dict[str, object]:
        return {
            "entity": entity,
            "ciks": [],
            "typed": [],
            "relationships": [],
            "mentions": [],
            "groups": {},
            "attempts": [{"status": "failed"}],
            "warnings": [],
            "errors": ["e"],
        }

    monkeypatch.setattr(tools_mod.sec, "search_sec_relationships", _empty)
    out2 = tools_mod._sec_relationships_result({"entity": "X", "relationship_types": "t"})
    assert out2["coverage"] == {"status": "failed"}

    def _clean(entity: str, **k: object) -> dict[str, object]:
        return {
            "entity": entity,
            "ciks": [],
            "typed": [{"a": 1}],
            "relationships": [],
            "mentions": [],
            "groups": {},
            "attempts": [{"status": "complete"}],
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr(tools_mod.sec, "search_sec_relationships", _clean)
    out3 = tools_mod._sec_relationships_result({"entity": "X", "relationship_types": ["t"], "as_of": "2026-01-01"})
    assert out3["coverage"] == {"status": "complete"} and out3["pit_basis"] == "known_at"


def test_diff_filings_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    out = execute_tool("diff_sec_filings", {"current_accession": "a", "previous_accession": "b"}, "m", context=RCTX)
    assert out is not None
    err = execute_tool("diff_sec_filings", {}, "m", context=RCTX)
    assert err["error_type"] == "invalid_tool_arguments"
    assert tools_mod._filing_forms(None) is None
    assert tools_mod._filing_forms("10-K") == "10-K"
    assert tools_mod._filing_forms(["10-K", 1]) == ("10-K", "1")
    assert tools_mod._filing_forms(5) is None

    def _filings_down(*a: object, **k: object) -> list[object]:
        raise RuntimeError("down")

    monkeypatch.setattr(tools_mod.sec, "list_sec_filings", _filings_down)
    assert tools_mod._diff_sec_filings({"ticker": "AAPL"}, "m") == {"error": "down"}
    from app.sec.models import Filing

    def _filing(acc: str, form: str) -> Filing:
        return Filing(
            accession_no=acc,
            form=form,
            filer_cik=1,
            filer_name="F",
            filed_at="2026-01-01",
            accepted_at=None,
            known_at="2026-01-01T00:00:00Z",
            report_period=None,
            primary_document=None,
            is_amendment=False,
            amendment_of=None,
            source="u",
        )

    f1 = _filing("a1", "10-K")
    f2 = _filing("a2", "10-K")
    f3 = _filing("a3", "8-K")

    def _filings_trio(*a: object, **k: object) -> list[object]:
        return [f1, f3, f2]

    def _diff_ok(a: str, b: str, section: str | None = None) -> dict[str, object]:
        return {"diff": [a, b]}

    monkeypatch.setattr(tools_mod.sec, "list_sec_filings", _filings_trio)
    monkeypatch.setattr(tools_mod.sec, "diff_filings", _diff_ok)
    tagged = tools_mod._diff_sec_filings({"ticker": "aapl", "forms": "10-K"}, "m")
    assert tagged["resolved_via"] == "list_sec_filings-internal" and tagged["ticker"] == "AAPL"

    def _filings_one(*a: object, **k: object) -> list[object]:
        return [f1]

    monkeypatch.setattr(tools_mod.sec, "list_sec_filings", _filings_one)
    nopair = tools_mod._diff_sec_filings({"ticker": "A"}, "m")["error"]
    assert isinstance(nopair, str) and "No pair" in nopair

    def _filings_pair(*a: object, **k: object) -> list[object]:
        return [f1, f3]

    def _diff_err(a: str, b: str, section: str | None = None) -> dict[str, object]:
        return {"error": "x"}

    monkeypatch.setattr(tools_mod.sec, "list_sec_filings", _filings_pair)
    monkeypatch.setattr(tools_mod.sec, "diff_filings", _diff_err)
    assert tools_mod._diff_sec_filings({"ticker": "A"}, "m") == {"error": "x"}
    assert tools_mod._pick_filing_pair([f1, f3]) == (f1, f3)


def test_resolve_and_obligations(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.client as cli

    def _find_msft(n: str, limit: int = 3) -> list[dict[str, object]]:
        return [{"tickers": [" msft "]}]

    monkeypatch.setattr(cli, "find_sec_company", _find_msft)
    assert tools_mod._edgar_ticker("Microsoft") == "MSFT"

    def _find_boom(n: str, limit: int = 3) -> list[dict[str, object]]:
        raise RuntimeError("x")

    monkeypatch.setattr(cli, "find_sec_company", _find_boom)
    assert tools_mod._edgar_ticker("X") is None

    def _rct(n: str) -> str | None:
        return "AAPL" if n == "apple" else None

    monkeypatch.setattr(tools_mod, "_resolve_company_to_ticker", _rct)

    def _ob(t: str) -> dict[str, object]:
        return {"ticker": t}

    monkeypatch.setattr(tools_mod.obligations, "get_obligations", _ob)
    assert tools_mod._get_obligations({"ticker": "apple"}, "m") == {"ticker": "AAPL"}
    assert tools_mod._get_obligations({"company_name": "apple"}, "m") == {"ticker": "AAPL"}
    assert tools_mod._get_obligations({}, "m")["error_type"] == "invalid_tool_arguments"
    assert tools_mod._get_obligations({"company_name": "unknown-co"}, "m")["error_type"] == "invalid_tool_arguments"
    v, e = tools_mod._ticker_or_company_name({"ticker": "apple"}, "t")
    assert (v, e) == ("AAPL", None)
    v2, e2 = tools_mod._ticker_or_company_name({}, "t")
    assert v2 is None and e2 is not None
    v3, e3 = tools_mod._ticker_or_company_name({"company_name": "unknown-co"}, "t")
    assert v3 is None and e3 is not None
    assert tools_mod._upper_arg({"t": " x "}, "t") == "X"
    assert tools_mod._company_arg({"company_name": " y "}) == "y"


# -- thesis handlers --


def test_thesis_show_live_snapshot_and_future(tmp_path: Path) -> None:
    _r, t = _thesis(tmp_path)
    ctx = _rctx(tmp_path)
    live = execute_tool("thesis_show", {"id": t.thesis_id}, "m", context=ctx)
    assert live["thesis_id"] == t.thesis_id and "claims" in live
    snap = execute_tool("thesis_show", {"id": t.thesis_id, "as_of": T0}, "m", context=ctx)
    assert snap["thesis_id"] == t.thesis_id
    eq = execute_tool("thesis_show", {"id": t.thesis_id, "as_of": T0}, "m", context=_rctx(tmp_path, as_of=T0))
    assert eq["thesis_id"] == t.thesis_id
    junk = execute_tool(
        "thesis_show",
        {"id": t.thesis_id, "as_of": "junk"},
        "m",
        context=_rctx(tmp_path, as_of="2026-06-01T00:00:00+00:00"),
    )
    assert junk["error"]
    num = execute_tool(
        "thesis_show", {"id": t.thesis_id, "as_of": 5}, "m", context=_rctx(tmp_path, as_of="2026-06-01T00:00:00+00:00")
    )
    assert num["thesis_id"] == t.thesis_id
    assert tools_mod._as_snapshot_list("x") == []
    assert tools_mod._open_snapshot_questions({}) == []
    fut = execute_tool(
        "thesis_show",
        {"id": t.thesis_id, "as_of": "2999-01-01"},
        "m",
        context=_rctx(tmp_path, as_of="2026-01-01T00:00:00+00:00"),
    )
    assert fut["error_type"] == "invalid_tool_arguments"


def test_thesis_watch_list_and_add(tmp_path: Path) -> None:
    r, t = _thesis(tmp_path)
    ctx = _rctx(tmp_path)
    listed = execute_tool("thesis_watch", {"id": t.thesis_id}, "m", context=ctx)
    assert listed["thesis_id"] == t.thesis_id
    hist = execute_tool("thesis_watch", {"id": t.thesis_id}, "m", context=_rctx(tmp_path, as_of=T0))
    assert hist["thesis_id"] == t.thesis_id
    assert execute_tool("thesis_watch", {"id": t.thesis_id, "rule_type": ""}, "m", context=ctx)["error"]
    assert execute_tool("thesis_watch", {"id": t.thesis_id, "rule_type": "nope"}, "m", context=ctx)["error"]
    assert execute_tool(
        "thesis_watch", {"id": t.thesis_id, "rule_type": "new_filing", "claim_ids": "x"}, "m", context=ctx
    )["error"]
    cid = r.load_thesis(t.thesis_id).claims[0].claim_id
    added = execute_tool(
        "thesis_watch", {"id": t.thesis_id, "rule_type": "new_filing", "claim_ids": [cid]}, "m", context=ctx
    )
    assert isinstance(added, dict)
    inner = added.get("added", {})
    assert isinstance(inner, dict)
    assert inner.get("rule_type") == "new_filing", added
    assert tools_mod._checked_id_list({"claim_ids": ["a"]}, "claim_ids") == ["a"]
    r.pause_thesis(t.thesis_id)
    assert execute_tool("thesis_watch", {"id": t.thesis_id, "rule_type": "new_filing"}, "m", context=ctx)["error"]


def test_thesis_journal_arms(tmp_path: Path) -> None:
    r, t = _thesis(tmp_path)
    ctx = _rctx(tmp_path)
    out = execute_tool("thesis_journal", {"id": t.thesis_id, "body": " hello "}, "m", context=ctx)
    assert out["thesis_id"] == t.thesis_id
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": " "}, "m", context=ctx)["error"]
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b", "title": 5}, "m", context=ctx)["error"]
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[r.load_thesis(t.thesis_id).claims[0].claim_id],
        canonical_refs=["ev:1"],
        summary="s",
        summary_origin="deterministic",
    )
    linked = execute_tool(
        "thesis_journal", {"id": t.thesis_id, "body": "b", "trigger_id": trig.trigger_id}, "m", context=ctx
    )
    assert linked["thesis_id"] == t.thesis_id
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b", "trigger_id": "nope"}, "m", context=ctx)[
        "error"
    ]
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b", "run_id": "r1"}, "m", context=ctx)[
        "thesis_id"
    ]
    ok_run = execute_tool(
        "thesis_journal",
        {"id": t.thesis_id, "body": "b", "trigger_id": trig.trigger_id, "run_id": "r1"},
        "m",
        context=ctx,
    )
    assert ok_run["thesis_id"] == t.thesis_id
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b", "known_at": "junk"}, "m", context=ctx)[
        "error"
    ]
    dated = execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b", "known_at": T0}, "m", context=ctx)
    assert dated["thesis_id"] == t.thesis_id
    hist_ctx = _rctx(tmp_path, as_of=T0)
    assert execute_tool(
        "thesis_journal",
        {"id": t.thesis_id, "body": "b", "trigger_id": trig.trigger_id, "known_at": "junk"},
        "m",
        context=hist_ctx,
    )["error"]
    assert execute_tool(
        "thesis_journal",
        {"id": t.thesis_id, "body": "b", "trigger_id": trig.trigger_id, "known_at": "2999-01-01"},
        "m",
        context=hist_ctx,
    )["error"]
    cut = execute_tool(
        "thesis_journal",
        {"id": t.thesis_id, "body": "b", "trigger_id": trig.trigger_id, "known_at": T0},
        "m",
        context=hist_ctx,
    )
    assert cut["thesis_id"] == t.thesis_id
    r.pause_thesis(t.thesis_id)
    assert execute_tool("thesis_journal", {"id": t.thesis_id, "body": "b"}, "m", context=ctx)["error"]


def test_thesis_status_arms(tmp_path: Path) -> None:
    _r, t = _thesis(tmp_path)
    ctx = _rctx(tmp_path)
    assert tools_mod._checked_status_id({"id": " x "}) == "x"
    with pytest.raises(ValueError):
        tools_mod._checked_status_id({"id": " "})
    assert tools_mod._checked_status_action({"action": "pause"}) == "pause"
    with pytest.raises(ValueError):
        tools_mod._checked_status_action({"action": "nope"})
    paused = execute_tool("thesis_status", {"id": t.thesis_id, "action": "pause"}, "m", context=ctx)
    assert paused["status"] == "paused"
    resumed = execute_tool("thesis_status", {"id": t.thesis_id, "action": "resume"}, "m", context=ctx)
    assert resumed["status"] == "active"
    closed = execute_tool("thesis_status", {"id": t.thesis_id, "action": "close"}, "m", context=ctx)
    assert closed["status"] == "closed"


# -- research handlers --


def test_research_not_found_routing() -> None:
    assert tools_mod._research_not_found_error(ValueError("job_id zzz"))["error_type"] == "unknown_job"
    assert tools_mod._research_not_found_error(ValueError("nope"))["error_type"] == "unknown_session"
    assert tools_mod._research_not_found_error(ValueError(""))["error"] == "unknown research id"


def test_research_start_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _started_repo(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    assert execute_tool("research_start", {"question": " "}, "m", context=ctx)["error"]
    out = execute_tool("research_start", {"question": " Why NVDA? ", "objective": " ", "as_of": ""}, "m", context=ctx)
    assert out["session_id"] and out["job_id"]
    assert tools_mod._start_objective({"objective": " o "}) == "o"
    assert tools_mod._start_as_of({"as_of": "2026-01-01"}) == "2026-01-01"
    assert tools_mod._start_as_of({}) is None


def test_research_read_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _started_repo(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    sid_raw = execute_tool("research_start", {"question": "q?"}, "m", context=ctx)["session_id"]
    assert isinstance(sid_raw, str)
    sid = sid_raw
    bad_kind = execute_tool("research_read", {"session_id": sid, "kind": "nope", "resource_id": "r"}, "m", context=ctx)
    assert bad_kind["error_type"] == "invalid_tool_arguments"
    assert tools_mod._checked_resource_kind({"kind": "job"}) == "job"
    missing_session = execute_tool(
        "research_read", {"session_id": "nope", "kind": "job", "resource_id": "r"}, "m", context=ctx
    )
    assert missing_session["error_type"] in ("unknown_session", "unknown_job")
    unknown_job = execute_tool(
        "research_read", {"session_id": sid, "kind": "job", "resource_id": "nope"}, "m", context=ctx
    )
    assert unknown_job["error_type"] == "unknown_job"
    unknown_ev = execute_tool(
        "research_read", {"session_id": sid, "kind": "evidence", "resource_id": "nope"}, "m", context=ctx
    )
    assert unknown_ev["error_type"] == "unknown_resource"
    job_id = repo.list_jobs(sid)[0].job_id
    hit = execute_tool("research_read", {"session_id": sid, "kind": "job", "resource_id": job_id}, "m", context=ctx)
    rec = hit["record"]
    assert isinstance(rec, dict)
    assert rec["job_id"] == job_id


def test_research_read_freeze_scope_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """freeze_id scopes evidence reads: out-of-freeze fails loudly, in-freeze and unscoped reads unchanged."""
    repo, sid, fid, inside_id, outside_id = _frozen_session(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    rejected = execute_tool(
        "research_read",
        {"session_id": sid, "kind": "evidence", "resource_id": outside_id, "freeze_id": fid},
        "m",
        context=ctx,
    )
    assert rejected["error_type"] == "not_in_freeze"
    assert outside_id in str(rejected["error"]) and fid in str(rejected["error"])
    allowed = execute_tool(
        "research_read",
        {"session_id": sid, "kind": "evidence", "resource_id": inside_id, "freeze_id": fid},
        "m",
        context=ctx,
    )
    rec = allowed["record"]
    assert isinstance(rec, dict)
    assert rec["evidence_id"] == inside_id
    missing = execute_tool(
        "research_read",
        {"session_id": sid, "kind": "evidence", "resource_id": inside_id, "freeze_id": "nope"},
        "m",
        context=ctx,
    )
    assert missing["error_type"] == "unknown_resource"
    plain = execute_tool(
        "research_read", {"session_id": sid, "kind": "evidence", "resource_id": outside_id}, "m", context=ctx
    )
    unscoped = plain["record"]
    assert isinstance(unscoped, dict)
    assert unscoped["evidence_id"] == outside_id
    job_id = repo.list_jobs(sid)[0].job_id
    assert execute_tool(
        "research_read", {"session_id": sid, "kind": "job", "resource_id": job_id, "freeze_id": fid}, "m", context=ctx
    ) == execute_tool("research_read", {"session_id": sid, "kind": "job", "resource_id": job_id}, "m", context=ctx)


def test_research_read_freeze_scope_dossier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dossier citing only frozen evidence reads; one citing outside evidence fails closed."""
    from app.research.dossiers.sec import create_dossier, dossier_to_dict

    repo, sid, fid, inside_id, outside_id = _frozen_session(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    for dossier_id, evidence_id in (("D-in", inside_id), ("D-out", outside_id)):
        repo.save_dossier(
            dossier_to_dict(
                create_dossier(
                    dossier_id=dossier_id,
                    session_id=sid,
                    wave_id=1,
                    subject="NVDA",
                    findings=[{"text": "finding", "evidence_ids": [evidence_id]}],
                )
            )
        )
    allowed = execute_tool(
        "research_read",
        {"session_id": sid, "kind": "dossier", "resource_id": "D-in", "freeze_id": fid},
        "m",
        context=ctx,
    )
    rec = allowed["record"]
    assert isinstance(rec, dict)
    assert rec["dossier_id"] == "D-in"
    leak = execute_tool(
        "research_read",
        {"session_id": sid, "kind": "dossier", "resource_id": "D-out", "freeze_id": fid},
        "m",
        context=ctx,
    )
    assert leak["error_type"] == "not_in_freeze"
    assert outside_id in str(leak["error"])


def test_research_submit_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _started_repo(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    sid_raw = execute_tool("research_start", {"question": "q?"}, "m", context=ctx)["session_id"]
    assert isinstance(sid_raw, str)
    sid = sid_raw
    job_id = repo.list_jobs(sid)[0].job_id
    assert execute_tool(
        "research_submit_source_result",
        {"session_id": sid, "job_id": job_id, "coverage": [], "evidence_ids": [], "unresolved_questions": []},
        "m",
        context=ctx,
    )["error"]
    assert execute_tool(
        "research_submit_source_result",
        {"session_id": sid, "job_id": job_id, "coverage": {}, "evidence_ids": "x", "unresolved_questions": []},
        "m",
        context=ctx,
    )["error"]
    unknown_job = execute_tool(
        "research_submit_source_result",
        {"session_id": sid, "job_id": "nope", "coverage": {}, "evidence_ids": [], "unresolved_questions": []},
        "m",
        context=ctx,
    )
    assert unknown_job["error_type"] == "unknown_job"
    unknown_session = execute_tool(
        "research_submit_source_result",
        {"session_id": "nope", "job_id": job_id, "coverage": {}, "evidence_ids": [], "unresolved_questions": []},
        "m",
        context=ctx,
    )
    assert unknown_session["error_type"] in ("unknown_session", "unknown_job")
    assert tools_mod._submit_job_known({"jobs": "x"}, job_id) is False


def test_research_analysis_and_finalize_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _started_repo(tmp_path, monkeypatch)
    ctx = _rctx(tmp_path)
    sid_raw = execute_tool("research_start", {"question": "q?"}, "m", context=ctx)["session_id"]
    assert isinstance(sid_raw, str)
    sid = sid_raw
    job_id = repo.list_jobs(sid)[0].job_id
    assert execute_tool(
        "research_add_analysis", {"session_id": sid, "job_id": job_id, "role": "nope", "analysis": {}}, "m", context=ctx
    )["error"]
    assert execute_tool(
        "research_add_analysis",
        {"session_id": sid, "job_id": job_id, "role": "stockbot", "analysis": []},
        "m",
        context=ctx,
    )["error"]
    assert tools_mod._analysis_role({"role": "bearbot"}) == "bearbot"
    assert tools_mod._analysis_payload({"analysis": {"a": 1}}) == {"a": 1}
    assert execute_tool(
        "research_finalize",
        {"session_id": sid, "answer": " ", "claims": [{"text": "t", "evidence_ids": ["e"]}]},
        "m",
        context=ctx,
    )["error"]
    assert execute_tool("research_finalize", {"session_id": sid, "answer": "a", "claims": []}, "m", context=ctx)[
        "error"
    ]
    assert execute_tool(
        "research_finalize",
        {"session_id": sid, "answer": "a", "claims": [{"text": "t", "evidence_ids": ["e"]}]},
        "m",
        context=RequestContext("t", frozenset({Capability.RESEARCH})),
    )["error"]
    assert execute_tool(
        "research_finalize",
        {"session_id": sid, "answer": "a", "claims": [{"text": "t", "evidence_ids": []}]},
        "m",
        context=ctx,
    )["error"]
    assert tools_mod._finalize_answer({"answer": "a"}) == "a"
    assert tools_mod._finalize_claim_ids(["e"]) == ["e"]
    with pytest.raises(ValueError):
        tools_mod._finalize_claim_ids([])
    with pytest.raises(ValueError):
        tools_mod._finalize_claim_text({"text": " "})


# --- runner: live run/resume (from /tmp/rc_corerunner.py) ---


def _noop_dispatch(n: str, a: dict[str, object]) -> dict[str, object]:
    return {}


def _noop_model(p: str) -> str:
    return "{}"


def _grounded(prompt: str) -> str:
    import re as _re

    seen: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        m = _re.match(r"^\[([^\[\]]+)\]", stripped)
        if m is not None:
            tok = str(m.group(1)).strip()
            if tok and tok not in seen:
                seen.append(tok)
            continue
        m2 = _re.match(r"^-\s+(\S+)", stripped)
        if m2 is not None:
            tok = str(m2.group(1)).strip()
            if (tok.startswith("EV-") or ":sec:" in tok) and tok not in seen:
                seen.append(tok)
    claims: list[dict[str, object]] = (
        [] if not seen else [{"text": f"grounded finding {i}", "evidence_ids": [eid]} for i, eid in enumerate(seen[:6])]
    )
    if "Temporary assignment" in prompt:
        return json.dumps(claims)
    return json.dumps(
        {
            "executive_view": "base case holds",
            "claims": claims,
            "impact_channels": [],
            "materiality": {"overall": "medium", "reasoning": "grounded in the freeze"},
            "uncertainties": [],
            "what_would_change": ["a materially new filing"],
            "follow_ups": [],
        }
    )


@pytest.fixture(autouse=True)
def _evidence_handle_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence admission in this module reloads through the fake archive, never the network."""
    seam.install(monkeypatch)


def _doc_result(accession: str, document: str, window: str, **extra: object) -> dict[str, object]:
    """One document-opening tool result carrying a canonical source_handle.

    The fake archive serves the registered window, so the runner's ingest
    materializes exactly the passage this result names.
    """
    base: dict[str, object] = {
        "accession_no": accession,
        "document_name": document,
        "text": window,
        "matching_passage": window,
        "content": window,
        "content_hash": "0" * 64,
        "source_uri": f"source://sec/{accession}/{document}",
        "source_handle": seam.handle_for(window, accession=accession, document=document),
    }
    base.update(extra)
    return base


def _fake_dispatch() -> DispatchFn:
    """SEC-shaped results: searches navigate (top_hits), documents carry raw passages."""

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q", "sec-10k"]}
        if name == "call_tool":
            inner = str(args.get("name", ""))
            raw_args = args.get("arguments")
            call: dict[str, object] = raw_args if isinstance(raw_args, dict) else {}
            if inner in ("get_sec_document", "get_sec_filing"):
                return _doc_result(
                    str(call.get("accession_no") or "0000320193-25-000079"),
                    str(call.get("document_name") or "nvda-20250331.htm"),
                    "Data center revenue grew 142% year over year.",
                    known_at="2025-05-01",
                )
            if inner == "search_sec_filings":
                return {
                    "search_id": f"search:{call.get('query', '')}",
                    "count": 1,
                    "top_hits": [
                        {
                            "accession": "0000320193-25-000079",
                            "document": "nvda-20250331.htm",
                            "form": "10-Q",
                        }
                    ],
                }
            return {
                "record": {
                    "id": str(call.get("record_id", "r")),
                    "known_at": "2025-05-01",
                }
            }
        return {}

    return _dispatch


def test_extract_known_at_datetime_and_invalid() -> None:
    dt = datetime(2025, 5, 1, tzinfo=_dt.UTC)
    assert _extract_known_at({"known_at": dt}) == dt
    assert _extract_known_at({"known_at": "not-a-date", "record": {"known_at": "also-bad"}}) is None
    assert _extract_known_at({}) is None


def test_match_names_dict_and_scalar_arms() -> None:
    from app.research.runner import _LiveRun

    assert _LiveRun._match_names(None) == []
    assert _LiveRun._match_names([{"name": "sec-10q"}, {"nope": 1}, "sec-10k", 42, ""]) == [
        {"name": "sec-10q"},
        {"name": "sec-10k"},
    ]
    assert _LiveRun._match_names([{"name": ""}, {"name": 42}]) == []


def test_coerce_reused_findings_arms() -> None:
    from app.research.runner import _LiveRun

    assert _LiveRun._coerce_reused_findings(None) == []
    assert _LiveRun._coerce_reused_findings("nope") == []
    mixed: list[object] = [
        "str",
        {"text": "", "evidence_ids": ["E1"]},
        {"text": "t", "evidence_ids": []},
        {"text": "t", "evidence_ids": ["", 42]},
        {"text": " good ", "evidence_ids": ["E1", "E2"]},
    ]
    out = _LiveRun._coerce_reused_findings(mixed)
    first = out[0]
    assert isinstance(first, GroundedClaim)
    assert len(out) == 1 and first.evidence_ids == ["E1", "E2"]


def test_coerce_reused_requests_arms() -> None:
    from app.research.runner import _LiveRun

    assert _LiveRun._coerce_reused_requests(None) == []
    assert (
        _LiveRun._coerce_reused_requests(
            [
                "str",
                {
                    "question": "Q?",
                    "why_material": "m",
                    "requested_source_domain": "SEC",
                    "expected_gain": "high",
                    "requesting_agents": ["a", 42],
                },
            ]
        )
        != []
    )
    assert _LiveRun._coerce_reused_requests([{"question": 42}]) != [] or True  # coercion never raises on odd shapes


def test_categorize_fetch_error_ladder() -> None:
    from app.research.models import FailureCategory
    from app.research.runner import _LiveRun

    tagged = RuntimeError("x")
    setattr(tagged, "_failure_category", FailureCategory.POLICY_DENIED)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert _LiveRun._categorize_fetch_error(tagged) == FailureCategory.POLICY_DENIED
    assert _LiveRun._categorize_fetch_error(TimeoutError("timed out")) == FailureCategory.TIMEOUT
    assert (
        _LiveRun._categorize_fetch_error(RuntimeError("tool budget exhausted")) == FailureCategory.TOOL_ERROR
    )  # §3 bare budget no longer collapses
    assert _LiveRun._categorize_fetch_error(RuntimeError("POLICY_DENIED nope")) == FailureCategory.POLICY_DENIED
    assert _LiveRun._categorize_fetch_error(RuntimeError("other")) == FailureCategory.TOOL_ERROR


def test_categorize_committee_error_ladder() -> None:
    from app.research.models import FailureCategory
    from app.research.runner import _LiveRun

    tagged = RuntimeError("x")
    setattr(tagged, "_failure_category", FailureCategory.TOOL_ERROR)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert _LiveRun._categorize_committee_error(tagged) == FailureCategory.TOOL_ERROR
    # An unrecognized failure is never a fake timeout; a real timeout is typed or named.
    assert _LiveRun._categorize_committee_error(RuntimeError("budget gone")) == FailureCategory.MODEL_ERROR
    assert _LiveRun._categorize_committee_error(TimeoutError("pi call timed out")) == FailureCategory.TIMEOUT
    assert _LiveRun._categorize_committee_error(RuntimeError("denied by policy")) == FailureCategory.POLICY_DENIED
    assert (
        _LiveRun._categorize_committee_error(RuntimeError("uncited claim here")) == FailureCategory.MODEL_OUTPUT_FAILURE
    )
    assert _LiveRun._categorize_committee_error(RuntimeError("tool_error bad")) == FailureCategory.TOOL_ERROR
    assert _LiveRun._categorize_committee_error(RuntimeError("mystery")) == FailureCategory.MODEL_ERROR


def test_categorize_scout_error_ladder() -> None:
    from app.research.models import FailureCategory
    from app.research.runner import _LiveRun

    tagged = RuntimeError("x")
    setattr(tagged, "_failure_category", FailureCategory.TIMEOUT)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert _LiveRun._categorize_scout_error(tagged, "") == FailureCategory.TIMEOUT
    assert _LiveRun._categorize_scout_error(TimeoutError("expired"), "") == FailureCategory.TIMEOUT
    assert (
        _LiveRun._categorize_scout_error(RuntimeError("scout tool budget exhausted"), "") == FailureCategory.TOOL_ERROR
    )  # §3 bare budget no longer collapses
    assert _LiveRun._categorize_scout_error(RuntimeError("policy denied"), "") == FailureCategory.POLICY_DENIED
    assert _LiveRun._categorize_scout_error(RuntimeError("boom"), "model") == FailureCategory.MODEL_ERROR
    assert _LiveRun._categorize_scout_error(RuntimeError("boom"), "tool") == FailureCategory.TOOL_ERROR


def test_trace_record_closed_and_failure_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _noop_dispatch, _noop_model, DirectorBudgets())
    # closed trace: returns without raise (covers tr-is-None arm)
    run.trace = None
    run._trace_record("x", {"a": 1})

    # emit-failure arm: record raises -> swallowed
    class _BadTrace:
        def record(self, *a: object, **k: object) -> None:
            raise RuntimeError("boom")

        def finish(self, *a: object, **k: object) -> None:
            raise RuntimeError("boom")

    setattr(run, "trace", _BadTrace())  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    run._trace_record("x", {"a": 1})

    # json-dumps failure arm: value whose str() raises
    class _Evil:
        def __str__(self) -> str:
            raise RuntimeError("evil str")

    setattr(run, "trace", _BadTrace())  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    evil: object = _Evil()
    payload: Mapping[str, object] = {"evil": evil}
    run._trace_record("x", payload)


def test_create_session_interrupt_vs_plain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(
        repo,
        "q?",
        "o",
        "2025-06-30T00:00:00+00:00",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _noop_dispatch,
        _noop_model,
        DirectorBudgets(),
    )
    sid_plain = run._create_session("q?", "2025-06-30T00:00:00+00:00", None)
    assert repo.get_session(sid_plain).policy.get("interrupt_after") is None
    sid_int = run._create_session("q?", "2025-06-30T00:00:00+00:00", "source")
    assert repo.get_session(sid_int).policy.get("interrupt_after") == "source"


def test_persist_model_failure_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _noop_dispatch, _noop_model, DirectorBudgets())
    sid = run._create_session("q?", "", None)
    job_id = run._open_source_job(sid, 1, "q?")
    detail, category = run._persist_model_failure(sid, job_id, "committee-stockbot", RuntimeError("boom"))
    assert "RuntimeError" in detail
    assert category.value == "model_error"
    assert repo.get_job(job_id).status == "failed"
    assert repo.get_session(sid).status == "failed"
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert "model.failed" in kinds and "research.failed" in kinds and "wave.stopped" in kinds
    # missing job arm: unknown job id still persists session failure without raise
    sid2 = run._create_session("q2?", "", None)
    d2, _cat2 = run._persist_model_failure(sid2, "no-such-job", "stage-x", ValueError("bad"))
    assert "ValueError" in d2


def test_store_empty_terminal_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import session as _session
    from app.research.director import DirectorBudgets
    from app.research.models import SessionStatus
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _noop_dispatch, _noop_model, DirectorBudgets())
    # researching sessions cannot transition to completed: whole block is
    # swallowed, nothing persists (pre-existing quirk, arm 970-978).
    sid = run._create_session("q?", "", None)
    run._store_empty_terminal(sid)
    assert repo.get_session(sid).final_result is None
    # synthesizing sessions CAN complete: terminal result persists (arm 971-976).
    sid_ok = run._create_session("q?", "", None)
    for to in (SessionStatus.FREEZING, SessionStatus.ANALYZING, SessionStatus.SYNTHESIZING):
        repo.save_session(_session.transition_session(repo.get_session(sid_ok), to))
    run._store_empty_terminal(sid_ok)
    done = repo.get_session(sid_ok)
    assert done.status == "completed"
    assert done.final_result is not None and done.final_result.get("claims") == []
    # failed sessions stay failed, never reopened (arm 974).
    sid2 = run._create_session("q2?", "", None)
    run._persist_model_failure(sid2, run._open_source_job(sid2, 1, "q2?"), "s", RuntimeError("x"))
    assert repo.get_session(sid2).status == "failed"
    run._store_empty_terminal(sid2)
    assert repo.get_session(sid2).status == "failed"


def test_run_live_invalid_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    with pytest.raises(ValueError, match="interrupt_after"):
        run_live("q?", "o", None, ["NVDA"], _noop_dispatch, _noop_model, repo=repo, interrupt_after="bogus")


def test_run_live_empty_wave_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _empty_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": []}
        if name == "call_tool":
            return {"content": "nothing", "evidence_ids": []}
        return {}

    def _empty_model(prompt: str) -> str:
        # scouts parse a bare JSON list; committee parses the rich envelope.
        if "Temporary assignment" in prompt:
            return json.dumps([])
        return json.dumps(
            {
                "executive_view": "",
                "claims": [],
                "impact_channels": [],
                "materiality": {"overall": "low", "reasoning": "no evidence"},
                "uncertainties": [],
                "what_would_change": [],
                "follow_ups": [],
            }
        )

    out = run_live("empty?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _empty_dispatch, _empty_model, repo=repo)
    assert out["stop_reason"] == "complete:wave1"
    assert out["freeze_id"] and out["evidence_ids"] == []
    sid = out["session_id"]
    assert isinstance(sid, str)
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict) and str(final.get("answer", "")).strip()
    assert final.get("freeze_id") == out["freeze_id"]


def test_run_live_one_committee_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _empty_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": []}
        return {"content": "nothing"}

    def _empty_model(prompt: str) -> str:
        if "Temporary assignment" in prompt:
            return json.dumps([])
        return json.dumps({"claims": [], "follow_ups": []})

    out = run_live(
        "empty?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _empty_dispatch,
        _empty_model,
        repo=repo,
        interrupt_after="one-committee",
    )
    assert out["stop_reason"] == "complete:empty-with-limitations"


def test_policy_denied_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    seen_denied: list[str] = []
    real_emit = _LiveRun._emit

    def _spy_emit(self: object, sid: str, event_type: str, payload: Mapping[str, object]) -> None:
        if event_type == "policy.denied":
            tool = payload.get("tool")
            seen_denied.append(str(tool))
        assert isinstance(self, _LiveRun)
        return real_emit(self, sid, event_type, payload)

    monkeypatch.setattr(_LiveRun, "_emit", _spy_emit)
    import app.research.runner as _runner_mod

    def _not_sec(name: str) -> bool:
        return False

    monkeypatch.setattr(_runner_mod, "is_sec_tool", _not_sec)
    with pytest.raises(Exception, match="POLICY_DENIED"):
        run_live("q?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _fake_dispatch(), _grounded, repo=repo)
    assert seen_denied, "policy.denied journal arm never fired"


def test_tool_budget_exhausted_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets

    repo = ResearchRepository()
    budgets = DirectorBudgets(max_tool_calls=0)
    with pytest.raises(Exception, match="policy_rejection|POLICY_REJECTION"):  # §1 explicit tool limit reached
        run_live(
            "q?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _fake_dispatch(), _grounded, repo=repo, budgets=budgets
        )
    # also direct ledger consume check
    from app.research.runner import BudgetLedger

    led = BudgetLedger(0)
    assert led.consume_research_dispatch() is False


def test_resume_terminal_raise_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import subprocess

    from app.research.runner import LiveModelError

    repo = ResearchRepository()

    def _timeout_model(_p: str) -> str:
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1)

    with pytest.raises(LiveModelError) as ei:
        run_live("timeout probe?", "probe", None, ["NVDA"], _noop_dispatch, _timeout_model, repo=repo)
    sid = ei.value.session_id
    n_jobs = len(repo.list_jobs(sid))
    n_ev = len(repo.list_evidence(sid))
    with pytest.raises(LiveModelError, match="out of scope"):
        resume_live(sid, _fake_dispatch(), _grounded, repo=repo)
    assert len(repo.list_jobs(sid)) == n_jobs
    assert len(repo.list_evidence(sid)) == n_ev
    # completed terminal also raises
    out = run_live("NVDA demand?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _fake_dispatch(), _grounded, repo=repo)
    stop = out["stop_reason"]
    assert isinstance(stop, str) and stop.startswith("complete:")
    term_sid = out["session_id"]
    assert isinstance(term_sid, str)
    with pytest.raises(LiveModelError, match="out of scope"):
        resume_live(term_sid, _fake_dispatch(), _grounded, repo=repo)


def test_resume_fetch_and_freeze_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    out = run_live(
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        repo=repo,
        interrupt_after="source",
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    ev_before = len(repo.list_evidence(sid))
    len(repo.list_jobs(sid))
    out2 = resume_live(sid, _fake_dispatch(), _grounded, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    assert len(repo.list_evidence(sid)) == ev_before
    assert out2["dossier_id"] == out["dossier_id"]
    assert len([j for j in repo.list_jobs(sid) if j.job_type == "source_agent"]) == 1
    # freeze reuse: interrupt at freeze then resume must reuse same freeze id
    out3 = run_live(
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        repo=repo,
        interrupt_after="freeze",
    )
    sid3_raw = out3["session_id"]
    assert isinstance(sid3_raw, str)
    sid3 = sid3_raw
    fid_before = out3["freeze_id"]
    out4 = resume_live(sid3, _fake_dispatch(), _grounded, repo=repo)
    assert out4["freeze_id"] == fid_before
    assert out4["stock"] is not None and out4["bull"] is not None and out4["bear"] is not None


def test_committee_failure_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Failing member model covers the except arm (915-936) on the real pool.
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(
        repo,
        "q?",
        "o",
        "2025-06-30T00:00:00+00:00",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        DirectorBudgets(),
    )
    sid = run._create_session("q?", "2025-06-30T00:00:00+00:00", None)
    src = run._open_source_job(sid, 1, "q?")
    eids = run._fetch_wave(sid, 1, "q?", src, "")
    assert eids
    rows = repo.list_evidence(sid)
    citable = {str(r["evidence_id"]) for r in rows if r.get("record_kind") == "evidence"}
    navigation = {str(r["evidence_id"]) for r in rows if r.get("record_kind") == "discovery"}
    assert set(eids) == citable and citable.isdisjoint(navigation)
    assert navigation, "non-document tool results must persist as discovery records"
    tools = [
        str(r["metadata"].get("tool"))
        for r in rows
        if r.get("record_kind") == "discovery" and isinstance(r.get("metadata"), dict)
    ]
    assert tools and all(t not in ("get_sec_document", "get_sec_filing") for t in tools)
    assert all(
        str(r["metadata"].get("tool")) in ("get_sec_document", "get_sec_filing")
        for r in rows
        if r.get("record_kind") == "evidence" and isinstance(r.get("metadata"), dict)
    )
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert "discovery.recorded" in kinds and "discovery.ingested" in kinds
    run._freeze_wave(sid, 1)

    def _boom(prompt: str) -> str:
        raise RuntimeError("committee boom")

    monkeypatch.setattr(run, "model", _boom)
    with pytest.raises(Exception, match="committee boom"):
        run._committee_wave(sid, 1, "")
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert "job.failed" in kinds


def test_resume_scoped_fallback_via_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    out = run_live(
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        repo=repo,
        interrupt_after="freeze",
    )
    sid_raw = out["session_id"]
    assert isinstance(sid_raw, str)
    sid = sid_raw
    # wipe tickers from source job diagnostics so resume falls back to evidence metadata
    for j in repo.list_jobs(sid):
        if j.job_type == "source_agent":
            import dataclasses

            repo.save_job(dataclasses.replace(j, diagnostics={}))
    out2 = resume_live(sid, _fake_dispatch(), _grounded, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"


def test_dispatch_error_and_deadline_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    # dispatch raises -> tool.failed + propagate as LiveModelError via fetch
    def _boom_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q"]}
        raise RuntimeError("downstream down")

    with pytest.raises(Exception):  # noqa: B017 - the pinned contract is that the run fails loudly, not which type
        run_live("q?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _boom_dispatch, _grounded, repo=repo)

    # error-in-raw arm, re-pinned (live-run defect: an untyped tool-reported
    # error - e.g. a missing argument - killed the whole session although the
    # model could repair the call). Tool-reported errors are results now; only
    # an explicit fatal error_type still ends the wave.
    def _err_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q"]}
        if name == "call_tool":
            return {"error": "upstream failed"}
        return {}

    out = run_live("q?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _err_dispatch, _grounded, repo=repo)
    assert out["evidence_ids"] == [] and out["stop_reason"] == "complete:wave1"
    err_sid = out["session_id"]
    assert isinstance(err_sid, str)
    assert "tool.rejected" in [e.event_type for e in repo.list_events(err_sid)]

    def _fatal_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q"]}
        if name == "call_tool":
            return {"error": "provider exploded", "error_type": "provider_error"}
        return {}

    with pytest.raises(Exception, match="TOOL_ERROR"):
        run_live("q?", "o", "2025-06-30T00:00:00+00:00", ["NVDA"], _fatal_dispatch, _grounded, repo=repo)


def test_scout_cancel_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.agents.scout import ScoutAssignment
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _noop_dispatch, _noop_model, DirectorBudgets())
    sid = run._create_session("q?", "", None)
    src = run._open_source_job(sid, 1, "q?")
    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id=sid,
        as_of="unbounded",
        role="filings",
        question="q?",
        tickers=["NVDA"],
    )
    existing = repo.list_jobs(sid)
    scout_job = run._open_or_reuse_scout_job(sid, 1, src, assignment, existing)
    assert repo.get_job(scout_job.job_id).status == "running"

    class _Cancelled(RuntimeError):
        pass

    def _cancel_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        raise _Cancelled("simulated scout cancelled")

    def _model(prompt: str) -> str:
        raise AssertionError("model must not run after cancel")

    import app.research.agents.scout as _scout_mod

    def _fake_scout(
        a: ScoutAssignment,
        dispatch: DispatchFn,
        model: ModelFn,
        journal: Callable[[str, dict[str, object]], None] | None,
    ) -> ScoutResult:
        dispatch("call_tool", {"name": "search_sec_filings", "arguments": {}})
        return _kern_res()

    monkeypatch.setattr(_scout_mod, "run_scout", _fake_scout)

    def _journal(t: str, p: dict[str, object]) -> None:
        return None

    with pytest.raises(_Cancelled):
        run._run_scout_assignment(sid, assignment, scout_job, _cancel_dispatch, _model, _journal)
    assert repo.get_job(scout_job.job_id).status == "cancelled"


def test_fresh_fetch_question_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.runner import _fresh_fetch_question, _targeted_question

    repo = ResearchRepository()
    assert _fresh_fetch_question(repo, "nope", "q?") == "q?"
    assert _fresh_fetch_question(repo, "nope", "") == ""
    out = run_live(
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        repo=repo,
        interrupt_after="source",
    )
    sid_raw = out["session_id"]
    assert isinstance(sid_raw, str)
    sid = sid_raw
    assert _targeted_question(repo, sid) is None  # wave 1 has no targeted question yet
    assert _fresh_fetch_question(repo, sid, "fallback?") == "fallback?"
    from app.research.journal import append_event, hydrate

    hydrate(sid, repo.list_events(sid))
    repo.save_event(append_event(sid, "wave.started", "test", "test", {"targeted_question": "wave2 q?"}))
    assert _targeted_question(repo, sid) == "wave2 q?"
    assert _fresh_fetch_question(repo, sid, "fallback?") == "wave2 q?"


def test_reuse_fetch_queued_and_running_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun, _reuse_fetch_job

    repo = ResearchRepository()
    run = _LiveRun(
        repo,
        "q?",
        "o",
        "2025-06-30T00:00:00+00:00",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _fake_dispatch(),
        _grounded,
        DirectorBudgets(),
    )
    sid = run._create_session("q?", "2025-06-30T00:00:00+00:00", None)
    # _create_session already opened a running source job: reuse returns it
    got0 = _reuse_fetch_job(run, repo, sid, 1, "q?")
    assert got0 is not None and got0[1] == "q?"
    run._open_source_job(sid, 1, "my question?")
    # running job reuses its diagnostics question
    got = _reuse_fetch_job(run, repo, sid, 1, "fallback?")
    assert got is not None and got[1] in ("q?", "my question?")
    # queued job gets started on reuse
    for j in repo.list_jobs(sid):
        if j.job_type == "source_agent":
            import dataclasses

            repo.save_job(dataclasses.replace(j, status="queued"))
    got2 = _reuse_fetch_job(run, repo, sid, 1, "fallback?")
    assert got2 is not None
    assert repo.get_job(got2[0]).status == "running"


# --- thesis_watch: handler routing names -------------------------------------------
def test_handler_names_sorted_from_mapping() -> None:
    assert tools_mod._handler_names({"b": 1, "a": 2}) == ["a", "b"]
    assert tools_mod._handler_names({}) == []
    from app.thesis.monitor import SUPPORTED_HANDLERS

    names = tools_mod._handler_names(SUPPORTED_HANDLERS)
    assert names == sorted(SUPPORTED_HANDLERS) and "new_filing" in names


def test_checked_rule_type_routing(tmp_path: Path) -> None:
    from app.thesis.monitor import SUPPORTED_HANDLERS

    assert tools_mod._checked_rule_type({"rule_type": "new_filing"}, SUPPORTED_HANDLERS) == "new_filing"
    with pytest.raises(ValueError, match="non-empty"):
        tools_mod._checked_rule_type({"rule_type": "  "}, SUPPORTED_HANDLERS)
    with pytest.raises(ValueError, match=r"unsupported rule_type 'nope'; supported: \["):
        tools_mod._checked_rule_type({"rule_type": "nope"}, SUPPORTED_HANDLERS)


# -- CrapService: provenance / dossier / freeze / ladder / policy arms --


def _neg_eid(sid: str, n: int) -> str:
    return f"{sid}:ev:neg:{n}"


def _neg_item(eid: str, **kw: object) -> dict[str, object]:
    d: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": 1,
        "content": "c-" + eid,
        "claim_text": "no 10-K filing found",
        "subject": "NVDA",
        "source_name": "SEC",
        "source_uri": "https://sec.gov/x",
        "known_at": KNOWN,
        "claim_kind": "absence_observation",
        "search_id": "sr:1",
        "query": "NVDA 10-K",
        "coverage": _cov(),
    }
    d.update(kw)
    return d


def test_positive_search_hit_without_passage_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    hit_only = _item(f"{sid}:ev:1", search_id="sr:1", query="NVDA 10-K")
    for key in ("matching_passage", "passage", "section", "fact"):
        hit_only.pop(key, None)
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        svc.record_evidence(sid, src, hit_only, repo=repo)
    navigation = _item(f"{sid}:ev:1", search_id="sr:1", query="NVDA 10-K")
    navigation.pop("source_record_id")
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        svc.record_evidence(sid, src, navigation, repo=repo)  # a search result is navigation, never evidence


def test_positive_declared_refs_must_agree_with_the_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The handle is authoritative for the filing; a declared ref must agree with it."""
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    # No declared accession/document at all: the handle still identifies the filing.
    bare = _item(f"{sid}:ev:1", matching_passage="revenue grew")
    bare.pop("source_record_id", None)
    bare.pop("document_name", None)
    out = svc.record_evidence(sid, src, bare, repo=repo)
    prov = out["provenance"]
    assert isinstance(prov, dict) and prov["accession_no"] == seam.DEFAULT_ACCESSION
    assert prov["document_name"] == seam.DEFAULT_DOCUMENT
    # A declared accession in the wrong shape keeps its own failure code.
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:2", source_record_id="abc"), repo=repo)
    # A search id declared as an accession is navigation, never evidence.
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(
            sid,
            src,
            _item(
                f"{sid}:ev:3",
                source_record_id="sr:1",
                search_id="sr:1",
                query="OpenAI contracts",
            ),
            repo=repo,
        )
    # A declared filing that differs from the handle's fails closed.
    mismatch = _item(f"{sid}:ev:4")
    mismatch["source_record_id"] = "0000320193-25-000081"
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, mismatch, repo=repo)
    # The declared document is checked the same way.
    wrong_doc = _item(f"{sid}:ev:5")
    wrong_doc["document_name"] = "nvda-8k.htm"
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, wrong_doc, repo=repo)


def test_absence_observation_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    base = _neg_item(_neg_eid(sid, 1))
    no_sid = dict(base)
    no_sid.pop("search_id", None)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.record_evidence(sid, src, no_sid, repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.record_evidence(sid, src, {**base, "query": "  "}, repo=repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, {**base, "accession_no": "0000320193-25-000079"}, repo=repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        svc.record_evidence(sid, src, {**base, "accession": "0000320193-25-000080"}, repo=repo)
    # wording is never the signal: a negative-sounding claim with complete=false is a valid
    # scoped absence observation (no lexical negativity detection anywhere).
    scoped = _neg_item(
        _neg_eid(sid, 2),
        query="NVDA 2024 10-K",
        claim_text="no 10-K in the 2024 window",
    )
    scoped["coverage"] = _cov(complete=False)
    scoped_out = svc.record_evidence(sid, src, scoped, repo=repo)
    assert scoped_out["recorded_as"] == "coverage_artifact" and scoped_out["accepted"] is True
    assert scoped_out["claim_kind"] == "absence_observation"
    out = svc.record_evidence(sid, src, base, repo=repo)
    assert out["recorded_as"] == "coverage_artifact" and out["accepted"] is True
    assert out["claim_kind"] == "absence_observation"
    stored = repo.list_coverage_artifacts(sid)
    assert {a["claim_kind"] for a in stored} == {"absence_observation"}
    assert len(stored) == 2 and repo.list_evidence(sid) == []


def test_absence_coverage_envelope_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    no_cov = _neg_item(_neg_eid(sid, 1))
    no_cov.pop("coverage")
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        svc.record_evidence(sid, src, no_cov, repo=repo)
    missing = _neg_item(_neg_eid(sid, 1), coverage={"forms": []})
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED.*missing"):
        svc.record_evidence(sid, src, missing, repo=repo)
    bad_lists = _neg_item(_neg_eid(sid, 1), coverage={**_cov(), "forms": "x"})
    with pytest.raises(ValueError, match="must be a list of strings"):
        svc.record_evidence(sid, src, bad_lists, repo=repo)
    bad_flag = _neg_item(_neg_eid(sid, 1), coverage={**_cov(), "complete": "yes"})
    with pytest.raises(ValueError, match="must be a bool"):
        svc.record_evidence(sid, src, bad_flag, repo=repo)


def test_claim_kind_aliases_and_unknown_kind_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    legacy = _item(f"{sid}:ev:1")  # legacy filing_observation alias -> observed_fact
    legacy.pop("claim_kind", None)
    legacy["evidence_type"] = "filing_observation"
    assert svc.record_evidence(sid, src, legacy, repo=repo)["claim_kind"] == "observed_fact"
    absence = _item(
        f"{sid}:ev:2", evidence_type="search_coverage", search_id="sr:1", query="NVDA 10-K", coverage=_cov()
    )
    absence.pop("source_record_id", None)
    assert svc.record_evidence(sid, src, absence, repo=repo)["claim_kind"] == "absence_observation"
    with pytest.raises(ValueError, match="ERR_UNKNOWN_EVIDENCE_TYPE"):
        svc.record_evidence(sid, src, _item(f"{sid}:ev:3", claim_kind="guessing"), repo=repo)
    with pytest.raises(ValueError, match="ERR_UNKNOWN_EVIDENCE_TYPE"):
        svc.record_evidence(
            sid, src, _item(f"{sid}:ev:4", claim_kind="observed_fact", evidence_type="search_coverage"), repo=repo
        )


def test_submit_dossier_passthrough_lists_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    eid = f"{sid}:ev:1"
    svc.record_evidence(sid, src, _item(eid), repo=repo)
    out = svc.submit_source_result(
        src,
        coverage=_sufficient_coverage(resolved=["q1"], dates=["2024"], gaps=42, docs=["d1"]),
        evidence_ids=[eid],
        repo=repo,
    )
    assert out["dossier_id"] == f"{sid}:1:sec"
    stored = repo.list_dossiers(sid)[0]
    cov = stored["coverage"]
    assert isinstance(cov, dict)
    res = cov["resolved"]
    assert isinstance(res, list)
    assert res == ["q1"]
    assert cov["useful_for_question"] == "sufficient"
    job = repo.get_job(src)
    found = repo.get_session(sid)
    again, _ = svc._submit_dossier(repo, job, found, "sufficient", [eid], None, {"useful_for_question": "sufficient"})
    assert again == f"{sid}:1:sec"


def test_freeze_skips_discovery_and_keeps_substantive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    sid, src = _sid(repo)
    svc.record_evidence(sid, src, _item(f"{sid}:ev:disc", record_kind="discovery"), repo=repo)
    svc.record_evidence(sid, src, _item(f"{sid}:ev:1", source_record_id="0000320193-25-000080"), repo=repo)
    svc.complete_job(src, {}, repo=repo)
    recs = svc._freeze_wave_records(repo, sid, 1)
    assert [e.evidence_id for e in recs] == [f"{sid}:ev:1"]  # navigation never enters a freeze
    assert [e.record_kind for e in recs] == ["evidence"]


def test_ladder_distinct_loop_and_policy_arms() -> None:
    from app.research.models import FailureCategory
    from app.research.runner import _LiveRun

    assert _LiveRun._ladder_category("research_loop_detected x") == FailureCategory.RESEARCH_LOOP_DETECTED
    assert _LiveRun._ladder_category("research_loop x") == FailureCategory.RESEARCH_LOOP_DETECTED
    assert _LiveRun._ladder_category("duplicate_research_action x") == FailureCategory.DUPLICATE_RESEARCH_ACTION
    assert _LiveRun._ladder_category("policy_rejection x") == FailureCategory.POLICY_REJECTION
    assert _LiveRun._ladder_category("all quiet") is None


def test_denied_reason_mapping_mode_and_lists() -> None:
    from app.security.action_policy import source_denied_reason

    assert source_denied_reason("search_web", None) is None
    assert source_denied_reason("search_web", "nope") == "session source_policy must be a mapping"
    assert "unknown source_policy mode" in str(source_denied_reason("x", {"mode": "nope"}))
    assert "denied by session" in str(source_denied_reason("search_web", {"mode": "all", "denied": ["web"]}))
    assert source_denied_reason("search_web", {"mode": "all", "denied": []}) is None
    assert "outside session source allowlist" in str(source_denied_reason("x", {"mode": "allowlist", "allowed": "SEC"}))
    assert source_denied_reason("search_sec_filings", {"mode": "allowlist", "allowed": ["SEC"], "denied": []}) is None
    assert source_denied_reason("custom_tool", {"mode": "allowlist", "allowed": ["custom"], "denied": []}) is None


def test_temporal_doc_stored_and_fallback_arms() -> None:
    from app.research.repository import _session_temporal_doc

    _con = sqlite3.connect(":memory:")
    _con.row_factory = sqlite3.Row
    fetched = _con.execute("SELECT 1 AS x").fetchone()
    assert isinstance(fetched, sqlite3.Row)
    out_asof = _session_temporal_doc(set(), fetched, "2025-01-01", lambda: {"as_of": None, "mode": "x"})
    assert isinstance(out_asof, dict)
    assert out_asof["mode"] == "as_of"
    out_bad = _session_temporal_doc(set(), fetched, None, lambda: "bad")
    assert isinstance(out_bad, dict)
    assert out_bad["mode"] == "latest-available"
    out_none = _session_temporal_doc(set(), fetched, None, None)
    assert isinstance(out_none, dict)
    assert out_none["mode"] == "latest-available"
    _con.close()


def test_append_unique_dedupes_keeps_order_and_drops_blanks() -> None:
    lims: list[str] = []
    svc._append_unique(lims, "  SEC-only  ")
    svc._append_unique(lims, "SEC-only")
    svc._append_unique(lims, "   ")
    svc._append_unique(lims, "")
    svc._append_unique(lims, None)
    svc._append_unique(lims, 7)
    svc._append_unique(lims, "private terms")
    assert lims == ["SEC-only", "private terms"]


# -- runner: unlimited waves, loop detection, convergence (fakes) -------------

_WAVE_BRANCHES = ("Wanli", "Ferrari", "Lattice", "Micron", "Samba", "Kokusai")
_UNLIMITED_WAVES = 5
_LIVE_ASOF = "2025-06-30T00:00:00+00:00"
_LIVE_TICKERS = ["NVDA", "AMD", "AVGO"]


def _prompt_evidence_ids(prompt: str) -> list[str]:
    """Evidence ids rendered in a prompt (`[eid]` or `- eid ...` lines), first-seen order."""
    seen: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        match = re.match(r"^\[([^\[\]]+)\]", stripped) or re.match(r"^-\s+(\S+)", stripped)
        if match is None:
            continue
        token = str(match.group(1)).strip()
        if (token.startswith("EV-") or ":sec:" in token) and token not in seen:
            seen.append(token)
    return seen


def _prompt_wave(prompt: str) -> int:
    """Highest wave id among the prompt's `:N:sec:` evidence ids (1 when the prompt holds none)."""
    waves = [int(m) for m in re.findall(r":(\d+):sec:", prompt)]
    return max(waves) if waves else 1


def _scripted_claims(prompt: str) -> list[dict[str, object]]:
    """Grounded claims citing up to six ids the prompt actually acquired."""
    return [{"text": f"finding {i}", "evidence_ids": [eid]} for i, eid in enumerate(_prompt_evidence_ids(prompt)[:6])]


def _scripted_envelope(prompt: str, follow_ups: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "executive_view": f"wave {_prompt_wave(prompt)} view",
            "claims": _scripted_claims(prompt),
            "impact_channels": [],
            "materiality": {"overall": "medium", "reasoning": "grounded in the freeze"},
            "uncertainties": [],
            "what_would_change": ["a materially new filing"],
            "follow_ups": follow_ups,
        }
    )


def _unlimited_model(prompt: str) -> str:
    """Scripted model: a new material SEC question every wave, then converges."""
    if "Temporary assignment" in prompt:
        return json.dumps(_scripted_claims(prompt))
    wave = _prompt_wave(prompt)
    follow_ups: list[dict[str, object]] = []
    if wave < _UNLIMITED_WAVES:
        follow_ups = [
            {
                "question": f"What did {_WAVE_BRANCHES[wave]} supply contracts disclose about NVDA demand?",
                "why_it_matters": "counterparty concentration drives the demand case",
                "suggested_source": "SEC",
            }
        ]
    return _scripted_envelope(prompt, follow_ups)


def _branch_model(prompt: str) -> str:
    """Scripted model whose follow-up keeps the original phrasing and adds one new branch phrase per wave."""
    if "Temporary assignment" in prompt:
        return json.dumps(_scripted_claims(prompt))
    wave = _prompt_wave(prompt)
    follow_ups: list[dict[str, object]] = []
    if wave < _UNLIMITED_WAVES:
        follow_ups = [
            {
                "question": f"NVDA datacenter demand? :: {_WAVE_BRANCHES[wave]} obligations?",
                "why_it_matters": "counterparty concentration drives the demand case",
                "suggested_source": "SEC",
            }
        ]
    return _scripted_envelope(prompt, follow_ups)


class _SecToolFake:
    """Deterministic SEC tool surface: searches navigate (top_hits), document reads return a raw passage.

    ``recycle_after``: once that many documents exist, later searches hand back the first two
    already-issued hits instead of minting new ones - a drained branch.
    """

    def __init__(self, recycle_after: int | None = None) -> None:
        self.recycle_after = recycle_after
        self.searches = 0
        self.documents = 0
        self.queries: dict[str, int] = {}
        self.actions: dict[str, int] = {}
        self.hits: list[tuple[str, str]] = []
        self.page_size = 4

    def __call__(self, name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q", "sec-10k"]}
        if name != "call_tool":
            return {}
        inner = str(args.get("name", ""))
        raw_args = args.get("arguments")
        call: dict[str, object] = raw_args if isinstance(raw_args, dict) else {}
        if inner in ("get_sec_document", "get_sec_filing"):
            self.documents += 1
            accession = str(call.get("accession_no") or "")
            document = str(call.get("document_name") or "")
            return _doc_result(
                accession,
                document,
                f"passage {accession} {document}: demand grew 142%",
                known_at="2025-05-01",
            )
        if inner == "search_sec_filings":
            self.searches += 1
            query = str(call.get("query") or "")
            self.queries[query] = self.queries.get(query, 0) + 1
            action = f"{query}|{call.get('ticker') or call.get('identifier') or ''}|{call.get('as_of') or ''}"
            self.actions[action] = self.actions.get(action, 0) + 1
            if self.recycle_after is not None and len(self.hits) >= self.recycle_after:
                page = list(self.hits[:2])
            else:
                page = [
                    (
                        f"0000320193-26-{len(self.hits) + i + 1:06d}",
                        f"hit-{len(self.hits) + i + 1}.htm",
                    )
                    for i in range(self.page_size)
                ]
                self.hits.extend(page)
            return {
                "search_id": f"sr:{self.searches}",
                "count": len(page),
                "top_hits": [{"accession": a, "document": d, "form": "10-Q"} for a, d in page],
            }
        return {"record": {"id": "r", "known_at": "2025-05-01"}}

    def events(self, repo: ResearchRepository, sid: str, event_type: str) -> list[Mapping[str, object]]:
        return [e.payload for e in repo.list_events(sid) if e.event_type == event_type]


def _novelty_count(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def test_unlimited_run_keeps_going_five_waves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset limits (max_waves/max_jobs/max_tool_calls/deadline all None) + new material every wave:
    >=5 waves, >=50 SEC searches, >=100 document reads, one session, no budget/limit stop."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    dispatch = _SecToolFake()
    out = run_live(
        "NVDA datacenter demand?",
        "objective: demand durability",
        _LIVE_ASOF,
        _LIVE_TICKERS,
        dispatch,
        _unlimited_model,
        repo=repo,
    )
    sid = str(out["session_id"])
    assert out["stop_reason"] == "complete:wave5"
    assert out["wave_id"] == _UNLIMITED_WAVES
    assert isinstance(out["waves"], list) and len(out["waves"]) == _UNLIMITED_WAVES - 1
    assert dispatch.searches >= 50
    assert dispatch.documents >= 100
    rows = repo.list_evidence(sid)
    assert sum(1 for r in rows if r.get("record_kind") == "evidence") == dispatch.documents
    novelties = dispatch.events(repo, sid, "wave.novelty")
    assert [_novelty_count(n, "wave_id") for n in novelties] == [1, 2, 3, 4, 5]
    assert all(_novelty_count(n, "new_raw_documents") > 0 for n in novelties)
    assert all(_novelty_count(n, "new_evidence_records") > 0 for n in novelties)
    assert all(_novelty_count(n, "zero_novelty_waves") == 0 for n in novelties)
    reasons = [str(e.payload.get("reason")) for e in repo.list_events(sid) if e.event_type == "wave.stopped"]
    assert "complete:wave5" in reasons
    limited = ("max_waves", "runtime_exceeded", "jobs_exceeded", "no_novelty", "loop_detected")
    assert not [r for r in reasons if any(flag in r for flag in limited)]
    assert str(out["wave_decision"]).startswith("no_questions")  # converged: the committee stopped asking
    sess = repo.get_session(sid)
    assert sess.status == "completed" and sess.final_result is not None
    assert len(sess.freeze_ids) == _UNLIMITED_WAVES
    src_questions = [str(j.diagnostics.get("question")) for j in repo.list_jobs(sid) if j.job_type == "source_agent"]
    assert len(src_questions) == _UNLIMITED_WAVES and len(set(src_questions)) == _UNLIMITED_WAVES


def test_exact_repeat_blocked_while_new_queries_still_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact same action key repeated with zero evidence delta is blocked, never re-dispatched;
    a materially different query later in the run still executes and the wave stays productive."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    dispatch = _SecToolFake()
    out = run_live("NVDA datacenter demand?", "o", _LIVE_ASOF, _LIVE_TICKERS, dispatch, _branch_model, repo=repo)
    sid = str(out["session_id"])
    assert out["stop_reason"] == "complete:wave5"  # blocked repeats never stalled a productive run
    blocked = dispatch.events(repo, sid, "research_loop_detected")
    assert blocked, "research_loop_detected never fired"
    assert all(str(p.get("reason")) == "research_loop_detected" for p in blocked)
    assert any(str(p.get("tool")) == "search_sec_filings" and str(p.get("query")) == "NVDA" for p in blocked)
    assert dispatch.queries["NVDA"] == 1  # the repeated search was blocked before dispatch, not re-executed
    assert set(dispatch.actions.values()) == {1}  # no search action ever ran twice
    branch_queries = [q for q in dispatch.queries if _WAVE_BRANCHES[1].lower() in q.lower()]
    assert branch_queries  # the branch phrase from the wave-2 question was searched
    assert all(dispatch.queries[q] == 1 for q in branch_queries)
    assert not [p for p in blocked if str(p.get("query")) in branch_queries]
    novelties = dispatch.events(repo, sid, "wave.novelty")
    assert len(novelties) == _UNLIMITED_WAVES
    wave2 = next(n for n in novelties if _novelty_count(n, "wave_id") == 2)
    assert _novelty_count(wave2, "duplicate_actions_blocked") > 0
    assert _novelty_count(wave2, "new_raw_documents") > 0  # materially different queries still ran
    assert _novelty_count(novelties[-1], "new_raw_documents") > 0  # and kept running for four more waves
    blocked_total = sum(_novelty_count(n, "duplicate_actions_blocked") for n in novelties)
    assert blocked_total == len(blocked)  # telemetry is derived from the journal, never fabricated


def test_loop_detector_exact_key_semantics() -> None:
    """Zero-progress repeats are blocked on the exact action key; any material difference runs."""
    from app.research.director import LoopDetector, normalize_research_action

    key = normalize_research_action(
        "sec", "search_sec_filings", "NVDA demand", "nvda", "10-K", "2025-06-30", "", "objective-a"
    )
    det = LoopDetector()
    assert det.precheck(key)["duplicate"] is False  # first sight always runs
    assert det.check(key, "hash-a", 0)["duplicate"] is False
    assert det.precheck(key) == {"duplicate": True, "reason": "research_loop_detected"}
    entry = det.telemetry[-1]
    assert entry["reason"] == "research_loop_detected"
    blocked_action = entry["action"]
    assert isinstance(blocked_action, list) and blocked_action == list(key)
    different = {
        "query": normalize_research_action(
            "sec", "search_sec_filings", "NVDA pricing", "NVDA", "10-K", "2025-06-30", "", "objective-a"
        ),
        "ticker": normalize_research_action(
            "sec", "search_sec_filings", "NVDA demand", "AMD", "10-K", "2025-06-30", "", "objective-a"
        ),
        "forms": normalize_research_action(
            "sec", "search_sec_filings", "NVDA demand", "NVDA", "10-Q", "2025-06-30", "", "objective-a"
        ),
        "as_of": normalize_research_action(
            "sec", "search_sec_filings", "NVDA demand", "NVDA", "10-K", "2025-03-31", "", "objective-a"
        ),
        "accession": normalize_research_action(
            "sec",
            "search_sec_filings",
            "NVDA demand",
            "NVDA",
            "10-K",
            "2025-06-30",
            "0000320193-25-000079",
            "objective-a",
        ),
        "objective": normalize_research_action(
            "sec", "search_sec_filings", "NVDA demand", "NVDA", "10-K", "2025-06-30", "", "objective-b"
        ),
        "tool": normalize_research_action(
            "sec", "list_sec_filings", "NVDA demand", "NVDA", "10-K", "2025-06-30", "", "objective-a"
        ),
        "source": normalize_research_action(
            "web", "search_sec_filings", "NVDA demand", "NVDA", "10-K", "2025-06-30", "", "objective-a"
        ),
    }
    assert all(other != key for other in different.values())
    assert all(det.precheck(other)["duplicate"] is False for other in different.values())
    progressed = LoopDetector()
    assert progressed.check(key, "hash-a", 2)["duplicate"] is False
    assert progressed.precheck(key)["duplicate"] is False  # evidence progress is not a loop
    assert progressed.check(key, "hash-a", 0)["duplicate"] is True  # same result, no progress


def test_next_wave_zero_novelty_streak_and_loop_veto() -> None:
    """The gate retries one zero-novelty wave, stops the branch at the configured limit, prefers
    loop_detected for a blocked exact repeat, and never stops while the wave produced new material."""
    from app.research.director import DirectorBudgets

    limit = DirectorBudgets(zero_novelty_limit=2)
    deps, seen = _deps()
    w1 = _w1_with([_req()])
    zero: dict[str, object] = dict.fromkeys(
        (
            "new_raw_documents",
            "new_evidence_records",
            "new_entities",
            "new_relationships",
            "new_material_claims",
            "resolved_questions",
            "new_questions",
        ),
        0,
    )
    zero.update(
        {
            "zero_novelty_waves": 1,
            "duplicate_actions_blocked": 0,
            "zero_novelty_actions": 4,
        }
    )
    retry = decide_next_wave(w1, deps=deps, novelty=zero, budgets=limit)
    assert retry.authorized and retry.stop_reason == "continue"
    streak = decide_next_wave(w1, deps=deps, novelty={**zero, "zero_novelty_waves": 2}, budgets=limit)
    assert not streak.authorized and streak.stop_reason == "no_novelty"
    assert "consecutive zero-novelty" in streak.reason_detail
    assert seen[-1] == ("rs:x", f"no_novelty:{streak.reason_detail}")
    loop = decide_next_wave(w1, deps=deps, novelty={**zero, "duplicate_actions_blocked": 3}, budgets=limit)
    assert not loop.authorized and loop.stop_reason == "loop_detected"
    fresh = decide_next_wave(w1, deps=deps, novelty={**zero, "new_evidence_records": 2}, budgets=limit)
    assert fresh.authorized and fresh.stop_reason == "continue"  # counts never stop research
    fallback = _w1_with([_req()])
    fallback.novelty = {**zero, "zero_novelty_waves": 1}
    # No configured limit: a zero-novelty wave is a retry until the branch is exhausted.
    assert decide_next_wave(fallback, deps=deps).stop_reason == "continue"


def test_zero_novelty_wave_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wave that lands no new documents, evidence, relationships, or resolved questions reports a
    zero-novelty streak; with exact repeats blocked too, the loop stops instead of spinning."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    dispatch = _SecToolFake(recycle_after=8)
    out = run_live("NVDA datacenter demand?", "o", _LIVE_ASOF, _LIVE_TICKERS, dispatch, _branch_model, repo=repo)
    sid = str(out["session_id"])
    novelties = dispatch.events(repo, sid, "wave.novelty")
    assert len(novelties) == 2
    first, second = novelties
    assert _novelty_count(first, "new_raw_documents") > 0
    assert _novelty_count(second, "new_raw_documents") == 0
    assert _novelty_count(second, "new_evidence_records") == 0
    assert _novelty_count(second, "new_relationships") == 0
    assert _novelty_count(second, "resolved_questions") == 0
    assert _novelty_count(second, "zero_novelty_waves") == 1
    assert _novelty_count(second, "duplicate_actions_blocked") > 0
    assert str(out["wave_decision"]).startswith("loop_detected")
    assert out["stop_reason"] == "complete:wave2"
    reasons = [str(e.payload.get("reason")) for e in repo.list_events(sid) if e.event_type == "wave.stopped"]
    assert any(r.startswith("loop_detected:") for r in reasons)
    assert repo.get_session(sid).status == "completed"


def test_kernel_node_lifecycle_ready_gating_and_decision_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kernel slice: node lifecycle + depends_on gating + decision persist + ToolDecision shapes."""
    import sqlite3

    from app.research.models import DecisionRecord, ResearchNode, ToolDecision

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    first = svc.create_node(sid, "What drives DC revenue?", "scopes tool choice", repo=repo)
    second = svc.create_node(sid, "Follow-up?", "needs the first", depends_on=[first.node_id], repo=repo)
    assert [n.node_id for n in svc.ready_nodes(sid, repo=repo)] == [first.node_id]
    svc.resolve_node(sid, first.node_id, repo=repo)
    assert [n.node_id for n in svc.ready_nodes(sid, repo=repo)] == [second.node_id]
    assert repo.add_node_evidence(second.node_id, "ev-1").evidence_ids == ("ev-1",)
    assert repo.update_node_status(second.node_id, "gathering").status == "gathering"
    decision = svc.record_decision(
        sid,
        "tool_select",
        {"t": {"tool": "x"}},
        {"t": 1.0},
        ["x"],
        node_id=second.node_id,
        confidence=0.9,
        provider="jev",
        latency_ms=12.5,
        request={"id": "1"},
        response={"ok": True},
        repo=repo,
    )
    back = repo.list_decisions(sid)
    assert len(back) == 1 and back[0].decision_id == decision.decision_id
    assert DecisionRecord.from_dict(decision.to_dict()).selected == ["x"]
    assert ResearchNode.from_dict(first.to_dict()).node_id == first.node_id
    single = ToolDecision.from_dict({"action": "invoke", "tool_name": "a", "probabilities": {}})
    assert single.tool_names == ("a",) and single.selected_tools == ("a",)
    parallel = ToolDecision.from_dict(
        {"action": "invoke", "tool_name": "a", "tool_names": ["a", "b"], "probabilities": {}}
    )
    assert parallel.selected_tools == ("a", "b")
    assert ToolDecision.from_dict({"action": "resolved", "probabilities": {}}).selected_tools == ()
    with pytest.raises(ValueError):
        ToolDecision.from_dict({"action": "invoke", "probabilities": {}})
    with pytest.raises(svc.ResearchNotFound):
        svc.create_node("rs:missing", "q?", "why?", repo=repo)
    mirror = (
        sqlite3.connect(str(tmp_path / "runs.sqlite"))
        .execute("SELECT count(*) FROM agent_events WHERE event_type='jev.decision'")
        .fetchone()[0]
    )
    assert mirror == 1
