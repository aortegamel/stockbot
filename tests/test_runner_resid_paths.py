"""RunnerResid paths: branch tests for the over-10 fns in app/research/runner.py.

_merge_disagreement, _LiveRun._reused_scout_result, _find_completed_scout,
_LiveRun._start_reused_scout, _LiveRun._run_next_wave, _LiveRun._complete_wave_tail,
_run_one_committee, then the wave-items/record-identity and resume phases:
_dossier_items + _LiveRun._wave_items, _LiveRun._build_evidence_record (evidence vs
discovery classification), _freeze_or_reuse, resume_live. Live-run defect contract:
rejected tool calls stay agent-visible (never ingested, retry blocked by the loop
detector), scout assignment failures degrade instead of ending the session, plus
_close_wave1_result/_substantive_ids/_merge_session_dossiers.
Plain pytest, tmp_path DBs, fakes for models/dispatch.
"""

from __future__ import annotations

import json
import re as _re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tests.research_source_seam as seam
from app.research.repository import ResearchRepository

if TYPE_CHECKING:
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.director import Wave1Result
    from app.research.evidence import Evidence
    from app.research.models import JSONValue
    from app.research.runner import _LiveRun


def _grounded(prompt: str) -> str:
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
        []
        if not seen
        else [
            {"text": f"grounded finding {chr(97 + i)} suggests exposure", "evidence_ids": [eid]}
            for i, eid in enumerate(seen[:6])
        ]
    )
    if "Temporary assignment" in prompt:
        return json.dumps(claims)
    channels: list[dict[str, object]] = (
        [{"text": "channel finding", "direction": "up", "evidence_ids": [seen[0]]}] if seen else []
    )
    return json.dumps(
        {
            "executive_view": "base case holds",
            "claims": claims,
            "impact_channels": channels,
            "materiality": {"overall": "medium", "reasoning": "grounded in the freeze"},
            "uncertainties": ["scope limits"],
            "what_would_change": ["a materially new filing"],
            "follow_ups": [],
        }
    )


@pytest.fixture(autouse=True)
def _evidence_handle_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence admission in this module reloads through the fake archive, never the network."""
    seam.install(monkeypatch)


def _with_handle(result: dict[str, object]) -> dict[str, object]:
    """Attach the canonical source_handle for the passage this result returns."""
    text = next(
        (result[k] for k in ("text", "content", "matching_passage") if isinstance(result.get(k), str)),
        "",
    )
    result["source_handle"] = seam.handle_for(
        str(text),
        accession=str(result.get("accession_no") or seam.DEFAULT_ACCESSION),
        document=str(result.get("document_name") or seam.DEFAULT_DOCUMENT),
    )
    return result


def _fake_dispatch() -> Callable[[str, dict[str, object]], dict[str, object]]:
    """SEC-shaped tool results: search runs navigate (top_hits), documents carry raw passages."""

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q", "sec-10k"]}
        if name == "call_tool":
            inner = str(args.get("name", ""))
            raw_args = args.get("arguments")
            call: dict[str, object] = raw_args if isinstance(raw_args, dict) else {}
            if inner in ("get_sec_document", "get_sec_filing"):
                return _with_handle(
                    {
                        "accession_no": str(call.get("accession_no") or "0000320193-25-000079"),
                        "document_name": str(call.get("document_name") or "nvda-20250331.htm"),
                        "matching_passage": "Data center revenue grew 142% year over year.",
                        "content": "Data center revenue grew 142% year over year.",
                        "known_at": "2025-05-01",
                    }
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


def _make_run(
    repo: ResearchRepository, dispatch: Callable[[str, dict[str, object]], dict[str, object]] | None = None
) -> _LiveRun:
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    return _LiveRun(
        repo,
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        dispatch or _fake_dispatch(),
        _grounded,
        DirectorBudgets(),
    )


def _trio(sid: str, wave: int, fid: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis

    claim = [GroundedClaim(text="likely holds share", evidence_ids=["E1"])]
    stock = StockbotAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=["E1"],
        as_of="x",
        question="q?",
        answer="a",
        base_case="base",
        claims=list(claim),
    )
    bull = BullAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=["E1"],
        as_of="x",
        question="q?",
        stance="bullish",
        bull_case="up",
        claims=list(claim),
    )
    bear = BearAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=["E1"],
        as_of="x",
        question="q?",
        stance="bearish",
        bear_case="down",
        claims=list(claim),
    )
    return stock, bull, bear


def test_merge_disagreement_dedup_and_newest_requests() -> None:
    from app.research.agents import ResearchRequest
    from app.research.runner import _merge_disagreement
    from app.research.synthesis.committee import CommitteeDisagreement

    r1 = ResearchRequest(
        question="Q1?",
        why_material="m",
        requested_source_domain="SEC",
        expected_gain="high",
        requesting_agents=["stockbot"],
    )
    r1b = ResearchRequest(
        question="Q1?",
        why_material="m",
        requested_source_domain="SEC",
        expected_gain="high",
        requesting_agents=["bullbot"],
    )
    r1c = ResearchRequest(
        question="Q1?",
        why_material="m",
        requested_source_domain="SEC",
        expected_gain="high",
        requesting_agents=["bearbot"],
    )
    r2 = ResearchRequest(
        question="Q2?",
        why_material="m",
        requested_source_domain="SEC",
        expected_gain="high",
        requesting_agents=["bearbot"],
    )
    d1 = CommitteeDisagreement(
        session_id="s",
        wave_id=1,
        freeze_id="f1",
        agreement=["a1", "a2", "a1"],
        disagreement=["x"],
        critical_uncertainties=["u1"],
        requested_research=[r1],
    )
    d2 = CommitteeDisagreement(
        session_id="s",
        wave_id=2,
        freeze_id="f2",
        agreement=["a2", "a3"],
        disagreement=["x", "y"],
        critical_uncertainties=["u1", "u2"],
        requested_research=[r1b, r2],
    )
    merged = _merge_disagreement(d1, d2)
    assert (merged.session_id, merged.wave_id, merged.freeze_id) == ("s", 2, "f2")
    assert merged.agreement == ["a1", "a2", "a3"]
    assert merged.disagreement == ["x", "y"]
    assert merged.critical_uncertainties == ["u1", "u2"]
    # routing follows the newest committee read: an already-executed question never starves a newer one
    assert [r.question for r in merged.requested_research] == ["Q1?", "Q2?"]
    assert merged.requested_research[0].requesting_agents == ["bullbot"]
    assert merged.requested_research[1].requesting_agents == ["bearbot"]
    # the same question asked twice by the latest wave still unions its requesting agents
    merged2 = _merge_disagreement(
        d1,
        CommitteeDisagreement(
            session_id="s", wave_id=2, freeze_id="f2", critical_uncertainties=[], requested_research=[r1b, r1c]
        ),
    )
    assert [r.requesting_agents for r in merged2.requested_research] == [["bullbot", "bearbot"]]


def test_reused_scout_result_coercion_and_default_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import jobs as _jobs
    from app.research.agents.scout import ScoutAssignment
    from app.research.models import JobType

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("q?", "", None)
    src = run._open_source_job(sid, 1, "q?")
    asg = ScoutAssignment(
        assignment_id="scout-filings",
        session_id=sid,
        as_of="unbounded",
        role="filings",
        question="q?",
        tickers=["NVDA"],
    )
    sess, child = _jobs.create_job(
        repo.get_session(sid), repo.list_jobs(sid), job_type=JobType.SCOUT, owner="t", wave_id=1, parent_job_id=src
    )
    repo.save_session(sess)
    repo.save_job(child)
    res: dict[str, object] = {
        "assignment_id": "scout-filings",
        "coverage": "full",
        "findings": [{"text": "good finding", "evidence_ids": ["E1"]}, {"text": "  ", "evidence_ids": ["E2"]}, "junk"],
        "unknowns": ["u", 42],
        "limitations": [1, "lim"],
        "follow_up_requests": [
            {
                "question": "FQ?",
                "why_material": "m",
                "requested_source_domain": "SEC",
                "expected_gain": "high",
                "requesting_agents": ["a", 7],
            }
        ],
    }
    out = run._reused_scout_result(sid, asg, child, res)
    assert out.coverage == "full" and out.assignment_id == "scout-filings"
    assert [f.text for f in out.findings] == ["good finding"]
    assert out.unknowns == ["u"] and out.limitations == ["lim"]
    assert len(out.follow_up_requests) == 1
    assert [e.event_type for e in repo.list_events(sid)].count("scout.reused") == 1
    out2 = run._reused_scout_result(sid, asg, child, {k: v for k, v in res.items() if k != "coverage"})
    assert out2.coverage == "reused"


def test_find_completed_scout_hit_and_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import jobs as _jobs
    from app.research.agents.scout import ScoutAssignment
    from app.research.models import JobType

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("q?", "", None)
    src = run._open_source_job(sid, 1, "q?")
    asg = ScoutAssignment(
        assignment_id="scout-filings",
        session_id=sid,
        as_of="unbounded",
        role="filings",
        question="q?",
        tickers=["NVDA"],
    )
    assert run._find_completed_scout(sid, src, asg, repo.list_jobs(sid)) is None
    sess, child = _jobs.create_job(
        repo.get_session(sid), repo.list_jobs(sid), job_type=JobType.SCOUT, owner="t", wave_id=1, parent_job_id=src
    )
    repo.save_session(sess)
    repo.save_job(_jobs.start_job(child))
    repo.save_job(
        _jobs.complete_job(
            repo.get_job(child.job_id),
            result={
                "assignment_id": "scout-filings",
                "findings": [{"text": "t", "evidence_ids": ["E1"]}],
            },
        )
    )
    got = run._find_completed_scout(sid, src, asg, repo.list_jobs(sid))
    assert got is not None and got.assignment_id == "scout-filings"
    assert run._find_completed_scout(sid, "no-such-parent", asg, repo.list_jobs(sid)) is None


def test_start_reused_scout_queued_and_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import dataclasses

    from app.research import jobs as _jobs
    from app.research.models import JobType

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("q?", "", None)
    src = run._open_source_job(sid, 1, "q?")
    sess, queued = _jobs.create_job(
        repo.get_session(sid), repo.list_jobs(sid), job_type=JobType.SCOUT, owner="t", wave_id=1, parent_job_id=src
    )
    queued = dataclasses.replace(queued, diagnostics={"assignment_id": "scout-filings"})
    repo.save_session(sess)
    repo.save_job(queued)
    run._start_reused_scout(sid, queued, "scout-filings")
    assert repo.get_job(queued.job_id).status == "running"
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert "job.started" in kinds and "scout.retried" in kinds
    running = repo.get_job(queued.job_id)
    run._start_reused_scout(sid, running, "")
    assert repo.get_job(queued.job_id).status == "running"


def test_run_one_committee_empty_and_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.runner import _run_one_committee

    repo = ResearchRepository()
    run = _make_run(repo)
    run_empty = _make_run(repo)

    def _empty_fetch(session_id: str) -> list[str]:
        return []

    run_empty._fetch = _empty_fetch
    out0 = _run_one_committee(run_empty, repo, "q?", "", 1)
    assert out0["stop_reason"] == "complete:empty-with-limitations"
    assert out0["stock"] is None and out0["freeze_id"] == ""
    out = _run_one_committee(run, repo, "NVDA demand?", "2025-06-30T00:00:00+00:00", 1)
    assert out["stop_reason"] == "interrupted:one-committee"
    assert out["stock"] is not None and out["bull"] is None and out["bear"] is None
    assert out["freeze_id"] and out["evidence_ids"]
    kinds = [e.event_type for e in repo.list_events(str(out["session_id"]))]
    assert "committee.completed" in kinds and "wave.stopped" in kinds


def _real_wave1(run: _LiveRun, sid: str) -> Wave1Result:
    from app.research.director import Wave1Result
    from app.research.synthesis.committee import compute_disagreement

    src = run._open_source_job(sid, 1, "NVDA demand?")
    e1 = run._fetch_wave(sid, 1, "NVDA demand?", src, "")
    assert e1
    f1 = run._freeze_wave(sid, 1)
    s1, b1, r1 = run._committee_wave(sid, 1, "")
    d1 = compute_disagreement(s1, b1, r1)
    return Wave1Result(
        session_id=sid,
        wave_id=1,
        freeze_id=f1,
        evidence_ids=e1,
        stock=s1,
        bull=b1,
        bear=r1,
        disagreement=d1,
        question="NVDA demand?",
    )


def test_run_next_wave_empty_then_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    w1 = _real_wave1(run, sid)

    def _no_wave(sid: str, wave: int, q: str, src_job_id: str, prefix: str) -> list[str]:
        return []

    run._fetch_wave = _no_wave
    assert run._run_next_wave(w1, "TQ?") is None
    assert repo.get_session(sid).current_wave == 2
    events = repo.list_events(sid)
    kinds = [e.event_type for e in events]
    assert "wave.stopped" in kinds
    stopped = [e.payload.get("reason") for e in events if e.event_type == "wave.stopped"]
    assert "complete:empty-with-limitations" in stopped


def test_run_next_wave_full_merges_waves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import Wave1Result
    from app.research.synthesis.committee import CommitteeDisagreement

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    w1 = _real_wave1(run, sid)
    out = run._run_next_wave(w1, "TQ follow-up?")
    assert out is not None
    assert out["wave_id"] == 2
    assert out["targeted"] == "TQ follow-up?" and out["evidence_ids"]
    assert out["dossier_id"] == run.dossier_ids[-1]
    w2result = out["result"]
    assert isinstance(w2result, Wave1Result) and w2result.wave_id == 2
    assert w2result.freeze_id == out["freeze_id"] and w2result.evidence_ids == out["evidence_ids"]
    assert w2result.question == "NVDA demand?"
    w2dis = out["disagreement"]
    assert isinstance(w2dis, CommitteeDisagreement) and w2dis.wave_id == 2
    assert w2dis.freeze_id == out["freeze_id"]
    assert w1.disagreement is not None
    assert set(w1.disagreement.critical_uncertainties) <= set(w2dis.critical_uncertainties)


def test_run_next_wave_empty_dossier_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import Wave1Result
    from app.research.synthesis.committee import CommitteeDisagreement

    repo = ResearchRepository()
    run = _make_run(repo)
    assert run.dossier_ids == []
    sid = run._create_session("q?", "", None)
    stock, bull, bear = _trio(sid, 2, "f2x")
    d1 = CommitteeDisagreement(
        session_id=sid, wave_id=1, freeze_id="f1", agreement=["a1"], disagreement=["x"], critical_uncertainties=[]
    )
    w1 = Wave1Result(
        session_id=sid,
        wave_id=1,
        freeze_id="f1",
        evidence_ids=["E1"],
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=d1,
    )

    def _fetch(sid: str, wave: int, q: str, src_job_id: str, prefix: str) -> list[str]:
        return ["E1"]

    def _freeze(session_id: str, wave: int) -> str:
        return "f2x"

    def _committee(session_id: str, wave: int, prefix: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        return (stock, bull, bear)

    monkeypatch.setattr(run, "_fetch_wave", _fetch)
    monkeypatch.setattr(run, "_freeze_wave", _freeze)
    monkeypatch.setattr(run, "_committee_wave", _committee)
    out = run._run_next_wave(w1, "tq?")
    assert out is not None and out["dossier_id"] == ""
    tail_dis = out["disagreement"]
    assert isinstance(tail_dis, CommitteeDisagreement) and tail_dis.freeze_id == "f2x"
    tail_result = out["result"]
    assert isinstance(tail_result, Wave1Result) and tail_result.freeze_id == "f2x"


def test_complete_wave_tail_synth_and_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import Wave1Result

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    w1 = _real_wave1(run, sid)
    w2 = run._run_next_wave(w1, "TQ follow-up?")
    assert w2 is not None
    w2_result = w2["result"]
    assert isinstance(w2_result, Wave1Result)
    tail = run._complete_wave_tail(w2_result, str(w2["dossier_id"]), "gate:x", [w2])
    assert set(tail) == {
        "session_id",
        "wave_id",
        "freeze_id",
        "evidence_ids",
        "dossier_id",
        "stock",
        "bull",
        "bear",
        "disagreement",
        "stop_reason",
        "wave_decision",
        "novelty",
        "waves",
        "bundle_dir",
    }
    assert tail["stop_reason"] == "complete:wave2"
    assert tail["wave_id"] == 2
    assert tail["freeze_id"] == w2["freeze_id"]
    assert tail["evidence_ids"] == w2["evidence_ids"]
    assert tail["wave_decision"] == "gate:x"
    assert tail["novelty"] == dict(w2_result.novelty)
    assert tail["waves"] == [
        {
            "wave_id": 2,
            "targeted": "TQ follow-up?",
            "freeze_id": w2["freeze_id"],
            "evidence_ids": w2["evidence_ids"],
            "dossier_id": w2["dossier_id"],
            "disagreement": w2["disagreement"],
        }
    ]
    done = repo.get_session(sid)
    assert done.status == "completed" and done.final_result is not None
    assert done.final_result["freeze_id"] == w2["freeze_id"]
    final_claims = done.final_result["claims"]
    assert isinstance(final_claims, list) and final_claims
    for claim in final_claims:
        assert isinstance(claim, dict)
        assert set(claim) == {"text", "claim_type", "evidence_ids", "datapoint_ids"}
        assert claim["datapoint_ids"] == []
    none_result = Wave1Result(
        session_id=sid,
        wave_id=2,
        freeze_id="f2y",
        evidence_ids=["E1"],
        disagreement=None,
        novelty={"new_evidence_records": 0, "zero_novelty_waves": 2},
    )
    tail2 = run._complete_wave_tail(
        none_result,
        "D1",
        "gate:no_novelty",
        [
            {
                "wave_id": 2,
                "targeted": "t",
                "freeze_id": "f2y",
                "evidence_ids": ["E1"],
                "dossier_id": "D1",
                "disagreement": None,
            },
        ],
    )
    assert tail2["stop_reason"] == "complete:wave2"
    assert tail2["freeze_id"] == "f2y" and tail2["stock"] is None and tail2["disagreement"] is None
    assert tail2["novelty"] == {"new_evidence_records": 0, "zero_novelty_waves": 2}
    tail2_waves = tail2["waves"]
    assert isinstance(tail2_waves, list) and len(tail2_waves) == 1
    tail2_first = tail2_waves[0]
    assert isinstance(tail2_first, dict) and tail2_first["freeze_id"] == "f2y"
    unchanged = repo.get_session(sid).final_result
    assert unchanged is not None and unchanged["freeze_id"] == w2["freeze_id"]


def test_document_result_without_passage_persists_as_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document read carrying accession + name but no raw passage cannot ground evidence:
    it is kept as a navigation record (accession as source id, no SEC provenance)."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    base = _fake_dispatch()

    def _passage_less(name: str, args: dict[str, object]) -> dict[str, object]:
        raw = base(name, args)
        if name == "call_tool" and str(args.get("name", "")) in ("get_sec_document", "get_sec_filing"):
            return {k: v for k, v in raw.items() if k not in ("matching_passage", "content")}
        return raw

    run = _make_run(repo, _passage_less)
    sid = run._create_session("NVDA demand?", "", None)
    eids = run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    assert eids == []  # nothing citable came back, so nothing is advertised
    docs = [
        row
        for row in repo.list_evidence(sid)
        if isinstance(row.get("metadata"), dict)
        and row["metadata"].get("tool") in ("get_sec_document", "get_sec_filing")
    ]
    assert docs, "the passage-less document reads are still recorded for provenance"
    for row in docs:
        meta = row.get("metadata")
        assert isinstance(meta, dict)
        assert row["record_kind"] == "discovery"
        assert row["provenance"] == {"kind": "none"}
        assert row["source_record_id"] == "0000320193-25-000079"
        assert meta["accession_no"] == "0000320193-25-000079"
        assert meta["document_name"] == "nvda-20250331.htm"
        assert row["claim_text"] == "get_sec_document finding for NVDA"


def test_wave_novelty_counts_persisted_relationships_claims_and_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dossier rows feed novelty accounting: a relationship and a claim land once, and a
    question that stopped being open in the next wave is reported as resolved."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "relationships": [{"relationship_id": "rel-1"}, {"relationship_id": "  "}, "junk"],
            "findings": [{"text": "claim one"}, {"text": " "}, 7],
            "open_questions": ["q-1", "  "],
            "unknowns": ["q-2"],
            "coverage": {"material_open_questions": ["q-3"], "unresolved": ["q-4"], "open_questions": ["q-1"]},
        }
    )
    first = run._wave_novelty(sid, 1)
    assert (first["new_relationships"], first["new_material_claims"]) == (1, 1)
    assert first["new_questions"] == 4 and first["resolved_questions"] == 0
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:2:sec",
            "session_id": sid,
            "wave_id": 2,
            "relationships": [{"relationship_id": "rel-1"}],
            "findings": [{"text": "claim one"}],
            "open_questions": ["q-1"],
            "coverage": "junk",
        }
    )
    second = run._wave_novelty(sid, 2)
    assert (second["new_relationships"], second["new_material_claims"], second["new_questions"]) == (0, 0, 0)
    assert second["resolved_questions"] == 3
    journaled = [e.payload for e in repo.list_events(sid) if e.event_type == "wave.novelty"]
    assert journaled[-1]["resolved_questions"] == 3  # the journal carries what the gate read


def test_document_result_without_ids_keeps_accession_as_record_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document read that echoes neither accession nor name is still navigation, but its
    accession (from the call args) stays the record id so the artifact remains traceable."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _id_less(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["sec-10q"]}
        if name == "call_tool":
            inner = str(args.get("name", ""))
            if inner == "search_sec_filings":
                return {
                    "search_id": "sr:1",
                    "count": 1,
                    "top_hits": [{"accession": "0000320193-25-000079", "form": "10-Q"}],
                }
            if inner in ("get_sec_document", "get_sec_filing"):
                return {"content": "data center demand grew"}
            return {"record": {"id": "r"}}
        return {}

    run = _make_run(repo, _id_less)
    sid = run._create_session("NVDA demand?", "", None)
    assert run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "") == []
    rows = [
        row
        for row in repo.list_evidence(sid)
        if isinstance(row.get("metadata"), dict) and row["metadata"].get("tool") == "get_sec_document"
    ]
    assert rows
    for row in rows:
        assert row["record_kind"] == "discovery"  # no document name: never citable
        assert row["source_record_id"] == "0000320193-25-000079"
        assert row["provenance"] == {"kind": "none"}
        assert row["content"] == "data center demand grew"


def test_evidence_row_keeps_raw_passage_and_document_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grounded document read persists the archive's passage as the record body - the tool's
    display content never replaces what a claim must cite - and is labelled by its document
    identity."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _grounded_doc(name: str, args: dict[str, object]) -> dict[str, object]:
        raw = _fake_dispatch()(name, args)
        if name == "call_tool" and str(args.get("name", "")) in (
            "get_sec_document",
            "get_sec_filing",
        ):
            return _with_handle(
                {
                    **raw,
                    "text": "data center revenue grew 142%",
                    "content": "exhibit 99.1 cover page",
                }
            )
        return raw

    run = _make_run(repo, _grounded_doc)
    sid = run._create_session("NVDA demand?", "", None)
    eids = run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    rows = [row for row in repo.list_evidence(sid) if row.get("record_kind") == "evidence"]
    assert rows and {str(row["evidence_id"]) for row in rows} == set(eids)
    for row in rows:
        assert row["content"] == "data center revenue grew 142%"
        assert row["claim_text"] == "nvda-20250331.htm 0000320193-25-000079"
        assert row["subject"] == "NVDA demand?"
        provenance = row["provenance"]
        assert isinstance(provenance, dict) and provenance.get("kind") == "sec_source"


def test_wave_items_identity_survives_row_without_metadata_accession(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted rows are the identity source: a row carrying its accession only in
    source_record_id (service-written evidence) still names its document."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import dataclasses

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    assert run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    rec = next(r for r in run.ledger.list_session(sid) if r.record_kind == "evidence")
    assert rec.source_record_id == "0000320193-25-000079"
    legacy = dataclasses.replace(
        rec,
        evidence_id=f"{rec.evidence_id}:legacy",
        metadata={k: v for k, v in rec.metadata.items() if k != "accession_no"},
    )
    run.ledger.append(legacy)
    # the legacy row is the same document, never a second one with a blank accession
    assert run._wave_items(sid, 1)["documents"] == {"0000320193-25-000079|nvda-20250331.htm"}


def test_wave_items_slice_each_wave_by_ledger_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave items are that wave's ledger rows: a later wave's evidence never leaks into an earlier one."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    first = run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    second = run._fetch_wave(
        sid, 2, "NVDA demand? Wanli obligations?", run._open_source_job(sid, 2, "NVDA demand? Wanli obligations?"), ""
    )
    assert first and second and set(first).isdisjoint(second)
    wave1 = run._wave_items(sid, 1)
    assert wave1["evidence"] == set(first)
    assert wave1["evidence_keys"] and wave1["documents"]
    assert all(key.count("|") == 2 for key in wave1["evidence_keys"])  # accession|document|content_hash
    assert run._wave_items(sid, 2)["evidence"] == set(second)


def _empty_dispatch() -> Callable[[str, dict[str, object]], dict[str, object]]:
    """Navigation-only SEC surface: no document read carries a passage, so no evidence can land."""

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": []}
        if name == "call_tool":
            return {"content": "nothing"}
        return {}

    return _dispatch


def test_resume_reuses_running_source_job_and_ends_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume whose source job is still open reruns the wave on that same job; a wave that
    lands no evidence terminates with the limitations answer and keeps its dossier."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.runner import resume_live

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    src = run.source_jobs[0]
    out = resume_live(sid, _empty_dispatch(), _grounded, repo=repo)
    assert out["stop_reason"] == "complete:empty-with-limitations"
    assert out["wave_id"] == 1 and out["freeze_id"] == "" and out["evidence_ids"] == []
    assert out["stock"] is None and out["bull"] is None and out["bear"] is None
    assert [j.job_id for j in repo.list_jobs(sid) if j.job_type == "source_agent"] == [src]
    assert repo.get_job(src).status == "completed"  # the reused job carried the wave
    dossier_id = str(out["dossier_id"])
    assert dossier_id.startswith(f"{sid}:1:")
    reasons = [str(e.payload.get("reason")) for e in repo.list_events(sid) if e.event_type == "wave.stopped"]
    assert reasons[-1] == "complete:empty-with-limitations"


def test_resume_survives_trace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trace that cannot be reattached never aborts a resume: the wave still closes."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import app.research.runner as _runner

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)

    def _trace_down(*args: object, **kwargs: object) -> None:
        raise RuntimeError("trace store down")

    monkeypatch.setattr(_runner, "_resume_trace", _trace_down)
    out = _runner.resume_live(sid, _fake_dispatch(), _grounded, repo=repo)
    assert out["stop_reason"] == "complete:wave1"
    assert out["stock"] is not None and out["evidence_ids"]


def test_freeze_or_reuse_keeps_wave_ids_without_usable_freeze_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved freeze id always wins; a row without a usable id list leaves this wave's ids standing."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.runner import _freeze_or_reuse

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    eids = run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    assert eids
    fid = f"{sid}:1:freeze"
    repo.save_freeze({"freeze_id": fid, "session_id": sid, "wave_id": 1})  # no evidence_ids at all
    assert _freeze_or_reuse(run, repo, sid, 1, eids) == (fid, eids)
    repo.save_freeze(
        {"freeze_id": f"{sid}:2:freeze", "session_id": sid, "wave_id": 2, "evidence_ids": [eids[0], 7]}
    )  # partly malformed ids are never half-trusted
    assert _freeze_or_reuse(run, repo, sid, 2, ["EV-x"]) == (f"{sid}:2:freeze", ["EV-x"])
    fresh_fid, fresh_ids = _freeze_or_reuse(run, repo, sid, 3, ["EV-y"])
    assert fresh_fid == f"{sid}:3:freeze" and fresh_ids == ["EV-y"]  # no row: this wave freezes fresh


def test_resume_freeze_reuse_drops_navigation_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-fix freeze carrying discovery ids still resumes on it - never refrozen - but the wave
    closes on substantive evidence only: navigation artifacts never reach the gate or the answer."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import freeze as _freeze
    from app.research.evidence import evidence_from_dict
    from app.research.runner import resume_live, run_live

    repo = ResearchRepository()
    base = _fake_dispatch()

    def _dated_search(name: str, args: dict[str, object]) -> dict[str, object]:
        raw = base(name, args)
        if name == "call_tool" and str(args.get("name", "")) == "search_sec_filings":
            return {**raw, "known_at": "2025-05-01"}  # navigation rows are dated too
        return raw

    out = run_live(
        "NVDA demand?",
        "o",
        "2025-06-30T00:00:00+00:00",
        ["NVDA"],
        _dated_search,
        _grounded,
        repo=repo,
        interrupt_after="source",
    )
    sid = out["session_id"]
    assert isinstance(sid, str) and out["evidence_ids"]
    frozen_wave = [evidence_from_dict(row) for row in repo.list_evidence(sid)]
    substantive = {r.evidence_id for r in frozen_wave if r.record_kind == "evidence"}
    navigation = {r.evidence_id for r in frozen_wave if r.record_kind == "discovery"}
    assert substantive and navigation  # the fake surface opens documents and navigates both
    fid = f"{sid}:1:freeze"
    repo.save_freeze(
        _freeze.freeze_to_dict(
            _freeze.create_freeze(
                freeze_id=fid,
                session_id=sid,
                wave_id=1,
                records=frozen_wave,
                as_of="2025-06-30T00:00:00+00:00",
            )
        )
    )
    out2 = resume_live(sid, _fake_dispatch(), _grounded, repo=repo)
    assert out2["freeze_id"] == fid  # the resolved freeze id wins, never refrozen
    resumed_ids = out2["evidence_ids"]
    assert isinstance(resumed_ids, list)
    assert set(resumed_ids) == substantive
    assert navigation.isdisjoint(resumed_ids)
    assert out2["stop_reason"] == "complete:wave1"


# --- scout deadline guards (plan: optional deadlines; liveness is heartbeat-only) ---


def test_deadline_guard_none_future_and_expired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No deadline means no stop; an expired deadline fails closed with a TimeoutError naming the step."""
    from datetime import datetime, timedelta

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    run = _make_run(ResearchRepository())
    # No deadline configured (unlimited by default).
    run._scout_deadline = None
    run._raise_if_deadline_exceeded("before model call")
    run._raise_if_past_deadline("sid", "search_sec_filings", {}, 1.0)
    # Future deadline still proceeds.
    run._scout_deadline = datetime.now(UTC) + timedelta(minutes=5)
    run._raise_if_deadline_exceeded("before model call")
    run._raise_if_past_deadline("sid", "search_sec_filings", {}, 1.0)
    # Expired aware deadline fails closed, naming the step.
    run._scout_deadline = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(TimeoutError, match="before model call"):
        run._raise_if_deadline_exceeded("before model call")
    # Naive datetimes are read as UTC rather than crashing the guard.
    run._scout_deadline = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)
    with pytest.raises(TimeoutError, match="naive before"):
        run._raise_if_deadline_exceeded("naive before")


def test_past_deadline_after_dispatch_fails_the_tool_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dispatch that outlives the scout deadline is a timeout, journaled on the session."""
    from datetime import datetime, timedelta

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00")
    run._scout_deadline = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(TimeoutError, match="scout deadline exceeded after dispatch"):
        run._raise_if_past_deadline(sid, "search_sec_filings", {"query": "x"}, 3.0)
    events = [e.event_type for e in repo.list_events(sid)]
    assert "tool.failed" in events
    # A non-datetime deadline never aborts the best-effort guard.
    setattr(run, "_scout_deadline", object())  # noqa: B010 - deliberate wrong type exercises the guard
    run._raise_if_past_deadline(sid, "search_sec_filings", {"query": "x"}, 3.0)


# --- repairable tool calls: a malformed model call must not end the wave ---


def test_malformed_tool_call_returns_to_the_model_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """invalid_tool_arguments/unknown_tool/unknown_search come back as results; real failures still raise."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    calls: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        inner = str(args.get("name", name))
        raw_args = args.get("arguments")
        call: dict[str, object] = raw_args if isinstance(raw_args, dict) else {}
        calls.append(inner)
        if inner == "get_material_events":
            # A tool-reported argument error, the live defect's exact shape (no error_type).
            return {"error": "Missing required argument 'since' for tool 'get_material_events'"}
        if inner == "search_sec_filings":
            return {"error": "unknown search", "error_type": "unknown_search"}
        if inner == "list_sec_documents":
            return {"error": "provider exploded", "error_type": str(call.get("error_type") or "provider_error")}
        return {"content": "ok"}

    run = _make_run(repo, _dispatch)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    run._open_source_job(sid, 1, "q?")
    # Untyped and repairable: the error payload is handed back with its reason text.
    raw, _dur = run._guarded_tool_call(
        sid, "get_material_events", "call_tool", {"name": "get_material_events", "arguments": {}}
    )
    assert "Missing required argument" in str(raw.get("error")) and not raw.get("error_type")
    raw_search, _ = run._guarded_tool_call(
        sid, "search_sec_filings", "call_tool", {"name": "search_sec_filings", "arguments": {}}
    )
    assert raw_search.get("error_type") == "unknown_search"
    rejected = [e for e in repo.list_events(sid) if e.event_type == "tool.rejected"]
    assert len(rejected) == 2
    assert "Missing required argument" in str(rejected[0].payload.get("error"))  # reason text, never opaque
    assert [e.payload.get("error_type") for e in rejected] == ["", "unknown_search"]
    assert "tool.failed" not in [e.event_type for e in repo.list_events(sid)]
    # Fatal categories: an explicit error_type the model cannot repair still ends the wave.
    for fatal_type in ("provider_error", "auth_required", "source_unavailable", "deadline_exceeded"):
        with pytest.raises(ValueError, match="TOOL_ERROR"):
            run._guarded_tool_call(
                sid,
                "list_sec_documents",
                "call_tool",
                {"name": "list_sec_documents", "arguments": {"error_type": fatal_type}},
            )
    assert "tool.failed" in [e.event_type for e in repo.list_events(sid)]


def test_wave_completes_when_a_malformed_call_is_corrected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live failure shape: one malformed tool call, then corrected calls still produce a wave."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    seen = {"bad": 0}

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        inner = str(args.get("name", name))
        call = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
        if inner == "get_material_events" and seen["bad"] == 0:
            seen["bad"] += 1
            return {
                "error": "Missing required argument(s) for tool 'get_material_events': since",
                "error_type": "invalid_tool_arguments",
                "tool": inner,
            }
        if inner in ("get_sec_document", "get_sec_filing"):
            return _with_handle(
                {
                    "accession_no": "0000320193-25-000079",
                    "document_name": "nvda-20250331.htm",
                    "matching_passage": "Data center revenue grew.",
                    "content": "Data center revenue grew.",
                    "known_at": "2025-05-01",
                }
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
        if inner == "get_material_events":
            return _with_handle(
                {
                    "accession_no": "0000320193-25-000079",
                    "document_name": "nvda-8k.htm",
                    "matching_passage": "Material events.",
                    "content": "Material events.",
                    "known_at": "2025-05-01",
                }
            )
        return {"content": "ok"}

    run = _make_run(repo, _dispatch)
    sid = run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    src = run._open_source_job(sid, 1, "NVDA demand?")
    eids = run._fetch_wave(sid, 1, "NVDA demand?", src, "")
    assert eids, "the corrected wave still produced evidence"
    assert seen["bad"] == 1


# --- scout findings: a fabricated citation is dropped, never fatal ---


def test_scout_drops_unknown_ids_and_keeps_grounded_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scout stage: an invented id loses its own claim only; the wave keeps the grounded rest."""
    import json as _json

    from app.research.agents.scout import ScoutAssignment, run_scout

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    journal_events: list[tuple[str, dict[str, object]]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        inner = str(args.get("name", name))
        if inner == "search_sec_filings":
            return {
                "search_id": "s1",
                "count": 1,
                "top_hits": [{"accession": "0000320193-25-000079", "document": "nvda.htm", "form": "10-Q"}],
            }
        if inner in ("get_sec_document", "get_sec_filing"):
            return {
                "evidence_ids": [
                    {"evidence_id": "EV-1", "known_at": "2025-05-01", "claim_text": "revenue grew"},
                    {"evidence_id": "EV-2", "known_at": "2025-05-02", "claim_text": "margin up"},
                ]
            }
        return {"content": "ok"}

    def _model(prompt: str) -> str:
        if "Acquired evidence" in prompt:
            return _json.dumps(
                [
                    {"text": "revenue grew", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]},
                    {"text": "invented", "claim_type": "observed_fact", "evidence_ids": ["EV-999"]},
                    {"text": "unknown gap", "claim_type": "unknown", "evidence_ids": []},
                ]
            )
        return _json.dumps([])

    assignment = ScoutAssignment(
        assignment_id="a1",
        session_id="rs:x",
        as_of="unbounded",
        role="filings",
        question="q?",
        tickers=["NVDA"],
        context={},
        queries=[],
        baseline=[],
    )
    result = run_scout(assignment, dispatch=_dispatch, model=_model, journal=lambda t, p: journal_events.append((t, p)))
    texts = [c.text for c in result.findings]
    assert texts == ["revenue grew", "unknown gap"], texts
    assert any(t == "claim.rejected" for t, _ in journal_events)
    assert any("dropped" in line for line in result.limitations)
    # A non-JSON model answer contributes nothing (reported), never aborts the wave.
    empty = run_scout(assignment, dispatch=_dispatch, model=lambda p: "no json here", journal=lambda t, pl: None)
    assert empty.findings == []
    assert any("dropped" in line for line in empty.limitations)


def test_scout_tolerates_fenced_and_prose_wrapped_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real model shapes (fenced JSON, prose-wrapped list, a bare object) still ground; junk is reported."""
    from app.research.agents import parse_grounded_claims_tolerant

    frozen = ["EV-1"]
    fenced = 'Here you go:\n```json\n[{"text": "revenue grew", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]}]\n```\n'
    assert [c.text for c in parse_grounded_claims_tolerant(fenced, frozen=frozen)] == ["revenue grew"]
    prose = 'Findings: [{"text": "margin up", "claim_type": "inference", "evidence_ids": ["EV-1"]}] — done.'
    assert [c.text for c in parse_grounded_claims_tolerant(prose, frozen=frozen)] == ["margin up"]
    single = '{"text": "one finding", "claim_type": "inference", "evidence_ids": ["EV-1"]}'
    assert [c.text for c in parse_grounded_claims_tolerant(single, frozen=frozen)] == ["one finding"]
    # Unparseable output contributes nothing and is reported, never fatal at the scout boundary.
    seen: list[tuple[str, str]] = []
    assert (
        parse_grounded_claims_tolerant(
            "I could not find anything.", frozen=frozen, on_reject=lambda item, why: seen.append((item, why))
        )
        == []
    )
    assert seen and "not a JSON claim list" in seen[0][1]


# --- dossier as_of: the unbounded sentinel is "no cutoff", not a malformed date ---


def test_dossier_accepts_unbounded_as_of() -> None:
    from app.research.dossiers.sec import create_dossier

    dossier = create_dossier(
        dossier_id="d",
        session_id="rs:t",
        wave_id=1,
        as_of="unbounded",
        coverage={},
        findings=[],
        unknowns=[],
        limitations=[],
    )
    assert dossier.as_of is None
    # A genuinely malformed date still fails loudly.
    with pytest.raises(Exception, match="ISO-8601"):
        create_dossier(
            dossier_id="d",
            session_id="rs:t",
            wave_id=1,
            as_of="not-a-date",
            coverage={},
            findings=[],
            unknowns=[],
            limitations=[],
        )


# --- binary document bytes: navigation provenance only, never citable ---


def test_binary_document_text_persists_as_nul_free_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document read whose text is raw binary (PDF bytes carrying NULs) is not readable
    text: it persists as a NUL-free discovery record and the rejection is journaled."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    pdf = "%PDF-1.5\x00\x00/Filter/FlateDecode\x00endobj"

    def _binary_doc(name: str, args: dict[str, object]) -> dict[str, object]:
        raw = _fake_dispatch()(name, args)
        if name == "call_tool" and str(args.get("name", "")) in (
            "get_sec_document",
            "get_sec_filing",
        ):
            return _with_handle({**raw, "text": pdf, "content": pdf})
        return raw

    run = _make_run(repo, _binary_doc)
    sid = run._create_session("NVDA demand?", "", None)
    eids = run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "")
    assert eids == []  # raw bytes can never ground a claim, so nothing is advertised
    docs = [
        row
        for row in repo.list_evidence(sid)
        if isinstance(row.get("metadata"), dict)
        and row["metadata"].get("tool") in ("get_sec_document", "get_sec_filing")
    ]
    assert docs, "the binary document reads are still recorded for provenance"
    for row in docs:
        assert row["record_kind"] == "discovery"
        assert row["content"] == "%PDF-1.5/Filter/FlateDecodeendobj"  # stored NUL-free
    rejected = [e for e in repo.list_events(sid) if e.event_type == "evidence.rejected"]
    assert {str(e.payload.get("reason")) for e in rejected} == {"binary_source_unreadable"}
    assert {str(e.payload.get("evidence_id")) for e in rejected} == {str(row["evidence_id"]) for row in docs}


def test_model_failure_category_matches_the_real_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashing model call is a MODEL_ERROR, never a fake timeout; a real timeout stays TIMEOUT."""
    from app.research.models import FailureCategory

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    crash_sid = run._create_session("NVDA demand?", "", None)
    crash_job = run._open_source_job(crash_sid, 1, "NVDA demand?")
    detail, category = run._persist_model_failure(
        crash_sid, crash_job, "committee-stockbot", ValueError("embedded null byte")
    )
    assert "ValueError" in detail
    assert category == FailureCategory.MODEL_ERROR
    crash = repo.get_job(crash_job).failure
    assert crash is not None and crash.category == "model_error"
    crash_session = repo.get_session(crash_sid).failure
    assert crash_session is not None and crash_session.category == "model_error"
    assert [e.payload.get("reason") for e in repo.list_events(crash_sid) if e.event_type == "research.failed"] == [
        "model-error:ValueError"
    ]
    # Provider quota exhaustion is an external block: reported as provider_error, never a fake timeout.
    quota_sid = run._create_session("quota?", "", None)
    quota_job = run._open_source_job(quota_sid, 1, "quota?")
    _q_detail, quota_category = run._persist_model_failure(
        quota_sid,
        quota_job,
        "committee-stockbot",
        RuntimeError('Pi model call failed: 429: {"type":"GoUsageLimitError","message":"Go usage limit exceeded"}'),
    )
    assert quota_category == FailureCategory.PROVIDER_ERROR
    quota_failure = repo.get_job(quota_job).failure
    assert quota_failure is not None and quota_failure.category == "provider_error"
    timeout_sid = run._create_session("timeout?", "", None)
    timeout_job = run._open_source_job(timeout_sid, 1, "timeout?")
    run._persist_model_failure(timeout_sid, timeout_job, "source-scout", TimeoutError("deadline"))
    timed = repo.get_job(timeout_job).failure
    assert timed is not None and timed.category == "timeout"
    assert [e.payload.get("reason") for e in repo.list_events(timeout_sid) if e.event_type == "research.failed"] == [
        "timeout:model-call"
    ]


# --- committee failure categories: a crash is never reported as a timeout ---


def test_committee_category_keeps_the_real_cause() -> None:
    from app.research.models import FailureCategory
    from app.research.runner import LiveModelError, _LiveRun

    assert _LiveRun._categorize_committee_error(TimeoutError("pi call timed out")) == FailureCategory.TIMEOUT
    # An argv/OS failure says nothing about time: reporting TIMEOUT would mislead resume/retry.
    assert (
        _LiveRun._categorize_committee_error(OSError("[Errno 7] Argument list too long: 'pi'"))
        == FailureCategory.MODEL_ERROR
    )
    # A model-stage failure carries the category it already persisted.
    err = LiveModelError("rs:1", "committee-stockbot", "boom", FailureCategory.MODEL_ERROR)
    assert _LiveRun._categorize_committee_error(err) == FailureCategory.MODEL_ERROR
    # Ladder arm: loop/policy text is named as such instead of a generic model error.
    assert (
        _LiveRun._categorize_committee_error(RuntimeError("policy_rejection: explicit tool limit reached"))
        == FailureCategory.POLICY_REJECTION
    )
    assert (
        _LiveRun._categorize_committee_error(RuntimeError("research_loop_detected: duplicate action"))
        == FailureCategory.RESEARCH_LOOP_DETECTED
    )
    # Keyword arm: contract breaks the shared ladder does not claim keep their own name.
    assert (
        _LiveRun._categorize_committee_error(RuntimeError("model_output_failure: uncited finding"))
        == FailureCategory.MODEL_OUTPUT_FAILURE
    )
    assert (
        _LiveRun._categorize_committee_error(RuntimeError("tool failed: upstream down")) == FailureCategory.TOOL_ERROR
    )


# --- rejected tool calls: a result for the agent, never a wave failure ---


def test_rejected_tool_call_never_ingests_and_blocks_its_own_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live defect: an untyped tool error (missing argument) is a result, not a session failure.

    The rejected call lands no discovery row, is journaled with its reason text, and
    registers as a zero-progress action - so the two later identical calls of the
    same role tool never reach the provider again, while the rest of the wave runs.
    """
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    calls: list[str] = []
    base = _fake_dispatch()

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "call_tool":
            inner = str(args.get("name", ""))
            calls.append(inner)
            if inner == "list_sec_filings":
                return {"error": "Missing required argument 'since' for tool 'list_sec_filings'"}
        return base(name, args)

    run = _make_run(repo, _dispatch)
    sid = run._create_session("NVDA demand?", "", None)
    src = run._open_source_job(sid, 1, "NVDA demand?")
    eids = run._fetch_wave(sid, 1, "NVDA demand?", src, "")
    assert eids, "the other role tools still produced evidence"
    # filings/financials/risk all call list_sec_filings identically: one dispatch, two blocks.
    assert calls.count("list_sec_filings") == 1
    assert run.wave_actions[1]["duplicate_actions_blocked"] >= 2
    kinds = [e.event_type for e in repo.list_events(sid)]
    assert "research_loop_detected" in kinds
    rejected = [e for e in repo.list_events(sid) if e.event_type == "tool.rejected"]
    assert [str(e.payload.get("tool")) for e in rejected] == ["list_sec_filings"]
    assert "Missing required argument" in str(rejected[0].payload.get("error"))
    assert not [
        row
        for row in repo.list_evidence(sid)
        if isinstance(row.get("metadata"), dict) and row["metadata"].get("tool") == "list_sec_filings"
    ]


def test_rejected_action_registration_blocks_the_repeat_via_the_loop_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration a rejected call gets is what stops an unbounded malformed retry."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    calls: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        calls.append(str(args.get("name", name)))
        return {"error": "No pair of filings found for NVDA"}

    run = _make_run(repo, _dispatch)
    sid = run._create_session("NVDA demand?", "", None)
    run._open_source_job(sid, 1, "NVDA demand?")
    args: dict[str, object] = {"name": "diff_sec_filings", "arguments": {"ticker": "NVDA"}}
    raw, _ = run._guarded_tool_call(sid, "diff_sec_filings", "call_tool", args)
    assert raw.get("error") == "No pair of filings found for NVDA"
    run._register_rejected_action("diff_sec_filings", args, raw)
    raw_again, _ = run._guarded_tool_call(sid, "diff_sec_filings", "call_tool", args)
    assert raw_again == {"evidence_ids": []}  # blocked before the provider call
    assert calls == ["diff_sec_filings"]
    assert [e.event_type for e in repo.list_events(sid)].count("research_loop_detected") == 1


# --- scout assignments: one failure is a limitation, not the end of the wave ---


def test_timed_out_scout_assignment_leaves_the_wave_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scout model-call timeout fails that assignment only: the others still land evidence."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    draft_calls = {"n": 0}

    def _model(prompt: str) -> str:
        if "Respond with JSON only" in prompt:  # one scout's finding-draft call
            draft_calls["n"] += 1
            if draft_calls["n"] == 1:
                raise TimeoutError("pi model call timed out after 300s")
        return _grounded(prompt)

    run = _make_run(repo)
    run.model = _model
    sid = run._create_session("NVDA demand?", "", None)
    src = run._open_source_job(sid, 1, "NVDA demand?")
    eids = run._fetch_wave(sid, 1, "NVDA demand?", src, "")
    assert eids, "the remaining assignments completed"
    scouts = [j for j in repo.list_jobs(sid) if j.job_type == "scout"]
    assert [j.status for j in scouts].count("failed") == 1
    dead = next(j for j in scouts if j.status == "failed")
    assert dead.failure is not None and dead.failure.category == "timeout"
    assert repo.get_session(sid).failure is None  # the session is not failed by one assignment
    assert "scout.degraded" in [e.event_type for e in repo.list_events(sid)]
    dossier = repo.get_dossier(f"{sid}:1:sec")
    limitations = dossier.get("limitations")
    assert isinstance(limitations, list) and any("scout-filings unavailable" in str(line) for line in limitations)


def test_failed_scout_degrades_but_runlevel_faults_stay_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider/model trouble in one assignment returns a limitation; a tool fault
    or a run-level denial still ends the wave."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.agents.scout import ScoutAssignment
    from app.research.models import FailureCategory

    repo = ResearchRepository()
    run = _make_run(repo)
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
    scout_job = run._open_or_reuse_scout_job(sid, 1, src, assignment, repo.list_jobs(sid))
    slow = TimeoutError("pi model call timed out")
    setattr(slow, "_failure_category", FailureCategory.TIMEOUT)  # noqa: B010 - pre-tagged category, like the runner tags it
    degraded = run._degrade_scout_assignment(sid, assignment, scout_job, slow)
    assert degraded.assignment_id == "scout-filings" and degraded.findings == []
    assert any("scout-filings unavailable (timeout)" in line for line in degraded.limitations)
    assert [e.event_type for e in repo.list_events(sid)].count("scout.degraded") == 1
    # A tool dispatch that raised is an infrastructure fault, not provider slowness.
    with pytest.raises(RuntimeError, match="downstream down"):
        run._degrade_scout_assignment(sid, assignment, scout_job, RuntimeError("downstream down"))
    denied = RuntimeError("policy_rejection: explicit tool limit reached")
    setattr(denied, "_failure_category", FailureCategory.POLICY_REJECTION)  # noqa: B010 - pre-tagged category
    with pytest.raises(RuntimeError, match="policy_rejection"):
        run._degrade_scout_assignment(sid, assignment, scout_job, denied)

    class _Cancelled(RuntimeError):
        pass

    with pytest.raises(_Cancelled):
        run._degrade_scout_assignment(sid, assignment, scout_job, _Cancelled("scout cancelled"))


# --- wave1 close + resume helpers ---


def test_close_wave1_result_interrupt_and_empty_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted wave returns its evidence with no invented analyses; a wave with
    nothing at all closes through the limitations terminal."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.director import Wave1Result
    from app.research.runner import _close_wave1_result

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    interrupted = Wave1Result(
        session_id=sid, wave_id=1, freeze_id=f"{sid}:1:freeze", evidence_ids=["E1"], question="NVDA demand?"
    )
    out = _close_wave1_result(run, interrupted, "D1", "freeze")
    assert out["stop_reason"] == "interrupted:freeze"
    assert (out["stock"], out["bull"], out["bear"], out["disagreement"]) == (None, None, None, None)
    assert out["evidence_ids"] == ["E1"] and out["dossier_id"] == "D1"
    empty = _close_wave1_result(
        run, Wave1Result(session_id=sid, wave_id=2, freeze_id="", evidence_ids=[], question="nothing?"), "", None
    )
    assert empty["stop_reason"] == "complete:empty-with-limitations"
    assert empty["wave_id"] == 2 and empty["freeze_id"] == "" and empty["evidence_ids"] == []
    assert repo.list_evidence(sid) == []  # the empty close invents nothing
    # The trio ran: the completed close drives the gate/synthesis tail, not a stub payload.
    done_run = _make_run(repo)
    done_sid = done_run._create_session("NVDA demand?", "2025-06-30T00:00:00+00:00", None)
    completed = _close_wave1_result(done_run, _real_wave1(done_run, done_sid), "", None)
    assert str(completed["stop_reason"]).startswith("complete:")
    assert completed["stock"] is not None and completed["evidence_ids"]
    final = repo.get_session(done_sid)
    assert final.final_result is not None and final.final_result["freeze_id"] == completed["freeze_id"]
    # Walked to synthesizing first (the real run's state): the terminal answer persists.
    from app.research import session as _session
    from app.research.models import SessionStatus

    empty_sid = run._create_session("nothing?", "", None)
    sess = repo.get_session(empty_sid)
    for step in (SessionStatus.FREEZING, SessionStatus.ANALYZING, SessionStatus.SYNTHESIZING):
        sess = _session.transition_session(sess, step)
        repo.save_session(sess)
    closed = _close_wave1_result(
        run, Wave1Result(session_id=empty_sid, wave_id=1, freeze_id="", evidence_ids=[], question="nothing?"), "", None
    )
    assert closed["stop_reason"] == "complete:empty-with-limitations"
    closed_final = repo.get_session(empty_sid)
    assert closed_final.status == "completed"
    assert closed_final.final_result is not None and closed_final.final_result["claims"] == []


def test_merge_session_dossiers_appends_new_ids_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume keeps the run's dossier order and adds only unseen session ids."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import dataclasses

    from app.research.runner import _merge_session_dossiers

    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("q?", "", None)
    run.dossier_ids.append("D-run")
    sess = dataclasses.replace(repo.get_session(sid), dossier_ids=["D-run", "", "D-2", "D-2"])
    _merge_session_dossiers(run, sess)
    assert run.dossier_ids == ["D-run", "D-2"]
    _merge_session_dossiers(run, dataclasses.replace(sess, dossier_ids=[]))
    assert run.dossier_ids == ["D-run", "D-2"]


def test_substantive_ids_filters_frozen_ids_against_the_ledger() -> None:
    """A reused freeze keeps the session's substantive ids; with no substantive rows
    yet, every frozen id stands."""
    from app.research.runner import _substantive_ids

    assert _substantive_ids(["E1", "EV-nav"], {"E1"}) == ["E1"]
    assert _substantive_ids(["E1", "EV-nav"], set()) == ["E1", "EV-nav"]
    assert _substantive_ids([], {"E1"}) == []
    assert _substantive_ids(["A", "B", "A"], {"A", "B"}) == ["A", "B", "A"]


# --- runner retention: fetch/materialize never archives (service acceptance does) ---


def test_runner_retention_no_archive_on_reject_or_admit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bogus handle demotes to discovery and a valid handle admits as evidence, but
    neither path archives: the runner materializes from the archive, only service
    acceptance persists to it. The record_kind=='evidence' assertion below pins
    candidate/materialized evidence without archival; it is not equivalent to
    production evidence admission."""
    from app.sec.archive import iter_archived_documents

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    acc, doc = "0000320193-25-000079", "nvda-20250331.htm"
    root = tmp_path / "raw"
    base = _fake_dispatch()

    def _bogus(name: str, args: dict[str, object]) -> dict[str, object]:
        raw = base(name, args)
        if name == "call_tool" and str(args.get("name", "")) in (
            "get_sec_document",
            "get_sec_filing",
        ):
            raw["source_handle"] = {"basis": "guessed"}
        return raw

    before = list(iter_archived_documents(acc, doc, root=root))
    assert before == []
    run = _make_run(repo, _bogus)
    sid = run._create_session("NVDA demand?", "", None)
    assert run._fetch_wave(sid, 1, "NVDA demand?", run.source_jobs[0], "") == []
    docs = [
        row
        for row in repo.list_evidence(sid)
        if isinstance(row.get("metadata"), dict)
        and row["metadata"].get("tool") in ("get_sec_document", "get_sec_filing")
    ]
    assert docs, "the bogus-handle reads are still recorded for provenance"
    assert all(row["record_kind"] == "discovery" for row in docs)
    assert [row for row in repo.list_evidence(sid) if row.get("record_kind") == "evidence"] == []
    assert [e for e in repo.list_events(sid) if e.event_type == "evidence.accepted"] == []
    assert repo.get_session(sid).evidence_ids == []
    assert list(iter_archived_documents(acc, doc, root=root)) == before == []

    run2 = _make_run(repo)
    sid2 = run2._create_session("NVDA demand?", "", None)
    raw_valid = base(
        "call_tool",
        {"name": "get_sec_document", "arguments": {"accession_no": acc, "document_name": doc}},
    )
    before2 = list(iter_archived_documents(acc, doc, root=root))
    record = run2._build_evidence_record(
        sid2,
        1,
        "NVDA demand?",
        run2.source_jobs[0],
        "get_sec_document",
        {"name": "get_sec_document", "arguments": {"accession_no": acc, "document_name": doc}},
        raw_valid,
        "EV-admit",
    )
    # Candidate/materialized evidence only: must not be read as production admission.
    assert record.record_kind == "evidence"
    assert list(iter_archived_documents(acc, doc, root=root)) == before2 == []
    assert repo.list_evidence(sid2) == []
    assert [e for e in repo.list_events(sid2) if e.event_type == "evidence.accepted"] == []
    assert repo.get_session(sid2).evidence_ids == []


def _bundle_record(
    evidence_id: str = "EV-bundle",
    session_id: str = "bundle-session",
    content: str = "Revenue grew.\nExact source text.",
) -> Evidence:
    from app.research.evidence import Evidence, evidence_content_hash

    return Evidence(
        evidence_id=evidence_id,
        session_id=session_id,
        wave_id=1,
        source_type="sec",
        source_name="get_sec_document",
        subject="NVDA",
        claim_text="Revenue grew.",
        content=content,
        content_hash=evidence_content_hash(content),
        retrieved_at=datetime(2025, 5, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("url", "excerpt"),
    [("", "quote"), ("https://example.com", " "), (12, "quote"), ("https://example.com", None)],
)
def test_web_source_ref_rejects_missing_required_text(url: object, excerpt: object) -> None:
    from app.research.evidence import web_source_ref

    with pytest.raises(ValueError, match="web_source_ref: url and excerpt must be non-empty strings"):
        web_source_ref(url=url, excerpt=excerpt)


@pytest.mark.parametrize("representation", [None, " ", 17, " html "])
def test_web_source_ref_trims_and_bounds_provider_content(representation: object) -> None:
    from app.research.evidence import web_source_ref

    ref = web_source_ref(
        url=" https://example.com/page ",
        excerpt=" quoted text ",
        title=" Headline ",
        domain=" example.com ",
        published_at=" 2025-04-30 ",
        retrieved_at=" 2025-05-01 ",
        tool_result_id=" result-1 ",
        source_content=" " + "a" * 12001 + " ",
        source_representation=representation,
    )
    assert ref == {
        "kind": "web_source",
        "url": "https://example.com/page",
        "excerpt": "quoted text",
        "title": "Headline",
        "domain": "example.com",
        "published_at": "2025-04-30",
        "retrieved_at": "2025-05-01",
        "tool_result_id": "result-1",
        "source_content": "a" * 12000,
        "source_representation": "html" if representation == " html " else "text",
    }
    excerpt_only = web_source_ref(
        url="https://example.com/page",
        excerpt="quoted text",
        title=" ",
        domain=17,
        source_content=" ",
        source_representation="html",
    )
    assert excerpt_only == {"kind": "web_source", "url": "https://example.com/page", "excerpt": "quoted text"}


@pytest.mark.parametrize(
    ("source_uri", "record_id", "provenance", "expected_url", "expected_accession"),
    [
        (
            " https://explicit.test/doc ",
            "record",
            {"source_uri": "https://provenance.test/doc"},
            "https://explicit.test/doc",
            "record",
        ),
        (None, "record", {"source_uri": " https://provenance.test/doc "}, "https://provenance.test/doc", "record"),
        (
            " ",
            "0000320193-25-000079",
            {},
            "https://www.sec.gov/Archives/edgar/data/000032019325000079",
            "0000320193-25-000079",
        ),
        (
            None,
            "record",
            {"accession_no": "0000320193-25-000080"},
            "https://www.sec.gov/Archives/edgar/data/000032019325000080",
            "0000320193-25-000080",
        ),
        (None, None, {"source_uri": " ", "accession_no": ""}, "", ""),
        (
            None,
            "record-id",
            {"source_uri": 17, "accession_no": 42},
            "https://www.sec.gov/Archives/edgar/data/recordid",
            "record-id",
        ),
        (None, "---", {}, "", "---"),
    ],
)
def test_bundle_entry_source_url_precedence(
    source_uri: str | None,
    record_id: str | None,
    provenance: dict[str, JSONValue],
    expected_url: str,
    expected_accession: str,
) -> None:
    from dataclasses import replace

    from app.research.evidence import evidence_bundle_entry

    rec = replace(_bundle_record(), source_uri=source_uri, source_record_id=record_id)
    # Legacy mutable provenance can hold fields beyond the canonical constructor shape.
    rec.provenance.update(provenance)
    entry = evidence_bundle_entry(rec)
    assert entry["source_url"] == expected_url
    assert entry["accession"] == expected_accession
    assert entry["evidence_id"] == rec.evidence_id
    assert entry["session_id"] == rec.session_id


@pytest.mark.parametrize(
    ("provenance", "metadata", "locator", "artifact", "form"),
    [
        (
            {
                "accession_no": "0000320193-25-000079",
                "document_name": "source.htm",
                "section": "Risk",
                "offset": 0,
                "end": 20,
            },
            {"document_name": "fallback.htm", "source_bytes": "archived", "form": "10-Q"},
            {"document": "source.htm", "section": "Risk", "start": 0, "end": 20},
            "source://sec/0000320193-25-000079/source.htm",
            "10-Q",
        ),
        (
            {"document_name": "", "section": 1, "offset": True, "end": False},
            {"document_name": "fallback.htm", "filing_form": "10-K", "source_bytes": "archived"},
            {"document": "fallback.htm", "section": None, "start": None, "end": None},
            None,
            "10-K",
        ),
        (
            {"document_name": 42, "offset": "0", "end": None},
            {"document_name": 12, "form": 7},
            {"document": "12", "section": None, "start": None, "end": None},
            None,
            None,
        ),
        ({}, {"document_name": None}, {"document": "", "section": None, "start": None, "end": None}, None, None),
    ],
)
def test_bundle_entry_preserves_locator_and_archived_identity(
    provenance: dict[str, JSONValue],
    metadata: dict[str, JSONValue],
    locator: dict[str, JSONValue],
    artifact: str | None,
    form: str | None,
) -> None:
    from app.research.evidence import evidence_bundle_entry

    rec = _bundle_record()
    rec.provenance.update(provenance)
    rec.metadata.update(metadata)
    entry = evidence_bundle_entry(rec, accepted_at="2025-06-01T00:00:00+00:00")
    assert entry["source_locator"] == locator
    assert entry["source_artifact"] == artifact
    assert entry["form"] == form
    assert entry["accepted_at"] == "2025-06-01T00:00:00+00:00"
    assert entry["retrieved_at"] == "2025-05-01T00:00:00+00:00"
    assert entry["rendered_to_model"] == rec.content
    assert entry["source_hash"] == f"sha256:{rec.content_hash}"


@settings(derandomize=True, max_examples=40)
@given(
    evidence_id=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=40),
    source_uri=st.one_of(st.none(), st.text(alphabet=" abcdef:/._-", max_size=40)),
    accession=st.one_of(st.none(), st.text(alphabet="0123456789-", max_size=20)),
)
def test_bundle_entry_preserves_id_and_string_url(
    evidence_id: str,
    source_uri: str | None,
    accession: str | None,
) -> None:
    from dataclasses import replace

    from app.research.evidence import evidence_bundle_entry

    rec = replace(_bundle_record(evidence_id), source_uri=source_uri, source_record_id=accession)
    entry = evidence_bundle_entry(rec)
    assert entry["evidence_id"] == evidence_id
    assert isinstance(entry["source_url"], str)
    assert entry["rendered_to_model"] == rec.content
    assert entry["accepted_at"] == entry["retrieved_at"]
    if source_uri is not None and source_uri.strip():
        assert entry["source_url"] == source_uri.strip()


def test_write_bundle_selects_substantive_unique_rows_and_uses_ledger_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from app.research.evidence import evidence_to_dict

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("Exact question?", "", None)
    ledger_record = _bundle_record("EV-ledger", sid)
    run.ledger.append(ledger_record)
    persisted_record = _bundle_record("EV-persisted", sid, "Second source text.")
    navigation = replace(ledger_record, evidence_id="EV-navigation", record_kind="discovery")
    rows: list[object] = [
        None,
        {},
        {"evidence_id": 7},
        {"evidence_id": ""},
        {"evidence_id": "EV-ledger", "content": "corrupt persisted copy"},
        evidence_to_dict(persisted_record),
        evidence_to_dict(navigation),
        {"evidence_id": "EV-ledger"},
        evidence_to_dict(persisted_record),
    ]

    def _rows(session_id: str) -> list[object]:
        return rows

    monkeypatch.setattr(repo, "list_evidence", _rows)
    bundle = run.write_bundle(sid)
    assert {path.name for path in (bundle / "evidence").iterdir()} == {"EV-ledger.json", "EV-persisted.json"}
    exported = json.loads((bundle / "evidence" / "EV-ledger.json").read_text())
    assert exported["rendered_to_model"] == ledger_record.content
    assert exported["evidence_id"] == "EV-ledger"
    assert json.loads((bundle / "eval.json").read_text()) == {
        "session_id": sid,
        "artifacts": [f"sha256_{ledger_record.content_hash}", f"sha256_{persisted_record.content_hash}"],
    }
    request = json.loads((bundle / "request.json").read_text())
    session = repo.get_session(sid)
    assert session.as_of is not None
    assert request == {"question": "Exact question?", "as_of": session.as_of.isoformat()}


def test_write_bundle_skips_invalid_export_entries_and_sanitizes_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from app.research.journal import append_event

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("Bundle question?", "2025-06-30T00:00:00+00:00", None)
    repo.save_session(replace(repo.get_session(sid), final_result={"answer": "Exact answer."}))
    repo.save_event(append_event(sid, "tool.completed", "runner", "runner", {"tool": "get_sec_document"}))
    rows: list[object] = [
        None,
        {},
        {"evidence_id": 1},
        {"evidence_id": ""},
        {"evidence_id": "EV/a:bé.-", "text": "exact"},
    ]

    def _entries(session_id: str) -> tuple[list[object], list[str]]:
        return rows, ["artifact-name"]

    def _agents(session_id: str) -> dict[str, object]:
        return {"bull/role": {"result": "exact"}}

    monkeypatch.setattr(run, "_bundle_evidence_entries", _entries)
    monkeypatch.setattr(run, "_bundle_agents", _agents)
    bundle = run.write_bundle(sid)
    assert {path.name for path in (bundle / "evidence").iterdir()} == {"EV_a_bé.-.json"}
    assert json.loads((bundle / "evidence" / "EV_a_bé.-.json").read_text()) == rows[-1]
    assert json.loads((bundle / "agents" / "bull_role.json").read_text()) == {"result": "exact"}
    assert json.loads((bundle / "answer.json").read_text()) == {
        "answer": "Exact answer.",
        "final_result": {"answer": "Exact answer."},
    }
    assert json.loads((bundle / "eval.json").read_text()) == {"session_id": sid, "artifacts": ["artifact-name"]}
    calls = [json.loads(line) for line in (bundle / "tool_calls.jsonl").read_text().splitlines()]
    # job./tool. events go to the export; session.created does not.
    assert [call["event_type"] for call in calls] == ["job.created", "tool.completed"]
    assert calls[0]["payload"] == {"job_id": run.source_jobs[-1], "job_type": "source_agent"}
    assert calls[1]["payload"] == {"tool": "get_sec_document"}
    assert (bundle / "artifacts").is_dir()


def test_evidence_record_preserves_archive_dates_and_navigation_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _make_run(repo)
    sid = run._create_session("NVDA demand?", "", None)
    now = datetime(2025, 6, 1, tzinfo=UTC)
    monkeypatch.setattr("app.research.runner.utcnow", lambda: now)
    args: dict[str, object] = {
        "arguments": {"ticker": " NVDA ", "query": " revenue "},
    }
    raw = _with_handle(
        {
            "accession_no": seam.DEFAULT_ACCESSION,
            "document_name": seam.DEFAULT_DOCUMENT,
            "text": "Revenue grew.",
            "source_uri": "https://tool.test/doc",
            "known_at": "2025-05-02",
        }
    )
    doc = run._build_evidence_record(sid, 1, "NVDA demand?", "job-1", "get_sec_document", args, raw, "EV-document")
    assert doc.retrieved_at == datetime(2025, 5, 1, tzinfo=UTC)
    assert doc.published_at == datetime(2025, 4, 30, tzinfo=UTC)
    assert doc.known_at == datetime(2025, 5, 1, tzinfo=UTC)
    assert (
        doc.source_uri
        == f"https://www.sec.gov/Archives/edgar/data/{seam.DEFAULT_ACCESSION.replace('-', '')}/{seam.DEFAULT_DOCUMENT}"
    )
    assert doc.job_id == "job-1"
    assert doc.metadata["search_scope"] == "NVDA"
    assert doc.metadata["query"] == "revenue"
    nav = run._build_evidence_record(
        sid,
        1,
        "NVDA demand?",
        "",
        "search_sec_filings",
        args,
        {"search_id": " search-1 ", "known_at": "2025-05-02", "source_uri": "https://tool.test/search"},
        "EV-navigation",
    )
    assert nav.retrieved_at == now
    assert nav.published_at is None
    assert nav.known_at == datetime(2025, 5, 2, tzinfo=UTC)
    assert nav.source_uri == "https://tool.test/search"
    assert nav.job_id is None
    assert nav.metadata == {
        "tool": "search_sec_filings",
        "tickers": "NVDA",
        "record_kind": "discovery",
        "query": "revenue",
        "search_id": "search-1",
        "search_scope": "NVDA",
    }
    assert nav.provenance == {"kind": "search_run", "search_id": "search-1", "query": "revenue"}
