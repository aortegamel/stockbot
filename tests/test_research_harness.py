"""Permanent §44/§45 gates: transitions, budgets, journal, PIT, append-only,
freeze immutability + cumulative E2, dossier refs, same-freeze parallel
committee, resume at source/freeze/one-member. All fakes, no network."""

import dataclasses
import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tests.research_source_seam as seam
from app.research import jobs as _jobs
from app.research import session as _session
from app.research.agents.bearbot import BearAnalysis
from app.research.agents.bullbot import BullAnalysis
from app.research.agents.stockbot import StockbotAnalysis
from app.research.dossiers.sec import (
    DossierIntegrityError,
    create_dossier,
    validate_dossier,
)
from app.research.evidence import (
    Evidence,
    EvidenceLedger,
    EvidenceRejectedError,
    evidence_content_hash,
    ingest_evidence,
)
from app.research.freeze import FreezeIntegrityError, create_freeze, verify_freeze
from app.research.journal import append_event
from app.research.models import Job, JSONValue, SessionStatus
from app.research.repository import ResearchRepository
from app.research.runner import run_live

ASOF = datetime(2025, 6, 30, tzinfo=UTC)


def _committee_output(claims: list[dict[str, object]], follow_ups: Sequence[object] = ()) -> str:
    """Rich committee envelope (executive_view, claims, channels, materiality, uncertainties,
    what_would_change, follow_ups): the role contract every trio prompt now demands."""
    import json as _json

    ids: list[str] = []
    for claim in claims:
        raw_ids = claim.get("evidence_ids")
        if isinstance(raw_ids, list):
            ids.extend(eid for eid in raw_ids if isinstance(eid, str))
    channels: list[dict[str, object]] = (
        [{"text": "frozen-evidence channel", "direction": "pressure", "evidence_ids": [ids[0]]}] if ids else []
    )
    return _json.dumps(
        {
            "executive_view": "Balanced read of the frozen evidence.",
            "claims": claims,
            "impact_channels": channels,
            "materiality": {"overall": "medium", "reasoning": "effect size read off the frozen evidence"},
            "uncertainties": ["the freeze does not settle timing"],
            "what_would_change": ["a new filing disclosing the terms"],
            "follow_ups": list(follow_ups),
        }
    )


def _grounded(prompt: str, follow_ups: Sequence[object] = ()) -> str:
    """Fake grounded model: cite only freeze/acquired ids listed in the prompt."""
    import json as _json
    import re as _re

    seen: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        bracket = _re.match(r"^\[([^\[\]]+)\]", stripped)
        if bracket is not None:
            token = str(bracket.group(1)).strip()
            if token and token not in seen:
                seen.append(token)
            continue
        acquired = _re.match(r"^-\s+(\S+)", stripped)
        if acquired is not None:
            token = str(acquired.group(1)).strip()
            if (token.startswith("EV-") or ":sec:" in token) and token not in seen:
                seen.append(token)
    if not seen:
        claims: list[dict[str, object]] = []
    else:
        claims = [{"text": f"grounded finding {i}", "evidence_ids": [eid]} for i, eid in enumerate(seen[:6])]
    if "Temporary assignment" in prompt:
        return _json.dumps(claims)
    return _committee_output(claims, follow_ups)


def _ev(eid: str, sid: str, wave: int, known: datetime) -> Evidence:
    return Evidence(
        evidence_id=eid,
        session_id=sid,
        wave_id=wave,
        source_type="sec",
        source_name="SEC",
        source_uri="https://sec.gov/x",
        source_record_id="r",
        subject="NVDA",
        claim_text="c",
        content="c-" + eid,
        content_hash=evidence_content_hash("c-" + eid),
        known_at=known,
        retrieved_at=datetime(2025, 6, 1, tzinfo=UTC),
        job_id="J-1",
        agent_id="s-A",
    )


def _frozen_write(obj: object, name: str, value: object) -> None:
    """Write one attribute through the instance __setattr__ (frozen dataclasses raise)."""
    setattr(obj, name, value)


def test_transitions_valid_and_invalid() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    s2 = _session.transition_session(s, SessionStatus.PLANNING)
    assert s2.status == SessionStatus.PLANNING.value
    with pytest.raises(ValueError):
        _session.transition_session(s, SessionStatus.COMPLETED)


def test_budgets_enforced_from_policy() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    jobs: list[Job] = []
    last = s
    created = 0
    with pytest.raises(ValueError, match="max_parallel|max_total_jobs"):
        for _ in range(21):
            last, j = _jobs.create_job(last, jobs, job_type="scout", owner="t")
            jobs.append(j)
            created += 1
    assert created <= 20


def test_budgets_enforced_total_jobs() -> None:
    """Unlimited by default: no max_total_jobs ceiling until policy configures one."""
    s = _session.create_session("q?", "o", as_of=ASOF)
    jobs: list[Job] = []
    last = s
    for _ in range(21):
        last, j = _jobs.create_job(last, jobs, job_type="scout", owner="t")
        jobs.append(_jobs.complete_job(_jobs.start_job(j)))
    assert len(jobs) == 21  # no configured cap: creation never raises
    policy: dict[str, JSONValue] = dict(last.policy)
    raw: object = policy.get("research", {})
    assert isinstance(raw, dict)
    section: dict[str, JSONValue] = dict(raw)
    section["max_total_jobs"] = 21
    policy["research"] = section
    capped = dataclasses.replace(last, policy=policy)
    with pytest.raises(ValueError, match="max_total_jobs"):
        _jobs.create_job(capped, jobs, job_type="scout", owner="t")


def test_scout_cannot_have_children() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    _, parent = _jobs.create_job(s, [], job_type="scout", owner="t")
    with pytest.raises(ValueError):
        _jobs.create_job(s, [parent], job_type="scout", owner="t", parent_job_id=parent.job_id)


def test_journal_sequences_and_pit_rejection_logged() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    e1 = append_event(s.session_id, "research.created", "t", "t", {})
    e2 = append_event(s.session_id, "research.started", "t", "t", {})
    assert e2.sequence == e1.sequence + 1
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-X", s.session_id, 1, datetime(2025, 7, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen and seen[0][0] == "evidence.rejected"
    assert seen[0][1]["reason"] == "PIT_VIOLATION"


def test_ledger_append_only_and_supersede() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    ingest_evidence(led, _ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)), as_of=ASOF, on_reject=None)
    with pytest.raises(ValueError):
        led.append(_ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)))
    fix = led.get("EV-1")
    led.supersede(dataclasses.replace(fix, evidence_id="EV-2", superseded_by="EV-1"))
    assert led.get("EV-1").evidence_id == "EV-1"


def test_freeze_immutable_and_cumulative_e2() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    for eid, wave in (("EV-1", 1), ("EV-2", 1), ("EV-3", 2)):
        ingest_evidence(led, _ev(eid, s.session_id, wave, datetime(2025, 5, 1, tzinfo=UTC)), as_of=ASOF, on_reject=None)
    e1 = [e for e in led.list_session(s.session_id) if e.wave_id <= 1]
    e2 = [e for e in led.list_session(s.session_id) if e.wave_id <= 2]
    f1 = create_freeze(freeze_id="E1", session_id=s.session_id, wave_id=1, records=e1, as_of=ASOF)
    f2 = create_freeze(freeze_id="E2", session_id=s.session_id, wave_id=2, records=e2, as_of=ASOF)
    assert set(f1.evidence_ids) < set(f2.evidence_ids)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _frozen_write(f2, "evidence_ids", ("EV-1",))
    verify_freeze(f2, e2)
    with pytest.raises(FreezeIntegrityError):
        verify_freeze(f2, e1)


def test_dossier_rejects_dangling_refs() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    cov: dict[str, object] = {
        "entities": ["NVDA"],
        "forms": ["10-Q"],
        "time_range": {"start": "2025-01-01", "end": "2025-06-30"},
        "sources_examined": ["e"],
        "complete": True,
        "exclusions": [],
    }
    bad = create_dossier(
        dossier_id="SEC-D1",
        session_id=s.session_id,
        wave_id=1,
        subject="NVDA",
        coverage=cov,
        findings=[{"text": "dangling", "evidence_ids": ["EV-NOPE"]}],
        unknowns=[],
        limitations=[],
        open_questions=[],
    )
    with pytest.raises(DossierIntegrityError):
        validate_dossier(bad, set())


DispatchFn = Callable[[str, dict[str, object]], dict[str, object]]
ModelFn = Callable[[str], str]


def _fake_document(index: int, args: Mapping[str, object]) -> dict[str, object]:
    """One raw filing document result (accession + document_name + passage): the only evidence shape."""
    raw_accession = args.get("accession_no")
    accession = (
        raw_accession.strip()
        if isinstance(raw_accession, str) and raw_accession.strip()
        else f"0000320193-25-{index:06d}"
    )
    raw_document = args.get("document_name")
    document = (
        raw_document.strip() if isinstance(raw_document, str) and raw_document.strip() else f"nvda-20250331-{index}.htm"
    )
    passage = "Data Center revenue grew year over year on accelerated computing demand."
    return {
        "content": passage,
        "text": passage,
        "accession_no": accession,
        "document_name": document,
        "matching_passage": passage,
        "source_handle": seam.handle_for(passage, accession=accession, document=document),
        "known_at": "2025-05-01",
        "uri": f"https://www.sec.gov/Archives/edgar/data/1045810/{accession.replace('-', '')}/{document}",
    }


def _fake_dispatch() -> DispatchFn:
    """SEC fake: searches return navigation packets (top_hits), document opens return raw passages.

    Only ``get_sec_document``/``get_sec_filing`` results carrying an accession, a
    document name, and a passage become evidence; every other result is navigation.
    """
    counter = itertools.count(1)

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["search_sec_filings", "list_sec_filings", "get_sec_document", "get_sec_filing"]}
        if name == "call_tool":
            n = next(counter)
            inner = str(args.get("name") or "")
            inner_args = args.get("arguments")
            requested: dict[str, object] = inner_args if isinstance(inner_args, dict) else {}
            if inner in ("get_sec_document", "get_sec_filing"):
                return _fake_document(n, requested)
            return {
                "search_id": f"s{n}",
                "query": str(requested.get("query") or ""),
                "count": 1,
                "top_hits": [
                    {"accession": f"0000320193-25-{n:06d}", "document": f"nvda-20250331-{n}.htm", "form": "10-Q"}
                ],
            }
        return {}

    return _dispatch


def _run_wave(repo: ResearchRepository, model: ModelFn) -> dict[str, object]:
    return run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=model,
        repo=repo,
        budgets=None,
    )


def test_same_freeze_parallel_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _model(prompt: str) -> str:
        time.sleep(2)
        assert prompt
        return _grounded(prompt)

    out = _run_wave(repo, _model)
    assert out["stop_reason"] == "complete:wave1"
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    assert stock.evidence_ids == bull.evidence_ids == bear.evidence_ids
    sid_obj = out["session_id"]
    assert isinstance(sid_obj, str)
    jobs = repo.list_jobs(sid_obj)
    assert all(j.status == "completed" for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot"))


def test_parallel_beats_serial_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    spans: list[tuple[float, float]] = []

    def _slow(prompt: str) -> str:
        assert prompt
        start = time.monotonic()
        time.sleep(1)
        spans.append((start, time.monotonic()))
        return _grounded(prompt)

    _run_wave(repo, _slow)
    committee = spans[3:6]  # first three calls are serial scouts
    assert len(committee) == 3
    assert max(s for s, _ in committee) < min(e for _, e in committee)


def test_resume_at_source_freeze_one_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    for hook, reason in (
        ("source", "interrupted:source"),
        ("freeze", "interrupted:freeze"),
        ("one-committee", "interrupted:one-committee"),
    ):
        out = run_live(
            question="NVDA demand?",
            objective="o",
            as_of="2025-06-30T00:00:00+00:00",
            tickers=["NVDA"],
            dispatch=_fake_dispatch(),
            model=_ok,
            repo=repo,
            budgets=None,
            interrupt_after=hook,
        )
        assert out["stop_reason"] == reason
        resume_sid = out["session_id"]
        assert isinstance(resume_sid, str)
        n_jobs = len(repo.list_jobs(resume_sid))
        state = repo.resume(resume_sid)
        assert len(repo.list_jobs(resume_sid)) == n_jobs  # no dup on resume
        assert state.session is not None


def test_committee_prompts_carry_identical_freeze_payload() -> None:
    prompts: list[str] = []

    def _cap(prompt: str) -> str:
        prompts.append(prompt)
        return _grounded(prompt)

    s = _session.create_session("NVDA demand?", "o", as_of=ASOF)
    led = EvidenceLedger()
    ingest_evidence(led, _ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)), as_of=ASOF, on_reject=None)
    rec = led.get("EV-1")
    payload = f"[{rec.evidence_id}] {rec.subject} | claim: {rec.claim_text}\ncontent: {rec.content}"
    from app.research.agents.bearbot import run_bearbot
    from app.research.agents.bullbot import run_bullbot
    from app.research.agents.stockbot import run_stockbot

    run_stockbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    run_bullbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    run_bearbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    assert len(prompts) == 3
    bodies = [p.split("Evidence (cite ids; do not invent):\n", 1)[1] for p in prompts]
    assert bodies[0] == bodies[1] == bodies[2]
    assert "EV-1" in bodies[0] and "c-ev-1" in bodies[0].lower()


def test_source_timestamp_never_invented_and_post_cutoff_rejected() -> None:
    from app.research.runner import _extract_known_at, _extract_source_ref

    assert _extract_known_at({}) is None
    assert _extract_known_at({"record": {}}) is None
    assert _extract_source_ref({}) == (None, None)
    assert _extract_source_ref({"source": "SEC EDGAR"}) == (None, None)
    known = _extract_known_at(
        {
            "meta": {
                "source_refs": {
                    "record_id": "0001045810-25-000023",
                    "uri": "https://sec.gov/x",
                    "known_at": "2025-05-28",
                }
            }
        }
    )
    assert known is not None and (known.year, known.month, known.day) == (2025, 5, 28)
    uri, ref = _extract_source_ref(
        {"meta": {"source_refs": {"record_id": "0001045810-25-000023", "uri": "https://sec.gov/x"}}}
    )
    assert (uri, ref) == ("https://sec.gov/x", "0001045810-25-000023")
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-FUT", s.session_id, 1, datetime(2026, 1, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen[0][1]["reason"] == "PIT_VIOLATION"
    assert led.ids() == ()


def test_resume_after_source_completes_reusing_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    assert out["stop_reason"] == "interrupted:source"
    sid = out["session_id"]
    assert isinstance(sid, str)
    eids_raw = out["evidence_ids"]
    assert isinstance(eids_raw, list) and all(isinstance(e, str) for e in eids_raw)
    eids_before: list[str] = list(eids_raw)
    did_before = out["dossier_id"]
    ev_before = len(repo.list_evidence(sid))
    jobs_before = repo.list_jobs(sid)
    assert (
        len(jobs_before) == 4
        and sum(1 for j in jobs_before if j.job_type == "source_agent") == 1
        and sum(1 for j in jobs_before if j.job_type == "scout") == 3
    )
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    eids_after = out2["evidence_ids"]
    assert isinstance(eids_after, list)
    # The resumed run reuses exactly the persisted wave evidence: every dossier-cited id
    # is in it, it never advertises a navigation artifact, and nothing new was fetched.
    persisted = [dict(r) for r in repo.list_evidence(sid)]
    evidence_rows = {str(r.get("evidence_id")) for r in persisted if r.get("record_kind") == "evidence"}
    discovery_rows = {str(r.get("evidence_id")) for r in persisted if r.get("record_kind") == "discovery"}
    assert set(eids_before) <= set(eids_after) == evidence_rows
    assert discovery_rows and not (set(eids_after) & discovery_rows)
    assert len(repo.list_evidence(sid)) == ev_before  # no dup evidence
    assert out2["dossier_id"] == did_before  # dossier reused, not recreated
    jobs_after = repo.list_jobs(sid)
    assert len([j for j in jobs_after if j.job_type == "source_agent"]) == 1  # no dup source job
    assert len([j for j in jobs_after if j.job_type == "scout"]) == 3  # completed scouts reused, no dup scouts
    assert len(jobs_after) == len(jobs_before) + 3  # trio added once
    assert out2["freeze_id"] == f"{sid}:1:freeze"
    assert repo.get_session(sid).status == "completed"


def test_resume_after_freeze_skips_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="freeze",
    )
    assert out["stop_reason"] == "interrupted:freeze"
    sid = out["session_id"]
    assert isinstance(sid, str)
    fid = out["freeze_id"]
    assert isinstance(fid, str) and fid
    calls = [0]
    base = _fake_dispatch()

    def _counting(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "call_tool":
            calls[0] += 1
        return base(name, args)

    out2 = resume_live(sid, _counting, _ok, repo=repo)
    assert calls[0] == 0  # fetch skipped: no tool evidence calls
    assert out2["stop_reason"] == "complete:wave1"
    assert out2["freeze_id"] == fid  # same freeze reused
    assert out2["stock"] is not None and out2["bull"] is not None and out2["bear"] is not None


def test_resume_after_one_committee_reruns_trio_on_same_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="one-committee",
    )
    assert out["stop_reason"] == "interrupted:one-committee"
    sid = out["session_id"]
    assert isinstance(sid, str)
    fid = out["freeze_id"]
    jobs_before = repo.list_jobs(sid)
    assert len(jobs_before) == 5  # source + 3 scouts + stock-only
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    # Analyses are not persisted, so resume always reruns the full trio on the
    # same freeze id with new jobs (never reuses the stock-only partial result).
    assert out2["freeze_id"] == fid
    assert out2["stock"] is not None and out2["bull"] is not None and out2["bear"] is not None
    assert out2["stop_reason"] == "complete:wave1"
    jobs_after = repo.list_jobs(sid)
    assert len(jobs_after) == len(jobs_before) + 3
    sess = repo.get_session(sid)
    assert len(sess.committee_runs) == 2
    second = sess.committee_runs[1]
    assert isinstance(second, dict) and second.get("freeze_id") == fid
    second_jobs = second.get("jobs")
    assert isinstance(second_jobs, list) and len(second_jobs) == 3


def test_resume_dossier_resource_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.resources import read_resource

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    did = out["dossier_id"]
    assert isinstance(did, str) and did
    stores = repo.resource_stores(sid)
    resolved = read_resource(f"dossier://{did}", dossiers=stores["dossier"])
    assert isinstance(resolved, dict) and resolved.get("dossier_id") == did
    resolved_full = read_resource(
        f"dossier://{did}",
        evidence=stores["evidence"],
        freezes=stores["freeze"],
        dossiers=stores["dossier"],
        jobs=stores["job"],
        sessions=stores["research"],
    )
    assert resolved_full == resolved


def test_resume_failed_session_reports_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from app.research.runner import LiveModelError, resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _timeout_model(_prompt: str) -> str:
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1)

    with pytest.raises(LiveModelError) as excinfo:
        run_live("timeout probe?", "probe", None, ["NVDA"], lambda n, a: {}, _timeout_model, repo=repo)
    sid = excinfo.value.session_id
    assert repo.get_session(sid).status == "failed"
    n_jobs = len(repo.list_jobs(sid))
    n_ev = len(repo.list_evidence(sid))

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    with pytest.raises(LiveModelError, match="out of scope"):
        resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert len(repo.list_jobs(sid)) == n_jobs  # failed resume writes nothing
    assert len(repo.list_evidence(sid)) == n_ev


def test_scouts_run_serially_with_ordered_results() -> None:
    from app.research.agents.scout import run_scout
    from app.research.agents.sec_agent import run_sec_assignment

    seen: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name
        assert args is not None
        return {}

    def _model(prompt: str) -> str:
        assert prompt
        seen.append(prompt[:40])
        return "[]"

    dossier = run_sec_assignment(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        as_of="2025-06-30",
        tickers=["NVDA"],
        dispatch=_dispatch,
        model=_model,
        spawn=lambda a: run_scout(a, dispatch=_dispatch, model=_model),
    )
    assert len(seen) == 3  # filings, financials, risk in assignment order
    assert isinstance(dossier, object) and dossier is not None


def test_pit_unverified_historical_rejected_current_accepted() -> None:
    from datetime import datetime

    from app.research.evidence import (
        Evidence,
        EvidenceLedger,
        EvidenceRejectedError,
        evidence_content_hash,
        ingest_evidence,
    )

    hist = datetime(2025, 6, 30, tzinfo=UTC)

    def _mk(eid: str, known: datetime | None) -> Evidence:
        content = "content"
        return Evidence(
            eid,
            "rs:pit",
            1,
            "sec",
            "search_sec_filings",
            "s",
            "claim",
            content,
            evidence_content_hash(content),
            datetime(2025, 5, 1, tzinfo=UTC),
            "https://sec.gov/x",
            "r1",
            None,
            known,
            None,
            "job:1",
            "sec_scout",
            (),
            (),
            None,
            None,
            {},
            None,
        )

    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError) as ei:
        ingest_evidence(led, _mk("EV-U1", None), as_of=hist, on_reject=lambda t, p: seen.append((t, p)))
    assert ei.value.reason == "PIT_UNVERIFIED"
    assert seen[0][1]["reason"] == "PIT_UNVERIFIED"
    assert led.ids() == ()
    out = ingest_evidence(led, _mk("EV-U2", None), as_of=None)
    assert out.evidence_id == "EV-U2"
    from app.research.agents.scout import _is_pit_eligible

    assert _is_pit_eligible(None, "2025-06-30") is False
    assert _is_pit_eligible(None, "unbounded") is True


def test_budget_resume_hydrates_cumulative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    assert out["stop_reason"] == "interrupted:source"
    sid = out["session_id"]
    assert isinstance(sid, str)
    sess = repo.get_session(sid)
    used = sess.budget.get("tool_calls_used")
    assert isinstance(used, int) and used > 0
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    sess2 = repo.get_session(sid)
    used2 = sess2.budget.get("tool_calls_used")
    assert isinstance(used2, int) and used2 == used  # no reset to zero on resume


def test_two_wave_e2_tree_and_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A material wave-1 follow-up drives a second wave on cumulative evidence; an explicit
    configured max_waves (never an architectural ceiling) then settles the run at complete:wave2."""
    from app.research.director import DirectorBudgets

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _model(prompt: str) -> str:
        assert prompt
        return _grounded(prompt, ["What drove the Q2 delta?"])

    out = run_live(
        question="NVDA historical demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_model,
        repo=repo,
        budgets=DirectorBudgets(max_waves=2),
    )
    assert out["stop_reason"] == "complete:wave2"
    assert str(out["wave_decision"]).startswith("max_waves")  # explicit int honored, waves are sequence numbers
    sid = out["session_id"]
    assert isinstance(sid, str)
    jobs = repo.list_jobs(sid)
    src_jobs = [j for j in jobs if j.job_type == "source_agent"]
    scout_jobs = [j for j in jobs if j.job_type == "scout"]
    committee_jobs = [j for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot")]
    assert len(src_jobs) == 2 and len(scout_jobs) == 6 and len(committee_jobs) == 6
    src_ids = {j.job_id for j in src_jobs}
    assert all(j.parent_job_id in src_ids for j in scout_jobs)
    assert all(j.status == "completed" for j in jobs)
    assert len(jobs) == 14
    # The committee's wave-1 request (never evidence) drove the targeted wave-2 source job.
    wave2_src = src_jobs[-1]
    assert (wave2_src.diagnostics or {}).get("question") == "What drove the Q2 delta?"
    sess = repo.get_session(sid)
    assert len(sess.freeze_ids) >= 2
    raw1: object = repo.get_freeze(sess.freeze_ids[0]).get("evidence_ids", [])
    raw2: object = repo.get_freeze(sess.freeze_ids[-1]).get("evidence_ids", [])
    e1 = {e for e in raw1 if isinstance(e, str)} if isinstance(raw1, list) else set()
    e2 = {e for e in raw2 if isinstance(e, str)} if isinstance(raw2, list) else set()
    assert e1 and e2 and e1 < e2  # cumulative E2 superset of E1


def test_grounded_claims_reject_unknown_and_empty() -> None:
    import json as _json

    from app.research.agents import ModelOutputFailure, parse_grounded_claims

    claims = parse_grounded_claims(
        _json.dumps([{"text": "revenue grew", "evidence_ids": ["EV-1"]}]), frozen=["EV-1", "EV-2"]
    )
    assert [c.text for c in claims] and claims[0].evidence_ids == ["EV-1"]
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(_json.dumps([{"text": "invented trend", "evidence_ids": ["EV-999"]}]), frozen=["EV-1"])
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            _json.dumps([{"text": "Revenue grew with no citation.", "evidence_ids": []}]), frozen=["EV-1"]
        )


def test_committee_refs_derive_from_claims_not_whole_freeze() -> None:
    from app.research.agents import claims_refs
    from app.research.agents.stockbot import run_stockbot

    analysis = run_stockbot(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1", "EV-2"],
        as_of="x",
        model=lambda prompt: _committee_output([{"text": "only first matters", "evidence_ids": ["EV-1"]}]),
        evidence_text="[EV-1] a\n[EV-2] b",
    )
    assert claims_refs(analysis.claims) == ["EV-1"]
    assert [c.evidence_ids for c in analysis.claims] == [["EV-1"]]


def test_committee_unknown_id_fails_model_output() -> None:
    from app.research.agents import ModelOutputFailure
    from app.research.agents.bearbot import run_bearbot
    from app.research.agents.bullbot import run_bullbot

    with pytest.raises(ModelOutputFailure, match="EV-999"):
        run_bullbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: _committee_output([{"text": "bad", "evidence_ids": ["EV-999"]}]),
            evidence_text="[EV-1] a",
        )
    with pytest.raises(ModelOutputFailure, match="uncited"):
        run_bearbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: _committee_output([{"text": "Bearish with no citation.", "evidence_ids": []}]),
            evidence_text="[EV-1] a",
        )


def test_scout_findings_cite_only_acquired_ids() -> None:
    """Scout boundary: only acquired ids ground; a bad record is dropped and reported, never fatal."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    assignment = ScoutAssignment(
        assignment_id="scout-filings", session_id="rs:t", as_of="2025-06-30", role="filings", question="Q?", tickers=[]
    )

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name and args is not None
        if name == "browse_tools":
            return {}
        return {"evidence_ids": [{"evidence_id": "EV-1", "known_at": "2025-05-01"}]}

    import json as _json

    good = run_scout(
        assignment, dispatch=_dispatch, model=lambda prompt: _json.dumps([{"text": "found", "evidence_ids": ["EV-1"]}])
    )
    assert [c.evidence_ids for c in good.findings] == [["EV-1"]]
    # A fabricated id loses its own claim; the grounded sibling survives with a limitation.
    mixed = run_scout(
        assignment,
        dispatch=_dispatch,
        model=lambda prompt: _json.dumps(
            [{"text": "found", "evidence_ids": ["EV-1"]}, {"text": "bad", "evidence_ids": ["EV-999"]}]
        ),
    )
    assert [c.text for c in mixed.findings] == ["found"]
    assert any("dropped" in line for line in mixed.limitations)
    # An uncited non-unknown claim is dropped too: nothing ungrounded reaches the dossier.
    uncited = run_scout(
        assignment,
        dispatch=_dispatch,
        model=lambda prompt: _json.dumps([{"text": "Something factual uncited.", "evidence_ids": []}]),
    )
    assert uncited.findings == []
    assert any("dropped" in line for line in uncited.limitations)


def test_dossier_preserves_per_claim_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    out = _run_wave(repo, _grounded)
    assert out["stop_reason"] == "complete:wave1"
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    from app.research.agents import claims_refs as _crefs

    for analysis in (stock, bull, bear):
        frozen = set(analysis.evidence_ids)
        for claim in analysis.claims:
            assert claim.evidence_ids and set(claim.evidence_ids) <= frozen
        assert set(_crefs(analysis.claims)) <= frozen
    from app.research.synthesis.committee import CommitteeDisagreement
    from app.research.synthesis.final import synthesize_final

    sid = out["session_id"]
    fid = out["freeze_id"]
    eids = out["evidence_ids"]
    disagreement = out["disagreement"]
    assert isinstance(sid, str) and isinstance(fid, str)
    assert isinstance(eids, list) and isinstance(disagreement, CommitteeDisagreement)
    synth = synthesize_final(
        "NVDA demand?",
        session_id=sid,
        wave_id=1,
        freeze_id=fid,
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
    )
    from app.research.agents import claims_refs as _crefs2

    frozen_raw: object = repo.get_freeze(fid).get("evidence_ids", [])
    frozen_ids = {e for e in frozen_raw if isinstance(e, str)} if isinstance(frozen_raw, list) else set()
    assert frozen_ids  # the freeze is the committee's evidence universe
    for claim in synth.claims:
        assert claim.evidence_ids and set(claim.evidence_ids) <= frozen_ids
        # Per-claim mapping survives synthesis: same text -> exactly the trio's cited ids.
        expected = {
            eid
            for analysis in (stock, bull, bear)
            for c in analysis.claims
            if c.text == claim.text
            for eid in c.evidence_ids
        }
        assert set(claim.evidence_ids) == expected
    # Refs derive from the trio's per-claim mapping, never the whole freeze.
    assert set(_crefs2(synth.claims)) == set(_crefs(stock.claims)) | set(_crefs(bull.claims)) | set(_crefs(bear.claims))


def test_resume_partial_source_reuses_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    from app.research.repository import get_research_db_path
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    base = _fake_dispatch()
    scouts_started = {"n": 0}

    def _crash(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            scouts_started["n"] += 1
            # Deterministic cut: each scout opens with browse_tools, so the third one
            # starting means two completed and this one is mid-fetch (keying on a raw
            # call count would drift with the scout's tool volume).
            if scouts_started["n"] == 3:
                raise KeyboardInterrupt("simulated crash mid-source-fetch")
        return base(name, args)

    def _ok(prompt: str) -> str:
        return _grounded(prompt)

    import pytest as _pt

    with _pt.raises(KeyboardInterrupt):
        run_live(
            question="NVDA demand?",
            objective="o",
            as_of="2025-06-30T00:00:00+00:00",
            tickers=["NVDA"],
            dispatch=_crash,
            model=_ok,
            repo=repo,
            budgets=None,
        )
    db = get_research_db_path()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT DISTINCT session_id FROM jobs").fetchall()
    assert len(rows) == 1
    sid = str(rows[0][0])
    jobs_before = repo.list_jobs(sid)
    src_before = [j for j in jobs_before if j.job_type == "source_agent"]
    scout_before = [j for j in jobs_before if j.job_type == "scout"]
    assert len(src_before) == 1 and src_before[0].status == "running"
    assert len([j for j in scout_before if j.status == "completed"]) == 2
    out = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out["stop_reason"] == "complete:wave1"
    jobs_after = repo.list_jobs(sid)
    assert len([j for j in jobs_after if j.job_type == "source_agent"]) == 1
    completed = [j for j in jobs_after if j.job_type == "scout" and j.status == "completed"]
    aids = [(j.diagnostics or {}).get("assignment_id") for j in completed]
    assert sorted(a for a in aids if isinstance(a, str)) == [
        "scout-filings",
        "scout-financials",
        "scout-risk",
    ]
    eids_final = [str(r.get("evidence_id")) for r in repo.list_evidence(sid)]
    assert len(eids_final) == len(set(eids_final))


def test_run_resume_produce_complete_append_only_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.evals.traces import get_trace_events, list_traces
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    traces = list_traces(sid)
    assert len(traces) == 1
    tid = traces[0].trace_id
    evs = get_trace_events(tid)
    types = [e.event_type for e in evs]
    for required in (
        "trace.opened",
        "tool.completed",
        "evidence.ingested",
        "discovery.ingested",
        "job.created",
        "job.completed",
        "model.completed",
    ):
        assert required in types
    seqs = [e.seq for e in evs]
    assert len(set(seqs)) == len(seqs) and seqs == sorted(seqs)
    tools = [e for e in evs if e.event_type == "tool.completed"]
    assert tools and all(isinstance(e.payload.get("tool"), str) and e.payload.get("tool") for e in tools)
    # Every completion carries its args plus exactly one kind marker: the evidence id
    # (raw document opened) or record_kind=discovery (navigation artifact, never citable).
    assert all("args" in e.payload for e in tools)
    assert all(("evidence_id" in e.payload) != (e.payload.get("record_kind") == "discovery") for e in tools)
    assert any("evidence_id" in e.payload for e in tools)
    assert any(e.payload.get("record_kind") == "discovery" for e in tools)
    models = [e for e in evs if e.event_type == "model.completed"]
    assert models and all("prompt" in e.payload and "output" in e.payload for e in models)
    discs = [e for e in evs if e.event_type == "discovery.completed"]
    assert discs and all("args" in e.payload and "matches" in e.payload for e in discs)
    n_before = len(evs)
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    assert list_traces(sid)[0].trace_id == tid
    evs2 = get_trace_events(tid)
    assert len(evs2) > n_before
    seqs2 = [e.seq for e in evs2]
    assert len(set(seqs2)) == len(seqs2) and seqs2 == sorted(seqs2)
    all_types = [e.event_type for e in evs2]
    assert "trace.resumed" in all_types
    resumed = next(e for e in evs2 if e.event_type == "trace.resumed")
    assert resumed.payload.get("trace_id") == tid and resumed.payload.get("session_id") == sid
    from app.research.evals.traces import get_trace

    header = get_trace(tid)
    assert header is not None and header.status == "completed" and header.conclusion


def _svc_sid(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jobs = repo.list_jobs(sid)
    return sid, jobs[0].job_id


@pytest.fixture(autouse=True)
def _evidence_handle_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence admission in this module reloads through the fake archive, never the network."""
    seam.install(monkeypatch)


def _svc_item(
    eid: str,
    wave: int = 1,
    passage: str = "Accelerated computing demand grew in the reporting period.",
    **over: object,
) -> dict[str, object]:
    """Observed-fact item with the canonical handle for its cited passage (overrides win)."""
    base: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": "c-" + eid,
        "subject": "NVDA",
        "source_name": "SEC",
        "source_uri": "https://sec.gov/x",
        "source_record_id": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
        "matching_passage": passage,
        "source_handle": seam.handle_for(passage),
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    if "passage" in over and "source_handle" not in over and "matching_passage" not in over:
        text = str(over.pop("passage"))
        base["matching_passage"] = text
        base["source_handle"] = seam.handle_for(text)
    base.update(over)
    return base


def _committee_analysis(
    eid: str, follow_ups: Sequence[object] = (), claim_text: str = "finding", claim_type: str = "inference"
) -> dict[str, object]:
    """Rich committee envelope mapping for service.record_committee_analysis (all required keys)."""
    return {
        "executive_view": f"{claim_text} read over the freeze.",
        "claims": [{"text": claim_text, "claim_type": claim_type, "evidence_ids": [eid]}],
        "impact_channels": [{"text": "exposure channel", "direction": "pressure", "evidence_ids": [eid]}],
        "materiality": {"overall": "medium", "reasoning": "read off the freeze"},
        "uncertainties": ["open terms"],
        "what_would_change": ["a new filing disclosing the terms"],
        "follow_ups": list(follow_ups),
    }


def _svc_ana(eid: str) -> dict[str, object]:
    return _committee_analysis(eid)


def _svc_trio(repo: ResearchRepository, sid: str, eid: str) -> None:
    from app.research import service as _svc

    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)


def test_committee_cannot_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    out = execute_agent_tool("search_web", {"query": "x", "session_id": sid}, RuntimeToolSession(session_id="t1"))
    assert "forbids" in str(out.get("error", ""))


def test_committee_cannot_add_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bear = str(_svc.start_job(sid, "bearbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError):
        _svc.record_evidence(sid, bear, _svc_item(f"{sid}:ev:2"), repo=repo)
    with pytest.raises(ValueError, match="forbids"):
        _svc.authorize_and_consume_dispatch(sid, bear, "search_web", repo=repo)


def test_role_job_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    stock = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="!="):
        _svc.record_committee_analysis(sid, stock, "bearbot", _svc_ana(eid), repo=repo)


def test_evidence_rejected_on_completed_or_committee_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    with pytest.raises(ValueError, match="running"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:9"), repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bull = str(_svc.start_job(sid, "bullbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError):
        _svc.record_evidence(sid, bull, _svc_item(f"{sid}:ev:3"), repo=repo)


def test_restart_after_e1_resume_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    fresh = ResearchRepository()
    assert _svc.resume_research(sid, repo=fresh)["session"] is not None
    _svc_trio(fresh, sid, eid)
    _svc.decide_next_wave(sid, repo=fresh)
    out = _svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=fresh)
    assert out["freeze_id"] == f"{sid}:1:freeze"


def test_restart_after_2_of_3_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    ids = [str(_svc.start_job(sid, r, repo=repo, wave_id=1)["job_id"]) for r in ("stockbot", "bullbot", "bearbot")]
    for jid, role in zip(ids[:2], ("stockbot", "bullbot")):
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)
    fresh = ResearchRepository()
    _svc.record_committee_analysis(sid, ids[2], "bearbot", _svc_ana(eid), repo=fresh)
    out = _svc.finalize_session(sid, "answer", [{"text": "finding", "evidence_ids": [eid]}], repo=fresh)
    assert out["freeze_id"] == f"{sid}:1:freeze"


def test_source_terminal_before_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    with pytest.raises(ValueError, match="still open"):
        _svc.freeze_session(sid, 1, repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == f"{sid}:1:freeze"


def test_budget_stops_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(jid), tool_budget=1))
    ctx = RuntimeToolSession(session_id="t-budget")
    ctx.active_research_session_id = sid
    ctx.active_research_job_id = jid
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        calls.append((name, dict(arguments)))
        return {"result_type": "web_search", "query": arguments.get("query"), "results": [], "source": "exa"}

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    first = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    assert "error" not in first, first
    second = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    # §3: explicit per-job kernel exhaustion passes through verbatim (no budget_exhausted collapse).
    assert "tool_budget exhausted" in str(second.get("error", "")), second
    assert second.get("error_type") != "budget_exhausted", second
    assert len(calls) == 1
    assert calls[0][0] == "search_web" and calls[0][1] == {"query": "NVDA demand"}
    assert repo.get_job(jid).tool_budget == 0
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1
    kept = execute_agent_tool(
        "research_add_evidence", {"session_id": sid, "job_id": jid, "item": _svc_item(f"{sid}:ev:1")}, ctx
    )
    assert "error" not in kept, kept
    assert repo.get_job(jid).tool_budget == 0
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1


def test_research_bound_dispatch_carries_research_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC discovery sees the attached research session (exhaustive default), None when unattached."""
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    seen: dict[str, object] = {}

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        seen["context_session"] = getattr(context, "research_session_id", "missing")
        return {"search_id": "s1", "count": 0, "source": "SEC EDGAR"}

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    bound = RuntimeToolSession(
        session_id="pi-bound",
        active_research_session_id=sid,
        active_research_job_id=jid,
    )
    out = execute_agent_tool("search_sec_filings", {"query": "Apple"}, bound)
    assert "error" not in out, out
    assert seen["context_session"] == sid

    seen.clear()
    unbound = RuntimeToolSession(session_id="pi-unbound")
    out = execute_agent_tool("search_sec_filings", {"query": "Apple"}, unbound)
    assert "error" not in out, out
    assert seen["context_session"] is None


def test_concurrent_dispatch_race_admits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(jid), tool_budget=1))
    sess = repo.get_session(sid)
    policy = dict(sess.policy)
    raw: object = policy.get("research", {})
    assert isinstance(raw, dict)
    section: dict[str, JSONValue] = dict(raw)
    section["max_tool_calls"] = 1
    policy["research"] = section
    repo.save_session(dataclasses.replace(sess, policy=policy))
    barrier = threading.Barrier(2)
    outcomes: list[object] = [None, None]

    def _worker(idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            outcomes[idx] = _svc.authorize_and_consume_dispatch(sid, jid, "search_web", repo=repo)
        except Exception as exc:  # noqa: BLE001 — race outcome is the assertion
            outcomes[idx] = exc

    threads = [threading.Thread(target=_worker, args=(i,)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(thread.is_alive() is False for thread in threads)
    successes = [o for o in outcomes if isinstance(o, dict)]
    failures = [o for o in outcomes if isinstance(o, Exception)]
    assert len(successes) == 1, outcomes
    assert len(failures) == 1, outcomes
    assert "budget exhausted" in str(failures[0]).lower(), outcomes
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1
    assert repo.get_job(jid).tool_budget == 0
    assert [j.job_id for j in repo.list_jobs(sid)] == [jid]


def test_stage_allows_common_read_controls() -> None:
    from app.research.stage import check_stage_tool

    for stage in ("SOURCE_RESEARCH", "COMMITTEE", "FINAL"):
        check_stage_tool(stage, "research_read")
        check_stage_tool(stage, "research_cancel")
    for stage in ("COMMITTEE", "FINAL"):
        with pytest.raises(ValueError, match="forbids"):
            check_stage_tool(stage, "research_add_evidence")
        with pytest.raises(ValueError, match="forbids"):
            check_stage_tool(stage, "search_web")


def test_queued_scout_blocks_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    found = repo.get_session(sid)
    existing = repo.list_jobs(sid)
    updated, scout = _jobs.create_job(found, existing, job_type="scout", owner="service", wave_id=1)
    repo.save_session(updated)
    repo.save_job(scout)
    with pytest.raises(ValueError, match="still open") as exc:
        _svc.freeze_session(sid, 1, repo=repo)
    assert scout.job_id in str(exc.value)
    assert repo.get_job(scout.job_id).status == "queued"


def test_start_job_rejects_malformed_budget_wave_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    for bad in ("2", 2.0, True, None):
        with pytest.raises(ValueError, match="wave_id"):
            _svc.start_job(sid, "source_agent", budget={"wave_id": bad}, repo=repo)
    ok = _svc.start_job(sid, "source_agent", budget={"wave_id": 1}, repo=repo)
    assert ok["job_id"]


def test_job_runtime_identity_merges_partial_then_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """attach_job_runtime accumulates accepted keys in diagnostics across partial calls."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _sid, src = _svc_sid(repo)
    partial = _svc.attach_job_runtime(
        src, {"runtime": "omp", "runtime_agent_id": "agent-1", "not_accepted": "x"}, repo=repo
    )
    assert partial["diagnostics"] == {"runtime": "omp", "runtime_agent_id": "agent-1"}
    done = _svc.attach_job_runtime(
        src,
        {
            "runtime_agent_id": "",
            "runtime_parent_agent_id": "agent-0",
            "runtime_task_call_id": "call-7",
            "runtime_agent_type": "scout",
            "runtime_session_file": "/tmp/session.jsonl",
        },
        repo=repo,
    )
    assert done["diagnostics"] == {
        "runtime": "omp",
        "runtime_agent_id": "agent-1",
        "runtime_parent_agent_id": "agent-0",
        "runtime_task_call_id": "call-7",
        "runtime_agent_type": "scout",
        "runtime_session_file": "/tmp/session.jsonl",
    }
    assert ResearchRepository().get_job(src).diagnostics == done["diagnostics"]
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.attach_job_runtime("nope", {"runtime": "omp"}, repo=repo)


def test_job_fail_cancel_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fail_job marks failed with category/message; cancel_job marks cancelled; terminal is a no-op."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _, src = _svc_sid(repo)
    with pytest.raises(ValueError):
        _svc.fail_job(src, "bogus-category", "msg", repo=repo)
    assert repo.get_job(src).status == "running"
    failed = _svc.fail_job(src, "timeout", "deadline hit", repo=repo)
    assert failed["status"] == "failed"
    failure = failed["failure"]
    assert isinstance(failure, dict)
    assert failure["category"] == "timeout"
    assert failure["message"] == "deadline hit"
    assert repo.get_job(src).status == "failed"
    assert _svc.cancel_job(src, repo=repo)["status"] == "failed"
    assert _svc.fail_job(src, "timeout", "again", repo=repo)["status"] == "failed"
    _, src2 = _svc_sid(repo)
    cancelled = _svc.cancel_job(src2, repo=repo)
    assert cancelled["status"] == "cancelled"
    assert repo.get_job(src2).status == "cancelled"
    assert _svc.fail_job(src2, "timeout", "late", repo=repo)["status"] == "cancelled"
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.fail_job("nope", "timeout", "msg", repo=repo)
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.cancel_job("nope", repo=repo)


def test_start_job_owner_override_records_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing budget.owner seam records caller jobs; default stays kernel."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    assert repo.get_job(src).owner == "kernel"
    created = _svc.start_job(sid, "source_agent", budget={"owner": "thesis"}, repo=repo)
    assert created["owner"] == "thesis"
    assert ResearchRepository().get_job(str(created["job_id"])).owner == "thesis"


def test_record_evidence_rejects_bad_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    base = _svc_item(f"{sid}:ev:badmeta")
    bads: list[object] = [{"ok": object()}, {1: "x"}, "not-a-dict", [("k", "v")]]
    for i, bad in enumerate(bads):
        item = dict(base)
        item["evidence_id"] = f"{sid}:ev:bad{i}"
        item["metadata"] = bad
        with pytest.raises(ValueError, match="metadata"):
            _svc.record_evidence(sid, src, item, repo=repo)
    good = dict(base)
    good["evidence_id"] = f"{sid}:ev:goodmeta"
    good["metadata"] = {"source": "sec", "page": 3}
    out = _svc.record_evidence(sid, src, good, repo=repo)
    meta = out["metadata"]
    assert isinstance(meta, dict)
    assert meta["source"] == "sec" and meta["page"] == 3
    assert meta["source_bytes"] == "archived" and "identity_key" in meta


# ---------------------------------------------------------------------------
# SEC-only NVDA/Anthropic regression + context/coverage/dedup/provenance.
# Forward-compatible: new APIs via lazy getattr; behavior asserts only,
# never exact call counts.
# ---------------------------------------------------------------------------


def _sec_only_policy() -> dict[str, JSONValue]:
    """SEC-only allowlist policy shape per contract (research_sources in)."""
    return {"research_sources": {"mode": "allowlist", "sources": ["SEC"]}}


def test_sec_only_policy_persisted_and_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sess = _session.create_session("NVDA AI demand?", "o", as_of=ASOF, policy=_sec_only_policy())
    doc = sess.to_dict()
    for key in (
        "session_id",
        "query",
        "objective",
        "as_of",
        "status",
        "policy",
        "budget",
        "source_policy",
        "temporal_scope",
    ):
        assert key in doc
    sp = doc["source_policy"]
    assert isinstance(sp, dict)
    assert sp.get("mode") == "allowlist"
    allowed = sp.get("allowed")
    assert isinstance(allowed, list) and "SEC" in allowed
    assert sp.get("denied") == []
    ts = doc["temporal_scope"]
    assert isinstance(ts, dict)
    for key in ("as_of", "start", "end", "mode", "raw"):
        assert key in ts
    assert ts.get("mode") == "as_of"
    # Budgets kept alongside the new policy keys.
    budget = doc["budget"]
    assert isinstance(budget, dict)
    for key in ("deadline_seconds", "total_tool_budget", "total_token_budget", "total_cost_budget"):
        assert key in budget, key
    # Persist + reload: SQLite round-trips both fields.
    repo.save_session(sess)
    loaded = repo.get_session(sess.session_id)
    assert loaded.source_policy == sess.source_policy
    assert loaded.temporal_scope == sess.temporal_scope
    # Round-trip via from_dict/validate preserves the contract keys.
    from app.research.models import ResearchSession as _RS

    back = _RS.from_dict(doc)
    back.validate()
    assert back.to_dict()["source_policy"] == doc["source_policy"]
    assert back.to_dict()["temporal_scope"] == doc["temporal_scope"]
    # Temporal kwarg and default latest-available mode.
    scoped = _session.create_session("NVDA demand?", "o", as_of=None, temporal="between 2025-01-01 and 2025-06-30")
    assert scoped.temporal_scope.get("mode") == "range"
    latest = _session.create_session("NVDA demand?", "o", as_of=None)
    assert latest.temporal_scope.get("mode") == "latest-available"
    assert latest.temporal_scope.get("as_of") is not None


def test_sec_only_nvda_anthropic_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    calls: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["search_sec_filings", "list_sec_filings"]}
        if name == "call_tool":
            inner = args.get("name")
            inner_name = inner if isinstance(inner, str) and inner else "unknown_tool"
            calls.append(inner_name)
            # No-issuer probe (Anthropic, private): empty record, never an error.
            if "anthropic" in str(args).lower() and "nvda" not in str(args).lower():
                return {"record": {}, "evidence_ids": []}
            return {
                "record": {
                    "id": "r-" + str(len(calls)),
                    "known_at": "2025-05-01",
                    "uri": "https://sec.gov/x",
                    "record_id": "0001045810-25-000023",
                },
                "evidence_ids": ["EV-1", "EV-2"],
            }
        return {}

    from app.research.agents.source_agent import build_research_context

    ctx = build_research_context("NVDA AI demand vs Anthropic private AI concepts?", ["NVDA"])
    blob = str(ctx).lower()
    assert "nvda" in blob
    assert "anthropic" in blob or "ai" in blob
    out = run_live(
        question="NVDA AI demand vs Anthropic private AI concepts?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_dispatch,
        model=_grounded,
        repo=repo,
        budgets=None,
    )
    # No-issuer does not stop the session; NVDA filings + conceptual searches ran.
    assert out["stop_reason"] in ("complete:wave1", "complete:wave2")
    assert calls, "expected NVDA filing/conceptual SEC calls"
    # No SEC count cap: more than 3 useful calls allowed (behavior, not exact N).
    assert len([c for c in calls if c]) >= 3
    assert "up to 3" not in str(out).lower()
    # Freeze shared by trio; final answer produced with SEC-only limits explicit.
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    assert stock.evidence_ids == bull.evidence_ids == bear.evidence_ids
    sid = out["session_id"]
    assert isinstance(sid, str)
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict) and str(final.get("answer", "")).strip()
    assert final.get("freeze_id") == out["freeze_id"]


def test_dedup_collapses_cosmetic_repeats() -> None:
    from app.research.agents.scout import normalize_query
    from app.research.agents.source_agent import normalize_query as _sa_norm

    assert normalize_query("AI demand") == normalize_query("AI   demand")
    assert normalize_query("AI Demand") == normalize_query("ai demand")
    assert _sa_norm("AI demand") == normalize_query("AI   demand")
    # Single execution oracle: normalized equivalents share one slot.
    executed: set[str] = set()
    for variant in ("AI demand", "AI   demand", "ai DEMAND"):
        key = normalize_query(variant)
        assert isinstance(key, str)
        executed.add(key)
    assert len(executed) == 1


def test_scout_executes_normalized_repeat_once() -> None:
    from app.research.agents.scout import ScoutAssignment, run_scout

    seen: list[tuple[str, object]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("browse_tools", "search_tools"):
            return {"matches": []}
        if name == "call_tool" and args.get("name") == "search_sec_filings":
            inner = args.get("arguments")
            query = inner.get("query") if isinstance(inner, dict) else None
            seen.append((str(query), inner.get("as_of") if isinstance(inner, dict) else None))
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["AI demand", "AI   demand", "ai DEMAND"],
        baseline=[],
    )
    result = run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")
    assert len(seen) == 1
    assert seen[0][0] == "AI demand"
    assert seen[0][1] == "2025-06-30"
    assert result.unknowns and result.unknowns[0] == "no PIT-eligible SEC evidence returned"


def test_scout_material_events_args_always_carry_since() -> None:
    """The harness emits schema-valid role calls: `since` is required, bounded or not.

    Live defect: unbounded sessions (as_of "unbounded"/blank) built get_material_events
    with no `since`, so the tool layer answered invalid_tool_arguments and the filings
    scout silently lost its material-events coverage. The unbounded window floors at the
    documented 2024-01-01 because there is no cutoff to measure a lookback back from.
    """
    from app.research.agents.scout import ScoutAssignment, run_scout
    from app.tools import _validate_tool_arguments

    calls: list[tuple[str, dict[str, object]]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = args.get("name")
        inner_args = args.get("arguments")
        assert isinstance(inner, str) and isinstance(inner_args, dict)
        calls.append((inner, inner_args))
        return {"evidence_ids": []}

    def _role_calls(as_of: str) -> list[tuple[str, dict[str, object]]]:
        calls.clear()
        run_scout(
            ScoutAssignment(
                assignment_id="scout-filings",
                session_id="rs:t",
                as_of=as_of,
                role="filings",
                question="What changed at NVDA?",
                tickers=["NVDA"],
            ),
            dispatch=_dispatch,
            model=lambda prompt: "[]",
        )
        # Every emitted role call passes the real schema check, not a hand-written shape.
        for tool, args in calls:
            assert _validate_tool_arguments(tool, args) is None, (as_of, tool, args)
        return list(calls)

    for no_cutoff in ("unbounded", "", "   "):
        events = [args for tool, args in _role_calls(no_cutoff) if tool == "get_material_events"]
        assert events and [args["since"] for args in events] == ["2024-01-01"], no_cutoff
        # No cutoff: nothing is date-filtered on the way to the tool.
        assert all("as_of" not in args for args in events)
        searches = [args for tool, args in calls if tool == "search_sec_filings"]
        assert searches and all("as_of" not in args for args in searches)

    events = [args for tool, args in _role_calls("2026-08-10") if tool == "get_material_events"]
    assert [args["since"] for args in events] == ["2025-08-10"]
    assert [args["as_of"] for args in events] == ["2026-08-10"]
    searches = [args for tool, args in calls if tool == "search_sec_filings"]
    assert searches and all(args["as_of"] == "2026-08-10" for args in searches)


def test_pit_eligibility_short_circuits_only_without_a_cutoff() -> None:
    """No bounded as_of means no PIT filtering; a bounded as_of keeps the strict rule."""
    from app.research.agents.scout import _is_pit_eligible

    dated = "2011-03-16T16:33:51+00:00"
    for no_cutoff in ("unbounded", "", "   ", None):
        # The sentinel reaches only the scout (the ledger gate sees None), so a non-ISO
        # as_of must mean "no cutoff" instead of raising inside the PIT gate.
        assert _is_pit_eligible(dated, no_cutoff) is True, no_cutoff
        assert _is_pit_eligible(None, no_cutoff) is True, no_cutoff
    assert _is_pit_eligible(dated, "2025-06-30") is True
    assert _is_pit_eligible("2025-07-01T00:00:00+00:00", "2025-06-30") is False
    assert _is_pit_eligible(None, "2025-06-30") is False
    assert _is_pit_eligible("", "2025-06-30") is False
    assert _is_pit_eligible(7, "2025-06-30") is False
    # A bounded cutoff that cannot be parsed is never verifiable: fail closed, never admit.
    assert _is_pit_eligible(dated, "2025-06-30T25:00:00+00:00") is False


def test_scout_unbounded_citations_reach_the_model() -> None:
    """An unbounded scout keeps its dated citations: the model gets citable ids.

    Live defect: 64/64 unbounded citations were rejected as PIT-ineligible, so the
    prompt listed no ids and every claim was dropped as uncited.
    """
    from app.research.agents.scout import ScoutAssignment, ScoutResult, run_scout

    prompts: list[str] = []
    journal: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        return {
            "evidence_ids": [
                {
                    "evidence_id": "EV-1",
                    "known_at": "2011-03-16T16:33:51+00:00",
                    "claim_text": "Data center revenue grew",
                }
            ],
            "top_hits": [],
        }

    def _model(prompt: str) -> str:
        prompts.append(prompt)
        return "[]"

    def _journal(kind: str, payload: dict[str, object]) -> None:
        journal.append(kind)

    def _scout(no_cutoff: str) -> ScoutResult:
        prompts.clear()
        journal.clear()
        return run_scout(
            ScoutAssignment(
                assignment_id="scout-filings",
                session_id="rs:t",
                as_of=no_cutoff,
                role="filings",
                question="What changed at NVDA?",
                tickers=["NVDA"],
            ),
            dispatch=_dispatch,
            model=_model,
            journal=_journal,
        )

    for no_cutoff in ("unbounded", ""):
        result = _scout(no_cutoff)
        assert "evidence.rejected" not in journal, no_cutoff
        assert "EV-1 (known_at=2011-03-16T16:33:51+00:00) :: Data center revenue grew" in prompts[0]
        assert result.limitations == [], no_cutoff
        assert result.unknowns == [], no_cutoff


def test_scout_executes_unscoped_counterparty_queries_before_the_model() -> None:
    """Global counterparty searches run with no ticker/cik, pre-model; their hits are opened."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    seq: list[str] = []
    searched: list[dict[str, object]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = args.get("name")
        inner_args = args.get("arguments")
        assert isinstance(inner, str) and isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            seq.append(f"search:{inner_args.get('query')}")
            searched.append(inner_args)
            return {
                "top_hits": [{"accession": "0001-25-000001", "document": "ex-10.htm"}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            seq.append(f"open:{inner_args.get('accession_no')}")
            return {
                "evidence_ids": [
                    {
                        "evidence_id": "EV-1",
                        "known_at": "2025-05-01",
                        "claim_text": "ok",
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="What happens to Microsoft if OpenAI goes bankrupt?",
        tickers=["MSFT"],
        unscoped_queries=["OpenAI", "OpenAI exposure"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: seq.append("model") or "[]")

    unscoped = [a for a in searched if "query" in a]
    assert [a["query"] for a in unscoped] == ["OpenAI", "OpenAI exposure"]
    assert all("ticker" not in a and "cik" not in a for a in unscoped)
    assert [e for e in seq if e.startswith("open:")] == ["open:0001-25-000001"]
    assert seq[-1] == "model"  # every search, and the document read it surfaced, ran first


def test_scout_expansion_searches_derived_queries_and_opens_new_documents() -> None:
    """Expansion: passage-derived queries run, their documents open, and the loop stops at fixpoint."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    searches: list[str] = []
    opens: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = str(args.get("name"))
        inner_args = args.get("arguments")
        assert isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            query = str(inner_args.get("query"))
            searches.append(query)
            hit = ("ACC-1", "one.htm") if query == "assigned query" else ("ACC-2", "two.htm")
            return {
                "top_hits": [{"accession": hit[0], "document": hit[1]}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            accession = str(inner_args.get("accession_no"))
            opens.append(accession)
            text = "Hyperscaler concentration grew" if accession == "ACC-1" else "ok"
            return {
                "evidence_ids": [
                    {
                        "evidence_id": f"EV-{accession}",
                        "known_at": "2025-05-01",
                        "claim_text": text,
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["assigned query"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")

    # Round 1 = assigned query; round 2 = the terms the opened ACC-1 passage names.
    assert searches == ["assigned query", "Hyperscaler", "concentration"]
    assert opens == [
        "ACC-1",
        "ACC-2",
    ]  # the document the derived searches surfaced, opened once


def test_scout_expansion_never_reexecutes_a_seen_query() -> None:
    """A derived term already executed (any case/whitespace form) is suppressed, never re-run."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    searches: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = str(args.get("name"))
        inner_args = args.get("arguments")
        assert isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            searches.append(str(inner_args.get("query")))
            return {
                "top_hits": [{"accession": "ACC-1", "document": "one.htm"}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            return {
                "evidence_ids": [
                    {
                        "evidence_id": "EV-1",
                        "known_at": "2025-05-01",
                        "claim_text": "concentration. Ferrari up.",
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["CONCENTRATION"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")
    assert searches == ["CONCENTRATION", "Ferrari"]


def test_scout_prompt_renders_context_baseline_and_queries() -> None:
    """Scout prompt: context lines + relationships, latest-filing baseline, assigned queries."""
    from app.research.agents.scout import ScoutAssignment, build_scout_prompt

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2026-08-10",
        role="filings",
        question="NVDA data center growth?",
        tickers=["NVDA"],
        context={
            "primary_entities": ["NVDA"],
            "concepts": ["accelerated computing", "  "],
            "relationships": [
                {"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"},
                {"subject": "NVDA", "relation": "  ", "object": "dropped"},
                "not-a-triple",
            ],
        },
        baseline=["NVDA 10-Q data center segment", "NVDA 10-K competition"],
        queries=["NVDA data center revenue"],
    )
    prompt = build_scout_prompt(assignment)
    assert "Context primary_entities: NVDA" in prompt
    assert "Context concepts: accelerated computing" in prompt
    assert "Context relationships: NVDA supplies hyperscalers" in prompt
    assert "dropped" not in prompt and "not-a-triple" not in prompt
    assert "Latest-filing baseline (as_of-filtered, target searches with its terms):" in prompt
    assert "- NVDA 10-K competition" in prompt
    assert "Assigned queries (search each; skip exact repeats already executed):" in prompt
    assert "- NVDA data center revenue" in prompt

    bare = build_scout_prompt(
        ScoutAssignment(
            assignment_id="scout-risk", session_id="rs:t", as_of="unbounded", role="risk", question="Q?", tickers=[]
        )
    )
    assert "Context " not in bare and "Latest-filing baseline" not in bare
    assert "Assigned queries" not in bare and "NO ticker and NO cik" not in bare
    assert "scope tickers TBD" in bare and "As of: unbounded" in bare
    assert "Workflow:" in bare and "Respond with JSON only" in bare


def test_merge_context_unions_model_terms() -> None:
    from app.research.agents.source_agent import _merge_context, build_research_context

    base = build_research_context("NVDA AI demand?", ["NVDA"])
    merged = _merge_context(
        base,
        {
            "concepts": ["accelerated computing"],
            "relationships": [{"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"}],
        },
    )
    assert "accelerated computing" in str(merged["concepts"]).lower()
    rels = merged["relationships"]
    assert isinstance(rels, list)
    assert any(isinstance(r, dict) and r.get("object") == "hyperscalers" for r in rels)
    assert _merge_context(base, "not-a-dict") == base


def test_expand_queries_mines_findings_and_context() -> None:
    from app.research.agents.source_agent import build_research_context, expand_queries

    ctx = build_research_context("NVDA AI demand?", ["NVDA"])
    out = expand_queries(["nvda demand"], ["Hyperscaler concentration grew in filings"], ctx)
    assert out and all(q.strip() for q in out)
    assert "nvda demand" not in [q.lower() for q in out]


def test_expansion_stop_prefers_info_over_counts() -> None:
    from app.research.agents.source_agent import expansion_stop

    assert expansion_stop(sec_answerable_remaining=False)[0] is True
    assert expansion_stop(new_queries=False)[0] is True
    assert expansion_stop(new_queries=True) == (False, "continue")


@pytest.mark.parametrize(
    ("question", "tickers", "must_contain"),
    [
        ("Spirit AeroSystems Boeing 737 relationship and backlog?", ["SPR"], ("aerospace", "boeing")),
        ("Novo Nordisk GLP-1 diabetes obesity outlook?", ["NVO"], ("glp", "diabetes", "obesity")),
        ("Arista cloud networking datacenter Ethernet demand?", ["ANET"], ("cloud", "network", "datacenter")),
        ("Albemarle lithium brine battery demand?", ["ALB"], ("lithium", "battery")),
        ("Apple China supply chain and tariffs?", ["AAPL"], ("china", "supply")),
    ],
)
def test_cross_domain_context_carries_industry_terms(
    question: str, tickers: list[str], must_contain: tuple[str, ...]
) -> None:
    from app.research.agents.source_agent import (
        build_query_families,
        build_research_context,
    )

    ctx = build_research_context(question, tickers)
    blob = str(ctx).lower()
    assert any(term in blob for term in must_contain), blob[:500]
    # Industry/relationship/risk content beyond the raw query words.
    for key in ("industries", "relationships", "risks", "concepts"):
        assert key in ctx, sorted(ctx.keys())
    # No issuer-specific branches: generic families serve every domain.
    families = build_query_families(ctx)
    assert families and len({f.lower() for f in families}) == len(families)
    assert "Anthropic" not in str(families)


def test_private_counterparty_queries_reach_filings_scout() -> None:
    """MSFT/OpenAI: the private counterparty is searched unscoped, never dead-ended."""
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
        decompose_question,
    )

    question = "What happens to Microsoft if OpenAI goes bankrupt?"
    groups = _grouped_queries(build_research_context(question, ["MSFT"]))
    assert {"OpenAI", "OpenAI exposure", "OpenAI agreement", "OpenAI 8-K"} <= set(groups["cp"])

    def _catalog_only(name: str, args: dict[str, object]) -> dict[str, object]:
        return {}  # discovery hint only; no query execution on this path

    assignments = decompose_question(
        question, session_id="rs:t", as_of="2026-08-10", tickers=["MSFT"], dispatch=_catalog_only
    )
    filings = next(a for a in assignments if a.role == "filings")
    assert "OpenAI" in filings.queries  # bare name -> unscoped EDGAR full-text search
    # The counterparty family is wired into the filings role and nowhere else.
    for other in (a for a in assignments if a.role != "filings"):
        assert set(groups["cp"]).isdisjoint(other.queries)


def test_no_counterparty_question_keeps_existing_families() -> None:
    """NVDA-only question: counterparty family stays empty, every other family unchanged."""
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
    )

    ctx = build_research_context("NVDA data center revenue growth", ["NVDA"])
    assert ctx["related_entities"] == []
    assert _grouped_queries(ctx) == {
        "cp": [],
        "a": ["NVDA"],
        "b": [],
        "c": [],
        "d": [],
        "e": [],
        "f": ["NVDA risk factors"],
        "related": [],
        "supp_bare": ["data", "center", "revenue", "growth"],
        "supp_filing": [
            "data 8-K",
            "data proxy",
            "data N-PX",
            "data agreement",
            "center 8-K",
            "center proxy",
            "center N-PX",
            "center agreement",
            "revenue 8-K",
            "revenue proxy",
            "revenue N-PX",
            "revenue agreement",
            "growth 8-K",
            "growth proxy",
            "growth N-PX",
            "growth agreement",
        ],
        "supp_risk": ["data risk factor", "center risk factor", "revenue risk factor", "growth risk factor"],
    }


def test_counterparty_family_includes_every_related_entity() -> None:
    """Counterparty family: all eight related entities, four variants each, scope tickers excluded."""
    from app.research.agents.source_agent import _grouped_queries

    ctx: dict[str, object] = {
        "primary_entities": ["MSFT"],
        "related_entities": ["MSFT", *(f"Counter {i}" for i in range(8))],
    }
    cp = _grouped_queries(ctx)["cp"]
    assert len(cp) == 32  # 8 entities x (bare + exposure/agreement/8-K); no entity cap
    assert "MSFT" not in cp and "Counter 7" in cp and "Counter 7 8-K" in cp


def test_role_assignments_dispatch_every_material_query() -> None:
    """No first-two-per-family cap: every planned family query reaches its role's assignment."""
    from app.research.agents.source_agent import _grouped_queries, _role_assignments

    ctx: dict[str, object] = {
        "primary_entities": ["MSFT"],
        "related_entities": ["Alpha", "Beta", "Gamma", "Delta"],
        "industries": ["industrials", "software", "semiconductors", "services"],
        "products": ["Azure", "Office", "Dynamics"],
        "technologies": [],
        "concepts": ["cloud"],
        "risks": ["credit", "litigation", "regulatory", "supply"],
        "catalysts": [],
    }
    groups = _grouped_queries(ctx)
    assignments = _role_assignments("Q?", "rs:t", "2026-08-10", ["MSFT"], ctx, groups, [])
    by_role = {a.role: a.queries for a in assignments}
    for family, role in (
        ("b", "filings"),
        ("cp", "filings"),
        ("related", "filings"),
        ("supp_filing", "filings"),
        ("c", "financials"),
        ("d", "financials"),
        ("supp_bare", "financials"),
        ("f", "risk"),
        ("supp_risk", "risk"),
    ):
        assert len(groups[family]) > 2, (family, groups[family])
        assert [q for q in by_role[role] if q in set(groups[family])] == groups[family], family


def test_scout_prompt_demands_global_search_for_non_filers() -> None:
    """Scout prompt: counterparty queries are listed as global - no ticker, no cik, no scope shortcut."""
    from app.research.agents.scout import ScoutAssignment, build_scout_prompt

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2026-08-10",
        role="filings",
        question="What happens to Microsoft if OpenAI goes bankrupt?",
        tickers=["MSFT"],
        queries=["MSFT OpenAI"],
        unscoped_queries=["OpenAI", "OpenAI exposure"],
    )
    prompt = build_scout_prompt(assignment)
    assert "NO ticker and NO cik" in prompt
    assert "- OpenAI exposure" in prompt
    assert "never conclude a relationship is absent from a scoped search" in prompt
    # The branch must be truthful about what already ran: the unscoped searches execute
    # before the model call and the documents their hits named are opened for it.
    assert "these queries have already run with NO ticker and NO cik" in prompt
    assert "already run" in prompt and "opened below" in prompt
    # ...and it leads the prompt, before the scoped query list.
    assert prompt.index("Counterparty branch first") < prompt.index("Assigned queries")
    # Without counterparty queries the block is absent: no instruction drift for ordinary questions.
    scoped_only = build_scout_prompt(
        ScoutAssignment(
            assignment_id="scout-filings",
            session_id="rs:t",
            as_of="2026-08-10",
            role="filings",
            question="NVDA data center revenue growth",
            tickers=["NVDA"],
            queries=["NVDA"],
        )
    )
    assert "NO ticker and NO cik" not in scoped_only


def test_no_date_means_latest_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    sess = _session.create_session("NVDA demand?", "o", as_of=None)
    doc = sess.to_dict()
    assert doc["as_of"] is None
    ts = doc["temporal_scope"]
    assert isinstance(ts, dict)
    assert ts.get("mode") == "latest-available"
    assert ts.get("as_of") is not None


def test_as_of_excludes_later_evidence() -> None:
    sess = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-FUT2", sess.session_id, 1, datetime(2025, 7, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen and seen[0][1]["reason"] == "PIT_VIOLATION"
    assert led.ids() == ()
    # Latest-doc respects cutoff: eligible doc ingests, later doc rejects.
    ingest_evidence(led, _ev("EV-OK", sess.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)), as_of=ASOF, on_reject=None)
    assert led.ids() == ("EV-OK",)


def test_coverage_shape_and_insufficient_never_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["Anthropic private revenue"],
        "source_limitations": ["SEC-only: no private-issuer filings"],
    }
    out = _svc.submit_source_result(
        src, coverage=coverage, evidence_ids=[], unresolved_questions=["Anthropic private revenue"], repo=repo
    )
    assert out["job_status"] == "completed"
    # submit = source job complete only; session never terminal here.
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")
    # Coverage round-trips on the completed job result when the field lands.
    stored = repo.get_job(src).result or {}
    result_cov = stored.get("coverage") if isinstance(stored, dict) else None
    if isinstance(result_cov, dict):
        assert result_cov.get("useful_for_question") == "insufficient"
        for key in ("resolved", "partially_resolved", "unresolved", "source_limitations"):
            if key in result_cov:
                assert isinstance(result_cov[key], list)
    # Insufficient still freezes + synthesizes via the kernel path.
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == f"{sid}:1:freeze"


def test_negatives_cite_searchrun_only() -> None:
    import dataclasses as _dc

    from app.sec.models import DocumentMatch, MatchingPassage, SearchRun

    fields = {f.name for f in _dc.fields(SearchRun)}
    for required in (
        "id",
        "source",
        "query",
        "filters",
        "executed_at",
        "as_of",
        "matched_entities",
        "matched_documents",
        "matched_passages",
    ):
        assert required in fields, sorted(fields)
    # Negative-claim shape: grounded positives cite document/passage/accession,
    # negatives cite the SearchRun id only — never 'not found' as 'does not exist'.
    doc_fields = {f.name for f in _dc.fields(DocumentMatch)}
    assert doc_fields >= {"accession", "matching_passages"}
    passage_fields = {f.name for f in _dc.fields(MatchingPassage)}
    assert passage_fields >= {"document", "query", "score"}


def test_sec_gates_deny_non_sec_even_via_browse() -> None:
    from app.research.agents.source_agent import is_sec_tool

    for denied in ("query_finra", "get_short_interest", "search_web", "get_market_snapshot", "get_analyst_estimates"):
        assert is_sec_tool(denied) is False, denied
    for allowed in ("search_sec_filings", "list_sec_filings", "get_sec_document"):
        assert is_sec_tool(allowed) is True, allowed
    # Runner gate surfaces POLICY_DENIED (not a prompt-text refusal).
    import inspect as _inspect

    from app.research.runner import _LiveRun

    src = _inspect.getsource(_LiveRun._guarded_tool_call)
    assert "POLICY_DENIED" in src
    assert "is_sec_tool" in src


def test_resolve_source_policy_sec_only_allowlist() -> None:
    from app.research.models import resolve_source_policy, validate_source_policy

    sec_only = resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": ["SEC"]}})
    assert sec_only["mode"] == "allowlist"
    allowed = sec_only["allowed"]
    assert isinstance(allowed, list) and "SEC" in allowed
    assert sec_only["denied"] == []
    # Kernel default is SEC-only even with no policy input.
    assert resolve_source_policy(None)["mode"] == "allowlist"
    # Round-trip through the validator preserves the contract shape.
    assert validate_source_policy(sec_only) == sec_only
    with pytest.raises(ValueError, match="mode"):
        resolve_source_policy({"research_sources": {"mode": "someday", "sources": ["SEC"]}})


def test_resolve_temporal_scope_modes() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope, validate_temporal_scope

    now = _dt(2025, 6, 30, tzinfo=UTC)
    # No date -> latest-available with cutoff set, never invented/None.
    latest = resolve_temporal_scope(query="NVDA demand?", now=now)
    assert latest["mode"] == "latest-available"
    assert latest["as_of"] is not None
    # as-of-2025-01-01 excludes later docs.
    asof = resolve_temporal_scope(as_of="2025-01-01", now=now)
    assert asof["mode"] == "as_of"
    assert str(asof["as_of"])[:10] == "2025-01-01"
    # Interval start/end bounds.
    interval = resolve_temporal_scope(temporal="between 2025-01-01 and 2025-06-30", now=now)
    assert interval["mode"] == "range"
    assert str(interval["start"])[:10] == "2025-01-01"
    assert str(interval["end"])[:10] == "2025-06-30"
    # Last-quarter range resolves start/end around the pinned clock.
    quarter = resolve_temporal_scope(temporal="this quarter", now=now)
    assert quarter["mode"] == "range"
    assert quarter["start"] is not None and quarter["end"] is not None
    assert validate_temporal_scope(dict(latest)) == latest


# ---------------------------------------------------------------------------
# RegressionEval §17 suites (deterministic, fakes only, no network).
# Naming: test_reg_<suite>_<behavior>. Contract stubs resolve via getattr
# with a precise failure naming the missing source hook; no source edits.
# ---------------------------------------------------------------------------


def _reg_svc_sid(repo: ResearchRepository, q: str = "GS OpenAI exposure?") -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _reg_cov(useful: str = "sufficient") -> dict[str, object]:
    """sufficient now needs the full structured envelope + no residual branches/questions."""
    return {
        "useful_for_question": useful,
        "resolved": ["GS direct OpenAI exposure"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "major_entities_investigated": ["GS", "OpenAI"],
        "relationship_types_checked": ["investment", "commercial"],
        "forms_examined": ["10-K"],
        "exhibits_examined": ["EX-10.1"],
        "material_open_questions": [],
        "search_runs": ["s1"],
        "covered_branches": ["GS direct OpenAI exposure"],
    }


def _absence_cov(**over: object) -> dict[str, object]:
    """Full absence-observation coverage envelope (forms/dates/partitions/entities/docs/gaps + flags)."""
    cov: dict[str, object] = {
        "forms": ["10-K"],
        "dates": ["2025-02-14"],
        "partitions": ["efts"],
        "entities": ["GS"],
        "docs": ["0000886982-26-000001"],
        "gaps": [],
        "pagination_complete": True,
        "complete": True,
    }
    cov.update(over)
    return cov


def _reg_item(eid: str, wave: int = 1, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": f"GS OpenAI-linked exposure per filing {eid}",
        "subject": "GS",
        "source_name": "SEC",
        "source_uri": "https://www.sec.gov/Archives/edgar/data/886982/000088698226000001/primary.htm",
        "source_record_id": "0000886982-26-000001",
        "document_name": "gs-10q-20260331.htm",
        "matching_passage": "The firm discloses OpenAI-linked exposure in the filing.",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    base.update(over)
    if "source_handle" not in over:
        passage = str(base["matching_passage"])
        base["source_handle"] = seam.handle_for(
            passage,
            accession=str(base["source_record_id"]),
            document=str(base["document_name"]),
            known_at=str(base["known_at"]) if isinstance(base.get("known_at"), str) else None,
        )
    return base


def test_reg_lifecycle_running_until_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    assert repo.get_job(src).status == "running"
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    assert repo.get_job(src).status == "running"
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")


def test_reg_lifecycle_atomic_submit_completes_job_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    assert repo.get_job(src).status == "completed"
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")


def test_reg_lifecycle_double_submit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="CLOSED|closed|completed|running"):
        _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)


def test_reg_lifecycle_closed_rejects_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="running|closed|completed"):
        _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:2"), repo=repo)


# --- unlimited (4): 100+ distinct allowed, no default kill, dup-no-progress, diff queries ---


def _reg_dispatch(repo: ResearchRepository, sid: str, jid: str, tool: str) -> None:
    from app.research import service as _svc

    _svc.authorize_and_consume_dispatch(sid, jid, tool, repo=repo)


def test_reg_unlimited_100_distinct_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("GS OpenAI exposure?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jobs = repo.list_jobs(sid)
    assert len(jobs) == 1
    jid = jobs[0].job_id
    assert jobs[0].tool_budget is None, (
        "missing contract (LimitsLoopBudget owns "
        "app/research/jobs.py:job_tool_budget + app/research/models.py:DEFAULT_BUDGET): "
        "unlimited=None; source job tool_budget must default None"
    )
    for _ in range(100):
        _reg_dispatch(repo, sid, jid, "get_sec_document")
    assert repo.get_session(sid).budget.get("tool_calls_used") == 100


def test_reg_unlimited_no_default_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.models import DEFAULT_BUDGET

    assert DEFAULT_BUDGET.get("total_tool_budget") is None, (
        "missing contract (LimitsLoopBudget owns "
        "app/research/models.py:default_budget/DEFAULT_BUDGET): unlimited=None; "
        f"total_tool_budget must default None, got {DEFAULT_BUDGET.get('total_tool_budget')!r}"
    )
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _reg_svc_sid(repo)
    for _ in range(25):
        _reg_dispatch(repo, sid, jid, "get_sec_document")
    assert repo.get_session(sid).budget.get("tool_calls_used") == 25


def test_reg_unlimited_duplicate_no_progress_rejected() -> None:
    import app.research.director as _director

    normalize = getattr(_director, "normalize_research_action", None)
    loop_cls = getattr(_director, "LoopDetector", None)
    assert callable(normalize) and loop_cls is not None, (
        "missing source hook (LimitsLoopBudget owns "
        "app/research/director.py:normalize_research_action + LoopDetector): contract normalize("
        "source,tool,query,ticker,forms,as_of,accession,objective)->tuple; "
        "LoopDetector.check(action,result_hash,evidence_delta)->{duplicate,reason} + telemetry list"
    )
    loop = loop_cls()
    action = normalize("sec", "get_sec_document", "GS 10-K", "GS", ("10-K",), "2025-06-30", "ACC-1", "exposure?")
    first = loop.check(action, "hash-a", 3)
    assert first.get("duplicate") is False
    repeat = loop.check(action, "hash-a", 0)
    assert repeat.get("duplicate") is True, repeat


def test_reg_unlimited_different_queries_allowed() -> None:
    import app.research.director as _director

    normalize = getattr(_director, "normalize_research_action", None)
    loop_cls = getattr(_director, "LoopDetector", None)
    assert callable(normalize) and loop_cls is not None, (
        "missing source hook (LimitsLoopBudget owns "
        "app/research/director.py:normalize_research_action + LoopDetector): contract normalize("
        "source,tool,query,ticker,forms,as_of,accession,objective)->tuple; "
        "LoopDetector.check(action,result_hash,evidence_delta)->{duplicate,reason} + telemetry list"
    )
    loop = loop_cls()
    a = normalize("sec", "get_sec_document", "GS 10-K risk", "GS", ("10-K",), "2025-06-30", "ACC-1", "exposure?")
    b = normalize("sec", "get_sec_document", "GS 10-Q MD&A", "GS", ("10-Q",), "2025-06-30", "ACC-2", "exposure?")
    assert loop.check(a, "hash-a", 2).get("duplicate") is False
    assert loop.check(b, "hash-b", 2).get("duplicate") is False


# --- dispatch-boundary loop gate: exact no-progress repeats carry research_loop_detected ---
def test_dispatch_loop_gate_first_allowed_repeat_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First dispatch runs; the identical immediate repeat is refused and journaled, never re-executed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure", "ticker": "GS", "forms": ["10-K"]}
    first = _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    assert first["tool_calls_used"] == 1
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    blocked = [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"]
    assert len(blocked) == 1
    assert blocked[0].payload["reason"] == "research_loop_detected"
    assert blocked[0].payload["tool"] == "search_sec_filings"
    assert blocked[0].payload["query"] == "GS OpenAI exposure"
    assert blocked[0].payload["job_id"] == src


def test_dispatch_loop_gate_new_evidence_readmits_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New evidence is progress: the same action runs again, then its own zero-progress repeat is refused."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:1"), repo=repo)
    again = _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    assert again["tool_calls_used"] == 2
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)


def test_dispatch_loop_gate_survives_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracked actions are persisted: a fresh repository still refuses the no-progress repeat."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    fresh = ResearchRepository()
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=fresh)


def test_dispatch_loop_gate_distinct_arguments_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One tool with different arguments (documents, queries, forms) is a different action every time."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    calls = [
        ("get_sec_document", {"accession_no": "0000320193-25-000079", "document_name": "nvda-20250331.htm"}),
        ("get_sec_document", {"accession_no": "0000320193-25-000079", "document_name": "nvda-20250331ex10.htm"}),
        ("search_sec_filings", {"query": "GS OpenAI exposure"}),
        ("search_sec_filings", {"query": "GS OpenAI exposure", "forms": ["10-Q"]}),
    ]
    for tool, args in calls:
        _svc.authorize_and_consume_dispatch(sid, src, tool, arguments=args, repo=repo)
    assert repo.get_session(sid).budget.get("tool_calls_used") == len(calls)
    assert [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"] == []


def test_dispatch_loop_gate_state_is_per_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking is per session+job: another job in the same session may run the same action."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    scout = str(_svc.start_job(sid, "scout", repo=repo, wave_id=1)["job_id"])
    assert (
        _svc.authorize_and_consume_dispatch(sid, scout, "search_sec_filings", arguments=args, repo=repo)["job_id"]
        == scout
    )


def test_dispatch_loop_gate_gateway_repeat_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model path passes arguments through the gateway: the repeat is refused before the tool runs."""
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _reg_svc_sid(repo)
    ctx = RuntimeToolSession(session_id="t-loop")
    ctx.active_research_session_id = sid
    ctx.active_research_job_id = jid
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        calls.append((name, dict(arguments)))
        return {"result_type": "sec_search", "query": arguments.get("query"), "count": 1, "results": []}

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    first = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert "error" not in first, first
    second = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert "research_loop_detected" in str(second.get("error", "")), second
    assert len(calls) == 1  # the repeat never reached the tool
    third = execute_agent_tool("search_sec_filings", {"query": "GS 10-Q risk factors"}, ctx)
    assert "error" not in third, third
    assert len(calls) == 2


def test_dispatch_loop_gate_no_staged_job_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged session with no active job keeps today's refusal verbatim; the loop gate never sees it."""
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _src = _reg_svc_sid(repo)
    ctx = RuntimeToolSession(session_id="t-nojob")
    ctx.active_research_session_id = sid
    out = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert out.get("error_type") == "invalid_research_context"
    assert "Active research job is required" in str(out.get("error"))
    assert repo.get_session(sid).budget.get("tool_calls_used", 0) == 0
    assert [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"] == []


def test_reg_freshness_latest_default() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope

    latest = resolve_temporal_scope(
        query="What happens to Goldman Sachs if OpenAI goes bankrupt?", now=_dt(2025, 6, 30, tzinfo=UTC)
    )
    assert latest["mode"] == "latest-available"
    assert latest["as_of"] is not None


def test_reg_freshness_latest_10k_pinned() -> None:
    from app.research.models import select_latest_baseline

    filings: list[object] = [
        {"form": "10-K", "known_at": "2025-02-14", "filed_at": "2025-02-14", "accession_no": "OLD"},
        {"form": "10-K", "known_at": "2025-06-20", "filed_at": "2025-06-20", "accession_no": "NEW"},
        {"form": "10-K/A", "known_at": "2025-06-25", "filed_at": "2025-06-25", "accession_no": "AMD"},
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    annual = base["annual_10k"]
    acc = annual.get("accession_no") if isinstance(annual, dict) else getattr(annual, "accession_no", None)
    assert acc in ("NEW", "AMD"), acc


def test_reg_freshness_superseded_only_current_rejected() -> None:
    from app.research.models import select_latest_baseline, superseded_current_violation

    filings: list[object] = [
        {
            "form": "10-K",
            "known_at": "2024-02-10",
            "filed_at": "2024-02-10",
            "accession_no": "OLD",
            "superseded_by": "NEW",
        },
        {"form": "10-K", "known_at": "2025-02-14", "filed_at": "2025-02-14", "accession_no": "NEW"},
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    annual = base["annual_10k"]
    acc = annual.get("accession_no") if isinstance(annual, dict) else getattr(annual, "accession_no", None)
    assert acc == "NEW", acc
    assert superseded_current_violation(filings, "OLD", as_of="2025-06-30") is not None
    assert superseded_current_violation(filings, "NEW", as_of="2025-06-30") is None


def test_reg_freshness_historical_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PIT-eligible historical evidence ingests; a future known_at rejects.
    # Distinct accession per item: dedupe keys on source_record_id, so a shared
    # accession would return duplicate_of instead of reaching the PIT gate.
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("GS 2024 exposure?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jid = repo.list_jobs(sid)[0].job_id
    out = _svc.record_evidence(sid, jid, _reg_item(f"{sid}:ev:1", known_at="2024-12-01T00:00:00+00:00"), repo=repo)
    assert out["evidence_id"] == f"{sid}:ev:1"
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected|ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            jid,
            _reg_item(
                f"{sid}:ev:2",
                source_record_id="0000886982-26-000002",
                source_uri="https://www.sec.gov/Archives/edgar/data/886982/000088698226000002/primary.htm",
                source_handle=seam.handle_for(
                    "The firm discloses OpenAI-linked exposure in the filing.",
                    accession="0000886982-26-000002",
                    document="gs-10q-20260331.htm",
                    known_at="2025-07-01T00:00:00+00:00",
                ),
            ),
            repo=repo,
        )


def test_reg_negatives_coverage_present() -> None:
    """The closed vocabularies carry the claim/provenance contract; default coverage keeps its keys."""
    from app.research.dossiers.sec import default_coverage
    from app.research.evidence import CLAIM_KINDS, PROVENANCE_KINDS

    # Evidence is kernel-replayed claims only: a search-derived absence is coverage state.
    assert CLAIM_KINDS == ("observed_fact",)
    assert PROVENANCE_KINDS == ("sec_source", "sec_record", "finra_record", "web_source", "search_run", "none")
    cov = default_coverage()
    for key in ("forms", "sources_examined", "complete", "exclusions"):
        assert key in cov, sorted(cov.keys())


def test_reg_negatives_scoped_no_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scoped no-hit observation records as a coverage artifact with its full coverage envelope."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    scoped_cov = _absence_cov(complete=False, gaps=["sections 4-6 unread"], pagination_complete=True)
    out = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "no OpenAI bankruptcy exposure disclosed in sections 1-3 of the scoped GS 10-K",
            "claim_text": "not found in sections 1-3 of the scoped GS 10-K: OpenAI bankruptcy exposure",
            "claim_kind": "absence_observation",
            "subject": "GS",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "s1",
            "query": "GS OpenAI bankruptcy",
            "coverage": scoped_cov,
        },
        repo=repo,
    )
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    assert out.get("search_id") == "s1" and out.get("query") == "GS OpenAI bankruptcy"
    assert out.get("coverage") == scoped_cov
    stored = repo.list_coverage_artifacts(sid)[0]
    assert stored.get("search_id") == "s1" and stored.get("coverage") == scoped_cov
    assert repo.list_evidence(sid) == []  # coverage state never lands in the ledger
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(
            sid,
            src,
            {
                "wave_id": 1,
                "content": "no exposure anywhere",
                "claim_text": "no OpenAI exposure in any filing",
                "claim_kind": "absence_observation",
                "subject": "GS",
                "source_name": "SEC",
                "known_at": "2025-06-29T00:00:00+00:00",
                "search_id": "s1",
                "query": "GS OpenAI",
                "coverage": {
                    "forms": ["10-K"],
                    "dates": [],
                    "partitions": [],
                    "docs": [],
                    "gaps": [],
                    "complete": False,
                },
            },
            repo=repo,
        )


def test_reg_negatives_incomplete_stays_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:1"), repo=repo)
    out = _svc.submit_source_result(
        src,
        coverage={
            "useful_for_question": "insufficient",
            "resolved": [],
            "partially_resolved": [],
            "unresolved": ["OpenAI private terms"],
            "source_limitations": ["SEC-only"],
        },
        evidence_ids=[],
        unresolved_questions=["OpenAI private terms"],
        repo=repo,
    )
    assert out["job_status"] == "completed"
    stored = repo.get_job(src).result or {}
    assert isinstance(stored, dict)
    cov = stored.get("coverage")
    assert isinstance(cov, dict) and cov.get("useful_for_question") == "insufficient"
    assert cov.get("complete") in (None, False)


# --- PIT (3): accepted_at==known_at, TZ preserves instant, future rejected ---


def test_reg_pit_accepted_at_is_known_at() -> None:
    from typing import override

    from edgar import Filing as EdgarFiling

    from app.sec.normalization import filing_from_edgar

    class _StubFiling(EdgarFiling):
        acceptance_datetime = "2025-02-14T17:30:00Z"

        @override
        @property
        def period_of_report(self) -> str:
            return "2024-12-31"

        @override
        @property
        def document(self) -> str:
            return "primary.htm"

    filing = _StubFiling(
        cik=886982, company="Goldman Sachs", form="10-K", filing_date="2025-02-14", accession_no="0000886982-26-000001"
    )
    meta = filing_from_edgar(filing)
    assert meta.known_at == "2025-02-14T17:30:00Z"
    assert meta.accepted_at == meta.known_at


def test_reg_pit_tz_preserves_instant() -> None:
    from app.research.models import pit_violated

    assert pit_violated("2025-06-30T00:00:00+00:00", "2025-06-29T20:00:00-04:00") is False
    assert pit_violated("2025-06-30T00:00:00+00:00", "2025-06-30T01:00:00+00:00") is True


def test_reg_pit_future_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    with pytest.raises(ValueError):
        _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:9", known_at="2025-07-01T00:00:00+00:00"), repo=repo)


# --- freeze/committee (5): immutable, identical freeze ID, no source tools, new wave, decide ---
def test_reg_freeze_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses as _dc
    from datetime import datetime as _dt

    from app.research import service as _svc
    from app.research.freeze import EvidenceFreeze

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    frozen = _svc.freeze_session(sid, 1, repo=repo)
    fid = str(frozen["freeze_id"])
    stored = repo.get_freeze(fid)
    assert stored["freeze_id"] == fid
    now = _dt(2025, 6, 30, tzinfo=UTC)
    with pytest.raises(_dc.FrozenInstanceError):
        frozen_obj = EvidenceFreeze(
            freeze_id=fid, session_id=sid, wave_id=1, created_at=now, as_of=now, evidence_ids=(eid,), content_hash="h"
        )
        _frozen_write(frozen_obj, "evidence_ids", ("tampered",))
    assert repo.get_freeze(fid)["freeze_id"] == fid


def test_reg_freeze_identical_id_for_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3
    jobs = repo.list_jobs(sid)
    assert repo.get_session(sid).freeze_ids[-1] == fid
    assert all(j.wave_id == 1 for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot"))


def test_reg_committee_cannot_call_source_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bear = str(_svc.start_job(sid, "bearbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="forbids"):
        _svc.authorize_and_consume_dispatch(sid, bear, "get_sec_document", repo=repo)


def test_reg_freeze_new_wave_new_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    f1 = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    assert f1 == f"{sid}:1:freeze"
    assert f"{sid}:2:freeze" != f1


def test_reg_freeze_director_finalize_or_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    out = _svc.decide_next_wave(sid, repo=repo)
    # §3: director gate set has no budget_exhausted (deleted); runtime_exceeded stays.
    assert out["stop_reason"] in (
        "no_questions",
        "low_gain",
        "not_actionable",
        "continue",
        "max_waves",
        "jobs_exceeded",
        "runtime_exceeded",
    )


# --- synthesis (4): supported trace, inference labeled, unknown, manageable ---


def test_reg_synthesis_supported_traces_to_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid, jid, role, _committee_analysis(eid, claim_text="GS discloses OpenAI-linked exposure"), repo=repo
        )
    out = _svc.finalize_session(
        sid,
        "GS exposure is filing-backed.",
        [{"text": "GS discloses OpenAI-linked exposure", "evidence_ids": [eid]}],
        repo=repo,
    )
    assert out["freeze_id"] == f"{sid}:1:freeze"
    final = repo.get_session(sid).final_result or {}
    assert isinstance(final, dict)
    claims_raw = final.get("claims")
    assert isinstance(claims_raw, list) and claims_raw
    first = claims_raw[0]
    assert isinstance(first, dict) and first.get("evidence_ids") == [eid]


def test_reg_synthesis_inference_labeled() -> None:
    """A declared inference stays labeled inference through synthesis: a citation never upgrades it."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="OpenAI stress may widen GS spreads", evidence_ids=["EV-1"])
    assert claim.claim_type == "inference"
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[claim],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
        claims=[claim],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
        claims=[claim],
    )
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    raw_claims = synth.to_dict()["claims"]
    assert isinstance(raw_claims, list)
    rows = [r for r in raw_claims if isinstance(r, dict) and r.get("text") == claim.text]
    assert rows and all(r.get("claim_type") == "inference" for r in rows)
    assert "(inference)" in synth.answer and "observed_fact" not in synth.answer


def test_reg_synthesis_unknown_stays_unknown() -> None:
    from app.research.evals.evaluators import EvalInput, evaluate

    inp = EvalInput(
        scenario_name="gs-openai-sec-only",
        answer_text="OpenAI private revenue share: UNKNOWN (no filing discloses it).",
        evidence_ids=("EV-1",),
        requires_evidence=True,
    )
    assert evaluate(inp).passed


def test_reg_synthesis_manageable_rejected_as_fact() -> None:
    from app.research.agents import ModelOutputFailure, parse_grounded_claims

    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "GS will manageably absorb any OpenAI loss", "evidence_ids": []}]', frozen=["EV-1"]
        )


# --- evidence record_kind + claim labels (contract pins, stub-safe) ---


def test_reg_evidence_record_kind_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.research.evidence import (
        RECORD_KINDS,
        DiscoveryRecord,
        EvidenceRecord,
        discovery_only,
        substantive_records,
    )

    assert RECORD_KINDS == frozenset({"discovery", "evidence"})
    assert DiscoveryRecord(record_id="d1", session_id="s", tool="search_sec_filings", query="GS 10-K").search_id is None
    assert EvidenceRecord(record_id="e1", session_id="s", evidence_id="EV-1", claim_text="c").evidence_id == "EV-1"
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    # A scoping search records as a coverage artifact: search scope, never citable evidence.
    out = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "scoping search over GS filings",
            "claim_text": "scoping search",
            "claim_kind": "absence_observation",
            "subject": "GS",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "s1",
            "query": "GS 10-K",
            "coverage": _absence_cov(),
            "record_kind": "discovery",
        },
        repo=repo,
    )
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    assert out.get("claim_kind") == "absence_observation"
    assert "evidence_id" not in out and repo.list_evidence(sid) == []
    assert repo.list_coverage_artifacts(sid)[0].get("search_id") == "s1"
    assert discovery_only([{"record_kind": "discovery"}]) is True
    assert substantive_records([{"record_kind": "discovery"}, {"record_kind": "evidence"}]) == [
        {"record_kind": "evidence"}
    ]


def test_reg_evidence_claim_labels() -> None:
    """Claim type is declared, never inferred from wording: absent type stays inference, and
    observed_fact/contradicted need at least one freeze id (unknown may cite none)."""
    from app.research.agents import (
        CLAIM_TYPES,
        GroundedClaim,
        ModelOutputFailure,
        parse_grounded_claims,
    )

    assert tuple(CLAIM_TYPES) == ("observed_fact", "inference", "unknown", "contradicted")
    assert GroundedClaim(text="GS revenue grew", evidence_ids=["EV-1"]).claim_type == "inference"
    declared = parse_grounded_claims(
        '[{"text": "GS revenue grew", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]}]', frozen=["EV-1"]
    )
    assert declared[0].claim_type == "observed_fact"
    # Wording and citation never type a claim: an undeclared confident sentence stays inference.
    undeclared = parse_grounded_claims(
        '[{"text": "GS revenue grew per the filing, definitely observed.", "evidence_ids": ["EV-1"]}]', frozen=["EV-1"]
    )
    assert undeclared[0].claim_type == "inference"
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "GS revenue grew", "claim_type": "observed_fact", "evidence_ids": []}]', frozen=["EV-1"]
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims('[{"text": "unclear", "claim_type": "shrug", "evidence_ids": ["EV-1"]}]', frozen=["EV-1"])
    unknown = parse_grounded_claims(
        '[{"text": "OpenAI terms UNKNOWN", "claim_type": "unknown", "evidence_ids": []}]', frozen=["EV-1"]
    )
    assert unknown[0].claim_type == "unknown" and unknown[0].evidence_ids == []


def test_crap_source_policy_modes_and_errors() -> None:
    from app.research.models import resolve_source_policy, source_domain_allowed

    # Bare research_sources mapping (no outer key) resolves identically.
    bare = resolve_source_policy({"mode": "all", "sources": []})
    assert bare["mode"] == "all" and bare["allowed"] == []
    # Mode "all" admits any domain not denied; allowlist pins SEC.
    assert source_domain_allowed(bare, "anything.example") is True
    assert source_domain_allowed(None, None) is True
    sec = resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": [" SEC "]}})
    assert sec["allowed"] == ["SEC"]
    assert source_domain_allowed(sec, "SEC") is True
    assert source_domain_allowed(sec, "web") is False
    denied = {"allowed": ["SEC"], "denied": ["SEC"], "mode": "allowlist"}
    assert source_domain_allowed(denied, "SEC") is False
    with pytest.raises(ValueError, match="policy"):
        resolve_source_policy("nope")
    with pytest.raises(ValueError, match="research_sources"):
        resolve_source_policy({"research_sources": "nope"})
    with pytest.raises(ValueError, match="sources"):
        resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": []}})
    with pytest.raises(ValueError, match="sources"):
        resolve_source_policy({"research_sources": {"mode": "all", "sources": [""]}})
    with pytest.raises(ValueError, match="source_domain"):
        source_domain_allowed(sec, "  ")


def test_crap_temporal_patterns() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope

    now = _dt(2025, 6, 30, tzinfo=UTC)
    assert resolve_temporal_scope(temporal="show all history please", now=now)["mode"] == "unbounded"
    assert resolve_temporal_scope(temporal="as of 2024-03-31", now=now)["as_of"] == "2024-03-31"
    last2 = resolve_temporal_scope(temporal="last 2 years", now=now)
    assert last2["mode"] == "range" and str(last2["start"])[:4] == "2023"
    lastyr = resolve_temporal_scope(temporal="last year", now=now)
    assert (lastyr["start"], lastyr["end"]) == ("2024-01-01", "2024-12-31")
    lastq = resolve_temporal_scope(query="What drove NVDA revenue last quarter?", now=now)
    assert lastq["mode"] == "range" and (lastq["start"], lastq["end"]) == ("2025-01-01", "2025-03-31")
    thisw = resolve_temporal_scope(query="Apple news this week", now=now)
    assert thisw["mode"] == "range" and thisw["start"] == "2025-06-23" and str(thisw["end"])[:10] == "2025-06-30"
    lastw = resolve_temporal_scope(temporal="last week", now=now)
    assert lastw["mode"] == "range" and (lastw["start"], lastw["end"]) == ("2025-06-16", "2025-06-22")
    assert resolve_temporal_scope(temporal="before earnings", now=now)["mode"] == "as_of"
    assert resolve_temporal_scope(temporal="latest available data", now=now)["mode"] == "latest-available"
    assert resolve_temporal_scope(temporal="most recent filing", now=now)["mode"] == "as_of"
    with pytest.raises(ValueError, match="after end"):
        resolve_temporal_scope(temporal="between 2025-06-30 and 2025-01-01", now=now)
    with pytest.raises(ValueError, match="temporal"):
        resolve_temporal_scope(temporal="  ", now=now)
    with pytest.raises(ValueError, match="invalid calendar"):
        resolve_temporal_scope(temporal="as of 2025-02-30", now=now)


def test_crap_baseline_current_and_quarterly_picks() -> None:
    from app.research.models import select_latest_baseline

    filings: list[object] = [
        {"form": "10-Q", "known_at": "2025-03-31", "accession_no": "Q1"},
        {"form": "10-Q", "known_at": "2025-06-20", "accession_no": "Q2"},
        {"form": "8-K", "known_at": "2025-06-10", "accession_no": "K1"},
        {"form": "8-K", "known_at": "2025-06-25", "accession_no": "K2"},
        {"form": "10-K", "known_at": "2030-01-01", "accession_no": "FUTURE"},
        {"form": "DEF-14A", "known_at": "2025-01-01", "accession_no": "PROXY"},
        {"form": "10-K"},
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    q = base["quarterly_10q"]
    assert isinstance(q, dict) and q.get("accession_no") == "Q2"
    mats = base["material_8k"]
    assert isinstance(mats, list)
    first = mats[0]
    assert isinstance(first, dict) and first.get("accession_no") == "K2"
    assert all(isinstance(f, dict) and f.get("accession_no") != "FUTURE" for f in mats)


def test_crap_superseded_violation_edges() -> None:
    from app.research.models import superseded_current_violation

    assert superseded_current_violation([{"form": "8-K", "known_at": "2025-01-01"}], "X") is None
    filings: list[object] = [
        {"form": "10-K", "known_at": "2024-02-10", "accession_no": "OLD", "superseded_by": "NEW"},
        {"form": "10-K", "known_at": "2025-02-14", "accession_no": "NEW"},
    ]
    assert superseded_current_violation(filings, [], as_of="2025-06-30") is None
    assert superseded_current_violation(filings, ["NEW", 42, " "], as_of="2025-06-30") is None
    assert superseded_current_violation(filings, 42, as_of="2025-06-30") is None
    hit = superseded_current_violation(filings, " OLD ", as_of="2025-06-30")
    assert hit is not None and "NEW" in hit


def test_crap_discovery_only_edges() -> None:
    from app.research.evidence import (
        DiscoveryRecord,
        discovery_only,
        substantive_records,
    )

    assert discovery_only([]) is False
    assert discovery_only("nope") is False
    assert discovery_only([{"record_kind": "discovery"}, {"record_kind": "evidence"}]) is False
    assert discovery_only([{"metadata": {"record_kind": "discovery"}}]) is True
    assert discovery_only([DiscoveryRecord(record_id="d", session_id="s", tool="t", query="q")]) is False
    assert discovery_only([{"record_kind": "discovery", "metadata": {}}]) is True
    assert substantive_records([{"record_kind": "discovery"}, {}]) == [{}]


# ---------------------------------------------------------------------------
# Evals slice: committee invariants + finalization UX (offline fakes only).
# Same freeze across the trio; 3 distinct jobs created before the run;
# concurrent; completed-job cross-role write rejected; roles cannot mutate the
# freeze; claims resolve to frozen evidence; research_requests stay separate
# from evidence. Finalize success auto-renders a substantive structured answer
# in the same turn; a bare "finalized/N claims" with no answer fails.
# ---------------------------------------------------------------------------


def _eval_sid(repo: ResearchRepository) -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research("MSFT OpenAI exposure?", "o", as_of="2026-08-10T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _eval_item(eid: str, wave: int = 1) -> dict[str, object]:
    return {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": "MSFT Azure OpenAI-linked exposure per filing " + eid,
        "subject": "MSFT",
        "source_name": "SEC",
        "source_uri": "https://www.sec.gov/Archives/edgar/data/789790/000095017026001234/primary.htm",
        "source_record_id": "0000950170-26-001234",
        "document_name": "msft-10q-20260630.htm",
        "matching_passage": "Azure OpenAI-linked exposure is disclosed in the filing.",
        "source_handle": seam.handle_for(
            "Azure OpenAI-linked exposure is disclosed in the filing.",
            accession="0000950170-26-001234",
            document="msft-10q-20260630.htm",
        ),
        "known_at": "2026-08-01T00:00:00+00:00",
    }


def _eval_cov() -> dict[str, object]:
    return {
        "useful_for_question": "sufficient",
        "resolved": ["MSFT OpenAI exposure"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "major_entities_investigated": ["MSFT", "OpenAI"],
        "relationship_types_checked": ["investment", "commercial"],
        "forms_examined": ["10-K"],
        "exhibits_examined": ["EX-10.1"],
        "material_open_questions": [],
        "search_runs": ["s1"],
        "covered_branches": ["MSFT Azure OpenAI-linked exposure"],
    }


def _eval_trio_ids(repo: ResearchRepository, sid: str, eid: str) -> tuple[str, str, str, str]:
    from app.research import service as _svc

    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3 and len(set(created)) == 3
    return fid, str(created[0]), str(created[1]), str(created[2])


def test_eval_committee_same_freeze_and_trio_created_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    fid, stock_jid, bull_jid, bear_jid = _eval_trio_ids(repo, sid, eid)
    assert repo.get_session(sid).freeze_ids[-1] == fid
    assert len({stock_jid, bull_jid, bear_jid}) == 3
    for jid, role in ((stock_jid, "stockbot"), (bull_jid, "bullbot"), (bear_jid, "bearbot")):
        job = repo.get_job(jid)
        assert job.job_type == role and job.status == "running"
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    assert all(repo.get_job(jid).status == "completed" for jid in (stock_jid, bull_jid, bear_jid))
    assert repo.get_session(sid).freeze_ids[-1] == fid


def test_eval_committee_cross_role_write_rejected_and_freeze_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses as _dc
    from datetime import datetime as _dt

    from app.research import service as _svc
    from app.research.freeze import EvidenceFreeze

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    fid, stock_jid, _bull_jid, _bear_jid = _eval_trio_ids(repo, sid, eid)
    with pytest.raises(ValueError, match="!="):
        _svc.record_committee_analysis(sid, stock_jid, "bearbot", _committee_analysis(eid), repo=repo)
    now = _dt(2026, 8, 10, tzinfo=UTC)
    with pytest.raises(_dc.FrozenInstanceError):
        frozen_obj = EvidenceFreeze(
            freeze_id=fid, session_id=sid, wave_id=1, created_at=now, as_of=now, evidence_ids=(eid,), content_hash="h"
        )
        _frozen_write(frozen_obj, "evidence_ids", ("tampered",))
    _svc.record_committee_analysis(
        sid, stock_jid, "stockbot", _committee_analysis(eid, ["Probe Azure terms?"]), repo=repo
    )
    assert repo.get_freeze(fid)["freeze_id"] == fid


def test_eval_committee_claims_resolve_and_requests_not_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _fid, stock_jid, _bull_jid, _bear_jid = _eval_trio_ids(repo, sid, eid)
    with pytest.raises(ValueError):
        _svc.record_committee_analysis(sid, stock_jid, "stockbot", _committee_analysis("EV-NOPE"), repo=repo)
    n_evidence = len(repo.list_evidence(sid))
    _svc.record_committee_analysis(
        sid, stock_jid, "stockbot", _committee_analysis(eid, ["Probe Azure terms?"]), repo=repo
    )
    assert len(repo.list_evidence(sid)) == n_evidence


def test_eval_finalize_renders_answer_same_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.research.evals.evaluators import EvalInput, evaluate

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid, jid, role, _committee_analysis(eid, claim_text="MSFT Azure exposure"), repo=repo
        )
    _svc.decide_next_wave(sid, repo=repo)
    out = _svc.finalize_session(
        sid,
        "MSFT Azure exposure is filing-backed.",
        [{"text": "MSFT Azure exposure", "evidence_ids": [eid]}],
        repo=repo,
    )
    final = repo.get_session(sid).final_result or {}
    assert isinstance(final, dict) and str(final.get("answer", "")).strip()
    assert "filing-backed" in str(final.get("answer"))
    assert out["freeze_id"] == f"{sid}:1:freeze"
    rendered = _svc.inspect_research(sid, repo=repo)
    assert str(rendered.get("final", "") or final.get("answer", "")).strip()
    bare = EvalInput(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="finalized 1 claims",
        evidence_ids=(eid,),
        requires_evidence=True,
        finalized_claim_count=1,
        answered=False,
    )
    assert "finalized-without-answer" in evaluate(bare).violations


# ---------------------------------------------------------------------------
# Phase 16 invariants: raw-provenance evidence integrity, scoped absence,
# sufficiency envelope, committee freeze atomicity + role boundary, declared
# claim typing, the per-freeze wave gate (N > 2 never finalizes by itself),
# and finalize rendering (answer only in final_result).
# External behavior only: returned payloads, persisted rows, raised error codes.
# ---------------------------------------------------------------------------


def _inv_sid(repo: ResearchRepository, q: str = "NVDA OpenAI exposure?") -> tuple[str, str]:
    """Fresh session with its running source job: the only record_evidence boundary."""
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _typed_committee_analysis(eid: str, text: str, claim_type: str) -> dict[str, object]:
    """Rich committee envelope with one explicitly typed claim."""
    env = _committee_analysis(eid, claim_text=text)
    env["claims"] = [{"text": text, "claim_type": claim_type, "evidence_ids": [eid]}]
    return env


def test_invariant_search_hit_is_never_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search hit is navigation: no raw passage means ERR_RAW_SOURCE_REQUIRED and no persisted row."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    hit = {
        "evidence_id": f"{sid}:ev:1",
        "wave_id": 1,
        "content": "NVDA 10-Q hit: accelerated computing demand",
        "claim_text": "search hit",
        "subject": "NVDA",
        "source_name": "SEC",
        "search_id": "s1",
        "query": "NVDA demand",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED") as excinfo:
        _svc.record_evidence(sid, src, hit, repo=repo)
    assert "navigation artifact" in str(excinfo.value)
    assert repo.list_evidence(sid) == []


def test_invariant_search_id_never_passes_as_accession(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search id offered as the filing accession is a provenance mismatch; junk fails the format gate."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid, src, {**_svc_item(f"{sid}:ev:1"), "source_record_id": "s1", "search_id": "s1"}, repo=repo
        )
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT"):
        _svc.record_evidence(sid, src, {**_svc_item(f"{sid}:ev:2"), "source_record_id": "s-1"}, repo=repo)
    assert repo.list_evidence(sid) == []


def test_invariant_raw_document_is_accepted_and_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accession + document_name + passage is the accepted fact shape; a bare 18-digit run normalizes."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    out = _svc.record_evidence(
        sid,
        src,
        {**_svc_item(eid), "source_record_id": "000032019325000079"},
        repo=repo,
    )
    assert out["evidence_id"] == eid
    assert out.get("claim_kind") == "observed_fact"
    prov = out.get("provenance")
    assert isinstance(prov, dict)
    passage = "Accelerated computing demand grew in the reporting period."
    assert prov.get("kind") == "sec_source" and prov.get("accession_no") == "0000320193-25-000079"
    assert prov.get("document_name") == "nvda-20250331.htm" and prov.get("passage") == passage
    # The kernel stores the archive's bytes with the coordinates and hash it reloaded them at.
    assert (prov.get("offset"), prov.get("end")) == (0, len(passage))
    assert prov.get("text_hash") == sha256(passage.encode("utf-8")).hexdigest()
    assert out.get("record_kind") == "evidence"
    assert out.get("source_record_id") == "0000320193-25-000079"


def test_invariant_dossier_summary_is_never_observed_fact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dossier/model summaries carry no raw passage: declared observed_fact they fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    summary = {
        "evidence_id": f"{sid}:ev:1",
        "wave_id": 1,
        "content": "Dossier summary: NVDA demand is rising.",
        "claim_text": "summary finding",
        "subject": "NVDA",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, summary, repo=repo)
    cited_only = {
        **summary,
        "evidence_id": f"{sid}:ev:2",
        "source_record_id": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, cited_only, repo=repo)
    assert repo.list_evidence(sid) == []


def test_invariant_absence_observation_needs_search_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absence = search run + full coverage + no accession, recorded as coverage state, never evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    base = {
        "wave_id": 1,
        "content": "no NVDA capacity constraint disclosed",
        "claim_text": "not disclosed in the searched scope",
        "claim_kind": "absence_observation",
        "subject": "NVDA",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
        "search_id": "s1",
        "query": "NVDA capacity",
    }
    out = _svc.record_evidence(sid, src, {**base, "coverage": _absence_cov()}, repo=repo)
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    artifact_id = str(out.get("artifact_id"))
    assert artifact_id.startswith(f"{sid}:cov:")
    assert out.get("evidence_ids") == []
    # Not evidence: neither the ledger nor the session evidence set grows...
    assert repo.list_evidence(sid) == []
    assert repo.get_session(sid).evidence_ids == []
    # ...and the artifact stays inspectable as coverage state.
    artifacts = repo.list_coverage_artifacts(sid)
    assert [a.get("artifact_id") for a in artifacts] == [artifact_id]
    assert artifacts[0].get("search_id") == "s1" and artifacts[0].get("coverage") == _absence_cov()
    coverage_store = repo.resource_stores(sid)["coverage"]
    stored = coverage_store[artifact_id]
    assert isinstance(stored, dict) and stored.get("query") == "NVDA capacity"
    # One scope is one artifact: a repeat is an idempotent write, not a second record.
    again = _svc.record_evidence(sid, src, {**base, "coverage": _absence_cov()}, repo=repo)
    assert again.get("accepted") is False and again.get("artifact_id") == artifact_id
    assert len(repo.list_coverage_artifacts(sid)) == 1
    bad_covs: list[dict[str, object]] = [
        {"forms": ["10-K"]},  # missing keys
        _absence_cov(pagination_complete="yes"),  # flag is not a bool
        _absence_cov(complete=None),  # flag is not a bool
    ]
    for i, bad_cov in enumerate(bad_covs):
        with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
            _svc.record_evidence(
                sid,
                src,
                {**base, "query": f"NVDA capacity {i}", "coverage": bad_cov},
                repo=repo,
            )
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(sid, src, {**base, "query": "", "coverage": _absence_cov()}, repo=repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            src,
            {
                **base,
                "source_record_id": "0000320193-25-000079",
                "coverage": _absence_cov(),
            },
            repo=repo,
        )
    # The same absence wording is not a fact without raw provenance.
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, {**base, "claim_kind": "observed_fact"}, repo=repo)
    assert repo.list_evidence(sid) == []  # nothing retroactively became evidence
    assert len(repo.list_coverage_artifacts(sid)) == 1


def test_invariant_claim_kind_never_routed_by_wording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative wording is gated by provenance, not lexis: raw provenance accepts it, none fails closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    wording = "No OpenAI exposure exists anywhere in the GS filings and the business is unaffected."
    eid = f"{sid}:ev:1"
    out = _svc.record_evidence(
        sid,
        src,
        {
            **_svc_item(eid, passage=wording),
            "claim_text": wording,
            "content": wording,
            "matching_passage": wording,
        },
        repo=repo,
    )
    assert out.get("claim_kind") == "observed_fact"
    prov = out.get("provenance")
    assert isinstance(prov, dict) and prov.get("kind") == "sec_source" and prov.get("passage") == wording
    bare = {
        "evidence_id": f"{sid}:ev:2",
        "wave_id": 1,
        "content": wording,
        "claim_text": wording,
        "subject": "GS",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT|ERR_RAW_SOURCE_REQUIRED") as excinfo:
        _svc.record_evidence(sid, src, bare, repo=repo)
    assert "negat" not in str(excinfo.value).lower()  # no lexical negativity gate anywhere


def test_invariant_unknown_stays_unknown_and_absence_needs_coverage() -> None:
    """Absence renders as searched-scope language; a generic unknown never becomes an absence claim."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import (
        SEC_SCOPE_ABSENCE,
        scoped_absence,
        synthesize_final,
    )
    from app.tool_render import render_final_result

    raw_claim = "Redeployment economics cannot be determined from the filings available"
    assert scoped_absence(raw_claim) == f"{SEC_SCOPE_ABSENCE}: {raw_claim}"
    assert scoped_absence("  ").startswith(SEC_SCOPE_ABSENCE)
    unknown = GroundedClaim(text=raw_claim, claim_type="unknown", evidence_ids=[])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[unknown],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
        claims=[unknown],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
        claims=[unknown],
    )
    # No coverage artifacts: the unknown stays an unknown, and no absence claim is invented from it.
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    assert synth.absence_observations == []
    assert raw_claim in synth.unknowns
    assert not any(SEC_SCOPE_ABSENCE in item for item in synth.unknowns)
    # With a coverage artifact, absence text is that artifact's, still scoped.
    scoped = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        absence_observations=["no OpenAI funding terms disclosed"],
    )
    assert scoped.absence_observations == [f"{SEC_SCOPE_ABSENCE}: no OpenAI funding terms disclosed"]
    assert f"- {SEC_SCOPE_ABSENCE}" in scoped.answer
    assert raw_claim in scoped.unknowns  # the coverage artifact does not reclassify the unknown
    rendered = render_final_result(scoped.to_dict())
    assert SEC_SCOPE_ABSENCE in rendered


def test_invariant_direct_evidence_is_canonical_not_committee_labels() -> None:
    """`observed_fact` on a committee claim never renders as direct evidence; canonical rows do."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    labeled = GroundedClaim(
        text="Microsoft will absorb the OpenAI loss",
        claim_type="observed_fact",
        evidence_ids=["EV-1"],
    )
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[labeled],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
    )
    observation = {
        "evidence_id": "EV-1",
        "claim_text": "OpenAI investment carrying value",
        "record_kind": "evidence",
        "source_name": "SEC-10-K",
        "provenance": {
            "kind": "sec_source",
            "accession_no": "0000320193-25-000079",
            "document_name": "msft-10k.htm",
            "passage": "OpenAI carrying value was $13.3B",
        },
    }
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        observations=[observation],
    )
    direct = [str(row.get("text")) for row in synth.direct_evidence]
    assert len(direct) == 1 and "msft-10k.htm" in direct[0] and "OpenAI carrying value was $13.3B" in direct[0]
    assert all(labeled.text not in line for line in direct)  # a model label is not direct evidence
    assert synth.direct_evidence[0]["evidence_ids"] == ["EV-1"]
    # The committee reading still renders (as an interpretation), so nothing is silently dropped.
    assert [row["text"] for row in synth.first_order_effects] == [labeled.text]
    assert "What the evidence directly shows" in synth.answer and "msft-10k.htm" in synth.answer
    # Without canonical observations the section is empty rather than model-labeled.
    bare = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    assert bare.direct_evidence == []


def test_invariant_sufficient_needs_structured_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`sufficient` needs the full structured envelope; the legacy bare-sufficient fallback is gone."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    legacy: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA demand"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
    }
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED") as excinfo:
        _svc.submit_source_result(src, coverage=legacy, evidence_ids=[eid], repo=repo)
    assert "major_entities_investigated" in str(excinfo.value)  # the missing envelope is named
    assert repo.get_job(src).status == "running"  # a failed gate never completes the job
    assert repo.get_session(sid).status not in ("completed", "synthesizing", "failed")


def test_invariant_sufficient_rejects_residual_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any residual branch/question in a sufficient envelope fails; a clean envelope completes."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    residuals: list[tuple[str, object]] = [
        ("remaining_branches", ["private-lab terms"]),
        ("material_open_questions", ["Is the exposure hedged?"]),
        ("routes_unsearched", ["8-K exhibits"]),
        ("major_entities_missing", ["OpenAI"]),
    ]
    for key, value in residuals:
        cov: dict[str, object] = dict(_reg_cov())
        cov[key] = value
        with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
            _svc.submit_source_result(src, coverage=cov, evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
        _svc.submit_source_result(
            src, coverage=_reg_cov(), evidence_ids=[eid], unresolved_questions=["still open?"], repo=repo
        )
    assert repo.get_job(src).status == "running"
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    # Unknown search_run ids stay advisory (readable SEC ledger -> warnings, never a failure).
    assert isinstance(out.get("warnings"), list)
    dossiers = repo.list_dossiers(sid)
    assert dossiers
    stored_cov = dossiers[-1].get("coverage")
    assert isinstance(stored_cov, dict)
    assert stored_cov.get("source_domain") == "SEC" and stored_cov.get("source_sufficiency") == "sufficient"


def test_invariant_committee_trio_running_before_roles_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three committee jobs exist RUNNING before any role finishes; all share one freeze."""
    import sqlite3

    from app.research.repository import get_research_db_path

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    db = str(get_research_db_path())

    def _committee_statuses() -> dict[str, str]:
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT job_type, status FROM jobs WHERE job_type IN ('stockbot','bullbot','bearbot')"
            ).fetchall()
        return {str(job_type): str(status) for job_type, status in rows}

    snapshots: list[dict[str, str]] = []

    def _model(prompt: str) -> str:
        if "Temporary assignment" not in prompt:  # a committee role call, not a scout
            snapshots.append(_committee_statuses())
        return _grounded(prompt)

    out = _run_wave(repo, _model)
    assert out["stop_reason"] == "complete:wave1"
    assert len(snapshots) == 3  # one snapshot per role model call
    assert all(len(snap) == 3 and set(snap.values()) == {"running"} for snap in snapshots)
    sid = out["session_id"]
    assert isinstance(sid, str)
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis

    assert isinstance(stock, StockbotAnalysis) and isinstance(bull, BullAnalysis) and isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    stored = {
        str((repo.get_job(j.job_id).result or {}).get("freeze_id"))
        for j in repo.list_jobs(sid)
        if j.job_type in ("stockbot", "bullbot", "bearbot")
    }
    assert stored == {out["freeze_id"]}


def test_invariant_committee_roles_stay_out_of_the_tool_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every committee role job is freeze-only: SEC/research tools and evidence writes are forbidden."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3
    for jid in created:
        for tool in ("search_sec_filings", "get_sec_document", "get_sec_filing", "research_add_evidence", "search_web"):
            with pytest.raises(ValueError, match="forbids"):
                _svc.authorize_and_consume_dispatch(sid, str(jid), tool, repo=repo)
        with pytest.raises(ValueError):
            _svc.record_evidence(sid, str(jid), _svc_item(f"{sid}:ev:2"), repo=repo)


def test_invariant_committee_follow_ups_route_through_director_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committee requests drive the next-wave decision (never evidence) or stop at the gate."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    follow_up = "Do the exhibits cover the Azure terms?"
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid, jid, role, _committee_analysis(eid, [follow_up] if i == 0 else []), repo=repo
        )
    assert len(repo.list_evidence(sid)) == 1  # requests never become evidence
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_question"] == follow_up
    sess = repo.get_session(sid)
    assert sess.current_wave == 2 and sess.status == "targeted_research" and sess.targeted_question == follow_up
    # A request outside the SEC lane routes to a non-authorized gate instead of a wave.
    sid2, src2 = _inv_sid(repo, "GS OpenAI exposure?")
    eid2 = f"{sid2}:ev:1"
    _svc.record_evidence(sid2, src2, {**_svc_item(eid2), "subject": "GS"}, repo=repo)
    _svc.complete_job(src2, {}, repo=repo)
    _svc.freeze_session(sid2, 1, repo=repo)
    off_lane = {"question": "What does the press say about the terms?", "suggested_source": "web"}
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid2, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid2, jid, role, _committee_analysis(eid2, [off_lane] if i == 0 else []), repo=repo
        )
    out2 = _svc.decide_next_wave(sid2, repo=repo)
    assert out2["authorized"] is False and out2["stop_reason"] == "not_actionable"
    assert repo.get_session(sid2).status == "synthesizing"
    assert len(repo.list_evidence(sid2)) == 1


def test_invariant_insufficient_coverage_drives_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An honest insufficient dossier with actionable SEC residuals authorizes the next wave from its
    coverage alone (the committee asked nothing); without residuals the run settles with unknowns."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    question = "NVDA OpenAI exposure?"
    sid, src = _inv_sid(repo, question)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    residual = "OpenAI private-lab terms"
    _svc.submit_source_result(
        src,
        coverage={
            "useful_for_question": "insufficient",
            "major_entities_missing": [residual],
            "unresolved": ["Is the exposure hedged?"],
        },
        evidence_ids=[eid],
        unresolved_questions=["Is the exposure hedged?"],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_domain"] == "SEC"
    assert out["targeted_question"] == f"{question} :: targeted follow-up: {residual}"
    assert residual in str(out["reason_detail"])
    sess = repo.get_session(sid)
    assert sess.status == "targeted_research" and sess.current_wave == 2
    assert sess.targeted_question == out["targeted_question"]
    # No actionable SEC residual: coverage challenges nothing, the committee gate settles the run.
    sid2, src2 = _inv_sid(repo, "AMD private-lab terms?")
    eid2 = f"{sid2}:ev:1"
    _svc.record_evidence(sid2, src2, _svc_item(eid2), repo=repo)
    _svc.submit_source_result(
        src2,
        coverage={
            "useful_for_question": "insufficient",
            "source_limitations": ["SEC-only: no private-issuer filings"],
        },
        evidence_ids=[eid2],
        repo=repo,
    )
    _svc.freeze_session(sid2, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid2, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid2, jid, role, _committee_analysis(eid2), repo=repo)
    out2 = _svc.decide_next_wave(sid2, repo=repo)
    assert out2["authorized"] is False and out2["stop_reason"] == "no_questions"
    assert "coverage challenge" not in str(out2["reason_detail"])
    assert repo.get_session(sid2).status == "synthesizing"


def test_invariant_claims_only_envelope_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A claims+follow_ups-only committee envelope fails ERR_COMMITTEE_ENVELOPE_INCOMPLETE everywhere."""
    import json as _json

    from app.research import service as _svc
    from app.research.agents import ModelOutputFailure
    from app.research.agents.stockbot import run_stockbot

    old_shape = _json.dumps({"claims": [{"text": "finding", "evidence_ids": ["EV-1"]}], "follow_ups": []})
    with pytest.raises(ModelOutputFailure, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        run_stockbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: old_shape,
            evidence_text="[EV-1] a",
        )
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    jid = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        _svc.record_committee_analysis(
            sid, jid, "stockbot", {"claims": [{"text": "finding", "evidence_ids": [eid]}], "follow_ups": []}, repo=repo
        )
    assert repo.get_job(jid).status == "running"  # the incomplete envelope never closes the job


def test_invariant_claim_type_is_declared_never_promoted() -> None:
    """Absent type stays inference; observed_fact/contradicted need a freeze id; citations never upgrade."""
    from app.research.agents import ModelOutputFailure
    from app.research.agents.stockbot import run_stockbot

    def _run(model: Callable[[str], str]) -> StockbotAnalysis:
        return run_stockbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=model,
            evidence_text="[EV-1] a",
        )

    undeclared = _run(
        lambda prompt: _committee_output([{"text": "Confirmed: revenue grew.", "evidence_ids": ["EV-1"]}])
    )
    assert [c.claim_type for c in undeclared.claims] == ["inference"]
    assert undeclared.claims[0].evidence_ids == ["EV-1"]
    declared = _run(
        lambda prompt: _committee_output(
            [{"text": "Revenue grew.", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]}]
        )
    )
    assert [c.claim_type for c in declared.claims] == ["observed_fact"]
    with pytest.raises(ModelOutputFailure, match="uncited"):
        _run(
            lambda prompt: _committee_output(
                [{"text": "Revenue grew.", "claim_type": "observed_fact", "evidence_ids": []}]
            )
        )
    with pytest.raises(ModelOutputFailure, match="EV-999"):
        _run(
            lambda prompt: _committee_output(
                [{"text": "Revenue grew.", "claim_type": "observed_fact", "evidence_ids": ["EV-999"]}]
            )
        )


def test_invariant_director_waves_continue_past_two() -> None:
    """A material question advances wave N -> N+1 forever on new evidence; a zero-novelty wave stops
    the run only once nothing unexplored is left, or at an explicitly configured operator limit."""
    from app.research.agents import ResearchRequest
    from app.research.director import (
        DirectorBudgets,
        DirectorDeps,
        Wave1Result,
        decide_next_wave,
    )
    from app.research.synthesis.committee import CommitteeDisagreement

    stops: list[tuple[str, str]] = []

    def _committee(_sid: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        raise AssertionError("this gate test never runs a committee")

    deps = DirectorDeps(
        create_session=lambda q, a: "rs:x",
        fetch_wave_evidence=lambda s: [],
        create_freeze=lambda s: "F1",
        run_committee=_committee,
        record_stop=lambda s, r: stops.append((s, r)),
    )

    def _wave(wave: int, question: str) -> Wave1Result:
        request = ResearchRequest(
            question=question,
            why_material="unresolved in the last freeze",
            requested_source_domain="SEC",
            expected_gain="medium",
        )
        disagreement = CommitteeDisagreement(
            session_id="rs:x",
            wave_id=wave,
            freeze_id=f"F{wave}",
            requested_research=[request],
        )
        return Wave1Result(
            session_id="rs:x",
            wave_id=wave,
            freeze_id=f"F{wave}",
            evidence_ids=["EV-1"],
            disagreement=disagreement,
            question="Is GS exposed?",
        )

    novelty = {
        "new_raw_documents": 1,
        "new_evidence_records": 2,
        "new_relationships": 1,
        "resolved_questions": 1,
        "zero_novelty_waves": 0,
        "duplicate_actions_blocked": 0,
    }
    questions = {
        "What drove the Q2 delta?": 2,
        "Which exhibits cover the terms?": 3,
        "How does the hedge work?": 4,
        "What if the hedge fails?": 5,
    }
    for question, wave in questions.items():
        decision = decide_next_wave(
            _wave(wave - 1, question),
            deps=deps,
            budgets=DirectorBudgets(),
            waves_used=wave - 1,
            novelty=novelty,
        )
        assert decision.authorized is True and decision.stop_reason == "continue"
        assert decision.targeted_question == question and decision.targeted_domain == "SEC"
        assert len(stops) == wave - 1
    # Convergence, not a wave number or a streak: a zero-novelty wave with the committee question
    # still unanswered keeps the branch alive, however long the streak already is.
    zero = dict.fromkeys(
        (
            "new_raw_documents",
            "new_evidence_records",
            "new_relationships",
            "resolved_questions",
        ),
        0,
    )
    zero.update({"zero_novelty_waves": 3, "duplicate_actions_blocked": 0})
    still_open = decide_next_wave(
        _wave(5, "What else?"),
        deps=deps,
        budgets=DirectorBudgets(),
        waves_used=5,
        novelty=zero,
    )
    assert still_open.authorized is True and still_open.stop_reason == "continue"
    answered = _wave(5, "What else?")
    answered.disagreement = CommitteeDisagreement(session_id="rs:x", wave_id=5, freeze_id="F5")
    stopped = decide_next_wave(answered, deps=deps, budgets=DirectorBudgets(), waves_used=5, novelty=zero)
    assert stopped.authorized is False and stopped.stop_reason == "no_novelty"
    # An explicit operator runaway guard stops a branch that still has questions, at its own count.
    limited = decide_next_wave(
        _wave(5, "What else?"),
        deps=deps,
        budgets=DirectorBudgets(zero_novelty_limit=2),
        waves_used=5,
        novelty={**zero, "zero_novelty_waves": 2},
    )
    assert limited.authorized is False and limited.stop_reason == "no_novelty"
    assert "zero_novelty_limit=2" in limited.reason_detail


def test_invariant_finalize_card_never_duplicates_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The answer lives once in final_result; the finalize card is a short confirmation, not the answer."""
    from app.research import service as _svc
    from app.tool_render import render_tool_result

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    _svc.decide_next_wave(sid, repo=repo)
    answer = "MSFT Azure exposure is filing-backed and the agreement terms stay undisclosed."
    out = _svc.finalize_session(sid, answer, [{"text": "MSFT Azure exposure", "evidence_ids": [eid]}], repo=repo)
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict)
    assert final.get("answer") == answer and str(final.get("content", "")).strip()
    assert final.get("freeze_id") == f"{sid}:1:freeze"
    card = render_tool_result(out)
    assert "Research finalized" in card
    assert f"{sid}:1:freeze" in card
    assert "Claims:" in card and "impact channels:" in card
    assert answer not in card and "filing-backed" not in card
    assert "Answer delivered separately" in card
    # The answer is stored once (final_result.answer); the card only counts what was frozen.
    assert str(final.get("content", "")).startswith("Bottom line:")


# --- property invariants: accession + typed-provenance parsing (plan: raw-source-boundary) ---

_ACCESSION_ALPHABET = "0123456789abcdefABCDEF-"


@settings(max_examples=200, derandomize=True)
@given(st.text(max_size=24))
def test_prop_accession_normalization_is_idempotent(text: str) -> None:
    """Any accepted accession normalizes to the canonical dashed form and stays stable."""
    from app.research.evidence import ACCESSION_RE, normalize_accession

    try:
        canonical = normalize_accession(text)
    except ValueError:
        return
    assert ACCESSION_RE.match(canonical)
    assert normalize_accession(canonical) == canonical


@settings(max_examples=200, derandomize=True)
@given(
    st.text(alphabet=_ACCESSION_ALPHABET, max_size=24), st.text(alphabet="0123456789abcdef", min_size=1, max_size=12)
)
def test_prop_provenance_accepts_exactly_canonical_accessions(blob: str, search_id: str) -> None:
    """sec_source provenance accepts exactly what accession normalization accepts; search runs stay search runs."""
    from app.research.evidence import (
        EvidenceIntegrityError,
        normalize_accession,
        validate_provenance,
    )

    item: dict[str, object] = {"kind": "sec_source", "accession_no": blob, "document_name": "d.htm", "passage": "p"}
    try:
        canonical = normalize_accession(blob)
    except ValueError:
        with pytest.raises(EvidenceIntegrityError):
            validate_provenance(item)
        return
    assert validate_provenance(item)["accession_no"] == canonical
    assert validate_provenance({"kind": "search_run", "search_id": search_id, "query": "q"})["kind"] == "search_run"


def test_telemetry_derived_from_persisted_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan Phase 13: search/evidence telemetry derives from persisted rows + journal, never model counts."""
    from app.research import service as _svc
    from app.research.journal import append_event

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    repo.save_event(
        append_event(
            sid,
            "tool.completed",
            "runner",
            "runner",
            {"tool": "search_sec_filings", "args": {"query": "NVDA OpenAI supply"}},
        )
    )
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["private contract terms"],
        "source_limitations": ["SEC-only"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    # Journal-derived searches/queries (no coverage search_runs in this submission).
    assert telemetry.get("searches_count") == 1
    assert telemetry.get("queries_attempted") == ["NVDA OpenAI supply"]
    # Row-derived document/filing/evidence counts.
    assert telemetry.get("evidence_records") == 1
    assert telemetry.get("filings_opened") == 1
    assert telemetry.get("documents_opened") == 1 and telemetry.get("raw_documents_used") == 1
    # Relationships are relationship rows; an evidence row is not a relationship.
    assert "material_relationships_found" not in telemetry
    assert telemetry.get("telemetry_gaps") == []


def test_telemetry_gaps_mark_underivable_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No persisted search activity -> searches/queries are null and named in telemetry_gaps, never guessed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _sid, src = _svc_sid(repo)
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["nothing yet"],
        "source_limitations": ["SEC-only"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry.get("searches_count") is None and telemetry.get("queries_attempted") is None
    gaps = telemetry.get("telemetry_gaps")
    assert isinstance(gaps, list) and "searches_count" in gaps and "queries_attempted" in gaps
    assert telemetry.get("evidence_records") == 0


def test_accession_forms_normalize_to_canonical() -> None:
    """Canonical + bare-18-digit forms normalize; search-id-shaped values are rejected outright."""
    from app.research.evidence import normalize_accession

    assert normalize_accession("0000320193-25-000079") == "0000320193-25-000079"
    assert normalize_accession("000032019325000079") == "0000320193-25-000079"
    for bad in ("7768855a3f91", "0000320193-25-00007", "0000320193/25/000079", "", "0000320193-25-000079x"):
        with pytest.raises(ValueError):
            normalize_accession(bad)


# ---------------------------------------------------------------------------
# CRAP gate: novelty/director-gate walk, telemetry signal derivation, committee
# trio atomicity, claim shaping. External behavior only: returned payloads,
# persisted rows, raised errors.
# ---------------------------------------------------------------------------


def _nov_row(
    sid: str,
    eid: str,
    wave: int,
    *,
    accession: str = "",
    document: str = "",
    record_id: str = "",
    subject: str = "NVDA",
    digest: str | None = None,
) -> dict[str, object]:
    """Persisted evidence row for the novelty walk: raw provenance, record-id fallback, or no identity."""
    row: dict[str, object] = {
        "evidence_id": eid,
        "session_id": sid,
        "wave_id": wave,
        "subject": subject,
        "content_hash": digest or f"h-{eid}",
    }
    if accession:
        row["provenance"] = {"kind": "sec_source", "accession_no": accession, "document_name": document, "passage": "p"}
    if record_id:
        row["source_record_id"] = record_id
        row["source_name"] = "SEC 10-K"
    return row


def test_gate_novelty_walks_rows_relationships_and_streak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Novelty counts only documents/rows the run does not already hold; dossier relationships count once."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:1", 1, accession="0000320193-25-000079", document="a.htm"))
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:2", 1, record_id="r-1"))  # accession fallback identity
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:3", 1, subject="AMD"))  # no identity at all
    repo.save_evidence(
        _nov_row(sid, f"{sid}:ev:4", 2, accession="0000320193-25-000079", document="a.htm", digest=f"h-{sid}:ev:1")
    )  # re-read of a held document
    repo.save_evidence(
        _nov_row(sid, f"{sid}:ev:5", 2, accession="0000320193-25-000081", document="b.htm", subject="AMD")
    )
    repo.save_event(append_event(sid, "research_loop_detected", "runner", "runner", {}))
    repo.save_event(append_event(sid, "wave.novelty", "runner", "runner", {"zero_novelty_actions": 2}))
    found = dataclasses.replace(repo.get_session(sid), current_wave=2)
    repo.save_session(found)
    novelty = _svc._gate_novelty(repo, found)
    assert set(novelty) == {
        "new_raw_documents",
        "new_evidence_records",
        "new_entities",
        "new_relationships",
        "new_material_claims",
        "resolved_questions",
        "new_questions",
        "zero_novelty_waves",
        "duplicate_actions_blocked",
        "zero_novelty_actions",
    }
    # Only ev:5 (new document) is new; ev:4 is the same document+hash the run already held.
    assert novelty["new_raw_documents"] == 1 and novelty["new_evidence_records"] == 1
    assert novelty["new_entities"] == 0  # NVDA + AMD were already subjects of the prior wave
    assert novelty["new_relationships"] == 0 and novelty["new_material_claims"] == 0
    assert novelty["resolved_questions"] == 0 and novelty["new_questions"] == 0
    assert novelty["zero_novelty_waves"] == 0  # wave 2 contributed a fresh row
    assert novelty["duplicate_actions_blocked"] == 1 and novelty["zero_novelty_actions"] == 2

    # A dossier relationship is a persisted novelty signal; a re-listed one is not new again.
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:2:sec",
            "session_id": sid,
            "wave_id": 2,
            "coverage": {},
            "relationships": [{"kind": "supplier"}, "junk", {"kind": "customer"}],
        }
    )
    with_rels = _svc._gate_novelty(repo, found)
    assert with_rels["new_relationships"] == 2  # the non-mapping entry is not a relationship
    assert with_rels["new_evidence_records"] == 1
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {},
            "relationships": [{"kind": "supplier"}],
        }
    )
    # A relationship any other wave already lists is not fresh for the current wave.
    assert _svc._gate_novelty(repo, found)["new_relationships"] == 1

    # A wave with no evidence rows counts into the zero-novelty streak, back to the last productive wave.
    ahead = dataclasses.replace(repo.get_session(sid), current_wave=4)
    assert _svc._gate_novelty(repo, ahead)["zero_novelty_waves"] == 2  # waves 4 and 3 added nothing

    # Unreadable dossier storage means no relationship signal, never a guessed one.
    def _boom(_session_id: str) -> list[dict[str, object]]:
        raise RuntimeError("dossier storage unavailable")

    monkeypatch.setattr(repo, "list_dossiers", _boom)
    degraded = _svc._gate_novelty(repo, found)
    assert degraded["new_relationships"] == 0 and degraded["new_evidence_records"] == 1


def test_telemetry_ledger_unresolvable_and_journal_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable search_runs ledger falls back to journal queries; an unreadable ledger does the same."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    repo.save_event(
        append_event(
            sid, "tool.completed", "runner", "runner", {"tool": "search_sec_filings", "args": {"query": "NVDA 10-K"}}
        )
    )
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "search_runs": ["sr:not-persisted"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry.get("searches_count") == 1  # the coverage's search ids, not the journal call
    assert telemetry.get("queries_attempted") == ["NVDA 10-K"]  # journal fallback: the ledger holds no query
    assert telemetry.get("telemetry_gaps") == []


def test_telemetry_no_coverage_relationships_and_storage_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No persisted coverage/identity is a named gap; relationships come from dossiers or are omitted."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_evidence({"evidence_id": f"{sid}:ev:1", "session_id": sid, "wave_id": 1, "subject": "NVDA"})
    bare = _svc._derive_telemetry(repo, sid)
    assert bare.get("branches_covered") is None and bare.get("branches_remaining") is None
    assert bare.get("filings_opened") is None and bare.get("documents_opened") is None
    assert bare.get("raw_documents_used") is None and bare.get("evidence_records") == 1
    gaps = bare.get("telemetry_gaps")
    assert isinstance(gaps, list)
    assert {
        "branches_covered",
        "branches_remaining",
        "filings_opened",
        "documents_opened",
        "raw_documents_used",
        "searches_count",
        "queries_attempted",
    } <= set(gaps)
    assert "material_relationships_found" not in bare  # no dossier: the key is omitted, never zero

    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {},
            "relationships": [{"kind": "supplier"}, "junk"],
        }
    )
    with_rels = _svc._derive_telemetry(repo, sid)
    assert with_rels.get("material_relationships_found") == 1
    assert with_rels.get("evidence_records") == 1

    # The coverage read succeeds, the relationship read then fails: omit the key, keep the rest.
    inner = repo.list_dossiers
    calls = itertools.count()

    def _flaky(session_id: str) -> list[dict[str, JSONValue]]:
        if next(calls) == 0:
            return inner(session_id)
        raise RuntimeError("dossier storage unavailable")

    monkeypatch.setattr(repo, "list_dossiers", _flaky)
    degraded = _svc._derive_telemetry(repo, sid)
    assert "material_relationships_found" not in degraded
    assert degraded.get("evidence_records") == 1


def test_committee_jobs_partial_trio_rollback_leaves_reused_role_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trio failing mid-creation fails only the jobs this call started; a reused role is left running."""
    from app.research import service as _svc
    from app.research.models import FailureCategory

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    found = repo.get_session(sid)
    cur, preexisting = _jobs.create_job(found, repo.list_jobs(sid), job_type="bullbot", owner="pi", wave_id=1)
    repo.save_session(cur)
    repo.save_job(_jobs.start_job(preexisting))
    real_start = _jobs.start_job
    calls = itertools.count()

    def _flaky(job: Job) -> Job:
        if next(calls) == 1:
            raise RuntimeError("start failed")
        return real_start(job)

    monkeypatch.setattr(_jobs, "start_job", _flaky)
    with pytest.raises(RuntimeError, match="start failed"):
        _svc.create_committee_jobs(sid, 1, repo=repo)
    committee = {j.job_type: j for j in repo.list_jobs(sid) if j.job_type in ("stockbot", "bullbot", "bearbot")}
    assert set(committee) == {"stockbot", "bullbot"}  # bearbot never persisted
    assert committee["bullbot"].job_id == preexisting.job_id and committee["bullbot"].status == "running"
    rolled_back = committee["stockbot"]
    assert rolled_back.status == "failed"
    failure = rolled_back.failure
    assert failure is not None and failure.category == FailureCategory.COMMITTEE_DEADLOCK.value
    assert failure.message == "partial trio: RuntimeError"

    # A retry reuses the running role, creates the missing one, and flags the pending freeze.
    monkeypatch.setattr(_jobs, "start_job", real_start)
    retry = _svc.create_committee_jobs(sid, 1, repo=repo)
    retried = retry["jobs"]
    assert isinstance(retried, list) and len(retried) == 3 and len(set(retried)) == 3
    assert retried[0] == rolled_back.job_id and retried[1] == preexisting.job_id
    verb = retry["pending_next_action"]
    assert isinstance(verb, dict) and verb.get("freeze_pending") == f"{sid}:1:freeze"


def test_finalize_claims_shapes_and_guards() -> None:
    """Caller claims narrow to text+evidence_ids; every guard keeps its pinned ValueError code."""
    from app.research import service as _svc
    from app.research.agents import ModelOutputFailure

    shaped = _svc._finalize_claims(
        "sid",
        [
            {
                "claim": "  inference from the freeze  ",
                "evidence_ids": ("EV-1",),
                "claim_type": None,
            },
            {
                "claim_text": "declared fact",
                "evidence_ids": ["EV-1"],
                "claim_type": "observed_fact",
            },
        ],
        ["EV-1"],
    )
    assert [(c.text, c.claim_type, c.evidence_ids) for c in shaped] == [
        ("inference from the freeze", "inference", ["EV-1"]),
        ("declared fact", "observed_fact", ["EV-1"]),
    ]
    # A claim with no text under any alias is never silently grounded.
    with pytest.raises(ModelOutputFailure, match="non-empty text"):
        _svc._finalize_claims("sid", [{"evidence_ids": ["EV-1"]}], ["EV-1"])
    not_a_list: object = {"text": "x"}
    with pytest.raises(ValueError, match="must be a list"):
        _svc._finalize_claims("sid", not_a_list, ["EV-1"])
    with pytest.raises(ValueError, match="non-empty grounded"):
        _svc._finalize_claims("sid", [], ["EV-1"])
    non_mapping: object = "x"
    with pytest.raises(ValueError, match="each claim must be a mapping"):
        _svc._finalize_claims("sid", [non_mapping], ["EV-1"])
    with pytest.raises(ValueError, match="JSON-able"):
        _svc._finalize_claims("sid", [{"text": object(), "evidence_ids": ["EV-1"]}], ["EV-1"])
    # A coverage artifact id is coverage state: it can never ground a claim, because
    # the freeze's citable ids are ledger evidence ids only.
    artifact_id = "rs:sid:cov:9d3f1a2b"
    with pytest.raises(ModelOutputFailure, match="unknown evidence id"):
        _svc._finalize_claims(
            "sid",
            [{"text": "absence as a finding", "evidence_ids": [artifact_id]}],
            ["EV-1"],
        )


def test_unbounded_as_of_is_no_cutoff_for_every_pit_gate() -> None:
    """`unbounded` means no cutoff: dated documents stay eligible and never `pit_violated`."""
    from app.research.agents.scout import _is_pit_eligible
    from app.research.models import NO_CUTOFF_AS_OF, pit_unverified, pit_violated

    assert NO_CUTOFF_AS_OF == frozenset({"unbounded"})
    dated = "2011-03-16T16:33:51+00:00"
    # The live defect: pit_violated raised on the sentinel, so every dated document was rejected.
    assert pit_violated("unbounded", dated) is False
    assert pit_unverified("unbounded", None) is False
    assert _is_pit_eligible(dated, "unbounded") is True
    assert _is_pit_eligible(None, "unbounded") is True
    # Bounded scopes keep the strict rule.
    assert pit_violated("2025-06-30T00:00:00+00:00", "2026-01-01T00:00:00+00:00") is True
    assert _is_pit_eligible("2025-01-01T00:00:00+00:00", "2025-06-30T00:00:00+00:00") is True
    assert _is_pit_eligible("2026-01-01T00:00:00+00:00", "2025-06-30T00:00:00+00:00") is False
    assert _is_pit_eligible(None, "2025-06-30T00:00:00+00:00") is False


def test_counterparty_queries_are_assigned_unscoped() -> None:
    """A private counterparty's queries reach the filings scout as global (no ticker/cik) queries."""
    from app.research.agents.scout import build_scout_prompt
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
        decompose_question,
    )

    ctx = build_research_context("What happens to Microsoft if OpenAI goes bankrupt?", ["MSFT"])
    groups = _grouped_queries(ctx)
    assert "OpenAI" in groups["cp"]
    assignments = decompose_question(
        "What happens to Microsoft if OpenAI goes bankrupt?",
        session_id="rs:t",
        as_of="2026-08-10",
        tickers=["MSFT"],
        dispatch=lambda name, args: {},
    )
    filings = next(a for a in assignments if a.role == "filings")
    assert "OpenAI" in filings.unscoped_queries
    prompt = build_scout_prompt(filings)
    assert "NO ticker and NO cik" in prompt
    # Other scouts keep their scoped assignment: no global counterparty work.
    assert all(not a.unscoped_queries for a in assignments if a.role != "filings")
    # A question without a counterparty assigns no global queries.
    plain = decompose_question(
        "NVDA data center revenue growth",
        session_id="rs:t",
        as_of="2026-08-10",
        tickers=["NVDA"],
        dispatch=lambda name, args: {},
    )
    assert all(not a.unscoped_queries for a in plain)


# ---------------------------------------------------------------------------
# Hedgefund MVP slice: SEC+FINRA+WEB domains under the kernel's source policy.
# All offline (fake archive seam, no network). Live golden coverage stays with
# the opt-in verify:hedgefund-live script, never pytest.
# - domains SEC|FINRA|WEB; provenance sec_source|sec_record|finra_record|web_source|search_run|none (closed;
#   FINRA/WEB/SEC-replay only via persisted-result replay, bare rows fail closed with ERR_RAW_SOURCE_REQUIRED);
# - integrity SEC=PRIMARY_DOCUMENT/CANONICAL_STRUCTURED FINRA=CANONICAL_STRUCTURED WEB=EXTERNAL_SOURCE
# - session source policy {mode:allowlist,allowed:[]} default SEC-only;
# - agents sec-agent/finra-agent/exa-agent + scouts; stockbot.yml async off, depth 3;
# - Wave1 one OMP task batch with 3 desks; targeted waves only requested desks;
# - committee same freeze via research_read only.
# ---------------------------------------------------------------------------


def _hf_policy(*sources: str) -> dict[str, JSONValue]:
    """Session policy requesting exactly these source domains (allowlist)."""
    return {"research_sources": {"mode": "allowlist", "sources": list(sources)}}


def _hf_sid(repo: ResearchRepository, *sources: str, q: str = "NVDA demand?") -> tuple[str, str]:
    """Fresh session (+ first running job) under the requested sources; default SEC-only."""
    from app.research import service as _svc

    sid = _svc.create_research(
        q, "o", as_of="2025-06-30T00:00:00+00:00", policy=_hf_policy(*sources) if sources else None, repo=repo
    )
    return sid, repo.list_jobs(sid)[0].job_id


def test_hf_provenance_maps_domain_and_integrity_in_kernel() -> None:
    """Kernel owns provenance->domain/integrity; evidence_to_dict exposes both on every row."""
    from app.research.evidence import (
        evidence_domain,
        evidence_integrity,
        finra_record_ref,
        search_run_ref,
        sec_source_ref,
        web_source_ref,
    )

    sec = sec_source_ref(
        accession_no="0000320193-25-000079",
        document_name="d.htm",
        passage="hello world",
        offset=0,
        end=5,
        basis="rendered",
        text_hash="ab" * 32,
    )
    assert evidence_domain(sec) == "SEC" and evidence_integrity(sec) == "PRIMARY_DOCUMENT"
    fin = finra_record_ref(tool_name="query_finra", record_identity="row-1")
    assert evidence_domain(fin) == "FINRA" and evidence_integrity(fin) == "CANONICAL_STRUCTURED"
    web = web_source_ref(url="https://example.com/a", excerpt="highlight text")
    assert evidence_domain(web) == "WEB" and evidence_integrity(web) == "EXTERNAL_SOURCE"
    assert evidence_domain(search_run_ref(search_id="s1", query="q")) == "WEB"
    assert evidence_domain({"kind": "none"}) == "SOURCE" and evidence_domain(None) == "SOURCE"
    from app.research.evidence import Evidence, evidence_to_dict

    content = "hello world"
    rec = Evidence(
        evidence_id="EV-1",
        session_id="rs:t",
        wave_id=1,
        source_type="pi",
        source_name="SEC",
        subject="NVDA",
        claim_text=content,
        content=content,
        content_hash=__import__("app.research.evidence", fromlist=["evidence_content_hash"]).evidence_content_hash(
            content
        ),
        retrieved_at=__import__("datetime", fromlist=["datetime"]).datetime(
            2025, 6, 29, tzinfo=__import__("datetime", fromlist=["UTC"]).UTC
        ),
        provenance=dict(sec),
    )
    stored = evidence_to_dict(rec)
    assert stored["source_domain"] == "SEC" and stored["integrity_class"] == "PRIMARY_DOCUMENT"
    from app.research.evidence import EvidenceIntegrityError, evidence_from_dict

    assert evidence_from_dict(dict(stored)).evidence_id == "EV-1"
    with pytest.raises(EvidenceIntegrityError, match="source_domain"):
        evidence_from_dict({**stored, "source_domain": "FINRA"})
    with pytest.raises(EvidenceIntegrityError, match="integrity_class"):
        evidence_from_dict({**stored, "integrity_class": "EXTERNAL_SOURCE"})
    legacy = {k: v for k, v in stored.items() if k not in ("source_domain", "integrity_class")}
    assert evidence_from_dict(legacy).evidence_id == "EV-1"


def test_hf_sources_carry_kernel_integrity() -> None:
    """Sources derive domain+integrity from observations via the kernel (caller never assigns)."""
    from app.research.agents import GroundedClaim
    from app.research.synthesis.final import _normalize_sources, _source_line

    obs = {
        "evidence_id": "EV-1",
        "provenance": {"kind": "finra_record", "tool_name": "query_finra", "record_identity": "row-1"},
    }
    rows = _normalize_sources(None, [GroundedClaim(text="t", evidence_ids=["EV-1"])], [obs])
    assert rows == [
        {"evidence_id": "EV-1", "domain": "FINRA", "document": "", "integrity_class": "CANONICAL_STRUCTURED"}
    ]
    assert "CANONICAL_STRUCTURED" in _source_line(rows[0])
    # Caller-supplied rows still get kernel integrity from the matching observation.
    caller = [{"evidence_id": "EV-1", "domain": "FINRA", "document": "FINRA"}]
    rows2 = _normalize_sources(caller, [GroundedClaim(text="t", evidence_ids=["EV-1"])], [obs])
    assert rows2[0]["integrity_class"] == "CANONICAL_STRUCTURED"


def test_hf_synthesize_keeps_trio_what_would_change() -> None:
    """Regression: synthesize_final unions the trio what_would_change (dropped wiring broke test_final_branches)."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="exposure", evidence_ids=["EV-1"])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[claim],
        what_would_change=["c1"],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[claim],
        what_would_change=["c1"],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[claim],
        what_would_change=["c2"],
    )
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    assert synth.what_would_change == ["c1", "c2"]


def test_hf_normalize_coverage_keeps_submit_keys_verbatim() -> None:
    """Dossier submit keys survive normalize + lines under their own names (no aliasing)."""
    from app.research.synthesis.final import _coverage_lines, _normalize_coverage

    coverage: dict[str, object] = {
        "sec": {"docs": ["0000320193-25-000079"], "forms_examined": ["10-K"], "search_runs": ["s1"]},
        "finra": {
            "datasets_queried": ["short-interest"],
            "settlement_windows_covered": ["2025-06-15"],
            "tickers_covered": ["NVDA"],
        },
        "web": {"queries_executed": ["NVDA news"], "results_inspected": ["https://example.com/a"]},
    }
    norm = _normalize_coverage(coverage)
    sec = norm["sec"]
    finra = norm["finra"]
    web = norm["web"]
    assert isinstance(sec, dict) and isinstance(finra, dict) and isinstance(web, dict)
    assert sec["docs"] == ["0000320193-25-000079"]
    assert finra["datasets_queried"] == ["short-interest"]
    assert finra["settlement_windows_covered"] == ["2025-06-15"]
    assert web["queries_executed"] == ["NVDA news"]
    assert "source_domain" not in finra and "useful_for_question" not in finra
    assert any("datasets_queried: short-interest" in line for line in _coverage_lines(norm))


def _hf_cov(**over: object) -> dict[str, object]:
    """Insufficient coverage envelope (submit-shaped, no residuals required)."""
    cov: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["open branch"],
        "source_limitations": [],
        "major_entities_investigated": ["NVDA"],
        "relationship_types_checked": [],
        "forms_examined": [],
        "exhibits_examined": [],
        "material_open_questions": ["open branch"],
        "remaining_branches": ["open branch"],
        "routes_unsearched": [],
        "search_runs": ["s1"],
        "covered_branches": [],
    }
    cov.update(over)
    return cov


def _hf_freeze_id(sid: str, wave: int = 1) -> str:
    return f"{sid}:{wave}:freeze"


def _hf_prov(out: Mapping[str, object], key: str) -> object:
    """Provenance field of a record_evidence return (pyrefly-clean accessor)."""
    prov = out.get("provenance")
    assert isinstance(prov, dict)
    return prov.get(key)


def test_hf_default_session_is_sec_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel default: no policy input means SEC-only allowlist."""

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    assert repo.get_session(sid).source_policy == {"allowed": ["SEC"], "denied": [], "mode": "allowlist"}


def test_hf_sec_only_denies_finra_and_web_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-only allowlist: FINRA/WEB job creation fails closed with policy_denied."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    for domain in ("FINRA", "WEB", "finra", "web"):
        with pytest.raises(ValueError, match="policy_denied"):
            _svc.start_job(sid, "source_agent", source=domain, repo=repo, wave_id=1)
    ok = _svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)
    assert repo.get_job(str(ok["job_id"])).source_domain == "SEC"


def test_hf_multisource_policy_allows_three_desks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC+FINRA+WEB allowlist: one running OMP-owned job per desk, distinct ids."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    assert repo.get_session(sid).source_policy == {
        "allowed": ["SEC", "FINRA", "WEB"],
        "denied": [],
        "mode": "allowlist",
    }
    ids = [
        str(_svc.start_job(sid, "source_agent", source=d, budget={"owner": "omp"}, repo=repo, wave_id=1)["job_id"])
        for d in ("SEC", "FINRA", "WEB")
    ]
    assert len(set(ids)) == 3
    by_id = {jid: repo.get_job(jid) for jid in ids}
    assert {j.source_domain for j in by_id.values()} == {"SEC", "FINRA", "WEB"}
    assert all(j.owner == "omp" and j.status == "running" for j in by_id.values())


def test_hf_finra_job_cannot_submit_sec_evidence_as_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-domain isolation: submit cites only session-ledger ids; a dangling id fails
    ERR_EVIDENCE_NOT_FOUND, and a FINRA submit can never invent SEC ids."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, sec = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(fin, coverage=_hf_cov(), evidence_ids=[f"{sid}:ev:ghost"], repo=repo)
    assert repo.get_job(fin).status == "running"
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-own")
    marker = rid.rsplit(":tr:", 1)[-1]
    fin_eid = f"{sid}:ev:fin-own"
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": fin_eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": f"NVDA short interest 12345 shares {marker}",
        },
        repo=repo,
    )
    sec_eid = f"{sid}:ev:sec-other"
    _svc.record_evidence(sid, sec, _svc_item(sec_eid), repo=repo)
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(fin, coverage=_hf_cov(), evidence_ids=[sec_eid], repo=repo)
    assert repo.get_job(fin).status == "running"


def test_hf_exa_job_cannot_submit_finra_evidence_as_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-domain isolation: a WEB submit cites only its wave-domain lane, never FINRA evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(web, coverage=_hf_cov(), evidence_ids=[f"{sid}:ev:ghost"], repo=repo)
    assert repo.get_job(web).status == "running"
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-own")
    web_eid = f"{sid}:ev:web-own"
    _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": web_eid,
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": web_rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-other")
    fin_marker = fin_rid.rsplit(":tr:", 1)[-1]
    fin_eid = f"{sid}:ev:fin-other"
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": fin_eid,
            "content": f"NVDA short interest 12345 shares {fin_marker}",
            "claim_text": f"NVDA short interest 12345 shares {fin_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": fin_rid,
            "matching_passage": f"NVDA short interest 12345 shares {fin_marker}",
        },
        repo=repo,
    )
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(web, coverage=_hf_cov(), evidence_ids=[fin_eid], repo=repo)
    assert repo.get_job(web).status == "running"


def test_hf_no_cross_job_attach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job id from another session is rejected on evidence and submit paths."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid_a, _ = _hf_sid(repo, "SEC", "FINRA", "WEB", q="session A?")
    sid_b, src_b = _hf_sid(repo, "SEC", "FINRA", "WEB", q="session B?")
    with pytest.raises(ValueError, match="belongs to"):
        _svc.record_evidence(sid_a, src_b, _svc_item(f"{sid_a}:ev:1"), repo=repo)
    with pytest.raises(ValueError, match="belongs to"):
        _svc.record_evidence(sid_a, src_b, _svc_item(f"{sid_a}:ev:2"), repo=repo)
    with pytest.raises(ValueError, match="belongs to"):
        _svc.authorize_and_consume_dispatch(sid_a, src_b, "search_sec_filings", repo=repo)
    assert repo.list_evidence(sid_a) == [] and repo.list_evidence(sid_b) == []


def test_hf_replay_rejects_cross_wave_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wave-1 persisted result cited from a wave-2 job fails ERR_PROVENANCE_MISMATCH."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    w1 = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    w1_rid = _hf_persist_finra(repo, sid, w1, f"{sid}:tr:w1", as_of_date="2025-06-15", published_at="2025-06-15")
    w1_marker = w1_rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        w1,
        {
            "evidence_id": f"{sid}:ev:w1",
            "content": f"NVDA short interest 12345 shares {w1_marker}",
            "claim_text": f"NVDA short interest 12345 shares {w1_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": w1_rid,
            "matching_passage": f"NVDA short interest 12345 shares {w1_marker}",
        },
        repo=repo,
    )
    for job in repo.list_jobs(sid):
        if job.job_type in ("source_agent", "scout") and job.wave_id == 1 and job.status in ("queued", "running"):
            _svc.complete_job(job.job_id, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    w2 = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=2)["job_id"])
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            w2,
            {
                "evidence_id": f"{sid}:ev:w2-cross",
                "content": f"NVDA short interest 12345 shares {w1_marker}",
                "claim_text": f"NVDA short interest 12345 shares {w1_marker}",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": w1_rid,
                "matching_passage": f"NVDA short interest 12345 shares {w1_marker}",
            },
            repo=repo,
        )
    w2_rid = _hf_persist_finra(repo, sid, w2, f"{sid}:tr:w2", as_of_date="2025-06-15", published_at="2025-06-15")
    w2_marker = w2_rid.rsplit(":tr:", 1)[-1]
    out = _svc.record_evidence(
        sid,
        w2,
        {
            "evidence_id": f"{sid}:ev:w2",
            "content": f"NVDA short interest 12345 shares {w2_marker}",
            "claim_text": f"NVDA short interest 12345 shares {w2_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": w2_rid,
            "matching_passage": f"NVDA short interest 12345 shares {w2_marker}",
        },
        repo=repo,
    )
    assert _hf_prov(out, "tool_result_id") == w2_rid


def test_hf_admission_rejects_fabricated_sec_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No handle, garbage handle, unknown document, and declared-ref mismatch all fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    no_handle = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, no_handle, repo=repo)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:2", source_handle={"bogus": 1}), repo=repo)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            src,
            _svc_item(
                f"{sid}:ev:3",
                source_record_id="0000000000-00-000000",
                document_name="nope.htm",
                source_handle={
                    "accession_no": "0000000000-00-000000",
                    "document_name": "nope.htm",
                    "basis": "raw",
                    "offset": 0,
                    "max_chars": 5,
                    "text_hash": "x" * 64,
                    "source_content_hash": "a" * 64,
                },
            ),
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:4", source_record_id="0000320193-25-000080"), repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_admission_rejects_fabricated_finra_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FINRA-shaped row with no SEC archive handle is navigation at best: ERR_RAW_SOURCE_REQUIRED."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    finra_row = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    finra_row.update({"content": "short interest 5%", "claim_text": "short interest 5%", "source_name": "FINRA"})
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, finra_row, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_admission_rejects_fabricated_web_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare web URL with no archive handle never becomes citable evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    web_row = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    web_row.update(
        {
            "content": "news says up",
            "claim_text": "news says up",
            "source_name": "web",
            "source_uri": "https://example.com/news",
        }
    )
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, web_row, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_pit_rejects_newer_than_as_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filing newer than as_of never becomes evidence: the archive refuses the read or the ingest gate rejects it."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    passage = "Later passage a day past the cutoff."
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected|ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            src,
            _svc_item(
                f"{sid}:ev:late",
                passage=passage,
                source_record_id="0000320193-25-000080",
                document_name="nvda-late.htm",
                source_uri="https://sec.gov/late",
                source_handle=seam.handle_for(
                    passage,
                    accession="0000320193-25-000080",
                    document="nvda-late.htm",
                    known_at="2025-07-01T00:00:00+00:00",
                ),
                known_at="2025-07-01T00:00:00+00:00",
            ),
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_pit_unknown_time_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded as_of with no known_at fails PIT_UNVERIFIED, never defaults to eligible."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    passage = "Undated passage with no PIT proof."
    item = _svc_item(
        f"{sid}:ev:undated",
        passage=passage,
        source_record_id="0000320193-25-000081",
        document_name="nvda-undated.htm",
        source_uri="https://sec.gov/undated",
        source_handle=seam.handle_for(passage, accession="0000320193-25-000081", document="nvda-undated.htm"),
    )
    item.pop("known_at", None)
    # Archive is the PIT authority: caller known_at is overwritten by the
    # materialized timing, so unknown time is driven by the archive row.
    seam._SOURCE_TIMING[("0000320193-25-000081", "nvda-undated.htm")] = ("", "2025-04-30", "2025-05-01T00:00:00Z")

    with pytest.raises(ValueError, match="PIT_UNVERIFIED|rejected"):
        _svc.record_evidence(sid, src, item, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_multisource_freeze_hash_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same evidence set freezes to a stable hash: order-insensitive, byte-identical on re-read."""
    from app.research import freeze as _freeze
    from app.research import service as _svc
    from app.research.evidence import evidence_from_dict

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    e1, e2 = f"{sid}:ev:1", f"{sid}:ev:2"
    _svc.record_evidence(sid, src, _svc_item(e1), repo=repo)
    _svc.record_evidence(
        sid,
        src,
        _svc_item(
            e2,
            passage="Second frozen passage.",
            source_record_id="0000320193-25-000080",
            document_name="nvda-2.htm",
            source_uri="https://sec.gov/2",
            source_handle=seam.handle_for(
                "Second frozen passage.", accession="0000320193-25-000080", document="nvda-2.htm"
            ),
        ),
        repo=repo,
    )
    _svc.complete_job(src, {}, repo=repo)
    fid = _hf_freeze_id(sid)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == fid
    recs = [evidence_from_dict(r) for r in repo.list_evidence(sid)]
    frozen = _freeze.freeze_from_dict(repo.get_freeze(fid))
    _freeze.verify_freeze(frozen, recs)
    _freeze.verify_freeze(frozen, list(reversed(recs)))
    assert _freeze.freeze_content_hash(recs) == frozen.content_hash


def test_hf_finra_only_followup_single_desk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Targeted FINRA-only follow-up: only a FINRA job is opened; SEC/WEB stay denied."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "FINRA")
    fin = str(
        _svc.start_job(sid, "source_agent", source="FINRA", budget={"owner": "omp"}, repo=repo, wave_id=1)["job_id"]
    )
    assert repo.get_job(fin).source_domain == "FINRA"
    with pytest.raises(ValueError, match="policy_denied"):
        _svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)
    with pytest.raises(ValueError, match="policy_denied"):
        _svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)
    assert [j.source_domain for j in repo.list_jobs(sid)] == [None, "FINRA"]


def test_hf_committee_same_freeze_no_data_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Committee: one freeze across the trio; data/task tools blocked, research_read passes."""
    from app.research import service as _svc
    from app.research.stage import check_stage_tool, stage_for_session

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    fid = _hf_freeze_id(sid)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == fid
    out = _svc.create_committee_jobs(sid, 1, repo=repo)
    out_jobs = out["jobs"]
    assert isinstance(out_jobs, list) and len(out_jobs) == 3
    assert out["freeze_id"] == fid
    trio = {str(jid): repo.get_job(str(jid)) for jid in out_jobs if isinstance(jid, str)}
    assert len(trio) == 3
    assert {j.job_type for j in trio.values()} == {"stockbot", "bullbot", "bearbot"}
    assert all(j.wave_id == 1 and j.status == "running" for j in trio.values())
    assert stage_for_session(repo.get_session(sid), repo.list_jobs(sid)) == "COMMITTEE"
    for jid in trio:
        for tool in (
            "search_web",
            "get_sec_document",
            "research_add_evidence",
            "research_submit_source_result",
            "task",
        ):
            with pytest.raises(ValueError, match="forbids|cannot dispatch"):
                _svc.authorize_and_consume_dispatch(sid, jid, tool, repo=repo)
    check_stage_tool("COMMITTEE", "research_read")
    jobs_out = [jid for jid in out_jobs if isinstance(jid, str)]
    assert len(jobs_out) == 3
    for role, jid in zip(("stockbot", "bullbot", "bearbot"), jobs_out):
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)
    assert repo.get_session(sid).freeze_ids == [fid]


def _hf_run_to_freeze(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str, str]:
    """One evidence row -> insufficient submit -> freeze; returns (sid, src, fid)."""
    from app.research import service as _svc

    sid, src = _hf_sid(repo, q=q)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_hf_cov(), evidence_ids=[eid], repo=repo)
    return sid, src, str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])


def test_hf_resume_mid_source_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume mid-source: read-only snapshot, open job kept, no dup evidence/jobs."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    snap = _svc.resume_research(sid, repo=repo)
    open_ids = snap["open_job_ids"]
    assert isinstance(open_ids, list) and src in open_ids
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert ResearchRepository().resume(sid).open_job_ids == [src]


def test_hf_resume_after_source_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after source submit: still no new jobs/evidence, source job completed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_hf_cov(), evidence_ids=[eid], repo=repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    _svc.resume_research(sid, repo=repo)
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert repo.get_job(src).status == "completed"


def test_hf_resume_after_freeze_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after freeze: same freeze id, no fetch rerun, no dup jobs/evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _, fid = _hf_run_to_freeze(repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    _svc.resume_research(sid, repo=repo)
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert repo.get_session(sid).freeze_ids == [fid]


def test_hf_resume_after_one_committee_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after one committee member: same freeze, evidence/jobs untouched, trio resumable."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _, fid = _hf_run_to_freeze(repo)
    eid = f"{sid}:ev:1"
    first = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    _svc.record_committee_analysis(sid, first, "stockbot", _svc_ana(eid), repo=repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)), list(repo.get_session(sid).freeze_ids))
    _svc.resume_research(sid, repo=repo)
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)), list(repo.get_session(sid).freeze_ids)) == before
    assert repo.get_job(first).status == "completed"
    rest = _svc.create_committee_jobs(sid, 1, repo=repo)
    rest_jobs_raw = rest["jobs"]
    assert isinstance(rest_jobs_raw, list)
    rest_jobs = [jid for jid in rest_jobs_raw if isinstance(jid, str)]
    assert fid in (str(rest["freeze_id"]), repo.get_session(sid).freeze_ids[-1])
    assert len(set(rest_jobs)) == 3


def test_hf_provenance_vocab_closed() -> None:
    """Provenance kinds are closed at six: sec_source|sec_record|finra_record|web_source|search_run|none all validate.

    Bare FINRA rows / bare URLs still fail closed at admission (ERR_RAW_SOURCE_REQUIRED):
    only tool_result_id refs replayed against persisted tool results ground finra/web evidence.
    """
    from app.research.evidence import (
        PROVENANCE_KINDS,
        finra_record_ref,
        search_run_ref,
        sec_record_ref,
        sec_source_ref,
        validate_provenance,
        web_source_ref,
    )

    assert set(PROVENANCE_KINDS) == {"sec_source", "sec_record", "finra_record", "web_source", "search_run", "none"}
    assert validate_provenance({"kind": "none"}) == {"kind": "none"}
    assert validate_provenance(search_run_ref(search_id="s1", query="q"))["kind"] == "search_run"
    sec = sec_source_ref(
        accession_no="0000320193-25-000079",
        document_name="d.htm",
        passage="hello world",
        offset=0,
        end=5,
        basis="rendered",
        text_hash="ab" * 32,
    )
    assert validate_provenance(dict(sec))["kind"] == "sec_source"
    rec = sec_record_ref(tool_name="get_sec_filing", record_identity="row-1")
    assert validate_provenance(dict(rec))["kind"] == "sec_record"
    fin = finra_record_ref(tool_name="query_finra", record_identity="row-1")
    assert validate_provenance(dict(fin))["kind"] == "finra_record"
    web = web_source_ref(url="https://example.com/a", excerpt="highlight text")
    assert validate_provenance(dict(web))["kind"] == "web_source"
    with pytest.raises(ValueError, match="kind"):
        validate_provenance({"kind": "carrier_pigeon"})


def _hf_persist_finra(
    repo: ResearchRepository,
    sid: str,
    job_id: str,
    rid: str,
    *,
    as_of_date: str | None = "2025-06-15",
    published_at: str | None = "2025-06-20",
) -> str:
    """Persist one staged FINRA result; returns its tool_result_id."""
    from app.research import service as _svc

    marker = rid.rsplit(":tr:", 1)[-1]
    payload: dict[str, object] = {
        "records": [{"symbol": "NVDA", "shortInterest": 12345, "marker": marker}],
        "briefing": f"NVDA short interest 12345 shares {marker}",
    }
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date
    if published_at is not None:
        payload["published_at"] = published_at
    return str(_svc.persist_tool_result(sid, job_id, "query_finra", rid, payload, repo=repo)["tool_result_id"])


def _hf_persist_web(repo: ResearchRepository, sid: str, job_id: str, rid: str) -> str:
    """Persist one staged search_web result; returns its tool_result_id."""
    from app.research import service as _svc

    return str(
        _svc.persist_tool_result(
            sid,
            job_id,
            "search_web",
            rid,
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": "NVDA rallies on demand",
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )


def _hf_desks(repo: ResearchRepository, sid: str) -> tuple[str, str]:
    """Fresh running FINRA + WEB jobs for one session."""
    from app.research import service as _svc

    fin = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    web = str(_svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)["job_id"])
    return fin, web


def test_hf_persist_accepts_matching_domain_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_tool_result: FINRA evidence tool on FINRA jobs, search_web on WEB jobs."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    assert _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-ok") == f"{sid}:tr:fin-ok"
    assert _hf_persist_web(repo, sid, web, f"{sid}:tr:web-ok") == f"{sid}:tr:web-ok"
    assert _svc.get_tool_result(f"{sid}:tr:fin-ok", repo=repo)["tool_name"] == "query_finra"
    assert _svc.get_tool_result(f"{sid}:tr:web-ok", repo=repo)["tool_name"] == "search_web"


def test_hf_persist_rejects_cross_domain_and_sec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_tool_result: cross-domain tools and every SEC-job persist fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, sec = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    with pytest.raises(ValueError, match="cannot ground FINRA evidence"):
        _svc.persist_tool_result(sid, fin, "search_web", f"{sid}:tr:x1", {"evidence": []}, repo=repo)
    with pytest.raises(ValueError, match="cannot ground WEB evidence"):
        _svc.persist_tool_result(sid, web, "query_finra", f"{sid}:tr:x2", {"records": []}, repo=repo)
    with pytest.raises(ValueError, match="cannot ground FINRA evidence"):
        _svc.persist_tool_result(sid, fin, "list_finra_datasets", f"{sid}:tr:x3", {"datasets": []}, repo=repo)
    for tool in ("query_finra", "search_web", "get_sec_document"):
        with pytest.raises(ValueError, match="persists no tool results"):
            _svc.persist_tool_result(sid, sec, tool, f"{sid}:tr:x-{tool}", {}, repo=repo)


def test_hf_persist_is_idempotent_first_bytes_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume-safe persist: repeating a tool_result_id keeps the first bytes, same id back."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = f"{sid}:tr:repeat"
    marker = rid.rsplit(":tr:", 1)[-1]
    _hf_persist_finra(repo, sid, fin, rid)
    again = _svc.persist_tool_result(
        sid, fin, "query_finra", rid, {"records": [{"symbol": "ZZZ"}], "briefing": "other"}, repo=repo
    )
    assert again["tool_result_id"] == rid
    stored = _svc.get_tool_result(rid, repo=repo)
    result = stored["result"]
    assert isinstance(result, dict)
    assert result["briefing"] == f"NVDA short interest 12345 shares {marker}"


def test_hf_replay_admits_finra_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA job + tool_result_id replay admits finra_record provenance with publication known_at."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-replay")
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-1",
            "content": "NVDA short interest 12345 shares fin-replay",
            "claim_text": "NVDA short interest 12345 shares fin-replay",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": "NVDA short interest 12345 shares fin-replay",
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "finra_record"
    assert _hf_prov(out, "tool_name") == "query_finra"
    assert _hf_prov(out, "known_at") == "2025-06-20"
    assert out["known_at"] == "2025-06-20T00:00:00+00:00"


def test_hf_sec_record_identity_keys_on_record_and_tool_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC replay dedupe keys on record_identity + tool_result_id, never subject/claim alone.

    Regression: claim-less SEC replay rows under one subject collided on a blank
    accession/document key, so a later distinct record deduped onto the first.
    """
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    sec = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    payload: dict[str, object] = {
        "filings": ["NVDA Form 4 accession 0001045810-25-000001", "NVDA 10-Q accession 0001045810-25-000002"],
        "filed_at": "2025-06-01",
    }
    rid_a = str(
        _svc.persist_tool_result(sid, sec, "list_sec_filings", f"{sid}:tr:a", payload, repo=repo)["tool_result_id"]
    )
    rid_b = str(
        _svc.persist_tool_result(sid, sec, "list_sec_filings", f"{sid}:tr:b", payload, repo=repo)["tool_result_id"]
    )

    def _cite(eid: str, rid: str, passage: str) -> dict[str, JSONValue]:
        # No claim_text/subject: the scheduler path supplies content only.
        item: dict[str, object] = {
            "evidence_id": f"{sid}:ev:{eid}",
            "content": passage,
            "source_name": "SEC",
            "tool_result_id": rid,
            "matching_passage": passage,
        }
        return _svc.record_evidence(sid, sec, item, repo=repo)

    form4 = "NVDA Form 4 accession 0001045810-25-000001"
    first = _cite("a-form4", rid_a, form4)
    other_record = _cite("a-10q", rid_a, "NVDA 10-Q accession 0001045810-25-000002")
    other_result = _cite("b-form4", rid_b, form4)
    repeat = _cite("a-form4-again", rid_a, form4)
    assert _hf_prov(first, "kind") == "sec_record"
    assert "duplicate_of" not in other_record and other_record["evidence_id"] == f"{sid}:ev:a-10q"
    assert "duplicate_of" not in other_result and other_result["evidence_id"] == f"{sid}:ev:b-form4"
    assert repeat == {"evidence_id": f"{sid}:ev:a-form4", "accepted": False, "duplicate_of": f"{sid}:ev:a-form4"}
    assert repo.list_evidence_ids(sid) == [f"{sid}:ev:a-form4", f"{sid}:ev:a-10q", f"{sid}:ev:b-form4"]


def test_hf_replay_admits_web_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB job + tool_result_id replay admits web_source provenance (url/excerpt/title/domain)."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-replay")
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-1",
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "web_source"
    assert _hf_prov(out, "url") == "https://example.com/a"
    assert _hf_prov(out, "excerpt") == "NVDA rallies on demand"
    assert _hf_prov(out, "published_at") == "2025-06-20"


def test_hf_replay_uses_explicit_store_and_rejects_cross_session(tmp_path: Path) -> None:
    """Explicit-repo replay reads the caller's DB and rejects another session's result."""
    from app.research import service as _svc

    repo_a = ResearchRepository(tmp_path / "a.sqlite")
    repo_b = ResearchRepository(tmp_path / "b.sqlite")
    sid_a, _ = _hf_sid(repo_a, "SEC", "FINRA", "WEB")
    fin_a, web_a = _hf_desks(repo_a, sid_a)
    fin_rid = _hf_persist_finra(repo_a, sid_a, fin_a, f"{sid_a}:tr:fin-xstore")
    web_rid = _hf_persist_web(repo_a, sid_a, web_a, f"{sid_a}:tr:web-xstore")
    fin_out = _svc.record_evidence(
        sid_a,
        fin_a,
        {
            "evidence_id": f"{sid_a}:ev:fin-xstore",
            "content": "NVDA short interest 12345 shares fin-xstore",
            "claim_text": "NVDA short interest 12345 shares fin-xstore",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": fin_rid,
            "matching_passage": "NVDA short interest 12345 shares fin-xstore",
        },
        repo=repo_a,
    )
    assert _hf_prov(fin_out, "kind") == "finra_record"
    web_out = _svc.record_evidence(
        sid_a,
        web_a,
        {
            "evidence_id": f"{sid_a}:ev:web-xstore",
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": web_rid,
            "source_record_id": "https://example.com/a",
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo_a,
    )
    assert _hf_prov(web_out, "kind") == "web_source"
    sid_b, _ = _hf_sid(repo_b, "SEC", "FINRA", "WEB")
    fin_b = str(_svc.start_job(sid_b, "source_agent", source="FINRA", repo=repo_b, wave_id=1)["job_id"])
    stored_a = repo_a.get_tool_result(fin_rid)
    repo_b.save_tool_result(dict(stored_a))
    with pytest.raises(ValueError, match="PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid_b,
            fin_b,
            {
                "evidence_id": f"{sid_b}:ev:fin-cross",
                "content": "NVDA short interest 12345 shares fin-xstore",
                "claim_text": "NVDA short interest 12345 shares fin-xstore",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": fin_rid,
                "matching_passage": "NVDA short interest 12345 shares fin-xstore",
            },
            repo=repo_b,
        )


def test_hf_replay_wrong_result_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A citation the persisted result does not reproduce fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-wrong")
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-wrong")
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            fin,
            {
                "evidence_id": f"{sid}:ev:fin-x",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": fin_rid,
                "matching_passage": "totally absent values 99999",
            },
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            web,
            {
                "evidence_id": f"{sid}:ev:web-x",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "web",
                "tool_result_id": web_rid,
                "matching_passage": "absent quote xyz",
                "known_at": "2025-06-29T00:00:00+00:00",
            },
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(
            sid,
            fin,
            {
                "evidence_id": f"{sid}:ev:fin-unknown",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": f"{sid}:tr:nope",
                "matching_passage": "NVDA short interest 12345 shares",
            },
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_finra_caller_known_at_never_overrides_result_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA known_at is the persisted publication time: a caller backdate cannot move it."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-caller-time")
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-caller-time",
            "content": "NVDA short interest 12345 shares fin-caller-time",
            "claim_text": "NVDA short interest 12345 shares fin-caller-time",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": "NVDA short interest 12345 shares fin-caller-time",
            "known_at": "2025-01-01T00:00:00+00:00",
        },
        repo=repo,
    )
    assert out["known_at"] == "2025-06-20T00:00:00+00:00"


def test_hf_finra_known_at_gates_pit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA known_at comes from the persisted publication time: early admits, late PIT_VIOLATION, missing PIT_UNVERIFIED."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")

    def _item(eid: str, rid: str, passage: str) -> dict[str, object]:
        return {
            "evidence_id": eid,
            "content": passage,
            "claim_text": passage,
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": passage,
        }

    early_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    early_rid = _hf_persist_finra(
        repo, sid, early_job, f"{sid}:tr:pit-early", as_of_date="2025-06-15", published_at="2025-06-15"
    )
    _svc.record_evidence(
        sid, early_job, _item(f"{sid}:ev:pit-early", early_rid, "NVDA short interest 12345 shares pit-early"), repo=repo
    )
    late_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    late_rid = _hf_persist_finra(
        repo, sid, late_job, f"{sid}:tr:pit-late", as_of_date="2025-06-15", published_at="2025-07-15"
    )
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid,
            late_job,
            _item(f"{sid}:ev:pit-late", late_rid, "NVDA short interest 12345 shares pit-late"),
            repo=repo,
        )
    bare_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    bare_rid = _hf_persist_finra(repo, sid, bare_job, f"{sid}:tr:pit-none", as_of_date=None, published_at=None)
    with pytest.raises(ValueError, match="PIT_UNVERIFIED|rejected"):
        _svc.record_evidence(
            sid,
            bare_job,
            _item(f"{sid}:ev:pit-none", bare_rid, "NVDA short interest 12345 shares pit-none"),
            repo=repo,
        )
    assert [r["evidence_id"] for r in repo.list_evidence(sid)] == [f"{sid}:ev:pit-early"]


def test_hf_finra_settlement_is_not_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-cutoff settlementDate published after cutoff is PIT_VIOLATION, never admitted."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    rid = _hf_persist_finra(repo, sid, job, f"{sid}:tr:pit-settle", as_of_date="2025-06-15", published_at="2025-07-15")
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid,
            job,
            {
                "evidence_id": f"{sid}:ev:pit-settle",
                "content": "NVDA short interest 12345 shares pit-settle",
                "claim_text": "NVDA short interest 12345 shares pit-settle",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": rid,
                "matching_passage": "NVDA short interest 12345 shares pit-settle",
            },
            repo=repo,
        )


def test_hf_get_tool_result_roundtrip_and_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_tool_result round-trips the persisted payload; unknown ids raise ResearchNotFound."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:roundtrip")
    stored = _svc.get_tool_result(rid, repo=repo)
    assert stored["tool_result_id"] == rid and stored["tool_name"] == "query_finra"
    assert stored["session_id"] == sid and stored["job_id"] == fin
    with pytest.raises(_svc.ResearchNotFound, match="unknown tool_result_id"):
        _svc.get_tool_result(f"{sid}:tr:nope", repo=repo)


def test_hf_gate_finra_authorized_when_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multisource session: a material FINRA committee follow-up authorizes the next wave."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(
        src,
        coverage=_hf_cov(unresolved=[], material_open_questions=[], remaining_branches=[]),
        evidence_ids=[eid],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    follow_up = {"question": "What does FINRA short interest show for NVDA?", "suggested_source": "finra"}
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid, jid, role, _committee_analysis(eid, [follow_up] if i == 0 else []), repo=repo
        )
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_domain"] == "finra"
    assert out["targeted_question"] == "What does FINRA short interest show for NVDA?"
    sess = repo.get_session(sid)
    assert (
        sess.current_wave == 2
        and sess.status == "targeted_research"
        and sess.targeted_question == out["targeted_question"]
    )


def test_hf_gate_web_denied_when_sec_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-only session: a material WEB committee follow-up stops not_actionable (fails closed)."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(
        src,
        coverage=_hf_cov(unresolved=[], material_open_questions=[], remaining_branches=[]),
        evidence_ids=[eid],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    off_lane = {"question": "What does the press say about the NVDA terms?", "suggested_source": "WEB"}
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid, jid, role, _committee_analysis(eid, [off_lane] if i == 0 else []), repo=repo
        )
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is False and out["stop_reason"] == "not_actionable"
    assert repo.get_session(sid).status == "synthesizing"


def _hf_finra_sufficient(**over: object) -> dict[str, object]:
    """FINRA sufficient envelope: datasets/tickers/windows/reads/branches, no residuals."""
    cov: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA short interest"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "datasets_queried": ["short-interest"],
        "tickers_covered": ["NVDA"],
        "settlement_windows_covered": ["2025-06-15"],
        "dataset_reads": ["read-1"],
        "covered_branches": ["short interest"],
        "material_open_questions": [],
        "major_entities_missing": [],
        "remaining_branches": [],
        "routes_unsearched": [],
    }
    cov.update(over)
    return cov


def _hf_web_sufficient(**over: object) -> dict[str, object]:
    """WEB sufficient envelope: semantic branches/queries/results/branches, no residuals."""
    cov: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA news"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "semantic_branches_covered": ["company news"],
        "queries_executed": ["NVDA news"],
        "results_inspected": ["https://example.com/a"],
        "covered_branches": ["company news"],
        "material_open_questions": [],
        "major_entities_missing": [],
        "remaining_branches": [],
        "routes_unsearched": [],
    }
    cov.update(over)
    return cov


def test_hf_finra_sufficient_accepts_own_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient with datasets/tickers/windows/reads/branches completes; SEC keys not required."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-suff")
    eid = f"{sid}:ev:fin-suff"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": f"NVDA short interest 12345 shares {marker}",
        },
        repo=repo,
    )
    out = _svc.submit_source_result(fin, coverage=_hf_finra_sufficient(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    stored = repo.get_dossier(str(out["dossier_id"]))["coverage"]
    assert isinstance(stored, dict)
    assert stored.get("source_domain") == "FINRA" and stored.get("datasets_queried") == ["short-interest"]


def test_hf_web_sufficient_accepts_own_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB sufficient with semantic branches/queries/results/branches completes; SEC keys not required."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-suff")
    eid = f"{sid}:ev:web-suff"
    _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": eid,
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    out = _svc.submit_source_result(web, coverage=_hf_web_sufficient(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    stored = repo.get_dossier(str(out["dossier_id"]))["coverage"]
    assert isinstance(stored, dict)
    assert stored.get("source_domain") == "WEB" and stored.get("queries_executed") == ["NVDA news"]


def test_hf_finra_sufficient_missing_own_key_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient without datasets_queried fails ERR_COVERAGE_REQUIRED; job stays running."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-miss")
    eid = f"{sid}:ev:fin-miss"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": f"NVDA short interest 12345 shares {marker}",
        },
        repo=repo,
    )
    bad = _hf_finra_sufficient()
    del bad["datasets_queried"]
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.submit_source_result(fin, coverage=bad, evidence_ids=[eid], repo=repo)
    assert repo.get_job(fin).status == "running"


def test_hf_finra_sufficient_with_residual_fails_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient with a remaining branch fails ERR_COVERAGE_INCOMPLETE; SEC regression stays green."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-resid")
    eid = f"{sid}:ev:fin-resid"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": f"NVDA short interest 12345 shares {marker}",
        },
        repo=repo,
    )
    with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
        _svc.submit_source_result(
            fin, coverage=_hf_finra_sufficient(remaining_branches=["Reg SHO"]), evidence_ids=[eid], repo=repo
        )
    assert repo.get_job(fin).status == "running"
    sec_sid, sec = _hf_sid(repo)
    sec_eid = f"{sec_sid}:ev:1"
    _svc.record_evidence(sec_sid, sec, _svc_item(sec_eid), repo=repo)
    assert (
        _svc.submit_source_result(sec, coverage=_reg_cov(), evidence_ids=[sec_eid], repo=repo)["job_status"]
        == "completed"
    )


def _hf_replay_item(eid: str, rid: str, passage: str, **over: object) -> dict[str, object]:
    """FINRA/WEB replay item: distinct evidence ids per citation so dedupe never masks the gate."""
    item: dict[str, object] = {
        "evidence_id": eid,
        "content": passage,
        "claim_text": passage,
        "subject": "NVDA",
        "source_name": "FINRA",
        "tool_result_id": rid,
        "matching_passage": passage,
    }
    item.update(over)
    return item


def test_hf_replay_rejects_padded_finra_citation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strict substring citation records; the same text plus a fabricated tail fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-pad")
    marker = rid.rsplit(":tr:", 1)[-1]
    briefing = f"NVDA short interest 12345 shares {marker}"
    out = _svc.record_evidence(sid, fin, _hf_replay_item(f"{sid}:ev:fin-exact", rid, briefing), repo=repo)
    assert _hf_prov(out, "kind") == "finra_record"
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid, fin, _hf_replay_item(f"{sid}:ev:fin-padded", rid, briefing + " fabricated tail"), repo=repo
        )


def test_hf_replay_rejects_padded_web_citation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strict substring excerpt records; the same highlight plus a fabricated tail fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-pad")
    highlight = "NVDA rallies on demand"
    out = _svc.record_evidence(
        sid,
        web,
        _hf_replay_item(f"{sid}:ev:web-exact", rid, highlight, source_name="web", known_at="2025-06-29T00:00:00+00:00"),
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "web_source"
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            web,
            _hf_replay_item(
                f"{sid}:ev:web-padded",
                rid,
                highlight + " fabricated tail",
                source_name="web",
                known_at="2025-06-29T00:00:00+00:00",
            ),
            repo=repo,
        )


def test_hf_stored_provenance_preserves_tool_result_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored FINRA/WEB provenance keeps the replayed tool_result_id for audit."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-id")
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-id")
    marker = fin_rid.rsplit(":tr:", 1)[-1]
    fin_out = _svc.record_evidence(
        sid, fin, _hf_replay_item(f"{sid}:ev:fin-id", fin_rid, f"NVDA short interest 12345 shares {marker}"), repo=repo
    )
    web_out = _svc.record_evidence(
        sid,
        web,
        _hf_replay_item(
            f"{sid}:ev:web-id",
            web_rid,
            "NVDA rallies on demand",
            source_name="web",
            known_at="2025-06-29T00:00:00+00:00",
        ),
        repo=repo,
    )
    assert _hf_prov(fin_out, "tool_result_id") == fin_rid
    assert _hf_prov(web_out, "tool_result_id") == web_rid


def test_hf_rejected_evidence_journals_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PIT-rejected record_evidence journals an evidence.rejected event with the refusal reason."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-rej", as_of_date="2025-06-15", published_at="2025-07-15")
    marker = rid.rsplit(":tr:", 1)[-1]
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid, fin, _hf_replay_item(f"{sid}:ev:fin-rej", rid, f"NVDA short interest 12345 shares {marker}"), repo=repo
        )
    rejected = [e for e in repo.list_events(sid) if e.event_type == "evidence.rejected"]
    assert rejected and rejected[0].payload.get("evidence_id") == f"{sid}:ev:fin-rej"
    assert rejected[0].payload.get("reason") == "PIT_VIOLATION"


def test_start_job_request_id_dedups_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same request_id returns the winner; no second job row persists."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    first = _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    second = _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    assert second["job_id"] == first["job_id"]
    assert len(repo.list_jobs(sid)) == 2  # seed job + one winner


def test_start_job_request_id_type_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request_id bound to source_agent rejects a committee type."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    with pytest.raises(ValueError, match="already identifies"):
        _svc.start_job(sid, "stockbot", repo=repo, wave_id=1, request_id="attempt-1")


def test_evidence_insert_and_session_link_commit_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence row + session.evidence_ids land in one batch."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:atomic"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    assert eid in repo.get_session(sid).evidence_ids
    assert repo.get_evidence(eid)["evidence_id"] == eid


def test_write_bundle_propagates_store_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing session raises instead of writing a fallback bundle."""
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _fake_dispatch(), _ok, DirectorBudgets())
    with pytest.raises(KeyError):
        run.write_bundle("no-such-session")


def test_write_bundle_propagates_evidence_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt persisted evidence raises instead of skipping the row."""
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    repo = ResearchRepository()
    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _fake_dispatch(), _ok, DirectorBudgets())

    def _corrupt(_row: object) -> object:
        raise ValueError("corrupt row")

    monkeypatch.setattr("app.research.evidence.evidence_from_dict", _corrupt)
    with pytest.raises(ValueError, match="corrupt row"):
        run._bundle_evidence_entries(sid)
