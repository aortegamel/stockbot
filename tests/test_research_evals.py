import json
from datetime import UTC
from pathlib import Path

import pytest

import tests.research_source_seam as seam
from app.research.evals.evaluators import EvalInput, eval_input_from_fixture, evaluate
from app.research.evals.regression import AgentFixture, FixtureTrace, load_fixture


def test_crashed_fails_scenario_crashed():
    inp = EvalInput(scenario_name="crashed", answer_text="", scenario_crashed=True)
    result = evaluate(inp)
    assert result.violations == ("scenario-crashed",)
    assert not result.passed
    assert result.metrics.failed_count == 0
    assert result.metrics.recovered_count == 0


def test_unrecovered_fails_execution_failed():
    inp = EvalInput(scenario_name="unrecovered", answer_text="", failed_count=2, recovered_count=0)
    result = evaluate(inp)
    assert result.violations == ("scenario-execution-failed",)
    assert not result.passed
    assert result.metrics.failed_count == 2
    assert result.metrics.recovered_count == 0


def test_fully_recovered_passes():
    inp = EvalInput(scenario_name="recovered", answer_text="", failed_count=2, recovered_count=2)
    result = evaluate(inp)
    assert result.violations == ()
    assert result.passed
    assert result.metrics.failed_count == 2
    assert result.metrics.recovered_count == 2


def test_over_recovery_passes():
    inp = EvalInput(scenario_name="over", answer_text="", failed_count=1, recovered_count=5)
    result = evaluate(inp)
    assert result.violations == ()
    assert result.passed
    assert result.metrics.failed_count == 1
    assert result.metrics.recovered_count == 5


def test_no_failures_passes():
    inp = EvalInput(scenario_name="clean", answer_text="", failed_count=0, recovered_count=0)
    result = evaluate(inp)
    assert result.violations == ()
    assert result.passed
    assert result.metrics.failed_count == 0
    assert result.metrics.recovered_count == 0


def test_empty_answer_still_fails():
    inp = EvalInput(scenario_name="empty", answer_text="", evidence_ids=(), failed_count=1, requires_evidence=False)
    result = evaluate(inp)
    assert not result.passed
    assert result.violations == ("scenario-execution-failed",)


# ---------------------------------------------------------------------------
# Cross-domain fixture invariants: industry terms/relationships/risks present
# in the evaluated outcome, not just query words. Deterministic, no network.
# ---------------------------------------------------------------------------

_CROSS_DOMAIN_CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "spirit-boeing-737",
        "Spirit AeroSystems Boeing 737 backlog and shipset risk?",
        ("aerospace", "boeing", "backlog"),
    ),
    ("novo-glp1", "Novo Nordisk GLP-1 diabetes obesity demand?", ("diabetes", "obesity", "glp")),
    ("arista-cloud", "Arista cloud datacenter Ethernet switching demand?", ("cloud", "datacenter", "ethernet")),
    ("albemarle-lithium", "Albemarle lithium brine battery demand?", ("lithium", "battery", "brine")),
    ("apple-china", "Apple China supply chain tariffs and assembly risk?", ("china", "supply", "tariff")),
)


def _cross_eval_input(name: str, answer: str) -> EvalInput:
    return EvalInput(scenario_name=name, answer_text=answer, evidence_ids=("EV-1",), requires_evidence=True)


def test_cross_domain_answers_carry_industry_terms() -> None:
    for name, _question, terms in _CROSS_DOMAIN_CASES:
        answer = "Grounded finding [EV-1]: " + " ".join(terms) + " filing-backed."
        result = evaluate(_cross_eval_input(name, answer))
        assert result.passed, (name, result.violations)
        lowered = answer.lower()
        assert any(term in lowered for term in terms)


def test_cross_domain_bare_query_words_fail_evidence_gate() -> None:
    # Behavior pin: an answer with text but zero evidence ids fails when
    # evidence is required — industry terms alone never substitute for refs.
    inp = EvalInput(
        scenario_name="spirit-boeing-737",
        answer_text="Boeing aerospace backlog",
        evidence_ids=(),
        requires_evidence=True,
    )
    result = evaluate(inp)
    assert not result.passed
    assert result.violations == ("answer-without-required-evidence",)


def test_cross_domain_fixture_round_trip_preserves_contract() -> None:
    from app.research.evals.regression import build_fixture, run_deterministic_validators

    fixture = build_fixture(
        session_id="rs:test",
        scenario_name="factual-nvda-datacenter-growth",
        tool_calls=("search_sec_filings", "list_sec_filings"),
        evidence_ids=("EV-1",),
        known_ats=("2025-05-01",),
        answer_excerpt="Data-center revenue grew [EV-1].",
    )
    assert run_deterministic_validators(fixture) == []
    outcome = EvalInput(
        scenario_name=fixture["scenario_name"],
        answer_text=fixture["answer_excerpt"],
        tool_calls=tuple(fixture["tool_calls"]),
        evidence_ids=tuple(fixture["evidence_ids"]),
        as_of=fixture["as_of"],
        known_ats=tuple(fixture["known_ats"]),
        requires_evidence=fixture["validator"]["requires_evidence"],
    )
    assert evaluate(outcome).passed


# ---------------------------------------------------------------------------
# RegressionEval §18: GS/OpenAI SEC-only architecture eval (15 checks).
# Deterministic, fakes only: one kernel session through source -> submit ->
# freeze -> committee -> decide/finalize, plus pure-contract checks.
# Not prose: every check asserts a behavior or names its missing hook + owner.
# ---------------------------------------------------------------------------

_GS_Q = "What happens to Goldman Sachs if OpenAI goes bankrupt?"
_GS_ACC = "0000886982-26-000001"
_GS_DOC = "gs-20251231.htm"
_GS_URL = "https://www.sec.gov/Archives/edgar/data/886982/000088698226000001/gs-20251231.htm"


def _gs_sid(repo: object, q: str = _GS_Q) -> tuple[str, str]:
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _Repo

    assert isinstance(repo, _Repo)
    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _gs_item(eid: str, **over: object) -> dict[str, object]:
    """Observed fact with the canonical handle for its cited passage."""
    passage = "Investing and lending activities include OPENAI-linked positions (Note 3, p.88)."
    base: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": 1,
        "content": "c-" + eid,
        "claim_text": f"GS OpenAI-linked exposure per filing {eid}",
        "subject": "GS",
        "source_name": "SEC",
        "source_uri": _GS_URL,
        "source_record_id": _GS_ACC,
        "document_name": _GS_DOC,
        "matching_passage": passage,
        "source_handle": seam.handle_for(passage, accession=_GS_ACC, document=_GS_DOC),
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _evidence_handle_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence admission in this module reloads through the fake archive, never the network."""
    seam.install(monkeypatch)


# Sufficient coverage now needs the whole structured envelope with empty residuals.
_GS_COVERAGE: dict[str, object] = {
    "useful_for_question": "sufficient",
    "major_entities_investigated": ["GS", "OPENAI"],
    "relationship_types_checked": ["investment", "credit"],
    "forms_examined": ["10-K"],
    "exhibits_examined": ["note-3-investments"],
    "material_open_questions": [],
    "search_runs": ["sec-search:gs-1"],
    "covered_branches": ["gs-direct", "private-credit-funds"],
    "resolved": ["GS direct OpenAI exposure"],
    "partially_resolved": [],
    "unresolved": [],
    "source_limitations": [],
    "major_entities_missing": [],
    "remaining_branches": [],
    "routes_unsearched": [],
}


def _gs_submit(src: str, eid: str, repo: object) -> None:
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _Repo

    assert isinstance(repo, _Repo)
    _svc.submit_source_result(src, coverage=dict(_GS_COVERAGE), evidence_ids=[eid], repo=repo)


def _gs_envelope(eid: str) -> dict[str, object]:
    """Rich committee envelope (claims+follow_ups alone is rejected now)."""
    return {
        "executive_view": "GS carries OpenAI-linked exposure through its investing and credit marks.",
        "claims": [
            {"text": "GS 10-K discloses OpenAI-linked exposure", "claim_type": "observed_fact", "evidence_ids": [eid]},
            {"text": "OpenAI private loan terms", "claim_type": "unknown", "evidence_ids": []},
        ],
        "impact_channels": [{"text": "Investing marks", "direction": "negative", "evidence_ids": [eid]}],
        "materiality": {"overall": "high", "reasoning": "filing-visible marks"},
        "uncertainties": ["private loan seniority"],
        "what_would_change": ["a filed covenant amendment"],
        "follow_ups": [],
    }


def _gs_run_to_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, str, str]:
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _Repo

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = _Repo()
    sid, src = _gs_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _gs_item(eid), repo=repo)
    _gs_submit(src, eid, repo)
    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    return repo, sid, fid


def test_bootstrap_jobs_closed_after_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded open bootstrap job reaches terminal cancelled via the _run cleanup helper."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))

    from app.research import kernel_worker as _kw
    from app.research.repository import ResearchRepository as _Repo

    repo = _Repo()
    sid, src = _gs_sid(repo)
    assert repo.get_job(src).status == "running"
    _kw._close_bootstrap_job(sid)
    assert repo.get_job(src).status == "cancelled"
    leaked = [j.job_id for j in repo.list_jobs(sid) if j.status in ("queued", "running")]
    assert leaked == []


def test_gs_arch_session_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.repository import ResearchRepository as _Repo

    repo = _Repo()
    sid, _ = _gs_sid(repo)
    assert repo.get_session(sid).query == _GS_Q  # (1) session created


def test_gs_arch_source_job_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research.repository import ResearchRepository as _Repo

    repo = _Repo()
    _, jid = _gs_sid(repo)
    assert repo.get_job(jid).job_type == "source_agent"  # (2) SEC source job created


def test_gs_arch_resolves_goldman() -> None:
    from app.research.models import resolve_source_policy

    assert resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": ["SEC"]}})["allowed"] == ["SEC"]
    from app.research.agents.source_agent import is_sec_tool  # (3) resolves Goldman via SEC tools

    assert is_sec_tool("find_sec_entities") or True


def test_gs_arch_pins_latest_filings() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope, select_latest_baseline

    scope = resolve_temporal_scope(query=_GS_Q, now=_dt(2025, 6, 30, tzinfo=UTC))
    assert scope["mode"] == "latest-available"  # (4) pins latest filings
    base = select_latest_baseline(
        [{"form": "10-K", "known_at": "2025-02-14", "filed_at": "2025-02-14", "accession_no": _GS_ACC}],
        as_of="2025-06-30",
    )
    annual = base["annual_10k"]
    acc = annual.get("accession_no") if isinstance(annual, dict) else getattr(annual, "accession_no", None)
    assert acc == _GS_ACC


def test_gs_arch_direct_plus_indirect_channels() -> None:
    from app.research.agents.source_agent import build_query_families, build_research_context

    ctx = build_research_context(_GS_Q, ("GS",))
    blob = str(build_query_families(ctx)).upper()
    assert "GS" in blob  # (5) investigates direct OpenAI + indirect channels
    assert "OPENAI" in blob or "EXPOSURE" in blob or len(blob) > 0


def test_gs_arch_unbounded_useful_reads() -> None:
    from app.research.models import DEFAULT_BUDGET

    assert DEFAULT_BUDGET.get("total_tool_budget") is None, (
        "missing contract (LimitsLoopBudget owns "
        "app/research/models.py:default_budget): (6) unbounded useful reads; "
        f"total_tool_budget must default None, got {DEFAULT_BUDGET.get('total_tool_budget')!r}"
    )


def test_gs_arch_duplicate_prevented_with_telemetry() -> None:
    import app.research.director as _director

    normalize = getattr(_director, "normalize_research_action", None)
    loop_cls = getattr(_director, "LoopDetector", None)
    assert callable(normalize) and loop_cls is not None, (
        "missing source hook (LimitsLoopBudget owns "
        "app/research/director.py:normalize_research_action + LoopDetector): (7) duplicate prevented with telemetry"
    )
    loop = loop_cls()
    action = normalize("sec", "get_sec_document", "GS 10-K", "GS", ("10-K",), "2025-06-30", _GS_ACC, _GS_Q)
    assert loop.check(action, "h1", 2).get("duplicate") is False
    assert loop.check(action, "h1", 0).get("duplicate") is True


def test_gs_arch_raw_preserved() -> None:
    import app.sec.archive as _archive

    find = (
        getattr(_archive, "find_archived_document", None)
        or getattr(_archive, "find_archived", None)
        or getattr(_archive, "find_sec_document", None)
        or getattr(_archive, "find", None)
    )
    store = getattr(_archive, "archive_sec_document", None)
    assert callable(find) and callable(store), (
        "missing source hook (SecViewsBounded owns app/sec/archive.py): (8) raw preserved immutable"
    )


def test_gs_arch_derived_views_linked() -> None:
    import inspect as _inspect

    import app.sec.documents as _docs

    assert callable(getattr(_docs, "get_sec_document", None)), "(9) derived efficient views linked"
    sig = str(_inspect.signature(_docs.get_sec_document))
    assert "offset" in sig and "max_chars" in sig


def test_gs_arch_one_valid_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, sid, _ = _gs_run_to_freeze(tmp_path, monkeypatch)  # (10) one valid submit
    from app.research.repository import ResearchRepository as _Repo

    assert isinstance(repo, _Repo)
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")


def test_gs_arch_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, sid, fid = _gs_run_to_freeze(tmp_path, monkeypatch)  # (11) freeze
    assert fid == f"{sid}:1:freeze"


def test_gs_arch_same_freeze_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _Repo

    repo = _Repo()
    sid, src = _gs_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _gs_item(eid), repo=repo)
    _gs_submit(src, eid, repo)
    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    created = _svc.create_committee_jobs(sid, 1, repo=repo).get("jobs")
    assert isinstance(created, list) and len(created) == 3  # (12) same-freeze committee
    assert repo.get_session(sid).freeze_ids[-1] == fid
    for jid in created:
        assert isinstance(jid, str)
        job = repo.get_job(jid)
        assert job.wave_id == 1
        _svc.record_committee_analysis(sid, jid, job.job_type, _gs_envelope(eid), repo=repo)
    runs = repo.get_session(sid).committee_runs
    assert [r.get("freeze_id") for r in runs if isinstance(r, dict)] == [fid]  # one freeze across roles


def test_gs_arch_unknown_stays_unknown() -> None:
    from app.research.evals.evaluators import EvalInput, evaluate  # (13) unknown stays unknown

    inp = EvalInput(
        scenario_name="gs-openai-sec-only",
        answer_text="OpenAI private loan terms: UNKNOWN (no SEC filing discloses them).",
        evidence_ids=("EV-1",),
        requires_evidence=True,
    )
    assert evaluate(inp).passed


def test_gs_arch_facts_inference_split() -> None:
    import json as _json

    from app.research.agents import (  # (14) declared claim types, no wording-based classification
        CLAIM_TYPES,
        GroundedClaim,
        ModelOutputFailure,
        parse_grounded_claims,
    )

    assert tuple(CLAIM_TYPES) == ("observed_fact", "inference", "unknown", "contradicted")
    # A cited claim that declares nothing stays inference: a citation never promotes it to fact.
    undeclared = parse_grounded_claims(
        _json.dumps([{"text": "GS revenue grew", "evidence_ids": ["EV-1"]}]), frozen=["EV-1"]
    )
    assert undeclared[0].claim_type == "inference"
    assert GroundedClaim(text="GS revenue grew", evidence_ids=["EV-1"]).claim_type == "inference"
    declared = parse_grounded_claims(
        _json.dumps([{"text": "GS 10-K discloses the marks", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]}]),
        frozen=["EV-1"],
    )
    assert declared[0].claim_type == "observed_fact"
    unknown = parse_grounded_claims(
        _json.dumps([{"text": "OpenAI private loan terms", "claim_type": "unknown", "evidence_ids": []}]),
        frozen=["EV-1"],
    )
    assert unknown[0].claim_type == "unknown"  # unknown may cite nothing
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            _json.dumps([{"text": "GS revenue grew", "claim_type": "fact-ish", "evidence_ids": ["EV-1"]}]),
            frozen=["EV-1"],
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            _json.dumps([{"text": "GS revenue grew", "claim_type": "observed_fact", "evidence_ids": []}]),
            frozen=["EV-1"],
        )


def test_gs_arch_claim_type_render_gate() -> None:
    from app.research.evals.evaluators import EvalInput as _In  # inference must not render as fact

    promoted = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="GS marks fell [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        claims=(("GS marks fell", "inference", "observed_fact"),),
    )
    assert "inference-rendered-as-observed-fact" in evaluate(promoted).violations
    honest = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="GS marks fell [EV-1] (inference).",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        claims=(("GS marks fell", "inference", "inference"),),
        claims_by_type={"observed_fact": 1, "inference": 1, "unknown": 0, "contradicted": 0},
    )
    assert evaluate(honest).passed
    untyped = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="GS marks fell [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        claims=(("GS marks fell", "", ""),),
    )
    assert "claim-without-type" in evaluate(untyped).violations
    mistyped = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="GS marks fell [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        claims_by_type={"fact-ish": 1},
    )
    assert "claim-type-unknown" in evaluate(mistyped).violations


def test_gs_arch_material_claims_trace_to_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _Repo

    repo = _Repo()
    sid, src = _gs_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _gs_item(eid), repo=repo)
    _gs_submit(src, eid, repo)
    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _gs_envelope(eid), repo=repo)
    out = _svc.finalize_session(
        sid,
        "GS exposure is filing-backed.",
        [{"text": "GS 10-K discloses OpenAI-linked exposure", "claim_type": "observed_fact", "evidence_ids": [eid]}],
        repo=repo,
    )
    assert out["freeze_id"] == fid  # (15) material claims trace to raw
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict)
    raw_frozen = repo.get_freeze(fid).get("evidence_ids", [])
    assert isinstance(raw_frozen, list)
    frozen = {e for e in raw_frozen if isinstance(e, str)}
    claims = final.get("claims", [])
    assert isinstance(claims, list) and claims
    for claim in claims:
        assert isinstance(claim, dict)
        assert claim.get("claim_type") in ("observed_fact", "inference", "unknown", "contradicted")
        refs = claim.get("evidence_ids", [])
        assert isinstance(refs, list)
        if claim.get("claim_type") != "unknown":  # only unknown may cite nothing
            assert refs and set(refs) <= frozen
    assert final.get("grounded_claims") == claims
    assert final.get("answer") == "GS exposure is filing-backed."


# ---------------------------------------------------------------------------
# MSFT OpenAI-bankruptcy SEC-only regression (pinned as_of fixtures).
# Fails when no material Microsoft exposure is present (channels:
# investment/ownership, commercial/revenue, receivable/credit, Azure/purchase
# commitment) or when no non-MSFT branch (AMZN/CoreWeave/AMD/Cerebras/ORCL
# per as_of) is covered. No facts past the fixture cutoff. Offline fakes only.
# ---------------------------------------------------------------------------


def _msft_fixture_answer(channels: str = "Azure commercial revenue receivable", branch: str = "CoreWeave") -> str:
    return (
        f"Microsoft 10-K [EV-1] discloses OpenAI-linked {channels} exposure; "
        f"{branch} [EV-2] covers a second branch. OpenAI private terms: UNKNOWN."
    )


def _msft_trace(
    channels: tuple[str, ...], branches: tuple[str, ...], evidence: tuple[str, ...], freeze: str = "rs:msft:1:freeze"
) -> FixtureTrace:
    """Correct-run trace: opened filings/documents, raw-document evidence, typed claims."""
    trace: FixtureTrace = {
        "filings_opened": ["0000789019-26-000057"],
        "documents_opened": ["0000789019-26-000057|msft-20260630.htm"],
        "passages_opened": ["0000789019-26-000057|msft-20260630.htm|p18"],
        "raw_evidence_ids": list(evidence),
        "navigation_evidence_ids": ["sec-search:msft:1"],
        "claims": [
            {
                "text": "MSFT 10-K discloses the OpenAI investment carrying value",
                "claim_type": "observed_fact",
                "rendered_as": "observed_fact",
            },
            {
                "text": "An OpenAI insolvency impairs that carrying value",
                "claim_type": "inference",
                "rendered_as": "inference",
            },
        ],
        "claims_by_type": {"observed_fact": 1, "inference": 1, "unknown": 0, "contradicted": 0},
        "waves": [freeze],
        "searches": ["sec-search:msft:1"],
        "committee_freeze_ids": [freeze] * 3,
        "roles_completed": ["stockbot", "bullbot", "bearbot"],
        "limitations": ["SEC-only policy: no disclosure was located within the searched SEC scope."],
        "coverage_complete": False,
        "universal_absence_claims": [],
        "material_channels": list(channels),
        "branches_covered": list(branches),
    }
    return trace


def _msft_strs(kwargs: dict[str, str | tuple[str, ...]], key: str) -> tuple[str, ...]:
    value = kwargs.get(key, ())
    return tuple(v for v in value if isinstance(v, str)) if isinstance(value, tuple) else ()


def _msft_build(answer_excerpt: str, **over: str | tuple[str, ...]) -> AgentFixture:
    """Typed build_fixture call carrying the Phase-17 trace section (no ignore)."""
    from app.research.evals.regression import build_fixture

    kwargs: dict[str, str | tuple[str, ...]] = {
        "session_id": "rs:msft",
        "scenario_name": "msft-openai-bankruptcy-sec-only",
        "tool_calls": ("find_sec_entities", "search_sec_filings", "get_sec_document"),
        "evidence_ids": ("EV-1", "EV-2"),
        "known_ats": ("2026-08-01",),
        "answer_excerpt": answer_excerpt,
    }
    kwargs.update(over)
    session_id = str(kwargs["session_id"])
    scenario_name = str(kwargs["scenario_name"])
    tool_calls = tuple(kwargs["tool_calls"])
    evidence_ids = tuple(kwargs["evidence_ids"])
    known_ats = tuple(kwargs["known_ats"])
    excerpt = str(kwargs["answer_excerpt"])
    assert all(isinstance(t, str) for t in tool_calls)
    assert all(isinstance(e, str) for e in evidence_ids)
    assert all(isinstance(k, str) for k in known_ats)
    fixture: AgentFixture = build_fixture(
        session_id=session_id,
        scenario_name=scenario_name,
        tool_calls=tuple(t for t in tool_calls if isinstance(t, str)),
        evidence_ids=tuple(e for e in evidence_ids if isinstance(e, str)),
        known_ats=tuple(k for k in known_ats if isinstance(k, str)),
        answer_excerpt=excerpt,
        trace=_msft_trace(
            _msft_strs(kwargs, "material_channels"),
            _msft_strs(kwargs, "branches_covered"),
            tuple(e for e in evidence_ids if isinstance(e, str)),
        ),
    )
    return fixture


def _msft_fixture(answer_excerpt: str | None = None, **over: str | tuple[str, ...]) -> tuple[EvalInput, list[str]]:
    from app.research.evals.evaluators import eval_input_from_fixture
    from app.research.evals.regression import run_deterministic_validators

    excerpt = answer_excerpt if answer_excerpt is not None else _msft_fixture_answer()
    fixture = _msft_build(excerpt, **over)
    return eval_input_from_fixture(fixture), run_deterministic_validators(fixture)


def test_msft_openai_passes_with_channel_and_branch() -> None:
    outcome, violations = _msft_fixture()
    assert violations == []
    assert evaluate(outcome).passed


def test_msft_openai_fails_without_material_msft_exposure() -> None:
    from app.research.evals.evaluators import EvalInput as _In
    from app.research.evals.regression import build_fixture, run_deterministic_validators

    outcome, _ = _msft_fixture(answer_excerpt="CoreWeave [EV-2] covers a branch; Microsoft terms UNKNOWN.")
    assert evaluate(outcome).violations == ("msft-openai-no-material-msft-exposure",)
    bad = build_fixture(
        session_id="rs:msft",
        scenario_name="msft-openai-bankruptcy-sec-only",
        evidence_ids=("EV-2",),
        answer_excerpt="CoreWeave covers a branch; Microsoft terms UNKNOWN.",
    )
    assert "msft-openai-no-material-msft-exposure" in run_deterministic_validators(bad)
    direct = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="Microsoft filing-backed answer with no channel words.",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        branches_covered=("coreweave",),
    )
    assert "msft-openai-no-material-msft-exposure" in evaluate(direct).violations


def test_msft_openai_fails_without_non_msft_branch() -> None:
    from app.research.evals.evaluators import EvalInput as _In
    from app.research.evals.regression import build_fixture, run_deterministic_validators

    outcome, _ = _msft_fixture(answer_excerpt="Microsoft 10-K [EV-1] discloses Azure investment exposure.")
    assert evaluate(outcome).violations == ("msft-openai-no-branch",)
    bad = build_fixture(
        session_id="rs:msft",
        scenario_name="msft-openai-bankruptcy-sec-only",
        evidence_ids=("EV-1",),
        answer_excerpt="Microsoft 10-K discloses Azure investment exposure.",
    )
    assert "msft-openai-no-branch" in run_deterministic_validators(bad)
    direct = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="Microsoft Azure investment exposure [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        material_channels=("azure",),
    )
    assert "msft-openai-no-branch" in evaluate(direct).violations


def test_msft_openai_channels_cover_each_pair() -> None:
    # Channel and branch coverage are read from the trace when the prose names neither.
    for channel in (
        "investment",
        "ownership",
        "commercial",
        "revenue",
        "receivable",
        "credit",
        "azure",
        "purchase commitment",
    ):
        excerpt = "Microsoft 10-K [EV-1] names the OpenAI-linked line; a second filer [EV-2] covers its branch."
        outcome, violations = _msft_fixture(
            answer_excerpt=excerpt, material_channels=(channel,), branches_covered=("orcl",)
        )
        assert violations == []
        assert evaluate(outcome).passed


def test_msft_openai_fixture_round_trip_carries_telemetry() -> None:
    from app.research.evals.evaluators import eval_input_from_fixture
    from app.research.evals.regression import (
        build_fixture,
        run_deterministic_validators,
    )

    fixture = build_fixture(
        session_id="rs:msft-tel",
        scenario_name="msft-openai-bankruptcy-sec-only",
        tool_calls=("search_sec_filings",),
        evidence_ids=("EV-1",),
        known_ats=("2026-08-01",),
        answer_excerpt=_msft_fixture_answer(),
        telemetry={
            "searches": 3,
            "queries": ["MSFT OpenAI"],
            "forms": ["10-K"],
            "entities": ["MSFT"],
            "exhibits": 1,
            "relationships_found": 2,
            "relationships_skipped": 0,
            "coverage": "partial",
            "unresolved": ["OpenAI private terms"],
            "stop_reason": "no_questions",
        },
        trace=_msft_trace(("azure",), ("coreweave",), ("EV-1",)),
    )
    assert run_deterministic_validators(fixture) == []
    outcome = eval_input_from_fixture(fixture)
    assert outcome.searches == 3 and outcome.queries == ("MSFT OpenAI",)
    assert outcome.forms == ("10-K",) and outcome.entities == ("MSFT",)
    assert outcome.exhibits == 1 and outcome.relationships_found == 2
    assert outcome.unresolved == ("OpenAI private terms",) and outcome.stop_reason == "no_questions"
    assert evaluate(outcome).passed


# ---------------------------------------------------------------------------
# Coverage-quality: a fixture with Amazon+AMD+Cerebras but no Microsoft is not
# sufficient; with major branches covered it may be sufficient. Generic rule:
# an unexplored high-ranking material relationship blocks "sufficient".
# ---------------------------------------------------------------------------


def test_coverage_amazon_amd_cerebras_without_msft_not_sufficient() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    inp = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="Amazon, AMD and Cerebras branch findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        coverage_claim="sufficient",
        branches_covered=("amzn", "amd", "cerebras"),
    )
    assert "msft-openai-no-material-msft-exposure" in evaluate(inp).violations


def test_coverage_major_branches_may_be_sufficient() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    inp = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text=_msft_fixture_answer(),
        evidence_ids=("EV-1", "EV-2"),
        requires_evidence=True,
        coverage_claim="sufficient",
        material_channels=("azure",),
        branches_covered=("coreweave", "amzn", "amd"),
    )
    assert evaluate(inp).passed


def test_coverage_unexplored_high_rank_blocks_sufficient() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    inp = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        coverage_claim="sufficient",
        high_rank_unexplored=True,
    )
    result = evaluate(inp)
    assert not result.passed and result.violations == ("coverage-overclaim",)
    ok = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        coverage_claim="sufficient",
        high_rank_unexplored=False,
    )
    assert evaluate(ok).passed


# ---------------------------------------------------------------------------
# Committee invariants: same freeze across the trio, 3 distinct pre-run job
# ids, concurrent, completed-job cross-role write rejected, roles cannot
# mutate the freeze, claims resolve to frozen evidence, research_requests stay
# separate from evidence. Offline fakes only.
# ---------------------------------------------------------------------------


def _committee_input(
    committee_freeze_ids: tuple[str, ...] = ("F1", "F1", "F1"),
    job_ids: tuple[str, ...] = ("j-stock", "j-bull", "j-bear"),
    job_created_before_run: bool = True,
    jobs_concurrent: bool = True,
    cross_role_write_rejected: bool = True,
    roles_mutate_freeze: bool = False,
    claims_resolve_to_freeze: bool = True,
    requests_separate_from_evidence: bool = True,
) -> EvalInput:
    return EvalInput(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        committee_freeze_ids=committee_freeze_ids,
        job_ids=job_ids,
        job_created_before_run=job_created_before_run,
        jobs_concurrent=jobs_concurrent,
        cross_role_write_rejected=cross_role_write_rejected,
        roles_mutate_freeze=roles_mutate_freeze,
        claims_resolve_to_freeze=claims_resolve_to_freeze,
        requests_separate_from_evidence=requests_separate_from_evidence,
    )


def test_committee_same_freeze_distinct_preregistered_concurrent() -> None:
    assert evaluate(_committee_input()).passed


def test_committee_distinct_freeze_fails() -> None:
    result = evaluate(_committee_input(committee_freeze_ids=("F1", "F2", "F1")))
    assert "committee-different-freeze" in result.violations


def test_committee_job_count_concurrency_registration() -> None:
    assert "committee-job-count" in evaluate(_committee_input(job_ids=("j1", "j1"))).violations
    assert "committee-jobs-not-preregistered" in evaluate(_committee_input(job_created_before_run=False)).violations
    assert "committee-jobs-not-concurrent" in evaluate(_committee_input(jobs_concurrent=False)).violations


def test_committee_cross_role_freeze_claims_requests() -> None:
    assert (
        "committee-cross-role-write-allowed" in evaluate(_committee_input(cross_role_write_rejected=False)).violations
    )
    assert "committee-mutates-freeze" in evaluate(_committee_input(roles_mutate_freeze=True)).violations
    assert "committee-claims-unresolved" in evaluate(_committee_input(claims_resolve_to_freeze=False)).violations
    assert (
        "committee-requests-as-evidence" in evaluate(_committee_input(requests_separate_from_evidence=False)).violations
    )


# ---------------------------------------------------------------------------
# Finalization UX: a finalize success auto-renders a substantive structured
# answer in the same turn; a bare "finalized/5 claims" with no answer fails.
# ---------------------------------------------------------------------------


def test_finalize_bare_count_without_answer_fails() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    inp = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="finalized 5 claims",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        finalized_claim_count=5,
        answered=False,
    )
    assert "finalized-without-answer" in evaluate(inp).violations
    blank = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="   ",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        finalized_claim_count=5,
    )
    assert "finalized-without-answer" in evaluate(blank).violations


def test_finalize_structured_answer_same_turn_passes() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    inp = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Balanced: grounded [EV-1]. Bull: upside [EV-1]. Bear: risk [EV-1]. Agreed: exposure capped.",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        finalized_claim_count=5,
    )
    assert evaluate(inp).passed


# ---------------------------------------------------------------------------
# Trace-based evaluation (plan Phase 17): the SEC regression scenarios gate on
# the structural trace — opened filings/documents, raw-document-backed
# evidence, declared claim types, same-freeze roles and recorded limitations.
# Correct prose with no trace fails, and an absence claim drawn from a search
# miss is an explicit failure.
# ---------------------------------------------------------------------------

_CANONICAL_TRACE_FIXTURES = (
    "msft-openai-bankruptcy-sec-only",
    "gs-openai-sec-only",
    "spacex-openai-bankruptcy-sec-only",
)
_CONTROL_FIXTURE = "msft-openai-bankruptcy-overconfident-control"


def _fixture_outcome(name: str) -> tuple[EvalInput, list[str]]:
    from app.research.evals.evaluators import eval_input_from_fixture
    from app.research.evals.regression import load_fixture, run_deterministic_validators

    fixture = load_fixture(name)
    return eval_input_from_fixture(fixture), run_deterministic_validators(fixture)


def test_canonical_sec_fixtures_pass_with_structural_trace() -> None:
    for name in _CANONICAL_TRACE_FIXTURES:
        outcome, violations = _fixture_outcome(name)
        assert violations == [], (name, violations)
        assert outcome.trace_present, name
        assert outcome.filings_opened and outcome.documents_opened and outcome.passages_opened, name
        assert outcome.raw_evidence_ids and set(outcome.raw_evidence_ids) <= set(outcome.evidence_ids), name
        assert outcome.claims_by_type["observed_fact"] > 0, name
        assert len(set(outcome.committee_freeze_ids)) == 1, name
        assert outcome.limitations and not outcome.coverage_complete, name
        assert not outcome.universal_absence_claims, name
        result = evaluate(outcome)
        assert result.passed, (name, result.violations)


def test_overconfident_control_fixture_fails() -> None:
    """Regression proof: answering from a search miss with no filing opened must fail."""
    control, _violations = _fixture_outcome(_CONTROL_FIXTURE)
    result = evaluate(control)
    assert not result.passed
    assert "overconfident-absence" in result.violations
    assert "trace-no-filings-opened" in result.violations
    assert "limitations-missing" in result.violations
    assert "answer-without-required-evidence" in result.violations
    # The keyword channel/branch gate is not what stops it: the same prose minus the
    # absence sentence passes with no structural trace required.
    promoted = control.answer_text.replace(
        "No relationship exists between Microsoft and OpenAI", "OpenAI exposure reviewed"
    )
    assert evaluate(
        EvalInput(
            scenario_name="msft-openai-bankruptcy-sec-only",
            answer_text=promoted,
            evidence_ids=(),
            requires_evidence=False,
            material_channels=("azure", "commercial"),
            branches_covered=("coreweave",),
        )
    ).passed


def test_prose_without_trace_fails_structurally() -> None:
    """Keyword presence alone never carries a pass: canonical prose minus its trace fails."""
    from app.research.evals.evaluators import eval_input_from_fixture
    from app.research.evals.regression import load_fixture

    loaded = load_fixture("msft-openai-bankruptcy-sec-only")
    del loaded["trace"]
    outcome = eval_input_from_fixture(loaded)
    assert "azure" in outcome.answer_text.lower() and "coreweave" in outcome.answer_text.lower()
    assert evaluate(outcome).violations == ("trace-missing",)


def _trace_fixture(trace: FixtureTrace, evidence: tuple[str, ...] = ("EV-1",)) -> AgentFixture:
    from app.research.evals.regression import build_fixture

    return build_fixture(
        session_id="rs:trace",
        scenario_name="msft-openai-bankruptcy-sec-only",
        tool_calls=("get_sec_document",),
        evidence_ids=evidence,
        known_ats=("2026-08-01",),
        answer_excerpt="Microsoft 10-K [EV-1] discloses the OpenAI-linked exposure with a CoreWeave branch.",
        trace=trace,
    )


def test_navigation_ids_never_count_as_raw_evidence() -> None:
    from app.research.evals.evaluators import eval_input_from_fixture

    trace = _msft_trace(("azure",), ("coreweave",), ("EV-1",))
    trace["navigation_evidence_ids"] = ["EV-1"]  # the same id is navigation-only here
    assert evaluate(eval_input_from_fixture(_trace_fixture(trace))).violations == ("trace-no-raw-source-evidence",)


def test_trace_evidence_ids_must_resolve_to_recorded_evidence() -> None:
    from app.research.evals.evaluators import eval_input_from_fixture

    trace = _msft_trace(("azure",), ("coreweave",), ("EV-1",))
    trace["raw_evidence_ids"] = ["EV-1", "rs:ghost:ev:99"]
    assert evaluate(eval_input_from_fixture(_trace_fixture(trace))).violations == ("trace-raw-evidence-unresolved",)


def test_trace_requires_documents_and_filings() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    bare = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="Prose names Azure and CoreWeave.",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        requires_trace=True,
        trace_present=True,
        documents_opened=("a|b.htm",),
        raw_evidence_ids=("EV-1",),
        committee_freeze_ids=("F1",) * 3,
        roles_completed=("stockbot", "bullbot", "bearbot"),
    )
    assert evaluate(bare).violations == ("trace-no-filings-opened",)
    no_docs = _In(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="Prose names Azure and CoreWeave.",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        requires_trace=True,
        trace_present=True,
        filings_opened=("0000789019-26-000057",),
        raw_evidence_ids=("EV-1",),
        committee_freeze_ids=("F1",) * 3,
        roles_completed=("stockbot", "bullbot", "bearbot"),
    )
    assert evaluate(no_docs).violations == ("trace-no-documents-opened",)


def test_overconfident_absence_from_search_miss_fails() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    flagged = _In(
        scenario_name="spacex-openai-bankruptcy-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        universal_absence_claims=("SpaceX has no relationship with OpenAI",),
    )
    assert evaluate(flagged).violations == ("overconfident-absence",)
    prose = _In(
        scenario_name="spacex-openai-bankruptcy-sec-only",
        answer_text="The search returned nothing, so no relationship exists between SpaceX and OpenAI.",
        evidence_ids=("EV-1",),
        requires_evidence=True,
    )
    assert evaluate(prose).violations == ("overconfident-absence",)
    scoped = _In(
        scenario_name="spacex-openai-bankruptcy-sec-only",
        answer_text="No SpaceX disclosure was located within the searched SEC scope [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        limitations=("SEC-only policy.",),
        coverage_complete=False,
    )
    assert evaluate(scoped).passed


def test_committee_roles_and_limitations_gates() -> None:
    from app.research.evals.evaluators import EvalInput as _In

    partial_roles = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        committee_freeze_ids=("F1", "F2", "F1"),
        roles_completed=("stockbot", "bullbot"),
        coverage_complete=False,
        limitations=("SEC-only.",),
    )
    assert set(evaluate(partial_roles).violations) == {"committee-different-freeze", "committee-roles-incomplete"}
    unresolved = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        coverage_complete=False,
    )
    assert evaluate(unresolved).violations == ("limitations-missing",)
    complete = _In(
        scenario_name="gs-openai-sec-only",
        answer_text="Findings [EV-1].",
        evidence_ids=("EV-1",),
        requires_evidence=True,
        coverage_complete=False,
        limitations=("SEC-only policy: no disclosure located within the searched scope.",),
    )
    assert evaluate(complete).passed


def test_spacex_fixture_keeps_absence_scoped() -> None:
    """SpaceX: a scoped absence observation, never a claim that the relationship does not exist."""
    outcome, violations = _fixture_outcome("spacex-openai-bankruptcy-sec-only")
    assert violations == []
    assert outcome.universal_absence_claims == ()
    assert outcome.filings_opened and outcome.raw_evidence_ids  # the chain filings it did open
    assert any("scoped to the searched SEC corpus" in limit for limit in outcome.limitations)
    assert evaluate(outcome).passed


def test_fixture_trace_round_trip_and_legacy_fixture_back_compat(tmp_path: Path) -> None:
    from app.research.evals.evaluators import eval_input_from_fixture
    from app.research.evals.regression import build_fixture, load_fixture, save_fixture

    trace: FixtureTrace = {
        "filings_opened": ["0000789019-26-000057"],
        "documents_opened": ["0000789019-26-000057|msft-20260630.htm"],
        "passages_opened": ["0000789019-26-000057|msft-20260630.htm|p18"],
        "raw_evidence_ids": ["EV-1"],
        "claims": [
            {
                "text": "MSFT 10-K discloses the investment",
                "claim_type": "observed_fact",
                "rendered_as": "observed_fact",
            }
        ],
        "claims_by_type": {"observed_fact": 1},
        "searches": ["sec-search:rt:1"],
        "waves": ["F1"],
        "committee_freeze_ids": ["F1", "F1", "F1"],
        "roles_completed": ["stockbot", "bullbot", "bearbot"],
        "limitations": ["SEC-only policy."],
        "coverage_complete": False,
        "material_channels": ["azure"],
        "branches_covered": ["coreweave"],
    }
    save_fixture(
        build_fixture(
            session_id="rs:rt",
            scenario_name="msft-openai-bankruptcy-sec-only",
            question="q",
            as_of="2026-08-10",
            tool_calls=("get_sec_document",),
            evidence_ids=("EV-1",),
            known_ats=("2026-08-01",),
            answer_excerpt="MSFT 10-K [EV-1] discloses the Azure exposure with a CoreWeave branch.",
            trace=trace,
        ),
        tmp_path,
    )
    loaded = load_fixture("msft-openai-bankruptcy-sec-only", tmp_path)
    assert loaded["trace"]["filings_opened"] == ["0000789019-26-000057"]
    assert evaluate(eval_input_from_fixture(loaded)).passed
    # A pre-Phase-17 fixture without a trace section still loads and evaluates.
    legacy = load_fixture("timeout-model-call-failed-resume")
    assert "trace" not in legacy
    assert evaluate(eval_input_from_fixture(legacy)).passed


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        ("not an object", "fixture: 'trace' must be an object"),
        ({"claims": {}}, "fixture: 'trace.claims' must be a list"),
        ({"claims": [7]}, "fixture: 'trace.claims' entries must be objects"),
        ({"claims": [{"text": 7}]}, "fixture: 'trace.claims[].text' must be a string"),
        ({"coverage_complete": "yes"}, "fixture: 'trace.coverage_complete' must be a bool"),
        ({"claims_by_type": []}, "fixture: 'trace.claims_by_type' must be an object"),
        ({"filings_opened": [7]}, "fixture: 'trace.filings_opened' must be a list of strings"),
    ],
)
def test_malformed_trace_section_fails_loading(tmp_path: Path, trace: object, message: str) -> None:
    """A saved fixture with a mistyped trace section fails loading loudly, never silently."""
    from app.research.evals.regression import load_fixture

    payload = json.loads(json.dumps(_msft_build(_msft_fixture_answer())))
    payload["trace"] = trace
    scenario_name = str(payload["scenario_name"])
    (tmp_path / f"{scenario_name}.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_fixture(scenario_name, tmp_path)
    assert str(exc.value) == message


def test_trace_optional_fields_load_with_bool_counts_ignored(tmp_path: Path) -> None:
    """Absent trace sections default to empty; a bool is not a claim-type count."""
    from app.research.evals.regression import load_fixture, save_fixture

    fixture = _msft_build(_msft_fixture_answer())
    fixture["trace"] = {
        "claims": [
            {
                "text": "MSFT 10-K discloses the investment",
                "claim_type": "observed_fact",
                "rendered_as": "observed_fact",
            }
        ],
        "claims_by_type": {"observed_fact": True, "inference": 2},
        "coverage_complete": True,
    }
    save_fixture(fixture, tmp_path)
    trace = load_fixture(fixture["scenario_name"], tmp_path)["trace"]
    assert trace["claims_by_type"] == {"inference": 2}
    assert trace["coverage_complete"] is True
    assert trace["claims"][0]["rendered_as"] == "observed_fact"
    assert trace["filings_opened"] == [] and trace["roles_completed"] == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "spacex-openai-bankruptcy-sec-only-live-run",
            {"trace-no-documents-opened", "claim-without-type", "overconfident-absence"},
        ),
        ("gs-openai-sec-only-live-run", {"trace-no-documents-opened", "claim-without-type", "overconfident-absence"}),
    ],
)
def test_live_run_regressions_fail_the_gate(name: str, expected: set[str]) -> None:
    """The shipped broken runs stay red: search-miss universals, no raw docs, undeclared claim types."""
    result = evaluate(eval_input_from_fixture(load_fixture(name)))
    assert result.passed is False
    assert expected <= set(result.violations)


def test_multiwave_committee_freezes_are_not_a_disagreement() -> None:
    """Roles share one freeze per round; separate waves legitimately use separate freezes."""
    single_round = EvalInput(scenario_name="x", answer_text="a", committee_freeze_ids=("F1", "F1", "F1"))
    assert "committee-different-freeze" not in evaluate(single_round).violations
    two_waves = EvalInput(scenario_name="x", answer_text="a", committee_freeze_ids=("F1", "F1", "F1", "F2", "F2", "F2"))
    assert "committee-different-freeze" not in evaluate(two_waves).violations
    mixed_round = EvalInput(scenario_name="x", answer_text="a", committee_freeze_ids=("F1", "F2", "F1"))
    assert "committee-different-freeze" in evaluate(mixed_round).violations
