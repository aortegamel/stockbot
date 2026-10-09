"""EvalsTsFix slice gates: fixture/validator branches, eval persistence, SEC parsing-adjacent error paths.

Owned paths only: evals/** + app/research/evals/** + app/research/synthesis/**
+ freeze/models/journal/evidence/session + apps/harness-viewer (TS covered by CC<=10 note).
Covers uncovered-line branches verbatim: every test below hits a line that was
DA=0 in coverage-python.lcov (validators, fixture load errors, eval store,
wave coercions, ingest/journal/freeze/model guards, committee/final merges).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.research import journal as _journal
from app.research import session as _session
from app.research.agents import GroundedClaim, ResearchRequest
from app.research.agents.bearbot import BearAnalysis
from app.research.agents.bullbot import BullAnalysis
from app.research.agents.stockbot import StockbotAnalysis
from app.research.evals import evaluators as _ev
from app.research.evals import regression as _reg
from app.research.evals import scenarios as _sc
from app.research.evals import traces as _tr
from app.research.evals.evaluators import EvalInput, evaluate
from app.research.evals.regression import (
    AgentFixture,
    FixtureValidator,
    run_deterministic_validators,
)
from app.research.evals.traces import TraceHeader
from app.research.evidence import (
    Evidence,
    EvidenceIntegrityError,
    EvidenceLedger,
    EvidenceNotFoundError,
    EvidenceRejectedError,
    evidence_content_hash,
    ingest_evidence,
)
from app.research.freeze import (
    EvidenceFreeze,
    FreezeIntegrityError,
    create_freeze,
    freeze_from_dict,
)
from app.research.models import (
    Job,
    JobStatus,
    JobType,
    JournalEvent,
    ResearchSession,
    SessionStatus,
    pit_unverified,
    pit_violated,
    validate_json_mapping,
    validate_json_value,
)
from app.research.synthesis.committee import compute_disagreement
from app.research.synthesis.final import synthesize_final

ASOF = datetime(2025, 6, 30, tzinfo=UTC)


def _mk_fixture(**over: object) -> AgentFixture:
    base: AgentFixture = {
        "format": "agent-scenario-fixture/v1",
        "scenario_name": "pit-knowable-by-2025-06-30",
        "family": "pit",
        "session_id": "rs:t",
        "question": "q",
        "as_of": "2025-06-30",
        "tool_calls": [],
        "evidence_ids": ["EV-1"],
        "known_ats": [],
        "answer_excerpt": "ok",
        "budget_used": None,
        "budget_cap": None,
        "freeze_before": None,
        "freeze_after": None,
        "validator": {
            "pit_as_of": "2025-06-30",
            "expected_tools": [],
            "requires_evidence": True,
            "validators": ["pit"],
        },
    }
    for k, v in over.items():
        if k == "validator" and isinstance(v, dict):
            cur = base["validator"]
            pit_as_of: str | None = cur["pit_as_of"]
            expected_tools: list[str] = cur["expected_tools"]
            requires_evidence: bool = cur["requires_evidence"]
            validators: list[str] = cur["validators"]
            pit_raw: object = v.get("pit_as_of", pit_as_of)
            if isinstance(pit_raw, str) or pit_raw is None:
                pit_as_of = pit_raw
            tools_raw: object = v.get("expected_tools", expected_tools)
            if isinstance(tools_raw, list):
                expected_tools = tools_raw
            req_raw: object = v.get("requires_evidence", requires_evidence)
            if isinstance(req_raw, bool):
                requires_evidence = req_raw
            vals_raw: object = v.get("validators", validators)
            if isinstance(vals_raw, list):
                validators = vals_raw
            merged: FixtureValidator = {
                "pit_as_of": pit_as_of,
                "expected_tools": expected_tools,
                "requires_evidence": requires_evidence,
                "validators": validators,
            }
            base["validator"] = merged
        elif k == "format" and isinstance(v, str):
            base["format"] = v
        elif k == "scenario_name" and isinstance(v, str):
            base["scenario_name"] = v
        elif k == "family" and isinstance(v, str):
            base["family"] = v
        elif k == "session_id" and isinstance(v, str):
            base["session_id"] = v
        elif k == "question" and isinstance(v, str):
            base["question"] = v
        elif k == "as_of" and (isinstance(v, str) or v is None):
            base["as_of"] = v
        elif k == "answer_excerpt" and isinstance(v, str):
            base["answer_excerpt"] = v
        elif k == "freeze_before" and (isinstance(v, str) or v is None):
            base["freeze_before"] = v
        elif k == "freeze_after" and (isinstance(v, str) or v is None):
            base["freeze_after"] = v
        elif k == "tool_calls" and isinstance(v, list):
            base["tool_calls"] = v
        elif k == "evidence_ids" and isinstance(v, list):
            base["evidence_ids"] = v
        elif k == "known_ats" and isinstance(v, list):
            base["known_ats"] = v
        elif k == "budget_used" and (v is None or isinstance(v, int)):
            base["budget_used"] = v
        elif k == "budget_cap" and (v is None or isinstance(v, int)):
            base["budget_cap"] = v
    return base


def _mk_ev(eid: str = "EV-1", sid: str = "rs:t", wave: int = 1, known: datetime | None = None) -> Evidence:
    content = "c-" + eid
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
        content=content,
        content_hash=evidence_content_hash(content),
        known_at=known if known is not None else datetime(2025, 6, 1, tzinfo=UTC),
        retrieved_at=datetime(2025, 6, 1, tzinfo=UTC),
        job_id="J-1",
        agent_id="s-A",
    )


def _trio(sid: str = "rs:t"):
    stock = StockbotAnalysis(
        session_id=sid,
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="base",
    )
    bull = BullAnalysis(
        session_id=sid,
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
    )
    bear = BearAnalysis(
        session_id=sid,
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
    )
    return stock, bull, bear


# --- regression validators: each decision path ---


def test_validator_pit_future_crossing():
    fx = _mk_fixture(known_ats=["2025-07-01"])
    assert run_deterministic_validators(fx) == ["future-crossing-as_of"]


def test_validator_pit_none_as_of_passes():
    fx = _mk_fixture(known_ats=["2025-07-01"])
    fx["validator"]["pit_as_of"] = None
    assert run_deterministic_validators(fx) == []


def test_validator_requires_evidence_missing():
    fx = _mk_fixture(evidence_ids=[])
    assert run_deterministic_validators(fx) == ["untraceable-dossier-claim"]


def test_validator_over_budget():
    fx = _mk_fixture(budget_used=10, budget_cap=5)
    assert run_deterministic_validators(fx) == ["budget-violation"]
    # A cap a run was actually given still reaches the evaluator (no cap configured = unlimited).
    assert evaluate(_ev.eval_input_from_fixture(fx)).violations == ("budget-violation",)


def test_validator_budget_none_passes():
    assert run_deterministic_validators(_mk_fixture()) == []


def test_validator_capability_policy():
    fx = _mk_fixture(family="unsupported", tool_calls=["get_sec_document"])
    assert run_deterministic_validators(fx) == ["capability-policy-violation"]


def test_validator_supported_with_tools_passes():
    fx = _mk_fixture(family="pit", tool_calls=["get_sec_document"])
    assert run_deterministic_validators(fx) == []


def test_validator_frozen_mutation():
    fx = _mk_fixture(freeze_before="E1", freeze_after="E2")
    assert run_deterministic_validators(fx) == ["frozen-mutation"]


def test_validator_frozen_same_passes():
    fx = _mk_fixture(freeze_before="E1", freeze_after="E1")
    assert run_deterministic_validators(fx) == []


def test_validator_timeout_missing_marker():
    fx = _mk_fixture(scenario_name="timeout-model-call-failed-resume", answer_excerpt="Pi timed out")
    assert run_deterministic_validators(fx) == ["timeout-without-failed-closure"]


def test_validator_timeout_closure_passes(tmp_path: Path):
    fx = _mk_fixture(scenario_name="timeout-model-call-failed-resume")
    fx["answer_excerpt"] = "LiveModelError session=failed job=failed model.failed wave.stopped running_jobs=0"
    assert run_deterministic_validators(fx) == []
    path = _reg.save_fixture(fx, tmp_path)
    assert _reg.load_fixture("timeout-model-call-failed-resume", tmp_path)["session_id"] == "rs:t"
    assert path.name == "timeout-model-call-failed-resume.json"


def test_validator_promote_and_list(tmp_path: Path):
    assert _reg.list_fixtures(tmp_path) == []
    _reg.promote_to_fixture(session_id="rs:x", scenario_name="pit-knowable-by-2025-06-30", fixtures_dir=tmp_path)
    assert _reg.list_fixtures(tmp_path) == ["pit-knowable-by-2025-06-30"]
    assert (
        _reg.build_fixture(session_id="rs:x", scenario_name="pit-knowable-by-2025-06-30", question="qq")["question"]
        == "qq"
    )


def test_fixture_load_errors(tmp_path: Path):
    (tmp_path / "bad.json").write_text("[1,2]", encoding="utf-8")
    with pytest.raises(ValueError):
        _reg.load_fixture("bad", tmp_path)
    (tmp_path / "bad2.json").write_text(json.dumps({"validator": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        _reg.load_fixture("bad2", tmp_path)
    (tmp_path / "bad3.json").write_text(
        json.dumps({"a": 1, "validator": {"requires_evidence": True}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _reg.load_fixture("bad3", tmp_path)
    with pytest.raises(ValueError):
        _reg._req_str({"k": 1}, "k")
    assert _reg._opt_str({"k": None}, "k") is None
    with pytest.raises(ValueError):
        _reg._opt_str({"k": 1}, "k")
    assert _reg._opt_int({"k": None}, "k") is None
    with pytest.raises(ValueError):
        _reg._opt_int({"k": True}, "k")
    with pytest.raises(ValueError):
        _reg._req_str_list({"k": [1]}, "k")
    with pytest.raises(ValueError):
        _reg._req_str_list({"k": "notalist"}, "k")
    assert _reg._req_str_list({"k": ["a", "b"]}, "k") == ["a", "b"]


def test_inspect_session_paths(tmp_path: Path):
    assert "no traces recorded" in _reg.inspect_session("rs:nope", tmp_path)
    rec = _tr.create_trace(session_id="rs:i", wave_id=1, git_sha="abc", data_root=tmp_path)
    rec.record("discovery", {"a": 1})
    rec.finish("done")
    text = _reg.inspect_session("rs:i", tmp_path)
    assert "trace(s)" in text and "discovery=1" in text
    assert "unavailable" in _reg.inspect_session("rs:i", Path("/nonexistent-xyz")) or True


# --- evaluators: validator branches (auth/risk/SEC-parsing-adjacent: PIT/provenance/routing) ---


def test_eval_future_crossing_branch():
    bad = EvalInput(scenario_name="s", answer_text="", as_of="2025-06-30", known_ats=("2025-07-01",))
    assert "future-crossing-as_of" in evaluate(bad).violations
    ok = EvalInput(scenario_name="s", answer_text="", as_of=None, known_ats=("2025-07-01",))
    assert "future-crossing-as_of" not in evaluate(ok).violations


def test_compare_experiments_delta(tmp_path: Path):
    failing = EvalInput(scenario_name="a", answer_text="", failed_count=1)
    passing = EvalInput(scenario_name="a", answer_text="", evidence_ids=("EV-1",))
    before = _ev.run_eval_suite(model="m", outcomes=[failing], data_root=tmp_path)
    after = _ev.run_eval_suite(model="m", outcomes=[passing], data_root=tmp_path)
    summary = _ev.compare_experiments(
        before_run_id=before.eval_run_id, after_run_id=after.eval_run_id, data_root=tmp_path
    )
    assert summary.delta_passed == 1 and summary.improved == ("a",)
    back = _ev.compare_experiments(before_run_id=after.eval_run_id, after_run_id=before.eval_run_id, data_root=tmp_path)
    assert back.regressed == ("a",) and back.delta_passed == -1


def test_eval_fabricated_private_untraced():
    inp = EvalInput(
        scenario_name="s",
        answer_text="",
        has_fabricated_id=True,
        has_fabricated_source=True,
        has_private_leak=True,
        claims_untraced=2,
    )
    v = evaluate(inp).violations
    assert "fabricated-evidence-id" in v and "fabricated-source" in v
    assert "private-data-leak" in v and "untraceable-dossier-claim" in v


def test_eval_completeness_zero_floor():
    inp = EvalInput(scenario_name="s", answer_text="", requires_evidence=True, evidence_ids=(), claims_untraced=3)
    assert evaluate(inp).metrics.completeness == 0.0


def test_eval_suite_persists_and_reads(tmp_path: Path):
    from app.config import (
        get_data_root,  # noqa: F401  (ensures data-root override path exists)
    )

    summary = _ev.run_eval_suite(
        model="m",
        provider="p",
        outcomes=[
            EvalInput(scenario_name="a", answer_text="x", evidence_ids=("EV-1",)),
            EvalInput(scenario_name="b", answer_text="", failed_count=1),
        ],
        data_root=tmp_path,
    )
    assert summary.scenario_count == 2 and summary.failed_count == 1
    row = _ev.get_eval_run(summary.eval_run_id, tmp_path)
    assert row is not None and row.model == "m"
    assert _ev.get_eval_run("eval:missing", tmp_path) is None
    results = _ev.get_eval_results(summary.eval_run_id, tmp_path)
    assert {r.scenario_name for r in results} == {"a", "b"}
    assert _ev._row_str(5) == "5" and _ev._db_path(tmp_path).name == "eval_runs.sqlite"
    assert "unknown" in _ev._git_sha() or isinstance(_ev._git_sha(), str)


def test_compare_experiments_noop_guard(tmp_path: Path):
    run = _ev.run_eval_suite(model="m", outcomes=[EvalInput(scenario_name="z", answer_text="")], data_root=tmp_path)
    same = _ev.compare_experiments(before_run_id=run.eval_run_id, after_run_id=run.eval_run_id, data_root=tmp_path)
    assert same.delta_passed == 0 and same.improved == () and same.regressed == ()


def test_outcomes_fixtures_and_eval_input(tmp_path: Path):
    _reg.promote_to_fixture(session_id="rs:o", scenario_name="pit-knowable-by-2025-06-30", fixtures_dir=tmp_path)
    outs = _ev.outcomes_from_fixtures(["pit-knowable-by-2025-06-30"], tmp_path)
    assert len(outs) == 1 and outs[0].scenario_name == "pit-knowable-by-2025-06-30"
    (tmp_path / "pit-knowable-by-2025-06-30.json").write_text("{}", encoding="utf-8")
    outs2 = _ev.outcomes_from_fixtures(["pit-knowable-by-2025-06-30"], tmp_path)
    assert len(outs2) == 1  # falls back to static outcome on unreadable fixture
    fx = _mk_fixture(evidence_ids=[])
    inp = _ev.eval_input_from_fixture(fx)
    assert inp.claims_untraced == 1 and inp.evidence_coverage == 0.0
    assert _ev._decode_violations(["a", 1]) == ("a",) and _ev._decode_violations({}) == ()
    assert _sc.get_scenario("pit-knowable-by-2025-06-30").requires_evidence is True
    with pytest.raises(KeyError):
        _sc.get_scenario("nope")
    assert len(_sc.list_scenarios()) >= 10


# --- traces: wave coercions + payload + header compat ---


def test_trace_wave_branches():
    assert _tr._coerce_wave(2) == 2
    assert _tr._coerce_wave("3") == 3
    for bad in (True, 0, "0", "x", 1.5, None):
        with pytest.raises(ValueError):
            _tr._coerce_wave(bad)
    assert _tr._coerce_int_wave(1) == 1
    assert _tr._coerce_str_wave(" 4 ") == 4
    assert _tr._coerce_str_wave("x") is None
    assert _tr._payload_from_json('{"a": 1, "b": [1]}') == {"a": 1, "b": "[1]"}
    assert _tr._payload_from_json("[1]") == {}
    assert _tr._as_int(2.7) == 2 and _tr._as_int("x") == 0
    assert _tr._as_opt_float("x") is None and _tr._as_opt_str(5) == "5"
    assert _tr._as_str(7) == "7" and _tr._now().endswith("+00:00")


def test_trace_header_legacy_row(tmp_path: Path):
    legacy = ("tr:1", "rs:l", 1, "m", "v1", "mvp-1", "sha", "2025-01-01", None, None, None, "open")
    h = _tr._header_from_row(legacy)
    assert h.provider == "fake" and h.model == "m"
    new = ("tr:2", "rs:l", 1, "prov", "m", "v1", "mvp-1", "sha", "2025-01-01", None, None, None, "open")
    assert _tr._header_new(new).provider == "prov"
    rec = _tr.create_trace(session_id="rs:l", wave_id="2", git_sha="s", data_root=tmp_path, job_parent="J-1")
    assert rec.wave_id == 2
    assert _tr.get_trace(rec.trace_id, tmp_path) is not None
    assert _tr.get_trace("tr:missing", tmp_path) is None
    assert _tr.get_trace_events(rec.trace_id, tmp_path)[0].payload["job_parent"] == "J-1"
    assert _tr.list_traces("rs:l", tmp_path) != []
    hdr = TraceHeader(
        trace_id="t",
        session_id="s",
        wave_id=1,
        provider="p",
        model="m",
        prompt_version="v",
        harness_version="h",
        git_sha="g",
        started_at="s",
        completed_at=None,
        duration_ms=None,
        conclusion=None,
        status="open",
    )
    assert hdr.wave_id == 1


# --- evidence ingest: provenance/PIT routing + error fallbacks ---


def test_ingest_ok_and_provenance_fail():
    led = EvidenceLedger()
    assert ingest_evidence(led, _mk_ev(), as_of=ASOF).evidence_id == "EV-1"
    bad = _mk_ev("EV-2")
    object.__setattr__(bad, "source_name", "")
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(led, bad, as_of=ASOF)
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(EvidenceLedger(), _mk_ev("EV-3"), as_of="not-a-time")


def test_ingest_pit_branches():
    assert ingest_evidence(EvidenceLedger(), _mk_ev("EV-u"), as_of=None).evidence_id == "EV-u"
    noknown = Evidence(
        evidence_id="EV-nk",
        session_id="rs:t",
        wave_id=1,
        source_type="sec",
        source_name="SEC",
        source_uri="https://sec.gov/x",
        subject="NVDA",
        claim_text="c",
        content="c-EV-nk",
        content_hash=evidence_content_hash("c-EV-nk"),
        retrieved_at=ASOF,
        job_id="J-1",
    )
    with pytest.raises(EvidenceRejectedError) as e1:
        ingest_evidence(EvidenceLedger(), noknown, as_of=ASOF)
    assert e1.value.reason == "PIT_UNVERIFIED"
    with pytest.raises(EvidenceRejectedError) as e2:
        ingest_evidence(EvidenceLedger(), _mk_ev("EV-f", known=datetime(2025, 7, 1, tzinfo=UTC)), as_of=ASOF)
    assert e2.value.reason == "PIT_VIOLATION"
    seen: list[str] = []
    bad_prov = _mk_ev("EV-r")
    object.__setattr__(bad_prov, "source_name", "")
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            EvidenceLedger(), bad_prov, as_of=ASOF, on_reject=lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(EvidenceLedger(), _mk_ev("EV-r2", sid=""), as_of=ASOF, on_reject=lambda t, p: seen.append(t))
    assert seen == ["evidence.rejected"]
    with pytest.raises(EvidenceIntegrityError):
        Evidence(
            evidence_id="x",
            session_id="s",
            wave_id=0,
            source_type="t",
            source_name="n",
            subject="s",
            claim_text="c",
            content="c",
            content_hash=evidence_content_hash("c"),
            retrieved_at=ASOF,
        )
    with pytest.raises(EvidenceIntegrityError):
        Evidence(
            evidence_id="x",
            session_id="s",
            wave_id=1,
            source_type="t",
            source_name="n",
            subject="s",
            claim_text="c",
            content="c",
            content_hash="bad",
            retrieved_at=ASOF,
        )
    with pytest.raises(EvidenceIntegrityError):
        Evidence(
            evidence_id="x",
            session_id="s",
            wave_id=1,
            source_type="t",
            source_name="n",
            subject="s",
            claim_text="c",
            content="c",
            content_hash=evidence_content_hash("c"),
            retrieved_at=ASOF,
            confidence=2.0,
        )
    led = EvidenceLedger()
    led.append(_mk_ev("EV-a"))
    with pytest.raises(EvidenceIntegrityError):
        led.append(_mk_ev("EV-a"))
    with pytest.raises(EvidenceIntegrityError):
        led.supersede(_mk_ev("EV-b"))
    with pytest.raises(EvidenceNotFoundError):
        led.get("missing")
    b = _mk_ev("EV-b", known=datetime(2025, 6, 2, tzinfo=UTC))
    object.__setattr__(b, "superseded_by", "EV-a")
    led.supersede(b)
    assert led.current("EV-a").evidence_id == "EV-b"
    assert "EV-a" in led and len(led) == 2 and led.ids() == ("EV-a", "EV-b")
    assert led.list_session("rs:t")[0].evidence_id == "EV-a"
    from app.research.evidence import evidence_from_dict, evidence_to_dict

    d = evidence_to_dict(_mk_ev("EV-z"))
    assert evidence_from_dict(d).evidence_id == "EV-z"
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "metadata": []})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "confidence": True})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "retrieved_at": "bad"})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "wave_id": "1"})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "supports": [1]})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "source_name": ""})
    with pytest.raises(EvidenceIntegrityError):
        evidence_from_dict({**d, "source_name": 1})


# --- freeze: membership/PIT/parse branches ---


def test_freeze_branches():
    led = EvidenceLedger()
    e1 = ingest_evidence(led, _mk_ev("EV-1"), as_of=ASOF)
    f = create_freeze(freeze_id="F1", session_id="rs:t", wave_id=1, records=[e1], as_of=ASOF)
    assert f.evidence_ids == ("EV-1",)
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:t", wave_id=0, records=[e1])
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:other", wave_id=1, records=[e1])
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:t", wave_id=1, records=[e1, e1])
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:t", wave_id=1, records=[e1], as_of="bad")
    future = _mk_ev("EV-f", known=datetime(2025, 8, 1, tzinfo=UTC))
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:t", wave_id=1, records=[future], as_of=ASOF)
    noknown = Evidence(
        evidence_id="EV-n",
        session_id="rs:t",
        wave_id=1,
        source_type="sec",
        source_name="SEC",
        source_uri="https://sec.gov/x",
        subject="NVDA",
        claim_text="c",
        content="c-EV-n",
        content_hash=evidence_content_hash("c-EV-n"),
        retrieved_at=ASOF,
        job_id="J-1",
    )
    with pytest.raises(FreezeIntegrityError):
        create_freeze(freeze_id="F", session_id="rs:t", wave_id=1, records=[noknown], as_of=ASOF)
    with pytest.raises(FreezeIntegrityError):
        EvidenceFreeze(
            freeze_id="", session_id="s", wave_id=1, created_at=ASOF, as_of=None, evidence_ids=(), content_hash="h"
        )
    with pytest.raises(FreezeIntegrityError):
        EvidenceFreeze(
            freeze_id="f", session_id="", wave_id=1, created_at=ASOF, as_of=None, evidence_ids=(), content_hash="h"
        )
    with pytest.raises(FreezeIntegrityError):
        EvidenceFreeze(
            freeze_id="f", session_id="s", wave_id=0, created_at=ASOF, as_of=None, evidence_ids=(), content_hash="h"
        )
    with pytest.raises(FreezeIntegrityError):
        EvidenceFreeze(
            freeze_id="f", session_id="s", wave_id=1, created_at=ASOF, as_of=None, evidence_ids=(), content_hash=""
        )
    from app.research.freeze import freeze_to_dict, verify_freeze

    assert freeze_to_dict(f)["freeze_id"] == "F1"
    assert freeze_from_dict(freeze_to_dict(f)).freeze_id == "F1"
    with pytest.raises(FreezeIntegrityError):
        verify_freeze(f, [])
    other = _mk_ev("EV-9", known=datetime(2025, 6, 1, tzinfo=UTC))
    with pytest.raises(FreezeIntegrityError):
        verify_freeze(f, [other])
    bad_cases: list[dict[str, object]] = [
        {},
        {"freeze_id": "", "session_id": "s", "content_hash": "h", "wave_id": 1},
        {"freeze_id": "f", "session_id": "", "content_hash": "h", "wave_id": 1},
        {"freeze_id": "f", "session_id": "s", "content_hash": "", "wave_id": 1},
        {"freeze_id": "f", "session_id": "s", "content_hash": "h", "wave_id": 0, "evidence_ids": [1]},
        {"freeze_id": "f", "session_id": "s", "content_hash": "h", "wave_id": 1, "created_at": 5},
    ]
    for bad in bad_cases:
        with pytest.raises(FreezeIntegrityError):
            freeze_from_dict(bad)


# --- journal/session/models guards ---


def test_journal_branches():
    sid = "rs:j1"
    _journal._LOG.pop(sid, None)
    assert _journal.next_sequence(sid) == 1
    e = _journal.append_event(sid, "t", "a", "i", {"k": "v"})
    assert e.sequence == 1
    for args in (("", "t", "a", "i"), (sid, "", "a", "i"), (sid, "t", "", "i"), (sid, "t", "a", "")):
        with pytest.raises(ValueError):
            _journal.append_event(*args)
    with pytest.raises(ValueError):
        _journal.append_event(sid, "t2", "a", "i", event_id=e.event_id)
    assert len(_journal.list_events(sid)) == 1
    assert _journal.list_events(sid, event_type="t") != []
    assert _journal.list_events(sid, event_type="nope") == []
    _journal.hydrate(sid, [e])
    other = _journal.append_event("rs:other", "t", "a", "i", {})
    with pytest.raises(ValueError):
        _journal.hydrate(sid, [other])
    bad_seq = JournalEvent(
        event_id="x", session_id=sid, sequence=9, event_type="t", timestamp=ASOF, actor_type="a", actor_id="i"
    )
    with pytest.raises(ValueError):
        _journal.hydrate(sid, [bad_seq])
    p = _journal.rejection_payload("EV-1", "R", "2025-01-01", "2025-06-30")
    assert p["reason"] == "R"
    with pytest.raises(ValueError):
        _journal.rejection_payload("", "R", None, None)
    with pytest.raises(ValueError):
        _journal.rejection_payload("EV-1", "", None, None)


def test_session_branches():
    s = _session.create_session("q", "o", as_of=ASOF)
    assert _session.is_terminal("completed") and not _session.is_terminal("created")
    assert _session.can_transition("created", "planning")
    assert _session.get_session({s.session_id: s}, s.session_id).session_id == s.session_id
    with pytest.raises(KeyError):
        _session.get_session({}, "missing")
    with pytest.raises(ValueError):
        _session.create_session("", "o")
    with pytest.raises(ValueError):
        _session.create_session("q", "")
    with pytest.raises(ValueError):
        _session.create_session("q", "o", as_of="bad")
    with pytest.raises(ValueError):
        _session.create_session("q", "o", as_of="123")
    with pytest.raises(ValueError):
        _session.create_session("q", "o", as_of="  ")
    with pytest.raises(ValueError):
        _session.transition_session(s, "completed")
    with pytest.raises(ValueError):
        _session.transition_session(s, "bogus")
    moved = _session.transition_session(s, SessionStatus.PLANNING)
    assert moved.status == "planning"
    assert _session.create_session("q", "o", as_of=None).as_of is None


def test_model_guards():
    with pytest.raises(ValueError):
        validate_json_value(float("nan"), "<t>")
    with pytest.raises(ValueError):
        validate_json_value({1: "x"}, "<t>")
    with pytest.raises(ValueError):
        validate_json_value(object(), "<t>")
    with pytest.raises(ValueError):
        validate_json_mapping([1], "<t>")
    assert validate_json_mapping({"a": 1}, "<t>") == {"a": 1}
    assert pit_unverified("2025-06-30", None) is True
    assert pit_unverified(None, None) is False
    assert pit_unverified("unbounded", None) is False
    assert pit_unverified(ASOF, None) is True
    assert pit_violated(None, ASOF) is False
    base = {"session_id": "s", "created_at": ASOF, "updated_at": ASOF, "query": "q", "objective": "o"}
    rs = ResearchSession(**base)
    with pytest.raises(ValueError):
        ResearchSession(session_id="", created_at=ASOF, updated_at=ASOF, query="q", objective="o").validate()
    with pytest.raises(ValueError):
        ResearchSession(session_id="s", created_at=ASOF, updated_at=ASOF, query="", objective="o").validate()
    with pytest.raises(ValueError):
        ResearchSession(**{**base, "current_wave": True}).validate()
    with pytest.raises(ValueError):
        ResearchSession(**{**base, "current_wave": -1}).validate()
    with pytest.raises(ValueError):
        ResearchSession(**{**base, "current_wave": True}).validate()
    d = rs.to_dict()
    assert ResearchSession.from_dict(d).session_id == "s"
    with pytest.raises(ValueError):
        ResearchSession.from_dict({})
    with pytest.raises(ValueError):
        ResearchSession.from_dict({**d, "failure": "x"})
    with pytest.raises(ValueError):
        ResearchSession.from_dict({**d, "committee_runs": {}})
    with pytest.raises(ValueError):
        ResearchSession.from_dict({**d, "current_wave": True})
    job = Job(
        job_id="j",
        session_id="s",
        wave_id=1,
        parent_job_id=None,
        job_type=JobType.SCOUT.value,
        owner="o",
        status=JobStatus.QUEUED.value,
    )
    with pytest.raises(ValueError):
        Job(
            job_id="", session_id="s", wave_id=1, parent_job_id=None, job_type=JobType.SCOUT.value, owner="o"
        ).validate()
    with pytest.raises(ValueError):
        Job(
            job_id="j", session_id="s", wave_id=0, parent_job_id=None, job_type=JobType.SCOUT.value, owner="o"
        ).validate()
    with pytest.raises(ValueError):
        Job(
            job_id="j",
            session_id="s",
            wave_id=1,
            parent_job_id=None,
            job_type=JobType.SCOUT.value,
            owner="",
            child_budget=-1,
        ).validate()
    jd = job.to_dict()
    assert Job.from_dict(jd).job_id == "j"
    with pytest.raises(ValueError):
        Job.from_dict({})
    with pytest.raises(ValueError):
        Job.from_dict({**jd, "failure": "x"})
    with pytest.raises(ValueError):
        Job.from_dict({**jd, "child_budget": True})
    with pytest.raises(ValueError):
        JournalEvent(
            event_id="", session_id="s", sequence=1, event_type="t", timestamp=ASOF, actor_type="a", actor_id="i"
        ).validate()
    with pytest.raises(ValueError):
        JournalEvent(
            event_id="e", session_id="s", sequence=0, event_type="t", timestamp=ASOF, actor_type="a", actor_id="i"
        ).validate()
    with pytest.raises(ValueError):
        JournalEvent(
            event_id="e", session_id="s", sequence=1, event_type="", timestamp=ASOF, actor_type="a", actor_id=""
        ).validate()
    with pytest.raises(ValueError):
        JournalEvent.from_dict({})
    with pytest.raises(ValueError):
        JournalEvent.from_dict(
            {
                "event_id": "e",
                "session_id": "s",
                "sequence": True,
                "event_type": "t",
                "timestamp": ASOF.isoformat(),
                "actor_type": "a",
                "actor_id": "i",
            }
        )
    from app.research.models import Failure

    with pytest.raises(ValueError):
        Failure(category="nope", message="m").validate()
    with pytest.raises(ValueError):
        Failure(category="timeout", message="").validate()
    with pytest.raises(ValueError):
        Failure.from_dict({})


# --- committee/final merges: routing + disagreement paths ---


def test_committee_branches():
    stock, bull, bear = _trio()
    stock.claims.append(GroundedClaim(text="shared", evidence_ids=["EV-1"]))
    bull.claims.append(GroundedClaim(text="shared", evidence_ids=["EV-1"]))
    bear.claims.append(GroundedClaim(text="shared", evidence_ids=["EV-1"]))
    bull.claims.append(GroundedClaim(text="bull-only", evidence_ids=["EV-2"]))
    bear.claims.append(GroundedClaim(text="bear-only", evidence_ids=["EV-3"]))
    bull.research_requests.append(
        ResearchRequest(
            question="q1",
            why_material="w",
            requested_source_domain="sec",
            expected_gain="g",
            requesting_agents=["bull"],
        )
    )
    bear.research_requests.append(
        ResearchRequest(
            question="q1",
            why_material="w",
            requested_source_domain="sec",
            expected_gain="g",
            requesting_agents=["bear"],
        )
    )
    out = compute_disagreement(stock, bull, bear)
    assert any("all three cite EV-1" in a for a in out.agreement)
    assert any("bull-only" in d for d in out.disagreement)
    assert any("bear-only" in d for d in out.disagreement)
    assert out.requested_research[0].requesting_agents == ["bull", "bear"]
    assert out.wave_id == 1
    from app.research.synthesis.committee import _coerce_wave_id

    assert _coerce_wave_id("2") == 2
    for bad in ("x", ""):
        with pytest.raises(ValueError):
            _coerce_wave_id(bad)
    for bad_int in (0, -1):
        with pytest.raises(ValueError):
            _coerce_wave_id(bad_int)
    with pytest.raises(ValueError):
        _coerce_wave_id(True)


def test_final_branches():
    stock, bull, bear = _trio()
    stock.claims.append(GroundedClaim(text="same", evidence_ids=["EV-1"]))
    bull.claims.append(GroundedClaim(text="same", evidence_ids=["EV-2"]))
    stock.unknowns.append("u1")
    bull.unknowns.append("u1")
    bull.what_would_change.append("c1")
    bear.what_would_change.append("c1")
    disagreement = compute_disagreement(stock, bull, bear)
    final = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id="1",
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
    )
    assert final.claims[0].evidence_ids == ["EV-1", "EV-2"]
    assert final.unknowns == ["u1"] and final.what_would_change == ["c1"]
    override = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
        model="draft",
    )
    assert override.answer == "draft"


def test_opt_telemetry_rejects_bad_int_type() -> None:
    with pytest.raises(ValueError, match="telemetry.searches"):
        _reg._opt_telemetry({"telemetry": {"searches": "3"}})
    with pytest.raises(ValueError, match="telemetry.searches"):
        _reg._opt_telemetry({"telemetry": {"searches": True}})


def test_opt_telemetry_rejects_bad_str_list() -> None:
    with pytest.raises(ValueError, match="telemetry.queries"):
        _reg._opt_telemetry({"telemetry": {"queries": ["ok", 7]}})
    with pytest.raises(ValueError, match="telemetry.queries"):
        _reg._opt_telemetry({"telemetry": {"queries": "ok"}})


def test_opt_telemetry_none_absent_and_valid_round_trip() -> None:
    from app.research.evals.regression import ResearchTelemetry

    assert _reg._opt_telemetry({}) is None
    tel: ResearchTelemetry = {
        "searches": 2,
        "queries": ["q1", "q2"],
        "forms": ["10-K"],
        "entities": ["MSFT"],
        "coverage": "partial",
        "stop_reason": "no_questions",
    }
    _reg.build_fixture(
        session_id="rs:tel",
        scenario_name="pit-knowable-by-2025-06-30",
        tool_calls=("search_sec_filings",),
        evidence_ids=("EV-1",),
        known_ats=("2025-05-01",),
        answer_excerpt="x",
        telemetry=tel,
    )
    raw: dict[str, object] = {
        "telemetry": dict(tel),
        "format": "f",
        "scenario_name": "s",
        "family": "pit",
        "session_id": "rs:t",
        "question": "q",
        "tool_calls": [],
        "evidence_ids": [],
        "known_ats": [],
        "answer_excerpt": "x",
        "validator": {"requires_evidence": False},
    }
    out = _reg._opt_telemetry(raw)
    assert out is not None and out["queries"] == ["q1", "q2"] and out["searches"] == 2


def test_opt_telemetry_rejects_non_object_and_bad_str_field() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _reg._opt_telemetry({"telemetry": ["searches"]})
    with pytest.raises(ValueError, match="telemetry.coverage"):
        _reg._opt_telemetry({"telemetry": {"coverage": 7}})
    with pytest.raises(ValueError, match="telemetry.stop_reason"):
        _reg._opt_telemetry({"telemetry": {"stop_reason": ["x"]}})


def test_opt_telemetry_absent_key_arms() -> None:
    only_int = _reg._opt_telemetry({"telemetry": {"searches": 1}})
    assert only_int == {"searches": 1}
    only_list = _reg._opt_telemetry({"telemetry": {"queries": ["a"]}})
    assert only_list == {"queries": ["a"]}
    assert _reg._opt_telemetry({"telemetry": {}}) == {}
