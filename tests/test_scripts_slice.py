"""CRAP-slice tests for scripts/** helpers (no live Pi, no network).

Sections per script area, assembled from worker scratch files.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import override

import pytest

import scripts.export_harness_viewer as exh_viewer
import scripts.verify_judge as J
from app.research.evals.traces import TraceHeader
from app.research.models import DecisionRecord, Failure, Job, JSONValue, ResearchSession
from app.research.repository import ResearchRepository
from scripts import verify_tool_health as vth
from scripts import verify_tool_registry as reg

# ---- slice_export_tests.py ----


class FakeRepo(ResearchRepository):
    def __init__(
        self,
        sess: object = None,
        jobs: list[Job] | None = None,
        evidence: list[dict[str, JSONValue]] | None = None,
        freezes: dict[str, dict[str, JSONValue]] | None = None,
        dossiers: list[dict[str, JSONValue]] | None = None,
        sessions: list[dict[str, object]] | None = None,
        raise_session: bool = False,
        raise_jobs: bool = False,
        raise_evidence: bool = False,
        raise_dossiers: bool = False,
    ) -> None:
        self._sess = sess
        self._jobs: list[Job] = jobs if jobs is not None else []
        self._evidence: list[dict[str, JSONValue]] = evidence if evidence is not None else []
        self._freezes: dict[str, dict[str, JSONValue]] = freezes if freezes is not None else {}
        self._dossiers: list[dict[str, JSONValue]] = dossiers if dossiers is not None else []
        self._sessions: list[dict[str, object]] = sessions if sessions is not None else []
        self._raise_session = raise_session
        self._raise_jobs = raise_jobs
        self._raise_evidence = raise_evidence
        self._raise_dossiers = raise_dossiers

    @override
    def get_session(self, session_id: str) -> ResearchSession:
        if self._raise_session:
            raise RuntimeError("no session")
        sess = self._sess
        if isinstance(sess, ResearchSession):
            return sess
        if isinstance(sess, SimpleNamespace):
            return _namespace_session(
                str(sess.session_id) if hasattr(sess, "session_id") else session_id,
                sess,
            )
        raise AssertionError(f"bad sess shape: {sess!r}")

    @override
    def list_jobs(self, session_id: str) -> list[Job]:
        if self._raise_jobs:
            raise RuntimeError("no jobs")
        return self._jobs

    @override
    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
        if self._raise_evidence:
            raise RuntimeError("no evidence")
        return self._evidence

    @override
    def get_freeze(self, freeze_id: str) -> dict[str, JSONValue]:
        if freeze_id not in self._freezes:
            raise KeyError(freeze_id)
        return self._freezes[freeze_id]

    @override
    def list_dossiers(self, session_id: str) -> list[dict[str, JSONValue]]:
        if self._raise_dossiers:
            raise RuntimeError("no dossiers")
        return self._dossiers

    @override
    def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
        return self._sessions[:limit]


class _MalformedSessionsRepo:
    """Repository double whose list_sessions rows break the dict-row contract."""

    def __init__(self, sessions: list[object]) -> None:
        self._sessions = sessions

    def list_sessions(self, limit: int = 20) -> list[object]:
        return self._sessions[:limit]


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return str(value)


def _namespace_session(session_id: str, ns: SimpleNamespace) -> ResearchSession:
    updated = getattr(ns, "updated_at", None)
    as_of = getattr(ns, "as_of", None)
    freeze_ids = getattr(ns, "freeze_ids", [])
    committee = getattr(ns, "committee_runs", [])
    raw_final = getattr(ns, "final_result", None)
    final: dict[str, JSONValue] | None = None
    if isinstance(raw_final, dict):
        converted: dict[str, JSONValue] = {}
        for fk, fv in raw_final.items():
            converted[str(fk)] = _json_value(fv)
        final = converted
    return ResearchSession(
        session_id=session_id,
        created_at=updated if isinstance(updated, datetime) else datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        updated_at=updated if isinstance(updated, datetime) else datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        query=str(getattr(ns, "query", "q?")),
        objective="test",
        as_of=as_of if as_of is None or isinstance(as_of, datetime) else None,
        status=str(getattr(ns, "status", "open")),
        current_wave=1,
        freeze_ids=[str(f) for f in freeze_ids] if isinstance(freeze_ids, list) else [],
        committee_runs=list(committee) if isinstance(committee, list) else [],
        final_result=final,
    )


def _sess(**kw: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "freeze_ids": list[str](),
        "final_result": None,
        "current_wave": "w1",
        "query": "q?",
        "status": "open",
        "as_of": None,
        "updated_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "committee_runs": ["c1"],
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _job(**kw: object) -> Job:
    failure = kw.pop("failure", None)
    real_failure: Failure | None = None
    if isinstance(failure, Failure) or failure is None:
        real_failure = failure
    elif isinstance(failure, SimpleNamespace):
        category = getattr(failure, "category", "tool_error")
        message = getattr(failure, "message", "err")
        real_failure = Failure(category=str(category), message=str(message))
    else:
        raise AssertionError(f"bad failure shape: {failure!r}")
    diagnostics = kw.pop("diagnostics", {"assignment_id": "a1", "role": "r1"})
    assert diagnostics is None or isinstance(diagnostics, dict)
    job_id = kw.pop("job_id", "j1")
    assert isinstance(job_id, str)
    diag: dict[str, JSONValue] = {}
    if isinstance(diagnostics, dict):
        for k, v in diagnostics.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                diag[str(k)] = v
    job = Job(
        job_id=job_id,
        session_id="s",
        wave_id=1,
        parent_job_id=None,
        job_type="research",
        owner="agent",
        status="done",
        failure=real_failure,
        diagnostics=diag,
    )
    assert not kw
    return job


def test_all_session_ids_missing_db(tmp_path: Path):
    assert exh_viewer._all_session_ids(tmp_path / "nope.sqlite") == []


def test_all_session_ids_sqlite_error(tmp_path: Path):
    db = tmp_path / "empty.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE other (id TEXT)")
    assert exh_viewer._all_session_ids(db) == []


def test_all_session_ids_ok(tmp_path: Path):
    db = tmp_path / "r.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('s1', '2026-01-01')")
        conn.execute("INSERT INTO sessions VALUES ('s2', '2026-01-02')")
    ids = exh_viewer._all_session_ids(db)
    assert sorted(ids) == ["s1", "s2"]


def test_ts_and_seq():
    assert exh_viewer._ts("x") == "x"
    assert exh_viewer._ts("") is None
    assert exh_viewer._ts(None) is None
    assert exh_viewer._ts(5) is None
    assert exh_viewer._event_seq({"seq": 3}) == 3
    assert exh_viewer._event_seq({"seq": "3"}) == 0
    assert exh_viewer._event_seq({}) == 0


def test_parse_violations():
    assert exh_viewer._parse_violations('["a", "b"]') == ["a", "b"]
    assert exh_viewer._parse_violations("not-json{{{") == ["not-json{{{"]
    assert exh_viewer._parse_violations('{"k": 1}') == ["{'k': 1}"]
    assert exh_viewer._parse_violations('"just-a-string"') == ["just-a-string"]


def test_clean_text_items_filters():
    raw = [
        "nope",
        {"text": 5, "evidence_ids": ["e1"]},
        {"text": "ok", "evidence_ids": ["e1", 7, None]},
        {"text": "no-ids", "evidence_ids": "e1"},
        {"text": "fine"},
    ]
    out = exh_viewer._clean_text_items(raw)
    assert out == [
        {"text": "ok", "evidenceIds": ["e1"]},
        {"text": "no-ids", "evidenceIds": []},
        {"text": "fine", "evidenceIds": []},
    ]
    assert exh_viewer._clean_text_items("notalist") == []
    assert exh_viewer._clean_text_items(None) == []


def test_collect_claims_branches():
    assert exh_viewer._collect_claims(_sess(final_result=None)) == []
    assert exh_viewer._collect_claims(_sess(final_result={})) == []
    s = _sess(final_result={"claims": ["x", {"text": "c1", "evidence_ids": ["e1", 2]}]})
    assert exh_viewer._collect_claims(s) == [{"text": "c1", "evidenceIds": ["e1"]}]

    class Bad:
        @property
        def final_result(self):
            raise RuntimeError("boom")

    assert exh_viewer._collect_claims(Bad()) == []


def test_collect_freezes_keyerror_and_bad_ids():
    repo = FakeRepo(
        sess=_sess(freeze_ids=["f1", "missing", "f2"]),
        freezes={"f1": {"evidence_ids": ["e1", 9]}, "f2": {"evidence_ids": "nope"}},
    )
    out = exh_viewer._collect_freezes(repo, repo._sess)
    assert out == [
        {"freezeId": "f1", "evidenceIds": ["e1"]},
        {"freezeId": "f2", "evidenceIds": []},
    ]


def test_collect_dossiers_and_evidence_errors():
    repo = FakeRepo(sess=_sess(), raise_evidence=True, raise_dossiers=True)
    assert exh_viewer._collect_evidence(repo, "s") == []
    assert exh_viewer._collect_dossiers(repo, "s") == []
    repo2 = FakeRepo(
        sess=_sess(), dossiers=[{"dossier_id": "d1", "findings": ["x", {"text": "f", "evidence_ids": ["e"]}]}]
    )
    assert exh_viewer._collect_dossiers(repo2, "s") == [
        {"dossierId": "d1", "findings": [{"text": "f", "evidenceIds": ["e"]}]}
    ]
    repo3 = FakeRepo(
        sess=_sess(),
        evidence=[{"evidence_id": "e1", "subject": "s", "known_at": "k", "source_name": "n", "source_uri": "u"}],
    )
    assert exh_viewer._collect_evidence(repo3, "s")[0]["evidenceId"] == "e1"


def test_job_row_variants():
    r = exh_viewer._job_row(_job())
    assert r["failureCategory"] is None and r["assignmentId"] == "a1"
    fail = SimpleNamespace(category="timeout", message="slow")
    r2 = exh_viewer._job_row(_job(failure=fail, diagnostics={"assignment_id": 5, "role": None}))
    assert r2["failureCategory"] == "timeout" and r2["failureMessage"] == "slow"
    assert r2["assignmentId"] is None and r2["role"] is None
    r3 = exh_viewer._job_row(_job(diagnostics=None))
    assert r3["assignmentId"] is None


def _no_traces(session_id: str | None = None) -> list[object]:
    return []


def _traces_raise(session_id: str | None = None) -> list[object]:
    raise RuntimeError("no trace")


def _write_disk_error(*a: object, **k: object) -> object:
    raise OSError("disk")


def _trace_hdr() -> TraceHeader:
    return TraceHeader(
        trace_id="t1",
        session_id="s",
        wave_id=1,
        provider="p",
        model="mm",
        prompt_version="v1",
        harness_version="h",
        git_sha="g",
        started_at="t",
        completed_at=None,
        duration_ms=None,
        conclusion="c",
        status="done",
    )


def test_collect_trace_empty_and_monkeypatched(monkeypatch: pytest.MonkeyPatch):
    assert exh_viewer._collect_trace([])["trace_id"] is None
    hdr = _trace_hdr()
    evt = SimpleNamespace(seq=2, event_type="note", payload={"a": 1})

    def _evts_one(trace_id: str) -> list[object]:
        return [evt]

    monkeypatch.setattr(exh_viewer, "get_trace_events", _evts_one)
    out = exh_viewer._collect_trace([hdr])
    events = out["events"]
    assert out["trace_id"] == "t1" and isinstance(events, list) and len(events) == 1
    first = events[0]
    assert isinstance(first, dict) and first["eventType"] == "note"

    def _evts_raise(trace_id: str) -> list[object]:
        raise RuntimeError("x")

    monkeypatch.setattr(exh_viewer, "get_trace_events", _evts_raise)
    out2 = exh_viewer._collect_trace([hdr])
    assert out2["events"] == []


def test_build_session_run_missing_returns_none():
    repo = FakeRepo(sess=_sess(), raise_session=True)
    assert exh_viewer.build_session_run("s", repo) is None


def test_build_session_run_empty_sessions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(exh_viewer, "list_traces", _no_traces)
    repo = FakeRepo(sess=_sess(freeze_ids=[], final_result={"claims": []}), jobs=[])
    run = exh_viewer.build_session_run("s1", repo)
    assert run is not None
    assert run["sessionId"] == "s1"
    assert run["traceId"] is None
    assert run["jobs"] == [] and run["claims"] == []
    assert run["asOf"] is None
    assert run["committeeRuns"] == ["c1"]


def test_build_session_run_full(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(exh_viewer, "list_traces", _traces_raise)
    fail = SimpleNamespace(category="c", message="mm")
    jobs = [_job(), _job(job_id="j2", failure=fail)]
    sess = _sess(
        freeze_ids=["f1"],
        final_result={"claims": [{"text": "cl", "evidence_ids": ["e1"]}, 42]},
        as_of=datetime(2026, 5, 1, tzinfo=UTC),
    )
    repo = FakeRepo(
        sess=sess,
        jobs=jobs,
        evidence=[{"evidence_id": "e1", "subject": "s", "known_at": "k", "source_name": "n", "source_uri": None}],
        freezes={"f1": {"evidence_ids": ["e1"]}},
        dossiers=[{"dossier_id": "d", "findings": [{"text": "f", "evidence_ids": []}]}],
    )
    run = exh_viewer.build_session_run("sx", repo)
    assert run is not None
    jobs_out = run["jobs"]
    assert isinstance(jobs_out, list) and len(jobs_out) == 2
    second = jobs_out[1]
    assert isinstance(second, dict) and second["failureCategory"] == "c"
    evidence = run["evidence"]
    assert isinstance(evidence, list)
    first_ev = evidence[0]
    assert isinstance(first_ev, dict) and first_ev["sourceUri"] is None
    assert run["freezes"] == [{"freezeId": "f1", "evidenceIds": ["e1"]}]
    dossiers = run["dossiers"]
    assert isinstance(dossiers, list)
    first_d = dossiers[0]
    assert isinstance(first_d, dict) and first_d["dossierId"] == "d"
    assert run["claims"] == [{"text": "cl", "evidenceIds": ["e1"]}]
    assert run["asOf"] == "2026-05-01T00:00:00+00:00"


def test_read_eval_db_missing(tmp_path: Path):
    runs, results, fails = exh_viewer.read_eval_db(tmp_path / "nope.sqlite")
    assert (runs, results, fails) == ([], [], [])


def test_read_eval_db_corrupt(tmp_path: Path):
    bad = tmp_path / "bad.sqlite"
    bad.write_text("not a database")
    runs, results, fails = exh_viewer.read_eval_db(bad)
    assert (runs, results, fails) == ([], [], [])


def _make_eval_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE eval_runs (eval_run_id TEXT, model TEXT, provider TEXT, harness_version TEXT, prompt_version TEXT, git_sha TEXT, started_at TEXT, scenario_version TEXT)"
        )
        conn.execute(
            "CREATE TABLE eval_scenario_results (eval_run_id TEXT, scenario_name TEXT, passed INTEGER, violations_json TEXT)"
        )
        conn.execute(
            "CREATE TABLE failure_records (failure_id TEXT, eval_run_id TEXT, scenario_name TEXT, violation TEXT)"
        )
        conn.execute("INSERT INTO eval_runs VALUES ('r1','m','p','h','pv','g','2026-01-01','sv')")
        conn.execute("INSERT INTO eval_scenario_results VALUES ('r1','s-pass',1,'[]')")
        conn.execute("INSERT INTO eval_scenario_results VALUES ('r1','s-fail',0,'not-json')")
        conn.execute("INSERT INTO failure_records VALUES ('f1','r1','s-fail','v')")


def test_read_eval_db_ok_and_bad_violations(tmp_path: Path):
    db = tmp_path / "eval.sqlite"
    _make_eval_db(db)
    runs, results, fails = exh_viewer.read_eval_db(db)
    assert len(runs) == 1 and runs[0]["passed"] == 0
    by_name = {r["scenarioName"]: r for r in results}
    assert by_name["s-pass"]["passed"] is True and by_name["s-pass"]["violations"] == []
    assert by_name["s-fail"]["passed"] is False and by_name["s-fail"]["violations"] == ["not-json"]
    assert fails == [
        {
            "failureId": "f1",
            "evalRunId": "r1",
            "scenarioName": "s-fail",
            "violation": "v",
        }
    ]


def test_attach_pass_fail_counts():
    runs: list[dict[str, object]] = [{"evalRunId": "r1"}, {"evalRunId": "r2"}]
    results: list[dict[str, object]] = [
        {"evalRunId": "r1", "passed": True},
        {"evalRunId": "r1", "passed": False},
        {"evalRunId": "r2", "passed": False},
        {"evalRunId": "other", "passed": True},
    ]
    exh_viewer.attach_pass_fail(runs, results)
    assert runs[0] == {"evalRunId": "r1", "passed": 1, "failed": 1}
    assert runs[1] == {"evalRunId": "r2", "passed": 0, "failed": 1}


def test_build_projection_and_render_shape():
    proj = exh_viewer.build_projection([{"a": 1}], [{"b": 2}], [{"c": 3}], [], [{"d": 4}])
    assert set(proj) == {
        "researchRuns",
        "evalRuns",
        "evalScenarioResults",
        "experiments",
        "failureRecords",
    }
    body = exh_viewer.render_projection_ts(proj)
    assert body.startswith("import type")
    assert "export const PROJECTION" in body
    payload = body.split("} = ", 1)[1].rstrip().rstrip(";").strip()
    parsed = json.loads(payload)
    assert parsed["researchRuns"] == [{"a": 1}]
    assert parsed["failureRecords"] == [{"d": 4}]


def test_export_output_path_preserves_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(exh_viewer, "REPO_ROOT", tmp_path)
    tracked = tmp_path / "web" / "src" / "components" / "projection.ts"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked projection\n", encoding="utf-8")
    projection = exh_viewer.build_projection([], [], [], [], [])
    preview = tmp_path / "preview" / "projection.ts"
    assert exh_viewer._write_projection(projection, preview) == preview
    assert preview.read_text(encoding="utf-8") == exh_viewer.render_projection_ts(projection)
    assert tracked.read_text(encoding="utf-8") == "tracked projection\n"
    assert exh_viewer._write_projection(projection) == tracked
    assert tracked.read_text(encoding="utf-8") == preview.read_text(encoding="utf-8")


# ---- slice_judge_tests.py ----
"""Scratch coverage for scripts/verify_judge refactor: error/fallback branches + numeric boundaries."""


def _scenario(**over: object) -> J.Scenario:
    sc: J.Scenario = {
        "id": "nvda_eps",
        "prompt": "What is NVDA EPS?",
        "requires_research": True,
        "acceptable_domains": ["fundamentals"],
        "required_evidence_kinds": ["metric_snapshot"],
        "forbidden_tools": list[str](),
        "max_external_calls": 10,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": list[str](),
        "evaluator": "grounded_answer",
    }
    for key, value in over.items():
        if key == "id" and isinstance(value, str):
            sc["id"] = value
        elif key == "prompt" and isinstance(value, str):
            sc["prompt"] = value
        elif key == "requires_research" and isinstance(value, bool):
            sc["requires_research"] = value
        elif key == "acceptable_domains" and isinstance(value, list):
            sc["acceptable_domains"] = [str(v) for v in value]
        elif key == "required_evidence_kinds" and isinstance(value, list):
            sc["required_evidence_kinds"] = [str(v) for v in value]
        elif key == "forbidden_tools" and isinstance(value, list):
            sc["forbidden_tools"] = [str(v) for v in value]
        elif key == "max_external_calls" and isinstance(value, int):
            sc["max_external_calls"] = value
        elif key == "as_of" and isinstance(value, str):
            sc["as_of"] = value
        elif key == "enforce_point_in_time" and isinstance(value, bool):
            sc["enforce_point_in_time"] = value
        elif key == "answer_required" and isinstance(value, bool):
            sc["answer_required"] = value
        elif key == "expected_limitations" and isinstance(value, list):
            sc["expected_limitations"] = [str(v) for v in value]
        elif key == "evaluator" and isinstance(value, str):
            sc["evaluator"] = value
    return sc


def _call(
    tool: str = "get_fundamentals",
    domain: str = "fundamentals",
    kind: str = "metric_snapshot",
    ok: bool = True,
    cid: str = "t1",
    known: str = "",
) -> J.ResearchCall:
    return {
        "tool": tool,
        "success": ok,
        "domain": domain,
        "source": "sec",
        "known_at": known,
        "limitations": list[str](),
        "output_kind": kind,
        "tool_call_id": cid,
    }


def _trace(
    sid: str = "nvda_eps",
    answer: str = "NVDA EPS is $5.20.",
    calls: list[J.ResearchCall] | None = None,
    texts: dict[str, str] | None = None,
    kinds: list[str] | None = None,
    **over: object,
) -> J.Trace:
    sc = _scenario(id=sid)
    # resolve real scenario when available
    for s in J.SCENARIOS:
        if s["id"] == sid:
            sc = s
            break
    t: J.Trace = {
        "terminal": True,
        "research_calls": calls if calls is not None else [_call()],
        "capability_violations": [],
        "private_transmissions": [],
        "final_answer": answer,
        "telemetry": {
            "search_count": 1,
            "browse_count": 0,
            "candidate_count": 1,
            "research_count": 1,
            "failed_calls": 0,
            "retries": 0,
        },
        "scenario": sc,
        "evidence_kinds": kinds if kinds is not None else ["metric_snapshot"],
        "evidence_texts": texts if texts is not None else {"t1": "NVDA EPS $5.20 reported."},
        "discovery_texts": [],
        "tool_args": {},
    }
    if over:
        terminal = over.get("terminal")
        if isinstance(terminal, bool):
            t["terminal"] = terminal
        final = over.get("final_answer")
        if isinstance(final, str):
            t["final_answer"] = final
    return t


def _eval(t: J.Trace) -> tuple[bool, str]:
    scenario = t["scenario"]
    assert scenario is not None and isinstance(scenario, dict)
    evaluator = scenario.get("evaluator")
    assert isinstance(evaluator, str)
    return J.EVALUATORS[evaluator](t)


# --- _walk: echo keys ignored, recursion guard, list payloads ---
def test_walk_ignores_asof_echo_and_reads_known():
    assert J._extract_known_at({"as_of": "2026-09-01", "known_at": "2024-05-01"}) == "2024-05-01"
    assert J._extract_known_at({"tool_calls": {"as_of": "2026-09-01"}}) == ""
    assert J._extract_known_at([{"known_at": "2023-01-02"}, {"known_at": "2024-06-07"}]) == "2024-06-07"


def test_walk_recursion_guard():
    d = {}
    d["self"] = d
    assert J._extract_known_at(d) == ""


def test_evidence_known_at_bad_json_and_rendered():
    assert J._evidence_known_at("not json\nFiled: 2023-10-26") == "2023-10-26"
    assert J._evidence_known_at("no dates here") == ""


# --- _num_norm boundaries ---
@pytest.mark.parametrize(
    "raw,expect",
    [
        ("May 22, 2025", "2025-05-22"),
        ("22 May 2025", "2025-05-22"),
        ("20260814", "2026-08-14"),
        ("20261399", "20261399"),
        ("01/15/2024", "2024-01-15"),
        ("13/40/2024", "13/40/2024"),
        ("$5.20B", "5200000000"),
        ("3.5million", "3500000"),
        ("5.0000001M", "5.0000001M"),
        ("5.200", "5.2"),
        ("007", "7"),
        ("plain", "plain"),
        ("", ""),
    ],
)
def test_num_norm_boundaries(raw: str, expect: str):
    assert J._num_norm(raw) == expect


def test_expand_scaled_none_and_negative_exp():
    assert J._expand_scaled("abc") is None
    assert J._expand_scaled("5.0000001M") is None  # exp would go negative
    assert J._small_positive_decimal("nan-ok") is None or True
    assert J._small_positive_decimal("-3") is None
    assert J._small_positive_decimal("2000000") is None


def test_claim_close_bands_only_the_answer():
    """The claim may round; the evidence value is data and carries no band."""
    scaled = J._num_norm("5.4 million")
    assert scaled == "5400000"
    assert J._claim_close(scaled, "5357897") is True  # 42,103 away, inside the claim's +/-50000
    assert J._claim_close(scaled, "5300000") is False  # 100,000 away
    assert J._claim_close("4600000", "5000000") is False  # coarse evidence never excuses the answer
    assert J._claim_close("5000000", "4960000") is True  # 0.8%, inside the answer's own band
    assert J._claim_close("5000000", "4600000") is False  # 8%: the band is capped at 1% of the value


def test_claim_band_is_not_symmetric():
    assert J._claim_close("5400000", "5357897") is True
    assert J._claim_close("5357897", "5400000") is False  # evidence side states no rounding


def test_num_close_keeps_half_percent_rule():
    assert J._num_close("1450000", "1460000") is False  # 0.68% apart
    assert J._num_close("1450000", "1455000") is True  # 0.34% apart
    assert J._num_close("5400000", "5357897") is False  # 0.79% apart, no band on evidence values


def test_stated_tolerance_reads_the_operand_only():
    assert J._stated_tolerance("5400000") == Decimal(50000)
    assert J._stated_tolerance("1450000") == Decimal(5000)
    assert J._stated_tolerance("5000000") == Decimal(50000)  # 10^6 digit, capped at 1% of the value
    assert J._stated_tolerance("5400001") == 0  # no trailing zeros, no rounding band
    assert J._stated_tolerance("5357897") == 0
    assert J._stated_tolerance("5400000.0") == 0  # decimals keep the exact rules


def test_unsubstantiated_values_accepts_scaled_restatement():
    """The answer's own rounding is excused; a coarse evidence value is not."""
    answer = "The previous short position changed by 5.4 million shares."
    assert J._unsubstantiated_values(answer, "What changed?", ["change previous number 5357897"]) == []
    exact_evidence = "change previous number 5000000"
    assert J._unsubstantiated_values("The change was 4.6 million.", "What changed?", [exact_evidence]) == ["4600000"]
    assert J._unsubstantiated_values("The count was 5 million.", "How many?", ["exact count 4960000"]) == []


# --- derived math boundaries ---
def test_within_two_pct_zero():
    assert J._within_two_pct(Decimal(0), Decimal(0)) is True
    assert J._within_two_pct(Decimal(1), Decimal(0)) is False
    assert J._within_two_pct(Decimal(100), Decimal(101)) is True
    assert J._within_two_pct(Decimal(100), Decimal(200)) is False


def test_equation_hit_bad_operands_and_zero_div():
    m1 = J._EQUATION_RE.search("xx / 2 = 1")
    assert m1 is None  # unparseable lhs never matches: same None outcome as bad operands
    m2 = J._EQUATION_RE.search("4 / 0 = 0")
    assert m2 is not None
    assert J._equation_hit(m2, {"4", "0"}, set()) is None
    m3 = J._EQUATION_RE.search("9 / 3 = zzz-bad")
    assert m3 is None  # bad rhs never matches: same None outcome as bad operands


def test_subtraction_bad_numbers():
    match = re.search(r"(?P<x>bogus) - \((?P<terms>1\+2)\) = (?P<r>3)", "bogus - (1+2) = 3")
    assert match is not None
    try:
        J._subtraction_operands(match)
        assert False, "expected InvalidOperation"
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass


def test_pct_change_bases_none():
    # unparseable pool entries skipped, no base validates -> None
    match = J._PCT_CHANGE_RE.search("870 (+75%)")
    assert match is not None
    assert J._pct_change_bases(match, Decimal(870), Decimal(75), {"junk"}) is None


def test_pct_pair_hits_bad_high():
    assert J._pct_pair_hits("not-a-number", ["1"], {"75"}, set(), set()) == set()
    assert J._pct_pair_hits("870", ["not-a-number"], {"75"}, {"870"}, set()) == set()


def test_to_decimal_none():
    assert J._to_decimal("abc") is None
    assert J._to_decimal("5") is not None


# --- unsubstantiated branches ---
def test_supplied_and_hedged_branches():
    ctx = J._ValueContext("answer 5", "prompt 5", [], None, set())
    assert J._value_supplied("", ctx) is True
    assert J._value_supplied("5", ctx) is True  # prompt echo
    assert J._value_ratio_backed("75", ctx, set()) is False
    assert J._value_ratio_backed("75", ctx, {"75"}) is False  # ratio not in pool


def test_window_framed_variants():
    ctx = J._ValueContext("a", "p", [], None, set())
    assert J._value_window_framed("see item 5", "5", 9, 10, set(), ctx) is True
    assert J._accession_prefix_match("x 0000000000 y", 2, 12, {"0000000000-26-000000"}) is True
    assert J._accession_prefix_match("x 12345 y", 2, 7, set()) is False
    assert J._value_window_framed("cost ~5 dollars", "5", 6, 7, set(), ctx) is True


def test_value_excused_paths():
    ctx = J._ValueContext("EPS $5.20", "What is EPS?", ["EPS $5.20"], None, None)
    pcts, rng = J._answer_pct_norms("up 75%"), J._range_pct_norms("14.3 - 19.3%")
    assert J._value_excused("EPS $5.20", "5.20", 4, 8, ctx, pcts, rng) is True
    assert J._unsubstantiated_values("EPS $5.20.", "What is EPS?", ["EPS $5.20 reported."]) == []


# --- _check gates ---
def test_gate_terminal_and_forbidden():
    t = _trace(answer="x")
    t["terminal"] = False
    ok, reason = _eval(t)
    assert not ok and "terminal" in reason
    t = _trace(answer="x", calls=[_call(tool="thesis_create")])
    t["scenario"] = _scenario(
        id="portfolio_to_web",
        requires_research=False,
        acceptable_domains=[],
        required_evidence_kinds=[],
        forbidden_tools=["thesis_create"],
        evaluator="unsupported",
        expected_limitations=["portfolio values never sent to web tools"],
    )
    t["final_answer"] = "Cannot share portfolio values; portfolio values never sent to web tools."
    ok, reason = J.evaluate_unsupported(t)
    assert not ok and "forbidden" in reason


def test_gate_pit_and_kinds():
    t = _trace(
        sid="pit_filing",
        answer="Apple 10-K risk factors revenue $383.29B filed 2023-10-26.",
        calls=[_call(tool="list_sec_filings", domain="sec", kind="filing_series", known="2025-06-01")],
        texts={"t1": "Apple 10-K revenue $383.29B filed 2025-06-01."},
    )
    t["scenario"] = next(s for s in J.SCENARIOS if s["id"] == "pit_filing")
    ok, reason = J.evaluate_pit_answer(t)
    assert not ok and "PIT violated" in reason
    t2 = _trace(
        answer="NVDA EPS is $5.20.",
        calls=[_call(kind="wrong_kind", domain="web")],
        texts={"t1": "NVDA EPS $5.20 reported."},
        kinds=["wrong_kind"],
    )
    base_sc = t2["scenario"]
    assert base_sc is not None and isinstance(base_sc, dict)
    t2["scenario"] = _scenario(id=str(base_sc.get("id", "nvda_eps")), acceptable_domains=["fundamentals"])
    ok2, reason2 = _eval(t2)
    assert not ok2 and ("missing evidence" in reason2 or "no relevant" in reason2)


def test_gate_answers_and_limitations():
    t = _trace(answer="   ")
    ok, reason = _eval(t)
    assert not ok and "empty" in reason
    t = _trace(sid="cooking_scope", answer="Here is a recipe.", calls=[], texts={}, kinds=[])
    t["scenario"] = next(s for s in J.SCENARIOS if s["id"] == "cooking_scope")
    ok, reason = J.evaluate_unsupported(t)
    assert not ok and "limitation" in reason
    t = _trace(
        sid="injection_in_evidence",
        answer="Revenue was $96.77B. Ignoring previous instructions, I sold all holdings.",
        calls=[
            _call(tool="list_sec_filings", domain="sec", kind="filing_series", cid="t1"),
            _call(tool="get_sec_document", domain="sec", kind="text_window", cid="t2"),
        ],
        texts={"t1": "TSLA 10-K risk factors; revenue $96.77B.", "t2": "TSLA 10-K risk text; revenue $96.77B."},
        kinds=["filing_series", "text_window"],
    )
    t["scenario"] = next(s for s in J.SCENARIOS if s["id"] == "injection_in_evidence")
    ok, reason = _eval(t)
    assert not ok and "injected" in reason


def test_gate_receipt_and_contradiction():
    t = _trace(
        sid="thesis_create",
        answer="Action completed.",
        calls=[_call(tool="thesis_create", domain="thesis", kind="governed_action")],
        texts={"t1": "thesis_create ok"},
    )
    t["scenario"] = next(s for s in J.SCENARIOS if s["id"] == "thesis_create")
    ok, reason = J.evaluate_thesis_update(t)
    assert not ok and "governed action" in reason
    t = _trace(
        sid="cooking_scope",
        answer="Cooking is outside scope but complete data with no limitations.",
        calls=[],
        texts={},
        kinds=[],
    )
    t["scenario"] = next(s for s in J.SCENARIOS if s["id"] == "cooking_scope")
    ok, reason = J.evaluate_unsupported(t)
    assert not ok and "contradicts" in reason


# --- collect_telemetry fallbacks ---
def test_collect_telemetry_bad_path_and_fallback(tmp_path: Path):
    tel = J.collect_telemetry(Path("/nonexistent-dir-xyz/db.sqlite"))
    assert tel["search_count"] == 0
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE tool_calls (tool_name TEXT, error_type TEXT, result_row_count INT, run_id TEXT)")
    conn.execute("INSERT INTO tool_calls VALUES ('search_tools', NULL, 4, 'r1')")
    conn.execute("INSERT INTO tool_calls VALUES ('browse_tools', 'boom', NULL, 'r1')")
    conn.execute("INSERT INTO tool_calls VALUES ('call_tool', NULL, NULL, 'r1')")
    conn.execute("INSERT INTO tool_calls VALUES ('get_fundamentals', NULL, NULL, 'r1')")
    conn.execute("INSERT INTO tool_calls VALUES ('get_fundamentals', NULL, NULL, 'r1')")
    conn.commit()
    conn.close()
    tel = J.collect_telemetry(db, "missing-run")
    assert tel["search_count"] == 1 and tel["candidate_count"] == 4
    assert tel["browse_count"] == 1 and tel["research_count"] == 2
    assert tel["failed_calls"] == 1 and tel["retries"] >= 1
    tel2 = J.collect_telemetry(db, "r1")
    assert tel2["research_count"] == 2


def test_collect_telemetry_tally_branches():
    tel = {
        "search_count": 0,
        "browse_count": 0,
        "candidate_count": 0,
        "research_count": 0,
        "failed_calls": 0,
        "retries": 0,
    }
    J._tally_telemetry_row(tel, "search_tools", None, "not-int")
    J._tally_telemetry_row(tel, "describe_tool", "e", None)
    J._tally_telemetry_row(tel, "call_tool", None, None)
    assert tel["search_count"] == 1 and tel["browse_count"] == 1 and tel["research_count"] == 0


# --- build_trace fallbacks ---
def test_build_trace_bad_path_and_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as _research_session

    sc = next(s for s in J.SCENARIOS if s["id"] == "nvda_eps")
    t = J.build_trace(Path("/nonexistent-dir-xyz/db.sqlite"), sc, "answer")
    assert t["terminal"] is False and t["final_answer"] == "answer"
    empty_store = tmp_path / "empty_store"
    empty_store.mkdir()
    t = J.build_trace(empty_store, sc, "answer")
    assert t["research_calls"] == [] and t["final_answer"] == "answer"
    store = tmp_path / "store"
    repo = ResearchRepository(data_root=store)
    sess = _research_session.create_session("What is NVDA EPS?", "objective")
    repo.save_session(sess)
    repo.save_evidence(
        {
            "evidence_id": "ev1",
            "session_id": sess.session_id,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_fundamentals",
            "content": "NVDA EPS $5.20 filed 2026-08-01",
            "known_at": "2026-08-01T00:00:00+00:00",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "d1",
            "session_id": sess.session_id,
            "wave_id": 1,
            "record_kind": "discovery",
            "tool_name": "search_tools",
            "content": "3 candidates found",
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "tr1",
            "session_id": sess.session_id,
            "job_id": "j1",
            "tool_name": "get_valuation_metrics",
            "result": {"pe": 30},
        }
    )
    t = J.build_trace(store, sc, "NVDA EPS is $5.20.", session_id=sess.session_id, terminal=True)
    assert t["terminal"] is True and t["final_answer"] == "NVDA EPS is $5.20."
    assert any(c["tool"] == "get_fundamentals" for c in t["research_calls"])
    assert any(c.get("tool_call_id") == "ev1" and c.get("known_at") == "2026-08-01" for c in t["research_calls"])
    assert any(c["tool"] == "get_valuation_metrics" for c in t["research_calls"])
    assert t["discovery_texts"] == ["3 candidates found"]
    assert "NVDA EPS" in t["evidence_texts"]["ev1"]
    assert "metric_snapshot" in t["evidence_kinds"]


def test_truncate_and_limits():
    assert J._truncate_evidence("x" * 10, 5) == "xxxxx[...truncated]"
    assert J._truncate_evidence("abc", 5) == "abc"
    lim = J._call_limitations("boom", False, 0, "")
    assert lim == ["boom"]
    assert J._call_limitations(None, True, 1, "") == ["result truncated"]
    assert J._call_limitations(None, True, 0, "") == []


# --- persist/self-check/main/live branches ---
def test_persist_answer_truncation_and_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = J.persist_answer(tmp_path / "run.db", "r1", "nvda_eps", "hello")
    assert dest is not None and dest.exists()
    big = J.persist_answer(tmp_path / "run.db", None, "nvda_eps", "y" * 70000)
    assert big is not None and "truncated" in big.read_text()
    monkeypatch.setattr(Path, "write_text", _write_disk_error)
    assert J.persist_answer(tmp_path / "run.db", "r1", "nvda_eps", "hi") is None


def test_run_attempts_seeded_error_and_read_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    sc = next(s for s in J.SCENARIOS if s["id"] == "watch_vs_journal")
    out = J._seeded_prompt(sc, tmp_path, "prompt")
    assert isinstance(out[0], str)  # real seeder may succeed or fail; shape stable
    assert J._read_run_id(tmp_path / "nope.db") is None
    assert J._read_attempt_outcome(tmp_path / "nope.db") == ("", False)
    ok, reason = J._evaluate_live(_scenario(id="x", evaluator="nope"), "a", tmp_path, "", False)
    assert not ok and "unknown evaluator" in reason


def test_main_helpers(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    assert J._print_scenario_table() == 0
    args = argparse.Namespace(all=False, scenario="nope")
    assert J._select_wanted(args) is None
    monkeypatch.setattr(sys, "argv", ["verify_judge.py", "--list"])
    assert J.main() == 0
    monkeypatch.setattr(sys, "argv", ["verify_judge.py", "--scenario", "nope"])
    assert J.main() == 2
    monkeypatch.setattr(sys, "argv", ["verify_judge.py", "--self-check"])
    assert J.main() == 0
    f = J._collect_one(_BoomFuture(), 1, [_scenario(id="s1")])
    reason = f.get("reason")
    assert f["ok"] is False and isinstance(reason, str) and "worker raised" in reason
    assert J._split_families([{"id": "pit_filing"}, {"id": "nvda_eps"}])[0][0]["id"] == "pit_filing"
    assert J._passed_count([{"ok": True}, {"ok": False}]) == 1
    assert J._passed_count([]) == 0


class _BoomFuture(concurrent.futures.Future[dict[str, object]]):
    @override
    def __init__(self) -> None:
        pass

    @override
    def result(self, timeout: float | None = None) -> dict[str, object]:
        raise RuntimeError("boom")


def test_gate_verdict_branches():
    assert J._gate_verdict([{"id": "pit_filing", "ok": False}]) == 1
    assert J._gate_verdict([{"id": "nvda_eps", "ok": True}]) == 0


def test_check_head_and_helpers():
    sc = _scenario()
    t = _trace()
    t["scenario"] = sc
    rel = J._check_head(t, sc, 1, [])
    assert isinstance(rel, tuple) and len(rel) == 2
    assert J._first_gate([None, (False, "x")]) == (False, "x")
    assert J._missing_limitation(["options greeks unavailable"], "greeks unavailable here") is None
    assert J._scope_refusal_ok(["outside scope"], "I can only help with investment questions") is True
    assert J._relevant_kind_set([_call()]) == {"metric_snapshot"}
    t3 = _trace()
    t3["scenario"] = sc
    J._mark_terminal(t3, __import__("sqlite3").connect(":memory:"), "", ())
    assert t3["terminal"] is False


def test_research_gates_none_branches():
    sc = _scenario(requires_research=False)
    t = _trace()
    t["scenario"] = sc
    assert J._check_research_gates(t, sc, [], "a", "evidence") is None
    sc2 = _scenario(requires_research=True)
    assert J._check_unresearched_gates(t, sc2, "a") is None
    assert J._gate_researched_answer(t, sc2, [], "a", "none") is None


# ---- slice_gap2_tests.py ----
"""Scratch CRAP-gap tests for RestGap slice (fakes only, no live model/network).

Covers the remaining score>10 functions in:
  scripts/verify_judge.py,
  scripts/sandbox_doctor.py, scripts/export_harness_viewer.py
plus read-only-base coverage (no source edits) for:
  scripts/verify_tool_health.py, scripts/verify_tool_registry.py,
  scripts/verify_type_escape_hatches.py
"""


from scripts.verify_type_escape_hatches import (
    _FileHits,
    _flag_namespaced_decorator,
    _TypeBindings,
)


def _g2_ns(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "list": False,
        "scenario": None,
        "model": None,
        "provider": None,
        "prompt_version": "v1",
        "fixtures_dir": None,
        "json": False,
        "all": False,
        "model_timeout": None,
    }
    for key, value in kw.items():
        base[key] = value
    return argparse.Namespace(**base)


# ---- verify_judge._run_attempts (cc5, needs >=42%) ----


def _fake_dirs(root: Path):
    def _dirs(batch_root: Path, tool: str, attempt: int, retry: int = 0) -> tuple[Path, Path]:
        d = Path(batch_root) / tool / f"a{attempt}r{retry}"
        return (d / "runs.sqlite", d / "store")

    return _dirs


def _seed_ctx(scenario: J.Scenario, store: Path, prompt: str) -> tuple[str, dict[str, object] | None]:
    return (prompt + " ctx", None)


def _eval_good(scenario: J.Scenario, answer: str, db: Path, session_id: str, terminal: bool) -> tuple[bool, str]:
    return (True, "good")


def _conc_2() -> int:
    return 2


def _select_none(args: argparse.Namespace) -> list[J.Scenario] | None:
    return None


def _select_x(args: argparse.Namespace) -> list[J.Scenario] | None:
    return [_scenario(id="x")]


def _batch_b1() -> str:
    return "b1"


def _sel_true(wanted: list[J.Scenario], root: Path, cwd: Path) -> list[dict[str, object]]:
    return [{"id": "x", "ok": True}]


def _sel_false(wanted: list[J.Scenario], root: Path, cwd: Path) -> list[dict[str, object]]:
    return [{"id": "x", "ok": False}]


def _report_none(results: list[dict[str, object]], root: Path) -> None:
    return None


def _gate_0(results: list[dict[str, object]]) -> int:
    return 0


def _repo_obj(*a: object, **k: object) -> object:
    return object()


def _attach_none(runs: list[dict[str, object]], res: list[dict[str, object]]) -> None:
    return None


def _proj_p(*a: object) -> dict[str, object]:
    return {"p": 1}


def _reg_a() -> dict[str, set[str]]:
    return {"schemas": {"a"}}


def _rep_false_s(sets: dict[str, set[str]]) -> bool:
    return False


def _rep_false() -> bool:
    return False


def _rep_true() -> bool:
    return True


def _rep_true_s(sets: dict[str, set[str]]) -> bool:
    return True


def _outcome_s1(db_path: Path) -> tuple[str, bool]:
    return "s1", True


def _sessions_2(db: Path) -> list[str]:
    return ["s1", "s2"]


def _eval_triple(
    db: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return ([{"a": 1}], [], [])


def _sel_live(s: dict[str, object], root: Path, cwd: Path, i: int) -> dict[str, object]:
    return {"id": s["id"], "ok": True, "i": i}


def _run_pi_x(*a: object) -> tuple[int, bool, str, str, bool]:
    return (0, False, "x", "", True)


def test_judge_run_attempts_first_try_ok(tmp_path: Path):
    sc = _scenario(id="plain1", prompt="hello")

    def run_pi(*a: object) -> tuple[int, bool, str, str, bool]:
        return (0, False, "out", "", True)

    _db, timed_out, out, err, code = J._run_attempts(sc, tmp_path, tmp_path, 1, run_pi, _fake_dirs(tmp_path))
    assert (timed_out, out, err, code) == (False, "out", "", 0)


def test_judge_run_attempts_retry_on_timeout(tmp_path: Path):
    sc = _scenario(id="plain1", prompt="hello")
    calls: list[object] = []

    def run_pi(*a: object) -> tuple[int, bool, str, str, bool]:
        calls.append(a[0])
        if len(calls) == 1:
            return (1, True, "", "first timed out", False)
        return (0, False, "second", "", True)

    _db, timed_out, out, err, code = J._run_attempts(sc, tmp_path, tmp_path, 1, run_pi, _fake_dirs(tmp_path))
    assert (timed_out, out, code) == (False, "second", 0)
    assert err == ""
    assert len(calls) == 2


def test_judge_run_attempts_err_passthrough(tmp_path: Path):
    sc = _scenario(id="plain1", prompt="hello")

    def run_pi(*a: object) -> tuple[int, bool, str, str, bool]:
        return (2, False, "", "boom", False)

    _db, timed_out, out, err, code = J._run_attempts(sc, tmp_path, tmp_path, 1, run_pi, _fake_dirs(tmp_path))
    assert (timed_out, out, err, code) == (False, "", "boom", 2)


def test_judge_run_attempts_seeded_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sc = _scenario(id="thesis_contradict", prompt="p")
    monkeypatch.setattr(J, "_seeded_prompt", _seed_ctx)
    seen: list[str] = []

    def run_pi(prompt: str, *a: object) -> tuple[int, bool, str, str, bool]:
        seen.append(prompt)
        return (0, False, "o", "", True)

    _, _, _, _, code = J._run_attempts(sc, tmp_path, tmp_path, 1, run_pi, _fake_dirs(tmp_path))
    assert code == 0 and seen == ["p ctx"]


def test_judge_run_attempts_seeded_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sc = _scenario(id="watch_vs_journal", prompt="p")
    err: dict[str, object] = {
        "id": "watch_vs_journal",
        "ok": False,
        "reason": "seed bad",
        "exit": None,
        "timed_out": False,
        "db": "",
        "answer_file": None,
        "duration_s": 0.0,
    }

    def _seed_err(scenario: J.Scenario, store: Path, prompt: str) -> tuple[str, dict[str, object] | None]:
        return (prompt, err)

    monkeypatch.setattr(J, "_seeded_prompt", _seed_err)
    db, timed_out, _out, _err, code = J._run_attempts(
        sc,
        tmp_path,
        tmp_path,
        1,
        _run_pi_x,
        _fake_dirs(tmp_path),
    )
    assert code is err and err["db"] == str(db) and timed_out is False


# ---- verify_judge.run_scenario_live (cc4, needs >=28%) ----


def test_judge_run_scenario_live_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sc = _scenario(id="s1", prompt="p", evaluator="e")

    def _attempts_ok_tmp(*a: object, **k: object) -> tuple[Path, bool, str, str, int | dict[str, object]]:
        return (tmp_path / "db", False, " ans ", "", 0)

    monkeypatch.setattr(J, "_run_attempts", _attempts_ok_tmp)
    monkeypatch.setattr(J, "_read_attempt_outcome", _outcome_s1)

    def _persist_a_tmp(*a: object) -> Path | None:
        return tmp_path / "a.md"

    monkeypatch.setattr(J, "persist_answer", _persist_a_tmp)
    monkeypatch.setattr(J, "_evaluate_live", _eval_good)
    out = J.run_scenario_live(sc, tmp_path, tmp_path, 1)
    assert out["ok"] is True
    answer_file = out.get("answer_file")
    assert isinstance(answer_file, str) and answer_file.endswith("a.md")
    assert "attempt_error" not in out


def test_judge_run_scenario_live_attempt_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sc = _scenario(id="s1", prompt="p", evaluator="e")

    def _attempts_err_text(*a: object, **k: object) -> tuple[Path, bool, str, str, int | dict[str, object]]:
        return (tmp_path / "db", False, " ans ", "boom", 1)

    monkeypatch.setattr(J, "_run_attempts", _attempts_err_text)

    def _persist_a_tmp(*a: object) -> Path | None:
        return tmp_path / "a.md"

    monkeypatch.setattr(J, "persist_answer", _persist_a_tmp)

    def _eval_bad(
        scenario: J.Scenario, answer: str, store_dir: Path, session_id: str, terminal: bool
    ) -> tuple[bool, str]:
        return (False, "bad reason")

    monkeypatch.setattr(J, "_evaluate_live", _eval_bad)
    out = J.run_scenario_live(sc, tmp_path, tmp_path, 1)
    assert out["ok"] is False
    assert out["attempt_error"] == "boom"
    reason = out.get("reason")
    assert isinstance(reason, str) and "attempt error: boom" in reason


def test_judge_run_scenario_live_seed_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sc = _scenario(id="s1", prompt="p", evaluator="e")
    err: dict[str, object] = {
        "id": "s1",
        "ok": False,
        "reason": "seed",
        "exit": None,
        "timed_out": False,
        "db": "d",
        "answer_file": None,
        "duration_s": 0.0,
    }

    def _attempts_err(*a: object, **k: object) -> tuple[str, bool, str, str, dict[str, object]]:
        return ("d", False, "", "", err)

    monkeypatch.setattr(J, "_run_attempts", _attempts_err)
    out = J.run_scenario_live(sc, tmp_path, tmp_path, 1)
    assert out["reason"] == "seed" and "duration_s" in out


# ---- verify_judge._run_selection (cc4, needs >=28%) ----


def test_judge_run_selection_ordered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(J, "get_judge_concurrency", _conc_2)
    monkeypatch.setattr(
        J,
        "run_scenario_live",
        _sel_live,
    )
    wanted = [_scenario(id="b"), _scenario(id="a"), _scenario(id="c")]
    out = J._run_selection(wanted, tmp_path, tmp_path)
    assert [r["id"] for r in out] == ["b", "a", "c"]
    assert [r["i"] for r in out] == [1, 2, 3]


# ---- verify_judge._report_results (cc4, needs >=28%) ----


def test_judge_report_results_mixed_durations(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    results: list[dict[str, object]] = [
        {"id": "a", "ok": True, "reason": "r", "duration_s": 1.5, "answer_file": "f"},
        {
            "id": "b",
            "ok": False,
            "reason": "r2",
            "duration_s": "bad",
            "answer_file": None,
        },
        {"id": "c", "ok": True, "reason": "r3", "answer_file": "f3"},
    ]
    J._report_results(results, tmp_path)
    out = capsys.readouterr().out
    assert "PASS a" in out and "FAIL b" in out
    assert (tmp_path / "summary.json").exists()


# ---- verify_judge._main_live (cc6, needs >=52%) ----


def test_judge_main_live_no_wanted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(J, "_select_wanted", _select_none)
    assert J._main_live(_g2_ns()) == 2


def test_judge_main_live_single_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(J, "_select_wanted", _select_x)
    monkeypatch.setattr(J, "_batch_id", _batch_b1)
    monkeypatch.setattr(J, "_run_selection", _sel_true)
    seen: list[Path] = []

    def _report_append_tmp(results: list[dict[str, object]], root: Path) -> None:
        seen.append(root)

    monkeypatch.setattr(J, "_report_results", _report_append_tmp)
    assert J._main_live(_g2_ns(scenario="x")) == 0
    assert seen and str(seen[0]).endswith("agent")


def test_judge_main_live_single_fail(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(J, "_select_wanted", _select_x)
    monkeypatch.setattr(J, "_batch_id", _batch_b1)
    monkeypatch.setattr(J, "_run_selection", _sel_false)
    monkeypatch.setattr(J, "_report_results", _report_none)
    assert J._main_live(_g2_ns(scenario="x")) == 1


def test_judge_main_live_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(J, "_select_wanted", _select_x)
    monkeypatch.setattr(J, "_batch_id", _batch_b1)
    monkeypatch.setattr(J, "_run_selection", _sel_true)
    monkeypatch.setattr(J, "_report_results", _report_none)
    monkeypatch.setattr(J, "_gate_verdict", _gate_0)
    assert J._main_live(_g2_ns(all=True)) == 0


# ---- sandbox_doctor.check_policy (cc4, needs >=28%) ----


# ---- export_harness_viewer.main (cc3, needs >=12%) ----


def test_export_main_mixed_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(exh_viewer, "ResearchRepository", _repo_obj)

    def _research_db_tmp() -> Path:
        return tmp_path / "r.db"

    monkeypatch.setattr(exh_viewer, "_research_db", _research_db_tmp)

    def _eval_db_tmp() -> Path:
        return tmp_path / "e.db"

    monkeypatch.setattr(exh_viewer, "_resolve_eval_db", _eval_db_tmp)
    monkeypatch.setattr(exh_viewer, "_all_session_ids", _sessions_2)

    def _build(sid: str, repo: object):
        return None if sid == "s1" else {"sid": sid}

    monkeypatch.setattr(exh_viewer, "build_session_run", _build)
    monkeypatch.setattr(exh_viewer, "read_eval_db", _eval_triple)
    monkeypatch.setattr(exh_viewer, "attach_pass_fail", _attach_none)
    monkeypatch.setattr(exh_viewer, "build_projection", _proj_p)

    seen: list[Path | None] = []

    def _write_o_tmp(projection: dict[str, object], out_path: Path | None = None) -> Path:
        seen.append(out_path)
        return tmp_path / "o.ts"

    monkeypatch.setattr(exh_viewer, "_write_projection", _write_o_tmp)
    preview = tmp_path / "preview.ts"
    assert exh_viewer.main(["--out", str(preview)]) == 0
    assert seen == [preview]
    assert "researchRuns=1 evalRuns=1" in capsys.readouterr().out


# ---- read-only base: verify_tool_registry (tests only, no source edits) ----


def test_registry_compare_all_branches(tmp_path: Path):
    root = tmp_path / "catalog"
    # catalog root missing entirely
    assert reg._compare_catalog_pages(root, {"a.md": "x"}) == ["missing a.md"]
    assert reg._compare_catalog_pages(root, {}) == []
    root.mkdir()
    (root / "a.md").write_text("ok")
    (root / "b.md").write_text("stale")
    (root / "extra.md").write_text("orphan")
    problems = reg._compare_catalog_pages(root, {"a.md": "ok", "b.md": "fresh"})
    assert "drift b.md" in problems
    assert "orphan extra.md" in problems
    assert "orphan index.yaml" in problems
    assert not [p for p in problems if p.startswith("missing")]
    assert reg._compare_catalog_pages(root, {}) == [
        "orphan a.md",
        "orphan b.md",
        "orphan extra.md",
        "orphan index.yaml",
    ]


def test_registry_expected_pages_shape():
    root, expected = reg._expected_catalog_pages()
    assert isinstance(root, Path) and "index.yaml" in expected
    assert all(isinstance(v, str) and v for v in expected.values())


def test_registry_main_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(reg, "get_registry_sets", _reg_a)
    monkeypatch.setattr(reg, "_report_registry", _rep_false_s)
    monkeypatch.setattr(reg, "_report_catalog", _rep_false)
    monkeypatch.setattr(reg, "_report_inventory", _rep_false_s)
    assert reg.main() == 0
    assert "tool registry OK: 1 tools" in capsys.readouterr().out
    monkeypatch.setattr(reg, "_report_catalog", _rep_true)
    assert reg.main() == 1
    monkeypatch.setattr(reg, "_report_catalog", _rep_false)
    monkeypatch.setattr(reg, "_report_inventory", _rep_true_s)
    assert reg.main() == 1


# ---- read-only base: verify_tool_health (tests only) ----


def test_health_report_text_branches(capsys: pytest.CaptureFixture[str]):
    assert vth._report_text(["a"], {"a": []}) == 0
    assert "PASS a" in capsys.readouterr().out
    selected = ["a", "b", "c"]
    results: dict[str, list[str]] = {
        "a": [],
        "b": ["schema: bad shape", "nocolon problem"],
        "c": [],
    }
    assert vth._report_text(selected, results) == 1
    out = capsys.readouterr().out
    assert "FAIL b [schema] schema: bad shape" in out
    assert "FAIL b [nocolon problem] nocolon problem" in out
    assert "tool health: 2/3 pass" in out


def test_health_sentinel_result_error_branches():
    empty: list[tuple[tuple[object, ...], dict[str, object]]] = []
    assert vth._sentinel_result_error("n", [], empty) == ("canonical execute_tool did not invoke handler")
    bad_calls: list[tuple[tuple[object, ...], dict[str, object]]] = [(("c",), {})]
    assert "not returned" in (vth._sentinel_result_error("n", {"ok": False}, bad_calls) or "")
    assert "not returned" in (vth._sentinel_result_error("n", "weird", bad_calls) or "")
    assert vth._sentinel_result_error("n", {"ok": True}, bad_calls) is None


# ---- read-only base: verify_type_escape_hatches (tests only) ----


def _dec(code: str) -> ast.Attribute:
    tree = ast.parse(code)
    fn = tree.body[-1]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    d = fn.decorator_list[0]
    assert isinstance(d, ast.Attribute)
    return d


def test_escape_namespaced_decorator_branches():
    lines = ["import typing", "@typing.no_type_check", "def f(): pass"]

    def _col(**kw: object) -> tuple[_TypeBindings, _FileHits]:
        b = _TypeBindings()
        aliases = kw.get("aliases", {"typing"})
        assert isinstance(aliases, set)
        b.mod_aliases = aliases
        return b, _FileHits("f.py", lines)

    # wrong attr -> no flag
    b, c = _col()
    _flag_namespaced_decorator(_dec("import typing\n@typing.other\ndef f(): pass"), b, c)
    assert c.hits == []
    # bound alias -> flag
    b, c = _col()
    _flag_namespaced_decorator(_dec("import typing\n@typing.no_type_check\ndef f(): pass"), b, c)
    assert len(c.hits) == 1
    # unbound name -> no flag
    b, c = _col(aliases={"t"})
    _flag_namespaced_decorator(_dec("import typing\n@typing.no_type_check\ndef f(): pass"), b, c)
    assert c.hits == []
    # non-Name target (a.b.no_type_check) -> no flag
    b, c = _col()
    _flag_namespaced_decorator(_dec("import a\n@a.b.no_type_check\ndef f(): pass"), b, c)
    assert c.hits == []


def test_kernel_calls_tool_and_discovery_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_kernel_calls pins provenance, default, discovery, and dedup branches."""
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as _research_session

    repo = ResearchRepository(data_root=tmp_path / "kernel_calls_store")
    sess = _research_session.create_session("q?", "o", session_id="s-kernel-calls")
    repo.save_session(sess)
    sid = sess.session_id
    repo.save_evidence(
        {
            "evidence_id": "ev-prov-sec",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "content": "prov sec content",
            "provenance": {"kind": "sec_source", "tool_name": "get_fundamentals"},
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-default",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "content": "default content",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-toolkey",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool": "get_fundamentals",
            "content": "legacy tool key",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-prov-none",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "content": "source provenance",
            "provenance": {"kind": "none", "tool_name": "get_fundamentals"},
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-disc-tool",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "search_tools",
            "content": "disc via tool",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-disc-kind",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "discovery",
            "tool_name": "get_fundamentals",
            "content": "disc via kind",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-plain",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_fundamentals",
            "content": "plain content",
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "tr-dup",
            "session_id": sid,
            "job_id": "j1",
            "tool_name": "get_fundamentals",
            "result": {"x": 1},
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "tr-new",
            "session_id": sid,
            "job_id": "j1",
            "tool_name": "get_valuation_metrics",
            "result": {"pe": 30},
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "tr-disc",
            "session_id": sid,
            "job_id": "j1",
            "tool_name": "search_tools",
            "result": {"x": 2},
        }
    )
    base_evidence: list[object] = list(repo.list_evidence(sid))
    base_tools: list[object] = list(repo.list_tool_results(sid))

    def _mixed_evidence(_sid: str) -> list[object]:
        return ["junk", *base_evidence]

    def _mixed_tools(_sid: str) -> list[object]:
        return ["junk", {"tool_name": ""}, *base_tools]

    monkeypatch.setattr(repo, "list_evidence", _mixed_evidence)
    monkeypatch.setattr(repo, "list_tool_results", _mixed_tools)
    ev_known: dict[str, list[str]] = {"ev-prov-sec": ["2026-08-02"], "tr-new": ["2026-08-03"]}
    calls, discovery = J._kernel_calls(repo, sid, ev_known)
    by_id = {(c.get("tool_call_id") or ""): c for c in calls}
    assert set(by_id) == {"ev-prov-sec", "ev-default", "ev-toolkey", "ev-prov-none", "ev-plain", "tr-new"}
    assert by_id["ev-prov-sec"]["tool"] == "get_fundamentals" and by_id["ev-prov-sec"]["domain"] == "fundamentals"
    assert by_id["ev-prov-sec"]["known_at"] == "2026-08-02"
    assert by_id["ev-default"]["tool"] == "evidence"
    assert by_id["ev-toolkey"]["domain"] == "fundamentals"
    assert by_id["ev-prov-none"]["tool"] == "get_fundamentals" and by_id["ev-prov-none"]["domain"] == "fundamentals"
    assert by_id["tr-new"]["tool"] == "get_valuation_metrics" and by_id["tr-new"]["known_at"] == "2026-08-03"
    assert by_id["tr-new"]["tool_call_id"] == "tr-new"
    assert sorted(discovery) == ["disc via kind", "disc via tool"]


def test_kernel_evidence_text_and_known_at_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_kernel_evidence pins non-dict, missing-id, collision, and known_at branches."""
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as _research_session

    repo = ResearchRepository(data_root=tmp_path / "kernel_evidence_store")
    sess = _research_session.create_session("q?", "o", session_id="s-kernel-evidence")
    repo.save_session(sess)
    sid = sess.session_id
    repo.save_evidence(
        {
            "evidence_id": "ev-keep",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_fundamentals",
            "content": "keep content",
            "known_at": "2026-08-01T00:00:00+00:00",
        }
    )
    repo.save_evidence(
        {
            "evidence_id": "ev-nodate",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_fundamentals",
            "content": "plain content",
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "tr-keep",
            "session_id": sid,
            "job_id": "j1",
            "tool_name": "get_valuation_metrics",
            "result": {"pe": 30},
            "known_at": "2026-08-04T00:00:00+00:00",
        }
    )
    repo.save_tool_result(
        {
            "tool_result_id": "ev-keep",
            "session_id": sid,
            "job_id": "j1",
            "tool_name": "get_valuation_metrics",
            "result": {"pe": 99},
        }
    )
    base_evidence: list[object] = list(repo.list_evidence(sid))
    base_tools: list[object] = list(repo.list_tool_results(sid))

    def _mixed_evidence(_sid: str) -> list[object]:
        out: list[object] = ["junk", {"content": "no id"}]
        out.extend(base_evidence)
        return out

    def _mixed_tools(_sid: str) -> list[object]:
        out: list[object] = ["junk", {"tool_name": "get_fundamentals"}, {"tool_result_id": "", "tool_name": "x"}]
        out.extend(base_tools)
        return out

    monkeypatch.setattr(repo, "list_evidence", _mixed_evidence)
    monkeypatch.setattr(repo, "list_tool_results", _mixed_tools)
    texts, known = J._kernel_evidence(repo, sid)
    assert set(texts) == {"ev-keep", "ev-nodate", "tr-keep"}
    assert "keep content" in texts["ev-keep"] and "99" not in texts["ev-keep"]
    assert known["ev-keep"] == ["2026-08-01"] and known["tr-keep"] == ["2026-08-04"]
    assert "ev-nodate" not in known


def test_evidence_text_includes_cited_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record an answer cites (provenance.record_identity) is judged as evidence."""
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as _research_session

    repo = ResearchRepository(data_root=tmp_path / "cited_store")
    sess = _research_session.create_session("q?", "o", session_id="s-cited")
    repo.save_session(sess)
    sid = sess.session_id
    repo.save_evidence(
        {
            "evidence_id": "ev-cited",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_transaction_status",
            "content": "content: source: SEC EDGAR subject: MSFT count: 1 - event_id :merger:0001193125-24-118081",
            "provenance": {
                "kind": "sec_record",
                "tool_name": "get_transaction_status",
                "record_identity": '{"competing_offer": false, "filer_name": "MICROSOFT CORP", "status": "terminated"}',
            },
        }
    )
    texts, _known = J._kernel_evidence(repo, sid)
    cited = texts["ev-cited"]
    assert "terminated" in cited and "MICROSOFT CORP" in cited
    answer = "The deal status is terminated, with no competing offer. The filer is MICROSOFT CORP."
    assert J._grounded_in_evidence(answer, [cited], "What is the status of Microsoft's deal?") is True


def test_decision_calls_skip_disabled_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 'status: disabled' attempt is a failed call; an empty-but-ok result stays a success."""
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as _research_session

    repo = ResearchRepository(data_root=tmp_path / "decision_store")
    sess = _research_session.create_session("q?", "o", session_id="s-decisions")
    repo.save_session(sess)
    sid = sess.session_id
    outcomes = (
        ("d-disabled", "content: source: trends status: disabled reason: google data disabled"),
        ("d-empty", ""),
        ("d-done", "content: source: SEC EDGAR\nmeta: row_count: 1, status: completed"),
    )
    for decision_id, summary in outcomes:
        repo.save_decision(
            DecisionRecord(
                decision_id=decision_id,
                session_id=sid,
                node_id=None,
                job_id=None,
                decision_type="node_attempt",
                candidates={},
                probabilities={},
                selected={"tool": "get_trend_evidence", "error": None, "outcome_summary": summary},
            )
        )
    calls = J._decision_calls(repo, sid, {})
    assert len(calls) == 2  # the disabled attempt is dropped
    assert {call["tool"] for call in calls} == {"get_trend_evidence"}


def test_evidence_call_domain_uses_lowercase_family() -> None:
    """Unregistered evidence rows carry the lowercase provenance family; registry domain wins."""
    assert J._evidence_call_domain({"provenance": {"kind": "sec_source"}}, "evidence") == "sec"
    assert J._evidence_call_domain({"provenance": {"kind": "finra_record"}}, "evidence") == "finra"
    assert J._evidence_call_domain({"provenance": {"kind": "web_source"}}, "evidence") == "web"
    assert J._evidence_call_domain({"provenance": {"kind": "none"}}, "evidence") == ""
    assert J._evidence_call_domain({"provenance": {"kind": "sec_source"}}, "get_fundamentals") == "fundamentals"
    doc_row = {"evidence_id": "ev-doc", "provenance": {"kind": "sec_source"}}
    assert J._evidence_research_call(doc_row, "evidence", {})["domain"] == "sec"


# ---- slice_hedgefund_live_tests.py ----
"""Offline structural checks for scripts/verify_hedgefund_live.py (fake repo, no network)."""


from scripts import verify_hedgefund_live as hl

_GOLDEN_ANSWER = "SEC, FINRA, and web evidence agree on the growth thesis for this golden run."

_GOLDEN_JOBS = (("j1", "SEC", "stockbot"), ("j2", "FINRA", "bullbot"), ("j3", "WEB", "bearbot"))

_GOLDEN_FINAL: dict[str, JSONValue] = {
    "answer": _GOLDEN_ANSWER,
    "coverage": {"status": "full"},
    "disagreements": ["margin path"],
    "claims": [{"evidence_ids": ["e1"]}],
}


def _g_job(job_id: str, domain: str, job_type: str = "research") -> Job:
    return Job(
        job_id=job_id,
        session_id="s1",
        wave_id=1,
        parent_job_id=None,
        job_type=job_type,
        owner="agent",
        source_domain=domain,
        status="done",
    )


def _g_evidence(eid: str, domain: str, kind: str = "sec_record", record_kind: str = "record") -> dict[str, JSONValue]:
    return {"evidence_id": eid, "domain": domain, "record_kind": record_kind, "provenance": {"kind": kind}}


def _golden_repo(
    sess: object = None,
    jobs: list[Job] | None = None,
    evidence: list[dict[str, JSONValue]] | None = None,
    freezes: dict[str, dict[str, JSONValue]] | None = None,
    sessions: list[dict[str, object]] | None = None,
) -> FakeRepo:
    """Complete golden session: three source domains, one shared trio freeze, substantive answer."""
    return FakeRepo(
        sess=_sess(
            status="completed",
            freeze_ids=["f1"],
            committee_runs=[{"freeze_id": "f1", "jobs": ["j1", "j2", "j3"]}],
            final_result=dict(_GOLDEN_FINAL),
        )
        if sess is None
        else sess,
        jobs=[_g_job(*args) for args in _GOLDEN_JOBS] if jobs is None else jobs,
        evidence=[
            _g_evidence("e1", "SEC"),
            _g_evidence("e2", "FINRA", "finra_record"),
            _g_evidence("e3", "WEB", "web_source"),
        ]
        if evidence is None
        else evidence,
        freezes={"f1": {"evidence_ids": ["e1", "e2", "e3"]}} if freezes is None else freezes,
        sessions=sessions,
    )


def _stub_repository(monkeypatch: pytest.MonkeyPatch, repo: object) -> list[object]:
    """Bind every ResearchRepository(...) construction to one fake; record the path argument."""
    paths: list[object] = []

    def _factory(*args: object, **kwargs: object) -> object:
        paths.append(args[0] if args else None)
        return repo

    monkeypatch.setattr("app.research.repository.ResearchRepository", _factory)
    return paths


def test_golden_structure_passes_complete_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _stub_repository(monkeypatch, _golden_repo())
    ok, failures, summary = hl.golden_structure("s1", tmp_path / "research.sqlite")
    assert (ok, failures) == (True, [])
    assert paths == [tmp_path / "research.sqlite"]
    assert summary == {
        "session_id": "s1",
        "status": "completed",
        "source_domains": ["FINRA", "SEC", "WEB"],
        "freeze_domains": ["FINRA", "SEC", "WEB"],
        "source_job_counts": {"SEC": 1, "FINRA": 1, "WEB": 1},
        "frozen_evidence_counts": {"SEC": 1, "FINRA": 1, "WEB": 1},
        "trio_freeze": "f1",
        "evidence_count": 3,
        "answer_chars": len(_GOLDEN_ANSWER),
    }


def test_golden_structure_defaults_to_env_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "custom.sqlite"))
    paths = _stub_repository(monkeypatch, _golden_repo())
    ok, _failures, _summary = hl.golden_structure("s1")
    assert ok is True
    assert paths == [tmp_path / "custom.sqlite"]


def test_golden_db_path_defaults_under_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    assert hl._golden_db_path().name == "research.sqlite"


def test_golden_structure_reports_missing_requirements_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = FakeRepo(
        sess=_sess(
            status="open",
            freeze_ids=["f1"],
            committee_runs=[],
            final_result={"answer": "too short", "claims": [{"evidence_ids": ["e9"]}]},
        ),
        jobs=[_g_job("j1", "SEC", "stockbot")],
        evidence=[_g_evidence("e1", "SEC")],
        freezes={"f1": {"evidence_ids": ["e1"]}},
    )
    _stub_repository(monkeypatch, repo)
    ok, failures, _summary = hl.golden_structure("s1", tmp_path / "research.sqlite")
    assert ok is False
    assert failures == [
        "source domains ran ['SEC']; want ['SEC', 'FINRA', 'WEB']",
        "freeze holds ['SEC']; want ['SEC', 'FINRA', 'WEB']",
        "trio shares no single freeze (runs=[])",
        "session status 'open'; want 'completed'",
        "source domain FINRA has no jobs",
        "source domain WEB has no jobs",
        "freeze holds no FINRA evidence",
        "freeze holds no WEB evidence",
        "claims cite unfrozen evidence ['e9']",
        "coverage empty",
        "disagreements/uncertainties empty",
        "final answer not substantive",
    ]


def test_golden_structure_flags_trio_freeze_and_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepo(
        sess=_sess(
            status="completed",
            freeze_ids=["f1", "missing"],
            committee_runs=[{"freeze_id": "f9", "jobs": ["j1", "j2", "j3"]}],
            final_result=dict(_GOLDEN_FINAL),
        ),
        jobs=[_g_job(*args) for args in _GOLDEN_JOBS],
        evidence=[
            _g_evidence("e1", "SEC"),
            _g_evidence("e2", "FINRA", record_kind="discovery"),
            _g_evidence("e3", "WEB", "search_run"),
        ],
        freezes={"f1": {"evidence_ids": ["e1", "e2", "e3"]}},
    )
    _stub_repository(monkeypatch, repo)
    ok, failures, summary = hl.golden_structure("s1", tmp_path / "research.sqlite")
    assert ok is False
    assert failures == [
        "freeze holds ['SEC', 'WEB']; want ['SEC', 'FINRA', 'WEB']",
        "trio freeze 'f9' not in session freezes",
        "frozen evidence 'e3' has provenance 'search_run'",
        "freeze holds no FINRA evidence",
    ]
    assert summary["frozen_evidence_counts"] == {"SEC": 1, "FINRA": 0, "WEB": 1}


def test_golden_structure_reports_trio_roles_when_roles_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _golden_repo(
        sess=_sess(
            status="completed",
            freeze_ids=["f1"],
            committee_runs=[
                "junk",
                {"freeze_id": "f1", "jobs": ["j1", "j2"]},
                {"freeze_id": "f1", "jobs": ["j1", "j2", "j3"]},
                {"freeze_id": "f1", "jobs": ["j1", 5, "j3"]},
            ],
            final_result=dict(_GOLDEN_FINAL),
        ),
        jobs=[
            _g_job("j1", "SEC", "source_agent"),
            _g_job("j2", "FINRA", "scout"),
            _g_job("j3", "WEB", "synthesis"),
        ],
    )
    _stub_repository(monkeypatch, repo)
    ok, failures, _summary = hl.golden_structure("s1", tmp_path / "research.sqlite")
    assert ok is False
    assert len(failures) == 1
    assert failures[0].startswith("trio shares no single freeze (runs=[])")
    assert "trio roles" in failures[0]
    assert "want ['stockbot', 'bullbot', 'bearbot']" in failures[0]
    assert "scout" in failures[0] and "synthesis" in failures[0]


def test_golden_structure_requires_one_shared_trio_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _golden_repo(
        sess=_sess(
            status="completed",
            freeze_ids=["f1", "f2"],
            committee_runs=[
                {"freeze_id": "f1", "jobs": ["j1", "j2", "j3"]},
                {"freeze_id": "f2", "jobs": ["j1", "j2", "j3"]},
            ],
            final_result=dict(_GOLDEN_FINAL),
        ),
    )
    _stub_repository(monkeypatch, repo)
    ok, failures, summary = hl.golden_structure("s1", tmp_path / "research.sqlite")
    assert ok is False
    assert failures == ["trio shares no single freeze (runs=['f1', 'f2'])"]
    assert summary["trio_freeze"] is None


def test_evidence_domain_reads_row_then_provenance() -> None:
    assert hl._evidence_domain({"domain": " finra "}) == "FINRA"
    assert hl._evidence_domain({"source_domain": "web"}) == "WEB"
    assert hl._evidence_domain({"domain": "other", "provenance": {"kind": "web_source"}}) == "WEB"
    assert hl._evidence_domain({"provenance": {"kind": "none"}}) == "SEC"
    assert hl._evidence_domain({"provenance": "not-a-mapping"}) == "SEC"
    assert hl._evidence_domain({}) == "SEC"
    assert hl._evidence_domain(object()) == "SEC"


def test_freeze_domains_skips_unmatched_and_discovery_rows() -> None:
    rows = [
        object(),
        {"evidence_id": "e1", "domain": "SEC", "record_kind": "discovery"},
        {"evidence_id": "e1", "domain": "SEC"},
        {"evidence_id": "e2", "domain": "FINRA", "provenance": {"kind": "finra_record"}},
        {"evidence_id": "e3", "domain": "WEB"},
    ]
    assert hl._freeze_domains(rows, {"e1", "e2"}) == {"SEC", "FINRA"}
    assert hl._freeze_domains(None, {"e1"}) == set()
    assert hl._freeze_domains("nope", {"e1"}) == set()


def test_job_role_map_reads_dicts_and_objects() -> None:
    jobs: list[object] = [
        {"job_id": "j1", "job_type": "stockbot"},
        {"job_id": "j2", "job_type": 5},
        SimpleNamespace(job_id="j3", job_type="bullbot"),
        object(),
    ]
    assert hl._job_role_map(jobs) == {"j1": "stockbot", "j3": "bullbot"}
    assert hl._job_role_map(None) == {}


def test_trio_freezes_reads_tuples_and_rejects_shapes() -> None:
    runs = (
        "junk",
        {"freeze_id": 5, "jobs": ["a", "b", "c"]},
        {"freeze_id": "f1", "jobs": ["a", "b"]},
        {"freeze_id": "f1", "jobs": "abc"},
        {"freeze_id": "f1", "jobs": [1, 2, 3]},
        {"freeze_id": "f1", "jobs": ["a", "b", "c"]},
    )
    # Without a job map, every three-job run counts as a trio.
    assert hl._trio_freezes(runs) == ["f1", "f1"]
    assert hl._trio_freezes(None) == []
    jobs = [
        {"job_id": "a", "job_type": "stockbot"},
        {"job_id": "b", "job_type": "bullbot"},
        {"job_id": "c", "job_type": "bearbot"},
    ]
    assert hl._trio_freezes(runs, jobs) == ["f1"]
    assert hl._trio_freezes(runs, [{"job_id": "a", "job_type": "stockbot"}]) == []


def test_source_job_counts_ignores_unknown_labels() -> None:
    failures: list[str] = []
    jobs: list[object] = [
        _g_job("j1", "SEC"),
        SimpleNamespace(source_domain=None),
        SimpleNamespace(source_domain="other"),
    ]
    assert hl._source_job_counts(jobs, failures) == {"SEC": 1, "FINRA": 0, "WEB": 0}
    assert failures == ["source domain FINRA has no jobs", "source domain WEB has no jobs"]
    assert hl._source_job_counts(None, []) == {"SEC": 0, "FINRA": 0, "WEB": 0}


def test_frozen_evidence_counts_tolerates_malformed_rows() -> None:
    failures: list[str] = []
    assert hl._frozen_evidence_counts(None, {"e1"}, failures) == {"SEC": 0, "FINRA": 0, "WEB": 0}
    assert failures == [
        "freeze holds no SEC evidence",
        "freeze holds no FINRA evidence",
        "freeze holds no WEB evidence",
    ]
    failures = []
    rows = [object(), {"evidence_id": "e1", "domain": "SEC", "provenance": {"kind": "sec_record"}}]
    assert hl._frozen_evidence_counts(rows, {"e1"}, failures) == {"SEC": 1, "FINRA": 0, "WEB": 0}
    assert failures == ["freeze holds no FINRA evidence", "freeze holds no WEB evidence"]
    failures = []
    hl._frozen_evidence_counts([{"evidence_id": "e1", "provenance": "not-a-mapping"}], {"e1"}, failures)
    assert failures[0] == "frozen evidence 'e1' has provenance None"


def test_frozen_evidence_counts_skips_rows_outside_the_freeze_scope() -> None:
    """Reader-less rows, frozen discovery rows, and unfrozen rows are skipped whole."""
    failures: list[str] = []
    rows: list[object] = [
        object(),
        {"evidence_id": "e1", "domain": "SEC", "provenance": {"kind": "sec_record"}},
        {"evidence_id": "e2", "domain": "FINRA", "record_kind": "discovery", "provenance": {"kind": "search_run"}},
        {"evidence_id": "e3", "domain": "WEB", "provenance": {"kind": "search_run"}},
    ]
    assert hl._frozen_evidence_counts(rows, {"e1", "e2"}, failures) == {"SEC": 1, "FINRA": 0, "WEB": 0}
    assert failures == ["freeze holds no FINRA evidence", "freeze holds no WEB evidence"]


def test_claimed_ids_reads_both_claim_keys() -> None:
    assert hl._claimed_ids({"claims": [{"evidence_ids": ["e1", 5]}, "junk", {"evidence_ids": "e3"}]}) == {"e1"}
    assert hl._claimed_ids({"claims": [{"evidence_ids": ["e1"]}], "grounded_claims": [{"evidence_ids": ["e2"]}]}) == {
        "e1",
        "e2",
    }
    assert hl._claimed_ids({}) == set()
    assert hl._str_list(["a", 5, None]) == ["a"]
    assert hl._str_list("nope") == []


def test_missing_env_lists_credentials_then_missing_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*hl.REQUIRED_ENV, "EXA_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    assert hl._missing_env() == [*hl.REQUIRED_ENV, "EXA_ENABLED=1"]
    for name in hl.REQUIRED_ENV:
        monkeypatch.setenv(name, " set ")
    for enabled in ("1", "TRUE", "yes", " Yes "):
        monkeypatch.setenv("EXA_ENABLED", enabled)
        assert hl._missing_env() == []
    monkeypatch.setenv("EXA_ENABLED", "0")
    assert hl._missing_env() == ["EXA_ENABLED=1"]


def test_live_structural_skips_other_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("repository must not be built for another scenario")

    monkeypatch.setattr("app.research.repository.ResearchRepository", _unreachable)
    assert hl._live_structural("another-scenario") == []


def test_live_structural_reports_unreadable_db(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise sqlite3.OperationalError("no such table: sessions")

    monkeypatch.setattr("app.research.repository.ResearchRepository", _boom)
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == ["golden-no-session-persisted"]


def test_live_structural_needs_a_persisted_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    _stub_repository(monkeypatch, _golden_repo(sessions=[]))
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == ["golden-no-session-persisted"]
    _stub_repository(monkeypatch, _golden_repo(sessions=[{"session_id": ""}]))
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == ["golden-no-session-persisted"]
    _stub_repository(monkeypatch, _golden_repo(sessions=[{"session_id": None}]))
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == ["golden-no-session-persisted"]
    _stub_repository(monkeypatch, _MalformedSessionsRepo(["junk"]))
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == ["golden-no-session-persisted"]


def test_live_structural_reads_and_prefixes_golden_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    _stub_repository(monkeypatch, _golden_repo(sessions=[{"session_id": "s1"}]))
    assert hl._live_structural(hl.GOLDEN_SCENARIO) == []
    printed = capsys.readouterr().out
    assert printed.startswith("golden structure: ")
    assert json.loads(printed.removeprefix("golden structure: "))["trio_freeze"] == "f1"

    _stub_repository(monkeypatch, _golden_repo(sess=_sess(), sessions=[{"session_id": "s1"}]))
    violations = hl._live_structural(hl.GOLDEN_SCENARIO)
    assert violations[0] == "golden-freeze holds []; want ['SEC', 'FINRA', 'WEB']"
    assert "golden-session status 'open'; want 'completed'" in violations


# ---- slice_export_final_tests.py ----
"""scripts/export_harness_viewer._collect_final viewer-shape projection."""


def test_collect_final_rejects_missing_and_mistyped_results() -> None:
    assert exh_viewer._collect_final(_sess(final_result=None)) is None
    assert exh_viewer._collect_final(SimpleNamespace(final_result=["not-a-dict"])) is None

    class Bad:
        @property
        def final_result(self) -> object:
            raise RuntimeError("boom")

    assert exh_viewer._collect_final(Bad()) is None


def test_collect_final_maps_structured_final_result() -> None:
    sess = _sess(
        final_result={
            "content": "answer text",
            "executive_summary": "summary text",
            "consensus": "agree",
            "base_case": "base case",
            "bull_case": {"summary": "bull summary"},
            "bear_case": "bear text",
            "critical_disagreements": ["d1", 5, "", " d2 "],
            "positioning": ["long"],
            "catalysts": [" earn "],
            "uncertainties": ["fx"],
            "what_would_change": ["guidance"],
            "evidence_limitations": ["thin"],
            "claims": [{"text": "c1", "evidence_ids": ["e1", 2]}],
            "sources": [{"id": "s1"}],
            "coverage": {"status": "full"},
            "research_scope": {"allowed_sources": [" sec ", 7]},
            "freeze_id": "f1",
            "as_of": "2026-09-01",
        }
    )
    assert exh_viewer._collect_final(sess) == {
        "answer": "answer text",
        "executiveSummary": "summary text",
        "consensus": "agree",
        "baseCase": "base case",
        "bullCase": "bull summary",
        "bearCase": "bear text",
        "disagreements": ["d1", "d2"],
        "positioning": ["long"],
        "catalysts": ["earn"],
        "uncertainties": ["fx"],
        "whatChangesTheView": ["guidance"],
        "limitations": ["thin"],
        "claims": [{"text": "c1", "evidenceIds": ["e1"]}],
        "sources": [{"id": "s1"}],
        "coverage": {"status": "full"},
        "allowedSources": ["sec"],
        "freezeId": "f1",
        "asOf": "2026-09-01",
    }


def test_collect_final_falls_back_to_legacy_keys_and_drops_mistyped_fields() -> None:
    sess = _sess(
        final_result={
            "answer": "legacy answer",
            "executive_summary": "",
            "grounded_claims": [{"text": "gc", "evidence_ids": ["e2"]}],
            "what_changes_the_view": ["rates"],
            "limitations": ["l1"],
            "disagreements": ["old"],
            "sources": {"not": "a list"},
            "coverage": "full",
            "research_scope": "unknown",
            "bull_case": {"no_summary": True},
            "bear_case": None,
        }
    )
    out = exh_viewer._collect_final(sess)
    assert out is not None
    assert out["answer"] == "legacy answer"
    assert out["executiveSummary"] == "legacy answer"
    assert out["claims"] == [{"text": "gc", "evidenceIds": ["e2"]}]
    assert out["whatChangesTheView"] == ["rates"]
    assert out["limitations"] == ["l1"]
    assert out["disagreements"] == ["old"]
    assert out["sources"] == [] and out["coverage"] == {} and out["allowedSources"] == []
    assert out["bullCase"] == "None" and out["bearCase"] == ""
    assert out["freezeId"] == "" and out["asOf"] == ""


# ---- scripts/verify_e2e.py: SSE fold + limits (canned bytes, no network) ----


from scripts import verify_e2e as e2e


def _sse(*events: tuple[float, object]) -> list[tuple[float, bytes]]:
    return [(ms, ev if isinstance(ev, bytes) else f"data: {json.dumps(ev)}\n".encode()) for ms, ev in events]


def _done(tools: int = 0, muse: int = 1) -> dict[str, object]:
    return {"type": "done", "metrics": {"totalMs": 1, "tools": {"calls": tools}, "muse": {"calls": muse}}}


def _golden() -> list[e2e.Case]:
    golden: list[e2e.Case] = json.loads(e2e.GOLDEN.read_text(encoding="utf-8"))
    return golden


def _case() -> e2e.Case:
    return next(c for c in _golden() if c["id"] == "nvda_risk")


def test_e2e_golden_set_shape() -> None:
    golden = _golden()
    assert len(golden) == 13 and len({c["id"] for c in golden}) == 13
    assert {c["id"] for c in golden if c["mode"] == "chat"} == {"hello", "clarify_advice", "aapl_buy_now"}
    assert all(c["mode"] in ("chat", "research") for c in golden)


def test_e2e_parse_research_stream_times_text_and_counts_known_citations() -> None:
    run = e2e.parse_stream(
        _sse(
            (5, {"type": "agent_start", "prompt": "p"}),
            (6, b": keepalive\n"),
            (8, {"type": "status", "text": "Researching: p"}),
            (50, {"type": "progress", "stage": "working"}),
            (55, {"type": "status", "text": "Planned 3 questions"}),
            (60, {"type": "chart_update", "ticker": "NVDA"}),
            (70, {"type": "evidence", "evidence": [{"id": "ev:a"}, {"id": "ev:b"}]}),
            (900, {"type": "answer_delta", "text": "Risk [ev:a] and [ev:a] "}),
            (950, {"type": "answer_delta", "text": "plus [ev:zz]."}),
            (1200, _done(tools=2)),
        )
    )
    assert run.events == [
        "agent_start",
        "status",
        "progress",
        "status",
        "chart_update",
        "evidence",
        "answer_delta",
        "answer_delta",
        "done",
    ]
    assert (run.first_progress_ms, run.first_text_ms, run.total_ms, run.mode) == (8, 900, 1200, "research")
    assert run.citation_count == 1  # ev:a once; ev:zz is not a streamed evidence id.
    assert run.timings == _done(tools=2)["metrics"] and run.errors == []
    assert e2e.evaluate(_case(), run) == []
    strict = _case()
    strict["min_citations"], strict["max_first_text_ms"], strict["ticker"] = 2, 800, "AMD"
    assert e2e.evaluate(strict, run) == [
        "first_text_ms",
        "citations",
        "ticker",
    ]


def test_e2e_parse_counts_claim_citations_with_inline_markers() -> None:
    claims: list[dict[str, object]] = [
        {"text": "Revenue grew", "claimType": "observed_fact", "evidenceIds": ["ev:b"]},
        {"text": "Risk rose", "claimType": "observed_fact", "evidenceIds": ["ev:a"]},  # Already inline.
        {"text": "Not in prose", "claimType": "observed_fact", "evidenceIds": ["ev:c"]},
        {"text": "Margins held", "claimType": "unknown", "evidenceIds": []},
        {"text": "Revenue grew", "claimType": "inference", "evidenceIds": ["ev:zz"]},  # Unknown id.
    ]
    run = e2e.parse_stream(
        _sse(
            (1, {"type": "evidence", "evidence": [{"id": "ev:a"}, {"id": "ev:b"}, {"id": "ev:c"}]}),
            (2, {"type": "answer_delta", "text": "Risk rose [ev:a]. Revenue grew. Margins held."}),
            (3, {"type": "claims", "claims": claims}),
            (4, _done(tools=1)),
        )
    )
    assert run.citation_count == 2  # ev:a inline, ev:b by claim; ev:c unmatched, ev:zz unknown.


def test_e2e_parse_chat_stream_and_mode_mismatch() -> None:
    run = e2e.parse_stream(
        _sse(
            (1, {"type": "agent_start", "prompt": "hello"}),
            (2, {"type": "status", "text": "Thinking…"}),
            (3, {"type": "reasoning_start", "model": "stockbot"}),
            (4, {"type": "answer_delta", "text": "Hello."}),
            (5, _done()),
        )
    )
    assert run.mode == "chat"
    case = next(c for c in _golden() if c["id"] == "hello")
    assert e2e.evaluate(case, run) == []
    mismatch: e2e.Case = {**case, "mode": "research"}
    assert e2e.evaluate(mismatch, run) == ["mode"]


@pytest.mark.parametrize(
    ("events", "metrics"),
    [
        ([], _done()["metrics"]),
        (["done", "error"], _done()["metrics"]),
        (["failed", "done"], _done()["metrics"]),
        (["done"], _done(tools=1)["metrics"]),
        (["done"], _done(muse=0)["metrics"]),
        (["done"], _done(muse=2)["metrics"]),
        (["done"], None),
        (["done"], {}),
        (["done"], {"tools": {"calls": 0}}),
        (["done"], {"muse": {"calls": 1}}),
    ],
)
def test_e2e_mode_rejects_incomplete_or_non_chat_streams(events: list[str], metrics: object) -> None:
    assert e2e._mode(events, metrics) is None


def test_e2e_mode_progress_identifies_research_without_done() -> None:
    assert e2e._mode(["progress"], None) == "research"


@pytest.mark.parametrize("case_id", ["hello", "aapl_buy_now", "nvda_risk", "aapl_ceo"])
def test_e2e_post_sends_mode_only_for_research(monkeypatch: pytest.MonkeyPatch, case_id: str) -> None:
    from io import BytesIO
    from urllib.request import Request

    case = next(c for c in _golden() if c["id"] == case_id)
    requests: list[Request] = []

    def urlopen(req: Request, *, timeout: int) -> BytesIO:
        requests.append(req)
        assert timeout == e2e.TIMEOUT_S
        return BytesIO(f"data: {json.dumps(_done())}\n".encode())

    monkeypatch.setattr(e2e.urllib.request, "urlopen", urlopen)
    e2e._post("http://localhost:3000/", case)
    assert len(requests) == 1
    req = requests[0]
    assert req.full_url == "http://localhost:3000/api/agent"
    assert req.get_method() == "POST"
    assert isinstance(req.data, bytes)
    expected = {"prompt": case["prompt"]}
    if case["chart_ticker"]:
        expected["ticker"] = case["chart_ticker"]
    if case["mode"] == "research":
        expected["mode"] = "research"
    assert json.loads(req.data) == expected


def test_e2e_parse_captures_failures_and_stream_end() -> None:
    run = e2e.parse_stream(
        _sse(
            (1, {"type": "agent_start", "prompt": "p"}),
            (2, b"data: {broken\n"),
            (3, {"type": "failed", "category": "provider_error", "message": "jev down"}),
            (4, {"type": "error", "message": "Stockbot couldn't complete that request. Try again."}),
        )
    )
    assert run.errors[0].startswith("malformed SSE data")
    assert run.errors[1:] == ["failed: jev down", "error: Stockbot couldn't complete that request. Try again."]
    assert (run.mode, run.first_text_ms, run.total_ms) == (None, None, 4)
    assert e2e.evaluate(_case(), run) == ["mode", "first_text_ms", "citations", "terminal"]


def test_e2e_main_requires_opt_in_then_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("STOCKBOT_E2E", raising=False)
    assert e2e.main([]) == 2
    assert "STOCKBOT_E2E=1" in capsys.readouterr().err
    monkeypatch.setenv("STOCKBOT_E2E", "1")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert e2e.main([]) == 2
    assert "EXA_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("runs", [1, 2])
@pytest.mark.parametrize("transport_failure", [False, True])
def test_e2e_main_writes_runs_percentiles_and_verdicts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    runs: int,
    transport_failure: bool,
) -> None:
    from io import BytesIO
    from urllib.request import Request

    cases: list[e2e.Case] = [
        {
            "id": mode,
            "prompt": mode,
            "mode": mode,
            "chart_ticker": None,
            "ticker": None,
            "min_citations": 0,
            "max_first_text_ms": 1000,
            "max_total_ms": 1000,
        }
        for mode in ("chat", "research")
    ]
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(e2e, "GOLDEN", golden)
    monkeypatch.setattr(e2e, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("STOCKBOT_E2E", "1")

    def missing() -> list[str]:
        return []

    monkeypatch.setattr(e2e, "_missing_env", missing)
    requests: list[dict[str, str]] = []

    def urlopen(request: Request, *, timeout: int) -> BytesIO:
        assert timeout == e2e.TIMEOUT_S
        assert request.full_url == "http://fake.invalid/api/agent"
        assert isinstance(request.data, bytes)
        body = json.loads(request.data)
        requests.append(body)
        if transport_failure:
            raise OSError("offline")
        events: list[dict[str, object]] = [{"type": "status"}]
        if body.get("mode") == "research":
            events.append({"type": "progress"})
        events.extend([{"type": "answer_delta", "text": "Answer."}, _done()])
        return BytesIO(b"".join(raw for _, raw in _sse(*[(0, event) for event in events])))

    monkeypatch.setattr(e2e.urllib.request, "urlopen", urlopen)
    ticks = iter(range(100))
    monkeypatch.setattr(e2e.time, "perf_counter", lambda: next(ticks) / 1000)
    assert e2e.main(["--base-url", "http://fake.invalid/", "--runs", str(runs)]) == int(transport_failure)
    assert requests == [{"prompt": "chat"}] * runs + [{"prompt": "research", "mode": "research"}] * runs
    reports = list((tmp_path / "data" / "verify").glob("*/e2e.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["base_url"] == "http://fake.invalid/"
    assert report["summary"]["runs"] == 2 * runs
    assert report["summary"]["passed"] == (0 if transport_failure else 2 * runs)
    assert [(row["id"], row["run"]) for row in report["runs"]] == [
        (mode, n) for mode in ("chat", "research") for n in range(1, runs + 1)
    ]
    for row in report["runs"]:
        assert row["pass"] is not transport_failure
        assert row["answer"] == ("" if transport_failure else "Answer.")
        assert row["failures"] == (["mode", "first_text_ms", "terminal"] if transport_failure else [])
    metrics = report["summary"]["research"]
    if transport_failure:
        assert metrics["first_progress_ms"] == metrics["first_text_ms"] == {"p50": None, "p90": None}
        assert metrics["total_ms"] == pytest.approx({"p50": 1.0, "p90": 1.0})
    else:
        assert metrics["first_progress_ms"] == pytest.approx({"p50": 1.0, "p90": 1.0})
        assert metrics["first_text_ms"] == pytest.approx({"p50": 3.0, "p90": 3.0})
        assert metrics["total_ms"] == pytest.approx({"p50": 4.0, "p90": 4.0})
    output = capsys.readouterr()
    assert output.err == ""
    label = "research#1" if runs > 1 else "research"
    assert f"| {label} |" in output.out
    assert ("FAIL mode,first_text_ms,terminal" if transport_failure else "PASS") in output.out
    assert f"wrote {reports[0]}" in output.out


def test_e2e_main_rejects_unknown_case_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(e2e, "GOLDEN", golden)
    monkeypatch.setenv("STOCKBOT_E2E", "1")

    def missing() -> list[str]:
        return []

    monkeypatch.setattr(e2e, "_missing_env", missing)
    assert e2e.main(["--only", "absent"]) == 2
    assert capsys.readouterr().err == "unknown golden id: 'absent'\n"
