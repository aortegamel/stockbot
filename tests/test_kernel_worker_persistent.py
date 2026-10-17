"""Focused kernel-worker persistence test: shared JEV identity + ready-first main."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import NamedTuple, Never, override
from unittest import mock

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

import app.research.kernel_worker as kw
from app import child_spawn
from app.research import scheduler as sched
from app.research.models import DecisionRecord, Job, JSONValue, ResearchNode
from app.research.repository import ResearchRepository


class _FakeJev:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.decide_calls = 0

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    async def decide(
        self,
        state: object,
        questions: Mapping[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        self.decide_calls += 1
        out: dict[str, object] = {}
        for qid in questions:
            out[qid] = {"choice": "analyze"}
        return out


def _proposals(n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "id": f"rs:t-q{i}",
                "objectiveId": "rs:t",
                "question": f"q{i}?",
                "dependsOn": [],
                "whyItMatters": "Route question.",
            }
        )
    return rows


def _needle_hook(**kwargs: object) -> dict[str, object]:
    return {"tool": "t", "arguments": {}}


def _empty_strs() -> list[str]:
    out: list[str] = []
    return out


def _empty_objs() -> list[object]:
    out: list[object] = []
    return out


@pytest.fixture
def _offline_research(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Kernel(sched._Kernel):
        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return {"session_id": session_id, "objective": "test", "query": "", "as_of": None}

    kernel = _Kernel()

    def _get_kernel() -> sched._Kernel:
        return kernel

    def _list_evidence(self: ResearchRepository, session_id: str) -> list[dict[str, JSONValue]]:
        return []

    def _list_nodes(self: ResearchRepository, session_id: str) -> list[ResearchNode]:
        return []

    def _list_decisions(self: ResearchRepository, session_id: str) -> list[DecisionRecord]:
        return []

    def _list_jobs(self: ResearchRepository, session_id: str) -> list[Job]:
        return []

    def _no_database(self: ResearchRepository) -> Never:
        raise AssertionError("Persistent worker tests must not access the database")

    monkeypatch.setattr(sched, "_default_kernel", _get_kernel)
    monkeypatch.setattr(ResearchRepository, "list_evidence", _list_evidence)
    monkeypatch.setattr(ResearchRepository, "list_nodes", _list_nodes)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _list_decisions)
    monkeypatch.setattr(ResearchRepository, "list_jobs", _list_jobs)
    monkeypatch.setattr(ResearchRepository, "_connect", _no_database)


def test_two_requests_share_one_jev_identity(monkeypatch: pytest.MonkeyPatch, _offline_research: None) -> None:
    monkeypatch.setattr(kw, "_JEV", None)
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    fake = _FakeJev()

    def _get_jev() -> _FakeJev:
        return fake

    def _get_generate() -> Callable[..., object]:
        return _needle_hook

    monkeypatch.setattr(kw, "_shared_jev", _get_jev)
    monkeypatch.setattr(kw, "_shared_needle_generate", _get_generate)
    seen: list[object] = []

    async def _fake_run(sid: str, **hooks: object) -> dict[str, object]:
        seen.append(hooks.get("jev"))
        assert hooks.get("needle_generate") is not None
        nodes: list[object] = []
        out: dict[str, object] = {"status": "complete", "nodes": nodes}
        return out

    def _no_nodes(sid: str, objective: str, admitted: object) -> None:
        return None

    def _noop_node(*args: object, **kwargs: object) -> None:
        return None

    async def _two_intake(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return _proposals(2)

    def _no_hit(*args: object, **kwargs: object) -> list[str]:
        return _empty_strs()

    def _research(*args: object, **kwargs: object) -> str:
        return "rs:t"

    monkeypatch.setattr("app.research.scheduler.run", _fake_run)
    monkeypatch.setattr(kw, "_create_nodes_topological", _no_nodes)
    monkeypatch.setattr(kw, "_graph_intake", _two_intake)
    monkeypatch.setattr(kw, "_registry_portfolio_hit", _no_hit)

    from app.research import service

    monkeypatch.setattr(service, "create_research", _research)
    monkeypatch.setattr(service, "create_node", _noop_node)

    out1 = kw._run({"id": "r1", "op": "run", "prompt": "alpha?"})
    out2 = kw._run({"id": "r2", "op": "run", "prompt": "beta?"})
    assert out1["id"] == "r1" and out2["id"] == "r2"
    assert out1["sessionId"] == "rs:t"
    assert out2["sessionId"] == "rs:t"
    assert seen == [fake, fake]  # same scheduler-run identity across both requests
    assert fake.decide_calls == 2  # one entry disposition round per request; nodes follow in scheduler.run


def test_stalled_run_reports_escalated_without_guard(monkeypatch: pytest.MonkeyPatch, _offline_research: None) -> None:
    monkeypatch.setattr(kw, "_JEV", None)
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    fake = _FakeJev()

    def _get_jev() -> _FakeJev:
        return fake

    def _get_generate() -> Callable[..., object]:
        return _needle_hook

    monkeypatch.setattr(kw, "_shared_jev", _get_jev)
    monkeypatch.setattr(kw, "_shared_needle_generate", _get_generate)

    async def _stalled_run(sid: str, **hooks: object) -> dict[str, object]:
        assert hooks.get("jev") is fake
        return {
            "status": "stalled",
            "nodes": _empty_objs(),
            "incomplete_guard": False,
            "reason": "stalled: no progress",
        }

    def _no_hit(*args: object, **kwargs: object) -> list[str]:
        return _empty_strs()

    async def _empty_intake(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return []

    def _research(*args: object, **kwargs: object) -> str:
        return "rs:stalled"

    monkeypatch.setattr("app.research.scheduler.run", _stalled_run)
    monkeypatch.setattr(kw, "_graph_intake", _empty_intake)
    monkeypatch.setattr(kw, "_registry_portfolio_hit", _no_hit)

    from app.research import service

    def _noop_node(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(service, "create_research", _research)
    monkeypatch.setattr(service, "create_node", _noop_node)
    out = kw._run({"id": "stall-1", "op": "run", "prompt": "stall?"})
    assert out["id"] == "stall-1"
    assert out["sessionId"] == "rs:stalled"
    assert out["stalled"] is True
    assert out["escalated"] is True
    assert out["incomplete_guard"] is False
    assert out["escalations"] == 1
    assert out["failures"] == {"stalled": 1}
    assert out["stalled_reason"] == "stalled: no progress"


def _response_row(rid: str, objective: str) -> dict[str, object]:
    return {
        "id": rid,
        "objective": objective,
        "evidence": _empty_objs(),
        "nodes": _empty_objs(),
        "decisions": _empty_objs(),
        "unresolved": _empty_objs(),
        "incomplete_guard": False,
        "toolExecutions": _empty_objs(),
        "needleDecisions": _empty_objs(),
        "toolCalls": _empty_objs(),
        "failures": {},
        "escalations": 0,
        "escalated": False,
    }


def test_main_ready_precedes_responses_and_closes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeJev()
    monkeypatch.setattr(kw, "_JEV", fake)
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    started = {"needle": 0}

    def _needle_start() -> None:
        started["needle"] += 1

    def _needle_close() -> None:
        started["needle"] += 1000

    needle_mod = types.SimpleNamespace(start=_needle_start, close=_needle_close, generate_arguments=_needle_hook)
    monkeypatch.setitem(sys.modules, "app.needle_client", needle_mod)

    def _startup() -> _FakeJev:
        fake.start()
        needle_mod.start()
        return fake

    monkeypatch.setattr(kw, "_startup", _startup)
    queue: list[dict[str, object]] = [_response_row("a", "A"), _response_row("b", "B")]

    def _fake_run_call(req: object, **kwargs: object) -> dict[str, object]:
        assert isinstance(req, dict)
        row = queue.pop(0)
        raw = req.get("id")
        row["id"] = raw if isinstance(raw, str) else "?"
        return row

    shutdowns = {"n": 0}

    def _shutdown() -> None:
        shutdowns["n"] += 1
        fake.close()
        needle_mod.close()

    monkeypatch.setattr(kw, "_run", _fake_run_call)
    monkeypatch.setattr(kw, "_shutdown", _shutdown)

    body = (
        json.dumps({"id": "a", "op": "run", "prompt": "A?"})
        + "\n"
        + json.dumps({"id": "b", "op": "run", "prompt": "B?"})
        + "\n"
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    def _no_signal(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr("signal.signal", _no_signal)

    kw.main()

    parsed: list[object] = []
    for line in buf.getvalue().splitlines():
        if line.strip():
            parsed.append(json.loads(line))
    assert fake.started == 1 and started["needle"] % 1000 == 1  # both started exactly once
    assert parsed[0] == {"type": "ready"}  # readiness precedes every response
    ids: list[object] = []
    for row in parsed[1:]:
        assert isinstance(row, dict)
        ids.append(row.get("id"))
    assert ids == ["a", "b"]  # request IDs preserved
    assert fake.closed == 1 and started["needle"] == 1001  # EOF closed once


def test_route_served_while_run_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeJev()
    entered = threading.Event()
    release = threading.Event()

    def _blocked_run(req: object, **kwargs: object) -> dict[str, object]:
        assert isinstance(req, dict) and req.get("id") == "run1"
        entered.set()
        assert release.wait(timeout=10)
        return _response_row("run1", "R")

    def _fast_route(req: object, **kwargs: object) -> dict[str, object]:
        assert isinstance(req, dict)
        raw = req.get("id")
        return {"id": raw if isinstance(raw, str) else "?", "route": "research_required", "personas": None}

    monkeypatch.setattr(kw, "_startup", lambda: fake)
    monkeypatch.setattr(kw, "_run", _blocked_run)
    monkeypatch.setattr(kw, "_route", _fast_route)
    monkeypatch.setattr(kw, "_shutdown", lambda: None)

    def _no_signal(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr("signal.signal", _no_signal)
    body = (
        json.dumps({"id": "run1", "op": "run", "prompt": "x"})
        + "\n"
        + json.dumps({"id": "r1", "op": "route", "prompt": "hi"})
        + "\n"
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))
    buf = io.StringIO()
    real_stdout = sys.stdout
    monkeypatch.setattr(sys, "stdout", buf)
    worker = threading.Thread(target=kw.main, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        deadline = time.monotonic() + 5
        while '"r1"' not in buf.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert '"r1"' in buf.getvalue()  # route answered while run still blocked (not 60s)
    finally:
        release.set()
        worker.join(timeout=10)
    monkeypatch.setattr(sys, "stdout", real_stdout)
    rows = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    by_id = {row.get("id") for row in rows if isinstance(row, dict) and "id" in row}
    assert {"run1", "r1"} <= by_id  # ids still correlate under concurrency


def test_watchlist_served_while_run_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page-load watchlist op runs on its own worker: it never queues behind a blocked run."""
    fake = _FakeJev()
    entered = threading.Event()
    release = threading.Event()
    served = threading.Event()

    def _blocked_run(req: object, **kwargs: object) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=10)
        return _response_row("run1", "R")

    def _watch(req: object) -> dict[str, object]:
        assert isinstance(req, dict) and req.get("watchlist") == ["NVDA"]
        served.set()
        return {"id": "w1", "watchlist": []}

    def _no_signal(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(kw, "_startup", lambda: fake)
    monkeypatch.setattr(kw, "_run", _blocked_run)
    monkeypatch.setattr(kw, "_watchlist", _watch)
    monkeypatch.setattr(kw, "_shutdown", lambda: None)
    monkeypatch.setattr("signal.signal", _no_signal)
    body = (
        json.dumps({"id": "run1", "op": "run", "prompt": "x"})
        + "\n"
        + json.dumps({"id": "w1", "op": "watchlist", "watchlist": ["NVDA"]})
        + "\n"
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))
    buf = io.StringIO()
    real_stdout = sys.stdout
    monkeypatch.setattr(sys, "stdout", buf)
    worker = threading.Thread(target=kw.main, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        assert served.wait(timeout=5)  # answered while the run is still held
        assert not release.is_set()
    finally:
        release.set()
        worker.join(timeout=10)
    monkeypatch.setattr(sys, "stdout", real_stdout)
    rows = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert {"run1", "w1"} <= {row.get("id") for row in rows if isinstance(row, dict)}


def test_main_answers_bad_lines_and_direct_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank lines report nothing; bad JSON, unknown ops, and failing direct ops answer in order."""
    fake = _FakeJev()
    monkeypatch.setattr(kw, "_startup", lambda: fake)
    monkeypatch.setattr(kw, "_shutdown", lambda: None)

    def _no_signal(signum: int, handler: object) -> None:
        return None

    def _route(req: Mapping[str, JSONValue], jev: object = None) -> dict[str, JSONValue]:
        assert jev is fake
        return {"id": kw._request_id(req), "route": "research_required", "personas": None}

    def _boom(req: Mapping[str, JSONValue], jev: object = None) -> dict[str, JSONValue]:
        raise RuntimeError("assess down")

    monkeypatch.setattr("signal.signal", _no_signal)
    monkeypatch.setattr(kw, "_route", _route)
    monkeypatch.setattr(kw, "_assess_final_response", _boom)
    bad_op = f"op must be one of {', '.join(kw._KERNEL_OPS)}"
    body = "\n".join(
        [
            "",
            "{oops",
            "[1, 2]",
            json.dumps({"op": "nope", "id": "n1"}),
            json.dumps({"id": "r1", "op": "route", "prompt": "hi"}),
            json.dumps({"id": "a1", "op": "assess_final_response", "text": "x"}),
        ]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(body + "\n"))
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    kw.main()

    raw_rows = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    rows = [row for row in raw_rows if isinstance(row, dict)]
    assert len(rows) == len(raw_rows) == 6  # the blank line answered nothing
    assert rows[0] == {"type": "ready"}
    assert [row.get("id") for row in rows[1:]] == ["?", "?", "n1", "r1", "a1"]
    assert rows[1]["terminal"] == {"category": "invalid_params", "message": "invalid JSON"}
    assert rows[2]["terminal"] == {"category": "invalid_params", "message": bad_op}
    assert rows[3]["terminal"] == rows[2]["terminal"]
    assert rows[4] == {"id": "r1", "route": "research_required", "personas": None}
    assert rows[5]["error"] == "worker failed: assess down"
    assert rows[5]["terminal"] == {"category": "provider_error", "message": "worker failed: assess down"}


_FRAMED_503 = (
    "The requested data is unavailable: one or more tool calls failed.\n\n"
    "Tool: get_threshold_securities\n  error: HTTP 503 from FINRA\n\n"
    "Next step: retry."
)


def test_watchlist_fetch_rows_keep_each_source_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fan-out row reports its own status, scalar metadata, and string refs; failures keep their text."""
    results: dict[tuple[str, str], object] = {
        ("AAPL", "list_sec_filings"): {
            "content": "rows",
            "meta": {
                "as_of": "2026-10-09",
                "returned_count": 3,
                "truncated": True,
                "total": 9,
                "source_refs": {"record_id": "AAPL:10-K", "uri": "https://example.test/AAPL", "known_at": 5},
            },
        },
        ("AAPL", "get_short_interest"): {
            "content": "",
            "meta": {"as_of": "2026-10-01", "returned_count": 0, "source_refs": {"known_at": None}},
        },
        ("AAPL", "get_threshold_securities"): {"error": _FRAMED_503},
        ("QQQ", "list_sec_filings"): ["not", "a", "mapping"],
        ("QQQ", "get_short_interest"): {"content": "rows"},
        ("QQQ", "get_threshold_securities"): {
            "content": "rows",
            "meta": {"as_of": {"day": 1}, "returned_count": 2, "total": None, "source_refs": "AAPL"},
        },
    }

    async def _invoke(
        invoke: object, to_outcome: object, tool: str, args: Mapping[str, object], *rest: object
    ) -> tuple[object, None]:
        symbol = str(args.get("identifier") or args.get("ticker"))
        if (symbol, tool) == ("AAPL", "get_reg_sho_volume"):
            await asyncio.Event().wait()  # never answers inside the budget
        if (symbol, tool) == ("QQQ", "get_reg_sho_volume"):
            raise RuntimeError("pool closed")
        return results[(symbol, tool)], None

    monkeypatch.setattr(sched, "_invoke_attempt_tool", _invoke)
    rows = asyncio.run(kw._watchlist_fetch(["AAPL", "QQQ"], 0.2))

    def head(symbol: str, tool: str) -> dict[str, JSONValue]:
        return {"symbol": symbol, "source": kw._WATCHLIST_SOURCES[tool], "tool": tool}

    assert rows == [
        {
            **head("AAPL", "list_sec_filings"),
            "status": "ok",
            "as_of": "2026-10-09",
            "returned_count": 3,
            "truncated": True,
            "total": 9,
            "ref": {"record_id": "AAPL:10-K", "uri": "https://example.test/AAPL"},
        },
        {**head("AAPL", "get_short_interest"), "status": "no_records", "as_of": "2026-10-01", "returned_count": 0},
        {**head("AAPL", "get_reg_sho_volume"), "status": "timeout", "error": "no result within 0.2s"},
        {**head("AAPL", "get_threshold_securities"), "status": "error", "error": "error: HTTP 503 from FINRA"},
        {**head("QQQ", "list_sec_filings"), "status": "error", "error": "no result"},
        {**head("QQQ", "get_short_interest"), "status": "ok"},
        {**head("QQQ", "get_reg_sho_volume"), "status": "error", "error": "RuntimeError: pool closed"},
        {**head("QQQ", "get_threshold_securities"), "status": "ok", "returned_count": 2},
    ]


def test_source_outcome_cites_evidence_only_for_answered_sources() -> None:
    """Intake records carry evidence_id; only ok/no_records answers keep it, and timeout wins over text."""
    tool = "list_sec_filings"
    head: dict[str, JSONValue] = {"symbol": "NVDA", "source": "sec_filings", "tool": tool}
    answered = {"result": {"meta": {"returned_count": 1}}, "evidence_id": "ev:NVDA:sec"}
    empty = {"result": {"meta": {"returned_count": 0}}, "evidence_id": "ev:NVDA:empty"}
    ok = kw._source_outcome("NVDA", tool, answered)
    assert ok == {**head, "status": "ok", "returned_count": 1, "evidence_id": "ev:NVDA:sec"}
    assert kw._source_outcome("NVDA", tool, empty) == {
        **head,
        "status": "no_records",
        "returned_count": 0,
        "evidence_id": "ev:NVDA:empty",
    }
    for blank in ("", 7, None):
        assert kw._source_outcome("NVDA", tool, {"result": {}, "evidence_id": blank}) == {**head, "status": "ok"}
    failures: list[tuple[Mapping[str, object], dict[str, JSONValue]]] = [
        ({"result": {"error": "HTTP 503"}, "evidence_id": "ev:x"}, {"status": "error", "error": "HTTP 503"}),
        # A record-level error fails the row even beside a mapping result.
        (
            {"result": {"meta": {"returned_count": 4}}, "error": "boom", "evidence_id": "ev:x"},
            {"status": "error", "error": "boom"},
        ),
        (
            {"error": "No data found for NVDA", "error_type": "timeout", "evidence_id": "ev:x"},
            {"status": "timeout", "error": "No data found for NVDA"},
        ),
        (
            {"error": "No data found for NVDA: filings"},
            {"status": "no_records", "error": "No data found for NVDA: filings"},
        ),
        # Frame-only text keeps the raw text rather than an empty error.
        ({"error": "Next step: retry"}, {"status": "error", "error": "Next step: retry"}),
        ({"error": "x" * 400}, {"status": "error", "error": "x" * 300}),
    ]
    for record, expected in failures:
        assert kw._source_outcome("NVDA", tool, record) == {**head, **expected}, record


def test_background_main_on_real_pipe_starts_no_hangup_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-main-thread main over a real pipe answers requests and starts no extra thread."""
    fake = _FakeJev()
    answered = threading.Event()

    def _route(req: Mapping[str, JSONValue], jev: object = None) -> dict[str, JSONValue]:
        answered.set()
        return {"id": kw._request_id(req), "route": "research_required", "personas": None}

    monkeypatch.setattr(kw, "_startup", lambda: fake)
    monkeypatch.setattr(kw, "_shutdown", lambda: None)
    monkeypatch.setattr(kw, "_route", _route)
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    before = set(threading.enumerate())
    worker = threading.Thread(target=kw.main, daemon=True)
    worker.start()
    try:
        os.write(write_fd, json.dumps({"id": "r1", "op": "route", "prompt": "hi"}).encode() + b"\n")
        assert answered.wait(timeout=10)
        assert set(threading.enumerate()) - before == {worker}  # the writer is open: a watcher would still poll
    finally:
        os.close(write_fd)
        worker.join(timeout=10)
        stdin.close()
    assert not worker.is_alive()


_SIGNUMS = st.sampled_from([int(signal.SIGTERM), int(signal.SIGINT)])
_GUARD_STEPS = st.lists(
    st.one_of(
        _SIGNUMS.map(lambda signum: ("signal", [signum])),
        st.lists(_SIGNUMS, max_size=3).map(lambda signums: ("nested", signums)),
        st.just(("thread", [int(signal.SIGTERM)])),
    ),
    max_size=8,
)


def _inside_background_context(check: Callable[[], None]) -> None:
    """Run check on the main thread while a background thread sits inside its own spawn context."""
    entered = threading.Event()
    leave = threading.Event()

    def _background() -> None:
        with child_spawn.registering_child():
            entered.set()
            leave.wait(timeout=10)

    thread = threading.Thread(target=_background)
    thread.start()
    try:
        assert entered.wait(timeout=10)
        check()
    finally:
        leave.set()
        thread.join(timeout=10)


@seed(20261016)
@settings(max_examples=100, derandomize=True, deadline=None)
@given(_GUARD_STEPS)
def test_prop_spawn_guard_replays_only_after_outer_registration(steps: list[tuple[str, list[int]]]) -> None:
    """Held signals replay in arrival order, only after the outer registration, with the guard already clear."""
    registered = False
    expected: list[int] = []
    replayed: list[tuple[int, bool, bool]] = []

    def _raise_signal(signum: int) -> None:
        # The OS signal boundary: record registration state and whether the guard still holds signals.
        replayed.append((signum, registered, child_spawn.defer_termination(signum)))

    def _held(signum: int) -> None:
        assert child_spawn.defer_termination(signum) is True
        expected.append(signum)

    def _not_held() -> None:
        assert child_spawn.defer_termination(signal.SIGTERM) is False

    with mock.patch.object(child_spawn.signal, "raise_signal", _raise_signal):
        _inside_background_context(_not_held)  # a background context creates no shared state
        with child_spawn.registering_child():
            for kind, signums in steps:
                if kind == "nested":
                    with child_spawn.registering_child():
                        for signum in signums:
                            _held(signum)
                elif kind == "thread":
                    _inside_background_context(lambda signum=signums[0]: _held(signum))  # the main context stays
                else:
                    _held(signums[0])
                assert replayed == []  # neither a nested nor a background exit replays
            registered = True
        assert replayed == [(signum, True, False) for signum in expected]
        _not_held()


# Real-subprocess shutdown regressions: the real kernel main, start, and spawn paths over stub providers.
_ROOT = Path(__file__).resolve().parent.parent
_WAIT_S = 60.0

# One stub serves both providers: Needle runs it as a .py server, the decision client runs it as `bun`.
_STUB = """\
import json, os, signal, sys

role = "needle" if sys.argv[0].endswith(".py") else "bun"


def event(text):
    fd = os.open(os.environ["KW_TEST_EVENTS"], os.O_WRONLY)
    os.write(fd, (text + "\\n").encode())
    os.close(fd)


if role == "needle" and os.fork() == 0:
    signal.pause()  # a grandchild in the Needle group: only the group kill ends it
for line in sys.stdin:
    event(role + "-ping")
    if os.environ.get("KW_TEST_HOLD") == role:
        open(os.environ["KW_TEST_RELEASE"]).readline()
    sys.stdout.write(json.dumps({"id": json.loads(line)["id"], "ready": True}) + "\\n")
    sys.stdout.flush()
signal.pause()  # caller EOF never ends a stub: only owner cleanup does
"""

_DRIVER = """\
import os, signal, subprocess, sys, threading
from pathlib import Path

sys.path.insert(0, os.environ["KW_TEST_ROOT"])
import app.needle_client as needle
import app.research.kernel_worker as kw

stub = os.environ["KW_TEST_STUB"]
needle._server = lambda: Path(stub)
needle._python = lambda: sys.executable
kw._prewarm = lambda: None  # no registry or SEC network work


def event(text):
    fd = os.open(os.environ["KW_TEST_EVENTS"], os.O_WRONLY)
    os.write(fd, (text + "\\n").encode())
    os.close(fd)


gate = os.environ.get("KW_TEST_GATE")
if gate:
    signum = int(os.environ["KW_TEST_GATE_SIGNAL"])
    real_popen = subprocess.Popen

    def gated_popen(argv, *args, **kwargs):
        proc = real_popen(argv, *args, **kwargs)
        if (gate == "decision" and argv[0] == "bun") or (gate == "needle" and stub in argv):
            # The signal lands after Popen returns and before the owner registers the child.
            if os.environ.get("KW_TEST_GATE_THREAD"):
                sender = threading.Thread(target=signal.pthread_kill, args=(threading.main_thread().ident, signum))
                sender.start()
                sender.join()
            else:
                signal.raise_signal(signum)
        return proc

    subprocess.Popen = gated_popen

if os.environ.get("KW_TEST_HOLD") == "run":

    def held_run(req, **kwargs):
        event("run-entered")
        open(os.environ["KW_TEST_RELEASE"]).readline()
        return kw._terminal(kw._request_id(req), "provider_error", "released")

    kw._run = held_run

if os.environ.get("KW_TEST_LOST_SIGNAL"):

    def lost_kill(ident, signum):
        # The watcher holds the claim, but its OS signal never arrives.
        event("hangup-claimed")
        threading.Event().wait()

    signal.pthread_kill = lost_kill

if os.environ.get("KW_TEST_EXIT_REPEAT"):

    def armed_prewarm():
        # Needle startup returned, so its child is registered. Exit cleanup repeats SIGTERM before it kills.
        destroy = needle._destroy_proc

        def repeat_destroy(proc):
            signal.raise_signal(signal.SIGTERM)
            destroy(proc)

        needle._destroy_proc = repeat_destroy
        event("exit-cleanup-armed")
        threading.Event().wait()

    kw._prewarm = armed_prewarm


class ReportingThread(threading.Thread):
    # Every driver thread reports its exit through the event FIFO, so tests never read thread names.
    def run(self):
        try:
            super().run()
        finally:
            event("thread-done")


threading.Thread = ReportingThread
kw.main()
"""


class _Child(NamedTuple):
    proc: subprocess.Popen[bytes]
    token: str
    events: int
    release: int
    seen: list[str]


def _has_token(pid: int | str, token: str) -> bool:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return f"KW_TEST_TOKEN={token}".encode() in environ.split(b"\0")


def _token_pids(token: str) -> list[int]:
    """Live processes that inherited this test's token: the kernel tree, whatever its sessions."""
    return [
        int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit() and _has_token(entry.name, token)
    ]


@contextlib.contextmanager
def _kernel_child(tmp_path: Path, stdin: int = subprocess.PIPE, **knobs: str) -> Generator[_Child]:
    """Run the real kernel main over stub providers. Every exit kills the whole test tree."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = tmp_path / "needle_server.py"
    stub.write_text(_STUB)
    bun = bin_dir / "bun"
    bun.write_text(f"#!{sys.executable}\n{_STUB}")
    bun.chmod(0o755)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    os.mkfifo(tmp_path / "events")
    os.mkfifo(tmp_path / "release")
    # An O_RDWR FIFO end opens at once and never reports EOF, so no stub open ever blocks.
    events = os.open(tmp_path / "events", os.O_RDWR | os.O_NONBLOCK)
    release = os.open(tmp_path / "release", os.O_RDWR)
    token = uuid.uuid4().hex
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "STOCKBOT_DATA_DIR": str(tmp_path / "data"),
        "KW_TEST_ROOT": str(_ROOT),
        "KW_TEST_STUB": str(stub),
        "KW_TEST_EVENTS": str(tmp_path / "events"),
        "KW_TEST_RELEASE": str(tmp_path / "release"),
        "KW_TEST_TOKEN": token,
        **knobs,
    }
    proc = subprocess.Popen(
        [sys.executable, str(driver)], stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    try:
        yield _Child(proc, token, events, release, [])
    finally:
        proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        deadline = time.monotonic() + _WAIT_S
        # A descendant can fork during cleanup, so rescan until no live match remains.
        while (pids := _token_pids(token)) and time.monotonic() < deadline:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    pidfd = os.pidfd_open(pid)
                    try:
                        # The pidfd pins one process: a PID reused after the open fails this token read.
                        if _has_token(pid, token):
                            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                            poller = select.poll()
                            poller.register(pidfd, select.POLLIN)
                            poller.poll(max(0.0, deadline - time.monotonic()) * 1000)
                    finally:
                        os.close(pidfd)
        os.close(events)
        os.close(release)
        assert pids == [], f"token processes survived cleanup: {pids}"


def _await_event(child: _Child, name: str) -> None:
    """Block on the stub event FIFO until name arrives."""
    poller = select.poll()
    poller.register(child.events, select.POLLIN)
    deadline = time.monotonic() + _WAIT_S
    while name not in child.seen:
        left_ms = (deadline - time.monotonic()) * 1000
        if left_ms <= 0 or not poller.poll(left_ms):
            pytest.fail(f"no {name} event; saw {child.seen}")
        child.seen.extend(os.read(child.events, 4096).decode().split())


def _finish(child: _Child) -> tuple[list[dict[str, object]], str]:
    """Wait for the kernel to exit: (stdout rows, stderr text)."""
    # Keep stdin open until the kernel exits. communicate() closes stdin.
    child.proc.wait(timeout=_WAIT_S)
    if child.proc.stdin is not None and child.proc.stdin.closed:
        child.proc.stdin = None
    out, err = child.proc.communicate(timeout=_WAIT_S)
    rows: list[dict[str, object]] = []
    for line in out.decode().splitlines():
        row: object = json.loads(line)
        assert isinstance(row, dict)
        rows.append(row)
    return rows, err.decode()


def _assert_tree_gone(child: _Child) -> None:
    """Every process with this test's token exits. A pidfd wait replaces a fixed sleep."""
    survivors: list[int] = []
    for pid in _token_pids(child.token):
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            if not poller.poll(5000):
                survivors.append(pid)
        finally:
            os.close(pidfd)
    assert survivors == [], f"owned descendants outlived the kernel: {survivors}"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_startup_signal_under_needle_lock_exits_and_reaps(tmp_path: Path, signum: int) -> None:
    """A signal while startup waits under the Needle lock exits 128+signum and reaps every child."""
    with _kernel_child(tmp_path, KW_TEST_HOLD="needle") as child:
        _await_event(child, "needle-ping")
        child.proc.send_signal(signum)
        rows, err = _finish(child)
        assert child.proc.returncode == 128 + signum, err
        assert rows == []
        _assert_tree_gone(child)


@pytest.mark.parametrize(
    ("gate", "signum", "thread"),
    [
        ("needle", signal.SIGTERM, False),
        ("needle", signal.SIGINT, True),
        ("decision", signal.SIGINT, False),
        ("decision", signal.SIGTERM, True),
    ],
)
def test_signal_inside_spawn_window_reaps_new_child(tmp_path: Path, gate: str, signum: int, thread: bool) -> None:
    """A signal between Popen and owner registration replays after registration, so cleanup sees the child."""
    knobs = {"KW_TEST_GATE": gate, "KW_TEST_GATE_SIGNAL": str(signum)}
    if thread:
        knobs["KW_TEST_GATE_THREAD"] = "1"
    with _kernel_child(tmp_path, **knobs) as child:
        rows, err = _finish(child)
        assert child.proc.returncode == 128 + signum, err
        assert rows == []
        _assert_tree_gone(child)


@pytest.mark.parametrize("provider", ["bun", "needle"])
def test_caller_pipe_eof_during_startup_exits_and_reaps(tmp_path: Path, provider: str) -> None:
    """Caller EOF cancels a pending provider handshake and reaps every owned process."""
    with _kernel_child(tmp_path, KW_TEST_HOLD=provider) as child:
        _await_event(child, f"{provider}-ping")
        assert child.proc.stdin is not None
        child.proc.stdin.close()
        rows, err = _finish(child)
        assert child.proc.returncode == 128 + signal.SIGTERM, err
        assert rows == []
        _assert_tree_gone(child)


def test_connected_caller_reaches_ready_and_closes_cleanly(tmp_path: Path) -> None:
    """A connected caller completes startup, receives an answer, and closes without cancellation."""
    with _kernel_child(tmp_path, KW_TEST_HOLD="needle") as child:
        _await_event(child, "needle-ping")
        assert child.proc.stdin is not None
        assert child.proc.stdout is not None
        assert child.proc.poll() is None
        os.write(child.release, b"go\n")
        assert select.select([child.proc.stdout], [], [], _WAIT_S)[0], "worker did not become ready"
        assert json.loads(child.proc.stdout.readline()) == {"type": "ready"}
        assert child.proc.poll() is None
        child.proc.stdin.write(json.dumps({"id": "connected", "op": "nope"}).encode() + b"\n")
        child.proc.stdin.close()
        rows, err = _finish(child)
        assert child.proc.returncode == 0, err
        assert len(rows) == 1
        assert rows[0]["id"] == "connected"
        terminal = rows[0]["terminal"]
        assert isinstance(terminal, dict)
        assert terminal["category"] == "invalid_params"
        _assert_tree_gone(child)


def test_caller_socket_half_close_during_startup_exits_and_reaps(tmp_path: Path) -> None:
    """A Unix socket caller that shuts down its write side during startup ends startup."""
    caller, kernel_end = socket.socketpair()
    try:
        with _kernel_child(tmp_path, stdin=kernel_end.fileno(), KW_TEST_HOLD="needle") as child:
            kernel_end.close()
            _await_event(child, "needle-ping")
            caller.shutdown(socket.SHUT_WR)
            rows, err = _finish(child)
            assert child.proc.returncode == 128 + signal.SIGTERM, err
            assert rows == []
            _assert_tree_gone(child)
    finally:
        caller.close()
        kernel_end.close()


def test_queued_request_at_caller_eof_is_answered(tmp_path: Path) -> None:
    """A request queued before the hangup keeps startup running: ready, the answer, then a clean exit."""
    with _kernel_child(tmp_path, KW_TEST_HOLD="needle") as child:
        _await_event(child, "needle-ping")
        assert child.proc.stdin is not None
        child.proc.stdin.write(json.dumps({"id": "q1", "op": "nope"}).encode() + b"\n")
        child.proc.stdin.close()
        _await_event(child, "thread-done")  # the watcher decided with the request still queued
        os.write(child.release, b"go\n")
        rows, err = _finish(child)
        assert child.proc.returncode == 0, err
        assert rows[0] == {"type": "ready"}
        assert [row.get("id") for row in rows[1:]] == ["q1"]
        _assert_tree_gone(child)


def test_caller_eof_after_ready_keeps_normal_shutdown(tmp_path: Path) -> None:
    """After readiness the watcher never signals: an in-flight run answers, then normal shutdown reaps."""
    with _kernel_child(tmp_path, KW_TEST_HOLD="run") as child:
        assert child.proc.stdin is not None
        child.proc.stdin.write(json.dumps({"id": "run1", "op": "run", "prompt": "x"}).encode() + b"\n")
        child.proc.stdin.flush()
        _await_event(child, "run-entered")
        child.proc.stdin.close()
        _await_event(child, "thread-done")  # the watcher decided after readiness
        os.write(child.release, b"go\n")
        rows, err = _finish(child)
        assert child.proc.returncode == 0, err
        assert rows[0] == {"type": "ready"}
        assert [row.get("id") for row in rows[1:]] == ["run1"]
        _assert_tree_gone(child)


def test_watcher_claim_before_ready_exits_from_main(tmp_path: Path) -> None:
    """The watcher claims the hangup before readiness but its signal waits: main exits 143 and reaps."""
    with _kernel_child(tmp_path, KW_TEST_HOLD="needle", KW_TEST_LOST_SIGNAL="1") as child:
        _await_event(child, "needle-ping")
        assert child.proc.stdin is not None
        child.proc.stdin.close()
        _await_event(child, "hangup-claimed")
        os.write(child.release, b"go\n")
        child.proc.wait(timeout=10)  # a blocking claim never returns
        rows, err = _finish(child)
        assert child.proc.returncode == 128 + signal.SIGTERM, err
        assert rows == []
        _assert_tree_gone(child)


def test_repeated_signal_during_exit_cleanup_still_reaps(tmp_path: Path) -> None:
    """A second SIGTERM inside owned-child cleanup never interrupts that cleanup."""
    with _kernel_child(tmp_path, KW_TEST_EXIT_REPEAT="1") as child:
        _await_event(child, "exit-cleanup-armed")
        child.proc.send_signal(signal.SIGTERM)
        rows, err = _finish(child)
        assert child.proc.returncode == 128 + signal.SIGTERM, err
        assert rows == []
        _assert_tree_gone(child)


def test_spawn_guard_replays_and_clears_after_spawn_failure() -> None:
    """A failed spawn still replays cancellation and clears the guard for the next caller."""
    replayed: list[int] = []
    with mock.patch.object(child_spawn.signal, "raise_signal", replayed.append):
        with pytest.raises(OSError, match="spawn failed"), child_spawn.registering_child():
            assert child_spawn.defer_termination(signal.SIGTERM) is True
            raise OSError("spawn failed")
        assert replayed == [signal.SIGTERM]
        assert child_spawn.defer_termination(signal.SIGINT) is False
