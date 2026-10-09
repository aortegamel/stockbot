"""Focused kernel-worker persistence test: shared JEV identity + ready-first main."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping
from typing import Never, override

import pytest

import app.research.kernel_worker as kw
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
        return {"id": raw if isinstance(raw, str) else "?", "route": "research_required"}

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
