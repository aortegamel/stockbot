"""Fail-closed guard contracts: failures propagate, envelopes preserved."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import override
from unittest import mock

import pytest

from app.decision_client import JevClient
from app.research import kernel_worker as kw
from app.research import scheduler as sched
from app.research.models import JSONValue
from app.research.repository import ResearchRepository


def test_research_registry_excludes_portfolio_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live research registry exposes no portfolio tool, even with the broker enabled."""
    from app.tools import PORTFOLIO_AUTHORIZED_TOOLS

    monkeypatch.setenv("BROKER_ENABLED", "true")
    names = {entry["name"] for entry in sched.build_registry()}
    assert PORTFOLIO_AUTHORIZED_TOOLS
    assert not names & PORTFOLIO_AUTHORIZED_TOOLS
    assert kw._registry_portfolio_hit() == []
    leaked = [*sched.build_registry(), {"name": "get_portfolio_snapshot"}]
    with mock.patch("app.research.scheduler.build_registry", return_value=leaked):
        assert kw._registry_portfolio_hit() == ["get_portfolio_snapshot"]


def test_arguments_context_carries_ticker_without_raw_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw evidence/attempts never cross; the ticker rides beside grounded scalar args."""
    gen = _FailGenerate(
        {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "confidence": 0.8, "reasoning": "r"}
    )
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    out = kw._arguments(
        {
            "id": "a1",
            "op": "arguments",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "ticker": "NVDA",
            "context": {
                "evidence": [{"evidence_id": "ev-1", "content": "raw bytes"}],
                "attempts": [{"tool": "list_sec_filings"}],
                "grounding_hint": "do not send",
                "as_of": None,
                "temporal_scope": {"mode": "latest-available"},
                "today_utc": "2026-10-12",
                "ticker": "NVDA",
            },
            "schema": {"type": "object"},
        }
    )
    assert out["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    seen = gen.seen[0]
    assert isinstance(seen, dict)
    context = seen.get("context")
    assert isinstance(context, dict)
    assert context.get("ticker") == "NVDA"
    assert "evidence" not in context and "attempts" not in context and "grounding_hint" not in context
    grounded = context.get("grounded_arguments")
    assert isinstance(grounded, dict) and grounded.get("identifier") == "NVDA"


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("can you give me buying advice", "NVDA"),
        ("Should I buy Apple?", "AAPL"),
        ("Should I buy MissingCo?", "NVDA"),
    ],
)
def test_arguments_context_and_withhold_preserve_subject_precedence(
    monkeypatch: pytest.MonkeyPatch, objective: str, expected: str
) -> None:
    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL", "NVDA": "NVDA"}.get(name)

    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _resolve)
    generated = _FailGenerate(None)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: generated)
    out = kw._arguments({"id": "context", "tool": "list_sec_filings", "objective": objective, "ticker": "NVDA"})
    payload = generated.seen[0]
    assert isinstance(payload, dict)
    context = payload["context"]
    assert isinstance(context, dict)
    assert context.get("ticker") == expected
    grounded = context["grounded_arguments"]
    assert isinstance(grounded, dict)
    assert grounded["identifier"] == expected
    arguments = out["arguments"]
    assert isinstance(arguments, dict) and arguments["identifier"] == expected


class _FailGenerate:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.seen: list[object] = []

    def __call__(self, **kwargs: object) -> object:
        self.seen.append(kwargs)
        return self.payload


@pytest.mark.parametrize("schema_source", ["registry", "empty", "inline"])
def test_arguments_uses_the_caller_schema_or_registered_schema(
    monkeypatch: pytest.MonkeyPatch, schema_source: str
) -> None:
    registered: dict[str, JSONValue] = {
        "type": "object",
        "title": "registered",
        "properties": {"ticker": {"type": "string"}},
    }
    inline: dict[str, JSONValue] = {"type": "object", "title": "caller", "properties": {"ticker": {"type": "string"}}}
    generated = _FailGenerate(
        {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "grounded"}
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: generated)
    monkeypatch.setattr(sched, "build_registry", lambda: [{"name": "query_finra", "parameters": registered}])
    request: dict[str, JSONValue] = {"id": "schema", "tool": "query_finra", "objective": "short interest?"}
    if schema_source != "registry":
        request["schema"] = inline if schema_source == "inline" else dict[str, JSONValue]()
    out = kw._direct_op({"op": "arguments", **request}, None)
    assert out is not None
    assert out["id"] == "schema" and out["tool"] == "query_finra"
    assert out["arguments"] == {"ticker": "NVDA", "dataset": "consolidatedShortInterest"}
    assert out["confidence"] == 0.9 and out["reasoning"] == "grounded"
    assert len(generated.seen) == 1
    payload = generated.seen[0]
    assert isinstance(payload, dict)
    assert payload["tool"] == "query_finra"
    assert payload["schema"] == (inline if schema_source == "inline" else registered)
    assert payload["objective"] == "short interest?"


def test_arguments_mismatch_is_error_never_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kw,
        "_shared_needle_generate",
        lambda: _FailGenerate({"tool": "other", "arguments": {}}),
    )
    out = kw._arguments({"id": "a2", "op": "arguments", "tool": "query_finra"})
    assert out["id"] == "a2" and "error" in out
    assert "error" in kw._arguments({"id": "a3"})


def test_arguments_withhold_seeds_sec_and_sho(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine Needle withhold (returns None) seeds SEC/SHO; unseedable stays error."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA", "Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _withhold(**kwargs: object) -> object:
        return None

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _withhold)
    sho = kw._arguments(
        {
            "id": "s1",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}
    sec = kw._arguments(
        {
            "id": "s2",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
        }
    )
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    none = kw._arguments(
        {
            "id": "s3",
            "tool": "get_reg_sho_volume",
            "objective": "Which filings mention Elon Musk?",
        }
    )
    assert none["id"] == "s3" and "error" in none


def test_arguments_generation_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising Needle generate reaches the outer error envelope, never seeded success."""

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("needle boom")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments(
        {
            "id": "s9",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
        }
    )
    assert out["id"] == "s9" and "error" in out
    assert "needle boom" in str(out["error"])
    assert "arguments" not in out


def test_arguments_schema_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _FailGenerate(
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "confidence": 0.9,
            "reasoning": "r",
        }
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)

    def _bad_schema(name: str, reg: object) -> object:
        raise RuntimeError("schema boom")

    monkeypatch.setattr("app.research.scheduler._schema_for", _bad_schema)
    out = kw._arguments({"id": "a9", "op": "arguments", "tool": "query_finra"})
    assert out["id"] == "a9" and "error" in out
    assert "schema boom" in str(out["error"])


def test_arguments_repair_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _FailGenerate(
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "confidence": 0.9,
            "reasoning": "r",
        }
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    monkeypatch.setattr(
        "app.research.scheduler.build_registry",
        lambda: [{"name": "query_finra", "parameters": {"type": "object"}}],
    )

    def _bad_repair(*a: object, **k: object) -> object:
        raise RuntimeError("repair boom")

    monkeypatch.setattr("app.research.scheduler._repair_tool_arguments", _bad_repair)
    out = kw._arguments(
        {
            "id": "a10",
            "tool": "query_finra",
            "objective": "q?",
            "schema": {"type": "object"},
        }
    )
    assert out["id"] == "a10" and "error" in out
    assert "repair boom" in str(out["error"])


def test_arguments_repairs_needle_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Org ticker repairs through the shared scheduler path; generated forms stay verbatim for tool rejection."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _gen(**kwargs: object) -> object:
        tool = kwargs.get("tool")
        if tool == "list_sec_filings":
            return {
                "tool": tool,
                "arguments": {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]},
                "confidence": 0.8,
                "reasoning": "r",
            }
        return {
            "tool": tool,
            "arguments": {"ticker": "FINRA", "company_name": "Apple"},
            "confidence": 0.8,
            "reasoning": "r",
        }

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)
    sec = kw._arguments(
        {
            "id": "r1",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "schema": {"type": "object"},
        }
    )
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]}
    sho = kw._arguments(
        {
            "id": "r2",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
            "schema": {"type": "object"},
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}


def test_request_call_reuses_original_deadline_and_cancels_only_expired_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tests.research_source_seam as seam

    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-1": "NVIDIA supplies accelerated computing systems."})
    store = ResearchRepository()
    evidence = store.list_evidence(sid)
    import time as _real_time

    clock = [_real_time.time()]
    monkeypatch.setattr(kw.time, "time", lambda: clock[0])
    events: list[str] = []

    async def _service(wait: bool) -> str:
        events.append("started")
        if wait:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                events.append("cancelled")
                raise
        events.append("completed")
        return "ready"

    request: dict[str, JSONValue] = {"id": "deadline", "deadlineAt": int(clock[0] * 1000) + 10_000}
    assert kw._request_call(request, "first", _service(False)) == "ready"
    clock[0] = clock[0] + 20.0
    with pytest.raises(TimeoutError, match="deadline reached during second"):
        kw._request_call(request, "second", _service(True))
    assert events == ["started", "completed", "started", "cancelled"]
    assert store.list_evidence(sid) == evidence
    assert kw._request_call({"id": "next"}, "next", _service(False)) == "ready"
    assert events[-2:] == ["started", "completed"]


def test_startup_jev_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadJev(JevClient):
        @override
        def start(self) -> None:
            raise RuntimeError("jev start down")

    monkeypatch.setattr(kw, "_shared_jev", lambda: _BadJev())
    monkeypatch.setattr(kw, "_prewarm", lambda: None)
    with pytest.raises(RuntimeError, match="jev start down"):
        kw._startup()


def test_startup_needle_failure_stays_live(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    jev = JevClient()

    def _noop_start(self: JevClient) -> None:
        return None

    monkeypatch.setattr(kw, "_shared_jev", lambda: jev)
    monkeypatch.setattr(JevClient, "start", _noop_start)
    monkeypatch.setattr(kw, "_prewarm", lambda: None)
    real_import = importlib.import_module

    def _bad_needle(name: str, package: str | None = None) -> ModuleType:
        if name == "app.needle_client":
            raise RuntimeError("needle down")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _bad_needle)
    assert kw._startup() is jev


@pytest.mark.parametrize("stage", ["repository", "list", "cancel"])
def test_close_bootstrap_job_failure_logs(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.research import service
    from app.research.models import Job

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    failure = RuntimeError(f"{stage} down")
    job = Job(
        job_id="j1",
        session_id="rs:x",
        wave_id=1,
        parent_job_id=None,
        job_type="source_agent",
        owner="k",
    )

    def _one_job(self: ResearchRepository, sid: str) -> list[Job]:
        return [job]

    if stage == "repository":
        target = "app.research.repository.ResearchRepository"
    elif stage == "list":
        target = "app.research.repository.ResearchRepository.list_jobs"
    else:
        monkeypatch.setattr(ResearchRepository, "list_jobs", _one_job)
        target = "app.research.service.cancel_job"
    with mock.patch(target, side_effect=failure) as cleanup:
        kw._close_bootstrap_job("rs:x")
    assert cleanup.call_count == 1
    records = [record for record in caplog.records if "bootstrap cleanup failed" in record.message]
    assert len(records) == 1
    assert "sid=rs:x" in records[0].message
    assert records[0].exc_info is not None and records[0].exc_info[1] is failure

    if stage == "cancel":
        done_source = replace(job, job_id="j0", status="completed")
        open_other = replace(job, job_id="j2", job_type="stockbot")
        with (
            mock.patch.object(
                ResearchRepository,
                "list_jobs",
                return_value=[open_other, done_source, job],
            ),
            mock.patch.object(service, "cancel_job") as cancel,
        ):
            kw._close_bootstrap_job("rs:x")
        cancel.assert_called_once_with("j1")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_node_creation_closes_bootstrap_without_replacing_setup_error(
    cleanup_fails: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))

    def _no_ticker(objective: str, supplied_ticker: object = None) -> None:
        return None

    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr(sched, "_objective_subject_ticker", _no_ticker)
    repo = ResearchRepository()
    failure = RuntimeError("node creation down")
    cleanup_failure = RuntimeError("cancel down")
    sessions: list[str] = []
    cancellations: list[str] = []
    original_cancel = service.cancel_job

    def _session(stage: str, detail: dict[str, object]) -> None:
        if stage == "session":
            sid = detail["session_id"]
            assert isinstance(sid, str)
            sessions.append(sid)
            assert [job.status for job in repo.list_jobs(sid)] == ["running"]

    def _failed_node(*args: object, **kwargs: object) -> object:
        raise failure

    def _cancel(job_id: str) -> dict[str, JSONValue]:
        cancellations.append(job_id)
        if cleanup_fails:
            raise cleanup_failure
        return original_cancel(job_id)

    monkeypatch.setattr(service, "create_node", _failed_node)
    monkeypatch.setattr(service, "cancel_job", _cancel)
    with pytest.raises(RuntimeError, match="node creation down") as raised:
        kw.run_graph_prompt("q?", jev=JevClient(), progress=_session)
    assert raised.value is failure
    assert len(sessions) == 1
    sid = sessions[0]
    session = repo.get_session(sid)
    assert session.query == "q?"
    jobs = repo.list_jobs(sid)
    assert len(jobs) == 1
    assert cancellations == [jobs[0].job_id]
    assert jobs[0].status == ("running" if cleanup_fails else "cancelled")
    if not cleanup_fails:
        assert not [job for job in jobs if job.status in ("queued", "running")]
    cleanup_records = [record for record in caplog.records if "bootstrap cleanup failed" in record.message]
    assert len(cleanup_records) == int(cleanup_fails)
    if cleanup_fails:
        assert cleanup_records[0].exc_info is not None
        assert cleanup_records[0].exc_info[1] is cleanup_failure


@pytest.mark.parametrize(
    ("line", "expected_message"),
    [
        ("{oops", "invalid JSON"),
        ("[1, 2]", f"op must be one of {', '.join(kw._KERNEL_OPS)}"),
        ('{"op": "nope"}', f"op must be one of {', '.join(kw._KERNEL_OPS)}"),
    ],
)
def test_request_line_reports_bad_input(line: str, expected_message: str) -> None:
    """Invalid JSON, a non-mapping request, and an unknown op all answer invalid_params."""
    req, response = kw._request_or_response(line)
    assert req is None
    assert response is not None and response["id"] == "?"
    assert response["terminal"] == {"category": "invalid_params", "message": expected_message}


def test_request_line_keeps_known_id_and_blank_reports_nothing() -> None:
    """A blank line is answered with nothing; an unknown op keeps the caller's own id."""
    assert kw._request_or_response("  \n") == (None, None)
    req, response = kw._request_or_response('{"op": "nope", "id": "keep-me"}')
    assert req is None
    assert response is not None and response["id"] == "keep-me"
    valid, response = kw._request_or_response('{"op": "arguments", "id": "r1", "tool": "add_widget"}')
    assert response is None
    assert valid == {"op": "arguments", "id": "r1", "tool": "add_widget"}


def test_direct_op_failure_is_worker_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising direct op answers with the worker failure terminal, never a raise."""

    def _boom(*a: object, **k: object) -> dict[str, JSONValue]:
        raise RuntimeError("op blew up")

    monkeypatch.setattr(kw, "_arguments", _boom)
    out = kw._direct_op({"id": "d2", "op": "arguments"}, None)
    assert out["id"] == "d2"
    assert out["error"] == "worker failed: op blew up"
    assert out["terminal"] == {"category": "provider_error", "message": out["error"]}


def test_configure_logging_maps_level_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """STOCKBOT_LOG_LEVEL selects the level; an unknown name falls back to WARNING."""
    seen: list[dict[str, object]] = []

    def _basic_config(**kwargs: object) -> None:
        seen.append(dict(kwargs))

    monkeypatch.setattr(kw.logging, "basicConfig", _basic_config)
    monkeypatch.setenv("STOCKBOT_LOG_LEVEL", "debug")
    kw._configure_logging()
    monkeypatch.setenv("STOCKBOT_LOG_LEVEL", "chatty")
    kw._configure_logging()
    assert [row["level"] for row in seen] == [kw.logging.DEBUG, kw.logging.WARNING]


def test_prewarm_does_not_block_requests_when_sources_fail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import threading

    import app.sec.client as sec_client

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _registry() -> frozenset[str]:
        entered.set()
        if not release.wait(2):
            raise AssertionError("test did not release prewarm")
        raise RuntimeError("registry down")

    class _WarmThread(threading.Thread):
        @override
        def run(self) -> None:
            try:
                assert self.daemon
                super().run()
            finally:
                finished.set()

    def _lookup(key: str) -> None:
        raise RuntimeError("lookup down")

    monkeypatch.setattr(kw.threading, "Thread", _WarmThread)

    monkeypatch.setattr("app.research.tool_runner.research_tool_names", _registry)
    monkeypatch.setattr(sec_client, "ensure_identity", lambda: None)
    monkeypatch.setattr(sec_client, "_fetch_lookup_frame", _lookup)
    with caplog.at_level(logging.DEBUG, logger=kw.__name__):
        kw._prewarm()
        try:
            assert entered.wait(1)
            assert not release.is_set()
            assert kw._direct_op({"id": "warm", "op": "arguments"}, None) == {
                "id": "warm",
                "error": "tool required",
            }
        finally:
            release.set()
            assert finished.wait(1)
    errors = [record.exc_info[1] for record in caplog.records if record.exc_info is not None]
    assert any(isinstance(error, RuntimeError) and str(error) == "registry down" for error in errors)
    assert any(isinstance(error, RuntimeError) and str(error) == "lookup down" for error in errors)


def test_shutdown_closes_jev_needle_and_pools_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown is idempotent; a JEV close that raises never blocks the remaining closes."""
    import sys as _sys
    from types import SimpleNamespace

    closed: list[str] = []

    class _Jev:
        def close(self) -> None:
            closed.append("jev")
            raise OSError("already gone")

    def _pools() -> None:
        closed.append("pools")

    def _needle_close() -> None:
        closed.append("needle")

    monkeypatch.setattr(kw, "_JEV", _Jev())
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    monkeypatch.setattr(kw, "_shutdown_pools", _pools)
    monkeypatch.setitem(_sys.modules, "app.needle_client", SimpleNamespace(close=_needle_close))
    kw._shutdown()
    kw._shutdown()
    assert sorted(closed) == ["jev", "needle", "pools"]


def test_shutdown_survives_missing_needle_and_dead_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Needle module or a pool without shutdown never breaks worker exit."""
    import sys as _sys

    from app.research import scheduler as _sched

    calls: list[str] = []
    real_pools = kw._shutdown_pools

    def _counting_pools() -> None:
        calls.append("pools")
        real_pools()

    monkeypatch.setattr(kw, "_JEV", None)
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    monkeypatch.setattr(kw, "_shutdown_pools", _counting_pools)
    monkeypatch.setitem(_sys.modules, "app.needle_client", None)
    monkeypatch.setattr(_sched, "_TOOL_POOL", object())
    monkeypatch.setattr("app.research.tool_runner._SEC_POOL", object())
    monkeypatch.setattr("app.research.tool_runner._TOOL_POOL", object())
    kw._shutdown()
    kw._shutdown()
    assert calls == ["pools"]  # the second call returns before touching anything


def test_arguments_requires_tool_before_anything_else() -> None:
    """A blank or missing tool is denied without touching Needle."""
    for req in ({"id": "a0"}, {"id": "a0", "tool": ""}, {"id": "a0", "tool": 7}, {"id": "a0", "objective": "q?"}):
        assert kw._arguments(req) == {"id": "a0", "error": "tool required"}


@pytest.mark.parametrize("tool", ["research_resume", "research_status", "research_cancel"])
def test_arguments_session_carry_skips_needle(monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    """resume/status/cancel carry the caller's session id without grounding or invoking the tool."""

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run for a sid carry")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments({"id": "s1", "tool": tool, "arguments": {"session_id": " rs:1 "}})
    assert out["id"] == "s1" and out["tool"] == tool
    assert out["arguments"] == {"session_id": "rs:1"}
    assert out["confidence"] == 1.0


def test_arguments_read_search_carries_ids_or_denies() -> None:
    """read_search carries session_id+search_id from node and arguments, and denies a missing search_id."""
    carried = kw._arguments(
        {
            "id": "r1",
            "tool": "research_read_search",
            "node": {"session_id": "rs:9"},
            "arguments": {"search_id": " s1 "},
        }
    )
    assert carried["id"] == "r1" and carried["tool"] == "research_read_search"
    assert carried["arguments"] == {"session_id": "rs:9", "search_id": "s1"}
    assert carried["confidence"] == 1.0
    denied = kw._arguments({"id": "r2", "tool": "research_read_search", "node": {"session_id": "rs:9"}})
    assert denied == {
        "id": "r2",
        "error": "research_read_search needs search_id from a prior search_sec_filings result",
    }


def test_arguments_session_id_falls_back_to_node() -> None:
    """A blank arguments.session_id falls back to node.session_id."""
    carried = kw._arguments(
        {"id": "r3", "tool": "research_status", "arguments": {"session_id": " "}, "node": {"session_id": "rs:5"}}
    )
    assert carried["arguments"] == {"session_id": "rs:5"}


def test_arguments_bad_ticker_is_answered_without_fail_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed ticker channel is its own envelope: the call never reaches the fail-open boundary."""
    import logging

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run with a malformed ticker")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    with caplog.at_level(logging.WARNING, logger=kw.__name__):
        out = kw._arguments({"id": "t1", "tool": "list_sec_filings", "objective": "q?", "ticker": 7})
    assert out["id"] == "t1" and "ticker" in str(out["error"])
    assert caplog.records == []


def _needle_withhold(**kwargs: object) -> None:
    """A genuine Needle withhold: generate() answers None instead of a tool call."""


def test_arguments_result_drops_non_numeric_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-mapping generate result carries no confidence or reasoning; a boolean confidence reads as None."""

    def _repair(*a: object, **k: object) -> dict[str, object]:
        return {"ticker": "NVDA"}

    def _not_a_mapping(**kwargs: object) -> object:
        return "not-a-mapping"

    monkeypatch.setattr("app.research.scheduler._repair_tool_arguments", _repair)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _not_a_mapping)
    assert kw._arguments({"id": "n1", "tool": "query_finra", "objective": "NVDA?"}) == {
        "id": "n1",
        "tool": "query_finra",
        "arguments": {"ticker": "NVDA"},
    }
    generated: dict[str, object] = {"tool": "query_finra", "arguments": {}, "confidence": True, "reasoning": 7}

    def _boolean_confidence(**kwargs: object) -> dict[str, object]:
        return generated

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boolean_confidence)
    out = kw._arguments({"id": "n2", "tool": "query_finra", "objective": "NVDA?"})
    assert out["confidence"] is None and out["reasoning"] == ""


@pytest.mark.parametrize(
    ("text", "qualified"),
    [
        ("The filings reviewed do not establish termination triggers.", True),
        ("The filings did not establish termination triggers.", True),
        ("The filings establish termination triggers.", False),
    ],
)
def test_unknown_qualifier_reads_do_not_establish(text: str, qualified: bool) -> None:
    from app.research.service import _QUALIFIERS

    assert bool(_QUALIFIERS["unknown"].search(text)) is qualified


def test_plural_form_names_are_never_numbers() -> None:
    """ "their latest 10-Qs" names a form, so its 10 never needs a datapoint."""
    from app.research.service import _masked, _numbers

    assert _numbers(_masked("Both latest 10-Qs and 8-K/A filings show growth of 12 percent.")) == [("12", 51, 53)]


@pytest.mark.parametrize("wave,frozen", [(3, False), (0, True)])
def test_admit_datapoint_uses_live_wave(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wave: int, frozen: bool
) -> None:
    import tests.research_source_seam as seam

    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-prose": "NVDA FY2025 Revenue was 46.7."})
    store = ResearchRepository()
    session = store.get_session(sid)
    store.save_session(replace(session, current_wave=wave, freeze_ids=[f"{sid}:1:freeze"] if frozen else []))
    result = kw._direct_op(
        {
            "id": "wave",
            "op": "admit_datapoint",
            "sessionId": sid,
            "evidenceId": "ev-prose",
            "metric": "Revenue",
            "value": "46.7",
            "entity": "NVDA",
            "period": "FY2025",
        },
        None,
    )
    assert "error" not in result, result
    eid = result["evidence_id"]
    assert isinstance(eid, str)
    assert store.get_evidence(eid)["wave_id"] == (2 if frozen else wave)


def test_admit_datapoint_drops_a_declared_currency_symbol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A table puts "$" in its own cell: a "$96,221" declaration proves the "96,221" token."""
    import tests.research_source_seam as seam

    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-table": "Income statement\nRevenue | $ | 96,221"})
    result = kw._direct_op(
        {
            "id": "usd",
            "op": "admit_datapoint",
            "sessionId": sid,
            "evidenceId": "ev-table",
            "metric": "Revenue",
            "value": "$96,221",
            "entity": "Income statement",
            "period": "Income statement",
        },
        None,
    )
    point = result.get("datapoint")
    assert isinstance(point, dict) and point.get("value") == "96,221", result


def test_admit_datapoint_recomputes_a_declared_growth_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A growth declaration over two session dp: ids admits only the ledger's own recompute."""
    import tests.research_source_seam as seam
    from app.research import service

    table = (
        "NVIDIA Corporation and Subsidiaries\n(In millions)\nThree Months Ended | Six Months Ended\n"
        "Jul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025\n"
        "Revenue | $ | 96,221 | $ | 46,743 | $ | 177,837 | $ | 90,805\n"
        "Cost of revenue | $ | 26,000 | $ | 12,000 | $ | 50,000 | $ | 24,000"
    )
    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-table": table})
    base = {"op": "admit_datapoint", "sessionId": sid, "evidenceId": "ev-table", "metric": "Revenue"}
    base |= {"entity": "NVIDIA Corporation", "unit": "millions"}
    ids = []
    for value, date in (("96,221", "Jul 26, 2026"), ("46,743", "Jul 27, 2025")):
        result = kw._direct_op({**base, "id": value, "value": value, "period": f"Three Months Ended {date}"}, None)
        point = result.get("datapoint")
        assert isinstance(point, dict), result
        ids.append(point["id"])
    period = "Three Months Ended Jul 26, 2026 versus Three Months Ended Jul 27, 2025"
    growth = {**base, "id": "g", "metric": "Revenue growth", "unit": "%", "period": period}
    growth |= {"formula": "percent_change@2", "inputs": ids}
    result = kw._direct_op({**growth, "value": "105.9"}, None)
    point = result.get("datapoint")
    assert isinstance(point, dict) and point["inputs"] == ids, result
    claim = f"NVIDIA Corporation reported Revenue growth of 105.9% for {period}."
    assert service._binding_defect(claim, [point], "calculation") is None
    assert service._wording_defect(claim, [point]) is None
    # "the" before each period reads naturally and binds the same periods; a missing prior period still fails.
    natural = claim.replace("for Three", "for the Three").replace("versus Three", "versus the Three")
    assert service._binding_defect(natural, [point], "calculation") is None
    assert service._binding_defect(natural.split(" versus")[0] + ".", [point], "calculation") is not None
    for bad in ({"value": "105.8"}, {"value": "105.9", "inputs": ids[::-1]}, {"value": "105.9", "inputs": ids[:1]}):
        assert "error" in kw._direct_op({**growth, **bad}, None), bad
    assert "not datapoints" in str(
        kw._direct_op({**growth, "value": "105.9", "inputs": ["dp:x", ids[1]]}, None)["error"]
    )
    # Inline observed inputs are admitted first through the observed guards, then feed the same recompute.
    # Their singular scale ("million") binds the source's "(In millions)".
    shared = {"evidence_id": "ev-table", "metric": "Revenue", "entity": "NVIDIA Corporation", "unit": "million"}
    first: dict[str, JSONValue] = {**shared, "value": "96,221", "period": "Three Months Ended Jul 26, 2026"}
    second: dict[str, JSONValue] = {**shared, "value": "46,743", "period": "Three Months Ended Jul 27, 2025"}
    inline: list[JSONValue] = [first, second]
    assert kw._direct_op({**growth, "value": "105.9", "inputs": inline}, None)["datapoint"] == point
    unproven: list[JSONValue] = [first, {**second, "value": "46,744"}]
    assert "'46,744'" in str(kw._direct_op({**growth, "value": "105.9", "inputs": unproven}, None)["error"])
    nested: list[JSONValue] = [first, {**growth, "value": "105.9"}]
    assert "observed number declaration" in str(
        kw._direct_op({**growth, "value": "1", "inputs": nested}, None)["error"]
    )
    # Operands of another metric, or an output period that names other periods, never publish as growth.
    cost: list[JSONValue] = [first, {**second, "metric": "Cost of revenue", "value": "12,000"}]
    mixed = kw._direct_op({**growth, "value": "701.8", "inputs": cost}, None)
    assert "different metrics" in str(mixed.get("error")), mixed
    unrelated = kw._direct_op({**growth, "value": "105.9", "period": "FY2031 versus FY2030"}, None)
    assert "name both input periods" in str(unrelated.get("error")), unrelated
    # A second entity spelling the source also states is its own datapoint, never a mismatched dedupe hit.
    longer = {**first, "id": "long", "entity": "NVIDIA Corporation and Subsidiaries", "evidenceId": "ev-table"}
    longer |= {"op": "admit_datapoint", "sessionId": sid}
    point_long = kw._direct_op(longer, None).get("datapoint")
    assert isinstance(point_long, dict) and point_long["entity"] == "NVIDIA Corporation and Subsidiaries", point_long
    # A declared "Jul 26,2026" admits the source's "Jul 26, 2026"; spacing never makes a declaration differ.
    tight = {**longer, "id": "tight", "period": "Three Months Ended Jul 26,2026"}
    point_tight = kw._direct_op(tight, None).get("datapoint")
    assert isinstance(point_tight, dict) and point_tight["period"] == "Three Months Ended Jul 26, 2026", point_tight
    # A growth period written with other spacing and "the" still names both input periods.
    loose = {**growth, "id": "loose", "value": "105.9"}
    loose["period"] = "the Three Months Ended Jul 26,2026 versus the Three Months Ended Jul 27,2025"
    assert isinstance(kw._direct_op(loose, None).get("datapoint"), dict)


def test_admit_datapoint_reuses_filtered_document_handle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tests.research_source_seam as seam
    from app.research import service

    passage = "NVDA FY2025 Revenue was $46.7 billion."
    prefix = "An unrelated document introduction.\n"
    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-prose": prefix + passage})
    store = ResearchRepository()
    handle = {
        **seam.handle_for_registered(passage),
        "offset": 0,
        "section": "income",
        "query": "Revenue",
    }
    filtered = seam.fake_sec_document(
        seam.DEFAULT_ACCESSION,
        seam.DEFAULT_DOCUMENT,
        offset=len(prefix),
        max_chars=len(passage),
    )
    with mock.patch("app.sec.documents.get_sec_document", return_value=filtered) as reload:
        admitted = service.record_evidence(
            sid,
            store.list_jobs(sid)[0].job_id,
            {
                "content": passage,
                "source_name": "SEC",
                "subject": "NVDA",
                "source_handle": handle,
                "matching_passage": passage,
            },
            repo=store,
        )
        request: dict[str, JSONValue] = {
            "id": "filtered",
            "op": "admit_datapoint",
            "sessionId": sid,
            "evidenceId": admitted["evidence_id"],
            "metric": "Revenue",
            "value": "46.7",
            "entity": "NVDA",
            "period": "FY2025",
            "unit": "billion",
        }
        result = kw._direct_op(request, None)
    assert "error" not in result, result
    assert reload.call_count == 2
    assert reload.call_args.kwargs["section"] == "income"
    assert reload.call_args.kwargs["query"] == "Revenue"
    eid = admitted["evidence_id"]
    assert isinstance(eid, str)
    original = store.get_evidence(eid)
    metadata = original["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_handle"] == handle
    legacy = {**original, "metadata": {key: value for key, value in metadata.items() if key != "source_handle"}}
    with mock.patch.object(ResearchRepository, "get_evidence", return_value=legacy):
        rejected = kw._direct_op(request, None)
    assert "filtered document view" in str(rejected["error"])
    assert "datapoint" not in rejected


@pytest.mark.parametrize("domain,tool", [("SEC", "get_xbrl_facts"), ("FINRA", "query_finra"), ("WEB", "search_web")])
def test_admit_datapoint_replays_persisted_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, domain: str, tool: str
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    store = ResearchRepository()
    sid = service.create_research(
        "Revenue?",
        "Revenue?",
        policy={
            "research_sources": {"mode": "allowlist", "sources": ["SEC", "FINRA", "WEB"]},
        },
        repo=store,
    )
    job = str(service.start_job(sid, source=domain, repo=store)["job_id"])
    passage = "NVDA FY2025 Revenue was 46.7."
    row = {"symbol": "NVDA", "period": "FY2025", "Revenue": "46.7"}
    locator = passage if domain == "WEB" else json.dumps(row, sort_keys=True)
    payload = (
        {
            "evidence": [
                {"url": "https://example.com/decoy", "highlight": passage},
                {"url": "https://example.com/revenue", "highlight": passage},
            ]
        }
        if domain == "WEB"
        else {"records": [row]}
    )
    ref = service.persist_tool_result(sid, job, tool, "revenue", payload, repo=store)["tool_result_id"]
    original = service.record_evidence(
        sid,
        job,
        {
            "content": passage,
            "source_name": domain,
            "tool_result_id": ref,
            "matching_passage": locator,
            "subject": "NVDA",
            **({"source_record_id": "https://example.com/revenue"} if domain == "WEB" else {}),
        },
        repo=store,
    )["evidence_id"]
    service.complete_job(job, repo=store)
    result = kw._direct_op(
        {
            "id": "record",
            "op": "admit_datapoint",
            "sessionId": sid,
            "evidenceId": original,
            "metric": "Revenue",
            "value": "46.7",
            "entity": "NVDA",
            "period": "FY2025",
        },
        None,
    )
    assert "error" not in result, result
    assert result["evidence_id"] != original
    assert isinstance(result["datapoint"], dict)
    assert result["datapoint"]["value"] == "46.7"
    if domain == "WEB":
        eid = result["evidence_id"]
        assert isinstance(eid, str)
        provenance = store.get_evidence(eid)["provenance"]
        assert isinstance(provenance, dict)
        assert provenance["url"] == "https://example.com/revenue"


def test_admit_datapoint_rejects_numbers_absent_from_source_property(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hypothesis import given, seed, settings
    from hypothesis import strategies as st

    import tests.research_source_seam as seam

    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-prose": "NVDA FY2025 Revenue was $46.7 billion."})

    @seed(20261027)
    @settings(max_examples=15, deadline=None)
    @given(st.integers(min_value=100, max_value=999))
    def check(value: int) -> None:
        result = kw._direct_op(
            {
                "id": "property",
                "op": "admit_datapoint",
                "sessionId": sid,
                "evidenceId": "ev-prose",
                "metric": "Revenue",
                "value": str(value),
                "entity": "NVDA",
                "period": "FY2025",
            },
            None,
        )
        assert "error" in result
        assert "datapoint" not in result
        assert len(ResearchRepository().list_evidence(sid)) == 1

    check()


def test_shutdown_cancels_queued_work_without_waiting_for_running_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()
    entered = [threading.Event(), threading.Event()]
    pools = [ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=1)]

    def _running(started: threading.Event) -> None:
        started.set()
        release.wait(2)

    running = [pool.submit(_running, started) for pool, started in zip(pools, entered)]
    queued = [pool.submit(lambda: None) for pool in pools]
    monkeypatch.setattr(sched, "_TOOL_POOL", pools[1])
    monkeypatch.setattr("app.research.tool_runner._SEC_POOL", pools[0])
    monkeypatch.setattr("app.research.tool_runner._TOOL_POOL", pools[1])
    try:
        assert all(started.wait(1) for started in entered)
        kw._shutdown_pools()
        assert all(future.cancelled() for future in queued)
        assert all(not future.done() for future in running)
    finally:
        release.set()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def test_arguments_caller_context_scope_reaches_needle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller temporal scope and as_of reach Needle; raw evidence never does."""
    seen: list[dict[str, object]] = []

    def _gen(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)
    out = kw._arguments(
        {
            "id": "c1",
            "op": "arguments",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "context": {
                "temporal_scope": {"mode": "as_of", "as_of": "2026-06-30"},
                "as_of": "2026-06-30",
                "evidence": [{"evidence_id": "ev-1", "content": "raw"}],
            },
            "schema": {"type": "object"},
        }
    )
    assert out["id"] == "c1" and out["tool"] == "list_sec_filings"
    context = seen[0]["context"]
    assert isinstance(context, dict)
    assert context.get("as_of") == "2026-06-30"
    assert context.get("temporal_scope") == {"mode": "as_of", "as_of": "2026-06-30"}
    assert "evidence" not in context


def test_arguments_error_envelopes_keep_500_char_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both error envelopes truncate at 500 chars, never 501."""

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("x" * 600)

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    outer = kw._arguments({"id": "e1", "tool": "list_sec_filings", "objective": "NVDA revenue?"})
    assert isinstance(outer["error"], str) and len(outer["error"]) == 500
    ticker = kw._arguments({"id": "e2", "tool": "list_sec_filings", "objective": "q?", "ticker": "X" * 600})
    assert isinstance(ticker["error"], str) and len(ticker["error"]) == 500
    assert "ticker" in str(ticker["error"])


def test_arguments_prompt_grounds_when_objective_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt-only request grounds like its objective; without either, the objective is empty."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _withhold(**kwargs: object) -> object:
        return None

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _withhold)
    seeded = kw._arguments({"id": "p1", "tool": "list_sec_filings", "prompt": "What drove NVDA revenue last quarter?"})
    assert seeded["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}


@pytest.mark.parametrize("payload", [{"objective": 7}, {"objective": None, "prompt": 7}])
def test_arguments_non_string_objective_cannot_seed_a_call(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, JSONValue]
) -> None:
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _needle_withhold)
    out = kw._arguments({"id": "bad-objective", "tool": "list_sec_filings", **payload})
    assert out == {"id": "bad-objective", "error": "needle arguments failed"}


def test_arguments_missing_needle_tool_defaults_to_jev_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Needle payload without a tool key still grounds the JEV-selected tool."""

    def _gen(**kwargs: object) -> dict[str, object]:
        return {"arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _schema(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments({"id": "m1", "tool": "query_finra", "objective": "short interest?"})
    assert out["tool"] == "query_finra" and "error" not in out


def test_arguments_non_mapping_needle_arguments_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dict Needle arguments repair from empty instead of raising into the envelope."""

    def _gen(**kwargs: object) -> dict[str, object]:
        return {"tool": "query_finra", "arguments": ["ticker"], "confidence": 0.9, "reasoning": "r"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _schema(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments({"id": "n1", "tool": "query_finra", "objective": "short interest?"})
    assert "error" not in out and isinstance(out["arguments"], dict)


def test_arguments_node_session_id_carries_without_arguments_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node-carried session_id shortcuts without Needle when arguments holds no mapping."""

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run for a sid carry")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments({"id": "n1", "tool": "research_status", "node": {"session_id": "rs:5"}})
    assert out["arguments"] == {"session_id": "rs:5"}


def test_tool_result_archive_failure_prevents_persisting_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "entry.sqlite"))
    sid = service.create_research("NVDA filings?")
    store = ResearchRepository()
    job_id = service.start_job(sid, source="SEC")["job_id"]
    assert isinstance(job_id, str)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("archive unavailable")

    monkeypatch.setattr("app.storage.raw_archive.store_evidence_artifact", unavailable)
    with pytest.raises(OSError, match="archive unavailable"):
        service.persist_tool_result(sid, job_id, "list_sec_filings", None, {"filings": []})
    assert store.list_tool_results(sid) == []


@pytest.mark.parametrize("choice", ["low", "medium", "high", "xhigh", "invalid"])
def test_effort_dispatch_uses_scope_limits_and_strict_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, choice: str
) -> None:
    seen: list[object] = []

    def transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append(state)
        probabilities: dict[str, JSONValue] = {
            label: 1.0 if label == choice else 0.0 for label in ("low", "medium", "high", "xhigh")
        }
        return {
            "answers": {
                "reasoning_effort": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": probabilities,
                }
            }
        }

    def no_persist(**kwargs: object) -> None:
        return None

    client = JevClient(transport=transport, data_root=tmp_path)
    monkeypatch.setattr(client, "_persist", no_persist)
    monkeypatch.setattr(kw.time, "time", lambda: 1000.0)
    request: dict[str, JSONValue] = {
        "id": "effort-1",
        "op": "effort",
        "sessionId": "session-1",
        "prompt": "Explain these filings",
        "ticker": "NVDA",
        "watchlist": ["AAPL", "MSFT"],
        "deadlineAt": 1030000,
        "maxOutputTokens": 2048,
        "answerTokenBudget": 4096,
        "maxAnswerChars": 6000,
    }
    parsed, error = kw._request_or_response(json.dumps(request))
    assert parsed == request and error is None
    out = kw._direct_op(request, client)
    if choice == "invalid":
        terminal = out["terminal"]
        assert isinstance(terminal, dict) and terminal["category"] == "provider_error"
        assert "malformed_decision_answer" in str(out["error"])
    else:
        assert out == {"id": "effort-1", "effort": choice}
    state = seen[0]
    assert isinstance(state, dict)
    assert state["prompt"] == "Explain these filings"
    assert json.loads(state["scope"].split(": ", 1)[1]) == {"watchlist": ["AAPL", "MSFT"]}
    assert state["limits"] == {
        "deadlineMs": 30000.5,
        "maxOutputTokens": 2048,
        "answerTokenBudget": 4096,
        "maxAnswerChars": 6000,
    }


def test_sections_dispatch_keeps_parts_at_half_probability_in_answer_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each part is one noul. 0.50 keeps a part, 0.49 drops it, and the declared order wins over answer order.

    Every decision lands in the one shared chat session, created on first use and reused after.
    """
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "chat.sqlite"))
    seen: list[tuple[object, Mapping[str, JSONValue]]] = []
    probability = {
        "risks": 0.5,
        "short_term": 0.9,
        "long_term": 0.49,
        "analysts": 1.0,
        "bull_case": 0.0,
        "bear_case": 0.7,
    }

    def transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append((state, questions))
        return {"answers": {qid: {"type": "noul", "noul": p} for qid, p in probability.items()}}

    client = JevClient(transport=transport, data_root=tmp_path)
    request: dict[str, JSONValue] = {
        "id": "sections-1",
        "op": "sections",
        "prompt": "What do the filings say?",
        "ticker": "NVDA",
    }
    parsed, error = kw._request_or_response(json.dumps(request))
    assert parsed == request and error is None
    assert kw._direct_op(request, client) == {
        "id": "sections-1",
        "sections": ["short_term", "analysts", "bear_case", "risks"],
    }
    state, questions = seen[0]
    assert isinstance(state, dict) and json.loads(state["scope"].split(": ", 1)[1]) == {"ticker": "NVDA"}
    assert set(questions) == set(probability)
    kw._direct_op({**request, "id": "sections-2"}, client)
    store = ResearchRepository()
    assert store.get_session("rs:chat").query == "default chat"
    assert [d.decision_type for d in store.list_decisions("rs:chat")] == ["answer_sections", "answer_sections"]
    assert [s["session_id"] for s in store.list_sessions()] == ["rs:chat"]


def test_sections_dispatch_rejects_a_missing_prompt_and_a_malformed_answer(tmp_path: Path) -> None:
    assert kw._direct_op({"id": "s0", "op": "sections", "prompt": " "}, None)["terminal"] == {
        "category": "invalid_params",
        "message": "prompt required",
    }

    def malformed(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return {"answers": {"short_term": {"type": "noul", "noul": 2}}}

    client = JevClient(transport=malformed, data_root=tmp_path)
    out = kw._direct_op({"id": "s1", "op": "sections", "prompt": "q"}, client)
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "provider_error"


@pytest.mark.parametrize("key", ["maxOutputTokens", "answerTokenBudget", "maxAnswerChars"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2000"])
def test_effort_rejects_invalid_limits_before_jev(monkeypatch: pytest.MonkeyPatch, key: str, value: JSONValue) -> None:
    def unexpected() -> JevClient:
        pytest.fail("Invalid limits must not call JEV")

    monkeypatch.setattr(kw, "_shared_jev", unexpected)
    out = kw._direct_op({"id": "bad", "op": "effort", "prompt": "Explain", key: value}, None)
    assert out["terminal"] == {"category": "invalid_params", "message": f"{key} must be a positive integer"}


def test_effort_deadline_expires_before_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected() -> JevClient:
        pytest.fail("An expired deadline must not call JEV")

    monkeypatch.setattr(kw, "_shared_jev", unexpected)
    out = kw._direct_op({"id": "late", "op": "effort", "prompt": "Explain", "deadlineAt": 1}, None)
    assert out["terminal"] == {"category": "timeout", "message": "deadline reached before effort"}


def test_effort_accepts_unset_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "effort.sqlite"))
    seen: list[object] = []

    def transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append(state)
        return {
            "answers": {
                "reasoning_effort": {
                    "type": "choice",
                    "choice": "low",
                    "confidence": 1.0,
                    "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0, "xhigh": 0.0},
                }
            }
        }

    def no_persist(**kwargs: object) -> None:
        return None

    client = JevClient(transport=transport, data_root=tmp_path)
    monkeypatch.setattr(client, "_persist", no_persist)
    assert kw._direct_op({"id": "simple", "op": "effort", "prompt": "Hello"}, client) == {
        "id": "simple",
        "effort": "low",
    }
    assert seen == [{"prompt": "Hello", "scope": "", "limits": {}}]


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": "Explain", "ticker": 3},
        {"prompt": "Explain", "watchlist": "NVDA"},
        {"prompt": "Explain", "context": "untrusted"},
    ],
)
def test_effort_rejects_invalid_prompt_or_scope(monkeypatch: pytest.MonkeyPatch, fields: dict[str, JSONValue]) -> None:
    def unexpected() -> JevClient:
        pytest.fail("Invalid prompt or scope must not call JEV")

    monkeypatch.setattr(kw, "_shared_jev", unexpected)
    out = kw._direct_op({"id": "bad-scope", "op": "effort", **fields}, None)
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "invalid_params"


@pytest.mark.parametrize("existing_session", [False, True])
def test_effort_persists_decision_in_a_real_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing_session: bool
) -> None:
    from app.research.session import create_session

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "effort-persist.sqlite"))
    store = ResearchRepository()
    request: dict[str, JSONValue] = {"id": "1", "op": "effort", "prompt": "Explain NVDA filings", "ticker": "NVDA"}
    if existing_session:
        session = create_session("Explain NVDA filings", "Explain NVDA filings", ticker="NVDA")
        store.save_session(session)
        request["sessionId"] = session.session_id

    def transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return {
            "answers": {
                "reasoning_effort": {
                    "type": "choice",
                    "choice": "low",
                    "confidence": 1.0,
                    "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0, "xhigh": 0.0},
                }
            }
        }

    client = JevClient(transport=transport, data_root=tmp_path)
    assert kw._direct_op(request, client) == {"id": "1", "effort": "low"}
    sessions = store.list_sessions()
    assert len(sessions) == 1
    sid = sessions[0]["session_id"]
    assert isinstance(sid, str) and sid != "1"
    if existing_session:
        assert sid == request["sessionId"]
    persisted = store.get_session(sid)
    assert persisted.query == "Explain NVDA filings" and persisted.ticker == "NVDA"
    decisions = store.list_decisions(sid)
    assert len(decisions) == 1 and decisions[0].decision_type == "reasoning_effort"
    assert decisions[0].selected == {
        "reasoning_effort": {
            "kind": "choice",
            "choice": "low",
            "confidence": 1.0,
            "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0, "xhigh": 0.0},
        }
    }
    assert store.list_jobs(sid) == []


def test_effort_does_not_replace_an_unknown_supplied_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "effort-missing.sqlite"))

    def transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return {
            "answers": {
                "reasoning_effort": {
                    "type": "choice",
                    "choice": "low",
                    "confidence": 1.0,
                    "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0, "xhigh": 0.0},
                }
            }
        }

    client = JevClient(transport=transport, data_root=tmp_path)
    out = kw._direct_op({"id": "missing", "op": "effort", "prompt": "Explain", "sessionId": "not-a-session"}, client)
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "provider_error"
    assert "unknown session_id" in str(out["error"])
    assert ResearchRepository().list_sessions() == []


def _scope_client(probability: float | None) -> JevClient:
    """JevClient whose scope noul answers with one probability; None raises like an outage."""

    class _Scope(JevClient):
        @override
        async def request_in_scope(self, prompt: str, *, session_id: str) -> bool:
            if probability is None:
                raise RuntimeError("decision sidecar down")
            return probability >= 0.50

    def _unused(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return {}

    return _Scope(transport=_unused)


def test_scope_op_answers_a_cooking_prompt_without_tools() -> None:
    """The judge cooking_scope prompt gets the runtime's fixed refusal, and it states the limitation."""
    out = kw._direct_op(
        {"id": "sc1", "op": "scope", "prompt": "How do I bake sourdough bread at home?"}, _scope_client(0.1)
    )
    assert out == {"id": "sc1", "in_scope": False, "answer": kw.OUT_OF_SCOPE_ANSWER}
    assert "outside the scope" in kw.OUT_OF_SCOPE_ANSWER
    assert "unsupported" in kw.OUT_OF_SCOPE_ANSWER
    assert not any(char.isdigit() for char in kw.OUT_OF_SCOPE_ANSWER)


@pytest.mark.parametrize(
    "prompt",
    [
        "Create a thesis: NVDA beats earnings on AI datacenter demand.",
        "Refine my NVDA thesis: AI demand actually collapsed — update the position.",
        "Watch my NVDA thesis for the upcoming earnings and journal that I'm nervous about guidance.",
        "What is the web saying about my largest portfolio holding?",
        "what time is it",
    ],
)
def test_scope_op_keeps_judge_prompts_in_scope(prompt: str) -> None:
    """Thesis, portfolio, and clock prompts stay in scope, so research still runs."""
    out = kw._direct_op({"id": "sc2", "op": "scope", "prompt": prompt}, _scope_client(0.8))
    assert out == {"id": "sc2", "in_scope": True}


def test_scope_op_jev_failure_never_blocks_research() -> None:
    """A scope outage answers with an error and no verdict, so the caller keeps researching."""
    out = kw._direct_op({"id": "sc3", "op": "scope", "prompt": "NVDA revenue?"}, _scope_client(None))
    assert out["id"] == "sc3"
    assert "in_scope" not in out and "answer" not in out
    assert "scope gate failed" in str(out["error"])


def test_scope_op_requires_a_prompt() -> None:
    out = kw._direct_op({"id": "sc4", "op": "scope", "prompt": "  "}, _scope_client(0.9))
    assert out["error"] == "prompt required"
    assert out["id"] == "sc4"
    assert out["terminal"] == {"category": "invalid_params", "message": "prompt required"}
    assert out["failures"] == {"invalid_params": 1}


@pytest.mark.parametrize("session_id", [None, "scope-session"])
def test_scope_op_passes_the_prompt_and_session(
    session_id: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller session is reused; without one the op creates a real session, so the decision can persist."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "scope.sqlite"))
    seen: list[tuple[str, str]] = []

    class Scope(JevClient):
        @override
        async def request_in_scope(self, prompt: str, *, session_id: str) -> bool:
            seen.append((prompt, session_id))
            return True

    req: dict[str, JSONValue] = {"id": "scope-request", "op": "scope", "prompt": "NVDA revenue?"}
    if session_id is not None:
        req["session_id"] = session_id
    assert kw._direct_op(req, Scope()) == {"id": "scope-request", "in_scope": True}
    ((prompt, used),) = seen
    assert prompt == "NVDA revenue?"
    if session_id is not None:
        assert used == session_id
    else:
        assert used != "scope-request"
        assert ResearchRepository().get_session(used).objective == "NVDA revenue?"


def test_scope_op_reports_the_deadline_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(kw.time, "time", lambda: next(clock, 2.0))
    out = kw._direct_op(
        {"id": "scope-deadline", "op": "scope", "prompt": "NVDA revenue?", "deadlineAt": 1000},
        _scope_client(0.8),
    )
    assert out == {"id": "scope-deadline", "error": "scope gate failed: deadline reached during scope"}


def test_scope_op_bounds_an_outage_error() -> None:
    class Scope(JevClient):
        @override
        async def request_in_scope(self, prompt: str, *, session_id: str) -> bool:
            raise RuntimeError("x" * 600)

    out = kw._direct_op({"id": "scope-outage", "op": "scope", "prompt": "NVDA revenue?"}, Scope())
    assert out == {"id": "scope-outage", "error": "scope gate failed: " + "x" * 481}


def test_scope_op_is_accepted_by_the_worker_protocol() -> None:
    """The harness sends op:scope; the worker protocol must accept exactly that name."""
    request, response = kw._request_or_response('{"op": "scope", "id": "sc5", "prompt": "q?"}')
    assert response is None
    assert request == {"op": "scope", "id": "sc5", "prompt": "q?"}
    assert "scope" in kw._KERNEL_OPS


# --- native research loop: open_research, research_tool, prove_claims -------------------------------

_FINRA_ROW: dict[str, object] = {
    "symbol": "NVDA",
    "period": "Q3",
    "current_short_position": 1234567,
}
_FINRA_POLICY: dict[str, object] = {"research_sources": {"mode": "allowlist", "sources": ["FINRA"]}}


def _short_interest_fixture(store: ResearchRepository, sid: str, ref: str) -> tuple[dict[str, object], str, object]:
    """(tool result, job id, runtime outcome) for one persisted FINRA result, as a tool call leaves them."""
    from app.research import service
    from app.tool_runtime import outcome_from_result

    job = str(service.start_job(sid, source="FINRA", repo=store)["job_id"])
    payload = {"status": "ok", "published_at": "2026-10-01", "records": [dict(_FINRA_ROW)]}
    tool_result_id = service.persist_tool_result(sid, job, "get_short_interest", ref, payload, repo=store)[
        "tool_result_id"
    ]
    result: dict[str, object] = {"tool_result_id": tool_result_id, "result": payload}
    return result, job, outcome_from_result("get_short_interest", result)


def test_research_tool_returns_admitted_ids_evidence_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loop reads one summary line per admitted row, each prefixed with its evidence id."""
    from app.research import service, tool_runner

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "wire.sqlite"))
    store = ResearchRepository()
    sid = service.create_research("short interest?", "short interest?", policy=_FINRA_POLICY, repo=store)
    seen: list[dict[str, object]] = []

    def _runtime(tool: str, arguments: dict[str, object], tool_session: object, **kwargs: object) -> dict[str, object]:
        seen.append({"tool": tool, "arguments": arguments, "kwargs": kwargs})
        job_id = str(kwargs["active_research_job_id"])
        payload = {"status": "ok", "published_at": "2026-10-01", "records": [dict(_FINRA_ROW)]}
        ref = service.persist_tool_result(sid, job_id, tool, "tr:wire", payload, repo=store)["tool_result_id"]
        return {"tool_result_id": ref, "result": payload}

    monkeypatch.setattr(tool_runner, "execute_agent_tool", _runtime)
    out = asyncio.run(tool_runner.run_tool(tool_runner.Kernel(store), sid, "get_short_interest", {"ticker": "NVDA"}))

    assert out["ok"] is True and out["tool"] == "get_short_interest"
    assert seen and seen[0]["arguments"] == {"ticker": "NVDA"}
    rows = out["evidence"]
    assert isinstance(rows, list) and rows
    evidence = [row for row in rows if isinstance(row, dict)]
    assert len(evidence) == len(rows)
    assert out["admitted"] == len(evidence)
    stored = {row["evidence_id"] for row in store.list_evidence(sid)}
    assert {row["id"] for row in evidence} <= stored
    assert all(row["source"] and row["retrievedAt"] for row in evidence)
    assert any(row.get("datapoints") for row in evidence)
    # Rows with a datapoint lead the summary: they are the citable ones.
    first = str(out["summary"]).split("\n", 1)[0]
    assert first.endswith("] datapoints"), first
    assert first[1 : first.index("]")] in {row["id"] for row in evidence if row.get("datapoints")}


def test_evidence_summary_prints_the_datapoint_identity() -> None:
    """A number sentence copies metric, value, unit, entity, and period from its datapoint line.

    The proof binds a stricter sentence than the passage states: a claim must state the datapoint's
    unit, so the model can only write a passing number sentence when the summary shows it.
    """
    from app.research import tool_runner

    row: dict[str, JSONValue] = {
        "id": "ev:1",
        "content": "Basic: 2.47 (NVIDIA Corporation, Three Months Ended Jul 26, 2026)",
        "datapoints": [
            {
                "id": "dp:1",
                "metric": "Basic",
                "value": "2.47",
                "entity": "NVIDIA Corporation",
                "period": "Three Months Ended Jul 26, 2026",
                "unit": "per share",
            }
        ],
    }

    summary = tool_runner.evidence_summary([row])

    assert summary.startswith("[ev:1] datapoints")
    for part in ("dp:1", "metric=Basic", "value=2.47", "unit=per share", "period=Three Months Ended Jul 26, 2026"):
        assert part in summary
    assert "Basic: 2.47" in summary


def test_research_tool_reports_a_tool_error_as_a_failed_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A provider error becomes ok:false with no evidence row and the tool's own text."""
    from app.research import service, tool_runner

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "fail.sqlite"))
    store = ResearchRepository()
    sid = service.create_research("short interest?", "short interest?", policy=_FINRA_POLICY, repo=store)

    def _failing_runtime(*args: object, **kwargs: object) -> dict[str, object]:
        return {"error": "provider exploded", "error_type": "tool_error"}

    monkeypatch.setattr(tool_runner, "execute_agent_tool", _failing_runtime)

    out = asyncio.run(tool_runner.run_tool(tool_runner.Kernel(store), sid, "get_short_interest", {"ticker": "NVDA"}))

    assert out["ok"] is False and out["evidence"] == [] and out["admitted"] == 0
    assert "provider exploded" in str(out["error"])
    assert "provider exploded" in str(out["summary"])
    assert store.list_evidence(sid) == []


def test_research_tool_denies_a_tool_outside_the_research_registry() -> None:
    """The registry allowlist is the trust boundary: an unlisted tool never runs."""
    from app.research import tool_runner

    out = asyncio.run(tool_runner.run_tool(tool_runner.Kernel(), "rs:none", "get_portfolio_snapshot", {}))

    assert out["ok"] is False and out["evidence"] == [] and out["admitted"] == 0
    assert "not a research tool" in str(out["error"])


def test_research_tool_requires_session_tool_and_object_arguments() -> None:
    assert kw._research_tool({"id": "t1", "tool": "search_web"}) == {"id": "t1", "error": "sessionId and tool required"}
    assert (
        kw._research_tool({"id": "t2", "sessionId": "rs:1", "tool": "search_web", "arguments": []})["error"]
        == "arguments must be an object"
    )


def test_research_tool_unknown_session_reports_a_failed_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "absent.sqlite"))
    out = kw._research_tool({"id": "t3", "sessionId": "rs:absent", "tool": "search_web", "arguments": {}})

    assert out["id"] == "t3" and out["ok"] is False and out["evidence"] == []
    assert out["error"]


def _proven_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, str, dict[str, JSONValue], str]:
    """(session, evidence id, canonical datapoint, the ledger row's own content) for one admitted number."""
    from app.research import service, tool_runner

    sid = service.create_research("short interest?", "short interest?", policy=_FINRA_POLICY, repo=ResearchRepository())
    store = ResearchRepository()
    result, job, outcome = _short_interest_fixture(store, sid, "tr:proof")
    tool_runner.admit_tool_result(tool_runner.Kernel(store), sid, job, "get_short_interest", result, outcome)
    row = next(item for item in store.list_evidence(sid) if tool_runner.row_datapoints(item))
    point = tool_runner.row_datapoints(row)[0]
    assert isinstance(point, dict)
    return sid, str(row["evidence_id"]), point, str(row["content"])


def test_prove_claims_passes_a_proven_number_and_rejects_an_unbound_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A number bound to a verified datapoint publishes; a number with no datapoint is rejected."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "proof.sqlite"))
    sid, eid, point, content = _proven_number(monkeypatch, tmp_path)

    out = kw._prove_claims(
        {
            "id": "p1",
            "sessionId": sid,
            "claims": [
                {
                    "text": content,
                    "claim_type": "observed_fact",
                    "evidence_ids": [eid],
                    "datapoint_ids": [point["id"]],
                },
                {
                    "text": "Short interest reached 9999999 shares.",
                    "claim_type": "observed_fact",
                    "evidence_ids": [eid],
                },
            ],
        }
    )

    verdicts = out["verdicts"]
    assert isinstance(verdicts, list) and len(verdicts) == 2
    assert verdicts[0] == {
        "claim_type": "observed_fact",
        "support_answer": "yes",
        "support": 1.0,
        "evidence_state": "sufficient_support",
    }
    rejected = verdicts[1]
    assert isinstance(rejected, dict)
    assert rejected["support_answer"] == "no" and str(rejected["error"]).startswith("proof:")


def test_prove_claims_is_index_aligned_and_rejects_malformed_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every input claim answers at its own index; a malformed claim is its own rejection."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "aligned.sqlite"))
    sid, eid, _point, _content = _proven_number(monkeypatch, tmp_path)

    out = kw._prove_claims(
        {
            "id": "p2",
            "sessionId": sid,
            "claims": [
                None,
                {"text": "  ", "claim_type": "observed_fact"},
                {"text": "X.", "claim_type": "observed_fact", "evidence_ids": [eid]},
            ],
        }
    )

    verdicts = out["verdicts"]
    assert isinstance(verdicts, list) and len(verdicts) == 3
    judged = [verdict for verdict in verdicts if isinstance(verdict, dict)]
    assert len(judged) == len(verdicts)
    assert [verdict["claim_type"] for verdict in judged] == ["", "observed_fact", "observed_fact"]
    for verdict in judged:
        assert verdict["support_answer"] == "no"
        assert str(verdict["error"]).startswith("proof:")


def test_prove_claims_reports_an_unknown_session_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing ledger is an outage, never a rejection: no verdict list crosses."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "missing.sqlite"))

    out = kw._prove_claims(
        {
            "id": "p3",
            "sessionId": "rs:absent",
            "claims": [{"text": "X.", "claim_type": "observed_fact", "evidence_ids": ["ev:1"]}],
        }
    )

    assert out["id"] == "p3" and "verdicts" not in out
    assert not str(out["error"]).startswith("proof:")


def test_research_ops_report_an_expired_deadline_without_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-expired deadline answers in the op's own shape and runs nothing."""
    ran: list[str] = []

    def _unexpected(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        ran.append("tool")
        return {}

    monkeypatch.setattr(kw, "_research_tool", _unexpected)

    tool = kw._direct_op(
        {
            "id": "d1",
            "op": "research_tool",
            "sessionId": "rs:x",
            "tool": "search_web",
            "arguments": {},
            "deadlineAt": 1,
        },
        None,
    )
    assert tool["id"] == "d1" and tool["ok"] is False and tool["tool"] == "search_web"
    assert "deadline reached before research_tool" in str(tool["error"])
    assert ran == []

    proof = kw._direct_op(
        {
            "id": "d2",
            "op": "prove_claims",
            "sessionId": "rs:x",
            "claims": [{"claim_type": "observed_fact"}, {"claim_type": "unknown"}],
            "deadlineAt": 1,
        },
        None,
    )
    verdicts = proof["verdicts"]
    assert isinstance(verdicts, list)
    judged = [verdict for verdict in verdicts if isinstance(verdict, dict)]
    assert len(judged) == len(verdicts)
    assert [verdict["claim_type"] for verdict in judged] == ["observed_fact", "unknown"]
    for verdict in judged:
        assert verdict["support_answer"] == "no"
        assert not str(verdict["error"]).startswith("proof:")


def test_open_research_creates_the_scoped_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ticker and asOf scope the new session; a watchlist request keeps no session ticker; no node exists."""
    from app.research import tool_runner

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "open.sqlite"))
    store = ResearchRepository()

    out = kw._open_research({"id": "o1", "prompt": "  NVDA risk factors?  ", "ticker": "nvda", "asOf": "2026-01-02"})
    session = store.get_session(str(out["sessionId"]))
    assert out["id"] == "o1"
    raw_tools = out["tools"]
    assert isinstance(raw_tools, list)
    assert raw_tools == sorted(tool_runner.RESEARCH_TOOLS)  # the model only sees executable tools
    assert session.objective == "NVDA risk factors?"
    assert session.ticker == "NVDA"
    assert str(session.as_of).startswith("2026-01-02")
    assert store.list_nodes(session.session_id) == []

    watch = kw._open_research({"id": "o2", "prompt": "my watchlist?", "ticker": "nvda", "watchlist": ["nvda", "qqq"]})
    assert store.get_session(str(watch["sessionId"])).ticker is None


@pytest.mark.parametrize(
    "request_",
    [
        {"prompt": "   "},
        {"prompt": 7},
        {"prompt": "q?", "ticker": 7},
        {"prompt": "q?", "watchlist": "NVDA"},
        {"prompt": "q?", "watchlist": []},
        {"prompt": "q?", "watchlist": ["NVDA", "SEC"]},
    ],
)
def test_open_research_rejects_a_bad_request_before_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request_: dict[str, JSONValue]
) -> None:
    """A malformed prompt, ticker, or watchlist answers an error and creates no session."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "bad.sqlite"))
    out = kw._open_research({"id": "o3", **request_})

    assert out["id"] == "o3" and out["error"] and "sessionId" not in out
    assert ResearchRepository().list_sessions() == []


def test_open_research_advertises_exactly_the_tools_research_tool_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The advertised set is the accepted set: one constant drives both, so they cannot drift."""
    from app.research import tool_runner

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "advertised.sqlite"))
    raw = kw._open_research({"id": "o4", "prompt": "q?"})["tools"]
    assert isinstance(raw, list)
    advertised = [name for name in raw if isinstance(name, str)]
    assert len(advertised) == len(raw)

    assert advertised == sorted(tool_runner.RESEARCH_TOOLS)
    assert set(advertised) == tool_runner.research_tool_names()
    for name in advertised:
        assert name in tool_runner.RESEARCH_TOOLS
    for name in (
        "call_tool",
        "browse_tools",
        "search_tools",
        "list_tool_domains",
        "describe_tool",
        "research_start",
        "research_read",
        "research_read_search",
        "research_resume",
        "set_chart_symbol",
        "update_watchlist",
        "add_widget",
    ):
        assert name not in advertised
        denied = asyncio.run(tool_runner.run_tool(tool_runner.Kernel(), "rs:none", name, {}))
        assert denied["ok"] is False and "not a research tool" in str(denied["error"])


@pytest.mark.parametrize(
    "change,error",
    [
        ({"evidenceId": "missing"}, "missing"),
        ({"value": "99.9"}, "99.9"),
        ({"period": "FY2024"}, "period"),
        ({"sessionId": None}, "sessionId and evidenceId required"),
        ({"sessionId": "missing"}, "another session"),
    ],
)
def test_admit_datapoint_rejection_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: dict[str, JSONValue], error: str
) -> None:
    import tests.research_source_seam as seam

    sid = seam.proven_session(monkeypatch, tmp_path, {"ev-prose": "NVDA FY2025 Revenue was $46.7 billion."})
    request: dict[str, JSONValue] = {
        "id": "rejected",
        "op": "admit_datapoint",
        "sessionId": sid,
        "evidenceId": "ev-prose",
        "metric": "Revenue",
        "value": "46.7",
        "entity": "NVDA",
        "period": "FY2025",
        "unit": "billion",
        **change,
    }
    result = kw._direct_op(request, None)
    assert error in str(result.get("error")), result
    assert "datapoint" not in result


@pytest.mark.parametrize("kind", ["news", "filing", "insider", "alert"])
def test_request_context_preserves_valid_feed_fields(kind: str) -> None:
    context: dict[str, JSONValue] = {
        "kind": kind,
        "title": "A headline",
        "source": "SEC",
        "at": "",
        "url": "https://example.com/news",
        "sym": "NVDA",
    }
    assert kw._request_context({"context": context}) == context
    minimal: dict[str, JSONValue] = {"kind": kind, "title": "A headline", "source": "SEC", "at": "now"}
    assert kw._request_context({"context": minimal}) == minimal
    assert kw._request_context({}) is None


@pytest.mark.parametrize(
    "change",
    [
        {"extra": True},
        {"kind": "other"},
        {"title": ""},
        {"title": "a" * 301},
        {"source": None},
        {"url": "file:///tmp/private"},
        {"url": "a" * 2001},
        {"at": None},
        {"at": "a" * 41},
        {"sym": ""},
    ],
)
def test_request_context_rejects_malformed_feed_fields(change: dict[str, JSONValue]) -> None:
    context: dict[str, JSONValue] = {"kind": "news", "title": "Headline", "source": "SEC", "at": "now", **change}
    with pytest.raises(ValueError):
        kw._request_context({"context": context})


@pytest.mark.parametrize("context", [[], "news", 42])
def test_request_context_rejects_nonobjects(context: JSONValue) -> None:
    with pytest.raises(ValueError, match="must be a feed item"):
        kw._request_context({"context": context})
