"""Observability: redaction units and recorder-independent budget reserves.

RUNS_DB_PATH is isolated per session by the root conftest fixture.
"""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.policy import RequestContext
from app.redact import redact_json, redact_text, redact_value
from app.runtime import ExecutionBudget
from app.storage.runs import (
    RunRecorder,
    finalize_failed_run,
    reset_current_recorder,
    set_current_recorder,
)


def test_redact_text_units():
    assert redact_text("Authorization: Bearer abc123") == "Authorization: Bearer [REDACTED]"
    assert redact_text("key=sk-or-v1-abcdefghijklmnop end") == "key=sk-or-v1-[REDACTED] end"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    assert redact_text(jwt) == "eyJ[REDACTED JWT]"
    assert redact_text("plain text") == "plain text"
    # Already-redacted output is stable under re-redaction.
    assert redact_text("Bearer [REDACTED]") == "Bearer [REDACTED]"
    assert redact_text("sk-or-v1-[REDACTED]") == "sk-or-v1-[REDACTED]"
    # Account identifiers in free text (review round 2 P1).
    assert redact_text("My account_number is 12345678") == "My account_number is [REDACTED]"
    assert redact_text("My account_id=87654321") == "My account_id=[REDACTED]"
    # Bare digit runs stay untouched (CIKs/accession numbers are structural).
    assert redact_text("cik 0000320193 filing") == "cik 0000320193 filing"
    assert redact_text("account 2026 taxes") == "account 2026 taxes"


def test_redact_value_units():
    value = {
        "accountNumber": "12345678",
        "nested": {"client_secret": "s3cret", "position_id": "pos-42"},
        "tags": ["a", "Bearer tok"],
        "count": 3,
        "flag": None,
        "provider_instrument_id": "inst-7",
    }
    redacted = redact_value(value)
    assert isinstance(redacted, dict)
    assert redacted["accountNumber"] == "[REDACTED]"
    assert redacted["nested"]["client_secret"] == "[REDACTED]"
    # Structural research identifiers pass through.
    assert redacted["nested"]["position_id"] == "pos-42"
    assert redacted["provider_instrument_id"] == "inst-7"
    assert redacted["tags"] == ["a", "Bearer [REDACTED]"]
    assert redacted["count"] == 3
    assert redacted["flag"] is None


def test_redact_json_units():
    assert redact_json('{"token": "abc", "ticker": "AAPL"}') == ('{"token": "[REDACTED]", "ticker": "AAPL"}')
    assert redact_json("not json") == "not json"


def test_run_limits_default_unbounded_evidence():
    """RunLimits defaults to no evidence-token ceiling (counters stay telemetry)."""
    from app.policy import RunLimits

    assert RunLimits().max_evidence_tokens is None


def test_reserve_methods_enforce_runtime():
    """Reserves refuse once elapsed runtime is gone, even with call slots left."""
    budget = ExecutionBudget(
        max_tool_calls=2,
        max_runtime=1.0,
        max_evidence_tokens=48000,
    )
    assert budget.reserve_tool_call() is True
    assert budget.reserve_search_call() is True
    budget._started -= 60  # pretend the budget started 60s ago
    assert budget.runtime_remaining() <= 0
    assert budget.reserve_search_call() is False
    assert budget.reserve_tool_call() is False


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


def test_concurrent_recorder_writes_unique_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """8 threads x 10 tool+evidence rows keep every row with a unique ID."""
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    recorder = _recorder("run-conc-1")
    with recorder:

        def worker() -> None:
            for _ in range(10):
                seq = recorder.next_tool_seq()
                tc_id = f"{recorder.run_id}:tc:{seq}"
                now = datetime.now(UTC).isoformat()
                recorder.record_tool_call(
                    tool_call_id=tc_id,
                    round=0,
                    tool_name="search_tools",
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
                recorder.record_evidence(
                    evidence_id=f"{recorder.run_id}:evid:{recorder.next_evidence_seq():04d}",
                    run_id=recorder.run_id,
                    tool_call_id=tc_id,
                    round=0,
                    tool_name="search_tools",
                    rendered_hash="h",
                    rendered_bytes=1,
                    estimated_tokens=1,
                    source_names="[]",
                    source_freshness="{}",
                    as_of=None,
                    rendered_text="t",
                )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    conn = sqlite3.connect(str(tmp_path / "runs.sqlite"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == 80
        assert conn.execute("SELECT COUNT(DISTINCT tool_call_id) FROM tool_calls").fetchone()[0] == 80
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 80
        assert conn.execute("SELECT COUNT(DISTINCT evidence_id) FROM evidence").fetchone()[0] == 80
    finally:
        conn.close()


def test_tool_call_telemetry_columns_migrated_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pre-telemetry DBs gain the columns on open; queue/handler/cache persist."""
    path = tmp_path / "runs.sqlite"
    conn = sqlite3.connect(str(path))
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
    with _recorder("run-tel-1") as recorder:
        now = datetime.now(UTC).isoformat()
        recorder.record_tool_call(
            tool_call_id="run-tel-1:tc:1",
            round=0,
            tool_name="search_tools",
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
            protocol_id="proto-1",
            bridge_queue_ms=3.5,
            handler_ms=12.25,
            cache_hit=True,
            cache_type="stockbot_parsed",
        )
    # Reopen proves the migration is idempotent.
    with _recorder("run-tel-2"):
        pass
    conn = sqlite3.connect(str(path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
        assert {"protocol_id", "bridge_queue_ms", "handler_ms", "cache_hit", "cache_type"} <= cols
        row = conn.execute(
            "SELECT protocol_id, bridge_queue_ms, handler_ms, cache_hit, cache_type"
            " FROM tool_calls WHERE tool_call_id = 'run-tel-1:tc:1'"
        ).fetchone()
        assert row == ("proto-1", 3.5, 12.25, 1, "stockbot_parsed")
    finally:
        conn.close()


def test_execute_agent_tool_ids_and_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pi-supplied call IDs become run-scoped rows; handler/queue/cache persist."""
    import app.tool_runtime as gateway

    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))

    def _fake_execute_tool(
        name: str, arguments: dict[str, object], model: str, *, context: RequestContext
    ) -> dict[str, object]:
        return {"ok": True, "cache_hit": True, "cache_type": "unit_test"}

    monkeypatch.setattr(gateway, "execute_tool", _fake_execute_tool)
    session = gateway.RuntimeToolSession(session_id="s1")
    with _recorder("run-pi-1") as recorder:
        token = set_current_recorder(recorder)
        try:
            fallback = gateway.execute_agent_tool("search_tools", {"query": "telemetry probe"}, session)
            assert fallback.get("content")
            correlated = gateway.execute_agent_tool(
                "search_tools",
                {"query": "telemetry probe"},
                session,
                tool_call_id="call-9",
                protocol_id="proto-9",
                bridge_queue_ms=7.5,
            )
            assert correlated.get("content")
            content = correlated["content"]
            assert isinstance(content, str)
            assert "proto-9" not in content
        finally:
            reset_current_recorder(token)
    conn = sqlite3.connect(str(tmp_path / "runs.sqlite"))
    try:
        rows = {
            row[0]: row[1:]
            for row in conn.execute(
                "SELECT tool_call_id, protocol_id, bridge_queue_ms, handler_ms, cache_hit, cache_type FROM tool_calls"
            )
        }
        seq_id, pi_id = "run-pi-1:tc:1", "run-pi-1:tc:call-9"
        assert rows[seq_id][0] is None
        assert rows[seq_id][1] == 0.0
        assert rows[seq_id][2] is not None and rows[seq_id][2] >= 0.0
        assert tuple(rows[seq_id][3:]) == (1, "unit_test")
        assert rows[pi_id][:3] == ("proto-9", 7.5, rows[pi_id][2])
        assert rows[pi_id][2] is not None and rows[pi_id][2] >= 0.0
        assert tuple(rows[pi_id][3:]) == (1, "unit_test")
    finally:
        conn.close()


def test_finalize_failed_run_reconstructs_orphan_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Orphaned runs terminalize as failed with aggregates rebuilt from child rows."""
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    now = datetime.now(UTC).isoformat()
    with _recorder("run-orphan-pop") as recorder:
        recorder.record_model_call(
            round=2,
            provider="p",
            model="t",
            started_at=now,
            completed_at=now,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 999, "cost": 0.25},
        )
        recorder.record_tool_call(
            tool_call_id="tc-1",
            round=3,
            tool_name="t",
            arguments_json="{}",
            started_at=now,
            completed_at=now,
            status="completed",
            result_row_count=1,
            returned_count=1,
            truncated=False,
            result_bytes=10,
            result_hash="h",
            source_names="",
            source_freshness="",
            as_of=None,
            error_type=None,
            error_message=None,
        )
        recorder.record_event("turn_end", round=1)
    with _recorder("run-orphan-empty"):
        pass
    assert finalize_failed_run("run-orphan-pop", error_type="tool_timeout", error_message="boom") is True
    assert finalize_failed_run("run-orphan-empty", error_type="tool_timeout", error_message="boom") is True
    conn = sqlite3.connect(str(tmp_path / "runs.sqlite"))
    try:
        cols = (
            "status, completed_at, duration_ms, round_count, model_call_count,"
            " tool_call_count, input_tokens, output_tokens, total_tokens,"
            " estimated_model_cost, estimated_total_cost, error_type"
        )
        pop = conn.execute(f"SELECT {cols} FROM agent_runs WHERE run_id = ?", ("run-orphan-pop",)).fetchone()
        assert pop[0] == "failed"
        assert pop[1] is not None and pop[2] is not None
        assert tuple(pop[3:]) == (3, 1, 1, 100, 50, 150, 0.25, 0.25, "tool_timeout")
        empty = conn.execute(f"SELECT {cols} FROM agent_runs WHERE run_id = ?", ("run-orphan-empty",)).fetchone()
        assert empty[0] == "failed"
        assert tuple(empty[3:]) == (0, 0, 0, 0, 0, 0, 0.0, 0.0, "tool_timeout")
    finally:
        conn.close()
