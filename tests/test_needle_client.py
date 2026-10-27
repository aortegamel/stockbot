"""Persistent Needle worker: one process reused, IDs correlated, fail closed."""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import ClassVar, override

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

import app.needle_client as nc


def _request_id(raw_line: str) -> object:
    decoded: object = json.loads(raw_line)
    assert isinstance(decoded, dict)
    return decoded.get("id")


class _FakeProc:
    instances: ClassVar[list[_FakeProc]] = []
    behavior: ClassVar[str] = "echo"

    class _In:
        def __init__(self, proc: _FakeProc) -> None:
            self._proc = proc

        def write(self, s: str) -> None:
            self._proc._write(s)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _Out:
        def __init__(self, proc: _FakeProc) -> None:
            self._proc = proc

        def readline(self) -> str:
            return self._proc._readline()

        def close(self) -> None:
            pass

    def __init__(self, *args: object, **kwargs: object) -> None:
        _FakeProc.instances.append(self)
        self.cmd: object = args[0] if args else None
        self.written: list[str] = []
        self.stdin: _FakeProc._In | None = _FakeProc._In(self)
        self.stdout: _FakeProc._Out | None = _FakeProc._Out(self)
        self.stderr: io.StringIO | None = io.StringIO("")
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []

    def _write(self, s: str) -> None:
        if _FakeProc.behavior == "broken-once" and len(_FakeProc.instances) == 1:
            raise BrokenPipeError(32, "Broken pipe")
        if _FakeProc.behavior == "broken-always":
            raise BrokenPipeError(32, "Broken pipe")
        self.written.append(s)

    def _readline(self) -> str:
        raw: object = json.loads(self.written[-1])
        assert isinstance(raw, dict)
        if raw.get("action") == "ping":
            return json.dumps({"id": raw.get("id"), "ready": True}) + "\n"
        if _FakeProc.behavior == "malformed":
            return "not json\n"
        if raw.get("action") == "extract":
            if _FakeProc.behavior == "mismatch":
                return json.dumps({"id": raw.get("id"), "record": "wrong", "fields": {}}) + "\n"
            rec = raw.get("record")
            name = rec.get("name") if isinstance(rec, dict) else None
            return json.dumps({"id": raw.get("id"), "record": name, "fields": {"q": 1}}) + "\n"
        if raw.get("action") == "embed":
            return json.dumps({"id": raw.get("id"), "embedding": [0.1, 0.2]}) + "\n"
        if _FakeProc.behavior == "mismatch":
            return json.dumps({"id": "needle:wrong", "tool": raw.get("tool"), "arguments": {}}) + "\n"
        return json.dumps({"id": raw.get("id"), "tool": raw.get("tool"), "arguments": {"q": 1}}) + "\n"

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_timeouts.append(timeout)
        return self.returncode


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    nc.close()
    _FakeProc.instances.clear()
    _FakeProc.behavior = "echo"
    monkeypatch.setattr(subprocess, "Popen", _FakeProc)

    def _skip_weights() -> None:
        return None

    monkeypatch.setattr(nc, "_ensure_weights", _skip_weights)
    yield
    nc.close()
    _FakeProc.instances.clear()
    _FakeProc.behavior = "echo"


def test_extract_fields_round_trip(worker: None) -> None:
    """extract_fields sends one extract action and returns declared record fields."""
    record: dict[str, object] = {
        "name": "web_row",
        "description": "one row",
        "parameters": {"type": "object"},
    }
    out = nc.extract_fields(record, "NVDA news https://x.example/a up 5%")
    assert out["record"] == "web_row"
    assert out["fields"] == {"q": 1}
    assert len(_FakeProc.instances) == 1
    proc = _FakeProc.instances[0]
    sent: object = json.loads(proc.written[-1])
    assert isinstance(sent, dict) and sent.get("action") == "extract"


def test_embed_round_trip(worker: None) -> None:
    """embed sends one embed action and returns the server vector."""
    vec = nc.embed("NVDA revenue")
    assert vec == [0.1, 0.2]
    proc = _FakeProc.instances[0]
    sent: object = json.loads(proc.written[-1])
    assert isinstance(sent, dict) and sent.get("action") == "embed"


def test_extract_fields_rejects_blank_inputs(worker: None) -> None:
    """Blank record/passage fail before any server write."""
    with pytest.raises(ValueError, match="record"):
        nc.extract_fields({}, "x")
    with pytest.raises(ValueError, match="passage"):
        nc.extract_fields({"name": "r"}, "  ")
    assert _FakeProc.instances == []


def test_extract_fields_record_mismatch_fails_closed(worker: None) -> None:
    """Server answering a different record name is a malformed response."""
    _FakeProc.behavior = "mismatch"
    with pytest.raises(RuntimeError, match="malformed"):
        nc.extract_fields({"name": "web_row"}, "some passage")


def test_embed_round_trip_rejects_blank(worker: None) -> None:
    """Blank embed text fails before spawn; server shape covered by exchange unit."""
    with pytest.raises(ValueError, match="text"):
        nc.embed("  ")
    assert _FakeProc.instances == []


def test_generate_arguments_transmits_only_execution_inputs(worker: None) -> None:
    """generateArguments sends request metadata plus explicit arguments; raw records never enter."""
    marker = "MARKER-" + "x" * 4096
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"identifier": {"type": "string"}, "ticker": {"type": "string"}, "tags": {"type": "array"}},
    }
    out = nc.generate_arguments(
        tool="list_sec_filings",
        schema=schema,
        objective="objective text",
        node={
            "question": "What ticker?",
            "node_id": "n1",
            "id": "i1",
            "type": "t",
            "evidence": [marker],
            "attempts": [marker],
            "feedback": marker,
            "intake": marker,
            "reasoning": marker,
        },
        context={
            "as_of": "2026-01-01",
            "today_utc": "2026-10-12",
            "ticker": "AAPL",
            "temporal_scope": {
                "mode": "latest-available",
                "as_of": "2026-01-01",
                "start": "2026-01-01",
                "end": "2026-02-01",
                "raw": marker,
            },
            "grounded_arguments": {
                "identifier": "AAPL",
                "ticker": "AAPL",
                "tags": ["a", 1, None],
                "nested": {"a": 1},
                "evil": marker,
            },
            "evidence": [{"evidence_id": "ev-1", "content": marker}],
            "attempts": [{"result": marker}],
            "grounding_hint": "do not send",
            "feedback": marker,
            "intake": marker,
            "reasoning": marker,
        },
    )
    assert out["tool"] == "list_sec_filings"
    proc = _FakeProc.instances[0]
    sent: object = json.loads(proc.written[-1])
    assert isinstance(sent, dict)
    assert set(sent) == {"id", "action", "tool", "schema", "objective", "node", "context"}
    assert sent["node"] == {"question": "What ticker?", "node_id": "n1", "id": "i1", "type": "t"}
    assert sent["context"] == {
        "as_of": "2026-01-01",
        "today_utc": "2026-10-12",
        "ticker": "AAPL",
        "temporal_scope": {
            "mode": "latest-available",
            "as_of": "2026-01-01",
            "start": "2026-01-01",
            "end": "2026-02-01",
        },
        "grounded_arguments": {"identifier": "AAPL", "ticker": "AAPL", "tags": ["a", 1, None]},
    }
    assert marker not in json.dumps(sent, ensure_ascii=False)
    assert "do not send" not in json.dumps(sent, ensure_ascii=False)


def test_start_ping_then_two_calls_reuse_one_process(worker: None) -> None:
    nc.start()
    nc.start()
    assert len(_FakeProc.instances) == 1
    proc = _FakeProc.instances[0]
    assert isinstance(proc.cmd, list) and str(proc.cmd[1]).endswith("server.py")
    ping_raw: object = json.loads(proc.written[0])
    assert isinstance(ping_raw, dict)
    assert ping_raw.get("action") == "ping" and isinstance(ping_raw.get("id"), str)

    first = nc.generate_arguments(tool="search_sec_filings", schema={}, objective="o", node="n", context={})
    assert first["withheld"] is False
    second = nc.generate_arguments(
        {
            "tool": "search_sec_filings",
            "schema": {},
            "objective": "o",
            "node": "n",
            "context": {},
        }
    )
    assert first == {
        "tool": "search_sec_filings",
        "arguments": {"q": 1},
        "reasoning": "",
        "confidence": None,
        "withheld": False,
    }
    assert second == first
    assert len(_FakeProc.instances) == 1
    ids = [_request_id(w) for w in proc.written[1:]]
    assert len(ids) == 2 and ids[0] != ids[1]


def test_mismatched_id_fails_closed(worker: None) -> None:
    _FakeProc.behavior = "mismatch"
    with pytest.raises(RuntimeError, match="malformed"):
        nc.generate_arguments(tool="search_sec_filings")
    assert len(_FakeProc.instances) == 1


def test_malformed_response_fails_closed(worker: None) -> None:
    _FakeProc.behavior = "malformed"
    with pytest.raises(RuntimeError, match="malformed"):
        nc.generate_arguments(tool="search_sec_filings")


def test_broken_pipe_restarts_once(worker: None) -> None:
    _FakeProc.behavior = "broken-once"
    out = nc.generate_arguments(tool="search_sec_filings")
    assert out == {
        "tool": "search_sec_filings",
        "arguments": {"q": 1},
        "reasoning": "",
        "confidence": None,
        "withheld": False,
    }
    assert len(_FakeProc.instances) == 2


@pytest.mark.parametrize("operation", ["generate", "start", "extract", "embed"])
def test_startup_timeout_reports_startup_budget(worker: None, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    assert nc._TIMEOUT_S == 120.0
    assert not hasattr(nc, "_GENERATE_TIMEOUT_S")

    def silent_startup(proc: _FakeProc, timeout_s: float) -> str:
        request = json.loads(proc.written[-1])
        assert request["action"] == "ping"
        assert timeout_s == 120.0
        raise TimeoutError("startup silence")

    monkeypatch.setattr(nc, "_readline", silent_startup)
    with pytest.raises(RuntimeError, match="startup timed out after 120s"):
        if operation == "generate":
            nc.generate_arguments(tool="search_sec_filings")
        elif operation == "start":
            nc.start()
        elif operation == "extract":
            nc.extract_fields({"name": "web_row"}, "some passage")
        else:
            nc.embed("some passage")
    proc = _FakeProc.instances[0]
    assert proc.returncode is not None
    assert proc.wait_timeouts and all(timeout is not None for timeout in proc.wait_timeouts)
    assert nc._PROC is None


def test_generation_timeout_reaps_worker_then_recovers(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_first_worker(proc: _FakeProc, timeout_s: float) -> str:
        request = json.loads(proc.written[-1])
        if request["action"] == "ping":
            assert timeout_s == 120.0
            return proc._readline()
        # No generation ceiling: generation reads share the worker timeout.
        assert timeout_s == nc._TIMEOUT_S
        if len(_FakeProc.instances) == 1:
            raise TimeoutError("generation silence")
        return proc._readline()

    def reject_group_signal(pgid: int, sig: int) -> None:
        pytest.fail(f"The client signaled an unowned fake process group: {pgid}, {sig}")

    monkeypatch.setattr(nc, "_readline", timeout_first_worker)
    monkeypatch.setattr(os, "killpg", reject_group_signal)
    with pytest.raises(RuntimeError, match="arguments.generate timed out after 120s"):
        nc.generate_arguments(tool="search_sec_filings")
    first = _FakeProc.instances[0]
    assert first.returncode is not None
    assert first.wait_timeouts and all(timeout is not None for timeout in first.wait_timeouts)
    assert nc._PROC is None
    assert nc.generate_arguments(tool="search_sec_filings")["arguments"] == {"q": 1}
    assert len(_FakeProc.instances) == 2
    actions = [json.loads(line)["action"] for line in _FakeProc.instances[1].written]
    assert actions == ["ping", "arguments.generate"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups and /proc are required")
@pytest.mark.parametrize("shutdown", ["timeout", "close"])
def test_real_worker_cleanup_kills_owned_descendants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shutdown: str
) -> None:
    nc.close()
    descendants_file = tmp_path / "descendants.json"
    server = tmp_path / "stub_server.py"
    server.write_text(
        "import json, os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child_code = '''import json, os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        "with open(sys.argv[1], 'w') as stream:\n"
        "    json.dump([os.getpid(), grandchild.pid], stream)\n"
        "time.sleep(60)\n"
        "'''\n"
        f"child = subprocess.Popen([sys.executable, '-c', child_code, {str(descendants_file)!r}])\n"
        f"while not os.path.exists({str(descendants_file)!r}):\n"
        "    time.sleep(0.005)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['action'] == 'ping':\n"
        "        print(json.dumps({'id': request['id'], 'ready': True}), flush=True)\n"
        "    else:\n"
        "        time.sleep(60)\n"
    )
    monkeypatch.setattr(nc, "_server", lambda: server)
    monkeypatch.setattr(nc, "_python", lambda: sys.executable)
    monkeypatch.setattr(nc, "_ensure_weights", lambda: None)
    proc: subprocess.Popen[str] | None = None
    descendants: list[int] = []
    try:
        nc.start()
        if shutdown == "timeout":
            # Shrink the shared worker timeout only after the real ping succeeded.
            monkeypatch.setattr(nc, "_TIMEOUT_S", 0.05)
        proc = nc._PROC
        assert proc is not None
        descendants = json.loads(descendants_file.read_text())
        assert len(descendants) == 2
        assert os.getpgid(proc.pid) == proc.pid
        assert all(os.getpgid(pid) == proc.pid for pid in descendants)
        if shutdown == "timeout":
            with pytest.raises(RuntimeError, match="arguments.generate timed out after 0.05s"):
                nc.generate_arguments(tool="search_sec_filings")
        else:
            started = time.monotonic()
            nc.close()
            assert time.monotonic() - started < 2.0
        assert nc._PROC is None
        assert proc.returncode is not None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            live = []
            for pid in descendants:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
                except FileNotFoundError:
                    continue
                if state != "Z":
                    live.append(pid)
            if not live:
                break
            time.sleep(0.005)
        assert not live, f"Owned descendants survived: {live}"
    finally:
        if not descendants and descendants_file.exists():
            descendants = json.loads(descendants_file.read_text())
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc is not None:
            proc.kill()
            proc.wait(timeout=2.0)
        nc.close()


def _stub_needle(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a scriptable needle stub module (the venv has no needle package)."""
    import sys

    stub = types.ModuleType("needle")

    class _FakeNeedle:
        instances: ClassVar[list[object]] = []
        calls: ClassVar[list[tuple[str, object]]] = []
        envelope: ClassVar[dict[str, object]] = {"type": "none"}
        vector: ClassVar[object] = [0.1, 0.2, 0.3]

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            _FakeNeedle.instances.append(self)

        def complete(self, text: str, max_new_tokens: int = 512) -> dict[str, object]:
            _FakeNeedle.calls.append(("complete", (text, max_new_tokens)))
            return dict(_FakeNeedle.envelope)

        def reset(self) -> None:
            _FakeNeedle.calls.append(("reset", None))

        def close(self) -> None:
            _FakeNeedle.calls.append(("close", None))

        def embed(self, text: str) -> object:
            assert text
            _FakeNeedle.calls.append(("embed", text))
            return _FakeNeedle.vector

    monkeypatch.setattr(stub, "Needle", _FakeNeedle, raising=False)
    monkeypatch.setitem(sys.modules, "needle", stub)
    return _FakeNeedle


def _import_server() -> types.ModuleType:
    import importlib.util

    # Match the module key that mutmut derives from the source path.
    spec = importlib.util.spec_from_file_location(
        "needle-harness.lib.needle.server", "needle-harness/lib/needle/server.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CATALOG = [
    {
        "name": "get_sec_filing",
        "description": (
            "Returns one filing's record (filer, subject when known, form, filed/accepted/known "
            "dates, period, primary document, amendment link, source URL) by accession number. "
            "When the accession number is unknown, find it with list_sec_filings or "
            "search_sec_filings first; never invent it from a ticker or company name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "accession_no": {
                    "type": "string",
                    "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    "pattern": "^\\d{10}-\\d{2}-\\d{6}$",
                },
                "as_of": {
                    "type": "string",
                    "description": "Point-in-time date YYYY-MM-DD.",
                },
            },
            "required": ["accession_no"],
        },
    },
    {
        "name": "get_reg_sho_volume",
        "description": "Self-contained daily short-sale volume by venue for one ticker over the rolling 12 months.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "tradeDate": {
                    "type": "string",
                    "description": "Optional trade date YYYY-MM-DD.",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                },
            },
            "required": ["ticker"],
        },
    },
]


def _load_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """Import server.py with the needle extension stubbed (venv has no needle pkg)."""
    _stub_needle(monkeypatch)
    path = tmp_path / ".needle-catalog.json"
    path.write_text(json.dumps(_CATALOG))
    monkeypatch.setenv("NEEDLE_CATALOG", str(path))
    return _import_server()


def _import_server_with_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str | None) -> types.ModuleType:
    """Import server.py against one catalog file; None leaves the file missing."""
    _stub_needle(monkeypatch)
    path = tmp_path / ".needle-catalog.json"
    if text is not None:
        path.write_text(text)
    monkeypatch.setenv("NEEDLE_CATALOG", str(path))
    return _import_server()


def test_bound_entry_keeps_full_params_and_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    schema = {
        "type": "object",
        "properties": {
            "accession_no": {
                "type": "string",
                "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                "pattern": "^\\d{10}-\\d{2}-\\d{6}$",
            },
            "tradeDate": {
                "type": "string",
                "description": "Optional trade date YYYY-MM-DD.",
                "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            },
        },
        "required": ["accession_no"],
    }
    entry = srv._tool_entry("get_sec_filing", schema)
    props = entry["parameters"]["properties"]
    assert "Named accession_no, not accession_number" in props["accession_no"]["description"]
    assert props["accession_no"]["pattern"] == "^\\d{10}-\\d{2}-\\d{6}$"
    assert props["tradeDate"]["pattern"] == "^\\d{4}-\\d{2}-\\d{2}$"
    assert len(entry["description"]) > 150  # full catalog text, not the TOOLS slim
    fallback = srv._tool_entry("get_sec_filing", {})
    assert (
        "Named accession_no, not accession_number"
        in fallback["parameters"]["properties"]["accession_no"]["description"]
    )


def test_bound_entry_carries_no_trigger_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bound tool entries never carry triggers; the native SDK decides call versus withhold."""
    srv = _load_server(monkeypatch, tmp_path)
    assert "triggers" not in srv._tool_entry("get_sec_filing", {})
    assert "triggers" not in srv._tool_entry("research_resume", {})


def test_bound_system_carries_temporal_decoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    system = srv._bound_system()
    assert "Today UTC is" in system and "decode relative dates before choosing" in system


def test_legacy_global_catalog_stays_slim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    assert all(len(t["description"]) <= 150 for t in srv.TOOLS)
    assert "description" not in json.dumps([t["parameters"] for t in srv.TOOLS])
    assert srv._strip_descriptions({"a": {"description": "x", "type": "string"}}) == {"a": {"type": "string"}}


def test_extract_agent_binds_single_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Extraction binds exactly one record shape, never the registry."""
    srv = _load_server(monkeypatch, tmp_path)
    record = {
        "name": "web_row",
        "description": "one row",
        "parameters": {"type": "object"},
    }
    agent = srv._extract_agent(record, None)
    tools = agent.kwargs["tools"]
    assert isinstance(tools, list) and tools == [record]


def test_extract_agent_rejects_bad_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="bad record"):
        srv._extract_agent({}, None)
    with pytest.raises(ValueError, match="bad record"):
        srv._extract_agent({"name": ""}, None)


def test_extract_decision_reads_withheld_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Withheld record lands in suppressed_calls; extract reads it (unlike args path)."""
    srv = _load_server(monkeypatch, tmp_path)
    r: dict[str, object] = {
        "function_calls": [],
        "suppressed_calls": [{"name": "web_row", "arguments": {"u": "x"}}],
    }
    out = srv._extract_decision("web_row", r)
    assert out["record"] == "web_row" and out["fields"] == {"u": "x"} and out["withheld"] is True


def test_extract_decision_empty_is_null_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    out = srv._extract_decision("web_row", {"function_calls": [], "suppressed_calls": []})
    assert out["record"] is None and out["fields"] == {} and out["withheld"] is False


def test_extract_decision_mismatch_and_ungrounded_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        srv._extract_decision("web_row", {"function_calls": [{"name": "other", "arguments": {}}]})
    bad: dict[str, object] = {
        "function_calls": [{"name": "web_row", "arguments": {}}],
        "validation": {"ungrounded": ["web_row.u"]},
    }
    with pytest.raises(ValueError, match="not grounded"):
        srv._extract_decision("web_row", bad)
    ok = srv._extract_decision("web_row", bad, strict=False)
    assert ok["record"] == "web_row"


def test_handle_extract_rejects_bad_shapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bad record/passage/max tokens fail closed before any decode."""
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)
    rec = {"name": "web_row"}
    bad_passage = _json.dumps({"id": "e1", "action": "extract", "record": rec, "passage": "  "})
    assert srv.handle(bad_passage)["error"] == "bad passage"
    bad_record = _json.dumps({"id": "e2", "action": "extract", "record": {}, "passage": "x"})
    assert srv.handle(bad_record)["error"] == "bad record"
    bad_tokens = _json.dumps(
        {
            "id": "e3",
            "action": "extract",
            "record": rec,
            "passage": "x",
            "max_new_tokens": 0,
        }
    )
    assert srv.handle(bad_tokens)["error"] == "bad max_new_tokens"


def test_handle_embed_rejects_blank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)
    out = srv.handle(_json.dumps({"id": "m1", "action": "embed", "text": "  "}))
    assert out["error"] == "bad text"


def test_handle_embed_returns_vector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)
    out = srv.handle(_json.dumps({"id": "m2", "action": "embed", "text": "NVDA revenue"}))
    assert out["embedding"] == [0.1, 0.2, 0.3]


def test_noncall_detail_plain_and_suppressed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_noncall_detail pins type/suppressed/confidence/reasoning in one line."""
    srv = _load_server(monkeypatch, tmp_path)
    plain = srv._noncall_detail({"type": "noul", "suppressed_calls": [], "confidence": 0.5, "reasoning": "why"})
    assert "noul" in plain and "0.5" in plain and "why" in plain
    supp = srv._noncall_detail(
        {
            "type": "withheld",
            "suppressed_calls": [{"name": "get_sec_filing"}],
            "confidence": 0.9,
            "reasoning": "  a  b  ",
        }
    )
    assert "get_sec_filing" in supp and "0.9" in supp


def test_handle_mismatch_error_carries_detail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Needle None mismatch error carries type/suppressed/confidence/reasoning."""
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)

    class _R:
        def complete(self, prompt: str, max_new_tokens: int = 0) -> dict[str, object]:
            return {
                "type": "noul",
                "function_calls": [],
                "suppressed_calls": [{"name": "x"}],
                "confidence": 0.4,
                "reasoning": "need grounding",
            }

    class _B:
        def complete(self, prompt: str, max_new_tokens: int = 0) -> dict[str, object]:
            return _R().complete(prompt, max_new_tokens)

        def close(self) -> None:
            pass

    def _bound(tool: str, schema: dict[str, object], facts: str) -> _B:
        return _B()

    monkeypatch.setattr(srv, "_bound_agent", _bound)
    line = _json.dumps(
        {
            "id": "m1",
            "action": "arguments.generate",
            "tool": "search_sec_filings",
            "schema": {},
            "objective": "o",
        }
    )
    out = srv.handle(line)
    assert "error" in out
    err = str(out["error"])
    assert "mismatch" in err and "suppressed" in err and "confidence" in err and "reasoning" in err


def test_handle_withheld_same_name_returns_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validated suppressed same-name decision reaches the response (tool + withheld:true)."""
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)

    class _B:
        def complete(self, prompt: str, max_new_tokens: int = 0) -> dict[str, object]:
            return {
                "type": "withheld",
                "function_calls": [],
                "suppressed_calls": [{"name": "search_sec_filings", "arguments": {"q": "x"}}],
                "confidence": 0.8,
                "reasoning": "withheld grounding",
            }

        def close(self) -> None:
            pass

    def _bound(tool: str, schema: dict[str, object], facts: str) -> _B:
        return _B()

    monkeypatch.setattr(srv, "_bound_agent", _bound)
    line = _json.dumps(
        {
            "id": "w1",
            "action": "arguments.generate",
            "tool": "search_sec_filings",
            "schema": {},
            "objective": "o",
        }
    )
    out = srv.handle(line)
    assert out["tool"] == "search_sec_filings" and out["withheld"] is True
    assert out["arguments"] == {"q": "x"}


def test_handle_rejects_engine_failure_ungrounded_and_missing_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Engine failure, SDK grounding flags, or no request text never return arguments."""
    srv = _load_server(monkeypatch, tmp_path)
    clean: dict[str, object] = {
        "type": "call",
        "success": True,
        "error": None,
        "function_calls": [{"name": "list_sec_filings", "arguments": {"identifier": "AAPL"}}],
        "suppressed_calls": [],
        "confidence": 0.9,
        "reasoning": "AAPL from the question",
    }
    envelopes: dict[str, dict[str, object]] = {
        "clean": clean,
        "engine": {"type": "error", "error": "needle_init not called"},
        "ungrounded": {
            **clean,
            "validation": {
                "ungrounded": ["list_sec_filings.identifier"],
                "negation": False,
            },
        },
        "negated": {**clean, "validation": {"ungrounded": [], "negation": True}},
    }
    envelope: dict[str, object] = clean

    class _B:
        def complete(self, text: str, max_new_tokens: int = 0) -> dict[str, object]:
            return envelope

        def close(self) -> None:
            pass

    def _bound(tool: str, schema: dict[str, object], facts: str) -> _B:
        return _B()

    monkeypatch.setattr(srv, "_bound_agent", _bound)

    def _generate(rid: str, question: str) -> dict[str, object]:
        line = json.dumps(
            {
                "id": rid,
                "action": "arguments.generate",
                "tool": "list_sec_filings",
                "node": {"question": question},
            }
        )
        out: dict[str, object] = srv.handle(line)
        return out

    for name, env in envelopes.items():
        envelope = env
        out = _generate(name, "AAPL 10-Q and 10-K filings")
        if name == "clean":
            assert out["tool"] == "list_sec_filings" and out["arguments"] == {"identifier": "AAPL"}
        else:
            assert "error" in out and "arguments" not in out, name
    envelope = clean
    blank = _generate("blank", "  ")
    assert "error" in blank and "arguments" not in blank


_ARGS = st.dictionaries(st.text(max_size=8), st.integers() | st.text(max_size=8), max_size=3)


def test_prop_flagged_or_failed_envelope_never_returns_arguments() -> None:
    """Any ungrounded field, negation, or engine-failure marker yields an error and no arguments."""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as tmp:
        srv = _load_server(mp, Path(tmp))
        envelope: dict[str, object] = {}

        def _complete(self: object, text: str, max_new_tokens: int = 0) -> dict[str, object]:
            return envelope

        mp.setattr(srv.needle.Needle, "complete", _complete)

        @seed(20261011)
        @settings(max_examples=200, derandomize=True)
        @given(
            _ARGS,
            st.lists(st.text(max_size=20), max_size=3),
            st.booleans(),
            st.sampled_from(["none", "type", "success", "error"]),
            st.booleans(),
        )
        def check(
            args: dict[str, object],
            ungrounded: list[str],
            negation: bool,
            failure: str,
            withheld: bool,
        ) -> None:
            nonlocal envelope
            call = [{"name": "list_sec_filings", "arguments": args}]
            envelope = {
                "type": "error" if failure == "type" else "call",
                "success": failure != "success",
                "error": "engine failed" if failure == "error" else None,
                "function_calls": [] if withheld else call,
                "suppressed_calls": call if withheld else [],
                "validation": {"ungrounded": ungrounded, "negation": negation},
            }
            line = json.dumps(
                {
                    "id": "p",
                    "action": "arguments.generate",
                    "tool": "list_sec_filings",
                    "node": {"question": "q"},
                }
            )
            out = srv.handle(line)
            if ungrounded or negation or failure != "none":
                assert "error" in out and "arguments" not in out and "tool" not in out
            else:
                assert out["arguments"] == args and out["withheld"] is withheld

        check()


def test_decision_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_decision: call, suppressed same-name (withheld), suppressed other-name, plain."""
    srv = _load_server(monkeypatch, tmp_path)
    call = srv._decision({"type": "call", "function_calls": [{"name": "t", "arguments": {"a": 1}}]}, "t")
    assert call["tool"] == "t" and call["arguments"] == {"a": 1} and call["withheld"] is False
    same = srv._decision(
        {
            "type": "withheld",
            "function_calls": [],
            "suppressed_calls": [{"name": "t", "arguments": {"a": 2}}],
        },
        "t",
    )
    assert same["tool"] == "t" and same["arguments"] == {"a": 2} and same["withheld"] is True
    other = srv._decision(
        {
            "type": "withheld",
            "function_calls": [],
            "suppressed_calls": [{"name": "x", "arguments": {}}],
        },
        "t",
    )
    assert other["tool"] is None and other["withheld"] is False
    empty = srv._decision({"type": "call", "function_calls": [], "suppressed_calls": []}, "t")
    assert empty["tool"] is None and empty["withheld"] is False
    missing = srv._decision({"type": "call"}, "t")
    assert missing["tool"] is None and missing["withheld"] is False


def test_generate_withheld_accept_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client accepts a withheld same-name call and retains the shared worker timeout."""
    import json as _json

    nc.close()
    _FakeProc.instances.clear()
    _FakeProc.behavior = "echo"

    def _fake_readline(proc: _FakeProc, timeout_s: float) -> str:
        assert proc is _FakeProc.instances[-1]
        raw: object = _json.loads(proc.written[-1])
        assert isinstance(raw, dict)
        if raw.get("action") == "ping":
            return _json.dumps({"id": raw.get("id"), "ready": True}) + "\n"
        return (
            _json.dumps(
                {
                    "id": raw.get("id"),
                    "tool": raw.get("tool"),
                    "arguments": {"q": 1},
                    "withheld": True,
                }
            )
            + "\n"
        )

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(nc, "_ensure_weights", lambda: None)
    monkeypatch.setattr(nc, "_readline", _fake_readline)
    out = nc.generate_arguments(tool="search_sec_filings")
    assert out["withheld"] is True
    assert out["tool"] == "search_sec_filings"
    nc.close()
    _FakeProc.instances.clear()

    def _slow(proc: _FakeProc, timeout_s: float) -> str:
        raw2: object = _json.loads(proc.written[-1])
        assert isinstance(raw2, dict)
        if raw2.get("action") == "ping":
            assert timeout_s == nc._TIMEOUT_S
            return _json.dumps({"id": raw2.get("id"), "ready": True}) + "\n"
        assert timeout_s == nc._TIMEOUT_S
        raise TimeoutError("slow")

    monkeypatch.setattr(nc, "_readline", _slow)
    with pytest.raises(RuntimeError, match="timed out after 120s"):
        nc.generate_arguments(tool="search_sec_filings")
    nc.close()
    _FakeProc.instances.clear()
    _FakeProc.behavior = "echo"


def test_arguments_facts_keep_only_execution_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bound facts keep request metadata plus explicit arguments; raw records never bind."""
    srv = _load_server(monkeypatch, tmp_path)
    marker = "MARKER-" + "x" * 4096
    schema: dict[str, object] = {"type": "object", "properties": {"identifier": {}, "ticker": {}, "tags": {}}}
    text = srv._arguments_facts(
        "objective",
        {
            "question": "q",
            "node_id": "n1",
            "id": "i1",
            "type": "t",
            "evidence": [marker],
            "attempts": [marker],
            "feedback": marker,
            "intake": marker,
            "reasoning": marker,
        },
        {
            "as_of": "2026-01-01",
            "today_utc": "2026-10-12",
            "ticker": "AAPL",
            "temporal_scope": {
                "mode": "latest-available",
                "as_of": "2026-01-01",
                "start": "2026-01-01",
                "end": "2026-02-01",
                "raw": marker,
            },
            "grounded_arguments": {
                "identifier": "AAPL",
                "ticker": "AAPL",
                "tags": ["a", 1, None],
                "nested": {"a": 1},
                "evil": marker,
            },
            "evidence": [{"evidence_id": "ev-1", "content": marker}],
            "attempts": [{"result": marker}],
            "grounding_hint": "do not send",
            "feedback": marker,
            "intake": marker,
            "reasoning": marker,
        },
        schema,
    )
    assert "node node_id: n1" in text and "node id: i1" in text and "node type: t" in text
    assert "as_of: 2026-01-01" in text and "ticker: AAPL" in text
    assert "temporal_scope mode: latest-available" in text and "temporal_scope start: 2026-01-01" in text
    assert "argument identifier: AAPL" in text and "argument tags:" in text
    assert marker not in text and "do not send" not in text
    assert "evidence" not in text and "attempt" not in text


def test_arguments_facts_reject_nonfinite_and_nested_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """NaN, Infinity, mappings, and off-schema values never bind as Needle facts."""
    srv = _load_server(monkeypatch, tmp_path)
    schema = {"type": "object", "properties": {"count": {"type": "integer"}, "rows": {"type": "array"}}}
    text = srv._arguments_facts(
        "objective",
        {"node_id": "n1"},
        {"grounded_arguments": {"count": float("nan"), "rows": [{"a": 1}], "evil": "x"}},
        schema,
    )
    assert "argument count" not in text and "argument rows" not in text and "evil" not in text


def _production_schema(name: str) -> dict[str, object]:
    """Exact parameters the kernel binds for one production tool."""
    from app.tools import TOOLS

    for tool in TOOLS:
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name") == name and isinstance(fn.get("parameters"), dict):
            return fn["parameters"]
    raise AssertionError(f"no production schema for {name}")


_TSLA = "What did TSLA file recently?"
_TESLA = "Quick lookup of Tesla"
_LATEST_5 = "Show TSLA's latest 5 filings"
_UP_TO_37 = "Show up to 37 records of TSLA filings"
_NO_COUNT = "TSLA 10-K filings since 2025-01-01 for CIK 0001318605, accession 0001318605-25-000010"


@pytest.mark.parametrize(
    ("tool", "request_text", "emitted", "grounded", "outcome"),
    [
        # User omitted the count: the declared handler default grounds.
        ("list_sec_filings", _TSLA, {"identifier": "TSLA", "limit": 50}, None, "accepted"),
        ("find_sec_entities_bounded", _TESLA, {"query": "Tesla", "limit": 20}, None, "accepted"),
        # Dates, CIKs, accessions, and 10-K are not counts: the default still grounds.
        ("list_sec_filings", _NO_COUNT, {"identifier": "TSLA", "forms": ["10-K"], "limit": 50}, None, "accepted"),
        # Unknown numbers stay rejected.
        ("list_sec_filings", _TSLA, {"identifier": "TSLA", "limit": 37}, None, "not grounded"),
        ("find_sec_entities_bounded", _TESLA, {"query": "Tesla", "limit": 50}, None, "not grounded"),
        # A "(default 20)" description is not a declared default.
        ("find_sec_entities", _TESLA, {"query": "Tesla", "limit": 20}, None, "not grounded"),
        # An explicit grounded limit is authoritative; the default never replaces it.
        ("list_sec_filings", _TSLA, {"identifier": "TSLA", "limit": 50}, {"limit": 10}, "not grounded"),
        ("list_sec_filings", _TSLA, {"identifier": "TSLA", "limit": 10}, {"limit": 10}, "accepted"),
        # An explicit request count is authoritative; the default, another number, or omission never replaces it.
        ("list_sec_filings", _LATEST_5, {"identifier": "TSLA", "limit": 5}, None, "accepted"),
        ("list_sec_filings", _LATEST_5, {"identifier": "TSLA", "limit": 50}, None, "limit"),
        ("list_sec_filings", _LATEST_5, {"identifier": "TSLA"}, None, "asks for 5"),
        ("list_sec_filings", _UP_TO_37, {"identifier": "TSLA", "limit": 37}, None, "accepted"),
        ("list_sec_filings", _UP_TO_37, {"identifier": "TSLA", "limit": 50}, None, "limit"),
        # Plain integers are never silently discarded because of length or year-like digits.
        ("list_sec_filings", "TSLA latest 10000 filings", {"identifier": "TSLA", "limit": 10000}, None, "accepted"),
        ("list_sec_filings", "TSLA latest 10000 filings", {"identifier": "TSLA", "limit": 50}, None, "limit"),
        ("list_sec_filings", "TSLA filings, limit 2000", {"identifier": "TSLA", "limit": 2000}, None, "accepted"),
        ("list_sec_filings", "TSLA filings, limit 2000", {"identifier": "TSLA", "limit": 50}, None, "limit"),
        ("list_sec_filings", "TSLA up to 2000 records", {"identifier": "TSLA", "limit": 2000}, None, "accepted"),
        ("list_sec_filings", "TSLA up to 2000 records", {"identifier": "TSLA", "limit": 50}, None, "limit"),
        # A risk percentage is not a count; the declared record default stays available.
        (
            "list_sec_filings",
            "TSLA filings, position limit of 10% of float",
            {"identifier": "TSLA", "limit": 50},
            None,
            "accepted",
        ),
        (
            "list_sec_filings",
            "TSLA latest 5 filings, position limit of 10 % of float",
            {"identifier": "TSLA", "limit": 5},
            None,
            "accepted",
        ),
        # A record count never overrides the character-window limit on a document read.
        (
            "get_sec_document",
            "Summarize risk factors in TSLA latest 2 10-K filings",
            {"accession_no": "0001318605-25-000010", "limit": 12000},
            {"accession_no": "0001318605-25-000010", "limit": 12000},
            "accepted",
        ),
        # Conflicting counts fail before the engine runs.
        ("list_sec_filings", f"{_LATEST_5}, up to 37 records", {"identifier": "TSLA", "limit": 5}, None, "conflicting"),
        ("list_sec_filings", _LATEST_5, {"identifier": "TSLA", "limit": 5}, {"limit": 10}, "conflicting"),
        # Compound, separated, or unsupported number words fail before the engine instead of a partial count.
        (
            "list_sec_filings",
            "TSLA latest one hundred filings",
            {"identifier": "TSLA", "limit": 1},
            None,
            "plain digits",
        ),
        (
            "list_sec_filings",
            "TSLA latest thirty five filings",
            {"identifier": "TSLA", "limit": 30},
            None,
            "plain digits",
        ),
        (
            "list_sec_filings",
            "List TSLA filings, limit 1,000",
            {"identifier": "TSLA", "limit": 1},
            None,
            "plain digits",
        ),
    ],
)
def test_generate_applies_count_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool: str,
    request_text: str,
    emitted: dict[str, object],
    grounded: dict[str, object] | None,
    outcome: str,
) -> None:
    """Structured limit, then explicit request count, then declared default; anything else fails."""
    srv = _load_server(monkeypatch, tmp_path)
    engines: list[object] = []

    class _Engine:
        def __init__(self, **kwargs: object) -> None:
            self.system = str(kwargs["system"])
            engines.append(self)

        def complete(self, text: str, max_new_tokens: int = 0) -> dict[str, object]:
            numbers = set(re.findall(r"(?<![\w.,])\d+(?!\d)", f"{text}\n{self.system}"))
            flagged = [
                f"{tool}.{key}"
                for key, value in emitted.items()
                if isinstance(value, int) and str(value) not in numbers
            ]
            return {
                "type": "call",
                "function_calls": [{"name": tool, "arguments": emitted}],
                "validation": {"ungrounded": flagged, "negation": False},
            }

        def close(self) -> None:
            pass

    class _Clock(datetime):
        @override
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> _Clock:
            return cls(2026, 1, 2, 3, 4, tzinfo=UTC)

    # Fixed clock: the bound date/time facts never license a tested number.
    monkeypatch.setattr(srv, "datetime", _Clock)
    monkeypatch.setattr(srv.needle, "Needle", _Engine)
    context: dict[str, object] = {"grounded_arguments": grounded} if grounded is not None else {}
    out = srv.handle(
        json.dumps(
            {
                "id": "d1",
                "action": "arguments.generate",
                "tool": tool,
                "schema": _production_schema(tool),
                "objective": request_text,
                "node": {"node_id": "n1"},
                "context": context,
            }
        )
    )
    if outcome == "accepted":
        assert out["arguments"] == emitted
    else:
        assert outcome in str(out["error"]) and "arguments" not in out
    assert len(engines) == (0 if outcome in ("conflicting", "plain digits") else 1)


_COUNT_TEMPLATES = (
    "Show {ticker}'s latest {n} filings",
    "List the last {n} 10-K filings for {ticker}",
    "Give me up to {n} records for {ticker}",
    "{ticker}: the {n} most recent Form 4 transactions",
    "Top {n} holders of {ticker}, limit {n}",
    "At most {n} SEC filings for {ticker} since 2025-01-01",
)
_NON_COUNTS = (
    "{ticker} 10-K and 10-Q filings",
    "{ticker} filings since 2025-01-01",
    "{ticker} filings for CIK 0001318605",
    "{ticker} accession 0001318605-25-000010",
    "last 3 years of {ticker} filings",
    "{ticker} 8-K filings in 2024",
    "latest 2024 filings for {ticker}",
    "{ticker} filings with a position limit of 10% of float",
    "{ticker} filings with a position limit of 10 percent of float",
)


def test_prop_explicit_counts_ignore_dates_ciks_and_forms() -> None:
    """A count phrase yields exactly its number; dates, CIKs, accessions, and form names never yield one."""
    schema = _production_schema("list_sec_filings")
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as tmp:
        srv = _load_server(mp, Path(tmp))

        @seed(20261013)
        @settings(max_examples=200, derandomize=True)
        @given(
            st.one_of(st.integers(min_value=1, max_value=1899), st.integers(min_value=2100, max_value=100000)),
            st.sampled_from(_COUNT_TEMPLATES),
            st.sampled_from(_NON_COUNTS),
            st.sampled_from(["TSLA", "AMZN", "META"]),
        )
        def check(n: int, counted: str, plain: str, ticker: str) -> None:
            text = plain.format(ticker=ticker)
            assert srv._explicit_counts(text) == set()
            assert "default limit: 50" in srv._arguments_facts(text, None, {}, schema)
            assert srv._explicit_counts(f"{text}. {counted.format(ticker=ticker, n=n)}") == {n}

        check()


def test_handle_passes_raw_request_text_to_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bound complete receives the raw question, else the raw objective; never a JSON wrapper."""
    srv = _load_server(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    class _B:
        def complete(self, text: str, max_new_tokens: int = 0) -> dict[str, object]:
            seen["text"] = text
            return {
                "type": "call",
                "success": True,
                "error": None,
                "function_calls": [{"name": "list_sec_filings", "arguments": {"identifier": "AAPL"}}],
                "suppressed_calls": [],
            }

        def close(self) -> None:
            pass

    def _bound(tool: str, schema: dict[str, object], facts: str) -> _B:
        seen["facts"] = facts
        return _B()

    monkeypatch.setattr(srv, "_bound_agent", _bound)
    question = "List all Form 10-Q filings for AAPL?"
    out = srv.handle(
        json.dumps(
            {
                "id": "q1",
                "action": "arguments.generate",
                "tool": "list_sec_filings",
                "objective": "other objective",
                "node": {"question": question},
            }
        )
    )
    assert out["arguments"] == {"identifier": "AAPL"}
    assert seen["text"] == question
    assert question not in str(seen["facts"])
    out = srv.handle(
        json.dumps(
            {
                "id": "q2",
                "action": "arguments.generate",
                "tool": "list_sec_filings",
                "objective": "fallback objective",
                "node": {},
            }
        )
    )
    assert out["arguments"] == {"identifier": "AAPL"}
    assert seen["text"] == "fallback objective"


@pytest.mark.parametrize("tool", ["get_sec_filing", "get_sec_document", "list_sec_documents"])
def test_generate_exposes_grounded_accession_in_user_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str
) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    accession = "0001045810-26-000075"
    srv.needle.Needle.envelope = {
        "type": "call",
        "function_calls": [{"name": tool, "arguments": {"accession_no": accession}}],
    }
    req = {
        "id": "sec",
        "action": "arguments.generate",
        "tool": tool,
        "schema": _production_schema(tool),
        "objective": "NVDA latest 10-Q risk factors",
        "context": {"grounded_arguments": {"accession_no": accession}},
    }
    assert srv.handle(json.dumps(req))["arguments"] == {"accession_no": accession}
    texts = [payload[0] for name, payload in srv.needle.Needle.calls if name == "complete"]
    assert texts == [f"NVDA latest 10-Q risk factors\naccession_no: {accession}"]
    srv.needle.Needle.envelope["function_calls"] = [
        {"name": tool, "arguments": {"accession_no": "0001045810-26-000052"}}
    ]
    assert "changed grounded accession_no" in srv.handle(json.dumps(req))["error"]
    req["context"] = {}
    assert "needs a grounded accession_no" in srv.handle(json.dumps(req))["error"]
    assert len([name for name, _ in srv.needle.Needle.calls if name == "complete"]) == 2


@pytest.mark.parametrize("form", ["10-Q", "10-K", "8-K"])
def test_generate_exposes_form_and_default_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form: str) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    arguments = {"identifier": "NVDA", "forms": [form], "limit": 50}
    srv.needle.Needle.envelope = {
        "type": "call",
        "function_calls": [{"name": "list_sec_filings", "arguments": arguments}],
    }
    req = {
        "id": "forms",
        "action": "arguments.generate",
        "tool": "list_sec_filings",
        "schema": _production_schema("list_sec_filings"),
        "objective": f"NVDA latest {form} risk factors",
    }
    assert srv.handle(json.dumps(req))["arguments"] == arguments
    texts = [payload[0] for name, payload in srv.needle.Needle.calls if name == "complete"]
    assert texts == [f"NVDA latest {form} risk factors\nforms: {form}\nlimit: 50"]
    arguments["limit"] = int(form.split("-")[0])
    assert "changed grounded limit" in srv.handle(json.dumps(req))["error"]


def _run_operation(operation: str) -> object:
    """Run one public client operation that talks to the worker."""
    if operation == "generate":
        return nc.generate_arguments(tool="search_sec_filings")
    if operation == "extract":
        return nc.extract_fields({"name": "web_row"}, "some passage")
    return nc.embed("some passage")


def _script_replies(monkeypatch: pytest.MonkeyPatch, reply: Callable[[dict[str, object]], str]) -> None:
    """Answer pings normally; answer every other action with the raw line `reply`."""

    def _fake_readline(proc: _FakeProc, timeout_s: float) -> str:
        request: object = json.loads(proc.written[-1])
        assert isinstance(request, dict)
        if request.get("action") == "ping":
            return json.dumps({"id": request.get("id"), "ready": True}) + "\n"
        return reply(request)

    monkeypatch.setattr(nc, "_readline", _fake_readline)


def _echo_id(tail: dict[str, object]) -> Callable[[dict[str, object]], str]:
    """Reply carrying the request id plus `tail`."""

    def _reply(request: dict[str, object]) -> str:
        return json.dumps({"id": request.get("id"), **tail}) + "\n"

    return _reply


def test_spawn_failure_fails_closed(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused spawn raises the OS reason and owns no child afterwards."""

    def _refuse_spawn(*args: object, **kwargs: object) -> None:
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", _refuse_spawn)
    with pytest.raises(RuntimeError, match="needle spawn failed: .*No such file or directory"):
        nc.embed("some passage")
    assert nc._PROC is None


def test_spawn_without_pipes_fails_closed(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A child missing any pipe is destroyed and never registered as owned."""
    killed: list[object] = []

    class _NoPipesProc(_FakeProc):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.stdin = None
            self.stdout = None
            self.stderr = None

        @override
        def kill(self) -> None:
            killed.append(self)
            super().kill()

    monkeypatch.setattr(subprocess, "Popen", _NoPipesProc)
    with pytest.raises(RuntimeError, match="missing pipes"):
        nc.embed("some passage")
    assert len(killed) == 1
    assert nc._PROC is None


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("generate", "needle arguments.generate failed for 'search_sec_filings': server stdin missing"),
        ("extract", "needle extract failed for 'web_row': server stdin missing"),
        ("embed", "needle embed failed: server stdin missing"),
    ],
)
def test_exchange_without_stdin_fails_closed(worker: None, operation: str, expected: str) -> None:
    """A live child that lost its stdin fails each exchange before writing."""
    _run_operation(operation)
    _FakeProc.instances[0].stdin = None
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _run_operation(operation)


_MALFORMED_REPLIES: dict[str, Callable[[dict[str, object]], str]] = {
    "non-json": lambda request: "not json\n",
    "json-list": lambda request: json.dumps([1, 2]) + "\n",
    "json-string": lambda request: json.dumps("ok") + "\n",
    "json-number": lambda request: json.dumps(7) + "\n",
    "json-null": lambda request: json.dumps(None) + "\n",
    "wrong-id": lambda request: json.dumps({"id": "needle:wrong", "ready": True}) + "\n",
    "error-envelope": lambda request: json.dumps({"id": request.get("id"), "error": "boom"}) + "\n",
}


@pytest.mark.parametrize("operation", ["generate", "extract", "embed"])
@pytest.mark.parametrize("reply_name", sorted(_MALFORMED_REPLIES))
def test_malformed_server_responses_fail_closed(
    worker: None, monkeypatch: pytest.MonkeyPatch, operation: str, reply_name: str
) -> None:
    """Undecodable, uncorrelated, or error envelopes fail every exchange with its own label."""
    _script_replies(monkeypatch, _MALFORMED_REPLIES[reply_name])
    expected = "boom" if reply_name == "error-envelope" else "malformed server response"
    with pytest.raises(RuntimeError, match=expected):
        _run_operation(operation)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("extract", "needle extract for 'web_row' timed out after 120s"),
        ("embed", "needle embed timed out after 120s"),
    ],
)
def test_silent_server_reaps_worker(
    worker: None, monkeypatch: pytest.MonkeyPatch, operation: str, expected: str
) -> None:
    """A silent child is killed and reaped, and the failure names the operation and budget."""

    def _silent_after_ping(proc: _FakeProc, timeout_s: float) -> str:
        request: object = json.loads(proc.written[-1])
        assert isinstance(request, dict)
        if request.get("action") == "ping":
            return proc._readline()
        assert timeout_s == nc._TIMEOUT_S
        raise TimeoutError("silence")

    monkeypatch.setattr(nc, "_readline", _silent_after_ping)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _run_operation(operation)
    assert nc._PROC is None
    assert _FakeProc.instances[0].returncode is not None


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("generate", "needle arguments.generate failed for 'search_sec_filings': server unavailable"),
        ("extract", "needle extract for 'web_row' failed: server unavailable"),
        ("embed", "needle embed failed: server unavailable"),
    ],
)
def test_persistent_broken_pipe_gives_up_after_one_restart(worker: None, operation: str, expected: str) -> None:
    """A second broken pipe reports the server as unavailable instead of looping."""
    _FakeProc.behavior = "broken-always"
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _run_operation(operation)
    assert len(_FakeProc.instances) == 2
    assert _FakeProc.instances[1].returncode is not None
    assert nc._PROC is None


def test_generate_arguments_request_forms(worker: None) -> None:
    """One positional request dict or kwargs; any other call shape fails before spawning."""
    with pytest.raises(ValueError, match="tool must be a nonempty tool name"):
        nc.generate_arguments()
    with pytest.raises(ValueError, match="tool must be a nonempty tool name"):
        nc.generate_arguments(tool="")
    with pytest.raises(ValueError, match="tool must be a nonempty tool name"):
        nc.generate_arguments(tool=["search_sec_filings"])
    with pytest.raises(TypeError, match="pass kwargs"):
        nc.generate_arguments("search_sec_filings")
    with pytest.raises(TypeError, match="pass kwargs"):
        nc.generate_arguments({"tool": "search_sec_filings"}, {"tool": "other"})
    with pytest.raises(TypeError, match="pass kwargs"):
        nc.generate_arguments({"tool": "search_sec_filings"}, schema={})
    assert _FakeProc.instances == []
    out = nc.generate_arguments({"tool": "search_sec_filings", "objective": "objective text"})
    assert out["tool"] == "search_sec_filings"
    sent: object = json.loads(_FakeProc.instances[0].written[-1])
    assert isinstance(sent, dict) and sent["objective"] == "objective text"


def test_missing_server_and_weights_fail_loudly(worker: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing server file or missing weights raises before any spawn."""
    monkeypatch.setattr(nc, "_server", lambda: tmp_path / "server.py")
    with pytest.raises(RuntimeError, match="needle arguments.generate unavailable"):
        nc.generate_arguments(tool="search_sec_filings")
    with pytest.raises(RuntimeError, match="needle extract unavailable"):
        nc.extract_fields({"name": "web_row"}, "some passage")
    with pytest.raises(RuntimeError, match="needle embed unavailable"):
        nc.embed("some passage")
    assert _FakeProc.instances == []

    def _no_weights() -> None:
        raise RuntimeError("needle weights missing: /nowhere/needle3.cact")

    monkeypatch.setattr(nc, "_ensure_weights", _no_weights)
    monkeypatch.setattr(nc, "_server", lambda: Path(nc.__file__))
    for operation in ("generate", "extract", "embed"):
        with pytest.raises(RuntimeError, match="needle weights missing"):
            _run_operation(operation)
    assert _FakeProc.instances == []


def test_extract_options_validation_and_wire_payload(worker: None) -> None:
    """Decoding options are validated before spawn and travel verbatim on the wire."""
    with pytest.raises(TypeError, match="max_new_tokens must be an int in 1..4096"):
        nc.extract_fields({"name": "web_row"}, "some passage", max_new_tokens=0)
    with pytest.raises(TypeError, match="max_new_tokens must be an int in 1..4096"):
        nc.extract_fields({"name": "web_row"}, "some passage", max_new_tokens=4097)
    with pytest.raises(TypeError, match="max_new_tokens must be an int in 1..4096"):
        nc.extract_fields({"name": "web_row"}, "some passage", max_new_tokens=True)
    with pytest.raises(TypeError, match="system must be a string or None"):
        nc._check_extract_options(512, 5, True)
    with pytest.raises(TypeError, match="strict must be a bool"):
        nc._check_extract_options(512, None, "yes")
    with pytest.raises(ValueError, match="record must carry a nonempty name"):
        nc.extract_fields({"name": ""}, "some passage")
    with pytest.raises(ValueError, match="passage must be a nonempty string"):
        nc.extract_fields({"name": "web_row"}, "  ")
    assert _FakeProc.instances == []
    out = nc.extract_fields({"name": "web_row"}, "some passage", system="custom", max_new_tokens=1, strict=False)
    assert out["record"] == "web_row"
    sent: object = json.loads(_FakeProc.instances[0].written[-1])
    assert isinstance(sent, dict)
    assert sent["system"] == "custom" and sent["max_new_tokens"] == 1 and sent["strict"] is False


def test_generate_response_shapes(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or foreign tool, a non-mapping arguments value, and odd flags fail or normalize."""
    _script_replies(monkeypatch, _echo_id({"arguments": {"q": 1}}))
    with pytest.raises(ValueError, match="needle tool mismatch"):
        nc.generate_arguments(tool="search_sec_filings")

    _script_replies(monkeypatch, _echo_id({"tool": "other", "arguments": {}}))
    with pytest.raises(ValueError, match="needle tool mismatch"):
        nc.generate_arguments(tool="search_sec_filings")

    _script_replies(monkeypatch, _echo_id({"tool": "search_sec_filings", "arguments": "x"}))
    with pytest.raises(TypeError, match="must be a mapping"):
        nc.generate_arguments(tool="search_sec_filings")

    _script_replies(
        monkeypatch,
        _echo_id({"tool": "search_sec_filings", "arguments": {}, "withheld": "yes", "reasoning": 5}),
    )
    out = nc.generate_arguments(tool="search_sec_filings")
    assert out["withheld"] is False and out["reasoning"] == "5" and out["confidence"] is None
    assert out["arguments"] == {}


def test_extract_response_shapes(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Null records pass through; a foreign record name or non-mapping fields fail closed."""
    _script_replies(monkeypatch, _echo_id({"fields": "x"}))
    with pytest.raises(TypeError, match="must be a mapping"):
        _run_operation("extract")

    _script_replies(monkeypatch, _echo_id({"record": None}))
    out = nc.extract_fields({"name": "web_row"}, "some passage")
    assert out["record"] is None and out["fields"] == {} and out["reasoning"] == ""

    _script_replies(monkeypatch, _echo_id({"record": "other", "fields": {}}))
    with pytest.raises(RuntimeError, match="malformed server response"):
        nc.extract_fields({"name": "web_row"}, "some passage")


@pytest.mark.parametrize(
    "embedding",
    [[], "abc", {"a": 1}, [1, True], ["1"], [None], [[1.0]], 7, None],
)
def test_embed_rejects_malformed_vectors(worker: None, monkeypatch: pytest.MonkeyPatch, embedding: object) -> None:
    """Only a non-empty list of JSON numbers becomes a vector; nothing else passes."""
    _script_replies(monkeypatch, _echo_id({"embedding": embedding}))
    with pytest.raises((RuntimeError, TypeError), match="malformed server response"):
        nc.embed("some passage")


def test_embed_accepts_json_numbers(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integer and float JSON numbers become floats in reply order."""
    _script_replies(monkeypatch, _echo_id({"embedding": [1, 2.5, -3]}))
    assert nc.embed("some passage") == [1.0, 2.5, -3.0]


def test_similarity_order_paths(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank references, short or over-cap sets, and embed failures keep the caller's order."""
    assert nc.similarity_order(["a", "b"], "  ") is None
    assert nc._nonblank_reference(5) is False
    assert nc.similarity_order(["only"], "reference") is None
    assert nc.similarity_order([], "reference") is None
    assert nc.similarity_order(["a", "b", "c"], "reference", max_texts=2) is None
    assert _FakeProc.instances == []

    def _embed(text: str) -> list[float]:
        assert text.strip()
        return [1.0, 0.0] if text.startswith("near") else [0.0, 1.0]

    monkeypatch.setattr(nc, "embed", _embed)
    assert nc.similarity_order(["far", "near"], "reference") == [0, 1]
    assert nc.similarity_order(["near", "far"], "reference") == [1, 0]
    assert nc.similarity_order(("near", "far"), "reference") == [1, 0]
    # Blank texts sink below every embedded text; equal scores keep input order.
    assert nc.similarity_order(["  ", "far", "near"], "reference") == [1, 2, 0]
    assert nc.similarity_order(["far", "far"], "reference") == [0, 1]

    def _broken_embed(text: str) -> list[float]:
        raise RuntimeError("embed down")

    monkeypatch.setattr(nc, "embed", _broken_embed)
    assert nc.similarity_order(["far", "near"], "reference") is None


_JSON_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12)
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3),
    max_leaves=6,
)


def test_prop_generate_arguments_shapes_replies_or_fails_closed() -> None:
    """Every JSON reply either yields the fixed result shape for the requested tool or raises."""
    with pytest.MonkeyPatch.context() as mp:
        nc.close()
        _FakeProc.instances.clear()
        _FakeProc.behavior = "echo"
        mp.setattr(subprocess, "Popen", _FakeProc)
        mp.setattr(nc, "_ensure_weights", lambda: None)
        reply_body: dict[str, dict[str, object]] = {"body": {}}
        reply_flags: dict[str, bool] = {"match_id": True}

        def _fake_readline(proc: _FakeProc, timeout_s: float) -> str:
            request: object = json.loads(proc.written[-1])
            assert isinstance(request, dict)
            if request.get("action") == "ping":
                return json.dumps({"id": request.get("id"), "ready": True}) + "\n"
            body = dict(reply_body["body"])
            body["id"] = request.get("id") if reply_flags["match_id"] else "needle:wrong"
            return json.dumps(body) + "\n"

        mp.setattr(nc, "_readline", _fake_readline)

        @seed(20261014)
        @settings(max_examples=200, derandomize=True)
        @given(
            st.dictionaries(st.text(max_size=6), _JSON_VALUES, max_size=4),
            st.booleans(),
            st.booleans(),
        )
        def check(payload: dict[str, object], match_id: bool, tool_matches: bool) -> None:
            body = {key: value for key, value in payload.items() if key not in ("id", "tool")}
            if tool_matches:
                body["tool"] = "search_sec_filings"
            reply_body["body"] = body
            reply_flags["match_id"] = match_id
            try:
                out = nc.generate_arguments(tool="search_sec_filings")
            except (RuntimeError, TypeError, ValueError) as exc:
                assert str(exc)
                return
            assert set(out) == {"tool", "arguments", "withheld", "reasoning", "confidence"}
            assert out["tool"] == "search_sec_filings"
            assert isinstance(out["arguments"], dict) and isinstance(out["withheld"], bool)
            assert isinstance(out["reasoning"], str)

        check()
        nc.close()
        _FakeProc.instances.clear()


def test_prop_embed_shapes_vector_replies_or_fails_closed() -> None:
    """A non-empty list of JSON numbers becomes floats; any other reply shape raises."""
    with pytest.MonkeyPatch.context() as mp:
        nc.close()
        _FakeProc.instances.clear()
        _FakeProc.behavior = "echo"
        mp.setattr(subprocess, "Popen", _FakeProc)
        mp.setattr(nc, "_ensure_weights", lambda: None)
        reply_body: dict[str, dict[str, object]] = {"body": {}}

        def _fake_readline(proc: _FakeProc, timeout_s: float) -> str:
            request: object = json.loads(proc.written[-1])
            assert isinstance(request, dict)
            if request.get("action") == "ping":
                return json.dumps({"id": request.get("id"), "ready": True}) + "\n"
            return json.dumps({"id": request.get("id"), **reply_body["body"]}) + "\n"

        mp.setattr(nc, "_readline", _fake_readline)

        @seed(20261014)
        @settings(max_examples=200, derandomize=True)
        @given(
            st.one_of(
                st.lists(
                    st.integers(min_value=-(10**6), max_value=10**6) | st.floats(allow_nan=False, allow_infinity=False),
                    min_size=1,
                    max_size=4,
                ),
                st.lists(
                    st.booleans() | st.none() | st.text(max_size=4) | st.lists(st.integers(max_value=9), max_size=2),
                    max_size=3,
                ),
                st.sampled_from(["abc", {"a": 1}, 7, None, [[1.0]]]),
            )
        )
        def check(embedding: object) -> None:
            reply_body["body"] = {"embedding": embedding}
            numeric = (
                isinstance(embedding, list)
                and bool(embedding)
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in embedding)
            )
            if numeric:
                assert nc.embed("some passage") == [float(v) for v in embedding]
                return
            with pytest.raises((RuntimeError, TypeError)):
                nc.embed("some passage")

        check()
        nc.close()
        _FakeProc.instances.clear()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not json at all",
        '{"tools": []}',
        "[]",
        "null",
        '[5, "junk", {"description": "no name"}]',
    ],
)
def test_load_catalog_exits_on_unusable_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str | None
) -> None:
    """A missing, unreadable, non-list, or nameless catalog stops the server before weights load."""
    with pytest.raises(SystemExit):
        _import_server_with_catalog(monkeypatch, tmp_path, raw)


def test_load_catalog_slims_usable_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The shared catalog keeps names, truncates descriptions, and drops parameter schemas."""
    catalog = [
        "junk",
        5,
        {"description": "no name"},
        {
            "name": "blank_desc",
            "description": "   ",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
        {"name": "long_desc", "description": "x" * 400},
    ]
    srv = _import_server_with_catalog(monkeypatch, tmp_path, json.dumps(catalog))
    assert [entry["name"] for entry in srv.TOOLS] == ["blank_desc", "long_desc"]
    assert srv.TOOLS[0]["description"] == "blank_desc"
    assert srv.TOOLS[1]["description"] == "x" * 150
    assert all(entry["parameters"] == {"type": "object"} for entry in srv.TOOLS)


def test_handle_rejects_malformed_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unparsable, untagged, oversized, and unknown-action lines answer instead of raising."""
    srv = _load_server(monkeypatch, tmp_path)
    assert srv.handle("not json") == {"id": "?", "error": "bad_request"}
    assert srv.handle("[1, 2]") == {"id": "?", "error": "bad_request"}
    assert srv.handle(json.dumps({"action": "ping"})) == {"id": "?", "error": "bad_request"}
    assert srv.handle(json.dumps({"id": 5, "action": "ping"})) == {"id": "?", "error": "bad_request"}
    assert srv.handle("x" * (srv.MAX_LINE_CHARS + 1)) == {"id": "?", "error": "bad_request"}
    assert srv.handle(json.dumps({"id": "p1", "action": "ping"})) == {"id": "p1", "ready": True}
    assert srv.handle(json.dumps({"id": "u1", "action": "nope"})) == {"id": "u1", "error": "bad_action"}
    assert srv.handle(json.dumps({"id": "u2"})) == {"id": "u2", "error": "bad_action"}
    assert srv.handle(json.dumps({"id": "u3", "action": ["ping"]})) == {"id": "u3", "error": "bad_action"}
    assert srv.handle(json.dumps({"id": "g1", "action": "arguments.generate"}))["error"] == "bad tool"


def test_handle_legacy_start_and_step_return_decisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Legacy start/step still run on the shared agent and answer with one decision shape."""
    srv = _load_server(monkeypatch, tmp_path)
    srv.needle.Needle.envelope = {
        "type": "call",
        "function_calls": [{"name": "list_sec_filings", "arguments": {"identifier": "AAPL"}}],
        "confidence": 0.7,
        "reasoning": "AAPL from the request",
    }
    assert srv.handle(json.dumps({"id": "s1", "action": "start", "prompt": "AAPL 10-Q filings"})) == {
        "id": "s1",
        "tool": "list_sec_filings",
        "arguments": {"identifier": "AAPL"},
        "withheld": False,
        "confidence": 0.7,
        "reasoning": "AAPL from the request",
    }
    assert srv.handle(json.dumps({"id": "s2", "action": "step", "result": {"a": 1}}))["tool"] == "list_sec_filings"
    assert srv.handle(json.dumps({"id": "s3", "action": "start", "prompt": 5}))["error"] == "bad prompt"
    texts = [payload[0] for name, payload in srv.needle.Needle.calls if name == "complete"]
    assert texts == ["AAPL 10-Q filings", json.dumps({"a": 1})]
    assert ("reset", None) in srv.needle.Needle.calls


def test_handle_extract_binds_one_record_and_honours_strict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Extract binds one record shape, carries the custom system, and keeps strict gating."""
    srv = _load_server(monkeypatch, tmp_path)
    srv.needle.Needle.envelope = {
        "type": "call",
        "function_calls": [{"name": "web_row", "arguments": {"u": "x"}}],
        "confidence": 0.5,
        "reasoning": "one row",
    }
    record = {"name": "web_row", "parameters": {"type": "object"}}
    out = srv.handle(
        json.dumps(
            {
                "id": "x1",
                "action": "extract",
                "record": record,
                "passage": "NVDA news up 5%",
                "system": "custom extraction system",
                "max_new_tokens": 8,
            }
        )
    )
    assert out["record"] == "web_row" and out["fields"] == {"u": "x"} and out["withheld"] is False
    bound = srv.needle.Needle.instances[-1]
    assert bound.kwargs["system"] == "custom extraction system" and bound.kwargs["tools"] == [record]
    assert ("complete", ("NVDA news up 5%", 8)) in srv.needle.Needle.calls

    srv.needle.Needle.envelope = {
        "type": "call",
        "function_calls": [{"name": "web_row", "arguments": {"u": "hallucinated"}}],
        "validation": {"ungrounded": ["web_row.u"], "negation": False},
    }
    for strict in (True, "nope"):
        flagged = srv.handle(
            json.dumps(
                {
                    "id": "x2",
                    "action": "extract",
                    "record": record,
                    "passage": "NVDA news",
                    "strict": strict,
                }
            )
        )
        assert "not grounded" in str(flagged["error"])
    assert srv.handle(
        json.dumps({"id": "x3", "action": "extract", "record": record, "passage": "NVDA news", "strict": False})
    )["fields"] == {"u": "hallucinated"}


def test_handle_embed_returns_and_rejects_vectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The shared agent vector passes through; empty, non-list, and non-numeric vectors fail."""
    srv = _load_server(monkeypatch, tmp_path)
    srv.needle.Needle.vector = [1, 2.5]
    assert srv.handle(json.dumps({"id": "m1", "action": "embed", "text": "NVDA revenue"})) == {
        "id": "m1",
        "embedding": [1, 2.5],
    }
    bad_vectors: tuple[object, ...] = ([], "abc", [1, True], [None])
    for vector in bad_vectors:
        srv.needle.Needle.vector = vector
        out = srv.handle(json.dumps({"id": "m2", "action": "embed", "text": "NVDA revenue"}))
        assert out["error"] == "bad embedding"


def test_read_line_returns_one_line_and_reports_overlong(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One line per call; an overlong line is drained and reported once, keeping later lines."""
    srv = _load_server(monkeypatch, tmp_path)
    monkeypatch.setattr(srv, "MAX_LINE_CHARS", 8)
    monkeypatch.setattr(sys, "stdin", io.StringIO("12345678\n123456789\n\nok\ntail"))
    assert srv._read_line() == ("12345678\n", False)
    assert srv._read_line() == (None, True)
    assert srv._read_line() == ("\n", False)
    assert srv._read_line() == ("ok\n", False)
    assert srv._read_line() == ("tail", False)
    assert srv._read_line() == ("", False)


def test_read_line_drains_overlong_line_across_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An overlong line longer than one read chunk is drained to its newline, not left for later."""
    srv = _load_server(monkeypatch, tmp_path)
    monkeypatch.setattr(srv, "MAX_LINE_CHARS", 8)
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 70_000 + "\n" + "next\n"))
    assert srv._read_line() == (None, True)
    assert srv._read_line() == ("next\n", False)
    assert srv._read_line() == ("", False)


def test_main_answers_every_line_until_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() writes exactly one JSON line per request, skipping blanks, until EOF."""
    srv = _load_server(monkeypatch, tmp_path)
    lines = [
        json.dumps({"id": "p1", "action": "ping"}),
        "",
        "not json",
        json.dumps({"id": "u1", "action": "nope"}),
        json.dumps({"id": "e1", "action": "embed", "text": "NVDA revenue"}),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    srv.main()
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"id": "p1", "ready": True},
        {"id": "?", "error": "bad_request"},
        {"id": "u1", "error": "bad_action"},
        {"id": "e1", "embedding": [0.1, 0.2, 0.3]},
    ]
    ping = json.dumps({"id": "p2", "action": "ping"})
    # The cap admits the follow-up request, so the drained overlong line must not eat it.
    monkeypatch.setattr(srv, "MAX_LINE_CHARS", len(ping))
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 60 + "\n" + ping + "\n"))
    srv.main()
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"id": "?", "error": "bad_request"},
        {"id": "p2", "ready": True},
    ]


def test_extract_decision_rejects_non_mapping_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A call with non-mapping arguments, or no mapping at all, fails instead of returning fields."""
    srv = _load_server(monkeypatch, tmp_path)
    for arguments in ("x", [1]):
        with pytest.raises(TypeError, match="extract fields must be a mapping"):
            srv._extract_decision("web_row", {"function_calls": [{"name": "web_row", "arguments": arguments}]})
    with pytest.raises(ValueError, match="mismatch"):
        srv._extract_decision("web_row", {"function_calls": ["nope"]})


def test_decision_ignores_malformed_call_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-mapping call entry yields no tool and never a withheld decision."""
    srv = _load_server(monkeypatch, tmp_path)
    plain = srv._decision({"type": "call", "function_calls": ["nope"]}, "t")
    assert plain["tool"] is None and plain["arguments"] == {} and plain["withheld"] is False
    withheld = srv._decision({"type": "withheld", "function_calls": [], "suppressed_calls": ["nope"]}, "t")
    assert withheld["tool"] is None and withheld["withheld"] is False


def test_prop_handle_always_answers_with_id_tagged_mapping() -> None:
    """Any stdin line yields a JSON-ready mapping tagged with an id; handle never raises."""
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as tmp:
        srv = _load_server(mp, Path(tmp))

        @seed(20261014)
        @settings(max_examples=200, derandomize=True)
        @given(
            st.one_of(
                st.text(max_size=40),
                st.dictionaries(
                    st.sampled_from(
                        [
                            "id",
                            "action",
                            "tool",
                            "record",
                            "passage",
                            "text",
                            "prompt",
                            "schema",
                            "node",
                            "objective",
                            "context",
                            "strict",
                            "max_new_tokens",
                            "result",
                        ]
                    ),
                    _JSON_VALUES,
                    max_size=6,
                ),
            )
        )
        def check(line: object) -> None:
            raw = line if isinstance(line, str) else json.dumps(line)
            out = srv.handle(raw)
            assert isinstance(out, dict) and isinstance(out.get("id"), str)
            assert json.dumps(out)

        check()


def test_requested_limit_reads_node_question_without_objective(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A blank objective falls back to the node question; a limit-less schema requests nothing."""
    srv = _load_server(monkeypatch, tmp_path)
    schema = _production_schema("list_sec_filings")
    assert srv._requested_limit(None, {"question": "Show TSLA's latest 5 filings"}, {}, schema) == 5
    assert srv._requested_limit("   ", {"question": "Show TSLA's latest 5 filings"}, {}, schema) == 5
    assert srv._requested_limit("Show TSLA filings", {"question": "Show TSLA's latest 5 filings"}, {}, schema) is None
    assert srv._requested_limit("Show TSLA's latest 5 filings", {}, {}, {"type": "object"}) is None
    assert srv._requested_limit("Show TSLA's latest 5 filings", {}, {"grounded_arguments": {"limit": 5}}, schema) == 5
    with pytest.raises(ValueError, match="conflicting explicit counts"):
        srv._requested_limit("Show TSLA's latest 5 filings", {}, {"grounded_arguments": {"limit": 10}}, schema)
