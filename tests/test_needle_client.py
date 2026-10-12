"""Persistent Needle worker: one process reused, IDs correlated, fail closed."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

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
        self.stdin = _FakeProc._In(self)
        self.stdout = _FakeProc._Out(self)
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []

    def _write(self, s: str) -> None:
        if _FakeProc.behavior == "broken-once" and len(_FakeProc.instances) == 1:
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


def _load_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """Import server.py with the needle extension stubbed (venv has no needle pkg)."""
    import importlib.util
    import sys

    stub = types.ModuleType("needle")

    class _FakeNeedle:
        instances: ClassVar[list[object]] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            _FakeNeedle.instances.append(self)

        def complete(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {"type": "none"}

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

        def embed(self, text: str) -> list[float]:
            assert text
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(stub, "Needle", _FakeNeedle, raising=False)
    monkeypatch.setitem(sys.modules, "needle", stub)
    catalog = [
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
    path = tmp_path / ".needle-catalog.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv("NEEDLE_CATALOG", str(path))
    spec = importlib.util.spec_from_file_location("needle_server_under_test", "needle-harness/lib/needle/server.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_evidence_fact_keeps_block_and_all_remaining_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One SDK evidence fact keeps the shared block and every non-block field."""
    srv = _load_server(monkeypatch, tmp_path)
    record = {
        "evidence_id": "ev-1",
        "content": "Total revenue: 12.5%\u00a0of FY26 — see Note 9.",
        "content_hash": "h1",
        "provenance": "SEC",
        "claim_text": "Total revenue: 12.5%",
        "subject": "NVDA",
    }
    facts = srv._evidence_facts([record])
    assert len(facts) == 1
    assert facts[0].startswith("[GROUNDED evidence_id=ev-1]")
    assert "Total revenue: 12.5%\u00a0of FY26 — see Note 9." in facts[0]
    assert json.loads(facts[0].rsplit(" evidence[0] extra: ", 1)[1]) == {
        "claim_text": "Total revenue: 12.5%",
        "subject": "NVDA",
    }


def test_evidence_facts_keep_observation_entries_current_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Observation summaries keep full JSON and never carry the grounded label."""
    srv = _load_server(monkeypatch, tmp_path)
    observation = {
        "tool": "list_sec_filings",
        "outcome_summary": "timed out",
        "error": "TimeoutError",
    }
    facts = srv._evidence_facts([observation])
    assert facts == [json.dumps(observation, ensure_ascii=False)]


def test_arguments_facts_routes_evidence_and_excludes_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wired SDK facts reuse grounded evidence and still exclude wrapper instructions."""
    srv = _load_server(monkeypatch, tmp_path)
    record = {"evidence_id": "ev-1", "content": "12.5%", "claim_text": "12.5%"}
    text = srv._arguments_facts(
        "objective",
        {"question": "q", "node_id": "n1"},
        {"evidence": [record], "grounding_hint": "do not send"},
    )
    assert "[GROUNDED evidence_id=ev-1]" in text
    assert "do not send" not in text


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
