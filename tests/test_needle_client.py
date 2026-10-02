"""Persistent Needle worker: one process reused, IDs correlated, fail closed."""

from __future__ import annotations

import io
import json
import subprocess
import types
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

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

    def __init__(self, *args: object, **kwargs: object) -> None:
        _FakeProc.instances.append(self)
        self.cmd: object = args[0] if args else None
        self.written: list[str] = []
        self.stdin = _FakeProc._In(self)
        self.stdout = _FakeProc._Out(self)
        self.stderr = io.StringIO("")
        self.returncode: int | None = None

    def _write(self, s: str) -> None:
        if _FakeProc.behavior == "broken-once" and len(_FakeProc.instances) == 1:
            raise BrokenPipeError(32, "Broken pipe")
        self.written.append(s)

    def _readline(self) -> str:
        raw: object = json.loads(self.written[-1])
        assert isinstance(raw, dict)
        if raw.get("action") == "ping":
            return json.dumps({"id": raw.get("id"), "ready": True}) + "\n"
        if _FakeProc.behavior == "mismatch":
            return json.dumps({"id": "needle:wrong", "tool": raw.get("tool"), "arguments": {}}) + "\n"
        if _FakeProc.behavior == "malformed":
            return "not json\n"
        return json.dumps({"id": raw.get("id"), "tool": raw.get("tool"), "arguments": {"q": 1}}) + "\n"

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
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
    second = nc.generate_arguments(
        {"tool": "search_sec_filings", "schema": {}, "objective": "o", "node": "n", "context": {}}
    )
    assert first == {"tool": "search_sec_filings", "arguments": {"q": 1}, "reasoning": "", "confidence": None}
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
    assert out == {"tool": "search_sec_filings", "arguments": {"q": 1}, "reasoning": "", "confidence": None}
    assert len(_FakeProc.instances) == 2


def _load_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """Import server.py with the needle extension stubbed (venv has no needle pkg)."""
    import importlib.util
    import sys

    stub = types.ModuleType("needle")

    class _FakeNeedle:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def complete(self, *args: object, **kwargs: object) -> dict[str, object]:
            return {"type": "none"}

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

    stub.Needle = _FakeNeedle  # type: ignore[attr-defined]
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
                    "as_of": {"type": "string", "description": "Point-in-time date YYYY-MM-DD."},
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
    assert entry["triggers"] == [".+"]
    fallback = srv._tool_entry("get_sec_filing", {})
    assert (
        "Named accession_no, not accession_number"
        in fallback["parameters"]["properties"]["accession_no"]["description"]
    )


def test_arguments_prompt_states_grounding_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    decoded: object = json.loads(srv._arguments_prompt("get_sec_filing", {}, "o", "n", {}))
    assert isinstance(decoded, dict)
    instruction = str(decoded["instruction"])
    for needle in (
        "must-call-or-error",
        "never chain",
        "never judge",
        "accession_no",
        "search_sec_filings",
        "AAPL",
        "FINRA",
        "YYYY-MM-DD",
        "this week",
        "group/name",
        "never fabricate",
        "Today UTC is",
        "decode relative dates before choosing",
        "Monday-now NYC range",
    ):
        assert needle in instruction
    assert str(decoded["objective"]) == "o"


def test_arguments_prompt_objective_raw_verbatim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    once: object = json.loads(srv._arguments_prompt("get_sec_filing", {}, "o", "n", {}))
    assert isinstance(once, dict) and once["objective"] == "o"
    stamped: object = json.loads(srv._arguments_prompt("get_sec_filing", {}, "[Today UTC 2026-01-01] o", "n", {}))
    assert isinstance(stamped, dict) and stamped["objective"] == "[Today UTC 2026-01-01] o"
    blank: object = json.loads(srv._arguments_prompt("get_sec_filing", {}, "", "n", {}))
    assert isinstance(blank, dict) and blank["objective"] == ""


def test_bound_system_carries_temporal_decoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    system = srv._bound_system()
    assert "Today UTC is" in system and "decode relative dates before choosing" in system


def test_legacy_global_catalog_stays_slim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    srv = _load_server(monkeypatch, tmp_path)
    assert all(len(t["description"]) <= 150 for t in srv.TOOLS)
    assert "description" not in json.dumps([t["parameters"] for t in srv.TOOLS])
    assert srv._strip_descriptions({"a": {"description": "x", "type": "string"}}) == {"a": {"type": "string"}}


def test_handle_forwards_node_context_into_arguments_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Live path: handle arguments.generate forwards node/context into _arguments_prompt."""
    import json as _json

    srv = _load_server(monkeypatch, tmp_path)
    seen: dict[str, object] = {}
    orig = srv._arguments_prompt

    def _spy(tool: object, schema: object, objective: object, node: object, context: object) -> str:
        seen["node"] = node
        seen["context"] = context
        return orig(tool, schema, objective, node, context)  # type: ignore[arg-type]

    monkeypatch.setattr(srv, "_arguments_prompt", _spy)
    node = {"node_id": "n1"}
    context = {"today_utc": "2026-09-27", "temporal_scope": {"mode": "latest-available"}}
    line = _json.dumps(
        {
            "id": "r1",
            "action": "arguments.generate",
            "tool": "get_sec_filing",
            "schema": {},
            "objective": "o",
            "node": node,
            "context": context,
        }
    )
    out = srv.handle(line)
    assert seen["node"] == node
    assert seen["context"] == context
    assert out["id"] == "r1"


def test_arguments_prompt_omits_as_of_without_dates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Omit rule pins as_of only (since/start/end stay out of the scrub path)."""
    srv = _load_server(monkeypatch, tmp_path)
    decoded: object = json.loads(srv._arguments_prompt("get_sec_filing", {}, "o", "n", {}))
    assert isinstance(decoded, dict)
    instruction = str(decoded["instruction"])
    assert "as_of only from an explicit YYYY-MM-DD date" in instruction
    assert "omit as_of entirely (latest-available)" in instruction
