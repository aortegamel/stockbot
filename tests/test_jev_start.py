"""JevClient.start() startup behavior: ping handshake + single sidecar reuse."""

from __future__ import annotations

import asyncio
import json
import select
import shutil
import subprocess
from pathlib import Path

import pytest

from app.decision_client import JevClient


class _FakeStdin:
    def __init__(self) -> None:
        self.last = ""
        self.writes: list[dict[str, object]] = []

    def write(self, s: str) -> None:
        self.last = s
        self.writes.append(json.loads(s))

    def flush(self) -> None:
        pass


class _FakeStdout:
    def __init__(self, stdin: _FakeStdin, *, bad_ack: bool = False) -> None:
        self._stdin = stdin
        self._bad_ack = bad_ack

    def readline(self) -> str:
        payload = json.loads(self._stdin.last)
        if self._bad_ack:
            return json.dumps({"id": "wrong", "ready": True}) + "\n"
        if payload.get("op") == "ping":
            return json.dumps({"id": payload["id"], "ready": True}) + "\n"
        return (
            json.dumps({"id": payload["id"], "raw": {}, "decisions": {"q": {"kind": "noul", "probability": 0.5}}})
            + "\n"
        )


class _FakeProc:
    def __init__(self, *, bad_ack: bool = False) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(self.stdin, bad_ack=bad_ack)

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bad_ack: bool = False) -> tuple[JevClient, _FakeProc]:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts")
    proc = _FakeProc(bad_ack=bad_ack)
    monkeypatch.setattr(client, "_proc", proc)
    return client, proc


def test_start_pings_and_reuses_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, proc = _client(tmp_path, monkeypatch)
    client.start()
    client.start()
    assert client._proc is proc
    assert [w.get("op") for w in proc.stdin.writes] == ["ping", "ping"]
    assert proc.stdin.writes[0]["id"] != proc.stdin.writes[1]["id"]


def test_start_rejects_bad_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(tmp_path, monkeypatch, bad_ack=True)
    with pytest.raises(RuntimeError, match="ping failed"):
        client.start()


def test_start_then_decide_uses_one_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, proc = _client(tmp_path, monkeypatch)

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)
    client.start()
    out = asyncio.run(
        client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")
    )
    assert out == {"q": {"kind": "noul", "probability": 0.5}}
    assert client._proc is proc


def test_runtime_ping_over_bun() -> None:
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("bun unavailable")
    root = Path(__file__).resolve().parent.parent
    rt = root / "decision" / "runtime.ts"
    if not rt.exists():
        pytest.skip("runtime.ts missing")
    proc = subprocess.Popen(
        [bun, str(rt)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=str(root),
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"id": "ping:t1", "op": "ping"}) + "\n")
        proc.stdin.flush()
        no_fds: list[int] = []
        ready, _, _ = select.select([proc.stdout], no_fds, no_fds, 30)
        assert ready, "sidecar did not answer ping"
        assert json.loads(proc.stdout.readline()) == {"id": "ping:t1", "ready": True}
        assert proc.poll() is None
    finally:
        proc.terminate()


def test_tool_options_prompt_keeps_last_three_failures() -> None:
    from app.decision_client import _tool_options_prompt
    from app.research.models import JSONValue

    reg: list[dict[str, JSONValue]] = [{"name": "a", "description": "A tool"}]
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    attempts: list[JSONValue] = [{"tool": f"t{i}", "error": f"err{i}"} for i in range(10)]
    _, prompt = _tool_options_prompt(reg, node, [], attempts)
    assert prompt.count("Prior attempt") == 3
    assert "t7" in prompt and "t8" in prompt and "t9" in prompt
    assert "t0" not in prompt and "t6" not in prompt
    assert "Today UTC is" in prompt and "decode relative dates before choosing" in prompt
    assert "page with research_read_search beyond display_limit" in prompt
    assert "[Today UTC " in prompt and "q?" in prompt


def test_tool_options_prompt_stamp_idempotent() -> None:
    from app.decision_client import _tool_options_prompt
    from app.research.models import JSONValue, query_with_today_utc

    reg: list[dict[str, JSONValue]] = [{"name": "a", "description": "A tool"}]
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    _, prompt = _tool_options_prompt(reg, node, [], [])
    assert "[Today UTC " in prompt
    stamped = query_with_today_utc("q?")
    node2: dict[str, JSONValue] = {"node_id": "n1", "question": stamped}
    _, prompt2 = _tool_options_prompt(reg, node2, [], [])
    assert prompt2.count("[Today UTC ") == 1
    assert query_with_today_utc("") == "" and query_with_today_utc("   ") == "   "
    assert query_with_today_utc(stamped) == stamped


def test_tool_options_prompt_carries_mangled_decode_hint() -> None:
    from app.decision_client import _mangled_select_hint, _tool_options_prompt
    from app.research.models import JSONValue

    reg: list[dict[str, JSONValue]] = [{"name": "a", "description": "A tool"}]
    mangled: dict[str, JSONValue] = {
        "node_id": "n1",
        "question": "after orcls manjure what will happen to apple stock?",
    }
    _, prompt = _tool_options_prompt(reg, mangled, [], [])
    assert "seems mangled" in prompt and "orcls" in prompt
    clean: dict[str, JSONValue] = {"node_id": "n1", "question": "What is AAPL EPS?"}
    _, clean_prompt = _tool_options_prompt(reg, clean, [], [])
    assert "seems mangled" not in clean_prompt
    assert _mangled_select_hint("What is AAPL EPS?") == ""


def test_outcome_dict_truncates_64k_content_keeps_error() -> None:
    from app.decision_client import _outcome_dict

    big = "x" * 65536
    err = "e" * 5000
    out = _outcome_dict({"tool": "t", "content": big, "error": err, "error_type": "boom"})
    assert isinstance(out["content"], str) and len(out["content"]) <= 2000
    assert out["error"] == err
    assert out["error_type"] == "boom"
