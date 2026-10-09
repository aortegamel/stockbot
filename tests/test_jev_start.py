"""JevClient.start() startup behavior: ping handshake + single sidecar reuse."""

from __future__ import annotations

import asyncio
import io
import json
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from pathlib import Path
from types import TracebackType
from typing import Self, override

import pytest

from app.decision_client import JevClient
from app.research.models import JSONValue


class _FakeStdin(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.last = ""
        self.writes: list[dict[str, object]] = []

    @override
    def write(self, s: str, /) -> int:
        self.last = s
        self.writes.append(json.loads(s))
        return len(s)

    @override
    def flush(self) -> None:
        pass


class _FakeStdout(io.StringIO):
    def __init__(self, stdin: _FakeStdin, *, bad_ack: bool = False) -> None:
        super().__init__()
        self._stdin = stdin
        self._bad_ack = bad_ack

    @override
    def readline(self, size: int | None = -1, /) -> str:
        payload = json.loads(self._stdin.last)
        if self._bad_ack:
            return json.dumps({"id": "wrong", "ready": True}) + "\n"
        if payload.get("op") == "ping":
            return json.dumps({"id": payload["id"], "ready": True}) + "\n"
        return (
            json.dumps({"id": payload["id"], "raw": {}, "decisions": {"q": {"kind": "noul", "probability": 0.5}}})
            + "\n"
        )


class _FakeProc(subprocess.Popen[str]):
    def __init__(self, *, bad_ack: bool = False) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(self.stdin, bad_ack=bad_ack)

    @override
    def poll(self) -> None:
        return None

    @override
    def terminate(self) -> None:
        pass

    @override
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
    assert isinstance(proc.stdin, _FakeStdin)
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
    from app.research.models import query_with_today_utc

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


def test_outcome_dict_truncates_64k_content_keeps_error() -> None:
    from app.decision_client import _outcome_dict

    big = "x" * 65536
    err = "e" * 5000
    out = _outcome_dict({"tool": "t", "content": big, "error": err, "error_type": "boom"})
    assert isinstance(out["content"], str) and len(out["content"]) <= 2000
    assert out["error"] == err
    assert out["error_type"] == "boom"


class _HungStdout(io.StringIO):
    """readline blocks until terminate releases it, then reports EOF."""

    def __init__(self) -> None:
        super().__init__()
        self._released = threading.Event()

    @override
    def readline(self, size: int | None = -1, /) -> str:
        self._released.wait(timeout=30)
        return ""


class _HungProc(subprocess.Popen[str]):
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _HungStdout()
        self.terminated = threading.Event()

    @override
    def poll(self) -> None:
        return None

    @override
    def terminate(self) -> None:
        self.terminated.set()
        assert isinstance(self.stdout, _HungStdout)
        self.stdout._released.set()

    @override
    def kill(self) -> None:
        self.terminate()


def test_hung_sidecar_timeout_keeps_loop_responsive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=0.2)
    hung = _HungProc()
    client._proc = hung

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _decide_hung() -> None:
        await client.decide(
            {"s": 1},
            {"q": {"type": "noul", "instructions": "x"}},
            decision_type="t",
            session_id="s",
        )

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timeout"):
        asyncio.run(_decide_hung())
    assert time.monotonic() - start < 5

    async def _probe() -> None:
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=2)

    asyncio.run(_probe())
    assert hung.terminated.is_set()
    deadline = time.monotonic() + 5
    while client._lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not client._lock.locked()

    async def _stub(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return {"answers": {"q": {"type": "noul", "noul": 0.7}}}

    client._transport = _stub
    out = asyncio.run(
        client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")
    )
    assert out == {"q": {"kind": "noul", "probability": 0.7}}


def test_outer_cancel_closes_hung_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=60)
    hung = _HungProc()
    client._proc = hung

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _decide_hung() -> None:
        await client.decide(
            {"s": 1},
            {"q": {"type": "noul", "instructions": "x"}},
            decision_type="t",
            session_id="s",
        )

    async def _outer() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.ensure_future(_decide_hung()), timeout=0.2)

    start = time.monotonic()
    asyncio.run(_outer())
    elapsed = time.monotonic() - start
    assert elapsed < 5

    async def _probe() -> None:
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=2)

    asyncio.run(_probe())
    assert hung.terminated.is_set()
    assert client._proc is None or hung.terminated.is_set()
    deadline = time.monotonic() + 5
    while client._lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not client._lock.locked()
    assert client._lock.acquire(blocking=False)
    client._lock.release()


class _GateStdout(io.StringIO):
    """readline blocks until the gate opens, then answers the latest write."""

    def __init__(self, stdin: _FakeStdin, gate: threading.Event) -> None:
        super().__init__()
        self._stdin = stdin
        self._gate = gate

    @override
    def readline(self, size: int | None = -1, /) -> str:
        assert self._gate.wait(timeout=10), "sidecar gate never opened"
        payload = json.loads(self._stdin.last)
        return (
            json.dumps({"id": payload["id"], "raw": {}, "decisions": {"q": {"kind": "noul", "probability": 0.5}}})
            + "\n"
        )


class _GateProc(subprocess.Popen[str]):
    def __init__(self, gate: threading.Event) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _GateStdout(self.stdin, gate)
        self.terminated = threading.Event()

    @override
    def poll(self) -> None:
        return None

    @override
    def terminate(self) -> None:
        self.terminated.set()

    @override
    def kill(self) -> None:
        self.terminate()


def test_outer_cancel_waiter_leaves_holder_sidecar_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=30)
    gate = threading.Event()
    proc = _GateProc(gate)
    client._proc = proc

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _scenario() -> None:
        questions: dict[str, JSONValue] = {"q": {"type": "noul", "instructions": "x"}}
        holder = asyncio.ensure_future(client.decide({"s": 1}, questions, decision_type="t", session_id="s"))
        await asyncio.sleep(0.3)
        waiter = asyncio.ensure_future(client.decide({"s": 1}, questions, decision_type="t", session_id="s"))
        await asyncio.sleep(0.3)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0.2)
        assert not proc.terminated.is_set()
        gate.set()
        assert await asyncio.wait_for(holder, timeout=5) == {"q": {"kind": "noul", "probability": 0.5}}
        assert not proc.terminated.is_set()
        assert client._proc is proc

    asyncio.run(_scenario())


class _BlockingHttpResp:
    """urlopen response whose read blocks until close releases it."""

    def __init__(self, entered: threading.Event) -> None:
        self._entered = entered
        self._release = threading.Event()
        self.closed = threading.Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def read(self) -> bytes:
        self._entered.set()
        assert self._release.wait(timeout=10), "http release never set"
        if self.closed.is_set():
            raise OSError("closed")
        return b'{"answers": {"q": {"type": "noul", "noul": 0.5}}}'

    def close(self) -> None:
        self.closed.set()
        self._release.set()


def test_outer_cancel_stops_hung_http_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from app.decision_client import _SidecarUnavailable

    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=30)
    entered = threading.Event()
    resp = _BlockingHttpResp(entered)

    def _noop(**kwargs: object) -> None:
        return None

    def _missing(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        raise _SidecarUnavailable("no sidecar")

    def _urlopen(request: urllib.request.Request, *, timeout: float) -> _BlockingHttpResp:
        return resp

    monkeypatch.setattr(client, "_persist", _noop)
    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    async def _scenario() -> None:
        call = asyncio.ensure_future(
            client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")
        )
        assert await asyncio.to_thread(entered.wait, 3)
        await asyncio.sleep(0.2)
        start = time.monotonic()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert time.monotonic() - start < 3
        assert resp.closed.is_set()

    asyncio.run(_scenario())


def test_cancelled_waiter_never_starts_after_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A call cancelled while queued must not write once it finally gets the lock."""
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=30)
    gate = threading.Event()
    proc = _GateProc(gate)
    client._proc = proc

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _scenario() -> None:
        questions: dict[str, JSONValue] = {"q": {"type": "noul", "instructions": "x"}}
        holder = asyncio.ensure_future(client.decide({"s": 1}, questions, decision_type="t", session_id="s"))
        await asyncio.sleep(0.3)
        waiter = asyncio.ensure_future(client.decide({"s": 1}, questions, decision_type="t", session_id="s"))
        await asyncio.sleep(0.3)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        gate.set()
        assert await asyncio.wait_for(holder, timeout=5) == {"q": {"kind": "noul", "probability": 0.5}}
        await asyncio.sleep(0.5)
        assert isinstance(proc.stdin, _FakeStdin)
        assert len(proc.stdin.writes) == 1

    asyncio.run(_scenario())


def test_http_fallback_runs_on_client_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback uses run_in_executor on the client pool, never to_thread."""
    import urllib.request

    from app.decision_client import _SidecarUnavailable

    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=30)
    seen: dict[str, object] = {}
    entered = threading.Event()
    resp = _BlockingHttpResp(entered)

    def _noop(**kwargs: object) -> None:
        return None

    def _missing(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        raise _SidecarUnavailable("no sidecar")

    def _urlopen(request: urllib.request.Request, *, timeout: float) -> _BlockingHttpResp:
        return resp

    async def _run_and_close() -> object:
        loop = asyncio.get_running_loop()
        orig = loop.run_in_executor

        def _spy[*Args, Result](
            executor: Executor | None, func: Callable[[*Args], Result], *args: *Args
        ) -> asyncio.Future[Result]:
            seen["executor"] = executor
            return orig(executor, func, *args)

        monkeypatch.setattr(loop, "run_in_executor", _spy)
        call = asyncio.ensure_future(
            client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")
        )
        assert await asyncio.to_thread(entered.wait, 3)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return seen.get("executor")

    monkeypatch.setattr(client, "_persist", _noop)
    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert asyncio.run(_run_and_close()) is client._http_pool
