"""JevClient.start() startup behavior: ping handshake + single sidecar reuse."""

from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from http.client import HTTPMessage
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Self, override

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from app.decision_client import REASON_SENTINEL, RESOLVED_SENTINEL, JevClient
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
            json.dumps(
                {
                    "id": payload["id"],
                    "raw": {},
                    "decisions": {"q": {"kind": "noul", "probability": 0.5}},
                }
            )
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
        client.decide(
            {"s": 1},
            {"q": {"type": "noul", "instructions": "x"}},
            decision_type="t",
            session_id="s",
        )
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
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


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
    assert "candidate" not in out


def test_outcome_dict_excludes_candidate_records() -> None:
    from app.decision_client import _outcome_dict

    content = "x" * 65536
    candidate: dict[str, JSONValue] = {
        "source_handle": {"accession_no": "a"},
        "content": "y" * 32000,
        "matching_passage": "y" * 32000,
    }
    out = _outcome_dict({"tool": "get_sec_document", "content": content, "candidate": candidate})
    assert out == {
        "tool": "get_sec_document",
        "content": content[:2000],
        "error": None,
        "error_type": None,
        "tool_result_id": None,
    }


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
        client.decide(
            {"s": 1},
            {"q": {"type": "noul", "instructions": "x"}},
            decision_type="t",
            session_id="s",
        )
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
            json.dumps(
                {
                    "id": payload["id"],
                    "raw": {},
                    "decisions": {"q": {"kind": "noul", "probability": 0.5}},
                }
            )
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


_DECISION_URL = "https://decisions.test/v1/chat/completions"
_DECISION_KEY = "sk-test-decision-key"
_DECISION_MODEL = "test-decision-model"
_NATIVE_URL = "https://decisions.test/api/alpha/decisions"
_NATIVE_MODEL = "typesafe/jev-1.13"
_FORMATS = ("chat_completions", "decisions")


def _decision_env(monkeypatch: pytest.MonkeyPatch, api_format: str = "chat_completions") -> None:
    native = api_format == "decisions"
    monkeypatch.setenv("DECISION_API_FORMAT", api_format)
    monkeypatch.setenv("DECISION_API_URL", _NATIVE_URL if native else _DECISION_URL)
    monkeypatch.setenv("DECISION_API_KEY", _DECISION_KEY)
    monkeypatch.setenv("DECISION_MODEL", _NATIVE_MODEL if native else _DECISION_MODEL)
    # app.config loads the repository .env; each test sets REASONING_EFFORT itself.
    monkeypatch.delenv("REASONING_EFFORT", raising=False)


def _chat_body(content: object, **extra: object) -> bytes:
    """One Chat Completions response; dict/list content is JSON-encoded like a model reply."""
    text = content if isinstance(content, str) or content is None else json.dumps(content)
    message: dict[str, object] = {"role": "assistant", "content": text}
    choice: dict[str, object] = {"index": 0, "message": message, "finish_reason": "stop"}
    body: dict[str, object] = {"choices": [choice], **extra}
    return json.dumps(body).encode()


def _native_body(answers: object, **extra: object) -> bytes:
    """One native Decisions response: answers sit at the top level beside model/usage."""
    return json.dumps({"answers": answers, **extra}).encode()


def _body(api_format: str, answers: object, **extra: object) -> bytes:
    if api_format == "decisions":
        return _native_body(answers, **extra)
    return _chat_body({"answers": answers}, **extra)


def _provider_usage(api_format: str, usage: Mapping[str, object]) -> dict[str, object]:
    """Chat-shaped token counts in the wire keys of the selected format."""
    if api_format == "chat_completions":
        return dict(usage)
    names = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}
    return {names.get(k, k): v for k, v in usage.items()}


class _BlockingHttpResp:
    """urlopen response whose read blocks until close releases it."""

    def __init__(self, entered: threading.Event) -> None:
        self._entered = entered
        self._release = threading.Event()
        self.closed = threading.Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        self._entered.set()
        assert self._release.wait(timeout=10), "http release never set"
        if self.closed.is_set():
            raise OSError("closed")
        return _chat_body({"answers": {"q": {"type": "noul", "noul": 0.5}}})

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
    monkeypatch.setattr(client._http_opener, "open", _urlopen)
    _decision_env(monkeypatch)

    async def _scenario() -> None:
        call = asyncio.ensure_future(
            client.decide(
                {"s": 1},
                {"q": {"type": "noul", "instructions": "x"}},
                decision_type="t",
                session_id="s",
            )
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
            client.decide(
                {"s": 1},
                {"q": {"type": "noul", "instructions": "x"}},
                decision_type="t",
                session_id="s",
            )
        )
        assert await asyncio.to_thread(entered.wait, 3)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return seen.get("executor")

    monkeypatch.setattr(client, "_persist", _noop)
    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(client._http_opener, "open", _urlopen)
    _decision_env(monkeypatch)
    assert asyncio.run(_run_and_close()) is client._http_pool


_HTTP_QUESTIONS: dict[str, JSONValue] = {
    "q": {"type": "noul", "instructions": "Estimate whether the proposition holds."},
    "pick": {"type": "choice", "instructions": "Select a label.", "criteria": {"a": "A", "b": "B"}},
}
_HTTP_ANSWERS: dict[str, JSONValue] = {
    "q": {"type": "noul", "noul": 0.75},
    "pick": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.8},
}


def _http_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes | None, api_format: str = "chat_completions"
) -> tuple[JevClient, dict[str, object]]:
    import urllib.request

    from app.decision_client import _SidecarUnavailable

    _decision_env(monkeypatch, api_format)
    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path, runtime_path=tmp_path / "no-runtime.ts")

    def _missing(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        raise _SidecarUnavailable("no sidecar")

    def _urlopen(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        seen["request"] = request
        seen["timeout"] = timeout
        assert body is not None
        return io.BytesIO(body)

    def _persist(**kwargs: object) -> None:
        seen["persist"] = kwargs

    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(client, "_persist", _persist)
    if body is not None:
        monkeypatch.setattr(client._http_opener, "open", _urlopen)
    return client, seen


def _http_run(
    client: JevClient, questions: Mapping[str, JSONValue] = _HTTP_QUESTIONS
) -> dict[str, dict[str, JSONValue]]:
    return asyncio.run(client.decide({"source": "test evidence"}, questions, decision_type="t", session_id="s"))


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("host", ["decision.test", "localhost", "127.0.0.1", "[::1]"])
def test_http_decision_rejects_http_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, host: str
) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
    monkeypatch.setenv("DECISION_API_URL", f"http://{host}/v1/decide?token={_DECISION_KEY}")
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_run(client)
        assert str(exc.value) == "decide: decision_request_failed: DECISION_API_URL must use HTTPS"
        assert _DECISION_KEY not in str(exc.value)
        assert seen == {}
    finally:
        client._http_pool.shutdown(wait=True)


class _RedirectHttpResp(io.BytesIO):
    """HTTP response at urllib's socket boundary; the real opener handles redirects."""

    def __init__(self, body: bytes, code: int, location: str | None) -> None:
        super().__init__(body)
        self.code = code
        self.msg = "test response"
        self.headers = HTTPMessage()
        if location is not None:
            self.headers["Location"] = location

    def info(self) -> HTTPMessage:
        return self.headers


def _redirect_http_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_format: str,
    replies: Sequence[tuple[int, str | None]],
) -> tuple[JevClient, dict[str, object], list[object], list[_RedirectHttpResp]]:
    import urllib.request

    client, seen = _http_client(tmp_path, monkeypatch, None, api_format)
    requests: list[object] = []
    responses: list[_RedirectHttpResp] = []

    def _open(_handler: urllib.request.AbstractHTTPHandler, request: urllib.request.Request) -> _RedirectHttpResp:
        assert len(requests) < len(replies), "unexpected redirect transport call"
        code, location = replies[len(requests)]
        requests.append(request)
        body = _body(api_format, _HTTP_ANSWERS) if code == 200 else f"redirect {_DECISION_KEY}".encode()
        response = _RedirectHttpResp(body, code, location)
        responses.append(response)
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", _open)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", _open)
    return client, seen, requests, responses


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("host", ["decision.test", "localhost", "127.0.0.1", "[::1]"])
@pytest.mark.parametrize("secure_hop", [False, True])
def test_http_decision_rejects_redirect_downgrade_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, host: str, secure_hop: bool
) -> None:
    import urllib.request

    replies: list[tuple[int, str | None]] = [(302, "/secure")] if secure_hop else []
    replies.append((302, f"http://{host}/decide?token={_DECISION_KEY}"))
    client, seen, requests, responses = _redirect_http_client(tmp_path, monkeypatch, api_format, replies)
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_run(client)
        assert str(exc.value) == "decide: decision_request_failed: decision redirects must use HTTPS"
        assert _DECISION_KEY not in str(exc.value)
        assert len(requests) == (2 if secure_hop else 1)
        for request in requests:
            assert isinstance(request, urllib.request.Request)
            assert request.full_url.startswith("https://")
        assert all(response.closed for response in responses)
        assert "persist" not in seen
    finally:
        client._http_pool.shutdown(wait=True)


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("status", [301, 302, 303])
@pytest.mark.parametrize("location", ["/resolved", "https://decisions.test/resolved", "https://other.test/resolved"])
def test_http_decision_follows_secure_redirect_with_native_request_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, status: int, location: str
) -> None:
    import urllib.parse
    import urllib.request

    client, seen, requests, responses = _redirect_http_client(
        tmp_path, monkeypatch, api_format, [(status, location), (200, None)]
    )
    try:
        assert _http_run(client)["q"] == {"kind": "noul", "probability": 0.75}
        assert len(requests) == 2
        first, redirected = requests
        assert isinstance(first, urllib.request.Request)
        assert isinstance(redirected, urllib.request.Request)
        assert first.get_method() == "POST"
        assert isinstance(first.data, bytes)
        body = json.loads(first.data)
        state = {"source": "test evidence"}
        if api_format == "decisions":
            assert first.full_url == _NATIVE_URL
            assert body == {"model": _NATIVE_MODEL, "state": state, "questions": _HTTP_QUESTIONS}
        else:
            assert first.full_url == _DECISION_URL
            assert set(body) == {"model", "messages", "stream"}
            assert body["model"] == _DECISION_MODEL and body["stream"] is False
            system, user = body["messages"]
            assert system["role"] == "system" and system["content"].startswith("You are the decision authority.")
            assert user["role"] == "user"
            assert json.loads(user["content"]) == {"state": state, "questions": _HTTP_QUESTIONS}
        assert first.get_header("Authorization") == f"Bearer {_DECISION_KEY}"
        assert first.get_header("Content-type") == "application/json"
        assert redirected.full_url == urllib.parse.urljoin(first.full_url, location)
        assert redirected.get_method() == "GET" and redirected.data is None

        def _origin(url: str) -> tuple[str, str, int | None]:
            parts = urllib.parse.urlsplit(url)
            default = 443 if parts.scheme == "https" else 80 if parts.scheme == "http" else None
            return (parts.scheme.lower(), (parts.hostname or "").lower(), parts.port or default)

        if _origin(redirected.full_url) == _origin(first.full_url):
            assert redirected.get_header("Authorization") == f"Bearer {_DECISION_KEY}"
        else:
            assert redirected.get_header("Authorization") is None
        assert redirected.get_header("Content-type") is None
        assert redirected.get_header("Content-length") is None
        assert all(response.closed for response in responses)
        assert "persist" in seen
    finally:
        client._http_pool.shutdown(wait=True)


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("status", [307, 308])
def test_http_decision_retains_urllib_post_redirect_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, status: int
) -> None:
    client, seen, requests, _ = _redirect_http_client(tmp_path, monkeypatch, api_format, [(status, "/secure")])
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_run(client)
        assert str(exc.value) == f"decide: decision_request_failed: HTTP {status}: redirect [redacted]"
        assert _DECISION_KEY not in str(exc.value)
        assert len(requests) == 1
        assert "persist" not in seen
    finally:
        client._http_pool.shutdown(wait=True)


@pytest.mark.parametrize("api_format", _FORMATS)
def test_http_decision_secure_redirect_loop_stops_at_urllib_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str
) -> None:
    client, seen, requests, _ = _redirect_http_client(tmp_path, monkeypatch, api_format, [(302, "/loop")] * 5)
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_run(client)
        assert str(exc.value) == "decide: decision_request_failed: HTTP 302: redirect [redacted]"
        assert _DECISION_KEY not in str(exc.value)
        assert len(requests) == 5
        assert "persist" not in seen
    finally:
        client._http_pool.shutdown(wait=True)


def test_http_decision_uses_configured_chat_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    client, seen = _http_client(
        tmp_path,
        monkeypatch,
        _chat_body(
            {"answers": _HTTP_ANSWERS}, model="provider-revision", usage={"prompt_tokens": 12, "completion_tokens": 7}
        ),
    )
    out = _http_run(client)
    assert out == {
        "pick": {"kind": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.8},
        "q": {"kind": "noul", "probability": 0.75},
    }
    req = seen["request"]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == _DECISION_URL
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {_DECISION_KEY}"
    assert req.get_header("Content-type") == "application/json"
    assert isinstance(req.data, bytes)
    body = json.loads(req.data)
    assert set(body) == {"model", "messages", "stream"}
    assert body["model"] == _DECISION_MODEL and body["stream"] is False
    assert len(body["messages"]) == 2
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].startswith("You are the decision authority.")
    assert user["role"] == "user"
    assert json.loads(user["content"]) == {"state": {"source": "test evidence"}, "questions": _HTTP_QUESTIONS}
    persisted = seen["persist"]
    assert isinstance(persisted, dict)
    assert persisted["via"] == "http"
    assert persisted["jev_in"] == 12 and persisted["jev_out"] == 7
    assert persisted["jev_model"] == "provider-revision"
    assert persisted["raw"] == {
        "answers": _HTTP_ANSWERS,
        "model": "provider-revision",
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }
    assert seen["timeout"] == 60.0


# Request and 200 response examples from the OpenRouter alpha Decisions API reference.
_NATIVE_QUESTIONS: dict[str, JSONValue] = {
    "is_bug": {
        "criteria": {
            "false": "The customer is asking a question or requesting a feature.",
            "true": "The customer describes broken or unexpected product behavior.",
        },
        "instructions": "Is the customer reporting a software defect?",
        "type": "noul",
    },
    "team": {
        "criteria": {
            "account": "Login, permissions, or profile issues.",
            "frontend": "Rendering, layout, or browser compatibility issues.",
            "payments": "Checkout, billing, or payment processing issues.",
        },
        "instructions": "Which team should own this ticket?",
        "type": "choice",
    },
    "urgency": {
        "criteria": ["Can wait for the next release", "Should be fixed this week", "Blocking revenue right now"],
        "instructions": "How urgent is this ticket?",
        "type": "score",
    },
}
_NATIVE_LEGEND: dict[str, JSONValue] = {
    "0": "Can wait for the next release",
    "1": "Should be fixed this week",
    "2": "Blocking revenue right now",
}
_NATIVE_ANSWERS: dict[str, JSONValue] = {
    "is_bug": {"noul": 0.96, "type": "noul"},
    "team": {
        "choice": "payments",
        "confidence": 0.75,
        "probabilities": {"account": 0, "frontend": 0.16, "payments": 0.84},
        "type": "choice",
    },
    "urgency": {
        "confidence": 0.99,
        "legend": _NATIVE_LEGEND,
        "probabilities": {"0": 0, "1": 0.01, "2": 0.99},
        "score": 1.99,
        "type": "score",
    },
}


def test_http_decision_uses_configured_native_decisions_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    client, seen = _http_client(
        tmp_path,
        monkeypatch,
        _native_body(
            _NATIVE_ANSWERS,
            id="gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
            model="typesafe/jev-1.13-20260917",
            provider="TypeSafe",
            usage={"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
        ),
        "decisions",
    )
    state: dict[str, JSONValue] = {
        "customer_tier": "enterprise",
        "ticket": "My checkout page shows a blank screen after I click Pay. I have tried two browsers.",
    }
    out = asyncio.run(client.decide(state, _NATIVE_QUESTIONS, decision_type="t", session_id="s"))
    assert out == {
        "is_bug": {"kind": "noul", "probability": 0.96},
        "team": {
            "kind": "choice",
            "choice": "payments",
            "probabilities": {"account": 0.0, "frontend": 0.16, "payments": 0.84},
            "confidence": 0.75,
        },
        "urgency": {
            "kind": "score",
            "score": 1.99,
            "confidence": 0.99,
            "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
            "raw": _NATIVE_LEGEND,
        },
    }
    req = seen["request"]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == _NATIVE_URL
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {_DECISION_KEY}"
    assert req.get_header("Content-type") == "application/json"
    assert isinstance(req.data, bytes)
    assert json.loads(req.data) == {"model": _NATIVE_MODEL, "state": state, "questions": _NATIVE_QUESTIONS}
    persisted = seen["persist"]
    assert isinstance(persisted, dict)
    assert persisted["via"] == "http"
    assert persisted["jev_in"] == 476 and persisted["jev_out"] == 70
    assert persisted["jev_model"] == "typesafe/jev-1.13-20260917"
    assert persisted["raw"] == {
        "answers": _NATIVE_ANSWERS,
        "model": "typesafe/jev-1.13-20260917",
        "usage": {"input_tokens": 476, "output_tokens": 70},
    }
    assert seen["timeout"] == 60.0


@pytest.mark.parametrize("api_format", _FORMATS)
def test_http_decision_format_comes_only_from_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str
) -> None:
    """The native URL path and model name never override the configured format."""
    import urllib.request

    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
    monkeypatch.setenv("DECISION_API_URL", _NATIVE_URL)
    monkeypatch.setenv("DECISION_MODEL", _NATIVE_MODEL)
    _http_run(client)
    req = seen["request"]
    assert isinstance(req, urllib.request.Request) and isinstance(req.data, bytes)
    assert req.full_url == _NATIVE_URL
    body = json.loads(req.data)
    expected = {"model", "state", "questions"} if api_format == "decisions" else {"model", "messages", "stream"}
    assert set(body) == expected and body["model"] == _NATIVE_MODEL


@pytest.mark.parametrize("api_format", _FORMATS)
def test_http_decision_reasoning_effort_field_per_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str
) -> None:
    """Chat Completions gets reasoning_effort unchanged; the Decisions API documents no reasoning field."""
    import urllib.request

    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
    monkeypatch.setenv("REASONING_EFFORT", "xhigh")
    _http_run(client)
    req = seen["request"]
    assert isinstance(req, urllib.request.Request) and isinstance(req.data, bytes)
    body = json.loads(req.data)
    if api_format == "decisions":
        assert set(body) == {"model", "state", "questions"}
    else:
        assert set(body) == {"model", "messages", "stream", "reasoning_effort"}
        assert body["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("value", ["XHIGH", "x high", "1high", "a" * 33])
def test_http_decision_rejects_bad_reasoning_effort_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, value: str
) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
    monkeypatch.setenv("REASONING_EFFORT", value)
    with pytest.raises(RuntimeError, match=r"decide: decision_request_failed: REASONING_EFFORT must match"):
        _http_run(client)
    assert "request" not in seen and "persist" not in seen


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("reported", [None, "", 7, ["provider-revision"]])
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": "12", "completion_tokens": True}, {"prompt_tokens": 0}])
def test_http_decision_does_not_invent_usage_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, reported: object, usage: object
) -> None:
    extra: dict[str, object] = {}
    if isinstance(usage, dict):
        extra["usage"] = _provider_usage(api_format, usage)
    if reported is not None:
        extra["model"] = reported
    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS, **extra), api_format)
    _http_run(client)
    persisted = seen["persist"]
    assert isinstance(persisted, dict)
    configured = _NATIVE_MODEL if api_format == "decisions" else _DECISION_MODEL
    assert persisted["jev_model"] == configured
    assert persisted["jev_out"] is None
    assert persisted["jev_in"] == (0 if usage == {"prompt_tokens": 0} else None)
    raw = persisted["raw"]
    assert isinstance(raw, dict)
    assert raw["model"] == configured
    no_usage: dict[str, JSONValue] | None = None if usage is None else {}
    assert raw.get("usage") == ({"input_tokens": 0} if usage == {"prompt_tokens": 0} else no_usage)


_MALFORMED_CHAT: list[bytes] = [
    b"not JSON",
    b"[]",
    b'{"choices":[]}',
    b'{"choices":[{}]}',
    b'{"choices":[{"message":{"content":{}}}]}',
    b'{"choices":[{"message":{"content":" "}}]}',
    b'{"choices":[{"message":{"content":"{}","refusal":"denied"}}]}',
    b'{"choices":[{"message":{"content":"{}"},"finish_reason":"length"}]}',
    _chat_body("not JSON"),
    _chat_body([]),
    _chat_body({"answers": []}),
    _chat_body('{"answers":{"q":{"type":"noul","noul":NaN}}}'),
    _chat_body('{"answers":{"q":{"type":"noul","noul":Infinity}}}'),
    _chat_body('{"answers":{"q":{"type":"noul","noul":-Infinity}}}'),
    _native_body(_HTTP_ANSWERS),
]
_MALFORMED_NATIVE: list[bytes] = [
    b"not JSON",
    b"[]",
    b'"answers"',
    b"{}",
    b'{"answers":null}',
    _native_body([]),
    _native_body("answers"),
    b'{"answers":{"q":{"type":"noul","noul":NaN}}}',
    b'{"answers":{"q":{"type":"noul","noul":Infinity}}}',
    b'{"answers":{"q":{"type":"noul","noul":-Infinity}}}',
    _chat_body({"answers": _HTTP_ANSWERS}),
]


@pytest.mark.parametrize(
    ("api_format", "body"),
    [("chat_completions", b) for b in _MALFORMED_CHAT] + [("decisions", b) for b in _MALFORMED_NATIVE],
)
def test_http_decision_rejects_malformed_provider_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, body: bytes
) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, body, api_format)
    with pytest.raises(TypeError, match="malformed_decision_response"):
        _http_run(client)
    assert "persist" not in seen


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_http_decision_rejects_non_utf8_provider_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, encoding: str
) -> None:
    """Invalid UTF-8, including a well-formed UTF-16/32 body, is the existing malformed-body error."""
    for body in (b"\xff", _body(api_format, _HTTP_ANSWERS).decode().encode(encoding)):
        client, seen = _http_client(tmp_path, monkeypatch, body, api_format)
        with pytest.raises(TypeError, match=r"^decide: malformed_decision_response: body is not JSON$") as exc:
            _http_run(client)
        assert exc.value.__suppress_context__ is True
        assert "persist" not in seen


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize(
    ("answers", "error"),
    [
        ({"q": {"type": "noul", "noul": 0.5}}, "decision answers do not match questions"),
        (
            {**_HTTP_ANSWERS, "extra": {"type": "noul", "noul": 0.5}},
            "decision answers do not match questions",
        ),
        (
            {
                **_HTTP_ANSWERS,
                "pick": {"type": "choice", "choice": "c", "probabilities": {"a": 0.2, "b": 0.8}, "confidence": 0.8},
            },
            "malformed_decision_answer for pick",
        ),
        (
            {
                **_HTTP_ANSWERS,
                "pick": {"type": "choice", "choice": "a", "probabilities": {"a": -0.1, "b": 1.1}, "confidence": 0.8},
            },
            "malformed_decision_answer for pick",
        ),
        (
            {
                **_HTTP_ANSWERS,
                "pick": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 1.0},
            },
            "malformed_decision_answer for pick",
        ),
        (
            {**_HTTP_ANSWERS, "q": {"type": "noul", "noul": -0.001}},
            "malformed_decision_answer for q",
        ),
        (
            {**_HTTP_ANSWERS, "q": {"type": "noul", "noul": 1.001}},
            "malformed_decision_answer for q",
        ),
        (
            {**_HTTP_ANSWERS, "q": {"type": "noul", "noul": True}},
            "malformed_decision_answer for q",
        ),
    ],
)
def test_http_decision_preserves_answer_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, answers: object, error: str
) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, answers), api_format)
    with pytest.raises(ValueError, match=error):
        _http_run(client)
    assert "persist" not in seen


@pytest.mark.parametrize("api_format", _FORMATS)
@settings(max_examples=30, derandomize=True, deadline=None)
@given(
    st.one_of(
        st.just(0.0), st.just(1.0), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
)
def test_prop_http_noul_probability_roundtrip(api_format: str, probability: float) -> None:
    # Each example owns its monkeypatch; no network or mutable pytest fixture crosses examples.
    with pytest.MonkeyPatch.context() as monkeypatch:
        client, _ = _http_client(
            Path("/tmp"), monkeypatch, _body(api_format, {"q": {"type": "noul", "noul": probability}}), api_format
        )
        try:
            assert _http_run(client, {"q": {"type": "noul"}}) == {"q": {"kind": "noul", "probability": probability}}
        finally:
            client._http_pool.shutdown(wait=True)


@pytest.mark.parametrize("api_format", _FORMATS)
@seed(20261017)
@settings(max_examples=60, deadline=None)
@given(
    scheme=st.one_of(
        st.sampled_from(["https", "HTTPS", "hTtPs", "http", "HTTP", "ftp", "file", "ws", "wss"]),
        st.from_regex(r"[A-Za-z][A-Za-z0-9+.-]{0,12}", fullmatch=True),
    ),
    host=st.sampled_from(["decision.test", "localhost", "127.0.0.1", "[::1]"]),
)
def test_prop_http_decision_accepts_only_https(api_format: str, scheme: str, host: str) -> None:
    import urllib.request

    with pytest.MonkeyPatch.context() as monkeypatch:
        client, seen = _http_client(Path("/tmp"), monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
        url = f"{scheme}://{host}/v1/decide"
        monkeypatch.setenv("DECISION_API_URL", url)
        try:
            if scheme.lower() == "https":
                out = _http_run(client)
                assert out["q"] == {"kind": "noul", "probability": 0.75}
                request = seen["request"]
                assert isinstance(request, urllib.request.Request)
                assert request.full_url == url
                assert request.get_header("Authorization") == f"Bearer {_DECISION_KEY}"
                assert "persist" in seen
            else:
                with pytest.raises(RuntimeError) as exc:
                    _http_run(client)
                assert str(exc.value) == "decide: decision_request_failed: DECISION_API_URL must use HTTPS"
                assert _DECISION_KEY not in str(exc.value)
                assert seen == {}
        finally:
            client._http_pool.shutdown(wait=True)


@pytest.mark.parametrize("api_format", _FORMATS)
@pytest.mark.parametrize("setting", ["DECISION_API_FORMAT", "DECISION_API_URL", "DECISION_API_KEY", "DECISION_MODEL"])
@pytest.mark.parametrize("value", [None, " \t"])
def test_http_decision_requires_each_live_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str, setting: str, value: str | None
) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS), api_format)
    if value is None:
        monkeypatch.delenv(setting)
    else:
        monkeypatch.setenv(setting, value)
    with pytest.raises(RuntimeError, match=f"decision_request_failed: missing {setting}"):
        _http_run(client)
    assert "request" not in seen and "persist" not in seen


@pytest.mark.parametrize(
    "value", ["openai", "chat", "decision", "Decisions", "responses", "chat_completions,decisions"]
)
def test_http_decision_rejects_unsupported_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    client, seen = _http_client(tmp_path, monkeypatch, _native_body(_HTTP_ANSWERS), "decisions")
    monkeypatch.setenv("DECISION_API_FORMAT", value)
    with pytest.raises(RuntimeError, match="decision_request_failed: unsupported DECISION_API_FORMAT"):
        _http_run(client)
    assert "request" not in seen and "persist" not in seen


def test_http_decision_redacts_key_from_provider_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request
    from email.message import Message

    client, seen = _http_client(tmp_path, monkeypatch, b"")

    def _error(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        raise urllib.error.HTTPError(
            _DECISION_URL, 401, "Unauthorized", Message(), io.BytesIO(f"invalid {_DECISION_KEY}".encode())
        )

    monkeypatch.setattr(client._http_opener, "open", _error)
    with pytest.raises(RuntimeError, match=r"decision_request_failed: HTTP 401: invalid \[redacted\]") as exc:
        _http_run(client)
    assert _DECISION_KEY not in str(exc.value)
    assert "persist" not in seen


def test_http_decision_redacts_key_from_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    client, _ = _http_client(tmp_path, monkeypatch, b"")

    def _error(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        raise OSError(f"network error {_DECISION_KEY}")

    monkeypatch.setattr(client._http_opener, "open", _error)
    with pytest.raises(RuntimeError, match=r"decision_request_failed: network error \[redacted\]") as exc:
        _http_run(client)
    assert _DECISION_KEY not in str(exc.value)
    assert exc.value.__suppress_context__ is True


@pytest.mark.parametrize(("via", "api_format"), [("http", "chat_completions"), ("http", "decisions"), ("stub", None)])
@pytest.mark.parametrize(("input_tokens", "output_tokens"), [(12, 7), (None, None), (0, None), (None, 0)])
def test_decision_persists_model_and_endpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
    api_format: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    from types import SimpleNamespace

    model = "provider-revision"
    raw: dict[str, JSONValue] = {"answers": _HTTP_ANSWERS, "model": model}
    normalized: dict[str, JSONValue] = {}
    usage: dict[str, object] = {}
    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
        usage["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
        usage["completion_tokens"] = output_tokens
    if normalized:
        raw["usage"] = normalized
    if api_format is not None:
        extra: dict[str, object] = {"model": model}
        if usage:
            extra["usage"] = _provider_usage(api_format, usage)
        client, _ = _http_client(tmp_path, monkeypatch, _body(api_format, _HTTP_ANSWERS, **extra), api_format)
        monkeypatch.setattr(client, "_persist", JevClient._persist.__get__(client, JevClient))
    else:

        def _transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
            return raw

        client = JevClient(transport=_transport, data_root=tmp_path)
    monkeypatch.setenv("DECISION_API_URL", "https://user:secret@decisions.test:8443/custom/chat?secret=query#fragment")
    domain: dict[str, object] = {}
    events: list[dict[str, object]] = []
    models: list[dict[str, object]] = []

    def _domain(record: object, **kwargs: object) -> None:
        domain.update(kwargs)

    def _event(event_type: str, **kwargs: object) -> None:
        events.append({"event_type": event_type, **kwargs})

    def _model(**kwargs: object) -> float:
        models.append(kwargs)
        return 0.0

    recorder = SimpleNamespace(enabled=True, record_event=_event, record_model_call=_model)
    monkeypatch.setattr(client, "_persist_domain", _domain)
    monkeypatch.setattr("app.decision_client.get_current_recorder", lambda: recorder)
    _http_run(client)
    provider = "https://decisions.test:8443/custom/chat" if via == "http" else None
    assert domain["provider"] == provider
    assert events[0]["event_type"] == "jev_decision"
    assert events[0]["model"] == model
    metadata = events[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["provider"] == provider and metadata["via"] == via
    assert metadata["model"] == model
    assert metadata["usage"] == {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if via == "http":
        assert len(models) == 1
        assert models[0]["provider"] == provider and models[0]["model"] == model
        if input_tokens is not None and output_tokens is not None:
            usage["total_tokens"] = input_tokens + output_tokens
        assert models[0]["usage"] == (usage or None)
    else:
        assert models == []
    encoded = json.dumps({"events": events, "models": models, "domain": domain})
    assert _DECISION_KEY not in encoded
    assert "user:secret" not in encoded and "secret=query" not in encoded


def test_injected_transport_needs_no_live_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DECISION_API_FORMAT", "DECISION_API_URL", "DECISION_API_KEY", "DECISION_MODEL"):
        monkeypatch.delenv(key, raising=False)
    client, _ = _capturing_client(tmp_path, monkeypatch)
    assert _http_run(client, {"q": {"type": "noul"}}) == {"q": {"kind": "noul", "probability": 0.5}}


def _answers(questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Stub decision answers: first sorted option wins each choice. Noul 0.5."""
    out: dict[str, JSONValue] = {}
    for qid, q in questions.items():
        criteria = q.get("criteria") if isinstance(q, dict) else None
        if isinstance(criteria, dict):
            names = sorted(criteria)
            probs: dict[str, JSONValue] = {n: 0.5 for n in names}
            out[qid] = {
                "type": "choice",
                "choice": names[0],
                "probabilities": probs,
                "confidence": 0.5,
            }
        else:
            out[qid] = {"type": "noul", "noul": 0.5}
    return {"answers": out}


def _capturing_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[JevClient, list[tuple[object, dict[str, JSONValue]]]]:
    """Client whose transport records the exact (state, questions) JEV receives."""
    seen: list[tuple[object, dict[str, JSONValue]]] = []

    def _transport(state: object, questions: dict[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append((state, questions))
        return _answers(questions)

    client = JevClient(transport=_transport, data_root=tmp_path)

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)
    return client, seen


_MARKERS = (
    "RAW_CONTENT_MARKER",
    "PROVENANCE_MARKER",
    "META_MARKER",
    "REASONING_MARKER",
    "INTAKE_MARKER",
)
_GUIDANCE_LINE = "Do not repeat a tool call with identical arguments to an earlier attempt."


def _raw_context() -> tuple[list[JSONValue], list[JSONValue]]:
    """Scheduler-shaped ctx (non-record, admitted row, 12 observations, guidance) plus one attempt with raw payload."""
    admitted: dict[str, JSONValue] = {
        "evidence_id": "ev-1",
        "session_id": "s",
        "wave_id": 1,
        "source_name": "SEC",
        "subject": "AAPL",
        "claim_text": "c" * 201,
        "content": "RAW_CONTENT_MARKER " * 400,
        "provenance": {"kind": "sec_source", "accession_no": "PROVENANCE_MARKER"},
        "metadata": {"identity_key": "META_MARKER"},
        "job_id": "j-admit",
    }
    observations: list[JSONValue] = [
        {
            "tool": f"t{i}",
            "job_id": f"j{i}",
            "outcome_summary": "s" * 2000,
            "error": "e" * 500 if i == 11 else None,
            "error_type": "tool_error" if i == 11 else None,
            "tool_result_ref": f"r{i}",
        }
        for i in range(12)
    ]
    guidance: dict[str, JSONValue] = {
        "type": "selection_guidance",
        "intake_evidence": ["ev-1: INTAKE_MARKER"],
        "instruction": _GUIDANCE_LINE,
        "prior_calls": [f"p{i}" for i in range(12)],
    }
    attempts: list[JSONValue] = [
        {
            "tool": "a",
            "arguments": {"q": "x"},
            "outcome_summary": "o" * 200,
            "error": "f" * 201,
            "error_type": "tool_error",
            "reasoning": "REASONING_MARKER",
            "confidence": 0.9,
            "job_id": "j9",
            "evidence_id": "ev-9",
            "tool_result_ref": "r9",
            "generate_ms": 1.0,
            "tool_ms": 2.0,
            "assess_ms": 3.0,
        }
    ]
    return ["NOT_A_RECORD", admitted, *observations, guidance], attempts


def _compact_evidence_expected(with_guidance: bool) -> list[JSONValue]:
    observed: list[JSONValue] = [
        {
            "tool": f"t{i}",
            "job_id": f"j{i}",
            "tool_result_ref": f"r{i}",
            "outcome_summary": "s" * 200,
            "error": "e" * 200 if i == 11 else None,
            "error_type": "tool_error" if i == 11 else None,
        }
        for i in range(2, 12)
    ]
    guidance: list[JSONValue] = [
        {
            "type": "selection_guidance",
            "instruction": _GUIDANCE_LINE,
            "prior_calls": [f"p{i}" for i in range(2, 12)],
        }
    ]
    admitted: JSONValue = {
        "evidence_id": "ev-1",
        "source_name": "SEC",
        "subject": "AAPL",
        "claim_text": "c" * 200,
    }
    return [admitted, *observed, *(guidance if with_guidance else [])]


def test_select_tool_sends_only_compact_state_and_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JEV select request: 4-field evidence, bounded observations, compact attempts, compact options."""
    client, seen = _capturing_client(tmp_path, monkeypatch)
    registry: list[dict[str, JSONValue]] = [
        {
            "name": "a",
            "description": "DESCRIPTION_MARKER " * 50,
            "domain": "SEC",
            "purpose": "Read  filings",
            "intent": "INTENT_MARKER",
            "keyInputs": "req(x)",
            "outputKind": "rows",
            "useWhen": "need text",
            "avoidWhen": "no accession",
            "prerequisites": "needs search_id",
            "pitSupport": "PIT-scoped",
            "conflicts": "b",
            "nextTools": "c",
            "parameters": {"type": "object"},
        }
    ]
    evidence, attempts = _raw_context()
    before = copy.deepcopy((registry, evidence, attempts))
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}

    decision = asyncio.run(client.select_tool("obj", node, registry, evidence, attempts, session_id="s"))

    assert decision.tool_name == "a"
    assert (registry, evidence, attempts) == before
    state, questions = seen[0]
    assert isinstance(state, dict)
    assert state["evidence"] == _compact_evidence_expected(with_guidance=True)
    assert state["attempts"] == [
        {
            "tool": "a",
            "arguments": {"q": "x"},
            "job_id": "j9",
            "evidence_id": "ev-9",
            "tool_result_ref": "r9",
            "outcome_summary": "o" * 200,
            "error": "f" * 200,
            "error_type": "tool_error",
        }
    ]
    wire = json.dumps({"state": state, "questions": questions})
    for marker in (*_MARKERS, "DESCRIPTION_MARKER", "INTENT_MARKER"):
        assert marker not in wire
    sel = questions["tool_selection"]
    assert isinstance(sel, dict)
    criteria = sel["criteria"]
    assert isinstance(criteria, dict)
    assert (
        criteria["a"]
        == "a Purpose: Read filings. Use: need text. Avoid: no accession. Needs: needs search_id. PIT: PIT-scoped."
    )
    assert {"reasoning_required", "node_resolved"} <= set(criteria)
    prompt = sel["instructions"]
    assert isinstance(prompt, str)
    assert "Evidence on hand: 15 item(s) [ev-1,j0," in prompt
    assert f"Observed t3 ref=r3: {'s' * 200}." in prompt
    assert f"Prior attempt a failed: {'f' * 200}." in prompt


def test_select_tool_compacts_tuple_evidence_and_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuple inputs (valid Sequence) reach JEV compacted, never dropped."""
    client, seen = _capturing_client(tmp_path, monkeypatch)
    evidence, attempts = _raw_context()
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    registry: list[dict[str, JSONValue]] = [{"name": "a"}]

    asyncio.run(client.select_tool("obj", node, registry, tuple(evidence), tuple(attempts), session_id="s"))

    state, _questions = seen[0]
    assert isinstance(state, dict)
    assert state["evidence"] == _compact_evidence_expected(with_guidance=True)
    tool_attempts = state["attempts"]
    assert isinstance(tool_attempts, list) and len(tool_attempts) == 1
    assert isinstance(tool_attempts[0], dict) and tool_attempts[0]["arguments"] == {"q": "x"}


@pytest.mark.parametrize("field", ["evidence", "attempts"])
def test_records_rejects_mapping(field: str) -> None:
    """A dict for evidence/attempts is a caller defect: TypeError before select_tool/assess_result request JEV."""
    from app.decision_client import _records

    with pytest.raises(TypeError, match=f"jev {field} must be a sequence of records, not a mapping"):
        _records({"evidence_id": "ev-1"}, field)


def test_select_tool_none_context_sends_empty_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, seen = _capturing_client(tmp_path, monkeypatch)
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    asyncio.run(client.select_tool("obj", node, [{"name": "a"}], None, None, session_id="s"))
    state, _questions = seen[0]
    assert isinstance(state, dict) and state["evidence"] == [] and state["attempts"] == []


@pytest.mark.parametrize("prior_calls", [None, "p1"])
def test_select_tool_invalid_prior_calls_send_empty_list(
    prior_calls: JSONValue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, seen = _capturing_client(tmp_path, monkeypatch)
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    guidance: list[JSONValue] = [
        {
            "type": "selection_guidance",
            "instruction": _GUIDANCE_LINE,
            "prior_calls": prior_calls,
        }
    ]
    asyncio.run(client.select_tool("obj", node, [{"name": "a"}], guidance, None, session_id="s"))
    state, _questions = seen[0]
    assert isinstance(state, dict)
    assert state["evidence"] == [{"type": "selection_guidance", "instruction": _GUIDANCE_LINE, "prior_calls": []}]


def test_manifest_line_needs_no_description() -> None:
    from app.decision_client import _manifest_line

    assert _manifest_line({"name": "bare"}) == "bare"
    assert _manifest_line({"name": "t", "useWhen": "  ", "pitSupport": "PIT-blind"}) == "t PIT: PIT-blind."


def test_assess_result_keeps_raw_outcome_and_compacts_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JEV assess request: raw current outcome capped at 2000, evidence compacted, caller untouched."""
    client, seen = _capturing_client(tmp_path, monkeypatch)
    evidence, _ = _raw_context()
    ctx = evidence[:-1]  # scheduler passes no guidance to assessment
    before = copy.deepcopy(ctx)
    outcome: dict[str, JSONValue] = {
        "tool_name": "a",
        "content": "x" * 5000,
        "error": None,
        "tool_result_id": "r1",
    }
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}

    out = asyncio.run(client.assess_result(node, outcome, ctx, session_id="s", job_id="j1"))

    assert out["continuation"] == "continue_research"
    assert ctx == before
    state, _questions = seen[0]
    assert isinstance(state, dict)
    assert state["outcome"] == {
        "tool": "a",
        "content": "x" * 2000,
        "error": None,
        "error_type": None,
        "tool_result_id": "r1",
    }
    assert state["evidence"] == _compact_evidence_expected(with_guidance=False)
    wire = json.dumps(state)
    for marker in _MARKERS:
        assert marker not in wire


_RAW_KEYS = st.sampled_from(["content", "reasoning", "provenance", "metadata", "generate_ms", "outcome"])


@settings(max_examples=200, derandomize=True)
@given(
    st.text(max_size=400),
    st.text(max_size=400),
    st.dictionaries(_RAW_KEYS, st.text(max_size=50), max_size=6),
)
def test_prop_compact_projection_clips_text_and_drops_raw(text: str, other: str, raw: dict[str, str]) -> None:
    """Projection keeps exactly the approved fields. Text fields equal their first 200 chars."""
    from app.decision_client import _compact_attempts, _compact_context

    admitted: dict[str, JSONValue] = {
        **raw,
        "evidence_id": "ev",
        "source_name": "S",
        "subject": "X",
        "claim_text": text,
    }
    observed: dict[str, JSONValue] = {
        **raw,
        "tool": "t",
        "outcome_summary": text,
        "error": other,
    }
    attempt: dict[str, JSONValue] = {
        **raw,
        "tool": "t",
        "arguments": {"k": "v"},
        "outcome_summary": text,
        "error": other,
    }
    assert _compact_context([admitted, observed]) == [
        {
            "evidence_id": "ev",
            "source_name": "S",
            "subject": "X",
            "claim_text": text[:200],
        },
        {
            "tool": "t",
            "job_id": None,
            "tool_result_ref": None,
            "outcome_summary": text[:200],
            "error": other[:200],
            "error_type": None,
        },
    ]
    assert _compact_attempts([attempt]) == [
        {
            "tool": "t",
            "arguments": {"k": "v"},
            "job_id": None,
            "evidence_id": None,
            "tool_result_ref": None,
            "outcome_summary": text[:200],
            "error": other[:200],
            "error_type": None,
        }
    ]


def _final_stub(prob: float, state: str) -> Callable[[object, Mapping[str, JSONValue]], dict[str, JSONValue]]:
    def _run(state_arg: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        assert sorted(questions.keys()) == ["evidence_state", "support"]
        return {
            "answers": {
                "support": {"type": "noul", "noul": prob},
                "evidence_state": {
                    "type": "choice",
                    "choice": state,
                    "probabilities": {
                        "sufficient_support": 0.8 if state == "sufficient_support" else 0.05,
                        "sufficient_contradiction": 0.8 if state == "sufficient_contradiction" else 0.05,
                        "conflicted": 0.8 if state == "conflicted" else 0.05,
                        "insufficient": 0.8 if state == "insufficient" else 0.05,
                    },
                    "confidence": 0.9,
                },
            }
        }

    return _run


class _FinalJev(JevClient):
    def __init__(self, tmp_path: Path, prob: float, state: str) -> None:
        root = Path(__file__).resolve().parent.parent
        super().__init__(
            data_root=tmp_path,
            runtime_path=root / "decision" / "runtime.ts",
            transport=_final_stub(prob, state),
        )
        self.decide_types: list[tuple[str, dict[str, JSONValue]]] = []

    @override
    def _persist(self, **kwargs: object) -> None:
        return None

    @override
    async def decide(
        self,
        state: object,
        questions: Mapping[str, JSONValue],
        *,
        decision_type: str,
        session_id: str,
        node_id: str | None = None,
        job_id: str | None = None,
        choice_options: Mapping[str, Mapping[str, str]] | None = None,
    ) -> dict[str, dict[str, JSONValue]]:
        self.decide_types.append((decision_type, dict(questions)))
        return await super().decide(
            state,
            questions,
            decision_type=decision_type,
            session_id=session_id,
            node_id=node_id,
            job_id=job_id,
            choice_options=choice_options,
        )


def _final_client(tmp_path: Path, prob: float, state: str) -> _FinalJev:
    return _FinalJev(tmp_path, prob, state)


def test_assess_final_response_supported(tmp_path: Path) -> None:
    client = _final_client(tmp_path, 0.85, "sufficient_support")
    out = asyncio.run(
        client.assess_final_response(
            "Revenue grew per the filing.",
            [{"id": "ev-1", "content": "10-Q shows revenue grew."}],
            session_id="s1",
            confidence=0.7,
        )
    )
    assert out["support"] == 0.85 and out["support_answer"] == "yes"
    assert out["evidence_state"] == "sufficient_support" and out["confidence"] == 0.7
    assert client.decide_types[0][0] == "final_response_assessment"
    support = client.decide_types[0][1]["support"]
    assert isinstance(support, dict)
    instructions = support["instructions"]
    assert isinstance(instructions, str)
    assert "lacks inventory or cited support" in instructions


def test_assess_final_response_conflicted(tmp_path: Path) -> None:
    client = _final_client(tmp_path, 0.55, "conflicted")
    out = asyncio.run(client.assess_final_response("Growth maybe.", [], session_id="s1"))
    assert out["support_answer"] == "unsure" and out["evidence_state"] == "conflicted"
    assert out["confidence"] is None


def test_assess_final_claim_attributed(tmp_path: Path) -> None:
    client = _final_client(tmp_path, 0.9, "sufficient_support")
    out = asyncio.run(
        client.assess_final_claim(
            "WSJ reported the CEO resigned.",
            "attributed_report",
            [{"id": "ev-2", "content": "WSJ: CEO resigned."}],
            session_id="s1",
        )
    )
    assert out["support_answer"] == "yes" and out["evidence_state"] == "sufficient_support"
    assert client.decide_types[0][0] == "final_claim_assessment"


def test_assess_final_claim_conditional_and_missing(tmp_path: Path) -> None:
    client = _final_client(tmp_path, 0.2, "insufficient")
    out = asyncio.run(
        client.assess_final_claim(
            "If tariffs rise, margins may fall.",
            "conditional_risk",
            [],
            session_id="s1",
        )
    )
    assert out["support_answer"] == "no" and out["evidence_state"] == "insufficient"
    with pytest.raises(ValueError):
        asyncio.run(client.assess_final_claim("  ", "inference", [], session_id="s1"))
    with pytest.raises(ValueError):
        asyncio.run(client.assess_final_claim("x", "  ", [], session_id="s1"))


def test_assess_final_ops_fail_closed() -> None:
    import app.research.kernel_worker as kw

    blank = kw._assess_final_response({"id": "r1", "op": "assess_final_response", "text": "  "})
    assert blank["support"] is None and blank["support_answer"] is None and blank["evidence_state"] == "insufficient"
    assert blank["error"] == "text required"
    noclaim = kw._assess_final_claim({"id": "c1", "op": "assess_final_claim", "claim": {"text": "x"}})
    assert noclaim["support"] is None and noclaim["evidence_state"] == "insufficient"

    class _Boom(JevClient):
        @override
        async def assess_final_response(
            self,
            text: str,
            evidence: Mapping[str, JSONValue] | Sequence[JSONValue] | None = None,
            *,
            session_id: str,
            job_id: str | None = None,
            confidence: float | None = None,
            claims: Sequence[JSONValue] | None = None,
        ) -> dict[str, JSONValue]:
            raise RuntimeError("jev down")

        @override
        async def assess_final_claim(
            self,
            text: str,
            claim_type: str,
            evidence: Mapping[str, JSONValue] | Sequence[JSONValue] | None = None,
            *,
            session_id: str,
            job_id: str | None = None,
            confidence: float | None = None,
        ) -> dict[str, JSONValue]:
            raise RuntimeError("jev down")

    down_r = kw._assess_final_response({"id": "r2", "text": "x", "evidence": []}, jev=_Boom())
    assert down_r["support"] is None and down_r["support_answer"] is None and down_r["evidence_state"] == "insufficient"
    down_c = kw._assess_final_claim(
        {
            "id": "c2",
            "claim": {"text": "x", "claim_type": "inference", "evidence_ids": []},
            "evidence": [],
        },
        jev=_Boom(),
    )
    assert down_c["support"] is None and down_c["evidence_state"] == "insufficient"


def test_assess_final_ops_shape_and_alias(tmp_path: Path) -> None:
    import app.research.kernel_worker as kw

    client = _final_client(tmp_path, 0.8, "sufficient_support")
    resp = kw._assess_final_response(
        {
            "id": "r3",
            "op": "assess_final_response",
            "text": "WSJ reported X; if Y then Z.",
            "claims": [
                {
                    "text": "WSJ reported X.",
                    "claim_type": "attributed_report",
                    "evidence_ids": ["ev-1"],
                },
                {
                    "text": "If Y then Z.",
                    "claim_type": "conditional_risk",
                    "evidence_ids": ["ev-2"],
                    "confidence": 0.6,
                },
            ],
            "evidence": [
                {"id": "ev-1", "content": "WSJ: X."},
                {"id": "ev-2", "content": "Y risk."},
                {"id": "x"},
            ],
            "sessionId": "sess-9",
            "confidence": 0.75,
        },
        jev=client,
    )
    assert resp["id"] == "r3" and resp["support_answer"] == "yes" and "error" not in resp
    assert client.decide_types[0][0] == "final_response_assessment"
    claim = kw._assess_final_claim(
        {
            "id": "c3",
            "op": "assess_final_claim",
            "claim": {
                "text": "WSJ reported X.",
                "claim_type": "attributed_report",
                "evidence_ids": ["ev-1"],
            },
            "evidence": [{"id": "ev-1", "content": "WSJ: X."}],
            "session_id": "sess-9",
        },
        jev=client,
    )
    assert claim["id"] == "c3" and claim["support_answer"] == "yes" and "error" not in claim


def _noop_persist(**kwargs: object) -> None:
    return None


def _runtime_path() -> Path:
    return Path(__file__).resolve().parent.parent / "decision" / "runtime.ts"


class _PipeStdin(io.StringIO):
    """stdin that fails its first writes with a broken pipe, then accepts the retry."""

    def __init__(self, *, fail_writes: int = 0) -> None:
        super().__init__()
        self.writes: list[dict[str, object]] = []
        self._fails = fail_writes

    @override
    def write(self, s: str, /) -> int:
        if self._fails > 0:
            self._fails -= 1
            raise BrokenPipeError("sidecar pipe closed")
        self.writes.append(json.loads(s))
        return len(s)

    @override
    def flush(self) -> None:
        pass


class _ReplyStdout(io.StringIO):
    """readline returns a canned reply line, a ping ack for the last request, or "" for EOF."""

    def __init__(self, stdin: _PipeStdin, reply: str | None) -> None:
        super().__init__()
        self._stdin = stdin
        self._reply = reply

    @override
    def readline(self, size: int | None = -1, /) -> str:
        if self._reply is not None:
            return self._reply
        if not self._stdin.writes:
            return ""
        return json.dumps({"id": self._stdin.writes[-1]["id"], "ready": True}) + "\n"


class _ScriptedProc(subprocess.Popen[str]):
    """Scripted sidecar handle: canned reply line, optional broken write, kill/terminate flags."""

    def __init__(self, reply: str | None = None, *, fail_writes: int = 0) -> None:
        self.stdin = _PipeStdin(fail_writes=fail_writes)
        self.stdout = _ReplyStdout(self.stdin, reply)
        self.killed = False
        self.terminated = False

    @override
    def poll(self) -> None:
        return None

    @override
    def terminate(self) -> None:
        self.terminated = True

    @override
    def kill(self) -> None:
        self.killed = True


def test_sidecar_restarts_once_after_broken_pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead write pipe is discarded and the call retries on a fresh sidecar process."""
    import app.decision_client as dc

    dead = _ScriptedProc(fail_writes=1)
    fresh = _ScriptedProc()
    client = JevClient(data_root=tmp_path, runtime_path=_runtime_path())
    monkeypatch.setattr(client, "_proc", dead)

    def _spawn(*args: object, **kwargs: object) -> _ScriptedProc:
        return fresh

    monkeypatch.setattr(dc.subprocess, "Popen", _spawn)

    client.start()

    assert dead.killed and not dead.terminated
    assert client._proc is fresh
    assert isinstance(fresh.stdin, _PipeStdin)
    assert [w.get("op") for w in fresh.stdin.writes] == ["ping"]


def test_sidecar_eof_raises_unavailable_and_discards_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar that closes without answering reports unavailable and drops the dead handle."""
    from app.decision_client import _LIVE_PROCS, _SidecarUnavailable

    proc = _ScriptedProc("")
    client = JevClient(data_root=tmp_path, runtime_path=_runtime_path())
    monkeypatch.setattr(client, "_proc", proc)
    _LIVE_PROCS.add(proc)

    try:
        with pytest.raises(_SidecarUnavailable, match="sidecar closed"):
            client.start()
    finally:
        _LIVE_PROCS.discard(proc)

    assert proc.killed
    assert client._proc is None


@pytest.mark.parametrize(
    ("reply", "error"),
    [("not json\n", RuntimeError), ("[1, 2]\n", TypeError), ('"text"\n', TypeError)],
)
def test_sidecar_rejects_unusable_reply_lines(
    reply: str, error: type[Exception], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply line that is not a JSON object fails the call and keeps the live handle."""
    proc = _ScriptedProc(reply)
    client = JevClient(data_root=tmp_path, runtime_path=_runtime_path())
    monkeypatch.setattr(client, "_proc", proc)

    with pytest.raises(error, match="malformed sidecar response"):
        client.start()

    assert not proc.killed and client._proc is proc


def _sidecar_answering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: Mapping[str, object]) -> JevClient:
    """Client whose sidecar answers each request with the canned reply, keeping the request id."""
    client = JevClient(data_root=tmp_path, runtime_path=tmp_path / "no-runtime.ts")

    def _roundtrip(payload: Mapping[str, JSONValue]) -> dict[str, object]:
        return {"id": payload["id"], **dict(response)}

    monkeypatch.setattr(client, "_sidecar_roundtrip", _roundtrip)
    monkeypatch.setattr(client, "_persist", _noop_persist)
    return client


@pytest.mark.parametrize(
    ("response", "error", "match"),
    [
        (
            {"id": "other", "raw": {}, "decisions": {"q": {"kind": "noul", "probability": 0.5}}},
            TypeError,
            "malformed sidecar response",
        ),
        ({"error": "boom", "raw": {}, "decisions": {}}, RuntimeError, "decide: boom"),
        ({"raw": {}}, TypeError, "malformed sidecar response"),
        ({"raw": {}, "decisions": {}}, TypeError, "malformed sidecar response"),
        ({"raw": {}, "decisions": {"q": "not-a-decision"}}, TypeError, "malformed sidecar response"),
        ({"raw": set(), "decisions": {"q": {"kind": "noul"}}}, ValueError, "not a JSON value"),
    ],
)
def test_decide_rejects_unusable_sidecar_replies(
    response: Mapping[str, object],
    error: type[Exception],
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar reply must carry this request's id, no error, and per-question decision objects."""
    client = _sidecar_answering(tmp_path, monkeypatch, response)

    with pytest.raises(error, match=match):
        asyncio.run(client.decide({"s": 1}, {"q": {"type": "noul"}}, decision_type="t", session_id="s"))


@pytest.mark.parametrize(
    ("decision_type", "session_id", "questions", "message"),
    [
        ("", "s", {"q": {"type": "noul"}}, "decision_type required"),
        ("t", "", {"q": {"type": "noul"}}, "session_id required"),
        ("t", "s", {}, "missing_questions"),
    ],
)
def test_decide_rejects_bad_request_before_transport(
    decision_type: str,
    session_id: str,
    questions: Mapping[str, JSONValue],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank type or session and empty questions fail before any transport call."""
    client, seen = _capturing_client(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=message):
        asyncio.run(client.decide({"s": 1}, questions, decision_type=decision_type, session_id=session_id))

    assert seen == []


@pytest.mark.parametrize(
    ("decision_type", "session_id", "questions", "message"),
    [
        (7, "s", {"q": {"type": "noul"}}, "decision_type required"),
        ("t", None, {"q": {"type": "noul"}}, "session_id required"),
        ("t", "s", [], "missing_questions"),
    ],
)
def test_require_decision_request_rejects_non_text_and_non_mapping(
    decision_type: object, session_id: object, questions: object, message: str
) -> None:
    """A non-text type or session and non-mapping questions are rejected by the request guard."""
    from app.decision_client import _require_decision_request

    with pytest.raises(ValueError, match=message):
        _require_decision_request(decision_type, session_id, questions)


def test_decide_rejects_unserializable_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A state json cannot encode fails as unserializable before any transport call."""
    client, seen = _capturing_client(tmp_path, monkeypatch)

    with pytest.raises(TypeError, match="unserializable request"):
        asyncio.run(client.decide({"bad": object()}, {"q": {"type": "noul"}}, decision_type="t", session_id="s"))

    assert seen == []


def _choice_transport(
    picks: Mapping[str, str],
    probabilities: Mapping[str, Mapping[str, float]] | None = None,
) -> Callable[[object, Mapping[str, JSONValue]], dict[str, JSONValue]]:
    """Stub answer envelope: pick the named label per question, with optional explicit probabilities."""

    def _run(_state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        answers: dict[str, JSONValue] = {}
        for qid, question in questions.items():
            criteria = question.get("criteria") if isinstance(question, dict) else None
            labels: list[str] = sorted(criteria) if isinstance(criteria, dict) else []
            chosen = picks[qid]
            assert chosen in labels, (qid, chosen)
            given: Mapping[str, float] | None = probabilities.get(qid) if probabilities else None
            rest = 0.1 / max(len(labels) - 1, 1)
            probs: dict[str, JSONValue] = {}
            for label in labels:
                probs[label] = given[label] if given is not None else (0.9 if label == chosen else rest)
            answers[qid] = {"type": "choice", "choice": chosen, "probabilities": probs, "confidence": 0.9}
        return {"answers": answers}

    return _run


def _stub_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: Callable[..., object]) -> JevClient:
    client = JevClient(transport=transport, data_root=tmp_path)
    monkeypatch.setattr(client, "_persist", _noop_persist)
    return client


@pytest.mark.parametrize(
    ("pick", "action"),
    [("alpha", "invoke"), (REASON_SENTINEL, "reason"), (RESOLVED_SENTINEL, "resolved")],
)
def test_adjudicate_maps_sidecar_choice_to_tool_decision(
    pick: str, action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adjudication over the whole registry returns invoke, reason, or resolved for the winner."""
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    registry: list[dict[str, JSONValue]] = [{"name": "alpha"}]
    client = _stub_client(tmp_path, monkeypatch, _choice_transport({"tool_selection": pick}))

    out = asyncio.run(client.adjudicate({"analyses": []}, node, session_id="s", registry=registry))

    assert out.action == action
    assert out.tool_name == ("alpha" if action == "invoke" else None)
    assert out.tool_names == (("alpha",) if action == "invoke" else ())
    assert out.confidence == 0.9


def test_adjudicate_fans_out_close_runner_ups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner-ups above the parallel floor and inside the window join the fan-out, capped at three."""
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    registry: list[dict[str, JSONValue]] = [
        {"name": "alpha"},
        {"name": "beta"},
        {"name": "gamma"},
        {"name": "delta"},
    ]
    scores = {
        "tool_selection": {
            "alpha": 0.60,
            "beta": 0.55,
            "gamma": 0.50,
            "delta": 0.20,
            REASON_SENTINEL: 0.05,
            RESOLVED_SENTINEL: 0.05,
        }
    }
    client = _stub_client(tmp_path, monkeypatch, _choice_transport({"tool_selection": "alpha"}, scores))

    out = asyncio.run(client.adjudicate({"analyses": []}, node, session_id="s", registry=registry))

    assert out.action == "invoke" and out.tool_name == "alpha"
    assert out.tool_names[0] == "alpha"
    assert out.tool_names == ("alpha", "beta", "gamma")


def test_select_tool_rejects_sidecar_decision_defects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar decision that is not a choice, not in the registry, or not numeric fails the selection."""
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    registry: list[dict[str, JSONValue]] = [{"name": "alpha"}]
    cases: list[tuple[JSONValue, type[Exception], str]] = [
        ({"kind": "noul", "probability": 0.5}, TypeError, "malformed tool_selection decision"),
        ({"kind": "choice", "choice": "alpha"}, TypeError, "malformed tool_selection decision"),
        ({"kind": "choice", "choice": "outside", "probabilities": {"outside": 0.9}}, ValueError, "not in registry"),
        (
            {"kind": "choice", "choice": "alpha", "probabilities": {"alpha": "x"}, "confidence": 0.9},
            TypeError,
            "malformed tool_selection probabilities",
        ),
    ]

    for decision, error, match in cases:
        client = _sidecar_answering(tmp_path, monkeypatch, {"raw": {}, "decisions": {"tool_selection": decision}})
        with pytest.raises(error, match=match):
            asyncio.run(client.select_tool("obj", node, registry, session_id="s"))


@pytest.mark.parametrize(
    ("decision", "match"),
    [
        # A non-mapping decision fails the sidecar response typing before the kind check,
        # so that defect names the response, not the selection (app/decision_client.py:1173).
        (None, "malformed sidecar response"),
        ({}, "malformed tool_selection decision"),
        ({"choice": []}, "malformed tool_selection decision"),
        ({"kind": "choice", "choice": []}, "malformed tool_selection decision"),
    ],
)
def test_select_tool_rejects_missing_sidecar_decision(
    decision: JSONValue, match: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or shapeless tool_selection decision never becomes a ToolDecision."""
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    client = _sidecar_answering(tmp_path, monkeypatch, {"raw": {}, "decisions": {"tool_selection": decision}})

    with pytest.raises(TypeError, match=match):
        asyncio.run(client.select_tool("obj", node, [{"name": "alpha"}], session_id="s"))


# --- decision answer parsing boundaries ---------------------------------------------------------

_CHOICE_QUESTIONS: dict[str, JSONValue] = {"pick": {"type": "choice", "criteria": {"alpha": "A", "beta": "B"}}}
_SCORE_QUESTIONS: dict[str, JSONValue] = {"s": {"type": "score", "criteria": ["low", "mid", "high"]}}
_BOUNDARY_FLOATS = st.one_of(
    st.just(0.0),
    st.just(1.0),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    # NaN and the infinities carry no range, so they come from their own branch: the bounded
    # branches above cannot mix them with min_value/max_value.
    st.sampled_from((math.nan, math.inf, -math.inf)),
)


def test_parse_decisions_requires_an_answers_envelope() -> None:
    """The response envelope must be an object whose answers field is an object."""
    from app.decision_client import parse_decisions

    questions: dict[str, JSONValue] = {"q": {"type": "noul"}}
    raws: tuple[object, ...] = (None, [], "{}", {}, {"answers": None}, {"answers": []}, {"answers": "x"})
    for raw in raws:
        with pytest.raises(TypeError, match="malformed_decision_response"):
            parse_decisions(questions, raw)


def test_parse_decisions_requires_matching_answer_ids() -> None:
    """Answers must cover exactly the supplied question ids."""
    from app.decision_client import parse_decisions

    questions: dict[str, JSONValue] = {"q": {"type": "noul"}, "other": {"type": "noul"}}
    answer: JSONValue = {"type": "noul", "noul": 0.5}
    with pytest.raises(ValueError, match="decision answers do not match questions"):
        parse_decisions(questions, {"answers": {"q": answer}})
    with pytest.raises(ValueError, match="decision answers do not match questions"):
        parse_decisions(questions, {"answers": {"q": answer, "other": answer, "extra": answer}})


@pytest.mark.parametrize(
    "answer",
    [
        None,
        [],
        "noul",
        {},
        {"type": "choice"},
        {"type": "noul"},
        {"type": "noul", "noul": "0.5"},
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": -0.01},
        {"type": "noul", "noul": 1.01},
        {"type": "noul", "noul": float("nan")},
        {"type": "noul", "noul": float("inf")},
    ],
)
def test_parse_noul_rejects_malformed_answers(answer: JSONValue) -> None:
    """A noul answer is a probability in [0,1]; any other shape is malformed."""
    from app.decision_client import parse_decisions

    with pytest.raises(ValueError, match="malformed_decision_answer for q"):
        parse_decisions({"q": {"type": "noul"}}, {"answers": {"q": answer}})


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"type": "choice"},
        {"type": "choice", "choice": "alpha"},
        {"type": "choice", "choice": "gamma", "probabilities": {"alpha": 0.5, "beta": 0.5}, "confidence": 0.5},
        {"type": "choice", "choice": "alpha", "probabilities": [], "confidence": 0.5},
        {"type": "choice", "choice": "alpha", "probabilities": {"alpha": 0.5}, "confidence": 0.5},
        {
            "type": "choice",
            "choice": "alpha",
            "probabilities": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0},
            "confidence": 0.5,
        },
        {"type": "choice", "choice": "alpha", "probabilities": {"alpha": 0.5, "beta": 1.5}, "confidence": 0.5},
        {"type": "choice", "choice": "alpha", "probabilities": {"alpha": 0.5, "beta": 0.5}, "confidence": True},
        {"type": "choice", "choice": "alpha", "probabilities": {"alpha": 0.5, "beta": 0.5}, "confidence": 1.5},
    ],
)
def test_parse_choice_rejects_malformed_answers(answer: JSONValue) -> None:
    """A choice answer declares a listed label, its exact probability map, and an in-range confidence."""
    from app.decision_client import parse_decisions

    with pytest.raises(ValueError, match="malformed_decision_answer for pick"):
        parse_decisions(_CHOICE_QUESTIONS, {"answers": {"pick": answer}})


def test_parse_choice_orders_probabilities_by_label() -> None:
    """The stored probability map lists the declared labels in label order, whatever order the provider used."""
    from app.decision_client import parse_decisions

    answer: JSONValue = {
        "type": "choice",
        "choice": "beta",
        "probabilities": {"beta": 0.8, "alpha": 0.2},
        "confidence": 0.8,
    }
    out = parse_decisions(_CHOICE_QUESTIONS, {"answers": {"pick": answer}})

    assert out == {
        "pick": {
            "kind": "choice",
            "choice": "beta",
            "probabilities": {"alpha": 0.2, "beta": 0.8},
            "confidence": 0.8,
        }
    }
    probabilities = out["pick"]["probabilities"]
    assert isinstance(probabilities, dict)
    assert list(probabilities) == ["alpha", "beta"]


def test_parse_decisions_prefers_supplied_choice_options() -> None:
    """Supplied choice options replace the question criteria; an empty entry falls back to the criteria."""
    from app.decision_client import parse_decisions

    questions: dict[str, JSONValue] = {"pick": {"type": "choice", "criteria": {"alpha": "A"}}}
    raw: JSONValue = {
        "answers": {"pick": {"type": "choice", "choice": "gamma", "probabilities": {"gamma": 1.0}, "confidence": 1.0}}
    }

    assert parse_decisions(questions, raw, {"pick": {"gamma": "G"}}) == {
        "pick": {
            "kind": "choice",
            "choice": "gamma",
            "probabilities": {"gamma": 1.0},
            "confidence": 1.0,
        }
    }
    with pytest.raises(ValueError, match="malformed_decision_answer for pick"):
        parse_decisions(questions, raw, {"pick": {}})


def test_parse_score_keeps_optional_fields_and_legend() -> None:
    """A score answer keeps its score, distribution, confidence, and legend in that order."""
    from app.decision_client import parse_decisions

    answer: JSONValue = {
        "type": "score",
        "score": 1.5,
        "confidence": 0.99,
        "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
        "legend": {"0": "low"},
    }

    assert parse_decisions(_SCORE_QUESTIONS, {"answers": {"s": answer}}) == {
        "s": {
            "kind": "score",
            "score": 1.5,
            "confidence": 0.99,
            "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
            "raw": {"0": "low"},
        }
    }


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {"type": "score"},
        {"type": "score", "score": True},
        {"type": "score", "score": "1"},
        {"type": "score", "score": float("nan")},
        {"type": "score", "score": float("inf")},
        {"type": "score", "score": 3.0},
        {"type": "score", "score": -0.5},
        {"type": "score", "score": 1.0, "confidence": 1.2},
        {"type": "score", "score": 1.0, "probabilities": []},
        {"type": "score", "score": 1.0, "probabilities": {"0": 2.0}},
    ],
)
def test_parse_score_rejects_malformed_answers(answer: JSONValue) -> None:
    """A score answer is a finite number inside the rubric range, with optional valid extras."""
    from app.decision_client import parse_decisions

    with pytest.raises(ValueError, match="malformed_decision_answer for s"):
        parse_decisions(_SCORE_QUESTIONS, {"answers": {"s": answer}})


def test_parse_score_range_comes_from_criteria_or_levels() -> None:
    """The rubric is the question's criteria or levels array; anything shorter leaves the range open."""
    from app.decision_client import parse_decisions

    for key in ("criteria", "levels"):
        questions: dict[str, JSONValue] = {"s": {"type": "score", key: ["a", "b"]}}
        assert parse_decisions(questions, {"answers": {"s": {"type": "score", "score": 1}}}) == {
            "s": {"kind": "score", "score": 1}
        }
        with pytest.raises(ValueError, match="malformed_decision_answer for s"):
            parse_decisions(questions, {"answers": {"s": {"type": "score", "score": 2}}})
    single: dict[str, JSONValue] = {"s": {"type": "score", "criteria": ["only"]}}
    assert parse_decisions(single, {"answers": {"s": {"type": "score", "score": 7}}}) == {
        "s": {"kind": "score", "score": 7}
    }


def test_parse_decisions_infers_kind_from_the_answer_type() -> None:
    """An undeclared question type falls back to the answer's own type tag."""
    from app.decision_client import parse_decisions

    out = parse_decisions(
        {"s": {"instructions": "score it"}},
        {"answers": {"s": {"type": "score", "score": 2, "confidence": 0.5}}},
    )

    assert out == {"s": {"kind": "score", "score": 2, "confidence": 0.5}}


@settings(max_examples=60, derandomize=True, deadline=None)
@given(_BOUNDARY_FLOATS)
def test_prop_noul_probability_boundary(probability: float) -> None:
    """A noul probability round-trips inside [0,1]; NaN, infinities, and outliers are malformed."""
    from app.decision_client import parse_decisions

    raw = {"answers": {"q": {"type": "noul", "noul": probability}}}
    if 0.0 <= probability <= 1.0:
        assert parse_decisions({"q": {"type": "noul"}}, raw) == {"q": {"kind": "noul", "probability": probability}}
    else:
        with pytest.raises(ValueError, match="malformed_decision_answer for q"):
            parse_decisions({"q": {"type": "noul"}}, raw)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(_BOUNDARY_FLOATS, _BOUNDARY_FLOATS, _BOUNDARY_FLOATS)
def test_prop_choice_probability_boundary(alpha: float, beta: float, confidence: float) -> None:
    """A choice answer needs every declared label and an in-range confidence; in-range values round-trip."""
    from app.decision_client import parse_decisions

    raw = {
        "answers": {
            "pick": {
                "type": "choice",
                "choice": "alpha",
                "probabilities": {"alpha": alpha, "beta": beta},
                "confidence": confidence,
            }
        }
    }
    if 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and 0.0 <= confidence <= 1.0:
        assert parse_decisions(_CHOICE_QUESTIONS, raw) == {
            "pick": {
                "kind": "choice",
                "choice": "alpha",
                "probabilities": {"alpha": alpha, "beta": beta},
                "confidence": confidence,
            }
        }
    else:
        with pytest.raises(ValueError, match="malformed_decision_answer for pick"):
            parse_decisions(_CHOICE_QUESTIONS, raw)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(_BOUNDARY_FLOATS)
def test_prop_score_range_boundary(score: float) -> None:
    """A score round-trips inside the rubric range; non-finite and out-of-range scores are malformed."""
    from app.decision_client import parse_decisions

    raw = {"answers": {"s": {"type": "score", "score": score}}}
    if 0.0 <= score <= 2.0:
        assert parse_decisions(_SCORE_QUESTIONS, raw) == {"s": {"kind": "score", "score": score}}
    else:
        with pytest.raises(ValueError, match="malformed_decision_answer for s"):
            parse_decisions(_SCORE_QUESTIONS, raw)


@settings(max_examples=60, derandomize=True, deadline=None)
@given(st.sampled_from(_FORMATS), _BOUNDARY_FLOATS)
def test_prop_envelope_rejects_non_finite_probability_bodies(api_format: str, probability: float) -> None:
    """NaN and Infinity are not JSON: those bodies are rejected; finite probabilities round-trip."""
    from app.decision_client import _decision_envelope

    answers = {"q": {"type": "noul", "noul": probability}}
    body = _body(api_format, answers)
    if math.isfinite(probability):
        assert _decision_envelope(body, "configured", api_format) == {"answers": answers, "model": "configured"}
    else:
        with pytest.raises(TypeError, match="malformed_decision_response"):
            _decision_envelope(body, "configured", api_format)


def test_decision_envelope_normalizes_usage_keys_per_format() -> None:
    """Each protocol's token counts land in the envelope's own keys; the configured model is the fallback."""
    from app.decision_client import _decision_envelope

    chat = _chat_body({"answers": {}}, model="reported", usage={"prompt_tokens": 3, "completion_tokens": 4, "cost": 1})
    native = _native_body({}, model="reported", usage={"input_tokens": 5, "output_tokens": 6, "cost": 1})

    chat_out = _decision_envelope(chat, "configured", "chat_completions")
    native_out = _decision_envelope(native, "configured", "decisions")

    assert chat_out == {"answers": {}, "model": "reported", "usage": {"input_tokens": 3, "output_tokens": 4}}
    assert native_out == {"answers": {}, "model": "reported", "usage": {"input_tokens": 5, "output_tokens": 6}}
    usage = chat_out["usage"]
    assert isinstance(usage, dict)
    assert list(usage) == ["input_tokens", "output_tokens"]
    assert _decision_envelope(_chat_body({"answers": {}}), "configured", "chat_completions") == {
        "answers": {},
        "model": "configured",
    }
    assert _decision_envelope(_native_body({}), "configured", "decisions") == {"answers": {}, "model": "configured"}


def test_decision_envelope_rejects_chat_defects() -> None:
    """A body without a usable choice, or with a refused, truncated, or empty message, is rejected."""
    from app.decision_client import _decision_envelope

    bodies = [
        b'{"choices": []}',
        b'{"choices": [{}]}',
        b'{"choices": [{"message": {"content": {}}}]}',
        b'{"choices": [{"message": {"refusal": "no"}}]}',
        b'{"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}',
        b'{"choices": [{"message": {"content": " "}}]}',
        _chat_body("not JSON"),
        _chat_body([1, 2]),
    ]
    for body in bodies:
        with pytest.raises(TypeError, match="malformed_decision_response"):
            _decision_envelope(body, "configured", "chat_completions")
    with pytest.raises(TypeError, match="malformed_decision_response"):
        _decision_envelope(b"[]", "configured", "decisions")


@pytest.mark.parametrize("api_format", _FORMATS)
def test_decision_envelope_accepts_utf8_non_ascii_body(api_format: str) -> None:
    """A raw UTF-8 non-ASCII body decodes; the reported model keeps its characters."""
    from app.decision_client import _decision_envelope

    choice: JSONValue = {"message": {"content": '{"answers": {}}'}, "finish_reason": "stop"}
    fields: dict[str, JSONValue] = {"answers": {}} if api_format == "decisions" else {"choices": [choice]}
    body = json.dumps({**fields, "model": "modèle-é"}, ensure_ascii=False).encode()
    assert b"\\u" not in body
    assert _decision_envelope(body, "configured", api_format) == {"answers": {}, "model": "modèle-é"}


# --- registry resolution ------------------------------------------------------------------------


def test_auto_registry_prefers_scheduler_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty scheduler registry is returned as-is; the degraded build never runs."""
    import app.decision_client as dc

    manifest: list[dict[str, JSONValue]] = [{"name": "alpha"}]

    def _unused() -> list[dict[str, JSONValue]]:
        raise AssertionError("fallback ran for a healthy scheduler registry")

    monkeypatch.setattr("app.research.scheduler.build_registry", lambda: manifest)
    monkeypatch.setattr(dc, "_fallback_registry", _unused)

    assert dc._auto_registry() is manifest


def test_auto_registry_falls_back_when_scheduler_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheduler that raises, or returns nothing, yields the degraded canonical registry."""
    import app.decision_client as dc
    from app.research.scheduler import build_registry

    def _boom() -> list[dict[str, JSONValue]]:
        raise AttributeError("build_registry is gone")

    monkeypatch.setattr("app.research.scheduler.build_registry", _boom)
    fallback = dc._auto_registry()

    assert fallback and all(isinstance(entry.get("name"), str) for entry in fallback)
    assert list(fallback[0]) == list(build_registry()[0])

    monkeypatch.setattr("app.research.scheduler.build_registry", list)
    assert [entry["name"] for entry in dc._auto_registry()] == [entry["name"] for entry in fallback]


def test_fallback_registry_renders_discovery_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known tool carries its discovery metadata and declared prerequisites; unknown names stay blank."""
    import app.decision_client as dc
    import app.tools
    from app.research.scheduler import _HANDLE_PARAMS
    from app.tools import TOOL_DISCOVERY_REGISTRY

    known = next(iter(TOOL_DISCOVERY_REGISTRY))
    handle = next(iter(sorted(_HANDLE_PARAMS)))
    unknown = "zz_not_a_canonical_tool"

    def _tools(*args: object, **kwargs: object) -> list[object]:
        return [
            "not-a-tool",
            {},
            {"function": "not-a-mapping"},
            {"function": {"name": ""}},
            {"function": {"name": 7}},
            {"function": {"name": known, "parameters": {"required": [handle, 7]}}},
            {"function": {"name": unknown, "description": "described", "parameters": "not-a-mapping"}},
        ]

    monkeypatch.setattr(app.tools, "tools_for_capabilities", _tools)
    entries = dc._fallback_registry()
    meta = TOOL_DISCOVERY_REGISTRY[known]

    assert [entry["name"] for entry in entries] == [known, unknown]
    assert entries[0]["purpose"] == meta.summary
    assert entries[0]["useWhen"] == "; ".join(meta.choose_when)
    assert entries[0]["avoidWhen"] == "; ".join(meta.reject_when)
    assert entries[0]["conflicts"] == ", ".join(meta.conflicts_with)
    assert entries[0]["nextTools"] == ", ".join(meta.related_tools)
    assert entries[0]["evidence"] == meta.output_kind
    assert entries[0]["prerequisites"] == f"needs {handle}"
    assert entries[0]["parameters"] == {"required": [handle, 7]}
    assert entries[1]["description"] == "described"
    assert entries[1]["domain"] == "unknown"
    assert entries[1]["purpose"] == "" and entries[1]["prerequisites"] == ""
    assert entries[1]["nextTools"] == "" and entries[1]["keyInputs"] == "req()"
    assert entries[1]["parameters"] == {}


def test_fallback_registry_survives_missing_scheduler_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the scheduler handle and blind lists the degraded manifest stays conservative."""
    import app.decision_client as dc

    monkeypatch.delattr("app.research.scheduler._HANDLE_PARAMS")
    monkeypatch.delattr("app.research.scheduler._PIT_BLIND_TOOLS")

    assert dc._scheduler_metadata() == (frozenset(), frozenset())
    entries = dc._fallback_registry()
    assert entries
    assert all(entry["pitSupport"] == "PIT-scoped" for entry in entries)
    assert all(entry["prerequisites"] == "" for entry in entries)


# --- persistence: domain record plus run recording ----------------------------------------------


def _persist_call(
    client: JevClient,
    decisions: dict[str, dict[str, JSONValue]],
    *,
    questions: dict[str, JSONValue] | None = None,
    raw: object = None,
    via: str = "http",
    jev_in: int | None = None,
    jev_out: int | None = None,
    jev_model: str | None = None,
) -> None:
    client._persist(
        decision_type="tool_selection",
        session_id="s1",
        node_id="n1",
        job_id=None,
        questions=questions if questions is not None else {"q": {"type": "noul"}},
        decisions=decisions,
        raw=raw,
        via=via,
        latency_ms=1.5,
        started_at="2026-10-14T00:00:00+00:00",
        completed_at="2026-10-14T00:00:01+00:00",
        jev_in=jev_in,
        jev_out=jev_out,
        jev_model=jev_model,
    )


def test_persist_projects_every_decision_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each answer kind lands in the record's probabilities projection; the run event keeps the round."""
    from types import SimpleNamespace

    from app.decision_client import _decision_record
    from app.research.models import DecisionRecord

    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        seen.update({"record": record, "domain": kwargs})

    def _record_event(*args: object, **kwargs: object) -> None:
        seen.update({"event": kwargs})

    def _record_model_call(**kwargs: object) -> None:
        seen.update({"model": kwargs})

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr(
        "app.decision_client.get_current_recorder",
        lambda: SimpleNamespace(enabled=True, record_event=_record_event, record_model_call=_record_model_call),
    )
    monkeypatch.setenv("DECISION_API_URL", "https://decisions.test/v1/chat")
    decisions: dict[str, dict[str, JSONValue]] = {
        "n": {"kind": "noul", "probability": 0.5},
        "c": {"kind": "choice", "choice": "a", "probabilities": {"a": 1.0}},
        "s": {"kind": "score", "score": 2, "probabilities": {"a": 0.5}, "confidence": 0.7},
        "u": {"kind": "other", "note": "kept"},
    }

    _persist_call(client, decisions, raw={"answers": {}}, jev_in=3, jev_out=4, jev_model="m")

    record = seen["record"]
    assert isinstance(record, DecisionRecord)
    assert record.probabilities == {
        "n": {"probability": 0.5},
        "c": {"a": 1.0},
        "s": {"score": 2, "a": 0.5, "confidence": 0.7},
        "u": {"kind": "other", "note": "kept"},
    }
    assert record.selected == decisions
    assert record.confidence is None
    assert record.candidates == {"q": {"type": "noul"}}
    domain = seen["domain"]
    assert isinstance(domain, dict)
    assert domain["provider"] == "https://decisions.test/v1/chat"
    assert domain["request"] == {"state": None, "questions": {"q": {"type": "noul"}}}
    event = seen["event"]
    assert isinstance(event, dict)
    assert event["model"] == "m" and event["success"] is True and event["duration_ms"] == 1.5
    metadata = event["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["usage"] == {"input_tokens": 3, "output_tokens": 4}
    assert metadata["session_id"] == "s1" and metadata["node_id"] == "n1"
    assert event["result_summary"] == json.dumps(decisions, sort_keys=True)
    model_call = seen["model"]
    assert isinstance(model_call, dict)
    assert model_call["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert model_call["provider"] == "https://decisions.test/v1/chat"
    junk: dict[str, object] = {**decisions, "junk": "not-a-decision"}
    with_junk = _decision_record(
        decision_type="tool_selection",
        session_id="s1",
        node_id="n1",
        job_id=None,
        questions={"q": {"type": "noul"}},
        decisions=junk,
    )
    assert with_junk.probabilities == record.probabilities
    assert with_junk.selected == junk


def test_persist_sole_decision_carries_its_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-decision round keeps its confidence when it is a probability."""
    from app.decision_client import _decision_record
    from app.research.models import DecisionRecord

    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        seen.update({"record": record})

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr("app.decision_client.get_current_recorder", lambda: None)

    _persist_call(client, {"q": {"kind": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 0.4}})
    record = seen["record"]
    assert isinstance(record, DecisionRecord)
    assert record.confidence == 0.4

    _persist_call(client, {"q": {"kind": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 1.4}})
    record = seen["record"]
    assert isinstance(record, DecisionRecord)
    assert record.confidence is None

    non_decision = _decision_record(
        decision_type="tool_selection",
        session_id="s1",
        node_id="n1",
        job_id=None,
        questions={"q": {"type": "noul"}},
        decisions={"q": "not-a-decision"},
    )
    assert non_decision.confidence is None


@pytest.mark.parametrize(
    ("jev_in", "jev_out", "expected"),
    [
        (3, 4, {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
        (3, None, {"prompt_tokens": 3}),
        (None, 4, {"completion_tokens": 4}),
        (None, None, None),
    ],
)
def test_persist_runs_records_usage_variants(
    jev_in: int | None,
    jev_out: int | None,
    expected: dict[str, object] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the counts the provider reported reach the recorded model call."""
    from types import SimpleNamespace

    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        return None

    def _record_event(*args: object, **kwargs: object) -> None:
        seen.update({"event": kwargs})

    def _record_model_call(**kwargs: object) -> None:
        seen.update({"model": kwargs})

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr(
        "app.decision_client.get_current_recorder",
        lambda: SimpleNamespace(enabled=True, record_event=_record_event, record_model_call=_record_model_call),
    )
    monkeypatch.setenv("DECISION_API_URL", "https://decisions.test/v1/chat")

    _persist_call(client, {"q": {"kind": "noul", "probability": 0.5}}, jev_in=jev_in, jev_out=jev_out, jev_model="m")

    model_call = seen.get("model")
    assert isinstance(model_call, dict)
    assert model_call["usage"] == expected


@pytest.mark.parametrize("recorder", [None, SimpleNamespace(enabled=False), SimpleNamespace(enabled=None)])
def test_persist_runs_skips_without_an_enabled_recorder(
    recorder: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No recorder, a disabled recorder, and a stub round record no events, but still store the record."""
    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        seen.update({"domain": 1})

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr("app.decision_client.get_current_recorder", lambda: recorder)

    _persist_call(client, {"q": {"kind": "noul", "probability": 0.5}}, via="stub")

    assert seen == {"domain": 1}


def test_persist_skips_unserializable_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision that is not a JSON value is skipped: no record, no run event, no raise."""
    from app.decision_client import _decision_record

    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        seen.update({"domain": 1})

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr("app.decision_client.get_current_recorder", lambda: seen.update({"recorder": 1}))

    _persist_call(client, {"q": {"kind": "other", "bad": math.nan}})

    assert seen == {}
    with pytest.raises(ValueError, match="not a JSON value"):
        _decision_record(
            decision_type="tool_selection",
            session_id="s1",
            node_id="n1",
            job_id=None,
            questions={"q": {"type": "noul"}},
            decisions={"q": {"kind": "other", "bad": object()}},
        )


def test_persist_swallows_domain_and_runs_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing domain store or recorder never breaks the decision call."""
    from types import SimpleNamespace

    def _domain_boom(record: object, **kwargs: object) -> None:
        raise OSError("disk gone")

    def _event_boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("recorder gone")

    def _model_boom(**kwargs: object) -> None:
        raise ValueError("usage rejected")

    client = JevClient(data_root=tmp_path)
    monkeypatch.setattr(client, "_persist_domain", _domain_boom)
    monkeypatch.setattr(
        "app.decision_client.get_current_recorder",
        lambda: SimpleNamespace(enabled=True, record_event=_event_boom, record_model_call=_model_boom),
    )
    monkeypatch.setenv("DECISION_API_URL", "https://decisions.test/v1/chat")

    _persist_call(client, {"q": {"kind": "noul", "probability": 0.5}}, jev_in=1, jev_out=2, jev_model="m")


def test_persist_swallows_a_failing_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recorder that rejects the model call still keeps the decision round."""
    from types import SimpleNamespace

    seen: dict[str, object] = {}
    client = JevClient(data_root=tmp_path)

    def _record_domain(record: object, **kwargs: object) -> None:
        seen.update({"domain": 1})

    def _record_event(*args: object, **kwargs: object) -> None:
        seen.update({"event": 1})

    def _record_model_call(**kwargs: object) -> None:
        raise ValueError("usage rejected")

    monkeypatch.setattr(client, "_persist_domain", _record_domain)
    monkeypatch.setattr(
        "app.decision_client.get_current_recorder",
        lambda: SimpleNamespace(enabled=True, record_event=_record_event, record_model_call=_record_model_call),
    )
    monkeypatch.setenv("DECISION_API_URL", "https://decisions.test/v1/chat")

    _persist_call(client, {"q": {"kind": "noul", "probability": 0.5}}, jev_in=1, jev_out=2, jev_model="m")

    assert seen == {"domain": 1, "event": 1}


# --- post-tool verdict, prompt phases, and logging projections ----------------------------------

_ENTRY_REGISTRY: list[dict[str, JSONValue]] = [{"name": "get_current_time"}]


def test_assess_entry_tool_rejects_defective_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank prompt, a missing tool name, or a defective registry fails before any decision call."""
    from app.decision_client import _assess_text

    client, seen = _capturing_client(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="blank prompt"):
        asyncio.run(client.assess_entry_tool("   ", "get_current_time"))
    with pytest.raises(ValueError, match="tool required"):
        asyncio.run(client.assess_entry_tool("what time is it?", ""))
    assert seen == []
    with pytest.raises(ValueError, match="blank prompt"):
        _assess_text(7, "get_current_time")
    with pytest.raises(ValueError, match="tool required"):
        _assess_text("what time is it?", 7)

    monkeypatch.setattr("app.decision_client._auto_registry", list)
    with pytest.raises(ValueError, match="empty registry"):
        asyncio.run(client.assess_entry_tool("what time is it?", "get_current_time"))

    cases: list[tuple[list[dict[str, JSONValue]], str]] = [
        ([{}], "needs a name"),
        ([{"name": ""}], "needs a name"),
        ([{"name": REASON_SENTINEL}], "duplicate tool"),
    ]
    for registry, error in cases:
        monkeypatch.setattr("app.decision_client._auto_registry", lambda registry=registry: registry)
        with pytest.raises(ValueError, match=error):
            asyncio.run(client.assess_entry_tool("what time is it?", "get_current_time"))

    assert seen == []


def test_assess_entry_tool_returns_the_winner_and_shapes_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-tool verdict returns a declared winner and shows the clipped outcome in the question."""
    monkeypatch.setattr("app.decision_client._auto_registry", lambda: _ENTRY_REGISTRY)
    client, seen = _capturing_client(tmp_path, monkeypatch)

    verdict = asyncio.run(
        client.assess_entry_tool(
            "what time is it?",
            "get_current_time",
            {"tz": "UTC"},
            {"ok": True, "content": "x" * 4000, "error": "e" * 900, "category": "time"},
        )
    )

    assert verdict == "get_current_time"
    state, questions = seen[0]
    assert isinstance(state, dict)
    assert state["tool"] == "get_current_time" and state["arguments"] == {"tz": "UTC"}
    outcome = state["outcome"]
    assert isinstance(outcome, str)
    assert outcome.startswith("tool get_current_time ok=True category='time'")
    assert f"content={'x' * 1500!r}" in outcome and f"error={'e' * 500!r}" in outcome
    assess = questions["assess"]
    assert isinstance(assess, dict)
    criteria = assess["criteria"]
    assert isinstance(criteria, dict)
    assert set(criteria) == {
        "get_current_time",
        REASON_SENTINEL,
        RESOLVED_SENTINEL,
        "research_required",
    }
    instructions = assess["instructions"]
    assert isinstance(instructions, str)
    assert "the one registry tool to run next." in instructions
    assert "decode relative dates before choosing" in instructions


def test_assess_entry_tool_rejects_a_winner_outside_the_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar verdict outside the declared options is a defect, never a guessed tool."""
    monkeypatch.setattr("app.decision_client._auto_registry", lambda: _ENTRY_REGISTRY)
    decisions: dict[str, JSONValue] = {"assess": {"kind": "choice", "choice": "nope", "probabilities": {"nope": 1.0}}}
    client = _sidecar_answering(tmp_path, monkeypatch, {"raw": {}, "decisions": decisions})

    with pytest.raises(ValueError, match="winner 'nope' not in options"):
        asyncio.run(client.assess_entry_tool("what time is it?", "get_current_time"))


def test_node_dict_reads_dataclasses_to_dict_and_attributes() -> None:
    """Node projections come from a mapping, a to_dict method, dataclass fields, or plain attributes."""
    from dataclasses import dataclass

    from app.decision_client import _node_dict

    @dataclass
    class _Node:
        node_id: str
        question: str

    class _ToDict:
        def to_dict(self) -> object:
            return {"id": "n2", "question": "why?"}

    class _NotAMapping:
        def to_dict(self) -> object:
            return "nope"

    class _Attrs:
        node_id = "n3"
        question = "how?"
        depends_on = ("a", "b")

    assert _node_dict({"node_id": "n1", "question": "q?", "unlisted": 1}) == {"node_id": "n1", "question": "q?"}
    assert _node_dict(_Node(node_id="n4", question="q4?")) == {"node_id": "n4", "question": "q4?"}
    assert _node_dict(_ToDict()) == {"id": "n2", "question": "why?"}
    assert _node_dict(_NotAMapping()) == {}
    assert _node_dict(_Attrs()) == {"node_id": "n3", "question": "how?", "depends_on": ["a", "b"]}
    assert _node_dict(object()) == {}


class _HostileDict(dict[str, JSONValue]):
    """Mapping whose lookups always fail, standing in for a corrupt decision payload."""

    @override
    def get(self, *args: object, **kwargs: object) -> JSONValue:
        raise TypeError("boom")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, (None, None, None)),
        ({}, (None, None, None)),
        ({"model": ""}, (None, None, None)),
        ({"model": "m"}, (None, None, "m")),
        ({"model": "m", "usage": {"input_tokens": 3, "output_tokens": 4}}, (3, 4, "m")),
        ({"model": "m", "usage": {"input_tokens": True, "output_tokens": "4"}}, (None, None, "m")),
        ({"model": 7, "usage": {"input_tokens": 0}}, (0, None, None)),
    ],
)
def test_jev_usage_reads_only_wire_token_counts(
    raw: JSONValue, expected: tuple[int | None, int | None, str | None]
) -> None:
    """Usage extraction keeps integer token counts and a non-empty model name, and nothing else."""
    from app.decision_client import _jev_usage

    assert _jev_usage(raw) == expected


def test_logging_projections_never_raise_on_hostile_mappings() -> None:
    """Usage and choice summaries are logging-only: a hostile mapping yields the fallback text."""
    from app.decision_client import _choice_summary, _jev_usage

    assert _jev_usage(_HostileDict()) == (None, None, None)
    assert _choice_summary(_HostileDict()) == "winner=? conf=? top3=[] margin=?"


def test_choice_summary_shapes_winner_top3_and_margin() -> None:
    """The logging one-liner reports the winner, its confidence, the top three, and the margin."""
    from app.decision_client import _choice_summary

    assert _choice_summary(None) == "winner=- conf=- top3=[] margin=0.00"
    assert _choice_summary(
        {"choice": "a", "confidence": 0.5, "probabilities": {"a": 0.9, "b": 0.6, "c": 0.2, "d": 0.1}}
    ) == ("winner=a conf=0.50 top3=[a:0.90,b:0.60,c:0.20] margin=0.30")
    assert _choice_summary({"choice": "a", "confidence": 1.5, "probabilities": {"a": 2.0, "b": "x"}}) == (
        "winner=a conf=1.50 top3=[] margin=0.00"
    )
    assert _choice_summary({"choice": "", "confidence": True, "probabilities": [1]}) == (
        "winner=- conf=- top3=[] margin=0.00"
    )


def test_observation_lines_caps_at_five_recent_records() -> None:
    """Only the last ten records are inspected and at most five observation lines come back."""
    from app.decision_client import _observation_lines

    assert _observation_lines(None) == []
    assert _observation_lines({"a": 1}) == []
    many: list[JSONValue] = [{"tool": f"t{i}", "outcome_summary": "s"} for i in range(12)]

    assert _observation_lines(many) == [f"t{i}: s" for i in range(2, 7)]


def test_tool_options_prompt_rejects_defective_input() -> None:
    """An empty registry, a nameless node, and a duplicate or colliding entry are defects."""
    from app.decision_client import _registry_options, _tool_options_prompt

    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    registry: list[dict[str, JSONValue]] = [{"name": "a"}]

    with pytest.raises(ValueError, match="empty registry"):
        _tool_options_prompt([], node, [], [])
    bad_nodes: tuple[Mapping[str, JSONValue], ...] = ({}, {"node_id": "n1"}, {"question": "q?"}, {"id": ""})
    for bad_node in bad_nodes:
        with pytest.raises(ValueError, match="needs nodeId and question"):
            _tool_options_prompt(registry, bad_node, [], [])
    with pytest.raises(ValueError, match="needs a name"):
        _tool_options_prompt([{}], node, [], [])
    with pytest.raises(ValueError, match="tool_selection: registry entry needs a name"):
        _registry_options(["junk"], {}, "tool_selection")
    with pytest.raises(ValueError, match="duplicate tool a"):
        _tool_options_prompt([{"name": "a"}, {"name": "a"}], node, [], [])
    with pytest.raises(ValueError, match="collides with sentinel"):
        _tool_options_prompt([{"name": REASON_SENTINEL}], node, [], [])


def test_tool_options_prompt_renders_evidence_observations_and_why() -> None:
    """The question text carries the node question, why-it-matters, evidence ids, and observations."""
    from app.decision_client import _SENTINEL_DESCRIPTIONS, _tool_options_prompt

    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?", "why_it_matters": " because "}
    evidence: list[JSONValue] = [
        {"evidence_id": "ev-1", "claim_text": "c"},
        {"tool": "t", "tool_result_ref": "r", "outcome_summary": "  one   two  "},
        {"tool": "u", "outcome_summary": "x", "error": "  boom  "},
        {"evidence_id": "ev-2"},
        {"id": "ev-3"},
        "junk",
        {"outcome_summary": "   "},
    ]

    options, prompt = _tool_options_prompt(
        [{"name": "a"}], node, evidence, [{"tool": "b"}, "junk", {"tool": "c", "error": ""}]
    )

    assert options == {
        "a": "a",
        REASON_SENTINEL: _SENTINEL_DESCRIPTIONS[REASON_SENTINEL],
        RESOLVED_SENTINEL: _SENTINEL_DESCRIPTIONS[RESOLVED_SENTINEL],
    }
    assert prompt.startswith("Which single tool runs next for research node n1? Question: [Today UTC ")
    assert " Why it matters:  because " in prompt
    assert " Evidence on hand: 7 item(s) [ev-1,r,ev-2,ev-3]." in prompt
    assert " Observed t ref=r: one two." in prompt
    assert " Observed u FAILED: boom." in prompt
    assert "Prior attempt" not in prompt

    _, number_prompt = _tool_options_prompt([{"name": "a"}], {"node_id": "n1", "question": 7}, [], [])
    assert "Question: 7" in number_prompt and "Latest revenue" not in number_prompt


def test_assess_final_response_states_declared_claim_types(tmp_path: Path) -> None:
    """The whole-response question states the declared claim count and its sorted, unique types."""
    client = _final_client(tmp_path, 0.9, "sufficient_support")

    asyncio.run(
        client.assess_final_response(
            "WSJ reported X; if Y then Z.",
            [],
            session_id="s1",
            claims=[
                {"claim_type": "conditional_risk"},
                {"claim_type": "attributed_report"},
                {"claim_type": "attributed_report"},
                "junk",
            ],
        )
    )

    support = client.decide_types[0][1]["support"]
    assert isinstance(support, dict)
    instructions = support["instructions"]
    assert isinstance(instructions, str)
    assert " Declared claims: 4 (types: attributed_report,conditional_risk)." in instructions
    assert "Cited evidence: 0 item(s)." in instructions


def test_assess_final_response_requires_text_and_bounds_confidence(tmp_path: Path) -> None:
    """A blank or non-text response is rejected; a confidence outside [0,1] is not carried."""
    from app.decision_client import _checked_excerpt

    client = _final_client(tmp_path, 0.9, "insufficient")

    with pytest.raises(ValueError, match="assess_final_response: text required"):
        asyncio.run(client.assess_final_response("  ", [], session_id="s1"))
    for text in (7, None):
        with pytest.raises(ValueError, match="assess_final_response: text required"):
            _checked_excerpt(text, "assess_final_response")

    out = asyncio.run(client.assess_final_response("Growth maybe.", None, session_id="s1", confidence=1.5))

    assert out["confidence"] is None
    support = client.decide_types[0][1]["support"]
    assert isinstance(support, dict)
    instructions = support["instructions"]
    assert isinstance(instructions, str)
    assert " Declared claims: 0." in instructions and "(types:" not in instructions
