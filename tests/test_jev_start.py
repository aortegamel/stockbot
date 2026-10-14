"""JevClient.start() startup behavior: ping handshake + single sidecar reuse."""

from __future__ import annotations

import asyncio
import copy
import io
import json
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from pathlib import Path
from types import TracebackType
from typing import Self, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
    for question, wanted, absent in (
        (
            "What did NVDA's latest 10-Q disclose about revenue?",
            "get_sec_document section reads (Item 1, Item 2, Item 1A)",
            "with a revenue query",
        ),
        (
            "What drove NVDA revenue last quarter?",
            "get_sec_document with a revenue query",
            "section reads",
        ),
        (
            "What is ORCL's latest revenue?",
            "Latest revenue means the latest reported financial period",
            "with invented start",
        ),
    ):
        _, revenue_prompt = _tool_options_prompt(reg, {"node_id": "n1", "question": question}, [], [])
        assert wanted in revenue_prompt and absent not in revenue_prompt
    assert "open the 10-Q accession" not in prompt


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
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
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
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes, api_format: str = "chat_completions"
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
        return io.BytesIO(body)

    def _persist(**kwargs: object) -> None:
        seen["persist"] = kwargs

    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(client, "_persist", _persist)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return client, seen


def _http_run(
    client: JevClient, questions: Mapping[str, JSONValue] = _HTTP_QUESTIONS
) -> dict[str, dict[str, JSONValue]]:
    return asyncio.run(client.decide({"source": "test evidence"}, questions, decision_type="t", session_id="s"))


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

    monkeypatch.setattr(urllib.request, "urlopen", _error)
    with pytest.raises(RuntimeError, match=r"decision_request_failed: HTTP 401: invalid \[redacted\]") as exc:
        _http_run(client)
    assert _DECISION_KEY not in str(exc.value)
    assert "persist" not in seen


def test_http_decision_redacts_key_from_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    client, _ = _http_client(tmp_path, monkeypatch, b"")

    def _error(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
        raise OSError(f"network error {_DECISION_KEY}")

    monkeypatch.setattr(urllib.request, "urlopen", _error)
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
def test_select_tool_rejects_mapping_before_request(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dict for evidence/attempts is a caller defect: TypeError, no JEV request."""
    client, seen = _capturing_client(tmp_path, monkeypatch)
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    # cast: deliberately violates the Sequence contract to prove runtime rejection.
    bad = cast(Sequence[JSONValue], {"evidence_id": "ev-1"})
    evidence = bad if field == "evidence" else None
    attempts = bad if field == "attempts" else None
    with pytest.raises(TypeError, match=f"jev {field} must be a sequence of records, not a mapping"):
        asyncio.run(client.select_tool("obj", node, [{"name": "a"}], evidence, attempts, session_id="s"))
    assert seen == []


def test_assess_result_rejects_mapping_evidence_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, seen = _capturing_client(tmp_path, monkeypatch)
    node: dict[str, JSONValue] = {"node_id": "n1", "question": "q?"}
    bad = cast(Sequence[JSONValue], {"evidence_id": "ev-1"})  # contract violation on purpose
    with pytest.raises(TypeError, match="jev evidence must be a sequence of records, not a mapping"):
        asyncio.run(client.assess_result(node, {"content": "x"}, bad, session_id="s"))
    assert seen == []


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
