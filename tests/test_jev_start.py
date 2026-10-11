"""JevClient.start() startup behavior: ping handshake + single sidecar reuse."""

from __future__ import annotations

import asyncio
import json
import select
import shutil
import subprocess
import threading
import time
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
    for question, wanted, absent in (
        (
            "What did NVDA's latest 10-Q disclose about revenue?",
            "get_sec_document section reads (Item 1, Item 2, Item 1A)",
            "with a revenue query",
        ),
        ("What drove NVDA revenue last quarter?", "get_sec_document with a revenue query", "section reads"),
    ):
        _, revenue_prompt = _tool_options_prompt(reg, {"node_id": "n1", "question": question}, [], [])
        assert wanted in revenue_prompt and absent not in revenue_prompt
    assert "open the 10-Q accession" not in prompt


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


def test_outcome_dict_truncates_64k_content_keeps_error() -> None:
    from app.decision_client import _outcome_dict

    big = "x" * 65536
    err = "e" * 5000
    out = _outcome_dict({"tool": "t", "content": big, "error": err, "error_type": "boom"})
    assert isinstance(out["content"], str) and len(out["content"]) <= 2000
    assert out["error"] == err
    assert out["error_type"] == "boom"
    assert "candidate" not in out


def test_outcome_dict_passes_candidate_unclipped_content_stays_capped() -> None:
    from app.decision_client import _outcome_dict

    big = "x" * 65536
    window = "y" * 32000
    cand: dict[str, object] = {"source_handle": {"accession_no": "a"}, "content": window, "matching_passage": window}
    out = _outcome_dict({"tool": "get_sec_document", "content": big}, cand)
    assert isinstance(out["content"], str) and len(out["content"]) == 2000
    assert out["candidate"] == cand
    got = out["candidate"]
    assert isinstance(got, dict) and got["content"] == window


def test_outcome_dict_drops_unserializable_candidate_content_stays_capped() -> None:
    from app.decision_client import _outcome_dict

    big = "x" * 65536
    out = _outcome_dict({"tool": "t", "content": big}, object())
    assert isinstance(out["content"], str) and len(out["content"]) == 2000
    assert "candidate" not in out


class _HungStdout:
    """readline blocks until terminate releases it, then reports EOF."""

    def __init__(self) -> None:
        self._released = threading.Event()

    def readline(self) -> str:
        self._released.wait(timeout=30)
        return ""


class _HungProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _HungStdout()
        self.terminated = threading.Event()

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated.set()
        self.stdout._released.set()

    def kill(self) -> None:
        self.terminate()


def test_hung_sidecar_timeout_keeps_loop_responsive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=0.2)
    hung = _HungProc()
    client._proc = hung  # type: ignore[assignment]

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

    async def _stub(state: object, questions: object) -> object:
        return {"answers": {"q": {"type": "noul", "noul": 0.7}}}

    client._transport = _stub  # type: ignore[assignment]
    out = asyncio.run(
        client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")
    )
    assert out == {"q": {"kind": "noul", "probability": 0.7}}


def test_outer_cancel_closes_hung_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=60)
    hung = _HungProc()
    client._proc = hung  # type: ignore[assignment]

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


class _GateStdout:
    """readline blocks until the gate opens, then answers the latest write."""

    def __init__(self, stdin: _FakeStdin, gate: threading.Event) -> None:
        self._stdin = stdin
        self._gate = gate

    def readline(self) -> str:
        assert self._gate.wait(timeout=10), "sidecar gate never opened"
        payload = json.loads(self._stdin.last)
        return (
            json.dumps({"id": payload["id"], "raw": {}, "decisions": {"q": {"kind": "noul", "probability": 0.5}}})
            + "\n"
        )


class _GateProc:
    def __init__(self, gate: threading.Event) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _GateStdout(self.stdin, gate)
        self.terminated = threading.Event()

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated.set()

    def kill(self) -> None:
        self.terminate()


def test_outer_cancel_waiter_leaves_holder_sidecar_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parent.parent
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", timeout_s=30)
    gate = threading.Event()
    proc = _GateProc(gate)
    client._proc = proc  # type: ignore[assignment]

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _scenario() -> None:
        kw: dict[str, str] = {"decision_type": "t", "session_id": "s"}
        questions = {"q": {"type": "noul", "instructions": "x"}}
        holder = asyncio.ensure_future(client.decide({"s": 1}, questions, **kw))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)
        waiter = asyncio.ensure_future(client.decide({"s": 1}, questions, **kw))  # type: ignore[arg-type]
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

    def __enter__(self) -> _BlockingHttpResp:  # noqa: PYI034 - test fake returns self
        return self

    def __exit__(self, *exc: object) -> None:
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

    def _missing(payload: object) -> object:
        raise _SidecarUnavailable("no sidecar")

    monkeypatch.setattr(client, "_persist", _noop)
    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: resp)  # type: ignore[arg-type]
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    async def _scenario() -> None:
        call = asyncio.ensure_future(
            client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")  # type: ignore[arg-type]
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
    client._proc = proc  # type: ignore[assignment]

    def _noop(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(client, "_persist", _noop)

    async def _scenario() -> None:
        kw: dict[str, str] = {"decision_type": "t", "session_id": "s"}
        questions = {"q": {"type": "noul", "instructions": "x"}}
        holder = asyncio.ensure_future(client.decide({"s": 1}, questions, **kw))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)
        waiter = asyncio.ensure_future(client.decide({"s": 1}, questions, **kw))  # type: ignore[arg-type]
        await asyncio.sleep(0.3)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        gate.set()
        assert await asyncio.wait_for(holder, timeout=5) == {"q": {"kind": "noul", "probability": 0.5}}
        await asyncio.sleep(0.5)
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

    def _missing(payload: object) -> object:
        raise _SidecarUnavailable("no sidecar")

    async def _run_and_close() -> object:
        loop = asyncio.get_running_loop()
        orig = loop.run_in_executor

        def _spy(executor: object, func: object, *args: object) -> object:
            seen["executor"] = executor
            return orig(executor, func, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(loop, "run_in_executor", _spy)
        call = asyncio.ensure_future(
            client.decide({"s": 1}, {"q": {"type": "noul", "instructions": "x"}}, decision_type="t", session_id="s")  # type: ignore[arg-type]
        )
        assert await asyncio.to_thread(entered.wait, 3)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return seen.get("executor")

    monkeypatch.setattr(client, "_persist", _noop)
    monkeypatch.setattr(client, "_sidecar_roundtrip", _missing)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: resp)  # type: ignore[arg-type]
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert asyncio.run(_run_and_close()) is client._http_pool


def _final_stub(prob: float, state: str) -> object:
    async def _run(state_arg: object, questions: dict[str, object]) -> object:
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

    return _run  # type: ignore[return-value]


def _final_client(tmp_path: Path, prob: float, state: str) -> JevClient:
    from collections.abc import Mapping

    from app.research.models import JSONValue

    root = Path(__file__).resolve().parent.parent
    stub = _final_stub(prob, state)
    client = JevClient(data_root=tmp_path, runtime_path=root / "decision" / "runtime.ts", transport=stub)  # type: ignore[arg-type]
    seen: list[tuple[str, dict[str, object]]] = []
    orig = client.decide

    async def _spy(
        state_arg: object, questions: Mapping[str, JSONValue], **kw: object
    ) -> dict[str, dict[str, JSONValue]]:
        seen.append((str(kw.get("decision_type")), dict(questions)))
        out = await orig(state_arg, questions, **kw)  # type: ignore[arg-type]
        client.decide_types = seen  # type: ignore[attr-defined]
        return out

    client.decide = _spy  # type: ignore[method-assign]
    return client


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
    assert client.decide_types[0][0] == "final_response_assessment"  # type: ignore[attr-defined]
    instructions = client.decide_types[0][1]["support"]["instructions"]  # type: ignore[attr-defined]
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
    assert client.decide_types[0][0] == "final_claim_assessment"  # type: ignore[attr-defined]


def test_assess_final_claim_conditional_and_missing(tmp_path: Path) -> None:
    client = _final_client(tmp_path, 0.2, "insufficient")
    out = asyncio.run(
        client.assess_final_claim("If tariffs rise, margins may fall.", "conditional_risk", [], session_id="s1")
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

    class _Boom:
        async def assess_final_response(self, *a: object, **k: object) -> object:
            raise RuntimeError("jev down")

        async def assess_final_claim(self, *a: object, **k: object) -> object:
            raise RuntimeError("jev down")

    down_r = kw._assess_final_response({"id": "r2", "text": "x", "evidence": []}, jev=_Boom())  # type: ignore[arg-type]
    assert down_r["support"] is None and down_r["support_answer"] is None and down_r["evidence_state"] == "insufficient"
    down_c = kw._assess_final_claim(
        {"id": "c2", "claim": {"text": "x", "claim_type": "inference", "evidence_ids": []}, "evidence": []},
        jev=_Boom(),  # type: ignore[arg-type]
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
                {"text": "WSJ reported X.", "claim_type": "attributed_report", "evidence_ids": ["ev-1"]},
                {"text": "If Y then Z.", "claim_type": "conditional_risk", "evidence_ids": ["ev-2"], "confidence": 0.6},
            ],
            "evidence": [{"id": "ev-1", "content": "WSJ: X."}, {"id": "ev-2", "content": "Y risk."}, {"id": "x"}],
            "sessionId": "sess-9",
            "confidence": 0.75,
        },
        jev=client,  # type: ignore[arg-type]
    )
    assert resp["id"] == "r3" and resp["support_answer"] == "yes" and "error" not in resp
    assert client.decide_types[0][0] == "final_response_assessment"  # type: ignore[attr-defined]
    claim = kw._assess_final_claim(
        {
            "id": "c3",
            "op": "assess_final_claim",
            "claim": {"text": "WSJ reported X.", "claim_type": "attributed_report", "evidence_ids": ["ev-1"]},
            "evidence": [{"id": "ev-1", "content": "WSJ: X."}],
            "session_id": "sess-9",
        },
        jev=client,  # type: ignore[arg-type]
    )
    assert claim["id"] == "c3" and claim["support_answer"] == "yes" and "error" not in claim
