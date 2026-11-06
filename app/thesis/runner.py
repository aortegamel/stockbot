"""Bounded kernel research runs over a thesis trigger (stdlib + PyYAML only).

Automated monitoring calls the agent API's native research loop per pending trigger.
The runner acknowledges the stable trigger ID only after the stream reports done
and a durable trigger-linked journal entry exists. A failed run leaves the trigger
pending with valid partial tool writes intact (at-least-once retry, no rollback).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from socket import SHUT_RDWR, socket
from threading import Timer
from urllib.parse import urlsplit

from app.policy import Capability
from app.storage.ids import run_id as new_run_id
from app.thesis.context import ResearchContext, build_live_context
from app.thesis.models import JSONValue, Trigger
from app.thesis.repository import ThesisRepository

_GRANTS: dict[str, Capability] = {
    "broker-market-read": Capability.BROKER_MARKET_READ,
    "portfolio-read": Capability.PORTFOLIO_READ,
}


def _grant_cap(grant: str) -> Capability:
    """Capability for one grant string (unknown grants stay loud)."""
    if grant not in _GRANTS:
        raise ValueError(f"<runner>: unknown grant {grant!r}; expected one of {sorted(_GRANTS)}")
    return _GRANTS[grant]


def capabilities_for_grants(grants: list[str]) -> frozenset[Capability]:
    """Map explicit CLI grant strings to capabilities; reject anything else."""
    return frozenset(_grant_cap(g) for g in grants or [])


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    thesis_id: str
    trigger_id: str
    journal_path: str = ""
    evidence_ids: tuple[str, ...] = ()
    processed: bool = False
    tools_used: tuple[str, ...] = field(default_factory=tuple)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stringify_prompt_text(text: object, ref: str) -> str | None:
    """Non-str text stringified; withheld-marker when unstringifiable."""
    try:
        text = str(text)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return f"[unsafe content withheld ref={ref}]"
    return text or None


def _coerce_prompt_text(text: object, ref: str) -> str | None:
    """Raw text coerced to str; None when blank, withheld-marker when unstringifiable."""
    if text is None or text == "":
        return None
    if isinstance(text, str):
        return text
    return _stringify_prompt_text(text, ref)


def _scan_prompt_text(text: str, ref: str) -> str:
    """Injection-scanner verdict for coerced text."""
    try:
        from app.security.prompt_injection import (
            assess,  # local: keep thesis import graph acyclic
        )

        found = assess(text)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return f"[unsafe content withheld ref={ref}]"
    if found.verdict in ("BLOCK", "QUARANTINE"):
        return f"[unsafe content withheld ref={ref}]"
    return text


def _safe_prompt_text(text: str, ref: str) -> str:
    """Gate stored free text via the shared injection scanner; provenance labels never bypass it."""
    coerced = _coerce_prompt_text(text, ref)
    if coerced is None:
        return ""
    if coerced.startswith("[unsafe content withheld"):
        return coerced
    return _scan_prompt_text(coerced, ref)


def _prompt_field(raw: object, ref: str) -> str:
    """One free-text prompt field gated through the injection scanner."""
    if raw is None or raw == "":
        return ""
    return _safe_prompt_text(raw if isinstance(raw, str) else str(raw), ref=ref)


def _prompt_ref(value: object) -> str:
    """Provenance ref for a prompt row; unknown when blank."""
    return value if isinstance(value, str) and value else "unknown"


def _safe_trigger_row(trig: object, tid_ref: str) -> object:
    """Trigger row with its summary gated; non-dicts pass through."""
    if not isinstance(trig, dict):
        return trig
    out = dict(trig)
    if "summary" in out:
        out["summary"] = _prompt_field(out.get("summary", ""), ref=f"trigger:{tid_ref}")
    return out


def _safe_evidence_row(e: object) -> object:
    """Evidence row with its summary gated; non-dicts pass through."""
    if not isinstance(e, dict):
        return e
    ed = dict(e)
    ed["summary"] = _prompt_field(ed.get("summary", ""), ref=f"evidence:{_prompt_ref(ed.get('evidence_id'))}")
    return ed


def _safe_journal_row(j: object) -> object:
    """Journal row with its excerpt gated; non-dicts pass through."""
    if not isinstance(j, dict):
        return j
    jd = dict(j)
    jd["excerpt"] = _prompt_field(jd.get("excerpt", ""), ref=f"journal:{_prompt_ref(jd.get('journal'))}")
    return jd


def _prompt_sections(
    thesis_id: str,
    trigger: Trigger,
    data_cutoff: str,
    ctx: ResearchContext,
    run_id: str,
    refs: str,
    summary_line: str,
    packet: dict[str, JSONValue],
    trig_out: object,
    evidence_out: list[object],
    journals_out: list[object],
) -> list[str]:
    """Ordered prompt lines: identity, cutoff, state, trigger/evidence, journals, close."""
    return [
        f"thesis_id: {thesis_id}",
        f"trigger_id: {trigger.trigger_id}",
        f"run_id: {run_id}",
        f"TRIGGER DATA CUTOFF: {data_cutoff}",
        f"trigger summary: {summary_line}",
        f"trigger canonical refs: {refs[:500]}",
        "The supplied trigger and stored-evidence packet is bounded by the trigger data cutoff. You may perform additional live research using currently available tools. Preserve the real timing/provenance of anything newly found.",
        "CURRENT THESIS STATE:",
        json.dumps(packet, sort_keys=True),
        "TRIGGER/EVIDENCE AVAILABLE TO THIS MONITOR TICK:",
        json.dumps({"trigger": trig_out, "evidence": evidence_out}, sort_keys=True),
        "CURRENT PRIOR JOURNAL CONTEXT:",
        json.dumps(journals_out, sort_keys=True),
        "Only trigger and evidence inputs are bounded by the trigger data cutoff; thesis, state, watch, questions, memory, and prior journals are current.",
        "Record material findings, supporting and counterevidence, with the thesis_journal tool.",
        (
            "Before completing, write a material thesis_journal entry using"
            f" trigger_id '{trigger.trigger_id}' and run_id '{run_id}'."
        ),
        "Do not copy the trigger data cutoff into journal known_at; omit known_at unless it independently represents when the journal information became known.",
    ]


def _build_prompt(*, thesis_id: str, trigger: Trigger, data_cutoff: str, ctx: ResearchContext, run_id: str) -> str:
    refs = ", ".join(trigger.canonical_refs) or "(none)"
    packet = dict(ctx.thesis_packet)
    trig = packet.pop("trigger", {})
    tid_ref = trigger.trigger_id or "unknown"
    summary_line = _safe_prompt_text(trigger.summary or "", ref=f"trigger:{tid_ref}")[:500]
    trig_out = _safe_trigger_row(trig, tid_ref)
    evidence_out = [_safe_evidence_row(e) for e in ctx.evidence_refs]
    journals_out = [_safe_journal_row(j) for j in ctx.journal_excerpts]
    return "\n".join(
        _prompt_sections(
            thesis_id,
            trigger,
            data_cutoff,
            ctx,
            run_id,
            refs,
            summary_line,
            packet,
            trig_out,
            evidence_out,
            journals_out,
        )
    )


def _fail(run_id: str, exc: Exception) -> None:
    try:
        from app.storage.runs import finalize_failed_run

        finalize_failed_run(run_id, error_type=type(exc).__name__, error_message=str(exc))
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass


def _stop_research_stream(sock: socket) -> None:
    """Interrupt a stalled read when the total research deadline expires."""
    with suppress(OSError):
        sock.shutdown(SHUT_RDWR)


def _default_run_kernel(
    prompt: str,
    *,
    as_of: str | None,
    transport: Callable[..., HTTPConnection] | None = None,
) -> None:
    """Wait for the native research loop. Failed or incomplete streams raise."""
    base = os.environ.get("STOCKBOT_AGENT_API_URL", f"http://127.0.0.1:{os.environ.get('STOCKBOT_API_PORT') or '3001'}")
    timeout = float(os.environ.get("STOCKBOT_THESIS_RESEARCH_TIMEOUT_SECONDS", "300"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("STOCKBOT_THESIS_RESEARCH_TIMEOUT_SECONDS must be finite and positive")
    payload = {"prompt": prompt, "mode": "research", **({"asOf": as_of} if as_of else {})}
    url = urlsplit(f"{base.rstrip('/')}/api/agent")
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("STOCKBOT_AGENT_API_URL must be an HTTP(S) URL")
    connect = transport or (HTTPSConnection if url.scheme == "https" else HTTPConnection)
    connection = connect(url.hostname, url.port, timeout=timeout)
    deadline = time.monotonic() + timeout
    timer: Timer | None = None
    try:
        connection.connect()
        if connection.sock is None:
            raise RuntimeError("Research API connection has no socket")
        timer = Timer(max(0, deadline - time.monotonic()), _stop_research_stream, (connection.sock,))
        timer.daemon = True
        timer.start()
        connection.request(
            "POST",
            url.path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"Research API returned HTTP {response.status}")
            for line in response:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Research API timed out")
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:])
                if event.get("type") in {"failed", "error"}:
                    raise RuntimeError(str(event.get("message") or "Research failed"))
                if event.get("type") == "done":
                    return
    except (OSError, HTTPException) as exc:
        raise RuntimeError(f"Research API failed: {exc}") from exc
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
    raise RuntimeError("Research API stream ended without done")


_RUN_KERNEL: Callable[..., None] | None = None


def _run_kernel(prompt: str, *, as_of: str | None, thesis_id: str, trigger_id: str, run_id: str) -> None:
    """Injectable kernel seam (tests monkeypatch module _RUN_KERNEL)."""
    if _RUN_KERNEL is not None:
        fake = _RUN_KERNEL
        try:
            fake(prompt, thesis_id=thesis_id, trigger_id=trigger_id, run_id=run_id)
        except TypeError:
            fake(prompt)
        return
    _default_run_kernel(prompt, as_of=as_of)


def run_trigger(
    repository: ThesisRepository,
    thesis_id: str,
    trigger_id: str,
    *,
    known_at: str | None = None,
) -> RunOutcome:
    """Run the shared graph fan-out for one pending trigger; ack that trigger ID only.

    Failure (bad state, over-budget context, kernel failure, or no durable
    trigger-linked journal) raises before acknowledgement, so the trigger
    stays pending and valid partial tool writes are retained for the retry.
    """
    known_at = known_at or _utcnow()
    thesis = repository.load_thesis(thesis_id)
    tid = thesis.thesis_id
    trigger = next((t for t in repository.load_triggers(tid) if t.trigger_id == trigger_id), None)
    if trigger is None:
        raise KeyError(f"unknown trigger: {trigger_id!r}")
    if trigger.status != "pending":
        raise ValueError(f"<runner>: trigger {trigger_id!r} is {trigger.status}, not pending")
    # PIT/budget gate: raises before any kernel call when context is over budget.
    ctx = build_live_context(repository, tid, trigger, data_cutoff=known_at)
    rid = new_run_id()
    prompt = _build_prompt(thesis_id=tid, trigger=trigger, data_cutoff=known_at, ctx=ctx, run_id=rid)
    try:
        _run_kernel(prompt, as_of=known_at, thesis_id=tid, trigger_id=trigger.trigger_id, run_id=rid)
        repository.load_triggers(tid)  # re-read: surface corrupt YAML instead of acking blind
        if not repository.has_journal_for_trigger(tid, trigger.trigger_id, run_id=rid):
            raise RuntimeError(
                f"<runner>: no durable journal for trigger {trigger.trigger_id!r} (thesis {tid!r});"
                f" the kernel run must write a material thesis_journal entry with trigger_id {trigger.trigger_id!r}"
                f" and run_id {rid!r} before the trigger can be acknowledged;"
                " leaving pending for retry"
            )
        repository.mark_trigger_processed(tid, trigger.trigger_id, rid)
    except Exception as exc:
        _fail(rid, exc)
        raise
    return RunOutcome(run_id=rid, thesis_id=tid, trigger_id=trigger.trigger_id, processed=True)
