"""JEV bridge Python client: typed decide + select_tool/assess_result/adjudicate.

Transport only + persistence. The TypeSafe SDK call and answer-shape
validation live in decision/runtime.ts (JSONL sidecar); Python owns
persistence: DecisionRecord domain fields -> research.sqlite, full
request/response + latency -> runs.sqlite via the ambient RunRecorder.
Fallbacks: injectable transport stub (tests), direct HTTPS systemOne when
the sidecar is absent. stdlib only.
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.config import get_data_root
from app.research.models import (
    DecisionRecord,
    JSONValue,
    ToolDecision,
    new_decision_id,
    query_with_today_utc,
    utcnow,
    validate_json_mapping,
    validate_json_value,
)
from app.research.repository import ResearchRepository, get_research_db_path
from app.storage.runs import get_current_recorder

logger = logging.getLogger(__name__)

__all__ = ["JevClient"]


@runtime_checkable
class _HasToDict(Protocol):
    def to_dict(self) -> object: ...


_PROVIDER = "typesafe"
_DEFAULT_MODEL = "jev-latest"

# Mirror of decision/jev.ts TOOL_SELECTION_SENTINELS (names are the contract).
REASON_SENTINEL = "reasoning_required"
RESOLVED_SENTINEL = "node_resolved"
_SENTINEL_DESCRIPTIONS = {
    REASON_SENTINEL: "Escalate to the reasoner (decompose/analyze proposals); no tool call fits this node.",
    RESOLVED_SENTINEL: "Existing evidence resolves the node; no further tool call needed.",
}

# Entry routing (JEV-before-kernel): fast-path + research sentinels framing the whole canonical registry.
_ENTRY_OPTIONS = {
    REASON_SENTINEL: "Answerable by reasoning/explanation from user-supplied context without new external evidence.",
    "research_required": "Needs current/external facts, evidence retrieval, source verification, or tool execution.",
}
# Opt-in parallel fan-out: runner-up joins only when close to the winner and above floor (cap keeps blast radius small).
_PARALLEL_MIN_PROB = 0.35
_PARALLEL_WINDOW = 0.15
_PARALLEL_CAP = 3

# Mirror of decision/jev.ts EVIDENCE_STATE_OPTIONS (choice labels are the contract).
EVIDENCE_STATE_OPTIONS = {
    "sufficient_support": "The cited evidence sufficiently supports the claim.",
    "sufficient_contradiction": "The cited evidence sufficiently contradicts the claim.",
    "conflicted": "The evidence both supports and contradicts the claim; do not resolve by guessing.",
    "insufficient": "The evidence is missing or too weak to support or contradict the claim.",
}

CONTINUE_OPTIONS = {
    "resolve_node": "Existing evidence resolves the node; stop gathering.",
    "continue_research": "Evidence is useful but the node needs more tool calls.",
    "reason_over_evidence": "Escalate to the reasoner over the gathered evidence.",
}


class _SidecarUnavailable(RuntimeError):
    """Sidecar missing or dead; caller falls back to direct HTTPS."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_prob(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0


def _parse_noul(name: str, qid: str, ans: object) -> dict[str, JSONValue]:
    bad = ValueError(f"{name}: malformed_typesafe_answer for {qid}")
    if not isinstance(ans, dict) or ans.get("type") != "noul":
        raise bad
    if not _is_prob(ans.get("noul")):
        raise bad
    return {"kind": "noul", "probability": float(ans["noul"])}


def _parse_choice(name: str, qid: str, ans: object, options: Mapping[str, str]) -> dict[str, JSONValue]:
    bad = ValueError(f"{name}: malformed_typesafe_answer for {qid}")
    if not isinstance(ans, dict) or ans.get("type") != "choice":
        raise bad
    choice = ans.get("choice")
    probs = ans.get("probabilities")
    conf = ans.get("confidence")
    if not isinstance(choice, str) or choice not in options:
        raise bad
    if not isinstance(probs, dict) or not isinstance(conf, (int, float)) or isinstance(conf, bool):
        raise bad
    want = sorted(options)
    got = sorted(probs)
    if got != want or not _is_prob(conf):
        raise bad
    out: dict[str, JSONValue] = {}
    for k in want:
        if not _is_prob(probs[k]):
            raise bad
        out[k] = float(probs[k])
    return {"kind": "choice", "choice": choice, "probabilities": out, "confidence": float(conf)}


def _parse_score(name: str, qid: str, ans: object, max_score: float | None) -> dict[str, JSONValue]:
    bad = ValueError(f"{name}: malformed_typesafe_answer for {qid}")
    if not isinstance(ans, dict) or ans.get("type") != "score":
        raise bad
    score = ans.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not __import__("math").isfinite(score):
        raise bad
    if max_score is not None and not 0 <= float(score) <= max_score:
        raise bad
    out: dict[str, JSONValue] = {"kind": "score", "score": score}
    if ans.get("confidence") is not None:
        if not _is_prob(ans.get("confidence")):
            raise bad
        out["confidence"] = float(ans["confidence"])
    if ans.get("probabilities") is not None:
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            raise bad
        keep: dict[str, JSONValue] = {}
        for k, v in probs.items():
            if not _is_prob(v):
                raise bad
            keep[str(k)] = float(v)
        out["probabilities"] = keep
    if "legend" in ans:
        out["raw"] = ans["legend"]
    return out


def _score_max_from_question(q: object) -> float | None:
    if not isinstance(q, dict):
        return None
    for key in ("criteria", "levels"):
        seq = q.get(key)
        if isinstance(seq, list) and len(seq) >= 2:
            return float(len(seq) - 1)
    return None


def _options_from_question(q: object) -> dict[str, str]:
    if isinstance(q, dict) and isinstance(q.get("criteria"), dict):
        return {str(k): str(k) for k in q["criteria"]}
    return {}


def parse_decisions(
    questions: Mapping[str, JSONValue],
    raw: object,
    choice_options: Mapping[str, Mapping[str, str]] | None = None,
    name: str = "decide",
) -> dict[str, dict[str, JSONValue]]:
    """Validate a raw SystemOne payload into per-question decisions (mirrors askDecisions)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise ValueError(f"{name}: malformed_typesafe_response")
    answers: dict[str, JSONValue] = raw["answers"]
    if sorted(answers) != sorted(questions):
        raise ValueError(f"{name}: typesafe answers do not match questions")
    out: dict[str, dict[str, JSONValue]] = {}
    for qid in sorted(questions):
        q = questions[qid]
        ans = answers[qid]
        qtype = q.get("type") if isinstance(q, dict) else None
        atype = ans.get("type") if isinstance(ans, dict) else None
        kind = qtype if qtype in ("choice", "score") else atype
        if kind == "choice":
            opts = (choice_options or {}).get(qid) or _options_from_question(q)
            out[qid] = _parse_choice(name, qid, ans, opts)
        elif kind == "score":
            out[qid] = _parse_score(name, qid, ans, _score_max_from_question(q))
        else:
            out[qid] = _parse_noul(name, qid, ans)
    return out


def _field(obj: object, *names: str, default: JSONValue | None = None) -> JSONValue | None:
    for name in names:
        candidate: object = (
            obj.get(name) if isinstance(obj, dict) and obj.get(name) is not None else getattr(obj, name, None)
        )
        if candidate is None:
            continue
        try:
            return validate_json_value(candidate, "<decision_client>: '_field'")
        except ValueError:
            continue
    return default


def _node_dict(node: object) -> dict[str, JSONValue]:
    raw: dict[str, JSONValue]
    if isinstance(node, dict):
        raw = validate_json_mapping(node, "<node_dict>")
    elif isinstance(node, _HasToDict):
        try:
            got = node.to_dict()
            raw = validate_json_mapping(got, "<node_dict>") if isinstance(got, dict) else {}
        except Exception:
            raw = {}
    else:
        raw = {}
        try:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(node) and not isinstance(node, type):
                candidate = asdict(node)
                raw = validate_json_mapping(candidate, "<node_dict>") if isinstance(candidate, dict) else {}
        except Exception:
            raw = {}
        if not raw:
            for key in (
                "node_id",
                "id",
                "session_id",
                "question",
                "why_it_matters",
                "depends_on",
                "status",
                "evidence_ids",
                "missing_evidence",
            ):
                attr: object = getattr(node, key, None)
                if attr is not None:
                    raw[key] = validate_json_value(attr, "<node_dict>")
    out: dict[str, JSONValue] = {}
    for key in (
        "node_id",
        "id",
        "session_id",
        "question",
        "why_it_matters",
        "depends_on",
        "status",
        "evidence_ids",
        "missing_evidence",
    ):
        value: object = raw.get(key)
        if value is None:
            continue
        if isinstance(value, tuple):
            out[key] = validate_json_value(list(value), "<decision_client>: '_node_dict'")
        else:
            out[key] = validate_json_value(value, "<decision_client>: '_node_dict'")
    return out


_MAX_PRIOR_FAILURES = 3
_MAX_OUTCOME_CONTENT = 2000


def _outcome_dict(outcome: object) -> dict[str, JSONValue]:
    def text(value: object) -> str | None:
        return value if isinstance(value, str) else (None if value is None else str(value))

    content = text(_field(outcome, "content"))
    return {
        "tool": _field(outcome, "tool_name", "tool"),
        "content": content[:_MAX_OUTCOME_CONTENT] if isinstance(content, str) else content,
        "error": text(_field(outcome, "error")),
        "error_type": text(_field(outcome, "error_type")),
        "tool_result_id": text(_field(outcome, "tool_result_id", "tool_result_ref")),
    }


def _evidence_ids(evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None) -> list[str]:
    ids: list[str] = []
    if not isinstance(evidence, list):
        return ids
    for item in evidence:
        if isinstance(item, dict):
            for key in ("evidence_id", "id", "job_id", "tool_result_ref"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
        if len(ids) >= 10:
            break
    return ids


def _observation_lines(evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None) -> list[str]:
    """Recent unadmitted tool observations (ctx evidence without an evidence id)."""
    lines: list[str] = []
    if not isinstance(evidence, list):
        return lines
    for item in evidence[-10:]:
        if not isinstance(item, dict) or item.get("evidence_id") or item.get("id"):
            continue
        summary = item.get("outcome_summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        tool = item.get("tool")
        ref = item.get("tool_result_ref")
        head = str(tool) if isinstance(tool, str) and tool else "tool"
        if isinstance(ref, str) and ref.strip():
            head += f" ref={ref.strip()[:60]}"
        error = item.get("error")
        if isinstance(error, str) and error.strip():
            head += f" FAILED: {error.strip()[:200]}"
        else:
            head += f": {' '.join(summary.split())[:200]}"
        lines.append(head)
        if len(lines) >= 5:
            break
    return lines


def _manifest_line(entry: Mapping[str, JSONValue]) -> str:
    """One compact option line per registry entry (mirrors decision/jev.ts manifestLine)."""

    def opt(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        return " ".join(value.split())

    desc = entry.get("description")
    raw_name = entry.get("name", "")
    line = " ".join(desc.split()) if isinstance(desc, str) and desc else (raw_name if isinstance(raw_name, str) else "")
    purpose = opt(entry.get("purpose"))
    if purpose and purpose != line:
        line += f" Purpose: {purpose}."
    intent = opt(entry.get("intent"))
    if intent:
        line += f" Intent: {intent}."
    inputs = opt(entry.get("keyInputs"))
    if inputs:
        line += f" Inputs: {inputs}."
    output = opt(entry.get("outputKind"))
    if output:
        line += f" Output: {output}."
    evidence = opt(entry.get("evidence"))
    if evidence:
        line += f" Evidence: {evidence}."
    prereq = opt(entry.get("prerequisites"))
    if prereq:
        line += f" Needs: {prereq}."
    pit = opt(entry.get("pitSupport"))
    if pit:
        line += f" PIT: {pit}."
    use = opt(entry.get("useWhen"))
    if use:
        line += f" Use: {use}."
    avoid = opt(entry.get("avoidWhen"))
    if avoid:
        line += f" Avoid: {avoid}."
    conflicts = opt(entry.get("conflicts"))
    if conflicts:
        line += f" Conflicts: {conflicts}."
    nxt = opt(entry.get("nextTools"))
    if nxt:
        line += f" Next: {nxt}."
    domain = opt(entry.get("domain"))
    if domain:
        line = f"[{domain}] {line}"
    return line


def _tool_options_prompt(
    registry: Sequence[Mapping[str, JSONValue]],
    node: Mapping[str, JSONValue],
    evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None,
    attempts: Sequence[JSONValue] | Mapping[str, JSONValue] | None,
) -> tuple[dict[str, str], str]:
    if not registry:
        raise ValueError("tool_selection: empty registry")
    node_id = node.get("node_id") or node.get("id")
    question = node.get("question")
    if not node_id or not question:
        raise ValueError("tool_selection: node needs nodeId and question")
    options: dict[str, str] = {}
    for entry in registry:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("tool_selection: registry entry needs a name")
        if name in options:
            raise ValueError(f"tool_selection: duplicate tool {name}")
        options[name] = _manifest_line(entry) if isinstance(entry, dict) else name
    for key, desc in _SENTINEL_DESCRIPTIONS.items():
        if key in options:
            raise ValueError(f"tool_selection: registry collides with sentinel {key}")
        options[key] = desc
    stamped = query_with_today_utc(question) if isinstance(question, str) else question
    prompt = f"Which single tool runs next for research node {node_id}? Question: {stamped} JEV owns this selection and every transition over the whole canonical registry; Needle runs args only and never selects, chains, or judges. Choose exactly one winner."
    why = node.get("why_it_matters")
    if isinstance(why, str) and why:
        prompt += f" Why it matters: {why}"
    ids = _evidence_ids(evidence)
    count = len(evidence) if isinstance(evidence, list) else 0
    prompt += f" Evidence on hand: {count} item(s){f' [{chr(44).join(ids)}]' if ids else ''}."
    for line in _observation_lines(evidence):
        prompt += f" Observed {line}."
    if isinstance(attempts, list):
        recent: list[tuple[str, str]] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            tool = attempt.get("tool")
            error = attempt.get("error")
            if isinstance(tool, str) and isinstance(error, str) and tool and error:
                recent.append((tool, error))
        for tool, error in recent[-_MAX_PRIOR_FAILURES:]:
            prompt += f" Prior attempt {tool} failed: {error[:200]}."
    prompt += f" Today UTC is {utcnow().date().isoformat()}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week'/'today'/'last quarter' as arg values."
    prompt += " Search packets list candidates only — page with research_read_search beyond display_limit and open each accession via get_sec_filing/get_sec_document until answered or guard trips."
    _qtext = stamped if isinstance(stamped, str) else (question if isinstance(question, str) else "")
    if "revenue" in _qtext.lower() and "quarter" in _qtext.lower():
        prompt += " Revenue-quarter answers come from the 10-Q (MD&A/segment revenue), not the 8-K: after the filing list, open the 10-Q accession via get_sec_document with a revenue query before re-reading any 8-K."
    return options, prompt


def _auto_registry() -> list[dict[str, JSONValue]]:
    try:
        from app.research.scheduler import build_registry

        reg = build_registry()
        if reg:
            return reg
    except Exception:
        pass
    from app.policy import Capability
    from app.security.action_policy import TOOL_DOMAINS
    from app.tools import TOOL_DISCOVERY_REGISTRY, tools_for_capabilities

    try:  # scheduler import failed above; reuse its handle/blind lists when available
        from app.research.scheduler import _HANDLE_PARAMS as _HANDLES
        from app.research.scheduler import _PIT_BLIND_TOOLS as _BLIND
    except Exception:
        _HANDLES = frozenset()
        _BLIND = frozenset()

    out: list[dict[str, JSONValue]] = []
    for tool in tools_for_capabilities(frozenset({Capability.RESEARCH})):
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        params = dict(params) if isinstance(params, dict) else {}
        required = params.get("required")
        req_list: list[str] = [str(k) for k in required if isinstance(k, str)] if isinstance(required, list) else []
        required = req_list
        meta = TOOL_DISCOVERY_REGISTRY.get(name)
        out.append(
            {
                "name": name,
                "domain": TOOL_DOMAINS.get(name, "unknown"),
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "purpose": meta.summary if meta is not None else "",
                "keyInputs": f"req({', '.join(required)})" if required else "req()",
                "outputKind": meta.output_kind if meta is not None else "",
                "evidence": meta.output_kind if meta is not None else "",
                "prerequisites": f"needs {', '.join(needs)}"
                if (needs := [k for k in required if k in _HANDLES])
                else "",
                # ponytail: degraded path (scheduler unimportable); blind list best-effort.
                "pitSupport": "PIT-blind: current state only" if name in _BLIND else "PIT-scoped",
                "intent": meta.intent if meta is not None else "",
                "useWhen": "; ".join(meta.choose_when) if meta is not None else "",
                "avoidWhen": "; ".join(meta.reject_when) if meta is not None else "",
                "conflicts": ", ".join(meta.conflicts_with) if meta is not None else "",
                "nextTools": ", ".join(meta.related_tools) if meta is not None else "",
                "parameters": params,
            }
        )
    return out


_LIVE_PROCS: set[subprocess.Popen[str]] = set()
_ATEXIT_ARMED = False


def _arm_atexit() -> None:
    global _ATEXIT_ARMED
    if _ATEXIT_ARMED:
        return
    _ATEXIT_ARMED = True

    def _teardown() -> None:
        for proc in list(_LIVE_PROCS):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001, S110 - best-effort teardown at exit
                pass

    atexit.register(_teardown)


def _trunc200(value: object) -> str:
    """Compact single-line truncation for prompts/objectives; never raises (logging only)."""
    try:
        s = value if isinstance(value, str) else ("" if value is None else str(value))
        return " ".join(s.split())[:200]
    except Exception:  # noqa: BLE001 - logging-only helper, never raises
        return "?"


def _rank_prob(item: tuple[str, float]) -> float:
    return item[1]


def _choice_summary(decision: object) -> str:
    """Winner + confidence + top-3 + margin one-liner; never raises (logging only)."""
    try:
        if not isinstance(decision, dict):
            return "winner=- conf=- top3=[] margin=0.00"
        winner = decision.get("choice")
        w = winner if isinstance(winner, str) and winner else "-"
        conf = decision.get("confidence")
        c = f"{float(conf):.2f}" if isinstance(conf, (int, float)) and not isinstance(conf, bool) else "-"
        probs = decision.get("probabilities")
        floats: dict[str, float] = {}
        if isinstance(probs, dict):
            for k, v in probs.items():
                if _is_prob(v):
                    floats[str(k)] = float(v)  # type: ignore[arg-type]
        ranked = sorted(floats.items(), key=_rank_prob, reverse=True)
        top = ",".join(f"{k}:{v:.2f}" for k, v in ranked[:3])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) >= 2 else 0.0
        return f"winner={w} conf={c} top3=[{top}] margin={margin:.2f}"
    except Exception:  # noqa: BLE001 - logging-only helper, never raises
        return "winner=? conf=? top3=[] margin=?"


def _args_summary(args: Mapping[str, JSONValue]) -> str:
    """Arg keys + byte size only; values never logged (may carry secrets)."""
    try:
        size = len(json.dumps(dict(args), default=str))
        return f"argkeys=[{','.join(sorted(str(k) for k in args))}] argbytes={size}"
    except Exception:  # noqa: BLE001 - logging-only helper, never raises
        return "argkeys=[] argbytes=?"


def _jev_usage(raw: object) -> tuple[int | None, int | None, str | None]:
    """Extract (input_tokens, output_tokens, model) from a raw SystemOne result; never raises."""
    try:
        if not isinstance(raw, dict):
            return None, None, None
        model_raw = raw.get("model")
        model = model_raw if isinstance(model_raw, str) and model_raw else None
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return None, None, model
        in_raw = usage.get("input_tokens")
        out_raw = usage.get("output_tokens")
        jev_in = in_raw if isinstance(in_raw, int) and not isinstance(in_raw, bool) else None
        jev_out = out_raw if isinstance(out_raw, int) and not isinstance(out_raw, bool) else None
        return jev_in, jev_out, model
    except Exception:  # noqa: BLE001 - logging/persistence helper, never raises
        return None, None, None


class JevClient:
    """JEV bridge: typed decide over the sidecar, multi-tool-aware select, persistence."""

    def __init__(
        self,
        *,
        transport: Callable[..., object] | None = None,
        runtime_path: Path | str | None = None,
        timeout_s: float = 60.0,
        data_root: Path | None = None,
        provider: str = _PROVIDER,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._transport = transport
        self._timeout_s = timeout_s
        self._data_root = Path(data_root) if data_root is not None else get_data_root()
        self._provider = provider
        self._model = model
        # ponytail: single-flight lock; pipeline if sidecar throughput matters.
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        root = Path(__file__).resolve().parent.parent
        self._runtime_ts = Path(runtime_path) if runtime_path is not None else root / "decision" / "runtime.ts"
        self._repo_root = root

    def start(self) -> None:
        """Zero-arg ping handshake: ensure the sidecar is alive, reuse the proc."""
        payload: dict[str, JSONValue] = {"id": f"jev:ping:{uuid.uuid4().hex[:12]}", "op": "ping"}
        response = self._sidecar_roundtrip(payload)
        if response.get("id") != payload["id"] or response.get("ready") is not True:
            raise RuntimeError(f"jev ping failed: unexpected sidecar ack for {payload['id']!r}")

    def close(self) -> None:
        if self._lock.acquire(blocking=False):
            try:
                proc, self._proc = self._proc, None
            finally:
                self._lock.release()
        else:
            # A roundtrip holds the lock while blocked in readline; waiting
            # here would freeze the event loop on a hung sidecar. Terminate
            # the live handle lock-free so its readline returns EOF; the
            # holder then discards it, releases the lock, and the next call
            # restarts via _ensure_proc.
            proc = self._proc
        if proc is not None:
            _LIVE_PROCS.discard(proc)
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001, S110 - best-effort close
                pass

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            _LIVE_PROCS.discard(self._proc)
            self._proc = None
        if not self._runtime_ts.exists():
            raise _SidecarUnavailable(f"sidecar missing: {self._runtime_ts}")
        try:
            proc = subprocess.Popen(
                ["bun", str(self._runtime_ts)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=str(self._repo_root),
            )
        except FileNotFoundError as exc:
            raise _SidecarUnavailable("bun unavailable for sidecar") from exc
        _arm_atexit()
        _LIVE_PROCS.add(proc)
        self._proc = proc
        return proc

    def _sidecar_roundtrip(self, payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        with self._lock:
            try:
                proc = self._ensure_proc()
            except _SidecarUnavailable:
                raise
            line = json.dumps(payload) + "\n"
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(line)
                proc.stdin.flush()
                raw_line = proc.stdout.readline()
            except (BrokenPipeError, OSError):
                _LIVE_PROCS.discard(proc)
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001, S110 - best-effort restart
                    pass
                self._proc = None
                proc = self._ensure_proc()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(line)
                proc.stdin.flush()
                raw_line = proc.stdout.readline()
            if not raw_line:
                _LIVE_PROCS.discard(proc)
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001, S110 - best-effort restart
                    pass
                self._proc = None
                raise _SidecarUnavailable("sidecar closed (EOF)")
            try:
                response = json.loads(raw_line)
            except ValueError as exc:
                raise RuntimeError(f"decide: malformed sidecar response: {exc}") from exc
            if not isinstance(response, dict):
                raise RuntimeError("decide: malformed sidecar response")
            return validate_json_mapping(response, "<decision_client>: 'sidecar'")

    def _http_system_one(self, state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("decide: jev unavailable (no sidecar, no TYPESAFE_API_KEY for direct call)")
        base = (os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        body: dict[str, JSONValue] = {"state": validate_json_value(state, "<state>"), "questions": dict(questions)}
        model = (os.environ.get("TYPESAFE_DEFAULT_MODEL") or "").strip()
        if model:
            body["model"] = model
        req = urllib.request.Request(
            f"{base}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as exc:
            raise RuntimeError(f"decide: typesafe_request_failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("decide: malformed_typesafe_response")
        return validate_json_mapping(raw, "<decision_client>: 'http'")

    async def _invoke(
        self,
        state: object,
        questions: Mapping[str, JSONValue],
        choice_options: Mapping[str, Mapping[str, str]] | None,
    ) -> tuple[dict[str, dict[str, JSONValue]], JSONValue, str]:
        if self._transport is not None:
            raw = self._transport(state, dict(questions))
            if inspect.isawaitable(raw):
                raw = await raw
            logger.debug(
                "toolflow invoke_detail sid=- nid=- via=stub qids=[%s]", ",".join(sorted(str(k) for k in questions))
            )
            return parse_decisions(questions, raw, choice_options), validate_json_value(raw, "<raw>"), "stub"
        payload: dict[str, JSONValue] = {
            "id": f"jev:{uuid.uuid4().hex[:12]}",
            "op": "decide",
            "state": validate_json_value(state, "<state>"),
            "questions": dict(questions),
        }
        if choice_options:
            payload["choiceOptions"] = {k: dict(v) for k, v in choice_options.items()}
        try:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._sidecar_roundtrip, payload), timeout=self._timeout_s
                )
            except _SidecarUnavailable:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(self._http_system_one, state, questions), timeout=self._timeout_s
                )
                logger.debug(
                    "toolflow invoke_detail sid=- nid=- via=http qids=[%s]", ",".join(sorted(str(k) for k in questions))
                )
                return parse_decisions(questions, raw, choice_options), raw, "http"
            except TimeoutError as exc:
                self.close()
                raise RuntimeError(f"decide: typesafe timeout after {self._timeout_s}s") from exc
        except asyncio.CancelledError:
            self.close()
            raise
        if not isinstance(response, dict) or response.get("id") != payload["id"]:
            raise RuntimeError("decide: malformed sidecar response")
        if "error" in response:
            raise RuntimeError(f"decide: {response['error']}")
        decisions = response.get("decisions")
        if not isinstance(decisions, dict) or not decisions:
            raise RuntimeError("decide: malformed sidecar response")
        typed: dict[str, dict[str, JSONValue]] = {}
        for qid, val in validate_json_mapping(decisions, "<decision_client>: 'decisions'").items():
            if not isinstance(val, dict):
                raise RuntimeError("decide: malformed sidecar response")
            typed[qid] = validate_json_mapping(val, "<decision_client>: 'decision'")
        raw_out: object = response.get("raw")
        logger.debug(
            "toolflow invoke_detail sid=- nid=- via=sidecar qids=[%s]", ",".join(sorted(str(k) for k in questions))
        )
        return typed, validate_json_value(raw_out, "<raw>"), "sidecar"

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
        """Typed JEV decide; persists the round best-effort, raises on any defect."""
        if not decision_type or not isinstance(decision_type, str):
            raise ValueError("decide: decision_type required")
        if not session_id or not isinstance(session_id, str):
            raise ValueError("decide: session_id required")
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("decide: missing_questions")
        try:
            json.dumps({"state": state, "questions": dict(questions), "options": choice_options})
        except TypeError as exc:
            raise ValueError(f"decide: unserializable request: {exc}") from exc
        started_at = _now()
        start = time.perf_counter()
        try:
            decisions, raw, via = await self._invoke(state, questions, choice_options)
        except Exception as exc:
            logger.warning(
                "toolflow decide_defect sid=%s nid=%s type=%s err=%s: %s",
                session_id,
                node_id if isinstance(node_id, str) and node_id else "-",
                decision_type,
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        completed_at = _now()
        latency_ms = (time.perf_counter() - start) * 1000.0
        jev_in, jev_out, jev_model = _jev_usage(raw)
        self._persist(
            decision_type=decision_type,
            session_id=session_id,
            node_id=node_id,
            job_id=job_id,
            questions=dict(questions),
            decisions=decisions,
            raw=raw,
            via=via,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=completed_at,
            jev_in=jev_in,
            jev_out=jev_out,
            jev_model=jev_model,
        )
        logger.info(
            "toolflow decide_exit sid=%s nid=%s type=%s qids=[%s] via=%s latency_ms=%.1f in=%s out=%s",
            session_id,
            node_id if isinstance(node_id, str) and node_id else "-",
            decision_type,
            ",".join(sorted(str(k) for k in decisions)),
            via,
            latency_ms,
            jev_in,
            jev_out,
        )
        logger.debug(
            "toolflow decide_detail sid=%s nid=%s type=%s options=%s",
            session_id,
            node_id if isinstance(node_id, str) and node_id else "-",
            decision_type,
            {str(k): (len(v) if isinstance(v, Mapping) else -1) for k, v in choice_options.items()}
            if isinstance(choice_options, Mapping)
            else "-",
        )
        return decisions

    async def select_tool(
        self,
        objective: str,
        node: object,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None = None,
        attempts: Sequence[JSONValue] | Mapping[str, JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
    ) -> ToolDecision:
        """JEV owns ALL tool selection/transitions over the whole canonical registry every round; Needle is args-only (never selects/chains/judges). Caller assembles the whole registry; single-winner choice + 2 sentinels."""
        objective = query_with_today_utc(objective) if isinstance(objective, str) else objective
        reg = list(registry) if registry else _auto_registry()
        node_d = _node_dict(node)
        nid = node_d.get("node_id") or node_d.get("id")
        nid_s = nid if isinstance(nid, str) and nid else "-"
        try:
            options, prompt = _tool_options_prompt(reg, node_d, evidence, attempts)
        except Exception as exc:
            logger.warning(
                "toolflow select_defect sid=%s nid=%s registry=%d err=%s: %s",
                session_id,
                nid_s,
                len(reg),
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        logger.info(
            "toolflow select_entry sid=%s nid=%s registry=%d options=%d objective=%r",
            session_id,
            nid_s,
            len(reg),
            len(options),
            _trunc200(objective),
        )
        logger.debug(
            "toolflow select_detail sid=%s nid=%s question=%r evidence=%d attempts=%d",
            session_id,
            nid_s,
            _trunc200(node_d.get("question")),
            len(evidence) if isinstance(evidence, (list, dict)) else 0,
            len(attempts) if isinstance(attempts, (list, dict)) else 0,
        )
        questions: dict[str, JSONValue] = {
            "tool_selection": {
                "type": "choice",
                "instructions": prompt,
                "criteria": validate_json_mapping(options, "<decision_client>: 'criteria'"),
            }
        }
        state = {"objective": objective, "node": node_d, "evidence": evidence or [], "attempts": attempts or []}
        decisions = await self.decide(
            state,
            questions,
            decision_type="tool_selection",
            session_id=session_id,
            node_id=nid if isinstance(nid, str) else None,
            job_id=job_id,
            choice_options={"tool_selection": options},
        )
        return self._to_tool_decision(
            decisions.get("tool_selection"),
            options,
            set_of_registry={o for o in options} - set(_SENTINEL_DESCRIPTIONS),
            sid=session_id,
            nid=nid if isinstance(nid, str) else None,
            event="select",
        )

    async def route_entry(
        self,
        prompt: str,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
    ) -> str:
        """JEV-first entry route: one choice over reasoning_required + whole canonical registry + research_required. Uses _invoke directly so nothing persists (never decide); raises on blank prompt, empty registry, or any JEV outage — the caller fail-opens to research_required."""
        text = query_with_today_utc(prompt.strip()) if isinstance(prompt, str) else ""
        if not text:
            logger.warning("toolflow route_defect sid=- nid=- err=blank_prompt")
            raise ValueError("route_entry: blank prompt")
        reg = list(registry) if registry else _auto_registry()
        try:
            if not reg:
                raise ValueError("route_entry: empty registry")
            options: dict[str, str] = dict(_ENTRY_OPTIONS)
            for entry in reg:
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str) or not name:
                    raise ValueError("route_entry: registry entry needs a name")
                if name in options:
                    raise ValueError(f"route_entry: duplicate tool {name}")
                options[name] = _manifest_line(entry)
        except Exception as exc:
            logger.warning(
                "toolflow route_defect sid=- nid=- registry=%d err=%s: %s",
                len(reg),
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        logger.info(
            "toolflow route_entry sid=- nid=- registry=%d options=%d prompt=%r",
            len(reg),
            len(options),
            _trunc200(text),
        )
        questions: dict[str, JSONValue] = {
            "entry": {
                "type": "choice",
                "instructions": f"Route this entry prompt with exactly one winner: {text} reasoning_required answers from user-supplied context with no new external evidence and no tool; a registry tool runs single-shot first when it fits; otherwise research_required. Today UTC is {utcnow().date().isoformat()}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week'/'today'/'last quarter' as arg values.",
                "criteria": validate_json_mapping(options, "<decision_client>: 'criteria'"),
            }
        }
        start = time.perf_counter()
        try:
            decisions, raw, _via = await self._invoke({"prompt": text}, questions, {"entry": options})
        except Exception as exc:
            logger.warning("toolflow route_defect sid=- nid=- err=%s: %s", type(exc).__name__, str(exc)[:200])
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        d = decisions.get("entry")
        winner = d.get("choice") if isinstance(d, dict) else None
        if not isinstance(winner, str) or winner not in options:
            logger.warning(
                "toolflow route_defect sid=- nid=- winner=%r err=winner_not_in_options %s",
                winner,
                _choice_summary(d),
            )
            raise ValueError(f"route_entry: winner {winner!r} not in options")
        jev_in, jev_out, _ = _jev_usage(raw)
        logger.info(
            "toolflow route_exit sid=- nid=- %s via=%s latency_ms=%.1f in=%s out=%s",
            _choice_summary(d),
            _via,
            latency_ms,
            jev_in,
            jev_out,
        )
        return winner

    async def assess_entry_tool(
        self,
        prompt: str,
        tool: str,
        arguments: Mapping[str, JSONValue] | None = None,
        result: Mapping[str, JSONValue] | None = None,
    ) -> str:
        """Non-persisted post-tool verdict for the direct single-shot path.

        One choice over reasoning_required + research_required + node_resolved
        + the whole canonical registry: the sentinels stop (resolve/reason/
        research), a registry tool name chains one more round. Uses _invoke
        directly (never decide: nothing persists); raises on bad input or a
        winner outside the options — the worker fail-opens to research_required.
        """
        text = query_with_today_utc(prompt.strip()) if isinstance(prompt, str) else ""
        if not text:
            logger.warning(
                "toolflow assess_defect sid=- nid=- tool=%s err=blank_prompt",
                tool if isinstance(tool, str) and tool else "-",
            )
            raise ValueError("assess_entry_tool: blank prompt")
        if not isinstance(tool, str) or not tool:
            logger.warning("toolflow assess_defect sid=- nid=- err=tool_required")
            raise ValueError("assess_entry_tool: tool required")
        reg = _auto_registry()
        try:
            if not reg:
                raise ValueError("assess_entry_tool: empty registry")
            options: dict[str, str] = dict(_ENTRY_OPTIONS)
            if RESOLVED_SENTINEL not in options:
                options[RESOLVED_SENTINEL] = _SENTINEL_DESCRIPTIONS[RESOLVED_SENTINEL]
            for entry in reg:
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str) or not name:
                    raise ValueError("assess_entry_tool: registry entry needs a name")
                if name in options:
                    raise ValueError(f"assess_entry_tool: duplicate tool {name}")
                options[name] = _manifest_line(entry)
            args = validate_json_mapping(
                dict(arguments) if isinstance(arguments, dict) else {}, "<decision_client>: 'arguments'"
            )
            res = validate_json_mapping(dict(result) if isinstance(result, dict) else {}, "<decision_client>: 'result'")
        except Exception as exc:
            logger.warning(
                "toolflow assess_defect sid=- nid=- tool=%s registry=%d err=%s: %s",
                tool,
                len(reg),
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        ok = res.get("ok")
        content = res.get("content") if isinstance(res.get("content"), str) else ""
        error = res.get("error") if isinstance(res.get("error"), str) else ""
        category = res.get("category") if isinstance(res.get("category"), str) else ""
        outcome = f"tool {tool} ok={ok!r} category={category!r} content={content[:1500]!r} error={error[:500]!r}"
        logger.info(
            "toolflow assess_entry sid=- nid=- tool=%s registry=%d options=%d prompt=%r %s ok=%r category=%r",
            tool,
            len(reg),
            len(options),
            _trunc200(text),
            _args_summary(args),
            ok,
            category,
        )
        questions: dict[str, JSONValue] = {
            "assess": {
                "type": "choice",
                "instructions": (
                    f"Assess this single-shot tool result for the entry prompt: {text} "
                    + f"Arguments: {json.dumps(args, default=str)[:2000]} Outcome: {outcome} "
                    + "node_resolved when the result answers the prompt; reasoning_required when no "
                    + "further tool helps; research_required when a full session is needed; otherwise "
                    + f"the one registry tool to run next. Today UTC is {utcnow().date().isoformat()}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week'/'today'/'last quarter' as arg values.",
                ),
                "criteria": validate_json_mapping(options, "<decision_client>: 'criteria'"),
            }
        }
        start = time.perf_counter()
        try:
            decisions, raw, _via = await self._invoke(
                {"prompt": text, "tool": tool, "arguments": args, "outcome": outcome},
                questions,
                {"assess": options},
            )
        except Exception as exc:
            logger.warning(
                "toolflow assess_defect sid=- nid=- tool=%s err=%s: %s",
                tool,
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        d = decisions.get("assess")
        winner = d.get("choice") if isinstance(d, dict) else None
        if not isinstance(winner, str) or winner not in options:
            logger.warning(
                "toolflow assess_defect sid=- nid=- tool=%s winner=%r err=winner_not_in_options %s",
                tool,
                winner,
                _choice_summary(d),
            )
            raise ValueError(f"assess_entry_tool: winner {winner!r} not in options")
        jev_in, jev_out, _ = _jev_usage(raw)
        logger.info(
            "toolflow assess_exit sid=- nid=- tool=%s %s via=%s latency_ms=%.1f in=%s out=%s",
            tool,
            _choice_summary(d),
            _via,
            latency_ms,
            jev_in,
            jev_out,
        )
        return winner

    async def adjudicate(
        self,
        proposal: object,
        node: object,
        *,
        session_id: str,
        job_id: str | None = None,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
    ) -> ToolDecision:
        """JEV adjudicates a reasoner proposal over the same whole-registry options as select_tool; proposal lives in state only, never filters the registry. Needle never selects/chains/judges."""
        reg = list(registry) if registry else _auto_registry()
        node_d = _node_dict(node)
        nid = node_d.get("node_id") or node_d.get("id")
        nid_s = nid if isinstance(nid, str) and nid else "-"
        try:
            options, prompt = _tool_options_prompt(reg, node_d, None, None)
        except Exception as exc:
            logger.warning(
                "toolflow adjudicate_defect sid=%s nid=%s registry=%d err=%s: %s",
                session_id,
                nid_s,
                len(reg),
                type(exc).__name__,
                str(exc)[:200],
            )
            raise
        logger.info(
            "toolflow adjudicate_entry sid=%s nid=%s registry=%d options=%d question=%r",
            session_id,
            nid_s,
            len(reg),
            len(options),
            _trunc200(node_d.get("question")),
        )
        logger.debug(
            "toolflow adjudicate_detail sid=%s nid=%s proposal=%r",
            session_id,
            nid_s,
            _trunc200(proposal),
        )
        questions: dict[str, JSONValue] = {
            "tool_selection": {
                "type": "choice",
                "instructions": prompt,
                "criteria": validate_json_mapping(options, "<decision_client>: 'criteria'"),
            }
        }
        state = {"proposal": proposal, "node": node_d, "objective": node_d.get("question")}
        decisions = await self.decide(
            state,
            questions,
            decision_type="reason_adjudication",
            session_id=session_id,
            node_id=nid if isinstance(nid, str) else None,
            job_id=job_id,
            choice_options={"tool_selection": options},
        )
        return self._to_tool_decision(
            decisions.get("tool_selection"),
            options,
            set_of_registry={o for o in options} - set(_SENTINEL_DESCRIPTIONS),
            sid=session_id,
            nid=nid if isinstance(nid, str) else None,
            event="adjudicate",
        )

    async def assess_result(
        self,
        node: object,
        outcome: object,
        evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
    ) -> dict[str, JSONValue]:
        """Post-tool assessment: relevance noul + evidence-state choice + continue choice."""
        node_d = _node_dict(node)
        nid = node_d.get("node_id") or node_d.get("id")
        question = node_d.get("question") or ""
        questions: dict[str, JSONValue] = {
            "relevance": {"type": "noul", "instructions": f"Do the available facts support resolving: {question}"},
            "evidence_state": {
                "type": "choice",
                "instructions": "What best characterizes the evidence state for this node?",
                "criteria": dict(EVIDENCE_STATE_OPTIONS),
            },
            "continue": {
                "type": "choice",
                "instructions": "What should happen next for this node?",
                "criteria": dict(CONTINUE_OPTIONS),
            },
        }
        state = {"node": node_d, "outcome": _outcome_dict(outcome), "evidence": evidence or []}
        decisions = await self.decide(
            state,
            questions,
            decision_type="result_assessment",
            session_id=session_id,
            node_id=nid if isinstance(nid, str) else None,
            job_id=job_id,
            choice_options={"evidence_state": dict(EVIDENCE_STATE_OPTIONS), "continue": dict(CONTINUE_OPTIONS)},
        )
        relevance = decisions.get("relevance") or {}
        ev_state = decisions.get("evidence_state") or {}
        cont = decisions.get("continue") or {}
        probs: dict[str, JSONValue] = {}
        if isinstance(relevance.get("probability"), (int, float)):
            probs["relevance"] = relevance["probability"]
        if isinstance(ev_state.get("probabilities"), dict):
            probs.update(ev_state["probabilities"])
        if isinstance(cont.get("probabilities"), dict):
            probs.update(cont["probabilities"])
        continuation = cont.get("choice") if isinstance(cont.get("choice"), str) else "continue_research"
        ev_choice = ev_state.get("choice") if isinstance(ev_state.get("choice"), str) else None
        return {
            "probabilities": probs,
            "confidence": cont.get("confidence"),
            "continuation": continuation,
            "continue": continuation,
            "action": continuation,
            "evidence": None,
            "candidate": None,
            "admit": None,
            "evidence_state": ev_choice,
            "decision": ev_choice,
            "relevance": relevance.get("probability"),
        }

    @staticmethod
    def _to_tool_decision(
        decision: object,
        options: Mapping[str, str],
        set_of_registry: set[str],
        *,
        sid: str | None = None,
        nid: str | None = None,
        event: str = "choice",
    ) -> ToolDecision:
        sid_s = sid if isinstance(sid, str) and sid else "-"
        nid_s = nid if isinstance(nid, str) and nid else "-"
        if not isinstance(decision, dict) or decision.get("kind") != "choice":
            logger.warning("toolflow %s_defect sid=%s nid=%s err=malformed_decision", event, sid_s, nid_s)
            raise ValueError("decide: malformed tool_selection decision")
        winner = decision.get("choice")
        probs = decision.get("probabilities")
        conf = decision.get("confidence")
        if not isinstance(winner, str) or not isinstance(probs, dict):
            logger.warning("toolflow %s_defect sid=%s nid=%s err=malformed_decision", event, sid_s, nid_s)
            raise ValueError("decide: malformed tool_selection decision")
        try:
            floats: dict[str, float] = {str(k): float(v) for k, v in probs.items()}
        except Exception:
            logger.warning(
                "toolflow %s_defect sid=%s nid=%s winner=%r err=malformed_probabilities", event, sid_s, nid_s, winner
            )
            raise
        probabilities = validate_json_mapping(floats, "<decision_client>: 'probabilities'")
        confidence = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None
        if winner == REASON_SENTINEL:
            out = ToolDecision(action="reason", probabilities=probabilities, confidence=confidence)
        elif winner == RESOLVED_SENTINEL:
            out = ToolDecision(action="resolved", probabilities=probabilities, confidence=confidence)
        else:
            if winner not in set_of_registry:
                logger.warning(
                    "toolflow %s_defect sid=%s nid=%s winner=%r err=winner_not_in_registry %s",
                    event,
                    sid_s,
                    nid_s,
                    winner,
                    _choice_summary(decision),
                )
                raise ValueError(f"decide: tool_selection winner {winner!r} not in registry")
            winner_prob = floats.get(winner, 0.0)

            def _rank_key(t: str) -> float:
                return floats.get(t, 0.0)

            ranked = sorted(
                (
                    t
                    for t in set_of_registry
                    if floats.get(t, 0.0) >= _PARALLEL_MIN_PROB and floats.get(t, 0.0) >= winner_prob - _PARALLEL_WINDOW
                ),
                key=_rank_key,
                reverse=True,
            )[:_PARALLEL_CAP]
            names = (winner, *(t for t in ranked if t != winner))[:_PARALLEL_CAP]
            out = ToolDecision(
                action="invoke",
                tool_name=winner,
                tool_names=names,
                probabilities=probabilities,
                confidence=confidence,
            )
        out.validate("<decision_client>")
        if out.action in ("reason", "resolved"):
            logger.info(
                "toolflow %s_sentinel sid=%s nid=%s sentinel=%s %s",
                event,
                sid_s,
                nid_s,
                out.action,
                _choice_summary(decision),
            )
        else:
            logger.info(
                "toolflow %s_exit sid=%s nid=%s %s fanout=[%s]",
                event,
                sid_s,
                nid_s,
                _choice_summary(decision),
                ",".join(out.tool_names),
            )
        return out

    def _persist(
        self,
        *,
        decision_type: str,
        session_id: str,
        node_id: str | None,
        job_id: str | None,
        questions: dict[str, JSONValue],
        decisions: dict[str, dict[str, JSONValue]],
        raw: object,
        via: str,
        latency_ms: float,
        started_at: str,
        completed_at: str,
        jev_in: int | None = None,
        jev_out: int | None = None,
        jev_model: str | None = None,
    ) -> None:
        try:
            probabilities: dict[str, JSONValue] = {}
            for qid, decision in decisions.items():
                if not isinstance(decision, dict):
                    continue
                kind = decision.get("kind")
                if kind == "noul" and isinstance(decision.get("probability"), (int, float)):
                    probabilities[qid] = {"probability": decision["probability"]}
                elif kind == "choice" and isinstance(decision.get("probabilities"), dict):
                    probabilities[qid] = dict(decision["probabilities"])
                elif kind == "score":
                    entry: dict[str, JSONValue] = {"score": decision.get("score")}
                    if isinstance(decision.get("probabilities"), dict):
                        entry.update(decision["probabilities"])
                    if decision.get("confidence") is not None:
                        entry["confidence"] = decision["confidence"]
                    probabilities[qid] = entry
                else:
                    probabilities[qid] = dict(decision) if isinstance(decision, dict) else {"result": decision}
            confidence: float | None = None
            if len(decisions) == 1:
                sole = next(iter(decisions.values()))
                raw_conf = sole.get("confidence") if isinstance(sole, dict) else None
                if (
                    isinstance(raw_conf, (int, float))
                    and not isinstance(raw_conf, bool)
                    and 0.0 <= float(raw_conf) <= 1.0
                ):
                    confidence = float(raw_conf)
            record = DecisionRecord(
                decision_id=new_decision_id(),
                session_id=session_id,
                node_id=node_id,
                job_id=job_id,
                decision_type=decision_type,
                candidates=validate_json_mapping(dict(questions), "<decision_client>: 'candidates'"),
                probabilities=validate_json_mapping(probabilities, "<decision_client>: 'probabilities'"),
                selected=validate_json_value(decisions, "<decision_client>: 'selected'"),
                confidence=confidence,
                created_at=utcnow(),
            )
            record.validate("<decision_client>")
        except Exception as exc:  # noqa: BLE001 - persistence never breaks a decision
            logger.debug("jev persist: skipping record build (%s: %s)", type(exc).__name__, exc)
            return
        request: dict[str, JSONValue] = {"state": None, "questions": questions}
        try:
            self._persist_domain(record, request=request, response=raw, latency_ms=latency_ms)
        except Exception as exc:  # noqa: BLE001 - persistence never breaks a decision
            logger.debug("jev persist: domain skipped (%s: %s)", type(exc).__name__, exc)
        try:
            self._persist_runs(
                record,
                request=request,
                response=raw,
                via=via,
                latency_ms=latency_ms,
                started_at=started_at,
                completed_at=completed_at,
                decisions=decisions,
                jev_in=jev_in,
                jev_out=jev_out,
                jev_model=jev_model,
            )
        except Exception as exc:  # noqa: BLE001 - persistence never breaks a decision
            logger.debug("jev persist: runs skipped (%s: %s)", type(exc).__name__, exc)

    def _persist_domain(
        self, record: DecisionRecord, *, request: dict[str, JSONValue], response: object, latency_ms: float
    ) -> None:
        try:
            repo = ResearchRepository(data_root=self._data_root)
            save = getattr(repo, "save_decision", None)
            if callable(save):
                try:
                    save(
                        record,
                        request=request,
                        response=response,
                        provider=self._provider,
                        latency_ms=latency_ms,
                    )
                except TypeError:
                    save(record)
                return
        except Exception:
            pass
        # ponytail: local jev_decisions table until KernelPersistence's
        # save_decision lands; then the duck-typed path above wins.
        path = get_research_db_path(self._data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = record.to_dict()
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jev_decisions ("
                " decision_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, node_id TEXT, job_id TEXT,"
                " decision_type TEXT NOT NULL, candidates TEXT NOT NULL, probabilities TEXT NOT NULL,"
                " selected TEXT NOT NULL, confidence REAL, created_at TEXT NOT NULL,"
                " request TEXT NOT NULL, response TEXT NOT NULL, provider TEXT NOT NULL, latency_ms REAL NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO jev_decisions (decision_id, session_id, node_id, job_id,"
                " decision_type, candidates, probabilities, selected, confidence, created_at,"
                " request, response, provider, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc["decision_id"],
                    doc["session_id"],
                    doc["node_id"],
                    doc["job_id"],
                    doc["decision_type"],
                    json.dumps(doc["candidates"], sort_keys=True),
                    json.dumps(doc["probabilities"], sort_keys=True),
                    json.dumps(doc["selected"], sort_keys=True),
                    doc["confidence"],
                    doc["created_at"],
                    json.dumps(request, sort_keys=True, default=str),
                    json.dumps(response, sort_keys=True, default=str),
                    self._provider,
                    latency_ms,
                ),
            )
            conn.commit()

    def _persist_runs(
        self,
        record: DecisionRecord,
        *,
        request: dict[str, JSONValue],
        response: object,
        via: str,
        latency_ms: float,
        started_at: str,
        completed_at: str,
        decisions: dict[str, dict[str, JSONValue]],
        jev_in: int | None = None,
        jev_out: int | None = None,
        jev_model: str | None = None,
    ) -> None:
        recorder = get_current_recorder()
        if recorder is None or not getattr(recorder, "enabled", True):
            return
        doc = record.to_dict()
        metadata: dict[str, object] = {
            "decision_id": doc["decision_id"],
            "decision_type": doc["decision_type"],
            "session_id": doc["session_id"],
            "node_id": doc["node_id"],
            "job_id": doc["job_id"],
            "provider": self._provider,
            "via": via,
            "latency_ms": latency_ms,
            "request": request,
            "response": response,
            "usage": {"input_tokens": jev_in, "output_tokens": jev_out},
            "model": jev_model,
        }
        recorder.record_event(
            "jev_decision",
            model=self._model,
            result_summary=json.dumps(decisions, sort_keys=True),
            success=True,
            metadata=metadata,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=latency_ms,
        )
        try:
            in_tokens = jev_in if isinstance(jev_in, int) else 0
            out_tokens = jev_out if isinstance(jev_out, int) else 0
            recorder.record_model_call(
                round=0,
                provider=self._provider,
                model=jev_model or self._model,
                started_at=started_at,
                completed_at=completed_at,
                usage={
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": in_tokens + out_tokens,
                },
                tool_call_count=0,
            )
        except Exception as exc:  # noqa: BLE001 - model-call row is best-effort, never breaks a decision
            logger.debug("jev persist: model call skipped (%s: %s)", type(exc).__name__, exc)
