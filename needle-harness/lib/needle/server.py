import importlib.util
import json
import logging
import os
import sys
import threading
from datetime import UTC, datetime

os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

import needle

logger = logging.getLogger(__name__)


def _strip_descriptions(node):
    """Drop every `description` key from a tool-parameter schema tree, keep structure."""
    if isinstance(node, dict):
        return {k: _strip_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip_descriptions(v) for v in node]
    return node


def _catalog_path():
    """Catalog file: NEEDLE_CATALOG override, else the harness-generated file."""
    env = os.environ.get("NEEDLE_CATALOG")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", ".needle-catalog.json"))


def load_catalog():
    path = _catalog_path()
    try:
        with open(path) as f:
            tools = json.load(f)
    except Exception:
        logger.exception("needle server: cannot load tool catalog at %s", path)
        sys.exit(1)
    if not isinstance(tools, list) or not tools:
        sys.stderr.write(f"needle server: tool catalog at {path} is empty or invalid\n")
        sys.exit(1)
    # ponytail: full describe text (~58KB catalog) exceeds the Needle init
    # budget (needle_init code -1); truncated top-level descriptions ground
    # routing (name-only misroutes, e.g. clock->insider), stripped params fit.
    # Init budget is tight (~29KB slim fails, ~28KB passes): drop parameter
    # properties at init — the bound per-tool call carries full params.
    slim = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        raw_desc = tool.get("description")
        desc = raw_desc if isinstance(raw_desc, str) and raw_desc.strip() else tool["name"]
        slim.append({"name": tool["name"], "description": desc[:150], "parameters": {"type": "object"}})
    if not slim:
        sys.stderr.write(f"needle server: tool catalog at {path} is empty or invalid\n")
        sys.exit(1)
    return slim


TOOLS = load_catalog()


def _grounding_formatter():
    """Shared grounded formatter, loaded directly so the worker avoids app imports."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "..", "app", "research", "grounding.py"))
    try:
        spec = importlib.util.spec_from_file_location("needle_grounding", path)
        if spec is None or spec.loader is None:
            raise ValueError("missing loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        formatter = module.format_grounded_block
    except Exception:
        logger.exception("needle server: cannot load grounded formatter at %s", path)
        sys.exit(1)
    if not callable(formatter):
        sys.stderr.write(f"needle server: grounded formatter at {path} is invalid\n")
        sys.exit(1)
    return formatter


format_grounded_block = _grounding_formatter()


def resolve_weights():
    # Mirror of needle-harness/scripts/needle.ts:44-48: relative pins resolve
    # against HARNESS_DIR, absolute pins pass through.
    here = os.path.dirname(os.path.abspath(__file__))
    harness = os.path.normpath(os.path.join(here, "..", ".."))
    env = os.environ.get("NEEDLE_WEIGHTS")
    if env:
        if os.path.isabs(env):
            return env
        return os.path.normpath(os.path.join(harness, env))
    root_blob = os.path.normpath(os.path.join(harness, "..", "needle3.cact"))
    if os.path.exists(root_blob):
        return root_blob
    return None


def _today():
    """UTC date fact, computed per request so long-lived servers never go stale."""
    return datetime.now(UTC).strftime("%a %Y-%m-%d")


def _legacy_system():
    return (
        f"date: {_today()} UTC; locale: en-US; "
        "Route retrieval only: call a tool only with entities/terms from the request or prior results. "
        "SEC questions: prefer find_sec_entities then search_sec_filings then get_sec_document chains. "
        "Return no call when evidence suffices."
    )


def _bound_system():
    today = datetime.now(UTC).date().isoformat()
    return f"date: {_today()} UTC; locale: en-US; Today UTC is {today}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week' or 'today' or 'last quarter' as arg values."


weights = resolve_weights()
kwargs = {
    "tools": TOOLS,
    # Kernel arguments path binds its own facts-only system per request (_arguments_facts); shared system stays legacy until loop.ts cutover removes start/step.
    "system": _legacy_system(),
    "buffer_size": 65536,
}
if weights is not None:
    kwargs["weights"] = weights
agent = needle.Needle(**kwargs)

# Legacy start/step share the module-global agent; reset/complete are not
# reentrant, so serialize them. Oversize lines are rejected in handle().
_agent_lock = threading.Lock()
MAX_LINE_CHARS = 1_000_000


def _tool_entry(tool, schema):
    """Single-tool binding for one arguments.generate call: original full metadata, no triggers."""
    params = schema if isinstance(schema, dict) else {}
    full = None
    with open(_catalog_path()) as f:
        for t in json.load(f):
            if isinstance(t, dict) and t.get("name") == tool:
                full = t
                break
    full = full if isinstance(full, dict) else {}
    raw_desc = full.get("description")
    desc = raw_desc if isinstance(raw_desc, str) and raw_desc.strip() else tool
    full_params = full.get("parameters")
    # ponytail: bound call is small (one tool); keep full params the kernel
    # passes in, else fall back to the catalog entry. Legacy TOOLS slim stays
    # stripped for the shared start/step agent init budget.
    bound_params = params or (full_params if isinstance(full_params, dict) else {})
    entry = {
        "name": tool,
        "description": desc,
        "parameters": bound_params if bound_params else {"type": "object"},
    }
    return entry


# Wrapper instructions, not facts: the SDK system carries facts only (docs: "facts, never instructions").
_INSTRUCTION_KEYS = frozenset({"grounding_hint"})


def _fact(value):
    """One fact value verbatim: strings raw, structures as full JSON (never clipped)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


_GROUNDED_FIELDS = frozenset({"content", "content_hash", "evidence_id", "provenance"})


def _evidence_fact(index, record):
    """One admitted record: shared block first, then every remaining field as JSON."""
    extras = {k: v for k, v in record.items() if k not in _GROUNDED_FIELDS}
    text = format_grounded_block(record)
    if extras:
        text += f" evidence[{index}] extra: {json.dumps(extras, ensure_ascii=False)}"
    return text


def _evidence_facts(records):
    """Admitted records in order; malformed or observation entries keep current JSON."""
    facts = []
    for index, record in enumerate(records):
        if isinstance(record, dict) and isinstance(record.get("evidence_id"), str) and record["evidence_id"]:
            facts.append(_evidence_fact(index, record))
        else:
            facts.append(_fact(record))
    return facts


def _context_facts(key, value):
    """One context entry: evidence lists reuse the grounded renderer, others use JSON."""
    if key == "evidence" and isinstance(value, list):
        return _evidence_facts(value)
    return [f"{key}: {_fact(value)}"]


def _arguments_facts(objective, node, context):
    """SDK system facts: UTC date, objective, node metadata, context (evidence, attempts, temporal anchors)."""
    facts = [f"date: {datetime.now(UTC).strftime('%Y-%m-%d %a %H:%M')} UTC", "locale: en-US"]
    if isinstance(objective, str) and objective.strip():
        facts.append(f"objective: {objective}")
    if isinstance(node, dict):
        # The question is the request text itself, not a fact.
        facts.extend(f"node {k}: {_fact(v)}" for k, v in node.items() if k != "question")
    if isinstance(context, dict):
        for k, v in context.items():
            if k not in _INSTRUCTION_KEYS:
                facts.extend(_context_facts(k, v))
    return "; ".join(facts)


def _request_text(node, objective):
    """Raw node question, else raw objective; no real request text fails."""
    question = node.get("question") if isinstance(node, dict) else None
    for text in (question, objective):
        if isinstance(text, str) and text.strip():
            return text
    raise ValueError("arguments.generate needs a node question or objective")


def _bound_agent(tool, schema, facts):
    """Fresh Needle bound to exactly the JEV-selected tool (docs: one tool per action)."""
    bound_kwargs = dict(kwargs)
    bound_kwargs["tools"] = [_tool_entry(tool, schema)]
    bound_kwargs["system"] = facts
    return needle.Needle(**bound_kwargs)


def _extract_agent(record, system=None):
    """Fresh Needle bound to exactly one record shape (extraction is one-tool calling)."""
    if not isinstance(record, dict):
        raise TypeError("bad record")
    if not isinstance(record.get("name"), str) or not record["name"]:
        raise ValueError("bad record")
    bound_kwargs = dict(kwargs)
    bound_kwargs["tools"] = [record]
    if isinstance(system, str) and system.strip():
        bound_kwargs["system"] = system
    else:
        bound_kwargs["system"] = _bound_system()
    return needle.Needle(**bound_kwargs)


def _grounding_flags(r):
    """SDK grounding flags (ungrounded fields, negated request), sorted; empty list when clean."""
    validation = r.get("validation") or {}
    flagged = [str(f) for f in validation.get("ungrounded") or []]
    if validation.get("negation"):
        flagged.append("negated request")
    return sorted(flagged)


def _extract_decision(record_name, r, strict=True):
    """First call (or withheld call) as extracted fields; null record when nothing matched."""
    calls = r.get("function_calls") or []
    withheld = False
    if not calls:
        calls = r.get("suppressed_calls") or []
        withheld = bool(calls)
    if not calls:
        return {
            "record": None,
            "fields": {},
            "withheld": False,
            "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning") or "",
        }
    call = calls[0] if isinstance(calls[0], dict) else {}
    name = call.get("name")
    if name != record_name:
        raise ValueError(f"extract record mismatch: declared {record_name!r}, engine emitted {name!r}")
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        raise TypeError("extract fields must be a mapping")
    flagged = _grounding_flags(r) if strict else []
    if flagged:
        raise ValueError(f"extraction returned values not grounded in the input: {', '.join(flagged)}")
    return {
        "record": record_name,
        "fields": args,
        "withheld": withheld,
        "confidence": r.get("confidence"),
        "reasoning": r.get("reasoning") or "",
    }


def _noncall_detail(r):
    """One-line detail for a Needle non-call: type, suppressed names, confidence, truncated reasoning."""
    rtype = r.get("type")
    suppressed = r.get("suppressed_calls") or []
    names = [c.get("name") for c in suppressed if isinstance(c, dict)][:5]
    conf = r.get("confidence")
    reasoning = " ".join(str(r.get("reasoning") or "").split())[:200]
    return f"type={rtype!r} suppressed={names!r} confidence={conf!r} reasoning={reasoning!r}"


def _decision(r, want=None):
    calls = r.get("function_calls") or []
    withheld = False
    if not calls:
        calls = r.get("suppressed_calls") or []
        withheld = bool(calls)
    if (r.get("type") == "call" and calls) or (
        withheld and (calls[0].get("name") if isinstance(calls[0], dict) else None) == want
    ):
        call = calls[0] if isinstance(calls[0], dict) else {}
        return {
            "tool": call.get("name"),
            "arguments": call.get("arguments") or {},
            "withheld": withheld,
            "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning") or "",
        }
    return {
        "tool": None,
        "arguments": {},
        "withheld": False,
        "confidence": r.get("confidence"),
        "reasoning": r.get("reasoning") or "",
    }


def validate_needle_tool(jev_tool, needle_tool):
    """Runtime gate: Needle output must invoke the exact JEV-selected tool.

    JEV owns tool selection; Needle never selects, chains, or judges
    sufficiency. A mismatch (including None/escalation) raises — the caller
    treats it as retryable and returns to JEV with the full registry again.
    Never silently substitute.
    """
    if not isinstance(jev_tool, str) or not jev_tool:
        raise TypeError("jev_tool must be a nonempty tool name")
    if needle_tool != jev_tool:
        raise ValueError(f"needle tool mismatch: jev selected {jev_tool!r}, needle emitted {needle_tool!r}")
    return needle_tool


def handle(line):
    # Bound stdin memory: a huge prompt never reaches json/model work.
    if len(line) > MAX_LINE_CHARS:
        return {"id": "?", "error": "bad_request"}
    try:
        req = json.loads(line)
        rid = req["id"]
        action = req.get("action", "")
        if not isinstance(rid, str):
            raise TypeError("bad types")
    except Exception:
        logger.exception("needle server: bad request")
        return {"id": "?", "error": "bad_request"}
    try:
        # Ping-only gate: answers after imports load weights; never touches the model.
        if action == "ping":
            return {"id": rid, "ready": True}
        # Legacy non-kernel path: kernel path uses arguments.generate only; JEV owns transitions (start/step stay until loop.ts cutover).
        if action == "start":
            prompt = req["prompt"]
            if not isinstance(prompt, str):
                raise TypeError("bad prompt")
            with _agent_lock:
                agent.reset()
                r = agent.complete(prompt, max_new_tokens=256)
        elif action == "step":
            with _agent_lock:
                r = agent.complete(json.dumps(req.get("result")), max_new_tokens=256)
        elif action == "arguments.generate":
            # Narrow execution worker: exactly one JEV-selected tool. No
            # registry inspection, no tool choice, no chaining, no
            # sufficiency judgment. start/step (old Needle-owned loop) intact.
            tool = req.get("tool")
            if not isinstance(tool, str) or not tool:
                raise TypeError("bad tool")
            import time as _t

            _gen_t0 = _t.perf_counter()
            _init_ms = 0.0
            node = req.get("node")
            objective = req.get("objective")
            text = _request_text(node, objective)
            bound = _bound_agent(tool, req.get("schema"), _arguments_facts(objective, node, req.get("context")))
            _init_ms = (_t.perf_counter() - _gen_t0) * 1000.0
            try:
                r = bound.complete(text, max_new_tokens=512)
            finally:
                bound.close()
            _gen_ms = (_t.perf_counter() - _gen_t0) * 1000.0
            sys.stderr.write(f"needle timing tool={tool} init_ms={_init_ms:.1f} generate_ms={_gen_ms:.1f}\n")
            sys.stderr.flush()
            if r.get("type") == "error" or r.get("success") is False or r.get("error"):
                raise ValueError(f"needle engine failed for {tool!r}: {r.get('error') or r.get('type')}")
            decision = _decision(r, tool)
            try:
                validate_needle_tool(tool, decision["tool"])
            except TypeError:
                raise
            except ValueError:
                if decision["tool"] is None:
                    raise ValueError(
                        f"needle tool mismatch: jev selected {tool!r}, needle emitted None ({_noncall_detail(r)})"
                    ) from None
                raise
            flagged = _grounding_flags(r)
            if flagged:
                raise ValueError(f"needle arguments for {tool!r} not grounded in the request: {', '.join(flagged)}")
        elif action == "extract":
            # Structured extraction: one record shape in, typed fields out.
            # Never hands the full registry — the grammar admits exactly this record.
            record = req.get("record")
            passage = req.get("passage")
            if not isinstance(record, dict) or not record.get("name"):
                raise TypeError("bad record")
            if not isinstance(passage, str) or not passage.strip():
                raise TypeError("bad passage")
            max_tokens = req.get("max_new_tokens", 512)
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 4096:
                raise TypeError("bad max_new_tokens")
            strict = req.get("strict", True)
            strict = strict if isinstance(strict, bool) else True
            bound = _extract_agent(record, req.get("system"))
            try:
                r = bound.complete(passage, max_new_tokens=max_tokens)
            finally:
                bound.close()
            out = {"id": rid}
            out.update(_extract_decision(str(record["name"]), r, strict))
            return out
        elif action == "embed":
            # Retrieval embedding off the shared agent; serial like start/step.
            text = req.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise TypeError("bad text")
            with _agent_lock:
                vec = agent.embed(text)
            if not isinstance(vec, list):
                raise TypeError("bad embedding")
            if not vec or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vec):
                raise ValueError("bad embedding")
            return {"id": rid, "embedding": vec}
        else:
            return {"id": rid, "error": "bad_action"}
        out = {"id": rid}
        out.update(decision if action == "arguments.generate" else _decision(r))
        return out
    except Exception as e:
        logger.exception("needle server: request failed id=%s", rid)
        return {"id": rid, "error": str(e)}


def _read_line():
    """Next stdin line, capped at MAX_LINE_CHARS. Returns (line, overlong)."""
    # Chunked reads so one huge prompt cannot balloon memory before handle()
    # ever sees it; the excess is drained and answered once as bad_request.
    parts = []
    total = 0
    while True:
        chunk = sys.stdin.readline(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_LINE_CHARS + 1:  # +1: trailing newline is not content
            while not chunk.endswith("\n"):
                chunk = sys.stdin.readline(65536)
                if not chunk:
                    break
            return None, True
        parts.append(chunk)
        if chunk.endswith("\n"):
            break
    return "".join(parts), False


def main():
    while True:
        line, overlong = _read_line()
        if overlong:
            sys.stdout.write(json.dumps({"id": "?", "error": "bad_request"}) + "\n")
            sys.stdout.flush()
            continue
        if not line:
            break  # EOF
        line = line.strip()
        if not line:
            continue
        sys.stdout.write(json.dumps(handle(line)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
