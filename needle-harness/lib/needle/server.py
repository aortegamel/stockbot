import json
import logging
import math
import os
import re
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


def _slim_catalog(tools):
    """Slim catalog entries for the shared init budget; unusable entries drop out."""
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
    return slim


def load_catalog():
    path = _catalog_path()
    try:
        with open(path) as f:
            tools = json.load(f)
    except Exception:
        logger.exception("needle server: cannot load tool catalog at %s", path)
        sys.exit(1)
    slim = _slim_catalog(tools) if isinstance(tools, list) else []
    if not slim:
        sys.stderr.write(f"needle server: tool catalog at {path} is empty or invalid\n")
        sys.exit(1)
    return slim


TOOLS = load_catalog()


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
    return f"date: {_today()} UTC; locale: en-US"


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


def _catalog_entry(tool):
    """Find the original catalog metadata for one bound tool."""
    with open(_catalog_path()) as f:
        for entry in json.load(f):
            if isinstance(entry, dict) and entry.get("name") == tool:
                return entry
    return {}


def _tool_entry(tool, schema):
    """Single-tool binding for one arguments.generate call: original full metadata, no triggers."""
    params = schema if isinstance(schema, dict) else {}
    full = _catalog_entry(tool)
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


_NODE_KEYS = ("node_id", "id", "type")
_SCOPE_KEYS = ("mode", "as_of", "start", "end")


def _scope_facts(scope):
    """Persisted temporal window without generated wording: string window fields only."""
    if not isinstance(scope, dict):
        return []
    facts = []
    for key in _SCOPE_KEYS:
        value = scope.get(key)
        if isinstance(value, str):
            facts.append(f"temporal_scope {key}: {value}")
    return facts


def _fact(value):
    """Keep one explicit execution value without truncation."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _scalar(value):
    """JSON scalar without non-finite floats, so NaN/Infinity never ground arguments."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _argument_value(value):
    """Admit scalars and scalar lists only; records and nested structures never bind."""
    return _scalar(value) or (isinstance(value, list) and all(_scalar(item) for item in value))


def _grounded_argument_facts(values, schema):
    """Explicit schema properties only; malformed, off-schema, or nested values never bind."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(values, dict) or not isinstance(properties, dict):
        return []
    facts = []
    for key, value in values.items():
        if key in properties and _argument_value(value):
            facts.append(f"argument {key}: {_fact(value)}")
    return facts


def _default_facts(schema, explicit):
    """Schema-declared `default` values only; an explicit argument for the key wins."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return []
    return [
        f"default {key}: {_fact(spec['default'])}"
        for key, spec in properties.items()
        if isinstance(spec, dict) and "default" in spec and key not in explicit and _argument_value(spec["default"])
    ]


_COUNT_WORDS = {
    word: n
    for n, word in enumerate(
        (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ),
        1,
    )
} | {"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
# Number words that are never a supported count on their own; a request using them gets a digits instruction.
_UNSUPPORTED_WORDS = ("sixty", "seventy", "eighty", "ninety", "thousand", "million", "dozen")
_NUMBER_WORD = rf"(?:{'|'.join((*_COUNT_WORDS, *_UNSUPPORTED_WORDS))})\b"
# Plain digits are read whole. Temporal phrases exclude year-shaped tokens separately below.
_COUNT = rf"(\d+|{'|'.join(_COUNT_WORDS)})(?![\w./:%-]|,\d)(?!\s*(?:%|percent\b|per\s+cent\b))"
# Counts the matcher never reads partially: separators or decimals (1,000; 2.5), compounds (one hundred,
# thirty five, twenty-five), and unsupported number words.
_BAD_COUNT = (
    rf"(\d+(?:[.,]\d+)+|{_NUMBER_WORD}(?:[\s-]+(?:and[\s-]+)?{_NUMBER_WORD})+|(?:{'|'.join(_UNSUPPORTED_WORDS)})\b)"
)
_RECORDS = (
    r"(?:filings?|records?|results?|rows?|items?|entry|entries|hits?|matches|candidates?|forms?|reports?|"
    r"documents?|transactions?|trades?|events?|sales?|notices?|changes?|offerings?|holders?|owners?|"
    r"companies|entities|stocks?|tickers?|names?|observations?|publications?|articles?|sources?)\b"
)
# Up to three qualifier words (10-K, insider, SEC) may sit between the count and the record noun. A time unit
# never does, so "last 3 years of filings" is a window, not a count. A number word never does, so the rest of
# a compound count is never skipped as a qualifier.
_QUALIFIERS = (
    rf"(?:(?!(?:days?|weeks?|months?|quarters?|years?|hours?|minutes?|sessions?)\b|{_NUMBER_WORD})[\w/&.-]+\s+){{0,3}}?"
)


def _count_phrases(count):
    """Recognized count phrases with `count` (one capture group) in the number position."""
    return re.compile(
        rf"\b(?:latest|last|most\s+recent|newest|recent|past|previous)\s+"
        rf"(?!(?:19|20)\d\d\b){count}\s+{_QUALIFIERS}{_RECORDS}"
        rf"|\b(?:top|first|up\s+to|at\s+most|no\s+more\s+than|max(?:imum)?(?:\s+of)?)\s+"
        rf"{count}\s+{_QUALIFIERS}{_RECORDS}"
        rf"|\b(?!(?:19|20)\d\d\b){count}\s+(?:latest|most\s+recent|newest|recent|last)\s+{_QUALIFIERS}{_RECORDS}"
        rf"|\blimit(?:\s*[:=]\s*|\s+(?:of\s+|to\s+)?){count}(?!\s*(?:%|percent\b|per\s+cent\b))",
        re.IGNORECASE,
    )


_COUNT_RE = _count_phrases(_COUNT)
_BAD_COUNT_RE = _count_phrases(_BAD_COUNT)


def _explicit_counts(text):
    """Distinct counts named by recognized count phrases (latest 5 filings, up to 37 records, limit 10).

    A count phrase whose number is a compound, separated, decimal, or unsupported word fails instead of
    yielding a smaller count or leaving the default in place.
    """
    text = text if isinstance(text, str) else ""
    bad = _BAD_COUNT_RE.search(text)
    if bad is not None:
        raw = next(group for group in bad.groups() if group)
        raise ValueError(f"unsupported count {raw!r} in the request; write the count as plain digits, e.g. 100")
    counts = set()
    for match in _COUNT_RE.finditer(text):
        raw = next(group for group in match.groups() if group)
        counts.add(int(raw) if raw.isdigit() else _COUNT_WORDS[raw.lower()])
    return counts


def _requested_limit(objective, node, context, schema):
    """One requested limit from a structured grounded limit or the original request text; None when absent.

    The original request (objective) is the authority; the node question stands in only without one.
    Distinct counts fail rather than choosing one.
    """
    properties = schema.get("properties") if isinstance(schema, dict) else None
    # get_sec_document's limit counts characters, not records. Its max_chars alias identifies that schema.
    if not isinstance(properties, dict) or "limit" not in properties or "max_chars" in properties:
        return None
    counts = _limit_counts(objective, node, context)
    if len(counts) > 1:
        raise ValueError(f"conflicting explicit counts {sorted(counts)} in the request; ask for one count")
    return next(iter(counts), None)


def _limit_counts(objective, node, context):
    """Collect request counts and the explicit structured limit before conflict checks."""
    text = objective if isinstance(objective, str) and objective.strip() else _request_text(node, objective)
    counts = _explicit_counts(text)
    grounded = context.get("grounded_arguments") if isinstance(context, dict) else None
    structured = grounded.get("limit") if isinstance(grounded, dict) else None
    if isinstance(structured, int) and not isinstance(structured, bool):
        counts.add(structured)
    return counts


def _check_limit(tool, arguments, limit):
    """A requested limit must reach execution exactly; a default or other number never replaces it."""
    value = arguments.get("limit") if isinstance(arguments, dict) else None
    if limit is not None and (isinstance(value, bool) or value != limit):
        raise ValueError(f"needle arguments for {tool!r} set limit {value!r}; the request asks for {limit}")


def _metadata_facts(node):
    """Keep node metadata before execution context and its temporal scope."""
    facts = []
    if isinstance(node, dict):
        for key in _NODE_KEYS:
            value = node.get(key)
            if isinstance(value, str):
                facts.append(f"node {key}: {value}")
    return facts


def _context_facts(context):
    """Keep explicit context strings and the persisted temporal window."""
    facts = []
    if isinstance(context, dict):
        for key in ("as_of", "today_utc", "ticker"):
            value = context.get(key)
            if isinstance(value, str):
                facts.append(f"{key}: {value}")
        facts.extend(_scope_facts(context.get("temporal_scope")))
    return facts


def _arguments_facts(objective, node, context, schema=None, limit=None):
    """Bind request facts, explicit arguments, and declared schema defaults. Never bind research records or feedback."""
    facts = [f"date: {datetime.now(UTC).strftime('%Y-%m-%d %a %H:%M')} UTC", "locale: en-US"]
    if isinstance(objective, str) and objective.strip():
        facts.append(f"objective: {objective}")
    facts.extend(_metadata_facts(node))
    grounded = context.get("grounded_arguments") if isinstance(context, dict) else None
    explicit = dict(grounded) if isinstance(grounded, dict) else {}
    if limit is not None:
        explicit["limit"] = limit
    facts.extend(_context_facts(context))
    facts.extend(_grounded_argument_facts(explicit, schema))
    facts.extend(_default_facts(schema, explicit))
    return "; ".join(facts)


def _request_text(node, objective):
    """Raw node question, else raw objective; no real request text fails."""
    question = node.get("question") if isinstance(node, dict) else None
    for text in (question, objective):
        if isinstance(text, str) and text.strip():
            return text
    raise ValueError("arguments.generate needs a node question or objective")


def _generation_text(text, objective, context, schema, limit):
    """Expose execution handles in user text. Form numbers never supply a result count."""
    properties = schema.get("properties", {})
    grounded = context.get("grounded_arguments", {}) if isinstance(context, dict) else {}
    expected = {}
    if "accession_no" in schema.get("required", []):
        accession = grounded.get("accession_no") if isinstance(grounded, dict) else None
        if accession is None:
            match = re.search(r"\b\d{10}-?\d{2}-?\d{6}\b", f"{objective or ''} {text}")
            accession = match.group(0) if match else None
        if not isinstance(accession, str) or not re.fullmatch(r"\d{10}-?\d{2}-?\d{6}", accession):
            raise ValueError("arguments.generate needs a grounded accession_no from a successful filing listing")
        expected["accession_no"] = accession
    if "forms" in properties:
        forms = sorted(
            {
                form.upper()
                for form in re.findall(r"\b(?:10-[QK]|8-K)(?:/A)?\b", f"{objective or ''} {text}", re.IGNORECASE)
            }
        )
        if forms:
            expected["forms"] = forms
            default = properties.get("limit", {}).get("default")
            if limit is not None or isinstance(default, int):
                expected["limit"] = limit if limit is not None else default
    for key, value in expected.items():
        rendered = ", ".join(value) if isinstance(value, list) else str(value)
        text += f"\n{key}: {rendered}"
    return text, expected


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


def _response_calls(r):
    """Prefer actual calls. Use suppressed calls only when actual calls are absent."""
    calls = r.get("function_calls") or []
    if calls:
        return calls, False
    calls = r.get("suppressed_calls") or []
    return calls, bool(calls)


def _first_call(calls):
    """First call as a mapping; a malformed entry becomes an empty mapping."""
    return calls[0] if calls and isinstance(calls[0], dict) else {}


def _extract_decision(record_name, r, strict=True):
    """First call (or withheld call) as extracted fields; null record when nothing matched."""
    calls, withheld = _response_calls(r)
    if not calls:
        return {
            "record": None,
            "fields": {},
            "withheld": False,
            "confidence": r.get("confidence"),
            "reasoning": r.get("reasoning") or "",
        }
    call = _first_call(calls)
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
    calls, withheld = _response_calls(r)
    call = _first_call(calls)
    name = call.get("name")
    if (r.get("type") == "call" and calls) or (withheld and name == want):
        return {
            "tool": name,
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


def _handle_start(req):
    """Run the legacy initial prompt on the serialized shared agent."""
    prompt = req["prompt"]
    if not isinstance(prompt, str):
        raise TypeError("bad prompt")
    with _agent_lock:
        agent.reset()
        r = agent.complete(prompt, max_new_tokens=256)
    return _decision(r)


def _handle_step(req):
    """Run a legacy result on the serialized shared agent."""
    with _agent_lock:
        r = agent.complete(json.dumps(req.get("result")), max_new_tokens=256)
    return _decision(r)


def _generated_decision(tool, r, limit):
    """Validate the engine result before arguments cross the execution boundary."""
    if r.get("type") == "error" or r.get("success") is False or r.get("error"):
        raise ValueError(f"needle engine failed for {tool!r}: {r.get('error') or r.get('type')}")
    decision = _decision(r, tool)
    try:
        validate_needle_tool(tool, decision["tool"])
    except ValueError:
        if decision["tool"] is None:
            raise ValueError(
                f"needle tool mismatch: jev selected {tool!r}, needle emitted None ({_noncall_detail(r)})"
            ) from None
        raise
    flagged = _grounding_flags(r)
    if flagged:
        raise ValueError(f"needle arguments for {tool!r} not grounded in the request: {', '.join(flagged)}")
    _check_limit(tool, decision["arguments"], limit)
    return decision


def _handle_generate(req):
    """Generate arguments for exactly one JEV-selected tool.

    Narrow execution worker: no registry inspection, no tool choice, no
    chaining, no sufficiency judgment. Legacy start/step (the old Needle-owned
    loop) stay until the loop.ts cutover.
    """
    tool = req.get("tool")
    if not isinstance(tool, str) or not tool:
        raise TypeError("bad tool")
    import time as _t

    _gen_t0 = _t.perf_counter()
    _init_ms = 0.0
    node = req.get("node")
    objective = req.get("objective")
    text = _request_text(node, objective)
    entry = _tool_entry(tool, req.get("schema"))
    schema = entry["parameters"]
    context = req.get("context")
    # Conflicting counts fail here, before the engine runs.
    limit = _requested_limit(objective, node, context, schema)
    text, expected = _generation_text(text, objective, context, schema, limit)
    bound = _bound_agent(tool, schema, _arguments_facts(objective, node, context, schema, limit))
    _init_ms = (_t.perf_counter() - _gen_t0) * 1000.0
    try:
        r = bound.complete(text, max_new_tokens=512)
    finally:
        bound.close()
    _gen_ms = (_t.perf_counter() - _gen_t0) * 1000.0
    sys.stderr.write(f"needle timing tool={tool} init_ms={_init_ms:.1f} generate_ms={_gen_ms:.1f}\n")
    sys.stderr.flush()
    decision = _generated_decision(tool, r, limit)
    for key, value in expected.items():
        if decision["arguments"].get(key) != value:
            raise ValueError(f"needle arguments for {tool!r} changed grounded {key}: expected {value!r}")
    return decision


def _handle_extract(req):
    """Decode one record shape, never the full registry."""
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
    return _extract_decision(str(record["name"]), r, strict)


def _handle_embed(req):
    """Retrieve one vector from the serialized shared agent."""
    text = req.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise TypeError("bad text")
    with _agent_lock:
        vec = agent.embed(text)
    if not isinstance(vec, list):
        raise TypeError("bad embedding")
    if not vec or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vec):
        raise ValueError("bad embedding")
    return {"embedding": vec}


# One handler per request action. Handlers return the response body without
# the id, which handle() adds. start/step are the legacy non-kernel path: the
# kernel path uses arguments.generate only and JEV owns transitions.
_HANDLERS = {
    "start": _handle_start,
    "step": _handle_step,
    "arguments.generate": _handle_generate,
    "extract": _handle_extract,
    "embed": _handle_embed,
}


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
        # Ping answers after imports load weights. It never touches the model.
        if action == "ping":
            return {"id": rid, "ready": True}
        handler = _HANDLERS.get(action) if isinstance(action, str) else None
        if handler is None:
            return {"id": rid, "error": "bad_action"}
        out = {"id": rid}
        out.update(handler(req))
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
