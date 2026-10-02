import json
import os
import sys
import threading
from datetime import datetime, timezone

os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

import needle  # noqa: E402


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
    except Exception as e:
        sys.stderr.write(f"needle server: cannot load tool catalog at {path}: {e}\n")
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
    return datetime.now(timezone.utc).strftime("%a %Y-%m-%d")


def _legacy_system():
    return (
        f"date: {_today()} UTC; locale: en-US; "
        "Route retrieval only: call a tool only with entities/terms from the request or prior results. "
        "SEC questions: prefer find_sec_entities then search_sec_filings then get_sec_document chains. "
        "Return no call when evidence suffices."
    )


def _bound_system():
    today = datetime.now(timezone.utc).date().isoformat()
    return f"date: {_today()} UTC; locale: en-US; Today UTC is {today}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week' or 'today' or 'last quarter' as arg values."


weights = resolve_weights()
kwargs = {
    "tools": TOOLS,
    # Binding args-only rule lives per-request in _arguments_prompt (kernel path); shared system stays legacy until loop.ts cutover removes start/step.
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


def _tool_triggers(tool):
    """Trigger gate: data tools must call; control tools must withhold when ungrounded."""
    # ponytail: universal [".+"] forces a call on any input — proven to
    # fabricate session_id from question text for research_resume
    # ({"session_id": "NVDA revenue last quarter"}). Control/session tools
    # carry caller sid via kernel_worker shortcut; Needle must withhold here.
    if tool.startswith(("research_", "thesis_")):
        return []
    return [".+"]


def _tool_entry(tool, schema):
    """Single-tool binding for one arguments.generate call: grammar admits exactly this tool."""
    params = schema if isinstance(schema, dict) else {}
    full = None
    try:
        with open(_catalog_path()) as f:
            for t in json.load(f):
                if isinstance(t, dict) and t.get("name") == tool:
                    full = t
                    break
    except Exception:
        full = None
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
    triggers = _tool_triggers(tool)
    if triggers:
        entry["triggers"] = triggers
    return entry


def _bound_agent(tool, schema):
    """Fresh Needle bound to exactly the JEV-selected tool (docs: one tool per action)."""
    bound_kwargs = dict(kwargs)
    bound_kwargs["tools"] = [_tool_entry(tool, schema)]
    # ponytail: bound call is args-only — facts-only system. Instructions
    # ("route retrieval only", chain preferences, "no call when evidence
    # suffices") withhold the call on multi-hop objectives (docs: system =
    # facts never instructions). Shared agent keeps legacy system for start/step.
    bound_kwargs["system"] = _bound_system()
    return needle.Needle(**bound_kwargs)


def _decision(r):
    calls = r.get("function_calls") or []
    if r.get("type") == "call" and calls:
        tool = calls[0]["name"]
        args = calls[0].get("arguments") or {}
    else:
        tool = None
        args = {}
    return {
        "tool": tool,
        "arguments": args,
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
        raise ValueError("jev_tool must be a nonempty tool name")
    if needle_tool != jev_tool:
        raise ValueError(f"needle tool mismatch: jev selected {jev_tool!r}, needle emitted {needle_tool!r}")
    return needle_tool


def _arguments_prompt(tool, schema, objective, node, context):
    """Single-shot constrained prompt: exactly one tool, grounded args only."""
    return json.dumps(
        {
            "instruction": (
                f"You must call exactly the tool {tool!r} once with valid grounded arguments, "
                "or return an error (must-call-or-error). Never call another tool, never chain "
                "tools, never judge sufficiency or completion. Ground every argument: "
                "accession_no only from a prior search_sec_filings packet, never invented from "
                "tickers or names; ticker is a stock symbol like AAPL, never FINRA/SEC/org or "
                "query words — when the symbol is unstated pass company_name instead and omit ticker; "
                "dataset must be canonical group/name. as_of only from an explicit YYYY-MM-DD date or relative wording "
                "in the query/objective, never memory or priors: no date wording means omit as_of entirely (latest-available). Today UTC is "
                f"{datetime.now(timezone.utc).date().isoformat()}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week' or 'today' or 'last quarter' as arg values. On insufficient grounding raise/return an error, never fabricate, never stay silent. "
                "For get_sec_document on a revenue-quarter question always include query 'revenue increased'; "
            ),
            "tool": tool,
            "schema": schema,
            "objective": objective,
            "node": node,
            "context": context,
        }
    )


def handle(line):
    # Bound stdin memory: a huge prompt never reaches json/model work.
    if len(line) > MAX_LINE_CHARS:
        return {"id": "?", "error": "bad_request"}
    try:
        req = json.loads(line)
        rid = req["id"]
        action = req.get("action", "")
        if not isinstance(rid, str):
            raise ValueError("bad types")
    except Exception:
        return {"id": "?", "error": "bad_request"}
    try:
        # Ping-only gate: answers after imports load weights; never touches the model.
        if action == "ping":
            return {"id": rid, "ready": True}
        # Legacy non-kernel path: kernel path uses arguments.generate only; JEV owns transitions (start/step stay until loop.ts cutover).
        if action == "start":
            prompt = req["prompt"]
            if not isinstance(prompt, str):
                raise ValueError("bad prompt")
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
                raise ValueError("bad tool")
            bound = _bound_agent(tool, req.get("schema"))
            try:
                objective = req.get("objective")
                prompt = _arguments_prompt(
                    tool,
                    req.get("schema"),
                    objective
                    if isinstance(objective, str) and objective.strip()
                    else json.dumps({"tool": tool, "schema": req.get("schema")}),
                    req.get("node"),
                    req.get("context"),
                )
                r = bound.complete(prompt, max_new_tokens=512)
            finally:
                try:
                    bound.close()
                except Exception:
                    pass
            validate_needle_tool(tool, _decision(r)["tool"])
        else:
            return {"id": rid, "error": "bad_action"}
        out = {"id": rid}
        out.update(_decision(r))
        return out
    except Exception as e:
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
