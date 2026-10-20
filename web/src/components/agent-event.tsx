import { redactArgs, type AgentEvent, type FailureCategory } from "../../../needle-harness/lib/agent/types";

const LIMIT = "The research reached its work limit before it finished.";
const MODEL = "The model could not produce a checked answer.";
const SOURCE = "A data source failed while answering.";
// Prod failure text: one plain sentence per category. No raw message, stack, or tool name.
const FAILURE_TEXT: Record<FailureCategory, string> = {
  timeout: "The research took too long and stopped.", deadline_exceeded: "The research took too long and stopped.",
  token_budget_exhausted: LIMIT, tool_budget_exhausted: LIMIT, job_budget_exhausted: LIMIT, wave_budget_exhausted: LIMIT,
  parallelism_exceeded: LIMIT, depth_exceeded: LIMIT, context_capacity_exceeded: LIMIT, response_too_large: LIMIT,
  tool_output_limit: LIMIT, duplicate_research_action: LIMIT, research_loop_detected: LIMIT,
  policy_denied: "This request is not allowed.", policy_rejection: "This request is not allowed.",
  tool_error: SOURCE, storage_error: SOURCE, freeze_mismatch: SOURCE,
  model_error: MODEL, model_output_failure: MODEL, provider_error: MODEL, synthesis_failed: MODEL, committee_deadlock: MODEL,
  no_evidence: "No evidence was found for this question.",
};

function FailureView({ text, onRetry }: { text: string; onRetry?: () => void }) {
  return (
    <div className="text-red-400" role="alert">
      ✗ {text}{" "}
      {onRetry && <button type="button" onClick={onRetry} className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400">Try again</button>}
    </div>
  );
}

function conf(c: number | null): string {
  return c === null ? "n/a" : `${Math.round(c * 100)}%`;
}

function NeedleDecisionView({ event }: { event: Extract<AgentEvent, { type: "needle_decision" }> }) {
  return (
    <div>
      <span className="rounded border border-green-900 px-1 text-[10px] text-green-500">NEEDLE/LOCAL</span>{" "}
      <span style={{ color: "#4ade80" }}>
        ● needle / {event.tool ?? "(escalate)"}
        {event.tool ? ` / confidence ${conf(event.confidence)}` : ""}
      </span>
      {event.tool && (
        <div className="pl-4 text-zinc-500">
          ├─ {event.tool}({JSON.stringify(redactArgs(event.arguments))})
        </div>
      )}
    </div>
  );
}

function WorkEventView({ event }: { event: AgentEvent }) {
  switch (event.type) {
    case "needle_decision":
      return <NeedleDecisionView event={event} />;
    case "tool_start":
      return <div className="pl-4 text-zinc-500">├─ {event.tool}…</div>;
    case "tool_result":
      return (
        <div className="pl-4 text-zinc-500">
          ├─ {event.tool} / {event.evidenceId ? `✓ ${event.evidenceId}` : "✗"} — {event.preview}
        </div>
      );
    case "progress": {
      const detail = event.detail !== undefined ? ` ${JSON.stringify(event.detail).slice(0, 160)}` : "";
      return <div className="pl-4 text-zinc-500">├─ {event.stage}{detail}</div>;
    }
    case "tool_failed":
      return <div className="pl-4 text-zinc-500">├─ {event.tool} / ✗ {event.category} — {event.preview}</div>;
  }
}

export function AgentEventView({ event, debug = false, onRetry }: { event: AgentEvent; debug?: boolean; onRetry?: () => void }) {
  switch (event.type) {
    case "agent_start":
      return <div className="text-zinc-100">&gt; {event.prompt}</div>;
    case "reasoning_start":
      return (
        <div>
          <span className="rounded border border-violet-900 px-1 text-[10px] text-violet-400">STOCKBOT</span>{" "}
          <span style={{ color: "#a78bfa" }}>● stockbot / working…</span>
        </div>
      );
    case "answer_delta":
      return <span>{event.text}</span>;
    case "done": {
      const m = event.metrics;
      // Phase timings are debug output (.env STOCKBOT_DEBUG=1); an absent phase never renders.
      const timings = debug ? Object.entries(m.timings ?? {}).flatMap(([k, v]) => (v === undefined ? [] : [`${k} ${Math.round(v)}ms`])) : [];
      return (
        <div className="pt-2 text-xs text-zinc-500">
          <div>── metrics ──</div>
          <div>
            Needle calls: {m.needle.calls} · Tool calls: {m.tools.calls} · Stockbot calls: {m.muse.calls} · Evidence:{" "}
            {m.evidence.count} · Evidence chars: {m.evidence.characters} · Stockbot input: {m.muse.inputTokens ?? "n/a"}{" "}
            · Stockbot output: {m.muse.outputTokens ?? "n/a"}
          </div>
          <div>
            Needle: {Math.round(m.needle.totalMs)}ms · Tools: {Math.round(m.tools.totalMs)}ms · Stockbot:{" "}
            {Math.round(m.muse.totalMs)}ms · Total: {Math.round(m.totalMs)}ms
          </div>
          {timings.length > 0 && <div>Timings: {timings.join(" · ")}</div>}
        </div>
      );
    }
    case "error":
      // Transport-level text (request failed, stalled, ended early): bounded client prose, not a stack.
      return <FailureView text={event.message} onRetry={onRetry} />;
    case "failed":
      return debug ? <div className="text-red-400">✗ {event.category}: {event.message}</div> : <FailureView text={FAILURE_TEXT[event.category] ?? MODEL} onRetry={onRetry} />;
    default:
      return <WorkEventView event={event} />;
  }
}
