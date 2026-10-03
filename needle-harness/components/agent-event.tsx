import { redactArgs, type AgentEvent } from "@/lib/agent/types";

function conf(c: number | null): string {
  return c === null ? "n/a" : `${Math.round(c * 100)}%`;
}

export function AgentEventView({ event }: { event: AgentEvent }) {
  switch (event.type) {
    case "agent_start":
      return <div className="text-zinc-100">&gt; {event.prompt}</div>;
    case "needle_decision":
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
    case "tool_start":
      return <div className="pl-4 text-zinc-500">├─ {event.tool}…</div>;
    case "tool_result":
      return (
        <div className="pl-4 text-zinc-500">
          ├─ {event.tool} / {event.evidenceId ? `✓ ${event.evidenceId}` : "✗"} — {event.preview}
        </div>
      );
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
        </div>
      );
    }
    case "error":
      return <div className="text-red-400">✗ {event.message}</div>;
    case "tool_failed":
      return <div className="pl-4 text-zinc-500">├─ {event.tool} / ✗ {event.category} — {event.preview}</div>;
    case "failed":
      return <div className="text-red-400">✗ {event.category}: {event.message}</div>;
  }
}
