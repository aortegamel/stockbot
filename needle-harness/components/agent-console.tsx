import type { AgentEvent } from "@/lib/agent/types";
import { AgentEventView } from "./agent-event";

export function AgentConsole({ events, reasoning, debug }: { events: AgentEvent[]; reasoning: boolean; debug: boolean }) {
  if (events.length === 0) {
    return (
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4 text-sm">
        <div className="text-zinc-100">Stockbot — local-first investment research</div>
        <div className="text-zinc-500">
          Hedge-fund-style research infrastructure for individual investors — research and data foundation, not automated trading or portfolio management. I read
          primary sources and show my evidence; deterministic calculations, not model arithmetic.
        </div>
        <div className="space-y-1 text-zinc-500">
          <div>· SEC EDGAR — resolve a company to its CIK, search and list filings (10-K/10-Q/8-K), read bounded document windows, diff risk factors across periods.</div>
          <div>· Fundamentals — EPS, dividends, balance-sheet items, full statements, XBRL facts, obligations and valuation multiples.</div>
          <div>· Ownership &amp; insiders — beneficial stakes, holder changes, Forms 3/4/5 trades, Form 144 planned sales, offerings and dilution.</div>
          <div>· FINRA — short interest and leaderboards, Reg SHO volume, threshold securities, dataset queries.</div>
          <div>· Market context — analyst consensus, S&amp;P 500 weight, macro rates and inflation. No live quotes or option chains in this harness.</div>
          <div>· Web + time — Exa web highlights as research leads only (canonical SEC/FINRA wins on facts); current time for point-in-time queries.</div>
          <div>· Deep research — multi-question sessions with evidence, point-in-time known_at/as_of, raw archiving and provenance.</div>
          <div>· Full Stockbot optionally reads Robinhood portfolio, quotes, and options when BROKER_ENABLED — disabled here. Research only, not investment advice.</div>
        </div>
        <div className="text-zinc-600">Try: “What time is it?” · “NVDA risk factors?” · “AAPL insider sales last quarter?”</div>
      </div>
    );
  }
  const answer = events.filter((e) => e.type === "answer_delta").map((e) => e.text).join("");
  // Prod hides internal work rows; debug (.env STOCKBOT_DEBUG=1) shows all.
  const visible = debug ? events : events.filter((e) => e.type === "agent_start" || e.type === "answer_delta" || e.type === "reasoning_start" || e.type === "done" || e.type === "error" || e.type === "failed");
  const head = visible.filter(
    (e) => e.type !== "answer_delta" && e.type !== "done" && e.type !== "error" && e.type !== "failed" && e.type !== "reasoning_start",
  );
  const tail = visible.filter((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const streaming = reasoning && tail.length === 0;
  return (
    <div className="flex-1 space-y-2 overflow-y-auto px-4 py-4 text-sm">
      {head.map((e, i) => (
        <AgentEventView key={i} event={e} />
      ))}
      {streaming && <AgentEventView event={{ type: "reasoning_start", model: "stockbot" }} />}
      {answer && <div className="whitespace-pre-wrap pt-2 text-zinc-100">{answer}</div>}
      {tail.map((e, i) => (
        <AgentEventView key={`tail-${i}`} event={e} />
      ))}
    </div>
  );
}