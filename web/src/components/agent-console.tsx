import { useState } from "react";
import type { AgentEvent, Evidence } from "../../../needle-harness/lib/agent/types";
import { AgentEventView } from "./agent-event";
import { SourcePanel } from "./source-panel";

// ponytail: citations are exact Evidence.ids in brackets, never a parsed link scheme.
function splitAnswer(answer: string, known: Map<string, Evidence>): { text: string; citation?: Evidence }[] {
  const entries = [...known.keys()];
  if (entries.length === 0) return [{ text: answer }];
  const parts: { text: string; citation?: Evidence }[] = [];
  const pattern = new RegExp(`\\[(${entries.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\]`, "g");
  let last = 0;
  answer.replace(pattern, (match, id, offset) => {
    if (offset > last) parts.push({ text: answer.slice(last, offset) });
    parts.push({ text: match, citation: known.get(id) });
    last = offset + match.length;
    return match;
  });
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return parts;
}

function evidenceById(events: AgentEvent[]): Map<string, Evidence> {
  // Each evidence event carries the full merged list, so the last one wins.
  const evidence = events.findLast((e) => e.type === "evidence")?.evidence ?? [];
  return new Map(evidence.map((entry) => [entry.id, entry]));
}

type SourceSelection = { evidenceId: string; position: number } | null;

function SelectedSource({
  selected,
  known,
  setSelected,
}: {
  selected: SourceSelection;
  known: Map<string, Evidence>;
  setSelected: React.Dispatch<React.SetStateAction<SourceSelection>>;
}) {
  if (!selected) return null;
  const sources = known.get(selected.evidenceId)?.sources || [];
  if (sources.length === 0) return null;
  const position = selected.position % sources.length;
  return <SourcePanel key={sources[position].id} reference={sources[position]} position={position} count={sources.length} onNavigate={(direction) => setSelected((current) => current && { evidenceId: current.evidenceId, position: (current.position + direction + sources.length) % sources.length })} onClose={() => setSelected(null)} />;
}

// ponytail: selection keeps evidence id + index so follow-up merges refresh the panel instead of a stale snapshot.
export function AgentConsole({ events, reasoning, debug }: { events: AgentEvent[]; reasoning: boolean; debug: boolean }) {
  const [selected, setSelected] = useState<SourceSelection>(null);
  const known = evidenceById(events);
  const sourceControls = [...known.values()].filter((entry) => entry.sources?.length);
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
  // The evidence event stays out of prod rows: citations + source controls read it directly.
  const visible = debug ? events : events.filter((e) => e.type === "agent_start" || e.type === "answer_delta" || e.type === "reasoning_start" || e.type === "done" || e.type === "error" || e.type === "failed");
  const head = visible.filter(
    (e) => e.type !== "answer_delta" && e.type !== "done" && e.type !== "error" && e.type !== "failed" && e.type !== "reasoning_start" && e.type !== "evidence",
  );
  const tail = visible.filter((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const streaming = reasoning && tail.length === 0;
  return (
    <div className="flex-1 space-y-2 overflow-y-auto px-4 py-4 text-sm">
      {head.map((e, i) => (
        <AgentEventView key={i} event={e} />
      ))}
      {streaming && <AgentEventView event={{ type: "reasoning_start", model: "stockbot" }} />}
      {answer && <div className="whitespace-pre-wrap pt-2 text-zinc-100">{splitAnswer(answer, known).map((part, i) => {
        const citation = part.citation;
        return citation?.sources?.length
          ? <button key={i} type="button" onClick={() => setSelected({ evidenceId: citation.id, position: 0 })} aria-label={`Open source for ${citation.id}`} className="rounded border border-blue-800 px-1 text-blue-300 underline decoration-dotted hover:bg-blue-950">{part.text}</button>
          : <span key={i}>{part.text}</span>;
      })}</div>}
      {sourceControls.length > 0 && <div className="flex flex-wrap gap-2 pt-2" role="group" aria-label="Captured sources">
        {sourceControls.map((entry) => (
          <button key={entry.id} type="button" onClick={() => setSelected({ evidenceId: entry.id, position: 0 })} className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400" aria-label={`Open captured sources for ${entry.id}`}>
            Sources: {entry.id}
          </button>
        ))}
      </div>}
      <SelectedSource selected={selected} known={known} setSelected={setSelected} />
      {tail.map((e, i) => (
        <AgentEventView key={`tail-${i}`} event={e} />
      ))}
    </div>
  );
}
