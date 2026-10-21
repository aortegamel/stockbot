import { useEffect, useRef, useState } from "react";
import { FINAL_PERSONAS, type AgentEvent, type Evidence, type FinalClaim, type SourceReference } from "../../../needle-harness/lib/agent/types";
import { fetchSourceDocument } from "../lib/research-client";
import { routeHash } from "../lib/route";
import { sourceLabel } from "../lib/source-viewer";
import { ActionCards, TrioCard, type WidgetOutcomes } from "./action-cards";
import { AgentEventView, WorkRow } from "./agent-event";
import { Toasts } from "./toasts";

type AnswerPart = { text: string; citation?: Evidence; number?: number };

// The sentence around [start, end): a newline or a terminator before the next sentence bounds it.
// A terminator followed by a bracket marker does not end the sentence, so trailing markers stay inside.
function sentenceAround(answer: string, start: number, end: number): string {
  const head = answer.slice(0, start);
  const from = Math.max(head.lastIndexOf("\n"), ...[...head.matchAll(/[.!?]\s/g)].map((m) => m.index + 1)) + 1;
  const stop = answer.slice(end).search(/[.!?](?=\s+[^\s\[]|\s*$)|\n/);
  return answer.slice(from, stop < 0 ? answer.length : end + stop + 1);
}

// ponytail: citations are exact Evidence.ids in brackets, never a parsed link scheme.
// A gated claim adds a numbered chip after the first exact occurrence of its text, one per known
// evidence id, unless its sentence already cites that id inline. Distinct ids number in answer order.
// An inline marker shows its number too, so a raw evidence id never reaches the reader.
function splitAnswer(answer: string, known: Map<string, Evidence>, claims: FinalClaim[]): AnswerPart[] {
  const entries = [...known.keys()];
  if (entries.length === 0) return [{ text: answer }];
  const marks: { at: number; end: number; id: string; marker?: string }[] = [];
  const pattern = new RegExp(`\\[(${entries.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\]`, "g");
  for (const m of answer.matchAll(pattern)) marks.push({ at: m.index, end: m.index + m[0].length, id: m[1], marker: m[0] });
  for (const claim of claims) {
    const at = claim.text ? answer.indexOf(claim.text) : -1;
    if (at < 0) continue;
    const end = at + claim.text.length;
    const sentence = sentenceAround(answer, at, end);
    for (const id of claim.evidenceIds) {
      if (known.has(id) && !sentence.includes(`[${id}]`) && !marks.some((m) => m.at === end && m.id === id && !m.marker)) marks.push({ at: end, end, id });
    }
  }
  marks.sort((a, b) => a.at - b.at);
  const numbers = new Map<string, number>();
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const mark of marks) {
    const at = Math.max(mark.at, last);
    if (at > last) parts.push({ text: answer.slice(last, at) });
    const number = numbers.get(mark.id) ?? numbers.size + 1;
    numbers.set(mark.id, number);
    parts.push({ text: `[${number}]`, citation: known.get(mark.id), number });
    last = Math.max(last, mark.end);
  }
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return parts;
}

function evidenceById(events: AgentEvent[]): Map<string, Evidence> {
  // Each evidence event carries the full merged list, so the last one wins.
  const evidence = events.findLast((e) => e.type === "evidence")?.evidence ?? [];
  return new Map(evidence.map((entry) => [entry.id, entry]));
}

// Readable labels read once from each captured document. A failed read keeps the caller's fallback label.
function useSourceLabels(references: SourceReference[]): Map<string, string> {
  const requested = useRef(new Set<string>());
  const mounted = useRef(false);
  const [labels, setLabels] = useState(new Map<string, string>());
  const ids = JSON.stringify([...new Set(references.map((reference) => reference.id))]);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    for (const id of JSON.parse(ids) as string[]) {
      if (requested.current.has(id)) continue;
      requested.current.add(id);
      void fetchSourceDocument(id).then((source) => {
        const label = sourceLabel(source);
        if (label && mounted.current) setLabels((current) => new Map(current).set(id, label));
      }, () => undefined);
    }
  }, [ids]);
  return labels;
}

// Each listed source once, in evidence order, with the evidence row that cites it.
function sourceLinks(known: Map<string, Evidence>): { reference: SourceReference; evidence: Evidence }[] {
  const links = new Map<string, { reference: SourceReference; evidence: Evidence }>();
  for (const evidence of known.values()) {
    for (const reference of evidence.sources ?? []) if (!links.has(reference.id)) links.set(reference.id, { reference, evidence });
  }
  return [...links.values()];
}

// One trio follow-up per offered session; null events means the user declined.
type TrioState = { sessionId: string; events: AgentEvent[] | null } | null;

// ponytail: the kernel buffers the trio answer, so the reply is read whole, not streamed.
async function fetchTrio(sessionId: string): Promise<AgentEvent[]> {
  try {
    const response = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op: "personas", sessionId, personas: FINAL_PERSONAS }),
    });
    if (!response.ok) return [{ type: "error", message: `request failed: ${response.status}` }];
    const events: AgentEvent[] = [];
    for (const line of (await response.text()).split("\n")) {
      if (!line.startsWith("data: ")) continue;
      try {
        events.push(JSON.parse(line.slice(6)));
      } catch {
        // A malformed frame never hides the rest of the answer.
      }
    }
    if (!events.some((e) => e.type === "done" || e.type === "error" || e.type === "failed")) events.push({ type: "error", message: "stream ended without a terminal event" });
    return events;
  } catch (error) {
    return [{ type: "error", message: error instanceof Error ? error.message : String(error) }];
  }
}

// A citation opens the provenance page for its first source in a new tab.
// onRetry reruns the request behind these events; without it failures offer no Try again.
// widgetOutcomes: what the workspace did with each widget receipt, for the toast text.
export function AgentConsole({ events, reasoning, debug, onRetry, widgetOutcomes }: { events: AgentEvent[]; reasoning: boolean; debug: boolean; onRetry?: () => void; widgetOutcomes?: WidgetOutcomes }) {
  const [trio, setTrio] = useState<TrioState>(null);
  const known = evidenceById(events);
  const links = sourceLinks(known);
  const answer = events.filter((e) => e.type === "answer_delta").map((e) => e.text).join("");
  const parts = splitAnswer(answer, known, events.findLast((e) => e.type === "claims")?.claims ?? []);
  const cited = parts.flatMap((part) => part.citation?.sources?.slice(0, 1) ?? []);
  const labels = useSourceLabels([...cited, ...links.map((link) => link.reference)]);
  const labelOf = (reference: SourceReference, evidence: Evidence, index: number): string =>
    labels.get(reference.id) ?? evidence.title ?? reference.title ?? `${evidence.source.toUpperCase() || "Source"} ${index}`;
  if (events.length === 0) {
    return (
      <div className="flex-1 space-y-3 overflow-y-auto p-3 text-sm">
        <div className="text-zinc-100">Stockbot — local-first investment research</div>
        <div className="text-xs text-zinc-500">
          Hedge-fund-style research infrastructure for individual investors — research and data foundation, not automated trading or portfolio management. I read
          primary sources and show my evidence; deterministic calculations, not model arithmetic.
        </div>
        <div className="space-y-1 text-xs text-zinc-500">
          <div>· SEC EDGAR — resolve a company to its CIK, search and list filings (10-K/10-Q/8-K), read bounded document windows, diff risk factors across periods.</div>
          <div>· Fundamentals — EPS, dividends, balance-sheet items, full statements, XBRL facts, obligations and valuation multiples.</div>
          <div>· Ownership &amp; insiders — beneficial stakes, holder changes, Forms 3/4/5 trades, Form 144 planned sales, offerings and dilution.</div>
          <div>· FINRA — short interest and leaderboards, Reg SHO volume, threshold securities, dataset queries.</div>
          <div>· Market context — analyst consensus, S&amp;P 500 weight, macro rates and inflation. No live quotes or option chains in this harness.</div>
          <div>· Web + time — Exa web highlights as research leads only (canonical SEC/FINRA wins on facts); current time for point-in-time queries.</div>
          <div>· Deep research — multi-question sessions with evidence, point-in-time known_at/as_of, raw archiving and provenance.</div>
          <div>· Full Stockbot optionally reads Robinhood portfolio, quotes, and options when BROKER_ENABLED — disabled here. Research only, not investment advice.</div>
        </div>
        <div className="text-xs text-zinc-600">Try: “What time is it?” · “NVDA risk factors?” · “AAPL insider sales last quarter?”</div>
      </div>
    );
  }
  // Prod hides internal work rows; debug (.env STOCKBOT_DEBUG=1) shows all.
  // The evidence and claims events stay out of prod rows: citations + source controls read them directly.
  // Status lines render in their own block in both modes, so the rows never repeat them.
  const visible = debug ? events : events.filter((e) => e.type === "agent_start" || e.type === "answer_delta" || e.type === "reasoning_start" || e.type === "done" || e.type === "error" || e.type === "failed");
  const head = visible.filter(
    (e) => e.type !== "answer_delta" && e.type !== "done" && e.type !== "error" && e.type !== "failed" && e.type !== "reasoning_start" && e.type !== "evidence" && e.type !== "claims" && e.type !== "offer_trio" && e.type !== "status",
  );
  const statuses = events.flatMap((e, i) => (e.type === "status" ? [<div key={i}>{e.text}</div>] : []));
  const tail = visible.filter((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const streaming = reasoning && tail.length === 0;
  // The offer appears once the stockbot answer is final; Yes or No answers it once per session.
  const offer = tail.length > 0 ? events.findLast((e) => e.type === "offer_trio") : undefined;
  const current = offer && trio?.sessionId === offer.sessionId ? trio : null;
  const askTrio = (sessionId: string): void => {
    setTrio({ sessionId, events: [{ type: "reasoning_start", model: "stockbot" }] });
    void fetchTrio(sessionId).then((trioEvents) => setTrio((state) => (state?.sessionId === sessionId ? { sessionId, events: trioEvents } : state)));
  };
  return (
    <div className="flex-1 space-y-2 overflow-y-auto p-3 text-sm">
      {head.map((e, i) => (
        <AgentEventView key={i} event={e} />
      ))}
      {statuses.length > 0 && (answer
        ? <details className="text-xs text-zinc-500"><summary className="cursor-pointer">Progress · {statuses.length} {statuses.length === 1 ? "step" : "steps"}</summary>{statuses}</details>
        : <div role="status" className="space-y-0.5 text-xs text-zinc-500">{statuses}</div>)}
      {streaming && <AgentEventView event={{ type: "reasoning_start", model: "stockbot" }} />}
      {answer && <div className="whitespace-pre-wrap pt-2 text-zinc-100">{parts.map((part, i) => {
        const citation = part.citation;
        const reference = citation?.sources?.[0];
        if (!citation || !reference) return <span key={i}>{part.text}</span>;
        const label = labelOf(reference, citation, part.number ?? 0);
        return <a key={i} href={routeHash({ page: "source", id: reference.id })} target="_blank" rel="noopener noreferrer" title={label} aria-label={`Open source: ${label}`} className="rounded border border-blue-800 px-1 text-blue-300 underline decoration-dotted hover:bg-blue-950">{part.text}</a>;
      })}</div>}
      <WorkRow events={events} debug={debug} />
      <ActionCards events={events} />
      {links.length > 0 && <div className="flex flex-wrap items-center gap-2 pt-2" role="group" aria-label="Captured sources">
        <span className="text-xs text-zinc-500">Sources:</span>
        {links.map(({ reference, evidence }, index) => (
          <a key={reference.id} href={routeHash({ page: "source", id: reference.id })} target="_blank" rel="noopener noreferrer" className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400">
            {labelOf(reference, evidence, index + 1)}
          </a>
        ))}
      </div>}
      {tail.map((e, i) => (
        <AgentEventView key={`tail-${i}`} event={e} debug={debug} onRetry={onRetry} />
      ))}
      {offer && !current && <TrioCard onYes={() => askTrio(offer.sessionId)} onNo={() => setTrio({ sessionId: offer.sessionId, events: null })} />}
      {current?.events && <AgentConsole events={current.events} reasoning={current.events.some((e) => e.type === "reasoning_start")} debug={debug} onRetry={() => askTrio(current.sessionId)} />}
      <Toasts events={events} onRetry={onRetry} widgetOutcomes={widgetOutcomes} />
    </div>
  );
}
