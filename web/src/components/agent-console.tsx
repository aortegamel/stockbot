import { useEffect, useRef, useState } from "react";
import { FINAL_PERSONAS, type AgentEvent, type Evidence, type FinalClaim, type SourceReference } from "../../../needle-harness/lib/agent/types";
import { fetchSourceDocument } from "../lib/research-client";
import { originalSourceUrl, safeSourceUrl } from "../lib/source-viewer";
import { ActionCards, TrioCard } from "./action-cards";
import { AgentEventView, WorkRow } from "./agent-event";
import { SourcePanel } from "./source-panel";
import { Toasts } from "./toasts";

type AnswerPart = { text: string; citation?: Evidence };

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
    if (!numbers.has(mark.id)) numbers.set(mark.id, numbers.size + 1);
    parts.push({ text: mark.marker ?? `[${numbers.get(mark.id)}]`, citation: known.get(mark.id) });
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

function useCitationUrls(parts: AnswerPart[]): Map<string, string | undefined> {
  const requested = useRef(new Set<string>());
  const mounted = useRef(false);
  const [urls, setUrls] = useState(new Map<string, string | undefined>());
  const references = new Map<string, SourceReference>();
  for (const part of parts) {
    const reference = part.citation?.sources?.[0];
    if (reference && !references.has(reference.id)) references.set(reference.id, { id: reference.id, url: reference.url });
  }
  const sources = JSON.stringify([...references.values()]);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    for (const reference of JSON.parse(sources) as SourceReference[]) {
      if (requested.current.has(reference.id)) continue;
      requested.current.add(reference.id);
      void fetchSourceDocument(reference.id)
        .then((source) => originalSourceUrl(source, true))
        .catch(() => safeSourceUrl(reference.url))
        .then((url) => {
          if (mounted.current) setUrls((current) => new Map(current).set(reference.id, url));
        });
    }
  }, [sources]);
  return urls;
}

type SourceSelection = { evidenceId: string; position: number } | null;
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
// onRetry reruns the request behind these events; without it failures offer no Try again.
export function AgentConsole({ events, reasoning, debug, onRetry }: { events: AgentEvent[]; reasoning: boolean; debug: boolean; onRetry?: () => void }) {
  const [selected, setSelected] = useState<SourceSelection>(null);
  const [trio, setTrio] = useState<TrioState>(null);
  const known = evidenceById(events);
  const sourceControls = [...known.values()].filter((entry) => entry.sources?.length);
  const answer = events.filter((e) => e.type === "answer_delta").map((e) => e.text).join("");
  const parts = splitAnswer(answer, known, events.findLast((e) => e.type === "claims")?.claims ?? []);
  const citationUrls = useCitationUrls(parts);
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
  const visible = debug ? events : events.filter((e) => e.type === "agent_start" || e.type === "answer_delta" || e.type === "reasoning_start" || e.type === "done" || e.type === "error" || e.type === "failed");
  const head = visible.filter(
    (e) => e.type !== "answer_delta" && e.type !== "done" && e.type !== "error" && e.type !== "failed" && e.type !== "reasoning_start" && e.type !== "evidence" && e.type !== "claims" && e.type !== "offer_trio",
  );
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
      {streaming && <AgentEventView event={{ type: "reasoning_start", model: "stockbot" }} />}
      {answer && <div className="whitespace-pre-wrap pt-2 text-zinc-100">{parts.map((part, i) => {
        const citation = part.citation;
        const reference = citation?.sources?.[0];
        if (!citation || !reference) return <span key={i}>{part.text}</span>;
        const url = citationUrls.has(reference.id) ? citationUrls.get(reference.id) : safeSourceUrl(reference.url);
        const className = "rounded border border-blue-800 px-1 text-blue-300 underline decoration-dotted hover:bg-blue-950";
        return url
          ? <a key={i} href={url} target="_blank" rel="noopener noreferrer" aria-label={`Open source for ${citation.id}`} className={className}>{part.text}</a>
          : <button key={i} type="button" onClick={() => setSelected({ evidenceId: citation.id, position: 0 })} aria-label={`Open source for ${citation.id}`} className={className}>{part.text}</button>;
      })}</div>}
      <WorkRow events={events} debug={debug} />
      <ActionCards events={events} />
      {sourceControls.length > 0 && <div className="flex flex-wrap gap-2 pt-2" role="group" aria-label="Captured sources">
        {sourceControls.map((entry) => (
          <button key={entry.id} type="button" onClick={() => setSelected({ evidenceId: entry.id, position: 0 })} className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400" aria-label={`Open captured sources for ${entry.id}`}>
            Sources: {entry.id}
          </button>
        ))}
      </div>}
      <SelectedSource selected={selected} known={known} setSelected={setSelected} />
      {tail.map((e, i) => (
        <AgentEventView key={`tail-${i}`} event={e} debug={debug} onRetry={onRetry} />
      ))}
      {offer && !current && <TrioCard onYes={() => askTrio(offer.sessionId)} onNo={() => setTrio({ sessionId: offer.sessionId, events: null })} />}
      {current?.events && <AgentConsole events={current.events} reasoning={current.events.some((e) => e.type === "reasoning_start")} debug={debug} onRetry={() => askTrio(current.sessionId)} />}
      <Toasts events={events} onRetry={onRetry} />
    </div>
  );
}
