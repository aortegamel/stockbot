import { useEffect, useRef, useState } from "react";
import type { AgentEvent, ChatTurn, Evidence, FinalClaim, SourceReference } from "../../../needle-harness/lib/agent/types";
import { fetchSourceDocument } from "../lib/research-client";
import { routeHash } from "../lib/route";
import { originalLink, sourceLabel, type OriginalLink } from "../lib/source-viewer";
import { ActionCards, type WidgetOutcomes } from "./action-cards";
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

type LoadedSource = { label: string; link: OriginalLink | undefined };

// Resolve each displayed source once. A failed read keeps its label but never creates an original link.
function useLoadedSources(references: SourceReference[]): Map<string, LoadedSource | null> {
  const requested = useRef(new Set<string>());
  const mounted = useRef(false);
  const [loaded, setLoaded] = useState(new Map<string, LoadedSource | null>());
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
        if (mounted.current) setLoaded((current) => new Map(current).set(id, { label: sourceLabel(source), link: originalLink(source) }));
      }, () => {
        if (mounted.current) setLoaded((current) => new Map(current).set(id, null));
      });
    }
  }, [ids]);
  return loaded;
}

// Each listed source once, in evidence order, with the evidence row that cites it.
function sourceLinks(known: Map<string, Evidence>): { reference: SourceReference; evidence: Evidence }[] {
  const links = new Map<string, { reference: SourceReference; evidence: Evidence }>();
  for (const evidence of known.values()) {
    for (const reference of evidence.sources ?? []) if (!links.has(reference.id)) links.set(reference.id, { reference, evidence });
  }
  return [...links.values()];
}

// Citations open resolved originals. Source cards keep the audit page under a separate provenance action.
// onRetry reruns the request behind these events; without it failures offer no Try again.
// widgetOutcomes: what the workspace did with each widget receipt, for the toast text.
// history: earlier messages of the stored chat, above the live turn in the same scroll column.
export function AgentConsole({ events, history = [], reasoning, debug, onRetry, widgetOutcomes }: { events: AgentEvent[]; history?: ChatTurn[]; reasoning: boolean; debug: boolean; onRetry?: () => void; widgetOutcomes?: WidgetOutcomes }) {
  const known = evidenceById(events);
  const links = sourceLinks(known);
  const finalAnswer = events.filter((e) => e.type === "answer_delta").map((e) => e.text).join("");
  const partial = events.findLast((e) => e.type === "answer_partial");
  const answer = partial?.text ?? finalAnswer;
  // A checked snapshot replaces the streamed draft.
  const done = events.some((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const showSources = answer.length > 0 || done;
  const claims = partial?.claims ?? events.findLast((e) => e.type === "claims")?.claims ?? [];
  const parts = splitAnswer(answer, known, claims);
  const cited = parts.flatMap((part) => part.citation?.sources?.slice(0, 1) ?? []);
  const loaded = useLoadedSources(showSources ? [...cited, ...links.map((link) => link.reference)] : []);
  const labelOf = (reference: SourceReference, evidence: Evidence, index: number): string =>
    loaded.get(reference.id)?.label || evidence.title || reference.title || `${evidence.source.toUpperCase() || "Source"} ${index}`;
  const scrollRef = useRef<HTMLDivElement>(null);
  // Stays pinned to the newest text until the reader scrolls up.
  const followTail = useRef(true);
  useEffect(() => {
    const box = scrollRef.current;
    if (box && followTail.current) box.scrollTop = box.scrollHeight;
  }, [history.length, events.length, answer]);
  if (events.length === 0 && history.length === 0) {
    return (
      <div className="flex-1 space-y-2 overflow-y-auto p-3 text-sm">
        <div className="text-zinc-100">Stockbot — local-first investment research</div>
        <div className="text-xs text-zinc-500">
          Ask about a stock, a filing, or the news. I read primary sources and show the evidence under each answer.
        </div>
        <div className="space-y-1 text-xs text-zinc-500">
          <div>· “What moved NVDA today?”</div>
          <div>· “NVDA risk factors?”</div>
          <div>· “AAPL insider sales last quarter?”</div>
        </div>
        <div className="text-xs text-zinc-600">Research only, not investment advice.</div>
      </div>
    );
  }
  // Prod hides internal work rows; debug (.env STOCKBOT_DEBUG=1) shows all.
  // The evidence and claims events stay out of prod rows: citations + source controls read them directly.
  // Show readable status lines live. Collapse their history when a terminal event arrives.
  const visible = debug ? events : events.filter((e) => e.type === "agent_start" || e.type === "answer_delta" || e.type === "reasoning_start" || e.type === "done" || e.type === "error" || e.type === "failed");
  const head = visible.filter(
    (e) => e.type !== "answer_delta" && e.type !== "answer_partial" && e.type !== "done" && e.type !== "error" && e.type !== "failed" && e.type !== "reasoning_start" && e.type !== "evidence" && e.type !== "claims" && e.type !== "status",
  );
  const statuses = events.flatMap((e, i) => (e.type === "status" ? [<div key={i}>{e.text}</div>] : []));
  const tail = visible.filter((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const streaming = reasoning && tail.length === 0;
  return (
    <div
      ref={scrollRef}
      className="flex-1 space-y-2 overflow-y-auto p-3 text-sm"
      onScroll={() => {
        const box = scrollRef.current;
        if (box) followTail.current = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      }}
    >
      {history.length > 0 && (
        <div aria-label="Earlier conversation" className="space-y-3">
          {history.map((turn, i) => (
            <div
              key={i}
              className={turn.role === "user"
                ? "text-[18px] font-semibold italic leading-snug text-zinc-100"
                : "whitespace-pre-wrap leading-relaxed text-zinc-100"}
            >
              {turn.role === "user" ? `> ${turn.content}` : turn.content}
            </div>
          ))}
        </div>
      )}
      {head.map((e, i) => (
        <AgentEventView key={i} event={e} />
      ))}
      {statuses.length > 0 && (tail.length > 0
        ? <details className="text-xs text-zinc-500"><summary className="cursor-pointer">Progress · {statuses.length} {statuses.length === 1 ? "step" : "steps"}</summary>{statuses}</details>
        : <div role="status" className="space-y-0.5 text-xs text-zinc-500">{statuses}</div>)}
      {streaming && !answer && <AgentEventView event={{ type: "reasoning_start", model: "stockbot" }} />}
      {!done && answer && <div role="status" className="pt-2 text-xs text-zinc-400">Partial checked answer · research continues</div>}
      {answer && <div className="whitespace-pre-wrap pt-2 leading-relaxed text-zinc-100">{parts.map((part, i) => {
        const citation = part.citation;
        const reference = citation?.sources?.[0];
        if (!citation || !reference) return <span key={i}>{part.text}</span>;
        const label = labelOf(reference, citation, part.number ?? 0);
        const link = loaded.get(reference.id)?.link;
        if (!link) {
          const status = loaded.has(reference.id) ? "Original source unavailable" : "Loading original source…";
          return <span key={i} title={`${label} — ${status}`} aria-label={`${status}: ${label}`} className="rounded border border-zinc-700 px-1 text-zinc-400">{part.text}</span>;
        }
        return <a key={i} href={link.href} target="_blank" rel="noopener noreferrer" title={label} aria-label={`Open source: ${label}`} className="rounded border border-blue-800 px-1 text-blue-300 underline decoration-dotted hover:bg-blue-950">{part.text}</a>;
      })}</div>}
      <WorkRow events={events} debug={debug} />
      <ActionCards events={events} />
      {showSources && links.length > 0 && <div className="flex flex-wrap items-center gap-2 pt-2" role="group" aria-label="Captured sources">
        <span className="text-xs text-zinc-500">Sources:</span>
        {links.map(({ reference, evidence }, index) => {
          const link = loaded.get(reference.id)?.link;
          return (
            <span key={reference.id} className="flex items-center gap-2 rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300">
              <span>{labelOf(reference, evidence, index + 1)}</span>
              {link
                ? <a href={link.href} target="_blank" rel="noopener noreferrer" className="text-blue-300 underline">{link.highlighted ? `Open on ${link.host} ↗` : `Original on ${link.host} ↗`}</a>
                : <span role="status" className="text-zinc-500">{loaded.has(reference.id) ? "Original source unavailable" : "Loading original source…"}</span>}
              <a href={routeHash({ page: "source", id: reference.id })} target="_blank" rel="noopener noreferrer" className="text-blue-300 underline">View provenance</a>
            </span>
          );
        })}
      </div>}
      {tail.map((e, i) => (
        <AgentEventView key={`tail-${i}`} event={e} debug={debug} onRetry={onRetry} />
      ))}
      <Toasts events={events} onRetry={onRetry} widgetOutcomes={widgetOutcomes} />
    </div>
  );
}
