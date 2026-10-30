import { useEffect, useState } from "react";
import { redactArgs, type AgentEvent, type AskContext, type FailureCategory } from "../../../needle-harness/lib/agent/types";

const LIMIT = "The research reached its work limit before it finished.";
const MODEL = "The model failed while preparing the answer.";
const SOURCE = "A data source failed while answering.";
// Prod failure text: one plain sentence per category. No raw message, stack, or tool name.
const FAILURE_TEXT: Record<FailureCategory, string> = {
  timeout: "The research took too long and stopped.", deadline_exceeded: "The research took too long and stopped.",
  token_budget_exhausted: LIMIT, tool_budget_exhausted: LIMIT, job_budget_exhausted: LIMIT, wave_budget_exhausted: LIMIT,
  parallelism_exceeded: LIMIT, depth_exceeded: LIMIT, context_capacity_exceeded: LIMIT, response_too_large: LIMIT,
  tool_output_limit: LIMIT, duplicate_research_action: LIMIT, research_loop_detected: LIMIT,
  policy_denied: "This request is not allowed.", policy_rejection: "This request is not allowed.",
  tool_error: SOURCE, storage_error: SOURCE, freeze_mismatch: SOURCE,
  model_error: MODEL,
  model_output_failure: "The model returned an unusable answer.",
  provider_error: "The model provider failed to complete the request.",
  synthesis_failed: "The research could not combine the findings into an answer.",
  committee_deadlock: "The research models could not agree on an answer.",
  no_evidence: "No evidence was found for this question.", no_records: "A data source had no records for this request.",
};

export function failureText(category: FailureCategory): string {
  return FAILURE_TEXT[category] ?? "The research failed before it finished.";
}

// Prod step text: one plain line per needle catalog tool (.needle-catalog.json). Debug shows the raw name.
const TOOL_PHRASES: Record<string, string> = {
  get_fundamentals: "Read a reported fundamental", find_sec_entities: "Looked up the company at the SEC",
  find_sec_entities_bounded: "Looked up the company at the SEC", search_sec_filings: "Searched SEC filings",
  search_sec_filings_bounded: "Searched SEC filings", list_sec_filings: "Listed SEC filings",
  search_sec_relationships: "Checked ownership and deal links", get_sec_search_coverage: "Checked SEC data coverage",
  get_sec_filing: "Read an SEC filing record", list_sec_documents: "Listed filing documents", get_sec_document: "Read SEC filing",
  diff_sec_filings: "Compared SEC filings", get_material_events: "Read recent material events",
  get_beneficial_ownership: "Read major shareholder stakes", get_ownership_changes: "Compared shareholder stakes",
  get_insider_activity: "Read insider trades", get_planned_insider_sales: "Read planned insider sales",
  get_offering_history: "Read offering history", get_dilution_profile: "Calculated dilution",
  get_governance_events: "Read governance filings", get_transaction_status: "Read deal filings",
  get_short_pressure_profile: "Checked short pressure", get_recent_ownership_filings: "Read recent ownership filings",
  diff_risk_factors: "Compared risk factors", get_financial_statements: "Read financial statements",
  get_xbrl_facts: "Read a reported financial fact", get_short_interest: "Read short interest",
  get_short_interest_leaderboard: "Read the most-shorted list", get_reg_sho_volume: "Read short-sale volume",
  get_threshold_securities: "Read threshold securities", get_analyst_estimates: "Read analyst estimates",
  get_sp500_weight: "Read the S&P 500 weight", get_obligations: "Read future obligations",
  get_valuation_metrics: "Calculated valuation multiples", list_finra_datasets: "Listed FINRA datasets",
  describe_finra_dataset: "Read a FINRA dataset description", get_finra_datapoints: "Read FINRA data",
  query_finra: "Analyzed FINRA data", search_web: "Searched the web",
  thesis_create: "Saved a thesis", thesis_show: "Read a thesis", thesis_refine: "Updated a thesis",
  thesis_watch: "Checked thesis watch rules", thesis_journal: "Added a thesis note", thesis_status: "Changed thesis monitoring",
  research_start: "Started a research session", research_resume: "Resumed a research session",
  research_status: "Checked research status", research_cancel: "Cancelled a research session",
  research_read: "Read saved research", research_read_search: "Read saved search results",
  research_add_evidence: "Recorded a finding", research_submit_source_result: "Completed a source check",
  research_add_analysis: "Recorded an analysis", research_finalize: "Finalized the research",
  find_alternative_signals: "Read alternative signals", get_trend_evidence: "Read search trends",
  investigate_social_arbitrage_candidate: "Checked a trend candidate", get_macro_context: "Read macro data",
  search_company_patents: "Searched company patents", get_current_time: "Checked the current time",
  set_chart_symbol: "Changed the chart", update_watchlist: "Updated the watchlist",
};
// Worker progress stages (app/research/kernel_worker.py).
const STAGE_PHRASES: Record<string, string> = {
  session: "Opened a research session", intake_start: "Started gathering sources", intake_done: "Gathered first sources",
  intake_evidence: "Reviewed first evidence", reasoner_start: "Planned the research", reasoner_retry: "Re-planned the research",
  reasoner_done: "Finished the plan", nodes_done: "Finished the research steps",
};

type Step = Extract<AgentEvent, { type: "tool_start" | "progress" }>;

function stepLine(step: Step, debug: boolean): string {
  if (step.type === "tool_start") return debug ? step.tool : TOOL_PHRASES[step.tool] ?? "Used a research tool";
  // Never print raw detail JSON: stage names stay, blobs go behind the JSONL log.
  return debug ? step.stage : STAGE_PHRASES[step.stage] ?? "Worked on the research";
}

// ponytail: events carry no timestamps, so the first render that sees an event stamps it.
const SEEN_AT = new WeakMap<AgentEvent, number>();
function seenAt(event: AgentEvent): number {
  const at = SEEN_AT.get(event) ?? Date.now();
  SEEN_AT.set(event, at);
  return at;
}

// One live line while the run works, one summary line after it ends; the steps fold out below.
export function WorkRow({ events, debug }: { events: AgentEvent[]; debug: boolean }) {
  const [, tick] = useState(0);
  const terminal = events.find((e) => e.type === "done" || e.type === "error" || e.type === "failed");
  const live = terminal === undefined;
  const steps = events.filter((e): e is Step => e.type === "tool_start" || e.type === "progress");
  const ticking = live && steps.length > 0;
  useEffect(() => {
    if (!ticking) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [ticking]);
  if (steps.length === 0) return null;
  const count = `${steps.length} ${steps.length === 1 ? "step" : "steps"}`;
  const start = seenAt(events[0]);
  const label = live
    ? `Working · ${count} · ${Math.max(0, Math.floor((Date.now() - start) / 1000))}s`
    : `Worked for ${Math.round((terminal.type === "done" ? terminal.metrics.totalMs : seenAt(terminal) - start) / 1000)}s · ${count}`;
  return (
    <div className="pt-2 text-xs text-zinc-400">
      <details>
        <summary className="cursor-pointer">{label}</summary>
        <ol className="pl-4 text-zinc-500">
          {steps.map((step, i) => <li key={i}>{stepLine(step, debug)}</li>)}
        </ol>
      </details>
    </div>
  );
}

function FailureView({ text, onRetry }: { text: string; onRetry?: () => void }) {
  return (
    <div className="text-red-400" role="alert">
      {onRetry ? <>✗ {text} <button type="button" onClick={onRetry} className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400">Try again</button></> : <>✗ {text}</>}
    </div>
  );
}

function conf(c: number | null): string {
  return c === null ? "n/a" : `${Math.round(c * 100)}%`;
}

function NeedleDecisionView({ event }: { event: Extract<AgentEvent, { type: "needle_decision" }> }) {
  return (
    <div>
      <span className="rounded border border-green-900 px-1 text-[13px] text-green-500">NEEDLE/LOCAL</span>{" "}
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
      return <div className="pl-4 text-zinc-500">├─ {event.stage}</div>;
    }
    case "tool_failed":
      return <div className="pl-4 text-zinc-500">├─ {event.tool} / ✗ {event.category} — {event.preview}</div>;
  }
}

// A readable feed-item chip: "News · Stocktwits · NVDA — title". CSS cuts a long title; title= holds it whole.
export function AskContextChip({ context, onRemove }: { context: AskContext; onRemove?: () => void }) {
  const kind = context.kind[0].toUpperCase() + context.kind.slice(1);
  return (
    <div className="pill" aria-label="Attached feed item" title={context.title} style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0, maxWidth: "100%" }}>
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
        {[kind, context.source, context.sym].filter(Boolean).join(" · ")} — {context.title}
      </span>
      {onRemove && <button type="button" className="ghost" aria-label="Remove context" onClick={onRemove}>×</button>}
    </div>
  );
}

export function AgentEventView({ event, debug = false, onRetry }: { event: AgentEvent; debug?: boolean; onRetry?: () => void }) {
  switch (event.type) {
    case "agent_start":
      return <div className="text-zinc-100">&gt; {event.prompt}{event.context && <AskContextChip context={event.context} />}</div>;
    case "reasoning_start":
      return (
        <div>
          <span className="rounded border border-violet-900 px-1 text-[13px] text-violet-400">STOCKBOT</span>{" "}
          <span style={{ color: "#a78bfa" }}>● stockbot / working…</span>
        </div>
      );
    case "answer_delta":
    case "answer_partial":
      return <span>{event.text}</span>;
    case "done": {
      const m = event.metrics;
      // Phase timings are debug output (.env STOCKBOT_DEBUG=1); an absent phase never renders.
      const timings = debug ? Object.entries(m.timings ?? {}).flatMap(([k, v]) => (v === undefined ? [] : [`${k} ${Math.round(v)}ms`])) : [];
      return (
        <div className="pt-2 text-xs text-zinc-500">
          <div role="status">{event.status === "completed_with_limitations" ? "Completed with limitations" : event.status === "failed" ? "Failed" : event.status === "cancelled" ? "Cancelled" : "Completed"}</div>
          {event.limitations && event.limitations.length > 0 && <ul className="list-disc pl-4">{event.limitations.map((limitation, i) => <li key={i}>{limitation}</li>)}</ul>}
          <div>── metrics ──</div>
          <div>
            Needle calls: {m.needle.calls} · Tool calls: {m.tools.calls} · Stockbot calls: {m.muse.calls}
            {m.muse.providerCalls !== undefined && <> · Provider calls: {m.muse.providerCalls}</>} · Evidence:{" "}
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
      // Raw diagnostics remain in debug output.
      return <FailureView text={debug ? [event.stage, event.code, event.message].filter(Boolean).join(": ") : "The request failed. Please retry."} onRetry={onRetry} />;
    case "failed":
      return debug ? <div className="text-red-400">✗ {[event.category, event.stage, event.code, event.message].filter(Boolean).join(": ")}</div> : <FailureView text={failureText(event.category)} onRetry={onRetry} />;
    default:
      return <WorkEventView event={event} />;
  }
}
