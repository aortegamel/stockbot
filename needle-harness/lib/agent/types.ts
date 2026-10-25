// Shared agent types. Single owner for Tool/Evidence/Event/Metrics contracts.

export type FailureCategory =
  | "timeout"
  | "token_budget_exhausted"
  | "tool_budget_exhausted"
  | "job_budget_exhausted"
  | "wave_budget_exhausted"
  | "parallelism_exceeded"
  | "depth_exceeded"
  | "policy_denied"
  | "tool_error"
  | "model_error"
  | "model_output_failure"
  | "no_evidence"
  | "freeze_mismatch"
  | "committee_deadlock"
  | "synthesis_failed"
  | "context_capacity_exceeded"
  | "response_too_large"
  | "deadline_exceeded"
  | "provider_error"
  | "tool_output_limit"
  | "storage_error"
  | "policy_rejection"
  | "duplicate_research_action"
  | "research_loop_detected"
  // The source answered but holds no matching records ("No data found for ..."), as the kernel worker reports it.
  | "no_records";

// ponytail: research caps live in the kernel (None=null=unlimited); no TS budget mirror (DISPLAY_CHAR_LIMIT in loop.ts is a model-view bound, never termination).

type JSONSchema = {
  type: string;
  properties?: Record<string, unknown>;
  required?: string[];
  [k: string]: unknown;
};

// Server-created reference to an immutable capture; id never holds a filesystem path.
export type SourceReference = { id: string; title?: string; url?: string };

// ponytail: interim worker buffer (§11/§12: navigation result, not kernel evidence). Durable evidence requires kernel admission with source identity/hash/session/job/wave + PIT/provenance/identity gates; promotion via research.evidence.add lands with job/evidence wiring.
export type Evidence = {
  id: string;
  source: string;
  title?: string;
  url?: string;
  retrievedAt: string;
  content: string;
  sourceHandle?: Record<string, unknown>;
  sourceRefs?: Record<string, unknown>;
  sources?: SourceReference[];
  datapoints?: Datapoint[];
};

// Kernel-built canonical datapoint of one ledger row (Python evidence.canonical_datapoint /
// calculation_datapoint). Model context only: the worker gate re-verifies it before a claim may cite it.
export type Datapoint = {
  id: string;
  metric: string;
  value: string;
  entity: string;
  period: string;
  unit?: string;
  statement: string;
  inputs?: string[];
};

function text(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}
// Trust boundary: keep only well-formed kernel descriptors (dp: id, every identity field a string); never repair.
export function readDatapoints(raw: unknown): Datapoint[] {
  if (!Array.isArray(raw)) return [];
  const out: Datapoint[] = [];
  for (const item of raw) {
    if (item === null || typeof item !== "object" || Array.isArray(item)) continue;
    const { id, metric, value, entity, period, unit, statement, inputs } = item as Record<string, unknown>;
    if (!text(id) || !id.startsWith("dp:") || !text(metric) || !text(value) || !text(entity) || !text(period) || !text(statement)) continue;
    const point: Datapoint = { id, metric, value, entity, period, statement };
    if (text(unit)) point.unit = unit;
    if (Array.isArray(inputs)) point.inputs = inputs.filter(text);
    out.push(point);
  }
  return out;
}

// Match the source endpoint. References never contain paths or live URLs.
export const SOURCE_ID_RE = /^(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})$/;

// One candidate reference: only a well-formed id plus public title/url fields
// survive; every other shape drops the row instead of repairing it.
function readSourceReference(raw: unknown): SourceReference | undefined {
  if (raw === null || typeof raw !== "object") return undefined;
  const { id, title, url } = raw as Record<string, unknown>;
  if (typeof id !== "string" || !SOURCE_ID_RE.test(id)) return undefined;
  const ref: SourceReference = { id };
  if (typeof title === "string" && title) ref.title = title.slice(0, 300);
  if (typeof url === "string" && /^https?:\/\//i.test(url)) ref.url = url.slice(0, 2000);
  return ref;
}
// Trust boundary: keep only well-formed server references, in source order and
// deduplicated by id; anything else is dropped, never repaired.
export function sourceReferences(raw: unknown): SourceReference[] {
  if (!Array.isArray(raw)) return [];
  const out: SourceReference[] = [];
  const seen = new Set<string>();
  for (const r of raw) {
    const ref = readSourceReference(r);
    if (ref === undefined || seen.has(ref.id)) continue;
    seen.add(ref.id);
    out.push(ref);
  }
  return out;
}
// Final-answer claim inventory: model-declared types mirror Python CLAIM_TYPES
// exactly (snake_case over the wire); confidence is model-declared 0..1,
// carried through, never a reject threshold.
export const FINAL_CLAIM_TYPES = [
  "observed_fact",
  "calculation",
  "inference",
  "assumption",
  "unknown",
  "contradicted",
  "conditional_risk",
  "attributed_report",
] as const;
export type FinalClaimType = (typeof FINAL_CLAIM_TYPES)[number];

// Claims never state value/entity/period/unit themselves: every number they state
// resolves through datapointIds (snake_case datapoint_ids on the wire) to a kernel
// descriptor of a cited row, and the Python gate binds the wording to it.
export type FinalClaim = {
  text: string;
  claimType: FinalClaimType;
  evidenceIds: string[];
  datapointIds?: string[];
  confidence?: number;
};

// One model answer before any gate: narrative prose plus the model's declared
// claims. The claims stay unproven declarations until the final gates return
// kept text. Nothing here is ledger proof.
export type NarrativeResult = { prose: string; claims: FinalClaim[]; confidence?: number };

export type FinalPersonaResult = NarrativeResult & { persona: Persona };

// Final-answer personas in fixed output order; each runs as its own Muse call.
export const FINAL_PERSONAS = ["stockbot", "bearbot", "bullbot"] as const;
export type Persona = (typeof FINAL_PERSONAS)[number];

// Saved workspace widget kinds in fixed order; must match WIDGET_TYPES in app/tools.py.
// The add_widget tool accepts every kind except tv-link (it needs a user-pasted URL).
export const WIDGET_TYPES: readonly string[] = ["price-volume", "financials", "news", "monthly-returns", "short-interest", "short-volume", "insider", "filings", "multiples", "tv-link"];

// Feed item kinds. An Ask-about-this context names one item of these kinds.
export const ASK_CONTEXT_KINDS = ["news", "filing", "insider", "alert"] as const;
// A user-picked feed item beside the verbatim prompt. It is untrusted data, never instructions.
export type AskContext = { kind: (typeof ASK_CONTEXT_KINDS)[number]; title: string; url?: string; source: string; sym?: string; at: string };
// A link is safe only when the viewer could fetch it directly: web scheme, host present, no credentials.
// Shared by the web viewer and the agent route's feed-context check; no DOM dependency.
function fetchableHttpUrl(parsed: URL): boolean {
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
  return parsed.hostname !== "" && !parsed.username && !parsed.password;
}

export function safeSourceUrl(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const parsed = new URL(value);
    if (fetchableHttpUrl(parsed)) return parsed.href;
  } catch {
    return undefined;
  }
  return undefined;
}

export type ToolResult = { ok: true; evidence: Evidence; chartSelection?: unknown; watchlistUpdate?: unknown; widgetSelection?: unknown } | { ok: false; error: string; category: FailureCategory; retryable: boolean };

export type Tool = {
  description: string;
  parameters: JSONSchema;
  execute(args: Record<string, unknown>, opts: { sessionId: string }): Promise<ToolResult>;
};

export type AgentEvent =
  | { type: "agent_start"; prompt: string; context?: AskContext }
  | { type: "chart_update"; ticker: string }
  | { type: "watchlist_update"; action: "add" | "remove"; ticker: string }
  | { type: "widget_update"; ticker: string; widgetType: string }
  | { type: "needle_decision"; step: number; tool: string | null; arguments: Record<string, unknown>; confidence: number | null }
  | { type: "tool_start"; tool: string }
  | { type: "tool_result"; tool: string; evidenceId?: string; preview: string }
  | { type: "evidence"; evidence: Evidence[]; sessionId?: string }
  | { type: "progress"; stage: string; detail?: Record<string, unknown> }
  | { type: "reasoning_start"; model: string }
  | { type: "answer_delta"; text: string }
  | { type: "done"; metrics: Metrics }
  // The gated research answer's cited claim inventory; each text is a verbatim answer substring.
  | { type: "claims"; claims: FinalClaim[] }
  | { type: "error"; message: string }
  // A stockbot-only research answer offers the trio over the same session evidence.
  | { type: "offer_trio"; sessionId: string }
  | { type: "tool_failed"; tool: string; category: FailureCategory; preview: string }
  | { type: "failed"; category: FailureCategory; message: string }
  // A readable progress line for the user, never answer text. Build it with statusEvent.
  | { type: "status"; text: string };

export type Metrics = {
  totalMs: number;
  needle: { calls: number; totalMs: number; escalations: number };
  tools: { calls: number; totalMs: number };
  muse: {
    calls: number;
    providerCalls?: number;
    inputTokens?: number;
    outputTokens?: number;
    cachedTokens?: number;
    totalMs: number;
  };
  evidence: { count: number; characters: number };
  failures: Partial<Record<FailureCategory, number>>;
  // Phase milliseconds; a phase with no measurement stays absent, never 0.
  timings?: Timings;
};

export type Timings = Partial<Record<"routeMs" | "intakeMs" | "reasonerMs" | "admitMs" | "nodesMs" | "personasMs" | "narrativeMs" | "binderMs" | "gatesMs" | "firstTextMs", number>>;

const SENSITIVE_KEY_NORMS: Record<string, true> = {
  accesstoken: true,
  refreshtoken: true,
  token: true,
  authorization: true,
  auth: true,
  clientsecret: true,
  clientid: true,
  secret: true,
  password: true,
  apikey: true,
  cookie: true,
  crumb: true,
  accountnumber: true,
  accountid: true,
};

function redactValue(value: unknown): unknown {
  if (typeof value === "string") return value.slice(0, 2000);
  if (Array.isArray(value)) return value.map(redactValue);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = SENSITIVE_KEY_NORMS[k.toLowerCase().replace(/[^a-z0-9]/g, "")] ? "[REDACTED]" : redactValue(v);
    }
    return out;
  }
  return value;
}

// TS key-based redaction is a UI safeguard only; the authoritative recorder boundary is app/redact.py (key norms PLUS free-text Bearer/sk/JWT patterns — a secret smuggled inside a query value is invisible to key matching).
export function redactArgs(a: Record<string, unknown>): Record<string, unknown> {
  return redactValue(a) as Record<string, unknown>;
}

// One collapsed line bounded to 120 chars: a status can quote the prompt, a search query,
// or model-chosen arguments. Clients render it as plain text.
export function statusEvent(text: string): AgentEvent {
  const line = text.replace(/\s+/g, " ").trim();
  return { type: "status", text: line.length > 120 ? `${line.slice(0, 119)}…` : line };
}

// Present-tense [verb, object] for common tools; any other tool reads "Running <tool name>".
const TOOL_STATUS: Record<string, readonly [string, string]> = {
  get_insider_activity: ["Reading", "Form 4 insider filings"], get_planned_insider_sales: ["Reading", "Form 144 planned sales"],
  get_beneficial_ownership: ["Reading", "major shareholder stakes"], get_short_interest: ["Checking", "FINRA short interest"],
  get_reg_sho_volume: ["Checking", "FINRA short-sale volume"], get_short_pressure_profile: ["Checking", "short pressure"],
  get_fundamentals: ["Reading", "reported fundamentals"], get_financial_statements: ["Reading", "financial statements"],
  get_xbrl_facts: ["Reading", "reported financial facts"], get_valuation_metrics: ["Calculating", "valuation multiples"],
  get_analyst_estimates: ["Reading", "analyst estimates"], get_material_events: ["Reading", "recent material events"],
  search_sec_filings: ["Searching", "SEC filings"], search_sec_filings_bounded: ["Searching", "SEC filings"],
  list_sec_filings: ["Listing", "SEC filings"], get_sec_document: ["Reading", "the SEC filing"],
  find_sec_entities: ["Finding", "the company at the SEC"], find_sec_entities_bounded: ["Finding", "the company at the SEC"],
  search_web: ["Searching", "the web"], get_current_time: ["Checking", "the current time"],
  query_finra: ["Analyzing", "FINRA data"], get_finra_datapoints: ["Reading", "FINRA data"],
};

// "Reading Form 4 insider filings for NVDA"; search_web names its query ("Searching the web: <query>").
// verb replaces the tool verb, e.g. "Looking up" before the arguments exist.
export function toolStatus(tool: string, subject?: string, verb?: string): string {
  const [toolVerb, what] = Object.hasOwn(TOOL_STATUS, tool) ? TOOL_STATUS[tool] : ["Running", tool.replaceAll("_", " ")];
  const suffix = subject ? (tool === "search_web" ? `: ${subject}` : ` for ${subject}`) : "";
  return `${verb ?? toolVerb} ${what}${suffix}`;
}

export function foundStatus(count: number): string {
  return count > 0 ? `Found ${count} ${count === 1 ? "source" : "sources"}` : "Found nothing new";
}
