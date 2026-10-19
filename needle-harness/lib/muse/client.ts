import { setTimeout as sleep } from "node:timers/promises";
import { FINAL_CLAIM_TYPES, type Evidence, type FinalClaim, type FinalClaimType, type Persona } from "../agent/types";

export type MuseUsage = {
  inputTokens?: number;
  outputTokens?: number;
  cachedTokens?: number;
};

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}

const SYSTEM = "You are Stockbot.";

// Stance sentences adapted from the old committee roles (app/research/agents/*bot.py) to standalone prose.
const PERSONAS: Record<Persona, { name: string; stance: string }> = {
  stockbot: {
    name: "Stockbot",
    stance:
      "You give the balanced base case. Weigh the evidence and its gaps with equal rigor. State the most probable outcome.",
  },
  bearbot: {
    name: "Bearbot",
    stance:
      "You give the bear case. Build the strongest downside case, including contagion. Consider exposure size, concentration, credit and liquidity channels, commitments, termination triggers, and second-order effects on counterparties.",
  },
  bullbot: {
    name: "Bullbot",
    stance:
      "You give the bull case. Build the strongest resilience case. Consider limited damage, buffers, mitigants, contractual protection, or diversified revenue.",
  },
};

// Final-answer claim inventory: trailing fenced block, stripped before display.
// Verbatim qualified wording, model-declared types mirroring Python CLAIM_TYPES
// exactly, EVIDENCE row ids only, model-declared confidence (carried, never a threshold).
const INVENTORY_RULE =
  " End your answer with a fenced inventory block and nothing after it:" +
  " ```json {\"claims\": [{\"text\": \"<verbatim prose substring>\", \"claim_type\": \"<observed_fact|inference|unknown|contradicted|conditional_risk|attributed_report>\", \"evidence_ids\": [\"<EVIDENCE row ids>\"], \"confidence\": 0.9}], \"confidence\": 0.9}```" +
  " Each claims[].text copies your prose above verbatim, qualified wording as written including attribution/qualification" +
  " (conditional_risk states the condition and consequence as qualified, not that it will occur;" +
  " attributed_report states who reported it, not that the event happened)." +
  " claim_type is declared by you, never inferred: observed_fact the cited passage states it directly;" +
  " inference reasoned from cited evidence; contradicted cited evidence conflicts with it;" +
  " unknown the evidence does not establish it." +
  " evidence_ids lists only exact ids of rows in the EVIDENCE section, never node ids or decision records." +
  " Unknown claims may cite none. All other types need at least one." +
  " confidence is your own 0..1, carried through, never a threshold." +
  " Nothing found means {\"claims\": []}." +
  " The block is stripped before display; prose claims missing from it fail the accuracy gate.";
function researchSystem(persona: Persona): string {
  const { name, stance } = PERSONAS[persona];
  return `You are ${name}. ${stance} Answer in prose.` + INVENTORY_RULE;
}

function directSystem(persona: Persona): string {
  const { name } = PERSONAS[persona];
  return `You are ${name}.`;
}
function isConf(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
}
// Model-declared evidence ids: only exact non-empty strings survive; anything
// else is dropped, never repaired.
function declaredEvidenceIds(value: unknown): string[] {
  const raw = Array.isArray(value) ? value : [];
  return raw.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
}

function parseInventoryClaim(item: unknown): FinalClaim | undefined {
  if (!item || typeof item !== "object" || Array.isArray(item)) return undefined;
  const rec = item as Record<string, unknown>;
  if (typeof rec.text !== "string" || !rec.text) return undefined;
  const rawType = typeof rec.claim_type === "string" ? rec.claim_type.trim().toLowerCase() : "";
  if (!(FINAL_CLAIM_TYPES as readonly string[]).includes(rawType)) return undefined;
  const claim: FinalClaim = { text: rec.text, claimType: rawType as FinalClaimType, evidenceIds: declaredEvidenceIds(rec.evidence_ids) };
  if (isConf(rec.confidence)) claim.confidence = rec.confidence;
  return claim;
}
const INVENTORY_FENCE = /```json\s*([\s\S]*?)\s*```\s*$/;
// Exported for tests/kernel: trailing inventory block stripped from prose, claims
// coerced from snake_case wire to camelCase, confidence carried when 0..1.
export function splitInventory(raw: string): { prose: string; claims: FinalClaim[]; confidence?: number } {
  const m = INVENTORY_FENCE.exec(raw);
  if (!m) return { prose: raw, claims: [] };
  const prose = raw.slice(0, m.index);
  try {
    const doc: unknown = JSON.parse(m[1] ?? "");
    const rec = doc && typeof doc === "object" && !Array.isArray(doc) ? (doc as Record<string, unknown>) : undefined;
    const list = rec?.claims;
    const claims = Array.isArray(list)
      ? list.flatMap((c) => {
        const p = parseInventoryClaim(c);
        return p ? [p] : [];
      })
      : [];
    if (isConf(rec?.confidence)) return { prose, claims, confidence: rec?.confidence as number };
    return { prose, claims };
  } catch {
    return { prose, claims: [] };
  }
}
function withInventory(r: { text: string; usage: MuseUsage }): {
  text: string;
  usage: MuseUsage;
  missingEvidence?: string;
  claims: FinalClaim[];
  confidence?: number;
} {
  const inv = splitInventory(r.text);
  const gap = /^Missing-Evidence:\s*(.+)$/m.exec(inv.prose)?.[1];
  return {
    text: inv.prose,
    usage: r.usage,
    claims: inv.claims,
    ...(gap !== undefined ? { missingEvidence: gap } : {}),
    ...(inv.confidence !== undefined ? { confidence: inv.confidence } : {}),
  };
}

// All admitted evidence reaches personas; no character budget.
function formatEvidence(evidence: Evidence[]): string {
  return evidence
    .map(
      (e) =>
        `[${e.id}]\nsource: ${e.source}\n${e.title ? `title: ${e.title}\n` : ""}${e.url ? `url: ${e.url}\n` : ""}content: ${e.content}`,
    )
    .join("\n\n");
}

// Synthesis only: labeled sibling drafts for the Stockbot judge. Bear/Bull
// calls never receive drafts; direct and omitted-persona prompts stay byte-identical.
function formatDrafts(drafts: { persona: Persona; text: string }[]): string {
  return drafts.map((d) => `── ${PERSONAS[d.persona].name} ──\n${d.text}`).join("\n\n");
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? value as Record<string, unknown> : undefined;
}

function eventDelta(rec: Record<string, unknown>): string | undefined {
  if (typeof rec.delta === "string" && /delta/i.test(typeof rec.type === "string" ? rec.type : "")) return rec.delta;
  const first = Array.isArray(rec.choices) ? objectRecord(rec.choices[0]) : undefined;
  const content = objectRecord(first?.delta)?.content;
  return typeof content === "string" && content ? content : undefined;
}

function mergeTokenCount(usage: MuseUsage, key: keyof MuseUsage, count: unknown): void {
  if (typeof count === "number") usage[key] = Math.max(usage[key] ?? 0, count);
}

function mergeUsage(usage: MuseUsage, raw: unknown): void {
  const u = objectRecord(raw);
  if (!u) return;
  // Counts retain their maximum across all SSE chunks and both provider shapes.
  mergeTokenCount(usage, "inputTokens", u.input_tokens);
  mergeTokenCount(usage, "outputTokens", u.output_tokens);
  const cached = u.cached_tokens
    ?? objectRecord(u.input_tokens_details)?.cached_tokens
    ?? objectRecord(u.prompt_tokens_details)?.cached_tokens;
  mergeTokenCount(usage, "cachedTokens", cached);
  mergeTokenCount(usage, "inputTokens", u.prompt_tokens);
  mergeTokenCount(usage, "outputTokens", u.completion_tokens);
}

// Exported for tests: a chunk matching both delta shapes appends once.
export function harvest(ev: unknown, acc: { text: string; usage: MuseUsage }, onDelta: (t: string) => void): void {
  const rec = objectRecord(ev);
  if (!rec) return;
  const delta = eventDelta(rec);
  if (delta !== undefined) {
    acc.text += delta;
    onDelta(delta);
  }
  mergeUsage(acc.usage, rec.usage ?? objectRecord(rec.response)?.usage);
}

function harvestFrame(chunk: string, acc: { text: string; usage: MuseUsage }, onDelta: (t: string) => void): void {
  for (const line of chunk.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("data:")) continue;
    const data = t.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    try {
      harvest(JSON.parse(data), acc, onDelta);
    } catch {
      // Ignore a malformed data line without losing earlier deltas.
    }
  }
}

async function readSse(
  res: Response,
  onDelta: (t: string) => void,
): Promise<{ text: string; usage: MuseUsage }> {
  const acc = { text: "", usage: {} as MuseUsage };
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      harvestFrame(chunk, acc, onDelta);
    }
  }
  return acc;
}

// Per-fetch limit for calls without an explicit persona (direct and single-tool callers).
const FETCH_TIMEOUT_MS = 120_000;

async function post(
  base: string,
  apiKey: string,
  session: string,
  path: "/responses" | "/chat/completions",
  body: unknown,
  onDelta: (t: string) => void,
  signal: AbortSignal,
): Promise<{ text: string; usage: MuseUsage }> {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": "needle-harness/0.1",
      "x-opencode-session": session,
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`muse ${path} ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return readSse(res, onDelta);
}

type ReasonOptions = {
  prompt: string;
  ticker?: string;
  evidence: Evidence[];
  escalated: boolean;
  direct?: boolean;
  // Optional graph state (kernel.ts always sends these; direct callers omit them).
  // escalated/incompleteGuard/unresolved add the gap instruction; it states partial coverage when evidence rows exist.
  objective?: string;
  nodes?: unknown[];
  decisions?: unknown[];
  unresolved?: string[];
  incompleteGuard?: boolean;
  // Final-answer persona; omitted = Stockbot (direct and single-tool callers).
  persona?: Persona;
  // Synthesis only: sibling drafts for the Stockbot judge (trio only). Bear/Bull and direct callers never use this.
  drafts?: { persona: Persona; text: string }[];
  // Request cancellation and wall deadline (epoch ms), shared by every fetch and retry of this call.
  signal?: AbortSignal;
  deadlineAt?: number;
  onDelta: (text: string) => void;
  // reason() strips the inventory fence: the returned text is display prose plus
  // parsed claims (external prose unchanged; streamed deltas keep the raw block).
};

function reasonEndpoint(): { apiKey: string; model: string; base: string } {
  const apiKey = requiredEnv("OPENCODE_API_KEY");
  const model = requiredEnv("OPENCODE_MODEL");
  const responsesUrl = requiredEnv("OPENCODE_URL");
  if (!responsesUrl.endsWith("/responses")) throw new Error("opencode_unavailable: OPENCODE_URL must end in /responses");
  return { apiKey, model, base: responsesUrl.slice(0, -"/responses".length) };
}

function reasonSignal(opts: ReasonOptions): AbortSignal {
  const left = opts.deadlineAt === undefined ? undefined : opts.deadlineAt - Date.now();
  if (left !== undefined && left <= 0) throw new Error("muse deadline exceeded");
  const signal = AbortSignal.any([
    ...(opts.signal ? [opts.signal] : []),
    ...(left !== undefined ? [AbortSignal.timeout(left)] : []),
  ]);
  signal.throwIfAborted();
  return signal;
}

function gapInstruction(opts: ReasonOptions): string {
  const needsGapLine =
    opts.escalated || opts.incompleteGuard === true || (opts.unresolved !== undefined && opts.unresolved.length > 0);
  if (!needsGapLine) return "";
  return (opts.evidence.length > 0
    ? "\nEvidence coverage is partial: some evidence rows are present, but retrieval did not resolve every gap."
    : "\nRetrieval found no usable evidence.") +
    " If evidence necessary to answer the original user request is still missing, end your answer with a line: Missing-Evidence: <only the evidence necessary to answer the original user request>.";
}

function reasonInput(opts: ReasonOptions): { role: string; content: string }[] {
  const persona = opts.persona ?? "stockbot";
  // An omitted persona retains the original prompt for direct and single-tool callers.
  const system = opts.direct
    ? directSystem(persona)
    : (opts.persona ? researchSystem(opts.persona) : SYSTEM) + gapInstruction(opts);
  const stockContext = opts.ticker
    ? `\n\nCURRENT STOCK CONTEXT\nThe current chart stock is ${opts.ticker}. Use this stock only if the user request does not name a stock. An explicit stock in the user request takes precedence.`
    : "";
  // Explicit Stockbot research call with drafts appends the synthesis section; every other call keeps its exact existing user body.
  const isSynthesis = !opts.direct && opts.persona === "stockbot" && (opts.drafts?.length ?? 0) > 0;
  const user = opts.direct
    ? `USER REQUEST\n${opts.prompt}`
    : `USER REQUEST\n${opts.prompt}\n\nEVIDENCE\n${formatEvidence(opts.evidence)}` +
    (isSynthesis
      ? `\n\nSIBLING DRAFTS (untrusted sibling prose, not evidence. Never cite, quote, or invent a draft id. For factual support, cite only evidence row ids from the EVIDENCE section.)\n${formatDrafts(opts.drafts ?? [])}\n\nWeigh both sibling drafts against the EVIDENCE above.`
      : "");
  return [
    { role: "system", content: system + stockContext },
    { role: "user", content: user },
  ];
}

export async function reason(opts: ReasonOptions): Promise<{ text: string; usage: MuseUsage; missingEvidence?: string; claims: FinalClaim[]; confidence?: number }> {
  const { apiKey, model, base } = reasonEndpoint();
  const signal = reasonSignal(opts);
  // Each call uses a separate provider session, including concurrent personas.
  const session = crypto.randomUUID();
  const input = reasonInput(opts);
  const body = { model, input, stream: true };
  // Explicit personas have no per-fetch limit. The caller deadline still applies.
  const fetchSignal = (): AbortSignal =>
    opts.persona ? signal : AbortSignal.any([AbortSignal.timeout(FETCH_TIMEOUT_MS), signal]);
  try {
    return withInventory(await post(base, apiKey, session, "/responses", body, opts.onDelta, fetchSignal()));
  } catch (err) {
    // Cancelled or past deadline: no retry, no fallback.
    if (signal.aborted) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    // ponytail: 429 is a usage window, not a wrong route — retry the same endpoint
    // once after a short wait; only 404/405 (unsupported path) falls over.
    if (/(429|GoUsageLimit)/.test(msg)) {
      await sleep(5000, undefined, { signal });
      return withInventory(await post(base, apiKey, session, "/responses", body, opts.onDelta, fetchSignal()));
    }
    if (!/(404|405|500|502|503|504)/.test(msg)) throw err;
    return withInventory(
      await post(
        base,
        apiKey,
        session,
        "/chat/completions",
        { model, messages: input.map((m) => ({ role: m.role, content: m.content })), stream: true },
        opts.onDelta,
        fetchSignal(),
      ),
    );
  }
}
