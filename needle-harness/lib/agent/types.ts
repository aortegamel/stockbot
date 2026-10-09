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
  | "incomplete_evidence"
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
  | "research_loop_detected";

// ponytail: research caps live in the kernel (None=null=unlimited); no TS budget mirror (DISPLAY_CHAR_LIMIT in loop.ts is a model-view bound, never termination).

type JSONSchema = {
  type: string;
  properties?: Record<string, unknown>;
  required?: string[];
  [k: string]: unknown;
};

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
};

export type ToolResult = { ok: true; evidence: Evidence } | { ok: false; error: string; category: FailureCategory; retryable: boolean };

export type Tool = {
  description: string;
  parameters: JSONSchema;
  execute(args: Record<string, unknown>, opts: { sessionId: string }): Promise<ToolResult>;
};

export type AgentEvent =
  | { type: "agent_start"; prompt: string }
  | { type: "needle_decision"; step: number; tool: string | null; arguments: Record<string, unknown>; confidence: number | null }
  | { type: "tool_start"; tool: string }
  | { type: "tool_result"; tool: string; evidenceId?: string; preview: string }
  | { type: "progress"; stage: string; detail?: Record<string, unknown> }
  | { type: "reasoning_start"; model: string }
  | { type: "answer_delta"; text: string }
  | { type: "done"; metrics: Metrics }
  | { type: "error"; message: string }
  | { type: "tool_failed"; tool: string; category: FailureCategory; preview: string }
  | { type: "failed"; category: FailureCategory; message: string };

export type Metrics = {
  totalMs: number;
  needle: { calls: number; totalMs: number; escalations: number };
  tools: { calls: number; totalMs: number };
  muse: {
    calls: number;
    inputTokens?: number;
    outputTokens?: number;
    cachedTokens?: number;
    totalMs: number;
  };
  evidence: { count: number; characters: number };
  failures: Partial<Record<FailureCategory, number>>;
};

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
