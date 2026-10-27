// Concrete effort values pass unchanged. Auto, unset, and empty use the provider default.
// Research answer calls can supply a JEV-selected override.
// Python twin: app/config.py reasoning_fields.
const SHAPE = /^[a-z][a-z0-9_-]{0,31}$/;

export function reasoningFields(protocol: "responses" | "chat_completions" | "decisions", override?: string): Record<string, unknown> {
  const effort = (override ?? process.env.REASONING_EFFORT)?.trim();
  if (!effort || effort === "auto") return {};
  if (!SHAPE.test(effort)) throw new Error(`REASONING_EFFORT must match ${SHAPE.source}`);
  if (protocol === "responses") return { reasoning: { effort } };
  if (protocol === "chat_completions") return { reasoning_effort: effort };
  // The Decisions API request schema (DecisionsRequest) documents no reasoning field: send none.
  return {};
}
