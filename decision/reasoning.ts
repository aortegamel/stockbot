// REASONING_EFFORT: the model's reasoning effort for every remote model call, on any provider.
// A set value passes through unchanged in the protocol's own request field, so each
// provider's own levels work. Unset or empty sends no field: the provider default applies.
// A bad shape throws before any request. Python twin: app/config.py reasoning_fields.
const SHAPE = /^[a-z][a-z0-9_-]{0,31}$/;

export function reasoningFields(protocol: "responses" | "chat_completions" | "decisions"): Record<string, unknown> {
  const effort = process.env.REASONING_EFFORT?.trim();
  if (!effort) return {};
  if (!SHAPE.test(effort)) throw new Error(`REASONING_EFFORT must match ${SHAPE.source}`);
  if (protocol === "responses") return { reasoning: { effort } };
  if (protocol === "chat_completions") return { reasoning_effort: effort };
  // The Decisions API request schema (DecisionsRequest) documents no reasoning field: send none.
  return {};
}
