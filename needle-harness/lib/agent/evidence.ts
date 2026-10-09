import { createHash } from "node:crypto";
import type { Evidence } from "./types";

const DISPLAY_CONTENT_LIMIT = 8000;

function stable(value: unknown): string {
  if (value === undefined || typeof value === "function") return "null";
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stable(v)}`).join(",")}}`;
}

export function makeEvidence(
  source: string,
  content: string,
  opts?: { title?: string; url?: string; sourceHandle?: Record<string, unknown>; sourceRefs?: Record<string, unknown> },
): Evidence {
  const fullHash = createHash("sha256").update(content).digest("hex");
  const id = `ev:${createHash("sha256").update(`${source}\n${stable(opts?.sourceHandle ?? opts?.sourceRefs ?? null)}\n${fullHash}`).digest("hex").slice(0, 16)}`;
  const truncated = content.length > DISPLAY_CONTENT_LIMIT;
  return {
    id,
    source,
    title: opts?.title,
    url: opts?.url,
    retrievedAt: new Date().toISOString(),
    content: truncated ? `${content.slice(0, DISPLAY_CONTENT_LIMIT)}\n…[truncated]` : content,
    ...(opts?.sourceHandle !== undefined ? { sourceHandle: opts.sourceHandle } : {}),
    ...(opts?.sourceRefs !== undefined ? { sourceRefs: opts.sourceRefs } : {}),
  };
}

