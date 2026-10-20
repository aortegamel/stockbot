import { useEffect, useState } from "react";

type Dep = { ok?: unknown; required?: unknown };
type Tone = "checking" | "ok" | "warn" | "down";
type Status = { tone: Tone; failing: string[] };

const COLORS: Record<Tone, string> = { checking: "#93A0B8", ok: "#3FB950", warn: "#D29922", down: "#F85149" };

// Reads the /api/health body: red when the service or a required dependency fails, amber for an optional one.
export function healthStatus(body: unknown): Status {
  if (typeof body !== "object" || body === null || !("ok" in body) || body.ok !== true) return { tone: "down", failing: ["api"] };
  const deps = "deps" in body && typeof body.deps === "object" && body.deps !== null ? Object.entries(body.deps) : [];
  const failing = deps.filter(([, dep]: [string, Dep | null]) => dep?.ok !== true);
  const tone = failing.some(([, dep]: [string, Dep | null]) => dep?.required !== false) ? "down" : failing.length ? "warn" : "ok";
  return { tone, failing: failing.map(([name]) => name) };
}

export function HealthDot() {
  const [status, setStatus] = useState<Status>({ tone: "checking", failing: [] });
  useEffect(() => {
    const ctrl = new AbortController();
    fetch("/api/health", { signal: ctrl.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => setStatus(healthStatus(body)))
      .catch(() => {
        if (!ctrl.signal.aborted) setStatus(healthStatus(null));
      });
    return () => ctrl.abort();
  }, []);
  const label = status.failing.length ? `health ${status.tone}: ${status.failing.join(", ")} (run bun run doctor)` : `health ${status.tone}`;
  return (
    <span
      role="status"
      aria-label={label}
      title={label}
      style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: COLORS[status.tone] }}
    />
  );
}
