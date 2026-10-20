import { existsSync } from "node:fs";
import { join } from "node:path";
import { fingerprintKey } from "@/lib/env";

export async function GET(): Promise<Response> {
  const root =
    process.env.STOCKBOT_REPO_ROOT ??
    (process.cwd().endsWith("needle-harness") ? process.cwd().replace(/\/needle-harness$/, "") : process.cwd());
  const needleWeights = process.env.NEEDLE_WEIGHTS ?? null;
  const opencodeKey = process.env.OPENCODE_API_KEY ?? "";
  const weightsPresent =
    needleWeights !== null ? existsSync(needleWeights) : existsSync(join(process.cwd(), "..", "needle3.cact"));
  const env = (k: string): string => {
    const v = (process.env[k] ?? "").trim();
    return v.toLowerCase().startsWith("your") ? "" : v;
  };
  const on = (k: string): boolean => ["1", "true", "yes"].includes(env(k).toLowerCase());
  const all = (...keys: string[]): boolean => keys.every((k) => env(k) !== "");
  const venvPresent = existsSync(join(root, "venv/bin/python"));
  const deps: Record<string, { ok: boolean; required: boolean }> = {
    venv: { ok: venvPresent, required: true },
    needle: { ok: weightsPresent, required: true },
    sec: { ok: env("SEC_EDGAR_IDENTITY").includes("@"), required: true },
    jev: { ok: all("DECISION_API_FORMAT", "DECISION_API_URL", "DECISION_API_KEY", "DECISION_MODEL"), required: true },
    opencode: { ok: all("OPENCODE_API_KEY", "OPENCODE_URL", "OPENCODE_MODEL"), required: true },
  };
  if (on("EXA_ENABLED")) deps.exa = { ok: env("EXA_API_KEY") !== "", required: false };
  if (env("FINRA_CLIENT_ID") || env("FINRA_CLIENT_SECRET")) {
    deps.finra = { ok: all("FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET"), required: false };
  }
  if (on("BROKER_ENABLED")) deps.broker = { ok: true, required: false };
  return Response.json(
    {
      ok: true,
      port: process.env.STOCKBOT_API_PORT ?? process.env.PORT ?? "3000",
      hasOpencodeKey: Boolean(opencodeKey),
      needleWeights,
      weightsPresent,
      venvPresent,
      deps,
      // Best-effort: set by the Bun host after prewarmKernel().
      // Env-only so health never pulls the kernel singleton into its module graph.
      kernelPrewarmed: process.env.KERNEL_PREWARMED === "1",
      debug: ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase()),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
