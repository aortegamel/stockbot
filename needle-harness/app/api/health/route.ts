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
  return Response.json(
    {
      ok: true,
      port: process.env.STOCKBOT_API_PORT ?? process.env.PORT ?? "3000",
      hasOpencodeKey: Boolean(opencodeKey),
      needleWeights,
      weightsPresent,
      venvPresent: existsSync(join(root, "venv/bin/python")),
      // Best-effort: set by the Bun host after prewarmKernel().
      // Env-only so health never pulls the kernel singleton into its module graph.
      kernelPrewarmed: process.env.KERNEL_PREWARMED === "1",
      debug: ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase()),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
