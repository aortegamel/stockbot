import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fingerprintKey } from "@/lib/env";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const needleWeights = process.env.NEEDLE_WEIGHTS ?? null;
  const opencodeKey = process.env.OPENCODE_API_KEY ?? "";
  const weightsPresent =
    needleWeights !== null ? existsSync(needleWeights) : existsSync(join(process.cwd(), "..", "needle3.cact"));
  return Response.json(
    {
      ok: true,
      port: process.env.PORT ?? "3000",
      hasOpencodeKey: Boolean(opencodeKey),
      opencodeFingerprint: opencodeKey ? fingerprintKey(opencodeKey) : null,
      needleWeights,
      weightsPresent,
      venvPresent: existsSync(`${homedir()}/.cache/needle-harness/.needle/bin/python`),
      // Best-effort: set by instrumentation register() after prewarmKernel().
      // Env-only so health never pulls the kernel singleton into its module graph.
      kernelPrewarmed: process.env.KERNEL_PREWARMED === "1",
      debug: ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase()),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
