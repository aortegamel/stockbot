// Native Bun API host: dispatches the existing route handlers without a framework.
// Run: bun scripts/api.ts with cwd needle-harness. Listens on STOCKBOT_API_PORT (the actual
// port is written back to the env so /api/health reports it); PORT stays Vite. Importing this
// module only exposes dispatch: listening happens in startApiServer().
import type { Server } from "bun";
import { POST as agentPOST } from "@/app/api/agent/route";
import { GET as healthGET } from "@/app/api/health/route";
import { GET as researchGET } from "@/app/api/research/[...path]/route";
import { kernelRouter, prewarmKernel } from "@/lib/agent/kernel";

// Former cactus API paths owned by Bun. Wrong methods are 405; unknown paths are 404.
export async function dispatch(req: Request): Promise<Response> {
  const path = new URL(req.url).pathname;
  if (path === "/api/agent" || path === "/api/agent/") {
    if (req.method !== "POST") return Response.json({ error: "method not allowed" }, { status: 405 });
    return agentPOST(req);
  }
  if (path === "/api/health" || path === "/api/health/") {
    if (req.method !== "GET") return Response.json({ error: "method not allowed" }, { status: 405 });
    return healthGET();
  }
  if (path === "/api/research") return researchGET(req);
  if (path.startsWith("/api/research/")) {
    if (req.method !== "GET") return Response.json({ error: "method not allowed" }, { status: 405 });
    return researchGET(req);
  }
  return Response.json({ error: "not found" }, { status: 404 });
}

export function resolveApiPort(): number {
  const n = Number(process.env.STOCKBOT_API_PORT ?? "3000");
  return Number.isInteger(n) && n >= 0 && n <= 65535 ? n : 3000;
}

// Startup prewarm: warms the kernel worker so request #1 == request #500.
// Never throws; prewarm failure only logs.
export async function registerPrewarm(): Promise<void> {
  try {
    await prewarmKernel();
    process.env.KERNEL_PREWARMED = "1";
    console.log("[web] prewarm kernel ready");
  } catch (e) {
    console.error(`[web] kernel prewarm failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export async function startApiServer(): Promise<Server<undefined>> {
  const requested = resolveApiPort();
  const server = Bun.serve({ hostname: "127.0.0.1", port: requested, idleTimeout: 0, fetch: dispatch });
  process.env.STOCKBOT_API_PORT = String(server.port ?? requested);
  await registerPrewarm();
  for (const sig of ["SIGINT", "SIGTERM"] as const) {
    process.on(sig, () => {
      // Kernel owns its child; close() kills it. The server socket closes via exit.
      try {
        kernelRouter.close();
      } catch { }
      try {
        void server.stop();
      } catch { }
      process.exit(0);
    });
  }
  console.log(`[web] bun api listening http://127.0.0.1:${server.port ?? requested}`);
  return server;
}

if (import.meta.main) {
  await startApiServer();
}
