// Native Bun API host: dispatches the existing route handlers without a framework.
// Run: bun scripts/api.ts with cwd needle-harness. Listens on STOCKBOT_API_PORT (the actual
// port is written back to the env so /api/health reports it); PORT stays Vite. Importing this
// module only exposes dispatch: listening happens in startApiServer().
import type { Server } from "bun";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { POST as agentPOST } from "@/app/api/agent/route";
import { GET as healthGET } from "@/app/api/health/route";
import { GET as researchGET } from "@/app/api/research/[...path]/route";
import { GET as watchlistGET } from "@/app/api/watchlist/route";
import { kernelRouter, prewarmKernel } from "@/lib/agent/kernel";
import { fingerprintKey, loadDotenvAuthoritative } from "@/lib/env";

const HARNESS_DIR = dirname(dirname(fileURLToPath(import.meta.url)));
const DOTENV = join(dirname(HARNESS_DIR), ".env");
// .env is authoritative: stale inherited exports (daemon broker, shell) lose. Loaded at module
// scope before startApiServer() reads env; the only module-scope env read elsewhere is
// STOCKBOT_REPO_ROOT in the kernel, which the cwd default already resolves for this launch.
const { loaded: dotenvLoaded, overridden } = loadDotenvAuthoritative(DOTENV);
for (const o of overridden) {
  process.stdout.write(`api [env]: override ${o.key} inherited ${o.oldRedacted} → .env ${o.newRedacted}\n`);
}
process.stdout.write(dotenvLoaded ? `api [env]: env loaded ${DOTENV}\n` : `api [env]: no .env at ${DOTENV}, using shell env\n`);
if (process.env.OPENCODE_API_KEY) {
  process.stdout.write(`api [env]: opencode identity fp=${fingerprintKey(process.env.OPENCODE_API_KEY)}\n`);
}

// Former cactus API paths owned by Bun. Wrong methods are 405. Unknown paths are 404.
function dispatchMethod(req: Request, method: string, handler: (req: Request) => Promise<Response>): Promise<Response> | Response {
  if (req.method !== method) return Response.json({ error: "method not allowed" }, { status: 405 });
  return handler(req);
}

export async function dispatch(req: Request): Promise<Response> {
  const path = new URL(req.url).pathname;
  switch (path) {
    case "/api/agent":
    case "/api/agent/":
      return dispatchMethod(req, "POST", agentPOST);
    case "/api/watchlist":
    case "/api/watchlist/":
      return dispatchMethod(req, "GET", watchlistGET);
    case "/api/health":
    case "/api/health/":
      return dispatchMethod(req, "GET", healthGET);
  }
  if (path === "/api/research") return researchGET(req);
  if (path.startsWith("/api/research/")) return dispatchMethod(req, "GET", researchGET);
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
  const server = Bun.serve({ hostname: "0.0.0.0", port: requested, idleTimeout: 0, fetch: dispatch });
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
  console.log(`[web] bun api listening http://0.0.0.0:${server.port ?? requested} (all IPv4 interfaces, no login)`);
  return server;
}

if (import.meta.main) {
  await startApiServer();
}
