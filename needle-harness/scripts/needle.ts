import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fingerprintKey, loadDotenvAuthoritative } from "../lib/env";
import { pickFreePort } from "../lib/port";

const HARNESS_DIR = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT_DIR = dirname(HARNESS_DIR);
const WEB_DIR = join(ROOT_DIR, "web");
const SERVER = join(HARNESS_DIR, "lib/needle/server.py");

const REPO_PYTHON = join(ROOT_DIR, "venv/bin/python");
const WARMUP_TIMEOUT_MS = 120_000;
const owned = new Set<ChildProcess>();
// Services spawn detached so one signal reaches the whole child tree;
// group kill falls back to direct kill when the group is already gone.
export function killService(child: ChildProcess, sig: NodeJS.Signals): void {
  const pid = child.pid;
  if (typeof pid === "number" && pid > 0 && child.exitCode === null) {
    try {
      process.kill(-pid, sig);
      return;
    } catch {
      // Already gone or no group on this platform; direct kill below.
    }
  }
  try {
    child.kill(sig);
  } catch {
    // Already gone.
  }
}
let gotSignal: NodeJS.Signals | null = null;

function fail(msg: string): never {
  for (const child of [...owned]) killService(child, "SIGTERM");
  process.stderr.write(msg + "\n");
  process.exit(1);
}

// Mirror of scheduler._JEV_REGISTRY_EXCLUDED: JEV meta tools, never a Needle tool.
const META_EXCLUDED: Record<string, true> = { call_tool: true, browse_tools: true, search_tools: true, list_tool_domains: true, describe_tool: true };

type DescribeEntry = { function?: { name?: unknown; description?: unknown; parameters?: unknown } };

export async function readBridgeReply(bridge: ChildProcess): Promise<{ line: string; errTail: string }> {
  const { promise, resolve, reject } = Promise.withResolvers<string>();
  let buf = "";
  let errTail = "";
  const timer = setTimeout(() => reject(new Error("timeout")), WARMUP_TIMEOUT_MS);
  bridge.stdout?.on("data", (d: Buffer) => {
    buf += d.toString();
    let i: number;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      try {
        const parsed: unknown = JSON.parse(line);
        if (parsed && typeof parsed === "object" && "tools" in parsed) {
          clearTimeout(timer);
          resolve(line);
          break;
        }
      } catch {
        // Non-JSON log line before the describe reply — skip.
      }
    }
  });
  bridge.on("error", reject);
  bridge.on("close", (code) => {
    clearTimeout(timer);
    reject(new Error(`bridge exited ${code ?? "unknown"} before describe reply`));
  });
  bridge.stderr?.on("data", (d: Buffer) => {
    const s = d.toString();
    process.stderr.write(s);
    errTail = (errTail + s).slice(-2000);
  });
  bridge.stdin?.write('{"id":"describe-1","op":"describe"}\n');
  const line = await promise.catch((e: Error) => {
    killService(bridge, "SIGTERM");
    owned.delete(bridge);
    fail(`needle catalog failed: ${e.message}\nbridge stderr tail: ${errTail || "(empty)"}`);
  });
  return { line, errTail };
}

function catalogEntries(entries: DescribeEntry[]): { name: string; description: string; parameters: unknown }[] {
  const catalog: { name: string; description: string; parameters: unknown }[] = [];
  for (const entry of entries) {
    const fn = entry?.function;
    // Invalid entries and JEV meta tools both drop out of the Needle catalog.
    if (!fn || typeof fn.name !== "string" || META_EXCLUDED[fn.name]) continue;
    const rawDesc = typeof fn.description === "string" && fn.description.trim() ? fn.description : fn.name;
    catalog.push({
      name: fn.name,
      description: rawDesc,
      parameters: fn.parameters && typeof fn.parameters === "object" ? fn.parameters : { type: "object" },
    });
  }
  return catalog;
}

function readCatalogTools(line: string, errTail: string): DescribeEntry[] {
  let tools: unknown;
  try {
    const body: unknown = JSON.parse(line);
    tools = body && typeof body === "object" && "tools" in body ? body.tools : undefined;
  } catch {
    fail(`needle catalog failed: bad JSON ${line}\nbridge stderr tail: ${errTail || "(empty)"}`);
  }
  if (!Array.isArray(tools)) {
    fail(`needle catalog failed: describe tools not an array: ${line}\nbridge stderr tail: ${errTail || "(empty)"}`);
  }
  return tools as DescribeEntry[];
}

export function parseCatalog(line: string, errTail: string) {
  // Bridge describe replies contain untrusted JSON. Recheck each consumed field.
  const catalog = catalogEntries(readCatalogTools(line, errTail));
  if (catalog.length === 0) {
    fail("needle catalog failed: describe returned no usable tools");
  }
  for (const name of ["search_web", "find_sec_entities", "search_sec_filings", "get_sec_document"]) {
    if (!catalog.some((c) => c.name === name)) {
      fail(`needle catalog failed: required tool "${name}" missing from describe output`);
    }
  }
  if (!catalog.some((c) => c.name === "get_current_time")) {
    catalog.push(
      { name: "get_current_time", description: "Get the current UTC date and time", parameters: { type: "object", properties: {} } },
    );
  }
  return catalog;
}

export async function fetchCatalog(catalogPath: string, spawnBridge: typeof spawn = spawn): Promise<void> {
  process.stdout.write("needle [ai]: fetching tool catalog via tool_bridge describe…\n");
  const bridge = spawnBridge(REPO_PYTHON, [`${ROOT_DIR}/scripts/tool_bridge.py`], {
    cwd: ROOT_DIR,
    stdio: ["pipe", "pipe", "pipe"],
    detached: true,
  });
  owned.add(bridge);
  bridge.on("close", () => owned.delete(bridge));
  const { line, errTail } = await readBridgeReply(bridge);
  killService(bridge, "SIGTERM");
  owned.delete(bridge);
  if (gotSignal) process.exit(1);
  const catalog = parseCatalog(line, errTail);
  await writeFile(catalogPath, JSON.stringify(catalog, null, 2));
  process.stdout.write(`needle [ai]: catalog ok ${catalogPath} (${catalog.map((c) => c.name).join(",")})\n`);
}

export async function runAssets(childEnv: NodeJS.ProcessEnv, spawnAssets: typeof spawn = spawn): Promise<void> {
  process.stdout.write("needle [web]: copying pdfjs assets…\n");
  const child = spawnAssets("bun", ["scripts/copy-pdfjs-assets.ts"], {
    cwd: HARNESS_DIR,
    stdio: "inherit",
    env: childEnv,
    detached: true,
  });
  owned.add(child);
  const code = await new Promise<number | null>((resolve) => {
    child.on("error", () => resolve(1));
    child.on("close", (c) => resolve(c));
  });
  owned.delete(child);
  if (gotSignal) process.exit(1);
  if (code !== 0) fail(`needle assets failed: copy-pdfjs-assets exited ${code ?? "signal"}`);
}

type ServiceName = "api" | "market" | "web";
type FirstExit = { name: ServiceName; child: ChildProcess; code: number | null; signal: NodeJS.Signals | null };

export function waitForFirstExit(services: { name: ServiceName; child: ChildProcess }[]): Promise<FirstExit> {
  const { promise, resolve } = Promise.withResolvers<FirstExit>();
  let firstSeen = false;
  function noteExit(name: ServiceName, child: ChildProcess, code: number | null, signal: NodeJS.Signals | null): void {
    if (firstSeen) return;
    firstSeen = true;
    resolve({ name, child, code, signal });
  }
  for (const { name, child } of services) {
    child.on("error", (err) => {
      process.stderr.write(`needle [${name}]: spawn failed ${err instanceof Error ? err.message : String(err)}\n`);
      noteExit(name, child, 1, null);
    });
    child.on("close", (code, signal) => noteExit(name, child, code, signal));
  }
  return promise;
}

if (import.meta.main) {
  if (!existsSync(SERVER)) {
    process.stderr.write(`needle: harness not found at ${HARNESS_DIR}\n`);
    process.exit(1);
  }

  const DOTENV = join(ROOT_DIR, ".env");
  // .env is authoritative: stale inherited exports (daemon broker, shell) lose.
  const { loaded: dotenvLoaded, overridden } = loadDotenvAuthoritative(DOTENV);
  for (const o of overridden) {
    process.stdout.write(`needle [env]: override ${o.key} inherited ${o.oldRedacted} → .env ${o.newRedacted}\n`);
  }
  if (process.env.OPENCODE_API_KEY) {
    process.stdout.write(`needle [env]: opencode identity fp=${fingerprintKey(process.env.OPENCODE_API_KEY)}\n`);
  }

  // Strictly increasing preferred ports so three pickFreePort calls never duplicate before bind.
  const vitePreferred = Number(process.env.PORT ?? "3000");
  const PORT_NUM = await pickFreePort(vitePreferred);
  const apiRaw = Number(process.env.STOCKBOT_API_PORT ?? String(PORT_NUM + 1));
  const apiPreferred = Number.isInteger(apiRaw) && apiRaw > PORT_NUM ? apiRaw : PORT_NUM + 1;
  const API_NUM = await pickFreePort(apiPreferred);
  const marketRaw = Number(process.env.STOCKBOT_MARKET_PORT ?? String(API_NUM + 1));
  const marketPreferred = Number.isInteger(marketRaw) && marketRaw > API_NUM ? marketRaw : API_NUM + 1;
  const MARKET_NUM = await pickFreePort(marketPreferred);
  const PORT = String(PORT_NUM);
  const STOCKBOT_API_PORT = String(API_NUM);
  const STOCKBOT_MARKET_PORT = String(MARKET_NUM);
  const CATALOG = process.env.NEEDLE_CATALOG ?? join(HARNESS_DIR, ".needle-catalog.json");
  process.env.PORT = PORT;
  process.env.STOCKBOT_API_PORT = STOCKBOT_API_PORT;
  process.env.STOCKBOT_MARKET_PORT = STOCKBOT_MARKET_PORT;
  process.env.STOCKBOT_REPO_ROOT = ROOT_DIR;
  process.env.NEEDLE_CATALOG = CATALOG;
  const childEnv: NodeJS.ProcessEnv = {
    ...process.env,
    PORT,
    STOCKBOT_API_PORT,
    STOCKBOT_MARKET_PORT,
    STOCKBOT_REPO_ROOT: ROOT_DIR,
    NEEDLE_CATALOG: CATALOG,
  };

  // Vite owns its subtree; the supervisor only marks an unhealthy web child.
  const WEB_UNHEALTHY_MARKER = join(HARNESS_DIR, ".needle-web-unhealthy");
  process.stdout.write(`needle [web]: harness ${HARNESS_DIR} → http://0.0.0.0:${PORT} (all IPv4 interfaces, no login, cwd ${process.cwd()})\n`);
  process.stdout.write(dotenvLoaded ? `needle [web]: env loaded ${DOTENV}\n` : `needle [web]: no .env at ${DOTENV}, using shell env\n`);

  for (const sig of ["SIGINT", "SIGTERM"] as const) {
    process.on(sig, () => {
      gotSignal ??= sig;
      for (const child of [...owned]) killService(child, sig);
      if (owned.size === 0) process.exit(1);
    });
  }
  if (!existsSync(REPO_PYTHON)) {
    fail("Repository venv missing. Run: bun run setup");
  }
  process.stdout.write(`needle [ai]: venv ok ${REPO_PYTHON}\n`);

  const pinned = process.env.NEEDLE_WEIGHTS;
  const repoBlob = join(ROOT_DIR, "needle3.cact");
  if (pinned) {
    const resolved = isAbsolute(pinned) ? pinned : join(HARNESS_DIR, pinned);
    process.env.NEEDLE_WEIGHTS = resolved;
    childEnv.NEEDLE_WEIGHTS = resolved;
    process.stdout.write(`needle [ai]: weights ${resolved} (${existsSync(resolved) ? "exists" : "missing"})\n`);
  } else if (existsSync(repoBlob)) {
    process.stdout.write(`needle [ai]: weights ${repoBlob} (exists)\n`);
  } else {
    process.stderr.write("weights not pinned — falling back to HF cache\n");
  }
  await fetchCatalog(CATALOG);
  if (gotSignal) process.exit(1);

  try {
    await rm(WEB_UNHEALTHY_MARKER, { force: true });
  } catch {
    // Stale marker stays; an unhealthy exit below overwrites it.
  }

  process.stdout.write("needle [kernel]: starting\n");
  process.stdout.write("needle [jev]: starting\n");
  process.stdout.write("needle [needle]: loading weights\n");
  await runAssets(childEnv);
  if (gotSignal) process.exit(1);
  process.stdout.write(`needle [api]: starting bun scripts/api.ts (cwd ${HARNESS_DIR}, STOCKBOT_API_PORT=${STOCKBOT_API_PORT})…\n`);
  process.stdout.write(`needle [market]: starting uvicorn api.main:app (cwd ${ROOT_DIR}, STOCKBOT_MARKET_PORT=${STOCKBOT_MARKET_PORT})…\n`);
  process.stdout.write(`needle [web]: starting vite dev (cwd ${WEB_DIR}, PORT=${PORT})…\n`);
  process.stdout.write("needle [web]: vite dev output follows — wait for Ready…\n");
  const api = spawn("bun", [join(HARNESS_DIR, "scripts/api.ts")], {
    cwd: HARNESS_DIR,
    stdio: "inherit",
    env: childEnv,
    detached: true,
  });
  owned.add(api);
  const market = spawn(REPO_PYTHON, ["-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", STOCKBOT_MARKET_PORT], {
    cwd: ROOT_DIR,
    stdio: "inherit",
    env: childEnv,
    detached: true,
  });
  owned.add(market);
  if (process.env.STOCKBOT_THESIS_MONITOR === "1") {
    const thesisRoot = resolve(ROOT_DIR, process.env.STOCKBOT_DATA_DIR ?? "data", "thesis");
    for (const d of existsSync(thesisRoot) ? readdirSync(thesisRoot, { withFileTypes: true }) : []) {
      if (!d.isDirectory()) continue;
      owned.add(spawn(REPO_PYTHON, ["cli.py", "thesis", "monitor", d.name, "--interval-seconds", "900"], { cwd: ROOT_DIR, stdio: "inherit", env: childEnv, detached: true }));
    }
  }
  const web = spawn("bun", ["x", "vite", "--host", "0.0.0.0", "--port", PORT, "--strictPort"], {
    cwd: WEB_DIR,
    stdio: "inherit",
    env: childEnv,
    detached: true,
  });
  owned.add(web);
  for (const child of owned) {
    child.on("close", () => owned.delete(child));
  }
  const first = await waitForFirstExit([
    { name: "api", child: api },
    { name: "market", child: market },
    { name: "web", child: web },
  ]);
  for (const child of [api, market, web]) {
    if (child !== first.child) killService(child, gotSignal ?? "SIGTERM");
  }
  await Promise.all(
    [api, market, web].map(
      (child) =>
        new Promise<void>((resolve) => {
          if (child.exitCode !== null || child.signalCode !== null) resolve();
          else {
            child.on("close", () => resolve());
            child.on("error", () => resolve());
          }
        }),
    ),
  );
  owned.clear();
  if (gotSignal !== null) process.exit(1);
  const exitCode = first.code ?? 1;
  if (first.name === "web") {
    if (exitCode === 0) {
      process.stdout.write(`needle [web]: vite dev exited with code ${exitCode}\n`);
    } else {
      const detail = typeof first.code === "number" ? `code ${first.code}` : first.signal ? `signal ${first.signal}` : "signal exit";
      await writeFile(WEB_UNHEALTHY_MARKER, `unhealthy ${detail} at ${new Date().toISOString()}\n`);
      process.stderr.write(`needle [web]: unhealthy web child (${detail}); marker ${WEB_UNHEALTHY_MARKER}\n`);
      process.stderr.write(`needle [web]: vite dev exited with code ${exitCode}\n`);
    }
  } else {
    const detail = typeof first.code === "number" ? `code ${first.code}` : first.signal ? `signal ${first.signal}` : "spawn failed";
    process.stderr.write(`needle [${first.name}]: exited (${detail}); stopping remaining services\n`);
  }
  process.exit(exitCode);
}
