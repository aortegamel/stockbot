import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { fingerprintKey, loadDotenvAuthoritative } from "../lib/env";

const HARNESS_DIR = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT_DIR = dirname(HARNESS_DIR);
const SERVER = join(HARNESS_DIR, "lib/needle/server.py");

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

const VENV_PYTHON = `${homedir()}/.cache/needle-harness/.needle/bin/python`;
const WARMUP_TIMEOUT_MS = 120_000;
const PORT = process.env.PORT ?? "3000";
const CATALOG = process.env.NEEDLE_CATALOG ?? join(HARNESS_DIR, ".needle-catalog.json");

// Next owns its subtree; the supervisor only marks an unhealthy web child.
const WEB_UNHEALTHY_MARKER = join(HARNESS_DIR, ".needle-web-unhealthy");
process.stdout.write(`needle [web]: harness ${HARNESS_DIR} → http://127.0.0.1:${PORT} (cwd ${process.cwd()})\n`);
process.stdout.write(dotenvLoaded ? `needle [web]: env loaded ${DOTENV}\n` : `needle [web]: no .env at ${DOTENV}, using shell env\n`);

function fail(msg: string): never {
  process.stderr.write(msg + "\n");
  process.exit(1);
}

if (!existsSync(VENV_PYTHON)) {
  fail("needle venv missing — run: bash scripts/setup-needle.sh");
}
process.stdout.write(`needle [ai]: venv ok ${VENV_PYTHON}\n`);

const pinned = process.env.NEEDLE_WEIGHTS;
const repoBlob = join(ROOT_DIR, "needle3.cact");
if (pinned) {
  const resolved = isAbsolute(pinned) ? pinned : join(HARNESS_DIR, pinned);
  process.env.NEEDLE_WEIGHTS = resolved;
  process.stdout.write(`needle [ai]: weights ${resolved} (${existsSync(resolved) ? "exists" : "missing"})\n`);
} else if (existsSync(repoBlob)) {
  process.stdout.write(`needle [ai]: weights ${repoBlob} (exists)\n`);
} else {
  process.stderr.write("weights not pinned — falling back to HF cache\n");
}

// Mirror of scheduler._JEV_REGISTRY_EXCLUDED: JEV meta tools, never a Needle tool.
const META_EXCLUDED: Record<string, true> = { call_tool: true, browse_tools: true, search_tools: true, list_tool_domains: true, describe_tool: true };

type DescribeEntry = { function?: { name?: unknown; description?: unknown; parameters?: unknown } };

function stripDescriptions(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(stripDescriptions);
  if (node && typeof node === "object") {
    return Object.fromEntries(
      Object.entries(node as Record<string, unknown>)
        .filter(([k]) => k !== "description")
        .map(([k, v]) => [k, stripDescriptions(v)]),
    );
  }
  return node;
}

async function fetchCatalog(): Promise<void> {
  process.stdout.write("needle [ai]: fetching tool catalog via tool_bridge describe…\n");
  const bridge = spawn(`${ROOT_DIR}/venv/bin/python`, [`${ROOT_DIR}/scripts/tool_bridge.py`], {
    cwd: ROOT_DIR,
    stdio: ["pipe", "pipe", "pipe"],
  });
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
        const parsed = JSON.parse(line) as { tools?: unknown };
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
    bridge.kill();
    fail(`needle catalog failed: ${e.message}\nbridge stderr tail: ${errTail || "(empty)"}`);
  });
  bridge.kill();
  let tools: unknown;
  try {
    tools = (JSON.parse(line) as { tools?: unknown }).tools;
  } catch {
    fail(`needle catalog failed: bad JSON ${line}\nbridge stderr tail: ${errTail || "(empty)"}`);
  }
  if (!Array.isArray(tools)) {
    fail(`needle catalog failed: describe tools not an array: ${line}\nbridge stderr tail: ${errTail || "(empty)"}`);
  }
  const catalog: { name: string; description: string; parameters: unknown }[] = [];
  for (const entry of tools as DescribeEntry[]) {
    const fn = entry?.function;
    if (!fn || typeof fn.name !== "string") continue;
    if (META_EXCLUDED[fn.name]) continue;
    const rawDesc = typeof fn.description === "string" && fn.description.trim() ? fn.description : fn.name;
    catalog.push({
      name: fn.name,
      // Slim by design: full describe text (~58KB catalog) exceeds the Needle init budget; truncated top-level descriptions ground routing (name-only misroutes), stripped params fit.
      description: rawDesc.slice(0, 150),
      parameters: fn.parameters && typeof fn.parameters === "object" ? stripDescriptions(fn.parameters) : { type: "object" },
    });
  }
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
  await writeFile(CATALOG, JSON.stringify(catalog, null, 2));
  process.stdout.write(`needle [ai]: catalog ok ${CATALOG} (${catalog.map((c) => c.name).join(",")})\n`);
}

await fetchCatalog();

try {
  await rm(WEB_UNHEALTHY_MARKER, { force: true });
} catch {
  // Stale marker stays; an unhealthy exit below overwrites it.
}

process.stdout.write("needle [kernel]: starting\n");
process.stdout.write("needle [jev]: starting\n");
process.stdout.write("needle [needle]: loading weights\n");
process.stdout.write(`needle [web]: starting next dev (cwd ${HARNESS_DIR}, PORT=${PORT})…\n`);
process.stdout.write("needle [web]: next dev output follows — wait for Ready…\n");
const dev = spawn("bun", ["run", "dev"], {
  cwd: HARNESS_DIR,
  stdio: "inherit",
  env: { ...process.env, PORT, STOCKBOT_REPO_ROOT: ROOT_DIR },
});
for (const sig of ["SIGINT", "SIGTERM"] as const) {
  process.on(sig, () => dev.kill(sig));
}
const { promise: exited, resolve: onExit } = Promise.withResolvers<{ code: number | null; signal: NodeJS.Signals | null }>();
dev.on("close", (code, signal) => onExit({ code, signal }));
const { code, signal } = await exited;
const exitCode = code ?? 1;
if (exitCode === 0) {
  process.stdout.write(`needle [web]: next dev exited with code ${exitCode}\n`);
} else {
  const detail = typeof code === "number" ? `code ${code}` : signal ? `signal ${signal}` : "signal exit";
  await writeFile(WEB_UNHEALTHY_MARKER, `unhealthy ${detail} at ${new Date().toISOString()}\n`);
  process.stderr.write(`needle [web]: unhealthy web child (${detail}); marker ${WEB_UNHEALTHY_MARKER}\n`);
  process.stderr.write(`needle [web]: next dev exited with code ${exitCode}\n`);
}
process.exit(exitCode);
