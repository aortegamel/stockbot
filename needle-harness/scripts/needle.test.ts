import { afterEach, describe, expect, mock, spyOn, test, vi } from "bun:test";
import { spawn, type ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildWeb, fetchCatalog, killService, monitoredTheses, parseCatalog, readBridgeReply, runAssets, waitForFirstExit, webSettings } from "./needle";

const HARNESS_DIR = join(import.meta.dir, "..");
const ROOT_DIR = join(HARNESS_DIR, "..");
const REQUIRED = ["search_web", "find_sec_entities", "search_sec_filings", "get_sec_document"];

const requiredEntries = () => REQUIRED.map((name) => ({ function: { name } }));

/** Bridge and service child stand-in: writable streams, recorded signals, EventEmitter events. */
class StubChild extends EventEmitter {
  readonly written: string[] = [];
  readonly killCalls: NodeJS.Signals[] = [];
  pid: number | undefined;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  readonly stdin = {
    write: (data: string): boolean => {
      this.written.push(data);
      return true;
    },
  };
  readonly stdout = {
    on: (_event: "data", listener: (chunk: Buffer) => void): void => {
      this.stdoutListeners.push(listener);
    },
  };
  readonly stderr = {
    on: (_event: "data", listener: (chunk: Buffer) => void): void => {
      this.stderrListeners.push(listener);
    },
  };
  private readonly stdoutListeners: ((chunk: Buffer) => void)[] = [];
  private readonly stderrListeners: ((chunk: Buffer) => void)[] = [];

  constructor(pid?: number) {
    super();
    this.pid = pid;
  }

  kill(signal?: NodeJS.Signals): boolean {
    this.killCalls.push(signal ?? "SIGTERM");
    return true;
  }

  send(chunk: string): void {
    for (const listener of this.stdoutListeners) listener(Buffer.from(chunk));
  }

  sendErr(chunk: string): void {
    for (const listener of this.stderrListeners) listener(Buffer.from(chunk));
  }
}

const asChild = (stub: StubChild): ChildProcess => stub as unknown as ChildProcess;

/**
  * Observes a rejection eagerly and returns its message.
  * Bun stalls the run when a `rejects` matcher subscribes after the rejection already landed,
  * so this handler attaches at call time, like the pending-work observers in client.test.ts.
  */
async function exitFailure(pending: Promise<unknown>): Promise<string> {
  const error: unknown = await pending.then(
    () => null,
    (reason: unknown) => reason,
  );
  if (!(error instanceof Error)) throw new Error(`expected an Error rejection, got ${String(error)}`);
  return error.message;
}

function spawnRecorder(stub: StubChild): { spawnFn: typeof spawn; calls: unknown[][] } {
  const calls: unknown[][] = [];
  const spawnFn = ((...args: unknown[]) => {
    calls.push(args);
    return stub;
  }) as unknown as typeof spawn;
  return { spawnFn, calls };
}

function captureOutput() {
  const stdout = spyOn(process.stdout, "write").mockReturnValue(true);
  const stderr = spyOn(process.stderr, "write").mockReturnValue(true);
  const exit = spyOn(process, "exit").mockImplementation((code) => {
    throw new Error(`process.exit(${code})`);
  });
  return { stdout, stderr, exit };
}

afterEach(() => {
  mock.restore();
  vi.useRealTimers();
});

describe("Needle catalog parsing", () => {
  test("keeps order, descriptions, schemas, and duplicates while dropping invalid and meta entries", () => {
    const tools = [
      null,
      {},
      { function: null },
      { function: { name: 42 } },
      ...["call_tool", "browse_tools", "search_tools", "list_tool_domains", "describe_tool"].map((name) => ({ function: { name } })),
      { function: { name: "search_web", description: "  Search the web  ", parameters: { type: "object", required: ["query"] } } },
      { function: { name: "find_sec_entities", description: " \n", parameters: null } },
      { function: { name: "search_sec_filings", description: 7, parameters: [] } },
      { function: { name: "get_sec_document", parameters: false } },
      { function: { name: "search_web", description: "Second search", parameters: 42 } },
    ];
    expect(parseCatalog(JSON.stringify({ tools }), "")).toEqual([
      { name: "search_web", description: "  Search the web  ", parameters: { type: "object", required: ["query"] } },
      { name: "find_sec_entities", description: "find_sec_entities", parameters: { type: "object" } },
      { name: "search_sec_filings", description: "search_sec_filings", parameters: [] },
      { name: "get_sec_document", description: "get_sec_document", parameters: { type: "object" } },
      { name: "search_web", description: "Second search", parameters: { type: "object" } },
      { name: "get_current_time", description: "Get the current UTC date and time", parameters: { type: "object", properties: {} } },
    ]);
  });

  test("keeps a bridge-provided clock tool instead of appending a second one", () => {
    const clock = { name: "get_current_time", description: "Bridge clock", parameters: { custom: true } };
    const catalog = parseCatalog(JSON.stringify({ tools: [{ function: clock }, ...requiredEntries()] }), "");
    expect(catalog[0]).toEqual(clock);
    expect(catalog.map((tool) => tool.name)).toEqual(["get_current_time", ...REQUIRED]);
  });

  test("reports malformed replies with the bridge stderr tail", () => {
    const output = captureOutput();
    for (const [line, detail] of [
      ["not-json", "bad JSON"],
      ["42", "describe tools not an array:"],
      ["null", "describe tools not an array:"],
      ["{}", "describe tools not an array:"],
      ['{"tools":{}}', "describe tools not an array:"],
    ]) {
      expect(() => parseCatalog(line!, "bridge diagnostic")).toThrow("process.exit(1)");
      expect(output.stderr).toHaveBeenLastCalledWith(`needle catalog failed: ${detail} ${line}\nbridge stderr tail: bridge diagnostic\n`);
    }
    expect(() => parseCatalog("broken", "")).toThrow("process.exit(1)");
    expect(output.stderr).toHaveBeenLastCalledWith("needle catalog failed: bad JSON broken\nbridge stderr tail: (empty)\n");
  });

  test("rejects a catalog with no usable tools", () => {
    const output = captureOutput();
    for (const tools of [[], [null, {}, { function: { name: "call_tool" } }]]) {
      expect(() => parseCatalog(JSON.stringify({ tools }), "")).toThrow("process.exit(1)");
      expect(output.stderr).toHaveBeenLastCalledWith("needle catalog failed: describe returned no usable tools\n");
    }
  });

  test("reports the first missing required tool", () => {
    const output = captureOutput();
    for (let index = 0; index < REQUIRED.length; index += 1) {
      const tools = [...requiredEntries().slice(0, index), ...requiredEntries().slice(index + 1), { function: { name: "extra" } }];
      expect(() => parseCatalog(JSON.stringify({ tools }), "")).toThrow("process.exit(1)");
      expect(output.stderr).toHaveBeenLastCalledWith(`needle catalog failed: required tool "${REQUIRED[index]}" missing from describe output\n`);
    }
  });
});

describe("Needle bridge reader", () => {
  test("skips logs and unrelated JSON, joins chunks, and returns the first describe reply", async () => {
    const output = captureOutput();
    const stub = new StubChild();
    const pending = readBridgeReply(asChild(stub));
    stub.send('log\n{bad}\nnull\n42\n{"ready":true}\n{"to');
    stub.sendErr("diagnostic");
    stub.send('ols":[]}\n{"tools":["later"]}\n');
    expect(await pending).toEqual({ line: '{"tools":[]}', errTail: "diagnostic" });
    expect(stub.written).toEqual(['{"id":"describe-1","op":"describe"}\n']);
    expect(output.stderr).toHaveBeenCalledWith("diagnostic");
    stub.emit("close", 0);
  });

  test("waits for a newline and fails at the 120s warmup deadline with the last stderr tail", async () => {
    vi.useFakeTimers();
    const output = captureOutput();
    const stub = new StubChild();
    const failure = exitFailure(readBridgeReply(asChild(stub)));
    stub.sendErr("discard" + "x".repeat(2000));
    stub.send('{"tools":[]}');
    vi.advanceTimersByTime(119_999);
    expect(output.exit).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(await failure).toBe("process.exit(1)");
    expect(output.stderr).toHaveBeenLastCalledWith(`needle catalog failed: timeout\nbridge stderr tail: ${"x".repeat(2000)}\n`);
    expect(stub.killCalls).toEqual(["SIGTERM"]);
    stub.emit("close", null);
  });

  test("reports bridge exit codes, missing codes, and spawn errors", async () => {
    const output = captureOutput();
    for (const code of [7, null]) {
      const stub = new StubChild();
      const failure = exitFailure(readBridgeReply(asChild(stub)));
      stub.emit("close", code);
      expect(await failure).toBe("process.exit(1)");
      expect(output.stderr).toHaveBeenLastCalledWith(`needle catalog failed: bridge exited ${code ?? "unknown"} before describe reply\nbridge stderr tail: (empty)\n`);
      expect(stub.killCalls).toEqual(["SIGTERM"]);
    }
    const stub = new StubChild();
    const failure = exitFailure(readBridgeReply(asChild(stub)));
    stub.emit("error", new Error("cannot spawn"));
    expect(await failure).toBe("process.exit(1)");
    expect(output.stderr).toHaveBeenLastCalledWith("needle catalog failed: cannot spawn\nbridge stderr tail: (empty)\n");
    stub.emit("close", 1);
  });
});

describe("Needle catalog fetch", () => {
  test("writes the describe reply to the catalog file with the exact spawn, request, and JSON format", async () => {
    const output = captureOutput();
    const stub = new StubChild();
    const { spawnFn, calls } = spawnRecorder(stub);
    const directory = await mkdtemp(join(tmpdir(), "needle-catalog-"));
    const catalogPath = join(directory, "catalog.json");
    try {
      const pending = fetchCatalog(catalogPath, spawnFn);
      stub.send(JSON.stringify({ tools: requiredEntries() }) + "\n");
      await pending;
      expect(calls).toEqual([[join(ROOT_DIR, "venv/bin/python"), [join(ROOT_DIR, "scripts/tool_bridge.py")], { cwd: ROOT_DIR, stdio: ["pipe", "pipe", "pipe"], detached: true }]]);
      expect(stub.written).toEqual(['{"id":"describe-1","op":"describe"}\n']);
      expect(stub.killCalls).toEqual(["SIGTERM"]);
      const expected: { name: string; description: string; parameters: unknown }[] = REQUIRED.map((name) => ({ name, description: name, parameters: { type: "object" } }));
      expected.push({ name: "get_current_time", description: "Get the current UTC date and time", parameters: { type: "object", properties: {} } });
      expect(await readFile(catalogPath, "utf8")).toBe(JSON.stringify(expected, null, 2));
      expect(output.stdout).toHaveBeenLastCalledWith(`needle [ai]: catalog ok ${catalogPath} (${expected.map((tool) => tool.name).join(",")})\n`);
      stub.emit("close", 0);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe("Needle child supervision", () => {
  test("signals the detached group first and falls back to the direct child", () => {
    const stub = new StubChild(4321);
    const kill = spyOn(process, "kill").mockReturnValue(true);
    killService(asChild(stub), "SIGINT");
    expect(kill).toHaveBeenCalledWith(-4321, "SIGINT");
    expect(stub.killCalls).toEqual([]);
    kill.mockImplementation(() => {
      throw new Error("group gone");
    });
    killService(asChild(stub), "SIGTERM");
    expect(stub.killCalls).toEqual(["SIGTERM"]);
  });

  test("kills directly for missing, invalid, or exited PIDs and tolerates a vanished child", () => {
    const kill = spyOn(process, "kill").mockReturnValue(true);
    for (const pid of [undefined, 0, -2, 4321]) {
      const stub = new StubChild(pid);
      if (pid === 4321) stub.exitCode = 0;
      killService(asChild(stub), "SIGTERM");
      expect(stub.killCalls).toEqual(["SIGTERM"]);
    }
    expect(kill).not.toHaveBeenCalled();
    const gone = new StubChild();
    gone.kill = () => {
      throw new Error("gone");
    };
    expect(() => killService(asChild(gone), "SIGTERM")).not.toThrow();
  });

  test("resolves on the first service exit and keeps logging later failures", async () => {
    const output = captureOutput();
    const api = new StubChild();
    const market = new StubChild();
    const web = new StubChild();
    const pending = waitForFirstExit([
      { name: "api", child: asChild(api) },
      { name: "market", child: asChild(market) },
      { name: "web", child: asChild(web) },
    ]);
    market.emit("close", null, "SIGTERM");
    api.emit("close", 0, null);
    web.emit("error", new Error("late failure"));
    web.emit("error", "late string failure");
    const first = await pending;
    expect(first.name).toBe("market");
    expect(first.child).toBe(asChild(market));
    expect(first.code).toBeNull();
    expect(first.signal).toBe("SIGTERM");
    expect(output.stderr).toHaveBeenCalledWith("needle [web]: spawn failed late failure\n");
    expect(output.stderr).toHaveBeenLastCalledWith("needle [web]: spawn failed late string failure\n");
  });

  test("treats a spawn error as the first service failure", async () => {
    captureOutput();
    const stub = new StubChild();
    const pending = waitForFirstExit([{ name: "api", child: asChild(stub) }]);
    stub.emit("error", new Error("cannot spawn"));
    stub.emit("close", null, null);
    const first = await pending;
    expect(first.name).toBe("api");
    expect(first.child).toBe(asChild(stub));
    expect(first.code).toBe(1);
    expect(first.signal).toBeNull();
  });

  test("copies pdfjs assets with the inherited child environment", async () => {
    captureOutput();
    const stub = new StubChild();
    const { spawnFn, calls } = spawnRecorder(stub);
    const env = { PORT: "33000" };
    const pending = runAssets(env, spawnFn);
    stub.emit("close", 0);
    await pending;
    expect(calls).toEqual([["bun", ["scripts/copy-pdfjs-assets.ts"], { cwd: HARNESS_DIR, stdio: "inherit", env, detached: true }]]);
  });

  test("reports asset exit codes, signal exits, and spawn failures", async () => {
    const output = captureOutput();
    for (const code of [9, null, "error"] as const) {
      const stub = new StubChild();
      const { spawnFn } = spawnRecorder(stub);
      const failure = exitFailure(runAssets({}, spawnFn));
      if (code === "error") stub.emit("error", new Error("cannot spawn"));
      else stub.emit("close", code);
      expect(await failure).toBe("process.exit(1)");
      expect(output.stderr).toHaveBeenLastCalledWith(`needle assets failed: copy-pdfjs-assets exited ${code === "error" ? 1 : code ?? "signal"}\n`);
    }
  });
});

describe("Needle thesis monitors", () => {
  test("start for theses with an enabled watch rule, or for every thesis when forced", async () => {
    const root = await mkdtemp(join(tmpdir(), "needle-thesis-"));
    const watch = {
      watched: "rules:\n- claim_ids:\n  - claim:1\n  enabled: true\n  rule_type: new_filing\nschema_version: 1\n",
      disabled: "rules:\n- claim_ids:\n  - claim:1\n  enabled: false\n  rule_type: new_external_evidence\nschema_version: 1\n",
      empty: "rules: []\nschema_version: 1\n",
    };
    try {
      for (const [slug, text] of Object.entries(watch)) {
        await mkdir(join(root, slug));
        await writeFile(join(root, slug, "watch.yaml"), text);
      }
      await mkdir(join(root, "no-watch-file"));
      await writeFile(join(root, "stray.yaml"), "enabled: true\n");
      expect(monitoredTheses(root, false)).toEqual(["watched"]);
      expect(monitoredTheses(root, true).sort()).toEqual(["disabled", "empty", "no-watch-file", "watched"]);
      expect(monitoredTheses(join(root, "missing"), true)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

describe("Needle web listener", () => {
  test("defaults to a production build on every interface; STOCKBOT_LAN=0 keeps loopback", () => {
    expect(webSettings({})).toEqual({ mode: "prod", host: "0.0.0.0" });
    expect(webSettings({ STOCKBOT_WEB_MODE: "dev" })).toEqual({ mode: "dev", host: "0.0.0.0" });
    expect(webSettings({ STOCKBOT_LAN: "0", STOCKBOT_LAN_TOKEN: "t" })).toEqual({ mode: "prod", host: "127.0.0.1" });
    expect(webSettings({ STOCKBOT_LAN_TOKEN: "t", STOCKBOT_WEB_MODE: "dev" })).toEqual({ mode: "dev", host: "0.0.0.0" });
  });

  test("refuses an unknown web mode", () => {
    const output = captureOutput();
    expect(() => webSettings({ STOCKBOT_WEB_MODE: "preview" })).toThrow("process.exit(1)");
    expect(output.stderr).toHaveBeenLastCalledWith('needle [web]: STOCKBOT_WEB_MODE must be "prod" or "dev", got "preview"\n');
  });

  test("builds the web bundle in the web directory and fails fast on a build error", async () => {
    const output = captureOutput();
    const ok = new StubChild();
    const { spawnFn, calls } = spawnRecorder(ok);
    const env = { PORT: "33000" };
    const pending = buildWeb(env, spawnFn);
    ok.emit("close", 0);
    await pending;
    expect(calls).toEqual([["bun", ["x", "vite", "build"], { cwd: join(ROOT_DIR, "web"), stdio: "inherit", env, detached: true }]]);
    const broken = new StubChild();
    const failure = exitFailure(buildWeb(env, spawnRecorder(broken).spawnFn));
    broken.emit("close", 2);
    expect(await failure).toBe("process.exit(1)");
    expect(output.stderr).toHaveBeenLastCalledWith("needle web build failed: vite build exited 2\n");
  });
});
