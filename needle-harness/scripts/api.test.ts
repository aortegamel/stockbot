import { describe, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { dispatch, registerPrewarm, resolveApiPort, startApiServer } from "./api";
import { kernelRouter, type KernelResponse } from "@/lib/agent/kernel";

describe("bun api startup env", () => {
  // Real startup in a child process: the child's env and cwd never touch this suite.
  // Only the repo-root .env read is faked, so the real .env stays unread and unchanged.
  test("API startup uses root OpenCode settings from a foreign cwd and preserves inherited PORT", () => {
    const rootDotenv = join(import.meta.dir, "..", "..", ".env");
    const rootText = 'OPENCODE_API_KEY="root-key"\nOPENCODE_URL="http://127.0.0.1:9/root/responses"\nOPENCODE_MODEL="root-model"\nPORT="3000"\n';
    const dir = mkdtempSync(join(tmpdir(), "stockbot-api-env-"));
    try {
      writeFileSync(join(dir, ".env"), 'OPENCODE_API_KEY="cwd-key"\nOPENCODE_URL="http://127.0.0.1:9/cwd/responses"\nOPENCODE_MODEL="cwd-model"\n');
      writeFileSync(
        join(dir, "preload.ts"),
        `import { mock } from "bun:test";
import * as fs from "node:fs";
const real = { ...fs };
const root = ${JSON.stringify(rootDotenv)};
mock.module("node:fs", () => ({
  ...real,
  existsSync: (p: string) => p === root || real.existsSync(p),
  readFileSync: (p: string, ...a: unknown[]) => (p === root ? ${JSON.stringify(rootText)} : (real.readFileSync as (...x: unknown[]) => unknown)(p, ...a)),
}));
`,
      );
      writeFileSync(
        join(dir, "probe.ts"),
        `import ${JSON.stringify(join(import.meta.dir, "api.ts"))};
const pick = ["OPENCODE_API_KEY", "OPENCODE_URL", "OPENCODE_MODEL", "PORT", "STOCKBOT_ENVTEST_INHERITED"];
console.log(JSON.stringify(Object.fromEntries(pick.map((k) => [k, process.env[k]]))));
`,
      );
      const child = Bun.spawnSync([process.execPath, "--preload", "./preload.ts", "./probe.ts"], {
        cwd: dir,
        env: {
          PATH: process.env.PATH,
          HOME: process.env.HOME,
          OPENCODE_API_KEY: "stale-key",
          OPENCODE_URL: "http://127.0.0.1:9/stale/responses",
          OPENCODE_MODEL: "stale-model",
          PORT: "3101",
          STOCKBOT_ENVTEST_INHERITED: "kept",
        },
      });
      const out = child.stdout.toString();
      if (child.exitCode !== 0) throw new Error(child.stderr.toString());
      expect(JSON.parse(out.trim().split("\n").at(-1) ?? "")).toEqual({
        OPENCODE_API_KEY: "root-key",
        OPENCODE_URL: "http://127.0.0.1:9/root/responses",
        OPENCODE_MODEL: "root-model",
        PORT: "3101",
        STOCKBOT_ENVTEST_INHERITED: "kept",
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 30000);
});

describe("bun api host", () => {
  test("unknown path is 404 without starting a listener", async () => {
    const res = await dispatch(new Request("http://localhost/api/missing"));
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: "not found" });
  });

  test("agent rejects a wrong method without invoking the handler", async () => {
    const res = await dispatch(new Request("http://localhost/api/agent"));
    expect(res.status).toBe(405);
  });

  test("health reports the actual API listener port, not Vite PORT", async () => {
    const prevApi = process.env.STOCKBOT_API_PORT;
    const prevPort = process.env.PORT;
    process.env.STOCKBOT_API_PORT = "4101";
    process.env.PORT = "3101";
    try {
      const res = await dispatch(new Request("http://localhost/api/health"));
      expect(res.status).toBe(200);
      expect(await res.json()).toMatchObject({ ok: true, port: "4101" });
      expect(resolveApiPort()).toBe(4101);
    } finally {
      if (prevApi === undefined) delete process.env.STOCKBOT_API_PORT;
      else process.env.STOCKBOT_API_PORT = prevApi;
      if (prevPort === undefined) delete process.env.PORT;
      else process.env.PORT = prevPort;
    }
  });

  test("research rejects a wrong method without reading state", async () => {
    const res = await dispatch(new Request("http://localhost/api/research/sessions", { method: "POST" }));
    expect(res.status).toBe(405);
  });

  test("watchlist page load forwards the deduplicated list with one budgeted deadline", async () => {
    const rows = [
      { symbol: "NVDA", source: "sec_filings", tool: "list_sec_filings", status: "ok" },
      { symbol: "QQQ", source: "finra_threshold_list", tool: "get_threshold_securities", status: "no_records" },
    ];
    const calls: Record<string, unknown>[] = [];
    const spy = spyOn(kernelRouter, "call").mockImplementation(async (body: Record<string, unknown>) => {
      calls.push(body);
      return { watchlist: rows } as unknown as KernelResponse;
    });
    try {
      const before = Date.now();
      const res = await dispatch(new Request("http://localhost/api/watchlist?symbols=nvda,QQQ,nvda"));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ watchlist: rows });
      expect(calls.length).toBe(1);
      expect(calls[0]?.op).toBe("watchlist");
      expect(calls[0]?.watchlist).toEqual(["NVDA", "QQQ"]);
      const deadline = calls[0]?.deadlineAt as number;
      expect(deadline).toBeGreaterThanOrEqual(before + 15_000);
      expect(deadline).toBeLessThanOrEqual(Date.now() + 15_000);
    } finally {
      spy.mockRestore();
    }
  });

  test("watchlist worker failure is a 502 with the worker error, never an empty list", async () => {
    for (const [reply, error] of [
      [{ error: "<worker>: 'watchlist' must be a list of 1-25 tickers" }, "<worker>: 'watchlist' must be a list of 1-25 tickers"],
      [{ route: "x" }, "malformed watchlist reply"],
    ] as const) {
      const spy = spyOn(kernelRouter, "call").mockImplementation(async () => reply as unknown as KernelResponse);
      try {
        const res = await dispatch(new Request("http://localhost/api/watchlist?symbols=NVDA"));
        expect(res.status).toBe(502);
        expect(await res.json()).toEqual({ error });
      } finally {
        spy.mockRestore();
      }
    }
  });

  test("watchlist worker outage is a 502 carrying the failure message", async () => {
    for (const [thrown, message] of [[new Error("worker down"), "worker down"], ["raw boom", "raw boom"]] as const) {
      const spy = spyOn(kernelRouter, "call").mockImplementation(async () => {
        throw thrown;
      });
      try {
        const res = await dispatch(new Request("http://localhost/api/watchlist?symbols=NVDA,AAPL"));
        expect(res.status).toBe(502);
        expect(await res.json()).toEqual({ error: message });
      } finally {
        spy.mockRestore();
      }
    }
  });

  test("every owned path reaches its handler with or without a trailing slash", async () => {
    const spy = spyOn(kernelRouter, "call").mockImplementation(async () => ({ watchlist: [] }) as unknown as KernelResponse);
    try {
      const results: unknown[] = [];
      for (const path of ["/api/health", "/api/health/"]) {
        const res = await dispatch(new Request(`http://localhost${path}`));
        results.push([res.status, (await res.json()).ok]);
      }
      for (const path of ["/api/watchlist?symbols=NVDA", "/api/watchlist/?symbols=NVDA"]) {
        const res = await dispatch(new Request(`http://localhost${path}`));
        results.push([res.status, res.headers.get("Cache-Control")]);
      }
      const agent = await dispatch(new Request("http://localhost/api/agent/", { method: "GET" }));
      results.push(agent.status);
      expect(results).toEqual([
        [200, true], [200, true],
        [200, "no-store"], [200, "no-store"],
        405,
      ]);
      expect(spy).toHaveBeenCalledTimes(2);
    } finally {
      spy.mockRestore();
    }
  });

  test("research root and prefixed paths fail loudly without a bridge read", async () => {
    const research = await dispatch(new Request("http://localhost/api/research", { method: "GET" }));
    expect(research.status).toBe(400);
    expect(await research.json()).toEqual({ error: "bad session_id" });
    const agent = await dispatch(new Request("http://localhost/api/agent", { method: "POST", body: "{" }));
    expect(agent.status).toBe(400);
    expect(await agent.json()).toEqual({ error: "invalid JSON" });
  });

  test("malformed watchlist symbols are rejected before any worker call", async () => {
    const spy = spyOn(kernelRouter, "call");
    try {
      for (const query of ["", "?symbols=", "?symbols=SEC", "?symbols=NVDA,,AAPL", `?symbols=${Array.from({ length: 26 }, (_, i) => `T${i}`).join(",")}`]) {
        const res = await dispatch(new Request(`http://localhost/api/watchlist${query}`));
        expect(res.status).toBe(400);
      }
      expect((await dispatch(new Request("http://localhost/api/watchlist?symbols=NVDA", { method: "POST" }))).status).toBe(405);
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });

  test("startup prewarm sets KERNEL_PREWARMED without throwing", async () => {
    const prev = process.env.KERNEL_PREWARMED;
    delete process.env.KERNEL_PREWARMED;
    try {
      await registerPrewarm();
      expect(process.env.KERNEL_PREWARMED as string | undefined).toBe("1");
    } finally {
      if (prev === undefined) delete process.env.KERNEL_PREWARMED;
      else process.env.KERNEL_PREWARMED = prev;
    }
  }, 130_000);

  test("listener answers health on a non-primary loopback address, so it binds every IPv4 interface", async () => {
    // A 127.0.0.1-only bind refuses 127.0.0.2 on Linux; 0.0.0.0 accepts it. No LAN interface needed.
    const saved = { api: process.env.STOCKBOT_API_PORT, prewarmed: process.env.KERNEL_PREWARMED };
    const before = { SIGINT: process.listeners("SIGINT"), SIGTERM: process.listeners("SIGTERM") };
    process.env.STOCKBOT_API_PORT = "0";
    const server = await startApiServer();
    try {
      const res = await fetch(`http://127.0.0.2:${server.port}/api/health`);
      expect(res.status).toBe(200);
      expect(await res.json()).toMatchObject({ ok: true, port: String(server.port) });
    } finally {
      await server.stop(true);
      for (const sig of ["SIGINT", "SIGTERM"] as const) {
        for (const fn of process.listeners(sig)) if (!before[sig].includes(fn)) process.removeListener(sig, fn);
      }
      if (saved.api === undefined) delete process.env.STOCKBOT_API_PORT;
      else process.env.STOCKBOT_API_PORT = saved.api;
      if (saved.prewarmed === undefined) delete process.env.KERNEL_PREWARMED;
      else process.env.KERNEL_PREWARMED = saved.prewarmed;
    }
  }, 130_000);
});
