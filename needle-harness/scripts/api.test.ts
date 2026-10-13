import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { dispatch, registerPrewarm, resolveApiPort, startApiServer } from "./api";

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
  }, 30000);

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
  }, 30000);
});
