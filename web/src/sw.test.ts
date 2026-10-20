import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

const SRC = readFileSync(new URL("../public/sw.js", import.meta.url), "utf8");
const ORIGIN = "https://app.test";

type Req = { url: string; method: string; mode: string };

// Runs public/sw.js as a classic worker against fake self/caches/fetch and records every cache touch.
function loadWorker() {
  const listeners: Record<string, (event: unknown) => void> = {};
  const cacheOps: string[] = [];
  const stored = Promise.withResolvers<void>();
  const caches = {
    match: async (req: Req | string) => {
      cacheOps.push(`match ${typeof req === "string" ? req : req.url}`);
      return undefined;
    },
    open: async () => ({
      put: async (req: Req) => {
        cacheOps.push(`put ${req.url}`);
        stored.resolve();
      },
    }),
  };
  const self = {
    location: { origin: ORIGIN },
    addEventListener: (type: string, fn: (event: unknown) => void) => {
      listeners[type] = fn;
    },
  };
  new Function("self", "caches", "fetch", SRC)(self, caches, async () => new Response("net"));

  async function dispatch(path: string, method = "GET"): Promise<boolean> {
    let responded: Promise<Response> | undefined;
    listeners.fetch({ request: { url: ORIGIN + path, method, mode: "cors" }, respondWith: (p: Promise<Response>) => (responded = p) });
    if (responded) await responded;
    return responded !== undefined;
  }
  return { dispatch, cacheOps, stored: stored.promise };
}

describe("service worker cache policy", () => {
  test("never serves or stores /api/ requests", async () => {
    const sw = loadWorker();
    expect(await sw.dispatch("/api/quote?symbol=NVDA")).toBe(false);
    expect(await sw.dispatch("/api/agent", "POST")).toBe(false);
    expect(sw.cacheOps).toEqual([]);
  });

  test("caches hashed assets on first fetch", async () => {
    const sw = loadWorker();
    expect(await sw.dispatch("/assets/index-abc123.js")).toBe(true);
    await sw.stored;
    expect(sw.cacheOps).toEqual([`match ${ORIGIN}/assets/index-abc123.js`, `put ${ORIGIN}/assets/index-abc123.js`]);
  });
});
