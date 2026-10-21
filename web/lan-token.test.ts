import { afterEach, describe, expect, test } from "bun:test";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { lanTokenMiddleware } from "./lan-token";

const TOKEN = "s3cret token/+";
const LAN = { STOCKBOT_LAN: "1", STOCKBOT_LAN_TOKEN: TOKEN };
let server: Server | undefined;

afterEach(() => {
  server?.close();
  server = undefined;
});

/** Real HTTP listener: the gate, then a stand-in for the proxy that answers 200 "passed". */
async function serve(env: NodeJS.ProcessEnv): Promise<string> {
  const gate = lanTokenMiddleware(env);
  server = createServer((req, res) => gate(req, res, () => res.end("passed")));
  await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

describe("LAN token gate", () => {
  test("passes every request through when no token is set", async () => {
    for (const env of [{}, { STOCKBOT_LAN_TOKEN: "" }, { STOCKBOT_LAN: "0" }, { STOCKBOT_LAN: "1" }]) {
      const base = await serve(env);
      const res = await fetch(`${base}/api/health`, { redirect: "manual" });
      expect(res.status).toBe(200);
      expect(await res.text()).toBe("passed");
      server!.close();
    }
  });


  test("answers 401 JSON on /api without a token or with a wrong one", async () => {
    const base = await serve(LAN);
    for (const init of [{}, { headers: { "x-stockbot-token": "wrong" } }, { headers: { "x-stockbot-token": `${TOKEN}x` } }, { headers: { cookie: "stockbot_lan_token=wrong" } }]) {
      for (const path of ["/api", "/api/health", "/api/research/abc"]) {
        const res = await fetch(base + path, init);
        expect(res.status).toBe(401);
        expect(res.headers.get("content-type")).toBe("application/json");
        expect(await res.json()).toEqual({ error: "lan_token_required" });
      }
    }
  });

  test("passes non-API paths without a token", async () => {
    const base = await serve(LAN);
    for (const path of ["/", "/assets/index.js", "/apiary"]) {
      expect(await (await fetch(base + path)).text()).toBe("passed");
    }
  });

  test("passes /api with the token header", async () => {
    const res = await fetch(`${await serve(LAN)}/api/health`, { headers: { "x-stockbot-token": TOKEN } });
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("passed");
  });

  test("query token sets the cookie, strips the token, and the cookie then passes /api", async () => {
    const base = await serve(LAN);
    const res = await fetch(`${base}/workspace?x=1&token=${encodeURIComponent(TOKEN)}`, { redirect: "manual" });
    expect(res.status).toBe(302);
    expect(res.headers.get("location")).toBe("/workspace?x=1");
    const setCookie = res.headers.get("set-cookie") ?? "";
    expect(setCookie).toBe(`stockbot_lan_token=${encodeURIComponent(TOKEN)}; Path=/; HttpOnly; SameSite=Strict`);
    const cookie = setCookie.split(";")[0]!;
    const api = await fetch(`${base}/api/health`, { headers: { cookie: `other=1; ${cookie}` } });
    expect(api.status).toBe(200);
    expect(await api.text()).toBe("passed");
  });

  test("a wrong query token sets no cookie", async () => {
    const res = await fetch(`${await serve(LAN)}/api/health?token=wrong`, { redirect: "manual" });
    expect(res.status).toBe(401);
    expect(res.headers.get("set-cookie")).toBeNull();
  });

  describe("CSRF guard (runs with and without a token)", () => {
    for (const env of [{}, LAN]) {
      test(`foreign-Origin POST to /api/agent is 403 (token ${env.STOCKBOT_LAN_TOKEN ? "set" : "unset"})`, async () => {
        const base = await serve(env);
        const res = await fetch(`${base}/api/agent`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Origin: "https://evil.example", host: new URL(base).host, ...(env.STOCKBOT_LAN_TOKEN ? { "x-stockbot-token": TOKEN } : {}) },
          body: JSON.stringify({ prompt: "x" }),
        });
        expect(res.status).toBe(403);
        expect(await res.json()).toEqual({ error: "forbidden_origin" });
      });
      test(`same-host Origin POST passes CSRF (token ${env.STOCKBOT_LAN_TOKEN ? "set" : "unset"})`, async () => {
        const base = await serve(env);
        const host = new URL(base).host;
        const res = await fetch(`${base}/api/agent`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Origin: `http://${host}`, host, ...(env.STOCKBOT_LAN_TOKEN ? { "x-stockbot-token": TOKEN } : {}) },
          body: JSON.stringify({ prompt: "x" }),
        });
        expect(res.status).toBe(200);
      });
      test(`form-encoded POST is 403 (token ${env.STOCKBOT_LAN_TOKEN ? "set" : "unset"})`, async () => {
        const base = await serve(env);
        const res = await fetch(`${base}/api/agent`, {
          method: "POST",
          headers: { "Content-Type": "text/plain", host: new URL(base).host, ...(env.STOCKBOT_LAN_TOKEN ? { "x-stockbot-token": TOKEN } : {}) },
          body: "prompt=x",
        });
        expect(res.status).toBe(403);
      });
      test(`body-less same-host POST needs a JSON content type (token ${env.STOCKBOT_LAN_TOKEN ? "set" : "unset"})`, async () => {
        const base = await serve(env);
        const host = new URL(base).host;
        const headers = { Origin: `http://${host}`, host, ...(env.STOCKBOT_LAN_TOKEN ? { "x-stockbot-token": TOKEN } : {}) };
        const bare = await fetch(`${base}/api/theses/x/check`, { method: "POST", headers });
        expect(bare.status).toBe(403);
        const json = await fetch(`${base}/api/theses/x/check`, { method: "POST", headers: { ...headers, "Content-Type": "application/json" } });
        expect(json.status).toBe(200);
      });
    }
    test("GET with a foreign Origin still passes", async () => {
      const base = await serve({});
      const res = await fetch(`${base}/api/health`, { headers: { Origin: "https://evil.example", host: new URL(base).host } });
      expect(res.status).toBe(200);
    });
  });
});
