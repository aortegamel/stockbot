import { describe, expect, test } from "bun:test";
import { dispatch, registerPrewarm, resolveApiPort } from "./api";

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
  });
});
