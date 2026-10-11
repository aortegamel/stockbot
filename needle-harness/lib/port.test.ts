import { describe, expect, test } from "bun:test";
import { createServer } from "node:net";
import { pickFreePort } from "./port";

describe("pickFreePort", () => {
  test("skips an occupied port and returns the preferred one when free", async () => {
    const busy = createServer();
    const { promise, resolve } = Promise.withResolvers<void>();
    busy.listen(0, "127.0.0.1", resolve);
    await promise;
    const addr = busy.address();
    if (typeof addr !== "object" || addr === null) throw new Error("expected a bound TCP address");
    const taken = addr.port;
    try {
      expect(await pickFreePort(taken)).toBeGreaterThan(taken);
    } finally {
      busy.close();
    }
    expect(await pickFreePort(taken)).toBe(taken);
  });

  test("treats invalid input as 3000", async () => {
    expect(await pickFreePort(Number("junk"))).toBeGreaterThanOrEqual(3000);
    expect(await pickFreePort(-1)).toBeGreaterThanOrEqual(3000);
  });
});