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

  test("skips a port held on another local address such as 127.0.0.2", async () => {
    // Linux routes all of 127/8 to lo. A 127.0.0.1-only probe binds beside this listener and
    // wrongly reports the port free; a 0.0.0.0 probe collides with it.
    const busy = createServer();
    const { promise, resolve } = Promise.withResolvers<void>();
    busy.listen(0, "127.0.0.2", resolve);
    await promise;
    const addr = busy.address();
    if (typeof addr !== "object" || addr === null) throw new Error("expected a bound TCP address");
    try {
      expect(await pickFreePort(addr.port)).toBeGreaterThan(addr.port);
    } finally {
      busy.close();
    }
  });

  test("treats invalid input as 3000", async () => {
    expect(await pickFreePort(Number("junk"))).toBeGreaterThanOrEqual(3000);
    expect(await pickFreePort(-1)).toBeGreaterThanOrEqual(3000);
  });
});