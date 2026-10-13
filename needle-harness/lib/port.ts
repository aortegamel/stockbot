import { createServer } from "node:net";

/** First port free on every IPv4 interface (0.0.0.0) at or above the preferred port (default 3000). */
export async function pickFreePort(preferred = 3000): Promise<number> {
  const start = Number.isInteger(preferred) && preferred > 0 ? preferred : 3000;
  // ponytail: 100-try ceiling; raise the bound if a host ever packs denser.
  for (let port = start, n = 0; n < 100 && port <= 65535; port++, n++) {
    if (await isFree(port)) return port;
  }
  throw new Error(`no free port at or above ${start}`);
}

function isFree(port: number): Promise<boolean> {
  const { promise, resolve } = Promise.withResolvers<boolean>();
  const server = createServer();
  server.once("error", () => resolve(false));
  server.listen(port, "0.0.0.0", () => server.close(() => resolve(true)));
  return promise;
}
