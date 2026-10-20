import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

const COOKIE = "stockbot_lan_token";

type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => void;

/** LAN gate for the web listener. A no-op unless STOCKBOT_LAN=1; then /api/* needs the token. */
export function lanTokenMiddleware(env: NodeJS.ProcessEnv = process.env): Middleware {
  if (env.STOCKBOT_LAN !== "1") return (_req, _res, next) => next();
  const token = env.STOCKBOT_LAN_TOKEN;
  if (!token) throw new Error("STOCKBOT_LAN=1 requires a non-empty STOCKBOT_LAN_TOKEN");
  const expected = Buffer.from(token);
  // The cookie stores the URL-encoded token, so the cookie compare needs no decode step.
  const expectedCookie = Buffer.from(encodeURIComponent(token));
  const same = (value: string | undefined, want: Buffer) => {
    if (value === undefined) return false;
    const got = Buffer.from(value);
    return got.length === want.length && timingSafeEqual(got, want);
  };
  return (req, res, next) => {
    const url = new URL(req.url ?? "/", "http://lan");
    if (same(url.searchParams.get("token") ?? undefined, expected)) {
      url.searchParams.delete("token");
      res.statusCode = 302;
      res.setHeader("Set-Cookie", `${COOKIE}=${expectedCookie.toString()}; Path=/; HttpOnly; SameSite=Strict`);
      res.setHeader("Location", url.pathname + url.search);
      res.end();
      return;
    }
    if (url.pathname !== "/api" && !url.pathname.startsWith("/api/")) return next();
    const header = req.headers["x-stockbot-token"];
    const cookie = req.headers.cookie
      ?.split(";")
      .map((part) => part.trim())
      .find((part) => part.startsWith(`${COOKIE}=`))
      ?.slice(COOKIE.length + 1);
    if (same(typeof header === "string" ? header : undefined, expected) || same(cookie, expectedCookie)) return next();
    res.statusCode = 401;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "lan_token_required" }));
  };
}
