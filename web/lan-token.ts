import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

const COOKIE = "stockbot_lan_token";

type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => void;

/** LAN gate for the web listener. Open when no token is set; /api/* needs the token once set.
 * The CSRF check always runs: non-GET /api requests need a same-host Origin
 * (or no Origin, e.g. curl) and a JSON content type, so a foreign page cannot
 * drive the API through the user's browser. */
export function lanTokenMiddleware(env: NodeJS.ProcessEnv = process.env): Middleware {
 const token = env.STOCKBOT_LAN_TOKEN;
 if (!token) {
  // Open mode: only the CSRF check runs.
  return (req, res, next) => csrfGuard(req, res, () => next());
 }
 const expected = Buffer.from(token);
 // The cookie stores the URL-encoded token, so the cookie compare needs no decode step.
 const expectedCookie = Buffer.from(encodeURIComponent(token));
 const same = (value: string | undefined, want: Buffer) => {
  if (value === undefined) return false;
  const got = Buffer.from(value);
  return got.length === want.length && timingSafeEqual(got, want);
 };
 return (req, res, next) => csrfGuard(req, res, () => {
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
 });
}

// CSRF: state-changing API calls must come from our own pages (same-host Origin)
// or non-browser clients (no Origin), with a JSON body a cross-origin form cannot send.
function csrfGuard(req: IncomingMessage, res: ServerResponse, next: () => void): void {
 const url = new URL(req.url ?? "/", "http://lan");
 if ((url.pathname === "/api" || url.pathname.startsWith("/api/")) && (req.method ?? "GET") !== "GET") {
  const origin = req.headers.origin;
  const host = req.headers.host;
  if (typeof origin === "string" && origin !== "" && host !== undefined) {
   let sameHost = false;
   try {
    sameHost = new URL(origin).host === host;
   } catch {
    sameHost = false;
   }
   if (!sameHost) return deny(res);
  }
  if ((req.headers["content-type"] ?? "").split(";")[0].trim() !== "application/json") return deny(res);
 }
 next();
}

function deny(res: ServerResponse): void {
 res.statusCode = 403;
 res.setHeader("Content-Type", "application/json");
 res.end(JSON.stringify({ error: "forbidden_origin" }));
}
