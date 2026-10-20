import { useCallback, useEffect, useState } from "react";

// Match the validated ticker channel in the kernel.
export const TICKER_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;

export type Route = { page: "home" } | { page: "workspace"; sym: string } | { page: "feed" } | { page: "harness" };

// Unknown hashes and invalid tickers return null; the caller picks the default page.
export function parseRoute(hash: string): Route | null {
  const [page, raw, ...rest] = hash.replace(/^#\/?/, "").split("/");
  if (rest.length > 0) return null;
  if (page === "workspace") {
    let sym = "";
    try {
      sym = decodeURIComponent(raw ?? "").trim().toUpperCase();
    } catch {
      return null;
    }
    return TICKER_RE.test(sym) ? { page, sym } : null;
  }
  if (raw !== undefined) return null;
  return page === "home" || page === "feed" || page === "harness" ? { page } : null;
}

export function routeHash(route: Route): string {
  return route.page === "workspace" ? `#/workspace/${encodeURIComponent(route.sym)}` : `#/${route.page}`;
}

// Returns the parsed hash (null when absent or unknown) and a navigate function.
// Navigate updates the state at once, so a render never sees the old route after a pick.
export function useHashRoute(): [Route | null, (route: Route) => void] {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const sync = () => setHash(window.location.hash);
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  const navigate = useCallback((route: Route) => {
    window.location.hash = routeHash(route);
    setHash(window.location.hash);
  }, []);
  return [parseRoute(hash), navigate];
}
