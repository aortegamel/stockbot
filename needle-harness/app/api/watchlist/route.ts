import { kernelRouter, parseWatchlist, WATCHLIST_BUDGET_MS } from "@/lib/agent/kernel";

// Page-load trigger: GET /api/watchlist?symbols=NVDA,AAPL. Read-only fan-out of SEC filings and
// the three FINRA datasets per symbol through the kernel worker's tool handlers and caches.
// Every (symbol, source) row comes back with its own status; a malformed list is rejected whole.
export async function GET(req: Request): Promise<Response> {
  const raw = new URL(req.url).searchParams.get("symbols");
  const watchlist = parseWatchlist(raw === null ? undefined : raw.split(","));
  if (!watchlist) return Response.json({ error: "symbols must list 1-25 valid tickers" }, { status: 400 });
  // The deadline travels with the request, so a call queued behind another page load keeps the same budget.
  const deadlineAt = Date.now() + WATCHLIST_BUDGET_MS;
  try {
    const res = await kernelRouter.call({ op: "watchlist", watchlist, deadlineAt }, { signal: req.signal, timeoutMs: WATCHLIST_BUDGET_MS * 2 });
    if (typeof res.error === "string" && res.error) return Response.json({ error: res.error }, { status: 502 });
    if (!Array.isArray(res.watchlist)) return Response.json({ error: "malformed watchlist reply" }, { status: 502 });
    return Response.json({ watchlist: res.watchlist }, { headers: { "Cache-Control": "no-store" } });
  } catch (err) {
    return Response.json({ error: err instanceof Error ? err.message : String(err) }, { status: 502 });
  }
}
