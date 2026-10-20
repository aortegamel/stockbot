import { memo, useCallback, useEffect, useRef, useState } from "react";
import { fetchFeed, type FeedItem } from "../lib/market-data";
import { toastAlert } from "../components/toasts";
import { routeHash } from "../lib/route";
import { ago } from "../lib/time";
import type { ShellState } from "../layout/Shell";

const POLL_MS = 60_000;
// ponytail: cap the first page at 200 rows. Add virtualization if this limit grows.
const FIRST_PAGE = 200;
const KINDS = ["news", "filing", "insider", "alert", "short"];

function newestFirst(list: FeedItem[]): FeedItem[] {
  return [...list].sort((a, b) => Date.parse(b.at) - Date.parse(a.at));
}

type Props = Pick<ShellState, "selected" | "watchlist" | "setInput"> & { openDrawer?: () => void };

const FeedRow = memo(function FeedRow({ item, now, onAsk }: { item: FeedItem; now: number; onAsk: (item: FeedItem) => void }) {
  // Alerts open the workspace that holds the thesis panel.
  const href = item.kind === "alert" ? (item.sym ? routeHash({ page: "workspace", sym: item.sym }) : null) : item.link ?? null;
  return (
    <li className="feed-row">
      <time dateTime={item.at} title={item.at} className="feed-muted num">{ago(item.at, now)}</time>
      <span className="feed-muted">{item.source}</span>
      <span className="feed-kind">{item.kind}</span>
      <span className="feed-title" title={item.title}>{item.title}</span>
      {item.sym && <a className="pill" href={routeHash({ page: "workspace", sym: item.sym })}>{item.sym}</a>}
      <button type="button" className="pill" onClick={() => onAsk(item)}>Ask about this</button>
      {href && (
        <a className="pill" href={href} {...(item.kind === "alert" ? {} : { target: "_blank", rel: "noreferrer" })}>Open source</a>
      )}
    </li>
  );
});

export default function Feed({ selected, watchlist, setInput, openDrawer }: Props) {
  // null: the first page is loading.
  const [items, setItems] = useState<FeedItem[] | null>(null);
  // Polled items that arrived while the list was scrolled away from the top.
  const [pending, setPending] = useState<FeedItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sourceErrors, setSourceErrors] = useState(0);
  const [now, setNow] = useState(0);
  const [retry, setRetry] = useState(0);
  const [kind, setKind] = useState("all");
  const [ticker, setTicker] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const openDrawerRef = useRef(openDrawer);
  useEffect(() => {
    openDrawerRef.current = openDrawer;
  }, [openDrawer]);
  // The selected workspace rides with the watchlist; ShellState exposes no other open workspaces.
  const symsKey = [...new Set([selected, ...watchlist])].join(",");

  useEffect(() => {
    setItems(null);
    setPending([]);
    setError(null);
    setSourceErrors(0);
    let alive = true;
    let loaded = false;
    let since: string | undefined;
    let ctrl: AbortController | null = null;
    const seen = new Set<string>();
    async function load() {
      ctrl?.abort();
      const mine = new AbortController();
      ctrl = mine;
      try {
        const page = await fetchFeed(symsKey.split(","), since, mine.signal);
        if (!alive || mine.signal.aborted) return;
        page.items.forEach(toastAlert);
        const fresh = newestFirst(page.items.filter((item) => !seen.has(item.id)));
        for (const item of fresh) seen.add(item.id);
        if (fresh[0] && (!since || Date.parse(fresh[0].at) > Date.parse(since))) since = fresh[0].at;
        setNow(Date.now());
        setSourceErrors(page.errors.length);
        setError(null);
        if (!loaded) {
          loaded = true;
          setItems(fresh.slice(0, FIRST_PAGE));
        } else if (fresh.length > 0) {
          if ((scrollRef.current?.scrollTop ?? 0) < 40) setItems((prev) => newestFirst([...fresh, ...(prev ?? [])]));
          else setPending((prev) => newestFirst([...fresh, ...prev]));
        }
      } catch {
        // A failed poll keeps the loaded list. Only the first page shows the error.
        if (alive && !mine.signal.aborted && !loaded) setError("Feed unavailable.");
      }
    }
    const visible = () => {
      if (!document.hidden) void load();
    };
    void load();
    const id = setInterval(visible, POLL_MS);
    document.addEventListener("visibilitychange", visible);
    return () => {
      alive = false;
      ctrl?.abort();
      clearInterval(id);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [symsKey, retry]);

  const ask = useCallback((item: FeedItem) => {
    setInput(`About ${item.kind}${item.sym ? ` ${item.sym}` : ""}: ${item.title}${item.link ? ` (${item.link})` : ""} — `);
    openDrawerRef.current?.();
    requestAnimationFrame(() => document.querySelector<HTMLInputElement>('[aria-label="Ask the research agent"]')?.focus());
  }, [setInput]);

  function showPending() {
    setItems((prev) => newestFirst([...pending, ...(prev ?? [])]));
    setPending([]);
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }

  const sym = ticker.trim().toUpperCase();
  const visibleItems = (items ?? []).filter((item) => (kind === "all" || item.kind === kind) && (!sym || (item.sym ?? "").includes(sym)));

  return (
    <div className="page-scroll" ref={scrollRef}>
      <section className="glass page-card">
        <div className="feed-filters">
          <select aria-label="Kind" value={kind} onChange={(e) => setKind(e.target.value)} className="pill">
            <option value="all">all kinds</option>
            {KINDS.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <input aria-label="Ticker filter" placeholder="Ticker" value={ticker} onChange={(e) => setTicker(e.target.value)} className="pill" />
          {pending.length > 0 && (
            <button type="button" className="pill feed-new num" onClick={showPending}>{pending.length} new</button>
          )}
        </div>
        {sourceErrors > 0 && (
          <p className="feed-muted">
            <span className="num">{sourceErrors}</span> sources unavailable.{" "}
            <button type="button" className="pill" onClick={() => setRetry((n) => n + 1)}>Retry</button>
          </p>
        )}
        {error !== null ? (
          <p role="alert">
            {error}{" "}
            <button type="button" className="pill" onClick={() => setRetry((n) => n + 1)}>Retry</button>
          </p>
        ) : items === null ? (
          <ul className="feed-list" aria-busy="true">
            {[0, 1, 2, 3, 4].map((n) => (
              <li key={n} className="feed-row feed-skeleton" aria-hidden />
            ))}
          </ul>
        ) : visibleItems.length === 0 ? (
          <p className="feed-muted">No items for this filter — try Ask Stockbot: what's new with {selected}?</p>
        ) : (
          <ul className="feed-list">
            {visibleItems.map((item) => <FeedRow key={item.id} item={item} now={now} onAsk={ask} />)}
          </ul>
        )}
      </section>
    </div>
  );
}
