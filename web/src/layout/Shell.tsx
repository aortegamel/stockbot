import { useCallback, useContext, useEffect, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { AIPanel, watchlistFor } from "../components/market-panels";
import type { WidgetOutcomes } from "../components/action-cards";
import { HealthDot } from "../components/health-dot";
import { QUOTE_POLL_MS, fetchMarketStatus, fetchMarketWatchlist, fetchQuote, refreshMarket, saveMarketWatchlist, type MarketStatus } from "../lib/market-data";
import { parseRoute, routeHash, TICKER_RE, useHashRoute, type Route } from "../lib/route";
import type { AgentEvent, AskContext } from "../../../needle-harness/lib/agent/types";
import { fetchJournal, fetchSnapshot, listSessions, type JournalPage, type ResearchSnapshot } from "../lib/research-client";
import CommandBar from "./CommandBar";
import { DEFAULT_WORKSPACE_ID, type WorkspaceDraft } from "../pages/Workspace";
import { MarketRefreshContext } from "../components/home-panels";

// The existing watchdog stops once buffered persona reasoning starts.
const STALL_TIMEOUT_MS = 120_000;

const STRIP_DEFAULTS = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ"];

const DRAWER_KEY = "stockbot.drawer";

// Saved choice wins; otherwise the drawer starts open on wide screens.
function initialDrawer(): boolean {
  try {
    const saved = window.localStorage.getItem(DRAWER_KEY);
    if (saved) return saved === "open";
  } catch {
    // Storage can be disabled; fall back to the width default.
  }
  return window.innerWidth >= 1024;
}

const TICKER_ORG: Record<string, true> = { FINRA: true, SEC: true, NYSE: true, NASDAQ: true, OTC: true, EDGAR: true, SHO: true, REG: true };

// App state that outlives a page switch. Pages receive it as props.
export type ShellState = {
  selected: string;
  pick: (sym: string) => void;
  watchlist: string[];
  watchlistLoading?: boolean;
  onWatchlist: (symbols: string[]) => void;
  events: AgentEvent[];
  input: string;
  setInput: (value: string) => void;
  research: boolean;
  setResearch: (value: boolean) => void;
  // A feed item for the next Send. It rides as data beside the prompt; null clears it.
  setAskContext: (context: AskContext | null) => void;
  onSend: () => void;
  openDrawer: () => void;
  // Opens the drawer and sends text as a new run (the palette Ask row).
  ask?: (text: string) => void;
  busy: boolean;
  // The workspace reports which agent widget receipts it added or the cap blocked.
  onWidgetOutcomes?: (outcomes: WidgetOutcomes) => void;
  debug: boolean;
  sessions: { session_id?: string; status?: string; updated_at?: string; query?: string }[];
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  snapshot: ResearchSnapshot | null;
  journal: JournalPage | null;
  seq: number;
  workspace: WorkspaceDraft | null;
  setWorkspace: Dispatch<SetStateAction<WorkspaceDraft | null>>;
  // Shell owns the agent state. The workspace supplies its movable window frame.
  chat: ReactNode;
};

function Strip() {
  const refresh = useContext(MarketRefreshContext);
  const [prices, setPrices] = useState<Record<string, number | null>>({});
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      const next: Record<string, number | null> = {};
      await Promise.all(
        STRIP_DEFAULTS.map(async (sym) => {
          try {
            const q = await fetchQuote(sym, ctrl.signal);
            next[sym] = typeof q.price === "number" ? q.price : null;
          } catch {
            next[sym] = null;
          }
        }),
      );
      if (alive) setPrices(next);
    }
    void load();
    const id = setInterval(() => void load(), QUOTE_POLL_MS);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, [refresh]);
  const items = [...STRIP_DEFAULTS, ...STRIP_DEFAULTS];
  return (
    <div className="ticker-track">
      {items.map((sym, i) => (
        <span key={i} className="ticker-item">
          <b>{sym}</b>{" "}
          {prices[sym] == null ? "—" : prices[sym]!.toLocaleString(undefined, { maximumFractionDigits: 2 })}{" "}
          <span className="ticker-delayed">DELAYED</span>
        </span>
      ))}
    </div>
  );
}

// Frames can span network chunks. Ignore malformed data without changing event order.
function pushAgentFrame(frame: string, push: (event: AgentEvent) => void) {
  for (const line of frame.split("\n")) {
    const data = line.trim();
    if (!data.startsWith("data:")) continue;
    try {
      const event = JSON.parse(data.slice(5)) as AgentEvent;
      if (event.type !== "agent_start") push(event);
    } catch {
      // Preserve the existing parser behavior for malformed frames.
    }
  }
}

async function receiveAgentResponse(response: Response, push: (event: AgentEvent) => void, arm: () => void) {
  if (!response.ok || !response.body) {
    push({ type: "error", message: `request failed: ${response.status}` });
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    arm();
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      pushAgentFrame(frame, push);
    }
  }
}

function MarketControls({ error, setError, onCollected }: {
  error: string;
  setError: (error: string) => void;
  onCollected: () => void;
}) {
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [refreshPending, setRefreshPending] = useState(false);
  const refreshRequested = useRef(false);
  const collected = useRef(onCollected);
  collected.current = onCollected;

  useEffect(() => {
    const ctrl = new AbortController();
    let previous: MarketStatus | null = null;
    async function poll() {
      try {
        const status = await fetchMarketStatus(ctrl.signal);
        if (ctrl.signal.aborted) return;
        // Only requested refreshes set running. Routine cycles do not reload panels.
        if (!status.running && previous?.running === true) {
          refreshRequested.current = false;
          setRefreshPending(false);
          collected.current();
        }
        previous = status;
        setMarketStatus(status);
      } catch {
        if (!ctrl.signal.aborted) setError("Market status unavailable.");
      }
    }
    void poll();
    const id = setInterval(() => void poll(), 3_000);
    return () => { ctrl.abort(); clearInterval(id); };
  }, [setError]);

  async function requestMarketRefresh() {
    if (refreshRequested.current || marketStatus?.running) return;
    refreshRequested.current = true;
    setRefreshPending(true);
    setError("");
    try {
      await refreshMarket();
    } catch {
      refreshRequested.current = false;
      setRefreshPending(false);
      setError("Market refresh failed. Please retry.");
    }
  }
  const running = refreshPending || marketStatus?.running;
  return <>
    <span className="num" aria-live="polite">
      as of {marketStatus?.oldest_as_of == null ? "--:--" : new Date(marketStatus.oldest_as_of * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}
    </span>
    <button type="button" className="pill" disabled={running} onClick={() => void requestMarketRefresh()}>
      {running ? "Refreshing…" : "Refresh"}
    </button>
    {error && <span role="status">{error}</span>}
  </>;
}

function applyWatchlistReceipt(event: AgentEvent, symbols: string[], onWatchlist: (symbols: string[]) => void) {
  if (event.type !== "watchlist_update") return;
  const { action, ticker } = event;
  if ((action === "add" || action === "remove") && typeof ticker === "string" && TICKER_RE.test(ticker) && !TICKER_ORG[ticker]) {
    onWatchlist(action === "remove" ? symbols.filter((symbol) => symbol !== ticker)
      : symbols.includes(ticker) ? symbols : [...symbols, ticker]);
  }
}

export default function Shell({ children }: { children: (route: Route, state: ShellState) => ReactNode }) {
  const [hashRoute, navigate] = useHashRoute();
  // The last workspace ticker. A workspace hash wins; other pages keep the last pick.
  const [lastSym, setLastSym] = useState(() => (hashRoute?.page === "workspace" ? hashRoute.sym : "NVDA"));
  const route: Route = hashRoute ?? { page: "workspace", sym: lastSym };
  const selected = route.page === "workspace" ? route.sym : lastSym;
  if (selected !== lastSym) setLastSym(selected);
  const [savedWatchlist, setWatchlist] = useState<string[]>([]);
  const [watchlistLoading, setWatchlistLoading] = useState(true);
  const watchlistRef = useRef(savedWatchlist);
  const watchlistWrites = useRef(Promise.resolve());
  const [marketError, setMarketError] = useState("");
  const onWatchlist = useCallback((symbols: string[]) => {
    watchlistRef.current = symbols;
    setWatchlist(symbols);
    setWatchlistLoading(false);
    setMarketError("");
    watchlistWrites.current = watchlistWrites.current.then(async () => {
      const saved = await saveMarketWatchlist(symbols);
      if (watchlistRef.current === symbols) {
        watchlistRef.current = saved.symbols;
        setWatchlist(saved.symbols);
      }
    }).catch(() => setMarketError("Watchlist save failed. Please retry your edit."));
  }, []);
  const watchlist = watchlistFor(selected, savedWatchlist);
  const [marketRefresh, setMarketRefresh] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    const initial = watchlistRef.current;
    fetchMarketWatchlist(ctrl.signal).then((data) => {
      if (!ctrl.signal.aborted && watchlistRef.current === initial) {
        watchlistRef.current = data.symbols;
        setWatchlist(data.symbols);
        setWatchlistLoading(false);
      }
    }).catch(() => !ctrl.signal.aborted && setMarketError("Watchlist unavailable. Reload to retry."));
    return () => ctrl.abort();
  }, []);

  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [widgetOutcomes, setWidgetOutcomes] = useState<WidgetOutcomes>(new Map());
  const onWidgetOutcomes = useCallback((outcomes: WidgetOutcomes) => setWidgetOutcomes((previous) => new Map([...previous, ...outcomes])), []);
  const [busy, setBusy] = useState(false);
  const [debug, setDebug] = useState(false);
  const [sessions, setSessions] = useState<ShellState["sessions"]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<ResearchSnapshot | null>(null);
  const [journal, setJournal] = useState<JournalPage | null>(null);
  const [seq, setSeq] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const [input, setInput] = useState("");
  const [research, setResearch] = useState(false);
  const [askContext, setAskContext] = useState<AskContext | null>(null);
  const [drawer, setDrawerState] = useState(initialDrawer);
  const [workspace, setWorkspace] = useState<WorkspaceDraft | null>(null);

  useEffect(() => {
    if (!workspace?.dirty.length) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [workspace?.dirty.length]);
  const [, setClock] = useState("--:--:--");

  // Clock ticks on its own interval — no synchronous setState in the effect body.
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-GB")), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    fetch("/api/health", { signal: ctrl.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((health) => setDebug(health?.debug === true))
      .catch(() => { });
    return () => ctrl.abort();
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);
  // Backend sessions for the state panel.
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      try {
        const r = await listSessions(20, ctrl.signal);
        if (alive) {
          setSessions(r.sessions ?? []);
        }
      } catch {
        if (alive) {
          setSessions([]);
        }
      }
    }
    void load();
    const id = setInterval(() => void load(), 15_000);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, []);

  // Snapshot + journal poll for the selected research session (durable state + logs).
  useEffect(() => {
    setSnapshot(null);
    setJournal(null);
    setSeq(0);
    if (sessionId === null) {
      return;
    }
    const current = sessionId;
    let alive = true;
    const ctrl = new AbortController();
    let cursor = 0;
    async function pollJournal() {
      try {
        const page = await fetchJournal(current, cursor, ctrl.signal);
        if (!alive) {
          return;
        }
        const rows = page.events ?? [];
        if (rows.length > 0) {
          const head = rows[rows.length - 1];
          const tailSeq = head && typeof head.sequence === "number" ? head.sequence : cursor;
          cursor = Math.max(cursor, tailSeq);
          setSeq(cursor);
          setJournal((prev) => ({
            session_id: page.session_id,
            after_seq: cursor,
            events: [...(prev?.events ?? []), ...rows].slice(-200),
          }));
        } else {
          setJournal((prev) => prev ?? page);
        }
      } catch {
        if (alive) {
          setJournal((prev) => prev ?? { error: "journal unavailable" });
        }
      }
    }
    async function poll() {
      try {
        const snap = await fetchSnapshot(current, ctrl.signal);
        if (!alive) {
          return;
        }
        setSnapshot(snap);
      } catch {
        if (alive) {
          setSnapshot({ error: "snapshot unavailable" });
        }
      }
      await pollJournal();
    }
    void poll();
    const id = setInterval(() => void poll(), 5_000);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, [sessionId]);
  // A pick never changes the page. On the workspace it also rewrites the hash.
  // Read the live hash: an agent stream can call pick after the user leaves the workspace.
  function pick(sym: string) {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    setLastSym(clean);
    if ((parseRoute(window.location.hash) ?? route).page === "workspace") navigate({ page: "workspace", sym: clean });
  }
  function setDrawer(open: boolean) {
    setDrawerState(open);
    try {
      window.localStorage.setItem(DRAWER_KEY, open ? "open" : "closed");
    } catch {
      // Storage can be disabled; the choice then lasts for this page load only.
    }
  }

  // The palette's ticker rows always land on the workspace.
  function goTicker(sym: string) {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    pick(clean);
    navigate({ page: "workspace", sym: clean });
  }

  // text: a prompt from outside the panel input (the palette Ask row). Only the panel Send uses the feed item.
  async function onSend(text?: string) {
    const context = text === undefined ? askContext : null;
    const raw = text ?? input;
    const prefix = /^\s*\/research\b\s*/i;
    const researchMode = research || prefix.test(raw);
    const q = raw.replace(prefix, "").trim() || (context ? `What does this mean${context.sym ? ` for ${context.sym}` : ""}?` : "");
    if (!q || busy || abortRef.current) return;
    const prompt = q;
    setBusy(true);
    setEvents([{ type: "agent_start", prompt, ...(context ? { context } : {}) }]);
    setWidgetOutcomes(new Map());
    setInput("");
    setAskContext(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let terminal = false;
    let stalled = false;
    let reasoningStarted = false;
    let timer: number | undefined;
    const arm = () => {
      clearTimeout(timer);
      if (reasoningStarted) return;
      timer = window.setTimeout(() => {
        stalled = true;
        ctrl.abort();
      }, STALL_TIMEOUT_MS);
    };
    const push = (event: AgentEvent) => {
      if (event.type === "chart_update") {
        const ticker = typeof event.ticker === "string" ? event.ticker.trim().toUpperCase() : "";
        if (!TICKER_RE.test(ticker) || TICKER_ORG[ticker]) return;
        pick(ticker);
      }
      applyWatchlistReceipt(event, watchlistRef.current, onWatchlist);
      if (event.type === "done" || event.type === "error" || event.type === "failed") terminal = true;
      if (event.type === "reasoning_start") {
        reasoningStarted = true;
        clearTimeout(timer);
      }
      setEvents((previous) => [...previous, event]);
    };
    arm();
    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Watchlist rides as separate context; the prompt stays verbatim and the server decides scope.
        body: JSON.stringify({ prompt, ticker: selected, currentChartTicker: selected, ...(watchlist.length ? { watchlist } : {}), ...(context ? { context } : {}), ...(researchMode ? { mode: "research" } : {}) }),
        signal: ctrl.signal,
      });
      await receiveAgentResponse(response, push, arm);
      if (!terminal) push({ type: "error", message: "stream ended without a terminal event" });
    } catch (error) {
      if (!terminal) {
        push({
          type: "error",
          message: stalled
            ? `stream stalled: no data for ${STALL_TIMEOUT_MS / 1000}s`
            : error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      clearTimeout(timer);
      abortRef.current = null;
      setBusy(false);
    }
  }

  const ask = (text: string) => {
    setDrawer(true);
    void onSend(text);
  };
  // The chat stays mounted for a windowed Agent. The drawer toggle gates only the docked aside.
  const chat = <AIPanel events={events} input={input} setInput={setInput} research={research} setResearch={setResearch} context={askContext} onClearContext={() => setAskContext(null)} onSend={() => void onSend()} sym={selected} busy={busy} debug={debug} widgetOutcomes={widgetOutcomes} />;
  const state: ShellState = {
    selected, pick, watchlist, watchlistLoading, onWatchlist, events, input, setInput, research, setResearch, setAskContext, onSend: () => void onSend(),
    busy, debug, sessions, sessionId, setSessionId, snapshot, journal, seq, openDrawer: () => setDrawer(true), ask, onWidgetOutcomes, chat,
    workspace, setWorkspace,
  };

  const navLink = (label: string, to: Route) => (
    <a href={routeHash(to)} aria-current={route.page === to.page ? "page" : undefined} className="pill">
      {label}
    </a>
  );
  const currentWorkspace = workspace?.docs.find((doc) => doc.id === workspace.activeId);
  const windowedAgent = route.page === "workspace" && currentWorkspace?.widgets.some((widget) => widget.type === "agent");

  return (
    <main className="app">
      <span aria-hidden className="orbs">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
      </span>
      <span className="grain" aria-hidden />

      <header className="thinbar app-header">
        <span className="app-header-row">
          <b className="brand">NEEDLE TERMINAL</b>
          <nav className="nav" aria-label="Pages">
            {navLink(`Workspaces: ${currentWorkspace?.name ?? "default"}`, { page: "workspace", sym: selected })}
            {navLink("Feed", { page: "feed" })}
            {debug && navLink("Harness", { page: "harness" })}
          </nav>
          <CommandBar
            watchlist={watchlist}
            sessions={sessions}
            onTicker={goTicker}
            onScan={() => {
              setWorkspace((draft) => draft ? { ...draft, activeId: DEFAULT_WORKSPACE_ID } : draft);
              navigate({ page: "workspace", sym: selected });
            }}
            onSession={(id) => {
              setSessionId(id);
              navigate({ page: "workspace", sym: selected });
            }}
            onAsk={ask}
          />
          <span className="nav">
            <MarketControls error={marketError} setError={setMarketError} onCollected={() => setMarketRefresh((n) => n + 1)} />
            <HealthDot />
            <button
              type="button"
              onClick={() => setDrawer(!drawer)}
              aria-label={drawer ? "Close agent drawer" : "Open agent drawer"}
              aria-expanded={drawer}
              className="icon-btn"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <rect x="1.75" y="2.75" width="12.5" height="10.5" rx="1.5" />
                <path d="M10 2.75v10.5" />
              </svg>
            </button>
          </span>
        </span>
      </header>

      <MarketRefreshContext.Provider value={marketRefresh}>
        <div className="shell-body">
          {children(route, state)}
          {drawer && !windowedAgent && <aside className="glass ws-cell drawer" aria-label="Agent drawer">{chat}</aside>}
        </div>

        <footer className="app-footer">
          <span className="strip ticker-strip">
            <Strip />
          </span>
          <details className="footer-disclaimer">
            <summary>Disclaimers</summary>
            <p>Stockbot is for research and informational purposes only. It is not investment, legal, tax, or financial advice. Verify source data and make your own decisions.</p>
          </details>
          <span>delayed feed · Yahoo OHLCV + SEC EDGAR + multi-source news · research snapshot + journal</span>
          <span>{selected}</span>
        </footer>
      </MarketRefreshContext.Provider>
    </main>
  );
}
