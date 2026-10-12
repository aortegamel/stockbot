import { useEffect, useRef, useState } from "react";
import TradingChart from "./components/trading-chart";
import {
  WatchlistPanel,
  FinancialsPanel,
  NewsPanel,
  AIPanel,
  SessionPanel,
  JournalPanel,
} from "./components/market-panels";
import { fetchQuote, searchTickers, type SearchHit } from "./lib/market-data";
import type { AgentEvent } from "../../needle-harness/lib/agent/types";
import { fetchJournal, fetchSnapshot, listSessions, type JournalPage, type ResearchSnapshot } from "./lib/research-client";

// The existing watchdog stops once buffered persona reasoning starts.
const STALL_TIMEOUT_MS = 120_000;

const BG =
  "radial-gradient(120% 90% at 12% 0%, rgba(31,111,235,0.20), transparent 55%), radial-gradient(100% 80% at 100% 8%, rgba(41,72,150,0.35), transparent 60%), radial-gradient(90% 70% at 50% 112%, rgba(6,182,212,0.14), transparent 60%), linear-gradient(180deg, #141A30 0%, #101528 48%, #0C101F 100%)";

const STRIP_DEFAULTS = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ"];

function Strip() {
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
    const id = setInterval(() => void load(), 60_000);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, []);
  const items = [...STRIP_DEFAULTS, ...STRIP_DEFAULTS];
  return (
    <div style={{ display: "inline-block", animation: "tick 40s linear infinite" }}>
      {items.map((sym, i) => (
        <span key={i} style={{ marginRight: 18 }}>
          <b>{sym}</b>{" "}
          {prices[sym] == null ? "—" : prices[sym]!.toLocaleString(undefined, { maximumFractionDigits: 2 })}{" "}
          <span style={{ color: "var(--muted, #93A0B8)", fontSize: 12 }}>DELAYED</span>
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState("NVDA");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [debug, setDebug] = useState(false);
  const [sessions, setSessions] = useState<{ session_id?: string; status?: string; updated_at?: string; query?: string }[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<ResearchSnapshot | null>(null);
  const [journal, setJournal] = useState<JournalPage | null>(null);
  const [seq, setSeq] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const [input, setInput] = useState("");
  const [clock, setClock] = useState("--:--:--");

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
    void poll();
    const id = setInterval(() => void poll(), 5_000);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, [sessionId]);
  // Any-ticker search: Yahoo validates the symbol, selecting it loads real data.
  useEffect(() => {
    const q = query.trim();
    let alive = true;
    const ctrl = new AbortController();
    if (q.length < 2) {
      const id = setTimeout(() => alive && setHits([]), 0);
      return () => {
        alive = false;
        clearTimeout(id);
        ctrl.abort();
      };
    }
    const id = setTimeout(() => {
      searchTickers(q, ctrl.signal)
        .then((page) => alive && setHits(page.hits.slice(0, 8)))
        .catch(() => { });
    }, 250);
    return () => {
      alive = false;
      clearTimeout(id);
      ctrl.abort();
    };
  }, [query]);

  function pick(sym: string) {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    setSelected(clean);
    setQuery("");
    setHits([]);
  }

  async function onSend() {
    const q = input.trim();
    if (!q || busy || abortRef.current) return;
    const prompt = `[${selected}] ${q}`;
    setBusy(true);
    setEvents([{ type: "agent_start", prompt }]);
    setInput("");
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
        body: JSON.stringify({ prompt }),
        signal: ctrl.signal,
      });
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
      }
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

  return (
    <main
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: BG,
        color: "var(--text, #E6ECF5)",
        fontSize: 20,
        position: "relative",
        overflow: "hidden",
      }}
    >
      <style>{`@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}`}</style>
      <style>{`
        .orb{position:absolute;border-radius:50%;filter:blur(90px);pointer-events:none;z-index:0;mix-blend-mode:screen}
        .orb-a{width:54vw;height:54vw;left:-14vw;top:-22vw;background:#1f6feb;opacity:.34;animation:orb-drift1 28s ease-in-out infinite}
        .orb-b{width:46vw;height:46vw;right:-12vw;top:-12vw;background:#294896;opacity:.40;animation:orb-drift2 34s ease-in-out infinite}
        .orb-c{width:50vw;height:50vw;left:20vw;bottom:-30vw;background:#06b6d4;opacity:.20;animation:orb-drift3 40s ease-in-out infinite}
        @keyframes orb-drift1{0%,100%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(6vw,4vw,0) scale(1.12)}}
        @keyframes orb-drift2{0%,100%{transform:translate3d(0,0,0) scale(1.05)}50%{transform:translate3d(-7vw,6vw,0) scale(.94)}}
        @keyframes orb-drift3{0%,100%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(4vw,-6vw,0) scale(1.1)}}
        .grain{position:absolute;inset:0;z-index:0;pointer-events:none;opacity:.16;mix-blend-mode:overlay;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.45'/%3E%3C/svg%3E")}
        .glass{position:relative;background:rgba(20,26,48,0.42);backdrop-filter:blur(48px) saturate(180%);-webkit-backdrop-filter:blur(48px) saturate(180%);border:1px solid rgba(255,255,255,0.11);border-radius:20px;box-shadow:0 16px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.20);overflow:hidden}
        .glass::before{content:"";position:absolute;top:0;left:10%;right:10%;height:1px;background:radial-gradient(ellipse at center, rgba(255,255,255,0.5), transparent 70%);pointer-events:none;z-index:3}
        .glass > div{position:relative;z-index:1;background:transparent !important;border-color:rgba(255,255,255,0.08) !important;flex:1;min-height:0;min-width:0}
        .glass input{background:rgba(255,255,255,0.07) !important;border-color:rgba(255,255,255,0.12) !important;color:#E6ECF5 !important}
        .glass table td,.glass table th{border-color:rgba(255,255,255,0.08) !important}
        .thinbar{background:rgba(255,255,255,0.06);backdrop-filter:blur(48px) saturate(180%);-webkit-backdrop-filter:blur(48px) saturate(180%);border-bottom:0.5px solid rgba(255,255,255,0.14)}
        .strip{background:rgba(255,255,255,0.05);backdrop-filter:blur(48px) saturate(180%);-webkit-backdrop-filter:blur(48px) saturate(180%);border:1px solid rgba(255,255,255,0.12);box-shadow:inset 0 1px 0 rgba(255,255,255,0.22)}
        @media (prefers-reduced-motion:reduce){.orb{animation:none !important}}
      `}</style>
      <span aria-hidden style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
      </span>
      <span className="grain" aria-hidden />

      <header
        className="thinbar"
        style={{ position: "relative", zIndex: 1, height: 46 }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 18px", height: 46, width: "100%", fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif" }}>
          <b style={{ fontSize: 18, fontWeight: 700, letterSpacing: "0.12em", color: "var(--text, #E6ECF5)" }}>NEEDLE TERMINAL</b>
          <span style={{ position: "relative" }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") pick(hits[0]?.sym ?? query);
                if (e.key === "Escape") {
                  setQuery("");
                  setHits([]);
                }
              }}
              placeholder="Search any US ticker…"
              style={{ fontSize: 14, padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.07)", color: "#E6ECF5", width: 220 }}
            />
            {hits.length > 0 && (
              <div style={{ position: "absolute", top: 30, left: 0, right: 0, background: "#141A30", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, overflow: "hidden", zIndex: 10 }}>
                {hits.map((h) => (
                  <button
                    key={h.sym}
                    onClick={() => pick(h.sym)}
                    style={{ display: "block", width: "100%", textAlign: "left", padding: "6px 10px", background: "transparent", border: 0, color: "#E6ECF5", cursor: "pointer", fontSize: 14 }}
                  >
                    <b>{h.sym}</b> <span style={{ opacity: 0.6 }}>{h.name ?? ""}</span>
                  </button>
                ))}
              </div>
            )}
          </span>
          <span style={{ marginLeft: "auto", color: "var(--muted, #93A0B8)", fontSize: 15 }}>
            <span style={{ color: "var(--down, #EF5350)" }}>○ DELAYED</span> · {clock}
          </span>
        </span>
      </header>

      <div
        className="strip"
        style={{ position: "relative", zIndex: 1, margin: "8px 14px 0", borderRadius: 14, overflow: "hidden", whiteSpace: "nowrap", padding: "7px 0", fontSize: 15, color: "var(--text, #E6ECF5)" }}
      >
        <Strip />
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "280px 1fr 470px",
          gap: 8,
          padding: 8,
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <div className="glass" style={{ flexShrink: 0 }}>
            <WatchlistPanel selected={selected} onSelect={pick} />
          </div>
          <div className="glass" style={{ flexShrink: 0 }}>
            <SessionPanel sessions={sessions} sessionId={sessionId} onSelect={setSessionId} snapshot={snapshot} />
          </div>
          <div className="glass" style={{ flexShrink: 0 }}>
            <JournalPanel journal={journal} seq={seq} />
          </div>
        </div>
        <div style={{ overflow: "hidden", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <div className="glass" style={{ flex: "1.4 1 0", minHeight: 0 }}>
            <TradingChart sym={selected} />
          </div>
          <div style={{ flex: "0.9 1 0", minHeight: 0, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, overflow: "hidden" }}>
            <div className="glass" style={{ minHeight: 0, minWidth: 0, overflow: "hidden", display: "flex" }}>
              <FinancialsPanel sym={selected} />
            </div>
            <div className="glass" style={{ minHeight: 0, minWidth: 0, overflow: "hidden", display: "flex" }}>
              <NewsPanel sym={selected} />
            </div>
          </div>
        </div>
        <div className="glass" style={{ overflow: "hidden", minHeight: 0, display: "flex" }}>
          <AIPanel events={events} input={input} setInput={setInput} onSend={() => void onSend()} sym={selected} busy={busy} debug={debug} />
        </div>
      </div>

      <footer
        style={{ position: "relative", zIndex: 1, fontSize: 13, padding: "4px 12px", display: "flex", justifyContent: "space-between", color: "rgba(235,235,245,0.78)" }}
      >
        <span>delayed feed · Yahoo OHLCV + SEC EDGAR + multi-source news · research snapshot + journal</span>
        <span>{selected}</span>
      </footer>
    </main>
  );
}
