import { useEffect, useState } from "react";
import { fetchFilings, fetchNews, fetchQuote, fetchWatchlistSources, type SourceRow } from "../lib/market-data";

import { AgentConsole } from "./agent-console";
import { useLoad } from "./home-panels";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import type { JournalPage, ResearchSnapshot } from "../lib/research-client";
import { sessionTitle } from "../lib/session-title";
import { ago } from "../lib/time";

// Panels sit inside a glass card or tab, which already draws the surface and border.
const panel: React.CSSProperties = {
  padding: 16,
  overflow: "hidden",
};
const h: React.CSSProperties = {
  fontSize: 12,
  letterSpacing: "0.06em",
  color: "var(--muted, #93A0B8)",
  marginBottom: 8,
  fontWeight: 600,
};
const muted = "var(--muted, #93A0B8)";
const down = "var(--down, #EF5350)";

const WATCH_DEFAULTS = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG"];

// The actual UI watchlist: the selected symbol is always present even when it
// isn't in the defaults (any-ticker search). The agent receives this same list.
export function watchlistFor(selected: string): string[] {
  const extra = selected && !WATCH_DEFAULTS.includes(selected) ? [selected] : [];
  return [...extra, ...WATCH_DEFAULTS];
}

const SOURCE_LABEL: Record<SourceRow["source"], string> = {
  sec_filings: "SEC",
  finra_short_interest: "SI",
  finra_short_sale_volume: "VOL",
  finra_threshold_list: "THR",
};

function sourceText(row: SourceRow): string {
  if (row.status === "ok") return row.as_of ? row.as_of.slice(0, 10) : "ok";
  return row.status === "no_records" ? "none" : row.status;
}

// Watchlist: real delayed quotes from /api/quote, plus one page-load fan-out of
// SEC filings and the three FINRA datasets per symbol. Every source shows its own status.
export function WatchlistPanel({ selected, onSelect, symbols }: { selected: string; onSelect: (s: string) => void; symbols?: string[] }) {
  const syms = symbols ?? watchlistFor(selected);
  const symKey = syms.join(",");
  const [prices, setPrices] = useState<Record<string, number | null>>({});
  const [sources, setSources] = useState<SourceRow[]>([]);
  const [sourceError, setSourceError] = useState(false);
  const [sourceAttempt, setSourceAttempt] = useState(0);
  useEffect(() => {
    const ctrl = new AbortController();
    // Deferred one tick: a StrictMode remount cancels the timer, so one page load sends one request.
    const id = setTimeout(() => {
      setSources([]);
      setSourceError(false);
      fetchWatchlistSources(symKey.split(","), ctrl.signal)
        .then(setSources)
        .catch(() => !ctrl.signal.aborted && setSourceError(true));
    }, 0);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [symKey, sourceAttempt]);
  useEffect(() => {
    const current = symKey.split(",");
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      const next: Record<string, number | null> = {};
      await Promise.all(
        current.map(async (sym) => {
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
  }, [symKey]);
  return (
    <div style={{ ...panel, height: "100%", overflowY: "auto" }}>
      <div style={h}>WATCHLIST · DELAYED</div>
      {sourceError && (
        <p role="alert" style={{ fontSize: 14, color: muted, marginBottom: 8 }}>
          We could not load this data. <button type="button" className="pill" onClick={() => setSourceAttempt((attempt) => attempt + 1)}>Retry</button>
        </p>
      )}
      {syms.map((sym) => (
        <button
          key={sym}
          onClick={() => onSelect(sym)}
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignContent: "center",
            alignItems: "center",
            lineHeight: "20px",
            width: "100%",
            height: 44,
            padding: "0 8px",
            textAlign: "left",
            background: sym === selected ? "var(--panel2, #151C29)" : "transparent",
            border: sym === selected ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
            borderRadius: 4,
            color: "var(--text, #E6ECF5)",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          <b style={{ fontWeight: 600 }}>{sym}</b>
          <span className="num">{prices[sym] == null ? "—" : prices[sym]!.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          <span style={{ flexBasis: "100%", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: "16px", color: muted, fontSize: 11 }}>
            DELAYED
            <span aria-label={`${sym} sources`} style={{ marginLeft: 4 }}>
              {sources
                .filter((row) => row.symbol === sym)
                .map((row) => (
                  <span
                    key={row.source}
                    title={[row.tool, row.status, row.as_of, row.ref?.record_id].filter(Boolean).join(" · ")}
                    style={{ color: row.status === "error" || row.status === "timeout" ? down : undefined }}
                  >
                    {`${SOURCE_LABEL[row.source]} ${sourceText(row)} `}
                  </span>
                ))}
            </span>
          </span>
        </button>
      ))}
    </div>
  );
}

// Financials: delayed window stats from the same /api/candles the chart draws
// plus the latest SEC filing link. No static table.
export function FinancialsPanel({ sym }: { sym: string }) {
  const { data, error: transportError, retry } = useLoad(sym, (signal) => fetchFilings(sym, signal));
  const filings = data?.items.slice(0, 3) ?? [];
  const error = transportError ?? data?.error;
  const loading = !data && !error;
  return (
    <div style={{ height: "100%", minHeight: 0, overflowY: "auto" }}>
      {loading && <span className="skeleton" role="status" aria-label="Loading filings" aria-busy="true" />}
      {error && (
        <p role="alert" style={{ fontSize: 14, color: muted }}>
          We could not load this data. <button type="button" className="pill" onClick={retry}>Retry</button>
        </p>
      )}
      {!loading && filings.length === 0 && !error && (
        <p style={{ fontSize: 14, color: muted }}>No filings for {sym}. Try asking: Summarize the latest {sym} disclosures.</p>
      )}
      {filings.map((f, i) => (
        <div key={i} style={{ fontSize: 14, marginBottom: 8, lineHeight: 1.4 }}>
          <span style={{ color: muted, fontSize: 12 }} title={`updated ${f.updated}`}>
            filed {f.filed}
          </span>
          <div>
            {f.link ? (
              <a href={f.link} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>
                {f.title}
              </a>
            ) : (
              f.title
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// News: Google News RSS headlines via /api/news. No hardcoded headlines.
export function NewsPanel({ sym }: { sym: string }) {
  const { data, error: transportError, retry } = useLoad(sym, (signal) => fetchNews(sym, signal));
  const items = data?.items.slice(0, 8) ?? [];
  const error = transportError ?? data?.error;
  const loading = !data && !error;
  return (
    <div style={{ height: "100%", minHeight: 0, overflowY: "auto" }}>
      {loading && <span className="skeleton" role="status" aria-label="Loading news" aria-busy="true" />}
      {error && (
        <p role="alert" style={{ fontSize: 14, color: muted }}>
          We could not load this data. <button type="button" className="pill" onClick={retry}>Retry</button>
        </p>
      )}
      {items.map((n, i) => (
        <div key={i} style={{ fontSize: 14, marginBottom: 8, lineHeight: 1.4 }}>
          <span style={{ color: muted, fontSize: 12 }}>
            {(n.published || "").slice(0, 16)} · {n.source}
          </span>
          <div>
            {n.link ? (
              <a href={n.link} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>
                {n.title}
              </a>
            ) : (
              n.title
            )}
          </div>
        </div>
      ))}
      {!loading && items.length === 0 && !error && (
        <p style={{ fontSize: 14, color: muted }}>No news for {sym}. Try asking: What moved {sym} today?</p>
      )}
    </div>
  );
}

export function AIPanel({
  events,
  input,
  setInput,
  onSend,
  sym,
  busy,
  debug,
}: {
  events: AgentEvent[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  sym: string;
  busy: boolean;
  debug: boolean;
}) {
  return (
    <div style={{ ...panel, padding: 0, display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* Mixed case on purpose: the ticker context reads "viewing NVDA", not "VIEWING". */}
      <div className="card-head" style={{ textTransform: "none" }}>AGENT · viewing {sym}{debug ? " · DEBUG" : ""}</div>
      <AgentConsole events={events} reasoning={events.some((event) => event.type === "reasoning_start")} debug={debug} />
      <form onSubmit={(event) => { event.preventDefault(); onSend(); }} style={{ display: "flex", gap: 6, padding: "0 12px 12px" }}>
        <input
          aria-label="Ask the research agent"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          disabled={busy}
          placeholder={busy ? "Research in progress…" : "Ask about this symbol…"}
          className="input"
          style={{ flex: 1, minWidth: 0 }}
        />
        <button type="submit" disabled={busy || !input.trim()} className="btn-primary" style={{ cursor: busy ? "wait" : "pointer" }}>
          {busy ? "Thinking…" : "Send"}
        </button>
      </form>
    </div>
  );
}
type SessionSummary = { session_id?: string; status?: string; updated_at?: string; query?: string };
const clip: React.CSSProperties = { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };

// Backend-state panel: durable sessions from research.sqlite + live snapshot.
export function SessionPanel({
  sessions,
  sessionId,
  onSelect,
  snapshot,
}: {
  sessions: SessionSummary[];
  sessionId: string | null;
  onSelect: (id: string | null) => void;
  snapshot: ResearchSnapshot | null;
}) {
  const jobs = snapshot?.jobs ?? [];
  const open = jobs.filter((j) => j.status === "running" || j.status === "queued");
  return (
    <div style={panel}>
      <div style={h}>BACKEND STATE</div>
      {sessions.length === 0 && (
        <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>no research sessions yet</div>
      )}
      {sessions.slice(0, 8).map((s) => (
        <button
          key={s.session_id ?? "?"}
          title={s.session_id}
          onClick={() => onSelect(s.session_id ? (s.session_id === sessionId ? null : s.session_id) : null)}
          style={{
            display: "block",
            width: "100%",
            textAlign: "left",
            padding: "4px 6px",
            background: s.session_id === sessionId ? "var(--panel2, #151C29)" : "transparent",
            border: s.session_id === sessionId ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
            borderRadius: 4,
            color: "var(--text, #D5DBE5)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          <div style={{ ...clip, fontSize: 14, fontWeight: 600 }}>
            {sessionTitle(s.query ?? "") || (s.session_id ?? "?").slice(0, 8)}
          </div>
          <div style={{ ...clip, color: "var(--muted, #7A8599)", fontSize: 12 }}>
            {[s.status ?? "?", s.updated_at && ago(s.updated_at, Date.now())].filter(Boolean).join(" · ")}
          </div>
        </button>
      ))}
      {snapshot && !snapshot.error && (
        <div style={{ fontSize: 12, marginTop: 6, color: "var(--muted, #7A8599)" }}>
          jobs {jobs.length} · open {open.length} · {String(snapshot.session?.status ?? "?")}
        </div>
      )}
      {snapshot?.error && <div style={{ fontSize: 12, color: down }}>{snapshot.error}</div>}
    </div>
  );
}

// Log tail: hash-chained journal events polled from research.sqlite.
export function JournalPanel({ journal, seq }: { journal: JournalPage | null; seq: number }) {
  const events = journal?.events ?? [];
  return (
    <div style={panel}>
      <div style={h}>LOGS · seq {seq}</div>
      {journal?.error && <div style={{ fontSize: 12, color: down }}>{journal.error}</div>}
      {!journal && <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>select a session for logs</div>}
      {journal && !journal.error && events.length === 0 && (
        <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>no journal rows yet</div>
      )}
      <div style={{ maxHeight: 220, overflowY: "auto", fontSize: 12 }}>
        {events.slice(-60).map((e) => (
          <div key={e.event_id ?? `${e.sequence}`} style={{ marginBottom: 4 }}>
            <span style={{ color: "var(--muted, #7A8599)" }}>
              #{e.sequence} {e.event_type}
            </span>{" "}
            <span style={{ color: "var(--text, #D5DBE5)" }}>{e.actor_type}:{e.actor_id}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
