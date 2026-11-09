import { useContext, useEffect, useState } from "react";
import { QUOTE_POLL_MS, fetchFilings, fetchNews, fetchQuote, fetchWatchlistSources, type SourceRow } from "../lib/market-data";

import { AgentConsole } from "./agent-console";
import type { WidgetOutcomes } from "./action-cards";
import { AskContextChip } from "./agent-event";
import { MarketRefreshContext, useLoad } from "./home-panels";
import type { AgentEvent, AskContext, ChatTurn } from "../../../needle-harness/lib/agent/types";
import type { ChatSummary } from "../lib/chat-client";
import type { JournalPage, ResearchSnapshot } from "../lib/research-client";
import { sessionTitle } from "../lib/session-title";
import { ago } from "../lib/time";

// Panels sit inside a glass card or tab, which already draws the surface and border.
const panel: React.CSSProperties = {
  padding: 16,
  overflow: "hidden",
};
const h: React.CSSProperties = {
  fontSize: "var(--fs-label)",
  letterSpacing: "0.06em",
  color: "var(--muted, #93A0B8)",
  marginBottom: 8,
  fontWeight: 600,
};
const muted = "var(--muted, #93A0B8)";
const down = "var(--down, #EF5350)";

// The selected symbol stays visible without changing the saved watchlist.
export function watchlistFor(selected: string, symbols: string[]): string[] {
  return selected && !symbols.includes(selected) ? [selected, ...symbols] : symbols;
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
  const syms = watchlistFor(selected, symbols ?? []);
  const refresh = useContext(MarketRefreshContext);
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
  }, [symKey, sourceAttempt, refresh]);
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
    const id = setInterval(() => void load(), QUOTE_POLL_MS);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, [symKey, refresh]);
  return (
    <div style={{ ...panel, height: "100%", overflowY: "auto" }}>
      <div style={h}>WATCHLIST · DELAYED</div>
      {sourceError && (
        <p role="alert" style={{ fontSize: "var(--fs-body)", color: muted, marginBottom: 8 }}>
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
            fontSize: "var(--fs-body)",
          }}
        >
          <b style={{ fontWeight: 600 }}>{sym}</b>
          <span className="num">{prices[sym] == null ? "—" : prices[sym]!.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          <span style={{ flexBasis: "100%", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: "16px", color: muted, fontSize: "var(--fs-xs)" }}>
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
        <p role="alert" style={{ fontSize: "var(--fs-body)", color: muted }}>
          We could not load this data. <button type="button" className="pill" onClick={retry}>Retry</button>
        </p>
      )}
      {!loading && filings.length === 0 && !error && (
        <p style={{ fontSize: "var(--fs-body)", color: muted }}>No filings for {sym}. Try asking: Summarize the latest {sym} disclosures.</p>
      )}
      {filings.map((f, i) => (
        <div key={i} style={{ fontSize: "var(--fs-body)", marginBottom: 8, lineHeight: 1.4 }}>
          <span style={{ color: muted, fontSize: "var(--fs-label)" }} title={`updated ${f.updated}`}>
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
        <p role="alert" style={{ fontSize: "var(--fs-body)", color: muted }}>
          We could not load this data. <button type="button" className="pill" onClick={retry}>Retry</button>
        </p>
      )}
      {items.map((n, i) => (
        <div key={i} style={{ fontSize: "var(--fs-body)", marginBottom: 8, lineHeight: 1.4 }}>
          <span style={{ color: muted, fontSize: "var(--fs-label)" }}>
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
        <p style={{ fontSize: "var(--fs-body)", color: muted }}>No news for {sym}. Try asking: What moved {sym} today?</p>
      )}
    </div>
  );
}

export function AIPanel({
  events,
  turns = [],
  onNewChat,
  chats = [],
  activeChat,
  onOpenChat,
  onDeleteChat,
  input,
  setInput,
  research,
  setResearch,
  context,
  onClearContext,
  onSend,
  sym,
  busy,
  debug,
  widgetOutcomes,
}: {
  events: AgentEvent[];
  // Stored messages of the open chat, oldest first. The console shows only the live turn.
  turns?: ChatTurn[];
  onNewChat?: () => void;
  // Stored conversations of this installation, newest first, with the open one marked.
  chats?: ChatSummary[];
  activeChat?: string | null;
  onOpenChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  input: string;
  setInput: (v: string) => void;
  research: boolean;
  setResearch: (value: boolean) => void;
  // A feed item attached to the next Send; Send works with an empty input while it is set.
  context?: AskContext | null;
  onClearContext?: () => void;
  onSend: () => void;
  sym: string;
  busy: boolean;
  debug: boolean;
  widgetOutcomes?: WidgetOutcomes;
}) {
  const empty = turns.length === 0 && events.length === 0;
  return (
    <div style={{ ...panel, padding: 0, display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* Mixed case on purpose: the ticker context reads "viewing NVDA", not "VIEWING". */}
      <div className="card-head" style={{ textTransform: "none", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <span>AGENT · viewing {sym}{debug ? " · DEBUG" : ""}</span>
        {onNewChat && <button type="button" className="pill" aria-label="New chat" disabled={empty} onClick={onNewChat}>New chat</button>}
      </div>
      {onOpenChat && chats.length > 0 && (
        <details style={{ padding: "0 12px 8px" }}>
          <summary style={{ cursor: "pointer", fontSize: "var(--fs-label)", color: muted }}>History · {chats.length}</summary>
          <div aria-label="Stored chats" style={{ maxHeight: 160, overflowY: "auto", marginTop: 4 }}>
            {chats.map((chat) => (
              <div key={chat.chatId} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <button
                  type="button"
                  onClick={() => onOpenChat(chat.chatId)}
                  aria-current={chat.chatId === activeChat ? "true" : undefined}
                  title={chat.title}
                  style={{
                    flex: 1, minWidth: 0, textAlign: "left", padding: "2px 4px", background: "transparent", cursor: "pointer",
                    color: chat.chatId === activeChat ? "var(--accent, #2962FF)" : "var(--text, #D5DBE5)",
                    border: "1px solid transparent", borderRadius: 4, fontSize: "var(--fs-label)", ...clip,
                  }}
                >
                  {chat.title}
                </button>
                {onDeleteChat && (
                  <button type="button" className="ghost" aria-label={`Delete chat ${chat.title}`} title="Delete this chat" onClick={() => onDeleteChat(chat.chatId)}>×</button>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
      {/* Backend debug mode must not expose diagnostic events in chat. */}
      <AgentConsole events={events} history={turns} reasoning={events.some((event) => event.type === "reasoning_start")} debug={false} widgetOutcomes={widgetOutcomes} />
      {context && <div style={{ padding: "0 12px 6px" }}><AskContextChip context={context} onRemove={onClearContext} /></div>}
      <form onSubmit={(event) => { event.preventDefault(); onSend(); }} style={{ display: "flex", gap: 6, padding: "0 12px 12px" }}>
        <input
          aria-label="Ask the research agent"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          disabled={busy}
          placeholder={busy ? "Research in progress…" : research ? "Deep research…" : "Ask about this symbol… (/research for deep research)"}
          className="input"
          style={{ flex: 1, minWidth: 0 }}
        />
        <label className="pill"><input type="checkbox" aria-label="Research mode" checked={research} onChange={(event) => setResearch(event.target.checked)} /> Research</label>
        <button type="submit" disabled={busy || (!input.trim() && !context)} className="btn-primary" style={{ cursor: busy ? "wait" : "pointer" }}>
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
        <div style={{ fontSize: "var(--fs-label)", color: "var(--muted, #7A8599)" }}>no research sessions yet</div>
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
            fontSize: "var(--fs-label)",
          }}
        >
          <div style={{ ...clip, fontSize: "var(--fs-body)", fontWeight: 600 }}>
            {sessionTitle(s.query ?? "") || (s.session_id ?? "?").slice(0, 8)}
          </div>
          <div style={{ ...clip, color: "var(--muted, #7A8599)", fontSize: "var(--fs-label)" }}>
            {[s.status ?? "?", s.updated_at && ago(s.updated_at, Date.now())].filter(Boolean).join(" · ")}
          </div>
        </button>
      ))}
      {snapshot && !snapshot.error && (
        <div style={{ fontSize: "var(--fs-label)", marginTop: 6, color: "var(--muted, #7A8599)" }}>
          jobs {jobs.length} · open {open.length} · {String(snapshot.session?.status ?? "?")}
        </div>
      )}
      {snapshot?.error && <div style={{ fontSize: "var(--fs-label)", color: down }}>{snapshot.error}</div>}
    </div>
  );
}

// Log tail: hash-chained journal events polled from research.sqlite.
export function JournalPanel({ journal, seq }: { journal: JournalPage | null; seq: number }) {
  const events = journal?.events ?? [];
  return (
    <div style={panel}>
      <div style={h}>LOGS · seq {seq}</div>
      {journal?.error && <div style={{ fontSize: "var(--fs-label)", color: down }}>{journal.error}</div>}
      {!journal && <div style={{ fontSize: "var(--fs-label)", color: "var(--muted, #7A8599)" }}>select a session for logs</div>}
      {journal && !journal.error && events.length === 0 && (
        <div style={{ fontSize: "var(--fs-label)", color: "var(--muted, #7A8599)" }}>no journal rows yet</div>
      )}
      <div style={{ maxHeight: 220, overflowY: "auto", fontSize: "var(--fs-label)" }}>
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
