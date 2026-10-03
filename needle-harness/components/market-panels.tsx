"use client";

import { useEffect, useState } from "react";
import { fetchFilings, fetchNews, fetchQuote, type FilingItem, type NewsItem } from "@/lib/market-data";

export type ChatMsg = { role: "user" | "ai"; text: string };

const panel: React.CSSProperties = {
  background: "var(--panel, #11161F)",
  border: "1px solid var(--border, #1E2635)",
  borderRadius: 6,
  padding: 10,
  overflow: "hidden",
};
const h: React.CSSProperties = {
  fontSize: 11,
  letterSpacing: "0.08em",
  color: "var(--muted, #7A8599)",
  marginBottom: 8,
  fontWeight: 700,
};

const WATCH_DEFAULTS = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG"];

// Watchlist: real delayed quotes from /api/quote. The selected symbol is
// always present even when it isn't in the defaults (any-ticker search).
export function WatchlistPanel({ selected, onSelect }: { selected: string; onSelect: (s: string) => void }) {
  const extra = selected && !WATCH_DEFAULTS.includes(selected) ? [selected] : [];
  const syms = [...extra, ...WATCH_DEFAULTS];
  const symKey = syms.join(",");
  const [prices, setPrices] = useState<Record<string, number | null>>({});
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
    <div style={panel}>
      <div style={h}>WATCHLIST · DELAYED</div>
      {syms.map((sym) => (
        <button
          key={sym}
          onClick={() => onSelect(sym)}
          style={{
            display: "flex",
            justifyContent: "space-between",
            width: "100%",
            padding: "6px 6px",
            background: sym === selected ? "var(--panel2, #151C29)" : "transparent",
            border: sym === selected ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
            borderRadius: 4,
            color: "var(--text, #D5DBE5)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          <b>{sym}</b>
          <span style={{ textAlign: "right" }}>
            <div>{prices[sym] == null ? "—" : (prices[sym] as number).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            <div style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>DELAYED</div>
          </span>
        </button>
      ))}
    </div>
  );
}

// Filings: latest SEC filings for the viewed symbol. No static table.
export function FinancialsPanel({ sym }: { sym: string }) {
  const [filings, setFilings] = useState<FilingItem[]>([]);
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    fetchFilings(sym, ctrl.signal)
      .then((page) => alive && setFilings(page.items.slice(0, 3)))
      .catch(() => { });
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [sym]);
  return (
    <div style={panel}>
      <div style={h}>FILINGS · {sym}</div>
      {filings.length === 0 && (
        <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>no SEC filings found</div>
      )}
      {filings.map((f, i) => (
        <div key={i} style={{ fontSize: 12, marginBottom: 8 }}>
          <span style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>{f.published.slice(0, 10)}</span>
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
  const [items, setItems] = useState<NewsItem[]>([]);
  const [stale, setStale] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    fetchNews(sym, ctrl.signal)
      .then((page) => {
        if (!alive) return;
        setItems(page.items.slice(0, 8));
        setStale(page.error ?? null);
      })
      .catch(() => {
        if (alive) setStale("news unavailable");
      });
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [sym]);
  return (
    <div style={panel}>
      <div style={h}>NEWS · {sym}</div>
      {stale && <div style={{ fontSize: 11, color: "var(--muted, #7A8599)", marginBottom: 8 }}>{stale}</div>}
      {items.map((n, i) => (
        <div key={i} style={{ fontSize: 12, marginBottom: 8 }}>
          <span style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>
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
      {items.length === 0 && !stale && (
        <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>loading headlines…</div>
      )}
    </div>
  );
}

// Chat shell kept; send shows the delayed quote for the viewed symbol.
// The agent itself is NOT wired: GET /api/viewer-context?sym=X is the
// one-fetch plug-in point the future agent loop reads for viewer context.
export function AIPanel({
  chat,
  input,
  setInput,
  onSend,
  sym,
}: {
  chat: ChatMsg[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  sym: string;
}) {
  return (
    <div style={{ ...panel, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={h}>AGENT · viewing {sym} · unwired</div>
      <div style={{ fontSize: 11, color: "var(--muted, #7A8599)", marginBottom: 8 }}>
        context: GET /api/viewer-context?sym={sym}
      </div>
      <div style={{ flex: 1, overflowY: "auto", fontSize: 12, marginBottom: 8 }}>
        {chat.map((m, i) => (
          <div key={i} style={{ marginBottom: 6 }}>
            <span style={{ color: m.role === "ai" ? "var(--accent, #2962FF)" : "var(--up, #26A69A)", fontWeight: 700 }}>
              {m.role === "ai" ? "AI" : "YOU"}:{" "}
            </span>
            {m.text}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSend()}
          placeholder="Ask… (agent not wired yet)"
          style={{
            flex: 1,
            background: "var(--panel2, #151C29)",
            border: "1px solid var(--border, #1E2635)",
            borderRadius: 4,
            color: "var(--text, #D5DBE5)",
            padding: "6px 8px",
            fontSize: 12,
          }}
        />
        <button
          onClick={onSend}
          style={{
            background: "var(--accent, #2962FF)",
            color: "#fff",
            border: 0,
            borderRadius: 4,
            padding: "6px 12px",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
