"use client";

import { useEffect, useState } from "react";
import TradingChart from "@/components/trading-chart";
import {
  WatchlistPanel,
  FinancialsPanel,
  NewsPanel,
  AIPanel,
  type ChatMsg,
} from "@/components/market-panels";
import { fetchQuote, searchTickers, type SearchHit } from "@/lib/market-data";

export default function Home() {
  const [selected, setSelected] = useState("NVDA");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [chat, setChat] = useState<ChatMsg[]>([
    { role: "ai", text: "Terminal online · delayed data. Type a ticker above — any US symbol works." },
  ]);
  const [input, setInput] = useState("");
  const [clock, setClock] = useState("--:--:--");

  // Clock ticks on its own interval — no synchronous setState in the effect body.
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-GB")), 1000);
    return () => clearInterval(id);
  }, []);

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
    if (!q) return;
    setChat((c) => [...c, { role: "user", text: q }]);
    setInput("");
    try {
      const quote = await fetchQuote(selected);
      const px = typeof quote.price === "number" ? quote.price.toFixed(2) : "—";
      setChat((c) => [
        ...c,
        { role: "ai", text: `${selected} @ ${px} (delayed) — agent context at /api/viewer-context?sym=${selected} (unwired).` },
      ]);
    } catch {
      setChat((c) => [...c, { role: "ai", text: `${selected}: quote unavailable — delayed feed down.` }]);
    }
  }

  return (
    <main
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "var(--bg, #0A0E14)",
        color: "var(--text, #D5DBE5)",
        fontSize: 13,
      }}
    >
      <style>{`@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}`}</style>

      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          borderBottom: "1px solid var(--border, #1E2635)",
          background: "var(--panel, #11161F)",
        }}
      >
        <b style={{ letterSpacing: "0.12em" }}>NEEDLE TERMINAL</b>
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
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid var(--border, #1E2635)",
              background: "var(--panel2, #151C29)",
              color: "var(--text, #D5DBE5)",
              width: 200,
            }}
          />
          {hits.length > 0 && (
            <div
              style={{
                position: "absolute",
                top: 28,
                left: 0,
                right: 0,
                background: "var(--panel2, #151C29)",
                border: "1px solid var(--border, #1E2635)",
                borderRadius: 6,
                overflow: "hidden",
                zIndex: 10,
              }}
            >
              {hits.map((hit) => (
                <button
                  key={hit.sym}
                  onClick={() => pick(hit.sym)}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    padding: "6px 10px",
                    background: "transparent",
                    border: 0,
                    color: "var(--text, #D5DBE5)",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  <b>{hit.sym}</b> <span style={{ opacity: 0.6 }}>{hit.name ?? ""}</span>
                </button>
              ))}
            </div>
          )}
        </span>
        <span style={{ color: "var(--muted, #7A8599)", fontSize: 12 }}>
          <span style={{ color: "var(--down, #EF5350)" }}>○ DELAYED</span> · {clock}
        </span>
      </header>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "220px 1fr 300px",
          gap: 8,
          padding: 8,
        }}
      >
        <div style={{ overflowY: "auto", minHeight: 0 }}>
          <WatchlistPanel selected={selected} onSelect={pick} />
        </div>

        <div style={{ overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ height: 560, flexShrink: 0 }}>
            <TradingChart sym={selected} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <FinancialsPanel sym={selected} />
            <NewsPanel sym={selected} />
          </div>
        </div>

        <div style={{ minHeight: 0, display: "flex" }}>
          <AIPanel chat={chat} input={input} setInput={setInput} onSend={() => void onSend()} sym={selected} />
        </div>
      </div>

      <footer
        style={{
          borderTop: "1px solid var(--border, #1E2635)",
          background: "var(--panel, #11161F)",
          color: "var(--muted, #7A8599)",
          fontSize: 11,
          padding: "4px 12px",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>delayed feed · Yahoo + SEC + Google News RSS · agent context: /api/viewer-context (unwired)</span>
        <span>{selected}</span>
      </footer>
    </main>
  );
}
