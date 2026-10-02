"use client";

import { useEffect, useState } from "react";
import TradingChart from "@/components/trading-chart";
import {
  WatchlistPanel,
  FinancialsPanel,
  NewsPanel,
  PositionsPanel,
  AIPanel,
  type ChatMsg,
  type AgentInfo,
} from "@/components/market-panels";
import { SYMBOLS, TICKER_ITEMS } from "@/lib/market-mock";

const AGENTS_MOCK: AgentInfo[] = [
  { name: "Scanner", task: "gap + volume scan", active: true },
  { name: "Analyst", task: "earnings summarizer", active: true },
  { name: "Risk", task: "position limits", active: false },
];

export default function Home() {
  const [selected, setSelected] = useState("NVDA");
  const [chat, setChat] = useState<ChatMsg[]>([
    { role: "ai", text: "Terminal online. Ask about any symbol, e.g. NVDA earnings risk." },
  ]);
  const [input, setInput] = useState("");
  const [clock, setClock] = useState("--:--:--");

  useEffect(() => {
    setClock(new Date().toLocaleTimeString("en-GB"));
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-GB")), 1000);
    return () => clearInterval(id);
  }, []);

  function onSend() {
    const q = input.trim();
    if (!q) return;
    const m = SYMBOLS[selected];
    setChat((c) => [
      ...c,
      { role: "user", text: q },
      { role: "ai", text: `${selected} @ ${m.base} (${m.chg >= 0 ? "+" : ""}${m.chg}%) — mocked note: ${m.name}, 52W ${m.low52}–${m.high52}.` },
    ]);
    setInput("");
  }

  const m = SYMBOLS[selected];

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
          padding: "8px 12px",
          borderBottom: "1px solid var(--border, #1E2635)",
          background: "var(--panel, #11161F)",
        }}
      >
        <b style={{ letterSpacing: "0.12em" }}>NEEDLE TERMINAL</b>
        <span style={{ color: "var(--muted, #7A8599)", fontSize: 12 }}>
          <span style={{ color: "var(--up, #26A69A)" }}>● LIVE</span> · {clock}
        </span>
      </header>

      <div
        style={{
          overflow: "hidden",
          whiteSpace: "nowrap",
          borderBottom: "1px solid var(--border, #1E2635)",
          padding: "4px 0",
          fontSize: 12,
        }}
      >
        <div style={{ display: "inline-block", animation: "tick 40s linear infinite" }}>
          {[...TICKER_ITEMS, ...TICKER_ITEMS].map((t, i) => (
            <span key={i} style={{ marginRight: 18 }}>
              <b>{t.sym}</b> {t.val}{" "}
              <span style={{ color: t.chg.startsWith("-") ? "var(--down, #EF5350)" : "var(--up, #26A69A)" }}>
                {t.chg}
              </span>
            </span>
          ))}
        </div>
      </div>

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
          <WatchlistPanel selected={selected} onSelect={setSelected} />
        </div>

        <div style={{ overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ height: 560, flexShrink: 0 }}>
            <TradingChart sym={selected} base={m.base} volLabel={m.vol} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
            <FinancialsPanel sym={selected} />
            <NewsPanel sym={selected} />
            <PositionsPanel />
          </div>
        </div>

        <div style={{ minHeight: 0, display: "flex" }}>
          <AIPanel chat={chat} input={input} setInput={setInput} onSend={onSend} agents={AGENTS_MOCK} />
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
        <span>mock feed · no backend</span>
        <span>
          {selected} · {Object.keys(SYMBOLS).length} symbols · {(Object.values(SYMBOLS).reduce((a, s) => a + s.chg, 0) / Object.keys(SYMBOLS).length).toFixed(2)}% avg
        </span>
      </footer>
    </main>
  );
}
