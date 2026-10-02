import { useEffect, useState } from "react";
import { SYMBOLS, NEWS, POSITIONS } from "../lib/market-mock";

export type ChatMsg = { role: "user" | "ai"; text: string };
export type AgentInfo = { name: string; task: string; active: boolean };

const up = "var(--up, #26A69A)";
const down = "var(--down, #EF5350)";
const panel: React.CSSProperties = {
  background: "var(--panel, #11161F)",
  border: "1px solid var(--border, #1E2635)",
  borderRadius: 6,
  padding: 16,
  overflow: "hidden",
};
const h: React.CSSProperties = {
  fontSize: 20,
  letterSpacing: "0.08em",
  color: "var(--muted, #93A0B8)",
  marginBottom: 10,
  fontWeight: 700,
};

export function WatchlistPanel({ selected, onSelect }: { selected: string; onSelect: (s: string) => void }) {
  const [prices, setPrices] = useState<Record<string, number>>(() =>
    Object.fromEntries(Object.entries(SYMBOLS).map(([k, v]) => [k, v.base])),
  );
  useEffect(() => {
    const id = setInterval(() => {
      setPrices((p) => {
        const n = { ...p };
        for (const k of Object.keys(n)) n[k] = n[k] * (1 + (Math.random() - 0.5) * 0.002);
        return n;
      });
    }, 1500);
    return () => clearInterval(id);
  }, []);
  return (
    <div style={{ ...panel, height: "100%", overflowY: "auto" }}>
      <div style={h}>WATCHLIST</div>
      {Object.entries(SYMBOLS).map(([sym, m]) => {
        const px = prices[sym] ?? m.base;
        const neg = m.chg < 0;
        return (
          <button
            key={sym}
            onClick={() => onSelect(sym)}
            style={{
              display: "flex",
              justifyContent: "space-between",
              width: "100%",
              padding: "10px 12px",
              background: sym === selected ? "var(--panel2, #151C29)" : "transparent",
              border: sym === selected ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
              borderRadius: 4,
              color: "var(--text, #E6ECF5)",
              cursor: "pointer",
              fontSize: 20,
            }}
          >
            <span>
              <b>{sym}</b> <span style={{ color: "var(--muted, #93A0B8)" }}>{m.name}</span>
            </span>
            <span style={{ textAlign: "right" }}>
              <div>{px.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
              <div style={{ color: neg ? down : up, fontSize: 20 }}>
                {neg ? "" : "+"}
                {m.chg}%
              </div>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function FinancialsPanel({ sym }: { sym: string }) {
  const m = SYMBOLS[sym];
  if (!m) return null;
  const pos = Math.min(100, Math.max(0, ((m.base - m.low52) / (m.high52 - m.low52)) * 100));
  const stats: [string, string][] = [
    ["Mkt Cap", m.mktCap],
    ["P/E", m.pe],
    ["EPS", m.eps],
    ["Revenue", m.rev],
    ["Div", m.div],
    ["Beta", m.beta],
    ["Shares", m.shares],
    ["ROE", m.roe],
    ["D/E", m.de],
  ];
  return (
    <div style={{ ...panel, height: "100%", minHeight: 0, overflowY: "auto" }}>
      <div style={h}>FINANCIALS · {sym}</div>
      <div style={{ fontSize: 20, color: "var(--muted, #93A0B8)", marginBottom: 4 }}>
        52W {m.low52} — {m.high52}
      </div>
      <div style={{ height: 6, background: "var(--panel2, #151C29)", borderRadius: 3, marginBottom: 8 }}>
        <div style={{ width: `${pos}%`, height: "100%", background: "var(--accent, #2962FF)", borderRadius: 3 }} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, fontSize: 20 }}>
        {stats.map(([k, v]) => (
          <div key={k}>
            <div style={{ color: "var(--muted, #93A0B8)", fontSize: 20 }}>{k}</div>
            <div>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function NewsPanel({ sym }: { sym: string }) {
  const items = [...(NEWS[sym] ?? []), ...(NEWS._global ?? [])].slice(0, 8);
  return (
    <div style={{ ...panel, height: "100%", minHeight: 0, overflowY: "auto" }}>
      <div style={h}>NEWS</div>
      {items.map((n, i) => (
        <div key={i} style={{ fontSize: 20, marginBottom: 12, lineHeight: 1.5 }}>
          <span style={{ color: "var(--muted, #93A0B8)", fontSize: 20 }}>
            {n.ts} · {n.cat}
          </span>
          <div>{n.headline}</div>
        </div>
      ))}
      {items.length === 0 && <div style={{ fontSize: 20, color: "var(--muted, #93A0B8)" }}>No news.</div>}
    </div>
  );
}

export function PositionsPanel() {
  const totVal = POSITIONS.reduce((a, p) => a + p.qty * p.last, 0);
  const totPnl = POSITIONS.reduce((a, p) => a + p.pnl, 0);
  return (
    <div style={{ ...panel, height: "100%", minHeight: 0, overflowY: "auto" }}>
      <div style={h}>POSITIONS</div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, fontSize: 20 }}>
        <div style={{ flex: 1, background: "var(--panel2, #151C29)", borderRadius: 4, padding: 8 }}>
          <div style={{ color: "var(--muted, #93A0B8)", fontSize: 20 }}>VALUE</div>
          <b>${totVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b>
        </div>
        <div style={{ flex: 1, background: "var(--panel2, #151C29)", borderRadius: 4, padding: 8 }}>
          <div style={{ color: "var(--muted, #93A0B8)", fontSize: 20 }}>P&amp;L</div>
          <b style={{ color: totPnl < 0 ? down : up }}>
            {totPnl < 0 ? "−" : "+"}${Math.abs(totPnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </b>
        </div>
      </div>
      <table style={{ width: "100%", fontSize: 20, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "var(--muted, #93A0B8)", textAlign: "left" }}>
            <th>Sym</th>
            <th>Qty</th>
            <th>Avg</th>
            <th>Last</th>
            <th>P&amp;L</th>
          </tr>
        </thead>
        <tbody>
          {POSITIONS.map((p) => (
            <tr key={p.sym} style={{ borderTop: "1px solid var(--border, #1E2635)" }}>
              <td>
                <b>{p.sym}</b>
              </td>
              <td>{p.qty}</td>
              <td>{p.avg}</td>
              <td>{p.last}</td>
              <td style={{ color: p.pnl < 0 ? down : up }}>
                {p.pnl < 0 ? "" : "+"}
                {p.pnl.toFixed(0)} ({p.pct}%)
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AIPanel({
  chat,
  input,
  setInput,
  onSend,
  agents,
}: {
  chat: ChatMsg[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  agents: AgentInfo[];
}) {
  return (
    <div style={{ ...panel, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={h}>AI AGENTS</div>
      {agents.map((a) => (
        <div key={a.name} style={{ fontSize: 20, marginBottom: 6 }}>
          <span style={{ color: a.active ? up : "var(--muted, #93A0B8)" }}>●</span> <b>{a.name}</b>{" "}
          <span style={{ color: "var(--muted, #93A0B8)" }}>{a.task}</span>
        </div>
      ))}
      <div style={{ ...h, marginTop: 8 }}>CHAT</div>
      <div style={{ flex: 1, overflowY: "auto", fontSize: 20, marginBottom: 8, lineHeight: 1.55 }}>
        {chat.map((m, i) => (
          <div key={i} style={{ marginBottom: 9 }}>
            <span style={{ color: m.role === "ai" ? "var(--accent, #2962FF)" : up, fontWeight: 700 }}>
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
          placeholder="Ask…"
          style={{
            flex: 1,
            background: "var(--panel2, #151C29)",
            border: "1px solid var(--border, #1E2635)",
            borderRadius: 4,
            color: "var(--text, #E6ECF5)",
            padding: "10px 12px",
            fontSize: 20,
          }}
        />
        <button
          onClick={onSend}
          style={{
            background: "var(--accent, #2962FF)",
            color: "#fff",
            border: 0,
            borderRadius: 4,
            padding: "10px 18px",
            cursor: "pointer",
            fontSize: 20,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
