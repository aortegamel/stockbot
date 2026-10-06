"use client";

import FluidOrb from "@/components/ui/fluid-orb";
import type { Candle, WatchItem } from "@/lib/market-client";
import type { JournalPage, ResearchSnapshot } from "@/lib/research-client";

export type ChatMsg = { role: "user" | "ai" | "ai-stream"; text: string };

const up = "var(--up, #26A69A)";
const down = "var(--down, #EF5350)";
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

// Watchlist renders caller-supplied delayed quotes only — no local jitter, no RNG.
export function WatchlistPanel({
  items,
  selected,
  onSelect,
}: {
  items: WatchItem[];
  selected: string;
  onSelect: (s: string) => void;
}) {
  return (
    <div style={panel}>
      <div style={h}>WATCHLIST · DELAYED</div>
      {items.length === 0 && (
        <div style={{ fontSize: 12, color: "var(--muted, #7A8599)" }}>loading delayed quotes…</div>
      )}
      {items.map((t) => (
        <button
          key={t.sym}
          onClick={() => onSelect(t.sym)}
          style={{
            display: "flex",
            justifyContent: "space-between",
            width: "100%",
            padding: "6px 6px",
            background: t.sym === selected ? "var(--panel2, #151C29)" : "transparent",
            border: t.sym === selected ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
            borderRadius: 4,
            color: "var(--text, #D5DBE5)",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          <span>
            <b>{t.sym}</b> <span style={{ color: "var(--muted, #7A8599)" }}>{t.name}</span>
          </span>
          <span style={{ textAlign: "right" }}>
            <div>{t.price === null ? "—" : t.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            <div style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>DELAYED</div>
          </span>
        </button>
      ))}
    </div>
  );
}

// Financials derive from the same delayed candles the chart draws — no static table.
export function FinancialsPanel({ sym, candles, price }: { sym: string; candles: Candle[]; price: number | null }) {
  const closes = candles.map((c) => c.close);
  const last = price ?? (closes.length ? closes[closes.length - 1] ?? null : null);
  const hi = closes.length ? Math.max(...closes) : null;
  const lo = closes.length ? Math.min(...closes) : null;
  const chg = closes.length > 1 && closes[0] ? (((closes[closes.length - 1] ?? 0) - (closes[0] ?? 0)) / (closes[0] ?? 1)) * 100 : null;
  const rows: [string, string][] = [
    ["Last (delayed)", last === null ? "—" : last.toLocaleString(undefined, { maximumFractionDigits: 2 })],
    ["Range (window)", hi === null || lo === null ? "—" : `${lo.toFixed(2)} — ${hi.toFixed(2)}`],
    ["Change (window)", chg === null ? "—" : `${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`],
    ["Rows", String(candles.length)],
  ];
  return (
    <div style={panel}>
      <div style={h}>SESSION · {sym} · DELAYED</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 12 }}>
        {rows.map(([k, v]) => (
          <div key={k}>
            <div style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>{k}</div>
            <div>{v}</div>
          </div>
        ))}
      </div>
      {candles.length === 0 && (
        <div style={{ fontSize: 11, color: "var(--muted, #7A8599)", marginTop: 6 }}>no delayed rows yet</div>
      )}
    </div>
  );
}

type SessionSummary = { session_id?: string; status?: string; updated_at?: string; query?: string };

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
            fontSize: 11,
          }}
        >
          <b>{(s.session_id ?? "?").slice(0, 8)}</b>{" "}
          <span style={{ color: "var(--muted, #7A8599)" }}>{s.status ?? "?"}</span>
          <div style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>{(s.query ?? "").slice(0, 60)}</div>
        </button>
      ))}
      {snapshot && !snapshot.error && (
        <div style={{ fontSize: 11, marginTop: 6, color: "var(--muted, #7A8599)" }}>
          jobs {jobs.length} · open {open.length} · {String(snapshot.session?.status ?? "?")}
        </div>
      )}
      {snapshot?.error && <div style={{ fontSize: 11, color: down }}>{snapshot.error}</div>}
    </div>
  );
}

// Log tail: hash-chained journal events polled from research.sqlite.
export function JournalPanel({ journal, seq }: { journal: JournalPage | null; seq: number }) {
  const events = journal?.events ?? [];
  return (
    <div style={panel}>
      <div style={h}>LOGS · seq {seq}</div>
      {journal?.error && <div style={{ fontSize: 11, color: down }}>{journal.error}</div>}
      {!journal && <div style={{ fontSize: 11, color: "var(--muted, #7A8599)" }}>select a session for logs</div>}
      {journal && !journal.error && events.length === 0 && (
        <div style={{ fontSize: 11, color: "var(--muted, #7A8599)" }}>no journal rows yet</div>
      )}
      <div style={{ maxHeight: 220, overflowY: "auto", fontSize: 11 }}>
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

export function AIPanel({
  chat,
  input,
  setInput,
  onSend,
  busy,
}: {
  chat: ChatMsg[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  busy: boolean;
}) {
  return (
    <div style={{ ...panel, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ ...h, display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <FluidOrb size={28} color={busy ? "#26A69A" : "#2962FF"} style={{ flexShrink: 0 }} />
        <span>AGENT · SSE /api/agent{busy ? " · thinking…" : ""}</span>
      </div>
      <div style={{ flex: 1, overflowY: "auto", fontSize: 12, marginBottom: 8 }}>
        {chat.map((m, i) => (
          <div key={i} style={{ marginBottom: 6, opacity: m.role === "ai-stream" ? 0.9 : 1 }}>
            <span style={{ color: m.role === "user" ? up : "var(--accent, #2962FF)", fontWeight: 700 }}>
              {m.role === "user" ? "YOU" : "AI"}:{" "}
            </span>
            {m.text}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && onSend()}
          placeholder="Ask…"
          disabled={busy}
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
          disabled={busy}
          style={{
            background: "var(--accent, #2962FF)",
            color: "#fff",
            border: 0,
            borderRadius: 4,
            padding: "6px 12px",
            cursor: busy ? "wait" : "pointer",
            fontSize: 12,
            opacity: busy ? 0.6 : 1,
          }}
        >
          {busy ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
