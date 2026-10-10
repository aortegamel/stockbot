"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import TradingChart from "@/components/trading-chart";
import {
  WatchlistPanel,
  FinancialsPanel,
  SessionPanel,
  JournalPanel,
  AIPanel,
  type ChatMsg,
} from "@/components/market-panels";
import { fetchCandles, fetchWatchlist, type Candle, type WatchItem } from "@/lib/market-client";
import { fetchJournal, fetchSnapshot, listSessions, type JournalPage, type ResearchSnapshot } from "@/lib/research-client";
import type { AgentEvent } from "@/lib/agent/types";

const TABS = ["5M", "15M", "1H", "4H", "1D", "1W"] as const;

function appendAiText(chat: ChatMsg[], text: string): ChatMsg[] {
  const out = [...chat];
  const last = out[out.length - 1];
  if (last && last.role === "ai-stream") {
    out[out.length - 1] = { role: "ai-stream", text: last.text + text };
  } else {
    out.push({ role: "ai-stream", text });
  }
  return out;
}

export default function Terminal() {
  const [selected, setSelected] = useState("NVDA");
  const [tab, setTab] = useState<string>("1D");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [price, setPrice] = useState<number | null>(null);
  const [candlesError, setCandlesError] = useState<string | null>(null);
  const [watch, setWatch] = useState<WatchItem[]>([]);
  const [chat, setChat] = useState<ChatMsg[]>([
    { role: "ai", text: "Terminal online. Quotes are DELAYED. Ask about any symbol, e.g. NVDA earnings risk." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [clock, setClock] = useState("--:--:--");
  const [sessions, setSessions] = useState<{ session_id?: string; status?: string; updated_at?: string; query?: string }[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<ResearchSnapshot | null>(null);
  const [journal, setJournal] = useState<JournalPage | null>(null);
  const [seq, setSeq] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setClock(new Date().toLocaleTimeString("en-GB"));
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-GB")), 1000);
    return () => clearInterval(id);
  }, []);

  // Watchlist: real closes from the same-origin candles proxy, refreshed every 60s.
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      try {
        const items = await fetchWatchlist(ctrl.signal);
        if (alive) {
          setWatch(items);
        }
      } catch {
        if (alive) {
          setWatch([]);
        }
      }
    }
    void load();
    const id = setInterval(() => void load(), 60_000);
    return () => {
      alive = false;
      ctrl.abort();
      clearInterval(id);
    };
  }, []);

  // Candles for the selected symbol + timeframe.
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      setCandlesError(null);
      try {
        const r = await fetchCandles(selected, tab, ctrl.signal);
        if (!alive) {
          return;
        }
        setCandles(r.candles ?? []);
        const last = r.candles?.length ? r.candles[r.candles.length - 1]?.close ?? null : null;
        setPrice(r.price ?? last);
      } catch (err) {
        if (!alive) {
          return;
        }
        setCandles([]);
        setPrice(null);
        setCandlesError(err instanceof Error ? err.message : String(err));
      }
    }
    void load();
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [selected, tab]);

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

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const onSend = useCallback(async () => {
    const q = input.trim();
    if (!q || busy) {
      return;
    }
    setBusy(true);
    setChat((c) => [...c, { role: "user", text: q }]);
    setInput("");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: `[${selected}] ${q}` }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        setChat((c) => [...c, { role: "ai", text: `request failed: ${res.status}` }]);
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let terminal = false;
      for (; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buf += dec.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of frame.split("\n")) {
            const t = line.trim();
            if (!t.startsWith("data:")) {
              continue;
            }
            let ev: AgentEvent;
            try {
              ev = JSON.parse(t.slice(5)) as AgentEvent;
            } catch {
              continue;
            }
            if (ev.type === "answer_delta") {
              setChat((c) => appendAiText(c, ev.text));
            } else if (ev.type === "needle_decision") {
              setChat((c) => [...c, { role: "ai", text: `● ${ev.tool ?? "(escalate)"} ${ev.confidence === null ? "" : `${Math.round(ev.confidence * 100)}%`}` }]);
            } else if (ev.type === "tool_result") {
              setChat((c) => [...c, { role: "ai", text: `├─ ${ev.tool} — ${ev.preview.slice(0, 160)}` }]);
            } else if (ev.type === "tool_failed" || ev.type === "failed" || ev.type === "error") {
              const msg = ev.type === "error" ? ev.message : `${ev.category}: ${ev.type === "failed" ? ev.message : ev.preview}`;
              setChat((c) => [...c, { role: "ai", text: `✗ ${msg.slice(0, 300)}` }]);
              terminal = terminal || ev.type === "failed" || ev.type === "error";
            } else if (ev.type === "done") {
              terminal = true;
              const m = ev.metrics;
              setChat((c) => [
                ...c,
                { role: "ai", text: `── done: tools ${m.tools.calls} · evidence ${m.evidence.count} · ${Math.round(m.totalMs)}ms` },
              ]);
            }
          }
        }
      }
      if (!terminal) {
        setChat((c) => [...c, { role: "ai", text: "stream ended without a terminal event" }]);
      }
    } catch (err) {
      if (!ctrl.signal.aborted) {
        setChat((c) => [...c, { role: "ai", text: err instanceof Error ? err.message : String(err) }]);
      }
    } finally {
      setBusy(false);
    }
  }, [input, busy, selected]);

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
          <span style={{ color: "var(--down, #EF5350)" }}>○ DELAYED</span> · {clock}
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
          {[...watch, ...watch].map((t, i) => (
            <span key={i} style={{ marginRight: 18 }}>
              <b>{t.sym}</b> {t.price === null ? "—" : t.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}{" "}
              <span style={{ color: "var(--muted, #7A8599)", fontSize: 10 }}>DELAYED</span>
            </span>
          ))}
          {watch.length === 0 && <span style={{ color: "var(--muted, #7A8599)" }}>loading delayed quotes…</span>}
        </div>
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "220px 1fr 340px",
          gap: 8,
          padding: 8,
        }}
      >
        <div style={{ overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <WatchlistPanel items={watch} selected={selected} onSelect={setSelected} />
          <SessionPanel sessions={sessions} sessionId={sessionId} onSelect={setSessionId} snapshot={snapshot} />
        </div>

        <div style={{ overflowY: "auto", minHeight: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ height: 560, flexShrink: 0 }}>
            <TradingChart sym={selected} candles={candles} price={price} tab={tab} onTab={setTab} tabs={TABS} error={candlesError} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <FinancialsPanel sym={selected} candles={candles} price={price} />
            <JournalPanel journal={journal} seq={seq} />
          </div>
        </div>

        <div style={{ minHeight: 0, display: "flex" }}>
          <AIPanel chat={chat} input={input} setInput={setInput} onSend={() => void onSend()} busy={busy} />
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
        <span>delayed feed · backend state via research snapshot + journal</span>
        <span>
          {selected} · {candles.length} rows{sessionId ? ` · session ${sessionId.slice(0, 8)}` : ""}
        </span>
      </footer>
    </main>
  );
}
