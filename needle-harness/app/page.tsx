"use client";

import { useEffect, useState } from "react";
import { AgentConsole } from "@/components/agent-console";
import { PromptBox } from "@/components/prompt-box";
import type { AgentEvent } from "@/lib/agent/types";

// ponytail: stall watchdog — no chunk for this long aborts the stream; mirrors muse FETCH_TIMEOUT_MS.
const STALL_TIMEOUT_MS = 120_000;

export default function Home() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [debug, setDebug] = useState(false);
  const reasoning = events.some((e) => e.type === "reasoning_start");

  // Server .env flag: STOCKBOT_DEBUG=1 renders unfiltered working logs.
  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => setDebug(h?.debug === true))
      .catch(() => { });
  }, []);

  async function ask(prompt: string) {
    setBusy(true);
    setEvents([{ type: "agent_start", prompt }]);
    const ctrl = new AbortController();
    let terminal = false;
    let stalled = false;
    let timer: number | undefined = undefined;
    const arm = () => {
      clearTimeout(timer);
      timer = window.setTimeout(() => {
        stalled = true;
        ctrl.abort();
      }, STALL_TIMEOUT_MS);
    };
    const push = (ev: AgentEvent) => {
      if (ev.type === "done" || ev.type === "error" || ev.type === "failed") terminal = true;
      setEvents((prev) => [...prev, ev]);
    };
    arm();
    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        push({ type: "error", message: `request failed: ${res.status}` });
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (; ;) {
        const { done, value } = await reader.read();
        if (done) break;
        arm();
        buf += dec.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of frame.split("\n")) {
            const t = line.trim();
            if (!t.startsWith("data:")) continue;
            try {
              const ev = JSON.parse(t.slice(5)) as AgentEvent;
              if (ev.type !== "agent_start") push(ev);
            } catch {
              // Partial frame; next chunk completes it.
            }
          }
        }
      }
      if (!terminal) push({ type: "error", message: "stream ended without a terminal event" });
    } catch (err) {
      if (!terminal) {
        push({
          type: "error",
          message: stalled
            ? `stream stalled: no data for ${STALL_TIMEOUT_MS / 1000}s`
            : err instanceof Error
              ? err.message
              : String(err),
        });
      }
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  return (
    <main className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2 text-sm">
        <span className="font-bold tracking-widest">
          STOCKBOT{debug && <span className="ml-2 rounded border border-amber-900 px-1 text-[10px] text-amber-400">DEBUG</span>}
        </span>
        <span className="text-zinc-500">
          LOCAL <span className="text-green-500">●</span>
        </span>
      </header>
      <AgentConsole events={events} reasoning={reasoning} debug={debug} />
      <PromptBox onSubmit={ask} busy={busy} />
    </main>
  );
}
