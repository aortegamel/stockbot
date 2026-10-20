import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import type { FeedItem } from "../lib/market-data";
import { actionSummary, OpenWorkspace } from "./action-cards";
import { failureText } from "./agent-event";

const TOAST_MS = 5000;
export const SEEN_ALERTS_KEY = "stockbot.seen_alerts";
const SEEN_LIMIT = 500;
// This tab's ids: dedups even when storage is unavailable or full.
const seenHere = new Set<string>();

function seenAlerts(): string[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(SEEN_ALERTS_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

// A feed poll hands every item here: an unseen alert id gives one toast and, when granted, one system notification.
// Plain DOM so it works on any page, with or without the agent drawer open.
export function toastAlert(item: FeedItem): void {
  const seen = seenAlerts();
  if (item.kind !== "alert" || seen.includes(item.id) || seenHere.has(item.id)) return;
  seenHere.add(item.id);
  try {
    localStorage.setItem(SEEN_ALERTS_KEY, JSON.stringify([...seen, item.id].slice(-SEEN_LIMIT)));
  } catch {
    // Storage unavailable or full: seenHere still dedups this tab.
  }
  if (typeof Notification !== "undefined" && Notification.permission === "granted") new Notification(item.title, { body: item.source });
  let stack = document.querySelector<HTMLElement>('[aria-label="Alerts"]');
  if (!stack) {
    stack = Object.assign(document.createElement("div"), { className: "fixed right-4 top-4 z-50 flex w-72 flex-col gap-2 text-xs" });
    stack.setAttribute("role", "status");
    stack.setAttribute("aria-label", "Alerts");
    document.body.append(stack);
  }
  const toast = Object.assign(document.createElement("div"), { className: "strip rounded-xl px-3 py-2 text-zinc-200", textContent: `Alert: ${item.title}` });
  stack.append(toast);
  setTimeout(() => toast.remove(), TOAST_MS);
}

// One event gives at most one toast: an applied action or a terminal failure.
function toastFor(event: AgentEvent): { text: string; sym?: string; failed: boolean } | undefined {
  if (event.type === "error") return { text: "The request failed. Please retry.", failed: true };
  if (event.type === "failed") return { text: failureText(event.category), failed: true };
  const summary = actionSummary(event);
  return summary && { ...summary, failed: false };
}

// ponytail: one timer drops the oldest toast at a time; per-toast timers if stacks grow long.
export function Toasts({ events, onRetry }: { events: AgentEvent[]; onRetry?: () => void }) {
  // The stack portals to <body>: a backdrop-filter ancestor would otherwise pin "fixed" to the panel.
  const [host, setHost] = useState<HTMLElement | null>(null);
  const [gone, setGone] = useState<ReadonlySet<AgentEvent>>(new Set());
  useEffect(() => setHost(document.body), []);
  const shown = events.flatMap((event, i) => {
    const toast = gone.has(event) ? undefined : toastFor(event);
    return toast ? [{ event, i, ...toast }] : [];
  });
  const oldest = shown[0]?.event;
  const dismiss = (event: AgentEvent): void => setGone((previous) => new Set(previous).add(event));
  useEffect(() => {
    if (!oldest) return;
    const id = setTimeout(() => dismiss(oldest), TOAST_MS);
    return () => clearTimeout(id);
  }, [oldest]);
  if (!host || shown.length === 0) return null;
  return createPortal(
    <div role="status" aria-label="Notifications" className="fixed bottom-4 right-4 z-50 flex w-72 flex-col gap-2 text-xs">
      {shown.map((toast) => (
        <div key={toast.i} className={`strip flex items-center justify-between gap-2 rounded-xl px-3 py-2 ${toast.failed ? "text-red-300" : "text-zinc-200"}`}>
          <span>{toast.failed ? "✗ " : ""}{toast.text}</span>
          {toast.sym && <OpenWorkspace sym={toast.sym} />}
          {toast.failed && onRetry && (
            <button type="button" onClick={() => { dismiss(toast.event); onRetry(); }} className="rounded border border-zinc-700 px-2 py-1 text-zinc-300 hover:border-zinc-400">Try again</button>
          )}
        </div>
      ))}
    </div>,
    host,
  );
}
