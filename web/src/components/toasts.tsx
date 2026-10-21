import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import type { FeedItem } from "../lib/market-data";
import { actionSummary, OpenWorkspace, type WidgetOutcomes } from "./action-cards";
import { failureText } from "./agent-event";

const TOAST_MS = 5000;
export const SEEN_ALERTS_KEY = "stockbot.seen_alerts";
const SEEN_LIMIT = 500;
// This tab's ids: dedups even when storage is unavailable or full.
const seenHere = new Set<string>();
// Opaque so drawer text never shows through. z-[1] ties .shell-body and paints after it, but stays under the header (z 2) and its palette.
const TOAST = "rounded-xl border border-white/12 bg-[#141A30] px-3 py-2 shadow-lg";
const STACK = "flex flex-col gap-2";

// One bottom-right corner, above the 30px footer, holds both the alert stack and the agent toast stack.
function toastHost(): HTMLElement {
  let host = document.getElementById("toasts");
  if (!host) {
    host = Object.assign(document.createElement("div"), { id: "toasts", className: `fixed bottom-10 right-4 z-[1] w-72 text-xs ${STACK}` });
    document.body.append(host);
  }
  return host;
}

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
  let stack = document.querySelector<HTMLElement>('#toasts [aria-label="Alerts"]');
  if (!stack) {
    stack = Object.assign(document.createElement("div"), { className: STACK });
    stack.setAttribute("role", "status");
    stack.setAttribute("aria-label", "Alerts");
    toastHost().append(stack);
  }
  const toast = Object.assign(document.createElement("div"), { className: `${TOAST} text-zinc-200`, textContent: `Alert: ${item.title}` });
  stack.append(toast);
  setTimeout(() => toast.remove(), TOAST_MS);
}

// One event gives at most one toast: an applied action or a terminal failure.
function toastFor(event: AgentEvent, widgetOutcomes?: WidgetOutcomes): { text: string; sym?: string; failed: boolean } | undefined {
  if (event.type === "error") return { text: "The request failed. Please retry.", failed: true };
  if (event.type === "failed") return { text: failureText(event.category), failed: true };
  const summary = actionSummary(event, widgetOutcomes);
  return summary && { ...summary, failed: false };
}

// ponytail: one timer drops the oldest toast at a time; per-toast timers if stacks grow long.
export function Toasts({ events, onRetry, widgetOutcomes }: { events: AgentEvent[]; onRetry?: () => void; widgetOutcomes?: WidgetOutcomes }) {
  // The stack portals to the shared corner host: a backdrop-filter ancestor would otherwise pin "fixed" to the panel.
  const [host, setHost] = useState<HTMLElement | null>(null);
  const [gone, setGone] = useState<ReadonlySet<AgentEvent>>(new Set());
  useEffect(() => setHost(toastHost()), []);
  const shown = events.flatMap((event, i) => {
    const toast = gone.has(event) ? undefined : toastFor(event, widgetOutcomes);
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
    <div role="status" aria-label="Notifications" className={STACK}>
      {shown.map((toast) => (
        <div key={toast.i} className={`${TOAST} flex items-center justify-between gap-2 ${toast.failed ? "text-red-300" : "text-zinc-200"}`}>
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
