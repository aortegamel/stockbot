import { useState } from "react";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import { MAX_WIDGETS } from "../lib/market-data";
import { routeHash, TICKER_RE } from "../lib/route";

const CARD = "strip flex items-center justify-between gap-2 rounded-xl px-3 py-2 text-xs text-zinc-200";
const ACTION = "rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-400";

// What the workspace did with each widget_update receipt. No entry: no workspace applied it yet.
export type WidgetOutcomes = ReadonlyMap<AgentEvent, "added" | "blocked">;

// An applied action the chat reports once: a card inline and a toast. `sym` links to its workspace.
// A widget receipt says "Added" only after the workspace confirmed it.
export function actionSummary(event: AgentEvent, widgetOutcomes?: WidgetOutcomes): { text: string; sym?: string; alert?: boolean } | undefined {
  if (event.type === "tool_result" && event.tool === "thesis_create" && event.evidenceId) return { text: "Thesis saved" };
  // An alert is set once a thesis_watch add returns its rule; the rendered result has an "added:" line, a list call has none.
  // ponytail: reads the 160-char preview (the rule_type ends by char ~155); a structured receipt if the preview shape changes.
  if (event.type === "tool_result" && event.tool === "thesis_watch" && event.evidenceId) {
    const added = /^added: .*\brule_type: ([a-z_]+)/m.exec(event.preview);
    if (added) return { text: `Alert set: ${added[1].replaceAll("_", " ")}`, alert: true };
    return undefined;
  }
  if (event.type !== "chart_update" && event.type !== "watchlist_update" && event.type !== "widget_update") return undefined;
  const ticker = event.ticker;
  if (!TICKER_RE.test(ticker)) return undefined;
  if (event.type === "chart_update") return { text: `Chart set to ${ticker}`, sym: ticker };
  if (event.type === "widget_update") {
    const outcome = widgetOutcomes?.get(event);
    if (outcome === "added") return { text: `Added the ${event.widgetType} widget to ${ticker}`, sym: ticker };
    if (outcome === "blocked") return { text: `Workspace limit reached: the ${event.widgetType} widget was not added to ${ticker} (${MAX_WIDGETS} widgets max)`, sym: ticker };
    return { text: `Widget requested: ${event.widgetType} for ${ticker}`, sym: ticker };
  }
  return { text: event.action === "remove" ? `Removed ${ticker} from the watchlist` : `Added ${ticker} to the watchlist`, sym: ticker };
}

// Asks once for system notifications; hidden when unsupported or already answered.
export function EnableNotifications() {
  const [permission, setPermission] = useState(() => (typeof Notification === "undefined" ? "denied" : Notification.permission));
  if (permission !== "default") return null;
  return <button type="button" onClick={() => void Notification.requestPermission().then(setPermission)} className={ACTION}>Enable notifications</button>;
}

export function OpenWorkspace({ sym }: { sym: string }) {
  return <a href={routeHash({ page: "workspace", sym })} className={ACTION}>Open workspace</a>;
}

// Widget receipts collapse into one card per workspace at the first receipt: "NVDA summary · 2 widgets".
export function ActionCards({ events }: { events: AgentEvent[] }) {
  const widgets = new Map<string, number>();
  for (const event of events) if (event.type === "widget_update" && TICKER_RE.test(event.ticker)) widgets.set(event.ticker, (widgets.get(event.ticker) ?? 0) + 1);
  const cards = events.flatMap((event) => {
    if (event.type !== "widget_update") return actionSummary(event) ?? [];
    const count = widgets.get(event.ticker);
    if (!count) return [];
    widgets.delete(event.ticker);
    return { text: `${event.ticker} summary · ${count} widget${count === 1 ? "" : "s"}`, sym: event.ticker, alert: false };
  });
  if (cards.length === 0) return null;
  return (
    <div className="space-y-2 pt-2" role="group" aria-label="Results">
      {cards.map((card, i) => (
        <div key={i} className={CARD}>
          <span>{card.text}</span>
          {card.sym && <OpenWorkspace sym={card.sym} />}
          {card.alert && <EnableNotifications />}
        </div>
      ))}
    </div>
  );
}

