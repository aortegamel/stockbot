import { useCallback, useEffect, useState } from "react";
import { WIDGET_TYPES, type AgentEvent } from "../../../needle-harness/lib/agent/types";
import { SessionPanel, JournalPanel } from "../components/market-panels";
import { WatchlistChatPanel } from "../components/watchlist-chat-panel";
import { ThesisPanel } from "../components/thesis-panel";
import { PortfolioPanel } from "../components/portfolio-panel";
import { EnableNotifications } from "../components/action-cards";
import { isWidgetType, WidgetGrid, widgetLabel } from "../components/widgets";
import { fetchQuote, fetchWorkspace, saveWorkspace, type WorkspaceWidget } from "../lib/market-data";
import { routeHash, TICKER_RE } from "../lib/route";
import type { ShellState } from "../layout/Shell";

const TABS_KEY = "stockbot.tabs";
const DEFAULT_WIDGETS = ["price-volume", "financials", "news"];
const BOTTOM_TABS = ["Theses", "Alerts", "Sessions", "Journal"] as const;
// Agent widget receipts apply once per page load, even when the page remounts.
const applied = new WeakSet<AgentEvent>();

// One widget per type: the type doubles as the widget id.
const widgetOf = (type: string, sym: string): WorkspaceWidget => ({ id: type, type, ticker: sym });
const defaults = (sym: string) => DEFAULT_WIDGETS.map((type) => widgetOf(type, sym));

function savedTabs(): string[] {
  try {
    const raw: unknown = JSON.parse(window.localStorage.getItem(TABS_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((sym): sym is string => typeof sym === "string" && TICKER_RE.test(sym)) : [];
  } catch {
    return [];
  }
}

type Trigger = { trigger_id: string; trigger_type: string; status: string; summary?: string; created_at?: string };

// Pending triggers of the theses scoped to this ticker. No per-ticker endpoint exists, so the list and details compose here.
function AlertsPanel({ sym }: { sym: string }) {
  const [state, setState] = useState<{ rows: (Trigger & { thesis: string })[]; error: string | null } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    setState(null);
    const ctrl = new AbortController();
    const get = async <T,>(path: string): Promise<T> => {
      const res = await fetch(path, { signal: ctrl.signal, headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`request failed: ${res.status}`);
      return (await res.json()) as T;
    };
    (async () => {
      const list = await get<unknown>("/api/theses");
      const mine = (Array.isArray(list) ? list : []).filter(
        (t): t is { thesis_id: string; slug?: string; scope?: string } => typeof t?.thesis_id === "string" && String(t.scope ?? "").toUpperCase() === sym,
      );
      const details = await Promise.all(mine.map((t) => get<{ triggers?: Trigger[] }>(`/api/theses/${encodeURIComponent(t.thesis_id)}`)));
      return details.flatMap((d, i) => (Array.isArray(d.triggers) ? d.triggers : [])
        .filter((t) => t.status === "pending")
        .map((t) => ({ ...t, thesis: mine[i].slug ?? mine[i].thesis_id })));
    })()
      .then((rows) => !ctrl.signal.aborted && setState({ rows, error: null }))
      .catch(() => !ctrl.signal.aborted && setState({ rows: [], error: "alerts unavailable" }));
    return () => ctrl.abort();
  }, [sym, attempt]);
  return (
    <div className="ws-alerts">
      <EnableNotifications />
      {state === null && <span className="skeleton" aria-label="Loading alerts" aria-busy="true" />}
      {state?.error && (
        <p role="alert" className="feed-muted">
          Alerts unavailable. <button type="button" className="pill" onClick={() => setAttempt((n) => n + 1)}>Retry</button>
        </p>
      )}
      {state && !state.error && state.rows.length === 0 && <p className="feed-muted">No pending alerts for {sym}. Try asking about its latest risks.</p>}
      {state?.rows.map((t) => (
        <div key={t.trigger_id}>
          <span className="feed-muted"><span className="num">{(t.created_at ?? "").slice(0, 16)}</span> · {t.thesis}</span> {t.summary || t.trigger_type}
        </div>
      ))}
    </div>
  );
}

export default function Workspace(s: ShellState) {
  const sym = s.selected;
  const [tabs, setTabs] = useState(() => [...new Set([...savedTabs(), sym])]);
  // The open ticker always has a tab; a pick or a hash adds it.
  const shown = tabs.includes(sym) ? tabs : [...tabs, sym];
  const tabList = shown.join(",");
  const [adding, setAdding] = useState(false);
  // status: the last edit or save outcome for this ticker's workspace.
  const [doc, setDoc] = useState<{ sym: string; widgets: WorkspaceWidget[]; status: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [bottom, setBottom] = useState<(typeof BOTTOM_TABS)[number]>("Theses");
  const loaded = doc?.sym === sym ? doc : null;
  const widgets = loaded?.widgets ?? defaults(sym);

  useEffect(() => {
    setTabs((previous) => previous.includes(sym) ? previous : [...previous, sym]);
  }, [sym]);

  useEffect(() => {
    try {
      window.localStorage.setItem(TABS_KEY, JSON.stringify(tabList.split(",")));
    } catch {
      // Storage can be disabled; the tabs then last for this page load only.
    }
  }, [tabList]);

  // Last delayed price per open tab; a failed quote shows a dash.
  const [prices, setPrices] = useState<Record<string, number | null>>({});
  useEffect(() => {
    const ctrl = new AbortController();
    for (const tab of tabList.split(",")) {
      fetchQuote(tab, ctrl.signal).then(
        (q) => setPrices((p) => ({ ...p, [tab]: typeof q.price === "number" ? q.price : null })),
        () => undefined,
      );
    }
    return () => ctrl.abort();
  }, [tabList]);

  // A never-saved (or unreachable) workspace shows the default chart, financials, and news.
  useEffect(() => {
    const ctrl = new AbortController();
    fetchWorkspace(sym, ctrl.signal)
      .then((d) => !ctrl.signal.aborted && setDoc({ sym, widgets: d.updated_at ? d.widgets : defaults(sym), status: "" }))
      .catch(() => !ctrl.signal.aborted && setDoc({ sym, widgets: defaults(sym), status: "workspace unavailable" }));
    return () => ctrl.abort();
  }, [sym, attempt]);

  async function save(next: WorkspaceWidget[]) {
    const target = sym;
    let status = "save failed";
    try {
      const d = await saveWorkspace(target, next);
      status = `saved ${(d.updated_at ?? "").slice(11, 19)}`;
    } catch {
      // The status line reports the failure; the widgets stay on screen, unsaved.
    }
    setDoc((d) => (d?.sym === target ? { ...d, status } : d));
  }

  function edit(next: WorkspaceWidget[]) {
    setDoc({ sym, widgets: next, status: "unsaved" });
  }

  const removeWidget = useCallback((id: string) => {
    setDoc((d) => d?.sym === sym ? { ...d, widgets: d.widgets.filter((w) => w.id !== id), status: "unsaved" } : d);
  }, [sym]);

  // A validated agent receipt adds its widget to this workspace and saves it.
  useEffect(() => {
    if (!loaded) return;
    const added: string[] = [];
    for (const event of s.events) {
      if (event.type !== "widget_update" || event.ticker !== sym || applied.has(event)) continue;
      applied.add(event);
      if (isWidgetType(event.widgetType) && !loaded.widgets.some((w) => w.type === event.widgetType) && !added.includes(event.widgetType)) added.push(event.widgetType);
    }
    if (added.length === 0) return;
    const next = [...loaded.widgets, ...added.map((type) => widgetOf(type, sym))];
    setDoc({ sym, widgets: next, status: "" });
    void save(next);
  }, [s.events, loaded, sym]);

  function close(tab: string) {
    const rest = shown.filter((t) => t !== tab);
    if (rest.length === 0) return;
    setTabs(rest);
    if (tab === sym) s.pick(rest[0]);
  }

  const missing = WIDGET_TYPES.filter((type) => !widgets.some((w) => w.type === type));
  const status = loaded?.status ?? "";

  return (
    <div className="workspace">
      <div className="ws-left">
        <div className="glass">
          <WatchlistChatPanel selected={s.selected} onSelect={s.pick} events={s.events} onChange={s.onWatchlist} symbols={s.watchlist} />
        </div>
        <div className="glass">
          <PortfolioPanel onSelectTicker={s.pick} />
        </div>
      </div>
      <div className="ws-mid">
        <div className="strip ws-tabs">
          <span role="tablist" aria-label="Open workspaces" className="ws-tablist">
            {shown.map((tab) => (
              <span key={tab} className="ws-tab">
                <a role="tab" aria-selected={tab === sym} href={routeHash({ page: "workspace", sym: tab })} className="pill">
                  <span style={{ fontWeight: 600 }}>{tab}</span>{" "}
                  <span className="ws-tab-price">{prices[tab] == null ? "—" : prices[tab]!.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </a>
                {shown.length > 1 && <button type="button" aria-label={`Close ${tab}`} onClick={() => close(tab)} className="ws-tool">×</button>}
              </span>
            ))}
            {adding ? (
              <input
                autoFocus
                aria-label="Open ticker"
                placeholder="Ticker"
                className="ws-tab-input"
                onBlur={() => setAdding(false)}
                onKeyDown={(e) => {
                  const value = e.currentTarget.value.trim().toUpperCase();
                  if (e.key === "Escape") setAdding(false);
                  if (e.key === "Enter" && TICKER_RE.test(value)) {
                    setAdding(false);
                    s.pick(value);
                  }
                }}
              />
            ) : (
              <button type="button" aria-label="Open another ticker" onClick={() => setAdding(true)} className="ws-tool">+</button>
            )}
          </span>
          <span className="ws-actions">
            <select
              aria-label="Add widget"
              value=""
              disabled={!loaded}
              onChange={(e) => isWidgetType(e.target.value) && edit([...widgets, widgetOf(e.target.value, sym)])}
              className="pill"
            >
              <option value="">Add widget</option>
              {missing.map((type) => <option key={type} value={type}>{widgetLabel(type)}</option>)}
            </select>
            <button type="button" disabled={!loaded} onClick={() => void save(widgets)} className="pill">Save</button>
            {status && <span className="feed-muted num">{status}</span>}
            {status === "workspace unavailable" && <button type="button" className="pill" onClick={() => setAttempt((n) => n + 1)}>Retry</button>}
            {status === "save failed" && <button type="button" className="pill" onClick={() => void save(widgets)}>Retry</button>}
          </span>
        </div>
        {!loaded ? <span className="skeleton" aria-label="Loading workspace" aria-busy="true" /> : <WidgetGrid widgets={widgets} onRemove={removeWidget} />}
        <div className="glass ws-bottom">
          <div role="tablist" aria-label={`${sym} panels`} className="ws-tablist">
            {BOTTOM_TABS.map((tab) => (
              <button key={tab} type="button" role="tab" aria-selected={bottom === tab} onClick={() => setBottom(tab)} className="pill">{tab}</button>
            ))}
          </div>
          {/* Hidden panels stay mounted: their polls and selections survive a tab switch. */}
          <div role="tabpanel" aria-label="Theses" hidden={bottom !== "Theses"}>
            <ThesisPanel onSelectTicker={s.pick} />
          </div>
          <div role="tabpanel" aria-label="Alerts" hidden={bottom !== "Alerts"}>
            {bottom === "Alerts" && <AlertsPanel key={sym} sym={sym} />}
          </div>
          <div role="tabpanel" aria-label="Sessions" hidden={bottom !== "Sessions"}>
            <SessionPanel sessions={s.sessions} sessionId={s.sessionId} onSelect={s.setSessionId} snapshot={s.snapshot} />
          </div>
          <div role="tabpanel" aria-label="Journal" hidden={bottom !== "Journal"}>
            <JournalPanel journal={s.journal} seq={s.seq} />
          </div>
        </div>
      </div>
    </div>
  );
}
