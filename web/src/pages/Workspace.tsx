import { useEffect, useState, type ReactNode } from "react";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import { SessionPanel, JournalPanel } from "../components/market-panels";
import { WatchlistChatPanel } from "../components/watchlist-chat-panel";
import { ThesisPanel } from "../components/thesis-panel";
import { PortfolioPanel } from "../components/portfolio-panel";
import { EnableNotifications } from "../components/action-cards";
import { isWidgetType, WidgetGrid, widgetLabel, WORKSPACE_TYPES } from "../components/widgets";
import HomePanels, { HOME_TYPES } from "../components/home-workspace";
import { TvLinkForm } from "../components/tv-link";
import { listWorkspaces, MAX_WIDGETS, saveWorkspace, type WorkspaceDoc, type WorkspaceWidget } from "../lib/market-data";
import { TICKER_RE } from "../lib/route";
import type { ShellState } from "../layout/Shell";

const ACTIVE_KEY = "stockbot.workspace";
export const DEFAULT_WORKSPACE_ID = "00000000-0000-4000-8000-000000000000";
// Shell retains drafts across page changes. Only Save writes them to the server.
export type WorkspaceDraft = {
  docs: WorkspaceDoc[];
  activeId: string;
  dirty: string[];
  status: Record<string, string>;
  saving: boolean;
};
const applied = new WeakSet<AgentEvent>();

// getRandomValues also works on HTTP LAN origins, where randomUUID is unavailable.
function newId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function windowOf(type: string, ticker: string, index: number): WorkspaceWidget {
  return {
    id: newId(), type, ticker,
    geometry: { x: 32 + (index % 8) * 32, y: 32 + (index % 8) * 32, width: 560, height: 360 },
  };
}

// The default tiles the browser window once, at creation. Saved layouts keep their saved pixels.
// 58 px: workspace padding, viewport border, and the 40 px desktop margin in WidgetGrid. 258 px: shell and toolbar chrome plus that margin.
function defaultWorkspace(ticker: string): WorkspaceDoc {
  const g = 8;
  // Proportions come from the user's saved 2502x1190 layout. 1240 px keeps every window at the server's 280 px minimum width.
  const w = Math.max(1240, window.innerWidth - 58);
  const h = Math.max(640, window.innerHeight - 258);
  const right = Math.round(w * 0.3);
  const rightX = w - right;
  const left = Math.max(280, Math.round(w * 0.14));
  const chartX = left + g;
  const top = Math.round(h * 0.55);
  const bottomY = top + g;
  const bottom = h - bottomY;
  const portfolio = Math.max(280, Math.round(w * 0.2));
  const marketsX = portfolio + g;
  const detail = Math.max(280, Math.round(w * 0.2));
  const detailX = rightX - g - detail;
  const tape = 180;
  const news = Math.round((h - tape - g) * 0.4);
  const agentY = tape + g + news + g;
  const initial = [
    ["watchlist", 0, 0, left, top],
    ["price-volume", chartX, 0, rightX - g - chartX, top],
    ["portfolio", 0, bottomY, portfolio, bottom],
    ["markets", marketsX, bottomY, detailX - g - marketsX, bottom],
    ["market-detail", detailX, bottomY, detail, bottom],
    ["index-tape", rightX, 0, right, tape],
    ["news", rightX, tape + g, right, news],
    ["agent", rightX, agentY, right, h - agentY],
  ] as const;
  return {
    id: DEFAULT_WORKSPACE_ID, name: "default", updated_at: null,
    widgets: initial.map(([type, x, y, width, height]) => ({
      id: newId(), type, ticker, geometry: { x, y, width, height },
    })),
  };
}


function edit(draft: WorkspaceDraft, doc: WorkspaceDoc): WorkspaceDraft {
  return {
    ...draft,
    docs: draft.docs.map((d) => d.id === doc.id ? doc : d),
    dirty: draft.dirty.includes(doc.id) ? draft.dirty : [...draft.dirty, doc.id],
    status: { ...draft.status, [doc.id]: "Unsaved changes" },
  };
}

type Receipt = { type: string; ticker: string; event: AgentEvent };
function receiptOf(event: AgentEvent): Receipt | null {
  if (event.type !== "widget_update" && event.type !== "chart_update") return null;
  const type = event.type === "chart_update" ? "price-volume" : event.widgetType;
  const symbol = typeof event.ticker === "string" ? event.ticker.trim().toUpperCase() : "";
  if (!TICKER_RE.test(symbol) || !isWidgetType(type) || type === "tv-link") return null;
  return { type, ticker: symbol, event };
}
function applyReceipts(widgets: WorkspaceWidget[], events: AgentEvent[]): { widgets: WorkspaceWidget[]; outcomes: Map<AgentEvent, "added" | "blocked">; changed: boolean } {
  const next = [...widgets];
  const outcomes = new Map<AgentEvent, "added" | "blocked">();
  for (const event of events) {
    if (applied.has(event)) continue;
    applied.add(event);
    const receipt = receiptOf(event);
    if (!receipt) continue;
    if (next.some((w) => w.type === receipt.type && w.ticker === receipt.ticker)) continue;
    if (next.length >= MAX_WIDGETS) {
      if (receipt.event.type === "widget_update") outcomes.set(receipt.event, "blocked");
      continue;
    }
    next.push(windowOf(receipt.type, receipt.ticker, next.length));
    if (receipt.event.type === "widget_update") outcomes.set(receipt.event, "added");
  }
  return { widgets: next, outcomes, changed: next.length !== widgets.length };
}
function initialDraft(workspaces: WorkspaceDoc[], selected: string): WorkspaceDraft {
  const docs = workspaces.some((doc) => doc.id === DEFAULT_WORKSPACE_ID)
    ? workspaces : [defaultWorkspace(selected), ...workspaces];
  let activeId = DEFAULT_WORKSPACE_ID;
  try {
    const saved = window.localStorage.getItem(ACTIVE_KEY);
    if (docs.some((d) => d.id === saved)) activeId = saved!;
  } catch {
    // Disabled storage does not prevent server-backed workspaces.
  }
  return { docs, activeId, dirty: [], status: {}, saving: false };
}
function extraPanel(s: ShellState, widget: WorkspaceWidget) {
  switch (widget.type) {
    case "watchlist": return <WatchlistChatPanel selected={s.selected} onSelect={s.pick} events={s.events} onChange={s.onWatchlist} symbols={s.watchlist} />;
    case "portfolio": return <PortfolioPanel onSelectTicker={s.pick} />;
    case "agent": return s.chat;
    case "theses": return <ThesisPanel onSelectTicker={s.pick} />;
    case "alerts": return <AlertsPanel sym={widget.ticker} />;
    case "sessions": return <SessionPanel sessions={s.sessions} sessionId={s.sessionId} onSelect={s.setSessionId} snapshot={s.snapshot} />;
    case "journal": return <JournalPanel journal={s.journal} seq={s.seq} />;
    default: return null;
  }
}
function WorkspaceTabs({ draft, activeId, onSelect, onNew }: { draft: WorkspaceDraft; activeId: string; onSelect: (id: string) => void; onNew: () => void }) {
  return (
    <div className="strip ws-tabs">
      <div role="tablist" aria-label="Workspaces" className="ws-tablist">
        {draft.docs.map((doc) => <button key={doc.id} type="button" role="tab" aria-selected={doc.id === activeId} className="pill" onClick={() => onSelect(doc.id)}>
          {doc.name || "Untitled"}{draft.dirty.includes(doc.id) ? " *" : ""}
        </button>)}
      </div>
      <button type="button" className="pill" aria-label="New workspace" title="New workspace" onClick={onNew}>+</button>
    </div>
  );
}
function CreatorForm({ name, setName, fromDefault, setFromDefault, onCreate, onCancel }: {
  name: string; setName: (v: string) => void; fromDefault: boolean; setFromDefault: (v: boolean) => void; onCreate: () => void; onCancel: () => void;
}) {
  return (
    <form className="strip ws-actions ws-toolbar" onSubmit={(e) => { e.preventDefault(); onCreate(); }}>
      <input autoFocus className="input" aria-label="Workspace name" placeholder="Workspace name" value={name} maxLength={60} required onChange={(e) => setName(e.target.value)} />
      <label className="pill"><input type="checkbox" checked={fromDefault} onChange={(e) => setFromDefault(e.target.checked)} /> Start from default</label>
      <button type="submit" className="btn-primary" disabled={!name.trim()}>Create workspace</button>
      <button type="button" className="pill" onClick={onCancel}>Cancel</button>
    </form>
  );
}
function EditorBar({ current, status, dirty, saving, ticker, setTicker, onName, onAddSelect, onSave }: {
  current: WorkspaceDoc; status: string; dirty: boolean; saving: boolean; ticker: string; setTicker: (v: string) => void;
  onName: (v: string) => void; onAddSelect: (v: string) => void; onSave: () => void;
}) {
  return (
    <div className="strip ws-actions ws-toolbar">
      <input className="input ws-name" aria-label="Workspace name" value={current.name} readOnly={current.id === DEFAULT_WORKSPACE_ID} maxLength={60} required onChange={(e) => onName(e.target.value)} />
      <input className="input ws-ticker" aria-label="New window ticker" value={ticker} maxLength={10} pattern={TICKER_RE.source} title="Ticker for new windows" onChange={(e) => setTicker(e.target.value.trim().toUpperCase())} />
      <select aria-label="Add widget" value="" className="pill" disabled={current.widgets.length >= MAX_WIDGETS || !TICKER_RE.test(ticker)} onChange={(e) => onAddSelect(e.target.value)}>
        <option value="">Add widget</option>
        {WORKSPACE_TYPES.map((type) => <option key={type} value={type}>{widgetLabel(type)}</option>)}
      </select>
      <button type="button" className="btn-primary" disabled={saving || !current.name.trim()} onClick={onSave}>{saving ? "Saving…" : "Save"}</button>
      <span role="status" className="feed-muted">{status || (dirty ? "Unsaved changes" : current.updated_at ? "Saved" : "Default layout")}</span>
      {current.widgets.length >= MAX_WIDGETS && <span className="feed-muted">Workspace limit: {MAX_WIDGETS} widgets</span>}
    </div>
  );
}
type Trigger = { trigger_id: string; trigger_type: string; status: string; summary?: string; created_at?: string };

// The API has no ticker endpoint for alerts. Compose the thesis list and details.
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
      .catch(() => !ctrl.signal.aborted && setState({ rows: [], error: "Alerts unavailable" }));
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
  const draft = s.workspace;
  const current = draft?.docs.find((d) => d.id === draft.activeId);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [fromDefault, setFromDefault] = useState(false);
  const [ticker, setTicker] = useState(s.selected);
  const [linking, setLinking] = useState(false);
  const saving = draft?.saving ?? false;

  useEffect(() => setTicker(s.selected), [s.selected]);
  useEffect(() => {
    if (draft) return;
    const ctrl = new AbortController();
    setError("");
    listWorkspaces(ctrl.signal)
      .then(({ workspaces }) => !ctrl.signal.aborted && s.setWorkspace(initialDraft(workspaces, s.selected)))
      .catch(() => !ctrl.signal.aborted && setError("Workspaces unavailable. Your saved workspaces did not load."));
    return () => ctrl.abort();
  }, [draft, attempt, s.selected, s.setWorkspace]);

  useEffect(() => {
    if (!draft) return;
    try {
      window.localStorage.setItem(ACTIVE_KEY, draft.activeId);
    } catch {
      // Disabled storage only prevents remembering the selected workspace.
    }
  }, [draft?.activeId]);

  function mergeSave(draft: WorkspaceDraft, snapshot: WorkspaceDoc, saved: WorkspaceDoc): WorkspaceDraft {
    const unchanged = draft.docs.some((doc) => doc === snapshot);
    return {
      ...draft,
      docs: draft.docs.map((doc) => doc === snapshot ? saved : doc),
      dirty: unchanged ? draft.dirty.filter((id) => id !== snapshot.id) : draft.dirty,
      status: { ...draft.status, [snapshot.id]: unchanged ? `Saved ${(saved.updated_at ?? "").slice(11, 19)}` : "Saved earlier version. Unsaved changes" },
    };
  }
  // Receipts target the selected named workspace. They retain their own ticker.
  useEffect(() => {
    if (!draft || !current) return;
    const applied = applyReceipts(current.widgets, s.events);
    if (applied.changed) s.setWorkspace((d) => d ? edit(d, { ...current, widgets: applied.widgets }) : d);
    if (applied.outcomes.size) s.onWidgetOutcomes?.(applied.outcomes);
  }, [draft, current, s.events, s.setWorkspace, s.onWidgetOutcomes]);

  function update(change: (doc: WorkspaceDoc) => WorkspaceDoc) {
    if (!current) return;
    s.setWorkspace((d) => {
      const doc = d?.docs.find((doc) => doc.id === current.id);
      if (!d || !doc) return d;
      const next = change(doc);
      return next === doc ? d : edit(d, next);
    });
  }

  async function save() {
    if (!current || saving || !current.name.trim()) return;
    const snapshot = current;
    s.setWorkspace((d) => d ? { ...d, saving: true } : d);
    try {
      const saved = await saveWorkspace(snapshot.id, snapshot.name.trim(), snapshot.widgets);
      s.setWorkspace((d) => (d && d.docs.some((doc) => doc === snapshot)) ? mergeSave(d, snapshot, saved) : d);
    } catch {
      s.setWorkspace((d) => d ? { ...d, status: { ...d.status, [snapshot.id]: "Save failed. Your changes remain unsaved." } } : d);
    } finally {
      s.setWorkspace((d) => d ? { ...d, saving: false } : d);
    }
  }

  function add(type: string, params?: Record<string, unknown>) {
    if (!TICKER_RE.test(ticker)) return;
    update((doc) => doc.widgets.length >= MAX_WIDGETS ? doc : {
      ...doc, widgets: [...doc.widgets, { ...windowOf(type, ticker, doc.widgets.length), ...(params ? { params } : {}) }],
    });
  }

  if (!draft || !current) return <div className="workspace">
    {error ? <p role="alert">{error} <button type="button" className="pill" onClick={() => setAttempt((n) => n + 1)}>Retry</button></p>
      : <span className="skeleton" aria-label="Loading workspace" aria-busy="true" />}
  </div>;

  const dirty = draft.dirty.includes(current.id);
  const select = (id: string) => {
    setLinking(false);
    setCreating(false);
    s.setWorkspace((d) => d ? { ...d, activeId: id } : d);
  };
  const create = () => {
    if (!name.trim()) return;
    const template = fromDefault ? draft.docs.find((doc) => doc.id === DEFAULT_WORKSPACE_ID)?.widgets ?? [] : [];
    const doc: WorkspaceDoc = {
      id: newId(), name: name.trim(), updated_at: null,
      widgets: template.map((widget) => ({ ...structuredClone(widget), id: newId() })),
    };
    s.setWorkspace((d) => d ? { ...d, docs: [...d.docs, doc], activeId: doc.id, dirty: [...d.dirty, doc.id] } : d);
    setCreating(false);
    setLinking(false);
  };
  const choose = (value: string) => {
    if (value === "tv-link") setLinking(true);
    else if (WORKSPACE_TYPES.includes(value)) add(value);
  };
  const windows = (renderHome: (type: string, ticker?: string) => ReactNode) => (
    <WidgetGrid key={current.id} widgets={current.widgets} renderExtra={(widget) => renderHome(widget.type, widget.ticker) ?? extraPanel(s, widget)}
      onChange={(widgets) => update((doc) => ({ ...doc, widgets }))}
      onRemove={(id) => update((doc) => ({ ...doc, widgets: doc.widgets.filter((w) => w.id !== id) }))}
      onParams={(id, params) => update((doc) => ({ ...doc, widgets: doc.widgets.map((w) => w.id === id ? { ...w, params } : w) }))}
      onType={(id, type) => update((doc) => ({ ...doc, widgets: doc.widgets.map((w) => w.id === id ? { id: w.id, ticker: w.ticker, geometry: w.geometry, type } : w) }))}
      onTicker={(id, next) => update((doc) => ({ ...doc, widgets: doc.widgets.map((w) => w.id === id ? { ...w, ticker: next } : w) }))}
    />
  );
  return (
    <div className="workspace">
      <WorkspaceTabs draft={draft} activeId={current.id} onSelect={select} onNew={() => { setName(""); setFromDefault(false); setCreating(true); }} />
      {creating
        ? <CreatorForm name={name} setName={setName} fromDefault={fromDefault} setFromDefault={setFromDefault} onCreate={create} onCancel={() => setCreating(false)} />
        : <EditorBar current={current} status={draft.status[current.id]} dirty={dirty} saving={saving} ticker={ticker} setTicker={setTicker}
          onName={(next) => update((doc) => ({ ...doc, name: next }))} onAddSelect={choose} onSave={() => void save()} />}
      <p className="ws-hint">Drag a title to move a window. Drag an edge to resize it. Use the panel menu to replace its contents.</p>
      {linking && <div className="glass ws-link-form"><TvLinkForm onSubmit={(params) => { add("tv-link", params); setLinking(false); }} onCancel={() => setLinking(false)} /></div>}
      {current.widgets.some((widget) => HOME_TYPES.includes(widget.type))
        ? <HomePanels key={current.id} {...s}>{windows}</HomePanels>
        : windows(() => null)}
    </div>
  );
}
