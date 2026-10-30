import { memo, useCallback, useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { Rnd } from "react-rnd";
import { WIDGET_TYPES } from "../../../needle-harness/lib/agent/types";
import { fetchCandles, fetchFilings, fetchWidgetData, type Candle, type FilingItem, type WidgetData, type WorkspaceWidget } from "../lib/market-data";
import { TICKER_RE } from "../lib/route";
import { FinancialsPanel, NewsPanel } from "./market-panels";
import TradingChart, { ChartControls } from "./trading-chart";
import { useLoad } from "./home-panels";
import { TvLinkForm, TvLinkWidget } from "./tv-link";

export const WORKSPACE_TYPES = [...WIDGET_TYPES, "watchlist", "portfolio", "agent", "theses", "alerts", "sessions", "journal"];
const LABELS: Record<string, string> = {
  "price-volume": "Price & volume",
  financials: "Financials",
  news: "News",
  "monthly-returns": "Monthly returns",
  "short-interest": "Short interest",
  "short-volume": "Short volume",
  insider: "Insider timeline",
  filings: "Filings timeline",
  multiples: "Multiples",
  "tv-link": "TradingView link",
  watchlist: "Watchlist",
  portfolio: "Portfolio",
  agent: "Agent",
  theses: "Theses",
  alerts: "Alerts",
  sessions: "Sessions",
  journal: "Journal",
};

export const isWidgetType = (type: unknown): type is string => typeof type === "string" && WIDGET_TYPES.includes(type);
export const widgetLabel = (type: string): string => LABELS[type] ?? type;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const muted = { color: "var(--muted, #93A0B8)" } as const;
const cell = { padding: "0 8px", height: 28, textAlign: "right" } as const;
const head = { ...cell, ...muted, fontSize: "var(--fs-label)", fontWeight: 500 } as const;

// Month-over-month close returns keyed by UTC year; a second bar in one month (the live bar) replaces the first.
export function monthlyReturns(candles: Candle[]): Map<number, (number | null)[]> {
  const months: { year: number; month: number; close: number }[] = [];
  for (const bar of candles) {
    const at = new Date(bar.t * 1000);
    const row = { year: at.getUTCFullYear(), month: at.getUTCMonth(), close: bar.close };
    const last = months.at(-1);
    if (last && last.year === row.year && last.month === row.month) months[months.length - 1] = row;
    else months.push(row);
  }
  const out = new Map<number, (number | null)[]>();
  for (let i = 1; i < months.length; i++) {
    const prior = months[i - 1].close;
    if (!prior) continue;
    const { year, month, close } = months[i];
    const row = out.get(year) ?? Array<number | null>(12).fill(null);
    row[month] = close / prior - 1;
    out.set(year, row);
  }
  return out;
}

function Status({ error, empty, retry }: { error: string | null | undefined; empty: string | null; retry: () => void }) {
  if (error) return <p role="alert" style={muted}>Data unavailable. <button type="button" className="pill" onClick={retry}>Retry</button></p>;
  return empty ? <p style={muted}>{empty}</p> : null;
}

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const num = (value: unknown) => (typeof value === "number" ? value.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "—");

function MonthlyReturns({ sym }: { sym: string }) {
  const { data, error, retry } = useLoad(sym, (signal) => fetchCandles(sym, "5y", "1mo", signal));
  if (!data && !error) return <span className="skeleton" aria-label="Loading monthly returns" aria-busy="true" />;
  const years = [...monthlyReturns(data?.candles ?? []).entries()].sort((a, b) => b[0] - a[0]);
  return (
    <>
      <Status error={error ?? data?.error} empty={years.length === 0 ? `No monthly returns for ${sym}. Try asking about its price history.` : null} retry={retry} />
      {years.length > 0 && (
        <table className="num" aria-label={`${sym} monthly returns`} style={{ fontSize: "var(--fs-body)", borderCollapse: "collapse" }}>
          <thead><tr><th />{MONTHS.map((m) => <th key={m} style={head}>{m}</th>)}</tr></thead>
          <tbody>
            {years.map(([year, row]) => (
              <tr key={year}>
                <th style={head}>{year}</th>
                {row.map((r, i) => (
                  <td key={i} style={{ ...cell, background: r === null ? undefined : `rgba(${r < 0 ? "239,83,80" : "38,166,154"},${Math.min(Math.abs(r) * 4, 0.8)})` }}>
                    {r === null ? "" : pct(r)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

// Server-backed rows as a plain table; `columns` maps row keys to headers.
function SeriesTable({ type, sym, columns }: { type: string; sym: string; columns: [string, string][] }) {
  const { data, error, retry } = useLoad(`${type}:${sym}`, (signal) => fetchWidgetData(type, sym, signal));
  if (!data && !error) return <span className="skeleton" aria-label={`Loading ${widgetLabel(type)}`} aria-busy="true" />;
  const rows = data?.rows ?? [];
  return (
    <>
      <Status error={error ?? data?.error} empty={rows.length === 0 ? `No ${widgetLabel(type).toLowerCase()} data for ${sym}. Try asking about its latest activity.` : null} retry={retry} />
      {rows.length > 0 && (
        <table className="num" style={{ fontSize: "var(--fs-body)", borderCollapse: "collapse", width: "100%" }}>
          <thead><tr>{columns.map(([key, label]) => <th key={key} style={head}>{label}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map(([key]) => (
                  <td key={key} style={cell}>{key === "short_pct" && typeof row[key] === "number" ? `${num(row[key])}%` : typeof row[key] === "number" ? num(row[key]) : (row[key] ?? "—")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function FilingsTimeline({ sym }: { sym: string }) {
  const { data, error, retry } = useLoad(sym, (signal) => fetchFilings(sym, signal));
  if (!data && !error) return <span className="skeleton" aria-label="Loading filings" aria-busy="true" />;
  const items: FilingItem[] = data?.items ?? [];
  return (
    <>
      <Status error={error ?? data?.error} empty={items.length === 0 ? `No filings for ${sym}. Try asking about its latest disclosures.` : null} retry={retry} />
      {items.map((f, i) => (
        <div key={i} style={{ fontSize: "var(--fs-body)", marginBottom: 8 }}>
          <span className="num" style={muted}>{f.filed}</span>{" "}
          {f.link ? <a href={f.link} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>{f.title}</a> : f.title}
        </div>
      ))}
    </>
  );
}

// ponytail: current multiples only; no point-in-time EPS history source exists yet.
function Multiples({ sym }: { sym: string }) {
  const { data, error, retry } = useLoad(`multiples:${sym}`, (signal) => fetchWidgetData("multiples", sym, signal));
  if (!data && !error) return <span className="skeleton" aria-label="Loading multiples" aria-busy="true" />;
  const d: WidgetData | undefined = data;
  const empty = d && [d.price, d.ttm_eps, d.pe, d.earnings_yield].every((value) => value == null);
  return (
    <>
      <Status error={error ?? d?.error} empty={empty ? `No multiples for ${sym}. Try asking about its earnings.` : null} retry={retry} />
      {d && !error && !d.error && !empty && (
        <table className="num" style={{ fontSize: "var(--fs-body)" }}>
          <tbody>
            <tr><th style={head}>Price</th><td style={cell}>{num(d.price)}</td></tr>
            <tr><th style={head}>TTM EPS</th><td style={cell}>{num(d.ttm_eps)}</td></tr>
            <tr><th style={head}>P/E</th><td style={cell}>{num(d.pe)}</td></tr>
            <tr><th style={head}>Earnings yield</th><td style={cell}>{typeof d.earnings_yield === "number" ? pct(d.earnings_yield) : "—"}</td></tr>
          </tbody>
        </table>
      )}
      <div style={{ ...muted, fontSize: "var(--fs-label)" }}>History unavailable: no point-in-time EPS series.</div>
    </>
  );
}

const WIDGETS: Record<string, (w: WorkspaceWidget) => ReactNode> = {
  "price-volume": (w) => <TradingChart sym={w.ticker} params={w.params} />,
  financials: (w) => <FinancialsPanel sym={w.ticker} />,
  news: (w) => <NewsPanel sym={w.ticker} />,
  "monthly-returns": (w) => <MonthlyReturns sym={w.ticker} />,
  "short-interest": (w) => <SeriesTable type="short-interest" sym={w.ticker} columns={[["date", "Settled"], ["short_qty", "Short shares"], ["days_to_cover", "Days to cover"]]} />,
  "short-volume": (w) => <SeriesTable type="short-volume" sym={w.ticker} columns={[["date", "Date"], ["short_pct", "Short %"], ["short_qty", "Short"], ["total_qty", "Total"]]} />,
  insider: (w) => <SeriesTable type="insider" sym={w.ticker} columns={[["date", "Filed"], ["title", "Transaction"]]} />,
  filings: (w) => <FilingsTimeline sym={w.ticker} />,
  multiples: (w) => <Multiples sym={w.ticker} />,
  "tv-link": (w) => <TvLinkWidget params={w.params} />,
};

function WidgetBody({ widget, renderExtra }: { widget: WorkspaceWidget; renderExtra: (widget: WorkspaceWidget) => ReactNode }) {
  return WIDGETS[widget.type] ? WIDGETS[widget.type](widget) : renderExtra(widget);
}

function Tool({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return <button type="button" aria-label={label} title={label} onClick={onClick} className="ws-tool">{children}</button>;
}

type Geometry = WorkspaceWidget["geometry"];

function boundedGeometry(geometry: Geometry): Geometry {
  return {
    x: Math.max(0, Math.min(20000, Math.round(geometry.x))),
    y: Math.max(0, Math.min(20000, Math.round(geometry.y))),
    width: Math.max(280, Math.min(4000, Math.round(geometry.width))),
    height: Math.max(180, Math.min(4000, Math.round(geometry.height))),
  };
}

function keyTarget(key: string, shift: boolean): { field: keyof Geometry; delta: number } | null {
  const left = key === "ArrowLeft" ? -10 : key === "ArrowRight" ? 10 : null;
  if (left !== null) return { field: shift ? "width" : "x", delta: left };
  const vertical = key === "ArrowUp" ? -10 : key === "ArrowDown" ? 10 : null;
  if (vertical !== null) return { field: shift ? "height" : "y", delta: vertical };
  return null;
}
const WidgetFrame = memo(function WidgetFrame({ widget, focused, hidden, viewport, order, onFocus, onRaise, onGeometry, onRemove, onParams, onType, onTicker, renderExtra }: {
  widget: WorkspaceWidget;
  focused: boolean;
  hidden: boolean;
  viewport: Geometry;
  order: number;
  onFocus: (id: string) => void;
  onRaise: (id: string) => void;
  onGeometry: (id: string, geometry: Geometry) => void;
  onRemove: (id: string) => void;
  onParams: (id: string, params: Record<string, unknown>) => void;
  onType: (id: string, type: string) => void;
  onTicker: (id: string, ticker: string) => void;
  renderExtra: (widget: WorkspaceWidget) => ReactNode;
}) {
  const [rev, setRev] = useState(0);
  const [editing, setEditing] = useState(false);
  const [ticker, setTicker] = useState(widget.ticker);
  const instructions = useId();
  useEffect(() => setTicker(widget.ticker), [widget.ticker]);
  useEffect(() => setEditing(false), [widget.type]);
  const tvLink = widget.type === "tv-link";
  const market = isWidgetType(widget.type) || widget.type === "alerts";
  const title = tvLink && typeof widget.params?.label === "string" ? widget.params.label : widgetLabel(widget.type);
  const label = market ? `${title} · ${widget.ticker}` : title;
  const geometry = focused ? viewport : widget.geometry;
  function keyGeometry(e: ReactKeyboardEvent<HTMLElement>): void {
    if (e.target !== e.currentTarget || focused || e.altKey || e.ctrlKey || e.metaKey) return;
    const target = keyTarget(e.key, e.shiftKey);
    if (!target) return;
    e.preventDefault();
    onGeometry(widget.id, boundedGeometry({ ...widget.geometry, [target.field]: widget.geometry[target.field] + target.delta }));
  }
  return (
    <Rnd
      position={{ x: geometry.x, y: geometry.y }}
      size={{ width: geometry.width, height: geometry.height }}
      bounds="parent"
      dragHandleClassName="ws-window-header"
      cancel="input,textarea,button,select,form,a,[role=combobox]"
      minWidth={focused ? viewport.width : 280}
      minHeight={focused ? viewport.height : 180}
      maxWidth={focused ? viewport.width : 4000}
      maxHeight={focused ? viewport.height : 4000}
      disableDragging={focused}
      enableResizing={!focused}
      resizeHandleComponent={{ bottomRight: <span data-workspace-resize="bottom-right" aria-label={`Resize ${label}`} style={{ display: "block", width: "100%", height: "100%" }} /> }}
      onDragStart={() => onRaise(widget.id)}
      onResizeStart={() => onRaise(widget.id)}
      onDragStop={(_, data) => onGeometry(widget.id, boundedGeometry({ ...widget.geometry, x: data.x, y: data.y }))}
      onResizeStop={(_, __, element, ___, position) => onGeometry(widget.id, boundedGeometry({ ...position, width: element.offsetWidth, height: element.offsetHeight }))}
      style={{ zIndex: focused ? 100 : order, visibility: hidden ? "hidden" : undefined }}
    >
      <section aria-label={label} data-widget-type={widget.type} className={`glass ws-widget${focused ? " ws-widget-focus" : ""}`} onPointerDownCapture={() => onRaise(widget.id)} onFocusCapture={() => onRaise(widget.id)} style={{ width: "100%", height: "100%" }}>
        <header className="card-head ws-window-header">
          <span className="ws-window-title" tabIndex={0} role="group" aria-label={`Move or resize ${label}`} aria-describedby={instructions} onKeyDown={keyGeometry}>{label}</span>
          <span id={instructions} hidden>Use arrow keys to move 10 pixels. Use Shift and arrow keys to resize 10 pixels.</span>
          <span className="card-head-actions">
            <select aria-label={`Panel type for ${label}`} value={widget.type} onChange={(e) => onType(widget.id, e.target.value)}>
              {WORKSPACE_TYPES.filter((type) => type !== "tv-link" || tvLink).map((type) => <option key={type} value={type}>{widgetLabel(type)}</option>)}
            </select>
            {widget.type === "price-volume" && <ChartControls params={widget.params} onChange={(next) => onParams(widget.id, next)} />}
            {tvLink && <Tool label={`Edit ${label}`} onClick={() => setEditing((on) => !on)}>✎</Tool>}
            <Tool label={`Refresh ${label}`} onClick={() => setRev((n) => n + 1)}>↻</Tool>
            <Tool label={`${focused ? "Collapse" : "Expand"} ${label}`} onClick={() => onFocus(widget.id)}>⤢</Tool>
            <Tool label={`Remove ${label}`} onClick={() => onRemove(widget.id)}>×</Tool>
          </span>
        </header>
        {market && (
          <form className="ws-window-ticker" onSubmit={(e) => { e.preventDefault(); if (TICKER_RE.test(ticker) && ticker !== widget.ticker) onTicker(widget.id, ticker); }} onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); setTicker(widget.ticker); } }}>
            <input aria-label={`Ticker for ${title}`} value={ticker} pattern={TICKER_RE.source} required maxLength={10} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
            <button type="submit" className="pill">Set ticker</button>
          </form>
        )}
        <div className="ws-widget-body">
          {editing
            ? <TvLinkForm initial={widget.params} onSubmit={(params) => { setEditing(false); onParams(widget.id, params); }} onCancel={() => setEditing(false)} />
            : <WidgetBody key={rev} widget={widget} renderExtra={renderExtra} />}
        </div>
      </section>
    </Rnd>
  );
});

// A window Esc belongs to the innermost owner: a field, a form, an open menu, or the palette handles its own Esc first.
function freeEscape(e: KeyboardEvent): boolean {
  return e.key === "Escape" && !e.defaultPrevented
    && !(e.target instanceof window.Element && e.target.closest("input,textarea,select,form,[role=combobox],details[open]"));
}

// Geometry and stacking change only through onChange. Focus does not change saved geometry.
export function WidgetGrid({ widgets, renderExtra, onChange, onRemove, onParams, onType, onTicker }: {
  widgets: WorkspaceWidget[] | null;
  renderExtra: (widget: WorkspaceWidget) => ReactNode;
  onChange: (widgets: WorkspaceWidget[]) => void;
  onRemove: (id: string) => void;
  onParams: (id: string, params: Record<string, unknown>) => void;
  onType: (id: string, type: string) => void;
  onTicker: (id: string, ticker: string) => void;
}) {
  const [focused, setFocused] = useState<string | null>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState<Geometry>({ x: 0, y: 0, width: 1200, height: 800 });
  const focus = useCallback((id: string) => setFocused((previous) => previous === id ? null : id), []);
  const visible = widgets ?? [];
  const focusing = focused !== null && visible.some((w) => w.id === focused);
  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    const measure = () => setViewport({ x: element.scrollLeft, y: element.scrollTop, width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    element.addEventListener("scroll", measure);
    return () => { observer.disconnect(); element.removeEventListener("scroll", measure); };
  }, []);
  useEffect(() => {
    if (!focusing) return;
    const onKey = (e: KeyboardEvent) => {
      if (!freeEscape(e)) return;
      e.preventDefault();
      setFocused(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focusing]);
  function raise(id: string) {
    if (!widgets || widgets.at(-1)?.id === id) return;
    const widget = widgets.find((item) => item.id === id);
    if (widget) onChange([...widgets.filter((item) => item.id !== id), widget]);
  }
  function changeGeometry(id: string, geometry: Geometry) {
    if (!widgets) return;
    const widget = widgets.find((item) => item.id === id);
    if (!widget) return;
    const same = Object.keys(geometry).every((key) => geometry[key as keyof Geometry] === widget.geometry[key as keyof Geometry]);
    if (same && widgets.at(-1)?.id === id) return;
    onChange([...widgets.filter((item) => item.id !== id), { ...widget, geometry }]);
  }
  const width = Math.max(1200, viewport.width, ...visible.map((widget) => widget.geometry.x + widget.geometry.width + 40));
  const height = Math.max(800, viewport.height, ...visible.map((widget) => widget.geometry.y + widget.geometry.height + 40));
  return (
    <div ref={viewportRef} className="ws-viewport" style={{ overflow: focusing ? "hidden" : "auto", position: "relative" }}>
      <div className="ws-desktop" style={{ position: "relative", width, height }}>
        {widgets === null && <span className="skeleton" aria-label="Loading workspace" aria-busy="true" />}
        {widgets !== null && visible.length === 0 && <p className="feed-muted">No widgets yet. Choose Add widget or ask for a chart.</p>}
        {visible.map((widget, order) => (
          <WidgetFrame key={widget.id} widget={widget} focused={focusing && focused === widget.id} hidden={focusing && focused !== widget.id} viewport={viewport} order={order} onFocus={focus} onRaise={raise} onGeometry={changeGeometry} onRemove={onRemove} onParams={onParams} onType={onType} onTicker={onTicker} renderExtra={renderExtra} />
        ))}
      </div>
    </div>
  );
}
