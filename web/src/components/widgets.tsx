import { memo, useCallback, useState, type ReactNode } from "react";
import { WIDGET_TYPES } from "../../../needle-harness/lib/agent/types";
import { fetchCandles, fetchFilings, fetchWidgetData, type Candle, type FilingItem, type WidgetData, type WorkspaceWidget } from "../lib/market-data";
import { FinancialsPanel, NewsPanel } from "./market-panels";
import TradingChart from "./trading-chart";
import { useLoad } from "./home-panels";

// Menu labels for the fixed widget set; WIDGET_TYPES (the add_widget whitelist) sets the order.
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
};

export const isWidgetType = (type: unknown): type is string => typeof type === "string" && WIDGET_TYPES.includes(type);
export const widgetLabel = (type: string): string => LABELS[type] ?? type;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const muted = { color: "var(--muted, #93A0B8)" } as const;
const cell = { padding: "0 8px", height: 28, textAlign: "right" } as const;
const head = { ...cell, ...muted, fontSize: 12, fontWeight: 500 } as const;

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
        <table className="num" aria-label={`${sym} monthly returns`} style={{ fontSize: 14, borderCollapse: "collapse" }}>
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
        <table className="num" style={{ fontSize: 14, borderCollapse: "collapse", width: "100%" }}>
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
        <div key={i} style={{ fontSize: 14, marginBottom: 8 }}>
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
        <table className="num" style={{ fontSize: 14 }}>
          <tbody>
            <tr><th style={head}>Price</th><td style={cell}>{num(d.price)}</td></tr>
            <tr><th style={head}>TTM EPS</th><td style={cell}>{num(d.ttm_eps)}</td></tr>
            <tr><th style={head}>P/E</th><td style={cell}>{num(d.pe)}</td></tr>
            <tr><th style={head}>Earnings yield</th><td style={cell}>{typeof d.earnings_yield === "number" ? pct(d.earnings_yield) : "—"}</td></tr>
          </tbody>
        </table>
      )}
      <div style={{ ...muted, fontSize: 12 }}>History unavailable: no point-in-time EPS series.</div>
    </>
  );
}

const WIDGETS: Record<string, (sym: string) => ReactNode> = {
  "price-volume": (sym) => <TradingChart sym={sym} />,
  financials: (sym) => <FinancialsPanel sym={sym} />,
  news: (sym) => <NewsPanel sym={sym} />,
  "monthly-returns": (sym) => <MonthlyReturns sym={sym} />,
  "short-interest": (sym) => <SeriesTable type="short-interest" sym={sym} columns={[["date", "Settled"], ["short_qty", "Short shares"], ["days_to_cover", "Days to cover"]]} />,
  "short-volume": (sym) => <SeriesTable type="short-volume" sym={sym} columns={[["date", "Date"], ["short_pct", "Short %"], ["short_qty", "Short"], ["total_qty", "Total"]]} />,
  insider: (sym) => <SeriesTable type="insider" sym={sym} columns={[["date", "Filed"], ["title", "Transaction"]]} />,
  filings: (sym) => <FilingsTimeline sym={sym} />,
  multiples: (sym) => <Multiples sym={sym} />,
};

function WidgetBody({ widget }: { widget: WorkspaceWidget }) {
  return (WIDGETS[widget.type]?.(widget.ticker) as ReactNode) ?? null;
}

function Tool({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return <button type="button" aria-label={label} title={label} onClick={onClick} className="ws-tool">{children}</button>;
}

const WidgetFrame = memo(function WidgetFrame({ widget, expanded, onExpand, onRemove }: {
  widget: WorkspaceWidget;
  expanded: boolean;
  onExpand: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  const [rev, setRev] = useState(0);
  const label = `${widgetLabel(widget.type)} · ${widget.ticker}`;
  return (
    <section aria-label={label} className={`glass ws-widget${expanded ? " ws-widget-wide" : ""}`}>
      <header className="card-head">
        <span>{label}</span>
        <span className="card-head-actions">
          <Tool label={`Refresh ${label}`} onClick={() => setRev((n) => n + 1)}>↻</Tool>
          <Tool label={`${expanded ? "Collapse" : "Expand"} ${label}`} onClick={() => onExpand(widget.id)}>⤢</Tool>
          <Tool label={`Remove ${label}`} onClick={() => onRemove(widget.id)}>×</Tool>
        </span>
      </header>
      <div className="ws-widget-body"><WidgetBody key={rev} widget={widget} /></div>
    </section>
  );
});

// Refresh remounts one widget. Expand spans the full row.
export function WidgetGrid({ widgets, onRemove }: { widgets: WorkspaceWidget[]; onRemove: (id: string) => void }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const expand = useCallback((id: string) => setExpanded((previous) => previous === id ? null : id), []);
  const visible = widgets.filter((w) => isWidgetType(w.type));
  return (
    <div className="ws-grid">
      {visible.length === 0 && <p className="feed-muted">No widgets yet. Choose Add widget or ask for a chart.</p>}
      {visible.map((widget) => <WidgetFrame key={widget.id} widget={widget} expanded={expanded === widget.id} onExpand={expand} onRemove={onRemove} />)}
    </div>
  );
}
