import { memo, useCallback, useEffect, useState, type ReactNode } from "react";
import {
  fetchBriefing, fetchIdentity, fetchPositioning, fetchScans, fetchSparklines,
  type MarketRow, type MarketSort, type ScanDef,
} from "../lib/market-data";
import { routeHash } from "../lib/route";

const INDEXES = ["SPY", "QQQ", "DIA", "IWM", "CL=F", "GC=F", "^TNX"];
const SUMMARY_KEY = "stockbot.home.summary";
const SORTS: [MarketSort, string][] = [["chg_pct", "Chg %"], ["volume", "Volume"], ["short_pct", "Short %"]];

type Load<T> = { data?: T; error?: string; retry: () => void };

// One request per key. A new key, a retry, or unmount aborts the previous request,
// so a stale response never lands. attempt > 0 marks a user retry.
export function useLoad<T>(key: string, fn: (signal: AbortSignal, attempt: number) => Promise<T>): Load<T> {
  const [attempt, setAttempt] = useState(0);
  const [got, setGot] = useState<{ at: string; data?: T; error?: string }>({ at: "" });
  const at = `${key}#${attempt}`;
  useEffect(() => {
    const ctrl = new AbortController();
    fn(ctrl.signal, attempt).then(
      (data) => !ctrl.signal.aborted && setGot({ at, data }),
      () => !ctrl.signal.aborted && setGot({ at, error: "We could not load this data." }),
    );
    return () => ctrl.abort();
    // fn reads only key and attempt; both are in `at`.
  }, [at]); // eslint-disable-line react-hooks/exhaustive-deps
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return got.at === at ? { data: got.data, error: got.error, retry } : { retry };
}

function LoadError({ retry }: { retry: () => void }) {
  return (
    <p className="home-error" role="alert">
      We could not load this data. <button type="button" className="pill" onClick={(e) => { e.stopPropagation(); retry(); }}>Retry</button>
    </p>
  );
}

// Skeleton while loading, plain error + Retry, one-line empty state.
export function View<T>({ load, empty, children }: { load: Load<T>; empty?: (d: T) => string | null; children: (d: T) => ReactNode }) {
  if (load.error) return <LoadError retry={load.retry} />;
  if (load.data === undefined) return (
    <div role="status" aria-label="Loading data" aria-busy="true">
      {[0, 1, 2].map((n) => <span key={n} className="skeleton" aria-hidden="true" />)}
    </div>
  );
  const none = empty?.(load.data);
  return none ? <p className="home-empty">{none}</p> : <>{children(load.data)}</>;
}

const num = (v: number | null | undefined, digits = 2) => (v === null || v === undefined ? "—" : v.toFixed(digits));
const signed = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`);
const tone = (v: number | null | undefined) => (v === null || v === undefined ? "" : v >= 0 ? "up" : "down");
const change = (closes: number[]) => (closes.length > 1 && closes[0] ? ((closes[closes.length - 1] - closes[0]) / closes[0]) * 100 : null);

export function Sparkline({ closes, w = 120, h = 32 }: { closes: number[]; w?: number; h?: number }) {
  if (closes.length < 2) return <span className="home-empty">no data</span>;
  const lo = Math.min(...closes);
  const span = Math.max(...closes) - lo || 1;
  const pts = closes.map((c, i) => `${((i / (closes.length - 1)) * w).toFixed(1)},${(h - ((c - lo) / span) * h).toFixed(1)}`);
  return (
    <svg className={`spark ${tone(change(closes))}`} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`${closes.length} point chart`}>
      <path className="spark-area" d={`M0,${h} L${pts.join(" L")} L${w},${h} Z`} />
      <polyline className="spark-line" points={pts.join(" ")} />
    </svg>
  );
}

export function IndexBand({ watchlist }: { watchlist: string[] }) {
  const syms = [...INDEXES, ...watchlist.filter((s) => !INDEXES.includes(s)).slice(0, 3)];
  const load = useLoad(syms.join(","), (s) => fetchSparklines(syms, "1d", s));
  return (
    <View load={load}>
      {(d) => (
        <section className="home-indexes">
          {syms.map((sym, i) => {
            const line = d.syms[sym];
            const closes = line && !("error" in line) ? line.closes : [];
            return (
              <article key={sym} className={`glass home-card ${i === 0 ? "home-big" : "home-mini"}`}>
                <header className="home-head">
                  <b>{sym}</b>
                  <span className="num">{num(closes.at(-1))}</span>
                  <span className={`num ${tone(change(closes))}`}>{signed(change(closes))}</span>
                </header>
                {line && "error" in line ? <LoadError retry={load.retry} /> : closes.length < 2 ? (
                  <p className="home-empty">No chart data. Try asking: What moved {sym} today?</p>
                ) : <Sparkline closes={closes} w={i === 0 ? 480 : 120} h={i === 0 ? 120 : 32} />}
              </article>
            );
          })}
        </section>
      )}
    </View>
  );
}

export function Briefing({ watchlist }: { watchlist: string[] }) {
  const [all, setAll] = useState(false);
  const load = useLoad(watchlist.join(","), (s, attempt) => fetchBriefing(watchlist, attempt > 0, s));
  return (
    <section className="glass home-card home-brief">
      <header className="home-head">
        <b>Market Briefing</b>
        <button type="button" className="pill" onClick={load.retry}>Refresh</button>
      </header>
      <View load={load} empty={(d) => (d.items.length || d.errors.length ? null : "No briefing items yet. Try asking: What moved my watchlist today?")}>
        {(d) => (
          <>
            <ul className="home-list">
              {(all ? d.items : d.items.slice(0, 5)).map((it, i) => (
                <li key={i}>
                  <b>{it.sym}</b> <a href={it.link} target="_blank" rel="noreferrer">{it.title}</a>{" "}
                  <span className="home-muted">{it.filing_form ?? it.kind} · {it.published ?? it.filed ?? ""}</span>
                </li>
              ))}
            </ul>
            {d.items.length > 5 && (
              <button type="button" className="pill" onClick={() => setAll(!all)}>{all ? "Show less" : `Show all ${d.items.length}`}</button>
            )}
            <p className="home-muted">{d.date} · {d.cached ? "cached" : "fresh"}{d.errors.length ? ` · ${d.errors.length} source errors` : ""}</p>
            {d.errors.length > 0 && <LoadError retry={load.retry} />}
          </>
        )}
      </View>
    </section>
  );
}

export function Scans({ active, onScan, onCreate }: { active: string; onScan: (scan: ScanDef) => void; onCreate: () => void }) {
  const load = useLoad("scans", (s) => fetchScans(s));
  const row = (scan: ScanDef) => (
    <li key={scan.id}>
      <button type="button" className="home-scan" aria-pressed={active === scan.id} onClick={() => onScan(scan)}>{scan.name}</button>
    </li>
  );
  return (
    <section className="glass home-card">
      <b>Market Scans</b>
      <View load={load} empty={(d) => (d.scans.length ? null : "No scans yet. Try asking: Scan for unusual volume.")}>
        {(d) => {
          const yours = d.scans.filter((s) => s.kind === "custom");
          return (
            <>
              <h4 className="home-sub">Yours</h4>
              {yours.length ? <ul className="home-list">{yours.map(row)}</ul> : <p className="home-empty">No saved scans. Try asking: Scan for heavily shorted names.</p>}
              <h4 className="home-sub">Featured</h4>
              {d.scans.some((s) => s.kind === "preset") ? <ul className="home-list">{d.scans.filter((s) => s.kind === "preset").map(row)}</ul> : <p className="home-empty">No featured scans. Try asking: Scan for unusual volume.</p>}
            </>
          );
        }}
      </View>
      <button type="button" className="pill" onClick={onCreate}>Create your own</button>
    </section>
  );
}

export function Detail({ sym }: { sym: string }) {
  const spark = useLoad(sym, (s) => fetchSparklines([sym], "1d", s));
  const pos = useLoad(sym, (s) => fetchPositioning(sym, s));
  const id = useLoad(sym, (s) => fetchIdentity(sym, s));
  const [hidden, setHidden] = useState(() => {
    try {
      return window.localStorage.getItem(SUMMARY_KEY) === "hidden";
    } catch {
      return false;
    }
  });
  function toggle() {
    setHidden(!hidden);
    try {
      window.localStorage.setItem(SUMMARY_KEY, hidden ? "shown" : "hidden");
    } catch {
      // Storage can be disabled; the choice then lasts for this page load only.
    }
  }
  const bar = (label: string, pct: number, text: string, cls: string) => (
    <div className="home-bar">
      <span>{label}</span>
      <span className="home-bar-track"><span className={`home-bar-fill ${cls}`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} /></span>
      <span className="num">{text}</span>
    </div>
  );
  return (
    <section className="glass home-card" aria-label={`${sym} detail`}>
      <header className="home-head">
        <b>{sym}</b>
        <a href={routeHash({ page: "workspace", sym })}>Open workspace</a>
        <button type="button" className="pill" onClick={toggle}>{hidden ? "Show summary" : "Hide summary"}</button>
      </header>
      <View load={spark}>
        {(d) => {
          const line = d.syms[sym];
          if (line && "error" in line) return <LoadError retry={spark.retry} />;
          return line && line.closes.length >= 2 ? <Sparkline closes={line.closes} w={280} h={64} /> : (
            <p className="home-empty">No chart data. Try asking: Show the price history for {sym}.</p>
          );
        }}
      </View>
      {!hidden && (
        <>
          <View load={id}>
            {(i) => i.error ? <LoadError retry={id.retry} /> : i.resolved ? (
              <p className="home-muted">
                entity {i.entity_id ?? "—"}<br />
                security {i.security_id ?? "—"} · {i.method ?? "—"}
              </p>
            ) : <p className="home-empty">No company details. Try asking: Tell me about {sym}.</p>}
          </View>
          <View load={pos} empty={(p) => (p.short_pct !== null || p.insider_net_90d || Object.keys(p.errors).length ? null : `No positioning data. Try asking: Research ${sym} short interest.`)}>
            {(p) => {
              const ins = p.insider_net_90d;
              const max = Math.max(ins?.buys ?? 0, ins?.sells ?? 0) || 1;
              return (
                <div className="home-pos">
                  {bar("Short %", p.short_pct ?? 0, p.short_pct === null ? "—" : `${p.short_pct.toFixed(1)}%`, "down")}
                  {ins ? (
                    <>
                      {bar("Insider buys 90d", (ins.buys / max) * 100, ins.buys.toLocaleString("en-US"), "up")}
                      {bar("Insider sells 90d", (ins.sells / max) * 100, ins.sells.toLocaleString("en-US"), "down")}
                    </>
                  ) : null}
                  {Object.keys(p.errors).length > 0 && <LoadError retry={pos.retry} />}
                </div>
              );
            }}
          </View>
        </>
      )}
    </section>
  );
}

const MarketTableRow = memo(function MarketTableRow({ row: r, selected, favorite, tabStop, onSelect, onToggle, onRetry }: {
  row: MarketRow;
  selected: boolean;
  favorite: boolean;
  tabStop: boolean;
  onSelect: (sym: string) => void;
  onToggle: (sym: string) => void;
  onRetry: () => void;
}) {
  return (
    <tr
      aria-selected={selected}
      tabIndex={tabStop ? 0 : -1}
      onClick={() => onSelect(r.sym)}
      onFocus={(e) => { if (e.target === e.currentTarget) onSelect(r.sym); }}
    >
      <td>
        <button
          type="button"
          className="home-heart"
          aria-label={favorite ? `Remove ${r.sym} from watchlist` : `Add ${r.sym} to watchlist`}
          onClick={(e) => { e.stopPropagation(); onToggle(r.sym); }}
        >{favorite ? "♥" : "♡"}</button>
      </td>
      <td><b>{r.sym}</b></td>
      <td className="num">{num(r.last)}</td>
      <td className={`num ${tone(r.chg_pct)}`}>{signed(r.chg_pct)}</td>
      <td className="num">{r.volume === null ? "—" : r.volume.toLocaleString("en-US")}</td>
      <td className="num">{r.short_pct === null ? "—" : `${r.short_pct.toFixed(1)}%`}</td>
      <td>
        {r.last_filing ? `${r.last_filing.form} ${r.last_filing.filed}` : "—"}
        {r.error && <LoadError retry={onRetry} />}
      </td>
    </tr>
  );
});

export function MarketsTable({ rows, total, sel, sort, watchlist, onSort, onSelect, onWatchlist, onRetry }: {
  rows: MarketRow[];
  total: number;
  sel: string | undefined;
  sort: MarketSort;
  watchlist: string[];
  onSort: (sort: MarketSort) => void;
  onSelect: (sym: string) => void;
  onWatchlist: (symbols: string[]) => void;
  onRetry: () => void;
}) {
  const onToggle = useCallback((sym: string) => {
    onWatchlist(watchlist.includes(sym) ? watchlist.filter((s) => s !== sym) : [...watchlist, sym]);
  }, [watchlist, onWatchlist]);
  const tabSym = rows.some((r) => r.sym === sel) ? sel : rows[0]?.sym;
  return (
    <>
      <table className="home-table num">
        <thead>
          <tr>
            <th /><th>Symbol</th><th>Last</th>
            {SORTS.map(([key, label]) => (
              <th key={key}><button type="button" className="home-sort" aria-pressed={sort === key} onClick={() => onSort(key)}>{label}</button></th>
            ))}
            <th>Last filing</th>
          </tr>
        </thead>
        <tbody onKeyDown={(e) => {
          const items = Array.from(e.currentTarget.rows);
          const index = items.findIndex((row) => row === e.target);
          if (index < 0) return;
          const current = items[index];
          let next = index;
          if (e.key === "ArrowDown") next = Math.min(items.length - 1, index + 1);
          else if (e.key === "ArrowUp") next = Math.max(0, index - 1);
          else if (e.key === "Home") next = 0;
          else if (e.key === "End") next = items.length - 1;
          else if (e.key === "Enter" || e.key === " ") current.click();
          else return;
          e.preventDefault();
          items[next]?.focus();
        }}>
          {rows.map((r) => (
            <MarketTableRow
              key={r.sym}
              row={r}
              selected={r.sym === sel}
              favorite={watchlist.includes(r.sym)}
              tabStop={r.sym === tabSym}
              onSelect={onSelect}
              onToggle={onToggle}
              onRetry={onRetry}
            />
          ))}
        </tbody>
      </table>
      {/* ponytail: slice at 200 rows; add virtualization if scans grow past that. */}
      {total > rows.length && <p className="home-muted">showing {rows.length} of {total}</p>}
    </>
  );
}

