import { useState, type ReactNode } from "react";
import { Briefing, Detail, IndexTape, indexSyms, MarketsTable, Scans, SpyChart, useLoad, View } from "./home-panels";
import type { ShellState } from "../layout/Shell";
import { fetchMarkets, fetchSparklines, type MarketSort, type ScanDef } from "../lib/market-data";

export const HOME_TYPES = ["index-tape", "spy-chart", "briefing", "scans", "markets", "market-detail"];
const MAX_ROWS = 200;

type Props = Pick<ShellState, "watchlist" | "onWatchlist" | "setInput" | "setAskContext" | "openDrawer"> & {
  children: (render: (type: string) => ReactNode) => ReactNode;
};

export default function HomePanels(s: Props) {
  const [scan, setScan] = useState<ScanDef | null>(null);
  const [sort, setSort] = useState<MarketSort>("chg_pct");
  const [picked, setPicked] = useState<string | null>(null);
  // One sparkline request feeds both the SPY chart and the index tape.
  const idxSyms = indexSyms(s.watchlist);
  const indexes = useLoad(idxSyms.join(","), (signal) => fetchSparklines(idxSyms, "1d", signal));
  // Custom scans store only a filter: run the watchlist scan over their symbols.
  const custom = scan?.kind === "custom";
  const scanId = custom || !scan ? "watchlist" : scan.id;
  const syms = custom && scan.filter.symbols?.length ? scan.filter.symbols : s.watchlist;
  const markets = useLoad(`${scanId}|${syms.join(",")}|${sort}`, (signal) => fetchMarkets(scanId, syms, sort, signal));
  const rows = markets.data?.rows ?? [];
  const sel = picked ?? rows[0]?.sym;
  return s.children((type) => {
    switch (type) {
      case "index-tape": return <IndexTape load={indexes} syms={idxSyms.slice(1)} />;
      case "spy-chart": return <SpyChart load={indexes} />;
      case "briefing": return <Briefing watchlist={s.watchlist} />;
      case "scans": return <Scans
        active={scan?.id ?? "watchlist"}
        onScan={(next) => { setScan(next); setPicked(null); }}
        onCreate={() => { s.setInput("scan for "); s.setAskContext(null); s.openDrawer(); }}
      />;
      case "market-detail": return sel ? <Detail sym={sel} /> : null;
      case "markets": return (
        <section className="glass home-card home-markets">
          <header className="card-head">
            <b>Markets · {scan?.name ?? "Watchlist"}</b>
            {(markets.data?.errors.length ?? 0) > 0 && (
              <span className="home-note" role="alert">
                Some market data is unavailable. <button type="button" className="ghost" onClick={markets.retry}>Retry</button>
              </span>
            )}
          </header>
          <div className="home-scroll">
            <View load={markets} empty={(d) => (d.rows.length || d.errors.length ? null : "No rows for this scan. Try: Ask Stockbot: scan for unusual volume")}>
              {(d) => (
                <MarketsTable
                  rows={d.rows.slice(0, MAX_ROWS)} sel={sel} sort={sort} watchlist={s.watchlist}
                  onSort={setSort} onSelect={setPicked} onWatchlist={s.onWatchlist} onRetry={markets.retry}
                />
              )}
            </View>
          </div>
          {/* ponytail: slice at 200 rows. Add virtualization if scans exceed that limit. */}
          {rows.length > MAX_ROWS && <footer className="home-foot">showing {MAX_ROWS} of {rows.length}</footer>}
        </section>
      );
      default: return null;
    }
  });
}
