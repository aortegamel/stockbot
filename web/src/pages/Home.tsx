import { useState } from "react";
import { PortfolioPanel } from "../components/portfolio-panel";
import { Briefing, Detail, IndexTape, indexSyms, MarketsTable, Scans, SpyChart, useLoad, View } from "../components/home-panels";
import type { ShellState } from "../layout/Shell";
import { fetchMarkets, fetchSparklines, type MarketSort, type ScanDef } from "../lib/market-data";

const MAX_ROWS = 200;

export default function Home(s: Pick<ShellState, "watchlist" | "onWatchlist" | "pick" | "setInput" | "setAskContext" | "openDrawer">) {
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
  return (
    <div className="page-scroll home">
      <IndexTape load={indexes} syms={idxSyms.slice(1)} />
      <div className="home-mid">
        <SpyChart load={indexes} />
        <Briefing watchlist={s.watchlist} />
      </div>
      <div className="home-grid">
        <Scans
          active={scan?.id ?? "watchlist"}
          onScan={(next) => {
            setScan(next);
            setPicked(null);
          }}
          onCreate={() => {
            s.setInput("scan for ");
            s.setAskContext(null);
            s.openDrawer();
          }}
        />
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
                  rows={d.rows.slice(0, MAX_ROWS)}
                  sel={sel}
                  sort={sort}
                  watchlist={s.watchlist}
                  onSort={setSort}
                  onSelect={setPicked}
                  onWatchlist={s.onWatchlist}
                  onRetry={markets.retry}
                />
              )}
            </View>
          </div>
          {/* ponytail: slice at 200 rows; add virtualization if scans grow past that. */}
          {rows.length > MAX_ROWS && <footer className="home-foot">showing {MAX_ROWS} of {rows.length}</footer>}
        </section>
        <div className="home-right">
          {sel && <Detail sym={sel} />}
          <PortfolioPanel onSelectTicker={s.pick} />
        </div>
      </div>
    </div>
  );
}
