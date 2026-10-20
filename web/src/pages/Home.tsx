import { useState } from "react";
import { PortfolioPanel } from "../components/portfolio-panel";
import { Briefing, Detail, IndexBand, MarketsTable, Scans, useLoad, View } from "../components/home-panels";
import type { ShellState } from "../layout/Shell";
import { fetchMarkets, type MarketSort, type ScanDef } from "../lib/market-data";

const MAX_ROWS = 200;

export default function Home(s: Pick<ShellState, "watchlist" | "onWatchlist" | "pick" | "setInput" | "openDrawer">) {
  const [scan, setScan] = useState<ScanDef | null>(null);
  const [sort, setSort] = useState<MarketSort>("chg_pct");
  const [picked, setPicked] = useState<string | null>(null);
  // Custom scans store only a filter: run the watchlist scan over their symbols.
  const custom = scan?.kind === "custom";
  const scanId = custom || !scan ? "watchlist" : scan.id;
  const syms = custom && scan.filter.symbols?.length ? scan.filter.symbols : s.watchlist;
  const markets = useLoad(`${scanId}|${syms.join(",")}|${sort}`, (signal) => fetchMarkets(scanId, syms, sort, signal));
  const rows = markets.data?.rows ?? [];
  const sel = picked ?? rows[0]?.sym;
  return (
    <div className="page-scroll home">
      <div className="home-band">
        <IndexBand watchlist={s.watchlist} />
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
            s.openDrawer();
          }}
        />
        <section className="glass home-card">
          <b>Markets · {scan?.name ?? "Watchlist"}</b>
          <View load={markets} empty={(d) => (d.rows.length || d.errors.length ? null : "No rows for this scan. Try: Ask Stockbot: scan for unusual volume")}>
            {(d) => (
              <>
                {d.errors.length > 0 && (
                  <p className="home-error" role="alert">
                    Some market data is unavailable. <button type="button" className="pill" onClick={markets.retry}>Retry</button>
                  </p>
                )}
                <MarketsTable
                  rows={d.rows.slice(0, MAX_ROWS)}
                  total={d.rows.length}
                  sel={sel}
                  sort={sort}
                  watchlist={s.watchlist}
                  onSort={setSort}
                  onSelect={setPicked}
                  onWatchlist={s.onWatchlist}
                  onRetry={markets.retry}
                />
              </>
            )}
          </View>
        </section>
        <div className="home-right">
          {sel && <Detail sym={sel} />}
          <PortfolioPanel onSelectTicker={s.pick} />
        </div>
      </div>
    </div>
  );
}
