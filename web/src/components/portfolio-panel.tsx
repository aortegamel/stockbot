import { useEffect, useState } from "react";

export type Holding = {
  ticker: string;
  quantity: string;
  average_cost: string | null;
  market_price: string | null;
  market_value: string | null;
  unrealized_gain: string | null;
  unrealized_gain_pct: string | null;
};

export type PortfolioView = {
  synced_at: string | null;
  cash: string | null;
  invested_value: string | null;
  total_value: string | null;
  positions: Holding[];
};

const muted = "var(--muted, #93A0B8)";
const down = "var(--down, #EF5350)";
const up = "var(--up, #26A69A)";

// Same base as market-data getJSON. A 404 means BROKER_ENABLED is off or OAuth is missing: null hides the panel.
async function load(path: string, init?: RequestInit): Promise<PortfolioView | null> {
  const base = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
  const res = await fetch(`${base}${path}`, init);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return (await res.json()) as PortfolioView;
}

const money = (v: string | null) => (v === null ? "—" : Number(v).toFixed(2));

// Holdings from the cached snapshot (GET /api/portfolio). Sync runs one read-only Robinhood pull.
export function PortfolioPanel({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  // undefined = first load in flight, null = broker unavailable (hidden).
  const [view, setView] = useState<PortfolioView | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let alive = true;
    load("/api/portfolio")
      .then((v) => alive && setView(v))
      .catch((e: unknown) => alive && setError(`portfolio unavailable: ${e instanceof Error ? e.message : String(e)}`));
    return () => {
      alive = false;
    };
  }, []);
  const sync = () => {
    setBusy(true);
    setError(null);
    load("/api/portfolio/sync", { method: "POST" })
      .then(setView)
      .catch((e: unknown) => setError(`sync failed: ${e instanceof Error ? e.message : String(e)}`))
      .finally(() => setBusy(false));
  };
  if (view === null || (view === undefined && !error)) return null;
  const cell: React.CSSProperties = { padding: "4px 6px", textAlign: "right" };
  return (
    <div
      style={{
        background: "var(--panel, #11161F)",
        border: "1px solid var(--border, #1E2635)",
        borderRadius: 6,
        padding: 16,
        height: "100%",
        minHeight: 0,
        overflowY: "auto",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 20, letterSpacing: "0.08em", color: muted, fontWeight: 700 }}>PORTFOLIO</div>
        <button type="button" onClick={sync} disabled={busy} style={{ fontSize: 16, cursor: busy ? "wait" : "pointer" }}>
          {busy ? "Syncing…" : "Sync"}
        </button>
      </div>
      <div style={{ fontSize: 14, color: muted, marginBottom: 8 }}>
        {view?.synced_at ? `last sync ${view.synced_at}` : "never synced"}
        {view?.total_value != null && ` · total ${money(view.total_value)}`}
      </div>
      {error && <div style={{ fontSize: 14, color: down, marginBottom: 8 }}>{error}</div>}
      {view && view.positions.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 16 }}>
          <thead>
            <tr style={{ color: muted }}>
              <th style={{ ...cell, textAlign: "left" }}>Ticker</th>
              <th style={cell}>Qty</th>
              <th style={cell}>Avg cost</th>
              <th style={cell}>Mkt value</th>
              <th style={cell}>P/L</th>
            </tr>
          </thead>
          <tbody>
            {view.positions.map((p, i) => {
              const gain = p.unrealized_gain === null ? null : Number(p.unrealized_gain);
              return (
                <tr key={i} onClick={() => onSelectTicker(p.ticker)} style={{ cursor: "pointer" }}>
                  <td style={{ ...cell, textAlign: "left" }}>
                    <button type="button" style={{ background: "none", border: 0, color: "inherit", font: "inherit", padding: 0, cursor: "pointer" }}>
                      {p.ticker}
                    </button>
                  </td>
                  <td style={cell}>{p.quantity}</td>
                  <td style={cell}>{money(p.average_cost)}</td>
                  <td style={cell}>{money(p.market_value)}</td>
                  <td style={{ ...cell, color: gain === null ? muted : gain < 0 ? down : up }}>
                    {money(p.unrealized_gain)}
                    {p.unrealized_gain_pct !== null && ` (${(Number(p.unrealized_gain_pct) * 100).toFixed(2)}%)`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {view && view.positions.length === 0 && <div style={{ fontSize: 16, color: muted }}>no holdings in the last sync</div>}
    </div>
  );
}
