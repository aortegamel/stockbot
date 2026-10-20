import { useEffect, useRef, useState } from "react";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import { WatchlistPanel, watchlistFor } from "./market-panels";

const TICKER_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const ORGANIZATIONS: Record<string, true> = { FINRA: true, SEC: true, NYSE: true, NASDAQ: true, OTC: true, EDGAR: true, SHO: true, REG: true };

// The list lives only in this mounted terminal, like the chart selection.
export function WatchlistChatPanel({ selected, onSelect, events, onChange }: {
  selected: string;
  onSelect: (ticker: string) => void;
  events: AgentEvent[];
  onChange: (symbols: string[]) => void;
}) {
  const [symbols, setSymbols] = useState(() => watchlistFor(selected));
  const seen = useRef(new WeakSet<object>());
  useEffect(() => {
    setSymbols((previous) => previous.includes(selected) ? previous : [selected, ...previous]);
  }, [selected]);
  useEffect(() => {
    for (const event of events) {
      if (event.type !== "watchlist_update" || seen.current.has(event)) continue;
      seen.current.add(event);
      const { action, ticker } = event;
      if ((action !== "add" && action !== "remove") || typeof ticker !== "string"
        || !TICKER_RE.test(ticker) || ORGANIZATIONS[ticker]) continue;
      setSymbols((previous) => action === "remove"
        ? previous.filter((symbol) => symbol !== ticker)
        : previous.includes(ticker) ? previous : [...previous, ticker]);
    }
  }, [events]);
  useEffect(() => onChange(symbols), [symbols, onChange]);
  return <>
    <p role="status" style={{ color: "var(--muted, #93A0B8)", padding: "8px 16px", margin: 0 }}>
      Quotes are delayed. A dash means unavailable. Source dates show freshness. Failed sources show error or timeout.
    </p>
    {symbols.length > 0
      ? <WatchlistPanel selected={selected} onSelect={onSelect} symbols={symbols} />
      : <p role="status">Your watchlist is empty.</p>}
  </>;
}
