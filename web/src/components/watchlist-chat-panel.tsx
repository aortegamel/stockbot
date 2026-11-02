import { WatchlistPanel } from "./market-panels";

// Shell loads the saved list and persists edits, including agent receipts.
export function WatchlistChatPanel({ selected, onSelect, symbols }: {
  selected: string;
  onSelect: (ticker: string) => void;
  symbols: string[];
}) {
  return <>
    <p role="status" style={{ color: "var(--muted, #93A0B8)", padding: "8px 16px 0", margin: 0, fontSize: "var(--fs-xs)", lineHeight: "16px" }}>
      Quotes are delayed. A dash means unavailable. Source dates show freshness. Failed sources show error or timeout.
    </p>
    <WatchlistPanel selected={selected} onSelect={onSelect} symbols={symbols} />
  </>;
}
