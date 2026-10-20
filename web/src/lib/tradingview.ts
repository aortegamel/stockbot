// Pasted TradingView links → a parsed kind with validated IDs only.
// Render code builds every image, embed, and open URL from these fields, never from the pasted URL.
export type TradingViewLink =
  | { kind: "snapshot"; id: string }
  | { kind: "symbol"; symbol: string }
  | { kind: "chart-params"; symbol: string; interval?: string }
  | { kind: "idea"; id: string }
  | { kind: "layout"; id: string };

const HOST = /^(?:www\.)?tradingview\.com$/;
const ID = /^[A-Za-z0-9]{8}$/;
const IDEA = /^([A-Za-z0-9]{8})(?:-|$)/;
const PART = "[A-Z0-9_.!]{1,30}";
const SYMBOL = new RegExp(`^(?:${PART}:)?${PART}$`);
// TradingView chart interval tokens → the saved chart interval values (trading-chart INTERVALS).
const INTERVAL: Record<string, string> = { "5": "5m", "15": "15m", "60": "1h", "240": "4h", D: "D", "1D": "D", W: "W", "1W": "W" };

export const MAX_URL = 300;
export const MAX_LABEL = 60;

type Page = { url: URL; parts: string[] };

function trustedPage(raw: string): Page | null {
  const text = raw.trim();
  if (text.length === 0 || text.length > MAX_URL) return null;
  try {
    const url = new URL(text);
    const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    // Only https on tradingview.com or www., with no port or credentials.
    const trusted = url.protocol === "https:" && HOST.test(url.hostname) && !url.port && !url.username && !url.password;
    return trusted ? { url, parts } : null;
  } catch {
    return null;
  }
}

function symbolOf(raw: string | null | undefined): string | null {
  const symbol = (raw ?? "").toUpperCase();
  return SYMBOL.test(symbol) ? symbol : null;
}

function chartParams({ url }: Page): TradingViewLink | null {
  const symbol = symbolOf(url.searchParams.get("symbol"));
  if (!symbol) return null;
  const interval = INTERVAL[url.searchParams.get("interval")?.toUpperCase() ?? ""];
  return interval ? { kind: "chart-params", symbol, interval } : { kind: "chart-params", symbol };
}

// Ideas: /chart/<SYMBOL>/<id>-<slug>/.
function ideaLink({ parts }: Page): TradingViewLink | null {
  const idea = IDEA.exec(parts[2]);
  return idea ? { kind: "idea", id: idea[1] } : null;
}

// "<first path part>/<part count>" → the matcher for that path shape.
// Keys always hold "/", so no Object.prototype name can match a pasted path.
const SHAPES: Record<string, (page: Page) => TradingViewLink | null> = {
  "x/2": ({ parts }) => ID.test(parts[1]) ? { kind: "snapshot", id: parts[1] } : null,
  "chart/1": chartParams,
  "chart/2": ({ parts }) => ID.test(parts[1]) ? { kind: "layout", id: parts[1] } : null,
  "chart/3": ideaLink,
};

export function parseTradingViewLink(raw: string): TradingViewLink | null {
  const page = trustedPage(raw);
  if (!page) return null;
  const [head, a] = page.parts;
  if (head === "symbols") {
    // /symbols/NASDAQ-NVDA/ → NASDAQ:NVDA; the first dash splits exchange and ticker.
    const symbol = symbolOf(a?.replace("-", ":"));
    return symbol ? { kind: "symbol", symbol } : null;
  }
  return SHAPES[`${head}/${page.parts.length}`]?.(page) ?? null;
}

// Snapshot images live under the lowercased first ID character (checked live 2026-10-20).
export const snapshotImage = (id: string) => `https://s3.tradingview.com/snapshots/${id[0].toLowerCase()}/${id}.png`;
