import { useCallback, useEffect, useRef, useState } from "react";

// Present-day market data via the public TradingView Advanced-chart
// embed: a plain iframe, no scraping/automation/credentials/dependency.
const WIDGET_BASE = "https://s.tradingview.com/widgetembed/";

// Chart settings saved in the price-volume widget params. Only these values are stored.
export const INTERVALS = ["5m", "15m", "1h", "4h", "D", "W"];
export const STYLES = ["candles", "line", "area"];
export const STUDIES = ["Volume", "MA20", "MA50", "MA200", "EMA", "RSI", "MACD", "BB", "VWAP"];
export type ChartParams = { interval: string; style: string; studies: string[] };

const EMBED_INTERVAL: Record<string, string> = { "5m": "5", "15m": "15", "1h": "60", "4h": "240", D: "D", W: "W" };
const STUDY_LABEL: Record<string, string> = { MA20: "MA 20", MA50: "MA 50", MA200: "MA 200", BB: "Bollinger Bands" };
const ma = (length: number) => ({ id: "MASimple@tv-basicstudies", inputs: { length } });
// Volume is the embed's built-in pane (hide_volume), not a study.
const EMBED_STUDY: Record<string, object> = {
  MA20: ma(20),
  MA50: ma(50),
  MA200: ma(200),
  EMA: { id: "MAExp@tv-basicstudies" },
  RSI: { id: "RSI@tv-basicstudies" },
  MACD: { id: "MACD@tv-basicstudies" },
  BB: { id: "BB@tv-basicstudies" },
  VWAP: { id: "VWAP@tv-basicstudies" },
};

// Saved params → known values only. A missing or unknown value falls back to the default chart (D, candles, volume).
export function chartSettings(params: Record<string, unknown> = {}): ChartParams {
  const { interval, style, studies } = params;
  return {
    interval: typeof interval === "string" && INTERVALS.includes(interval) ? interval : "D",
    style: typeof style === "string" && STYLES.includes(style) ? style : "candles",
    studies: Array.isArray(studies) ? STUDIES.filter((s) => studies.includes(s)) : ["Volume"],
  };
}

export function chartSrc(symbol: string, params?: Record<string, unknown>): string {
  const s = chartSettings(params);
  const query = new URLSearchParams({
    symbol,
    interval: EMBED_INTERVAL[s.interval],
    theme: "dark",
    style: String(STYLES.indexOf(s.style) + 1),
    locale: "en",
    timezone: "exchange",
  });
  const studies = s.studies.filter((s) => s !== "Volume").map((s) => EMBED_STUDY[s]);
  // tv.js sends object-form studies as a JSON array.
  if (studies.length > 0) query.set("studies", JSON.stringify(studies));
  if (!s.studies.includes("Volume")) query.set("hide_volume", "1");
  return `${WIDGET_BASE}?${query}`;
}

// Card-title controls; each change reports the full normalized settings.
export function ChartControls({ params, onChange }: { params?: Record<string, unknown>; onChange: (next: ChartParams) => void }) {
  const s = chartSettings(params);
  const menu = useRef<HTMLDetailsElement>(null);
  // Close the indicators menu on any click outside it.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (menu.current?.open && !menu.current.contains(e.target as Node)) menu.current.open = false;
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);
  return (
    <>
      <select aria-label="Chart interval" value={s.interval} onChange={(e) => onChange({ ...s, interval: e.target.value })} className="ws-tool">
        {INTERVALS.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
      <select aria-label="Chart style" value={s.style} onChange={(e) => onChange({ ...s, style: e.target.value })} className="ws-tool">
        {STYLES.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
      <details
        ref={menu}
        style={{ position: "relative" }}
        onKeyDown={(e) => {
          // Esc closes the open menu first and claims the key, so window Esc listeners skip it.
          if (e.key !== "Escape" || e.defaultPrevented || !menu.current?.open) return;
          e.preventDefault();
          menu.current.open = false;
          menu.current.querySelector("summary")?.focus();
        }}
      >
        <summary className="ws-tool" style={{ listStyle: "none", cursor: "pointer" }}>Indicators</summary>
        <div role="group" aria-label="Indicators" className="glass" style={{ position: "absolute", right: 0, top: "100%", zIndex: 20, padding: 8, display: "grid", gap: 4, minWidth: 160 }}>
          {STUDIES.map((study) => (
            <label key={study} style={{ display: "flex", gap: 8, alignItems: "center", whiteSpace: "nowrap", textTransform: "none" }}>
              <input
                type="checkbox"
                checked={s.studies.includes(study)}
                onChange={() => onChange({ ...s, studies: STUDIES.filter((x) => x === study ? !s.studies.includes(x) : s.studies.includes(x)) })}
              />
              {STUDY_LABEL[study] ?? study}
            </label>
          ))}
        </div>
      </details>
    </>
  );
}

const alertStyle = { padding: 16, fontSize: 14, color: "rgba(235,235,245,0.72)" } as const;

function PlainAlert({ text, action, label }: { text: string; action?: () => void; label?: string }) {
  return (
    <div role="alert" style={{ ...alertStyle, display: "flex", gap: 12, alignItems: "center" }}>
      <span>{text}</span>
      {action && label && (
        <button
          onClick={action}
          style={{ fontSize: 14, padding: "3px 12px", cursor: "pointer", color: "#fff", background: "#0A84FF", border: "none", borderRadius: 999 }}
        >
          {label}
        </button>
      )}
    </div>
  );
}

export default function TradingChart({ sym, params }: { sym: string; params?: Record<string, unknown> }) {
  const [loaded, setLoaded] = useState(false);
  const [online, setOnline] = useState(
    () => typeof navigator === "undefined" || navigator.onLine !== false,
  );
  const [failed, setFailed] = useState(false);
  const attachError = useCallback((frame: HTMLIFrameElement | null) => {
    if (frame === null) return undefined;
    const onError = () => setFailed(true);
    frame.addEventListener("error", onError);
    return () => frame.removeEventListener("error", onError);
  }, []);

  useEffect(() => {
    const goOnline = () => setOnline(true);
    const goOffline = () => setOnline(false);
    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, []);

  const symbol = sym.trim().toUpperCase();
  const src = chartSrc(symbol, params);

  useEffect(() => {
    setFailed(false);
    setLoaded(false);
  }, [src]);

  if (!symbol) {
    return <PlainAlert text="chart unavailable — no symbol selected. Try: search a ticker above." />;
  }

  if (!online) {
    return <PlainAlert text="chart unavailable — you are offline." action={() => setOnline(navigator.onLine !== false)} label="Retry chart" />;
  }

  if (failed) {
    return <PlainAlert text={`TradingView chart unavailable for ${symbol}`} action={() => { setFailed(false); setLoaded(false); }} label="Retry chart" />;
  }


  return (
    <div
      role="region"
      aria-label={`${symbol} price chart`}
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "transparent",
        border: "none",
        color: "#F5F7FF",
      }}
    >
      {!loaded && <div className="skeleton" style={{ flex: 1, minHeight: 200 }} aria-label="Loading chart" />}
      <iframe
        key={src}
        ref={attachError}
        src={src}
        title={`${symbol} TradingView chart`}
        onLoad={() => setLoaded(true)}
        style={{ flex: 1, width: "100%", minHeight: 0, border: "none", display: loaded ? undefined : "none" }}
      />
    </div>
  );
}
