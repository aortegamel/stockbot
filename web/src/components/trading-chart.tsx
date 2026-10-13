import { useCallback, useEffect, useMemo, useState } from "react";

// Present-day market data via the public TradingView Advanced-chart
// embed: a plain iframe, no scraping/automation/credentials/dependency.
const WIDGET_BASE = "https://s.tradingview.com/widgetembed/";

const INTERVALS = [
  { label: "5M", param: "5" },
  { label: "15M", param: "15" },
  { label: "1H", param: "60" },
  { label: "4H", param: "240" },
  { label: "1D", param: "D" },
  { label: "1W", param: "W" },
] as const;

type IntervalLabel = (typeof INTERVALS)[number]["label"];
type ChartType = "candles" | "line";

// TradingView chart styles: 1 = candles, 2 = line.
const STYLE: Record<ChartType, string> = { candles: "1", line: "2" };

const alertStyle = { padding: 16, fontSize: 20, color: "rgba(235,235,245,0.72)" } as const;

export default function TradingChart({ sym }: { sym: string }) {
  const [interval, setInterval] = useState<IntervalLabel>("1D");
  const [chartType, setChartType] = useState<ChartType>("candles");
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
  const intervalParam = INTERVALS.find((t) => t.label === interval)?.param ?? "D";
  const src = useMemo(
    () =>
      `${WIDGET_BASE}?symbol=${encodeURIComponent(symbol)}&interval=${intervalParam}&theme=dark&style=${STYLE[chartType]}&locale=en&timezone=exchange`,
    [symbol, intervalParam, chartType],
  );

  useEffect(() => {
    setFailed(false);
  }, [src]);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  const bdr = "rgba(255,255,255,0.12)";

  if (!symbol) {
    return (
      <div role="alert" style={alertStyle}>
        chart unavailable — no symbol selected
      </div>
    );
  }

  if (!online) {
    return (
      <div role="alert" style={alertStyle}>
        chart unavailable — you are offline
      </div>
    );
  }

  if (failed) {
    return (
      <div role="alert" style={{ ...alertStyle, display: "flex", gap: 12, alignItems: "center" }}>
        <span>TradingView chart unavailable for {symbol}</span>
        <button
          onClick={() => setFailed(false)}
          style={{
            fontSize: 20,
            padding: "3px 12px",
            cursor: "pointer",
            color: "#fff",
            background: "#0A84FF",
            border: "none",
            borderRadius: 999,
          }}
        >
          Retry chart
        </button>
      </div>
    );
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
      <div
        role="toolbar"
        aria-label="Chart controls"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 14px",
          borderBottom: `1px solid ${bdr}`,
          flexShrink: 0,
          minHeight: 38,
        }}
      >
        <span style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.01em", color: "#F5F7FF" }}>
          {symbol}
        </span>
        <div role="group" aria-label="Chart interval" style={{ display: "flex", gap: 2 }}>
          {INTERVALS.map((t) => (
            <button
              key={t.label}
              aria-pressed={interval === t.label}
              onClick={() => setInterval(t.label)}
              style={{
                fontSize: 20,
                padding: "3px 10px",
                cursor: "pointer",
                color: interval === t.label ? "#fff" : "rgba(235,235,245,0.6)",
                background: interval === t.label ? "#0A84FF" : "transparent",
                border: "none",
                borderRadius: 999,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div role="group" aria-label="Chart style" style={{ display: "flex", gap: 2 }}>
          {(["line", "candles"] as const).map((m) => (
            <button
              key={m}
              aria-pressed={chartType === m}
              onClick={() => setChartType(m)}
              style={{
                fontSize: 20,
                padding: "3px 12px",
                cursor: "pointer",
                textTransform: "capitalize",
                color: chartType === m ? "#fff" : "rgba(235,235,245,0.6)",
                background: chartType === m ? "#0A84FF" : "rgba(10,132,255,0.22)",
                border: "none",
                borderRadius: 999,
              }}
            >
              {m === "line" ? "Line" : "Candles"}
            </button>
          ))}
        </div>
      </div>
      <iframe
        key={src}
        ref={attachError}
        src={src}
        title={`${symbol} TradingView chart`}
        style={{ flex: 1, width: "100%", minHeight: 0, border: "none" }}
      />
      <div
        style={{
          flexShrink: 0,
          padding: "2px 14px",
          borderTop: `1px solid ${bdr}`,
          fontSize: 12,
        }}
      >
        <a
          href="https://www.tradingview.com/"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "rgba(235,235,245,0.6)" }}
        >
          Chart by TradingView
        </a>
      </div>
    </div>
  );
}
