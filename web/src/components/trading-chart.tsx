import { useCallback, useEffect, useState } from "react";

// Present-day market data via the public TradingView Advanced-chart
// embed: a plain iframe, no scraping/automation/credentials/dependency.
const WIDGET_BASE = "https://s.tradingview.com/widgetembed/";

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

export default function TradingChart({ sym }: { sym: string }) {
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
  const src = `${WIDGET_BASE}?symbol=${encodeURIComponent(symbol)}&interval=D&theme=dark&style=1&locale=en&timezone=exchange`;

  useEffect(() => {
    setFailed(false);
    setLoaded(false);
  }, [src]);

  const bdr = "rgba(255,255,255,0.12)";

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
      <div
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
        <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.01em", color: "#F5F7FF" }}>
          {symbol}
        </span>
      </div>
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
