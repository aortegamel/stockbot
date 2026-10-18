import { useCallback, useEffect, useState } from "react";

// Present-day market data via the public TradingView Advanced-chart
// embed: a plain iframe, no scraping/automation/credentials/dependency.
const WIDGET_BASE = "https://s.tradingview.com/widgetembed/";

const alertStyle = { padding: 16, fontSize: 20, color: "rgba(235,235,245,0.72)" } as const;

export default function TradingChart({ sym }: { sym: string }) {
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
      </div>
      <iframe
        key={src}
        ref={attachError}
        src={src}
        title={`${symbol} TradingView chart`}
        style={{ flex: 1, width: "100%", minHeight: 0, border: "none" }}
      />
    </div>
  );
}
