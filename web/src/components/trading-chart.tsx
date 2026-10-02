import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { calcMA, calcMACD, calcRSI, genCandles } from "../lib/market-mock";

const UP = "#26A69A";
const DOWN = "#EF5350";
const ACCENT = "#2962FF";

const TABS = ["5M", "15M", "1H", "4H", "1D", "1W"];

type CandlePoint = { open: number; high: number; low: number; close: number };

function isCandlePoint(v: unknown): v is CandlePoint {
  if (!v || typeof v !== "object") return false;
  if (!("open" in v && "high" in v && "low" in v && "close" in v)) return false;
  return (
    typeof v.open === "number" &&
    typeof v.high === "number" &&
    typeof v.low === "number" &&
    typeof v.close === "number"
  );
}

const fmt = (v: number) =>
  v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Range-Bar shape: Bar dataKey "hl" gives x/y/width/height for the high-low
// range, open/close are interpolated — no axis-scale plumbing needed.
function CandleShape(props: {
  x?: number | string;
  y?: number | string;
  width?: number | string;
  height?: number | string;
  payload?: unknown;
}) {
  const { x, y, width, height, payload } = props;
  if (
    typeof x !== "number" ||
    typeof y !== "number" ||
    typeof width !== "number" ||
    typeof height !== "number"
  )
    return <g />;
  if (!isCandlePoint(payload)) return <g />;
  const { open, high, low, close } = payload;
  const range = high - low || 1;
  const yOpen = y + ((high - open) / range) * height;
  const yClose = y + ((high - close) / range) * height;
  const c = close >= open ? UP : DOWN;
  const cx = x + width / 2;
  const bTop = Math.min(yOpen, yClose);
  const bH = Math.max(Math.abs(yOpen - yClose), 1);
  const hw = Math.max(width * 0.35, 1);
  return (
    <g>
      <line x1={cx} y1={y} x2={cx} y2={y + height} stroke={c} strokeWidth={1} />
      <rect x={cx - hw} y={bTop} width={hw * 2} height={bH} fill={c} opacity={0.85} />
    </g>
  );
}

function CandleTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: unknown }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!isCandlePoint(d)) return null;
  const up = d.close >= d.open;
  return (
    <div
      style={{
        fontSize: 20,
        background: "rgba(20,22,48,0.82)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(255,255,255,0.16)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.2)",
        borderRadius: 10,
        color: "#F5F7FF",
        padding: "6px 10px",
        lineHeight: 1.8,
      }}
    >
      {(["O", "H", "L", "C"] as const).map((k, i) => (
        <div key={k} style={{ display: "flex", gap: 10 }}>
          <span style={{ color: "rgba(235,235,245,0.55)", width: 10 }}>{k}</span>
          <span style={{ color: k === "C" ? (up ? UP : DOWN) : "#F5F7FF" }}>
            {fmt([d.open, d.high, d.low, d.close][i])}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function TradingChart({
  sym,
  base,
  volLabel,
}: {
  sym: string;
  base: number;
  volLabel: string;
}) {
  const [chartType, setChartType] = useState<"candles" | "line">("candles");
  const [tab, setTab] = useState("1H");

  const candles = useMemo(() => genCandles(base, 80), [sym, base]);
  const ma20 = useMemo(() => calcMA(candles, 20), [candles]);
  const ma50 = useMemo(() => calcMA(candles, 50), [candles]);
  const rsi = useMemo(() => calcRSI(candles), [candles]);
  const macdData = useMemo(() => calcMACD(candles), [candles]);

  const data = candles.map((c, i) => ({
    ...c,
    i,
    hl: [c.low, c.high] as [number, number],
    ma20: ma20[i],
    ma50: ma50[i],
    rsi: rsi[i],
    macdLine: macdData[i].macd,
    signal: macdData[i].signal,
    hist: macdData[i].hist,
  }));

  const last = candles[candles.length - 1];
  const up = last.close >= candles[0].open;
  const pct = (((last.close - candles[0].open) / candles[0].open) * 100).toFixed(2);
  const lastRsi = rsi[rsi.length - 1] ?? 50;
  const lastMacd = macdData[macdData.length - 1];
  const yDomain: [number, number] = [
    Math.min(...candles.map((c) => c.low)) * 0.9995,
    Math.max(...candles.map((c) => c.high)) * 1.0005,
  ];
  const gridC = "rgba(255,255,255,0.12)";
  const tick = { fontSize: 20, fill: "rgba(235,235,245,0.72)" };
  const tx = "#F5F7FF";
  const mut = "rgba(235,235,245,0.72)";
  const bdr = "rgba(255,255,255,0.12)";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "transparent",
        border: "none",
        color: "#F5F7FF",
      }}
    >
      {/* Toolbar: symbol + timeframe tabs + chart-type toggle + last price */}
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
        <span style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.01em", color: tx }}>{sym}</span>
        <div style={{ display: "flex", gap: 2 }}>
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                fontSize: 20,
                padding: "3px 10px",
                cursor: "pointer",
                color: tab === t ? "#fff" : "rgba(235,235,245,0.6)",
                background: tab === t ? "#0A84FF" : "transparent",
                border: "none",
                borderRadius: 999,
              }}
            >
              {t}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 2 }}>
          {(["line", "candles"] as const).map((m) => (
            <button
              key={m}
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
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 30, fontWeight: 700, color: up ? UP : DOWN }}>{fmt(last.close)}</span>
          <span style={{ fontSize: 20, color: up ? UP : DOWN }}>
            {up ? "▲" : "▼"} {up ? "+" : ""}
            {pct}%
          </span>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 10,
          padding: "4px 14px",
          borderBottom: `1px solid ${bdr}`,
          flexShrink: 0,
          minHeight: 26,
          fontSize: 20,
        }}
      >
        {(
          [
            ["O", candles[0].open],
            ["H", Math.max(...candles.map((c) => c.high))],
            ["L", Math.min(...candles.map((c) => c.low))],
            ["C", last.close],
          ] satisfies Array<[string, number]>
        ).map(([k, v]) => (
          <span key={k}>
            <span style={{ color: mut }}>{k} </span>
            <span style={{ color: tx, fontWeight: 600 }}>{fmt(v)}</span>
          </span>
        ))}
        <span>
          <span style={{ color: ACCENT }}>MA20 </span>
          <span style={{ color: tx, fontWeight: 600 }}>
            {ma20[ma20.length - 1]?.toFixed(2) ?? "—"}
          </span>
        </span>
        <span>
          <span style={{ color: "#FF9800" }}>MA50 </span>
          <span style={{ color: tx, fontWeight: 600 }}>
            {ma50[ma50.length - 1]?.toFixed(2) ?? "—"}
          </span>
        </span>
        <span style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
          <span style={{ color: mut }}>
            RSI{" "}
            <span style={{ color: lastRsi > 70 ? DOWN : lastRsi < 30 ? UP : tx, fontWeight: 600 }}>
              {lastRsi.toFixed(1)}
            </span>
          </span>
          <span style={{ color: mut }}>
            MACD{" "}
            <span style={{ color: lastMacd.hist != null && lastMacd.hist >= 0 ? UP : DOWN, fontWeight: 600 }}>
              {lastMacd.macd.toFixed(2)}
            </span>
          </span>
          <span style={{ color: mut }}>VOL {volLabel}</span>
        </span>
      </div>

      {/* Panes */}
      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", padding: "4px 2px 2px" }}>
        <div style={{ flex: 3, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 6" stroke={gridC} vertical={false} />
              <XAxis dataKey="i" type="number" domain={[0, data.length - 1]} hide />
              <YAxis domain={yDomain} tick={tick} width={56} />
              <Tooltip content={<CandleTooltip />} />
              {chartType === "candles" ? (
                <Bar dataKey="hl" shape={<CandleShape />} isAnimationActive={false} fill="transparent" />
              ) : (
                <Line
                  dataKey="close"
                  stroke={ACCENT}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              <Line dataKey="ma20" stroke={ACCENT} strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
              <Line dataKey="ma50" stroke="#FF9800" strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div style={{ flex: 0.6, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 0, right: 8, left: -8, bottom: 0 }}>
              <XAxis dataKey="i" hide />
              <YAxis hide />
              <Bar dataKey="vol" fill={ACCENT} opacity={0.3} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ flex: 0.7, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 0, right: 8, left: -8, bottom: 0 }}>
              <XAxis dataKey="i" hide />
              <YAxis domain={[0, 100]} ticks={[30, 70]} tick={tick} width={56} />
              <ReferenceLine y={70} stroke={`${DOWN}55`} strokeDasharray="2 3" />
              <ReferenceLine y={30} stroke={`${UP}55`} strokeDasharray="2 3" />
              <Line dataKey="rsi" stroke={DOWN} strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div style={{ flex: 0.7, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 0, right: 8, left: -8, bottom: 2 }}>
              <XAxis dataKey="i" hide />
              <YAxis tick={tick} width={56} />
              <ReferenceLine y={0} stroke={gridC} />
              <Bar dataKey="hist" fill={ACCENT} opacity={0.45} isAnimationActive={false} />
              <Line dataKey="macdLine" stroke={ACCENT} strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
              <Line dataKey="signal" stroke="#FF9800" strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
