import { useState, type FormEvent } from "react";
import { MAX_LABEL, MAX_URL, parseTradingViewLink, snapshotImage, type TradingViewLink } from "../lib/tradingview";
import { chartSrc } from "./trading-chart";

// C0 probe (2026-10-20, headless Chromium, this sandbox + no-referrer):
// live: s.tradingview.com/widgetembed (symbol, chart params), s.tradingview.com/embed/<idea id>, s3 snapshot PNGs.
// blocked (frame-ancestors 'none'): every www.tradingview.com page: /x/, /symbols/, /chart/, idea pages, layouts.
const SANDBOX = "allow-scripts allow-same-origin allow-popups";
const muted = { color: "var(--muted, #93A0B8)", fontSize: "var(--fs-label)" } as const;
const field = { height: 28, padding: "0 8px", border: "1px solid var(--hair)", borderRadius: 6, background: "var(--fill)", color: "inherit", font: "inherit", minWidth: 0 } as const;

function Open({ href }: { href: string }) {
  return <a href={href} target="_blank" rel="noopener noreferrer" className="pill">Open in TradingView</a>;
}

function Frame({ src, title }: { src: string; title: string }) {
  return <iframe src={src} title={title} sandbox={SANDBOX} referrerPolicy="no-referrer" style={{ flex: 1, width: "100%", minHeight: 300, height: "100%", border: "none" }} />;
}

function Snapshot({ id, label }: { id: string; label: string }) {
  // A query string defeats the image cache on Refresh.
  const [rev, setRev] = useState(0);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 12 }}>
      <img src={rev ? `${snapshotImage(id)}?r=${rev}` : snapshotImage(id)} alt={label} referrerPolicy="no-referrer" style={{ maxWidth: "100%", objectFit: "contain" }} />
      <span style={{ display: "flex", gap: 8 }}>
        <button type="button" className="pill" onClick={() => setRev((n) => n + 1)}>Refresh image</button>
        <Open href={`https://www.tradingview.com/x/${id}/`} />
      </span>
    </div>
  );
}

// Every URL below comes from parsed fields, never from the pasted link.
function TvLinkView({ link, label }: { link: TradingViewLink; label: string }) {
  if (link.kind === "snapshot") return <Snapshot id={link.id} label={label} />;
  if (link.kind === "symbol") return <Frame src={chartSrc(link.symbol)} title={`${label} TradingView embed`} />;
  if (link.kind === "chart-params") return <Frame src={chartSrc(link.symbol, link.interval ? { interval: link.interval } : undefined)} title={`${label} TradingView embed`} />;
  if (link.kind === "idea") return <Frame src={`https://s.tradingview.com/embed/${link.id}/`} title={`${label} TradingView embed`} />;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 12, fontSize: "var(--fs-body)" }}>
      <span>TradingView does not allow saved layouts inside other sites.</span>
      <span style={muted}>To show this chart here, open it, choose the camera icon, then Copy link. Paste that snapshot link with ✎.</span>
      <span><Open href={`https://www.tradingview.com/chart/${link.id}/`} /></span>
    </div>
  );
}

export function TvLinkWidget({ params }: { params?: Record<string, unknown> }) {
  const link = typeof params?.url === "string" ? parseTradingViewLink(params.url) : null;
  const label = typeof params?.label === "string" ? params.label : "TradingView";
  if (!link) return <p role="alert" style={{ fontSize: "var(--fs-body)", padding: 12 }}>This TradingView link is not supported. Choose ✎ to paste another link.</p>;
  return <TvLinkView link={link} label={label} />;
}

// A link the form can save: parsed, with a normalized href within MAX_URL, and the label to show and store.
function savable(url: string, label: string) {
  const link = parseTradingViewLink(url);
  if (!link) return null;
  const href = new URL(url.trim()).href;
  if (href.length > MAX_URL) return null;
  return { link, href, name: label.trim() || `TradingView ${"symbol" in link ? link.symbol : link.kind}` };
}

// Add and ✎ edit. The stored url is the normalized href of a link the parser accepts.
export function TvLinkForm({ initial, onSubmit, onCancel }: {
  initial?: Record<string, unknown>;
  onSubmit: (params: { url: string; label: string }) => void;
  onCancel: () => void;
}) {
  const [url, setUrl] = useState(typeof initial?.url === "string" ? initial.url : "");
  const [label, setLabel] = useState(typeof initial?.label === "string" ? initial.label : "");
  const draft = savable(url, label);
  function submit(e: FormEvent) {
    e.preventDefault();
    if (draft) onSubmit({ url: draft.href, label: draft.name.slice(0, MAX_LABEL) });
  }
  return (
    <form aria-label="TradingView link" onSubmit={submit} onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); onCancel(); } }} style={{ display: "flex", flexDirection: "column", gap: 8, padding: 4, minHeight: 0 }}>
      <span style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <input autoFocus aria-label="TradingView URL" placeholder="https://www.tradingview.com/x/…" value={url} maxLength={MAX_URL} onChange={(e) => setUrl(e.target.value)} style={{ ...field, flex: "2 1 240px" }} />
        <input aria-label="Widget label" placeholder="Label" value={label} maxLength={MAX_LABEL} onChange={(e) => setLabel(e.target.value)} style={{ ...field, flex: "1 1 120px" }} />
        <button type="submit" className="pill" disabled={!draft}>{initial ? "Save link" : "Add link"}</button>
        <button type="button" className="pill" onClick={onCancel}>Cancel</button>
      </span>
      {url.trim() && !draft && <span role="alert" style={muted}>Paste an https tradingview.com snapshot, symbol, chart, idea, or layout link.</span>}
      {draft && (
        <div aria-label="Link preview" style={{ display: "flex", flexDirection: "column", minHeight: 240, border: "1px solid var(--hair)", borderRadius: 6, overflow: "hidden" }}>
          <span style={{ ...muted, padding: "4px 8px" }}>Preview · {draft.link.kind}</span>
          <TvLinkView link={draft.link} label={draft.name} />
        </div>
      )}
    </form>
  );
}
