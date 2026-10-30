import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act, useState } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import { WIDGET_TYPES, type AgentEvent } from "../../../needle-harness/lib/agent/types";
import Workspace, { type WorkspaceDraft } from "../pages/Workspace";
import type { WorkspaceDoc } from "../lib/market-data";
import { TvLinkWidget } from "./tv-link";
import { widgetLabel } from "./widgets";

// React DOM reads browser globals at import time. The test must install those globals first.
const { createRoot } = await import("react-dom/client");
const realFetch = globalThis.fetch;
let mounted: Root | null = null;
const root = () => dom.window.document.getElementById("root")!;
const text = () => root().textContent ?? "";
const SNAPSHOT = "https://www.tradingview.com/x/8ot166mV/?utm_source=evil";
const SYMBOL = "https://www.tradingview.com/symbols/NASDAQ-NVDA/?utm_source=evil";

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  dom.window.localStorage.clear();
});

async function render(node: ReactNode) {
  mounted ??= createRoot(root());
  await act(async () => mounted!.render(node));
  for (let i = 0; i < 12; i++) await act(async () => { await Promise.resolve(); });
}
async function click(el: Element) {
  await act(async () => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })));
  for (let i = 0; i < 12; i++) await act(async () => { await Promise.resolve(); });
}
async function type(target: HTMLInputElement, value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value")?.set?.call(target, value);
    target.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
}
function find<T extends Element>(selector: string): T {
  const el = root().querySelector<T>(selector);
  if (!el) throw new Error(`missing ${selector}`);
  return el;
}
const button = (label: string) => [...root().querySelectorAll("button")].find((b) => b.textContent === label)!;

function expectSafeFrame(frame: HTMLIFrameElement, src: string) {
  expect(frame.getAttribute("src")).toBe(src);
  expect(frame.getAttribute("sandbox")).toBe("allow-scripts allow-same-origin allow-popups");
  expect(frame.getAttribute("referrerpolicy")).toBe("no-referrer");
}

describe("TvLinkWidget", () => {
  test("shows a snapshot image built from the ID with refresh and open", async () => {
    await render(<TvLinkWidget params={{ url: SNAPSHOT, label: "Setup" }} />);
    const img = find<HTMLImageElement>("img");
    expect(img.getAttribute("src")).toBe("https://s3.tradingview.com/snapshots/8/8ot166mV.png");
    expect(img.getAttribute("referrerpolicy")).toBe("no-referrer");
    expect(img.alt).toBe("Setup");
    await click(button("Refresh image"));
    expect(find<HTMLImageElement>("img").getAttribute("src")).toBe("https://s3.tradingview.com/snapshots/8/8ot166mV.png?r=1");
    const open = find<HTMLAnchorElement>("a");
    expect(open.getAttribute("href")).toBe("https://www.tradingview.com/x/8ot166mV/");
    expect(open.getAttribute("rel")).toBe("noopener noreferrer");
    expect(root().querySelector("iframe")).toBeNull();
  });

  test("embeds symbols, chart params, and ideas through sandboxed embed URLs only", async () => {
    await render(<TvLinkWidget params={{ url: SYMBOL, label: "NVDA" }} />);
    expectSafeFrame(find("iframe"), "https://s.tradingview.com/widgetembed/?symbol=NASDAQ%3ANVDA&interval=D&theme=dark&style=1&locale=en&timezone=exchange");
    await render(<TvLinkWidget params={{ url: "https://www.tradingview.com/chart/?symbol=NASDAQ%3ANVDA&interval=240", label: "NVDA 4h" }} />);
    expectSafeFrame(find("iframe"), "https://s.tradingview.com/widgetembed/?symbol=NASDAQ%3ANVDA&interval=240&theme=dark&style=1&locale=en&timezone=exchange");
    await render(<TvLinkWidget params={{ url: "https://www.tradingview.com/chart/AUDUSD/gRwCMwkV-AUDUSD-is-Nearing/?utm_source=evil", label: "Idea" }} />);
    expectSafeFrame(find("iframe"), "https://s.tradingview.com/embed/gRwCMwkV/");
    expect(root().innerHTML).not.toContain("utm_source");
  });

  test("offers an open button and snapshot guidance for layouts, and an alert for bad links", async () => {
    await render(<TvLinkWidget params={{ url: "https://www.tradingview.com/chart/AbCd1234/", label: "Layout" }} />);
    expect(root().querySelector("iframe")).toBeNull();
    expect(find<HTMLAnchorElement>("a").getAttribute("href")).toBe("https://www.tradingview.com/chart/AbCd1234/");
    expect(text()).toContain("Copy link");
    await render(<TvLinkWidget params={{ url: "javascript:alert(1)", label: "Bad" }} />);
    expect(find('[role="alert"]').textContent).toContain("not supported");
    expect(root().querySelector("iframe, img, a")).toBeNull();
  });
});

function serve(saved: WorkspaceDoc["widgets"]) {
  const id = "a16ec102-2e68-49fb-b818-e9671a3d2474";
  const docs: Record<string, WorkspaceDoc> = { [id]: { id, name: "Links", widgets: saved, updated_at: "2026-10-20T10:00:00Z" } };
  const puts: WorkspaceDoc["widgets"][] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), dom.window.location.href);
    if (url.pathname === "/api/workspaces") return Response.json({ workspaces: Object.values(docs) });
    if (url.pathname.startsWith("/api/workspaces/") && init?.method === "PUT") {
      const id = url.pathname.split("/").at(-1)!;
      const body = JSON.parse(String(init.body)) as { name: string; widgets: WorkspaceDoc["widgets"] };
      puts.push(body.widgets);
      docs[id] = { id, ...body, updated_at: "2026-10-20T12:00:00+00:00" };
      return Response.json(docs[id]);
    }
    if (url.pathname === "/api/portfolio") return new Response(null, { status: 404 });
    if (url.pathname === "/api/theses") return Response.json([]);
    if (url.pathname === "/api/watchlist") return Response.json({ watchlist: [] });
    return Response.json({});
  }) as typeof fetch;
  return { saved: () => docs[id], puts };
}

function Page({ events = [] }: { events?: AgentEvent[] }) {
  const [workspace, setWorkspace] = useState<WorkspaceDraft | null>(null);
  return <Workspace selected="NVDA" pick={() => { }} watchlist={["NVDA"]} onWatchlist={() => { }}
    events={events} input="" setInput={() => { }} research={false} setResearch={() => { }} setAskContext={() => { }} onSend={() => { }} openDrawer={() => { }} busy={false} debug={false}
    workspace={workspace} setWorkspace={setWorkspace} sessions={[]} sessionId={null} setSessionId={() => { }} snapshot={null} journal={null} seq={0} chat={null} />;
}

async function chooseTvLink() {
  await act(async () => {
    const select = find<HTMLSelectElement>('select[aria-label="Add widget"]');
    select.value = "tv-link";
    select.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}

describe("Workspace TradingView links", () => {
  test("every widget type has a menu label", async () => {
    serve([]);
    await render(<Page />);
    for (const type of WIDGET_TYPES) expect(widgetLabel(type)).not.toBe(type);
  });

  test("adds several links through the form with preview, saves unique ids, and edits with ✎", async () => {
    const { saved, puts } = serve([]);
    await render(<Page />);
    await chooseTvLink();
    await type(find('input[aria-label="TradingView URL"]'), "http://www.tradingview.com/x/8ot166mV/");
    expect(find('form [role="alert"]').textContent).toContain("https tradingview.com");
    expect(button("Add link").disabled).toBe(true);
    await type(find('input[aria-label="TradingView URL"]'), ` ${SNAPSHOT} `);
    await type(find('input[aria-label="Widget label"]'), "Setup");
    expect(find('[aria-label="Link preview"] img').getAttribute("src")).toBe("https://s3.tradingview.com/snapshots/8/8ot166mV.png");
    await click(button("Add link"));
    expect(root().querySelector('form[aria-label="TradingView link"]')).toBeNull();
    // The menu keeps tv-link after one is added; the label defaults from the parsed kind.
    expect(find<HTMLSelectElement>('select[aria-label="Add widget"]').querySelector('option[value="tv-link"]')).not.toBeNull();
    await chooseTvLink();
    await type(find('input[aria-label="TradingView URL"]'), SYMBOL);
    await click(button("Add link"));
    expect(find('[aria-label="Setup · NVDA"] img')).not.toBeNull();
    expectSafeFrame(find('[aria-label="TradingView NASDAQ:NVDA · NVDA"] iframe'), "https://s.tradingview.com/widgetembed/?symbol=NASDAQ%3ANVDA&interval=D&theme=dark&style=1&locale=en&timezone=exchange");
    await click(button("Save"));
    expect(saved().widgets.map(({ type, ticker, params }) => ({ type, ticker, params }))).toEqual([
      { type: "tv-link", ticker: "NVDA", params: { url: "https://www.tradingview.com/x/8ot166mV/?utm_source=evil", label: "Setup" } },
      { type: "tv-link", ticker: "NVDA", params: { url: SYMBOL, label: "TradingView NASDAQ:NVDA" } },
    ]);
    expect(new Set(saved().widgets.map((w) => w.id)).size).toBe(2);
    await click(find('[aria-label="Edit Setup · NVDA"]'));
    expect(find<HTMLInputElement>('[aria-label="Setup · NVDA"] input[aria-label="TradingView URL"]').value).toBe("https://www.tradingview.com/x/8ot166mV/?utm_source=evil");
    await type(find('[aria-label="Setup · NVDA"] input[aria-label="TradingView URL"]'), "https://www.tradingview.com/chart/AbCd1234/");
    await click(button("Save link"));
    expect(puts).toHaveLength(1);
    await click(button("Save"));
    expect(puts.at(-1)?.find((w) => w.params?.label === "Setup")?.params).toEqual({ url: "https://www.tradingview.com/chart/AbCd1234/", label: "Setup" });
    expect(find('[aria-label="Setup · NVDA"]').textContent).toContain("Copy link");
  });

  test("ignores an agent tv-link receipt, which carries no URL", async () => {
    const { puts } = serve([]);
    await render(<Page events={[{ type: "widget_update", ticker: "NVDA", widgetType: "tv-link" }]} />);
    expect(puts).toHaveLength(0);
    expect(root().querySelectorAll(".ws-widget")).toHaveLength(0);
  });
});
