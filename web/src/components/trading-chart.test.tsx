import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import type { Root } from "react-dom/client";
import { JSDOM } from "jsdom";
import TradingChart, { ChartControls, chartSettings, chartSrc, INTERVALS, STUDIES, STYLES } from "./trading-chart";

const dom = new JSDOM(`<body><div id="root"></div></body>`, { url: "https://example.com/" });
const registry = globalThis as Record<string, unknown>;
const DOM_GLOBALS = ["IS_REACT_ACT_ENVIRONMENT", "window", "document", "navigator", "requestAnimationFrame", "cancelAnimationFrame"];
const savedGlobals = new Map<string, PropertyDescriptor | undefined>();
const realFetch = globalThis.fetch;
let mounted: Root | null = null;
let onlineValue = true;

beforeAll(() => {
  for (const key of DOM_GLOBALS) savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  registry.IS_REACT_ACT_ENVIRONMENT = true;
  registry.window = dom.window;
  registry.document = dom.window.document;
  registry.navigator = dom.window.navigator;
  registry.requestAnimationFrame = (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0) as unknown as number;
  registry.cancelAnimationFrame = (frame: number) => clearTimeout(frame);
  Object.defineProperty(dom.window.navigator, "onLine", {
    configurable: true,
    get: () => onlineValue,
  });
});

afterEach(async () => {
  onlineValue = true;
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
});

afterAll(() => {
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
  dom.window.close();
});

async function mount(element: ReactNode): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(element));
  return container;
}

async function rerender(element: ReactNode): Promise<void> {
  await act(async () => mounted?.render(element));
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 200 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));
  }
  if (!ready()) throw new Error("the chart did not reach the expected state");
}

function frame(root: HTMLElement): HTMLIFrameElement | null {
  return root.querySelector("iframe");
}

function control(root: HTMLElement, label: string): HTMLButtonElement {
  const found = [...root.querySelectorAll("button")].find((entry) => entry.textContent === label);
  if (!found) throw new Error(`missing ${label} control`);
  return found;
}

async function click(target: Element): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

describe("TradingChart", () => {
  test("maps the current ticker to the TradingView widget source with no legacy fetch", async () => {
    const requests: string[] = [];
    globalThis.fetch = ((input: string | URL | Request) => {
      requests.push(String(input));
      return Promise.reject(new Error("chart must not fetch"));
    }) as unknown as typeof fetch;
    const root = await mount(<TradingChart sym="nvda" />);
    await until(() => frame(root)?.src.includes("symbol=NVDA") ?? false);
    const src = frame(root)?.src ?? "";
    expect(src.startsWith("https://s.tradingview.com/widgetembed/")).toBe(true);
    expect(src).not.toContain("6mo");
    expect(requests).toEqual([]);
    expect(frame(root)?.title).toContain("NVDA");
    await rerender(<TradingChart sym="AAPL" />);
    await until(() => frame(root)?.src.includes("symbol=AAPL") ?? false);
  });

  test("reports offline, empty-symbol, and widget-failure states with recovery", async () => {
    const empty = await mount(<TradingChart sym="  " />);
    expect(empty.querySelector('[role="alert"]')?.textContent).toContain("no symbol");
    await rerender(<TradingChart sym="NVDA" />);
    await until(() => frame(empty) !== null);

    onlineValue = false;
    await act(async () => {
      dom.window.dispatchEvent(new dom.window.Event("offline"));
    });
    expect(empty.querySelector('[role="alert"]')?.textContent).toContain("offline");
    expect(frame(empty)).toBeNull();
    onlineValue = true;
    await act(async () => {
      dom.window.dispatchEvent(new dom.window.Event("online"));
    });
    await until(() => frame(empty) !== null);

    await act(async () => {
      frame(empty)?.dispatchEvent(new dom.window.Event("error"));
    });
    await until(() => empty.querySelector('[role="alert"]')?.textContent?.includes("unavailable for NVDA") ?? false);
    await click(control(empty, "Retry chart"));
    await until(() => frame(empty) !== null);
  });

  test("a detached frame cannot fail the current chart after a symbol replacement", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    const bySymbol = frame(root);
    if (!bySymbol) throw new Error("missing symbol frame");
    await rerender(<TradingChart sym="AAPL" />);
    await until(() => (frame(root)?.src ?? "").includes("symbol=AAPL"));
    expect(bySymbol.isConnected).toBe(false);
    expect(frame(root)).not.toBe(bySymbol);
    await act(async () => {
      bySymbol.dispatchEvent(new dom.window.Event("error"));
    });
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect((frame(root)?.src ?? "").includes("symbol=AAPL")).toBe(true);
  });

  test("a widget failure clears when the symbol changes", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    await act(async () => {
      frame(root)?.dispatchEvent(new dom.window.Event("error"));
    });
    await until(() => root.querySelector('[role="alert"]')?.textContent?.includes("unavailable for NVDA") ?? false);
    await rerender(<TradingChart sym="AAPL" />);
    await until(() => (frame(root)?.src ?? "").includes("symbol=AAPL"));
    expect(root.querySelector('[role="alert"]')).toBeNull();
  });

  test("a frame error after remount fails the remounted chart, not the old frame", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    const beforeUnmount = frame(root);
    if (!beforeUnmount) throw new Error("missing pre-unmount frame");
    await act(async () => mounted?.unmount());
    mounted = null;
    expect(beforeUnmount.isConnected).toBe(false);
    await act(async () => {
      beforeUnmount.dispatchEvent(new dom.window.Event("error"));
    });
    const remounted = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(remounted) !== null);
    expect(remounted.querySelector('[role="alert"]')).toBeNull();
    await act(async () => {
      frame(remounted)?.dispatchEvent(new dom.window.Event("error"));
    });
    await until(() => remounted.querySelector('[role="alert"]')?.textContent?.includes("unavailable for NVDA") ?? false);
    expect(frame(remounted)).toBeNull();
  });

  test("the offline alert takes precedence over a widget failure and the online chart recovers", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    await act(async () => {
      frame(root)?.dispatchEvent(new dom.window.Event("error"));
    });
    await until(() => root.querySelector('[role="alert"]')?.textContent?.includes("unavailable for NVDA") ?? false);
    onlineValue = false;
    await act(async () => {
      dom.window.dispatchEvent(new dom.window.Event("offline"));
    });
    expect(root.querySelector('[role="alert"]')?.textContent).toContain("you are offline");
    expect(frame(root)).toBeNull();
    onlineValue = true;
    await act(async () => {
      dom.window.dispatchEvent(new dom.window.Event("online"));
    });
    await until(() => root.querySelector('[role="alert"]')?.textContent?.includes("unavailable for NVDA") ?? false);
    await click(control(root, "Retry chart"));
    await until(() => frame(root) !== null);
    expect(root.querySelector('[role="alert"]')).toBeNull();
  });
});

describe("chart settings", () => {
  const query = (src: string) => new URL(src).searchParams;

  test("the default chart keeps the original embed URL", () => {
    expect(chartSrc("NVDA")).toBe("https://s.tradingview.com/widgetembed/?symbol=NVDA&interval=D&theme=dark&style=1&locale=en&timezone=exchange");
    expect(chartSrc("NVDA", {})).toBe(chartSrc("NVDA"));
  });

  test("maps every interval and style to the embed values", () => {
    const intervals = INTERVALS.map((interval) => query(chartSrc("NVDA", { interval })).get("interval"));
    expect(intervals).toEqual(["5", "15", "60", "240", "D", "W"]);
    const styles = STYLES.map((style) => query(chartSrc("NVDA", { style })).get("style"));
    expect(styles).toEqual(["1", "2", "3"]);
  });

  test("maps indicators to embed studies and Volume to the built-in volume pane", () => {
    const all = query(chartSrc("NVDA", { studies: STUDIES }));
    expect(all.get("hide_volume")).toBeNull();
    expect(JSON.parse(all.get("studies") ?? "[]")).toEqual([
      { id: "MASimple@tv-basicstudies", inputs: { length: 20 } },
      { id: "MASimple@tv-basicstudies", inputs: { length: 50 } },
      { id: "MASimple@tv-basicstudies", inputs: { length: 200 } },
      { id: "MAExp@tv-basicstudies" },
      { id: "RSI@tv-basicstudies" },
      { id: "MACD@tv-basicstudies" },
      { id: "BB@tv-basicstudies" },
      { id: "VWAP@tv-basicstudies" },
    ]);
    const none = query(chartSrc("NVDA", { studies: [] }));
    expect(none.get("hide_volume")).toBe("1");
    expect(none.get("studies")).toBeNull();
  });

  test("unknown saved values fall back to the defaults and studies keep the menu order", () => {
    expect(chartSettings({ interval: "2m", style: 3, studies: ["RSI", "evil", "MA20", "RSI"] })).toEqual({ interval: "D", style: "candles", studies: ["MA20", "RSI"] });
    expect(chartSettings({ studies: "RSI" })).toEqual({ interval: "D", style: "candles", studies: ["Volume"] });
  });

  test("the chart frame follows its params and the controls report full settings", async () => {
    const changes: unknown[] = [];
    const root = await mount(
      <>
        <ChartControls params={{ interval: "1h", studies: ["Volume"] }} onChange={(next) => changes.push(next)} />
        <TradingChart sym="NVDA" params={{ interval: "1h", style: "area" }} />
      </>,
    );
    await until(() => frame(root) !== null);
    expect(query(frame(root)?.src ?? "").get("interval")).toBe("60");
    expect(query(frame(root)?.src ?? "").get("style")).toBe("3");
    const select = (label: string) => root.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;
    expect(select("Chart interval").value).toBe("1h");
    await act(async () => {
      select("Chart style").value = "line";
      select("Chart style").dispatchEvent(new dom.window.Event("change", { bubbles: true }));
    });
    const box = [...root.querySelectorAll("label")].find((label) => label.textContent === "MA 20")?.querySelector("input");
    if (!box) throw new Error("missing MA 20 indicator");
    await click(box);
    expect(changes).toEqual([
      { interval: "1h", style: "line", studies: ["Volume"] },
      { interval: "1h", style: "candles", studies: ["Volume", "MA20"] },
    ]);
  });
});
