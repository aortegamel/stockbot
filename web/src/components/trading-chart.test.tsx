import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import type { Root } from "react-dom/client";
import { JSDOM } from "jsdom";
import TradingChart from "./trading-chart";

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

  test("every interval control issues a distinct widget source", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    const seen = new Set<string>();
    for (const [label, param] of [["5M", "interval=5"], ["15M", "interval=15"], ["1H", "interval=60"], ["4H", "interval=240"], ["1D", "interval=D"], ["1W", "interval=W"]] as const) {
      await click(control(root, label));
      await until(() => (frame(root)?.src ?? "").includes(param));
      expect(control(root, label).getAttribute("aria-pressed")).toBe("true");
      seen.add(frame(root)?.src ?? "");
    }
    expect(seen.size).toBe(6);
  });

  test("the style toggle issues a distinct widget source", async () => {
    const root = await mount(<TradingChart sym="NVDA" />);
    await until(() => frame(root) !== null);
    await click(control(root, "Line"));
    await until(() => (frame(root)?.src ?? "").includes("style=2"));
    await click(control(root, "Candles"));
    await until(() => (frame(root)?.src ?? "").includes("style=1"));
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

  test("a detached frame cannot fail the current chart after a symbol or interval replacement", async () => {
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

    const byInterval = frame(root);
    if (!byInterval) throw new Error("missing interval frame");
    await click(control(root, "1H"));
    await until(() => (frame(root)?.src ?? "").includes("interval=60"));
    expect(byInterval.isConnected).toBe(false);
    await act(async () => {
      byInterval.dispatchEvent(new dom.window.Event("error"));
    });
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect((frame(root)?.src ?? "").includes("interval=60")).toBe(true);
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
