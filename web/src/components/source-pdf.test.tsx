import { afterAll, afterEach, beforeAll, describe, expect, mock, test } from "bun:test";
import { act } from "react";
import { createRoot } from "react-dom/client";
import type { Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
import type { SourceDocument } from "../lib/research-client";
import type { QuoteResolution } from "../lib/source-viewer";

// PDF.js is the external rendering boundary. The viewer still creates the page DOM and quote marks.
let pageText: string[][] = [];
let failure: unknown;
let documentGate: Promise<void> | undefined;
let destroyed = 0;
let cleaned = 0;
const requests: Record<string, unknown>[] = [];
const renderedPages: number[] = [];
const workerOptions = { workerSrc: "" };
mock.module("pdfjs-dist", () => ({
  GlobalWorkerOptions: workerOptions,
  getDocument: (options: Record<string, unknown>) => {
    requests.push(options);
    return {
      destroy: async () => { destroyed++; },
      promise: (async () => {
        await documentGate;
        if (failure !== undefined) throw failure;
        return {
          numPages: pageText.length,
          getPage: async (page: number) => {
            return {
              getViewport: ({ scale }: { scale: number }) => ({ scale, width: 100.5, height: 200.5 }),
              render: () => {
                renderedPages.push(page);
                return { promise: Promise.resolve() };
              },
              getTextContent: async () => ({ items: pageText[page - 1].map((str) => ({ str })) }),
              cleanup: () => { cleaned++; },
            };
          },
          cleanup: async () => { cleaned++; },
        };
      })(),
    };
  },
  TextLayer: class {
    constructor(private options: { container: HTMLDivElement; textContentSource: { items: { str: string }[] } }) { }
    async render() {
      for (const item of this.options.textContentSource.items) {
        const span = this.options.container.ownerDocument.createElement("span");
        span.textContent = item.str;
        this.options.container.append(span);
      }
    }
  },
}));
// Dynamic import: static import would bind the real pdfjs-dist before mock.module installs the fake engine.
const { default: SourcePdf } = await import("./source-pdf");

const dom = new JSDOM('<body><div id="pdf-root"></div></body>', { url: "https://viewer.example/" });
const savedGlobals = new Map<string, PropertyDescriptor | undefined>();
const scrolls: { element: HTMLElement; options: boolean | ScrollIntoViewOptions | undefined }[] = [];
let mounted: Root | null = null;
beforeAll(() => {
  const values: Record<string, unknown> = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator, IS_REACT_ACT_ENVIRONMENT: true };
  for (const [key, value] of Object.entries(values)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  dom.window.HTMLElement.prototype.scrollIntoView = function(this: HTMLElement, options?: boolean | ScrollIntoViewOptions) {
    scrolls.push({ element: this, options });
  };
});
afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  pageText = [];
  failure = undefined;
  documentGate = undefined;
  destroyed = 0;
  cleaned = 0;
  requests.length = 0;
  renderedPages.length = 0;
  scrolls.length = 0;
});
afterAll(() => {
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
  dom.window.close();
});

async function mountPdf(source: SourceDocument, resolutions: (QuoteResolution | null)[]): Promise<HTMLElement> {
  const root = dom.window.document.getElementById("pdf-root");
  if (!root) throw new Error("missing PDF test root");
  mounted = createRoot(root);
  await act(async () => {
    mounted?.render(<SourcePdf source={source} onResolve={(value) => {
      if (typeof value === "function") throw new Error("unexpected resolution updater");
      resolutions.push(value);
    }} />);
  });
  return root;
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 100 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  }
  if (!ready()) throw new Error("the PDF viewer did not reach the expected state");
}

const PDF: SourceDocument = {
  id: "evidence:pdf:1",
  title: "Notice",
  representation: "pdf",
  captureKind: "original",
  quote: "quoted",
  linkStatus: "stockbot_render",
  originalUrl: "https://www.sec.gov/Archives/edgar/data/1/notice.pdf",
};

describe("SourcePdf", () => {
  test("rejects captures that are not original PDFs", () => {
    for (const source of [{ ...PDF, captureKind: "provider_extraction" as const }, { ...PDF, representation: "html" as const }]) {
      expect(renderToStaticMarkup(<SourcePdf source={source} onResolve={() => undefined} />)).toContain("only renders original SEC captures");
    }
  });

  test("accepts original PDF captures with an empty page container before rendering", () => {
    const out = renderToStaticMarkup(<SourcePdf source={PDF} onResolve={() => undefined} />);
    expect(out).not.toContain("only renders original SEC captures");
    expect(out).not.toContain("no citation quote");
    expect(out).toContain('aria-label="PDF pages rendered"');
    expect(out).toContain('<div aria-label="Captured PDF pages"></div>');
  });

  test("shows scanned guidance when no quote exists", () => {
    const out = renderToStaticMarkup(<SourcePdf source={{ ...PDF, quote: "   " }} onResolve={() => undefined} />);
    expect(out).toContain("no citation quote");
  });

  test("renders all pages and highlights a quote across the page boundary", async () => {
    pageText = [["Before Rev", "enue"], [" grew", " after"]];
    const resolutions: (QuoteResolution | null)[] = [];
    const root = await mountPdf({ ...PDF, quote: "Revenue grew" }, resolutions);
    await until(() => resolutions.at(-1)?.status === "resolved");
    expect(resolutions).toEqual([null, { status: "resolved", start: 7, end: 19 }]);
    expect(renderedPages).toEqual([1, 2]);
    expect(root.querySelector<HTMLProgressElement>("progress")?.value).toBe(2);
    expect(root.querySelector<HTMLProgressElement>("progress")?.max).toBe(2);
    const pages = [...root.querySelectorAll<HTMLElement>(".source-pdf-page")];
    expect(pages.map((page) => page.querySelector(".textLayer")?.textContent)).toEqual(["Before Revenue", " grew after"]);
    expect([...root.querySelectorAll("mark")].map((mark) => mark.textContent)).toEqual(["Rev", "enue", " grew"]);
    expect(root.querySelectorAll(".mgl-active-element-highlight").length).toBe(3);
    expect(scrolls[0].element).toBe(pages[0]);
    expect(scrolls[0].options).toEqual({ behavior: "smooth", block: "start" });
    expect(scrolls[1].element).toBe(root.querySelector<HTMLElement>("mark")!);
    expect(pages[0].style.getPropertyValue("--total-scale-factor")).toBe("1.25");
    expect(pages[0].querySelector<HTMLCanvasElement>("canvas")?.width).toBe(101);
    expect(pages[0].querySelector<HTMLCanvasElement>("canvas")?.getAttribute("aria-hidden")).toBe("true");
    expect(requests[0]).toMatchObject({
      url: "/api/research/sources/evidence%3Apdf%3A1/pdf",
      cMapUrl: "/pdfjs/cmaps/",
      standardFontDataUrl: "/pdfjs/standard_fonts/",
      wasmUrl: "/pdfjs/wasm/",
      iccUrl: "/pdfjs/iccs/",
    });
    expect(workerOptions.workerSrc).toBe("https://viewer.example/pdfjs/pdf.worker.min.mjs");
    expect(cleaned).toBe(3);
  });

  test("keeps ambiguous and missing PDF passages unselected", async () => {
    for (const [quote, status] of [["risk", "ambiguous"], ["missing", "unresolved"]] as const) {
      pageText = [["risk"], ["risk"]];
      const resolutions: (QuoteResolution | null)[] = [];
      const root = await mountPdf({ ...PDF, quote }, resolutions);
      await until(() => resolutions.at(-1)?.status === status);
      expect(resolutions.at(-1)).toEqual(status === "ambiguous" ? { status, count: 2 } : { status });
      expect(root.querySelector("mark")).toBeNull();
      expect(root.querySelector('[role="alert"]')).toBeNull();
      await act(async () => mounted?.unmount());
      mounted = null;
    }
  });

  test("shows rendering failures without claiming a selected passage", async () => {
    failure = new Error("The captured PDF is damaged.");
    const resolutions: (QuoteResolution | null)[] = [];
    const root = await mountPdf(PDF, resolutions);
    await until(() => root.querySelector('[role="alert"]') !== null);
    expect(root.querySelector('[role="alert"]')?.textContent).toBe("The captured PDF is damaged.");
    expect(resolutions).toEqual([null, { status: "unresolved" }]);
    expect(root.querySelector("mark")).toBeNull();
    expect(renderedPages).toEqual([]);
  });

  test("destroys a pending document after the viewer closes", async () => {
    const gate = Promise.withResolvers<void>();
    documentGate = gate.promise;
    pageText = [["quoted"]];
    const resolutions: (QuoteResolution | null)[] = [];
    await mountPdf(PDF, resolutions);
    expect(requests.length).toBe(1);
    await act(async () => mounted?.unmount());
    mounted = null;
    await act(async () => gate.resolve());
    await until(() => destroyed === 2);
    expect(renderedPages).toEqual([]);
    expect(resolutions).toEqual([null]);
  });
});
