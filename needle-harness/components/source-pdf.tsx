"use client";

// Same-origin /pdfjs/* assets come from the copy-pdfjs build step. No CDN or upstream fetch.
import { getDocument, GlobalWorkerOptions, TextLayer } from "pdfjs-dist";
import type { PDFDocumentLoadingTask, PDFPageProxy } from "pdfjs-dist";
import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import type { SourceDocument } from "@/lib/research-client";
import { activateQuote, markQuote, resolveQuote } from "@/lib/source-viewer";
import type { QuoteResolution } from "@/lib/source-viewer";

type PageLayer = { layer: HTMLDivElement; start: number; element: HTMLDivElement };

async function renderPdfPage(container: HTMLDivElement, pageProxy: PDFPageProxy, start: number): Promise<PageLayer & { text: string }> {
  const viewport = pageProxy.getViewport({ scale: 1.25 });
  const element = window.document.createElement("div");
  element.className = "source-pdf-page";
  // PDF.js text layer CSS sizes spans and the layer from --total-scale-factor; the canvas uses the same box.
  element.style.setProperty("--total-scale-factor", String(viewport.scale));
  const canvas = window.document.createElement("canvas");
  canvas.width = Math.ceil(viewport.width);
  canvas.height = Math.ceil(viewport.height);
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.height = `${Math.floor(viewport.height)}px`;
  const layer = window.document.createElement("div");
  layer.className = "textLayer";
  element.append(canvas, layer);
  container.append(element);
  await pageProxy.render({ canvas, viewport }).promise;
  const content = await pageProxy.getTextContent();
  await new TextLayer({ container: layer, textContentSource: content, viewport }).render();
  const text = layer.textContent ?? "";
  pageProxy.cleanup();
  return { layer, start, element, text };
}

function markPdfQuote(container: HTMLDivElement, layers: PageLayer[], text: string, quote: string, location: SourceDocument["location"], onResolve: Dispatch<SetStateAction<QuoteResolution | null>>): void {
  const found = resolveQuote(text, quote, location);
  onResolve(found);
  if (found.status !== "resolved") return;
  const marked: HTMLElement[] = [];
  for (const { layer, start } of layers) {
    const layerLength = layer.textContent?.length ?? 0;
    if (found.end > start && found.start < start + layerLength) {
      marked.push(...markQuote(layer, Math.max(0, found.start - start), Math.min(start + layerLength, found.end) - start));
    }
  }
  if (!marked.length) {
    onResolve({ status: "unresolved" });
  } else {
    layers.findLast((entry) => entry.start <= found.start)?.element.scrollIntoView({ behavior: "smooth", block: "start" });
    activateQuote(container, marked);
  }
}


export default function SourcePdf({ source, onResolve }: {
  source: SourceDocument;
  onResolve: Dispatch<SetStateAction<QuoteResolution | null>>;
}) {
  const pages = useRef<HTMLDivElement>(null);
  const [count, setCount] = useState(0);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const invalid = source.representation !== "pdf" || source.captureKind !== "original";
  const id = source.id;
  const location = source.location;
  const quote = source.quote;

  useEffect(() => {
    if (invalid) return;
    const rendered: HTMLDivElement | null = pages.current;
    if (!rendered) return;
    const container: HTMLDivElement = rendered;
    let cancelled = false;
    let documentTask: PDFDocumentLoadingTask | undefined;
    container.replaceChildren();
    onResolve(null);

    async function render() {
      // Same-origin worker and packaged assets only. PDF bytes load from the same immutable capture as the JSON.
      GlobalWorkerOptions.workerSrc = new URL("/pdfjs/pdf.worker.min.mjs", window.location.origin).href;
      const task = getDocument({
        url: `/api/research/sources/${encodeURIComponent(id)}/pdf`,
        cMapUrl: "/pdfjs/cmaps/",
        cMapPacked: true,
        standardFontDataUrl: "/pdfjs/standard_fonts/",
        wasmUrl: "/pdfjs/wasm/",
        iccUrl: "/pdfjs/iccs/",
        stopAtErrors: false,
      });
      documentTask = task;
      const pdf = await task.promise;
      if (cancelled) { await documentTask?.destroy().catch(() => undefined); return; }
      setTotal(pdf.numPages);
      const layers: PageLayer[] = [];
      let text = "";
      for (let page = 1; page <= pdf.numPages; page++) {
        if (cancelled) { await documentTask?.destroy().catch(() => undefined); return; }
        const rendered = await renderPdfPage(container, await pdf.getPage(page), text.length);
        layers.push(rendered);
        text += rendered.text;
        if (cancelled) { await documentTask?.destroy().catch(() => undefined); return; }
        setCount(page);
      }
      if (cancelled) { await documentTask?.destroy().catch(() => undefined); return; }
      markPdfQuote(container, layers, text, quote, location, onResolve);
      await pdf.cleanup();
    }

    render().catch((failure: unknown) => {
      if (!cancelled) {
        setError(failure instanceof Error ? failure.message : "The PDF failed to render.");
        onResolve({ status: "unresolved" });
      }
    });
    return () => { cancelled = true; documentTask?.destroy().catch(() => undefined); };
  }, [id, invalid, location, onResolve, quote]);

  return (
    <div>
      {invalid && <p role="alert" className="py-3 text-red-300">This viewer only renders original SEC captures as PDFs.</p>}
      {error && <p role="alert" className="py-3 text-red-300">{error}</p>}
      <progress value={count} max={total} className="mb-2 w-full" aria-label="PDF pages rendered" />
      {!source.quote.trim() && <p role="status">This original PDF has no citation quote. Scanned pages show the page images without claimed text placement.</p>}
      <div ref={pages} aria-label="Captured PDF pages" />
    </div>
  );
}
