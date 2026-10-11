"use client";

import type { WindowLike } from "dompurify";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import type { SourceReference } from "@/lib/agent/types";
import { fetchSourceDocument } from "@/lib/research-client";
import type { SourceDocument } from "@/lib/research-client";
import { activateQuote, highlightMappedQuote, mapSourceHtml, markQuote, originalSourceUrl, resolveQuote, sanitizeSourceHtml, tagFragmentAnchors } from "@/lib/source-viewer";
import type { QuoteResolution } from "@/lib/source-viewer";

const SourcePdf = lazy(() => import("./source-pdf"));

const CAPTURE_LABELS: Record<SourceDocument["captureKind"], string> = {
  original: "Original document",
  provider_extraction: "Provider extraction — not the original document",
  structured_record: "Structured record",
  search_excerpt: "Search excerpt only",
};

function useSourceDocument(reference: SourceReference, retry: number): { source: SourceDocument | null; error: string | null } {
  const [source, setSource] = useState<SourceDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchSourceDocument(reference.id, controller.signal)
      .then((loaded) => {
        if (!controller.signal.aborted) {
          setSource(loaded);
          setError(null);
        }
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : "The source request failed.");
      });
    return () => controller.abort();
  }, [reference.id, retry]);

  return { source, error };
}

function renderSourceText(root: HTMLDivElement, source: SourceDocument): void {
  if (source.representation === "html") {
    const fragment = sanitizeSourceHtml(window as unknown as WindowLike, source.content ?? "");
    tagFragmentAnchors(fragment);
    root.append(fragment);
  } else root.textContent = source.content ?? "";
}

function useSourceHighlight(content: React.RefObject<HTMLDivElement | null>, source: SourceDocument | null): { resolution: QuoteResolution | null; marks: React.RefObject<HTMLElement[]> } {
  const marks = useRef<HTMLElement[]>([]);
  const [resolution, setResolution] = useState<QuoteResolution | null>(null);

  const paint = (root: HTMLDivElement, current: SourceDocument): QuoteResolution => {
    root.replaceChildren();
    if (current.representation === "html" && current.matchingText !== undefined) {
      const map = mapSourceHtml(root, window as unknown as WindowLike, current.content ?? "");
      const highlighted = highlightMappedQuote(root, map, current);
      marks.current = highlighted.marks;
      if (marks.current.length) activateQuote(root, marks.current);
      return highlighted.resolution;
    }
    renderSourceText(root, current);
    const location = current.representation === "html" && current.location?.basis === "raw"
      ? { prefix: current.location.prefix, suffix: current.location.suffix }
      : current.location;
    const found = resolveQuote(root.textContent ?? "", current.quote, location);
    marks.current = found.status === "resolved" ? markQuote(root, found.start, found.end) : [];
    if (marks.current.length) activateQuote(root, marks.current);
    return found;
  };

  useEffect(() => {
    const root = content.current;
    if (!source || !root || source.representation === "pdf") return;
    const frame = requestAnimationFrame(() => setResolution(paint(root, source)));
    return () => cancelAnimationFrame(frame);
  }, [content, source]);

  return { resolution, marks };
}

function SourceHeader({ source }: { source: SourceDocument }) {
  return (<>
    <h3 className="mt-3 text-base font-semibold">{source.title}</h3>
    <p className="mt-1 text-amber-200">{CAPTURE_LABELS[source.captureKind]}</p>
    <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-zinc-400">
      <dt>Publisher</dt><dd>{source.publisher || "Not recorded"}</dd>
      <dt>Published</dt><dd>{source.publishedAt || "Not recorded"}</dd>
      <dt>Captured</dt><dd>{source.capturedAt || "Not recorded"}</dd>
      <dt>Capture hash</dt><dd className="break-all font-mono">{source.contentHash || "Not recorded"}</dd>
      <dt>Representation</dt><dd>{source.representation}</dd>
    </dl>
  </>);
}

function SourceQuote({ source, resolution }: { source: SourceDocument; resolution: QuoteResolution | null }) {
  return (<>
    {source.quote && <blockquote className="my-3 border-l-2 border-zinc-500 pl-3"><span className="text-xs text-zinc-400">Exact citation quote</span><p className="whitespace-pre-wrap">{source.quote}</p></blockquote>}
    {resolution && <p role="status" className="mb-2 text-sm text-zinc-300">
      {resolution.status === "resolved" ? "The quote matches the captured source." : resolution.status === "ambiguous" ? `Ambiguous quote: ${resolution.count} matches remain. No passage is selected.` : "The quote does not resolve in the rendered source. No passage is selected."}
    </p>}
  </>);
}


export function SourcePanel({ reference, position, count, onNavigate, onClose }: {
  reference: SourceReference;
  position: number;
  count: number;
  onNavigate: (direction: number) => void;
  onClose: () => void;
}) {
  const panel = useRef<HTMLElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const [retry, setRetry] = useState(0);
  const { source, error } = useSourceDocument(reference, retry);
  const { resolution, marks } = useSourceHighlight(content, source);
  const [pdfResolution, setPdfResolution] = useState<QuoteResolution | null>(null);
  const status = source?.representation === "pdf" ? pdfResolution : resolution;

  useEffect(() => {
    panel.current?.focus();
  }, [reference.id, retry]);

  const original = source ? originalSourceUrl(source, status?.status === "resolved") : undefined;
  return (
    <aside ref={panel} tabIndex={-1} aria-label="Captured source" className="source-panel mt-4 rounded border border-zinc-600 bg-zinc-950 p-3 outline-none focus-visible:ring-2 focus-visible:ring-blue-400" onKeyDown={(event) => {
      if (event.key === "Escape") { event.preventDefault(); onClose(); }
    }}>
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-semibold">Captured source</h2>
        <span className="text-zinc-400">{position + 1} of {count}</span>
        <button type="button" disabled={count < 2} onClick={() => onNavigate(-1)} className="source-control">Previous source</button>
        <button type="button" disabled={count < 2} onClick={() => onNavigate(1)} className="source-control">Next source</button>
        <button type="button" onClick={onClose} className="source-control ml-auto">Close source</button>
      </div>
      {!source && !error && <p role="status" className="py-3">Loading captured source…</p>}
      {error && <div role="alert" className="py-3 text-red-300"><p>{error}</p><button type="button" className="source-control mt-2" onClick={() => setRetry((value) => value + 1)}>Retry source</button></div>}
      {source && <>
        <SourceHeader source={source} />
        {original && <a className="mt-2 inline-block text-blue-300 underline" href={original} target="_blank" rel="noopener noreferrer">Open original</a>}
        <SourceQuote source={source} resolution={status} />
        {status?.status === "resolved" && source.representation !== "pdf" && <button type="button" className="source-control mb-2" onClick={() => {
          if (content.current) activateQuote(content.current, marks.current);
        }}>Go to quote</button>}
        {source.representation === "pdf" ? <Suspense fallback={<p role="status">Loading the PDF renderer…</p>}><SourcePdf source={source} onResolve={setPdfResolution} /></Suspense> : <div ref={content} aria-label="Captured source content" className={`max-h-[65vh] overflow-auto rounded bg-white p-4 text-sm text-zinc-900 ${source.representation === "html" ? "source-html" : "whitespace-pre-wrap font-mono"}`} />}
        <p className="mt-2 text-xs text-zinc-400">This pane shows the immutable capture. The original link can show a newer document.</p>
      </>}
    </aside>
  );
}
