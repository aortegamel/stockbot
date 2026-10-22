import type { WindowLike } from "dompurify";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { fetchSourceDocument } from "../lib/research-client";
import type { SourceDocument } from "../lib/research-client";
import { activateQuote, highlightMappedQuote, mapSourceHtml, markQuote, originalLink, resolveQuote, sanitizeSourceHtml, sourceLabel, tagFragmentAnchors } from "../lib/source-viewer";
import type { QuoteResolution } from "../lib/source-viewer";

const SourcePdf = lazy(() => import("./source-pdf"));

const CAPTURE_LABELS: Record<SourceDocument["captureKind"], string> = {
  original: "Original document",
  provider_extraction: "Provider extraction — not the original document",
  structured_record: "Structured record — no official page",
  search_excerpt: "Search excerpt only",
};

function useSourceDocument(id: string, retry: number): { source: SourceDocument | null; error: string | null } {
  const [source, setSource] = useState<SourceDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchSourceDocument(id, controller.signal)
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
  }, [id, retry]);

  return { source, error };
}

function renderSourceText(root: HTMLDivElement, source: SourceDocument): void {
  if (source.representation === "html") {
    const fragment = sanitizeSourceHtml(window as unknown as WindowLike, source.content ?? "");
    tagFragmentAnchors(fragment);
    root.append(fragment);
  } else root.textContent = source.content ?? "";
}

const QUOTE_BLOCKS = "p, div, article, section, main, h1, h2, h3, h4, h5, h6, ul, ol, li, dl, dt, dd, blockquote, pre, br, hr, table, caption, tr";

// Exa stores raw HTML highlights. The viewer projects them through the same sanitizer to rendered characters.
// Block ends become line breaks for display. Matching ignores whitespace, so the breaks never change a match.
function renderedQuote(quote: string): string {
  const fragment = sanitizeSourceHtml(window as unknown as WindowLike, quote);
  for (const block of fragment.querySelectorAll(QUOTE_BLOCKS)) block.after("\n");
  return (fragment.textContent ?? "").replace(/\s*\n\s*/gu, "\n").trim();
}

type PaintedQuote = { resolution: QuoteResolution; quote: string };

function resolveRenderedQuote(text: string, current: SourceDocument, location: SourceDocument["location"]): PaintedQuote {
  let chosen: PaintedQuote = { resolution: resolveQuote(text, current.quote, location), quote: current.quote };
  // A literal match, even an ambiguous one, wins. Literal angle brackets never select a projected passage.
  if (chosen.resolution.status === "unresolved" && current.representation === "html" && (current.captureKind === "provider_extraction" || current.captureKind === "search_excerpt")) {
    const projected = renderedQuote(current.quote);
    const resolution = resolveQuote(text, projected, location);
    if (resolution.status !== "unresolved") chosen = { resolution, quote: projected };
  }
  return chosen;
}

function paintSource(root: HTMLDivElement, current: SourceDocument): PaintedQuote & { marks: HTMLElement[] } {
  root.replaceChildren();
  if (current.representation === "html" && current.matchingText !== undefined) {
    const map = mapSourceHtml(root, window as unknown as WindowLike, current.content ?? "");
    const highlighted = highlightMappedQuote(root, map, current);
    return { ...highlighted, quote: current.quote };
  }
  renderSourceText(root, current);
  const location = current.representation === "html" && current.location?.basis === "raw"
    ? { prefix: current.location.prefix, suffix: current.location.suffix }
    : current.location;
  const chosen = resolveRenderedQuote(root.textContent ?? "", current, location);
  const found = chosen.resolution;
  return { ...chosen, marks: found.status === "resolved" ? markQuote(root, found.start, found.end) : [] };
}

function useSourceHighlight(content: React.RefObject<HTMLDivElement | null>, source: SourceDocument | null): { painted: PaintedQuote | null; marks: React.RefObject<HTMLElement[]> } {
  const marks = useRef<HTMLElement[]>([]);
  const [painted, setPainted] = useState<PaintedQuote | null>(null);

  const paint = (root: HTMLDivElement, current: SourceDocument): PaintedQuote => {
    const { marks: found, resolution, quote } = paintSource(root, current);
    marks.current = found;
    if (found.length) activateQuote(root, found);
    return { resolution, quote };
  };

  useEffect(() => {
    const root = content.current;
    if (!source || !root || source.representation === "pdf") return;
    const frame = requestAnimationFrame(() => setPainted(paint(root, source)));
    return () => cancelAnimationFrame(frame);
  }, [content, source]);

  return { painted, marks };
}

const NOT_RECORDED = "Not recorded";

// This page is the Stockbot view. The one external link opens the official page; a missing official page shows none.
function SourceHeader({ source }: { source: SourceDocument }) {
  const original = originalLink(source);
  return (<>
    <h1 className="mt-1 text-base font-semibold">{sourceLabel(source) || source.title}</h1>
    <p className="mt-1 text-amber-200">{CAPTURE_LABELS[source.captureKind]}</p>
    {original && <p className="mt-1"><a className="text-blue-300 underline" href={original.href} target="_blank" rel="noopener noreferrer">
      {original.highlighted ? `Open on ${original.host} (highlighted) ↗` : `Original on ${original.host} ↗`}
    </a></p>}
    <h2 className="mt-3 text-sm font-semibold">How we got this</h2>
    <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-zinc-400">
      <dt>Source</dt><dd>{source.publisher || NOT_RECORDED}</dd>
      <dt>Published</dt><dd>{source.publishedAt || NOT_RECORDED}</dd>
      <dt>Tool</dt><dd className="font-mono">{source.toolName || NOT_RECORDED}</dd>
      <dt>Arguments</dt><dd>{source.toolArgs ? <pre className="whitespace-pre-wrap font-mono">{JSON.stringify(source.toolArgs, null, 2)}</pre> : NOT_RECORDED}</dd>
      <dt>Retrieved</dt><dd>{source.capturedAt || NOT_RECORDED}</dd>
      <dt>Known at</dt><dd>{source.knownAt || NOT_RECORDED}</dd>
      <dt>Content hash</dt><dd className="break-all font-mono">{source.contentHash || NOT_RECORDED}</dd>
    </dl>
  </>);
}

function SourceQuote({ quote, resolution }: { quote: string; resolution: QuoteResolution | null }) {
  return (<>
    {quote && <blockquote className="my-3 border-l-2 border-zinc-500 pl-3"><span className="text-xs text-zinc-400">Exact citation quote</span><p className="whitespace-pre-wrap">{quote}</p></blockquote>}
    {resolution && <p role="status" className="my-2 text-sm text-zinc-300">
      {resolution.status === "resolved" ? "The quote matches the captured source." : resolution.status === "ambiguous" ? `Ambiguous quote: ${resolution.count} matches remain. No passage is selected.` : "The quote does not resolve in the rendered source. No passage is selected."}
    </p>}
  </>);
}

function CapturedSource({ source, painted, pdfResolution, content, marks, onPdfResolve }: {
  source: SourceDocument;
  painted: PaintedQuote | null;
  pdfResolution: QuoteResolution | null;
  content: React.RefObject<HTMLDivElement | null>;
  marks: React.RefObject<HTMLElement[]>;
  onPdfResolve: React.Dispatch<React.SetStateAction<QuoteResolution | null>>;
}) {
  const status: QuoteResolution | null = source.representation === "pdf" ? pdfResolution : painted?.resolution ?? null;
  // The display uses the quote that the passage resolved against. PDF never paints here.
  const quote = painted?.quote ?? source.quote ?? "";
  return (<>
    <SourceHeader source={source} />
    <SourceQuote quote={quote} resolution={status} />
    {status?.status === "resolved" && source.representation !== "pdf" && <button type="button" className="source-control mb-2" onClick={() => {
      if (content.current) activateQuote(content.current, marks.current);
    }}>Go to quote</button>}
    {source.representation === "pdf" ? <Suspense fallback={<p role="status">Loading the PDF renderer…</p>}><SourcePdf source={source} onResolve={onPdfResolve} /></Suspense>
      : <div ref={content} aria-label="Captured source content" className={`max-h-[65vh] overflow-auto rounded bg-white p-4 text-sm text-zinc-900 ${source.representation === "html" ? "source-html" : "whitespace-pre-wrap font-mono"}`} />}
    <p className="mt-2 text-xs text-zinc-400">This page shows the immutable capture. The original link can show a newer document.</p>
  </>);
}

// Standalone provenance page for #/source/<id>: it reads the capture by id and needs no agent state.
export function SourcePage({ id }: { id: string }) {
  const content = useRef<HTMLDivElement>(null);
  const [retry, setRetry] = useState(0);
  const { source, error } = useSourceDocument(id, retry);
  const { painted, marks } = useSourceHighlight(content, source);
  const [pdfResolution, setPdfResolution] = useState<QuoteResolution | null>(null);

  return (
    <div className="page-scroll">
      <article aria-label="Source provenance" className="glass source-panel p-4">
        {/* `.glass > div` clears backgrounds; this wrapper keeps the white capture box intact. */}
        <div>
          {!source && !error && <p role="status" className="py-3">Loading captured source…</p>}
          {error && <div role="alert" className="py-3 text-red-300"><p>{error}</p><button type="button" className="source-control mt-2" onClick={() => setRetry((value) => value + 1)}>Retry source</button></div>}
          {source && <CapturedSource source={source} painted={painted} pdfResolution={pdfResolution} content={content} marks={marks} onPdfResolve={setPdfResolution} />}
        </div>
      </article>
    </div>
  );
}
