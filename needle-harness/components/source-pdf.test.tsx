import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import SourcePdf from "./source-pdf";
import type { SourceDocument } from "@/lib/research-client";

// Server rendering runs no effects, so these tests pin markup only.
// Under Bun, pdfjs-dist runs in Node mode: it reads relative PDF URLs from the filesystem and jsdom canvases have no 2D context.
// Real page rendering and text-layer highlights therefore need the browser smoke check.
// source-panel.test.tsx pins in jsdom that a non-original PDF never requests PDF bytes.
const PDF: SourceDocument = {
  id: "evidence:pdf:1",
  title: "Notice",
  representation: "pdf",
  captureKind: "original",
  quote: "quoted",
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
});
