import { afterEach, describe, expect, test } from "bun:test";
import { fetchJournal, fetchSnapshot, fetchSourceDocument, listSessions } from "./research-client";
import type { SourceDocument } from "./research-client";

const originalFetch = globalThis.fetch;
const id = `capture:${"b".repeat(64)}`;
function document(overrides: Partial<SourceDocument> = {}): SourceDocument {
  return {
    id,
    title: "10-K",
    representation: "text",
    captureKind: "original",
    content: "Revenue grew.",
    quote: "Revenue grew.",
    ...overrides,
  };
}

function respond(body: unknown, status = 200): void {
  const fetchState = globalThis as { fetch?: typeof fetch };
  const mocked = () =>
    Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
  fetchState.fetch = mocked as unknown as typeof fetch;
}

afterEach(() => {
  (globalThis as { fetch?: typeof fetch }).fetch = originalFetch;
});

describe("fetchSourceDocument", () => {
  test("accepts a captured document and preserves the server id", async () => {
    respond(document({ contentHash: "abc" }));
    expect(await fetchSourceDocument(id)).toMatchObject({ id, title: "10-K", contentHash: "abc" });
  });

  test("preserves the optional SEC matching basis without changing the original content", async () => {
    const source = document({ representation: "html", content: "<table><tr><td>Revenue</td><td>10</td></tr></table>", matchingText: "Revenue | 10", quote: "Revenue | 10" });
    respond(source);
    expect(await fetchSourceDocument(id)).toStrictEqual(source);
    respond(document({ matchingText: "" }));
    expect((await fetchSourceDocument(id)).matchingText).toBe("");
  });

  test("rejects mismatched ids, wrong shapes, and non-original PDFs", async () => {
    for (const body of [
      document({ id: "other" }),
      { ...document(), title: 42 },
      { ...document(), representation: "pdf", captureKind: "provider_extraction" },
      { ...document(), representation: "html", content: undefined },
      { ...document(), location: { offset: -1 } },
    ]) {
      respond(body);
      await expect(fetchSourceDocument(id)).rejects.toThrow();
    }
  });

  test("surfaces server errors without inventing a document", async () => {
    respond({ error: "source capture not found" }, 404);
    await expect(fetchSourceDocument(id)).rejects.toThrow("source capture not found");
  });

  test("returns exactly the validated fields for every representation and capture kind", async () => {
    const metadata = {
      url: "https://example.com/10-k",
      publisher: "SEC",
      publishedAt: "2026-01-02",
      capturedAt: "2026-01-03T00:00:00Z",
      contentHash: "abc",
    };
    for (const body of [
      document({ representation: "html", captureKind: "provider_extraction" }),
      document({ representation: "json", captureKind: "structured_record" }),
      document({ representation: "text", captureKind: "search_excerpt", ...metadata }),
      { id, title: "10-K", representation: "pdf", captureKind: "original", quote: "Revenue grew.", pdfUrl: "/files/10-k.pdf", ...metadata },
    ] satisfies SourceDocument[]) {
      respond(body);
      expect(await fetchSourceDocument(id)).toStrictEqual(body);
    }
  });

  test("preserves zero offsets, empty context, and partial locations", async () => {
    for (const location of [
      { offset: 0, end: 0, basis: "content", prefix: "", suffix: "" },
      { offset: 0 },
      { end: 13 },
      { basis: "content" },
      { prefix: "Before ", suffix: " after" },
      {},
    ]) {
      const body = document({ location });
      respond(body);
      expect(await fetchSourceDocument(id)).toStrictEqual(body);
    }
  });

  test("rejects malformed source locations", async () => {
    for (const location of [
      null,
      [],
      "0",
      5,
      { offset: -1 },
      { end: -1 },
      { offset: 1.5 },
      { end: "3" },
      { offset: 2 ** 53 },
      { basis: 5 },
      { prefix: null },
      { suffix: {} },
    ]) {
      respond({ ...document(), location });
      await expect(fetchSourceDocument(id)).rejects.toThrow();
    }
  });

  test("rejects malformed source shapes and metadata types", async () => {
    for (const body of [
      null,
      [],
      "source",
      { ...document(), id: undefined },
      { ...document(), quote: undefined },
      { ...document(), quote: 42 },
      { ...document(), representation: "markdown" },
      { ...document(), captureKind: "summary" },
      { ...document(), representation: "pdf", captureKind: "structured_record" },
      { ...document(), representation: "pdf", captureKind: "search_excerpt" },
      ...["url", "publisher", "publishedAt", "capturedAt", "contentHash", "content", "matchingText", "pdfUrl"].map((key) => ({
        ...document(),
        [key]: 42,
      })),
    ]) {
      respond(body);
      await expect(fetchSourceDocument(id)).rejects.toThrow();
    }
  });

  test("reports the HTTP status when a server error has no message", async () => {
    respond({ error: 42 }, 503);
    await expect(fetchSourceDocument(id)).rejects.toThrow("503");
  });

  test("reports validation failures in a fixed order with their own messages", async () => {
    const cases: [unknown, string][] = [
      [{ ...document(), id: "other", url: 42, location: null }, "The source response is invalid."],
      [{ ...document(), url: 42, location: null }, "The source metadata is invalid."],
      [{ ...document(), content: undefined, location: null }, "The captured source content is missing."],
      [{ ...document(), location: null }, "The source location is invalid."],
      [{ ...document(), representation: "pdf", captureKind: "provider_extraction", content: undefined, location: null }, "The PDF does not identify an original capture."],
    ];
    for (const [body, message] of cases) {
      respond(body);
      await expect(fetchSourceDocument(id)).rejects.toThrow(message);
    }
    respond({ error: 42 }, 503);
    await expect(fetchSourceDocument(id)).rejects.toThrow("Source request failed (503).");
  });
});

describe("research reads", () => {
  const urls: string[] = [];
  function capture(body: unknown, status = 200): void {
    const fetchState = globalThis as { fetch?: typeof fetch };
    fetchState.fetch = ((input: string | URL | Request) => {
      urls.push(String(input));
      return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
    }) as unknown as typeof fetch;
  }

  afterEach(() => {
    urls.length = 0;
  });

  test("reads a snapshot, a journal tail, and a bounded session list from their own endpoints", async () => {
    capture({ session: { session_id: "rs:1" }, jobs: [] });
    expect(await fetchSnapshot("rs:1")).toEqual({ session: { session_id: "rs:1" }, jobs: [] });
    capture({ session_id: "rs:1", after_seq: 7, events: [{ sequence: 8, event_type: "source_agent" }] });
    expect(await fetchJournal("rs:1", 7)).toEqual({ session_id: "rs:1", after_seq: 7, events: [{ sequence: 8, event_type: "source_agent" }] });
    capture({ sessions: [{ session_id: "rs:1" }] });
    expect(await listSessions(5)).toEqual({ sessions: [{ session_id: "rs:1" }] });
    capture({ sessions: [] });
    expect(await listSessions()).toEqual({ sessions: [] });
    expect(urls).toEqual([
      "/api/research/rs%3A1",
      "/api/research/rs%3A1/journal?after_seq=7",
      "/api/research/sessions?limit=5",
      "/api/research/sessions?limit=20",
    ]);
  });

  test("surfaces the server message for a failed read before the generic status text", async () => {
    capture({ error: "unknown session: 'rs:9'" }, 404);
    await expect(fetchSnapshot("rs:9")).rejects.toThrow("unknown session: 'rs:9'");
    capture({ after_seq: 0, error: 42 }, 500);
    await expect(fetchJournal("rs:9", 0)).rejects.toThrow("journal 500");
    capture({}, 503);
    await expect(listSessions(3)).rejects.toThrow("sessions 503");
    capture({}, 502);
    await expect(fetchSnapshot("rs:9")).rejects.toThrow("snapshot 502");
  });
});
