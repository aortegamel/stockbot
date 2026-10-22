import { afterEach, describe, expect, test } from "bun:test";
import { fetchJournal, fetchSnapshot, fetchSourceDocument, fetchSourceLabel, listSessions } from "./research-client";
import type { SourceAudit, SourceDocument, SourceLabel } from "./research-client";

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
    linkStatus: "missing",
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

  test.each(["SOURCE_MISMATCH", "SOURCE_AMBIGUOUS"])("preserves %s separately from the API audit and existing source fields", async (failure) => {
    const source = document({
      captureKind: "structured_record", quote: "", content: '{"shortVolume":100}', publisher: "FINRA",
      toolName: "get_reg_sho_volume", toolResultId: "result:1", toolArgs: { symbol: "NVDA" }, dataset: "regSho",
      knownAt: "2026-10-21", contentHash: "sha256:api",
      sourceConflict: { failure, reason: "The published file disagrees with the API record." },
      audit: { evidenceId: "EV-1", claimType: "observed_fact", verification: "verified", verificationReason: "", value: "100" },
    });
    respond(source);
    expect(await fetchSourceDocument(id)).toStrictEqual(source);
  });

  test("rejects malformed source conflicts and conflicts with an available public link", async () => {
    for (const sourceConflict of [null, [], "conflict", {}, { failure: 1, reason: "mismatch" }, { failure: "SOURCE_MISMATCH", reason: null }]) {
      respond({ ...document(), sourceConflict });
      await expect(fetchSourceDocument(id)).rejects.toThrow("The source conflict is invalid.");
    }
    respond(document({
      linkStatus: "stockbot_render", originalUrl: "https://www.finra.org/file.txt",
      sourceConflict: { failure: "SOURCE_MISMATCH", reason: "The published file disagrees." },
    }));
    await expect(fetchSourceDocument(id)).rejects.toThrow("The source links are invalid.");
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
      document({ linkStatus: "stockbot_render", originalUrl: "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260102.txt" }),
      document({ linkStatus: "highlighted_original", originalUrl: "https://www.sec.gov/a.htm", highlightUrl: "https://www.sec.gov/a.htm#f-30:~:text=Revenue,81%2C615" }),
      document({ representation: "text", captureKind: "search_excerpt", ...metadata }),
      document({ representation: "text", captureKind: "structured_record", content: '{\n  "symbolCode": "ACME"\n}', quote: "" }),
      { id, title: "10-K", representation: "pdf", captureKind: "original", quote: "Revenue grew.", linkStatus: "missing", pdfUrl: "/files/10-k.pdf", ...metadata },
    ] satisfies SourceDocument[]) {
      respond(body);
      expect(await fetchSourceDocument(id)).toStrictEqual(body);
    }
  });

  test("keeps how-we-got-it fields and plain-object tool arguments; old sessions omit them", async () => {
    const how = {
      toolName: "get_short_interest",
      toolResultId: "rs:1:tr:7",
      dataset: "otcMarket/consolidatedShortInterest",
      knownAt: "2026-01-02",
      toolArgs: { ticker: "ACME", filters: { limit: 5 } },
    };
    const body = document(how);
    respond(body);
    expect(await fetchSourceDocument(id)).toStrictEqual(body);
    respond(document());
    const old = await fetchSourceDocument(id);
    for (const key of Object.keys(how)) expect(key in old).toBe(false);
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
      // JSON rows are never a representation; structured records are plain text.
      { ...document(), representation: "json" },
      { ...document(), representation: "pdf", captureKind: "search_excerpt" },
      ...["url", "publisher", "publishedAt", "capturedAt", "contentHash", "content", "matchingText", "pdfUrl", "toolName", "toolResultId", "dataset", "knownAt", "originalUrl", "highlightUrl"].map((key) => ({
        ...document(),
        [key]: 42,
      })),
      ...[null, [], "query=ACME", 5].map((toolArgs) => ({ ...document(), toolArgs })),
    ]) {
      respond(body);
      await expect(fetchSourceDocument(id)).rejects.toThrow();
    }
  });

  test("rejects link fields that do not match their status or are not safe web URLs", async () => {
    const page = "https://www.sec.gov/a.htm";
    const highlight = `${page}#:~:text=Revenue`;
    for (const links of [
      { linkStatus: undefined },
      { linkStatus: "public_page" },
      { linkStatus: "stockbot_render", originalUrl: page, captureKind: "structured_record" },
      { linkStatus: "highlighted_original", originalUrl: page, highlightUrl: highlight, captureKind: "structured_record" },
      { linkStatus: "missing", originalUrl: page },
      { linkStatus: "missing", highlightUrl: highlight },
      { linkStatus: "stockbot_render" },
      { linkStatus: "stockbot_render", originalUrl: page, highlightUrl: highlight },
      { linkStatus: "stockbot_render", originalUrl: highlight },
      { linkStatus: "highlighted_original", originalUrl: page },
      { linkStatus: "highlighted_original", highlightUrl: highlight },
      ...["javascript:alert(1)", "https://user:secret@www.sec.gov/a.htm", "file:///etc/passwd", "not a url"].flatMap((unsafe) => [
        { linkStatus: "stockbot_render", originalUrl: unsafe },
        { linkStatus: "highlighted_original", originalUrl: page, highlightUrl: unsafe },
      ]),
    ]) {
      respond({ ...document(), ...links });
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
      [{ ...document(), linkStatus: "stockbot_render", content: undefined, location: null }, "The source links are invalid."],
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

describe("fetchSourceLabel", () => {
  const evidenceId = "evidence:rs:1:ev:7";
  const label = (overrides: Partial<SourceLabel> = {}): SourceLabel => ({ id: evidenceId, title: "10-K", linkStatus: "missing", ...overrides });
  const requested: { url: string; init?: RequestInit }[] = [];
  function reply(body: unknown, status = 200): void {
    (globalThis as { fetch?: typeof fetch }).fetch = ((input: string | URL | Request, init?: RequestInit) => {
      requested.push({ url: String(input), init });
      return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
    }) as unknown as typeof fetch;
  }

  afterEach(() => {
    requested.length = 0;
  });

  test("reads the label endpoint, never the full source endpoint", async () => {
    reply(label());
    expect(await fetchSourceLabel(evidenceId)).toStrictEqual(label());
    expect(requested.map((request) => [request.url, request.init?.cache])).toEqual([["/api/research/sources/evidence%3Ars%3A1%3Aev%3A7/label", "no-store"]]);
  });

  test("keeps exactly the persisted label fields and drops everything the viewer resolves", async () => {
    const full = label({ publisher: "SEC", publishedAt: "2026-01-01", knownAt: "2026-01-02", linkStatus: "stockbot_render", originalUrl: "https://www.sec.gov/a.htm" });
    reply({ ...full, content: "body", quote: "q", audit: { evidenceId: "x" }, highlightUrl: "https://www.sec.gov/a.htm#:~:text=q", toolArgs: { q: 1 } });
    expect(await fetchSourceLabel(evidenceId)).toStrictEqual(full);
  });

  test("rejects labels that claim a highlight, mismatch their links, or name another source", async () => {
    const page = "https://www.sec.gov/a.htm";
    for (const body of [
      label({ id: "evidence:other" }),
      { ...label(), title: 5 },
      { ...label(), linkStatus: "highlighted_original", originalUrl: page },
      { ...label(), linkStatus: undefined },
      label({ originalUrl: page }),
      label({ linkStatus: "stockbot_render" }),
      label({ linkStatus: "stockbot_render", originalUrl: `${page}#:~:text=Revenue` }),
      label({ linkStatus: "stockbot_render", originalUrl: "javascript:alert(1)" }),
      label({ linkStatus: "stockbot_render", originalUrl: "https://user:pw@www.sec.gov/a.htm" }),
      ...["publisher", "publishedAt", "knownAt"].map((key) => ({ ...label(), [key]: 7 })),
      null,
      [],
    ]) {
      reply(body);
      await expect(fetchSourceLabel(evidenceId)).rejects.toThrow();
    }
  });

  test("surfaces the controlled server error, then the status", async () => {
    reply({ id: evidenceId, error: "source capture not found", status: 404, failure: "SOURCE_IDENTITY_MISSING" }, 404);
    await expect(fetchSourceLabel(evidenceId)).rejects.toThrow("source capture not found");
    reply({}, 502);
    await expect(fetchSourceLabel(evidenceId)).rejects.toThrow("Source request failed (502).");
  });
});

describe("fetchSourceDocument audit", () => {
  const node = (overrides: Partial<SourceAudit> = {}): SourceAudit => ({ evidenceId: "EV-1", claimType: "observed_fact", verification: "verified", verificationReason: "", ...overrides });
  const calculation = (audit: unknown): Record<string, unknown> => ({ ...document({ captureKind: "calculation", content: "a - b", quote: "" }), audit });

  test("keeps a nested calculation audit with each input's own source id and no API link", async () => {
    const audit = node({
      evidenceId: "EV-C", sourceId: "evidence:EV-C", claimType: "calculation", value: "3481", entity: "NVDA", period: "Q1", unit: "USD millions", knownAt: "2025-05-01",
      formula: "a - b", formulaId: "difference", formulaVersion: "1", recomputedValue: "3481",
      inputs: [
        node({ evidenceId: "EV-A", sourceId: "evidence:EV-A", value: "26,044", source: { name: "SEC", document: "x.htm", accession: "0001045810-24-000124", hash: "sha256:abc" } }),
        node({ evidenceId: "EV-B", verification: "unverified", verificationReason: "absent" }),
      ],
    });
    respond(calculation(audit));
    expect((await fetchSourceDocument(id)).audit).toStrictEqual(audit);
  });

  test("preserves canonical descriptors on verified observations and calculations", async () => {
    const observed = {
      id: "dp:observation", metric: "revenue", field: "Revenue", value: "100", entity: "NVDA", period: "2026",
      unit: "USD", statement: "NVDA revenue for 2026 was 100 USD.",
    };
    const derived = {
      ...observed, id: "dp:calculation", metric: "revenue_change", field: "formula", formula: "difference@1",
      statement: "revenue_change = difference@1(revenue=100, revenue=90) = 10", value: "10",
      inputs: ["dp:observation", "dp:prior"],
    };
    const audit = node({
      claimType: "calculation", formula: "a - b", value: "10", datapoint: derived,
      inputs: [node({ value: "100", datapoint: observed })],
    });
    respond(calculation(audit));
    expect((await fetchSourceDocument(id)).audit).toStrictEqual(audit);
  });

  test("does not expose canonical datapoints for unverified or conflicted audits", async () => {
    for (const verification of ["unverified", "conflicted"] as const) {
      const audit = node({ verification, value: "100" });
      respond(calculation({ ...audit, datapoint: { id: "dp:untrusted" } }));
      expect((await fetchSourceDocument(id)).audit).toStrictEqual(audit);
    }
  });

  test("rejects malformed canonical descriptors on verified audits", async () => {
    const datapoint = { id: "dp:observation", metric: "revenue", value: "100", entity: "NVDA", period: "2026", statement: "Revenue was 100." };
    for (const invalid of [null, [], {}, { ...datapoint, id: "EV-1" }, { ...datapoint, metric: 1 }, { ...datapoint, field: 1 }, { ...datapoint, formula: 1 }]) {
      respond(calculation({ ...node(), datapoint: invalid }));
      await expect(fetchSourceDocument(id)).rejects.toThrow("The source audit is invalid.");
    }
    respond(calculation({ ...node(), datapoint: { ...datapoint, unit: null } }));
    expect((await fetchSourceDocument(id)).audit?.datapoint).toStrictEqual(datapoint);
  });

  test("keeps repair history as typed display text, never a citation link", async () => {
    const audit = node({
      verification: "unverified",
      verificationReason: "no proof",
      repairs: [{ repair_id: "repair:abc", status: "verified", category: "eps_without_accession", datapoint: "eps_diluted", value: "0.76" }],
    });
    respond(calculation(audit));
    expect((await fetchSourceDocument(id)).audit).toStrictEqual(audit);
    for (const repairs of [[{ status: "verified" }], [{ repair_id: "" }], [{ repair_id: "repair:abc", status: 1 }], "repairs"]) {
      respond(calculation(node({ repairs: repairs as never })));
      await expect(fetchSourceDocument(id)).rejects.toThrow("The source audit is invalid.");
    }
  });
  test("drops unknown audit and audit-source keys", async () => {
    respond(calculation({ ...node({ source: { name: "SEC" } }), upstreamUrl: "https://data.sec.gov/x.json", source: { name: "SEC", url: "https://data.sec.gov/x.json" } }));
    expect((await fetchSourceDocument(id)).audit).toStrictEqual(node({ source: { name: "SEC" } }));
  });

  test("rejects malformed audits, links that are not server source ids, and calculations with an original link", async () => {
    const bad: unknown[] = [
      null, [], "audit",
      { ...node(), evidenceId: undefined },
      { ...node(), claimType: 1 },
      { ...node(), verification: "passed" },
      { ...node(), verificationReason: undefined },
      node({ sourceId: "https://data.sec.gov/x.json" }),
      node({ sourceId: "evidence:../x" }),
      { ...node(), value: 3481 },
      { ...node(), source: "SEC" },
      { ...node(), source: { hash: 1 } },
      { ...node(), inputs: {} },
      { ...node(), inputs: [{ ...node(), verification: "ok" }] },
    ];
    for (const audit of bad) {
      respond(calculation(audit));
      await expect(fetchSourceDocument(id)).rejects.toThrow("The source audit is invalid.");
    }
    respond({ ...calculation(node()), linkStatus: "stockbot_render", originalUrl: "https://www.sec.gov/a.htm" });
    await expect(fetchSourceDocument(id)).rejects.toThrow("The source links are invalid.");
  });

  test("bounds audit depth at the backend proof limit", async () => {
    const chain = (depth: number): SourceAudit => (depth === 0 ? node() : node({ claimType: "calculation", inputs: [chain(depth - 1)] }));
    respond(calculation(chain(16)));
    expect((await fetchSourceDocument(id)).audit).toStrictEqual(chain(16));
    respond(calculation(chain(17)));
    await expect(fetchSourceDocument(id)).rejects.toThrow("The source audit is invalid.");
  });

  test("a controlled viewer error surfaces only its public message", async () => {
    respond({ id, error: "source read failed", status: 502, failure: "SOURCE_UNAVAILABLE" }, 502);
    // Exactly the public message: the internal failure class never reaches the reader.
    await expect(fetchSourceDocument(id)).rejects.toThrow(/^source read failed$/u);
  });
});
