import { beforeEach, describe, expect, mock, test } from "bun:test";
import type { SpawnSyncReturns } from "node:child_process";

let result: SpawnSyncReturns<string> = {
  pid: 1,
  output: [],
  stdout: JSON.stringify({ ok: true, data: {} }),
  stderr: "",
  status: 0,
  signal: null,
};
const calls: unknown[][] = [];
mock.module("node:child_process", () => ({
  spawnSync: (...args: unknown[]) => {
    calls.push(args);
    return result;
  },
}));
// Dynamic import: the route binds spawnSync at load time, so the mock installs first (module-loading boundary test).
const { GET } = await import("./route");
beforeEach(() => {
  calls.length = 0;
  result = { pid: 1, output: [], stdout: JSON.stringify({ ok: true, data: {} }), stderr: "", status: 0, signal: null };
});

const captureId = `capture:${"a".repeat(64)}`;
function sourcePath(id: string, suffix = ""): string {
  return `/api/research/sources/${encodeURIComponent(id)}${suffix}`;
}
function read(path: string): Promise<Response> {
  return GET(new Request(`http://localhost${path}`));
}
function payload(): unknown {
  const args = calls[0]?.[1];
  if (!Array.isArray(args)) throw new Error("The bridge did not receive arguments.");
  return JSON.parse(args[2] as string);
}
function expectSourceHeaders(response: Response): void {
  expect(response.headers.get("Cache-Control")).toBe("no-store");
  expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
}

describe("research source route", () => {
  test("returns the stored source document with bounded bridge output", async () => {
    const document = { source_id: captureId, title: "Filing", blocks: [{ text: "Revenue increased." }] };
    result.stdout = JSON.stringify({ ok: true, data: document });
    const response = await read(sourcePath(captureId));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(document);
    expectSourceHeaders(response);
    expect(payload()).toEqual({ kind: "source", source_id: captureId });
    expect(calls[0]?.[2]).toMatchObject({ encoding: "utf8", timeout: 15_000, maxBuffer: 32 * 1024 * 1024 });
    const args = calls[0]?.[1];
    expect(Array.isArray(args) && args[1]).toContain("read_source_document(req[\"source_id\"])");
  });

  test("returns only the PDF bytes from the bridge", async () => {
    const pdf = Buffer.from("JVBERg==", "base64");
    result.stdout = JSON.stringify({ ok: true, data: { pdf_base64: pdf.toString("base64") } });
    const response = await read(sourcePath(captureId, "/pdf"));
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe("application/pdf");
    expectSourceHeaders(response);
    expect(Buffer.from(await response.arrayBuffer())).toEqual(pdf);
    expect(payload()).toEqual({ kind: "source_pdf", source_id: captureId });
    const args = calls[0]?.[1];
    expect(Array.isArray(args) && args[1]).toContain("base64.b64encode(read_source_pdf(req[\"source_id\"]))");
  });

  test("accepts encoded evidence IDs at the length boundary", async () => {
    for (const id of ["evidence:a:b_c-1", `evidence:${"a".repeat(256)}`]) {
      calls.length = 0;
      const response = await read(sourcePath(id));
      expect(response.status).toBe(200);
      expect(payload()).toEqual({ kind: "source", source_id: id });
    }
  });

  test("rejects URLs, paths, invalid IDs, and invalid encoding without the bridge", async () => {
    const invalidIds = [
      "https://example.com/file.pdf",
      "/etc/passwd",
      "../capture",
      "evidence:../../secret",
      "capture:not-a-hash",
      `capture:${"A".repeat(64)}`,
      `capture:${"a".repeat(63)}`,
      `capture:${"a".repeat(65)}`,
      "evidence:",
      `evidence:${"a".repeat(257)}`,
      "evidence:with space",
      "evidence:a\n",
      "evidence:a\r",
      `${captureId}\n`,
    ];
    for (const path of [...invalidIds.map((id) => sourcePath(id)), "/api/research/sources/%ZZ"]) {
      const response = await read(path);
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "bad source_id" });
      expectSourceHeaders(response);
    }
    expect(calls.length).toBe(0);
  });

  test("rejects missing and extra path segments without the bridge", async () => {
    for (const path of [
      "/api/research/sources",
      "/api/research/sources/",
      sourcePath(captureId, "/"),
      sourcePath(captureId, "/raw"),
      sourcePath(captureId, "/pdf/extra"),
      sourcePath(captureId, "/pdf/"),
      `/api/research/sources//${captureId}`,
    ]) {
      const response = await read(path);
      expect(response.ok).toBe(false);
      expect(typeof (await response.json()).error).toBe("string");
      expectSourceHeaders(response);
    }
    expect(calls.length).toBe(0);
  });

  test("returns safe errors for missing, corrupt, and unreadable captures", async () => {
    for (const [status, error] of [
      [404, "source capture not found"],
      [409, "invalid source capture"],
      [502, "source read failed"],
    ] as const) {
      result.status = 1;
      result.stdout = JSON.stringify({ ok: false, status, error: "Traceback: /private/data/capture.pdf" });
      for (const suffix of ["", "/pdf"]) {
        const response = await read(sourcePath(captureId, suffix));
        expect(response.status).toBe(status);
        expect(await response.json()).toEqual({ error });
        expectSourceHeaders(response);
      }
    }
  });

  test("hides process errors, tracebacks, and unrecognized bridge failures", async () => {
    const failures: Partial<SpawnSyncReturns<string>>[] = [
      { error: new Error("spawn /private/venv/bin/python ENOENT") },
      { status: 1, stdout: "", stderr: "Traceback: /private/data/capture.pdf" },
      { status: 1, stdout: JSON.stringify({ error: "missing /private/data/capture.pdf" }) },
      { status: 1, stdout: JSON.stringify({ status: 200, error: "Traceback" }) },
    ];
    for (const failure of failures) {
      result = { pid: 1, output: [], stdout: "", stderr: "", status: 0, signal: null, ...failure };
      const response = await read(sourcePath(captureId));
      expect(response.status).toBe(502);
      expect(await response.json()).toEqual({ error: "source read failed" });
      expectSourceHeaders(response);
    }
  });

  test("rejects malformed PDF bridge responses", async () => {
    for (const data of [null, {}, { pdf_base64: 42 }, { pdf_base64: "" }, { pdf_base64: "not base64!" }, { pdf_base64: "JVBERg==\n" }]) {
      result.stdout = JSON.stringify({ ok: true, data });
      const response = await read(sourcePath(captureId, "/pdf"));
      expect(response.status).toBe(502);
      expect(await response.json()).toEqual({ error: "invalid source PDF response" });
      expectSourceHeaders(response);
    }
  });

  test("rejects malformed bridge JSON and missing data", async () => {
    for (const stdout of ["Traceback: /private/data", JSON.stringify({ ok: true })]) {
      result.stdout = stdout;
      const response = await read(sourcePath(captureId));
      expect(response.status).toBe(502);
      expect(await response.json()).toEqual({ error: "source read failed" });
      expectSourceHeaders(response);
    }
  });
});

describe("existing research routes", () => {
  test("preserves session listing, snapshots, and journal cursors", async () => {
    const body = { marker: "unchanged" };
    result.stdout = JSON.stringify({ ok: true, data: body });
    for (const [path, expected] of [
      ["/api/research/sessions", { kind: "sessions", limit: 20 }],
      ["/api/research/sessions?limit=999", { kind: "sessions", limit: 100 }],
      ["/api/research/sessions?limit=invalid", { kind: "sessions", limit: 20 }],
      ["/api/research/session_1", { kind: "snapshot", session_id: "session_1" }],
      ["/api/research/session_1/journal", { kind: "journal", session_id: "session_1", after_seq: 0 }],
      ["/api/research/session_1/journal?after_seq=5", { kind: "journal", session_id: "session_1", after_seq: 5 }],
      ["/api/research/session_1/journal?after=-3", { kind: "journal", session_id: "session_1", after_seq: 0 }],
    ] as const) {
      calls.length = 0;
      const response = await read(path);
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual(body);
      expect(response.headers.get("Cache-Control")).toBe("no-store");
      expect(payload()).toEqual(expected);
    }
  });

  test("preserves unknown session errors", async () => {
    result.status = 1;
    result.stdout = JSON.stringify({ ok: false, error: "unknown session: missing" });
    const response = await read("/api/research/missing");
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: "unknown session: missing" });
  });
  test("maps non-source bridge failures without leaking process output", async () => {
    const cases: { failure: Partial<SpawnSyncReturns<string>>; status: number; error: string }[] = [
      { failure: { error: new Error("spawn /private/venv/bin/python ENOENT") }, status: 502, error: "bridge spawn failed: spawn /private/venv/bin/python ENOENT" },
      { failure: { status: 1, stdout: JSON.stringify({ ok: false, error: "unknown session: missing" }) }, status: 404, error: "unknown session: missing" },
      { failure: { status: 1, stdout: "Traceback: /private/data", stderr: "Traceback: /private/journal" }, status: 404, error: "Traceback: /private/journal" },
      { failure: { status: 2, stdout: "Traceback: /private/data", stderr: "" }, status: 502, error: "research read failed" },
      { failure: { status: 0, stdout: "Traceback: /private/data" }, status: 502, error: "research bridge bad json" },
      { failure: { status: 0, stdout: JSON.stringify({ ok: true }) }, status: 502, error: "research bridge bad json" },
    ];
    for (const { failure, status, error } of cases) {
      result = { pid: 1, output: [], stdout: "", stderr: "", status: 0, signal: null, ...failure };
      const response = await read("/api/research/session_1");
      expect(response.status).toBe(status);
      expect(await response.json()).toEqual({ error });
      expect(response.headers.get("Cache-Control")).toBe("no-store");
    }
  });


  const sessionId = "rs:4d6534bd-0c1e-4b8a-9f2d-3a5b6c7d8e9f";
  function sessionPath(id: string, suffix = ""): string {
    return `/api/research/${encodeURIComponent(id)}${suffix}`;
  }

  test("decodes encoded and literal session IDs once for snapshots and journals", async () => {
    const maxId = `rs:${"a".repeat(125)}`;
    for (const [path, expected] of [
      [sessionPath(sessionId), { kind: "snapshot", session_id: sessionId }],
      [sessionPath(sessionId, "/journal?after_seq=5"), { kind: "journal", session_id: sessionId, after_seq: 5 }],
      [`/api/research/${sessionId}`, { kind: "snapshot", session_id: sessionId }],
      [`/api/research/${sessionId}/journal?after=-3`, { kind: "journal", session_id: sessionId, after_seq: 0 }],
      [sessionPath(maxId), { kind: "snapshot", session_id: maxId }],
      ["/api/research/%73essions", { kind: "snapshot", session_id: "sessions" }],
    ] as const) {
      calls.length = 0;
      const response = await read(path);
      expect(response.status).toBe(200);
      expect(response.headers.get("Cache-Control")).toBe("no-store");
      expect(payload()).toEqual(expected);
    }
  });

  test("rejects malformed encodings and invalid decoded session IDs without the bridge", async () => {
    const invalidIds = [
      "rs/../x",
      "rs/x",
      "rs\0x",
      "rs x",
      "rs\nx",
      "..",
      `rs:${"a".repeat(126)}`,
      encodeURIComponent(sessionId),
    ];
    const malformed = ["%ZZ", "rs%ZZ", `${encodeURIComponent(sessionId)}%`, "%E0%A4%A", "%C0%AF", "%FF"];
    for (const path of [
      ...invalidIds.flatMap((id) => [sessionPath(id), sessionPath(id, "/journal")]),
      ...malformed.flatMap((segment) => [`/api/research/${segment}`, `/api/research/${segment}/journal`]),
      "/api/research/",
      "/api/research",
    ]) {
      const response = await read(path);
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "bad session_id" });
    }
    expect(calls.length).toBe(0);
  });

  test("keeps unknown session path shapes as not found without the bridge", async () => {
    for (const path of [sessionPath(sessionId, "/raw"), sessionPath(sessionId, "/journal/extra"), "/api/research/rs/4d6534bd"]) {
      const response = await read(path);
      expect(response.status).toBe(404);
      expect(await response.json()).toEqual({ error: "unknown research path" });
    }
    expect(calls.length).toBe(0);
  });
  test("treats a collapsed empty segment as a literal journal identifier", async () => {
    const response = await read("/api/research//journal");
    expect(response.status).toBe(200);
    expect(payload()).toEqual({ kind: "snapshot", session_id: "journal" });
  });
});
