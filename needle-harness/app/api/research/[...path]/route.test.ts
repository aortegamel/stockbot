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
      ["/api/research/sessions?limit=999", { kind: "sessions", limit: 100 }],
      ["/api/research/sessions?limit=invalid", { kind: "sessions", limit: 20 }],
      ["/api/research/session_1", { kind: "snapshot", session_id: "session_1" }],
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
});
