import { afterEach, describe, expect, mock, test } from "bun:test";
import * as childProcess from "node:child_process";
import type { ExecFileOptions } from "node:child_process";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const root = resolve(import.meta.dir, "../../../../..");
const route = join(import.meta.dir, "route.ts");
const python = join(root, "venv/bin/python");
const execute = promisify(childProcess.execFile);
type ReadResult = { status: number; cache: string | null; body: Record<string, unknown> };

// A provided python script becomes a private venv/bin/python; otherwise the fixture links the real venv.
async function fixture(realApp = true, python?: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "research-route-"));
  if (python === undefined) await symlink(join(root, "venv"), join(directory, "venv"));
  else {
    await mkdir(join(directory, "venv/bin"), { recursive: true });
    await writeFile(join(directory, "venv/bin/python"), python, { mode: 0o755 });
  }
  if (realApp) await symlink(join(root, "app"), join(directory, "app"));
  else await mkdir(join(directory, "app/research"), { recursive: true });
  return directory;
}

// Each child owns its cwd and environment, so the route resolves ROOT to the temporary fixture.
async function runChild(directory: string, script: string, env: Record<string, string> = {}): Promise<ReadResult[]> {
  const child = Bun.spawn([process.execPath, "-e", script], {
    cwd: directory,
    env: { ...process.env, RESEARCH_DB_PATH: join(directory, "research.sqlite"), ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (code !== 0) throw new Error(`Route child failed (${code}): ${stderr}`);
  return JSON.parse(stdout) as ReadResult[];
}

function readRoutes(directory: string, paths: string[]): Promise<ReadResult[]> {
  return runChild(directory, `
    import { GET } from ${JSON.stringify(route)};
    const results = [];
    for (const path of ${JSON.stringify(paths)}) {
      const response = await GET(new Request("http://localhost" + path));
      results.push({ status: response.status, cache: response.headers.get("Cache-Control"), body: await response.json() });
    }
    console.log(JSON.stringify(results));
  `);
}

// Bun records LCOV for the test process alone, so route behavior that runs inside a spawned
// child is invisible to coverage. These in-process tests drive the route against a replaced
// execFile. bridgeReply = null passes the call through to the real bridge.
type BridgeOutcome = { stdout: string } | { failed: { stdout?: string; code?: unknown } } | { rejected: unknown };
type ExecFileCallback = (error: unknown, result?: { stdout: string; stderr: string }) => void;
let bridgeReply: ((payload: Record<string, unknown>) => BridgeOutcome) | null = null;
let bridgeCalls: Record<string, unknown>[] = [];
// Captured before the replacement installs, so the passthrough can never recurse into the stub.
const realExecFile = childProcess.execFile;
// execFile -> promisify custom shape: the route awaits { stdout, stderr }.
const promisified = (callback: ExecFileCallback) =>
  (error: unknown, result?: { stdout: string; stderr: string }) => callback(error, error ? undefined : result);

mock.module("node:child_process", () => ({
  ...childProcess,
  execFile: (file: string, args: string[], options: ExecFileOptions, callback: ExecFileCallback): unknown => {
    if (!bridgeReply) {
      return (realExecFile as unknown as (...a: unknown[]) => unknown)(file, args, options, (error: unknown, stdout: string, stderr: string) =>
        promisified(callback)(error, { stdout, stderr }),
      );
    }
    const payload = JSON.parse(args[2]) as Record<string, unknown>;
    bridgeCalls.push(payload);
    const outcome = bridgeReply(payload);
    if ("stdout" in outcome) callback(null, { stdout: outcome.stdout, stderr: "" });
    else if ("failed" in outcome) callback(Object.assign(new Error("bridge command failed"), outcome.failed));
    else callback(outcome.rejected);
    return undefined;
  },
}));

afterEach(() => {
  bridgeReply = null;
  bridgeCalls = [];
});

type InProcessResult = { status: number; cache: string | null; nosniff: string | null; type: string | null; text: string; bytes: string };
// Dynamic import cannot be static here: the replacement above must install before route.ts binds
// node:child_process at module scope. The query keeps this instance separate from any copy another
// test file loaded first, which would otherwise pin the real bridge.
const IN_PROCESS_ROUTE = "./route.ts?in-process";

// One route call inside this process; the body stays readable as JSON text and as raw base64.
async function readInProcess(path: string): Promise<InProcessResult> {
  const { GET } = await import(IN_PROCESS_ROUTE);
  const response = await GET(new Request("http://localhost" + path));
  const buffer = Buffer.from(await response.arrayBuffer());
  return {
    status: response.status,
    cache: response.headers.get("Cache-Control"),
    nosniff: response.headers.get("X-Content-Type-Options"),
    type: response.headers.get("Content-Type"),
    text: buffer.toString("utf8"),
    bytes: buffer.toString("base64"),
  };
}

describe("research read route", () => {
  test("reads snapshots, sessions, and the journal from a temporary store", async () => {
    const directory = await fixture();
    try {
      const { stdout } = await execute(python, ["-c", `
from app.research import service
print(service.create_research("Route test question"))
      `], { cwd: root, env: { ...process.env, RESEARCH_DB_PATH: join(directory, "research.sqlite") } });
      const id = stdout.trim();
      const [snapshot, sessions, journal, after] = await readRoutes(directory, [
        `/api/research/${id}`,
        "/api/research/sessions",
        `/api/research/${id}/journal`,
        `/api/research/${id}/journal?after_seq=1`,
      ]);
      for (const response of [snapshot, sessions, journal, after]) {
        expect(response.status).toBe(200);
        expect(response.cache).toBe("no-store");
      }
      expect(snapshot.body.session).toMatchObject({ session_id: id, query: "Route test question" });
      expect(snapshot.body.jobs).toHaveLength(1);
      expect(sessions.body.sessions).toEqual(expect.arrayContaining([expect.objectContaining({ session_id: id })]));
      expect(journal.body.events).toEqual([expect.objectContaining({ session_id: id, sequence: 1, event_type: "job.started" })]);
      expect(after.body.events).toEqual([]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("returns 404 for explicit unknown sessions, including the journal", async () => {
    const directory = await fixture();
    try {
      const responses = await readRoutes(directory, ["/api/research/missing", "/api/research/missing/journal"]);
      for (const response of responses) {
        expect(response.status).toBe(404);
        expect(response.cache).toBe("no-store");
        expect(response.body.error).toContain("unknown session");
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("returns 502 for database failures on each read endpoint", async () => {
    const directory = await fixture();
    try {
      await writeFile(join(directory, "research.sqlite"), "not a sqlite database");
      const responses = await readRoutes(directory, ["/api/research/session", "/api/research/session/journal", "/api/research/sessions"]);
      for (const response of responses) {
        expect(response.status).toBe(502);
        expect(response.cache).toBe("no-store");
        expect(response.body.error).toContain("database");
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("catches Python import failures inside the bridge", async () => {
    const directory = await fixture(false);
    try {
      await writeFile(join(directory, "app/research/service.py"), 'raise ImportError("route import failed")\n');
      const [response] = await readRoutes(directory, ["/api/research/session"]);
      expect(response.status).toBe(502);
      expect(response.cache).toBe("no-store");
      expect(response.body.error).toBe("route import failed");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("does not treat unrelated Python KeyError as a missing session", async () => {
    const directory = await fixture(false);
    try {
      await writeFile(join(directory, "app/research/service.py"), `
class ResearchNotFound(KeyError):
    pass

def inspect_research(session_id):
    raise KeyError("internal field")
`);
      await writeFile(join(directory, "app/research/repository.py"), "class ResearchRepository: pass\n");
      const [response] = await readRoutes(directory, ["/api/research/session"]);
      expect(response.status).toBe(502);
      expect(response.body.error).toContain("internal field");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  // Uses the production 15-second timeout against a real sleeping child; the route exposes no clock to inject.
  test("maps a real bridge timeout to 502", async () => {
    const directory = await fixture(false, "#!/bin/sh\nexec sleep 20\n");
    try {
      const [response] = await readRoutes(directory, ["/api/research/session"]);
      expect(response.status).toBe(502);
      expect(response.cache).toBe("no-store");
      expect(response.body.error).toContain("execution failed");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }, 20_000);

  test("maps unstructured exit code 1 failures to 502", async () => {
    const directory = await fixture(false, "#!/bin/sh\necho 'Python failed before the envelope' >&2\nexit 1\n");
    try {
      const [response] = await readRoutes(directory, ["/api/research/session"]);
      expect(response.status).toBe(502);
      expect(response.cache).toBe("no-store");
      expect(response.body.error).toContain("execution failed");
      // execFile's message embeds the full bridge command line, so the 200-character cap always applies.
      expect(response.body.error).toHaveLength("bridge execution failed: ".length + 200);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("only a failed string-error envelope with exit code 1 is a missing session", async () => {
    for (const envelope of [
      '{"ok":true,"kind":"unknown_session","error":"unknown session: x"}',
      '{"ok":false,"kind":"unknown_session","error":42}',
    ]) {
      const directory = await fixture(false, `#!/bin/sh\nprintf '%s' '${envelope}'\nexit 1\n`);
      try {
        const [response] = await readRoutes(directory, ["/api/research/session"]);
        expect(response.status).toBe(502);
        expect(response.cache).toBe("no-store");
        expect(typeof response.body.error).toBe("string");
        expect(response.body.error).toContain("execution failed");
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  });

  test("returns 502 for invalid success envelopes", async () => {
    for (const stdout of ["not json", '{"ok":true}', "", "null", "42", "[]"]) {
      const directory = await fixture(false, `#!/bin/sh\nprintf '%s' '${stdout}'\n`);
      try {
        const [response] = await readRoutes(directory, ["/api/research/session"]);
        expect(response.status).toBe(502);
        expect(response.body.error).toMatch(/bad (json|shape)/);
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  });

  test("serves a real Python read that waits for a gate released after GET starts", async () => {
    const directory = await fixture(false);
    const gate = join(directory, "gate");
    try {
      await execute("mkfifo", [gate]);
      await writeFile(join(directory, "app/research/repository.py"), "class ResearchRepository: pass\n");
      await writeFile(join(directory, "app/research/service.py"), `
import os

class ResearchNotFound(KeyError):
    pass

def inspect_research(session_id):
    with open(os.environ["ROUTE_TEST_GATE"]) as gate:
        return {"session_id": session_id, "gate": gate.read()}
`);
      // A blocking bridge cannot return from GET before Python exits, so the FIFO writer never opens.
      const [response] = await runChild(directory, `
        import { writeFile } from "node:fs/promises";
        import { GET } from ${JSON.stringify(route)};
        const pending = GET(new Request("http://localhost/api/research/session"));
        const released = writeFile(process.env.ROUTE_TEST_GATE, "released");
        const response = await pending;
        await released;
        console.log(JSON.stringify([{ status: response.status, cache: response.headers.get("Cache-Control"), body: await response.json() }]));
      `, { ROUTE_TEST_GATE: gate });
      expect(response.status).toBe(200);
      expect(response.cache).toBe("no-store");
      expect(response.body).toEqual({ session_id: "session", gate: "released" });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("rejects invalid session ids and unknown research paths without reading", async () => {
    const directory = await fixture(false);
    try {
      const [badId, unknownPath] = await readRoutes(directory, ["/api/research/bad.id", "/api/research/session/unrecognized"]);
      expect(badId).toMatchObject({ status: 400, body: { error: "bad session_id" } });
      expect(unknownPath).toMatchObject({ status: 404, body: { error: "unknown research path" } });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

// Fake venv/bin/python: echoes the bridge payload (argv[3] after "-c script") inside the given envelope.
function echoPython(envelope: string, exit = 0): string {
  return `#!/usr/bin/env python3
import json, sys
payload = json.loads(sys.argv[3])
print(json.dumps(${envelope}))
sys.exit(${exit})
`;
}

async function readRaw(directory: string, path: string): Promise<{ status: number; headers: Record<string, string | null>; bytes: string; text: string }> {
  const [result] = (await runChild(directory, `
    import { GET } from ${JSON.stringify(route)};
    const response = await GET(new Request("http://localhost" + ${JSON.stringify(path)}));
    const buffer = Buffer.from(await response.arrayBuffer());
    console.log(JSON.stringify([{
      status: response.status,
      headers: { cache: response.headers.get("Cache-Control"), nosniff: response.headers.get("X-Content-Type-Options"), type: response.headers.get("Content-Type") },
      bytes: buffer.toString("base64"),
      text: buffer.toString("utf8"),
    }]));
  `)) as unknown as { status: number; headers: Record<string, string | null>; bytes: string; text: string }[];
  return result;
}

const captureId = `capture:${"a".repeat(64)}`;
function sourcePath(id: string, suffix = ""): string {
  return `/api/research/sources/${encodeURIComponent(id)}${suffix}`;
}

describe("research source route", () => {
  test("returns the stored source document for the requested capture", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": {"title": "Filing", "payload": payload}}`));
    try {
      const response = await readRaw(directory, sourcePath(captureId));
      expect(response.status).toBe(200);
      expect(response.headers).toMatchObject({ cache: "no-store", nosniff: "nosniff" });
      expect(JSON.parse(response.text)).toEqual({ title: "Filing", payload: { kind: "source", source_id: captureId } });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("returns only the PDF bytes from the bridge", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": {"pdf_base64": "JVBERg==", "kind": payload["kind"]}}`));
    try {
      const response = await readRaw(directory, sourcePath(captureId, "/pdf"));
      expect(response.status).toBe(200);
      expect(response.headers).toMatchObject({ cache: "no-store", nosniff: "nosniff", type: "application/pdf" });
      expect(response.bytes).toBe("JVBERg==");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("accepts encoded evidence IDs at the length boundary", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": payload}`));
    try {
      for (const id of ["evidence:a:b_c-1", `evidence:${"a".repeat(256)}`]) {
        const response = await readRaw(directory, sourcePath(id));
        expect(response.status).toBe(200);
        expect(JSON.parse(response.text)).toEqual({ kind: "source", source_id: id });
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("rejects URLs, paths, invalid IDs, bad encoding, and extra segments without the bridge", async () => {
    // A bridge call would answer 200; every case must stop before it.
    const directory = await fixture(false, echoPython(`{"ok": True, "data": {}}`));
    try {
      const invalid = [
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
        `${captureId}\n`,
      ];
      for (const path of [...invalid.map((id) => sourcePath(id)), "/api/research/sources/%ZZ"]) {
        const response = await readRaw(directory, path);
        expect(response.status).toBe(400);
        expect(JSON.parse(response.text)).toEqual({ error: "bad source_id" });
        expect(response.headers).toMatchObject({ cache: "no-store", nosniff: "nosniff" });
      }
      for (const [path, status, error] of [
        ["/api/research/sources", 404, "unknown source path"],
        ["/api/research/sources/", 400, "bad source_id"],
        [sourcePath(captureId, "/"), 404, "unknown source path"],
        [sourcePath(captureId, "/raw"), 404, "unknown source path"],
        [sourcePath(captureId, "/pdf/extra"), 404, "unknown source path"],
        [`/api/research/sources//${captureId}`, 404, "unknown source path"],
        ["/api/research/sources/%ZZ/raw", 404, "unknown source path"],
        ["/api/research/sources/%ZZ/pdf", 400, "bad source_id"],
        ["/api/research/sources//pdf", 400, "bad source_id"],
        [`/api//research/sources/${captureId}`, 404, "unknown source path"],
      ] as const) {
        const response = await readRaw(directory, path);
        expect(response.status).toBe(status);
        expect(JSON.parse(response.text)).toEqual({ error });
        expect(response.headers).toMatchObject({ cache: "no-store", nosniff: "nosniff" });
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("maps missing, corrupt, and unreadable captures to fixed messages", async () => {
    for (const [status, error] of [
      [404, "source capture not found"],
      [409, "invalid source capture"],
      [502, "source read failed"],
    ] as const) {
      const directory = await fixture(
        false,
        echoPython(`{"ok": False, "status": ${status}, "error": "Traceback: /private/data/capture.pdf"}`, 1),
      );
      try {
        for (const suffix of ["", "/pdf"]) {
          const response = await readRaw(directory, sourcePath(captureId, suffix));
          expect(response.status).toBe(status);
          expect(JSON.parse(response.text)).toEqual({ error });
        }
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  });

  test("hides tracebacks and unrecognized bridge failures", async () => {
    for (const python of [
      "#!/bin/sh\necho 'Traceback: /private/data/capture.pdf' >&2\nexit 1\n",
      echoPython(`{"ok": False, "error": "missing /private/data/capture.pdf"}`, 2),
      echoPython(`{"ok": False, "status": 200, "error": "Traceback"}`, 1),
      echoPython(`{"ok": False, "status": 404, "error": "Traceback"}`, 2),
      echoPython(`{"ok": False, "status": "404", "error": "Traceback"}`, 1),
      echoPython(`{"ok": False, "status": 404, "error": 42}`, 1),
      echoPython(`{"ok": True, "status": 404, "error": "Traceback"}`, 1),
      echoPython(`None`, 1),
      echoPython(`42`, 1),
      echoPython(`{"error": "Traceback"}`, 1),
      echoPython(`{"ok": False}`, 1),
      "#!/bin/sh\nprintf 'Traceback: /private/data'\n",
      echoPython(`{"ok": True}`),
    ]) {
      const directory = await fixture(false, python);
      try {
        const response = await readRaw(directory, sourcePath(captureId));
        expect(response.status).toBe(502);
        expect(JSON.parse(response.text)).toEqual({ error: "source read failed" });
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  });

  test("rejects malformed PDF bridge responses", async () => {
    for (const data of ["None", "42", '"text"', "{}", `{"pdf_base64": 42}`, `{"pdf_base64": ""}`, `{"pdf_base64": "not base64!"}`, `{"pdf_base64": "JVBERg==\\n"}`]) {
      const directory = await fixture(false, echoPython(`{"ok": True, "data": ${data}}`));
      try {
        const response = await readRaw(directory, sourcePath(captureId, "/pdf"));
        expect(response.status).toBe(502);
        expect(JSON.parse(response.text)).toEqual({ error: "invalid source PDF response" });
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    }
  });
});

describe("research session id decoding", () => {
  const sessionId = "rs:4d6534bd-0c1e-4b8a-9f2d-3a5b6c7d8e9f";

  test("decodes encoded and literal session IDs once for snapshots and journals", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": payload}`));
    try {
      const paths = [
        `/api/research/${encodeURIComponent(sessionId)}`,
        `/api/research/${encodeURIComponent(sessionId)}/journal?after_seq=5`,
        `/api/research/${sessionId}`,
        "/api/research/%73essions",
      ];
      const responses = await readRoutes(directory, paths);
      expect(responses.map((r) => r.body)).toEqual([
        { kind: "snapshot", session_id: sessionId },
        { kind: "journal", session_id: sessionId, after_seq: 5 },
        { kind: "snapshot", session_id: sessionId },
        { kind: "snapshot", session_id: "sessions" },
      ]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("rejects malformed encodings and invalid decoded IDs", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": {}}`));
    try {
      const paths = [
        ...["rs/../x", "rs x", "rs\nx", `rs:${"a".repeat(126)}`, encodeURIComponent(sessionId)].map(
          (id) => `/api/research/${encodeURIComponent(id)}`,
        ),
        "/api/research/%ZZ",
        "/api/research/%ZZ/journal",
        "/api/research/%C0%AF",
      ];
      for (const response of await readRoutes(directory, paths)) {
        expect(response.status).toBe(400);
        expect(response.body).toEqual({ error: "bad session_id" });
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe("research query boundaries", () => {
  test("defaults and clamps session limits after decimal integer parsing", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": payload}`));
    try {
      const cases = [
        ["", 20],
        ["?limit=", 20],
        ["?limit=bad", 20],
        ["?limit=Infinity", 20],
        ["?limit=-1", 1],
        ["?limit=0", 1],
        ["?limit=1", 1],
        ["?limit=100", 100],
        ["?limit=101", 100],
        ["?limit=2.9", 2],
        ["?limit=7tail", 7],
        ["?limit=0x10", 1],
        ["?limit=%20%2B8%20", 8],
        ["?limit=4&limit=99", 4],
      ] as const;
      const responses = await readRoutes(directory, cases.map(([query]) => `/api/research/sessions${query}`));
      expect(responses).toEqual(cases.map(([, limit]) => ({
        status: 200,
        cache: "no-store",
        body: { kind: "sessions", limit },
      })));
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("preserves journal cursor precedence and lower bounds", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": payload}`));
    try {
      const cases = [
        ["", 0],
        ["?after=7", 7],
        ["?after_seq=5&after=7", 5],
        ["?after_seq=&after=7", 0],
        ["?after_seq=bad&after=7", 0],
        ["?after_seq=Infinity", 0],
        ["?after_seq=-4", 0],
        ["?after_seq=0", 0],
        ["?after_seq=2.9", 2],
        ["?after_seq=7tail", 7],
        ["?after_seq=0x10", 0],
        ["?after_seq=1000000", 1000000],
        ["?after_seq=%20%2B8%20", 8],
        ["?after_seq=4&after_seq=99", 4],
      ] as const;
      const responses = await readRoutes(directory, cases.map(([query]) => `/api/research/session/journal${query}`));
      expect(responses).toEqual(cases.map(([, after_seq]) => ({
        status: 200,
        cache: "no-store",
        body: { kind: "journal", session_id: "session", after_seq },
      })));
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test("validates session IDs before unknown paths without cache headers", async () => {
    const directory = await fixture(false, echoPython(`{"ok": True, "data": payload}`));
    try {
      const responses = await readRoutes(directory, [
        "/api/research",
        "/api/research/%ZZ/extra",
        "/api/research/bad.id/extra",
        "/api/research/session/extra",
        "/api/research/sessions/extra",
      ]);
      expect(responses).toEqual([
        { status: 400, cache: null, body: { error: "bad session_id" } },
        { status: 400, cache: null, body: { error: "bad session_id" } },
        { status: 400, cache: null, body: { error: "bad session_id" } },
        // A valid id with an unrecognized trailing segment is an unknown path, not a bad id.
        { status: 404, cache: null, body: { error: "unknown research path" } },
        { status: 404, cache: null, body: { error: "unknown research path" } },
      ]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe("research route in-process", () => {
  test("serves snapshots, journals, and sessions from the bridge payload", async () => {
    bridgeReply = (payload) => ({ stdout: JSON.stringify({ ok: true, data: payload }) });
    const cases: [string, number, string, unknown][] = [
      ["/api/research/session", 200, "no-store", { kind: "snapshot", session_id: "session" }],
      ["/api/research/%73essions", 200, "no-store", { kind: "snapshot", session_id: "sessions" }],
      ["/api/research/rs:4d65/journal", 200, "no-store", { kind: "journal", session_id: "rs:4d65", after_seq: 0 }],
      ["/api/research/session/journal?after=9", 200, "no-store", { kind: "journal", session_id: "session", after_seq: 9 }],
      ["/api/research/session/journal?after_seq=3&after=9", 200, "no-store", { kind: "journal", session_id: "session", after_seq: 3 }],
      ["/api/research/sessions", 200, "no-store", { kind: "sessions", limit: 20 }],
      ["/api/research/sessions?limit=250", 200, "no-store", { kind: "sessions", limit: 100 }],
      ["/api/research/sessions?limit=-4", 200, "no-store", { kind: "sessions", limit: 1 }],
      ["/api/research/sessions?limit=junk", 200, "no-store", { kind: "sessions", limit: 20 }],
    ];
    for (const [path, status, cache, body] of cases) {
      const response = await readInProcess(path);
      expect({ path, status: response.status, cache: response.cache, body: JSON.parse(response.text) }).toEqual({ path, status, cache, body });
    }
    expect(bridgeCalls).toEqual([
      { kind: "snapshot", session_id: "session" },
      { kind: "snapshot", session_id: "sessions" },
      { kind: "journal", session_id: "rs:4d65", after_seq: 0 },
      { kind: "journal", session_id: "session", after_seq: 9 },
      { kind: "journal", session_id: "session", after_seq: 3 },
      { kind: "sessions", limit: 20 },
      { kind: "sessions", limit: 100 },
      { kind: "sessions", limit: 1 },
      { kind: "sessions", limit: 20 },
    ]);
  });

  test("rejects bad session ids and unknown paths without a bridge call", async () => {
    bridgeReply = (payload) => ({ stdout: JSON.stringify({ ok: true, data: payload }) });
    const cases: [string, number, unknown][] = [
      ["/api/research", 400, { error: "bad session_id" }],
      ["/api/research/%ZZ", 400, { error: "bad session_id" }],
      ["/api/research/bad.id/extra", 400, { error: "bad session_id" }],
      ["/api/research/%C0%AF", 400, { error: "bad session_id" }],
      ["/api/research/session/other", 404, { error: "unknown research path" }],
      ["/api/research/sessions/other", 404, { error: "unknown research path" }],
    ];
    for (const [path, status, body] of cases) {
      const response = await readInProcess(path);
      expect({ path, status: response.status, cache: response.cache, body: JSON.parse(response.text) })
        .toEqual({ path, status, cache: null, body });
    }
    expect(bridgeCalls).toEqual([]);
  });

  test("maps bridge envelopes, rejections, and malformed output to fixed statuses", async () => {
    const cases: [BridgeOutcome, number, string | RegExp][] = [
      [{ failed: { stdout: '{"ok":false,"status":404,"error":"unknown session: gone"}', code: 1 } }, 404, "unknown session: gone"],
      [{ failed: { stdout: '{"ok":false,"error":"database is locked"}', code: 2 } }, 502, "database is locked"],
      [{ failed: { stdout: '{"ok":false,"error":"worker died"}', code: 1 } }, 404, "worker died"],
      [{ failed: { stdout: '{"ok":true,"error":"ignored"}', code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: '{"ok":false,"error":42}', code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: "not json", code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ rejected: "raw string failure" }, 502, "bridge execution failed: research read failed"],
      [{ failed: { stdout: "null", code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: '"text"', code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: '{"status":404,"error":"gone"}', code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: '{"ok":false}', code: 1 } }, 502, "bridge execution failed: bridge command failed"],
      [{ failed: { stdout: '{"ok":false,"error":"worker died"}', code: 0 } }, 502, "worker died"],
      [{ stdout: "not json" }, 502, "research bridge bad json"],
      [{ stdout: "" }, 502, "research bridge bad shape"],
      [{ stdout: '{"ok":true}' }, 502, "research bridge bad shape"],
      [{ stdout: "null" }, 502, "research bridge bad shape"],
      [{ stdout: "42" }, 502, "research bridge bad shape"],
    ];
    for (const [outcome, status, error] of cases) {
      bridgeReply = () => outcome;
      const response = await readInProcess("/api/research/session");
      expect({ outcome, status: response.status, cache: response.cache, body: JSON.parse(response.text) })
        .toEqual({ outcome, status, cache: "no-store", body: { error } });
    }
    // A rejection carrying no value still surfaces as a bridge execution failure.
    bridgeReply = () => ({ rejected: undefined });
    const rejected = await readInProcess("/api/research/session");
    expect(rejected.status).toBe(502);
    expect(JSON.parse(rejected.text).error).toMatch(/^bridge execution failed: /);
  });

  test("serves source documents and PDF bytes from the bridge payload", async () => {
    bridgeReply = (payload) => ({
      stdout: JSON.stringify(payload.kind === "source_pdf"
        ? { ok: true, data: { pdf_base64: "JVBERg==", kind: payload.kind } }
        : { ok: true, data: { title: "Filing", payload } }),
    });
    const document = await readInProcess(sourcePath(captureId));
    expect({ status: document.status, cache: document.cache, nosniff: document.nosniff, type: document.type, body: JSON.parse(document.text) }).toEqual({
      status: 200, cache: "no-store", nosniff: "nosniff", type: "application/json;charset=utf-8",
      body: { title: "Filing", payload: { kind: "source", source_id: captureId } },
    });
    const pdf = await readInProcess(sourcePath(captureId, "/pdf"));
    expect({ status: pdf.status, cache: pdf.cache, nosniff: pdf.nosniff, type: pdf.type, bytes: pdf.bytes }).toEqual({
      status: 200, cache: "no-store", nosniff: "nosniff", type: "application/pdf", bytes: "JVBERg==",
    });
    expect(pdf.text).toBe("%PDF");
    expect(bridgeCalls).toEqual([{ kind: "source", source_id: captureId }, { kind: "source_pdf", source_id: captureId }]);
  });

  test("stops invalid source paths and ids before the bridge", async () => {
    bridgeReply = (payload) => ({ stdout: JSON.stringify({ ok: true, data: payload }) });
    const invalidIds = ["https://example.com/file.pdf", "/etc/passwd", "../capture", "evidence:../../secret", "capture:not-a-hash",
      `capture:${"A".repeat(64)}`, `capture:${"a".repeat(63)}`, `capture:${"a".repeat(65)}`, "evidence:",
      `evidence:${"a".repeat(257)}`, "evidence:with space", "evidence:a\n", `${captureId}\n`];
    const cases: [string, number, string][] = [
      ...invalidIds.map((id): [string, number, string] => [sourcePath(id), 400, "bad source_id"]),
      ["/api/research/sources", 404, "unknown source path"],
      ["/api/research/sources/", 400, "bad source_id"],
      ["/api/research/sources/%ZZ", 400, "bad source_id"],
      ["/api/research/sources/%ZZ/raw", 404, "unknown source path"],
      ["/api/research/sources/%ZZ/pdf", 400, "bad source_id"],
      ["/api/research/sources//pdf", 400, "bad source_id"],
      [sourcePath(captureId, "/"), 404, "unknown source path"],
      [sourcePath(captureId, "/raw"), 404, "unknown source path"],
      [sourcePath(captureId, "/pdf/extra"), 404, "unknown source path"],
      [`/api/research/sources//${captureId}`, 404, "unknown source path"],
      [`/api//research/sources/${captureId}`, 404, "unknown source path"],
    ];
    for (const [path, status, error] of cases) {
      const response = await readInProcess(path);
      expect({ path, status: response.status, cache: response.cache, nosniff: response.nosniff, body: JSON.parse(response.text) })
        .toEqual({ path, status, cache: "no-store", nosniff: "nosniff", body: { error } });
    }
    expect(bridgeCalls).toEqual([]);
  });

  test("hides source bridge failures and rejects malformed PDF payloads", async () => {
    const envelope = (status: unknown, code: unknown, error: unknown = "Traceback: /private/data/capture.pdf"): BridgeOutcome =>
      ({ failed: { stdout: JSON.stringify({ ok: false, status, error }), code } });
    const cases: [BridgeOutcome, number, string][] = [
      [envelope(404, 1), 404, "source capture not found"],
      [envelope(409, 1), 409, "invalid source capture"],
      [envelope(502, 1), 502, "source read failed"],
      [envelope(404, 2), 502, "source read failed"],
      [envelope("404", 1), 502, "source read failed"],
      [envelope(500, 1), 502, "source read failed"],
      [envelope(404, 1, 42), 502, "source read failed"],
      [{ failed: { stdout: '{"ok":true,"status":404,"error":"Traceback"}', code: 1 } }, 502, "source read failed"],
      [{ failed: { stdout: "Traceback: /private/data/capture.pdf", code: 1 } }, 502, "source read failed"],
      [{ failed: { code: 1 } }, 502, "source read failed"],
      [{ rejected: "raw string failure" }, 502, "source read failed"],
      [{ stdout: '{"ok":true}' }, 502, "source read failed"],
      [{ stdout: "not json" }, 502, "source read failed"],
      [{ stdout: '{"ok":true,"data":"text"}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":null}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":{}}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":{"pdf_base64":42}}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":{"pdf_base64":""}}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":{"pdf_base64":"not base64!"}}' }, 502, "invalid source PDF response"],
      [{ stdout: '{"ok":true,"data":{"pdf_base64":"JVBERg==\\n"}}' }, 502, "invalid source PDF response"],
    ];
    for (const [outcome, status, error] of cases) {
      bridgeReply = () => outcome;
      const response = await readInProcess(sourcePath(captureId, "/pdf"));
      expect({ outcome, status: response.status, cache: response.cache, nosniff: response.nosniff, body: JSON.parse(response.text) })
        .toEqual({ outcome, status, cache: "no-store", nosniff: "nosniff", body: { error } });
    }
    bridgeReply = () => envelope(404, 1);
    const document = await readInProcess(sourcePath(captureId));
    expect({ status: document.status, cache: document.cache, nosniff: document.nosniff, body: JSON.parse(document.text) })
      .toEqual({ status: 404, cache: "no-store", nosniff: "nosniff", body: { error: "source capture not found" } });
  });

  test("reads a real research store through the production bridge", async () => {
    // The route resolves ROOT from cwd, so this call runs the repo venv exactly as the server does.
    const directory = await mkdtemp(join(tmpdir(), "research-route-parent-"));
    const database = join(directory, "research.sqlite");
    const previous = process.env.RESEARCH_DB_PATH;
    process.env.RESEARCH_DB_PATH = database;
    bridgeReply = null;
    try {
      const { stdout } = await execute(python, ["-c", `
from app.research import service
print(service.create_research("In-process route question"))
      `], { cwd: root, env: { ...process.env, RESEARCH_DB_PATH: database } });
      const id = stdout.trim();
      const snapshot = await readInProcess(`/api/research/${id}`);
      const sessions = await readInProcess("/api/research/sessions");
      const journal = await readInProcess(`/api/research/${id}/journal?after_seq=1`);
      expect({ status: snapshot.status, cache: snapshot.cache, body: JSON.parse(snapshot.text) }).toMatchObject({
        status: 200, cache: "no-store", body: { session: { session_id: id, query: "In-process route question" } },
      });
      expect(JSON.parse(sessions.text).sessions).toEqual(expect.arrayContaining([expect.objectContaining({ session_id: id })]));
      expect(JSON.parse(journal.text)).toMatchObject({ session_id: id, after_seq: 1, events: [] });
    } finally {
      if (previous === undefined) delete process.env.RESEARCH_DB_PATH;
      else process.env.RESEARCH_DB_PATH = previous;
      await rm(directory, { recursive: true, force: true });
    }
  }, 20_000);
});
