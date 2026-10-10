import { describe, expect, test } from "bun:test";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const root = resolve(import.meta.dir, "../../../../..");
const route = join(import.meta.dir, "route.ts");
const python = join(root, "venv/bin/python");
const execute = promisify(execFile);
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
    for (const stdout of ["not json", '{"ok":true}']) {
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
