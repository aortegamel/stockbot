import { execFile } from "node:child_process";
import { join } from "node:path";
import { promisify } from "node:util";

export const dynamic = "force-dynamic";

// Durable-state reads. /terminal polls these for backend state and logs:
// snapshot = inspect_research (session + jobs + next action),
// journal  = hash-chained journal tail (log tail + poll cursor).
// Both run read-only against data/research.sqlite via venv python stdlib.
const ROOT = process.cwd().endsWith("needle-harness") ? join(process.cwd(), "..") : process.cwd();

type BridgeOut = { status: number; body: unknown };
const execFileAsync = promisify(execFile);

async function runBridge(payload: Record<string, unknown>): Promise<BridgeOut> {
  const py = join(ROOT, "venv/bin/python");
  const script = `
import json, sys
sys.path.insert(0, ${JSON.stringify(ROOT)})
try:
    from app.research import service as svc
    from app.research.repository import ResearchRepository
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)[:300]}))
    sys.exit(2)
try:
    req = json.loads(sys.argv[1])
    kind = req.get("kind")
    if kind == "snapshot":
        out = svc.inspect_research(req["session_id"])
    elif kind == "journal":
        store = ResearchRepository()
        svc._require_session(store, req["session_id"])
        events = store.list_events(req["session_id"])
        after = req.get("after_seq", 0)
        rows = [e.to_dict() for e in events if e.sequence > after]
        out = {"session_id": req["session_id"], "after_seq": after, "events": rows[-200:]}
    elif kind == "sessions":
        store = ResearchRepository()
        out = {"sessions": store.list_sessions(req.get("limit", 20))}
    else:
        print(json.dumps({"error": "unknown kind"})); sys.exit(2)
    print(json.dumps({"ok": True, "data": out}, default=str))
except svc.ResearchNotFound as e:
    print(json.dumps({"ok": False, "error": f"unknown session: {e}"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)[:300]}))
    sys.exit(2)
`;
  let stdout: string;
  try {
    ({ stdout } = await execFileAsync(py, ["-c", script, JSON.stringify(payload)], { encoding: "utf8", timeout: 15_000 }));
  } catch (error: unknown) {
    const output = error instanceof Error && "stdout" in error && typeof error.stdout === "string" ? error.stdout : "";
    try {
      const raw: unknown = JSON.parse(output);
      if (raw && typeof raw === "object" && "ok" in raw && raw.ok === false && "error" in raw && typeof raw.error === "string") {
        const missing = error instanceof Error && "code" in error && error.code === 1;
        return { status: missing ? 404 : 502, body: { error: raw.error } };
      }
    } catch {
      // Execution failures can omit the bridge envelope.
    }
    const message = error instanceof Error ? error.message : "research read failed";
    return { status: 502, body: { error: `bridge execution failed: ${message.slice(0, 200)}` } };
  }
  try {
    const raw: unknown = JSON.parse(stdout || "{}");
    if (!raw || typeof raw !== "object" || !("data" in raw)) {
      return { status: 502, body: { error: "research bridge bad shape" } };
    }
    return { status: 200, body: raw.data };
  } catch {
    return { status: 502, body: { error: "research bridge bad json" } };
  }
}

export async function GET(req: Request): Promise<Response> {
  const u = new URL(req.url);
  const parts = u.pathname.split("/").filter(Boolean);
  // /api/research/sessions | /api/research/:id | /api/research/:id/journal
  if (parts.length === 3 && parts[2] === "sessions") {
    const limitRaw = u.searchParams.get("limit") ?? "20";
    const parsed = Number.parseInt(limitRaw, 10);
    const limit = Number.isFinite(parsed) ? Math.min(100, Math.max(1, parsed)) : 20;
    const r = await runBridge({ kind: "sessions", limit });
    return Response.json(r.body, { status: r.status, headers: { "Cache-Control": "no-store" } });
  }
  const sessionId = parts[2] ?? "";
  if (!sessionId || !/^[A-Za-z0-9:_-]{1,128}$/.test(sessionId)) {
    return Response.json({ error: "bad session_id" }, { status: 400 });
  }
  if (parts.length === 4 && parts[3] === "journal") {
    const afterRaw = u.searchParams.get("after_seq") ?? u.searchParams.get("after") ?? "0";
    const parsed = Number.parseInt(afterRaw, 10);
    const afterSeq = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
    const r = await runBridge({ kind: "journal", session_id: sessionId, after_seq: afterSeq });
    return Response.json(r.body, { status: r.status, headers: { "Cache-Control": "no-store" } });
  }
  if (parts.length === 3) {
    const r = await runBridge({ kind: "snapshot", session_id: sessionId });
    return Response.json(r.body, { status: r.status, headers: { "Cache-Control": "no-store" } });
  }
  return Response.json({ error: "unknown research path" }, { status: 404 });
}
