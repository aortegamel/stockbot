import { execFile } from "node:child_process";
import { join } from "node:path";
import { promisify } from "node:util";

// Durable-state reads. /terminal polls these for backend state and logs:
// snapshot = inspect_research (session + jobs + next action),
// journal  = hash-chained journal tail (log tail + poll cursor),
// source   = immutable source capture (document JSON or original PDF bytes).
// All run read-only against data/ via venv python stdlib.
const ROOT = process.cwd().endsWith("needle-harness") ? join(process.cwd(), "..") : process.cwd();

type BridgeOut = { status: number; body: unknown };
const execFileAsync = promisify(execFile);
const SOURCE_ERRORS: Record<number, string> = {
  404: "source capture not found",
  409: "invalid source capture",
  502: "source read failed",
};

type FailureEnvelope = { error: string; status?: unknown };

function readFailureEnvelope(output: string): FailureEnvelope | null {
  try {
    const raw: unknown = JSON.parse(output);
    if (raw && typeof raw === "object" && "ok" in raw && raw.ok === false && "error" in raw && typeof raw.error === "string") {
      return { error: raw.error, status: "status" in raw ? raw.status : undefined };
    }
  } catch {
    // Execution failures can omit the bridge envelope.
  }
  return null;
}

function sourceBridgeFailure(envelope: FailureEnvelope | null, code: unknown): BridgeOut {
  // Only the bridge's own status envelope maps; tracebacks and paths never reach the client.
  const status = envelope && code === 1 && typeof envelope.status === "number" && envelope.status in SOURCE_ERRORS ? envelope.status : 502;
  return { status, body: { error: SOURCE_ERRORS[status] } };
}

function sessionBridgeFailure(error: unknown, envelope: FailureEnvelope | null): BridgeOut {
  if (envelope) return { status: error instanceof Error && "code" in error && error.code === 1 ? 404 : 502, body: { error: envelope.error } };
  const message = error instanceof Error ? error.message : "research read failed";
  return { status: 502, body: { error: `bridge execution failed: ${message.slice(0, 200)}` } };
}

function bridgeFailure(error: unknown, isSource: boolean): BridgeOut {
  const output = error instanceof Error && "stdout" in error && typeof error.stdout === "string" ? error.stdout : "";
  const envelope = readFailureEnvelope(output);
  if (!isSource) return sessionBridgeFailure(error, envelope);
  return sourceBridgeFailure(envelope, error instanceof Error && "code" in error ? error.code : undefined);
}

function bridgeSuccess(stdout: string, isSource: boolean): BridgeOut {
  try {
    const raw: unknown = JSON.parse(stdout || "{}");
    if (!raw || typeof raw !== "object" || !("data" in raw)) {
      return { status: 502, body: { error: isSource ? SOURCE_ERRORS[502] : "research bridge bad shape" } };
    }
    return { status: 200, body: raw.data };
  } catch {
    return { status: 502, body: { error: isSource ? SOURCE_ERRORS[502] : "research bridge bad json" } };
  }
}

async function runBridge(payload: Record<string, unknown>): Promise<BridgeOut> {
  const py = join(ROOT, "venv/bin/python");
  const isSource = payload.kind === "source" || payload.kind === "source_pdf";
  const script = `
import base64, json, sys
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
    if kind in ("source", "source_pdf"):
        try:
            from app.research.sources import read_source_document, read_source_pdf
            if kind == "source":
                out = read_source_document(req["source_id"])
            else:
                out = {"pdf_base64": base64.b64encode(read_source_pdf(req["source_id"])).decode("ascii")}
        except (KeyError, FileNotFoundError):
            print(json.dumps({"ok": False, "status": 404, "error": "source capture not found"}))
            sys.exit(1)
        except ValueError:
            print(json.dumps({"ok": False, "status": 409, "error": "invalid source capture"}))
            sys.exit(1)
        except OSError:
            print(json.dumps({"ok": False, "status": 502, "error": "source read failed"}))
            sys.exit(1)
    elif kind == "snapshot":
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
    ({ stdout } = await execFileAsync(py, ["-c", script, JSON.stringify(payload)], {
      encoding: "utf8",
      timeout: 15_000,
      // Source PDFs ride back as base64; no output cap.
      maxBuffer: Infinity,
    }));
  } catch (error: unknown) {
    return bridgeFailure(error, isSource);
  }
  return bridgeSuccess(stdout, isSource);
}

function sourceIdFromPath(parts: string[], headers: Record<string, string>): string | Response {
  if ((parts.length !== 5 && parts.length !== 6) || (parts.length === 6 && parts[5] !== "pdf")) {
    return Response.json({ error: "unknown source path" }, { status: 404, headers });
  }
  let sourceId: string;
  try {
    sourceId = decodeURIComponent(parts[4]);
  } catch {
    return Response.json({ error: "bad source_id" }, { status: 400, headers });
  }
  if (/^(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})$/.exec(sourceId)?.[0] !== sourceId) {
    return Response.json({ error: "bad source_id" }, { status: 400, headers });
  }
  return sourceId;
}

async function readSource(pathname: string): Promise<Response> {
  const headers = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" };
  // /api/research/sources/:id | /api/research/sources/:id/pdf (raw split keeps empty segments visible)
  const parts = pathname.split("/");
  const sourceId = sourceIdFromPath(parts, headers);
  if (sourceId instanceof Response) return sourceId;
  const isPdf = parts.length === 6;
  const r = await runBridge({ kind: isPdf ? "source_pdf" : "source", source_id: sourceId });
  if (!isPdf || r.status !== 200) return Response.json(r.body, { status: r.status, headers });
  return sourcePdfResponse(r.body, headers);
}

function sourcePdfResponse(body: unknown, headers: Record<string, string>): Response {
  if (
    !body ||
    typeof body !== "object" ||
    !("pdf_base64" in body) ||
    typeof body.pdf_base64 !== "string" ||
    !body.pdf_base64 ||
    /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.exec(body.pdf_base64)?.[0] !== body.pdf_base64
  ) {
    return Response.json({ error: "invalid source PDF response" }, { status: 502, headers });
  }
  return new Response(Buffer.from(body.pdf_base64, "base64"), { headers: { ...headers, "Content-Type": "application/pdf" } });
}

async function readResearch(payload: Record<string, unknown>): Promise<Response> {
  const r = await runBridge(payload);
  return Response.json(r.body, { status: r.status, headers: { "Cache-Control": "no-store" } });
}

function readSessions(params: URLSearchParams): Promise<Response> {
  const limitRaw = params.get("limit") ?? "20";
  const parsed = Number.parseInt(limitRaw, 10);
  const limit = Number.isFinite(parsed) ? Math.min(100, Math.max(1, parsed)) : 20;
  return readResearch({ kind: "sessions", limit });
}

function sessionPayload(sessionId: string, u: URL, parts: string[]): Record<string, unknown> | undefined {
  if (parts.length === 4 && parts[3] === "journal") {
    const afterRaw = u.searchParams.get("after_seq") ?? u.searchParams.get("after") ?? "0";
    const parsed = Number.parseInt(afterRaw, 10);
    const afterSeq = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
    return { kind: "journal", session_id: sessionId, after_seq: afterSeq };
  }
  if (parts.length === 3) return { kind: "snapshot", session_id: sessionId };
}

async function readSession(u: URL, parts: string[]): Promise<Response> {
  let sessionId: string;
  try {
    sessionId = decodeURIComponent(parts[2] ?? "");
  } catch {
    return Response.json({ error: "bad session_id" }, { status: 400 });
  }
  if (!/^[A-Za-z0-9:_-]{1,128}$/.test(sessionId)) {
    return Response.json({ error: "bad session_id" }, { status: 400 });
  }
  const payload = sessionPayload(sessionId, u, parts);
  if (payload) return readResearch(payload);
  return Response.json({ error: "unknown research path" }, { status: 404 });
}

export async function GET(req: Request): Promise<Response> {
  const u = new URL(req.url);
  const parts = u.pathname.split("/").filter(Boolean);
  if (parts[2] === "sources") return readSource(u.pathname);
  // /api/research/sessions | /api/research/:id | /api/research/:id/journal
  if (parts.length === 3 && parts[2] === "sessions") return readSessions(u.searchParams);
  return readSession(u, parts);
}
