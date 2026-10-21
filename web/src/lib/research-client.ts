// Backend-state reads: research snapshot + journal tail for the terminal.
// Snapshot restores state on reload; journal polls for logs; live work streams
// over POST /api/agent SSE. No mock chat or positions here.
export type ResearchSnapshot = {
  session?: { session_id?: string; status?: string; query?: string; objective?: string; updated_at?: string };
  jobs?: { job_id?: string; job_type?: string; status?: string; owner?: string }[];
  jobs_by_status?: Record<string, unknown>;
  pending_next_action?: unknown;
  latest_freeze?: unknown;
  error?: string;
};

export type JournalPage = {
  session_id?: string;
  after_seq?: number;
  events?: {
    event_id?: string;
    sequence?: number;
    event_type?: string;
    timestamp?: string;
    actor_type?: string;
    actor_id?: string;
    payload?: unknown;
  }[];
  error?: string;
};

export type SourceDocument = {
  id: string;
  title: string;
  url?: string;
  publisher?: string;
  publishedAt?: string;
  capturedAt?: string;
  contentHash?: string;
  representation: "html" | "text" | "json" | "pdf";
  captureKind: "original" | "provider_extraction" | "structured_record" | "search_excerpt";
  content?: string;
  matchingText?: string;
  pdfUrl?: string;
  quote: string;
  location?: { offset?: number; end?: number; basis?: string; prefix?: string; suffix?: string };
  // How the source was fetched. Sessions recorded before these fields omit them.
  toolName?: string;
  toolResultId?: string;
  dataset?: string;
  knownAt?: string;
  upstreamUrl?: string;
  publicUrl?: string;
  toolArgs?: Record<string, unknown>;
};

const SOURCE_REPRESENTATIONS: SourceDocument["representation"][] = ["html", "text", "json", "pdf"];
const CAPTURE_KINDS: SourceDocument["captureKind"][] = ["original", "provider_extraction", "structured_record", "search_excerpt"];
const OPTIONAL_TEXT_KEYS = ["url", "publisher", "publishedAt", "capturedAt", "contentHash", "content", "matchingText", "pdfUrl", "toolName", "toolResultId", "dataset", "knownAt", "upstreamUrl", "publicUrl"] as const;
const LOCATION_TEXT_KEYS = ["basis", "prefix", "suffix"] as const;

function fieldOf(record: object, key: string): unknown {
  return key in record ? (record as Record<string, unknown>)[key] : undefined;
}

function validateLocationNumbers(record: Record<string, unknown>): void {
  for (const key of ["offset", "end"]) {
    const value = record[key];
    if (value !== undefined && (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)) {
      throw new Error("The source location is invalid.");
    }
  }
}

function validateLocationText(record: Record<string, unknown>): void {
  for (const key of LOCATION_TEXT_KEYS) {
    if (record[key] !== undefined && typeof record[key] !== "string") throw new Error("The source location is invalid.");
  }
}

function copyLocation(record: Record<string, unknown>): NonNullable<SourceDocument["location"]> {
  const out: NonNullable<SourceDocument["location"]> = {};
  if (typeof record.offset === "number") out.offset = record.offset;
  if (typeof record.end === "number") out.end = record.end;
  for (const key of LOCATION_TEXT_KEYS) {
    if (typeof record[key] === "string") out[key] = record[key];
  }
  return out;
}

function locationOf(source: object): SourceDocument["location"] | undefined {
  const location = fieldOf(source, "location");
  if (location === undefined) return undefined;
  if (!location || typeof location !== "object" || Array.isArray(location)) throw new Error("The source location is invalid.");
  const record = location as Record<string, unknown>;
  validateLocationNumbers(record);
  validateLocationText(record);
  return copyLocation(record);
}

export async function fetchSnapshot(sessionId: string, signal?: AbortSignal): Promise<ResearchSnapshot> {
  const res = await fetch(`/api/research/${encodeURIComponent(sessionId)}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  const body = (await res.json()) as ResearchSnapshot;
  if (!res.ok) {
    throw new Error(typeof body.error === "string" ? body.error : `snapshot ${res.status}`);
  }
  return body;
}

export async function fetchJournal(
  sessionId: string,
  afterSeq: number,
  signal?: AbortSignal,
): Promise<JournalPage> {
  const res = await fetch(`/api/research/${encodeURIComponent(sessionId)}/journal?after_seq=${afterSeq}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  const body = (await res.json()) as JournalPage;
  if (!res.ok) {
    throw new Error(typeof body.error === "string" ? body.error : `journal ${res.status}`);
  }
  return body;
}

export async function listSessions(limit = 20, signal?: AbortSignal): Promise<{ sessions?: { session_id?: string; status?: string; updated_at?: string; query?: string }[] }> {
  const res = await fetch(`/api/research/sessions?limit=${limit}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  const body = (await res.json()) as {
    sessions?: { session_id?: string; status?: string; updated_at?: string; query?: string }[];
    error?: string;
  };
  if (!res.ok) {
    throw new Error(typeof body.error === "string" ? body.error : `sessions ${res.status}`);
  }
  return body;
}

function sourceIdentity(source: object, sourceId: string): SourceDocument {
  const representation = fieldOf(source, "representation");
  const captureKind = fieldOf(source, "captureKind");
  if (fieldOf(source, "id") !== sourceId || typeof fieldOf(source, "title") !== "string" || typeof fieldOf(source, "quote") !== "string" ||
    !SOURCE_REPRESENTATIONS.some((value) => value === representation) ||
    !CAPTURE_KINDS.some((value) => value === captureKind)) {
    throw new Error("The source response is invalid.");
  }
  return {
    id: sourceId,
    title: fieldOf(source, "title") as string,
    representation: representation as SourceDocument["representation"],
    captureKind: captureKind as SourceDocument["captureKind"],
    quote: fieldOf(source, "quote") as string,
  };
}

function copySourceMetadata(source: object, document: SourceDocument): void {
  for (const key of OPTIONAL_TEXT_KEYS) {
    const value = fieldOf(source, key);
    if (value !== undefined) {
      if (typeof value !== "string") throw new Error("The source metadata is invalid.");
      document[key] = value;
    }
  }
  const args = fieldOf(source, "toolArgs");
  if (args !== undefined) {
    if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("The source metadata is invalid.");
    document.toolArgs = { ...args };
  }
}

function validateCapturedContent(document: SourceDocument): void {
  if (document.representation === "pdf") {
    if (document.captureKind !== "original") throw new Error("The PDF does not identify an original capture.");
  } else if (document.content === undefined) throw new Error("The captured source content is missing.");
}

export async function fetchSourceDocument(sourceId: string, signal?: AbortSignal): Promise<SourceDocument> {
  const response = await fetch(`/api/research/sources/${encodeURIComponent(sourceId)}`, {
    signal,
    cache: "no-store",
    headers: { Accept: "application/json" },
  });
  const body: unknown = await response.json();
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("The source response is invalid.");
  const source: object = body;
  if (!response.ok) {
    const error = fieldOf(source, "error");
    throw new Error(typeof error === "string" ? error : `Source request failed (${response.status}).`);
  }
  const document = sourceIdentity(source, sourceId);
  copySourceMetadata(source, document);
  validateCapturedContent(document);
  const location = locationOf(source);
  if (location !== undefined) document.location = location;
  return document;
}
