// Backend-state reads: research snapshot + journal tail for the terminal.
// Snapshot restores state on reload; journal polls for logs; live work streams
// over POST /api/agent SSE. No mock chat or positions here.
import { readDatapoints, safeSourceUrl, SOURCE_ID_RE } from "../../../needle-harness/lib/agent/types";
import type { Datapoint } from "../../../needle-harness/lib/agent/types";

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
  representation: "html" | "text" | "pdf";
  // structured_record: the held SEC/FINRA data as JSON text, only when no official page exists (linkStatus "missing").
  // calculation: Stockbot's formula text; it never has one original (linkStatus "missing"), its audit links each input.
  captureKind: "original" | "provider_extraction" | "structured_record" | "search_excerpt" | "calculation";
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
  toolArgs?: Record<string, unknown>;
  // The official human page. highlightUrl carries a backend-verified text fragment; originalUrl never does.
  linkStatus: "highlighted_original" | "stockbot_render" | "missing";
  originalUrl?: string;
  highlightUrl?: string;
  // The backend replays the observation or calculation proof when it resolves the source. Evidence sources only.
  audit?: SourceAudit;
  // A published-file conflict does not change the captured API record's proof verdict.
  sourceConflict?: { failure: string; reason: string };
};

// One verified observation or calculation, read-only. sourceId opens that record's own Stockbot view.
export type SourceAudit = {
  evidenceId: string;
  claimType: string;
  verification: "verified" | "unverified" | "conflicted";
  verificationReason: string;
  sourceId?: string;
  value?: string;
  entity?: string;
  period?: string;
  unit?: string;
  knownAt?: string;
  formula?: string;
  formulaId?: string;
  formulaVersion?: string;
  recomputedValue?: string;
  datapoint?: Datapoint & { field?: string; formula?: string };
  source?: { name?: string; document?: string; accession?: string; hash?: string };
  inputs?: SourceAudit[];
  repairs?: SourceRepair[];
};

// Immutable legacy repair history shown beside the ledger verdict. Never a verified citation.
type SourceRepair = {
  repair_id: string;
  status?: string;
  reason?: string;
  category?: string;
  datapoint?: string;
  value?: string;
};

// Citation label metadata: persisted fields only. It never carries a highlighted link.
export type SourceLabel = Pick<SourceDocument, "id" | "title" | "publisher" | "publishedAt" | "knownAt" | "originalUrl"> & {
  linkStatus: "stockbot_render" | "missing";
};

const SOURCE_REPRESENTATIONS: SourceDocument["representation"][] = ["html", "text", "pdf"];
const CAPTURE_KINDS: SourceDocument["captureKind"][] = ["original", "provider_extraction", "structured_record", "search_excerpt", "calculation"];
const LINK_STATUSES: SourceDocument["linkStatus"][] = ["highlighted_original", "stockbot_render", "missing"];
const OPTIONAL_TEXT_KEYS = ["url", "publisher", "publishedAt", "capturedAt", "contentHash", "content", "matchingText", "pdfUrl", "toolName", "toolResultId", "dataset", "knownAt", "originalUrl", "highlightUrl"] as const;
const LOCATION_TEXT_KEYS = ["basis", "prefix", "suffix"] as const;
const AUDIT_TEXT_KEYS = ["value", "entity", "period", "unit", "knownAt", "formula", "formulaId", "formulaVersion", "recomputedValue"] as const;
const AUDIT_SOURCE_KEYS = ["name", "document", "accession", "hash"] as const;
const REPAIR_TEXT_KEYS = ["status", "reason", "category", "datapoint", "value"] as const;
const VERIFICATIONS: SourceAudit["verification"][] = ["verified", "unverified", "conflicted"];
const LABEL_TEXT_KEYS = ["publisher", "publishedAt", "knownAt", "originalUrl"] as const;
const MAX_AUDIT_DEPTH = 16;

function fieldOf(record: object, key: string): unknown {
  return key in record ? (record as Record<string, unknown>)[key] : undefined;
}

function optionalText(record: object, key: string, message: string): string | undefined {
  const value = fieldOf(record, key);
  if (value !== undefined && typeof value !== "string") throw new Error(message);
  return value;
}

function plainObject(value: unknown, message: string): object {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(message);
  return value;
}

function auditSource(raw: unknown): NonNullable<SourceAudit["source"]> {
  const record = plainObject(raw, "The source audit is invalid.");
  const source: NonNullable<SourceAudit["source"]> = {};
  for (const key of AUDIT_SOURCE_KEYS) {
    const value = optionalText(record, key, "The source audit is invalid.");
    if (value !== undefined) source[key] = value;
  }
  return source;
}

function auditDatapoint(raw: unknown): NonNullable<SourceAudit["datapoint"]> {
  const record = plainObject(raw, "The source audit is invalid.");
  const datapoint: NonNullable<SourceAudit["datapoint"]> | undefined = readDatapoints([record])[0];
  if (!datapoint) throw new Error("The source audit is invalid.");
  for (const key of ["field", "formula"] as const) {
    const value = optionalText(record, key, "The source audit is invalid.");
    if (value !== undefined) datapoint[key] = value;
  }
  return datapoint;
}

// Trust boundary: every audit node is typed, bounded in depth, and links only a server source id.
function auditOf(raw: unknown, depth: number): SourceAudit {
  const record = plainObject(raw, "The source audit is invalid.");
  const [evidenceId, claimType, verification, verificationReason] = ["evidenceId", "claimType", "verification", "verificationReason"].map((key) => fieldOf(record, key));
  if (depth > MAX_AUDIT_DEPTH || typeof evidenceId !== "string" || typeof claimType !== "string" || typeof verificationReason !== "string" ||
    !VERIFICATIONS.some((value) => value === verification)) throw new Error("The source audit is invalid.");
  const audit: SourceAudit = { evidenceId, claimType, verification: verification as SourceAudit["verification"], verificationReason };
  const sourceId = optionalText(record, "sourceId", "The source audit is invalid.");
  if (sourceId !== undefined && !SOURCE_ID_RE.test(sourceId)) throw new Error("The source audit is invalid.");
  if (sourceId !== undefined) audit.sourceId = sourceId;
  for (const key of AUDIT_TEXT_KEYS) {
    const value = optionalText(record, key, "The source audit is invalid.");
    if (value !== undefined) audit[key] = value;
  }
  if (fieldOf(record, "source") !== undefined) audit.source = auditSource(fieldOf(record, "source"));
  const datapoint = fieldOf(record, "datapoint");
  if (datapoint !== undefined && audit.verification === "verified") audit.datapoint = auditDatapoint(datapoint);
  const inputs = fieldOf(record, "inputs");
  if (inputs !== undefined) {
    if (!Array.isArray(inputs)) throw new Error("The source audit is invalid.");
    audit.inputs = inputs.map((input) => auditOf(input, depth + 1));
  }
  const repairs = fieldOf(record, "repairs");
  if (repairs !== undefined) {
    if (!Array.isArray(repairs)) throw new Error("The source audit is invalid.");
    audit.repairs = repairs.map((repair) => repairOf(repair));
  }
  return audit;
}

// Trust boundary: repair history is typed display text only, never a citation link.
function repairOf(raw: unknown): SourceRepair {
  const record = plainObject(raw, "The source audit is invalid.");
  const repair_id = fieldOf(record, "repair_id");
  if (typeof repair_id !== "string" || !repair_id) throw new Error("The source audit is invalid.");
  const repair: SourceRepair = { repair_id };
  for (const key of REPAIR_TEXT_KEYS) {
    const value = optionalText(record, key, "The source audit is invalid.");
    if (value !== undefined) repair[key] = value;
  }
  return repair;
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
  const linkStatus = fieldOf(source, "linkStatus");
  if (fieldOf(source, "id") !== sourceId || typeof fieldOf(source, "title") !== "string" || typeof fieldOf(source, "quote") !== "string" ||
    !SOURCE_REPRESENTATIONS.some((value) => value === representation) ||
    !CAPTURE_KINDS.some((value) => value === captureKind) ||
    !LINK_STATUSES.some((value) => value === linkStatus)) {
    throw new Error("The source response is invalid.");
  }
  return {
    id: sourceId,
    title: fieldOf(source, "title") as string,
    representation: representation as SourceDocument["representation"],
    captureKind: captureKind as SourceDocument["captureKind"],
    quote: fieldOf(source, "quote") as string,
    linkStatus: linkStatus as SourceDocument["linkStatus"],
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
  const conflict = fieldOf(source, "sourceConflict");
  if (conflict !== undefined) {
    const record = plainObject(conflict, "The source conflict is invalid.");
    const failure = optionalText(record, "failure", "The source conflict is invalid.");
    const reason = optionalText(record, "reason", "The source conflict is invalid.");
    if (!failure || !reason) throw new Error("The source conflict is invalid.");
    document.sourceConflict = { failure, reason };
  }
}

// Each link status names exactly the links it carries. Links are safe web URLs; the plain original has no text fragment.
// A structured record stands in only for a missing official page. A calculation never has one original.
function validateLinks(document: SourceDocument): void {
  const { linkStatus, originalUrl, highlightUrl } = document;
  const shaped = linkStatus === "missing"
    ? originalUrl === undefined && highlightUrl === undefined
    : originalUrl !== undefined && (linkStatus === "highlighted_original") === (highlightUrl !== undefined) && document.captureKind !== "structured_record" && document.captureKind !== "calculation";
  const safe = [originalUrl, highlightUrl].every((url) => url === undefined || safeSourceUrl(url) !== undefined);
  if (!shaped || !safe || originalUrl?.includes(":~:") || (document.sourceConflict && linkStatus !== "missing")) throw new Error("The source links are invalid.");
}

function validateCapturedContent(document: SourceDocument): void {
  if (document.representation === "pdf") {
    if (document.captureKind !== "original") throw new Error("The PDF does not identify an original capture.");
  } else if (document.content === undefined) throw new Error("The captured source content is missing.");
}

// One source endpoint read. A failed read becomes an Error carrying the backend's controlled message.
async function readSourceJson(path: string, signal?: AbortSignal): Promise<object> {
  const response = await fetch(path, {
    signal,
    cache: "no-store",
    headers: { Accept: "application/json" },
  });
  const body: unknown = await response.json();
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("The source response is invalid.");
  if (!response.ok) {
    const error = fieldOf(body, "error");
    throw new Error(typeof error === "string" ? error : `Source request failed (${response.status}).`);
  }
  return body;
}

// Full source read: the backend resolves, replays, and archives the source for citations and the provenance viewer.
export async function fetchSourceDocument(sourceId: string, signal?: AbortSignal): Promise<SourceDocument> {
  const source = await readSourceJson(`/api/research/sources/${encodeURIComponent(sourceId)}`, signal);
  const document = sourceIdentity(source, sourceId);
  copySourceMetadata(source, document);
  validateLinks(document);
  validateCapturedContent(document);
  const location = locationOf(source);
  if (location !== undefined) document.location = location;
  if (fieldOf(source, "audit") !== undefined) document.audit = auditOf(fieldOf(source, "audit"), 0);
  return document;
}

// Persisted label metadata only. This endpoint does not resolve the full source or download its original.
export async function fetchSourceLabel(sourceId: string, signal?: AbortSignal): Promise<SourceLabel> {
  const source = await readSourceJson(`/api/research/sources/${encodeURIComponent(sourceId)}/label`, signal);
  const title = fieldOf(source, "title");
  const linkStatus = fieldOf(source, "linkStatus");
  if (fieldOf(source, "id") !== sourceId || typeof title !== "string" || (linkStatus !== "stockbot_render" && linkStatus !== "missing")) {
    throw new Error("The source label is invalid.");
  }
  const label: SourceLabel = { id: sourceId, title, linkStatus };
  for (const key of LABEL_TEXT_KEYS) {
    const value = optionalText(source, key, "The source label is invalid.");
    if (value !== undefined) label[key] = value;
  }
  const { originalUrl } = label;
  if ((linkStatus === "missing") !== (originalUrl === undefined) || (originalUrl !== undefined && (safeSourceUrl(originalUrl) === undefined || originalUrl.includes(":~:")))) {
    throw new Error("The source links are invalid.");
  }
  return label;
}
