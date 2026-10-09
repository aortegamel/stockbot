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
