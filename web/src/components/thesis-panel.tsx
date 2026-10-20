import { useEffect, useState } from "react";

type ThesisSummary = {
  thesis_id: string;
  slug: string;
  status: string;
  scope: string;
  user_thesis: string;
  updated_at: string;
  alerts: number;
  pending: number;
};
type ThesisDetail = ThesisSummary & {
  claims: { claim_id: string; statement: string; status: string }[];
  assessment: string;
  watch_rules: { rule_id: string; rule_type: string; enabled: boolean; support_status: string }[];
  journal: { entry_id: string; created_at: string; title: string; body: string }[];
  last_tick: string | null;
};

const POLL_MS = 60_000;
// Same explicit-scope rule as the monitor's targets_for_thesis.
const TICKER = /^[A-Za-z]{1,5}$/;
const muted = "var(--muted, #7A8599)";
const down = "var(--down, #EF5350)";
const small = { fontSize: 12 } as const;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { Accept: "application/json" } });
  const body = (await res.json().catch(() => ({}))) as T & { detail?: unknown };
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `request failed: ${res.status}`);
  return body;
}

// Persistent theses from /api/theses: polled every 60s (no push), one expandable detail.
export function ThesisPanel({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  const [theses, setTheses] = useState<ThesisSummary[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ThesisDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [rev, setRev] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    const fail = (e: unknown) => {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    };
    const load = () => {
      call<ThesisSummary[]>("/api/theses", { signal: ctrl.signal })
        .then((rows) => {
          setTheses(rows);
          setError(null);
        })
        .catch(fail);
      if (openId) {
        call<ThesisDetail>(`/api/theses/${encodeURIComponent(openId)}`, { signal: ctrl.signal }).then(setDetail).catch(fail);
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      clearInterval(id);
      ctrl.abort();
    };
  }, [openId, rev]);

  function open(t: ThesisSummary) {
    if (TICKER.test(t.scope)) onSelectTicker(t.scope.toUpperCase());
    setDetail(null);
    setOpenId(t.thesis_id === openId ? null : t.thesis_id);
  }

  async function act(action: "check" | "pause" | "resume") {
    if (!openId || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      await call(`/api/theses/${encodeURIComponent(openId)}/${action}`, { method: "POST" });
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setRev((r) => r + 1);
    }
  }

  const shown = detail?.thesis_id === openId ? detail : null;
  return (
    <div style={{ padding: "var(--sp-3)" }}>
      {error && <div style={{ ...small, color: down }}>{error}</div>}
      {theses.length === 0 && !error && <div style={{ ...small, color: muted }}>no theses yet</div>}
      {theses.map((t) => (
        <div key={t.thesis_id}>
          <button
            onClick={() => open(t)}
            aria-expanded={t.thesis_id === openId}
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              padding: "4px 6px",
              background: t.thesis_id === openId ? "var(--fill-active)" : "transparent",
              border: t.thesis_id === openId ? "1px solid var(--accent, #2962FF)" : "1px solid transparent",
              borderRadius: 4,
              color: "var(--text, #D5DBE5)",
              cursor: "pointer",
              lineHeight: "20px",
              fontSize: 14,
            }}
          >
            <b style={{ fontWeight: 600 }}>{t.scope}</b> <span style={{ ...small, color: muted }}>{t.status}</span>
            {t.alerts > 0 && (
              <span
                aria-label={`${t.alerts} alerts, ${t.pending} pending`}
                style={{ ...small, marginLeft: 6, padding: "0 5px", borderRadius: 8, background: t.pending > 0 ? down : "var(--accent, #2962FF)", color: "#fff" }}
              >
                {t.alerts}
              </span>
            )}
            <div style={{ ...small, lineHeight: "16px", color: muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.user_thesis}</div>
          </button>
          {t.thesis_id === openId && shown && (
            <div style={{ ...small, padding: "6px 8px", color: "var(--text, #D5DBE5)" }}>
              <div style={{ color: muted }}>
                {shown.assessment} · last tick {shown.last_tick ?? "never"}
              </div>
              <div style={{ display: "flex", gap: 6, margin: "6px 0" }}>
                <button disabled={busy} onClick={() => act("check")}>
                  Check now
                </button>
                <button disabled={busy} onClick={() => act(shown.status === "paused" ? "resume" : "pause")}>
                  {shown.status === "paused" ? "Resume" : "Pause"}
                </button>
              </div>
              {actionError && <div style={{ color: down }}>{actionError}</div>}
              <div aria-label="claims">
                {shown.claims.map((c) => (
                  <div key={c.claim_id}>• {c.statement} <span style={{ color: muted }}>[{c.status}]</span></div>
                ))}
              </div>
              <div aria-label="watch rules" style={{ marginTop: 4 }}>
                {shown.watch_rules.length === 0 && <div style={{ color: muted }}>no watch rules</div>}
                {shown.watch_rules.map((r) => (
                  <div key={r.rule_id} style={{ color: r.enabled ? undefined : muted }}>
                    watch {r.rule_type} {r.enabled ? "" : `(${r.support_status})`}
                  </div>
                ))}
              </div>
              <div aria-label="journal" style={{ marginTop: 4, maxHeight: 200, overflowY: "auto" }}>
                {shown.journal.length === 0 && <div style={{ color: muted }}>no journal entries</div>}
                {shown.journal.map((j) => (
                  <div key={j.entry_id} style={{ marginBottom: 4 }}>
                    <span style={{ color: muted }}>{j.created_at.slice(0, 16)}</span> <b>{j.title}</b>
                    <div style={{ whiteSpace: "pre-wrap" }}>{j.body.slice(0, 400)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
