import { useEffect, useState } from "react";
import { fetchScans, searchTickers, type ScanDef, type SearchHit } from "../lib/market-data";
import { sessionTitle } from "../lib/session-title";
import { ago } from "../lib/time";
import type { ShellState } from "./Shell";

type Thesis = { thesis_id: string; scope: string; user_thesis: string };
// A row without run is a one-line notice (empty or failed group).
type Row = { group: string; label: string; detail?: string; run?: () => void };

const PER_GROUP = 5;

// Global palette: `/` (outside text fields) or Ctrl/Cmd-K opens it, Esc closes it.
export default function CommandBar({ watchlist, sessions, onTicker, onScan, onSession, onAsk }: {
  watchlist: string[];
  sessions: ShellState["sessions"];
  onTicker: (sym: string) => void;
  onScan: (scan: ScanDef) => void;
  onSession: (id: string) => void;
  onAsk: (text: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  // null: the request failed.
  const [hits, setHits] = useState<SearchHit[] | null>([]);
  const [scans, setScans] = useState<ScanDef[]>([]);
  const [theses, setTheses] = useState<Thesis[] | null>([]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as Element | null)?.closest?.("input, textarea, select, [contenteditable]");
      if (((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") || (e.key === "/" && !typing)) {
        e.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Scans and theses load on each open, so a closed palette sends no requests.
  useEffect(() => {
    if (!open) return;
    const ctrl = new AbortController();
    fetchScans(ctrl.signal).then((page) => setScans(page.scans), () => setScans([]));
    fetch("/api/theses", { signal: ctrl.signal })
      .then((res) => (res.ok ? (res.json() as Promise<Thesis[]>) : Promise.reject(new Error(`${res.status}`))))
      .then(setTheses, () => !ctrl.signal.aborted && setTheses(null));
    return () => ctrl.abort();
  }, [open]);

  const q = query.trim();
  useEffect(() => {
    if (!open || !q) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      searchTickers(q, ctrl.signal).then((page) => setHits(page.hits), () => !ctrl.signal.aborted && setHits(null));
    }, 200);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [open, q]);

  if (!open) {
    return (
      <button type="button" className="cmd-trigger" onClick={() => setOpen(true)}>
        Search or do anything <kbd>/</kbd>
      </button>
    );
  }

  const lc = q.toLowerCase();
  const has = (...fields: (string | undefined)[]) => !lc || fields.some((f) => f?.toLowerCase().includes(lc));
  const group = (name: string, rows: Row[], empty: string) => (rows.length ? rows.slice(0, PER_GROUP) : [{ group: name, label: empty }]);
  const rows: Row[] = [
    ...(q
      ? group("Tickers", (hits ?? []).map((h) => ({ group: "Tickers", label: h.sym, detail: h.name, run: () => onTicker(h.sym) })),
        hits === null ? "ticker search unavailable" : "no matching tickers")
      : []),
    ...group("Watchlist", watchlist.filter((s) => has(s)).map((s) => ({ group: "Watchlist", label: s, run: () => onTicker(s) })), "no watchlist match"),
    // Saved scans show only when the endpoint returns data.
    ...scans.filter((s) => has(s.name)).slice(0, PER_GROUP).map((s) => ({ group: "Saved scans", label: s.name, run: () => onScan(s) })),
    ...group("Theses", (theses ?? []).filter((t) => has(t.scope, t.user_thesis)).map((t) => ({
      group: "Theses", label: t.scope, detail: t.user_thesis.slice(0, 60), run: () => onTicker(t.scope),
    })), theses === null ? "theses unavailable" : "no theses"),
    ...group("Recent sessions", sessions.filter((s) => s.session_id && has(s.session_id, s.query, sessionTitle(s.query ?? ""))).map((s) => ({
      group: "Recent sessions",
      label: sessionTitle(s.query ?? "") || s.session_id!,
      detail: [s.status, s.updated_at && ago(s.updated_at, Date.now())].filter(Boolean).join(" · "),
      run: () => onSession(s.session_id!),
    })), "no recent sessions"),
    ...(q ? [{ group: "Ask", label: `Ask Stockbot: ${q}`, run: () => onAsk(q) }] : []),
  ];
  const actions = rows.filter((r) => r.run);
  const current = actions[Math.min(active, actions.length - 1)];

  function close() {
    setOpen(false);
    setQuery("");
    setActive(0);
    setHits([]);
  }
  function run(row: Row | undefined) {
    if (!row?.run) return;
    close();
    row.run();
  }

  return (
    <span className="cmd">
      <input
        autoFocus
        role="combobox"
        aria-expanded="true"
        aria-controls="cmd-list"
        aria-label="Search or do anything"
        className="cmd-input"
        value={query}
        placeholder="Search or do anything"
        onBlur={close}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            close();
          }
          else if (e.key === "Enter") run(current);
          else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && actions.length) {
            e.preventDefault();
            const step = e.key === "ArrowDown" ? 1 : -1;
            setActive((i) => (Math.min(i, actions.length - 1) + step + actions.length) % actions.length);
          }
        }}
      />
      <div id="cmd-list" role="listbox" aria-label="Commands" className="cmd-list">
        {rows.map((r, i) => (
          <div key={i}>
            {r.group !== rows[i - 1]?.group && <div className="cmd-group">{r.group}</div>}
            {r.run ? (
              // mousedown would blur the input first and close the palette before the click lands.
              <div role="option" aria-selected={r === current} className="cmd-row" onMouseDown={(e) => e.preventDefault()} onClick={() => run(r)}>
                <b>{r.label}</b> {r.detail && <span className="cmd-detail">{r.detail}</span>}
              </div>
            ) : (
              <div className="cmd-row cmd-empty">{r.label}</div>
            )}
          </div>
        ))}
      </div>
    </span>
  );
}
