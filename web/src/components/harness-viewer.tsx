import { useEffect, useState } from "react";
import type {
  EvalRun,
  EvidenceView,
  HarnessProjection,
  ResearchRun,
} from "./harness-viewer-types";
import { fetchSnapshot, type ResearchSnapshot } from "../lib/research-client";
import { sessionTitle } from "../lib/session-title";
import { ago } from "../lib/time";

function snapshotToRun(sessionId: string, snapshot: ResearchSnapshot): ResearchRun | null {
  const session = snapshot.session;
  if (session === undefined || session.session_id === undefined || session.session_id !== sessionId) {
    return null;
  }
  const jobs = Array.isArray(snapshot.jobs)
    ? snapshot.jobs.map((job, index) => ({
      jobId: job.job_id ?? `${sessionId}-job-${index}`,
      parentJobId: null,
      jobType: job.job_type ?? "research",
      owner: job.owner ?? "kernel",
      waveId: 1,
      status: job.status ?? "unknown",
      failureCategory: null,
      failureMessage: null,
      assignmentId: null,
      role: null,
    }))
    : [];
  return {
    sessionId,
    waveId: 1,
    question: session.query ?? session.objective ?? sessionId,
    status: session.status ?? "unknown",
    asOf: null,
    updatedAt: session.updated_at ?? "",
    traceId: null,
    conclusion: null,
    traceStatus: null,
    provider: null,
    model: null,
    jobs,
    waves: [1],
    events: [],
    claims: [],
    evidence: [],
    freezes: [],
    dossiers: [],
    coverageArtifacts: [],
    finalResult: null,
    committeeRuns: [],
  };
}

type Run = ResearchRun;
type RunJob = Run["jobs"][number];

function waveIds(run: Run): number[] {
  const stored = run.waves ?? [];
  if (stored.length !== 0) {
    return [...stored].sort((a, b) => a - b);
  }
  const ids = run.jobs.map((job) => job.waveId);
  const unique = [...new Set(ids.length === 0 ? [run.waveId] : ids)];
  return unique.sort((a, b) => a - b);
}

function jobsForWave(run: Run, wave: number): RunJob[] {
  return run.jobs.filter((job) => job.waveId === wave);
}

function frozenIds(run: Run): string[] {
  return run.freezes.flatMap((fr) => fr.evidenceIds);
}

function deskCounts(run: Run): string[] {
  return (["SEC", "FINRA", "WEB"] as const).map((domain) => {
    const count = run.evidence.filter((ev) => (ev.domain ?? "").toUpperCase() === domain).length;
    return `${domain} ${count}`;
  });
}

function TopHeader({ run }: { run: Run }) {
  const names = run.evidence.map((ev) => ev.sourceName).filter((name) => name !== "");
  const sources = [...new Set(names)];
  return (
    <>
      <h1>Session</h1>
      <p>Question: {run.question}</p>
      <p>
        Status: {run.status} — wave {run.waveId} — waves [{waveIds(run).join(", ")}]
      </p>
      <p>
        Evidence: {run.evidence.length} records — sources: {sources.length} ({sources.join(", ") || "none"})
      </p>
      <p>Evidence by desk: {deskCounts(run).join(", ")}</p>
      <p>
        Provider: {run.provider ?? "unknown"} model: {run.model ?? "unknown"}
      </p>
      <p>
        <a href={`research://${run.sessionId}`}>{run.sessionId}</a> asOf: {run.asOf ?? "unbounded"} updated:{" "}
        {run.updatedAt}
      </p>
      {run.traceId === null ? <p>Trace missing for this run; showing Job persistence only.</p> : null}
      {run.status !== "completed" && run.status !== "failed" && run.status !== "cancelled" ? (
        <p>
          Run in progress [{run.status}]; jobs and events below are partial.
        </p>
      ) : null}
      {run.status === "failed" ? <p>Run failed; terminal failures listed under Jobs without exceptions.</p> : null}
      {run.evidence.length === 0 ? <p>Empty evidence for this run.</p> : null}
    </>
  );
}

function DeskItem({ job }: { job: RunJob }) {
  const detail = [
    job.jobType,
    job.sourceDomain ?? "domain unknown",
    job.role === null ? null : `role ${job.role}`,
    job.assignmentId === null ? null : `assignment ${job.assignmentId}`,
    `[${job.status}]`,
    job.failureCategory === null ? null : `${job.failureCategory}: ${job.failureMessage ?? ""}`,
  ]
    .filter((part) => part !== null)
    .join(" ");
  return (
    <>
      <a href={`job://${job.jobId}`}>{job.jobId}</a> {detail}
    </>
  );
}

function WaveDesks({ run }: { run: Run }) {
  const waves = waveIds(run);
  return (
    <>
      <h2>Waves and desks</h2>
      {waves.map((wave) => (
        <section key={wave}>
          <h3>Wave {wave}</h3>
          {jobsForWave(run, wave).length === 0 ? (
            <p>No jobs persisted for wave {wave}.</p>
          ) : (
            <ul>
              {jobsForWave(run, wave).map((job) => (
                <li key={job.jobId}>
                  <DeskItem job={job} />
                </li>
              ))}
            </ul>
          )}
        </section>
      ))}
    </>
  );
}

function FreezeCommittee({ run }: { run: Run }) {
  const frozen = frozenIds(run);
  const failed = run.jobs.filter((job) => job.status === "failed" || job.status === "cancelled");
  const finalResult = run.finalResult ?? null;
  const disagreements = finalResult === null ? [] : finalResult.disagreements;
  return (
    <>
      <h2>Freeze and committee</h2>
      {run.freezes.length === 0 ? <p>No freeze persisted.</p> : null}
      <ul>
        {run.freezes.map((fr) => (
          <li key={fr.freezeId}>
            <a href={`freeze://${fr.freezeId}`}>{fr.freezeId}</a> [{fr.evidenceIds.join(", ") || "empty"}]
          </li>
        ))}
      </ul>
      <p>Committee runs: {run.committeeRuns.length}</p>
      {disagreements.length === 0 ? (
        <p>No disagreements persisted.</p>
      ) : (
        <>
          <h3>Disagreement</h3>
          <ul>
            {disagreements.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </>
      )}
      {failed.length === 0 ? null : (
        <>
          <h3>Failures</h3>
          <ul>
            {failed.map((job) => (
              <li key={job.jobId}>
                <a href={`job://${job.jobId}`}>{job.jobId}</a> {job.jobType} [{job.status}]{" "}
                {job.failureCategory ?? "no category"}: {job.failureMessage ?? "no message"}
              </li>
            ))}
          </ul>
        </>
      )}
      <p>Freeze-contained claims only; ids outside [{frozen.join(", ") || "no freeze"}] render as outside freeze.</p>
    </>
  );
}

function NamedList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <>
      <h3>{title}</h3>
      <ul>
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </>
  );
}

function FinalNarrative({ final }: { final: NonNullable<Run["finalResult"]> }) {
  return (<>
    <p>{final.answer || "No answer persisted."}</p>
    {final.executiveSummary !== "" && final.executiveSummary !== final.answer ? (
      <p>Bottom line: {final.executiveSummary}</p>
    ) : null}
    {final.consensus !== "" ? <p>Consensus: {final.consensus}</p> : null}
    {final.baseCase !== "" ? <p>Base case: {final.baseCase}</p> : null}
    {final.bullCase !== "" ? <p>Bull case: {final.bullCase}</p> : null}
    {final.bearCase !== "" ? <p>Bear case: {final.bearCase}</p> : null}
  </>);
}

function FinalClaims({ claims, frozen }: { claims: NonNullable<Run["finalResult"]>["claims"]; frozen: string[] }) {
  return claims.length === 0 ? <p>No validated claims.</p> : (
    <>
      <h3>Claims</h3>
      <ul>
        {claims.map((claim, index) => (
          <li key={index}>
            {claim.text} [
            {claim.evidenceIds.map((eid, inner) => (
              <span key={eid}>
                {inner > 0 ? ", " : ""}
                <a href={`freeze://${eid}`}>{eid}</a>
                {frozen.includes(eid) ? "" : " (outside freeze)"}
              </span>
            ))}
            ]
          </li>
        ))}
      </ul>
    </>
  );
}

function FinalSection({ run }: { run: Run }) {
  const final = run.finalResult ?? null;
  if (final === null) {
    return (
      <>
        <h2>Final</h2>
        <p>No final persisted.</p>
      </>
    );
  }
  const frozen = frozenIds(run);
  return (
    <>
      <h2>Final</h2>
      <FinalNarrative final={final} />
      <NamedList title="Positioning" items={final.positioning} />
      <NamedList title="Catalysts" items={final.catalysts} />
      <NamedList title="Uncertainties" items={final.uncertainties} />
      <NamedList title="What changes the view" items={final.whatChangesTheView} />
      <NamedList title="Limitations" items={final.limitations} />
      <FinalClaims claims={final.claims} frozen={frozen} />
      {final.sources.length === 0 ? null : (
        <>
          <h3>Sources</h3>
          <ul>
            {final.sources.map((source) => (
              <li key={source.evidenceId}>
                {source.domain}{source.integrityClass ? ` [${source.integrityClass}]` : ""} {source.document} [{source.evidenceId}]
              </li>
            ))}
          </ul>
        </>
      )}
      <p>
        Coverage: {JSON.stringify(final.coverage)} — allowed sources: {final.allowedSources.join(", ") || "unknown"} —
        freeze: {final.freezeId || "unknown"}
      </p>
    </>
  );
}

function AuditLog({ run }: { run: Run }) {
  const events = [...run.events].sort((a, b) => a.seq - b.seq);
  const artifacts = run.coverageArtifacts ?? [];
  return (
    <>
      <h2>Audit log</h2>
      <details>
        <summary>
          {events.length} journal events, {run.jobs.length} jobs, {artifacts.length} coverage artifacts
        </summary>
        <ol>
          {events.map((evt) => (
            <li key={`${evt.seq}`}>
              [{evt.seq}] {evt.eventType} {JSON.stringify(evt.payload)}
            </li>
          ))}
        </ol>
        {artifacts.length === 0 ? null : (
          <ul>
            {artifacts.map((artifact) => (
              <li key={artifact.artifactId}>
                {artifact.artifactId} wave {artifact.waveId} search {artifact.searchId} query {artifact.query}:{" "}
                {artifact.claimText}
              </li>
            ))}
          </ul>
        )}
      </details>
    </>
  );
}

function RunTimeline({ run }: { run: Run }) {
  return (
    <>
      <TopHeader run={run} />
      <WaveDesks run={run} />
      <FreezeCommittee run={run} />
      <FinalSection run={run} />
      <AuditLog run={run} />
      <h2>Conclusion</h2>
      <p>{run.conclusion ?? "No conclusion persisted."}</p>
    </>
  );
}

function missingSessionLabel(sessionId: string): string {
  return sessionId === "" ? "unselected" : sessionId;
}

function SessionsList({ runs, selected, onSelect }: { runs: Run[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <>
      <h1>Runs</h1>
      {runs.length === 0 ? (
        <p>No research runs persisted yet. Run a live research session, then re-run scripts/export_harness_viewer.py.</p>
      ) : (
        <p>
          {runs.length} sessions (read-only projections from trace persistence)
        </p>
      )}
      <ul>
        {runs.map((run) => (
          <li key={run.sessionId}>
            <button onClick={() => onSelect(run.sessionId)} aria-pressed={selected === run.sessionId}>
              {run.sessionId}
            </button>{" "}
            <b title={run.question}>{sessionTitle(run.question)}</b>{" "}
            {[run.status, run.updatedAt && ago(run.updatedAt, Date.now())].filter(Boolean).join(" · ")} — wave {run.waveId},{" "}
            {run.evidence.length} evidence, {run.freezes.length} freezes, {run.dossiers.length} dossiers, waves [
            {(run.waves ?? []).join(", ")}]
          </li>
        ))}
      </ul>
    </>
  );
}

function EvidenceDrill({ runs }: { runs: Run[] }) {
  const rows = runs.flatMap((run) => run.evidence.map((ev: EvidenceView) => ({ run, ev })));
  return (
    <>
      <h1>Evidence drill</h1>
      {rows.length === 0 ? (
        <p>No freeze-contained evidence persisted yet.</p>
      ) : (
        <p>{rows.length} evidence records with freeze-contained claim links below.</p>
      )}
      <ul>
        {rows.map(({ run, ev }) => (
          <li key={`${run.sessionId}:${ev.evidenceId}`}>
            <a href={`evidence://${ev.evidenceId}`}>{ev.evidenceId}</a> [{ev.domain ?? "SEC"}] {ev.subject} [
            {ev.knownAt ?? "known_at unknown"}] {ev.sourceName} <a href={`research://${run.sessionId}`}>{run.sessionId}</a>
          </li>
        ))}
      </ul>
    </>
  );
}

function EvalCompare({
  projection,
  activeId,
  onSelect,
}: {
  projection: HarnessProjection;
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  const runs: EvalRun[] = projection.evalRuns;
  const results =
    activeId === null ? [] : projection.evalScenarioResults.filter((row) => row.evalRunId === activeId);
  const failures =
    activeId === null ? [] : projection.failureRecords.filter((row) => row.evalRunId === activeId);
  return (
    <>
      <h1>Eval compare</h1>
      <p>
        {runs.length} eval runs, {projection.experiments.length} experiments
        (projections: evalRuns, evalScenarioResults, experiments)
      </p>
      <ul>
        {runs.map((run) => (
          <li key={run.evalRunId}>
            <button onClick={() => onSelect(run.evalRunId)} aria-pressed={activeId === run.evalRunId}>
              {run.evalRunId}
            </button>{" "}
            {run.model} {run.provider} passed {run.passed} failed {run.failed}
          </li>
        ))}
      </ul>
      <ul>
        {results.map((result) => (
          <li key={result.scenarioName}>
            {result.passed ? "PASS" : "FAIL"} {result.scenarioName}
            {result.violations.length === 0 ? null : <> violations: {result.violations.join("; ")}</>}
          </li>
        ))}
      </ul>
      <ul>
        {projection.experiments.map((exp) => (
          <li key={exp.experimentId}>
            {exp.experimentId} {exp.beforeRunId} → {exp.afterRunId} delta {exp.deltaPassed}
          </li>
        ))}
      </ul>
      <ul>
        {failures.map((row) => (
          <li key={row.failureId}>
            {row.failureId} {row.scenarioName} {row.violation}
          </li>
        ))}
      </ul>
    </>
  );
}

export function HarnessViewer({ projection, sessionId }: { projection: HarnessProjection; sessionId?: string }) {
  const [selectedSession, setSelectedSession] = useState<string | null>(sessionId ?? null);
  const [selectedEvalRun, setSelectedEvalRun] = useState<string | null>(null);
  const runs = projection.researchRuns;
  const run = selectedSession === null ? null : (runs.find((row) => row.sessionId === selectedSession) ?? null);
  const activeEvalId = selectedEvalRun ?? projection.evalRuns[0]?.evalRunId ?? null;
  return (
    <main>
      <SessionsList runs={runs} selected={selectedSession} onSelect={setSelectedSession} />
      {selectedSession === null || run === null ? (
        <>
          <h1>Session timeline</h1>
          <p>No trace persisted for this session (research://{missingSessionLabel(selectedSession ?? "")}).</p>
        </>
      ) : (
        <RunTimeline run={run} />
      )}
      <EvidenceDrill runs={runs} />
      <EvalCompare projection={projection} activeId={activeEvalId} onSelect={setSelectedEvalRun} />
    </main>
  );
}

export function HarnessSessionViewer({ sessionId }: { sessionId: string | null }) {
  const [projection, setProjection] = useState<HarnessProjection | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (sessionId === null) {
      return;
    }
    const current = sessionId;
    let alive = true;
    const ctrl = new AbortController();
    async function load() {
      try {
        const snapshot = await fetchSnapshot(current, ctrl.signal);
        const run = snapshotToRun(current, snapshot);
        if (alive) {
          if (run === null) {
            setError("snapshot unavailable");
          } else {
            setProjection({
              researchRuns: [run],
              evalRuns: [],
              evalScenarioResults: [],
              experiments: [],
              failureRecords: [],
            });
          }
        }
      } catch {
        if (alive) {
          setError("snapshot unavailable");
        }
      }
    }
    void load();
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [sessionId]);
  if (sessionId === null) {
    return (
      <main>
        <h1>Session timeline</h1>
        <p>No trace persisted for this session (research://unselected).</p>
      </main>
    );
  }
  if (error !== null) {
    return (
      <main>
        <h1>Session timeline</h1>
        <p>Snapshot unavailable for this session (research://{sessionId}).</p>
      </main>
    );
  }
  if (projection === null) {
    return (
      <main>
        <h1>Session timeline</h1>
        <p>Loading session (research://{sessionId}).</p>
      </main>
    );
  }
  return <HarnessViewer projection={projection} sessionId={sessionId} />;
}
