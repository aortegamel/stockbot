// Read-only projections from the local SQLite ledgers.
interface TraceJob {
  jobId: string;
  parentJobId: string | null;
  jobType: string;
  owner: string;
  waveId: number;
  status: string;
  failureCategory: string | null;
  failureMessage: string | null;
  assignmentId: string | null;
  role: string | null;
  sourceDomain?: string | null;
}

interface TraceEvent {
  seq: number;
  eventType: string;
  payload: Record<string, string | number | boolean | null>;
}

interface GroundedClaimView {
  text: string;
  evidenceIds: string[];
}

export interface EvidenceView {
  evidenceId: string;
  subject: string;
  knownAt: string | null;
  sourceName: string;
  sourceUri: string | null;
  domain?: string;
}

interface CoverageArtifactView {
  artifactId: string;
  waveId: number;
  claimText: string;
  searchId: string;
  query: string;
}

interface FinalResultView {
  answer: string;
  executiveSummary: string;
  consensus: string;
  baseCase: string;
  bullCase: string;
  bearCase: string;
  disagreements: string[];
  positioning: string[];
  catalysts: string[];
  uncertainties: string[];
  whatChangesTheView: string[];
  limitations: string[];
  claims: GroundedClaimView[];
  sources: { evidenceId: string; domain: string; document: string; integrityClass?: string }[];
  coverage: Record<string, unknown>;
  allowedSources: string[];
  freezeId: string;
  asOf: string;
}

export interface ResearchRun {
  sessionId: string;
  waveId: number;
  question: string;
  status: string;
  asOf: string | null;
  updatedAt: string;
  traceId: string | null;
  conclusion: string | null;
  traceStatus: string | null;
  provider: string | null;
  model: string | null;
  jobs: TraceJob[];
  waves?: number[];
  events: TraceEvent[];
  claims: GroundedClaimView[];
  evidence: EvidenceView[];
  freezes: { freezeId: string; evidenceIds: string[] }[];
  dossiers: { dossierId: string; findings: GroundedClaimView[] }[];
  coverageArtifacts?: CoverageArtifactView[];
  finalResult?: FinalResultView | null;
  committeeRuns: unknown[];
}

export interface EvalRun {
  evalRunId: string;
  model: string;
  provider: string;
  harnessVersion: string;
  promptVersion: string;
  gitSha: string;
  scenarioVersion: string;
  startedAt: string;
  passed: number;
  failed: number;
}

export interface EvalScenarioResult {
  evalRunId: string;
  scenarioName: string;
  passed: boolean;
  violations: string[];
}

export interface Experiment {
  experimentId: string;
  beforeRunId: string;
  afterRunId: string;
  deltaPassed: number;
}

export interface FailureRecord {
  failureId: string;
  evalRunId: string;
  scenarioName: string;
  violation: string;
}

export interface HarnessProjection {
  researchRuns: ResearchRun[];
  evalRuns: EvalRun[];
  evalScenarioResults: EvalScenarioResult[];
  experiments: Experiment[];
  failureRecords: FailureRecord[];
}
