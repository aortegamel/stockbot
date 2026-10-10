# Stockbot runtime: needle harness → kernel scheduler

Launch: `bun run stockbot` runs `bun needle-harness/scripts/needle.ts`,
which refreshes the tool catalog via `scripts/tool_bridge.py describe` and
serves the Next.js harness. The harness posts prompts to
`needle-harness/app/api/agent/route.ts`, which runs
`runKernelAgent` (`needle-harness/lib/agent/kernel.ts`): one
`app/research/kernel_worker.py` call per request, which persists the prompt
verbatim as the objective, runs the intake (SEC filing metadata + search
snippets per resolved ticker plus one exact-text Exa search, settled with
`continuation=False` so it admits evidence but never resolves), builds a
verbatim digest, runs Reasoner decompose over raw query + digest (optional
tickers/corrected_query hints feed at most one more intake round), then JEV
proposal disposition → per-proposal nodes, then `app/research/scheduler.py
run` over ready nodes. Each intake round logs one `intake_digest` decision in
`runs.sqlite` `agent_events`: the request holds the raw query plus per-call
summary views, the response holds the digest plus Reasoner output.
Progress lines (`session`, `intake_start`, `intake_done`, `reasoner_*`,
`tool_start`, `tool_done`) stream as `progress` events; prod forwards a
stripped `working` stage to keep the UI stall watchdog fed.

Architecture: the kernel scheduler owns the agent loop (JEV selects over
the whole registry every round, Needle fills tool arguments, tools execute
via `app/tool_runtime.py` against canonical `app/tools.py`). Thesis
monitoring runs the same scheduler in-process via
`app/thesis/runner.py run_trigger` (injectable `_RUN_KERNEL` seam; the
default runs the shared graph fan-out via `run_graph_prompt` +
`scheduler.run`). No OMP/Pi entry or superseded loop remains.

SEC replay evidence uses the tool name, record identity, and tool-result ID
to distinguish records. Repeated citations of the same record and result
reuse the existing evidence ID. Identical records from different result IDs
remain separate. Existing sessions retain their stored identity keys.

Duplicate admissions retain their evidence IDs but do not count as new
evidence or scheduler progress. The harness links a tool call only to its
explicit evidence ID in the session. Calls without an ID receive no link.

`app/research/runner.py` is a deterministic eval harness only (retired live
loop kept as a test helper): the scheduler owns orchestration, the kernel
owns evidence, PIT, freeze, and stages. Production and eval orchestration
never call `run_live`/`resume_live`.

Live golden (opt-in only, never CI): `bun run verify:hedgefund-live` runs one
golden scenario through the production kernel path with real SEC/FINRA/Exa
credentials (`HEDGEFUND_LIVE=1`, `SEC_EDGAR_IDENTITY`, `FINRA_CLIENT_ID` /
`FINRA_CLIENT_SECRET`, `EXA_ENABLED=1` + `EXA_API_KEY`) and re-exports the
read-only harness-viewer projection. Without the opt-in flag it exits 2.

## SEC discovery labels

The stored description supplies the section label. The file type or form supplies the fallback.
The stored snippet supplies the display window. Without a snippet, the section supplies the window.
Needle extracts only a candidate term from the snippet. Identity metadata and section headings never enter that extraction passage.
Description-only hits skip Needle. The shared term helper can extract financial subjects from their descriptions.

Terms quote one stored field in its original case. The helper excludes filer names regardless of case.
The helper preserves complete amounts, scale words, qualifiers, and clause negation.
Terms contain at most eight words. If that limit cannot preserve essential meaning, the helper returns `None`.
The helper never invents a term from an absent query. Empty source fields remain absent.
These labels aid navigation. They do not replace raw filing evidence.

## Typed runtime contracts

The scheduler uses concrete `_Kernel`, `JevClient`, `ReasonerClient`, and
`ToolOutcome` contracts. Runtime callers and test fakes use these contracts.
The intake journal retains its request and response payloads.

The updated SEC adapters propagate SDK and parser failures.
Wrong input types raise `TypeError`. Invalid values retain their domain errors.
Progress callbacks, heartbeat updates, and failure recording propagate errors.

Explicit failure records and JSON error envelopes retain their response fields.
These boundaries log exception tracebacks. The worker converts uncaught run errors
to its existing `provider_error` envelope.

JEV choice answers must name an own configured option. The parser rejects
inherited names. It retains explicitly configured names such as `__proto__`
without changing the result object's prototype.

## TypeScript usage checks

`knip.json` treats `decision/runtime.ts` as a root entry. The Python process
starts this file. The harness uses its own workspace and manifest.
Its project pattern includes CSS imports.

Knip treats `python3.14`, `poly-crap`, `gitleaks`, `osv-scanner`, and
`llama-server` as external executables. Package dependencies do not provide
these executables.

The guided SEC accession errors carry the tool name in a `tool` key. The error
text and self-correcting hint stay unchanged. The tool-health gate reads this
key for its missing-argument checks.

The clock tool reads time through a `_utc_now` helper. The tool-health gate
doubles this helper with a fixed UTC instant. Production output keeps its
`utc_now` and `source` fields.

The live judge runs each scenario through `create_research`, `create_node`,
and `run_node` in an isolated per-attempt store. JEV, tool invocation, and the
evaluator all use that store. The evaluator reads session status, final
results, evidence, and tool results from the same store. Attempt errors persist
in the result record.

Root TypeScript uses the classic 5.x compiler. The Stryker checker requires
its classic API. The `mutation:ts` script runs the mutation gate.

## Failure cleanup and terminal reads

If graph setup fails after session creation, the worker cancels the open
bootstrap source job. The worker keeps the session record. Cleanup errors
appear in the worker log and do not replace the original result or error.

The terminal clears its snapshot, journal, and sequence when the selected
session changes. It ignores responses from the previous session.

The research API uses an asynchronous Python subprocess with a 15-second
timeout. Unknown sessions return HTTP 404. Import, database, execution, and
bridge response errors return HTTP 502.

The harness binds development and production servers to `127.0.0.1`.
It does not advertise LAN access or provide authentication for remote use.
