# Stockbot runtime: needle harness → kernel scheduler

Launch: `bun run stockbot` runs `bun needle-harness/scripts/needle.ts`,
which refreshes the tool catalog via `scripts/tool_bridge.py describe` and
starts three loopback services: the Bun API host (`needle-harness/scripts/api.ts`),
FastAPI (`api.main:app`), and Vite. The Bun host dispatches `POST /api/agent`,
`GET /api/health`, and `GET /api/research/*` to the existing handlers. Vite
proxies those paths to Bun and the existing market `/api/*` paths to FastAPI.
`needle-harness/app/api/agent/route.ts` runs `runKernelAgent`
(`needle-harness/lib/agent/kernel.ts`, prewarmed by the Bun host): one
`app/research/kernel_worker.py` call per request, which persists the prompt
verbatim as the objective, runs the intake (SEC filing metadata + search
snippets per resolved ticker plus one exact-text Exa search, settled with
`continuation=False` so it admits evidence but never resolves), builds a
verbatim digest, runs Reasoner decompose over raw query + digest (optional
tickers/corrected_query hints feed at most one more intake round), then JEV
proposal disposition → per-proposal nodes, then `app/research/scheduler.py
run` over ready nodes. Each intake round logs one `intake_digest` decision
in the research database. The request contains the raw query and summaries of tool calls.
The response contains the intake digest and Reasoner output.
Progress lines (`session`, `intake_start`, `intake_done`, `reasoner_*`,
`tool_start`, `tool_done`) stream as `progress` events; prod forwards a
stripped `working` stage to keep the UI stall watchdog fed.

The live Reasoner prompt states the exact required `objectiveId`.
Its JSON example uses that value and an objective-scoped proposal ID.
The validator rejects proposals that belong to another objective.

The worker gives the existing second attempt the validation error from a rejected response.
It quotes that error as JSON data, not instructions.
It states the required contract separately.
Both attempts use the remaining setup budget.
The worker retains hints only from a valid response.
This repair does not change transport, quota, configuration, or deadline rules.

Explicit 10-Q revenue requests list only `10-Q`.
Latest requests remove ungrounded `start_date` and `end_date` values.
Explicit user dates remain intact.
SEC catalog evidence uses a persisted tool result for replay validation.
The scheduler does not treat a catalog `source_refs` receipt as a document `source_handle`.
The scheduler resolves the catalog receipt to persisted rows.
It selects the latest exact `10-Q` row and reads its document.
The first document read for an undated latest 10-Q revenue request fetches `Item 1`, `Item 2`, and `Item 1A`.
The scheduler reads these sections in parallel without a query filter.
Each section uses separate assessment and admission checks.
The round applies an assessment verdict only after every section read settles.
Later reads, dated requests, and other revenue requests keep one `query='revenue'` read.
A failed section read returns to the selector.
The accession-family and repeated-failure breakers distinguish sibling sections.
For an undated latest 10-Q revenue request, the scheduler uses grounded listing and document arguments without Needle generation.
Requests with explicit dates, extra required arguments, or summary-only listings keep Needle generation.
Document evidence keeps visible inline XBRL facts and the full bounded canonical passage.
The document view excludes hidden XBRL resources.
Attempt summaries stay bounded at 2,000 characters.

On a 10-Q revenue node, a SEC listing cannot resolve the node before a document read succeeds.
On any node, `reason_over_evidence` reloads admitted evidence for a fresh JEV selection within the node round cap.
A rejected document call keeps the whole registry available for JEV selection.
Evidence gates, request walls, scheduling deadlines, and watchdogs stay unchanged.

Before `query_finra` or `get_finra_datapoints` runs, the scheduler checks the dataset against installed FINRA dataset names.
It accepts a bare dataset name or the name after a group prefix.
`get_finra_datapoints` also requires a nonempty list of nonblank field names.
A rejected dataset or field list starts JEV selection instead of a tool call.
The selector can request `list_finra_datasets` or `describe_finra_dataset`.

Before `get_xbrl_facts` runs, the scheduler checks the generated `concept`.
The objective, the node question, or a successful source record must name the concept.
Within source records, only actual content, claims, or financial tag/value rows can ground a concept.
An exact tag word matches.
A `CANONICAL_CONCEPTS` group name or alias also matches whole words, regardless of case, separators, or spacing.
Other friendly labels do not match.

Generated arguments, summaries, metadata, error text, reasoning, and failed sources never ground a concept.
Neither `today_utc` nor the `[Today UTC ...]` prefix grounds a concept.
The scheduler rejects date-shaped concepts, such as a `YYYY` placeholder or a four-digit year.
It also rejects relative-date words, such as `today` or `last quarter`, and concepts without letters.
An ungrounded concept starts JEV selection before the tool runs.
The scheduler never substitutes a fallback concept.

Entry routing sends one `op:route` call.
JEV returns `{route, personas}`.
`personas: null` means the request neither selects nor excludes a persona.
An explicit list is nonempty, unique, and canonical, in the order Stockbot, Bearbot, Bullbot.
Positive requests retain only requested personas that the user does not exclude.
Exclusion-only requests select the remaining personas.
A programmatic route becomes a `route_hint` while JEV selects personas.
A selector failure returns `{id, error}` without an assumed persona set.
No remaining persona also fails the request.

Without an explicit selection, direct and single-tool answers use one Stockbot call.
Research uses all three personas.
An explicit selection always enters research.
Partial selections run the selected personas concurrently over the same graph, evidence, decisions, unresolved items, and authority limits.
Partial selections use no synthesis or draft retry.

The full trio uses synthesis.
Bearbot and Bullbot draft concurrently over the shared snapshot.
Stockbot judges both drafts against that snapshot and writes the middle verdict.
The kernel buffers drafts instead of streaming them.
An empty graph retains the direct prompt without evidence.
Exactly one failed or empty draft retries once.
A second draft failure or a failed or empty synthesis fails the report without prose.

If a first-pass draft or synthesis reports `Missing-Evidence`, the kernel combines unique gaps into one follow-up pass.
The follow-up uses the same session.
The kernel merges the evidence and repeats persona generation with the same synthesis rules.
The follow-up still runs after persona generation exceeds the first research wall.
The kernel never runs a third research pass.
The second pass is final even when a gap remains.

Final gates check the claim inventory and the whole response.
If the whole response fails, the gates check each claim.
The kernel removes each rejected claim substring.
Other prose stays unchanged, including uncertainty and `Missing-Evidence`.
Node IDs and decision records never count as evidence IDs.

An inventory with an unknown evidence ID retries once for each affected persona on the same evidence snapshot.
An inventory whose `claims[].text` is not a verbatim prose substring uses the same retry.
The correction states valid evidence IDs and requires exact evidence row IDs and verbatim claim text.
The correction includes rejected prose and inventory as untrusted output.
The corrected answer passes the same gates.
A still-invalid, empty, or failed correction fails without prose or another retry.
Other inventory defects do not retry.
Direct answers carry no inventory and never retry.
A gate failure, an empty persona, or rejection of all claims fails the report without prose.

A remaining gap does not emit `failed` when the final gates pass.
The kernel emits one `answer_delta` with labeled persona sections.
The trio emits Bearbot and Bullbot drafts before the Stockbot verdict.
Partial selections emit sections in canonical order.
Metrics count all attempted calls and token usage across retries and passes.
Metrics also report the final stage duration.
Each research persona targets 250 words of display prose.
The fenced claim inventory stays outside that target.
Muse receives all projected evidence without an aggregate character cap.

Research wall: the 120-second `REQUEST_WALL_MS` deadline applies to each worker research pass.
The first pass starts its wall at route entry, or at the caller `deadlineAt`.
The follow-up pass gets a fresh wall: `deadlineAt` is the follow-up start plus 120 seconds.
Its worker call timeout is also 120 seconds.
Worst-case research latency is therefore 240 seconds plus unbounded persona time.
One gap adds one follow-up research pass and one more persona generation round.
Persona generation, the draft retry, and Stockbot synthesis have no application time limit.
The kernel waits for them without a deadline, even after the wall passes.
The caller cancellation signal still reaches every persona call and aborts it.

The kernel scheduler owns the agent loop.
JEV selects from the whole registry when deterministic routing does not select a tool.
Needle fills tool arguments only.
Tools execute through `app/tool_runtime.py` against canonical `app/tools.py`.
Thesis monitoring runs the same scheduler in-process through `app/thesis/runner.py run_trigger`.
Its injectable `_RUN_KERNEL` uses `run_graph_prompt` and `scheduler.run` by default.
No OMP/Pi entry or superseded loop remains.

SEC replay evidence uses the tool name, record identity, and tool-result ID
to distinguish records. Repeated citations of the same record and result
reuse the existing evidence ID. Identical records from different result IDs
remain separate. Existing sessions retain their stored identity keys.

Duplicate admissions retain their evidence IDs but do not count as new
evidence or scheduler progress. The harness links a tool call only to its
explicit evidence ID in the session. Calls without an ID receive no link.

JEV receives compact, deterministic context. No LLM summarizes evidence.
Each admitted evidence record carries four fields: `evidence_id`,
`source_name`, `subject`, and `claim_text`. Only `claim_text` is clipped, to
200 characters. Bounded unadmitted observations and do-not-repeat guidance
remain in the context.
The guidance record retains the `instruction` key and the ten most recent `prior_calls`.

Attempt history keeps the tool, its arguments, and its IDs. Summaries and
errors are clipped to 200 characters. Attempt reasoning and metrics are not
sent to JEV.

The JEV client accepts sequences of records, including lists and tuples.
It rejects dictionary inputs with a `TypeError` before it requests a
decision.

Tool options render six fields: name, purpose, useWhen, avoidWhen,
Needs(prerequisites), and PIT(pitSupport). Compact metadata retains the
safety restrictions that descriptions once carried. JEV sees every
registry tool. Needle keeps the full descriptions and parameter schemas.

The argument worker calls native `Needle.complete()` with the raw node question and the SDK native `max_new_tokens=512` generation default.
It binds exactly the tool that JEV selects.
The catalog preserves the original description and parameter schema.
Structured extraction keeps its native 512-token default and 1-4096 validation bound.

The SDK receives every admitted evidence record without clipping.
The SDK facts reuse the shared grounded formatter, `format_grounded_block`.
Each SDK fact block carries the evidence ID, content hash, provenance, complete stored content, and all remaining record fields.
The formatter does not clip SDK facts or Reasoner prompts.
Each Reasoner block carries the evidence ID, content hash, provenance, and complete stored content.
Complete stored content is not complete source text.

Source text that intake clipped before storage stays clipped.
Wrapper instructions such as `grounding_hint` do not enter the SDK facts.

The worker rejects explicit engine failures, SDK-reported ungrounded fields, and negated requests.
The emitted tool must match JEV's selected tool.

The process supervisor retains the 120-second startup budget and shared 120-second worker request timeout.
Argument generation has no separate 10s/20s timeout; the shared worker timeout still bounds each generation call.
SDK `close()` cannot interrupt a blocked inference call.
The research wall and scheduling deadline can still end a request.

Tool selection receives compact session evidence. Result assessment
receives admitted evidence only from the current node's jobs: the current
job and the jobs in its attempt history. It also receives any evidence row
that the node's attempts reference by ID, such as a duplicate winner stored
under another job. Other intake and other-node evidence never support the
node. The raw current outcome stays capped at 2000 characters.

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
