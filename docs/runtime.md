# Stockbot runtime: needle harness → kernel scheduler

Launch: `bun run stockbot` runs `bun needle-harness/scripts/needle.ts`,
which refreshes the tool catalog via `scripts/tool_bridge.py describe` and
starts three loopback services: the Bun API host (`needle-harness/scripts/api.ts`),
FastAPI (`api.main:app`), and Vite. The Bun host dispatches `POST /api/agent`,
`GET /api/watchlist`, `GET /api/health`, and `GET /api/research/*` to the existing handlers. Vite
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
The gate requests carry the research session id of the run they judge.
A gate error rejects its claim or response; it never keeps ungraded prose.
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
The browser sends the selected ticker separately from the original question.
The route and research session retain that ticker context.
Pure clock questions use the clock tool even when a ticker context exists.

The browser also sends its actual watchlist as a separate `watchlist` list.
It never adds the symbols to the prompt.
The route rejects a malformed list with HTTP 400. It accepts 1-25 valid tickers and removes duplicates.
A prompt that names the watchlist ("watchlist" or "watch list") takes watchlist scope.
That request drops the selected-ticker scope: the list is the requested scope.
The route passes the list to JEV as separate entry context, in the same `op:"route"` call.
JEV sees the scope before it decides whether one registry tool fits, so the watchlist never requires a code-side route override.
JEV's returned route governs dispatch: `no_session` and `reasoning_required` answer direct, a tool name runs single-shot, anything else researches.
On a watchlist request the same list rides `op:"run"`, and the intake fan-out covers every symbol.
Live JEV behavior for a watchlist prompt is unverified until a live run checks it; the mocked route replies only prove the context and the dispatch wiring.
Every other prompt keeps the selected-ticker behavior and ignores the list, so a selected-ticker question never becomes watchlist-wide.

Watchlist sources load at two triggers through one worker path:

1. Page load: the watchlist panel sends `GET /api/watchlist?symbols=...` once per watchlist.
   The worker `op:"watchlist"` runs on its own thread, so it never waits behind a research run.
   It creates no session, job, or evidence.
2. Watchlist question: round 1 of the intake fans out the same calls.
   The existing boundary admits each successful result as evidence.

Each symbol gets four fixed-argument calls through the existing tool handlers.
`list_sec_filings` uses `limit=5` and no forms filter, so fund forms stay in.
The three FINRA calls are `get_short_interest`, `get_reg_sho_volume` (the handler's current NYC week), and `get_threshold_securities`.
Every call is an independent task on the existing SEC and tool pools under the 15s intake budget.
A call past the budget is cancelled and reported as a timeout.
Each (symbol, source) row carries `status` (`ok`, `no_records`, `error`, `timeout`), `as_of`, source references, and the evidence ID when admitted.
A FINRA "No data found" answer is `no_records`. Any other failure is `error`.
The intake_digest decision persists the rows.
The writer prompt lists every row under WATCHLIST COVERAGE, so a failed source is disclosed.
Reuse follows the existing caches: FINRA results use the `app.cache` 3600s TTL.
SEC submissions use the edgartools HTTP cache with its 30s revalidation.
A fresh entry serves the agent request without source work. An expired entry refreshes through the same path.

Tools execute through `app/tool_runtime.py` against canonical `app/tools.py`.
Thesis monitoring runs the same scheduler in-process through `app/thesis/runner.py run_trigger`.
Its injectable `_RUN_KERNEL` uses `run_graph_prompt` and `scheduler.run` by default.
No OMP/Pi entry or superseded loop remains.

SEC replay evidence uses the tool name, record identity, and tool-result ID
to distinguish records. Repeated citations of the same record and result
reuse the existing evidence ID. Identical records from different result IDs
remain separate. Existing sessions retain their stored identity keys.

FINRA replay evidence uses the tool name and record identity only, so an
identical repeated call dedupes onto the stored row. The record identity is
the first canonical row when the result carries rows. `get_short_interest`
and `get_reg_sho_volume` answer with an analysis envelope and no rows, so
their identity is the envelope query JSON, which names the one symbol and
window the result covers. The briefing source stays the last fallback; it is
one constant string, so using it as the identity collapses every symbol onto
the first admitted row and leaves the other symbols uncitable.

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

The argument SDK receives no research evidence, raw tool results, or attempt history.
Its context contains temporal anchors, the selected ticker, and explicitly grounded argument values.
The server filters context fields before it binds the selected tool.
Durable storage retains complete evidence records.
JEV retains its evidence context outside the argument SDK.
Downstream checks retain evidence for identifier validation and quote validation.
Each Reasoner block carries the evidence ID, content hash, provenance, and complete stored content.
Complete stored content is not complete source text.
Source text that intake clipped before storage stays clipped.
Wrapper instructions such as `grounding_hint` do not enter the SDK facts.
The worker rejects explicit engine failures, SDK-reported ungrounded fields, and negated requests.
The emitted tool must match JEV's selected tool.
For record-count `limit` arguments, the worker applies one order: a structured grounded limit, then an explicit count in the original request, then a declared schema default.
Record counts never constrain the character-window `limit` on `get_sec_document` (whose schema has `max_chars`).
Recognized count phrases include "latest 5 filings", "5 most recent filings", "up to 37 records", and "limit 10".
Plain digit counts are read whole without a length ceiling. "limit 2000" and "up to 2000 records" request 2000.
Dates, CIKs, accession numbers, form names such as 10-K, and year text such as "latest 2024 filings" are not counts.
Percentages, including "position limit of 10% of float", are not record counts.
Two different requested counts fail before generation.
A count written as a compound ("one hundred", "thirty five"), with separators ("1,000"), or with an unsupported word ("seventy") also fails before generation.
The error asks for plain digits.
A generated limit that differs from the requested count fails, and an omitted limit fails too.
SEC `forms` filters must be exact form types from the installed edgartools registries, or a single `/A` suffix on a registered non-amendment form.
Exact registered amendments such as `N-MFP2/A` pass even without a registered base form. Unregistered doubled `/A` values fail.
An unknown form, such as a ticker, fails as `invalid_tool_arguments` before any EDGAR call.
The scheduler passes generated forms through unchanged.

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

The terminal Harness tab renders `HarnessSessionViewer` with the selected session id. It reads the existing session plus snapshot research APIs only. A missing or failed snapshot stays unavailable for that session, never another session. Static `HarnessViewer` with a projection prop remains for untracked preview files; export previews write only the `--out` path.

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

## JEV decision API

JEV calls one configured decision endpoint. `DECISION_API_FORMAT` selects the protocol for that endpoint.
The `decision/runtime.ts` sidecar and the direct HTTP path in `app/decision_client.py` read the same settings:

- `DECISION_API_FORMAT`: required. Accepted values are `chat_completions` and `decisions`.
- `DECISION_API_URL`: the full endpoint URL. Stockbot sends each `POST` to this URL verbatim and appends no path.
- `DECISION_API_KEY`: the credential. Stockbot sends it as `Authorization: Bearer <key>`.
- `DECISION_MODEL`: the one model for every existing JEV operation. These operations include entry routing, proposal disposition, tool selection, result assessment, and adjudication.

Set these values in the root `.env` to select the protocol, endpoint, key, and model. No code change is necessary.
The code supplies no format, URL, or model default. A live call fails when a value is blank.
A missing, blank, or unsupported `DECISION_API_FORMAT` fails before any HTTP request.
Stockbot never infers the format from the key, model name, host, or URL path.
Tests inject fake values.

### `chat_completions` format

Use this value for an OpenAI-compatible Chat Completions endpoint, such as `https://<provider-host>/v1/chat/completions`.
The request body contains `model`, `messages`, and `stream: false`.
One fixed system message carries the answer instructions.
One user message carries the JSON text of `{state, questions}`.
The request uses no tools, `response_format`, JSON Schema, or logprobs.

The client reads `choices[0].message.content`, which must be a nonempty string that contains a JSON object.
It rejects HTTP failures, absent choices, empty or nonstring content, refusals, and truncated output (`finish_reason: "length"`).
It parses the string and rejects malformed JSON or a parsed value that is not a JSON object.
The model reports noul probabilities, choice distributions, and confidence values in that JSON.
These values are model estimates, not calibrated measurements.

### `decisions` format

Use this value for a native Decisions API endpoint.
For example, set `DECISION_MODEL="typesafe/jev-1.13"`, `DECISION_API_FORMAT="decisions"`, and `DECISION_API_URL="https://openrouter.ai/api/alpha/decisions"` for that OpenRouter model.
These values are a configuration example only. The code does not default to them.
A Chat Completions request for this model returned HTTP 400.

The request body is exactly `{model, state, questions}`.
It contains no `messages`, system prompt, or `stream` field.
The response is a JSON object with a top-level `answers` object, `model`, and `usage` with `input_tokens` and `output_tokens`.
The client rejects HTTP failures and a response without a top-level `answers` object.
The provider supplies the probabilities. The client keeps the numeric probabilities and score legend unchanged.
It does not add new persistence fields or billing logic for other response fields, such as `id`, `provider`, or `usage.cost`.

### Shared validation and records

Both formats feed the same `{answers, model, usage}` result to the existing parsers. The `usage` field is optional.
The answer IDs must match the question IDs exactly. Each answer must match its question type, labels, and score range.
Errors use `decision_request_failed`, `malformed_decision_response`, `malformed_decision_answer`, and `decision answers do not match questions`.
A noul probability outside [0, 1] keeps the `invalid_probability` error.
The client makes no retry and uses no fallback model.
The existing thresholds, fan-out rules, and evidence gates stay unchanged for both formats.

Decision records keep the provider-reported model and token usage as `model`, `usage.input_tokens`, and `usage.output_tokens`.
Usage records include only integer token counts.
If the response omits the model or reports an invalid model, the record uses `DECISION_MODEL`.
The client never invents probabilities, confidence values, usage, or answers.
Errors and artifacts never include the API key.

The `JevClient` name and the historical event names stay unchanged. These names do not select a provider.
The Reasoner keeps its separate `OPENCODE_*` settings.

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
