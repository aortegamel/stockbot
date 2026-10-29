# Stockbot runtime: needle harness → kernel scheduler

Launch: `bun run stockbot` runs `bun needle-harness/scripts/needle.ts`,
which refreshes the tool catalog via `scripts/tool_bridge.py describe` and
starts three services: the Bun API host (`needle-harness/scripts/api.ts`),
FastAPI (`api.main:app`), and Vite. The Bun host dispatches `POST /api/agent`,
`GET /api/watchlist`, `GET /api/health`, and `GET /api/research/*` to the existing handlers. Vite
proxies those paths to Bun and the existing market `/api/*` paths to FastAPI.
For research requests, `needle-harness/app/api/agent/route.ts` runs `runKernelAgent`
(`needle-harness/lib/agent/kernel.ts`, prewarmed by the Bun host).
One `app/research/kernel_worker.py` call per research request persists the prompt
verbatim as the objective, runs the intake (SEC filing metadata + search
snippets per resolved ticker plus one exact-text Exa search, settled with
`continuation=False` so it admits evidence but never resolves), builds a
verbatim digest, runs Reasoner decompose over raw query + digest (optional
tickers/corrected_query hints feed at most one more intake round), then JEV
proposal disposition → per-proposal nodes, then `app/research/scheduler.py
run` over ready nodes. Each intake round logs one `intake_digest` decision
in the research database. The request contains the raw query and summaries of tool calls.
The response contains the intake digest and Reasoner output.
Readable `status` lines show live while the run works and collapse into the
Progress details block once the answer or a terminal event arrives.
`progress`/`tool_start` rows stay in the work summary; chat shows plain
phrases, debug views show raw names.
Every query also appends ordered `{timestamp, type, payload}` JSON lines to
`data/logs/agent-events/<UTC-timestamp>-<uuid>.jsonl` for post-query review.

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
An undated `list_sec_filings` call takes its arguments from the node's own question without Needle: the question's ticker or company, the 10-Q, 10-K, and 8-K names it contains, and the schema default limit.
A child node therefore lists its own company, not the first ticker of the session objective.
A calendar bound anywhere in the request (objective, node question, or raw scope: a year, a month name, or as of, before, after, since, until, between), a dated scope, or a cutoff keeps Needle generation.
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

Default chat uses one plain model answer with curated read-only research tools.
The model may call the six SEC EDGAR tools plus `search_web`, `get_current_time`, `get_material_events`, `get_financial_statements`, `get_analyst_estimates`, `get_valuation_metrics`, `get_insider_activity`, and `get_short_interest` through the bridge, for up to 8 rounds.
Chat sends reasoning effort `low`. `REASONING_EFFORT` does not change chat.
Before the model call, the route fetches four rows for the stock in view: 30 days of news with `search_web`, the 10 latest 8-K, 10-Q, and 10-K filings with `list_sec_filings`, `get_analyst_estimates`, and `get_valuation_metrics`. The feed-item ticker comes first, then the chart ticker.
At the same time, the kernel worker `sections` op asks JEV which investor parts fit the request: short term, long term, what analysts say, bull case, bear case, and risks to watch.
JEV answers one independent probability per part. A part at 0.50 or more stays. A failed or malformed JEV answer keeps every part.
Every chat decision is stored in one shared research session, `rs:chat` ("default chat"). The worker creates it on first use, so the sessions list shows one chat row.
The user message carries the system-clock date, the fetched rows, and the kept parts. A failed fetch leaves its section out.
The model writes the facts first, then each kept part under its label. It reports analyst targets and ratings as the analysts' view. It never says buy, sell, or hold, and it gives no price target of its own.
A model `get_sec_document` read returns at most 6000 characters. The tool default is 12000.
The model writes plain text in ASD-STE100 style and writes no URLs.
The route removes leftover Markdown emphasis, heading marks, and bullet marks.
The route sends one `evidence` event with every successful tool row. The UI lists their captured sources below the answer as clickable links.
A failed tool call returns an error to the model. It does not fail the chat.
Default chat runs no research, evidence inventory, or final gates, and JEV picks only the investor parts. The answer text is not checked against the filings.
The server handles three deterministic UI commands in `needle-harness/app/api/agent/route.ts`:

- Watchlist commands contain `add` or `remove` plus a watchlist request.
  Needle generates arguments for the fixed `update_watchlist` tool.
- Widget commands contain `add` followed by `widget` or `widgets` within the same sentence.
  Needle generates arguments for the fixed `add_widget` tool.
- Chart commands match the whole prompt, such as `show AAPL` or `open the chart for $nvda`.
  The server calls the fixed `set_chart_symbol` tool.

These commands do not use JEV or research.
The harness emits `{type: "chart_update", ticker}` only after canonical chart success.
The chart receipt does not enter factual evidence.

The client recognizes the `/research ` prompt prefix.
The research toggle also selects research mode.
Both send `mode: "research"` to `POST /api/agent`.
The client removes the prefix before it sends the prompt.
The server rejects other supplied mode values with HTTP 400.
Research always runs the full pipeline with Bearbot, Bullbot, and Stockbot.
It uses no entry selection or direct-tool shortcut.

During research, canonical chart success emits progress with `stage: "chart_update"` and `detail: {ticker}`.
The scheduler validates `result.meta.chart_selection` before it emits this progress.
It does not parse formatted chart text or infer success from the final execution list.
A chart receipt alone cannot resolve a substantive stock question.

Research intake uses the supplied ticker only when the objective names no stock.
An explicit objective stock takes precedence over this context.
Deterministic fallback arguments use the same subject precedence as Needle.

Research always uses the full trio with synthesis.
Bearbot and Bullbot draft concurrently over the shared snapshot.
Stockbot weighs both drafts against that snapshot and writes its response.
The stockbot synthesis starts as soon as both drafts settle.
At the same time, the final gates check each draft separately.
Each draft that passes its gates publishes at once: Bearbot first, then Bullbot.
An empty research graph makes no persona or gate call. The fixed line answers, and the limitations name `research retrieved no evidence`.
Exactly one failed or empty draft retries once.
If synthesis fails, times out, or loses every claim to the gates after a draft published, the run ends `completed_with_limitations`.
The limitations name the unavailable synthesis, and the run emits no `failed` event.
If no draft passes its gates and synthesis fails with a provider error, the run fails without prose.
If no draft passes and the gates reject every synthesis claim, the run answers the fixed line described below.

The kernel runs exactly one worker research run and one persona round.
An unresolved node, a tripped guard, or an escalation never starts more research.
The persona prompts never ask for a missing-evidence line.
The graph projection carries no open-items list or partial-coverage line, and no persona stance mentions gaps.
One exception exists: each failed tool call shows as a `LIMITATIONS` line with its tool, category, and error.
That line is untrusted and is not evidence. It has no citable ID.
The model states a failed retrieval only as an `unknown` claim and never states what that source contains.
A failed retrieval never becomes an evidence row.
Otherwise, an answer never announces missing evidence: what the run did not find is left out.

Final gates check the claim inventory and the whole response.
The kernel checks all personas and all of their claims concurrently and processes the results in persona order.
The gate requests carry the research session id of the run they judge.
A gate verdict whose `error` starts with `proof:` is a deterministic rejection of its claim.
Any other gate error (a JEV transport failure, a deadline) means the gate is unavailable: the claim counts as a gate failure, never as a rejection.
A `support_answer` of `yes` with `evidence_state: "insufficient"` rejects the claim.

When the gates reject every claim, the kernel publishes one fixed runtime-written line: "I couldn't determine this from the retrieved sources."
No model text appears in that answer. The run ends `completed_with_limitations`, and the limitations include the rejection message.
A run still fails without prose when a gate outage, provider error, or timeout leaves nothing to publish.
If the whole response fails, the gates check each claim.
The kernel removes each rejected claim substring.
Other prose stays unchanged, including uncertainty.
Node IDs and decision records never count as evidence IDs.

An inventory with an unknown evidence ID or an unknown datapoint ID retries once for each affected persona on the same evidence snapshot.
An inventory whose `claims[].text` is not a verbatim prose substring uses the same retry.
This local check is a pre-filter only. A local pass never counts as proof.
The correction states valid evidence IDs and requires exact evidence row IDs and verbatim claim text.
The correction includes rejected prose and inventory as untrusted output.
The corrected answer passes the same gates.
A still-invalid, empty, or failed correction gets no further retry. If nothing survives, the fixed line answers.
Other inventory defects do not retry.
In the trio, a gate failure or a fully rejected persona removes only that persona's block.
When no persona block survives, the fixed line answers. A gate outage or provider failure still fails without prose.
The claim inventory is the last `json` fenced block of a response.
The kernel drops any text after its closing fence.
An unclosed (cut-off) block never shows raw JSON: the prose ends at the fence.

An unresolved node does not emit `failed` when the final gates pass.
A research answer with no cited claim ships its gated prose with no warning banner.
The trio emits one `answer_delta` per published persona block, each followed by a cumulative `claims` event.
The reasoner returns narrative prose plus declared claims. The claims stay unproven until the gates keep them.
The reasoner's raw stream (`onDelta`) is internal diagnostics only, because it carries the inventory fence.
The kernel and the route pass no stream callback, so the client sees only the gated text and the kept claims.
Metrics count all attempted calls and token usage across retries.
Metrics also report the final stage duration.
Runtime model prompts retain persona tasks, required response formats, and untrusted-data protections.
They no longer impose advice bans, authority bans, evidence-only reasoning, word targets, or a prose outline.
Required JSON shapes, claim inventories, tool arguments, identifiers, and date formats remain.
Existing code validation, risk checks, authorization, and execution limits remain unchanged.
The configured provider can still add a disclaimer.
Muse receives all projected evidence without an aggregate character cap.

Research wall: the 120-second `REQUEST_WALL_MS` deadline applies to the worker research run.
The run starts its wall at route entry, or at the caller `deadlineAt`.
Its worker call timeout is the time left on that wall.
Persona generation, the draft retry, Stockbot synthesis, and the final gates never receive the wall.
Each Muse call keeps the client's 120-second per-call fetch limit. Each gate call keeps the worker's 10-minute call timeout.
A run whose research used most of the wall therefore still writes and gates its answer.
The caller cancellation signal still reaches every persona call and aborts it.

Output caps (experiment): two optional env knobs cap Muse output tokens.
`MUSE_MAX_OUTPUT_TOKENS` caps each call. `MUSE_ANSWER_TOKEN_BUDGET` caps all Muse calls of one request (brief, personas, retries).
Each call sends `max_output_tokens = min(per-call cap, remaining budget)`.
The remaining budget drops by each response's reported output tokens.
Concurrent trio drafts read the same remaining budget, so together they can exceed it.
Under 16 tokens left, the next call fails before any request: the provider rejects `max_output_tokens` below 16.
Unset knobs send no cap. The chat-completions fallback sends no cap.
The provider counts reasoning tokens inside `max_output_tokens`, so a low cap can return little or no prose.

Reasoning effort: `REASONING_EFFORT` defaults to `auto` when unset or empty.
In auto mode, JEV chooses `low`, `medium`, `high`, or `xhigh` before each research run.
The choice uses the prompt, scope, remaining deadline, and answer limits.
Only the research Muse answers use that choice, including persona drafts, synthesis, retries, and protocol fallbacks.
JEV, the Reasoner, and default chat send no effort field in auto mode.
An effort selection error fails the request with `EFFORT_SELECTION_FAILED` at stage `effort`. Stockbot logs the selected effort.
A concrete value applies unchanged to every remote model call. It must match `^[a-z][a-z0-9_-]{0,31}$`.
A bad value fails before any request. Responses sends `reasoning: {effort}`. Chat Completions sends `reasoning_effort`.
The Decisions API documents no reasoning field, so the `decisions` format sends none.
`decision/reasoning.ts` and `app/config.py` `reasoning_fields` own this mapping.

Startup prewarm: the Bun host waits up to 120 seconds for the kernel worker `ready` message.
At 120 seconds, prewarm kills the silent worker and logs the failure. The API host keeps serving.

Before readiness, the worker handles `SIGINT` and `SIGTERM` without acquiring the Needle lock.
The worker unwinds startup before its exit handlers remove owned child processes.
During child creation, the worker defers termination until the child enters its owner's registry.
This deferral does not change the child's signal mask.

A POSIX watcher detects caller disconnection without reading stdin.
The watcher preserves queued requests and stops signaling once startup completes.
Normal request handling and post-ready shutdown remain unchanged.
An explicit `SIGKILL` cannot run this cleanup.

The kernel scheduler owns the agent loop.
JEV selects necessary operations from the whole registry based on the objective and current evidence.
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
On a watchlist request the same list accompanies `op:"run"`. JEV selects the necessary operations for that scope.
Live JEV behavior for a watchlist prompt is unverified until a live run checks it; the mocked route replies only prove the context and the dispatch wiring.
Every other prompt keeps the selected-ticker behavior and ignores the list, so a selected-ticker question never becomes watchlist-wide.

The watchlist panel loads sources through `GET /api/watchlist?symbols=...` once per watchlist.
The worker `op:"watchlist"` uses a separate thread. It creates no session, job, or evidence.
Research questions use the scheduler instead of this fixed source fan-out.

Each symbol gets four fixed-argument calls through the existing tool handlers.
`list_sec_filings` uses `limit=5` and no forms filter, so fund forms stay in.
The three FINRA calls are `get_short_interest`, `get_reg_sho_volume` (the handler's current NYC week), and `get_threshold_securities`.
Every call is an independent task on the existing SEC and tool pools under the 8s intake budget.
A call past the budget is cancelled and reported as a timeout.
Each (symbol, source) row carries `status` (`ok`, `no_records`, `error`, `timeout`), `as_of`, source references, and the evidence ID when admitted.
A FINRA "No data found" answer is `no_records`. Any other failure is `error`.
Research actions persist their objective, purpose, navigation, and assessment in the existing decision journal.
The optional `coverage` response reports SEC document availability and inspection separately from evidence.
Reuse follows the existing caches: FINRA results use the `app.cache` 3600s TTL.
SEC submissions use the edgartools HTTP cache with its 30s revalidation.
A fresh entry serves the agent request without source work. An expired entry refreshes through the same path.

Research starts with one objective node. It does not automatically search SEC and web sources or decompose the objective.
JEV can request reasoning through the existing analyze, adjudicate, and expand stages.
Only explicit JEV admission creates a proposed follow-up node.
Missing-evidence children run before their parent. True refinements can depend on the parent.
The existing dependency gate controls both paths.

The scheduler admits a relevant passage only after canonical source proof succeeds.
The result assessment must explicitly report topical relevance as `yes`.
Resolution support controls continuation. A passage need not resolve the entire objective to enter the ledger.
Catalog records remain navigation. They do not establish disclosure content.
Per-claim assessment checks ledger identity, archived bytes, session ownership, point-in-time availability, numbers, quotations, and qualifiers.
JEV then judges qualitative paraphrases against canonical ledger passages.
Standalone finalization retains its stricter deterministic wording gate.
Canonical numerical claims with complete deterministic proof need no repeated arithmetic judgment from JEV.

Each worker operation receives the original `deadlineAt`. No stage renews that deadline.
Cancellation stops the current client operation instead of the shared worker.
Admission uses the existing SEC pool. The service checks the owning job again immediately before an evidence commit.
A timeout preserves admitted evidence and reports its stage.
The journal retains completed navigation and attempts for further research in the same session.

SEC coverage distinguishes listed, located, retrieved, inspected, and admitted content.
It also reports incomplete and unavailable documents.
Inspection covers the returned document window. A retrieved window does not establish a complete filing read.
Coverage limitations are not citable evidence.


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

The argument worker calls native `Needle.complete()` with the node question and grounded execution handles in user text.
It keeps the SDK native `max_new_tokens=512` generation default.
For accession-based tools, the scheduler selects the handle from persisted same-session filing rows with issuer, form, and cutoff checks.
The worker rejects a missing accession before generation.
For discovery, a named SEC form enters the user text with the requested count or declared default count.
Form numbers never supply a record count.
It binds exactly the tool that JEV selects.
The catalog preserves the original description and parameter schema.
Structured extraction keeps its native 512-token default and 1-4096 validation bound.

The argument SDK receives no research evidence, raw tool results, or attempt history.
Its context contains temporal anchors, the selected ticker, and explicitly grounded argument values.
The server filters context fields before it binds the selected tool.
Fresh nodes exclude `research_read` and `research_read_search` because they lack saved resource handles.
If Needle rejects `search_web` through a tool mismatch or grounding error, the scheduler uses the node question.
It removes a leading `[Today UTC YYYY-MM-DD]` instruction before it trims whitespace and truncates the query to 200 characters.
It preserves dates within the question.
A blank or non-string question remains invalid.
The scheduler makes no additional generation call.
One deterministic Needle failure excludes its tool for the next round when another tool remains.
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
For named 10-Q, 10-K, or 8-K requests, the scheduler supplies that form filter.
Other generated form filters remain unchanged.

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

## Numeric datapoints

A number in a published claim needs a verified datapoint: a ledger row whose proof rebuilds `{metric, value, entity, period, unit}` from the archived source bytes. One row holds at most one datapoint. Two paths create them.

Structured fan-out: after the scheduler admits a FINRA or SEC structured record, it admits one datapoint row per numeric field of the cited row, up to 12 fields.
A FINRA single-ticker envelope cites its `latest_record`: the newest raw source row, scalar fields only.
An XBRL fact row binds its concept to its value. Entity and period come from the row's own fields.
Envelope numbers such as counts and limits never become datapoints. A field the proof rejects stays absent.

Claim-time admission: a persona claim that states a number without an existing `dp:` id declares it in `numbers: [{evidence_id, metric, value, entity, period, unit}]`.
Before the final gates, the kernel sends each declaration to the worker op `admit_datapoint`, at most four at a time.
The worker rebuilds the cited row's admission item, admits the datapoint under a new job, and returns the descriptor.
An SEC document row stores its original `source_handle` (with section and query) in `metadata.source_handle` at admission, so claim-time admission reuses the exact window. Rows admitted before this change carry no handle and fail closed.
The worker rejects a returned descriptor whose fields differ from the declaration. A failed admission leaves the claim unchanged, and the gate rejects its number.
Persona prompts write each number as its own sentence: `<entity> reported <metric> of <value> for <period>.` The sentence names no data source and adds no words outside the datapoint fields, because the gate rejects them.

Table periods: a statement table whose duration header (`Three Months Ended | Six Months Ended`) spans groups of value columns binds a period such as `Three Months Ended Jul 26, 2026` to the value in that group and date column. A repeated bare date stays ambiguous and fails closed.

Repeated values: a value that appears more than once in a passage (`Net revenue` and `Total net revenue` rows with the same figures) binds the one occurrence whose row or sentence matches every claimed label. When no occurrence matches, the proof fails. When two or more match, it stays ambiguous and fails closed.

Growth rates: a persona states a growth rate as a `calculation` claim. Its `numbers` entry adds `formula: "percent_change@2"` and `inputs: [current, prior]`, with unit `%` and period `<current period> versus <prior period>`. Each input is an existing `dp:` id or an inline observed declaration, which the worker admits first through the observed guards. The worker then admits a calculation row. The ledger recomputes the value (one decimal). It checks that the inputs share one entity, metric, and unit, name two distinct periods, and that the output period names both input periods. A value that differs from the recompute fails closed. Every declared label keys observed and calculated rows, so a second source spelling becomes its own datapoint and never a mismatched duplicate.

In the gate, an entity spelled with a trailing period (`Advanced Micro Devices, Inc.`) does not end its sentence.

A question that names two or more companies starts one research node per company, `<question> (<TICKER> only)`, instead of one root node. The `(TICKER only)` suffix is that node's issuer. Other nodes resolve the filing issuer from the node question first, then from the session objective.

A datapoint never drops a scale the source states. When the declaration has no unit but the value's statement carries a scale word (`$46.7 billion`) or an earlier heading states one (`(In millions, except per share data)`), the proof fails closed. A declared scale unit may be singular: `million` binds the source's `millions`, and the stored unit keeps the source spelling. Personas write `$<value> million` with the scale word singular after the value. The worker compares the stored descriptor with the declaration without regard to case.

`get_financial_statements` returns the whole statement as `label | <period> ...` rows with whole-number values, never pandas' truncated display. It also returns `line_items`: one `{ticker, period, unit, <label>: value}` row per consolidated line item and period column. Breakdown and abstract rows are dropped. The unit comes from the line item's own XBRL facts: `USD`, `USD/share`, or `shares`. A concept with several units, or another unit, has no unit field. The scheduler fans out a datapoint for each line item, up to 60 rows, so these numbers are provable and a growth rate over two years of one line item recomputes. A claim states the unit: `$215938000000`.

A published answer drops a citation-only fragment (`2026. [ev:1]`) together with the sentence before it, so a rejected sentence never leaves a bare citation behind.

Readable display: the gates check the raw source figure, and the reader sees a rounded one. After every gate, the kernel rewrites displayed answer and claim text the same way, so citations still find their claims. A value followed by a scale word (`$96,221 million`) becomes `$96.2 billion`. A whole number of 7+ digits becomes `$215.9 billion` or `69.4 million` only when it equals a verified datapoint value of the run. A camelCase metric label of a datapoint reads as words (`current short position quantity`). Text inside citation brackets, CIKs, accession numbers, dates, and all other numbers stay verbatim. Values round to one decimal, and a trailing `.0` drops.

A growth claim that also lists its inputs' `dp:` ids drops each input whose value the sentence does not state. A `dp:` id that no row of the run carries (a mistyped id) also leaves the claim, and every stated number still needs a bound datapoint at the gate. The gate ignores the article "the" when it matches an entity or period, so `for the Three Months Ended ...` binds like `for Three Months Ended ...`. A growth period may differ from its inputs' periods in spacing and "the". Plural form names (`10-Qs`) are names, never numbers. An unknown claim may say `do not establish`. The worker ignores inner spacing when it compares a declaration with the stored descriptor, and a padded declaration still differs. The kernel logs each claim it drops before the gates (`[ai] <persona> claim dropped before gates: <reason>`) and each withheld persona answer (`[ai] persona answer withheld: <reason>`).

Claim alignment: the kernel removes only claims that are wrong, not claims written in another spelling. Before any check, a `claims[].text` that is not an exact prose substring maps to the prose span it names. Spacing, quote style, case, citation brackets, and trailing punctuation do not matter. Failing that, the prose sentence that shares most of the claim's words (Jaccard at least 0.6) and exactly its numbers stands in. The gates then judge that published prose text. A claim with no such match names nothing the answer says, and it drops. A claim that cites a real `dp:` id but not its row gains that row, and the gate still proves the datapoint there.

Correction loop: a persona whose gated answer lost anything gets up to 3 correction rounds. Each round lists every rejected claim of the best answer so far with its reason: the local inventory defect, the worker's `proof:` error, or the evidence judge's verdict. It also lists every sentence that no claim covered. Under each rejection the round shows a concrete rewrite the gates accept instead of describing one:

- an unbound number: the exact number sentence built from the datapoint's own fields (`NVIDIA Corporation reported Revenue of $96,221 million for Three Months Ended Jul 26, 2026.`);
- an unqualified claim: the claim wrapped in its qualifier (`This suggests …`, `The evidence does not establish whether …`, `Assuming …`);
- an uncovered sentence: the claims row to add, with the sentence and its cited ids;
- a bad id: the valid EVIDENCE ids.

A rejection kind without a mechanical rewrite (stray numbers, wrong claim kind, the judge's verdict) gets a worked before/after example. The best answer so far rides beside the list as untrusted text. The kernel admits the corrected answer's numbers and gates it again. A corrected answer becomes the best only when it publishes more verified claims, or the same number with fewer rejections. The loop ends when nothing is rejected, when a round does not improve, after 3 rounds, or when a correction call fails. A gate outage never starts the loop. The kernel logs `[ai] <persona> correction round <n>: <before> -> <after> verified claims, <before> -> <after> rejections, kept|discarded`.

A judge rejection names the judge's evidence state: contradicted, conflicted, or too weak for the claim as written. The kernel logs every rejection as `[ai] <persona> claim rejected: <reason> | <claim text>`.

Unknown claims never reach the judge. No evidence can show that something is absent, so a claim declared `unknown` passes once its wording carries the unknown qualifier (`not found`, `unclear`, `does not establish`, ...) and it states no unproven number (a year is allowed).

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
- `DECISION_API_URL`: the full HTTPS endpoint URL. Stockbot sends each `POST` to this URL verbatim and appends no path.
- `DECISION_API_KEY`: the credential. Stockbot sends it as `Authorization: Bearer <key>`.
- `DECISION_MODEL`: the one model for every existing JEV operation. These operations include entry routing, proposal disposition, tool selection, result assessment, and adjudication.

Set these values in the root `.env` to select the protocol, endpoint, key, and model. No code change is necessary.
The code supplies no format, URL, or model default. A live call fails when a value is blank.
A missing, blank, or unsupported `DECISION_API_FORMAT` fails before any HTTP request.
Both transports reject a non-HTTPS `DECISION_API_URL` before network access, including HTTP on loopback.
Both transports reject every non-HTTPS redirect before the next request, including loopback HTTP.
Relative redirects resolve against the current HTTPS URL.
urllib follows POST 301/302/303 as GET.
It rejects POST 307/308.
Fetch follows POST 301/302/303 as GET and preserves POST bodies for 307/308.

Fetch removes Authorization on redirects to another origin.
urllib retains its native HTTPS redirect headers.
Python retains urllib's redirect-loop limits and the caller's total timeout.
Fetch permits at most 20 redirects within one shared 10-second timeout.
Stockbot never infers the format from the key, model name, host, or URL path.
Tests inject fake values.

### `chat_completions` format

Use this value for an OpenAI-compatible Chat Completions endpoint, such as `https://<provider-host>/v1/chat/completions`.
The request contains `model`, `messages`, and `stream: false`, plus `reasoning_effort` for a concrete `REASONING_EFFORT` value.
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
The sidecar (`decision/client.ts`) retries once when `fetch` itself throws, such as on a connection reset, under the same 10-second timeout.
An HTTP error status never retries. The direct Python HTTP fallback makes no retry. No path uses a fallback model.
Both formats use the same support policy. Fan-out rules and deterministic evidence gates remain separate.

Noul parsers keep the raw probability. Policy normalization adds `{outcome, probability, strength}`:

- `0.70 <= p <= 1`: `outcome: "yes"` with `strength: "confident"`.
- `0.50 <= p < 0.70`: `outcome: "yes"` with `strength: "tentative"`.
- `0 <= p < 0.50`: `outcome: "no"` with `strength: null`.
- Missing, invalid, or unassessable values: `outcome: "unknown"`, `probability: null`, and `strength: null`.

Normalization preserves each valid numeric probability. Unknown support never becomes negative support.
Raw `choice`, `noul`, and `score` results keep their existing shapes.
Malformed transport responses still produce explicit errors. Normalization does not suppress those errors.
Final assessments expose `support_result` and `support_strength` beside the existing `support_answer`.
Result assessments expose the corresponding `resolution_support_*` and `topical_relevance_*` fields.
Both affirmative strengths produce `yes` for existing answer consumers.
Evidence insufficiency and conflict remain separate from probability.
Positive JEV support cannot override source identity, content hashes, point-in-time checks, or numeric proof.
JEV support never authorizes an external action without user approval.

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
The web project pattern includes CSS files, so Knip resolves CSS imports.

Knip treats `uv`, `poly-crap`, `gitleaks`, `osv-scanner`, and
`llama-server` as external executables. Package dependencies do not provide
these executables.

The launcher starts the Bun API host by its absolute path
(`needle-harness/scripts/api.ts`). The working directory stays `needle-harness`.
Kernel types without outside users stay local to `kernel.ts`.

The project uses one root `venv/` with uv-managed Python 3.14.
`bun run setup` reads `.python-version` and installs all requirements, including `cactus-needle`.
Stockbot and Needle use `venv/bin/python` without a system-Python fallback.
Docker runs the same setup in `/app`.

The guided SEC accession errors carry the tool name in a `tool` key. The error
text and self-correcting hint stay unchanged. The tool-health gate reads this
key for its missing-argument checks.

The clock tool reads time through a `_utc_now` helper. The tool-health gate
doubles this helper with a fixed UTC instant. Production output keeps its
`utc_now` and `source` fields.

FINRA freshness tests freeze the local date that the FINRA client reads.
They check records at 90 and 91 days, including a year boundary.
SEC facts tests freeze the UTC clock.
They check that an omitted `as_of` uses the UTC date when the local date differs.

Scheduler scope tests and the intake subject test replace company resolution with local doubles.
They require no SEC lookup.

Caller-disconnect tests run real workers with provider stubs.
Coverage must collect child processes through `COVERAGE_PROCESS_START`.
Use branch and parallel coverage for both parent and child processes.
Parent-only coverage does not measure the watcher.

The live judge runs each scenario through `create_research`, `create_node`,
and `run_node` in an isolated per-attempt store. JEV, tool invocation, and the
evaluator all use that store. The evaluator reads session status, final
results, evidence, and tool results from the same store. Attempt errors persist
in the result record.

Root TypeScript uses the classic 5.x compiler. The Stryker checker requires
its classic API. The `mutation:ts` script runs the mutation gate.

## Python type integrity

Public functions keep strict parameter and return types. Private guards
accept `object` and narrow it with `isinstance` checks or `TypeGuard`.
Malformed runtime values reach the existing error path. The rules forbid
casts, type-ignore comments, and weaker Pyrefly rules.
Tests that need malformed repository rows use a repository double that
returns those rows. The tests do not cast them.

Source resolution in `app/research/sources.py` types the injected store
with a private structural `Protocol` and a `TypeGuard`. A store qualifies
when it has callable `get_evidence` and `get_tool_result` methods.
`ResearchRepository` and test doubles both qualify. Other values fall back to
a `ResearchRepository` for the data root.

`pyrefly.toml` sets `python-interpreter-path = "venv/bin/python"`. Pyrefly
resolves this relative path from the directory of `pyrefly.toml`. The CLI
therefore resolves imports from the repository venv, not from the
`python3` on `PATH`. The language server reads the same configuration.

pandas ships no `py.typed` marker. `requirements.txt` therefore pins
`pandas-stubs`. Pyrefly reads the pandas types from these stubs in the
repository venv. The stubs do not change runtime pandas behavior.

## Mutation sandbox

The `source_paths` list in `setup.cfg` names the mutated sources:

- `app/`
- `api/market_providers.py`
- `calculations/`
- `scripts/tool_bridge.py`
- `needle-harness/lib/needle/server.py`

mutmut accepts directories and individual files in this list. It copies each
source into `mutants/`. It then copies `tests/`, `setup.cfg`, and the
`also_copy` list. Last, it writes the mutants over the copied sources.
A source that also sits inside an `also_copy` directory therefore receives
mutants. The other files in that directory stay unmodified support copies.
For example, `api/market_providers.py` receives mutants, but `api/main.py`
stays a support copy.

mutmut associates tests through a function's module name. The server test
loader uses `needle-harness.lib.needle.server`. This name matches the
server's path-derived key. The production subprocess still uses `__main__`,
which does not select named mutants.

mutmut 3.8.0 caches the mapping from functions to tests.
Its configuration fingerprint excludes `source_paths`.
Its incremental collection runs only new test IDs.
Existing tests can therefore leave new modules without recorded calls.

Refresh this mapping after changes to `source_paths`, loader module names,
or the functions that existing tests call:

1. Wait until every mutation process stops.
2. Rename `mutants/mutmut-stats.json` to a unique backup name.
3. Keep the `.meta` files to preserve completed verdicts.
4. Run the affected mutants by their exact keys.
5. Check the new module's function keys before trusting a `NoTests` result.

The next run collects fresh statistics when the statistics file is absent.
New statistics alone do not rerun cached `NoTests` verdicts.

The `also_copy` list adds the other files that the collected suite reads:

- `scripts/`: tests import `scripts.*` modules. `tests/security/test_sandbox.py`
  runs `scripts/host/pi-opencode-go-secret`.
- `api/`: `tests/test_market_providers.py` imports `api.main` and
  `api.market_providers`.
- `calculations/`: `tests/test_calculations.py` imports the package.
- `cli.py`: several tests import `cli`.
- `conftest.py`: the root fixture isolates `RUNS_DB_PATH` and
  `STOCKBOT_DATA_DIR` in temporary directories.
- `pytest.ini`: it sets `pythonpath`, `testpaths`, the default marker
  filter for live tests, and the marker names.
- `pyrefly.toml`: `tests/test_verify_type_escape_hatches.py` reads its
  `project-includes`.
- `docker-compose.yml`, `sandbox/stockbot/`, `package.json`, `.omp/`:
  `tests/security/test_sandbox.py` reads them. `tests/test_coverage_scripts.py`
  runs the `package.json` scripts.
- `evals/`: `app.research.evals.regression` and
  `scripts/strict_routing_harness.py` read its fixtures.
- `needle-harness/lib/`: `tests/test_exa_agent.py` reads `muse/client.ts`.
  `tests/test_needle_client.py` loads `needle/server.py`. The web sources
  import `agent/types.ts`.
- `decision/`: `tests/test_jev_start.py` runs `decision/runtime.ts` with Bun.
  The runtime imports only its sibling files and Node built-in modules.
- `web/src/`, `web/index.html`, `web/package.json`, `web/bun.lock`,
  `web/tsconfig.json`, `web/vite.config.ts`: the browser test runs Vite in
  `mutants/web`. `web/src/` comes first because mutmut needs the parent
  directory before it copies individual files.

The list copies support files only. It filters no tests and excludes no
mutants.

Install the sandbox web dependencies from the copied lockfile before mutation:

```bash
venv/bin/python -c 'from mutmut.utils.file_utils import copy_also_copy_files; copy_also_copy_files()'
bun install --frozen-lockfile --cwd mutants/web
SEC_EDGAR_IDENTITY='Stockbot tests tests@example.invalid' venv/bin/mutmut run
```

The baseline resolves a ticker through EDGAR's bundled index.
The test identity permits that offline lookup. Do not copy the root `.env`
into the sandbox.

Do not copy `web/node_modules` through `also_copy`. The copy operation
dereferences its executable links, which breaks the Vite entrypoint.
The separate installation preserves those links and isolates Vite's cache.

### Survivor repair scope

The repair uses saved `Survived` verdicts outside `calculations/`.
Pending, missing, and `NoTests` cases remain unverified.
The calculations checks cover the named portfolio risk, valuation, execution,
performance, and fundamentals functions. The application imports no functions
from `calculations.factors` or `calculations.technicals`.

`tests/test_calculations.py` checks known answers, defined input boundaries,
and fixed-seed properties. These tests preserve the existing formulas and units.
The repair adds no tests to `*_crap_paths.py`.

Before a focused mutation run, preserve the existing metadata and test map.
Collect a fresh test map for the affected normal test modules.
Run exact mutant keys from one source module.
Keep equivalent and excluded mutants in the report.
Report retained verdicts separately from new verdicts.
Keep mutation and CRAP thresholds unchanged.

### TypeScript mutation

Use the existing decision configuration for decision sources:

```bash
bunx stryker run --mutate decision/jev.ts
```

Use the harness configuration for harness sources:

```bash
bunx stryker run stryker.harness.conf.json --mutate needle-harness/lib/agent/kernel.ts
```

Both configurations retain the TypeScript checker. The harness configuration
uses `needle-harness/tsconfig.json` and runs the existing Bun suite from that
directory. It lists production files explicitly, without negative globs.
Add new production files to its `mutate` list.

Stryker reads every file below the repository root as campaign input. The
harness `ignorePatterns` list removes generated content from that input:

- `/venv`, `/mutants` (the mutmut sandbox), `/data`, and `/reports`
- `/coverage-harness`, `/coverage-ts`, `/coverage-python.lcov`, and
  `/.coverage`
- `/.hypothesis`, `/.pytest_cache`, `/.ruff_cache`, and `__pycache__` at any
  depth
- `/web/public/pdfjs`, the output of `scripts/copy-pdfjs-assets.ts`

A leading `/` anchors a pattern to the repository root. An unanchored pattern
matches that name in any directory. The list excludes no application source,
no test, and no `mutate` target. Stryker skips the tracked runtime file
`data/mandate.json` during the scan. The `inPlace` run keeps excluded files
on disk. Tests and the checker still read them.

Anchor each new pattern. Do not exclude source or tests.
The decision configuration has no input exclusions.

The harness command reads the test scope from `STRYKER_TEST_SCOPE`. An unset
value runs the whole harness suite. Two harness test files dominate the run
time: `scripts/api.test.ts` (50s) and
`app/api/research/[...path]/route.test.ts` (29s). The other eight files run in
about 3s together.

A scoped run trades time for certainty. A mutant can survive a scoped run and
die in the full suite, because the excluded tests can kill it. Confirm every
survivor of a scoped run with an unset `STRYKER_TEST_SCOPE`:

```bash
bunx stryker run stryker.harness.conf.json --concurrency 6 --timeoutMS 300000 \
  --reporters clear-text,html,json --mutate '<survivor ranges>'
```

Stryker accepts `file:startLine-endLine` ranges in `--mutate`. Merge the
survivor ranges per file before the confirm run. The `timeoutMS` value covers
the two slow files. Without it, Stryker derives a short timeout and reports
slow mutants as `Timeout`.

The harness writes `reports/mutation/harness.html` and `harness.json`. It does
not replace the decision report at `reports/mutation/mutation.html`.

#### Restore after an interrupted run

Both configurations use `inPlace`. Stryker writes instrumented code and
`// @ts-nocheck` comments into the working files. Stryker keeps the originals
in `.stryker-tmp/backup-<id>`.

If the process stops before cleanup finishes, backup files can remain.
A reset log line alone does not prove that Stryker restored every file.

Before any new Stryker run, restore the files. A new run saves contaminated
files as its originals. Before the normal checks, restore the files.
A `// @ts-nocheck` comment hides type errors, and instrumented code changes
what the tests run.

If you edited files after Stryker started, save those edits before you restore the backup.

1. Check that no Stryker process runs.
2. Find the backup directory in the `Find your backup at` log line.
3. Copy the backup files into the repository root.
4. Compare every restored file with its backup.

```bash
backup='.stryker-tmp/backup-<id>'
cp -a "$backup"/. .
(cd "$backup" && find . -type f) | while read -r f; do cmp "$backup/$f" "$f"; done
```

An empty `cmp` output means that every file matches its backup. Do not use
`git restore`, `git checkout`, or `git reset`. They discard uncommitted source
changes. Run the normal checks only after the restoration.

## Failure cleanup and terminal reads

If graph setup fails after session creation, the worker cancels the open
bootstrap source job. The worker keeps the session record. Cleanup errors
appear in the worker log and do not replace the original result or error.

The terminal clears its snapshot, journal, and sequence when the selected
session changes. It ignores responses from the previous session.

TradingView owns the timeframe and chart-style controls inside the embedded terminal chart.

The research API uses an asynchronous Python subprocess with a 15-second
timeout. Unknown sessions return HTTP 404. Import, database, execution, and
bridge response errors return HTTP 502.

The Bun API and the market service always bind `127.0.0.1`
(`needle-harness/scripts/api.ts`, `needle-harness/scripts/needle.ts`).
The Vite listener binds `0.0.0.0` by default (`STOCKBOT_LAN=0` keeps
`127.0.0.1`); `STOCKBOT_LAN_TOKEN`, when set, is required on every `/api/*`
request (`web/lan-token.ts`). See the README "LAN access" section.
