# Stockbot runtime: native research tools and ledger proof

Launch: `bun run stockbot` runs `bun needle-harness/scripts/needle.ts`.
The launcher refreshes the tool catalog through `scripts/tool_bridge.py describe`.
It starts three services: the Bun API host (`needle-harness/scripts/api.ts`), FastAPI (`api.main:app`), and Vite.
The Bun host dispatches `POST /api/agent`, `GET /api/watchlist`, `GET /api/health`, and `GET /api/research/*`.
Vite proxies those paths to the Bun host.
It sends the existing `/api/*` and `/api/market/*` market paths to FastAPI.

`GET /api/market/watchlist` loads the saved list.
`PUT /api/market/watchlist` saves `{symbols}`.
The selected ticker stays visible without entering the saved list.
The shell persists validated watchlist receipts even when no watchlist panel exists.
`GET /api/market/status` supplies `running`, `last_cycle_at`, and `oldest_as_of` in epoch seconds.
The header shows the oldest timestamp in local time as `as of HH:MM`.
Its Refresh button sends `POST /api/market/refresh` and polls status every three seconds.
When collection completes, panels read the stored data again.
Quotes, candles, news, and filings retain their data when `stale` is true.
Their `as_of` fields contain ISO timestamps.

Research requests use `runKernelAgent` in `needle-harness/lib/agent/kernel.ts`.
The kernel calls `app/research/kernel_worker.py` through its shared worker.
The native model selects tools and supplies their arguments.
The scheduler does not select tools for this path.
Research uses no intake decomposition, catalog selection, result assessment, or graph expansion.

The native sequence is:

1. `open_research` stores the objective and creates its session.
   It returns the session ID and the allowed research tool names.
   It runs no model or source tool.
2. The kernel matches those names to canonical tool schemas.
   The model requests tools through native function calls.
3. Each `research_tool` request executes one allowed tool through `app/research/tool_runner.py`.
   The runner creates a source job and calls `app/tool_runtime.py`.
   The service admits evidence through deterministic structural checks.
4. The model receives the tool summary and admitted evidence rows.
   It can request more tools before it writes its answer.
   Independent calls in one model turn execute concurrently.
5. The kernel admits declared numbers through `admit_datapoint`.
   It checks the claim inventory and calls `prove_claims` for valid claims.
6. The kernel removes unsupported text from the final answer.
   It emits verified claims and replaces the displayed answer when necessary.

The model writes one answer with Bearbot, Bullbot, and Stockbot labels.
These labels do not start separate agents, drafts, synthesis calls, or correction rounds.
The model's prose streams before final proof.
The inventory fence does not stream to the user.
The final `answer_partial` event replaces provisional text when proof removes content or number formatting changes it.
An answer without surviving claims returns the fixed undetermined answer with `completed_with_limitations`.
A proof outage fails the request instead of declaring its claims false.

The tool allowlist excludes discovery, session controls, chart controls, watchlist updates, and widget creation.
`open_research` advertises the same set that `research_tool` accepts.
SEC document calls use a maximum 12,000-character window in native research.
Tool failures return errors to the model and create no evidence rows.
Evidence summaries use at most 1,200 characters per row and 8,000 characters overall.
The returned evidence rows retain their stored content.

The request uses the original `deadlineAt`, or a 180-second wall when the caller supplies none.
The kernel passes that deadline to session setup, source calls, model generation, and claim proof.
Cancellation reaches the model and the active worker request.
The worker runs tool, proof, and datapoint operations on four threads.
It processes watchlist reads on a separate thread.

Readable status lines show during research.
The interface retains tool activity in the Progress details.
Each query also writes ordered `{timestamp, type, payload}` records to `data/logs/agent-events/<UTC-timestamp>-<uuid>.jsonl`.

## Default chat


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
Research uses the native sequence described above.
It uses no entry-routing operation or direct-tool shortcut.

The worker accepts these operations:
`open_research`, `research_tool`, `prove_claims`, `watchlist`, `effort`, `sections`, `scope`, `arguments`, and `admit_datapoint`.
It rejects retired `run`, `route`, and `assess_final_claims` operations.
The `arguments` operation remains available for chat commands that use Needle.

Output caps remain optional.
`MUSE_MAX_OUTPUT_TOKENS` limits each model call.
`MUSE_ANSWER_TOKEN_BUDGET` limits the answer's model calls, including tool turns.
Each call uses the smaller available cap.
The provider counts reasoning tokens within that cap.
An exhausted budget fails before another call.
The chat-completions fallback sends no output cap.

`REASONING_EFFORT` defaults to `auto` when unset or empty.
The `effort` operation remains available for callers that select answer effort.
The native kernel passes its caller's `reasoningEffort` to the model.
It does not select effort through a scheduler round.
Responses sends `reasoning: {effort}`.
Chat Completions sends `reasoning_effort`.
The Decisions API receives no reasoning field.


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

## Scope

The browser sends the selected ticker separately from the original question.
The browser also sends its watchlist as a separate list.
The route validates watchlist values before dispatch.
The native kernel passes a supplied watchlist to `open_research` instead of ticker scope.
The worker stores no ticker when a watchlist applies.
The worker accepts `asOf` for a point-in-time session.
Source calls use the cutoff that the session or original request states.

The watchlist panel loads sources through `GET /api/watchlist?symbols=...` once per watchlist.
The worker `op:"watchlist"` uses a separate thread. It creates no session, job, or evidence.
Research questions use native model tool calls instead of this fixed source fan-out.

Each symbol gets four fixed-argument calls through the existing tool handlers.
`list_sec_filings` uses `limit=5` and no forms filter, so fund forms stay in.
The three FINRA calls are `get_short_interest`, `get_reg_sho_volume` (the handler's current NYC week), and `get_threshold_securities`.
Every call is an independent task on the existing SEC and tool pools under the 8s intake budget.
A call past the budget is cancelled and reported as a timeout.
Each (symbol, source) row carries `status` (`ok`, `no_records`, `error`, `timeout`), `as_of`, source references, and the evidence ID when admitted.
A FINRA "No data found" answer is `no_records`. Any other failure is `error`.
The watchlist read creates no research session, job, or evidence.
The optional coverage response distinguishes SEC document availability from inspection.
FINRA and SEC clients retain their existing caches.

## Deterministic evidence gates

`tool_runner.py` executes tools through `app/tool_runtime.py` against canonical `app/tools.py`.
`service.admit_evidence` delegates to `record_evidence`.
Admission checks the session, owning job, source domain, canonical provenance, source identity, and point-in-time availability.
The service reloads SEC document handles or persisted structured tool results.
It checks that the cited passage belongs to those source bytes.
It checks content hashes and any observation or calculation proof.
The service checks the owning job before it commits evidence.
Admission uses no model verdict for topical relevance or resolution support.
SEC navigation records remain discovery artifacts, not proof of document content.

`prove_claims` parses each claim and calls `service.require_verified_claims` with no semantic deferral.
The gate uses the session ledger, not evidence text supplied by the caller.
It checks citation ownership, archived bytes, hashes, point-in-time proof, quotations, and SEC accession identity.
Every referenced datapoint must belong to a cited verified row.
Observed and calculated datapoints must match the declared claim type.
Number wording must bind the metric, value, entity, period, and unit.
Non-observation claims need their explicit qualifiers.
A qualitative observed fact must match a located statement.
No model judges a qualitative paraphrase after these checks.

A deterministic rejection returns an error that starts with `proof:`.
Any other proof error means that the gate is unavailable.
The kernel rejects malformed or misaligned verdict arrays.
It never treats an unavailable gate as evidence against a claim.

## Thesis monitoring

`cli.py thesis monitor <slug>` calls `app/thesis/runner.py run_trigger`.
The default runner posts the prompt to `/api/agent` with `mode: "research"`.
An optional `asOf` cutoff reaches `open_research` unchanged.
The agent runs the same native research loop as interactive research.
The retired research rounds remain deleted.

The runner waits for the SSE stream to emit `done`.
It raises on `failed` or `error`, preserving the event message.
A non-200 response, a transport failure, a timeout, or an incomplete stream also raises.
These failures leave the trigger pending for retry.
Even after `done`, the existing journal check requires a durable entry for that trigger and run.

- `STOCKBOT_AGENT_API_URL` sets the API base URL.
  Its default is `http://127.0.0.1:${STOCKBOT_API_PORT:-3001}`.
- `STOCKBOT_THESIS_RESEARCH_TIMEOUT_SECONDS` sets a positive, finite timeout in seconds.
  Its default is `300`.
  The runner bounds connection waits and interrupts the stream when the total deadline expires.


SEC replay evidence uses the tool name, record identity, and tool-result ID
to distinguish records. Repeated citations of the same record and result
reuse the existing evidence ID. Identical records from different result IDs
remain separate. Existing sessions retain their stored identity keys.

FINRA evidence cites the first canonical row when the result contains records.
For an analysis envelope without records, the runner cites the complete canonical envelope JSON.
It never substitutes the query JSON or a copied metric for the persisted result.

Repeated admissions retain their evidence IDs.
The native kernel adds each returned ID only once to its evidence list.
Tool calls without admitted evidence remain uncitable observations.

## Needle arguments

The worker retains `arguments` for chat commands.
It binds the selected tool's canonical description and parameter schema.
The SDK receives temporal anchors, ticker context, and grounded argument values.
It receives no research evidence, raw tool results, or attempt history.
The worker rejects explicit engine failures, ungrounded fields, and mismatched tools.
The shared worker timeout still bounds argument generation.


The terminal Harness tab renders `HarnessSessionViewer` with the selected session id. It reads the existing session plus snapshot research APIs only. A missing or failed snapshot stays unavailable for that session, never another session. Static `HarnessViewer` with a projection prop remains for untracked preview files; export previews write only the `--out` path.

## Market data store and collector

`data/market.sqlite` (`app/storage/market.py`) keeps market history for local reuse. The file uses WAL mode, so the collector writes while readers read.

| Table | Contents | Retention |
|---|---|---|
| `watchlist` | One saved watchlist per install. The first run seeds it from `DEFAULT_WATCHLIST`. | Kept |
| `bars`, `quotes`, `filings`, `ciks` | Candles per interval, quotes tagged by source, SEC Atom filings, ticker→CIK | Kept (Robinhood quotes: 30 days) |
| `snapshots` | One JSON payload per kind, symbol, and UTC day: estimates, FINRA, US Google Trends, `rh_options` | Kept (`rh_options`: 30 days) |
| `news` | Headlines, article text, summary state | 30 days |
| `refresh_log` | Last success, last error, covered range per kind, symbol, and params | Kept |
| `budget` | Exa dollar spend per month | Kept |

The collector (`app/services/collector.py`) runs as one daemon thread in the FastAPI lifespan. `STOCKBOT_COLLECTOR=0` disables it. The thread wakes every 60 seconds and runs each job whose interval has passed. Jobs run one at a time:

- quote: 5 minutes. Candles and news: 15 minutes. Filings: 1 hour.
- Analyst estimates, SEC companyfacts, FINRA short interest and Reg SHO, Exa news, US Trends: daily.
- Article text and news summaries: small batches every cycle. Retention: daily.

A failed job records `last_error` and runs again after its interval. A failed job does not change `last_ok_at`.

Reads (`api/market_providers.py`) return stored data with `as_of` and `stale`. A stale read does not fetch. A read fetches live only when the store has no data, or when a candle range starts before `covered_from`. The header Refresh button (`POST /api/market/refresh`) runs every job once. `status.running` is true only while a requested refresh is queued or running.

Quotes run in two collector threads, one per source, separate from the 60-second cycle, so a slow Yahoo sweep never delays the 5-second broker poll. They cover the watchlist plus every symbol the UI has read a quote for (`market.quote_symbols()`), such as the index tape and searched tickers. Neither provider publishes a rate limit, so the intervals sit below observed ceilings:

| Source | Interval | Calls | Evidence |
|---|---|---|---|
| Robinhood (broker connected) | 5 s | one batched `get_equity_quotes` call for all tracked symbols | about 4 calls/s sustained, throttling near 8/s, as measured by kevin1chun/robinhood-for-agents |
| Yahoo `quoteSummary` | 60 s | one call per symbol, bypassing `valuation`'s 5-minute price cache | about 2,000 requests/hour per IP is the commonly reported ceiling |

A quote row is saved only when its price differs from the newest saved price of the same source. `refresh_log` records every successful poll. `GET /api/quote` returns the price from whichever source was confirmed most recently, with `source` set to `robinhood` or `yahoo`. A read is `stale` when that source missed two of its intervals or its last poll failed. Reads call a provider only when no quote is saved yet: Robinhood first when connected, then Yahoo for symbols Robinhood does not quote, such as indexes and futures. Each source is tried at most once per its interval, so repeated reads of an unquotable symbol never hammer a provider. The UI polls quotes every 5 seconds (`QUOTE_POLL_MS`). Option chains are saved once a day under `rh_options` while the broker is connected. Robinhood quotes and `rh_options` snapshots are deleted after 30 days. Without a broker, no code calls Robinhood.

Exa: every caller reserves an upper-bound cost in `budget` before the request. The client then replaces the reservation with `costDollars.total`. The monthly cap is `EXA_MONTHLY_LIMIT` (default $9) for all callers. The collector also has its own cap, `EXA_COLLECTOR_MONTHLY_LIMIT` (default $3). Set a matching key budget in the Exa dashboard. Only that budget stops spending from other machines.

News summaries (`app/services/news_summary.py`): the OpenCode reasoner reads each new story. The prompt is `STORY_SYSTEM_PROMPT` in `app/reasoner_client.py`, and the reply is checked JSON. A relevant story with high or medium impact becomes `data/news/YYYY-MM-DD/<TICKER>-<slug>-<hash8>.md`. The collector requests each article body at most once. A story gets two summary attempts.

Retention deletes these items when they are older than 30 days: `news` rows, `data/news/` day folders, `data/briefing/*.json`, and `data/raw/viewer/capture` entries. Retention refuses symlinked roots.

## Numeric datapoints

A published number needs a verified datapoint.
Its proof reconstructs the metric, value, entity, period, and unit from archived source bytes.
One evidence row holds at most one datapoint.
The runner can admit datapoints from structured tool records.
The kernel can also send model declarations to `admit_datapoint`, with at most four concurrent requests.

The worker reconstructs the cited row's admission input.
SEC document declarations reuse the original source handle stored in the row's metadata.
A missing handle or a descriptor mismatch fails closed.
A failed declaration never makes an unproven number publishable.

Calculation declarations identify their formula and observed inputs.
The ledger recomputes the result and checks input identity, metric, unit, and periods.
Claim proof distinguishes calculations from observations.
It checks each stated number against its cited datapoints.

The kernel aligns declared claim text with the answer before it checks the inventory.
It removes sentences that no supported claim covers.
It applies the same readable-number formatting to the final answer and kept claims.
Formatting preserves citation brackets, identifiers, accession numbers, and dates.
The native path runs no model correction loop after proof.


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

The native path uses `ResearchKernel`, `RuntimeToolSession`, and `ToolOutcome` contracts.
Thesis monitoring keeps only the session bootstrap.

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
- `DECISION_MODEL`: the model for the surviving JEV operations, including scope, answer sections, and answer effort.

Native research tools and final claim proof do not call JEV or the Reasoner for decisions.
The native model selects tools directly.
The ledger decides final claim proof deterministically.

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
`state` is the JSON text of the state object, not the object itself: Respan models (for example `respan/span-01-lite:free`) reject an object state with HTTP 400, and the TS sidecar and the direct HTTP path in `app/decision_client.py` both send the string.
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

## Named workspaces

The Workspaces page contains freeform windows through `react-rnd`.
The former Home page is the `default` workspace.
The header shows `Workspaces: <name>` for the selected workspace.
Market indexes, the SPY chart, briefing, scans, markets, company details, and portfolio each occupy a movable window.
Scans and markets retain their shared selection.
The company details window follows the selected market row.
The window picker also offers these panels in other workspaces.

Drag a window title to move it.
Drag an edge or corner to resize it.
The panel menu replaces a window's contents without changing its position or size.
The **?** control beside the workspace tabs shows these instructions on hover or keyboard focus.
The instructions do not occupy a workspace row.
Market windows retain independent tickers.
The ticker beside Add widget applies only to new windows.
Watchlist, Portfolio, Agent, Theses, Alerts, Sessions, and Journal use the same window controls.

The default workspace starts with Watchlist, Portfolio, Price & volume, Markets, Company details, Market indexes, News, and Agent.
Company details starts on its window ticker. A Markets row click switches it to that row.
The client sizes that starter layout to the browser window once, when it builds the layout.
The Agent window replaces the agent drawer, so chat moves and resizes like other windows.

Create a named workspace with the **+** button after the workspace tabs.
New workspaces start empty.
Select **Start from default** to copy the current default workspace.
The copy includes unsaved window edits, positions, sizes, tickers, and panel settings.
Each copy receives independent window IDs.
Changes to a copy do not change the default workspace.

Edit a custom workspace's name in the toolbar.
The default workspace keeps its name.
Workspace tabs retain unsaved drafts during page and workspace changes.
Only Save writes names, window order, geometry, tickers, and panel settings to the server.
Agent receipts also remain unsaved until Save.
A failed save retains the draft.
The browser warns before a reload or close discards unsaved drafts.

Expand fills the visible workspace area.
Escape restores the original window geometry.
Forms and menus handle Escape before the window.
Focus a window title to use arrow keys for 10-pixel moves.
Shift with an arrow key resizes the window by 10 pixels.
The desktop scrolls when saved windows exceed the visible area.

`GET /api/workspaces` lists named documents.
`PUT /api/workspaces/<uuid>` saves `{name, widgets}`.
Each widget includes `{id, type, ticker, params?, geometry: {x, y, width, height}}`.
The server validates names, UUIDs, unique widget IDs, widget types, tickers, geometry, and chart/link settings.
Each workspace supports at most 24 windows.
The server stores named documents in `data/named-workspaces/<uuid>.json`.
The default workspace uses UUID `00000000-0000-4000-8000-000000000000`.
The client supplies its starter layout when no saved default document exists.
Only Save stores edits to that layout.
The client preserves existing saved workspaces and the remembered workspace selection.
It does not read, import, modify, or delete legacy `data/workspaces/` files.
