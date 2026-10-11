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

Explicit 10-Q revenue requests list only `10-Q`.
Latest requests remove ungrounded `start_date` and `end_date` values. Explicit user dates remain intact.
SEC catalog evidence uses a persisted tool result for replay validation.
The scheduler does not treat a catalog `source_refs` receipt as a document `source_handle`.
The scheduler resolves the catalog receipt to persisted rows. It selects the latest exact `10-Q` row and reads its document with `query='revenue'`.
The first document read for an undated latest 10-Q revenue request fetches three sections: `Item 1`, `Item 2`, and `Item 1A`.
The scheduler reads these sections in parallel without a query filter.
Each section uses separate assessment and admission checks.
The round applies an assessment verdict only after every section read settles.
Later reads, dated requests, and other revenue requests keep one `query='revenue'` read.
A failed section read returns to the selector on its own. The accession-family and repeated-failure breakers do not count sibling sections as one repeated call.
For an undated latest 10-Q revenue request, the scheduler uses grounded listing and document arguments without Needle generation.
Requests with explicit dates, extra required arguments, or summary-only listings keep Needle generation.
Document evidence keeps visible inline XBRL facts and the full bounded canonical passage.
The document view excludes hidden XBRL resources.
Attempt summaries stay bounded at 2,000 characters.

For canonical `get_sec_document` results, JEV receives one verbatim source quote capped at 4,000 characters.
The quote centers on the first revenue match.
If the passage has no revenue match, the quote starts at its beginning.

Prior evidence retains its IDs, source names, subjects, and claim text capped at 200 characters.
Prior observations retain tool identities, errors, references, and summaries capped at 200 characters.
This compact view changes only document assessment.
Admission and persona evidence retain the full passage.

Tool selection, other JEV calls, evidence gates, and deadlines stay unchanged.
On a 10-Q revenue node, a `reason_over_evidence` assessment of a SEC listing does not end the node before one document read succeeds.
On any node, a `reason_over_evidence` assessment reloads admitted evidence and starts a fresh JEV selection within the node round cap.
A rejected document call keeps recovery tools for JEV re-selection.
All evidence gates and deadlines stay unchanged.
The Needle argument decoder has a 20-second ceiling in Python and TypeScript.

In-node JEV selection (`scheduler._node_context`) always withholds `research_start`, `research_read_search`, and `research_resume`.
When `stage_for_session` derives the SOURCE stage, selection also withholds `research_add_analysis` and `research_finalize`.
An unknown or created stage derives SOURCE.
TARGETED_RESEARCH is SOURCE, even after a prior freeze. A frozen ANALYZING session is not SOURCE.
The filter runs before the selection copy and before any generation.
A SOURCE node never fabricates a committee payload.
`build_registry`, the COMMITTEE and FINAL tool sets, and selection in other stages stay unchanged.
The `reason_over_evidence` path, the gates, the caps, the retries, and the deadlines stay unchanged.

JEV tool selection (`_select_round`) receives a compact copy of its evidence and attempts.
Each item keeps its IDs, tool, arguments, and scalar fields capped at 200 characters.
Nested payloads, such as raw `result` rows and `metadata`, stay out of the selection copy.
When the copy drops text, Needle `embed` ranks up to 24 passage chunks of 2,000 characters from the four most recent candidates.
Needle `extract_fields` then reads one `selection_fact` record from the best chunk.
JEV receives one `selection_digest` item with the top three ranked IDs, their scores and summaries, and the extracted fields. JEV never receives vectors.
The Needle calls run serially on one worker with an 8-second budget, capped at a quarter of the remaining run deadline.
A Needle error, a timeout, or a spent budget keeps the compact copy without the digest. Selection then continues.
The registry options, the gates, the caps, the retries, and other JEV calls stay unchanged.
Assessment, admission, and personas keep the full evidence and attempt bytes.

Needle argument generation receives a bounded copy of the node evidence and attempts.
The copy keeps the four most recent evidence items and the ten most recent attempts.
Each item keeps fixed fields only. Text fields stay capped at 200 characters.
Each prior-call argument value stays capped at 2,000 characters.

Generation omits an oversized nested argument instead of truncating its serialized value.
Identity handles stay whole: evidence IDs, tool result references, job IDs, source record IDs, source URIs, tickers, CIKs, accession numbers, and document names.
An accession that exists only in the full nested record joins the copy as `accession_no`.
The copy drops `result`, `outcome`, `metadata`, `provenance`, and `reasoning`.
The `selection_guidance` do-not-repeat list builds from the bounded copy, with at most ten calls and 2,000 characters per call.
The objective, the node, `as_of`, `temporal_scope`, and `today_utc` reach generation unchanged.

When the selection round produced a `selection_digest`, generation receives that same digest text.
Generation makes no extra Needle `embed` or `extract_fields` call.
Without a selection digest, for example after a Needle error, generation receives no digest.
A missing fact stays absent. The 20-second decoder ceiling and the withholding rules stay unchanged.
Storage, admission, duplicate detection, accession carry, and concept checks keep the full evidence and attempts.

Before `get_xbrl_facts` runs, the scheduler checks the generated `concept`.
The objective, the node question, or a successful source record must name the concept.
Within source records, only actual content, claims, or financial tag/value rows can ground a concept.
An exact tag word matches. A `CANONICAL_CONCEPTS` group name or alias also matches whole words, without regard to case, separators, or spacing.
Other friendly labels do not match.

Generated arguments, summaries, metadata, error text, reasoning, and failed sources never ground a concept.
Neither `today_utc` nor the `[Today UTC ...]` prefix grounds a concept.
The scheduler rejects date-shaped concepts, such as a `YYYY` placeholder or a four-digit year.
It also rejects relative-date words, such as `today` or `last quarter`, and concepts without letters.
An ungrounded concept starts JEV re-selection before the tool runs. The scheduler never substitutes a fallback concept.

Entry and final personas: `route.ts` first sends one `op:route` call. JEV
returns `{route, personas}`. `personas: null` means the request neither
selected nor excluded any persona. An explicit list is nonempty, unique, and
canonical, in the fixed order Stockbot, Bearbot, Bullbot. Positive selections
drop any excluded personas; exclusion-only requests use the remaining
personas. A route outage, a selector error, a malformed selection, or no
remaining persona (`no_personas_remaining`) fails the request with `failed` +
`error`. The route never guesses a default. Without a selection, direct and
single-tool answers stay single-call Stockbot, and research uses all three
personas. An explicit selection always enters research, so the final layer
owns the evidence lifecycle. In research, partial selections run each selected
persona as its own concurrent Muse call over the same graph projection,
evidence, decisions, unresolved items, and authority limits, with no synthesis
and no retry. The full trio synthesizes: Bearbot and Bullbot draft
concurrently over the shared snapshot (buffered, never streamed), then
Stockbot judges both drafts against the same snapshot and writes the middle
verdict. An empty graph keeps the direct no-evidence prompt. Drafts stay
buffered and never stream. Exactly one failed or empty draft retries once;
a second failure, a failed or empty synthesis, or
any failure after the retry fails the report with no prose. If any draft or
the synthesis reports `Missing-Evidence` on the first pass, the kernel combines
the unique gaps into one shared follow-up pass in the same session. The kernel
merges the follow-up evidence and regenerates per the same synthesis rule.
The follow-up always runs, even after persona generation passes the first
research wall. The kernel never runs a third research pass. The second pass is
final even when a gap remains. Then the final gates check the prose like a
gap-free answer: the claim inventory, the whole-response check, and on reject
the per-claim checks. Each rejected claim substring disappears from the prose.
Unsupported inventory claims therefore leave the answer. Prose outside rejected
claims stays as written, including uncertainty statements and the
`Missing-Evidence` line. An inventory that cites an unknown evidence id or a claims[].text that is not a verbatim prose substring (never
any other inventory defect) retries once per affected persona on the same
evidence snapshot: the kernel replays the same prompt plus a correction that
lists the valid evidence ids, requires exact EVIDENCE row ids, requires every claims[].text as a verbatim prose substring, and includes
the rejected prose and inventory as untrusted output, then runs
the corrected results through the same gates. Node ids and decision records
may name graph structure, but they never count as evidence ids. A corrected
answer still passes the source gates; a still-invalid, empty, or failed
correction fails the report with no prose and no second retry. Direct answers
carry no inventory and never retry. Other inventory defects, a gate failure,
an empty persona, or all claims rejected still fail the report with no prose.
A remaining gap never emits `failed`. When the final
gates pass, one `answer_delta` emits the labeled
sections: the full trio emits `── Bearbot ──` and `── Bullbot ──` drafts, then
the `── Stockbot ──` middle verdict; partial selections emit their labeled
sections (`── Stockbot ──`, …) in fixed order. Metrics count every attempted
call including the retry, sum token usage over calls and passes, and report
final-stage wall time.
Each research persona targets 250 words of display prose. The complete fenced claim inventory stays outside this target.
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

Architecture: the kernel scheduler owns the agent loop (JEV selects over
the node-filtered registry every round, Needle fills arguments only, tools execute
via `app/tool_runtime.py` against canonical `app/tools.py`). Thesis
monitoring runs the same scheduler in-process via
`app/thesis/runner.py run_trigger` (injectable `_RUN_KERNEL` seam; the
default runs the shared graph fan-out via `run_graph_prompt` +
`scheduler.run`). No OMP/Pi entry or superseded loop remains.

`app/research/runner.py` is a deterministic eval harness only (retired live
loop kept as a test helper): the scheduler owns orchestration, the kernel
owns evidence, PIT, freeze, and stages. Production and eval orchestration
never call `run_live`/`resume_live`.

Live golden (opt-in only, never CI): `bun run verify:hedgefund-live` runs one
golden scenario through the production kernel path with real SEC/FINRA/Exa
credentials (`HEDGEFUND_LIVE=1`, `SEC_EDGAR_IDENTITY`, `FINRA_CLIENT_ID` /
`FINRA_CLIENT_SECRET`, `EXA_ENABLED=1` + `EXA_API_KEY`) and re-exports the
read-only harness-viewer projection. Without the opt-in flag it exits 2.
