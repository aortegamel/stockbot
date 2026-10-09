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
owns the evidence lifecycle. In research, every selected persona runs as its
own concurrent Muse call over the same graph projection, evidence, decisions,
unresolved items, and authority limits. An empty graph keeps the direct no-evidence
prompt. Drafts stay buffered and never stream. If any persona reports
`Missing-Evidence`, the kernel combines the unique gaps into one shared
follow-up pass in the same session. Then it regenerates every selected
persona. A failed persona, an empty or whitespace-only persona response, a
second gap, or too little time before the request deadline fails the report
with no prose. Only when every selected persona returns prose without a gap
does one `answer_delta` emit the labeled sections (`── Stockbot ──`, …) in
fixed order. Metrics count every attempted persona call, sum token usage over
calls and passes, and report final-stage wall time.

Architecture: the kernel scheduler owns the agent loop (JEV selects over
the whole registry every round, Needle fills arguments only, tools execute
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
