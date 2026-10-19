# Needle Harness

Local routing (Cactus Needle 3), remote reasoning (Muse Spark via OpenCode Go API), Bun orchestration, Vite terminal UI.

## 1. Requirements

- Bun 1.4.2, uv, and the repo-root `needle3.cact` weights
- `SEC_EDGAR_IDENTITY="Name email@example.com"` in the repo-root `.env` for SEC tools

## 2. Setup

```bash
bun run --cwd .. setup
bun install
# Set OPENCODE_API_KEY in the repo-root ../.env (value from ../scripts/host/pi-opencode-go-secret).
# Do not export it: an empty OPENCODE_API_KEY="" in ../.env replaces any shell export.
# Set SEC_EDGAR_IDENTITY="Name email@example.com" in the repo-root ../.env.
```

Stockbot and Needle share the repository-root `venv/`.
The harness does not create a separate Python environment.

## 3. Env vars

See ../.env.example (single repo-root .env). Startup reads the repo-root .env from the harness location, never from the cwd. Never commit .env.

Values in .env replace inherited shell exports. An inherited PORT is the one exception and stays. Shell exports apply only to keys absent from .env. An empty `OPENCODE_API_KEY=""` line counts as present and replaces an exported key.

## 4. How to run

```bash
cp ../.env.example ../.env  # once from repo root, then fill values
bun run stockbot  # from repo root → Vite on PORT, free-picked from 3000, bound to all IPv4 interfaces
# staged logs per step; browser misbehaves? first check the Bun API port in the startup log, then curl that /api/health
bun run lint && bun run typecheck
BENCHMARK=1 bun run stockbot   # write benchmarks/*.json per run
```

Vite (`PORT`), the Bun API (`STOCKBOT_API_PORT`), and the FastAPI market service (`STOCKBOT_MARKET_PORT`) bind `0.0.0.0`.
The startup log prints each selected port.
From another device, open `http://<host LAN IP>:<PORT>`, for example `http://192.168.1.20:3000`.
`0.0.0.0` is a bind address, not a client address.
These endpoints have no login protection. Use them only on a trusted network. Never forward the ports to the internet.

## 5. Architecture

```
prompt + current chart + exclusive ticker/watchlist scope
  → JEV selects the original answer/research route, personas, and an optional chart focus
  → EDGAR resolves only the selected stock label
  → Needle generates set_chart_symbol arguments → canonical tool → chart_update
  → original answer/research route → evidence → reasoner → answer + metrics
Bun streams AgentEvents as SSE to the terminal UI.
```

JEV alone selects the route and chart focus. Needle generates tool arguments. Bun orchestrates. Tools retrieve. The reasoner answers.

### Current stock and chart actions

The API sends the displayed stock as `currentChartTicker`, separately from the exclusive research `ticker` or `watchlist` scope.
The existing frontend `ticker` supplies the displayed stock unless the request supplies a valid `currentChartTicker`.
The API validates each stock symbol with the existing ticker validator.
The user prompt remains verbatim.

For a question with several stocks, JEV chooses the focus in the same call that selects the original route.
The worker resolves only that chosen ticker or company label through EDGAR.
It returns `chartTicker` only when the resolved focus differs from the current chart.
It returns `chartError` when chart selection or resolution fails.
Invalid chart metadata never authorizes a chart change or discards a valid original route.

Before a direct answer, single-tool answer, or persona research run, the API executes any approved chart action.
Needle generates arguments for the fixed `set_chart_symbol` tool using the selected focus.
The API checks the generated ticker against that focus.
The canonical tool must return a matching `chart_selection` receipt before the API emits `{type:"chart_update", ticker}`.
The bridge carries that structured receipt in `result.meta.chart_selection`.
The TypeScript adapter preserves it as untrusted `ToolResult.chartSelection` only for `set_chart_symbol`.
The API checks that object, not the tool's formatted content. Existing metrics remain scoped to the original answer path.
The pre-action shares the existing request deadline and existing worker/bridge cancellation behavior. It adds no retry or separate timeout policy.
The chart receipt supplies no factual answer evidence or research seed evidence.
The API adds no assessment call and never reroutes the question for this action.
The API rejects `set_chart_symbol` as an original entry route or a post-tool answer transition.

If the chart action fails, the API emits a warning as answer text and continues the original question.
The reasoner keeps the prior chart context after failure.
After success, direct, single-tool, persona, follow-up, and synthesis calls use the applied ticker as separate reasoner context.
Watchlist calls omit that reasoner ticker and keep their exclusive watchlist research scope.
The reasoner treats current-stock context as context, not evidence.

A successful canonical research invocation emits the same event through validated worker progress.
Production SSE retains `chart_update` while ordinary progress remains redacted.
The API does not infer chart actions from tool previews or answer prose.

### Captured evidence

Answer citations open the original page in a new tab when the source supplies a safe HTTP or HTTPS URL.
The browser fetches each cited source once per console instance to add its quote fragment.
Text-only updates do not restart source resolution.
The browser uses a range for multiline quotes or single-line quotes that exceed 80 characters and ten words.
Each range covers the first and last five words.
The browser can highlight that range across paragraph boundaries if the original page still contains it.
All other quotes use one complete text fragment.
Without a safe URL, the citation opens the immutable source capture beside the answer.
The `Sources: …` controls still open that capture.
The server creates every source ID. The browser never builds one.

- `capture:<sha256>` names a capture that the bridge archived after the runtime accepted a tool result.
- `evidence:<id>` names a stored evidence row. The server resolves it from archived SEC bytes or the recorded tool result.

The viewer requests only `/api/research/sources/<id>` and its `/pdf` path.
The viewer does not retrieve arbitrary source URLs.

Each capture carries one explicit kind label:

- **Provider extraction**: Exa returned page text for a `search_web` row. It is not the original document.
- **Search excerpt**: Exa returned only a highlight for the row. The viewer shows only that excerpt.
- **Structured record**: A FINRA or SEC tool returned a record. The viewer shows it as JSON.
- **Original document**: `get_sec_document` retrieved SEC bytes. The archive keeps those exact bytes.

The viewer marks only an exact quote. Matching ignores whitespace and nothing else. It never makes fuzzy matches.
SEC HTML matching runs against the server's rendered canonical text.
The viewer maps preserved table cells onto that basis before it trusts stored coordinates.
Exa highlights may contain provider HTML. The viewer tries the stored quote first. If that fails, it projects
provider HTML through the same sanitizer and retries the exact match against the rendered characters.
An exact literal match, even an ambiguous one, always wins over the projection.
Stored full-document coordinates or prefix and suffix context select one occurrence of a repeated quote.
If more than one match remains, the viewer reports an ambiguous quote and marks no passage.
If no match exists, the viewer reports an unresolved quote and marks no passage.

The read-only source API checks archive hashes before it returns content or PDF bytes.
If a capture is missing or corrupt, the viewer shows an error instead of fetching a replacement.
The original link can show a newer document.

If archiving raises an exception, the tool call fails with `source_capture_failed`.
The runtime records no successful evidence for that call.
The tool returns no partial source references.

A successful SEC result requires a pinned original archive.
The call fails when its original bytes or hash cannot be archived or verified.
A legacy web row can resolve to a search excerpt without page text.
A legacy row without a matching archive or excerpt shows an error.

The viewer sanitizes captured HTML with DOMPurify.
PDF.js renders only original SEC PDF captures, with selectable text and quote highlights.
The client rejects a PDF with any other capture kind.
The supervisor runs `copy-pdfjs` before Vite starts.
That script copies the packaged PDF.js worker, fonts, character maps, WebAssembly, and ICC assets to `web/public/pdfjs`.
The viewer loads these assets from the same origin. It uses no CDN.

The navigation helpers adapt [Magellan](https://github.com/magellan-extension/magellan) under the MIT license.
`lib/source-viewer.ts` includes its copyright and license notice.

## 6. Benchmarks

With `BENCHMARK=1`, each run writes `benchmarks/<YYYYMMDD-HHmmss>-<6rand>.json`
with prompt, needle decisions, tool calls, full evidence, Muse usage, answer, metrics.
Never committed (gitignored).
