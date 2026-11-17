# Needle Harness

Remote reasoning through one OpenCode model, Bun orchestration, Vite web UI.
The optional Needle experiment adds a small local model for chat command arguments.

## 1. Requirements

- Bun 1.4.2 and uv
- `OPENCODE_API_KEY`, `OPENCODE_URL`, and `OPENCODE_MODEL` in the repo-root `.env`
- `SEC_EDGAR_IDENTITY="Name email@example.com"` in the repo-root `.env` for SEC tools
- Optional Needle experiment: `NEEDLE_ENABLED=true`, the `cactus-needle` package, and the repo-root `needle3.cact` weights

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
bun run stockbot  # from repo root → vite build, then vite preview on PORT (free-picked from 3000), every IPv4 interface
bun run dev       # from repo root → same launcher with the vite dev server (STOCKBOT_WEB_MODE=dev)
# staged logs per step; browser misbehaves? first check the Bun API port in the startup log, then curl that /api/health
bun run lint && bun run typecheck
BENCHMARK=1 bun run stockbot   # write benchmarks/*.json per run
```

Vite (`PORT`) binds `0.0.0.0` by default (`STOCKBOT_LAN=0` keeps `127.0.0.1`). The Bun API (`STOCKBOT_API_PORT`) and the FastAPI market service (`STOCKBOT_MARKET_PORT`) always bind `127.0.0.1` and are reached through the Vite `/api/*` proxy.
The startup log prints each selected port.
`STOCKBOT_LAN_TOKEN=<token> bun run stockbot` requires the token on `/api/*`; open `http://<host LAN IP>:<PORT>/?token=<token>` once per device.
See the root README "LAN access" section.

## 5. Architecture

```
prompt + current chart + exclusive ticker/watchlist scope
  → exact UI command? → tool arguments → canonical tool → receipt event
  → otherwise → one OpenCode model call with curated read-only tools → answer + sources
research mode (exclusive scope)
  → open_research → native tool loop → evidence admission
  → model writes the answer and its claims → deterministic claim proof
Bun streams AgentEvents as SSE to the web UI.
```

Bun orchestrates. The model selects tools. Python tools retrieve and admit evidence. Code proves each claim. JEV and Needle are optional experiments; see the root README.

### Current stock and chart actions

The API sends the displayed stock as `currentChartTicker`, separately from the exclusive research `ticker` or `watchlist` scope.
The existing frontend `ticker` supplies the displayed stock unless the request supplies a valid `currentChartTicker`.
The API validates each stock symbol with the existing ticker validator.
The user prompt remains verbatim.

A chart command is one exact shape: `show AAPL`, `open the chart for $nvda`, or `switch to MSFT`.
The route parses the ticker and calls `set_chart_symbol` directly. No model is involved.
The canonical tool must return a matching `chart_selection` receipt before the route emits `{type:"chart_update", ticker}`.
The bridge carries that receipt in `result.meta.chart_selection`, and the TypeScript adapter keeps it as untrusted `ToolResult.chartSelection` only for `set_chart_symbol`.
The API checks that object, not the tool's formatted content.
The action shares the request deadline and the existing worker and bridge cancellation.
The chart receipt supplies no factual answer evidence or research seed evidence.
A failure answers as text and does not end the request.

A watchlist command (`add NVDA to my watchlist`) or a widget command (`add a short volume widget`) needs tool arguments.
The route asks Needle for them when `NEEDLE_ENABLED=true`. Otherwise it asks the OpenCode model for one JSON object from the tool schema. A Needle failure also falls back to the model.
The canonical tool validates the fields. The bridge preserves its `watchlist_update` or `widget_selection` receipt.
The route emits the matching update event only after canonical success.
The widget receipt ticker must equal `currentChartTicker`.
The receipt supplies no factual evidence. A failure answers as text without research.

### Stored conversations

Chat keeps its messages in `data/chat.sqlite`. `STOCKBOT_DATA_DIR` moves that file.
The tables are `chats` and `chat_messages`. One installation is one account, so no owner column exists.
The first send creates a chat and names it with the prompt.
The API returns the chat id in the `X-Chat-Id` response header.
A later send names that id, and the API reads the earlier messages from the store.
The provider reads the last 20 messages, and each message is cut to 4000 characters.
A failed or cancelled run stores the question alone.
`GET /api/chat` lists the chats. `GET /api/chat?id=` returns one chat. `DELETE /api/chat?id=` deletes one chat.
The chat panel lists the stored chats, and it deletes a chat when the user clicks the delete control.
A store failure still answers the request, and the response then has no chat id.
Deep research keeps its own sessions, so stored chat messages do not change a research run.

### Provider generation

The reasoner generates prose and its claim inventory in one provider call.
It appends the inventory as a fenced JSON block.
The parser removes that block before display.
No mandatory binder or second answer call follows generation.
Missing or malformed inventory supplies no trusted claims.
The existing publication gates still check claims before display.

Qualitative observed facts can use accurate paraphrases of cited passages.
Quotes must retain the source wording.
Numeric claims still require exact evidence IDs and datapoint IDs.
Feed items and sibling drafts remain untrusted data, not evidence.
Direct answers retain their separate rules for requests without evidence.

Every model call uses the shared caller deadline when the caller supplies one.
Otherwise, one 120-second limit covers generation, the bounded retry, and protocol fallback.
Abort and error paths retain observed token counts, provider POST counts, and generation time.
Research timings report `narrativeMs`. They do not report an absent binder stage.

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

`muse.calls` counts logical Stockbot calls.
`providerCalls` counts provider POSTs, including a bounded retry or protocol fallback.
Token totals count each POST once.
Debug timings report research generation as `narrativeMs`.
