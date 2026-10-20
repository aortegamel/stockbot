# Stockbot

Stockbot is a local-first AI investment research system designed to give individual investors hedge-fund-style research infrastructure. Today it is the research and data foundation for that direction, not an automated trading or consumer portfolio-management product.

## Current capabilities

- SEC EDGAR research, filing extraction, and deterministic financial calculations
- FINRA datasets, short-interest screens, and Regulation SHO analysis
- Point-in-time `known_at` data handling, provenance, raw-data archiving, normalized Parquet datasets, and DuckDB analytics
- Obligation and valuation analysis
- A tool-using research agent backed by Pi
- Analyst-consensus and market-data adapters
- Optional Robinhood portfolio, quote, option, and saved-scanner **read** integration with portfolio analytics
- Optional Exa-backed external web research (`search_web`): bounded current-evidence search with highlights; results are research evidence, not canonical financial records

## Architecture principles

Stockbot keeps raw source data and normalized analytical data separate, preserves when data became knowable, and favors deterministic calculations over model arithmetic. The LLM selects bounded tools and explains evidence; it is not the primary database or calculator.

## Grounded hypothetical reasoning

Hypotheticals ("what if X does terrible?") are model-invented assumptions, each declared with an `assumptionId` — never cited as evidence.
Every impact number must cite exact evidence ids in `evidenceRefs` plus a `numbers[]` entry `{value, evidenceId, quote}`.
Each `numbers[]` quote reproduces the source bytes verbatim; live traces prove the run reasoned on exact bytes, not summaries.
The model never multiplies: it states the formula in words ("exposure times haircut, divided by 100").
Code attaches `computed_impact` (`exposure_pct x haircut_pct / 100`) to each kept analysis via
`attach_scenario_impact` in `app/research/grounding.py` (fail-open: unparseable pairs ride unchanged).
Unknown refs and non-verbatim quotes fail closed: the validator raises `ValueError` and the caller drops the analysis, never accepts it.
Live traces (`cli.py inspect`) show the exact GROUNDED bytes the run reasoned on.
`reason_adjudication` DecisionRecords persist `evidence_refs` + `numbers` + `assumptions` + `computed_impact` in `selected`, so inspect links each interpretation back to its source rows.
Example: "If the OpenAI IPO does terrible, how does NVDA move?"
-> assumption `openai-ipo-terrible` (invented, labeled) carries the 30% haircut; it needs no evidence ref.
NVDA's 12.5% OpenAI exposure is quoted verbatim from `ev-nvda-10k`; code computes `12.5 x 30% = 3.75`, the model does not.

## Quick start

On a fresh machine:

1. Install [uv](https://docs.astral.sh/uv/) and [Bun](https://bun.sh), then clone this repository.
2. Run `bun run setup && bun install --cwd web && bun install --cwd needle-harness`.
3. Run `cp .env.example .env`. Fill every key that `.env.example` labels `MVP-required`: SEC, JEV decisions, and OpenCode reasoner.
4. Run `bun run doctor`. Apply the `Fix:` on each `✗` line. Repeat until the doctor exits 0.
5. Run `bun run stockbot`. Open the Vite URL from the startup log, for example `http://localhost:3000`.

## Requirements

- uv (installs and manages Python 3.14)
- An SEC EDGAR identity for SEC requests
- A working Pi setup for model inference (Stockbot shells to `pi`)
- Bun (task runner) — `package.json` scripts are zero-dependency wrappers around Python tooling; required for `bun run setup`, `bun run test`, `bun run verify`.

## Local setup

```bash
cp .env.example .env
# Fill only the credentials for integrations you intend to use.
bun run setup
```

`bun run setup` reads `.python-version` and creates the only Python environment at `venv/`.
Stockbot, Needle, and all Python tools use this environment.
The setup installs their shared dependencies from `requirements.txt`.

Offline tests do not need FINRA, Robinhood, or SEC credentials.

### Offline browser regression

The default Python suite runs Selenium against the actual Vite UI.
The test uses local HTTP fixtures for every market and research request.
It does not start the live API servers or call SEC, Yahoo, or broker services.
At 1440x900 and 1280x720, it checks that the agent input and Send button stay visible.
It also sends native wheel input to scroll the transcript and the captured source content.

Install Chromium and a ChromeDriver version that matches the Chromium major version.
The test searches `PATH` for `chromium`, `chromium-browser`, or `google-chrome`, plus `chromedriver`.
Set `STOCKBOT_CHROMIUM_BINARY` and `STOCKBOT_CHROMEDRIVER` to use explicit executable paths.
The test fails if either executable is missing.
It never downloads a browser or driver through Selenium Manager.

```bash
bun run setup
bun install --cwd web
chromium --version
chromedriver --version
bun run test:browser
```

The repository pins [Selenium 4.51.0](https://pypi.org/project/selenium/4.51.0/).
Its package metadata supports Python 3.10 or later, including the repository's Python 3.14.
The test uses headless Chromium without a display server.
`bun run test` and `bun run verify` include this regression through the default Python suite.

The web component suite runs with `bun test --cwd web`.
Its App tests await deferred watchlist results inside React `act` boundaries.
Fake-timer tests unmount components before restoring the real timers.

## Environment variables

See [`.env.example`](.env.example) for the current list and whether each setting is required, optional, or testing-only. `SEC_EDGAR_IDENTITY` is required only for SEC features. FINRA and Robinhood are optional integrations.

JEV decisions use one configured endpoint. Set `DECISION_API_FORMAT` to `chat_completions` for an OpenAI-compatible Chat Completions endpoint or to `decisions` for a native Decisions API endpoint, such as OpenRouter `/api/alpha/decisions`. Set `DECISION_API_URL`, `DECISION_API_KEY`, and `DECISION_MODEL` to select the endpoint, credential, and model. The code supplies no format, URL, or model default. See [docs/runtime.md](docs/runtime.md#jev-decision-api).

Python and Bun startup both load only the repository root `.env`, regardless of the current directory.
Root `.env` values replace inherited shell values.
Inherited keys absent from `.env` stay in effect.
An inherited `PORT` stays in effect, so `PORT=3101 bun run stockbot` works.

## Running research

```bash
bun run stockbot
```

Pi owns model selection and the agent loop; Stockbot exposes its tools and
security gates over the stdio bridge. `cli.py` remains for admin tasks only
(`runs`, `inspect`, `refresh-data`, `log-server`, `robinhood-login`).

### LAN access

`bun run stockbot` binds three listeners to every IPv4 interface (`0.0.0.0`):

- Vite UI on `PORT` (free-picked from 3000).
- Bun API on `STOCKBOT_API_PORT` (free-picked above the Vite port).
- FastAPI market service on `STOCKBOT_MARKET_PORT` (free-picked above the API port).

The startup log prints each selected port.
From another device, use the host LAN IP, for example `http://192.168.1.20:3000`.
Find the host LAN IP with `ip -4 addr` on Linux.
`0.0.0.0` is a bind address, not a client address. Do not put it in a browser.
Each service also answers health checks directly, for example `http://192.168.1.20:3001/api/health`.

These endpoints have no login protection.
Anyone who can reach the host can use the UI, run research, and read market and research data.
Run Stockbot only on a trusted network.
Do not forward these ports through a router or expose them to the internet.

## Running tests
```bash
bun run test
bun run typecheck
bun run lint
bun run test-collect
# Fresh-environment verification (offline):
bun run verify
# Live agent checks (manual, opt-in; requires Pi plus model credentials):
bun run verify:live
# Live end-to-end (manual, opt-in):
bun run verify:e2e
```
The default suite is offline; Pi model credentials are required only for `bun run verify:live`. FINRA and Robinhood smoke tests are opt-in bun scripts (smoke-mock, smoke-prod, smoke-robinhood). `bun run verify:live` and `bun run verify:e2e` are manual/opt-in and are not part of `bun run verify` or CI.
It uses the configured Pi model and root `.env`, forces FINRA production mode
(`FINRA_USE_MOCK=0`), calls real SEC/FINRA/Exa services, and writes isolated
evidence under `data/verify/<UTC batch>/agent-loop/`. It can consume model/API
quota and requires the relevant `.env` values and network access.

## Optional Robinhood integration

Set `BROKER_ENABLED=true` and complete the local OAuth login flow (`cli.py robinhood-login`) before using the integration. OAuth and MCP transport are pinned to `https://agent.robinhood.com/mcp/trading`; persisted tokens/client registration are bound to that HTTPS origin. Stockbot only permits explicitly allowlisted read operations and blocks unknown operations. It blocks order placement, order cancellation/replacement, option exercise, withdrawals, deposits, and transfers. It does not support trading or modifying saved scanners.

## Optional Exa integration

Set `EXA_ENABLED=true` and `EXA_API_KEY` to enable external web research via
the `search_web` tool. It returns bounded highlights for current qualitative
evidence: news, announcements, competitive/industry developments,
publications, specialist commentary, and counterevidence. Canonical
SEC/FINRA/Robinhood/local-warehouse data remains authoritative — Exa never
writes canonical datasets, and exact financial facts, portfolio state, and
deterministic screens must come from the canonical tools. Searches appear in
run traces (`cli.py inspect`) with their evidence.

## Data storage

Runtime data is local and ignored by Git under `data/`. Stockbot archives raw source responses and stores normalized datasets as Parquet for DuckDB analytics. Robinhood OAuth state is intended to stay local at `~/.stockbot/robinhood/oauth.json` with restrictive permissions.

## Privacy and external services

Stockbot runs locally, but selected functionality calls external APIs including SEC EDGAR, FINRA, Robinhood MCP, and market-data sources. When Pi runs inference, prompts and selected research/tool context—including applicable portfolio research context—are sent to the user's configured Pi model for inference. Raw Robinhood OAuth credentials remain local; Stockbot omits brokerage account identifiers from the model-facing portfolio payload.

The analyst-consensus adapter uses Yahoo Finance's unofficial `quoteSummary` endpoint with a cookie/crumb workflow. It is isolated in `app/analyst_client.py`, is not required for startup, and fails gracefully; it is not a guaranteed supported API.

## Current limitations

- No hosted, multi-user, or deployment environment
- No brokerage trading, money movement, or scanner writes
- Some research adapters depend on external source availability
- Yahoo Finance consensus data is unofficial and best-effort
- The broader thesis-monitoring and mandate product is not implemented

## Roadmap

See [docs/architecture-roadmap.md](docs/architecture-roadmap.md). It describes intended architecture; some components are not implemented.

## Disclaimer

Stockbot is for research and informational purposes only. It is not investment, legal, tax, or financial advice. Verify source data and make your own decisions.

## License

[AGPL-3.0-only](LICENSE)
