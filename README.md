# Stockbot

Stockbot is a local research workstation for individual investors. It collects filings, market data, and news, then answers questions with one AI model and a deterministic proof step.

Every number in a final answer must trace back to a stored source. Stockbot cannot place orders or move money.

## What it does

- Reads SEC EDGAR filings and extracts financial facts.
- Reads FINRA datasets for short interest and Regulation SHO.
- Tracks market data for your watchlist.
- Pulls analyst estimates and valuation metrics.
- Answers questions about companies, filings, and market events.
- Shows the source of each answer.
- Optional: reads Robinhood portfolio data with no write access.
- Optional: searches the web for current news.

## How an answer is built

1. You ask a question in the web UI.
2. The model selects tools from a fixed list. It does not run code.
3. Each tool result is checked before it becomes evidence. The check covers the source, the filing bytes, the point in time, and the data shape.
4. The model writes the answer and declares every claim.
5. Code verifies each claim against the evidence. A claim without support is removed.

The model writes prose. Code does the arithmetic.

## Chat commands

The chat also accepts a few exact commands:

- `add NVDA to my watchlist`
- `remove NVDA from my watchlist`
- `add a short volume widget`
- `show AAPL`, or `open the chart for AAPL`

Everything else goes to the model as a research question.

## Experiments

Two experiments are off by default. The default runtime needs neither.

### JEV decisions (`JEV_ENABLED`)

JEV adds three decisions around the model. It can reject a request that is outside stock research, pick the parts of a chat answer, and select a reasoning effort. It calls a separate decision endpoint.

Set `JEV_ENABLED=true` and the four `DECISION_API_*` values to turn it on. When it is off, every request is in scope, every answer part is kept, and no reasoning effort is selected. See [docs/runtime.md](docs/runtime.md#jev-decision-api).

### Needle argument grounding (`NEEDLE_ENABLED`)

Needle is a small local model. It turns a chat command into tool arguments.

To use it, install the optional package and provide the weights file:

```bash
venv/bin/pip install -r requirements-needle.txt
```

Put `needle3.cact` at the repository root, or set `NEEDLE_WEIGHTS` to its path. Git does not track the weights file.

Set `NEEDLE_ENABLED=true` in `.env`. When Needle is off, the OpenCode model produces the same arguments from the tool schema. A Needle failure also falls back to the model.

## Requirements

- [uv](https://docs.astral.sh/uv/) manages Python 3.14.
- An SEC EDGAR identity: your name and email.
- An OpenCode Responses endpoint and model: `OPENCODE_API_KEY`, `OPENCODE_URL`, and `OPENCODE_MODEL`.
- [Bun](https://bun.sh) runs the project scripts.

The default runtime does not need JEV, Needle, FINRA, Robinhood, or Exa.

## Quick start

1. Install uv and Bun, then clone the repository.
2. Run `bun run setup && bun install --cwd web && bun install --cwd needle-harness`.
3. Run `cp .env.example .env`. Fill `SEC_EDGAR_IDENTITY` and the three `OPENCODE_*` values.
4. Run `bun run doctor`. Fix every `✗` line and repeat until the doctor exits 0.
5. Run `bun run stockbot`. Open the URL from the startup log, for example `http://localhost:3000`.

## How it runs

`bun run stockbot` starts three local services:

- the Vite web UI on `PORT`, default 3000,
- the Bun API host,
- the FastAPI market-data service.

Only the web UI listens on your network. The other two bind to `127.0.0.1`, and the browser reaches them through the Vite proxy.

`bun run dev` starts the same stack with hot reload.

The Bun kernel runs one native tool loop through `app/research/kernel_worker.py`. The model selects tools and supplies their arguments. The kernel admits evidence and proves each claim before it shows the answer. [docs/runtime.md](docs/runtime.md) describes the full sequence.

`cli.py` remains for admin tasks only: `runs`, `inspect`, `refresh-data`, `log-server`, and `robinhood-login`.

### LAN access

`bun run stockbot` serves the web UI on every IPv4 interface (`0.0.0.0`) by default, so other devices on your network can open it. It starts three listeners:

- Vite UI on `PORT`, free-picked from 3000, bound to `0.0.0.0`.
- Bun API on `STOCKBOT_API_PORT`, free-picked above the Vite port, always bound to `127.0.0.1`.
- FastAPI market service on `STOCKBOT_MARKET_PORT`, free-picked above the API port, always bound to `127.0.0.1`.

The browser reaches the Bun API and the market service only through the Vite `/api/*` proxy. The startup log prints each selected port.

`STOCKBOT_LAN=0 bun run stockbot` keeps the UI on loopback only.

Open `http://<host LAN IP>:<PORT>/` from the other device. Find the host LAN IP with `ip -4 addr`.

`0.0.0.0` is a bind address, not a client address. Do not put it in a browser.

Optional token: set `STOCKBOT_LAN_TOKEN` to require it on `/api/*`, with the header `x-stockbot-token` or an HttpOnly cookie set once through `?token=…`.

Without a token, anyone on your network can run the agent, spend model quota, and read theses. Use it only on a trusted network. Do not forward the port through a router or expose it to the internet.

## Environment variables

See [`.env.example`](.env.example) for the full list. Each setting is labeled required, optional, or experiment-only.

`SEC_EDGAR_IDENTITY` is required only for SEC features. FINRA, Robinhood, Exa, JEV, and Needle are optional.

Python and Bun load only the repository-root `.env`, regardless of the current directory. A root `.env` value replaces an inherited shell value. An inherited key absent from `.env` stays in effect. An inherited `PORT` stays in effect, so `PORT=3101 bun run stockbot` works.

## Tests

The default suite runs offline. External services are faked.

```bash
bun run test        # Python, decision, harness, and web suites
bun run typecheck   # Python types
bun run lint        # ruff plus TypeScript checks
bun run verify      # fresh environment, offline
```

`bun run test:browser` runs the Selenium UI regression. It needs Chromium and a matching ChromeDriver on `PATH`. The default Python suite also runs it. Set `STOCKBOT_CHROMIUM_BINARY` and `STOCKBOT_CHROMEDRIVER` to use explicit executable paths.

Live checks are opt-in: `verify:e2e` for the real agent loop, `smoke-prod` for FINRA, and `smoke-robinhood` for the broker. They do not run through `bun run verify` or CI.

## Optional Robinhood integration

Set `BROKER_ENABLED=true` and complete the local OAuth login with `cli.py robinhood-login`.

OAuth and MCP transport are pinned to `https://agent.robinhood.com/mcp/trading`. Persisted tokens and client registration are bound to that HTTPS origin.

Stockbot permits only explicitly allowlisted read operations. It blocks unknown operations, order placement, order cancellation or replacement, option exercise, withdrawals, deposits, and transfers. It does not support trading or changing saved scanners.

## Optional Exa integration

Set `EXA_ENABLED=true` and `EXA_API_KEY` to enable the `search_web` research tool. It returns bounded highlights for current qualitative evidence: news, announcements, competitive and industry developments, publications, specialist commentary, and counterevidence.

SEC, FINRA, Robinhood, and local data remain authoritative. Exa never writes canonical datasets. Exact financial facts, portfolio state, and deterministic screens come from the canonical tools.

## Stored data

Runtime data stays on your machine under `data/`, which Git ignores:

- `data/market.sqlite` keeps watchlists, quotes, filings, estimates, and news.
- `data/research.sqlite` keeps research sessions, evidence, and decisions.
- `data/chat.sqlite` keeps chat conversations.
- `data/cache.db` keeps tool caches.
- `data/logs/` keeps per-request JSONL traces.

Robinhood OAuth state stays at `~/.stockbot/robinhood/oauth.json` with restrictive permissions.

## Hypothetical questions

A hypothetical question, such as "what if OpenAI has a terrible IPO?", needs assumptions. Stockbot labels each assumption with an `assumptionId`. An assumption is never cited as evidence.

Every impact number must cite evidence ids and a verbatim source quote. Code computes the impact from the cited numbers. The model states the formula in words and does not multiply.

The validator fails closed. An unknown evidence id or a quote that does not match the source drops the analysis. An answer is never accepted with unverified numbers.

`cli.py inspect` shows the stored record for one run.

## Privacy

Stockbot runs on your machine. Some features call external services:

- SEC EDGAR, FINRA, Yahoo Finance, and news providers.
- The OpenCode endpoint you configure. Prompts and the selected research context go to that endpoint.
- Exa, when `EXA_ENABLED=true`.
- Robinhood, when `BROKER_ENABLED=true`. Credentials stay local.

Stockbot omits brokerage account identifiers from the portfolio data it sends to the model.

## Current limitations

- No hosted, multi-user, or deployment environment.
- No brokerage trading, money movement, or scanner writes.
- Some research adapters depend on external source availability.
- Yahoo Finance consensus data is unofficial and best effort.
- The broader thesis-monitoring and mandate product is not complete.

## Roadmap

See [docs/architecture-roadmap.md](docs/architecture-roadmap.md). It describes intended architecture. Some components are not implemented.

## Disclaimer

Stockbot is for research and information only. It is not investment, legal, tax, or financial advice. Verify source data and make your own decisions.

## License

[AGPL-3.0-only](LICENSE)
