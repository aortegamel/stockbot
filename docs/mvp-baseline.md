# MVP baseline (Phase 1.4)

Measured 2026-10-20 UTC, commits through `ae89fbb` (Phase 1.1 timings),
harness `bun run stockbot` on 127.0.0.1:3000, `TIMEOUT_S=400`.
Raw runs: `data/verify/20261020T014846Z/e2e.json`,
`data/verify/20261020T020907Z/e2e.json`,
`data/verify/20261020T023148Z/e2e.json`.
Phase timings land at `runs[].timings.timings` in those files
(the runner stores whole `done.metrics` as `run.timings`).

Targets: research first `answer_delta` p90 < 10 s, `done` p90 < 45 s,
each research answer ≥ 1 citation.

## Aggregate (research prompts, 8 prompts × 3 runs = 24 rows)

| metric | p50 | p90 |
|---|---|---|
| first_text_ms | 114742 | 188018 |
| total_ms | 114742 | 188018 |

`first_text_ms == total_ms` on every research row: the research answer
arrives as one `answer_delta` at the end. Citations: 0 on all 24 rows.

## Per prompt (first_ms / total_ms / citations, 3 runs)

| id | run 1 | run 2 | run 3 |
|---|---|---|---|
| hello | 5298 / 5385 / 0 PASS | 1684 / 1695 / 0 PASS | 3474 / 3578 / 0 PASS |
| time | 6950 / 7107 / 0 PASS | 7230 / 7353 / 1 PASS | 6383 / 6574 / 0 PASS |
| clarify_advice | 2 / 2 / 0 PASS | 2 / 2 / 0 PASS | 2 / 2 / 0 PASS |
| aapl_buy_now | 5051 / 7776 / 0 PASS | 8782 / 12611 / 0 FAIL total | 3323 / 4708 / 0 PASS |
| aapl_ceo | 117561 / 117561 / 0 FAIL | 108019 / 108019 / 0 FAIL | 157548 / 157548 / 0 FAIL |
| nvda_risk | - / 169095 / 0 FAIL | 106522 / 106522 / 0 FAIL | 115227 / 115227 / 0 FAIL |
| aapl_insider | - / 168196 / 0 FAIL | 239956 / 239956 / 0 FAIL | 138126 / 138126 / 0 FAIL |
| gme_short | 95980 / 95980 / 0 FAIL | 111073 / 111073 / 0 FAIL | 102220 / 102220 / 0 FAIL |
| msft_revenue | 80433 / 80433 / 0 FAIL | 114617 / 114617 / 0 FAIL | 350574 / 350574 / 0 FAIL |
| nvda_amd_compare | - / 58332 / 0 FAIL | 105495 / 105495 / 0 FAIL | 124876 / 124876 / 0 FAIL |
| tsla_move | - / 143307 / 0 FAIL | 111089 / 111089 / 0 FAIL | 188018 / 188018 / 0 FAIL |
| pltr_bull_bear | 114742 / 114742 / 0 FAIL | 128010 / 128010 / 0 FAIL | 169841 / 169841 / 0 FAIL |

`-` = no `answer_delta` before stream end (transport/timeout row).
Run-1 `None` routes for nvda_risk/aapl_insider/tsla_move were the 130 s
runner guard truncating streams; `TIMEOUT_S=400` fixed those in runs 2-3.
Remaining totals (100-350 s) are server-side: two 120 s walls plus
unbounded persona time (`kernel.ts` `REQUEST_WALL_MS`).

## What each Phase 2 step must beat

| prompt class | current | target |
|---|---|---|
| direct (hello/time/clarify) | pass, 2 ms - 7 s | keep passing |
| aapl_buy_now (reasoning) | 5-13 s, flaky over 10 s | first < 10 s always |
| research ×8 | p90 188 s, 0 citations | first p90 < 10 s, done p90 < 45 s, ≥ 1 citation |
