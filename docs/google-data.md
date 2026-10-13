# Google public-data layer (branch `google-data`)

Bounded consumer-interest discovery, social-arbitrage research, macro and
geographic context, patents, weather, and developer adoption. SEC EDGAR stays
authoritative; Google sources are research evidence only. Everything works
offline without credentials; every failure is explicit.

## Credentials (three separate things)

- `GOOGLE_CLOUD_API_KEY` — shared Cloud key for **YouTube Data v3 only**.
- `DATACOMMONS_API_KEY` — separate key for **Data Commons V2** (required;
  without it the source reports `DATACOMMONS_AUTH_REQUIRED` with zero HTTP).
- BigQuery uses **ADC plus `GOOGLE_CLOUD_PROJECT`** and never an API key,
  against a dedicated billing-disabled project (verified before every
  submission; enabled/unknown billing refuses).
The BigQuery project must also have the Cloud Billing API (`cloudbilling.googleapis.com`) enabled: `_billing_state()` calls `v1/projects/{project}/billingInfo` before every real query and fails closed (`billing_unknown`) otherwise; queries themselves stay in the free tier against a billing-disabled project.

There is no `YOUTUBE_API_KEY`, `BIGQUERY_ENABLED`, `DATACOMMONS_ENABLED`, or
`YOUTUBE_ENABLED`. There is no `GOOGLE_KG_API_KEY`: Knowledge Graph entity
resolution was removed; investigation uses SEC-confirmed entity mappings.

Setup: copy `.env.example`, set `GOOGLE_DATA_ENABLED="true"` plus only the
keys for sources you use. Missing keys/project/ADC are explicit external
prerequisites — the repo never creates credentials or touches Cloud IAM.

## Byte and quota caps

- BigQuery per-query cap `BIGQUERY_MAX_BYTES_PER_QUERY` (default 1 GiB);
  estimates over it refuse with `BIGQUERY_QUERY_TOO_LARGE` and zero submits.
- Daily `BIGQUERY_DAILY_BYTES_LIMIT` (default 10 GiB) and monthly
  `BIGQUERY_MONTHLY_BYTES_LIMIT` (default 500 GiB) reservations; exhausted
  day/month refuses with `BIGQUERY_FREE_LIMIT_REACHED` and zero submits.
- YouTube search reservations `YOUTUBE_SEARCH_DAILY_LIMIT` (default 80);
  the 81st search refuses with `YOUTUBE_QUOTA_EXHAUSTED`. Failed HTTP still
  consumes one unit.

## Available vs pending

Available: BigQuery Trends daily tables (US top/rising, international
top/rising), Data Commons observations, YouTube analytics (transient,
15-minute display), patents detail/stats, Census ACS detail, NOAA GSOD,
Stack Overflow tag activity. Pending: official Trends API
(`GOOGLE_TRENDS_API_ENABLED="false"`; reservation returns
`GOOGLE_TRENDS_API_PENDING_ACCESS`) and hourly Trends tables
(`hourly_unavailable` in coverage).

Data Commons filters observation periods with inclusive date bounds.
Year-only dates cover the full calendar year.
Month-only dates cover the full calendar month, including leap days.
Full dates cover one day.
The source keeps observations whose periods overlap the requested window.
Each facet retains its original date, value, unit, provider, and metadata.

## Retention and point-in-time

Observations land in the `google_observations` warehouse
(`observation_id=sha256(source|table|period|geo|term|list_kind)`,
`content_hash` over metrics, features, evidence, and `calc_version`). Identical rows preserve the first
`known_at`; revised content is invisible before its new `known_at`. Only the
last ~30 Trends `refresh_date` partitions are queryable (TTL); each carries
rolling 5-year `week` backfill. Scores/ranks compare only within one
refresh/week/geo/granularity. Omitted `week_start`/`week_end` default to the
trailing 14-day week window ending at `end_date`; deeper history needs
explicit `week_start`/`week_end`. `US`/country rows are grouped cross-subregion `AVG(score)` with `dma_count`/`region_count` and carry no Google national rank; `score` with `score_basis=mean_list_score_where_listed` is the mean source score among subregions where the term appears in the published list (non-listed subregions contribute no row and are excluded), not a Google-provided nationwide score. YouTube titles/counts are never persisted
(RAM-only panel); only quota counters touch disk.

## Error codes

`GOOGLE_DATA_DISABLED`, `DATACOMMONS_DISABLED`,
`DATACOMMONS_AUTH_REQUIRED` (auth), `DATACOMMONS_RATE_LIMITED`,
`BIGQUERY_QUERY_TOO_LARGE`, `BIGQUERY_FREE_LIMIT_REACHED`,
`YOUTUBE_QUOTA_EXHAUSTED`, `GOOGLE_TRENDS_API_DISABLED`,
`GOOGLE_TRENDS_API_PENDING_ACCESS`, plus per-query `source_unavailable`,
`cost_limit_exceeded`, `monthly_limit_exceeded`, `daily_limit_exceeded`,
`billing_enabled`, `billing_unknown`, `ledger_corrupt`, `missing_coverage`,
`unsupported_join`, `malformed_row`, `invalid_params`, `invalid_config`.

## Smoke (live, chargeable — explicit authorization only)

```bash
RUN_BIGQUERY_SMOKE=1 venv/bin/python -m pytest tests/test_google_data.py -q -k smoke_bigquery
RUN_DATACOMMONS_SMOKE=1 venv/bin/python -m pytest tests/test_google_data.py -q -k smoke_datacommons
RUN_YOUTUBE_SMOKE=1 venv/bin/python -m pytest tests/test_google_data.py -q -k smoke_youtube
RUN_GOOGLE_TRENDS_API_SMOKE=1 venv/bin/python -m pytest tests/test_google_data.py -q -k smoke_trends_api
```

Without markers the suite reports offline proof and live validation stays
unperformed; official Trends is not a merge blocker. Tool discovery stays in
`scripts/verify_tool_registry.py`; there is no separate Google tool list.
