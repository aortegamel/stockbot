# get_reg_sho_volume

Domain: finra
Family: short-sale-volume
Intent: daily_short_sale_volume
Output kind: daily_series
Source: finra
Entity scope: single_security
Time mode: date_range_or_latest

Self-contained daily short-sale volume by venue for one ticker: FINRA Reg SHO volume, rolling 12 months.

## Choose when

- Daily short-sale volume or venue breakdowns for one ticker.

## Reject when

- Do NOT use for biweekly short interest positions (get_short_interest).
- Do NOT call describe_finra_dataset or get_finra_datapoints; dataset and fields resolve internally.

## Conflicts with

- get_short_interest

## Related tools

- get_short_interest
- query_finra

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `company_name` (string): Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.
- `ticker` (string): Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.
- `tradeDate` (string): Optional trade date YYYY-MM-DD.
