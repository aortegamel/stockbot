# query_finra

Domain: finra
Family: short-interest
Intent: analyze_historical_finra_records
Output kind: distribution_or_trend
Source: finra
Entity scope: single_dataset
Time mode: date_range

Analyzed FINRA briefing with trends and metrics over any named dataset, no raw rows.

## Choose when

- Analyzing a FINRA dataset's coverage, distribution, and changes over time.

## Reject when

- Do NOT use for exact source values (get_finra_datapoints).
- Do NOT use for one ticker's current short position (get_short_interest).

## Conflicts with

- get_finra_datapoints
- get_short_interest

## Related tools

- describe_finra_dataset
- get_finra_datapoints
- get_short_interest
- list_finra_datasets

## Prerequisites

None

## Required arguments

- `dataset` (string): Canonical id group/name (e.g. otcMarket/regShoDaily); unambiguous bare names resolve, unknown/ambiguous ones are rejected.

## Optional arguments

- `analysis_goal` (string): Optional: what the user needs answered (e.g. 'trend over the last 12 months'). Guides the briefing; deterministic metrics are always computed.
- `end_date` (string): YYYY-MM-DD.
- `filters` (array): Extra compare filters (field names must exist on the dataset — when unknown, call describe_finra_dataset first).
- `limit` (integer): Max records to return (clamped to 1..1000).
- `offset` (integer): 0-based record offset for pagination (FINRA max 500000). Rejected for datasets whose catalog entry has supportsRecordOffset=false.
- `start_date` (string): YYYY-MM-DD. Combined with end_date as a range.
- `ticker` (string): Issue symbol when the dataset is symbol-level.
