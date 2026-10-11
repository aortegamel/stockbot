# get_short_interest_leaderboard

Domain: finra
Family: screens
Intent: rank_short_interest
Output kind: leaderboard
Source: finra_sec
Entity scope: market_wide
Time mode: latest_or_as_of

Market-wide most-shorted screen: ranked stocks by short interest as a percent of SEC shares.

## Choose when

- Screening which stocks are the most shorted across the market.

## Reject when

- Do NOT use for one ticker's short interest (get_short_interest).
- Not percent of public float and not real-time. Excludes funds, ETFs, preferreds, and unmapped symbols.

## Conflicts with

- get_short_interest

## Related tools

- get_short_interest

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `as_of` (string): Optional knowledge horizon (YYYY-MM-DD). Only data knowable on or before this date is used. Defaults to today; pass an explicit date for a historical screen.
- `limit` (integer): Number of ranked stocks to return; default 10, maximum 25.
- `settlement_date` (string): Optional FINRA settlement date (YYYY-MM-DD). Omit for the latest published FINRA cycle.
