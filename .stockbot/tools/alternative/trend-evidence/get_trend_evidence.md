# get_trend_evidence

Domain: alternative
Family: trend-evidence
Intent: retrieve_known_trend_evidence
Output kind: evidence_series
Source: google_trends
Entity scope: single_topic
Time mode: date_range

Evidence for one known trend: search interest, rising queries, and geography.

## Choose when

- Backing a known trend claim with dated, geography-specific search-interest evidence.

## Reject when

- Not for discovering new signals.
- Not for search-volume claims. Returns top/rising list membership only.

## Conflicts with

None

## Related tools

- find_alternative_signals

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `end_date` (string): Range end YYYY-MM-DD. Optional; both omitted defaults to trailing 7 days ending today UTC.
- `geo` (string): Single geography shorthand for geos.
- `geos` (array): Geographies, e.g. [US].
- `limit` (integer): Max rows (default 100).
- `start_date` (string): Range start YYYY-MM-DD. Optional; both omitted defaults to trailing 7 days ending today UTC.
- `term` (string): Optional substring filter over collected terms.
- `week_end` (string): Interest-week end YYYY-MM-DD; omitted defaults to the trailing 14-day week window ending at end_date.
- `week_start` (string): Interest-week start YYYY-MM-DD; omitted defaults to the trailing 14-day week window ending at end_date.
