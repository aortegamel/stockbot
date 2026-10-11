# find_alternative_signals

Domain: alternative
Family: signal-discovery
Intent: discover_emerging_signals
Output kind: ranked_candidates
Source: google_trends
Entity scope: market_wide
Time mode: latest_or_as_of

Discovery scan for rising search-term and diffusion signals worth investigating.

## Choose when

- Screening for emerging trend or attention signals across terms.

## Reject when

- Not for evidence on one known trend.
- Not for a dated, geography-specific trend question.
- Candidates only, never materiality or investment claims. Features stay null unless exactly one PIT-valid scope matches.

## Conflicts with

None

## Related tools

- get_trend_evidence
- investigate_social_arbitrage_candidate

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; candidates known after it are excluded.
- `geo` (string): Geography filter, e.g. US.
- `limit` (integer): Max candidates (default 20).
- `query` (string): Substring filter over candidate terms.
