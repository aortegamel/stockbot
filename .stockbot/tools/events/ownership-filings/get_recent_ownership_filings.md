# get_recent_ownership_filings

Domain: events
Family: ownership-filings
Intent: retrieve_latest_ownership_filings
Output kind: filing_series
Source: sec
Entity scope: market_wide
Time mode: latest

Market-wide feed of the most recent SC 13D/13G filings from roughly the last 24 hours.

## Choose when

- Finding the latest market-wide SC 13D/G filings when no ticker is given; what just came out, newly filed.

## Reject when

- Not for one company's current holders.
- Not proof that a filing caused a price move. Do not drill into get_beneficial_ownership unless asked.

## Conflicts with

None

## Related tools

- get_beneficial_ownership
- get_material_events

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `form_type` (string)
- `limit` (integer)
