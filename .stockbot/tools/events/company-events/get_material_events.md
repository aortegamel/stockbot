# get_material_events

Domain: events
Family: company-events
Intent: retrieve_recent_material_events
Output kind: event_series
Source: sec
Entity scope: single_security
Time mode: since_or_as_of

Deterministic 8-K-derived recent event feed with accession citations for what changed since a date.

## Choose when

- Finding what changed recently: recent 8-K-derived events for a company since a date.

## Reject when

- Does not cover market reaction or news commentary.
- Answer from the event feed; do not open filing documents unless the question needs document text.
- Do NOT use for a full filing list by ticker or form (list_sec_filings).

## Conflicts with

- list_sec_filings

## Related tools

- get_sec_document
- search_web
- get_recent_ownership_filings
- list_sec_filings

## Prerequisites

None

## Required arguments

- `since` (string): YYYY-MM-DD; events known on or after this date.
- `ticker` (string): Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `company_name` (string): Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.
