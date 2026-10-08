# list_sec_filings

Domain: sec
Family: filing-catalog
Intent: list_entity_filings
Output kind: filing_series
Source: sec
Entity scope: single_entity
Time mode: date_range_or_as_of

List EDGAR filings for an exact ticker or CIK (Required: identifier, e.g. identifier="AAPL"); filterable by form and date range.

## Choose when

- Listing what a company filed lately; recent filings for an exact ticker or CIK, optionally filtered by form or date.
- Required identifier (ticker or CIK, e.g. identifier="AAPL"); optional forms, start_date, end_date, as_of, limit.

## Reject when

- Do not guess an identifier from a bare company name; use the exact ticker when known, otherwise resolve the company's exact identifier first.
- Do NOT use for disclosure search without known identifier (search_sec_filings).
- Do NOT use for 8-K-derived what-changed event feed since a date (get_material_events).

## Conflicts with

- search_sec_filings
- get_material_events

## Related tools

- get_sec_filing
- search_sec_filings
- find_sec_entities
- get_material_events

## Prerequisites

None

## Required arguments

- `identifier` (string): Ticker or CIK, e.g. AAPL.

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; filings known after it are excluded.
- `end_date` (string): YYYY-MM-DD.
- `forms` (array)
- `limit` (integer)
- `start_date` (string): YYYY-MM-DD. Combined with end_date as a range.
