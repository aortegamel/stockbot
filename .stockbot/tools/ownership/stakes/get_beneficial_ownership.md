# get_beneficial_ownership

Domain: ownership
Family: stakes
Intent: retrieve_current_beneficial_owners
Output kind: current_snapshot
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Current 5%+ beneficial-ownership stakes (SC 13D/G): holder, shares, percent, voting powers.

## Choose when

- Finding who owns more than 5% of a company.

## Reject when

- Not for stake changes over time (get_ownership_changes).
- Not for relationship links in either direction (search_sec_relationships).
- Answer from these records; do not open filings or pull changes unless asked.

## Conflicts with

- get_ownership_changes
- search_sec_relationships

## Related tools

- get_ownership_changes
- search_sec_relationships

## Prerequisites

None

## Required arguments

- `ticker` (string): Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `company_name` (string): Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.
- `limit` (integer)
