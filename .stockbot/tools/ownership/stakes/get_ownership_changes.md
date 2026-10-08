# get_ownership_changes

Domain: ownership
Family: stakes
Intent: compare_ownership_stakes
Output kind: change_series
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Deterministic diffs between a holder's consecutive 13D/G filings: share and percent changes, changed stakes and positions.

## Choose when

- Comparing consecutive 13D/G filings for changes in a holder's stake.

## Reject when

- Not for the current snapshot of holders (get_beneficial_ownership).
- Not for relationship links in either direction (search_sec_relationships).

## Conflicts with

- get_beneficial_ownership
- search_sec_relationships

## Related tools

- get_beneficial_ownership
- search_sec_relationships

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `limit` (integer)
