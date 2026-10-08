# find_sec_entities_bounded

Domain: sec
Family: entity-discovery
Intent: resolve_sec_entity_bounded
Output kind: candidate_records
Source: sec
Entity scope: entity_query
Time mode: current

Quick bounded entity lookup: same verified candidates, fast routes only, capped at limit.

## Choose when

- One identity check when full candidate coverage is not needed.

## Reject when

- Not for full candidate coverage (find_sec_entities).
- Unneeded when the exact ticker or CIK is already known.

## Conflicts with

- find_sec_entities

## Related tools

- find_sec_entities
- list_sec_filings

## Prerequisites

None

## Required arguments

- `query` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; former names apply only within their known/valid interval.
- `limit` (integer): Max candidates returned (default 20).
