# find_sec_entities

Domain: sec
Family: entity-discovery
Intent: resolve_sec_entity
Output kind: candidate_records
Source: sec
Entity scope: entity_query
Time mode: current

Resolve a company name, ticker, or CIK to verified SEC entity candidates with CIKs and tickers.

## Choose when

- Starting from a company name when the exact ticker or CIK is not known.

## Reject when

- Unneeded when the exact ticker or CIK is already known.
- Not for the quick bounded lookup (find_sec_entities_bounded).

## Conflicts with

- find_sec_entities_bounded

## Related tools

- find_sec_entities_bounded
- list_sec_filings
- search_sec_filings

## Prerequisites

None

## Required arguments

- `query` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; former names apply only within their known/valid interval.
- `exhaustive` (boolean): true searches every entity route (the default when dispatched inside a research session); false requests the quick bounded lookup (default outside a research session).
- `limit` (integer): Bounded lookups return at most this many candidates (default 20); exhaustive lookups return every candidate found across routes (local source cap 50) and limit only bounds the display packet.
