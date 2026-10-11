# search_sec_relationships

Domain: ownership
Family: stakes
Intent: search_ownership_relationships
Output kind: relationship_records
Source: sec
Entity scope: single_entity
Time mode: latest_or_as_of

Ownership and transaction relationship links for an entity: 13D/G owners, 13F holdings, deal links.

## Choose when

- Mapping who owns, holds, or transacts with an entity in either direction.

## Reject when

- Not for current 5%+ stake sizes (get_beneficial_ownership).
- Not for consecutive-filing stake diffs (get_ownership_changes).
- Mentions never become verified links. Transaction status stays unknown without closing evidence.
- Do not resolve a ticker first. A company_name alone works (the server maps the name).

## Conflicts with

- get_beneficial_ownership
- get_ownership_changes

## Related tools

- get_beneficial_ownership
- get_ownership_changes

## Prerequisites

None

## Required arguments

- `entity` (string): CIK, ticker, entity id, or candidate dict. If unknown, pass company_name instead; never call with neither.

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `company_name` (string): Company name (e.g. Apple) when entity is unknown; the server maps it to a ticker.
- `exhaustive` (boolean): Exhaust all applicable relationship indexes and SEC routes; the returned model context remains bounded.
- `limit` (integer)
- `relationship_types` (array): Optional open-vocabulary type filter (e.g. beneficial_owner, holding_manager).
