# get_governance_events

Domain: governance
Family: events
Intent: retrieve_governance_events
Output kind: event_series
Source: sec
Entity scope: single_security
Time mode: since_or_as_of

Proxy and governance filing context (DEF 14A, meetings, votes) with retrieval pointers.

## Choose when

- Finding shareholder-meeting, proxy-vote, or board-compensation records.

## Reject when

- Not for merger-deal status.

## Conflicts with

None

## Related tools

- get_transaction_status

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `since` (string): YYYY-MM-DD; events known on or after this date.
