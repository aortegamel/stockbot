# get_transaction_status

Domain: transactions
Family: deals
Intent: retrieve_transaction_status
Output kind: event_series
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

M&A filing context: tender offers, 14D-9 recommendations, S-4s, and merger proxies.

## Choose when

- Checking merger, acquisition, or tender-offer filing context for a ticker.

## Reject when

- Not for governance or proxy votes.

## Conflicts with

None

## Related tools

- get_governance_events
- get_sec_document

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `limit` (integer)
