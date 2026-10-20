# update_watchlist

Domain: market
Family: watchlist
Intent: update_watchlist
Output kind: watchlist_update
Source: local
Entity scope: single_security
Time mode: current

Add one stock to, or remove one stock from, the local watchlist.

## Choose when

- The user asks to add a stock to, or remove a stock from, the watchlist.

## Reject when

- The receipt is not factual research evidence. Not for questions about watchlist stocks.

## Conflicts with

None

## Related tools

None

## Prerequisites

None

## Required arguments

- `action` (string)
- `ticker` (string)

## Optional arguments

None
