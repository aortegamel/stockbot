# get_planned_insider_sales

Domain: insider
Family: trades
Intent: retrieve_planned_insider_sales
Output kind: notice_series
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Planned Form 144 sale notices not yet executed: proposed insider sales for one ticker.

## Choose when

- Proposed insider sales reported on Form 144 for one ticker.

## Reject when

- Do NOT use for completed insider trades (get_insider_activity).

## Conflicts with

- get_insider_activity

## Related tools

- get_insider_activity

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `limit` (integer)
