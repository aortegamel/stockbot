# get_obligations

Domain: fundamentals
Family: obligations
Intent: retrieve_future_obligations
Output kind: obligation_schedule
Source: sec
Entity scope: single_security
Time mode: latest

Future cash obligations from 10-K/10-Q notes: amounts, horizons, certainty language.

## Choose when

- What a company is obligated to pay in the future for one ticker, including contracts and commitments.

## Reject when

- Do NOT use for valuation multiples (get_valuation_metrics).
- Do not resolve a ticker first. A company_name alone works (the server maps the name).

## Conflicts with

None

## Related tools

- get_valuation_metrics
- get_financial_statements

## Prerequisites

None

## Required arguments

- `ticker` (string): Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.

## Optional arguments

- `company_name` (string): Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.
