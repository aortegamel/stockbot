# get_valuation_metrics

Domain: valuation
Family: multiples
Intent: calculate_valuation_multiples
Output kind: derived_snapshot
Source: yahoo_finance_sec
Entity scope: single_security
Time mode: latest

Cheap-vs-expensive earnings multiples at live price: trailing plus forward P/E.

## Choose when

- Whether a company is cheap or expensive on earnings multiples for one ticker.

## Reject when

- Do NOT use for reported EPS alone (get_fundamentals).
- Do NOT use for forward consensus alone (get_analyst_estimates).
- Anchored to the live price at query time. Not a historical point-in-time valuation.

## Conflicts with

- get_analyst_estimates
- get_fundamentals

## Related tools

- get_analyst_estimates
- get_obligations
- get_fundamentals

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

None
