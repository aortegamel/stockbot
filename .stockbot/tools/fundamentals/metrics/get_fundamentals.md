# get_fundamentals

Domain: fundamentals
Family: metrics
Intent: retrieve_reported_metric
Output kind: metric_snapshot
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Single reported fundamental for one ticker: EPS, dividends, balance-sheet item, or shares outstanding.

## Choose when

- One specific reported historical numeric fundamental for one ticker: basic/diluted/TTM EPS, dividends, or shares outstanding.

## Reject when

- Do NOT use for full statements (get_financial_statements).
- Do NOT use for XBRL facts by concept (get_xbrl_facts).
- Do NOT use for cheap-vs-expensive multiples (get_valuation_metrics).
- Do NOT use for forward consensus (get_analyst_estimates).
- Not for public float: shares_outstanding is SEC-reported shares outstanding. Undeclared dividend estimates are never included.

## Conflicts with

- get_analyst_estimates
- get_financial_statements
- get_valuation_metrics
- get_xbrl_facts

## Related tools

- get_xbrl_facts
- get_financial_statements
- get_valuation_metrics
- get_analyst_estimates

## Prerequisites

None

## Required arguments

- `metric` (string)
- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time query date YYYY-MM-DD; store-backed for eps/shares_outstanding/dividends; live results are labeled data_source=live.
