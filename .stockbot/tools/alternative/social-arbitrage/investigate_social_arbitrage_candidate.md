# investigate_social_arbitrage_candidate

Domain: alternative
Family: social-arbitrage
Intent: assess_attention_demand_gap
Output kind: derived_analysis
Source: google_trends
Entity scope: single_topic
Time mode: latest_or_as_of

Enrichment of one social-arbitrage candidate with corroboration and exposure gap. Social signals vetting.

## Choose when

- Testing whether online attention around one candidate corresponds to real demand.

## Reject when

- Not for broad signal discovery.
- Never establishes causality and never trades.

## Conflicts with

None

## Related tools

- find_alternative_signals
- get_trend_evidence

## Prerequisites

None

## Required arguments

- `term` (string): Discovery term to investigate.

## Optional arguments

- `geo` (string): Geography, e.g. US.
- `limit` (integer): Max evidence rows per source (default 5; YouTube never above 5).
