# get_short_pressure_profile

Domain: finra
Family: short-interest
Intent: assess_short_pressure
Output kind: derived_composite
Source: finra_sec
Entity scope: single_security
Time mode: latest_or_as_of

Short pressure vs shares outstanding for one ticker: FINRA positioning plus SEC shares and ratio.

## Choose when

- Short positioning relative to shares outstanding for one ticker.

## Reject when

- Do NOT use for biweekly short position alone (get_short_interest).
- Do NOT use for daily short-sale volume (get_reg_sho_volume).
- Not for manipulation or causation questions. Describes positioning only.

## Conflicts with

- get_short_interest

## Related tools

- get_short_interest
- query_finra
- get_reg_sho_volume

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

None
