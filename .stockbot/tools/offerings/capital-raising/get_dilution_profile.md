# get_dilution_profile

Domain: offerings
Family: capital-raising
Intent: calculate_dilution
Output kind: derived_analysis
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Deterministic dilution math for diluted shareholders: inputs, formula, and source accessions always shown.

## Choose when

- Quantifying share-count impact from offerings, converts, or warrants.

## Reject when

- Not for offering-terms history (get_offering_history).

## Conflicts with

- get_offering_history

## Related tools

- get_offering_history

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
