# get_offering_history

Domain: offerings
Family: capital-raising
Intent: retrieve_offering_history
Output kind: offering_series
Source: sec
Entity scope: single_security
Time mode: latest_or_as_of

Offering history from S-1/S-3/424B filings: offering terms with source-registration links.

## Choose when

- Reviewing past offerings, shelf registrations, or IPO terms for a ticker, including share-count impact context for converts or warrants.

## Reject when

- Not for dilution math (get_dilution_profile).

## Conflicts with

- get_dilution_profile

## Related tools

- get_dilution_profile

## Prerequisites

None

## Required arguments

- `ticker` (string)

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD.
- `limit` (integer)
