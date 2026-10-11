# get_macro_context

Domain: macro
Family: statistics
Intent: retrieve_macro_statistics
Output kind: statistic_series
Source: datacommons
Entity scope: geography
Time mode: latest

Macro statistics for a geography such as California: population (how many people live there), unemployment, inflation, GDP, rates.

## Choose when

- Retrieving population, labor, inflation, GDP, or rate statistics for a geography.

## Reject when

- Not for company-specific facts.
- Do NOT use for outside news, commentary, or why a stock moved (search_web).
- Needs explicit geography and variable IDs.

## Conflicts with

- search_web

## Related tools

- search_web

## Prerequisites

None

## Required arguments

- `geos` (array): Geography DCIDs, e.g. [geoId/06].
- `variables` (array): Statistical variable IDs, e.g. [Count_Person].

## Optional arguments

- `end_date` (string): Range end YYYY-MM-DD.
- `limit` (integer): Max observations (default 100).
- `start_date` (string): Range start YYYY-MM-DD.
