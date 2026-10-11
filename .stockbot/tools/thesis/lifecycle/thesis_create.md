# thesis_create

Domain: thesis
Family: lifecycle
Intent: create_thesis
Output kind: governed_action
Source: local
Entity scope: single_thesis
Time mode: current

Start a new investment thesis proposal with scope, claims, and open questions.

## Choose when

- Creating a new investment thesis to track and test.

## Reject when

- Not for reading an existing thesis.
- Not for changing an existing thesis ID (thesis_refine).

## Conflicts with

None

## Related tools

- thesis_show
- thesis_refine

## Prerequisites

None

## Required arguments

- `user_thesis` (string): The user's investment thesis in their own words.

## Optional arguments

- `assumptions` (array)
- `claims` (array)
- `expressions` (array)
- `invalidators` (array)
- `questions` (array)
- `scope` (string): Ticker scope (e.g. NVDA) or 'unknown'.
- `unknowns` (array)
