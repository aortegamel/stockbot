# thesis_refine

Domain: thesis
Family: lifecycle
Intent: refine_thesis
Output kind: governed_action
Source: local
Entity scope: single_thesis
Time mode: current

Update a thesis with clarifications and deltas. Pass the thesis ID as thesis:<uuid>.

## Choose when

- Revising a thesis after new evidence or feedback.

## Reject when

- Not for routine operator notes without changing claims (thesis_journal).
- Refuses paused or closed theses.

## Conflicts with

- thesis_journal

## Related tools

- thesis_show
- thesis_journal

## Prerequisites

None

## Required arguments

- `clarification` (string): New information or correction in the user's own words.
- `id` (string): Thesis ID or slug.

## Optional arguments

- `assumptions` (array)
- `claims` (array)
- `expressions` (array)
- `invalidators` (array)
- `questions` (array)
- `scope` (string): Ticker scope (e.g. NVDA) or 'unknown'.
- `unknowns` (array)
