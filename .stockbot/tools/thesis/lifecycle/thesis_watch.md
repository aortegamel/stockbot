# thesis_watch

Domain: thesis
Family: lifecycle
Intent: list_or_add_thesis_watch
Output kind: governed_action
Source: local
Entity scope: single_thesis
Time mode: current

List existing watch rules, or add a validated monitoring rule that alerts when a thesis condition triggers.

## Choose when

- Listing what is watched for a thesis, or setting an alert on an invalidator or trigger; what am I watching for, watch rules.

## Reject when

- Not for logging notes (thesis_journal).
- Not for reading thesis status and assessment (thesis_show).
- Never modifies existing watch rules.

## Conflicts with

- thesis_journal
- thesis_show

## Related tools

- thesis_show

## Prerequisites

None

## Required arguments

- `id` (string): Thesis ID or slug.

## Optional arguments

- `claim_ids` (array)
- `expression_ids` (array)
- `rule_type` (string): Semantic monitor name to add (omit to only list rules).
