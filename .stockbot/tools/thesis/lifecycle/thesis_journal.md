# thesis_journal

Domain: thesis
Family: lifecycle
Intent: append_thesis_journal
Output kind: governed_action
Source: local
Entity scope: single_thesis
Time mode: current

Append an operator note or journal entry to a thesis log. Pass the thesis ID as thesis:<uuid>.

## Choose when

- Appending an operator note about ongoing monitoring without creating or changing a watch rule.

## Reject when

- Not for revising thesis claims or deltas (thesis_refine).
- Not for setting alerts (thesis_watch).
- Active theses only.

## Conflicts with

- thesis_watch
- thesis_refine

## Related tools

- thesis_refine

## Prerequisites

None

## Required arguments

- `body` (string): Note body (Markdown).
- `id` (string): Thesis ID or slug.

## Optional arguments

- `known_at` (string): PIT cutoff this entry is known at (ISO-8601).
- `run_id` (string): Live run this entry completes (trigger-linked only).
- `title` (string)
- `trigger_id` (string): Trigger this entry completes (omit for ordinary notes).
