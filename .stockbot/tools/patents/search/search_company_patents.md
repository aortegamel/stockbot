# search_company_patents

Domain: patents
Family: search
Intent: search_company_patents
Output kind: patent_records
Source: google_patents
Entity scope: single_company
Time mode: date_range_or_latest

Company patent search: publications, assignees, counts, and classifications.

## Choose when

- Finding patents a company filed or patented lately, with publication counts and classifications.

## Reject when

- Not for financial or filing questions.
- Answer from patent records, never search_web. Counts are publications, never inventions or bullish signals.
- Needs documented assignee names for the company.

## Conflicts with

None

## Related tools

None

## Prerequisites

None

## Required arguments

- `assignees` (array): Documented assignee aliases (verified, never inferred from matching text).
- `company_id` (string): Documented assignee name from existing company evidence.

## Optional arguments

- `end_date` (string): Range end YYYY-MM-DD.
- `limit` (integer): Max publications (default 20).
- `start_date` (string): Range start YYYY-MM-DD.
