# get_sec_search_coverage

Domain: sec
Family: filing-search
Intent: inspect_ingestion_coverage
Output kind: coverage_status
Source: sec
Entity scope: dataset_partition
Time mode: current

Persisted SEC ingestion coverage and backfill-job status for a form, source, or date partition.

## Choose when

- Checking whether a form or date partition is covered or still queued before searching.

## Reject when

- Does not retrieve filing content.
- Never infers completeness from result rows. Reads persisted coverage and backfill jobs only.

## Conflicts with

None

## Related tools

- search_sec_filings

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `form` (string)
- `limit` (integer)
- `search_id` (string)
- `source` (string)
