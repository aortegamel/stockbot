# diff_sec_filings

Domain: sec
Family: filing-diff
Intent: compare_full_filings
Output kind: diff
Source: sec
Entity scope: filing_pair_or_security
Time mode: latest_or_as_of

Self-contained full-filing diff for one ticker or two accessions: amendment versus prior version.

## Choose when

- Comparing a ticker's latest amendment filing versus its predecessor filing.
- Comparing two known filing accessions for amendment or restatement changes.

## Reject when

- Do NOT use for risk-factor-only year-over-year diffs (diff_risk_factors).
- Do NOT call list_sec_filings first; ticker resolution is internal.
- Do NOT use for disclosure search without change framing (search_sec_filings).

## Conflicts with

- diff_risk_factors
- search_sec_filings

## Related tools

- diff_risk_factors
- get_sec_filing

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; filings known after it are excluded.
- `current_accession` (string): SEC accession number, e.g. 0000320193-25-000079. Named current_accession/previous_accession for diffs.
- `forms` (array)
- `previous_accession` (string): SEC accession number, e.g. 0000320193-25-000079. Named current_accession/previous_accession for diffs.
- `section` (string)
- `ticker` (string)
