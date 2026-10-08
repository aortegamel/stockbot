# search_sec_filings_bounded

Domain: sec
Family: filing-search
Intent: search_filing_text_bounded
Output kind: search_results
Source: sec
Entity scope: multi_entity
Time mode: date_range_or_as_of

Quick bounded EDGAR lookup: same filing-text search, fast routes only, capped at limit.

## Choose when

- One mention check when full coverage is not needed.
- Required: at least one of query, ticker, cik, company_name, person_name, domain, accession_no, security_identifier; e.g. query="risk factors", ticker="AAPL".

## Reject when

- Not for all/every-mention or full-coverage questions (search_sec_filings).
- Not a filing lister for a known ticker (list_sec_filings).

## Conflicts with

- search_sec_filings

## Related tools

- search_sec_filings
- list_sec_filings

## Prerequisites

None

## Required arguments

None

## Optional arguments

- `accession_no` (string): SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.
- `as_of` (string): Point-in-time date YYYY-MM-DD; records known after it are excluded.
- `cik` (string)
- `company_name` (string)
- `domain` (string)
- `end_date` (string): YYYY-MM-DD.
- `forms` (array)
- `limit` (integer): Max hits returned (default 20).
- `person_name` (string)
- `query` (string)
- `security_identifier` (string): Ticker, CUSIP, ISIN, or class title; never treated as issuer identity.
- `start_date` (string): YYYY-MM-DD. Combined with end_date as a range.
- `ticker` (string)
