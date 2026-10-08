# search_sec_filings

Domain: sec
Family: filing-search
Intent: search_filing_text
Output kind: search_results
Source: sec
Entity scope: multi_entity
Time mode: date_range_or_as_of

General EDGAR full-text disclosure search across entity, EFTS, and 10-K/10-Q routes, with mentions.

## Choose when

- Searching disclosed filing text, risk-factor language, and mentions when the accession number is unknown.
- SEC filings or filing full-text search when accession is unknown.
- Required: at least one of query, ticker, cik, company_name, person_name, domain, accession_no, security_identifier; e.g. query="risk factors", ticker="AAPL".

## Reject when

- Not a filing lister for a known ticker (list_sec_filings).
- Do NOT use for year-over-year risk-factor changes (diff_risk_factors).
- Do NOT use for full-filing diffs between accessions (diff_sec_filings).
- Do NOT use for one filing metadata record by accession (get_sec_filing).
- Not for the quick bounded lookup (search_sec_filings_bounded).

## Conflicts with

- diff_risk_factors
- diff_sec_filings
- list_sec_filings
- get_sec_filing
- search_sec_filings_bounded

## Related tools

- search_sec_filings_bounded
- list_sec_filings
- get_sec_filing
- find_sec_entities
- diff_risk_factors

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
- `exhaustive` (boolean): true drains every applicable route (the default when dispatched inside a research session); false requests the quick bounded lookup (default outside a research session).
- `forms` (array)
- `limit` (integer): Max hits in the returned packet (default 20); under exhaustive retrieval it does not reduce retrieval.
- `person_name` (string)
- `query` (string)
- `security_identifier` (string): Ticker, CUSIP, ISIN, or class title; never treated as issuer identity.
- `start_date` (string): YYYY-MM-DD. Combined with end_date as a range.
- `ticker` (string)
