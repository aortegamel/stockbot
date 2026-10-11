# list_sec_documents

Domain: sec
Family: filing-catalog
Intent: list_filing_documents
Output kind: record_series
Source: sec
Entity scope: single_filing
Time mode: as_of

Index of documents and exhibits attached to one filing, looked up by accession number.

## Choose when

- Listing documents and exhibits attached to a known filing accession.

## Reject when

- Do NOT use for document text windows (get_sec_document).
- Do NOT use for filing metadata records (get_sec_filing).
- Does not discover filings. Find an unknown accession with list_sec_filings first.

## Conflicts with

- get_sec_document
- get_sec_filing

## Related tools

- get_sec_filing
- get_sec_document
- list_sec_filings

## Prerequisites

None

## Required arguments

- `accession_no` (string): SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.

## Optional arguments

- `as_of` (string): Point-in-time date YYYY-MM-DD; filings known after it are excluded.
