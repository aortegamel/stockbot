# research_read_search

Domain: research
Family: session
Intent: read_search_hits
Output kind: current_snapshot
Source: local
Entity scope: single_session
Time mode: latest_or_as_of

Page the persisted ranked hit universe of one SEC search, with retrieval truth.

## Choose when

- Reading hits beyond the compact top_hits/additional_hits packet of a search_sec_filings result.
- Paging a large search to exhaustion instead of rerunning the search with a higher limit.

## Reject when

- Not for running a new SEC search (search_sec_filings).
- Hits are navigation artifacts, not evidence. Open the filing and cite a raw passage first.

## Conflicts with

None

## Related tools

- research_read
- research_add_evidence

## Prerequisites

None

## Required arguments

- `search_id` (string): Search id returned by search_sec_filings.
- `session_id` (string): Research session ID returned by search_sec_filings (direct calls create one).

## Optional arguments

- `forms` (array): Optional form filter, e.g. ["10-K", "8-K"].
- `limit` (integer): Hits per page (default 50).
- `offset` (integer): Hits to skip, best score first (default 0).
