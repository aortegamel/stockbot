# search_web

Domain: web
Family: search
Intent: search_external_web
Output kind: search_results
Source: exa
Entity scope: open_query
Time mode: latest

External web news and commentary for price moves, headlines, and industry developments: what outside commentators and people are saying, business risks.

## Choose when

- Finding recent news, announcements, catalysts, market reaction, why a stock moved/rose/fell, or recent commentary outside structured sources.

## Reject when

- Not for FINRA short data.
- Do NOT use for bounded geography statistics like population or rates (get_macro_context).
- Not for exact financial facts, portfolio state, point-in-time history, mandate math, or deterministic screens.
- Never for market-wide screening or historical completeness claims. The published_at date differs from retrieved_at.
- Not for patent records (search_company_patents).
- Never put account, portfolio, or personal identifiers in the query.

## Conflicts with

- get_macro_context

## Related tools

- get_material_events
- query_finra
- get_macro_context

## Prerequisites

None

## Required arguments

- `query` (string): Search query: ticker/company/industry plus the research question. Never include account, portfolio, or personal identifiers.

## Optional arguments

- `category` (string): Optional category to narrow the search.
- `end_published_date` (string): Optional end publication date YYYY-MM-DD, inclusive.
- `exclude_domains` (array): Optional domains to exclude from results.
- `include_domains` (array): Optional domains to restrict results to.
- `limit` (integer): Maximum results, 1-25 (default 5).
- `search_type` (string): Optional search mode (default auto).
- `start_published_date` (string): Optional start publication date YYYY-MM-DD, inclusive.
