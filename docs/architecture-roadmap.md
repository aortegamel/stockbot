# Stock Analyst AI: Target Architecture and Implementation Roadmap

> **Roadmap notice:** This document describes the intended architecture and roadmap. Some components are not implemented yet.

This document is the build plan for turning this repository into a research system that can ingest large amounts of market and company data, find unusual relationships, and let an AI agent investigate them without placing the full dataset in the model's context window.

The intended product is a research assistant and opportunity-ranking system, not an oracle or autonomous trading system. Its edge should come from joining public facts earlier and more consistently than a person can, preserving exactly when each fact became public, and showing the evidence behind every thesis.

## 1. Core architectural decision

Do not make the vector database the main database, and do not ask the LLM to scan the entire corpus during a chat.

Use four different layers:

```text
External sources
    SEC, FINRA, prices, company IR, trials, FDA, publications
                         |
                         v
1. Immutable raw archive
    Original JSON, HTML, XML, PDFs, and response metadata
                         |
                         v
2. Normalized point-in-time data
    Companies, securities, facts, filings, transactions, trials, events
                         |
                         +----------------------+
                         |                      |
                         v                      v
3a. Analytical feature store          3b. Search index
    Ratios, changes, percentiles,          Filing and document chunks
    peer comparisons, signal scores        Dense + lexical retrieval
                         |                      |
                         +-----------+----------+
                                     v
4. Research agent
    Plans an investigation, queries bounded results, retrieves evidence,
    checks contradictions, and writes a sourced thesis
```

The division of responsibility is:

- SQL/analytical storage answers exact questions: values, dates, trends, rankings, joins, screens, and correlations.
- Qdrant answers narrative questions: where management discussed purchase obligations, why guidance changed, which risk language is new, or what a trial endpoint means.
- Deterministic Python calculates financial metrics and signals.
- Extraction models convert narrative disclosures into structured events, always retaining evidence spans.
- The research LLM chooses tools, develops hypotheses, and explains results. It does not perform bulk scanning or become the calculator.

## 2. Recommended technology shape

Keep provider choices behind interfaces so they can change without rewriting the domain logic.

### Initial local/development stack

- Raw archive: local `data/raw/` files, excluded from Git.
- Normalized application data: PostgreSQL when available; SQLite can remain an early development fallback.
- Large scans and backtests: partitioned Parquet queried with DuckDB.
- Narrative search: Qdrant.
- Dense embeddings: configurable provider, initially an OpenAI embedding model.
- Lexical retrieval: Qdrant sparse vectors/BM25 or a separate full-text index.
- Background ingestion: a scheduler plus idempotent worker jobs.
- Research agent loop: needle harness → kernel scheduler (`needle-harness/scripts/needle.ts` → `runKernelAgent` → `app/research/kernel_worker.py` → `app/research/scheduler.py run_node` → `app/tool_runtime.py` → canonical `app/tools.py`).

### Why Qdrant plus embeddings

Qdrant is an index for document chunks, not the authoritative copy of the documents. Store a stable `chunk_id` and metadata in Qdrant; store the canonical text and provenance in the main database/raw archive.

Use hybrid retrieval:

1. Dense embeddings find semantically similar language.
2. Sparse/lexical retrieval finds exact terms such as an XBRL concept, drug name, contract phrase, or accounting term.
3. Metadata filters restrict results by company, form, section, filing date, source, or trial.
4. Rank fusion combines dense and lexical candidates.
5. A reranker selects the most relevant and non-duplicative evidence.

Qdrant supports named dense and sparse vectors, hybrid queries, rank fusion, and payload filtering. Relevant documentation:

- <https://qdrant.tech/documentation/search/text-search/hybrid-search/>
- <https://qdrant.tech/documentation/search/filtering/>

OpenAI also offers hosted vector stores and file search. Those are useful for a fast prototype, but this project will likely need custom chunking, richer domain metadata, point-in-time filtering, retrieval experiments, and provider independence. Qdrant is therefore the preferred long-term index. Keep hosted file search as a possible prototype adapter, not a permanent dependency. OpenAI vector-store documentation: <https://developers.openai.com/api/reference/resources/vector_stores>

### Configuration requirements

Never scatter model names or dimensions through application code. Add configuration such as:

```text
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=<configured-model-name>
EMBEDDING_DIMENSIONS=<configured-dimension-count>
EMBEDDING_VERSION=v1
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=...
QDRANT_COLLECTION=document_chunks_v1
```

Store the embedding provider, model, dimensions, chunking version, and creation time with every indexed chunk. A model or chunking change must create a new index version so the old index can remain usable during migration.

## 3. Target repository structure

The exact names can change, but responsibilities should remain separate.

```text
app/
  api/                         # HTTP request/response layer
  agent/
    orchestrator.py            # Research loop and context budgeting
    planner.py                 # Question -> investigation plan
    tool_registry.py           # Small semantic tool surface
    evidence.py                # Evidence packets and citations
    prompts.py
  domain/
    entities.py                # Company, security, person, drug, trial IDs
    documents.py
    facts.py
    events.py
    signals.py
  normalization.py             # Deterministic SEC/FINRA normalizers
  services/
    research_data.py           # Fetch -> archive -> normalize -> Parquet (CLI: python cli.py refresh-data)
  storage/
    database.py
    raw_archive.py
    repositories.py
    parquet.py
  retrieval/
    chunker.py
    embeddings.py
    qdrant_index.py
    lexical.py
    hybrid.py
    reranker.py
  analytics/
    metrics.py
    features.py
    anomalies.py
    screens.py
    backtests.py
  extraction/
    commitments.py
    insider_transactions.py
    guidance.py
    biotech_events.py
  workers/
    scheduler.py
    jobs.py
migrations/                    # Database migrations
tests/
  unit/
  integration/
  retrieval/
  evals/
  fixtures/
data/                          # Runtime data; ignored by Git
```

Rules for the codebase:

- Connectors fetch data but contain no investment logic.
- Normalizers map provider formats into stable domain schemas.
- Analytics consumes normalized data, never provider response shapes.
- Agent tools call repositories/analytics services, never external APIs directly.
- Every ingestion and extraction step is idempotent and versioned.
- Vendor-specific identifiers are aliases; internal stable IDs are primary keys.

## 4. Canonical point-in-time schema

All important data must preserve when it became knowable. Without this, later machine-learning results will contain look-ahead bias.

### Shared provenance fields

Every fact, document, and event should carry as many of these as apply:

```text
source_name
source_record_id
source_url
document_id
accession_number
retrieved_at
published_at / filed_at
effective_at / period_end
known_at
content_hash
parser_version
extraction_version
evidence_location
confidence
```

`known_at` is the earliest timestamp at which the system believes a market participant could have observed the record. All historical screens and backtests must filter with `known_at <= as_of`.

### Essential tables

#### Identity

- `entities`: companies, people, agencies, subsidiaries, counterparties.
- `securities`: share classes, bonds, options identifiers if licensed.
- `entity_aliases`: ticker history, CIK, LEI, company-name aliases.
- `entity_relationships`: parent/subsidiary, company/counterparty, company/drug sponsor.

Never use ticker as the permanent company identifier. Tickers and names change.

#### Documents

- `documents`: one row per filing, exhibit, release, trial record version, FDA document, or publication.
- `document_sections`: structural sections and tables.
- `document_chunks`: canonical chunk text and its coordinates.
- `document_versions`: hashes and supersession links.

#### Structured facts

- `financial_facts`: canonical concept, original concept, value, unit, duration/instant, dimensions, period, filing, amendment status.
- `financial_statements`: normalized statement presentation and line ordering.
- `short_interest`: settlement date, publication date, position, previous position, average volume, days to cover.
- `short_sale_volume`: trade date, venue/reporting facility, short, exempt, and total volume.
- `market_prices`: adjusted and unadjusted OHLCV plus corporate-action factors.
- `insider_transactions`: owner, relationship, transaction code, shares, price, ownership after transaction, direct/indirect ownership, plan indicators.
- `commitments`: counterparty, amount/range, currency, start/end dates, conditions, accounting treatment, evidence.
- `offerings_and_dilution`: shelves, ATMs, convertibles, warrants, equity plans, repurchases, issuance.

#### Research objects

- `events`: normalized material event with event type, entities, dates, attributes, confidence, and evidence IDs.
- `features`: entity, `as_of`, feature name, value, feature version, calculation lineage.
- `signals`: signal type, score, rank, generated time, feature snapshot, evidence IDs, status.
- `research_runs`: question, plan, tool calls, evidence used, answer, model versions, token/cost metrics.

## 5. Document chunking and indexing

Do not split filings into arbitrary fixed character windows first. Financial documents are structured documents.

### Chunking pipeline

1. Parse the filing/document into sections, subsections, paragraphs, lists, and tables.
2. Preserve SEC item names, note titles, exhibit names, table headers, and page/paragraph coordinates.
3. Split oversized sections into roughly 350-800 token chunks with limited overlap.
4. Keep tables with their titles, headers, units, and nearby explanatory footnotes.
5. Add a short generated descriptor only if it improves retrieval; never replace the original text.
6. Hash the normalized chunk so unchanged text is not embedded again.

### Qdrant point payload

Each point should include:

```json
{
  "chunk_id": "stable-id",
  "document_id": "stable-id",
  "entity_id": "stable-id",
  "security_id": "stable-id-or-null",
  "source": "sec",
  "document_type": "10-Q",
  "section": "MD&A/Liquidity and Capital Resources",
  "accession_number": "...",
  "published_at": "...",
  "known_at": "...",
  "period_end": "...",
  "content_hash": "...",
  "chunking_version": "...",
  "embedding_version": "..."
}
```

Create payload indexes for fields used in filters. Keep complete text in the main store and optionally a copy in the Qdrant payload for fast retrieval.

### Retrieval pipeline

Classify every request before searching:

| Request | Primary path |
|---|---|
| Exact revenue, EPS, shares, dates, ratios | SQL/feature store |
| Narrative reason, risk, tone, contract language | Hybrid document retrieval |
| Screen all companies | Analytical query; retrieve documents only for finalists |
| Mixed thesis | SQL first, then targeted narrative retrieval |

Recommended search sequence:

1. Convert the question into one or more focused search queries.
2. Apply entity, source, form, section, and `known_at` filters before vector search.
3. Retrieve a wider dense and lexical candidate set.
4. Fuse rankings.
5. Rerank the best candidates.
6. Remove near-duplicates and enforce document/date diversity.
7. Return 6-12 evidence chunks by default, with a hard token limit.
8. Let the agent request more evidence explicitly rather than silently adding it.

Create a retrieval evaluation set before tuning chunk sizes or ranking weights. Measure whether the correct evidence appears in the top 5 and top 10 results.

## 6. Agent and tool architecture

The final agent should see a small set of domain tools, not one tool per upstream dataset.

### Proposed tools

```text
resolve_entity(name_or_ticker, as_of=None)
get_company_snapshot(entity_id, as_of, fields=None)
get_metric_series(entity_id, metrics, start, end, frequency)
compare_entities(entity_ids, metrics, as_of)
find_events(entity_id, event_types, start, end, as_of)
screen_entities(universe, conditions, as_of, limit)
search_documents(query, filters, as_of, limit)
get_evidence(evidence_ids)
get_signal(signal_id)
explain_feature(entity_id, feature_name, as_of)
```

Do not expose unrestricted SQL to the model. Use a validated screen/query DSL or strongly typed tool arguments.

### Research loop

1. **Interpret:** identify entity, timeframe, requested outcome, and whether the question is factual, comparative, or investigative.
2. **Plan:** create 2-6 testable hypotheses and specify which data can confirm or reject each.
3. **Survey:** fetch a compact company snapshot and available-data manifest.
4. **Calculate:** query normalized facts and feature tables.
5. **Investigate:** retrieve narrative evidence only for anomalies or missing explanations.
6. **Challenge:** search for disconfirming evidence, stale data, accounting differences, and alternative explanations.
7. **Synthesize:** report thesis, evidence, counterevidence, confidence, data freshness, and unresolved questions.

### Context-window control

- Tool results must declare `row_count`, `returned_count`, `next_cursor`, `truncated`, `as_of`, and source freshness.
- Never truncate serialized JSON in the middle of an object.
- Default numerical responses to a few dozen rows and aggregate larger datasets server-side.
- Store complete intermediate results under `research_run_id`; put only summaries and evidence IDs in the model conversation.
- Compress old conversational turns into a research-state summary.
- Give every run a configurable evidence-token budget.
- Prefer tables of calculated values over raw records.
- Fetch exact excerpts only after the system knows which documents matter.

For corpus-wide questions, use map/reduce outside the chat context:

1. Batch-extract a typed schema from each relevant document.
2. Validate and store the extracted events.
3. Aggregate events with SQL.
4. Give the agent only the aggregate, outliers, and evidence for selected companies.

## 7. Phase-by-phase implementation roadmap

Complete phases in order. A phase is complete only when its acceptance criteria pass.

### Phase 0 - Shared foundations

- [x] Define stable IDs for entities, securities, documents, chunks, events, and evidence.
      (entities/securities/documents done: `sec:cik:…`, `sec:equity:…`, hash-keyed doc IDs;
      chunk/event/evidence IDs land with their phases)
- [ ] Add database migrations and repository interfaces.
- [x] Implement the immutable raw archive with content hashes.
      (`app/storage/raw_archive.py`: write-once `data/raw/` payloads + first-write retrieval manifests.
      Identical bytes dedup to one payload. `refresh_sec_tickers` also appends each successful retrieval
      (sha256, timestamp taken after the fetch, url) to the key's `retrievals.jsonl`. Historical ticker
      lookups take the newest event inside the inclusive 31-day window at `as_of`, hash-verify its bytes,
      and stamp `known_at` with that event's time. A malformed log or bad newest bytes fails closed. The
      31-day TTL bounds snapshot age only: it cannot establish ticker ownership between observations.)
- [x] Add ingestion checkpoints, retry policy, rate limiting, and structured logs.
      (`app/normalization.py` + `app/services/research_data.py`: deterministic normalizers;
      fetch -> archive -> normalize -> Parquet refresh with retry and pacing)
- [x] Add `known_at`, `published_at`, and `effective_at` conventions.
      (SEC facts: `known_at` = filed date; FINRA snapshots: `known_at` = first complete-archive time,
      `published_at` NULL where the source exposes no publication timestamp)
- [x] Add parser/extractor/feature version fields.
      (`parser_version` / calculation version on every dataset and screen run)
- [x] Establish fixture-based tests that never require live APIs.
- [ ] Add data-quality reports: duplicates, gaps, stale sources, schema drift.

Acceptance criteria:

- [x] Re-running an ingestion job creates no duplicates.
- [x] Every normalized record can link back to the exact raw response/document.
- [x] An `as_of` query cannot see records published later.
- [x] A parser change can be replayed from raw data without downloading again.

### Phase 1 - SEC ingestion and normalization

Treat the existing SEC integration as a starting point, then meet this production definition of complete.

- [ ] Backfill submissions and company facts using SEC bulk/API sources.
- [ ] Store filing metadata and original filings/exhibits by accession number.
- [ ] Normalize XBRL facts by concept, unit, dimensions, duration, form, accession, and amendment.
- [ ] Distinguish quarter-only, year-to-date, annual, and instantaneous facts.
- [ ] Build canonical mappings for common metrics while retaining original tags.
- [ ] Parse 10-K, 10-Q, 8-K, S-1/S-3, DEF 14A, Forms 3/4/5, 13D/G, and relevant exhibits.
- [ ] Version and diff narrative sections by accession.
- [ ] Normalize insider transactions rather than storing only Form 4 text.
- [ ] Extract offering, dilution, debt, commitment, guidance, and material-agreement events.
- [ ] Chunk and index narrative sections with evidence coordinates.

Acceptance criteria:

- A metric series agrees with sampled filing values and shows its accession/unit.
- Restatements and amendments do not silently create duplicate quarters.
- Queries work for a historical `as_of` date.
- Every extracted claim links to an exact filing excerpt.
- Retrieval finds known risk, commitment, guidance, and dilution passages in a labeled test set.

### Phase 2 - FINRA ingestion and short-data analytics

- [x] Build dataset definitions with explicit schemas, sorting, pagination, freshness, and retention.
- [x] Archive every FINRA response and request parameters.
- [ ] Backfill consolidated short-interest history.
      (pipeline exists and can backfill any cycle; a full history backfill is an operational run)
- [x] Ingest Reg SHO/short-sale volume separately from short interest.
- [ ] Ingest threshold-list membership and relevant OTC/ATS activity.
- [x] Store settlement date, reporting/publication date, and `known_at` separately.
- [ ] Normalize symbols through security history.
- [ ] Add completeness checks for expected reporting cycles and facilities.
- [x] Add a data dictionary warning that short-sale volume is not short interest.

Derived features:

- [ ] Short interest as percent of defensible float.
      (deliberately deferred; the interim product ranks by SEC-reported shares outstanding only)
- [x] Cycle-over-cycle and multi-cycle short-interest change.
- [ ] Days-to-cover trend.
- [ ] Industry and market-cap peer percentiles.
- [ ] Short interest versus price, volume, and volatility divergence.
- [ ] Short-sale-volume ratios by facility, kept clearly labeled as flow.
- [ ] Threshold-list entry, duration, and exit events.

Acceptance criteria:

- [x] Latest and historical cycles are explicitly sorted and reproducible.
- [x] Publication lag is respected by historical queries.
- [x] Missing cycles/facilities are visible rather than treated as zeros.
- [x] FINRA unit tests use frozen fixtures and cover pagination, errors, and schema changes.
- [x] A `screen_short_setups` query can rank a universe without invoking an LLM.

### Phase 3 - Market and reference data

SEC and FINRA signals are difficult to interpret without prices, corporate actions, and a reliable universe.

- [ ] Choose a licensed price/reference-data provider.
- [ ] Ingest adjusted and unadjusted OHLCV.
- [ ] Ingest splits, dividends, symbol changes, mergers, and delistings.
- [ ] Build historical index/universe membership where possible.
- [ ] Add sector, industry, exchange, country, and market-cap histories.
- [ ] Add options, securities-lending, borrow fee, or analyst-estimate data only if licensing and cost justify them.

Acceptance criteria:

- Historical returns cannot accidentally use today's security universe.
- Prices reconcile across corporate actions.
- Delisted companies remain in backtests.
- Feature queries join by stable security ID rather than current ticker.

### Phase 4 - Search and retrieval platform

- [ ] Implement structure-aware document chunking.
- [ ] Implement embedding batching, caching, retries, and cost reporting.
- [ ] Create versioned Qdrant collections.
- [ ] Create indexed payload fields for common filters.
- [ ] Implement dense, lexical, and hybrid retrieval adapters.
- [ ] Add reranking and duplicate suppression.
- [ ] Implement `search_documents` and `get_evidence` tools.
- [ ] Create labeled retrieval questions and expected evidence chunks.
- [ ] Measure retrieval recall before connecting it to the main agent.

Acceptance criteria:

- Reindexing is resumable and does not re-embed unchanged chunks.
- All results obey entity/date/source filters.
- Top-k recall meets a documented target on the retrieval eval set.
- Returned evidence contains source, date, document ID, and location.
- The model receives a bounded evidence packet, never an entire filing by default.

### Phase 5 - Event extraction and feature engine

Build typed extraction schemas. Do not store a model-written paragraph as the only result.

- [ ] Commitment and purchase-obligation extraction.
- [ ] Shelf, ATM, convertible, warrant, and share-issuance extraction.
- [ ] Guidance issue/revision/withdrawal extraction.
- [ ] Insider-sale classification and cluster detection.
- [ ] Debt, covenant, maturity, refinancing, and going-concern extraction.
- [ ] Customer/supplier/counterparty relationship extraction.
- [ ] Evidence-span validation and confidence scoring.
- [ ] Human review queue for low-confidence or high-impact events.
- [ ] Nightly feature calculation and versioned feature definitions.

Acceptance criteria:

- Extracted numeric values are parsed and validated by deterministic code.
- Every event has evidence and extraction-version lineage.
- Low-confidence events are not promoted into strong signals automatically.
- Feature recomputation is deterministic for a fixed input snapshot.

### Phase 6 - Screening and signal engine

Start with transparent rules and statistics. The system should discover candidates before the agent starts reading documents.

- [ ] Define a versioned universe for each daily run.
- [ ] Calculate peer-relative percentiles and robust z-scores.
- [ ] Detect time-series change points and unusual acceleration/deceleration.
- [ ] Create signal families rather than one opaque master score.
- [ ] Store the complete feature snapshot behind every signal.
- [ ] Add scheduled screens and watchlist alerts.
- [ ] Track whether signals later strengthened, weakened, or resolved.

Initial signal families:

#### Dilution and future funding burden

- Share-count acceleration.
- Stock compensation relative to revenue and free cash flow.
- Repurchases minus issuance.
- Shelf/ATM capacity and recent usage.
- Warrants, convertibles, earn-outs, and equity plans.
- Contractual commitments relative to cash and expected cash generation.
- Capex guidance versus balance-sheet capacity.
- Counterparty-linked commitments to avoid seeing only one side of a contract.

#### Fundamental deterioration plus short pressure

- Revenue/guidance and gross-margin deterioration.
- Receivables or inventory growth exceeding sales growth.
- Accruals and operating-cash-flow divergence.
- Customer concentration or covenant risk.
- Insider-selling clusters after filtering tax, gift, exercise, and plan transactions.
- Rising short interest, days to cover, borrow stress if licensed, and adverse price/volume behavior.

#### Narrative change

- New or materially expanded risk language.
- Guidance wording changes.
- Liquidity language becoming more cautious.
- New commitments, contingencies, or subsequent events.
- Differences between management narrative and calculated financial trends.

Acceptance criteria:

- Every score can be decomposed into named features.
- Every qualitative component links to evidence.
- A signal can be recreated later from its `as_of` snapshot.
- Alerts state what changed, not merely that a score crossed a threshold.

### Phase 7 - Research agent

- [ ] Replace source-specific raw tools with the semantic tools described above.
- [ ] Add explicit planning and hypothesis tracking.
- [ ] Add a research-run store outside the context window.
- [ ] Add per-tool row, byte, and token budgets.
- [ ] Add evidence and contradiction passes.
- [ ] Require claim-level source references for factual conclusions.
- [ ] Report data freshness and missing sources.
- [ ] Separate facts, inference, uncertainty, and informational opinion.
- [ ] Add timeout, retry, and partial-result behavior.

Recommended final answer shape:

```text
Question / scope
Current thesis
What is unusual
Evidence supporting the thesis
Evidence against it / alternative explanations
Upcoming catalysts
Data freshness and gaps
Confidence and what would change the view
```

Acceptance criteria:

- The agent cannot state a sourced number without a matching tool result/evidence ID.
- It can answer multi-source questions without receiving raw corpora.
- Repeated runs produce traceable research plans and tool histories.
- It explicitly reports missing, stale, or contradictory data.
- Citation tests verify that the cited span supports the claim.

### Phase 8 - Machine learning and backtesting

Do not train an investment model until the point-in-time feature store and evaluation harness are trustworthy.

- [ ] Define targets before training: future abnormal return, drawdown, earnings deterioration, financing event, or catalyst success.
- [ ] Use walk-forward/rolling time splits, never random train/test splits.
- [ ] Apply realistic reporting delays through `known_at`.
- [ ] Include delisted securities and historical universe membership.
- [ ] Include transaction costs, liquidity limits, and borrow availability for short tests.
- [ ] Establish simple rule and linear-model baselines.
- [ ] Try interpretable tree/ranking models before deep models.
- [ ] Track feature importance, calibration, turnover, drawdown, and regime stability.
- [ ] Register model version, training window, feature version, and evaluation results.
- [ ] Monitor drift and disable models that fail documented thresholds.

Use embeddings for retrieval and clustering. Do not assume embedding similarity itself predicts returns. The predictive model should train on point-in-time structured features and events; the LLM should explain and challenge its output.

Acceptance criteria:

- No feature contains information published after the prediction timestamp.
- Results beat simple baselines out of sample after costs.
- Performance is not concentrated in a tiny number of securities or one market regime.
- Every prediction is accompanied by its most influential features and data freshness.

### Phase 9 - Biotech and medical catalyst data

Biotech should be an additional domain built on the same raw -> normalized -> event -> feature -> signal pattern.

#### Data sources

- [ ] ClinicalTrials.gov API and version/history tracking.
- [ ] FDA approvals, labels, adverse-event data, recalls, shortages, advisory committees, and accelerated-approval status.
- [ ] Company filings and investor-relations releases.
- [ ] PubMed and peer-reviewed publications.
- [ ] Conference programs/abstracts where licensing and access allow.
- [ ] Patents and exclusivity information where relevant.

ClinicalTrials.gov API documentation: <https://clinicaltrials.gov/data-about-studies/learn-about-api>

openFDA drug API documentation: <https://open.fda.gov/apis/drug/event/>

#### Biotech domain model

- `drug_assets`: normalized asset, molecule, modality, aliases.
- `indications`: disease, line of therapy, biomarker population.
- `trials`: NCT ID, sponsor, phase, design, enrollment, arms, status.
- `trial_endpoints`: primary/secondary endpoint, measurement, time frame.
- `trial_versions`: full history and field-level changes.
- `regulatory_events`: submission, acceptance, advisory committee, approval, rejection, withdrawal.
- `publications`: DOI/PubMed ID, trial links, result attributes.
- `asset_company_relationships`: originator, licensee, partner, economics, effective dates.

#### Biotech features and alerts

- [ ] Enrollment or completion-date changes.
- [ ] Primary endpoint, comparator, arm, or sample-size changes.
- [ ] Trial status changes and unexplained delays.
- [ ] New safety language or discontinuation imbalance.
- [ ] Upcoming readout windows and regulatory catalysts.
- [ ] Cash runway relative to expected catalyst date.
- [ ] Partnership, licensing, milestone, and royalty economics.
- [ ] Trial-quality features: randomization, blinding, control, statistical power, endpoint hierarchy, follow-up.
- [ ] Effect size with confidence interval, not p-value alone.
- [ ] Comparable historical trial and regulatory outcomes.

Important cautions:

- Adverse-event reporting systems can generate hypotheses but generally cannot establish causality or incidence on their own.
- Registry changes may be administrative rather than scientifically meaningful.
- Conference abstracts can be incomplete and later change in peer-reviewed publication.
- Drug aliases and company licensing relationships require careful entity resolution.
- The agent should distinguish clinical significance from statistical significance.

Acceptance criteria:

- Every trial change is a diff between known versions with timestamps.
- A drug can be followed across aliases, sponsors, partners, indications, and trials.
- Catalyst alerts link to the exact registry/FDA/publication evidence.
- Clinical interpretations expose uncertainty and never imply unsupported causality.

## 8. Evaluation strategy

Maintain separate eval suites because a single end-to-end score will hide where failures occur.

### Data correctness

- Fixture response -> expected normalized rows.
- XBRL unit/duration/amendment cases.
- FINRA reporting-cycle and missing-data cases.
- Corporate-action and ticker-history cases.

### Retrieval

- Query -> relevant chunk IDs.
- Measure recall at 5/10, ranking quality, filter accuracy, and duplication.
- Include exact keyword, semantic paraphrase, table, and cross-document questions.

### Extraction

- Document excerpt -> expected typed event.
- Score field accuracy, numeric accuracy, evidence-span accuracy, and abstention.

### Agent behavior

- Correct tool choice.
- Correct `as_of` handling.
- Citation completeness and citation entailment.
- Contradiction handling.
- Missing-data honesty.
- Context/token budget compliance.

### Signal and ML evaluation

- Point-in-time reproducibility.
- Walk-forward performance after costs.
- Calibration and false-positive analysis.
- Sector, size, liquidity, and regime breakdowns.

Keep a small set of historical case studies, but do not tune only to famous outcomes. Each case should freeze the information available before the event and verify whether the system surfaced the relevant public clues without using later knowledge.

## 9. Operational and safety requirements

- [ ] Keep API keys only in environment/secret storage; never commit `.env`.
- [ ] Respect source rate limits, terms, redistribution rights, and data licenses.
- [ ] Encrypt sensitive configuration and authenticated provider traffic.
- [ ] Record provider request IDs and ingestion failures.
- [ ] Add backups for normalized data and raw-source manifests.
- [ ] Add retention and deletion policies for hosted vector stores.
- [ ] Track embedding, extraction, reranking, and generation costs separately.
- [ ] Protect tools with strict schemas and bounded queries.
- [ ] Do not allow generated code or unrestricted SQL to mutate production data.
- [ ] Label research as informational and require human review before any trade execution.

## 10. Recommended immediate sequence

This is the shortest path from the current project to a useful research platform:

1. Finish Phase 0's stable schema, raw archive, provenance, and `as_of` rules.
2. Bring SEC to the Phase 1 acceptance criteria instead of adding more chat tools.
3. Finish normalized FINRA history and deterministic short-pressure features.
4. Add market prices, corporate actions, and a historical security universe.
5. Build Qdrant indexing and retrieval evals on SEC narrative sections.
6. Add `search_documents`, `get_evidence`, `get_metric_series`, and `screen_entities`.
7. Build the first complete signal family: dilution/future funding burden or fundamental deterioration plus short pressure.
8. Connect the research agent only after the screens and retrieval layers work independently.
9. Accumulate point-in-time history and validate rules before training predictive ML.
10. Add biotech as a separate domain once the shared event/evidence pipeline is stable.

The first valuable end-to-end milestone is not “the model can access every dataset.” It is:

> On a historical date, the system can rank a universe for one transparent thesis, show the exact features that caused the ranking, retrieve the best supporting and contradicting evidence, and reproduce the result later without seeing future data.

Once that works, new datasets become connectors and new investment ideas become feature/signal definitions rather than rewrites of the agent.
