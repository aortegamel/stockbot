"""Tool implementations + OpenAI-format JSON schemas for Pi.

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from zoneinfo import ZoneInfo

from . import (
    analyst_client,
    edgar_client,
    exa_client,
    finra_client,
    obligations,
    sec,
    valuation,
)
from .analytics import screens
from .analytics.options import analyze_option, compare_options
from .analytics.portfolio import largest_positions, portfolio_concentration
from .config import broker_enabled, get_data_root, get_robinhood_mcp_url
from .domain.market.entities import EntityRelationship
from .domain.portfolio.models import PortfolioSnapshot, Position
from .policy import Capability, RequestContext
from .robinhood import RobinhoodClient, capabilities
from .robinhood.auth import DEFAULT_TOKEN_PATH, OAuthConfig, has_valid_tokens
from .robinhood.client import RobinhoodAuthRequired
from .robinhood.options import OptionQuote, normalize_option_quote
from .robinhood.portfolio import RobinhoodPortfolioProvider
from .sec.models import SECSearchResult
from .services import risk as risk_service
from .services import sec_facts
from .services.portfolio_research import (
    SEC_CONCEPTS,
    PortfolioResearchPosition,
    enrich_portfolio_research,
)
from .services.portfolio_sync import read_latest_snapshot, sync_robinhood_portfolio

if TYPE_CHECKING:
    from .domain.risk.breaches import RiskBreach
    from .domain.risk.evaluation import EvaluationIssue, RiskEvaluation
    from .sec.models import Filing
    from .thesis.intake import IntakeProposal
    from .thesis.models import (
        JSONValue,
        Thesis,
        ThesisQuestion,
        ThesisStateSnapshot,
        WatchRule,
    )
    from .thesis.repository import ThesisRepository

logger = logging.getLogger(__name__)

# Structured thesis-proposal delta fields shared by thesis_create/refine.
# IntakeProposal.from_dict remains the source of truth for value shapes.
_THESIS_DELTA_PROPERTIES = {
    "scope": {"type": "string", "description": "Ticker scope (e.g. NVDA) or 'unknown'."},
    "claims": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"statement": {"type": "string"}},
            "required": ["statement"],
        },
    },
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "invalidators": {"type": "array", "items": {"type": "string"}},
    "unknowns": {"type": "array", "items": {"type": "string"}},
    "expressions": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "instrument": {"type": "string"},
                "direction": {"type": "string"},
                "structure": {"type": "string"},
                "horizon": {"type": "string"},
            },
            "required": list[str](),
        },
    },
    "questions": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "question_type": {"type": "string"}},
            "required": ["question"],
        },
    },
}

_THESIS_PROPOSAL_KEYS = tuple(_THESIS_DELTA_PROPERTIES)

TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Single reported fundamental for one ticker: a specific numeric fundamental (EPS, "
            "dividends, balance sheet line item, shares outstanding) for a ticker. "
            "Note: shares_outstanding is SEC-reported shares outstanding, "
            "not public float. Call this for any request for a specific "
            "numeric metric. Do NOT use for full financial statements (get_financial_statements), XBRL-tagged facts by concept name (get_xbrl_facts), cheap-vs-expensive multiples (get_valuation_metrics), or forward consensus expectations (get_analyst_estimates). When presenting EPS, show basic and diluted EPS side by side in a markdown table "
            "with period, basic EPS, and diluted EPS columns, including TTM for both when available. Dividends responses include last paid and next "
            "SEC-declared (upcoming) dividends with filing provenance; undeclared estimates are never included. Render past, present, and future-declared dividends under separate headings; anything undeclared is an estimate and must never appear under NEXT DECLARED. Dividend metrics are tool-computed; interpret, never recalculate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "metric": {
                        "type": "string",
                        "enum": [
                            "eps",
                            "dividends",
                            "balance_sheet",
                            "shares_outstanding",
                            "overview",
                        ],
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time query date YYYY-MM-DD; store-backed for eps/shares_outstanding/dividends; live results are labeled data_source=live.",
                    },
                },
                "required": ["ticker", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_sec_entities",
            "description": "Resolves a company name, ticker, or CIK to verified SEC entity candidates (CIK, tickers, verification status), including no-ticker registrants and former names. Ties and fuzzy-only matches stay ambiguous; verify identity before list_sec_filings. Dispatched inside a deep research session, every entity route is searched by default and limit bounds only the returned packet (default 20); pass exhaustive=false for the quick bounded lookup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; former names apply only within their known/valid interval.",
                    },
                    "exhaustive": {
                        "type": "boolean",
                        "description": "true searches every entity route (the default when dispatched inside a research session); false requests the quick bounded lookup (default outside a research session).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Bounded lookups return at most this many candidates (default 20); exhaustive lookups return every candidate found across routes (local source cap 50) and limit only bounds the display packet.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_sec_entities_bounded",
            "description": "Quick bounded entity lookup: same verified SEC entity candidates as find_sec_entities, but only the fast routes, capped at limit (default 20). Pick this for one identity check when full coverage is not needed; pick find_sec_entities when the question needs every candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; former names apply only within their known/valid interval.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max candidates returned (default 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": 'EDGAR discovery over entity, full-text (EFTS), filer-submissions, global filing, and local routes. Dispatched inside a deep research session, every applicable route is drained by default and limit bounds only the returned hit packet (default 20; page the rest with research_read_search); pass exhaustive=false for the quick bounded lookup, which is also the default outside a research session. Hits are text mentions: each names the filer (filer_name/filer_cik) and the exact matched document, never inferred subject identity. Returns coverage, attempts, counts, PIT basis, warnings/errors, auto-queued backfill jobs, and bounded evidence IDs. Required: at least one of query, ticker, cik, company_name, person_name, domain, accession_no, security_identifier (a call with none is rejected). Optional: forms, start_date, end_date, as_of, exhaustive, limit. Example: {"query": "risk factors", "ticker": "AAPL", "forms": ["10-K"], "limit": 10}.',
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ticker": {"type": "string"},
                    "cik": {"type": "string"},
                    "company_name": {"type": "string"},
                    "person_name": {"type": "string"},
                    "domain": {"type": "string"},
                    "security_identifier": {
                        "type": "string",
                        "description": "Ticker, CUSIP, ISIN, or class title; never treated as issuer identity.",
                    },
                    "accession_no": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    },
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD. Combined with end_date as a range.",
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; records known after it are excluded.",
                    },
                    "exhaustive": {
                        "type": "boolean",
                        "description": "true drains every applicable route (the default when dispatched inside a research session); false requests the quick bounded lookup (default outside a research session).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max hits in the returned packet (default 20); under exhaustive retrieval it does not reduce retrieval.",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings_bounded",
            "description": "Quick bounded EDGAR lookup: same filing-text search as search_sec_filings, but fast routes only, capped at limit (default 20). Pick this for one mention check; pick search_sec_filings when the question needs all/every mention or full coverage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ticker": {"type": "string"},
                    "cik": {"type": "string"},
                    "company_name": {"type": "string"},
                    "person_name": {"type": "string"},
                    "domain": {"type": "string"},
                    "security_identifier": {
                        "type": "string",
                        "description": "Ticker, CUSIP, ISIN, or class title; never treated as issuer identity.",
                    },
                    "accession_no": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    },
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD. Combined with end_date as a range.",
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; records known after it are excluded.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max hits returned (default 20).",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sec_filings",
            "description": 'Lists SEC EDGAR filings for an exact ticker or CIK. Required: identifier (ticker or CIK, e.g. identifier="AAPL"); missing identifier is rejected; does NOT search company names; resolve names via find_sec_entities first. Optional: forms, start_date, end_date, as_of, limit. Example: {"identifier": "AAPL", "forms": ["10-K"], "limit": 10}.',
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Ticker or CIK, e.g. AAPL.",
                    },
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD. Combined with end_date as a range.",
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; filings known after it are excluded.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sec_relationships",
            "description": "Ownership and transaction relationships for an entity (CIK, ticker, or entity id), both directions: 13D/G beneficial owners, 13F manager holdings (either direction), insider issuer/owner links, transaction filer/target/acquirer, offering filer/registrant, plus verified workflow links and observed mentions. Mentions never flatten into verified links; transaction status stays unknown without closing evidence. Pass an entity, or a company_name (e.g. Apple); the server maps the name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "CIK, ticker, entity id, or candidate dict. If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when entity is unknown; the server maps it to a ticker.",
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional open-vocabulary type filter (e.g. beneficial_owner, holding_manager).",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                    "exhaustive": {
                        "type": "boolean",
                        "description": (
                            "Exhaust all applicable relationship indexes and SEC routes; "
                            "the returned model context remains bounded."
                        ),
                        "default": True,
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_search_coverage",
            "description": "Reads persisted SEC ingestion coverage and backfill jobs only (never infers completeness from rows). Use to check whether a form/source/date partition is covered or still queued/running/failed, or to revisit one search's ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "form": {"type": "string"},
                    "search_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filing",
            "description": "Returns one filing's record (filer, subject when known, form, filed/accepted/known dates, period, primary document, amendment link, source URL) by accession number. When the accession number is unknown, find it with list_sec_filings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accession_no": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; filings known after it are excluded.",
                    },
                },
                "required": ["accession_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sec_documents",
            "description": "Lists the documents and exhibits attached to one filing by accession number. When the accession number is unknown, find it with list_sec_filings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accession_no": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; filings known after it are excluded.",
                    },
                },
                "required": ["accession_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_document",
            "description": 'Returns a bounded window of one filing document\'s text (default: primary document) by accession number. Required: accession_no. Optional: document_name (exact file), section (within-document heading filter), query (within-document substring filter), cursor (alias for offset, default 0; overrides offset when both are given), limit (alias for max_chars, 1..32000, default 12000; overrides max_chars when both are given), offset/max_chars (legacy aliases), as_of, raw (default false; true returns raw source text, still bounded). Returns text plus metadata, section, source_refs [{accession, document, offset}], cursor/next_cursor for pagination, and a canonical source_handle for the returned window (pass it to research_add_evidence: the kernel reloads that window from the archive and materializes the cited passage itself). Example: {"accession_no": "0000320193-25-000079", "section": "Risk Factors", "cursor": 0, "limit": 12000}; next page with {"accession_no": "...", "cursor": <next_cursor>}. Load only the document relevant to the question, never full history. When the accession is already known, pass it; use get_material_events only to discover what changed.',
            "parameters": {
                "type": "object",
                "properties": {
                    "accession_no": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named accession_no, not accession_number.",
                    },
                    "document_name": {
                        "type": "string",
                        "description": "Exact filing file name; omit for the primary document.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Within-document heading filter, e.g. Risk Factors.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Within-document substring filter.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; filings known after it are excluded.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset into the document text (default 0). Legacy alias; cursor overrides it when both are given.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Characters to return, 1..32000 (default 12000). Legacy alias; limit overrides it when both are given.",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Character offset into the document text (default 0). Overrides offset when both are given.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Characters to return, 1..32000 (default 12000). Overrides max_chars when both are given.",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw source text, still bounded (default false).",
                    },
                },
                "required": ["accession_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_sec_filings",
            "description": "Self-contained full-filing diff for one ticker: pass ticker (plus optional forms hint like S-3/A) and the two most recent matching filings resolve internally; or pass two accession numbers directly. Returns added/removed language. Do NOT call list_sec_filings first. Do NOT use for risk-factor-only year-over-year diffs for one ticker (diff_risk_factors).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "forms": {"type": "array", "items": {"type": "string"}},
                    "current_accession": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named current_accession/previous_accession for diffs.",
                    },
                    "previous_accession": {
                        "type": "string",
                        "pattern": "^\\d{10}-?\\d{2}-?\\d{6}$",
                        "description": "SEC accession number, e.g. 0000320193-25-000079. Named current_accession/previous_accession for diffs.",
                    },
                    "section": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD; filings known after it are excluded.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_events",
            "description": "What changed since a date: deterministic 8-K-derived events with accession citations. Call for 'what changed/what's new' questions. Pass a ticker (e.g. AAPL) or a company_name (e.g. Apple); the server maps the name. Never call with neither.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.",
                    },
                    "since": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD; events known on or after this date.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                },
                "required": ["ticker", "since"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_beneficial_ownership",
            "description": "5%+ beneficial-ownership records (SC 13D/G): holder, shares, percent, voting/dispositive powers. Deterministic numbers, never web prose. Use for current 5%+ stakes; use get_ownership_changes for stake changes. Pass a ticker (e.g. AAPL) or a company_name (e.g. Apple); the server maps the name. Never call with neither.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ownership_changes",
            "description": "Deterministic diffs between a holder's consecutive 13D/G filings: share and percent changes plus voting/text changes. Use for stake changes; use get_beneficial_ownership for current 5%+ stakes. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insider_activity",
            "description": "Executed insider buys/sells for one ticker: insider transactions (Forms 3/4/5) with SEC transaction codes mapped to purchase/sale/exercise/grant/gift/conversion/withholding/other. Disposals are never defaulted to bearish selling. Use for actual insider purchases and sales by executives and directors. Do NOT use for planned but unexecuted Form 144 sales (get_planned_insider_sales). Pass a ticker (e.g. AAPL) or a company_name (e.g. Apple); the server maps the name. Never call with neither.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_planned_insider_sales",
            "description": "Planned Form 144 sale notices not yet executed for one ticker: proposed insider sales. Use for proposed or planned insider sales. Do NOT use for completed insider trades (get_insider_activity); compare with get_insider_activity for follow-through. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_offering_history",
            "description": "Financing history (S-1/S-3/424B/EFFECT): offering terms with source-registration links. Unknown terms stay unknown, never estimated. Pair with get_dilution_profile for financing/dilution work. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dilution_profile",
            "description": "Deterministic dilution math: inputs, formula, and source accessions always shown. Unquantifiable terms return not_quantifiable. Pair with get_offering_history for financing/dilution work. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_governance_events",
            "description": "Proxy/governance filing context (DEF 14A, contested forms, information statements) with retrieval pointers. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "since": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD; events known on or after this date.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_status",
            "description": "M&A filing context (tender offers, 14D-9, S-4, merger proxies). Deal status is unknown until structured parsers land; use get_sec_document for filing text. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Point-in-time date YYYY-MM-DD.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_short_pressure_profile",
            "description": "Short pressure vs shares outstanding for one ticker: FINRA positioning plus SEC shares outstanding and their ratio. Do NOT use for one ticker's biweekly short position alone (get_short_interest) or daily short-sale volume (get_reg_sho_volume). Describes positioning only; never assesses manipulation or causation. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": "Search for relevant Stockbot tools when the active tools cannot perform the task. Returns compact routing cards with ambiguity groups; browse_tools remains the hierarchical catalog path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Non-empty capability description (not a company or ticker); never call with an empty query.",
                    },
                    "domain": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tool_domains",
            "description": "List the tool-domain catalog: domain names with one-line descriptions.",
            "parameters": {"type": "object", "properties": {}, "required": list[str]()},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_tool",
            "description": "Show full metadata for one named Stockbot tool: domain, family, intent, output kind, source, entity scope, time mode, choose-when and reject-when notes, conflicts, related tools, prerequisites, required and optional arguments. Call with the exact tool name, or describe several tools in one call with names. Use for 'what can X do' questions; do not run the subject tool instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Describe several tools in one call, in order.",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_tools",
            "description": "Browse the hierarchical Stockbot tool catalog: root domains, one domain's families, one family's tools with contrast, or one tool's full metadata with canonical parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "family": {"type": "string"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": "Call one catalog tool by its exact canonical name with arguments; arguments defaults to {} (required for zero-argument tools the model may omit); validates against the canonical schema and executes through the standard gateway path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_ownership_filings",
            "description": "Lists the most recent SC 13D/13G filings market-wide "
            "(SEC current-filings feed, ~24h window): issuer, filer, stake "
            "percent/shares, filed date, accession. Call for 'most recent' "
            "or 'latest' big-investor filings when no ticker is given. Do not drill into get_beneficial_ownership unless asked. It lists filings, it "
            "does not establish that a filing caused a price move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "form_type": {
                        "type": "string",
                        "enum": ["SC 13D", "SC 13G", "both"],
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_risk_factors",
            "description": "Self-contained Risk Factors year-over-year diff for one ticker: what changed in Risk Factors language "
            "vs. the prior filing. Takes a ticker alone; filings resolve internally so do NOT call list_sec_filings first. Call for risk-disclosure change framing (what is new/changed). Do NOT use for full-filing diffs between two accessions (diff_sec_filings). "
            "Do not use for disclosure or mention questions without change framing; use search_sec_filings instead.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_statements",
            "description": "Full parsed statements for one ticker: income statement, "
            "balance sheet, and cash flow from 10-K or 10-Q filings. Do NOT use for a single metric like EPS (get_fundamentals) or a single XBRL-tagged fact (get_xbrl_facts). Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "statement_type": {
                        "type": "string",
                        "enum": ["income_statement", "balance_sheet", "cash_flow"],
                    },
                },
                "required": ["ticker", "statement_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_xbrl_facts",
            "description": "Single XBRL-tagged fact by concept name for one ticker: XBRL financial metrics (Revenue, Net Income, "
            'Cash, Debt, Equity, etc.) for any company. Do not use this for EPS; for EPS use get_fundamentals(metric="eps"). Use exact XBRL tag names (e.g. NetIncomeLoss for net income), never friendly labels. Do NOT use for full statements (get_financial_statements). Takes a ticker and concept.',
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "concept": {"type": "string"},
                },
                "required": ["ticker", "concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_short_interest",
            "description": "Biweekly short position for one ticker: FINRA consolidated short interest "
            "(current/previous short position, days to cover, average daily "
            "volume, percent change). Call for short interest, short float, "
            "or days-to-cover questions. Do NOT use for daily short-sale volume by venue (get_reg_sho_volume), market-wide most-shorted screens (get_short_interest_leaderboard), or short-vs-shares-outstanding context (get_short_pressure_profile). For change-over-time or trend "
            "questions, prefer query_finra. When the user asks to show "
            "figures or values, or names exact fields, prefer "
            "get_finra_datapoints. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "settlementDate": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional settlement date YYYY-MM-DD. Omit to return recent cycles.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_short_interest_leaderboard",
            "description": "Market-wide most-shorted screen: FINRA short-interest leaderboard, short interest as a percentage of SEC-reported shares outstanding for tickers that map 1:1 to an SEC CIK whose security is classified as common equity and that has a shares-outstanding fact knowable on or before the as-of date (default: today). Excludes symbols that cannot be mapped to a single SEC entity, are not classified as common equity (funds, ETFs, preferred issues), lack a usable shares-outstanding fact, or have invalid short-interest quantities; every exclusion is counted and returned in coverage. Use for questions such as 'which stock has the highest short interest', 'most shorted stock', or 'short interest as a percent of total shares'. Do NOT use for one ticker's short interest (get_short_interest). This is a deterministic, complete FINRA settlement-date screen; it is NOT percent of public float and is not real-time short interest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of ranked stocks to return; default 10, maximum 25.",
                    },
                    "settlement_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional FINRA settlement date (YYYY-MM-DD). Omit for the latest published FINRA cycle.",
                    },
                    "as_of": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional knowledge horizon (YYYY-MM-DD). Only data knowable on or before this date is used. Defaults to today; pass an explicit date for a historical screen.",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reg_sho_volume",
            "description": "Self-contained daily short-sale volume by venue for one ticker: FINRA daily Reg SHO short-sale volume "
            "ticker (short, short-exempt, and total share quantity by "
            "reporting facility). Omit tradeDate for the Monday-now NYC week-to-date range plus short-volume ratio; pass one tradeDate for that single day only. Rolling 12 months. Pass a ticker (e.g. AAPL) or a company_name (e.g. Apple); dataset and fields resolve internally so do NOT call describe_finra_dataset or get_finra_datapoints. Do NOT use for biweekly short interest positions (get_short_interest).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.",
                    },
                    "tradeDate": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional trade date YYYY-MM-DD.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_threshold_securities",
            "description": "Returns FINRA OTC Regulation SHO / Rule 4320 "
            "threshold securities. Optionally filter by ticker and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "tradeDate": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional trade date YYYY-MM-DD.",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_estimates",
            "description": "Forward sell-side consensus expectations for one ticker: sell-side consensus estimates "
            "from Yahoo Finance: latest quote, analyst 12-month price "
            "targets (mean/median/high/low) and recommendation rating, "
            "forward EPS and revenue estimates per period (current quarter, "
            "next quarter, current fiscal year, next fiscal year) with "
            "growth rates, plus EPS estimate-revision trend (7/30/60 days "
            "ago). Call for analyst estimates, price targets, consensus "
            "expectations, forward growth, or valuation-vs-consensus "
            "questions. Do NOT use for reported historical EPS (get_fundamentals). Consensus moves daily; the response includes the "
            "as-of timestamp. Always state the as-of date.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sp500_weight",
            "description": "Returns a company's current weight in the S&P 500 "
            "index (rank, weight as percent of index market cap) from the "
            "Slickcharts constituent list. Call for 'what percent of the "
            "S&P 500 is [ticker]' or index-weight questions. To estimate "
            "total S&P 500 market cap, divide market_cap from "
            "get_analyst_estimates by weight_pct/100.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_obligations",
            "description": "Future cash obligations from 10-K/10-Q notes for one ticker: quantified contractual obligations and "
            "commitments disclosed in the latest 10-Q/10-K notes: "
            "manufacturing/supply/capacity commitments, cloud service "
            "agreements, vendor commitments, operating leases, and "
            "facility lease guarantees, each with the amount, the "
            "filing's own certainty language (contractual = "
            "non-cancelable/firm; contingent = cancellable, reducible, "
            "terminable, or default-triggered), payment horizon, and "
            "source excerpt. Call for purchase obligations, supply "
            "commitments, cloud commitments, lease obligations, "
            "guarantees, or any 'what is the company obligated to pay "
            "in the future' question. Do NOT use for cheap-vs-expensive multiples (get_valuation_metrics). Contingent items are NOT counted "
            "in adjusted EPS. Treat on-balance-sheet (already accrued) items as informational and never double-count them; never present contingent or off-balance-sheet obligations as certain. Pass a ticker (e.g. AAPL) or a company_name (e.g. Apple); the server maps the name to a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker (e.g. AAPL). If unknown, pass company_name instead; never call with neither.",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name (e.g. Apple) when the ticker is unknown; the server maps it to a ticker.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_valuation_metrics",
            "description": "Cheap-vs-expensive earnings multiples at live price for one ticker: valuation metrics anchored to the live "
            "price as of the query: trailing P/E (SEC GAAP TTM EPS), "
            "consensus forward P/E (Yahoo), plus three clearly separated "
            "EPS figures: consensus forward EPS; adjusted forward EPS "
            "(consensus minus only contractual obligations — "
            "non-cancelable/firm per the 10-Q/10-K notes — annualized "
            "per share); and a stress-scenario forward EPS (also "
            "subtracting contingent obligations: cancellable, reducible, "
            "terminable, or default-triggered). The per-share obligation "
            "drag is shown explicitly. Use for 'is the stock cheap', "
            "P/E, forward earnings, or obligation-adjusted valuation "
            "questions. Do NOT use for reported EPS alone (get_fundamentals) or forward consensus alone (get_analyst_estimates). Never present the stress scenario as 'adjusted'. Always state which ledger tier you are citing plus the live price and its timestamp. Takes a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_finra_datasets",
            "description": "Lists public FINRA Query API datasets (filing cabinet "
            "catalog): concise entries with canonical id group/name, group, "
            "description, and ticker/date support. Optional group or search "
            "filters. Prefer calling the analysis tool directly with a known dataset ID; use this listing only to resolve an ID search did not surface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {
                        "type": "string",
                        "description": "Optional dataset group filter (e.g. otcMarket, fixedIncomeMarket, finra).",
                    },
                    "search": {
                        "type": "string",
                        "description": "Optional substring match on name/description.",
                    },
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_finra_dataset",
            "description": "Describes one FINRA dataset: fields with types and "
            "descriptions, ticker/date fields, documented filter values, and "
            "supported methods. Takes the dataset ID directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Canonical group/name "
                        "(e.g. otcMarket/regShoDaily); unambiguous bare names resolve, unknown/ambiguous ones are rejected.",
                    }
                },
                "required": ["dataset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_finra_datapoints",
            "description": "Exact raw rows from any named FINRA dataset: returns exact source values "
            "for explicit data requests ONLY (e.g. 'show the last five "
            "settlement-date values' or 'show recent position figures'). "
            "Requires a 'fields' list and at least one narrowing "
            "condition (ticker, date/date range, or filters). IMPORTANT: "
            "when the user requests named datapoints with friendly "
            "labels (e.g. 'days to cover', 'average daily volume'), "
            "call describe_finra_dataset FIRST and use the "
            "metadata's exact field names (e.g. daysToCoverQuantity, "
            "averageDailyVolumeQuantity) in the fields list — never "
            "friendly labels. For 'latest five' / 'last five' / 'most "
            'recent\' requests, add sort_fields ["-<dateField>"] or '
            'sort_order "desc" (or "asc" for oldest first); the '
            "client resolves the sort against the dataset's partitions "
            "automatically. Do NOT use for ordinary analysis — query_finra "
            "and the specific helper tools return analyzed briefings "
            "instead. Common short-interest fields: settlementDate, symbolCode, "
            "currentShortPositionQuantity, previousShortPositionQuantity, "
            "averageDailyVolumeQuantity, daysToCoverQuantity. For other datasets, "
            "call describe_finra_dataset first for exact field names — ticker "
            "plus dataset is enough to begin. Returns at "
            "most 25 rows containing only the requested fields. Exact "
            "source values are guaranteed for normal scalar data; "
            "oversized text fields are rendered as a marked excerpt "
            "(table cells are capped at 200 characters to keep the tool "
            "message compact). Takes the dataset ID directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Canonical id group/name "
                        "(e.g. otcMarket/regShoDaily); unambiguous bare names resolve, unknown/ambiguous ones are rejected.",
                    },
                    "fields": {
                        "type": "array",
                        "description": "Exact field names to return (e.g. settlementDate, symbolCode, currentShortPositionQuantity for short interest).",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Issue symbol when the dataset is symbol-level.",
                    },
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD. Combined with end_date as a range.",
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD.",
                    },
                    "filters": {
                        "type": "array",
                        "description": "Extra compare filters (field names must exist on the dataset — when unknown, call describe_finra_dataset first).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": [
                                        "EQUAL",
                                        "GREATER",
                                        "LESSER",
                                        "GTE",
                                        "LTE",
                                        "NOT_EQUAL",
                                        "BEGINS_WITH",
                                    ],
                                },
                                "value": {"type": "string"},
                            },
                            "required": ["field", "value"],
                        },
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (clamped to 1..25; default 10).",
                    },
                    "sort_fields": {
                        "type": "array",
                        "description": "FINRA sortFields syntax: '+field' "
                        "ascending, '-field' descending, e.g. "
                        '["-settlementDate"] returns newest first. Use for '
                        "'latest five' / 'last five' / 'most recent' data "
                        "requests. Fields must exist on the dataset.",
                        "items": {"type": "string"},
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Convenience: sort by the dataset's "
                        "date field ('desc' = newest first, for 'latest "
                        "five' requests). Rejected when the dataset has no "
                        "date field — use sort_fields instead.",
                    },
                },
                "required": ["dataset", "fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_finra",
            "description": "Analyzed FINRA briefing with trends and metrics over any named dataset: queries a FINRA dataset by canonical group/name "
            "(or legacy bare name) and returns an analyzed briefing: "
            "query provenance, coverage dates, deterministic metrics "
            "(min/max/mean/median/sum, latest-vs-prior change), derived "
            "trends, data-quality warnings, and a concise prose briefing. "
            "Raw source records are NOT returned. Prefer get_short_interest "
            "/ get_reg_sho_volume / get_threshold_securities for those "
            "specific questions. Takes the dataset ID directly with a bounded limit. "
            "Use get_finra_datapoints only when the user explicitly asks "
            "to see exact source values. For more records, paginate with offset using the returned next_offset/may_have_more indicators. If a result is flagged stale or historical (newest date older than 90 days), say so explicitly and never present it as current market data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Canonical id group/name "
                        "(e.g. otcMarket/regShoDaily); unambiguous bare names resolve, unknown/ambiguous ones are rejected.",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Issue symbol when the dataset is symbol-level.",
                    },
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD. Combined with end_date as a range.",
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "YYYY-MM-DD.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max records to return (clamped to 1..1000).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "0-based record offset for pagination "
                        "(FINRA max 500000). Rejected for datasets whose "
                        "catalog entry has supportsRecordOffset=false.",
                    },
                    "filters": {
                        "type": "array",
                        "description": "Extra compare filters (field names must exist on the dataset — when unknown, call describe_finra_dataset first).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": [
                                        "EQUAL",
                                        "GREATER",
                                        "LESSER",
                                        "GTE",
                                        "LTE",
                                        "NOT_EQUAL",
                                        "BEGINS_WITH",
                                    ],
                                },
                                "value": {"type": "string"},
                            },
                            "required": ["field", "value"],
                        },
                    },
                    "analysis_goal": {
                        "type": "string",
                        "description": "Optional: what the user needs answered "
                        "(e.g. 'trend over the last 12 months'). Guides the "
                        "briefing; deterministic metrics are always computed.",
                    },
                },
                "required": ["dataset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_snapshot",
            "description": "Returns a read-only Robinhood MCP stock quote with last, bid, ask, and retrieval time.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_option_chain",
            "description": "Returns a bounded read-only Robinhood option chain filtered by type, DTE, and strike.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "option_type": {"type": "string", "enum": ["put", "call"]},
                    "min_dte": {"type": "integer", "minimum": 0},
                    "max_dte": {"type": "integer", "minimum": 0},
                    "strike_min": {"type": "number"},
                    "strike_max": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
                "required": ["ticker", "option_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_option_contract",
            "description": "Analyzes one Robinhood option contract using observed quote fields and deterministic expiration payoff math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "expiration": {"type": "string", "description": "YYYY-MM-DD"},
                    "strike": {"type": "number"},
                    "option_type": {"type": "string", "enum": ["put", "call"]},
                    "target_price": {"type": "number"},
                },
                "required": ["ticker", "expiration", "strike", "option_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_options",
            "description": "Compares bounded Robinhood option contracts at a target expiration price, including spreads, liquidity, Greeks, and deterministic payoff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "option_type": {"type": "string", "enum": ["put", "call"]},
                    "target_price": {"type": "number"},
                    "min_dte": {"type": "integer", "minimum": 0},
                    "max_dte": {"type": "integer", "minimum": 0},
                    "strike_min": {"type": "number"},
                    "strike_max": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
                "required": ["ticker", "option_type", "target_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_snapshot",
            "description": "Returns the user's current Robinhood portfolio with deterministic valuation, weights, cash, concentration, and available SEC/FINRA research context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "refresh": {
                        "type": "boolean",
                        "description": "If true, refresh account and quote data from Robinhood before returning the snapshot.",
                    }
                },
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scanner_filter_specs",
            "description": "Lists every valid Robinhood scanner filter type and usage (read-only catalog).",
            "parameters": {
                "type": "object",
                "properties": dict[str, object](),
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_mandate",
            "description": "Deterministic risk/mandate evaluation of the latest portfolio snapshot against data/mandate.json: sector exposure, single-position weight, minimum cash, prohibited assets. Breaches are computed by Stockbot; explain them, do not recalculate.",
            "parameters": {
                "type": "object",
                "properties": dict[str, object](),
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scans",
            "description": "Lists the user's saved Robinhood scanners (screeners): id, title, active filters, configured columns, sort order, and whether the scan is Cortex-managed (read-only).",
            "parameters": {
                "type": "object",
                "properties": dict[str, object](),
                "required": list[str](),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_scan",
            "description": "Executes a saved Robinhood scanner and returns live, real-time market results (bounded to limit rows). Requires a scan_id from get_scans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {
                        "type": "string",
                        "description": "The scan identifier to execute.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Maximum result rows to return (default 20).",
                    },
                },
                "required": ["scan_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Current qualitative evidence from the web (news, announcements, competitive/industry developments, management commentary, publications, specialist commentary, counterevidence) with bounded highlights. NOT a source for exact financial facts, portfolio state, historical point-in-time facts, mandate calculations, or deterministic screens — use the canonical SEC/FINRA/Robinhood/local-warehouse tools for those. Distinguish published_at from retrieved_at and never claim historical completeness. Never use for market-wide screening; deterministic screens generate candidates first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query: ticker/company/industry plus the research question. Never include account, portfolio, or personal identifiers.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["news", "company", "publication", "financial report"],
                        "description": "Optional category to narrow the search.",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional domains to restrict results to.",
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional domains to exclude from results.",
                    },
                    "start_published_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional start publication date YYYY-MM-DD, inclusive.",
                    },
                    "end_published_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Optional end publication date YYYY-MM-DD, inclusive.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["auto", "fast", "deep-lite"],
                        "description": "Optional search mode (default auto).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Maximum results, 1-25 (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_create",
            "description": "Creates a thesis from a structured proposal. Returns thesis_id, scope, initial supported watch rules, or setup-needed state with missing questions when no target resolves. Never invents thresholds. Takes a bare investment view with no existing thesis ID (thesis_refine needs an existing ID).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_thesis": {
                        "type": "string",
                        "description": "The user's investment thesis in their own words.",
                    },
                    **_THESIS_DELTA_PROPERTIES,
                },
                "required": ["user_thesis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_show",
            "description": "Reads one thesis with its assessment, watch rules, and open questions. Nonmutating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Thesis ID or slug."},
                    "as_of": {
                        "type": "string",
                        "description": "Point-in-time cutoff (ISO-8601); omit for current state.",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_refine",
            "description": "Refines a thesis with a clarification plus optional structured deltas. Adds claims/expressions and supported watch rules; never overwrites user-disabled rules. Refuses paused/closed theses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Thesis ID or slug."},
                    "clarification": {
                        "type": "string",
                        "description": "New information or correction in the user's own words.",
                    },
                    **_THESIS_DELTA_PROPERTIES,
                },
                "required": ["id", "clarification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_watch",
            "description": "Lists a thesis's watch rules, or appends one validated supported rule (IDs and domain input only). Never modifies existing rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Thesis ID or slug."},
                    "rule_type": {
                        "type": "string",
                        "description": "Semantic monitor name to add (omit to only list rules).",
                    },
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                    "expression_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_journal",
            "description": "Appends one operator note to a thesis journal (active theses only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Thesis ID or slug."},
                    "title": {"type": "string"},
                    "body": {"type": "string", "description": "Note body (Markdown)."},
                    "trigger_id": {
                        "type": "string",
                        "description": "Trigger this entry completes (omit for ordinary notes).",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Live run this entry completes (trigger-linked only).",
                    },
                    "known_at": {
                        "type": "string",
                        "description": "PIT cutoff this entry is known at (ISO-8601).",
                    },
                },
                "required": ["id", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thesis_status",
            "description": "Pauses, resumes, or closes thesis monitoring. Pause suspends monitor ticks without deleting rules; resume re-activates; close ends monitoring permanently. Returns thesis_id, slug, and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Thesis ID or slug."},
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "close"],
                        "description": "Status change to apply.",
                    },
                },
                "required": ["id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_start",
            "description": "Starts a research session for a question and returns session_id with its first job, status, and pending next action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Research question."},
                    "objective": {
                        "type": "string",
                        "description": "Objective (defaults to the question).",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "Point-in-time cutoff (ISO-8601); omit for current state.",
                    },
                    "policy": {
                        "type": "object",
                        "description": "Optional session policy ({research_sources: {mode, sources}}); omit for SEC-only default.",
                    },
                    "research_sources": {
                        "type": "object",
                        "description": "Optional source allowlist ({mode: allowlist, sources}); omit for SEC-only default.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional source allowlist shorthand (e.g. [SEC, FINRA, WEB]); omit for SEC-only default.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_resume",
            "description": "Resumes a research session: read-only snapshot with session, wave, budgets, open jobs, and pending next action. Nonmutating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_status",
            "description": "Reads a research session with its jobs and pending next action. Nonmutating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_cancel",
            "description": "Cancels a research session; cancelling a terminal session returns its current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_read",
            "description": "Reads one persisted research resource (evidence, freeze, dossier, coverage artifact, job, or session) by ID. Coverage artifacts are a search's scope records (absence observations): inspection only, never citable as raw evidence. Nonmutating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Research session ID."},
                    "kind": {
                        "type": "string",
                        "enum": ["evidence", "freeze", "dossier", "coverage", "job", "research"],
                        "description": "Resource store to read.",
                    },
                    "resource_id": {
                        "type": "string",
                        "description": "Evidence, freeze, dossier, coverage-artifact, job, or session ID.",
                    },
                    "freeze_id": {
                        "type": "string",
                        "description": "Optional freeze scope: an evidence read must be a member of that freeze's evidence set.",
                    },
                },
                "required": ["session_id", "kind", "resource_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_read_search",
            "description": "Pages the persisted SEC search universe of one search_id: every ranked hit that search retrieved, with retrieval truth (pagination_complete/source_exhausted) and coverage. display_limit only bounded the earlier model packet, never the stored set, so this is how the full remainder is read. Nonmutating; hits are navigation artifacts — open the filing and cite a raw passage before recording evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                    "search_id": {
                        "type": "string",
                        "description": "Search id returned by search_sec_filings.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Hits to skip, best score first (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Hits per page (default 50).",
                    },
                    "forms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Optional form filter, e.g. ["10-K", "8-K"].',
                    },
                },
                "required": ["session_id", "search_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_add_evidence",
            "description": "Records one typed finding on a running research job. observed_fact requires the canonical source_handle returned by get_sec_document plus the passage/matching_passage/section you are citing: the kernel reloads that document window from the SEC archive itself, slices the passage out of it, and stores those bytes, so a hallucinated passage fails ERR_PASSAGE_NOT_IN_SOURCE and a changed document fails ERR_SEC_HANDLE_STALE (a hit or a handle-less citation fails ERR_RAW_SOURCE_REQUIRED; SEC search results are navigation artifacts). absence_observation requires search_id + query + coverage of what was searched and is recorded as a session coverage artifact (search scope, never citable evidence), not as an evidence record. Provenance, point-in-time, and IDs are kernel-validated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Running job ID the finding belongs to.",
                    },
                    "item": {
                        "type": "object",
                        "description": "Finding whose provenance must match its claim_kind and the owning job's domain: SEC jobs cite the get_sec_document source_handle + cited passage, or the persisted tool_result_id of an SEC structured response they read plus the cited record values, for observed_fact; FINRA/WEB jobs cite the persisted tool_result_id of the FINRA/search_web response they read plus the cited record values/highlight (the kernel replays the persisted result itself); search scope for absence_observation.",
                        "properties": {
                            "claim_kind": {
                                "type": "string",
                                "enum": ["observed_fact", "absence_observation"],
                                "description": "Defaults to observed_fact (evidence). absence_observation records search-scope coverage instead of evidence; evidence_type/ev_type are legacy aliases (filing_observation/search_coverage).",
                            },
                            "claim_text": {
                                "type": "string",
                                "description": "The finding, stated over its provenance.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Recorded content (defaults to claim_text).",
                            },
                            "source_record_id": {
                                "type": "string",
                                "description": "SEC accession of the opened filing (also accession_no/accession; 18 bare digits normalize).",
                            },
                            "document_name": {
                                "type": "string",
                                "description": "Document the passage came from (also document).",
                            },
                            "passage": {
                                "type": "string",
                                "description": "Passage/section you are citing from that document window (also matching_passage/section/fact); the kernel stores its own copy of the matched span.",
                            },
                            "source_handle": {
                                "type": "object",
                                "description": "The source_handle get_sec_document returned for the window you read (accession_no, document_name, basis, offset, max_chars, text_hash, source_content_hash). The kernel reloads it and materializes the passage itself.",
                            },
                            "tool_result_id": {
                                "type": "string",
                                "description": "FINRA/WEB jobs only: the persisted tool_result_id the kernel stored for the FINRA/search_web response you read (returned in that response). The kernel replays it; a copied number without it fails closed.",
                            },
                            "url": {
                                "type": "string",
                                "description": "WEB jobs only: the result URL of the persisted search_web row you are citing (also source_record_id/source_uri).",
                            },
                            "excerpt": {
                                "type": "string",
                                "description": "WEB jobs only: the highlight text of the persisted search_web row you are citing (also matching_passage/passage).",
                            },
                            "record_identity": {
                                "type": "string",
                                "description": "FINRA jobs only: the record values of the persisted FINRA row you are citing (also matching_passage/passage).",
                            },
                            "source_uri": {
                                "type": "string",
                                "description": "URL of the opened document, when known.",
                            },
                            "known_at": {
                                "type": "string",
                                "description": "ISO-8601 public-knowledge timestamp (e.g. filing date); never a retrieval time.",
                            },
                            "search_id": {
                                "type": "string",
                                "description": "Absence observations only: search-run id the scope covers.",
                            },
                            "query": {
                                "type": "string",
                                "description": "Absence observations only: the exact query that was run.",
                            },
                            "coverage": {
                                "type": "object",
                                "description": "Absence observations only: what was searched and whether paging was exhausted.",
                                "properties": {
                                    "forms": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Forms searched (also dates/partitions/entities/docs/gaps).",
                                    },
                                    "pagination_complete": {
                                        "type": "boolean",
                                        "description": "True only when every page of the search was retrieved.",
                                    },
                                    "complete": {
                                        "type": "boolean",
                                        "description": "True only when the searched scope is the whole intended scope.",
                                    },
                                },
                            },
                        },
                    },
                },
                "required": ["session_id", "job_id", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_submit_source_result",
            "description": "Complete one running source job with validated coverage; evidence stays mutation-only. coverage requires useful_for_question. SEC sufficient additionally requires major_entities_investigated, relationship_types_checked, forms_examined, exhibits_examined, material_open_questions, search_runs (persisted search ids), and covered_branches; FINRA sufficient requires datasets_queried, tickers_covered, settlement_windows_covered, dataset_reads, and covered_branches; WEB sufficient requires semantic_branches_covered, queries_executed, results_inspected, and covered_branches; every domain must leave material_open_questions, major_entities_missing, remaining_branches, routes_unsearched, and the unresolved_questions argument empty (else ERR_COVERAGE_REQUIRED / ERR_COVERAGE_INCOMPLETE). insufficient keeps evidence optional and residuals allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Running source job ID to complete.",
                    },
                    "coverage": {
                        "type": "object",
                        "description": "Coverage envelope: what was searched and how far it got.",
                        "properties": {
                            "useful_for_question": {
                                "type": "string",
                                "enum": ["sufficient", "insufficient"],
                                "description": "Required. sufficient = SCC scope drained with no material open branch; insufficient = honest residual coverage.",
                            },
                            "major_entities_investigated": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty — entities actually investigated in filings.",
                            },
                            "relationship_types_checked": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty — relationship channels checked.",
                            },
                            "forms_examined": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty — forms opened.",
                            },
                            "exhibits_examined": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty — exhibits/documents opened.",
                            },
                            "material_open_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: present and empty.",
                            },
                            "search_runs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty search ids backing the coverage.",
                            },
                            "covered_branches": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: non-empty — material branches the run covered.",
                            },
                            "major_entities_missing": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: must be empty.",
                            },
                            "remaining_branches": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: must be empty.",
                            },
                            "routes_unsearched": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "sufficient: must be empty.",
                            },
                            "resolved": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional existing envelope: resolved questions (also partially_resolved/unresolved/source_limitations/dates/partitions/docs/gaps).",
                            },
                            "datasets_queried": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "FINRA sufficient: non-empty — datasets queried.",
                            },
                            "tickers_covered": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "FINRA sufficient: non-empty — tickers covered.",
                            },
                            "settlement_windows_covered": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "FINRA sufficient: non-empty — settlement/date windows covered.",
                            },
                            "dataset_reads": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "FINRA sufficient: non-empty — dataset reads backing the coverage.",
                            },
                            "semantic_branches_covered": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "WEB sufficient: non-empty — semantic branches covered.",
                            },
                            "queries_executed": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "WEB sufficient: non-empty — queries executed.",
                            },
                            "results_inspected": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "WEB sufficient: non-empty — results inspected.",
                            },
                        },
                        "required": ["useful_for_question"],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Evidence IDs grounding a sufficient result (empty only with insufficient).",
                    },
                    "unresolved_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Open questions left by the source run; must be empty with sufficient.",
                    },
                },
                "required": [
                    "session_id",
                    "job_id",
                    "coverage",
                    "evidence_ids",
                    "unresolved_questions",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_add_analysis",
            "description": "Records one committee analysis (stockbot, bullbot, or bearbot) on a running job; claim refs are validated against the frozen evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Running job ID the analysis belongs to.",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["stockbot", "bullbot", "bearbot"],
                        "description": "Committee role authoring the analysis.",
                    },
                    "analysis": {
                        "type": "object",
                        "description": "Committee output with claims grounded in frozen evidence IDs.",
                    },
                },
                "required": ["session_id", "job_id", "role", "analysis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_finalize",
            "description": "Persists the trio-joined synthesis and completes a research session; every claim must cite frozen evidence IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Research session ID.",
                    },
                    "answer": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Final synthesis prose.",
                    },
                    "claims": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Grounded findings; each claim needs non-empty text and non-empty evidence IDs from the freeze.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Finding text.",
                                },
                                "evidence_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string", "minLength": 1},
                                    "description": "Frozen evidence IDs grounding this claim.",
                                },
                            },
                            "required": ["text", "evidence_ids"],
                        },
                    },
                },
                "required": ["session_id", "answer", "claims"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_alternative_signals",
            "description": "Reads locally collected Google public-data discovery candidates (top/rising lists) with persistence/diffusion features only when exactly one PIT-valid v2 feature scope matches; otherwise features are null with available_feature_scopes listed. Candidates only, never materiality or investment claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring filter over candidate terms.",
                    },
                    "geo": {
                        "type": "string",
                        "description": "Geography filter, e.g. US.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": "Point-in-time date YYYY-MM-DD; candidates known after it are excluded.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max candidates (default 20).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_evidence",
            "description": "Bounded Google Trends discovery collection over public top/rising lists with stable source identity and retrieval timestamps. List membership only, never search-volume claims. Takes a named trend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Range start YYYY-MM-DD. Optional; both omitted defaults to trailing 7 days ending today UTC.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Range end YYYY-MM-DD. Optional; both omitted defaults to trailing 7 days ending today UTC.",
                    },
                    "geos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Geographies, e.g. [US].",
                    },
                    "geo": {
                        "type": "string",
                        "description": "Single geography shorthand for geos.",
                    },
                    "term": {
                        "type": "string",
                        "description": "Optional substring filter over collected terms.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Max rows (default 100).",
                    },
                    "week_start": {
                        "type": "string",
                        "description": "Interest-week start YYYY-MM-DD; omitted defaults to the trailing 14-day week window ending at end_date.",
                    },
                    "week_end": {
                        "type": "string",
                        "description": "Interest-week end YYYY-MM-DD; omitted defaults to the trailing 14-day week window ending at end_date.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate_social_arbitrage_candidate",
            "description": "Bounded enrichment for one discovery term: local signal evidence and SEC-confirmed/unresolved entity mappings plus a pointer to transient YouTube corroboration. Returns evidence and explicit gaps; never fabricates causality and never trades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Discovery term to investigate.",
                    },
                    "geo": {"type": "string", "description": "Geography, e.g. US."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Max evidence rows per source (default 5; YouTube never above 5).",
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_context",
            "description": "Bounded Data Commons statistical observations for explicit geography/variable IDs with unit/facet/provider provenance. Distinct facets stay distinct; never splices incompatible series. Takes explicit geography/variable IDs; do not use search_web for macro statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "geos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Geography DCIDs, e.g. [geoId/06].",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Statistical variable IDs, e.g. [Count_Person].",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Range start YYYY-MM-DD.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Range end YYYY-MM-DD.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max observations (default 100).",
                    },
                },
                "required": ["geos", "variables"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_company_patents",
            "description": "Bounded patent-publication search for documented company assignees via checked-in BigQuery templates. Counts publications explicitly; never labels counts as inventions or bullish signals. Patent records are authoritative here; never use search_web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_id": {
                        "type": "string",
                        "description": "Documented assignee name from existing company evidence.",
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Documented assignee aliases (verified, never inferred from matching text).",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Range start YYYY-MM-DD.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Range end YYYY-MM-DD.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Max publications (default 20).",
                    },
                },
                "required": ["company_id", "assignees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Current UTC date and time from the system clock. Call for 'what time is it' questions; never for market data, filings, or historical point-in-time facts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
# TOOL_REGISTRY_VERSION moved below TOOL_DISCOVERY_REGISTRY (hashes schemas + routing metadata).


def _robinhood_client(
    *,
    account_tools: frozenset[str] = frozenset(),
) -> RobinhoodClient:
    """Construct a broker client with only the MCP reads this handler needs."""
    if not broker_enabled():
        raise RuntimeError("Robinhood integration is disabled; set BROKER_ENABLED=true")
    url = get_robinhood_mcp_url()
    oauth = OAuthConfig(url)
    if not has_valid_tokens(oauth.server_origin, DEFAULT_TOKEN_PATH):
        raise RobinhoodAuthRequired("Robinhood OAuth is not set up or has expired")
    return RobinhoodClient(
        url,
        oauth=oauth,
        market_tools=capabilities.MARKET_READ_TOOLS,
        account_tools=account_tools,
    )


def authorize_robinhood_browser() -> bool:
    """Run the full OAuth flow now (browser + loopback callback), persisting
    tokens. True on success. Works regardless of BROKER_ENABLED — this is
    the setup step."""
    try:
        url = get_robinhood_mcp_url()
        client = RobinhoodClient(url, oauth=OAuthConfig(url))
        client.list_tools()  # SDK performs discovery + OAuth when required
        return True
    except Exception:
        logger.warning("Robinhood authorization failed", exc_info=True)
        return False


def _norm_row(row: dict[str, object]) -> dict[str, object]:
    """One provider row with string keys (provider JSON is untyped)."""
    return {str(k): v for k, v in row.items()}


def _parse_text_block(text: str) -> object:
    try:
        parsed: object = json.loads(text)
        return parsed
    except TypeError, ValueError:
        return {"text": text}


def _payload_from_content(value: dict[str, object]) -> object:
    """MCP content-blocks payload: first text block wins, else the raw dict."""
    content = value.get("content")
    if isinstance(content, list):
        for block in content:
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                return _parse_text_block(text)
    return value


def _provider_payload(value: object) -> object:
    """Unwrap MCP envelope (genuinely dynamic provider JSON)."""
    if isinstance(value, dict):
        structured = value.get("structured_content") or value.get("structuredContent")
        if structured is not None:
            payload: object = structured
            return payload
        assert isinstance(value, dict)
        return _payload_from_content(value)
    return value


def _direct_rows(unwrapped: object) -> list[dict[str, object]] | None:
    """Payload that is already rows (or never rows): list, non-dict, else None."""
    if isinstance(unwrapped, list):
        return [_norm_row(row) for row in unwrapped if isinstance(row, dict)]
    return [] if not isinstance(unwrapped, dict) else None


def _keyed_rows(unwrapped: dict[str, object], keys: tuple[str, ...]) -> list[dict[str, object]] | None:
    """First caller-keyed list hit, else None."""
    for key in keys:
        value = unwrapped.get(key)
        if isinstance(value, list):
            return [_norm_row(row) for row in value if isinstance(row, dict)]
    return None


def _default_hit(value: object, keys: tuple[str, ...]) -> list[dict[str, object]] | None:
    """One conventional-key hit: list rows, or non-empty nested rows, else None."""
    if isinstance(value, list):
        return [_norm_row(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        nested = _rows(value, *keys)
        if nested:
            return nested
    return None


def _default_rows(unwrapped: dict[str, object], keys: tuple[str, ...]) -> list[dict[str, object]]:
    """Conventional data/results/items/records keys (one nested level), else the dict itself."""
    for key in ("data", "results", "items", "records"):
        hit = _default_hit(unwrapped.get(key), keys)
        if hit is not None:
            return hit
    return [_norm_row(unwrapped)]


def _rows(payload: object, *keys: str) -> list[dict[str, object]]:
    unwrapped = _provider_payload(payload)
    direct = _direct_rows(unwrapped)
    if direct is not None:
        return direct
    assert isinstance(unwrapped, dict)
    hit = _keyed_rows(unwrapped, keys)
    if hit is not None:
        return hit
    return _default_rows(unwrapped, keys)


def _first(value: object, *keys: str) -> object:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if value.get(key) is not None:
            found: object = value[key]
            return found
    return None


def _quote_row(value: object) -> object:
    if isinstance(value, dict) and isinstance(value.get("quote"), dict):
        return value["quote"]
    return value


def get_market_snapshot(ticker: str) -> dict[str, object]:
    ticker = ticker.strip().upper()
    provider = RobinhoodPortfolioProvider(_robinhood_client())
    quote = provider.get_equity_quotes([ticker]).get(ticker)
    if quote is None:
        return {"error": f"No Robinhood quote found for {ticker}", "source": "robinhood_mcp"}
    return {
        "result_type": "market_snapshot",
        "ticker": ticker,
        "last": str(quote.last) if quote.last is not None else None,
        "bid": str(quote.bid) if quote.bid is not None else None,
        "ask": str(quote.ask) if quote.ask is not None else None,
        "retrieved_at": quote.retrieved_at.isoformat(),
        # *_local fields are the process host's local timezone, not the end user's.
        "retrieved_at_local": quote.retrieved_at.astimezone().isoformat(),
        "source": "robinhood_mcp",
    }


_PORTFOLIO_TOP_POSITIONS = 15
_PORTFOLIO_TOP_LARGEST = 5


def _breach_row(breach: RiskBreach) -> dict[str, object]:
    """One mandate breach with Decimal-safe string rendering."""
    return {
        "metric": breach.metric,
        "target": breach.target,
        "severity": breach.severity,
        "actual": str(breach.actual) if breach.actual is not None else None,
        "limit": str(breach.limit) if breach.limit is not None else None,
        "excess": str(breach.excess) if breach.excess is not None else None,
        "note": breach.note,
        "unit": breach.unit,
    }


def _mandate_issue_row(issue: EvaluationIssue) -> dict[str, object]:
    """One mandate issue with stable key order."""
    return {
        "code": issue.code,
        "metric": issue.metric,
        "target": issue.target,
        "position_id": issue.position_id,
        "ticker": issue.ticker,
    }


def _mandate_evaluation(evaluation: RiskEvaluation) -> dict[str, object]:
    """Evaluation dataclass -> model packet (snapshot ids, breaches, issues)."""
    return {
        "result_type": "mandate_evaluation",
        "snapshot_id": evaluation.snapshot_id,
        "snapshot_created_at": evaluation.created_at.isoformat(),
        "snapshot_created_at_local": evaluation.created_at.astimezone().isoformat(),
        "breaches": [_breach_row(breach) for breach in evaluation.breaches],
        "sector_exposures": {sector: str(weight) for sector, weight in evaluation.sector_exposures.items()},
        "issues": [_mandate_issue_row(issue) for issue in evaluation.issues],
        "source": "mandate",
    }


def evaluate_mandate(data_root: Path | None = None, mandate_path: Path | None = None) -> dict[str, object]:
    """Deterministic mandate evaluation over the latest persisted snapshot."""
    path = mandate_path or get_data_root() / "mandate.json"
    try:
        evaluation = risk_service.evaluate_latest_mandate(path, data_root=data_root)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    return _mandate_evaluation(evaluation)


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    """Lenient tool-JSON int coercion (None stays None, garbage raises)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    return int(str(value))


def _tool_function(tool: dict[str, object]) -> dict[str, object]:
    """OpenAI schema function dict (TOOLS entries are untyped app-side JSON)."""
    function = tool.get("function")
    if isinstance(function, dict):
        return {str(k): v for k, v in function.items()}
    return {}


ModelHandler = Callable[[dict[str, object], str], dict[str, object]]
ContextHandler = Callable[[dict[str, object], RequestContext], dict[str, object]]


def _freshness_key(item: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(item.get("sec_latest_filed_at") or "0000-00-00"),
        str(item.get("finra_settlement_date") or "0000-00-00"),
        str(item.get("finra_retrieved_at") or ""),
    )


def _bases_count_key(bases: list[object]) -> Callable[[str], int]:
    def _count(s: str) -> int:
        return bases.count(s)

    return _count


def _research_freshness(freshness_items: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-position research freshness to one latest non-empty dict."""
    non_empty = [item for item in freshness_items if item]
    if not non_empty:
        return {}
    return max(non_empty, key=_freshness_key)


def _position_research_row(position: Position, research_item: PortfolioResearchPosition | None) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": position.ticker,
        "quantity": str(position.quantity),
        "market_price": _str_or_none(position.market_price),
        "price_type": position.price_type,
        "market_value": _str_or_none(position.market_value),
        "portfolio_weight": _str_or_none(position.portfolio_weight),
        "unrealized_gain": _str_or_none(position.unrealized_gain),
        "security_id": position.security_id,
        "entity_id": position.entity_id,
        "resolved": position.entity_id is not None,
    }
    if research_item is not None:
        sec: dict[str, object] = {}
        for concept in SEC_CONCEPTS:
            fact = research_item.latest_sec_metrics.get(concept)
            if isinstance(fact, dict) and fact:
                sec[concept] = {
                    "value": _str_or_none(fact.get("value")),
                    "period_end": fact.get("period_end") or None,
                }
        row["sec"] = sec
        finra = research_item.latest_finra_metrics
        if finra:
            row["finra"] = {
                "short_position": _str_or_none(finra.get("short_position")),
                "prev_position": _str_or_none(finra.get("prev_position")),
                "change": _str_or_none(finra.get("short_interest_change")),
                "change_pct": _str_or_none(finra.get("short_interest_change_pct")),
                "days_to_cover": _str_or_none(finra.get("days_to_cover")),
                "settlement_date": finra.get("settlement_date") or None,
            }
    return row


def _get_portfolio_snapshot(arguments: dict[str, object], model: str) -> dict[str, object]:
    """Bounded, deterministic portfolio snapshot (spec §23)."""
    del model
    refresh = bool(arguments.get("refresh", False))
    provider = RobinhoodPortfolioProvider(
        _robinhood_client(
            account_tools=frozenset(
                {
                    "get_accounts",
                    "get_portfolio",
                    "get_equity_positions",
                }
            )
        )
    )
    if refresh:
        snapshot = sync_robinhood_portfolio(provider, data_root=None)
    else:
        snapshot = read_latest_snapshot(data_root=None) or sync_robinhood_portfolio(provider, data_root=None)
    research = _snapshot_research(snapshot)
    positions_by_id = _snapshot_positions_by_id(snapshot)
    position_rows = _snapshot_position_rows(snapshot, positions_by_id, research)
    omitted_count = max(0, len(snapshot.positions) - len(position_rows))
    return _snapshot_envelope(snapshot, position_rows, omitted_count, research)


def _snapshot_research(snapshot: PortfolioSnapshot) -> dict[str, PortfolioResearchPosition]:
    """Research items keyed by position id."""
    return {item.position.position_id: item for item in enrich_portfolio_research(snapshot)}


def _snapshot_positions_by_id(snapshot: PortfolioSnapshot) -> dict[str, Position]:
    """Positions keyed by position id."""
    return {position.position_id: position for position in snapshot.positions}


def _snapshot_position_rows(
    snapshot: PortfolioSnapshot,
    positions_by_id: dict[str, Position],
    research: dict[str, PortfolioResearchPosition],
) -> list[dict[str, object]]:
    """Top-N position rows in rank order."""
    ranked = largest_positions(
        [(position.position_id, position.market_value) for position in snapshot.positions],
        limit=_PORTFOLIO_TOP_POSITIONS,
    )
    return [
        _position_research_row(positions_by_id[position_id], research.get(position_id)) for position_id, _ in ranked
    ]


def _snapshot_largest(snapshot: PortfolioSnapshot) -> list[dict[str, object]]:
    """Largest positions by ticker for the envelope."""
    return [
        {"ticker": ticker, "market_value": _str_or_none(value)}
        for ticker, value in largest_positions(
            [(position.ticker, position.market_value) for position in snapshot.positions],
            limit=_PORTFOLIO_TOP_LARGEST,
        )
    ]


def _snapshot_envelope(
    snapshot: PortfolioSnapshot,
    position_rows: list[dict[str, object]],
    omitted_count: int,
    research: dict[str, PortfolioResearchPosition],
) -> dict[str, object]:
    """Result envelope for a loaded snapshot."""
    return {
        "result_type": "portfolio_snapshot",
        # Persistent snapshot/account identifiers stay local. Tool results are
        # rendered into model context, where they are not needed.
        "created_at": snapshot.created_at.isoformat(),
        "created_at_local": snapshot.created_at.astimezone().isoformat(),
        "broker": snapshot.broker,
        "account_count": len(snapshot.account_ids),
        "total_value": _str_or_none(snapshot.total_value),
        "cash": _str_or_none(snapshot.cash),
        "invested_value": _str_or_none(snapshot.invested_value),
        "position_count": len(snapshot.positions),
        "priced_position_count": sum(1 for position in snapshot.positions if position.market_value is not None),
        "unresolved_position_count": sum(1 for position in snapshot.positions if position.entity_id is None),
        "concentration": _str_or_none(
            portfolio_concentration([position.portfolio_weight for position in snapshot.positions])
        ),
        "positions": position_rows,
        "omitted_count": omitted_count,
        "largest_positions": _snapshot_largest(snapshot),
        "unresolved": [position.ticker for position in snapshot.positions if position.entity_id is None],
        "freshness": {
            "snapshot_created_at": snapshot.created_at.isoformat(),
            "snapshot_created_at_local": snapshot.created_at.astimezone().isoformat(),
            **_research_freshness([item.research_data_freshness for item in research.values()]),
        },
        "source": "robinhood_mcp",
    }


_SCAN_SPECS_CAP = 60
_SCAN_LIST_CAP = 60
_SCAN_RESULTS_ROWS = 20
_SCAN_WRITE_PREVIEW_ROWS = 10


def _scan_rows(data: dict[str, object]) -> list[dict[str, object]]:
    """Instrument rows from a scan payload under any of the common keys."""
    rows = _rows(data, "results", "instruments", "rows", "items")
    return rows if rows is not None else []


def _get_scanner_filter_specs(arguments: dict[str, object], model: str) -> dict[str, object]:
    del arguments, model
    data = RobinhoodPortfolioProvider(_robinhood_client()).get_scanner_filter_specs()
    specs = data.get("filter_specs")
    if isinstance(specs, list):
        rows = [{str(k): v for k, v in row.items()} for row in specs if isinstance(row, dict)]
    else:
        rows = [{k: v for k, v in row.items()} for row in _scan_rows(data)]
    result: dict[str, object] = {
        "result_type": "scan_specs",
        "count": len(rows),
        "specs": rows[:_SCAN_SPECS_CAP],
        "omitted_count": max(0, len(rows) - _SCAN_SPECS_CAP),
        "source": "robinhood_mcp",
    }
    return result


def _get_scans(arguments: dict[str, object], model: str) -> dict[str, object]:
    del arguments, model
    rows = RobinhoodPortfolioProvider(_robinhood_client(account_tools=frozenset({"get_scans"}))).get_scans()
    result: dict[str, object] = {
        "result_type": "scan_list",
        "count": len(rows),
        "scans": rows[:_SCAN_LIST_CAP],
        "omitted_count": max(0, len(rows) - _SCAN_LIST_CAP),
        "source": "robinhood_mcp",
    }
    return result


def _run_scan(arguments: dict[str, object], model: str) -> dict[str, object]:
    del model
    scan_id = str(arguments["scan_id"])
    limit = max(1, min(int(str(arguments.get("limit") or _SCAN_RESULTS_ROWS)), 25))
    data = RobinhoodPortfolioProvider(_robinhood_client(account_tools=frozenset({"run_scan"}))).run_scan(scan_id)
    rows = _scan_rows(data)
    return {
        "result_type": "scan_results",
        "scan_id": scan_id,
        "title": str(_first(data, "title", "name") or ""),
        "total": _first(data, "total", "total_matches", "match_count", "count"),
        "rows": rows[:limit],
        "omitted": max(0, len(rows) - limit),
        "sort": _first(data, "sort", "sort_order"),
        "filters": _first(data, "filters", "active_filters"),
        "live": True,
        "source": "robinhood_mcp",
    }


def _call_broker_tool(client: RobinhoodClient, name: str, arguments: dict[str, object]) -> object:
    """Broker tool call without static MCP typing."""
    return client.call_tool(name, arguments)


def _chain_id_for(client: RobinhoodClient, ticker: str) -> object:
    """First chain id for the ticker, else None (provider JSON is untyped)."""
    chain = _provider_payload(_call_broker_tool(client, "get_option_chains", {"underlying_symbol": ticker}))
    chain_rows = _rows(chain, "chains", "option_chains")
    return _first(chain_rows[0], "chain_id", "chainId", "id") if chain_rows else None


def _instrument_args(ticker: str, option_type: str, chain_id: object, filters: dict[str, object]) -> dict[str, object]:
    """Provider instrument query: chain scoping plus optional expiry/state."""
    instrument_args: dict[str, object] = {"chain_symbol": ticker, "type": option_type}
    if chain_id:
        instrument_args["chain_id"] = chain_id
    if filters.get("expiration_date") is not None:
        instrument_args["expiration_dates"] = filters["expiration_date"]
    if filters.get("state") is not None:
        instrument_args["state"] = filters["state"]
    return instrument_args


def _of_option_type(instruments: list[dict[str, object]], option_type: str) -> list[dict[str, object]]:
    """Keep only rows of the requested put/call type (provider echoes both)."""
    return [
        row
        for row in instruments
        if str(_first(row, "type", "option_type", "optionType") or option_type).lower() in {option_type, option_type[0]}
    ]


def _row_dte(row: dict[str, object], today: date) -> int | None:
    """Days to expiry for one instrument row, None when unparseable."""
    expiration = str(_first(row, "expiration", "expiration_date", "expirationDate") or "")[:10]
    try:
        return (date.fromisoformat(expiration) - today).days
    except ValueError:
        return None


def _row_strike_value(row: dict[str, object]) -> Decimal | None:
    """Strike for one instrument row, None when missing/non-numeric."""
    strike = _first(row, "strike", "strike_price", "strikePrice")
    try:
        return Decimal(str(strike))
    except ValueError, TypeError, ArithmeticError:
        return None


def _dte_in_range(dte: int | None, filters: dict[str, object]) -> bool:
    """min_dte/max_dte window; unparseable expiry never passes a bound."""
    min_dte = filters.get("min_dte")
    if min_dte is not None and (dte is None or dte < int(str(min_dte))):
        return False
    max_dte = filters.get("max_dte")
    return not (max_dte is not None and (dte is None or dte > int(str(max_dte))))


def _strike_in_range(strike_value: Decimal | None, filters: dict[str, object]) -> bool:
    """strike_min/strike_max window; missing strike never passes a bound."""
    strike_min = filters.get("strike_min")
    if strike_min is not None and (strike_value is None or strike_value < Decimal(str(strike_min))):
        return False
    strike_max = filters.get("strike_max")
    return not (strike_max is not None and (strike_value is None or strike_value > Decimal(str(strike_max))))


def _filter_instruments(
    instruments: list[dict[str, object]], filters: dict[str, object], today: date
) -> list[dict[str, object]]:
    """DTE/strike window over instrument rows."""
    return [
        row
        for row in instruments
        if _dte_in_range(_row_dte(row, today), filters) and _strike_in_range(_row_strike_value(row), filters)
    ]


def _quotes_by_id(client: RobinhoodClient, instruments: list[dict[str, object]]) -> dict[str, object]:
    """Quote rows keyed by instrument id (empty when no instruments)."""
    ids = [_first(row, "id", "instrument_id", "contract_id") for row in instruments]
    ids = [str(value) for value in ids if value]
    quotes = (
        _rows(
            _call_broker_tool(client, "get_option_quotes", {"instrument_ids": ids}),
            "quotes",
            "option_quotes",
            "results",
        )
        if ids
        else []
    )
    quotes_by_id: dict[str, object] = {}
    for row in quotes:
        quote = _quote_row(row)
        quote_id = _first(quote, "id", "instrument_id", "contract_id")
        if quote_id:
            quotes_by_id[str(quote_id)] = quote
    return quotes_by_id


def _merge_quotes(
    instruments: list[dict[str, object]], quotes_by_id: dict[str, object], ticker: str
) -> list[OptionQuote]:
    """Instrument + quote merge normalized to OptionQuotes (un-normalizable rows dropped)."""
    normalized: list[OptionQuote] = []
    for instrument in instruments:
        instrument_id = str(_first(instrument, "id", "instrument_id", "contract_id") or "")
        merged: dict[str, object] = dict(instrument)
        raw_quote = quotes_by_id.get(instrument_id)
        if isinstance(raw_quote, dict):
            merged.update(raw_quote)
        merged["contract_id"] = instrument_id
        merged["ticker"] = ticker
        try:
            normalized.append(normalize_option_quote(merged, ticker=ticker))
        except ValueError:
            continue
    return normalized


def _load_option_quotes(ticker: str, option_type: str, **filters: object) -> list[OptionQuote]:
    client = _robinhood_client()
    chain_id = _chain_id_for(client, ticker)
    instruments = _rows(
        client.call_tool("get_option_instruments", _instrument_args(ticker, option_type, chain_id, filters)),
        "instruments",
        "option_instruments",
    )
    instruments = _filter_instruments(_of_option_type(instruments, option_type), filters, datetime.now(UTC).date())
    return _merge_quotes(instruments, _quotes_by_id(client, instruments), ticker)


def _filter_quotes(
    quotes: list[OptionQuote], min_dte: object, max_dte: object, strike_min: object, strike_max: object
) -> list[OptionQuote]:
    """DTE/strike window over normalized quotes (loader already applied the same window pre-quote)."""
    today = datetime.now(UTC).date()
    return [
        quote
        for quote in quotes
        if (min_dte is None or (quote.expiration - today).days >= int(str(min_dte)))
        and (max_dte is None or (quote.expiration - today).days <= int(str(max_dte)))
        and (strike_min is None or quote.strike >= Decimal(str(strike_min)))
        and (strike_max is None or quote.strike <= Decimal(str(strike_max)))
    ]


def _no_quotes_error(ticker: str, option_type: str) -> dict[str, object]:
    """Empty-filter envelope shared by chain/compare (ticker already uppercased)."""
    return {
        "error": f"No Robinhood {option_type} contracts matched the requested filters for {ticker}",
        "source": "robinhood_mcp",
    }


def get_option_chain(
    ticker: str,
    option_type: str,
    min_dte: object = None,
    max_dte: object = None,
    strike_min: object = None,
    strike_max: object = None,
    limit: object = 20,
) -> dict[str, object]:
    ticker = ticker.strip().upper()
    option_type = option_type.lower()
    quotes = _load_option_quotes(
        ticker,
        option_type,
        min_dte=min_dte,
        max_dte=max_dte,
        strike_min=strike_min,
        strike_max=strike_max,
    )
    filtered = _filter_quotes(quotes, min_dte, max_dte, strike_min, strike_max)
    if not filtered:
        return _no_quotes_error(ticker, option_type)
    bounded = max(1, min(int(str(limit or 20)), 30))
    return {
        "result_type": "option_chain",
        "ticker": ticker,
        "option_type": option_type,
        "contracts": [analyze_option(quote) for quote in filtered[:bounded]],
        "matched": len(filtered),
        "returned": min(len(filtered), bounded),
        "filters": {"min_dte": min_dte, "max_dte": max_dte, "strike_min": strike_min, "strike_max": strike_max},
        "source": "robinhood_mcp",
    }


def analyze_option_contract(
    ticker: str, expiration: str, strike: object, option_type: str, target_price: object = None
) -> dict[str, object]:
    quotes = _load_option_quotes(ticker.strip().upper(), option_type.lower(), expiration_date=expiration)
    matches = [
        quote for quote in quotes if quote.expiration.isoformat() == expiration and quote.strike == Decimal(str(strike))
    ]
    if not matches:
        return {"error": "No matching Robinhood option contract found", "source": "robinhood_mcp"}
    return {
        "result_type": "option_analysis",
        **analyze_option(matches[0], target_price=(str(target_price) if target_price is not None else None)),
        "source": "robinhood_mcp",
    }


def compare_robinhood_options(
    ticker: str,
    option_type: str,
    target_price: object,
    min_dte: object = None,
    max_dte: object = None,
    strike_min: object = None,
    strike_max: object = None,
    limit: object = 20,
) -> dict[str, object]:
    quotes = _load_option_quotes(
        ticker.strip().upper(),
        option_type.lower(),
        min_dte=min_dte,
        max_dte=max_dte,
        strike_min=strike_min,
        strike_max=strike_max,
    )
    filtered = _filter_quotes(quotes, min_dte, max_dte, strike_min, strike_max)
    if not filtered:
        return _no_quotes_error(ticker.strip().upper(), option_type.lower())
    return {
        "result_type": "option_comparison",
        "ticker": ticker.upper(),
        "source": "robinhood_mcp",
        **compare_options(
            filtered,
            target_price=(str(target_price) if target_price is not None else None),
            limit=int(str(limit or 20)),
        ),
    }


def _search_web(args: dict[str, object], model: str) -> dict[str, object]:
    """Exa-backed web search; harness-level soft failures must not stop the run."""
    raw_inc = args.get("include_domains")
    if isinstance(raw_inc, list):
        include_domains: list[str] | None = [str(x) for x in raw_inc]
    else:
        include_domains = None
    raw_exc = args.get("exclude_domains")
    if isinstance(raw_exc, list):
        exclude_domains: list[str] | None = [str(x) for x in raw_exc]
    else:
        exclude_domains = None
    start, err = _finra_date(args.get("start_published_date"), "search_web", "start_published_date")
    if err is not None:
        return err
    end, err = _finra_date(args.get("end_published_date"), "search_web", "end_published_date")
    if err is not None:
        return err
    result = exa_client.search(
        str(args["query"]),
        category=_str_or_none(args.get("category")),
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        start_published_date=start,
        end_published_date=end,
        search_type=str(args.get("search_type") or "auto"),
        limit=args.get("limit") or exa_client.EXA_DEFAULT_LIMIT,
    )
    if isinstance(result, dict) and "error" in result:
        result["soft"] = True  # harness-level: soft failures must not stop the run
    return result


def plan_public_search_queries(
    primary_name: str | None = None,
    primary_ticker: str | None = None,
    related_names: Sequence[object] = (),
) -> list[dict[str, object]]:
    """PUBLIC search targets → search_web args (planning only, no numbers).

    Targets come only from the user request, canonical/public
    relationships, public screens, or explicitly named companies.
    Queries carry names only — never quantities/weights/cost
    basis/account IDs (`authorize_egress` + `private_pattern_hit`
    remain the gate, unchanged).
    """
    # ponytail: names-only mapping, no query-builder lib for 3 strings
    targets: list[str] = []
    seen: set[str] = set()

    def _add(name: object) -> None:
        if not isinstance(name, str):
            return
        key = name.strip()
        if not key or key.casefold() in seen:
            return
        seen.add(key.casefold())
        targets.append(key)

    if isinstance(primary_name, str) and primary_name.strip():
        _add(primary_name.strip())
    elif isinstance(primary_ticker, str) and primary_ticker.strip():
        _add(primary_ticker.strip().upper())
    for name in related_names or ():
        _add(name)
    return [{"query": f"{name} recent announcements"} for name in targets[:3]]


def _other_end(rel: EntityRelationship, primary_entity_id: str) -> str | None:
    """The far end of one relationship touching the primary, else None."""
    try:
        if rel.from_entity_id == primary_entity_id:
            return rel.to_entity_id
        if rel.to_entity_id == primary_entity_id:
            return rel.from_entity_id
    except AttributeError:
        return None
    return None


def _related_names(
    primary_entity_id: str, relationships: Sequence[EntityRelationship], names_by_entity: dict[str, str] | None
) -> list[str]:
    """Names one hop from the primary via EntityRelationships."""
    if not isinstance(names_by_entity, dict):
        return []
    others: list[str] = []
    for rel in relationships or ():
        other = _other_end(rel, primary_entity_id)
        if other:
            others.append(other)
    return [names_by_entity[other] for other in others if other in names_by_entity]


def suggest_public_search_queries(
    primary_entity_id: str | None,
    primary_name: str | None,
    primary_ticker: str | None,
    relationships: Sequence[EntityRelationship] = (),
    names_by_entity: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Warehouse-aware wrapper: single-hop EntityRelationships."""
    related = _related_names(primary_entity_id, relationships, names_by_entity) if primary_entity_id else []
    return plan_public_search_queries(primary_name, primary_ticker, related)


def _google_soft(result: dict[str, object]) -> dict[str, object]:
    """Collector passthrough: error dicts get soft:true like _search_web."""
    if isinstance(result, dict) and "error" in result and "soft" not in result:
        result = dict(result)
        result["soft"] = True
    return result


def _google_import_error(source: str, exc: Exception) -> dict[str, object]:
    return {
        "status": "unavailable",
        "source": source,
        "soft": True,
        "error": f"Google data unavailable: {exc}",
        "error_type": "source_unavailable",
    }


def _arg_str(args: dict[str, object], key: str) -> str | None:
    """JSON-boundary narrow: schema strings only, None otherwise."""
    raw = args.get(key)
    return raw if isinstance(raw, str) else None


def _arg_str_list(args: dict[str, object], key: str) -> list[str]:
    raw = args.get(key)
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, str)]
    return []


def _arg_int(args: dict[str, object], key: str, default: int) -> int:
    raw = args.get(key, default)
    return int(raw) if isinstance(raw, (int, str)) else default


def _signals_capped(rows: object, limit: int) -> bool:
    """Continuation flag: row count reaches the requested limit."""
    count = len(rows) if isinstance(rows, list) else 0
    try:
        return count >= max(1, limit)
    except TypeError, ValueError:
        return False


def _local_signals_packet(args: dict[str, object], rows: object, limit: int) -> dict[str, object]:
    """Local-signal ok packet with continuation capped at the requested limit."""
    capped = _signals_capped(rows, limit)
    assert isinstance(rows, list)
    return {
        "status": "ok",
        "source": "google",
        "signals": rows,
        "count": len(rows),
        "coverage": {"query": _arg_str(args, "query"), "geo": _arg_str(args, "geo"), "as_of": _arg_str(args, "as_of")},
        "warnings": [],
        "continuation": capped,
    }


def _find_alternative_signals(args: dict[str, object], model: str) -> dict[str, object]:
    """No persisted signal store: ephemeral per-collection compute only, never raises."""
    # Seam: ephemeral per-collection compute via collect_trends + normalization; caller logs to the run bundle + raw_archive; NOTE: warehouse slots behind live readers.
    try:
        from .google_data import signals as _signals  # noqa: F401 - seam anchor: signals normalize live per collection
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _google_import_error("google", exc)
    try:
        limit = _arg_int(args, "limit", 20)
        return _local_signals_packet(args, [], limit)
    except Exception as exc:
        logger.exception("find_alternative_signals failed")
        return {"error": f"Tool 'find_alternative_signals' failed: {exc}", "soft": True, "source": "google"}


def _trend_geos(args: dict[str, object]) -> list[str]:
    """Geos for the trends collector: explicit list wins, else geo, else US."""
    geo = _arg_str(args, "geo")
    raw_geos = args.get("geos")
    str_geos: list[str] = [g for g in raw_geos if isinstance(g, str)] if isinstance(raw_geos, list) else []
    return str_geos or ([geo] if geo else ["US"])


def _trend_window(args: dict[str, object]) -> tuple[str | None, str | None]:
    """Date window for trends: explicit dates win, else the trailing 7 days."""
    start_date = _arg_str(args, "start_date")
    end_date = _arg_str(args, "end_date")
    if start_date is None and end_date is None:
        _today = datetime.now(UTC).date()
        return (_today - timedelta(days=6)).isoformat(), _today.isoformat()
    return start_date, end_date


def _get_trend_evidence(args: dict[str, object], model: str) -> dict[str, object]:
    try:
        from .google_data import trends as _trends
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _google_import_error("trends", exc)
    try:
        start_date, end_date = _trend_window(args)
        return _google_soft(
            _trends.collect_trends(
                start_date=start_date,
                end_date=end_date,
                geos=_trend_geos(args),
                limit=_arg_int(args, "limit", 100),
                data_root=get_data_root(),
                week_start=_arg_str(args, "week_start"),
                week_end=_arg_str(args, "week_end"),
                term=_arg_str(args, "term"),
            )
        )
    except Exception as exc:
        logger.exception("get_trend_evidence failed")
        return {"error": f"Tool 'get_trend_evidence' failed: {exc}", "soft": True, "source": "trends"}


def _investigate_term(args: dict[str, object]) -> str:
    """Search term for the candidate: schema string, else stringified."""
    term_raw = args.get("term", "")
    return term_raw if isinstance(term_raw, str) else str(term_raw or "")


def _collect_investigate_signals(
    term: str, geo: str, per_source: int, evidence: dict[str, object], gaps: list[str]
) -> None:
    """No persisted signal store: ephemeral per-collection compute only; record a gap."""
    # Seam: ephemeral per-collection compute via collect_trends + normalization; caller logs to the run bundle + raw_archive; NOTE: warehouse slots behind live readers.
    del term, geo, per_source, evidence
    gaps.append("signals unavailable: no persisted signal store (ephemeral per-collection compute only)")


def _classify_investigate_entity(ent: object, term: str, confirmed: list[object], unresolved: list[object]) -> None:
    """One SEC candidate -> confirmed (verified + CIK) or unresolved."""
    cik = getattr(ent, "cik", None)
    entry = {
        "name": getattr(ent, "name", None) or term,
        "cik": cik,
        "verification_status": getattr(ent, "verification_status", None),
    }
    if getattr(ent, "verification_status", None) == "verified" and cik:
        confirmed.append(entry)
    else:
        unresolved.append(entry)


def _resolve_investigate_ticker(term: str, confirmed: list[object], unresolved: list[object]) -> None:
    """Ticker-alias corroboration via provider candidates; unresolved ticker only when nothing else confirmed anything."""
    from .data_sources import SourceGateway
    from .domain.market.identity import resolve_ticker_aliases as _resolve_alias

    as_of = datetime.now(UTC)
    candidates = SourceGateway().ticker_candidates(term.upper(), as_of)
    resolution = _resolve_alias(term.upper(), candidates, as_of=as_of)
    if resolution.resolved:
        confirmed.append(
            {
                "ticker": term.upper(),
                "entity_id": resolution.entity_id,
                "security_id": resolution.security_id,
                "via": "ticker_alias",
            }
        )
    elif not confirmed and not unresolved:
        unresolved.append({"ticker": term.upper(), "reason": "unresolved"})


def _collect_investigate_entities(
    term: str, confirmed: list[object], unresolved: list[object], gaps: list[str]
) -> None:
    """SEC + ticker-alias corroboration; failures become gaps, never raises."""
    try:
        from .sec.discovery.service import find_sec_entities as _find_sec

        sec = _find_sec(query=term, max_results=5, data_root=get_data_root())
        for ent in list(getattr(sec, "entities", None) or []):
            _classify_investigate_entity(ent, term, confirmed, unresolved)
        _resolve_investigate_ticker(term, confirmed, unresolved)
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        gaps.append(f"entity resolution unavailable: {exc}")


def _investigate_social_arbitrage_candidate(args: dict[str, object], model: str) -> dict[str, object]:
    """Evidence + gaps for one term; corroboration capped, causality never claimed."""
    term = _investigate_term(args)
    geo = _arg_str(args, "geo") or "US"
    per_source = min(max(_arg_int(args, "limit", 5), 1), 25)
    evidence: dict[str, object] = {}
    confirmed: list[object] = []
    unresolved: list[object] = []
    gaps: list[str] = []
    result: dict[str, object] = {
        "term": term,
        "geo": geo,
        "source": "google",
        "status": "ok",
        "evidence": evidence,
        "entities": {"confirmed": confirmed, "unresolved": unresolved},
        "gaps": gaps,
    }
    _collect_investigate_signals(term, geo, per_source, evidence, gaps)
    _collect_investigate_entities(term, confirmed, unresolved, gaps)
    # ponytail: no YouTube imports/calls/data here — evidence table has no expiry, so API content must not enter tool results
    gaps.append(
        "youtube metrics excluded from saved evidence; run /youtube-analytics <thesis-id-or-slug> for the retention-safe view"
    )
    if len(gaps) >= 3 and not evidence:
        result.update({"status": "unavailable", "soft": True, "error": "; ".join(gaps)})
    return result


def _get_macro_context(args: dict[str, object], model: str) -> dict[str, object]:
    try:
        from .google_data import datacommons as _dc
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _google_import_error("datacommons", exc)
    try:
        return _google_soft(
            _dc.get_macro_context(
                _arg_str_list(args, "geos"),
                _arg_str_list(args, "variables"),
                start_date=_arg_str(args, "start_date"),
                end_date=_arg_str(args, "end_date"),
                limit=_arg_int(args, "limit", 100),
            )
        )
    except Exception as exc:
        logger.exception("get_macro_context failed")
        return {"error": f"Tool 'get_macro_context' failed: {exc}", "soft": True, "source": "datacommons"}


def _patent_company_id(args: dict[str, object]) -> str:
    """Validated company_id for the patents collector (non-empty string)."""
    company_id = args["company_id"]
    if not isinstance(company_id, str) or not company_id:
        raise TypeError(f"company_id must be a non-empty string, got {type(company_id).__name__}")
    return company_id


def _patent_assignees(args: dict[str, object]) -> list[str] | None:
    """Optional assignee filter: strings only, else None."""
    assignees_raw = args.get("assignees")
    return [a for a in assignees_raw if isinstance(a, str)] if isinstance(assignees_raw, list) else None


def _search_company_patents(args: dict[str, object], model: str) -> dict[str, object]:
    try:
        from .google_data import patents as _patents
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _google_import_error("patents", exc)
    try:
        return _google_soft(
            _patents.search_company_patents(
                _patent_company_id(args),
                start_date=_arg_str(args, "start_date"),
                end_date=_arg_str(args, "end_date"),
                limit=_arg_int(args, "limit", 20),
                assignees=_patent_assignees(args),
            )
        )
    except Exception as exc:
        logger.exception("search_company_patents failed")
        return {"error": f"Tool 'search_company_patents' failed: {exc}", "soft": True, "source": "patents"}


def _get_current_time(args: dict[str, object], model: str) -> dict[str, object]:
    del args, model
    return {"utc_now": datetime.now(UTC).isoformat(), "source": "system-clock"}


def _wrap_one(record: object) -> object:
    """One SEC list item: to_dict when available, else a plain copy."""
    if hasattr(record, "to_dict"):
        to_dict = getattr(record, "to_dict")  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
        if callable(to_dict):
            return to_dict()
        return record
    if isinstance(record, dict):
        return dict(record)
    if isinstance(record, (list, tuple)):
        return list(record)
    return record


def _wrap_list(identifier: object, records: object, key: str) -> dict[str, object]:
    """SEC list results: identifier echo, count, to_dict records, source."""
    rec_list: list[object] = list(records) if isinstance(records, (list, tuple)) else []
    items = [_wrap_one(r) for r in rec_list]
    return {"subject": identifier, "count": len(items), key: items, "source": "SEC EDGAR"}


# Typed discovery metadata for progressive tool discovery.


@dataclass(frozen=True)
class ToolDiscovery:
    """Catalog metadata for one RESEARCH tool (generic lexical search source)."""

    domain: str
    family: str
    intent: str
    output_kind: str
    source: str
    entity_scope: str
    time_mode: str
    summary: str
    choose_when: tuple[str, ...]
    reject_when: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    related_tools: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    direct_activation: bool = True


# Single source for domain descriptions (list_tool_domains + catalog generator share this).
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "alternative": "Alternative and non-filing signals outside standard SEC and market feeds.",
    "analyst": "Analyst estimates and expectations for earnings, revenue, and price targets.",
    "events": "Material company events derived from 8-K and filing activity.",
    "sec": "SEC filing discovery, retrieval, and document reading via EDGAR.",
    "finra": "FINRA short interest, short volume, and threshold-securities data.",
    "fundamentals": "Reported numeric fundamentals such as EPS, dividends, and balance-sheet items.",
    "governance": "Proxy, meeting, vote, and board-compensation records.",
    "insider": "Insider transactions and planned sales from Forms 3/4/5 and 144.",
    "macro": "Macroeconomic context such as employment, inflation, and rates.",
    "market": "Market data such as index weights, option contracts, and trend evidence.",
    "offerings": "Financing history, offering terms, and dilution math.",
    "ownership": "Beneficial ownership stakes, holder changes, and relationship links.",
    "patents": "Patent records and innovation activity.",
    "research": "Live research sessions: questions, jobs, evidence, freezes, and dossiers.",
    "thesis": "Thesis tracking, refinement, obligations, and operator notes.",
    "time": "Current time from the system clock.",
    "transactions": "Transaction status and mandate evaluation for deals.",
    "valuation": "Valuation multiples and financial-statement analysis.",
    "web": "General web search for facts outside structured financial sources.",
}


TOOL_DISCOVERY_REGISTRY: dict[str, ToolDiscovery] = {
    "describe_finra_dataset": ToolDiscovery(
        domain="finra",
        family="catalog",
        intent="inspect_finra_dataset_schema",
        output_kind="schema",
        source="finra",
        entity_scope="single_dataset",
        time_mode="current",
        summary="One FINRA dataset's fields, types, filter values, and supported methods.",
        choose_when=(
            "Learning a named FINRA dataset's fields, types, filters, and coverage before querying.",
            "what is in.",
            "fields and coverage.",
        ),
        reject_when=(
            "Not for finding which dataset covers a question (list_finra_datasets).",
            "Not for analyzed briefings.",
        ),
        conflicts_with=("list_finra_datasets",),
        related_tools=(
            "list_finra_datasets",
            "query_finra",
            "get_finra_datapoints",
        ),
        prerequisites=(),
    ),
    "diff_risk_factors": ToolDiscovery(
        domain="sec",
        family="filing-diff",
        intent="compare_risk_factors",
        output_kind="diff",
        source="sec",
        entity_scope="single_security",
        time_mode="latest",
        summary="Self-contained Risk Factors year-over-year diff for one ticker: what is new or changed.",
        choose_when=("What is new or changed in a company's risk disclosures for one ticker.",),
        reject_when=(
            "Do NOT use for full-filing diffs between accessions (diff_sec_filings).",
            "Do NOT use for disclosure search without change framing (search_sec_filings).",
            "Self-contained for one ticker; do NOT call list_sec_filings before or after.",
        ),
        conflicts_with=(
            "diff_sec_filings",
            "search_sec_filings",
        ),
        related_tools=(
            "diff_sec_filings",
            "search_sec_filings",
        ),
        prerequisites=(),
    ),
    "diff_sec_filings": ToolDiscovery(
        domain="sec",
        family="filing-diff",
        intent="compare_full_filings",
        output_kind="diff",
        source="sec",
        entity_scope="filing_pair_or_security",
        time_mode="latest_or_as_of",
        summary="Self-contained full-filing diff for one ticker or two accessions: amendment versus prior version.",
        choose_when=(
            "Comparing a ticker's latest amendment filing versus its predecessor filing.",
            "Comparing two known filing accessions for amendment or restatement changes.",
        ),
        reject_when=(
            "Do NOT use for risk-factor-only year-over-year diffs (diff_risk_factors).",
            "Do NOT call list_sec_filings first; ticker resolution is internal.",
            "Do NOT use for disclosure search without change framing (search_sec_filings).",
        ),
        conflicts_with=(
            "diff_risk_factors",
            "search_sec_filings",
        ),
        related_tools=(
            "diff_risk_factors",
            "get_sec_filing",
        ),
        prerequisites=(),
    ),
    "find_alternative_signals": ToolDiscovery(
        domain="alternative",
        family="signal-discovery",
        intent="discover_emerging_signals",
        output_kind="ranked_candidates",
        source="google_trends",
        entity_scope="market_wide",
        time_mode="latest_or_as_of",
        summary="Discovery scan for rising search-term and diffusion signals worth investigating.",
        choose_when=("Screening for emerging trend or attention signals across terms.",),
        reject_when=(
            "Not for evidence on one known trend.",
            "Not for a dated, geography-specific trend question.",
        ),
        conflicts_with=(),
        related_tools=(
            "get_trend_evidence",
            "investigate_social_arbitrage_candidate",
        ),
        prerequisites=(),
    ),
    "find_sec_entities": ToolDiscovery(
        domain="sec",
        family="entity-discovery",
        intent="resolve_sec_entity",
        output_kind="candidate_records",
        source="sec",
        entity_scope="entity_query",
        time_mode="current",
        summary="Resolve a company name, ticker, or CIK to verified SEC entity candidates with CIKs and tickers.",
        choose_when=("Starting from a company name when the exact ticker or CIK is not known.",),
        reject_when=(
            "Unneeded when the exact ticker or CIK is already known.",
            "Not for the quick bounded lookup (find_sec_entities_bounded).",
        ),
        conflicts_with=("find_sec_entities_bounded",),
        related_tools=(
            "find_sec_entities_bounded",
            "list_sec_filings",
            "search_sec_filings",
        ),
        prerequisites=(),
    ),
    "find_sec_entities_bounded": ToolDiscovery(
        domain="sec",
        family="entity-discovery",
        intent="resolve_sec_entity_bounded",
        output_kind="candidate_records",
        source="sec",
        entity_scope="entity_query",
        time_mode="current",
        summary="Quick bounded entity lookup: same verified candidates, fast routes only, capped at limit.",
        choose_when=("One identity check when full candidate coverage is not needed.",),
        reject_when=(
            "Not for full candidate coverage (find_sec_entities).",
            "Unneeded when the exact ticker or CIK is already known.",
        ),
        conflicts_with=("find_sec_entities",),
        related_tools=(
            "find_sec_entities",
            "list_sec_filings",
        ),
        prerequisites=(),
    ),
    "get_analyst_estimates": ToolDiscovery(
        domain="analyst",
        family="estimates",
        intent="retrieve_forward_consensus",
        output_kind="forecast_snapshot",
        source="yahoo_finance",
        entity_scope="single_security",
        time_mode="latest",
        summary="Forward sell-side consensus expectations: targets, ratings, forward EPS/revenue, revisions.",
        choose_when=("What analysts expect for one ticker: targets, consensus EPS, or estimate revisions.",),
        reject_when=(
            "Do NOT use for reported historical EPS (get_fundamentals).",
            "Do NOT use for cheap-vs-expensive multiples (get_valuation_metrics).",
        ),
        conflicts_with=(
            "get_fundamentals",
            "get_valuation_metrics",
        ),
        related_tools=(
            "get_valuation_metrics",
            "get_fundamentals",
        ),
        prerequisites=(),
    ),
    "get_beneficial_ownership": ToolDiscovery(
        domain="ownership",
        family="stakes",
        intent="retrieve_current_beneficial_owners",
        output_kind="current_snapshot",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Current 5%+ beneficial-ownership stakes (SC 13D/G): holder, shares, percent, voting powers.",
        choose_when=("Finding who owns more than 5% of a company.",),
        reject_when=(
            "Not for stake changes over time (get_ownership_changes).",
            "Not for relationship links in either direction (search_sec_relationships).",
            "Answer from these records; do not open filings or pull changes unless asked.",
        ),
        conflicts_with=(
            "get_ownership_changes",
            "search_sec_relationships",
        ),
        related_tools=(
            "get_ownership_changes",
            "search_sec_relationships",
        ),
        prerequisites=(),
    ),
    "get_dilution_profile": ToolDiscovery(
        domain="offerings",
        family="capital-raising",
        intent="calculate_dilution",
        output_kind="derived_analysis",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Deterministic dilution math for diluted shareholders: inputs, formula, and source accessions always shown.",
        choose_when=("Quantifying share-count impact from offerings, converts, or warrants.",),
        reject_when=("Not for offering-terms history (get_offering_history).",),
        conflicts_with=("get_offering_history",),
        related_tools=("get_offering_history",),
        prerequisites=(),
    ),
    "get_financial_statements": ToolDiscovery(
        domain="fundamentals",
        family="statements",
        intent="retrieve_financial_statement",
        output_kind="statement",
        source="sec",
        entity_scope="single_security",
        time_mode="latest",
        summary="Full parsed statements for one ticker: income statement, balance sheet, and cash flow.",
        choose_when=("Full financial statements rather than one numeric metric.",),
        reject_when=(
            "Do NOT use for a single metric like EPS (get_fundamentals).",
            "Do NOT use for a single XBRL fact (get_xbrl_facts).",
        ),
        conflicts_with=(
            "get_fundamentals",
            "get_xbrl_facts",
        ),
        related_tools=(
            "get_fundamentals",
            "get_xbrl_facts",
        ),
        prerequisites=(),
    ),
    "get_finra_datapoints": ToolDiscovery(
        domain="finra",
        family="short-interest",
        intent="retrieve_finra_records",
        output_kind="raw_records",
        source="finra",
        entity_scope="single_dataset",
        time_mode="date_range_or_latest",
        summary="Exact raw rows from any named FINRA dataset: only the requested fields, up to 25 rows.",
        choose_when=("Exact fields and values from a named FINRA dataset for an explicit data request.",),
        reject_when=(
            "Do NOT use for analyzed briefings or trends (query_finra).",
            "Do NOT use for one ticker's current short position (get_short_interest).",
        ),
        conflicts_with=(
            "get_short_interest",
            "query_finra",
        ),
        related_tools=(
            "describe_finra_dataset",
            "query_finra",
            "list_finra_datasets",
        ),
        prerequisites=(),
    ),
    "get_fundamentals": ToolDiscovery(
        domain="fundamentals",
        family="metrics",
        intent="retrieve_reported_metric",
        output_kind="metric_snapshot",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Single reported fundamental for one ticker: EPS, dividends, balance-sheet item, or shares outstanding.",
        choose_when=(
            "One specific reported historical numeric fundamental for one ticker: basic/diluted/TTM EPS, dividends, or shares outstanding.",
        ),
        reject_when=(
            "Do NOT use for full statements (get_financial_statements).",
            "Do NOT use for XBRL facts by concept (get_xbrl_facts).",
            "Do NOT use for cheap-vs-expensive multiples (get_valuation_metrics).",
            "Do NOT use for forward consensus (get_analyst_estimates).",
        ),
        conflicts_with=(
            "get_analyst_estimates",
            "get_financial_statements",
            "get_valuation_metrics",
            "get_xbrl_facts",
        ),
        related_tools=(
            "get_xbrl_facts",
            "get_financial_statements",
            "get_valuation_metrics",
            "get_analyst_estimates",
        ),
        prerequisites=(),
    ),
    "get_governance_events": ToolDiscovery(
        domain="governance",
        family="events",
        intent="retrieve_governance_events",
        output_kind="event_series",
        source="sec",
        entity_scope="single_security",
        time_mode="since_or_as_of",
        summary="Proxy and governance filing context (DEF 14A, meetings, votes) with retrieval pointers.",
        choose_when=("Finding shareholder-meeting, proxy-vote, or board-compensation records.",),
        reject_when=("Not for merger-deal status.",),
        conflicts_with=(),
        related_tools=("get_transaction_status",),
        prerequisites=(),
    ),
    "get_insider_activity": ToolDiscovery(
        domain="insider",
        family="trades",
        intent="retrieve_executed_insider_trades",
        output_kind="transaction_series",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Executed insider buys/sells for one ticker: actual purchases and sales from Forms 3/4/5.",
        choose_when=("One ticker's executed insider sales (buys/sells) by executives and directors (Forms 3/4/5).",),
        reject_when=("Do NOT use for planned but unexecuted Form 144 sales (get_planned_insider_sales).",),
        conflicts_with=("get_planned_insider_sales",),
        related_tools=(
            "get_planned_insider_sales",
            "get_beneficial_ownership",
        ),
        prerequisites=(),
    ),
    "get_macro_context": ToolDiscovery(
        domain="macro",
        family="statistics",
        intent="retrieve_macro_statistics",
        output_kind="statistic_series",
        source="datacommons",
        entity_scope="geography",
        time_mode="latest",
        summary="Macro statistics for a geography such as California: population (how many people live there), unemployment, inflation, GDP, rates.",
        choose_when=("Retrieving population, labor, inflation, GDP, or rate statistics for a geography.",),
        reject_when=(
            "Not for company-specific facts.",
            "Do NOT use for outside news, commentary, or why a stock moved (search_web).",
        ),
        conflicts_with=("search_web",),
        related_tools=("search_web",),
        prerequisites=(),
    ),
    "get_material_events": ToolDiscovery(
        domain="events",
        family="company-events",
        intent="retrieve_recent_material_events",
        output_kind="event_series",
        source="sec",
        entity_scope="single_security",
        time_mode="since_or_as_of",
        summary="Deterministic 8-K-derived recent event feed with accession citations for what changed since a date.",
        choose_when=("Finding what changed recently: recent 8-K-derived events for a company since a date.",),
        reject_when=(
            "Does not cover market reaction or news commentary.",
            "Answer from the event feed; do not open filing documents unless the question needs document text.",
            "Do NOT use for a full filing list by ticker or form (list_sec_filings).",
        ),
        conflicts_with=("list_sec_filings",),
        related_tools=(
            "get_sec_document",
            "search_web",
            "get_recent_ownership_filings",
            "list_sec_filings",
        ),
        prerequisites=(),
    ),
    "get_obligations": ToolDiscovery(
        domain="fundamentals",
        family="obligations",
        intent="retrieve_future_obligations",
        output_kind="obligation_schedule",
        source="sec",
        entity_scope="single_security",
        time_mode="latest",
        summary="Future cash obligations from 10-K/10-Q notes: amounts, horizons, certainty language.",
        choose_when=(
            "What a company is obligated to pay in the future for one ticker, including contracts and commitments.",
        ),
        reject_when=("Do NOT use for valuation multiples (get_valuation_metrics).",),
        conflicts_with=(),
        related_tools=(
            "get_valuation_metrics",
            "get_financial_statements",
        ),
        prerequisites=(),
    ),
    "get_offering_history": ToolDiscovery(
        domain="offerings",
        family="capital-raising",
        intent="retrieve_offering_history",
        output_kind="offering_series",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Offering history from S-1/S-3/424B filings: offering terms with source-registration links.",
        choose_when=(
            "Reviewing past offerings, shelf registrations, or IPO terms for a ticker, including share-count impact context for converts or warrants.",
        ),
        reject_when=("Not for dilution math (get_dilution_profile).",),
        conflicts_with=("get_dilution_profile",),
        related_tools=("get_dilution_profile",),
        prerequisites=(),
    ),
    "get_ownership_changes": ToolDiscovery(
        domain="ownership",
        family="stakes",
        intent="compare_ownership_stakes",
        output_kind="change_series",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Deterministic diffs between a holder's consecutive 13D/G filings: share and percent changes, changed stakes and positions.",
        choose_when=("Comparing consecutive 13D/G filings for changes in a holder's stake.",),
        reject_when=(
            "Not for the current snapshot of holders (get_beneficial_ownership).",
            "Not for relationship links in either direction (search_sec_relationships).",
        ),
        conflicts_with=(
            "get_beneficial_ownership",
            "search_sec_relationships",
        ),
        related_tools=(
            "get_beneficial_ownership",
            "search_sec_relationships",
        ),
        prerequisites=(),
    ),
    "get_planned_insider_sales": ToolDiscovery(
        domain="insider",
        family="trades",
        intent="retrieve_planned_insider_sales",
        output_kind="notice_series",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Planned Form 144 sale notices not yet executed: proposed insider sales for one ticker.",
        choose_when=("Proposed insider sales reported on Form 144 for one ticker.",),
        reject_when=("Do NOT use for completed insider trades (get_insider_activity).",),
        conflicts_with=("get_insider_activity",),
        related_tools=("get_insider_activity",),
        prerequisites=(),
    ),
    "get_recent_ownership_filings": ToolDiscovery(
        domain="events",
        family="ownership-filings",
        intent="retrieve_latest_ownership_filings",
        output_kind="filing_series",
        source="sec",
        entity_scope="market_wide",
        time_mode="latest",
        summary="Market-wide feed of the most recent SC 13D/13G filings from roughly the last 24 hours.",
        choose_when=(
            "Finding the latest market-wide SC 13D/G filings when no ticker is given; what just came out, newly filed.",
        ),
        reject_when=("Not for one company's current holders.",),
        conflicts_with=(),
        related_tools=(
            "get_beneficial_ownership",
            "get_material_events",
        ),
        prerequisites=(),
    ),
    "get_reg_sho_volume": ToolDiscovery(
        domain="finra",
        family="short-sale-volume",
        intent="daily_short_sale_volume",
        output_kind="daily_series",
        source="finra",
        entity_scope="single_security",
        time_mode="date_range_or_latest",
        summary="Self-contained daily short-sale volume by venue for one ticker: FINRA Reg SHO volume, rolling 12 months.",
        choose_when=("Daily short-sale volume or venue breakdowns for one ticker.",),
        reject_when=(
            "Do NOT use for biweekly short interest positions (get_short_interest).",
            "Do NOT call describe_finra_dataset or get_finra_datapoints; dataset and fields resolve internally.",
        ),
        conflicts_with=("get_short_interest",),
        related_tools=(
            "get_short_interest",
            "query_finra",
        ),
        prerequisites=(),
    ),
    "get_sec_document": ToolDiscovery(
        domain="sec",
        family="filing-catalog",
        intent="read_filing_document",
        output_kind="text_window",
        source="sec",
        entity_scope="single_document",
        time_mode="as_of",
        summary="Bounded text window of one filing document by accession number (Required: accession_no); page with cursor/limit via next_cursor.",
        choose_when=(
            "Reading a specific section such as MD&A or risk factors from a known accession.",
            "What the main document in a filing says; main-document text for a known accession.",
            'Example: accession_no="0000320193-25-000079", section="Risk Factors", cursor=0, limit=12000; next page with cursor=next_cursor.',
        ),
        reject_when=(
            "Do NOT use for filing metadata by accession (get_sec_filing).",
            "Do NOT use to list a filing's documents or exhibits (list_sec_documents).",
            "Not for what-changed questions.",
        ),
        conflicts_with=(
            "get_sec_filing",
            "list_sec_documents",
        ),
        related_tools=(
            "get_sec_filing",
            "get_material_events",
        ),
        prerequisites=(),
    ),
    "get_sec_filing": ToolDiscovery(
        domain="sec",
        family="filing-catalog",
        intent="retrieve_filing_metadata",
        output_kind="record",
        source="sec",
        entity_scope="single_filing",
        time_mode="as_of",
        summary="One filing's record by accession number: filer, form, dates, primary document, source URL.",
        choose_when=("Fetching filing metadata after its accession number is known.",),
        reject_when=(
            "Does not discover filings; list or search for the accession when unknown.",
            "Do NOT use for document text windows (get_sec_document).",
            "Do NOT use to list a filing's documents or exhibits (list_sec_documents).",
            "Does not search filing text; use full-text search when accession is unknown (search_sec_filings).",
        ),
        conflicts_with=(
            "get_sec_document",
            "list_sec_documents",
            "search_sec_filings",
        ),
        related_tools=(
            "list_sec_filings",
            "list_sec_documents",
            "search_sec_filings",
        ),
        prerequisites=(),
    ),
    "get_sec_search_coverage": ToolDiscovery(
        domain="sec",
        family="filing-search",
        intent="inspect_ingestion_coverage",
        output_kind="coverage_status",
        source="sec",
        entity_scope="dataset_partition",
        time_mode="current",
        summary="Persisted SEC ingestion coverage and backfill-job status for a form, source, or date partition.",
        choose_when=("Checking whether a form or date partition is covered or still queued before searching.",),
        reject_when=("Does not retrieve filing content.",),
        conflicts_with=(),
        related_tools=("search_sec_filings",),
        prerequisites=(),
    ),
    "get_short_interest": ToolDiscovery(
        domain="finra",
        family="short-interest",
        intent="current_reported_short_position",
        output_kind="current_snapshot",
        source="finra",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Biweekly short position for one ticker: FINRA short interest, days to cover, percent change.",
        choose_when=("One ticker's current short interest, short float, or days to cover.",),
        reject_when=(
            "Do NOT use for daily short-sale volume by venue (get_reg_sho_volume).",
            "Do NOT use for market-wide most-shorted screens (get_short_interest_leaderboard).",
            "Do NOT use for short-vs-shares context (get_short_pressure_profile).",
            "Do NOT use for exact source values (get_finra_datapoints).",
            "Do NOT use for analyzed briefings or trends over a dataset (query_finra).",
        ),
        conflicts_with=(
            "get_finra_datapoints",
            "get_reg_sho_volume",
            "get_short_pressure_profile",
            "query_finra",
            "get_short_interest_leaderboard",
        ),
        related_tools=(
            "query_finra",
            "get_finra_datapoints",
            "get_reg_sho_volume",
            "get_short_pressure_profile",
            "get_short_interest_leaderboard",
        ),
        prerequisites=(),
    ),
    "get_short_interest_leaderboard": ToolDiscovery(
        domain="finra",
        family="screens",
        intent="rank_short_interest",
        output_kind="leaderboard",
        source="finra_sec",
        entity_scope="market_wide",
        time_mode="latest_or_as_of",
        summary="Market-wide most-shorted screen: ranked stocks by short interest as a percent of SEC shares.",
        choose_when=("Screening which stocks are the most shorted across the market.",),
        reject_when=("Do NOT use for one ticker's short interest (get_short_interest).",),
        conflicts_with=("get_short_interest",),
        related_tools=("get_short_interest",),
        prerequisites=(),
    ),
    "get_short_pressure_profile": ToolDiscovery(
        domain="finra",
        family="short-interest",
        intent="assess_short_pressure",
        output_kind="derived_composite",
        source="finra_sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="Short pressure vs shares outstanding for one ticker: FINRA positioning plus SEC shares and ratio.",
        choose_when=("Short positioning relative to shares outstanding for one ticker.",),
        reject_when=(
            "Do NOT use for biweekly short position alone (get_short_interest).",
            "Do NOT use for daily short-sale volume (get_reg_sho_volume).",
        ),
        conflicts_with=("get_short_interest",),
        related_tools=(
            "get_short_interest",
            "query_finra",
            "get_reg_sho_volume",
        ),
        prerequisites=(),
    ),
    "get_sp500_weight": ToolDiscovery(
        domain="market",
        family="index-membership",
        intent="retrieve_sp500_weight",
        output_kind="current_snapshot",
        source="slickcharts",
        entity_scope="single_security",
        time_mode="latest",
        summary="A company's current weight and rank in the S&P 500 index from the constituent list.",
        choose_when=("Answering what percent of the S&P 500 a ticker represents.",),
        reject_when=("Do not use for valuation or short-positioning questions.",),
        conflicts_with=(),
        related_tools=("get_analyst_estimates",),
        prerequisites=(),
    ),
    "get_threshold_securities": ToolDiscovery(
        domain="finra",
        family="threshold-securities",
        intent="retrieve_threshold_status",
        output_kind="status_series",
        source="finra",
        entity_scope="single_security_or_market",
        time_mode="date_or_latest",
        summary="FINRA OTC Regulation SHO threshold securities, optionally filtered by ticker and date.",
        choose_when=("Checking whether securities appear on the Reg SHO threshold list.",),
        reject_when=("Not for ordinary short interest levels.",),
        conflicts_with=(),
        related_tools=("get_short_interest",),
        prerequisites=(),
    ),
    "get_transaction_status": ToolDiscovery(
        domain="transactions",
        family="deals",
        intent="retrieve_transaction_status",
        output_kind="event_series",
        source="sec",
        entity_scope="single_security",
        time_mode="latest_or_as_of",
        summary="M&A filing context: tender offers, 14D-9 recommendations, S-4s, and merger proxies.",
        choose_when=("Checking merger, acquisition, or tender-offer filing context for a ticker.",),
        reject_when=("Not for governance or proxy votes.",),
        conflicts_with=(),
        related_tools=(
            "get_governance_events",
            "get_sec_document",
        ),
        prerequisites=(),
    ),
    "get_trend_evidence": ToolDiscovery(
        domain="alternative",
        family="trend-evidence",
        intent="retrieve_known_trend_evidence",
        output_kind="evidence_series",
        source="google_trends",
        entity_scope="single_topic",
        time_mode="date_range",
        summary="Evidence for one known trend: search interest, rising queries, and geography.",
        choose_when=("Backing a known trend claim with dated, geography-specific search-interest evidence.",),
        reject_when=("Not for discovering new signals.",),
        conflicts_with=(),
        related_tools=("find_alternative_signals",),
        prerequisites=(),
    ),
    "get_valuation_metrics": ToolDiscovery(
        domain="valuation",
        family="multiples",
        intent="calculate_valuation_multiples",
        output_kind="derived_snapshot",
        source="yahoo_finance_sec",
        entity_scope="single_security",
        time_mode="latest",
        summary="Cheap-vs-expensive earnings multiples at live price: trailing plus forward P/E.",
        choose_when=("Whether a company is cheap or expensive on earnings multiples for one ticker.",),
        reject_when=(
            "Do NOT use for reported EPS alone (get_fundamentals).",
            "Do NOT use for forward consensus alone (get_analyst_estimates).",
        ),
        conflicts_with=(
            "get_analyst_estimates",
            "get_fundamentals",
        ),
        related_tools=(
            "get_analyst_estimates",
            "get_obligations",
            "get_fundamentals",
        ),
        prerequisites=(),
    ),
    "get_xbrl_facts": ToolDiscovery(
        domain="fundamentals",
        family="metrics",
        intent="retrieve_xbrl_concept",
        output_kind="fact_records",
        source="sec",
        entity_scope="single_security",
        time_mode="latest",
        summary="Single XBRL-tagged fact by concept name: revenue, net income, cash, debt, or equity.",
        choose_when=("One tagged line-item value by exact XBRL concept name.",),
        reject_when=(
            "Do NOT use for EPS (get_fundamentals).",
            "Do NOT use for full statements (get_financial_statements).",
        ),
        conflicts_with=(
            "get_financial_statements",
            "get_fundamentals",
        ),
        related_tools=(
            "get_fundamentals",
            "get_financial_statements",
        ),
        prerequisites=(),
    ),
    "investigate_social_arbitrage_candidate": ToolDiscovery(
        domain="alternative",
        family="social-arbitrage",
        intent="assess_attention_demand_gap",
        output_kind="derived_analysis",
        source="google_trends",
        entity_scope="single_topic",
        time_mode="latest_or_as_of",
        summary="Enrichment of one social-arbitrage candidate with corroboration and exposure gap. Social signals vetting.",
        choose_when=("Testing whether online attention around one candidate corresponds to real demand.",),
        reject_when=("Not for broad signal discovery.",),
        conflicts_with=(),
        related_tools=(
            "find_alternative_signals",
            "get_trend_evidence",
        ),
        prerequisites=(),
    ),
    "list_finra_datasets": ToolDiscovery(
        domain="finra",
        family="catalog",
        intent="discover_finra_dataset",
        output_kind="catalog",
        source="finra",
        entity_scope="dataset_catalog",
        time_mode="current",
        summary="Catalog of public FINRA datasets with canonical ids, groups, and ticker/date support.",
        choose_when=("Finding which FINRA dataset covers a question before querying.",),
        reject_when=("Does not return dataset fields or schemas (describe_finra_dataset).",),
        conflicts_with=("describe_finra_dataset",),
        related_tools=("describe_finra_dataset",),
        prerequisites=(),
    ),
    "list_sec_documents": ToolDiscovery(
        domain="sec",
        family="filing-catalog",
        intent="list_filing_documents",
        output_kind="record_series",
        source="sec",
        entity_scope="single_filing",
        time_mode="as_of",
        summary="Index of documents and exhibits attached to one filing, looked up by accession number.",
        choose_when=("Listing documents and exhibits attached to a known filing accession.",),
        reject_when=(
            "Do NOT use for document text windows (get_sec_document).",
            "Do NOT use for filing metadata records (get_sec_filing).",
        ),
        conflicts_with=(
            "get_sec_document",
            "get_sec_filing",
        ),
        related_tools=(
            "get_sec_filing",
            "get_sec_document",
            "list_sec_filings",
        ),
        prerequisites=(),
    ),
    "list_sec_filings": ToolDiscovery(
        domain="sec",
        family="filing-catalog",
        intent="list_entity_filings",
        output_kind="filing_series",
        source="sec",
        entity_scope="single_entity",
        time_mode="date_range_or_as_of",
        summary='List EDGAR filings for an exact ticker or CIK (Required: identifier, e.g. identifier="AAPL"); filterable by form and date range.',
        choose_when=(
            "Listing what a company filed lately; recent filings for an exact ticker or CIK, optionally filtered by form or date.",
            'Required identifier (ticker or CIK, e.g. identifier="AAPL"); optional forms, start_date, end_date, as_of, limit.',
        ),
        reject_when=(
            "Do not guess an identifier from a bare company name; use the exact ticker when known, otherwise resolve the company's exact identifier first.",
            "Do NOT use for disclosure search without known identifier (search_sec_filings).",
            "Do NOT use for 8-K-derived what-changed event feed since a date (get_material_events).",
        ),
        conflicts_with=(
            "search_sec_filings",
            "get_material_events",
        ),
        related_tools=(
            "get_sec_filing",
            "search_sec_filings",
            "find_sec_entities",
            "get_material_events",
        ),
        prerequisites=(),
    ),
    "query_finra": ToolDiscovery(
        domain="finra",
        family="short-interest",
        intent="analyze_historical_finra_records",
        output_kind="distribution_or_trend",
        source="finra",
        entity_scope="single_dataset",
        time_mode="date_range",
        summary="Analyzed FINRA briefing with trends and metrics over any named dataset, no raw rows.",
        choose_when=("Analyzing a FINRA dataset's coverage, distribution, and changes over time.",),
        reject_when=(
            "Do NOT use for exact source values (get_finra_datapoints).",
            "Do NOT use for one ticker's current short position (get_short_interest).",
        ),
        conflicts_with=(
            "get_finra_datapoints",
            "get_short_interest",
        ),
        related_tools=(
            "describe_finra_dataset",
            "get_finra_datapoints",
            "get_short_interest",
            "list_finra_datasets",
        ),
        prerequisites=(),
    ),
    "search_company_patents": ToolDiscovery(
        domain="patents",
        family="search",
        intent="search_company_patents",
        output_kind="patent_records",
        source="google_patents",
        entity_scope="single_company",
        time_mode="date_range_or_latest",
        summary="Company patent search: publications, assignees, counts, and classifications.",
        choose_when=(
            "Finding patents a company filed or patented lately, with publication counts and classifications.",
        ),
        reject_when=(
            "Not for financial or filing questions.",
            "Answer from patent records.",
        ),
        conflicts_with=(),
        related_tools=(),
        prerequisites=(),
    ),
    "search_sec_filings": ToolDiscovery(
        domain="sec",
        family="filing-search",
        intent="search_filing_text",
        output_kind="search_results",
        source="sec",
        entity_scope="multi_entity",
        time_mode="date_range_or_as_of",
        summary="General EDGAR full-text disclosure search across entity, EFTS, and 10-K/10-Q routes, with mentions.",
        choose_when=(
            "Searching disclosed filing text, risk-factor language, and mentions when the accession number is unknown.",
            "SEC filings or filing full-text search when accession is unknown.",
            'Required: at least one of query, ticker, cik, company_name, person_name, domain, accession_no, security_identifier; e.g. query="risk factors", ticker="AAPL".',
        ),
        reject_when=(
            "Not a filing lister for a known ticker (list_sec_filings).",
            "Do NOT use for year-over-year risk-factor changes (diff_risk_factors).",
            "Do NOT use for full-filing diffs between accessions (diff_sec_filings).",
            "Do NOT use for one filing metadata record by accession (get_sec_filing).",
            "Not for the quick bounded lookup (search_sec_filings_bounded).",
        ),
        conflicts_with=(
            "diff_risk_factors",
            "diff_sec_filings",
            "list_sec_filings",
            "get_sec_filing",
            "search_sec_filings_bounded",
        ),
        related_tools=(
            "search_sec_filings_bounded",
            "list_sec_filings",
            "get_sec_filing",
            "find_sec_entities",
            "diff_risk_factors",
        ),
        prerequisites=(),
    ),
    "search_sec_filings_bounded": ToolDiscovery(
        domain="sec",
        family="filing-search",
        intent="search_filing_text_bounded",
        output_kind="search_results",
        source="sec",
        entity_scope="multi_entity",
        time_mode="date_range_or_as_of",
        summary="Quick bounded EDGAR lookup: same filing-text search, fast routes only, capped at limit.",
        choose_when=(
            "One mention check when full coverage is not needed.",
            'Required: at least one of query, ticker, cik, company_name, person_name, domain, accession_no, security_identifier; e.g. query="risk factors", ticker="AAPL".',
        ),
        reject_when=(
            "Not for all/every-mention or full-coverage questions (search_sec_filings).",
            "Not a filing lister for a known ticker (list_sec_filings).",
        ),
        conflicts_with=("search_sec_filings",),
        related_tools=(
            "search_sec_filings",
            "list_sec_filings",
        ),
        prerequisites=(),
    ),
    "search_sec_relationships": ToolDiscovery(
        domain="ownership",
        family="stakes",
        intent="search_ownership_relationships",
        output_kind="relationship_records",
        source="sec",
        entity_scope="single_entity",
        time_mode="latest_or_as_of",
        summary="Ownership and transaction relationship links for an entity: 13D/G owners, 13F holdings, deal links.",
        choose_when=("Mapping who owns, holds, or transacts with an entity in either direction.",),
        reject_when=(
            "Not for current 5%+ stake sizes (get_beneficial_ownership).",
            "Not for consecutive-filing stake diffs (get_ownership_changes).",
        ),
        conflicts_with=(
            "get_beneficial_ownership",
            "get_ownership_changes",
        ),
        related_tools=(
            "get_beneficial_ownership",
            "get_ownership_changes",
        ),
    ),
    "search_web": ToolDiscovery(
        domain="web",
        family="search",
        intent="search_external_web",
        output_kind="search_results",
        source="exa",
        entity_scope="open_query",
        time_mode="latest",
        summary="External web news and commentary for price moves, headlines, and industry developments: what outside commentators and people are saying, business risks.",
        choose_when=(
            "Finding recent news, announcements, catalysts, market reaction, why a stock moved/rose/fell, or recent commentary outside structured sources.",
        ),
        reject_when=(
            "Not for FINRA short data.",
            "Do NOT use for bounded geography statistics like population or rates (get_macro_context).",
        ),
        conflicts_with=("get_macro_context",),
        related_tools=(
            "get_material_events",
            "query_finra",
            "get_macro_context",
        ),
        prerequisites=(),
    ),
    "thesis_create": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="create_thesis",
        output_kind="governed_action",
        source="local",
        entity_scope="single_thesis",
        time_mode="current",
        summary="Start a new investment thesis proposal with scope, claims, and open questions.",
        choose_when=("Creating a new investment thesis to track and test.",),
        reject_when=("Not for reading an existing thesis.",),
        conflicts_with=(),
        related_tools=(
            "thesis_show",
            "thesis_refine",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "thesis_journal": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="append_thesis_journal",
        output_kind="governed_action",
        source="local",
        entity_scope="single_thesis",
        time_mode="current",
        summary="Append an operator note or journal entry to a thesis log. Pass the thesis ID as thesis:<uuid>.",
        choose_when=("Appending an operator note about ongoing monitoring without creating or changing a watch rule.",),
        reject_when=(
            "Not for revising thesis claims or deltas (thesis_refine).",
            "Not for setting alerts (thesis_watch).",
        ),
        conflicts_with=(
            "thesis_watch",
            "thesis_refine",
        ),
        related_tools=("thesis_refine",),
        prerequisites=(),
        direct_activation=False,
    ),
    "thesis_status": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="change_thesis_status",
        output_kind="governed_action",
        source="local",
        entity_scope="single_thesis",
        time_mode="current",
        summary="Pause, resume, or close thesis monitoring. Pass the thesis ID as thesis:<uuid>.",
        choose_when=("Pausing monitoring without deleting rules, resuming a paused thesis, or closing a thesis.",),
        reject_when=(
            "Not for reading thesis state (thesis_show).",
            "Not for editing claims or rules (thesis_refine, thesis_watch).",
        ),
        conflicts_with=(),
        related_tools=("thesis_show",),
        prerequisites=(),
        direct_activation=False,
    ),
    "thesis_refine": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="refine_thesis",
        output_kind="governed_action",
        source="local",
        entity_scope="single_thesis",
        time_mode="current",
        summary="Update a thesis with clarifications and deltas. Pass the thesis ID as thesis:<uuid>.",
        choose_when=("Revising a thesis after new evidence or feedback.",),
        reject_when=("Not for routine operator notes without changing claims (thesis_journal).",),
        conflicts_with=("thesis_journal",),
        related_tools=(
            "thesis_show",
            "thesis_journal",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "thesis_show": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="retrieve_thesis",
        output_kind="current_snapshot",
        source="local",
        entity_scope="single_thesis",
        time_mode="latest_or_as_of",
        summary="Read a thesis: its status, assessment, and current state. Pass the thesis ID as thesis:<uuid>.",
        choose_when=("Checking a thesis and its current assessment.",),
        reject_when=(
            "Not for changing a thesis.",
            "Not for listing or adding monitoring rules and alerts (thesis_watch).",
        ),
        conflicts_with=("thesis_watch",),
        related_tools=(
            "thesis_create",
            "thesis_refine",
            "thesis_journal",
            "thesis_watch",
        ),
        prerequisites=(),
    ),
    "thesis_watch": ToolDiscovery(
        domain="thesis",
        family="lifecycle",
        intent="list_or_add_thesis_watch",
        output_kind="governed_action",
        source="local",
        entity_scope="single_thesis",
        time_mode="current",
        summary="List existing watch rules, or add a validated monitoring rule that alerts when a thesis condition triggers.",
        choose_when=(
            "Listing what is watched for a thesis, or setting an alert on an invalidator or trigger; what am I watching for, watch rules.",
        ),
        reject_when=(
            "Not for logging notes (thesis_journal).",
            "Not for reading thesis status and assessment (thesis_show).",
        ),
        conflicts_with=(
            "thesis_journal",
            "thesis_show",
        ),
        related_tools=("thesis_show",),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_start": ToolDiscovery(
        domain="research",
        family="session",
        intent="start_research",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Start a research session for a question; returns the session ID with its first job and next action.",
        choose_when=("Starting research on a new question.",),
        reject_when=(
            "Not for checking session state (research_status).",
            "Not for cancelling a session (research_cancel).",
        ),
        conflicts_with=(
            "research_status",
            "research_cancel",
        ),
        related_tools=(
            "research_status",
            "research_cancel",
            "research_resume",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_resume": ToolDiscovery(
        domain="research",
        family="session",
        intent="resume_research",
        output_kind="current_snapshot",
        source="local",
        entity_scope="single_session",
        time_mode="latest_or_as_of",
        summary="Resume a research session: read-only snapshot with wave, budgets, open jobs, and next action.",
        choose_when=("Resuming or re-entering an existing research session.",),
        reject_when=("Not for checking jobs and next action without wave state (research_status).",),
        conflicts_with=("research_status",),
        related_tools=(
            "research_status",
            "research_start",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_status": ToolDiscovery(
        domain="research",
        family="session",
        intent="inspect_research",
        output_kind="current_snapshot",
        source="local",
        entity_scope="single_session",
        time_mode="latest_or_as_of",
        summary="Read a research session with its jobs and pending next action.",
        choose_when=("Checking a research session and its current state.",),
        reject_when=(
            "Not for starting a session (research_start).",
            "Not for resuming wave and budget state (research_resume).",
        ),
        conflicts_with=(
            "research_start",
            "research_resume",
        ),
        related_tools=(
            "research_start",
            "research_resume",
            "research_read",
            "research_cancel",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_cancel": ToolDiscovery(
        domain="research",
        family="session",
        intent="cancel_research",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Cancel a research session; terminal sessions return current state.",
        choose_when=("Stopping a research session that is no longer needed.",),
        reject_when=("Not for starting a session (research_start).",),
        conflicts_with=("research_start",),
        related_tools=(
            "research_start",
            "research_status",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_read": ToolDiscovery(
        domain="research",
        family="session",
        intent="read_research_resource",
        output_kind="current_snapshot",
        source="local",
        entity_scope="single_session",
        time_mode="latest_or_as_of",
        summary="Read one persisted research resource: evidence, freeze, dossier, job, or session record.",
        choose_when=("Reading a single evidence, freeze, dossier, job, or session record.",),
        reject_when=("Not for session state overviews (research_status).",),
        conflicts_with=(),
        related_tools=("research_status",),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_read_search": ToolDiscovery(
        domain="research",
        family="session",
        intent="read_search_hits",
        output_kind="current_snapshot",
        source="local",
        entity_scope="single_session",
        time_mode="latest_or_as_of",
        summary="Page the persisted ranked hit universe of one SEC search, with retrieval truth.",
        choose_when=(
            "Reading hits beyond the compact top_hits/additional_hits packet of a search_sec_filings result.",
            "Paging a large search to exhaustion instead of rerunning the search with a higher limit.",
        ),
        reject_when=("Not for running a new SEC search (search_sec_filings).",),
        conflicts_with=(),
        related_tools=(
            "research_read",
            "research_add_evidence",
        ),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_add_evidence": ToolDiscovery(
        domain="research",
        family="session",
        intent="add_research_evidence",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Record one finding on a research job; provenance, point-in-time, and IDs are kernel-validated.",
        choose_when=("Recording a finding from a dispatched research job.",),
        reject_when=("Not for session state overviews (research_status).",),
        conflicts_with=(),
        related_tools=("research_status",),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_submit_source_result": ToolDiscovery(
        domain="research",
        family="session",
        intent="submit_source_result",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Complete one running source job with validated coverage; evidence stays mutation-only.",
        choose_when=(
            "Completing a source investigation with validated coverage.",
            "Sufficient coverage means major EDGAR-visible channels and counterparties investigated with no material open questions or branches — never just N evidence rows.",
        ),
        reject_when=("Not for session state overviews (research_status).",),
        conflicts_with=(),
        related_tools=("research_status",),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_add_analysis": ToolDiscovery(
        domain="research",
        family="session",
        intent="add_committee_analysis",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Record one committee analysis (stockbot, bullbot, or bearbot); claim refs are validated against frozen evidence.",
        choose_when=("Recording a trio analysis grounded in the frozen evidence.",),
        reject_when=("Not for session state overviews (research_status).",),
        conflicts_with=(),
        related_tools=("research_status",),
        prerequisites=(),
        direct_activation=False,
    ),
    "research_finalize": ToolDiscovery(
        domain="research",
        family="session",
        intent="finalize_research",
        output_kind="governed_action",
        source="local",
        entity_scope="single_session",
        time_mode="current",
        summary="Persist the trio-joined synthesis and complete a research session with frozen-evidence claims.",
        choose_when=(
            "Finalizing a trio-complete research session with grounded claims.",
            "Delivering the substantive structured answer in the same turn — Bottom line through evidence refs plus the searched-source scope line; a bare finalized-status note is not a completion.",
        ),
        reject_when=("Not for session state overviews (research_status).",),
        conflicts_with=(),
        related_tools=("research_status",),
        prerequisites=(),
        direct_activation=False,
    ),
    "get_current_time": ToolDiscovery(
        domain="time",
        family="clock",
        intent="retrieve_current_time",
        output_kind="current_snapshot",
        source="system",
        entity_scope="global",
        time_mode="current",
        summary="Current UTC date and time from the system clock.",
        choose_when=("What time is it now; the current UTC date and time.",),
        reject_when=("Not for market data, filings, or historical point-in-time facts.",),
        conflicts_with=(),
        related_tools=(),
        prerequisites=(),
    ),
}


_DISCOVERY_TEXT_LIMIT = 200


_SLUG_KEBAB_RE = __import__("re").compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
_SLUG_SNAKE_RE = __import__("re").compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _check_discovery_domain(name: str, meta: ToolDiscovery, known_domains: set[str]) -> None:
    """Registry entry lives in a known domain."""
    if meta.domain not in known_domains:
        raise AssertionError(f"tool discovery {name!r} has unknown domain {meta.domain!r}")


def _check_discovery_slugs(name: str, meta: ToolDiscovery) -> None:
    """Domain/family/source are kebab slugs; intent/output/scope/mode are snake."""
    for field_name in ("domain", "family", "source"):
        value = getattr(meta, field_name)
        if not value or not _SLUG_KEBAB_RE.match(value):
            raise AssertionError(f"tool discovery {name!r} has non-slug {field_name} {value!r}")
    for field_name in ("intent", "output_kind", "entity_scope", "time_mode"):
        value = getattr(meta, field_name)
        if not value or not _SLUG_SNAKE_RE.match(value):
            raise AssertionError(f"tool discovery {name!r} has non-snake {field_name} {value!r}")


def _check_discovery_texts(name: str, meta: ToolDiscovery) -> None:
    """Summary plus every bullet bounded and non-empty; both bullets required."""
    if not meta.summary or len(meta.summary) > _DISCOVERY_TEXT_LIMIT:
        raise AssertionError(f"tool discovery {name!r} has empty/overlong summary")
    if not meta.choose_when or not meta.reject_when:
        raise AssertionError(f"tool discovery {name!r} needs >=1 choose_when and >=1 reject_when")
    for bullet in (
        *meta.choose_when,
        *meta.reject_when,
        *meta.related_tools,
        *meta.prerequisites,
        *meta.conflicts_with,
    ):
        if not bullet or len(bullet) > _DISCOVERY_TEXT_LIMIT:
            raise AssertionError(f"tool discovery {name!r} has empty/overlong bullet {bullet!r}")


def _check_discovery_refs(name: str, meta: ToolDiscovery) -> None:
    """Related/prerequisite tools exist in the registry."""
    for ref in (*meta.related_tools, *meta.prerequisites):
        if ref not in TOOL_DISCOVERY_REGISTRY:
            raise AssertionError(f"tool discovery {name!r} references unknown tool {ref!r}")


def _check_discovery_peer(name: str, meta: ToolDiscovery, peer: str) -> None:
    """One conflict edge: known peer, reciprocated, and named in reject_when."""
    if peer not in TOOL_DISCOVERY_REGISTRY:
        raise AssertionError(f"tool discovery {name!r} conflicts with unknown tool {peer!r}")
    if name not in TOOL_DISCOVERY_REGISTRY[peer].conflicts_with:
        raise AssertionError(
            f"tool discovery {name!r} conflicts with {peer!r} but {peer!r} does not reciprocate {name!r}"
        )
    if peer not in " ".join(meta.reject_when):
        raise AssertionError(f"tool discovery {name!r} conflicts with {peer!r} but never names {peer!r} in reject_when")


def _check_discovery_conflicts(name: str, meta: ToolDiscovery) -> None:
    """No self/duplicate conflicts; every edge reciprocated and named."""
    if name in meta.conflicts_with:
        raise AssertionError(f"tool discovery {name!r} self-conflicts with {name!r}")
    if len(set(meta.conflicts_with)) != len(meta.conflicts_with):
        raise AssertionError(f"tool discovery {name!r} has duplicate conflicts_with entry")
    for peer in meta.conflicts_with:
        _check_discovery_peer(name, meta, peer)


def _check_discovery_entry(name: str, meta: ToolDiscovery, known_domains: set[str]) -> None:
    """All per-entry gates for one registry row."""
    _check_discovery_domain(name, meta, known_domains)
    _check_discovery_slugs(name, meta)
    _check_discovery_texts(name, meta)
    _check_discovery_refs(name, meta)
    _check_discovery_conflicts(name, meta)


def _uncovered_research_tools(raw_caps: dict[object, object]) -> list[str]:
    """RESEARCH-capability tools missing from the discovery registry."""
    return sorted(
        str(tool)
        for tool, cap in raw_caps.items()
        if cap is Capability.RESEARCH
        and str(tool) not in {"search_tools", "list_tool_domains", "describe_tool", "browse_tools", "call_tool"}
        and str(tool) not in TOOL_DISCOVERY_REGISTRY
    )


def _check_discovery_activation(raw_caps: dict[object, object]) -> None:
    """Direct-activatable rows are RESEARCH-capability tools."""
    for name, meta in TOOL_DISCOVERY_REGISTRY.items():
        if meta.direct_activation and raw_caps.get(name) is not Capability.RESEARCH:
            raise AssertionError(f"tool discovery {name!r} is direct-activatable but not Capability.RESEARCH")


def _check_discovery_capabilities() -> None:
    """Registry covers every RESEARCH tool; activation matches capability."""
    raw_caps = globals().get("TOOL_CAPABILITIES")
    if not isinstance(raw_caps, dict):
        return
    uncovered = _uncovered_research_tools(raw_caps)
    if uncovered:
        raise AssertionError(f"tool discovery registry missing RESEARCH tools: {uncovered}")
    _check_discovery_activation(raw_caps)


def validate_tool_discovery_registry() -> dict[str, ToolDiscovery]:
    """Fail loudly on registry drift; returns the registry for verify scripts."""
    known_domains = set(DOMAIN_DESCRIPTIONS)
    for name in sorted(TOOL_DISCOVERY_REGISTRY):
        _check_discovery_entry(name, TOOL_DISCOVERY_REGISTRY[name], known_domains)
    _check_discovery_capabilities()
    return TOOL_DISCOVERY_REGISTRY


# Content-derived registry version for observability records (schemas + routing metadata).
_TOOL_DISCOVERY_FINGERPRINT = json.dumps(
    {
        name: [
            meta.domain,
            meta.family,
            meta.intent,
            meta.output_kind,
            meta.source,
            meta.entity_scope,
            meta.time_mode,
            meta.summary,
            list(meta.choose_when),
            list(meta.reject_when),
            sorted(meta.conflicts_with),
            sorted(meta.related_tools),
            sorted(meta.prerequisites),
            meta.direct_activation,
        ]
        for name, meta in sorted(TOOL_DISCOVERY_REGISTRY.items())
    },
    sort_keys=True,
)
TOOL_REGISTRY_VERSION = hashlib.sha256(
    (json.dumps(TOOLS, sort_keys=True) + _TOOL_DISCOVERY_FINGERPRINT).encode()
).hexdigest()[:12]


validate_tool_discovery_registry()


def dynamically_activatable_tool_names() -> list[str]:
    """Sorted canonical names eligible for direct small-model activation."""
    return sorted(name for name, meta in validate_tool_discovery_registry().items() if meta.direct_activation)


def _routing_card(name: str) -> dict[str, object]:
    """Compact routing card: registry metadata plus canonical arg lists, never full parameters."""
    meta = TOOL_DISCOVERY_REGISTRY[name]
    _, required, optional = _canonical_tool_schema(name)
    return {
        "name": name,
        "domain": meta.domain,
        "family": meta.family,
        "summary": meta.summary,
        "intent": meta.intent,
        "output_kind": meta.output_kind,
        "source": meta.source,
        "entity_scope": meta.entity_scope,
        "time_mode": meta.time_mode,
        "choose_when": list(meta.choose_when),
        "reject_when": list(meta.reject_when),
        "required": required,
        "optional": optional,
    }


def build_prerequisite_graph_from_tool_metadata() -> dict[str, frozenset[str]]:
    """Direct prerequisite edges from the registry (no transitive expansion)."""
    return {name: frozenset(meta.prerequisites) for name, meta in TOOL_DISCOVERY_REGISTRY.items() if meta.prerequisites}


def _normalize_discovery_text(value: str) -> list[str]:
    """Lowercase, de-punctuate, and de-pluralize discovery text into tokens."""
    lowered = (
        value.lower()
        .replace("p/e", "pe")
        .replace("13-d", "13d")
        .replace("contractual", "contract")
        .replace("trended", "trend")
        .replace("changed", "change")
    )
    cleaned = "".join(c if c.isalnum() or c == " " else " " for c in lowered)
    raw = " ".join(cleaned.split()).split()
    merged: list[str] = []
    skip = False
    for i, token in enumerate(raw):
        if skip:
            skip = False
            continue
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        pair = (token, nxt.rstrip("s"))
        if token in ("10", "8") and nxt and pair in (("10", "k"), ("10", "q"), ("8", "k")):
            merged.append(pair[0] + pair[1])
            skip = True
        else:
            merged.append(token)
    tokens: list[str] = []
    for token in merged:
        if len(token) > 3:
            if token.endswith("ies"):
                token = token[:-3] + "y"
            elif token.endswith("s"):
                token = token[:-1]
        tokens.append(token)
    return tokens


# Generic stopwords for catalog search (standard filler, never domain/intent terms).
_DISCOVERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "with",
        "show",
        "tell",
        "give",
    }
)


def _discovery_keywords(value: str) -> set[str]:
    """Normalized discovery tokens minus stopwords and single characters."""
    return {t for t in _normalize_discovery_text(value) if len(t) > 1 and t not in _DISCOVERY_STOPWORDS}


def _find_root(parent: dict[str, str], name: str) -> str:
    """Union-find root with path halving over the ranked-name forest."""
    while parent[name] != name:
        parent[name] = parent[parent[name]]
        name = parent[name]
    return name


def _union_names(parent: dict[str, str], first: str, second: str) -> None:
    """Union two ranked names by root (no-op when already joined)."""
    first_root, second_root = _find_root(parent, first), _find_root(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root


def _conflict_components(ranked_names: list[str]) -> list[list[str]]:
    """Connected components of conflicts_with among the ranked names."""
    parent: dict[str, str] = {n: n for n in ranked_names}
    present = set(ranked_names)
    for name in ranked_names:
        for peer in TOOL_DISCOVERY_REGISTRY[name].conflicts_with:
            if peer in present:
                _union_names(parent, name, peer)
    comps: dict[str, list[str]] = {}
    for name in ranked_names:
        comps.setdefault(_find_root(parent, name), []).append(name)
    return list(comps.values())


def _is_ambiguity_group(component: list[str]) -> bool:
    """A real ambiguity group: 2+ names with at least one internal conflict edge."""
    if len(component) < 2:
        return False
    return any(any(peer in component for peer in TOOL_DISCOVERY_REGISTRY[name].conflicts_with) for name in component)


def _choose_bit(name: str) -> str:
    """One candidate's distinguishing bit: name plus its primary choose_when."""
    return (
        f"{name} \u2014 {TOOL_DISCOVERY_REGISTRY[name].choose_when[0]}"
        if TOOL_DISCOVERY_REGISTRY[name].choose_when
        else name
    )


def _ambiguity_card(component: list[str], index: dict[str, int]) -> dict[str, object]:
    """Ranked-order candidates, domain/family paths, and the distinguishing question."""

    def _order_key(n: str) -> int:
        return index[n]

    ordered = sorted(component, key=_order_key)
    paths = sorted({f"{TOOL_DISCOVERY_REGISTRY[n].domain}/{TOOL_DISCOVERY_REGISTRY[n].family}" for n in ordered})
    bits = [_choose_bit(n) for n in ordered]
    return {
        "candidates": ordered,
        "paths": paths,
        "distinguishing_question": "Which outcome do you need: " + "; ".join(bits) + "?",
    }


def _group_sort_key(group: dict[str, object]) -> str:
    """First candidate name (groups are non-empty by construction)."""
    cands = group.get("candidates")
    if isinstance(cands, list) and cands and isinstance(cands[0], str):
        return cands[0]
    return ""


def _ambiguity_groups(ranked_names: list[str]) -> tuple[bool, list[dict[str, object]]]:
    """Connected components of conflicts_with among returned matches."""
    if len(ranked_names) < 2:
        return False, []
    index = {name: i for i, name in enumerate(ranked_names)}
    groups = [_ambiguity_card(comp, index) for comp in _conflict_components(ranked_names) if _is_ambiguity_group(comp)]
    groups.sort(key=_group_sort_key)
    return (len(groups) > 0), groups


_SEARCH_EMPTY_HINT = "No matches. Retry with the full user question as the query (never a ticker or one word); omit domain unless certain of it."


def _search_params(args: dict[str, object]) -> tuple[str, str | None]:
    """Normalized query plus validated domain filter (unknown domains ignored)."""
    query = str(args.get("query") or "")
    domain = str(args.get("domain") or "").strip().lower() or None
    if domain is not None and domain not in DOMAIN_DESCRIPTIONS:
        domain = None
    return query, domain


def _empty_search_result() -> dict[str, object]:
    """Zero-match envelope with the retry hint."""
    return {"matches": [], "count": 0, "ambiguous": False, "ambiguity_groups": [], "hint": _SEARCH_EMPTY_HINT}


def _tool_name_bonus(name: str, query_norm: str) -> int:
    """Exact tool-name match bonus (10)."""
    if query_norm and query_norm == " ".join(_normalize_discovery_text(name.replace("_", " "))):
        return 10
    return 0


def _tool_phrase_bonus(meta: ToolDiscovery, query_norm: str) -> int:
    """Exact phrase in summary/choose_when bonus (5, once)."""
    for text in (meta.summary, *meta.choose_when):
        phrase = " ".join(_normalize_discovery_text(text))
        if query_norm and phrase and (query_norm in phrase or phrase in query_norm):
            return 5
    return 0


def _tool_domain_bonus(meta: ToolDiscovery, query_tokens: set[str]) -> int:
    """Domain-name overlap bonus (1)."""
    if query_tokens and set(_normalize_discovery_text(meta.domain)) <= query_tokens:
        return 1
    return 0


def _tool_field_tokens(name: str, meta: ToolDiscovery) -> set[str]:
    """Scorable token set: name/domain/family/intent/output/summary/choose_when."""
    return _discovery_keywords(
        " ".join(
            (
                name.replace("_", " "),
                meta.domain,
                meta.family.replace("-", " "),
                meta.intent.replace("_", " "),
                meta.output_kind.replace("_", " "),
                meta.summary,
                " ".join(meta.choose_when),
            )
        )
    )


def _score_one_tool(name: str, meta: ToolDiscovery, query_norm: str, query_tokens: set[str]) -> int:
    """Lexical score for one registry row (name + phrase + overlap + domain)."""
    score = _tool_name_bonus(name, query_norm) + _tool_phrase_bonus(meta, query_norm)
    score += len(query_tokens & _tool_field_tokens(name, meta))
    return score + _tool_domain_bonus(meta, query_tokens)


def _rank_scored(scored: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Relative-noise gate (within 4 of best) plus deterministic score/name order."""
    if scored:
        best = max(score for score, _ in scored)
        scored = [(score, name) for score, name in scored if score >= best - 4]

    def _rank_key(hit: tuple[int, str]) -> tuple[int, str]:
        return (-hit[0], hit[1])

    scored.sort(key=_rank_key)
    return scored


def _score_registry(query_norm: str, query_tokens: set[str], domain: str | None) -> list[tuple[int, str]]:
    """Score every in-domain row, then gate and rank."""
    scored: list[tuple[int, str]] = []
    for name, meta in TOOL_DISCOVERY_REGISTRY.items():
        if domain and meta.domain != domain:
            continue
        score = _score_one_tool(name, meta, query_norm, query_tokens)
        if score > 0:
            scored.append((score, name))
    # Relative noise gate: keep hits within 4 points of the best; wide-open
    # queries with no strong match keep everything.
    return _rank_scored(scored)


def _expand_conflicts(ranked_names: list[str], domain: str | None) -> list[str]:
    """Top primaries plus their direct conflicts (same domain), bounded to 5."""
    expanded = list(ranked_names)
    for name in ranked_names:
        for peer in TOOL_DISCOVERY_REGISTRY[name].conflicts_with:
            if peer not in expanded:
                if domain and TOOL_DISCOVERY_REGISTRY[peer].domain != domain:
                    continue
                expanded.append(peer)
    return expanded[:5]


def _search_tools(args: dict[str, object], model: str) -> dict[str, object]:
    """Generic lexical ranking over TOOL_DISCOVERY_REGISTRY fields only.

    Signals (small generic weights, no per-intent boosts): exact tool-name
    match (10) > exact phrase in summary/choose_when (5) > token overlap over
    name/domain/family/intent/output_kind/summary/choose_when
    (1 per token) > domain-name overlap (1). Ties break alphabetically for
    determinism. reject_when and related_tools never score: reject text names
    rivals, so counting it boosts a tool for the queries it must lose.
    A relative-noise margin keeps only hits within 4 points of
    the best score. Returns compact routing cards only: up to 3 lexical primaries plus direct conflicts, bounded to 5, ambiguity over the expanded set.
    """
    del model
    query, domain = _search_params(args)
    if not query.strip():
        return _empty_search_result()
    scored = _score_registry(" ".join(_normalize_discovery_text(query)), _discovery_keywords(query), domain)
    expanded = _expand_conflicts([name for _, name in scored[:3]], domain)
    ranked = [_routing_card(name) for name in expanded]
    ambiguous, groups = _ambiguity_groups(expanded)
    if not ranked:
        return _empty_search_result()
    return {"matches": ranked, "count": len(ranked), "ambiguous": ambiguous, "ambiguity_groups": groups}


def _list_tool_domains(args: dict[str, object], model: str) -> dict[str, object]:
    """Sorted domain catalog from the shared DOMAIN_DESCRIPTIONS map."""
    del args
    del model
    return {
        "domains": [{"name": name, "description": DOMAIN_DESCRIPTIONS[name]} for name in sorted(DOMAIN_DESCRIPTIONS)],
    }


def _schema_arg_lists(name: str) -> tuple[list[str], list[str]]:
    """Required/optional argument names for one tool from its canonical schema."""
    params, required, optional = _canonical_tool_schema(name)
    del params
    return required, optional


def _describe_one(name: str) -> dict[str, object]:
    """Full metadata for one named tool from the registry plus its canonical schema."""
    meta = TOOL_DISCOVERY_REGISTRY.get(name)
    if meta is None:
        return {"error": "unknown_tool", "name": name}
    required, optional = _schema_arg_lists(name)
    return {
        "name": name,
        "domain": meta.domain,
        "family": meta.family,
        "summary": meta.summary,
        "intent": meta.intent,
        "output_kind": meta.output_kind,
        "source": meta.source,
        "entity_scope": meta.entity_scope,
        "time_mode": meta.time_mode,
        "choose_when": list(meta.choose_when),
        "reject_when": list(meta.reject_when),
        "conflicts_with": list(meta.conflicts_with),
        "related_tools": list(meta.related_tools),
        "prerequisites": list(meta.prerequisites),
        "required_arguments": required,
        "optional_arguments": optional,
    }


def _parse_describe_names(name: str) -> list[object] | None:
    """Agents serialize the names array into the name string; coerce instead of failing."""
    # ponytail: coerce instead of failing their describe-then-call flow.
    try:
        parsed = json.loads(name)
    except json.JSONDecodeError, TypeError:
        return None
    return parsed if isinstance(parsed, list) and parsed else None


def _describe_tool(args: dict[str, object], model: str) -> dict[str, object]:
    """Full metadata for one named tool, or several tools in order with `names`."""
    del model
    raw = args.get("names")
    if isinstance(raw, list):
        return {"tools": [_describe_one(str(name)) for name in raw]}
    name = args.get("name") or ""
    if isinstance(name, str) and name.strip().startswith("["):
        parsed = _parse_describe_names(name)
        if parsed is not None:
            return {"tools": [_describe_one(str(item)) for item in parsed]}
    return _describe_one(str(name))


def _browse_key(nm: str) -> tuple[str, str, str]:
    """Sort key for the full catalog: (domain, family, name)."""
    meta = TOOL_DISCOVERY_REGISTRY[nm]
    return (meta.domain, meta.family, nm)


def _browse_selection(args: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    """Normalized (name, domain, family) browse coordinates, each None when blank."""
    raw_name = args.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    raw_domain = args.get("domain")
    domain = raw_domain.strip().lower() if isinstance(raw_domain, str) and raw_domain.strip() else None
    raw_family = args.get("family")
    family = raw_family.strip().lower() if isinstance(raw_family, str) and raw_family.strip() else None
    return name, domain, family


def _browse_named(name: str) -> dict[str, object]:
    """One tool by exact name (name-authoritative over guessed coordinates)."""
    # Name-authoritative: the 1B model copies the whole routing card into
    # browse (name plus a guessed domain/family). The exact name uniquely
    # identifies the tool, so resolve it instead of rejecting.
    if TOOL_DISCOVERY_REGISTRY.get(name) is None:
        return {"error": "unknown_tool", "name": name}
    info = _describe_one(name)
    params, _, _ = _canonical_tool_schema(name)
    return {**info, "parameters": params}


def _browse_family_names(domain: str, family: str) -> list[str]:
    """Catalog-sorted tool names for one domain/family path."""
    return sorted(
        (n for n, m in TOOL_DISCOVERY_REGISTRY.items() if m.domain == domain and m.family == family),
        key=_browse_key,
    )


def _browse_family_card(name: str) -> dict[str, object]:
    """Compact card for one family member (summary/intent/output only)."""
    meta = TOOL_DISCOVERY_REGISTRY[name]
    return {
        "name": name,
        "domain": meta.domain,
        "family": meta.family,
        "summary": meta.summary,
        "intent": meta.intent,
        "output_kind": meta.output_kind,
    }


def _browse_contrast_row(name: str) -> dict[str, object]:
    """One contrast row: primary use plus what it must not be used for."""
    meta = TOOL_DISCOVERY_REGISTRY[name]
    return {
        "tool": name,
        "use_it_for": meta.choose_when[0] if meta.choose_when else "",
        "do_not_use_it_for": " ".join(meta.reject_when),
    }


def _browse_family(domain: str, family: str) -> dict[str, object]:
    """Family path: member cards plus the choose/reject contrast table."""
    names = _browse_family_names(domain, family)
    if not names:
        return {
            "error": "unknown_family",
            "domain": domain,
            "families": sorted({m.family for n, m in TOOL_DISCOVERY_REGISTRY.items() if m.domain == domain}),
        }
    tools = [_browse_family_card(n) for n in names]
    return {
        "path": f"/{domain}/{family}",
        "domain": domain,
        "family": family,
        "tools": tools,
        "count": len(tools),
        "contrast_table": [_browse_contrast_row(n) for n in names],
    }


def _browse_domain(domain: str) -> dict[str, object]:
    """Domain path: families with tool counts."""
    fams: dict[str, list[str]] = {}
    for n, m in TOOL_DISCOVERY_REGISTRY.items():
        if m.domain == domain:
            fams.setdefault(m.family, []).append(n)
    families = [{"name": f, "path": f"/{domain}/{f}", "tool_count": len(v)} for f, v in sorted(fams.items())]
    return {"path": f"/{domain}", "domain": domain, "families": families}


def _browse_root() -> dict[str, object]:
    """Catalog root: domains only."""
    domains = [{"name": d, "path": f"/{d}", "description": DOMAIN_DESCRIPTIONS[d]} for d in sorted(DOMAIN_DESCRIPTIONS)]
    return {"path": "/", "domains": domains}


def _browse_tools(args: dict[str, object], model: str) -> dict[str, object]:
    """Hierarchical catalog: root domains, domain families, family tools + contrast, or one tool."""
    del model
    name, domain, family = _browse_selection(args)
    if name:
        return _browse_named(name)
    if family and not domain:
        return {"error": "family_requires_domain", "hint": "call browse_tools with domain and family"}
    if domain and domain not in DOMAIN_DESCRIPTIONS:
        return {"error": "unknown_domain", "domains": sorted(DOMAIN_DESCRIPTIONS)}
    if domain and family:
        return _browse_family(domain, family)
    if domain:
        return _browse_domain(domain)
    return _browse_root()


def _envelope_dict(raw: object) -> dict[str, object]:
    """String-keyed dict from envelope JSON, else empty."""
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def _envelope_attempts(raw: object) -> list[dict[str, object]]:
    """Attempt dicts from envelope JSON, else empty."""
    return (
        [{str(k): v for k, v in a.items()} for a in raw if isinstance(a, dict)]
        if isinstance(raw, (list, tuple))
        else []
    )


def _mention_hit(hit: dict[str, object]) -> dict[str, object]:
    """One text hit as a mention-role packet (unresolved subject)."""
    base: dict[str, object] = {str(k): v for k, v in hit.items()}
    base["match_role"] = "mention"
    base["subject_cik"] = None
    base["subject_name"] = None
    return base


def _passage_row(hit: dict[str, object]) -> dict[str, object]:
    """One matching passage: document + query + score + section + terms."""
    score = hit.get("score")
    return {
        "document": hit.get("matched_document"),
        "query": hit.get("query"),
        "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
        "section": hit.get("file_description") or hit.get("file_type"),
        "term": _passage_term(hit),
    }


def _passage_term(hit: dict[str, object]) -> str | None:
    """First exposure term in the hit description, else the query topic."""
    import re as _re

    text = str(hit.get("file_description") or "")
    match = _re.search(r"contract|concentration|investments?|commitments?|counterpart\w*|openai", text, _re.IGNORECASE)
    if match:
        return match.group(0).lower()
    query = str(hit.get("query") or "").strip()
    return query or None


def _document_matches(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    """Hits grouped by accession with their matching passages."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for hit in hits:
        key = hit.get("accession_no")
        accession = key if isinstance(key, str) and key else ""
        grouped.setdefault(accession, []).append(_passage_row(hit))
    return [{"accession": accession, "matching_passages": passages} for accession, passages in grouped.items()]


def _hit_window(hit: dict[str, object]) -> dict[str, object]:
    """One compact top hit: identity, document, relevance, and remainder pointer."""
    return {
        "accession": hit.get("accession_no"),
        "form": hit.get("form"),
        "filed_at": hit.get("filed_at"),
        "document": hit.get("matched_document"),
        "section": hit.get("file_description") or hit.get("file_type"),
        "term": _passage_term(hit),
        "window": hit.get("snippet") or hit.get("file_description"),
        "relevance_reason": list(hit.get("relevance_reason") or [])
        if isinstance(hit.get("relevance_reason"), (list, tuple))
        else [],
        "snippet": hit.get("snippet") or hit.get("file_description"),
        "resource_uri": hit.get("resource_uri"),
    }


_SEARCH_RETRIEVAL_NOTE = (
    "SEC search hits are navigation artifacts, never evidence: open the "
    "underlying filing/document and cite a raw passage before recording "
    "anything. This packet is a bounded display window — display_limit is a "
    "context saver, NOT retrieval completeness; the full ranked hit set is "
    "persisted and paged with research_read_search(session_id, search_id)."
)


def _discovery_packet(
    hits: list[dict[str, object]], search_id: object, limit: int | None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compact discovery packet: bounded top_hits + paging pointer (display only)."""
    top = [_hit_window(hit) for hit in hits] if limit is None else [_hit_window(hit) for hit in hits[:limit]]
    rest = 0 if limit is None else max(len(hits) - len(top), 0)
    remainder: dict[str, object] = {"count": rest}
    if rest and isinstance(search_id, str) and search_id:
        remainder["page_with"] = "research_read_search"
        remainder["next_offset"] = len(top)
        remainder["note"] = _SEARCH_RETRIEVAL_NOTE
    return top, remainder


def _mention_hits(raw: object) -> list[dict[str, object]]:
    """Text hits as mention packets."""
    if not isinstance(raw, (list, tuple)):
        return []
    return [_mention_hit(hit) for hit in raw if isinstance(hit, dict)]


def _search_runs(raw: object) -> list[dict[str, object]]:
    """SearchRun dicts from the result packet, else empty."""
    if not isinstance(raw, (list, tuple)):
        return []
    return [{str(k): v for k, v in run.items()} for run in raw if isinstance(run, dict)]


def _envelope_pit_basis(attempts: list[dict[str, object]]) -> str | None:
    """Most common attempt pit_basis, else None."""
    bases: list[object] = [a.get("pit_basis") for a in attempts if a.get("pit_basis") is not None]
    return max({str(b) for b in bases}, key=_bases_count_key(bases)) if bases else None


def _envelope_counts(data: dict[str, object], cov: dict[str, object]) -> dict[str, object]:
    """Reported/retrieved/pages plus entity/filing/document sizes."""

    def _len_of(key: str) -> int:
        value = data.get(key)
        return len(value) if isinstance(value, (list, tuple)) else 0

    return {
        "results_reported": cov.get("results_reported", 0),
        "results_retrieved": cov.get("results_retrieved", 0),
        "pages": cov.get("pages", 0),
        "entities": _len_of("entities"),
        "filings": _len_of("filings"),
        "documents": _len_of("documents"),
    }


def _envelope_backfill(cov: dict[str, object]) -> list[object]:
    """Pending backfill jobs, else empty."""
    return list(cov["pending_backfill_jobs"]) if isinstance(cov.get("pending_backfill_jobs"), (list, tuple)) else []


def _search_envelope(result: SECSearchResult, *, limit: int | None = 20) -> dict[str, object]:
    """SECSearchResult -> model packet: roles, ledger, PIT, jobs, evidence."""
    data = result.to_dict()
    cov = _envelope_dict(data.get("coverage"))
    attempts = _envelope_attempts(data.get("attempts"))
    hits = _mention_hits(data.get("text_hits"))
    request = _envelope_dict(data.get("request"))
    runs = _search_runs(data.get("search_runs"))
    documents = _document_matches(hits)
    top_hits, additional_hits = _discovery_packet(hits, data.get("search_id"), limit)
    return {
        "subject": request.get("query") or request.get("company_name"),
        "query": request.get("query"),
        "search_id": data.get("search_id"),
        "request": request,
        "scope": _envelope_scope(request),
        "count": len(hits),
        "retrieval": {
            "hits_are": "navigation_artifacts",
            "display_limit": limit,
            "display_limit_note": "context saver only; the full ranked hit set is persisted and never truncated by display_limit",
            "page_with": "research_read_search",
            "note": _SEARCH_RETRIEVAL_NOTE,
        },
        "top_hits": top_hits,
        "additional_hits": additional_hits,
        "entities": data.get("entities"),
        "entity_matches": data.get("entities"),
        "filings": data.get("filings"),
        "filing_candidates": data.get("filings"),
        "documents": data.get("documents"),
        "document_matches": documents,
        "parties": data.get("relationships"),
        "relationships": data.get("relationships"),
        "relationship_matches": data.get("relationships"),
        "hits": hits,
        "coverage": cov,
        "attempts": attempts,
        "search_runs": runs,
        "counts": _envelope_counts(data, cov),
        "pit_basis": _envelope_pit_basis(attempts),
        "warnings": data.get("warnings"),
        "errors": data.get("errors"),
        "backfill_jobs": _envelope_backfill(cov),
        "evidence_packet_ids": data.get("evidence_packet_ids"),
        "source": "SEC EDGAR",
    }


def _envelope_scope(request: dict[str, object]) -> dict[str, object]:
    """Issuer scope echo: ticker only (identity resolves server-side to CIK)."""
    scope: dict[str, object] = {}
    if isinstance(request.get("ticker"), str) and request.get("ticker"):
        scope["ticker"] = request.get("ticker")
    return scope


def _discovery_exhaustive(args: dict[str, object], context: RequestContext) -> bool:
    """Exhaustive SEC discovery by default in a research session; explicit caller choice wins."""
    explicit = args.get("exhaustive")
    if explicit is not None:
        return bool(explicit)
    return context.research_session_id is not None


def _discovery_bounds(args: dict[str, object], context: RequestContext) -> tuple[bool, int | None, int | None]:
    """(exhaustive, backend max_results, packet limit): exhaustive retrieval is never limit-bounded."""
    exhaustive = _discovery_exhaustive(args, context)
    packet = _optional_int(args.get("limit")) if args.get("limit") is not None else 20
    return exhaustive, (None if exhaustive else packet), packet


def _find_sec_entities(args: dict[str, object], context: RequestContext) -> dict[str, object]:
    """Entity discovery -> envelope with candidate verification statuses."""
    as_of, as_of_err = _sec_date_arg(args, "find_sec_entities", "as_of")
    if as_of_err is not None:
        return as_of_err
    exhaustive, max_results, packet = _discovery_bounds(args, context)
    return _search_envelope(
        sec.find_sec_entities(
            str(args["query"]),
            as_of=as_of,
            exhaustive=exhaustive,
            max_results=max_results,
            data_root=get_data_root(),
        ),
        limit=packet,
    )


def _sec_search_result(args: dict[str, object], context: RequestContext) -> dict[str, object]:
    """Discovery search -> envelope with jobs + evidence IDs; exhaustive in a research session."""
    if not any(
        (value.strip() if isinstance(value, str) else value)
        for key in (
            "query",
            "ticker",
            "cik",
            "company_name",
            "person_name",
            "domain",
            "accession_no",
            "security_identifier",
        )
        if (value := args.get(key)) is not None
    ):
        return _invalid_args_error(
            "search_sec_filings",
            "search_sec_filings needs one of: query, ticker, cik, "
            "company_name, person_name, domain, accession_no, "
            "security_identifier",
        )
    exhaustive, max_results, packet = _discovery_bounds(args, context)
    start, start_err = _sec_date_arg(args, "search_sec_filings", "start_date")
    if start_err is not None:
        return start_err
    end, end_err = _sec_date_arg(args, "search_sec_filings", "end_date")
    if end_err is not None:
        return end_err
    as_of, as_of_err = _sec_date_arg(args, "search_sec_filings", "as_of")
    if as_of_err is not None:
        return as_of_err
    raw_forms = args.get("forms")
    if isinstance(raw_forms, str):
        forms: tuple[str, ...] | None = (raw_forms,)
    elif isinstance(raw_forms, (list, tuple)):
        forms = tuple(str(x) for x in raw_forms)
    else:
        forms = None
    request = sec.SECSearchRequest(
        query=_str_or_none(args.get("query")),
        ticker=_str_or_none(args.get("ticker")),
        cik=_str_or_none(args.get("cik")),
        company_name=_str_or_none(args.get("company_name")),
        person_name=_str_or_none(args.get("person_name")),
        domain=_str_or_none(args.get("domain")),
        accession_no=_str_or_none(args.get("accession_no")),
        security_identifier=_str_or_none(args.get("security_identifier")),
        forms=forms,
        start_date=start,
        end_date=end,
        as_of=as_of,
        exhaustive=exhaustive,
        max_results=max_results,
    )
    return _search_envelope(sec.SECDiscoveryService(data_root=get_data_root()).search(request), limit=packet)


def _find_sec_entities_bounded(args: dict[str, object], context: RequestContext) -> dict[str, object]:
    """Bounded entity lookup: forced exhaustive=false through the shared handler."""
    return _find_sec_entities({**args, "exhaustive": False}, context)


def _sec_search_result_bounded(args: dict[str, object], context: RequestContext) -> dict[str, object]:
    """Bounded filings lookup: forced exhaustive=false through the shared handler."""
    return _sec_search_result({**args, "exhaustive": False}, context)


class _DocView(TypedDict, total=False):
    """get_sec_document view kwargs (section/query/raw only when supplied)."""

    section: str
    query: str
    raw: bool


def _doc_offset(raw: object) -> int:
    """Lenient offset coercion (bool/float/str all narrow to int, None is 0)."""
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    if raw is None:
        return 0
    if isinstance(raw, str):
        return int(raw.strip()) if raw.strip() else 0
    return int(str(raw))


def _doc_max_chars(raw: object) -> int | None:
    """Lenient max_chars coercion (None/blank keep the 12k model bound)."""
    if raw is None:
        return 12_000
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        return int(raw.strip()) if raw.strip() else 12_000
    return int(str(raw))


def _doc_view_kwargs(args: dict[str, object]) -> _DocView:
    """section/query/raw only when the model supplied them (legacy fakes stay green)."""
    extra: _DocView = {}
    section = _str_or_none(args.get("section"))
    query = _str_or_none(args.get("query"))
    if section is not None:
        extra["section"] = section
    if query is not None:
        extra["query"] = query
    if args.get("raw") is not None:
        extra["raw"] = bool(args.get("raw", False))
    return extra


_SEC_ACCESSION_HINT = (
    "pass accession_no like 0001628280-26-044069 from the search_sec_filings packet; "
    "accession_number is not a valid key"
)


def _sec_accession_value(args: dict[str, object], tool: str) -> tuple[str | None, dict[str, object] | None]:
    """Validated accession_no or a self-correcting invalid_tool_arguments error."""
    raw = args.get("accession_no")
    if not isinstance(raw, str) or not raw.strip():
        return None, {
            "error": f"tool '{tool}': missing accession_no {raw!r}; {_SEC_ACCESSION_HINT}",
            "error_type": "invalid_tool_arguments",
        }
    try:
        sec.normalize_accession_no(raw)
    except Exception:  # noqa: BLE001 - any normalize failure is a bad accession, guidance owns it
        return None, {
            "error": f"tool '{tool}': invalid accession_no {raw!r}; {_SEC_ACCESSION_HINT}",
            "error_type": "invalid_tool_arguments",
        }
    return raw.strip(), None


def _diff_accession_value(args: dict[str, object], tool: str, key: str) -> tuple[str | None, dict[str, object] | None]:
    """Validated diff accession (current/previous) or a self-correcting error."""
    raw = args.get(key)
    if raw is None:
        return None, None
    if not isinstance(raw, str) or not raw.strip():
        return None, {
            "error": f"tool '{tool}': invalid {key} {raw!r}; {_SEC_ACCESSION_HINT}",
            "error_type": "invalid_tool_arguments",
        }
    try:
        sec.normalize_accession_no(raw)
    except Exception:  # noqa: BLE001 - any normalize failure is a bad accession, guidance owns it
        return None, {
            "error": f"tool '{tool}': invalid {key} {raw!r}; {_SEC_ACCESSION_HINT}",
            "error_type": "invalid_tool_arguments",
        }
    return raw.strip(), None


def _get_sec_filing(args: dict[str, object], model: str) -> dict[str, object]:
    """One filing's record by accession; bad accessions get self-correcting guidance."""
    val, err = _sec_accession_value(args, "get_sec_filing")
    if err is not None or val is None:
        assert err is not None
        return err
    as_of, err = _sec_date_arg(args, "get_sec_filing", "as_of")
    if err is not None:
        return err
    try:
        return sec.get_sec_filing(val, as_of=as_of).to_dict()
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "error_type": "invalid_tool_arguments"}


def _get_sec_document(args: dict[str, object], model: str) -> dict[str, object]:
    """Archive-first document read; model callers always get a bounded window."""
    del model
    val, err = _sec_accession_value(args, "get_sec_document")
    if err is not None or val is None:
        assert err is not None
        return err
    as_of, err = _sec_date_arg(args, "get_sec_document", "as_of")
    if err is not None:
        return err
    try:
        cursor = args.get("cursor")
        limit = args.get("limit")
        return sec.get_sec_document(
            val,
            _str_or_none(args.get("document_name")),
            as_of=as_of,
            offset=_doc_offset(cursor if cursor is not None else args.get("offset", 0)),
            max_chars=_doc_max_chars(limit if limit is not None else args.get("max_chars", 12_000)),
            data_root=get_data_root(),
            **_doc_view_kwargs(args),
        )
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "error_type": "invalid_tool_arguments"}


def _list_sec_documents(args: dict[str, object], model: str) -> dict[str, object]:
    """Documents for one filing; bad accessions get self-correcting guidance."""
    del model
    val, err = _sec_accession_value(args, "list_sec_documents")
    if err is not None or val is None:
        assert err is not None
        return err
    as_of, err = _sec_date_arg(args, "list_sec_documents", "as_of")
    if err is not None:
        return err
    try:
        return _wrap_list(
            val,
            sec.list_sec_documents(val, as_of=as_of),
            "documents",
        )
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "error_type": "invalid_tool_arguments"}


_REL_PARTIAL_STATUSES = ("partial", "source_limited", "complete_within_source_limits", "retrying")


def _rel_types(raw: object) -> Sequence[str] | None:
    """relationship_types coercion: string, list/tuple of strings, else None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    return None


def _as_result_list(result: dict[str, object], key: str) -> list[object]:
    """List field from the relationships result, else empty."""
    raw = result.get(key)
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _rel_attempts(result: dict[str, object]) -> tuple[list[object], list[dict[str, object]]]:
    """Errors plus attempt dicts from the relationships result."""
    errors = _as_result_list(result, "errors")
    raw_attempts = result.get("attempts")
    attempts = (
        [{str(k): v for k, v in a.items()} for a in raw_attempts if isinstance(a, dict)]
        if isinstance(raw_attempts, list)
        else []
    )
    return errors, attempts


def _rel_attempt_flags(attempts: list[dict[str, object]]) -> tuple[bool, bool]:
    """(has_partial, has_failed) over attempt statuses."""
    has_partial = any(a.get("status") in _REL_PARTIAL_STATUSES for a in attempts)
    return has_partial, any(a.get("status") == "failed" for a in attempts)


def _rel_coverage_status(
    result: dict[str, object], errors: list[object], attempts: list[dict[str, object]], found: int
) -> str:
    """failed when errors/failures explain zero hits, partial on any caveat, else complete."""
    has_partial, has_failed = _rel_attempt_flags(attempts)
    if (errors and not found) or (has_failed and not found):
        return "failed"
    if errors or result.get("warnings") or has_partial or has_failed:
        return "partial"
    return "complete"


def _rel_ciks(result: dict[str, object]) -> list[object]:
    """CIK list from the relationships result, else empty."""
    return list(result["ciks"]) if isinstance(result.get("ciks"), (list, tuple)) else []


def _rel_request(args: dict[str, object]) -> dict[str, object]:
    """Echo of the relationship request coordinates."""
    return {
        "entity": args.get("entity"),
        "relationship_types": args.get("relationship_types"),
        "as_of": args.get("as_of"),
    }


def _sec_relationships_result(args: dict[str, object]) -> dict[str, object]:
    as_of, as_of_err = _sec_date_arg(args, "search_sec_relationships", "as_of")
    if as_of_err is not None:
        return as_of_err
    result = sec.search_sec_relationships(
        str(args["entity"]),
        relationship_types=_rel_types(args.get("relationship_types")),
        as_of=as_of,
        limit=int(str(args.get("limit", 50) or 50)),
        exhaustive=bool(args.get("exhaustive", True)),
    )
    typed_list = _as_result_list(result, "typed")
    rels_list = _as_result_list(result, "relationships")
    ment_list = _as_result_list(result, "mentions")
    found = len(typed_list) + len(rels_list) + len(ment_list)
    errors, attempts = _rel_attempts(result)
    return {
        "subject": args.get("entity"),
        "entity": result.get("entity"),
        "ciks": _rel_ciks(result),
        "request": _rel_request(args),
        "count": found,
        "groups": result.get("groups"),
        "parties": result.get("typed"),
        "relationships": result.get("relationships"),
        "mentions": result.get("mentions"),
        "coverage": {"status": _rel_coverage_status(result, errors, attempts, found)},
        "attempts": result.get("attempts"),
        "counts": {"typed": len(typed_list), "workflow": len(rels_list), "mentions": len(ment_list)},
        "pit_basis": "known_at" if args.get("as_of") else None,
        "warnings": result.get("warnings"),
        "errors": errors,
        "backfill_jobs": [],
        "source": "SEC EDGAR",
    }


def _list_sec_filings(args: dict[str, object], model: str) -> dict[str, object]:
    """List filings with lenient tool-JSON coercions (forms union narrowed here)."""
    del model
    start, err = _sec_date_arg(args, "list_sec_filings", "start_date")
    if err is not None:
        return err
    end, err = _sec_date_arg(args, "list_sec_filings", "end_date")
    if err is not None:
        return err
    as_of, err = _sec_date_arg(args, "list_sec_filings", "as_of")
    if err is not None:
        return err
    forms, err = _filing_forms_arg(args.get("forms"), "list_sec_filings")
    if err is not None:
        return err
    identifier = _remap_mixed_case(args, "identifier", str(args["identifier"]).strip().upper())
    return _wrap_list(
        identifier,
        sec.list_sec_filings(
            identifier,
            forms=forms,
            start_date=start,
            end_date=end,
            as_of=as_of,
            limit=_optional_int(args.get("limit", 50)),
        ),
        "filings",
    )


def _filing_forms(raw: object) -> str | list[str] | tuple[str, ...] | None:
    """forms union coercion shared by list/diff filings (string, tuple, else None)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    return None


_FILING_FORMS_HINT = "pass forms as SEC form types (e.g. 10-K, 10-Q, 8-K, 4); dates belong in start_date/end_date"


def _date_like_form(value: object) -> bool:
    """Date-shaped form values are misrouted date args, never valid SEC forms."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if "YYYY" in text.upper():
        return True
    if _FINRA_DATE_RE.match(text):
        return True
    return "/" in text and re.search(r"\d{4}", text) is not None


def _filing_forms_arg(
    raw: object, tool: str
) -> tuple[str | list[str] | tuple[str, ...] | None, dict[str, object] | None]:
    """Coerced forms or an invalid_tool_arguments error naming the date-like value."""
    forms = _filing_forms(raw)
    if forms is None:
        return None, None
    values = [forms] if isinstance(forms, str) else list(forms)
    if any(_date_like_form(v) for v in values):
        return None, _invalid_args_error(tool, f"tool '{tool}': invalid forms {raw!r}; {_FILING_FORMS_HINT}")
    return forms, None


def _recent_filings(ticker: str, args: dict[str, object]) -> object:
    """Up to 10 recent filings for ticker self-resolution; errors stay a dict."""
    forms, err = _filing_forms_arg(args.get("forms"), "diff_sec_filings")
    if err is not None:
        return err
    try:
        as_of, _ = _sec_date_arg(args, "diff_sec_filings", "as_of")
        return sec.list_sec_filings(ticker, forms=forms, as_of=as_of, limit=10)
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"error": str(exc)}


def _filing_accession(filing: object) -> str:
    """Accession number narrowed; raises on unexpected shapes."""
    accession = getattr(filing, "accession_no", None)
    if isinstance(accession, str) and accession:
        return accession
    raise TypeError(f"filing must carry accession_no, got {type(filing).__name__}")


def _pick_filing_pair(filings: list[Filing]) -> tuple[Filing, Filing]:
    """Latest plus the newest same-form predecessor (else the second filing)."""
    latest = filings[0]
    same = [f for f in filings[1:] if f.form == latest.form]
    return (latest, same[0]) if same else (filings[0], filings[1])


def _diff_resolved_pair(ticker: str, filings: list[Filing], section: str | None) -> dict[str, object]:
    """Diff the picked pair, tagged with the resolving ticker."""
    current, previous = _pick_filing_pair(filings)
    out = sec.diff_filings(_filing_accession(current), _filing_accession(previous), section=section)
    if isinstance(out, dict) and "error" not in out:
        out = {**out, "ticker": ticker.strip().upper(), "resolved_via": "list_sec_filings-internal"}
    return out


def _diff_sec_filings(args: dict[str, object], model: str) -> dict[str, object]:
    """Accession pair direct, or ticker self-resolution via sec.list_sec_filings."""
    del model
    cur, err = _diff_accession_value(args, "diff_sec_filings", "current_accession")
    if err is not None:
        return err
    prev, err = _diff_accession_value(args, "diff_sec_filings", "previous_accession")
    if err is not None:
        return err
    section = _str_or_none(args.get("section"))
    if cur and prev:
        return sec.diff_filings(cur, prev, section=section)
    ticker = _str_or_none(args.get("ticker"))
    if not ticker:
        return _invalid_args_error(
            "diff_sec_filings", "Provide ticker or current_accession+previous_accession for tool 'diff_sec_filings'"
        )
    filings = _recent_filings(ticker, args)
    if isinstance(filings, dict):
        return filings
    assert isinstance(filings, list)
    if len(filings) < 2:
        return {"error": f"No pair of filings found for {ticker}: {len(filings)} match"}
    return _diff_resolved_pair(ticker, filings, section)


def _edgar_ticker(name: str) -> str | None:
    """EDGAR company-index top hit tickers[0], else None (never raises)."""
    try:
        from app.sec.client import find_sec_company

        for cand in find_sec_company(name, limit=3):
            ticks = cand.get("tickers") if isinstance(cand, dict) else None
            if isinstance(ticks, list) and ticks and isinstance(ticks[0], str) and ticks[0].strip():
                return ticks[0].strip().upper()
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass
    return None


def _resolve_company_to_ticker(name: str) -> str | None:
    """Company name to ticker via the EDGAR company index top hit."""
    return _edgar_ticker(name)


_FINRA_ORG_WORDS = frozenset({"FINRA", "SEC", "NYSE", "NASDAQ", "OTC", "EDGAR"})


def _upper_arg(args: dict[str, object], key: str) -> str | None:
    """Uppercase ticker/entity arg, None when blank/non-string."""
    raw = args.get(key)
    return raw.strip().upper() if isinstance(raw, str) and raw.strip() else None


def _remap_mixed_case(args: dict[str, object], key: str, value: str) -> str:
    """Ticker or company name: exact-ticker passthrough, else EDGAR index remap."""
    raw = args.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return value
    try:
        from app.sec.client import resolve_cik as _resolve_cik

        if _resolve_cik(value) is not None:
            return value
    except Exception:  # noqa: BLE001, S110 - exact-ticker check never blocks company remap
        pass
    resolved = _resolve_company_to_ticker(raw.strip())
    if resolved is not None and resolved != value:
        return resolved
    return value


def _company_arg(args: dict[str, object]) -> str | None:
    """company_name arg stripped, None when blank/non-string."""
    raw = args.get("company_name")
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _get_obligations(args: dict[str, object], model: str) -> dict[str, object]:
    """Ticker or company name; the server maps names to tickers for single dispatch."""
    del model
    ticker = _upper_arg(args, "ticker")
    if ticker is not None:
        ticker = _remap_mixed_case(args, "ticker", ticker)
    if ticker is None:
        name = _company_arg(args)
        if name is None:
            return _invalid_args_error(
                "get_obligations",
                "Provide a ticker (e.g. AAPL) or company_name (e.g. Apple) for tool 'get_obligations'",
            )
        resolved = _resolve_company_to_ticker(name)
        if resolved is None:
            return _invalid_args_error("get_obligations", f"Unknown company name '{name}'; pass a ticker like AAPL")
        ticker = resolved
    return obligations.get_obligations(ticker)


def _ticker_or_company_name(
    args: dict[str, object], tool: str, key: str = "ticker"
) -> tuple[str | None, dict[str, object] | None]:
    """Ticker/entity value or company_name; maps names to tickers for single dispatch."""
    value = _upper_arg(args, key)
    if value is not None and value not in _FINRA_ORG_WORDS:
        return _remap_mixed_case(args, key, value), None
    name = _company_arg(args)
    if name is None:
        return None, _invalid_args_error(
            tool,
            f"Provide an entity/ticker (e.g. AAPL) or company_name (e.g. Apple) for tool '{tool}'; never call with neither",
        )
    resolved = _resolve_company_to_ticker(name)
    if resolved is None:
        return None, _invalid_args_error(tool, f"Unknown company name '{name}'; pass a {key} like AAPL")
    return resolved, None


def _get_beneficial_ownership(args: dict[str, object], model: str) -> dict[str, object]:
    """Ticker or company name."""
    del model
    ticker, err = _ticker_or_company_name(args, "get_beneficial_ownership")
    if err is not None:
        return err
    assert ticker is not None
    as_of, as_of_err = _sec_date_arg(args, "get_beneficial_ownership", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        ticker,
        sec.get_beneficial_ownership(
            ticker,
            as_of=as_of,
            limit=_optional_int(args.get("limit", 20)),
        ),
        "records",
    )


def _get_insider_activity(args: dict[str, object], model: str) -> dict[str, object]:
    """Ticker or company name."""
    del model
    ticker = _upper_arg(args, "ticker")
    if ticker is not None and ticker not in _FINRA_ORG_WORDS:
        ticker = _remap_mixed_case(args, "ticker", ticker)
    if ticker is None or ticker in _FINRA_ORG_WORDS:
        name = _company_arg(args)
        if name is None:
            return _invalid_args_error(
                "get_insider_activity",
                "Provide a ticker (e.g. AAPL) or company_name (e.g. Apple) for tool 'get_insider_activity'",
            )
        resolved = _resolve_company_to_ticker(name)
        if resolved is None:
            return _invalid_args_error(
                "get_insider_activity", f"Unknown company name '{name}'; pass a ticker like AAPL"
            )
        ticker = resolved
    as_of, as_of_err = _sec_date_arg(args, "get_insider_activity", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        ticker,
        sec.get_insider_activity(ticker, as_of=as_of, limit=_optional_int(args.get("limit", 50))),
        "transactions",
    )


def _search_sec_relationships(args: dict[str, object], model: str) -> dict[str, object]:
    """Entity or company name; tickers are valid entity values."""
    del model
    entity, err = _ticker_or_company_name(args, "search_sec_relationships", key="entity")
    if err is not None:
        return err
    assert entity is not None
    return _sec_relationships_result({**args, "entity": entity})


def _sec_date_arg(args: dict[str, object], tool: str, key: str) -> tuple[str | None, dict[str, object] | None]:
    """Validated YYYY-MM-DD SEC date (None when blank); one line per Q-path call site."""
    return _finra_date(args.get(key), tool, key)


def _get_material_events(args: dict[str, object], model: str) -> dict[str, object]:
    """8-K events; bad since/as_of get self-correcting guidance."""
    del model
    since, err = _sec_date_arg(args, "get_material_events", "since")
    if err is not None:
        return err
    if since is None:
        return _invalid_args_error(
            "get_material_events", "Provide a since date YYYY-MM-DD for tool 'get_material_events'"
        )
    as_of, err = _sec_date_arg(args, "get_material_events", "as_of")
    if err is not None:
        return err
    ticker = _upper_arg(args, "ticker")
    if ticker is None or ticker in _FINRA_ORG_WORDS:
        name = _company_arg(args)
        if name is None:
            return _invalid_args_error(
                "get_material_events",
                "Provide a ticker (e.g. AAPL) or company_name (e.g. Apple) for tool 'get_material_events'",
            )
        resolved = _resolve_company_to_ticker(name)
        if resolved is None:
            return _invalid_args_error("get_material_events", f"Unknown company name '{name}'; pass a ticker like AAPL")
        ticker = resolved
    else:
        ticker = _remap_mixed_case(args, "ticker", ticker)
    try:
        events = sec.get_material_events(ticker, since, as_of=as_of, limit=_optional_int(args.get("limit", 50)))
    except Exception as exc:
        if "not found" in str(exc).lower():
            return _invalid_args_error(
                "get_material_events",
                f"Unknown ticker '{ticker}'; resolve the company via find_sec_entities first, "
                "then retry with its ticker",
            )
        raise
    return _wrap_list(ticker, events, "events")


def _get_governance_events(args: dict[str, object], model: str) -> dict[str, object]:
    """Governance events; bad since/as_of get self-correcting guidance."""
    del model
    since, err = _sec_date_arg(args, "get_governance_events", "since")
    if err is not None:
        return err
    as_of, err = _sec_date_arg(args, "get_governance_events", "as_of")
    if err is not None:
        return err
    return _wrap_list(
        args.get("ticker"),
        sec.get_governance_events(
            str(args["ticker"]), since=since, as_of=as_of, limit=_optional_int(args.get("limit", 10))
        ),
        "events",
    )


def _get_short_pressure_profile(args: dict[str, object], model: str) -> dict[str, object]:
    """Short-vs-outstanding context; ticker routes through the FINRA normalizer."""
    del model
    ticker = _finra_ticker(args)
    if ticker is None:
        return _invalid_args_error(
            "get_short_pressure_profile", "Provide a ticker (e.g. AAPL) for tool 'get_short_pressure_profile'"
        )
    return sec.get_short_pressure_context(ticker)


# Direct-dispatch tools (EDGAR/analyst/obligations/valuation) — same
# registry pattern as the FINRA/Robinhood handler maps below.
def _get_ownership_changes(args: dict[str, object], model: str) -> dict[str, object]:
    """Deterministic 13D/G diffs; bad as_of gets self-correcting guidance."""
    del model
    as_of, as_of_err = _sec_date_arg(args, "get_ownership_changes", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        args.get("ticker"),
        sec.get_ownership_changes(
            str(args["ticker"]),
            as_of=as_of,
            limit=_optional_int(args.get("limit", 20)),
        ),
        "changes",
    )


def _get_planned_insider_sales(args: dict[str, object], model: str) -> dict[str, object]:
    """Planned Form 144 notices; bad as_of gets self-correcting guidance."""
    del model
    as_of, as_of_err = _sec_date_arg(args, "get_planned_insider_sales", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        args.get("ticker"),
        sec.get_planned_insider_sales(
            str(args["ticker"]),
            as_of=as_of,
            limit=_optional_int(args.get("limit", 20)),
        ),
        "proposed_sales",
    )


def _get_offering_history(args: dict[str, object], model: str) -> dict[str, object]:
    """Financing history; bad as_of gets self-correcting guidance."""
    del model
    as_of, as_of_err = _sec_date_arg(args, "get_offering_history", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        args.get("ticker"),
        sec.get_offering_history(
            str(args["ticker"]),
            as_of=as_of,
            limit=_optional_int(args.get("limit", 50)),
        ),
        "offerings",
    )


def _get_dilution_profile(args: dict[str, object], model: str) -> dict[str, object]:
    """Deterministic dilution math; bad as_of gets self-correcting guidance."""
    del model
    as_of, as_of_err = _sec_date_arg(args, "get_dilution_profile", "as_of")
    if as_of_err is not None:
        return as_of_err
    return sec.get_dilution_profile(str(args["ticker"]), as_of=as_of)


def _get_transaction_status(args: dict[str, object], model: str) -> dict[str, object]:
    """M&A filing context; bad as_of gets self-correcting guidance."""
    del model
    as_of, as_of_err = _sec_date_arg(args, "get_transaction_status", "as_of")
    if as_of_err is not None:
        return as_of_err
    return _wrap_list(
        args.get("ticker"),
        sec.get_transaction_status(
            str(args["ticker"]),
            as_of=as_of,
            limit=_optional_int(args.get("limit", 10)),
        ),
        "transactions",
    )


_MODEL_HANDLERS: dict[str, ModelHandler] = {
    "evaluate_mandate": lambda args, model: evaluate_mandate(),
    "get_fundamentals": lambda args, model: sec_facts.get_fundamentals(
        str(args["ticker"]), str(args["metric"]), as_of=_str_or_none(args.get("as_of"))
    ),
    "search_sec_relationships": _search_sec_relationships,
    "get_sec_search_coverage": lambda args, model: sec.get_sec_search_coverage(
        source=_str_or_none(args.get("source")),
        form=_str_or_none(args.get("form")),
        search_id=_str_or_none(args.get("search_id")),
        limit=int(str(args.get("limit", 200))),
    ),
    "list_sec_filings": _list_sec_filings,
    "get_sec_filing": _get_sec_filing,
    "list_sec_documents": _list_sec_documents,
    "get_sec_document": _get_sec_document,
    "diff_sec_filings": _diff_sec_filings,
    "get_material_events": _get_material_events,
    "get_beneficial_ownership": _get_beneficial_ownership,
    "get_ownership_changes": _get_ownership_changes,
    "get_insider_activity": _get_insider_activity,
    "get_planned_insider_sales": _get_planned_insider_sales,
    "get_offering_history": _get_offering_history,
    "get_dilution_profile": _get_dilution_profile,
    "get_governance_events": _get_governance_events,
    "get_transaction_status": _get_transaction_status,
    "get_short_pressure_profile": _get_short_pressure_profile,
    "search_tools": _search_tools,
    "list_tool_domains": _list_tool_domains,
    "describe_tool": _describe_tool,
    "browse_tools": _browse_tools,
    "get_recent_ownership_filings": lambda args, model: edgar_client.get_recent_ownership_filings(
        str(args.get("form_type", "both")), int(str(args.get("limit", 10)))
    ),
    "diff_risk_factors": lambda args, model: edgar_client.diff_risk_factors(str(args["ticker"])),
    "get_xbrl_facts": lambda args, model: sec_facts.get_xbrl_facts(str(args["ticker"]), str(args["concept"])),
    "get_financial_statements": lambda args, model: edgar_client.get_financial_statements(
        str(args["ticker"]), str(args["statement_type"])
    ),
    "get_analyst_estimates": lambda args, model: analyst_client.get_analyst_estimates(str(args["ticker"])),
    "get_sp500_weight": lambda args, model: analyst_client.get_sp500_weight(str(args["ticker"])),
    "get_obligations": _get_obligations,
    "get_valuation_metrics": lambda args, model: valuation.get_valuation_metrics(str(args["ticker"])),
    "search_web": _search_web,
    "find_alternative_signals": _find_alternative_signals,
    "get_trend_evidence": _get_trend_evidence,
    "investigate_social_arbitrage_candidate": _investigate_social_arbitrage_candidate,
    "get_macro_context": _get_macro_context,
    "search_company_patents": _search_company_patents,
    "get_current_time": _get_current_time,
}


_FINRA_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _finra_date(value: object, tool: str, key: str) -> tuple[str | None, dict[str, object] | None]:
    """Validated YYYY-MM-DD date or a self-correcting invalid_tool_arguments error."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    text = value.strip() if isinstance(value, str) else ""
    msg = f"tool '{tool}': invalid {key} {value!r}; expected YYYY-MM-DD"
    if not _FINRA_DATE_RE.match(text):
        return None, _invalid_args_error(tool, msg)
    try:
        date.fromisoformat(text)
    except ValueError:
        return None, _invalid_args_error(tool, msg)
    return text, None


_FINRA_DATASET_HINT = "pass dataset as canonical group/name (e.g. otcMarket/regShoDaily); unambiguous bare names resolve, unknown/ambiguous ones are rejected"


def _finra_dataset(args: dict[str, object], tool: str, key: str) -> tuple[str | None, dict[str, object] | None]:
    """Canonical group/name dataset (or unambiguous legacy bare name) or an invalid_tool_arguments error."""
    raw = args.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None, _invalid_args_error(tool, f"tool '{tool}': invalid {key} {raw!r}; {_FINRA_DATASET_HINT}")
    text = raw.strip()
    if "/" in text:
        return text, None
    try:
        entry = finra_client._resolve_dataset(text)
    except ValueError as exc:
        return None, _invalid_args_error(tool, f"tool '{tool}': invalid {key} {raw!r}; {exc}")
    return entry.dataset_id, None


def _finra_ticker(args: dict[str, object]) -> str | None:
    """Uppercase FINRA ticker; mixed-case values remap via the EDGAR index."""
    value = _upper_arg(args, "ticker")
    if value is None or value in _FINRA_ORG_WORDS:
        return None
    return _remap_mixed_case(args, "ticker", value)


def _get_short_interest(args: dict[str, object], model: str) -> dict[str, object]:
    """One ticker's biweekly short position; bad dates get self-correcting guidance."""
    del model
    ticker = _finra_ticker(args)
    if ticker is None:
        return _invalid_args_error("get_short_interest", "Provide a ticker (e.g. AAPL) for tool 'get_short_interest'")
    val, err = _finra_date(args.get("settlementDate"), "get_short_interest", "settlementDate")
    if err is not None:
        return err
    return finra_client.get_short_interest(ticker, val)


_SHO_NYC = ZoneInfo("America/New_York")


def _sho_week_range() -> tuple[str, str]:
    """Monday-NYC start through today-NYC end (exchanges run on NYC dates)."""
    now_utc = datetime.now(UTC)
    nyc_today = now_utc.astimezone(_SHO_NYC).date()
    monday = nyc_today - timedelta(days=nyc_today.weekday())
    return monday.isoformat(), nyc_today.isoformat()


def _sho_number(value: object) -> float | None:
    """Lenient numeric coercion for FINRA metric sums/latest (None when unparseable)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _sho_field_sum(metrics: dict[str, object], field: str) -> float | None:
    """Sum for one metrics.fields entry, else None."""
    fields = metrics.get("fields")
    if not isinstance(fields, dict):
        return None
    entry = fields.get(field)
    if not isinstance(entry, dict):
        return None
    return _sho_number(entry.get("sum"))


def _sho_latest(metrics: dict[str, object], field: str) -> float | None:
    """Latest value for one latest_vs_prior row, else None."""
    rows = metrics.get("latest_vs_prior")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("field") == field:
            return _sho_number(row.get("latest"))
    return None


def _sho_volumes(metrics: dict[str, object]) -> tuple[float | None, float | None]:
    """(combined_short, total): latest-day triple when present, else aggregate sums."""
    short_l = _sho_latest(metrics, "shortParQuantity")
    exempt_l = _sho_latest(metrics, "shortExemptParQuantity")
    total_l = _sho_latest(metrics, "totalParQuantity")
    if short_l is not None and exempt_l is not None and total_l is not None and total_l > 0:
        return short_l + exempt_l, total_l
    short_s = _sho_field_sum(metrics, "shortParQuantity")
    exempt_s = _sho_field_sum(metrics, "shortExemptParQuantity")
    total_s = _sho_field_sum(metrics, "totalParQuantity")
    if short_s is None or exempt_s is None or total_s is None or total_s <= 0:
        return None, None
    return short_s + exempt_s, total_s


def _sho_int(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else value


def _sho_enrich(result: dict[str, object]) -> dict[str, object]:
    """Add short-volume ratio keys to a Reg SHO briefing; errors pass through."""
    if not isinstance(result, dict) or "error" in result:
        return result
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return result
    short_volume, total_volume = _sho_volumes(metrics)
    ratio = round(short_volume / total_volume * 100, 2) if short_volume is not None and total_volume else None
    enriched = dict(result)
    enriched["short_volume"] = _sho_int(short_volume)
    enriched["total_volume"] = _sho_int(total_volume)
    enriched["short_volume_ratio_pct"] = ratio
    enriched["short_volume_ratio_elevated"] = ratio is not None and ratio >= 40
    enriched["short_volume_ratio_note"] = (
        "Combined short + short-exempt share of total volume; "
        "40-50% generally considered elevated (>=50% highly elevated)."
    )
    return enriched


def _get_reg_sho_volume(args: dict[str, object], model: str) -> dict[str, object]:
    """Ticker or company name; one ticker's daily Reg SHO volume."""
    del model
    ticker, terr = _ticker_or_company_name(args, "get_reg_sho_volume")
    if terr is not None:
        return terr
    assert ticker is not None
    val, err = _finra_date(args.get("tradeDate"), "get_reg_sho_volume", "tradeDate")
    if err is not None:
        return err
    if val is None:
        monday, today = _sho_week_range()
        return _sho_enrich(
            finra_client.query_dataset("otcMarket/regShoDaily", ticker, start_date=monday, end_date=today, limit=100)
        )
    return _sho_enrich(finra_client.get_reg_sho_volume(ticker, val))


def _finra_range(args: dict[str, object], tool: str) -> tuple[str | None, str | None, dict[str, object] | None]:
    """Validated start/end_date pair or a self-correcting invalid_tool_arguments error."""
    start, err = _finra_date(args.get("start_date"), tool, "start_date")
    if err is not None:
        return None, None, err
    end, err = _finra_date(args.get("end_date"), tool, "end_date")
    if err is not None:
        return None, None, err
    return start, end, None


def _get_threshold_securities(args: dict[str, object], model: str) -> dict[str, object]:
    """Threshold list, optionally filtered; bad dates get self-correcting guidance."""
    del model
    val, err = _finra_date(args.get("tradeDate"), "get_threshold_securities", "tradeDate")
    if err is not None:
        return err
    return finra_client.get_threshold_securities(_finra_ticker(args), val)


def _get_finra_datapoints(args: dict[str, object], model: str) -> dict[str, object]:
    """Exact datapoints; bad dates get self-correcting guidance."""
    del model
    start, end, err = _finra_range(args, "get_finra_datapoints")
    if err is not None:
        return err
    dataset, err = _finra_dataset(args, "get_finra_datapoints", "dataset")
    if err is not None or dataset is None:
        assert err is not None
        return err
    ticker = _finra_ticker(args) or _str_or_none(args.get("symbol"))
    sort_order = _str_or_none(args.get("sort_order"))
    if sort_order is not None and sort_order not in ("asc", "desc"):
        return _invalid_args_error(
            "get_finra_datapoints",
            f"tool 'get_finra_datapoints': invalid sort_order {sort_order!r}; expected one of ['asc', 'desc']",
        )
    return finra_client.get_finra_datapoints(
        dataset,
        fields=args.get("fields"),
        ticker=ticker,
        start_date=start,
        end_date=end,
        limit=_optional_int(args.get("limit")),
        filters=args.get("filters"),
        sort_fields=args.get("sort_fields"),
        sort_order=sort_order,
    )


def _query_finra(args: dict[str, object], model: str) -> dict[str, object]:
    """Analyzed briefing; bad dates get self-correcting guidance."""
    del model
    start, end, err = _finra_range(args, "query_finra")
    if err is not None:
        return err
    dataset, err = _finra_dataset(args, "query_finra", "dataset")
    if err is not None or dataset is None:
        assert err is not None
        return err
    ticker = _finra_ticker(args) or _str_or_none(args.get("symbol"))
    return finra_client.query_dataset(
        dataset,
        ticker=ticker,
        start_date=start,
        end_date=end,
        limit=_optional_int(args.get("limit")),
        offset=_optional_int(args.get("offset")),
        filters=args.get("filters"),
        analysis_goal=_str_or_none(args.get("analysis_goal")),
    )


def _get_short_interest_leaderboard(args: dict[str, object], model: str) -> dict[str, object]:
    """Most-shorted screen; bad dates get self-correcting guidance."""
    del model
    settlement_date, err = _finra_date(args.get("settlement_date"), "get_short_interest_leaderboard", "settlement_date")
    if err is not None:
        return err
    as_of, err = _finra_date(args.get("as_of"), "get_short_interest_leaderboard", "as_of")
    if err is not None:
        return err
    return screens.get_short_interest_leaderboard(
        limit=_optional_int(args.get("limit")),
        settlement_date=settlement_date,
        as_of=as_of,
    )


# FINRA dispatch registry — kept next to the FINRA tool schemas above so the
# parity test can prove every FINRA schema has an executable dispatcher.
def _describe_finra_dataset(args: dict[str, object], model: str) -> dict[str, object]:
    """Dataset metadata; bare ids get self-correcting canonical guidance."""
    del model
    key = "dataset_id" if args.get("dataset_id") is not None else "dataset"
    dataset, err = _finra_dataset(args, "describe_finra_dataset", key)
    if err is not None or dataset is None:
        assert err is not None
        return err
    return finra_client.describe_dataset(dataset)


_FINRA_HANDLERS: dict[str, ModelHandler] = {
    "get_short_interest_leaderboard": _get_short_interest_leaderboard,
    "get_short_interest": _get_short_interest,
    "get_reg_sho_volume": _get_reg_sho_volume,
    "get_threshold_securities": _get_threshold_securities,
    "list_finra_datasets": lambda args, model: finra_client.list_datasets(
        group=_str_or_none(args.get("group")), search=_str_or_none(args.get("search"))
    ),
    "describe_finra_dataset": _describe_finra_dataset,
    "get_finra_datapoints": _get_finra_datapoints,
    "query_finra": _query_finra,
}

_ROBINHOOD_HANDLERS: dict[str, ModelHandler] = {
    "get_market_snapshot": lambda args, model: get_market_snapshot(str(args["ticker"])),
    "get_option_chain": lambda args, model: get_option_chain(
        str(args["ticker"]),
        str(args["option_type"]),
        args.get("min_dte"),
        args.get("max_dte"),
        args.get("strike_min"),
        args.get("strike_max"),
        args.get("limit", 20),
    ),
    "analyze_option_contract": lambda args, model: analyze_option_contract(
        str(args["ticker"]), str(args["expiration"]), args["strike"], str(args["option_type"]), args.get("target_price")
    ),
    "compare_options": lambda args, model: compare_robinhood_options(
        str(args["ticker"]),
        str(args["option_type"]),
        args["target_price"],
        args.get("min_dte"),
        args.get("max_dte"),
        args.get("strike_min"),
        args.get("strike_max"),
        args.get("limit", 20),
    ),
    "get_portfolio_snapshot": _get_portfolio_snapshot,
    "get_scanner_filter_specs": _get_scanner_filter_specs,
    "get_scans": _get_scans,
    "run_scan": _run_scan,
}
# Every model-visible tool has one application-level capability. This is
# separate from the Robinhood MCP registry, which governs broker operations.
# Broker/account-connected market reads (Robinhood quotes/options/scans)
# require BROKER_MARKET_READ so generic research contexts never expose them;
# portfolio/account tools additionally require PORTFOLIO_READ.
TOOL_CAPABILITIES: dict[str, Capability] = {
    "evaluate_mandate": Capability.PORTFOLIO_READ,
    "get_fundamentals": Capability.RESEARCH,
    "find_sec_entities": Capability.RESEARCH,
    "find_sec_entities_bounded": Capability.RESEARCH,
    "search_sec_filings": Capability.RESEARCH,
    "search_sec_filings_bounded": Capability.RESEARCH,
    "search_sec_relationships": Capability.RESEARCH,
    "get_sec_search_coverage": Capability.RESEARCH,
    "list_sec_filings": Capability.RESEARCH,
    "get_sec_filing": Capability.RESEARCH,
    "list_sec_documents": Capability.RESEARCH,
    "get_sec_document": Capability.RESEARCH,
    "diff_sec_filings": Capability.RESEARCH,
    "get_material_events": Capability.RESEARCH,
    "get_beneficial_ownership": Capability.RESEARCH,
    "get_ownership_changes": Capability.RESEARCH,
    "get_insider_activity": Capability.RESEARCH,
    "get_planned_insider_sales": Capability.RESEARCH,
    "get_offering_history": Capability.RESEARCH,
    "get_dilution_profile": Capability.RESEARCH,
    "get_governance_events": Capability.RESEARCH,
    "get_transaction_status": Capability.RESEARCH,
    "get_short_pressure_profile": Capability.RESEARCH,
    "search_tools": Capability.RESEARCH,
    "list_tool_domains": Capability.RESEARCH,
    "describe_tool": Capability.RESEARCH,
    "browse_tools": Capability.RESEARCH,
    "call_tool": Capability.RESEARCH,
    "get_recent_ownership_filings": Capability.RESEARCH,
    "diff_risk_factors": Capability.RESEARCH,
    "get_financial_statements": Capability.RESEARCH,
    "get_xbrl_facts": Capability.RESEARCH,
    "get_short_interest": Capability.RESEARCH,
    "get_short_interest_leaderboard": Capability.RESEARCH,
    "get_reg_sho_volume": Capability.RESEARCH,
    "get_threshold_securities": Capability.RESEARCH,
    "get_analyst_estimates": Capability.RESEARCH,
    "get_sp500_weight": Capability.RESEARCH,
    "get_obligations": Capability.RESEARCH,
    "get_valuation_metrics": Capability.RESEARCH,
    "search_web": Capability.RESEARCH,
    "find_alternative_signals": Capability.RESEARCH,
    "get_trend_evidence": Capability.RESEARCH,
    "investigate_social_arbitrage_candidate": Capability.RESEARCH,
    "get_macro_context": Capability.RESEARCH,
    "search_company_patents": Capability.RESEARCH,
    "get_current_time": Capability.RESEARCH,
    "list_finra_datasets": Capability.RESEARCH,
    "describe_finra_dataset": Capability.RESEARCH,
    "get_finra_datapoints": Capability.RESEARCH,
    "query_finra": Capability.RESEARCH,
    "thesis_create": Capability.RESEARCH,
    "thesis_show": Capability.RESEARCH,
    "thesis_refine": Capability.RESEARCH,
    "thesis_watch": Capability.RESEARCH,
    "thesis_journal": Capability.RESEARCH,
    "thesis_status": Capability.RESEARCH,
    "research_start": Capability.RESEARCH,
    "research_resume": Capability.RESEARCH,
    "research_status": Capability.RESEARCH,
    "research_cancel": Capability.RESEARCH,
    "research_read": Capability.RESEARCH,
    "research_read_search": Capability.RESEARCH,
    "research_add_evidence": Capability.RESEARCH,
    "research_submit_source_result": Capability.RESEARCH,
    "research_add_analysis": Capability.RESEARCH,
    "research_finalize": Capability.RESEARCH,
    "get_market_snapshot": Capability.BROKER_MARKET_READ,
    "get_option_chain": Capability.BROKER_MARKET_READ,
    "analyze_option_contract": Capability.BROKER_MARKET_READ,
    "compare_options": Capability.BROKER_MARKET_READ,
    "get_scanner_filter_specs": Capability.BROKER_MARKET_READ,
    "get_portfolio_snapshot": Capability.PORTFOLIO_READ,
    "get_scans": Capability.PORTFOLIO_READ,
    "run_scan": Capability.PORTFOLIO_READ,
}
PORTFOLIO_AUTHORIZED_TOOLS: frozenset[str] = frozenset(
    name for name, capability in TOOL_CAPABILITIES.items() if capability is Capability.PORTFOLIO_READ
)


def tools_for_capabilities(capabilities: frozenset[Capability]) -> list[dict[str, object]]:
    """Return only schemas whose application capability is granted."""
    out: list[dict[str, object]] = []
    for tool in TOOLS:
        raw_name = _tool_function(tool).get("name")
        if isinstance(raw_name, str) and TOOL_CAPABILITIES.get(raw_name) in capabilities:
            out.append(tool)
    return out


def tool_is_permitted(name: str, context: RequestContext) -> bool:
    capability = TOOL_CAPABILITIES.get(name)
    return capability is not None and capability in context.capabilities


def _tool_schema(name: str) -> dict[str, object]:
    """Canonical schema function dict for one tool, else empty."""
    tool = next((t for t in TOOLS if _tool_function(t).get("name") == name), None)
    return _tool_function(tool) if tool is not None else {}


def _schema_dict(raw: object) -> dict[str, object]:
    """String-keyed dict from schema JSON, else empty."""
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def _required_keys(params: dict[str, object]) -> list[str]:
    """Required argument names from a parameters dict."""
    raw_required = params.get("required")
    return [str(k) for k in raw_required] if isinstance(raw_required, list) else []


def _tool_properties(name: str) -> dict[str, object]:
    """Property dict for one tool's parameters, else empty."""
    params = _schema_dict(_tool_schema(name).get("parameters"))
    return _schema_dict(params.get("properties"))


def _pattern_mismatch(name: str, key: str, value: object, pattern: str, prop: object) -> str | None:
    """Pattern violation message for a present non-blank arg, else None (blank stays handler-lenient)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        return f"Invalid argument for tool '{name}': '{key}' must be a string matching {pattern}, got {value!r}"
    if not re.fullmatch(pattern, value.strip()):
        desc = prop.get("description") if isinstance(prop, dict) else None
        tail = f"; {desc}" if isinstance(desc, str) and desc else ""
        return f"Invalid argument for tool '{name}': '{key}' {value!r} does not match {pattern}{tail}"
    return None


def _enum_mismatch(name: str, key: str, value: object, allowed: list[str]) -> str | None:
    """Enum violation message for a present non-blank arg, else None (blank stays handler-lenient)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str) and value in allowed:
        return None
    return f"Invalid argument for tool '{name}': '{key}' {value!r} is not one of {allowed}"


def _schema_value_mismatch(name: str, key: str, value: object, prop: object) -> str | None:
    """pattern/enum violation for one present argument, else None."""
    if not isinstance(prop, dict):
        return None
    if name in _SEC_GUIDED_TOOLS and key == "accession_no":
        return None  # handler owns the self-correcting message (key + format + packet source)
    pattern = prop.get("pattern")
    if isinstance(pattern, str) and pattern:
        hit = _pattern_mismatch(name, key, value, pattern, prop)
        if hit is not None:
            return hit
    allowed = prop.get("enum")
    if isinstance(allowed, list) and allowed and all(isinstance(v, str) for v in allowed):
        return _enum_mismatch(name, key, value, [str(v) for v in allowed])
    return None


_SEC_GUIDED_TOOLS = frozenset({"get_sec_filing", "get_sec_document", "list_sec_documents"})

_DATE_ARG_KEYS = frozenset(
    {
        "as_of",
        "since",
        "start_date",
        "end_date",
        "tradeDate",
        "settlementDate",
        "settlement_date",
        "start_published_date",
        "end_published_date",
    }
)
_RELATIVE_DATE_RE = re.compile(r"\b(today|now|yesterday|tomorrow)\b|this week|last week|last quarter")


def _relative_date_mismatch(name: str, key: str, value: object) -> str | None:
    """Reject today/now-style literals on date args with the YYYY-MM-DD hint."""
    if key not in _DATE_ARG_KEYS or not isinstance(value, str) or not value.strip():
        return None
    if _RELATIVE_DATE_RE.search(value.strip().lower()):
        return (
            f"Invalid argument for tool '{name}': '{key}' {value!r} is a relative date; "
            "decode it to YYYY-MM-DD first (today/now = Today UTC date)"
        )
    return None


def _validate_tool_arguments(name: str, arguments: object) -> str | None:
    """Schema-level argument check: object-ness, required keys, plus schema
    pattern/enum when present. Returns an error message, or None when the
    arguments are acceptable."""
    if not isinstance(arguments, dict):
        return f"Tool arguments must be a JSON object for tool '{name}'"
    if name in _SEC_GUIDED_TOOLS and "accession_no" not in arguments:
        return None  # handler owns the self-correcting message (key + format + packet source)
    required_keys = _required_keys(_schema_dict(_tool_schema(name).get("parameters")))
    missing = [key for key in required_keys if key not in arguments]
    if missing:
        return f"Missing required argument(s) for tool '{name}': {', '.join(missing)}"
    props = _tool_properties(name)
    for key, value in arguments.items():
        hit = _relative_date_mismatch(name, key, value)
        if hit is not None:
            return hit
        hit = _schema_value_mismatch(name, key, value, props.get(key))
        if hit is not None:
            return hit
    return None


def _canonical_tool_schema(name: str) -> tuple[dict[str, object], list[str], list[str]]:
    """Canonical parameters object plus required/optional lists via _tool_function."""
    params = _schema_dict(_tool_schema(name).get("parameters"))
    required = _required_keys(params)
    props = _schema_dict(params.get("properties"))
    return params, required, sorted(key for key in props if key not in required)


def _invalid_args_error(name: str, message: str) -> dict[str, object]:
    """Repairable validation shape reusing the canonical schema."""
    params, required, optional = _canonical_tool_schema(name)
    return {
        "error": message,
        "error_type": "invalid_tool_arguments",
        "tool": name,
        "parameters": params,
        "required": required,
        "optional": optional,
    }


def _unknown_tool_error(name: str) -> dict[str, object]:
    """Unknown-tool shape for call_tool dispatch (executes nothing)."""
    return {
        "error": f"unknown_tool '{name}'",
        "error_type": "unknown_tool",
        "tool": name,
        "hint": "call browse_tools with no arguments, then call_tool with an exact catalog name",
    }


def _thesis_repo_for(context: RequestContext) -> ThesisRepository:
    """Thesis repository rooted at the invocation's data root (never CWD)."""
    from app.thesis.repository import ThesisRepository

    base = getattr(context, "data_root", None) or get_data_root()
    return ThesisRepository(Path(str(base)) / "thesis")


def _effective_at(context: RequestContext) -> str | None:
    as_of = getattr(context, "as_of", None)
    return as_of if isinstance(as_of, str) and as_of else None


def _normalize_thesis_id(value: str) -> str:
    """Accept a bare UUID for a thesis:uuid (agents strip the prefix)."""
    text = value.strip()
    if text and ":" not in text:
        return f"thesis:{text}"
    return text


def _thesis_for_context(repo: ThesisRepository, id_or_slug: str, context: RequestContext) -> Thesis:
    """Thesis as seen at the run cutoff; live when the run has none."""
    id_or_slug = _normalize_thesis_id(id_or_slug)
    cutoff = _effective_at(context)
    if not cutoff:
        return repo.load_thesis(id_or_slug)
    from app.thesis.models import Thesis  # local: avoids a module cycle

    current = repo.load_thesis(id_or_slug)
    snap = repo.load_state_as_of(current.thesis_id, cutoff)
    return Thesis.from_dict(dict(snap.thesis), "<as_of>")


_PIT_INSTANT_TOOLS = frozenset({"thesis_show", "research_status", "research_resume", "research_read"})
_PIT_GOVERNED_MUTATORS = frozenset(
    {
        "thesis_create",
        "thesis_refine",
        "thesis_watch",
        "thesis_journal",
        "thesis_status",
        "research_start",
        "research_cancel",
        "research_add_evidence",
        "research_submit_source_result",
        "research_add_analysis",
        "research_finalize",
    }
)


def _pit_day(cutoff: str) -> str | None:

    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    dt = _as_dt(cutoff)
    if dt is None:
        return None
    return dt.astimezone(UTC).date().isoformat()


def _tool_has_as_of(name: str) -> bool:
    """Whether the tool's canonical schema accepts an as_of coordinate."""
    props = _schema_dict(_tool_schema(name).get("parameters")).get("properties")
    return isinstance(props, dict) and "as_of" in _schema_dict(props)


def _default_pit_value(
    name: str, args: dict[str, object], cutoff: str
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Blank as_of defaults: instant tools take the cutoff, others the cutoff day."""
    if name in _PIT_INSTANT_TOOLS:
        return {**args, "as_of": cutoff}, None
    day = _pit_day(cutoff)
    if day is None:
        return args, {"error": f"tool '{name}': bad run cutoff {cutoff!r}", "error_type": "invalid_tool_arguments"}
    return {**args, "as_of": day}, None


def _reject_future_as_of(
    name: str, args: dict[str, object], supplied: str, cutoff: str
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Reject a model as_of beyond the run cutoff (parseable ISO comparison)."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    supplied_dt, cutoff_dt = _as_dt(supplied), _as_dt(cutoff)
    if supplied_dt is not None and cutoff_dt is not None and supplied_dt > cutoff_dt:
        return args, {
            "error": f"tool '{name}': as_of {supplied!r} is beyond the run cutoff {cutoff!r}",
            "error_type": "invalid_tool_arguments",
        }
    return args, None


def _apply_pit_cutoff(
    name: str, arguments: object, context: RequestContext
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Default `as_of` to the run cutoff; reject a model value beyond it."""
    cutoff = _effective_at(context)
    if not isinstance(arguments, dict):
        return {}, {
            "error": f"Tool arguments must be a JSON object for tool '{name}'",
            "error_type": "invalid_tool_arguments",
        }
    args: dict[str, object] = arguments
    if not cutoff or not _tool_has_as_of(name):
        return args, None
    supplied = args.get("as_of")
    if supplied is None or (isinstance(supplied, str) and not supplied):
        return _default_pit_value(name, args, cutoff)
    if not isinstance(supplied, str):
        return args, None  # handler validation owns the message
    return _reject_future_as_of(name, args, supplied, cutoff)


def _thesis_proposal(arguments: dict[str, object], user_thesis: str, path: str) -> IntakeProposal:
    """Structured tool args -> validated IntakeProposal (raises ValueError)."""
    from app.thesis import intake as thesis_intake

    payload: dict[str, object] = {"user_thesis": user_thesis}
    for key in _THESIS_PROPOSAL_KEYS:
        if arguments.get(key) is not None:
            payload[key] = arguments[key]
    return thesis_intake.IntakeProposal.from_dict(payload, path)


def _thesis_create(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.thesis import intake as thesis_intake

    user_thesis = arguments.get("user_thesis")
    if not isinstance(user_thesis, str) or not user_thesis.strip():
        raise ValueError("thesis_create: 'user_thesis' must be a non-empty string")
    proposal = _thesis_proposal(arguments, user_thesis, "<thesis_create>")
    return thesis_intake.create_thesis_from_proposal(
        _thesis_repo_for(context), proposal, effective_at=_effective_at(context)
    )


def _reject_future_show_as_of(model_as_of: str, cutoff: str) -> None:
    """Reject a model as_of beyond the run cutoff (parseable ISO comparison)."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    model_dt, cutoff_dt = _as_dt(model_as_of), _as_dt(cutoff)
    if model_dt is not None and cutoff_dt is not None and model_dt > cutoff_dt:
        raise ValueError(f"thesis_show: as_of {model_as_of!r} is beyond the run cutoff {cutoff!r}")


def _show_as_of(arguments: dict[str, object], context: RequestContext) -> str | None:
    """Resolved as_of for show: model value wins, else the run cutoff; rejects future values."""
    model_as_of = arguments.get("as_of")
    cutoff = _effective_at(context)
    if isinstance(model_as_of, str) and model_as_of and cutoff:
        _reject_future_show_as_of(model_as_of, cutoff)
    resolved = model_as_of if isinstance(model_as_of, str) and model_as_of else cutoff
    return resolved if isinstance(resolved, str) and resolved else None


def _as_snapshot_list(raw: object) -> list[object]:
    """Snapshot list field (claims/expressions), else empty."""
    return list(raw) if isinstance(raw, list) else []


def _snapshot_rule_dicts(raw: object) -> list[dict[str, object]]:
    """Watch-rule dicts narrowed from a snapshot field."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({str(k): v for k, v in item.items()})
    return out


def _snapshot_rules(watch: dict[str, object] | dict[str, JSONValue]) -> list[dict[str, object]]:
    """Validated watch rules from a state snapshot."""
    return _snapshot_rule_dicts(watch.get("rules", []))


def _live_rules(rules: list[dict[str, object]]) -> list[dict[str, object]]:
    """Enabled + supported watch rules."""
    return [r for r in rules if r.get("enabled") and r.get("support_status") == "supported"]


def _open_snapshot_questions(questions: dict[str, object] | dict[str, JSONValue]) -> list[dict[str, object]]:
    """Open questions from a state snapshot."""
    return [q for q in _snapshot_rule_dicts(questions.get("questions", [])) if q.get("status") == "open"]


def _load_snapshot(repo: ThesisRepository, tid: str, as_of: str) -> ThesisStateSnapshot:
    """State snapshot without static repo typing gaps."""
    return repo.load_state_as_of(tid, as_of)


def _watch_rule_dicts(rules: list[WatchRule]) -> list[dict[str, object]]:
    """Watch rules serialized for the model packet."""
    out: list[dict[str, object]] = []
    for rule in rules:
        rendered = rule.to_dict()
        out.append({str(k): v for k, v in rendered.items()})
    return out


def _open_question_dicts(questions: list[ThesisQuestion]) -> list[dict[str, object]]:
    """Open questions serialized for the model packet."""
    out: list[dict[str, object]] = []
    for question in questions:
        if question.status == "open":
            rendered = question.to_dict()
            out.append({str(k): v for k, v in rendered.items()})
    return out


def _show_snapshot(repo: ThesisRepository, tid: str, as_of: str) -> dict[str, object]:
    """Show packet from a point-in-time snapshot."""
    snap = _load_snapshot(repo, tid, as_of)
    t, state, watch, questions = snap.thesis, snap.state, snap.watch, snap.questions
    rules = _snapshot_rules(watch)
    return {
        "thesis_id": tid,
        "slug": t.get("slug"),
        "status": t.get("status"),
        "user_thesis": t.get("user_thesis"),
        "scope": t.get("scope"),
        "claims": _as_snapshot_list(t.get("claims", [])),
        "expressions": _as_snapshot_list(t.get("expressions", [])),
        "assessment": state.get("assessment"),
        "rules": rules,
        "setup_needed": not _live_rules(rules),
        "open_questions": _open_snapshot_questions(questions),
    }


def _thesis_claim_dicts(thesis: Thesis) -> list[dict[str, object]]:
    """Thesis claims serialized for the model packet."""
    out: list[dict[str, object]] = []
    for claim in thesis.claims:
        rendered = claim.to_dict()
        out.append({str(k): v for k, v in rendered.items()})
    return out


def _thesis_expression_dicts(thesis: Thesis) -> list[dict[str, object]]:
    """Thesis expressions serialized for the model packet."""
    out: list[dict[str, object]] = []
    for expression in thesis.expressions:
        rendered = expression.to_dict()
        out.append({str(k): v for k, v in rendered.items()})
    return out


def _show_live(repo: ThesisRepository, thesis: Thesis, tid: str) -> dict[str, object]:
    """Show packet from live thesis state."""
    rules = _watch_rule_dicts(repo.load_watch_rules(tid))
    return {
        "thesis_id": tid,
        "slug": thesis.slug,
        "status": thesis.status,
        "user_thesis": thesis.user_thesis,
        "scope": thesis.scope,
        "claims": _thesis_claim_dicts(thesis),
        "expressions": _thesis_expression_dicts(thesis),
        "assessment": repo.load_state(tid).assessment,
        "rules": rules,
        "setup_needed": not _live_rules(rules),
        "open_questions": _open_question_dicts(repo.load_questions(tid)),
    }


def _thesis_show(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    repo = _thesis_repo_for(context)
    thesis = _thesis_for_context(repo, str(arguments["id"]), context)
    tid = thesis.thesis_id
    as_of = _show_as_of(arguments, context)
    if as_of is not None:
        return _show_snapshot(repo, tid, as_of)
    return _show_live(repo, thesis, tid)


def _refine_inputs(repo: ThesisRepository, arguments: dict[str, object], context: RequestContext) -> tuple[Thesis, str]:
    thesis_id = arguments.get("id")
    if not isinstance(thesis_id, str) or not thesis_id.strip():
        raise ValueError("thesis_refine: 'id' must be a non-empty string")
    thesis = _thesis_for_context(repo, thesis_id, context)
    clarification = arguments.get("clarification")
    if not isinstance(clarification, str) or not clarification.strip():
        raise ValueError("thesis_refine: 'clarification' must be a non-empty string")
    return thesis, clarification.strip()


def _refine_noop_result(thesis: Thesis) -> dict[str, object]:
    return {"thesis_id": thesis.thesis_id, "slug": thesis.slug, "applied": False}


def _refine_is_noop(plan: dict[str, object], thesis: Thesis) -> bool:
    merged = plan["merged"]
    merged_thesis = merged.get("user_thesis") if isinstance(merged, dict) else None
    return not plan["added_claims"] and not plan["added_expressions"] and merged_thesis == thesis.user_thesis


def _thesis_refine(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.thesis import intake as thesis_intake

    repo = _thesis_repo_for(context)
    thesis, clarification = _refine_inputs(repo, arguments, context)
    proposal = _thesis_proposal(arguments, f"{thesis.user_thesis}\n{clarification}", "<thesis_refine>")
    plan = thesis_intake.plan_refinement(thesis, proposal)
    if _refine_is_noop(plan, thesis):
        return _refine_noop_result(thesis)
    out = thesis_intake.apply_refinement(repo, thesis.thesis_id, plan, proposal, effective_at=_effective_at(context))
    return {"applied": True, **out}


def _watch_list(repo: ThesisRepository, tid: str, context: RequestContext) -> dict[str, object]:
    """Current watch rules: point-in-time snapshot under a cutoff, else live."""
    cutoff = _effective_at(context)
    if cutoff:
        snap = _load_snapshot(repo, tid, cutoff)
        rules = _snapshot_rules(snap.watch)
        return {"thesis_id": tid, "rules": rules, "setup_needed": not _live_rules(rules)}
    live_rules = _watch_rule_dicts(repo.load_watch_rules(tid))
    return {"thesis_id": tid, "rules": live_rules, "setup_needed": not _live_rules(live_rules)}


def _handler_names(handlers: Mapping[str, object]) -> list[str]:
    """Supported rule_type names narrowed from the handlers mapping."""
    return sorted(handlers)


def _checked_rule_type(arguments: dict[str, object], handlers: Mapping[str, object]) -> str:
    """Validated watch rule_type (non-empty, supported)."""
    rule_type = arguments["rule_type"]
    if not isinstance(rule_type, str) or not rule_type.strip():
        raise ValueError("thesis_watch: 'rule_type' must be a non-empty string")
    if not isinstance(handlers, dict) or rule_type not in handlers:
        raise ValueError(f"thesis_watch: unsupported rule_type {rule_type!r}; supported: {_handler_names(handlers)}")
    checked: str = rule_type
    return checked


def _checked_id_list(arguments: dict[str, object], key: str) -> list[str]:
    """Validated claim/expression id list (list of strings, defaults empty)."""
    vals = arguments.get(key, [])
    if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
        raise ValueError(f"thesis_watch: '{key}' must be a list of IDs")
    raw = arguments.get(key, [])
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _apply_watch_rule(repo: ThesisRepository, tid: str, rule: dict[str, object], context: RequestContext) -> None:
    """Append one watch rule without static repo typing gaps."""
    repo.apply_research_result(tid, {"watch_add": [rule]}, "", effective_at=_effective_at(context))


def _watch_add(
    repo: ThesisRepository,
    thesis: Thesis,
    arguments: dict[str, object],
    context: RequestContext,
    handlers: Mapping[str, object],
) -> dict[str, object]:
    """Validate and append one watch rule to an active thesis."""
    from app.thesis.models import new_rule_id

    tid = thesis.thesis_id
    rule_type = _checked_rule_type(arguments, handlers)
    if thesis.status != "active":
        raise ValueError(f"thesis {tid!r} is {thesis.status}; refusing watch change")
    rule: dict[str, object] = {
        "rule_id": new_rule_id(),
        "rule_type": rule_type,
        "enabled": True,
        "support_status": "supported",
        "support_reason": "",
        "claim_ids": _checked_id_list(arguments, "claim_ids"),
        "expression_ids": _checked_id_list(arguments, "expression_ids"),
    }
    _apply_watch_rule(repo, tid, rule, context)
    return {"thesis_id": tid, "added": rule}


def _thesis_watch(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.thesis.monitor import SUPPORTED_HANDLERS

    repo = _thesis_repo_for(context)
    thesis = _thesis_for_context(repo, str(arguments["id"]), context)
    if arguments.get("rule_type") is None:
        return _watch_list(repo, thesis.thesis_id, context)
    return _watch_add(repo, thesis, arguments, context, SUPPORTED_HANDLERS)


def _checked_journal_body(arguments: dict[str, object]) -> str:
    """Validated journal body (non-empty, stripped)."""
    body = arguments["body"]
    if not isinstance(body, str) or not body.strip():
        raise ValueError("thesis_journal: 'body' must be a non-empty string")
    return body.strip()


def _checked_journal_title(arguments: dict[str, object]) -> str:
    """Validated journal title (string, defaults to Operator note)."""
    title = arguments.get("title", "Operator note")
    if title is not None and not isinstance(title, str):
        raise ValueError("thesis_journal: 'title' must be a string")
    return title or "Operator note"


def _trigger_ids(repo: ThesisRepository, thesis_id: str) -> list[str]:
    """Trigger ids narrowed for the membership check."""
    return [trigger.trigger_id for trigger in repo.load_triggers(thesis_id)]


def _checked_trigger(repo: ThesisRepository, thesis_id: str, trigger_id: str) -> str:
    """Trigger belongs to the thesis, else raise."""
    if not isinstance(trigger_id, str) or not trigger_id:
        raise ValueError("thesis_journal: 'trigger_id' must be a non-empty string")
    if trigger_id not in _trigger_ids(repo, thesis_id):
        raise ValueError(f"thesis_journal: trigger {trigger_id!r} does not belong to thesis {thesis_id!r}")
    return trigger_id


def _checked_run_id(arguments: dict[str, object], run_id: str) -> str:
    """run_id requires trigger_id and a non-empty string."""
    if arguments.get("trigger_id") is None:
        raise ValueError("thesis_journal: 'run_id' requires 'trigger_id'")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("thesis_journal: 'run_id' must be a non-empty string")
    return run_id


def _checked_known_at(known_at: object) -> str:
    """Parseable ISO-8601 known_at, else raise."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    if not isinstance(known_at, str) or not known_at or _as_dt(known_at) is None:
        raise ValueError("thesis_journal: 'known_at' must be a parseable ISO-8601 string")
    return known_at


def _trigger_known_at(known_at: object, cutoff: str, entry: dict[str, object]) -> None:
    """Trigger-linked known_at under a cutoff: required and equal to the cutoff."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    if not isinstance(known_at, str) or not known_at or _as_dt(known_at) is None:
        raise ValueError(
            "thesis_journal: 'known_at' is required for trigger-linked entries and must be a parseable ISO-8601 string"
        )
    known_dt, cutoff_dt = _as_dt(known_at), _as_dt(cutoff)
    if (known_dt is not None or cutoff_dt is not None) and known_dt != cutoff_dt:
        raise ValueError(f"thesis_journal: 'known_at' {known_at!r} must equal the run cutoff {cutoff!r}")
    entry["known_at"] = known_at


def _attach_journal_trigger(
    repo: ThesisRepository, thesis_id: str, arguments: dict[str, object], entry: dict[str, object]
) -> str | None:
    """Trigger/run linkage: trigger_id validated, run_id attached when present."""
    trigger_id = arguments.get("trigger_id")
    if trigger_id is None:
        return None
    assert isinstance(trigger_id, str)
    entry["trigger_id"] = _checked_trigger(repo, thesis_id, trigger_id)
    run_id = arguments.get("run_id")
    if run_id is not None:
        assert isinstance(run_id, str)
        entry["run_id"] = _checked_run_id(arguments, run_id)
    trigger = entry["trigger_id"]
    if isinstance(trigger, str):
        return trigger
    raise TypeError(f"trigger_id must be a string, got {type(trigger).__name__}")


def _attach_journal_known_at(
    trigger_id: str | None, arguments: dict[str, object], context: RequestContext, entry: dict[str, object]
) -> None:
    """known_at: cutoff-equal for trigger-linked entries under PIT, else plain parseable."""
    known_at = arguments.get("known_at")
    if trigger_id is not None:
        cutoff = _effective_at(context)
        if cutoff:
            _trigger_known_at(known_at, cutoff, entry)
        elif known_at is not None:
            entry["known_at"] = _checked_known_at(known_at)
    elif known_at is not None:
        entry["known_at"] = _checked_known_at(known_at)


def _thesis_journal(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    repo = _thesis_repo_for(context)
    thesis = _thesis_for_context(repo, str(arguments["id"]), context)
    if thesis.status != "active":
        raise ValueError(f"thesis {thesis.thesis_id!r} is {thesis.status}; refusing journal append")
    entry: dict[str, object] = {
        "title": _checked_journal_title(arguments),
        "body": _checked_journal_body(arguments),
    }
    trigger_id = _attach_journal_trigger(repo, thesis.thesis_id, arguments, entry)
    _attach_journal_known_at(trigger_id, arguments, context, entry)
    dest = repo.append_journal_entry(thesis.thesis_id, entry)
    return {"thesis_id": thesis.thesis_id, "journal_path": str(dest)}


def _checked_status_id(arguments: dict[str, object]) -> str:
    """Validated thesis id for status transitions (non-empty, stripped)."""
    thesis_id = arguments.get("id")
    if not isinstance(thesis_id, str) or not thesis_id.strip():
        raise ValueError("thesis_status: 'id' must be a non-empty string")
    return thesis_id.strip()


def _checked_status_action(arguments: dict[str, object]) -> str:
    """Validated status action (pause/resume/close)."""
    action = arguments.get("action")
    if not isinstance(action, str) or action not in ("pause", "resume", "close"):
        raise ValueError("thesis_status: 'action' must be one of pause, resume, close")
    return action


def _thesis_status(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    repo = _thesis_repo_for(context)
    thesis_id = _checked_status_id(arguments)
    action = _checked_status_action(arguments)
    op = {"pause": repo.pause_thesis, "resume": repo.resume_thesis, "close": repo.close_thesis}[action]
    updated = op(thesis_id, effective_at=_effective_at(context))
    return {"thesis_id": updated.thesis_id, "slug": updated.slug, "status": updated.status, "action": action}


def _research_repo_for(context: RequestContext):
    """Research repository rooted at the invocation's data root (never CWD)."""
    from app.research.repository import ResearchRepository

    return ResearchRepository(data_root=context.data_root)


def _research_not_found_error(exc: Exception) -> dict[str, object]:
    """Named unknown-id error; never raises to the model beyond the envelope."""
    raw = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
    message = raw or "unknown research id"
    error_type = "unknown_job" if "job_id" in message else "unknown_session"
    return {"error": message, "error_type": error_type}


def _start_question(arguments: dict[str, object]) -> str:
    """Validated research question (non-empty, stripped)."""
    question = arguments.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("research_start: 'question' must be a non-empty string")
    return question.strip()


def _start_objective(arguments: dict[str, object]) -> str | None:
    """Optional research objective (non-blank string, else None)."""
    objective = arguments.get("objective")
    return objective.strip() if isinstance(objective, str) and objective.strip() else None


def _start_as_of(arguments: dict[str, object]) -> str | None:
    """Optional as_of coordinate (non-blank string, else None)."""
    as_of = arguments.get("as_of")
    return as_of if isinstance(as_of, str) and as_of else None


def _start_policy(arguments: dict[str, object]) -> object:
    """Optional session policy passthrough; None stays the SEC-only default."""
    policy = arguments.get("policy")
    if isinstance(policy, dict):
        return {str(k): v for k, v in policy.items()}
    research_sources = arguments.get("research_sources")
    sources = arguments.get("sources")
    if isinstance(research_sources, dict):
        return {"research_sources": dict(research_sources)}
    if isinstance(sources, list) and all(isinstance(s, str) and s.strip() for s in sources):
        return {
            "research_sources": {"mode": "allowlist", "sources": [s.strip() for s in sources if isinstance(s, str)]}
        }
    return None


def _inspect_research_snapshot(research_service: object, session_id: str, repo: object) -> dict[str, object]:
    """inspect_research without static service typing."""
    inspect = getattr(research_service, "inspect_research", None)
    if not callable(inspect):
        raise TypeError(f"research service must expose inspect_research, got {type(research_service).__name__}")
    snapshot = inspect(session_id, repo=repo)
    if not isinstance(snapshot, dict):
        raise TypeError(f"inspect_research must return a dict, got {type(snapshot).__name__}")
    return {str(k): v for k, v in snapshot.items()}


def _started_packet(research_service: object, repo: object, session_id: str) -> dict[str, object]:
    """Start packet: session id, first job id, status, and next action."""
    snapshot = _inspect_research_snapshot(research_service, session_id, repo)
    raw_jobs = snapshot.get("jobs")
    jobs: list[object] = list(raw_jobs) if isinstance(raw_jobs, list) else []
    first_raw = jobs[0] if jobs else None
    first: dict[str, object] = dict(first_raw) if isinstance(first_raw, dict) else {}
    raw_session = snapshot.get("session")
    status = raw_session.get("status") if isinstance(raw_session, dict) else None
    return {
        "session_id": session_id,
        "job_id": first.get("job_id"),
        "status": status,
        "pending_next_action": snapshot.get("pending_next_action"),
    }


def _research_start(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service

    repo = _research_repo_for(context)
    session_id = research_service.create_research(  # type: ignore[arg-type]
        _start_question(arguments),
        _start_objective(arguments),
        as_of=_start_as_of(arguments),
        policy=_start_policy(arguments),  # type: ignore[arg-type]
        repo=repo,
    )
    return _started_packet(research_service, repo, session_id)


def _research_resume(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    try:
        return dict(research_service.resume_research(session_id, repo=_research_repo_for(context)))
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


def _research_status(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    try:
        return dict(research_service.inspect_research(session_id, repo=_research_repo_for(context)))
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


def _research_cancel(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    try:
        return dict(research_service.cancel_research(session_id, repo=_research_repo_for(context)))
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


_RESEARCH_KINDS = ("evidence", "freeze", "dossier", "coverage", "job", "research")


def _checked_resource_kind(arguments: dict[str, object]) -> str:
    """Validated research resource kind, else an unknown_resource error is raised by the caller."""
    kind = arguments.get("kind")
    if not isinstance(kind, str) or kind not in _RESEARCH_KINDS:
        raise ValueError(f"unknown resource kind: {kind!r} (expected evidence|freeze|dossier|coverage|job|research)")
    return kind


def _unknown_resource_error(kind: str, resource_id: str, session_id: str) -> dict[str, object]:
    """Unknown id within a known store: jobs get unknown_job, others unknown_resource."""
    if kind == "job":
        return {"error": f"unknown job_id: {resource_id!r} in session {session_id!r}", "error_type": "unknown_job"}
    return {"error": f"unknown {kind} id: {resource_id!r} in session {session_id!r}", "error_type": "unknown_resource"}


def _record_evidence_ids(record: object) -> set[str]:
    """Non-empty evidence ids a stored freeze or dossier record carries."""
    if not isinstance(record, Mapping):
        return set()
    ids: set[str] = set()
    for key in ("evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids"):
        raw = record.get(key)
        if isinstance(raw, list):
            ids.update(v for v in raw if isinstance(v, str) and v)
    return ids


def _freeze_scope_error(
    kind: str, resource_id: str, record: object, session_id: str, freeze_id: object, freezes: Mapping[str, object]
) -> dict[str, object] | None:
    """Freeze-scoped read gate: unknown freeze, or evidence/dossier outside its membership, fails closed."""
    if kind not in ("evidence", "dossier") or not isinstance(freeze_id, str) or not freeze_id:
        return None
    freeze = freezes.get(freeze_id)
    if not isinstance(freeze, Mapping):
        return _unknown_resource_error("freeze", freeze_id, session_id)
    frozen = _record_evidence_ids(freeze)
    if kind == "evidence":
        if resource_id in frozen:
            return None
        return {
            "error": f"evidence id {resource_id!r} is not a member of freeze {freeze_id!r}",
            "error_type": "not_in_freeze",
        }
    outside = sorted(_record_evidence_ids(record) - frozen)
    if not outside:
        return None
    return {
        "error": f"dossier id {resource_id!r} cites evidence outside freeze {freeze_id!r}: {', '.join(outside)}",
        "error_type": "not_in_freeze",
    }


def _research_read(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    try:
        kind = _checked_resource_kind(arguments)
    except ValueError as exc:
        return {"error": str(exc), "error_type": "unknown_resource"}
    resource_id = str(arguments.get("resource_id"))
    try:
        stores = _research_repo_for(context).resource_stores(session_id)
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)
    store = stores[kind]
    if resource_id not in store:
        return _unknown_resource_error(kind, resource_id, session_id)
    scoped = _freeze_scope_error(
        kind, resource_id, store[resource_id], session_id, arguments.get("freeze_id"), stores.get("freeze", {})
    )
    if scoped is not None:
        return scoped
    record: object = store[resource_id]
    if kind == "evidence" and isinstance(record, Mapping):
        from app.research.evidence import evidence_domain, evidence_integrity

        prov = record.get("provenance")
        prov_map = prov if isinstance(prov, Mapping) else None
        filled = dict(record)
        filled["source_domain"] = evidence_domain(prov_map)
        filled["integrity_class"] = evidence_integrity(prov_map)
        record = filled
    return {"session_id": session_id, "kind": kind, "resource_id": resource_id, "record": record}


_READ_SEARCH_DEFAULT_LIMIT = 50
_READ_SEARCH_MAX_LIMIT = 500


def _read_search_count(arguments: dict[str, object], key: str, default: int, minimum: int) -> int:
    """Validated int page argument (bools are never counts)."""
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"research_read_search: '{key}' must be an integer >= {minimum}, got {value!r}")
    return value


def _read_search_forms(arguments: dict[str, object]) -> tuple[str, ...] | None:
    """Validated form filter of the page window (None when absent)."""
    forms = arguments.get("forms")
    if forms is None:
        return None
    if not isinstance(forms, (list, tuple)) or any(not isinstance(f, str) for f in forms):
        raise ValueError(f"research_read_search: 'forms' must be a list of strings, got {type(forms).__name__}")
    return tuple(forms)


def _read_search_page(arguments: dict[str, object]) -> tuple[int, int, tuple[str, ...] | None]:
    """Validated (offset, limit, forms) page window for research_read_search."""
    offset = _read_search_count(arguments, "offset", 0, 0)
    limit = min(_read_search_count(arguments, "limit", _READ_SEARCH_DEFAULT_LIMIT, 1), _READ_SEARCH_MAX_LIMIT)
    return offset, limit, _read_search_forms(arguments)


def _research_read_search(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    """No persisted SEC search universe remains; every read is unknown_search."""
    # Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind live readers, never here.
    session_id = str(arguments["session_id"])
    search_id = str(arguments["search_id"])
    _read_search_page(arguments)
    try:
        _research_repo_for(context).get_session(session_id)
    except KeyError as e:
        return _research_not_found_error(e)
    return {"error": f"unknown search_id: {search_id!r}", "error_type": "unknown_search"}


def _research_add_evidence(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    job_id = str(arguments["job_id"])
    item = arguments["item"]
    if not isinstance(item, dict):
        raise ValueError(f"research_add_evidence: 'item' must be an object, got {type(item).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    try:
        return dict(research_service.record_evidence(session_id, job_id, item, repo=_research_repo_for(context)))
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


def _submit_coverage(arguments: dict[str, object]) -> dict[str, object]:
    """Validated coverage mapping for submit_source_result."""
    coverage = arguments["coverage"]
    if not isinstance(coverage, dict):
        raise ValueError(f"research_submit_source_result: 'coverage' must be an object, got {type(coverage).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return coverage


def _submit_str_list(arguments: dict[str, object], key: str) -> list[str]:
    """Validated string list for submit_source_result (evidence_ids/unresolved)."""
    values = arguments[key]
    if not isinstance(values, list) or any(not isinstance(e, str) for e in values):
        raise ValueError(f"research_submit_source_result: '{key}' must be a list of strings")
    return values


def _submit_job_known(state: dict[str, object], job_id: str) -> bool:
    """Whether the session snapshot contains the job."""
    raw_jobs = state.get("jobs")
    jobs: list[object] = list(raw_jobs) if isinstance(raw_jobs, list) else []
    return any(isinstance(j, dict) and j.get("job_id") == job_id for j in jobs)


def _research_submit_source_result(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    job_id = str(arguments["job_id"])
    coverage = _submit_coverage(arguments)
    evidence_ids = _submit_str_list(arguments, "evidence_ids")
    unresolved = _submit_str_list(arguments, "unresolved_questions")
    try:
        state = research_service.inspect_research(session_id, repo=_research_repo_for(context))
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)
    if not _submit_job_known(state, job_id):
        return {"error": f"unknown job_id: {job_id!r} in session {session_id!r}", "error_type": "unknown_job"}
    try:
        return dict(
            research_service.submit_source_result(
                job_id, dict(coverage), list(evidence_ids), list(unresolved), repo=_research_repo_for(context)
            )
        )
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


def _analysis_role(arguments: dict[str, object]) -> str:
    """Validated committee role (stockbot/bullbot/bearbot)."""
    role = str(arguments["role"])
    if role not in ("stockbot", "bullbot", "bearbot"):
        raise ValueError(f"research_add_analysis: role must be stockbot|bullbot|bearbot, got {role!r}")
    return role


def _analysis_payload(arguments: dict[str, object]) -> dict[str, object]:
    """Validated committee analysis mapping."""
    analysis = arguments["analysis"]
    if not isinstance(analysis, dict):
        raise ValueError(f"research_add_analysis: 'analysis' must be an object, got {type(analysis).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return analysis


def _research_add_analysis(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    job_id = str(arguments["job_id"])
    role = _analysis_role(arguments)
    analysis = _analysis_payload(arguments)
    try:
        return dict(
            research_service.record_committee_analysis(
                session_id, job_id, role, analysis, repo=_research_repo_for(context)
            )
        )
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


def _finalize_answer(arguments: dict[str, object]) -> str:
    """Validated finalize answer (non-empty string)."""
    answer = arguments["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("research_finalize: 'answer' must be a non-empty string")
    return answer


def _finalize_claim_ids(evidence_ids: object) -> list[str]:
    """Validated claim evidence id list (non-empty strings)."""
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError("research_finalize: each claim 'evidence_ids' must be a non-empty list of non-empty strings")
    for item in evidence_ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "research_finalize: each claim 'evidence_ids' must be a non-empty list of non-empty strings"
            )
    return evidence_ids


def _finalize_claim_text(claim: object) -> None:
    """One claim has a non-empty text."""
    text = claim.get("text", "") if isinstance(claim, dict) else ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("research_finalize: each claim 'text' must be a non-empty string")


def _finalize_claims(arguments: dict[str, object]) -> list[object]:
    """Validated finalize claims (non-empty list of grounded claims)."""
    claims = arguments["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("research_finalize: 'claims' must be a non-empty list")
    for claim in claims:
        _finalize_claim_text(claim)
        _finalize_claim_ids(claim.get("evidence_ids", []) if isinstance(claim, dict) else [])
    return claims


def _research_finalize(arguments: dict[str, object], context: RequestContext) -> dict[str, object]:
    from app.research import service as research_service
    from app.research.service import ResearchNotFound

    session_id = str(arguments["session_id"])
    try:
        return dict(
            research_service.finalize_session(
                session_id, _finalize_answer(arguments), _finalize_claims(arguments), repo=_research_repo_for(context)
            )
        )
    except ValueError as e:
        return {"error": str(e)}
    except (ResearchNotFound, KeyError) as e:
        return _research_not_found_error(e)


_THESIS_HANDLERS: dict[str, ContextHandler] = {
    "thesis_create": _thesis_create,
    "thesis_show": _thesis_show,
    "thesis_refine": _thesis_refine,
    "thesis_watch": _thesis_watch,
    "thesis_journal": _thesis_journal,
    "thesis_status": _thesis_status,
}

_RESEARCH_HANDLERS: dict[str, ContextHandler] = {
    "research_start": _research_start,
    "research_resume": _research_resume,
    "research_status": _research_status,
    "research_cancel": _research_cancel,
    "research_read": _research_read,
    "research_read_search": _research_read_search,
    "research_add_evidence": _research_add_evidence,
    "research_submit_source_result": _research_submit_source_result,
    "research_add_analysis": _research_add_analysis,
    "research_finalize": _research_finalize,
}

_SEC_DISCOVERY_HANDLERS: dict[str, ContextHandler] = {
    "find_sec_entities": _find_sec_entities,
    "find_sec_entities_bounded": _find_sec_entities_bounded,
    "search_sec_filings": _sec_search_result,
    "search_sec_filings_bounded": _sec_search_result_bounded,
}

# Thesis/research tools are direct local dispatch (no broker) but take
# (arguments, context) instead of (arguments, model) for data-root scoping.
# SEC discovery takes (arguments, context) for the research-session default
# (exhaustive retrieval); its handlers stay out of _RESEARCH_HANDLERS because
# they hit provider seams, not local-only state.
# Merged view for backward compat (tests/scripts import _DIRECT_HANDLERS).
_DIRECT_HANDLERS: dict[str, object] = {
    **_MODEL_HANDLERS,
    **_THESIS_HANDLERS,
    **_RESEARCH_HANDLERS,
    **_SEC_DISCOVERY_HANDLERS,
}
_CONTEXT_CALL_HANDLERS = (
    frozenset(_THESIS_HANDLERS) | frozenset(_RESEARCH_HANDLERS) | frozenset(_SEC_DISCOVERY_HANDLERS)
)


def _permission_error(name: str, context: RequestContext) -> dict[str, object] | None:
    """Not-permitted envelope, else None."""
    if not tool_is_permitted(name, context):
        return {"error": f"Tool is not permitted: {name}"}
    return None


def _pit_unsafe_error(name: str, context: RequestContext) -> dict[str, object] | None:
    """Historical runs reject tools without an as_of coordinate (governed mutators exempt)."""
    if _effective_at(context) and name not in _PIT_GOVERNED_MUTATORS and not _tool_has_as_of(name):
        return {
            "error": f"Tool '{name}' is not point-in-time safe under this historical run.",
            "error_type": "pit_unsafe_tool",
            "soft": True,
        }
    return None


def _lookup_context_handler(name: str) -> ContextHandler | None:
    """Thesis/research/SEC-discovery handler for context-dispatched tools, else None."""
    return _THESIS_HANDLERS.get(name) or _RESEARCH_HANDLERS.get(name) or _SEC_DISCOVERY_HANDLERS.get(name)


def _lookup_model_handler(name: str) -> ModelHandler | None:
    """Model handler across the model/FINRA/Robinhood maps, else None."""
    return _MODEL_HANDLERS.get(name) or _FINRA_HANDLERS.get(name) or _ROBINHOOD_HANDLERS.get(name)


def _with_pit_flag(name: str, result: dict[str, object], context: RequestContext) -> dict[str, object]:
    """Mark non-as_of results pit_safe=False under a historical run (governed mutators exempt)."""
    if (
        _effective_at(context)
        and isinstance(result, dict)
        and not _tool_has_as_of(name)
        and name not in _PIT_GOVERNED_MUTATORS
    ):
        result.setdefault("pit_safe", False)
    return result


def _dispatch_tool(name: str, arguments: dict[str, object], model: str, context: RequestContext) -> dict[str, object]:
    """Validated dispatch: context handlers, then model handlers, else unknown-tool."""
    if name in _CONTEXT_CALL_HANDLERS:
        ctx_handler = _lookup_context_handler(name)
        if ctx_handler is None:
            return _unknown_tool_error(name)
        return ctx_handler(arguments, context)
    model_handler = _lookup_model_handler(name)
    if model_handler is None:
        return _unknown_tool_error(name)
    return _with_pit_flag(name, model_handler(arguments, model), context)


def _robinhood_failure(name: str) -> dict[str, object]:
    """Robinhood provider failure without request identifiers (never logged/sent)."""
    # Provider errors can echo request arguments. Do not log
    # exception details (no account identifiers in logs) or place
    # them in a tool message that is subsequently sent to the LLM.
    logger.warning("Robinhood tool '%s' failed; provider details withheld", name)
    return {"error": f"Robinhood tool '{name}' failed; provider details withheld."}


def _auth_required() -> dict[str, object]:
    """Robinhood OAuth soft failure (setup step, not an error)."""
    return {
        "error": "Robinhood data unavailable (not authorized). Run `cli.py robinhood-login` to authorize.",
        "error_type": "auth_required",
        "soft": True,
        "source": "robinhood_mcp",
    }


def execute_tool(
    name: str,
    arguments: dict[str, object],
    model: str,
    *,
    context: RequestContext,
) -> dict[str, object]:
    """Dispatch a tool call by name. Always returns a JSON-serializable dict;
    never raises — errors are returned as {"error": ...} so the model can
    report them honestly (guardrail behavior)."""
    try:
        denied = _permission_error(name, context)
        if denied is not None:
            return denied
        arguments, pit_error = _apply_pit_cutoff(name, arguments, context)
        if pit_error is not None:
            return pit_error
        unsafe = _pit_unsafe_error(name, context)
        if unsafe is not None:
            return unsafe
        invalid = _validate_tool_arguments(name, arguments)
        if invalid is not None:
            return _invalid_args_error(name, invalid)
        return _dispatch_tool(name, arguments, model, context)
    except KeyError as e:
        return {"error": f"Missing required argument {e} for tool '{name}'"}
    except RobinhoodAuthRequired:
        logger.info("Robinhood tool '%s' not authorized; soft failure", name)
        return _auth_required()
    except Exception as e:
        if name in _ROBINHOOD_HANDLERS:
            return _robinhood_failure(name)
        logger.exception("Tool '%s' failed", name)
        return {"error": f"Tool '{name}' failed: {e}"}


validate_tool_discovery_registry()
