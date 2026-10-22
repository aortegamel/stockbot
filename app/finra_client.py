"""FINRA Query API client with filing-cabinet catalog + metadata discovery.

tools.py never talks to FINRA HTTP endpoints directly.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from operator import itemgetter
from pathlib import Path
from typing import TypeGuard

import requests

from . import cache
from .config import (
    FINRA_API_BASE,
    FINRA_TOKEN_URL,
    finra_use_mock,
    get_finra_client_id,
    get_finra_client_secret,
)
from .finra_analysis import analyze_and_brief
from .storage import raw_archive

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
MAX_OFFSET = 500_000
CACHE_TTL_SECONDS = 3600
DISCOVERY_TTL_SECONDS = 86400
TOKEN_SKEW_SECONDS = 60

DATAPOINTS_DEFAULT_LIMIT = 10
DATAPOINTS_MAX_LIMIT = 25
DATAPOINTS_MAX_FIELDS = 10

# FINRA record-pagination headers (requests lowercases header names).
_RECORD_HEADERS = (
    "record-total",
    "record-offset",
    "record-limit",
    "record-max-limit",
)

_ALLOWED_COMPARE = {
    "EQUAL",
    "GREATER",
    "LESSER",
    "GTE",
    "LTE",
    "NOT_EQUAL",
    "BEGINS_WITH",
}

# Prefer these exact field names when detecting a ticker/symbol column.
_PREFERRED_SYMBOL_FIELDS = (
    "symbolCode",
    "issueSymbolIdentifier",
    "securitiesInformationProcessorSymbolIdentifier",
    "oldSymbolCode",
)

_WEEKLY_SUMMARY_TYPES = (
    "OTC_W_FIRM",
    "OTC_W_SMBL",
    "OTC_W_SMBL_FIRM",
    "OTC_W_VOL_STATS",
    "ATS_W_FIRM",
    "ATS_W_SMBL",
    "ATS_W_SMBL_FIRM",
    "ATS_W_VOL_STATS",
)

# Datasets whose authoritative/displayed date field differs from the date
# partition field. Values are verified partition fields used to order
# partition walking; walking is rejected when no verified mapping exists.
_DATE_PARTITION_MAPPINGS: dict[tuple[str, str], str] = {
    ("otcmarket", "weeklysummary"): "weekStartDate",
    ("otcmarket", "weeklysummaryhistoric"): "weekStartDate",
}

# Corrections applied on top of live catalog/metadata before exposure.
# Keys are lowercase (group, name).
_METADATA_OVERRIDES: dict[tuple[str, str], dict[str, object]] = {
    ("fixedincomemarket", "treasurydailyaggregates"): {
        "market_aggregate": True,
        "symbol_field": None,
        "date_field": "tradeDate",
    },
    ("fixedincomemarket", "treasurymonthlyaggregates"): {
        "market_aggregate": True,
        "symbol_field": None,
        "date_field": "beginningOfTheMonthDate",
    },
    ("otcmarket", "weeklysummary"): {
        "symbol_field": "issueSymbolIdentifier",
        "date_field": "summaryStartDate",
        "default_filters": (("summaryTypeCode", "OTC_W_SMBL"),),
        "valid_filter_values": {"summaryTypeCode": _WEEKLY_SUMMARY_TYPES},
    },
    ("otcmarket", "weeklysummaryhistoric"): {
        "symbol_field": "issueSymbolIdentifier",
        "date_field": "summaryStartDate",
        "default_filters": (("summaryTypeCode", "OTC_W_SMBL"),),
        "valid_filter_values": {"summaryTypeCode": _WEEKLY_SUMMARY_TYPES},
    },
    ("otcmarket", "monthlysummary"): {
        "symbol_field": "issueSymbolIdentifier",
        "date_field": "summaryStartDate",
    },
    ("otcmarket", "consolidatedshortinterest"): {
        "symbol_field": "symbolCode",
        "date_field": "settlementDate",
    },
    ("otcmarket", "regshodaily"): {
        "symbol_field": "securitiesInformationProcessorSymbolIdentifier",
        "date_field": "tradeReportDate",
    },
    ("otcmarket", "thresholdlist"): {
        "symbol_field": "issueSymbolIdentifier",
        "date_field": "tradeDate",
    },
    ("otcmarket", "otcdailylist"): {
        "symbol_field": "oldSymbolCode",
        "date_field": "dailyListDatetime",
    },
}

# Legacy bare names kept only for resolution hints / error messages.
DATASET_NAMES = (
    "consolidatedShortInterest",
    "regShoDaily",
    "thresholdList",
    "weeklySummary",
    "weeklySummaryHistoric",
    "monthlySummary",
    "blocksSummary",
    "otcBlocksSummary",
    "otcDailyList",
    "agencyTbaPricing",
    "agencyCmoPricing",
    "agencyMarketBreadth",
    "agencyMarketSentiment",
    "agencyMbsTradingActivity",
    "agencyMbsArmHybridPricing",
    "agencyMbsPricing",
    "collateralizedObligationPricing",
    "corporate144AMarketBreadth",
    "corporate144AMarketSentiment",
    "corporatesAndAgenciesCappedVolume",
    "corporateMarketBreadth",
    "corporateMarketSentiment",
    "dailyCmbsPricing",
    "weeklyCmbsPricing",
    "nonAgencyCmoAbsPricing",
    "nonAgencyCmoVintagePricing",
    "securitizedProductsCappedVolume",
    "securitizedProductErrata",
    "securitizedProductTradingActivity",
    "treasuryDailyAggregates",
    "treasuryMonthlyAggregates",
    "industrySnapshotFirmsByRegistrationType",
)

# FINRA's live catalog may return path segments in all caps even though the
# documented data, metadata, and partitions endpoints use camelCase. Keep a
# conservative registry for known paths; unknown values remain unchanged.
_CANONICAL_GROUP_NAMES = {
    "adf": "adf",
    "finra": "finra",
    "firm": "firm",
    "fixedincomemarket": "fixedIncomeMarket",
    "otcmarket": "otcMarket",
    "registration": "registration",
}
_CANONICAL_DATASET_NAMES = {name.casefold(): name for name in DATASET_NAMES}


@dataclass(frozen=True)
class CatalogEntry:
    group: str
    name: str
    description: str
    methods: tuple[str, ...] = ()
    supports_query: bool = True
    status: str = ""
    access: str = "unknown"
    supports_record_offset: bool | None = None

    @property
    def dataset_id(self) -> str:
        return f"{self.group}/{self.name}"


@dataclass(frozen=True)
class DatasetSpec:
    group: str
    name: str
    description: str
    fields: tuple[dict[str, object], ...] = ()
    partition_fields: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    symbol_field: str | None = None
    date_field: str | None = None
    market_aggregate: bool = False
    default_filters: tuple[tuple[str, str], ...] = ()
    valid_filter_values: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def dataset_id(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def field_names(self) -> frozenset[str]:
        names: list[str] = []
        for f in self.fields:
            candidate: object = f.get("name")
            if isinstance(candidate, str) and candidate:
                names.append(candidate)
        return frozenset(names)


_token_lock = threading.Lock()
_cached_token: str | None = None
_token_expires_at: float = 0.0

_discovery_lock = threading.Lock()
_catalog_mem: dict[str, list[CatalogEntry]] = {}
_metadata_mem: dict[tuple[str, str], DatasetSpec] = {}
_partitions_mem: dict[tuple[str, str], list[tuple[str, ...]]] = {}


def _dataset_rank(row: dict[str, object]) -> tuple[int, str]:
    """Newest-match-first ordering for token-ranked catalog entries."""
    return (-int(str(row.get("match_score", 0))), str(row.get("name", "")))


def list_datasets(group: str | None = None, search: str | None = None) -> dict[str, object]:
    """Concise filing-cabinet catalog. Never returns data rows.

    search is token-based and ranked: the phrase is normalized (e.g.
    "trading volume" -> "volume", OTC/ATS/weekly aliases) and every entry is
    scored by how many distinct query tokens match its group/name/description.
    Matched entries are returned best-first; entries with no token match are
    omitted.
    """
    try:
        entries = _get_catalog()
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"error": _catalog_error_message(e)}

    group_filter = (group or "").strip().lower() or None
    tokens = _search_tokens(search)

    datasets: list[dict[str, object]] = []
    for entry in entries:
        if group_filter and entry.group.lower() != group_filter:
            continue
        score = _list_match_score(entry, tokens)
        if score is None:
            continue
        datasets.append(_list_dataset_row(entry, score))

    if tokens:
        datasets.sort(key=_dataset_rank)
        for d in datasets:
            d.pop("match_score", None)

    return {
        "source": "FINRA Query API catalog",
        "count": len(datasets),
        "datasets": datasets,
    }


def _list_match_score(entry: CatalogEntry, tokens: list[str]) -> int | None:
    """Ranked match score, or None when the entry does not match."""
    if not tokens:
        return 0
    score = _search_score(entry, tokens)
    return score if score > 0 else None


def _list_dataset_caps(entry: CatalogEntry) -> tuple[bool | None, bool | None]:
    """Verified ticker/date capabilities from override corrections only.

    Unknown stays None until describe_finra_dataset fetches real metadata —
    never guessed from the dataset name.
    """
    override = _METADATA_OVERRIDES.get((entry.group.lower(), entry.name.lower()), {})
    supports_ticker: bool | None = None
    supports_date: bool | None = None
    if "symbol_field" in override:
        supports_ticker = bool(override["symbol_field"]) and not override.get("market_aggregate")
    if "date_field" in override:
        supports_date = bool(override["date_field"])
    if override.get("market_aggregate"):
        supports_ticker = False
    return supports_ticker, supports_date


def _list_dataset_row(entry: CatalogEntry, score: int) -> dict[str, object]:
    supports_ticker, supports_date = _list_dataset_caps(entry)
    return {
        "dataset": entry.dataset_id,
        "group": entry.group,
        "name": entry.name,
        "description": entry.description,
        "supports_ticker": supports_ticker,
        "supports_date": supports_date,
        "supports_record_offset": entry.supports_record_offset,
        "access": entry.access,
        "match_score": score,
    }


# Multi-word friendly labels that map onto catalog wording.
_SEARCH_PHRASE_ALIASES = {
    "trading volume": "volume",
    "daily volume": "volume",
    "trade volume": "volume",
    "threshold securities": "threshold",
    "registration type": "registration",
}

# Single-token synonym expansions (query token -> haystack tokens that count).
_SEARCH_TOKEN_VARIANTS = {
    "trading": {"trading", "trade", "traded", "trades"},
    "volume": {"volume", "vol", "volumes"},
    "weekly": {"weekly", "week"},
    "week": {"weekly", "week"},
    "daily": {"daily", "day"},
    "day": {"daily", "day"},
    "monthly": {"monthly", "month"},
    "month": {"monthly", "month"},
    "otc": {"otc"},
    "ats": {"ats"},
    "interest": {"interest"},
    "short": {"short"},
    "aggregates": {"aggregate", "aggregates"},
    "aggregate": {"aggregate", "aggregates"},
}

# Tokens that add no topical signal (tickers, fillers, stop words).
_SEARCH_STOP_TOKENS = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "to",
    "by",
    "what",
    "is",
    "are",
    "show",
    "showme",
    "me",
    "values",
    "data",
    "list",
}


def _split_words(text: str) -> list[str]:
    """Lowercase tokenization with camelCase and punctuation boundaries."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
    return re.findall(r"[a-z0-9]+", text)


def _search_tokens(search: str | None) -> list[str]:
    """Normalize a free-text search phrase into ranked match tokens.

    Compound aliases are collapsed first ("trading volume" -> "volume",
    "short interest" -> "short interest" tokens), then the phrase is split
    on non-alphanumerics and camelCase boundaries. Unknown tokens (e.g. a
    ticker like "AAPL") are dropped — they cannot match catalog wording.
    """
    text = (search or "").strip().lower()
    if not text:
        return []
    for phrase, alias in _SEARCH_PHRASE_ALIASES.items():
        text = text.replace(phrase, alias)
    return [t for t in _split_words(text) if t not in _SEARCH_STOP_TOKENS and len(t) >= 2]


def _search_score(entry: CatalogEntry, tokens: list[str]) -> int:
    """Number of distinct query tokens with a variant present in the entry.

    Name/group matches count double so a phrase like "OTC weekly trading
    volume" ranks weeklySummary (weekly in name, OTC in group, volume in
    description) above generic descriptions.
    """
    if not tokens:
        return 0
    name = set(_split_words(entry.name))
    hay = set(_split_words(entry.group)) | name | set(_split_words(entry.description))
    score = 0
    for token in tokens:
        variants = _SEARCH_TOKEN_VARIANTS.get(token, {token})
        if not (variants & hay):
            continue
        hits = len(variants & hay)
        if variants & name:
            score += 2 * hits
        else:
            score += hits
    return score


def describe_dataset(dataset_id: str) -> dict[str, object]:
    """Full field metadata for one dataset (filing-cabinet describe step)."""
    try:
        entry = _resolve_dataset(dataset_id)
        spec = _get_dataset_spec(entry)
    except ValueError as e:
        return {"error": str(e)}
    except requests.HTTPError as e:
        return _describe_http_error(dataset_id, e)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"error": _catalog_error_message(e)}
    return _describe_result(spec, entry, _describe_field_rows(spec))


def _describe_http_error(dataset_id: str, e: requests.HTTPError) -> dict[str, object]:
    status = e.response.status_code if e.response is not None else None
    if status in (401, 403):
        return {
            "error": (
                f"FINRA returned {status} for metadata of '{dataset_id}'. "
                "It may not be public or the configured credentials lack "
                "the required entitlement. Use list_finra_datasets to see "
                "available datasets."
            )
        }
    return {"error": _catalog_error_message(e)}


def _describe_field_rows(spec: DatasetSpec) -> list[dict[str, object]]:
    return [
        {
            "name": f.get("name"),
            "type": f.get("type"),
            "description": f.get("description") or "",
            **({"format": f["format"]} if f.get("format") else {}),
        }
        for f in spec.fields
        if f.get("name")
    ]


def _describe_result(spec: DatasetSpec, entry: CatalogEntry, fields_out: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "dataset": spec.dataset_id,
        "group": spec.group,
        "name": spec.name,
        "description": spec.description,
        "fields": fields_out,
        "partition_fields": list(spec.partition_fields),
        "ticker_field": spec.symbol_field,
        "date_field": spec.date_field,
        "supports_ticker": bool(spec.symbol_field) and not spec.market_aggregate,
        "supports_date": bool(spec.date_field),
        "market_aggregate": spec.market_aggregate,
        "supported_methods": list(spec.methods),
        "supports_record_offset": entry.supports_record_offset,
        "access": entry.access,
        "source": f"FINRA metadata {spec.group}/{spec.name}",
    }
    _apply_describe_extras(result, spec)
    return result


def _apply_describe_extras(result: dict[str, object], spec: DatasetSpec) -> None:
    if spec.valid_filter_values:
        result["valid_filter_values"] = {k: list(v) for k, v in spec.valid_filter_values.items()}
    if spec.default_filters:
        result["default_filters"] = [{"field": f, "value": v} for f, v in spec.default_filters]


def get_short_interest(ticker: str, settlement_date: str | None = None) -> dict[str, object]:
    result = query_dataset(
        "otcMarket/consolidatedShortInterest",
        ticker=ticker,
        start_date=settlement_date,
        end_date=settlement_date,
        # Recent settlement cycles are sufficient for the briefing and keep
        # stale-data recovery within the bounded partition-query budget.
        limit=5,
        prefer_latest=settlement_date is None,
    )
    if settlement_date is None and result.get("data_freshness") == "stale":
        as_of_value = result.get("as_of_date")
        return _stale_short_interest_error(ticker, str(as_of_value) if as_of_value is not None else None)
    return result


def get_reg_sho_volume(ticker: str, trade_date: str | None = None) -> dict[str, object]:
    return query_dataset(
        "otcMarket/regShoDaily",
        ticker=ticker,
        start_date=trade_date,
        end_date=trade_date,
        limit=100,
    )


def get_threshold_securities(ticker: str | None = None, trade_date: str | None = None) -> dict[str, object]:
    return query_dataset(
        "otcMarket/thresholdList",
        ticker=ticker,
        start_date=trade_date,
        end_date=trade_date,
        limit=200 if not ticker else 50,
    )


# Official public files behind API rows: the human citation page of a FINRA
# record. Only the datasets below publish a static cdn.finra.org file; the
# thresholdList/weeklySummary/... pages are SPAs fed by api.finra.org itself.
_OFFICIAL_FILE_KIND = "official_file"
# marketCode or reportingFacilityCode -> daily short-sale volume file prefix.
# API rows for 2026-10-15 (B/NCTRF, Q/NQTRF, N/NYTRF, O/ORF) equal the
# FNQC/FNSQ/FNYX/FORF file lines; "D = ADF" per FINRA regShoDaily metadata.
_REG_SHO_FILE_PREFIXES = {
    "B": "FNQC",
    "NCTRF": "FNQC",
    "Q": "FNSQ",
    "NQTRF": "FNSQ",
    "N": "FNYX",
    "NYTRF": "FNYX",
    "O": "FORF",
    "ORF": "FORF",
    "D": "FNRA",
}
# regShoDaily API field -> facility file column.
_REG_SHO_FILE_COLUMNS = {
    "shortParQuantity": "ShortVolume",
    "shortExemptParQuantity": "ShortExemptVolume",
    "totalParQuantity": "TotalVolume",
    "marketCode": "Market",
}


def resolve_finra_official(
    dataset: str, row: Mapping[str, object], payload: Mapping[str, object], *, root: Path
) -> dict[str, object] | None:
    """Official cdn.finra.org file and the exact cited line for one FINRA row.

    ``row`` is the cited row ({} = the payload's first record); analysis
    envelopes cite their query JSON, so the symbol may be ``ticker`` and the day
    the payload's ``as_of_date``. None when FINRA publishes no static file for
    the dataset or no symbol/day is known. Fetch failures raise; a missing or
    disagreeing line raises ValueError.
    """
    name = dataset.rpartition("/")[2].casefold().removesuffix("mock")
    if name == "regshodaily":
        symbol_field, date_field = "securitiesInformationProcessorSymbolIdentifier", "tradeReportDate"
    elif name.endswith("shortinterest"):
        symbol_field, date_field = "symbolCode", "settlementDate"
    else:
        return None
    records = payload.get("records")
    first = records[0] if isinstance(records, list) and records else None
    cited: Mapping[str, object] = row or (first if isinstance(first, Mapping) else {})
    query = payload.get("query")
    symbol = (
        cited.get(symbol_field) or cited.get("ticker") or (query.get("ticker") if isinstance(query, Mapping) else None)
    )
    day = _norm_date(cited.get(date_field) or payload.get("as_of_date"))
    if not (isinstance(symbol, str) and symbol.strip() and day):
        return None
    stamp = day.replace("-", "")
    columns: Mapping[str, str] | None = None
    if name == "regshodaily":
        facility = _REG_SHO_FILE_PREFIXES.get(str(cited.get("marketCode"))) or _REG_SHO_FILE_PREFIXES.get(
            str(cited.get("reportingFacilityCode"))
        )
        # ponytail: no facility (analysis envelope, trimmed fields) cites the consolidated line unchecked.
        url = f"https://cdn.finra.org/equity/regsho/daily/{facility or 'CNMS'}shvol{stamp}.txt"
        symbol_column = "Symbol"
        columns = _REG_SHO_FILE_COLUMNS if facility else {}
    else:
        url = f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{stamp}.csv"
        symbol_column = "symbolCode"
    data = _official_file(url, root)
    quote = _official_line(data.decode("utf-8", "replace"), url, symbol_column, symbol.strip(), cited, columns)
    return {
        "url": url,
        "payload": data,
        "sha256": raw_archive.content_hash(data),
        "media": "text",
        "quote": quote,
        "anchor": "",
    }


def _official_file(url: str, root: Path) -> bytes:
    """Archived bytes of one official file; fetched and archived once."""
    record = raw_archive.find("finra", _OFFICIAL_FILE_KIND, url, root=root / "raw")
    if record is not None:
        return record.payload_path.read_bytes()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw_archive.archive("finra", _OFFICIAL_FILE_KIND, url, resp.content, url=url, root=root / "raw")
    return resp.content


def _official_line(
    text: str,
    url: str,
    symbol_column: str,
    symbol: str,
    cited: Mapping[str, object],
    columns: Mapping[str, str] | None,
) -> str:
    """The pipe-delimited line for ``symbol``; its cells must equal the cited
    values (``columns``: API field -> file column, None = same names)."""
    lines = text.splitlines()
    header = [cell.strip('"') for cell in lines[0].split("|")] if lines else []
    if symbol_column not in header:
        raise ValueError(f"{url}: no {symbol_column} column")
    at = header.index(symbol_column)
    for line in lines[1:]:
        cells = [cell.strip('"') for cell in line.split("|")]
        if len(cells) != len(header) or cells[at] != symbol:
            continue
        values = dict(zip(header, cells, strict=True))
        for api_field, column in (columns if columns is not None else {h: h for h in header}).items():
            value = cited.get(api_field)
            if value is not None and column in values and not _same_value(value, values[column]):
                raise ValueError(f"{url}: {symbol} {column}={values[column]!r} disagrees with {api_field}={value!r}")
        return line
    raise ValueError(f"{url}: no line for {symbol}")


def _same_value(value: object, cell: str) -> bool:
    """Numbers compare by value (file 274457.043010 == API 274457.04301)."""
    try:
        return Decimal(str(value)) == Decimal(cell)
    except InvalidOperation:
        return str(value).strip() == cell.strip()


def _filter_list(filters: object) -> list[dict[str, object]] | None:
    """Validate raw tool-JSON filters once; None stays None, garbage raises ValueError."""
    if filters is None:
        return None
    if isinstance(filters, list):
        for index, item in enumerate(filters):
            if not isinstance(item, dict):
                raise ValueError(  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
                    f"Filter #{index} must be an object with 'field' and 'value'."
                )
        return [{str(k): v for k, v in f.items()} for f in filters if isinstance(f, dict)]
    raise ValueError("filters must be a list of {field, op, value} objects.")


def _query_unexpected_error(dataset: str, e: Exception) -> dict[str, object]:
    """Generic query failure: catalog message for config issues, else plain."""
    logger.exception("FINRA query failed for dataset %s", dataset)
    msg = str(e)
    if "FINRA_CLIENT" in msg or "catalog" in msg.lower():
        return {"error": _catalog_error_message(e)}
    return {"error": f"FINRA query failed: {e}"}


def _query_effective_window(payload: dict[str, object], records: list[dict[str, object]]) -> tuple[int, int, int]:
    """(effective_limit, effective_offset, returned_count) from the payload."""
    raw_limit = payload.get("limit", DEFAULT_LIMIT)
    raw_offset = payload.get("offset", 0)
    effective_limit = raw_limit if isinstance(raw_limit, int) else int(str(raw_limit))
    effective_offset = raw_offset if isinstance(raw_offset, int) else int(str(raw_offset))
    return effective_limit, effective_offset, len(records)


def _query_wants_latest_recovery(
    spec: DatasetSpec,
    freshness: str,
    start_date: str | None,
    end_date: str | None,
    filter_list: list[dict[str, object]] | None,
    effective_offset: int,
    prefer_latest: bool,
) -> bool:
    """Latest-data recovery applies: stale prefer_latest with no narrowing."""
    return bool(
        prefer_latest
        and freshness == "stale"
        and spec.date_field
        and spec.partition_fields
        and not start_date
        and not end_date
        and not filter_list
        and effective_offset == 0
    )


def _recovery_params(spec: DatasetSpec) -> tuple[list[str], list[str]]:
    """Selected fields + newest-first sort for the recovery walk."""
    selected = [str(f["name"]) for f in spec.fields if f.get("name")]
    return selected, [f"-{spec.date_field!s}"]


def _recovery_from_payload(
    spec: DatasetSpec,
    records: list[dict[str, object]],
    headers: dict[str, object],
    effective_offset: int,
    effective_limit: int,
    partition_queries: int,
) -> tuple[list[dict[str, object]], dict[str, object], int, str | None, str]:
    """Final recovery tuple from walk records/headers (pagination + freshness)."""
    returned_count = len(records)
    pagination = _parse_pagination(headers, effective_offset, effective_limit, returned_count)
    pagination.update(
        {
            "total_records": None,
            "may_have_more": (returned_count >= effective_limit and partition_queries < _MAX_PARTITION_QUERIES),
            "source": "partitions",
        }
    )
    as_of, freshness = _freshness_status(spec, records)
    return records, pagination, returned_count, as_of, freshness


def _query_latest_recovery(
    dataset: str,
    spec: DatasetSpec,
    entry: CatalogEntry,
    ticker: str | None,
    effective_limit: int,
    effective_offset: int,
    returned_count: int,
) -> tuple[
    tuple[list[dict[str, object]], dict[str, object], int, str | None, str] | None,
    dict[str, object] | None,
]:
    """Partition-walk recovery for stale prefer_latest queries.

    Returns ((records, pagination, returned_count, as_of, freshness), None) on
    success, (None, error_result) when the walk itself fails.
    """
    selected, sort = _recovery_params(spec)
    try:
        records, headers, partition_queries, _short = _datapoints_via_partitions(
            spec,
            entry,
            selected,
            ticker,
            None,
            None,
            None,
            effective_limit,
            sort,
        )
    except requests.HTTPError as e:
        return None, _http_error_result(dataset, e)
    except ValueError as e:
        return None, {"error": str(e)}
    return _recovery_from_payload(
        spec,
        records,
        headers,
        effective_offset,
        effective_limit,
        partition_queries,
    ), None


def _query_analysis_warnings(analysis: dict[str, object], as_of: str | None, freshness: str) -> list[object]:
    """Analysis warnings plus the stale-data warning when applicable."""
    raw_warnings: object = analysis.get("warnings")
    warnings: list[object] = list(raw_warnings) if isinstance(raw_warnings, list) else []
    stale = _stale_warning(as_of, freshness)
    if stale:
        warnings.append(stale)
    return warnings


def _query_result_row(
    spec: DatasetSpec,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: object,
    analysis: dict[str, object],
    warnings: list[object],
    as_of: str | None,
    freshness: str,
    returned_count: int,
    effective_limit: int,
    effective_offset: int,
    pagination: dict[str, object],
) -> dict[str, object]:
    """Final analyzed-briefing result row for query_dataset."""
    return {
        "dataset": spec.name,
        "group": spec.group,
        "dataset_id": spec.dataset_id,
        "source": f"FINRA Query API {spec.group}/{spec.name}",
        "query": {
            "ticker": (ticker or "").strip().upper() or None,
            "start_date": start_date,
            "end_date": end_date,
            "limit": effective_limit,
            "offset": effective_offset,
            "filters": filters,
        },
        "coverage": analysis["coverage"],
        "metrics": analysis["metrics"],
        "trends": analysis["trends"],
        "warnings": warnings,
        "briefing": analysis["briefing"],
        "briefing_source": analysis["briefing_source"],
        "analysis_model": analysis["analysis_model"],
        "as_of_date": as_of,
        "data_freshness": freshness,
        "environment": _environment(),
        "returned_count": returned_count,
        "limit": effective_limit,
        "offset": effective_offset,
        "next_offset": effective_offset + returned_count,
        "may_have_more": pagination["may_have_more"],
        "total_records": pagination["total_records"],
        "pagination_source": pagination["source"],
    }


def _query_apply_recovery(
    dataset: str,
    spec: DatasetSpec,
    entry: CatalogEntry,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filter_list: list[dict[str, object]] | None,
    effective_limit: int,
    effective_offset: int,
    prefer_latest: bool,
    current: tuple[list[dict[str, object]], dict[str, object], int, str | None, str],
) -> tuple[
    tuple[list[dict[str, object]], dict[str, object], int, str | None, str],
    dict[str, object] | None,
]:
    """Stale prefer_latest partition-walk recovery, else the input tuple.

    Returns (result_tuple, None) on success/skip; (current, error) when the
    walk itself fails.
    """
    _records, _pagination, returned_count, _as_of, freshness = current
    if not _query_wants_latest_recovery(
        spec,
        freshness,
        start_date,
        end_date,
        filter_list,
        effective_offset,
        prefer_latest,
    ):
        return current, None
    recovered, error = _query_latest_recovery(
        dataset,
        spec,
        entry,
        ticker,
        effective_limit,
        effective_offset,
        returned_count,
    )
    if error is not None:
        return current, error
    return (recovered or current), None


def query_dataset(
    dataset: str,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    filters: object = None,
    analysis_goal: str | None = None,
    prefer_latest: bool = False,
) -> dict[str, object]:
    """Query a FINRA dataset and return an analyzed briefing.

    The main model receives deterministic metrics, trends, warnings, and
    (when configured) validated prose — never the raw records.
    """
    try:
        entry = _resolve_dataset(dataset)
        spec = _get_dataset_spec(entry)
        filter_list = _filter_list(filters)
        payload = _build_payload(spec, entry, ticker, start_date, end_date, limit, filter_list, offset)
        records, headers = _cached_query(spec, payload)
    except ValueError as e:
        return {"error": str(e)}
    except requests.HTTPError as e:
        return _http_error_result(dataset, e)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _query_unexpected_error(dataset, e)
    if not records:
        what = ticker or dataset
        return {"error": f"No data found for {what}: {spec.name}"}
    effective_limit, effective_offset, returned_count = _query_effective_window(payload, records)
    pagination = _parse_pagination(headers, effective_offset, effective_limit, returned_count)
    as_of, freshness = _freshness_status(spec, records)
    current = (records, pagination, returned_count, as_of, freshness)
    final, recovery_error = _query_apply_recovery(
        dataset,
        spec,
        entry,
        ticker,
        start_date,
        end_date,
        filter_list,
        effective_limit,
        effective_offset,
        prefer_latest,
        current,
    )
    if recovery_error is not None:
        return recovery_error
    records, pagination, returned_count, as_of, freshness = final
    analysis = analyze_and_brief(
        spec,
        records,
        analysis_goal,
        _query_cache_key(spec, payload),
        pagination=pagination,
    )
    return _query_result_row(
        spec,
        ticker,
        start_date,
        end_date,
        filters,
        analysis,
        _query_analysis_warnings(analysis, as_of, freshness),
        as_of,
        freshness,
        returned_count,
        effective_limit,
        effective_offset,
        pagination,
    )


def _datapoints_fetch_plan(
    spec: DatasetSpec,
    entry: CatalogEntry,
    selected: list[str],
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filter_list: list[dict[str, object]] | None,
    sort: list[str],
    limit: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    int,
    bool,
    dict[str, object] | None,
]:
    """(records, headers, partition_queries, short_result, payload-or-None).

    Partition-walking sorts resolve through _datapoints_via_partitions;
    everything else goes through the direct _build_payload/_cached_query path.
    """
    via = _use_partition_flow(spec, sort, ticker, start_date, end_date, filter_list)
    if via:
        records, headers, queries, short = _datapoints_via_partitions(
            spec,
            entry,
            selected,
            ticker,
            start_date,
            end_date,
            filter_list,
            limit,
            sort,
        )
        return records, headers, queries, short, None
    payload = _build_payload(
        spec,
        entry,
        ticker,
        start_date,
        end_date,
        limit,
        filter_list,
        fields=selected,
        sort_fields=sort,
    )
    records, headers = _cached_query(spec, payload)
    return records, headers, 0, False, payload


def _datapoints_prepared(
    dataset: str,
    fields: object,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int | None,
    filters: object,
    sort_fields: object,
    sort_order: str | None,
) -> tuple[
    tuple[CatalogEntry, DatasetSpec, list[str], list[dict[str, object]] | None, list[str], int, str],
    dict[str, object] | None,
    tuple[list[dict[str, object]], dict[str, object], int, bool],
]:
    """Validated fetch plan: ((entry, spec, selected, filter_list, sort, limit,
    dataset_id), payload-or-None) plus fetched (records, headers, queries,
    short). Raises on invalid input; HTTP errors are returned, not raised.
    """
    entry = _resolve_dataset(dataset)
    spec = _get_dataset_spec(entry)
    selected = _validate_datapoint_fields(spec, fields)
    filter_list = _filter_list(filters)
    _require_datapoint_narrowing(ticker, start_date, end_date, filter_list)
    sort = _validate_sort(spec, sort_fields, sort_order)
    clamped = _clamp_limit(limit, default=DATAPOINTS_DEFAULT_LIMIT, maximum=DATAPOINTS_MAX_LIMIT)
    records, headers, queries, short, payload = _datapoints_fetch_plan(
        spec,
        entry,
        selected,
        ticker,
        start_date,
        end_date,
        filter_list,
        sort,
        clamped,
    )
    plan = (entry, spec, selected, filter_list, sort, clamped, spec.dataset_id)
    fetched = (records, headers, queries, short)
    return plan, payload, fetched


def _datapoints_http_error(
    dataset: str,
    dataset_id: str,
    payload: dict[str, object] | None,
    e: requests.HTTPError,
) -> dict[str, object]:
    """Structured error for a failed exact-datapoints request."""
    return _http_error_result(
        dataset,
        e,
        request_purpose="exact datapoints request (get_finra_datapoints)",
        payload=payload,
        dataset_id=dataset_id,
    )


def _datapoints_pagination(
    via_partitions: bool,
    headers: dict[str, object],
    effective_limit: int,
    reduced: list[dict[str, object]],
    partition_queries: int,
) -> dict[str, object]:
    """Header pagination, or partition-driven pagination across partitions."""
    if not via_partitions:
        return _parse_pagination(headers, 0, effective_limit, len(reduced))
    # Across partitions there is no single Record-Total; honesty over
    # estimates: mark pagination as partition-driven.
    return {
        "total_records": None,
        "may_have_more": (
            len(reduced) >= effective_limit and partition_queries > 0 and partition_queries < _MAX_PARTITION_QUERIES
        ),
        "source": "partitions",
    }


def _datapoints_warnings(
    short_result: bool,
    reduced: list[dict[str, object]],
    effective_limit: int,
    as_of: str | None,
    freshness: str,
) -> list[object]:
    """Stale-data plus complete-short-result warnings for datapoints."""
    stale = _stale_warning(as_of, freshness)
    warnings: list[object] = [stale] if stale else []
    if short_result:
        warnings.append(
            f"Complete short result: only {len(reduced)} matching records "
            f"were found across all relevant FINRA partitions (requested "
            f"{effective_limit})."
        )
    return warnings


def _datapoints_result_row(
    spec: DatasetSpec,
    selected: list[str],
    reduced: list[dict[str, object]],
    effective_limit: int,
    pagination: dict[str, object],
    as_of: str | None,
    freshness: str,
    warnings: list[object],
    sort: list[str],
    via_partitions: bool,
    partition_queries: int,
) -> dict[str, object]:
    """Final exact-datapoints result row."""
    result: dict[str, object] = {
        "dataset": spec.name,
        "group": spec.group,
        "dataset_id": spec.dataset_id,
        "source": f"FINRA Query API {spec.group}/{spec.name}",
        "fields": list(selected),
        "records": reduced,
        "returned_count": len(reduced),
        "limit": effective_limit,
        "offset": 0,
        "next_offset": len(reduced),
        "may_have_more": pagination["may_have_more"],
        "total_records": pagination["total_records"],
        "pagination_source": pagination["source"],
        "as_of_date": as_of,
        "data_freshness": freshness,
        "environment": _environment(),
        "warnings": warnings,
    }
    if sort:
        result["sort_fields"] = list(sort)
    if via_partitions:
        result["sort_source"] = "partitions"
        result["partition_queries"] = partition_queries
    return result


def _datapoints_used_partitions(
    sort: list[str],
    spec: DatasetSpec,
    partition_queries: int,
    short_result: bool,
    payload: dict[str, object] | None,
) -> bool:
    """True when the rows came from the partition-walk path."""
    if partition_queries > 0 or short_result or payload is None and sort:
        return (
            partition_queries > 0
            or short_result
            or (bool(sort) and sort[0][1:] == spec.date_field and _date_partition_field(spec) is not None)
        )
    return False


def _datapoints_reduced(
    records: list[dict[str, object]],
    sort: list[str],
    selected: list[str],
    effective_limit: int,
) -> list[dict[str, object]]:
    """Locally ordered rows reduced to the selected fields (capped at limit)."""
    ordered = _apply_local_sort(records, sort) if sort else records
    return [_select_fields(row, selected) for row in ordered[:effective_limit]]


def _datapoints_is_stale_short_interest(spec: DatasetSpec, sort: list[str], freshness: str) -> bool:
    """Stale sorted short-interest requests are refused, never returned."""
    return spec.name.casefold() == "consolidatedshortinterest" and bool(sort) and freshness == "stale"


def get_finra_datapoints(
    dataset: str,
    fields: object = None,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    filters: object = None,
    sort_fields: object = None,
    sort_order: str | None = None,
) -> dict[str, object]:
    """Exact requested fields from FINRA source records (explicit data asks).

    Requires a non-empty, metadata-validated fields list (at most
    DATAPOINTS_MAX_FIELDS) and at least one narrowing condition (ticker,
    date/date range, or filter). Returns only the selected fields per row,
    capped at DATAPOINTS_MAX_LIMIT. Exact values are guaranteed for normal
    scalar data; oversized text fields may be shown as a marked excerpt by
    the rendering layer.

    sort_fields uses FINRA sortFields syntax: '+field' ascending, '-field'
    descending (e.g. ["-settlementDate"] for newest first). sort_order is a
    convenience that sorts by the dataset's date field when one exists. When
    exactly one sort field is requested, rows are re-ordered deterministically
    before returning so 'latest five' style requests are stable regardless of
    source ordering.
    """
    payload: dict[str, object] | None = None
    dataset_id = dataset
    try:
        plan, payload, fetched = _datapoints_prepared(
            dataset,
            fields,
            ticker,
            start_date,
            end_date,
            limit,
            filters,
            sort_fields,
            sort_order,
        )
        _entry, spec, selected, _filters, sort, clamped, dataset_id = plan
        records, headers, partition_queries, short_result = fetched
    except ValueError as e:
        return {"error": str(e)}
    except requests.HTTPError as e:
        return _datapoints_http_error(dataset, dataset_id, payload, e)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _query_unexpected_error(dataset, e)
    if not records:
        what = ticker or dataset
        return {"error": f"No data found for {what}: {spec.name}"}
    effective_limit = clamped
    reduced = _datapoints_reduced(records, sort, selected, effective_limit)
    via_partitions = _datapoints_used_partitions(sort, spec, partition_queries, short_result, payload)
    pagination = _datapoints_pagination(via_partitions, headers, effective_limit, reduced, partition_queries)
    as_of, freshness = _freshness_status(spec, records)
    if _datapoints_is_stale_short_interest(spec, sort, freshness):
        return _stale_short_interest_error(ticker or dataset, as_of)
    return _datapoints_result_row(
        spec,
        selected,
        reduced,
        effective_limit,
        pagination,
        as_of,
        freshness,
        _datapoints_warnings(short_result, reduced, effective_limit, as_of, freshness),
        sort,
        via_partitions,
        partition_queries,
    )


def _http_error_result(
    dataset: str,
    exc: requests.HTTPError,
    request_purpose: str = "",
    payload: dict[str, object] | None = None,
    dataset_id: str | None = None,
) -> dict[str, object]:
    """Structured, credential-free FINRA error result.

    Carries the dataset, the request purpose, the HTTP status, and the
    sanitized response body so the agent loop and the renderer can tell the
    main model exactly what failed. The sanitized payload and response
    metadata are logged at debug level; credentials and Authorization
    headers are never logged.
    """
    status = exc.response.status_code if exc.response is not None else None
    raw_body = exc.response.text if exc.response is not None else ""
    body = _sanitize_finra_body(raw_body)[:500]
    logger.exception("FINRA HTTP error for dataset %s", dataset)
    logger.debug(
        "FINRA request failed: dataset=%s status=%s purpose=%s payload=%s response=%s",
        dataset,
        status,
        request_purpose or "FINRA data request",
        _sanitize_payload(payload),
        body,
    )
    result: dict[str, object] = {
        "dataset": dataset,
        "dataset_id": dataset_id or dataset,
        "request_purpose": request_purpose or "FINRA data request",
        "http_status": status,
        "finra_response": body,
        "environment": _environment(),
    }
    if status in (401, 403):
        result["error"] = (
            f"FINRA returned {status} for dataset '{dataset}'. "
            "It may not be public or the configured credentials lack "
            "the required entitlement. Use list_finra_datasets to see "
            "available datasets, or omit this request."
        )
        return result
    result["error"] = f"FINRA request failed ({status if status is not None else '?'}): {body}"
    return result


# ---------------------------------------------------------------------------
# Partition-aware sorting (FINRA requires EQUAL filters on every partition
# field before sortFields is accepted; otherwise the API returns HTTP 400).
# ---------------------------------------------------------------------------

_MAX_PARTITION_QUERIES = 12


def _sort_needs_partition_walk(spec: DatasetSpec, covered: set[str]) -> bool:
    """False when every partition field already has an EQUAL filter."""
    return not all(f in covered for f in spec.partition_fields)


def _reject_multi_field_sort(spec: DatasetSpec, sort: list[str]) -> str:
    """Sort field name for a single-field sort; raise for multi-field sorts."""
    if len(sort) != 1:
        raise ValueError(
            f"Multi-field sorting requires an exact EQUAL filter on every "
            f"partition field of '{spec.dataset_id}' ({', '.join(spec.partition_fields)}). "
            "FINRA rejects sortFields without those filters."
        )
    return sort[0][1:]


def _reject_mapped_date_range(
    spec: DatasetSpec,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Reject date-range sorts when the sort date is not the partition date."""
    if spec.date_field in spec.partition_fields:
        return
    start = _clean_date_opt(start_date)
    end = _clean_date_opt(end_date)
    if start and end and start != end:
        raise ValueError(
            f"Date-range sorting is not supported for '{spec.dataset_id}': "
            f"the partition date ({_date_partition_field(spec)}) differs "
            f"from the requested date field ({spec.date_field}), so the "
            "range cannot be narrowed to the relevant partitions without "
            "budget-dependent results. A single date does not pre-narrow "
            "partitions either; use exact EQUAL filters on every partition "
            "field, or drop the sort."
        )


def _reject_unwalkable_sort(spec: DatasetSpec, sort: list[str]) -> None:
    """Raise for sorts that are neither server-side-valid nor walkable."""
    raise ValueError(
        f"Sorting by '{sort[0]}' on '{spec.dataset_id}' requires an exact "
        f"EQUAL filter on every partition field "
        f"({', '.join(spec.partition_fields)}). FINRA rejects sortFields "
        "without those filters, and partition walking is only available "
        f"for the dataset's authoritative date field "
        f"{spec.date_field or '(none)'} with a verified date partition. "
        "Either add the required EQUAL filters or request a date-based "
        "latest/oldest sort."
    )


def _use_partition_flow(
    spec: DatasetSpec,
    sort: list[str],
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
) -> bool:
    """Decide how to honor a sort request.

    Returns True when server-side sortFields cannot be sent and the sort can
    be resolved by walking dataset partitions: single-field sorts on the
    dataset's authoritative date field, ordered by a verified date
    partition. Date ranges are rejected when the authoritative date field is
    not itself a partition field (the mapped case), because the range could
    not be narrowed to relevant partitions and would make results
    budget-dependent. Raises ValueError for sorts that are neither
    server-side-valid nor resolvable from partitions.
    """
    if not sort:
        return False
    if not spec.partition_fields:
        # FINRA allows sortFields when a dataset has no partition fields.
        return False
    covered = _partition_fields_with_equal(spec, ticker, start_date, end_date, filters)
    if not _sort_needs_partition_walk(spec, covered):
        return False  # caller already supplies valid partition EQUAL filters
    name = _reject_multi_field_sort(spec, sort)
    if name != spec.date_field or _date_partition_field(spec) is None:
        _reject_unwalkable_sort(spec, sort)
        return False  # unreachable; _reject_unwalkable_sort always raises
    _reject_mapped_date_range(spec, start_date, end_date)
    return True


def _clean_str(value: object) -> str:
    """Stripped text for an optional string (None/other -> "")."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _clean_opt(value: object) -> str | None:
    """Stripped text, or None when blank."""
    text = _clean_str(value)
    return text if text else None


def _as_name(value: object) -> str | None:
    """Stripped field name, or None when not a non-blank string."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _equal_filter_field(extra: object) -> str | None:
    """Field name when a filter entry is an EQUAL filter, else None."""
    if not isinstance(extra, dict):
        return None
    op = str(extra.get("op") or "EQUAL").upper()
    if op != "EQUAL":
        return None
    return _as_name(extra.get("field"))


def _equal_filter_fields(filters: list[dict[str, object]] | None) -> set[str]:
    """Field names carrying an explicit EQUAL filter."""
    covered: set[str] = set()
    for extra in filters or []:
        name = _equal_filter_field(extra)
        if name is not None:
            covered.add(name)
    return covered


def _implied_single_date_field(spec: DatasetSpec, start_date: str | None, end_date: str | None) -> str | None:
    """Date field implied by a single-date (or open) request, else None."""
    if not spec.date_field:
        return None
    start = _clean_str(start_date)
    end = _clean_str(end_date)
    if start == end and start:
        return spec.date_field
    if bool(start) != bool(end):
        return spec.date_field
    return None


def _implied_symbol_field(spec: DatasetSpec, ticker: str | None) -> str | None:
    """Symbol field implied by a ticker request, else None."""
    if not spec.symbol_field:
        return None
    if _clean_str(ticker):
        return spec.symbol_field
    return None


def _partition_fields_with_equal(
    spec: DatasetSpec,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
) -> set[str]:
    """Partition fields that already carry an EQUAL filter (explicit or implied)."""
    covered = _equal_filter_fields(filters)
    dated = _implied_single_date_field(spec, start_date, end_date)
    if dated is not None:
        covered.add(dated)
    symbol = _implied_symbol_field(spec, ticker)
    if symbol is not None:
        covered.add(symbol)
    return covered


def _walk_range_field(spec: DatasetSpec, start: str | None, end: str | None) -> str | None:
    """Date field to narrow the walk to, for a true start/end range."""
    if start and end and start != end:
        return spec.date_field
    return None


def _walk_range_tuples(tuples: list[dict[str, str]], range_field: str, start: str, end: str) -> list[dict[str, str]]:
    """Partition tuples whose range-field value falls inside [start, end]."""
    return [t for t in tuples if t.get(range_field) and _date_in_range(t[range_field], start, end)]


def _walk_tuples(
    spec: DatasetSpec,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
    descending: bool,
) -> tuple[list[dict[str, str]], str | None]:
    """Ordered partition tuples for the walk plus the range-narrowed field."""
    partitions = _get_partitions(spec)
    pinned = _pinned_partition_filters(spec, ticker, start_date, end_date, filters)
    tuples = _ordered_partition_tuples(spec, partitions, pinned, spec.date_field, descending)
    start = _clean_date_opt(start_date)
    end = _clean_date_opt(end_date)
    range_field = _walk_range_field(spec, start, end)
    if range_field and start and end and range_field in spec.partition_fields:
        tuples = _walk_range_tuples(tuples, range_field, start, end)
    return tuples, range_field


def _walk_partition_payload(
    spec: DatasetSpec,
    entry: CatalogEntry,
    selected: list[str],
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
    remaining: int,
    tuple_values: dict[str, str],
    range_field: str | None,
) -> dict[str, object]:
    """Query payload for one partition tuple (EQUAL filters, no sortFields)."""
    extra: list[dict[str, object]] = [
        {"field": f, "op": "EQUAL", "value": v} for f, v in tuple_values.items() if f != range_field
    ]
    return _build_payload(
        spec,
        entry,
        ticker,
        start_date,
        end_date,
        remaining,
        list(filters or []) + extra,
        fields=selected,
    )


def _walk_one_partition(
    spec: DatasetSpec,
    payload: dict[str, object],
) -> tuple[list[dict[str, object]] | None, requests.HTTPError | None]:
    """(records, None) on success; (None, error) when the query failed."""
    try:
        records, _headers = _cached_query(spec, payload)
    except requests.HTTPError as e:
        logger.debug(
            "Partition query failed (counted): dataset=%s status=%s payload=%s",
            spec.dataset_id,
            e.response.status_code if e.response is not None else "?",
            _sanitize_payload(payload),
        )
        return None, e
    return records, None


def _walk_partitions(
    spec: DatasetSpec,
    entry: CatalogEntry,
    selected: list[str],
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
    limit: int,
    tuples: list[dict[str, str]],
    range_field: str | None,
) -> tuple[list[dict[str, object]], int, bool, requests.HTTPError | None]:
    """Walk tuples until the limit fills or the query budget is exhausted."""
    accumulated: list[dict[str, object]] = []
    queries = 0
    last_error: requests.HTTPError | None = None
    for tuple_values in tuples:
        if len(accumulated) >= limit or queries >= _MAX_PARTITION_QUERIES:
            break
        queries += 1
        remaining = limit - len(accumulated)
        payload = _walk_partition_payload(
            spec,
            entry,
            selected,
            ticker,
            start_date,
            end_date,
            filters,
            remaining,
            tuple_values,
            range_field,
        )
        records, error = _walk_one_partition(spec, payload)
        if error is not None:
            last_error = error
            continue
        accumulated.extend((records or [])[:remaining])
    exhausted = queries >= _MAX_PARTITION_QUERIES and len(accumulated) < limit
    return accumulated, queries, exhausted, last_error


def _walk_outcome(
    spec: DatasetSpec,
    ticker: str | None,
    limit: int,
    accumulated: list[dict[str, object]],
    queries: int,
    exhausted: bool,
    last_error: requests.HTTPError | None,
) -> tuple[list[dict[str, object]], dict[str, object], int, bool]:
    """Final walk result: raise on budget exhaustion / no data, else rows."""
    if exhausted:
        if not accumulated and last_error is not None:
            raise last_error  # every attempt failed: report the concrete HTTP failure
        raise ValueError(
            f"Could not locate {limit} records for '{spec.dataset_id}' within "
            f"{_MAX_PARTITION_QUERIES} partition queries. Narrow the request "
            "with a ticker, date range, or filters and retry."
        )
    if not accumulated:
        if last_error is not None:
            raise last_error
        what = ticker or spec.dataset_id
        raise ValueError(f"No data found for {what}: {spec.name}")
    return accumulated, {}, queries, len(accumulated) < limit


def _datapoints_via_partitions(
    spec: DatasetSpec,
    entry: CatalogEntry,
    selected: list[str],
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
    limit: int,
    sort: list[str],
) -> tuple[list[dict[str, object]], dict[str, object], int, bool]:
    """Resolve a date sort by walking available partitions.

    Enumerates the partition tuples FINRA actually published (date
    partition ordered newest-first for 'desc', oldest-first for 'asc'),
    querying each tuple with the required EQUAL partition filters plus the
    caller's narrowing conditions — never with sortFields. Every attempted
    query counts against the fixed budget, including HTTP failures and
    no-data responses, so the walk never continues through unlimited
    failing partitions. Stops as soon as the requested limit is accumulated.

    Returns (records, headers, partition_queries, short_result).
    short_result is True when every relevant partition was examined and
    fewer records than the limit exist (complete short result). Raises when
    the bounded budget cannot establish the requested records.
    """
    descending = sort[0][0] == "-"
    tuples, range_field = _walk_tuples(spec, ticker, start_date, end_date, filters, descending)
    accumulated, queries, exhausted, last_error = _walk_partitions(
        spec,
        entry,
        selected,
        ticker,
        start_date,
        end_date,
        filters,
        limit,
        tuples,
        range_field,
    )
    return _walk_outcome(spec, ticker, limit, accumulated, queries, exhausted, last_error)


def _pinned_equal_filters(spec: DatasetSpec, filters: list[dict[str, object]] | None) -> dict[str, str]:
    """Partition fields pinned by caller-supplied EQUAL filter values."""
    pinned: dict[str, str] = {}
    for extra in filters or []:
        name = _equal_filter_field(extra)
        if name is not None and name in spec.partition_fields:
            pinned[name] = str(extra.get("value"))
    return pinned


def _pin_single_date(
    spec: DatasetSpec,
    pinned: dict[str, str],
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Pin the date partition for a single-date request (mapped case excluded)."""
    date_field = spec.date_field
    if date_field is None or date_field not in spec.partition_fields:
        return
    start = _clean_str(start_date)
    end = _clean_str(end_date)
    if not start:
        return
    if end and start != end:
        return
    pinned[date_field] = start


def _pinned_partition_filters(
    spec: DatasetSpec,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
) -> dict[str, str]:
    """Partition fields with caller-supplied EQUAL values (fixed during walk)."""
    pinned = _pinned_equal_filters(spec, filters)
    _pin_single_date(spec, pinned, start_date, end_date)
    return pinned


def _date_in_range(value: str, start: str, end: str) -> bool:
    norm = _norm_date(value)
    return norm is not None and start <= norm <= end


# ---------------------------------------------------------------------------
# Freshness + environment
# ---------------------------------------------------------------------------

STALE_AFTER_DAYS = 90


def _environment() -> str:
    return "mock" if finra_use_mock() else "production"


def _stale_short_interest_error(subject: str, as_of: str | None) -> dict[str, object]:
    dated = f" (newest available date: {as_of})" if as_of else ""
    return {
        "error": (
            f"Current FINRA short interest is unavailable for {subject}{dated}. "
            "FINRA returned only stale historical data, so it cannot answer a "
            "latest short-interest request."
        )
    }


def _freshness_status(spec: DatasetSpec, records: list[dict[str, object]]) -> tuple[str | None, str]:
    """as_of_date from the dataset's authoritative date_field (never derived
    from unrelated fields) plus a current/stale/unknown label."""
    if not spec.date_field or not records:
        return None, "unknown"
    dates = [d for d in (_norm_date(r.get(spec.date_field)) for r in records) if d]
    if not dates:
        return None, "unknown"
    as_of = max(dates)
    try:
        days = (date.today() - date.fromisoformat(as_of)).days  # noqa: DTZ011 - trading-calendar local date has no tz meaning
    except TypeError, ValueError:
        return as_of, "unknown"
    return as_of, ("stale" if days > STALE_AFTER_DAYS else "current")


def _stale_warning(as_of: str | None, freshness: str) -> str | None:
    if freshness != "stale" or not as_of:
        return None
    return (
        f"STALE/HISTORICAL DATA: newest record is {as_of} (over "
        f"{STALE_AFTER_DAYS} days old); this is historical data, not "
        "current market data."
    )


def _norm_date(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", s):
        return s[:10]
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return None


def _sanitize_finra_body(text: str) -> str:
    """Strip credential-shaped content from a FINRA response body."""
    if not text:
        return ""
    text = re.sub(r"Bearer\s+\S+", "[REDACTED]", text)
    for key in (
        "client_id",
        "client_secret",
        "authorization",
        "password",
        "secret",
        "access_token",
        "token",
    ):
        text = re.sub(
            rf'"{key}"\s*:\s*"[^"]*"',
            f'"{key}": "[REDACTED]"',
            text,
            flags=re.IGNORECASE,
        )
    return text.strip()


def _sanitize_payload(payload: dict[str, object] | None) -> str:
    if payload is None:
        return "{}"
    return json.dumps(payload, sort_keys=True, default=str)


def _validated_datapoint_name(spec: DatasetSpec, index: int, entry: object) -> str:
    """One normalized datapoint field name; raises for malformed/unknown names."""
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError(f"fields #{index} is malformed: a non-empty field name is required.")
    name = entry.strip()
    if spec.field_names and name not in spec.field_names:
        known = ", ".join(sorted(spec.field_names)[:30])
        raise ValueError(f"Dataset '{spec.dataset_id}' has no field '{name}'. Known fields include: {known}")
    return name


def _validate_datapoint_fields(spec: DatasetSpec, fields: object) -> list[str]:
    if not fields or not isinstance(fields, list):
        raise ValueError(
            "get_finra_datapoints requires a non-empty 'fields' list. "
            "Call describe_finra_dataset first to see available fields."
        )
    normalized: list[str] = []
    for index, f in enumerate(fields):
        name = _validated_datapoint_name(spec, index, f)
        if name not in normalized:
            normalized.append(name)
    if len(normalized) > DATAPOINTS_MAX_FIELDS:
        raise ValueError(
            f"get_finra_datapoints accepts at most {DATAPOINTS_MAX_FIELDS} "
            f"fields, got {len(normalized)}. Select the specific fields you "
            "need instead."
        )
    return normalized


def _sort_order_entry(spec: DatasetSpec, sort_order: str) -> str:
    """Single sort entry for the convenience sort_order ('asc'/'desc')."""
    order = sort_order.strip().lower()
    if order not in ("asc", "desc"):
        raise ValueError("sort_order must be 'asc' or 'desc'.")
    date_field = spec.date_field
    if not date_field:
        raise ValueError(
            f"Dataset '{spec.dataset_id}' has no date field, so "
            "'sort_order' cannot be applied; use 'sort_fields' instead."
        )
    return ("-" if order == "desc" else "+") + date_field


def _split_sort_entry(raw: str) -> tuple[str, str]:
    """Split a '+field'/'-field' entry into (sign, name)."""
    text = raw.strip()
    if text and text[0] in "+-":
        return text[0], text[1:]
    return "+", text


def _check_sort_name_known(spec: DatasetSpec, name: str) -> None:
    """Raise when a sort/filter field name is absent from metadata."""
    if spec.field_names and name not in spec.field_names:
        known = ", ".join(sorted(spec.field_names)[:30])
        raise ValueError(f"Dataset '{spec.dataset_id}' has no sortable field '{name}'. Known fields include: {known}")


def _normalize_sort_entry(spec: DatasetSpec, index: int, entry: str) -> str:
    """Validate one '+field'/'-field' string into canonical (sign, name)."""
    raw = entry.strip()
    sign, name = _split_sort_entry(raw)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"sort_fields #{index} is malformed: '{raw}' is not a valid "
            "FINRA sort field. Use '+field' (ascending) or '-field' "
            "(descending)."
        )
    _check_sort_name_known(spec, name)
    return sign + name


def _validate_sort(
    spec: DatasetSpec,
    sort_fields: object,
    sort_order: str | None,
) -> list[str]:
    """Normalize FINRA sortFields entries ('+field' / '-field').

    sort_order is a convenience mapping onto the dataset's date field; it is
    rejected when the dataset has no date field or when sort_fields is also
    given. Raises ValueError before any data request on invalid input.
    """
    if sort_order is not None:
        if sort_fields:
            raise ValueError("Provide either 'sort_fields' or 'sort_order', not both.")
        return [_sort_order_entry(spec, sort_order)]
    if not sort_fields:
        return []
    if not isinstance(sort_fields, list):
        raise ValueError("sort_fields must be a list of '+field'/'-field' strings.")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    normalized: list[str] = []
    for index, entry in enumerate(sort_fields):
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"sort_fields #{index} is malformed: expected '+field' (ascending) or '-field' (descending)."
            )
        normalized.append(_normalize_sort_entry(spec, index, entry))
    return normalized


def _apply_local_sort(rows: list[dict[str, object]], sort_fields: list[str]) -> list[dict[str, object]]:
    """Deterministic local ordering for a single FINRA sort field.

    Guarantees 'latest five' style requests return the requested order even
    when the response ordering is not honored. Missing values always sort
    last. Multi-field sorts rely on FINRA's server-side sortFields ordering.
    """
    if len(sort_fields) != 1:
        return rows
    sign, name = sort_fields[0][0], sort_fields[0][1:]
    descending = sign == "-"

    def _sortable(value: object) -> float | str | None:
        if value is None or value == "":
            return None
        try:
            return float(str(value).replace(",", ""))
        except TypeError, ValueError:
            return str(value)

    def _present_key(r: dict[str, object]) -> float | str:
        v = _sortable(r.get(name))
        return v if v is not None else ""

    def _present_str_key(r: dict[str, object]) -> str:
        return str(r.get(name))

    present = [r for r in rows if _sortable(r.get(name)) is not None]
    missing = [r for r in rows if _sortable(r.get(name)) is None]
    try:
        present.sort(key=_present_key, reverse=descending)
    except TypeError:
        # Mixed numeric/string values in one column: fall back to string order.
        present.sort(key=_present_str_key, reverse=descending)
    return present + missing


def _require_datapoint_narrowing(
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    filters: list[dict[str, object]] | None,
) -> None:
    has_ticker = bool((ticker or "").strip())
    has_date = bool((start_date or "").strip()) or bool((end_date or "").strip())
    has_filter = bool(filters)
    if not (has_ticker or has_date or has_filter):
        raise ValueError(
            "get_finra_datapoints requires at least one narrowing condition: "
            "ticker, date/date range, or filters. Unbounded raw-data requests "
            "are not allowed."
        )


def _select_fields(row: dict[str, object], fields: list[str]) -> dict[str, object]:
    return {f: row.get(f) for f in fields}


def _parse_pagination(headers: dict[str, object], offset: int, limit: int, returned_count: int) -> dict[str, object]:
    """Header-driven pagination; explicit estimate only when FINRA omits
    Record-Total. Self-contained metadata: offset/limit/returned_count are
    included so the analysis layer can prove full-query coverage."""
    total_raw = headers.get("record-total")
    total: int | None = None
    if total_raw is not None:
        try:
            if isinstance(total_raw, bool):
                total = int(total_raw)
            elif isinstance(total_raw, int):
                total = total_raw
            elif isinstance(total_raw, float):
                total = int(total_raw)
            else:
                total = int(str(total_raw).strip())
        except TypeError, ValueError:
            total = None
    base: dict[str, object] = {
        "offset": offset,
        "limit": limit,
        "returned_count": returned_count,
    }
    if total is not None:
        base.update(
            {
                "total_records": total,
                "may_have_more": (offset + returned_count) < total,
                "source": "finra_header",
            }
        )
        return base
    base.update(
        {
            "total_records": None,
            "may_have_more": returned_count >= limit,
            "source": "estimate",
        }
    )
    return base


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _catalog_error_message(exc: BaseException) -> str:
    msg = str(exc)
    if "FINRA_CLIENT" in msg:
        return msg
    return (
        f"FINRA catalog unavailable: {msg}. "
        "Configure FINRA_CLIENT_ID / FINRA_CLIENT_SECRET and ensure network "
        "access to api.finra.org, then retry list_finra_datasets."
    )


def _catalog_disk_entries(hit: object) -> list[CatalogEntry] | None:
    """Catalog entries restored from the SQLite disk cache, else None."""
    if not isinstance(hit, list) or not hit:
        return None
    entries = [_entry_from_dict(d) for d in hit if isinstance(d, dict)]
    with _discovery_lock:
        _catalog_mem[_environment()] = entries
    return entries


def _catalog_is_queryable(e: CatalogEntry) -> bool:
    """Keep queryable, non-retired entries; entitled-only dropped at catalog."""
    inactive = ("retired", "deprecated", "inactive", "terminated")
    if not e.supports_query or e.access == "entitled":
        return False
    return not e.status or e.status.lower() not in inactive


def _catalog_store(cache_key: str, entries: list[CatalogEntry]) -> list[CatalogEntry]:
    """Persist filtered catalog entries to disk + memory caches."""
    cache.set(cache_key, [_entry_to_dict(e) for e in entries])
    with _discovery_lock:
        _catalog_mem[_environment()] = entries
    return entries


def _get_catalog() -> list[CatalogEntry]:
    environment = _environment()
    with _discovery_lock:
        if environment in _catalog_mem:
            return _catalog_mem[environment]
    cache_key = f"finra:v2:{environment}:catalog"
    hit = cache.get(cache_key, ttl=DISCOVERY_TTL_SECONDS)
    restored = _catalog_disk_entries(hit)
    if restored is not None:
        return restored
    raw = _fetch_catalog_http()
    entries: list[CatalogEntry] = []
    for item in raw:
        entry = _normalize_catalog_item(item)
        if entry is not None and _catalog_is_queryable(entry):
            entries.append(entry)
    return _catalog_store(cache_key, entries)


def _catalog_dict_rows(data: list[dict[str, object]]) -> list[dict[str, object]]:
    """String-keyed copies of catalog dicts."""
    return [{str(k): v for k, v in item.items()} for item in data if isinstance(item, dict)]


def _catalog_nested_rows(data: dict[str, object]) -> list[dict[str, object]] | None:
    """Rows from a {"datasets"|"data"|"results"} envelope, else None."""
    for key in ("datasets", "data", "results"):
        nested = data.get(key)
        if isinstance(nested, list):
            return _catalog_dict_rows(nested)
    return None


def _catalog_items_from_json(data: object) -> list[dict[str, object]]:
    """A JSON list of catalog dicts, unwrapping {"datasets"|"data"|"results"}."""
    if isinstance(data, list):
        return _catalog_dict_rows(data)
    if isinstance(data, dict):
        nested = _catalog_nested_rows(data)
        if nested is not None:
            return nested
    raise ValueError("FINRA /datasets response did not contain a dataset list")


def _catalog_group_name(item: dict[str, object]) -> str | None:
    """Canonical group name, or None when the item names no group."""
    group = item.get("group") or item.get("datasetGroup") or item.get("datasetGroupName") or ""
    text = str(group).strip()
    return _canonical_group_name(text) if text else None


def _catalog_dataset_name(item: dict[str, object]) -> str | None:
    """Canonical dataset name, or None when the item names no dataset."""
    name = item.get("name") or item.get("datasetName") or item.get("dataset") or ""
    text = str(name).strip()
    return _canonical_dataset_name(text) if text else None


def _catalog_methods(item: dict[str, object]) -> tuple[str, ...]:
    """Supported HTTP methods from a string 'GET, POST' or a list."""
    methods_raw: object = item.get("supportedMethods") or item.get("methods") or list[object]()
    if isinstance(methods_raw, str):
        return tuple(m.strip() for m in methods_raw.split(",") if m.strip())
    if isinstance(methods_raw, (list, tuple)):
        return tuple(str(m) for m in methods_raw)
    return ()


def _catalog_supports_query(item: dict[str, object], methods: tuple[str, ...]) -> bool:
    """Explicit supportsQuery flag, else any/no-method means queryable."""
    supports_query = item.get("supportsQuery")
    if supports_query is not None:
        return bool(supports_query)
    upper = {m.upper() for m in methods}
    return (not methods) or ("POST" in upper) or ("GET" in upper)


def _catalog_description(item: dict[str, object], group: str, name: str) -> str:
    """Item description, falling back to 'group/name'."""
    return str(item.get("description") or "").strip() or f"{group}/{name}"


def _fetch_catalog_http() -> list[dict[str, object]]:
    url = f"{FINRA_API_BASE}/datasets"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "Accept": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return _catalog_items_from_json(resp.json())


def _normalize_catalog_item(item: dict[str, object]) -> CatalogEntry | None:
    if not isinstance(item, dict):
        return None
    group = _catalog_group_name(item)
    name = _catalog_dataset_name(item)
    if not group or not name:
        return None
    methods = _catalog_methods(item)
    return CatalogEntry(
        group=group,
        name=name,
        description=_catalog_description(item, group, name),
        methods=methods,
        supports_query=_catalog_supports_query(item, methods),
        status=str(item.get("status") or "").strip(),
        access=_parse_access(item),
        supports_record_offset=_parse_optional_bool(
            item.get("supportsRecordOffset"), item.get("supports_record_offset")
        ),
    )


def _parse_access(item: dict[str, object]) -> str:
    """Map any FINRA-provided access/credential metadata to a label.

    FINRA's /datasets response does not document an access field today, so
    real entries resolve to "unknown". We still parse the plausible shapes
    defensively and only claim "public" when explicitly confirmed.
    """
    raw = item.get("access") or item.get("accessType") or item.get("credentialType")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in ("public", "open"):
            return "public"
        if s in ("firm", "organization", "entitled", "restricted", "private"):
            return "entitled"
    is_public = item.get("isPublic")
    if is_public is None:
        is_public = item.get("public")
    if isinstance(is_public, bool):
        return "public" if is_public else "entitled"
    return "unknown"


def _parse_bool_text(value: object) -> bool | None:
    """Bool from one text value ('true/1/yes', 'false/0/no'), else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


def _parse_optional_bool(*values: object) -> bool | None:
    for v in values:
        parsed = _parse_bool_text(v)
        if parsed is not None:
            return parsed
    return None


def _canonical_group_name(value: str) -> str:
    """Return documented casing for a known FINRA dataset group."""
    stripped = value.strip()
    return _CANONICAL_GROUP_NAMES.get(stripped.casefold(), stripped)


def _canonical_dataset_name(value: str) -> str:
    """Return documented casing for a known FINRA dataset name.

    Catalog entries include test datasets with a Mock suffix. Preserve that
    suffix after normalizing the corresponding base name.
    """
    stripped = value.strip()
    folded = stripped.casefold()
    canonical = _CANONICAL_DATASET_NAMES.get(folded)
    if canonical:
        return canonical
    if folded.endswith("mock"):
        base = _CANONICAL_DATASET_NAMES.get(folded[:-4])
        if base:
            return base + "Mock"
    return stripped


def _entry_to_dict(e: CatalogEntry) -> dict[str, object]:
    return {
        "group": e.group,
        "name": e.name,
        "description": e.description,
        "methods": list(e.methods),
        "supports_query": e.supports_query,
        "status": e.status,
        "access": e.access,
        "supports_record_offset": e.supports_record_offset,
    }


def _entry_from_dict(d: dict[str, object]) -> CatalogEntry:
    raw_methods = d.get("methods")
    if isinstance(raw_methods, (list, tuple)):
        methods = tuple(str(m) for m in raw_methods)
    else:
        methods = ()
    raw_sro = d.get("supports_record_offset")
    supports_record_offset = raw_sro if isinstance(raw_sro, bool) else None
    return CatalogEntry(
        group=str(d["group"]),
        name=str(d["name"]),
        description=str(d.get("description") or ""),
        methods=methods,
        supports_query=bool(d.get("supports_query", True)),
        status=str(d.get("status") or ""),
        access=str(d.get("access") or "unknown"),
        supports_record_offset=supports_record_offset,
    )


def _catalog_index(
    entries: list[CatalogEntry],
) -> tuple[dict[str, CatalogEntry], dict[str, list[CatalogEntry]]]:
    """Index catalog entries by id and by lowercased bare name."""
    by_id = {e.dataset_id.lower(): e for e in entries}
    by_name: dict[str, list[CatalogEntry]] = {}
    for e in entries:
        by_name.setdefault(e.name.lower(), []).append(e)
    return by_id, by_name


def _scan_group_name(entries: list[CatalogEntry], raw: str) -> CatalogEntry | None:
    """Entry matching 'group/name' case-insensitively, else None."""
    parts = raw.split("/", 1)
    if len(parts) != 2:
        return None
    g, n = parts[0].strip().lower(), parts[1].strip().lower()
    for e in entries:
        if e.group.lower() == g and e.name.lower() == n:
            return e
    return None


def _resolve_group_name(entries: list[CatalogEntry], by_id: dict[str, CatalogEntry], raw: str) -> CatalogEntry:
    """Resolve a 'group/name' id (case-insensitive, whitespace-tolerant)."""
    if raw.lower() in by_id:
        return by_id[raw.lower()]
    match = _scan_group_name(entries, raw)
    if match is not None:
        return match
    raise ValueError(f"Unknown FINRA dataset '{raw}'. Call list_finra_datasets to browse available datasets.")


def _bare_name_error(dataset_id: str, by_name: dict[str, list[CatalogEntry]]) -> CatalogEntry:
    """Resolve a legacy bare name, or raise with ambiguity/suggestions."""
    raw = dataset_id.strip()
    matches = by_name.get(raw.lower(), [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(m.dataset_id for m in matches)
        raise ValueError(f"Ambiguous FINRA dataset name '{dataset_id}'. Specify one of: {ids}")
    return _bare_name_suggestions(dataset_id, raw)


def _bare_name_suggestions(dataset_id: str, raw: str) -> CatalogEntry:
    """Close bare-name suggestions, else well-known legacy names (never returns)."""
    entries = _get_catalog()
    close = [e for e in entries if raw.lower() in e.name.lower()]
    if close:
        suggestions = ", ".join(e.dataset_id for e in close[:8])
        raise ValueError(
            f"Unknown FINRA dataset '{dataset_id}'. "
            f"Did you mean: {suggestions}? "
            "Call list_finra_datasets to browse available datasets."
        )
    known_hint = ", ".join(DATASET_NAMES[:6])
    raise ValueError(
        f"Unknown FINRA dataset '{dataset_id}'. "
        f"Examples: {known_hint}. "
        "Call list_finra_datasets to browse available datasets."
    )


def _resolve_dataset(dataset_id: str) -> CatalogEntry:
    raw = (dataset_id or "").strip()
    if not raw:
        raise ValueError("Dataset id is required. Use list_finra_datasets to browse the catalog.")
    entries = _get_catalog()
    by_id, by_name = _catalog_index(entries)
    if "/" in raw:
        return _resolve_group_name(entries, by_id, raw)
    return _bare_name_error(dataset_id, by_name)


def _get_dataset_spec(entry: CatalogEntry) -> DatasetSpec:
    key = (_environment(), entry.dataset_id.lower())
    with _discovery_lock:
        if key in _metadata_mem:
            return _metadata_mem[key]

    cache_key = f"finra:v2:{_environment()}:metadata:{entry.group}/{entry.name}"
    hit = cache.get(cache_key, ttl=DISCOVERY_TTL_SECONDS)
    if isinstance(hit, dict) and hit.get("fields") is not None:
        spec = _spec_from_cached(entry, hit)
        with _discovery_lock:
            _metadata_mem[key] = spec
        return spec

    raw = _fetch_metadata_http(entry.group, entry.name)
    spec = _build_spec_from_metadata(entry, raw)
    cache.set(
        cache_key,
        {
            "fields": list(spec.fields),
            "partition_fields": list(spec.partition_fields),
            "description": spec.description,
            "methods": list(spec.methods),
            "symbol_field": spec.symbol_field,
            "date_field": spec.date_field,
            "market_aggregate": spec.market_aggregate,
            "default_filters": [list(p) for p in spec.default_filters],
            "valid_filter_values": {k: list(v) for k, v in spec.valid_filter_values.items()},
        },
    )
    with _discovery_lock:
        _metadata_mem[key] = spec
    return spec


def _fetch_metadata_http(group: str, name: str) -> dict[str, object]:
    # Metadata is public; mock-mode data uses a Mock suffix but metadata
    # is fetched for the base dataset name (fields match).
    url = f"{FINRA_API_BASE}/metadata/group/{group}/name/{name}"
    resp = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metadata response for {group}/{name}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return {str(k): v for k, v in data.items()}


def _get_partitions(spec: DatasetSpec) -> list[tuple[str, ...]]:
    """Cached available partition tuples in FINRA's returned order.

    Returns a list of ordered tuples with one value per partition field in
    spec.partition_fields order (e.g. ("2026-08-10", "T1")). Raises
    ValueError when the dataset has no partition fields or FINRA cannot be
    reached.
    """
    if not spec.partition_fields:
        raise ValueError(f"Dataset '{spec.dataset_id}' has no partition fields.")
    key = (_environment(), spec.dataset_id.lower())
    with _discovery_lock:
        if key in _partitions_mem:
            return _partitions_mem[key]

    cache_key = f"finra:v3:{_environment()}:partitions:{spec.group}/{spec.name}"
    hit = cache.get(cache_key, ttl=DISCOVERY_TTL_SECONDS)
    if _is_partition_tuple_cache(hit, len(spec.partition_fields)):
        parsed = [tuple(str(v) for v in entry) for entry in hit]
        with _discovery_lock:
            _partitions_mem[key] = parsed
        return parsed

    raw = _fetch_partitions_http(spec.group, _dataset_path_name(spec))
    parsed = _parse_partitions(raw, spec.partition_fields)
    cache.set(cache_key, [list(t) for t in parsed])
    with _discovery_lock:
        _partitions_mem[key] = parsed
    return parsed


def _is_partition_tuple_cache(hit: object, n_fields: int) -> TypeGuard[list[Sequence[object]]]:
    """Validate a JSON-safe cached partitions payload.

    The SQLite cache serializes tuples as JSON lists, so cached entries
    arrive as fixed-length lists (tuples come only from in-memory stores).
    Returns True only for a list of entries that each carry exactly one
    value per partition field; anything else — e.g. the old flattened
    {field: [values]} v1 shape — is rejected so the data is refetched.
    """
    if not isinstance(hit, list):
        return False
    for entry in hit:
        if not isinstance(entry, (list, tuple)):
            return False
        if len(entry) != n_fields:
            return False
        if not all(isinstance(v, (str, int, float)) and str(v) for v in entry):
            return False
    return True


def _fetch_partitions_http(group: str, name: str) -> dict[str, object]:
    url = f"{FINRA_API_BASE}/partitions/group/{group}/name/{name}"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "Accept": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected partitions response for {group}/{name}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return {str(k): v for k, v in data.items()}


def _partition_dict_tuple(item: dict[str, object], n: int) -> tuple[str, ...] | None:
    """Tuple for a {"partitions": [...]} entry, or None when malformed."""
    values = item.get("partitions")
    if not isinstance(values, list):
        return None
    vals = [str(v) for v in values]
    if len(vals) != n or not all(vals):
        return None
    return tuple(vals)


def _partition_scalar_tuple(item: object, n: int) -> tuple[str, ...] | None:
    """Tuple for a scalar entry (single-partition datasets only), else None."""
    if not isinstance(item, (str, int, float)):
        return None
    if n != 1:
        return None
    return (str(item),)


def _partition_entry_tuple(item: object, n: int) -> tuple[str, ...] | None:
    """Tuple for one availablePartitions entry, or None when unusable."""
    if isinstance(item, dict):
        return _partition_dict_tuple(item, n)
    return _partition_scalar_tuple(item, n)


def _parse_partitions(raw: dict[str, object], partition_fields: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Normalize availablePartitions into ordered partition tuples.

    Each entry carries one value per partition field in field order (FINRA
    tuple semantics); entry order is preserved. Scalar entries are only
    meaningful for single-partition datasets and are otherwise dropped —
    an ambiguous single value cannot be placed safely without inventing
    combinations FINRA never published.
    """
    if not partition_fields:
        return []
    items = raw.get("availablePartitions")
    if not isinstance(items, list):
        return []
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    n = len(partition_fields)
    for item in items:
        tup = _partition_entry_tuple(item, n)
        if tup is None or tup in seen:
            continue
        seen.add(tup)
        out.append(tup)
    return out


def _date_partition_field(spec: DatasetSpec) -> str | None:
    """The partition field that carries the authoritative date.

    Returns spec.date_field when it is itself a partition field. When the
    displayed/sort date differs from the partition date (e.g. weeklySummary
    sorts by summaryStartDate but partitions by weekStartDate), a verified
    mapping in _DATE_PARTITION_MAPPINGS is required; without one, partition
    walking is rejected (None).
    """
    if spec.date_field and spec.date_field in spec.partition_fields:
        return spec.date_field
    mapped = _DATE_PARTITION_MAPPINGS.get((spec.group.lower(), spec.name.lower()))
    if mapped and mapped in spec.partition_fields:
        return mapped
    return None


def _ordered_partition_tuples(
    spec: DatasetSpec,
    partitions: list[tuple[str, ...]],
    pinned: dict[str, str],
    date_field: str | None,
    descending: bool,
) -> list[dict[str, str]]:
    """Order the original FINRA partition tuples by the date partition.

    Only tuples FINRA actually published are used — never a Cartesian
    product of per-field values. Tuples pinned by the caller (EQUAL filter
    supplied) are fixed; the date partition drives global ordering
    (newest-first for descending sorts, oldest-first otherwise); equal
    dates keep FINRA's returned order.
    """
    primary = _date_partition_field(spec)
    if primary is None or not partitions:
        return []
    index = {f: i for i, f in enumerate(spec.partition_fields)}
    primary_idx = index.get(primary)
    if primary_idx is None:
        return []

    def _date_key(entry: tuple[str, ...]) -> tuple[int, str]:
        value = entry[primary_idx]
        norm = _norm_date(value)
        if norm is None:
            return (1, value)
        return (0, norm)

    filtered: list[tuple[tuple[int, str], dict[str, str]]] = []
    for entry in partitions:
        values = dict(zip(spec.partition_fields, entry))
        if any(values.get(f) != pinned[f] for f in pinned):
            continue
        filtered.append((_date_key(entry), values))
    filtered.sort(key=itemgetter(0), reverse=descending)
    return [values for _, values in filtered]


def reset_partitions_cache() -> None:
    """Test helper — clears the in-memory partitions cache."""
    global _partitions_mem
    with _discovery_lock:
        _partitions_mem = {}


def _normalize_metadata_field(f: object) -> dict[str, object] | None:
    """One metadata field dict, or None when it has no usable name."""
    if not isinstance(f, dict) or not f.get("name"):
        return None
    return {
        "name": str(f.get("name")),
        "type": str(f.get("type") or f.get("dataType") or ""),
        "description": str(f.get("description") or ""),
        **({"format": str(f.get("format"))} if f.get("format") else {}),
    }


def _metadata_fields(raw: dict[str, object]) -> list[dict[str, object]]:
    """Normalized field dicts from live metadata ('fields'/'datasetFields')."""
    raw_fields: object = raw.get("fields") or raw.get("datasetFields") or list[object]()
    if not isinstance(raw_fields, (list, tuple)):
        return []
    out: list[dict[str, object]] = []
    for f in raw_fields:
        row = _normalize_metadata_field(f)
        if row is not None:
            out.append(row)
    return out


def _metadata_partitions(raw: dict[str, object]) -> tuple[str, ...]:
    """Partition field names from live metadata ('partitionFields'/'partitions')."""
    raw_partitions: object = raw.get("partitionFields") or raw.get("partitions") or list[object]()
    if not isinstance(raw_partitions, (list, tuple)):
        return ()
    return tuple(str(p) for p in raw_partitions)


def _metadata_description(entry: CatalogEntry, raw: dict[str, object]) -> str:
    """Live description, else catalog description, else dataset id."""
    return str(raw.get("description") or "").strip() or entry.description or entry.dataset_id


def _parse_live_methods(raw_sm: object) -> tuple[str, ...]:
    """Live supportedMethods (string or list) as a tuple, else empty."""
    if isinstance(raw_sm, (list, tuple)):
        return tuple(str(m) for m in raw_sm)
    if isinstance(raw_sm, str):
        return tuple(m.strip() for m in raw_sm.split(",") if m.strip())
    return ()


def _metadata_methods(entry: CatalogEntry, raw: dict[str, object]) -> tuple[str, ...]:
    """Catalog methods, else live supportedMethods (string or list)."""
    if entry.methods:
        return entry.methods
    raw_sm: object = raw.get("supportedMethods") or list[object]()
    return _parse_live_methods(raw_sm)


def _opt_str(value: object) -> str | None:
    """Non-blank string, else None."""
    return value if isinstance(value, str) and value else None


def _pair_list(raw: object) -> tuple[tuple[str, str], ...]:
    """List of 2-element string pairs (default_filters shapes)."""
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple((str(pair[0]), str(pair[1])) for pair in raw if isinstance(pair, (list, tuple)) and len(pair) == 2)


def _str_tuple_map(raw: object) -> dict[str, tuple[str, ...]]:
    """Dict of string->tuple-of-strings (valid_filter_values shapes)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            out[str(k)] = tuple(str(x) for x in v)
    return out


def _override_named_fields(
    entry: CatalogEntry,
    detected_symbol: str | None,
    detected_date: str | None,
) -> tuple[str | None, str | None, bool]:
    """Registry symbol/date/market corrections (unvalidated against metadata)."""
    symbol_field = detected_symbol
    date_field = detected_date
    market_aggregate = False
    override = _METADATA_OVERRIDES.get((entry.group.lower(), entry.name.lower()), {})
    if "symbol_field" in override:
        symbol_field = _opt_str(override["symbol_field"])
    if "date_field" in override:
        date_field = _opt_str(override["date_field"])
    if override.get("market_aggregate"):
        market_aggregate = True
        symbol_field = None
    return symbol_field, date_field, market_aggregate


def _validated_override_fields(
    fields: list[dict[str, object]],
    partition_fields: tuple[str, ...],
    symbol_field: str | None,
    date_field: str | None,
) -> tuple[str | None, str | None]:
    """Override names validated against metadata; stale names fall back to detection."""
    field_names = {f["name"] for f in fields}
    if symbol_field and field_names and symbol_field not in field_names:
        symbol_field = _detect_symbol_field(fields)
    if date_field and field_names and date_field not in field_names:
        date_field = _detect_date_field(fields, partition_fields)
    return symbol_field, date_field


def _override_corrections(
    entry: CatalogEntry,
    fields: list[dict[str, object]],
    partition_fields: tuple[str, ...],
) -> tuple[str | None, str | None, bool, tuple[tuple[str, str], ...], dict[str, tuple[str, ...]]]:
    """Detected symbol/date fields corrected by registry overrides.

    Returns (symbol_field, date_field, market_aggregate, default_filters,
    valid_filter_values). Override names absent from metadata fall back to
    detection — live metadata stays authoritative over a stale registry.
    """
    symbol, dated, market = _override_named_fields(
        entry, _detect_symbol_field(fields), _detect_date_field(fields, partition_fields)
    )
    symbol, dated = _validated_override_fields(fields, partition_fields, symbol, dated)
    override = _METADATA_OVERRIDES.get((entry.group.lower(), entry.name.lower()), {})
    return (
        symbol,
        dated,
        market,
        _pair_list(override.get("default_filters")),
        _str_tuple_map(override.get("valid_filter_values")),
    )


def _field_name_set(fields: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Metadata fields indexed by name."""
    by_name: dict[str, dict[str, object]] = {}
    for f in fields:
        raw = f.get("name")
        if isinstance(raw, str) and raw:
            by_name[raw] = f
    return by_name


def _symbol_like_name(f: dict[str, object]) -> str | None:
    """Field name when it looks like a string symbol column, else None."""
    raw_n = f.get("name")
    n = raw_n if isinstance(raw_n, str) else ""
    if not n:
        return None
    raw_t = f.get("type")
    t = raw_t.lower() if isinstance(raw_t, str) else ""
    if "symbol" in n.lower() and t in ("string", "text", ""):
        return n
    return None


def _date_partition_hit(by_name: dict[str, dict[str, object]], partition: str) -> str | None:
    """Partition name when it carries (or looks like) a date, else None."""
    f = by_name.get(partition)
    if f is None:
        return None
    if _is_date_type(f.get("type")):
        return partition
    if re.search(r"date|datetime|time", partition, re.IGNORECASE):
        return partition
    return None


def _first_date_typed_field(fields: list[dict[str, object]]) -> str | None:
    """First field with a date-ish type annotation, else None."""
    for f in fields:
        if _is_date_type(f.get("type")):
            return _as_name(f.get("name"))
    return None


def _first_date_named_field(fields: list[dict[str, object]]) -> str | None:
    """First field with a date-ish name, else None."""
    for f in fields:
        raw_n = f.get("name")
        n = raw_n if isinstance(raw_n, str) else ""
        if not n:
            continue
        if re.search(r"(^|.*)(date|datetime)$", n, re.IGNORECASE) or "Date" in n:
            return n
    return None


def _build_spec_from_metadata(entry: CatalogEntry, raw: dict[str, object]) -> DatasetSpec:
    fields = _metadata_fields(raw)
    partition_fields = _metadata_partitions(raw)
    symbol_field, date_field, market, defaults, valid = _override_corrections(entry, fields, partition_fields)
    return DatasetSpec(
        group=entry.group,
        name=entry.name,
        description=_metadata_description(entry, raw),
        fields=tuple(fields),
        partition_fields=partition_fields,
        methods=_metadata_methods(entry, raw),
        symbol_field=symbol_field,
        date_field=date_field,
        market_aggregate=market,
        default_filters=defaults,
        valid_filter_values=valid,
    )


def _cached_str_tuple_map(hit: dict[str, object], key: str) -> dict[str, tuple[str, ...]]:
    """valid_filter_values restored from a cached metadata payload."""
    return _str_tuple_map(hit.get(key))


def _cached_field_rows(hit: dict[str, object]) -> list[dict[str, object]]:
    """Cached field rows with stringified keys."""
    raw_fields = hit.get("fields")
    if not isinstance(raw_fields, (list, tuple)):
        return []
    return [{str(k): v for k, v in item.items()} for item in raw_fields if isinstance(item, dict)]


def _cached_partitions(hit: dict[str, object]) -> tuple[str, ...]:
    """Cached partition field names."""
    raw_pf = hit.get("partition_fields")
    if not isinstance(raw_pf, (list, tuple)):
        return ()
    return tuple(str(p) for p in raw_pf)


def _cached_methods(entry: CatalogEntry, hit: dict[str, object]) -> tuple[str, ...]:
    """Cached supported methods, else the catalog entry's methods."""
    raw_methods = hit.get("methods")
    if isinstance(raw_methods, (list, tuple)):
        return tuple(str(m) for m in raw_methods)
    return entry.methods


def _cached_description(entry: CatalogEntry, hit: dict[str, object]) -> str:
    """Cached description, else the catalog entry's description."""
    raw_desc = hit.get("description")
    if isinstance(raw_desc, str) and raw_desc:
        return raw_desc
    return entry.description


def _spec_from_cached(entry: CatalogEntry, hit: dict[str, object]) -> DatasetSpec:
    return DatasetSpec(
        group=entry.group,
        name=entry.name,
        description=_cached_description(entry, hit),
        fields=tuple(_cached_field_rows(hit)),
        partition_fields=_cached_partitions(hit),
        methods=_cached_methods(entry, hit),
        symbol_field=_opt_str(hit.get("symbol_field")),
        date_field=_opt_str(hit.get("date_field")),
        market_aggregate=bool(hit.get("market_aggregate")),
        default_filters=_pair_list(hit.get("default_filters")),
        valid_filter_values=_cached_str_tuple_map(hit, "valid_filter_values"),
    )


def _detect_symbol_field(fields: list[dict[str, object]]) -> str | None:
    by_name = _field_name_set(fields)
    for preferred in _PREFERRED_SYMBOL_FIELDS:
        if preferred in by_name:
            return preferred
    for f in fields:
        hit = _symbol_like_name(f)
        if hit is not None:
            return hit
    return None


def _detect_date_field(fields: list[dict[str, object]], partition_fields: tuple[str, ...]) -> str | None:
    by_name = _field_name_set(fields)
    for p in partition_fields:
        hit = _date_partition_hit(by_name, p)
        if hit is not None:
            return hit
    typed = _first_date_typed_field(fields)
    if typed is not None:
        return typed
    return _first_date_named_field(fields)


def _is_date_type(t: object) -> bool:
    s = str(t or "").lower()
    return s in ("date", "datetime", "timestamp") or "date" in s


# ---------------------------------------------------------------------------
# Query payload + HTTP
# ---------------------------------------------------------------------------


def _payload_base(
    entry: CatalogEntry,
    limit: int | None,
    offset: int | None,
    fields: list[str] | None,
    sort_fields: list[str] | None,
) -> dict[str, object]:
    """Payload envelope: limit plus optional offset/fields/sortFields."""
    payload: dict[str, object] = {"limit": _clamp_limit(limit)}
    if offset is not None:
        payload["offset"] = _validate_offset(entry, offset)
    if fields:
        payload["fields"] = list(fields)
    if sort_fields:
        payload["sortFields"] = list(sort_fields)
    return payload


def _normalize_filter_name(index: int, extra: Mapping[str, object]) -> str:
    """Validated non-blank filter field name."""
    field_name = extra.get("field")
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError(f"Filter #{index} is malformed: a non-empty 'field' is required.")
    return field_name.strip()


def _normalize_filter_value(field_name: str, extra: Mapping[str, object]) -> object:
    """Validated filter value (present and non-blank)."""
    value = extra.get("value")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Filter for field '{field_name}' is malformed: 'value' is required.")
    return value


def _normalize_filter_op(extra: Mapping[str, object]) -> str:
    """Validated uppercase compare op."""
    op = str(extra.get("op") or "EQUAL").upper()
    if op not in _ALLOWED_COMPARE:
        raise ValueError(f"Unsupported compare op '{op}'. Use one of: {sorted(_ALLOWED_COMPARE)}")
    return op


def _check_filter_known(spec: DatasetSpec, field_name: str, value: object) -> None:
    """Raise for unknown filter fields or disallowed enum values."""
    if spec.field_names and field_name not in spec.field_names:
        known = ", ".join(sorted(spec.field_names)[:30])
        raise ValueError(
            f"Dataset '{spec.dataset_id}' has no filterable field '{field_name}'. Known fields include: {known}"
        )
    allowed = spec.valid_filter_values.get(field_name)
    if allowed is not None and str(value) not in allowed:
        raise ValueError(
            f"Invalid value '{value}' for field '{field_name}' on "
            f"'{spec.dataset_id}'. Allowed values: {', '.join(allowed)}"
        )


def _payload_filter_rows(
    spec: DatasetSpec, filters: Sequence[Mapping[str, object]] | None
) -> tuple[list[dict[str, object]], set[str]]:
    """Normalized (compare rows, explicit field names) for caller filters."""
    rows: list[dict[str, object]] = []
    explicit: set[str] = set()
    for index, extra in enumerate(filters or []):
        if not isinstance(extra, dict):
            raise ValueError(  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
                f"Filter #{index} must be an object with 'field' and 'value'."
            )
        field_name = _normalize_filter_name(index, extra)
        value = _normalize_filter_value(field_name, extra)
        op = _normalize_filter_op(extra)
        _check_filter_known(spec, field_name, value)
        explicit.add(field_name)
        rows.append({"compareType": op, "fieldName": field_name, "fieldValue": str(value)})
    return rows, explicit


def _payload_symbol_row(spec: DatasetSpec, ticker: str | None) -> dict[str, object] | None:
    """EQUAL ticker row, or None when no ticker was requested."""
    symbol = (ticker or "").strip().upper() or None
    if not symbol:
        return None
    if spec.market_aggregate or not spec.symbol_field:
        raise ValueError(
            f"Dataset '{spec.dataset_id}' is a market-wide aggregate with no "
            "ticker/symbol field. Omit ticker (and ISIN) filters; filter by "
            "date or other fields instead."
        )
    return {"compareType": "EQUAL", "fieldName": spec.symbol_field, "fieldValue": symbol}


def _payload_default_rows(spec: DatasetSpec, explicit_fields: set[str]) -> list[dict[str, object]]:
    """Registry default filters skipped when the caller filtered that field."""
    rows: list[dict[str, object]] = []
    for field_name, value in spec.default_filters:
        if field_name in explicit_fields:
            continue
        rows.append({"compareType": "EQUAL", "fieldName": field_name, "fieldValue": value})
    return rows


def _clean_date_opt(value: str | None) -> str | None:
    """Stripped date text, or None when blank."""
    text = (value or "").strip()
    return text if text else None


def _require_payload_date_field(spec: DatasetSpec) -> str:
    """Authoritative date field, or raise when the dataset has none."""
    if not spec.date_field:
        raise ValueError(f"Dataset '{spec.dataset_id}' has no date field to filter on.")
    return spec.date_field


def _payload_range_filter(date_field: str, start: str, end: str) -> dict[str, object]:
    """A FINRA dateRangeFilters entry for a true start/end range."""
    return {"fieldName": date_field, "startDate": start, "endDate": end}


def _payload_equal_date_row(date_field: str, value: object) -> dict[str, object]:
    """An EQUAL compare row for a single-date request."""
    return {"compareType": "EQUAL", "fieldName": date_field, "fieldValue": value}


def _payload_date_parts(
    spec: DatasetSpec,
    start_date: str | None,
    end_date: str | None,
    payload: dict[str, object],
    compare: list[dict[str, object]],
) -> None:
    """Attach the dateRange EQUAL/range filters for a start/end request."""
    start = _clean_date_opt(start_date)
    end = _clean_date_opt(end_date)
    if not start and not end:
        return
    date_field = _require_payload_date_field(spec)
    if start and end and start != end:
        payload["dateRangeFilters"] = [_payload_range_filter(date_field, start, end)]
        return
    compare.append(_payload_equal_date_row(date_field, start or end))


def _build_payload(
    spec: DatasetSpec,
    entry: CatalogEntry,
    ticker: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int | None,
    filters: Sequence[Mapping[str, object]] | None,
    offset: int | None = None,
    fields: list[str] | None = None,
    sort_fields: list[str] | None = None,
) -> dict[str, object]:
    payload = _payload_base(entry, limit, offset, fields, sort_fields)
    normalized_extras, explicit_fields = _payload_filter_rows(spec, filters)
    compare: list[dict[str, object]] = []
    symbol_row = _payload_symbol_row(spec, ticker)
    if symbol_row is not None:
        compare.append(symbol_row)
    # Default filters (e.g. weeklySummary summaryTypeCode) only when the
    # caller did not already filter that field — avoids conflicting filters.
    compare.extend(_payload_default_rows(spec, explicit_fields))
    _payload_date_parts(spec, start_date, end_date, payload, compare)
    compare.extend(normalized_extras)
    if compare:
        payload["compareFilters"] = compare
    return payload


def _clamp_limit(limit: int | None, *, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    if limit is None:
        return default
    try:
        n = limit
    except TypeError, ValueError:
        return default
    return max(1, min(n, maximum))


def _coerce_offset(offset: object) -> int:
    """Integer offset, or raise for non-integer input."""
    try:
        if isinstance(offset, bool):
            return int(offset)
        if isinstance(offset, (int, float)):
            return int(offset)
        if isinstance(offset, str):
            return int(offset.strip())
    except TypeError, ValueError:
        raise ValueError(f"offset must be an integer, got {offset!r}.")
    raise ValueError(f"offset must be an integer, got {offset!r}.")


def _check_offset_bounds(n: int) -> None:
    """Raise for negative offsets or offsets past FINRA's maximum."""
    if n < 0:
        raise ValueError("offset must be >= 0.")
    if n > MAX_OFFSET:
        raise ValueError(
            f"offset {n} exceeds FINRA's maximum of {MAX_OFFSET}. Use filters to narrow the result set instead."
        )


def _check_offset_supported(entry: CatalogEntry) -> None:
    """Raise when the catalog marks record-offset pagination unsupported."""
    if entry.supports_record_offset is False:
        raise ValueError(
            f"Dataset '{entry.dataset_id}' does not support record offset "
            "pagination (supportsRecordOffset=false in the FINRA catalog)."
        )


def _validate_offset(entry: CatalogEntry, offset: object) -> int:
    n = _coerce_offset(offset)
    _check_offset_bounds(n)
    _check_offset_supported(entry)
    return n


def _dataset_path_name(spec: DatasetSpec) -> str:
    if finra_use_mock():
        return spec.name + "Mock"
    return spec.name


def _query_cache_key(spec: DatasetSpec, payload: dict[str, object]) -> str:
    path_name = _dataset_path_name(spec)
    return f"finra:v2:{_environment()}:query:{spec.group}:{path_name}:{json.dumps(payload, sort_keys=True)}"


def _narrow_cached_dict_hit(hit: object) -> tuple[list[object], dict[object, object]] | None:
    """Raw (record list, header dict) from a dict cache hit, else None."""
    if not isinstance(hit, dict) or "records" not in hit:
        return None
    raw_records = hit.get("records")
    raw_headers = hit.get("headers") or {}
    if not isinstance(raw_records, list) or not isinstance(raw_headers, dict):
        return None
    return raw_records, raw_headers


def _cached_dict_hit(
    hit: object,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    """(records, headers) from a dict-shaped query cache hit, else None."""
    narrowed = _narrow_cached_dict_hit(hit)
    if narrowed is None:
        return None
    raw_records, raw_headers = narrowed
    records = [{str(k): v for k, v in r.items()} for r in raw_records if isinstance(r, dict)]
    headers: dict[str, object] = {str(k): v for k, v in raw_headers.items()}
    return records, headers


def _cached_list_hit(hit: object) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    """(records, {}) from a legacy plain-list cache hit, else None."""
    if not isinstance(hit, list):
        return None
    # Legacy cached plain list (pre-analysis layer): no headers.
    return [{str(k): v for k, v in r.items()} for r in hit if isinstance(r, dict)], {}


def _cached_query(spec: DatasetSpec, payload: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache_key = _query_cache_key(spec, payload)
    hit = cache.get(cache_key, ttl=CACHE_TTL_SECONDS)
    if hit is not None:
        restored = _cached_dict_hit(hit) or _cached_list_hit(hit)
        if restored is not None:
            return restored
    _, records, headers = ingestion_post_query(spec.group, _dataset_path_name(spec), payload)
    cache.set(cache_key, {"records": records, "headers": headers})
    return records, headers


def ingestion_post_query(
    group: str, dataset_name: str, payload: dict[str, object]
) -> tuple[bytes, list[dict[str, object]], dict[str, object]]:
    """Raw FINRA data-plane POST used by the research data refresh service (app/services/research_data.py).

    Returns (response body bytes, parsed records, captured pagination
    headers).  It bypasses the chat SQLite cache on purpose: the immutable
    raw archive is the durable store, and the pipeline decides what to
    re-fetch via its checkpoints.
    """
    url = f"{FINRA_API_BASE}/data/group/{group}/name/{dataset_name}"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    logger.debug(
        "FINRA data POST: url=%s status=%s payload=%s",
        url,
        resp.status_code,
        _sanitize_payload(payload),
    )
    resp.raise_for_status()
    headers: dict[str, object] = {
        name.lower(): value for name, value in resp.headers.items() if name.lower() in _RECORD_HEADERS
    }
    # FINRA can return a successful empty response for a partition with no
    # matching rows. Continue the partition walk instead of parsing it as JSON.
    if not resp.content or not resp.content.strip():
        return resp.content, [], headers
    data = resp.json()
    return resp.content, _extract_records(data), headers


def _record_rows(items: object) -> list[dict[str, object]]:
    """String-keyed dict rows from a JSON list, dropping non-dict entries."""
    if not isinstance(items, list):
        return []
    return [{str(k): v for k, v in item.items()} for item in items if isinstance(item, dict)]


def _nested_record_rows(data: dict[str, object]) -> list[dict[str, object]] | None:
    """Rows from a {"data"|"records"|"results"} envelope, else None."""
    for key in ("data", "records", "results"):
        nested = data.get(key)
        if isinstance(nested, list):
            return _record_rows(nested)
    return None


def _dict_envelope_rows(data: dict[str, object]) -> list[dict[str, object]]:
    """Rows from a dict payload: nested envelope, else the dict itself."""
    nested = _nested_record_rows(data)
    if nested is not None:
        return nested
    return [{str(k): v for k, v in data.items()}]


def _extract_records(data: object) -> list[dict[str, object]]:
    if data is None:
        return []
    if isinstance(data, list):
        return _record_rows(data)
    if isinstance(data, dict):
        return _dict_envelope_rows(data)
    return []


def _access_token() -> str:
    global _cached_token, _token_expires_at
    now = time.time()
    with _token_lock:
        if _cached_token and now < _token_expires_at:
            return _cached_token
        token, expires_in = _fetch_token()
        _cached_token = token
        _token_expires_at = now + max(expires_in - TOKEN_SKEW_SECONDS, 30)
        return token


def reset_token_cache() -> None:
    """Test helper."""
    global _cached_token, _token_expires_at
    with _token_lock:
        _cached_token = None
        _token_expires_at = 0.0


def reset_discovery_cache() -> None:
    """Test helper — clears in-memory catalog/metadata caches."""
    global _catalog_mem, _metadata_mem
    with _discovery_lock:
        _catalog_mem = {}
        _metadata_mem = {}


def _fetch_token() -> tuple[str, int]:
    client_id = get_finra_client_id()
    client_secret = get_finra_client_secret()
    resp = requests.post(
        FINRA_TOKEN_URL,
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise ValueError("FINRA token response missing access_token")
    expires_in = int(data.get("expires_in") or 3600)
    return token, expires_in
