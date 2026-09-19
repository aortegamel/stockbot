"""All SEC EDGAR access lives here. tools.py never imports edgartools directly."""

from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import itertools
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from .config import get_data_root, init_config

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(get_data_root() / "edgar"))

from edgar import Company, Filing
from edgar.urls import build_archive_url, build_company_facts_url

from . import cache

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        init_config()
        # edgartools identity lives in app/sec/client.py (the edgar boundary).
        from app.sec.client import ensure_identity

        ensure_identity()
        _initialized = True


def _no_data(ticker: str, what: str) -> dict[str, object]:
    return {"error": f"No data found for {ticker}: {what}"}


def get_company(ticker: str) -> Company:
    """edgartools Company handle for a ticker (lazy init inside)."""
    _ensure_init()
    from app.sec.client import get_company as _sec_get_company

    return _sec_get_company(ticker)


def get_latest_report(ticker: str, form_type: str = "10-K"):
    """Latest filing object + parsed document for one form, or None.

    The single seam behind the obligations/valuation report reads; keeps the
    ``edgar`` import boundary inside this module."""
    filings = get_company(ticker).get_filings(form=[form_type])
    if not filings:
        return None
    filing = filings[0]
    assert isinstance(filing, Filing)
    return filing, filing.obj()


SEC_RESULT_CACHE_TTL_SECONDS = 86_400


def _utc_now_iso() -> str:
    return datetime.now(_dt.UTC).isoformat()


def _result_content_hash(payload: dict[str, object]) -> str:
    tmp = {k: v for k, v in payload.items() if k not in ("cache_hit", "cache_type")}
    return hashlib.sha256(json.dumps(tmp, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _companyfacts_url(cik: object) -> str | None:
    """Provenance URL for a CIK via EdgarTools (mirror-aware, None when bad)."""
    if isinstance(cik, (int, float, str, bytes)):
        try:
            return build_company_facts_url(int(cik))
        except TypeError, ValueError:
            return None
    return None


def _filing_dir_url(cik: object, accession: object) -> str | None:
    """Filing-directory URL via EdgarTools (mirror-aware, None when bad)."""
    if not isinstance(cik, (int, float, str, bytes)):
        return None
    try:
        acc = str(accession).replace("-", "")
        return build_archive_url(f"data/{int(cik)}/{acc}/")
    except TypeError, ValueError:
        return None


def _cached_or_fetch(key: str, fetch: Callable[[], dict[str, object]]) -> dict[str, object]:
    """24h parsed-result cache; retrieved_at preserved, transient flags unstored."""
    hit = cache.get(key, ttl=SEC_RESULT_CACHE_TTL_SECONDS)
    if isinstance(hit, dict):
        out = dict(hit)
        out["cache_hit"] = True
        out["cache_type"] = "stockbot_parsed"
        return out
    # Non-dict hit (corrupt cache): fall through and refetch, as before.
    value = fetch()
    if isinstance(value, dict) and "error" not in value:
        canonical = dict(value)
        canonical.setdefault("retrieved_at", _utc_now_iso())
        if "result_content_hash" not in canonical:
            canonical["result_content_hash"] = _result_content_hash(canonical)
        try:
            cache.set(key, canonical)
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        out = dict(canonical)
        out["cache_hit"] = False
        out["cache_type"] = "live_or_edgartools_http"
        return out
    if isinstance(value, dict):
        out = dict(value)
        out["cache_hit"] = False
        out["cache_type"] = "live_or_edgartools_http"
        return out
    return value


_QUARTER_DAYS = (60, 115)
_YTD_DAYS = (240, 300)
_FY_DAYS = (330, 400)
_MISSING_QUARTER_GAP_DAYS = 130
_DERIVED_Q4_OFFSET_DAYS = 91


def _facts_dataframe(facts) -> pd.DataFrame:
    """EntityFacts frame with filing metadata when the SDK offers it."""
    try:
        return facts.to_dataframe(include_metadata=True)
    except TypeError:
        return facts.to_dataframe()


def _fact_duration_days(frame: pd.DataFrame) -> pd.Series[int]:
    """Duration in days between period_start and period_end (XBRL facts)."""
    import pandas as pd

    start = pd.to_datetime(frame["period_start"], errors="coerce")
    end = pd.to_datetime(frame["period_end"], errors="coerce")
    return (end - start).dt.days


def _dedup_latest(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate period_end facts, keeping the most recently filed one.

    XBRL company facts can carry restated values for the same period. Prefer
    the true filed order (filing_date, then largest accession as tiebreak);
    when the SDK frame carries no date columns, fall back to the largest
    fiscal_year proxy (restatements are tagged with the year reported in).
    """
    # ponytail: fiscal_year proxy ceiling — per-filing version chains if the
    # SDK ever stops tagging restatements with their report year.
    import pandas as pd

    date_col = next((c for c in ("filing_date", "filed", "filed_at") if c in frame.columns), None)
    if date_col is None:
        keys = ["period_end", "fiscal_year"] if "fiscal_year" in frame.columns else ["period_end"]
        return frame.sort_values(keys).drop_duplicates(subset=["period_end"], keep="last")
    work = frame.copy()
    work["_filed"] = pd.to_datetime(work[date_col], errors="coerce")
    keys = ["period_end", "_filed"]
    acc_col = next((c for c in ("accession", "accn") if c in work.columns), None)
    if acc_col is not None:
        keys.append(acc_col)
    if "fiscal_year" in work.columns:
        keys.append("fiscal_year")
    out = work.sort_values(keys, na_position="first").drop_duplicates(subset=["period_end"], keep="last")
    return out.drop(columns=["_filed"])


def _quarters_with_derived_q4(quarterly: pd.DataFrame, full_facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Return the last 4 quarterly facts, deriving a missing quarter end.

    XBRL company facts hold quarterly (~3-month), YTD (6-9 month), and
    full-year values for the same period_end. NVDA reports its Q4 diluted
    EPS only as a full-year fact, so the Q4 quarter is derived as
    FY_total - YTD_through_Q3 from the matching YTD and FY facts. Without
    this, the TTM window silently drops Q4 and double-counts a stale
    quarter (old behavior produced 8.13 instead of 6.53 for NVDA).
    """
    import pandas as pd

    quarter = quarterly.copy().sort_values("period_end")
    if len(quarter) < 2:
        return quarter.tail(4)
    ends = pd.to_datetime(quarter["period_end"], errors="coerce")
    gaps = ends.diff().dt.days
    if len(gaps) >= 2 and gaps.iloc[-1] is not None and float(gaps.iloc[-1]) > _MISSING_QUARTER_GAP_DAYS:
        # One quarter between the last two period_ends is missing (usually
        # Q4, reported only as a full-year fact). The missing quarter ends
        # ~91 days before the latest period_end.
        missing_end = ends.iloc[-1] - pd.Timedelta(days=_DERIVED_Q4_OFFSET_DAYS)
        derived = _derive_q4_from_facts(full_facts, concept, missing_end)
        if derived is not None:
            quarter = pd.concat([quarter, derived], ignore_index=True).sort_values("period_end")
    return quarter.tail(4)


def _derive_q4_from_facts(full_facts: pd.DataFrame, concept: str, fy_end: pd.Timestamp) -> pd.DataFrame | None:
    """Derive Q4 EPS = FY_total - YTD_through_Q3 for the fiscal year ending fy_end."""
    import pandas as pd

    facts = full_facts[full_facts["concept"].isin([concept, concept.split(":")[-1]])].copy()
    facts["duration_days"] = _fact_duration_days(facts)
    fy_end = pd.Timestamp(fy_end)
    fy = facts[(facts["duration_days"].between(*_FY_DAYS)) & (pd.to_datetime(facts["period_end"]) == fy_end)]
    if fy.empty:
        return None
    fy_total = float(fy["value"].iloc[0])
    ytd = facts[
        (facts["duration_days"].between(*_YTD_DAYS))
        & (pd.to_datetime(facts["period_end"]) >= fy_end - pd.Timedelta(days=_MISSING_QUARTER_GAP_DAYS))
        & (pd.to_datetime(facts["period_end"]) < fy_end)
    ]
    if ytd.empty:
        return None
    ytd_q3 = float(ytd.sort_values("period_end")["value"].iloc[-1])
    q4 = fy_total - ytd_q3
    row = fy.iloc[0].copy()
    row["value"] = q4
    row["fiscal_period"] = "Q4"
    return pd.DataFrame([row])


_DIVIDEND_CONCEPT = "CommonStockDividendsPerShareDeclared"
_DIVIDEND_SOURCE = "SEC EDGAR company facts (Declared dividends per share)"
_DIVIDEND_TTM_MAX_AGE_DAYS = 180


def _is_recent_dividend_period(
    period_end: object, as_of: _dt.date, max_age_days: int = _DIVIDEND_TTM_MAX_AGE_DAYS
) -> bool:
    """True only when period_end is on/before as_of and within max_age_days."""
    try:
        end = _dt.date.fromisoformat(str(period_end)[:10])
    except TypeError, ValueError, AttributeError:
        return False
    delta = (as_of - end).days
    return 0 <= delta <= max_age_days


def _null_dividend_payload(ticker: str) -> dict[str, object]:
    """Coverage uncertainty: concept absence is not proof of a nonpayer."""
    return {
        "ticker": ticker,
        "dividend_status": "insufficient_data",
        "ttm_dividend_per_share": None,
        "ttm_dividend_yield": None,
        "price": None,
        "price_source": None,
        "price_retrieved_at": None,
        "growth_1y": None,
        "growth_3y_cagr": None,
        "growth_5y_cagr": None,
        "growth_10y_cagr": None,
        "annual_history": [],
        "source": _DIVIDEND_SOURCE,
    }


def _has_contiguous_gaps(period_ends: list[object], day_range: tuple[int, int], count: int) -> bool:
    """True only for exactly count parseable ends with gaps inside day_range."""
    if len(period_ends) != count:
        return False
    try:
        ends = sorted(_dt.date.fromisoformat(str(p)[:10]) for p in period_ends)
    except TypeError, ValueError:
        return False
    return all(day_range[0] <= (b - a).days <= day_range[1] for a, b in itertools.pairwise(ends))


def _has_contiguous_quarters(period_ends: list[object]) -> bool:
    """True only for exactly four parseable ends with quarterly gaps."""
    return _has_contiguous_gaps(period_ends, _QUARTER_DAYS, 4)


def _dividend_growth(annual: dict[int, float]) -> dict[str, float | None]:
    """Exact-gap growth/CAGR over annual totals; missing gaps stay null."""
    out: dict[str, float | None] = {
        "growth_1y": None,
        "growth_3y_cagr": None,
        "growth_5y_cagr": None,
        "growth_10y_cagr": None,
    }
    if not annual:
        return out
    latest_year = max(annual)
    latest = annual[latest_year]
    for key, n in (("growth_1y", 1), ("growth_3y_cagr", 3), ("growth_5y_cagr", 5), ("growth_10y_cagr", 10)):
        prev = annual.get(latest_year - n)
        if prev is None or prev <= 0:
            continue
        if n == 1:
            out[key] = round((latest - prev) / prev, 4)
        else:
            out[key] = round((latest / prev) ** (1.0 / n) - 1, 4)
    return out


_NO_HISTORICAL_PRICE_STORE = "no_historical_price_store"


def _dividend_valuation_stub() -> dict[str, object]:
    """Historical-price valuation is unavailable (no OHLCV store): nulls with reason."""
    return {
        "historical_yield": None,
        "historical_yield_reason": _NO_HISTORICAL_PRICE_STORE,
        "yield_percentile": None,
        "yield_percentile_reason": _NO_HISTORICAL_PRICE_STORE,
        "total_return": None,
        "total_return_reason": _NO_HISTORICAL_PRICE_STORE,
        "shareholder_yield": None,
        "shareholder_yield_reason": _NO_HISTORICAL_PRICE_STORE,
    }


def _dividend_ttm_float(ttm: object) -> float | None:
    """Parsed TTM value or None when absent/unparseable."""
    if ttm is None:
        return None
    try:
        if isinstance(ttm, (int, float, str, bytes)):
            return float(ttm)
        return float(str(ttm))
    except TypeError, ValueError:
        return None


def _dividend_live_quote(ticker: str) -> dict[str, object] | None:
    """Live quote dict or None when unavailable/not a dict."""
    try:
        from . import valuation as _valuation

        quote = _valuation.get_live_quote(ticker)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    return dict(quote) if isinstance(quote, dict) else None


def _dividend_quote_price(quote: dict[str, object]) -> object:
    """Quote price or None when the lookup itself fails."""
    try:
        return quote.get("price") if quote.get("price") is not None else None
    except TypeError, ValueError:
        return None


def _dividend_valuation(ticker: str, ttm: object, *, include_price: bool) -> dict[str, object]:
    """Point-in-time valuation: stale/absent TTM or historical requests expose no price."""
    nulls = {
        "ttm_dividend_yield": None,
        "price": None,
        "price_source": None,
        "price_retrieved_at": None,
        **_dividend_valuation_stub(),
    }
    if not include_price or ttm is None:
        return dict(nulls)
    ttm_f = _dividend_ttm_float(ttm)
    if ttm_f is None:
        return dict(nulls)
    quote = _dividend_live_quote(ticker)
    if quote is None:
        return dict(nulls)
    price = _dividend_quote_price(quote)
    if not isinstance(price, (int, float)) or price <= 0:
        return dict(nulls)
    return {
        "ttm_dividend_yield": round(ttm_f / price, 4),
        "price": price,
        "price_source": "yahoo",
        "price_retrieved_at": quote.get("retrieved_at"),
        **_dividend_valuation_stub(),
    }


def _dividend_annual_history(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, float]]:
    """Full-year-duration facts keyed by calendar year of period_end."""
    by_year: dict[int, tuple[str, float]] = {}
    for r in rows:
        raw_value = r.get("value")
        if not isinstance(raw_value, (int, float, str, Decimal)):
            continue
        try:
            end = _dt.date.fromisoformat(str(r.get("period_end"))[:10])
            val = float(raw_value)
        except TypeError, ValueError:
            continue
        prev = by_year.get(end.year)
        if prev is None or str(r.get("period_end")) > prev[0]:
            by_year[end.year] = (str(r.get("period_end")), val)
    history: list[dict[str, object]] = []
    annual: dict[int, float] = {}
    for fy in sorted(by_year, reverse=True):
        total = round(by_year[fy][1], 4)
        annual[fy] = total
        history.append({"fiscal_year": fy, "dividend_per_share": total})
    return history, annual


def get_fundamentals(ticker: str, metric: str, *, include_dividend_price: bool = True) -> dict[str, object]:
    """Return a specific fundamental for ticker.

    metric: 'eps' | 'dividends' | 'balance_sheet' | 'shares_outstanding' | 'overview'

    'shares_float' is accepted as a deprecated alias for
    'shares_outstanding': it returns SEC-reported shares outstanding, not
    public float, and the response says so explicitly.
    """
    _ensure_init()
    if metric == "shares_float":
        metric = "shares_outstanding"
    key = f"fundamentals:{ticker}:{metric}"
    result = _cached_or_fetch(key, lambda: _fetch_fundamentals(ticker, metric))
    if metric == "dividends" and isinstance(result, dict) and "error" not in result:
        result = dict(result)
        latest_end = result.pop("_latest_dividend_period_end", None)
        if result.get("dividend_status") != "insufficient_data":
            if result.get("ttm_dividend_per_share") is None or not _is_recent_dividend_period(
                latest_end,
                _dt.date.today(),  # noqa: DTZ011 - trading-calendar local date has no tz meaning
            ):
                result["ttm_dividend_per_share"] = None
                result["dividend_status"] = "unknown"
            else:
                result["dividend_status"] = "paying"
        result.update(
            _dividend_valuation(ticker, result.get("ttm_dividend_per_share"), include_price=include_dividend_price)
        )
    return result


def _fact_field(row: pd.Series[float], name: str, *alts: str) -> str | None:
    """First present/non-blank meta field on a fact row (None when absent)."""
    for k in (name, *alts):
        try:
            v = row.get(k)
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
        if v is not None and str(v) not in ("", "nan", "NaT"):
            return str(v)
    return None


def _copy_fact_meta(r: pd.Series[float]) -> dict[str, object]:
    """accession/form/filed meta off a fact row (first alias wins)."""
    meta: dict[str, object] = {}
    for _k in ("accession", "accn", "form", "filed", "filed_at", "filing_date"):
        try:
            _v = r.get(_k)
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            _v = None
        if _v is not None and str(_v) not in ("", "nan", "NaT"):
            _out = "accession" if _k in ("accession", "accn") else ("filed" if _k in ("filed", "filed_at", "filing_date") else _k)
            meta.setdefault(_out, str(_v))
    return meta


def _fundamentals_overview(ticker: str, company: Company, _cik: object, facts_url: str | None) -> dict[str, object]:
    """Ticker overview payload (name/CIK/industry plus facts URL)."""
    out: dict[str, object] = {
        "ticker": ticker,
        "name": company.name,
        "cik": company.cik,
        "industry": getattr(company, "sic_description", None),
    }
    if facts_url:
        out["source_url"] = facts_url
    return out


def _fundamentals_shares(ticker: str, company: Company, cik: object, facts_url: str | None) -> dict[str, object]:
    """Latest SEC-reported shares outstanding with filing meta."""
    facts = company.get_facts()
    if facts is None:
        return _no_data(ticker, "company facts not available")
    df = _facts_dataframe(facts)
    shares = df[
        df["concept"].isin(
            [
                "us-gaap:CommonStockSharesOutstanding",
                "CommonStockSharesOutstanding",
                "dei:EntityCommonStockSharesOutstanding",
            ]
        )
    ]
    if shares.empty:
        return _no_data(ticker, "shares outstanding not found in company facts")
    latest = shares.sort_values("period_end").iloc[-1]
    out: dict[str, object] = {
        "ticker": ticker,
        "shares_outstanding": float(latest["value"]),
        "as_of": str(latest["period_end"]),
        "source": "SEC EDGAR company facts",
        "note": "SEC-reported shares outstanding, not public float",
    }
    if cik is not None:
        out["cik"] = cik
    if facts_url:
        out["source_url"] = facts_url
    for k, v in (
        ("accession", _fact_field(latest, "accession", "accn")),
        ("form", _fact_field(latest, "form")),
        ("filed", _fact_field(latest, "filed", "filed_at", "filing_date")),
    ):
        if v:
            out[k] = v
    return out


def _recent_quarterly_facts(df: pd.DataFrame, concept: str) -> pd.DataFrame | None:
    """Quarterly facts for concept (None when absent; empty when no quarterly rows)."""
    subset = df[df["concept"].isin([concept, concept.split(":")[-1]])].copy()
    if subset.empty:
        return None
    subset["duration_days"] = _fact_duration_days(subset)
    q = subset[(subset["duration_days"] >= _QUARTER_DAYS[0]) & (subset["duration_days"] <= _QUARTER_DAYS[1])].copy()
    if q.empty:
        return q
    q = _dedup_latest(q).sort_values("period_end")
    return _quarters_with_derived_q4(q, df, concept)


def _merge_eps_quarters(recent_diluted: pd.DataFrame, recent_basic: pd.DataFrame | None) -> list[dict[str, object]]:
    """Diluted quarters with matching basic EPS and filing meta merged in."""
    quarterly_eps: list[dict[str, object]] = []
    for _, r_diluted in recent_diluted.iterrows():
        q_entry: dict[str, object] = {
            "fiscal_year": str(r_diluted["fiscal_year"]),
            "fiscal_period": str(r_diluted["fiscal_period"]),
            "eps_diluted": round(float(r_diluted["value"]), 2),
            "period_end": str(r_diluted["period_end"]),
        }
        for k, v in _copy_fact_meta(r_diluted).items():
            q_entry.setdefault(k, v)
        if recent_basic is not None:
            matching = recent_basic[recent_basic["period_end"] == r_diluted["period_end"]]
            if not matching.empty:
                q_entry["eps_basic"] = round(float(matching.iloc[0]["value"]), 2)
        quarterly_eps.append(q_entry)
    return quarterly_eps


def _eps_result(
    ticker: str,
    quarterly_eps: list[dict[str, object]],
    recent_diluted: pd.DataFrame,
    recent_basic: pd.DataFrame | None,
    cik: object,
    facts_url: str | None,
) -> dict[str, object]:
    """EPS payload with TTM totals once four quarters are present."""
    result: dict[str, object] = {
        "ticker": ticker,
        "quarterly_eps": quarterly_eps,
        "source": "SEC EDGAR company facts (Basic & Diluted EPS)",
    }
    if cik is not None:
        result["cik"] = cik
    if facts_url:
        result["source_url"] = facts_url
    # SDK get_ttm stays display-only (split-adjusted, not PIT); TTM here sums filed quarterly facts.
    if len(recent_diluted) == 4:
        result["ttm_eps_diluted"] = round(sum(float(r["value"]) for _, r in recent_diluted.iterrows()), 2)
    if recent_basic is not None and len(recent_basic) == 4:
        result["ttm_eps_basic"] = round(sum(float(r["value"]) for _, r in recent_basic.iterrows()), 2)
    return result


def _fundamentals_eps(ticker: str, company: Company, cik: object, facts_url: str | None) -> dict[str, object]:
    """Quarterly diluted (+basic) EPS with TTM totals."""
    facts = company.get_facts()
    if facts is None:
        return _no_data(ticker, "company facts not available")
    df = _facts_dataframe(facts)
    recent_diluted = _recent_quarterly_facts(df, "us-gaap:EarningsPerShareDiluted")
    if recent_diluted is None:
        return _no_data(ticker, "diluted EPS not found in company facts")
    if recent_diluted.empty:
        return _no_data(ticker, "no quarterly diluted EPS facts found")
    recent_basic = _recent_quarterly_facts(df, "us-gaap:EarningsPerShareBasic")
    if recent_basic is not None and recent_basic.empty:
        recent_basic = None
    quarterly_eps = _merge_eps_quarters(recent_diluted, recent_basic)
    return _eps_result(ticker, quarterly_eps, recent_diluted, recent_basic, cik, facts_url)


def _dividends_ttm(recent: pd.DataFrame) -> float | None:
    """TTM dividend total over four contiguous quarters (None while building)."""
    if len(recent) != 4:
        return None
    if not _has_contiguous_quarters(list(recent["period_end"])):
        return None
    return round(sum(float(r["value"]) for _, r in recent.iterrows()), 4)


def _dividends_fy_rows(div: pd.DataFrame) -> list[dict[str, str | float]]:
    """Full-year dividend facts as period_end/value rows."""
    fy = div[(div["duration_days"] >= _FY_DAYS[0]) & (div["duration_days"] <= _FY_DAYS[1])].copy()
    if fy.empty:
        return []
    fy = _dedup_latest(fy)
    return [{"period_end": str(r["period_end"]), "value": float(r["value"])} for _, r in fy.iterrows()]


def _fundamentals_dividends(ticker: str, company: Company) -> dict[str, object]:
    """Quarterly TTM dividends plus full-year annual history."""
    facts = company.get_facts()
    if facts is None:
        return _no_data(ticker, "company facts not available")
    df = _facts_dataframe(facts)
    div = df[df["concept"].isin(["us-gaap:" + _DIVIDEND_CONCEPT, _DIVIDEND_CONCEPT])].copy()
    if div.empty:
        return _null_dividend_payload(ticker)
    div["duration_days"] = _fact_duration_days(div)
    q = div[(div["duration_days"] >= _QUARTER_DAYS[0]) & (div["duration_days"] <= _QUARTER_DAYS[1])].copy()
    if not q.empty:
        q = _dedup_latest(q).sort_values("period_end")
        recent = _quarters_with_derived_q4(q, df, "us-gaap:" + _DIVIDEND_CONCEPT)
    else:
        recent = q
    ttm = _dividends_ttm(recent)
    latest_end: str | None = str(recent.iloc[-1]["period_end"]) if len(recent) else None
    history, annual = _dividend_annual_history(_dividends_fy_rows(div))
    return {
        "ticker": ticker,
        "dividend_status": "paying" if ttm is not None else "unknown",
        "ttm_dividend_per_share": ttm,
        **_dividend_growth(annual),
        "annual_history": history,
        "source": _DIVIDEND_SOURCE,
        "_latest_dividend_period_end": latest_end,
    }


def _balance_sheet_stmt(financials: object) -> object:
    """Balance-sheet statement handle (None when the SDK exposes neither view)."""
    return getattr(financials, "balance_sheet", None) or getattr(financials, "get_balance_sheet", lambda: None)()


def _balance_sheet_text(bs: object) -> dict[str, object]:
    """Latest balance-sheet snapshot as a dict (raw string when unshaped)."""
    try:
        latest = bs.get_latest() if hasattr(bs, "get_latest") else bs
        return latest.to_dict() if hasattr(latest, "to_dict") else {"raw": str(latest)}
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"raw": str(bs)}


def _fundamentals_balance(ticker: str, company: Company, cik: object, facts_url: str | None) -> dict[str, object]:
    """Latest balance-sheet snapshot with provenance."""
    financials = company.get_financials()
    bs = _balance_sheet_stmt(financials)
    if bs is None:
        return _no_data(ticker, "balance sheet not available")
    data = _balance_sheet_text(bs)
    out: dict[str, object] = {"ticker": ticker, "balance_sheet": data, "source": "SEC EDGAR financials"}
    if cik is not None:
        out["cik"] = cik
    if facts_url:
        out["source_url"] = facts_url
    return out


def _fetch_fundamentals(ticker: str, metric: str) -> dict[str, object]:
    try:
        company = Company(ticker)
        _cik = getattr(company, "cik", None)
        _facts_url = _companyfacts_url(_cik)
        if metric == "overview":
            return _fundamentals_overview(ticker, company, _cik, _facts_url)
        if metric == "shares_outstanding":
            return _fundamentals_shares(ticker, company, _cik, _facts_url)
        if metric == "eps":
            return _fundamentals_eps(ticker, company, _cik, _facts_url)
        if metric == "dividends":
            return _fundamentals_dividends(ticker, company)
        if metric == "balance_sheet":
            return _fundamentals_balance(ticker, company, _cik, _facts_url)
        return {"error": f"Unknown metric '{metric}'"}
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("get_fundamentals(%s, %s) failed: %s", ticker, metric, e)
        return _no_data(ticker, f"error retrieving {metric}: {e}")


_OWNERSHIP_FEED_FORMS = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")
_OWNERSHIP_FEED_TTL_SECONDS = 3600  # SEC current-filings feed covers ~24h
_OWNERSHIP_TICKER_TTL_SECONDS = 7 * 86400  # ponytail: cached CIK->ticker map, refreshes weekly


def get_recent_ownership_filings(form_type: str = "both", limit: int = 10) -> dict[str, object]:
    """Most recent SC 13D/G filings market-wide (SEC current-filings feed, ~24h window)."""
    _ensure_init()
    key = f"ownership_feed:{form_type}:{limit}"
    hit = cache.get(key, ttl=_OWNERSHIP_FEED_TTL_SECONDS)
    if isinstance(hit, dict):
        return hit
    value = _fetch_recent_ownership_filings(form_type, limit)
    cache.set(key, value)
    return value


def _resolve_issuer_ticker(cik: int) -> str | None:
    """Best-effort CIK -> ticker for drill-down (cached; None when unresolvable)."""
    key = f"cik_ticker:{cik:010d}"
    hit = cache.get(key, ttl=_OWNERSHIP_TICKER_TTL_SECONDS)
    if isinstance(hit, str):
        return hit or None
    try:
        tickers = Company(cik).tickers
        value = tickers[0] if tickers else ""
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        value = ""
    cache.set(key, value)
    return value or None


def _ownership_filing_doc(filing: Filing) -> tuple[dict[str, object], object | None]:
    """Base feed row plus parsed doc (None doc degrades to filer-only)."""
    row: dict[str, object] = {
        "form": str(getattr(filing, "form", "")),
        "filed": str(getattr(filing, "filing_date", "")),
        "accession_no": getattr(filing, "accession_no", None),
    }
    try:
        return row, filing.obj()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        row["filer"] = str(getattr(filing, "company", ""))
        row["note"] = "filing detail unavailable"
        return row, None


def _ownership_percent(doc: object) -> float | None:
    """Beneficial-ownership percent or None when absent/unparseable."""
    if getattr(doc, "total_percent", None) is None:
        return None
    try:
        return round(float(getattr(doc, "total_percent")), 2)  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    except TypeError, ValueError:
        return None


def _ownership_shares(doc: object) -> int | None:
    """Beneficial-ownership share count or None when absent/unparseable."""
    if getattr(doc, "total_shares", None) is None:
        return None
    try:
        return int(getattr(doc, "total_shares"))  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    except TypeError, ValueError:
        return None


def _ownership_doc_detail(row: dict[str, object], doc: object) -> None:
    """Issuer/filer/percent/shares detail onto the feed row (never raises)."""
    try:
        persons = getattr(doc, "reporting_persons", None) or []
        row["filers"] = [p.name for p in persons[:5]]
        issuer = getattr(doc, "issuer_info", None)
        if issuer is not None:
            row["issuer"] = getattr(issuer, "name", None)
            row["issuer_cik"] = getattr(issuer, "cik", None)
        percent = _ownership_percent(doc)
        if percent is not None:
            row["percent"] = percent
        shares = _ownership_shares(doc)
        if shares is not None:
            row["shares"] = shares
        row["event_date"] = str(getattr(doc, "date_of_event", "") or "") or None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        row["note"] = "ownership detail unavailable (pre-XML filing)"


def _ownership_forms(form_type: object) -> tuple[str, list[str] | None]:
    """Normalized label plus feed forms (None forms = invalid form_type)."""
    label = (form_type or "both").strip().upper() if isinstance(form_type, str) else "BOTH"
    if label in ("BOTH", "13D/G", "13DG"):
        return label, list(_OWNERSHIP_FEED_FORMS)
    if label in ("SC 13D", "13D"):
        return label, ["SC 13D", "SC 13D/A"]
    if label in ("SC 13G", "13G"):
        return label, ["SC 13G", "SC 13G/A"]
    return label, None


def _ownership_limit(limit: object) -> int | None:
    """Clamped feed limit 1-25 (None when unparseable)."""
    candidate: object = 10 if limit is None else limit
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, float):
        if not candidate.is_integer():
            return None
        candidate = int(candidate)
    if isinstance(candidate, str):
        text = candidate.strip()
        try:
            candidate = int(float(text)) if "." in text else int(text)
        except ValueError:
            return None
    if not isinstance(candidate, int):
        return None
    return max(1, min(candidate, 25))


def _ownership_form_rows(forms: list[str]) -> list[dict[str, object]]:
    """Merged, deduped feed rows across form variants (one failure never sinks)."""
    from edgar import get_current_filings

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for form in forms:
        try:
            # ponytail: 10 filings per variant (40 merged max); raise page_size if daily 13D/G volume exceeds it
            feed = get_current_filings(form=form, page_size=10)
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue  # one variant failing must not sink the feed
        for filing in feed:
            accession = str(getattr(filing, "accession_no", ""))
            if accession in seen:
                continue
            seen.add(accession)
            rows.append(_ownership_feed_row(filing))
    return rows


def _attach_ownership_tickers(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """Newest-first rows trimmed to limit with best-effort issuer tickers."""
    rows.sort(key=_filed_key, reverse=True)
    trimmed = rows[:limit]
    for row in trimmed:
        issuer_cik = row.get("issuer_cik")
        if issuer_cik:
            try:
                row["ticker"] = _resolve_issuer_ticker(int(str(issuer_cik)))
            except TypeError, ValueError:
                row["ticker"] = None
    return trimmed


def _ownership_feed_row(filing: Filing) -> dict[str, object]:
    """One feed row with issuer/filer detail; never raises (detail degrades to filer-only)."""
    row, doc = _ownership_filing_doc(filing)
    if doc is None:
        return row
    _ownership_doc_detail(row, doc)
    if not row.get("filers"):
        row["filers"] = [str(getattr(filing, "company", ""))]
    return row


def _filed_key(row: dict[str, object]) -> str:
    """Sort key for ownership-feed rows (filed ISO date; missing sorts first)."""
    return str(row.get("filed", ""))


def _fetch_recent_ownership_filings(form_type: str, limit: object) -> dict[str, object]:
    label, forms = _ownership_forms(form_type)
    if forms is None:
        return {"error": f"Invalid form_type '{form_type}': use 'SC 13D', 'SC 13G', or 'both'"}
    clamped = _ownership_limit(limit)
    if clamped is None:
        return {"error": f"Invalid limit '{limit}': use 1-25"}
    try:
        rows = _attach_ownership_tickers(_ownership_form_rows(forms), clamped)
        return {
            "form_type": "both" if label in ("BOTH", "13D/G", "13DG") else label,
            "window": "SEC current-filings feed (~24h)",
            "count": len(rows),
            "filings": rows,
            "source": "SEC EDGAR current filings (SC 13D/G)",
        }
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("get_recent_ownership_filings(%s) failed: %s", form_type, e)
        return {"error": f"No data found: error retrieving recent {label} filings: {e}"}


def get_latest_earnings_release(ticker: str) -> dict[str, object]:
    """Return the text of the latest 8-K Item 2.02 press release."""
    _ensure_init()
    key = f"earnings_release:{ticker}"
    return _cached_or_fetch(key, lambda: _fetch_latest_earnings_release(ticker))


def _filing_202_text(eightk: object) -> str | None:
    """Press-release text when the 8-K carries Item 2.02 (None otherwise)."""
    items = getattr(eightk, "items", []) or []
    if not any("2.02" in item for item in items):
        return None
    press_releases = getattr(eightk, "press_releases", None) or []
    if not press_releases:
        return None
    text = press_releases[0].text()
    return str(text) if text is not None else None


def _earnings_out(ticker: str, filing: Filing, text: object, kind: str, cik: object) -> dict[str, object]:
    """Earnings payload for an 8-K release or a 10-Q MD&A fallback."""
    filed = filing.filing_date
    accession = getattr(filing, "accession_no", None)
    out: dict[str, object] = {
        "ticker": ticker,
        "filed": str(filed),  # SDK annotates str but runtime yields a date (Arrow date32 .as_py())
        "accession_no": accession,
        "accession": accession,
        "text": text,
        "source": f"{kind} filed {filed} (accession {accession})",
    }
    _url = getattr(filing, "filing_url", None) or getattr(filing, "url", None) or _filing_dir_url(cik, accession)
    if _url:
        out["source_url"] = str(_url)
    if cik is not None:
        out["cik"] = cik
    return out


def _tenq_mda_text(company: Company) -> tuple[Filing | None, str | None]:
    """Latest 10-Q MD&A text for the earnings fallback (None pair when absent)."""
    tenq_filings = company.get_filings(form=["10-Q"])
    if not tenq_filings:
        return None, None
    filing = tenq_filings[0]
    assert isinstance(filing, Filing)
    tenq = filing.obj()
    mda = getattr(tenq, "management_discussion", None)
    if mda is None:
        return None, None
    return filing, mda if isinstance(mda, str) else getattr(mda, "text", lambda: str(mda))()


def _fetch_latest_earnings_release(ticker: str) -> dict[str, object]:
    try:
        company = Company(ticker)
        _cik = getattr(company, "cik", None)
        filings = company.get_filings(form=["8-K"])
        for filing in filings:
            eightk = filing.obj()
            logger.debug("8-K %s items: %s", filing.accession_no, getattr(eightk, "items", []))
            text = _filing_202_text(eightk)
            if text is None:
                logger.debug("8-K %s has no usable Item 2.02 press release", filing.accession_no)
                continue
            return _earnings_out(ticker, filing, text, "8-K Item 2.02", _cik)
        # Fallback: try latest 10-Q MD&A as earnings narrative source
        logger.debug("No 8-K Item 2.02 found for %s; falling back to 10-Q MD&A", ticker)
        filing, text = _tenq_mda_text(company)
        if filing is not None and text is not None:
            return _earnings_out(ticker, filing, text, "10-Q MD&A", _cik)
        return _no_data(ticker, "no 8-K Item 2.02 or 10-Q filing found")
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("get_latest_earnings_release(%s) failed: %s", ticker, e)
        return _no_data(ticker, f"error retrieving earnings release: {e}")


def diff_risk_factors(ticker: str) -> dict[str, object]:
    """Unified diff of risk factors between the last two 10-Qs."""
    _ensure_init()
    key = f"risk_diff:{ticker}"
    return _cached_or_fetch(key, lambda: _fetch_diff_risk_factors(ticker))


def _risk_text(filing: Filing) -> str | None:
    try:
        rf = getattr(filing.obj(), "risk_factors", None)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if rf is None:
        return None
    text = rf if isinstance(rf, str) else getattr(rf, "text", lambda: str(rf))()
    return text if text and str(text).strip() else None


def _risk_pair_texts(company: Company) -> list[tuple[Filing, str]]:
    """First two filings (10-Q then 10-K) carrying risk-factor text."""
    with_text: list[tuple[Filing, str]] = []
    for form in (["10-Q"], ["10-K"]):
        for f in itertools.islice(company.get_filings(form=form), 8):
            text = _risk_text(f)
            if text is not None:
                with_text.append((f, text))
            if len(with_text) == 2:
                break
        if len(with_text) == 2:
            break
    return with_text


def _risk_diff_text(latest: Filing, latest_text: str, prior: Filing, prior_text: str) -> str:
    """Unified diff of prior vs latest risk-factor text (sentinel when unchanged)."""
    diff = "\n".join(
        difflib.unified_diff(
            prior_text.splitlines(),
            latest_text.splitlines(),
            fromfile=f"{prior.form} filed {prior.filing_date}",
            tofile=f"{latest.form} filed {latest.filing_date}",
            lineterm="",
        )
    )
    return diff if diff.strip() else "No changes in risk factors language between the two filings."


def _risk_diff_payload(
    ticker: str, latest: Filing, latest_text: str, prior: Filing, prior_text: str, cik: object
) -> dict[str, object]:
    """Risk-diff payload with filing provenance and source URLs."""
    out: dict[str, object] = {
        "ticker": ticker,
        "latest_filed": str(latest.filing_date),  # SDK annotates str but runtime yields a date (Arrow date32 .as_py())
        "prior_filed": str(prior.filing_date),  # SDK annotates str but runtime yields a date (Arrow date32 .as_py())
        "latest_accession": getattr(latest, "accession_no", None),
        "prior_accession": getattr(prior, "accession_no", None),
        "accessions": [getattr(prior, "accession_no", None), getattr(latest, "accession_no", None)],
        "diff": _risk_diff_text(latest, latest_text, prior, prior_text),
        "source": f"{prior.form}s filed {prior.filing_date} and {latest.form}s filed {latest.filing_date}",
    }
    _urls = []
    for _f in (prior, latest):
        _u = (
            getattr(_f, "filing_url", None)
            or getattr(_f, "url", None)
            or _filing_dir_url(cik, getattr(_f, "accession_no", None))
        )
        if _u:
            _urls.append(str(_u))
    if _urls:
        out["source_urls"] = _urls
        out["source_url"] = _urls[-1]
    if cik is not None:
        out["cik"] = cik
    return out


def _fetch_diff_risk_factors(ticker: str) -> dict[str, object]:
    try:
        company = Company(ticker)
        _cik = getattr(company, "cik", None)
        with_text = _risk_pair_texts(company)
        if len(with_text) < 2:
            return _no_data(ticker, "fewer than two filings with risk factors found")
        (latest, latest_text), (prior, prior_text) = with_text[0], with_text[1]
        return _risk_diff_payload(ticker, latest, latest_text, prior, prior_text, _cik)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("diff_risk_factors(%s) failed: %s", ticker, e)
        return _no_data(ticker, f"error diffing risk factors: {e}")


def get_financial_statements(ticker: str, statement_type: str) -> dict[str, object]:
    """Return parsed financial statement (income, balance sheet, or cash flow)."""
    _ensure_init()
    key = f"financial_statements:{ticker}:{statement_type}"
    return _cached_or_fetch(key, lambda: _fetch_financial_statements(ticker, statement_type))


def _select_statement(financials: object, statement_type: str) -> tuple[object | None, dict[str, object] | None]:
    """Statement handle for a known type (error dict for unknown types)."""
    if statement_type == "income_statement":
        return getattr(financials, "income_statement", getattr(financials, "income", None)), None
    if statement_type == "balance_sheet":
        return getattr(financials, "balance_sheet", getattr(financials, "balance", None)), None
    if statement_type == "cash_flow":
        return getattr(
            financials,
            "cash_flow_statement",
            getattr(financials, "cashflow_statement", getattr(financials, "cash_flow", None)),
        ), None
    return None, {"error": f"Unknown statement type '{statement_type}'"}


def _statement_text(stmt: object) -> str:
    """Statement rendered as readable text (raw string when converters fail)."""
    try:
        if hasattr(stmt, "to_dataframe"):
            return str(stmt.to_dataframe())
        if hasattr(stmt, "to_string"):
            return str(stmt.to_string())
        return str(stmt)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return str(stmt)


def _fetch_financial_statements(ticker: str, statement_type: str) -> dict[str, object]:
    try:
        company = Company(ticker)
        _cik = getattr(company, "cik", None)
        _facts_url = _companyfacts_url(_cik)
        stmt, err = _select_statement(company.get_financials(), statement_type)
        if err is not None:
            return err
        if stmt is None:
            return _no_data(ticker, f"{statement_type} not available")
        out: dict[str, object] = {
            "ticker": ticker,
            "statement_type": statement_type,
            "text": _statement_text(stmt),
            "source": f"SEC EDGAR {statement_type}",
        }
        if _cik is not None:
            out["cik"] = _cik
        if _facts_url:
            out["source_url"] = _facts_url
        return out
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("get_financial_statements(%s, %s) failed: %s", ticker, statement_type, e)
        return _no_data(ticker, f"error retrieving {statement_type}: {e}")


def get_xbrl_facts(ticker: str, concept: str) -> dict[str, object]:
    """Return XBRL financial facts for any metric (Revenue, NetIncome, etc.)."""
    _ensure_init()
    key = f"xbrl_facts:{ticker}:{concept}"
    return _cached_or_fetch(key, lambda: _fetch_xbrl_facts(ticker, concept))


def _xbrl_push_word(words: list[str], current: str) -> str:
    """Append a finished word and reset the accumulator."""
    if current:
        words.append(current)
    return ""


def _xbrl_tokens(concept: str) -> list[str]:
    """Significant query tokens split on case/space boundaries (len>2)."""
    words: list[str] = []
    current = ""
    for ch in concept:
        if ch.isalnum():
            if ch.isupper() and current and not current[-1].isupper():
                current = _xbrl_push_word(words, current)
            current += ch
        else:
            current = _xbrl_push_word(words, current)
    _xbrl_push_word(words, current)
    return [w.lower() for w in words if len(w) > 2]


def _xbrl_token_match(df: pd.DataFrame, concept: str):
    """Rows whose concept holds every significant query token."""
    tokens = _xbrl_tokens(concept)
    if not tokens:
        return df.iloc[0:0]
    lowered = df["concept"].str.lower()
    mask = lowered.str.contains(tokens[0], na=False, regex=False)
    for token in tokens[1:]:
        mask = mask & lowered.str.contains(token, na=False, regex=False)
    return df[mask]


def _xbrl_candidates(df: pd.DataFrame, concept: str):
    """Concept matches: substring, then spaceless, then all-tokens fallback."""
    concept_lower = concept.lower()
    matching = df[df["concept"].str.lower().str.contains(concept_lower, na=False)]
    if not matching.empty:
        return matching
    # Spaced guesses ("Net Income") never match spaceless GAAP names;
    # retry against space-stripped values before reporting no data.
    compact = concept_lower.replace(" ", "")
    if compact != concept_lower:
        stripped = df["concept"].str.lower().str.replace(" ", "", regex=False)
        matching = df[stripped.str.contains(compact, na=False, regex=False)]
        if not matching.empty:
            return matching
    # Last resort: all significant query tokens inside one concept
    # ("Total Revenues" -> Revenues, "TotalRevenue" -> Revenues).
    # Only runs when nothing matched.
    return _xbrl_token_match(df, concept)


def _xbrl_result_rows(matching: pd.DataFrame) -> list[dict[str, object]]:
    """Most recent five matches as concept/value/period rows with filing meta."""
    recent = matching.sort_values("period_end").tail(5)
    result_list: list[dict[str, object]] = []
    for _, r in recent.iterrows():
        _row: dict[str, object] = {
            "concept": str(r["concept"]),
            "value": float(r["value"]),
            "period_end": str(r["period_end"]),
            "fiscal_period": str(r.get("fiscal_period", "N/A")),
        }
        for k, v in _copy_fact_meta(r).items():
            _row.setdefault(k, v)
        result_list.append(_row)
    return result_list


def _fetch_xbrl_facts(ticker: str, concept: str) -> dict[str, object]:
    try:
        company = Company(ticker)
        facts = company.get_facts()
        if facts is None:
            return _no_data(ticker, "company facts not available")
        df = _facts_dataframe(facts)
        matching = _xbrl_candidates(df, concept)
        if matching.empty:
            return _no_data(ticker, f"no XBRL facts found for concept '{concept}'")
        if matching["concept"].nunique() > 1:
            return _no_data(
                ticker, f"ambiguous XBRL concept for '{concept}': {sorted(matching['concept'].unique().tolist())[:8]}"
            )
        result_list = _xbrl_result_rows(matching)
        _cik = getattr(company, "cik", None)
        _facts_url = _companyfacts_url(_cik)
        out: dict[str, object] = {
            "ticker": ticker,
            "concept_searched": concept,
            "matching_concepts": result_list,
            "count": len(result_list),
            "source": "SEC EDGAR XBRL facts",
        }
        if _cik is not None:
            out["cik"] = _cik
        if _facts_url:
            out["source_url"] = _facts_url
        return out
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("get_xbrl_facts(%s, %s) failed: %s", ticker, concept, e)
        return _no_data(ticker, f"error retrieving facts for '{concept}': {e}")
