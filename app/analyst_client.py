"""Analyst-consensus and index-weight data sources.

Two keyless feeds:

* Yahoo Finance's unofficial ``quoteSummary`` endpoint (crumb + cookie): sell-side consensus
  estimates (EPS/revenue by period), analyst price targets, recommendation
  ratings, and estimate-revision trends.
* Slickcharts S&P 500 constituents table: per-company index weight.

Both are cached in cache.db; the crumb is held in-process with a TTL and
refreshed automatically. Yahoo failures return a structured unavailable-data
response and never affect application startup. tools.py never talks to these
HTTP endpoints directly.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import requests

from . import cache
from .config import (
    SLICKCHARTS_SP500_URL,
    YAHOO_CRUMB_URL,
    YAHOO_QUERY_BASE,
)

logger = logging.getLogger(__name__)

ESTIMATES_CACHE_TTL_SECONDS = 3600
WEIGHT_CACHE_TTL_SECONDS = 86400
CRUMB_TTL_SECONDS = 900
REQUEST_TIMEOUT_SECONDS = 20

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"

_lock = threading.RLock()
_session: requests.Session | None = None
_crumb: str | None = None
_crumb_at = 0.0

_PERIOD_LABELS = {
    "0q": "current_quarter",
    "+1q": "next_quarter",
    "0y": "current_fiscal_year",
    "+1y": "next_fiscal_year",
}

_RECOMMENDATION_LABELS = {
    "strong_buy": "Strong Buy",
    "buy": "Buy",
    "hold": "Hold",
    "underperform": "Underperform",
    "sell": "Sell",
}


def _no_data(ticker: str, what: str) -> dict[str, object]:
    return {"error": f"No data found for {ticker}: {what}"}


def _session_get(session: requests.Session, url: str, headers: dict[str, str] | None = None) -> requests.Response:
    return session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)


def _ensure_session() -> requests.Session:
    """Session with the cookies Yahoo requires; thread-safe, lazily created."""
    global _session
    with _lock:
        if _session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": _UA})
            _session_get(session, YAHOO_CRUMB_URL)
            _session = session
        return _session


def _get_crumb() -> str:
    global _crumb, _crumb_at
    with _lock:
        if _crumb is None or (time.time() - _crumb_at) > CRUMB_TTL_SECONDS:
            _crumb = _session_get(_ensure_session(), f"{YAHOO_QUERY_BASE}/v1/test/getcrumb").text.strip()
            _crumb_at = time.time()
        return _crumb


def _reset_crumb() -> None:
    global _crumb, _crumb_at
    _crumb = None
    _crumb_at = 0.0


def _fetch_summary_response(ticker: str, modules: str) -> requests.Response:
    """GET quoteSummary; retry once with a fresh crumb on 401/403."""
    url = f"{YAHOO_QUERY_BASE}/v10/finance/quoteSummary/{ticker}?modules={modules}&crumb={_get_crumb()}"
    resp = _session_get(_ensure_session(), url, headers={"Accept": "application/json"})
    if resp.status_code in (401, 403):
        _reset_crumb()
        url = f"{YAHOO_QUERY_BASE}/v10/finance/quoteSummary/{ticker}?modules={modules}&crumb={_get_crumb()}"
        resp = _session_get(_ensure_session(), url, headers={"Accept": "application/json"})
    return resp


def _summary_result_list(summary: dict[str, object], ticker: str) -> list[object]:
    """Narrow quoteSummary to its non-empty result list (same ValueErrors)."""
    result_obj: object = summary.get("result") or []
    if not isinstance(result_obj, list) or not result_obj:
        err: object = summary.get("error") or {}
        detail: object = err.get("description") if isinstance(err, dict) else None
        raise ValueError(detail if isinstance(detail, str) else f"Yahoo returned no data for {ticker}")
    return result_obj


def _first_summary_result(decoded: object, ticker: str) -> dict[str, object]:
    """Narrow decoded payload to quoteSummary result[0] (same ValueErrors)."""
    if not isinstance(decoded, dict):
        raise ValueError(f"Yahoo returned no data for {ticker}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    summary: object = decoded.get("quoteSummary") or {}
    if not isinstance(summary, dict):
        raise ValueError(f"Yahoo returned no data for {ticker}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    first: object = _summary_result_list(summary, ticker)[0]
    if not isinstance(first, dict):
        raise ValueError(f"Yahoo returned no data for {ticker}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return first


def _quote_summary(ticker: str, modules: str) -> dict[str, object]:
    """Fetch quoteSummary for ticker; retries once after refreshing the crumb."""
    resp = _fetch_summary_response(ticker, modules)
    resp.raise_for_status()
    return _first_summary_result(resp.json(), ticker)


def _raw(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    raw: object = value.get("raw")
    if raw is None:
        return None
    if isinstance(raw, (int, float, str)):
        return float(raw)
    return None


def _int(value: object) -> int | None:
    raw = _raw(value)
    return int(raw) if raw is not None else None


def _obj(value: object) -> dict[str, object]:
    """Narrow an untrusted Yahoo JSON value to a mapping ({} when absent/mistyped)."""
    return value if isinstance(value, dict) else {}


def _trend_rows(data: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_trend = _obj(data.get("earningsTrend")).get("trend")
    if not isinstance(raw_trend, list):
        return rows
    for row in raw_trend:
        if not isinstance(row, dict):
            continue
        period = row.get("period")
        label = _PERIOD_LABELS.get(period, period) if isinstance(period, str) else period
        eps = _obj(row.get("earningsEstimate"))
        rev = _obj(row.get("revenueEstimate"))
        eps_trend = _obj(row.get("epsTrend"))
        rows.append(
            {
                "period": label,
                "period_end_date": row.get("endDate"),
                "eps_avg": _raw(eps.get("avg")),
                "eps_low": _raw(eps.get("low")),
                "eps_high": _raw(eps.get("high")),
                "eps_growth_pct": round((_raw(eps.get("growth")) or 0) * 100, 1),
                "eps_analysts": _int(eps.get("numberOfAnalysts")),
                "eps_year_ago": _raw(eps.get("yearAgoEps")),
                "revenue_avg": _raw(rev.get("avg")),
                "revenue_low": _raw(rev.get("low")),
                "revenue_high": _raw(rev.get("high")),
                "revenue_growth_pct": round((_raw(rev.get("growth")) or 0) * 100, 1),
                "revenue_analysts": _int(rev.get("numberOfAnalysts")),
                "eps_revision": {
                    "current": _raw(eps_trend.get("current")),
                    "days7_ago": _raw(eps_trend.get("7daysAgo")),
                    "days30_ago": _raw(eps_trend.get("30daysAgo")),
                    "days60_ago": _raw(eps_trend.get("60daysAgo")),
                },
            }
        )
    return rows


def _label_from_mean(mean: float) -> str:
    if mean <= 1.5:
        return "Strong Buy"
    if mean <= 2.5:
        return "Buy"
    if mean <= 3.5:
        return "Hold"
    if mean <= 4.5:
        return "Underperform"
    return "Sell"


def _recommendation_label(key: str | None, mean: float | None) -> str:
    if key and key in _RECOMMENDATION_LABELS:
        return _RECOMMENDATION_LABELS[key]
    if mean is not None:
        return _label_from_mean(mean)
    return "unknown"


def get_analyst_estimates(ticker: str) -> dict[str, object]:
    """Return normalized analyst consensus estimates for ticker (cached 1h)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return _no_data("", "empty ticker")
    key = f"analyst_estimates:{ticker}"
    hit = cache.get(key, ttl=ESTIMATES_CACHE_TTL_SECONDS)
    if isinstance(hit, dict):
        return hit
    try:
        data = _quote_summary(ticker, "financialData,earningsTrend,defaultKeyStatistics")
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("analyst estimates failed for %s: %s", ticker, e)
        return {"error": f"Analyst estimates unavailable for {ticker}: {e}"}

    fd = _obj(data.get("financialData"))
    ks = _obj(data.get("defaultKeyStatistics"))
    price = _raw(fd.get("currentPrice"))
    shares = _int(ks.get("sharesOutstanding"))
    float_shares = _int(ks.get("floatShares"))
    market_cap = _raw(ks.get("marketCap"))
    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares
    recommendation_key = fd.get("recommendationKey")
    recommendation_mean = _raw(fd.get("recommendationMean"))
    estimates = _trend_rows(data)

    value: dict[str, object] = {
        "ticker": ticker,
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Yahoo Finance sell-side consensus (unofficial endpoint)",
        "quote": {"price": price, "currency": "USD"},
        "shares_outstanding": shares,
        "float_shares": float_shares,
        "market_cap": round(market_cap) if market_cap is not None else None,
        "price_targets": {
            "mean": _raw(fd.get("targetMeanPrice")),
            "median": _raw(fd.get("targetMedianPrice")),
            "high": _raw(fd.get("targetHighPrice")),
            "low": _raw(fd.get("targetLowPrice")),
            "num_analysts": _int(fd.get("numberOfAnalystOpinions")),
            "recommendation_mean": recommendation_mean,
            "recommendation": _recommendation_label(
                recommendation_key if isinstance(recommendation_key, str) else None, recommendation_mean
            ),
        },
        "valuation": {
            "trailing_pe": _raw(ks.get("trailingPE")),
            "forward_pe": _raw(ks.get("forwardPE")),
        },
        "forward_estimates": estimates,
    }
    cache.set(key, value)
    return value


def _fetch_quote_price(ticker: str) -> float | None:
    data = _quote_summary(ticker, "price")
    price = _raw(_obj(data.get("price")).get("regularMarketPrice"))
    if price is None:
        data = _quote_summary(ticker, "financialData")
        price = _raw(_obj(data.get("financialData")).get("currentPrice"))
    return price


def get_quote_price(ticker: str) -> dict[str, object]:
    """Latest Yahoo price with retrieval instant (uncached; 5-min TTL lives in valuation)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"price": None, "retrieved_at": None}
    try:
        price = _fetch_quote_price(ticker)
        if price is None:
            raise ValueError(f"Yahoo returned no price for {ticker}")
        return {"price": price, "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("quote price failed for %s: %s", ticker, e)
        return {"price": None, "retrieved_at": None}


def get_sp500_weight(ticker: str) -> dict[str, object]:
    """Return the S&P 500 index weight for ticker (Slickcharts, cached 24h)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return _no_data("", "empty ticker")
    key = f"sp500_weight:{ticker}"
    hit = cache.get(key, ttl=WEIGHT_CACHE_TTL_SECONDS)
    if isinstance(hit, dict):
        return hit
    try:
        resp = _session_get(_ensure_session(), SLICKCHARTS_SP500_URL)
        if resp.status_code == 403:
            return {
                "error": (
                    f"S&P 500 index weights unavailable for {ticker}: Slickcharts rejected the request (HTTP 403)"
                )
            }
        resp.raise_for_status()
        match = _find_weight(resp.text, ticker)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        logger.warning("sp500 weight failed for %s: %s", ticker, e)
        return {"error": f"S&P 500 index weight unavailable for {ticker}: {e}"}
    if match is None:
        return _no_data(ticker, "not found in S&P 500 constituent list")
    value: dict[str, object] = {
        "ticker": ticker,
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Slickcharts S&P 500 constituents",
        "company": match["company"],
        "rank": match["rank"],
        "weight_pct": match["weight_pct"],
        "note": (
            "Index weight is market-cap based (float-adjusted per S&P "
            "methodology); to estimate total S&P 500 market cap, divide "
            "market_cap from get_analyst_estimates by weight_pct/100."
        ),
    }
    cache.set(key, value)
    return value


def _find_weight(html: str, ticker: str) -> dict[str, object] | None:
    body = re.search(r"<tbody>(.*?)</tbody>", html, re.DOTALL)
    if body is None:
        return None
    for row in re.findall(r"<tr>(.*?)</tr>", body.group(1), re.DOTALL):
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)]
        if len(cells) < 5:
            continue
        symbol = cells[2].split(".")[0].upper()
        if symbol != ticker:
            continue
        weight = cells[3].replace("%", "").strip()
        return {
            "rank": int(cells[0]) if cells[0].isdigit() else None,
            "company": cells[1],
            "weight_pct": float(weight) if _is_float(weight) else None,
        }
    return None


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
