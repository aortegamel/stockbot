"""Valuation metrics computed from live price, SEC EPS, consensus estimates,
and filing-disclosed obligations.

The user-visible numbers are price-anchored to the latest quote at query
time (short TTL), never a stale cached price. Three EPS figures are
reported, never conflated:

* ``consensus_forward_eps`` — Yahoo sell-side consensus (FY current/next).
* ``adjusted_forward_eps`` — consensus minus contractual (non-cancelable,
  firm) obligations annualized per the filing's own schedule. This is the
  "adjusted" number: only absolute, contractually obligated items go in.
* ``forward_eps_incl_contingent`` — consensus minus ALL disclosed
  obligations (contractual + contingent). A stress scenario, explicitly
  NOT called "adjusted".

Buybacks, dividends, and unquantifiable indemnities are excluded from both
adjusted numbers and reported separately.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import TypedDict

from . import analyst_client, cache, edgar_client, obligations
from .obligations import DEFAULT_TRIGGERED_TYPES
from .services import sec_facts

logger = logging.getLogger(__name__)

PRICE_CACHE_TTL_SECONDS = 300  # 5 minutes: numbers must be as-of-now.
VALUATION_CACHE_TTL_SECONDS = 900


def _finite_float(value: object) -> float | None:
    """Finite float for a real numeric input (bool/NaN/inf/str -> None)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calculation(formula: str, value: object, unit: str, inputs: Mapping[str, object]) -> dict[str, object]:
    """Explicitly derived numeric marker (unverified; ledger admission only)."""
    records: list[object] = [
        item if isinstance(item, Mapping) else {"id": name, "value": item} for name, item in inputs.items()
    ]
    return {
        "derived": True,
        "formulaId": formula,
        "formulaVersion": "1",
        "value": value,
        "unit": unit,
        "derivation_inputs": records,
        "verification": "unverified",
    }


def _no_data(ticker: str, what: str) -> dict[str, object]:
    return {"error": f"No valuation data for {ticker}: {what}"}


class LiveQuote(TypedDict):
    """Latest quote: price plus its retrieval instant (both keys always present)."""

    price: float | None
    retrieved_at: str | None


def _cached_quote(hit: object) -> LiveQuote | None:
    """Normalized LiveQuote from a price-cache row (None when no usable row)."""
    if hit is None:
        return None
    if isinstance(hit, dict):
        hit_price = hit.get("price")
        hit_retrieved = hit.get("retrieved_at")
        return {
            "price": _finite_float(hit_price),
            "retrieved_at": hit_retrieved if isinstance(hit_retrieved, str) else None,
        }
    return {"price": _finite_float(hit), "retrieved_at": None}


def _fetched_quote(quote: object) -> tuple[float | None, str | None]:
    """Price + retrieval instant from an analyst quote payload (None-safe)."""
    quote_price = quote.get("price") if isinstance(quote, dict) else None
    price = _finite_float(quote_price)
    quote_retrieved = quote.get("retrieved_at") if isinstance(quote, dict) else None
    return price, quote_retrieved if isinstance(quote_retrieved, str) else None


def get_live_quote(ticker: str) -> LiveQuote:
    """Latest quote with Yahoo retrieval instant (5-minute TTL, independent of the 1-hour analyst-estimates cache)."""
    key = f"live_price:{ticker}"
    cached = _cached_quote(cache.get(key, ttl=PRICE_CACHE_TTL_SECONDS))
    if cached is not None:
        return cached
    price, retrieved_at = _fetched_quote(analyst_client.get_quote_price(ticker))
    if price is None:
        return {"price": None, "retrieved_at": None}
    cache.set(key, {"price": price, "retrieved_at": retrieved_at})
    return {"price": price, "retrieved_at": retrieved_at}


def get_live_price(ticker: str) -> float | None:
    """Latest tradeable quote as of now (5-minute TTL)."""
    return get_live_quote(ticker).get("price")


def _annualized(amount_billions: float, years: float) -> float:
    if years <= 0:
        return 0.0
    return amount_billions / years


def _tier_eps(entry: Mapping[str, object] | None, key: str) -> float | None:
    """float EPS value from a forward-EPS tier entry (missing/non-numeric -> None)."""
    if entry is None:
        return None
    return _finite_float(entry.get(key))


class ObligationAnnualImpact(TypedDict):
    """Per-kind annualized $B impact split by EPS treatment (all values $B)."""

    contractual_annual_billions: float
    contingent_annual_billions: float
    default_triggered_annual_billions: float
    revenue_matched_annual_billions: float
    per_kind: dict[str, dict[str, object]]
    flat_annual_by_bucket: dict[str, float]
    impact_by_fiscal_year: dict[str, dict[str, float]]


def _schedule_entries(value: object) -> list[Mapping[str, object]]:
    """Mapping rows from a schedule-like payload (non-list/non-mapping -> [])."""
    raw: list[object] = value if isinstance(value, list) else []
    return [y for y in raw if isinstance(y, Mapping)]


def _schedule_total(entries: list[Mapping[str, object]]) -> float:
    """Summed amount_billions over schedule entries (non-numeric skipped)."""
    total = 0.0
    for y in entries:
        y_amount = y.get("amount_billions")
        if isinstance(y_amount, (int, float)):
            total += y_amount
    return total


def _entry_amount(entry: Mapping[str, object]) -> float:
    """Numeric amount_billions for one schedule entry (missing -> 0.0)."""
    entry_amount = entry.get("amount_billions")
    return float(entry_amount) if isinstance(entry_amount, (int, float)) else 0.0


def _add_fy_amount(impact_by_fy: dict[str, dict[str, float]], year: str, bucket: str, amount: float) -> None:
    """Add amount to one FY bucket (blank year ignored)."""
    year = (year or "").strip()
    if not year:
        return
    entry = impact_by_fy.setdefault(
        year, {"contractual": 0.0, "contingent": 0.0, "default_triggered": 0.0, "revenue_matched": 0.0}
    )
    entry[bucket] = entry.get(bucket, 0.0) + (amount or 0.0)


def _horizon_of(row: Mapping[str, object]) -> Mapping[str, object]:
    """payment_horizon mapping for a row (missing/non-mapping -> {})."""
    horizon_raw = row.get("payment_horizon")
    return horizon_raw if isinstance(horizon_raw, Mapping) else {}


def _row_annual(
    row: Mapping[str, object],
    amount_b: float,
    kind: str,
    years: int,
    schedule: list[Mapping[str, object]],
    total: float,
    horizon: Mapping[str, object],
) -> float:
    """Annualized $B for one row: disclosed schedule wins, else horizon, else flat."""
    if schedule and total > 0:
        return total / max(1, len(schedule))
    return _horizon_annual(row, amount_b, kind, years, horizon)


def _horizon_annual(
    row: Mapping[str, object], amount_b: float, kind: str, years: int, horizon: Mapping[str, object]
) -> float:
    """Annualized $B from the payment horizon (front-loaded remainder, else flat)."""
    # ponytail: front-loaded remainder (~0.75yr) + tail spread; flat fallback otherwise
    cloud_schedule = _schedule_entries(horizon.get("schedule"))
    if cloud_schedule:
        return _schedule_total(cloud_schedule) / max(1, len(cloud_schedule))
    near_raw = horizon.get("paid_in_remainder_billions")
    near_b = float(near_raw) if isinstance(near_raw, (int, float)) else 0.0
    if near_b:
        return _front_loaded_annual(horizon, near_b, years)
    if row.get("status") == "off_balance_sheet" and "lease" in kind:
        return _annualized(amount_b, 10)
    return _annualized(amount_b, years)


def _front_loaded_annual(horizon: Mapping[str, object], near_b: float, years: int) -> float:
    """Annualized $B for a front-loaded remainder + tail horizon."""
    tail_raw = horizon.get("paid_after_remainder_billions", 0.0)
    tail_b = float(tail_raw) if isinstance(tail_raw, (int, float)) else 0.0
    return near_b / 0.75 + tail_b / max(1, years - 1)


def _row_bucket(row: Mapping[str, object]) -> str | None:
    """EPS-treatment bucket for one row (None = on-balance-sheet, informational only)."""
    if bool(row.get("revenue_matched")):
        return "revenue_matched"
    if row.get("status") == "on_balance_sheet":
        # Already accrued/expensed (leases, debt, deferred revenue):
        # informational only, never a future EPS drag.
        return None
    if row.get("certainty") == "contractual":
        return "contractual"
    if bool(row.get("default_triggered")) or row.get("type") in DEFAULT_TRIGGERED_TYPES:
        return "default_triggered"
    return "contingent"


def _record_schedule_entries(
    impact_by_fy: dict[str, dict[str, float]], bucket: str, entries: list[Mapping[str, object]]
) -> None:
    """Attribute disclosed schedule entries to their FY buckets."""
    for y in entries:
        _add_fy_amount(impact_by_fy, str(y.get("fiscal_year") or ""), bucket, _entry_amount(y))


def _record_row_schedule(
    impact_by_fy: dict[str, dict[str, float]],
    flat_annual_by_bucket: dict[str, float],
    bucket: str,
    annual: float,
    row: Mapping[str, object],
    schedule: list[Mapping[str, object]],
    total: float,
    horizon: Mapping[str, object],
    years: int,
) -> None:
    """Attribute one row's impact to FY buckets (disclosed FYs) or the flat bucket."""
    if schedule and total > 0:
        _record_schedule_entries(impact_by_fy, bucket, schedule)
        return
    if _record_horizon_schedule(impact_by_fy, bucket, horizon):
        return
    if _record_front_loaded(impact_by_fy, bucket, horizon, years):
        return
    flat_annual_by_bucket[bucket] += annual


def _record_horizon_schedule(
    impact_by_fy: dict[str, dict[str, float]], bucket: str, horizon: Mapping[str, object]
) -> bool:
    """Attribute a horizon schedule to FYs (False = no horizon schedule)."""
    cloud_schedule = _schedule_entries(horizon.get("schedule"))
    if not cloud_schedule:
        return False
    _record_schedule_entries(impact_by_fy, bucket, cloud_schedule)
    return True


def _front_loaded_amounts(horizon: Mapping[str, object]) -> tuple[float, str] | None:
    """Remainder amount + FY label for a front-loaded horizon (None when absent)."""
    near_raw = horizon.get("paid_in_remainder_billions")
    near_flat = float(near_raw) if isinstance(near_raw, (int, float)) else 0.0
    remainder_year = str(horizon.get("paid_in_remainder_of_fy") or "").strip()
    return (near_flat, remainder_year) if near_flat and remainder_year else None


def _record_front_loaded(
    impact_by_fy: dict[str, dict[str, float]], bucket: str, horizon: Mapping[str, object], years: int
) -> bool:
    """Attribute a front-loaded remainder+tail horizon to FYs (False = no horizon)."""
    amounts = _front_loaded_amounts(horizon)
    if amounts is None:
        return False
    near_flat, remainder_year = amounts
    _add_fy_amount(impact_by_fy, remainder_year, bucket, near_flat)
    _record_tail_years(impact_by_fy, bucket, horizon, remainder_year, years)
    return True


def _record_tail_years(
    impact_by_fy: dict[str, dict[str, float]],
    bucket: str,
    horizon: Mapping[str, object],
    remainder_year: str,
    years: int,
) -> None:
    """Spread a front-loaded horizon's tail evenly over the following FYs."""
    tail_raw = horizon.get("paid_after_remainder_billions", 0.0)
    tail_flat = float(tail_raw) if isinstance(tail_raw, (int, float)) else 0.0
    tail_per = tail_flat / max(1, years - 1)
    if not tail_per:
        return
    base = _remainder_base_year(remainder_year)
    if base is None:
        return
    for i in range(1, years):
        _add_fy_amount(impact_by_fy, str(base + i), bucket, tail_per)


def _remainder_base_year(remainder_year: str) -> int | None:
    """Leading 4-digit year of a remainder FY label (None when unparseable)."""
    try:
        return int(remainder_year[:4])
    except ValueError:
        return None


def _per_kind_entry(
    row: Mapping[str, object], amount_b: float, annual: float, horizon: Mapping[str, object]
) -> dict[str, object]:
    """per_kind row: rounded totals plus the flags that chose its bucket."""
    return {
        "total_billions": round(amount_b, 3),
        "annualized_billions": round(annual, 3),
        "certainty": row.get("certainty"),
        "status": row.get("status"),
        "revenue_matched": bool(row.get("revenue_matched")),
        "default_triggered": bool(row.get("default_triggered")) or row.get("type") in DEFAULT_TRIGGERED_TYPES,
        "payment_horizon": horizon,
    }


def _obligation_annual_impact(obligations_rows: Sequence[Mapping[str, object]], years: int) -> ObligationAnnualImpact:
    """Per-kind annualized $B impact split by EPS treatment.

    Three buckets, never conflated:

    * ``contractual`` — non-cancelable/firm obligations (pure expense:
      leases). These reduce EPS; they are the only items folded into
      "adjusted forward EPS".
    * ``contingent`` — cancellable/reducible/terminable/default-triggered
      PURE EXPENSE obligations (cloud R&D, vendor, guarantees). Shown in
      the stress scenario, never in "adjusted".
    * ``revenue_matched`` — spend that buys inventory/COGS for products
      sold at gross margin (supply commitments). This is NOT a separate
      EPS drag: consensus revenue and COGS already embed it (the spend
      buys inventory sold at a margin; counting it again would
      double-count the cost). Reported separately with implied revenue
      coverage at the company's own gross margin.

    Annualization honors the filing's disclosed payment horizon when
    present: a front-loaded commitment ($95B paid in the remainder of the
    current fiscal year, tail over ~4 years) is annualized accordingly
    rather than spread flat.
    """
    per_kind: dict[str, dict[str, object]] = {}
    impact_by_fy: dict[str, dict[str, float]] = {}
    flat_annual_by_bucket: dict[str, float] = {
        "contractual": 0.0,
        "contingent": 0.0,
        "default_triggered": 0.0,
        "revenue_matched": 0.0,
    }

    totals = {"contractual": 0.0, "contingent": 0.0, "default_triggered": 0.0, "revenue_matched": 0.0}

    for row in obligations_rows:
        if row.get("schedule_component"):
            continue
        amount_raw = row.get("amount_billions")
        if not isinstance(amount_raw, (int, float)) or not amount_raw:
            continue
        amount_b = float(amount_raw)
        kind = str(row.get("type") or "other")
        schedule = _schedule_entries(row.get("schedule"))
        total = _schedule_total(schedule)
        horizon = _horizon_of(row)
        annual = _row_annual(row, amount_b, kind, years, schedule, total, horizon)
        bucket = _row_bucket(row)
        per_kind[kind] = _per_kind_entry(row, amount_b, annual, horizon)
        if bucket is None:
            continue
        totals[bucket] += annual
        _record_row_schedule(impact_by_fy, flat_annual_by_bucket, bucket, annual, row, schedule, total, horizon, years)
    return {
        "contractual_annual_billions": round(totals["contractual"], 3),
        "contingent_annual_billions": round(totals["contingent"], 3),
        "default_triggered_annual_billions": round(totals["default_triggered"], 3),
        "revenue_matched_annual_billions": round(totals["revenue_matched"], 3),
        "per_kind": per_kind,
        "flat_annual_by_bucket": {k: round(v, 3) for k, v in flat_annual_by_bucket.items()},
        "impact_by_fiscal_year": {
            year: {k: round(v, 3) for k, v in buckets.items()} for year, buckets in impact_by_fy.items()
        },
    }


# P/E multiples used for scenario share-price projections. 15x is a deep
# value/bear multiple, 25x a stable-growth tech multiple, 30x a premium
# growth multiple, and 35x a momentum/peak multiple.
PROJECTION_MULTIPLES = (15, 20, 25, 30, 35)


def _projected_prices(eps_by_tier: dict[str, float | None], price: float | None) -> dict[str, object]:
    """Share price per scenario EPS under a set of assumed P/E multiples.

    projected_price = scenario EPS x assumed P/E multiple. Each cell also
    carries the % change vs the current live price, so a user can see how
    far the stock must fall (or rise) if a scenario's EPS materializes at a
    given multiple.
    """
    tiers: list[dict[str, object]] = []
    for tier, eps in eps_by_tier.items():
        if eps is None:
            continue
        cells: dict[str, dict[str, object]] = {}
        for multiple in PROJECTION_MULTIPLES:
            projected = round(eps * multiple, 2)
            pct = round((projected / price - 1) * 100, 1) if price is not None and price > 0 else None
            cells[f"{multiple}x"] = {
                "price": projected,
                "pct_change_vs_current": pct,
                "derived": True,
                "calculations": {
                    "price": _calculation(
                        "valuation_projected_price", projected, "USD/share", {"eps": eps, "assumed_pe": multiple}
                    ),
                    "pct_change_vs_current": _calculation(
                        "valuation_projected_change", pct, "%", {"current_price": price, "projected_price": projected}
                    ),
                },
            }
        tiers.append({"tier": tier, "eps": round(eps, 2), "prices": cells})
    return {
        "assumption": (
            "projected price = scenario EPS x assumed P/E multiple; pct_change is vs the current live price"
        ),
        "multiples": list(PROJECTION_MULTIPLES),
        "derived": True,
        "current_price": round(price, 2) if price is not None else None,
        "tiers": tiers,
    }


def _tax_rate_from_note_markdown(md: str) -> float | None:
    """Effective rate from one tax-note table (income-tax expense / pre-tax)."""
    match = re.search(
        r"(?:income tax expense|provision for income taxes)"
        r"[^|]{0,80}\|\s*\$?([\d,]+)",
        md,
        re.IGNORECASE,
    )
    pre = re.search(
        r"income before income tax(?:es)?[^|]{0,80}\|\s*\$?([\d,]+)",
        md,
        re.IGNORECASE,
    )
    if not match or not pre:
        return None
    tax = float(match.group(1).replace(",", ""))
    pretax = float(pre.group(1).replace(",", ""))
    return tax / pretax if pretax > 0 else None


def _filing_tax_rate(ticker: str) -> float | None:
    """Effective rate from the latest 10-K tax notes (None on any failure)."""
    try:
        filings = edgar_client.get_latest_report(ticker, "10-K")
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if filings is None:
        return None
    _filing, doc = filings
    notes = getattr(doc, "notes", None)
    if notes is None:
        return None
    try:
        markdowns = [note.to_markdown() for note in notes.search("tax")[:3]]
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    for md in markdowns:
        try:
            rate = _tax_rate_from_note_markdown(md)
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
        if rate is not None:
            return rate
    return None


def _facts_tax_rate(ticker: str) -> float | None:
    """Effective rate from XBRL annual facts (|tax| / |pre-tax|, None on failure)."""
    try:
        facts_obj = edgar_client.get_company(ticker).get_facts()
        if facts_obj is None:
            raise ValueError("company facts unavailable")
        df = facts_obj.to_dataframe()
        tax = df[
            df["concept"].str.contains("IncomeTaxExpenseBenefit", case=False) & (df["fiscal_period"] == "FY")
        ].sort_values("period_end")
        pre = df[
            df["concept"].str.contains("IncomeLossFromContinuingOperationsBeforeIncomeTaxes", case=False)
            & (df["fiscal_period"] == "FY")
        ].sort_values("period_end")
        if tax.empty or pre.empty:
            return None
        tax_v = float(tax.iloc[-1]["value"])
        pre_v = float(pre.iloc[-1]["value"])
        return abs(tax_v / pre_v) if pre_v != 0 else None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _first_sane_rate(candidates: list[float | None]) -> float | None:
    """First candidate inside the 5%-35% band (None when none qualifies)."""
    for rate in candidates:
        if rate is not None and 0.05 <= rate <= 0.35:
            return round(rate, 3)
    return None


def _ob_ticker(ob_rows: Mapping[str, object]) -> str:
    """Ticker string from an obligations envelope (non-string -> '')."""
    ob_ticker = ob_rows.get("ticker", "")
    return ob_ticker if isinstance(ob_ticker, str) else ""


def _effective_tax_rate(ob_rows: Mapping[str, object]) -> float | None:
    """Company's own effective tax rate from its 10-K (income tax expense /
    pre-tax income), falling back to XBRL annual facts, else None. Rates
    outside a sane band (5%-35%) are rejected."""
    ticker = _ob_ticker(ob_rows)
    return _first_sane_rate([_filing_tax_rate(ticker), _facts_tax_rate(ticker)])


def _revenue_matched_margin(ticker: str) -> tuple[float | None, str]:
    """Company's own gross margin from latest FY XBRL facts (GrossProfit /
    Revenue), else None with a reason (never an invented default).

    Mirrors _effective_tax_rate: same company-facts frame, same defensive
    pattern (any failure -> None, never an exception). The source tag
    ("company_facts" vs "unavailable: ...") rides along so coverage can
    caveat it.
    """
    try:
        facts_obj = edgar_client.get_company(ticker).get_facts()
        if facts_obj is None:
            raise ValueError("company facts unavailable")
        df = facts_obj.to_dataframe()
        gp = df[
            df["concept"].str.fullmatch(r"(us-gaap:)?GrossProfit", case=False) & (df["fiscal_period"] == "FY")
        ].sort_values("period_end")
        rev = df.iloc[0:0]
        for concept in (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ):
            hit = df[
                df["concept"].str.fullmatch(rf"(us-gaap:)?{concept}", case=False) & (df["fiscal_period"] == "FY")
            ].sort_values("period_end")
            if not hit.empty:
                rev = hit
                break
        if not gp.empty and not rev.empty:
            gross = float(gp.iloc[-1]["value"])
            revenue = float(rev.iloc[-1]["value"])
            if revenue:
                margin = gross / revenue
                if 0.05 <= margin <= 0.95:
                    return round(margin, 3), "company_facts"
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass
    return None, "unavailable: gross margin fact missing"


def _scenario_hit(
    rows: list[dict[str, object]],
    name: str,
    pretax_billions: float,
    one_time: bool,
    note: str,
    shares_out: int | None,
    tax_rate: float | None,
) -> None:
    """Append one after-tax EPS scenario (missing rate/shares -> None with reason)."""
    if tax_rate is None:
        rows.append(
            {
                "scenario": name,
                "pretax_billions": round(pretax_billions, 3),
                "after_tax_billions": None,
                "eps_impact": None,
                "one_time": one_time,
                "tax_rate": None,
                "reason": "effective tax rate unavailable",
                "note": note,
            }
        )
        return
    after_tax = pretax_billions * (1 - tax_rate)
    if shares_out is None or shares_out <= 0:
        shares_out = None
        rows.append(
            {
                "scenario": name,
                "pretax_billions": round(pretax_billions, 3),
                "after_tax_billions": round(after_tax, 3),
                "eps_impact": None,
                "one_time": one_time,
                "tax_rate": round(tax_rate, 3),
                "reason": "diluted shares unavailable",
                "note": note,
            }
        )
        return
    eps = after_tax / (shares_out / 1e9)
    rows.append(
        {
            "scenario": name,
            "pretax_billions": round(pretax_billions, 3),
            "after_tax_billions": round(after_tax, 3),
            "eps_impact": round(-eps, 3),
            "one_time": one_time,
            "tax_rate": round(tax_rate, 3),
            "reason": None,
            "note": note,
            "derived": True,
            "calculations": {
                "after_tax_billions": _calculation(
                    "valuation_after_tax",
                    round(after_tax, 3),
                    "USD billions",
                    {"pretax_billions": pretax_billions, "effective_tax_rate": tax_rate},
                ),
                "eps_impact": _calculation(
                    "valuation_eps_impact",
                    round(-eps, 3),
                    "USD/share",
                    {"after_tax_billions": after_tax, "shares_billions": shares_out / 1e9},
                ),
            },
        }
    )


def _snapshot_of(ob_rows: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Current obligation snapshot rows (non-list/non-mapping entries dropped)."""
    snapshot_list = ob_rows.get("current_snapshot", ob_rows.get("obligations", []))
    return [s for s in snapshot_list if isinstance(s, Mapping)] if isinstance(snapshot_list, list) else []


def _purchase_exposure(snapshot: list[Mapping[str, object]]) -> float | None:
    """First purchase/supply commitment amount (None when absent/non-numeric)."""
    for row in snapshot:
        if row.get("type") in ("purchase_commitments", "supply_commitments"):
            purchase_raw = row.get("amount_billions")
            purchase = float(purchase_raw) if isinstance(purchase_raw, (int, float)) else None
            return purchase or None
    return None


def _total_future_commitments(snapshot: list[Mapping[str, object]]) -> float:
    """Summed future-cash + off-balance-sheet commitments (excl. supply)."""
    total = 0.0
    for row in snapshot:
        if (
            row.get("status") in ("future_cash_obligation", "off_balance_sheet")
            and row.get("type") != "supply_commitments"
        ):
            commit_raw = row.get("amount_billions")
            if isinstance(commit_raw, (int, float)):
                total += commit_raw
    return total


def _scenario_exposure(row: Mapping[str, object]) -> float:
    """Numeric amount_billions for a scenario row (non-numeric -> 0.0)."""
    trigger_raw = row.get("amount_billions")
    return float(trigger_raw) if isinstance(trigger_raw, (int, float)) else 0.0


def _add_purchase_scenarios(
    rows: list[dict[str, object]], purchase: float | None, shares_out: int | None, tax_rate: float | None
) -> None:
    """5/10/20% write-down scenarios for purchase commitments."""
    if not purchase:
        return
    for pct in (0.05, 0.10, 0.20):
        _scenario_hit(
            rows,
            f"purchase_commitments_write_down_{int(pct * 100)}pct",
            purchase * pct,
            True,
            f"{int(pct * 100)}% of ${purchase}B purchase commitments written down",
            shares_out,
            tax_rate,
        )


def _add_commitment_scenario(
    rows: list[dict[str, object]], total_commitments: float, shares_out: int | None, tax_rate: float | None
) -> None:
    """Annualized all-future-commitments scenario (skipped when zero)."""
    if total_commitments:
        _scenario_hit(
            rows,
            "all_future_commitments_annualized",
            total_commitments / 6.0,
            False,
            "disclosed future-cash and off-balance-sheet commitments spread over 6 years",
            shares_out,
            tax_rate,
        )


def _add_trigger_scenarios(
    rows: list[dict[str, object]], snapshot: list[Mapping[str, object]], shares_out: int | None, tax_rate: float | None
) -> None:
    """Default-triggered guarantee + tax-settlement scenarios."""
    for row in snapshot:
        if row.get("default_triggered"):
            _scenario_hit(
                rows,
                f"{row.get('type')}_call",
                _scenario_exposure(row),
                True,
                "default-triggered guarantee called",
                shares_out,
                tax_rate,
            )
        elif row.get("type") == "unrecognized_tax_benefits":
            _scenario_hit(
                rows,
                "tax_settlement",
                _scenario_exposure(row),
                True,
                "full adverse tax settlement",
                shares_out,
                tax_rate,
            )


def _scenario_assumption(tax_rate: float | None) -> str:
    """Assumption string naming the effective rate (or its absence)."""
    return (
        "after-tax EPS impact = pretax exposure x (1 - effective tax "
        f"rate {round(tax_rate, 3) if tax_rate is not None else 'unavailable; no rate used'})"
        " / diluted shares; one-time items are "
        "shown per event, recurring items annualized"
    )


def _obligation_eps_scenarios(
    ob_rows: Mapping[str, object], shares_out: int | None, tax_rate: float | None
) -> dict[str, object]:
    """Translate disclosed obligations into after-tax EPS-impact scenarios.

    Scenario math (Burry-style): each obligation's exposure is converted to
    an after-tax EPS hit using the company's own effective tax rate and
    diluted shares. One-time items (write-downs, settlements, guarantee
    calls) are shown as one-time hits; recurring items are annualized.
    No invented inputs: an unknown tax rate yields after_tax None, unknown
    shares yield eps_impact None, each with a reason (never 0.15/1-share).
    """
    rows: list[dict[str, object]] = []
    snapshot = _snapshot_of(ob_rows)
    _add_purchase_scenarios(rows, _purchase_exposure(snapshot), shares_out, tax_rate)
    _add_commitment_scenario(rows, _total_future_commitments(snapshot), shares_out, tax_rate)
    _add_trigger_scenarios(rows, snapshot, shares_out, tax_rate)
    return {
        "assumption": _scenario_assumption(tax_rate),
        "effective_tax_rate": round(tax_rate, 3) if tax_rate is not None else None,
        "derived": True,
        "scenarios": rows,
    }


def _implied_coverage(revenue_matched_b: float, gross_margin: float | None) -> float | None:
    """Implied revenue covering revenue-matched spend at own margin (missing -> None)."""
    if revenue_matched_b and gross_margin is not None:
        return revenue_matched_b / (1 - gross_margin)
    return None


def _eps_line(
    eps_value: float | None, contractual: float | None, contingent: float | None, label: str, price: float | None
) -> dict[str, object] | None:
    """One forward-EPS tier with contractual/contingent drags applied (None EPS -> None)."""
    if eps_value is None:
        return None
    line: dict[str, object] = {
        "eps": round(eps_value, 2),
        "price": round(price, 2) if price is not None else None,
        "pe": round(price / eps_value, 1) if (price is not None and eps_value) else None,
        "label": label,
        "derived": True,
        "calculations": {
            "pe": _calculation(
                "valuation_pe",
                round(price / eps_value, 1) if price is not None and eps_value else None,
                "ratio",
                {"price": price, "eps": eps_value},
            ),
        },
    }
    _apply_contractual_drag(line, eps_value, contractual, price)
    _apply_contingent_drag(line, eps_value, contractual, contingent, price)
    calculations = line["calculations"]
    if isinstance(calculations, dict):
        residual = _contingent_residual(eps_value, contractual, contingent) if contingent is not None else None
        for field, formula, inputs, unit in (
            ("eps_after_contractual", "valuation_adjusted_eps", {"eps": eps_value, "drag": contractual}, "USD/share"),
            (
                "eps_after_all_obligations",
                "valuation_scenario_eps",
                {"eps": eps_value, "contractual": contractual, "contingent": contingent},
                "USD/share",
            ),
            (
                "pe_after_contractual",
                "valuation_pe",
                {"price": price, "eps": eps_value - contractual if contractual is not None else None},
                "ratio",
            ),
            (
                "pe_after_all_obligations",
                "valuation_pe",
                {"price": price, "eps": residual},
                "ratio",
            ),
        ):
            if field in line:
                calculations[field] = _calculation(formula, line[field], unit, inputs)
    return line


def _apply_contractual_drag(
    line: dict[str, object], eps_value: float, contractual: float | None, price: float | None
) -> None:
    """Contractual per-share drag onto an EPS line (falsy drag skipped)."""
    if contractual:
        line["obligation_drag_per_share"] = round(contractual, 2)
        line["eps_after_contractual"] = round(eps_value - contractual, 2)
        line["pe_after_contractual"] = (
            round(price / max(0.01, eps_value - contractual), 1)
            if (price is not None and (eps_value - contractual) > 0)
            else None
        )


def _contingent_residual(eps_value: float, contractual: float | None, contingent: float) -> float:
    """EPS left after contractual + contingent drags."""
    return eps_value - (contractual or 0.0) - contingent


def _apply_contingent_drag(
    line: dict[str, object], eps_value: float, contractual: float | None, contingent: float | None, price: float | None
) -> None:
    """Contingent per-share drag onto an EPS line (falsy drag skipped)."""
    if not contingent:
        return
    residual = _contingent_residual(eps_value, contractual, contingent)
    line["contingent_drag_per_share"] = round(contingent, 2)
    line["eps_after_all_obligations"] = round(residual, 2)
    line["pe_after_all_obligations"] = (
        round(price / max(0.01, residual), 1) if (price is not None and residual > 0) else None
    )


def _fy_year(period: dict[str, object]) -> str | None:
    """4-digit FY year from a period's period_end_date (missing -> None)."""
    year = str((period or {}).get("period_end_date") or "")[:4]
    return year or None


def _fy_ps(
    impact_by_fy: dict[str, dict[str, float]],
    flat_map: dict[str, float],
    shares_out: int | None,
    year: str | None,
    key: str,
) -> float | None:
    """Per-share impact for one FY bucket (missing shares/year -> None)."""
    if shares_out is not None and shares_out > 0 and year:
        return (impact_by_fy.get(year, {}).get(key, 0.0) + flat_map.get(key, 0.0)) / (shares_out / 1e9)
    return None


def _fy_drags(
    impact_by_fy: dict[str, dict[str, float]], flat_map: dict[str, float], shares_out: int | None, year: str | None
) -> dict[str, float | None]:
    """All four per-share drags for one FY (contractual/contingent/default/revenue)."""
    return {
        "contractual": _fy_ps(impact_by_fy, flat_map, shares_out, year, "contractual"),
        "contingent": _fy_ps(impact_by_fy, flat_map, shares_out, year, "contingent"),
        "default_triggered": _fy_ps(impact_by_fy, flat_map, shares_out, year, "default_triggered"),
        "revenue_matched": _fy_ps(impact_by_fy, flat_map, shares_out, year, "revenue_matched"),
    }


def _drag_ps(annual_billions: float, shares_out: int | None) -> float | None:
    """Annual $B bucket as per-share drag (missing shares -> None)."""
    return annual_billions / (shares_out / 1e9) if shares_out is not None and shares_out > 0 else None


def _estimates_by_period(estimates: dict[str, object]) -> dict[str, dict[str, object]] | None:
    """Forward estimates keyed by period (None when the list is missing)."""
    forward_rows = estimates.get("forward_estimates")
    if not isinstance(forward_rows, list):
        return None
    return {r["period"]: r for r in forward_rows}


def _current_fy_eps(
    eps_current: float | None, cur: dict[str, float | None], price: float | None
) -> dict[str, dict[str, object] | None]:
    """Current-FY forward-EPS tiers (consensus/adjusted/scenario/worst-case)."""
    return {
        "consensus": _eps_line(eps_current, None, None, "consensus", price),
        "adjusted": _eps_line(
            eps_current,
            cur["contractual"],
            None,
            "adjusted forward EPS (contractual obligations included)",
            price,
        ),
        "scenario": _eps_line(
            eps_current,
            cur["contractual"],
            cur["contingent"],
            "forward EPS incl. contingent obligations (stress scenario, no counterparty default)",
            price,
        ),
        "scenario_with_defaults": _eps_line(
            eps_current,
            cur["contractual"],
            (cur["contingent"] or 0.0) + (cur["default_triggered"] or 0.0),
            "forward EPS incl. contingent obligations AND counterparty-default-triggered guarantees (pay only on counterparty default)",
            price,
        ),
        "worst_case": _eps_line(
            eps_current,
            (cur["contractual"] or 0.0) + (cur["revenue_matched"] or 0.0),
            (cur["contingent"] or 0.0) + (cur["default_triggered"] or 0.0),
            "worst-case EPS (all disclosed obligations incl. revenue-matched supply stranded AND counterparty defaults)",
            price,
        ),
    }


def _next_fy_eps(
    eps_next: float | None, nxt: dict[str, float | None], price: float | None
) -> dict[str, dict[str, object] | None]:
    """Next-FY forward-EPS tiers (consensus/adjusted/scenario/worst-case)."""
    return {
        "consensus_next_fy": _eps_line(eps_next, None, None, "consensus", price),
        "adjusted_next_fy": _eps_line(
            eps_next,
            nxt["contractual"],
            None,
            "adjusted forward EPS (contractual obligations included)",
            price,
        ),
        "scenario_next_fy": _eps_line(
            eps_next,
            nxt["contractual"],
            nxt["contingent"],
            "forward EPS incl. contingent obligations (stress scenario, no counterparty default)",
            price,
        ),
        "worst_case_next_fy": _eps_line(
            eps_next,
            (nxt["contractual"] or 0.0) + (nxt["revenue_matched"] or 0.0),
            (nxt["contingent"] or 0.0) + (nxt["default_triggered"] or 0.0),
            "worst-case EPS (all disclosed obligations incl. revenue-matched supply stranded AND counterparty defaults)",
            price,
        ),
    }


def _build_forward_eps(
    eps_current: float | None,
    eps_next: float | None,
    cur: dict[str, float | None],
    nxt: dict[str, float | None],
    price: float | None,
) -> dict[str, dict[str, object] | None]:
    """Nine-tier forward-EPS map (consensus/adjusted/scenario/worst-case x FY)."""
    return {**_current_fy_eps(eps_current, cur, price), **_next_fy_eps(eps_next, nxt, price)}


def _tier_label(prefix: str, year: str | None, fallback: str) -> str:
    """Tier label with FY year when known (else the generic fallback)."""
    return f"{prefix} FY{year}" if year else fallback


def _scenario_tier_label(kind: str, year: str | None) -> str:
    """Projected-price tier label for a current-FY scenario kind."""
    suffix = "(no default)" if kind == "no_default" else "(counterparty default)"
    return f"Scenario FY{year} {suffix}" if year else f"Scenario (current FY, {suffix[1:]}"


def _next_scenario_tier_label(kind: str, year: str | None) -> str:
    """Projected-price tier label for a next-FY scenario kind."""
    suffix = "(no default)" if kind == "no_default" else "(counterparty default)"
    return f"Scenario FY{year} {suffix}" if year else f"Scenario (next FY, {suffix[1:]}"


def _projected_eps_map(
    forward_eps: dict[str, dict[str, object] | None], year_cur: str | None, year_next: str | None
) -> dict[str, float | None]:
    """Scenario-EPS map consumed by the projected-prices matrix."""
    return {
        _tier_label("Consensus", year_cur, "Consensus (current FY)"): _tier_eps(forward_eps.get("consensus"), "eps"),
        _tier_label("Adjusted", year_cur, "Adjusted (current FY)"): _tier_eps(
            forward_eps.get("adjusted"), "eps_after_contractual"
        ),
        _scenario_tier_label("no_default", year_cur): _tier_eps(
            forward_eps.get("scenario"), "eps_after_all_obligations"
        ),
        _scenario_tier_label("default", year_cur): _tier_eps(
            forward_eps.get("scenario_with_defaults"), "eps_after_all_obligations"
        ),
        _tier_label("Worst case", year_cur, "Worst case (current FY)"): _tier_eps(
            forward_eps.get("worst_case"), "eps_after_all_obligations"
        ),
        _tier_label("Consensus", year_next, "Consensus (next FY)"): _tier_eps(
            forward_eps.get("consensus_next_fy"), "eps"
        ),
        _next_scenario_tier_label("no_default", year_next): _tier_eps(
            forward_eps.get("scenario_next_fy"), "eps_after_all_obligations"
        ),
        _next_scenario_tier_label("default", year_next): _tier_eps(
            forward_eps.get("scenario_with_defaults_next_fy"), "eps_after_all_obligations"
        ),
        _tier_label("Worst case", year_next, "Worst case (next FY)"): _tier_eps(
            forward_eps.get("worst_case_next_fy"), "eps_after_all_obligations"
        ),
    }


def _coverage_rows(ob_rows: Mapping[str, object]) -> list[dict[str, object]]:
    """Quantified obligation rows (non-dict/non-list -> [])."""
    obligations_raw = ob_rows.get("obligations", [])
    return [r for r in obligations_raw if isinstance(r, dict)] if isinstance(obligations_raw, list) else []


def _coverage_manifest(ob_coverage: Mapping[str, object]) -> list[object]:
    """Scan manifest list (non-list -> [])."""
    manifest_raw = ob_coverage.get("scan_manifest")
    return manifest_raw if isinstance(manifest_raw, list) else []


def _coverage_warning_list(ob_rows: Mapping[str, object], ob_coverage: Mapping[str, object]) -> list[object]:
    """Coverage warnings list (missing/non-list -> [])."""
    warnings_raw: object = ob_coverage.get("warnings") or ob_rows.get("warnings") or []
    return list(warnings_raw) if isinstance(warnings_raw, list) else []


def _filings_examined(ob_rows: Mapping[str, object], manifest: list[object], rows: list[dict[str, object]]) -> object:
    """Filings examined: explicit list wins, else manifest/row dates."""
    return (
        ob_rows.get("filings_examined")
        or sorted({str(m.get("filing_date")) for m in manifest if isinstance(m, Mapping) and m.get("filing_date")})
        or sorted({str(r.get("filed")) for r in rows if r.get("filed")})
    )


def _sections_examined(ob_rows: Mapping[str, object], manifest: list[object], rows: list[dict[str, object]]) -> object:
    """Sections examined: explicit list wins, else manifest/row sources."""
    return (
        ob_rows.get("sections_examined")
        or sorted({str(s) for m in manifest if isinstance(m, Mapping) for s in (m.get("sections_examined") or []) if s})
        or sorted({str(r.get("source")) for r in rows if r.get("source")})
    )


def _unquantified_count(ob_rows: Mapping[str, object], ob_coverage: Mapping[str, object]) -> int:
    """Unquantified exposure count (coverage wins, else longest alias list)."""
    unquantified_raw = (
        ob_rows.get("unquantified_exposures") or ob_rows.get("unquantified") or ob_rows.get("unquantified_items")
    )
    fallback = len(unquantified_raw) if isinstance(unquantified_raw, list) else 0
    count = ob_coverage.get("quantified_count", None)
    uncounted = ob_coverage.get("unquantified_count", fallback)
    del count
    return uncounted if isinstance(uncounted, int) else fallback


def _quantified_count(ob_coverage: Mapping[str, object], rows: list[dict[str, object]]) -> int:
    """Quantified exposure count (coverage wins, else row count)."""
    quantified = ob_coverage.get("quantified_count", len(rows))
    return quantified if isinstance(quantified, int) else len(rows)


def _build_coverage(ob_rows: Mapping[str, object], price_gap: str | None) -> dict[str, object]:
    """Coverage block: manifest, examined filings/sections, counts, warnings."""
    rows = _coverage_rows(ob_rows)
    coverage_raw = ob_rows.get("coverage")
    ob_coverage = coverage_raw if isinstance(coverage_raw, dict) else {}
    manifest = _coverage_manifest(ob_coverage)
    coverage_warnings = _coverage_warning_list(ob_rows, ob_coverage)
    coverage: dict[str, object] = {
        "scan_manifest": manifest,
        "filings_examined": _filings_examined(ob_rows, manifest, rows),
        "sections_examined": _sections_examined(ob_rows, manifest, rows),
        "quantified_count": _quantified_count(ob_coverage, rows),
        "unquantified_count": _unquantified_count(ob_rows, ob_coverage),
        "warnings": coverage_warnings,
    }
    if coverage["unquantified_count"] and not rows:
        coverage_warnings.append(
            f"{coverage['unquantified_count']} unquantified exposure(s) disclosed "
            "without dollar amounts; excluded from quantified obligations."
        )
    if price_gap is not None:
        coverage_warnings.append(price_gap)
    return coverage


def _live_price_gap(ticker: str) -> tuple[float | None, str | None]:
    """Live price plus the no-price gap note (None gap when priced)."""
    price = get_live_price(ticker)
    if price is not None:
        return price, None
    return None, (
        f"No live price for {ticker}: price-anchored multiples (trailing "
        "P/E, forward P/E) and projected-price moves vs current are "
        "unavailable; EPS and obligation figures below carry no price, "
        "and no price is estimated."
    )


def _period_eps(period: dict[str, object]) -> float | None:
    """Numeric eps_avg for a forward period (non-numeric -> None)."""
    eps_avg = period.get("eps_avg")
    return eps_avg if isinstance(eps_avg, (int, float)) else None


def _valuation_drags(
    impact: ObligationAnnualImpact, ticker: str, shares_out: int | None
) -> tuple[float | None, float | None, float | None, float | None, float | None, str, float | None]:
    """Per-share drags + revenue coverage for the metrics assembly."""
    contractual_ps = _drag_ps(impact["contractual_annual_billions"], shares_out)
    contingent_ps = _drag_ps(impact["contingent_annual_billions"], shares_out)
    default_triggered_ps = _drag_ps(impact["default_triggered_annual_billions"], shares_out)
    # Revenue-matched supply spend is NOT an EPS drag: it buys inventory
    # sold at gross margin and is already embedded in consensus revenue
    # and COGS. Shown separately with implied revenue coverage.
    revenue_matched_ps = _drag_ps(impact["revenue_matched_annual_billions"], shares_out)
    gross_margin, margin_source = _revenue_matched_margin(ticker)
    implied_coverage_b = _implied_coverage(impact["revenue_matched_annual_billions"], gross_margin)
    return (
        contractual_ps,
        contingent_ps,
        default_triggered_ps,
        revenue_matched_ps,
        gross_margin,
        margin_source,
        implied_coverage_b,
    )


def _valuation_forward_tiers(
    impact: ObligationAnnualImpact,
    shares_out: int | None,
    fy_current: dict[str, object],
    fy_next: dict[str, object],
    eps_current: float | None,
    eps_next: float | None,
    price: float | None,
) -> tuple[dict[str, dict[str, object] | None], str | None, str | None]:
    """Forward-EPS tiers + FY years for the metrics assembly."""
    impact_by_fy = impact.get("impact_by_fiscal_year") or {}
    flat_map = impact.get("flat_annual_by_bucket") or {}
    year_cur = _fy_year(fy_current)
    year_next = _fy_year(fy_next)
    cur = _fy_drags(impact_by_fy, flat_map, shares_out, year_cur)
    nxt = _fy_drags(impact_by_fy, flat_map, shares_out, year_next)
    forward = _build_forward_eps(eps_current, eps_next, cur, nxt, price)
    for key in forward:
        drags = nxt if key.endswith("_next_fy") else cur
        if not key.startswith("consensus") and any(value is None for value in drags.values()):
            forward[key] = None
    return forward, year_cur, year_next


def _cached_metrics(key: str) -> dict[str, object] | None:
    """Cached metrics dict (None unless a dict row is present)."""
    hit = cache.get(key, ttl=VALUATION_CACHE_TTL_SECONDS)
    return hit if isinstance(hit, dict) else None


def _ttm_eps(eps: dict[str, object]) -> float | None:
    """Numeric TTM diluted EPS from an eps envelope (non-numeric -> None)."""
    ttm_raw = eps.get("ttm_eps_diluted")  # envelope keeps payload keys
    return _finite_float(ttm_raw)


def _shares_from_estimates(estimates: dict[str, object]) -> int | None:
    """Shares outstanding from estimates (non-int -> None)."""
    shares_raw = estimates.get("shares_outstanding")
    return shares_raw if isinstance(shares_raw, int) and not isinstance(shares_raw, bool) and shares_raw > 0 else None


def _fetch_obligations(ticker: str) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Obligations envelope or the _no_data payload (never raises)."""
    ob_rows = obligations.get_obligations(ticker)
    if "error" in ob_rows:
        return {"error": f"No valuation data for {ticker}: {ob_rows.get('error') or ''!s}"}, {}
    return None, ob_rows


def _fetch_estimates(ticker: str) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    """Analyst estimates + forward periods, or the _no_data payload."""
    estimates = analyst_client.get_analyst_estimates(ticker)
    if "error" in estimates:
        return {"error": f"No valuation data for {ticker}: {estimates['error']!s}"}, {}
    estimates_by_period = _estimates_by_period(estimates)
    if estimates_by_period is None:
        return {"error": f"No valuation data for {ticker}: analyst estimates missing forward estimates"}, {}
    return None, estimates_by_period


def _fetch_eps(
    ticker: str, estimates: dict[str, object], source_inputs: dict[str, object] | None = None
) -> tuple[dict[str, object] | None, float | None, int | None]:
    """TTM EPS + shares, or the _no_data payload when the eps envelope errors."""
    eps = sec_facts.get_fundamentals(ticker, "eps")
    if source_inputs is not None:
        source_inputs["sec_eps"] = eps
    if "error" in eps:
        return {"error": f"No valuation data for {ticker}: {eps.get('error') or ''!s}"}, None, None
    return None, _ttm_eps(eps), _shares_from_estimates(estimates)


def _finish_metrics(
    ticker: str,
    key: str,
    estimates: dict[str, object],
    price: float | None,
    price_gap: str | None,
    ob_rows: dict[str, object],
    shares_out: int | None,
    ttm_eps_diluted: float | None,
    impact: ObligationAnnualImpact,
    eps_scenarios: dict[str, object],
    estimates_by_period: dict[str, dict[str, object]],
    source_inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Forward EPS + coverage + final dict (+ cache when fully priced)."""
    fy_current = estimates_by_period.get("current_fiscal_year") or {}
    fy_next = estimates_by_period.get("next_fiscal_year") or {}
    eps_current = _period_eps(fy_current)
    eps_next = _period_eps(fy_next)
    (
        contractual_ps,
        contingent_ps,
        default_triggered_ps,
        revenue_matched_ps,
        gross_margin,
        margin_source,
        implied_coverage_b,
    ) = _valuation_drags(impact, ticker, shares_out)
    forward_eps, year_cur, year_next = _valuation_forward_tiers(
        impact, shares_out, fy_current, fy_next, eps_current, eps_next, price
    )
    coverage = _build_coverage(ob_rows, price_gap)
    value = _assemble_value(
        ticker,
        estimates,
        price,
        price_gap,
        coverage,
        shares_out,
        ttm_eps_diluted,
        impact,
        contractual_ps,
        contingent_ps,
        default_triggered_ps,
        revenue_matched_ps,
        implied_coverage_b,
        gross_margin,
        margin_source,
        eps_scenarios,
        forward_eps,
        year_cur,
        year_next,
    )
    value["derived"] = True
    value["calculation_sources"] = {
        **(source_inputs or {}),
        "live_price_input": {"id": f"quote:{ticker}", "value": price, "unit": "USD/share"},
        "sec_eps": {
            "id": f"ttm_eps_diluted:{ticker}",
            "value": (source_inputs or {}).get("sec_eps"),
            "unit": "USD/share",
        },
        "analyst_estimates": estimates,
        "consensus_basis": {
            "kind": "assumption",
            "description": "Yahoo sell-side consensus forward EPS and shares; treated as assumption, never fact.",
            "as_of": estimates.get("as_of"),
        },
        "obligation_snapshot": ob_rows,
        "annualization_years": {"kind": "assumption", "value": 6},
        "annualization_policy": {
            "kind": "assumption",
            "description": "Use disclosed schedules first. Otherwise use six years, ten years for off-balance-sheet leases, and 0.75 years for a fiscal remainder.",
        },
    }
    value["calculations"] = {
        "trailing_pe": _calculation(
            "valuation_pe",
            value["trailing_pe"],
            "ratio",
            {"live_price": price, "sec_eps.ttm_eps_diluted": ttm_eps_diluted},
        )
    }
    if price_gap is None:  # never cache a quote gap; retry fresh next call
        cache.set(key, value)
    return value


def get_valuation_metrics(ticker: str) -> dict[str, object]:
    """Price-anchored valuation with obligation-aware forward EPS (cached 15m)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return _no_data("", "empty ticker")
    key = f"valuation:{ticker}"
    cached = _cached_metrics(key)
    if cached is not None:
        return cached

    price, price_gap = _live_price_gap(ticker)
    est_error, estimates_by_period = _fetch_estimates(ticker)
    if est_error is not None:
        return est_error
    estimates = analyst_client.get_analyst_estimates(ticker)
    source_inputs: dict[str, object] = {
        "live_price": {"id": f"quote:{ticker}", "value": price, "unit": "USD/share"},
    }
    eps_error, ttm_eps_diluted, shares_out = _fetch_eps(ticker, estimates, source_inputs)
    if eps_error is not None:
        return eps_error
    ob_error, ob_rows = _fetch_obligations(ticker)
    if ob_error is not None:
        return ob_error
    impact = _obligation_annual_impact(_snapshot_of(ob_rows), years=6)
    tax_rate = _effective_tax_rate(ob_rows)
    eps_scenarios = _obligation_eps_scenarios(ob_rows, shares_out, tax_rate)
    return _finish_metrics(
        ticker,
        key,
        estimates,
        price,
        price_gap,
        ob_rows,
        shares_out,
        ttm_eps_diluted,
        impact,
        eps_scenarios,
        estimates_by_period,
        source_inputs,
    )


def _assemble_value(
    ticker: str,
    estimates: dict[str, object],
    price: float | None,
    price_gap: str | None,
    coverage: dict[str, object],
    shares_out: int | None,
    ttm_eps_diluted: float | None,
    impact: ObligationAnnualImpact,
    contractual_ps: float | None,
    contingent_ps: float | None,
    default_triggered_ps: float | None,
    revenue_matched_ps: float | None,
    implied_coverage_b: float | None,
    gross_margin: float | None,
    margin_source: str,
    eps_scenarios: dict[str, object],
    forward_eps: dict[str, dict[str, object] | None],
    year_cur: str | None,
    year_next: str | None,
) -> dict[str, object]:
    """Final metrics dict: price, obligations, forward EPS, projections, note."""
    return {
        "ticker": ticker,
        "as_of": estimates.get("as_of"),
        "fiscal_year_current": year_cur,
        "fiscal_year_next": year_next,
        "source": "live price (Yahoo Finance quote) + SEC EDGAR EPS + Yahoo consensus + SEC 10-Q/10-K notes",
        "price": {
            "last": round(price, 2) if price is not None else None,
            "retrieved_as_of": estimates.get("as_of"),
        },
        "price_gap": price_gap,
        "coverage": coverage,
        "shares_outstanding": shares_out,
        "ttm_gaap_eps": ttm_eps_diluted,
        "trailing_pe": (round(price / ttm_eps_diluted, 1) if (price is not None and ttm_eps_diluted) else None),
        "obligations": _obligations_block(
            impact,
            contractual_ps,
            contingent_ps,
            default_triggered_ps,
            revenue_matched_ps,
            implied_coverage_b,
            gross_margin,
            margin_source,
        ),
        "obligation_eps_scenarios": eps_scenarios,
        "forward_eps": forward_eps,
        "projected_prices": _projected_prices(_projected_eps_map(forward_eps, year_cur, year_next), price),
        "note": (
            "'Adjusted' includes only contractual (non-cancelable/firm) "
            "obligations from the latest 10-Q/10-K notes, annualized. "
            "'Scenario' adds contingent obligations that are cancellable / "
            "reducible / terminable per the filing (cloud, vendor, supply "
            "commitments) — no counterparty default assumed. "
            "'Scenario (counterparty default)' also adds default-triggered "
            "guarantees, which pay only if a named counterparty defaults "
            "or becomes insolvent. 'Worst case' additionally treats "
            "revenue-matched supply commitments as stranded costs (demand "
            "fails). Revenue-matched spend is NOT subtracted from consensus "
            "EPS (already embedded); it is reported separately with implied "
            "revenue coverage at the company's own gross margin (None when "
            "the filed gross-margin fact is missing). Buybacks, dividends, "
            "and unquantifiable indemnities are excluded."
        ),
    }


def _obligations_block(
    impact: ObligationAnnualImpact,
    contractual_ps: float | None,
    contingent_ps: float | None,
    default_triggered_ps: float | None,
    revenue_matched_ps: float | None,
    implied_coverage_b: float | None,
    gross_margin: float | None,
    margin_source: str,
) -> dict[str, object]:
    """Obligations block: annual $B buckets, per-share drags, margin coverage."""
    return {
        "contractual_annual_billions": impact["contractual_annual_billions"],
        "contingent_annual_billions": impact["contingent_annual_billions"],
        "default_triggered_annual_billions": impact["default_triggered_annual_billions"],
        "revenue_matched_annual_billions": impact["revenue_matched_annual_billions"],
        "drag_per_share_contractual": contractual_ps,
        "drag_per_share_contingent": contingent_ps,
        "drag_per_share_default_triggered": default_triggered_ps,
        "revenue_matched_per_share": revenue_matched_ps,
        "revenue_matched_implied_revenue_billions": round(implied_coverage_b, 1)
        if implied_coverage_b is not None
        else None,
        "revenue_matched_gross_margin": gross_margin,
        "revenue_matched_margin_source": margin_source,
        "per_kind": impact["per_kind"],
        "derived": True,
        "calculations": {
            **{
                name: _calculation(
                    "valuation_per_share_impact",
                    value,
                    "USD/share",
                    {
                        "annual_billions": annual,
                        "shares_outstanding": "calculation_sources.analyst_estimates.shares_outstanding",
                    },
                )
                for name, value, annual in (
                    ("drag_per_share_contractual", contractual_ps, impact["contractual_annual_billions"]),
                    ("drag_per_share_contingent", contingent_ps, impact["contingent_annual_billions"]),
                    (
                        "drag_per_share_default_triggered",
                        default_triggered_ps,
                        impact["default_triggered_annual_billions"],
                    ),
                    ("revenue_matched_per_share", revenue_matched_ps, impact["revenue_matched_annual_billions"]),
                )
            },
            "revenue_matched_implied_revenue_billions": _calculation(
                "valuation_implied_revenue",
                round(implied_coverage_b, 1) if implied_coverage_b is not None else None,
                "USD billions",
                {"annual_billions": impact["revenue_matched_annual_billions"], "gross_margin": gross_margin},
            ),
        },
        "annualization": {
            "kind": "assumption",
            "annualization_years": 6,
            "description": (
                "Disclosed schedules first; six years; ten years for off-balance-sheet leases; "
                "0.75 years for a fiscal remainder."
            ),
            "inputs": {"id": "calculation_sources.obligation_snapshot", "kind": "source_snapshot"},
        },
    }
