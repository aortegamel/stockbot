"""Equity valuation engine: EV/EBITDA, peer multiples, DCF, football field.

Stdlib-only port of the FinRobot valuation_engine pattern. No overlap with
existing code: ``app/valuation.py`` is a price-anchored multiples snapshot
(trailing/forward P/E off live price), and
``app/domain/portfolio/valuation.py`` prices held positions (quantity x
quote) — neither does target-price methods (DCF/multiples/football
field), so this module owns that gap. All inputs are plain dicts supplied
by the caller from existing tool evidence (the engine never fetches,
never calls an LLM). Every method is None-safe: bad or missing inputs
yield a zero-price ValuationResult with confidence 0, never an exception.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

# Assumed net debt as a fraction of enterprise value. There is no
# balance-sheet input in the contract, so the debt haircut is fixed and
# explicit rather than inferred.
NET_DEBT_PCT_OF_EV = 0.10
_EQUITY_FACTOR = 1.0 - NET_DEBT_PCT_OF_EV

EV_EBITDA_METHOD = "EV/EBITDA"
PEER_METHOD = "Peer Comparison"
DCF_METHOD = "DCF"

_DCF_DEFAULTS: dict[str, float] = {
    "growth_rate_1_5": 0.10,
    "growth_rate_6_10": 0.05,
    "terminal_growth": 0.02,
    "wacc": 0.10,
    "projection_years": 10,
}


@dataclass
class ValuationResult:
    """One method's valuation: band around a mid target price."""

    method: str
    target_price: float
    low_estimate: float
    high_estimate: float
    assumptions: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0
    description: str = ""


def _num(value: object) -> float | None:
    """Finite float from an untrusted value (None when missing/non-numeric)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _num_list(values: object) -> list[float]:
    """Finite floats from an untrusted list ([] when absent/mistyped)."""
    if not isinstance(values, list):
        return []
    return [v for v in (_num(v) for v in values) if v is not None]


def _as_dict(value: object) -> dict[str, object]:
    """Narrow untrusted input to a mapping ({} when absent/mistyped)."""
    return value if isinstance(value, dict) else {}


def _zero_result(method: str, description: str, assumptions: dict[str, object] | None = None) -> ValuationResult:
    """Zero-price result for missing/invalid inputs (never raises)."""
    return ValuationResult(
        method=method,
        target_price=0.0,
        low_estimate=0.0,
        high_estimate=0.0,
        assumptions=assumptions or {},
        confidence=0.0,
        description=description,
    )


def _ev_to_price(ebitda: float, multiple: float, shares: float) -> float:
    """Equity value per share with the fixed net-debt haircut."""
    return max(ebitda * multiple * _EQUITY_FACTOR / shares, 0.0)


def _dcf_equity_value(fcf: float, g1: float, g2: float, gt: float, wacc: float, years: int) -> float | None:
    """Enterprise PV of 2-stage FCF + Gordon terminal, less net-debt haircut."""
    if wacc <= gt or years <= 0:
        return None
    pv = 0.0
    cash = fcf
    for year in range(1, years + 1):
        cash *= 1.0 + (g1 if year <= 5 else g2)
        pv += cash / (1.0 + wacc) ** year
    terminal = cash * (1.0 + gt) / (wacc - gt)
    pv += terminal / (1.0 + wacc) ** years
    return max(pv * _EQUITY_FACTOR, 0.0)


def _target_multiple(target_multiple: object, positive: list[float]) -> float | None:
    """Positive override multiple, else the history mean (None when neither exists)."""
    override = _num(target_multiple)
    if override is not None and override > 0:
        return override
    return statistics.mean(positive) if positive else None


def _history_stdev(positive: list[float]) -> float:
    """Sample stdev of the multiple history (0.0 below two points)."""
    try:
        return statistics.stdev(positive) if len(positive) >= 2 else 0.0
    except statistics.StatisticsError:
        return 0.0


def financial_data_from_evidence(
    valuation_metrics: dict[str, object] | None = None,
    dividend_fundamentals: dict[str, object] | None = None,
    ev_ebitda_history: list[float] | None = None,
    ebitda: float | None = None,
    free_cash_flow: float | None = None,
) -> dict[str, object]:
    """Pure producer for engine inputs from existing evidence envelopes (never fetches).

    Maps ``app/valuation.get_valuation_metrics`` ``price.last`` →
    ``current_price`` and ``shares_outstanding``, plus the dividend
    ``safety.ttm_fcf`` (OCF − CapEx, ``_safety_fcf``) → ``free_cash_flow``.
    Both are total-currency dollars over a share count, so the per-share
    math stays consistent. Explicit ``ebitda``/``free_cash_flow`` args win
    over envelope values (FCF override beats ``safety.ttm_fcf``); no XBRL
    EBITDA concept exists in the store, so ``ebitda`` stays ``None``
    unless the caller supplies it (EV/EBITDA + peer methods remain
    zero-confidence until then); ``ev_ebitda_history`` likewise comes
    only from the caller arg.
    """
    metrics = _as_dict(valuation_metrics)
    price = _num(_as_dict(metrics.get("price")).get("last"))
    shares = _num(metrics.get("shares_outstanding"))
    safety = _as_dict(_as_dict(dividend_fundamentals).get("safety"))
    fcf = _num(free_cash_flow) if free_cash_flow is not None else _num(safety.get("ttm_fcf"))
    return {
        "current_price": price if price is not None else 0.0,
        "shares_outstanding": shares if shares is not None else 0.0,
        "ebitda": _num(ebitda),
        "free_cash_flow": fcf,
        "ev_ebitda_history": _num_list(ev_ebitda_history),
    }


def from_evidence(
    valuation_metrics: dict[str, object] | None,
    peer_multiples: dict[str, float] | None = None,
    ev_ebitda_history: list[float] | None = None,
    dividend_fundamentals: dict[str, object] | None = None,
    ebitda: float | None = None,
    free_cash_flow: float | None = None,
) -> ValuationEngine:
    """Thin adapter: existing evidence envelopes → engine inputs.

    Not a parallel engine — the snapshot stays the single source for
    price-anchored multiples; this only maps its
    ``price.last``/``shares_outstanding`` keys, the dividend ``safety.ttm_fcf``
    → ``free_cash_flow`` (so DCF has a real producer), plus an optional
    peer-multiple map into the DCF/football-field inputs.
    Explicit ``ebitda``/``free_cash_flow`` override envelope values;
    EBITDA has no store source, so EV/EBITDA + peer methods stay
    zero-confidence until a caller supplies ``ebitda``; missing values
    yield zero-price results, never raises.
    """
    peers: dict[str, object] = {}
    for ticker, multiple in _as_dict(peer_multiples).items():
        value = _num(multiple)
        if value is not None and value > 0:
            peers[ticker] = {"ev_ebitda": value}
    financial_data = financial_data_from_evidence(
        valuation_metrics, dividend_fundamentals, ev_ebitda_history, ebitda, free_cash_flow
    )
    return ValuationEngine(financial_data, peers)


class ValuationEngine:
    """Three-method valuation over caller-supplied plain-dict inputs."""

    def __init__(self, financial_data: dict[str, object], peer_data: dict[str, object] | None = None) -> None:
        self.financial_data = _as_dict(financial_data)
        self.peer_data = _as_dict(peer_data)

    def _price(self) -> float:
        return _num(self.financial_data.get("current_price")) or 0.0

    def _shares(self) -> float | None:
        shares = _num(self.financial_data.get("shares_outstanding"))
        return shares if shares is not None and shares > 0 else None

    def _ebitda(self) -> float | None:
        ebitda = _num(self.financial_data.get("ebitda"))
        return ebitda if ebitda is not None and ebitda > 0 else None

    def _fcf(self) -> float | None:
        fcf = _num(self.financial_data.get("free_cash_flow"))
        return fcf if fcf is not None and fcf > 0 else None

    def calculate_ev_ebitda_valuation(self, target_multiple: float | None = None) -> ValuationResult:
        """EV/EBITDA target with an avg±stdev band over the multiple history."""
        ebitda = self._ebitda()
        shares = self._shares()
        history = _num_list(self.financial_data.get("ev_ebitda_history"))
        positive = [m for m in history if m > 0]
        multiple = _target_multiple(target_multiple, positive)
        base_assumptions = {
            "net_debt_pct_of_ev": NET_DEBT_PCT_OF_EV,
            "net_debt_note": "net debt assumed 10% of EV (equity = 0.9 * EV); no balance-sheet input",
            "history_count": len(positive),
        }
        if multiple is None or ebitda is None or shares is None:
            return _zero_result(
                EV_EBITDA_METHOD,
                "Missing EBITDA, shares outstanding, or EV/EBITDA multiple.",
                {**base_assumptions, "target_multiple": multiple},
            )
        stdev = _history_stdev(positive)
        low_m = max(multiple - stdev, 0.0)
        return ValuationResult(
            method=EV_EBITDA_METHOD,
            target_price=_ev_to_price(ebitda, multiple, shares),
            low_estimate=_ev_to_price(ebitda, low_m, shares),
            high_estimate=_ev_to_price(ebitda, multiple + stdev, shares),
            assumptions={**base_assumptions, "target_multiple": multiple, "history_stdev": stdev},
            confidence=0.7 if positive else 0.5,
            description=f"EBITDA x {multiple:.1f}x EV/EBITDA, ±{stdev:.1f}x history band.",
        )

    def calculate_peer_comparison_valuation(self) -> ValuationResult:
        """Peer-multiple target with a min/avg/max band over peer EV/EBITDA."""
        ebitda = self._ebitda()
        shares = self._shares()
        multiples = sorted(
            m
            for peer in self.peer_data.values()
            if isinstance(peer, dict)
            for m in [_num(peer.get("ev_ebitda"))]
            if m is not None and m > 0
        )
        base_assumptions: dict[str, object] = {
            "net_debt_pct_of_ev": NET_DEBT_PCT_OF_EV,
            "net_debt_note": "net debt assumed 10% of EV (equity = 0.9 * EV); no balance-sheet input",
            "peer_count": len(multiples),
        }
        if not multiples or ebitda is None or shares is None:
            return _zero_result(
                PEER_METHOD,
                "Missing EBITDA, shares outstanding, or peer EV/EBITDA multiples.",
                base_assumptions,
            )
        avg = statistics.mean(multiples)
        return ValuationResult(
            method=PEER_METHOD,
            target_price=_ev_to_price(ebitda, avg, shares),
            low_estimate=_ev_to_price(ebitda, multiples[0], shares),
            high_estimate=_ev_to_price(ebitda, multiples[-1], shares),
            assumptions={
                **base_assumptions,
                "avg_multiple": avg,
                "min_multiple": multiples[0],
                "max_multiple": multiples[-1],
            },
            confidence=0.6 if len(multiples) >= 2 else 0.5,
            description=f"EBITDA x {avg:.1f}x average peer EV/EBITDA ({len(multiples)} peers).",
        )

    def _dcf_params(self, assumptions: object) -> dict[str, float] | None:
        """Merged DCF params (None when the merged set is unusable)."""
        supplied = _as_dict(assumptions)
        merged = {**_DCF_DEFAULTS}
        for key in ("growth_rate_1_5", "growth_rate_6_10", "terminal_growth", "wacc"):
            value = _num(supplied.get(key))
            if value is not None:
                merged[key] = value
        years = _num(supplied.get("projection_years"))
        if years is not None and years > 0:
            merged["projection_years"] = int(years)
        if merged["wacc"] <= merged["terminal_growth"]:
            return None
        return merged

    def calculate_dcf_valuation(self, assumptions: dict[str, object] | None = None) -> ValuationResult:
        """2-stage FCF DCF with a WACC±1% band around the mid target."""
        fcf = self._fcf()
        shares = self._shares()
        params = self._dcf_params(assumptions)
        if params is None or fcf is None or shares is None:
            return _zero_result(
                DCF_METHOD,
                "Missing free cash flow, shares outstanding, or usable DCF assumptions "
                "(WACC must exceed terminal growth).",
                {"supplied": _as_dict(assumptions), "defaults": dict(_DCF_DEFAULTS)},
            )
        years = int(params["projection_years"])
        mid = _dcf_equity_value(
            fcf,
            params["growth_rate_1_5"],
            params["growth_rate_6_10"],
            params["terminal_growth"],
            params["wacc"],
            years,
        )
        if mid is None:
            return _zero_result(DCF_METHOD, "WACC must exceed terminal growth.", {"supplied": _as_dict(assumptions)})
        wacc = params["wacc"]
        hi_equity = _dcf_equity_value(
            fcf,
            params["growth_rate_1_5"],
            params["growth_rate_6_10"],
            params["terminal_growth"],
            wacc - 0.01,
            years,
        )
        lo_equity = _dcf_equity_value(
            fcf,
            params["growth_rate_1_5"],
            params["growth_rate_6_10"],
            params["terminal_growth"],
            wacc + 0.01,
            years,
        )
        mid_price = mid / shares
        return ValuationResult(
            method=DCF_METHOD,
            target_price=mid_price,
            low_estimate=(lo_equity / shares) if lo_equity is not None else mid_price,
            high_estimate=(hi_equity / shares) if hi_equity is not None else mid_price,
            assumptions={
                **params,
                "wacc_low": wacc - 0.01,
                "wacc_high": wacc + 0.01,
                "net_debt_pct_of_ev": NET_DEBT_PCT_OF_EV,
            },
            confidence=0.65,
            description=f"{years}y 2-stage DCF at {wacc:.1%} WACC, ±1% WACC band.",
        )

    def generate_football_field_data(self) -> dict[str, object]:
        """Per-method {low, mid, high} bands plus the current price."""
        results = [
            self.calculate_ev_ebitda_valuation(),
            self.calculate_peer_comparison_valuation(),
            self.calculate_dcf_valuation(),
        ]
        field: dict[str, object] = {
            r.method: {"low": r.low_estimate, "mid": r.target_price, "high": r.high_estimate} for r in results
        }
        field["current_price"] = self._price()
        return field

    def synthesize_valuation(self) -> dict[str, object]:
        """Confidence-weighted target, full range, and per-method results."""
        results = [
            self.calculate_ev_ebitda_valuation(),
            self.calculate_peer_comparison_valuation(),
            self.calculate_dcf_valuation(),
        ]
        usable = [r for r in results if r.confidence > 0]
        price = self._price()
        if not usable:
            return {
                "target_price": 0.0,
                "range": (0.0, 0.0),
                "methods_used": [],
                "current_price": price,
                "upside": None,
                "individual_results": results,
            }
        weight = sum(r.confidence for r in usable)
        target = sum(r.target_price * r.confidence for r in usable) / weight
        return {
            "target_price": target,
            "range": (min(r.low_estimate for r in usable), max(r.high_estimate for r in usable)),
            "methods_used": [r.method for r in usable],
            "current_price": price,
            "upside": (target / price - 1.0) if price > 0 else None,
            "individual_results": results,
        }

    def explain_valuation_differences(self) -> str:
        """One-paragraph explainer comparing method mids to each other and price."""
        results = [
            self.calculate_ev_ebitda_valuation(),
            self.calculate_peer_comparison_valuation(),
            self.calculate_dcf_valuation(),
        ]
        usable = [r for r in results if r.confidence > 0]
        if not usable:
            return "Insufficient inputs for valuation: no method produced a usable target."
        price = self._price()
        mids = {r.method: r.target_price for r in usable}
        spread = max(mids.values()) - min(mids.values())
        base = max(mids.values())
        lines = [
            " vs ".join(f"{m} ${p:,.2f}" for m, p in sorted(mids.items())) + ".",
            f"Spread ${spread:,.2f} ({spread / base:.0%} of the highest target).",
        ]
        if price > 0:
            lines.append(
                " vs current price: "
                + ", ".join(f"{m} implies {p / price - 1.0:+.0%}" for m, p in sorted(mids.items()))
                + "."
            )
        lines.append(
            "EV/EBITDA reflects what the market paid for these earnings; "
            "peers reflect relative pricing; DCF reflects cash generation, "
            "so growth-vs-multiple disagreement drives most of the gap."
        )
        return " ".join(lines)
