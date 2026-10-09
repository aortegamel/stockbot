"""Unit tests for the obligation-aware valuation metrics.

Deterministic and offline: price, consensus, EPS, and obligations inputs
are injected via monkeypatch; nothing touches the network or cache.db.
"""

import pytest

from app import valuation
from app.policy import LOCAL_CONTEXT
from app.tools import execute_tool


class FakeCache:
    store: dict[str, object]

    def __init__(self) -> None:
        self.store = {}

    def get(self, key: str, ttl: float | None = None) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


def _estimates() -> dict[str, object]:
    return {
        "as_of": "2026-08-26T00:00:00Z",
        "quote": {"price": 213.05, "currency": "USD"},
        "shares_outstanding": 24_221_000_000,
        "forward_estimates": [
            {
                "period": "current_quarter",
                "period_end_date": "2026-07-31",
                "eps_avg": 2.09,
            },
            {
                "period": "next_quarter",
                "period_end_date": "2026-10-31",
                "eps_avg": 2.37,
            },
            {
                "period": "current_fiscal_year",
                "period_end_date": "2027-01-31",
                "eps_avg": 9.02,
            },
            {
                "period": "next_fiscal_year",
                "period_end_date": "2028-01-31",
                "eps_avg": 13.04,
            },
        ],
    }


def _obligation_rows() -> list[dict[str, object]]:
    return [
        {
            "type": "supply_commitments",
            "amount_billions": 119.0,
            "certainty": "contingent",
            "status": "future_cash_obligation",
            "revenue_matched": True,
            "payment_horizon": {
                "paid_in_remainder_of_fy": "2027",
                "paid_in_remainder_billions": 95.0,
                "paid_after_remainder_billions": 24.0,
            },
        },
        {
            "type": "cloud_commitments",
            "amount_billions": 30.0,
            "certainty": "contingent",
            "status": "future_cash_obligation",
            "revenue_matched": False,
            "payment_horizon": {
                "schedule": [
                    {"fiscal_year": "2027", "amount_billions": 6.0},
                    {"fiscal_year": "2028", "amount_billions": 7.0},
                    {"fiscal_year": "2029", "amount_billions": 7.0},
                    {"fiscal_year": "2030", "amount_billions": 5.0},
                    {"fiscal_year": "2031", "amount_billions": 3.0},
                    {"fiscal_year": "2032", "amount_billions": 2.0},
                ]
            },
        },
        {
            "type": "vendor_commitments",
            "amount_billions": 6.0,
            "certainty": "contingent",
            "status": "future_cash_obligation",
            "revenue_matched": False,
        },
        {
            "type": "operating_leases",
            "amount_billions": 5.604,
            "certainty": "contractual",
            "status": "future_cash_obligation",
            "revenue_matched": False,
            "schedule": [
                {"fiscal_year": "2027", "amount_billions": 0.46},
                {"fiscal_year": "2028", "amount_billions": 0.626},
                {"fiscal_year": "2029", "amount_billions": 0.602},
                {"fiscal_year": "2030", "amount_billions": 0.53},
                {"fiscal_year": "2031", "amount_billions": 0.462},
                {"fiscal_year": "2032", "amount_billions": 2.924},
            ],
        },
        {
            "type": "facility_lease_guarantees",
            "amount_billions": 3.5,
            "certainty": "contingent",
            "status": "contingent",
            "revenue_matched": False,
        },
        {
            "type": "8k_guarantees",
            "amount_billions": 105.0,
            "certainty": "contingent",
            "status": "contingent",
            "revenue_matched": False,
        },
    ]


def _obligations() -> dict[str, object]:
    return {"obligations": _obligation_rows()}


@pytest.fixture
def fake_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_price(ticker: str) -> float:
        return 213.05

    def _fake_estimates(ticker: str) -> dict[str, object]:
        return _estimates()

    def _fake_fundamentals(ticker: str, metric: str) -> dict[str, object]:
        return {"ttm_eps_diluted": 6.53}

    def _fake_obligations(ticker: str) -> dict[str, object]:
        return _obligations()

    def _fake_margin(ticker: str) -> tuple[float | None, str]:
        return (0.75, "company_facts")

    monkeypatch.setattr(valuation, "cache", FakeCache())
    monkeypatch.setattr(valuation, "get_live_price", _fake_price)
    monkeypatch.setattr(valuation.analyst_client, "get_analyst_estimates", _fake_estimates)
    monkeypatch.setattr(valuation.edgar_client, "get_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(valuation.sec_facts, "get_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(valuation.obligations, "get_obligations", _fake_obligations)
    monkeypatch.setattr(valuation, "_revenue_matched_margin", _fake_margin)


def test_trailing_pe_from_live_price(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    price = result["price"]
    assert isinstance(price, dict)
    assert price["last"] == 213.05
    assert result["ttm_gaap_eps"] == 6.53
    assert result["trailing_pe"] == pytest.approx(32.6, abs=0.1)


def test_three_eps_figures_never_conflated(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    consensus = fe["consensus"]
    adjusted = fe["adjusted"]
    scenario = fe["scenario"]
    with_defaults = fe["scenario_with_defaults"]
    assert consensus["eps"] == 9.02
    assert adjusted["eps_after_contractual"] is not None
    assert scenario["eps_after_all_obligations"] is not None
    assert with_defaults["eps_after_all_obligations"] is not None
    # The OpenAI-default scenario is strictly worse than no-default.
    assert with_defaults["eps_after_all_obligations"] < scenario["eps_after_all_obligations"]
    # Labels are distinct and explicit.
    assert "contractual obligations included" in adjusted["label"]
    assert "no counterparty default" in scenario["label"]
    assert "default-triggered" in with_defaults["label"]
    assert scenario["label"] != adjusted["label"]


def test_default_triggered_guarantees_separate_from_contingent(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    scenario = fe["scenario"]
    with_defaults = fe["scenario_with_defaults"]
    # 8-K $105B + facility $3.5B = $108.5B / 6y / 24.221B shares.
    delta = scenario["eps_after_all_obligations"] - with_defaults["eps_after_all_obligations"]
    assert delta == pytest.approx(108.5 / 6.0 / 24.221, abs=0.01)


def test_supply_front_loaded_is_revenue_matched_not_drag(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    ob = result["obligations"]
    assert isinstance(ob, dict)
    supply = ob["per_kind"]["supply_commitments"]
    assert supply["total_billions"] == 119.0
    assert supply["certainty"] == "contingent"
    assert supply["revenue_matched"] is True
    assert ob["revenue_matched_annual_billions"] > 100
    assert ob["revenue_matched_implied_revenue_billions"] > 400
    # Scenario EPS must NOT subtract supply spend (double-count).
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    scenario = fe["scenario"]
    assert scenario["eps_after_all_obligations"] > 8.0


def test_horizon_less_supply_falls_back_flat() -> None:
    rows: list[dict[str, object]] = [
        {
            "type": "supply_commitments",
            "amount_billions": 119.0,
            "certainty": "contingent",
            "status": "future_cash_obligation",
            "revenue_matched": True,
        },
    ]
    impact = valuation._obligation_annual_impact(rows, years=6)
    assert impact["revenue_matched_annual_billions"] == pytest.approx(119.0 / 6.0, abs=0.01)


def test_next_fy_figures(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    assert fe["consensus_next_fy"]["eps"] == 13.04
    assert fe["consensus_next_fy"]["pe"] == pytest.approx(16.3, abs=0.1)
    assert fe["scenario_next_fy"]["eps_after_all_obligations"] is not None


def test_worst_case_tier_includes_revenue_matched_supply(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    worst = fe["worst_case"]
    scenario = fe["scenario"]
    # Worst case must be strictly worse than the stress scenario: it adds
    # the revenue-matched supply drag (stranded-cost bear case).
    assert worst["eps_after_all_obligations"] < scenario["eps_after_all_obligations"]
    assert "stranded" in worst["label"]
    assert worst["eps_after_all_obligations"] > 0


def test_projected_prices_matrix(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    pp = result["projected_prices"]
    assert isinstance(pp, dict)
    assert pp["current_price"] == 213.05
    assert pp["multiples"] == [15, 20, 25, 30, 35]
    tiers_raw = pp["tiers"]
    assert isinstance(tiers_raw, list)
    by_tier = {t["tier"]: t for t in tiers_raw if isinstance(t, dict)}
    worst = by_tier["Worst case FY2027"]
    # Fixture worst-case FY27 EPS = 9.02 - 0.02 (leases FY27 0.46) - 3.92
    # (supply FY27 95.0 stranded) - 0.29 (cloud FY27 6.0 + vendor 1.0) -
    # 0.75 (default-triggered) = 4.04 (FY-matched; tail 4.8 sits in FY28+).
    assert worst["eps"] == 4.04
    assert worst["prices"]["15x"]["price"] == pytest.approx(60.6, abs=0.1)
    assert worst["prices"]["30x"]["price"] == pytest.approx(121.2, abs=0.1)
    # pct change vs current: 60.6/213.05 - 1 ≈ -71.6%.
    assert worst["prices"]["15x"]["pct_change_vs_current"] == pytest.approx(-71.6, abs=0.2)
    # Consensus FY2027 at 25x should exceed current price.
    consensus = by_tier["Consensus FY2027"]
    assert consensus["prices"]["25x"]["price"] > 213.05


def test_projected_prices_math_direct() -> None:
    pp = valuation._projected_prices({"Worst case FY27": 2.56, "Consensus FY28": 13.04}, price=213.05)
    tiers_raw = pp["tiers"]
    assert isinstance(tiers_raw, list)
    tiers = {t["tier"]: t for t in tiers_raw if isinstance(t, dict)}
    assert tiers["Consensus FY28"]["prices"]["20x"]["price"] == 260.8
    assert tiers["Consensus FY28"]["prices"]["35x"]["pct_change_vs_current"] == pytest.approx(114.2, abs=0.2)


def test_obligation_annual_impact_direct() -> None:
    impact = valuation._obligation_annual_impact(_obligation_rows(), years=6)
    assert impact["contractual_annual_billions"] == pytest.approx(0.934, abs=0.01)
    # Contingent (non-default): cloud schedule avg (6+7+7+5+3+2)/6 + vendor 6/6.
    expected_contingent = 30.0 / 6.0 + 6.0 / 6.0
    assert impact["contingent_annual_billions"] == pytest.approx(expected_contingent, abs=0.1)
    # Default-triggered: facility 3.5/6 + 8-K 105/6.
    expected_default = (3.5 + 105.0) / 6.0
    assert impact["default_triggered_annual_billions"] == pytest.approx(expected_default, abs=0.1)
    # Revenue-matched: supply front-loaded (95/0.75 + 24/5).
    expected_matched = 95.0 / 0.75 + 24.0 / 5.0
    assert impact["revenue_matched_annual_billions"] == pytest.approx(expected_matched, abs=0.1)


def test_fy_schedule_separation_no_blended_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FY absent from impact_by_fiscal_year must not inherit other years' schedule."""
    scheduled: dict[str, object] = {
        "type": "vendor_commitments",
        "amount_billions": 30.0,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": False,
        "schedule": [
            {"fiscal_year": "2028", "amount_billions": 10.0},
            {"fiscal_year": "2029", "amount_billions": 10.0},
            {"fiscal_year": "2030", "amount_billions": 10.0},
        ],
    }
    flat_vendor: dict[str, object] = {
        "type": "vendor_commitments",
        "amount_billions": 6.0,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": False,
    }

    def _run(rows: list[dict[str, object]]):
        def _run_price(ticker: str) -> float:
            return 213.05

        def _run_estimates(ticker: str) -> dict[str, object]:
            return _estimates()

        def _run_fundamentals(ticker: str, metric: str) -> dict[str, object]:
            return {"ttm_eps_diluted": 6.53}

        def _run_obligations(ticker: str) -> dict[str, object]:
            return {"obligations": rows}

        def _run_margin(ticker: str) -> tuple[float | None, str]:
            return (0.75, "company_facts")

        monkeypatch.setattr(valuation, "cache", FakeCache())
        monkeypatch.setattr(valuation, "get_live_price", _run_price)
        monkeypatch.setattr(valuation.analyst_client, "get_analyst_estimates", _run_estimates)
        monkeypatch.setattr(valuation.edgar_client, "get_fundamentals", _run_fundamentals)
        monkeypatch.setattr(valuation.sec_facts, "get_fundamentals", _run_fundamentals)
        monkeypatch.setattr(valuation.obligations, "get_obligations", _run_obligations)
        monkeypatch.setattr(valuation, "_revenue_matched_margin", _run_margin)
        return valuation.get_valuation_metrics("NVDA")

    shares = 24.221  # 24_221_000_000 from _estimates
    result = _run([scheduled])
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    assert fe["scenario"].get("contingent_drag_per_share", 0.0) == 0.0
    assert fe["scenario_next_fy"]["contingent_drag_per_share"] == pytest.approx(10.0 / shares, abs=0.01)
    impact = valuation._obligation_annual_impact([scheduled], years=6)
    assert "2027" not in impact["impact_by_fiscal_year"]
    assert impact["flat_annual_by_bucket"]["contingent"] == 0.0

    result = _run([scheduled, flat_vendor])
    fe = result["forward_eps"]
    assert isinstance(fe, dict)
    assert fe["scenario"]["contingent_drag_per_share"] == pytest.approx(1.0 / shares, abs=0.01)
    impact = valuation._obligation_annual_impact([scheduled, flat_vendor], years=6)
    assert impact["impact_by_fiscal_year"].get("2027", {}).get("contingent", 0.0) == 0.0
    assert impact["flat_annual_by_bucket"]["contingent"] == pytest.approx(1.0, abs=0.01)


def test_generic_scheduled_impact_attributes_fys_no_bleed() -> None:
    """Per-year schedules (incl. Thereafter) hit their disclosed FYs only;
    genuinely unscheduled rows still average into the flat bucket."""
    scheduled = {
        "type": "supply_commitments",
        "amount_billions": 8.0,
        "certainty": "contractual",
        "status": "future_cash_obligation",
        "revenue_matched": True,
        "schedule": [
            {"fiscal_year": "2027", "amount_billions": 4.0},
            {"fiscal_year": "2028", "amount_billions": 3.0},
            {"fiscal_year": "Thereafter", "amount_billions": 1.0},
        ],
    }
    flat_vendor = {
        "type": "vendor_commitments",
        "amount_billions": 6.0,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": False,
    }
    impact = valuation._obligation_annual_impact([scheduled, flat_vendor], years=6)
    fy = impact["impact_by_fiscal_year"]
    assert fy["2027"]["revenue_matched"] == pytest.approx(4.0)
    assert fy["2028"]["revenue_matched"] == pytest.approx(3.0)
    assert fy["Thereafter"]["revenue_matched"] == pytest.approx(1.0)
    assert "2029" not in fy
    assert "2030" not in fy
    assert impact["flat_annual_by_bucket"]["revenue_matched"] == 0.0
    assert impact["flat_annual_by_bucket"]["contingent"] == pytest.approx(1.0, abs=0.01)


def _snap_obligations() -> dict[str, object]:
    old: dict[str, object] = {
        "type": "vendor_commitments",
        "amount_billions": 20.0,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": False,
        "filed": "2026-02-01",
    }
    new: dict[str, object] = {**old, "amount_billions": 13.0, "filed": "2026-04-01"}
    manifest: list[str] = []
    warnings: list[str] = []
    return {
        "obligations": [old, new],
        "current_snapshot": [new],
        "coverage": {"scan_manifest": manifest, "quantified_count": 2, "unquantified_count": 0, "warnings": warnings},
    }


def test_valuation_uses_snapshot_not_ledger(monkeypatch: pytest.MonkeyPatch, fake_deps: None) -> None:
    """A superseded $20B + current $13B values at $13B, never $33B."""

    def _snap_fake(ticker: str) -> dict[str, object]:
        return _snap_obligations()

    monkeypatch.setattr(valuation.obligations, "get_obligations", _snap_fake)
    result = valuation.get_valuation_metrics("SYN")
    obligations = result["obligations"]
    assert isinstance(obligations, dict)
    assert obligations["contingent_annual_billions"] == pytest.approx(13.0 / 6.0, abs=0.01)


def test_eps_scenarios_missing_inputs_yield_none_with_reason() -> None:
    ob = {
        "obligations": [
            {
                "type": "vendor_commitments",
                "amount_billions": 6.0,
                "certainty": "contingent",
                "status": "future_cash_obligation",
                "revenue_matched": False,
            }
        ]
    }
    out = valuation._obligation_eps_scenarios(ob, None, None)
    assert out["effective_tax_rate"] is None
    assert out["scenarios"]
    scenarios = out["scenarios"]
    assert isinstance(scenarios, list)
    for s in scenarios:
        assert s["after_tax_billions"] is None
        assert s["eps_impact"] is None
        assert s["reason"] == "effective tax rate unavailable"
    out2 = valuation._obligation_eps_scenarios(ob, None, 0.2)
    assert out2["effective_tax_rate"] == pytest.approx(0.2)
    scenarios2 = out2["scenarios"]
    assert isinstance(scenarios2, list)
    for s in scenarios2:
        assert s["after_tax_billions"] is not None
        assert s["eps_impact"] is None
        assert s["reason"] == "diluted shares unavailable"


def test_missing_margin_yields_none_with_reason(monkeypatch: pytest.MonkeyPatch, fake_deps: None) -> None:
    def _no_margin(ticker: str) -> tuple[float | None, str]:
        return (None, "unavailable: gross margin fact missing")

    monkeypatch.setattr(
        valuation,
        "_revenue_matched_margin",
        _no_margin,
    )
    result = valuation.get_valuation_metrics("NVDA")
    ob = result["obligations"]
    assert isinstance(ob, dict)
    assert ob["revenue_matched_gross_margin"] is None
    assert ob["revenue_matched_margin_source"] == "unavailable: gross margin fact missing"
    assert ob["revenue_matched_implied_revenue_billions"] is None


def test_dynamic_fy_labels(fake_deps: None) -> None:
    result = valuation.get_valuation_metrics("NVDA")
    pp = result["projected_prices"]
    assert isinstance(pp, dict)
    tiers_raw = pp["tiers"]
    assert isinstance(tiers_raw, list)
    by_tier = {t["tier"]: t for t in tiers_raw if isinstance(t, dict)}
    assert "Consensus FY2027" in by_tier
    assert "Consensus FY2028" in by_tier
    assert "Worst case FY2027" in by_tier
    assert "Worst case FY2028" in by_tier
    assert all("FY27" not in t and "FY28" not in t for t in by_tier)


def test_unquantified_only_valuation_caveat(monkeypatch: pytest.MonkeyPatch, fake_deps: None) -> None:
    def _unquantified_only(ticker: str) -> dict[str, object]:
        manifest: list[str] = []
        warnings: list[str] = []
        empty_rows: list[dict[str, object]] = []
        return {
            "obligations": empty_rows,
            "current_snapshot": empty_rows,
            "unquantified_exposures": [{"type": "indemnities", "trigger": "unknown"}],
            "coverage": {
                "scan_manifest": manifest,
                "quantified_count": 0,
                "unquantified_count": 1,
                "warnings": warnings,
            },
        }

    monkeypatch.setattr(valuation.obligations, "get_obligations", _unquantified_only)
    result = valuation.get_valuation_metrics("NVDA")
    obligations = result["obligations"]
    assert isinstance(obligations, dict)
    assert obligations["contingent_annual_billions"] == 0.0
    assert obligations["contractual_annual_billions"] == 0.0
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    warnings = coverage["warnings"]
    assert isinstance(warnings, list)
    assert any("unquantified" in w for w in warnings if isinstance(w, str))


def test_valuation_skips_schedule_components() -> None:
    """Reconciled table components never double-count in EPS impact."""
    headline = {
        "type": "vendor_commitments",
        "amount_billions": 13.3,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": False,
        "default_triggered": False,
    }
    comps = [
        {
            **headline,
            "amount_billions": 6.0,
            "fiscal_year": "2027",
            "schedule_component": True,
            "headline_type": "vendor_commitments",
        },
        {
            **headline,
            "amount_billions": 7.3,
            "fiscal_year": "2028",
            "schedule_component": True,
            "headline_type": "vendor_commitments",
        },
    ]
    assert valuation._obligation_annual_impact([headline, *comps], 6) == valuation._obligation_annual_impact(
        [headline], 6
    )
    assert valuation._obligation_annual_impact([headline, *comps], 6)["contingent_annual_billions"] == pytest.approx(
        13.3 / 6.0, abs=0.01
    )


def test_get_live_quote_caches_retrieval_instant(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FakeCache()
    monkeypatch.setattr(valuation, "cache", cache)

    def _fake_quote_100(ticker: str) -> dict[str, object]:
        return {"price": 100.0, "retrieved_at": "2026-08-10T12:00:00Z"}

    monkeypatch.setattr(
        valuation.analyst_client,
        "get_quote_price",
        _fake_quote_100,
    )
    quote = valuation.get_live_quote("KO")
    assert quote == {"price": 100.0, "retrieved_at": "2026-08-10T12:00:00Z"}
    assert cache.store["live_price:KO"] == {"price": 100.0, "retrieved_at": "2026-08-10T12:00:00Z"}


def test_get_live_quote_legacy_row_yields_none_retrieved_at(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FakeCache()
    cache.store["live_price:KO"] = {"price": 65.0}
    monkeypatch.setattr(valuation, "cache", cache)
    assert valuation.get_live_quote("KO") == {"price": 65.0, "retrieved_at": None}


def test_get_live_price_returns_float_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(valuation, "cache", FakeCache())

    def _fake_quote_65(ticker: str) -> dict[str, object]:
        return {"price": 65.0, "retrieved_at": "2026-08-10T12:00:00Z"}

    monkeypatch.setattr(
        valuation.analyst_client,
        "get_quote_price",
        _fake_quote_65,
    )
    assert valuation.get_live_price("KO") == 65.0


def test_get_live_quote_ignores_stale_estimates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FakeCache()
    cache.store["analyst_estimates:KO"] = {"quote": {"price": 100.0}, "as_of": "2026-08-10T12:01:00Z"}
    monkeypatch.setattr(valuation, "cache", cache)

    def _fake_summary(ticker: str, modules: str) -> dict[str, object]:
        return {"price": {"regularMarketPrice": {"raw": 101.0}}}

    monkeypatch.setattr(
        valuation.analyst_client,
        "_quote_summary",
        _fake_summary,
    )
    assert valuation.get_live_quote("KO")["price"] == 101.0


def test_get_live_quote_yahoo_failure_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(valuation, "cache", FakeCache())

    def _boom(ticker: str, modules: str) -> dict[str, object]:
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(valuation.analyst_client, "_quote_summary", _boom)
    assert valuation.get_live_quote("KO") == {"price": None, "retrieved_at": None}


def test_eps_gateway_failure_reaches_tool_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DownGateway:
        def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
            raise OSError("SEC companyfacts unreachable")

    def _entity(ticker: str, as_of: object, data_root: object) -> str:
        return "sec:cik:0001045810"

    def _price(ticker: str) -> float:
        return 213.05

    def _fake_estimates(ticker: str) -> dict[str, object]:
        return _estimates()

    cache = FakeCache()
    monkeypatch.setattr(valuation, "cache", cache)
    monkeypatch.setattr(valuation, "get_live_price", _price)
    monkeypatch.setattr(valuation.analyst_client, "get_analyst_estimates", _fake_estimates)
    monkeypatch.setattr(valuation.sec_facts, "_resolve_entity", _entity)
    monkeypatch.setattr(valuation.sec_facts, "_gateway", _DownGateway)
    message = "company facts unavailable for sec:cik:0001045810: SEC companyfacts unreachable"
    with pytest.raises(RuntimeError, match=message):
        valuation.get_valuation_metrics("NVDA")
    result = execute_tool("get_valuation_metrics", {"ticker": "NVDA"}, "test", context=LOCAL_CONTEXT)
    assert result == {"error": f"Tool 'get_valuation_metrics' failed: {message}"}
    assert cache.store == {}
