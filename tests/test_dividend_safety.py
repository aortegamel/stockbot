"""Dividend safety tests (Phase 5): FCF-based coverage on SEC inputs only.
Warehouse-removal seam: providers + raw_archive, no persisted facts; future warehouse slots in behind SourceGateway."""

from collections.abc import Mapping

import pytest

from app import valuation
from app.domain.market.securities import TickerAlias
from app.normalization import normalize_sec_company_facts, normalize_sec_tickers
from app.services import sec_facts
from app.services.sec_facts import _assemble_dividend_safety

KO_CIK = 21344
RETRIEVED_AT = "2026-08-01T00:00:00Z"
AS_OF = "2026-08-10"

DPS_TAG = "CommonStockDividendsPerShareDeclared"
EPS_TAG = "EarningsPerShareDiluted"
OCF_TAG = "NetCashProvidedByUsedInOperatingActivities"
CAPX_TAG = "PaymentsToAcquirePropertyPlantAndEquipment"
PAID_TAG = "PaymentsOfDividendsCommonStock"
CASH_TAG = "CashAndCashEquivalentsAtCarryingValue"
DEBT_TAG = "LongTermDebtCurrentAndNoncurrent"
INCOME_TAG = "NetIncomeLoss"

# Four contiguous recent quarters (Q3'25-Q2'26); latest end 2026-06-30.
QUARTERS = [
    ("2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28"),
    ("2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10"),
    ("2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28"),
    ("2026-04-01", "2026-06-30", 2026, "Q2", "2026-07-28"),
]


class _Gateway:
    """SourceGateway double: normalized rows in, PIT-filtered company_facts out."""

    def __init__(self) -> None:
        self.alias_rows: list[dict[str, object]] = []
        self.facts_by_cik: dict[int, dict[str, list[dict[str, object]]]] = {}

    def ticker_candidates(self, ticker: str, as_of: object = None) -> list[TickerAlias]:
        """All aliases for the ticker, unfiltered (PIT stays in resolve_ticker_aliases)."""
        del as_of
        want = str(ticker).strip().upper()
        out: list[TickerAlias] = []
        for row in self.alias_rows:
            if str(row.get("alias_value") or "").strip().upper() != want:
                continue
            security_id = row.get("security_id")
            out.append(
                TickerAlias(
                    alias_type=str(row.get("alias_type")),
                    alias_value=str(row.get("alias_value")),
                    entity_id=str(row.get("entity_id")),
                    security_id=str(security_id) if security_id else None,
                    source=str(row.get("source")),
                    valid_from=str(row.get("valid_from")) if row.get("valid_from") else None,
                    valid_to=str(row.get("valid_to")) if row.get("valid_to") else None,
                    known_at=str(row.get("known_at")) if row.get("known_at") else None,
                    retrieved_at=str(row.get("retrieved_at")) if row.get("retrieved_at") else None,
                )
            )
        return out

    def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
        """Full normalized dict, PIT-filtered to as_of like the live gateway."""
        facts = self.facts_by_cik.get(int(cik))
        if facts is None:
            return {"documents": [], "financial_facts": [], "securities": [], "dividend_events": []}
        out: dict[str, object] = {name: list(rows) for name, rows in facts.items()}
        if as_of is None:
            return out
        for name, rows in out.items():
            if isinstance(rows, list):
                out[name] = [
                    row for row in rows if isinstance(row, dict) and str(row.get("known_at") or "")[:10] <= as_of
                ]
        return out


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> _Gateway:
    """SourceGateway double behind sec_facts._gateway (no warehouse)."""
    gw = _Gateway()
    monkeypatch.setattr(sec_facts, "_gateway", lambda: gw)
    return gw


def _merge(gw: _Gateway, cik: int, datasets: dict[str, list[dict[str, object]]]) -> None:
    merged = dict(gw.facts_by_cik.get(cik, {}))
    for name, rows in datasets.items():
        merged[name] = list(merged.get(name, [])) + list(rows)
    gw.facts_by_cik[cik] = merged


def _seed_ticker(gw: _Gateway, cik: int, ticker: str) -> None:
    datasets = normalize_sec_tickers(
        {"0": {"cik_str": cik, "ticker": ticker, "title": f"{ticker} Corp"}},
        retrieved_at=RETRIEVED_AT,
        content_hash=f"tickers-{cik}",
    )
    gw.alias_rows.extend(datasets.get("entity_aliases", []))


def _qfact(val: float, start: str, end: str, fy: int, fp: str, filed: str, accn: str) -> dict[str, object]:
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "filed": filed}


def _seed_concepts(gw: _Gateway, cik: int, concepts: Mapping[str, object], suffix: str) -> None:
    """Seed canonical + per-share facts through normalization (unit-aware)."""
    payload = {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "us-gaap": {tag: {"units": units} for tag, units in concepts.items()},
        },
    }
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash=f"safety-{suffix}-{cik}",
        source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        source_record_id=f"safety-{suffix}-cik{cik:010d}",
    )
    _merge(gw, cik, datasets)


def _quarters(
    tag_values: list[tuple[str, str, tuple[float, ...]]], prefix: str
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """tag -> unit -> quarterly facts zipped over QUARTERS."""
    out: dict[str, dict[str, list[dict[str, object]]]] = {}
    for tag, unit, values in tag_values:
        out[tag] = {
            unit: [
                _qfact(v, s, e, fy, fp, filed, f"{prefix}-{tag}-{fp}{fy}")
                for v, (s, e, fy, fp, filed) in zip(values, QUARTERS)
            ]
        }
    return out


def _seed_healthy(
    gw: _Gateway,
    *,
    dps: tuple[float, ...] = (0.51,) * 4,
    eps: tuple[float, ...] = (0.80,) * 4,
    ocf: tuple[float, ...] = (1000.0,) * 4,
    capx: tuple[float, ...] = (-200.0,) * 4,
    paid: tuple[float, ...] = (-260.0,) * 4,
) -> None:
    _seed_ticker(gw, KO_CIK, "KO")
    _seed_concepts(
        gw,
        KO_CIK,
        _quarters(
            [
                (DPS_TAG, "USD/shares", dps),
                (EPS_TAG, "USD/shares", eps),
                (OCF_TAG, "USD", ocf),
                (CAPX_TAG, "USD", capx),
                (PAID_TAG, "USD", paid),
            ],
            "q",
        ),
        "quarters",
    )
    fy = {
        OCF_TAG: {
            "USD": [
                _qfact(3500.0, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "fy-ocf-2024"),
                _qfact(3800.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "fy-ocf-2025"),
            ]
        },
        CAPX_TAG: {
            "USD": [
                _qfact(-700.0, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "fy-capx-2024"),
                _qfact(-750.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "fy-capx-2025"),
            ]
        },
        PAID_TAG: {
            "USD": [
                _qfact(-900.0, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "fy-paid-2024"),
                _qfact(-1000.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "fy-paid-2025"),
            ]
        },
        INCOME_TAG: {
            "USD": [
                _qfact(2500.0, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "fy-ni-2024"),
                _qfact(2800.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "fy-ni-2025"),
            ]
        },
    }
    _seed_concepts(gw, KO_CIK, fy, "fy")
    _seed_concepts(
        gw,
        KO_CIK,
        {
            DPS_TAG: {
                "USD/shares": [
                    _qfact(1.90, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "fy-dps-2024"),
                    _qfact(2.00, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "fy-dps-2025"),
                ]
            },
            CASH_TAG: {
                "USD": [
                    {
                        "end": "2026-06-30",
                        "val": 5000.0,
                        "accn": "cash-2026",
                        "fy": 2026,
                        "fp": "Q2",
                        "filed": "2026-07-28",
                    },
                ]
            },
            DEBT_TAG: {
                "USD": [
                    {
                        "end": "2025-06-30",
                        "val": 8000.0,
                        "accn": "debt-2025",
                        "fy": 2025,
                        "fp": "Q2",
                        "filed": "2025-07-28",
                    },
                    {
                        "end": "2026-06-30",
                        "val": 9000.0,
                        "accn": "debt-2026",
                        "fy": 2026,
                        "fp": "Q2",
                        "filed": "2026-07-28",
                    },
                ]
            },
        },
        "fy-div-cash-debt",
    )


def _fail_on_price(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker: str) -> dict[str, object]:
        raise AssertionError("historical dividend query must not call Yahoo")

    monkeypatch.setattr(valuation, "get_live_quote", _boom)


def _flags(safety: dict[str, object]) -> dict[str, dict[str, object]]:
    flags = safety["risk_flags"]
    assert isinstance(flags, list)
    out: dict[str, dict[str, object]] = {}
    for item in flags:
        assert isinstance(item, dict)
        key = item["flag"]
        assert isinstance(key, str)
        out[key] = item
    return out


def test_healthy_safety_ratios(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_healthy(gateway)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["data_source"] == "live"
    safety = result["safety"]
    assert isinstance(safety, dict)
    assert safety["methodology"] == "common-stock EPS/FCF basis; not AFFO/FFO"
    assert safety["ttm_fcf"] == 3200.0
    assert safety["ttm_dividends_paid"] == 1040.0
    assert safety["earnings_payout_ratio"] == 0.6375
    assert safety["fcf_payout_ratio"] == 0.325
    assert safety["fcf_coverage"] == 3.0769
    assert safety["cash_to_annual_dividend"] == 4.8077
    assert safety["interest_coverage"] is None
    assert safety["interest_coverage_reason"]
    assert safety["debt_up_yoy"] is True
    flags = _flags(safety)
    assert flags["fcf_declined_yoy"]["status"] is False
    assert flags["fcf_payout_expanded"]["status"] is False
    assert flags["eps_declined_yoy"]["status"] is False
    assert flags["leverage_rising"]["status"] is True
    assert flags["high_absolute_yield"]["status"] is None
    assert flags["growth_decelerating"]["status"] is None
    assert safety["dividend_vs_fcf_growth_5y"]["verdict"] == "insufficient_data"


def test_negative_eps_nulls_payout_with_flag(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_healthy(gateway, eps=(-0.50,) * 4)
    safety = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["safety"]
    assert isinstance(safety, dict)
    assert safety["earnings_payout_ratio"] is None
    assert safety["earnings_payout_ratio_reason"]
    assert _flags(safety)["negative_eps"]["status"] is True
    # FCF leg is unaffected.
    assert safety["fcf_payout_ratio"] == 0.325
    assert safety["fcf_coverage"] == 3.0769


def test_negative_fcf_nulls_payout_and_coverage_with_flag(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_healthy(gateway, ocf=(100.0,) * 4, capx=(-500.0,) * 4)
    safety = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["safety"]
    assert isinstance(safety, dict)
    assert safety["ttm_fcf"] == -1600.0
    assert safety["fcf_payout_ratio"] is None
    assert safety["fcf_coverage"] is None
    assert _flags(safety)["negative_or_zero_fcf"]["status"] is True
    assert safety["earnings_payout_ratio"] == 0.6375


def test_zero_dividend_nulls_coverage_with_flag(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_healthy(gateway, dps=(0.0,) * 4, paid=(0.0,) * 4)
    safety = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["safety"]
    assert isinstance(safety, dict)
    assert safety["ttm_dividends_paid"] == 0.0
    assert safety["fcf_payout_ratio"] == 0.0
    assert safety["fcf_coverage"] is None
    assert _flags(safety)["zero_dividend"]["status"] is True
    assert safety["cash_to_annual_dividend"] is None


def test_payout_expanding_verdict(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_concepts(
        gateway,
        KO_CIK,
        {
            DPS_TAG: {
                "USD/shares": [
                    _qfact(1.00, "2020-01-01", "2020-12-31", 2020, "FY", "2021-02-10", "v-dps-2020"),
                    _qfact(2.00, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "v-dps-2025"),
                ]
            },
            OCF_TAG: {
                "USD": [
                    _qfact(3500.0, "2020-01-01", "2020-12-31", 2020, "FY", "2021-02-10", "v-ocf-2020"),
                    _qfact(3800.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "v-ocf-2025"),
                ]
            },
            CAPX_TAG: {
                "USD": [
                    _qfact(-500.0, "2020-01-01", "2020-12-31", 2020, "FY", "2021-02-10", "v-capx-2020"),
                    _qfact(-750.0, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "v-capx-2025"),
                ]
            },
        },
        "verdict",
    )
    safety = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["safety"]
    assert isinstance(safety, dict)
    comp = safety["dividend_vs_fcf_growth_5y"]
    assert comp["dividend_cagr"] == 0.1487
    assert comp["fcf_cagr"] == 0.0033
    assert comp["verdict"] == "payout_expanding"


def test_high_absolute_yield_flag_unit() -> None:
    base = {"ttm_dividend_per_share": 2.0, "growth_1y": 0.05, "growth_5y_cagr": 0.10}
    assert _flags(_assemble_dividend_safety([], base, ttm_yield=0.068))["high_absolute_yield"]["status"] is True
    assert _flags(_assemble_dividend_safety([], base, ttm_yield=0.03))["high_absolute_yield"]["status"] is False
    assert _flags(_assemble_dividend_safety([], base))["high_absolute_yield"]["status"] is None
