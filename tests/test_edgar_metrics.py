"""Offline tests for the corrected shares_outstanding fundamental.

The metric was renamed from 'shares_float' because the value is SEC-reported
shares outstanding, not public float. The old name remains a deprecated alias
that returns the same value with an explicit note.
"""

from typing import override

import pandas as pd
import pytest

from app import edgar_client
from app.sec.events8k import KNOWN_8K_ITEMS, parse_8k_events
from app.sec.material import EIGHT_K_ITEM_EVENTS, material_events_from_8k
from app.tools import TOOLS


class _FakeFacts:
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"concept": "dei:EntityCommonStockSharesOutstanding", "value": 1000, "period_end": "2026-08-01"},
            ]
        )


class _FakeCompany:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.name = "Fake Corp"
        self.cik = "0000000001"
        self.sic_description = "Fake Industry"

    def get_facts(self) -> _FakeFacts:
        return _FakeFacts()


class _FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str, ttl: float | None = None) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def fake_edgar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edgar_client, "Company", _FakeCompany)
    monkeypatch.setattr(edgar_client, "cache", _FakeCache())
    monkeypatch.setattr(edgar_client, "_ensure_init", lambda: None)


def _as_seq(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, dict)
    return value


def test_shares_outstanding_is_sec_shares_not_float():
    result = edgar_client.get_fundamentals("FAKE", "shares_outstanding")
    assert result["shares_outstanding"] == 1000
    assert result["as_of"] == "2026-08-01"
    note = result["note"]
    assert isinstance(note, str)
    assert "not public float" in note


def test_shares_float_alias_returns_shares_outstanding():
    result = edgar_client.get_fundamentals("FAKE", "shares_float")
    assert result["shares_outstanding"] == 1000
    note = result["note"]
    assert isinstance(note, str)
    assert "not public float" in note


def test_tool_schema_offers_shares_outstanding_not_shares_float():
    schema = next(
        item
        for item in TOOLS
        if isinstance(item, dict)
        and isinstance(item["function"], dict)
        and item["function"]["name"] == "get_fundamentals"
    )
    func = schema["function"]
    assert isinstance(func, dict)
    params = func["parameters"]
    assert isinstance(params, dict)
    props = params["properties"]
    assert isinstance(props, dict)
    metric = props["metric"]
    assert isinstance(metric, dict)
    enum = metric["enum"]
    assert isinstance(enum, list)
    assert "shares_outstanding" in enum
    assert "shares_float" not in enum
    desc = func["description"]
    assert isinstance(desc, str)
    assert "not public float" in desc


_EPS_ROWS = [
    # (value, period_start, period_end, fiscal_period, fiscal_year)
    (0.76, "2025-01-27", "2025-04-27", "Q1", 2026),
    (0.77, "2025-01-27", "2025-04-27", "Q1", 2026),  # basic
    (1.84, "2025-01-27", "2025-07-27", "Q2", 2026),  # YTD (6 months)
    (1.08, "2025-04-28", "2025-07-27", "Q2", 2026),  # quarterly
    (3.14, "2025-01-27", "2025-10-26", "Q3", 2026),  # YTD (9 months)
    (1.30, "2025-07-28", "2025-10-26", "Q3", 2026),  # quarterly (restated)
    (4.90, "2025-01-27", "2026-01-25", "FY", 2026),  # full year (Q4 only as FY)
    (2.39, "2026-01-26", "2026-04-26", "Q1", 2027),
    (2.40, "2026-01-26", "2026-04-26", "Q1", 2027),  # basic
]


def _eps_facts() -> pd.DataFrame:
    rows = []
    for i, (value, start, end, period, year) in enumerate(_EPS_ROWS):
        concept = "us-gaap:EarningsPerShareDiluted"
        if i in (1, 8):
            concept = "us-gaap:EarningsPerShareBasic"
        rows.append(
            {
                "concept": concept,
                "value": value,
                "period_start": start,
                "period_end": end,
                "fiscal_period": period,
                "fiscal_year": year,
            }
        )
    return pd.DataFrame(rows)


class _EpsFakeFacts(_FakeFacts):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @override
    def to_dataframe(self) -> pd.DataFrame:
        return self._df


class _EpsFakeCompany(_FakeCompany):
    @override
    def get_facts(self) -> _EpsFakeFacts:
        return _EpsFakeFacts(df=_eps_facts())


def test_eps_ttm_uses_quarterly_facts_and_derives_q4(monkeypatch: pytest.MonkeyPatch):
    """TTM must not sum YTD/full-year facts (old bug: NVDA TTM was 8.13
    instead of 6.53) and must derive Q4 from FY_total - YTD_through_Q3."""
    monkeypatch.setattr(edgar_client, "Company", _EpsFakeCompany)
    monkeypatch.setattr(edgar_client, "cache", _FakeCache())
    result = edgar_client.get_fundamentals("FAKE", "eps")
    by_end = {q["period_end"]: q["eps_diluted"] for q in _as_seq(result["quarterly_eps"])}
    assert by_end["2025-07-27"] == 1.08
    assert by_end["2025-10-26"] == 1.30
    assert by_end["2026-01-25"] == 1.76  # derived: 4.90 - 3.14
    assert by_end["2026-04-26"] == 2.39
    assert result["ttm_eps_diluted"] == 6.53


# Step-1 8-K mapping regression, migrated to the new suite after
# get_filing_section was retired: bankruptcy is Item 1.03 (never 2.06)
# and impairments is Item 2.06 end to end (parse -> event -> vocabulary).


def test_8k_bankruptcy_is_item_103_not_206():
    assert EIGHT_K_ITEM_EVENTS["1.03"] == "bankruptcy"
    assert EIGHT_K_ITEM_EVENTS["2.06"] == "impairment"
    assert "Bankruptcy" in KNOWN_8K_ITEMS["1.03"]
    assert "Impairment" in KNOWN_8K_ITEMS["2.06"]


def test_8k_events_route_103_and_206_texts():
    events = parse_8k_events(
        "0000000001-26-000001",
        {"Item 1.03": "bankruptcy text", "Item 2.06": "impairments text EX-99.1"},
    )
    by_number = {event.item_number: event for event in events}
    assert list(by_number["2.06"].exhibit_refs) == ["EX-99.1"]
    assert by_number["2.06"].text == "impairments text EX-99.1"


def test_8k_bankruptcy_maps_to_bankruptcy_event():
    events = material_events_from_8k(
        "0000000001-26-000001",
        parse_8k_events("0000000001-26-000001", {"Item 1.03": "bankruptcy text"}),
        issuer="FAKE",
    )
    assert [event.event_type for event in events] == ["bankruptcy"]
    assert events[0].severity == "critical"
