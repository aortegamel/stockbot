"""Offline regression tests for XBRL concept matching (all-tokens + ambiguity)."""

from typing import ClassVar

import pandas as pd
import pytest

from app import edgar_client


class _FakeFacts:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class _FakeCompany:
    rows: ClassVar[list[dict[str, object]]] = []

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.name = "Fake Corp"
        self.cik = "0000000001"
        self.sic_description = "Fake Industry"

    def get_facts(self) -> _FakeFacts:
        return _FakeFacts(type(self).rows)


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


def _row(concept: str, value: float = 1000) -> dict[str, object]:
    return {"concept": concept, "value": value, "period_end": "2026-08-01"}


def test_total_revenue_requires_all_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _FakeCompany,
        "rows",
        [
            _row("us-gaap:TotalAssets"),
            _row("us-gaap:RevenueFromContractWithCustomer"),
            _row("us-gaap:TotalRevenues"),
        ],
    )
    result = edgar_client.get_xbrl_facts("FAKE", "Revenue Total")
    assert "error" not in result
    matching = result["matching_concepts"]
    assert isinstance(matching, list)
    assert len(matching) >= 1
    assert all(r["concept"] == "us-gaap:TotalRevenues" for r in matching)


def test_ambiguous_concepts_return_no_data_without_mixing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _FakeCompany,
        "rows",
        [
            _row("us-gaap:TotalRevenues"),
            _row("us-gaap:TotalRevenuesNet"),
        ],
    )
    result = edgar_client.get_xbrl_facts("FAKE", "Revenue Total")
    assert "error" in result
    assert "ambiguous" in str(result["error"])
    assert "matching_concepts" not in result


def test_exact_tag_wins_over_substring_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _FakeCompany,
        "rows",
        [_row("us-gaap:Assets"), _row("us-gaap:AssetsCurrent"), _row("us-gaap:OtherAssetsNoncurrent")],
    )
    for concept in ("Assets", "us-gaap:Assets"):
        result = edgar_client.get_xbrl_facts("FAKE", concept)
        matching = result["matching_concepts"]
        assert isinstance(matching, list)
        assert [r["concept"] for r in matching] == ["us-gaap:Assets"]
    assert "ambiguous" in str(edgar_client.get_xbrl_facts("FAKE", "Asset")["error"])


def test_ownership_limit_clamps_and_rejects() -> None:
    assert edgar_client._ownership_limit(None) == 10
    assert edgar_client._ownership_limit(5) == 5
    assert edgar_client._ownership_limit(0) == 1
    assert edgar_client._ownership_limit(99) == 25
    assert edgar_client._ownership_limit(" 7 ") == 7
    assert edgar_client._ownership_limit("12.9") == 12
    assert edgar_client._ownership_limit(4.0) == 4
    assert edgar_client._ownership_limit(True) is None
    assert edgar_client._ownership_limit(2.5) is None
    assert edgar_client._ownership_limit("  ") is None
    assert edgar_client._ownership_limit("n/a") is None
    assert edgar_client._ownership_limit(object()) is None


def test_ownership_limit_rejects_at_feed_boundary() -> None:
    out = edgar_client._fetch_recent_ownership_filings("both", True)
    assert out.get("error") == "Invalid limit 'True': use 1-25"
    out = edgar_client._fetch_recent_ownership_filings("both", "n/a")
    assert out.get("error") == "Invalid limit 'n/a': use 1-25"
