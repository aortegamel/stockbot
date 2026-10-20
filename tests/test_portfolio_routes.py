"""Portfolio API routes over a mocked Robinhood client: gating, sync, no identifiers."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.portfolio_routes import SETUP_MESSAGE, router
from app import tools
from app.data_sources import SourceGateway
from app.domain.market.securities import TickerAlias
from app.services.account_identity import local_account_id
from tests.test_portfolio_sync import FakeClient, _happy_payloads

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))

    def _unknown(self: object, ticker: str, as_of: datetime) -> list[TickerAlias]:
        del self, ticker, as_of
        return []

    monkeypatch.setattr(SourceGateway, "ticker_candidates", _unknown)


def _fake_broker(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient(_happy_payloads())

    def _client(*, account_tools: frozenset[str] = frozenset()) -> FakeClient:
        assert account_tools == {"get_accounts", "get_portfolio", "get_equity_positions"}
        return fake

    monkeypatch.setattr(tools, "_robinhood_client", _client)
    return fake


@pytest.mark.parametrize(("method", "path"), [("GET", "/api/portfolio"), ("POST", "/api/portfolio/sync")])
def test_broker_disabled_is_404_with_setup_hint(monkeypatch: pytest.MonkeyPatch, method: str, path: str) -> None:
    monkeypatch.setenv("BROKER_ENABLED", "false")
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.json() == {"detail": SETUP_MESSAGE}
    assert "venv/bin/python cli.py robinhood-login" in SETUP_MESSAGE


@pytest.mark.parametrize(("method", "path"), [("GET", "/api/portfolio"), ("POST", "/api/portfolio/sync")])
def test_missing_oauth_is_404_without_client(monkeypatch: pytest.MonkeyPatch, method: str, path: str) -> None:
    monkeypatch.setenv("BROKER_ENABLED", "true")

    def _invalid_tokens(*a: object, **k: object) -> bool:
        return False

    monkeypatch.setattr(tools, "has_valid_tokens", _invalid_tokens)

    def _no_client(*a: object, **k: object) -> None:
        pytest.fail("RobinhoodClient must not be constructed")

    monkeypatch.setattr(tools, "RobinhoodClient", _no_client)
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.json() == {"detail": SETUP_MESSAGE}


def test_get_before_first_sync_is_empty_and_calls_no_broker_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_broker(monkeypatch)
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    assert response.json() == {
        "synced_at": None,
        "cash": None,
        "invested_value": None,
        "total_value": None,
        "positions": [],
    }
    assert fake.calls == []


def test_sync_then_get_serves_cached_holdings_without_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_broker(monkeypatch)
    synced = client.post("/api/portfolio/sync")
    assert synced.status_code == 200
    body = synced.json()
    assert [name for name, _ in fake.calls].count("get_accounts") == 1
    assert body["synced_at"]
    assert body["cash"] == "3234.56"
    wing = next(p for p in body["positions"] if p["ticker"] == "WING")
    assert wing == {
        "ticker": "WING",
        "quantity": "10",
        "average_cost": "95.50",
        "market_price": "116.84",
        "market_value": "1168.40",
        "unrealized_gain": "213.40",
        "unrealized_gain_pct": wing["unrealized_gain_pct"],
    }
    assert wing["unrealized_gain_pct"] is not None
    calls = len(fake.calls)
    cached = client.get("/api/portfolio")
    assert cached.status_code == 200
    assert len(fake.calls) == calls
    assert [p["ticker"] for p in cached.json()["positions"]] == [p["ticker"] for p in body["positions"]]
    assert cached.json()["synced_at"] == body["synced_at"]
    for payload in (body, cached.json()):
        text = json.dumps(payload)
        for secret in ("100000001", "100000002", local_account_id("100000001"), local_account_id("100000002")):
            assert secret not in text
        assert "account" not in text
        assert "position_id" not in text
