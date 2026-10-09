"""End-to-end Robinhood portfolio sync tests with a recorded fake client.

Coverage under test:

- the exact read-only call sequence (accounts, per-account positions and
  portfolio, one batched deduplicated quotes call);
- snapshot math (cash sum, invested/total value, weights, zero-quantity
  positions, Decimal market values, deterministic snapshot/position ids);
- fail-explicit behavior on malformed fixture rows;
- the persistence round trip through ``read_latest_snapshot``;
- missing quote prices degrading the snapshot without crashing;
- no OAuth/token data in any persisted row.

Warehouse-removal seam: identity resolution reads live providers via SourceGateway; snapshots persist to the portfolio.sqlite ops store only — a future warehouse slots in behind the gateway.
"""

import dataclasses
import json
from datetime import UTC, datetime, tzinfo
from decimal import Decimal
from pathlib import Path
from typing import override

import pytest

from app.data_sources import SourceGateway
from app.domain.market.securities import SecurityResolution, TickerAlias
from app.domain.portfolio import BrokeragePositionInput, PortfolioSnapshot, Position
from app.domain.portfolio.snapshot import build_portfolio_snapshot
from app.domain.portfolio.valuation import build_position
from app.robinhood.portfolio import RobinhoodPortfolioProvider
from app.services.account_identity import local_account_id
from app.services.portfolio_sync import (
    persist_snapshot,
    read_latest_snapshot,
    resolve_security,
    sync_robinhood_portfolio,
)

FIXTURES = Path(__file__).parent / "fixtures" / "robinhood"

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
SNAPSHOT_ID = "portfolio:robinhood:2026-08-25T12:00:00+00:00"
QUOTE_TIME = datetime(2026, 8, 25, 15, 0, tzinfo=UTC)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def _inline_positions(account_number: str) -> dict[str, object]:
    positions = {
        "100000001": [
            {
                "id": "pos-2",
                "instrument_id": "instr-aapl",
                "symbol": "AAPL",
                "quantity": "0.5",
                "average_buy_price": "150.25",
            },
            {
                "id": "pos-3",
                "instrument_id": "instr-tsla",
                "symbol": "TSLA",
                "quantity": "0",
                "average_buy_price": "200.00",
            },
            {
                "id": "pos-1",
                "instrument_id": "instr-wing",
                "symbol": "WING",
                "quantity": "10",
                "average_buy_price": "95.50",
            },
        ],
        "100000002": [
            {
                "id": "pos-5",
                "instrument_id": "instr-msft",
                "symbol": "MSFT",
                "quantity": "5",
                "average_buy_price": "400.00",
            },
            {
                "id": "pos-6",
                "instrument_id": "instr-wing-2",
                "symbol": "WING",
                "quantity": "1",
                "average_buy_price": "100.00",
            },
        ],
    }
    return {"data": {"positions": positions.get(account_number, [])}}


def _inline_balance(account_number: str) -> dict[str, object]:
    cash = {"100000001": "1234.56", "100000002": "2000.00"}[account_number]
    return {"data": {"cash": cash, "buying_power": {"buying_power": "2500.00"}}}


def _inline_quotes():
    return {
        "data": {
            "results": [
                {
                    "quote": {
                        "symbol": "WING",
                        "last_trade_price": "116.84",
                        "bid_price": "116.83",
                        "ask_price": "116.85",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                },
                {
                    "quote": {
                        "symbol": "AAPL",
                        "last_trade_price": "160.00",
                        "bid_price": "159.90",
                        "ask_price": "160.10",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                },
                {
                    "quote": {
                        "symbol": "TSLA",
                        "last_trade_price": "250.00",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                },
                {
                    "quote": {
                        "symbol": "MSFT",
                        "last_trade_price": "405.00",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                },
            ]
        }
    }


class FakeClient:
    """Records call_tool(name, args) and serves payloads keyed by tool name.

    Payloads may be plain values or callables receiving the arguments dict
    (used for per-account responses, mirroring the real MCP server).
    """

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, object]] = []

    def call_tool(self, name: str, arguments: dict[str, object] | None = None) -> object:
        self.calls.append((name, arguments))
        payload = self.payloads[name]
        if callable(payload):
            return payload(arguments or {})
        return payload


def _positions_for(args: dict[str, object]) -> dict[str, object]:
    account = args["account_number"]
    assert isinstance(account, str)
    return _inline_positions(account)


def _balance_for(args: dict[str, object]) -> dict[str, object]:
    account = args["account_number"]
    assert isinstance(account, str)
    return _inline_balance(account)


def _happy_payloads() -> dict[str, object]:
    return {
        "get_accounts": _fixture("accounts.json"),
        "get_equity_positions": _positions_for,
        "get_portfolio": _balance_for,
        "get_equity_quotes": _inline_quotes(),
    }


def _wing_alias(
    *,
    entity_id: str = "sec:cik:0000320193",
    security_id: str | None = "sec:equity:0000320193",
    known_at: str = "2026-08-25T00:00:00Z",
    source: str = "sec",
    valid_from: str | None = "2026-01-01",
    valid_to: str | None = None,
) -> TickerAlias:
    """Fixture WING alias; PIT stays in the resolver, never in the stub."""
    return TickerAlias(
        alias_type="ticker",
        alias_value="WING",
        entity_id=entity_id,
        security_id=security_id,
        source=source,
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known_at,
        retrieved_at=known_at,
    )


def _run_sync(
    data_root: Path, payloads: dict[str, object] | None = None
) -> tuple[FakeClient, RobinhoodPortfolioProvider, PortfolioSnapshot]:
    client = FakeClient(payloads or _happy_payloads())
    provider = RobinhoodPortfolioProvider(client)
    snapshot = sync_robinhood_portfolio(provider, data_root=data_root, now=NOW)
    return client, provider, snapshot


@pytest.fixture(autouse=True)
def _wing_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live-alias double: WING resolves, every other ticker is unknown."""
    wing = _wing_alias()

    def _candidates(self: object, ticker: str, as_of: datetime) -> list[TickerAlias]:
        del self, as_of
        return [wing] if ticker.strip().upper() == "WING" else []

    monkeypatch.setattr(SourceGateway, "ticker_candidates", _candidates)


def _stub_candidates(monkeypatch: pytest.MonkeyPatch, aliases: list[TickerAlias]) -> None:
    """Point the gateway at an explicit alias list (PIT stays in the resolver)."""

    def _candidates(self: object, ticker: str, as_of: datetime) -> list[TickerAlias]:
        del self, as_of
        want = ticker.strip().upper()
        return [alias for alias in aliases if alias.alias_value == want]

    monkeypatch.setattr(SourceGateway, "ticker_candidates", _candidates)


def test_sync_uses_exact_readonly_call_sequence(data_root: Path) -> None:
    client, _, snapshot = _run_sync(data_root)
    assert client.calls == [
        ("get_accounts", {}),
        ("get_equity_positions", {"account_number": "100000001"}),
        ("get_portfolio", {"account_number": "100000001"}),
        ("get_equity_positions", {"account_number": "100000002"}),
        ("get_portfolio", {"account_number": "100000002"}),
        ("get_equity_quotes", {"symbols": ["AAPL", "TSLA", "WING", "MSFT"]}),
    ]
    assert snapshot.snapshot_id == SNAPSHOT_ID


def test_sync_builds_valued_snapshot_with_two_accounts(data_root: Path) -> None:
    _, _, snapshot = _run_sync(data_root)
    assert isinstance(snapshot, PortfolioSnapshot)
    assert snapshot.broker == "robinhood"
    assert snapshot.account_ids == (local_account_id("100000001"), local_account_id("100000002"))
    assert snapshot.created_at == NOW
    assert snapshot.cash == Decimal("3234.56")
    assert snapshot.invested_value == Decimal("3390.24")
    assert snapshot.total_value == Decimal("6624.80")
    assert len(snapshot.positions) == 5
    by_id = {position.position_id: position for position in snapshot.positions}
    wing = by_id[f"{SNAPSHOT_ID}:{local_account_id('100000001')}:WING"]
    assert wing.market_value == Decimal("1168.40")
    assert wing.market_price == Decimal("116.84")
    assert wing.quantity == Decimal(10)
    assert wing.unrealized_gain == Decimal("213.40")
    assert wing.price_type == "last"
    assert wing.quote_retrieved_at == QUOTE_TIME
    assert wing.entity_id == "sec:cik:0000320193"
    assert wing.security_id == "sec:equity:0000320193"
    assert wing.source == "robinhood_mcp"
    aapl = by_id[f"{SNAPSHOT_ID}:{local_account_id('100000001')}:AAPL"]
    assert aapl.market_value == Decimal("80.00")
    msft = by_id[f"{SNAPSHOT_ID}:{local_account_id('100000002')}:MSFT"]
    assert msft.market_value == Decimal("2025.00")
    assert msft.account_id == local_account_id("100000002")
    weights = [position.portfolio_weight for position in snapshot.positions]
    assert all(weight is not None for weight in weights)
    assert snapshot.total_value is not None
    assert snapshot.invested_value is not None
    known_weights = [w for w in weights if w is not None]
    assert sum(known_weights, Decimal(0)) == pytest.approx(snapshot.invested_value / snapshot.total_value)


def test_local_account_id_is_opaque_and_deterministic() -> None:
    assert local_account_id("100000001") == local_account_id("100000001")
    assert local_account_id("100000001") == "local:91b1e23ddbc8bddc"
    assert "100000001" not in local_account_id("100000001")
    assert local_account_id("100000001") != local_account_id("100000002")


def test_sync_raises_on_malformed_quantity(data_root: Path) -> None:
    payloads = _happy_payloads()
    payloads["get_equity_positions"] = _fixture("positions.json")
    client = FakeClient(payloads)
    provider = RobinhoodPortfolioProvider(client)
    with pytest.raises(ValueError, match="quantity"):
        sync_robinhood_portfolio(provider, data_root=data_root, now=NOW)


def test_zero_quantity_position_is_valued_at_zero(data_root: Path) -> None:
    _, _, snapshot = _run_sync(data_root)
    tsla = next(position for position in snapshot.positions if position.ticker == "TSLA")
    assert tsla.quantity == Decimal(0)
    assert tsla.market_value == Decimal(0)
    assert tsla.portfolio_weight == Decimal(0)


def test_snapshot_and_positions_are_persisted_once(data_root: Path) -> None:
    _, _, snapshot = _run_sync(data_root)
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.snapshot_id == SNAPSHOT_ID
    assert len(restored.positions) == 5
    assert [position.price_type for position in restored.positions] == ["last"] * 5
    assert restored.total_value == Decimal("6624.80")
    assert sum(1 for position in restored.positions if position.market_value is not None) == 5
    assert sum(1 for position in restored.positions if position.entity_id is None) == 3
    persist_snapshot(snapshot, data_root=data_root)
    assert read_latest_snapshot(data_root=data_root) == restored


def _assert_position_close(actual: Position, expected: Position) -> None:
    assert actual.position_id == expected.position_id
    assert actual.account_id == expected.account_id
    assert actual.security_id == expected.security_id
    assert actual.entity_id == expected.entity_id
    assert actual.ticker == expected.ticker
    assert actual.source == expected.source
    assert actual.price_type == expected.price_type
    assert actual.quote_retrieved_at == expected.quote_retrieved_at
    for field in (
        "quantity",
        "average_cost",
        "market_price",
        "market_value",
        "unrealized_gain",
    ):
        # Money and quantity are stored at their exact scale (quantity *
        # price products fit decimal128(38, 14)); restored values equal
        # the live Decimals exactly.
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("unrealized_gain_pct", "portfolio_weight"):
        # Ratios are computed at Python's default context precision (28
        # significant digits) and can carry more fractional digits than
        # the (38, 28) ratio columns hold; the write boundary rounds the
        # overflow, so compare approximately.
        assert getattr(actual, field) == pytest.approx(getattr(expected, field))


def test_read_latest_snapshot_round_trips(data_root: Path) -> None:
    snapshot = _run_sync(data_root)[2]
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.created_at == snapshot.created_at
    assert restored.broker == "robinhood"
    assert restored.account_ids == (local_account_id("100000001"), local_account_id("100000002"))
    assert restored.cash == snapshot.cash
    assert restored.invested_value == snapshot.invested_value
    assert restored.total_value == snapshot.total_value
    assert [position.position_id for position in restored.positions] == [
        position.position_id for position in snapshot.positions
    ]
    for actual, expected in zip(restored.positions, snapshot.positions):
        _assert_position_close(actual, expected)
        assert actual.retrieved_at == restored.created_at


def test_read_latest_snapshot_none_when_empty(data_root: Path) -> None:
    assert read_latest_snapshot(data_root=data_root) is None


def test_cash_only_account_survives_round_trip(data_root: Path) -> None:
    # Positions only in account A; account B holds cash only.  Before
    # portfolio_accounts, the read side derived account ids from position
    # rows and dropped B.
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={
            local_account_id("100000001"): Decimal(1000),
            local_account_id("100000002"): Decimal(2000),
        },
        created_at=NOW,
    )
    persist_snapshot(snapshot, data_root=data_root)
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.account_ids == (
        local_account_id("100000001"),
        local_account_id("100000002"),
    )


def test_empty_portfolio_with_cash_round_trips_account_ids(data_root: Path) -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001")],
        positions=[],
        cash_balances={local_account_id("100000001"): Decimal(5000)},
        created_at=NOW,
    )
    persist_snapshot(snapshot, data_root=data_root)
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.account_ids == (local_account_id("100000001"),)
    assert restored.positions == ()


def test_account_ids_round_trip_preserves_order(data_root: Path) -> None:
    account_a = local_account_id("100000001")
    account_b = local_account_id("100000002")
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[account_a, account_b],
        positions=[_position(account_id=account_a), _position(account_id=account_b)],
        cash_balances={account_a: Decimal(1000), account_b: Decimal(2000)},
        created_at=NOW,
    )
    persist_snapshot(snapshot, data_root=data_root)
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.account_ids == snapshot.account_ids


def test_decimal_round_trip_is_exact(data_root: Path) -> None:
    # Values chosen to stress column scales: a small fractional quantity
    # whose price product needs all 14 fractional digits, a large dollar
    # amount, a repeating-fraction percentage, and non-integral decimals
    # whose storage-scale artifacts (trailing zeros) must canonicalize
    # away on read (0.5 -> not 0.5000...).
    position = Position(
        position_id="pos-1",
        account_id=local_account_id("100000001"),
        security_id="sec:equity:0000320193",
        entity_id="sec:cik:0000320193",
        ticker="AMD",
        quantity=Decimal("0.00000001"),
        average_cost=Decimal("123456789012.34"),
        market_price=Decimal("123456789012.123456"),
        market_value=Decimal("1234.56789012123456"),
        unrealized_gain=Decimal("0.1"),
        unrealized_gain_pct=Decimal("0.3333333333"),
        portfolio_weight=None,
        source="robinhood_mcp",
        retrieved_at=NOW,
    )
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001")],
        positions=[position],
        cash_balances={local_account_id("100000001"): Decimal("1234.56789012123456")},
        created_at=NOW,
    )
    persist_snapshot(snapshot, data_root=data_root)
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.cash == snapshot.cash
    assert restored.invested_value == snapshot.invested_value
    assert restored.total_value == snapshot.total_value
    (actual,) = restored.positions
    (expected,) = snapshot.positions
    for field in (
        "quantity",
        "average_cost",
        "market_price",
        "market_value",
        "unrealized_gain",
        "unrealized_gain_pct",
        "portfolio_weight",
    ):
        assert getattr(actual, field) == getattr(expected, field)


def test_missing_quote_price_degrades_snapshot_but_persists(data_root: Path) -> None:
    payloads = _happy_payloads()

    def _wing_only(args: dict[str, object]) -> dict[str, object]:
        account = args["account_number"]
        assert isinstance(account, str)
        if account == "100000001":
            return {
                "data": {
                    "positions": [
                        {
                            "id": "pos-1",
                            "instrument_id": "instr-wing",
                            "symbol": "WING",
                            "quantity": "10",
                            "average_buy_price": "95.50",
                        }
                    ]
                }
            }
        empty: list[dict[str, object]] = []
        return {"data": {"positions": empty}}

    payloads["get_equity_positions"] = _wing_only
    payloads["get_equity_quotes"] = {
        "data": {"results": [{"quote": {"symbol": "WING", "venue_last_trade_time": "2026-08-25T15:00:00Z"}}]}
    }
    _, _, snapshot = _run_sync(data_root, payloads)
    wing = snapshot.positions[0]
    assert wing.market_price is None
    assert wing.market_value is None
    assert wing.unrealized_gain is None
    assert wing.portfolio_weight is None
    assert wing.price_type is None
    assert wing.quote_retrieved_at == QUOTE_TIME
    assert snapshot.invested_value is None
    assert snapshot.total_value is None
    assert snapshot.cash == Decimal("3234.56")
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert len(restored.positions) == 1
    assert sum(1 for position in restored.positions if position.market_value is not None) == 0
    assert restored.total_value is None
    assert restored.positions[0].market_price is None
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.invested_value is None
    assert restored.positions[0].market_price is None


def test_partial_pricing_nils_total_and_weights(data_root: Path) -> None:
    payloads = _happy_payloads()

    def _two_accounts(args: dict[str, object]) -> dict[str, object]:
        account = args["account_number"]
        assert isinstance(account, str)
        if account == "100000001":
            return {
                "data": {
                    "positions": [
                        {
                            "id": "pos-1",
                            "instrument_id": "instr-wing",
                            "symbol": "WING",
                            "quantity": "10",
                            "average_buy_price": "95.50",
                        }
                    ]
                }
            }
        return {
            "data": {
                "positions": [
                    {
                        "id": "pos-5",
                        "instrument_id": "instr-aapl",
                        "symbol": "AAPL",
                        "quantity": "5",
                        "average_buy_price": "150.25",
                    }
                ]
            }
        }

    payloads["get_equity_positions"] = _two_accounts
    payloads["get_equity_quotes"] = {
        "data": {
            "results": [
                {
                    "quote": {
                        "symbol": "WING",
                        "last_trade_price": "116.84",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                }
            ]
        }
    }
    _, _, snapshot = _run_sync(data_root, payloads)
    assert snapshot.invested_value == Decimal("1168.40")
    assert snapshot.total_value is None
    assert snapshot.cash == Decimal("3234.56")
    assert all(position.portfolio_weight is None for position in snapshot.positions)


def test_zero_quantity_unpriced_does_not_block_completeness(data_root: Path) -> None:
    payloads = _happy_payloads()

    def _zero_wing(args: dict[str, object]) -> dict[str, object]:
        account = args["account_number"]
        assert isinstance(account, str)
        if account == "100000001":
            return {
                "data": {
                    "positions": [
                        {
                            "id": "pos-1",
                            "instrument_id": "instr-wing",
                            "symbol": "WING",
                            "quantity": "0",
                            "average_buy_price": "95.50",
                        }
                    ]
                }
            }
        return {
            "data": {
                "positions": [
                    {
                        "id": "pos-5",
                        "instrument_id": "instr-aapl",
                        "symbol": "AAPL",
                        "quantity": "5",
                        "average_buy_price": "150.25",
                    }
                ]
            }
        }

    payloads["get_equity_positions"] = _zero_wing
    payloads["get_equity_quotes"] = {
        "data": {
            "results": [
                {
                    "quote": {
                        "symbol": "AAPL",
                        "last_trade_price": "160.00",
                        "venue_last_trade_time": "2026-08-25T15:00:00Z",
                    }
                }
            ]
        }
    }
    _, _, snapshot = _run_sync(data_root, payloads)
    assert snapshot.invested_value == Decimal("800.00")
    assert snapshot.total_value is not None
    aapl = next(position for position in snapshot.positions if position.ticker == "AAPL")
    wing = next(position for position in snapshot.positions if position.ticker == "WING")
    assert aapl.portfolio_weight is not None
    assert wing.market_value == Decimal(0)
    assert wing.portfolio_weight == Decimal(0)


def test_empty_accounts_still_persists_empty_snapshot(data_root: Path) -> None:
    client = FakeClient({"get_accounts": {"accounts": []}})
    provider = RobinhoodPortfolioProvider(client)
    snapshot = sync_robinhood_portfolio(provider, data_root=data_root, now=NOW)
    assert snapshot.positions == ()
    assert snapshot.cash is None
    assert snapshot.invested_value is None
    assert snapshot.total_value is None
    assert snapshot.account_ids == ()
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    assert restored.snapshot_id == SNAPSHOT_ID
    assert restored.positions == ()
    assert restored.account_ids == ()
    assert snapshot.snapshot_id == SNAPSHOT_ID


def test_persisted_rows_contain_no_oauth_data(data_root: Path) -> None:
    _, _, snapshot = _run_sync(data_root)
    forbidden = ("token", "oauth", "secret", "access", "refresh", "authorization")

    def _texts(node: object) -> list[str]:
        if isinstance(node, dict):
            return [str(key) for key in node] + [text for value in node.values() for text in _texts(value)]
        if isinstance(node, (list, tuple)):
            return [text for item in node for text in _texts(item)]
        return [] if node is None else [str(node)]

    cells = _texts(dataclasses.asdict(snapshot))
    assert cells
    for text in cells:
        assert not any(part in text.lower() for part in forbidden), (
            f"persisted snapshot carries forbidden data: {text!r}"
        )


def test_persisted_rows_never_contain_raw_account_ids(data_root: Path) -> None:
    _, _, snapshot = _run_sync(data_root)
    assert snapshot.account_ids == (local_account_id("100000001"), local_account_id("100000002"))
    restored = read_latest_snapshot(data_root=data_root)
    assert restored is not None
    for position in restored.positions:
        assert "100000001" not in position.account_id and "100000002" not in position.account_id
    assert [position.account_id for position in restored.positions] == [
        local_account_id("100000001"),
        local_account_id("100000001"),
        local_account_id("100000001"),
        local_account_id("100000002"),
        local_account_id("100000002"),
    ]


def test_sync_without_explicit_now_uses_utc_now(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FrozenClock(datetime):
        """datetime subclass freezing now(); isinstance checks still pass."""

        calls = 0

        @classmethod
        @override
        def now(cls, tz: tzinfo | None = None) -> FrozenClock:
            FrozenClock.calls += 1
            return FrozenClock(2026, 8, 25, 12, 0, tzinfo=UTC)

        @classmethod
        @override
        def fromisoformat(cls, date_string: str, /) -> FrozenClock:
            parsed = datetime.fromisoformat(date_string)
            return cls(
                parsed.year,
                parsed.month,
                parsed.day,
                parsed.hour,
                parsed.minute,
                parsed.second,
                parsed.microsecond,
                tzinfo=parsed.tzinfo,
            )

    from app.services import portfolio_sync

    monkeypatch.setattr(portfolio_sync, "datetime", FrozenClock)
    client = FakeClient(_happy_payloads())
    snapshot = sync_robinhood_portfolio(RobinhoodPortfolioProvider(client), data_root=data_root)
    assert snapshot.created_at == NOW
    assert snapshot.snapshot_id == SNAPSHOT_ID


# ---------------------------------------------------------------------------
# resolve_security / build_position
# ---------------------------------------------------------------------------


def test_resolve_security_alias_learned_after_as_of_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    _stub_candidates(monkeypatch, [_wing_alias(known_at="2026-08-26T00:00:00Z")])
    late = resolve_security("WING", as_of=as_of)
    assert late.resolved is False
    assert late.resolution_method == "unresolved"
    _stub_candidates(
        monkeypatch,
        [
            _wing_alias(known_at="2026-08-26T00:00:00Z"),
            _wing_alias(known_at="2026-08-25T00:00:00Z", source="control"),
        ],
    )
    known = resolve_security("WING", as_of=as_of)
    assert known.resolved is True
    assert known.resolution_method == "entity_alias"
    assert known.entity_id == "sec:cik:0000320193"


def test_resolve_security_expired_alias_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_candidates(
        monkeypatch,
        [
            _wing_alias(valid_to="2026-08-24", known_at="2026-08-01T00:00:00Z"),
            _wing_alias(valid_to="2026-08-25", known_at="2026-08-02T00:00:00Z", source="control"),
        ],
    )
    as_of = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    expired = resolve_security("WING", as_of=as_of)
    assert expired.resolved is False
    assert expired.resolution_method == "unresolved"
    # Half-open boundary: a date-only valid_to is midnight, so
    # valid_to="2026-08-25" is already expired at 00:00 on the 25th.
    boundary = resolve_security("WING", as_of=datetime(2026, 8, 25, 0, 0, tzinfo=UTC))
    assert boundary.resolved is False
    assert boundary.resolution_method == "unresolved"


def test_resolve_security_ambiguous_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_candidates(
        monkeypatch,
        [
            _wing_alias(entity_id="sec:cik:0000320193", security_id="sec:equity:0000320193"),
            _wing_alias(
                entity_id="sec:cik:0000999999",
                security_id="sec:equity:0000999999",
                source="control",
            ),
        ],
    )
    resolution = resolve_security("WING", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is False
    assert resolution.resolution_method == "ambiguous"
    assert resolution.entity_id is None
    assert resolution.security_id is None


def test_resolve_security_same_entity_multiple_rows_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_candidates(
        monkeypatch,
        [
            _wing_alias(
                security_id=None,
                known_at="2026-08-01T00:00:00Z",
            ),
            _wing_alias(
                security_id="sec:equity:0000320193",
                known_at="2026-08-02T00:00:00Z",
                source="control",
            ),
        ],
    )
    resolution = resolve_security("WING", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.resolution_method == "entity_alias"
    assert resolution.entity_id == "sec:cik:0000320193"
    assert resolution.security_id == "sec:equity:0000320193"


def test_build_position_passes_asset_type() -> None:
    raw = BrokeragePositionInput(
        position_id="pos-1",
        account_id="acc-1",
        ticker="WING",
        quantity=Decimal(10),
        average_cost=Decimal("95.50"),
        retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=UTC),
        source="robinhood_mcp",
        asset_type="option",
    )
    position = build_position(
        raw,
        SecurityResolution(None, None, "WING", False, "unresolved"),
        None,
    )
    assert position.asset_type == "option"


def _position(
    market_value: Decimal | None = Decimal(500),
    *,
    account_id: str = local_account_id("100000001"),
) -> Position:
    return Position(
        position_id="pos-1",
        account_id=account_id,
        security_id="sec:equity:0000320193",
        entity_id="sec:cik:0000320193",
        ticker="AMD",
        quantity=Decimal(1),
        average_cost=Decimal(400),
        market_price=market_value,
        market_value=market_value,
        unrealized_gain=Decimal(100) if market_value is not None else None,
        unrealized_gain_pct=Decimal("0.25") if market_value is not None else None,
        portfolio_weight=None,
        source="robinhood_mcp",
        retrieved_at=NOW,
    )


def test_cash_complete_multi_account_sums() -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={local_account_id("100000001"): Decimal(1000), local_account_id("100000002"): Decimal(2000)},
        created_at=NOW,
    )
    assert snapshot.cash == Decimal(3000)
    assert snapshot.invested_value == Decimal(500)
    assert snapshot.total_value == Decimal(3500)


def test_partial_cash_nils_total_and_weights() -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={local_account_id("100000001"): Decimal(1000), local_account_id("100000002"): None},
        created_at=NOW,
    )
    assert snapshot.cash is None
    assert snapshot.total_value is None
    assert all(position.portfolio_weight is None for position in snapshot.positions)


def test_missing_balance_nils_total() -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={local_account_id("100000001"): Decimal(1000)},
        created_at=NOW,
    )
    assert snapshot.cash is None
    assert snapshot.total_value is None


def test_duplicate_balance_for_one_account_incomplete() -> None:
    # The mapping signature cannot represent duplicate balances (the
    # provider yields one CashBalance per account), so this collapses to a
    # missing-balance case and stays incomplete.
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={local_account_id("100000001"): Decimal(1000), local_account_id("100000001"): Decimal(2000)},
        created_at=NOW,
    )
    assert snapshot.cash is None
    assert snapshot.total_value is None


def test_mismatched_balance_account_ids_incomplete() -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001"), local_account_id("100000002")],
        positions=[_position()],
        cash_balances={local_account_id("100000001"): Decimal(1000), local_account_id("100000003"): Decimal(2000)},
        created_at=NOW,
    )
    assert snapshot.cash is None
    assert snapshot.total_value is None


def test_cash_only_portfolio_has_valid_totals() -> None:
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id("100000001")],
        positions=[],
        cash_balances={local_account_id("100000001"): Decimal(5000)},
        created_at=NOW,
    )
    assert snapshot.invested_value == Decimal(0)
    assert snapshot.cash == Decimal(5000)
    assert snapshot.total_value == Decimal(5000)
    assert snapshot.positions == ()


def test_snapshot_builder_is_provider_neutral() -> None:
    snapshot = build_portfolio_snapshot(
        broker="testbroker",
        account_ids=["acct"],
        positions=[],
        cash_balances={"acct": Decimal(100)},
        created_at=NOW,
    )
    assert snapshot.broker == "testbroker"
    assert snapshot.snapshot_id == "portfolio:testbroker:2026-08-25T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Provider-level extraction behavior
# ---------------------------------------------------------------------------


class TestProvider:
    def test_get_accounts_accepts_bare_list(self):
        client = FakeClient({"get_accounts": [{"id": "acc-1", "type": "individual"}]})
        accounts = RobinhoodPortfolioProvider(client).get_accounts()
        assert [account.account_id for account in accounts] == ["acc-1"]

    def test_get_accounts_wraps_bare_object(self):
        client = FakeClient({"get_accounts": {"id": "acc-1", "type": "individual"}})
        accounts = RobinhoodPortfolioProvider(client).get_accounts()
        assert [account.account_id for account in accounts] == ["acc-1"]

    def test_get_accounts_unknown_shape_raises(self):
        client = FakeClient({"get_accounts": "unexpected"})
        with pytest.raises(ValueError, match="get_accounts"):
            RobinhoodPortfolioProvider(client).get_accounts()

    def test_get_accounts_unwraps_structured_content_envelope(self):
        client = FakeClient({"get_accounts": _fixture("accounts.json")})
        accounts = RobinhoodPortfolioProvider(client).get_accounts()
        assert [account.account_id for account in accounts] == ["100000001", "100000002"]
        assert {account.account_type for account in accounts} == {"individual"}

    def test_get_accounts_falls_back_to_content_text_json(self):
        payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "data": {
                                "accounts": [
                                    {
                                        "account_number": "c-1",
                                        "type": "margin",
                                        "brokerage_account_type": "individual",
                                        "state": "active",
                                    }
                                ]
                            }
                        }
                    ),
                }
            ]
        }
        client = FakeClient({"get_accounts": payload})
        accounts = RobinhoodPortfolioProvider(client).get_accounts()
        assert [account.account_id for account in accounts] == ["c-1"]

    def test_get_positions_normalizes_rows(self):
        client = FakeClient(
            {
                "get_equity_positions": {
                    "positions": [
                        {"id": "p", "account_id": "acc-1", "ticker": "wing", "quantity": "0.5"},
                    ]
                }
            }
        )
        positions = RobinhoodPortfolioProvider(client).get_positions("acc-1")
        assert positions[0].ticker == "WING"
        assert positions[0].account_id == "acc-1"
        assert positions[0].quantity == Decimal("0.5")
        assert client.calls == [("get_equity_positions", {"account_number": "acc-1"})]

    def test_get_cash_balance_accepts_bare_object(self):
        client = FakeClient({"get_portfolio": {"account_id": "acc-1", "cash": "100.00"}})
        balance = RobinhoodPortfolioProvider(client).get_cash_balance("acc-1")
        assert balance.cash == Decimal("100.00")
        assert balance.account_id == "acc-1"

    def test_get_cash_balance_unwraps_nested_buying_power(self):
        client = FakeClient({"get_portfolio": {"data": {"cash": "100.00", "buying_power": {"buying_power": "250.00"}}}})
        balance = RobinhoodPortfolioProvider(client).get_cash_balance("acc-1")
        assert balance.cash == Decimal("100.00")
        assert balance.buying_power == Decimal("250.00")
        assert balance.account_id == "acc-1"
        assert client.calls == [("get_portfolio", {"account_number": "acc-1"})]

    def test_get_equity_quotes_merges_aliases_and_skips_missing_ticker(self):
        client = FakeClient(
            {
                "get_equity_quotes": {
                    "results": [
                        {"symbol": "wing", "last": "116.84", "bid_price": "116.83", "askPrice": "116.85"},
                        {"ticker": "AAPL", "price": "160.00"},
                        {"last": "10.00"},
                    ]
                }
            }
        )
        quotes = RobinhoodPortfolioProvider(client).get_equity_quotes(["wing", "AAPL"])
        assert list(quotes) == ["WING", "AAPL"]
        assert quotes["WING"].last == Decimal("116.84")
        assert quotes["WING"].bid == Decimal("116.83")
        assert quotes["WING"].ask == Decimal("116.85")
        assert quotes["AAPL"].last == Decimal("160.00")
        assert client.calls == [("get_equity_quotes", {"symbols": ["WING", "AAPL"]})]

    def test_get_equity_quotes_digs_into_result_quote_objects(self):
        client = FakeClient(
            {
                "get_equity_quotes": {
                    "data": {
                        "results": [
                            {
                                "quote": {
                                    "symbol": "WING",
                                    "last_trade_price": "116.84",
                                    "bid_price": "116.83",
                                    "ask_price": "116.85",
                                    "venue_last_trade_time": "2026-08-25T15:00:00Z",
                                }
                            },
                        ]
                    }
                }
            }
        )
        quotes = RobinhoodPortfolioProvider(client).get_equity_quotes(["WING"])
        assert quotes["WING"].last == Decimal("116.84")
        assert quotes["WING"].bid == Decimal("116.83")
        assert quotes["WING"].ask == Decimal("116.85")
        assert quotes["WING"].retrieved_at == QUOTE_TIME

    def test_get_equity_quotes_accepts_bare_list(self):
        client = FakeClient(
            {
                "get_equity_quotes": [
                    {"symbol": "WING", "last": "116.84"},
                ]
            }
        )
        quotes = RobinhoodPortfolioProvider(client).get_equity_quotes(["WING"])
        assert quotes["WING"].last == Decimal("116.84")

    def test_get_equity_quotes_empty_tickers_skips_call(self):
        client = FakeClient({})
        quotes = RobinhoodPortfolioProvider(client).get_equity_quotes([])
        assert quotes == {}
        assert client.calls == []


class TestScanProvider:
    def test_get_scanner_filter_specs_unwraps_envelope(self):
        client = FakeClient({"get_scanner_filter_specs": _fixture("scan_specs.json")})
        data = RobinhoodPortfolioProvider(client).get_scanner_filter_specs()
        assert client.calls == [("get_scanner_filter_specs", {})]
        filter_specs = data["filter_specs"]
        assert isinstance(filter_specs, list)
        first_spec = filter_specs[0]
        assert isinstance(first_spec, dict)
        assert first_spec["filter_type"] == "FILTER_TYPE_INSTRUMENT_TYPE"

    def test_get_scans_returns_rows(self):
        client = FakeClient({"get_scans": _fixture("scans.json")})
        scans = RobinhoodPortfolioProvider(client).get_scans()
        assert client.calls == [("get_scans", {})]
        assert [scan["id"] for scan in scans] == ["scan-rsi-1", "scan-cortex-2"]
        assert scans[1]["cortex_managed"] is True

    def test_run_scan_passes_scan_id(self):
        client = FakeClient({"run_scan": _fixture("scan_results.json")})
        data = RobinhoodPortfolioProvider(client).run_scan("scan-rsi-1")
        assert client.calls == [("run_scan", {"scan_id": "scan-rsi-1"})]
        assert data["total"] == 3
        results = data["results"]
        assert isinstance(results, list)
        first_result = results[0]
        assert isinstance(first_result, dict)
        assert first_result["ticker"] == "WING"
