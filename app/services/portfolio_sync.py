"""Portfolio synchronization service.

Resolves positions to Stockbot identities, values them against Robinhood
quotes, builds immutable portfolio snapshots, and persists/reads them from
the user-private portfolio.sqlite ops store. Never exposes provider internals.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ..config import get_data_root
from ..domain.market.identity import resolve_ticker_aliases
from ..domain.market.securities import SecurityResolution
from ..domain.portfolio import PortfolioSnapshot, Position
from ..domain.portfolio.snapshot import build_portfolio_snapshot
from ..domain.portfolio.valuation import build_position
from ..robinhood.account import BrokeragePosition, CashBalance
from ..robinhood.adapters import to_position_input, to_quote
from ..robinhood.portfolio import RobinhoodPortfolioProvider
from .account_identity import local_account_id

SNAPSHOT_SOURCE = "robinhood_mcp"
PARSER_VERSION = "robinhood-mcp-account-v1"
CALCULATION_VERSION = "portfolio-snapshot-v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  snapshot_id TEXT PRIMARY KEY, broker TEXT NOT NULL, created_at TEXT NOT NULL,
  cash TEXT, invested_value TEXT, total_value TEXT,
  account_count INTEGER NOT NULL, position_count INTEGER NOT NULL,
  priced_position_count INTEGER NOT NULL, unresolved_position_count INTEGER NOT NULL,
  source TEXT NOT NULL, parser_version TEXT NOT NULL, calculation_version TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_snapshots_created ON portfolio_snapshots(created_at);
CREATE TABLE IF NOT EXISTS portfolio_positions (
  position_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, account_id TEXT NOT NULL,
  security_id TEXT, entity_id TEXT, ticker TEXT NOT NULL, quantity TEXT NOT NULL,
  average_cost TEXT, market_price TEXT, price_type TEXT, market_value TEXT,
  unrealized_gain TEXT, unrealized_gain_pct TEXT, portfolio_weight TEXT,
  source TEXT NOT NULL, quote_retrieved_at TEXT, asset_type TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_positions_snapshot ON portfolio_positions(snapshot_id);
CREATE TABLE IF NOT EXISTS portfolio_accounts (
  snapshot_id TEXT NOT NULL, account_id TEXT NOT NULL,
  PRIMARY KEY (snapshot_id, account_id));
"""


def get_portfolio_db_path(data_root: Path) -> Path:
    """User-private ops store path for a data root."""
    return data_root / "portfolio.sqlite"


def _db_path(data_root: Path | None) -> Path:
    root = Path(data_root) if data_root else get_data_root()
    return get_portfolio_db_path(root)


def _connect(data_root: Path | None) -> sqlite3.Connection:
    path = _db_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def resolve_security(
    ticker: str,
    *,
    provider_instrument_id: str | None = None,
    as_of: datetime | None = None,
    data_root: Path | None = None,
) -> SecurityResolution:
    """Resolve a ticker to a Stockbot security/entity identity.

    ``as_of`` must be a timezone-aware datetime; ``date`` and naive values
    are rejected at runtime (TypeError/ValueError), and aware values are
    normalized to UTC (timestamp precision).
    ``provider_instrument_id`` is preserved on the position but is NOT used
    for resolution: Robinhood instrument IDs are not bridged to Stockbot
    identities yet.  Resolution is point-in-time over live company-tickers
    aliases via the SourceGateway seam (PIT applied by
    ``resolve_ticker_aliases``).  Multiple active entities for one ticker
    resolve as ``"ambiguous"`` — never source/order-wins.  No mappings are
    ever invented: unknown tickers resolve to ``resolved=False``.
    """
    del provider_instrument_id
    del data_root
    if as_of is None:
        as_of = datetime.now(UTC)
    elif not isinstance(as_of, datetime):
        raise TypeError(f"as_of must be a timezone-aware datetime, got {type(as_of).__name__}")
    elif as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    from ..data_sources import SourceGateway

    aliases = SourceGateway().ticker_candidates(ticker, as_of)
    return resolve_ticker_aliases(ticker, aliases, as_of=as_of)


def _canonical_decimal(value: Decimal | None) -> Decimal | None:
    """Strip sqlite TEXT round-trip scale artifacts (value-preserving)."""
    if value is None:
        return None
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return normalized.quantize(Decimal(1))
    return normalized


def _row_quantity(value: Decimal | None) -> Decimal:
    """Fail closed on a persisted position without a quantity (never default money)."""
    if value is None:
        raise ValueError("Position row is missing or has a malformed quantity")
    return value


def _position_from_row(row: Mapping[str, object], retrieved_at: datetime) -> Position:
    """Rebuild a position from a persisted sqlite row (retrieved_at = snapshot created_at)."""
    numeric = {
        key: _canonical_decimal(Decimal(str(row[key])) if row[key] is not None else None)
        for key in (
            "quantity",
            "average_cost",
            "market_price",
            "market_value",
            "unrealized_gain",
            "unrealized_gain_pct",
            "portfolio_weight",
        )
    }
    quote_retrieved_at = row.get("quote_retrieved_at")
    asset_type_value = row.get("asset_type")
    asset_type = asset_type_value if isinstance(asset_type_value, str) and asset_type_value else "equity"
    return Position(
        position_id=str(row["position_id"]),
        account_id=str(row["account_id"]),
        security_id=str(row["security_id"]) if row.get("security_id") else None,
        entity_id=str(row["entity_id"]) if row.get("entity_id") else None,
        ticker=str(row["ticker"]),
        quantity=_row_quantity(numeric["quantity"]),
        average_cost=numeric["average_cost"],
        market_price=numeric["market_price"],
        market_value=numeric["market_value"],
        unrealized_gain=numeric["unrealized_gain"],
        unrealized_gain_pct=numeric["unrealized_gain_pct"],
        portfolio_weight=numeric["portfolio_weight"],
        source=str(row["source"]),
        retrieved_at=retrieved_at,
        price_type=str(row["price_type"]) if row.get("price_type") else None,
        quote_retrieved_at=(
            datetime.fromisoformat(quote_retrieved_at)
            if isinstance(quote_retrieved_at, str) and quote_retrieved_at
            else None
        ),
        asset_type=asset_type,
    )


def _ratio_value(value: Decimal | None) -> Decimal | None:
    """Fit a computed ratio to 28 fractional digits.

    Ratios are divided at Python's default context precision (28
    significant digits), so a ratio below 0.1 can carry 29+ fractional
    digits.  Rounding those beyond the 28th digit keeps a small position
    from failing the whole write; the discarded digits are far past any
    meaningful precision.  Money and quantity columns are never rounded
    here: a value that exceeds their scale is a data error and must fail
    loudly.
    """
    if value is None:
        return None
    return value.quantize(Decimal("1e-28"))


def _money(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def persist_snapshot(snapshot: PortfolioSnapshot, *, data_root: Path | None = None) -> None:
    """Persist an immutable snapshot (idempotent: a rerun writes 0 rows).

    Never writes OAuth/token or raw provider payload data or raw broker account identifiers.
    Research bundles reference the snapshot by ``snapshot_id`` plus content
    hash only, never full snapshot rows.
    """
    conn = _connect(data_root)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO portfolio_snapshots "
            "(snapshot_id, broker, created_at, cash, invested_value, total_value, "
            "account_count, position_count, priced_position_count, unresolved_position_count, "
            "source, parser_version, calculation_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_id,
                snapshot.broker,
                snapshot.created_at.isoformat(),
                _money(snapshot.cash),
                _money(snapshot.invested_value),
                _money(snapshot.total_value),
                len(snapshot.account_ids),
                len(snapshot.positions),
                sum(1 for position in snapshot.positions if position.market_value is not None),
                sum(1 for position in snapshot.positions if position.entity_id is None),
                SNAPSHOT_SOURCE,
                PARSER_VERSION,
                CALCULATION_VERSION,
            ),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO portfolio_positions "
            "(position_id, snapshot_id, account_id, security_id, entity_id, ticker, quantity, "
            "average_cost, market_price, price_type, market_value, unrealized_gain, "
            "unrealized_gain_pct, portfolio_weight, source, quote_retrieved_at, asset_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    position.position_id,
                    snapshot.snapshot_id,
                    position.account_id,
                    position.security_id,
                    position.entity_id,
                    position.ticker,
                    _money(position.quantity),
                    _money(position.average_cost),
                    _money(position.market_price),
                    position.price_type,
                    _money(position.market_value),
                    _money(position.unrealized_gain),
                    _money(_ratio_value(position.unrealized_gain_pct)),
                    _money(_ratio_value(position.portfolio_weight)),
                    position.source,
                    position.quote_retrieved_at.isoformat() if position.quote_retrieved_at else None,
                    position.asset_type,
                )
                for position in snapshot.positions
            ),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO portfolio_accounts (snapshot_id, account_id) VALUES (?, ?)",
            [(snapshot.snapshot_id, account_id) for account_id in snapshot.account_ids],
        )
        conn.commit()
    finally:
        conn.close()


def _position_row(row: Mapping[str, object]) -> dict[str, object]:
    """Position row with ISO-string timestamps (existing boundary)."""
    out = dict(row)
    quote_at = out.get("quote_retrieved_at")
    if isinstance(quote_at, datetime):
        out["quote_retrieved_at"] = quote_at.isoformat()
    return out


def _snapshot_created_at(value: object) -> datetime:
    """Snapshot instant from a TEXT column value (existing boundary)."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _snapshot_header(row: Mapping[str, object]) -> tuple[str, datetime, str]:
    """(snapshot id, created_at, broker) from the newest snapshot row."""
    return str(row["snapshot_id"]), _snapshot_created_at(row["created_at"]), str(row["broker"])


def _snapshot_decimals(row: Mapping[str, object]) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """(cash, invested, total) as canonical decimals (None stays None)."""
    return (
        _canonical_decimal(Decimal(str(row["cash"])) if row.get("cash") is not None else None),
        _canonical_decimal(Decimal(str(row["invested_value"])) if row.get("invested_value") is not None else None),
        _canonical_decimal(Decimal(str(row["total_value"])) if row.get("total_value") is not None else None),
    )


def _snapshot_account_ids(
    snapshot_id: str, positions: tuple[object, ...], *, data_root: Path | None
) -> tuple[str, ...]:
    conn = _connect(data_root)
    try:
        account_rows = conn.execute(
            "SELECT account_id FROM portfolio_accounts WHERE snapshot_id = ? ORDER BY rowid",
            (snapshot_id,),
        ).fetchall()
    finally:
        conn.close()
    account_ids = [str(row["account_id"]) for row in account_rows]
    if account_ids:
        return tuple(account_ids)
    # Snapshots predating portfolio_accounts fall back to the
    # position-derived reconstruction (accounts without positions are
    # unrecoverable there, same as before this change).
    legacy: list[str] = []
    for position in positions:
        account_id = str(getattr(position, "account_id", None))
        if account_id not in legacy:
            legacy.append(account_id)
    return tuple(legacy)


def read_latest_snapshot(*, data_root: Path | None = None) -> PortfolioSnapshot | None:
    """Return the newest persisted snapshot, or None when none exists.

    ``account_ids`` comes from ``portfolio_accounts`` in write order;
    positions are returned in persisted write order (position_id embeds
    opaque account ids that do not sort by account).
    """
    conn = _connect(data_root)
    try:
        rows = conn.execute("SELECT * FROM portfolio_snapshots ORDER BY created_at DESC LIMIT 1").fetchall()
        if not rows:
            return None
        row = rows[0]
        snapshot_id, created_at, broker = _snapshot_header(row)
        position_rows = conn.execute(
            "SELECT * FROM portfolio_positions WHERE snapshot_id = ? ORDER BY rowid",
            (snapshot_id,),
        ).fetchall()
    finally:
        conn.close()
    positions = tuple(
        _position_from_row(_position_row(dict(position_row)), created_at) for position_row in position_rows
    )
    cash, invested_value, total_value = _snapshot_decimals(dict(row))
    return PortfolioSnapshot(
        snapshot_id=snapshot_id,
        created_at=created_at,
        broker=broker,
        account_ids=_snapshot_account_ids(snapshot_id, positions, data_root=data_root),
        cash=cash,
        invested_value=invested_value,
        total_value=total_value,
        positions=positions,
    )


def sync_robinhood_portfolio(
    provider: RobinhoodPortfolioProvider,
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
) -> PortfolioSnapshot:
    """Full read-only portfolio sync: accounts, positions, cash, one batched
    quote call, identity resolution, valuation, persistence."""
    created_at = now or datetime.now(UTC)
    accounts = provider.get_accounts()
    raw_positions: list[BrokeragePosition] = []
    cash_balances: list[CashBalance] = []
    for account in accounts:
        raw_positions.extend(provider.get_positions(account.account_id))
        cash_balances.append(provider.get_cash_balance(account.account_id))
    tickers = list(dict.fromkeys(position.ticker for position in raw_positions))
    quotes = provider.get_equity_quotes(tickers)
    positions = [
        build_position(
            to_position_input(raw),
            resolve_security(
                raw.ticker,
                provider_instrument_id=raw.provider_instrument_id,
                as_of=created_at,
                data_root=data_root,
            ),
            to_quote(quotes[raw.ticker]) if raw.ticker in quotes else None,
        )
        for raw in raw_positions
    ]
    snapshot = build_portfolio_snapshot(
        broker="robinhood",
        account_ids=[local_account_id(account.account_id) for account in accounts],
        positions=positions,
        cash_balances={local_account_id(balance.account_id): balance.cash for balance in cash_balances},
        created_at=created_at,
    )
    persist_snapshot(snapshot, data_root=data_root)
    return snapshot
