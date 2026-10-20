"""Read-only Robinhood portfolio routes for the UI holdings panel.

- GET /api/portfolio — the newest persisted snapshot (no broker call).
- POST /api/portfolio/sync — one read-only Robinhood sync, then the new snapshot.

Both routes return 404 when BROKER_ENABLED is off or OAuth is missing, so the
panel hides. Responses carry no account or position identifiers.
"""

from fastapi import APIRouter, HTTPException

from app.domain.portfolio import PortfolioSnapshot
from app.robinhood import RobinhoodClient
from app.robinhood.portfolio import RobinhoodPortfolioProvider
from app.services.portfolio_sync import read_latest_snapshot, sync_robinhood_portfolio

router = APIRouter()

SETUP_MESSAGE = (
    "Robinhood portfolio unavailable. Set BROKER_ENABLED=true and run `venv/bin/python cli.py robinhood-login`."
)
_ACCOUNT_TOOLS = frozenset({"get_accounts", "get_portfolio", "get_equity_positions"})


def _client() -> RobinhoodClient:
    """The tool layer's gated client: disabled broker or missing OAuth reads as 404."""
    from app import tools

    try:
        return tools._robinhood_client(account_tools=_ACCOUNT_TOOLS)
    except RuntimeError as exc:  # disabled broker; RobinhoodAuthRequired subclasses RuntimeError
        raise HTTPException(status_code=404, detail=SETUP_MESSAGE) from exc


def _money(value: object) -> str | None:
    return None if value is None else str(value)


def _body(snapshot: PortfolioSnapshot | None) -> dict[str, object]:
    if snapshot is None:
        return {"synced_at": None, "cash": None, "invested_value": None, "total_value": None, "positions": []}
    return {
        "synced_at": snapshot.created_at.isoformat(),
        "cash": _money(snapshot.cash),
        "invested_value": _money(snapshot.invested_value),
        "total_value": _money(snapshot.total_value),
        "positions": [
            {
                "ticker": p.ticker,
                "quantity": _money(p.quantity),
                "average_cost": _money(p.average_cost),
                "market_price": _money(p.market_price),
                "market_value": _money(p.market_value),
                "unrealized_gain": _money(p.unrealized_gain),
                "unrealized_gain_pct": _money(p.unrealized_gain_pct),
            }
            for p in snapshot.positions
        ],
    }


@router.get("/api/portfolio")
def portfolio() -> dict[str, object]:
    _client()
    return _body(read_latest_snapshot())


@router.post("/api/portfolio/sync")
def portfolio_sync() -> dict[str, object]:
    return _body(sync_robinhood_portfolio(RobinhoodPortfolioProvider(_client())))
