"""Per-ticker workspaces stored as data/workspaces/<SYM>.json, plus server-backed widget data.

- GET /api/workspaces/<sym> — {sym, widgets, updated_at}; never saved is widgets [] and updated_at null.
- PUT /api/workspaces/<sym> — {widgets: [{id, type, ticker, params?}]} (max 24); unknown keys are 400.
- GET /api/widget-data?type=&sym= — short-interest | short-volume | insider | multiples rows.
  Best-effort: source failures come back as an "error" string, never a 500.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, cast

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import cache
from app.config import get_data_root
from app.research.models import validate_ticker
from app.thesis.models import JSONValue
from app.thesis.yaml import atomic_write_json
from calculations.valuation import earnings_yield, pe

from . import market_providers as mp

router = APIRouter()

_INSIDER_DAYS = 180


def _ticker(value: object, where: str) -> str:
    ticker = validate_ticker(value, where)
    if ticker is None:
        raise ValueError(f"{where}: 'ticker' must be a non-empty string")
    return ticker


class Widget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    type: str
    ticker: str
    params: dict[str, object] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        from app.tools import WIDGET_TYPES

        if value not in WIDGET_TYPES:
            raise ValueError(f"must be one of {', '.join(WIDGET_TYPES)}")
        return value

    @field_validator("ticker", mode="before")
    @classmethod
    def _valid_ticker(cls, value: object) -> str:
        return _ticker(value, "widget")


class Workspace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    widgets: list[Widget] = Field(max_length=24)


def _sym(raw: str, what: str) -> str:
    try:
        return _ticker(raw, what)
    except ValueError as exc:
        raise HTTPException(400, f"invalid {what}: {exc}") from exc


@router.get("/api/workspaces/{sym:path}")
def get_workspace(sym: str) -> dict[str, object]:
    ticker = _sym(sym, "workspace")
    path = get_data_root() / "workspaces" / f"{ticker}.json"
    if not path.is_file():
        return {"sym": ticker, "widgets": [], "updated_at": None}
    stored: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return stored


@router.put("/api/workspaces/{sym:path}")
def put_workspace(sym: str, payload: Annotated[object, Body()] = None) -> dict[str, object]:
    ticker = _sym(sym, "workspace")
    try:
        workspace = Workspace.model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise HTTPException(400, f"invalid workspace: {detail}") from exc
    body: dict[str, object] = {
        "sym": ticker,
        "widgets": [w.model_dump(exclude_unset=True) for w in workspace.widgets],
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(f"workspaces/{ticker}.json", cast("dict[str, JSONValue]", body), get_data_root())
    return body


def _records(res: dict[str, object]) -> list[dict[str, object]]:
    rows = res.get("records")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _date_key(row: dict[str, object]) -> str:
    return str(row["date"])


def _short_interest(sym: str) -> dict[str, object]:
    from app import finra_client

    today = datetime.now(UTC).date()
    res = finra_client.get_finra_datapoints(
        "otcMarket/consolidatedShortInterest",
        fields=["settlementDate", "currentShortPositionQuantity", "daysToCoverQuantity"],
        ticker=sym,
        start_date=(today - timedelta(days=365)).isoformat(),
        end_date=today.isoformat(),
        limit=25,
    )
    if res.get("error"):
        return {"rows": [], "error": str(res["error"])}
    rows: list[dict[str, object]] = [
        {
            "date": r.get("settlementDate"),
            "short_qty": r.get("currentShortPositionQuantity"),
            "days_to_cover": r.get("daysToCoverQuantity"),
        }
        for r in _records(res)
    ]
    rows.sort(key=_date_key)
    return {"rows": rows}


def _short_window(sym: str, start: str, end: str) -> tuple[list[dict[str, object]], str | None]:
    """One date window of Reg SHO rows, or ([], error). Incomplete windows are an error."""
    from app import finra_client

    res = finra_client.get_finra_datapoints(
        "otcMarket/regShoDaily",
        fields=["tradeReportDate", "shortParQuantity", "totalParQuantity"],
        ticker=sym,
        start_date=start,
        end_date=end,
        limit=25,
    )
    if res.get("error"):
        return [], str(res["error"])
    if res.get("may_have_more"):
        return [], "Short-volume data is incomplete: a date window exceeds the 25-record limit."
    return _records(res), None


def _short_volume(sym: str) -> dict[str, object]:
    today = datetime.now(UTC).date()

    days: dict[str, list[float]] = {}
    errors: list[str] = []
    no_data: str | None = None
    # ponytail: four date windows replace unsupported offsets, capped at 100 records over 14 days.
    # Reject incomplete windows. Add client pagination if this bound becomes insufficient.
    for back in range(0, 14, 4):
        rows, error = _short_window(
            sym, (today - timedelta(days=min(back + 3, 13))).isoformat(), (today - timedelta(days=back)).isoformat()
        )
        if error:
            if error.startswith("No data found"):
                no_data = error
            else:
                errors.append(error)
            continue
        for r in rows:
            day = days.setdefault(str(r.get("tradeReportDate")), [0.0, 0.0])
            day[0] += float(cast("float", r.get("shortParQuantity") or 0))
            day[1] += float(cast("float", r.get("totalParQuantity") or 0))
    rows = [
        {"date": d, "short_qty": s, "total_qty": t, "short_pct": s / t * 100 if t else None}
        for d, (s, t) in sorted(days.items())
    ]
    error = "; ".join(dict.fromkeys(errors)) or (no_data if not rows else None)
    return {"rows": rows, "error": error} if error else {"rows": rows}


def _insider(sym: str) -> dict[str, object]:
    """Open-market insider activity only. Grants and other transaction kinds are not sales."""
    txns, err = mp._recent_insider(sym, _INSIDER_DAYS)
    if err:
        return {"rows": [], "error": err}
    rows: list[dict[str, object]] = [
        {"date": t.get("filed"), "title": mp._insider_title(t), "kind": t.get("kind"), "shares": t.get("shares")}
        for t in txns
        if t.get("kind") in ("open_market_purchase", "open_market_sale")
    ]
    rows.sort(key=_date_key, reverse=True)
    return {"rows": rows}


def _multiples(sym: str) -> dict[str, object]:
    from app.services import sec_facts

    quote = mp.get_quote(sym)
    eps = sec_facts.get_fundamentals(sym, "eps")
    price = quote.get("price")
    ttm = eps.get("ttm_eps_diluted")
    price_n = price if isinstance(price, (int, float)) else None
    ttm_n = ttm if isinstance(ttm, (int, float)) else None
    out: dict[str, object] = {
        "price": price_n,
        "ttm_eps": ttm_n,
        "pe": None,
        "earnings_yield": None,
        "as_of": quote.get("retrieved_at"),
        "history": None,
    }
    if price_n is not None and ttm_n is not None:
        try:
            out["pe"] = pe(price_n, ttm_n)
        except ValueError:
            out["pe"] = None
        try:
            out["earnings_yield"] = earnings_yield(ttm_n, price_n)
        except ValueError:
            out["earnings_yield"] = None
    errors = [str(e) for e in (quote.get("error"), eps.get("error")) if e]
    if errors:
        out["error"] = "; ".join(errors)
    return out


_WIDGET_DATA = {
    "short-interest": _short_interest,
    "short-volume": _short_volume,
    "insider": _insider,
    "multiples": _multiples,
}


@router.get("/api/widget-data")
def widget_data(type: str = "", sym: str = "") -> dict[str, object]:
    if type not in _WIDGET_DATA:
        raise HTTPException(400, f"unknown widget type: {type!r}; server-backed: {', '.join(_WIDGET_DATA)}")
    ticker = _sym(sym, "sym")
    key = f"widget:{type}:{ticker}"
    hit = cache.get(key, ttl=900)
    if isinstance(hit, dict):
        return hit
    try:
        data = _WIDGET_DATA[type](ticker)
    except Exception as e:  # noqa: BLE001 - best-effort boundary, envelope not tracebacks
        data = {"error": f"{type} unavailable for {ticker}: {e}"}
    out = {"sym": ticker, "type": type, **data}
    cache.set(key, out)
    return out
