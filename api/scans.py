"""Scan definitions: server-defined presets + custom filters stored as data/scans/<id>.json.

- GET /api/scans — {"scans": [...]}: presets in fixed order, then custom newest first.
- POST /api/scans — {name, filter?, kind?="custom"}; unknown keys are 400. Stores only.
- DELETE /api/scans/<id> — custom only; presets are 400, unknown ids 404.

Preset ids double as GET /api/markets?scan=<id>.
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_data_root
from app.thesis.models import JSONValue
from app.thesis.yaml import atomic_write_json

from .market_providers import MAX_SYMS, SCAN_PRESETS

router = APIRouter(prefix="/api/scans")

_ID_RE = re.compile(r"^scan-[0-9a-f]{12}$")


class ScanFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] | None = Field(default=None, max_length=MAX_SYMS)
    forms: list[str] | None = None
    min_short_pct: float | None = None
    min_chg_pct: float | None = None
    sector: str | None = None


class NewScan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    kind: Literal["custom"] = "custom"
    filter: ScanFilter = Field(default_factory=ScanFilter)


def _dir() -> Path:
    return get_data_root() / "scans"


def _presets() -> list[dict[str, object]]:
    return [
        {"id": sid, "name": name, "kind": "preset", "filter": {}, "created_at": None}
        for sid, name in SCAN_PRESETS.items()
    ]


def _created(scan: dict[str, object]) -> str:
    return str(scan.get("created_at") or "")


@router.get("")
def list_scans() -> dict[str, object]:
    custom = [json.loads(p.read_text(encoding="utf-8")) for p in _dir().glob("scan-*.json")]
    custom.sort(key=_created, reverse=True)
    return {"scans": _presets() + custom}


@router.post("", status_code=201)
def create_scan(payload: dict[str, object]) -> dict[str, object]:
    try:
        scan = NewScan.model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise HTTPException(400, f"invalid scan: {detail}") from exc
    flt = scan.filter.model_dump(exclude_none=True)
    if "symbols" in flt:
        flt["symbols"] = [s.strip().upper() for s in flt["symbols"] if s.strip()]
    body: dict[str, object] = {
        "id": f"scan-{uuid.uuid4().hex[:12]}",
        "name": scan.name,
        "kind": "custom",
        "filter": flt,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    root = get_data_root()
    atomic_write_json(f"scans/{body['id']}.json", cast("dict[str, JSONValue]", body), root)
    return body


@router.delete("/{scan_id}")
def delete_scan(scan_id: str) -> dict[str, object]:
    if scan_id in SCAN_PRESETS:
        raise HTTPException(400, f"preset scans are server-defined: {scan_id}")
    path = _dir() / f"{scan_id}.json"
    if not _ID_RE.match(scan_id) or not path.is_file():
        raise HTTPException(404, f"unknown scan: {scan_id}")
    path.unlink()
    return {"deleted": scan_id}
