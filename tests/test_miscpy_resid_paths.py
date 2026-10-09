"""MiscPy slice: resources + resource_stores + freeze gate + finra/edgar + show_live.

Owned-file gate targets (BEFORE on coverage-python.lcov):
  resources.py read_resource 30.0 / parse_resource_uri 20.0 (N/A: no SF entry)
  repository.resource_stores 11.1/cc7/56% (miss 643-646, 649-651)
  service._freeze_wave_records 10.1/cc10/90% (miss 852)
  finra _parse_live_methods 26.8/cc6/16.7% (miss 2454-2458)
  edgar _fundamentals_dividends 12.2/cc6/44% (miss 572-573, 577-579, 583-587)
  tools._show_live 16.7/cc6/33% (miss 4890-4891)
  evidence_resolution: max 9.0, nothing over 10 -> small guard test only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from app.policy import Capability, RequestContext

T0 = "2026-01-01T00:00:00+00:00"


# -- resources.py: pure helpers, zero coverage before --


def test_miscpy_parse_resource_uri_happy_and_errors():
    from app.research.resources import ResourceError, parse_resource_uri

    assert parse_resource_uri("evidence://ev-1") == ("evidence", "ev-1")
    assert parse_resource_uri(" Research://SID/sec ") == ("research", "SID/sec")
    with pytest.raises(ResourceError):
        parse_resource_uri("no-separator")
    with pytest.raises(ResourceError):
        parse_resource_uri("evidence://  ")
    with pytest.raises(ResourceError):
        parse_resource_uri("nope://x")


def test_miscpy_read_resource_happy_and_missing():
    from app.research.resources import (
        ResourceError,
        ResourceNotFoundError,
        read_resource,
    )

    assert read_resource("evidence://e1", evidence={"e1": {"a": 1}}) == {"a": 1}
    # research:// strips the /section suffix
    assert read_resource("research://SID/sec", sessions={"SID": {"s": 1}}) == {"s": 1}
    with pytest.raises(ResourceError):
        read_resource("evidence://e1")  # no store wired
    with pytest.raises(ResourceNotFoundError):
        read_resource("evidence://zzz", evidence={"e1": 1})
    with pytest.raises(ResourceNotFoundError):
        read_resource("research://NOPE/sec", sessions={"SID": 1})
    with pytest.raises(ResourceNotFoundError):
        read_resource("research:///sec", sessions={"SID": 1})


# -- repository.resource_stores: missing-freeze + non-str dossier branches --


def _research_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    return ResearchRepository()


def test_miscpy_resource_stores_skips_missing_freeze_and_non_str_dossier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import service as svc

    repo = _research_repo(tmp_path, monkeypatch)
    sid = svc.create_research("NVDA demand?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    sess = repo.get_session(sid)
    sess = dataclasses.replace(sess, freeze_ids=["ghost-freeze"])
    repo.save_session(sess)
    # one dossier without a str dossier_id + one good one
    repo.save_dossier({"dossier_id": "d-good", "session_id": sid, "body": "ok"})
    stores = repo.resource_stores(sid)
    assert stores["freeze"] == {}  # KeyError arm (643-646)
    assert set(stores["dossier"]) == {"d-good"}  # non-str arm (649-651)
    assert stores["research"] == {sid: repo.get_session(sid).to_dict()}


# -- service._freeze_wave_records: open-source-jobs ValueError arm --


def _evidence_record(sid: str, eid: str, wave: int, content: str = "nvda demand grows") -> dict[str, object]:
    from app.research.evidence import evidence_content_hash

    now = "2025-06-30T00:00:00+00:00"
    return {
        "evidence_id": eid,
        "session_id": sid,
        "wave_id": wave,
        "source_type": "filing",
        "source_name": "sec",
        "subject": "NVDA",
        "claim_text": "c",
        "content": content,
        "content_hash": evidence_content_hash(content),
        "retrieved_at": now,
        "known_at": now,
        "source_uri": "sec://x",
        "job_id": "j1",
        "agent_id": "a1",
    }


def test_miscpy_freeze_wave_records_open_source_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import jobs as _jobs
    from app.research import service as svc

    repo = _research_repo(tmp_path, monkeypatch)
    sid = svc.create_research("NVDA demand?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    repo.save_evidence(_evidence_record(sid, "ev-1", 1))
    # the create_research source_agent job is still running -> open_src arm (852)
    with pytest.raises(ValueError, match="still open"):
        svc._freeze_wave_records(repo, sid, 1)
    # completing it unblocks the happy path
    running = next(j for j in repo.list_jobs(sid) if j.status == "running")
    repo.save_job(_jobs.complete_job(running, result={"ok": True}))
    assert len(svc._freeze_wave_records(repo, sid, 1)) == 1
    assert svc._freeze_wave_records(repo, sid, 0) == []


# -- finra _parse_live_methods: tuple/list/str/other arms --


def test_miscpy_parse_live_methods_branches():
    from app.finra_client import _parse_live_methods

    assert _parse_live_methods(["a", 1]) == ("a", "1")  # list arm (2454-2455)
    assert _parse_live_methods(("x",)) == ("x",)
    assert _parse_live_methods(" a, ,b ") == ("a", "b")  # str arm (2456-2457)
    assert _parse_live_methods(None) == ()  # fallthrough (2458)
    assert _parse_live_methods(5) == ()


def test_miscpy_metadata_methods_prefers_catalog_or_live():
    from app.finra_client import CatalogEntry, _metadata_methods

    entry = CatalogEntry(group="g", name="n", description="d", methods=("m1",))
    assert _metadata_methods(entry, {"supportedMethods": ["m2"]}) == ("m1",)
    bare = CatalogEntry(group="g", name="n", description="d")
    assert _metadata_methods(bare, {"supportedMethods": "a, b"}) == ("a", "b")


# -- edgar _fundamentals_dividends: no-facts / empty-div / quarterly arms --


class _DivCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str, ttl: float | None = None) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


class _DivFacts:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame


class _DivCompany:
    rows: ClassVar[list[dict[str, object]]] = []
    none_facts: bool = False

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def get_facts(self) -> _DivFacts | None:
        if type(self).none_facts:
            return None
        return _DivFacts(pd.DataFrame(type(self).rows))


def _q_row(value: float, start: str, end: str, fy: int = 2025, fp: str = "Q1") -> dict[str, object]:
    return {
        "concept": "us-gaap:CommonStockDividendsPerShareDeclared",
        "value": value,
        "period_start": start,
        "period_end": end,
        "fiscal_period": fp,
        "fiscal_year": fy,
    }


def _fy_row(value: float, year: int) -> dict[str, object]:
    return {
        "concept": "us-gaap:CommonStockDividendsPerShareDeclared",
        "value": value,
        "period_start": f"{year}-01-01",
        "period_end": f"{year}-12-31",
        "fiscal_period": "FY",
        "fiscal_year": year,
    }


def test_miscpy_fundamentals_dividends_no_facts_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import edgar_client

    monkeypatch.setattr(edgar_client, "Company", _DivCompany)
    monkeypatch.setattr(edgar_client, "cache", _DivCache())
    monkeypatch.setattr(edgar_client, "_ensure_init", lambda: None)
    monkeypatch.setattr(_DivCompany, "none_facts", True)
    assert "error" in edgar_client.get_fundamentals("KO", "dividends")  # 564-565
    monkeypatch.setattr(_DivCompany, "none_facts", False)
    monkeypatch.setattr(
        _DivCompany,
        "rows",
        [
            {
                "concept": "us-gaap:Other",
                "value": 1.0,
                "period_start": "2025-01-01",
                "period_end": "2025-03-31",
                "fiscal_period": "Q1",
                "fiscal_year": 2025,
            }
        ],
    )
    out = edgar_client.get_fundamentals("KO", "dividends", include_dividend_price=False)  # 570-571
    assert out["dividend_status"] == "insufficient_data"


def test_miscpy_fundamentals_dividends_quarterly_and_fy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import edgar_client

    monkeypatch.setattr(edgar_client, "Company", _DivCompany)
    monkeypatch.setattr(edgar_client, "cache", _DivCache())
    monkeypatch.setattr(edgar_client, "_ensure_init", lambda: None)
    rows = [
        _q_row(0.50, "2025-07-01", "2025-09-30", 2025, "Q3"),
        _q_row(0.50, "2025-10-01", "2025-12-31", 2025, "Q4"),
        _q_row(0.50, "2026-01-01", "2026-03-31", 2026, "Q1"),
        _q_row(0.50, "2026-04-01", "2026-06-30", 2026, "Q2"),
        _fy_row(2.00, 2025),
    ]
    monkeypatch.setattr(_DivCompany, "rows", rows)
    out = edgar_client.get_fundamentals("KO", "dividends", include_dividend_price=False)  # 577-581
    assert out["ttm_dividend_per_share"] == 2.0
    assert out["dividend_status"] == "paying"
    monkeypatch.setattr(_DivCompany, "rows", [_fy_row(2.0, 2025)])
    monkeypatch.setattr(edgar_client, "cache", _DivCache())
    fy_only = edgar_client.get_fundamentals("KO", "dividends", include_dividend_price=False)  # 582-583
    assert fy_only["dividend_status"] == "unknown"
    assert fy_only["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.0}]


# -- tools._show_live: the live branch of thesis_show (miss 4890-4891) --


def _tctx(tmp_path: Path) -> RequestContext:
    return RequestContext("test", frozenset({Capability.RESEARCH}), data_root=tmp_path)


def test_miscpy_thesis_show_live_branch(tmp_path: Path) -> None:
    from app.thesis.repository import ThesisRepository
    from app.tools import _show_live, execute_tool

    repo = ThesisRepository(tmp_path / "thesis")
    thesis = repo.create_thesis("NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    tid = thesis.thesis_id
    packet = _show_live(repo, thesis, tid)  # direct: covers 4890-4891 defs
    assert packet["thesis_id"] == tid and packet["slug"] == thesis.slug
    live = execute_tool("thesis_show", {"id": tid}, "m", context=_tctx(tmp_path))
    assert live["thesis_id"] == tid and live["user_thesis"] == "NVDA thesis"


# -- evidence_resolution: nothing over 10 (max 9.0); guard the unresolved miss --


def test_miscpy_resolve_subject_blank_and_miss():
    from datetime import UTC, datetime

    import app.services.evidence_resolution as er

    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    blank = er.resolve_subject(
        ticker=None, name="   ", aliases_by_ticker=lambda t: [], name_to_ticker=lambda n: None, as_of=as_of
    )
    assert blank.resolved is False
    miss = er.resolve_subject(
        ticker=None,
        name="no-such-company-xyz",
        aliases_by_ticker=lambda t: [],
        name_to_ticker=lambda n: None,
        as_of=as_of,
    )
    assert miss.resolved is False
