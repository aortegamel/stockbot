"""Slice coverage for GoogleCliFix: patent/geo parsing edges, backfill resume,
CLI arg validation, FINRA spec cache. All executor/factory fakes; no network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path  # noqa: F401 - kept for test-slice import shape

import pytest

import cli
from app import finra_client
from app.google_data import (
    geo_context,
    patents,
    stackoverflow,
    trends_api,
    youtube,
)


def _enable(monkeypatch: pytest.MonkeyPatch, project: str = "test-proj") -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)


# --- patent parsing edge cases -------------------------------------------------


def test_patent_half_open_dates_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    err, _, _ = patents._check_dates("2020-01-01", None)
    assert err is not None and err["error_type"] == "invalid_params"


def test_patent_reversed_dates_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    err, _, _ = patents._check_dates("2021-01-01", "2020-01-01")
    assert err is not None and "after" in str(err["error"])


def test_patent_malformed_date_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    err, _, _ = patents._check_dates("Jan 2020", "2020-02-01")
    assert err is not None and err["error_type"] == "invalid_params"


def test_patent_stats_skips_garbage_rows_and_gaps_years(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows = [
        {"pub_year": 2020, "pub_count": 3, "family_count": 2, "total_citations": 9, "cpc_bag": [["G06N", 2]]},
        "not-a-row",
        {"pub_year": None, "pub_count": 1},
        {"pub_year": "bogus", "pub_count": 1},
        {"pub_year": 2021, "pub_count": "bad", "family_count": "bad", "total_citations": "bad", "cpc_bag": "bad"},
    ]
    out = patents.get_assignee_stats("c1", assignees=["Alias"], executor=lambda t, p: {"status": "ok", "rows": rows})
    assert out["status"] == "ok"
    years = out["years"]
    assert isinstance(years, list)
    by_year = {y["pub_year"]: y for y in years if isinstance(y, dict)}
    assert by_year[2020]["pub_count"] == 3
    assert by_year[2021]["pub_count"] == 0  # bad values keep zeroed bucket
    assert out["gaps"] == 2  # null + unparseable years


def test_patent_publication_non_dict_rows_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows: list[object] = ["junk", {"publication_id": "p1", "publication_date": "2020-01-01"}]
    got = patents._collect_publications(rows, 10)
    assert [p["publication_id"] for p in got] == ["p1"]


# --- geo parsing edge cases ----------------------------------------------------


def test_geo_noaa_sentinels_warn_and_null(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows = [{"station_id": "USW1", "observed_at": "2026-09-01", "temp": 9999.9, "frshtt": "12"}]
    out = geo_context.get_geo_context(
        ["USW1"], variables=["TEMP"], executor=lambda t, p: {"status": "ok", "rows": rows}
    )
    assert out["status"] == "ok"
    context = out["context"]
    assert isinstance(context, list)
    temps = [c for c in context if isinstance(c, dict) and c["variable"] == "temp"]
    assert temps and temps[0]["value"] is None
    blob = json.dumps(out).lower()
    assert "incomplete quality" in blob and "tornado/hail" in blob


def test_geo_census_sentinels_null(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows = [{"geo_id": "04000US06", "total_pop": -666666666, "median_age": 37.5, "provider": "Census"}]
    out = geo_context.get_geo_context(
        ["04000US06"],
        variables=["total_pop", "median_age"],
        table_suffix="state_2020_5yr",
        executor=lambda t, p: {"status": "ok", "rows": rows},
    )
    assert out["status"] == "ok"
    context = out["context"]
    assert isinstance(context, list)
    pops = [c for c in context if isinstance(c, dict) and c["variable"] == "total_pop"]
    assert pops and pops[0]["value"] is None


def test_geo_unknown_census_table_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    out = geo_context.get_geo_context(
        ["04000US06"], table_suffix="nope_2099_x", executor=lambda t, p: {"status": "ok", "rows": []}
    )
    assert out["status"] == "unavailable" and "available" in out


def test_geo_missing_station_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    out = geo_context.get_geo_context(
        ["USW1", "USW-MISSING"],
        variables=["TEMP"],
        executor=lambda t, p: {
            "status": "ok",
            "rows": [{"station_id": "USW1", "observed_at": "2026-09-01", "temp": 70.0}],
        },
    )
    assert out["status"] == "ok"
    missing = out["missing_geos"]
    assert isinstance(missing, list)
    assert "USW-MISSING" in missing


def test_geo_invalid_limit_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)

    def _exec(t: str, p: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "rows": []}

    out = geo_context.get_geo_context([], variables=["TEMP"], executor=_exec)
    assert out["status"] == "error" and out["error_type"] == "invalid_params"


def test_trends_api_terms_and_interval_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    err = trends_api.get_interest_over_time(terms=[])
    assert err["status"] == "error" and err["error_type"] == "invalid_params"
    err2 = trends_api.get_interest_over_time(terms=["x"], interval="fortnightly")
    assert err2["status"] == "error" and "interval" in str(err2["error"])


def test_trends_api_pending_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("GOOGLE_TRENDS_API_ENABLED", "true")
    out = trends_api.get_interest_over_time(terms="alpha", country="US")
    assert out["status"] == "unavailable"
    coverage = out["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["terms"] == ["alpha"]


def test_stackoverflow_tags_required_and_window_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    bad = stackoverflow.get_tag_activity([], executor=lambda t, p: {"status": "ok", "rows": []})
    assert bad["status"] == "error"
    ok = stackoverflow.get_tag_activity(
        ["python"], start_date="2020-01-01", end_date="2020-02-01", executor=lambda t, p: {"status": "ok", "rows": []}
    )
    assert ok["status"] == "ok"


def test_youtube_quota_ledger_validated() -> None:
    today = youtube._pacific_today()
    with pytest.raises(youtube._QuotaRefused):
        youtube._validate_days({"not-a-date": {"search": 1, "videos": 0}})
    assert youtube._validate_days({today: {"search": 1, "videos": 0}})


# --- CLI arg validation --------------------------------------------------------


def test_cli_google_data_bad_subcommand_exits() -> None:
    args = argparse.Namespace(
        google_data_command="bogus",
        source="trends",
        company=None,
        start_date=None,
        end_date=None,
        geo=None,
        variable=[],
        limit=25,
        data_root=None,
    )
    with pytest.raises(SystemExit):
        cli._cmd_google_data(args)


def test_cli_google_data_limit_clamped() -> None:
    args = argparse.Namespace(limit=10**9)
    assert cli._google_limit(args) == 1000
    assert cli._google_limit(argparse.Namespace(limit=None)) == 25


def test_cli_google_sources_all_expands() -> None:
    args = argparse.Namespace(source="all")
    assert set(cli._google_sources(args)) == {"trends", "patents", "macro", "geo", "stackoverflow"}


def test_cli_unknown_command_errors() -> None:
    assert cli._dispatch_command(argparse.Namespace(command="nope")) is False


def test_cli_patents_requires_company(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        google_data_command="collect",
        source="patents",
        company=None,
        start_date=None,
        end_date=None,
        geo=None,
        variable=[],
        limit=25,
        data_root=None,
    )
    cli._cmd_google_data(args)
    assert json.loads(capsys.readouterr().out)["sources"]["patents"]["status"] == "error"


# --- FINRA spec cache ----------------------------------------------------------


def test_finra_spec_from_cached_prefers_hit() -> None:
    entry = finra_client.CatalogEntry(group="g", name="n", description="entry-desc", methods=("m",))
    hit: dict[str, object] = {
        "description": "hit-desc",
        "fields": [{"name": "f"}],
        "partition_fields": ["p"],
        "methods": ["m2"],
        "symbol_field": "sym",
        "date_field": "dt",
        "market_aggregate": True,
        "default_filters": [["a", "b"]],
        "valid_filter_values": {"a": ["x"]},
    }
    spec = finra_client._spec_from_cached(entry, hit)
    assert spec.description == "hit-desc"
    assert spec.symbol_field == "sym" and spec.date_field == "dt"
    assert spec.market_aggregate is True
    assert spec.partition_fields == ("p",)


def test_finra_spec_from_cached_falls_back_to_entry() -> None:
    entry = finra_client.CatalogEntry(group="g", name="n", description="entry-desc", methods=("m",))
    spec = finra_client._spec_from_cached(entry, {})
    assert spec.description == "entry-desc"
    assert spec.methods == ("m",) and spec.fields == ()
