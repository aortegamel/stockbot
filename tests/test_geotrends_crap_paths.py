"""Slice coverage for GeoTrendsFix: parser edges + cache/backfill arms.

All executor/factory fakes; no network. Behavior-preserving: asserts current
shapes only. Robinhood OAuth Storage is internal to build_oauth_provider (not
exposed on the provider), so its empty-state arm is covered directly through
load_tokens_for_origin, which is the exact branch Storage.get_tokens funnels
through.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app import analyst_client, cache, finra_client
from app.google_data import (
    bigquery_client,
    datacommons,
    geo_context,
    patents,
    signals,
    stackoverflow,
    trends,
    trends_api,
    youtube,
)
from app.robinhood import auth as rh_auth
from app.robinhood.account import _json_value
from app.services import sec_facts


def _enable(monkeypatch: pytest.MonkeyPatch, project: str = "test-proj") -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)


# --- cache: conn reuse + TTL arms ------------------------------------------------


def test_cache_conn_reused_for_same_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    assert cache._conn() is cache._conn()


def test_cache_set_get_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    cache.set("k1", {"v": 1})
    assert cache.get("k1") == {"v": 1}


def test_cache_missing_key_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    assert cache.get("no-such-key") is None


def test_cache_expired_ttl_miss_but_untimed_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    cache.set("k2", [1, 2])
    con = sqlite3.connect(cache._db_path())
    con.execute("UPDATE cache SET created_at = 0 WHERE key = 'k2'")
    con.commit()
    con.close()
    assert cache.get("k2", ttl=60) is None
    assert cache.get("k2") == [1, 2]


# --- finra: legacy list hit + bool text + group/name scan ------------------------
def test_finra_cached_list_hit_legacy_arm() -> None:
    hit = finra_client._cached_list_hit([{"a": 1}, "junk"])
    assert hit is not None
    records, headers = hit
    assert records == [{"a": 1}] and headers == {}


def test_finra_cached_list_hit_non_list_none() -> None:
    assert finra_client._cached_list_hit({"records": []}) is None


def test_finra_parse_bool_true_arms() -> None:
    assert finra_client._parse_bool_text("true") is True
    assert finra_client._parse_bool_text(" YES ") is True
    assert finra_client._parse_bool_text("1") is True
    assert finra_client._parse_bool_text(True) is True


def test_finra_parse_bool_false_arms() -> None:
    assert finra_client._parse_bool_text("false") is False
    assert finra_client._parse_bool_text(" No ") is False
    assert finra_client._parse_bool_text("0") is False
    assert finra_client._parse_bool_text(False) is False


def test_finra_parse_bool_garbage_none() -> None:
    assert finra_client._parse_bool_text("maybe") is None
    assert finra_client._parse_bool_text(None) is None
    assert finra_client._parse_bool_text(42) is None


def _finra_entries() -> list[finra_client.CatalogEntry]:
    return [
        finra_client.CatalogEntry(group="OTC", name="ShortInterest", description="d"),
        finra_client.CatalogEntry(group="Equity", name="RegSho", description="d"),
    ]


def test_finra_scan_group_name_match() -> None:
    hit = finra_client._scan_group_name(_finra_entries(), "otc / shortinterest")
    assert hit is not None and hit.dataset_id == "OTC/ShortInterest"


def test_finra_scan_group_name_non_slash_none() -> None:
    assert finra_client._scan_group_name(_finra_entries(), "ShortInterest") is None


def test_finra_scan_group_name_no_match_none() -> None:
    assert finra_client._scan_group_name(_finra_entries(), "OTC/Nope") is None


# --- datacommons: required-field errors + disabled gate --------------------------


def test_datacommons_resolve_entities_nodes_required() -> None:
    out = datacommons.resolve_entities([])
    assert out["status"] == "error" and out["error_type"] == "invalid_params"
    assert "at least one node" in str(out["error"])


def test_datacommons_place_hierarchy_dcid_required() -> None:
    out = datacommons.get_place_hierarchy("  ")
    assert out["status"] == "error" and out["error_type"] == "invalid_params"
    assert "dcid" in str(out["error"])


def test_datacommons_hierarchy_gate_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_DATA_ENABLED", raising=False)
    out = datacommons._hierarchy_gate("geoId/06")
    assert out is not None and out["status"] == "disabled"


# --- geo: metric/finalize/submit/clean/collect arms ------------------------------


def test_geo_noaa_metric_absent_column_none() -> None:
    assert geo_context._noaa_metric({"station_id": "S"}, "temp", "Fahrenheit", "2024-01-01", []) is None


def test_geo_noaa_metric_sentinel_warns_and_nulls() -> None:
    warnings: list[str] = []
    entry = geo_context._noaa_metric({"station_id": "S1", "temp": 9999.9}, "temp", "Fahrenheit", "2024-01-01", warnings)
    assert entry is not None and entry["value"] is None
    assert any("incomplete quality" in w and "temp" in w for w in warnings)


def test_geo_finalize_missing_coverage_warning() -> None:
    out = geo_context._finalize("t", [{"vintage": "2024-01-01"}], 10, None, ["G1"], [])
    assert out["status"] == "ok"
    warnings = out["warnings"]
    assert isinstance(warnings, list)
    assert any("missing-coverage" in str(w) and "G1" in str(w) for w in warnings)


def test_geo_submit_error_usable_none() -> None:
    assert geo_context._submit_error({"status": "ok", "rows": []}) is None


def test_geo_clean_float_bool_passthrough() -> None:
    assert geo_context._clean_float(True) is True
    assert geo_context._clean_float(False) is False
    assert geo_context._clean_float(9999.9) is None
    assert geo_context._clean_float(1.5) == 1.5


def test_geo_collect_noaa_join_error_on_non_dict() -> None:
    err, context, missing, warnings = geo_context._collect_noaa([42], ["S1"])
    assert err is not None and err["error_type"] == "unsupported_join"
    assert context == [] and missing == [] and warnings == []


# --- bigquery: columns/table/billing arms ----------------------------------------


def test_bq_columns_clause_str_input() -> None:
    assert bigquery_client._columns_clause({"columns": "total_pop", "limit": 5}) == "total_pop"


def test_bq_columns_clause_bad_column_value_error() -> None:
    with pytest.raises(ValueError, match="unknown census columns"):
        bigquery_client._columns_clause({"columns": ["bogus_col"], "limit": 5})


def test_bq_resolve_table_from_params_happy() -> None:
    spec = dict(bigquery_client.TEMPLATES["trends_refreshes"])
    assert (
        bigquery_client._resolve_table(spec, {"table": "patents-public-data.patents.publications"})
        == "patents-public-data.patents.publications"
    )


def test_bq_resolve_table_unknown_value_error() -> None:
    spec = dict(bigquery_client.TEMPLATES["trends_refreshes"])
    with pytest.raises(ValueError, match="unknown table"):
        bigquery_client._resolve_table(spec, {"table": "evil.table.x"})


def test_bq_info_billing_dict_arms() -> None:
    assert bigquery_client._info_billing({"billingEnabled": False}) is True
    assert bigquery_client._info_billing({"billingEnabled": True}) is False


def test_bq_api_billing_no_creds_none_no_network() -> None:
    assert bigquery_client._api_billing(object(), "proj") is None


# --- trends: unavailable/scope/refresh arms --------------------------------------


def test_trends_submit_via_client_import_error_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.google_data as _gd_pkg

    monkeypatch.delattr(_gd_pkg, "bigquery_client", raising=False)
    monkeypatch.setitem(sys.modules, "app.google_data.bigquery_client", None)
    out = trends._submit_via_client("t", {"limit": 1}, None)
    assert out["error"] == "bigquery client unavailable"
    assert out["error_type"] == "source_unavailable"


def test_trends_region_geo_base_suffix_arm() -> None:
    assert trends._region_geo_base({"region_code": "CA", "country_code": "US"}, {"country": "US"}) == "US:CA"
    assert trends._region_geo_base({"country_code": "US"}, {"country": "US"}) == "US"


# --- signals: coverage passthrough + version match arms -------------------------


def test_signals_narrow_blob_coverage_preset_passthrough() -> None:
    periods, geos, rows = signals._narrow_blob_coverage({}, ["2024-Q1"], ["US"])
    assert periods == ["2024-Q1"] and geos == ["US"] and rows == []


# --- youtube: ledger/decode/consent/main arms ------------------------------------


def test_youtube_legacy_total_happy() -> None:
    assert youtube._legacy_total({"2024-01-01": 2, "2024-01-02": 3}) == 5


def test_youtube_legacy_total_garbage_refuses() -> None:
    with pytest.raises(youtube._QuotaRefused):
        youtube._legacy_total({"2024-01-01": -1})
    with pytest.raises(youtube._QuotaRefused):
        youtube._legacy_total({"2024-01-01": "lots"})


def test_youtube_decode_request_invalid_json() -> None:
    request, err = youtube._decode_request(b"{not json")
    assert request is None and err == "invalid_params"


def test_youtube_main_action_consent_required() -> None:
    out = youtube._main_action({})
    assert out["status"] == "unavailable" and out["error_type"] == "consent_required"


class _Buf:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data
        self.written = b""

    def read(self, n: int = -1) -> bytes:
        return self._data[:n]

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def flush(self) -> None:
        pass


class _Stream:
    def __init__(self, buf: _Buf) -> None:
        self.buffer = buf


def test_youtube_main_invalid_params_exit_0(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_in, fake_out = _Buf(b"not-json{{{ high"), _Buf()
    monkeypatch.setattr(sys, "stdin", _Stream(fake_in))
    monkeypatch.setattr(sys, "stdout", _Stream(fake_out))
    assert youtube.main() == 0
    assert json.loads(fake_out.written)["error_type"] == "invalid_params"


# --- analyst: label boundaries + empty-result error ------------------------------


@pytest.mark.parametrize(
    ("mean", "label"),
    [(1.5, "Strong Buy"), (2.5, "Buy"), (3.5, "Hold"), (4.5, "Underperform"), (5.0, "Sell")],
)
def test_analyst_label_from_mean_boundaries(mean: float, label: str) -> None:
    assert analyst_client._label_from_mean(mean) == label


def test_analyst_summary_result_list_empty_value_error() -> None:
    with pytest.raises(ValueError):
        analyst_client._summary_result_list({}, "AAA")
    with pytest.raises(ValueError):
        analyst_client._summary_result_list({"result": []}, "AAA")
    assert analyst_client._summary_result_list({"result": [{"a": 1}]}, "AAA") == [{"a": 1}]


# --- robinhood: origin gate + json value + token empty-state ---------------------


def test_robinhood_server_origin_happy() -> None:
    assert rh_auth.server_origin("https://agent.robinhood.com/mcp/trading") == "https://agent.robinhood.com"


def test_robinhood_server_origin_rejects() -> None:
    with pytest.raises(rh_auth.OAuthStoreError):
        rh_auth.server_origin("http://agent.robinhood.com/mcp/trading")
    with pytest.raises(rh_auth.OAuthStoreError):
        rh_auth.server_origin("https://agent.robinhood.com/mcp/trading?x=1")


def test_robinhood_json_value_nested() -> None:
    out = _json_value(
        {
            "d": Decimal("1.25"),
            "ts": datetime(2024, 1, 1, tzinfo=UTC),
            "xs": [Decimal(2), None],
        }
    )
    assert out == {"d": "1.25", "ts": "2024-01-01T00:00:00+00:00", "xs": ["2", None]}


def test_robinhood_tokens_none_when_empty(tmp_path: Path) -> None:
    path = tmp_path / "oauth.json"
    origin = rh_auth.server_origin(rh_auth.ROBINHOOD_MCP_URL)
    assert rh_auth.load_tokens(path) is None
    assert rh_auth.load_tokens_for_origin(origin, path) is None
    path.write_text(json.dumps({"tokens": {"access_token": "x"}}))
    assert rh_auth.load_tokens_for_origin(origin, path) is None


def test_robinhood_provider_builds_with_tmp_path(tmp_path: Path) -> None:
    cfg = rh_auth.OAuthConfig(server_url=rh_auth.ROBINHOOD_MCP_URL)
    assert rh_auth.build_oauth_provider(cfg, path=tmp_path / "oauth.json") is not None


# --- stackoverflow: tags/row arms -------------------------------------------------


def test_stackoverflow_clean_tags_str_input() -> None:
    assert stackoverflow._clean_tags("python") == ["python"]
    assert stackoverflow._clean_tags([" a ", "", "b"]) == [" a ", "b"]


def test_stackoverflow_get_tag_activity_tags_required() -> None:
    out = stackoverflow.get_tag_activity(None)
    assert out["status"] == "error" and out["error_type"] == "invalid_params"


def test_stackoverflow_activity_row_non_dict_none() -> None:
    assert stackoverflow._activity_row(42) is None
    assert stackoverflow._activity_row("nope") is None


# --- sec_facts + trends_api: no-network crafted-payload arms ---------------------


def test_sec_facts_xbrl_counts_empty_matching() -> None:
    assert sec_facts._xbrl_counts({}) == ([], 0)
    assert sec_facts._xbrl_counts({"matching_concepts": ["a", "b"]}) == (["a", "b"], 2)
    assert sec_facts._xbrl_counts({"matching_concepts": ["a"], "count": 5}) == (["a"], 5)


def test_trends_api_terms_required_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    out = trends_api.get_interest_over_time(terms=[])
    assert out["status"] == "error" and out["error_type"] == "invalid_params"


# --- residual CRAP>10 arms (fresh-coverage pass) -----------------------------------
class _FakeBQ:
    """Minimal bigquery fake: dry_run + submit + billing_enabled seam."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows if rows is not None else [{"n": 1}]
        self.billing_enabled: bool | None = False

    def dry_run(self, params: object) -> int:
        return 10

    def submit(self, params: object, max_bytes: int, job_id: str) -> dict[str, object]:
        return {"job_id": job_id, "total_bytes_billed": 10, "rows": list(self._rows), "state": "DONE"}


def _bq_factory(fake: _FakeBQ) -> Callable[[], object]:
    def _make(*args: object, **kwargs: object) -> _FakeBQ:
        return fake

    return _make


def test_bq_submit_happy_covers_submit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable(monkeypatch)
    factory = _bq_factory(_FakeBQ())
    res = bigquery_client.submit_template("trends_top", {"limit": 1}, client_factory=factory, data_root=tmp_path)
    assert res["status"] == "ok" and res["rows"] == [{"n": 1}]


def test_submit_direct_executor_error_all_files() -> None:
    def _boom(template: str, params: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("nope")

    geo_out: dict[str, object] = geo_context._direct_submit("t", {"limit": 1}, _boom)
    assert geo_out["status"] == "error" and geo_out["error_type"] == "executor_error"
    pat_out: dict[str, object] = patents._direct_submit("t", {"limit": 1}, _boom)
    assert pat_out["status"] == "error" and pat_out["error_type"] == "executor_error"
    so_out: dict[str, object] = stackoverflow._direct_submit("t", {"limit": 1}, _boom)
    assert so_out["status"] == "error" and so_out["error_type"] == "executor_error"


def test_geo_submit_error_normalizes_bare_dict() -> None:
    out = geo_context._submit_error({"error": "x", "error_type": "request_failed"})
    assert out is not None and out["status"] == "unavailable" and out["source"] == "geo_context"
    out2 = stackoverflow._submit_error({"error": "x"})
    assert out2 is not None and out2["status"] == "error" and out2["source"] == "stackoverflow"
    out3 = stackoverflow._submit_error({"error": "x", "error_type": "billing_enabled"})
    assert out3 is not None and out3["status"] == "unavailable"


def test_geo_collect_census_happy_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    rows: list[object] = [{"geo_id": "04000US06", "total_pop": 100, "median_age": 37.5}]
    err, context, missing = geo_context._collect_census(rows, ["total_pop"], "2020", ["04000US06", "G-MISS"])
    assert err is None
    assert [c["variable"] for c in context] == ["total_pop"]
    assert missing == ["G-MISS"]


def test_geo_noaa_row_expands_all_metrics() -> None:
    row: dict[str, object] = {"station_id": "S1", "observed_at": "2024-01-02", "temp": 70.0, "frshtt": "0"}
    entries = geo_context._noaa_row(row, [])
    assert {e["variable"] for e in entries} >= {"temp", "frshtt"}
    assert all(e["geo_id"] == "S1" for e in entries)


def test_geo_frshtt_warning_flag_arm() -> None:
    warnings: list[str] = []
    geo_context._frshtt_warning("2", "S9", "2024-01-03", warnings)
    assert any("tornado/hail" in w and "S9" in w for w in warnings)
    quiet: list[str] = []
    geo_context._frshtt_warning("0", "S9", "2024-01-03", quiet)
    assert quiet == []


def test_geo_missing_ids_reports_absent() -> None:
    rows: list[object] = [{"geo_id": "A"}, {"geo_id": "B"}]
    assert geo_context._missing_ids(rows, "geo_id", ["A", "C"]) == ["C"]


def test_geo_full_context_noaa_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows = [{"station_id": "USW1", "observed_at": "2024-01-02", "temp": 70.0, "prcp": 0.1}]
    out = geo_context.get_geo_context(
        ["USW1", "USW-GONE"], variables=["TEMP"], executor=lambda t, p: {"status": "ok", "rows": rows}
    )
    assert out["status"] == "ok"
    missing = out["missing_geos"]
    assert isinstance(missing, list) and "USW-GONE" in missing
    warned = out["warnings"]
    assert isinstance(warned, list) and any("missing-coverage" in str(w) for w in warned)


def test_finra_parse_live_methods_arms() -> None:
    assert finra_client._parse_live_methods(["a", 1]) == ("a", "1")
    assert finra_client._parse_live_methods("a, b ,") == ("a", "b")
    assert finra_client._parse_live_methods(None) == ()
    assert finra_client._metadata_methods(
        finra_client.CatalogEntry(group="g", name="n", description="d"), {"supportedMethods": "x, y"}
    ) == ("x", "y")


def test_bq_make_client_fake_seams() -> None:
    factory: Callable[[], object] = lambda: _FakeBQ()
    err, _client, billing = bigquery_client._make_client(factory, "proj")
    assert err is None and billing is True
    minimal: Callable[[], object] = lambda: object()
    err2, _c2, billing2 = bigquery_client._make_client(minimal, "proj")
    assert err2 is None and billing2 is True


def test_datacommons_resolve_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("DATACOMMONS_API_KEY", "k")

    class _R:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"entities": [{"dcid": "geoId/06"}]}

    def _post(*a: object, **k: object) -> object:
        return _R()

    monkeypatch.setattr(datacommons.requests, "post", _post)
    out = datacommons.resolve_entities(["California"])
    assert out["status"] == "ok" and out["source"] == "datacommons"
    assert isinstance(out["entities"], list)


def test_datacommons_hierarchy_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("DATACOMMONS_API_KEY", "k")

    class _R:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": {"x": 1}}

    def _post2(*a: object, **k: object) -> object:
        return _R()

    monkeypatch.setattr(datacommons.requests, "post", _post2)
    out = datacommons.get_place_hierarchy("geoId/06")
    assert out["status"] == "ok" and out["dcid"] == "geoId/06"
    assert out["hierarchy"] == {"x": 1}


def test_submit_via_client_happy_all_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _ok(template: str, params: dict[str, object], data_root: object = None) -> dict[str, object]:
        return {"status": "ok", "rows": [{"n": 1}]}

    monkeypatch.setattr(bigquery_client, "submit_template", _ok)
    geo_ok: dict[str, object] = geo_context._submit_via_client("t", {"limit": 1}, tmp_path)
    assert geo_ok["status"] == "ok" and geo_ok["rows"] == [{"n": 1}]
    pat_ok: dict[str, object] = patents._submit_via_client("t", {"limit": 1}, tmp_path)
    assert pat_ok["status"] == "ok" and pat_ok["rows"] == [{"n": 1}]
    so_ok: dict[str, object] = stackoverflow._submit_via_client("t", {"limit": 1}, tmp_path)
    assert so_ok["status"] == "ok" and so_ok["rows"] == [{"n": 1}]
    tr_ok: dict[str, object] = trends._submit_via_client("t", {"limit": 1}, tmp_path)
    assert tr_ok["status"] == "ok" and tr_ok["rows"] == [{"n": 1}]


def test_submit_via_client_executor_error_mapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _boom(template: str, params: dict[str, object], data_root: object = None) -> dict[str, object]:
        raise RuntimeError("ledger down")

    monkeypatch.setattr(bigquery_client, "submit_template", _boom)
    geo_err: dict[str, object] = geo_context._submit_via_client("t", {"limit": 1}, tmp_path)
    assert geo_err["status"] == "error" and geo_err["error_type"] == "executor_error"
    pat_err: dict[str, object] = patents._submit_via_client("t", {"limit": 1}, tmp_path)
    assert pat_err["status"] == "error" and pat_err["error_type"] == "executor_error"
    so_err: dict[str, object] = stackoverflow._submit_via_client("t", {"limit": 1}, tmp_path)
    assert so_err["status"] == "error" and so_err["error_type"] == "executor_error"
    tr_err: dict[str, object] = trends._submit_via_client("t", {"limit": 1}, tmp_path)
    assert tr_err["status"] == "error" and tr_err["error_type"] == "executor_error"


def test_submit_via_client_ledger_corrupt_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    LedgerCorrupt = type("LedgerCorrupt", (Exception,), {})

    def _corrupt(template: str, params: dict[str, object], data_root: object = None) -> dict[str, object]:
        raise LedgerCorrupt("bad ledger")

    monkeypatch.setattr(bigquery_client, "submit_template", _corrupt)
    with pytest.raises(LedgerCorrupt):
        geo_context._submit_via_client("t", {"limit": 1}, tmp_path)
    with pytest.raises(LedgerCorrupt):
        patents._submit_via_client("t", {"limit": 1}, tmp_path)
    with pytest.raises(LedgerCorrupt):
        stackoverflow._submit_via_client("t", {"limit": 1}, tmp_path)
    with pytest.raises(LedgerCorrupt):
        trends._submit_via_client("t", {"limit": 1}, tmp_path)


def test_submit_error_status_passthrough() -> None:
    err: dict[str, object] = {"status": "error", "error": "x", "source": "bigquery"}
    assert geo_context._submit_error(err) is err
    assert stackoverflow._submit_error(err) is err


def test_bq_make_client_import_error_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "google.cloud.bigquery", None)
    parent = _sys.modules.get("google.cloud")
    if parent is not None:
        monkeypatch.delattr(parent, "bigquery", raising=False)
    err, _client, billing = bigquery_client._make_client(None, "proj")
    assert err is not None and billing is None
    assert err["error_type"] == "source_unavailable"


def test_signals_narrow_blob_coverage_miss_arms() -> None:
    periods, geos, rows = signals._narrow_blob_coverage({"observations": []}, None, None)
    assert periods is None and geos is None and rows == []
    periods2, geos2, _ = signals._narrow_blob_coverage({}, None, ["EU"])
    assert periods2 is None and geos2 == ["EU"]


def test_bq_make_client_real_import_path_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys as _sys
    import types as _types

    parent = _types.ModuleType("google.cloud")
    setattr(parent, "__path__", ["google-cloud"])  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    child = _types.ModuleType("google.cloud.bigquery")
    seen: dict[str, object] = {}

    class _Client:
        def __init__(self, project: object = None) -> None:
            seen["project"] = project

    setattr(child, "Client", _Client)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(parent, "bigquery", child)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    monkeypatch.setitem(_sys.modules, "google.cloud", parent)
    monkeypatch.setitem(_sys.modules, "google.cloud.bigquery", child)
    err, client, billing = bigquery_client._make_client(None, "proj")
    assert err is None and isinstance(client, _Client)
    assert seen["project"] == "proj" and billing is None


def test_analyst_quote_price_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    assert analyst_client.get_quote_price("  ") == {"price": None, "retrieved_at": None}
    price_row: dict[str, object] = {"price": {"regularMarketPrice": {"raw": 10.5}}}
    fallback_row: dict[str, object] = {"financialData": {"currentPrice": {"raw": 7.25}}}
    empty_row: dict[str, object] = {}

    def _qs(_ticker: str, modules: str) -> dict[str, object]:
        return price_row if modules == "price" else fallback_row

    monkeypatch.setattr(analyst_client, "_quote_summary", _qs)
    direct = analyst_client.get_quote_price("aaa")
    assert direct["price"] == 10.5 and direct["retrieved_at"]

    def _qs_fallback(_ticker: str, modules: str) -> dict[str, object]:
        return empty_row if modules == "price" else fallback_row

    monkeypatch.setattr(analyst_client, "_quote_summary", _qs_fallback)
    fell = analyst_client.get_quote_price("BBB")
    assert fell["price"] == 7.25 and fell["retrieved_at"]

    def _qs_empty(_ticker: str, _modules: str) -> dict[str, object]:
        return empty_row

    monkeypatch.setattr(analyst_client, "_quote_summary", _qs_empty)
    assert analyst_client.get_quote_price("CCC") == {"price": None, "retrieved_at": None}


# --- youtube worker + topic params: decision paths ---------------------------------
def test_youtube_main_action_invalid_thesis_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # blank thesis id refuses before touching analytics
    out = youtube._main_action({"confirmed": True, "thesis_id": "   "})
    assert out["status"] == "unavailable" and out["error_type"] == "invalid_params"
    # defaults (no mode/region/days/limit keys) flow through to the entry point
    seen: dict[str, object] = {}

    def _fake(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(youtube, "get_youtube_analytics", _fake)
    assert youtube._main_action({"confirmed": True, "thesis_id": "t1"}) == {"status": "ok"}
    assert (seen["mode"], seen["region"], seen["days"], seen["limit"]) == (None, None, None, None)
    # bools are not ints: days=True narrows to None, entry point sees None
    seen.clear()
    youtube._main_action({"confirmed": True, "thesis_id": "t1", "days": True, "limit": True})
    assert seen["days"] is None and seen["limit"] is None

    # analytics raise -> source_unavailable (best-effort boundary)
    def _boom(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("down")

    monkeypatch.setattr(youtube, "get_youtube_analytics", _boom)
    out = youtube._main_action({"confirmed": True, "thesis_id": "t1"})
    assert out["status"] == "unavailable" and out["error_type"] == "source_unavailable"


def test_youtube_topic_params_decision_paths() -> None:
    ok_err, clean = youtube._topic_params("  nvda gpus  ", 7)
    assert (ok_err, clean) == (None, "nvda gpus")
    bad: list[tuple[object, object]] = [
        (None, 30),
        ("   ", 30),
        ("x" * 201, 30),
        ("bad\x01query", 30),
        ("bad\x7fquery", 30),
        ("x", 0),
        ("x", 91),
        ("x", None),
        ("x", True),
        ("x", "7"),
        (5, 7),
    ]
    for query, days in bad:
        err, got = youtube._topic_params(query, days)
        assert err is not None and err["error_type"] == "invalid_params" and got is None
