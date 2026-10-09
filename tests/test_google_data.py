"""Focused behavioral contracts for Google public-data (no live network).

Provider fakes sit at the network boundary only: BigQuery via
client_factory fakes (dry_run/submit protocol), HTTP via monkeypatched
requests/session objects. Nothing here invents sibling behavior; every
assertion follows the batch contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests

from app.google_data import bigquery_client as bq
from app.google_data import (
    datacommons,
    geo_context,
    patents,
    signals,
    stackoverflow,
    trends,
    youtube,
)
from app.thesis.models import Thesis

GOOGLE_ENV = (
    "GOOGLE_DATA_ENABLED",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATACOMMONS_API_KEY",
    "GOOGLE_CLOUD_API_KEY",
    "BIGQUERY_MAX_BYTES_PER_QUERY",
    "BIGQUERY_MONTHLY_BYTES_LIMIT",
    "BIGQUERY_DAILY_BYTES_LIMIT",
    "GOOGLE_TRENDS_API_ENABLED",
    "YOUTUBE_SEARCH_DAILY_LIMIT",
)

_Params = dict[str, object]
_Row = dict[str, object]
_Seen = list[tuple[str, _Params]]
_Executor = Callable[[str, _Params], _Row]
_RowsFor = Callable[[str, _Params], list[_Row]]


def _observations(result: _Row) -> list[_Row]:
    """Narrow a collect_trends observations payload for strict typing."""
    obs = result.get("observations")
    assert isinstance(obs, list)
    return obs


def _metrics(row: _Row) -> _Row:
    """Narrow a signal row metrics payload for strict typing."""
    metrics = row.get("metrics")
    assert isinstance(metrics, dict)
    return metrics


def _child(row: _Row, key: str) -> _Row:
    """Narrow one nested mapping level for strict typing."""
    value = row.get(key)
    assert isinstance(value, dict)
    return value


def _nested(row: _Row, *keys: str) -> object:
    """Read nested mapping levels, asserting each level is a dict."""
    value: object = row
    for key in keys:
        assert isinstance(value, dict)
        step: object = value.get(key)
        value = step
    return value


def _metrics_or_empty(row: _Row) -> _Row:
    """Metrics mapping, or {} when absent (mirrors the `(m or {})` default)."""
    metrics = row.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _start_date(params: _Params) -> str:
    """Narrow the executor-seam start_date for strict typing."""
    start = params["start_date"]
    assert isinstance(start, str)
    return start


def _first_inputs_hash(row: _Row) -> object:
    """Narrow the first available feature scope's inputs_hash."""
    scopes = row.get("available_feature_scopes")
    assert isinstance(scopes, list)
    first = scopes[0]
    assert isinstance(first, dict)
    first_hash: object = first.get("inputs_hash")
    return first_hash


@pytest.fixture(autouse=True)
def _clean_google_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in GOOGLE_ENV:
        monkeypatch.delenv(key, raising=False)


def _enable(monkeypatch: pytest.MonkeyPatch, project: str = "test-proj") -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)


class FakeBQ:
    """Boundary fake: dry_run(params)->bytes, submit(params, max_bytes, job_id)."""

    def __init__(
        self,
        total_bytes: int = 10,
        rows: list[_Row] | None = None,
        state: str = "DONE",
        billed: int | None = None,
        billing_enabled: bool | None = False,
    ) -> None:
        self.total_bytes = total_bytes
        self._rows = rows if rows is not None else [{"n": 1}]
        self.state = state
        self.billed = total_bytes if billed is None else billed
        self.billing_enabled = billing_enabled
        self.submits: list[_Row] = []
        self.dry_runs: list[_Row] = []

    def dry_run(self, params: _Params) -> int:
        self.dry_runs.append(dict(params))
        return self.total_bytes

    def submit(self, params: _Params, max_bytes: int, job_id: str) -> _Row:
        self.submits.append({"params": dict(params), "max_bytes": max_bytes, "job_id": job_id})
        return {"job_id": job_id, "total_bytes_billed": self.billed, "rows": list(self._rows), "state": self.state}


def _factory(fake: FakeBQ) -> Callable[..., FakeBQ]:
    def _make(*args: object, **kwargs: object) -> FakeBQ:
        return fake

    return _make


def _ledger(tmp_path: Path) -> _Row:
    raw: object = json.loads((tmp_path / "google_data" / "bq_ledger.json").read_text())
    assert isinstance(raw, dict)
    return {str(k): v for k, v in raw.items()}


class _Resp:
    def __init__(self, payload: _Row, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> _Row:
        return self._payload

    def raise_for_status(self) -> None:
        return None


# BigQuery executor ------------------------------------------------------


def test_billing_enabled_refuses_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    fake = FakeBQ(billing_enabled=True)
    result = bq.submit_template("trends_top", {"limit": 5}, client_factory=_factory(fake), data_root=tmp_path)
    assert result["error_type"] == "billing_enabled"
    assert result["source"] == "bigquery"
    assert fake.submits == [] and fake.dry_runs == []


def test_billing_unknown_refuses_submit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    fake = FakeBQ(billing_enabled=None)
    result = bq.submit_template("trends_top", {"limit": 5}, client_factory=_factory(fake), data_root=tmp_path)
    assert result["error_type"] == "billing_unknown"
    assert fake.submits == []


def test_check_billing_disabled_states() -> None:
    assert bq.check_billing_disabled({"billingEnabled": False}) is True
    assert bq.check_billing_disabled({"billingEnabled": True}) is False
    assert bq.check_billing_disabled({}) is False
    assert bq.check_billing_disabled(FakeBQ(billing_enabled=False)) is True
    assert bq.check_billing_disabled(FakeBQ(billing_enabled=True)) is False


def test_dry_run_above_cap_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("BIGQUERY_MAX_BYTES_PER_QUERY", "1000")
    fake = FakeBQ(total_bytes=5000)
    result = bq.submit_template("trends_top", {"limit": 5}, client_factory=_factory(fake), data_root=tmp_path)
    assert result["error_type"] == "cost_limit_exceeded"
    assert fake.submits == []


def test_monthly_ceiling_includes_pending(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("BIGQUERY_MONTHLY_BYTES_LIMIT", "3000")
    fake = FakeBQ(total_bytes=2000)
    first = bq.submit_template("trends_top", {"k": 1}, client_factory=_factory(fake), data_root=tmp_path)
    assert first["status"] == "ok"
    second = bq.submit_template("trends_top", {"k": 2}, client_factory=_factory(fake), data_root=tmp_path)
    assert second["error_type"] == "monthly_limit_exceeded"
    assert len(fake.submits) == 1


def test_retry_reuses_job_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    fake = FakeBQ(total_bytes=100)
    first = bq.submit_template("trends_top", {"k": 1}, client_factory=_factory(fake), data_root=tmp_path)
    second = bq.submit_template("trends_top", {"k": 1}, client_factory=_factory(fake), data_root=tmp_path)
    assert first["job_id"] == second["job_id"]
    assert len(fake.submits) == 1
    jobs = _ledger(tmp_path)["jobs"]
    assert isinstance(jobs, dict)
    assert len(jobs) == 1


def test_corrupt_ledger_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    root = tmp_path / "google_data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "bq_ledger.json").write_text("not json{{{")
    with pytest.raises(bq.LedgerCorrupt):
        bq.submit_template("trends_top", {"k": 1}, client_factory=_factory(FakeBQ()), data_root=tmp_path)
    (root / "bq_ledger.json").write_text(json.dumps({"jobs": [], "months": {}}))
    with pytest.raises(bq.LedgerCorrupt):
        bq.submit_template("trends_top", {"k": 1}, client_factory=_factory(FakeBQ()), data_root=tmp_path)


def test_cross_month_retains_unresolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    fake = FakeBQ(total_bytes=100, state="RUNNING")
    bq.submit_template("trends_top", {"k": "old"}, client_factory=_factory(fake), data_root=tmp_path)
    path = tmp_path / "google_data" / "bq_ledger.json"
    ledger = json.loads(path.read_text())
    assert len(ledger["jobs"]) == 1
    old_id = next(iter(ledger["jobs"]))
    ledger["jobs"][old_id]["month"] = "2000-01"
    path.write_text(json.dumps(ledger))
    done_fake = FakeBQ(total_bytes=50)
    second = bq.submit_template("trends_top", {"k": "new"}, client_factory=_factory(done_fake), data_root=tmp_path)
    assert second["status"] == "ok"
    kept = json.loads(path.read_text())["jobs"]
    assert kept[old_id]["month"] == "2000-01"
    assert len(kept) == 2


def test_allowlist_rejects_unknown_template(tmp_path: Path) -> None:
    result = bq.submit_template("definitely_not_a_template", {}, client_factory=_factory(FakeBQ()), data_root=tmp_path)
    assert "error" in result
    assert result["source"] == "bigquery"


def test_patent_template_over_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("BIGQUERY_MAX_BYTES_PER_QUERY", "1000")
    fake = FakeBQ(total_bytes=10**9)
    result = bq.submit_template(
        "patents_assignee",
        {"assignees": ["Acme"], "start_date": "2020-01-01", "end_date": "2026-01-01", "limit": 5},
        client_factory=_factory(fake),
        data_root=tmp_path,
    )
    assert result["error_type"] == "cost_limit_exceeded"
    assert fake.submits == []


# Signals math + identity -------------------------------------------------

PERIODS = ["2026-W01", "2026-W02", "2026-W03", "2026-W04"]
GEOS = ["US", "GB", "DE", "FR"]


def _math_batch() -> list[_Row]:
    rows: list[_Row] = []
    for period in PERIODS:
        for geo in GEOS:
            hit = period in PERIODS[:3] and geo in GEOS[:2]
            rows.append(
                {
                    "table": "trends_top",
                    "period": period,
                    "geo": geo,
                    "term": "alpha" if hit else None,
                    "list_kind": "rising",
                    "rank": 5 if hit else None,
                    "present": hit,
                }
            )
    rows.append(
        {
            "table": "trends_top",
            "period": "2026-W01",
            "geo": "US",
            "term": "alpha",
            "list_kind": "rising",
            "rank": 5,
            "present": True,
        }
    )  # duplicate: must not inflate
    return rows


def test_candidate_math_persistence_diffusion() -> None:
    features = signals.compute_candidate_features(_math_batch())
    assert features["persistence"] == pytest.approx(0.75)
    assert features["diffusion"] == pytest.approx(0.5)


def test_rank_improvement_within_same_table_list_geo() -> None:
    rows: list[dict[str, object]] = [
        {"table": "t", "period": "2026-W01", "geo": "US", "term": "alpha", "list_kind": "rising", "rank": 9},
        {"table": "t", "period": "2026-W02", "geo": "US", "term": "alpha", "list_kind": "rising", "rank": 4},
    ]
    assert signals.compute_candidate_features(rows)["rank_improvement"] == 5


def test_normalize_signal_id_and_first_known_at() -> None:
    record: dict[str, object] = {
        "table": "t",
        "period": "2026-W01",
        "geo": "US",
        "term": "alpha",
        "list_kind": "rising",
        "rank": 3,
    }
    first = signals.normalize_candidate(record, retrieved_at="2026-09-01T00:00:00+00:00")
    assert first["signal_id"] == hashlib.sha256(b"t|2026-W01|US|alpha|rising").hexdigest()
    assert first["status"] == "candidate"
    recollected = signals.normalize_candidate(
        record, retrieved_at="2026-09-02T00:00:00+00:00", known_at=first["known_at"]
    )
    assert recollected["signal_id"] == first["signal_id"]
    assert recollected["known_at"] == first["known_at"]


def _trend_rows() -> list[_Row]:
    return [
        {"table": "trends_top", "period": "2026-W01", "geo": "US", "term": "alpha", "list_kind": "top", "rank": 3},
        {"table": "trends_rising", "period": "2026-W01", "geo": "US", "term": "beta", "list_kind": "rising", "rank": 1},
    ]


def _trend_executor(rows: list[_Row]) -> _Executor:
    def _run(template: str, params: _Params) -> _Row:
        return {"status": "ok", "rows": [dict(r) for r in rows], "source": "bigquery"}

    return _run


def test_trends_usable_without_other_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    result = trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["US"],
        limit=10,
        data_root=tmp_path,
        executor=_trend_executor(_trend_rows()),
    )
    assert result["status"] == "ok"
    assert result["source"] == "trends"
    assert "alpha" in json.dumps(result)


def test_asof_excludes_future_evidence_and_recollect_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable(monkeypatch)
    run = lambda: trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["US"],
        limit=10,
        data_root=tmp_path,
        executor=_trend_executor(_trend_rows()),
    )
    first, second = run(), run()
    assert first["status"] == second["status"] == "ok"
    assert [o["signal_id"] for o in _observations(first)] == [o["signal_id"] for o in _observations(second)]
    assert len({o["signal_id"] for o in _observations(first)}) == len(_observations(first))


# Trends durability regressions -------------------------------------------------


def _scoped_trend_executor(*, refreshes: list[str], rows_for: _RowsFor, seen: _Seen) -> _Executor:
    """Params-aware executor: partitions for trends_refreshes, per-refresh rows otherwise."""

    def _run(template: str, params: _Params) -> _Row:
        seen.append((template, dict(params)))
        if template == "trends_refreshes":
            return {"status": "ok", "rows": [{"refresh_date": r} for r in refreshes], "source": "bigquery"}
        return {"status": "ok", "rows": [dict(r) for r in rows_for(template, dict(params))], "source": "bigquery"}

    return _run


def _dma_row(
    dma: str,
    term: str = "alpha",
    week: str = "2026-08-24",
    refresh: str = "2026-09-02",
    rank: int = 1,
    score: float = 90,
    kind: str = "top",
) -> _Row:
    return {
        "term": term,
        "week": week,
        "refresh_date": refresh,
        "dma_name": dma,
        "rank": rank,
        "score": score,
        "list_kind": kind,
    }


def _three_dma_rows() -> list[_Row]:
    return [
        _dma_row(dma, term, refresh="2026-09-02", rank=i + 1, score=90 - 10 * i)
        for i, (dma, term) in enumerate([("Chicago", "alpha"), ("Los Angeles", "beta"), ("New York", "gamma")])
    ]


def _national_row(
    term: str = "alpha",
    week: str = "2026-08-24",
    refresh: str = "2026-09-02",
    dma_count: int = 2,
    score: float = 85.0,
    kind: str = "top",
) -> _Row:
    return {
        "term": term,
        "week": week,
        "refresh_date": refresh,
        "dma_count": dma_count,
        "score": score,
        "list_kind": kind,
    }


def _intl_national_row(
    term: str = "alpha",
    week: str = "2026-08-24",
    refresh: str = "2026-09-02",
    country: str = "GB",
    region_count: int = 5,
    score: float = 80.0,
    kind: str = "top",
) -> _Row:
    return {
        "term": term,
        "week": week,
        "refresh_date": refresh,
        "country_code": country,
        "region_count": region_count,
        "score": score,
        "list_kind": kind,
    }


def test_plan_groups_keeps_national_us_with_other_geos() -> None:
    assert trends._plan_groups(["US", "GB"]) == [
        {"kind": "us", "national": True, "dmas": []},
        {"kind": "intl", "country": "GB"},
    ]
    assert trends._plan_groups(["US", "New York"]) == [
        {"kind": "us", "national": True, "dmas": []},
        {"kind": "us", "national": False, "dmas": ["New York"]},
    ]
    assert trends._plan_groups(["US", "GB", "New York"]) == [
        {"kind": "us", "national": True, "dmas": []},
        {"kind": "us", "national": False, "dmas": ["New York"]},
        {"kind": "intl", "country": "GB"},
    ]


def test_national_rollup_emits_every_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-01", "2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top_national":
            return []
        refresh = _start_date(params)
        assert refresh in refreshes
        assert "dmas" not in params and "all_dmas" not in params
        return [
            _national_row("alpha", refresh=refresh, dma_count=2, score=85.0),
            _national_row("beta", refresh=refresh, dma_count=3, score=70.0),
        ]

    result = trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["US"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    assert {t for t, _ in seen if t != "trends_refreshes"} <= {"trends_us_top_national", "trends_us_rising_national"}
    by_refresh: dict[str, list[_Row]] = {}
    for obs in _observations(result):
        assert obs["geo"] == "US"
        assert obs.get("rank") is None
        assert _metrics_or_empty(obs).get("rank") is None
        by_refresh.setdefault(str(_metrics(obs).get("refresh_date")), []).append(obs)
    assert sorted(by_refresh) == refreshes
    for refresh, obs_list in by_refresh.items():
        by_term = {o["term"]: o for o in obs_list}
        assert _metrics(by_term["alpha"])["dma_count"] == 2
        assert _metrics(by_term["alpha"])["score"] == pytest.approx(85.0)
        assert _metrics(by_term["beta"])["dma_count"] == 3
        assert _metrics(by_term["beta"])["score"] == pytest.approx(70.0)
        assert all(refresh in str(o["source_record_id"]) for o in obs_list)
        for obs in obs_list:
            assert _metrics(obs)["score_basis"] == "mean_list_score_where_listed"
            assert _child(obs, "features")["diffusion"] is None
            assert _nested(obs, "features", "coverage", "missing", "diffusion") == "subregion rows aggregated"
            assert _nested(obs, "features", "rules", "diffusion", "value") is None


def test_intl_national_nulls_diffusion_and_marks_score_basis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_intl_top_national":
            return []
        return [_intl_national_row("alpha", refresh="2026-09-02", country="GB", region_count=5, score=80.0)]

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["GB"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    assert result["observations"]
    for obs in _observations(result):
        assert _metrics(obs)["region_count"] == 5
        assert _metrics(obs)["score_basis"] == "mean_list_score_where_listed"
        assert _child(obs, "features")["diffusion"] is None
        assert _nested(obs, "features", "coverage", "missing", "diffusion") == "subregion rows aggregated"
        assert _nested(obs, "features", "rules", "diffusion", "value") is None


def test_explicit_dma_keeps_numeric_diffusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        return [_dma_row("New York", refresh=_start_date(params), rank=1)]

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    assert result["observations"]
    for obs in _observations(result):
        assert _child(obs, "features")["diffusion"] == pytest.approx(1.0)
        assert "score_basis" not in _metrics_or_empty(obs)


def test_national_default_scope_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _no_rows(template: str, params: _Params) -> list[_Row]:
        return []

    result = trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["US"],
        limit=10,
        data_root=tmp_path,
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_no_rows, seen=seen),
    )
    assert result["status"] == "ok"
    data_calls = [(t, p) for t, p in seen if t != "trends_refreshes"]
    assert data_calls
    assert {t for t, _ in data_calls} <= {"trends_us_top_national", "trends_us_rising_national"}
    for _, params in data_calls:
        assert params["limit"] == 1001
        assert "dmas" not in params and "all_dmas" not in params
        assert params["week_end"] == "2026-09-02"
        assert params["week_start"] == "2026-08-20"


def test_national_sentinel_fires_post_aggregation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    big_us = [_national_row(term=f"term-{i}", refresh="2026-09-02", dma_count=210, score=50.0) for i in range(1001)]
    big_gb = [
        _intl_national_row(term=f"term-{i}", refresh="2026-09-02", country="GB", region_count=12, score=50.0)
        for i in range(1001)
    ]

    def _collect(geos: list[str], big: list[_Row], template: str) -> tuple[_Row, _Seen]:
        seen: _Seen = []

        def _rows(t: str, params: _Params) -> list[_Row]:
            if t == template:
                return [dict(r) for r in big]
            return []

        return trends.collect_trends(
            start_date="2026-09-02",
            end_date="2026-09-02",
            geos=geos,
            limit=10,
            data_root=tmp_path,
            week_start="2026-08-01",
            week_end="2026-08-31",
            executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
        ), seen

    for geos, big, template in (
        (["US"], big_us, "trends_us_top_national"),
        (["GB"], big_gb, "trends_intl_top_national"),
    ):
        result, seen = _collect(geos, big, template)
        assert result["status"] == "unavailable"
        assert result["reason"] == "query_scope_too_large"
        assert result["error_type"] == "missing_coverage"
        assert template in {t for t, _ in seen}


def test_national_ok_below_sentinel_preserves_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template == "trends_us_top_national":
            return [
                _national_row(term=f"term-{i}", refresh="2026-09-02", dma_count=210, score=50.0) for i in range(1000)
            ]
        return []

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["US"],
        limit=1000,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    assert result["count"] == 1000
    assert all(_metrics_or_empty(o).get("dma_count") == 210 for o in _observations(result))


def test_explicit_dma_keeps_row_level_sql(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        return [_dma_row("New York", refresh=_start_date(params), rank=1)]

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    data_calls = [(t, p) for t, p in seen if t != "trends_refreshes"]
    assert data_calls
    assert {t for t, _ in data_calls} <= {"trends_us_top", "trends_us_rising"}
    top_calls = [p for t, p in data_calls if t == "trends_us_top"]
    assert top_calls and all(p["all_dmas"] is False for p in top_calls)
    assert all(p["week_start"] == "2026-08-01" and p["week_end"] == "2026-08-31" for p in top_calls)
    assert {o["geo"] for o in _observations(result)} == {"New York"}


def test_limit_truncates_response_not_retained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        return _three_dma_rows()

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["Chicago", "Los Angeles", "New York"],
        limit=1,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    data_calls = [params for template, params in seen if template != "trends_refreshes"]
    assert data_calls and all(params["limit"] == 1001 for params in data_calls)
    assert result["count"] == 1 and len(_observations(result)) == 1
    warnings = result["warnings"]
    assert isinstance(warnings, list)
    assert result["continuation"] is True and "truncated" in warnings


def test_term_filter_applies_before_limit_slice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []
    dmas = ["Chicago", "Los Angeles", "New York"]

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        fillers = [
            _dma_row(dmas[i % 3], term=f"filler-{i:03d}", rank=i + 1, score=90, refresh="2026-09-02")
            for i in range(100)
        ]
        return fillers + [_dma_row("New York", term="Stanley Cup target", rank=101, score=90, refresh="2026-09-02")]

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["Chicago", "Los Angeles", "New York"],
        limit=10,
        term="Stanley",
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen),
    )
    assert result["status"] == "ok"
    assert result["count"] == 1 and len(_observations(result)) == 1
    assert "Stanley" in str(_observations(result)[0]["term"])
    assert result["continuation"] is False

    unfiltered = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["Chicago", "Los Angeles", "New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
    )
    assert unfiltered["count"] == 10
    assert unfiltered["continuation"] is True


def test_query_scope_over_1000_never_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    seen: _Seen = []
    calls: dict[tuple[str, str], int] = {}
    big = [
        _dma_row("New York", term=f"term-{i}", week="2026-08-24", refresh="2026-09-02", rank=i + 1, score=90)
        for i in range(1001)
    ]

    def _run(template: str, params: _Params) -> _Row:
        seen.append((template, dict(params)))
        if template == "trends_refreshes":
            return {"status": "ok", "rows": [{"refresh_date": r} for r in refreshes], "source": "bigquery"}
        key = (template, json.dumps(params, sort_keys=True, default=str))
        n = calls.get(key, 0)
        calls[key] = n + 1
        job_id = f"job-{template}-{params['start_date']}"
        if n == 0:
            rows: list[_Row] = big if template == "trends_us_top" else []
            return {"status": "ok", "rows": [dict(r) for r in rows], "source": "bigquery", "job_id": job_id}
        return {"status": "ok", "cached": True, "job_id": job_id, "rows": list[_Row](), "source": "bigquery"}

    def _collect() -> _Row:
        return trends.collect_trends(
            start_date="2026-09-02",
            end_date="2026-09-02",
            geos=["New York"],
            limit=10,
            data_root=tmp_path,
            week_start="2026-08-01",
            week_end="2026-08-31",
            executor=_run,
        )

    first, second = _collect(), _collect()
    for result in (first, second):
        assert result["status"] == "unavailable"
        assert result["error_type"] == "missing_coverage"
        assert result["reason"] == "query_scope_too_large"


def test_feature_calculation_uses_latest_refresh_per_week(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-01", "2026-09-02"]
    seen: _Seen = []
    w1, w2, w3, w4 = "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template == "trends_us_top":
            refresh = _start_date(params)
            if refresh == "2026-09-01":
                return [
                    _dma_row("New York", term="alpha", week=w, refresh=refresh, rank=r, score=s)
                    for w, r, s in [(w1, 4, 10), (w2, 3, 20), (w3, 2, 30), (w4, 1, 100)]
                ]
            if refresh == "2026-09-02":
                return [_dma_row("New York", term="alpha", week=w4, refresh=refresh, rank=5, score=40)]
            return []
        if template == "trends_us_rising":
            refresh = _start_date(params)
            score = 70 if refresh == "2026-09-01" else 71
            return [_dma_row("New York", term="control", week=w4, refresh=refresh, rank=1, score=score, kind="rising")]
        return []

    executor = _scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=seen)
    result = trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=executor,
    )
    assert result["status"] == "ok"
    alpha = [o for o in _observations(result) if o.get("term") == "alpha"]
    assert alpha
    features = _child(alpha[0], "features")
    assert features["velocity"] == pytest.approx(7.5)
    assert features["acceleration"] == pytest.approx(0.0)
    assert features["rank_improvement"] == -3
    assert features["percentile"] == pytest.approx(1.0)
    week4 = [o for o in _observations(result) if o.get("period") == w4 and o.get("term") == "alpha"]
    assert len(week4) == 2
    by_refresh = {_metrics(o).get("refresh_date"): o for o in week4}
    assert set(by_refresh) == {"2026-09-01", "2026-09-02"}
    assert _metrics(by_refresh["2026-09-02"]).get("score") == 40
    seen.clear()
    replayed = trends.collect_trends(
        start_date="2026-09-01",
        end_date="2026-09-02",
        geos=["New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=executor,
    )
    assert replayed["status"] == "ok"

    def _strip(o: _Row) -> _Row:
        return {k: v for k, v in o.items() if k not in ("known_at", "retrieved_at")}

    def _skey(o: _Row) -> tuple[object, ...]:
        key_metrics = o.get("metrics")
        key_refresh = key_metrics.get("refresh_date") if isinstance(key_metrics, dict) else None
        return (o.get("signal_id"), o.get("period"), str(key_refresh), o.get("term"))

    assert sorted((_strip(o) for o in _observations(replayed)), key=_skey) == sorted(
        (_strip(o) for o in _observations(result)), key=_skey
    )
    replayed_alpha = [o for o in _observations(replayed) if o.get("term") == "alpha"]
    assert replayed_alpha and (replayed_alpha[0].get("features") or {}) == features
    assert seen and {t for t, _ in seen} >= {"trends_refreshes", "trends_us_top"}


def test_trend_features_stable_across_recollect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    w1, w2, w3, w4 = "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        return [
            _dma_row("New York", term="alpha", week=w, refresh=_start_date(params), rank=r, score=s)
            for w, r, s in [(w1, 4, 10), (w2, 3, 20), (w3, 2, 30), (w4, 1, 50)]
        ]

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["New York"],
        limit=10,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
    )
    assert result["status"] == "ok"
    assert result["observations"]
    collected = {str(o["term"]): _child(o, "features") for o in _observations(result)}
    assert collected["alpha"]
    for name, rule in _child(collected["alpha"], "rules").items():
        assert isinstance(rule, dict)
        assert rule["rule"] == f"trend_{name}_v2"
        assert rule["calc_version"] == "2"


def test_feature_calculation_isolates_mixed_geographies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    w1, w2, w3, w4 = "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"
    weeks = [w1, w2, w3, w4]
    us_scores = [10, 20, 30, 50]
    ny_scores, ny_ranks = [100, 70, 60, 40], [1, 2, 3, 4]
    gb_scores = [5, 5, 10, 30]

    def _rows(template: str, params: _Params) -> list[_Row]:
        refresh = _start_date(params)
        if template == "trends_us_top_national":
            return [
                _national_row("alpha", week=w, refresh=refresh, dma_count=2, score=s) for w, s in zip(weeks, us_scores)
            ]
        if template == "trends_us_top":
            return [
                _dma_row("New York", term="alpha", week=w, refresh=refresh, rank=r, score=s)
                for w, r, s in zip(weeks, ny_ranks, ny_scores)
            ]
        if template == "trends_intl_top_national":
            return [
                _intl_national_row("alpha", week=w, refresh=refresh, country="GB", region_count=5, score=s)
                for w, s in zip(weeks, gb_scores)
            ]
        return []

    result = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["US", "New York", "GB"],
        limit=20,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
    )
    assert result["status"] == "ok"
    by_geo: dict[str, list[_Row]] = {}
    for obs in _observations(result):
        by_geo.setdefault(str(obs["geo"]), []).append(obs)
    assert set(by_geo) == {"US", "New York", "GB"}
    us_features = _child(by_geo["US"][0], "features")
    ny_features = _child(by_geo["New York"][0], "features")
    gb_features = _child(by_geo["GB"][0], "features")
    assert us_features["velocity"] == pytest.approx(10.0)
    assert ny_features["velocity"] == pytest.approx(-15.0)
    assert gb_features["velocity"] == pytest.approx(6.25)
    assert us_features["acceleration"] == pytest.approx(5.0)
    assert ny_features["acceleration"] == pytest.approx(5.0)
    assert gb_features["acceleration"] == pytest.approx(10.0)
    assert us_features["percentile"] == pytest.approx(1.0)
    assert ny_features["percentile"] == pytest.approx(0.25)
    assert gb_features["percentile"] == pytest.approx(1.0)
    assert us_features["rank_improvement"] is None
    assert ny_features["rank_improvement"] == -1
    assert gb_features["rank_improvement"] is None
    assert us_features["diffusion"] is None
    assert _nested(us_features, "rules", "diffusion", "value") is None
    assert _nested(us_features, "coverage", "missing", "diffusion") == "subregion rows aggregated"
    assert _metrics(by_geo["US"][0])["dma_count"] == 2
    assert gb_features["diffusion"] is None
    assert _nested(gb_features, "rules", "diffusion", "value") is None
    assert _nested(gb_features, "coverage", "missing", "diffusion") == "subregion rows aggregated"
    assert _metrics(by_geo["GB"][0])["region_count"] == 5
    assert ny_features["diffusion"] == pytest.approx(1.0)
    assert _nested(ny_features, "coverage", "geos_covered") == ["New York"]


def test_scope_change_does_not_duplicate_source_observations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)

    refreshes = ["2026-09-02"]
    w1, w2, w3, w4 = "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"
    weeks = [w1, w2, w3, w4]

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        refresh = _start_date(params)
        return [
            _dma_row("New York", term="alpha", week=w, refresh=refresh, rank=r, score=s)
            for w, r, s in zip(weeks, [4, 3, 2, 1], [10, 20, 30, 50])
        ]

    first = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["New York"],
        limit=20,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
    )
    assert first["status"] == "ok"
    second = trends.collect_trends(
        start_date="2026-09-02",
        end_date="2026-09-02",
        geos=["New York", "Los Angeles"],
        limit=20,
        data_root=tmp_path,
        week_start="2026-08-01",
        week_end="2026-08-31",
        executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
    )
    assert second["status"] == "ok"
    ny_first = {o["signal_id"] for o in _observations(first) if o.get("geo") == "New York"}
    ny_second = {o["signal_id"] for o in _observations(second) if o.get("geo") == "New York"}
    assert ny_first and ny_first == ny_second


def test_checkpoint_replay_is_scope_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable(monkeypatch)
    refreshes = ["2026-09-02"]
    w1, w2, w3, w4 = "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"
    weeks = [w1, w2, w3, w4]

    def _rows(template: str, params: _Params) -> list[_Row]:
        if template != "trends_us_top":
            return []
        refresh = _start_date(params)
        return [
            _dma_row("New York", term="alpha", week=w, refresh=refresh, rank=r, score=s)
            for w, r, s in zip(weeks, [4, 3, 2, 1], [10, 20, 30, 50])
        ]

    def _collect(geos: list[str]) -> _Row:
        return trends.collect_trends(
            start_date="2026-09-02",
            end_date="2026-09-02",
            geos=geos,
            limit=20,
            data_root=tmp_path,
            week_start="2026-08-01",
            week_end="2026-08-31",
            executor=_scoped_trend_executor(refreshes=refreshes, rows_for=_rows, seen=[]),
        )

    assert _collect(["New York"])["status"] == "ok"
    assert _collect(["New York", "Los Angeles"])["status"] == "ok"
    replayed = _collect(["New York"])
    assert replayed["status"] == "ok"
    ny_obs = [o for o in _observations(replayed) if o.get("geo") == "New York"]
    assert ny_obs
    for obs in ny_obs:
        assert _child(obs, "features")["diffusion"] == pytest.approx(1.0)
        assert _nested(obs, "features", "coverage", "geos_covered") == ["New York"]


# Entity + macro ------------------------------------------------------------
# Migration note: Knowledge Graph entity resolution was removed from the
# research path (no GOOGLE_KG_API_KEY; investigation uses SEC-confirmed
# entity mappings). app/google_data/knowledge_graph.py was deleted; no KG
# behavior is pinned here.


def _dc_payload() -> _Row:
    return {
        "byVariable": {
            "Count_Person": {
                "byEntity": {
                    "geoId/06": {
                        "orderedFacets": [
                            {"facetId": "f1", "observations": [{"date": "2020", "value": 1.0}]},
                            {"facetId": "f2", "observations": [{"date": "2020", "value": 2.0}]},
                        ]
                    }
                }
            }
        },
        "facets": {"f1": {"unit": "people", "importName": "US Census"}, "f2": {"unit": "USD", "importName": "BEA"}},
    }


def test_datacommons_two_facets_stay_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("DATACOMMONS_API_KEY", "dc-test")
    seen: dict[str, object] = {}

    def _post(url: str, **kwargs: object) -> _Resp:
        seen.update(kwargs)
        seen["url"] = url
        return _Resp(_dc_payload())

    monkeypatch.setattr(requests, "post", _post)
    result = datacommons.get_macro_context(["geoId/06"], ["Count_Person"])
    assert result["status"] == "ok"
    series = result["series"]
    assert isinstance(series, list)
    assert len(series) == 2
    by_facet = {s["facet"]: s for s in series}
    assert by_facet["f1"]["unit"] == "people"
    assert by_facet["f2"]["unit"] == "USD"
    assert by_facet["f1"]["provider"] == "US Census"
    assert by_facet["f2"]["provider"] == "BEA"
    assert seen["url"] == "https://api.datacommons.org/v2/observation"
    assert seen["headers"] == {"X-API-Key": "dc-test"}
    assert "params" not in seen


def test_datacommons_missing_key_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    calls: list[tuple[object, ...]] = []

    def _fail_post(*a: object, **k: object) -> _Resp:
        calls.append(a)
        return _Resp({})

    monkeypatch.setattr(requests, "post", _fail_post)
    result = datacommons.get_macro_context(["geoId/06"], ["Count_Person"])
    assert result["status"] == "unavailable"
    assert result["error"] == "DATACOMMONS_AUTH_REQUIRED"
    assert result["error_type"] == "auth_required"
    assert calls == []


# YouTube analytics (memory-only, thesis-bound) ---------------------------------


def _yt_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "test-key")


def _yt_thesis(tmp_path: Path) -> Thesis:
    from app.thesis.repository import ThesisRepository

    repo = ThesisRepository(tmp_path / "thesis")
    return repo.create_thesis("NVDA demand stays strong", scope="NVDA", claims=["demand holds"])


class _YtPipe:
    """Streaming HTTP fake: status_code + iter_content + close."""

    def __init__(self, payload: dict[str, object] | None = None, status: int = 200, raw: bytes | None = None) -> None:
        self.status_code = status
        self._raw = raw if raw is not None else json.dumps(payload if payload is not None else {}).encode()
        self.closed = False

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for i in range(0, len(self._raw), chunk_size):
            yield self._raw[i : i + chunk_size]

    def close(self) -> None:
        self.closed = True


def _yt_search_item(video_id: str, title: str, channel: str, live: str = "none") -> _Row:
    return {
        "id": {"videoId": video_id},
        "snippet": {
            "title": title,
            "channelId": "chan-" + video_id,
            "channelTitle": channel,
            "publishedAt": "2026-01-01T00:00:00Z",
            "liveBroadcastContent": live,
        },
    }


def _yt_chart_item(video_id: str, title: str, channel: str, live: str = "none") -> _Row:
    snip = _yt_search_item(video_id, title, channel, live)["snippet"]
    return {"id": video_id, "snippet": snip}


def test_youtube_disabled_before_network_or_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _get(url: str, **kwargs: object) -> _YtPipe:
        calls.append(url)
        return _YtPipe()

    monkeypatch.setattr(requests, "get", _get)
    thesis = _yt_thesis(tmp_path)
    before = {p for p in tmp_path.rglob("*")}
    for mode, query in (("topic", "nvda"), ("popular", None)):
        result = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, data_root=tmp_path, mode=mode, query=query)
        assert result == {"status": "disabled", "source": "youtube", "reason": "google_disabled"}
    assert calls == []
    assert {p for p in tmp_path.rglob("*")} == before
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    missing = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, mode="popular", data_root=tmp_path)
    assert missing == {"status": "disabled", "source": "youtube", "reason": "missing_key"}
    assert calls == []


def test_youtube_topic_ok_preserves_order_and_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    _yt_enable(monkeypatch)
    thesis = _yt_thesis(tmp_path)
    search: _Row = {
        "regionCode": "US",
        "items": [
            _yt_search_item("vidA", "Alpha Marker", "Chan Alpha"),
            _yt_search_item("vidB", "Beta Marker", "Chan Beta", live="live"),
        ],
    }
    details: _Row = {
        "items": [
            {"id": "vidB", "statistics": {"viewCount": "9007199254740993", "commentCount": "7"}},
            {"id": "vidA", "statistics": {"viewCount": "12", "likeCount": "3", "commentCount": "1"}},
        ]
    }
    seen: dict[str, object] = {}

    def _get(url: str, **kwargs: object) -> _YtPipe:
        params = kwargs.get("params", {})
        assert isinstance(params, dict)
        if "search" in url:
            seen["search"] = params
            assert params["order"] == "viewCount" and params["q"] == "nvda gpus"
            return _YtPipe(search)
        seen["videos"] = params
        return _YtPipe(details)

    monkeypatch.setattr(requests, "get", _get)
    result = youtube.get_youtube_analytics(
        thesis_id=thesis.slug, mode="topic", query="nvda gpus", region="us", days=7, limit=2, data_root=tmp_path
    )
    assert result["status"] == "ok"
    assert result["order"] == "viewCount" and result["query"] == "nvda gpus"
    assert result["days"] == 7 and result["region"] == "US" and result["mode"] == "topic"
    assert result["thesis"] == {"thesis_id": thesis.thesis_id, "slug": thesis.slug}
    videos = result["videos"]
    assert isinstance(videos, list)
    assert [v["video_id"] for v in videos] == ["vidA", "vidB"]
    assert videos[1]["view_count"] == "9007199254740993"
    assert videos[1]["like_count"] is None
    assert videos[1]["live_broadcast_content"] == "live"
    assert result["warnings"] == ["live_ordering"]
    search_seen = seen["search"]
    assert isinstance(search_seen, dict)
    assert search_seen["publishedAfter"] < search_seen["publishedBefore"]
    retrieved_raw = result["retrieved_at"]
    assert isinstance(retrieved_raw, str)
    expires_raw = result["expires_at"]
    assert isinstance(expires_raw, str)
    retrieved = datetime.fromisoformat(retrieved_raw)
    expires = datetime.fromisoformat(expires_raw)
    assert expires - retrieved == timedelta(minutes=15)


def test_youtube_no_content_on_disk_and_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _yt_enable(monkeypatch)
    thesis = _yt_thesis(tmp_path)
    title, channel = "DISK-MARKER-TITLE-9Z3", "DISK-MARKER-CHANNEL-8Y2"

    def _get(url: str, **kwargs: object) -> _YtPipe:
        if "search" in url:
            return _YtPipe({"items": [_yt_search_item("v1", title, channel)]})
        return _YtPipe({"items": [{"id": "v1", "statistics": {"viewCount": "5"}}]})

    monkeypatch.setattr(requests, "get", _get)
    result = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, query="public term", limit=1, data_root=tmp_path)
    assert result["status"] == "ok"
    assert not (tmp_path / "google_data" / "youtube_cache.json").exists()
    assert not (tmp_path / "google_data" / "youtube_cache.json.tmp").exists()
    blob = "".join(p.read_text(errors="replace") for p in tmp_path.rglob("*") if p.is_file())
    assert title not in blob and channel not in blob


def test_youtube_popular_uses_chart_without_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _yt_enable(monkeypatch)
    thesis = _yt_thesis(tmp_path)
    seen: dict[str, object] = {}

    def _get(url: str, **kwargs: object) -> _YtPipe:
        chart_params = kwargs.get("params", {})
        assert isinstance(chart_params, dict)
        seen.update(chart_params)
        return _YtPipe({"items": [_yt_chart_item("p1", "Pop Title", "Pop Chan")]})

    monkeypatch.setattr(requests, "get", _get)
    result = youtube.get_youtube_analytics(
        thesis_id=thesis.thesis_id, mode="popular", region="GB", limit=1, data_root=tmp_path
    )
    assert result["status"] == "ok" and result["order"] == "mostPopular"
    assert result["query"] is None and result["days"] is None
    assert seen.get("chart") == "mostPopular" and seen.get("regionCode") == "GB"
    assert "q" not in seen and "publishedAfter" not in seen


def test_youtube_legacy_cache_refuses_without_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _yt_enable(monkeypatch)
    thesis = _yt_thesis(tmp_path)
    cache = tmp_path / "google_data" / "youtube_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"seeded": "STALE-MARKER"}))
    calls: list[str] = []

    def _get(url: str, **kwargs: object) -> _YtPipe:
        calls.append(url)
        return _YtPipe()

    monkeypatch.setattr(requests, "get", _get)
    result = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, query="x", data_root=tmp_path)
    assert result["error_type"] == "legacy_cache_present"
    assert calls == [] and "STALE-MARKER" not in json.dumps(result)


def test_youtube_invalid_params_and_private_query_refuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _yt_enable(monkeypatch)
    thesis = _yt_thesis(tmp_path)
    calls: list[str] = []

    def _get(url: str, **kwargs: object) -> _YtPipe:
        calls.append(url)
        return _YtPipe()

    monkeypatch.setattr(requests, "get", _get)
    cases = [
        {"mode": "trending", "query": "x"},
        {"query": "x", "region": "USA"},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": True},
        {"query": "   "},
        {"query": "x" * 201},
        {"query": "bad\x01query"},
        {"mode": "popular", "query": "x"},
        {"query": "x", "days": 91},
    ]
    # Runtime-validation probe: inputs are intentionally mistyped (limit=True),
    # so the call goes through a loose alias instead of the static signature.
    probe: Callable[..., _Row] = youtube.get_youtube_analytics
    for extra in cases:
        kwargs: dict[str, object] = {"thesis_id": thesis.thesis_id, "query": "x", "data_root": tmp_path}
        kwargs.update(extra)
        assert probe(**kwargs)["error_type"] == "invalid_params"
    unknown = youtube.get_youtube_analytics(thesis_id="thesis:missing", query="x", data_root=tmp_path)
    assert unknown["error_type"] == "invalid_thesis"
    private = youtube.get_youtube_analytics(
        thesis_id=thesis.thesis_id, query="my portfolio holdings are up $5000 today", data_root=tmp_path
    )
    assert private["error_type"] == "private_args_denied"
    assert calls == []
    assert not (tmp_path / "google_data" / "youtube_quota.json").exists()


def test_youtube_quota_exhaustion_and_failed_request_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _yt_enable(monkeypatch)
    monkeypatch.setenv("YOUTUBE_SEARCH_DAILY_LIMIT", "1")
    thesis = _yt_thesis(tmp_path)
    calls: list[str] = []

    def _fail(url: str, **kwargs: object) -> _YtPipe:
        calls.append(url)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", _fail)
    failed = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, query="x", limit=1, data_root=tmp_path)
    assert failed["error_type"] == "source_unavailable"
    blocked = youtube.get_youtube_analytics(thesis_id=thesis.thesis_id, query="y", limit=1, data_root=tmp_path)
    assert blocked["error_type"] == "quota_exhausted"
    assert len(calls) == 1, "failed request consumed the only quota unit; no second call"


# BigQuery-backed context -------------------------------------------------------


def test_patents_require_aliases_and_count_publications(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    calls: list[tuple[str, _Params]] = []

    def _run(template: str, params: _Params) -> _Row:
        calls.append((template, params))
        return {
            "status": "ok",
            "rows": [
                {
                    "publication_id": "US-1",
                    "publication_date": "2024-05-01",
                    "assignees": ["Acme Corp"],
                    "classes": ["G06F"],
                    "title": "Widget",
                }
            ],
            "source": "bigquery",
        }

    refused = patents.search_company_patents("Acme")
    assert refused["status"] == "unavailable"
    assert calls == []
    result = patents.search_company_patents("Acme", assignees=["Acme Corp"], limit=5, executor=_run)
    assert result["status"] == "ok"
    pubs = result["publications"]
    assert isinstance(pubs, list)
    assert [p["publication_id"] for p in pubs] == ["US-1"]
    assert "invention" not in json.dumps(result).lower()
    assert "start_yyyymmdd" in calls[0][1] and "end_yyyymmdd" in calls[0][1]
    end_ymd = calls[0][1]["end_yyyymmdd"]
    start_ymd = calls[0][1]["start_yyyymmdd"]
    assert isinstance(end_ymd, int) and isinstance(start_ymd, int)
    assert end_ymd >= start_ymd
    n_calls = len(calls)
    half_open = patents.search_company_patents("Acme", assignees=["Acme Corp"], start_date="2024-01-01", executor=_run)
    assert half_open["error_type"] == "invalid_params"
    assert len(calls) == n_calls


def test_geo_mismatch_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    empty = geo_context.get_geo_context(["__not_a_geo__"], executor=lambda t, p: {"status": "ok", "rows": []})
    assert empty["status"] in ("unavailable", "error", "disabled")
    if empty["status"] == "unavailable":
        assert "missing" in json.dumps(empty).lower()
    bad_rows = geo_context.get_geo_context(["geoId/06"], executor=lambda t, p: {"status": "ok", "rows": [{"foo": 1}]})
    assert bad_rows["error_type"] == "unsupported_join"


def test_stackoverflow_stale_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    rows = [
        {"tag": "python", "period": "2020-01", "activity_count": 5},
        {"tag": "python", "period": "2021-01", "activity_count": 7},
    ]
    result = stackoverflow.get_tag_activity(["python"], executor=lambda t, p: {"status": "ok", "rows": rows})
    assert result["status"] == "ok"
    assert result["last_covered"] == "2021-01"


# Integration boundary ----------------------------------------------------------


def test_tool_schemas_bounded() -> None:
    from app import tools as tools_mod
    from app.security.action_policy import TOOL_DOMAINS
    from app.security.context_gateway import TOOL_ENVELOPES

    schemas: dict[str, dict[str, object]] = {}
    for tool in tools_mod.TOOLS:
        function = tool.get("function")
        assert isinstance(function, dict)
        tool_name = function.get("name")
        assert isinstance(tool_name, str)
        schemas[tool_name] = function

    def _required(key: str) -> list[object]:
        params = schemas[key]["parameters"]
        assert isinstance(params, dict)
        required = params["required"]
        assert isinstance(required, list)
        return required

    expected_caps = {
        "find_alternative_signals": 100,
        "get_trend_evidence": 1000,
        "investigate_social_arbitrage_candidate": 25,
        "get_macro_context": 100,
        "search_company_patents": 20,
    }
    for name, cap in expected_caps.items():
        params = schemas[name]["parameters"]
        assert isinstance(params, dict)
        properties = params["properties"]
        assert isinstance(properties, dict)
        limit_spec = properties["limit"]
        assert isinstance(limit_spec, dict)
        assert limit_spec["maximum"] == cap
        assert tools_mod.TOOL_CAPABILITIES[name].name == "RESEARCH"
        assert TOOL_DOMAINS[name] == "financial_research"
        assert name in TOOL_ENVELOPES
    assert set(_required("investigate_social_arbitrage_candidate")) == {"term"}
    assert set(_required("get_macro_context")) == {"geos", "variables"}
    assert _required("search_company_patents") == ["company_id", "assignees"]


def test_handler_disabled_passthrough_makes_no_network_call() -> None:
    from app import tools as tools_mod

    handler = tools_mod._DIRECT_HANDLERS["get_trend_evidence"]
    assert isinstance(handler, Callable)
    result = handler({"start_date": "2026-09-01", "end_date": "2026-09-02", "geos": ["US"], "limit": 5}, "test-model")
    assert isinstance(result, dict)
    assert result["status"] == "disabled"


def test_trend_evidence_term_passed_to_collect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app import tools as tools_mod
    from app.google_data import trends as trends_mod

    payload: _Row = {
        "status": "ok",
        "observations": [{"term": "Stanley Cup"}],
        "rows": [{"term": "Stanley Cup"}],
        "count": 1,
    }
    seen: dict[str, object] = {}

    def _fake_collect(**kwargs: object) -> _Row:
        seen.update(kwargs)
        return payload

    monkeypatch.setattr(trends_mod, "collect_trends", _fake_collect)
    handler = tools_mod._DIRECT_HANDLERS["get_trend_evidence"]
    assert isinstance(handler, Callable)
    result = handler(
        {"start_date": "2026-09-01", "end_date": "2026-09-02", "geos": ["US"], "limit": 10, "term": "Stanley"},
        "test-model",
    )
    assert seen.get("term") == "Stanley"
    assert seen.get("limit") == 10
    assert result == payload


def test_investigate_returns_evidence_and_gaps() -> None:
    from app import tools as tools_mod

    handler = tools_mod._DIRECT_HANDLERS["investigate_social_arbitrage_candidate"]
    assert isinstance(handler, Callable)
    result = handler({"term": "Stanley"}, "test-model")
    assert result["term"] == "Stanley"
    assert "evidence" in result and "gaps" in result


def test_investigate_never_embeds_youtube(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib
    import sqlite3

    from app import tools as tools_mod
    from app.storage.runs import RunRecorder

    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "yt-key")
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    marker = "YT-MARKER-7Q2-never-persist"
    calls: list[_Row] = []

    def _fake_fetch(**kwargs: object) -> _Row:
        calls.append(kwargs)
        return {"status": "ok", "videos": [{"title": marker}]}

    monkeypatch.setattr(youtube, "get_youtube_analytics", _fake_fetch)
    handler = tools_mod._DIRECT_HANDLERS["investigate_social_arbitrage_candidate"]
    assert isinstance(handler, Callable)
    recorder = RunRecorder(
        run_id="r-yt",
        request_id="q-yt",
        question="stanley",
        as_of=None,
        model="t",
        provider="t",
        model_parameters={},
        agent_version="t",
        prompt_version="t",
        tool_registry_version="t",
        git_sha="t",
        data_root=tmp_path,
    )
    with recorder:
        result = handler({"term": "Stanley"}, "test-model")
        rendered = json.dumps(result)
        recorder.record_evidence(
            evidence_id="r-yt:evid:0001",
            run_id="r-yt",
            tool_call_id="c1",
            round=0,
            tool_name="investigate_social_arbitrage_candidate",
            rendered_hash=hashlib.sha256(rendered.encode()).hexdigest(),
            rendered_bytes=len(rendered.encode()),
            estimated_tokens=len(rendered) // 4,
            source_names="[]",
            source_freshness="[]",
            as_of=None,
            rendered_text=rendered,
        )
    assert calls == []
    assert marker not in rendered
    assert "youtube" not in json.dumps(result.get("evidence", {}))
    assert "youtube-analytics" in rendered
    rows = sqlite3.connect(tmp_path / "runs.sqlite").execute("SELECT rendered_text FROM evidence").fetchall()
    assert rows and all(marker not in (row[0] or "") for row in rows)


def test_cli_google_data_parses_and_patents_needs_company(capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    import cli

    args = cli._build_parser().parse_args(
        [
            "google-data",
            "collect",
            "--source",
            "trends",
            "--start-date",
            "2026-09-01",
            "--end-date",
            "2026-09-02",
            "--geo",
            "US",
            "--limit",
            "25",
        ]
    )
    assert args.source == "trends" and args.geo == ["US"] and args.limit == 25
    cli._cmd_google_data(
        argparse.Namespace(
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
    )
    out = json.loads(capsys.readouterr().out)
    assert out["sources"]["patents"]["status"] == "error"


# Live smoke (explicit markers only; offline suite skips) ----------------------


def test_smoke_bigquery_trends_partition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not os.environ.get("RUN_BIGQUERY_SMOKE"):
        pytest.skip("live validation unperformed (no RUN_BIGQUERY_SMOKE)")
    from datetime import UTC, datetime, timedelta

    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    end = datetime.now(UTC).date().isoformat()
    start = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    result = bq.submit_template(
        "trends_refreshes",
        {"table": "bigquery-public-data.google_trends.top_terms", "start_date": start, "end_date": end, "limit": 5},
        data_root=tmp_path,
    )
    assert result["status"] == "ok"


def test_smoke_datacommons_california_population() -> None:
    if not os.environ.get("RUN_DATACOMMONS_SMOKE"):
        pytest.skip("live validation unperformed (no RUN_DATACOMMONS_SMOKE)")
    result = datacommons.get_macro_context(["geoId/06"], ["Count_Person"], limit=5)
    assert result["status"] == "ok"
    assert result["series"], "expected at least one California population series"


def test_smoke_youtube_search_and_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not os.environ.get("RUN_YOUTUBE_SMOKE"):
        pytest.skip("live validation unperformed (no RUN_YOUTUBE_SMOKE)")
    monkeypatch.setenv("GOOGLE_DATA_ENABLED", "true")
    thesis = _yt_thesis(tmp_path)
    result = youtube.get_youtube_analytics(
        thesis_id=thesis.thesis_id, mode="topic", query="python programming", limit=1, data_root=tmp_path
    )
    assert result["status"] == "ok"
    assert result["videos"], "expected at least one video with batched statistics"


def test_smoke_trends_api_pending() -> None:
    if not os.environ.get("RUN_GOOGLE_TRENDS_API_SMOKE"):
        pytest.skip("live validation unperformed (no RUN_GOOGLE_TRENDS_API_SMOKE)")
    from app.google_data import trends_api

    result = trends_api.get_interest_over_time(terms=["python"], interval="weekly")
    if result.get("error") in ("GOOGLE_TRENDS_API_DISABLED", "GOOGLE_TRENDS_API_PENDING_ACCESS"):
        pytest.skip("official Trends pending alpha access")
    assert result["status"] == "ok"


# Executor contract fidelity --------------------------------------------------


def test_render_filters_to_referenced_params_with_sdk_types() -> None:
    bq_sdk = pytest.importorskip("google.cloud.bigquery")
    spec = bq.TEMPLATES["trends_us_top"]
    render_table = spec["table"]
    assert isinstance(render_table, str)
    sql, params = bq._render_sql(
        spec,
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-02",
            "dmas": ["New York NY"],
            "all_dmas": False,
            "week_start": "2020-01-01",
            "week_end": "2026-09-02",
            "limit": 10,
            "collector_version": "1",
            "sql_version": "1",
        },
        render_table,
    )
    assert "DATE(@start_date)" in sql and "CURRENT_DATE" not in sql
    assert set(params) == {"start_date", "end_date", "dmas", "all_dmas", "week_start", "week_end"}
    typed: dict[str, object] = {}
    for p in bq._query_parameters(bq_sdk, params):
        typed[str(getattr(p, "name"))] = p  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    assert getattr(typed["dmas"], "array_type") == "STRING"  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    assert getattr(typed["start_date"], "type_") == "STRING"  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    assert getattr(typed["all_dmas"], "type_") == "BOOL"  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green


def test_youtube_default_search_ceiling_is_80(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config as _config

    assert _config.get_youtube_search_daily_limit() == 80
    assert _config.get_bq_daily_bytes_limit() == 10737418240
    assert _config.get_bq_monthly_bytes_limit() == 536870912000
