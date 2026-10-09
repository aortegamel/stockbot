"""Tests for the risk/mandate domain, storage glue, CLI command, and tool."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from app import tools
from app.domain.portfolio import PortfolioSnapshot, Position
from app.domain.risk.evaluation import UNKNOWN_SECTOR, EvaluationIssue, evaluate_mandate
from app.domain.risk.mandate import Mandate, RiskLimit, parse_mandate
from app.services import risk as risk_service
from app.services.mandate import load_mandate_file
from app.services.portfolio_sync import persist_snapshot, read_latest_snapshot
from cli import _cmd_evaluate_mandate


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def _position(
    position_id: str,
    ticker: str,
    entity_id: str | None,
    weight: Decimal | None,
    *,
    cash: Decimal | None = None,
) -> Position:
    return Position(
        position_id=position_id,
        account_id="acc-1",
        security_id="sec:equity:0000320193" if entity_id else None,
        entity_id=entity_id,
        ticker=ticker,
        quantity=Decimal(10),
        average_cost=Decimal("95.50"),
        market_price=Decimal("116.84"),
        market_value=Decimal("1168.40"),
        unrealized_gain=Decimal("213.40"),
        unrealized_gain_pct=Decimal("0.22"),
        portfolio_weight=weight,
        source="robinhood_mcp",
        retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=UTC),
        price_type="last",
        quote_retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=UTC),
    )


_MISSING = object()


def _hand_built_snapshot(weight: Decimal | None | object = _MISSING) -> PortfolioSnapshot:
    resolved_weight: Decimal | None
    unresolved_weight: Decimal | None
    if isinstance(weight, Decimal) or weight is None:
        resolved_weight = weight
        unresolved_weight = weight
    else:
        assert weight is _MISSING
        resolved_weight = Decimal("0.75")
        unresolved_weight = Decimal("0.25")
    resolved = _position(
        "snap-1:acc-1:WING",
        "WING",
        "sec:cik:0000320193",
        resolved_weight,
    )
    unresolved = _position(
        "snap-1:acc-1:ZZZZ",
        "ZZZZ",
        None,
        unresolved_weight,
    )
    return PortfolioSnapshot(
        snapshot_id="portfolio:robinhood:2026-08-25T12:00:00+00:00",
        created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        broker="robinhood",
        account_ids=("acc-1",),
        cash=Decimal("1234.56"),
        invested_value=Decimal("1228.40"),
        total_value=Decimal("2462.96"),
        positions=(resolved, unresolved),
    )


def _mandate(limits: Sequence[RiskLimit], prohibited: Sequence[str] = ()) -> Mandate:
    return Mandate(limits=tuple(limits), prohibited_assets=tuple(prohibited))


def _limit(metric: str, operator: str, threshold: str | float | Decimal, **overrides: str) -> RiskLimit:
    values = {"metric": metric, "operator": operator, "threshold": Decimal(str(threshold))}
    values.update(overrides)
    return RiskLimit(**values)


def _write_mandate(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_mandate / load_mandate_file
# ---------------------------------------------------------------------------


def test_load_mandate_file_valid_json_with_defaults(tmp_path: Path):
    path = _write_mandate(
        tmp_path / "mandate.json",
        {
            "limits": [
                {"metric": "single_position_weight", "operator": "<=", "threshold": 0.25},
                {"metric": "minimum_cash", "operator": ">=", "threshold": 0.10},
                {
                    "metric": "sector_exposure",
                    "target": "semiconductors",
                    "operator": "<=",
                    "threshold": 0.20,
                    "severity": "critical",
                },
            ],
            "prohibited_assets": ["GME", "sec:cik:0000320193"],
            "extra_ignored": True,
        },
    )
    mandate = load_mandate_file(path)
    assert len(mandate.limits) == 3
    assert mandate.limits[0].severity == "warning"
    assert mandate.limits[0].unit == "ratio"
    assert mandate.limits[0].threshold == Decimal("0.25")
    assert mandate.limits[1].severity == "warning"
    assert mandate.limits[2].target == "semiconductors"
    assert mandate.limits[2].severity == "critical"
    assert mandate.prohibited_assets == ("GME", "sec:cik:0000320193")


def test_load_mandate_file_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_mandate_file(tmp_path / "nope.json")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"limits": {}},
        {"limits": [{"metric": "bogus", "operator": "<=", "threshold": 1}]},
        {"limits": [{"metric": "single_position_weight", "operator": ">", "threshold": 1}]},
        {"limits": [{"metric": "single_position_weight", "operator": "<="}]},
        {"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": 0}]},
        {"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": -0.5}]},
        {"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": "abc"}]},
        {"limits": [{"metric": "sector_exposure", "operator": "<=", "threshold": 0.2}]},
        {"limits": [{"metric": "sector_exposure", "target": "", "operator": "<=", "threshold": 0.2}]},
        {"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": 0.25, "unit": "dollars"}]},
        {
            "limits": [
                {
                    "metric": "sector_exposure",
                    "target": "semiconductors",
                    "operator": "<=",
                    "threshold": 0.20,
                    "unit": "dollars",
                }
            ]
        },
        {"limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": 1.5}]},
        {"limits": [{"metric": "sector_exposure", "target": "semiconductors", "operator": "<=", "threshold": 1.5}]},
        {"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 1.5}]},
        {"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 0.1}], "prohibited_assets": "GME"},
        {"limits": [{"metric": "minimum_cash", "operator": ">=", "threshold": 0.1}], "prohibited_assets": [""]},
        {"limits": "nope"},
    ],
)
def test_parse_mandate_rejects_bad_config(payload: Mapping[str, object]):
    with pytest.raises(ValueError):
        parse_mandate(payload)


def test_parse_mandate_normalizes_whitespace():
    mandate = parse_mandate(
        {
            "limits": [
                {"metric": "sector_exposure", "target": "  semiconductors  ", "operator": "<=", "threshold": 0.20},
            ],
            "prohibited_assets": [" GME ", " sec:cik:0000320193 "],
        }
    )
    assert mandate.limits[0].target == "semiconductors"
    assert mandate.prohibited_assets == ("GME", "sec:cik:0000320193")


# ---------------------------------------------------------------------------
# evaluate_mandate math
# ---------------------------------------------------------------------------


def test_single_position_weight_breach_and_clean_position():
    mandate = _mandate([_limit("single_position_weight", "<=", "0.25")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert len(evaluation.breaches) == 1
    breach = evaluation.breaches[0]
    assert breach.metric == "single_position_weight"
    assert breach.actual == Decimal("0.75")
    assert breach.limit == Decimal("0.25")
    assert breach.excess == Decimal("0.50")
    assert breach.note == "WING (snap-1:acc-1:WING)"
    assert breach.severity == "warning"
    assert breach.unit == "ratio"


def test_single_position_weight_at_threshold_no_breach():
    mandate = _mandate([_limit("single_position_weight", "<=", "0.75")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert evaluation.breaches == ()


def test_single_position_weight_breach_with_ge_operator():
    mandate = _mandate([_limit("single_position_weight", ">=", "0.80")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    breach = evaluation.breaches[0]
    assert breach.excess == Decimal("0.05")


def test_single_position_weight_none_weight_not_evaluable():
    snapshot = _hand_built_snapshot(weight=None)
    mandate = _mandate([_limit("single_position_weight", "<=", "0.25")])
    evaluation = evaluate_mandate(snapshot, mandate)
    assert evaluation.breaches == ()
    assert evaluation.sector_exposures == {}
    assert evaluation.issues == (
        EvaluationIssue(
            "position_weight_unavailable", "single_position_weight", ticker="WING", position_id="snap-1:acc-1:WING"
        ),
        EvaluationIssue(
            "position_weight_unavailable", "single_position_weight", ticker="ZZZZ", position_id="snap-1:acc-1:ZZZZ"
        ),
    )


def test_minimum_cash_ratio_breach_and_excess():
    mandate = _mandate([_limit("minimum_cash", ">=", "0.60")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert len(evaluation.breaches) == 1
    breach = evaluation.breaches[0]
    assert breach.metric == "minimum_cash"
    assert isinstance(breach.actual, Decimal)
    assert breach.actual == Decimal("1234.56") / Decimal("2462.96")
    assert breach.excess == Decimal("0.60") - breach.actual
    assert breach.unit == "ratio"


def test_minimum_cash_ratio_satisfied_no_breach():
    mandate = _mandate([_limit("minimum_cash", ">=", "0.10")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert evaluation.breaches == ()


def test_minimum_cash_dollars_unit():
    mandate = _mandate([_limit("minimum_cash", ">=", "5000", unit="dollars")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    breach = evaluation.breaches[0]
    assert breach.actual == Decimal("1234.56")
    assert breach.excess == Decimal(5000) - Decimal("1234.56")
    assert breach.unit == "dollars"


def test_minimum_cash_unavailable_not_evaluable():
    snapshot = _hand_built_snapshot()
    snapshot = PortfolioSnapshot(
        snapshot_id=snapshot.snapshot_id,
        created_at=snapshot.created_at,
        broker=snapshot.broker,
        account_ids=snapshot.account_ids,
        cash=None,
        invested_value=snapshot.invested_value,
        total_value=snapshot.total_value,
        positions=snapshot.positions,
    )
    mandate = _mandate([_limit("minimum_cash", ">=", "0.10")])
    evaluation = evaluate_mandate(snapshot, mandate)
    assert evaluation.breaches == ()
    assert evaluation.issues == (EvaluationIssue("cash_unavailable", "minimum_cash"),)


def test_minimum_cash_total_value_unavailable_not_evaluable():
    snapshot = _hand_built_snapshot()
    snapshot = PortfolioSnapshot(
        snapshot_id=snapshot.snapshot_id,
        created_at=snapshot.created_at,
        broker=snapshot.broker,
        account_ids=snapshot.account_ids,
        cash=Decimal("1234.56"),
        invested_value=snapshot.invested_value,
        total_value=None,
        positions=snapshot.positions,
    )
    mandate = _mandate([_limit("minimum_cash", ">=", "0.10")])
    evaluation = evaluate_mandate(snapshot, mandate)
    assert evaluation.breaches == ()
    assert evaluation.issues == (EvaluationIssue("total_value_unavailable", "minimum_cash"),)


def test_prohibited_assets_ticker_and_entity_matches():
    mandate = _mandate([], prohibited=("wing", "sec:cik:0000320193", "NOPE"))
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert len(evaluation.breaches) == 2
    for breach in evaluation.breaches:
        assert breach.metric == "prohibited_assets"
        assert breach.severity == "warning"
        assert breach.unit is None
        assert breach.actual == "WING"
        assert breach.note == "position WING (snap-1:acc-1:WING)"
    assert {breach.target for breach in evaluation.breaches} == {"wing", "sec:cik:0000320193"}


def test_sector_exposure_unpriced_position_not_evaluable():
    snapshot = _hand_built_snapshot(weight=None)
    mandate = _mandate([_limit("sector_exposure", "<=", "0.20", target="semiconductors")])
    evaluation = evaluate_mandate(
        snapshot,
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.breaches == ()
    assert evaluation.issues == (
        EvaluationIssue(
            "position_weight_unavailable", "sector_exposure", ticker="WING", position_id="snap-1:acc-1:WING"
        ),
        EvaluationIssue(
            "position_weight_unavailable", "sector_exposure", ticker="ZZZZ", position_id="snap-1:acc-1:ZZZZ"
        ),
    )


def test_minimum_cash_zero_total_value_not_evaluable():
    snapshot = _hand_built_snapshot()
    snapshot = PortfolioSnapshot(
        snapshot_id=snapshot.snapshot_id,
        created_at=snapshot.created_at,
        broker=snapshot.broker,
        account_ids=snapshot.account_ids,
        cash=Decimal(0),
        invested_value=Decimal(0),
        total_value=Decimal(0),
        positions=snapshot.positions,
    )
    mandate = _mandate([_limit("minimum_cash", ">=", "0.10")])
    evaluation = evaluate_mandate(snapshot, mandate)
    assert evaluation.breaches == ()
    assert evaluation.issues == (EvaluationIssue("total_value_zero", "minimum_cash"),)


def test_sector_exposure_buckets_unknown_and_breaches():
    mandate = _mandate([_limit("sector_exposure", "<=", "0.20", target="semiconductors")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.sector_exposures == {
        "semiconductors": Decimal("0.75"),
        UNKNOWN_SECTOR: Decimal("0.25"),
    }
    assert len(evaluation.breaches) == 1
    breach = evaluation.breaches[0]
    assert breach.metric == "sector_exposure"
    assert breach.target == "semiconductors"
    assert breach.actual == Decimal("0.75")
    assert breach.excess == Decimal("0.55")


def test_sector_exposure_missing_target_sector_is_zero():
    mandate = _mandate([_limit("sector_exposure", "<=", "0.20", target="aero")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.breaches == ()
    assert EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="aero") in evaluation.issues


def test_sector_max_known_below_threshold_with_unknown_not_evaluable():
    mandate = _mandate([_limit("sector_exposure", "<=", "0.80", target="semiconductors")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.breaches == ()
    assert EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="semiconductors") in evaluation.issues


def test_sector_max_known_above_threshold_with_unknown_breaches():
    mandate = _mandate([_limit("sector_exposure", "<=", "0.20", target="semiconductors")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert len(evaluation.breaches) == 1
    assert evaluation.breaches[0].actual == Decimal("0.75")
    assert (
        EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="semiconductors") not in evaluation.issues
    )


def test_sector_min_known_below_threshold_with_unknown_not_evaluable():
    mandate = _mandate([_limit("sector_exposure", ">=", "0.80", target="semiconductors")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.breaches == ()
    assert EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="semiconductors") in evaluation.issues


def test_sector_min_known_above_threshold_with_unknown_passes():
    mandate = _mandate([_limit("sector_exposure", ">=", "0.70", target="semiconductors")])
    evaluation = evaluate_mandate(
        _hand_built_snapshot(),
        mandate,
        sector_map={"sec:cik:0000320193": "semiconductors"},
    )
    assert evaluation.breaches == ()
    assert evaluation.issues == ()


def test_sector_no_unknown_keeps_deterministic_behavior():
    snapshot = PortfolioSnapshot(
        snapshot_id="portfolio:robinhood:2026-08-25T12:00:00+00:00",
        created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        broker="robinhood",
        account_ids=("acc-1",),
        cash=Decimal("1234.56"),
        invested_value=Decimal("1228.40"),
        total_value=Decimal("2462.96"),
        positions=(
            _position("snap-1:acc-1:WING", "WING", "sec:cik:0000320193", Decimal("0.75")),
            _position("snap-1:acc-1:ZZZZ", "ZZZZ", "sec:cik:0000999999", Decimal("0.25")),
        ),
    )
    sector_map = {"sec:cik:0000320193": "semiconductors", "sec:cik:0000999999": "aerospace"}
    below = evaluate_mandate(
        snapshot,
        _mandate([_limit("sector_exposure", "<=", "0.80", target="semiconductors")]),
        sector_map=sector_map,
    )
    assert below.breaches == ()
    assert below.issues == ()
    above = evaluate_mandate(
        snapshot,
        _mandate([_limit("sector_exposure", ">=", "0.70", target="semiconductors")]),
        sector_map=sector_map,
    )
    assert above.breaches == ()
    assert above.issues == ()
    breach = evaluate_mandate(
        snapshot,
        _mandate([_limit("sector_exposure", "<=", "0.20", target="semiconductors")]),
        sector_map=sector_map,
    )
    assert len(breach.breaches) == 1
    assert breach.breaches[0].actual == Decimal("0.75")
    assert breach.issues == ()


def test_empty_sector_map_sector_limits_not_falsely_safe():
    mandate = _mandate([_limit("sector_exposure", "<=", "0.20", target="semiconductors")])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate, sector_map={})
    assert evaluation.breaches == ()
    assert evaluation.sector_exposures == {UNKNOWN_SECTOR: Decimal("1.0")}
    assert EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="semiconductors") in evaluation.issues


def test_empty_mandate_zero_breaches():
    mandate = _mandate([])
    evaluation = evaluate_mandate(_hand_built_snapshot(), mandate)
    assert evaluation.breaches == ()
    assert evaluation.issues == ()
    assert evaluation.snapshot_id == "portfolio:robinhood:2026-08-25T12:00:00+00:00"
    assert evaluation.created_at == datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# load_sector_map
# ---------------------------------------------------------------------------


def _positions(entity_id: str | None) -> list[Position]:
    return [_position("snap-1:acc-1:WING", "WING", entity_id, Decimal("0.75"))]


def _sector(monkeypatch: pytest.MonkeyPatch, sector: str | None = "semiconductors") -> None:
    def _fake(cik: int) -> str | None:
        return sector if cik == 320193 else None

    monkeypatch.setattr(risk_service, "_provider_sector", _fake)


def test_load_sector_map_live_provider_sector(monkeypatch: pytest.MonkeyPatch):
    _sector(monkeypatch)
    assert risk_service.load_sector_map(_positions("sec:cik:0000320193")) == {"sec:cik:0000320193": "semiconductors"}


def test_load_sector_map_unknown_and_provider_failure(monkeypatch: pytest.MonkeyPatch):
    assert risk_service.load_sector_map(_positions(None)) == {}

    def _boom(cik: int) -> str | None:
        raise RuntimeError("provider down")

    monkeypatch.setattr(risk_service, "_provider_sector", _boom)
    with pytest.raises(RuntimeError, match="provider down"):
        risk_service.load_sector_map(_positions("sec:cik:0000320193"))


def test_load_sector_map_dedups_entity(monkeypatch: pytest.MonkeyPatch):
    seen: list[int] = []

    def _count(cik: int) -> str | None:
        seen.append(cik)
        return "aero"

    monkeypatch.setattr(risk_service, "_provider_sector", _count)
    positions = _positions("sec:cik:0000320193") + _positions("sec:cik:0000320193")
    assert risk_service.load_sector_map(positions) == {"sec:cik:0000320193": "aero"}
    assert seen == [320193]


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def _seed_snapshot_and_mandate(
    data_root: Path, mandate_payload: Mapping[str, object] | None = None, monkeypatch: pytest.MonkeyPatch | None = None
) -> Path:
    persist_snapshot(_hand_built_snapshot(), data_root=data_root)
    if monkeypatch is not None:
        _sector(monkeypatch)
    mandate_path = data_root / "mandate.json"
    return _write_mandate(
        mandate_path,
        mandate_payload
        if mandate_payload is not None
        else {
            "limits": [
                {"metric": "sector_exposure", "target": "semiconductors", "operator": "<=", "threshold": 0.20},
                {"metric": "single_position_weight", "operator": "<=", "threshold": 0.25},
            ],
            "prohibited_assets": [],
        },
    )


def test_cli_evaluate_mandate_reports(
    capsys: pytest.CaptureFixture[str], data_root: Path, monkeypatch: pytest.MonkeyPatch
):
    mandate_path = _seed_snapshot_and_mandate(data_root, monkeypatch=monkeypatch)
    _cmd_evaluate_mandate(mandate_path, str(data_root))
    out = capsys.readouterr().out
    assert f"Mandate: {mandate_path}" in out
    expected_created = f"Snapshot: portfolio:robinhood:2026-08-25T12:00:00+00:00 created {datetime(2026, 8, 25, 12, 0, tzinfo=UTC).astimezone().isoformat()}"
    assert expected_created in out
    assert "Sector exposures: semiconductors 75.0%, unknown_sector 25.0%" in out
    assert "[warning] sector_exposure semiconductors: actual 75.0%, limit 20.0%, excess 55.0%" in out
    assert "[warning] single_position_weight: actual 75.0%, limit 25.0%, excess 50.0%" in out
    assert "No breaches." not in out


def test_cli_evaluate_mandate_no_breaches(
    capsys: pytest.CaptureFixture[str], data_root: Path, monkeypatch: pytest.MonkeyPatch
):
    mandate_path = _seed_snapshot_and_mandate(
        data_root,
        {
            "limits": [{"metric": "single_position_weight", "operator": "<=", "threshold": 0.99}],
            "prohibited_assets": [],
        },
        monkeypatch,
    )
    _cmd_evaluate_mandate(mandate_path, str(data_root))
    out = capsys.readouterr().out
    assert "No breaches." in out


def test_cli_evaluate_mandate_missing_mandate_exits_1(capsys: pytest.CaptureFixture[str], data_root: Path):
    persist_snapshot(_hand_built_snapshot(), data_root=data_root)
    with pytest.raises(SystemExit) as excinfo:
        _cmd_evaluate_mandate(data_root / "mandate.json", str(data_root))
    assert excinfo.value.code == 1
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------


def _as_seq(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, dict)
    return value


def test_tool_evaluate_mandate_missing_mandate_error(data_root: Path):
    result = tools.evaluate_mandate(data_root=data_root, mandate_path=data_root / "mandate.json")
    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "mandate" in error.lower() or "no such file" in error.lower()


def test_tool_evaluate_mandate_missing_snapshot_error(data_root: Path):
    _write_mandate(data_root / "mandate.json", {"limits": [], "prohibited_assets": []})
    result = tools.evaluate_mandate(data_root=data_root, mandate_path=data_root / "mandate.json")
    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "snapshot" in error.lower()


def test_tool_evaluate_mandate_happy_path(data_root: Path, monkeypatch: pytest.MonkeyPatch):
    mandate_path = _seed_snapshot_and_mandate(data_root, monkeypatch=monkeypatch)
    result = tools.evaluate_mandate(data_root=data_root, mandate_path=mandate_path)
    assert result["result_type"] == "mandate_evaluation"
    assert result["snapshot_id"] == "portfolio:robinhood:2026-08-25T12:00:00+00:00"
    assert result["sector_exposures"] == {"semiconductors": "0.75", "unknown_sector": "0.25"}
    breaches = _as_seq(result["breaches"])
    assert len(breaches) == 2
    breach = breaches[0]
    assert breach["metric"] == "sector_exposure"
    assert breach["actual"] == "0.75"
    assert breach["excess"] == "0.55"
    assert all(breach["unit"] == "ratio" for breach in breaches)
    assert result["issues"] == []
    assert result["source"] == "mandate"


def test_snapshot_seeded_via_persist_reads_back_via_read_latest(data_root: Path):
    persist_snapshot(_hand_built_snapshot(), data_root=data_root)
    snapshot = read_latest_snapshot(data_root=data_root)
    assert snapshot is not None
    assert snapshot.snapshot_id == "portfolio:robinhood:2026-08-25T12:00:00+00:00"
    assert {p.ticker for p in snapshot.positions} == {"WING", "ZZZZ"}
