"""Offline tests for app/sec/lineage.py (no network)."""

import pytest

from app.sec import lineage


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def test_fact_lineage_projects_all_keys():
    row = {"concept": "Revenues", "value": 5.0, "extra": "dropped"}
    out = lineage.fact_lineage(row)
    assert sorted(out) == sorted(
        [
            "concept",
            "value",
            "period_start",
            "period_end",
            "fiscal_year",
            "fiscal_period",
            "filed_at",
            "accession",
            "source_url",
            "known_at",
        ]
    )
    assert out["concept"] == "Revenues" and out["value"] == 5.0
    assert out["filed_at"] is None and "extra" not in out


def test_period_lineage_restatement():
    rows = [
        {
            "period_end": "2024-12-31",
            "value": 1.0,
            "filed_at": "2025-02-01",
            "known_at": "2025-02-02",
            "accession": "a1",
            "concept": "Revenues",
        },
        {
            "period_end": "2024-12-31",
            "value": 2.0,
            "filed_at": "2025-03-01",
            "known_at": "2025-03-02",
            "accession": "a2",
            "concept": "Revenues",
        },
    ]
    out = lineage.period_lineage(rows)
    assert len(out) == 1 and out[0]["restated"] is True
    assert _as_dict(out[0]["originally_reported"])["value"] == 1.0
    assert _as_dict(out[0]["latest"])["value"] == 2.0
    assert out[0]["originally_reported"] != out[0]["latest"]


def _facts_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "documents": [],
        "financial_facts": rows,
        "securities": [],
        "dividend_events": [],
    }


def _fact_row(
    value: float,
    filed_at: str,
    accession: str,
    *,
    entity_id: str = "sec:cik:0000000001",
    concept: str = "Revenues",
) -> dict[str, object]:
    return {
        "fact_id": f"f-{accession}",
        "entity_id": entity_id,
        "concept": concept,
        "value": value,
        "period_end": "2024-12-31",
        "filed_at": filed_at,
        "known_at": filed_at,
        "retrieved_at": f"{filed_at}T00:00:00Z",
        "accession": accession,
    }


def test_xbrl_lineage_as_of_excludes_restatement(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.data_sources import SourceGateway

    rows = [
        _fact_row(1.0, "2025-02-01", "a1"),
        _fact_row(2.0, "2025-03-01", "a2"),
    ]

    def _fake_company_facts(self: object, cik: int, as_of: str | None = None) -> dict[str, object]:
        assert cik == 1
        if as_of is None:
            return _facts_payload(list(rows))
        return _facts_payload([row for row in rows if str(row.get("known_at") or "")[:10] <= as_of])

    monkeypatch.setattr(SourceGateway, "company_facts", _fake_company_facts)
    early = lineage.xbrl_lineage("sec:cik:0000000001", "Revenues", as_of="2025-02-15")
    assert len(early) == 1 and early[0]["value"] == 1.0
    assert lineage.period_lineage(early)[0]["restated"] is False
    assert len(lineage.xbrl_lineage("sec:cik:0000000001", "Revenues")) == 2
    with pytest.raises(ValueError):
        lineage.xbrl_lineage("sec:cik:0000000001", "Revenues", as_of="02/15/2025")
