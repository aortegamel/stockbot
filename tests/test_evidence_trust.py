"""Trust/taint regressions for the Exa ontology layer."""

import inspect
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from app.domain.evidence.models import ResolutionStatus, SourceTier
from app.domain.evidence.source_quality import classify_source
from app.domain.market.securities import SecurityResolution
from app.research import evidence
from app.research.models import JSONValue
from app.security.context import Integrity
from app.services.evidence_claims import build_evidence_claims, claim_to_enriched_dict
from app.tools import plan_public_search_queries, suggest_public_search_queries


def test_high_trust_hosts_elevate():
    c = classify_source("https://www.reuters.com/x")
    assert c.integrity == Integrity.HIGH_TRUST_REPORTED
    assert c.source_tier == SourceTier.HIGH_TRUST_NEWS
    assert c.publisher == "Reuters"


def test_spoof_ir_and_gov_never_elevate():
    for url in (
        "https://reuters.com.evil.com/x",
        "https://evil-reuters.com/x",
        "https://ir.evil.com/x",
        "https://investor.evil.com/x",
        "https://www.sec.gov/x",
        "https://ir.nvidia.com/x",
    ):
        c = classify_source(url)
        assert c.integrity == Integrity.EXTERNAL, url
        assert c.source_tier == SourceTier.UNKNOWN, url


def test_sec_gov_not_canonical_or_primary():
    c = classify_source("https://www.sec.gov/x")
    assert c.source_tier != SourceTier.CANONICAL
    assert c.integrity not in (Integrity.CANONICAL, Integrity.PRIMARY_EXTERNAL)


def _amb(ticker: str | None = None, name: str | None = None, as_of: datetime | None = None) -> SecurityResolution:
    return SecurityResolution(None, None, ticker or name or "UNKNOWN", False, "ambiguous")


def _unr(ticker: str | None = None, name: str | None = None, as_of: datetime | None = None) -> SecurityResolution:
    return SecurityResolution(None, None, ticker or name or "UNKNOWN", False, "unresolved")


def _item(ticker: str = "ABC") -> dict[str, object]:
    return {
        "subject_ticker": ticker,
        "subject_name": "Abc",
        "claim": "x",
        "claim_type": "other",
        "source_url": "https://example.com",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
    }


def test_ambiguous_ticker_strict_identity():
    (c,) = build_evidence_claims(
        reader_items=[_item("ABC")], resolve=_amb, retrieved_fallback="2026-01-01T00:00:00+00:00"
    )
    assert c.entity_id is None and c.security_id is None and c.ticker is None
    assert c.subject_resolution == ResolutionStatus.AMBIGUOUS
    assert c.reported_ticker == "ABC"
    d = claim_to_enriched_dict(c)
    assert d["ticker"] is None and d["reported_ticker"] == "ABC"
    assert d["subject_resolution"] == "ambiguous"


def test_unknown_ticker_unresolved():
    (c,) = build_evidence_claims(
        reader_items=[_item("ZZZ")], resolve=_unr, retrieved_fallback="2026-01-01T00:00:00+00:00"
    )
    assert c.ticker is None
    assert c.subject_resolution == ResolutionStatus.UNRESOLVED


def test_claim_enriched_dict_shapes_persisted_row(tmp_path: Path):
    del tmp_path
    (c,) = build_evidence_claims(
        reader_items=[_item("ABC")], resolve=_amb, retrieved_fallback="2026-01-01T00:00:00+00:00"
    )
    d = claim_to_enriched_dict(c)
    assert d["subject_resolution"] == "ambiguous" and d["reported_ticker"] == "ABC"


def test_planner_exposes_no_portfolio_or_snapshot():
    for fn in (plan_public_search_queries, suggest_public_search_queries):
        params = set(inspect.signature(fn).parameters)
        assert not {p for p in params if "portfolio" in p or "snapshot" in p or p == "data_root"}
        assert "include_portfolio" not in params
    queries = plan_public_search_queries(primary_name="Acme", primary_ticker="ACME", related_names=["Beta"])
    assert queries
    for q in queries:
        query = q["query"]
        assert isinstance(query, str)
        assert "TSLA" not in query
    queries2 = suggest_public_search_queries("e1", "Acme", "ACME", relationships=(), names_by_entity={})
    for q in queries2:
        query2 = q["query"]
        assert isinstance(query2, str)
        assert "TSLA" not in query2


_NVDA_TABLE = (
    "NVIDIA Corporation\n"
    "(in million)\n"
    "Three Months Ended | Six Months Ended\n"
    "Jul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025\n"
    "$ | $ | $ | $\n"
    "Revenue | $ | 96,221 | $ | 46,743 | $ | 177,837 | $ | 90,805"
)


@pytest.mark.parametrize(
    "period,value",
    [("Three Months Ended Jul 26, 2026", "96,221"), ("Six Months Ended Jul 26, 2026", "177,837")],
)
def test_two_level_period_proof_rebuild_and_wording(period: str, value: str) -> None:
    from app.research.service import _binding_defect

    raw = _NVDA_TABLE.encode()
    provenance = evidence.sec_source_ref(
        accession_no="0001045810-26-000001",
        document_name="nvda.htm",
        passage=_NVDA_TABLE,
        offset=0,
        end=len(_NVDA_TABLE),
        basis="raw",
        text_hash=sha256(raw).hexdigest(),
    )
    selectors = {
        "metric": "Revenue",
        "value": value,
        "entity": "NVIDIA Corporation",
        "period": period,
        "unit": "million",
    }
    proof = evidence.observation_proof(provenance=provenance, source_bytes=raw, **selectors)
    point = evidence.canonical_datapoint(provenance=provenance, source_bytes=raw, **selectors)
    locator = proof["locator"]
    assert isinstance(locator, dict)
    assert locator["datapoint"] == point
    assert point["period"] == period
    row = evidence.Evidence(
        evidence_id="EV-TABLE",
        session_id="S-TABLE",
        wave_id=1,
        source_type="sec",
        source_name="SEC",
        subject="NVDA",
        claim_text="Revenue",
        content=_NVDA_TABLE,
        content_hash=evidence.evidence_content_hash(_NVDA_TABLE),
        retrieved_at=datetime(2026, 10, 27, tzinfo=UTC),
        provenance=provenance,
        proof=proof,
        value=value,
        entity="NVIDIA Corporation",
        period=period,
        unit="million",
    )
    assert evidence.verify_observation(row, value=value, source_bytes=raw).status == "verified"
    claim = f"NVIDIA Corporation reported Revenue of ${value} million for the {period}"
    assert _binding_defect(claim, [point], "observed_fact") is None
    assert _binding_defect(claim + " with 123 extra", [point], "observed_fact") is not None
    with pytest.raises(ValueError, match="another column"):
        evidence.canonical_datapoint(provenance=provenance, source_bytes=raw, **{**selectors, "period": "Jul 26, 2026"})


@pytest.mark.parametrize(
    "passage,value",
    [
        (_NVDA_TABLE, "96,221"),
        (_NVDA_TABLE.replace("(in million)", "(In millions, except per share data)"), "96,221"),
        ("NVIDIA Corporation FY2025 Revenue was $46.7 billion.", "46.7"),
    ],
)
def test_omitted_unit_never_drops_a_stated_scale(passage: str, value: str) -> None:
    """A declaration without the source's scale fails closed; publishing "96,221" for 96,221 million is wrong."""
    raw = passage.encode()
    provenance = evidence.sec_source_ref(
        accession_no="0001045810-26-000001",
        document_name="nvda.htm",
        passage=passage,
        offset=0,
        end=len(passage),
        basis="raw",
        text_hash=sha256(raw).hexdigest(),
    )
    period = "FY2025" if "FY2025" in passage else "Three Months Ended Jul 26, 2026"
    selectors = {"metric": "Revenue", "value": value, "entity": "NVIDIA Corporation", "period": period}
    with pytest.raises(ValueError, match="states the scale"):
        evidence.canonical_datapoint(provenance=provenance, source_bytes=raw, **selectors)
    # The declared unit uses the source spelling ("millions" under "(In millions, ...)").
    unit = next(word for word in ("billion", "millions", "million") if word in passage)
    assert (
        evidence.canonical_datapoint(provenance=provenance, source_bytes=raw, **selectors, unit=unit)["value"] == value
    )


_AMD_TABLE = (
    "Advanced Micro Devices, Inc.\n"
    "Three Months Ended | Six Months Ended\n"
    "June 27,2026 | June 28,2025 | June 27,2026 | June 28,2025\n"
    "(In millions, except per share amounts)\n"
    "Net revenue | $ | 11,536 | $ | 7,685 | $ | 21,789 | $ | 15,123\n"
    "Cost of sales | 5,000 | 4,000 | 9,000 | 8,000\n"
    "Total net revenue | $ | 11,536 | $ | 7,685 | $ | 21,789 | $ | 15,123"
)


def _amd_point(passage: str, **selectors: str) -> dict[str, JSONValue]:
    raw = passage.encode()
    provenance = evidence.sec_source_ref(
        accession_no="0000002488-26-000001",
        document_name="amd.htm",
        passage=passage,
        offset=0,
        end=len(passage),
        basis="raw",
        text_hash=sha256(raw).hexdigest(),
    )
    claimed = {"value": "11,536", "entity": "Advanced Micro Devices, Inc.", "unit": "millions"}
    claimed["period"] = "Three Months Ended June 27,2026"
    return evidence.canonical_datapoint(provenance=provenance, source_bytes=raw, **{**claimed, **selectors})


@seed(20261027)
@given(order=st.permutations(["Net revenue", "Total net revenue"]))
def test_repeated_value_binds_the_one_row_its_metric_names(order: list[str]) -> None:
    """A value on several rows binds by its row label; a label that binds twice stays ambiguous."""
    from app.research.service import _binding_defect

    for metric in order:
        point = _amd_point(_AMD_TABLE, metric=metric)
        assert point["metric"] == metric
        assert str(point["statement"]).startswith(f"{metric} |")
        sentence = f"Advanced Micro Devices, Inc. reported {metric} of $11,536 million for the Three Months Ended June 27,2026."
        assert _binding_defect(sentence, [point], "observed_fact") is None
        assert _binding_defect(sentence + " It also had 42 more.", [point], "observed_fact") is not None
    with pytest.raises(ValueError, match="none of its 2 occurrences"):
        _amd_point(_AMD_TABLE, metric="Gross margin")
    with pytest.raises(ValueError, match="more than once"):
        _amd_point(_AMD_TABLE + "\nNet revenue | $ | 11,536 | $ | 1 | $ | 2 | $ | 3", metric="Net revenue")
    # A conventional "June 27, 2026" selects the source's "June 27,2026"; the source spelling is kept.
    spaced = _amd_point(_AMD_TABLE, metric="Net revenue", period="Three Months Ended June 27, 2026")
    assert spaced["period"] == "Three Months Ended June 27,2026"
    with pytest.raises(ValueError, match="none of its 2 occurrences"):
        _amd_point(_AMD_TABLE, metric="Net revenue", period="Three Months Ended June 27, 20 26")


@seed(20261027)
@given(width=st.integers(1, 6), index=st.integers(0, 11), chosen=st.integers(0, 1))
def test_two_level_period_never_borrows_another_group(width: int, index: int, chosen: int) -> None:
    count = width * 2
    index %= count
    groups = ["Three Months Ended", "Six Months Ended"]
    dates = [f"Jul {day + 1}, 2026" for day in range(width)] * 2
    before = " | ".join(groups) + "\n" + " | ".join(dates) + "\n"
    wanted = f"{groups[chosen]} {dates[index]}"
    if index // width == chosen:
        assert evidence._grouped_period(before, wanted, (index, count)) == wanted
    else:
        with pytest.raises(ValueError, match="duration group"):
            evidence._grouped_period(before, wanted, (index, count))


@seed(20261027)
@given(count=st.integers(1, 12), index=st.integers(0, 11), year=st.integers(2000, 2030))
def test_single_header_source_spelling_stays_unchanged(count: int, index: int, year: int) -> None:
    index %= count
    cells = [f"FY {year + i}" for i in range(count)]
    before = " | ".join(cells) + "\n"
    claimed = {"period": cells[index].lower(), "entity": "NVDA", "unit": None, "value": "100"}
    assert evidence._context_label(before, len(before), "Revenue", claimed, "period", (index, count)) == cells[index]


@pytest.mark.parametrize(
    "headers,reason",
    [
        ("Jul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025", "no duration"),
        (
            (
                "Three Months Ended | Six Months Ended | Nine Months Ended\n"
                "Jul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025"
            ),
            "uneven",
        ),
        (
            "Three Months Ended | Three Months Ended\nJul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025",
            "ambiguous",
        ),
        ("Three Months Ended | Six Months Ended\nJul 26, 2026 | Jul 26, 2026 | Jul 26, 2026 | Jul 27, 2025", "repeats"),
        (
            "Three Months Ended | Six Months Ended\nJul 27, 2025 | Jul 26, 2026 | Jul 26, 2026 | Jul 27, 2025",
            "another date",
        ),
        ("Three Months Ended | Six Months Ended\nno dates", "no aligned"),
    ],
)
def test_two_level_period_rejects_ambiguous_headers(headers: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        evidence._grouped_period(headers, "Three Months Ended Jul 26, 2026", (0, 4))


def test_finra_latest_record_proves_one_scalar_datapoint() -> None:
    row = {"symbolCode": "NVDA", "settlementDate": "2026-10-15", "currentShortPositionQuantity": 69350469}
    payload: dict[str, object] = {
        "dataset": "consolidatedShortInterest",
        "metrics": {"fields": {}},
        "latest_record": row,
    }
    raw = json.dumps(payload).encode()
    provenance = evidence.finra_record_ref(
        tool_name="get_short_interest",
        record_identity=json.dumps(row),
        dataset="otcMarket/consolidatedShortInterest",
        tool_result_id="tr-finra",
    )
    assert evidence.cited_record(payload, json.dumps(row)) == row
    proof = evidence.observation_proof(
        provenance=provenance,
        source_bytes=raw,
        metric="currentShortPositionQuantity",
        value="69350469",
        entity="NVDA",
        period="2026-10-15",
    )
    observed = evidence.Evidence(
        evidence_id="EV-FINRA",
        session_id="S-FINRA",
        wave_id=1,
        source_type="finra",
        source_name="FINRA",
        subject="NVDA",
        claim_text="Short interest",
        content="Short interest",
        content_hash=evidence.evidence_content_hash("Short interest"),
        retrieved_at=datetime(2026, 10, 27, tzinfo=UTC),
        provenance=provenance,
        proof=proof,
        value="69350469",
        entity="NVDA",
        period="2026-10-15",
    )
    assert evidence.verify_observation(observed, value="69350469", source_bytes=raw).status == "verified"


@pytest.mark.parametrize("key", ["metrics", "latest_record", "records"])
def test_citable_record_requires_complete_exact_json(key: str) -> None:
    row = {"symbolCode": "NVDA", "settlementDate": "2026-10-15", "label": "two  spaces", "v": 10}
    payload = {key: [row] if key == "records" else row}
    assert evidence.cited_record(payload, json.dumps(row, indent=2)) == row
    assert evidence.cited_record({"result": payload}, json.dumps(row)) == row
    assert evidence.cited_record(payload, json.dumps({"v": 10})) is None
    assert evidence.cited_record(payload, json.dumps({**row, "label": "two spaces"})) is None
    assert evidence.cited_record(payload, json.dumps(payload)) == payload
    for locator in ("", " ", "not JSON", "null", "[]", "10", None):
        assert evidence.cited_record(payload, locator) is None
    assert evidence.cited_record({key: 10}, json.dumps(row)) is None


def test_context_label_requires_identity_but_allows_no_unit() -> None:
    claimed: dict[str, str | None] = {"entity": None, "period": None, "unit": None, "value": "100"}
    assert evidence._context_label("", 0, "", claimed, "unit", (0, 2)) is None
    for label in ("entity", "period"):
        with pytest.raises(ValueError, match=f"datapoint {label} is required"):
            evidence._context_label("", 0, "", claimed, label, (0, 2))
    claimed["period"] = "FY2026"
    with pytest.raises(ValueError, match="period 'FY2026' is not stated in the value's table row"):
        evidence._context_label("", 0, "", claimed, "period", (0, 2))
