"""Unit tests for the deterministic SEC/FINRA normalizers (app/normalization.py).

Fixture payloads mirror the shapes seeded in tests/test_analytics_screens.py;
no network access.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandera.errors import SchemaError

from app.normalization import (
    COMPANY_FACTS_PARSER_VERSION,
    COMPANY_TICKERS_PARSER_VERSION,
    SHARES_OUTSTANDING_CONCEPT,
    SHORT_INTEREST_PARSER_VERSION,
    normalize_finra_short_interest,
    normalize_sec_company_facts,
    normalize_sec_tickers,
)

RETRIEVED_AT = "2026-08-10T12:00:00Z"


def _tickers_payload(cik: int = 1) -> dict[str, dict[str, str | int]]:
    return {"0": {"cik_str": cik, "ticker": "AAA", "title": "Alpha Corp"}}


def _facts_payload(cik: int = 1, facts: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": facts or []}},
            }
        },
    }


def _eps_payload(
    cik: int = 1,
    diluted: list[dict[str, object]] | None = None,
    basic: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Companyfacts payload with us-gaap EPS concepts (USD/shares units)."""
    us_gaap: dict[str, object] = {}
    if diluted is not None:
        us_gaap["EarningsPerShareDiluted"] = {"units": {"USD/shares": diluted}}
    if basic is not None:
        us_gaap["EarningsPerShareBasic"] = {"units": {"USD/shares": basic}}
    return {"cik": cik, "entityName": f"CIK{cik}", "facts": {"us-gaap": us_gaap}}


def _normalize(payload: object) -> dict[str, list[dict[str, object]]]:
    return normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )


def test_ticker_normalization():
    datasets = normalize_sec_tickers(_tickers_payload(cik=1), retrieved_at=RETRIEVED_AT, content_hash="h1")
    entities = datasets["entities"]
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "sec:cik:0000000001"
    assert entities[0]["source"] == "sec:company_tickers"
    assert entities[0]["known_at"] == RETRIEVED_AT
    assert entities[0]["retrieved_at"] == RETRIEVED_AT
    assert entities[0]["content_hash"] == "h1"
    assert entities[0]["parser_version"] == COMPANY_TICKERS_PARSER_VERSION

    alias = datasets["entity_aliases"][0]
    assert alias["alias_type"] == "ticker"
    assert alias["alias_value"] == "AAA"
    assert alias["entity_id"] == "sec:cik:0000000001"
    assert alias["security_id"] == "sec:equity:0000000001"
    assert alias["source"] == "sec:company_tickers"
    assert alias["known_at"] == RETRIEVED_AT
    assert alias["retrieved_at"] == RETRIEVED_AT
    assert alias["content_hash"] == "h1"
    assert alias["parser_version"] == COMPANY_TICKERS_PARSER_VERSION


def test_ticker_normalization_skips_malformed_rows():
    raw = {
        "0": {"cik_str": "not-a-cik", "ticker": "BAD"},
        "1": {"cik_str": 2, "ticker": ""},
        "2": "junk",
    }
    datasets = normalize_sec_tickers(raw, retrieved_at=RETRIEVED_AT, content_hash="h1")
    assert datasets["entities"] == []
    assert datasets["entity_aliases"] == []


def test_facts_canonical_concepts():
    payload = _facts_payload(facts=[{"end": "2026-06-30", "val": 10, "accn": "a1", "filed": "2026-08-02"}])
    facts_env = payload["facts"]
    assert isinstance(facts_env, dict)
    dei_env = facts_env["dei"]
    assert isinstance(dei_env, dict)
    dei_env["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [{"end": "2026-06-30", "val": 123.5, "accn": "a2", "filed": "2026-08-02"}]},
    }
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
        source_record_id="cik0000000001",
    )
    facts = {f["concept"]: f for f in datasets["financial_facts"]}
    assert facts["Revenue"]["original_concept"] == "dei:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert facts["Revenue"]["value"] == 123.5
    assert facts["Revenue"]["unit"] == "USD"
    assert facts["Revenue"]["namespace"] == "dei"
    assert facts["Revenue"]["taxonomy"] == "dei"
    assert facts["Revenue"]["source_identity"] == {
        "end": "2026-06-30",
        "val": 123.5,
        "accn": "a2",
        "filed": "2026-08-02",
    }
    assert facts[SHARES_OUTSTANDING_CONCEPT]["unit"] == "shares"


def test_fact_known_at_is_filed_at_not_retrieved_at():
    payload = _facts_payload(facts=[{"end": "2026-06-30", "val": 100, "accn": "a1", "filed": "2026-08-02"}])
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )
    fact = datasets["financial_facts"][0]
    assert fact["known_at"] == "2026-08-02"
    assert fact["retrieved_at"] == RETRIEVED_AT
    assert fact["known_at"] != fact["retrieved_at"]


def test_malformed_facts_are_skipped_without_crash():
    payload = _facts_payload(
        facts=[
            {"end": "2026-06-30", "val": "not-a-number", "accn": "a1", "filed": "2026-08-02"},
            {"end": "2026-06-30", "val": 1, "accn": "a2"},  # missing filed
            {"end": "2026-06-30", "val": 2, "filed": "2026-08-02"},  # missing accn
            {"val": 3, "accn": "a4", "filed": "2026-08-02"},  # missing end
        ]
    )
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )
    assert datasets["financial_facts"] == []


def test_non_usd_units_are_ignored():
    payload = {
        "cik": 1,
        "facts": {
            "dei": {
                "Revenues": {"units": {"EUR": [{"end": "2026-06-30", "val": 10, "accn": "a1", "filed": "2026-08-02"}]}},
            }
        },
    }
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )
    assert not any(f["concept"] == "Revenue" for f in datasets["financial_facts"])


def test_security_classification():
    with_facts = normalize_sec_company_facts(
        _facts_payload(facts=[{"end": "2026-06-30", "val": 100, "accn": "a1", "filed": "2026-08-02"}]),
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )
    assert with_facts["securities"][0]["security_type"] == "equity-common"

    empty = normalize_sec_company_facts(
        _facts_payload(),
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="cik0000000001",
    )
    assert empty["securities"][0]["security_type"] == "unknown"
    assert empty["securities"][0]["parser_version"] == COMPANY_FACTS_PARSER_VERSION


def test_short_interest_normalization():
    rows: list[dict[str, object]] = [
        {
            "symbolCode": " aaa ",
            "issueName": "  Alpha  ",
            "currentShortPositionQuantity": "-5",
            "settlementDate": "2026-08-14",
        }
    ]
    datasets = normalize_finra_short_interest(
        rows,
        settlement_date="2026-08-14",
        retrieved_at=RETRIEVED_AT,
        content_hash="h1",
        source_url="u",
        source_record_id="r",
    )
    (row,) = datasets["short_interest"]
    assert row["symbol_code"] == "AAA"
    assert row["issue_name"] == "Alpha"
    assert row["short_position"] is None  # negative position -> None
    assert row["days_to_cover"] is None  # missing -> None
    assert row["parser_version"] == SHORT_INTEREST_PARSER_VERSION
    assert row["row_id"] == "finra:row:2026-08-14:AAA:h1"
    assert row["known_at"] == RETRIEVED_AT  # conservative: publication unknown, retrieval is known_at
    assert row["retrieved_at"] == RETRIEVED_AT


def test_short_interest_corrected_snapshot_is_new_version():
    rows: list[dict[str, object]] = [
        {"symbolCode": "AAA", "currentShortPositionQuantity": 20, "settlementDate": "2026-08-14"}
    ]
    v1 = normalize_finra_short_interest(
        rows,
        settlement_date="2026-08-14",
        retrieved_at=RETRIEVED_AT,
        content_hash="v1-hash",
        source_url="u",
        source_record_id="r",
    )
    v2 = normalize_finra_short_interest(
        rows,
        settlement_date="2026-08-14",
        retrieved_at=RETRIEVED_AT,
        content_hash="v2-hash",
        source_url="u",
        source_record_id="r",
    )
    assert v1["short_interest"][0]["row_id"] != v2["short_interest"][0]["row_id"]


def test_short_interest_explicit_known_at_wins_with_bounds():
    datasets = normalize_finra_short_interest(
        [{"symbolCode": "AAA", "currentShortPositionQuantity": 20}],
        settlement_date="2025-12-15",
        known_at="2025-12-24T08:00:00Z",
        retrieved_at="2025-12-24T12:00:00Z",
        content_hash="h1",
        source_url="u",
        source_record_id="r",
    )
    (row,) = datasets["short_interest"]
    assert row["known_at"] == "2025-12-24T08:00:00Z"
    assert row["retrieved_at"] == "2025-12-24T12:00:00Z"


def test_short_interest_explicit_known_at_rejects_out_of_bounds():
    import pytest

    with pytest.raises(ValueError, match="precedes settlement_date"):
        normalize_finra_short_interest(
            [{"symbolCode": "AAA"}],
            settlement_date="2025-12-15",
            known_at="2025-12-14T23:00:00Z",
            retrieved_at="2025-12-24T12:00:00Z",
            content_hash="h1",
            source_url="u",
            source_record_id="r",
        )
    with pytest.raises(ValueError, match="exceeds retrieved_at"):
        normalize_finra_short_interest(
            [{"symbolCode": "AAA"}],
            settlement_date="2025-12-15",
            known_at="2025-12-25T00:00:00Z",
            retrieved_at="2025-12-24T12:00:00Z",
            content_hash="h1",
            source_url="u",
            source_record_id="r",
        )


def test_short_interest_known_at_compares_instants_not_strings():
    import pytest

    with pytest.raises(ValueError, match="exceeds retrieved_at"):
        normalize_finra_short_interest(
            [{"symbolCode": "AAA"}],
            settlement_date="2025-12-15",
            known_at="2025-12-24T10:00:00-08:00",
            retrieved_at="2025-12-24T17:00:00Z",
            content_hash="h1",
            source_url="u",
            source_record_id="r",
        )
    datasets = normalize_finra_short_interest(
        [{"symbolCode": "AAA"}],
        settlement_date="2025-12-15",
        known_at="2025-12-24T01:00:00-08:00",
        retrieved_at="2025-12-24T12:00:00Z",
        content_hash="h1",
        source_url="u",
        source_record_id="r",
    )
    (row,) = datasets["short_interest"]
    assert row["known_at"] == "2025-12-24T01:00:00-08:00"


def test_eps_facts_normalized_with_period_metadata():
    diluted: list[dict[str, object]] = [
        {
            "start": "2025-05-01",
            "end": "2025-07-31",
            "val": 1.5,
            "accn": "a1",
            "fy": 2025,
            "fp": "Q2",
            "filed": "2025-08-28",
        }
    ]
    basic: list[dict[str, object]] = [
        {
            "start": "2025-05-01",
            "end": "2025-07-31",
            "val": 1.52,
            "accn": "a2",
            "fy": 2025,
            "fp": "Q2",
            "filed": "2025-08-28",
        }
    ]
    datasets = _normalize(_eps_payload(diluted=diluted, basic=basic))
    facts = {f["concept"]: f for f in datasets["financial_facts"]}
    assert set(facts) == {"EarningsPerShareDiluted", "EarningsPerShareBasic"}
    diluted_row = facts["EarningsPerShareDiluted"]
    assert diluted_row["original_concept"] == "us-gaap:EarningsPerShareDiluted"
    assert diluted_row["unit"] == "USD/shares"
    assert diluted_row["value"] == 1.5
    assert diluted_row["duration_type"] == "duration"
    assert diluted_row["period_start"] == "2025-05-01"
    assert diluted_row["period_end"] == "2025-07-31"
    assert diluted_row["fiscal_year"] == 2025
    assert diluted_row["fiscal_period"] == "Q2"
    assert diluted_row["known_at"] == "2025-08-28"
    assert diluted_row["parser_version"] == "sec-companyfacts-v6"
    assert diluted_row["entity_id"] == "sec:cik:0000000001"
    assert diluted_row["cik"] == 1
    assert diluted_row["issuer_cik"] == 1
    assert diluted_row["namespace"] == "us-gaap"
    assert diluted_row["taxonomy"] == "us-gaap"
    assert diluted_row["source_identity"] == diluted[0]
    assert diluted_row["dimensions"] == {}
    assert diluted_row["filed_at"] == "2025-08-28"
    assert diluted_row["accession"] == "a1"
    assert diluted_row["source_url"] == "u"
    assert diluted_row["content_hash"] == "h1"
    assert diluted_row["source_record_id"] == "cik0000000001"
    assert diluted_row["retrieved_at"] == RETRIEVED_AT
    assert diluted_row["parser_version"] == COMPANY_FACTS_PARSER_VERSION


def test_eps_instant_fact_without_period_metadata_is_nullable():
    datasets = _normalize(
        _eps_payload(
            diluted=[
                {"end": "2025-01-31", "val": 0.01, "accn": "a1", "filed": "2025-03-03"},
            ]
        )
    )
    (row,) = datasets["financial_facts"]
    assert row["duration_type"] == "instant"
    assert row["period_start"] is None
    assert row["fiscal_year"] is None
    assert row["fiscal_period"] is None
    assert row["period_instant"] == "2025-01-31"
    assert row["dimensions"] == {}


@pytest.mark.parametrize("duration", [False, True], ids=["instant", "duration"])
def test_fact_identity_preserves_context_dimensions_and_source(duration: bool) -> None:
    raw_fact: dict[str, object] = {
        "end": "2025-07-31",
        "val": 1.5,
        "accn": "0000000001-25-000123",
        "fy": 2025,
        "fp": "Q2",
        "filed": "2025-08-28",
        "form": "10-Q/A",
        "dimensions": {"us-gaap:StatementBusinessSegmentsAxis": "example:CloudMember"},
        "document_name": "alpha-20250731.htm",
        "context_id": "cloud-quarter",
        "unit_id": "usd-per-share",
        "element_id": "diluted-eps-cloud",
        "locator": "#diluted-eps-cloud",
        "decimals": "2",
        "precision": "INF",
        "scale": "0",
        "presentation_location": "income-statement",
    }
    if duration:
        raw_fact["start"] = "2025-05-01"
    (row,) = _normalize(_eps_payload(diluted=[raw_fact]))["financial_facts"]
    assert row["source_identity"] == raw_fact
    assert row["source_identity"] is not raw_fact
    assert row["original_concept"] == "us-gaap:EarningsPerShareDiluted"
    assert row["concept"] == "EarningsPerShareDiluted"
    assert row["unit"] == "USD/shares"
    assert row["duration_type"] == ("duration" if duration else "instant")
    assert row["period_start"] == ("2025-05-01" if duration else None)
    assert row["period_end"] == "2025-07-31"
    assert row["period_instant"] == (None if duration else "2025-07-31")
    assert row["known_at"] == row["filed_at"] == "2025-08-28"
    for field in (
        "dimensions",
        "document_name",
        "context_id",
        "unit_id",
        "element_id",
        "locator",
        "decimals",
        "precision",
        "scale",
        "presentation_location",
    ):
        assert row[field] == raw_fact[field]


def test_equal_fact_values_keep_separate_contexts_and_amendment_identity() -> None:
    original: dict[str, object] = {
        "start": "2025-05-01",
        "end": "2025-07-31",
        "val": 1.5,
        "accn": "0000000001-25-000123",
        "filed": "2025-08-28",
        "form": "10-Q",
        "context_id": "consolidated",
        "dimensions": {},
    }
    segment = {
        **original,
        "context_id": "cloud",
        "dimensions": {"us-gaap:StatementBusinessSegmentsAxis": "example:CloudMember"},
    }
    amended = {
        **original,
        "accn": "0000000001-25-000124",
        "filed": "2025-09-05",
        "form": "10-Q/A",
        "context_id": "consolidated-amended",
    }
    rows = _normalize(_eps_payload(diluted=[original, segment, amended]))["financial_facts"]
    assert len(rows) == 3
    assert [row["source_identity"] for row in rows] == [original, segment, amended]
    assert [row["context_id"] for row in rows] == ["consolidated", "cloud", "consolidated-amended"]
    assert [row["dimensions"] for row in rows] == [{}, segment["dimensions"], {}]
    assert [row["known_at"] for row in rows] == ["2025-08-28", "2025-08-28", "2025-09-05"]


@pytest.mark.parametrize("reverse", [False, True], ids=["null-prefix", "duration-prefix"])
def test_mixed_facts_preserve_metadata_after_long_null_prefix(reverse: bool) -> None:
    instant: list[dict[str, object]] = [
        {
            "end": "2025-07-31",
            "val": 100 + index,
            "accn": f"instant-{index}",
            "filed": "2025-08-28",
            "start": None,
            "fy": None,
            "fp": None,
            "frame": None,
        }
        for index in range(128)
    ]
    duration: list[dict[str, object]] = [
        {
            "start": "2025-05-01",
            "end": "2025-07-31",
            "val": 1.5,
            "accn": "duration-full",
            "filed": "2025-08-28",
            "fy": 2025,
            "fp": "Q2",
            "frame": "CY2025Q2",
        },
        {
            "start": "2025-05-01",
            "end": "2025-07-31",
            "val": 1.52,
            "accn": "duration-null",
            "filed": "2025-08-28",
            "fy": None,
            "fp": None,
            "frame": None,
        },
    ]
    facts = instant + duration
    if reverse:
        facts.reverse()
    rows = _normalize(_eps_payload(diluted=facts))["financial_facts"]
    assert [row["accession"] for row in rows] == [fact["accn"] for fact in facts]
    metadata = {
        row["accession"]: (
            row["duration_type"],
            row["period_start"],
            row["fiscal_year"],
            row["fiscal_period"],
            row["frame"],
        )
        for row in rows
    }
    assert metadata == {
        **{fact["accn"]: ("instant", None, None, None, None) for fact in instant},
        "duration-full": ("duration", "2025-05-01", 2025, "Q2", "CY2025Q2"),
        "duration-null": ("duration", "2025-05-01", None, None, None),
    }
    assert type(next(row["fiscal_year"] for row in rows if row["accession"] == "duration-full")) is int


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(
        st.fixed_dictionaries(
            {},
            optional={
                "start": st.one_of(st.none(), st.sampled_from(["2025-01-01", "2025-05-01"])),
                "fy": st.one_of(st.none(), st.integers(min_value=1900, max_value=2100)),
                "fp": st.one_of(st.none(), st.sampled_from(["FY", "Q1", "Q2", "Q3", "Q4"])),
                "frame": st.one_of(st.none(), st.sampled_from(["CY2025", "CY2025Q2", "CY2025Q2I"])),
                "context_id": st.one_of(st.none(), st.sampled_from(["quarter", "year-to-date"])),
                "unit_id": st.one_of(st.none(), st.sampled_from(["usd-per-share", "usd-per-share-2"])),
                "element_id": st.one_of(st.none(), st.sampled_from(["eps-primary", "eps-repeat"])),
                "dimensions": st.dictionaries(
                    st.sampled_from(["us-gaap:StatementBusinessSegmentsAxis"]),
                    st.sampled_from(["example:CloudMember", "example:HardwareMember"]),
                    max_size=1,
                ),
            },
        ),
        min_size=1,
        max_size=8,
    )
)
def test_prop_optional_fact_metadata_preserves_values(metadata: list[dict[str, object]]) -> None:
    facts: list[dict[str, object]] = [
        {
            "end": "2025-07-31",
            "val": 1.5,
            "accn": f"a{index}",
            "filed": "2025-08-28",
            **fields,
        }
        for index, fields in enumerate(metadata)
    ]
    rows = _normalize(_eps_payload(diluted=facts))["financial_facts"]
    assert len(rows) == len(facts)
    for row, raw_fact, fields in zip(rows, facts, metadata, strict=True):
        assert row["source_identity"] == raw_fact
        assert row["dimensions"] == fields.get("dimensions", {})
        for field in ("context_id", "unit_id", "element_id"):
            if field in fields:
                assert row[field] == fields[field]
        assert (
            row["duration_type"],
            row["period_start"],
            row["fiscal_year"],
            row["fiscal_period"],
            row["frame"],
        ) == (
            "duration" if fields.get("start") else "instant",
            fields.get("start"),
            fields.get("fy"),
            fields.get("fp"),
            fields.get("frame"),
        )
        if fields.get("fy") is not None:
            assert type(row["fiscal_year"]) is int


@settings(max_examples=40, derandomize=True, deadline=None)
@given(st.integers(min_value=101, max_value=256), st.sampled_from(["first", "middle", "last"]))
def test_prop_fact_metadata_does_not_depend_on_order(null_count: int, position: str) -> None:
    facts: list[dict[str, object]] = [
        {"end": "2025-07-31", "val": index, "accn": f"a{index}", "filed": "2025-08-28"} for index in range(null_count)
    ]
    facts.insert(
        {"first": 0, "middle": null_count // 2, "last": null_count}[position],
        {
            "start": "2025-05-01",
            "end": "2025-07-31",
            "val": 1.5,
            "accn": "duration",
            "filed": "2025-08-28",
            "fy": 2025,
            "fp": "Q2",
            "frame": "CY2025Q2",
        },
    )
    forward = _normalize(_eps_payload(diluted=facts))["financial_facts"]
    backward = _normalize(_eps_payload(diluted=list(reversed(facts))))["financial_facts"]
    assert forward == list(reversed(backward))


def test_fact_validation_rejects_invalid_known_at() -> None:
    with pytest.raises(SchemaError, match="_timestamp"):
        _normalize(_eps_payload(diluted=[{"end": "2025-07-31", "val": 1.5, "accn": "a1", "filed": "not-a-date"}]))


def test_eps_non_usd_shares_units_are_ignored():
    payload = _eps_payload()
    facts_env = payload["facts"]
    assert isinstance(facts_env, dict)
    gaap_env = facts_env["us-gaap"]
    assert isinstance(gaap_env, dict)
    gaap_env["EarningsPerShareDiluted"] = {
        "units": {
            "USD": [{"end": "2025-07-31", "val": 1.5, "accn": "a1", "filed": "2025-08-28"}],
            "shares": [{"end": "2025-07-31", "val": 2, "accn": "a2", "filed": "2025-08-28"}],
        },
    }
    datasets = _normalize(payload)
    assert datasets["financial_facts"] == []


def test_malformed_eps_facts_are_skipped_without_crash():
    datasets = _normalize(
        _eps_payload(
            diluted=[
                {"end": "2025-07-31", "val": "not-a-number", "accn": "a1", "filed": "2025-08-28"},
                {"end": "2025-07-31", "val": 1.0, "accn": "a2"},  # missing filed
                {"end": "2025-07-31", "val": 1.0, "filed": "2025-08-28"},  # missing accn
                {"val": 1.0, "accn": "a4", "filed": "2025-08-28"},  # missing end
            ]
        )
    )
    assert datasets["financial_facts"] == []


def test_eps_rows_deterministic_on_rerun():
    payload = _eps_payload(
        diluted=[{"end": "2025-07-31", "val": 1.5, "accn": "a1", "filed": "2025-08-28"}],
        basic=[{"end": "2025-07-31", "val": 1.52, "accn": "a2", "filed": "2025-08-28"}],
    )
    first = _normalize(payload)["financial_facts"]
    second = _normalize(payload)["financial_facts"]
    assert len(first) == 2
    assert first == second  # pure normalizer: same payload, same rows
    assert {f["concept"] for f in first} == {"EarningsPerShareDiluted", "EarningsPerShareBasic"}


def test_ambiguous_xbrl_group_emits_undated_events():
    payload = {
        "cik": 21344,
        "entityName": "CIK21344",
        "facts": {
            "us-gaap": {
                "DividendsPayableAmountPerShare": {
                    "units": {
                        "USD/shares": [
                            {"val": 0.50, "accn": "A1", "filed": "2026-08-01"},
                            {"val": 0.54, "accn": "A1", "filed": "2026-08-01"},
                        ]
                    }
                },
                "DividendPayableDateToBePaidDayMonthAndYear": {
                    "units": {
                        "USD": [
                            {"val": "2026-09-01", "accn": "A1", "filed": "2026-08-01"},
                            {"val": "2026-10-01", "accn": "A1", "filed": "2026-08-01"},
                        ]
                    }
                },
            }
        },
    }
    events = _normalize(payload)["dividend_events"]
    assert len(events) == 2
    assert all(e["declaration_date"] is None and e["record_date"] is None and e["payment_date"] is None for e in events)
    assert len({e["dividend_event_id"] for e in events}) == 2


def test_single_amount_single_payment_stays_paired():
    payload = {
        "cik": 21344,
        "entityName": "CIK21344",
        "facts": {
            "us-gaap": {
                "DividendsPayableAmountPerShare": {
                    "units": {
                        "USD/shares": [
                            {"val": 0.54, "accn": "A2", "filed": "2026-08-01"},
                        ]
                    }
                },
                "DividendPayableDateToBePaidDayMonthAndYear": {
                    "units": {
                        "USD": [
                            {"val": "2026-09-15", "accn": "A2", "filed": "2026-08-01"},
                        ]
                    }
                },
                "DividendsPayableDateOfRecordDayMonthAndYear": {
                    "units": {
                        "USD": [
                            {"val": "2026-08-29", "accn": "A2", "filed": "2026-08-01"},
                        ]
                    }
                },
            }
        },
    }
    (event,) = _normalize(payload)["dividend_events"]
    assert event["amount_per_share"] == 0.54
    assert event["payment_date"] == "2026-09-15"
    assert event["record_date"] == "2026-08-29"


def test_multiple_record_dates_emit_undated():
    payload = {
        "cik": 21344,
        "entityName": "CIK21344",
        "facts": {
            "us-gaap": {
                "DividendsPayableAmountPerShare": {
                    "units": {
                        "USD/shares": [
                            {"val": 0.54, "accn": "A3", "filed": "2026-08-01"},
                        ]
                    }
                },
                "DividendPayableDateToBePaidDayMonthAndYear": {
                    "units": {
                        "USD": [
                            {"val": "2026-09-15", "accn": "A3", "filed": "2026-08-01"},
                        ]
                    }
                },
                "DividendsPayableDateOfRecordDayMonthAndYear": {
                    "units": {
                        "USD": [
                            {"val": "2026-08-29", "accn": "A3", "filed": "2026-08-01"},
                            {"val": "2026-08-30", "accn": "A3", "filed": "2026-08-01"},
                        ]
                    }
                },
            }
        },
    }
    (event,) = _normalize(payload)["dividend_events"]
    assert event["declaration_date"] is None
    assert event["record_date"] is None
    assert event["payment_date"] is None


def test_xbrl_lineage_over_gateway_shaped_rows(monkeypatch: pytest.MonkeyPatch):
    """xbrl_lineage(entity_id, concept, as_of=...) over live gateway fact rows."""
    from app.sec import lineage

    rows = _normalize(
        _facts_payload(
            facts=[
                {"end": "2026-06-30", "val": 100, "accn": "a1", "filed": "2026-08-02"},
                {"end": "2026-06-30", "val": 120, "accn": "a2", "filed": "2026-09-05"},
            ]
        )
    )["financial_facts"]

    class _Gateway:
        def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
            assert cik == 1
            if as_of is None:
                return {"financial_facts": rows}
            gated = [r for r in rows if str(r.get("known_at") or "")[:10] <= as_of]
            return {"financial_facts": gated}

    monkeypatch.setattr("app.data_sources.SourceGateway", _Gateway)

    full = lineage.xbrl_lineage("sec:cik:0000000001", SHARES_OUTSTANDING_CONCEPT)
    assert [r["value"] for r in full] == [120.0, 100.0]  # newest period_end first
    assert full[0]["accession"] == "a2"
    gated = lineage.xbrl_lineage("sec:cik:0000000001", SHARES_OUTSTANDING_CONCEPT, as_of="2026-08-15")
    assert [r["value"] for r in gated] == [100.0]  # a2 known 2026-09-05 is gated out


def test_xbrl_period_lineage_detects_restatement():
    """period_lineage groups one period_end: earliest filed is original, later restated."""
    from app.sec import lineage

    rows = _normalize(
        _facts_payload(
            facts=[
                {"end": "2026-06-30", "val": 100, "accn": "a1", "filed": "2026-08-02"},
                {"end": "2026-06-30", "val": 120, "accn": "a2", "filed": "2026-09-05"},
            ]
        )
    )["financial_facts"]

    groups = lineage.period_lineage(rows)
    assert len(groups) == 1
    group = groups[0]
    assert group["period_end"] == "2026-06-30"
    original = group["originally_reported"]
    latest = group["latest"]
    assert isinstance(original, dict) and original["value"] == 100.0
    assert isinstance(latest, dict) and latest["value"] == 120.0
    assert group["restated"] is True
