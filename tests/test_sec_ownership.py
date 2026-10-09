"""Offline tests for app/sec/ownership.py (no network)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.sec import ownership
from app.sec.models import BeneficialOwnership


class _Person:
    def __init__(
        self,
        cik: str,
        name: str,
        sole_v: int = 0,
        shared_v: int = 0,
        sole_d: int = 0,
        shared_d: int = 0,
        agg: int = 0,
        pct: float | str = 0.0,
    ) -> None:
        self.cik = cik
        self.name = name
        self.sole_voting_power = sole_v
        self.shared_voting_power = shared_v
        self.sole_dispositive_power = sole_d
        self.shared_dispositive_power = shared_d
        self.aggregate_amount = agg
        self.percent_of_class = pct


def _schedule(persons: list[_Person], purpose: str | None = None) -> SimpleNamespace:
    items = SimpleNamespace(purpose_of_transaction=purpose)
    return SimpleNamespace(reporting_persons=persons, items=items)


def test_two_person_schedule():
    sched = _schedule(
        [
            _Person("1", "Alice", sole_v=100, shared_v=50, sole_d=100, shared_d=50, agg=1000, pct=5.5),
            _Person("2", "Bob", sole_v=0, shared_v=200, sole_d=0, shared_d=200, agg=2000, pct="7.25"),
        ],
        purpose="control",
    )
    recs = ownership.normalize_schedule(sched, issuer="ACME", form="SC 13D", filed_at="2024-01-15", accession_no="a1")
    assert len(recs) == 2
    assert recs[0].shares == 1000 and recs[0].percent == 5.5
    assert recs[0].sole_voting == 100 and recs[0].shared_dispositive == 50
    assert recs[1].shares == 2000 and recs[1].percent == 7.25
    assert recs[0].purpose_text == "control"
    assert recs[0].is_amendment is False


def test_amendment_diff_numbers_and_voting():
    prev = BeneficialOwnership(
        "Alice",
        "1",
        "ACME",
        "SC 13D",
        "2024-01-15",
        "a1",
        shares=1000,
        percent=5.0,
        sole_voting=100,
        purpose_text="control",
    )
    curr = BeneficialOwnership(
        "Alice",
        "1",
        "ACME",
        "SC 13D/A",
        "2024-06-15",
        "a2",
        shares=1500,
        percent=7.0,
        sole_voting=200,
        purpose_text="activist",
    )
    event = ownership.diff_ownership(prev, curr)
    assert event.share_change == 500
    assert event.percent_change == 7.0 - 5.0
    assert event.voting_changed is True
    assert event.text_changed is True
    assert event.previous_accession == "a1"
    assert event.current_accession == "a2"


def test_text_changed_needs_both_purposes():
    base = {
        "filer_name": "A",
        "filer_cik": "1",
        "issuer": "ACME",
        "form": "SC 13D",
        "filed_at": "2024-01-15",
        "accession_no": "a1",
    }
    both_same = ownership.diff_ownership(
        BeneficialOwnership(**{**base, "purpose_text": "x"}),
        BeneficialOwnership(**{**base, "purpose_text": "x", "accession_no": "a2"}),
    )
    assert both_same.text_changed is False
    missing = ownership.diff_ownership(
        BeneficialOwnership(**{**base, "purpose_text": None}),
        BeneficialOwnership(**{**base, "purpose_text": "y", "accession_no": "a2"}),
    )
    assert missing.text_changed is False
    missing2 = ownership.diff_ownership(
        BeneficialOwnership(**{**base, "purpose_text": "x"}),
        BeneficialOwnership(**{**base, "purpose_text": None, "accession_no": "a2"}),
    )
    assert missing2.text_changed is False


def test_changes_group_by_filer_and_skip_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(accession_no="a1", form="SC 13D", filed_at="2024-01-15", company="ACME"),
        SimpleNamespace(accession_no="bad", form="SC 13D", filed_at="2024-03-01", company="ACME"),
        SimpleNamespace(accession_no="a2", form="SC 13D/A", filed_at="2024-06-15", company="ACME"),
    ]

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)

    def fake_load(accession_no: str) -> SimpleNamespace:
        if accession_no == "bad":
            raise RuntimeError("boom")
        n = 1000 if accession_no == "a1" else 1500
        return _schedule([_Person("1", "Alice", agg=n)], purpose="p")

    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    events = ownership.get_ownership_changes("ACME")
    assert len(events) == 1
    assert events[0].share_change == 500
    assert events[0].previous_accession == "a1"
    assert events[0].current_accession == "a2"


def test_unparseable_schedule_yields_no_records():
    assert (
        ownership.normalize_schedule(
            SimpleNamespace(reporting_persons="not-a-list", items=None),
            issuer="ACME",
            form="SC 13G",
            filed_at=None,
            accession_no="a9",
        )
        == []
    )


def test_live_ownership_per_accession(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import Filing
    from app.sec.store import query_beneficial_ownership

    assert query_beneficial_ownership(subject_cik=320193, root=tmp_path) == []
    assert query_beneficial_ownership(root=tmp_path) == []
    sched = _schedule([_Person("999001", "Owner LP", agg=5000000, pct=6.2)], purpose="control")
    sched.issuer_info = SimpleNamespace(cik="320193", name="Subject Co")
    filing = Filing(
        form="SC 13D",
        accession_no="ACC-13D",
        filed_at="2024-03-10",
        filer_cik=999001,
        filer_name="Owner LP",
        accepted_at=None,
        known_at="2024-03-10T00:00:00Z",
        report_period=None,
        primary_document="primary",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )

    import app.sec.store as sec_store_mod

    def _fake_meta(accession: str, as_of: str | None = None) -> Filing:
        if accession == "ACC-13D":
            return filing
        raise ValueError(f"unknown accession {accession!r}")

    def _fake_load(accession_no: str) -> SimpleNamespace:
        assert accession_no == "ACC-13D"
        return sched

    monkeypatch.setattr(sec_store_mod, "_gateway_get_filing", _fake_meta)
    monkeypatch.setattr(ownership, "load_schedule", _fake_load)
    rows = query_beneficial_ownership(accession="ACC-13D", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["filer_cik"] == "999001"
    assert rows[0]["subject_cik"] == "320193"
    assert rows[0]["subject_name"] == "Subject Co"
    # Filer and subject never share a fallback identity.
    assert rows[0]["filer_cik"] != rows[0]["subject_cik"]


def test_13f_cusip_normalizes_live_per_accession(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import insider as _ins
    from app.sec.models import Filing
    from app.sec.store import query_13f_holdings

    assert query_13f_holdings(manager_cik=103567, root=tmp_path) == []
    filing = Filing(
        form="13F-HR",
        accession_no="ACC-DASH",
        filed_at="2024-05-15",
        filer_cik=103567,
        filer_name="Sample Manager LLC",
        accepted_at=None,
        known_at="2024-05-15T00:00:00Z",
        report_period="2024-03-31",
        primary_document="infotable.xml",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )

    import app.sec.documents as sec_docs
    import app.sec.store as sec_store_mod

    def _fake_meta(accession: str, as_of: str | None = None) -> Filing:
        if accession == "ACC-DASH":
            return filing
        raise ValueError(f"unknown accession {accession!r}")

    monkeypatch.setattr(sec_store_mod, "_gateway_get_filing", _fake_meta)
    table = [{"Cusip": "0378-33100", "Issuer": "Apple Inc", "ReportPeriod": "2024-03-31", "Class": "COM"}]

    def _fake_edgar(accession: str) -> SimpleNamespace:
        assert accession == "ACC-DASH"
        return SimpleNamespace(obj=lambda: SimpleNamespace(infotable=table))

    monkeypatch.setattr(sec_docs, "get_by_accession_number", _fake_edgar)
    # Dashed input normalizes to canonical on live per-accession resolution.
    rows = query_13f_holdings(accession="ACC-DASH", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["cusip"] == "037833100"
    assert rows[0]["security_id"] == "cusip:037833100"
    # Pure 13F normalization keeps distinct rows and stable ids per filing.
    base_row = {
        "Cusip": "037833100",
        "Issuer": "Apple Inc",
        "ReportPeriod": "2024-03-31",
        "Class": "COM",
        "SharesPrnAmount": 1000,
        "Value": 50000,
        "InvestmentDiscretion": "Sole",
        "OtherManager": "1",
        "Type": "Shares",
        "SoleVoting": 1000,
        "SharedVoting": 0,
        "NonVoting": 0,
    }
    rows_in = [dict(base_row), dict(base_row)]
    holdings = _ins.normalize_13f_holdings(
        rows_in,
        manager_name="M",
        manager_cik="5",
        accession_no="ACC-ROWS",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="infotable.xml",
        known_at="2024-05-15T00:00:00Z",
    )
    assert [h.source_row for h in holdings] == [1, 2]
    assert [h.shares_prn_type for h in holdings] == ["SH", "SH"]
    assert len({h.holding_id for h in holdings}) == 2
    repeat = _ins.normalize_13f_holdings(
        rows_in,
        manager_name="M",
        manager_cik="5",
        accession_no="ACC-ROWS",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="infotable.xml",
        known_at="2024-05-15T00:00:00Z",
    )
    assert [h.holding_id for h in repeat] == [h.holding_id for h in holdings]


def test_13f_shared_filing_rows_normalize_distinctly() -> None:
    from app.sec import insider as _ins

    base_row = {
        "Cusip": "037833100",
        "Issuer": "Apple Inc",
        "ReportPeriod": "2024-03-31",
        "Class": "COM",
        "SharesPrnAmount": 1000,
        "Value": 50000,
        "InvestmentDiscretion": "Sole",
        "OtherManager": "1",
        "Type": "Shares",
        "SoleVoting": 1000,
        "SharedVoting": 0,
        "NonVoting": 0,
    }
    third_row = {
        **base_row,
        "SharesPrnAmount": 2000,
        "Value": 60000,
        "SoleVoting": 1500,
        "SharedVoting": 500,
        "OtherManager": "2",
        "Type": "Principal",
    }
    rows = [dict(base_row), dict(base_row), dict(third_row)]
    holdings = _ins.normalize_13f_holdings(
        rows,
        manager_name="M",
        manager_cik="5",
        accession_no="ACC-ROWS",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="infotable.xml",
        known_at="2024-05-15T00:00:00Z",
    )
    assert [h.source_row for h in holdings] == [1, 2, 3]
    assert [h.shares_prn_type for h in holdings] == ["SH", "SH", "PRN"]
    assert [h.discretion for h in holdings] == ["Sole", "Sole", "Sole"]
    assert [h.other_manager for h in holdings] == ["1", "1", "2"]
    holding_ids = [h.holding_id for h in holdings]
    assert len(set(holding_ids)) == 3
    repeat = _ins.normalize_13f_holdings(
        rows,
        manager_name="M",
        manager_cik="5",
        accession_no="ACC-ROWS",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="infotable.xml",
        known_at="2024-05-15T00:00:00Z",
    )
    assert [h.holding_id for h in repeat] == holding_ids


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("037833100", "037833100"),
        ("0378-33100", "037833100"),
        ("0378 33100", "037833100"),
        (" 037833100 ", "037833100"),
        (None, None),
        ("", None),
        ("---", None),
    ],
)
def test_cusip_contract(raw: str | None, expected: str | None) -> None:
    from app.sec.cusip import cusip_security_id, normalize_cusip

    assert normalize_cusip(raw) == expected
    assert cusip_security_id("0378-33100") == "cusip:037833100"
    assert cusip_security_id("---") is None


def test_empty_cusip_is_none() -> None:
    from app.sec import insider as _ins

    # Empty-after-strip normalizes to None at the row-record boundary.
    empty = _ins._holding_row_to_record(
        {"Cusip": "---", "Issuer": "Apple Inc", "ReportPeriod": "2024-03-31"},
        manager_name="M",
        manager_cik="5",
        accession_no="ACC-EMPTY",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="d",
        known_at="2024-05-15T00:00:00Z",
        source_url=None,
        source_row=1,
    )
    assert empty.cusip is None
