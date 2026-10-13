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


def _schedule(
    persons: list[_Person], purpose: str | None = None, issuer_info: SimpleNamespace | None = None
) -> SimpleNamespace:
    items = SimpleNamespace(purpose_of_transaction=purpose)
    schedule = SimpleNamespace(reporting_persons=persons, items=items)
    if issuer_info is not None:
        schedule.issuer_info = issuer_info
    return schedule


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


def _window_filings(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(accession_no=f"a{i:02d}", form="SC 13D", filed_at=f"2024-{(i % 12) + 1:02d}-15")
        for i in range(count)
    ]


def test_bounded_limit_scans_newest_window(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(10)
    seen: dict[str, object] = {}
    loads: list[str] = []

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        seen.update(kwargs)
        limit = kwargs.get("limit")
        assert limit == 3  # max(2, 2 + 1) newest filings for limit=2
        return filings[:limit]

    def fake_load(accession_no: str) -> SimpleNamespace:
        loads.append(accession_no)
        assert accession_no in {"a00", "a01", "a02"}
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    events = ownership.get_ownership_changes("ACME", limit=2)
    assert isinstance(events, list)
    assert sorted(loads) == ["a00", "a01", "a02"]
    assert "limit" not in seen or seen["limit"] == 3


def test_bounded_window_newest_first_and_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(4)

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        assert kwargs.get("limit") == 3  # max(2, 2 + 1)
        return filings[:3]

    def fake_load(accession_no: str) -> SimpleNamespace:
        shares = {"a00": 1000, "a01": 1500, "a02": 1800}[accession_no]
        return _schedule([_Person("1", "Alice", agg=shares)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    events = ownership.get_ownership_changes("ACME", limit=2)
    assert [(e.previous_accession, e.current_accession) for e in events] == [("a01", "a02"), ("a00", "a01")]
    assert [e.share_change for e in events] == [300, 500]
    assert str(events[0].filed_at) >= str(events[1].filed_at)
    assert isinstance(events, list)
    assert all(isinstance(e.filed_at, str | type(None)) for e in events)


def test_explicit_none_stays_exhaustive(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(6)
    seen: dict[str, object] = {}
    loads = 0

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        seen.update(kwargs)
        assert kwargs.get("limit") is None
        return filings

    def fake_load(accession_no: str) -> SimpleNamespace:
        nonlocal loads
        loads += 1
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    ownership.get_ownership_changes("ACME", limit=None)
    assert loads == len(filings)
    assert seen["limit"] is None


def test_coverage_discloses_scan_and_missing_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(3)

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    def fake_load(accession_no: str) -> SimpleNamespace:
        if accession_no == "a02":
            return _schedule([_Person("2", "Bob", agg=2000)], purpose="p")
        shares = {"a00": 1000, "a01": 1500}[accession_no]
        return _schedule([_Person("1", "Alice", agg=shares)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    covered = ownership.get_ownership_changes_with_coverage("ACME", limit=2)
    assert covered["scanned_accessions"] == ["a00", "a01", "a02"]
    assert covered["scanned_count"] == 3
    assert covered["incomplete_history"] is True
    paired = covered["events"]
    assert isinstance(paired, list) and len(paired) == 1
    assert paired[0].share_change == 500
    assert (paired[0].previous_accession, paired[0].current_accession) == ("a00", "a01")
    notices = covered["missing_baselines"]
    assert isinstance(notices, list) and len(notices) == 1
    assert notices[0]["filer_name"] == "Bob"
    assert notices[0]["accession_no"] == "a02"
    assert "no change inferred" in notices[0]["note"]


def test_tool_wrapper_preserves_shape_with_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import tools

    events = [
        SimpleNamespace(
            filer_name="Alice",
            filer_cik="1",
            issuer="ACME",
            previous_accession="a00",
            current_accession="a01",
            filed_at="2024-02-15",
            to_dict=lambda: {"previous_accession": "a00", "current_accession": "a01", "share_change": 500},
        )
    ]
    covered: dict[str, object] = {
        "events": events,
        "scanned_accessions": ["a00", "a01"],
        "scanned_count": 2,
        "incomplete_history": False,
        "missing_baselines": [],
    }

    def _fake_covered(*args: object, **kwargs: object) -> dict[str, object]:
        return covered

    monkeypatch.setattr(tools.sec, "get_ownership_changes_with_coverage", _fake_covered)
    out = tools._get_ownership_changes({"ticker": "ACME", "limit": 20}, "test")
    assert out["subject"] == "ACME"
    assert out["count"] == 1
    assert out["changes"] == [{"previous_accession": "a00", "current_accession": "a01", "share_change": 500}]
    assert out["source"] == "SEC EDGAR"
    assert out["scanned_accessions"] == ["a00", "a01"]
    assert out["scanned_count"] == 2
    assert out["incomplete_history"] is False
    assert out["missing_baselines"] == []


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


def test_default_limit_scans_the_bounded_newest_window(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(25)
    seen: dict[str, object] = {}

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        seen.update(kwargs)
        limit = kwargs.get("limit")
        assert isinstance(limit, int)
        return filings[:limit]

    def fake_load(accession_no: str) -> SimpleNamespace:
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    covered = ownership.get_ownership_changes_with_coverage("ACME")
    assert seen["limit"] == 21  # max(2, default limit 20 + 1) newest filings
    assert covered["scanned_count"] == 21
    assert covered["incomplete_history"] is True
    events = ownership.get_ownership_changes("ACME")
    assert seen["limit"] == 21
    assert len(events) == 20  # bounded to the default limit


def test_readers_forward_subject_as_of_forms_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    filings = _window_filings(3)

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        calls.append((args, dict(kwargs)))
        return filings

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)

    def fake_load(accession_no: object) -> SimpleNamespace:
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    ownership.get_ownership_changes_with_coverage("ACME", as_of="2024-06-30", limit=4)
    ownership.get_ownership_changes("ACME", as_of="2024-06-30", limit=4)
    ownership.get_beneficial_ownership("ACME", as_of="2024-06-30", limit=2)
    bounded = ownership.get_beneficial_ownership("ACME", limit=2)
    records = ownership.get_beneficial_ownership("ACME")

    assert len(bounded) == 2  # the reader stays bounded by its own limit
    assert [args for args, _ in calls] == [("ACME",)] * 5
    assert [kwargs["as_of"] for _, kwargs in calls] == ["2024-06-30", "2024-06-30", "2024-06-30", None, None]
    assert [kwargs["limit"] for _, kwargs in calls] == [5, 5, 2, 2, 20]
    assert [kwargs["forms"] for _, kwargs in calls] == [["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"]] * 5
    assert len(records) == 3  # default limit 20 keeps every scanned record


def test_single_filing_limit_still_scans_a_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(3)
    limits: list[object] = []

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        limit = kwargs.get("limit")
        limits.append(limit)
        return filings[:limit] if isinstance(limit, int) else filings

    def fake_load(accession_no: str) -> SimpleNamespace:
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    covered_one = ownership.get_ownership_changes_with_coverage("ACME", limit=1)
    covered_small = ownership.get_ownership_changes_with_coverage("ACME", limit=20)
    covered_none = ownership.get_ownership_changes_with_coverage("ACME", limit=None)
    # max(2, limit + 1) bounds integer limits; explicit None stays exhaustive.
    assert limits == [2, 21, None]
    assert covered_one["scanned_accessions"] == ["a00", "a01"]
    assert covered_one["incomplete_history"] is True
    assert covered_small["incomplete_history"] is False  # 3 scanned filers stay below the 21-filing window
    assert covered_none["scanned_count"] == 3
    assert covered_none["incomplete_history"] is False


def test_event_list_stays_bounded_when_owners_outnumber_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = _window_filings(2)

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    def fake_load(accession_no: str) -> SimpleNamespace:
        # Joint filers: two owners per filing, so the pairs outnumber the requested limit.
        shares = {"a00": 1000, "a01": 1500}[accession_no]
        return _schedule([_Person("1", "Alice", agg=shares), _Person("2", "Bob", agg=shares + 1)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    covered = ownership.get_ownership_changes_with_coverage("ACME", limit=1)
    events = covered["events"]
    assert isinstance(events, list) and len(events) == 1  # two owners make two pairs; the limit keeps one
    assert covered["missing_baselines"] == []
    assert covered["scanned_accessions"] == ["a00", "a01"]


def test_scanned_accessions_keep_text_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(form="SC 13D", filed_at="2024-01-15"),
        SimpleNamespace(accession_no="a01", form="SC 13G", filed_at="2024-02-15"),
    ]

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    def fake_load(accession_no: object) -> SimpleNamespace:
        return _schedule([_Person("1", "Alice", agg=1000)], purpose="p")

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    covered = ownership.get_ownership_changes_with_coverage("ACME", limit=None)
    assert covered["scanned_accessions"] == ["", "a01"]  # a filing without an accession reads as empty text
    assert covered["scanned_count"] == 2


def test_notice_fields_and_events_survive_one_bad_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(accession_no="a00", form="SC 13D", filed_at="2024-01-15"),
        SimpleNamespace(accession_no="a01", form="SC 13D/A", filed_at="2024-02-15"),
        SimpleNamespace(accession_no="a02", form="SC 13D/A", filed_at="2024-03-15"),
        SimpleNamespace(accession_no="a03", form="SC 13G", filed_at="2024-04-15"),
    ]
    shares = {"a00": 900, "a01": 1000, "a02": 1500, "a03": 500}

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    def fake_load(accession_no: str) -> SimpleNamespace:
        # The single-record owner reads first, so a break after its notice would drop the other owner.
        owner = "2" if accession_no == "a00" else "1"
        return _schedule([_Person(owner, f"Owner {owner}", agg=shares[accession_no])], purpose="p")

    real_diff = ownership.diff_ownership

    def flaky_diff(prev: BeneficialOwnership, curr: BeneficialOwnership) -> object:
        if curr.accession_no == "a02":
            raise RuntimeError("boom")
        return real_diff(prev, curr)

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    monkeypatch.setattr(ownership, "diff_ownership", flaky_diff)
    covered = ownership.get_ownership_changes_with_coverage("ACME", limit=None)

    events = covered["events"]
    assert isinstance(events, list) and len(events) == 1  # (a01, a02) failed; (a02, a03) still diffed
    event = events[0]
    assert (event.previous_accession, event.current_accession) == ("a02", "a03")
    assert (event.filer_name, event.filer_cik, event.issuer) == ("Owner 1", "1", "ACME")
    assert (event.filed_at, event.prev_shares, event.curr_shares, event.share_change) == (
        "2024-04-15",
        1500,
        500,
        -1000,
    )
    assert (event.prev_percent, event.curr_percent, event.percent_change) == (0.0, 0.0, 0.0)
    assert event.voting_changed is False
    assert event.text_changed is False

    notices = covered["missing_baselines"]
    assert notices == [
        {
            "filer_name": "Owner 2",
            "filer_cik": "2",
            "accession_no": "a00",
            "filed_at": "2024-01-15",
            "note": "single filing in scanned window; no prior baseline, no change inferred",
        }
    ]


def test_filing_metadata_reaches_records_and_sparse_filings_keep_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = SimpleNamespace(
        accession_no="a-full",
        form="SC 13D/A",
        filed_at="2024-05-01",
        known_at="2024-05-02T00:00:00Z",
        filer_name="Owner Co",
        primary_document="primary.htm",
        source="https://sec.example/full",
    )
    sparse = SimpleNamespace(form="SC 13D", filed_at="2024-02-15", filer_name="Fallback Co")
    undated = SimpleNamespace(accession_no="a-undated")
    loaded: list[object] = []

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return [full, sparse, undated]

    def fake_load(accession_no: object) -> SimpleNamespace:
        loaded.append(accession_no)
        subject = SimpleNamespace(cik="9", name="Subject Co") if accession_no == "a-full" else None
        return _schedule([_Person("1", "Alice", agg=1000, pct=5.0)], purpose="control", issuer_info=subject)

    monkeypatch.setattr(ownership, "list_sec_filings", fake_list)
    monkeypatch.setattr(ownership, "load_schedule", fake_load)
    records = ownership.get_beneficial_ownership("ACME")

    assert loaded == ["a-full", "", "a-undated"]
    assert len(records) == 3  # a filing without a form still normalizes
    assert [record.accession_no for record in records] == ["a-full", "", "a-undated"]
    assert [record.form for record in records] == ["SC 13D/A", "SC 13D", ""]
    assert [record.is_amendment for record in records] == [True, False, False]
    assert [record.filed_at for record in records] == ["2024-05-01", "2024-02-15", None]
    assert [record.known_at for record in records] == ["2024-05-02T00:00:00Z", "2024-02-15", None]
    assert [record.document_name for record in records] == ["primary.htm", None, None]
    assert [record.source_url for record in records] == ["https://sec.example/full", None, None]
    # Structured issuer info names the subject; the filer name only fills the issuer fallback.
    assert [record.issuer for record in records] == ["Subject Co", "Fallback Co", "ACME"]
    assert [record.subject_name for record in records] == ["Subject Co", None, None]
    assert [(record.filer_name, record.filer_cik) for record in records] == [("Alice", "1")] * 3


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
