"""Offline tests for app/sec/insider.py (no network)."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, override

import pytest

from app.sec import insider
from app.sec.models import InsiderTransaction, ProposedInsiderSale


class _Activity:
    def __init__(
        self,
        code: str | None = None,
        shares: object = None,
        price: object = None,
        date: str | None = None,
        title: str | None = None,
        ad: str | None = None,
        holdings: object = None,
        transaction_type: str | None = None,
    ) -> None:
        self.transaction_code = code
        self.shares = shares
        self.price = price
        self.transaction_date = date
        self.security_title = title
        self.acquired_disposed = ad
        self.holdings_after = holdings
        self.transaction_type = transaction_type


class _Obj:
    def __init__(self, rows: list[_Activity], name: str = "Jane Doe", cik: str = "999"):
        self.insider_name = name
        self.insider_cik = cik
        self._rows = rows

    def get_transaction_activities(self):
        return self._rows


def test_code_kinds_and_missing_fields():
    rows = [
        _Activity(code="P", shares="1,000", price="10.5", date="2024-01-10", title="Common", ad="A", holdings="5000"),
        _Activity(code="S", shares=200, price=11.0, date="2024-02-10", title="Common", ad="D", holdings=4800),
        _Activity(code="M", shares=50, price="0", date="2024-03-10", title="Option", ad="A", holdings=4850),
        _Activity(code="Z", shares=7, price=1.0, date="2024-04-10", title="Common", ad="A", holdings=4857),
        _Activity(),  # everything missing
    ]
    txns = insider.normalize_ownership_filing(
        _Obj(rows), issuer="ACME", form="4", filed_at="2024-05-01", accession_no="x1"
    )
    assert [t.transaction_kind for t in txns] == [
        "open_market_purchase",
        "open_market_sale",
        "exercise",
        "other",
        "other",
    ]
    # never default unknown disposals to bearish selling
    assert txns[3].transaction_kind == "other"
    assert txns[3].transaction_kind != "open_market_sale"
    blank = txns[4]
    assert blank.shares is None and blank.price is None
    assert blank.transaction_date is None and blank.security is None
    assert blank.transaction_code is None and blank.holdings_after is None
    assert txns[0].shares == 1000 and txns[0].price == 10.5


class _Row:
    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping

    def to_dict(self) -> dict[str, object]:
        return dict(self._mapping)


class _Frame:
    def __init__(self, rows: Sequence[object]) -> None:
        self._rows = [_Row(r) if isinstance(r, dict) else r for r in rows]

    def iterrows(self) -> list[object]:
        return list(enumerate(self._rows))


class _Table:
    def __init__(self, rows: Sequence[object]) -> None:
        self.transactions = SimpleNamespace(data=_Frame(rows))


class _EnrichedObj(_Obj):
    def __init__(
        self,
        rows: list[_Activity],
        table_rows: Sequence[object],
        name: str = "Jane Doe",
        cik: str = "999",
        reporting_period: str = "2024-01-31",
    ) -> None:
        super().__init__(rows, name, cik)
        self.non_derivative_table = _Table(table_rows)
        self.derivative_table = _Table([])
        self.reporting_period = reporting_period


def _enriched_txns() -> list[InsiderTransaction]:
    rows = [
        _Activity(code="S", shares=2605.75, price=360.134),
        _Activity(code="M", shares=6539.0, price=0.0),
    ]
    table = [
        {"Code": "M", "Shares": 6539.0, "Price": 0.0, "Date": "2024-03-05", "AcquiredDisposed": "A", "Remaining": 100},
        {
            "Code": "S",
            "Shares": 2605.75,
            "Price": 360.134,
            "Date": "2024-03-08",
            "AcquiredDisposed": "D",
            "Remaining": 50,
        },
    ]
    return insider.normalize_ownership_filing(
        _EnrichedObj(rows, table), issuer="ACME", form="4", filed_at="2024-03-10", accession_no="x9"
    )


def test_enrichment_keyed_not_positional() -> None:
    txns = _enriched_txns()
    by_code = {t.transaction_code: t for t in txns}
    assert by_code["S"].transaction_date == "2024-03-08"
    assert by_code["M"].transaction_date == "2024-03-05"
    assert by_code["S"].acquired_disposed == "D" and by_code["M"].acquired_disposed == "A"
    assert by_code["S"].shares == 2605 and by_code["S"].holdings_after == 50


def test_enrichment_permutation_stable() -> None:
    rows = [
        _Activity(code="S", shares=2605.75, price=360.134),
        _Activity(code="M", shares=6539.0, price=0.0),
    ]
    table = [
        {"Code": "M", "Shares": 6539.0, "Price": 0.0, "Date": "2024-03-05", "AcquiredDisposed": "A", "Remaining": 100},
        {
            "Code": "S",
            "Shares": 2605.75,
            "Price": 360.134,
            "Date": "2024-03-08",
            "AcquiredDisposed": "D",
            "Remaining": 50,
        },
    ]
    forward = insider.normalize_ownership_filing(
        _EnrichedObj(rows, table), issuer="ACME", form="4", filed_at="2024-03-10", accession_no="x9"
    )
    backward = insider.normalize_ownership_filing(
        _EnrichedObj(rows, list(reversed(table))), issuer="ACME", form="4", filed_at="2024-03-10", accession_no="x9"
    )
    assert {t.transaction_code: t.transaction_date for t in forward} == {
        t.transaction_code: t.transaction_date for t in backward
    }


def test_enrichment_duplicate_and_miss_fall_back() -> None:
    dup = [
        {"Code": "S", "Shares": 10, "Price": 1.0, "Date": "2024-03-01", "AcquiredDisposed": "D"},
        {"Code": "S", "Shares": 10, "Price": 1.0, "Date": "2024-03-02", "AcquiredDisposed": "D"},
    ]
    txns = insider.normalize_ownership_filing(
        _EnrichedObj([_Activity(code="S", shares=10, price=1.0)], dup, reporting_period="2024-02-29"),
        issuer="ACME",
        form="4",
        filed_at="2024-03-10",
        accession_no="xd",
    )
    assert txns[0].transaction_date == "2024-03-01"
    missed = insider.normalize_ownership_filing(
        _EnrichedObj(
            [_Activity(code="S", shares=10, price=1.0, transaction_type="sale")], [], reporting_period="2024-02-29"
        ),
        issuer="ACME",
        form="4",
        filed_at="2024-03-10",
        accession_no="xm",
    )
    assert missed[0].transaction_date == "2024-02-29"
    assert missed[0].acquired_disposed == "D"


class _FakeDF:
    columns: ClassVar[object] = ["Shares to be sold"]

    def __getitem__(self, key: str) -> list[str | None]:
        assert key == "Shares to be sold"
        return ["1,000", "500", None]


def test_normalize_144_sums_shares_column():
    form144 = SimpleNamespace(person_selling="John Smith", seller_cik="123", securities_to_be_sold=_FakeDF())
    sale = insider.normalize_144(form144, issuer="ACME", filed_at="2024-05-01", accession_no="t1")
    assert sale.seller_name == "John Smith"
    assert sale.shares_proposed == 1500


def test_normalize_144_never_raises():
    sale = insider.normalize_144(object(), issuer="ACME", filed_at=None, accession_no="t9")
    assert sale.shares_proposed is None


def test_compare_144_to_form4_date_filter():
    proposed = ProposedInsiderSale("John Smith", "123", "ACME", "2024-05-01", "t1", shares_proposed=1500)

    def txn(name: str, date: str, shares: int, kind: str = "open_market_sale") -> InsiderTransaction:
        return InsiderTransaction(
            name, "123", "ACME", "4", "2024-06-01", "f1", date, "Common", "S", kind, shares, 10.0, "D", 1000
        )

    txns = [
        txn("john smith", "2024-06-01", 400),  # later sale, case-insensitive
        txn("John Smith", "2024-04-01", 9999),  # earlier sale: ignored
        txn("John Smith", "2024-06-02", 100, kind="open_market_purchase"),
        txn("Someone Else", "2024-06-03", 500),
    ]
    result = insider.compare_144_to_form4(proposed, txns)
    assert result["executed_sale_shares"] == 400
    assert result["matched"] is True
    assert result["proposed_shares"] == 1500
    assert result["seller_name"] == "John Smith"


def test_compare_144_unmatched():
    proposed = ProposedInsiderSale("Jane Doe", None, "ACME", "2024-05-01", "t2", shares_proposed=100)
    result = insider.compare_144_to_form4(proposed, [])
    assert result["matched"] is False
    assert result["executed_sale_shares"] == 0


def test_get_insider_activity_propagates_failed_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(accession_no="g1", form="4", filed_at="2024-01-15", company="ACME"),
        SimpleNamespace(accession_no="bad", form="4", filed_at="2024-02-15", company="ACME"),
    ]
    error = RuntimeError("boom")

    def fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return filings

    monkeypatch.setattr(insider, "list_sec_filings", fake_list)

    def fake_load(accession_no: str) -> _Obj:
        if accession_no == "bad":
            raise error
        return _Obj([_Activity(code="P", shares=10)])

    monkeypatch.setattr(insider, "load_ownership", fake_load)
    with pytest.raises(RuntimeError, match="^boom$") as exc:
        insider.get_insider_activity("ACME")
    assert exc.value is error

    filings.pop()
    txns = insider.get_insider_activity("ACME")
    assert len(txns) == 1
    assert txns[0].transaction_kind == "open_market_purchase"
    assert txns[0].shares == 10
    filings.clear()
    assert insider.get_insider_activity("ACME") == []


def test_store_no_persist_and_live_insider_per_accession(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import Filing
    from app.sec.store import query_insider_transactions

    # Unseeded live queries stay empty; the per-accession live path below pins the contract.
    assert query_insider_transactions(issuer_cik=320193, root=tmp_path) == []
    assert query_insider_transactions(root=tmp_path) == []
    filing = Filing(
        form="4",
        accession_no="ACC-4",
        filed_at="2024-04-01",
        filer_cik=320193,
        filer_name="Issuer Inc",
        accepted_at=None,
        known_at="2024-04-01T00:00:00Z",
        report_period=None,
        primary_document="primary",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )

    import app.sec.store as sec_store_mod

    def _fake_meta(accession: str, as_of: str | None = None) -> Filing:
        if accession == "ACC-4":
            return filing
        raise ValueError(f"unknown accession {accession!r}")

    monkeypatch.setattr(sec_store_mod, "_gateway_get_filing", _fake_meta)

    def _fake_load(accession_no: str) -> _Obj:
        if accession_no == "ACC-4":
            return _Obj([_Activity(code="P", shares=100)], name="Jane Doe", cik="1206472")
        raise RuntimeError(f"unknown accession {accession_no!r}")

    monkeypatch.setattr(insider, "load_ownership", _fake_load)
    rows = query_insider_transactions(accession="ACC-4", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["insider_name"] == "Jane Doe"
    assert rows[0]["insider_cik"] == "1206472"
    assert rows[0]["issuer"] == "Issuer Inc"
    assert rows[0]["transaction_kind"] == "open_market_purchase"
    # Roles stay on their own side: issuer is never the owner.
    assert rows[0]["issuer"] != rows[0]["insider_name"]


def test_13f_live_per_accession_and_no_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import insider as _ins
    from app.sec.models import Filing
    from app.sec.store import query_13f_holdings

    rec = _ins._holding_row_to_record(
        {"Cusip": "037833100", "Issuer": "Apple Inc.", "ReportPeriod": "2024-03-31", "Class": "Common Stock"},
        manager_name="Berkshire",
        manager_cik="1067983",
        accession_no="ACC-13F-1",
        report_period="2024-03-31",
        filed_at="2024-05-15",
        document_name="infotable.xml",
        known_at="2024-05-15T00:00:00Z",
        source_url=None,
        source_row=1,
    )
    assert rec.security_id == "cusip:037833100" and rec.entity_id is None
    # Unseeded live queries stay empty; the per-accession live path below pins the contract.
    assert query_13f_holdings(manager_cik="1067983", root=tmp_path) == []
    filing = Filing(
        form="13F-HR",
        accession_no="ACC-13F-1",
        filed_at="2024-05-15",
        filer_cik=1067983,
        filer_name="Berkshire",
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
        if accession == "ACC-13F-1":
            return filing
        raise ValueError(f"unknown accession {accession!r}")

    monkeypatch.setattr(sec_store_mod, "_gateway_get_filing", _fake_meta)
    table = [{"Cusip": "037833100", "Issuer": "Apple Inc.", "ReportPeriod": "2024-03-31", "Class": "Common Stock"}]

    def _fake_edgar(accession: str) -> SimpleNamespace:
        assert accession == "ACC-13F-1"
        return SimpleNamespace(obj=lambda: SimpleNamespace(infotable=table))

    monkeypatch.setattr(sec_docs, "get_by_accession_number", _fake_edgar)
    rows = query_13f_holdings(accession="ACC-13F-1", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["cusip"] == "037833100"
    assert rows[0]["security_id"] == "cusip:037833100"
    assert rows[0]["manager_cik"] == "1067983"


def test_values_by_column_getitem_fast_path() -> None:
    class _Frame:
        columns: ClassVar[object] = ["shares"]

        def __getitem__(self, key: object) -> object:
            assert key == "shares"
            return [100, 200]

    assert insider._values_by_column(_Frame(), "shares") == [100, 200]


def test_values_by_column_row_scan_fallbacks() -> None:
    dict_rows: list[object] = [{"shares": "1,000"}, {"shares": None}, {"price": 5}]
    assert insider._values_by_column(dict_rows, "shares") == ["1,000"]
    tuple_rows: list[object] = [("a", "b"), ("c",)]
    assert insider._values_by_column(tuple_rows, 1) == ["b"]
    assert insider._values_by_column([{"shares": 1}], True) is None
    assert insider._values_by_column([{"shares": 1}], None) is None
    assert insider._values_by_column(42, "shares") is None


def test_values_by_column_propagates_row_errors() -> None:
    error = RuntimeError("boom")

    class _BadDict(dict[str, object]):
        @override
        def get(self, key: str, default: object = None) -> object:
            raise error

    with pytest.raises(RuntimeError, match="^boom$") as exc:
        insider._values_by_column([_BadDict({"shares": 1})], "shares")
    assert exc.value is error
    assert insider._values_by_column([object()], "shares") == []
    assert insider._values_by_column([], "shares") == []


def test_values_by_column_propagates_lookup_and_coercion_failures() -> None:
    error = RuntimeError("boom")

    class _BadIter:
        columns: ClassVar[object] = ["shares"]

        def __getitem__(self, key: object) -> object:
            raise error

        def __iter__(self) -> object:
            raise error

    with pytest.raises(RuntimeError, match="^boom$") as exc:
        insider._values_by_column(_BadIter(), "shares")
    assert exc.value is error

    class _BadRows(_BadIter):
        @override
        def __getitem__(self, key: object) -> object:
            return None

    with pytest.raises(RuntimeError, match="^boom$") as exc:
        insider._values_by_column(_BadRows(), "shares")
    assert exc.value is error
