"""Unit tests for the generic obligations engine (Layer 1 XBRL + Layer 2
note-text + Layer 3 balance sheet). Live-only seams: filing text is stubbed
through edgar_client, XBRL facts are fixture FinancialFactRows stubbed at
_xbrl_gateway_facts, and note text archives on acceptance via raw_archive.

Warehouse-removal seam: live providers (SourceGateway + normalization + raw_archive) serve reads, nothing is persisted; a future warehouse slots in behind the gateway.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app import obligations
from app.storage import raw_archive

FIXTURES = Path(__file__).parent / "fixtures" / "obligations"


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str, ttl: float | None = None) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


class FakeNotes:
    def __init__(self, by_title: dict[str, FakeNote]) -> None:
        self._by: dict[str, FakeNote] = by_title

    def to_markdown(self) -> str:
        return "\n\n".join(n.to_markdown() for n in self._by.values())

    def search(self, keyword: str) -> list[FakeNote]:
        matches: list[FakeNote] = []
        for title, note in self._by.items():
            if keyword.lower() in title.lower():
                matches.append(note)
        return matches


class FakeNote:
    def __init__(self, title: str, markdown: str) -> None:
        self.title = title
        self._md = markdown

    def to_markdown(self):
        return self._md


class FakeDoc:
    def __init__(self, notes: FakeNotes, balance_sheet_md: str = "") -> None:
        self.notes = notes
        self._bs = balance_sheet_md

    @property
    def financials(self):
        bs_md = self._bs

        class _FS:
            def balance_sheet(self):
                m = MagicMock()
                m.to_markdown = lambda: bs_md
                return m

        return _FS()


class FakeFiling:
    def __init__(self, date: str = "2026-02-25") -> None:
        self.filing_date: str = date
        self.accession_no: str = "0001"
        self._doc: FakeDoc | None = None

    def obj(self) -> object:
        return self._doc


def _note_md(ticker: str, title_fragment: str) -> str:
    data: dict[str, object] = json.loads((FIXTURES / f"{ticker}_10K_notes.json").read_text())
    notes = data["notes"]
    assert isinstance(notes, dict)
    for title, md in notes.items():
        assert isinstance(md, str)
        if title_fragment.lower() in title.lower():
            return md
    raise KeyError(f"{title_fragment} not in {ticker} fixture: {list(notes)}")


def test_sentence_amounts_nvda_supply_cloud():
    md = _note_md("NVDA", "Commitments")
    rows = obligations._parse_sentence_amounts(md)
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["amount_billions"])
    # 10-K fixture discloses $95.2B (10-Q updates it to $119B live).
    assert 95.2 in by_kind.get("supply", [])
    # Cloud total (27.0) plus its per-year schedule rows (7.0, 6.0, ...).
    assert 27.0 in by_kind.get("cloud", [])
    assert 11.4 in by_kind.get("investment", [])


def test_sentence_amounts_program_capacity_excluded():
    md = _note_md("NVDA", "Debt")
    rows = obligations._parse_sentence_amounts(md)
    # Commercial paper program capacity ($25B) is not an obligation.
    assert not any(r["amount_billions"] == 25.0 for r in rows)


def test_fiscal_year_table_parsing():
    md = _note_md("NVDA", "Leases")
    table = obligations._parse_fiscal_year_table(md)
    amounts = {r["fiscal_year"]: r["amount_millions"] for r in table}
    assert amounts.get("2027") in (460, 493)
    assert "Thereafter" in amounts or "2032 and thereafter" in amounts


def test_tax_note_does_not_flood_generic_obligations():
    md = _note_md("NVDA", "Income Taxes")
    rows = obligations._parse_sentence_amounts(md)
    assert all(r["kind"] == "other" for r in rows)


def test_classify_contractual_vs_contingent():
    assert obligations._classify("non-cancelable lease agreements") == "contractual"
    assert obligations._classify("commitments are cancellable, rescheduled") == "contingent"
    assert obligations._classify("capacity may be reduced or terminated") == "contingent"
    assert obligations._classify("plain commitment") == "contractual"


def test_amount_kind_priority():
    assert obligations._amount_kind("cloud service agreement commitments were $30 billion") == "cloud"
    assert obligations._amount_kind("investment commitments are $11.4 billion") == "investment"
    assert obligations._amount_kind("manufacturing, supply, and capacity commitments were $119 billion") == "supply"


def _fact(
    concept: str = "PurchaseObligations",
    value: float = 5e9,
    period_end: str = "2026-05-02",
    filed_at: str = "2026-08-26",
    known_at: str = "2026-08-27T00:00:00Z",
    accession: str = "000123-26-000001",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "concept": concept,
        "value": value,
        "period_start": "2026-02-01",
        "period_end": period_end,
        "fiscal_year": 2026,
        "fiscal_period": "Q1",
        "filed_at": filed_at,
        "accession": accession,
        "known_at": known_at,
        "source_url": "",
    }
    row.update(overrides)
    return row


def _install(
    monkeypatch: pytest.MonkeyPatch,
    notes_md: dict[str, str],
    bs_md: str = "",
    facts: list[dict[str, object]] | None = None,
) -> FakeFiling:
    notes = FakeNotes({t: FakeNote(t, md) for t, md in notes_md.items()})
    doc = FakeDoc(notes, bs_md)
    filing = FakeFiling()

    class FakeCompany:
        def __init__(self, ticker: str) -> None:
            pass

        def get_filings(self, form: list[str] | None = None) -> list[FakeFiling]:
            return [filing]

    # Filing text comes from the edgar seam; XBRL facts are fixture
    # FinancialFactRows at the gateway seam (empty by default, populated
    # per test); archiving stays live so acceptance still writes raw/.
    fixture_facts = list(facts) if facts is not None else []

    def _fake_gateway_facts(ticker: str) -> list[dict[str, object]]:
        return list(fixture_facts)

    monkeypatch.setattr(obligations.edgar_client, "get_company", FakeCompany)
    monkeypatch.setattr(obligations, "_xbrl_gateway_facts", _fake_gateway_facts)
    monkeypatch.setattr(obligations, "cache", FakeCache())
    filing._doc = doc
    return filing


def test_get_obligations_nvda_full(monkeypatch: pytest.MonkeyPatch):
    data: dict[str, object] = json.loads((FIXTURES / "NVDA_10K_notes.json").read_text())
    notes_md = data["notes"]
    assert isinstance(notes_md, dict)
    _install(monkeypatch, notes_md)
    result = obligations.get_obligations("NVDA")
    assert "error" not in result
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    types = {o["type"] for o in _obligations}
    assert "debt" in types
    assert "operating_leases" in types
    assert "purchase_commitments" in types
    # Every row carries provenance + status.
    for row in _obligations:
        assert row.get("status")
        assert row.get("filed")
        assert row.get("source")
        assert row.get("content_hash")
        assert row.get("parser_version") == obligations.PARSER_VERSION


def test_get_obligations_requires_quantified_data(monkeypatch: pytest.MonkeyPatch):
    _install(monkeypatch, {"Notes": "no dollar figures here"})
    result = obligations.get_obligations("NVDA")
    assert "error" in result or not result.get("obligations")


def test_get_obligations_empty_ticker():
    result = obligations.get_obligations("")
    assert "error" in result


def test_archive_on_acceptance_anchors_note_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Note text that became evidence archives write-once under raw/."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    md = "The company's non-cancelable supply commitments were $13.3 billion as of September 27, 2025."
    _install_layered(
        monkeypatch,
        {"10-K": ({"Commitments and Contingencies": md}, "2026-02-01")},
    )
    result = obligations.get_obligations("SYN")
    assert "error" not in result
    _rows = result["obligations"]
    assert isinstance(_rows, list)
    assert any(r["type"] == "supply" and r["amount_billions"] == 13.3 for r in _rows)
    record = raw_archive.find("sec", "filing-note-text", "filing-text:SYN:2026-02-01:acc-10-K", root=tmp_path / "raw")
    assert record is not None
    assert record.payload_path.read_bytes() == md.encode()


def test_8k_guarantee_parsing():
    text = (
        "NVIDIA's aggregate payment obligation is cumulatively capped at "
        "$105 billion for its initial commitment under the Agreements."
    )
    matches = list(obligations._8K_GUARANTEE_RE.finditer(text))
    assert matches
    assert obligations._billion(matches[0].group(1), matches[0].group(2)) == 105.0


def test_balance_sheet_lines(monkeypatch: pytest.MonkeyPatch):
    bs_md = "| Accounts payable | $13,097 |\n| Total liabilities | $64,000 |\n"
    data: dict[str, object] = json.loads((FIXTURES / "NVDA_10K_notes.json").read_text())
    notes_md = data["notes"]
    assert isinstance(notes_md, dict)
    _install(monkeypatch, notes_md, bs_md)
    rows = obligations._balance_sheet_liabilities("NVDA")
    assert len(rows) == 2
    by_type = {r["type"]: r["amount_billions"] for r in rows}
    assert by_type["bs_accounts_payable"] == pytest.approx(13.097)
    assert by_type["bs_total_liabilities"] == pytest.approx(64.0)
    assert all(r["status"] == "on_balance_sheet" for r in rows)


def _obligation_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "purchase_commitments",
        "amount_billions": 119.0,
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": True,
        "default_triggered": False,
        "fiscal_year": None,
        "excerpt": "NVIDIA's non-cancelable purchase obligations are $119.0 billion.",
        "source": "SEC EDGAR 2026-02-25 Commitments note",
        "filed": "2026-02-25",
        "as_of": "2026-02-25",
        "known_at": "2026-08-26T00:00:00Z",  # extraction time (never event known_at)
        "retrieved_at": "2026-08-26T00:00:00Z",
        "content_hash": "abc",
        "parser_version": obligations.PARSER_VERSION,
        "ticker": "NVDA",
    }
    row.update(overrides)
    return row


def test_table_schedule_keeps_thereafter_bucket():
    md = _note_md("AAPL", "Commitments")
    schedule = obligations._parse_table_schedule(md)
    assert schedule is not None
    by_year = {y["fiscal_year"]: y["amount_billions"] for y in schedule}
    assert by_year["2026"] == pytest.approx(4.752)
    assert by_year["Thereafter"] == pytest.approx(0.773)
    assert sum(by_year.values()) == pytest.approx(13.308, abs=0.01)


def test_prose_schedule_reconciles_to_own_amount():
    md = (
        "Supply commitments were $8.0 billion, for which $4.0 billion, "
        "$3.0 billion and $1.0 billion will be paid in fiscal years 2027, "
        "2028 and 2029, respectively. Cloud commitments were $27.0 billion, "
        "for which $14.0 billion and $13.0 billion will be paid in fiscal "
        "years 2027 and 2028, respectively."
    )
    sched = obligations._parse_prose_schedule(md, 8.0)
    assert sched is not None
    assert [(y["fiscal_year"], y["amount_billions"]) for y in sched] == [
        ("2027", 4.0),
        ("2028", 3.0),
        ("2029", 1.0),
    ]
    other = obligations._parse_prose_schedule(md, 27.0)
    assert other is not None
    assert [(y["fiscal_year"], y["amount_billions"]) for y in other] == [
        ("2027", 14.0),
        ("2028", 13.0),
    ]
    assert obligations._parse_prose_schedule(md, 95.2) is None


def test_prose_schedule_thereafter_tail():
    md = (
        "Investment commitments are $8.0 billion, for which $4.0 billion, "
        "$3.0 billion and $1.0 billion will be paid in fiscal years 2027 "
        "and 2028 and thereafter, respectively."
    )
    assert obligations._parse_prose_schedule(md, 8.0) == [
        {"fiscal_year": "2027", "amount_billions": 4.0},
        {"fiscal_year": "2028", "amount_billions": 3.0},
        {"fiscal_year": "Thereafter", "amount_billions": 1.0},
    ]


def test_collect_note_rows_supply_schedule_no_bleed():
    md = (
        "Supply commitments were $13.3 billion as of September 27, 2025. "
        "Future payments are as follows (in millions):\n"
        "| 2026 | $4,752 |\n| 2027 | $3,708 |\n| 2028 | $1,981 |\n"
        "| 2029 | $1,306 |\n| 2030 | $788 |\n| Thereafter | $773 |"
    )
    rows: list[dict[str, object]] = []
    obligations._collect_note_rows(rows, "Commitments and Contingencies", md, FakeFiling())
    (headline,) = [r for r in rows if r["type"] == "supply" and r["amount_billions"] == 13.3]
    _schedule = headline["schedule"]
    assert isinstance(_schedule, list)
    assert [y["fiscal_year"] for y in _schedule] == [
        "2026",
        "2027",
        "2028",
        "2029",
        "2030",
        "Thereafter",
    ]
    assert headline["payment_horizon"] is None


def test_collect_note_rows_unreconciled_supply_keeps_horizon():
    md = _note_md("NVDA", "Commitments")
    rows: list[dict[str, object]] = []
    obligations._collect_note_rows(rows, "Commitments and Contingencies", md, FakeFiling())
    supply = [r for r in rows if r["type"] == "supply"]
    assert supply and all(r["schedule"] is None for r in supply)
    horizons: list[object] = [r["payment_horizon"] for r in supply]
    assert any(isinstance(h, dict) and h.get("paid_in_remainder_of_fy") == "2027" for h in horizons)


def _install_layered(
    monkeypatch: pytest.MonkeyPatch,
    notes_by_form: dict[str, tuple[dict[str, str], str]],
    facts: list[dict[str, object]] | None = None,
) -> None:
    """Form-aware mocked EDGAR: {form: ({title: md}, filing_date)}; 8-K -> none."""

    def fake_get_company(ticker: str) -> object:
        class _C:
            def get_filings(self, form: list[str] | None = None) -> list[FakeFiling]:
                out: list[FakeFiling] = []
                for name in form or []:
                    if name in notes_by_form:
                        notes_md, date = notes_by_form[name]
                        notes = FakeNotes({t: FakeNote(t, md) for t, md in notes_md.items()})
                        doc = FakeDoc(notes, "")
                        filing = FakeFiling(date=date)
                        filing.accession_no = f"acc-{name}"
                        filing._doc = doc
                        out.append(filing)
                return out

        return _C()

    fixture_facts = list(facts) if facts is not None else []

    def _fake_gateway_facts(ticker: str) -> list[dict[str, object]]:
        return list(fixture_facts)

    monkeypatch.setattr(obligations.edgar_client, "get_company", fake_get_company)
    monkeypatch.setattr(obligations, "_xbrl_gateway_facts", _fake_gateway_facts)
    monkeypatch.setattr(obligations, "cache", FakeCache())


def test_snapshot_supersedes_older_filing(monkeypatch: pytest.MonkeyPatch):
    """10-K $20B superseded by 10-Q $13B: ledger keeps both, snapshot $13B."""
    _install_layered(
        monkeypatch,
        {
            "10-Q": (
                {"Commitments and Contingencies": "Supply commitments were $13.0 billion as of March 31, 2026."},
                "2026-04-01",
            ),
            "10-K": (
                {"Commitments and Contingencies": "Supply commitments were $20.0 billion as of December 31, 2025."},
                "2026-02-01",
            ),
        },
    )
    result = obligations.get_obligations("SYN")
    assert "error" not in result
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    assert sorted(o["amount_billions"] for o in _obligations) == [13.0, 20.0]
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    assert [s["amount_billions"] for s in _snapshot] == [13.0]
    for row in _snapshot:
        assert row["parser_version"] == "obligations-v4"
        assert row["content_hash"]
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    assert _coverage["quantified_count"] == 2
    _filings = result["filings_examined"]
    assert isinstance(_filings, list)
    assert {"2026-04-01", "2026-02-01"} <= set(_filings)


def test_unquantified_only_returns_buckets_not_error(monkeypatch: pytest.MonkeyPatch):
    _install_layered(
        monkeypatch,
        {
            "10-K": (
                {"Commitments and Contingencies": "The company may indemnify its officers against certain claims."},
                "2026-02-01",
            ),
        },
    )
    result = obligations.get_obligations("SYN")
    assert "error" not in result
    assert result["obligations"] == []
    assert result["current_snapshot"] == []
    _unquantified = result["unquantified_exposures"]
    assert isinstance(_unquantified, list)
    assert len(_unquantified) == 1
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    assert _coverage["quantified_count"] == 0
    assert _coverage["unquantified_count"] == 1


def test_unquantified_triggers_from_sentence_words():
    filing = FakeFiling(date="2026-02-01")
    (unknown,) = obligations._scan_unquantified_exposures(
        "Guarantees", "The company may indemnify its officers against certain claims.", filing
    )[0]
    assert unknown["trigger"] == "unknown"
    assert unknown["excerpt"]
    (defaulted,) = obligations._scan_unquantified_exposures(
        "Guarantees", "The guarantees pay only upon counterparty default.", filing
    )[0]
    assert defaulted["trigger"] == "counterparty_default"
    (cond,) = obligations._scan_unquantified_exposures(
        "Guarantees", "The guarantee is conditional upon regulatory approval.", filing
    )[0]
    assert cond["trigger"] == "conditional"


def test_buybacks_and_dividends_are_capital_not_exposures():
    filing = FakeFiling(date="2026-02-01")
    exps, caps = obligations._scan_unquantified_exposures(
        "Equity", "The board authorized a share repurchase program.", filing
    )
    assert exps == []
    assert [c["type"] for c in caps] == ["buybacks"]
    assert caps[0]["trigger"] == "board_discretion"
    exps, caps = obligations._scan_unquantified_exposures("Equity", "The company declared quarterly dividends.", filing)
    assert exps == []
    assert [c["type"] for c in caps] == ["dividends"]


def test_zero_finding_filing_appears_in_scan_manifest(monkeypatch: pytest.MonkeyPatch):
    _install_layered(
        monkeypatch,
        {
            "10-K": (
                {"Commitments and Contingencies": "The company may indemnify its officers against certain claims."},
                "2026-02-01",
            ),
        },
    )
    result = obligations.get_obligations("SYN")
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    _manifest = _coverage["scan_manifest"]
    assert isinstance(_manifest, list)
    scanned = [m for m in _manifest if m["status"] == "scanned"]
    assert any(m["quantified_count"] == 0 for m in scanned)
    assert any(m["form"] == "10-K" and m["filing_date"] == "2026-02-01" for m in scanned)


def test_schedule_change_changes_content_hash():
    base = {
        "type": "supply",
        "amount_billions": 13.3,
        "filed": "2026-02-01",
        "certainty": "contingent",
        "status": "future_cash_obligation",
        "revenue_matched": True,
        "default_triggered": False,
        "fiscal_year": None,
        "schedule": [
            {"fiscal_year": "2027", "amount_billions": 4.0},
            {"fiscal_year": "2028", "amount_billions": 3.0},
        ],
    }
    same = {**base, "schedule": [dict(y) for y in base["schedule"]]}
    assert obligations._content_hash(base) == obligations._content_hash(same)
    changed = {
        **base,
        "schedule": [
            {"fiscal_year": "2027", "amount_billions": 5.0},
            {"fiscal_year": "2028", "amount_billions": 3.0},
        ],
    }
    assert obligations._content_hash(base) != obligations._content_hash(changed)
    nosched = {k: v for k, v in base.items() if k != "schedule"}
    assert obligations._content_hash(base) != obligations._content_hash(nosched)


def _install_with_8k(
    monkeypatch: pytest.MonkeyPatch,
    notes_by_form: dict[str, tuple[dict[str, str], str]],
    filings_8k: list[tuple[str, list[str], str, str]],
    facts: list[dict[str, object]] | None = None,
) -> None:
    """Form-aware mocked EDGAR plus a canned 8-K stream.

    ``filings_8k``: [(filing_date, items, text, accession)].
    """

    class Fake8KObj:
        def __init__(self, items: list[str], document: str) -> None:
            self.items: list[str] = items
            self.document: str = document

    class Fake8KFiling:
        def __init__(self, date: str, items: list[str], text: str, accession: str) -> None:
            self.filing_date = date
            self.accession_no = accession
            self._obj = Fake8KObj(items, text)

        def obj(self) -> Fake8KObj:
            return self._obj

    made_8k = [Fake8KFiling(*spec) for spec in filings_8k]

    def fake_get_company(ticker: str) -> object:
        class _C:
            def get_filings(self, form: list[str] | None = None) -> list[FakeFiling] | list[Fake8KFiling]:
                if form == ["8-K"]:
                    return list(made_8k)
                out: list[FakeFiling] = []
                for name in form or []:
                    if name in notes_by_form:
                        notes_md, date = notes_by_form[name]
                        notes = FakeNotes({t: FakeNote(t, md) for t, md in notes_md.items()})
                        doc = FakeDoc(notes, "")
                        filing = FakeFiling(date=date)
                        filing.accession_no = f"acc-{name}"
                        filing._doc = doc
                        out.append(filing)
                return out

        return _C()

    fixture_facts = list(facts) if facts is not None else []

    def _fake_gateway_facts(ticker: str) -> list[dict[str, object]]:
        return list(fixture_facts)

    monkeypatch.setattr(obligations.edgar_client, "get_company", fake_get_company)
    monkeypatch.setattr(obligations, "_xbrl_gateway_facts", _fake_gateway_facts)
    monkeypatch.setattr(obligations, "cache", FakeCache())


def test_schedule_table_reconciles_to_headline(monkeypatch: pytest.MonkeyPatch):
    """$13.3B headline + $13.308B table expose ~$13.3B, never ~$26.6B."""
    md = (
        "Supply commitments were $13.3 billion as of September 27, 2025. "
        "Future payments are as follows (in millions):\n"
        "| 2026 | $4,752 |\n| 2027 | $3,708 |\n| 2028 | $1,981 |\n"
        "| 2029 | $1,306 |\n| 2030 | $788 |\n| Thereafter | $773 |"
    )
    _install_layered(
        monkeypatch,
        {
            "10-K": ({"Commitments and Contingencies": md}, "2026-02-01"),
        },
    )
    result = obligations.get_obligations("SYN")
    assert "error" not in result
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    (headline,) = [r for r in _obligations if r["type"] == "supply" and r["amount_billions"] == 13.3]
    assert headline["schedule"]
    comps = [r for r in _obligations if r.get("schedule_component")]
    assert len(comps) == 6
    assert all(c["headline_type"] == "supply" for c in comps)
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap_total = sum(r["amount_billions"] for r in _snapshot)
    assert snap_total == pytest.approx(13.3, abs=0.05)


def test_schedule_components_flag_and_legacy_backfill(monkeypatch: pytest.MonkeyPatch):
    """Headline + table reconcile once; dropping flags re-derives them."""
    md = (
        "Supply commitments were $13.3 billion as of September 27, 2025. "
        "Future payments are as follows (in millions):\n"
        "| 2026 | $4,752 |\n| 2027 | $3,708 |\n| 2028 | $1,981 |\n"
        "| 2029 | $1,306 |\n| 2030 | $788 |\n| Thereafter | $773 |"
    )
    _install_layered(
        monkeypatch,
        {
            "10-K": ({"Commitments and Contingencies": md}, "2026-02-01"),
        },
    )
    live = obligations.get_obligations("SYN")
    assert "error" not in live
    _live_obligations = live["obligations"]
    assert isinstance(_live_obligations, list)
    assert len(_live_obligations) == 7
    (live_headline,) = [r for r in _live_obligations if r["type"] == "supply" and r["amount_billions"] == 13.3]
    assert live_headline["schedule"]
    assert len([r for r in _live_obligations if r.get("schedule_component")]) == 6
    _live_snapshot = live["current_snapshot"]
    assert isinstance(_live_snapshot, list)
    assert sum(r["amount_billions"] for r in _live_snapshot) == pytest.approx(13.3, abs=0.05)
    unflagged = [dict(r) for r in _live_obligations]
    for row in unflagged:
        row.pop("schedule_component", None)
        row.pop("headline_type", None)
    obligations._apply_legacy_component_flags(unflagged)
    assert len([r for r in unflagged if r.get("schedule_component")]) == 6
    assert all(r["headline_type"] == "supply" for r in unflagged if r.get("schedule_component"))
    snapshot, _ = obligations._current_snapshot(unflagged)
    amounts = [a for r in snapshot if isinstance(a := r["amount_billions"], (int, float))]
    assert len(amounts) == len(snapshot)
    assert sum(amounts) == pytest.approx(13.3, abs=0.05)


def test_reconciliation_ambiguity_attaches_closest_and_warns():
    md = (
        "Supply commitments were $10.1 billion. Vendor commitments were "
        "$9.5 billion. Future payments (in millions):\n"
        "| 2027 | $6,000 |\n| 2028 | $4,000 |"
    )
    rows: list[dict[str, object]] = []
    obligations._collect_note_rows(rows, "Commitments and Contingencies", md, FakeFiling())
    comps = [r for r in rows if r.get("schedule_component")]
    assert len(comps) == 2
    assert all(c["headline_type"] == "supply" for c in comps)
    warnings = [r.get("_reconciliation_warning") for r in rows if r.get("_reconciliation_warning")]
    assert len(warnings) == 1
    _warning = warnings[0]
    assert isinstance(_warning, str)
    assert "ambiguous" in _warning


def test_xbrl_gateway_provenance_stamps_fact_dates(monkeypatch: pytest.MonkeyPatch):
    """An August-filed fact is stamped August, never the February 10-K proxy."""
    _install_layered(
        monkeypatch,
        {"10-K": ({"Commitments and Contingencies": "No dollar amounts here."}, "2026-02-01")},
        [_fact()],
    )
    rows = obligations._xbrl_obligations("SYN")
    (row,) = [r for r in rows if r["type"] == "purchase_commitments"]
    assert row["filed"] == "2026-08-26"
    assert row["known_at"] == "2026-08-27T00:00:00Z"
    assert row["as_of"] == "2026-05-02"
    assert row["_accession"] == "000123-26-000001"
    assert row["concept"] == "PurchaseObligations"


def test_three_indemnities_yield_three_rows(monkeypatch: pytest.MonkeyPatch):
    """Three distinct indemnity excerpts in one filing: three rows, three hashes."""
    md = (
        "The company may indemnify its officers against certain claims. "
        "The company agreed to indemnify licensors for intellectual property "
        "infringement claims. The company may indemnify customers for tax "
        "positions taken in the ordinary course."
    )
    _install_layered(
        monkeypatch,
        {
            "10-K": ({"Commitments and Contingencies": md}, "2026-02-01"),
        },
    )
    result = obligations.get_obligations("SYN")
    assert "error" not in result
    _unquantified = result["unquantified_exposures"]
    assert isinstance(_unquantified, list)
    assert len(_unquantified) == 3
    assert len({e["content_hash"] for e in _unquantified}) == 3


def test_8k_lifecycle_chain_sums_zero(monkeypatch: pytest.MonkeyPatch):
    """Jan $10B + Mar $6B amendment + May (Item 1.02) termination, same
    agreement identity: 3 ledger rows, $0 current exposure — never $16B."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                (
                    "The company amended the guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $6 billion under the Agreements."
                ),
                "acc-mar",
            ),
            (
                "2026-05-20",
                ["Item 1.02"],
                (
                    "The company terminated the guarantee agreement with Alpha Holdings, under which exposure "
                    "was capped at $6 billion."
                ),
                "acc-may",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 3
    assert sorted(r.get("lifecycle_status") for r in ledger_8k) == [
        "terminated",
        "unknown",
        "unknown",
    ]
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    assert [r for r in _snapshot if r["type"] == "8k_guarantees"] == []


def test_coexisting_8k_guarantees_warn_without_resolution(monkeypatch: pytest.MonkeyPatch):
    """Two guarantees with distinct agreement identities stay additive."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                (
                    "The company entered into a second guarantee agreement with Beta Holdings, with aggregate "
                    "payment obligation cumulatively capped at $6 billion under the Agreements."
                ),
                "acc-mar",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap_8k = [r for r in _snapshot if r["type"] == "8k_guarantees"]
    assert sorted(r["amount_billions"] for r in snap_8k) == [6.0, 10.0]
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    _warnings = _coverage["warnings"]
    assert isinstance(_warnings, list)
    assert any("2 unresolved 8-K guarantees" in w for w in _warnings)


def test_8k_amendment_does_not_kill_unrelated_guarantee(monkeypatch: pytest.MonkeyPatch):
    """Jan A $10B + Feb B $3B + Mar amend-A-to-$6B: snapshot is $6B A + $3B
    B = $9B — never $6B (B killed by A's amendment)."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-02-10",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Beta Holdings, with aggregate payment "
                    "obligation cumulatively capped at $3 billion under the Agreements."
                ),
                "acc-feb",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                (
                    "The company amended the guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $6 billion under the Agreements."
                ),
                "acc-mar",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 3
    by_amount = {r["amount_billions"]: r for r in ledger_8k}
    assert by_amount[10.0].get("lifecycle_status") == "unknown"
    assert "lifecycle_status" not in by_amount[3.0]
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap_8k = [r for r in _snapshot if r["type"] == "8k_guarantees"]
    assert sorted(r["amount_billions"] for r in snap_8k) == [3.0, 6.0]
    assert sum(r["amount_billions"] for r in snap_8k) == 9.0


def test_8k_bare_guarantees_stay_additive(monkeypatch: pytest.MonkeyPatch):
    """Jan $10B + Feb $3B + Mar amend-to-$6B with no counterparty/label:
    3 ledger rows, none marked, snapshot $19B — never $6B."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement, with aggregate payment "
                    "obligation cumulatively capped at $10 billion."
                ),
                "acc-jan",
            ),
            (
                "2026-02-10",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement, with aggregate payment "
                    "obligation cumulatively capped at $3 billion."
                ),
                "acc-feb",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                (
                    "The company amended the guarantee agreement, with aggregate payment "
                    "obligation cumulatively capped at $6 billion."
                ),
                "acc-mar",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 3
    assert all("lifecycle_status" not in r for r in ledger_8k)
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap_8k = [r for r in _snapshot if r["type"] == "8k_guarantees"]
    assert sorted(r["amount_billions"] for r in snap_8k) == [3.0, 6.0, 10.0]
    assert sum(r["amount_billions"] for r in snap_8k) == 19.0
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    _warnings = _coverage["warnings"]
    assert isinstance(_warnings, list)
    assert any("3 unresolved 8-K guarantees" in w for w in _warnings)


def test_8k_amountless_termination_zeroes_exposure(monkeypatch: pytest.MonkeyPatch):
    """Jan Alpha $10B + May Item 1.02 termination with no dollar figure:
    2 ledger rows (May amount-None terminated), $0 current exposure."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-05-20",
                ["Item 1.02"],
                "The company terminated the Guarantee Agreement with Alpha Holdings.",
                "acc-may",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 2
    by_filed = {r["filed"]: r for r in ledger_8k}
    assert by_filed["2026-01-15"]["amount_billions"] == 10.0
    assert by_filed["2026-01-15"].get("lifecycle_status") == "unknown"
    assert by_filed["2026-05-20"]["amount_billions"] is None
    assert by_filed["2026-05-20"].get("lifecycle_status") == "terminated"
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    assert [r for r in _snapshot if r["type"] == "8k_guarantees"] == []


def test_8k_amountless_amendment_retains_last_quantified(monkeypatch: pytest.MonkeyPatch):
    """Jan Alpha $10B + Mar amount-less amendment: Jan stays `amended`,
    snapshot retains $10B — never $0."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                "The company amended the Guarantee Agreement with Alpha Holdings.",
                "acc-mar",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 2
    by_filed = {r["filed"]: r for r in ledger_8k}
    assert by_filed["2026-01-15"]["amount_billions"] == 10.0
    assert by_filed["2026-01-15"].get("lifecycle_status") == "amended"
    assert by_filed["2026-03-10"]["amount_billions"] is None
    assert "lifecycle_status" not in by_filed["2026-03-10"]
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap_8k = [r for r in _snapshot if r["type"] == "8k_guarantees"]
    assert len(snap_8k) == 1 and snap_8k[0]["amount_billions"] == 10.0
    assert snap_8k[0].get("lifecycle_status") == "amended"
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    _warnings = _coverage["warnings"]
    assert isinstance(_warnings, list)
    assert any("did not disclose a replacement amount" in w for w in _warnings)


def test_8k_amended_then_terminated_zeroes_with_notice(monkeypatch: pytest.MonkeyPatch):
    """Jan $10B + Mar amount-less amendment + May amount-less termination:
    $0 snapshot with an unknown-canceled-amount notice."""
    _install_with_8k(
        monkeypatch,
        {},
        [
            (
                "2026-01-15",
                ["Item 1.01"],
                (
                    "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
                    "obligation cumulatively capped at $10 billion under the Agreements."
                ),
                "acc-jan",
            ),
            (
                "2026-03-10",
                ["Item 1.01"],
                "The company amended the Guarantee Agreement with Alpha Holdings.",
                "acc-mar",
            ),
            (
                "2026-05-20",
                ["Item 1.02"],
                "The company terminated the Guarantee Agreement with Alpha Holdings.",
                "acc-may",
            ),
        ],
    )
    result = obligations.get_obligations("SYN")
    _obligations = result["obligations"]
    assert isinstance(_obligations, list)
    ledger_8k = [r for r in _obligations if r["type"] == "8k_guarantees"]
    assert len(ledger_8k) == 3
    by_filed = {r["filed"]: r for r in ledger_8k}
    assert by_filed["2026-01-15"].get("lifecycle_status") == "unknown"
    assert by_filed["2026-05-20"]["amount_billions"] is None
    assert by_filed["2026-05-20"].get("lifecycle_status") == "terminated"
    _snapshot = result["current_snapshot"]
    assert isinstance(_snapshot, list)
    assert [r for r in _snapshot if r["type"] == "8k_guarantees"] == []
    _coverage = result["coverage"]
    assert isinstance(_coverage, dict)
    _warnings = _coverage["warnings"]
    assert isinstance(_warnings, list)
    assert any("canceled amount unknown" in w for w in _warnings)


def test_8k_lifecycle_amendment_marks_earlier_row(monkeypatch: pytest.MonkeyPatch):
    """A later amount-less amendment stamps the earlier quantified row amended; snapshot retains $10B."""
    jan = (
        "2026-01-15",
        ["Item 1.01"],
        (
            "The company entered into a guarantee agreement with Alpha Holdings, with aggregate payment "
            "obligation cumulatively capped at $10 billion under the Agreements."
        ),
        "acc-jan",
    )
    mar = (
        "2026-03-10",
        ["Item 1.01"],
        "The company amended the Guarantee Agreement with Alpha Holdings.",
        "acc-mar",
    )
    _install_with_8k(monkeypatch, {}, [jan])
    _jan_result = obligations.get_obligations("SYN")
    _jan_obligations = _jan_result["obligations"]
    assert isinstance(_jan_obligations, list)
    jan_only = [r for r in _jan_obligations if r["type"] == "8k_guarantees"]
    assert len(jan_only) == 1
    assert jan_only[0]["amount_billions"] == 10.0
    assert "lifecycle_status" not in jan_only[0]
    _install_with_8k(monkeypatch, {}, [jan, mar])
    _both_result = obligations.get_obligations("SYN")
    _both_obligations = _both_result["obligations"]
    assert isinstance(_both_obligations, list)
    both = [r for r in _both_obligations if r["type"] == "8k_guarantees"]
    assert len(both) == 2
    by_filed = {r["filed"]: r for r in both}
    assert by_filed["2026-01-15"]["amount_billions"] == 10.0
    assert by_filed["2026-01-15"].get("lifecycle_status") == "amended"
    assert by_filed["2026-03-10"]["amount_billions"] is None
    _snapshot = _both_result["current_snapshot"]
    assert isinstance(_snapshot, list)
    snap = [r for r in _snapshot if r["type"] == "8k_guarantees"]
    assert len(snap) == 1 and snap[0]["amount_billions"] == 10.0
    assert snap[0].get("lifecycle_status") == "amended"
    _coverage = _both_result["coverage"]
    assert isinstance(_coverage, dict)
    _warnings = _coverage["warnings"]
    assert isinstance(_warnings, list)
    assert any("did not disclose a replacement amount" in w for w in _warnings)


def test_capital_buyback_stays_capital_only(monkeypatch: pytest.MonkeyPatch):
    """A dollar-less buyback lands in capital only; quantified ledger and snapshot stay empty."""
    _install_layered(
        monkeypatch,
        {
            "10-K": ({"Stock Compensation": "The board authorized a share repurchase program."}, "2026-02-01"),
        },
    )
    result = obligations.get_obligations("SYN")
    _capital = result["capital_allocation"]
    assert isinstance(_capital, list)
    assert len(_capital) == 1
    assert result["obligations"] == [] and result["current_snapshot"] == []


def test_payment_timing_retunes_content_hash():
    """A 95/24 front-loaded horizon vs 90/29 retunes identity; identical rows hash stable."""
    horizon = {
        "paid_in_remainder_of_fy": "2027",
        "paid_in_remainder_billions": 95.0,
        "paid_after_remainder_billions": 24.0,
    }
    row = _obligation_row(payment_horizon=horizon, schedule=None, amount_billions=119.0)
    row["content_hash"] = obligations._content_hash(row)
    assert obligations._content_hash(dict(row)) == row["content_hash"]
    corrected = _obligation_row(
        payment_horizon={**horizon, "paid_in_remainder_billions": 90.0, "paid_after_remainder_billions": 29.0},
        schedule=None,
        amount_billions=119.0,
    )
    corrected["content_hash"] = obligations._content_hash(corrected)
    assert corrected["content_hash"] != row["content_hash"]


def test_unquantified_hash_tracks_excerpt_and_trigger():
    base = {
        "type": "indemnities",
        "amount_billions": None,
        "filed": "2026-02-01",
        "trigger": "unknown",
        "excerpt": "The company may indemnify its officers.",
        "_accession": "acc-10k",
    }
    assert obligations._content_hash(dict(base)) == obligations._content_hash(dict(base))
    assert obligations._content_hash(
        {**base, "excerpt": "  THE company MAY\nindemnify its   officers. "}
    ) == obligations._content_hash(dict(base))
    assert obligations._content_hash(
        {**base, "excerpt": "The company may indemnify its directors."}
    ) != obligations._content_hash(dict(base))
    assert obligations._content_hash({**base, "trigger": "counterparty_default"}) != obligations._content_hash(
        dict(base)
    )
    assert obligations._content_hash({**base, "_accession": "acc-10q"}) != obligations._content_hash(dict(base))
