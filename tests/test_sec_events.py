"""Offline tests for 8-K events, exhibits, and filing diffs (no network)."""

from types import SimpleNamespace
from typing import ClassVar, NoReturn

import pytest

from app.sec import diffs, documents, events8k

ACC = "0000000000-26-000001"


def test_parse_multi_item() -> None:
    items = {
        "Item 1.03": "bankruptcy filing text",
        "1.02": "",  # empty -> skipped
        "item_202": "earnings beat, see EX-99.1 attached",
        "9.01": "exhibits list",
        "Item 11.11": "unknown passes through untouched",
    }
    events = events8k.parse_8k_events(ACC, items, event_date="2026-01-01")
    assert [(e.item_number, e.item_name) for e in events] == [
        ("1.03", events8k.KNOWN_8K_ITEMS["1.03"]),
        ("2.02", events8k.KNOWN_8K_ITEMS["2.02"]),
        ("9.01", events8k.KNOWN_8K_ITEMS["9.01"]),
    ]
    assert events[1].exhibit_refs == ("EX-99.1",)
    assert all(e.accession_no == ACC and e.event_date == "2026-01-01" for e in events)
    assert events[0].to_dict()["item_number"] == "1.03"


def test_extract_wrapper_skips_falsy() -> None:
    texts = {"Item 1.03": "bankruptcy text", "Item 2.02": ""}

    class R:
        items: ClassVar[object] = ["Item 1.03", "Item 2.02"]

        def __getitem__(self, name: str) -> object:
            return texts[name]

    events = events8k.extract_8k_events(R(), ACC)
    assert [e.item_number for e in events] == ["1.03"]


def test_exhibits(monkeypatch: pytest.MonkeyPatch) -> None:
    atts = [
        SimpleNamespace(
            document_type="EX-99.1", description="Press release", document="ex991.htm", url="https://x/ex991.htm"
        ),
        SimpleNamespace(
            document_type="EX-10.1", description="Agreement", document="ex101.htm", url="https://x/ex101.htm"
        ),
    ]

    def _fake_get(accession_no: str) -> SimpleNamespace:
        return SimpleNamespace(exhibits=atts)

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_get)
    rows = documents.get_filing_exhibits(ACC)
    assert rows[0] == {
        "accession_no": ACC,
        "exhibit": "EX-99.1",
        "description": "Press release",
        "document": "ex991.htm",
        "url": "https://x/ex991.htm",
    }
    assert documents.get_filing_exhibit(ACC, "ex-99.1")["document"] == "ex991.htm"
    with pytest.raises(ValueError):
        documents.get_filing_exhibit(ACC, "EX-99.9")


def test_diff_specialization_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import filings

    def _fake_text(accession_no: str, document: object = None) -> str:
        return "a\nb\nc\n" if accession_no == "old" else "a\nB\nc\nd\n"

    def _fake_filing(accession_no: str) -> SimpleNamespace:
        return SimpleNamespace(form="10-K")

    monkeypatch.setattr(documents, "get_sec_filing_text", _fake_text)
    monkeypatch.setattr(filings, "get_sec_filing", _fake_filing)
    out = diffs.diff_filings("new", "old")
    assert out["specialization"] == "10-K/10-Q"
    assert out["added"] == 2 and out["removed"] == 1
    assert out["truncated"] is False


def test_diff_truncated_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import filings

    texts = {"n": "\n".join(str(i) for i in range(1000, 2000)), "o": "\n".join(str(i) for i in range(1000))}

    def _fake_text(accession_no: str, document: object = None) -> str:
        return texts[accession_no]

    def _fake_filing(accession_no: str) -> SimpleNamespace:
        return SimpleNamespace(form="8-K")

    monkeypatch.setattr(documents, "get_sec_filing_text", _fake_text)

    monkeypatch.setattr(filings, "get_sec_filing", _fake_filing)
    out = diffs.diff_filings("n", "o")
    diff_lines = out["diff_lines"]
    assert isinstance(diff_lines, list)
    assert out["truncated"] is True and len(diff_lines) == 500

    def boom(accession_no: str, document: object = None) -> NoReturn:
        raise ValueError("bad accession")

    monkeypatch.setattr(documents, "get_sec_filing_text", boom)
    assert "error" in diffs.diff_filings("n", "o")


def test_diff_section_full_uses_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import filings

    seen: list[object] = []

    def _fake_text(accession_no: str, document: object = None) -> str:
        seen.append(document)
        return "x\n"

    def _fake_filing(accession_no: str) -> object:
        return SimpleNamespace(form="10-K")

    monkeypatch.setattr(filings, "get_sec_filing", _fake_filing)
    monkeypatch.setattr(documents, "get_sec_filing_text", _fake_text)
    out = diffs.diff_filings("n", "o", section="full")
    assert "error" not in out
    assert seen == [None, None]


def test_diff_section_miss_names_available_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [SimpleNamespace(document_name="aapl-20250927.htm"), SimpleNamespace(document_name="exhibit211.htm")]
    seen: list[tuple[str, str | None]] = []

    def _fake_docs(accession_no: str) -> object:
        return docs

    def _missing_heading(accession_no: str, *, section: str | None = None) -> NoReturn:
        seen.append((accession_no, section))
        raise ValueError("section not found: 'risk_factors'; available sections: Item 1. Business; Item 2. Properties")

    def _no_full_text(accession_no: str, document: object = None) -> NoReturn:
        raise AssertionError("a missing heading must not return the full document")

    monkeypatch.setattr(documents, "list_sec_documents", _fake_docs)
    monkeypatch.setattr(documents, "get_sec_document", _missing_heading)
    monkeypatch.setattr(documents, "get_sec_filing_text", _no_full_text)
    out = diffs.diff_filings("n", "o", section="risk_factors")
    err = out["error"]
    assert isinstance(err, str)
    assert "risk_factors" in err
    assert "available sections: Item 1. Business; Item 2. Properties" in err
    assert seen == [("n", "risk_factors")]
