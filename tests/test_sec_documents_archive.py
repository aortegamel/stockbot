"""Archive-first bounded SEC documents (offline; edgar faked via monkeypatch)."""

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from app import tools
from app.policy import Capability, RequestContext
from app.sec import documents, filings

ACC = "0000000000-26-000001"
DOC = "primary.htm"
BIG = "0123456789abcdef" * 3125
assert len(BIG) == 50_000

PDF_BYTES = b"%PDF-1.4\nfake pdf bytes\n%%EOF\n"
PDF_TEXT = "PAGE\n%PDF-1.4\nfake pdf bytes\n%%EOF"


def _fake_pdftotext(tmp_path: Path, *, exit_code: int = 0) -> Callable[[str], str]:
    """Stand-in ``shutil.which``: an extractor that emits NUL/BEL/FF noise, then the file it was handed."""
    script = tmp_path / "pdftotext"
    script.write_text(f'#!/bin/sh\nprintf "PAGE\\000\\007\\f\\n"\n[ {exit_code} -eq 0 ] && cat "$3"\n')
    script.chmod(0o755)

    def _which(name: str) -> str:
        del name
        return str(script)

    return _which


def _no_pdftotext(name: str) -> None:
    """Stand-in ``shutil.which`` reporting no extractor installed."""
    del name


class _FakeAttachment:
    def __init__(self, text: str, document: str = DOC, accession: str = ACC) -> None:
        self.document = document
        self.description = "10-K primary"
        self.url = f"https://sec/{accession}/{document}"
        self.content = text


class _BytesAttachment:
    document = DOC
    description: str | None = None
    url = "https://sec/bytes"

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def download(self) -> bytes:
        return self._payload


class _FakeFiling:
    def __init__(
        self,
        text: str | _BytesAttachment,
        accession: str = ACC,
        *,
        extra_attachments: tuple[_FakeAttachment, ...] = (),
    ) -> None:
        self.form = "10-K"
        self.filing_date = "2026-01-15"
        self.acceptance_datetime: str | None = None
        self.accession_no = accession
        self.company = "Fake Corp"
        self.cik = 123
        self.homepage_url = f"https://sec/{accession}"
        self.period_of_report = "2025-12-31"
        self._att = _FakeAttachment(text, accession=accession) if isinstance(text, str) else text
        self._extra_attachments = extra_attachments

    @property
    def document(self) -> _FakeAttachment | _BytesAttachment:
        return self._att

    @property
    def attachments(self) -> list[_FakeAttachment | _BytesAttachment]:
        return [self._att, *self._extra_attachments]


def _ctx() -> RequestContext:
    return RequestContext("research", frozenset({Capability.RESEARCH}))


class _EmptyCompany:
    def get_filings(self, **kwargs: object) -> list[object]:
        return []


def _empty_company(ticker_or_cik: object) -> _EmptyCompany:
    return _EmptyCompany()


@pytest.fixture(autouse=True)
def _empty_company_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(filings, "get_company", _empty_company)


def _annual_html(
    risk_text: str, *, business: str = "Annual business text.", properties: str = "Annual properties text."
) -> str:
    return (
        "<html><body><h1>Annual Report</h1>"
        f"<h2>Item 1. Business</h2><p>{business}</p>"
        "<h2>Item 1A. Risk Factors</h2>"
        + "".join(f"<p>{line}</p>" for line in risk_text.splitlines())
        + f"<h2>Item 2. Properties</h2><p>{properties}</p></body></html>"
    )


def test_archive_first_bounded_windows_and_local_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))

    def _fake_get(accession_no: str) -> _FakeFiling:
        return _FakeFiling(BIG)

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_get)

    first = tools.execute_tool("get_sec_document", {"accession_no": ACC}, "test", context=_ctx())
    assert first["text"] == BIG[:12000]
    assert (first["offset"], first["end_offset"], first["total_chars"]) == (0, 12000, 50000)
    assert first["more_available"] is True
    assert first["cache_hit"] is False
    assert first["source_representation"] == "source_bytes"
    digest = hashlib.sha256(BIG.encode("utf-8")).hexdigest()
    assert first["content_hash"] == digest
    assert first["source_content_hash"] == digest
    # selective retention: a live read archives nothing until evidence is accepted
    assert first["raw_archive_path"] is None
    assert first["retrieved_at"] is None
    assert first["known_at"] and first["source_url"]

    # seed the archive directly; the window is served back with the network down
    from app.sec import archive as _archive

    record = _archive.archive_sec_document(
        ACC,
        DOC,
        BIG.encode("utf-8"),
        url=str(first["source_url"]),
        retrieved_at="2026-01-15T00:00:00Z",
        metadata={"known_at": first["known_at"], "filed_at": "2026-01-15", "representation": "normalized_text"},
        root=tmp_path / "raw",
    )

    def _boom(accession_no: str) -> NoReturn:
        raise RuntimeError("network down")

    monkeypatch.setattr(documents, "get_by_accession_number", _boom)
    second = tools.execute_tool(
        "get_sec_document", {"accession_no": ACC, "document_name": DOC, "offset": 12000}, "test", context=_ctx()
    )
    assert second["text"] == BIG[12000:24000]
    assert (second["offset"], second["end_offset"]) == (12000, 24000)
    assert second["more_available"] is True
    assert second["cache_hit"] is True
    assert second["content_hash"] == first["content_hash"]
    assert second["source_content_hash"] == first["source_content_hash"]
    assert second["known_at"] == first["known_at"]
    assert second["retrieved_at"] == "2026-01-15T00:00:00Z"
    assert second["raw_archive_path"] == str(record.payload_path)

    assert documents.get_sec_filing_text(ACC, DOC) == BIG

    for bad in ({"offset": -1}, {"max_chars": 0}, {"max_chars": 32001}, {"offset": 60000}):
        rejected = tools.execute_tool(
            "get_sec_document", {"accession_no": ACC, "document_name": DOC, **bad}, "test", context=_ctx()
        )
        assert rejected["error_type"] == "invalid_tool_arguments"


def test_conflicting_revisions_latest_warns_historical_conflicts(tmp_path: Path) -> None:
    from app.sec import archive as _archive

    acc = "0000000000-26-000002"
    known_at = "2026-02-01T00:00:00Z"
    _archive.archive_sec_document(
        acc,
        DOC,
        b"aaa revision text",
        url="https://sec/x",
        retrieved_at="2026-02-01T00:00:00Z",
        metadata={"known_at": known_at, "filed_at": "2026-02-01", "representation": "normalized_text"},
        root=tmp_path / "raw",
    )
    with pytest.warns(UserWarning, match="immutable revision"):
        _archive.archive_sec_document(
            acc,
            DOC,
            b"zzz revision text",
            url="https://sec/x",
            retrieved_at="2026-02-02T00:00:00Z",
            metadata={"known_at": known_at, "filed_at": "2026-02-01", "representation": "normalized_text"},
            root=tmp_path / "raw",
        )

    latest = documents.get_sec_document(acc, DOC, data_root=tmp_path)
    # same known_at: latest retrieved_at wins, with the conflict retained and named
    assert latest["text"] == "zzz revision text"
    assert latest["cache_hit"] is True
    warnings = latest.get("warnings", [])
    assert isinstance(warnings, list)
    assert any(isinstance(w, str) and "conflicting revisions" in w for w in warnings)

    conflict = documents.get_sec_document(acc, DOC, as_of="2026-03-01", data_root=tmp_path)
    assert conflict["error_type"] == "pit_revision_conflict"


def test_byte_attachment_archives_source_bytes(tmp_path: Path) -> None:
    from app.sec import archive as _archive

    payload = b"\xff\xfe binary \x00\x01 document bytes"

    acc = "0000000000-26-000003"
    monkeypatch_filing = _FakeFiling(_BytesAttachment(payload), accession=acc)

    def _fake_get(accession_no: str) -> _FakeFiling:
        return monkeypatch_filing

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(documents, "get_by_accession_number", _fake_get)
        out = documents.get_sec_document(acc, data_root=tmp_path)
    assert out["source_content_hash"] == hashlib.sha256(payload).hexdigest()
    assert out["content_hash"] == hashlib.sha256(payload.decode("utf-8", "replace").encode("utf-8")).hexdigest()
    assert out["source_content_hash"] != out["content_hash"]
    # selective retention: the live read archives nothing
    assert out["raw_archive_path"] is None
    assert out["source_representation"] == "binary_unsupported"
    assert out["text"] == ""
    assert out["total_chars"] == 0
    warnings = out["warnings"]
    assert isinstance(warnings, list)
    assert any("has no text representation" in str(w) for w in warnings)

    # seed the archive directly; the window is served back
    record = _archive.archive_sec_document(
        acc,
        DOC,
        payload,
        url="https://sec/bytes",
        retrieved_at="2026-01-15T00:00:00Z",
        metadata={"known_at": "2026-01-15", "filed_at": "2026-01-15", "representation": "source_bytes"},
        root=tmp_path / "raw",
    )
    served = documents.get_sec_document(acc, DOC, data_root=tmp_path)
    assert served["cache_hit"] is True
    assert served["source_content_hash"] == hashlib.sha256(payload).hexdigest()
    assert Path(str(served["raw_archive_path"])).read_bytes() == payload
    assert served["raw_archive_path"] == str(record.payload_path)
    # a binary payload never leaves as text: NUL-free empty text plus an explicit marker
    assert served["source_representation"] == "binary_unsupported"
    assert served["text"] == ""
    assert served["total_chars"] == 0
    assert served["cache_hit"] is True
    served_warnings = served["warnings"]
    assert isinstance(served_warnings, list)
    assert any("has no text representation" in str(w) for w in served_warnings)


def test_archived_bounded_uses_row_name_and_slice() -> None:
    row: dict[str, object] = {
        "document_name": "row-doc.htm",
        "text": "x" * 100,
        "source_url": "https://sec/row",
        "content_hash": "ch",
        "source_content_hash": "sch",
        "source_representation": "normalized_text",
        "raw_archive_path": "/tmp/raw",
        "filed_at": "2024-01-01",
        "known_at": "2024-01-02",
        "retrieved_at": "2024-01-03",
    }
    out = documents._archived_bounded(
        "0000000000-24-000001", "arg-doc.htm", row, "0123456789", "https://sec/row", 2, 4, ["w1"]
    )
    assert out["document_name"] == "row-doc.htm"
    assert out["text"] == "2345"
    assert out["cache_hit"] is True and out["cache_type"] == "stockbot_archive"
    assert out["warnings"] == ["w1"]


def test_archived_bounded_falls_back_to_arg_name() -> None:
    out = documents._archived_bounded("0000000000-24-000002", "arg-doc.htm", {}, "", None, 0, None, None)
    assert out["document_name"] == "arg-doc.htm"
    assert "warnings" not in out


def test_pdf_payload_without_pdftotext_is_binary_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acc = "0000000000-26-000004"
    filing = _FakeFiling(_BytesAttachment(PDF_BYTES), accession=acc)

    def _fake_get(accession_no: str) -> _FakeFiling:
        return filing

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_get)
    monkeypatch.setattr(shutil, "which", _no_pdftotext)
    out = documents.get_sec_document(acc, data_root=tmp_path)

    assert out["text"] == ""
    assert out["source_representation"] == "binary_unsupported"
    assert (out["offset"], out["end_offset"], out["total_chars"], out["more_available"]) == (0, 0, 0, False)
    assert out["source_content_hash"] == hashlib.sha256(PDF_BYTES).hexdigest()
    assert out["content_hash"] == hashlib.sha256(PDF_BYTES.decode("utf-8", "replace").encode("utf-8")).hexdigest()
    assert out["raw_archive_path"] is None
    warnings = out["warnings"]
    assert isinstance(warnings, list)
    assert any("pdftotext unavailable" in str(w) for w in warnings)


def test_live_pdf_bytes_extract_through_pdftotext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acc = "0000000000-26-000005"
    filing = _FakeFiling(_BytesAttachment(PDF_BYTES), accession=acc)

    def _fake_get(accession_no: str) -> _FakeFiling:
        return filing

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_get)
    monkeypatch.setattr(shutil, "which", _fake_pdftotext(tmp_path))
    out = documents.get_sec_document(acc, data_root=tmp_path)

    assert out["source_representation"] == "pdf_extracted"
    # the extractor echoed the temp file it was handed: the exact live payload bytes
    assert out["text"] == PDF_TEXT
    assert out["total_chars"] == len(PDF_TEXT)
    assert out["warnings"] == ["pdf text extracted via pdftotext"]
    assert out["source_content_hash"] == hashlib.sha256(PDF_BYTES).hexdigest()


def test_pdf_extraction_failure_stays_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acc = "0000000000-26-000006"
    filing = _FakeFiling(_BytesAttachment(PDF_BYTES), accession=acc)

    def _fake_get(accession_no: str) -> _FakeFiling:
        return filing

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_get)
    monkeypatch.setattr(shutil, "which", _fake_pdftotext(tmp_path, exit_code=1))
    out = documents.get_sec_document(acc, data_root=tmp_path)

    assert out["text"] == ""
    assert out["source_representation"] == "binary_unsupported"
    warnings = out["warnings"]
    assert isinstance(warnings, list)
    assert any("pdf text extraction failed" in str(w) for w in warnings)


def test_archived_pdf_row_extracts_text_and_drops_control_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.sec import archive as _archive

    acc = "0000000000-26-000007"
    record = _archive.archive_sec_document(
        acc,
        "form8k2.pdf",
        PDF_BYTES,
        url="https://sec/form8k2.pdf",
        retrieved_at="2015-04-29T00:00:00Z",
        metadata={"known_at": "2015-04-29", "filed_at": "2015-04-29", "representation": "source_bytes"},
        root=tmp_path / "raw",
    )
    monkeypatch.setattr(shutil, "which", _fake_pdftotext(tmp_path))

    out = documents.get_sec_document(acc, "form8k2.pdf", data_root=tmp_path)
    assert out["source_representation"] == "pdf_extracted"
    assert out["text"] == PDF_TEXT
    assert not any(char in PDF_TEXT for char in "\x00\x07\x0c")
    assert out["total_chars"] == len(PDF_TEXT)
    assert out["raw_archive_path"] == str(record.payload_path)
    assert out["warnings"] == ["pdf text extracted via pdftotext"]

    raw_out = documents.get_sec_document(acc, "form8k2.pdf", data_root=tmp_path, raw=True)
    assert raw_out["view"] == "raw"
    assert raw_out["text"] == PDF_TEXT + "\n"


def test_archived_row_without_known_at_rejected_historically_but_served_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row archived without known_at never passes a historical as_of; latest still wins."""
    from app.sec import archive as _archive

    acc = "0000000000-26-000009"
    _archive.archive_sec_document(
        acc,
        DOC,
        b"legacy archived text",
        url="https://sec/legacy",
        retrieved_at="2026-01-15T00:00:00Z",
        metadata={"evidence_id": "legacy", "session_id": "legacy"},
        root=tmp_path / "raw",
    )
    latest = documents.get_sec_document(acc, DOC, data_root=tmp_path)
    assert latest["text"] == "legacy archived text"
    assert latest["cache_hit"] is True
    calls: list[str] = []

    def _down(accession_no: str) -> NoReturn:
        calls.append(accession_no)
        raise AssertionError("live fetch must not run when archive rows exist but fail PIT")

    monkeypatch.setattr(documents, "get_by_accession_number", _down)
    rows = documents._archived_rows_of(acc, DOC, "2026-03-01", tmp_path, None)
    assert rows == []
    assert calls == []


def test_accepted_2026_filing_rejected_at_2025_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P0 end-to-end: a 2026 acceptance is never served to a 2025 historical read."""
    import tests.research_source_seam as seam
    from app.research import service as svc
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    real_get_sec_document = documents.get_sec_document
    seam.install(monkeypatch)
    try:
        sid = svc.create_research("NVDA demand?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
        src = repo.list_jobs(sid)[0].job_id
        passage = "Data center revenue grew 142 percent in fiscal 2026."
        handle = seam.handle_for(passage, accession=ACC, document=DOC, known_at="2026-01-20T00:00:00+00:00")
        stored = svc.record_evidence(
            sid,
            src,
            {
                "evidence_id": f"{sid}:ev:1",
                "wave_id": 1,
                "content": "c",
                "claim_text": "c",
                "subject": "NVDA",
                "source_name": "SEC",
                "source_uri": "https://sec.gov/x",
                "source_record_id": ACC,
                "document_name": DOC,
                "matching_passage": passage,
                "known_at": "2024-01-01T00:00:00+00:00",
                "source_handle": handle,
            },
            repo=repo,
        )
        row = repo.get_evidence(str(stored["evidence_id"]))
        assert str(row["known_at"]).startswith("2026")
        sid25 = svc.create_research("NVDA demand?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
        src25 = repo.list_jobs(sid25)[0].job_id
        with pytest.raises(ValueError, match="ERR_SEC_HANDLE_UNREADABLE"):
            svc.record_evidence(
                sid25,
                src25,
                {
                    "evidence_id": f"{sid25}:ev:1",
                    "wave_id": 1,
                    "content": "c",
                    "claim_text": "c",
                    "subject": "NVDA",
                    "source_name": "SEC",
                    "source_uri": "https://sec.gov/x",
                    "source_record_id": ACC,
                    "document_name": DOC,
                    "matching_passage": passage,
                    "known_at": "2026-01-20T00:00:00+00:00",
                    "source_handle": handle,
                },
                repo=repo,
            )
    finally:
        monkeypatch.setattr(documents, "get_sec_document", real_get_sec_document)
    latest = documents.get_sec_document(ACC, DOC, data_root=tmp_path)
    assert latest["text"] == passage
    assert latest["cache_hit"] is True

    assert documents._archived_rows_of(ACC, DOC, "2025-01-01", tmp_path, None) == []

    def _fake_2026_filing(accession_no: str) -> _FakeFiling:
        return _FakeFiling(BIG, accession=ACC)

    monkeypatch.setattr(documents, "get_by_accession_number", _fake_2026_filing)
    with pytest.raises(ValueError, match="not known as of"):
        documents.get_sec_document(ACC, DOC, as_of="2025-01-01", data_root=tmp_path)


# ---------------------------------------------------------------------------
# resolve_sec_official: official www.sec.gov page per cited row (fetch faked)
# ---------------------------------------------------------------------------

OFF_ACC = "0000320193-26-000010"
OFF_CIK = 320193
OFF_DIR = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000010"
OFF_INDEX = f"{OFF_DIR}/{OFF_ACC}-index.htm"
OFF_PRIMARY = f"{OFF_DIR}/fake-10q.htm"
OFF_INDEX_HTML = (
    f"<html><body><div>SEC Accession No. {OFF_ACC}</div><table>"
    '<tr><td>1</td><td><a href="/ix?doc=/Archives/edgar/data/320193/000032019326000010/fake-10q.htm">'
    "fake-10q.htm</a></td></tr>"
    f'<tr><td>2</td><td><a href="/Archives/edgar/data/320193/000032019326000010/{OFF_ACC}.txt">txt</a></td></tr>'
    "</table></body></html>"
)
_Q2 = "<xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate>"
_YTD = "<xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate>"
_SEGMENT = "<xbrli:segment><xbrldi:explicitMember dimension='a:SegmentAxis'>a:DataCenter</xbrldi:explicitMember></xbrli:segment>"


def _ix_context(context_id: str, period: str, segment: str = "") -> str:
    return (
        f'<xbrli:context id="{context_id}"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier>{segment}'
        f"</xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>"
    )


def _ix_fact(name: str, context: str, text: str, fact_id: str, scale: str = "6", extra: str = "") -> str:
    return (
        f'<ix:nonFraction name="{name}" contextRef="{context}" unitRef="usd" decimals="-6" scale="{scale}" '
        f'format="ixt:num-dot-decimal" id="{fact_id}"{extra}>{text}</ix:nonFraction>'
    )


OFF_PRIMARY_HTML = (
    '<html><body><div style="display:none"><ix:header><ix:hidden>'
    + _ix_fact("us-gaap:Revenues", "c-1", "81,615", "f-hidden")
    + "</ix:hidden><ix:resources>"
    + _ix_context("c-1", _Q2)
    + _ix_context("c-2", _Q2, _SEGMENT)
    + _ix_context("c-3", _YTD)
    + _ix_context("c-4", "<xbrli:instant>2026-07-20</xbrli:instant>")
    + "</ix:resources></ix:header></div>"
    + "<p>Shares outstanding as of July 20, 2026 was "
    + _ix_fact("dei:EntityCommonStockSharesOutstanding", "c-4", "24.3", "f-5", scale="9")
    + " billion.</p><table>"
    + f"<tr><td>Data center</td><td>$</td><td>{_ix_fact('us-gaap:Revenues', 'c-2', '81,615', 'f-seg')}</td></tr>"
    + f"<tr><td>Revenue</td><td>$</td><td>{_ix_fact('us-gaap:Revenues', 'c-1', '81,615', 'f-1')}</td>"
    + f"<td>$</td><td>{_ix_fact('us-gaap:Revenues', 'c-3', '160,000', 'f-2')}</td></tr>"
    + "<tr><td>Net loss</td><td>$</td><td>("
    + _ix_fact("us-gaap:NetIncomeLoss", "c-1", "1,234", "f-3", scale="3", extra=' sign="-"')
    + ")</td></tr>"
    + f"<tr><td>Diluted</td><td>$</td><td>{_ix_fact('us-gaap:EarningsPerShareDiluted', 'c-1', '1.08', 'f-6', scale='0')}"
    + "</td></tr></table><p>Notes</p>"
    + f"<table><tr><td>Total revenue</td><td>{_ix_fact('us-gaap:Revenues', 'c-1', '81,615', 'f-9')}</td></tr></table>"
    + "</body></html>"
)
F4_ACC = "0001197647-26-000011"
F4_DIR = "https://www.sec.gov/Archives/edgar/data/1045810/000119764726000011"
F4_INDEX_HTML = (
    f"<html><body><div>SEC Accession No. {F4_ACC}</div><table>"
    '<tr><td>1</td><td><a href="/Archives/edgar/data/1045810/000119764726000011/xslF345X06/form4.xml">form4.html</a></td></tr>'
    '<tr><td>1</td><td><a href="/Archives/edgar/data/1045810/000119764726000011/form4.xml">form4.xml</a></td></tr>'
    "</table></body></html>"
)
F4_HTML = (
    "<html><body><table>"
    "<tr><td>Common</td><td>57,378</td><td>D</td></tr>"
    "<tr><td>Common</td><td>10/07/2026</td><td>G</td><td>500,000</td><td>D</td><td>$0</td><td>23,671,360</td><td>I</td></tr>"
    "</table></body></html>"
)
COMPANY_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&owner=include&count=10"
COMPANY_HTML = "<html><body><div>APPLE INC CIK#: 0000320193 (see all company filings)</div></body></html>"


class _FakeSec:
    """www.sec.gov stand-in: canned pages, every fetch recorded, unknown URLs fail like a network error."""

    def __init__(self) -> None:
        self.pages = {
            OFF_INDEX: OFF_INDEX_HTML.encode(),
            OFF_PRIMARY: OFF_PRIMARY_HTML.encode(),
            f"{F4_DIR}/{F4_ACC}-index.htm": F4_INDEX_HTML.encode(),
            f"{F4_DIR}/xslF345X06/form4.xml": F4_HTML.encode(),
            COMPANY_URL: COMPANY_HTML.encode(),
        }
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.pages:
            raise OSError(f"fetch failed: {url}")
        return self.pages[url]


@pytest.fixture
def sec_www(monkeypatch: pytest.MonkeyPatch) -> _FakeSec:
    fake = _FakeSec()
    monkeypatch.setattr(documents, "_sec_get", fake)
    return fake


def _official(tmp_path: Path, tool: str, row: dict[str, object], payload: dict[str, object] | None = None):
    page = documents.resolve_sec_official(tool, row, payload or {}, root=tmp_path)
    assert page is not None
    payload_bytes = page["payload"]
    assert isinstance(payload_bytes, bytes)
    quote = page["quote"]
    assert isinstance(quote, str) and quote
    # The quote occurs in the same renderer output the viewer uses for its basis.
    assert quote in documents._render_text(payload_bytes.decode("utf-8", "replace"))
    assert page["sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    return page


def test_official_xbrl_fact_anchor_and_row_quote(tmp_path: Path, sec_www: _FakeSec) -> None:
    """Dimensional, hidden, YTD, and later duplicate facts all lose to the first visible statement row."""
    row: dict[str, object] = {"concept": "us-gaap:Revenues", "value": 81615000000.0, "period_end": "2026-06-30"}
    page = _official(tmp_path, "get_xbrl_facts", {**row, "accession": OFF_ACC}, {"cik": OFF_CIK})
    assert (page["url"], page["media"], page["anchor"]) == (OFF_PRIMARY, "html", "f-1")
    assert page["quote"] == "Revenue | $ | 81,615"


def test_official_fact_scale_and_sign(tmp_path: Path, sec_www: _FakeSec) -> None:
    loss: dict[str, object] = {"concept": "NetIncomeLoss", "period_end": "2026-06-30", "accession": OFF_ACC}
    page = _official(tmp_path, "get_xbrl_facts", {**loss, "value": -1234000.0, "cik": OFF_CIK})
    assert (page["anchor"], page["quote"]) == ("f-3", "Net loss | $ | (1,234)")
    # The wrong sign matches no reported fact: the filing index page stays the official page.
    unsigned = _official(tmp_path, "get_xbrl_facts", {**loss, "value": 1234000.0, "cik": OFF_CIK})
    assert (unsigned["url"], unsigned["anchor"], unsigned["quote"]) == (OFF_INDEX, "", f"SEC Accession No. {OFF_ACC}")


def test_official_eps_row_and_shares_envelope(tmp_path: Path, sec_www: _FakeSec) -> None:
    eps: dict[str, object] = {
        "fiscal_period": "Q2",
        "eps_diluted": 1.08,
        "period_end": "2026-06-30",
        "accession": OFF_ACC,
        "cik": OFF_CIK,
    }
    page = _official(tmp_path, "get_fundamentals", eps, {"metric": "eps"})
    assert (page["anchor"], page["quote"]) == ("f-6", "Diluted | $ | 1.08")
    envelope: dict[str, object] = {
        "metric": "shares_outstanding",
        "shares_outstanding": 24300000000.0,
        "as_of": "2026-07-20",
        "accession": OFF_ACC,
        "cik": OFF_CIK,
    }
    shares = _official(tmp_path, "get_fundamentals", {}, envelope)
    assert shares["anchor"] == "f-5"
    assert shares["quote"] == "Shares outstanding as of July 20, 2026 was 24.3 billion."


def test_official_filing_row_is_index_page_fetched_once(tmp_path: Path, sec_www: _FakeSec) -> None:
    row: dict[str, object] = {"accession_no": OFF_ACC, "form": "10-Q", "filer_cik": OFF_CIK}
    first = _official(tmp_path, "list_sec_filings", row)
    second = _official(tmp_path, "list_sec_filings", row)
    assert (first["url"], first["media"], first["quote"]) == (OFF_INDEX, "html", f"SEC Accession No. {OFF_ACC}")
    assert second["payload"] == first["payload"]
    assert sec_www.calls == [OFF_INDEX]  # the archived capture serves every later read
    assert list((tmp_path / "raw" / "sec" / "official_page").iterdir())


def test_official_insider_row_is_rendered_form_line(tmp_path: Path, sec_www: _FakeSec) -> None:
    row: dict[str, object] = {
        "accession_no": F4_ACC,
        "issuer_cik": 1045810,
        "transaction_date": "2026-10-07",
        "shares": 500000.0,
        "price": 0.0,
        "holdings_after": 23671360.0,
    }
    page = _official(tmp_path, "get_insider_activity", row)
    assert page["url"] == f"{F4_DIR}/xslF345X06/form4.xml"
    assert page["quote"] == "Common | 10/07/2026 | G | 500,000 | D | $0 | 23,671,360 | I"


def test_official_search_hit_is_matched_document(tmp_path: Path, sec_www: _FakeSec) -> None:
    hit: dict[str, object] = {
        "accession_no": OFF_ACC,
        "filer_cik": OFF_CIK,
        "query": '"TOTAL revenue"',
        "snippet": None,
    }
    page = _official(tmp_path, "search_sec_filings", {**hit, "matched_document": "fake-10q.htm"})
    assert (page["url"], page["quote"]) == (OFF_PRIMARY, "Total revenue")
    # A document name that is not one plain file name never builds a URL.
    unsafe = _official(tmp_path, "search_sec_filings", {**hit, "matched_document": "../x.htm"})
    assert unsafe["url"] == OFF_INDEX


def test_official_identity_gaps(tmp_path: Path, sec_www: _FakeSec) -> None:
    assert documents.resolve_sec_official("get_valuation_metrics", {}, {"ticker": "AAPL"}, root=tmp_path) is None
    assert sec_www.calls == []
    # A CIK-only EPS/balance row carries a value the company page never states: no official page.
    assert (
        documents.resolve_sec_official(
            "get_fundamentals",
            {"eps_diluted": 1.08, "period_end": "2026-06-30"},
            {"metric": "eps", "ticker": "AAPL", "cik": OFF_CIK},
            root=tmp_path,
        )
        is None
    )
    assert sec_www.calls == []
    company = _official(tmp_path, "get_fundamentals", {}, {"metric": "overview", "name": "APPLE INC", "cik": OFF_CIK})
    assert (company["url"], company["quote"]) == (COMPANY_URL, "APPLE INC CIK#: 0000320193 (see all company filings)")
    # No party CIK: the accession's filer-agent prefix CIK still names the accession directory.
    with pytest.raises(OSError, match="fetch failed"):
        documents.resolve_sec_official("get_material_events", {"source_accessions": [F4_ACC]}, {}, root=tmp_path)
    assert sec_www.calls[-1] == f"https://www.sec.gov/Archives/edgar/data/1197647/000119764726000011/{F4_ACC}-index.htm"
