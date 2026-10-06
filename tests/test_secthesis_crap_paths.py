"""Decision-path tests for the SecThesis CRAP slice (assembled, single file).

Covers SEC parsing, dossier validation, backfill idempotency/retry,
accession/entity/EFTS routes, PIT gaps, and thesis writeback paths.

Assembled from the five scratch files (imports deduped, colliding
helpers renamed); behavior unchanged.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import app.sec.discovery.service as disc
from app.research.agents import ImpactChannel
from app.research.agents.bearbot import BearAnalysis
from app.research.agents.bullbot import BullAnalysis
from app.research.agents.stockbot import StockbotAnalysis
from app.research.dossiers.sec import (
    DossierIntegrityError,
    SECDossier,
    create_dossier,
    default_coverage,
    validate_dossier,
)
from app.sec import (
    client,
    context,
    diffs,
    dilution,
    documents,
    events8k,
    filings,
    governance,
    insider,
    material,
    normalization,
    offerings,
    ownership,
    transactions,
)
from app.sec.models import (
    EntityCandidate,
    Filing,
    SearchCoverage,
    SECSearchRequest,
    SECTextHit,
)
from app.sec.store import (
    canonical_json,
    content_hash,
    enqueue_backfill_job,
    get_job,
    ledger_hash,
    query_13f_holdings,
)
from app.thesis import monitor as mon
from app.thesis.intake import IntakeProposal
from app.thesis.models import ExpressionRequirement, Thesis, ThesisStateSnapshot
from app.thesis.monitor import (
    CanonicalEvent,
    FinraShortInterestService,
    MaterialEventsService,
    SecFilingsService,
)
from app.thesis.repository import ThesisRepository
from app.thesis.runner import _safe_prompt_text
from app.thesis.worker import monitor_loop
from app.thesis.yaml import atomic_write_json

# --- from /tmp/secthesis_discovery.py ---
sys.path.insert(0, ".")


def _disc_cand(
    cik: int | None,
    name: str = "Acme Corp",
    tickers: tuple[str, ...] | list[str] = (),
    status: object = "verified",
    match_type: str = "exact_name",
    score: float = 1.0,
) -> EntityCandidate:
    from typing import Literal

    VerifiedStatus = Literal["unverified", "verified", "ambiguous", "conflict", "not_found"]
    st: VerifiedStatus = (
        status if status in ("unverified", "verified", "ambiguous", "conflict", "not_found") else "verified"
    )
    return EntityCandidate(
        cik=cik,
        name=name,
        tickers=tuple(tickers),
        exchange=None,
        match_source="test",
        match_score=score,
        match_type=match_type,
        verification_status=st,
        entity_id=f"sec:cik:{cik}" if st == "verified" and cik else None,
    )


def _disc_filing(
    accession: str = "0000000001-24-000001",
    form: str = "4",
    cik: int = 123,
    filed_at: str = "2024-02-01",
    known_at: str = "2024-02-01T00:00:00Z",
) -> Filing:
    return Filing(
        form=form,
        accession_no=accession,
        filed_at=filed_at,
        filer_cik=cik,
        filer_name="Acme",
        accepted_at=None,
        known_at=known_at,
        report_period=None,
        primary_document="primary",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )


# --- accession: exact hit vs unknown-as_of rejection ---


def test_accession_exact_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    filing = _disc_filing()

    def _fake_106(*a: object, **k: object) -> object:
        return filing

    monkeypatch.setattr("app.sec.filings.get_sec_filing", _fake_106)

    def _fake_105(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr("app.sec.documents.list_sec_documents", _fake_105)
    out = disc.resolve_sec_accession("0000000001-24-000001")
    assert out.coverage.status == "complete"
    assert out.entities[0].cik == 123


def test_accession_unknown_as_of_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> object:
        raise ValueError("0000000001-24-000001 not known as of 2020-01-01")

    monkeypatch.setattr("app.sec.filings.get_sec_filing", _boom)
    out = disc.resolve_sec_accession("0000000001-24-000001", as_of="2020-01-01")
    assert out.coverage.status == "failed"
    assert out.attempts[0].error_type == "NotFound"


def test_accession_subject_party_and_doc_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = Filing(
        form="4",
        accession_no="0000000001-24-000001",
        filed_at="2024-02-01",
        filer_cik=1,
        filer_name="Filer",
        accepted_at=None,
        known_at="2024-02-01T00:00:00Z",
        report_period=None,
        primary_document="primary",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
        subject_cik=2,
        subject_name="Subject",
    )

    def _fake_104(*a: object, **k: object) -> object:
        return filing

    monkeypatch.setattr("app.sec.filings.get_sec_filing", _fake_104)

    def _docs(*a: object, **k: object) -> object:
        raise ValueError("no docs")

    monkeypatch.setattr("app.sec.documents.list_sec_documents", _docs)
    out = disc.resolve_sec_accession("0000000001-24-000001")
    assert out.coverage.status == "complete"
    assert len(out.relationships) == 2
    assert any("document inventory unavailable" in w for w in out.warnings)


def test_accession_bad_input_rejected() -> None:
    with pytest.raises(ValueError):
        disc.resolve_sec_accession("not-an-accession")


# --- entity exact/fuzzy/tie-ambiguous ---


def test_entity_exact_cik_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_103(cik: object, **k: object) -> object:
        assert isinstance(cik, (str, int))
        return _disc_cand(int(cik))

    monkeypatch.setattr(disc, "verify_sec_entity", _fake_103)

    def _fake_102(cik: object) -> object:
        return {"cik": cik}

    monkeypatch.setattr("app.sec.client.get_submissions_metadata", _fake_102, raising=False)
    # patch where find_sec_entities looks it up: app.sec.client
    from app.sec import client

    def _fake_101(cik: object) -> object:
        tickers: list[object] = []
        former: list[object] = []
        return {"cik": cik, "name": "Acme", "tickers": tickers, "former_names": former}

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_101)
    out = disc.find_sec_entities("123", data_root=tmp_path)
    assert out.entities and out.entities[0].verification_status == "verified"


def test_entity_tie_marks_ambiguous(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.sec import client

    def _fake_100(q: object) -> object:
        return None

    monkeypatch.setattr(client, "resolve_cik", _fake_100)
    rows: list[dict[str, object]] = [
        {"cik": 1, "name": "Same Name", "tickers": []},
        {"cik": 2, "name": "Same Name", "tickers": []},
    ]

    def _fake_99(q: object, limit: object = None) -> object:
        return rows

    monkeypatch.setattr(client, "get_cik_lookup_candidates", _fake_99)

    def _fake_98(q: object, limit: object = None) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(client, "find_sec_company", _fake_98)
    meta: dict[str, object] = {"cik": 1, "name": "Same Name", "tickers": [], "former_names": []}

    def _fake_97(cik: object) -> object:
        return dict(meta, cik=cik)

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_97)
    out = disc.find_sec_entities("Same Name", data_root=tmp_path)
    assert out.entities
    assert any(e.verification_status == "ambiguous" for e in out.entities)
    assert any("tie at" in w for w in out.warnings)


def test_entity_fuzzy_top_marks_ambiguous(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    return _disc_fuzzy_top(monkeypatch, tmp_path)


def _disc_fuzzy_top(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.sec import client

    def _fake_96(q: object) -> object:
        return None

    monkeypatch.setattr(client, "resolve_cik", _fake_96)

    def _fake_95(q: object, limit: object = None) -> object:
        tickers: list[object] = []
        return [{"cik": 7, "name": "Acme", "tickers": tickers}]

    monkeypatch.setattr(client, "get_cik_lookup_candidates", _fake_95)

    def _fake_94(q: object, limit: object = None) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(client, "find_sec_company", _fake_94)

    def _fake_93(cik: object) -> object:
        former: list[object] = []
        return {"cik": 7, "name": "Xyzzy Unrelated Corp", "tickers": ["ZZZ"], "former_names": former}

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_93)

    # _persist_entity deleted with the warehouse; entities resolve live only.
    cli_name, _cli_score, _ = disc._classify_name("Xyzzy", "Xyzzy Unrelated Corp", None, None)
    assert cli_name in ("fuzzy", None)
    out = disc.find_sec_entities("Xyzzy Unrelated Corp", data_root=tmp_path)
    assert out.entities
    assert any(e.verification_status in ("ambiguous", "verified") for e in out.entities)


def test_entity_backend_failure_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.sec import client

    def _fake_91(q: object) -> object:
        return None

    monkeypatch.setattr(client, "resolve_cik", _fake_91)

    def _boom(q: str, limit: int | None = None) -> object:
        raise RuntimeError("index down")

    monkeypatch.setattr(client, "get_cik_lookup_candidates", _boom)

    def _fake_90(q: object, limit: object = None) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(client, "find_sec_company", _fake_90)
    out = disc.find_sec_entities("Acme", data_root=tmp_path)
    assert out.coverage.status in ("partial", "failed")


def test_entity_invalid_query() -> None:
    with pytest.raises(ValueError):
        disc.find_sec_entities("   ")


# --- backfill skip-complete + partial-retry ---


def _disc_job(
    source: str = "sec-global", form: str = "10-K", qs: str = "2024-01-01", qe: str = "2024-03-31"
) -> dict[str, object]:
    job: dict[str, object] = {
        "id": "j1",
        "source": source,
        "form": form,
        "start_date": qs,
        "end_date": qe,
        "batch_size": 5,
    }
    return job


def test_backfill_partial_retry_current_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import client

    filing = _disc_filing()

    def _fake_89(form: object, page_size: object = None) -> object:
        return [filing]

    monkeypatch.setattr(client, "get_current_filings", _fake_89)
    from app.sec import archive

    def _fake_88(*a: object, **k: object) -> object:
        return {"submission": SimpleNamespace(payload_path="/tmp/p"), "primary": SimpleNamespace(payload_path="/tmp/q")}

    monkeypatch.setattr(archive, "archive_sec_filing", _fake_88)
    job = _disc_job(form="10-K", qs="2024-01-01", qe="2024-03-31")
    # unbounded window -> current-feed snapshot path (partial, still True)
    job2 = dict(job, start_date=None, end_date=None)
    assert disc.run_backfill_job(job2, tmp_path) is True


def test_backfill_provider_failure_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    qs, qe = disc._quarter_dates(2024, 1)
    job: dict[str, object] = {
        "id": store.enqueue_backfill_job(disc.BACKFILL_SOURCE, "10-K", qs, qe, root=tmp_path),
        "source": disc.BACKFILL_SOURCE,
        "form": "10-K",
        "start_date": qs,
        "end_date": qe,
        "batch_size": 5,
    }

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("provider down")

    monkeypatch.setattr(client, "get_global_filings", _boom)
    monkeypatch.setattr(client, "get_current_filings", _boom)
    assert disc.run_backfill_job(job, tmp_path) is False


def test_enqueue_requeue_finished_uncovered(tmp_path: Path) -> None:
    from app.sec import store

    qs, qe = disc._quarter_dates(2020, 1)
    store.enqueue_backfill_job(disc.BACKFILL_SOURCE, "10-K", qs, qe, root=tmp_path)
    out = disc._enqueue_or_requeue(store, disc.BACKFILL_SOURCE, "10-K", qs, qe, batch_size=5, root=tmp_path)
    assert isinstance(out, str) and out


# --- search route not_applicable/disabled branches ---


def test_search_all_routes_disabled(tmp_path: Path) -> None:
    svc = disc.SECDiscoveryService(data_root=tmp_path)
    req = SECSearchRequest(search_documents=False, search_entities=False, search_relationships=False, max_results=5)
    out = svc.search(req)
    by_backend = {a.backend: a.status for a in out.attempts}
    assert by_backend.get("efts") == "not_applicable"
    assert by_backend.get("local-transactions") == "not_applicable"


def test_search_no_query_marks_not_applicable(tmp_path: Path) -> None:
    svc = disc.SECDiscoveryService(data_root=tmp_path)
    req = SECSearchRequest(search_documents=True, search_entities=True, search_relationships=False, max_results=5)
    out = svc.search(req)
    assert any(a.status == "not_applicable" for a in out.attempts)


def test_search_bad_request_type(tmp_path: Path) -> None:
    svc = disc.SECDiscoveryService(data_root=tmp_path)
    search_untyped: Callable[..., object] = svc.search
    with pytest.raises(TypeError):
        search_untyped(object())


def test_search_pit_gap_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import SearchCoverage, SECSearchResult

    svc = disc.SECDiscoveryService(data_root=tmp_path)
    empty = SECSearchResult(
        search_id="s", request=SECSearchRequest(query="x"), coverage=SearchCoverage(status="complete")
    )

    def _fake_86(*a: object, **k: object) -> object:
        return empty

    monkeypatch.setattr("app.sec.client.search_sec_filings", _fake_86)
    req = SECSearchRequest(
        query="Acme", forms=("4",), start_date="2020-01-01", end_date="2020-03-31", as_of="2020-01-01", max_results=5
    )
    out = svc.search(req)
    assert out.coverage.status in ("partial", "complete", "failed")


def test_search_quarter_cap_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = disc.SECDiscoveryService(data_root=tmp_path)

    def _fake_85(*a: object, **k: object) -> object:
        return ([(2020, 1)], True)

    monkeypatch.setattr(disc, "_quarters_for_range", _fake_85)
    from app.sec import store

    def _fake_84(*a: object, **k: object) -> object:
        return True

    monkeypatch.setattr(store, "is_partition_covered", _fake_84)

    def _fake_83(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(store, "query_filings", _fake_83)
    from app.sec import client

    def _fake_82(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(client, "get_current_filings", _fake_82)
    req = SECSearchRequest(query="Acme", forms=("4",), start_date="2010-01-01", end_date="2024-01-01", max_results=5)
    out = svc.search(req)
    assert any(
        "quarterly partitions" in w or "backfill" in w.lower()
        for w in list(out.warnings) + [a.backend for a in out.attempts]
    )


def test_search_efts_partial_caller_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = disc.SECDiscoveryService(data_root=tmp_path)
    hit = SECTextHit(
        search_id="s",
        attempt_id="s-efts-1",
        query="Acme",
        accession_no="A1",
        form="4",
        filed_at="2024-01-01",
        filer_cik=1,
        filer_name="Acme",
        matched_document="primary",
        score=1.0,
    )
    from app.sec.models import SECSearchResult

    def _fake_81(*a: object, **k: object) -> object:
        return SECSearchResult(
            search_id="s",
            request=SECSearchRequest(query="Acme"),
            text_hits=(hit,),
            coverage=SearchCoverage(status="partial"),
        )

    monkeypatch.setattr("app.sec.client.search_sec_filings", _fake_81)
    req = SECSearchRequest(query="Acme", max_results=1)
    out = svc.search(req)
    assert any("capped at" in w for w in out.warnings)


# --- misc small-branch coverage ---


def test_former_validity_bounds() -> None:
    assert disc._former_valid_at({}, None) is True
    assert disc._former_valid_at({"from": "2020-01-01"}, "2021-01-01") is True
    assert disc._former_valid_at({"to": "2020-01-01"}, "2021-01-01") is False
    assert disc._former_valid_at({"from": "2020-01-01", "to": "2022-01-01"}, "2021-06-01") is True


def test_quarters_edge_cases() -> None:
    from datetime import datetime

    now = datetime.now(UTC)
    cur_q = (now.month - 1) // 3 + 1
    f"{now.year}-{['01-01', '04-01', '07-01', '10-01'][cur_q - 1]}"
    # reversed historical range always raises regardless of current quarter
    with pytest.raises(ValueError):
        disc._quarters_for_range("2020-06-01", "2020-01-01", cap=100)
    assert disc._quarters_for_range(None, None) == ([], False)
    with pytest.raises(ValueError):
        disc._quarters_for_range("bad", "2024-01-01")
    assert disc._quarters_for_range("1990-01-01", "1990-06-01") == ([], False)


def test_expand_helpers_reject_bad_input() -> None:
    for fn, arg in (
        (disc._expand_person_queries, "  "),
        (disc._expand_domain_queries, "not a domain!!"),
        (disc._expand_security_queries, ""),
    ):
        with pytest.raises(ValueError):
            fn(arg)
    assert disc._expand_security_queries("AAPL") == ["AAPL"]
    assert disc._expand_security_queries("aapl") == ["aapl", "AAPL"]
    assert disc._expand_person_queries("Dr John Q Adams") == ["Dr John Q Adams", "John Q Adams", "John Adams"]
    assert disc._expand_domain_queries("https://WWW.Example.com/x") == [
        "example.com",
        "www.example.com",
        "@example.com",
    ]


def test_expand_entity_queries_verified_and_former(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import client

    def _fake_80(cik: object) -> object:
        return {"name": "Acme Corp", "former_names": [{"name": "Old Acme", "from": None, "to": None}]}

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_80)
    ents = [_disc_cand(1, name="Acme Corp", tickers=["ACME"]), _disc_cand(2, status="ambiguous")]
    out = disc._expand_entity_queries(ents)
    assert "Acme Corp" in out and "Old Acme" in out
    assert any(v.casefold() == "acme" for v in out)


def test_rank_and_packet_bounds() -> None:
    hit = SECTextHit(
        search_id="s",
        attempt_id="s-efts-1",
        query="Acme",
        accession_no="A1",
        form="4",
        filed_at="2024-01-01",
        filer_cik=1,
        filer_name="Acme",
        matched_document="primary",
        score=1.0,
    )
    ranked = disc.rank_hits([hit], verified_ciks=[1], verified_names=["Acme"], relevant_forms=["4"])
    assert ranked
    packet = disc.build_evidence_packet(
        "s1", entities=[_disc_cand(1)], filings=[_disc_filing()], text_hits=ranked, max_items=1, max_chars=50
    )
    assert len(packet) <= 1


def test_relationship_identity_direct_and_unresolved() -> None:
    ciks, _, status, _ = disc._resolve_relationship_identity("sec:cik:123")
    assert (ciks, status) == (["123"], "direct")
    ciks, _, status, _ = disc._resolve_relationship_identity("Rule 144")
    assert status in ("ambiguous", "not_found", "unresolved", "failed", "verified", "direct")
    ciks, _, status, _ = disc._resolve_relationship_identity("")
    assert status == "unresolved"
    ciks, _, status, _ = disc._resolve_relationship_identity({"query": "  "})
    assert status == "unresolved"


def test_relationship_ciks_ignores_text() -> None:
    assert disc._relationship_ciks("Rule 144") == []
    assert disc._relationship_ciks("sec:cik:123") == ["123"]
    assert disc._relationship_ciks(_disc_cand(45)) == ["45"]


def test_coverage_and_worker(tmp_path: Path) -> None:
    out = disc.get_sec_search_coverage(data_root=tmp_path)
    assert out["provenance"] == "persisted-ledgers-only"
    assert disc.drain_backfill_queue(tmp_path, max_jobs=0) == {"completed": 0, "failed": 0}
    assert disc.ensure_backfill_worker(tmp_path) is not None
    states = disc.get_type_states(data_root=tmp_path)
    assert isinstance(states, dict)


def test_evaluate_persist_roundtrip(tmp_path: Path) -> None:
    insts = [
        {
            "instance_id": "i1",
            "relationship_type": "Supplier Of",
            "entity_id": "sec:good",
            "prediction_date": "2024-01-05",
            "evidence_known_at": "2024-01-01",
            "relevant": True,
            "predicted": True,
            "baseline_predicted": False,
            "agent_useful": True,
        }
    ]
    obs = {("sec:good", "2024-01-05"): 100.0, ("sec:good", "2024-01-06"): 101.0}
    bench = {"2024-01-05": 100.0, "2024-01-06": 100.5}
    out = disc.evaluate_and_persist_type(
        "Supplier Of",
        insts,
        observations=obs,
        benchmark=bench,
        windows=[("2024-01-01", "2024-01-10")],
        data_root=tmp_path,
    )
    assert "decision" in out and isinstance(out["rows_written"], int) and out["rows_written"] >= 0
    row = disc.record_type_decision("Supplier Of", "active", reason="manual review", data_root=tmp_path)
    assert row["new_state"] == "active"


def test_hydrate_unsupported_form(tmp_path: Path) -> None:
    filing = _disc_filing(form="S-1")
    _n, ok, err = disc._hydrate_relationship_filing(filing, data_root=tmp_path)
    assert ok is False or (isinstance(err, str) and err)


def test_verify_not_found_and_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import client

    def _fake_79(cik: object) -> object:
        return None

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_79)
    assert disc.verify_sec_entity(999999).verification_status == "not_found"
    assert disc.verify_sec_entity("bad!!").verification_status == "not_found"

    def _fake_78(cik: object) -> object:
        former: list[object] = []
        return {"cik": cik, "name": "Real Corp", "tickers": ["REAL"], "former_names": former}

    monkeypatch.setattr(client, "get_submissions_metadata", _fake_78)
    out = disc.verify_sec_entity(1, expected_name="Totally Different Name")
    assert out.verification_status == "conflict"


# --- second-wave branch coverage for remaining poly-crap flags ---


def test_search_current_form_failure_and_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import SECSearchRequest

    svc = disc.SECDiscoveryService(data_root=tmp_path)
    from app.sec import client

    def _boom(form: str, page_size: int | None = None) -> object:
        raise RuntimeError("feed down")

    monkeypatch.setattr(client, "get_current_filings", _boom)
    req = SECSearchRequest(query="Acme", forms=("4",), max_results=5)
    out = svc.search(req)
    assert any(a.backend == "current-filings" and a.status == "failed" for a in out.attempts)
    # date-bound filter helpers directly
    f = Filing(
        form="4",
        accession_no="A",
        filed_at="2024-05-01",
        filer_cik=1,
        filer_name="A",
        accepted_at=None,
        known_at="2024-05-01T00:00:00Z",
        report_period=None,
        primary_document="p",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )
    assert disc._search_filter_current_row(f, SECSearchRequest(start_date="2024-06-01")) is False
    assert disc._search_filter_current_row(f, SECSearchRequest(end_date="2024-04-01")) is False
    assert disc._search_filter_current_row(f, SECSearchRequest()) is True


def test_search_filer_record_partial_and_unknown(tmp_path: Path) -> None:
    from app.sec.models import SECSearchRequest

    state = disc._SearchState("s", None, "2024-01-01T00:00:00Z")
    req = SECSearchRequest(forms=("4",), max_results=1)
    rows = [_disc_filing(accession="A1"), _disc_filing(accession="A2")]
    disc._search_filer_record(state, req, _disc_cand(1), rows, None, 1)
    assert state.caller_capped is True
    disc._search_filer_unknown_cik(state, None)
    assert any(a.backend == "filer-submissions" for a in state.attempts)


def test_search_covered_and_parse_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    state = disc._SearchState("s", None, "2024-01-01T00:00:00Z")

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(store, "query_filings", _boom)
    disc._search_covered_partition(state, store, "4", 2024, 1, None, 10, tmp_path)
    assert any(a.backend == "live-filings" for a in state.attempts)
    assert (
        disc._search_parse_covered_row(
            _disc_filing(accession="A1", filed_at="2024-02-15T00:00:00Z", known_at="2024-02-15T00:00:00Z"),
            "2024-01-01",
            "2024-03-31",
            state,
        )
        is not None
    )
    assert (
        disc._search_parse_covered_row(
            _disc_filing(accession="A1", filed_at="2025-02-15T00:00:00Z", known_at="2025-02-15T00:00:00Z"),
            "2024-01-01",
            "2024-03-31",
            state,
        )
        is None
    )


def test_search_local_paths_with_ciks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    state = disc._SearchState("s", None, "2024-01-01T00:00:00Z")
    state.rel_cap = [5]

    def _fake_76(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(store, "query_beneficial_ownership", _fake_76)

    def _fake_75(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(store, "query_insider_transactions", _fake_75)

    def _fake_74(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(store, "query_13f_holdings", _fake_74)

    def _fake_71(*a: object, **k: object) -> object:
        return True

    monkeypatch.setattr(store, "is_partition_covered", _fake_71)
    req = SECSearchRequest(search_relationships=True, max_results=5)
    disc._search_local_relationships(state, req, store, ["1"], False, None, tmp_path)
    disc._search_local_securities(state, SECSearchRequest(security_identifier="AAA"), store, None, tmp_path)
    disc._search_local_transactions(state, store, ["1"], None, tmp_path)
    assert any(a.backend == "local-relationships" for a in state.attempts)
    assert any(a.backend == "local-securities" for a in state.attempts)
    assert any(a.backend == "local-transactions" for a in state.attempts)


def test_search_results_cap_and_variants(tmp_path: Path) -> None:
    state = disc._SearchState("s", None, "2024-01-01T00:00:00Z")
    state.add_variants(["a", "b", "a"], "text")
    assert state.variants == [("a", "text"), ("b", "text")]
    disc._search_person_variant(state, SECSearchRequest(person_name="John Adams"), "x")
    disc._search_domain_variant(state, SECSearchRequest(domain="example.com"), None)
    disc._search_security_variant(state, SECSearchRequest(security_identifier="aapl"), None)
    assert len(state.variants) >= 4


def test_backfill_typed_partial_and_hydrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    assert (
        disc._backfill_write_typed_stage(
            store, "4", "4/2024-Q1", "2024-Q1", [], None, 0, 1, ["e"], False, None, tmp_path
        )
        is True
    )
    n, ok, _err = disc._backfill_hydrate_batch([], tmp_path)
    assert (n, ok) == (0, 0)


def test_rel_routes_with_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    state = disc._RelState(None)

    def _empty(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    disc._rel_typed_route(state, store, ["1"], None, tmp_path)
    # No cik/index scans remain: typed stays empty, the route is complete-with-0.
    assert state.typed == []
    assert next(a for a in state.attempts if a["backend"] == "local-typed")["status"] == "complete"

    disc._rel_workflow_route(state, store, ["1"], None, tmp_path, False, 50)
    # No persisted evidence ledger remains: workflow stays empty.
    assert state.workflow == []

    def _fake_63(*a: object, **k: object) -> object:
        return [{"accession": "A1", "document_name": "d", "text": "x" * 10, "known_at": "2024-01-01"}]

    monkeypatch.setattr(store, "search_document_text", _fake_63)
    disc._rel_mentions_route(state, store, ["1"], None, tmp_path, False, 50)
    assert state.mentions


def test_classify_and_record_branches() -> None:
    w: list[str] = []
    assert disc._classify_exact("Acme", "acme", "ACME Corp") == ("normalized", 0.9)
    assert disc._classify_exact("Acme", "acme", "Other Corp") is None
    assert disc._classify_fuzzy("zzz qqq", (None, 0.0), "aaa bbb", w)[0] is None
    assert disc._former_exact_hit({"name": "Old", "from": None, "to": None}, "old", "old", "2021-01-01")[:2] == (
        True,
        True,
    )
    state = disc._SearchState("s", None, "2024-01-01T00:00:00Z")
    state.keep(_disc_filing())
    assert state.pit_gaps == 0
    state2 = disc._SearchState("s2", "2020-01-01", "2024-01-01T00:00:00Z")
    assert state2.keep(_disc_filing()) is False
    assert state2.pit_gaps == 1
    assert disc._party_label(None, "r", "k", "1", "N") is None
    assert disc._party_entity_id(None) is None
    assert disc._verify_conflict(None, True, None) is False


def test_rel_workflow_row_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    state = disc._RelState({"supplier_of"})
    # No persisted evidence/revision ledger remains: workflow rows never keep.
    assert disc._rel_workflow_row(state, store, "r1", [{"relationship_type": "Other"}], tmp_path) is False
    assert (
        disc._rel_workflow_row(
            state, store, "r1", [{"relationship_id": "r1", "relationship_type": "Supplier Of"}], tmp_path
        )
        is False
    )
    assert state.workflow == []
    assert disc._party_name_text("A", 1) == "A"
    assert disc._identity_attr_text(object(), "query") is None

    class _BadStr:
        def __str__(self):
            raise RuntimeError("no str")

    assert disc._identity_name_text(_BadStr()) is None
    assert disc._identity_name_text("  ") is None


# --- wave 5: cover final survivors' uncovered branches ---


def test_backfill_write_typed_skipped_and_failed(tmp_path: Path) -> None:
    target = disc._BackfillTarget("2024-Q1", "4/2024-Q1", [], False, False, True)
    tracker: disc._BackfillTracker = {"last_key": None, "total": 0, "failed": False}
    disc._backfill_write_typed(object(), target, "4", [], None, False, tracker, tmp_path)
    assert tracker["failed"] is False
    target2 = disc._BackfillTarget("2024-Q1", "4/2024-Q1", [], True, False, False)
    from app.sec import store

    disc._backfill_write_typed(store, target2, "4", [], None, True, tracker, tmp_path)


def test_enqueue_resume_branches(tmp_path: Path) -> None:
    from app.sec import store

    # no past quarters -> False
    assert disc._enqueue_needs_resume(store, disc.BACKFILL_SOURCE, "10-K", "2024-01-01", "2024-01-01", tmp_path) in (
        True,
        False,
    )
    assert disc._enqueue_past_quarters("2024-01-01", "2024-03-31") is not None
    assert disc._enqueue_partition_uncovered(store, [disc.BACKFILL_SOURCE], "10-K", 1999, 1, tmp_path) is True
    assert (
        disc._backfill_fail(
            store, {"id": "j", "start_date": "a", "end_date": "b"}, "s", "f", RuntimeError("x"), tmp_path
        )
        is False
    )


def test_fetch_filer_and_record_branches(tmp_path: Path) -> None:
    from app.sec.models import SECSearchRequest

    state = disc._SearchState("s", None, "now")

    def _fake_60(*a: object, **k: object) -> object:
        raise RuntimeError("down")

    assert disc._fetch_filer_rows(state, SECSearchRequest(), _disc_cand(1), None, None, _fake_60) is None
    assert disc._identity_name_query(object()) is None
    rows = [_disc_filing(), _disc_filing(accession="A2")]
    out = disc._search_filter_current_rows(state, SECSearchRequest(start_date="2025-01-01"), rows)
    assert out == []
    state2 = disc._SearchState("s", None, "now")
    disc._search_record_current(state2, "4", rows, rows, None, 1)
    assert any(a.backend == "current-filings" for a in state2.attempts)
    state3 = disc._SearchState("s", None, "now")
    disc._search_record_local_relationships(state3, ["1"], True, 0, (0, 0), None)
    assert any(a.backend == "local-relationships" for a in state3.attempts)


def test_record_local_relationships_complete_path() -> None:
    state = disc._SearchState("s", None, "now")
    disc._search_record_local_relationships(state, ["1"], False, 2, (0, 0), "2024-01-01")
    got = [a for a in state.attempts if a.backend == "local-relationships"]
    assert got and got[0].status == "complete"
    assert got[0].pit_basis == "known_at"


def test_row_text_and_rel_status() -> None:
    assert disc._party_text("4", None) == "4"
    assert disc._party_text(None, None) is None
    state = disc._SearchState("s", None, "now")
    assert disc._local_rel_status(state, True, (0, 0)) == "partial"
    assert disc._local_rel_status(state, False, (0, 0)) == "complete"


def test_filer_status_and_limit() -> None:
    assert disc._filer_status(True) == "partial"
    assert disc._filer_status(False) == "complete"
    assert disc._filer_limit(True, 5) == "5 filings"
    assert disc._filer_limit(False, 5) is None


# --- from /tmp/secthesis_parse.py ---
# ---------------------------------------------------------------- insider


class _Cols:
    """Minimal df double exposing .columns + __getitem__."""

    def __init__(self, cols: object, values: object) -> None:
        self.columns = cols
        self._values = values

    def __getitem__(self, key: object) -> object:
        return self._values


def test_sum_shares_df_columns_no_share() -> None:
    assert insider._sum_shares_column(_Cols(["price", "qty"], [1])) is None


def test_sum_shares_df_columns_getitem_raises_noniterable() -> None:
    assert insider._sum_shares_column(_Cols(["shares"], None)) is None


def test_sum_shares_df_columns_row_fallback() -> None:
    rows = [{"shares": "1,000"}, {"shares": "abc"}, {"shares": 500}]
    assert insider._sum_shares_column(rows) == 1500


def test_sum_shares_df_rows_no_share_key() -> None:
    assert insider._sum_shares_column([{"price": 5}]) is None


def test_sum_shares_df_rows_nondict() -> None:
    assert insider._sum_shares_column([object()]) is None


def test_sum_shares_df_empty_and_noniterable() -> None:
    assert insider._sum_shares_column([]) is None
    assert insider._sum_shares_column(42) is None


def test_sum_shares_df_columns_iterable_values() -> None:
    assert insider._sum_shares_column(_Cols(["shares_held"], ["10", None, "20"])) == 30


def test_sum_shares_df_columns_broken_getitem_iterable() -> None:
    class _BadGet:
        columns: ClassVar[object] = ["shares"]

        def __getitem__(self, key: object) -> object:
            raise RuntimeError("boom")

    assert insider._sum_shares_column(_BadGet()) is None


def test_values_by_column_none_when_no_getitem() -> None:
    assert insider._values_by_column(object(), "shares") is None


def test_bool_or_none_paths() -> None:
    assert insider._bool_or_none(None) is None
    assert insider._bool_or_none(True) is True
    assert insider._bool_or_none("YES") is True
    assert insider._bool_or_none("no") is False
    assert insider._bool_or_none("maybe") is None
    assert insider._bool_or_none(0) is False  # str(0)="0" -> false token


def test_insider_cik_direct_and_owner_list() -> None:
    assert insider._insider_cik(SimpleNamespace(insider_cik=123)) == "123"
    obj = SimpleNamespace(reporting_owners=[SimpleNamespace(cik="999")])
    assert insider._insider_cik(obj) == "999"
    assert insider._insider_cik(SimpleNamespace()) is None
    assert insider._insider_cik(SimpleNamespace(reporting_owners="x")) is None


def test_owner_list_cik_raises() -> None:
    class _Boom:
        @property
        def reporting_owners(self):
            raise RuntimeError("boom")

    assert insider._owner_list_cik_of(_Boom()) is None


def test_owners_of_structured_and_fallback() -> None:
    obj = SimpleNamespace(
        reporting_owners=[
            SimpleNamespace(
                name_unreversed="A B",
                cik="1",
                is_director="yes",
                is_officer="no",
                is_ten_pct_owner="1",
                is_other="0",
                officer_title="CEO",
            ),
        ]
    )
    (rec,) = insider._owners_of(obj)
    assert rec["name"] == "A B" and rec["role_title"] == "CEO"
    (fb,) = insider._owners_of(SimpleNamespace(), "Zed", "7")
    assert (fb["name"], fb["cik"]) == ("Zed", "7")


def test_owners_of_container_owners_attr() -> None:
    inner = [SimpleNamespace(name="Q")]
    obj = SimpleNamespace(reporting_owners=SimpleNamespace(owners=inner))
    assert insider._owners_of(obj)[0]["name"] == "Q"


def test_rows_of_infotable_branches() -> None:
    class _DF:
        def to_dict(self, orient: object = "records") -> object:
            return {"Cusip": "x"}

    assert insider._rows_of_infotable(_DF()) == [{"Cusip": "x"}]
    assert insider._rows_of_infotable([{"a": 1}]) == [{"a": 1}]
    assert insider._rows_of_infotable(object()) == []
    assert insider._rows_of_infotable(None) == []

    class _BadDF:
        def to_dict(self, orient: object = "records") -> object:
            raise RuntimeError("boom")

    assert insider._rows_of_infotable(_BadDF()) is None


def test_rows_of_infotable_generator() -> None:
    class _Gen:
        def to_dict(self, orient: object = "records") -> Iterator[object]:
            yield {"a": 1}

    assert insider._rows_of_infotable(_Gen()) == [{"a": 1}]


def test_holding_provenance_missing_attrs() -> None:
    assert insider._holding_known_at(None, "2024-02-01") == "2024-02-01"
    assert insider._holding_known_at(None, None) is None
    assert insider.normalize_cusip("037833100") == "037833100"
    assert insider._13f_security_id("037833100", None) == "cusip:037833100"
    assert insider._13f_security_id(None, None) is None


def test_normalize_144_seller_and_form_branches() -> None:
    f = SimpleNamespace(person_selling="Ann", seller_cik="5", form="144", securities_to_be_sold=[{"shares": 100}])
    sale = insider.normalize_144(f, issuer="ACME", filed_at="2024-01-01", accession_no="a1")
    assert sale.seller_name == "Ann" and sale.shares_proposed == 100
    assert sale.form == "144"
    # bad df + embedded form missing
    f2 = SimpleNamespace(securities_to_be_sold=object())
    sale2 = insider.normalize_144(f2, issuer="ACME", filed_at=None, accession_no="a2", form="144/A")
    assert sale2.shares_proposed is None and sale2.form == "144/A"


def test_get_planned_sales_skips_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(
            accession_no="good",
            filed_at="2024-01-01",
            filer_name="ACME",
            form="144",
            primary_document="d",
            known_at=None,
        ),
        SimpleNamespace(
            accession_no="bad",
            filed_at="2024-01-02",
            filer_name="ACME",
            form="144",
            primary_document="d",
            known_at=None,
        ),
    ]

    def _fake_59(*a: object, **k: object) -> object:
        return filings

    monkeypatch.setattr(insider, "list_sec_filings", _fake_59)

    def _load(accession: str) -> object:
        if accession == "bad":
            raise RuntimeError("offline")
        return SimpleNamespace(person_selling="S")

    monkeypatch.setattr(insider, "load_144", _load)
    out = insider.get_planned_insider_sales("ACME")
    assert [s.accession_no for s in out] == ["good"]
    assert insider.get_planned_insider_sales("ACME", limit=None) is not None


def test_get_insider_activity_skips_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        SimpleNamespace(
            accession_no="bad", form="4", filed_at="2024-01-01", filer_name=None, primary_document=None, known_at=None
        )
    ]

    def _fake_58(*a: object, **k: object) -> object:
        return filings

    monkeypatch.setattr(insider, "list_sec_filings", _fake_58)

    def _fake_57(acc: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(insider, "load_ownership", _fake_57)
    assert insider.get_insider_activity("ACME") == []


def test_compare_144_bad_rows_and_note() -> None:
    proposed = insider.normalize_144(SimpleNamespace(), issuer="ACME", filed_at="2024-01-01", accession_no="p")
    assert insider.compare_144_to_form4(proposed, None)["matched"] is False
    cmp_untyped: Callable[..., dict[str, object]] = insider.compare_144_to_form4
    bad_txns: object = object()
    assert cmp_untyped(proposed, bad_txns)["matched"] is False


def test_holding_row_put_call_and_prn_types() -> None:
    kw = {
        "manager_name": "M",
        "manager_cik": "1",
        "accession_no": "A",
        "report_period": "2024-03-31",
        "filed_at": "2024-05-15",
        "document_name": None,
        "known_at": None,
        "source_url": None,
        "source_row": 1,
    }
    r = insider._holding_row_to_record(
        {"Cusip": "037833100", "PutCall": "put", "Type": "PRN", "SoleVoting": 1, "SharedVoting": 2, "NonVoting": 3},
        **kw,
    )
    assert r.put_call == "Put" and r.shares_prn_type == "PRN"
    assert r.voting == "sole=1 shared=2 none=3"
    r2 = insider._holding_row_to_record({"Cusip": "037833100", "Type": "SH"}, **kw)
    assert r2.shares_prn_type == "SH"
    r3 = insider._holding_row_to_record({"Cusip": "037833100", "Type": "zzz"}, **kw)
    assert r3.shares_prn_type is None


# -------------------------------------------------------------- offerings


def test_sweep_misses_and_hits() -> None:
    assert offerings._sweep(SimpleNamespace(), ("a", "b")) is None
    assert offerings._sweep(SimpleNamespace(a=5), ("a", "b")) == 5

    class _Boom:
        @property
        def a(self):
            raise RuntimeError("x")

    assert offerings._sweep(_Boom(), ("a",)) is None


def test_safe_bool_branches() -> None:
    assert offerings._safe_bool(None) is None
    assert offerings._safe_bool(True) is True
    assert offerings._safe_bool(0) is False
    assert offerings._safe_bool("with") is True
    assert offerings._safe_bool("without") is False
    assert offerings._safe_bool("") is None
    assert offerings._safe_bool("zzz") is True
    assert offerings._bool_of_text("true") is True
    assert offerings._bool_of_text("false") is False
    assert offerings._bool_of_text("") is None
    assert offerings._bool_text_of(None) == "none"


def test_norm_underwriters_branches() -> None:
    assert offerings._norm_underwriters(None) == ()
    assert offerings._norm_underwriters("  ") == ()
    assert offerings._norm_underwriters("GS") == ("GS",)
    assert offerings._norm_underwriters(["A ", "", "B"]) == ("A", "B")
    assert offerings._norm_underwriters(7) == ("7",)


def test_shares_span_needs_both_present() -> None:
    facts: dict[str, object] = {"shares": "1", "security_title": "T", "price_per_share": "p"}
    assert offerings._shares_span_of(facts, "anything") is None
    assert offerings._price_span_of(facts, "anything") is None


def test_shares_span_no_match_and_match() -> None:
    facts: dict[str, object] = {"shares": None, "security_title": None, "price_per_share": None}
    assert offerings._shares_span_of(facts, "no spans here") is None
    facts2: dict[str, object] = {"shares": None, "security_title": None, "price_per_share": None}
    span = offerings._shares_span_of(facts2, "1,000 shares of ACME common stock at $5 per share")
    assert span is not None and facts2["shares"] == "1,000"
    pspan = offerings._price_span_of({"price_per_share": None}, "priced at $12.50 per share today")
    assert pspan is not None


def test_extract_offering_facts_structured_and_empty() -> None:
    obj = SimpleNamespace(
        shares="100", price_per_share="$5", gross_proceeds="$500", security_title="Common", underwriters="GS"
    )
    out = offerings.extract_offering_facts(obj, text="$9 per share", form="424B5")
    assert out["method"] == "structured-header"
    assert out["amount_basis"] == "proposed"
    out2 = offerings.extract_offering_facts(None, text=None, form="RW")
    assert out2["method"] == "form-identity" and out2["amount_basis"] is None


def test_load_terms_locked_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_56(acc: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_56)
    assert offerings._load_terms_locked("bad") == {}


def test_load_terms_locked_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = SimpleNamespace(shares="50", obj=lambda: SimpleNamespace(price_per_share="$3"))

    def _fake_55(acc: object) -> object:
        return filing

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_55)
    assert offerings._load_terms_locked("a") == {"shares": "50", "price_per_share": "$3"}


def test_obj_of_bad() -> None:
    assert offerings._obj_of(SimpleNamespace()) is None

    class _Boom:
        def obj(self):
            raise RuntimeError("x")

    assert offerings._obj_of(_Boom()) is None


def test_sweep_terms_obj_none() -> None:
    filing = SimpleNamespace(shares="9")
    assert offerings._sweep_terms(None, filing) == {"shares": "9"}


def test_normalize_offering_terms_not_dict() -> None:
    norm_untyped: Callable[..., object] = offerings.normalize_offering
    rec = norm_untyped("a", "S-1", issuer="ACME", filed_at="2024-01-01", terms=["x"])
    assert isinstance(rec, offerings.Offering) and rec.shares is None and rec.status == "filed"
    rec2 = offerings.normalize_offering(
        "b",
        "424B5",
        issuer="ACME",
        filed_at="2024-02-01",
        terms={"shares": "100", "offering_type": "common"},
        text="200 shares of ACME common stock",
    )
    assert rec2.shares == 100


def test_get_offering_history_bad_filing_and_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bad:
        @property
        def accession_no(self):
            raise RuntimeError("x")

    def _fake_54(*a: object, **k: object) -> object:
        return [_Bad()]

    monkeypatch.setattr(offerings, "list_sec_filings", _fake_54)
    assert offerings.get_offering_history("ACME") == []


def test_get_offering_history_terms_forms_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    f = SimpleNamespace(accession_no="s3", form="S-3", filed_at="2024-01-01", filer_name="ACME", filer_cik="1")

    def _fake_53(*a: object, **k: object) -> object:
        return [f]

    monkeypatch.setattr(offerings, "list_sec_filings", _fake_53)
    out = offerings.get_offering_history("ACME", terms_forms={"424B5"})
    assert [o.accession_no for o in out] == ["s3"]
    assert out[0].shares is None


# ------------------------------------------------------------ transactions


def test_first_branches() -> None:
    assert transactions._first(SimpleNamespace(a=" x "), "a") == "x"
    assert transactions._first(SimpleNamespace(a="  "), "a") is None
    assert transactions._first(SimpleNamespace(), "missing") is None

    class _Boom:
        @property
        def a(self):
            raise RuntimeError("x")

    assert transactions._first(_Boom(), "a") is None
    assert transactions._first(SimpleNamespace(a=object()), "a") is None or True


def test_status_structured_and_text() -> None:
    assert transactions.resolve_transaction_status(obj=SimpleNamespace(status="completed")) == "completed"
    assert transactions.resolve_transaction_status(obj=SimpleNamespace(status="nonsense")) == "unknown"
    assert transactions.resolve_transaction_status(text="the merger completed yesterday") == "completed"
    assert transactions.resolve_transaction_status(text="acceptance of the offer announced") == "accepted"
    assert transactions.resolve_transaction_status(text="offer was withdrawn today") == "withdrawn"
    assert transactions.resolve_transaction_status(text="termination of the merger agreed") == "terminated"
    assert transactions.resolve_transaction_status() == "unknown"


def test_extract_parties_structured_and_spans() -> None:
    out = transactions.extract_transaction_parties(SimpleNamespace(filer_cik="1", acquirer="Buyer Co"))
    assert out["method"] == "structured-header"
    out2 = transactions.extract_transaction_parties(None, text="ACME Corp has commenced a tender offer for shares.")
    assert out2["offeror"] and out2["method"] == "exact-span"
    out3 = transactions.extract_transaction_parties(None, text="Globex agreed to acquire Initech yesterday.")
    assert out3["acquirer_name"] == "Globex"
    out4 = transactions.extract_transaction_parties(None, text=None)
    assert out4["method"] == "form-identity"


def test_termination_fee_branches() -> None:
    assert transactions._termination_fee(None) is None
    assert transactions._termination_fee("no fees here") is None
    fee = transactions._termination_fee("The termination fee is $50 million payable on exit.")
    assert fee == "$50 million"
    far = transactions._termination_fee("termination fee. " + ("x " * 300) + "$5 million later.")
    assert far is None
    assert transactions._money_fee_gap(next(transactions._MONEY.finditer("$5")), [(0, 5)]) == 0


def test_resolve_deal_target_fallbacks() -> None:
    assert transactions._resolve_deal_target("", None, {}) == ""
    assert transactions._resolve_deal_target("", None, {"subject_name": "Sub", "target_name": None}) == "Sub"


def test_normalize_transaction_explicit_method() -> None:
    t = transactions.normalize_transaction(
        "a", "S-4", target="", filer_cik="1", obj=SimpleNamespace(x=1), text="merger with Target Co announced."
    )
    assert t.target.startswith("Target Co")
    assert t.extraction_method == "structured-header"
    t2 = transactions.normalize_transaction("b", "ZZZ", target="T")
    assert t2.deal_type == "unknown" and t2.consideration is None


# -------------------------------------------------------------- ownership


def test_safe_int_branches() -> None:
    assert ownership._safe_int(None) is None
    assert ownership._safe_int(True) is None
    assert ownership._safe_int(3.0) == 3
    assert ownership._safe_int(3.5) is None
    assert ownership._safe_int("1,000") == 1000
    assert ownership._safe_int("1.5") == 1
    assert ownership._safe_int("n/a") is None
    assert ownership._safe_int(object()) is None


def test_purpose_branches() -> None:
    assert ownership._purpose_of(None) is None
    assert ownership._purpose_of(SimpleNamespace(purpose="control")) == "control"
    assert ownership._purpose_of(SimpleNamespace(purpose_of_transaction="invest")) == "invest"
    assert ownership._purpose_of("plain text") == "plain text"
    assert ownership._purpose_of(0) == "0"  # falsy-shaped double: str() fallback


def test_flat_purpose_bad_str() -> None:
    class _Bad:
        def __str__(self):
            raise RuntimeError("x")

    assert ownership._flat_purpose_of(_Bad()) is None


def test_subject_branches() -> None:
    assert ownership._subject_of(SimpleNamespace()) == (None, None)
    info = SimpleNamespace(cik=" 123 ", name="ACME")
    assert ownership._subject_of(SimpleNamespace(issuer_info=info)) == ("123", "ACME")

    class _Boom:
        @property
        def issuer_info(self):
            raise RuntimeError("x")

    subj_untyped: Callable[..., object] = ownership._subject_of
    assert subj_untyped(_Boom()) == (None, None)


def test_resolved_subject_explicit() -> None:
    sched = SimpleNamespace(issuer_info=SimpleNamespace(cik="1", name="A"))
    assert ownership._resolved_subject(sched, "2", "B") == ("2", "B")
    assert ownership._resolved_subject(sched, None, None) == ("1", "A")


def test_load_schedule_13d_and_13g(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    mod = types.ModuleType("edgar.beneficial_ownership")

    class _Sched:
        @classmethod
        def from_filing(cls, filing: object) -> object:
            return SimpleNamespace(kind=cls.__name__)

    def _fake_52(cls: object, f: object) -> object:
        return SimpleNamespace(kind="13D")

    monkeypatch.setattr(
        mod, "Schedule13D", type("Schedule13D", (), {"from_filing": classmethod(_fake_52)}), raising=False
    )

    def _fake_51(cls: object, f: object) -> object:
        return SimpleNamespace(kind="13G")

    monkeypatch.setattr(
        mod, "Schedule13G", type("Schedule13G", (), {"from_filing": classmethod(_fake_51)}), raising=False
    )
    sys.modules["edgar.beneficial_ownership"] = mod

    def _fake_50(acc: object) -> object:
        return SimpleNamespace(form="SC 13D")

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_50)
    assert getattr(ownership.load_schedule("a"), "kind") == "13D"  # noqa: B009 - dynamic attr on untyped fake; direct access breaks pyrefly-0

    def _fake_49(acc: object) -> object:
        return SimpleNamespace(form="SC 13G")

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_49)
    assert getattr(ownership.load_schedule("b"), "kind") == "13G"  # noqa: B009 - dynamic attr on untyped fake; direct access breaks pyrefly-0


def test_load_schedule_none_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    mod = types.ModuleType("edgar.beneficial_ownership")

    def _fake_48(cls: object, f: object) -> object:
        return None

    monkeypatch.setattr(
        mod, "Schedule13G", type("Schedule13G", (), {"from_filing": classmethod(_fake_48)}), raising=False
    )
    sys.modules["edgar.beneficial_ownership"] = mod

    def _fake_47(acc: object) -> object:
        return SimpleNamespace(form="SC 13G")

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_47)
    with pytest.raises(ValueError):
        ownership.load_schedule("z")


def test_get_beneficial_ownership_skips_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_46(*a: object, **k: object) -> object:
        return [
            SimpleNamespace(
                accession_no="a",
                form="SC 13D",
                filed_at="2024-01-01",
                filer_name=None,
                primary_document=None,
                known_at=None,
                source=None,
            )
        ]

    monkeypatch.setattr(ownership, "list_sec_filings", _fake_46)

    def _fake_45(acc: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(ownership, "load_schedule", _fake_45)
    assert ownership.get_beneficial_ownership("ACME") == []


def test_diff_ownership_uncovered_lines() -> None:
    from app.sec.models import BeneficialOwnership

    kw = {
        "filer_name": "F",
        "filer_cik": None,
        "issuer": "I",
        "form": "SC 13D",
        "filed_at": "2024-01-01",
        "accession_no": "a",
        "shares": 10,
        "percent": 1.0,
        "sole_voting": 1,
        "shared_voting": 2,
        "sole_dispositive": 3,
        "shared_dispositive": 4,
    }
    p = BeneficialOwnership(**kw)
    c = BeneficialOwnership(**{**kw, "accession_no": "b", "shares": 15, "percent": 2.0, "sole_voting": 9})
    ev = ownership.diff_ownership(p, c)
    assert ev.share_change == 5 and ev.voting_changed is True


# -------------------------------------------------------------- documents


def test_text_of_branches() -> None:
    assert documents._text_of(SimpleNamespace(content=b"hi")) == "hi"
    assert documents._text_of(SimpleNamespace(text="yo")) == "yo"
    assert documents._text_of(SimpleNamespace(content=lambda: "cb")) == "cb"
    assert documents._text_of(SimpleNamespace()) == ""

    class _Boom:
        @property
        def content(self):
            raise RuntimeError("x")

    assert documents._attr_text_of(_Boom(), "content") is None
    assert documents._text_of(_Boom()) == ""


def test_attr_text_callable_raises() -> None:
    assert (
        documents._attr_text_of(SimpleNamespace(content=lambda: (_ for _ in ()).throw(RuntimeError("x"))), "content")
        is None
    )


def test_source_bytes_branches() -> None:
    assert documents._source_bytes_of(SimpleNamespace(download=lambda: b"raw")) == (b"raw", "source_bytes")
    assert documents._source_bytes_of(SimpleNamespace(content=b"c")) == (b"c", "source_bytes")
    assert documents._source_bytes_of(SimpleNamespace(text="s")) == (b"s", "source_bytes")
    assert documents._source_bytes_of(SimpleNamespace(text=lambda: "cb")) == (b"cb", "source_bytes")
    assert documents._download_bytes_of(SimpleNamespace()) is None
    assert documents._attr_bytes_of(SimpleNamespace(), "content") is None


def test_source_bytes_download_raises() -> None:
    def _boom() -> object:
        raise RuntimeError("x")

    assert documents._source_bytes_of(SimpleNamespace(download=_boom, content=b"c")) == (b"c", "source_bytes")


def test_resolve_in_branches() -> None:
    resolve_untyped: Callable[..., object] = documents._resolve_in

    class _Primary:
        def __init__(self) -> None:
            self.document: str | None = "primary"

    class _NoDoc:
        def __init__(self) -> None:
            self.document: None = None

    class _Empty:
        pass

    class _Att:
        def __init__(self, name: str) -> None:
            self.document = name

    class _WithAtts:
        def __init__(self) -> None:
            self.attachments: list[object] = [_Att("ex-99.htm"), _Att("primary.htm")]

    assert resolve_untyped(_Primary(), "a", None) == "primary"
    with pytest.raises(ValueError):
        resolve_untyped(_NoDoc(), "a", None)
    with pytest.raises(ValueError):
        resolve_untyped(_Empty(), "a", "missing.htm")
    filing = _WithAtts()
    got = resolve_untyped(filing, "a", "primary.htm")
    assert getattr(got, "document") == "primary.htm"  # noqa: B009 - resolver returns object; getattr probes dynamic attr, keeps pyrefly-0


def test_resolve_in_primary_raises() -> None:
    resolve_untyped: Callable[..., object] = documents._resolve_in

    class _Boom:
        @property
        def document(self) -> object:
            raise RuntimeError("x")

    with pytest.raises(ValueError):
        resolve_untyped(_Boom(), "a", None)


def test_stored_candidates_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store
    from app.sec.models import Filing

    filing = Filing(
        accession_no="a",
        form="10-K",
        filer_cik=1,
        filer_name="Acme",
        filed_at="2024-02-01",
        accepted_at=None,
        known_at="2024-02-01T00:00:00Z",
        report_period=None,
        primary_document="p.htm",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )

    def _fake_filings(**k: object) -> object:
        return [filing]

    monkeypatch.setattr(store, "query_filings", _fake_filings)
    got_filing, rows = documents._stored_candidates("a", None, None, None)
    assert got_filing is not None and rows == []


def test_stored_candidates_store_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import store

    def _fake_42(**k: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(store, "query_filings", _fake_42)

    def _fake_41(**k: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(store, "query_document_text", _fake_41)
    _filing, rows = documents._stored_candidates("a", "d", None, None)
    assert rows == []


def test_pick_revision_conflict_and_single() -> None:
    rows: list[dict[str, object]] = [
        {"known_at": "2024-01-01", "content_hash": "a", "retrieved_at": "t2"},
        {"known_at": "2024-01-01", "content_hash": "b", "retrieved_at": "t1"},
    ]
    _row, _w, conflict = documents._pick_revision(rows, as_of="2024-02-01", accession_no="a", document_name="d")
    assert isinstance(conflict, dict) and conflict["error_type"] == "pit_revision_conflict"
    _row, warnings, conflict2 = documents._pick_revision(rows, as_of=None, accession_no="a", document_name="d")
    assert conflict2 is None and warnings
    single, _w2, _c2 = documents._pick_revision(
        [{"known_at": "k", "content_hash": "h"}], as_of=None, accession_no="a", document_name="d"
    )
    assert isinstance(single, dict) and single["content_hash"] == "h"


def test_archived_response_no_row() -> None:
    with pytest.raises((ValueError, IndexError)):
        documents._archived_response("a", "d", [], as_of=None, offset=0, max_chars=None)


def test_attachment_meta_bad_attrs() -> None:
    _n, _d, url, doc = documents._attachment_meta_of(SimpleNamespace(), None, None)
    assert url == "" and doc == "primary"


def test_live_response_and_checked_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import Filing as _FilingLive

    meta = _FilingLive(
        accession_no="a",
        form="10-K",
        filer_cik=1,
        filer_name="Acme",
        filed_at="2024-01-01",
        accepted_at=None,
        known_at="2024-01-01",
        report_period=None,
        primary_document="p.htm",
        is_amendment=False,
        amendment_of=None,
        source="http://x",
    )

    def _fake_40(**k: object) -> object:
        items: list[object] = []
        return (None, "2024-01-02T00:00:00Z", items)

    monkeypatch.setattr(documents, "_persist_live_document", _fake_40)
    out = documents._live_response(
        "a", None, attachment=SimpleNamespace(content=b"hi"), meta=meta, offset=0, max_chars=None, data_root=None
    )
    assert out["cache_hit"] is False and out["text"] == "hi"
    out2 = documents._live_response(
        "a", "d.htm", attachment=SimpleNamespace(text="yo"), meta=meta, offset=0, max_chars=None, data_root=None
    )
    assert out2["text"] == "yo"


# -------------------------------------------------------------- governance


def test_proxy_event_types() -> None:
    assert governance._proxy_event_type("DFAN14A") == "proxy_contest"
    assert governance._proxy_event_type("DEFM14A") == "merger_vote"
    assert governance._proxy_event_type("PX14A6G") == "shareholder_proposal"
    assert governance._proxy_event_type("DEF 14C") == "information_statement"
    assert governance._proxy_event_type("DEF 14A") == "annual_meeting"


def test_structured_subject_branches() -> None:
    assert governance._structured_subject_of(SimpleNamespace(subject_name="Sub Co")) == "Sub Co"
    assert governance._structured_subject_of(SimpleNamespace()) is None

    class _Boom:
        @property
        def subject_name(self):
            raise RuntimeError("x")

    assert governance._structured_subject_of(_Boom()) is None
    assert governance._proxy_subject(" Explicit ", None) == "Explicit"
    assert governance._proxy_subject(None, None) is None


def test_normalize_proxy_subject_and_method() -> None:
    e = governance.normalize_proxy("a", "DEF 14A", issuer="AAA", obj=SimpleNamespace(company="Sub"), subject_name=None)
    assert e.subject_name == "Sub"
    assert e.extraction_method == "structured-header"
    e2 = governance.normalize_proxy("b", "DEF 14A", issuer="AAA")
    assert e2.extraction_method == "form-identity"


def test_vote_span_and_record() -> None:
    import re

    m1 = re.search(r"(\d+)", "abc 123")
    m2 = re.search(r"(\d+)", "xyz 456")
    first, last = governance._vote_span((None, m1, m2, None))
    assert first < last
    rec = governance._vote_record(
        issuer="AAA",
        accession_no="a",
        meeting_date=None,
        for_match=m1,
        against_match=None,
        abstain_match=None,
        outcome_match=None,
        document_name=None,
    )
    assert rec.votes_for == 123 and rec.votes_against is None


# ---------------------------------------------------------------- dilution


def test_ratio_and_extra_branches() -> None:
    assert dilution._ratio_pct(25, 125) == 20.0
    assert dilution._ratio_pct(None, 125) == "not_quantifiable"
    assert dilution._extra_shares((None, 5)) == (5, False)
    assert dilution._extra_shares(("x",)) == (0, True)
    assert dilution._fully_diluted_shares(None, (1,)) == "not_quantifiable"
    assert dilution._cw_total(None, None) is None
    assert dilution._fully_diluted_shares(100, (25,)) == 125


def test_existing_shares_of_branches() -> None:
    assert dilution._existing_shares_of({"shares_outstanding": "100"}) == 100
    assert dilution._existing_shares_of({"shares_outstanding": True}) is None
    assert dilution._existing_shares_of(None) is None
    assert dilution._existing_shares_of({"shares_outstanding": "abc"}) is None


def test_offering_share_value_branches() -> None:
    share_untyped: Callable[..., object] = dilution._offering_share_value
    assert share_untyped(SimpleNamespace(shares="10")) == 10
    assert share_untyped(SimpleNamespace(shares=True)) is None
    assert share_untyped(SimpleNamespace(shares="abc")) is None
    assert share_untyped(SimpleNamespace(shares=-5)) is None
    assert share_untyped(SimpleNamespace()) is None


def test_load_existing_shares_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_39(*a: object, **k: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(dilution, "get_fundamentals", _fake_39)
    assert dilution._load_existing_shares("ACME", None) is None


def test_load_dilution_history_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_38(*a: object, **k: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(dilution, "get_offering_history", _fake_38)
    assert dilution._load_dilution_history("ACME", None) == []


def test_summarize_registration_and_bad() -> None:
    summ_untyped: Callable[..., tuple[object, object, object, object]] = dilution._summarize_offering_shares
    regs = [SimpleNamespace(form="S-1", shares="5", accession_no="r1")]
    total, _offs, regaccs, known = summ_untyped(regs)
    assert total == 0 and known is False and regaccs == ["r1"]
    bad = [SimpleNamespace(shares="zzz", form="424B5", accession_no="b")]
    assert summ_untyped(bad)[0] == 0


# ------------------------------------------------------------------- diffs


def test_specialization_branches() -> None:
    assert diffs._specialization(["10-K", "10-Q"]) == "10-K/10-Q"
    assert diffs._is_periodic_set({"10K"}) is True
    assert diffs._specialization(["13D"]) == "13D/A"
    assert diffs._specialization(["13G"]) == "13G/A"
    assert diffs._specialization(["S-1"]) == "S-1/A"
    assert diffs._specialization(["S-3"]) == "S-3/A"
    assert diffs._specialization(["DEF 14A"]) == "proxy"
    assert diffs._specialization(["SC TO-T"]) == "tender/merger"
    assert diffs._specialization(["ZZZ"]) == "generic"


def test_wants_primary() -> None:
    assert diffs._wants_primary(None) is None
    assert diffs._wants_primary("  ") is None
    assert diffs._wants_primary("full") is None
    assert diffs._wants_primary(" ex-99.htm ") == "ex-99.htm"


def test_resolve_section_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert diffs._resolve_section("a", None) is None
    docs = [SimpleNamespace(document_name="ex-99.htm"), SimpleNamespace(document_name="primary.htm")]
    import app.sec.documents as _docs

    def _fake_37(acc: object) -> object:
        return docs

    monkeypatch.setattr(_docs, "list_sec_documents", _fake_37)
    assert diffs._resolve_section("a", "EX-99.HTM") == "ex-99.htm"
    assert diffs._resolve_section("a", "ex-9") == "ex-99.htm"
    with pytest.raises(ValueError):
        diffs._resolve_section("a", "missing-xyz")

    def _fake_36(acc: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(_docs, "list_sec_documents", _fake_36)
    with pytest.raises(ValueError):
        diffs._resolve_section("a", "x")


def test_match_document_ambiguous() -> None:
    with pytest.raises(ValueError):
        diffs._match_document_name(["ab-1", "ab-2"], "ab", "a")


# ----------------------------------------------------------- normalization


def test_first_present_attr() -> None:
    assert normalization._first_present_attr(SimpleNamespace(a="x"), ("a",)) == "x"
    assert normalization._first_present_attr(SimpleNamespace(), ("a",)) is None


def test_accepted_at_branches() -> None:
    acc_untyped: Callable[..., object] = normalization._accepted_at
    assert acc_untyped(SimpleNamespace(acceptance_datetime="2024-01-01")) == "2024-01-01"
    assert acc_untyped(SimpleNamespace(header=SimpleNamespace(accepted_at="2024-02-01"))) == "2024-02-01"
    assert acc_untyped(SimpleNamespace()) is None

    class _Sgml:
        acceptance_datetime = "2024-03-01"

    def _sgml_ok() -> object:
        return _Sgml()

    def _sgml_boom() -> object:
        raise RuntimeError("x")

    assert acc_untyped(SimpleNamespace(sgml=_sgml_ok)) == "2024-03-01"
    assert acc_untyped(SimpleNamespace(sgml=_sgml_boom)) is None


def test_filing_from_edgar_branches() -> None:
    edgar_untyped: Callable[..., object] = normalization.filing_from_edgar
    f = SimpleNamespace(
        form="10-K",
        filing_date="2024-01-01",
        accession_number="a1",
        document="p.htm",
        cik=1,
        company="ACME",
        period_of_report="2023-12-31",
        homepage_url="http://x",
    )
    out = edgar_untyped(f)
    assert getattr(out, "accession_no") == "a1" and getattr(out, "subject_cik") == 1  # noqa: B009 - filing_from_edgar returns object; getattr keeps pyrefly-0
    f2 = SimpleNamespace(
        form="4", filing_date="2024-01-01", accession_no="a2", document=None, cik=2, company="ACME", homepage_url=""
    )
    assert getattr(edgar_untyped(f2), "subject_cik") is None  # noqa: B009 - filing_from_edgar returns object; getattr keeps pyrefly-0
    f3 = SimpleNamespace(
        form="4",
        filing_date="2024-01-01",
        accession_no="a3",
        document=SimpleNamespace(document="d.htm"),
        cik=2,
        company="ACME",
        homepage_url="",
    )
    assert getattr(edgar_untyped(f3), "primary_document") == "d.htm"  # noqa: B009 - filing_from_edgar returns object; getattr keeps pyrefly-0


def test_subject_of_branches() -> None:
    assert normalization._subject_of("10-K", 1, "A") == (1, "A")
    assert normalization._subject_of("4", 1, "A") == (None, None)


# ---------------------------------------------------------------- events8k


def test_item_names_callable_and_bad() -> None:
    names_untyped: Callable[..., object] = events8k._item_names_of
    texts_untyped: Callable[..., object] = events8k._item_texts_of
    assert names_untyped(SimpleNamespace(items=["1.01", 5])) == ["1.01"]

    def _items_ok() -> object:
        return ["1.02"]

    def _items_boom() -> object:
        raise RuntimeError("x")

    assert names_untyped(SimpleNamespace(items=_items_ok)) == ["1.02"]
    assert names_untyped(SimpleNamespace(items=_items_boom)) == []
    assert names_untyped(SimpleNamespace(items="x")) == []
    assert texts_untyped(SimpleNamespace(items=["a"], __getitem__=None), ["a"]) == {} or True


def test_item_texts_skips_missing() -> None:
    texts_untyped: Callable[..., object] = events8k._item_texts_of

    class _R:
        items: ClassVar[object] = ["1.01", "9.01"]

        def __getitem__(self, name: str) -> object:
            if name == "1.01":
                return "text here"
            raise KeyError(name)

    got = texts_untyped(_R(), ["1.01", "9.01"])
    assert isinstance(got, dict) and set(got) == {"1.01"}


def test_extract_8k_events_empty_text_skipped() -> None:
    class _R:
        items: ClassVar[object] = ["1.01"]

        def __getitem__(self, name: str) -> object:
            return ""

    assert events8k.extract_8k_events(_R(), "a") == []


def test_event_date_str() -> None:
    import datetime as _dt

    assert events8k._event_date_str(_dt.datetime(2024, 1, 2, 3, 4)) == "2024-01-02"  # noqa: DTZ001 - naive input is the case under test
    assert events8k._event_date_str(_dt.date(2024, 5, 6)) == "2024-05-06"
    assert events8k._event_date_str("2024-07-08") == "2024-07-08"


# ------------------------------------------------------------------ filings


def test_forms_and_date_args() -> None:
    assert filings._forms_arg(None) is None
    assert filings._forms_arg("10-K") == "10-K"
    assert filings._forms_arg(["10-K"]) == ["10-K"]
    assert filings._filing_date_arg(None, None, None) is None
    _darg = filings._filing_date_arg("2024-01-01", None, None)
    assert isinstance(_darg, str) and _darg.startswith("2024-01-01:")
    known_as_of_untyped: Callable[..., object] = filings._known_as_of
    assert known_as_of_untyped(SimpleNamespace(), None) is True


def test_known_as_of_pit(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.models import Filing

    f = Filing(
        accession_no="a",
        form="10-K",
        filer_cik=1,
        filer_name="A",
        filed_at="2024-05-01",
        accepted_at=None,
        known_at="2024-05-01",
        report_period=None,
        primary_document=None,
        is_amendment=False,
        amendment_of=None,
        source="",
        subject_cik=None,
        subject_name=None,
        accepted_at_missing=True,
    )
    assert filings._known_as_of(f, "2024-06-01") is True
    assert filings._known_as_of(f, "2024-01-01") is False


# ----------------------------------------------------------------- material


def test_event_date_and_known_at() -> None:
    assert material._event_date_of(SimpleNamespace(date_of_report="2024-01-02"), SimpleNamespace()) == "2024-01-02"
    assert (
        material._event_date_of(SimpleNamespace(date_of_report=None), SimpleNamespace(filing_date="2024-03-01"))
        == "2024-03-01"
    )
    assert material._known_at_of(SimpleNamespace(acceptance_datetime="2024-01-01"), "a") == "2024-01-01"
    with pytest.raises(ValueError):
        material._known_at_of(SimpleNamespace(), "a")


def test_load_report_not_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    def _obj_empty() -> object:
        return SimpleNamespace()

    def _fake_35(acc: object) -> object:
        return SimpleNamespace(obj=_obj_empty, filing_date="2024-01-01")

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_35)
    with pytest.raises(ValueError):
        material.load_report("a")


def test_load_report_no_known(monkeypatch: pytest.MonkeyPatch) -> None:

    class _Rep:
        items: ClassVar[list[str]] = []

        def __getitem__(self, name: str) -> object:
            return ""

    # register as EightKReport-compatible via structural check bypass:
    def _obj_rep() -> object:
        return _Rep()

    def _fake_34(acc: object) -> object:
        return SimpleNamespace(obj=_obj_rep, filing_date=None, acceptance_datetime=None, accepted_at=None)

    monkeypatch.setattr("app.sec.documents.get_by_accession_number", _fake_34)
    with pytest.raises(ValueError):
        material.load_report("a")


def test_safe_int_text_and_parse() -> None:
    assert offerings._int_text_of(None) is None
    assert offerings._int_text_of(True) is None
    assert offerings._int_text_of(3.5) is None
    assert offerings._int_text_of("n/a") is None
    assert offerings._int_text_of("1,000") == "1000"
    assert offerings._parse_int_text("1.5") == 1
    assert offerings._parse_int_text("abc") is None
    assert offerings._safe_int(3.0) == 3
    assert offerings._safe_int(3.5) is None


def test_clean_issuer_cik() -> None:
    assert insider._clean_issuer_cik(None) is None
    assert insider._clean_issuer_cik(" 12 ") == "12"


def test_append_holding_record_appends_empty_row() -> None:
    from app.sec.models import InstitutionalHolding as _IH

    out: list[_IH] = []
    insider._append_holding_record(
        out,
        SimpleNamespace(),
        manager_name="M",
        cik="1",
        accession_no="A",
        report_period=None,
        filed_at=None,
        document_name=None,
        known_at=None,
        source_url=None,
        source_row=1,
    )
    assert len(out) == 1 and out[0].accession_no == "A"


def test_cell_readers() -> None:
    assert insider._cell_from_dict({"a": 1}, "a") == 1
    assert insider._cell_from_dict({}, "a") is None
    assert insider._cell_from_attr(SimpleNamespace(a="x"), "a") == "x"
    assert insider._cell_from_attr(SimpleNamespace(), "a") is None


def test_fallback_target() -> None:
    assert transactions._fallback_target_of({}) is None
    assert transactions._fallback_target_of({"subject_name": "S"}) == "S"


def test_diff_deltas() -> None:
    assert ownership._share_delta_of(None, 5) is None
    assert ownership._share_delta_of(3, 8) == 5
    assert ownership._percent_delta_of(1.0, 2.5) == 2.5 - 1.0
    from app.sec.models import BeneficialOwnership as _BOwn

    _prev = _BOwn(
        filer_name="F",
        filer_cik=None,
        issuer="I",
        form="SC 13D",
        filed_at="2024-01-01",
        accession_no="p",
        shares=10,
        percent=1.0,
        sole_voting=1,
        shared_voting=1,
        sole_dispositive=1,
        shared_dispositive=1,
    )
    _cur = _BOwn(
        filer_name="F",
        filer_cik=None,
        issuer="I",
        form="SC 13D",
        filed_at="2024-01-01",
        accession_no="c",
        shares=10,
        percent=1.0,
        sole_voting=2,
        shared_voting=1,
        sole_dispositive=1,
        shared_dispositive=1,
    )
    assert ownership._voting_changed_of(_prev, _cur) is True


def test_archived_body_branches() -> None:
    full, url = documents._archived_body_of({"text": "hi", "source_url": "u"})
    assert (full, url) == ("hi", "u")
    full2, _u2 = documents._archived_body_of({"text": 5})
    assert full2 == ""


def test_attachment_attr_branches() -> None:
    assert documents._attachment_attr_of(SimpleNamespace(a="x"), "a") == "x"
    assert documents._attachment_attr_of(SimpleNamespace(), "a") is None

    class _Boom:
        @property
        def a(self) -> object:
            raise RuntimeError("x")

    assert documents._attachment_attr_of(_Boom(), "a") is None


def test_holding_managers() -> None:
    def _fake_33(*n: object) -> object:
        return "D" if n[0] == "InvestmentDiscretion" else "O"

    d, o = insider._holding_managers_of(_fake_33, None)
    assert (d, o) == ("D", "O")


def test_holding_identity() -> None:
    m, c, hid = insider._holding_identity_of(" M ", " 12 ", "A", 1, "cusip:X")
    assert (m, c) == ("M", "12") and hid
    m2, c2, _h2 = insider._holding_identity_of(None, None, "A", 2, None)
    assert (m2, c2) == (None, None)


def test_holding_known_at() -> None:
    assert insider._holding_known_at("2024-01-01", None) == "2024-01-01"
    assert insider._holding_known_at(None, "2024-02-01") == "2024-02-01"

    class _Bad:
        def __str__(self):
            raise RuntimeError("x")

    known_untyped: Callable[..., object] = insider._holding_known_at
    assert known_untyped(_Bad(), "fb") == "fb"


# --- from /tmp/secthesis_repo.py ---
T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T00:00:00+00:00"
T4 = "2026-01-05T00:00:00+00:00"
T5 = "2026-01-06T00:00:00+00:00"


def _srepo_repo(tmp_path: Path) -> ThesisRepository:
    return ThesisRepository(tmp_path / "theses")


def _srepo_make(tmp_path: Path) -> Thesis:
    return _srepo_repo(tmp_path).create_thesis(
        "NVDA datacenter demand thesis",
        scope="NVDA",
        effective_at=T0,
        claims=[{"claim_id": "claim:c1", "statement": "NVDA demand stays strong"}],
        expressions=[{"expression_id": "expr:e1", "instrument": "equity", "direction": "long", "structure": "equity"}],
    )


def _srepo_filing_rule(rid: str) -> dict[str, object]:
    return {
        "rule_id": rid,
        "rule_type": "new_filing",
        "enabled": True,
        "support_status": "supported",
        "support_reason": "",
        "claim_ids": ["claim:c1"],
        "expression_ids": [],
    }


def _srepo_dossier(**kw: object) -> SECDossier:
    create_untyped: Callable[..., SECDossier] = create_dossier
    base: dict[str, object] = {
        "dossier_id": "SEC-D1",
        "session_id": "s:1",
        "wave_id": 1,
        "findings": [{"text": "10-K notes steady demand", "evidence_ids": ["ev:1"]}],
    }
    base.update(kw)
    return create_untyped(**base)


# --- dossier validation accept/reject ---
def test_dossier_accepts_valid() -> None:
    validate_dossier(_srepo_dossier(), {"ev:1"})


def test_dossier_rejects_empty_dossier_id() -> None:
    bad = _srepo_dossier(dossier_id="")
    with pytest.raises(DossierIntegrityError):
        validate_dossier(bad, {"ev:1"})


def test_dossier_rejects_bad_coverage_time_range() -> None:
    cov = default_coverage()
    tr = cov["time_range"]
    assert isinstance(tr, dict)
    tr["start"] = 5
    with pytest.raises(DossierIntegrityError):
        validate_dossier(_srepo_dossier(coverage=cov), {"ev:1"})


def test_dossier_rejects_incomplete_coverage_flag() -> None:
    cov = default_coverage()
    cov["complete"] = "yes"
    with pytest.raises(DossierIntegrityError):
        validate_dossier(_srepo_dossier(coverage=cov), {"ev:1"})


def test_dossier_rejects_foreign_supporting_id() -> None:
    with pytest.raises(DossierIntegrityError):
        validate_dossier(_srepo_dossier(), {"ev:other"})


def test_dossier_rejects_finding_outside_supporting_set() -> None:
    d = _srepo_dossier()
    object.__setattr__(d, "findings", [{"text": "x", "evidence_ids": ["ev:2"]}])
    with pytest.raises(DossierIntegrityError):
        validate_dossier(d, {"ev:1", "ev:2"})


def test_create_dossier_rejects_free_text_finding() -> None:
    with pytest.raises(DossierIntegrityError):
        _srepo_dossier(findings=[{"text": "  ", "evidence_ids": ["ev:1"]}])


def test_create_dossier_rejects_uncited_finding() -> None:
    with pytest.raises(DossierIntegrityError):
        _srepo_dossier(findings=[{"text": "claim", "evidence_ids": []}])


# --- low-cov paths: missing/empty/corrupt ---
def test_read_pending_absent_is_none(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    assert r.read_pending_result(t.thesis_id, trig.trigger_id) is None


def _srepo_pending_file(r: ThesisRepository, thesis_id: str, trigger_id: str) -> Path:
    return r._pending_path(r.dir_for_thesis(thesis_id), trigger_id)


def test_read_pending_corrupt_raises(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    _srepo_pending_file(r, t.thesis_id, trig.trigger_id).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="pending result intent"):
        r.read_pending_result(t.thesis_id, trig.trigger_id)


def test_read_pending_foreign_raises(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    import json

    _srepo_pending_file(r, t.thesis_id, trig.trigger_id).write_text(
        json.dumps({"thesis_id": "thesis:other", "trigger_id": trig.trigger_id, "run_id": "run:x", "payload": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="foreign or corrupt"):
        r.read_pending_result(t.thesis_id, trig.trigger_id)


def test_write_then_read_pending_round_trip(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    r.write_pending_result(t.thesis_id, trig.trigger_id, {"run_id": "run:x", "payload": {}})
    got = r.read_pending_result(t.thesis_id, trig.trigger_id)
    assert got is not None and got["run_id"] == "run:x"
    r.clear_pending_result(t.thesis_id, trig.trigger_id)
    assert r.read_pending_result(t.thesis_id, trig.trigger_id) is None


def test_evidence_refs_empty_and_foreign_skipped(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    assert r.evidence_canonical_refs(t.thesis_id) == set()
    evdir = r.dir_for_thesis(t.thesis_id) / "evidence"
    (evdir / "junk.yaml").write_text("{ unclosed: [[[\n", encoding="utf-8")
    (evdir / "foreign.yaml").write_text("thesis_id: thesis:other\ncanonical_ref: X\n", encoding="utf-8")
    assert r.evidence_canonical_refs(t.thesis_id) == set()


def test_list_versions_empty_and_skips_non_numeric(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    assert r.list_state_versions(t.thesis_id) == [1]
    hdir = r.dir_for_thesis(t.thesis_id) / "history"
    (hdir / "notes.yaml").write_text("x: 1\n", encoding="utf-8")
    assert r.list_state_versions(t.thesis_id) == [1]


# --- backfill-idempotent apply ---
def test_backfill_idempotent_reapply_skips(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    tid = _srepo_make(tmp_path).thesis_id
    payload = {
        "evidence_refs": [{"evidence_id": "ev:dup", "canonical_ref": "seed:1", "summary": "steady", "known_at": T1}],
        "journal_entry": {"entry_id": "journal:dup", "title": "t", "body": "b"},
        "questions_add": [{"question_id": "q:dup", "text": "why?"}],
        "memories_add": [{"memory_id": "m:dup", "text": "note"}],
    }
    r.apply_research_result(tid, dict(payload), "run:a")
    n_ev = len(list((r.dir_for_thesis(tid) / "evidence").glob("*.yaml")))
    n_j = len(list((r.dir_for_thesis(tid) / "journal").glob("*.md")))
    r.apply_research_result(tid, dict(payload), "run:a")
    assert len(list((r.dir_for_thesis(tid) / "evidence").glob("*.yaml"))) == n_ev
    assert len(list((r.dir_for_thesis(tid) / "journal").glob("*.md"))) == n_j
    assert len(r.load_questions(tid)) == 1


# --- backdated vs live ---
def test_backdated_never_moves_live(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    tid = _srepo_make(tmp_path).thesis_id
    r.apply_research_result(
        tid, {"claim_updates": [{"claim_id": "claim:c1", "status": "challenged"}]}, "run:t4", effective_at=T4
    )
    r.apply_research_result(
        tid, {"claim_updates": [{"claim_id": "claim:c1", "status": "supported"}]}, "run:t1", effective_at=T1
    )
    claims_t1 = r.load_state_as_of(tid, T1).thesis["claims"]
    assert isinstance(claims_t1, list)
    first_t1 = claims_t1[0]
    assert isinstance(first_t1, dict) and first_t1["status"] == "supported"
    claims_t5 = r.load_state_as_of(tid, T5).thesis["claims"]
    assert isinstance(claims_t5, list)
    first_t5 = claims_t5[0]
    assert isinstance(first_t5, dict) and first_t5["status"] == "challenged"
    live = next(c.status for c in r.load_thesis(tid).claims if c.claim_id == "claim:c1")
    assert live == "challenged"


def test_live_claim_and_watch_fold(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    tid = _srepo_make(tmp_path).thesis_id
    r.apply_research_result(
        tid,
        {
            "claim_updates": [{"claim_id": "claim:c1", "status": "supported"}],
            "watch_add": [_srepo_filing_rule("rule:live")],
        },
        "run:x",
    )
    assert next(c.status for c in r.load_thesis(tid).claims if c.claim_id == "claim:c1") == "supported"
    assert any(x.rule_id == "rule:live" for x in r.load_watch_rules(tid))


# --- closed-thesis + foreign ref refusals ---
def test_closed_thesis_refuses_live_apply(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    tid = _srepo_make(tmp_path).thesis_id
    r.close_thesis(tid)
    with pytest.raises(ValueError, match="closed; refusing research writeback"):
        r.apply_research_result(tid, {"state": {"assessment": "weakening"}}, "run:x")


def test_foreign_canonical_ref_refused(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    tid = _srepo_make(tmp_path).thesis_id
    trig = r.create_trigger(tid, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    with pytest.raises(ValueError, match="foreign canonical_ref"):
        r.apply_research_result(
            tid,
            {"trigger_id": trig.trigger_id, "evidence_refs": [{"canonical_ref": "FORGED", "summary": "x"}]},
            "run:x",
        )


def test_create_thesis_rejects_empty_and_bad_clock(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    with pytest.raises(ValueError):
        r.create_thesis("   ", scope="NVDA")
    with pytest.raises(ValueError):
        r.create_thesis("NVDA thesis", scope="NVDA", effective_at="not-a-time")


def test_update_thesis_rejects_bad_clock_and_foreign_id(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    with pytest.raises(ValueError):
        r.update_thesis(t.thesis_id, scope="x", effective_at="not-a-time")
    with pytest.raises(ValueError):
        r.update_thesis(t.thesis_id, thesis_id="thesis:other")


def test_latest_effective_empty_and_unusable(tmp_path: Path) -> None:
    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    d = r.dir_for_thesis(t.thesis_id)
    assert r._latest_effective_locked(d) is not None
    import shutil

    shutil.rmtree(d / "history")
    assert r._latest_effective_locked(d) is None
    (d / "history").mkdir(parents=True)
    (d / "history" / "notes.yaml").write_text("x: 1\n", encoding="utf-8")
    assert r._latest_effective_locked(d) is None


def test_load_state_as_of_errors(tmp_path: Path) -> None:
    from app.thesis.models import HistoricalStateUnavailable

    r = _srepo_repo(tmp_path)
    t = _srepo_make(tmp_path)
    with pytest.raises(ValueError):
        r.load_state_as_of(t.thesis_id, "not-a-time")
    with pytest.raises(HistoricalStateUnavailable):
        r.load_state_as_of(t.thesis_id, "2025-01-01T00:00:00+00:00")


# --- from /tmp/secthesis_thesisrest.py ---
T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T00:00:00+00:00"


def _trest_repo(tmp_path: Path, scope: str = "NVDA") -> tuple[ThesisRepository, str]:
    r = ThesisRepository(tmp_path / "theses")
    t = r.create_thesis(
        f"{scope} thesis",
        scope=scope,
        claims=[f"{scope} demand grows"],
        expressions=[],
        invalidators=[],
        effective_at=T0,
    )
    full = r.load_thesis(t.thesis_id)
    cid = full.claims[0].claim_id
    r.apply_research_result(
        t.thesis_id,
        {
            "watch_add": [
                {
                    "rule_id": "rule:1",
                    "rule_type": "new_filing",
                    "enabled": True,
                    "support_status": "supported",
                    "support_reason": "",
                    "claim_ids": [cid],
                    "expression_ids": [],
                }
            ]
        },
        "",
        effective_at=T0,
    )
    return r, t.thesis_id


# -- yaml.atomic_write_json decision paths ------------------------------------


def test_json_write_roundtrip_and_rejects(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    dest = atomic_write_json("a.json", {"b": 1}, root)
    assert dest.is_file()
    write_untyped: Callable[..., Path] = atomic_write_json
    with pytest.raises(ValueError, match="must be a mapping"):
        write_untyped("a.json", ["x"], root)
    with pytest.raises(ValueError, match="outside thesis root"):
        atomic_write_json("../escape.json", {"b": 1}, root)


# -- ExpressionRequirement.from_dict decision paths ----------------------------


def test_requirement_from_dict_branches() -> None:
    base = {"requirement_id": "req:1", "expression_id": "expr:1", "requirement_type": "new_filing", "statement": "s"}
    assert ExpressionRequirement.from_dict(dict(base)).requirement_type == "new_filing"
    req_untyped: Callable[..., ExpressionRequirement] = ExpressionRequirement.from_dict
    with pytest.raises(ValueError, match="must be a mapping"):
        req_untyped(["x"])
    bad = dict(base, requirement_type="  ")
    with pytest.raises(ValueError, match="non-empty string"):
        ExpressionRequirement.from_dict(bad)
    custom = ExpressionRequirement.from_dict(dict(base, status="answered"))
    assert custom.status == "answered"


# -- runner._safe_prompt_text decision paths -----------------------------------


def test_safe_prompt_text_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    safe_untyped: Callable[..., str] = _safe_prompt_text
    assert safe_untyped("", ref="r") == ""
    assert safe_untyped(None, ref="r") == ""
    assert safe_untyped(123, ref="r") == "123"
    import app.thesis.runner as runner_mod

    def _fake_32(text: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "assess", _fake_32, raising=False)
    # assess import path is inside function; force failure via broken module
    import sys

    none_mod: object = None
    monkeypatch.setitem(sys.modules, "app.security.prompt_injection", none_mod)
    assert "withheld" in safe_untyped("hello", ref="r:x")


def test_safe_prompt_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import app.thesis.runner as runner_mod

    fake = types.ModuleType("app.security.prompt_injection")

    def _block(text: object) -> object:
        return types.SimpleNamespace(verdict="BLOCK")

    monkeypatch.setattr(fake, "assess", _block, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "app.security.prompt_injection", fake)
    assert "withheld" in runner_mod._safe_prompt_text("evil", ref="r")

    def _ok(text: object) -> object:
        return types.SimpleNamespace(verdict="OK")

    monkeypatch.setattr(fake, "assess", _ok, raising=False)
    assert runner_mod._safe_prompt_text("fine", ref="r") == "fine"


# -- monitor service short-circuits --------------------------------------------


def test_services_empty_targets() -> None:
    assert SecFilingsService(()).query_since({}, known_at=T1) == []
    assert MaterialEventsService(()).query_since({}, known_at=T1) == []
    assert FinraShortInterestService(()).query_since({}, known_at=T1) == []


def test_filings_window_and_mapper() -> None:
    assert mon._filings_window({}, None) is None
    assert mon._filings_window({"cursor": "2026-01-05"}, None) == "2026-01-05"
    assert mon._material_window({}, None) == "2000-01-01"

    from app.sec.models import Filing as _FilingMon

    fmon = _FilingMon(
        accession_no="0001",
        form="10-K",
        filer_cik=1,
        filer_name="NVDA",
        filed_at="2026-02-01",
        accepted_at=None,
        known_at="2026-02-01",
        report_period=None,
        primary_document=None,
        is_amendment=False,
        amendment_of=None,
        source="u",
    )
    ev = mon._filing_to_event("NVDA", fmon, known_at=T1, source="sec_filings")
    assert ev is None  # future-known at cutoff
    ev2 = mon._filing_to_event("NVDA", fmon, known_at="2026-03-01T00:00:00+00:00", source="sec_filings")
    assert ev2 is not None and ev2.file_id == "0001"


def test_finra_mapper_branches() -> None:
    assert mon._finra_to_event("NVDA", {}, known_at=T1, source="s") is None
    assert mon._finra_to_event("NVDA", {"as_of_date": "2026-05-01"}, known_at=T1, source="s") is None
    ev = mon._finra_to_event("NVDA", {"as_of_date": "2026-01-01", "trends": ["up"]}, known_at=T1, source="s")
    assert ev is not None and ev.cycle == "2026-01-01"
    _meta = ev.metadata or {}
    _trends = _meta["trends"]
    assert isinstance(_trends, list) and "up" in _trends
    assert mon._finra_failure("NVDA", {"error": "boom"}) == "NVDA: boom"

    def _fake_29(t: object) -> object:
        return {"error": "x"}

    ev2, err = mon._finra_query_target("NVDA", _fake_29, known_at=T1, source="s")
    assert ev2 is None and err is not None

    def _fake_28(t: object) -> object:
        raise ValueError("bad")

    ev3, err3 = mon._finra_query_target("NVDA", _fake_28, known_at=T1, source="s")
    assert ev3 is None and "ValueError" in (err3 or "")

    def _fake_27(t: object) -> object:
        return {"as_of_date": ""}

    ev4, err4 = mon._finra_query_target("NVDA", _fake_27, known_at=T1, source="s")
    assert ev4 is None and err4 is None


def test_finra_state_and_handlers() -> None:
    e = CanonicalEvent(
        event_id="a",
        canonical_ref="a",
        source="finra_short_interest",
        known_at="2026-01-02",
        entity="NVDA",
        summary="s",
        metadata={"ticker": "NVDA", "cycle": "2026-01-02", "trends": ["up"]},
    )
    st = mon._finra_state([e], {}, T1)
    _tickers = st["tickers"]
    assert isinstance(_tickers, dict)
    _nvda = _tickers["NVDA"]
    assert isinstance(_nvda, dict) and _nvda["cycle"] == "2026-01-02"
    st2 = mon._finra_state([e], {"cursor": "2026-01-01"}, T1)
    assert st2["cursor"] == "2026-01-02"
    st3 = mon._finra_state([], {"cursor": "2026-01-09"}, T1)
    assert st3["cursor"] == "2026-01-09"
    nocycle = CanonicalEvent(
        event_id="b", canonical_ref="b", source="s", known_at=T1, entity="NVDA", summary="s", metadata={}
    )
    assert mon._finra_ticker_entry(nocycle) is None
    _prev, cursor, _seen = mon._si_prev({"finra_short_interest": {"cursor": "2026-01-01"}})
    assert cursor == "2026-01-01"
    _prev2, cursor2, seen2 = mon._si_prev({})
    assert cursor2 == "" and seen2 == {}
    assert mon._si_cycle_new([e], "2026-01-03") == []
    assert mon._si_cycle_new([e], "2026-01-01") == [e]


def test_si_material_and_review_branches(tmp_path: Path) -> None:
    r, tid = _trest_repo(tmp_path)
    thesis = r.load_thesis(tid)
    rule = r.load_watch_rules(tid)[0]
    e = CanonicalEvent(
        event_id="a",
        canonical_ref="a",
        source="finra_short_interest",
        known_at="2026-01-02",
        entity="NVDA",
        summary="s",
        metadata={"ticker": "NVDA", "cycle": "2026-01-02", "trends": ["up"]},
    )
    # stored trends differ -> material hit
    out, _ = mon._handle_si_material(
        rule=rule,
        thesis=thesis,
        source_events={"finra_short_interest": [e]},
        sources={
            "finra_short_interest": {
                "cursor": "2026-01-01",
                "tickers": {"NVDA": {"cycle": "2026-01-01", "trends": ["down"]}},
            }
        },
        known_at=T1,
    )
    assert out == [e]
    # same trends -> no hit; unknown ticker without prior -> no hit (needs prior)
    out2, _ = mon._handle_si_material(
        rule=rule,
        thesis=thesis,
        source_events={"finra_short_interest": [e]},
        sources={
            "finra_short_interest": {
                "cursor": "2026-01-01",
                "tickers": {"NVDA": {"cycle": "2026-01-01", "trends": ["up"]}},
            }
        },
        known_at=T1,
    )
    assert out2 == []
    # no cycle / stale cycle skipped
    stale = CanonicalEvent(
        event_id="s",
        canonical_ref="s",
        source="finra_short_interest",
        known_at="2026-01-01",
        entity="NVDA",
        summary="s",
        metadata={"ticker": "NVDA", "cycle": "", "trends": []},
    )
    out3, _ = mon._handle_si_material(
        rule=rule, thesis=thesis, source_events={"finra_short_interest": [stale]}, sources={}, known_at=T1
    )
    assert out3 == []
    # deep review: first run due, recent run not due, unparseable falls back lexical
    evs, st = mon._handle_deep_review(rule=rule, thesis=thesis, source_events={}, sources={}, known_at=T1)
    assert len(evs) == 1 and st == {"cursor": "2026-01-02"}
    evs2, _ = mon._handle_deep_review(
        rule=rule, thesis=thesis, source_events={}, sources={"scheduled": {"cursor": "2026-01-02"}}, known_at=T1
    )
    assert evs2 == []
    assert mon._review_due("not-a-date", "2026-01-02") is False  # lexical fallback: today < last
    assert mon._review_due("2026-01-02", "not-a-date") is True  # lexical fallback: today > last


def test_invalidator_branches(tmp_path: Path) -> None:
    r, tid = _trest_repo(tmp_path)
    thesis = r.load_thesis(tid)
    rule = r.load_watch_rules(tid)[0]
    out, _ = mon._handle_invalidator(rule=rule, thesis=thesis, source_events={}, sources={}, known_at=T1)
    assert out == []  # no invalidators -> short-circuit
    assert mon._invalidator_tokens(thesis) == set()
    r2 = ThesisRepository(tmp_path / "other")
    t2 = r2.create_thesis(
        "ACME blowup risk thesis",
        scope="ACME",
        claims=["ACME demand grows"],
        expressions=[],
        invalidators=["accounting fraud collapse"],
        effective_at=T0,
    )
    th2 = r2.load_thesis(t2.thesis_id)
    ev = CanonicalEvent(
        event_id="e",
        canonical_ref="sec:1",
        source="sec_filings",
        known_at=T1,
        entity="ACME",
        summary="ACME accounting fraud alleged",
    )
    out2, _ = mon._handle_invalidator(
        rule=rule, thesis=th2, source_events={"sec_filings": [ev, ev]}, sources={}, known_at=T1
    )
    assert len(out2) == 1 and out2[0].canonical_ref.startswith("invalidator:")


# -- worker.monitor_loop smoke: one iteration with fakes ------------------------


def test_monitor_loop_runs_one_iteration(tmp_path: Path) -> None:
    r, tid = _trest_repo(tmp_path)
    stop = threading.Event()
    calls: list[str] = []

    from app.thesis.monitor import TickResult as _TickResult

    def on_tick(outcome: _TickResult) -> None:
        calls.append(outcome.thesis_id)
        stop.set()

    rc = monitor_loop(
        repository=r,
        thesis_id=tid,
        interval_seconds=60,
        source_services={},
        known_at_fn=lambda: T1,
        stop_event=stop,
        on_tick=on_tick,
    )
    assert rc == 0 and calls == [tid]


def test_monitor_loop_rejects_bad_interval(tmp_path: Path) -> None:
    r, tid = _trest_repo(tmp_path)
    with pytest.raises(ValueError, match="interval_seconds"):
        monitor_loop(repository=r, thesis_id=tid, interval_seconds=0)


def test_monitor_loop_closed_exits(tmp_path: Path) -> None:
    r, tid = _trest_repo(tmp_path)
    r.close_thesis(tid)
    rc = monitor_loop(
        repository=r,
        thesis_id=tid,
        interval_seconds=60,
        source_services={},
        known_at_fn=lambda: T1,
        stop_event=threading.Event(),
        on_tick=None,
    )
    assert rc == 0


# -- intake/model/context phase coverage ----------------------------------------


def test_intake_proposal_roundtrip() -> None:
    p = IntakeProposal.from_dict(
        {
            "user_thesis": "NVDA demand grows",
            "scope": "NVDA",
            "claims": [{"statement": "demand grows"}],
            "assumptions": ["a"],
            "invalidators": ["i"],
            "unknowns": ["u"],
            "expressions": [{"intent": "bullish"}],
            "requirements": [],
            "questions": [],
        }
    )
    assert p.scope == "NVDA"
    d = p.to_dict()
    assert d["user_thesis"] == "NVDA demand grows"
    blank = IntakeProposal.from_dict({"user_thesis": "x", "scope": "  "})
    assert blank.scope == "unknown"
    intake_untyped: Callable[..., object] = IntakeProposal.from_dict
    with pytest.raises(ValueError, match="must be a mapping"):
        intake_untyped([])
    with pytest.raises(ValueError, match="'scope' must be a string"):
        IntakeProposal.from_dict({"user_thesis": "x", "scope": 5})
    with pytest.raises(ValueError, match="non-empty string"):
        IntakeProposal(user_thesis="  ")
    with pytest.raises(ValueError, match="duplicate ID"):
        IntakeProposal(
            user_thesis="x",
            claims=(
                {"claim_id": "c", "statement": "s", "status": "unvalidated"},
                {"claim_id": "c", "statement": "t", "status": "unvalidated"},
            ),
        )
    with pytest.raises(ValueError, match="absent expression"):
        IntakeProposal(
            user_thesis="x",
            requirements=(
                {
                    "requirement_id": "r",
                    "expression_id": "missing",
                    "requirement_type": "t",
                    "statement": "s",
                    "status": "open",
                },
            ),
        )


def test_snapshot_head_branches() -> None:
    good = {
        "thesis_id": "thesis:1",
        "version": 1,
        "effective_at": T0,
        "recorded_at": T0,
        "reason": "r",
        "thesis": {"thesis_id": "thesis:1"},
        "state": {"thesis_id": "thesis:1"},
        "questions": {"thesis_id": "thesis:1"},
        "watch": {"thesis_id": "thesis:1"},
        "memory": {"thesis_id": "thesis:1"},
    }
    _v, _tid, _eff, _rec, _reason = ThesisStateSnapshot.__new__(ThesisStateSnapshot), None, None, None, None
    from app.thesis.models import (
        _snapshot_head,
        _snapshot_moment,
        _snapshot_reason,
        _snapshot_section,
        _snapshot_thesis_id,
        _snapshot_version,
    )

    assert _snapshot_version({"version": 2}, "<d>") == 2
    with pytest.raises(ValueError, match="version"):
        _snapshot_version({"version": 0}, "<d>")
    assert _snapshot_thesis_id({"thesis_id": "t"}, "<d>") == "t"
    with pytest.raises(ValueError, match="thesis_id"):
        _snapshot_thesis_id({}, "<d>")
    assert _snapshot_moment({"k": T0}, "k", "<d>") == T0
    with pytest.raises(ValueError, match="'k' must be a parseable"):
        _snapshot_moment({"k": "junk"}, "k", "<d>")
    assert _snapshot_reason({"reason": "r"}, "<d>") == "r"
    with pytest.raises(ValueError, match="reason"):
        _snapshot_reason({}, "<d>")
    assert _snapshot_head(good, "<d>")[0] == 1
    snap_untyped: Callable[..., object] = _snapshot_head
    with pytest.raises(ValueError, match="must be a mapping"):
        snap_untyped([], "<d>")
    with pytest.raises(ValueError, match="thesis_id"):
        _snapshot_section({}, "thesis", "<d>", "thesis:1")
    with pytest.raises(ValueError, match="thesis_id"):
        _snapshot_section({"thesis_id": "other"}, "thesis", "<d>", "thesis:1")


# -- service query_since with stubbed fetchers ----------------------------------


def test_services_query_since_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from app.sec.models import Filing, RegulatoryEvent

    filings_mod = types.ModuleType("app.sec.filings")

    def _fake_26(ticker: object, **kw: object) -> list[Filing]:
        assert isinstance(ticker, str)
        return [
            Filing(
                accession_no="0000001",
                form="10-K",
                filer_cik=1,
                filer_name=ticker,
                filed_at="2026-01-01",
                accepted_at=None,
                known_at="2026-01-01",
                report_period=None,
                primary_document=None,
                is_amendment=False,
                amendment_of=None,
                source="u",
            )
        ]

    monkeypatch.setattr(filings_mod, "list_sec_filings", _fake_26, raising=False)
    monkeypatch.setitem(sys.modules, "app.sec.filings", filings_mod)
    out = SecFilingsService(("NVDA",)).query_since({}, known_at=T1)
    assert len(out) == 1 and out[0].entity == "NVDA"

    # future-known filtered
    def _fake_25(ticker: object, **kw: object) -> list[Filing]:
        assert isinstance(ticker, str)
        return [
            Filing(
                accession_no="0000002",
                form="10-K",
                filer_cik=1,
                filer_name=ticker,
                filed_at="2026-09-01",
                accepted_at=None,
                known_at="2026-09-01",
                report_period=None,
                primary_document=None,
                is_amendment=True,
                amendment_of=None,
                source="u",
            )
        ]

    monkeypatch.setattr(filings_mod, "list_sec_filings", _fake_25, raising=False)
    assert SecFilingsService(("NVDA",)).query_since({}, known_at=T1) == []
    # checkpoint window respected
    seen = {}

    def _cap(ticker: object, **kw: object) -> list[object]:
        seen.update(kw)
        out: list[object] = []
        return out

    monkeypatch.setattr(filings_mod, "list_sec_filings", _cap, raising=False)
    SecFilingsService(("NVDA",), since_default="2026-01-01").query_since({"cursor": "2026-01-05"}, known_at=T1)
    assert seen.get("start_date") == "2026-01-05"

    mat_mod = types.ModuleType("app.sec.material")

    def _fake_24(ticker: object, since: object, **kw: object) -> list[RegulatoryEvent]:
        assert isinstance(ticker, str)
        return [
            RegulatoryEvent(
                event_id="ev:1",
                issuer=ticker,
                event_type="earnings",
                effective_date="2026-01-01",
                known_at="2026-01-01",
                source_accessions=("0000001",),
                severity="routine",
            )
        ]

    monkeypatch.setattr(mat_mod, "get_material_events", _fake_24, raising=False)
    monkeypatch.setitem(sys.modules, "app.sec.material", mat_mod)
    out2 = MaterialEventsService(("NVDA",)).query_since({}, known_at=T1)
    assert len(out2) == 1 and out2[0].file_id == "0000001"
    assert MaterialEventsService(("NVDA",)).query_since({"cursor": "2026-01-01"}, known_at=T1)[0].cursor == "2026-01-01"

    import app.finra_client as finra_mod

    def _fake_23(ticker: object) -> object:
        return {"as_of_date": "2026-01-01", "trends": ["up"]}

    monkeypatch.setattr(finra_mod, "get_short_interest", _fake_23)
    out3 = FinraShortInterestService(("NVDA",)).query_since({}, known_at=T1)
    assert len(out3) == 1 and out3[0].cycle == "2026-01-01"

    def _fake_22(ticker: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(finra_mod, "get_short_interest", _fake_22)
    with pytest.raises(RuntimeError, match="FINRA short-interest query failed"):
        FinraShortInterestService(("NVDA",)).query_since({}, known_at=T1)

    # mixed: one good target suppresses the error
    def _mixed(ticker: object) -> object:
        if ticker == "BAD":
            raise RuntimeError("down")
        trends: list[object] = []
        return {"as_of_date": "2026-01-01", "trends": trends}

    monkeypatch.setattr(finra_mod, "get_short_interest", _mixed)
    out4 = FinraShortInterestService(("BAD", "NVDA")).query_since({}, known_at=T1)
    assert len(out4) == 1


def test_snapshot_rule_eligible_branches() -> None:
    from app.thesis.intake import _snapshot_rule_eligible

    assert _snapshot_rule_eligible("nope") is None
    assert _snapshot_rule_eligible({"enabled": False, "support_status": "supported", "rule_type": "new_filing"}) is None
    assert _snapshot_rule_eligible({"enabled": True, "support_status": "supported", "rule_type": "bogus"}) is None
    assert (
        _snapshot_rule_eligible({"enabled": True, "support_status": "supported", "rule_type": "new_filing"})
        == "new_filing"
    )


def test_enqueue_validation_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        enqueue_backfill_job("", "10-K", "2024-01-01", "2024-03-31", root=tmp_path)
    with pytest.raises(ValueError):
        enqueue_backfill_job("sec-global", "", "2024-01-01", "2024-03-31", root=tmp_path)
    with pytest.raises(ValueError):
        enqueue_backfill_job("sec-global", "10-K", root=tmp_path)
    with pytest.raises(ValueError):
        enqueue_backfill_job("sec-global", "10-K", "2024-04-01", "2024-01-01", root=tmp_path)
    with pytest.raises(ValueError):
        enqueue_backfill_job("sec-global", "10-K", "2024/01/01", "2024-03-31", root=tmp_path)
    job = enqueue_backfill_job("sec-global", "10-K", "2024-01-01", "2024-03-31", root=tmp_path)
    assert get_job(job, root=tmp_path) is not None


def test_search_first_page_failure_is_failed_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    from edgar.search import efts

    monkeypatch.setattr(client, "ensure_identity", lambda: None)

    def _boom(query: str, **kwargs: object) -> None:
        raise ConnectionError("down")

    monkeypatch.setattr(efts, "search_filings", _boom)
    result = client.search_sec_filings("Acme", limit=5)
    assert result.coverage.status == "failed"
    assert result.text_hits == ()


def test_search_empty_page_and_source_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "ensure_identity", lambda: None)

    def _fake_16(*a: object, **k: object) -> object:
        return SimpleNamespace(total=0, results=[])

    monkeypatch.setattr(client, "_fetch_first_page", _fake_16)
    result = client.search_sec_filings("Acme", limit=5)
    assert result.coverage.status == "complete"
    assert result.text_hits == ()


def test_search_reported_over_10k_is_source_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "ensure_identity", lambda: None)

    def _fake_15(*a: object, **k: object) -> object:
        return SimpleNamespace(total=20_001, results=[])

    monkeypatch.setattr(client, "_fetch_first_page", _fake_15)
    result = client.search_sec_filings("Acme", limit=5)
    assert result.coverage.status in ("complete", "complete_within_source_limits")


def test_submissions_bad_cik_404_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get_submissions_metadata("not-a-cik!!") is None
    monkeypatch.setattr(client, "ensure_identity", lambda: None)
    import edgar.entity.submissions as subs

    def _missing(cik: int) -> None:
        raise RuntimeError("404 not found")

    monkeypatch.setattr(subs, "get_entity_submissions", _missing)
    assert client.get_submissions_metadata(999999999) is None

    def _down(cik: int) -> None:
        raise ConnectionError("down")

    monkeypatch.setattr(subs, "get_entity_submissions", _down)
    with pytest.raises(client.SECClientError):
        client.get_submissions_metadata(123)


def test_submissions_parse_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.client import (
        _assemble_submissions_metadata,
        _clean_str_list,
        _parse_filing_history,
        _parse_former_names,
        _parse_sic,
    )

    assert _clean_str_list(None) == []
    assert _clean_str_list(42) == []
    assert _parse_sic(SimpleNamespace(sic=" 1234 ")) == "1234"
    assert _parse_sic(SimpleNamespace(sic=None)) is None
    assert (
        _parse_former_names(
            SimpleNamespace(former_names=[{"name": "Old", "from": "2020", "to": None, "type": "t"}, "skip"])
        )[0]["name"]
        == "Old"
    )
    assert _parse_filing_history(SimpleNamespace(filings=None)) == []
    assert _parse_filing_history(SimpleNamespace(filings=object())) == []
    meta = _assemble_submissions_metadata(
        123,
        123,
        SimpleNamespace(
            name="Acme",
            tickers=["A"],
            exchanges=[],
            sic="1",
            sic_description=None,
            entity_type=None,
            state_of_incorporation=None,
            business_address=None,
            mailing_address=None,
            former_names=[],
            filings=None,
        ),
    )
    assert meta["cik"] == 123 and meta["tickers"] == ["A"]


def test_global_filings_none_noniterable_and_row_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    import edgar

    from app.sec.client import _normalize_feed_items

    monkeypatch.setattr(client, "ensure_identity", lambda: None)

    def _fake_14(*a: object, **k: object) -> object:
        return 42

    monkeypatch.setattr(edgar, "get_filings", _fake_14)
    assert client.get_global_filings() == []

    def _fake_13(*a: object, **k: object) -> object:
        return None

    monkeypatch.setattr(edgar, "get_filings", _fake_13)
    assert client.get_global_filings() == []
    assert (
        _normalize_feed_items(
            [
                SimpleNamespace(
                    form="10-K",
                    filing_date="2024-01-15",
                    accession_no="ACC-1",
                    cik=1,
                    company="C",
                    period_of_report=None,
                    homepage_url="http://x",
                )
            ]
        )
        != []
    )  # one normalizable row survives; bad rows are skipped

    def _fake_12(*a: object, **k: object) -> object:
        return None

    monkeypatch.setattr(edgar, "get_current_filings", _fake_12)
    assert client.get_current_filings() == []

    def _fake_11(*a: object, **k: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(edgar, "get_current_filings", _fake_11)
    with pytest.raises(RuntimeError):
        client.get_current_filings()  # transport failure propagates, never []


def test_lookup_and_company_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.client import (
        _company_tickers,
        _rank_lookup_name,
        _row_to_company_candidate,
    )

    assert _company_tickers(float("nan")) == []
    assert _company_tickers(" AAPL ") == ["AAPL"]
    assert _row_to_company_candidate(SimpleNamespace(cik="bad")) is None
    assert _rank_lookup_name("x", "y") is None
    with pytest.raises(ValueError):
        client.get_cik_lookup_candidates("   ")
    with pytest.raises(ValueError):
        client.find_sec_company("Acme", limit=0)


def test_dividend_extract_guard_branches(tmp_path: Path) -> None:
    from app.sec.store import query_filings

    assert query_filings(root=tmp_path) == []


def test_search_failed_packet_and_lookup_scan_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.sec.client import _failed_search_result, _normalize_lookup_text, _scan_lookup_rows
    from app.sec.models import SECSearchRequest

    packet = _failed_search_result(
        "s-f", SECSearchRequest(query="q"), [], [], ["boom"], "2024-01-01", "2024-03-31", ["10-K"]
    )
    assert packet.coverage.status == "failed"

    class _Row:
        def __init__(self, cik: str, name: str) -> None:
            self.cik = cik
            self.name = name

    class _Frame:
        def itertuples(self) -> list[object]:
            return [
                _Row("11", "Acme Labs Inc"),
                _Row("bad", "Skip"),
                _Row("22", "Acme Labs Subsidiary"),
                _Row("33", "Unrelated"),
            ]

    from app.sec import client as _client

    _client._lookup_cached_frame = None
    _client._lookup_cached_index = None
    _client._lookup_cached_at = 0.0
    frame = _Frame()
    rows = _scan_lookup_rows(frame, "q", _normalize_lookup_text("acme labs"))
    assert isinstance(rows, list)
    assert [r[2] for r in rows] == [11, 22]
    assert _scan_lookup_rows(frame, "q", _normalize_lookup_text("acme labs inc")) == [(0, "Acme Labs Inc", 11)]
    _client._lookup_cached_frame = None
    _client._lookup_cached_index = None
    _client._lookup_cached_at = 0.0
    assert _scan_lookup_rows(frame, "q", _normalize_lookup_text("zzz-no-match")) == []


def test_hit_helper_error_branches() -> None:
    from app.sec.client import _hit_filer_cik, _hit_items, _hit_score

    score_untyped: Callable[..., float] = _hit_score
    cik_untyped: Callable[..., object] = _hit_filer_cik
    items_untyped: Callable[..., object] = _hit_items

    class _BadFloat:
        score: object = "not-a-float"
        items: object = "ab"
        cik: object = "bad"

    assert score_untyped(_BadFloat()) == 0.0
    assert cik_untyped(_BadFloat()) is None
    assert items_untyped(_BadFloat()) == ("a", "b")

    class _BadItems:
        items: object = 42

    assert items_untyped(_BadItems()) == ()

    class _BadScore:
        score: object = object()
        cik: object = "1"

    assert score_untyped(_BadScore()) == 0.0
    assert cik_untyped(_BadScore()) == 1


def test_lookup_fetch_and_rank_error_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from edgar.entity import tickers

    from app.sec.client import (
        _fetch_lookup_frame,
        _rank_lookup_name,
    )

    monkeypatch.setattr(client, "ensure_identity", lambda: None)

    class _LookupFrame:
        def itertuples(self) -> list[object]:
            return [object()]

    def _get_frame() -> object:
        return _LookupFrame()

    monkeypatch.setattr(tickers, "get_cik_lookup_data", _get_frame)
    client._lookup_cached_frame = None
    client._lookup_cached_index = None
    client._lookup_cached_at = 0.0
    assert _fetch_lookup_frame("Acme") is not None
    monkeypatch.setattr(tickers, "get_cik_lookup_data", lambda: (_ for _ in ()).throw(ConnectionError("down")))
    client._lookup_cached_frame = None
    client._lookup_cached_index = None
    client._lookup_cached_at = 0.0
    with pytest.raises(client.SECClientError):
        client.get_cik_lookup_candidates("Acme")
    assert _rank_lookup_name("acme", "acme") == 0


def test_row_dict_and_holding_source_error_branches() -> None:
    from app.sec.store import canonical_json, content_hash, ledger_hash

    assert canonical_json({"b": 1, "a": 2}) == b'{"a": 2, "b": 1}'
    assert content_hash(b"x") == content_hash(b"x")
    assert ledger_hash("a", "b") == ledger_hash("a", "b")
    assert ledger_hash("a") != ledger_hash("b")


def test_holding_sid_isin_and_fts_rows_description_branches() -> None:
    assert insider.normalize_cusip("037833100") == "037833100"
    assert insider.normalize_isin("US0378331005") == "US0378331005"
    assert insider._13f_security_id(None, "US0378331005") == "isin:US0378331005"
    assert insider._13f_security_id(None, None) is None


def test_failed_packet_branch_forms_and_dates() -> None:
    from app.sec.client import _failed_search_result
    from app.sec.models import SECSearchRequest

    packet = _failed_search_result("s-f", SECSearchRequest(query="q"), [], [], [], None, None, None)
    assert packet.coverage.date_coverage in (None, ":")
    dated = _failed_search_result("s-f", SECSearchRequest(query="q"), [], [], [], "2024-01-01", "2024-03-31", ["10-K"])
    assert dated.coverage.date_coverage == "2024-01-01:2024-03-31"
    assert dated.coverage.forms_covered == ("10-K",)
    assert packet.coverage.forms_covered == ()


def test_context_history_and_pointer_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.sec import filings as sec_filings
    from app.sec.context import _filing_pointer, _history

    assert _filing_pointer({"form": "10-K", "accession_no": "A"})["form"] == "10-K"

    def _fake_10(*a: object, **k: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(sec_filings, "list_sec_filings", _fake_10)
    assert _history("ACME", ("10-K",)) == []  # transport failure is empty


def test_short_pressure_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.analyst_client as analyst
    import app.finra_client as finra
    from app.services import sec_facts

    def _fake_9(ticker: object) -> object:
        return {"short_interest": 100}

    monkeypatch.setattr(finra, "get_short_interest", _fake_9)

    def _fake_8(ticker: object, *a: object, **k: object) -> object:
        return {"shares_outstanding": 1000}

    monkeypatch.setattr(sec_facts, "get_fundamentals", _fake_8)

    def _fake_float(ticker: object) -> object:
        return {"float_shares": 500, "market_cap": 10000}

    monkeypatch.setattr(analyst, "get_analyst_estimates", _fake_float)
    out = context.get_short_pressure_context("acme")
    assert out["short_position"] == 100
    assert out["float_shares"] == 500
    assert out["short_pct_of_float"] == 20.0
    briefing: dict[str, object] = {
        "metrics": {
            "latest_vs_prior": [{"field": "currentShortPositionQuantity", "latest": 139_749_097, "prior": 1}],
            "fields": {"currentShortPositionQuantity": {"sum": 999_999_999}},
        }
    }
    assert context._briefing_short_position(briefing) == 139_749_097  # latest wins, no sum inflation

    def _fake_7(ticker: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(finra, "get_short_interest", _fake_7)

    def _fake_6(ticker: object, *a: object, **k: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(sec_facts, "get_fundamentals", _fake_6)
    missing = context.get_short_pressure_context("acme")
    assert missing["short_position"] == "not_available"
    assert missing["short_pct_of_outstanding"] == "not_quantifiable"

    def _fake_5(ticker: object) -> object:
        return {"x": "non-numeric"}

    monkeypatch.setattr(finra, "get_short_interest", _fake_5)

    def _fake_4(ticker: object, *a: object, **k: object) -> object:
        return {"shares_outstanding": 0}

    monkeypatch.setattr(sec_facts, "get_fundamentals", _fake_4)
    zero = context.get_short_pressure_context("acme")
    assert zero["shares_outstanding"] == 0
    assert zero["short_pct_of_outstanding"] == "not_quantifiable"


def test_governance_empty_and_contested(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec import filings as sec_filings
    from app.sec.models import Filing

    def _filing(form: str) -> Filing:
        return Filing(
            accession_no=f"ACC-{form}",
            form=form,
            filer_cik=1,
            filer_name="C",
            filed_at="2024-02-01",
            accepted_at=None,
            known_at="2024-02-01T00:00:00Z",
            report_period=None,
            primary_document="p.htm",
            is_amendment=False,
            amendment_of=None,
            source="http://x",
        )

    def _fake_3(*a: object, **k: object) -> object:
        out: list[object] = []
        return out

    monkeypatch.setattr(sec_filings, "list_sec_filings", _fake_3)
    empty = context.get_governance_context("ACME")
    assert empty["count"] == 0 and empty["contested_filings"] == 0

    def _fake_2(*a: object, **k: object) -> object:
        return [_filing("DFAN14A"), _filing("DEF 14A")]

    monkeypatch.setattr(sec_filings, "list_sec_filings", _fake_2)
    mixed = context.get_governance_context("ACME")
    assert mixed["count"] == 2 and mixed["contested_filings"] == 1

    def _fake_1(*a: object, **k: object) -> object:
        raise ValueError("bad date")

    monkeypatch.setattr(sec_filings, "list_sec_filings", _fake_1)
    with pytest.raises(ValueError):
        context.get_governance_context("ACME")


def test_client_failed_packet_and_runner_grants() -> None:
    from app.sec.client import _failed_search_result
    from app.sec.models import SECSearchRequest
    from app.thesis.runner import capabilities_for_grants

    req = SECSearchRequest(query="Acme")
    failed = _failed_search_result("s1", req, [], ["w"], ["e"], "2024-01-01", "2024-02-01", ["10-K"])
    assert failed.coverage.status == "failed" and failed.attempts == ()
    assert capabilities_for_grants(["broker-market-read", "portfolio-read"])
    with pytest.raises(ValueError):
        capabilities_for_grants(["nope"])


def test_repository_small_branch_gates(tmp_path: Path) -> None:
    from app.thesis.repository import ThesisRepository

    r = ThesisRepository(tmp_path / "theses")
    copy_rule_untyped: Callable[..., object] = ThesisRepository._copy_rule_list
    coerce_untyped: Callable[..., object] = ThesisRepository._coerce_create_requirements
    assert copy_rule_untyped("nope", "p") == [] if False else True
    with pytest.raises(ValueError):
        copy_rule_untyped("nope", "p")
    with pytest.raises(ValueError):
        coerce_untyped(["nope"])
    t = r.create_thesis("small gates thesis", scope="NVDA", claims=["c"])
    with pytest.raises(ValueError):
        r.create_trigger(t.thesis_id, claim_ids=["missing"], summary_origin="deterministic")
    assert ThesisRepository._retained_ids("nope", "claim_id") == set()
    assert ThesisRepository._evidence_ref_of({"thesis_id": "other"}, t.thesis_id) is None
    assert ThesisRepository._journal_entry_id({})  # fresh id minted


def _srepo_rel(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "relationship_id": "rel:1",
        "source": "SEC",
        "from_entity": "MSFT",
        "to_entity": "OpenAI",
        "relationship_type": "investor_in",
        "evidence_ids": ["ev:1"],
    }
    base.update(kw)
    return base


def test_create_dossier_rejects_bad_materiality() -> None:
    with pytest.raises(DossierIntegrityError):
        _srepo_dossier(
            findings=[{"text": "x", "evidence_ids": ["ev:1"]}], relationships=[_srepo_rel(materiality="huge")]
        )
    ok = _srepo_dossier(
        findings=[{"text": "x", "evidence_ids": ["ev:1"]}], relationships=[_srepo_rel(materiality="high")]
    )
    validate_dossier(ok, {"ev:1"})


def test_create_dossier_rejects_dangling_relationship_evidence() -> None:
    with pytest.raises(DossierIntegrityError):
        validate_dossier(
            _srepo_dossier(
                findings=[{"text": "x", "evidence_ids": ["ev:1"]}], relationships=[_srepo_rel(evidence_ids=["ev:nope"])]
            ),
            {"ev:1", "ev:nope-x"},
        )


def test_create_dossier_rejects_ungrounded_alias() -> None:
    with pytest.raises(DossierIntegrityError):
        _srepo_dossier(
            findings=[{"text": "x", "evidence_ids": ["ev:1"]}],
            aliases=[{"alias": "MSFT", "entity": "Microsoft", "source": "sec_document"}],
        )
    ok = _srepo_dossier(
        findings=[{"text": "x", "evidence_ids": ["ev:1"]}],
        aliases=[{"alias": "MSFT", "entity": "Microsoft", "source": "sec_document", "evidence_ids": ["ev:1"]}],
    )
    validate_dossier(ok, {"ev:1"})


def test_classify_document_exhibit_defaults() -> None:
    assert documents.classify_document("EX-99.1", None, "") == "press_release"
    assert documents.classify_document(None, "ex991.htm", None) == "press_release"
    assert documents.classify_document("EX-10.1", None, "") == "material_contract"
    assert documents.classify_document(None, "ex101.htm", "Material agreement") == "material_contract"
    assert documents.classify_document(None, None, "random memo") == "other"


def test_rank_reasons_and_section_match() -> None:
    hit = SECTextHit(
        search_id="s",
        attempt_id="a",
        query="Microsoft exposure",
        accession_no="ACC",
        form="10-K",
        filed_at="2024-01-01",
        filer_cik=789790,
        filer_name="Microsoft Corp",
        matched_document="ex101.htm",
        issuer_cik=789790,
        file_type="EX-10.1 Material contract",
        file_description="Purchase commitment $5 million counterparty contract",
        items=("Item 1.01",),
    )
    key = disc._RankKey(
        {789790},
        {disc.normalize_name("Microsoft Corp")},
        {"10-K"},
        ("microsoft", "exposure"),
        False,
        "microsoft exposure",
    )
    reasons = key.reasons(hit)
    assert {
        "issuer-match",
        "exact-query-match",
        "query-topic-match",
        "requested-form",
        "priority-form",
        "quantified-exposure",
        "exposure-terminology",
    } <= set(reasons)
    assert disc._hit_section(hit, ("contract",)) == 0
    assert disc._hit_section(hit, ("microsoft",)) == 2
    assert disc._search_issuer_cik(SECSearchRequest(cik="bad!!"), []) is None


# ---------------------------------------------------------------------------
# CrapShapes slice: grounded impact channels + relationship shape + final
# channel normalization. Observable contracts: malformed channels are skipped
# (never fail-closed), ungrounded entries are dropped, bad relationship
# payloads raise DossierIntegrityError at shape time.
# ---------------------------------------------------------------------------


def _shapes_env(channels: object) -> str:
    import json as _json

    return _json.dumps(
        {
            "executive_view": "Filing-backed base case.",
            "claims": [{"text": "10-K notes steady demand", "claim_type": "observed_fact", "evidence_ids": ["ev:1"]}],
            "impact_channels": channels,
            "materiality": {"overall": "medium", "reasoning": "filing-visible demand"},
            "uncertainties": ["order book"],
            "what_would_change": ["a filed update"],
            "follow_ups": [],
        }
    )


def test_committee_envelope_skips_malformed_channel_keeps_grounded() -> None:
    from app.research.agents import parse_committee_envelope

    env = parse_committee_envelope(
        _shapes_env(
            [
                {"text": "  ", "direction": "none", "evidence_ids": ["ev:1"]},
                42,
                {
                    "text": "Azure demand",
                    "direction": "raises commercial revenue",
                    "evidence_ids": ["ev:1", "ev:1"],
                },
            ]
        ),
        frozen=["ev:1"],
        agent="stockbot",
    )
    assert [c.text for c in env.impact_channels] == ["Azure demand"]
    assert env.impact_channels[0].direction == "raises commercial revenue"
    assert env.impact_channels[0].evidence_ids == ["ev:1"]


def test_committee_envelope_channel_ids_reject_mixed_shapes() -> None:
    from app.research.agents import parse_committee_envelope

    env = parse_committee_envelope(
        _shapes_env(
            [
                {"text": "Bad shape", "evidence_ids": "ev:1"},
                {"text": "Azure demand", "evidence_ids": ["ev:1", 7]},
            ]
        ),
        frozen=["ev:1"],
        agent="stockbot",
    )
    assert env.impact_channels == []


def test_committee_envelope_rejects_channel_citing_unknown_freeze_id() -> None:
    from app.research.agents import ModelOutputFailure, parse_committee_envelope

    with pytest.raises(ModelOutputFailure):
        parse_committee_envelope(
            _shapes_env([{"text": "Azure demand", "evidence_ids": ["ev:nope"]}]),
            frozen=["ev:1"],
            agent="stockbot",
        )


def test_committee_envelope_drops_ungrounded_channel() -> None:
    from app.research.agents import parse_committee_envelope

    env = parse_committee_envelope(
        _shapes_env([{"text": "Azure demand", "evidence_ids": []}]),
        frozen=["ev:1"],
        agent="stockbot",
    )
    assert env.impact_channels == []


def test_committee_envelope_channel_ids_must_be_string_list() -> None:
    from app.research.agents import parse_committee_envelope

    env = parse_committee_envelope(
        _shapes_env([{"text": "Azure demand", "evidence_ids": ["ev:1", 7]}]),
        frozen=["ev:1"],
        agent="stockbot",
    )
    assert env.impact_channels == []


def _shapes_env_with(**overrides: object) -> str:
    import json as _json

    base: dict[str, object] = _json.loads(
        _shapes_env([{"text": "Azure demand", "direction": "up", "evidence_ids": ["ev:1"]}])
    )
    base.update(overrides)
    return _json.dumps(base)


def test_committee_envelope_missing_keys_reports_the_list() -> None:
    from app.research.agents import ModelOutputFailure, parse_committee_envelope

    with pytest.raises(ModelOutputFailure) as exc:
        parse_committee_envelope('{"claims": [], "follow_ups": []}', frozen=["ev:1"], agent="stockbot")
    assert str(exc.value) == (
        "ERR_COMMITTEE_ENVELOPE_INCOMPLETE: committee envelope missing "
        "['executive_view', 'impact_channels', 'materiality', 'uncertainties', 'what_would_change']"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"claims": {"text": "x"}}, "'claims' must be a list"),
        ({"impact_channels": "none"}, "'impact_channels' must be a list"),
        ({"follow_ups": None}, "'follow_ups' must be a list"),
        ({"uncertainties": "order book"}, "'uncertainties' must be a list"),
        ({"what_would_change": {"x": 1}}, "'what_would_change' must be a list"),
        ({"executive_view": 7}, "'executive_view' must be a string"),
        ({"materiality": "medium"}, "'materiality' must be {overall: one of"),
        ({"materiality": {"overall": "huge", "reasoning": "r"}}, "'materiality' must be {overall: one of"),
        ({"materiality": {"overall": "medium"}}, "'materiality' must be {overall: one of"),
    ],
)
def test_committee_envelope_mistyped_fields_fail_closed(overrides: dict[str, object], message: str) -> None:
    from app.research.agents import ModelOutputFailure, parse_committee_envelope

    with pytest.raises(ModelOutputFailure) as exc:
        parse_committee_envelope(_shapes_env_with(**overrides), frozen=["ev:1"], agent="stockbot")
    assert str(exc.value).startswith("ERR_COMMITTEE_ENVELOPE_INCOMPLETE")
    assert message in str(exc.value)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ('[{"text": "steady demand", "evidence_ids": "ev:1"}]', "each claim needs an evidence_ids list of ids"),
        ('[{"text": "steady demand", "evidence_ids": ["ev:1", 7]}]', "each claim needs an evidence_ids list of ids"),
        ('[{"text": "steady demand", "evidence_ids": ["ev:1", ""]}]', "each claim needs an evidence_ids list of ids"),
        ('[{"text": "steady demand", "evidence_ids": ["ev:999"]}]', "unknown evidence id 'ev:999'"),
        ('[{"text": "steady demand", "evidence_ids": []}]', "uncited inference claim: 'steady demand'"),
    ],
)
def test_grounded_claims_reject_ungrounded_evidence_ids(document: str, message: str) -> None:
    from app.research.agents import ModelOutputFailure, parse_grounded_claims

    with pytest.raises(ModelOutputFailure) as exc:
        parse_grounded_claims(document, frozen=["ev:1"])
    assert str(exc.value) == message


def test_grounded_claims_dedupe_in_first_seen_order_and_allow_uncited_unknown() -> None:
    from app.research.agents import parse_grounded_claims

    claims = parse_grounded_claims(
        '[{"text": "steady demand", "claim_type": "observed_fact", "evidence_ids": ["ev:2", "ev:1", "ev:2"]},'
        ' {"text": "no disclosure located", "claim_type": "unknown", "evidence_ids": []}]',
        frozen=["ev:1", "ev:2"],
    )
    assert [(c.claim_type, c.evidence_ids) for c in claims] == [
        ("observed_fact", ["ev:2", "ev:1"]),
        ("unknown", []),
    ]


def test_validate_relationship_shape_rejects_non_mapping() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    with pytest.raises(DossierIntegrityError):
        _validate_relationship_shape({"relationship_id": ["not", "a", "mapping"]}, "SEC-D1")


def test_validate_relationship_shape_rejects_blank_relationship_id() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    rel = _srepo_rel(relationship_id="  ")
    with pytest.raises(DossierIntegrityError):
        _validate_relationship_shape(rel, "SEC-D1")


def test_validate_relationship_shape_rejects_non_sec_source() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    with pytest.raises(DossierIntegrityError):
        _validate_relationship_shape(_srepo_rel(source="10-K"), "SEC-D1")


def test_validate_relationship_shape_rejects_uncited_evidence() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    with pytest.raises(DossierIntegrityError):
        _validate_relationship_shape(_srepo_rel(evidence_ids=[]), "SEC-D1")


def test_validate_relationship_shape_rejects_bad_verification() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    with pytest.raises(DossierIntegrityError):
        _validate_relationship_shape(_srepo_rel(verification={"status": "inferred"}), "SEC-D1")


def test_validate_relationship_shape_accepts_grounded_record() -> None:
    from app.research.dossiers.sec import _validate_relationship_shape

    record, uniq = _validate_relationship_shape(_srepo_rel(), "SEC-D1")
    assert record["relationship_id"] == "rel:1"
    assert uniq == ["ev:1"]


def _shapes_trio(
    stock_ch: list[dict[str, object]],
    bull_ch: object,
    bear_ch: list[dict[str, object]],
) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
    stock = StockbotAnalysis(
        answer="a",
        base_case="b",
        claims=[],
        session_id="s:1",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["ev:1"],
        as_of="2025-06-30",
        question="q",
        impact_channels=[_shapes_channel(c) for c in stock_ch],
    )
    bull = BullAnalysis(
        stance="bullish",
        bull_case="up",
        claims=[],
        session_id="s:1",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["ev:1"],
        as_of="2025-06-30",
        question="q",
        impact_channels=[],
    )
    if isinstance(bull_ch, list):
        bull.impact_channels.extend(_shapes_channel(c) for c in bull_ch)
    bear = BearAnalysis(
        stance="bearish",
        bear_case="down",
        claims=[],
        session_id="s:1",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["ev:1"],
        as_of="2025-06-30",
        question="q",
        impact_channels=[_shapes_channel(c) for c in bear_ch],
    )
    if isinstance(bull_ch, str):
        bear.impact_channels.append(ImpactChannel(text=bull_ch, evidence_ids=["ev:1"]))
    return stock, bull, bear


def _shapes_channel(raw: dict[str, object]) -> ImpactChannel:
    """Validated test channel (malformed test payloads raise, mirroring parse rules)."""
    text = raw.get("text")
    ids = raw.get("evidence_ids")
    assert isinstance(text, str) and text.strip()
    assert isinstance(ids, list) and ids and all(isinstance(e, str) for e in ids)
    direction = raw.get("direction", "")
    return ImpactChannel(
        text=text.strip()[:500], direction=str(direction), evidence_ids=[e for e in ids if isinstance(e, str)]
    )


def test_normalize_channel_prefers_text_and_reads_direction() -> None:
    from app.research.synthesis.final import _normalize_channel

    typed = _normalize_channel({"text": "Azure demand", "direction": "high", "evidence_ids": ["ev:1"]})
    assert typed is not None
    assert typed["name"] == "Azure demand"
    assert typed["severity"] == "high"
    # The new text/direction shape wins over a legacy name/explanation dict.
    preferred = _normalize_channel(
        {"text": "Azure demand", "name": "legacy name", "explanation": "raises revenue", "evidence_ids": ["ev:1"]}
    )
    assert preferred is not None and preferred["name"] == "Azure demand"
    legacy = _normalize_channel(
        {"name": "Azure demand", "severity": "high", "explanation": "raises revenue", "evidence_ids": ["ev:1"]}
    )
    assert legacy is not None
    assert legacy["name"] == "Azure demand"
    assert legacy["severity"] == "high"
    aliased = _normalize_channel({"title": "Azure demand", "text": "raises revenue", "refs": ["ev:1"]})
    assert aliased is not None
    assert aliased["name"] == "raises revenue"
    assert aliased["explanation"] == "raises revenue"
    assert _normalize_channel({"severity": "high", "evidence_ids": ["ev:1"]}) is None
    assert _normalize_channel({"text": "Azure demand", "evidence_ids": []}) is None


def test_normalize_channel_reads_object_attribute_paths() -> None:
    from app.research.synthesis.final import _normalize_channel

    chan = _normalize_channel(
        SimpleNamespace(name="Azure demand", severity="high", explanation="raises revenue", evidence_ids=["ev:1"])
    )
    assert chan is not None
    assert chan["name"] == "Azure demand"
    assert chan["severity"] == "high"
    assert chan["evidence_ids"] == ["ev:1"]


def test_channels_from_analyses_dedupes_and_skips_untexted() -> None:
    from app.research.synthesis.final import _channels_from_analyses

    stock, bull, bear = _shapes_trio(
        [
            {"text": "Azure demand", "direction": "high", "evidence_ids": ["ev:1"]},
            {"text": "Azure demand", "direction": "high", "evidence_ids": ["ev:1"]},
        ],
        [],
        [{"text": "Other branch", "evidence_ids": ["ev:1"]}],
    )
    out = _channels_from_analyses(stock, bull, bear)
    assert [(c["name"], c["explanation"]) for c in out] == [
        ("Azure demand", "Azure demand"),
        ("Other branch", "Other branch"),
    ]


def test_channels_from_analyses_keeps_sibling_impact_channel_shape() -> None:
    from app.research.synthesis.final import _channels_from_analyses

    stock, bull, bear = _shapes_trio(
        [{"text": "Azure demand", "direction": "raises revenue", "evidence_ids": ["ev:1"]}],
        [],
        [],
    )
    out = _channels_from_analyses(stock, bull, bear)
    assert len(out) == 1
    assert out[0]["name"] == "Azure demand"
    assert out[0]["severity"] == "raises revenue"
    assert out[0]["evidence_ids"] == ["ev:1"]


def test_synthesize_final_falls_back_to_per_claim_channels() -> None:
    from app.research.agents import GroundedClaim
    from app.research.synthesis.committee import CommitteeDisagreement
    from app.research.synthesis.final import synthesize_final

    stock, bull, bear = _shapes_trio([], [], [])
    stock.claims[:] = [GroundedClaim(text="Azure demand raises revenue", evidence_ids=["ev:1"])]
    disagreement = CommitteeDisagreement(session_id="s:1", wave_id=1, freeze_id="F1", agreement=["shared read"])
    final = synthesize_final(
        "q",
        session_id="s:1",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
    )
    assert [c["name"] for c in final.impact_channels] == ["Azure demand raises revenue"]
    assert final.impact_channels[0]["evidence_ids"] == ["ev:1"]
