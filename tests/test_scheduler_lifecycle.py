"""Live research helpers and native tool evidence admission."""

from collections.abc import Mapping
from pathlib import Path
from typing import override
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.research import scheduler as sched
from app.research import tool_runner
from app.research.models import JSONValue, ResearchNode
from app.research.repository import ResearchRepository
from app.runtime import ToolResultMeta
from app.tool_runtime import ToolOutcome

_FINRA_ROW: dict[str, JSONValue] = {
    "settlementDate": "2024-01-01",
    "symbolCode": "XYZ",
    "currentShortPositionQuantity": 123,
}


def _node(node_id: str = "n1", session_id: str = "s1") -> ResearchNode:
    return ResearchNode(node_id=node_id, session_id=session_id, question="q?", why_it_matters="w")


class _Kernel(tool_runner.Kernel):
    """Fake kernel recording lifecycle order; evidence and tool-result stores are in-memory."""

    def __init__(
        self,
        tool_results: Mapping[str, dict[str, JSONValue]] | None = None,
        objective: str = "",
    ) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []
        self.evidence: list[str] = []
        self.decisions: list[tuple[str, object]] = []
        self.tool_results: dict[str, dict[str, JSONValue]] = dict(tool_results or {})
        self.objective = objective
        self.attempts: list[dict[str, JSONValue]] = []
        self.store = mock.Mock(spec=ResearchRepository)
        self.store.list_nodes.return_value = [_node(), _node("n0")]

    @override
    def _store(self) -> ResearchRepository:
        return self.store

    @override
    def start_job(
        self,
        session_id: str,
        type: str = "source_agent",
        source: str | None = None,
        *,
        owner: str = "kernel",
        request_id: str | None = None,
    ) -> dict[str, JSONValue]:
        jid = f"job-{len(self.calls)}"
        self.calls.append(("start", source))
        return {"job_id": jid}

    @override
    def heartbeat_job(self, job_id: str) -> None:
        pass

    @override
    def limit_job_deadline(self, job_id: str, deadline_at: float) -> None:
        pass

    @override
    def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
        assert job_id.startswith("job-"), job_id
        self.calls.append(("admit", job_id))
        eid = f"ev-{len(self.evidence)}"
        self.evidence.append(eid)
        return {"evidence_id": eid}

    @override
    def complete_job(self, job_id: str, outcome: Mapping[str, object] | None = None) -> None:
        self.calls.append(("complete", job_id))

    @override
    def fail_job(self, job_id: str, category: str, message: str) -> None:
        self.calls.append(("fail", job_id, category))

    @override
    def get_session(self, session_id: str) -> dict[str, JSONValue]:
        return {
            "session_id": session_id,
            "objective": self.objective,
            "query": "",
            "as_of": None,
        }

    @override
    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
        return []

    @override
    def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
        return self.tool_results[tool_result_id]


def _outcome(content: str = "record values here", error: str | None = None) -> ToolOutcome:
    return ToolOutcome(
        tool_name="query_finra",
        content=content,
        source_handle=None,
        source_refs=None,
        error=error,
        error_type=None,
        retryable=False,
        meta=ToolResultMeta(0, None, False, None, [], {}),
    )


def _resolve_nvda(name: str) -> str | None:
    return {"NVDA": "NVDA"}.get(name)


_LISTED_8K, _LISTED_10Q = "0001045810-26-000078", "0001045810-26-000075"


def test_finra_latest_record_fans_out_verified_datapoints_the_gate_accepts(tmp_path: Path) -> None:
    """Each numeric latest_record field becomes a verified datapoint row; envelope counts never do."""
    import json

    from app.research import service
    from app.research.agents import parse_grounded_claims
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research(
        "TSLA short interest?", policy={"mode": "allowlist", "sources": ["SEC", "FINRA"]}, repo=store
    )
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="FINRA", repo=store)["job_id"])
    row: dict[str, JSONValue] = {
        "symbolCode": "TSLA",
        "settlementDate": "2026-10-15",
        "currentShortPositionQuantity": 69350469,
        "daysToCoverQuantity": 2.22,
    }
    payload: dict[str, JSONValue] = {
        "dataset_id": "otcMarket/consolidatedShortInterest",
        "query": {"ticker": "TSLA"},
        "briefing": None,
        "metrics": {"fields": {}},
        "latest_record": row,
        "returned_count": 5,
    }
    ref = service.persist_tool_result(sid, jid, "get_short_interest", None, payload, repo=store)["tool_result_id"]
    result: dict[str, object] = {**payload, "tool_result_id": ref}
    outcome = outcome_from_result("get_short_interest", result)
    ids = tool_runner.admit_tool_result(kernel, sid, jid, "get_short_interest", result, outcome)
    eid = ids[0] if ids else None
    assert eid is not None
    points = {
        str(d["metric"]): (e, d)
        for e in store.list_evidence(sid)
        if (d := datapoint_descriptor(evidence_from_dict(e))) is not None
    }
    assert {m: d["value"] for m, (_, d) in points.items()} == {
        "currentShortPositionQuantity": "69350469",
        "daysToCoverQuantity": "2.22",
    }
    row_ev, point = points["currentShortPositionQuantity"]
    claim = {
        "text": "TSLA reported currentShortPositionQuantity of 69350469 for 2026-10-15.",
        "claim_type": "observed_fact",
        "evidence_ids": [row_ev["evidence_id"]],
        "datapoint_ids": [point["id"]],
    }
    claims = parse_grounded_claims(json.dumps([claim]), frozen=[str(row_ev["evidence_id"])])
    service.require_verified_claims(store, sid, claims, "assess_final")


def test_alt_candidate_admits_one_row_without_datapoint_fanout(tmp_path: Path) -> None:
    """An admitting ALT attempt stores the cited row as alt_record evidence and fans out no datapoints."""
    from app.research import service
    from app.research.evidence import evidence_from_dict
    from app.tool_runtime import outcome_from_result

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("weight loss demand?", policy={"mode": "allowlist", "sources": ["ALT"]}, repo=store)
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="ALT", repo=store)["job_id"])
    row: dict[str, JSONValue] = {
        "signal_id": "sig-1",
        "term": "weight loss drug",
        "geo": "US",
        "period": "2026-10-26",
        "rank": 12,
        "known_at": "2026-10-26T00:00:00+00:00",
    }
    payload: dict[str, JSONValue] = {
        "status": "ok",
        "source": "google_trends",
        "observations": [dict(row)],
        "rows": [dict(row)],
        "count": 1,
    }
    ref = service.persist_tool_result(sid, jid, "get_trend_evidence", None, payload, repo=store)
    result: dict[str, object] = {**payload, "tool_result_id": ref["tool_result_id"]}
    outcome = outcome_from_result("get_trend_evidence", result)
    ids = tool_runner.admit_tool_result(kernel, sid, jid, "get_trend_evidence", result, outcome)
    eid = ids[0] if ids else None
    assert eid is not None
    rows = store.list_evidence(sid)
    assert [row["evidence_id"] for row in rows] == [eid]
    stored = evidence_from_dict(rows[0])
    assert stored.provenance["kind"] == "alt_record"
    assert stored.known_at is not None and stored.known_at.isoformat() == "2026-10-26T00:00:00+00:00"


def test_sec_listing_admits_every_filing_row_as_citable_evidence(tmp_path: Path) -> None:
    """A 10-K + 10-Q listing stores one citable record per filing, each replaying against the tool result."""
    from app.research import service
    from app.research.evidence import cited_record, datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("Apple's latest 10-K and 10-Q?", repo=store)
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="SEC", repo=store)["job_id"])
    filings: list[dict[str, JSONValue]] = [
        {"form": "10-Q", "accession_no": "0000320193-26-000020", "filed_at": "2026-07-31", "ticker": "AAPL"},
        {"form": "10-Q", "accession_no": "0000320193-26-000011", "filed_at": "2026-05-01", "ticker": "AAPL"},
        {"form": "10-K", "accession_no": "0000320193-25-000079", "filed_at": "2025-10-31", "ticker": "AAPL"},
    ]
    payload: dict[str, JSONValue] = {"identifier": "AAPL", "filings": list[JSONValue](filings), "source": "SEC EDGAR"}
    ref = service.persist_tool_result(sid, jid, "list_sec_filings", None, payload, repo=store)["tool_result_id"]
    result: dict[str, object] = {**payload, "tool_result_id": ref}
    outcome = outcome_from_result("list_sec_filings", result)
    ids = tool_runner.admit_tool_result(kernel, sid, jid, "list_sec_filings", result, outcome)
    eid = ids[0] if ids else None
    assert eid is not None
    # Each filing row also fans out datapoints of its own; the row records are the ones without a descriptor.
    stored = [e for e in (evidence_from_dict(r) for r in store.list_evidence(sid)) if datapoint_descriptor(e) is None]
    identities = [e.provenance.get("record_identity") for e in stored]
    assert sorted(identities, key=str) == sorted((tool_runner._compact_json(f) for f in filings), key=str)
    assert stored[0].evidence_id == eid  # the parent still cites the newest filing
    persisted = {"result": payload}
    assert [cited_record(persisted, str(i)) for i in identities] == filings


def test_sec_listing_row_accession_and_filing_date_datapoints_the_gate_accepts(tmp_path: Path) -> None:
    """A filing row's accession and filing date verify as datapoints; another row's date or accession never does.

    The gate binds each metric label directly before its value, so the claim states the
    filing date before the accession. A form name is not datapoint wording, so it stays out.
    """
    import json

    from app.research import service
    from app.research.agents import parse_grounded_claims
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("Apple's latest 10-K and 10-Q?", repo=store)
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="SEC", repo=store)["job_id"])
    filings: list[dict[str, JSONValue]] = [
        {
            "accepted_at": "2026-07-31T10:01:02Z",
            "accession_no": "0000320193-26-000020",
            "filed_at": "2026-07-31",
            "filer_name": "Apple Inc.",
            "form": "10-Q",
            "subject_cik": "320193",
            "subject_name": "Apple Inc.",
        },
        {
            "accepted_at": "2026-05-01T10:01:02Z",
            "accession_no": "0000320193-26-000011",
            "filed_at": "2026-05-01",
            "filer_name": "Apple Inc.",
            "form": "10-Q",
            "subject_cik": "320193",
            "subject_name": "Apple Inc.",
        },
        {
            "accepted_at": "2025-10-31T10:01:02Z",
            "accession_no": "0000320193-25-000079",
            "filed_at": "2025-10-31",
            "filer_name": "Apple Inc.",
            "form": "10-K",
            "subject_cik": "320193",
            "subject_name": "Apple Inc.",
        },
    ]
    payload: dict[str, JSONValue] = {"identifier": "AAPL", "filings": list[JSONValue](filings), "source": "SEC EDGAR"}
    ref = service.persist_tool_result(sid, jid, "list_sec_filings", None, payload, repo=store)["tool_result_id"]
    result: dict[str, object] = {**payload, "tool_result_id": ref}
    outcome = outcome_from_result("list_sec_filings", result)
    ids = tool_runner.admit_tool_result(kernel, sid, jid, "list_sec_filings", result, outcome)
    eid = ids[0] if ids else None
    assert eid is not None

    ten_k = tool_runner._compact_json(filings[2])
    rows = [evidence_from_dict(row) for row in store.list_evidence(sid)]
    row_eid = next(
        e.evidence_id for e in rows if datapoint_descriptor(e) is None and e.provenance.get("record_identity") == ten_k
    )
    points = {
        str(d["metric"]): (e, d)
        for e in rows
        if (d := datapoint_descriptor(e)) is not None and e.provenance.get("record_identity") == ten_k
    }
    assert {metric: d["value"] for metric, (_, d) in points.items()} == {
        "accession_no": "0000320193-25-000079",
        "filed_at": "2025-10-31",
    }
    assert {(d["entity"], d["period"]) for _, d in points.values()} == {("Apple Inc.", "2025-10-31")}
    # Every datapoint is its own citable row, so a claim cites the 10-K row and the datapoint rows it states.
    cited = [row_eid, *(e.evidence_id for e, _ in points.values())]
    dates = [str(points["filed_at"][1]["id"]), str(points["accession_no"][1]["id"])]
    for text, point_ids, defect in (
        ("Apple Inc. filed on 2025-10-31 with accession number 0000320193-25-000079", dates, None),
        ("Apple Inc. filed 10-K on 2025-10-31 with accession number 0000320193-25-000079", dates, None),
        # "filing date" is the plain-English spelling of filed_at a model writes; without the alias this rejects.
        ("Apple Inc. filing date 2025-10-31, accession number 0000320193-25-000079", dates, None),
        ("Apple Inc. filed on 2025-10-31 with accession number 0000320193-25-999999", dates, "ERR_SOURCE_IDENTITY"),
        ("Apple Inc. filed on 2026-05-01 with accession number 0000320193-25-000079", dates, "ERR_DATAPOINT"),
    ):
        claims = parse_grounded_claims(
            json.dumps(
                [
                    {
                        "text": text,
                        "claim_type": "observed_fact",
                        "evidence_ids": cited,
                        "datapoint_ids": point_ids,
                    }
                ]
            ),
            frozen=cited,
        )
        if defect is None:
            service.require_verified_claims(store, sid, claims, "assess_final")
        else:
            with pytest.raises(ValueError, match=defect):
                service.require_verified_claims(store, sid, claims, "assess_final")


def test_sec_listing_admission_reads_the_session_once_per_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing fan-out costs the same session reads for 4 rows as for 8: no per-row ledger or journal re-read.

    The stored outcome of these rows is pinned by the two listing tests above; this pins the
    cost that made a 50-filing listing (150 rows) spend 20 s: every row used to rebuild the
    session ledger, re-read the ledger for its proof, and reload the whole journal to append one event.
    """
    from app.research import service
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    reads: dict[str, int] = {}
    original_evidence = ResearchRepository.list_evidence
    original_events = ResearchRepository.list_events

    def counting_evidence(repo: ResearchRepository, session_id: str):
        reads["evidence"] = reads.get("evidence", 0) + 1
        return original_evidence(repo, session_id)

    def counting_events(repo: ResearchRepository, session_id: str):
        reads["events"] = reads.get("events", 0) + 1
        return original_events(repo, session_id)

    monkeypatch.setattr(ResearchRepository, "list_evidence", counting_evidence)
    monkeypatch.setattr(ResearchRepository, "list_events", counting_events)

    def admit(row_count: int) -> tuple[int, int]:
        reads.clear()
        store = ResearchRepository(data_root=tmp_path / f"n{row_count}")
        sid = service.create_research("Apple's latest 10-K and 10-Q?", repo=store)
        kernel = tool_runner.Kernel(store)
        jid = str(service.start_job(sid, source="SEC", repo=store)["job_id"])
        filings: list[dict[str, JSONValue]] = [
            {
                "accepted_at": f"2026-0{month}-01T10:01:02Z",
                "accession_no": f"0000320193-26-0000{index:02d}",
                "filed_at": f"2026-0{month}-01",
                "filer_name": "Apple Inc.",
                "form": "10-Q",
                "subject_cik": "320193",
                "subject_name": "Apple Inc.",
            }
            for index, month in enumerate(range(1, row_count + 1))
        ]
        payload: dict[str, JSONValue] = {
            "identifier": "AAPL",
            "filings": list[JSONValue](filings),
            "source": "SEC EDGAR",
        }
        ref = service.persist_tool_result(sid, jid, "list_sec_filings", None, payload, repo=store)["tool_result_id"]
        result: dict[str, object] = {**payload, "tool_result_id": ref}
        outcome = outcome_from_result("list_sec_filings", result)
        ids = tool_runner.admit_tool_result(kernel, sid, jid, "list_sec_filings", result, outcome)
        eid = ids[0] if ids else None
        assert eid is not None
        admission_reads = (reads.get("evidence", 0), reads.get("events", 0))
        rows = [evidence_from_dict(row) for row in store.list_evidence(sid)]
        # One record row plus two listing-fact datapoints per filing.
        assert len(rows) == 3 * row_count
        assert sum(1 for row in rows if datapoint_descriptor(row) is None) == row_count
        events = store.list_events(sid)  # re-verifies the whole hash chain
        accepted = [event for event in events if event.event_type == "evidence.accepted"]
        assert len(accepted) == 3 * row_count
        assert all(event.payload.get("tool_result_id") == ref for event in accepted)
        assert admission_reads[0] <= 1
        return admission_reads

    small_evidence, small_events = admit(4)
    large_evidence, large_events = admit(8)
    assert (small_evidence, small_events) == (large_evidence, large_events)


def test_service_emit_picks_up_a_journal_another_writer_advanced(tmp_path: Path) -> None:
    """A stored journal row this process never appended is rehydrated before the next event (no sequence collision)."""
    from app.research import service
    from app.research.models import JournalEvent, utcnow

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("Apple's latest 10-K and 10-Q?", repo=store)
    last = store.last_event_sequence(sid)
    store.save_event(
        JournalEvent(
            event_id=f"{sid}:jev:foreign",
            session_id=sid,
            sequence=last + 1,
            event_type="tool_result.persisted",
            timestamp=utcnow(),
            actor_type="kernel",
            actor_id="other-process",
            payload={"tool_result_id": "foreign"},
        )
    )
    service._emit(store, sid, "evidence.accepted", {"evidence_id": "ev:foreign"})
    events = store.list_events(sid)  # raises when the chain or the sequence order broke
    assert [event.sequence for event in events] == list(range(1, last + 3))
    assert [event.event_type for event in events[-2:]] == ["tool_result.persisted", "evidence.accepted"]
    assert events[-1].payload == {"evidence_id": "ev:foreign"}


def test_statement_line_items_fan_out_verified_datapoints_the_gate_accepts(tmp_path: Path) -> None:
    """Every get_financial_statements line item becomes a verified datapoint; a wrong period never verifies."""
    import json

    from app.research import service
    from app.research.agents import parse_grounded_claims
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research(
        "NVDA revenue?", policy={"mode": "allowlist", "sources": ["SEC", "FINRA"]}, repo=store
    )
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="SEC", repo=store)["job_id"])
    items: list[JSONValue] = [
        {"ticker": "NVDA", "period": "2026-01-25 (FY)", "Revenue": 215938000000},
        {"ticker": "NVDA", "period": "2025-01-26 (FY)", "Revenue": 130497000000},
        {"ticker": "NVDA", "period": "2026-01-25 (FY)", "Basic": 4.93},
    ]
    payload: dict[str, JSONValue] = {
        "ticker": "NVDA",
        "cik": 1045810,
        "statement_type": "income_statement",
        "text": "label | 2026-01-25 (FY) | 2025-01-26 (FY)\nRevenue | 215938000000 | 130497000000",
        "line_items": items,
    }
    ref = service.persist_tool_result(sid, jid, "get_financial_statements", None, payload, repo=store)
    result: dict[str, object] = {**payload, "tool_result_id": ref["tool_result_id"]}
    outcome = outcome_from_result("get_financial_statements", result)
    ids = tool_runner.admit_tool_result(kernel, sid, jid, "get_financial_statements", result, outcome)
    eid = ids[0] if ids else None
    assert eid is not None
    points = {
        (str(d["metric"]), str(d["period"])): (e, d)
        for e in store.list_evidence(sid)
        if (d := datapoint_descriptor(evidence_from_dict(e))) is not None
    }
    assert {key: d["value"] for key, (_, d) in points.items()} == {
        ("Revenue", "2026-01-25 (FY)"): "215938000000",
        ("Revenue", "2025-01-26 (FY)"): "130497000000",
        ("Basic", "2026-01-25 (FY)"): "4.93",
    }
    row_ev, point = points["Revenue", "2026-01-25 (FY)"]
    for period, ok in (("2026-01-25 (FY)", True), ("2025-01-26 (FY)", False)):
        claim = {
            "text": f"NVDA reported Revenue of 215938000000 for {period}.",
            "claim_type": "observed_fact",
            "evidence_ids": [row_ev["evidence_id"]],
            "datapoint_ids": [point["id"]],
        }
        claims = parse_grounded_claims(json.dumps([claim]), frozen=[str(row_ev["evidence_id"])])
        if ok:
            service.require_verified_claims(store, sid, claims, "assess_final")
        else:
            with pytest.raises(ValueError, match="ERR_DATAPOINT"):
                service.require_verified_claims(store, sid, claims, "assess_final")


def test_sec_passage_rows_fan_out_verified_datapoints_the_gate_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """SEC passage rows verify through admission. Fabricated and ambiguous values stay absent."""
    import json
    import logging
    from time import perf_counter

    import tests.research_source_seam as seam
    from app.research import service
    from app.research.agents import parse_grounded_claims
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.tool_runtime import outcome_from_result

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    seam.install(monkeypatch)
    passage = (
        "NVIDIA Corporation and Subsidiaries\n"
        "Condensed Consolidated Statements of Income\n"
        "(In millions, except per share data)\n"
        "(Unaudited)\n"
        "Three Months Ended | Six Months Ended\n"
        "Jul 26, 2026 | Jul 27, 2025 | Jul 26, 2026 | Jul 27, 2025\n"
        "Revenue | $ | 96,221 | $ | 46,743 | $ | 177,837 | $ | 90,805\n"
        "Ambiguous | 5 | 5 | 6 | 7\n"
        "Ambiguous | 5 | 5 | 6 | 7\n"
        "Net income per share:\n"
        "Basic | $ | 2.47 | $ | 1.08 | $ | 4.87 | $ | 1.85\n"
        "Diluted | $ | 2.46 | $ | 1.08 | $ | 4.85 | $ | 1.84\n"
        "Basic (1) | $ | 2.47 | $ | 1.08 | $ | 4.87 | $ | 1.85"
    )
    handle = seam.handle_for(passage)
    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("NVDA basic EPS?", repo=store)
    kernel = tool_runner.Kernel(store)
    jid = str(service.start_job(sid, source="SEC", repo=store)["job_id"])
    result: dict[str, object] = {"text": passage, "source_handle": handle}
    outcome = outcome_from_result("get_sec_document", result)
    original = tool_runner._admit_passage_datapoints
    elapsed: list[float] = []

    def timed_fanout(
        kernel: tool_runner.Kernel, sid: str, job_id: str, candidate: Mapping[str, object], ids: list[str]
    ) -> None:
        started = perf_counter()
        original(kernel, sid, job_id, candidate, ids)
        elapsed.append((perf_counter() - started) * 1000)

    monkeypatch.setattr(tool_runner, "_admit_passage_datapoints", timed_fanout)
    with caplog.at_level(logging.INFO, logger=tool_runner.logger.name):
        ids = tool_runner.admit_tool_result(kernel, sid, jid, "get_sec_document", result, outcome)
        eid = ids[0] if ids else None
    print(f"SEC sample passage fan-out: {elapsed[0]:.3f} ms")
    assert eid is not None
    points = {
        str(d["metric"]): (e, d)
        for e in store.list_evidence(sid)
        if (d := datapoint_descriptor(evidence_from_dict(e))) is not None
    }
    assert {metric: d["value"] for metric, (_, d) in points.items()} == {
        "Revenue": "96,221",
        "Basic": "2.47",
        "Diluted": "2.46",
        "Basic (1)": "2.47",
    }
    period = "Three Months Ended Jul 26, 2026"
    assert all(d["period"] == period and d["entity"] == "NVIDIA Corporation" for _, d in points.values())
    assert points["Revenue"][1]["unit"] == "millions"
    assert points["Basic"][1]["unit"] == "per share"
    assert "metric=Ambiguous" in caplog.text and "more than once" in caplog.text
    # Each datapoint row carries a one-line label, never the whole passage: personas receive every row's content.
    assert {e["content"] for e, _ in points.values()} == {
        f"{m}: {d['value']} (NVIDIA Corporation, {period})" for m, (_, d) in points.items()
    }
    row_ev, point = points["Basic"]
    claim = {
        "text": f"NVIDIA Corporation reported Basic of 2.47 per share for {period}",
        "claim_type": "observed_fact",
        "evidence_ids": [eid, row_ev["evidence_id"]],
        "datapoint_ids": [point["id"]],
    }
    claims = parse_grounded_claims(json.dumps([claim]), frozen=[eid, str(row_ev["evidence_id"])])
    service.require_verified_claims(store, sid, claims, "assess_final")
    with pytest.raises(ValueError, match="not a whole token"):
        kernel.admit_evidence(
            sid,
            jid,
            {
                "content": passage,
                "matching_passage": passage,
                "source_handle": handle,
                "metric": "Basic",
                "value": "999.99",
                "entity": "NVIDIA Corporation",
                "period": period,
                "unit": "per share",
            },
        )
    assert len(store.list_evidence(sid)) == 5


def test_finra_envelope_identity_is_symbol_distinct_and_replay_valid() -> None:
    """Envelope identity is the whole payload: replay-valid via cited_record, distinct per symbol."""
    from app.research.evidence import cited_record

    def envelope(symbol: str, metrics_sum: int) -> dict[str, JSONValue]:
        return {
            "dataset_id": "otcMarket/consolidatedShortInterest",
            "briefing": None,
            "briefing_source": "deterministic_only",
            "query": {"ticker": symbol, "start_date": None, "end_date": None, "limit": 5, "offset": 0},
            "metrics": {"fields": {"currentShortPositionQuantity": {"sum": metrics_sum}}, "latest_vs_prior": []},
        }

    def identity(payload: dict[str, JSONValue]) -> str:
        persisted: dict[str, JSONValue] = {"result": payload, "tool_name": "get_short_interest"}
        result = {**payload, "tool_result_id": "s1:tr:si"}
        candidate = tool_runner.evidence_candidate(
            "get_short_interest", "FINRA", result, _outcome(), _Kernel({"s1:tr:si": persisted})
        )
        assert candidate is not None
        locator = str(candidate["record_identity"])
        assert locator
        # A model citing this identity must still replay against the persisted result.
        assert cited_record(persisted, locator) == payload
        return locator

    first = identity(envelope("NVDA", 100))
    assert first == identity(envelope("NVDA", 100))  # identical repeated call keeps one identity
    # Two symbols can return identical metrics; the identity still separates them.
    assert first != identity(envelope("QQQ", 100))


def test_sec_candidate_needs_handle_and_text() -> None:
    """SEC boundary pins: missing handle or empty window/content stays uncitable."""
    good_handle = {"tool_result_id": "s1:tr:sec"}
    good_result = {"source_handle": good_handle, "text": "window words"}
    assert tool_runner._sec_locator(tool_runner._persisted_shapes(good_result)) == "window words"
    assert tool_runner._outcome_summary(_outcome("words")) == "words"
    assert tool_runner._sec_evidence_candidate(good_result, _outcome("words"), _Kernel()) is not None
    assert tool_runner._sec_evidence_candidate({"text": "window words"}, _outcome("words"), _Kernel()) is None
    assert tool_runner._sec_evidence_candidate({"source_handle": good_handle}, _outcome(""), _Kernel()) is None


def test_alt_and_macro_candidates_cite_the_first_returned_row() -> None:
    """Trend/macro packets cite one returned row; disabled, error, and zero-row packets stay uncitable."""
    from app.research.evidence import cited_record

    trend_row: dict[str, JSONValue] = {
        "signal_id": "sig-1",
        "source": "google_trends",
        "term": "weight loss drug",
        "geo": "US",
        "period": "2026-10-26",
        "rank": 12,
        "known_at": "2026-10-26T00:00:00+00:00",
    }
    trend: dict[str, JSONValue] = {
        "status": "ok",
        "source": "google_trends",
        "observations": [dict(trend_row)],
        "rows": [dict(trend_row)],
        "count": 1,
    }
    persisted: dict[str, JSONValue] = {"result": trend, "tool_name": "get_trend_evidence"}
    result: dict[str, JSONValue] = {**trend, "tool_result_id": "s1:tr:trend"}
    candidate = tool_runner.evidence_candidate(
        "get_trend_evidence", "ALT", result, _outcome("trend rows"), _Kernel({"s1:tr:trend": persisted})
    )
    assert candidate is not None and candidate["tool_result_id"] == "s1:tr:trend"
    locator = str(candidate["record_identity"])
    assert locator == tool_runner._compact_json(trend_row)
    assert candidate["content"] == "trend rows"
    assert candidate["matching_passage"] == locator
    assert cited_record(persisted, locator) == trend_row

    macro_row: dict[str, JSONValue] = {
        "geo": "US",
        "variable": "UnemploymentRate_Person",
        "facet": "f1",
        "observations": [{"date": "2026-09", "value": 4.1}],
        "known_at": "2026-10-30T00:00:00+00:00",
    }
    macro: dict[str, JSONValue] = {"status": "ok", "source": "datacommons", "series": [dict(macro_row)], "count": 1}
    macro_persisted: dict[str, JSONValue] = {"result": macro, "tool_name": "get_macro_context"}
    macro_result: dict[str, JSONValue] = {**macro, "tool_result_id": "s1:tr:macro"}
    macro_candidate = tool_runner.evidence_candidate(
        "get_macro_context", "MACRO", macro_result, _outcome("macro rows"), _Kernel({"s1:tr:macro": macro_persisted})
    )
    assert macro_candidate is not None
    assert cited_record(macro_persisted, str(macro_candidate["record_identity"])) == macro_row
    assert macro_candidate["content"] == "macro rows"
    assert macro_candidate["matching_passage"] == macro_candidate["record_identity"]
    for status in ("disabled", "error", None, 1):
        rejected: dict[str, JSONValue] = {**macro, "status": status}
        rejected_result: dict[str, JSONValue] = {**rejected, "tool_result_id": "s1:tr:rejected"}
        assert (
            tool_runner.evidence_candidate(
                "get_macro_context",
                "MACRO",
                rejected_result,
                _outcome("macro rows"),
                _Kernel({"s1:tr:rejected": {"result": rejected}}),
            )
            is None
        )

    # Disabled, error, and rowless packets never become evidence.
    rowless: list[tuple[str, dict[str, JSONValue]]] = [
        ("get_macro_context", {"status": "disabled", "source": "datacommons", "reason": "google data disabled"}),
        ("get_macro_context", {"status": "error", "error": "geos required", "error_type": "invalid_params"}),
        ("find_alternative_signals", {"status": "ok", "source": "google", "signals": [], "count": 0}),
        ("get_trend_evidence", {"status": "ok", "source": "google_trends", "observations": [], "rows": [], "count": 0}),
        ("investigate_social_arbitrage_candidate", {"status": "ok", "source": "google", "evidence": {}}),
    ]
    for tool, payload in rowless:
        empty: dict[str, JSONValue] = {**payload, "tool_result_id": "s1:tr:empty"}
        kernel = _Kernel({"s1:tr:empty": {"result": payload}})
        assert tool_runner.evidence_candidate(tool, "ALT", empty, _outcome(), kernel) is None


def test_diff_risk_factors_candidate_cites_the_comparison() -> None:
    """An SEC risk-factor diff cites its whole persisted comparison; the two filings date it."""
    diff: dict[str, JSONValue] = {
        "ticker": "NVDA",
        "latest_filed": "2026-02-25",
        "prior_filed": "2025-02-26",
        "latest_accession": "0001045810-26-000010",
        "prior_accession": "0001045810-25-000010",
        "diff": "--- prior\n+++ latest\n-competition risk\n+export-control risk",
    }
    persisted: dict[str, JSONValue] = {"result": diff, "tool_name": "diff_risk_factors"}
    result: dict[str, JSONValue] = {**diff, "tool_result_id": "s1:tr:diff"}
    candidate = tool_runner.evidence_candidate(
        "diff_risk_factors", "SEC", result, _outcome("risk factors changed"), _Kernel({"s1:tr:diff": persisted})
    )
    assert candidate is not None
    assert candidate["record_identity"] == tool_runner._compact_json(diff)


def test_sec_list_receipt_routes_through_persisted_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_sec_filings source_refs is a catalog receipt, never a handle: it replays by tool_result_id."""
    import tests.research_source_seam as seam
    from app.research.service import _handle_fields, _sec_record_texts
    from app.tool_runtime import outcome_from_result

    filing: dict[str, JSONValue] = {
        "form": "8-K",
        "accession_no": "0000320193-25-000079",
        "url": "https://sec.gov/x",
        "known_at": "2025-05-01",
    }
    listed: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:list",
        "filings": [filing],
        "source": "SEC EDGAR",
    }
    receipt = outcome_from_result("list_sec_filings", dict(listed))
    assert receipt.source_handle is None and receipt.source_refs
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(receipt.source_refs)  # the receipt is no document handle
    persisted: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:list",
        "tool_name": "list_sec_filings",
        "result": listed,
    }
    reads: list[str] = []

    class _StoredKernel(_Kernel):
        @override
        def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
            reads.append(tool_result_id)
            return persisted

    kernel = _StoredKernel()
    cand = tool_runner.evidence_candidate("list_sec_filings", "SEC", listed, receipt, kernel)
    assert cand is not None and "source_handle" not in cand, cand
    assert cand["tool_result_id"] == "s1:tr:list" and reads == ["s1:tr:list"]
    assert cand["record_identity"] in _sec_record_texts(persisted)

    # get_sec_document keeps its canonical handle; a malformed explicit handle still reaches the validator.
    seam.install(monkeypatch)
    passage = "Data center revenue grew."
    handle = seam.handle_for(passage)
    doc: dict[str, object] = {
        "source_handle": handle,
        "text": passage,
        "content": passage,
    }
    doc_cand = tool_runner.evidence_candidate(
        "get_sec_document",
        "SEC",
        doc,
        outcome_from_result("get_sec_document", doc),
        kernel,
    )
    assert doc_cand is not None and doc_cand["source_handle"] == handle
    _handle_fields(handle)
    bogus: dict[str, object] = {
        "source_handle": {"basis": "guessed"},
        "text": passage,
        "content": passage,
    }
    bad = tool_runner.evidence_candidate(
        "get_sec_document",
        "SEC",
        bogus,
        outcome_from_result("get_sec_document", bogus),
        kernel,
    )
    assert bad is not None and bad["source_handle"] == {"basis": "guessed"}
    _bad_handle = bad["source_handle"]
    assert isinstance(_bad_handle, Mapping)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(_bad_handle)


def test_with_company_seeds_ticker_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """'What is happening with Oracle now' resolves via the EDGAR index (live oracle-query defect)."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Oracle": "ORCL", "oracle": "ORCL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._objective_company("What is happening with Oracle now") == (
        "Oracle",
        "ORCL",
    )
    assert sched._objective_subject_ticker("What is happening with Oracle now") == "ORCL"
    repaired = sched._repair_tool_arguments("get_material_events", {}, "What is happening with Oracle now", "s", "n")
    assert repaired == {"company_name": "Oracle", "ticker": "ORCL"}
    assert sched._objective_company("What is happening with oracle now") == (
        "oracle",
        "ORCL",
    )
    assert sched._objective_subject_ticker("What is happening with oracle now") == "ORCL"
    assert sched._objective_company("What drove growth this week?") is None
    assert sched._objective_company("news for growth this week") is None
    assert sched._objective_subject_ticker("news for growth this week") is None


def test_scalar_or_scalar_list_accepts_finite_scalars_only() -> None:
    """Grounded execution values accept scalars and scalar lists; non-finite numbers and maps drop."""
    assert sched._scalar_or_scalar_list("ORCL")
    assert sched._scalar_or_scalar_list(2.5)
    assert sched._scalar_or_scalar_list([1, "a", 2.5, None])
    assert not sched._scalar_or_scalar_list(float("inf"))
    assert not sched._scalar_or_scalar_list([float("nan")])
    assert not sched._scalar_or_scalar_list([])
    assert not sched._scalar_or_scalar_list({"a": 1})


def test_find_sec_entities_seeded_from_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """find_sec_entities seeds a grounded query (ticker-first) instead of erroring on empty args."""
    import app.tools as _tools
    from app.research import scheduler as _sched

    def _resolve(name: str) -> str | None:
        return {"Tesla": "TSLA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    seeded = _sched._fallback_sec_args(
        "find_sec_entities",
        "Did Tesla insiders actually sell shares last quarter, or only file planned-sale notices?",
    )
    assert seeded == {"query": "TSLA"}


# Vowel-free segments keep generated tags from spelling date words (Today, ThisWeek) the guard rightly rejects.


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    st.sampled_from(
        ["10-Q/10-K", "Oracle Corp", "10-Q 10-K", "2026-10-12", "YYYY-MM-DD", "TSLA", "CIK", "SEC"]
        + ["10-Q/A", "10-K/A", "SC 13D", "DEF 14A", "N-PORT", "1-A", "EFFECT"]
    ),
)
def test_prop_generated_forms_are_never_substituted(form: str) -> None:
    """Repair keeps generated forms verbatim; unknown forms fail before EDGAR, known forms reach it."""
    from app import tools
    from app.policy import Capability, RequestContext

    filled = sched._repair_tool_arguments(
        "list_sec_filings",
        {"identifier": "ORCL", "forms": [form]},
        "What is ORCL's latest revenue?",
        "s",
        "n",
    )
    assert filled["forms"] == [form]
    calls: list[object] = []

    def _provider(*args: object, **kwargs: object) -> list[object]:
        calls.append(kwargs.get("forms"))
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tools.sec, "list_sec_filings", _provider)
        result = tools.execute_tool(
            "list_sec_filings",
            dict[str, object](filled),
            "test",
            context=RequestContext("research", frozenset({Capability.RESEARCH})),
        )
    if form in ("10-Q/A", "10-K/A", "SC 13D", "DEF 14A", "N-PORT", "1-A", "EFFECT"):
        assert "error" not in result and calls == [(form,)]
    else:
        assert result["error_type"] == "invalid_tool_arguments" and form in str(result["error"])
        assert calls == []


# ---------------------------------------------------------------------------
# Post-refactor stage coverage: locators, registry manifests, seeds, session loop
# ---------------------------------------------------------------------------


def test_finra_locator_reads_rows_then_envelope_then_nothing() -> None:
    """Row JSON wins; a rows-less envelope cites its full JSON; a payload with neither stays uncitable."""
    row_text = tool_runner._finra_locator({"records": [_FINRA_ROW]})
    assert row_text == tool_runner._compact_json(_FINRA_ROW)
    envelope = {"briefing": "  short   interest  ", "query": {"ticker": "NVDA"}}
    assert tool_runner._finra_locator(envelope) == tool_runner._compact_json(envelope)
    assert tool_runner._finra_locator({}) is None
    assert tool_runner._finra_locator({"briefing": "   ", "records": []}) == tool_runner._compact_json(
        {"briefing": "   ", "records": []}
    )


def test_sec_replay_locator_reads_scalar_rows_and_envelope() -> None:
    """A payload with no mapping row cites its own JSON; bare scalars are never citable fragments."""
    from app.research.evidence import cited_record

    assert (
        tool_runner._sec_replay_locator({"filings": ["10-K 0000320193-25-000079"]})
        == '{"filings": ["10-K 0000320193-25-000079"]}'
    )
    assert tool_runner._sec_replay_locator({"rows": [42]}) == '{"rows": [42]}'
    assert tool_runner._sec_replay_locator({"filings": [None]}) == '{"filings": [null]}'
    assert tool_runner._sec_replay_locator({}) == "{}"
    assert cited_record({"filings": ["10-K 0000320193-25-000079"]}, '{"filings": ["10-K 0000320193-25-000079"]}') == {
        "filings": ["10-K 0000320193-25-000079"]
    }


def test_replay_locators_keep_padded_strings_exact() -> None:
    """Padded table text (a get_financial_statements envelope, a FINRA briefing) still replays byte-exact."""
    from app.research.evidence import cited_record

    statement = {"statement_type": "income_statement", "text": "      concept  ...   value\n0  Revenues  ...  96221"}
    assert cited_record({"result": statement}, tool_runner._sec_replay_locator(statement)) == statement
    row = {"briefing": "short   interest", "symbol": "TSLA"}
    assert cited_record({"result": {"records": [row]}}, tool_runner._finra_locator({"records": [row]})) == row


def test_tool_result_ref_reads_handle_and_result_ids() -> None:
    """A source_handle reference wins over the result-level ids; absent everywhere it stays None."""
    assert tool_runner._tool_result_ref({"source_handle": {"tool_result_id": "h1"}}) == "h1"
    assert tool_runner._tool_result_ref({"source_handle": {"source_handle_id": "h2"}}) == "h2"
    assert tool_runner._tool_result_ref({"result_id": "r1"}) == "r1"
    assert tool_runner._tool_result_ref({"source_handle_id": "r2"}) == "r2"
    assert tool_runner._tool_result_ref({"source_handle": {}, "tool_result_id": " "}) is None
    assert tool_runner._tool_result_ref(None, {"tool_result_ref": "o1"}) == "o1"


def test_build_registry_skips_malformed_entries_and_renders_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed and JEV-excluded entries never reach the registry; named tools render discovery fields."""
    import app.tools as _tools

    def _fake_tools(caps: object) -> list[dict[str, object]]:
        return [
            {"function": "not-a-mapping"},
            {"function": {"name": ""}},
            {"function": {"name": "call_tool"}},
            {
                "function": {
                    "name": "list_sec_filings",
                    "description": "List EDGAR filings.",
                    "parameters": {"required": ["accession_no", "concept"], "properties": {}},
                }
            },
            {"function": {"name": "unregistered_tool", "parameters": {}}},
        ]

    monkeypatch.setattr(_tools, "tools_for_capabilities", _fake_tools)
    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)
    manifests = sched.build_registry()
    assert [m["name"] for m in manifests] == ["list_sec_filings", "unregistered_tool"]
    known = manifests[0]
    assert known["domain"] == "financial_research"
    assert known["description"] == "List EDGAR filings."
    assert known["keyInputs"] == "req(accession_no, concept)"
    assert known["prerequisites"] == "needs accession_no"
    assert known["pitSupport"] == "PIT-scoped"
    assert str(known["purpose"]).strip() and str(known["useWhen"]).strip()
    unknown = manifests[1]
    assert unknown["domain"] == "unknown"
    assert (unknown["purpose"], unknown["outputKind"], unknown["nextTools"]) == ("", "", "")


def test_objective_company_token_reads_possessive_and_insider_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Possessive, multi-word and insider phrases resolve; org words, stop words and bare titles do not."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL", "Tesla": "TSLA", "Acme Systems": "ACME"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._objective_company_token("What are Apple's margins?") == ("Apple", "AAPL")
    assert sched._objective_company_token("What did Tesla insiders sell?") == ("Tesla", "TSLA")
    assert sched._objective_company_token("What are Acme Systems margins?") == ("Acme Systems", "ACME")
    # A leading title-case non-stop word glues into the multi-word match, so no name resolves alone.
    assert sched._objective_company_token("Compare Acme Systems revenue") is None
    assert sched._objective_company_token("What are FINRA short interest figures?") is None
    assert sched._objective_company_token("Growth slowed last quarter") is None
    assert sched._objective_company_token(None) is None


@settings(max_examples=30, derandomize=True, deadline=None)
@given(
    st.lists(st.sampled_from(["AMD", "NVDA"]), min_size=0, max_size=8),
    st.sampled_from(["", "[Today UTC 2026-10-27] "]),
)
def test_prop_node_issuer_requires_one_distinct_subject(labels: list[str], prefix: str) -> None:
    """Duplicate labels preserve one subject. Distinct subjects preserve the session issuer."""
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect={"AMD": "AMD", "NVDA": "NVDA"}.get):
        expected = labels[0] if labels and len(set(labels)) == 1 else "NVDA"
        assert sched._node_subject_ticker(prefix + " and ".join(labels), "Compare NVDA and AMD", "NVDA") == expected


def test_fallback_sec_args_seeds_entities_and_accession(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity queries seed the resolved ticker or the raw question; filing reads seed a stated accession."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._fallback_sec_args("find_sec_entities", "What are Apple's margins?") == {"query": "AAPL"}
    assert sched._fallback_sec_args("find_sec_entities", "Growth slowed last quarter") == {
        "query": "Growth slowed last quarter"
    }
    acc = "0000320193-25-000079"
    assert sched._fallback_sec_args("get_sec_filing", f"read filing {acc}") == {"accession_no": acc}
    assert sched._fallback_sec_args("get_sec_filing", "read the latest filing") is None
    assert sched._fallback_sec_args("list_sec_filings", "What is Apple's latest revenue?") == {
        "identifier": "AAPL",
        "forms": ["10-Q", "10-K"],
    }


def test_scrub_ungrounded_date_args_keeps_every_grounded_date() -> None:
    """Request, scope, session cutoff and Today UTC dates stay; ungrounded dates and relative wording drop."""
    from datetime import UTC, datetime

    today = datetime.now(UTC).date().isoformat()
    keys = ("as_of", "start_date", "end_date")
    grounded: dict[str, JSONValue] = {"as_of": today, "start_date": "2025-09-27", "end_date": "2024-01-01"}
    assert (
        sched._scrub_ungrounded_date_args(
            grounded,
            "What is ORCL's latest revenue?",
            scope={"as_of": "2025-09-27"},
            session_as_of="2024-01-01",
            keys=keys,
        )
        == []
    )
    assert grounded == {"as_of": today, "start_date": "2025-09-27", "end_date": "2024-01-01"}

    dropped: dict[str, JSONValue] = {"as_of": "2020-01-01", "start_date": "this week", "end_date": 20240101}
    assert sched._scrub_ungrounded_date_args(dropped, "What is ORCL's latest revenue?", keys=keys) == [
        "as_of",
        "start_date",
    ]
    assert dropped == {"end_date": 20240101}

    requested: dict[str, JSONValue] = {"as_of": "2023-05-04"}
    assert sched._scrub_ungrounded_date_args(requested, "Compare ORCL as of 2023-05-04") == []
    required: dict[str, JSONValue] = {"as_of": "2020-01-01"}
    assert sched._scrub_ungrounded_date_args(required, "q", required=["as_of"]) == []
    assert required == {"as_of": "2020-01-01"}


def test_repair_seeds_finra_dataset_and_risk_section() -> None:
    """query_finra defaults its dataset; a risk-factor read seeds Item 1A only when it carries no query."""
    assert sched._repair_tool_arguments("query_finra", {}, "q?", "s1", "n1") == {"dataset": "consolidatedShortInterest"}
    assert sched._repair_tool_arguments("query_finra", {"dataset": "regShoDaily"}, "q?", "s1", "n1") == {
        "dataset": "regShoDaily"
    }
    assert sched._repair_tool_arguments("get_sec_document", {}, "What are NVDA's risk factors?", "s1", "n1") == {
        "section": "Item 1A"
    }
    assert sched._repair_tool_arguments(
        "get_sec_document", {"query": "risk"}, "What are NVDA's risk factors?", "s1", "n1"
    ) == {"query": "risk"}


def test_repair_withholds_search_selectors_without_a_subject() -> None:
    """search_sec_filings receives no selector when the objective names no resolvable subject."""
    assert sched._repair_tool_arguments("search_sec_filings", {}, "Growth slowed last quarter", "s1", "n1") == {}


def test_fallback_ticker_args_adds_eps_only_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticker fallback adds the EPS metric only for explicit EPS wording; a subjectless objective withholds."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    assert sched._fallback_sec_args("get_fundamentals", "What was NVDA's EPS last quarter?") == {
        "ticker": "NVDA",
        "metric": "eps",
    }
    assert sched._fallback_sec_args("get_fundamentals", "How did NVDA revenue grow?") == {"ticker": "NVDA"}
    assert sched._fallback_sec_args("get_short_interest", "What is NVDA short interest?") == {"ticker": "NVDA"}
    assert sched._fallback_sec_args("get_insider_activity", "What drove Growth this week?") is None
    assert sched._fallback_sec_args("get_planned_insider_sales", None) is None


"""Scheduler survivor repairs: dependency order, deadlines, resume state, registries, replay scope, carried evidence, tool selection."""


def test_carry_company_token_rejects_leading_stop_phrase() -> None:
    """Leading-stop phrases (What now) never seed company args."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What now") is False
    assert sched._company_token_allowed("How now") is False


def test_carry_company_token_rejects_org_words() -> None:
    """Organization words never seed company args."""
    from app.research import scheduler as sched

    for word in ("FINRA", "SEC", "NYSE", "NASDAQ", "OTC", "EDGAR"):
        assert sched._company_token_allowed(word) is False


def test_carry_company_token_rejects_reserved_short_words() -> None:
    """Reserved short-interest words never seed company args."""
    from app.research import scheduler as sched

    for word in ("SHO", "REG", "EPS"):
        assert sched._company_token_allowed(word) is False
    assert sched._company_token_allowed("AAPL") is True


def test_carry_company_token_rejects_stop_and_suffix() -> None:
    """Stop words (What) and suffixes (Inc) never seed company args."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What") is False
    assert sched._company_token_allowed("Inc") is False
    assert sched._company_token_allowed("Apple") is True


def test_carry_company_token_second_word_stop_still_rejected() -> None:
    """Phrases starting with a stop word stay rejected even when later words are clean."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What Apple") is False
    assert sched._company_token_allowed("How Apple") is False


def test_carry_stale_identifier_keeps_seven_digit_cik() -> None:
    """Six-digit numeric identifiers reseed; seven-digit ones name a real subject."""
    from app.research import scheduler as sched

    assert sched._stale_listing_identifier("123456", "AAPL") is True
    assert sched._stale_listing_identifier("1234567", "AAPL") is False


def test_carry_stale_identifier_reseeds_placeholders() -> None:
    """Blank, YYYY, date-shaped, and org identifiers reseed; real subjects keep their identifier."""
    from app.research import scheduler as sched

    assert sched._stale_listing_identifier("", None) is True
    assert sched._stale_listing_identifier("2025-01-01", None) is True
    assert sched._stale_listing_identifier("FINRA", "AAPL") is True
    assert sched._stale_listing_identifier("AAPL", "AAPL") is False
    assert sched._stale_listing_identifier("MSFT", "AAPL") is True


def test_scrub_drops_empty_as_of() -> None:
    """An empty as_of is ungrounded and drops."""
    filled: dict[str, JSONValue] = {"as_of": ""}
    assert sched._scrub_ungrounded_date_args(filled, "q") == ["as_of"]
    assert filled == {}


def test_scrub_drops_relative_as_of() -> None:
    """A relative as_of is ungrounded and drops."""
    filled: dict[str, JSONValue] = {"as_of": "today"}
    assert sched._scrub_ungrounded_date_args(filled, "What revenue as of today?") == ["as_of"]
    assert filled == {}


def test_scrub_keeps_longer_value_when_day_scoped() -> None:
    """A longer value whose day scope text names stays."""
    filled: dict[str, JSONValue] = {"as_of": "2026-09-01T00:00"}
    assert sched._scrub_ungrounded_date_args(filled, "q", scope={"raw": "window 2026-09-01 report"}) == []
    assert filled == {"as_of": "2026-09-01T00:00"}


def test_scrub_keeps_timestamp_against_scope_day() -> None:
    """A full timestamp whose day matches the scope day stays."""
    filled: dict[str, JSONValue] = {"as_of": "2026-09-01T12:00:00"}
    assert sched._scrub_ungrounded_date_args(filled, "q", scope={"as_of": "2026-09-01"}) == []
    assert filled == {"as_of": "2026-09-01T12:00:00"}


def test_sec_replay_dict_row_keeps_full_json() -> None:
    """A long dict row cites its full canonical JSON, never a truncation."""
    row = {"k": "x" * 2500}
    assert tool_runner._sec_replay_locator({"filings": [row]}) == tool_runner._compact_json(row)


def test_sec_replay_documents_row() -> None:
    """A documents row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"documents": [{"a": 5}]}) == '{"a": 5}'


def test_sec_replay_envelope_keeps_full_json() -> None:
    """A payload with no citable row cites its own full JSON."""
    payload = {"note": "x" * 2500}
    assert tool_runner._sec_replay_locator(payload) == tool_runner._compact_json(payload)


def test_sec_replay_events_row() -> None:
    """An events row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"events": [{"a": 6}]}) == '{"a": 6}'


def test_sec_replay_facts_list_row() -> None:
    """A facts-list row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"facts": [{"a": 3}]}) == '{"a": 3}'


def test_sec_replay_quarterly_eps_row() -> None:
    """A quarterly_eps row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"quarterly_eps": [{"a": 2}]}) == '{"a": 2}'


def test_sec_replay_records_row() -> None:
    """A records row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"records": [{"a": 4}]}) == '{"a": 4}'


def test_sec_replay_scalar_row_cites_full_payload() -> None:
    """Scalar list rows are not citable fragments: the locator names the complete payload JSON."""
    assert tool_runner._sec_replay_locator({"rows": ["x" * 2500]}) == '{"rows": ["' + "x" * 2500 + '"]}'


def test_sec_replay_transactions_row() -> None:
    """A transactions row cites its own JSON, never the envelope."""
    assert tool_runner._sec_replay_locator({"transactions": [{"a": 1}]}) == '{"a": 1}'


def test_revenue_listing_forms_separates_explicit_ten_q() -> None:
    """Latest quarterly asks stay broad; only explicit 10-Q wording narrows to 10-Q."""
    assert sched._revenue_listing_forms("What is ORCL's latest quarterly revenue?") == ["10-Q", "10-K"]
    assert sched._revenue_listing_forms("What drove NVDA revenue last quarter? 10-Q") == ["10-Q"]
    assert sched._revenue_listing_forms("What drove NVDA revenue last quarter?") == ["10-Q", "10-K", "8-K"]
    assert sched._revenue_listing_forms("How many employees?") is None


def test_registry_advertises_sec_reader_identity_and_discovery() -> None:
    entry = next(manifest for manifest in sched.build_registry() if manifest["name"] == "get_sec_filing")
    parameters = entry["parameters"]
    assert isinstance(parameters, dict)
    required = parameters["required"]
    assert isinstance(required, list) and "accession_no" in required
    assert entry["purpose"] and entry["intent"] and entry["pitSupport"]


def test_registry_contains_unique_tool_names() -> None:
    registry = sched.build_registry()
    assert registry
    assert len(registry) == len({str(manifest.get("name")) for manifest in registry})


def test_sched_build_registry_none_capabilities_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools as _tools

    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)

    def _boom(caps: object) -> object:
        raise AssertionError("caps required")

    monkeypatch.setattr(
        _tools,
        "tools_for_capabilities",
        _boom,
    )
    with pytest.raises(AssertionError, match="caps required"):
        sched.build_registry()
    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)


def test_build_registry_from_empty_cache_returns_research_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh registry contains research tools and excludes broker execution tools."""
    from app.policy import Capability
    from app.tools import TOOL_CAPABILITIES

    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)
    names = {entry["name"] for entry in sched.build_registry()}
    assert "list_sec_filings" in names
    assert all(TOOL_CAPABILITIES[str(name)] is Capability.RESEARCH for name in names)


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Compare NVDA, AAPL, NVDA and MSFT.", ["NVDA", "AAPL", "MSFT"]),
        ("compare Apple and Microsoft", ["Apple", "Microsoft"]),
        ("compare Acme Systems versus Apple", ["Acme Systems", "Apple"]),
        ("Apple versus Microsoft", ["Apple", "Microsoft"]),
        ("buy Apple; compare Microsoft; with apple", ["Apple", "Microsoft"]),
        ("buy Apple; compare NVDA and Microsoft; hold AAPL", ["Apple", "NVDA", "Microsoft", "AAPL"]),
        ("What are Apple's margins? What did Tesla insiders sell?", ["Apple", "Tesla"]),
        ("What are Acme Systems margins?", ["Acme Systems"]),
        ("short interest for Apple this week", ["Apple"]),
        ("volume with Tesla now", ["Tesla"]),
        ("Should I buy Apple?", ["Apple"]),
        ("sell Microsoft", ["Microsoft"]),
        ("hold Apple", ["Apple"]),
        ("invest in Acme Systems", ["Acme Systems"]),
        ("how is Apple doing?", ["Apple"]),
        ("tell me about Apple", ["Apple"]),
        ("[Today UTC 2026-10-19] compare AAPL and NVDA", ["AAPL", "NVDA"]),
        ("[Today UTC 2026-10-19] hi", []),
        ("hello, good morning", []),
        ("hi ty", []),
        ("HI TY", ["HI", "TY"]),
        ("What are SEC FINRA REG SHO EPS figures?", []),
        ("Growth slowed last quarter", []),
        ("", []),
    ],
)
def test_objective_chart_candidates_are_ordered_lexical_labels(
    objective: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate extraction preserves explicit labels without calling the resolver."""
    from app import tools

    resolver = mock.Mock(side_effect=AssertionError("Lexical extraction must not resolve a company."))
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", resolver)
    assert sched._objective_chart_candidates(objective) == expected
    resolver.assert_not_called()


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(st.sampled_from(["NVDA", "AAPL", "MSFT", "ORCL", "HI", "TY"]), min_size=1, max_size=12),
    st.sampled_from([" ", "  ", "\t", "\n", " \t "]),
)
def test_prop_chart_ticker_candidates_preserve_first_order_and_deduplicate(labels: list[str], whitespace: str) -> None:
    """Repeated ticker labels retain their first position across whitespace changes."""
    objective = whitespace + ("," + whitespace).join(labels) + whitespace
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("No resolver call allowed.")):
        assert sched._objective_chart_candidates(objective) == list(dict.fromkeys(labels))


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(st.sampled_from(["Apple", "Microsoft", "Acme Systems", "Coca Cola"]), min_size=1, max_size=10),
    st.sampled_from([" ", "  ", "\t", "\n", " \t "]),
)
def test_prop_chart_company_candidates_normalize_whitespace_and_deduplicate(labels: list[str], whitespace: str) -> None:
    """Company labels keep their first order despite case changes and internal whitespace."""
    objective = "; ".join(f"buy {whitespace.join(label.split())}; with {label.lower()} now" for label in labels)
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("No resolver call allowed.")):
        assert sched._objective_chart_candidates(objective) == list(dict.fromkeys(labels))


def _resolve_chart_test_subject(label: str) -> str | None:
    return {"Apple": "AAPL", "MSFT": "MSFT"}.get(label)


@pytest.mark.parametrize(
    ("objective", "supplied", "expected"),
    [
        ("How is it doing?", " nvda ", "NVDA"),
        ("hi", " aapl ", "AAPL"),
        ("", " msft ", "MSFT"),
        ("How is it doing?", "", None),
        ("How is it doing?", " \t ", None),
        ("How is it doing?", None, None),
        ("How is it doing?", 42, None),
        ("buy Apple", "NVDA", "AAPL"),
        ("how is Apple doing?", "NVDA", "AAPL"),
        ("tell me about Apple", "NVDA", "AAPL"),
        ("What are Apple's margins? Compare MSFT", "NVDA", "MSFT"),
        ("buy Unlisted Company", "NVDA", "NVDA"),
        ("compare ZZZZX and Unknown Company", "NVDA", "NVDA"),
    ],
)
def test_objective_subject_prefers_explicit_labels_over_chart_context(
    objective: str, supplied: object, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit subjects retain old precedence and unresolved labels fall back to the chart ticker."""
    from app import tools

    resolver = mock.Mock(side_effect=_resolve_chart_test_subject)
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", resolver)
    assert sched._objective_subject_ticker(objective, supplied) == expected
    assert all(call.args != ("NVDA",) for call in resolver.call_args_list)


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("list_sec_filings", {"identifier": "NVDA"}),
        ("search_sec_filings", {"ticker": "NVDA", "query": "How is it doing?"}),
        ("search_sec_filings_bounded", {"ticker": "NVDA", "query": "How is it doing?"}),
        ("find_sec_entities", {"query": "NVDA"}),
        ("find_sec_entities_bounded", {"query": "NVDA"}),
        ("get_reg_sho_volume", {"ticker": "NVDA"}),
        ("get_short_interest", {"ticker": "NVDA"}),
        ("get_insider_activity", {"ticker": "NVDA"}),
        ("get_planned_insider_sales", {"ticker": "NVDA"}),
        ("get_fundamentals", {"ticker": "NVDA"}),
        ("set_chart_symbol", {"ticker": "NVDA"}),
        ("get_sec_document", None),
        ("get_sec_filing", None),
        ("get_stock_price", None),
    ],
)
def test_fallback_args_use_normalized_chart_context_without_an_explicit_subject(
    tool_name: str, expected: dict[str, JSONValue] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chart context supplies only the selectors each tool accepts, never a filing accession."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", mock.Mock(return_value=None))
    assert sched._fallback_sec_args(tool_name, "How is it doing?", " nvda ") == expected


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("list_sec_filings", {"identifier": "NVDA"}),
        ("search_sec_filings", {"ticker": "NVDA", "query": "buy Unlisted Company"}),
        ("get_reg_sho_volume", {"ticker": "NVDA"}),
        ("set_chart_symbol", {"ticker": "NVDA"}),
        ("find_sec_entities", {"query": "NVDA"}),
    ],
)
def test_fallback_args_use_chart_context_for_unresolved_explicit_subject(
    tool_name: str, expected: dict[str, JSONValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved explicit company falls back to the selected chart ticker."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", mock.Mock(return_value=None))
    assert sched._fallback_sec_args(tool_name, "buy Unlisted Company", "NVDA") == expected


@pytest.mark.parametrize(
    ("objective", "supplied"),
    [("What do you think about the CEO?", "AAPL"), ("Is AI hype hurting it?", "NVDA")],
)
def test_objective_subject_uses_chart_context_for_unresolved_acronyms(
    objective: str, supplied: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acronyms lose to chart context, even when they resolve to a real ticker."""
    from app import tools

    def _resolve_acronym(label: str) -> str | None:
        return {"AI": "AI", "CEO": "CEOO"}.get(label, _resolve_chart_test_subject(label))

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_acronym)
    assert sched._objective_subject_ticker(objective, supplied) == supplied


def test_objective_subject_resolves_acronym_without_chart_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone acronym subjects still resolve when no chart ticker is present."""
    from app import tools

    def _resolve_ai(label: str) -> str | None:
        return {"AI": "AI"}.get(label)

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_ai)
    assert sched._objective_ticker("Is AI hype hurting it?") == "AI"
    assert sched._objective_subject_ticker("Is AI hype hurting it?") == "AI"


_SEC_TOP_HIT: dict[str, dict[str, object]] = {
    "Microsoft": {"name": "MICROSOFT CORP", "tickers": ["MSFT"]},
    "Growth": {"name": "Vireo Growth Inc.", "tickers": ["VREOF"]},
    "Tesla": {"name": "Tesla, Inc.", "tickers": ["TSLA"]},
}


def _sec_top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
    hit = _SEC_TOP_HIT.get(query)
    return [hit] if hit is not None else []


def _no_resolution(label: str) -> str | None:
    return None


@pytest.mark.parametrize(
    ("objective", "supplied", "expected"),
    [
        ("Is Microsoft overvalued?", None, "MSFT"),
        ("Is Microsoft overvalued?", "NVDA", "MSFT"),
        ("Give me the bull and bear case for Tesla", None, "TSLA"),
        ("Which Growth names matter?", None, None),
        ("Which Growth names matter?", "NVDA", "NVDA"),
        ("Is it overvalued?", None, None),
    ],
)
def test_bare_company_name_resolves_only_on_first_word_match(
    objective: str, supplied: str | None, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare name resolves when it starts the top SEC company name; a word inside a name never does."""
    from app import tools
    from app.sec import client

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr(client, "find_sec_company", _sec_top_hit)
    assert sched._objective_subject_ticker(objective, supplied) == expected


def test_bare_company_lookup_failure_falls_back_to_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing company index leaves the chart ticker as the subject."""
    from app import tools
    from app.sec import client

    def _down(query: str, limit: int = 10) -> list[dict[str, object]]:
        raise OSError("index unavailable")

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr(client, "find_sec_company", _down)
    assert sched._objective_subject_ticker("Is Microsoft overvalued?", "NVDA") == "NVDA"
    assert sched._objective_subject_ticker("Is Microsoft overvalued?") is None


def test_objective_subject_candidates_resolve_top_hits_in_mention_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each capitalized mention becomes one (label, SEC name, ticker) candidate in mention order."""
    from app.sec import client

    hits: dict[str, dict[str, object]] = {
        "Google": {"name": "Alphabet Inc.", "tickers": ["GOOGL"]},
        "Tesla": {"name": "Tesla, Inc.", "tickers": ["TSLA"]},
    }

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        return [hits[query]] if query in hits else []

    monkeypatch.setattr(client, "find_sec_company", _top_hit)
    assert sched._objective_subject_candidates("What does Google's cloud do for Tesla?") == [
        ("Google", "Alphabet Inc.", "GOOGL"),
        ("Tesla", "Tesla, Inc.", "TSLA"),
    ]


def test_objective_subject_candidates_skip_stop_words_and_acronyms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop words and role acronyms never reach the SEC index."""
    from app.sec import client

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        pytest.fail(f"stop word looked up: {query}")

    monkeypatch.setattr(client, "find_sec_company", _top_hit)
    assert sched._objective_subject_candidates("What did the CEO say about Growth?") == []


def test_objective_subject_candidates_dedupe_by_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two labels of one issuer keep only the first mention."""
    from app.sec import client

    hits: dict[str, dict[str, object]] = {
        "Alphabet": {"name": "Alphabet Inc.", "tickers": ["GOOGL"]},
        "Google": {"name": "Alphabet Inc.", "tickers": ["GOOGL"]},
    }

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        return [hits[query]] if query in hits else []

    monkeypatch.setattr(client, "find_sec_company", _top_hit)
    assert sched._objective_subject_candidates("Alphabet and Google search") == [("Alphabet", "Alphabet Inc.", "GOOGL")]


def test_objective_subject_candidates_skip_unresolved_and_failed_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed lookup and a row without a name or ticker drop the candidate."""
    from app.sec import client

    rows: dict[str, list[dict[str, object]]] = {
        "Bravo": [{"name": "", "tickers": ["BRA"]}],
        "Charlie": [{"name": "Charlie Inc.", "tickers": []}],
        "Delta": [{"name": "Delta Inc.", "tickers": ["SEC"]}],
    }

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        if query == "Acme":
            raise OSError("index down")
        return rows.get(query, [])

    monkeypatch.setattr(client, "find_sec_company", _top_hit)
    assert sched._objective_subject_candidates("Acme Bravo Charlie Delta") == []


def test_objective_subject_candidates_cap_the_jev_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five distinct capitalized names yield the first four candidates; each costs one JEV question."""
    from app.sec import client

    names = ["Acme", "Bravo", "Charlie", "Delta", "Echo"]
    hits: dict[str, dict[str, object]] = {
        name: {"name": f"{name} Holdings Inc.", "tickers": [name[:3].upper()]} for name in names
    }

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        return [hits[query]] if query in hits else []

    monkeypatch.setattr(client, "find_sec_company", _top_hit)
    assert [ticker for _, _, ticker in sched._objective_subject_candidates(", ".join(names))] == [
        "ACM",
        "BRA",
        "CHA",
        "DEL",
    ]


@pytest.mark.parametrize(
    "tool_name",
    ["get_valuation_metrics", "get_analyst_estimates", "get_material_events", "diff_risk_factors", "get_xbrl_facts"],
)
def test_resolved_subject_seeds_the_ticker_argument(tool_name: str) -> None:
    """A resolved subject reaches these tools as a grounded ticker, so Needle never invents one."""
    schema = sched._schema_for(tool_name, sched.build_registry())
    assert sched._grounded_execution_values(tool_name, schema, "How is it doing?", " nvda ")["ticker"] == "NVDA"


_THESIS_OBJECTIVE = (
    "Refine my NVDA thesis: AI demand actually collapsed — update the position.\n\n"
    "Context: operate on the operator's existing thesis thesis:de8af495-1c88-413b-b435-a948a1cc04ae."
)
_THESIS_ID = "thesis:de8af495-1c88-413b-b435-a948a1cc04ae"


def test_thesis_tool_seeds_the_objective_named_thesis_id() -> None:
    """The objective's one thesis:<uuid> becomes the grounded id; zero or several ids seed nothing."""
    schema = sched._schema_for("thesis_show", sched.build_registry())
    assert sched._grounded_execution_values("thesis_show", schema, _THESIS_OBJECTIVE) == {"id": _THESIS_ID}
    assert sched._fallback_thesis_args("thesis_refine", _THESIS_OBJECTIVE) == {"id": _THESIS_ID}
    assert sched._fallback_thesis_args("thesis_show", "Refine my NVDA thesis.") is None
    several = f"{_THESIS_ID} and thesis:00000000-0000-0000-0000-000000000000"
    assert sched._fallback_thesis_args("thesis_show", several) is None
    assert sched._fallback_thesis_args("get_fundamentals", _THESIS_OBJECTIVE) is None


def test_request_cutoff_seeds_a_pit_scoped_tool_as_of() -> None:
    """A request cutoff becomes the tool's own as_of on a PIT-scoped tool and never on a PIT-blind one."""
    cutoff = "2023-06-30T00:00:00+00:00"
    insider = sched._schema_for("get_insider_activity", sched.build_registry())
    assert sched._grounded_execution_values(
        "get_insider_activity", insider, "Did insiders sell TSLA?", "TSLA", as_of=cutoff
    ) == {
        "ticker": "TSLA",
        "as_of": "2023-06-30",
    }
    assert sched._grounded_execution_values("get_insider_activity", insider, "Did insiders sell TSLA?", "TSLA") == {
        "ticker": "TSLA"
    }
    thesis = sched._schema_for("thesis_show", sched.build_registry())
    assert sched._grounded_execution_values("thesis_show", thesis, _THESIS_OBJECTIVE, None, as_of=cutoff) == {
        "id": _THESIS_ID
    }


def test_listing_repair_drops_a_one_day_start_date() -> None:
    """start_date equal to end_date asks for one filing day and finds nothing; end_date alone bounds the window."""
    filled = sched._repair_tool_arguments(
        "list_sec_filings",
        {"identifier": "AAPL", "start_date": "2024-01-01", "end_date": "2024-01-01", "forms": ["10-K"]},
        "What did AAPL's 10-K say as of 2024-01-01?",
        "s1",
        "n1",
        schema=sched._schema_for("list_sec_filings", sched.build_registry()),
    )
    assert filled == {"identifier": "AAPL", "end_date": "2024-01-01", "forms": ["10-K"]}


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {"evidence": [{"url": " https://example.com/a ", "highlight": " Revenue\n grew. "}]},
            ("https://example.com/a", "Revenue grew."),
        ),
        ({"evidence": [{"source": "Reuters", "highlight": "Revenue grew."}]}, (None, None)),
        ({"evidence": [None, "bad", {"url": 3, "highlight": []}]}, (None, None)),
        ({"evidence": "bad"}, (None, None)),
        (
            {"evidence": [{"url": " ", "highlight": "text"}, {"url": "https://example.com", "highlight": ""}]},
            (None, None),
        ),
    ],
)
def test_web_locator_requires_citable_url_and_passage(
    payload: dict[str, object], expected: tuple[str | None, str | None]
) -> None:
    assert tool_runner._web_locator(payload) == expected


@pytest.mark.parametrize(
    "schema,scope,objective,initial,expected",
    [
        (None, {}, "latest events", {}, {}),
        ({"required": "since"}, {}, "latest events", {}, {}),
        ({"required": []}, {}, "latest events", {}, {}),
        ({"required": ["since"]}, {"start": "2025-01-01"}, "latest events", {}, {}),
        ({"required": ["since"]}, {"end": "2025-06-01"}, "latest events", {}, {}),
        ({"required": ["since"]}, {}, "events since 2025-01-01", {}, {}),
        ({"required": ["since"]}, {}, "latest events", {}, {"since": "2025-05-31"}),
        ({"required": ["since"]}, {}, "latest events", {"since": "2025-05-31"}, {"since": "2025-05-31"}),
    ],
)
def test_seed_since_window_preserves_explicit_scope(
    schema: dict[str, JSONValue] | None,
    scope: dict[str, JSONValue],
    objective: str,
    initial: dict[str, JSONValue],
    expected: dict[str, JSONValue],
) -> None:
    filled = dict(initial)
    sched._seed_since_window(filled, "events", schema, objective, "", scope, "2025-06-30", "s", "n")
    assert filled == expected


def test_seed_since_window_uses_fixed_clock_for_invalid_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    clock = mock.Mock(wraps=datetime)
    clock.now.return_value = datetime(2025, 6, 30, tzinfo=UTC)
    monkeypatch.setattr(sched, "datetime", clock)
    filled: dict[str, JSONValue] = {}
    sched._seed_since_window(filled, "events", {"required": ["since"]}, "events", "", None, "bad", "s", "n")
    assert filled == {"since": "2025-05-31"}
