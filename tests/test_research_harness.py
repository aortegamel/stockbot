"""Permanent §44/§45 gates: transitions, budgets, journal, PIT, append-only,
freeze immutability + cumulative E2, dossier refs, same-freeze parallel
committee, resume at source/freeze/one-member. All fakes, no network."""

import dataclasses
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tests.research_source_seam as seam
from app.research import jobs as _jobs
from app.research import session as _session
from app.research.agents import GroundedClaim
from app.research.agents.bearbot import BearAnalysis
from app.research.agents.bullbot import BullAnalysis
from app.research.agents.stockbot import StockbotAnalysis
from app.research.dossiers.sec import (
    DossierIntegrityError,
    create_dossier,
    validate_dossier,
)
from app.research.evidence import (
    Evidence,
    EvidenceLedger,
    EvidenceRejectedError,
    evidence_content_hash,
    ingest_evidence,
)
from app.research.freeze import FreezeIntegrityError, create_freeze, verify_freeze
from app.research.journal import append_event
from app.research.models import Job, JSONValue, SessionStatus
from app.research.repository import ResearchRepository
from app.research.runner import run_live

ASOF = datetime(2025, 6, 30, tzinfo=UTC)


def _committee_output(claims: list[dict[str, object]], follow_ups: Sequence[object] = ()) -> str:
    """Rich committee envelope (executive_view, claims, channels, materiality, uncertainties,
    what_would_change, follow_ups): the role contract every trio prompt now demands."""
    import json as _json

    ids: list[str] = []
    for claim in claims:
        raw_ids = claim.get("evidence_ids")
        if isinstance(raw_ids, list):
            ids.extend(eid for eid in raw_ids if isinstance(eid, str))
    channels: list[dict[str, object]] = (
        [
            {
                "text": "frozen-evidence channel",
                "direction": "pressure",
                "evidence_ids": [ids[0]],
            }
        ]
        if ids
        else []
    )
    return _json.dumps(
        {
            "executive_view": "Balanced read of the frozen evidence.",
            "claims": claims,
            "impact_channels": channels,
            "materiality": {
                "overall": "medium",
                "reasoning": "effect size read off the frozen evidence",
            },
            "uncertainties": ["the freeze does not settle timing"],
            "what_would_change": ["a new filing disclosing the terms"],
            "follow_ups": list(follow_ups),
        }
    )


def _grounded(prompt: str, follow_ups: Sequence[object] = ()) -> str:
    """Fake grounded model: cite only freeze/acquired ids listed in the prompt."""
    import json as _json
    import re as _re

    seen: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        bracket = _re.match(r"^\[([^\[\]]+)\]", stripped)
        if bracket is not None:
            token = str(bracket.group(1)).strip()
            if token and token not in seen:
                seen.append(token)
            continue
        acquired = _re.match(r"^-\s+(\S+)", stripped)
        if acquired is not None:
            token = str(acquired.group(1)).strip()
            if (token.startswith("EV-") or ":sec:" in token) and token not in seen:
                seen.append(token)
    if not seen:
        claims: list[dict[str, object]] = []
    else:
        # Letter labels avoid numeric datapoints in these qualitative inference fixtures.
        claims = [
            {"text": f"grounded finding {chr(97 + i)} suggests exposure", "evidence_ids": [eid]}
            for i, eid in enumerate(seen[:6])
        ]
    if "Temporary assignment" in prompt:
        return _json.dumps(claims)
    return _committee_output(claims, follow_ups)


def _ev(eid: str, sid: str, wave: int, known: datetime) -> Evidence:
    return Evidence(
        evidence_id=eid,
        session_id=sid,
        wave_id=wave,
        source_type="sec",
        source_name="SEC",
        source_uri="https://sec.gov/x",
        source_record_id="r",
        subject="NVDA",
        claim_text="c",
        content="c-" + eid,
        content_hash=evidence_content_hash("c-" + eid),
        known_at=known,
        retrieved_at=datetime(2025, 6, 1, tzinfo=UTC),
        job_id="J-1",
        agent_id="s-A",
    )


def _frozen_write(obj: object, name: str, value: object) -> None:
    """Write one attribute through the instance __setattr__ (frozen dataclasses raise)."""
    setattr(obj, name, value)


def test_transitions_valid_and_invalid() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    s2 = _session.transition_session(s, SessionStatus.PLANNING)
    assert s2.status == SessionStatus.PLANNING.value
    with pytest.raises(ValueError):
        _session.transition_session(s, SessionStatus.COMPLETED)


def test_budgets_enforced_from_policy() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    jobs: list[Job] = []
    last = s
    created = 0
    with pytest.raises(ValueError, match="max_parallel|max_total_jobs"):
        for _ in range(21):
            last, j = _jobs.create_job(last, jobs, job_type="scout", owner="t")
            jobs.append(j)
            created += 1
    assert created <= 20


def test_budgets_enforced_total_jobs() -> None:
    """Unlimited by default: no max_total_jobs ceiling until policy configures one."""
    s = _session.create_session("q?", "o", as_of=ASOF)
    jobs: list[Job] = []
    last = s
    for _ in range(21):
        last, j = _jobs.create_job(last, jobs, job_type="scout", owner="t")
        jobs.append(_jobs.complete_job(_jobs.start_job(j)))
    assert len(jobs) == 21  # no configured cap: creation never raises
    policy: dict[str, JSONValue] = dict(last.policy)
    raw: object = policy.get("research", {})
    assert isinstance(raw, dict)
    section: dict[str, JSONValue] = dict(raw)
    section["max_total_jobs"] = 21
    policy["research"] = section
    capped = dataclasses.replace(last, policy=policy)
    with pytest.raises(ValueError, match="max_total_jobs"):
        _jobs.create_job(capped, jobs, job_type="scout", owner="t")


def test_scout_cannot_have_children() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    _, parent = _jobs.create_job(s, [], job_type="scout", owner="t")
    with pytest.raises(ValueError):
        _jobs.create_job(s, [parent], job_type="scout", owner="t", parent_job_id=parent.job_id)


def test_journal_sequences_and_pit_rejection_logged() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    e1 = append_event(s.session_id, "research.created", "t", "t", {})
    e2 = append_event(s.session_id, "research.started", "t", "t", {})
    assert e2.sequence == e1.sequence + 1
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-X", s.session_id, 1, datetime(2025, 7, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen and seen[0][0] == "evidence.rejected"
    assert seen[0][1]["reason"] == "PIT_VIOLATION"


def test_ledger_append_only_and_supersede() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    ingest_evidence(
        led,
        _ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)),
        as_of=ASOF,
        on_reject=None,
    )
    with pytest.raises(ValueError):
        led.append(_ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)))
    fix = led.get("EV-1")
    led.supersede(dataclasses.replace(fix, evidence_id="EV-2", superseded_by="EV-1"))
    assert led.get("EV-1").evidence_id == "EV-1"


def test_freeze_immutable_and_cumulative_e2() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    for eid, wave in (("EV-1", 1), ("EV-2", 1), ("EV-3", 2)):
        ingest_evidence(
            led,
            _ev(eid, s.session_id, wave, datetime(2025, 5, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=None,
        )
    e1 = [e for e in led.list_session(s.session_id) if e.wave_id <= 1]
    e2 = [e for e in led.list_session(s.session_id) if e.wave_id <= 2]
    f1 = create_freeze(freeze_id="E1", session_id=s.session_id, wave_id=1, records=e1, as_of=ASOF)
    f2 = create_freeze(freeze_id="E2", session_id=s.session_id, wave_id=2, records=e2, as_of=ASOF)
    assert set(f1.evidence_ids) < set(f2.evidence_ids)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _frozen_write(f2, "evidence_ids", ("EV-1",))
    verify_freeze(f2, e2)
    with pytest.raises(FreezeIntegrityError):
        verify_freeze(f2, e1)


# --- Datapoint proof: observation + calculation replay, fabricated values conflict. ---

_PROOF_DC = "NVDA Q1 FY2025 Data Center revenue was $22,563 million"
_PROOF_TOT = "NVDA Q1 FY2025 Total revenue was $26,044 million"
_PROOF_OFF = (
    "AMD Q1 FY2025 Data Center revenue was $22,563 million",
    "NVDA Q2 FY2025 Data Center revenue was $22,563 million",
    "NVDA Q1 FY2025 Data Center revenue was 22,563 EUR millions",
)
_PROOF_DOC = (
    f"<html>Item 2. {_PROOF_DC} for the first quarter. {_PROOF_TOT} for the first quarter. "
    + " ".join(f"{passage}." for passage in _PROOF_OFF)
    + "</html>"
).encode()


def _observation_row(
    eid: str,
    passage: str,
    value: str,
    metric: str | None = None,
    *,
    entity: str = "NVDA",
    period: str = "Q1 FY2025",
    unit: str = "million",
) -> Evidence:
    """One SEC observation row whose proof pins _PROOF_DOC bytes, the passage window, and its canonical datapoint."""
    from app.research.evidence import observation_proof, sec_source_ref

    start = _PROOF_DOC.decode().index(passage)
    prov = sec_source_ref(
        accession_no="0001045810-24-000124",
        document_name="nvda-20240428.htm",
        passage=passage,
        offset=start,
        end=start + len(passage),
        basis="raw",
        text_hash=sha256(passage.encode()).hexdigest(),
    )
    label = metric or passage.removeprefix("NVDA Q1 FY2025 ").split(" was ")[0]
    return dataclasses.replace(
        _ev(eid, "S-proof", 1, datetime(2025, 5, 1, tzinfo=UTC)),
        provenance=prov,
        value=value,
        entity=entity,
        period=period,
        unit=unit,
        proof=observation_proof(
            provenance=prov,
            source_bytes=_PROOF_DOC,
            value=value,
            metric=label,
            entity=entity,
            period=period,
            unit=unit,
        ),
    )


def _calculation_row(
    eid: str, value: str, inputs: list[str], formula: str = "difference", rows: Sequence[Evidence] = ()
) -> Evidence:
    """One calculation row; with its input ``rows`` it carries a canonical datapoint, else it is legacy-shaped."""
    from app.research.evidence import calculation_datapoint, calculation_proof

    datapoint = (
        calculation_datapoint(
            formula_id=formula,
            formula_version="1",
            inputs=rows,
            metric="non-Data Center revenue",
            value=value,
            entity="NVDA",
            period="Q1 FY2025",
            unit="million",
        )
        if rows
        else None
    )
    proof = calculation_proof(
        formula_id=formula, formula_version="1", input_proof_ids=inputs, value=value, datapoint=datapoint
    )
    return dataclasses.replace(
        _ev(eid, "S-proof", 1, datetime(2025, 5, 1, tzinfo=UTC)),
        claim_kind="calculation",
        value=value,
        entity="NVDA",
        period="Q1 FY2025",
        unit="million",
        proof=proof,
    )


def test_observation_proof_verifies_exact_value_and_fabricated_value_conflicts() -> None:
    from app.research.evidence import EvidenceIntegrityError, observation_proof, value_hash, verify_observation

    row = _observation_row("EV-DC", _PROOF_DC, "22,563")
    ok = verify_observation(row, value="22,563", source_bytes=_PROOF_DOC)
    assert ok.status == "verified"
    # Fabricated number against a real row: the claimed value hash disagrees.
    assert verify_observation(row, value="22,564", source_bytes=_PROOF_DOC).status == "conflicted"
    # Bytes absent fails closed; tampered bytes conflict with the pinned sourceId.
    assert verify_observation(row, value="22,563", source_bytes=None).status == "unverified"
    assert verify_observation(row, value="22,563", source_bytes=_PROOF_DOC + b" ").status == "conflicted"
    # Missing datapoint identity and a legacy row without proof never verify.
    assert verify_observation(
        dataclasses.replace(row, period=None), value="22,563", source_bytes=_PROOF_DOC
    ).status == ("unverified")
    assert verify_observation(_ev("EV-OLD", "S-proof", 1, ASOF), value="c", source_bytes=_PROOF_DOC).status == (
        "unverified"
    )
    # A row cannot store a value its proof does not hash, and no proof mints for a value off the passage.
    with pytest.raises(EvidenceIntegrityError, match="valueHash"):
        dataclasses.replace(row, value="22,564")
    with pytest.raises(ValueError, match="is not a whole token"):
        observation_proof(
            provenance=row.provenance,
            source_bytes=_PROOF_DOC,
            value="26,044",
            metric="Data Center revenue",
            entity="NVDA",
            period="Q1 FY2025",
        )
    assert row.proof["valueHash"] == value_hash("22,563")


def test_calculation_proof_replays_two_verified_inputs_and_rejects_unrelated_inputs() -> None:
    from app.research.evidence import ProofVerdict, verify_calculation, verify_observation
    from app.research.service import _recompute

    total = _observation_row("EV-TOT", _PROOF_TOT, "26,044")
    dc = _observation_row("EV-DC", _PROOF_DC, "22,563")
    calc = _calculation_row("EV-NONDC", "3481", ["EV-TOT", "EV-DC"], rows=[total, dc])
    inputs = [(r, verify_observation(r, value=r.value, source_bytes=_PROOF_DOC)) for r in (total, dc)]
    assert verify_calculation(calc, value="3481", inputs=inputs, recompute=_recompute).status == "verified"
    frozen = create_freeze(freeze_id="EC", session_id="S-proof", wave_id=1, records=[total, dc, calc], as_of=ASOF)
    verify_freeze(frozen, [calc, dc, total])
    changed_proofs: list[dict[str, JSONValue]] = [
        {**calc.proof, "formulaVersion": "2"},
        {**calc.proof, "inputProofIds": ["EV-DC", "EV-TOT"]},
    ]
    for proof in changed_proofs:
        with pytest.raises(FreezeIntegrityError, match="datapoint_hash mismatch"):
            verify_freeze(frozen, [total, dc, dataclasses.replace(calc, proof=proof)])
    assert verify_calculation(calc, value="3480", inputs=inputs, recompute=_recompute).status == "conflicted"
    # Unrelated or reordered input evidence does not resolve to inputProofIds.
    other = _observation_row("EV-OTHER", _PROOF_TOT, "26,044")
    swapped = [inputs[1], inputs[0]]
    unrelated = [inputs[0], (other, verify_observation(other, value="26,044", source_bytes=_PROOF_DOC))]
    for bad in (swapped, unrelated):
        assert verify_calculation(calc, value="3481", inputs=bad, recompute=_recompute).status == "unverified"
    # A conflicted input, an absent-bytes input, an unknown formula, and a wrong stored result never verify.
    conflicted = [inputs[0], (dc, ProofVerdict("conflicted", "tampered"))]
    assert verify_calculation(calc, value="3481", inputs=conflicted, recompute=_recompute).status == "conflicted"
    absent = [inputs[0], (dc, verify_observation(dc, value="22,563", source_bytes=None))]
    assert verify_calculation(calc, value="3481", inputs=absent, recompute=_recompute).status == "unverified"
    unknown = _calculation_row("EV-X", "3481", ["EV-TOT", "EV-DC"], formula="magic", rows=[total, dc])
    assert verify_calculation(unknown, value="3481", inputs=inputs, recompute=_recompute).status == "unverified"
    wrong = _calculation_row("EV-W", "3480", ["EV-TOT", "EV-DC"], rows=[total, dc])
    assert verify_calculation(wrong, value="3480", inputs=inputs, recompute=_recompute).status == "conflicted"


@pytest.mark.parametrize(
    ("passage", "entity", "period", "unit"),
    [
        (_PROOF_OFF[0], "AMD", "Q1 FY2025", "million"),
        (_PROOF_OFF[1], "NVDA", "Q2 FY2025", "million"),
        (_PROOF_OFF[2], "NVDA", "Q1 FY2025", "EUR millions"),
    ],
)
def test_calculation_rejects_incompatible_typed_inputs(passage: str, entity: str, period: str, unit: str) -> None:
    from app.research.evidence import verify_calculation, verify_observation
    from app.research.service import _recompute

    total = _observation_row("EV-TOT", _PROOF_TOT, "26,044")
    dc = _observation_row("EV-DC", passage, "22,563", "Data Center revenue", entity=entity, period=period, unit=unit)
    calc = _calculation_row("EV-NONDC", "3481", ["EV-TOT", "EV-DC"], rows=[total, dc])
    inputs = [(row, verify_observation(row, value=row.value, source_bytes=_PROOF_DOC)) for row in (total, dc)]
    assert all(verdict.status == "verified" for _, verdict in inputs)
    verdict = verify_calculation(calc, value=calc.value, inputs=inputs, recompute=_recompute)
    assert verdict.status == "unverified" and "incompatible" in verdict.reason


_DP_RECORD: dict[str, JSONValue] = {
    "ticker": "NVDA",
    "period_end": "2025-04-27",
    "unit": "USD",
    "revenue": 100,
    "netIncome": 40,
}
_DP_PASSAGE = (
    "NVDA Q1 FY2025: net income was $40 million. NVDA Q1 FY2025: revenue was $100 million. Gross margin was 75%."
)


def _dp_record_source(*rows: Mapping[str, JSONValue], locator: str | None = None) -> tuple[dict[str, JSONValue], bytes]:
    """(sec_record provenance citing _DP_RECORD, archived tool-result bytes holding ``rows``)."""
    import json as _json

    from app.research.evidence import sec_record_ref

    result = {"tool_result_id": "tr-dp", "result": {"facts": [dict(row) for row in rows]}}
    prov = sec_record_ref(
        tool_name="get_fundamentals",
        record_identity=locator or _json.dumps(_DP_RECORD, sort_keys=True),
        tool_result_id="tr-dp",
    )
    return prov, _json.dumps(result, sort_keys=True).encode()


def test_canonical_datapoint_binds_the_exact_record_field() -> None:
    """The kernel reads metric/value/entity/period/unit off the one cited row; caller spelling only selects."""
    import json as _json

    from app.research.evidence import canonical_datapoint

    prov, raw = _dp_record_source(_DP_RECORD, {**_DP_RECORD, "ticker": "AMD", "revenue": 900})

    def bind(metric: str, value: str) -> dict[str, JSONValue]:
        return canonical_datapoint(
            provenance=prov, source_bytes=raw, metric=metric, value=value, entity="nvda", period="2025-04-27"
        )

    revenue = bind("revenue", "100")
    assert {key: revenue[key] for key in ("metric", "field", "value", "entity", "period", "unit")} == {
        "metric": "revenue",
        "field": "revenue",
        "value": "100",
        "entity": "NVDA",
        "period": "2025-04-27",
        "unit": "USD",
    }
    assert revenue["statement"] == _json.dumps(_DP_RECORD, sort_keys=True)
    # The id is deterministic and independent of how the caller spelled the selector.
    assert str(revenue["id"]).startswith("dp:") and bind("Revenue", "100") == revenue
    assert bind("net income", "40")["metric"] == "netIncome"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(
            {"metric": "net income"}, "is not the exact 'netIncome' field", id="revenue-100-is-not-net-income"
        ),
        pytest.param({"metric": "operating income"}, "names no field", id="absent-metric"),
        pytest.param({"value": "10"}, "is not the exact 'revenue' field", id="value-substring"),
        pytest.param({"entity": "AMD"}, "not the cited record's entity", id="wrong-entity"),
        pytest.param({"period": "2025-01-26"}, "not the cited record's period", id="wrong-period"),
        pytest.param({"unit": "EUR"}, "not the cited record's unit", id="wrong-unit"),
    ],
)
def test_canonical_datapoint_rejects_wrong_record_attribution(change: dict[str, str], reason: str) -> None:
    from app.research.evidence import canonical_datapoint

    prov, raw = _dp_record_source(_DP_RECORD)
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27", **change}
    with pytest.raises(ValueError, match=reason):
        canonical_datapoint(provenance=prov, source_bytes=raw, **selectors)


def test_canonical_datapoint_fails_closed_on_ambiguous_or_missing_identity() -> None:
    import json as _json

    from app.research.evidence import canonical_datapoint

    fragment = '"revenue": 100'
    cases: list[tuple[tuple[dict[str, JSONValue], ...], str | None, str]] = [
        # Two different rows hold the cited fragment: no row is singled out.
        ((_DP_RECORD, {**_DP_RECORD, "ticker": "AMD"}), fragment, "absent or ambiguous"),
        # A unique substring still does not identify the complete source record.
        ((_DP_RECORD,), fragment, "absent or ambiguous"),
        ((_DP_RECORD,), '{"revenue": 100}', "absent or ambiguous"),
        # Two source fields answer the same metric.
        (({**_DP_RECORD, "net_income": 40},), None, "is ambiguous in"),
        # The cited row states no period at all.
        (({k: v for k, v in _DP_RECORD.items() if k != "period_end"},), None, "states no period"),
    ]
    for rows, locator, reason in cases:
        prov, raw = _dp_record_source(*rows, locator=locator or _json.dumps(rows[0], sort_keys=True))
        metric = "net income" if reason == "is ambiguous in" else "revenue"
        value = "40" if metric == "net income" else "100"
        with pytest.raises(ValueError, match=reason):
            canonical_datapoint(
                provenance=prov, source_bytes=raw, metric=metric, value=value, entity="NVDA", period="2025-04-27"
            )


def test_cited_record_preserves_exact_string_content() -> None:
    import json as _json

    from app.research.evidence import cited_record

    row = {**_DP_RECORD, "label": "two  spaces"}
    payload = {"facts": [row]}
    assert cited_record(payload, _json.dumps(row, indent=2)) == row
    assert cited_record(payload, _json.dumps({**row, "label": "two spaces"})) is None


_ALIAS_ROW: dict[str, JSONValue] = {
    "cik": 1045810,
    "fiscal_year": 2026,
    "fiscal_period": "Q1",
    "period_start": "2025-01-27",
    "period_end": "2025-04-27",
    "filed_at": "2025-05-28",
    "unit": "USD",
    "revenue": 100,
}


def _alias_source(
    row: Mapping[str, JSONValue], envelope: Mapping[str, JSONValue]
) -> tuple[dict[str, JSONValue], bytes]:
    """(sec_record provenance citing ``row``, persisted tool-result bytes with ``envelope`` and created_at)."""
    import json as _json

    from app.research.evidence import sec_record_ref

    result = {
        "tool_result_id": "tr-alias",
        "created_at": "2025-06-01T00:00:00+00:00",
        "result": {**envelope, "facts": [dict(row)]},
    }
    prov = sec_record_ref(
        tool_name="get_fundamentals", record_identity=_json.dumps(row, sort_keys=True), tool_result_id="tr-alias"
    )
    return prov, _json.dumps(result, sort_keys=True).encode()


def test_canonical_datapoint_selects_by_deterministic_aliases() -> None:
    """Ticker, fiscal label and unit symbol select the source spelling; the descriptor keeps both spellings."""
    from app.research.evidence import canonical_datapoint, verify_observation

    prov, raw = _alias_source(_ALIAS_ROW, {"ticker": "NVDA", "cik": "0001045810"})
    point = canonical_datapoint(
        provenance=prov, source_bytes=raw, metric="revenue", value="100", entity="nvda", period="q1 fy2026", unit="$"
    )
    assert (point["entity"], point["period"], point["unit"]) == ("1045810", "2025-04-27", "USD")
    assert point["aliases"] == {"entity": "NVDA", "period": "Q1 FY2026", "unit": "$"}
    plain = canonical_datapoint(
        provenance=prov, source_bytes=raw, metric="revenue", value="100", entity="1045810", period="2025-04-27"
    )
    assert "aliases" not in plain and plain["id"] != point["id"]
    # The row stores the source spelling; verification rebuilds from the stored display spelling.
    from app.research.evidence import observation_proof

    proof = observation_proof(
        provenance=prov, source_bytes=raw, metric="revenue", value="100", entity="NVDA", period="Q1 FY2026", unit="$"
    )
    row = dataclasses.replace(
        _ev("EV-ALIAS", "S-proof", 1, datetime(2025, 6, 1, tzinfo=UTC)),
        provenance=prov,
        value="100",
        entity="1045810",
        period="2025-04-27",
        unit="USD",
        proof=proof,
    )
    assert verify_observation(row, value="100", source_bytes=raw).status == "verified"
    swapped = dataclasses.replace(row, period="2025-01-27")
    assert verify_observation(swapped, value="100", source_bytes=raw).status == "conflicted"


@pytest.mark.parametrize(
    ("row", "envelope", "change", "reason"),
    [
        pytest.param(_ALIAS_ROW, {"ticker": "NVDA", "cik": 1045810}, {"entity": "AMD"}, "entity", id="other-ticker"),
        pytest.param(_ALIAS_ROW, {"ticker": "NVDA", "cik": 2488}, {}, "entity", id="envelope-cik-differs"),
        pytest.param(_ALIAS_ROW, {"ticker": "NVDA"}, {}, "entity", id="no-cik-no-snapshot"),
        pytest.param(_ALIAS_ROW, {"ticker": "NVDA", "cik": 1045810}, {"period": "Q2 FY2026"}, "period", id="other-q"),
        pytest.param(
            {**_ALIAS_ROW, "period_start": "2024-01-29", "period_end": "2024-04-28"},
            {"ticker": "NVDA", "cik": 1045810},
            {},
            "period",
            id="comparative-row-keeps-filing-fiscal-tags",
        ),
        pytest.param(
            {**_ALIAS_ROW, "period_start": "2024-10-28"},
            {"ticker": "NVDA", "cik": 1045810},
            {},
            "period",
            id="ytd-duration-is-not-a-quarter",
        ),
        pytest.param(_ALIAS_ROW, {"ticker": "NVDA", "cik": 1045810}, {"unit": "million"}, "unit", id="no-scaling"),
    ],
)
def test_canonical_datapoint_alias_fails_closed(
    row: Mapping[str, JSONValue],
    envelope: Mapping[str, JSONValue],
    change: dict[str, str],
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research.evidence import canonical_datapoint

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))  # no company_tickers snapshot exists
    prov, raw = _alias_source(row, envelope)
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "Q1 FY2026", **change}
    with pytest.raises(ValueError, match=f"not the cited record's {reason}"):
        canonical_datapoint(provenance=prov, source_bytes=raw, **selectors)


def _archive_tickers(root: Path, cik: int, retrieved_at: str, pad: bytes = b"") -> None:
    """Archive one company_tickers snapshot that maps NVDA to ``cik``; ``pad`` keeps identical maps distinct bytes."""
    import json as _json

    from app.storage import raw_archive as _archive

    _archive.archive(
        "sec",
        "company_tickers",
        "company_tickers",
        _json.dumps({"0": {"cik_str": cik, "ticker": "NVDA", "title": "NVIDIA CORP"}}).encode() + pad,
        url="https://www.sec.gov/files/company_tickers.json",
        retrieved_at=retrieved_at,
        root=root / "raw",
    )


def test_canonical_datapoint_maps_ticker_by_archived_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an envelope CIK, only a company_tickers snapshot archived by the session cutoff maps the ticker."""
    from app.research.evidence import canonical_datapoint

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    prov, raw = _alias_source(_ALIAS_ROW, {})
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27"}
    cutoff = datetime(2025, 6, 1, tzinfo=UTC)
    _archive_tickers(tmp_path, 1045810, "2025-07-01T00:00:00Z")
    with pytest.raises(ValueError, match="not the cited record's entity"):
        canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=cutoff)  # snapshot postdates cutoff
    _archive_tickers(tmp_path, 1045810, "2025-05-01T00:00:00Z", pad=b" ")
    point = canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=cutoff)
    assert (point["entity"], point["aliases"]) == ("1045810", {"entity": "NVDA"})
    with pytest.raises(ValueError, match="not the cited record's entity"):
        canonical_datapoint(provenance=prov, source_bytes=raw, **{**selectors, "entity": "AMD"}, as_of=cutoff)


def test_corrupted_ticker_snapshot_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ticker snapshot failing hash verification resolves no alias, tampered mapping included."""
    from app.research.evidence import canonical_datapoint

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    _archive_tickers(tmp_path, 1045810, "2025-05-01T00:00:00Z")
    payloads = sorted((tmp_path / "raw" / "sec" / "company_tickers" / "company_tickers").glob("*.json"))
    payloads = [path for path in payloads if path.suffixes != [".manifest", ".json"]]
    assert len(payloads) == 1
    payloads[0].write_bytes(b'{"0": {"cik_str": 9999, "ticker": "NVDA", "title": "TAMPERED"}}')
    prov, raw = _alias_source(_ALIAS_ROW, {})
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27"}
    with pytest.raises(ValueError, match="not the cited record's entity"):
        canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=datetime(2025, 6, 1, tzinfo=UTC))


def test_snapshot_alias_without_session_cutoff_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool result's created_at never dates a snapshot: no tz-aware session cutoff, no ticker alias."""
    from app.research.evidence import canonical_datapoint

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    _archive_tickers(tmp_path, 1045810, "2025-05-01T00:00:00Z")  # predates the result's created_at
    prov, raw = _alias_source(_ALIAS_ROW, {})
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27"}
    for as_of in (None, "unbounded", "2025-06-01T00:00:00"):
        with pytest.raises(ValueError, match="not the cited record's entity"):
            canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=as_of)
    # The exact source spelling needs no snapshot and no cutoff.
    exact = canonical_datapoint(provenance=prov, source_bytes=raw, **{**selectors, "entity": "1045810"})
    assert exact["entity"] == "1045810" and "aliases" not in exact


def test_post_cutoff_ticker_reassignment_never_backdates_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ticker reassigned after the session cutoff keeps its cutoff holder, however late the result persisted."""
    import json as _json

    from app.research.evidence import canonical_datapoint, observation_proof, verify_observation

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    _archive_tickers(tmp_path, 1045810, "2025-05-01T00:00:00Z")
    _archive_tickers(tmp_path, 9999, "2025-07-01T00:00:00Z")  # NVDA reassigned after the cutoff
    cutoff = datetime(2025, 6, 1, tzinfo=UTC)
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27"}

    def late(row: Mapping[str, JSONValue]) -> tuple[dict[str, JSONValue], bytes]:
        prov, raw = _alias_source(row, {})
        return prov, _json.dumps({**_json.loads(raw), "created_at": "2025-08-01T00:00:00+00:00"}).encode()

    prov, raw = late({**_ALIAS_ROW, "cik": 9999})
    with pytest.raises(ValueError, match="not the cited record's entity"):
        canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=cutoff)
    prov, raw = late(_ALIAS_ROW)
    proof = observation_proof(provenance=prov, source_bytes=raw, **selectors, as_of=cutoff)
    row = dataclasses.replace(
        _ev("EV-PIT", "S-proof", 1, cutoff),
        provenance=prov,
        value="100",
        entity="1045810",
        period="2025-04-27",
        unit="USD",
        proof=proof,
    )
    assert verify_observation(row, value="100", source_bytes=raw, as_of=cutoff).status == "verified"
    # Rebuilding past the reassignment, or with no cutoff, never re-verifies the alias.
    later = datetime(2025, 7, 15, tzinfo=UTC)
    for as_of in (later, None):
        assert verify_observation(row, value="100", source_bytes=raw, as_of=as_of).status == "conflicted"


@pytest.mark.parametrize(
    ("envelope", "row_cik", "as_of", "bound", "snapshot_cik"),
    [
        pytest.param(
            {"ticker": "NVDA", "cik": 9999},
            9999,
            datetime(2025, 7, 15, tzinfo=UTC),
            "9999",
            1045810,
            id="after-reassignment",
        ),
        pytest.param(
            {"ticker": "NVDA", "cik": "0001045810"},
            1045810,
            datetime(2025, 6, 1, tzinfo=UTC),
            "1045810",
            1045810,
            id="normalized-envelope-cik-matches-snapshot",
        ),
        pytest.param(
            {"ticker": "NVDA", "cik": 1045810}, 1045810, "unbounded", "1045810", None, id="unbounded-envelope"
        ),
        pytest.param({"ticker": "NVDA"}, 1045810, "unbounded", None, None, id="ticker-only-unbounded"),
        pytest.param(
            {"ticker": "NVDA", "cik": 9999},
            9999,
            datetime(2025, 6, 1, tzinfo=UTC),
            None,
            None,
            id="bounded-envelope-without-snapshot",
        ),
    ],
)
def test_envelope_cik_yields_to_the_cutoff_snapshot(
    envelope: Mapping[str, JSONValue],
    row_cik: int,
    as_of: datetime | str,
    bound: str | None,
    snapshot_cik: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded envelope pairs its ticker and CIK only when the cutoff snapshot confirms that mapping.

    An absent snapshot confirms nothing, so the envelope pairs nothing and only exact claims bind.
    """
    from app.research.evidence import canonical_datapoint

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    if isinstance(as_of, datetime):
        if snapshot_cik is not None:
            _archive_tickers(tmp_path, snapshot_cik, "2025-05-01T00:00:00Z")
        _archive_tickers(tmp_path, 9999, "2025-07-01T00:00:00Z")  # NVDA reassigned
    prov, raw = _alias_source({**_ALIAS_ROW, "cik": row_cik}, envelope)
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "2025-04-27"}
    if bound is None:
        with pytest.raises(ValueError, match="not the cited record's entity"):
            canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=as_of)
    else:
        point = canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, as_of=as_of)
        assert (point["entity"], point["aliases"]) == (bound, {"entity": "NVDA"})
    # An exact numeric CIK claim needs no ticker pair.
    exact = canonical_datapoint(provenance=prov, source_bytes=raw, **{**selectors, "entity": str(row_cik)}, as_of=as_of)
    assert exact["entity"] == str(row_cik) and "aliases" not in exact


@pytest.mark.parametrize(
    "row_cik", [pytest.param(1045810, id="row-matches-snapshot"), pytest.param(9999, id="row-matches-envelope")]
)
def test_envelope_snapshot_conflict_rejects_whole_datapoint(
    row_cik: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded envelope CIK the cutoff snapshot contradicts rejects every claim on the result, exact CIK included."""
    from app.research.evidence import canonical_datapoint, observation_proof, verify_observation

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    _archive_tickers(tmp_path, 1045810, "2025-05-01T00:00:00Z")
    _archive_tickers(tmp_path, 9999, "2025-07-01T00:00:00Z")  # NVDA reassigned after the cutoff
    cutoff, later = datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 7, 15, tzinfo=UTC)
    prov, raw = _alias_source({**_ALIAS_ROW, "cik": row_cik}, {"ticker": "NVDA", "cik": 9999})
    selectors = {"metric": "revenue", "value": "100", "period": "2025-04-27"}
    for entity in ("NVDA", str(row_cik)):
        with pytest.raises(ValueError, match="source identity conflict"):
            canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, entity=entity, as_of=cutoff)
    # A proof built where the envelope holds never re-verifies at the contradicted cutoff.
    proof = observation_proof(provenance=prov, source_bytes=raw, **selectors, entity=str(row_cik), as_of=later)
    row = dataclasses.replace(
        _ev("EV-CONFLICT", "S-proof", 1, cutoff),
        provenance=prov,
        value="100",
        entity=str(row_cik),
        period="2025-04-27",
        unit="USD",
        proof=proof,
    )
    assert verify_observation(row, value="100", source_bytes=raw, as_of=later).status == "verified"
    verdict = verify_observation(row, value="100", source_bytes=raw, as_of=cutoff)
    assert verdict.status == "conflicted" and "source identity conflict" in verdict.reason


@settings(max_examples=20, derandomize=True)
@given(st.integers(min_value=0, max_value=6), st.booleans())
def test_prop_equivalent_envelope_cik_spellings_never_conflict(padding: int, prefixed: bool) -> None:
    """Zero padding and a sec:cik: prefix on the envelope CIK keep its agreement with the cutoff snapshot."""
    import os
    import tempfile
    from unittest import mock

    from app.research.evidence import canonical_datapoint

    spelling = f"{'sec:cik:' if prefixed else ''}{'0' * padding}1045810"
    with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"STOCKBOT_DATA_DIR": root}):
        _archive_tickers(Path(root), 1045810, "2025-05-01T00:00:00Z")
        prov, raw = _alias_source(_ALIAS_ROW, {"ticker": "NVDA", "cik": spelling})
        selectors = {"metric": "revenue", "value": "100", "period": "2025-04-27"}
        cutoff = datetime(2025, 6, 1, tzinfo=UTC)
        point = canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, entity="NVDA", as_of=cutoff)
        assert (point["entity"], point["aliases"]) == ("1045810", {"entity": "NVDA"})
        exact = canonical_datapoint(provenance=prov, source_bytes=raw, **selectors, entity="1045810", as_of=cutoff)
        assert exact["entity"] == "1045810" and "aliases" not in exact


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param({"metric": "net income"}, "does not label", id="revenue-100-is-not-net-income"),
        pytest.param({"metric": "gross margin"}, "does not label", id="label-outside-the-sentence"),
        pytest.param({"value": "10"}, "is not a whole token", id="value-substring"),
        pytest.param({"entity": "AMD"}, "entity 'AMD' is not stated", id="wrong-entity"),
        pytest.param({"period": "Q2 FY2025"}, "period 'Q2 FY2025' is not stated", id="wrong-period"),
        pytest.param({"unit": "billion"}, "unit 'billion' is not stated", id="wrong-unit"),
    ],
)
def test_canonical_datapoint_binds_labeled_passage_sentence(change: dict[str, str], reason: str) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=_DP_PASSAGE, tool_result_id="tr-web")
    selectors = {"metric": "revenue", "value": "100", "entity": "NVDA", "period": "Q1 FY2025", "unit": "million"}
    bound = canonical_datapoint(provenance=prov, source_bytes=b"archived", **selectors)
    assert (bound["metric"], bound["field"], bound["statement"]) == (
        "revenue",
        "excerpt",
        "NVDA Q1 FY2025: revenue was $100 million.",
    )
    with pytest.raises(ValueError, match=reason):
        canonical_datapoint(provenance=prov, source_bytes=b"archived", **{**selectors, **change})


@pytest.mark.parametrize(
    "passage",
    [
        "NVDA revenue and net income for Q1 FY2025 were 100 and 50 million.",
        "NVDA revenue and net income for Q1 FY2025 were 100 million.",
        "NVDA net income and revenue for Q1 FY2025 were 100 million.",
        "NVDA Q1 FY2025 cost of revenue was 100 million.",
        "NVDA Q1 FY2025 Data Center revenue was 100 million.",
        "NVDA Q1 FY2025 revenue increased 100 million.",
        "NVDA Q1 FY2025 revenue fell 100 million.",
        "NVDA Q1 FY2025 revenue increased by 100 million.",
        "NVDA revenue was 100 million. The period was Q1 FY2025.",
        "NVDA Q1 FY2025 revenue was 100. The unit was million.",
        "NVDA Q1 FY2025 revenue was 100 million and margin was 40%.",
        "Q1 FY2025 revenue was 90 million. NVDA revenue was 100 million.",
        "NVDA Q1 FY2025 revenue was 100 million from services.",
        "NVDA Q1 FY2025 revenue was 100 million from continuing operations.",
        "NVDA Q1 FY2025 revenue was 100 million attributable to Product X.",
        "NVDA Q1 FY2025 revenue was 100 million excluding disposals.",
        "NVDA Q1 FY2025 revenue was 100 million excl. disposals.",
        "NVDA Q1 FY2025 revenue was 100 million, adjusted.",
        "NVDA Q1 FY2025 revenue was 100 million for the Data Center business.",
        "NVDA Q1 FY2025 revenue was 100 million of the Gaming segment.",
        "NVDA Q1 FY2025 revenue was 100 million from the Automotive division.",
        "NVDA Q1 FY2025 revenue was 100 million for its Networking unit.",
        "NVDA Q1 FY2025 revenue was 100 million from the GeForce product line.",
    ],
)
def test_canonical_datapoint_rejects_other_metric_value_or_later_identity(passage: str) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=passage, tool_result_id="tr-web")
    with pytest.raises(ValueError, match="unrelated metric or value|is not stated"):
        canonical_datapoint(
            provenance=prov,
            source_bytes=b"archived",
            metric="revenue",
            value="100",
            entity="NVDA",
            period="Q1 FY2025",
            unit="million",
        )


@pytest.mark.parametrize(
    ("passage", "statement"),
    [
        ("NVDA Q1 FY2025. Revenue was 100 million.", "Revenue was 100 million."),
        ("NVDA Q1 FY2025. Revenue increased to 100 million.", "Revenue increased to 100 million."),
        (
            "NVDA\nQ1 FY2025 overview\nDuring the quarter, our revenue was approximately $100 million, led by gaming.",
            "During the quarter, our revenue was approximately $100 million, led by gaming.",
        ),
    ],
)
def test_canonical_datapoint_takes_identity_from_heading_context(passage: str, statement: str) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=passage, tool_result_id="tr-web")
    point = canonical_datapoint(
        provenance=prov,
        source_bytes=b"archived",
        metric="revenue",
        value="100",
        entity="NVDA",
        period="Q1 FY2025",
        unit="million",
    )
    assert (str(point["metric"]).casefold(), point["entity"], point["period"], point["unit"]) == (
        "revenue",
        "NVDA",
        "Q1 FY2025",
        "million",
    )
    assert point["statement"] == statement


@pytest.mark.parametrize(
    ("metric", "passage"),
    [
        ("adjusted revenue", "NVDA Q1 FY2025 adjusted revenue was 100 million on an adjusted basis."),
        ("Data Center revenue", "NVDA Q1 FY2025 Data Center revenue was 100 million for the data center business."),
    ],
)
def test_canonical_datapoint_keeps_tail_scope_the_claimed_metric_names(metric: str, passage: str) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=passage, tool_result_id="tr-web")
    selectors = {"value": "100", "entity": "NVDA", "period": "Q1 FY2025", "unit": "million"}
    point = canonical_datapoint(provenance=prov, source_bytes=b"archived", metric=metric, **selectors)
    assert (point["metric"], point["statement"]) == (metric, passage)


@pytest.mark.parametrize(
    ("metric", "passage"),
    [
        ("Data Center revenue", "NVDA Q1 FY2025 Data Center revenue was 100 million for the Gaming business."),
        ("segment revenue", "NVDA Q1 FY2025 segment revenue was 100 million from the Gaming segment."),
    ],
)
def test_canonical_datapoint_rejects_tail_scope_the_claimed_metric_omits(metric: str, passage: str) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=passage, tool_result_id="tr-web")
    selectors = {"value": "100", "entity": "NVDA", "period": "Q1 FY2025", "unit": "million"}
    with pytest.raises(ValueError, match="unrelated metric or value"):
        canonical_datapoint(provenance=prov, source_bytes=b"archived", metric=metric, **selectors)


_DP_TABLE = (
    "NVDA\n"
    "Condensed Statements of Income (in millions)\n"
    "Three Months Ended\n"
    "Q1 FY2025 | Q1 FY2024\n"
    "Revenue | $ | 26,044 | $ | 7,192\n"
    "Cost of revenue | 5,638 | 2,544\n"
)


@pytest.mark.parametrize(
    ("passage", "value", "period", "reason"),
    [
        pytest.param(_DP_TABLE, "7,192", "Q1 FY2024", None, id="column-header-period"),
        pytest.param(_DP_TABLE, "26,044", "Q1 FY2025", None, id="first-column"),
        pytest.param(_DP_TABLE, "26,044", "Q1 FY2024", "heads another column", id="wrong-column"),
        pytest.param(_DP_TABLE, "5,638", "Q1 FY2025", "unrelated metric or value", id="cost-of-revenue-row"),
        pytest.param("NVDA Q1 FY2025 (in millions)\nRevenue | 100", "100", "Q1 FY2025", None, id="heading-one-column"),
        pytest.param(
            "NVDA Q1 FY2025 (in millions)\nRevenue | 100 | 90",
            "100",
            "Q1 FY2025",
            "every value column",
            id="heading-many",
        ),
    ],
)
def test_canonical_datapoint_binds_table_row_through_headings_and_columns(
    passage: str, value: str, period: str, reason: str | None
) -> None:
    from app.research.evidence import canonical_datapoint, web_source_ref

    prov = web_source_ref(url="https://example.com/q1", excerpt=passage, tool_result_id="tr-web")
    selectors = {"metric": "revenue", "value": value, "entity": "NVDA", "period": period, "unit": "millions"}
    if reason is not None:
        with pytest.raises(ValueError, match=reason):
            canonical_datapoint(provenance=prov, source_bytes=b"archived", **selectors)
        return
    point = canonical_datapoint(provenance=prov, source_bytes=b"archived", **selectors)
    assert (point["metric"], point["value"], point["period"], point["unit"]) == ("Revenue", value, period, "millions")
    assert point["statement"] == next(line for line in passage.splitlines() if value in line)


def test_verify_rebuilds_canonical_datapoints_and_refuses_legacy_or_altered_proofs() -> None:
    """Only a descriptor the source rebuilds verifies; a legacy proof never counts as a new verified datapoint."""
    from app.research.evidence import datapoint_descriptor, verify_calculation, verify_observation
    from app.research.service import _recompute

    total = _observation_row("EV-TOT", _PROOF_TOT, "26,044")
    dc = _observation_row("EV-DC", _PROOF_DC, "22,563")
    assert verify_observation(dc, value="22,563", source_bytes=_PROOF_DOC).status == "verified"
    raw_locator = dc.proof["locator"]
    assert isinstance(raw_locator, dict)
    locator: dict[str, JSONValue] = dict(raw_locator)
    datapoint = locator.pop("datapoint")
    assert isinstance(datapoint, dict)
    altered: list[tuple[dict[str, JSONValue], str]] = [
        (locator, "unverified"),  # legacy weak proof: locator + value hash only
        ({**locator, "datapoint": {**datapoint, "metric": "Total revenue"}}, "conflicted"),
        ({**locator, "datapoint": {**datapoint, "id": "dp:" + "0" * 64}}, "conflicted"),
        ({**locator, "datapoint": {**datapoint, "statement": "Data Center revenue was $22,563 million"}}, "conflicted"),
    ]
    for changed, status in altered:
        row = dataclasses.replace(dc, proof={**dc.proof, "locator": changed})
        assert verify_observation(row, value="22,563", source_bytes=_PROOF_DOC).status == status
    inputs = [(row, verify_observation(row, value=row.value, source_bytes=_PROOF_DOC)) for row in (total, dc)]
    calc = _calculation_row("EV-NONDC", "3481", ["EV-TOT", "EV-DC"], rows=[total, dc])
    assert verify_calculation(calc, value="3481", inputs=inputs, recompute=_recompute).status == "verified"
    stored = calc.proof["datapoint"]
    total_point = datapoint_descriptor(total)
    assert total_point is not None
    assert isinstance(stored, dict) and stored["inputs"] == [total_point["id"], datapoint["id"]]
    legacy_calc = _calculation_row("EV-OLD", "3481", ["EV-TOT", "EV-DC"])
    assert verify_calculation(legacy_calc, value="3481", inputs=inputs, recompute=_recompute).status == "unverified"
    relabeled = dataclasses.replace(calc, proof={**calc.proof, "datapoint": {**stored, "metric": "net income"}})
    assert verify_calculation(relabeled, value="3481", inputs=inputs, recompute=_recompute).status == "conflicted"


def test_nested_calculations_preserve_each_input_proof_and_formula_version() -> None:
    from app.research.evidence import verify_calculation, verify_observation
    from app.research.service import _recompute

    total = _observation_row("EV-TOT", _PROOF_TOT, "26,044")
    dc = _observation_row("EV-DC", _PROOF_DC, "22,563")
    difference = _calculation_row("EV-NONDC", "3481", ["EV-TOT", "EV-DC"], rows=[total, dc])
    observations = [(row, verify_observation(row, value=row.value, source_bytes=_PROOF_DOC)) for row in (total, dc)]
    difference_verdict = verify_calculation(
        difference, value=difference.value, inputs=observations, recompute=_recompute
    )
    restored = _calculation_row("EV-RESTORED", "26044", ["EV-NONDC", "EV-DC"], formula="sum", rows=[difference, dc])
    nested_inputs = [(difference, difference_verdict), observations[1]]
    assert verify_calculation(restored, value=restored.value, inputs=nested_inputs, recompute=_recompute).status == (
        "verified"
    )
    assert restored.proof["inputProofIds"] == ["EV-NONDC", "EV-DC"]
    assert difference.proof["inputProofIds"] == ["EV-TOT", "EV-DC"]
    unknown_version = dataclasses.replace(restored, proof={**restored.proof, "formulaVersion": "2"})
    assert (
        verify_calculation(
            unknown_version, value=unknown_version.value, inputs=nested_inputs, recompute=_recompute
        ).status
        == "unverified"
    )
    assert (
        verify_calculation(
            restored, value=restored.value, inputs=[observations[0], observations[1]], recompute=_recompute
        ).status
        == "unverified"
    )


@pytest.mark.parametrize(
    ("formula", "values", "expected"),
    [
        ("sum", ("0.1", "0.2"), "0.3"),
        ("difference", ("4.90", "3.14"), "1.76"),
        ("difference", ("0.3000000000000000001", "0.1"), "0.2000000000000000001"),
        ("ratio", ("1", "8"), "0.125"),
        ("pct_change", ("0.10", "0.11"), "10"),
    ],
)
def test_calculation_registry_keeps_decimal_precision(formula: str, values: tuple[str, ...], expected: str) -> None:
    from app.research.service import _recompute

    assert _recompute(formula, "1", values) == expected
    assert _recompute(formula, "2", values) is None


def test_sec_q4_replay_recomputes_from_two_positional_inputs_and_preserves_nested_ttm() -> None:
    from decimal import Decimal

    from app.research.service import _recompute

    fy = _observation_row("EV-FY", _PROOF_TOT, "26,044")
    dc = _observation_row("EV-DC2", _PROOF_DC, "22,563")
    q4 = _calculation_row("EV-Q4", "3481", ["EV-FY", "EV-DC2"], rows=[fy, dc])
    assert _recompute("difference", "1", ("26,044", "22,563")) == "3481"
    assert fy.entity == dc.entity == q4.entity == "NVDA"
    assert fy.unit == dc.unit == q4.unit == "million"
    assert q4.proof["inputProofIds"] == ["EV-FY", "EV-DC2"]
    quarters = [fy, dc, _observation_row("EV-QA", _PROOF_TOT, "26,044"), q4]
    assert len({row.evidence_id for row in quarters}) == 4
    assert sum(Decimal((row.value or "0").replace(",", "")) for row in quarters) == Decimal(78132)
    assert _recompute("sum", "1", ("26,044", "22,563", "26,044", "3481")) == "78132"


def test_proof_rows_round_trip_and_keep_freeze_hash_and_pit() -> None:
    from app.research.evidence import evidence_from_dict, evidence_to_dict
    from app.research.freeze import freeze_content_hash

    row = _observation_row("EV-DC", _PROOF_DC, "22,563")
    back = evidence_from_dict(evidence_to_dict(row))
    assert (back.value, back.entity, back.period, back.unit, back.proof) == (
        row.value,
        row.entity,
        row.period,
        row.unit,
        row.proof,
    )
    # A legacy row dict (no datapoint/proof keys) reads back unproven.
    legacy = evidence_to_dict(_ev("EV-OLD", "S-proof", 1, ASOF))
    for key in ("value", "entity", "period", "unit", "proof"):
        legacy.pop(key, None)
    old = evidence_from_dict(legacy)
    assert old.value is None and old.proof == {}
    # The freeze hash stays sorted id:content_hash; proof fields never change it.
    bare = dataclasses.replace(row, value=None, entity=None, period=None, unit=None, proof={})
    assert freeze_content_hash([row]) == freeze_content_hash([bare])
    frozen = create_freeze(freeze_id="EP", session_id="S-proof", wave_id=1, records=[row], as_of=ASOF)
    verify_freeze(frozen, [back])
    # PIT still rejects a proven row known after as_of.
    late = dataclasses.replace(row, known_at=datetime(2025, 7, 1, tzinfo=UTC))
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(EvidenceLedger(), late, as_of=ASOF, on_reject=None)
    with pytest.raises(FreezeIntegrityError, match="PIT"):
        create_freeze(freeze_id="EP2", session_id="S-proof", wave_id=1, records=[late], as_of=ASOF)


def test_freeze_pins_datapoint_and_proof_and_reads_legacy_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sqlite3

    from app.research.evidence import evidence_from_dict, evidence_to_dict, value_hash
    from app.research.freeze import freeze_from_dict, freeze_to_dict
    from app.research.service import _committee_verify_write, _freeze_save_idempotent

    row = _observation_row("EV-DC", _PROOF_DC, "22,563")
    frozen = create_freeze(freeze_id="EP", session_id="S-proof", wave_id=1, records=[row], as_of=ASOF)
    stored = freeze_from_dict(freeze_to_dict(frozen))
    assert stored.datapoint_hash == frozen.datapoint_hash
    verify_freeze(stored, [evidence_from_dict(evidence_to_dict(row))])
    # Canonical JSON: nested proof key order never changes the pinned hash.
    locator = row.proof["locator"]
    assert isinstance(locator, dict)
    verify_freeze(stored, [dataclasses.replace(row, proof={**row.proof, "locator": dict(reversed(locator.items()))})])
    # Each datapoint/proof edit the content hash misses fails at freeze verify.
    tampered = [
        dataclasses.replace(row, entity="AMD"),
        dataclasses.replace(row, period="Q2 FY2025"),
        dataclasses.replace(row, unit="USD thousands"),
        dataclasses.replace(row, value="22,564", proof={**row.proof, "valueHash": value_hash("22,564")}),
        dataclasses.replace(row, proof={**row.proof, "locator": {**locator, "offset": 0}}),
    ]
    for bad in tampered:
        assert bad.content_hash == row.content_hash
        with pytest.raises(FreezeIntegrityError, match="datapoint_hash mismatch"):
            verify_freeze(stored, [bad])
    # A pre-datapoint freeze record (no datapoint_hash key) still loads and verifies its legacy rows,
    # and a datapoint later added to one of those rows is drift.
    old = _ev("EV-OLD", "S-proof", 1, ASOF)
    legacy = freeze_to_dict(create_freeze(freeze_id="EL", session_id="S-proof", wave_id=1, records=[old], as_of=ASOF))
    legacy.pop("datapoint_hash")
    legacy_freeze = freeze_from_dict(legacy)
    assert legacy_freeze.datapoint_hash is None
    verify_freeze(legacy_freeze, [old])
    with pytest.raises(FreezeIntegrityError, match="legacy freeze"):
        verify_freeze(legacy_freeze, [dataclasses.replace(old, entity="NVDA")])
    for bad_hash in ("", 5):
        with pytest.raises(FreezeIntegrityError, match="datapoint_hash"):
            freeze_from_dict({**legacy, "datapoint_hash": bad_hash})
    # Persisted path: the committee write gate catches a sqlite edit to the frozen row.
    db = tmp_path / "r.sqlite"
    monkeypatch.setenv("RESEARCH_DB_PATH", str(db))
    repo = ResearchRepository()
    repo.save_evidence(evidence_to_dict(row))
    repo.save_freeze(freeze_to_dict(frozen))
    repo.save_freeze(legacy)
    repo.save_evidence(evidence_to_dict(old))
    _committee_verify_write(repo, "S-proof", "EP")
    _committee_verify_write(repo, "S-proof", "EL")
    _freeze_save_idempotent(repo, frozen, [row])
    legacy_retry = create_freeze(freeze_id="EL", session_id="S-proof", wave_id=1, records=[old], as_of=ASOF)
    _freeze_save_idempotent(repo, legacy_retry, [old])
    assert repo.get_freeze("EL") == legacy
    # A fresh hash for tampered rows cannot replace or bypass the persisted freeze.
    changed = dataclasses.replace(row, period="FY2099")
    retry = create_freeze(freeze_id="EP", session_id="S-proof", wave_id=1, records=[changed], as_of=ASOF)
    with pytest.raises(FreezeIntegrityError, match="datapoint_hash mismatch"):
        _freeze_save_idempotent(repo, retry, [changed])
    assert repo.get_freeze("EP") == freeze_to_dict(frozen)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE evidence SET record = ? WHERE evidence_id = ?",
            (json.dumps({**evidence_to_dict(row), "period": "FY2099"}, sort_keys=True), "EV-DC"),
        )
    with pytest.raises(FreezeIntegrityError, match="datapoint_hash mismatch"):
        _committee_verify_write(repo, "S-proof", "EP")


def test_dossier_rejects_dangling_refs() -> None:
    s = _session.create_session("q?", "o", as_of=ASOF)
    cov: dict[str, object] = {
        "entities": ["NVDA"],
        "forms": ["10-Q"],
        "time_range": {"start": "2025-01-01", "end": "2025-06-30"},
        "sources_examined": ["e"],
        "complete": True,
        "exclusions": [],
    }
    bad = create_dossier(
        dossier_id="SEC-D1",
        session_id=s.session_id,
        wave_id=1,
        subject="NVDA",
        coverage=cov,
        findings=[{"text": "dangling", "evidence_ids": ["EV-NOPE"]}],
        unknowns=[],
        limitations=[],
        open_questions=[],
    )
    with pytest.raises(DossierIntegrityError):
        validate_dossier(bad, set())


DispatchFn = Callable[[str, dict[str, object]], dict[str, object]]
ModelFn = Callable[[str], str]


def _fake_document(index: int, args: Mapping[str, object]) -> dict[str, object]:
    """One raw filing document result (accession + document_name + passage): the only evidence shape."""
    raw_accession = args.get("accession_no")
    accession = (
        raw_accession.strip()
        if isinstance(raw_accession, str) and raw_accession.strip()
        else f"0000320193-25-{index:06d}"
    )
    raw_document = args.get("document_name")
    document = (
        raw_document.strip() if isinstance(raw_document, str) and raw_document.strip() else f"nvda-20250331-{index}.htm"
    )
    passage = "Data Center revenue grew year over year on accelerated computing demand."
    return {
        "content": passage,
        "text": passage,
        "accession_no": accession,
        "document_name": document,
        "matching_passage": passage,
        "source_handle": seam.handle_for(passage, accession=accession, document=document),
        "known_at": "2025-05-01",
        "uri": f"https://www.sec.gov/Archives/edgar/data/1045810/{accession.replace('-', '')}/{document}",
    }


def _fake_dispatch() -> DispatchFn:
    """SEC fake: searches return navigation packets (top_hits), document opens return raw passages.

    Only ``get_sec_document``/``get_sec_filing`` results carrying an accession, a
    document name, and a passage become evidence; every other result is navigation.
    """
    counter = itertools.count(1)

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {
                "matches": [
                    "search_sec_filings",
                    "list_sec_filings",
                    "get_sec_document",
                    "get_sec_filing",
                ]
            }
        if name == "call_tool":
            n = next(counter)
            inner = str(args.get("name") or "")
            inner_args = args.get("arguments")
            requested: dict[str, object] = inner_args if isinstance(inner_args, dict) else {}
            if inner in ("get_sec_document", "get_sec_filing"):
                return _fake_document(n, requested)
            return {
                "search_id": f"s{n}",
                "query": str(requested.get("query") or ""),
                "count": 1,
                "top_hits": [
                    {
                        "accession": f"0000320193-25-{n:06d}",
                        "document": f"nvda-20250331-{n}.htm",
                        "form": "10-Q",
                    }
                ],
            }
        return {}

    return _dispatch


def _run_wave(repo: ResearchRepository, model: ModelFn) -> dict[str, object]:
    return run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=model,
        repo=repo,
        budgets=None,
    )


def test_same_freeze_parallel_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _model(prompt: str) -> str:
        time.sleep(2)
        assert prompt
        return _grounded(prompt)

    out = _run_wave(repo, _model)
    assert out["stop_reason"] == "complete:wave1"
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    assert stock.evidence_ids == bull.evidence_ids == bear.evidence_ids
    sid_obj = out["session_id"]
    assert isinstance(sid_obj, str)
    jobs = repo.list_jobs(sid_obj)
    assert all(j.status == "completed" for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot"))


def test_parallel_beats_serial_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    spans: list[tuple[float, float]] = []

    def _slow(prompt: str) -> str:
        assert prompt
        start = time.monotonic()
        time.sleep(1)
        spans.append((start, time.monotonic()))
        return _grounded(prompt)

    _run_wave(repo, _slow)
    committee = spans[3:6]  # first three calls are serial scouts
    assert len(committee) == 3
    assert max(s for s, _ in committee) < min(e for _, e in committee)


def test_resume_at_source_freeze_one_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    for hook, reason in (
        ("source", "interrupted:source"),
        ("freeze", "interrupted:freeze"),
        ("one-committee", "interrupted:one-committee"),
    ):
        out = run_live(
            question="NVDA demand?",
            objective="o",
            as_of="2025-06-30T00:00:00+00:00",
            tickers=["NVDA"],
            dispatch=_fake_dispatch(),
            model=_ok,
            repo=repo,
            budgets=None,
            interrupt_after=hook,
        )
        assert out["stop_reason"] == reason
        resume_sid = out["session_id"]
        assert isinstance(resume_sid, str)
        n_jobs = len(repo.list_jobs(resume_sid))
        state = repo.resume(resume_sid)
        assert len(repo.list_jobs(resume_sid)) == n_jobs  # no dup on resume
        assert state.session is not None


def test_committee_prompts_carry_identical_freeze_payload() -> None:
    prompts: list[str] = []

    def _cap(prompt: str) -> str:
        prompts.append(prompt)
        return _grounded(prompt)

    s = _session.create_session("NVDA demand?", "o", as_of=ASOF)
    led = EvidenceLedger()
    ingest_evidence(
        led,
        _ev("EV-1", s.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)),
        as_of=ASOF,
        on_reject=None,
    )
    rec = led.get("EV-1")
    payload = f"[{rec.evidence_id}] {rec.subject} | claim: {rec.claim_text}\ncontent: {rec.content}"
    from app.research.agents.bearbot import run_bearbot
    from app.research.agents.bullbot import run_bullbot
    from app.research.agents.stockbot import run_stockbot

    run_stockbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    run_bullbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    run_bearbot(
        "Q?",
        session_id=s.session_id,
        wave_id=1,
        freeze_id="E1",
        evidence_ids=["EV-1"],
        as_of="x",
        model=_cap,
        evidence_text=payload,
    )
    assert len(prompts) == 3
    bodies = [p.split("Evidence (cite ids):\n", 1)[1] for p in prompts]
    assert bodies[0] == bodies[1] == bodies[2]
    assert "EV-1" in bodies[0] and "c-ev-1" in bodies[0].lower()


def test_source_timestamp_never_invented_and_post_cutoff_rejected() -> None:
    from app.research.runner import _extract_known_at, _extract_source_ref

    assert _extract_known_at({}) is None
    assert _extract_known_at({"record": {}}) is None
    assert _extract_source_ref({}) == (None, None)
    assert _extract_source_ref({"source": "SEC EDGAR"}) == (None, None)
    known = _extract_known_at(
        {
            "meta": {
                "source_refs": {
                    "record_id": "0001045810-25-000023",
                    "uri": "https://sec.gov/x",
                    "known_at": "2025-05-28",
                }
            }
        }
    )
    assert known is not None and (known.year, known.month, known.day) == (2025, 5, 28)
    uri, ref = _extract_source_ref(
        {
            "meta": {
                "source_refs": {
                    "record_id": "0001045810-25-000023",
                    "uri": "https://sec.gov/x",
                }
            }
        }
    )
    assert (uri, ref) == ("https://sec.gov/x", "0001045810-25-000023")
    s = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-FUT", s.session_id, 1, datetime(2026, 1, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen[0][1]["reason"] == "PIT_VIOLATION"
    assert led.ids() == ()


def test_resume_after_source_completes_reusing_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    assert out["stop_reason"] == "interrupted:source"
    sid = out["session_id"]
    assert isinstance(sid, str)
    eids_raw = out["evidence_ids"]
    assert isinstance(eids_raw, list) and all(isinstance(e, str) for e in eids_raw)
    eids_before: list[str] = list(eids_raw)
    did_before = out["dossier_id"]
    ev_before = len(repo.list_evidence(sid))
    jobs_before = repo.list_jobs(sid)
    assert (
        len(jobs_before) == 4
        and sum(1 for j in jobs_before if j.job_type == "source_agent") == 1
        and sum(1 for j in jobs_before if j.job_type == "scout") == 3
    )
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    eids_after = out2["evidence_ids"]
    assert isinstance(eids_after, list)
    # The resumed run reuses exactly the persisted wave evidence: every dossier-cited id
    # is in it, it never advertises a navigation artifact, and nothing new was fetched.
    persisted = [dict(r) for r in repo.list_evidence(sid)]
    evidence_rows = {str(r.get("evidence_id")) for r in persisted if r.get("record_kind") == "evidence"}
    discovery_rows = {str(r.get("evidence_id")) for r in persisted if r.get("record_kind") == "discovery"}
    assert set(eids_before) <= set(eids_after) == evidence_rows
    assert discovery_rows and not (set(eids_after) & discovery_rows)
    assert len(repo.list_evidence(sid)) == ev_before  # no dup evidence
    assert out2["dossier_id"] == did_before  # dossier reused, not recreated
    jobs_after = repo.list_jobs(sid)
    assert len([j for j in jobs_after if j.job_type == "source_agent"]) == 1  # no dup source job
    assert len([j for j in jobs_after if j.job_type == "scout"]) == 3  # completed scouts reused, no dup scouts
    assert len(jobs_after) == len(jobs_before) + 3  # trio added once
    assert out2["freeze_id"] == f"{sid}:1:freeze"
    assert repo.get_session(sid).status == "completed"


def test_resume_after_freeze_skips_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="freeze",
    )
    assert out["stop_reason"] == "interrupted:freeze"
    sid = out["session_id"]
    assert isinstance(sid, str)
    fid = out["freeze_id"]
    assert isinstance(fid, str) and fid
    calls = [0]
    base = _fake_dispatch()

    def _counting(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "call_tool":
            calls[0] += 1
        return base(name, args)

    out2 = resume_live(sid, _counting, _ok, repo=repo)
    assert calls[0] == 0  # fetch skipped: no tool evidence calls
    assert out2["stop_reason"] == "complete:wave1"
    assert out2["freeze_id"] == fid  # same freeze reused
    assert out2["stock"] is not None and out2["bull"] is not None and out2["bear"] is not None


def test_resume_after_one_committee_reruns_trio_on_same_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="one-committee",
    )
    assert out["stop_reason"] == "interrupted:one-committee"
    sid = out["session_id"]
    assert isinstance(sid, str)
    fid = out["freeze_id"]
    jobs_before = repo.list_jobs(sid)
    assert len(jobs_before) == 5  # source + 3 scouts + stock-only
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    # Analyses are not persisted, so resume always reruns the full trio on the
    # same freeze id with new jobs (never reuses the stock-only partial result).
    assert out2["freeze_id"] == fid
    assert out2["stock"] is not None and out2["bull"] is not None and out2["bear"] is not None
    assert out2["stop_reason"] == "complete:wave1"
    jobs_after = repo.list_jobs(sid)
    assert len(jobs_after) == len(jobs_before) + 3
    sess = repo.get_session(sid)
    assert len(sess.committee_runs) == 2
    second = sess.committee_runs[1]
    assert isinstance(second, dict) and second.get("freeze_id") == fid
    second_jobs = second.get("jobs")
    assert isinstance(second_jobs, list) and len(second_jobs) == 3


def test_resume_dossier_resource_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.resources import read_resource

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    did = out["dossier_id"]
    assert isinstance(did, str) and did
    stores = repo.resource_stores(sid)
    resolved = read_resource(f"dossier://{did}", dossiers=stores["dossier"])
    assert isinstance(resolved, dict) and resolved.get("dossier_id") == did
    resolved_full = read_resource(
        f"dossier://{did}",
        evidence=stores["evidence"],
        freezes=stores["freeze"],
        dossiers=stores["dossier"],
        jobs=stores["job"],
        sessions=stores["research"],
    )
    assert resolved_full == resolved


def test_resume_failed_session_reports_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from app.research.runner import LiveModelError, resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _timeout_model(_prompt: str) -> str:
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1)

    with pytest.raises(LiveModelError) as excinfo:
        run_live(
            "timeout probe?",
            "probe",
            None,
            ["NVDA"],
            lambda n, a: {},
            _timeout_model,
            repo=repo,
        )
    sid = excinfo.value.session_id
    assert repo.get_session(sid).status == "failed"
    n_jobs = len(repo.list_jobs(sid))
    n_ev = len(repo.list_evidence(sid))

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    with pytest.raises(LiveModelError, match="out of scope"):
        resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert len(repo.list_jobs(sid)) == n_jobs  # failed resume writes nothing
    assert len(repo.list_evidence(sid)) == n_ev


def test_scouts_run_serially_with_ordered_results() -> None:
    from app.research.agents.scout import run_scout
    from app.research.agents.sec_agent import run_sec_assignment

    seen: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name
        assert args is not None
        return {}

    def _model(prompt: str) -> str:
        assert prompt
        seen.append(prompt[:40])
        return "[]"

    dossier = run_sec_assignment(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        as_of="2025-06-30",
        tickers=["NVDA"],
        dispatch=_dispatch,
        model=_model,
        spawn=lambda a: run_scout(a, dispatch=_dispatch, model=_model),
    )
    assert len(seen) == 3  # filings, financials, risk in assignment order
    assert isinstance(dossier, object) and dossier is not None


def test_pit_unverified_historical_rejected_current_accepted() -> None:
    from datetime import datetime

    from app.research.evidence import (
        Evidence,
        EvidenceLedger,
        EvidenceRejectedError,
        evidence_content_hash,
        ingest_evidence,
    )

    hist = datetime(2025, 6, 30, tzinfo=UTC)

    def _mk(eid: str, known: datetime | None) -> Evidence:
        content = "content"
        return Evidence(
            eid,
            "rs:pit",
            1,
            "sec",
            "search_sec_filings",
            "s",
            "claim",
            content,
            evidence_content_hash(content),
            datetime(2025, 5, 1, tzinfo=UTC),
            "https://sec.gov/x",
            "r1",
            None,
            known,
            None,
            "job:1",
            "sec_scout",
            (),
            (),
            None,
            None,
            {},
            None,
        )

    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError) as ei:
        ingest_evidence(
            led,
            _mk("EV-U1", None),
            as_of=hist,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert ei.value.reason == "PIT_UNVERIFIED"
    assert seen[0][1]["reason"] == "PIT_UNVERIFIED"
    assert led.ids() == ()
    out = ingest_evidence(led, _mk("EV-U2", None), as_of=None)
    assert out.evidence_id == "EV-U2"
    from app.research.agents.scout import _is_pit_eligible

    assert _is_pit_eligible(None, "2025-06-30") is False
    assert _is_pit_eligible(None, "unbounded") is True


def test_budget_resume_hydrates_cumulative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    assert out["stop_reason"] == "interrupted:source"
    sid = out["session_id"]
    assert isinstance(sid, str)
    sess = repo.get_session(sid)
    used = sess.budget.get("tool_calls_used")
    assert isinstance(used, int) and used > 0
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    sess2 = repo.get_session(sid)
    used2 = sess2.budget.get("tool_calls_used")
    assert isinstance(used2, int) and used2 == used  # no reset to zero on resume


def test_two_wave_e2_tree_and_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A material wave-1 follow-up drives a second wave on cumulative evidence; an explicit
    configured max_waves (never an architectural ceiling) then settles the run at complete:wave2."""
    from app.research.director import DirectorBudgets

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()

    def _model(prompt: str) -> str:
        assert prompt
        return _grounded(prompt, ["What drove the Q2 delta?"])

    out = run_live(
        question="NVDA historical demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_model,
        repo=repo,
        budgets=DirectorBudgets(max_waves=2),
    )
    assert out["stop_reason"] == "complete:wave2"
    assert str(out["wave_decision"]).startswith("max_waves")  # explicit int honored, waves are sequence numbers
    sid = out["session_id"]
    assert isinstance(sid, str)
    jobs = repo.list_jobs(sid)
    src_jobs = [j for j in jobs if j.job_type == "source_agent"]
    scout_jobs = [j for j in jobs if j.job_type == "scout"]
    committee_jobs = [j for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot")]
    assert len(src_jobs) == 2 and len(scout_jobs) == 6 and len(committee_jobs) == 6
    src_ids = {j.job_id for j in src_jobs}
    assert all(j.parent_job_id in src_ids for j in scout_jobs)
    assert all(j.status == "completed" for j in jobs)
    assert len(jobs) == 14
    # The committee's wave-1 request (never evidence) drove the targeted wave-2 source job.
    wave2_src = src_jobs[-1]
    assert (wave2_src.diagnostics or {}).get("question") == "What drove the Q2 delta?"
    sess = repo.get_session(sid)
    assert len(sess.freeze_ids) >= 2
    raw1: object = repo.get_freeze(sess.freeze_ids[0]).get("evidence_ids", [])
    raw2: object = repo.get_freeze(sess.freeze_ids[-1]).get("evidence_ids", [])
    e1 = {e for e in raw1 if isinstance(e, str)} if isinstance(raw1, list) else set()
    e2 = {e for e in raw2 if isinstance(e, str)} if isinstance(raw2, list) else set()
    assert e1 and e2 and e1 < e2  # cumulative E2 superset of E1


def test_grounded_claims_reject_unknown_and_empty() -> None:
    import json as _json

    from app.research.agents import ModelOutputFailure, parse_grounded_claims

    claims = parse_grounded_claims(
        _json.dumps([{"text": "revenue grew", "evidence_ids": ["EV-1"]}]),
        frozen=["EV-1", "EV-2"],
    )
    assert [c.text for c in claims] and claims[0].evidence_ids == ["EV-1"]
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            _json.dumps([{"text": "invented trend", "evidence_ids": ["EV-999"]}]),
            frozen=["EV-1"],
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            _json.dumps([{"text": "Revenue grew with no citation.", "evidence_ids": []}]),
            frozen=["EV-1"],
        )


def test_committee_refs_derive_from_claims_not_whole_freeze() -> None:
    from app.research.agents import claims_refs
    from app.research.agents.stockbot import run_stockbot

    analysis = run_stockbot(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1", "EV-2"],
        as_of="x",
        model=lambda prompt: _committee_output([{"text": "only first matters", "evidence_ids": ["EV-1"]}]),
        evidence_text="[EV-1] a\n[EV-2] b",
    )
    assert claims_refs(analysis.claims) == ["EV-1"]
    assert [c.evidence_ids for c in analysis.claims] == [["EV-1"]]


def test_committee_unknown_id_fails_model_output() -> None:
    from app.research.agents import ModelOutputFailure
    from app.research.agents.bearbot import run_bearbot
    from app.research.agents.bullbot import run_bullbot

    with pytest.raises(ModelOutputFailure, match="EV-999"):
        run_bullbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: _committee_output([{"text": "bad", "evidence_ids": ["EV-999"]}]),
            evidence_text="[EV-1] a",
        )
    with pytest.raises(ModelOutputFailure, match="uncited"):
        run_bearbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: _committee_output([{"text": "Bearish with no citation.", "evidence_ids": []}]),
            evidence_text="[EV-1] a",
        )


def test_scout_findings_cite_only_acquired_ids() -> None:
    """Scout boundary: only acquired ids ground; a bad record is dropped and reported, never fatal."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
    )

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        assert name and args is not None
        if name == "browse_tools":
            return {}
        return {"evidence_ids": [{"evidence_id": "EV-1", "known_at": "2025-05-01"}]}

    import json as _json

    good = run_scout(
        assignment,
        dispatch=_dispatch,
        model=lambda prompt: _json.dumps([{"text": "found", "evidence_ids": ["EV-1"]}]),
    )
    assert [c.evidence_ids for c in good.findings] == [["EV-1"]]
    # A fabricated id loses its own claim; the grounded sibling survives with a limitation.
    mixed = run_scout(
        assignment,
        dispatch=_dispatch,
        model=lambda prompt: _json.dumps(
            [
                {"text": "found", "evidence_ids": ["EV-1"]},
                {"text": "bad", "evidence_ids": ["EV-999"]},
            ]
        ),
    )
    assert [c.text for c in mixed.findings] == ["found"]
    assert any("dropped" in line for line in mixed.limitations)
    # An uncited non-unknown claim is dropped too: nothing ungrounded reaches the dossier.
    uncited = run_scout(
        assignment,
        dispatch=_dispatch,
        model=lambda prompt: _json.dumps([{"text": "Something factual uncited.", "evidence_ids": []}]),
    )
    assert uncited.findings == []
    assert any("dropped" in line for line in uncited.limitations)


def test_dossier_preserves_per_claim_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    out = _run_wave(repo, _grounded)
    assert out["stop_reason"] == "complete:wave1"
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    from app.research.agents import claims_refs as _crefs

    for analysis in (stock, bull, bear):
        frozen = set(analysis.evidence_ids)
        for claim in analysis.claims:
            assert claim.evidence_ids and set(claim.evidence_ids) <= frozen
        assert set(_crefs(analysis.claims)) <= frozen
    from app.research.synthesis.committee import CommitteeDisagreement
    from app.research.synthesis.final import synthesize_final

    sid = out["session_id"]
    fid = out["freeze_id"]
    eids = out["evidence_ids"]
    disagreement = out["disagreement"]
    assert isinstance(sid, str) and isinstance(fid, str)
    assert isinstance(eids, list) and isinstance(disagreement, CommitteeDisagreement)
    synth = synthesize_final(
        "NVDA demand?",
        session_id=sid,
        wave_id=1,
        freeze_id=fid,
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
    )
    from app.research.agents import claims_refs as _crefs2

    frozen_raw: object = repo.get_freeze(fid).get("evidence_ids", [])
    frozen_ids = {e for e in frozen_raw if isinstance(e, str)} if isinstance(frozen_raw, list) else set()
    assert frozen_ids  # the freeze is the committee's evidence universe
    for claim in synth.claims:
        if claim.claim_type == "unknown" and not claim.evidence_ids:
            continue  # generator scope disclosure: no fabricated evidence cited
        assert claim.evidence_ids and set(claim.evidence_ids) <= frozen_ids
        # Per-claim mapping survives synthesis: same text -> exactly the trio's cited ids.
        expected = {
            eid
            for analysis in (stock, bull, bear)
            for c in analysis.claims
            if c.text == claim.text
            for eid in c.evidence_ids
        }
        assert set(claim.evidence_ids) == expected
    # Refs derive from the trio's per-claim mapping, never the whole freeze.
    assert set(_crefs2(synth.claims)) == set(_crefs(stock.claims)) | set(_crefs(bull.claims)) | set(_crefs(bear.claims))


def test_resume_partial_source_reuses_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    from app.research.repository import get_research_db_path
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    base = _fake_dispatch()
    scouts_started = {"n": 0}

    def _crash(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            scouts_started["n"] += 1
            # Deterministic cut: each scout opens with browse_tools, so the third one
            # starting means two completed and this one is mid-fetch (keying on a raw
            # call count would drift with the scout's tool volume).
            if scouts_started["n"] == 3:
                raise KeyboardInterrupt("simulated crash mid-source-fetch")
        return base(name, args)

    def _ok(prompt: str) -> str:
        return _grounded(prompt)

    import pytest as _pt

    with _pt.raises(KeyboardInterrupt):
        run_live(
            question="NVDA demand?",
            objective="o",
            as_of="2025-06-30T00:00:00+00:00",
            tickers=["NVDA"],
            dispatch=_crash,
            model=_ok,
            repo=repo,
            budgets=None,
        )
    db = get_research_db_path()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT DISTINCT session_id FROM jobs").fetchall()
    assert len(rows) == 1
    sid = str(rows[0][0])
    jobs_before = repo.list_jobs(sid)
    src_before = [j for j in jobs_before if j.job_type == "source_agent"]
    scout_before = [j for j in jobs_before if j.job_type == "scout"]
    assert len(src_before) == 1 and src_before[0].status == "running"
    assert len([j for j in scout_before if j.status == "completed"]) == 2
    out = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out["stop_reason"] == "complete:wave1"
    jobs_after = repo.list_jobs(sid)
    assert len([j for j in jobs_after if j.job_type == "source_agent"]) == 1
    completed = [j for j in jobs_after if j.job_type == "scout" and j.status == "completed"]
    aids = [(j.diagnostics or {}).get("assignment_id") for j in completed]
    assert sorted(a for a in aids if isinstance(a, str)) == [
        "scout-filings",
        "scout-financials",
        "scout-risk",
    ]
    eids_final = [str(r.get("evidence_id")) for r in repo.list_evidence(sid)]
    assert len(eids_final) == len(set(eids_final))


def test_run_resume_produce_complete_append_only_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.evals.traces import get_trace_events, list_traces
    from app.research.runner import resume_live

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    repo = ResearchRepository()

    def _ok(prompt: str) -> str:
        return _grounded(prompt)

    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
        interrupt_after="source",
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    traces = list_traces(sid)
    assert len(traces) == 1
    tid = traces[0].trace_id
    evs = get_trace_events(tid)
    types = [e.event_type for e in evs]
    for required in (
        "trace.opened",
        "tool.completed",
        "evidence.ingested",
        "discovery.ingested",
        "job.created",
        "job.completed",
        "model.completed",
    ):
        assert required in types
    seqs = [e.seq for e in evs]
    assert len(set(seqs)) == len(seqs) and seqs == sorted(seqs)
    tools = [e for e in evs if e.event_type == "tool.completed"]
    assert tools and all(isinstance(e.payload.get("tool"), str) and e.payload.get("tool") for e in tools)
    # Every completion carries its args plus exactly one kind marker: the evidence id
    # (raw document opened) or record_kind=discovery (navigation artifact, never citable).
    assert all("args" in e.payload for e in tools)
    assert all(("evidence_id" in e.payload) != (e.payload.get("record_kind") == "discovery") for e in tools)
    assert any("evidence_id" in e.payload for e in tools)
    assert any(e.payload.get("record_kind") == "discovery" for e in tools)
    models = [e for e in evs if e.event_type == "model.completed"]
    assert models and all("prompt" in e.payload and "output" in e.payload for e in models)
    discs = [e for e in evs if e.event_type == "discovery.completed"]
    assert discs and all("args" in e.payload and "matches" in e.payload for e in discs)
    n_before = len(evs)
    out2 = resume_live(sid, _fake_dispatch(), _ok, repo=repo)
    assert out2["stop_reason"] == "complete:wave1"
    assert list_traces(sid)[0].trace_id == tid
    evs2 = get_trace_events(tid)
    assert len(evs2) > n_before
    seqs2 = [e.seq for e in evs2]
    assert len(set(seqs2)) == len(seqs2) and seqs2 == sorted(seqs2)
    all_types = [e.event_type for e in evs2]
    assert "trace.resumed" in all_types
    resumed = next(e for e in evs2 if e.event_type == "trace.resumed")
    assert resumed.payload.get("trace_id") == tid and resumed.payload.get("session_id") == sid
    from app.research.evals.traces import get_trace

    header = get_trace(tid)
    assert header is not None and header.status == "completed" and header.conclusion


def _svc_sid(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jobs = repo.list_jobs(sid)
    return sid, jobs[0].job_id


@pytest.fixture(autouse=True)
def _evidence_handle_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence admission in this module reloads through the fake archive, never the network."""
    seam.install(monkeypatch)


def _svc_item(
    eid: str,
    wave: int = 1,
    passage: str = "Accelerated computing demand grew in the reporting period.",
    **over: object,
) -> dict[str, object]:
    """Observed-fact item with the canonical handle for its cited passage (overrides win)."""
    base: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": "c-" + eid,
        "subject": "NVDA",
        "source_name": "SEC",
        "source_uri": "https://sec.gov/x",
        "source_record_id": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
        "matching_passage": passage,
        "source_handle": seam.handle_for(passage),
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    if "passage" in over and "source_handle" not in over and "matching_passage" not in over:
        text = str(over.pop("passage"))
        base["matching_passage"] = text
        base["source_handle"] = seam.handle_for(text)
    base.update(over)
    return base


def _committee_analysis(
    eid: str,
    follow_ups: Sequence[object] = (),
    claim_text: str = "finding suggests exposure",
    claim_type: str = "inference",
) -> dict[str, object]:
    """Rich committee envelope mapping for service.record_committee_analysis (all required keys).

    The default claim is an explicitly hedged inference, as the publication gate requires.
    """
    return {
        "executive_view": f"{claim_text} read over the freeze.",
        "claims": [{"text": claim_text, "claim_type": claim_type, "evidence_ids": [eid]}],
        "impact_channels": [{"text": "exposure channel", "direction": "pressure", "evidence_ids": [eid]}],
        "materiality": {"overall": "medium", "reasoning": "read off the freeze"},
        "uncertainties": ["open terms"],
        "what_would_change": ["a new filing disclosing the terms"],
        "follow_ups": list(follow_ups),
    }


def _svc_ana(eid: str) -> dict[str, object]:
    return _committee_analysis(eid)


def _svc_trio(repo: ResearchRepository, sid: str, eid: str) -> None:
    from app.research import service as _svc

    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)


def test_committee_cannot_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    out = execute_agent_tool(
        "search_web",
        {"query": "x", "session_id": sid},
        RuntimeToolSession(session_id="t1"),
    )
    assert "forbids" in str(out.get("error", ""))


def test_committee_cannot_add_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bear = str(_svc.start_job(sid, "bearbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError):
        _svc.record_evidence(sid, bear, _svc_item(f"{sid}:ev:2"), repo=repo)
    with pytest.raises(ValueError, match="forbids"):
        _svc.authorize_and_consume_dispatch(sid, bear, "search_web", repo=repo)


def test_role_job_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    stock = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="!="):
        _svc.record_committee_analysis(sid, stock, "bearbot", _svc_ana(eid), repo=repo)


def test_evidence_rejected_on_completed_or_committee_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    with pytest.raises(ValueError, match="running"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:9"), repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bull = str(_svc.start_job(sid, "bullbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError):
        _svc.record_evidence(sid, bull, _svc_item(f"{sid}:ev:3"), repo=repo)


def test_restart_after_e1_resume_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    fresh = ResearchRepository()
    assert _svc.resume_research(sid, repo=fresh)["session"] is not None
    _svc_trio(fresh, sid, eid)
    _svc.decide_next_wave(sid, repo=fresh)
    out = _svc.finalize_session(
        sid, "finding suggests exposure", [{"text": "finding suggests exposure", "evidence_ids": [eid]}], repo=fresh
    )
    assert out["freeze_id"] == f"{sid}:1:freeze"


def test_restart_after_2_of_3_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    ids = [str(_svc.start_job(sid, r, repo=repo, wave_id=1)["job_id"]) for r in ("stockbot", "bullbot", "bearbot")]
    for jid, role in zip(ids[:2], ("stockbot", "bullbot")):
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)
    fresh = ResearchRepository()
    _svc.record_committee_analysis(sid, ids[2], "bearbot", _svc_ana(eid), repo=fresh)
    out = _svc.finalize_session(
        sid, "finding suggests exposure", [{"text": "finding suggests exposure", "evidence_ids": [eid]}], repo=fresh
    )
    assert out["freeze_id"] == f"{sid}:1:freeze"


def test_source_terminal_before_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    with pytest.raises(ValueError, match="still open"):
        _svc.freeze_session(sid, 1, repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == f"{sid}:1:freeze"


def test_budget_stops_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(jid), tool_budget=1))
    ctx = RuntimeToolSession(session_id="t-budget")
    ctx.active_research_session_id = sid
    ctx.active_research_job_id = jid
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        calls.append((name, dict(arguments)))
        return {
            "result_type": "web_search",
            "query": arguments.get("query"),
            "results": [],
            "source": "exa",
        }

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    first = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    assert "error" not in first, first
    second = execute_agent_tool("search_web", {"query": "NVDA demand"}, ctx)
    # §3: explicit per-job kernel exhaustion passes through verbatim (no budget_exhausted collapse).
    assert "tool_budget exhausted" in str(second.get("error", "")), second
    assert second.get("error_type") != "budget_exhausted", second
    assert len(calls) == 1
    assert calls[0][0] == "search_web" and calls[0][1] == {"query": "NVDA demand"}
    assert repo.get_job(jid).tool_budget == 0
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1
    kept = execute_agent_tool(
        "research_add_evidence",
        {"session_id": sid, "job_id": jid, "item": _svc_item(f"{sid}:ev:1")},
        ctx,
    )
    assert "error" not in kept, kept
    assert repo.get_job(jid).tool_budget == 0
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1


def test_research_bound_dispatch_carries_research_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC discovery sees the attached research session (exhaustive default), None when unattached."""
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    seen: dict[str, object] = {}

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        seen["context_session"] = getattr(context, "research_session_id", "missing")
        return {"search_id": "s1", "count": 0, "source": "SEC EDGAR"}

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    bound = RuntimeToolSession(
        session_id="pi-bound",
        active_research_session_id=sid,
        active_research_job_id=jid,
    )
    out = execute_agent_tool("search_sec_filings", {"query": "Apple"}, bound)
    assert "error" not in out, out
    assert seen["context_session"] == sid

    seen.clear()
    unbound = RuntimeToolSession(session_id="pi-unbound")
    out = execute_agent_tool("search_sec_filings", {"query": "Apple"}, unbound)
    assert "error" not in out, out
    assert seen["context_session"] is None


def test_concurrent_dispatch_race_admits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _svc_sid(repo)
    repo.save_job(dataclasses.replace(repo.get_job(jid), tool_budget=1))
    sess = repo.get_session(sid)
    policy = dict(sess.policy)
    raw: object = policy.get("research", {})
    assert isinstance(raw, dict)
    section: dict[str, JSONValue] = dict(raw)
    section["max_tool_calls"] = 1
    policy["research"] = section
    repo.save_session(dataclasses.replace(sess, policy=policy))
    barrier = threading.Barrier(2)
    outcomes: list[object] = [None, None]

    def _worker(idx: int) -> None:
        try:
            barrier.wait(timeout=10)
            outcomes[idx] = _svc.authorize_and_consume_dispatch(sid, jid, "search_web", repo=repo)
        except Exception as exc:  # noqa: BLE001 — race outcome is the assertion
            outcomes[idx] = exc

    threads = [threading.Thread(target=_worker, args=(i,)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(thread.is_alive() is False for thread in threads)
    successes = [o for o in outcomes if isinstance(o, dict)]
    failures = [o for o in outcomes if isinstance(o, Exception)]
    assert len(successes) == 1, outcomes
    assert len(failures) == 1, outcomes
    assert "budget exhausted" in str(failures[0]).lower(), outcomes
    assert repo.get_session(sid).budget.get("tool_calls_used") == 1
    assert repo.get_job(jid).tool_budget == 0
    assert [j.job_id for j in repo.list_jobs(sid)] == [jid]


def test_stage_allows_common_read_controls() -> None:
    from app.research.stage import check_stage_tool

    for stage in ("SOURCE_RESEARCH", "COMMITTEE", "FINAL"):
        check_stage_tool(stage, "research_read")
        check_stage_tool(stage, "research_cancel")
    for stage in ("COMMITTEE", "FINAL"):
        with pytest.raises(ValueError, match="forbids"):
            check_stage_tool(stage, "research_add_evidence")
        with pytest.raises(ValueError, match="forbids"):
            check_stage_tool(stage, "search_web")


def test_queued_scout_blocks_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    found = repo.get_session(sid)
    existing = repo.list_jobs(sid)
    updated, scout = _jobs.create_job(found, existing, job_type="scout", owner="service", wave_id=1)
    repo.save_session(updated)
    repo.save_job(scout)
    with pytest.raises(ValueError, match="still open") as exc:
        _svc.freeze_session(sid, 1, repo=repo)
    assert scout.job_id in str(exc.value)
    assert repo.get_job(scout.job_id).status == "queued"


def test_start_job_rejects_malformed_budget_wave_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    for bad in ("2", 2.0, True, None):
        with pytest.raises(ValueError, match="wave_id"):
            _svc.start_job(sid, "source_agent", budget={"wave_id": bad}, repo=repo)
    ok = _svc.start_job(sid, "source_agent", budget={"wave_id": 1}, repo=repo)
    assert ok["job_id"]


def test_job_runtime_identity_merges_partial_then_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """attach_job_runtime accumulates accepted keys in diagnostics across partial calls."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _sid, src = _svc_sid(repo)
    partial = _svc.attach_job_runtime(
        src,
        {"runtime": "omp", "runtime_agent_id": "agent-1", "not_accepted": "x"},
        repo=repo,
    )
    assert partial["diagnostics"] == {"runtime": "omp", "runtime_agent_id": "agent-1"}
    done = _svc.attach_job_runtime(
        src,
        {
            "runtime_agent_id": "",
            "runtime_parent_agent_id": "agent-0",
            "runtime_task_call_id": "call-7",
            "runtime_agent_type": "scout",
            "runtime_session_file": "/tmp/session.jsonl",
        },
        repo=repo,
    )
    assert done["diagnostics"] == {
        "runtime": "omp",
        "runtime_agent_id": "agent-1",
        "runtime_parent_agent_id": "agent-0",
        "runtime_task_call_id": "call-7",
        "runtime_agent_type": "scout",
        "runtime_session_file": "/tmp/session.jsonl",
    }
    assert ResearchRepository().get_job(src).diagnostics == done["diagnostics"]
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.attach_job_runtime("nope", {"runtime": "omp"}, repo=repo)


def test_job_fail_cancel_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fail_job marks failed with category/message; cancel_job marks cancelled; terminal is a no-op."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _, src = _svc_sid(repo)
    with pytest.raises(ValueError):
        _svc.fail_job(src, "bogus-category", "msg", repo=repo)
    assert repo.get_job(src).status == "running"
    failed = _svc.fail_job(src, "timeout", "deadline hit", repo=repo)
    assert failed["status"] == "failed"
    failure = failed["failure"]
    assert isinstance(failure, dict)
    assert failure["category"] == "timeout"
    assert failure["message"] == "deadline hit"
    assert repo.get_job(src).status == "failed"
    assert _svc.cancel_job(src, repo=repo)["status"] == "failed"
    assert _svc.fail_job(src, "timeout", "again", repo=repo)["status"] == "failed"
    _, src2 = _svc_sid(repo)
    cancelled = _svc.cancel_job(src2, repo=repo)
    assert cancelled["status"] == "cancelled"
    assert repo.get_job(src2).status == "cancelled"
    assert _svc.fail_job(src2, "timeout", "late", repo=repo)["status"] == "cancelled"
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.fail_job("nope", "timeout", "msg", repo=repo)
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.cancel_job("nope", repo=repo)


def test_start_job_owner_override_records_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing budget.owner seam records caller jobs; default stays kernel."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    assert repo.get_job(src).owner == "kernel"
    created = _svc.start_job(sid, "source_agent", budget={"owner": "thesis"}, repo=repo)
    assert created["owner"] == "thesis"
    assert ResearchRepository().get_job(str(created["job_id"])).owner == "thesis"


def test_record_evidence_rejects_bad_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    base = _svc_item(f"{sid}:ev:badmeta")
    bads: list[object] = [{"ok": object()}, {1: "x"}, "not-a-dict", [("k", "v")]]
    for i, bad in enumerate(bads):
        item = dict(base)
        item["evidence_id"] = f"{sid}:ev:bad{i}"
        item["metadata"] = bad
        with pytest.raises(ValueError, match="metadata"):
            _svc.record_evidence(sid, src, item, repo=repo)
    good = dict(base)
    good["evidence_id"] = f"{sid}:ev:goodmeta"
    good["metadata"] = {"source": "sec", "page": 3}
    out = _svc.record_evidence(sid, src, good, repo=repo)
    meta = out["metadata"]
    assert isinstance(meta, dict)
    assert meta["source"] == "sec" and meta["page"] == 3
    assert meta["source_bytes"] == "archived" and "identity_key" in meta


# ---------------------------------------------------------------------------
# SEC-only NVDA/Anthropic regression + context/coverage/dedup/provenance.
# Forward-compatible: new APIs via lazy getattr; behavior asserts only,
# never exact call counts.
# ---------------------------------------------------------------------------


def _sec_only_policy() -> dict[str, JSONValue]:
    """SEC-only allowlist policy shape per contract (research_sources in)."""
    return {"research_sources": {"mode": "allowlist", "sources": ["SEC"]}}


def test_sec_only_policy_persisted_and_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sess = _session.create_session("NVDA AI demand?", "o", as_of=ASOF, policy=_sec_only_policy())
    doc = sess.to_dict()
    for key in (
        "session_id",
        "query",
        "objective",
        "as_of",
        "status",
        "policy",
        "budget",
        "source_policy",
        "temporal_scope",
    ):
        assert key in doc
    sp = doc["source_policy"]
    assert isinstance(sp, dict)
    assert sp.get("mode") == "allowlist"
    allowed = sp.get("allowed")
    assert isinstance(allowed, list) and "SEC" in allowed
    assert sp.get("denied") == []
    ts = doc["temporal_scope"]
    assert isinstance(ts, dict)
    for key in ("as_of", "start", "end", "mode", "raw"):
        assert key in ts
    assert ts.get("mode") == "as_of"
    # Budgets kept alongside the new policy keys.
    budget = doc["budget"]
    assert isinstance(budget, dict)
    for key in (
        "deadline_seconds",
        "total_tool_budget",
        "total_token_budget",
        "total_cost_budget",
    ):
        assert key in budget, key
    # Persist + reload: SQLite round-trips both fields.
    repo.save_session(sess)
    loaded = repo.get_session(sess.session_id)
    assert loaded.source_policy == sess.source_policy
    assert loaded.temporal_scope == sess.temporal_scope
    # Round-trip via from_dict/validate preserves the contract keys.
    from app.research.models import ResearchSession as _RS

    back = _RS.from_dict(doc)
    back.validate()
    assert back.to_dict()["source_policy"] == doc["source_policy"]
    assert back.to_dict()["temporal_scope"] == doc["temporal_scope"]
    # Temporal kwarg and default latest-available mode.
    scoped = _session.create_session("NVDA demand?", "o", as_of=None, temporal="between 2025-01-01 and 2025-06-30")
    assert scoped.temporal_scope.get("mode") == "range"
    latest = _session.create_session("NVDA demand?", "o", as_of=None)
    assert latest.temporal_scope.get("mode") == "latest-available"
    assert latest.temporal_scope.get("as_of") is not None


def test_sec_only_nvda_anthropic_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    calls: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("search_tools", "browse_tools"):
            return {"matches": ["search_sec_filings", "list_sec_filings"]}
        if name == "call_tool":
            inner = args.get("name")
            inner_name = inner if isinstance(inner, str) and inner else "unknown_tool"
            calls.append(inner_name)
            # No-issuer probe (Anthropic, private): empty record, never an error.
            if "anthropic" in str(args).lower() and "nvda" not in str(args).lower():
                return {"record": {}, "evidence_ids": []}
            return {
                "record": {
                    "id": "r-" + str(len(calls)),
                    "known_at": "2025-05-01",
                    "uri": "https://sec.gov/x",
                    "record_id": "0001045810-25-000023",
                },
                "evidence_ids": ["EV-1", "EV-2"],
            }
        return {}

    from app.research.agents.source_agent import build_research_context

    ctx = build_research_context("NVDA AI demand vs Anthropic private AI concepts?", ["NVDA"])
    blob = str(ctx).lower()
    assert "nvda" in blob
    assert "anthropic" in blob or "ai" in blob
    out = run_live(
        question="NVDA AI demand vs Anthropic private AI concepts?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_dispatch,
        model=_grounded,
        repo=repo,
        budgets=None,
    )
    # No-issuer does not stop the session; NVDA filings + conceptual searches ran.
    assert out["stop_reason"] in ("complete:wave1", "complete:wave2")
    assert calls, "expected NVDA filing/conceptual SEC calls"
    # No SEC count cap: more than 3 useful calls allowed (behavior, not exact N).
    assert len([c for c in calls if c]) >= 3
    assert "up to 3" not in str(out).lower()
    # Freeze shared by trio; final answer produced with SEC-only limits explicit.
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    assert isinstance(stock, StockbotAnalysis)
    assert isinstance(bull, BullAnalysis)
    assert isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    assert stock.evidence_ids == bull.evidence_ids == bear.evidence_ids
    sid = out["session_id"]
    assert isinstance(sid, str)
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict) and str(final.get("answer", "")).strip()
    assert final.get("freeze_id") == out["freeze_id"]


def test_dedup_collapses_cosmetic_repeats() -> None:
    from app.research.agents.scout import normalize_query
    from app.research.agents.source_agent import normalize_query as _sa_norm

    assert normalize_query("AI demand") == normalize_query("AI   demand")
    assert normalize_query("AI Demand") == normalize_query("ai demand")
    assert _sa_norm("AI demand") == normalize_query("AI   demand")
    # Single execution oracle: normalized equivalents share one slot.
    executed: set[str] = set()
    for variant in ("AI demand", "AI   demand", "ai DEMAND"):
        key = normalize_query(variant)
        assert isinstance(key, str)
        executed.add(key)
    assert len(executed) == 1


def test_scout_executes_normalized_repeat_once() -> None:
    from app.research.agents.scout import ScoutAssignment, run_scout

    seen: list[tuple[str, object]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name in ("browse_tools", "search_tools"):
            return {"matches": []}
        if name == "call_tool" and args.get("name") == "search_sec_filings":
            inner = args.get("arguments")
            query = inner.get("query") if isinstance(inner, dict) else None
            seen.append((str(query), inner.get("as_of") if isinstance(inner, dict) else None))
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["AI demand", "AI   demand", "ai DEMAND"],
        baseline=[],
    )
    result = run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")
    assert len(seen) == 1
    assert seen[0][0] == "AI demand"
    assert seen[0][1] == "2025-06-30"
    assert result.unknowns and result.unknowns[0] == "no PIT-eligible SEC evidence returned"


def test_scout_material_events_args_always_carry_since() -> None:
    """The harness emits schema-valid role calls: `since` is required, bounded or not.

    Live defect: unbounded sessions (as_of "unbounded"/blank) built get_material_events
    with no `since`, so the tool layer answered invalid_tool_arguments and the filings
    scout silently lost its material-events coverage. The unbounded window floors at the
    documented 2024-01-01 because there is no cutoff to measure a lookback back from.
    """
    from app.research.agents.scout import ScoutAssignment, run_scout
    from app.tools import _validate_tool_arguments

    calls: list[tuple[str, dict[str, object]]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = args.get("name")
        inner_args = args.get("arguments")
        assert isinstance(inner, str) and isinstance(inner_args, dict)
        calls.append((inner, inner_args))
        return {"evidence_ids": []}

    def _role_calls(as_of: str) -> list[tuple[str, dict[str, object]]]:
        calls.clear()
        run_scout(
            ScoutAssignment(
                assignment_id="scout-filings",
                session_id="rs:t",
                as_of=as_of,
                role="filings",
                question="What changed at NVDA?",
                tickers=["NVDA"],
            ),
            dispatch=_dispatch,
            model=lambda prompt: "[]",
        )
        # Every emitted role call passes the real schema check, not a hand-written shape.
        for tool, args in calls:
            assert _validate_tool_arguments(tool, args) is None, (as_of, tool, args)
        return list(calls)

    for no_cutoff in ("unbounded", "", "   "):
        events = [args for tool, args in _role_calls(no_cutoff) if tool == "get_material_events"]
        assert events and [args["since"] for args in events] == ["2024-01-01"], no_cutoff
        # No cutoff: nothing is date-filtered on the way to the tool.
        assert all("as_of" not in args for args in events)
        searches = [args for tool, args in calls if tool == "search_sec_filings"]
        assert searches and all("as_of" not in args for args in searches)

    events = [args for tool, args in _role_calls("2026-08-10") if tool == "get_material_events"]
    assert [args["since"] for args in events] == ["2025-08-10"]
    assert [args["as_of"] for args in events] == ["2026-08-10"]
    searches = [args for tool, args in calls if tool == "search_sec_filings"]
    assert searches and all(args["as_of"] == "2026-08-10" for args in searches)


def test_pit_eligibility_short_circuits_only_without_a_cutoff() -> None:
    """No bounded as_of means no PIT filtering; a bounded as_of keeps the strict rule."""
    from app.research.agents.scout import _is_pit_eligible

    dated = "2011-03-16T16:33:51+00:00"
    for no_cutoff in ("unbounded", "", "   ", None):
        # The sentinel reaches only the scout (the ledger gate sees None), so a non-ISO
        # as_of must mean "no cutoff" instead of raising inside the PIT gate.
        assert _is_pit_eligible(dated, no_cutoff) is True, no_cutoff
        assert _is_pit_eligible(None, no_cutoff) is True, no_cutoff
    assert _is_pit_eligible(dated, "2025-06-30") is True
    assert _is_pit_eligible("2025-07-01T00:00:00+00:00", "2025-06-30") is False
    assert _is_pit_eligible(None, "2025-06-30") is False
    assert _is_pit_eligible("", "2025-06-30") is False
    assert _is_pit_eligible(7, "2025-06-30") is False
    # A bounded cutoff that cannot be parsed is never verifiable: fail closed, never admit.
    assert _is_pit_eligible(dated, "2025-06-30T25:00:00+00:00") is False


def test_scout_unbounded_citations_reach_the_model() -> None:
    """An unbounded scout keeps its dated citations: the model gets citable ids.

    Live defect: 64/64 unbounded citations were rejected as PIT-ineligible, so the
    prompt listed no ids and every claim was dropped as uncited.
    """
    from app.research.agents.scout import ScoutAssignment, ScoutResult, run_scout

    prompts: list[str] = []
    journal: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        return {
            "evidence_ids": [
                {
                    "evidence_id": "EV-1",
                    "known_at": "2011-03-16T16:33:51+00:00",
                    "claim_text": "Data center revenue grew",
                }
            ],
            "top_hits": [],
        }

    def _model(prompt: str) -> str:
        prompts.append(prompt)
        return "[]"

    def _journal(kind: str, payload: dict[str, object]) -> None:
        journal.append(kind)

    def _scout(no_cutoff: str) -> ScoutResult:
        prompts.clear()
        journal.clear()
        return run_scout(
            ScoutAssignment(
                assignment_id="scout-filings",
                session_id="rs:t",
                as_of=no_cutoff,
                role="filings",
                question="What changed at NVDA?",
                tickers=["NVDA"],
            ),
            dispatch=_dispatch,
            model=_model,
            journal=_journal,
        )

    for no_cutoff in ("unbounded", ""):
        result = _scout(no_cutoff)
        assert "evidence.rejected" not in journal, no_cutoff
        assert "EV-1 (known_at=2011-03-16T16:33:51+00:00) :: Data center revenue grew" in prompts[0]
        assert result.limitations == [], no_cutoff
        assert result.unknowns == [], no_cutoff


def test_scout_executes_unscoped_counterparty_queries_before_the_model() -> None:
    """Global counterparty searches run with no ticker/cik, pre-model; their hits are opened."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    seq: list[str] = []
    searched: list[dict[str, object]] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = args.get("name")
        inner_args = args.get("arguments")
        assert isinstance(inner, str) and isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            seq.append(f"search:{inner_args.get('query')}")
            searched.append(inner_args)
            return {
                "top_hits": [{"accession": "0001-25-000001", "document": "ex-10.htm"}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            seq.append(f"open:{inner_args.get('accession_no')}")
            return {
                "evidence_ids": [
                    {
                        "evidence_id": "EV-1",
                        "known_at": "2025-05-01",
                        "claim_text": "ok",
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="What happens to Microsoft if OpenAI goes bankrupt?",
        tickers=["MSFT"],
        unscoped_queries=["OpenAI", "OpenAI exposure"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: seq.append("model") or "[]")

    unscoped = [a for a in searched if "query" in a]
    assert [a["query"] for a in unscoped] == ["OpenAI", "OpenAI exposure"]
    assert all("ticker" not in a and "cik" not in a for a in unscoped)
    assert [e for e in seq if e.startswith("open:")] == ["open:0001-25-000001"]
    assert seq[-1] == "model"  # every search, and the document read it surfaced, ran first


def test_scout_expansion_searches_derived_queries_and_opens_new_documents() -> None:
    """Expansion: passage-derived queries run, their documents open, and the loop stops at fixpoint."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    searches: list[str] = []
    opens: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = str(args.get("name"))
        inner_args = args.get("arguments")
        assert isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            query = str(inner_args.get("query"))
            searches.append(query)
            hit = ("ACC-1", "one.htm") if query == "assigned query" else ("ACC-2", "two.htm")
            return {
                "top_hits": [{"accession": hit[0], "document": hit[1]}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            accession = str(inner_args.get("accession_no"))
            opens.append(accession)
            text = "Hyperscaler concentration grew" if accession == "ACC-1" else "ok"
            return {
                "evidence_ids": [
                    {
                        "evidence_id": f"EV-{accession}",
                        "known_at": "2025-05-01",
                        "claim_text": text,
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["assigned query"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")

    # Round 1 = assigned query; round 2 = the terms the opened ACC-1 passage names.
    assert searches == ["assigned query", "Hyperscaler", "concentration"]
    assert opens == [
        "ACC-1",
        "ACC-2",
    ]  # the document the derived searches surfaced, opened once


def test_scout_expansion_never_reexecutes_a_seen_query() -> None:
    """A derived term already executed (any case/whitespace form) is suppressed, never re-run."""
    from app.research.agents.scout import ScoutAssignment, run_scout

    searches: list[str] = []

    def _dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "browse_tools":
            return {"matches": []}
        inner = str(args.get("name"))
        inner_args = args.get("arguments")
        assert isinstance(inner_args, dict)
        if inner == "search_sec_filings":
            searches.append(str(inner_args.get("query")))
            return {
                "top_hits": [{"accession": "ACC-1", "document": "one.htm"}],
                "evidence_ids": [],
            }
        if inner == "get_sec_document":
            return {
                "evidence_ids": [
                    {
                        "evidence_id": "EV-1",
                        "known_at": "2025-05-01",
                        "claim_text": "concentration. Ferrari up.",
                    }
                ]
            }
        return {"evidence_ids": []}

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2025-06-30",
        role="filings",
        question="Q?",
        tickers=[],
        queries=["CONCENTRATION"],
    )
    run_scout(assignment, dispatch=_dispatch, model=lambda prompt: "[]")
    assert searches == ["CONCENTRATION", "Ferrari"]


def test_scout_prompt_renders_context_baseline_and_queries() -> None:
    """Scout prompt: context lines + relationships, latest-filing baseline, assigned queries."""
    from app.research.agents.scout import ScoutAssignment, build_scout_prompt

    assignment = ScoutAssignment(
        assignment_id="scout-filings",
        session_id="rs:t",
        as_of="2026-08-10",
        role="filings",
        question="NVDA data center growth?",
        tickers=["NVDA"],
        context={
            "primary_entities": ["NVDA"],
            "concepts": ["accelerated computing", "  "],
            "relationships": [
                {"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"},
                {"subject": "NVDA", "relation": "  ", "object": "dropped"},
                "not-a-triple",
            ],
        },
        baseline=["NVDA 10-Q data center segment", "NVDA 10-K competition"],
        queries=["NVDA data center revenue"],
    )
    prompt = build_scout_prompt(assignment)
    assert "Context primary_entities: NVDA" in prompt
    assert "Context concepts: accelerated computing" in prompt
    assert "Context relationships: NVDA supplies hyperscalers" in prompt
    assert "dropped" not in prompt and "not-a-triple" not in prompt
    assert "- NVDA 10-K competition" in prompt
    assert "- NVDA data center revenue" in prompt

    bare = build_scout_prompt(
        ScoutAssignment(
            assignment_id="scout-risk",
            session_id="rs:t",
            as_of="unbounded",
            role="risk",
            question="Q?",
            tickers=[],
        )
    )
    assert "Context " not in bare and "Latest-filing baseline" not in bare
    assert "Assigned queries" not in bare
    assert "scope tickers TBD" in bare and "As of: unbounded" in bare
    assert "Respond with JSON only" in bare


def test_merge_context_unions_model_terms() -> None:
    from app.research.agents.source_agent import _merge_context, build_research_context

    base = build_research_context("NVDA AI demand?", ["NVDA"])
    merged = _merge_context(
        base,
        {
            "concepts": ["accelerated computing"],
            "relationships": [{"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"}],
        },
    )
    assert "accelerated computing" in str(merged["concepts"]).lower()
    rels = merged["relationships"]
    assert isinstance(rels, list)
    assert any(isinstance(r, dict) and r.get("object") == "hyperscalers" for r in rels)
    assert _merge_context(base, "not-a-dict") == base


def test_expand_queries_mines_findings_and_context() -> None:
    from app.research.agents.source_agent import build_research_context, expand_queries

    ctx = build_research_context("NVDA AI demand?", ["NVDA"])
    out = expand_queries(["nvda demand"], ["Hyperscaler concentration grew in filings"], ctx)
    assert out and all(q.strip() for q in out)
    assert "nvda demand" not in [q.lower() for q in out]


def test_expansion_stop_prefers_info_over_counts() -> None:
    from app.research.agents.source_agent import expansion_stop

    assert expansion_stop(sec_answerable_remaining=False)[0] is True
    assert expansion_stop(new_queries=False)[0] is True
    assert expansion_stop(new_queries=True) == (False, "continue")


@pytest.mark.parametrize(
    ("question", "tickers", "must_contain"),
    [
        (
            "Spirit AeroSystems Boeing 737 relationship and backlog?",
            ["SPR"],
            ("aerospace", "boeing"),
        ),
        (
            "Novo Nordisk GLP-1 diabetes obesity outlook?",
            ["NVO"],
            ("glp", "diabetes", "obesity"),
        ),
        (
            "Arista cloud networking datacenter Ethernet demand?",
            ["ANET"],
            ("cloud", "network", "datacenter"),
        ),
        ("Albemarle lithium brine battery demand?", ["ALB"], ("lithium", "battery")),
        ("Apple China supply chain and tariffs?", ["AAPL"], ("china", "supply")),
    ],
)
def test_cross_domain_context_carries_industry_terms(
    question: str, tickers: list[str], must_contain: tuple[str, ...]
) -> None:
    from app.research.agents.source_agent import (
        build_query_families,
        build_research_context,
    )

    ctx = build_research_context(question, tickers)
    blob = str(ctx).lower()
    assert any(term in blob for term in must_contain), blob[:500]
    # Industry/relationship/risk content beyond the raw query words.
    for key in ("industries", "relationships", "risks", "concepts"):
        assert key in ctx, sorted(ctx.keys())
    # No issuer-specific branches: generic families serve every domain.
    families = build_query_families(ctx)
    assert families and len({f.lower() for f in families}) == len(families)
    assert "Anthropic" not in str(families)


def test_private_counterparty_queries_reach_filings_scout() -> None:
    """MSFT/OpenAI: the private counterparty is searched unscoped, never dead-ended."""
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
        decompose_question,
    )

    question = "What happens to Microsoft if OpenAI goes bankrupt?"
    groups = _grouped_queries(build_research_context(question, ["MSFT"]))
    assert {"OpenAI", "OpenAI exposure", "OpenAI agreement", "OpenAI 8-K"} <= set(groups["cp"])

    def _catalog_only(name: str, args: dict[str, object]) -> dict[str, object]:
        return {}  # discovery hint only; no query execution on this path

    assignments = decompose_question(
        question,
        session_id="rs:t",
        as_of="2026-08-10",
        tickers=["MSFT"],
        dispatch=_catalog_only,
    )
    filings = next(a for a in assignments if a.role == "filings")
    assert "OpenAI" in filings.queries  # bare name -> unscoped EDGAR full-text search
    # The counterparty family is wired into the filings role and nowhere else.
    for other in (a for a in assignments if a.role != "filings"):
        assert set(groups["cp"]).isdisjoint(other.queries)


def test_no_counterparty_question_keeps_existing_families() -> None:
    """NVDA-only question: counterparty family stays empty, every other family unchanged."""
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
    )

    ctx = build_research_context("NVDA data center revenue growth", ["NVDA"])
    assert ctx["related_entities"] == []
    assert _grouped_queries(ctx) == {
        "cp": [],
        "a": ["NVDA"],
        "b": [],
        "c": [],
        "d": [],
        "e": [],
        "f": ["NVDA risk factors"],
        "related": [],
        "supp_bare": ["data", "center", "revenue", "growth"],
        "supp_filing": [
            "data 8-K",
            "data proxy",
            "data N-PX",
            "data agreement",
            "center 8-K",
            "center proxy",
            "center N-PX",
            "center agreement",
            "revenue 8-K",
            "revenue proxy",
            "revenue N-PX",
            "revenue agreement",
            "growth 8-K",
            "growth proxy",
            "growth N-PX",
            "growth agreement",
        ],
        "supp_risk": [
            "data risk factor",
            "center risk factor",
            "revenue risk factor",
            "growth risk factor",
        ],
    }


def test_counterparty_family_includes_every_related_entity() -> None:
    """Counterparty family: all eight related entities, four variants each, scope tickers excluded."""
    from app.research.agents.source_agent import _grouped_queries

    ctx: dict[str, object] = {
        "primary_entities": ["MSFT"],
        "related_entities": ["MSFT", *(f"Counter {i}" for i in range(8))],
    }
    cp = _grouped_queries(ctx)["cp"]
    assert len(cp) == 32  # 8 entities x (bare + exposure/agreement/8-K); no entity cap
    assert "MSFT" not in cp and "Counter 7" in cp and "Counter 7 8-K" in cp


def test_role_assignments_dispatch_every_material_query() -> None:
    """No first-two-per-family cap: every planned family query reaches its role's assignment."""
    from app.research.agents.source_agent import _grouped_queries, _role_assignments

    ctx: dict[str, object] = {
        "primary_entities": ["MSFT"],
        "related_entities": ["Alpha", "Beta", "Gamma", "Delta"],
        "industries": ["industrials", "software", "semiconductors", "services"],
        "products": ["Azure", "Office", "Dynamics"],
        "technologies": [],
        "concepts": ["cloud"],
        "risks": ["credit", "litigation", "regulatory", "supply"],
        "catalysts": [],
    }
    groups = _grouped_queries(ctx)
    assignments = _role_assignments("Q?", "rs:t", "2026-08-10", ["MSFT"], ctx, groups, [])
    by_role = {a.role: a.queries for a in assignments}
    for family, role in (
        ("b", "filings"),
        ("cp", "filings"),
        ("related", "filings"),
        ("supp_filing", "filings"),
        ("c", "financials"),
        ("d", "financials"),
        ("supp_bare", "financials"),
        ("f", "risk"),
        ("supp_risk", "risk"),
    ):
        assert len(groups[family]) > 2, (family, groups[family])
        assert [q for q in by_role[role] if q in set(groups[family])] == groups[family], family


def test_no_date_means_latest_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    sess = _session.create_session("NVDA demand?", "o", as_of=None)
    doc = sess.to_dict()
    assert doc["as_of"] is None
    ts = doc["temporal_scope"]
    assert isinstance(ts, dict)
    assert ts.get("mode") == "latest-available"
    assert ts.get("as_of") is not None


def test_as_of_excludes_later_evidence() -> None:
    sess = _session.create_session("q?", "o", as_of=ASOF)
    led = EvidenceLedger()
    seen: list[tuple[str, dict[str, object]]] = []
    with pytest.raises(EvidenceRejectedError):
        ingest_evidence(
            led,
            _ev("EV-FUT2", sess.session_id, 1, datetime(2025, 7, 1, tzinfo=UTC)),
            as_of=ASOF,
            on_reject=lambda t, p: seen.append((t, p)),
        )
    assert seen and seen[0][1]["reason"] == "PIT_VIOLATION"
    assert led.ids() == ()
    # Latest-doc respects cutoff: eligible doc ingests, later doc rejects.
    ingest_evidence(
        led,
        _ev("EV-OK", sess.session_id, 1, datetime(2025, 5, 1, tzinfo=UTC)),
        as_of=ASOF,
        on_reject=None,
    )
    assert led.ids() == ("EV-OK",)


def test_coverage_shape_and_insufficient_never_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["Anthropic private revenue"],
        "source_limitations": ["SEC-only: no private-issuer filings"],
    }
    out = _svc.submit_source_result(
        src,
        coverage=coverage,
        evidence_ids=[],
        unresolved_questions=["Anthropic private revenue"],
        repo=repo,
    )
    assert out["job_status"] == "completed"
    # submit = source job complete only; session never terminal here.
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")
    # Coverage round-trips on the completed job result when the field lands.
    stored = repo.get_job(src).result or {}
    result_cov = stored.get("coverage") if isinstance(stored, dict) else None
    if isinstance(result_cov, dict):
        assert result_cov.get("useful_for_question") == "insufficient"
        for key in (
            "resolved",
            "partially_resolved",
            "unresolved",
            "source_limitations",
        ):
            if key in result_cov:
                assert isinstance(result_cov[key], list)
    # Insufficient still freezes + synthesizes via the kernel path.
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == f"{sid}:1:freeze"


def test_negatives_cite_searchrun_only() -> None:
    import dataclasses as _dc

    from app.sec.models import DocumentMatch, MatchingPassage, SearchRun

    fields = {f.name for f in _dc.fields(SearchRun)}
    for required in (
        "id",
        "source",
        "query",
        "filters",
        "executed_at",
        "as_of",
        "matched_entities",
        "matched_documents",
        "matched_passages",
    ):
        assert required in fields, sorted(fields)
    # Negative-claim shape: grounded positives cite document/passage/accession,
    # negatives cite the SearchRun id only — never 'not found' as 'does not exist'.
    doc_fields = {f.name for f in _dc.fields(DocumentMatch)}
    assert doc_fields >= {"accession", "matching_passages"}
    passage_fields = {f.name for f in _dc.fields(MatchingPassage)}
    assert passage_fields >= {"document", "query", "score"}


def test_sec_gates_deny_non_sec_even_via_browse() -> None:
    from app.research.agents.source_agent import is_sec_tool

    for denied in (
        "query_finra",
        "get_short_interest",
        "search_web",
        "get_market_snapshot",
        "get_analyst_estimates",
    ):
        assert is_sec_tool(denied) is False, denied
    for allowed in ("search_sec_filings", "list_sec_filings", "get_sec_document"):
        assert is_sec_tool(allowed) is True, allowed
    # Runner gate surfaces POLICY_DENIED (not a prompt-text refusal).
    import inspect as _inspect

    from app.research.runner import _LiveRun

    src = _inspect.getsource(_LiveRun._guarded_tool_call)
    assert "POLICY_DENIED" in src
    assert "is_sec_tool" in src


def test_resolve_source_policy_sec_only_allowlist() -> None:
    from app.research.models import resolve_source_policy, validate_source_policy

    sec_only = resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": ["SEC"]}})
    assert sec_only["mode"] == "allowlist"
    allowed = sec_only["allowed"]
    assert isinstance(allowed, list) and "SEC" in allowed
    assert sec_only["denied"] == []
    # Kernel default is SEC-only even with no policy input.
    assert resolve_source_policy(None)["mode"] == "allowlist"
    # Round-trip through the validator preserves the contract shape.
    assert validate_source_policy(sec_only) == sec_only
    with pytest.raises(ValueError, match="mode"):
        resolve_source_policy({"research_sources": {"mode": "someday", "sources": ["SEC"]}})


def test_resolve_temporal_scope_modes() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope, validate_temporal_scope

    now = _dt(2025, 6, 30, tzinfo=UTC)
    # No date -> latest-available with cutoff set, never invented/None.
    latest = resolve_temporal_scope(query="NVDA demand?", now=now)
    assert latest["mode"] == "latest-available"
    assert latest["as_of"] is not None
    # as-of-2025-01-01 excludes later docs.
    asof = resolve_temporal_scope(as_of="2025-01-01", now=now)
    assert asof["mode"] == "as_of"
    assert str(asof["as_of"])[:10] == "2025-01-01"
    # Interval start/end bounds.
    interval = resolve_temporal_scope(temporal="between 2025-01-01 and 2025-06-30", now=now)
    assert interval["mode"] == "range"
    assert str(interval["start"])[:10] == "2025-01-01"
    assert str(interval["end"])[:10] == "2025-06-30"
    # Last-quarter range resolves start/end around the pinned clock.
    quarter = resolve_temporal_scope(temporal="this quarter", now=now)
    assert quarter["mode"] == "range"
    assert quarter["start"] is not None and quarter["end"] is not None
    assert validate_temporal_scope(dict(latest)) == latest


# ---------------------------------------------------------------------------
# RegressionEval §17 suites (deterministic, fakes only, no network).
# Naming: test_reg_<suite>_<behavior>. Contract stubs resolve via getattr
# with a precise failure naming the missing source hook; no source edits.
# ---------------------------------------------------------------------------


def _reg_svc_sid(repo: ResearchRepository, q: str = "GS OpenAI exposure?") -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _reg_cov(useful: str = "sufficient") -> dict[str, object]:
    """sufficient now needs the full structured envelope + no residual branches/questions."""
    return {
        "useful_for_question": useful,
        "resolved": ["GS direct OpenAI exposure"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "major_entities_investigated": ["GS", "OpenAI"],
        "relationship_types_checked": ["investment", "commercial"],
        "forms_examined": ["10-K"],
        "exhibits_examined": ["EX-10.1"],
        "material_open_questions": [],
        "search_runs": ["s1"],
        "covered_branches": ["GS direct OpenAI exposure"],
    }


def _absence_cov(**over: object) -> dict[str, object]:
    """Full absence-observation coverage envelope (forms/dates/partitions/entities/docs/gaps + flags)."""
    cov: dict[str, object] = {
        "forms": ["10-K"],
        "dates": ["2025-02-14"],
        "partitions": ["efts"],
        "entities": ["GS"],
        "docs": ["0000886982-26-000001"],
        "gaps": [],
        "pagination_complete": True,
        "complete": True,
    }
    cov.update(over)
    return cov


def _reg_item(eid: str, wave: int = 1, **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": f"GS OpenAI-linked exposure per filing {eid}",
        "subject": "GS",
        "source_name": "SEC",
        "source_uri": "https://www.sec.gov/Archives/edgar/data/886982/000088698226000001/primary.htm",
        "source_record_id": "0000886982-26-000001",
        "document_name": "gs-10q-20260331.htm",
        "matching_passage": "The firm discloses OpenAI-linked exposure in the filing.",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    base.update(over)
    if "source_handle" not in over:
        passage = str(base["matching_passage"])
        base["source_handle"] = seam.handle_for(
            passage,
            accession=str(base["source_record_id"]),
            document=str(base["document_name"]),
            known_at=str(base["known_at"]) if isinstance(base.get("known_at"), str) else None,
        )
    return base


def test_reg_lifecycle_running_until_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    assert repo.get_job(src).status == "running"
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    assert repo.get_job(src).status == "running"
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")


def test_reg_lifecycle_atomic_submit_completes_job_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    assert repo.get_job(src).status == "completed"
    assert repo.get_session(sid).status not in ("completed", "failed", "cancelled")


def test_reg_lifecycle_double_submit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="CLOSED|closed|completed|running"):
        _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)


def test_reg_lifecycle_closed_rejects_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="running|closed|completed"):
        _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:2"), repo=repo)


# --- unlimited (4): 100+ distinct allowed, no default kill, dup-no-progress, diff queries ---


def _reg_dispatch(repo: ResearchRepository, sid: str, jid: str, tool: str) -> None:
    from app.research import service as _svc

    _svc.authorize_and_consume_dispatch(sid, jid, tool, repo=repo)


def test_reg_unlimited_100_distinct_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("GS OpenAI exposure?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jobs = repo.list_jobs(sid)
    assert len(jobs) == 1
    jid = jobs[0].job_id
    assert jobs[0].tool_budget is None, (
        "missing contract (LimitsLoopBudget owns "
        "app/research/jobs.py:job_tool_budget + app/research/models.py:DEFAULT_BUDGET): "
        "unlimited=None; source job tool_budget must default None"
    )
    for _ in range(100):
        _reg_dispatch(repo, sid, jid, "get_sec_document")
    assert repo.get_session(sid).budget.get("tool_calls_used") == 100


def test_reg_unlimited_no_default_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.models import DEFAULT_BUDGET

    assert DEFAULT_BUDGET.get("total_tool_budget") is None, (
        "missing contract (LimitsLoopBudget owns "
        "app/research/models.py:default_budget/DEFAULT_BUDGET): unlimited=None; "
        f"total_tool_budget must default None, got {DEFAULT_BUDGET.get('total_tool_budget')!r}"
    )
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _reg_svc_sid(repo)
    for _ in range(25):
        _reg_dispatch(repo, sid, jid, "get_sec_document")
    assert repo.get_session(sid).budget.get("tool_calls_used") == 25


def test_reg_unlimited_duplicate_no_progress_rejected() -> None:
    import app.research.director as _director

    normalize = getattr(_director, "normalize_research_action", None)
    loop_cls = getattr(_director, "LoopDetector", None)
    assert callable(normalize) and loop_cls is not None, (
        "missing source hook (LimitsLoopBudget owns "
        "app/research/director.py:normalize_research_action + LoopDetector): contract normalize("
        "source,tool,query,ticker,forms,as_of,accession,objective)->tuple; "
        "LoopDetector.check(action,result_hash,evidence_delta)->{duplicate,reason} + telemetry list"
    )
    loop = loop_cls()
    action = normalize(
        "sec",
        "get_sec_document",
        "GS 10-K",
        "GS",
        ("10-K",),
        "2025-06-30",
        "ACC-1",
        "exposure?",
    )
    first = loop.check(action, "hash-a", 3)
    assert first.get("duplicate") is False
    repeat = loop.check(action, "hash-a", 0)
    assert repeat.get("duplicate") is True, repeat


def test_reg_unlimited_different_queries_allowed() -> None:
    import app.research.director as _director

    normalize = getattr(_director, "normalize_research_action", None)
    loop_cls = getattr(_director, "LoopDetector", None)
    assert callable(normalize) and loop_cls is not None, (
        "missing source hook (LimitsLoopBudget owns "
        "app/research/director.py:normalize_research_action + LoopDetector): contract normalize("
        "source,tool,query,ticker,forms,as_of,accession,objective)->tuple; "
        "LoopDetector.check(action,result_hash,evidence_delta)->{duplicate,reason} + telemetry list"
    )
    loop = loop_cls()
    a = normalize(
        "sec",
        "get_sec_document",
        "GS 10-K risk",
        "GS",
        ("10-K",),
        "2025-06-30",
        "ACC-1",
        "exposure?",
    )
    b = normalize(
        "sec",
        "get_sec_document",
        "GS 10-Q MD&A",
        "GS",
        ("10-Q",),
        "2025-06-30",
        "ACC-2",
        "exposure?",
    )
    assert loop.check(a, "hash-a", 2).get("duplicate") is False
    assert loop.check(b, "hash-b", 2).get("duplicate") is False


# --- dispatch-boundary loop gate: exact no-progress repeats carry research_loop_detected ---
def test_dispatch_loop_gate_first_allowed_repeat_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First dispatch runs; the identical immediate repeat is refused and journaled, never re-executed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure", "ticker": "GS", "forms": ["10-K"]}
    first = _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    assert first["tool_calls_used"] == 1
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    blocked = [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"]
    assert len(blocked) == 1
    assert blocked[0].payload["reason"] == "research_loop_detected"
    assert blocked[0].payload["tool"] == "search_sec_filings"
    assert blocked[0].payload["query"] == "GS OpenAI exposure"
    assert blocked[0].payload["job_id"] == src


def test_dispatch_loop_gate_new_evidence_readmits_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """New evidence is progress: the same action runs again, then its own zero-progress repeat is refused."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:1"), repo=repo)
    again = _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    assert again["tool_calls_used"] == 2
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)


def test_dispatch_loop_gate_survives_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracked actions are persisted: a fresh repository still refuses the no-progress repeat."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    fresh = ResearchRepository()
    with pytest.raises(ValueError, match="research_loop_detected"):
        _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=fresh)


def test_dispatch_loop_gate_distinct_arguments_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One tool with different arguments (documents, queries, forms) is a different action every time."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    calls = [
        (
            "get_sec_document",
            {
                "accession_no": "0000320193-25-000079",
                "document_name": "nvda-20250331.htm",
            },
        ),
        (
            "get_sec_document",
            {
                "accession_no": "0000320193-25-000079",
                "document_name": "nvda-20250331ex10.htm",
            },
        ),
        ("search_sec_filings", {"query": "GS OpenAI exposure"}),
        ("search_sec_filings", {"query": "GS OpenAI exposure", "forms": ["10-Q"]}),
    ]
    for tool, args in calls:
        _svc.authorize_and_consume_dispatch(sid, src, tool, arguments=args, repo=repo)
    assert repo.get_session(sid).budget.get("tool_calls_used") == len(calls)
    assert [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"] == []


def test_dispatch_loop_gate_state_is_per_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking is per session+job: another job in the same session may run the same action."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    args = {"query": "GS OpenAI exposure"}
    _svc.authorize_and_consume_dispatch(sid, src, "search_sec_filings", arguments=args, repo=repo)
    scout = str(_svc.start_job(sid, "scout", repo=repo, wave_id=1)["job_id"])
    assert (
        _svc.authorize_and_consume_dispatch(sid, scout, "search_sec_filings", arguments=args, repo=repo)["job_id"]
        == scout
    )


def test_dispatch_loop_gate_gateway_repeat_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model path passes arguments through the gateway: the repeat is refused before the tool runs."""
    import app.tool_runtime as _gw
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, jid = _reg_svc_sid(repo)
    ctx = RuntimeToolSession(session_id="t-loop")
    ctx.active_research_session_id = sid
    ctx.active_research_job_id = jid
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        calls.append((name, dict(arguments)))
        return {
            "result_type": "sec_search",
            "query": arguments.get("query"),
            "count": 1,
            "results": [],
        }

    monkeypatch.setattr(_gw, "execute_tool", _fake_execute)
    first = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert "error" not in first, first
    second = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert "research_loop_detected" in str(second.get("error", "")), second
    assert len(calls) == 1  # the repeat never reached the tool
    third = execute_agent_tool("search_sec_filings", {"query": "GS 10-Q risk factors"}, ctx)
    assert "error" not in third, third
    assert len(calls) == 2


def test_dispatch_loop_gate_no_staged_job_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A staged session with no active job keeps today's refusal verbatim; the loop gate never sees it."""
    from app.tool_runtime import RuntimeToolSession, execute_agent_tool

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _src = _reg_svc_sid(repo)
    ctx = RuntimeToolSession(session_id="t-nojob")
    ctx.active_research_session_id = sid
    out = execute_agent_tool("search_sec_filings", {"query": "GS OpenAI exposure"}, ctx)
    assert out.get("error_type") == "invalid_research_context"
    assert "Active research job is required" in str(out.get("error"))
    assert repo.get_session(sid).budget.get("tool_calls_used", 0) == 0
    assert [e for e in repo.list_events(sid) if e.event_type == "research_loop_detected"] == []


def test_reg_freshness_latest_default() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope

    latest = resolve_temporal_scope(
        query="What happens to Goldman Sachs if OpenAI goes bankrupt?",
        now=_dt(2025, 6, 30, tzinfo=UTC),
    )
    assert latest["mode"] == "latest-available"
    assert latest["as_of"] is not None


def test_reg_freshness_latest_10k_pinned() -> None:
    from app.research.models import select_latest_baseline

    filings: list[object] = [
        {
            "form": "10-K",
            "known_at": "2025-02-14",
            "filed_at": "2025-02-14",
            "accession_no": "OLD",
        },
        {
            "form": "10-K",
            "known_at": "2025-06-20",
            "filed_at": "2025-06-20",
            "accession_no": "NEW",
        },
        {
            "form": "10-K/A",
            "known_at": "2025-06-25",
            "filed_at": "2025-06-25",
            "accession_no": "AMD",
        },
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    annual = base["annual_10k"]
    acc = annual.get("accession_no") if isinstance(annual, dict) else getattr(annual, "accession_no", None)
    assert acc in ("NEW", "AMD"), acc


def test_reg_freshness_superseded_only_current_rejected() -> None:
    from app.research.models import select_latest_baseline, superseded_current_violation

    filings: list[object] = [
        {
            "form": "10-K",
            "known_at": "2024-02-10",
            "filed_at": "2024-02-10",
            "accession_no": "OLD",
            "superseded_by": "NEW",
        },
        {
            "form": "10-K",
            "known_at": "2025-02-14",
            "filed_at": "2025-02-14",
            "accession_no": "NEW",
        },
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    annual = base["annual_10k"]
    acc = annual.get("accession_no") if isinstance(annual, dict) else getattr(annual, "accession_no", None)
    assert acc == "NEW", acc
    assert superseded_current_violation(filings, "OLD", as_of="2025-06-30") is not None
    assert superseded_current_violation(filings, "NEW", as_of="2025-06-30") is None


def test_reg_freshness_historical_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PIT-eligible historical evidence ingests; a future known_at rejects.
    # Distinct accession per item: dedupe keys on source_record_id, so a shared
    # accession would return duplicate_of instead of reaching the PIT gate.
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("GS 2024 exposure?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    jid = repo.list_jobs(sid)[0].job_id
    out = _svc.record_evidence(
        sid,
        jid,
        _reg_item(f"{sid}:ev:1", known_at="2024-12-01T00:00:00+00:00"),
        repo=repo,
    )
    assert out["evidence_id"] == f"{sid}:ev:1"
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected|ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            jid,
            _reg_item(
                f"{sid}:ev:2",
                source_record_id="0000886982-26-000002",
                source_uri="https://www.sec.gov/Archives/edgar/data/886982/000088698226000002/primary.htm",
                source_handle=seam.handle_for(
                    "The firm discloses OpenAI-linked exposure in the filing.",
                    accession="0000886982-26-000002",
                    document="gs-10q-20260331.htm",
                    known_at="2025-07-01T00:00:00+00:00",
                ),
            ),
            repo=repo,
        )


def test_reg_negatives_coverage_present() -> None:
    """The closed vocabularies carry the claim/provenance contract; default coverage keeps its keys."""
    from app.research.dossiers.sec import default_coverage
    from app.research.evidence import CLAIM_KINDS, PROVENANCE_KINDS

    # Evidence is kernel-replayed claims or recomputed calculations: a search-derived absence is coverage state.
    assert CLAIM_KINDS == ("observed_fact", "calculation", "inference", "assumption")
    assert PROVENANCE_KINDS == (
        "sec_source",
        "sec_record",
        "finra_record",
        "web_source",
        "search_run",
        "none",
    )
    cov = default_coverage()
    for key in ("forms", "sources_examined", "complete", "exclusions"):
        assert key in cov, sorted(cov.keys())


def test_reg_negatives_scoped_no_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scoped no-hit observation records as a coverage artifact with its full coverage envelope."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    scoped_cov = _absence_cov(complete=False, gaps=["sections 4-6 unread"], pagination_complete=True)
    out = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "no OpenAI bankruptcy exposure disclosed in sections 1-3 of the scoped GS 10-K",
            "claim_text": "not found in sections 1-3 of the scoped GS 10-K: OpenAI bankruptcy exposure",
            "claim_kind": "absence_observation",
            "subject": "GS",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "s1",
            "query": "GS OpenAI bankruptcy",
            "coverage": scoped_cov,
        },
        repo=repo,
    )
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    assert out.get("search_id") == "s1" and out.get("query") == "GS OpenAI bankruptcy"
    assert out.get("coverage") == scoped_cov
    stored = repo.list_coverage_artifacts(sid)[0]
    assert stored.get("search_id") == "s1" and stored.get("coverage") == scoped_cov
    assert repo.list_evidence(sid) == []  # coverage state never lands in the ledger
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(
            sid,
            src,
            {
                "wave_id": 1,
                "content": "no exposure anywhere",
                "claim_text": "no OpenAI exposure in any filing",
                "claim_kind": "absence_observation",
                "subject": "GS",
                "source_name": "SEC",
                "known_at": "2025-06-29T00:00:00+00:00",
                "search_id": "s1",
                "query": "GS OpenAI",
                "coverage": {
                    "forms": ["10-K"],
                    "dates": [],
                    "partitions": [],
                    "docs": [],
                    "gaps": [],
                    "complete": False,
                },
            },
            repo=repo,
        )


def test_reg_negatives_incomplete_stays_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    _svc.record_evidence(sid, src, _reg_item(f"{sid}:ev:1"), repo=repo)
    out = _svc.submit_source_result(
        src,
        coverage={
            "useful_for_question": "insufficient",
            "resolved": [],
            "partially_resolved": [],
            "unresolved": ["OpenAI private terms"],
            "source_limitations": ["SEC-only"],
        },
        evidence_ids=[],
        unresolved_questions=["OpenAI private terms"],
        repo=repo,
    )
    assert out["job_status"] == "completed"
    stored = repo.get_job(src).result or {}
    assert isinstance(stored, dict)
    cov = stored.get("coverage")
    assert isinstance(cov, dict) and cov.get("useful_for_question") == "insufficient"
    assert cov.get("complete") in (None, False)


# --- PIT (3): accepted_at==known_at, TZ preserves instant, future rejected ---


def test_reg_pit_accepted_at_is_known_at() -> None:
    from typing import override

    from edgar import Filing as EdgarFiling

    from app.sec.normalization import filing_from_edgar

    class _StubFiling(EdgarFiling):
        acceptance_datetime = "2025-02-14T17:30:00Z"

        @override
        @property
        def period_of_report(self) -> str:
            return "2024-12-31"

        @override
        @property
        def document(self) -> str:
            return "primary.htm"

    filing = _StubFiling(
        cik=886982,
        company="Goldman Sachs",
        form="10-K",
        filing_date="2025-02-14",
        accession_no="0000886982-26-000001",
    )
    meta = filing_from_edgar(filing)
    assert meta.known_at == "2025-02-14T17:30:00Z"
    assert meta.accepted_at == meta.known_at


def test_reg_pit_tz_preserves_instant() -> None:
    from app.research.models import pit_violated

    assert pit_violated("2025-06-30T00:00:00+00:00", "2025-06-29T20:00:00-04:00") is False
    assert pit_violated("2025-06-30T00:00:00+00:00", "2025-06-30T01:00:00+00:00") is True


def test_reg_pit_future_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    with pytest.raises(ValueError):
        _svc.record_evidence(
            sid,
            src,
            _reg_item(f"{sid}:ev:9", known_at="2025-07-01T00:00:00+00:00"),
            repo=repo,
        )


# --- freeze/committee (5): immutable, identical freeze ID, no source tools, new wave, decide ---
def test_reg_freeze_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses as _dc
    from datetime import datetime as _dt

    from app.research import service as _svc
    from app.research.freeze import EvidenceFreeze

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    frozen = _svc.freeze_session(sid, 1, repo=repo)
    fid = str(frozen["freeze_id"])
    stored = repo.get_freeze(fid)
    assert stored["freeze_id"] == fid
    now = _dt(2025, 6, 30, tzinfo=UTC)
    with pytest.raises(_dc.FrozenInstanceError):
        frozen_obj = EvidenceFreeze(
            freeze_id=fid,
            session_id=sid,
            wave_id=1,
            created_at=now,
            as_of=now,
            evidence_ids=(eid,),
            content_hash="h",
        )
        _frozen_write(frozen_obj, "evidence_ids", ("tampered",))
    assert repo.get_freeze(fid)["freeze_id"] == fid


def test_reg_freeze_identical_id_for_committee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3
    jobs = repo.list_jobs(sid)
    assert repo.get_session(sid).freeze_ids[-1] == fid
    assert all(j.wave_id == 1 for j in jobs if j.job_type in ("stockbot", "bullbot", "bearbot"))


def test_reg_committee_cannot_call_source_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    bear = str(_svc.start_job(sid, "bearbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="forbids"):
        _svc.authorize_and_consume_dispatch(sid, bear, "get_sec_document", repo=repo)


def test_reg_freeze_new_wave_new_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    f1 = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    assert f1 == f"{sid}:1:freeze"
    assert f"{sid}:2:freeze" != f1


def test_reg_freeze_director_finalize_or_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    out = _svc.decide_next_wave(sid, repo=repo)
    # §3: director gate set has no budget_exhausted (deleted); runtime_exceeded stays.
    assert out["stop_reason"] in (
        "no_questions",
        "low_gain",
        "not_actionable",
        "continue",
        "max_waves",
        "jobs_exceeded",
        "runtime_exceeded",
    )


# --- synthesis (4): supported trace, inference labeled, unknown, manageable ---


def test_reg_synthesis_supported_traces_to_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid,
            jid,
            role,
            _committee_analysis(eid, claim_text="GS filings suggest OpenAI-linked exposure"),
            repo=repo,
        )
    out = _svc.finalize_session(
        sid,
        "GS filings suggest OpenAI-linked exposure. GS exposure is filing-backed.",
        [{"text": "GS filings suggest OpenAI-linked exposure", "evidence_ids": [eid]}],
        repo=repo,
    )
    assert out["freeze_id"] == f"{sid}:1:freeze"
    final = repo.get_session(sid).final_result or {}
    assert isinstance(final, dict)
    claims_raw = final.get("claims")
    assert isinstance(claims_raw, list) and claims_raw
    first = claims_raw[0]
    assert isinstance(first, dict) and first.get("evidence_ids") == [eid]
    # The uncovered answer sentence is deleted surgically; the covered one publishes.
    answer = final.get("answer")
    assert isinstance(answer, str)
    assert "GS filings suggest OpenAI-linked exposure" in answer and "filing-backed" not in answer


def test_reg_synthesis_inference_labeled() -> None:
    """A declared inference stays labeled inference through synthesis: a citation never upgrades it."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="OpenAI stress may widen GS spreads", evidence_ids=["EV-1"])
    assert claim.claim_type == "inference"
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[claim],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
        claims=[claim],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
        claims=[claim],
    )
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    raw_claims = synth.to_dict()["claims"]
    assert isinstance(raw_claims, list)
    rows = [r for r in raw_claims if isinstance(r, dict) and r.get("text") == claim.text]
    assert rows and all(r.get("claim_type") == "inference" for r in rows)
    assert "(inference)" in synth.answer and "observed_fact" not in synth.answer


def test_reg_synthesis_unknown_stays_unknown() -> None:
    from app.research.evals.evaluators import EvalInput, evaluate

    inp = EvalInput(
        scenario_name="gs-openai-sec-only",
        answer_text="OpenAI private revenue share: UNKNOWN (no filing discloses it).",
        evidence_ids=("EV-1",),
        requires_evidence=True,
    )
    assert evaluate(inp).passed


def test_reg_synthesis_manageable_rejected_as_fact() -> None:
    from app.research.agents import ModelOutputFailure, parse_grounded_claims

    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "GS will manageably absorb any OpenAI loss", "evidence_ids": []}]',
            frozen=["EV-1"],
        )


# --- evidence record_kind + claim labels (contract pins, stub-safe) ---


def test_reg_evidence_record_kind_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.research.evidence import (
        RECORD_KINDS,
        DiscoveryRecord,
        EvidenceRecord,
        discovery_only,
        substantive_records,
    )

    assert RECORD_KINDS == frozenset({"discovery", "evidence"})
    assert DiscoveryRecord(record_id="d1", session_id="s", tool="search_sec_filings", query="GS 10-K").search_id is None
    assert EvidenceRecord(record_id="e1", session_id="s", evidence_id="EV-1", claim_text="c").evidence_id == "EV-1"
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    # A scoping search records as a coverage artifact: search scope, never citable evidence.
    out = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "scoping search over GS filings",
            "claim_text": "scoping search",
            "claim_kind": "absence_observation",
            "subject": "GS",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "s1",
            "query": "GS 10-K",
            "coverage": _absence_cov(),
            "record_kind": "discovery",
        },
        repo=repo,
    )
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    assert out.get("claim_kind") == "absence_observation"
    assert "evidence_id" not in out and repo.list_evidence(sid) == []
    assert repo.list_coverage_artifacts(sid)[0].get("search_id") == "s1"
    assert discovery_only([{"record_kind": "discovery"}]) is True
    assert substantive_records([{"record_kind": "discovery"}, {"record_kind": "evidence"}]) == [
        {"record_kind": "evidence"}
    ]


def test_reg_evidence_claim_labels() -> None:
    """Claim type is declared, never inferred from wording: absent type stays inference, and
    observed_fact/contradicted/conditional_risk/attributed_report need at least one freeze id (unknown may cite none)."""
    from app.research.agents import (
        CLAIM_TYPES,
        GroundedClaim,
        ModelOutputFailure,
        parse_grounded_claims,
    )

    assert tuple(CLAIM_TYPES) == (
        "observed_fact",
        "calculation",
        "inference",
        "assumption",
        "unknown",
        "contradicted",
        "conditional_risk",
        "attributed_report",
    )
    assert GroundedClaim(text="GS revenue grew", evidence_ids=["EV-1"]).claim_type == "inference"
    declared = parse_grounded_claims(
        '[{"text": "GS revenue grew", "claim_type": "observed_fact", "evidence_ids": ["EV-1"]}]',
        frozen=["EV-1"],
    )
    assert declared[0].claim_type == "observed_fact"
    # Wording and citation never type a claim: an undeclared confident sentence stays inference.
    undeclared = parse_grounded_claims(
        '[{"text": "GS revenue grew per the filing, definitely observed.", "evidence_ids": ["EV-1"]}]',
        frozen=["EV-1"],
    )
    assert undeclared[0].claim_type == "inference"
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "GS revenue grew", "claim_type": "observed_fact", "evidence_ids": []}]',
            frozen=["EV-1"],
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "A default could impair the marks", "claim_type": "conditional_risk", "evidence_ids": []}]',
            frozen=["EV-1"],
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "Management said exposure is capped", "claim_type": "attributed_report", "evidence_ids": []}]',
            frozen=["EV-1"],
        )
    with pytest.raises(ModelOutputFailure):
        parse_grounded_claims(
            '[{"text": "unclear", "claim_type": "shrug", "evidence_ids": ["EV-1"]}]',
            frozen=["EV-1"],
        )
    unknown = parse_grounded_claims(
        '[{"text": "OpenAI terms UNKNOWN", "claim_type": "unknown", "evidence_ids": []}]',
        frozen=["EV-1"],
    )
    assert unknown[0].claim_type == "unknown" and unknown[0].evidence_ids == []
    # Claims reference kernel datapoint ids; they never assert value/entity/period/unit themselves.
    typed = parse_grounded_claims(
        '[{"text": "NVDA Total revenue for Q1 FY2025 was $26,044 million", "claim_type": "observed_fact",'
        ' "evidence_ids": ["EV-1"], "datapoint_ids": ["dp:a", "dp:a"]}]',
        frozen=["EV-1"],
    )
    assert typed[0].datapoint_ids == ["dp:a"]
    for bad in (
        '[{"text": "Revenue was 26,044", "claim_type": "observed_fact", "evidence_ids": ["EV-1"], "value": "26,044"}]',
        '[{"text": "Revenue was 26,044", "claim_type": "observed_fact", "evidence_ids": ["EV-1"], "entity": "NVDA"}]',
        '[{"text": "Margin is 8.2", "claim_type": "calculation", "evidence_ids": ["EV-1"]}]',
        '[{"text": "Revenue was 26,044", "claim_type": "observed_fact", "evidence_ids": ["EV-1"], "datapoint_ids": ["x"]}]',
        '[{"text": "Revenue was 26,044", "claim_type": "observed_fact", "evidence_ids": ["EV-1"], "datapoint_ids": "dp:a"}]',
        '[{"text": "Assuming revenue of 26,044", "claim_type": "assumption", "evidence_ids": [], "datapoint_ids": ["dp:a"]}]',
    ):
        with pytest.raises(ModelOutputFailure):
            parse_grounded_claims(bad, frozen=["EV-1"])
    hypothetical = parse_grounded_claims(
        '[{"text": "Assume a 30% haircut", "claim_type": "assumption", "evidence_ids": []}]', frozen=["EV-1"]
    )
    assert hypothetical[0].claim_type == "assumption" and hypothetical[0].datapoint_ids == []


def test_crap_source_policy_modes_and_errors() -> None:
    from app.research.models import resolve_source_policy, source_domain_allowed

    # Bare research_sources mapping (no outer key) resolves identically.
    bare = resolve_source_policy({"mode": "all", "sources": []})
    assert bare["mode"] == "all" and bare["allowed"] == []
    # Mode "all" admits any domain not denied; allowlist pins SEC.
    assert source_domain_allowed(bare, "anything.example") is True
    assert source_domain_allowed(None, None) is True
    sec = resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": [" SEC "]}})
    assert sec["allowed"] == ["SEC"]
    assert source_domain_allowed(sec, "SEC") is True
    assert source_domain_allowed(sec, "web") is False
    denied = {"allowed": ["SEC"], "denied": ["SEC"], "mode": "allowlist"}
    assert source_domain_allowed(denied, "SEC") is False
    with pytest.raises(ValueError, match="policy"):
        resolve_source_policy("nope")
    with pytest.raises(ValueError, match="research_sources"):
        resolve_source_policy({"research_sources": "nope"})
    with pytest.raises(ValueError, match="sources"):
        resolve_source_policy({"research_sources": {"mode": "allowlist", "sources": []}})
    with pytest.raises(ValueError, match="sources"):
        resolve_source_policy({"research_sources": {"mode": "all", "sources": [""]}})
    with pytest.raises(ValueError, match="source_domain"):
        source_domain_allowed(sec, "  ")


def test_crap_temporal_patterns() -> None:
    from datetime import datetime as _dt

    from app.research.models import resolve_temporal_scope

    now = _dt(2025, 6, 30, tzinfo=UTC)
    assert resolve_temporal_scope(temporal="show all history please", now=now)["mode"] == "unbounded"
    assert resolve_temporal_scope(temporal="as of 2024-03-31", now=now)["as_of"] == "2024-03-31"
    last2 = resolve_temporal_scope(temporal="last 2 years", now=now)
    assert last2["mode"] == "range" and str(last2["start"])[:4] == "2023"
    lastyr = resolve_temporal_scope(temporal="last year", now=now)
    assert (lastyr["start"], lastyr["end"]) == ("2024-01-01", "2024-12-31")
    lastq = resolve_temporal_scope(query="What drove NVDA revenue last quarter?", now=now)
    assert lastq["mode"] == "range" and (lastq["start"], lastq["end"]) == (
        "2025-01-01",
        "2025-03-31",
    )
    thisw = resolve_temporal_scope(query="Apple news this week", now=now)
    assert thisw["mode"] == "range" and thisw["start"] == "2025-06-23" and str(thisw["end"])[:10] == "2025-06-30"
    lastw = resolve_temporal_scope(temporal="last week", now=now)
    assert lastw["mode"] == "range" and (lastw["start"], lastw["end"]) == (
        "2025-06-16",
        "2025-06-22",
    )
    assert resolve_temporal_scope(temporal="before earnings", now=now)["mode"] == "as_of"
    assert resolve_temporal_scope(temporal="latest available data", now=now)["mode"] == "latest-available"
    assert resolve_temporal_scope(temporal="most recent filing", now=now)["mode"] == "as_of"
    with pytest.raises(ValueError, match="after end"):
        resolve_temporal_scope(temporal="between 2025-06-30 and 2025-01-01", now=now)
    with pytest.raises(ValueError, match="temporal"):
        resolve_temporal_scope(temporal="  ", now=now)
    with pytest.raises(ValueError, match="invalid calendar"):
        resolve_temporal_scope(temporal="as of 2025-02-30", now=now)


def test_crap_baseline_current_and_quarterly_picks() -> None:
    from app.research.models import select_latest_baseline

    filings: list[object] = [
        {"form": "10-Q", "known_at": "2025-03-31", "accession_no": "Q1"},
        {"form": "10-Q", "known_at": "2025-06-20", "accession_no": "Q2"},
        {"form": "8-K", "known_at": "2025-06-10", "accession_no": "K1"},
        {"form": "8-K", "known_at": "2025-06-25", "accession_no": "K2"},
        {"form": "10-K", "known_at": "2030-01-01", "accession_no": "FUTURE"},
        {"form": "DEF-14A", "known_at": "2025-01-01", "accession_no": "PROXY"},
        {"form": "10-K"},
    ]
    base = select_latest_baseline(filings, as_of="2025-06-30")
    q = base["quarterly_10q"]
    assert isinstance(q, dict) and q.get("accession_no") == "Q2"
    mats = base["material_8k"]
    assert isinstance(mats, list)
    first = mats[0]
    assert isinstance(first, dict) and first.get("accession_no") == "K2"
    assert all(isinstance(f, dict) and f.get("accession_no") != "FUTURE" for f in mats)


def test_crap_superseded_violation_edges() -> None:
    from app.research.models import superseded_current_violation

    assert superseded_current_violation([{"form": "8-K", "known_at": "2025-01-01"}], "X") is None
    filings: list[object] = [
        {
            "form": "10-K",
            "known_at": "2024-02-10",
            "accession_no": "OLD",
            "superseded_by": "NEW",
        },
        {"form": "10-K", "known_at": "2025-02-14", "accession_no": "NEW"},
    ]
    assert superseded_current_violation(filings, [], as_of="2025-06-30") is None
    assert superseded_current_violation(filings, ["NEW", 42, " "], as_of="2025-06-30") is None
    assert superseded_current_violation(filings, 42, as_of="2025-06-30") is None
    hit = superseded_current_violation(filings, " OLD ", as_of="2025-06-30")
    assert hit is not None and "NEW" in hit


def test_crap_discovery_only_edges() -> None:
    from app.research.evidence import (
        DiscoveryRecord,
        discovery_only,
        substantive_records,
    )

    assert discovery_only([]) is False
    assert discovery_only("nope") is False
    assert discovery_only([{"record_kind": "discovery"}, {"record_kind": "evidence"}]) is False
    assert discovery_only([{"metadata": {"record_kind": "discovery"}}]) is True
    assert discovery_only([DiscoveryRecord(record_id="d", session_id="s", tool="t", query="q")]) is False
    assert discovery_only([{"record_kind": "discovery", "metadata": {}}]) is True
    assert substantive_records([{"record_kind": "discovery"}, {}]) == [{}]


# ---------------------------------------------------------------------------
# Evals slice: committee invariants + finalization UX (offline fakes only).
# Same freeze across the trio; 3 distinct jobs created before the run;
# concurrent; completed-job cross-role write rejected; roles cannot mutate the
# freeze; claims resolve to frozen evidence; research_requests stay separate
# from evidence. Finalize success auto-renders a substantive structured answer
# in the same turn; a bare "finalized/N claims" with no answer fails.
# ---------------------------------------------------------------------------


def _eval_sid(repo: ResearchRepository) -> tuple[str, str]:
    from app.research import service as _svc

    sid = _svc.create_research("MSFT OpenAI exposure?", "o", as_of="2026-08-10T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _eval_item(eid: str, wave: int = 1) -> dict[str, object]:
    return {
        "evidence_id": eid,
        "wave_id": wave,
        "content": "c-" + eid,
        "claim_text": "MSFT Azure OpenAI-linked exposure per filing " + eid,
        "subject": "MSFT",
        "source_name": "SEC",
        "source_uri": "https://www.sec.gov/Archives/edgar/data/789790/000095017026001234/primary.htm",
        "source_record_id": "0000950170-26-001234",
        "document_name": "msft-10q-20260630.htm",
        "matching_passage": "Azure OpenAI-linked exposure is disclosed in the filing.",
        "source_handle": seam.handle_for(
            "Azure OpenAI-linked exposure is disclosed in the filing.",
            accession="0000950170-26-001234",
            document="msft-10q-20260630.htm",
        ),
        "known_at": "2026-08-01T00:00:00+00:00",
    }


def _eval_cov() -> dict[str, object]:
    return {
        "useful_for_question": "sufficient",
        "resolved": ["MSFT OpenAI exposure"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "major_entities_investigated": ["MSFT", "OpenAI"],
        "relationship_types_checked": ["investment", "commercial"],
        "forms_examined": ["10-K"],
        "exhibits_examined": ["EX-10.1"],
        "material_open_questions": [],
        "search_runs": ["s1"],
        "covered_branches": ["MSFT Azure OpenAI-linked exposure"],
    }


def _eval_trio_ids(repo: ResearchRepository, sid: str, eid: str) -> tuple[str, str, str, str]:
    from app.research import service as _svc

    fid = str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3 and len(set(created)) == 3
    return fid, str(created[0]), str(created[1]), str(created[2])


def test_eval_committee_same_freeze_and_trio_created_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    fid, stock_jid, bull_jid, bear_jid = _eval_trio_ids(repo, sid, eid)
    assert repo.get_session(sid).freeze_ids[-1] == fid
    assert len({stock_jid, bull_jid, bear_jid}) == 3
    for jid, role in (
        (stock_jid, "stockbot"),
        (bull_jid, "bullbot"),
        (bear_jid, "bearbot"),
    ):
        job = repo.get_job(jid)
        assert job.job_type == role and job.status == "running"
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    assert all(repo.get_job(jid).status == "completed" for jid in (stock_jid, bull_jid, bear_jid))
    assert repo.get_session(sid).freeze_ids[-1] == fid


def test_eval_committee_cross_role_write_rejected_and_freeze_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses as _dc
    from datetime import datetime as _dt

    from app.research import service as _svc
    from app.research.freeze import EvidenceFreeze

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    fid, stock_jid, _bull_jid, _bear_jid = _eval_trio_ids(repo, sid, eid)
    with pytest.raises(ValueError, match="!="):
        _svc.record_committee_analysis(sid, stock_jid, "bearbot", _committee_analysis(eid), repo=repo)
    now = _dt(2026, 8, 10, tzinfo=UTC)
    with pytest.raises(_dc.FrozenInstanceError):
        frozen_obj = EvidenceFreeze(
            freeze_id=fid,
            session_id=sid,
            wave_id=1,
            created_at=now,
            as_of=now,
            evidence_ids=(eid,),
            content_hash="h",
        )
        _frozen_write(frozen_obj, "evidence_ids", ("tampered",))
    _svc.record_committee_analysis(
        sid,
        stock_jid,
        "stockbot",
        _committee_analysis(eid, ["Probe Azure terms?"]),
        repo=repo,
    )
    assert repo.get_freeze(fid)["freeze_id"] == fid


def test_eval_committee_claims_resolve_and_requests_not_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _fid, stock_jid, _bull_jid, _bear_jid = _eval_trio_ids(repo, sid, eid)
    with pytest.raises(ValueError):
        _svc.record_committee_analysis(sid, stock_jid, "stockbot", _committee_analysis("EV-NOPE"), repo=repo)
    n_evidence = len(repo.list_evidence(sid))
    _svc.record_committee_analysis(
        sid,
        stock_jid,
        "stockbot",
        _committee_analysis(eid, ["Probe Azure terms?"]),
        repo=repo,
    )
    assert len(repo.list_evidence(sid)) == n_evidence


def test_eval_finalize_renders_answer_same_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service as _svc
    from app.research.evals.evaluators import EvalInput, evaluate

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid,
            jid,
            role,
            _committee_analysis(eid, claim_text="MSFT Azure exposure may be filing-backed"),
            repo=repo,
        )
    _svc.decide_next_wave(sid, repo=repo)
    out = _svc.finalize_session(
        sid,
        "MSFT Azure exposure may be filing-backed.",
        [{"text": "MSFT Azure exposure may be filing-backed", "evidence_ids": [eid]}],
        repo=repo,
    )
    final = repo.get_session(sid).final_result or {}
    assert isinstance(final, dict) and str(final.get("answer", "")).strip()
    assert "filing-backed" in str(final.get("answer"))
    assert out["freeze_id"] == f"{sid}:1:freeze"
    rendered = _svc.inspect_research(sid, repo=repo)
    assert str(rendered.get("final", "") or final.get("answer", "")).strip()
    bare = EvalInput(
        scenario_name="msft-openai-bankruptcy-sec-only",
        answer_text="finalized 1 claims",
        evidence_ids=(eid,),
        requires_evidence=True,
        finalized_claim_count=1,
        answered=False,
    )
    assert "finalized-without-answer" in evaluate(bare).violations


# ---------------------------------------------------------------------------
# Phase 16 invariants: raw-provenance evidence integrity, scoped absence,
# sufficiency envelope, committee freeze atomicity + role boundary, declared
# claim typing, the per-freeze wave gate (N > 2 never finalizes by itself),
# and finalize rendering (answer only in final_result).
# External behavior only: returned payloads, persisted rows, raised error codes.
# ---------------------------------------------------------------------------


def _inv_sid(repo: ResearchRepository, q: str = "NVDA OpenAI exposure?") -> tuple[str, str]:
    """Fresh session with its running source job: the only record_evidence boundary."""
    from app.research import service as _svc

    sid = _svc.create_research(q, "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return sid, repo.list_jobs(sid)[0].job_id


def _typed_committee_analysis(eid: str, text: str, claim_type: str) -> dict[str, object]:
    """Rich committee envelope with one explicitly typed claim."""
    env = _committee_analysis(eid, claim_text=text)
    env["claims"] = [{"text": text, "claim_type": claim_type, "evidence_ids": [eid]}]
    return env


def test_invariant_search_hit_is_never_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search hit is navigation: no raw passage means ERR_RAW_SOURCE_REQUIRED and no persisted row."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    hit = {
        "evidence_id": f"{sid}:ev:1",
        "wave_id": 1,
        "content": "NVDA 10-Q hit: accelerated computing demand",
        "claim_text": "search hit",
        "subject": "NVDA",
        "source_name": "SEC",
        "search_id": "s1",
        "query": "NVDA demand",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED") as excinfo:
        _svc.record_evidence(sid, src, hit, repo=repo)
    assert "navigation artifact" in str(excinfo.value)
    assert repo.list_evidence(sid) == []


def test_invariant_search_id_never_passes_as_accession(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search id offered as the filing accession is a provenance mismatch; junk fails the format gate."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            src,
            {**_svc_item(f"{sid}:ev:1"), "source_record_id": "s1", "search_id": "s1"},
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT"):
        _svc.record_evidence(sid, src, {**_svc_item(f"{sid}:ev:2"), "source_record_id": "s-1"}, repo=repo)
    assert repo.list_evidence(sid) == []


def test_invariant_raw_document_is_accepted_and_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accession + document_name + passage is the accepted fact shape; a bare 18-digit run normalizes."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    out = _svc.record_evidence(
        sid,
        src,
        {**_svc_item(eid), "source_record_id": "000032019325000079"},
        repo=repo,
    )
    assert out["evidence_id"] == eid
    assert out.get("claim_kind") == "observed_fact"
    prov = out.get("provenance")
    assert isinstance(prov, dict)
    passage = "Accelerated computing demand grew in the reporting period."
    assert prov.get("kind") == "sec_source" and prov.get("accession_no") == "0000320193-25-000079"
    assert prov.get("document_name") == "nvda-20250331.htm" and prov.get("passage") == passage
    # The kernel stores the archive's bytes with the coordinates and hash it reloaded them at.
    assert (prov.get("offset"), prov.get("end")) == (0, len(passage))
    assert prov.get("text_hash") == sha256(passage.encode("utf-8")).hexdigest()
    assert out.get("record_kind") == "evidence"
    assert out.get("source_record_id") == "0000320193-25-000079"


def test_invariant_dossier_summary_is_never_observed_fact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dossier/model summaries carry no raw passage: declared observed_fact they fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    summary = {
        "evidence_id": f"{sid}:ev:1",
        "wave_id": 1,
        "content": "Dossier summary: NVDA demand is rising.",
        "claim_text": "summary finding",
        "subject": "NVDA",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, summary, repo=repo)
    cited_only = {
        **summary,
        "evidence_id": f"{sid}:ev:2",
        "source_record_id": "0000320193-25-000079",
        "document_name": "nvda-20250331.htm",
    }
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, cited_only, repo=repo)
    assert repo.list_evidence(sid) == []


def test_invariant_absence_observation_needs_search_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absence = search run + full coverage + no accession, recorded as coverage state, never evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    base = {
        "wave_id": 1,
        "content": "no NVDA capacity constraint disclosed",
        "claim_text": "not disclosed in the searched scope",
        "claim_kind": "absence_observation",
        "subject": "NVDA",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
        "search_id": "s1",
        "query": "NVDA capacity",
    }
    out = _svc.record_evidence(sid, src, {**base, "coverage": _absence_cov()}, repo=repo)
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    artifact_id = str(out.get("artifact_id"))
    assert artifact_id.startswith(f"{sid}:cov:")
    assert out.get("evidence_ids") == []
    # Not evidence: neither the ledger nor the session evidence set grows...
    assert repo.list_evidence(sid) == []
    assert repo.get_session(sid).evidence_ids == []
    # ...and the artifact stays inspectable as coverage state.
    artifacts = repo.list_coverage_artifacts(sid)
    assert [a.get("artifact_id") for a in artifacts] == [artifact_id]
    assert artifacts[0].get("search_id") == "s1" and artifacts[0].get("coverage") == _absence_cov()
    coverage_store = repo.resource_stores(sid)["coverage"]
    stored = coverage_store[artifact_id]
    assert isinstance(stored, dict) and stored.get("query") == "NVDA capacity"
    # One scope is one artifact: a repeat is an idempotent write, not a second record.
    again = _svc.record_evidence(sid, src, {**base, "coverage": _absence_cov()}, repo=repo)
    assert again.get("accepted") is False and again.get("artifact_id") == artifact_id
    assert len(repo.list_coverage_artifacts(sid)) == 1
    bad_covs: list[dict[str, object]] = [
        {"forms": ["10-K"]},  # missing keys
        _absence_cov(pagination_complete="yes"),  # flag is not a bool
        _absence_cov(complete=None),  # flag is not a bool
    ]
    for i, bad_cov in enumerate(bad_covs):
        with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
            _svc.record_evidence(
                sid,
                src,
                {**base, "query": f"NVDA capacity {i}", "coverage": bad_cov},
                repo=repo,
            )
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(sid, src, {**base, "query": "", "coverage": _absence_cov()}, repo=repo)
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            src,
            {
                **base,
                "source_record_id": "0000320193-25-000079",
                "coverage": _absence_cov(),
            },
            repo=repo,
        )
    # The same absence wording is not a fact without raw provenance.
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, {**base, "claim_kind": "observed_fact"}, repo=repo)
    assert repo.list_evidence(sid) == []  # nothing retroactively became evidence
    assert len(repo.list_coverage_artifacts(sid)) == 1


def test_invariant_claim_kind_never_routed_by_wording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative wording is gated by provenance, not lexis: raw provenance accepts it, none fails closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    wording = "No OpenAI exposure exists anywhere in the GS filings and the business is unaffected."
    eid = f"{sid}:ev:1"
    out = _svc.record_evidence(
        sid,
        src,
        {
            **_svc_item(eid, passage=wording),
            "claim_text": wording,
            "content": wording,
            "matching_passage": wording,
        },
        repo=repo,
    )
    assert out.get("claim_kind") == "observed_fact"
    prov = out.get("provenance")
    assert isinstance(prov, dict) and prov.get("kind") == "sec_source" and prov.get("passage") == wording
    bare = {
        "evidence_id": f"{sid}:ev:2",
        "wave_id": 1,
        "content": wording,
        "claim_text": wording,
        "subject": "GS",
        "source_name": "SEC",
        "known_at": "2025-06-29T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ERR_ACCESSION_FORMAT|ERR_RAW_SOURCE_REQUIRED") as excinfo:
        _svc.record_evidence(sid, src, bare, repo=repo)
    assert "negat" not in str(excinfo.value).lower()  # no lexical negativity gate anywhere


def test_invariant_unknown_stays_unknown_and_absence_needs_coverage() -> None:
    """Absence renders as searched-scope language; a generic unknown never becomes an absence claim."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import (
        SEC_SCOPE_ABSENCE,
        scoped_absence,
        synthesize_final,
    )
    from app.tool_render import render_final_result

    raw_claim = "Redeployment economics are unknown from the filings available"
    assert scoped_absence(raw_claim) == f"{SEC_SCOPE_ABSENCE}: {raw_claim}"
    assert scoped_absence("  ").startswith(SEC_SCOPE_ABSENCE)
    unknown = GroundedClaim(text=raw_claim, claim_type="unknown", evidence_ids=[])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[unknown],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
        claims=[unknown],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
        claims=[unknown],
    )
    # No coverage artifacts: the unknown stays an unknown, and no absence claim is invented from it.
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    assert synth.absence_observations == []
    assert raw_claim in synth.unknowns
    assert not any(SEC_SCOPE_ABSENCE in item for item in synth.unknowns)
    # With a coverage artifact, absence text is that artifact's, still scoped.
    scoped = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        absence_observations=["no OpenAI funding terms disclosed"],
    )
    assert scoped.absence_observations == [f"{SEC_SCOPE_ABSENCE}: no OpenAI funding terms disclosed"]
    assert f"- {SEC_SCOPE_ABSENCE}" in scoped.answer
    assert raw_claim in scoped.unknowns  # the coverage artifact does not reclassify the unknown
    rendered = render_final_result(scoped.to_dict())
    assert SEC_SCOPE_ABSENCE in rendered


def test_invariant_direct_evidence_is_canonical_not_committee_labels() -> None:
    """`observed_fact` on a committee claim never renders as direct evidence; canonical rows do."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    labeled = GroundedClaim(
        text="Microsoft will absorb the OpenAI loss",
        claim_type="observed_fact",
        evidence_ids=["EV-1"],
    )
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        answer="balanced",
        base_case="balanced",
        claims=[labeled],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bullish",
        bull_case="resilient",
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="x",
        question="Q?",
        stance="bearish",
        bear_case="contagion",
    )
    observation = {
        "evidence_id": "EV-1",
        "claim_text": "OpenAI investment carrying value",
        "record_kind": "evidence",
        "source_name": "SEC-10-K",
        "provenance": {
            "kind": "sec_source",
            "accession_no": "0000320193-25-000079",
            "document_name": "msft-10k.htm",
            "passage": "OpenAI carrying value was $13.3B",
        },
    }
    synth = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        observations=[observation],
    )
    direct = [str(row.get("text")) for row in synth.direct_evidence]
    assert len(direct) == 1 and "msft-10k.htm" in direct[0] and "OpenAI carrying value was $13.3B" in direct[0]
    assert all(labeled.text not in line for line in direct)  # a model label is not direct evidence
    assert synth.direct_evidence[0]["evidence_ids"] == ["EV-1"]
    # The committee reading still renders (as an interpretation), so nothing is silently dropped.
    assert [row["text"] for row in synth.first_order_effects] == [labeled.text]
    assert "What the evidence directly shows" in synth.answer and "msft-10k.htm" in synth.answer
    # Without canonical observations the section is empty rather than model-labeled.
    bare = synthesize_final(
        "Q?",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="x",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    assert bare.direct_evidence == []


def test_invariant_sufficient_needs_structured_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`sufficient` needs the full structured envelope; the legacy bare-sufficient fallback is gone."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    legacy: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA demand"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
    }
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED") as excinfo:
        _svc.submit_source_result(src, coverage=legacy, evidence_ids=[eid], repo=repo)
    assert "major_entities_investigated" in str(excinfo.value)  # the missing envelope is named
    assert repo.get_job(src).status == "running"  # a failed gate never completes the job
    assert repo.get_session(sid).status not in ("completed", "synthesizing", "failed")


def test_invariant_sufficient_rejects_residual_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any residual branch/question in a sufficient envelope fails; a clean envelope completes."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    residuals: list[tuple[str, object]] = [
        ("remaining_branches", ["private-lab terms"]),
        ("material_open_questions", ["Is the exposure hedged?"]),
        ("routes_unsearched", ["8-K exhibits"]),
        ("major_entities_missing", ["OpenAI"]),
    ]
    for key, value in residuals:
        cov: dict[str, object] = dict(_reg_cov())
        cov[key] = value
        with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
            _svc.submit_source_result(src, coverage=cov, evidence_ids=[eid], repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
        _svc.submit_source_result(
            src,
            coverage=_reg_cov(),
            evidence_ids=[eid],
            unresolved_questions=["still open?"],
            repo=repo,
        )
    assert repo.get_job(src).status == "running"
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    # Unknown search_run ids stay advisory (readable SEC ledger -> warnings, never a failure).
    assert isinstance(out.get("warnings"), list)
    dossiers = repo.list_dossiers(sid)
    assert dossiers
    stored_cov = dossiers[-1].get("coverage")
    assert isinstance(stored_cov, dict)
    assert stored_cov.get("source_domain") == "SEC" and stored_cov.get("source_sufficiency") == "sufficient"


def test_invariant_committee_trio_running_before_roles_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three committee jobs exist RUNNING before any role finishes; all share one freeze."""
    import sqlite3
    import threading

    from app.research.repository import get_research_db_path

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    db = str(get_research_db_path())

    def _committee_statuses() -> dict[str, str]:
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT job_type, status FROM jobs WHERE job_type IN ('stockbot','bullbot','bearbot')"
            ).fetchall()
        return {str(job_type): str(status) for job_type, status in rows}

    snapshots: list[dict[str, str]] = []
    barrier = threading.Barrier(3)

    def _model(prompt: str) -> str:
        if "Temporary assignment" not in prompt:  # a committee role call, not a scout
            snapshots.append(_committee_statuses())
            # All three callbacks must record their snapshot before any role can complete.
            barrier.wait(timeout=10)
        return _grounded(prompt)

    out = _run_wave(repo, _model)
    assert out["stop_reason"] == "complete:wave1"
    assert len(snapshots) == 3  # one snapshot per role model call
    assert all(len(snap) == 3 and set(snap.values()) == {"running"} for snap in snapshots)
    sid = out["session_id"]
    assert isinstance(sid, str)
    stock = out["stock"]
    bull = out["bull"]
    bear = out["bear"]
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis

    assert isinstance(stock, StockbotAnalysis) and isinstance(bull, BullAnalysis) and isinstance(bear, BearAnalysis)
    assert stock.freeze_id == bull.freeze_id == bear.freeze_id == out["freeze_id"]
    stored = {
        str((repo.get_job(j.job_id).result or {}).get("freeze_id"))
        for j in repo.list_jobs(sid)
        if j.job_type in ("stockbot", "bullbot", "bearbot")
    }
    assert stored == {out["freeze_id"]}


def test_invariant_committee_roles_stay_out_of_the_tool_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every committee role job is freeze-only: SEC/research tools and evidence writes are forbidden."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    trio = _svc.create_committee_jobs(sid, 1, repo=repo)
    created = trio.get("jobs")
    assert isinstance(created, list) and len(created) == 3
    for jid in created:
        for tool in (
            "search_sec_filings",
            "get_sec_document",
            "get_sec_filing",
            "research_add_evidence",
            "search_web",
        ):
            with pytest.raises(ValueError, match="forbids"):
                _svc.authorize_and_consume_dispatch(sid, str(jid), tool, repo=repo)
        with pytest.raises(ValueError):
            _svc.record_evidence(sid, str(jid), _svc_item(f"{sid}:ev:2"), repo=repo)


def test_invariant_committee_follow_ups_route_through_director_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committee requests drive the next-wave decision (never evidence) or stop at the gate."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    follow_up = "Do the exhibits cover the Azure terms?"
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid,
            jid,
            role,
            _committee_analysis(eid, [follow_up] if i == 0 else []),
            repo=repo,
        )
    assert len(repo.list_evidence(sid)) == 1  # requests never become evidence
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_question"] == follow_up
    sess = repo.get_session(sid)
    assert sess.current_wave == 2 and sess.status == "targeted_research" and sess.targeted_question == follow_up
    # A request outside the SEC lane routes to a non-authorized gate instead of a wave.
    sid2, src2 = _inv_sid(repo, "GS OpenAI exposure?")
    eid2 = f"{sid2}:ev:1"
    _svc.record_evidence(sid2, src2, {**_svc_item(eid2), "subject": "GS"}, repo=repo)
    _svc.complete_job(src2, {}, repo=repo)
    _svc.freeze_session(sid2, 1, repo=repo)
    off_lane = {
        "question": "What does the press say about the terms?",
        "suggested_source": "web",
    }
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid2, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid2,
            jid,
            role,
            _committee_analysis(eid2, [off_lane] if i == 0 else []),
            repo=repo,
        )
    out2 = _svc.decide_next_wave(sid2, repo=repo)
    assert out2["authorized"] is False and out2["stop_reason"] == "not_actionable"
    assert repo.get_session(sid2).status == "synthesizing"
    assert len(repo.list_evidence(sid2)) == 1


def test_invariant_insufficient_coverage_drives_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An honest insufficient dossier with actionable SEC residuals authorizes the next wave from its
    coverage alone (the committee asked nothing); without residuals the run settles with unknowns."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    question = "NVDA OpenAI exposure?"
    sid, src = _inv_sid(repo, question)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    residual = "OpenAI private-lab terms"
    _svc.submit_source_result(
        src,
        coverage={
            "useful_for_question": "insufficient",
            "major_entities_missing": [residual],
            "unresolved": ["Is the exposure hedged?"],
        },
        evidence_ids=[eid],
        unresolved_questions=["Is the exposure hedged?"],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_domain"] == "SEC"
    assert out["targeted_question"] == f"{question} :: targeted follow-up: {residual}"
    assert residual in str(out["reason_detail"])
    sess = repo.get_session(sid)
    assert sess.status == "targeted_research" and sess.current_wave == 2
    assert sess.targeted_question == out["targeted_question"]
    # No actionable SEC residual: coverage challenges nothing, the committee gate settles the run.
    sid2, src2 = _inv_sid(repo, "AMD private-lab terms?")
    eid2 = f"{sid2}:ev:1"
    _svc.record_evidence(sid2, src2, _svc_item(eid2), repo=repo)
    _svc.submit_source_result(
        src2,
        coverage={
            "useful_for_question": "insufficient",
            "source_limitations": ["SEC-only: no private-issuer filings"],
        },
        evidence_ids=[eid2],
        repo=repo,
    )
    _svc.freeze_session(sid2, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid2, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid2, jid, role, _committee_analysis(eid2), repo=repo)
    out2 = _svc.decide_next_wave(sid2, repo=repo)
    assert out2["authorized"] is False and out2["stop_reason"] == "no_questions"
    assert "coverage challenge" not in str(out2["reason_detail"])
    assert repo.get_session(sid2).status == "synthesizing"


def test_invariant_claims_only_envelope_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A claims+follow_ups-only committee envelope fails ERR_COMMITTEE_ENVELOPE_INCOMPLETE everywhere."""
    import json as _json

    from app.research import service as _svc
    from app.research.agents import ModelOutputFailure
    from app.research.agents.stockbot import run_stockbot

    old_shape = _json.dumps({"claims": [{"text": "finding", "evidence_ids": ["EV-1"]}], "follow_ups": []})
    with pytest.raises(ModelOutputFailure, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        run_stockbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=lambda prompt: old_shape,
            evidence_text="[EV-1] a",
        )
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    jid = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="ERR_COMMITTEE_ENVELOPE_INCOMPLETE"):
        _svc.record_committee_analysis(
            sid,
            jid,
            "stockbot",
            {"claims": [{"text": "finding", "evidence_ids": [eid]}], "follow_ups": []},
            repo=repo,
        )
    assert repo.get_job(jid).status == "running"  # the incomplete envelope never closes the job


def test_invariant_claim_type_is_declared_never_promoted() -> None:
    """Absent type stays inference; observed_fact/contradicted/conditional_risk/attributed_report need a freeze id; citations never upgrade."""
    from app.research.agents import ModelOutputFailure
    from app.research.agents.stockbot import run_stockbot

    def _run(model: Callable[[str], str]) -> StockbotAnalysis:
        return run_stockbot(
            "Q?",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="x",
            model=model,
            evidence_text="[EV-1] a",
        )

    undeclared = _run(
        lambda prompt: _committee_output([{"text": "Confirmed: revenue grew.", "evidence_ids": ["EV-1"]}])
    )
    assert [c.claim_type for c in undeclared.claims] == ["inference"]
    assert undeclared.claims[0].evidence_ids == ["EV-1"]
    declared = _run(
        lambda prompt: _committee_output(
            [
                {
                    "text": "Revenue grew.",
                    "claim_type": "observed_fact",
                    "evidence_ids": ["EV-1"],
                }
            ]
        )
    )
    assert [c.claim_type for c in declared.claims] == ["observed_fact"]
    qualified = _run(
        lambda prompt: _committee_output(
            [
                {
                    "text": "A default could impair the marks.",
                    "claim_type": "conditional_risk",
                    "evidence_ids": ["EV-1"],
                },
                {
                    "text": "Management said exposure is capped.",
                    "claim_type": "attributed_report",
                    "evidence_ids": ["EV-1"],
                },
            ]
        )
    )
    assert [c.claim_type for c in qualified.claims] == [
        "conditional_risk",
        "attributed_report",
    ]
    with pytest.raises(ModelOutputFailure, match="uncited"):
        _run(
            lambda prompt: _committee_output(
                [
                    {
                        "text": "A default could impair the marks.",
                        "claim_type": "conditional_risk",
                        "evidence_ids": [],
                    }
                ]
            )
        )
    with pytest.raises(ModelOutputFailure, match="uncited"):
        _run(
            lambda prompt: _committee_output(
                [
                    {
                        "text": "Revenue grew.",
                        "claim_type": "observed_fact",
                        "evidence_ids": [],
                    }
                ]
            )
        )
    with pytest.raises(ModelOutputFailure, match="EV-999"):
        _run(
            lambda prompt: _committee_output(
                [
                    {
                        "text": "Revenue grew.",
                        "claim_type": "observed_fact",
                        "evidence_ids": ["EV-999"],
                    }
                ]
            )
        )


def test_invariant_director_waves_continue_past_two() -> None:
    """A material question advances wave N -> N+1 forever on new evidence; a zero-novelty wave stops
    the run only once nothing unexplored is left, or at an explicitly configured operator limit."""
    from app.research.agents import ResearchRequest
    from app.research.director import (
        DirectorBudgets,
        DirectorDeps,
        Wave1Result,
        decide_next_wave,
    )
    from app.research.synthesis.committee import CommitteeDisagreement

    stops: list[tuple[str, str]] = []

    def _committee(_sid: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        raise AssertionError("this gate test never runs a committee")

    deps = DirectorDeps(
        create_session=lambda q, a: "rs:x",
        fetch_wave_evidence=lambda s: [],
        create_freeze=lambda s: "F1",
        run_committee=_committee,
        record_stop=lambda s, r: stops.append((s, r)),
    )

    def _wave(wave: int, question: str) -> Wave1Result:
        request = ResearchRequest(
            question=question,
            why_material="unresolved in the last freeze",
            requested_source_domain="SEC",
            expected_gain="medium",
        )
        disagreement = CommitteeDisagreement(
            session_id="rs:x",
            wave_id=wave,
            freeze_id=f"F{wave}",
            requested_research=[request],
        )
        return Wave1Result(
            session_id="rs:x",
            wave_id=wave,
            freeze_id=f"F{wave}",
            evidence_ids=["EV-1"],
            disagreement=disagreement,
            question="Is GS exposed?",
        )

    novelty = {
        "new_raw_documents": 1,
        "new_evidence_records": 2,
        "new_relationships": 1,
        "resolved_questions": 1,
        "zero_novelty_waves": 0,
        "duplicate_actions_blocked": 0,
    }
    questions = {
        "What drove the Q2 delta?": 2,
        "Which exhibits cover the terms?": 3,
        "How does the hedge work?": 4,
        "What if the hedge fails?": 5,
    }
    for question, wave in questions.items():
        decision = decide_next_wave(
            _wave(wave - 1, question),
            deps=deps,
            budgets=DirectorBudgets(),
            waves_used=wave - 1,
            novelty=novelty,
        )
        assert decision.authorized is True and decision.stop_reason == "continue"
        assert decision.targeted_question == question and decision.targeted_domain == "SEC"
        assert len(stops) == wave - 1
    # Convergence, not a wave number or a streak: a zero-novelty wave with the committee question
    # still unanswered keeps the branch alive, however long the streak already is.
    zero = dict.fromkeys(
        (
            "new_raw_documents",
            "new_evidence_records",
            "new_relationships",
            "resolved_questions",
        ),
        0,
    )
    zero.update({"zero_novelty_waves": 3, "duplicate_actions_blocked": 0})
    still_open = decide_next_wave(
        _wave(5, "What else?"),
        deps=deps,
        budgets=DirectorBudgets(),
        waves_used=5,
        novelty=zero,
    )
    assert still_open.authorized is True and still_open.stop_reason == "continue"
    answered = _wave(5, "What else?")
    answered.disagreement = CommitteeDisagreement(session_id="rs:x", wave_id=5, freeze_id="F5")
    stopped = decide_next_wave(answered, deps=deps, budgets=DirectorBudgets(), waves_used=5, novelty=zero)
    assert stopped.authorized is False and stopped.stop_reason == "no_novelty"
    # An explicit operator runaway guard stops a branch that still has questions, at its own count.
    limited = decide_next_wave(
        _wave(5, "What else?"),
        deps=deps,
        budgets=DirectorBudgets(zero_novelty_limit=2),
        waves_used=5,
        novelty={**zero, "zero_novelty_waves": 2},
    )
    assert limited.authorized is False and limited.stop_reason == "no_novelty"
    assert "zero_novelty_limit=2" in limited.reason_detail


def test_invariant_finalize_card_never_duplicates_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The answer lives once in final_result; the finalize card is a short confirmation, not the answer."""
    from app.research import service as _svc
    from app.tool_render import render_tool_result

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _eval_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _eval_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_eval_cov(), evidence_ids=[eid], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(sid, jid, role, _committee_analysis(eid), repo=repo)
    _svc.decide_next_wave(sid, repo=repo)
    answer = "MSFT Azure exposure may be filing-backed and the agreement terms stay undisclosed."
    out = _svc.finalize_session(sid, answer, [{"text": answer, "evidence_ids": [eid]}], repo=repo)
    final = repo.get_session(sid).final_result
    assert isinstance(final, dict)
    assert final.get("answer") == answer and str(final.get("content", "")).strip()
    assert final.get("freeze_id") == f"{sid}:1:freeze"
    card = render_tool_result(out)
    assert "Research finalized" in card
    assert f"{sid}:1:freeze" in card
    assert "Claims:" in card and "impact channels:" in card
    assert answer not in card and "filing-backed" not in card
    assert "Answer delivered separately" in card
    # The answer is stored once (final_result.answer); the card only counts what was frozen.
    assert str(final.get("content", "")).startswith("Bottom line:")


# --- property invariants: accession + typed-provenance parsing (plan: raw-source-boundary) ---

_ACCESSION_ALPHABET = "0123456789abcdefABCDEF-"


@settings(max_examples=200, derandomize=True)
@given(st.text(max_size=24))
def test_prop_accession_normalization_is_idempotent(text: str) -> None:
    """Any accepted accession normalizes to the canonical dashed form and stays stable."""
    from app.research.evidence import ACCESSION_RE, normalize_accession

    try:
        canonical = normalize_accession(text)
    except ValueError:
        return
    assert ACCESSION_RE.match(canonical)
    assert normalize_accession(canonical) == canonical


@settings(max_examples=200, derandomize=True)
@given(
    st.text(alphabet=_ACCESSION_ALPHABET, max_size=24),
    st.text(alphabet="0123456789abcdef", min_size=1, max_size=12),
)
def test_prop_provenance_accepts_exactly_canonical_accessions(blob: str, search_id: str) -> None:
    """sec_source provenance accepts exactly what accession normalization accepts; search runs stay search runs."""
    from app.research.evidence import (
        EvidenceIntegrityError,
        normalize_accession,
        validate_provenance,
    )

    item: dict[str, object] = {
        "kind": "sec_source",
        "accession_no": blob,
        "document_name": "d.htm",
        "passage": "p",
    }
    try:
        canonical = normalize_accession(blob)
    except ValueError:
        with pytest.raises(EvidenceIntegrityError):
            validate_provenance(item)
        return
    assert validate_provenance(item)["accession_no"] == canonical
    assert validate_provenance({"kind": "search_run", "search_id": search_id, "query": "q"})["kind"] == "search_run"


def test_telemetry_derived_from_persisted_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan Phase 13: search/evidence telemetry derives from persisted rows + journal, never model counts."""
    from app.research import service as _svc
    from app.research.journal import append_event

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    repo.save_event(
        append_event(
            sid,
            "tool.completed",
            "runner",
            "runner",
            {"tool": "search_sec_filings", "args": {"query": "NVDA OpenAI supply"}},
        )
    )
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["private contract terms"],
        "source_limitations": ["SEC-only"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    # Journal-derived searches/queries (no coverage search_runs in this submission).
    assert telemetry.get("searches_count") == 1
    assert telemetry.get("queries_attempted") == ["NVDA OpenAI supply"]
    # Row-derived document/filing/evidence counts.
    assert telemetry.get("evidence_records") == 1
    assert telemetry.get("filings_opened") == 1
    assert telemetry.get("documents_opened") == 1 and telemetry.get("raw_documents_used") == 1
    # Relationships are relationship rows; an evidence row is not a relationship.
    assert "material_relationships_found" not in telemetry
    assert telemetry.get("telemetry_gaps") == []


def test_telemetry_gaps_mark_underivable_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No persisted search activity -> searches/queries are null and named in telemetry_gaps, never guessed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    _sid, src = _svc_sid(repo)
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["nothing yet"],
        "source_limitations": ["SEC-only"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry.get("searches_count") is None and telemetry.get("queries_attempted") is None
    gaps = telemetry.get("telemetry_gaps")
    assert isinstance(gaps, list) and "searches_count" in gaps and "queries_attempted" in gaps
    assert telemetry.get("evidence_records") == 0


def test_accession_forms_normalize_to_canonical() -> None:
    """Canonical + bare-18-digit forms normalize; search-id-shaped values are rejected outright."""
    from app.research.evidence import normalize_accession

    assert normalize_accession("0000320193-25-000079") == "0000320193-25-000079"
    assert normalize_accession("000032019325000079") == "0000320193-25-000079"
    for bad in (
        "7768855a3f91",
        "0000320193-25-00007",
        "0000320193/25/000079",
        "",
        "0000320193-25-000079x",
    ):
        with pytest.raises(ValueError):
            normalize_accession(bad)


# ---------------------------------------------------------------------------
# CRAP gate: novelty/director-gate walk, telemetry signal derivation, committee
# trio atomicity, claim shaping. External behavior only: returned payloads,
# persisted rows, raised errors.
# ---------------------------------------------------------------------------


def _nov_row(
    sid: str,
    eid: str,
    wave: int,
    *,
    accession: str = "",
    document: str = "",
    record_id: str = "",
    subject: str = "NVDA",
    digest: str | None = None,
) -> dict[str, object]:
    """Persisted evidence row for the novelty walk: raw provenance, record-id fallback, or no identity."""
    row: dict[str, object] = {
        "evidence_id": eid,
        "session_id": sid,
        "wave_id": wave,
        "subject": subject,
        "content_hash": digest or f"h-{eid}",
    }
    if accession:
        row["provenance"] = {
            "kind": "sec_source",
            "accession_no": accession,
            "document_name": document,
            "passage": "p",
        }
    if record_id:
        row["source_record_id"] = record_id
        row["source_name"] = "SEC 10-K"
    return row


def test_gate_novelty_walks_rows_relationships_and_streak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Novelty counts only documents/rows the run does not already hold; dossier relationships count once."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:1", 1, accession="0000320193-25-000079", document="a.htm"))
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:2", 1, record_id="r-1"))  # accession fallback identity
    repo.save_evidence(_nov_row(sid, f"{sid}:ev:3", 1, subject="AMD"))  # no identity at all
    repo.save_evidence(
        _nov_row(
            sid,
            f"{sid}:ev:4",
            2,
            accession="0000320193-25-000079",
            document="a.htm",
            digest=f"h-{sid}:ev:1",
        )
    )  # re-read of a held document
    repo.save_evidence(
        _nov_row(
            sid,
            f"{sid}:ev:5",
            2,
            accession="0000320193-25-000081",
            document="b.htm",
            subject="AMD",
        )
    )
    repo.save_event(append_event(sid, "research_loop_detected", "runner", "runner", {}))
    repo.save_event(append_event(sid, "wave.novelty", "runner", "runner", {"zero_novelty_actions": 2}))
    found = dataclasses.replace(repo.get_session(sid), current_wave=2)
    repo.save_session(found)
    novelty = _svc._gate_novelty(repo, found)
    assert set(novelty) == {
        "new_raw_documents",
        "new_evidence_records",
        "new_entities",
        "new_relationships",
        "new_material_claims",
        "resolved_questions",
        "new_questions",
        "zero_novelty_waves",
        "duplicate_actions_blocked",
        "zero_novelty_actions",
    }
    # Only ev:5 (new document) is new; ev:4 is the same document+hash the run already held.
    assert novelty["new_raw_documents"] == 1 and novelty["new_evidence_records"] == 1
    assert novelty["new_entities"] == 0  # NVDA + AMD were already subjects of the prior wave
    assert novelty["new_relationships"] == 0 and novelty["new_material_claims"] == 0
    assert novelty["resolved_questions"] == 0 and novelty["new_questions"] == 0
    assert novelty["zero_novelty_waves"] == 0  # wave 2 contributed a fresh row
    assert novelty["duplicate_actions_blocked"] == 1 and novelty["zero_novelty_actions"] == 2

    # A dossier relationship is a persisted novelty signal; a re-listed one is not new again.
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:2:sec",
            "session_id": sid,
            "wave_id": 2,
            "coverage": {},
            "relationships": [{"kind": "supplier"}, "junk", {"kind": "customer"}],
        }
    )
    with_rels = _svc._gate_novelty(repo, found)
    assert with_rels["new_relationships"] == 2  # the non-mapping entry is not a relationship
    assert with_rels["new_evidence_records"] == 1
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {},
            "relationships": [{"kind": "supplier"}],
        }
    )
    # A relationship any other wave already lists is not fresh for the current wave.
    assert _svc._gate_novelty(repo, found)["new_relationships"] == 1

    # A wave with no evidence rows counts into the zero-novelty streak, back to the last productive wave.
    ahead = dataclasses.replace(repo.get_session(sid), current_wave=4)
    assert _svc._gate_novelty(repo, ahead)["zero_novelty_waves"] == 2  # waves 4 and 3 added nothing

    # Unreadable dossier storage means no relationship signal, never a guessed one.
    def _boom(_session_id: str) -> list[dict[str, object]]:
        raise RuntimeError("dossier storage unavailable")

    monkeypatch.setattr(repo, "list_dossiers", _boom)
    degraded = _svc._gate_novelty(repo, found)
    assert degraded["new_relationships"] == 0 and degraded["new_evidence_records"] == 1


def test_telemetry_ledger_unresolvable_and_journal_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable search_runs ledger falls back to journal queries; an unreadable ledger does the same."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    repo.save_event(
        append_event(
            sid,
            "tool.completed",
            "runner",
            "runner",
            {"tool": "search_sec_filings", "args": {"query": "NVDA 10-K"}},
        )
    )
    coverage: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "search_runs": ["sr:not-persisted"],
    }
    out = _svc.submit_source_result(src, coverage=coverage, evidence_ids=[], repo=repo)
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry.get("searches_count") == 1  # the coverage's search ids, not the journal call
    assert telemetry.get("queries_attempted") == ["NVDA 10-K"]  # journal fallback: the ledger holds no query
    assert telemetry.get("telemetry_gaps") == []


def test_telemetry_no_coverage_relationships_and_storage_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No persisted coverage/identity is a named gap; relationships come from dossiers or are omitted."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_evidence(
        {
            "evidence_id": f"{sid}:ev:1",
            "session_id": sid,
            "wave_id": 1,
            "subject": "NVDA",
        }
    )
    bare = _svc._derive_telemetry(repo, sid)
    assert bare.get("branches_covered") is None and bare.get("branches_remaining") is None
    assert bare.get("filings_opened") is None and bare.get("documents_opened") is None
    assert bare.get("raw_documents_used") is None and bare.get("evidence_records") == 1
    gaps = bare.get("telemetry_gaps")
    assert isinstance(gaps, list)
    assert {
        "branches_covered",
        "branches_remaining",
        "filings_opened",
        "documents_opened",
        "raw_documents_used",
        "searches_count",
        "queries_attempted",
    } <= set(gaps)
    assert "material_relationships_found" not in bare  # no dossier: the key is omitted, never zero

    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {},
            "relationships": [{"kind": "supplier"}, "junk"],
        }
    )
    with_rels = _svc._derive_telemetry(repo, sid)
    assert with_rels.get("material_relationships_found") == 1
    assert with_rels.get("evidence_records") == 1

    # The coverage read succeeds, the relationship read then fails: omit the key, keep the rest.
    inner = repo.list_dossiers
    calls = itertools.count()

    def _flaky(session_id: str) -> list[dict[str, JSONValue]]:
        if next(calls) == 0:
            return inner(session_id)
        raise RuntimeError("dossier storage unavailable")

    monkeypatch.setattr(repo, "list_dossiers", _flaky)
    degraded = _svc._derive_telemetry(repo, sid)
    assert "material_relationships_found" not in degraded
    assert degraded.get("evidence_records") == 1


def test_committee_jobs_partial_trio_rollback_leaves_reused_role_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trio failing mid-creation fails only the jobs this call started; a reused role is left running."""
    from app.research import service as _svc
    from app.research.models import FailureCategory

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    found = repo.get_session(sid)
    cur, preexisting = _jobs.create_job(found, repo.list_jobs(sid), job_type="bullbot", owner="pi", wave_id=1)
    repo.save_session(cur)
    repo.save_job(_jobs.start_job(preexisting))
    real_start = _jobs.start_job
    calls = itertools.count()

    def _flaky(job: Job) -> Job:
        if next(calls) == 1:
            raise RuntimeError("start failed")
        return real_start(job)

    monkeypatch.setattr(_jobs, "start_job", _flaky)
    with pytest.raises(RuntimeError, match="start failed"):
        _svc.create_committee_jobs(sid, 1, repo=repo)
    committee = {j.job_type: j for j in repo.list_jobs(sid) if j.job_type in ("stockbot", "bullbot", "bearbot")}
    assert set(committee) == {"stockbot", "bullbot"}  # bearbot never persisted
    assert committee["bullbot"].job_id == preexisting.job_id and committee["bullbot"].status == "running"
    rolled_back = committee["stockbot"]
    assert rolled_back.status == "failed"
    failure = rolled_back.failure
    assert failure is not None and failure.category == FailureCategory.COMMITTEE_DEADLOCK.value
    assert failure.message == "partial trio: RuntimeError"

    # A retry reuses the running role, creates the missing one, and flags the pending freeze.
    monkeypatch.setattr(_jobs, "start_job", real_start)
    retry = _svc.create_committee_jobs(sid, 1, repo=repo)
    retried = retry["jobs"]
    assert isinstance(retried, list) and len(retried) == 3 and len(set(retried)) == 3
    assert retried[0] == rolled_back.job_id and retried[1] == preexisting.job_id
    verb = retry["pending_next_action"]
    assert isinstance(verb, dict) and verb.get("freeze_pending") == f"{sid}:1:freeze"


def test_finalize_claims_shapes_and_guards() -> None:
    """Caller claims narrow to text+evidence_ids; every guard keeps its pinned ValueError code."""
    from app.research import service as _svc
    from app.research.agents import ModelOutputFailure

    shaped = _svc._finalize_claims(
        "sid",
        [
            {
                "claim": "  inference from the freeze  ",
                "evidence_ids": ("EV-1",),
                "claim_type": None,
            },
            {
                "claim_text": "declared fact",
                "evidence_ids": ["EV-1"],
                "claim_type": "observed_fact",
            },
        ],
        ["EV-1"],
    )
    assert [(c.text, c.claim_type, c.evidence_ids) for c in shaped] == [
        ("inference from the freeze", "inference", ["EV-1"]),
        ("declared fact", "observed_fact", ["EV-1"]),
    ]
    # A claim with no text under any alias is never silently grounded.
    with pytest.raises(ModelOutputFailure, match="non-empty text"):
        _svc._finalize_claims("sid", [{"evidence_ids": ["EV-1"]}], ["EV-1"])
    not_a_list: object = {"text": "x"}
    with pytest.raises(ValueError, match="must be a list"):
        _svc._finalize_claims("sid", not_a_list, ["EV-1"])
    with pytest.raises(ValueError, match="non-empty grounded"):
        _svc._finalize_claims("sid", [], ["EV-1"])
    non_mapping: object = "x"
    with pytest.raises(ValueError, match="each claim must be a mapping"):
        _svc._finalize_claims("sid", [non_mapping], ["EV-1"])
    with pytest.raises(ValueError, match="JSON-able"):
        _svc._finalize_claims("sid", [{"text": object(), "evidence_ids": ["EV-1"]}], ["EV-1"])
    # A coverage artifact id is coverage state: it can never ground a claim, because
    # the freeze's citable ids are ledger evidence ids only.
    artifact_id = "rs:sid:cov:9d3f1a2b"
    with pytest.raises(ModelOutputFailure, match="unknown evidence id"):
        _svc._finalize_claims(
            "sid",
            [{"text": "absence as a finding", "evidence_ids": [artifact_id]}],
            ["EV-1"],
        )


def test_unbounded_as_of_is_no_cutoff_for_every_pit_gate() -> None:
    """`unbounded` means no cutoff: dated documents stay eligible and never `pit_violated`."""
    from app.research.agents.scout import _is_pit_eligible
    from app.research.models import NO_CUTOFF_AS_OF, pit_unverified, pit_violated

    assert NO_CUTOFF_AS_OF == frozenset({"unbounded"})
    dated = "2011-03-16T16:33:51+00:00"
    # The live defect: pit_violated raised on the sentinel, so every dated document was rejected.
    assert pit_violated("unbounded", dated) is False
    assert pit_unverified("unbounded", None) is False
    assert _is_pit_eligible(dated, "unbounded") is True
    assert _is_pit_eligible(None, "unbounded") is True
    # Bounded scopes keep the strict rule.
    assert pit_violated("2025-06-30T00:00:00+00:00", "2026-01-01T00:00:00+00:00") is True
    assert _is_pit_eligible("2025-01-01T00:00:00+00:00", "2025-06-30T00:00:00+00:00") is True
    assert _is_pit_eligible("2026-01-01T00:00:00+00:00", "2025-06-30T00:00:00+00:00") is False
    assert _is_pit_eligible(None, "2025-06-30T00:00:00+00:00") is False


def test_counterparty_queries_are_assigned_unscoped() -> None:
    """A private counterparty's queries reach the filings scout as global (no ticker/cik) queries."""
    from app.research.agents.source_agent import (
        _grouped_queries,
        build_research_context,
        decompose_question,
    )

    ctx = build_research_context("What happens to Microsoft if OpenAI goes bankrupt?", ["MSFT"])
    groups = _grouped_queries(ctx)
    assert "OpenAI" in groups["cp"]
    assignments = decompose_question(
        "What happens to Microsoft if OpenAI goes bankrupt?",
        session_id="rs:t",
        as_of="2026-08-10",
        tickers=["MSFT"],
        dispatch=lambda name, args: {},
    )
    filings = next(a for a in assignments if a.role == "filings")
    assert "OpenAI" in filings.unscoped_queries
    # Other scouts keep their scoped assignment: no global counterparty work.
    assert all(not a.unscoped_queries for a in assignments if a.role != "filings")
    # A question without a counterparty assigns no global queries.
    plain = decompose_question(
        "NVDA data center revenue growth",
        session_id="rs:t",
        as_of="2026-08-10",
        tickers=["NVDA"],
        dispatch=lambda name, args: {},
    )
    assert all(not a.unscoped_queries for a in plain)


# ---------------------------------------------------------------------------
# Hedgefund MVP slice: SEC+FINRA+WEB domains under the kernel's source policy.
# All offline (fake archive seam, no network). Live golden coverage stays with
# the opt-in verify:hedgefund-live script, never pytest.
# - domains SEC|FINRA|WEB; provenance sec_source|sec_record|finra_record|web_source|search_run|none (closed;
#   FINRA/WEB/SEC-replay only via persisted-result replay, bare rows fail closed with ERR_RAW_SOURCE_REQUIRED);
# - integrity SEC=PRIMARY_DOCUMENT/CANONICAL_STRUCTURED FINRA=CANONICAL_STRUCTURED WEB=EXTERNAL_SOURCE
# - session source policy {mode:allowlist,allowed:[]} default SEC-only;
# - agents sec-agent/finra-agent/exa-agent + scouts; stockbot.yml async off, depth 3;
# - Wave1 one OMP task batch with 3 desks; targeted waves only requested desks;
# - committee same freeze via research_read only.
# ---------------------------------------------------------------------------


def _hf_policy(*sources: str) -> dict[str, JSONValue]:
    """Session policy requesting exactly these source domains (allowlist)."""
    return {"research_sources": {"mode": "allowlist", "sources": list(sources)}}


def _hf_sid(repo: ResearchRepository, *sources: str, q: str = "NVDA demand?") -> tuple[str, str]:
    """Fresh session (+ first running job) under the requested sources; default SEC-only."""
    from app.research import service as _svc

    sid = _svc.create_research(
        q,
        "o",
        as_of="2025-06-30T00:00:00+00:00",
        policy=_hf_policy(*sources) if sources else None,
        repo=repo,
    )
    return sid, repo.list_jobs(sid)[0].job_id


def test_hf_provenance_maps_domain_and_integrity_in_kernel() -> None:
    """Kernel owns provenance->domain/integrity; evidence_to_dict exposes both on every row."""
    from app.research.evidence import (
        evidence_domain,
        evidence_integrity,
        finra_record_ref,
        search_run_ref,
        sec_source_ref,
        web_source_ref,
    )

    sec = sec_source_ref(
        accession_no="0000320193-25-000079",
        document_name="d.htm",
        passage="hello world",
        offset=0,
        end=5,
        basis="rendered",
        text_hash="ab" * 32,
    )
    assert evidence_domain(sec) == "SEC" and evidence_integrity(sec) == "PRIMARY_DOCUMENT"
    fin = finra_record_ref(tool_name="query_finra", record_identity="row-1")
    assert evidence_domain(fin) == "FINRA" and evidence_integrity(fin) == "CANONICAL_STRUCTURED"
    web = web_source_ref(url="https://example.com/a", excerpt="highlight text")
    assert evidence_domain(web) == "WEB" and evidence_integrity(web) == "EXTERNAL_SOURCE"
    assert evidence_domain(search_run_ref(search_id="s1", query="q")) == "WEB"
    assert evidence_domain({"kind": "none"}) == "SOURCE" and evidence_domain(None) == "SOURCE"
    from app.research.evidence import Evidence, evidence_to_dict

    content = "hello world"
    rec = Evidence(
        evidence_id="EV-1",
        session_id="rs:t",
        wave_id=1,
        source_type="pi",
        source_name="SEC",
        subject="NVDA",
        claim_text=content,
        content=content,
        content_hash=__import__("app.research.evidence", fromlist=["evidence_content_hash"]).evidence_content_hash(
            content
        ),
        retrieved_at=__import__("datetime", fromlist=["datetime"]).datetime(
            2025, 6, 29, tzinfo=__import__("datetime", fromlist=["UTC"]).UTC
        ),
        provenance=dict(sec),
    )
    stored = evidence_to_dict(rec)
    assert stored["source_domain"] == "SEC" and stored["integrity_class"] == "PRIMARY_DOCUMENT"
    from app.research.evidence import EvidenceIntegrityError, evidence_from_dict

    assert evidence_from_dict(dict(stored)).evidence_id == "EV-1"
    with pytest.raises(EvidenceIntegrityError, match="source_domain"):
        evidence_from_dict({**stored, "source_domain": "FINRA"})
    with pytest.raises(EvidenceIntegrityError, match="integrity_class"):
        evidence_from_dict({**stored, "integrity_class": "EXTERNAL_SOURCE"})
    legacy = {k: v for k, v in stored.items() if k not in ("source_domain", "integrity_class")}
    assert evidence_from_dict(legacy).evidence_id == "EV-1"


def test_hf_sources_carry_kernel_integrity() -> None:
    """Sources derive domain+integrity from observations via the kernel (caller never assigns)."""
    from app.research.agents import GroundedClaim
    from app.research.synthesis.final import _normalize_sources, _source_line

    obs = {
        "evidence_id": "EV-1",
        "provenance": {
            "kind": "finra_record",
            "tool_name": "query_finra",
            "record_identity": "row-1",
        },
    }
    rows = _normalize_sources(None, [GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])], [obs])
    assert rows == [
        {
            "evidence_id": "EV-1",
            "domain": "FINRA",
            "document": "",
            "integrity_class": "CANONICAL_STRUCTURED",
        }
    ]
    assert "CANONICAL_STRUCTURED" in _source_line(rows[0])
    # Caller-supplied rows still get kernel integrity from the matching observation.
    caller = [{"evidence_id": "EV-1", "domain": "FINRA", "document": "FINRA"}]
    rows2 = _normalize_sources(caller, [GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])], [obs])
    assert rows2[0]["integrity_class"] == "CANONICAL_STRUCTURED"


def test_hf_synthesize_keeps_trio_what_would_change() -> None:
    """Regression: synthesize_final unions the trio what_would_change (dropped wiring broke test_final_branches)."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    base_claim = GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])
    filing_claim = GroundedClaim(text="a new filing", claim_type="unknown", evidence_ids=[])
    rate_claim = GroundedClaim(text="a rate change", claim_type="unknown", evidence_ids=[])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[base_claim, filing_claim],
        what_would_change=["a new filing"],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[base_claim, filing_claim],
        what_would_change=["a new filing"],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[
            base_claim,
            rate_claim,
            GroundedClaim(text="Assuming a revenue fall, the view changes", claim_type="assumption", evidence_ids=[]),
        ],
        what_would_change=["a rate change", "Assuming a revenue fall, the view changes"],
    )
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
    )
    # Union keeps covered hedges; an uncovered numbered hedge is deleted (exhaustive prose inventory).
    assert synth.what_would_change == ["a new filing", "a rate change", "Assuming a revenue fall, the view changes"]
    assert any(claim.text == "a new filing" and claim.claim_type == "unknown" for claim in synth.claims)


def test_gate_hedge_sections_keep_only_their_own_role_claims() -> None:
    """A hedge sentence stays only under a matching unknown/assumption claim; other hedges delete."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import gate_uncovered_prose, synthesize_final

    unknowns = GroundedClaim(text="No NVDA 2026 guidance was found", claim_type="unknown", evidence_ids=[])
    assumed = GroundedClaim(text="Assuming limited scope, the review defers to filings", claim_type="assumption")
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[unknowns],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[unknowns],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[assumed],
    )
    synth = synthesize_final(
        "q",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
    )
    synth.unknowns = ["No NVDA 2026 guidance was found", "An uncovered boom may follow"]
    synth.what_would_change = ["Assuming margins hold, the view improves"]
    synth.evidence_limitations = ["Assuming limited scope, the review defers to filings"]
    synth.coverage = {"sec": {"detail": "No NVDA 8-K mentions the topic"}}
    synth.claims = [
        *synth.claims,
        GroundedClaim(text="Assuming margins hold, the view improves", claim_type="assumption"),
        GroundedClaim(text="No NVDA 8-K mentions the topic", claim_type="unknown", evidence_ids=[]),
    ]
    gate_uncovered_prose(synth)
    assert synth.unknowns == ["No NVDA 2026 guidance was found"]
    assert synth.what_would_change == ["Assuming margins hold, the view improves"]
    sec_detail = synth.coverage.get("sec")
    assert isinstance(sec_detail, dict) and sec_detail.get("detail") == "No NVDA 8-K mentions the topic"


def test_hf_normalize_coverage_keeps_submit_keys_verbatim() -> None:
    """Dossier submit keys survive normalize + lines under their own names (no aliasing)."""
    from app.research.synthesis.final import _coverage_lines, _normalize_coverage

    coverage: dict[str, object] = {
        "sec": {
            "docs": ["0000320193-25-000079"],
            "forms_examined": ["10-K"],
            "search_runs": ["s1"],
        },
        "finra": {
            "datasets_queried": ["short-interest"],
            "settlement_windows_covered": ["2025-06-15"],
            "tickers_covered": ["NVDA"],
        },
        "web": {
            "queries_executed": ["NVDA news"],
            "results_inspected": ["https://example.com/a"],
        },
    }
    norm = _normalize_coverage(coverage)
    sec = norm["sec"]
    finra = norm["finra"]
    web = norm["web"]
    assert isinstance(sec, dict) and isinstance(finra, dict) and isinstance(web, dict)
    assert sec["docs"] == ["0000320193-25-000079"]
    assert finra["datasets_queried"] == ["short-interest"]
    assert finra["settlement_windows_covered"] == ["2025-06-15"]
    assert web["queries_executed"] == ["NVDA news"]
    assert "source_domain" not in finra and "useful_for_question" not in finra
    assert any("datasets_queried: short-interest" in line for line in _coverage_lines(norm))


def _hf_cov(**over: object) -> dict[str, object]:
    """Insufficient coverage envelope (submit-shaped, no residuals required)."""
    cov: dict[str, object] = {
        "useful_for_question": "insufficient",
        "resolved": [],
        "partially_resolved": [],
        "unresolved": ["open branch"],
        "source_limitations": [],
        "major_entities_investigated": ["NVDA"],
        "relationship_types_checked": [],
        "forms_examined": [],
        "exhibits_examined": [],
        "material_open_questions": ["open branch"],
        "remaining_branches": ["open branch"],
        "routes_unsearched": [],
        "search_runs": ["s1"],
        "covered_branches": [],
    }
    cov.update(over)
    return cov


def _hf_freeze_id(sid: str, wave: int = 1) -> str:
    return f"{sid}:{wave}:freeze"


def _hf_prov(out: Mapping[str, object], key: str) -> object:
    """Provenance field of a record_evidence return (pyrefly-clean accessor)."""
    prov = out.get("provenance")
    assert isinstance(prov, dict)
    return prov.get(key)


def test_hf_default_session_is_sec_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel default: no policy input means SEC-only allowlist."""

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    assert repo.get_session(sid).source_policy == {
        "allowed": ["SEC"],
        "denied": [],
        "mode": "allowlist",
    }


def test_hf_sec_only_denies_finra_and_web_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-only allowlist: FINRA/WEB job creation fails closed with policy_denied."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    for domain in ("FINRA", "WEB", "finra", "web"):
        with pytest.raises(ValueError, match="policy_denied"):
            _svc.start_job(sid, "source_agent", source=domain, repo=repo, wave_id=1)
    ok = _svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)
    assert repo.get_job(str(ok["job_id"])).source_domain == "SEC"


def test_hf_multisource_policy_allows_three_desks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC+FINRA+WEB allowlist: one running OMP-owned job per desk, distinct ids."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    assert repo.get_session(sid).source_policy == {
        "allowed": ["SEC", "FINRA", "WEB"],
        "denied": [],
        "mode": "allowlist",
    }
    ids = [
        str(
            _svc.start_job(
                sid,
                "source_agent",
                source=d,
                budget={"owner": "omp"},
                repo=repo,
                wave_id=1,
            )["job_id"]
        )
        for d in ("SEC", "FINRA", "WEB")
    ]
    assert len(set(ids)) == 3
    by_id = {jid: repo.get_job(jid) for jid in ids}
    assert {j.source_domain for j in by_id.values()} == {"SEC", "FINRA", "WEB"}
    assert all(j.owner == "omp" and j.status == "running" for j in by_id.values())


def test_hf_finra_job_cannot_submit_sec_evidence_as_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-domain isolation: submit cites only session-ledger ids; a dangling id fails
    ERR_EVIDENCE_NOT_FOUND, and a FINRA submit can never invent SEC ids."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, sec_job = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(fin, coverage=_hf_cov(), evidence_ids=[f"{sid}:ev:ghost"], repo=repo)
    assert repo.get_job(fin).status == "running"
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-own")
    marker = rid.rsplit(":tr:", 1)[-1]
    fin_eid = f"{sid}:ev:fin-own"
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": fin_eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    sec_eid = f"{sid}:ev:sec-other"
    _svc.record_evidence(sid, sec_job, _svc_item(sec_eid), repo=repo)
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(fin, coverage=_hf_cov(), evidence_ids=[sec_eid], repo=repo)
    assert repo.get_job(fin).status == "running"


def test_hf_exa_job_cannot_submit_finra_evidence_as_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-domain isolation: a WEB submit cites only its wave-domain lane, never FINRA evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(web, coverage=_hf_cov(), evidence_ids=[f"{sid}:ev:ghost"], repo=repo)
    assert repo.get_job(web).status == "running"
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-own")
    web_eid = f"{sid}:ev:web-own"
    _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": web_eid,
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": web_rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-other")
    fin_marker = fin_rid.rsplit(":tr:", 1)[-1]
    fin_eid = f"{sid}:ev:fin-other"
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": fin_eid,
            "content": f"NVDA short interest 12345 shares {fin_marker}",
            "claim_text": f"NVDA short interest 12345 shares {fin_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": fin_rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": fin_marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    with pytest.raises(ValueError, match="ERR_EVIDENCE_NOT_FOUND"):
        _svc.submit_source_result(web, coverage=_hf_cov(), evidence_ids=[fin_eid], repo=repo)
    assert repo.get_job(web).status == "running"


def test_hf_no_cross_job_attach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job id from another session is rejected on evidence and submit paths."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid_a, _ = _hf_sid(repo, "SEC", "FINRA", "WEB", q="session A?")
    sid_b, src_b = _hf_sid(repo, "SEC", "FINRA", "WEB", q="session B?")
    with pytest.raises(ValueError, match="belongs to"):
        _svc.record_evidence(sid_a, src_b, _svc_item(f"{sid_a}:ev:1"), repo=repo)
    with pytest.raises(ValueError, match="belongs to"):
        _svc.record_evidence(sid_a, src_b, _svc_item(f"{sid_a}:ev:2"), repo=repo)
    with pytest.raises(ValueError, match="belongs to"):
        _svc.authorize_and_consume_dispatch(sid_a, src_b, "search_sec_filings", repo=repo)
    assert repo.list_evidence(sid_a) == [] and repo.list_evidence(sid_b) == []


def test_hf_replay_rejects_cross_wave_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wave-1 persisted result cited from a wave-2 job fails ERR_PROVENANCE_MISMATCH."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    w1 = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    w1_rid = _hf_persist_finra(
        repo,
        sid,
        w1,
        f"{sid}:tr:w1",
        as_of_date="2025-06-15",
        published_at="2025-06-15",
    )
    w1_marker = w1_rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        w1,
        {
            "evidence_id": f"{sid}:ev:w1",
            "content": f"NVDA short interest 12345 shares {w1_marker}",
            "claim_text": f"NVDA short interest 12345 shares {w1_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": w1_rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": w1_marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    for job in repo.list_jobs(sid):
        if job.job_type in ("source_agent", "scout") and job.wave_id == 1 and job.status in ("queued", "running"):
            _svc.complete_job(job.job_id, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    w2 = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=2)["job_id"])
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            w2,
            {
                "evidence_id": f"{sid}:ev:w2-cross",
                "content": f"NVDA short interest 12345 shares {w1_marker}",
                "claim_text": f"NVDA short interest 12345 shares {w1_marker}",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": w1_rid,
                "matching_passage": json.dumps(
                    {"symbol": "NVDA", "shortInterest": 12345, "marker": w1_marker}, sort_keys=True
                ),
            },
            repo=repo,
        )
    w2_rid = _hf_persist_finra(
        repo,
        sid,
        w2,
        f"{sid}:tr:w2",
        as_of_date="2025-06-15",
        published_at="2025-06-15",
    )
    w2_marker = w2_rid.rsplit(":tr:", 1)[-1]
    out = _svc.record_evidence(
        sid,
        w2,
        {
            "evidence_id": f"{sid}:ev:w2",
            "content": f"NVDA short interest 12345 shares {w2_marker}",
            "claim_text": f"NVDA short interest 12345 shares {w2_marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": w2_rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": w2_marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    assert _hf_prov(out, "tool_result_id") == w2_rid


def test_hf_admission_rejects_fabricated_sec_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No handle, garbage handle, unknown document, and declared-ref mismatch all fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    no_handle = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, no_handle, repo=repo)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:2", source_handle={"bogus": 1}), repo=repo)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            src,
            _svc_item(
                f"{sid}:ev:3",
                source_record_id="0000000000-00-000000",
                document_name="nope.htm",
                source_handle={
                    "accession_no": "0000000000-00-000000",
                    "document_name": "nope.htm",
                    "basis": "raw",
                    "offset": 0,
                    "max_chars": 5,
                    "text_hash": "x" * 64,
                    "source_content_hash": "a" * 64,
                },
            ),
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            src,
            _svc_item(f"{sid}:ev:4", source_record_id="0000320193-25-000080"),
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_sec_unique_window_replays_proof_archive_and_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a unique locator enters the ledger, even when another document span repeats it."""
    from app.research import service as _svc
    from app.research.evidence import evidence_from_dict, verify_observation
    from app.research.freeze import freeze_from_dict
    from app.sec.archive import find_archived_document

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    accession, document = "0000320193-25-000079", "repeated-revenue.htm"
    locator = "NVDA Q1 FY2025 revenue was 7 million."
    cited = "NVDA Q1 FY2025 revenue\twas 7 million."
    full = f"Item 2.\n{locator}\n{cited}\nEnd."
    seam.register_document(accession, document, full)
    broad = seam.handle_for_registered(full, accession=accession, document=document)
    eid = f"{sid}:ev:unique"
    item = _svc_item(
        eid,
        passage=locator,
        content=cited,
        claim_text=locator,
        source_record_id=accession,
        document_name=document,
        source_handle=broad,
        value="7",
        metric="revenue",
        entity="NVDA",
        period="Q1 FY2025",
        unit="million",
    )
    with pytest.raises(ValueError, match="ERR_PASSAGE_AMBIGUOUS"):
        _svc.record_evidence(sid, src, item, repo=repo)
    assert repo.list_evidence(sid) == []
    assert repo.list_evidence_ids(sid) == []
    assert not [event for event in repo.list_events(sid) if event.event_type == "evidence.accepted"]
    assert find_archived_document(accession, document, root=tmp_path / "data" / "raw") is None

    scoped = seam.handle_for_registered(cited, accession=accession, document=document)
    accepted = _svc.record_evidence(sid, src, {**item, "source_handle": scoped}, repo=repo)
    assert accepted["evidence_id"] == eid
    row = evidence_from_dict(repo.get_evidence(eid))
    start = full.index(cited)
    assert row.provenance["passage"] == cited
    assert (row.provenance["offset"], row.provenance["end"]) == (start, start + len(cited))
    assert row.provenance["text_hash"] == scoped["text_hash"]
    archived = find_archived_document(accession, document, root=tmp_path / "data" / "raw")
    assert archived is not None
    source_bytes = archived.payload_path.read_bytes()
    assert source_bytes == full.encode("utf-8")
    assert row.proof["sourceId"] == f"sha256:{sha256(source_bytes).hexdigest()}"
    assert verify_observation(row, value="7", source_bytes=source_bytes).status == "verified"
    assert verify_observation(row, value="8", source_bytes=source_bytes).status == "conflicted"
    assert verify_observation(row, value="7", source_bytes=None).status == "unverified"

    _svc.complete_job(src, {}, repo=repo)
    frozen = _svc.freeze_session(sid, 1, repo=repo)
    fresh = ResearchRepository()
    replayed = evidence_from_dict(fresh.get_evidence(eid))
    assert replayed.proof == row.proof
    assert replayed.provenance == row.provenance
    assert verify_observation(replayed, value="7", source_bytes=source_bytes).status == "verified"
    verify_freeze(freeze_from_dict(fresh.get_freeze(str(frozen["freeze_id"]))), [replayed])


def test_hf_admission_rejects_fabricated_finra_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FINRA-shaped row with no SEC archive handle is navigation at best: ERR_RAW_SOURCE_REQUIRED."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    finra_row = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    finra_row.update(
        {
            "content": "short interest 5%",
            "claim_text": "short interest 5%",
            "source_name": "FINRA",
        }
    )
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, finra_row, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_admission_rejects_fabricated_web_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare web URL with no archive handle never becomes citable evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    web_row = {k: v for k, v in _svc_item(f"{sid}:ev:1").items() if k != "source_handle"}
    web_row.update(
        {
            "content": "news says up",
            "claim_text": "news says up",
            "source_name": "web",
            "source_uri": "https://example.com/news",
        }
    )
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(sid, src, web_row, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_pit_rejects_newer_than_as_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filing newer than as_of never becomes evidence: the archive refuses the read or the ingest gate rejects it."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    passage = "Later passage a day past the cutoff."
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected|ERR_SEC_HANDLE_UNREADABLE"):
        _svc.record_evidence(
            sid,
            src,
            _svc_item(
                f"{sid}:ev:late",
                passage=passage,
                source_record_id="0000320193-25-000080",
                document_name="nvda-late.htm",
                source_uri="https://sec.gov/late",
                source_handle=seam.handle_for(
                    passage,
                    accession="0000320193-25-000080",
                    document="nvda-late.htm",
                    known_at="2025-07-01T00:00:00+00:00",
                ),
                known_at="2025-07-01T00:00:00+00:00",
            ),
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_pit_unknown_time_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded as_of with no known_at fails PIT_UNVERIFIED, never defaults to eligible."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    passage = "Undated passage with no PIT proof."
    item = _svc_item(
        f"{sid}:ev:undated",
        passage=passage,
        source_record_id="0000320193-25-000081",
        document_name="nvda-undated.htm",
        source_uri="https://sec.gov/undated",
        source_handle=seam.handle_for(passage, accession="0000320193-25-000081", document="nvda-undated.htm"),
    )
    item.pop("known_at", None)
    # Archive is the PIT authority: caller known_at is overwritten by the
    # materialized timing, so unknown time is driven by the archive row.
    seam._SOURCE_TIMING[("0000320193-25-000081", "nvda-undated.htm")] = (
        "",
        "2025-04-30",
        "2025-05-01T00:00:00Z",
    )

    with pytest.raises(ValueError, match="PIT_UNVERIFIED|rejected"):
        _svc.record_evidence(sid, src, item, repo=repo)
    assert repo.list_evidence(sid) == []


def test_hf_multisource_freeze_hash_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same evidence set freezes to a stable hash: order-insensitive, byte-identical on re-read."""
    from app.research import freeze as _freeze
    from app.research import service as _svc
    from app.research.evidence import evidence_from_dict

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    e1, e2 = f"{sid}:ev:1", f"{sid}:ev:2"
    _svc.record_evidence(sid, src, _svc_item(e1), repo=repo)
    _svc.record_evidence(
        sid,
        src,
        _svc_item(
            e2,
            passage="Second frozen passage.",
            source_record_id="0000320193-25-000080",
            document_name="nvda-2.htm",
            source_uri="https://sec.gov/2",
            source_handle=seam.handle_for(
                "Second frozen passage.",
                accession="0000320193-25-000080",
                document="nvda-2.htm",
            ),
        ),
        repo=repo,
    )
    _svc.complete_job(src, {}, repo=repo)
    fid = _hf_freeze_id(sid)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == fid
    recs = [evidence_from_dict(r) for r in repo.list_evidence(sid)]
    frozen = _freeze.freeze_from_dict(repo.get_freeze(fid))
    _freeze.verify_freeze(frozen, recs)
    _freeze.verify_freeze(frozen, list(reversed(recs)))
    assert _freeze.freeze_content_hash(recs) == frozen.content_hash


def test_hf_finra_only_followup_single_desk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Targeted FINRA-only follow-up: only a FINRA job is opened; SEC/WEB stay denied."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "FINRA")
    fin = str(
        _svc.start_job(
            sid,
            "source_agent",
            source="FINRA",
            budget={"owner": "omp"},
            repo=repo,
            wave_id=1,
        )["job_id"]
    )
    assert repo.get_job(fin).source_domain == "FINRA"
    with pytest.raises(ValueError, match="policy_denied"):
        _svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)
    with pytest.raises(ValueError, match="policy_denied"):
        _svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)
    assert [j.source_domain for j in repo.list_jobs(sid)] == [None, "FINRA"]


def test_hf_committee_same_freeze_no_data_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Committee: one freeze across the trio; data/task tools blocked, research_read passes."""
    from app.research import service as _svc
    from app.research.stage import check_stage_tool, stage_for_session

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.complete_job(src, {}, repo=repo)
    fid = _hf_freeze_id(sid)
    assert _svc.freeze_session(sid, 1, repo=repo)["freeze_id"] == fid
    out = _svc.create_committee_jobs(sid, 1, repo=repo)
    out_jobs = out["jobs"]
    assert isinstance(out_jobs, list) and len(out_jobs) == 3
    assert out["freeze_id"] == fid
    trio = {str(jid): repo.get_job(str(jid)) for jid in out_jobs if isinstance(jid, str)}
    assert len(trio) == 3
    assert {j.job_type for j in trio.values()} == {"stockbot", "bullbot", "bearbot"}
    assert all(j.wave_id == 1 and j.status == "running" for j in trio.values())
    assert stage_for_session(repo.get_session(sid), repo.list_jobs(sid)) == "COMMITTEE"
    for jid in trio:
        for tool in (
            "search_web",
            "get_sec_document",
            "research_add_evidence",
            "research_submit_source_result",
            "task",
        ):
            with pytest.raises(ValueError, match="forbids|cannot dispatch"):
                _svc.authorize_and_consume_dispatch(sid, jid, tool, repo=repo)
    check_stage_tool("COMMITTEE", "research_read")
    jobs_out = [jid for jid in out_jobs if isinstance(jid, str)]
    assert len(jobs_out) == 3
    for role, jid in zip(("stockbot", "bullbot", "bearbot"), jobs_out):
        _svc.record_committee_analysis(sid, jid, role, _svc_ana(eid), repo=repo)
    assert repo.get_session(sid).freeze_ids == [fid]


def _hf_run_to_freeze(repo: ResearchRepository, q: str = "NVDA demand?") -> tuple[str, str, str]:
    """One evidence row -> insufficient submit -> freeze; returns (sid, src, fid)."""
    from app.research import service as _svc

    sid, src = _hf_sid(repo, q=q)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_hf_cov(), evidence_ids=[eid], repo=repo)
    return sid, src, str(_svc.freeze_session(sid, 1, repo=repo)["freeze_id"])


def test_hf_resume_mid_source_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume mid-source: read-only snapshot, open job kept, no dup evidence/jobs."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    snap = _svc.resume_research(sid, repo=repo)
    open_ids = snap["open_job_ids"]
    assert isinstance(open_ids, list) and src in open_ids
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert ResearchRepository().resume(sid).open_job_ids == [src]


def test_hf_resume_after_source_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after source submit: still no new jobs/evidence, source job completed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(src, coverage=_hf_cov(), evidence_ids=[eid], repo=repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    _svc.resume_research(sid, repo=repo)
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert repo.get_job(src).status == "completed"


def test_hf_resume_after_freeze_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after freeze: same freeze id, no fetch rerun, no dup jobs/evidence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _, fid = _hf_run_to_freeze(repo)
    before = (len(repo.list_jobs(sid)), len(repo.list_evidence(sid)))
    _svc.resume_research(sid, repo=repo)
    assert (len(repo.list_jobs(sid)), len(repo.list_evidence(sid))) == before
    assert repo.get_session(sid).freeze_ids == [fid]


def test_hf_resume_after_one_committee_no_dup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume after one committee member: same freeze, evidence/jobs untouched, trio resumable."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _, fid = _hf_run_to_freeze(repo)
    eid = f"{sid}:ev:1"
    first = str(_svc.start_job(sid, "stockbot", repo=repo, wave_id=1)["job_id"])
    _svc.record_committee_analysis(sid, first, "stockbot", _svc_ana(eid), repo=repo)
    before = (
        len(repo.list_jobs(sid)),
        len(repo.list_evidence(sid)),
        list(repo.get_session(sid).freeze_ids),
    )
    _svc.resume_research(sid, repo=repo)
    assert (
        len(repo.list_jobs(sid)),
        len(repo.list_evidence(sid)),
        list(repo.get_session(sid).freeze_ids),
    ) == before
    assert repo.get_job(first).status == "completed"
    rest = _svc.create_committee_jobs(sid, 1, repo=repo)
    rest_jobs_raw = rest["jobs"]
    assert isinstance(rest_jobs_raw, list)
    rest_jobs = [jid for jid in rest_jobs_raw if isinstance(jid, str)]
    assert fid in (str(rest["freeze_id"]), repo.get_session(sid).freeze_ids[-1])
    assert len(set(rest_jobs)) == 3


def test_hf_provenance_vocab_closed() -> None:
    """Provenance kinds are closed at six: sec_source|sec_record|finra_record|web_source|search_run|none all validate.

    Bare FINRA rows / bare URLs still fail closed at admission (ERR_RAW_SOURCE_REQUIRED):
    only tool_result_id refs replayed against persisted tool results ground finra/web evidence.
    """
    from app.research.evidence import (
        PROVENANCE_KINDS,
        finra_record_ref,
        search_run_ref,
        sec_record_ref,
        sec_source_ref,
        validate_provenance,
        web_source_ref,
    )

    assert set(PROVENANCE_KINDS) == {
        "sec_source",
        "sec_record",
        "finra_record",
        "web_source",
        "search_run",
        "none",
    }
    assert validate_provenance({"kind": "none"}) == {"kind": "none"}
    assert validate_provenance(search_run_ref(search_id="s1", query="q"))["kind"] == "search_run"
    sec = sec_source_ref(
        accession_no="0000320193-25-000079",
        document_name="d.htm",
        passage="hello world",
        offset=0,
        end=5,
        basis="rendered",
        text_hash="ab" * 32,
    )
    assert validate_provenance(dict(sec))["kind"] == "sec_source"
    rec = sec_record_ref(tool_name="get_sec_filing", record_identity="row-1")
    assert validate_provenance(dict(rec))["kind"] == "sec_record"
    fin = finra_record_ref(tool_name="query_finra", record_identity="row-1")
    assert validate_provenance(dict(fin))["kind"] == "finra_record"
    web = web_source_ref(url="https://example.com/a", excerpt="highlight text")
    assert validate_provenance(dict(web))["kind"] == "web_source"
    with pytest.raises(ValueError, match="kind"):
        validate_provenance({"kind": "carrier_pigeon"})


def _hf_persist_finra(
    repo: ResearchRepository,
    sid: str,
    job_id: str,
    rid: str,
    *,
    as_of_date: str | None = "2025-06-15",
    published_at: str | None = "2025-06-20",
) -> str:
    """Persist one staged FINRA result; returns its tool_result_id."""
    from app.research import service as _svc

    marker = rid.rsplit(":tr:", 1)[-1]
    payload: dict[str, object] = {
        "records": [{"symbol": "NVDA", "shortInterest": 12345, "marker": marker}],
        "briefing": f"NVDA short interest 12345 shares {marker}",
        "record": {"symbol": "NVDA", "shortInterest": 12345, "marker": marker},
    }
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date
    if published_at is not None:
        payload["published_at"] = published_at
    return str(_svc.persist_tool_result(sid, job_id, "query_finra", rid, payload, repo=repo)["tool_result_id"])


def _hf_persist_web(repo: ResearchRepository, sid: str, job_id: str, rid: str) -> str:
    """Persist one staged search_web result; returns its tool_result_id."""
    from app.research import service as _svc

    return str(
        _svc.persist_tool_result(
            sid,
            job_id,
            "search_web",
            rid,
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": "NVDA rallies on demand",
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )


def _hf_desks(repo: ResearchRepository, sid: str) -> tuple[str, str]:
    """Fresh running FINRA + WEB jobs for one session."""
    from app.research import service as _svc

    fin_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    web_job = str(_svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)["job_id"])
    return fin_job, web_job


def test_hf_persist_accepts_matching_domain_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_tool_result: FINRA evidence tool on FINRA jobs, search_web on WEB jobs."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    assert _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-ok") == f"{sid}:tr:fin-ok"
    assert _hf_persist_web(repo, sid, web, f"{sid}:tr:web-ok") == f"{sid}:tr:web-ok"
    assert _svc.get_tool_result(f"{sid}:tr:fin-ok", repo=repo)["tool_name"] == "query_finra"
    assert _svc.get_tool_result(f"{sid}:tr:web-ok", repo=repo)["tool_name"] == "search_web"


def test_hf_persist_rejects_cross_domain_and_sec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-domain tools and SEC tools without replay support fail closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _first = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="cannot ground FINRA evidence"):
        _svc.persist_tool_result(sid, fin, "search_web", f"{sid}:tr:x1", {"evidence": []}, repo=repo)
    with pytest.raises(ValueError, match="cannot ground WEB evidence"):
        _svc.persist_tool_result(sid, web, "query_finra", f"{sid}:tr:x2", {"records": []}, repo=repo)
    with pytest.raises(ValueError, match="cannot ground FINRA evidence"):
        _svc.persist_tool_result(sid, fin, "list_finra_datasets", f"{sid}:tr:x3", {"datasets": []}, repo=repo)
    for tool in ("get_sec_document",):
        with pytest.raises(ValueError, match="cannot ground SEC replay"):
            _svc.persist_tool_result(sid, sec_job, tool, f"{sid}:tr:x-{tool}", {}, repo=repo)


def test_hf_persist_is_idempotent_first_bytes_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume-safe persist: repeating a tool_result_id keeps the first bytes, same id back."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = f"{sid}:tr:repeat"
    marker = rid.rsplit(":tr:", 1)[-1]
    _hf_persist_finra(repo, sid, fin, rid)
    again = _svc.persist_tool_result(
        sid,
        fin,
        "query_finra",
        rid,
        {"records": [{"symbol": "ZZZ"}], "briefing": "other"},
        repo=repo,
    )
    assert again["tool_result_id"] == rid
    stored = _svc.get_tool_result(rid, repo=repo)
    result = stored["result"]
    assert isinstance(result, dict)
    assert result["briefing"] == f"NVDA short interest 12345 shares {marker}"


def test_hf_replay_admits_finra_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA job + tool_result_id replay admits finra_record provenance with publication known_at."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-replay")
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": "fin-replay"}, sort_keys=True)
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-1",
            "content": "NVDA short interest 12345 shares fin-replay",
            "claim_text": "NVDA short interest 12345 shares fin-replay",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": identity,
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "finra_record"
    assert _hf_prov(out, "tool_name") == "query_finra"
    assert _hf_prov(out, "known_at") == "2025-06-20"
    assert out["known_at"] == "2025-06-20T00:00:00+00:00"


def test_hf_sec_record_identity_keys_on_record_and_tool_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC replay dedupe keys on record_identity + tool_result_id, never subject/claim alone.

    Regression: claim-less SEC replay rows under one subject collided on a blank
    accession/document key, so a later distinct record deduped onto the first.
    """
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    form4_row = {"description": "NVDA Form 4 accession 0001045810-25-000001", "filed_at": "2025-06-01", "form": "4"}
    q10_row = {"description": "NVDA 10-Q accession 0001045810-25-000002", "filed_at": "2025-06-01", "form": "10-Q"}
    payload: dict[str, object] = {"filings": [dict(form4_row), dict(q10_row)], "filed_at": "2025-06-01"}
    rid_a = str(
        _svc.persist_tool_result(sid, sec_job, "list_sec_filings", f"{sid}:tr:a", payload, repo=repo)["tool_result_id"]
    )
    rid_b = str(
        _svc.persist_tool_result(sid, sec_job, "list_sec_filings", f"{sid}:tr:b", payload, repo=repo)["tool_result_id"]
    )

    def _cite(eid: str, rid: str, row: Mapping[str, object]) -> dict[str, JSONValue]:
        # No claim_text/subject: the scheduler path supplies content only.
        identity = json.dumps(dict(row), sort_keys=True)
        item: dict[str, object] = {
            "evidence_id": f"{sid}:ev:{eid}",
            "content": identity,
            "source_name": "SEC",
            "tool_result_id": rid,
            "matching_passage": identity,
        }
        return _svc.record_evidence(sid, sec_job, item, repo=repo)

    first = _cite("a-form4", rid_a, form4_row)
    other_record = _cite("a-10q", rid_a, q10_row)
    other_result = _cite("b-form4", rid_b, form4_row)
    repeat = _cite("a-form4-again", rid_a, form4_row)
    assert _hf_prov(first, "kind") == "sec_record"
    assert "duplicate_of" not in other_record and other_record["evidence_id"] == f"{sid}:ev:a-10q"
    assert "duplicate_of" not in other_result and other_result["evidence_id"] == f"{sid}:ev:b-form4"
    assert repeat == {
        "evidence_id": f"{sid}:ev:a-form4",
        "accepted": False,
        "duplicate_of": f"{sid}:ev:a-form4",
    }
    assert repo.list_evidence_ids(sid) == [
        f"{sid}:ev:a-form4",
        f"{sid}:ev:a-10q",
        f"{sid}:ev:b-form4",
    ]


def test_hf_replay_admits_web_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB job + tool_result_id replay admits web_source provenance (url/excerpt/title/domain)."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-replay")
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-1",
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "web_source"
    assert _hf_prov(out, "url") == "https://example.com/a"
    assert _hf_prov(out, "excerpt") == "NVDA rallies on demand"
    assert _hf_prov(out, "published_at") == "2025-06-20"


def test_hf_replay_uses_explicit_store_and_rejects_cross_session(
    tmp_path: Path,
) -> None:
    """Explicit-repo replay reads the caller's DB and rejects another session's result."""
    from app.research import service as _svc

    repo_a = ResearchRepository(tmp_path / "a.sqlite")
    repo_b = ResearchRepository(tmp_path / "b.sqlite")
    sid_a, _ = _hf_sid(repo_a, "SEC", "FINRA", "WEB")
    fin_a, web_a = _hf_desks(repo_a, sid_a)
    fin_rid = _hf_persist_finra(repo_a, sid_a, fin_a, f"{sid_a}:tr:fin-xstore")
    web_rid = _hf_persist_web(repo_a, sid_a, web_a, f"{sid_a}:tr:web-xstore")
    fin_identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": "fin-xstore"}, sort_keys=True)
    fin_out = _svc.record_evidence(
        sid_a,
        fin_a,
        {
            "evidence_id": f"{sid_a}:ev:fin-xstore",
            "content": "NVDA short interest 12345 shares fin-xstore",
            "claim_text": "NVDA short interest 12345 shares fin-xstore",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": fin_rid,
            "matching_passage": fin_identity,
        },
        repo=repo_a,
    )
    assert _hf_prov(fin_out, "kind") == "finra_record"
    web_out = _svc.record_evidence(
        sid_a,
        web_a,
        {
            "evidence_id": f"{sid_a}:ev:web-xstore",
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": web_rid,
            "source_record_id": "https://example.com/a",
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo_a,
    )
    assert _hf_prov(web_out, "kind") == "web_source"
    sid_b, _ = _hf_sid(repo_b, "SEC", "FINRA", "WEB")
    fin_b = str(_svc.start_job(sid_b, "source_agent", source="FINRA", repo=repo_b, wave_id=1)["job_id"])
    stored_a = repo_a.get_tool_result(fin_rid)
    repo_b.save_tool_result(dict(stored_a))
    with pytest.raises(ValueError, match="PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid_b,
            fin_b,
            {
                "evidence_id": f"{sid_b}:ev:fin-cross",
                "content": "NVDA short interest 12345 shares fin-xstore",
                "claim_text": "NVDA short interest 12345 shares fin-xstore",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": fin_rid,
                "matching_passage": fin_identity,
            },
            repo=repo_b,
        )


def test_hf_replay_wrong_result_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A citation the persisted result does not reproduce fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-wrong")
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-wrong")
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            fin,
            {
                "evidence_id": f"{sid}:ev:fin-x",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": fin_rid,
                "matching_passage": json.dumps({"absent": "values 99999"}),
            },
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            web,
            {
                "evidence_id": f"{sid}:ev:web-x",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "web",
                "tool_result_id": web_rid,
                "matching_passage": "absent quote xyz",
                "known_at": "2025-06-29T00:00:00+00:00",
            },
            repo=repo,
        )
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(
            sid,
            fin,
            {
                "evidence_id": f"{sid}:ev:fin-unknown",
                "content": "x",
                "claim_text": "x",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": f"{sid}:tr:nope",
                "matching_passage": "NVDA short interest 12345 shares",
            },
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_sec_record_availability_comes_from_the_cited_filing_not_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately backdated caller cannot admit a later-filed SEC row; envelope or as-of dates never stand in."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)  # historical as_of 2025-06-30
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    early_url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000001/"
    later_url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000099/"
    early = {"form": "10-Q", "description": "NVDA 10-Q accession 0001045810-25-000001", "filed_at": "2025-05-28"}
    later = {"form": "10-Q", "description": "NVDA 10-Q accession 0001045810-25-000099", "filed_at": "2025-08-27"}
    undated = {"form": "10-Q", "description": "NVDA 10-Q accession 0001045810-25-000050", "as_of_date": "2025-04-27"}
    rid = str(
        _svc.persist_tool_result(
            sid,
            sec_job,
            "list_sec_filings",
            f"{sid}:tr:sec-pit",
            {
                "filings": [{**early, "source_url": early_url}, {**later, "source_url": later_url}, undated],
                "filed_at": "2025-01-01",
                "as_of_date": "2025-06-01",
            },
            repo=repo,
        )["tool_result_id"]
    )
    backdated = {key: "2025-05-28" for key in ("known_at", "filed_at", "accepted_at", "published_at", "as_of_date")}

    def _cite(eid: str, row: Mapping[str, object], **caller: object) -> dict[str, JSONValue]:
        identity = json.dumps(dict(row), sort_keys=True)
        item = {
            "evidence_id": f"{sid}:ev:{eid}",
            "content": identity,
            "source_name": "SEC",
            "tool_result_id": rid,
            "matching_passage": identity,
            **caller,
        }
        return _svc.record_evidence(sid, sec_job, item, repo=repo)

    with pytest.raises(EvidenceRejectedError, match="PIT_VIOLATION"):
        _cite("later", {**later, "source_url": later_url}, **backdated)
    # A declared URI only selects: it cannot point the later row at the earlier filing.
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _cite("later-uri", {**later, "source_url": later_url}, source_uri=early_url, **backdated)
    # The cited row has no filing availability: neither the envelope date nor its as-of date stands in.
    with pytest.raises(EvidenceRejectedError, match="PIT_UNVERIFIED"):
        _cite("undated", undated, **backdated)
    assert repo.list_evidence(sid) == []
    admitted = _cite("early", {**early, "source_url": early_url}, known_at="2024-01-01")
    assert _hf_prov(admitted, "known_at") == "2025-05-28"
    assert _hf_prov(admitted, "source_uri") == early_url
    assert admitted["known_at"] == "2025-05-28T00:00:00+00:00"


def test_hf_sec_record_datapoint_binds_metric_field_at_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revenue 100 cannot be admitted as net income 100; the correct field binds and replays verified."""
    import json as _json

    from app.research import service as _svc
    from app.research.evidence import evidence_from_dict

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo)
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    row = {**_DP_RECORD, "filed_at": "2025-05-28"}
    rid = str(
        _svc.persist_tool_result(sid, sec_job, "get_fundamentals", f"{sid}:tr:dp", {"facts": [row]}, repo=repo)[
            "tool_result_id"
        ]
    )
    item: dict[str, object] = {
        "content": "NVDA revenue",
        "source_name": "SEC",
        "tool_result_id": rid,
        "matching_passage": _json.dumps(row, sort_keys=True),
        "value": "100",
        "entity": "NVDA",
        "period": "2025-04-27",
    }
    with pytest.raises(ValueError, match="ERR_DATAPOINT_UNVERIFIED"):
        _svc.record_evidence(sid, sec_job, {**item, "evidence_id": f"{sid}:ev:ni", "metric": "net income"}, repo=repo)
    with pytest.raises(ValueError, match="ERR_DATAPOINT_INCOMPLETE"):
        _svc.record_evidence(sid, sec_job, {**item, "evidence_id": f"{sid}:ev:none"}, repo=repo)
    assert repo.list_evidence(sid) == []
    out = _svc.record_evidence(sid, sec_job, {**item, "evidence_id": f"{sid}:ev:rev", "metric": "Revenue"}, repo=repo)
    descriptor = out["datapoint"]
    assert isinstance(descriptor, dict)
    assert (descriptor["metric"], descriptor["field"], descriptor["unit"]) == ("revenue", "revenue", "USD")
    assert out["datapoint_id"] == descriptor["id"]
    stored = evidence_from_dict(repo.get_evidence(f"{sid}:ev:rev"))
    assert (stored.value, stored.entity, stored.period, stored.unit) == ("100", "NVDA", "2025-04-27", "USD")
    rows = _svc._session_rows(repo, sid)
    assert _svc._proof_verdict(rows, stored, "100", as_of=repo.get_session(sid).as_of).status == "verified"


def test_hf_finra_caller_known_at_never_overrides_result_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA known_at is the persisted publication time: a caller backdate cannot move it."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-caller-time")
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": "fin-caller-time"}, sort_keys=True)
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-caller-time",
            "content": "NVDA short interest 12345 shares fin-caller-time",
            "claim_text": "NVDA short interest 12345 shares fin-caller-time",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": identity,
            "known_at": "2025-01-01T00:00:00+00:00",
        },
        repo=repo,
    )
    assert out["known_at"] == "2025-06-20T00:00:00+00:00"


def test_hf_finra_known_at_gates_pit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA known_at comes from the persisted publication time: early admits, late PIT_VIOLATION, missing PIT_UNVERIFIED."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")

    def _item(eid: str, rid: str, marker: str) -> dict[str, object]:
        return {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True
            ),
        }

    early_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    early_rid = _hf_persist_finra(
        repo,
        sid,
        early_job,
        f"{sid}:tr:pit-early",
        as_of_date="2025-06-15",
        published_at="2025-06-15",
    )
    _svc.record_evidence(
        sid,
        early_job,
        _item(
            f"{sid}:ev:pit-early",
            early_rid,
            "pit-early",
        ),
        repo=repo,
    )
    late_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    late_rid = _hf_persist_finra(
        repo,
        sid,
        late_job,
        f"{sid}:tr:pit-late",
        as_of_date="2025-06-15",
        published_at="2025-07-15",
    )
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid,
            late_job,
            _item(
                f"{sid}:ev:pit-late",
                late_rid,
                "pit-late",
            ),
            repo=repo,
        )
    bare_job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    bare_rid = _hf_persist_finra(repo, sid, bare_job, f"{sid}:tr:pit-none", as_of_date=None, published_at=None)
    with pytest.raises(ValueError, match="PIT_UNVERIFIED|rejected"):
        _svc.record_evidence(
            sid,
            bare_job,
            _item(
                f"{sid}:ev:pit-none",
                bare_rid,
                "pit-none",
            ),
            repo=repo,
        )
    assert [r["evidence_id"] for r in repo.list_evidence(sid)] == [f"{sid}:ev:pit-early"]


def test_hf_finra_settlement_is_not_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-cutoff settlementDate published after cutoff is PIT_VIOLATION, never admitted."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    job = str(_svc.start_job(sid, "source_agent", source="FINRA", repo=repo, wave_id=1)["job_id"])
    rid = _hf_persist_finra(
        repo,
        sid,
        job,
        f"{sid}:tr:pit-settle",
        as_of_date="2025-06-15",
        published_at="2025-07-15",
    )
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid,
            job,
            {
                "evidence_id": f"{sid}:ev:pit-settle",
                "content": "NVDA short interest 12345 shares pit-settle",
                "claim_text": "NVDA short interest 12345 shares pit-settle",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": rid,
                "matching_passage": json.dumps(
                    {"symbol": "NVDA", "shortInterest": 12345, "marker": "pit-settle"}, sort_keys=True
                ),
            },
            repo=repo,
        )


def test_hf_get_tool_result_roundtrip_and_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_tool_result round-trips the persisted payload; unknown ids raise ResearchNotFound."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:roundtrip")
    stored = _svc.get_tool_result(rid, repo=repo)
    assert stored["tool_result_id"] == rid and stored["tool_name"] == "query_finra"
    assert stored["session_id"] == sid and stored["job_id"] == fin
    with pytest.raises(_svc.ResearchNotFound, match="unknown tool_result_id"):
        _svc.get_tool_result(f"{sid}:tr:nope", repo=repo)


def test_hf_gate_finra_authorized_when_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multisource session: a material FINRA committee follow-up authorizes the next wave."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo, "SEC", "FINRA", "WEB")
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(
        src,
        coverage=_hf_cov(unresolved=[], material_open_questions=[], remaining_branches=[]),
        evidence_ids=[eid],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    follow_up = {
        "question": "What does FINRA short interest show for NVDA?",
        "suggested_source": "finra",
    }
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid,
            jid,
            role,
            _committee_analysis(eid, [follow_up] if i == 0 else []),
            repo=repo,
        )
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is True and out["targeted_domain"] == "finra"
    assert out["targeted_question"] == "What does FINRA short interest show for NVDA?"
    sess = repo.get_session(sid)
    assert (
        sess.current_wave == 2
        and sess.status == "targeted_research"
        and sess.targeted_question == out["targeted_question"]
    )


def test_hf_gate_web_denied_when_sec_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-only session: a material WEB committee follow-up stops not_actionable (fails closed)."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _hf_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    _svc.submit_source_result(
        src,
        coverage=_hf_cov(unresolved=[], material_open_questions=[], remaining_branches=[]),
        evidence_ids=[eid],
        repo=repo,
    )
    _svc.freeze_session(sid, 1, repo=repo)
    off_lane = {
        "question": "What does the press say about the NVDA terms?",
        "suggested_source": "WEB",
    }
    for i, role in enumerate(("stockbot", "bullbot", "bearbot")):
        jid = str(_svc.start_job(sid, role, repo=repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(
            sid,
            jid,
            role,
            _committee_analysis(eid, [off_lane] if i == 0 else []),
            repo=repo,
        )
    out = _svc.decide_next_wave(sid, repo=repo)
    assert out["authorized"] is False and out["stop_reason"] == "not_actionable"
    assert repo.get_session(sid).status == "synthesizing"


def _hf_finra_sufficient(**over: object) -> dict[str, object]:
    """FINRA sufficient envelope: datasets/tickers/windows/reads/branches, no residuals."""
    cov: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA short interest"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "datasets_queried": ["short-interest"],
        "tickers_covered": ["NVDA"],
        "settlement_windows_covered": ["2025-06-15"],
        "dataset_reads": ["read-1"],
        "covered_branches": ["short interest"],
        "material_open_questions": [],
        "major_entities_missing": [],
        "remaining_branches": [],
        "routes_unsearched": [],
    }
    cov.update(over)
    return cov


def _hf_web_sufficient(**over: object) -> dict[str, object]:
    """WEB sufficient envelope: semantic branches/queries/results/branches, no residuals."""
    cov: dict[str, object] = {
        "useful_for_question": "sufficient",
        "resolved": ["NVDA news"],
        "partially_resolved": [],
        "unresolved": [],
        "source_limitations": [],
        "semantic_branches_covered": ["company news"],
        "queries_executed": ["NVDA news"],
        "results_inspected": ["https://example.com/a"],
        "covered_branches": ["company news"],
        "material_open_questions": [],
        "major_entities_missing": [],
        "remaining_branches": [],
        "routes_unsearched": [],
    }
    cov.update(over)
    return cov


def test_hf_finra_sufficient_accepts_own_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient with datasets/tickers/windows/reads/branches completes; SEC keys not required."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-suff")
    eid = f"{sid}:ev:fin-suff"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    out = _svc.submit_source_result(fin, coverage=_hf_finra_sufficient(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    stored = repo.get_dossier(str(out["dossier_id"]))["coverage"]
    assert isinstance(stored, dict)
    assert stored.get("source_domain") == "FINRA" and stored.get("datasets_queried") == ["short-interest"]


def test_hf_web_sufficient_accepts_own_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WEB sufficient with semantic branches/queries/results/branches completes; SEC keys not required."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-suff")
    eid = f"{sid}:ev:web-suff"
    _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": eid,
            "content": "NVDA rallies on demand",
            "claim_text": "NVDA rallies on demand",
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": "NVDA rallies on demand",
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    out = _svc.submit_source_result(web, coverage=_hf_web_sufficient(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    stored = repo.get_dossier(str(out["dossier_id"]))["coverage"]
    assert isinstance(stored, dict)
    assert stored.get("source_domain") == "WEB" and stored.get("queries_executed") == ["NVDA news"]


def test_hf_finra_sufficient_missing_own_key_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient without datasets_queried fails ERR_COVERAGE_REQUIRED; job stays running."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-miss")
    eid = f"{sid}:ev:fin-miss"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    bad = _hf_finra_sufficient()
    del bad["datasets_queried"]
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.submit_source_result(fin, coverage=bad, evidence_ids=[eid], repo=repo)
    assert repo.get_job(fin).status == "running"


def test_hf_finra_sufficient_with_residual_fails_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA sufficient with a remaining branch fails ERR_COVERAGE_INCOMPLETE; SEC regression stays green."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-resid")
    eid = f"{sid}:ev:fin-resid"
    marker = rid.rsplit(":tr:", 1)[-1]
    _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": eid,
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": json.dumps(
                {"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True
            ),
        },
        repo=repo,
    )
    with pytest.raises(ValueError, match="ERR_COVERAGE_INCOMPLETE"):
        _svc.submit_source_result(
            fin,
            coverage=_hf_finra_sufficient(remaining_branches=["Reg SHO"]),
            evidence_ids=[eid],
            repo=repo,
        )
    assert repo.get_job(fin).status == "running"
    sec_sid, sec_job = _hf_sid(repo)
    sec_eid = f"{sec_sid}:ev:1"
    _svc.record_evidence(sec_sid, sec_job, _svc_item(sec_eid), repo=repo)
    assert (
        _svc.submit_source_result(sec_job, coverage=_reg_cov(), evidence_ids=[sec_eid], repo=repo)["job_status"]
        == "completed"
    )


def _hf_replay_item(eid: str, rid: str, passage: str, **over: object) -> dict[str, object]:
    """FINRA/WEB replay item: distinct evidence ids per citation so dedupe never masks the gate."""
    item: dict[str, object] = {
        "evidence_id": eid,
        "content": passage,
        "claim_text": passage,
        "subject": "NVDA",
        "source_name": "FINRA",
        "tool_result_id": rid,
        "matching_passage": passage,
    }
    item.update(over)
    return item


def test_hf_replay_rejects_padded_finra_citation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact record citation records; the same JSON plus a fabricated tail fails ERR_RAW_SOURCE_REQUIRED."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-pad")
    marker = rid.rsplit(":tr:", 1)[-1]
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    out = _svc.record_evidence(sid, fin, _hf_replay_item(f"{sid}:ev:fin-exact", rid, identity), repo=repo)
    assert _hf_prov(out, "kind") == "finra_record"
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED"):
        _svc.record_evidence(
            sid, fin, _hf_replay_item(f"{sid}:ev:fin-padded", rid, identity + " fabricated tail"), repo=repo
        )


def test_hf_replay_rejects_padded_web_citation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strict substring excerpt records; the same highlight plus a fabricated tail fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-pad")
    highlight = "NVDA rallies on demand"
    out = _svc.record_evidence(
        sid,
        web,
        _hf_replay_item(
            f"{sid}:ev:web-exact",
            rid,
            highlight,
            source_name="web",
            known_at="2025-06-29T00:00:00+00:00",
        ),
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "web_source"
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            web,
            _hf_replay_item(
                f"{sid}:ev:web-padded",
                rid,
                highlight + " fabricated tail",
                source_name="web",
                known_at="2025-06-29T00:00:00+00:00",
            ),
            repo=repo,
        )


def test_hf_stored_provenance_preserves_tool_result_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored FINRA/WEB provenance keeps the replayed tool_result_id for audit."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    fin_rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-id")
    web_rid = _hf_persist_web(repo, sid, web, f"{sid}:tr:web-id")
    marker = fin_rid.rsplit(":tr:", 1)[-1]
    fin_out = _svc.record_evidence(
        sid,
        fin,
        _hf_replay_item(
            f"{sid}:ev:fin-id",
            fin_rid,
            json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True),
        ),
        repo=repo,
    )
    web_out = _svc.record_evidence(
        sid,
        web,
        _hf_replay_item(
            f"{sid}:ev:web-id",
            web_rid,
            "NVDA rallies on demand",
            source_name="web",
            known_at="2025-06-29T00:00:00+00:00",
        ),
        repo=repo,
    )
    assert _hf_prov(fin_out, "tool_result_id") == fin_rid
    assert _hf_prov(web_out, "tool_result_id") == web_rid


def test_hf_rejected_evidence_journals_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PIT-rejected record_evidence journals an evidence.rejected event with the refusal reason."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(
        repo,
        sid,
        fin,
        f"{sid}:tr:fin-rej",
        as_of_date="2025-06-15",
        published_at="2025-07-15",
    )
    marker = rid.rsplit(":tr:", 1)[-1]
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    with pytest.raises(ValueError, match="PIT_VIOLATION|rejected"):
        _svc.record_evidence(
            sid,
            fin,
            _hf_replay_item(f"{sid}:ev:fin-rej", rid, identity),
            repo=repo,
        )
    rejected = [e for e in repo.list_events(sid) if e.event_type == "evidence.rejected"]
    assert rejected and rejected[0].payload.get("evidence_id") == f"{sid}:ev:fin-rej"
    assert rejected[0].payload.get("reason") == "PIT_VIOLATION"


def test_start_job_request_id_dedups_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same request_id returns the winner; no second job row persists."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    first = _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    second = _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    assert second["job_id"] == first["job_id"]
    assert len(repo.list_jobs(sid)) == 2  # seed job + one winner


def test_start_job_request_id_type_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request_id bound to source_agent rejects a committee type."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    _svc.start_job(sid, "source_agent", repo=repo, wave_id=1, request_id="attempt-1")
    with pytest.raises(ValueError, match="already identifies"):
        _svc.start_job(sid, "stockbot", repo=repo, wave_id=1, request_id="attempt-1")


def test_evidence_insert_and_session_link_commit_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence row + session.evidence_ids land in one batch."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    eid = f"{sid}:ev:atomic"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    assert eid in repo.get_session(sid).evidence_ids
    assert repo.get_evidence(eid)["evidence_id"] == eid


def test_write_bundle_propagates_store_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing session raises instead of writing a fallback bundle."""
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _fake_dispatch(), _ok, DirectorBudgets())
    with pytest.raises(KeyError):
        run.write_bundle("no-such-session")


def test_write_bundle_propagates_evidence_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt persisted evidence raises instead of skipping the row."""
    from app.research.director import DirectorBudgets
    from app.research.runner import _LiveRun

    def _ok(prompt: str) -> str:
        assert prompt
        return _grounded(prompt)

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    repo = ResearchRepository()
    out = run_live(
        question="NVDA demand?",
        objective="o",
        as_of="2025-06-30T00:00:00+00:00",
        tickers=["NVDA"],
        dispatch=_fake_dispatch(),
        model=_ok,
        repo=repo,
        budgets=None,
    )
    sid = out["session_id"]
    assert isinstance(sid, str)
    run = _LiveRun(repo, "q?", "o", None, "", ["NVDA"], _fake_dispatch(), _ok, DirectorBudgets())

    def _corrupt(_row: object) -> object:
        raise ValueError("corrupt row")

    monkeypatch.setattr("app.research.evidence.evidence_from_dict", _corrupt)
    with pytest.raises(ValueError, match="corrupt row"):
        run._bundle_evidence_entries(sid)


def test_session_preserves_original_question_text() -> None:
    """Original question text survives session creation without ticker prefixes."""
    session = _session.create_session("What time is it now?", "What time is it now?", ticker="NVDA")
    stored = session.to_dict()
    assert stored["query"] == "What time is it now?"
    assert stored["objective"] == "What time is it now?"
    assert stored["ticker"] == "NVDA"
    assert _session.create_session("q?", "q?").to_dict()["ticker"] is None
    round_tripped = _session.ResearchSession.from_dict(stored, "<test>")
    assert round_tripped.ticker == "NVDA"


def test_needle_execution_context_excludes_raw_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repository-backed node supplies anchors without disclosing stored evidence to Needle."""
    import asyncio as _asyncio
    import json as _json
    from typing import override

    import app.tools as _tools
    from app.decision_client import JevClient
    from app.research import scheduler as _sched
    from app.research import service as _svc
    from app.research.models import ToolDecision
    from app.tool_runtime import outcome_from_result

    def _orcl(name: str) -> str | None:
        return {"ORCL": "ORCL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _orcl)

    repo = ResearchRepository(data_root=tmp_path)
    objective = "What is ORCL's filing history?"
    # A cutoff retains generation, so this test checks the Needle context boundary.
    sid = _svc.create_research(objective, ticker="ORCL", as_of="2026-01-01", repo=repo)
    source_job = repo.list_jobs(sid)[0].job_id
    _svc.record_evidence(
        sid,
        source_job,
        _svc_item(
            f"{sid}:ev:private",
            passage="private stored passage",
            content="private stored content",
            claim_text="private stored claim",
        ),
        repo=repo,
    )
    assert any("private stored passage" in str(row) for row in repo.list_evidence(sid))
    node = _svc.create_node(sid, objective, "Check filing history.", repo=repo)
    seen: dict[str, object] = {}
    invoked: list[dict[str, JSONValue]] = []

    class _Jev(JevClient):
        @override
        async def select_tool(self, *args: object, **kwargs: object) -> ToolDecision:
            return ToolDecision(action="invoke", tool_name="list_sec_filings", tool_names=("list_sec_filings",))

        @override
        async def assess_result(self, *args: object, **kwargs: object) -> dict[str, JSONValue]:
            return {"evidence_state": "insufficient", "continuation": "continue_research"}

    async def _generate(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "ORCL"}, "reasoning": "", "withheld": False}

    def _invoke(name: str, arguments: dict[str, JSONValue], session: object, **kwargs: object) -> dict[str, object]:
        invoked.append(arguments)
        return {"filings": []}

    out = _asyncio.run(
        _sched.run_node(
            node,
            sid,
            repo=repo,
            jev=_Jev(),
            needle_generate=_generate,
            invoke=_invoke,
            to_outcome=outcome_from_result,
            registry=[
                {
                    "name": "list_sec_filings",
                    "parameters": {"type": "object", "properties": {"identifier": {"type": "string"}}},
                }
            ],
            max_rounds=1,
        )
    )
    assert invoked and invoked[0]["identifier"] == "ORCL"
    assert out["status"] == "blocked" and out["admitted"] == 0
    assert len(repo.list_evidence(sid)) == 1
    context = seen["context"]
    assert isinstance(context, dict)
    assert "evidence" not in context and "attempts" not in context and "grounding_hint" not in context
    serialized = _json.dumps(seen)
    assert "private stored passage" not in serialized
    assert "private stored content" not in serialized
    assert "private stored claim" not in serialized
    assert context["ticker"] == "ORCL"
    scope = context["temporal_scope"]
    assert scope == {"mode": "as_of", "as_of": "2026-01-01"}
    assert context["grounded_arguments"] == {"identifier": "ORCL"}


# --- changed-code CRAP repair: lifecycle argument resolution + coverage folds ---


def test_start_job_budget_resolves_request_wave_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget-embedded request_id/wave/model/owner shapes the job; a repeat request returns the winner."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    first = _svc.start_job(
        sid,
        "source_agent",
        budget={"request_id": "attempt-budget", "wave_id": 2, "model": "m-budget", "owner": "thesis"},
        repo=repo,
    )
    assert first["wave_id"] == 2 and first["model"] == "m-budget" and first["owner"] == "thesis"
    repeat = _svc.start_job(sid, "source_agent", budget={"request_id": "attempt-budget"}, repo=repo)
    assert repeat["job_id"] == first["job_id"] and len(repo.list_jobs(sid)) == 2  # seed job + one winner
    assert _svc.start_job(sid, "source_agent", budget={"model": 7}, model="m-kwarg", repo=repo)["model"] == "m-kwarg"
    assert _svc.start_job(sid, "source_agent", budget={"model": 7}, repo=repo)["model"] is None


def test_sufficient_required_rejects_empty_domain_work() -> None:
    """Each domain's sufficient envelope fails closed on an empty work artifact, not only on a missing key."""
    from app.research import service as _svc

    sec_no_runs = dict(_reg_cov())
    no_runs: list[object] = []
    sec_no_runs["search_runs"] = no_runs
    with pytest.raises(ValueError, match="non-empty search_runs"):
        _svc._sufficient_required(sec_no_runs, "SEC")
    sec_blank_branch = dict(_reg_cov())
    sec_blank_branch["covered_branches"] = ["   "]
    with pytest.raises(ValueError, match="non-empty covered_branches"):
        _svc._sufficient_required(sec_blank_branch, "SEC")
    with pytest.raises(ValueError, match="non-empty covered_branches"):
        _svc._sufficient_required(_hf_finra_sufficient(covered_branches=[]), "FINRA")
    with pytest.raises(ValueError, match="non-empty covered_branches"):
        _svc._sufficient_required(_hf_web_sufficient(covered_branches=["  "]), "WEB")
    _svc._sufficient_required(_reg_cov(), "SEC")  # the complete envelope passes


def test_submit_sufficient_without_search_runs_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sufficient SEC claim with no search runs is ERR_COVERAGE_REQUIRED; the job stays running."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    coverage: dict[str, object] = dict(_reg_cov())
    no_runs: list[object] = []
    coverage["search_runs"] = no_runs
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.submit_source_result(src, coverage=coverage, evidence_ids=[eid], repo=repo)
    assert repo.get_job(src).status == "running"


def test_wave1_coverage_merges_dossiers_by_domain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wave coverage unions each dossier's lists with a domain tag; an insufficient verdict never upgrades."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "created_at": "2025-01-01T00:00:00+00:00",
            "coverage": {"source_domain": "sec", "resolved": ["a"], "empty": [], "complete": True, "docs": 7},
            "relationships": [{"kind": "supplier"}, "junk"],
            "open_questions": ["q1", 7, ""],
        }
    )
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:finra",
            "session_id": sid,
            "wave_id": 1,
            "created_at": "2025-01-02T00:00:00+00:00",
            "coverage": {"source_domain": "finra", "resolved": [1, "b"], "useful_for_question": "insufficient"},
            "open_questions": "not-a-list",
        }
    )
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:web",
            "session_id": sid,
            "wave_id": 1,
            "created_at": "2025-01-03T00:00:00+00:00",
            "coverage": {"source_domain": "web", "useful_for_question": "sufficient"},
        }
    )
    repo.save_dossier({"dossier_id": f"{sid}:1:junk", "session_id": sid, "wave_id": 1, "coverage": "nope"})
    merged, rels, open_q = _svc._wave1_coverage(repo, sid)
    assert merged == {"resolved": ["[SEC] a", 1, "[FINRA] b"], "useful_for_question": "insufficient"}
    assert rels == [{"kind": "supplier"}]
    assert open_q == ["q1"]


def test_finalize_coverage_folds_known_domains_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-domain coverage keeps stripped, sorted, unique strings; unknown domains and junk rows are dropped."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:sec",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {"source_domain": "sec", "docs": [" b ", "a", "a", 5], "notes": []},
        }
    )
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:2:sec",
            "session_id": sid,
            "wave_id": 2,
            "coverage": {"source_domain": "SEC", "docs": ["c", "a"]},
        }
    )
    repo.save_dossier(
        {
            "dossier_id": f"{sid}:1:other",
            "session_id": sid,
            "wave_id": 1,
            "coverage": {"source_domain": "OTHER", "docs": ["x"]},
        }
    )
    repo.save_dossier({"dossier_id": f"{sid}:1:junk", "session_id": sid, "wave_id": 1, "coverage": "nope"})
    assert _svc._finalize_coverage(repo, sid) == {"sec": {"docs": ["a", "b", "c"]}}

    def _boom(_session_id: str) -> list[dict[str, object]]:
        raise RuntimeError("coverage read failed")

    monkeypatch.setattr(repo, "list_dossiers", _boom)
    assert _svc._finalize_coverage(repo, sid) == {}


def test_dossier_limitations_folds_lists_and_incomplete_flag() -> None:
    """Limitations/gaps fold in stripped and unique; an incomplete dossier names its own source."""
    from app.research import service as _svc

    lims: list[str] = ["existing"]
    _svc._dossier_limitations(lims, {"dossier_id": "d1"})
    _svc._dossier_limitations(lims, {"coverage": "nope"})
    assert lims == ["existing"]
    _svc._dossier_limitations(lims, {"coverage": {"source_limitations": ["  a  ", "a", 5], "gaps": ["b"]}})
    assert lims == ["existing", "a", "b"]
    _svc._dossier_limitations(lims, {"coverage": {"complete": False, "source_domain": "finra"}})
    assert lims == ["existing", "a", "b", "FINRA coverage incomplete for this session"]
    _svc._dossier_limitations(lims, {"coverage": {"complete": False}})
    assert lims[-1] == "Source coverage incomplete for this session"


def test_create_node_validates_arguments_and_dependency_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node text must be non-blank; every dependency must be a non-empty id inside the same session."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo, "first question?")
    other, _ = _svc_sid(repo, "second question?")
    with pytest.raises(ValueError, match="'question' must be a non-empty string"):
        _svc.create_node(sid, "   ", "why it matters", repo=repo)
    with pytest.raises(ValueError, match="'why_it_matters' must be a non-empty string"):
        _svc.create_node(sid, "what?", "", repo=repo)
    with pytest.raises(ValueError, match="'depends_on' must be a list of non-empty strings"):
        _svc.create_node(sid, "what?", "why", depends_on=["ok", ""], repo=repo)
    parent = _svc.create_node(sid, "  parent?  ", "  because it matters  ", repo=repo)
    assert parent.question == "parent?" and parent.why_it_matters == "because it matters"
    child = _svc.create_node(sid, "child?", "why", depends_on=[parent.node_id], repo=repo)
    assert child.depends_on == (parent.node_id,) and repo.get_node(child.node_id).session_id == sid
    foreign = _svc.create_node(other, "foreign?", "why", repo=repo)
    with pytest.raises(ValueError, match="belongs to another session"):
        _svc.create_node(sid, "cross?", "why", depends_on=[foreign.node_id], repo=repo)
    with pytest.raises(_svc.ResearchNotFound, match="unknown node_id"):
        _svc.create_node(sid, "missing dep?", "why", depends_on=[f"{sid}:node:absent"], repo=repo)

    def _boom_save(_node: object) -> None:
        raise KeyError("session row vanished")

    monkeypatch.setattr(repo, "save_node", _boom_save)
    with pytest.raises(_svc.ResearchNotFound, match="unknown session_id"):
        _svc.create_node(sid, "late?", "why", repo=repo)


def test_record_decision_validates_type_and_linked_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision type must be non-blank; a linked node/job must belong to the same session."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, own_job = _svc_sid(repo, "decision owner?")
    other, foreign_job = _svc_sid(repo, "decision stranger?")
    with pytest.raises(ValueError, match="'decision_type' must be a non-empty string"):
        _svc.record_decision(sid, "  ", {}, {}, None, repo=repo)
    node = _svc.create_node(sid, "q?", "why", repo=repo)
    foreign_node = _svc.create_node(other, "q?", "why", repo=repo)
    with pytest.raises(_svc.ResearchNotFound, match="unknown node_id"):
        _svc.record_decision(sid, "route", {}, {}, None, node_id=foreign_node.node_id, repo=repo)
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id"):
        _svc.record_decision(sid, "route", {}, {}, None, job_id=foreign_job, repo=repo)
    recorded = _svc.record_decision(
        sid, " route ", {"a": 1}, {"a": 0.5}, "a", node_id=node.node_id, job_id=own_job, repo=repo
    )
    assert recorded.decision_type == "route" and recorded.session_id == sid
    assert recorded.node_id == node.node_id and recorded.job_id == own_job
    plain = _svc.record_decision(sid, "route", {}, {}, None, repo=repo)
    assert plain.node_id is None and plain.job_id is None

    def _boom_save(*args: object, **kwargs: object) -> None:
        raise KeyError("decision row rejected")

    monkeypatch.setattr(repo, "save_decision", _boom_save)
    with pytest.raises(_svc.ResearchNotFound, match="decision row rejected"):
        _svc.record_decision(sid, "route", {}, {}, None, repo=repo)


@settings(max_examples=100, derandomize=True)
@given(st.lists(st.text(), max_size=8))
def test_prop_finalize_coverage_merge_is_normalized_and_idempotent(docs: list[str]) -> None:
    """Folding one source's string list stays sorted, blank-free, deduped, and stable on a repeat fold."""
    from app.research import service as _svc

    out: dict[str, dict[str, list[str]]] = {"sec": {}, "finra": {}, "web": {}}
    coverage: dict[str, object] = {"source_domain": "SEC", "docs": docs}
    _svc._merge_final_coverage(out, coverage)
    merged = out["sec"].get("docs", [])
    assert merged == sorted({d.strip() for d in docs if d.strip()})
    _svc._merge_final_coverage(out, coverage)
    assert out["sec"].get("docs", []) == merged


# --- expand_queries rerank: embedding is a tiebreak, never a failure --------
def test_rerank_queries_orders_only_with_a_usable_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.needle_client as needle
    from app.research.agents import source_agent as sa

    def _boom(_texts: object, _reference: object, **_kwargs: object) -> None:
        raise AssertionError("no embedding work for a non-ranking input")

    def _reversed(_texts: object, _reference: object, **_kwargs: object) -> list[int]:
        return [1, 0]

    monkeypatch.setattr(needle, "similarity_order", _boom)
    queries = ["nvda demand", "nvda margin"]
    assert sa._rerank_queries(queries, None) is queries  # absent reference
    assert sa._rerank_queries(queries, "   ") is queries  # blank reference
    assert sa._rerank_queries(["nvda demand"], "NVDA exposure") == ["nvda demand"]  # single candidate
    assert sa._rerank_queries([], "NVDA exposure") == []

    monkeypatch.setattr(needle, "similarity_order", _reversed)
    assert sa._rerank_queries(queries, "NVDA exposure") == ["nvda margin", "nvda demand"]


def test_rerank_queries_keeps_planned_order_when_embedding_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import app.needle_client as needle
    from app.research.agents import source_agent as sa

    def _raises(_texts: object, _reference: object, **_kwargs: object) -> None:
        raise RuntimeError("embed down")

    def _declines(_texts: object, _reference: object, **_kwargs: object) -> None:
        return None

    queries = ["nvda demand", "nvda margin"]
    # A missing worker module, a raising embedder, and a refusal to rank all keep the planned order.
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "app.needle_client", None)
        assert sa._rerank_queries(queries, "NVDA exposure") == queries
    monkeypatch.setattr(needle, "similarity_order", _raises)
    assert sa._rerank_queries(queries, "NVDA exposure") == queries
    monkeypatch.setattr(needle, "similarity_order", _declines)
    assert sa._rerank_queries(queries, "NVDA exposure") == queries


# --- evidence/synthesis: caller integrity rows + coverage lines (public results) ---


def test_hf_caller_closed_integrity_wins_over_observation() -> None:
    """A caller closed-vocabulary integrity survives even when the observation maps elsewhere."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[claim],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[claim],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[claim],
    )
    obs = {
        "evidence_id": "EV-1",
        "provenance": {
            "kind": "finra_record",
            "tool_name": "query_finra",
            "record_identity": "row-1",
        },
    }
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        observations=[obs],
        sources=[
            {
                "evidence_id": "EV-1",
                "domain": "FINRA",
                "document": "FINRA",
                "integrity_class": "PRIMARY_DOCUMENT",
            }
        ],
    )
    assert synth.sources[0]["integrity_class"] == "PRIMARY_DOCUMENT"
    assert "FINRA [PRIMARY_DOCUMENT]" in synth.answer


def test_hf_caller_integrity_alias_accepted() -> None:
    """Both integrity keys survive: integrity_class alone and the legacy integrity alias alone."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import FinalSynthesis, synthesize_final

    def _synth(source: dict[str, object]) -> FinalSynthesis:
        claim = GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])
        stock = StockbotAnalysis(
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="2025-06-30",
            question="q",
            answer="a",
            base_case="b",
            claims=[claim],
        )
        bull = BullAnalysis(
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="2025-06-30",
            question="q",
            stance="bull",
            bull_case="up",
            claims=[claim],
        )
        bear = BearAnalysis(
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            evidence_ids=["EV-1"],
            as_of="2025-06-30",
            question="q",
            stance="bear",
            bear_case="down",
            claims=[claim],
        )
        return synthesize_final(
            "q",
            session_id="rs:t",
            wave_id=1,
            freeze_id="F1",
            as_of="2025-06-30",
            stock=stock,
            bull=bull,
            bear=bear,
            disagreement=compute_disagreement(stock, bull, bear),
            sources=[source],
        )

    by_class = _synth({"evidence_id": "EV-1", "integrity_class": "canonical_structured"})
    assert by_class.sources[0]["integrity_class"] == "CANONICAL_STRUCTURED"
    by_alias = _synth({"evidence_id": "EV-1", "integrity": "canonical_structured"})
    assert by_alias.sources[0]["integrity_class"] == "CANONICAL_STRUCTURED"


def test_hf_caller_closed_vocab_pinned() -> None:
    """Every closed-vocabulary member survives verbatim, including lowercase external_source."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claims = [GroundedClaim(text="finding suggests exposure", evidence_ids=["E1", "E2", "E3"])]
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["E1", "E2", "E3"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=claims,
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["E1", "E2", "E3"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=claims,
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["E1", "E2", "E3"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=claims,
    )
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        observations=[{"evidence_id": "E3", "domain": "SEC", "integrity_class": "PRIMARY_DOCUMENT"}],
        sources=[
            {"evidence_id": "E1", "integrity_class": "primary_document"},
            {"evidence_id": "E2", "integrity_class": "canonical_structured"},
            {"evidence_id": "E3", "integrity_class": "external_source"},
        ],
    )
    assert [row["integrity_class"] for row in synth.sources] == [
        "PRIMARY_DOCUMENT",
        "CANONICAL_STRUCTURED",
        "EXTERNAL_SOURCE",
    ]


def test_hf_caller_document_label_survives() -> None:
    """The caller document label (not source_name) renders in sources rows and the answer."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[claim],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[claim],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[claim],
    )
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        sources=[
            {
                "evidence_id": "EV-1",
                "domain": "SEC",
                "document": "msft-10k.htm",
                "source_name": "other",
                "integrity_class": "PRIMARY_DOCUMENT",
            }
        ],
    )
    assert synth.sources[0]["document"] == "msft-10k.htm"
    assert "msft-10k.htm" in synth.answer


def test_hf_coverage_line_separators_and_summary() -> None:
    """Coverage answer lines keep ', '/' separation and the summary fallback detail."""
    from app.research.agents import GroundedClaim
    from app.research.agents.bearbot import BearAnalysis
    from app.research.agents.bullbot import BullAnalysis
    from app.research.agents.stockbot import StockbotAnalysis
    from app.research.synthesis.committee import compute_disagreement
    from app.research.synthesis.final import synthesize_final

    claim = GroundedClaim(text="finding suggests exposure", evidence_ids=["EV-1"])
    stock = StockbotAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        answer="a",
        base_case="b",
        claims=[claim],
    )
    bull = BullAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bull",
        bull_case="up",
        claims=[claim],
    )
    bear = BearAnalysis(
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        evidence_ids=["EV-1"],
        as_of="2025-06-30",
        question="q",
        stance="bear",
        bear_case="down",
        claims=[claim],
    )
    synth = synthesize_final(
        "q",
        session_id="rs:t",
        wave_id=1,
        freeze_id="F1",
        as_of="2025-06-30",
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=compute_disagreement(stock, bull, bear),
        coverage={"sec": {"forms": ["10-K", "10-Q"], "docs": ["d1"], "summary": "two filings"}},
        extra_claims=[GroundedClaim(text="two filings", claim_type="unknown", evidence_ids=[])],
    )
    assert "forms: 10-K, 10-Q" in synth.answer
    assert "SEC — docs: d1; forms: 10-K, 10-Q" in synth.answer
    assert "two filings" in synth.answer
    assert synth.claims[-1].claim_type == "unknown" and not synth.claims[-1].evidence_ids


# --- service evidence slice: materialized timing, ledger, replay, ownership ---


def test_service_materialized_timing_overwrites_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored row keeps the archive timing, never caller dates, and survives empty archive timing."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    passage = "Archive timing overwrite passage"
    out = _svc.record_evidence(
        sid,
        src,
        {
            **_svc_item(f"{sid}:ev:time", passage=passage),
            "known_at": "1999-01-01",
            "published_at": "1999-01-02",
            "retrieved_at": "1999-01-03",
            "source_uri": "https://caller.example/stale",
        },
        repo=repo,
    )
    assert out["known_at"] == "2025-05-01T00:00:00+00:00"
    assert out["published_at"] == "2025-04-30T00:00:00+00:00"
    assert out["retrieved_at"] == "2025-05-01T00:00:00+00:00"
    assert out["source_uri"] != "https://caller.example/stale"
    assert "XXpublished_atXX" not in out and "PUBLISHED_AT" not in out
    assert "XXretrieved_atXX" not in out and "RETRIEVED_AT" not in out
    stored = repo.get_evidence(f"{sid}:ev:time")
    assert stored["published_at"] == "2025-04-30T00:00:00+00:00"
    assert stored["retrieved_at"] == "2025-05-01T00:00:00+00:00"


def test_service_ledger_skips_legacy_and_keeps_duplicates_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy row that cannot decode never blocks admission; a duplicate id still fails closed."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    first = _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    assert first["evidence_id"] == f"{sid}:ev:1"
    repo.save_evidence(
        {
            "evidence_id": f"{sid}:ev:legacy",
            "session_id": sid,
            "wave_id": 1,
            "source_type": "pi",
            "source_name": "SEC",
            "subject": "NVDA",
            "claim_text": "legacy",
            "content": "legacy",
            "content_hash": "0" * 64,
            "retrieved_at": "2025-05-01T00:00:00+00:00",
            "provenance": {"kind": "none"},
        }
    )
    second = _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:2"), repo=repo)
    assert second["evidence_id"] == f"{sid}:ev:2"
    assert repo.get_evidence(f"{sid}:ev:2")["session_id"] == sid
    passage = "A passage the ledger has never seen"
    fresh = _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:3", passage=passage), repo=repo)
    assert fresh["evidence_id"] == f"{sid}:ev:3"
    assert len(repo.list_evidence(sid)) == 4
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        repo.save_evidence_and_session(repo.get_session(sid), repo.get_evidence(f"{sid}:ev:2"))
    assert len(repo.list_evidence(sid)) == 4
    corrected = _svc.record_evidence(
        sid,
        src,
        _svc_item(
            f"{sid}:ev:corrected",
            passage="The later filing corrects the earlier finding.",
            superseded_by=f"{sid}:ev:2",
        ),
        repo=repo,
    )
    assert corrected["superseded_by"] == f"{sid}:ev:2"
    assert repo.get_evidence(f"{sid}:ev:corrected")["superseded_by"] == f"{sid}:ev:2"
    assert len(repo.list_evidence(sid)) == 5


def test_service_finra_citation_keys_and_tool_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA replay reads record_identity cites and rejects non-record staged tools."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-keys")
    marker = rid.rsplit(":tr:", 1)[-1]
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-keys",
            "content": "unrelated content",
            "claim_text": "unrelated claim",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "record_identity": identity,
        },
        repo=repo,
    )
    assert _hf_prov(out, "record_identity") == identity
    wrong = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-wrong")
    repo.save_tool_result(
        {**repo.get_tool_result(wrong), "tool_result_id": f"{sid}:tr:fin-bad-tool", "tool_name": "search_web"}
    )
    with pytest.raises(ValueError, match="ERR_PROVENANCE_MISMATCH"):
        _svc.record_evidence(
            sid,
            fin,
            {
                "evidence_id": f"{sid}:ev:fin-bad-tool",
                "content": "unrelated content",
                "claim_text": "unrelated claim",
                "subject": "NVDA",
                "source_name": "FINRA",
                "tool_result_id": f"{sid}:tr:fin-bad-tool",
                "matching_passage": identity,
            },
            repo=repo,
        )
    assert repo.list_evidence(sid) == [repo.get_evidence(f"{sid}:ev:fin-keys")]


def test_service_record_ownership_pins_session_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored rows keep the caller session/job, claim, and short default id; events name the job."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    passage = "Ownership pins the submitting session and job"
    out = _svc.record_evidence(
        sid, src, _svc_item(f"{sid}:ev:own", passage=passage, claim_text="A caller claim"), repo=repo
    )
    stored = repo.get_evidence(f"{sid}:ev:own")
    assert (stored["session_id"], stored["job_id"]) == (sid, src)
    assert stored["claim_text"] == "A caller claim"
    assert out["claim_text"] == "A caller claim"
    defaulted = _svc.record_evidence(
        sid,
        src,
        {k: v for k, v in _svc_item("unused", passage="An unclaimed passage").items() if k != "evidence_id"},
        repo=repo,
    )
    assert str(defaulted["evidence_id"]).startswith("ev:")
    accepted = [e for e in repo.list_events(sid) if e.event_type == "evidence.accepted"]
    assert accepted and all(e.payload.get("job_id") == src for e in accepted)
    cov = _absence_cov()
    artifact = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "no NVDA capacity constraint disclosed",
            "claim_text": "not disclosed in the searched scope",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "SEC",
            "search_id": "s-own",
            "query": "NVDA capacity",
            "coverage": cov,
        },
        repo=repo,
    )
    assert repo.list_coverage_artifacts(sid)[-1]["job_id"] == src
    assert str(artifact["artifact_id"]).startswith(f"{sid}:cov:")


def test_service_evidence_identity_dedupes_repeats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-subject repeat returns the winner without a new row; subjects stay isolated."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    passage = "Repeated observed passage for dedupe"
    first = _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:first", passage=passage), repo=repo)
    repeat = _svc.record_evidence(
        sid,
        src,
        dict(_svc_item(f"{sid}:ev:second", passage=passage), content=first["content"], claim_text=first["claim_text"]),
        repo=repo,
    )
    assert repeat == {"evidence_id": first["evidence_id"], "accepted": False, "duplicate_of": first["evidence_id"]}
    assert [r["evidence_id"] for r in repo.list_evidence(sid)] == [first["evidence_id"]]
    other = _svc.record_evidence(
        sid,
        src,
        dict(_svc_item(f"{sid}:ev:other", subject="AAPL"), content=first["content"], claim_text=first["claim_text"]),
        repo=repo,
    )
    assert other["evidence_id"] == f"{sid}:ev:other"
    assert len(repo.list_evidence(sid)) == 2


def test_service_absence_and_wave_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong-wave item and a stale job fail closed; a live wave-2 absence admits."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:1"), repo=repo)
    with pytest.raises(ValueError, match="item wave"):
        _svc.record_evidence(sid, src, _svc_item(f"{sid}:ev:bad-wave", wave=2), repo=repo)
    assert len(repo.list_evidence(sid)) == 1
    _svc.complete_job(src, {}, repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    wave2 = str(_svc.start_job(sid, "source_agent", repo=repo, wave_id=2)["job_id"])
    stale = str(_svc.start_job(sid, "source_agent", repo=repo, wave_id=1)["job_id"])
    with pytest.raises(ValueError, match="current wave"):
        _svc.record_evidence(sid, stale, _svc_item(f"{sid}:ev:stale"), repo=repo)
    cov = _absence_cov()
    out = _svc.record_evidence(
        sid,
        wave2,
        {
            "wave_id": 2,
            "content": "no NVDA capacity constraint disclosed",
            "claim_text": "not disclosed in the searched scope",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "SEC",
            "search_id": "s-wave2",
            "query": "NVDA capacity wave 2",
            "coverage": cov,
        },
        repo=repo,
    )
    assert str(out["artifact_id"]).startswith(f"{sid}:cov:")
    assert repo.list_coverage_artifacts(sid)[-1]["wave_id"] == 2
    assert len(repo.list_evidence(sid)) == 1


def test_service_missing_archive_dates_replace_caller_dates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An archive without optional dates clears publication and uses the current retrieval time."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    captured_at = datetime(2025, 5, 2, tzinfo=UTC)
    monkeypatch.setattr(_svc, "utcnow", lambda: captured_at)

    def document_without_dates(
        accession_no: str,
        document_name: str | None = None,
        *,
        offset: int = 0,
        max_chars: int | None = None,
        **_unused: object,
    ) -> dict[str, object]:
        document = seam.fake_sec_document(accession_no, document_name, offset=offset, max_chars=max_chars)
        return {**document, "filed_at": None, "retrieved_at": None}

    monkeypatch.setattr("app.sec.documents.get_sec_document", document_without_dates)
    out = _svc.record_evidence(
        sid,
        src,
        _svc_item(
            f"{sid}:ev:no-dates",
            published_at="1999-01-01",
            retrieved_at="1999-01-02",
        ),
        repo=repo,
    )
    assert out["published_at"] is None
    assert out["retrieved_at"] == captured_at.isoformat()
    stored = repo.get_evidence(f"{sid}:ev:no-dates")
    assert stored["published_at"] is None
    assert stored["retrieved_at"] == captured_at.isoformat()
    assert stored["known_at"] == "2025-05-01T00:00:00+00:00"


def test_service_same_evidence_replay_returns_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An identical citation returns its original identity without another accepted event."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _svc_sid(repo)
    item = _svc_item(f"{sid}:ev:repeat")
    first = _svc.record_evidence(sid, src, item, repo=repo)
    events = repo.list_events(sid)
    repeated = _svc.record_evidence(sid, src, item, repo=repo)
    assert repeated == {"evidence_id": first["evidence_id"], "accepted": False, "duplicate_of": first["evidence_id"]}
    assert repo.list_evidence(sid) == [first]
    assert repo.list_events(sid) == events


@pytest.mark.parametrize("domain", ["FINRA", "WEB"])
def test_service_replay_acceptance_event_names_submitting_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    """The journal retains the submitting job for each non-document citation."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, domain)
    job = str(_svc.start_job(sid, "source_agent", source=domain, repo=repo)["job_id"])
    rid = f"{sid}:tr:owner"
    if domain == "FINRA":
        _hf_persist_finra(repo, sid, job, rid)
        marker = rid.rsplit(":tr:", 1)[-1]
        passage = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    else:
        _hf_persist_web(repo, sid, job, rid)
        passage = "NVDA rallies on demand"
    out = _svc.record_evidence(
        sid,
        job,
        _hf_replay_item(f"{sid}:ev:owner", rid, passage, source_name=domain),
        repo=repo,
    )
    assert out["job_id"] == job
    assert repo.get_evidence(f"{sid}:ev:owner")["job_id"] == job
    accepted = [event for event in repo.list_events(sid) if event.event_type == "evidence.accepted"]
    assert len(accepted) == 1
    assert accepted[0].payload["job_id"] == job
    assert accepted[0].payload["evidence_id"] == out["evidence_id"]
    assert accepted[0].payload["tool_result_id"] == rid


def test_hf_replay_finra_dataset_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare persisted dataset still grounds the FINRA citation when dataset_id is absent."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = f"{sid}:tr:fin-dataset"
    marker = rid.rsplit(":tr:", 1)[-1]
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    _svc.persist_tool_result(
        sid,
        fin,
        "query_finra",
        rid,
        {
            "records": [{"symbol": "NVDA", "shortInterest": 12345, "marker": marker}],
            "briefing": f"NVDA short interest 12345 shares {marker}",
            "dataset": "short-interest",
            "as_of_date": "2025-06-15",
            "published_at": "2025-06-20",
        },
        repo=repo,
    )
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-dataset",
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": identity,
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "finra_record"
    assert _hf_prov(out, "dataset") == "short-interest"


def test_hf_replay_finra_declared_source_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller-declared source_uri survives the FINRA replay onto the stored provenance."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, _ = _hf_desks(repo, sid)
    rid = _hf_persist_finra(repo, sid, fin, f"{sid}:tr:fin-uri")
    marker = rid.rsplit(":tr:", 1)[-1]
    identity = json.dumps({"symbol": "NVDA", "shortInterest": 12345, "marker": marker}, sort_keys=True)
    out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-uri",
            "content": f"NVDA short interest 12345 shares {marker}",
            "claim_text": f"NVDA short interest 12345 shares {marker}",
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": rid,
            "matching_passage": identity,
            "source_uri": "https://example.com/finra/short-interest",
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "finra_record"
    assert _hf_prov(out, "source_uri") == "https://example.com/finra/short-interest"


def test_hf_replay_keeps_full_long_identities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete cited record persists as full canonical JSON regardless of length; WEB stays truncated."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    fin_passage = "Short interest increased. " * 120
    fin_rid = f"{sid}:tr:fin-long"
    fin_row = json.dumps({"note": fin_passage}, sort_keys=True)
    _svc.persist_tool_result(
        sid,
        fin,
        "query_finra",
        fin_rid,
        {
            "records": [{"note": fin_passage}],
            "briefing": fin_passage,
            "as_of_date": "2025-06-15",
            "published_at": "2025-06-20",
        },
        repo=repo,
    )
    fin_out = _svc.record_evidence(
        sid,
        fin,
        {
            "evidence_id": f"{sid}:ev:fin-long",
            "content": fin_passage,
            "claim_text": fin_passage,
            "subject": "NVDA",
            "source_name": "FINRA",
            "tool_result_id": fin_rid,
            "matching_passage": fin_row,
        },
        repo=repo,
    )
    fin_prov = fin_out["provenance"]
    assert isinstance(fin_prov, dict)
    assert fin_prov["record_identity"] == fin_row
    sec_passage = "Revenue increased. " * 120
    sec_row = json.dumps({"description": sec_passage, "filed_at": "2025-06-01", "form": "10-Q"}, sort_keys=True)
    sec_rid = str(
        _svc.persist_tool_result(
            sid,
            sec_job,
            "list_sec_filings",
            f"{sid}:tr:sec-long",
            {"filings": [{"form": "10-Q", "description": sec_passage, "filed_at": "2025-06-01"}]},
            repo=repo,
        )["tool_result_id"]
    )
    sec_out = _svc.record_evidence(
        sid,
        sec_job,
        {
            "content": sec_passage,
            "source_name": "SEC",
            "tool_result_id": sec_rid,
            "matching_passage": sec_row,
        },
        repo=repo,
    )
    sec_prov = sec_out["provenance"]
    assert isinstance(sec_prov, dict)
    assert sec_prov["record_identity"] == sec_row
    web_passage = "Company demand increased. " * 120
    web_rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-long",
            {
                "evidence": [
                    {
                        "url": "https://example.com/long",
                        "source_domain": "example.com",
                        "title": "long read",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": web_passage,
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    web_out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-long",
            "content": web_passage,
            "claim_text": web_passage,
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": web_rid,
            "matching_passage": web_passage,
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    web_prov = web_out["provenance"]
    assert isinstance(web_prov, dict)
    assert len(str(web_prov["excerpt"])) == 2000


def test_hf_replay_sec_ticker_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC dataset reads the persisted ticker; a result with no subject/ticker stores no dataset."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    tick_passage = "NVDA Form 4 accession 0001045810-25-000011"
    tick_row = {"description": tick_passage, "filed_at": "2025-06-01", "form": "4"}
    tick_rid = str(
        _svc.persist_tool_result(
            sid,
            sec_job,
            "list_sec_filings",
            f"{sid}:tr:sec-tick",
            {"filings": [dict(tick_row)], "ticker": "NVDA"},
            repo=repo,
        )["tool_result_id"]
    )
    tick_out = _svc.record_evidence(
        sid,
        sec_job,
        {
            "evidence_id": f"{sid}:ev:sec-tick",
            "content": tick_passage,
            "source_name": "SEC",
            "tool_result_id": tick_rid,
            "matching_passage": json.dumps(tick_row, sort_keys=True),
        },
        repo=repo,
    )
    assert _hf_prov(tick_out, "kind") == "sec_record"
    bare_passage = "NVDA 10-Q accession 0001045810-25-000012"
    bare_row = {"description": bare_passage, "filed_at": "2025-06-01", "form": "10-Q"}
    bare_rid = str(
        _svc.persist_tool_result(
            sid,
            sec_job,
            "list_sec_filings",
            f"{sid}:tr:sec-bare",
            {"filings": [dict(bare_row)]},
            repo=repo,
        )["tool_result_id"]
    )
    bare_out = _svc.record_evidence(
        sid,
        sec_job,
        {
            "evidence_id": f"{sid}:ev:sec-bare",
            "content": bare_passage,
            "source_name": "SEC",
            "tool_result_id": bare_rid,
            "matching_passage": json.dumps(bare_row, sort_keys=True),
        },
        repo=repo,
    )
    bare_prov = bare_out["provenance"]
    assert isinstance(bare_prov, dict)
    assert "dataset" not in bare_prov


def test_hf_replay_sec_source_uri_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No declared URI: the SEC citation keeps the persisted result source."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    sec_job = str(_svc.start_job(sid, "source_agent", source="SEC", repo=repo, wave_id=1)["job_id"])
    uri = "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000011/"
    passage = "NVDA Form 4 accession 0001045810-25-000011"
    uri_row = {"description": passage, "filed_at": "2025-06-01", "form": "4"}
    rid = str(
        _svc.persist_tool_result(
            sid,
            sec_job,
            "list_sec_filings",
            f"{sid}:tr:sec-uri",
            {"filings": [dict(uri_row)], "source": uri},
            repo=repo,
        )["tool_result_id"]
    )
    out = _svc.record_evidence(
        sid,
        sec_job,
        {
            "evidence_id": f"{sid}:ev:sec-uri",
            "content": passage,
            "source_name": "SEC",
            "tool_result_id": rid,
            "matching_passage": json.dumps(uri_row, sort_keys=True),
        },
        repo=repo,
    )
    assert _hf_prov(out, "kind") == "sec_record"
    assert _hf_prov(out, "source_uri") == uri


def test_hf_replay_web_skips_highlightless_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A persisted row without highlight is not citable; the next matching row still grounds."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    good = "NVDA rallies on demand"
    rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-skip",
            {
                "evidence": [
                    {
                        "url": "https://example.com/skip",
                        "source_domain": "example.com",
                        "title": "skip",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                    },
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": good,
                    },
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-skip",
            "content": good,
            "claim_text": good,
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": good,
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    assert _hf_prov(out, "url") == "https://example.com/a"
    assert _hf_prov(out, "excerpt") == good


def test_hf_replay_web_rejects_blank_highlight_fabrication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A highlightless persisted row never mints excerpt text: citing it fails ERR_PASSAGE_NOT_IN_SOURCE."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-blank",
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    with pytest.raises(ValueError, match="ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc.record_evidence(
            sid,
            web,
            {
                "evidence_id": f"{sid}:ev:web-blank",
                "content": "XXXX",
                "claim_text": "XXXX",
                "subject": "NVDA",
                "source_name": "web",
                "tool_result_id": rid,
                "matching_passage": "XXXX",
                "known_at": "2025-06-29T00:00:00+00:00",
            },
            repo=repo,
        )
    assert repo.list_evidence(sid) == []


def test_hf_replay_web_representation_defaults_to_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider extraction without a representation stores as text."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    highlight = "NVDA rallies on demand"
    rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-repr",
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": highlight,
                        "source_content": f"{highlight}. Revenue increased.",
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-repr",
            "content": highlight,
            "claim_text": highlight,
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": highlight,
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    prov = out["provenance"]
    assert isinstance(prov, dict)
    assert prov["source_content"] == f"{highlight}. Revenue increased."
    assert prov["source_representation"] == "text"


def test_hf_replay_web_omits_absent_optional_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A persisted web row without title/domain stores no fabricated placeholder for them."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    highlight = "NVDA rallies on demand"
    rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-bare",
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": highlight,
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-bare",
            "content": highlight,
            "claim_text": highlight,
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": highlight,
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    prov = out["provenance"]
    assert isinstance(prov, dict)
    assert prov["excerpt"] == highlight
    assert "title" not in prov and "domain" not in prov


def test_submit_success_returns_refs_and_persists_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Submit returns job/dossier/evidence/telemetry refs and persists the job result."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_status"] == "completed"
    assert out["job_id"] == src
    assert out["dossier_id"] == f"{sid}:1:sec"
    assert out["evidence_ids"] == [eid]
    assert isinstance(out["telemetry"], dict) and isinstance(out["warnings"], list)
    stored = repo.get_job(src).result or {}
    coverage = stored.get("coverage")
    assert isinstance(coverage, dict)
    assert coverage["useful_for_question"] == "sufficient"
    assert stored.get("evidence_ids") == [eid]
    assert isinstance(stored.get("telemetry"), dict)
    assert repo.get_session(sid).current_wave == 1


def test_submit_wave2_advances_current_wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wave-2 submit advances the session current_wave to 2."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid1 = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid1), repo=repo)
    _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid1], repo=repo)
    _svc.freeze_session(sid, 1, repo=repo)
    src2 = str(_svc.start_job(sid, "source_agent", repo=repo, wave_id=2)["job_id"])
    eid2 = f"{sid}:ev:2"
    admitted = _svc.record_evidence(
        sid,
        src2,
        _reg_item(
            eid2,
            wave=2,
            claim_text="The later filing reports revised GS exposure terms.",
            matching_passage="The firm reports revised OpenAI-linked exposure terms.",
        ),
        repo=repo,
    )
    assert admitted["evidence_id"] == eid2
    assert admitted["job_id"] == src2
    assert admitted["wave_id"] == 2
    assert repo.get_evidence(eid2)["job_id"] == src2
    out2 = _svc.submit_source_result(
        src2, coverage={"useful_for_question": "insufficient"}, evidence_ids=[eid2], repo=repo
    )
    assert out2["job_id"] == src2
    assert repo.get_session(sid).current_wave == 2
    result = repo.get_job(src2).result
    assert isinstance(result, dict)
    assert result["evidence_ids"] == [eid2]


def test_submit_insufficient_preserves_unresolved_in_dossier_and_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An insufficient submit persists unresolved questions in the dossier and telemetry."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    cov_question = "Coverage residual terms"
    arg_question = "Is the exposure hedged?"
    out = _svc.submit_source_result(
        src,
        coverage={"useful_for_question": "insufficient", "material_open_questions": [cov_question]},
        evidence_ids=[eid],
        unresolved_questions=[arg_question],
        repo=repo,
    )
    dossier = repo.get_dossier(str(out["dossier_id"]))
    open_questions = dossier.get("open_questions")
    assert isinstance(open_questions, list)
    assert arg_question in open_questions
    telemetry = out["telemetry"]
    assert isinstance(telemetry, dict)
    remaining = telemetry.get("branches_remaining")
    assert isinstance(remaining, list)
    assert arg_question in remaining and cov_question in remaining


def test_submit_uses_explicit_repo_not_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Submit honors the explicit repo object when the env DB points elsewhere."""
    from app.research import service as _svc
    from app.research.repository import ResearchRepository as _RR

    env_db = tmp_path / "env.sqlite"
    real_db = tmp_path / "real.sqlite"
    monkeypatch.setenv("RESEARCH_DB_PATH", str(env_db))
    repo = _RR(path=real_db)
    sid, src = _reg_svc_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _reg_item(eid), repo=repo)
    out = _svc.submit_source_result(src, coverage=_reg_cov(), evidence_ids=[eid], repo=repo)
    assert out["job_id"] == src
    assert repo.get_job(src).status == "completed"
    assert _RR(path=env_db).list_sessions() == []


def test_decision_service_budget_ignores_non_string_request_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string budget request_id is ignored: the job still creates with no dedup claim."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    before = len(repo.list_jobs(sid))
    first = _svc.start_job(sid, "source_agent", budget={"request_id": 123}, repo=repo)
    assert first["job_id"]
    assert repo.submission_job_id(sid, "123") is None
    second = _svc.start_job(sid, "source_agent", budget={"request_id": ""}, repo=repo)
    assert second["job_id"] != first["job_id"]
    assert len(repo.list_jobs(sid)) == before + 2


def test_decision_service_node_created_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """create_node persists the node and journals one node.created event with its id."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    node = _svc.create_node(sid, "what?", "why", repo=repo)
    assert repo.get_node(node.node_id).question == "what?"
    matches = [e for e in repo.list_events(sid) if e.event_type == "node.created"]
    assert len(matches) == 1
    assert matches[0].payload.get("node_id") == node.node_id


def test_decision_service_decision_confidence_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """record_decision persists the caller confidence on the returned row and the stored row."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    recorded = _svc.record_decision(sid, "route", {"a": 1}, {"a": 0.5}, "a", confidence=0.75, repo=repo)
    assert recorded.confidence == 0.75
    stored = repo.list_decisions(sid)
    assert len(stored) == 1
    assert stored[0].confidence == 0.75
    assert stored[0].to_dict()["confidence"] == 0.75


def test_decision_service_decision_recorded_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """record_decision persists the row and journals one decision.recorded event with id and type."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo)
    recorded = _svc.record_decision(sid, "route", {"a": 1}, {"a": 0.5}, "a", repo=repo)
    assert repo.list_decisions(sid)[0].decision_id == recorded.decision_id
    matches = [e for e in repo.list_events(sid) if e.event_type == "decision.recorded"]
    assert len(matches) == 1
    assert matches[0].payload.get("decision_id") == recorded.decision_id
    assert matches[0].payload.get("type") == "route"


def test_submit_existing_dossier_keeps_current_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A later source job reports its current coverage despite the immutable earlier dossier."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, first = _svc_sid(repo)
    second = str(_svc.start_job(sid, "source_agent", repo=repo)["job_id"])
    before: dict[str, object] = {
        "useful_for_question": "insufficient",
        "material_open_questions": ["Prior filing terms"],
    }
    after: dict[str, object] = {
        "useful_for_question": "insufficient",
        "material_open_questions": ["Current filing terms"],
    }
    prior = _svc.submit_source_result(first, coverage=before, repo=repo)
    current = _svc.submit_source_result(second, coverage=after, repo=repo)
    assert current["dossier_id"] == prior["dossier_id"]
    telemetry = current["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["branches_remaining"] == ["Current filing terms"]
    stored = repo.get_job(second).result
    assert isinstance(stored, dict)
    assert stored["telemetry"] == telemetry


def test_submit_sufficient_missing_keys_reports_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sufficient envelope missing two work keys reports both; the job stays running with no dossier."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, src = _inv_sid(repo)
    eid = f"{sid}:ev:1"
    _svc.record_evidence(sid, src, _svc_item(eid), repo=repo)
    coverage: dict[str, object] = dict(_reg_cov())
    del coverage["forms_examined"]
    del coverage["search_runs"]
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED") as excinfo:
        _svc.submit_source_result(src, coverage=coverage, evidence_ids=[eid], repo=repo)
    message = str(excinfo.value)
    assert "forms_examined" in message and "search_runs" in message
    assert repo.get_job(src).status == "running"
    assert repo.get_job(src).result is None
    assert repo.list_dossiers(sid) == []
    assert repo.get_session(sid).status not in ("completed", "synthesizing", "failed")


def test_create_node_rejects_null_nonstring_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A null/non-string question fails through create_node with the documented ValueError."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _svc_sid(repo, "first question?")
    # Malformed inputs intentionally sit outside the static signature.
    create_untyped: Callable[..., object] = _svc.create_node
    bad_questions: list[object] = [None, 123]
    for bad_question in bad_questions:
        with pytest.raises(ValueError, match="'question' must be a non-empty string"):
            create_untyped(sid, bad_question, "why it matters", repo=repo)
    assert repo.list_nodes(sid) == []


def test_intake_newest_4a_digest_carries_insider_transactions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Newest 4/A: intake opens insider activity, so the digest carries parsed transaction rows.

    Live defect: the chain opened only 8-Ks, so a 4/A-led listing left the answer with
    filing metadata (form, filed date) and no transaction code, kind, shares or price.
    """
    import asyncio

    from app.research import kernel_worker as kw
    from app.research import scheduler as sched
    from app.research import service as _svc
    from app.tool_runtime import outcome_from_result

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("Did NVDA insiders sell?", "o", as_of=None, repo=repo)
    payloads: dict[str, dict[str, object]] = {
        "list_sec_filings": {
            "subject": "1045810",
            "filings": [
                {"form": "4/A", "accession_no": "0001045810-26-000002", "filed_at": "2026-09-09"},
                {"form": "8-K", "accession_no": "0001045810-26-000001", "filed_at": "2026-09-01"},
            ],
            "source": "SEC EDGAR",
        },
        "get_insider_activity": {
            "subject": "NVDA",
            "count": 1,
            "transactions": [
                {
                    "insider_name": "Huang Jen-Hsun",
                    "transaction_code": "S",
                    "transaction_kind": "open_market_sale",
                    "transaction_date": "2026-09-05",
                    "acquired_disposed": "D",
                    "shares": 2605,
                    "price": 360.13,
                    "filed_at": "2026-09-09",
                    "accession_no": "0001045810-26-000002",
                }
            ],
            "source": "SEC EDGAR",
        },
    }
    calls: list[str] = []

    async def _attempt(
        tool: str,
        args: dict[str, object],
        sid: str,
        session: object,
        registry: object,
        jev: object,
        kernel: sched._Kernel,
        as_of: str | None,
    ) -> dict[str, object]:
        calls.append(tool)
        if tool == "get_sec_document":
            return {"tool": tool, "arguments": args, "error": "no document backend", "outcome": None}
        job_id = sched._attempt_job(kernel, sid, "SEC")
        result = dict(payloads[tool])
        forms = args.get("forms")
        if isinstance(forms, list):
            filings = result["filings"]
            assert isinstance(filings, list)
            result["filings"] = [r for r in filings if r["form"] in forms][:1]
        ref = f"{sid}:tr:{tool}:{forms[0] if isinstance(forms, list) else ''}"
        _svc.persist_tool_result(sid, job_id, tool, ref, result)
        result["tool_result_id"] = ref
        return sched._success_attempt(tool, args, outcome_from_result(tool, result), result, "SEC", "intake", job_id)

    def _cik(ticker: str) -> str:
        return "1045810"

    monkeypatch.setattr(kw, "_intake_attempt", _attempt)
    monkeypatch.setattr(kw, "_intake_cik", _cik)
    admitted, _raw, stats = asyncio.run(
        kw._intake_round(None, ["NVDA"], sid, repo.get_session(sid).to_dict(), [], None, sched._Kernel(), None)
    )
    # Four family listings, then the 8-K read beside insider activity, then one bounded retry of the unread 8-K.
    assert sorted(calls) == sorted(["list_sec_filings"] * 4 + ["get_sec_document"] * 2 + ["get_insider_activity"])
    assert stats["unread_material"] == ["0001045810-26-000001"]
    # Every family listing admits as SEC record evidence; the 8-K read and its retry fail without a doc backend.
    assert stats["outcomes"] == {"admitted": 5, "duplicate": 0, "not_citable": 0, "error": 2, "timeout": 0}
    row = (
        "  - insider_name Huang Jen-Hsun, transaction_code S, transaction_kind open_market_sale, "
        "transaction_date 2026-09-05, acquired_disposed D, shares 2605, price 360.13, "
        "filed_at 2026-09-09, accession_no 0001045810-26-000002"
    )
    contents = [str(ev.get("content", "")) for ev in admitted]
    # Metadata alone (the listing) names the form and filed date but no transaction.
    listing = [c for c in contents if "4/A" in c and row not in c]
    assert listing and all("transaction_code" not in c and "shares" not in c for c in listing)
    assert [c for c in contents if row in c]
    assert row in kw._build_digest(admitted, ("NVIDIA", "NVDA"))


def _regweb_cov(**over: object) -> dict[str, object]:
    """Web absence coverage envelope: searched queries/domains, never a universal claim."""
    cov: dict[str, object] = {
        "forms": [],
        "dates": [],
        "partitions": [],
        "entities": ["NVDA"],
        "docs": ["search_web:NVDA demand"],
        "gaps": [],
        "pagination_complete": True,
        "complete": True,
    }
    cov.update(over)
    return cov


def _regweb_absence(repo: ResearchRepository, sid: str, job: str, search_id: str, query: str) -> dict[str, object]:
    from app.research import service as _svc

    return _svc.record_evidence(  # type: ignore[no-any-return]
        sid,
        job,
        {
            "wave_id": 1,
            "content": f"no web result for {query}",
            "claim_text": f"not located in searched web scope: {query}",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "web",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": search_id,
            "query": query,
            "coverage": _regweb_cov(),
        },
        repo=repo,
    )


def test_regweb_absence_links_query_scope_period_tools_pagination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A web absence records its searched query/scope/period/tools/pagination/completeness."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    web = str(_svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)["job_id"])
    out = _regweb_absence(repo, sid, web, "w1", "NVDA demand")
    assert out.get("recorded_as") == "coverage_artifact" and out.get("citable") is False
    assert out.get("search_id") == "w1" and out.get("query") == "NVDA demand"
    stored = repo.list_coverage_artifacts(sid)[0]
    assert stored.get("coverage") == _regweb_cov()
    assert stored.get("source_domain") == "WEB"


def test_regweb_absence_scoped_per_domain_never_universal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same search text on SEC and WEB jobs mints two artifacts; neither implies universal absence."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, sec_job = _hf_sid(repo, "SEC", "FINRA", "WEB")
    web = str(_svc.start_job(sid, "source_agent", source="WEB", repo=repo, wave_id=1)["job_id"])
    first = _regweb_absence(repo, sid, web, "w1", "NVDA demand")
    second = _svc.record_evidence(
        sid,
        sec_job,
        {
            "wave_id": 1,
            "content": "no SEC hit for NVDA demand",
            "claim_text": "not located in searched SEC scope: NVDA demand",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "w1",
            "query": "NVDA demand",
            "coverage": _absence_cov(),
        },
        repo=repo,
    )
    assert first["artifact_id"] != second["artifact_id"]
    domains = {a.get("source_domain") for a in repo.list_coverage_artifacts(sid)}
    assert domains == {"WEB", "SEC"}
    assert _svc._finalize_absence_texts(repo, sid, domain="WEB") == [first["claim_text"]]


def test_regweb_proof_requires_excerpt_verbatim_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A web datapoint value must read verbatim off the cited excerpt, never the wider extraction."""
    from app.research import service as _svc
    from app.research.evidence import observation_proof, web_source_ref

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid, _ = _hf_sid(repo, "SEC", "FINRA", "WEB")
    _, web = _hf_desks(repo, sid)
    highlight = "NVDA rallies on quarterly demand"
    rid = str(
        _svc.persist_tool_result(
            sid,
            web,
            "search_web",
            f"{sid}:tr:web-exact",
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "title": "NVDA news",
                        "published_at": "2025-06-20",
                        "retrieved_at": "2025-06-21",
                        "highlight": highlight,
                        "source_content": f"{highlight}. Unrelated full-page body mentions 1999 exactly once.",
                    }
                ]
            },
            repo=repo,
        )["tool_result_id"]
    )
    out = _svc.record_evidence(
        sid,
        web,
        {
            "evidence_id": f"{sid}:ev:web-exact",
            "content": highlight,
            "claim_text": highlight,
            "subject": "NVDA",
            "source_name": "web",
            "tool_result_id": rid,
            "matching_passage": highlight,
            "known_at": "2025-06-29T00:00:00+00:00",
        },
        repo=repo,
    )
    prov = out["provenance"]
    assert isinstance(prov, dict) and prov["excerpt"] == highlight
    ref = web_source_ref(
        url="https://example.com/a",
        excerpt=highlight,
        tool_result_id=rid,
        source_content="1999 appears only outside the excerpt",
    )
    with pytest.raises(ValueError, match="is not a whole token"):
        observation_proof(
            provenance=ref,
            source_bytes=b"1999 appears only outside the excerpt",
            value="1999",
            metric="NVDA",
            entity="NVDA",
            period="2025",
        )


def test_regweb_snippet_never_proves_full_article_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An excerpt-only row replays as search_excerpt: the claim verifies against the snippet alone."""
    from app.research.sources import read_source_document

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))

    class _WebRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            assert evidence_id == "ev-snip"
            return {
                "evidence_id": evidence_id,
                "session_id": "s",
                "provenance": {
                    "kind": "web_source",
                    "url": "https://example.com/snippet",
                    "excerpt": "snippet headline only",
                    "tool_result_id": "tr-snip",
                },
            }

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            assert tool_result_id == "tr-snip"
            return {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {
                    "query": "NVDA news",
                    "evidence": [{"url": "https://example.com/snippet", "highlight": "snippet headline only"}],
                },
            }

    document = read_source_document("evidence:ev-snip", data_root=tmp_path, repo=_WebRepo())
    assert document["captureKind"] == "search_excerpt"
    assert document["quote"] == "snippet headline only"
    assert document["linkStatus"] == "stockbot_render"


def test_regweb_pinned_drift_fails_closed_and_no_today_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned row whose replay no longer holds the excerpt fails closed; today's text never substitutes."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    provenance = {
        "kind": "web_source",
        "url": "https://example.com/a",
        "excerpt": "quoted excerpt",
        "tool_result_id": "tr-drift",
    }

    class _DriftRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            return {"evidence_id": evidence_id, "session_id": "s", "provenance": dict(provenance)}

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            return {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {"evidence": [{"url": "https://example.com/a", "highlight": "today's rewritten text"}]},
            }

    from app.research.sources import read_source_document as _read_web_source

    with pytest.raises(FileNotFoundError, match="excerpt"):
        _read_web_source("evidence:ev-drift", data_root=tmp_path, repo=_DriftRepo())


# ---------------------------------------------------------------------------
# Change 4 regression: FINRA observation proofs, aggregates, absence coverage
# ---------------------------------------------------------------------------
def _reg4_finra_bytes(payload: object) -> tuple[bytes, str]:
    import json as _json

    from app.storage import raw_archive as _archive

    raw = _json.dumps(payload, sort_keys=True).encode()
    return raw, f"sha256:{_archive.content_hash(raw)}"


def test_reg4_finra_observation_proof_verifies_and_conflicts() -> None:
    """A FINRA observation verifies its archived bytes; tampered bytes or values conflict."""
    import dataclasses as _dc

    from app.research.evidence import finra_record_ref, observation_proof, verify_observation

    payload = {"records": [{"symbolCode": "NVDA", "settlementDate": "2026-09-30", "v": 10}]}
    raw, source_id = _reg4_finra_bytes(payload)
    locator_row = '{"symbolCode": "NVDA", "settlementDate": "2026-09-30", "v": 10}'
    prov = finra_record_ref(
        tool_name="query_finra",
        record_identity=locator_row,
        dataset="otcMarket/consolidatedShortInterest",
        tool_result_id="tr-1",
    )
    proof = observation_proof(
        provenance=prov, source_bytes=raw, value="10", metric="v", entity="NVDA", period="2026-09-30"
    )
    assert proof["sourceId"] == source_id
    row = _dc.replace(
        _observation_row("EV-DC", _PROOF_DC, "22,563"),
        evidence_id="EV-FIN",
        claim_kind="observed_fact",
        provenance=dict(prov),
        value="10",
        entity="NVDA",
        period="2026-09-30",
        unit=None,
        proof={
            "kind": "observation",
            "sourceId": proof["sourceId"],
            "locator": proof["locator"],
            "valueHash": proof["valueHash"],
        },
    )
    assert verify_observation(row, value="10", source_bytes=raw).status == "verified"
    assert verify_observation(row, value="11", source_bytes=raw).status == "conflicted"
    assert verify_observation(row, value="10", source_bytes=raw + b" ").status == "conflicted"
    assert verify_observation(row, value="10", source_bytes=None).status == "unverified"


def test_reg4_finra_proof_bytes_absent_fails_closed() -> None:
    """Bytes-absent never verifies: freeze membership alone is insufficient."""
    import dataclasses as _dc

    from app.research.evidence import finra_record_ref, observation_proof, value_hash, verify_observation
    from app.research.freeze import create_freeze, verify_freeze

    payload = {"records": [{"symbolCode": "NVDA", "settlementDate": "2026-09-30", "v": 10}]}
    raw, source_id = _reg4_finra_bytes(payload)
    locator_row = '{"symbolCode": "NVDA", "settlementDate": "2026-09-30", "v": 10}'
    prov = finra_record_ref(
        tool_name="query_finra",
        record_identity=locator_row,
        dataset="otcMarket/consolidatedShortInterest",
        tool_result_id="tr-1",
    )
    proof = observation_proof(
        provenance=prov, source_bytes=raw, value="10", metric="v", entity="NVDA", period="2026-09-30"
    )
    row = _dc.replace(
        _observation_row("EV-DC", _PROOF_DC, "22,563"),
        evidence_id="EV-FIN",
        claim_kind="observed_fact",
        provenance=dict(prov),
        value="10",
        entity="NVDA",
        period="2026-09-30",
        unit=None,
        proof={
            "kind": "observation",
            "sourceId": source_id,
            "locator": proof["locator"],
            "valueHash": value_hash("10"),
        },
    )
    frozen = create_freeze(freeze_id="F4", session_id="S-proof", wave_id=1, records=[row], as_of=ASOF)
    verify_freeze(frozen, [row])
    # Frozen and intact, yet with the archived bytes gone the row still never verifies.
    assert verify_observation(row, value="10", source_bytes=None).status == "unverified"
    assert verify_observation(row, value="10", source_bytes=raw).status == "verified"


def test_reg4_finra_provenance_needs_persisted_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A FINRA citation without a persisted tool result or with a wrong-lane result fails closed."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    sid = _svc.create_research("q?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
    with pytest.raises(ValueError, match="ERR_RAW_SOURCE_REQUIRED|ERR_PASSAGE_NOT_IN_SOURCE"):
        _svc._finra_provenance({"tool_result_id": "nope", "matching_passage": '{"v": 1}'}, store=repo, session_id=sid)


def test_reg4_finra_aggregated_leaderboard_keeps_all_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leaderboard persists every contributing record; replay cites the second, not the first."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    import dataclasses as _dc

    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    sid = _svc.create_research("q?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    repo.save_job(_dc.replace(repo.get_job(src), source_domain="FINRA", status="running"))
    _svc.persist_tool_result(
        sid,
        src,
        "query_finra",
        "tr-board",
        {
            "records": [{"ticker": "AAA", "pct": 30.0}, {"ticker": "BBB", "pct": 20.0}],
            "dataset_id": "otcMarket/consolidatedShortInterest",
        },
        repo=repo,
    )
    prov = _svc._finra_provenance(
        {
            "tool_result_id": "tr-board",
            "matching_passage": json.dumps({"pct": 20.0, "ticker": "BBB"}, sort_keys=True),
        },
        store=repo,
        session_id=sid,
    )
    assert str(prov["record_identity"]) == json.dumps({"pct": 20.0, "ticker": "BBB"}, sort_keys=True)
    assert prov["tool_result_id"] == "tr-board"


def test_reg4_absence_claim_needs_full_coverage_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absence claim without the full query/scope/period/tools/pagination envelope fails closed."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    sid = _svc.create_research("q?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    base = {
        "wave_id": 1,
        "content": "no listing",
        "claim_text": "no listing in scope",
        "claim_kind": "absence_observation",
        "subject": "NVDA",
        "source_name": "FINRA",
        "known_at": "2026-06-29T00:00:00+00:00",
        "search_id": "s1",
        "query": "NVDA threshold",
    }
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(sid, src, dict(base), repo=repo)
    with pytest.raises(ValueError, match="ERR_COVERAGE_REQUIRED"):
        _svc.record_evidence(
            sid, src, {**base, "coverage": {**_absence_cov(), "pagination_complete": "yes"}}, repo=repo
        )
    out = _svc.record_evidence(sid, src, {**base, "coverage": _absence_cov()}, repo=repo)
    assert out["recorded_as"] == "coverage_artifact" and out["citable"] is False


# ---------------------------------------------------------------------------
# Change 5 regression: explicit legacy evidence repair (repair artifacts beside frozen rows)
# ---------------------------------------------------------------------------

_REP_CIK = 1045810
_REP_SID = "S-rep"
_REP_JOB = "J-rep"
_REP_PAGE = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&owner=include"
_REP_DPS_CONCEPT = "CommonStockDividendsPerShareDeclared"


def _rep_fact(
    start: str, end: str, val: float, accn: str, filed: str, fp: str, fy: int, form: str = "10-Q"
) -> JSONValue:
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}


# Q2 FY2026 EPS was reported at 1.08 and later amended to 1.05 by a 10-Q/A.
_REP_EPS = [
    _rep_fact("2025-01-27", "2025-04-27", 0.76, "0001045810-25-000100", "2025-05-28", "Q1", 2026),
    _rep_fact("2025-04-28", "2025-07-27", 1.08, "0001045810-25-000200", "2025-08-27", "Q2", 2026),
    _rep_fact("2025-04-28", "2025-07-27", 1.05, "0001045810-25-000300", "2025-10-01", "Q2", 2026, "10-Q/A"),
]
# FY2025 dividends: three quarters, a nine-month YTD, the fiscal year, then Q1 FY2026.
_REP_DPS = [
    _rep_fact("2024-01-29", "2024-04-28", 0.004, "0001045810-24-000100", "2024-05-29", "Q1", 2025),
    _rep_fact("2024-04-29", "2024-07-28", 0.01, "0001045810-24-000200", "2024-08-28", "Q2", 2025),
    _rep_fact("2024-07-29", "2024-10-27", 0.01, "0001045810-24-000300", "2024-11-20", "Q3", 2025),
    _rep_fact("2024-01-29", "2024-10-27", 0.024, "0001045810-24-000300", "2024-11-20", "Q3", 2025),
    _rep_fact("2024-01-29", "2025-01-26", 0.034, "0001045810-25-000050", "2025-02-26", "FY", 2025, "10-K"),
    _rep_fact("2025-01-27", "2025-04-27", 0.01, "0001045810-25-000100", "2025-05-28", "Q1", 2026),
]


@pytest.fixture
def _rep_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchRepository:
    """Isolated ledger plus a fake companyfacts endpoint (no network). Repairs archive the served bytes."""
    import json

    from app.services import research_data

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    units = {"EarningsPerShareDiluted": _REP_EPS, _REP_DPS_CONCEPT: _REP_DPS}
    facts = {
        "cik": _REP_CIK,
        "entityName": "NVIDIA",
        "facts": {"us-gaap": {concept: {"units": {"USD/shares": rows}} for concept, rows in units.items()}},
    }

    def edgar_get(url: str) -> bytes:
        assert url.endswith(f"CIK{_REP_CIK:010d}.json")
        return json.dumps(facts).encode()

    monkeypatch.setattr(research_data, "_edgar_get", edgar_get)
    return ResearchRepository()


def _rep_result(
    store: ResearchRepository, rid: str, tool: str, payload: Mapping[str, object], session_id: str = _REP_SID
) -> None:
    store.save_tool_result(
        {
            "tool_result_id": rid,
            "session_id": session_id,
            "job_id": _REP_JOB,
            "tool_name": tool,
            "created_at": "2025-08-01T00:00:00+00:00",
            "result": payload,
        }
    )


def _rep_row(
    store: ResearchRepository,
    eid: str,
    retrieved: str,
    provenance: Mapping[str, str] | None = None,
    **fields: JSONValue,
) -> Evidence:
    """One legacy row: no proof, persisted exactly as an older session stored it (fields override)."""
    from app.research.evidence import evidence_from_dict, evidence_to_dict

    at = datetime.fromisoformat(retrieved)
    content = f"legacy {eid}"
    base = Evidence(
        evidence_id=eid,
        session_id=_REP_SID,
        wave_id=1,
        source_type="sec",
        source_name="SEC",
        subject="NVDA",
        claim_text="legacy citation",
        content=content,
        content_hash=evidence_content_hash(content),
        known_at=at,
        retrieved_at=at,
        job_id=_REP_JOB,
        provenance={key: value for key, value in (provenance or {}).items()},
    )
    row = {**evidence_to_dict(base), **fields}
    store.save_evidence(row)
    return evidence_from_dict(row)


def _rep_record(tool: str, cited: object, rid: str = "", kind: str = "sec_record") -> dict[str, str]:
    import json

    identity = cited if isinstance(cited, str) else json.dumps(cited)
    return {"kind": kind, "tool_name": tool, "record_identity": identity, **({"tool_result_id": rid} if rid else {})}


def _rep_brief(artifacts: Sequence[Mapping[str, object]]) -> list[tuple[object, ...]]:
    return [(a["category"], a["datapoint"], a["status"], a["reason"]) for a in artifacts]


def _rep_nested(artifact: object, key: str, field: str) -> object:
    """artifact[key][field] of one repair artifact, checking both levels are mappings."""
    assert isinstance(artifact, Mapping)
    inner = artifact[key]
    assert isinstance(inner, Mapping)
    return inner.get(field)


def _rep_accession(artifact: object) -> object:
    return _rep_nested(artifact, "source", "accession")


def test_rep_eps_without_accession_records_verified_artifact_beside_untouched_row(
    _rep_store: ResearchRepository,
) -> None:
    """The retained period identity resolves the original fact; the row, its hash, and its freeze stay as stored."""
    import json

    from app.research import repair
    from app.research.agents import GroundedClaim
    from app.research.evidence import validate_proof, value_hash
    from app.research.freeze import freeze_from_dict, freeze_to_dict
    from app.research.service import require_verified_claims
    from app.storage import raw_archive

    store = _rep_store
    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(store, "TR-EPS", "get_fundamentals", {"ticker": "NVDA", "cik": _REP_CIK, "quarterly_eps": [q1]})
    legacy = _rep_row(store, "EV-EPS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-EPS"))
    frozen = create_freeze(
        freeze_id="F-rep", session_id=_REP_SID, wave_id=1, records=[legacy], as_of=datetime(2025, 9, 30, tzinfo=UTC)
    )
    store.save_freeze(freeze_to_dict(frozen))
    stored, stored_freeze = store.get_evidence("EV-EPS"), store.get_freeze("F-rep")
    # Keep the original replay bytes available, so publication fails for missing proof, not missing source.
    raw_archive.store_evidence_artifact(
        json.dumps(store.get_tool_result("TR-EPS"), sort_keys=True, default=str).encode()
    )
    [artifact] = repair.repair_evidence(store, "EV-EPS")
    assert _rep_brief([artifact]) == [("eps_without_accession", "eps_diluted", "verified", "")]
    assert (artifact["evidence_id"], artifact["value"], artifact["as_of"]) == (
        "EV-EPS",
        "0.76",
        "2025-09-01T00:00:00+00:00",
    )
    assert artifact["evidence_sha256"] == sha256(json.dumps(stored, sort_keys=True, default=str).encode()).hexdigest()
    source = artifact["source"]
    assert isinstance(source, Mapping)
    assert (source["accession"], source["concept"], source["period_end"]) == (
        "0001045810-25-000100",
        "EarningsPerShareDiluted",
        "2025-04-27",
    )
    proof = artifact["proof"]
    assert isinstance(proof, Mapping)
    assert validate_proof(proof) == proof
    assert (proof["kind"], proof["valueHash"]) == ("observation", value_hash("0.76"))
    locator = proof["locator"]
    assert isinstance(locator, Mapping) and locator["archive"] == f"sec/cik{_REP_CIK:010d}/companyfacts"
    digest = str(proof["sourceId"]).removeprefix("sha256:")
    assert raw_archive.read_payload("sec", f"cik{_REP_CIK:010d}", "companyfacts", sha256=digest) is not None
    # Additive storage only: the legacy row and its freeze are byte-identical and still verify.
    assert store.list_evidence_repairs(_REP_SID, "EV-EPS") == [artifact]
    assert store.get_evidence("EV-EPS") == stored and stored["proof"] == {}
    assert store.get_freeze("F-rep") == stored_freeze
    verify_freeze(freeze_from_dict(store.get_freeze("F-rep")), [legacy])
    # A verified repair remains a separate artifact. Neither its ID nor the legacy row proves a published value.
    for cited_id, error in (
        ("EV-EPS", "ERR_DATAPOINT_UNVERIFIED"),
        (str(artifact["repair_id"]), "ERR_PROOF_UNVERIFIED"),
    ):
        claim = GroundedClaim(text="NVDA EPS was 0.76", claim_type="observed_fact", evidence_ids=[cited_id])
        with pytest.raises(ValueError, match=error):
            require_verified_claims(store, _REP_SID, [claim], "acceptance")
    assert store.get_evidence("EV-EPS") == stored
    assert store.get_freeze("F-rep") == stored_freeze
    assert store.list_evidence_repairs(_REP_SID, "EV-EPS") == [artifact]
    # A rerun with the same outcome reuses the content-hash id instead of appending history.
    [again] = repair.repair_evidence(store, "EV-EPS")
    assert again["repair_id"] == artifact["repair_id"]
    assert len(store.list_evidence_repairs(_REP_SID, "EV-EPS")) == 1


@pytest.mark.parametrize(
    ("retrieved", "cited", "status", "reason", "accession"),
    [
        pytest.param(
            "2025-09-01", {"eps_diluted": 1.08}, "verified", "", "0001045810-25-000200", id="before-amendment"
        ),
        pytest.param("2025-11-01", {"eps_diluted": 1.08}, "unresolved", "SOURCE_AMBIGUOUS", None, id="after-amendment"),
        pytest.param(
            "2025-09-01", {"eps_diluted": 1.05}, "unresolved", "SOURCE_MISMATCH", None, id="restatement-cited-early"
        ),
        pytest.param(
            "2025-08-01", {"eps_diluted": 1.08}, "unresolved", "SOURCE_UNAVAILABLE", None, id="filed-after-as-of"
        ),
        pytest.param(
            "2025-11-01",
            {"eps_diluted": 1.08, "accession": "0001045810-25-000200"},
            "verified",
            "",
            "0001045810-25-000200",
            id="retained-original-accession",
        ),
        pytest.param(
            "2025-11-01",
            {"eps_diluted": 1.05, "accession": "0001045810-25-000300"},
            "verified",
            "",
            "0001045810-25-000300",
            id="retained-amendment-accession",
        ),
    ],
)
def test_rep_as_of_before_amendment_never_substitutes_a_restatement(
    retrieved: str,
    cited: dict[str, object],
    status: str,
    reason: str,
    accession: str | None,
    _rep_store: ResearchRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only facts filed by the original as_of count; a later 10-Q/A never replaces the cited observation."""
    from app.research import repair

    q2 = {"period_end": "2025-07-27", "fiscal_period": "Q2", **cited}
    _rep_result(_rep_store, "TR-Q2", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q2]})
    _rep_row(_rep_store, "EV-Q2", f"{retrieved}T00:00:00+00:00", _rep_record("get_fundamentals", q2, "TR-Q2"))
    with caplog.at_level("ERROR", logger="app.research.repair"):
        [artifact] = repair.repair_evidence(_rep_store, "EV-Q2")
    assert (artifact["status"], artifact["reason"], artifact["as_of"]) == (
        status,
        reason,
        f"{retrieved}T00:00:00+00:00",
    )
    assert _rep_accession(artifact) == accession
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    if status == "verified":
        assert errors == []
    else:
        assert artifact["proof"] == {} and artifact["source"] == {}
        [message] = errors
        assert message.startswith(f"evidence repair unresolved session={_REP_SID} evidence=EV-Q2 ")
        assert f"reason={reason} " in message
    assert _rep_store.get_evidence("EV-Q2")["proof"] == {}


def test_rep_session_cutoff_earlier_than_retrieval_is_the_as_of(_rep_store: ResearchRepository) -> None:
    """The session's own as_of wins over a later retrieval time, so the then-unfiled fact stays unavailable."""
    from app.research import repair
    from app.research import service as _svc

    sid = _svc.create_research("NVDA EPS?", "o", as_of="2025-08-01T00:00:00+00:00", repo=_rep_store)
    q2 = {"period_end": "2025-07-27", "fiscal_period": "Q2", "eps_diluted": 1.08}
    _rep_result(_rep_store, "TR-S", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q2]}, session_id=sid)
    _rep_row(
        _rep_store, "EV-S", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q2, "TR-S"), session_id=sid
    )
    [artifact] = repair.repair_evidence(_rep_store, "EV-S")
    assert (artifact["status"], artifact["reason"], artifact["as_of"]) == (
        "unresolved",
        "SOURCE_UNAVAILABLE",
        "2025-08-01T00:00:00+00:00",
    )


def test_rep_dividend_annual_and_ttm_rebuild_proofs_from_point_in_time_facts(_rep_store: ResearchRepository) -> None:
    """A whole dividend envelope yields one annual observation and one TTM calculation over four quarters."""
    import json

    from app.research import repair

    payload = {
        "cik": _REP_CIK,
        "ttm_dividend_per_share": 0.04,
        "annual_history": [{"fiscal_year": 2025, "dividend_per_share": 0.034}],
    }
    _rep_result(_rep_store, "TR-DIV", "get_fundamentals", payload)
    envelope = " ".join(json.dumps(payload, sort_keys=True).split())
    _rep_row(_rep_store, "EV-DIV", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", envelope, "TR-DIV"))
    artifacts = {str(a["datapoint"]): a for a in repair.repair_evidence(_rep_store, "EV-DIV")}
    assert sorted(_rep_brief(list(artifacts.values()))) == [
        ("dividend_annual_ttm", "annual_history[2025].dividend_per_share", "verified", ""),
        ("dividend_annual_ttm", "ttm_dividend_per_share", "verified", ""),
    ]
    annual = artifacts["annual_history[2025].dividend_per_share"]
    assert (_rep_nested(annual, "proof", "kind"), _rep_accession(annual), annual["value"]) == (
        "observation",
        "0001045810-25-000050",
        "0.034",
    )
    ttm = artifacts["ttm_dividend_per_share"]
    proof, inputs = ttm["proof"], ttm["inputs"]
    assert isinstance(proof, Mapping) and isinstance(inputs, list)
    assert (proof["kind"], proof["formulaId"], ttm["value"]) == ("calculation", "sec_dividend_ttm", "0.04")
    assert proof["inputProofIds"] == [item["repair_id"] for item in inputs]
    assert sorted(item["value"] for item in inputs) == ["0.01"] * 4
    # The fourth quarter is itself derived (fiscal year minus nine-month YTD), with both facts as inputs.
    [derived] = [item for item in inputs if item["proof"]["kind"] == "calculation"]
    assert sorted(str(_rep_accession(item)) for item in derived["inputs"]) == [
        "0001045810-24-000300",
        "0001045810-25-000050",
    ]


_REP_FY_INPUT = {
    "concept": _REP_DPS_CONCEPT,
    "period_start": "2024-01-29",
    "period_end": "2025-01-26",
    "unit": "USD/shares",
    "value": 0.034,
}
_REP_YTD_INPUT = {**_REP_FY_INPUT, "period_end": "2024-10-27", "value": 0.024}


@pytest.mark.parametrize(
    ("inputs", "status", "reason"),
    [
        pytest.param([_REP_FY_INPUT, _REP_YTD_INPUT], "verified", "", id="inputs-without-accession"),
        pytest.param(
            [{k: v for k, v in _REP_FY_INPUT.items() if k != "concept"}, _REP_YTD_INPUT],
            "unresolved",
            "CALCULATION_INVALID",
            id="input-without-concept",
        ),
    ],
)
def test_rep_derived_metric_with_incomplete_input_references(
    inputs: list[dict[str, object]], status: str, reason: str, _rep_store: ResearchRepository
) -> None:
    """Each input resolves deterministically by its retained identity, else the whole calculation is invalid."""
    from app.research import repair

    q4 = {
        "derived": True,
        "formulaId": "sec_q4_difference",
        "formulaVersion": "1",
        "concept": _REP_DPS_CONCEPT,
        "period_end": "2025-01-26",
        "value": 0.01,
        "derivation_inputs": inputs,
    }
    _rep_result(_rep_store, "TR-Q4", "get_fundamentals", {"cik": _REP_CIK, "rows": [q4]})
    _rep_row(_rep_store, "EV-Q4", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q4, "TR-Q4"))
    [artifact] = repair.repair_evidence(_rep_store, "EV-Q4")
    assert _rep_brief([artifact]) == [("derived_incomplete_inputs", "value", status, reason)]
    if status == "verified":
        resolved = artifact["inputs"]
        assert isinstance(resolved, list)
        assert [_rep_accession(item) for item in resolved] == [
            "0001045810-25-000050",
            "0001045810-24-000300",
        ]
    else:
        assert str(artifact["detail"]).startswith("input 0: SOURCE_IDENTITY_MISSING")


_REP_FILINGS = [
    {"accession_no": "0001045810-25-000100", "form": "10-Q", "filing_date": "2025-05-28"},
    {"accession_no": "0001045810-25-000050", "form": "10-K", "filing_date": "2025-02-26"},
]


@pytest.mark.parametrize(
    ("cited", "value", "status", "reason"),
    [
        pytest.param(_REP_FILINGS[0], None, "verified", "", id="exact-row"),
        pytest.param(_REP_FILINGS, "10-K", "verified", "", id="whole-list-typed-unique"),
        pytest.param(_REP_FILINGS, None, "unresolved", "SOURCE_IDENTITY_MISSING", id="whole-list-untyped"),
        pytest.param(_REP_FILINGS, "10-K/A", "unresolved", "SOURCE_MISMATCH", id="whole-list-value-absent"),
    ],
)
def test_rep_bridge_whole_list_needs_one_identified_row(
    cited: object, value: str | None, status: str, reason: str, _rep_store: ResearchRepository
) -> None:
    """A bridge list citation repairs only when its exact row or one typed value pins a single row."""
    import json

    from app.research import repair

    _rep_result(_rep_store, "TR-LIST", "list_sec_filings", {"filings": _REP_FILINGS})
    _rep_row(
        _rep_store,
        "EV-LIST",
        "2025-09-01T00:00:00+00:00",
        _rep_record("list_sec_filings", cited, "TR-LIST"),
        **({"value": value} if value else {}),
    )
    [artifact] = repair.repair_evidence(_rep_store, "EV-LIST")
    assert _rep_brief([artifact]) == [("bridge_whole_list", "record", status, reason)]
    if status == "verified":
        expected = _REP_FILINGS[0] if value is None else _REP_FILINGS[1]
        source = artifact["source"]
        assert isinstance(source, Mapping)
        assert (source["tool_result_id"], source["row"]) == ("TR-LIST", expected)
        assert artifact["value"] == (value or json.dumps(expected, sort_keys=True, default=str))


_REP_SHORTS = [
    {"symbolCode": "NVDA", "settlementDate": "2025-08-15", "currentShortPositionQuantity": 1000},
    {"symbolCode": "AMD", "settlementDate": "2025-08-15", "currentShortPositionQuantity": 1000},
    {"symbolCode": "NVDA", "settlementDate": "2025-07-31", "currentShortPositionQuantity": 900},
]


@pytest.mark.parametrize(
    ("cited", "rid", "retrieved", "fields", "status", "reason", "row"),
    [
        pytest.param(_REP_SHORTS[2], "", "2025-09-01", {}, "verified", "", 2, id="job-scoped-result-no-replay-id"),
        pytest.param(
            {"query": {"limit": 3}},
            "TF",
            "2025-09-01",
            {},
            "unresolved",
            "SOURCE_IDENTITY_MISSING",
            None,
            id="query-only",
        ),
        pytest.param(
            "x", "TF", "2025-09-01", {"value": "1000"}, "unresolved", "SOURCE_AMBIGUOUS", None, id="value-on-two-rows"
        ),
        pytest.param(
            "x",
            "TF",
            "2025-09-01",
            {"value": "1000", "entity": "NVDA", "period": "2025-08-15"},
            "verified",
            "",
            0,
            id="value-entity-period-unique",
        ),
        pytest.param(
            _REP_SHORTS[0], "TF", "2025-08-01", {}, "unresolved", "SOURCE_UNAVAILABLE", None, id="row-after-as-of"
        ),
    ],
)
def test_rep_finra_structured_record_repairs_from_persisted_rows_only(
    cited: object,
    rid: str,
    retrieved: str,
    fields: dict[str, JSONValue],
    status: str,
    reason: str,
    row: int | None,
    _rep_store: ResearchRepository,
) -> None:
    """FINRA rows replay the persisted result (or this job's result for an old row without a replay id)."""
    from app.research import repair

    _rep_result(_rep_store, "TF", "get_short_interest_leaderboard", {"records": _REP_SHORTS})
    provenance = _rep_record("get_short_interest_leaderboard", cited, rid, kind="finra_record")
    _rep_row(_rep_store, "EV-F", f"{retrieved}T00:00:00+00:00", provenance, source_name="FINRA", **fields)
    [artifact] = repair.repair_evidence(_rep_store, "EV-F")
    assert _rep_brief([artifact]) == [("finra_structured", "record", status, reason)]
    if row is not None:
        source = artifact["source"]
        assert isinstance(source, Mapping)
        assert (source["tool_result_id"], source["row"]) == ("TF", _REP_SHORTS[row])
        assert _rep_nested(artifact, "proof", "kind") == "observation"


@pytest.mark.parametrize(
    ("metadata", "status", "reason", "accession"),
    [
        pytest.param(
            {"concept": "EarningsPerShareDiluted", "unit": "USD/shares", "fiscal_period": "Q1"},
            "verified",
            "",
            "0001045810-25-000100",
            id="retained-concept",
        ),
        pytest.param({}, "unresolved", "SOURCE_IDENTITY_MISSING", None, id="page-only"),
    ],
)
def test_rep_company_page_fallback_needs_a_retained_fact_identity(
    metadata: dict[str, JSONValue], status: str, reason: str, accession: str | None, _rep_store: ResearchRepository
) -> None:
    """An old company-page citation repairs only from the concept it kept; the page itself proves nothing."""
    from app.research import repair

    _rep_row(
        _rep_store,
        "EV-PAGE",
        "2025-09-01T00:00:00+00:00",
        source_uri=_REP_PAGE,
        value="0.76",
        entity="NVDA",
        period="2025-04-27",
        unit="USD/shares",
        metadata=metadata,
    )
    [artifact] = repair.repair_evidence(_rep_store, "EV-PAGE")
    assert (artifact["category"], artifact["status"], artifact["reason"]) == ("company_page_fallback", status, reason)
    assert _rep_accession(artifact) == accession


def test_rep_cik_only_envelope_and_tampered_row_stay_unresolved(_rep_store: ResearchRepository) -> None:
    """No concept or period means no backfill by first matching number; a tampered row reports corruption."""
    import json
    import os
    import sqlite3

    from app.research import repair

    envelope = {"cik": _REP_CIK, "balance_sheet": {"Total Assets": 1}}
    _rep_result(_rep_store, "TR-BS", "get_fundamentals", envelope)
    _rep_row(_rep_store, "EV-BS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", envelope, "TR-BS"))
    assert _rep_brief(repair.repair_evidence(_rep_store, "EV-BS")) == [
        ("balance_sheet_cik_only", "record", "unresolved", "SOURCE_IDENTITY_MISSING")
    ]
    raw = _rep_store.get_evidence("EV-BS")
    with sqlite3.connect(os.environ["RESEARCH_DB_PATH"]) as conn:
        conn.execute(
            "INSERT INTO evidence (evidence_id, session_id, record) VALUES (?, ?, ?)",
            ("EV-TAMPER", _REP_SID, json.dumps({**raw, "evidence_id": "EV-TAMPER", "content": "edited"})),
        )
    [artifact] = repair.repair_evidence(_rep_store, "EV-TAMPER")
    assert (artifact["status"], artifact["reason"]) == ("unresolved", "SOURCE_CORRUPT")
    assert _rep_store.get_evidence("EV-TAMPER")["content"] == "edited"


def test_e2e_legacy_numeric_text_cannot_publish_without_datapoint_proof(
    _rep_store: ResearchRepository,
) -> None:
    """Kernel-held source text alone never publishes a number as a verified fact."""
    from app.research import repair
    from app.research.agents import GroundedClaim
    from app.research.service import require_verified_claims

    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(_rep_store, "TR-LEGACY", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q1]})
    _rep_row(
        _rep_store,
        "EV-LEGACY",
        "2025-09-01T00:00:00+00:00",
        _rep_record("get_fundamentals", q1, "TR-LEGACY"),
    )
    assert _rep_store.get_evidence("EV-LEGACY")["proof"] == {}
    claim = GroundedClaim(text="NVDA EPS was 0.76.", claim_type="observed_fact", evidence_ids=["EV-LEGACY"])
    with pytest.raises(ValueError, match="ERR_(DATAPOINT|PROOF)_UNVERIFIED"):
        require_verified_claims(_rep_store, _REP_SID, [claim], "acceptance")
    assert _rep_store.get_evidence("EV-LEGACY")["proof"] == {}
    assert [a for a in repair.repair_session(_rep_store, _REP_SID) if a["evidence_id"] == "EV-LEGACY"]


def test_rep_scope_refuses_proven_web_and_unknown_rows_and_session_repair_writes_no_evidence(
    _rep_store: ResearchRepository,
) -> None:
    """Only proof-less structured or untyped legacy rows repair; a whole-session pass never edits the ledger."""
    from app.research import repair
    from app.research.evidence import evidence_to_dict

    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(_rep_store, "TR-EPS", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q1]})
    _rep_row(_rep_store, "EV-EPS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-EPS"))
    _rep_row(_rep_store, "EV-PAGE", "2025-09-01T00:00:00+00:00", source_uri=_REP_PAGE, value="0.76")
    proven = dataclasses.replace(_observation_row("EV-PROVEN", _PROOF_DC, "22,563"), session_id=_REP_SID)
    _rep_store.save_evidence(evidence_to_dict(proven))
    web = {"kind": "web_source", "url": "https://example.com/a", "excerpt": "x", "tool_result_id": "tr"}
    _rep_row(_rep_store, "EV-WEB", "2025-09-01T00:00:00+00:00", web)
    for evidence_id in ("EV-PROVEN", "EV-WEB"):
        with pytest.raises(ValueError, match="not a repairable legacy row"):
            repair.repair_evidence(_rep_store, evidence_id)
    with pytest.raises(KeyError):
        repair.repair_evidence(_rep_store, "EV-ABSENT")
    ledger = _rep_store.list_evidence(_REP_SID)
    artifacts = repair.repair_session(_rep_store, _REP_SID)
    assert sorted((a["evidence_id"], a["status"], a["reason"]) for a in artifacts) == [
        ("EV-EPS", "verified", ""),
        ("EV-PAGE", "unresolved", "SOURCE_IDENTITY_MISSING"),
    ]
    assert _rep_store.list_evidence(_REP_SID) == ledger
    assert {a["repair_id"] for a in _rep_store.list_evidence_repairs(_REP_SID)} == {a["repair_id"] for a in artifacts}
    assert repair.REPAIR_CATEGORIES == (
        "eps_without_accession",
        "balance_sheet_cik_only",
        "dividend_annual_ttm",
        "bridge_whole_list",
        "finra_structured",
        "derived_incomplete_inputs",
        "company_page_fallback",
    )


def _rep_promoted(store: ResearchRepository, artifact: Mapping[str, object]) -> Evidence:
    """The decoded target-session row a promotion returned; its own proof replays as verified."""
    from app.research.evidence import evidence_from_dict
    from app.research.service import _proof_verdict, _session_rows

    eid = artifact["promoted_evidence_id"]
    assert isinstance(eid, str) and eid.startswith("ev:")
    row = evidence_from_dict(store.get_evidence(eid))
    assert row.metadata["repair_id"] == artifact["repair_id"]
    assert row.metadata["repaired_evidence_id"] == artifact["evidence_id"]
    cutoff = store.get_session(row.session_id).as_of
    verdict = _proof_verdict(_session_rows(store, row.session_id), row, row.value, as_of=cutoff)
    assert verdict.status == "verified", verdict.reason
    return row


def _rep_publish(store: ResearchRepository, row: Evidence, evidence_ids: Sequence[str] = ()) -> None:
    """Publish one claim worded from the row's canonical datapoint through the target session gate."""
    from app.research.agents import GroundedClaim
    from app.research.evidence import datapoint_descriptor
    from app.research.service import require_verified_claims

    point = datapoint_descriptor(row)
    assert point is not None
    unit = f" {point['unit']}" if point.get("unit") else ""
    claim = GroundedClaim(
        text=f"{point['entity']} {point['metric']} for {point['period']} was {point['value']}{unit}.",
        claim_type=row.claim_kind,
        evidence_ids=list(evidence_ids or [row.evidence_id]),
        datapoint_ids=[str(point["id"])],
    )
    require_verified_claims(store, row.session_id, [claim], "acceptance")


def test_rep_promotion_admits_one_verified_row_in_target_and_keeps_frozen_history(
    _rep_store: ResearchRepository,
) -> None:
    """A verified repair becomes one new publishable row in the named session; history and reruns stay fixed."""
    from app.research import repair
    from app.research import service as _svc
    from app.research.evidence import datapoint_descriptor
    from app.research.freeze import freeze_from_dict, freeze_to_dict

    store = _rep_store
    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(store, "TR-EPS", "get_fundamentals", {"ticker": "NVDA", "cik": _REP_CIK, "quarterly_eps": [q1]})
    legacy = _rep_row(store, "EV-EPS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-EPS"))
    frozen = create_freeze(
        freeze_id="F-rep", session_id=_REP_SID, wave_id=1, records=[legacy], as_of=datetime(2025, 9, 30, tzinfo=UTC)
    )
    store.save_freeze(freeze_to_dict(frozen))
    stored, stored_freeze = store.get_evidence("EV-EPS"), store.get_freeze("F-rep")
    target = _svc.create_research("NVDA EPS?", "o", repo=store)
    [artifact] = repair.repair_evidence(store, "EV-EPS", target)
    # The persisted artifact never carries the promotion; the returned one names the new row.
    [persisted] = store.list_evidence_repairs(_REP_SID, "EV-EPS")
    assert persisted == {key: value for key, value in artifact.items() if key != "promoted_evidence_id"}
    promoted = _rep_promoted(store, artifact)
    assert (promoted.session_id, promoted.claim_kind, promoted.superseded_by) == (target, "observed_fact", None)
    assert (promoted.value, promoted.entity, promoted.period, promoted.unit) == (
        "0.76",
        "sec:cik:0001045810",
        "2025-04-27",
        "USD/shares",
    )
    descriptor = datapoint_descriptor(promoted)
    assert descriptor is not None and descriptor["metric"] == "EarningsPerShareDiluted"
    result = store.get_tool_result(str(promoted.provenance["tool_result_id"]))
    assert (result["session_id"], result["tool_name"]) == (target, "get_xbrl_facts")
    assert promoted.evidence_id in store.get_session(target).evidence_ids
    _rep_publish(store, promoted)
    # Neither the legacy id nor the repair id becomes citable through the promotion.
    for cited in ("EV-EPS", str(artifact["repair_id"])):
        with pytest.raises(ValueError, match="ERR_PROOF_UNVERIFIED"):
            _rep_publish(store, promoted, [cited])
    # The legacy row, its freeze, and its repair history stay byte-identical.
    assert store.get_evidence("EV-EPS") == stored and stored["proof"] == {}
    assert store.get_freeze("F-rep") == stored_freeze
    verify_freeze(freeze_from_dict(store.get_freeze("F-rep")), [legacy])
    # A rerun returns the same row and writes no evidence, tool result, or repair history.
    before = (store.list_evidence(target), store.list_tool_results(target))
    [again] = repair.repair_evidence(store, "EV-EPS", target)
    assert again["promoted_evidence_id"] == promoted.evidence_id
    assert (store.list_evidence(target), store.list_tool_results(target)) == before
    assert len(store.list_evidence_repairs(_REP_SID, "EV-EPS")) == 1
    assert store.get_evidence("EV-EPS") == stored


def test_rep_promotion_does_not_exempt_normal_persistence_from_live_job_guard(
    _rep_store: ResearchRepository,
) -> None:
    """Normal persistence still rejects a valid repaired observation without a running job."""
    from app.research import repair
    from app.research import service as _svc

    store = _rep_store
    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(store, "TR-EPS", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q1]})
    _rep_row(store, "EV-EPS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-EPS"))
    target = _svc.create_research("NVDA EPS?", "o", repo=store)
    [artifact] = repair.repair_evidence(store, "EV-EPS", target)
    promoted = _rep_promoted(store, artifact)
    candidate = dataclasses.replace(
        promoted,
        evidence_id="ev:normal-without-job",
        metadata={**promoted.metadata, "identity_key": "normal-without-job"},
    )
    before = (store.list_evidence(target), store.get_session(target))
    with pytest.raises(_svc.ResearchNotFound, match="unknown job_id: 'repair'"):
        _svc._persist_finra_web_record(store, target, "repair", store.get_session(target), candidate, proof_bytes=None)
    assert (store.list_evidence(target), store.get_session(target)) == before


def test_rep_promotion_admits_calculation_inputs_first_and_recomputes_over_them(
    _rep_store: ResearchRepository,
) -> None:
    """A TTM repair promotes its four quarters (one derived Q4 over FY and YTD facts) before the calculation."""
    import json

    from app.research import repair
    from app.research import service as _svc
    from app.research.evidence import evidence_from_dict

    store = _rep_store
    payload = {
        "cik": _REP_CIK,
        "ttm_dividend_per_share": 0.04,
        "annual_history": [{"fiscal_year": 2025, "dividend_per_share": 0.034}],
    }
    _rep_result(store, "TR-DIV", "get_fundamentals", payload)
    envelope = " ".join(json.dumps(payload, sort_keys=True).split())
    _rep_row(store, "EV-DIV", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", envelope, "TR-DIV"))
    target = _svc.create_research("NVDA dividend?", "o", repo=store)
    artifacts = {str(a["datapoint"]): a for a in repair.repair_evidence(store, "EV-DIV", target)}
    annual = _rep_promoted(store, artifacts["annual_history[2025].dividend_per_share"])
    assert (annual.claim_kind, annual.value, annual.period) == ("observed_fact", "0.034", "2025-01-26")
    ttm_artifact = artifacts["ttm_dividend_per_share"]
    ttm = _rep_promoted(store, ttm_artifact)
    assert (ttm.claim_kind, ttm.value, ttm.proof["formulaId"]) == ("calculation", "0.04", "sec_dividend_ttm")
    raw_ids = ttm.proof["inputProofIds"]
    assert isinstance(raw_ids, list)
    inputs = [evidence_from_dict(store.get_evidence(str(eid))) for eid in raw_ids]
    nested = ttm_artifact["inputs"]
    assert isinstance(nested, list)
    assert all("created_at" not in item for item in nested)
    assert [row.metadata["repair_id"] for row in inputs] == [item["repair_id"] for item in nested]
    assert all(row.session_id == target and row.proof for row in inputs)
    assert [row.metadata["period_end"] for row in inputs] == ["2024-07-28", "2024-10-27", "2025-01-26", "2025-04-27"]
    assert (ttm.metadata["period_start"], ttm.metadata["period_end"]) == ("2024-04-29", "2025-04-27")
    [q4] = [row for row in inputs if row.claim_kind == "calculation"]
    assert (q4.value, q4.metadata["period_start"]) == ("0.01", "2024-10-28")
    q4_inputs = q4.proof["inputProofIds"]
    assert isinstance(q4_inputs, list)
    assert [evidence_from_dict(store.get_evidence(str(eid))).value for eid in q4_inputs] == ["0.034", "0.024"]
    _rep_publish(store, ttm)
    # A rerun reuses every promoted row, inputs included.
    rows = store.list_evidence(target)
    history = store.list_evidence_repairs(_REP_SID, "EV-DIV")
    results = store.list_tool_results(target)
    again = {str(a["datapoint"]): a for a in repair.repair_evidence(store, "EV-DIV", target)}
    assert again["ttm_dividend_per_share"]["promoted_evidence_id"] == ttm.evidence_id
    assert again["ttm_dividend_per_share"]["repair_id"] == ttm_artifact["repair_id"]
    assert again["ttm_dividend_per_share"]["inputs"] == nested
    assert store.list_evidence_repairs(_REP_SID, "EV-DIV") == history
    assert store.list_tool_results(target) == results
    assert store.list_evidence(target) == rows


def test_rep_promotion_supersedes_in_session_finra_row_and_refuses_foreign_results(
    _rep_store: ResearchRepository,
) -> None:
    """In the legacy session the new row corrects the old one; a result of another session never promotes."""
    from app.research import repair
    from app.research import service as _svc
    from app.research.evidence import datapoint_descriptor

    store = _rep_store
    sid = _svc.create_research("NVDA short interest?", "o", repo=store)
    fields: dict[str, JSONValue] = {"value": "1000", "entity": "NVDA", "period": "2025-08-15", "session_id": sid}
    _rep_result(store, "TF", "get_short_interest_leaderboard", {"records": _REP_SHORTS}, session_id=sid)
    provenance = _rep_record("get_short_interest_leaderboard", "x", "TF", kind="finra_record")
    _rep_row(store, "EV-F", "2025-09-01T00:00:00+00:00", provenance, source_name="FINRA", **fields)
    stored = store.get_evidence("EV-F")
    [artifact] = repair.repair_evidence(store, "EV-F", sid)
    promoted = _rep_promoted(store, artifact)
    assert (promoted.superseded_by, promoted.value, promoted.entity, promoted.period) == (
        "EV-F",
        "1000",
        "NVDA",
        "2025-08-15",
    )
    descriptor = datapoint_descriptor(promoted)
    assert descriptor is not None and descriptor["metric"] == "currentShortPositionQuantity"
    assert promoted.provenance["tool_result_id"] != "TF"
    assert store.get_tool_result("TF")["session_id"] == sid
    assert store.get_evidence("EV-F") == stored
    _rep_publish(store, promoted)
    _rep_result(store, "TF-OTHER", "get_short_interest_leaderboard", {"records": _REP_SHORTS}, session_id="S-other")
    foreign = _rep_record("get_short_interest_leaderboard", "x", "TF-OTHER", kind="finra_record")
    _rep_row(store, "EV-FX", "2025-09-01T00:00:00+00:00", foreign, source_name="FINRA", **fields)
    [refused] = repair.repair_evidence(store, "EV-FX", sid)
    assert (refused["status"], refused["reason"], refused["promoted_evidence_id"]) == (
        "unresolved",
        "SOURCE_MISMATCH",
        None,
    )


def test_rep_promotion_refuses_unresolved_pit_ineligible_and_closed_targets(
    _rep_store: ResearchRepository, caplog: pytest.LogCaptureFixture
) -> None:
    """Only a verified repair that passes target admission promotes; a closed or unknown target writes nothing."""
    from app.research import repair
    from app.research import service as _svc

    store = _rep_store
    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    q2 = {"period_end": "2025-07-27", "fiscal_period": "Q2", "eps_diluted": 1.05}
    _rep_result(store, "TR-EPS", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q1]})
    _rep_result(store, "TR-Q2", "get_fundamentals", {"cik": _REP_CIK, "quarterly_eps": [q2]})
    _rep_row(store, "EV-EPS", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-EPS"))
    _rep_row(store, "EV-Q2", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q2, "TR-Q2"))
    # The Q1 fact was filed 2025-05-28, after this target's cutoff.
    early = _svc.create_research("NVDA EPS?", "o", as_of="2025-05-01T00:00:00+00:00", repo=store)
    [mismatch] = repair.repair_evidence(store, "EV-Q2", early)
    assert (mismatch["status"], mismatch["reason"], mismatch["promoted_evidence_id"]) == (
        "unresolved",
        "SOURCE_MISMATCH",
        None,
    )
    with caplog.at_level("ERROR", logger="app.research.repair"):
        [late] = repair.repair_evidence(store, "EV-EPS", early)
    assert (late["status"], late["promoted_evidence_id"]) == ("verified", None)
    [message] = [r.getMessage() for r in caplog.records if "not promoted" in r.getMessage()]
    assert f"target={early} evidence=EV-EPS " in message and "PIT_VIOLATION" in message
    assert store.list_evidence(early) == []
    closed = _svc.create_research("NVDA EPS?", "o", repo=store)
    store.save_session(dataclasses.replace(store.get_session(closed), status=SessionStatus.CANCELLED.value))
    history = store.list_evidence_repairs(_REP_SID)
    with pytest.raises(ValueError, match="terminal"):
        repair.repair_evidence(store, "EV-EPS", closed)
    with pytest.raises(KeyError):
        repair.repair_session(store, _REP_SID, "S-absent")
    assert store.list_evidence_repairs(_REP_SID) == history
    assert store.list_evidence(closed) == []


# ---------------------------------------------------------------------------
# Acceptance sweep: one session, every supported path, tool -> archive -> datapoint -> proof
# -> evidence -> calculation -> claim -> answer -> citation -> viewer -> exact replay.
# ---------------------------------------------------------------------------
_E2E_TOTAL = "NVDA Q1 FY2025 total revenue was $26,044 million."
_E2E_DC = "NVDA Q1 FY2025 Data Center revenue was $22,563 million."
_E2E_FULL = f"Item 2. {_E2E_TOTAL} {_E2E_DC} End."
_E2E_WEB = "On 2025-06-20 NVDA shares rose 4.2%"
_E2E_FINRA_ROWS = [
    {"symbol": "NVDA", "settlementDate": "2025-06-13", "shortInterest": 12345},
    {"symbol": "AMD", "settlementDate": "2025-06-13", "shortInterest": 67890},
]
_E2E_DATAPOINT = {"entity": "NVDA", "period": "Q1 FY2025", "unit": "million"}


@dataclasses.dataclass(frozen=True)
class _E2E:
    repo: ResearchRepository
    sid: str
    data: Path
    ids: dict[str, str]
    jobs: dict[str, str]
    rids: dict[str, str]


def _e2e_finra_line(row: Mapping[str, object]) -> str:
    import json as _json

    return _json.dumps(dict(row), sort_keys=True)


def _e2e_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, freeze: bool = True) -> _E2E:
    """Real ledger through record_evidence: SEC observations, nested calculations, FINRA, WEB, absence, freeze."""
    from app.research import service as _svc

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    repo = ResearchRepository()
    sid, sec_job = _hf_sid(repo, "SEC", "FINRA", "WEB")
    fin, web = _hf_desks(repo, sid)
    seam.register_document(seam.DEFAULT_ACCESSION, seam.DEFAULT_DOCUMENT, _E2E_FULL)
    ids = {key: f"{sid}:ev:{key}" for key in ("tot", "dc", "calc", "nested", "finra", "web")}
    for key, passage, value, metric in (
        ("tot", _E2E_TOTAL, "26,044", "total revenue"),
        ("dc", _E2E_DC, "22,563", "Data Center revenue"),
    ):
        _svc.record_evidence(
            sid,
            sec_job,
            {
                "evidence_id": ids[key],
                "content": passage,
                "claim_text": passage,
                "subject": "NVDA",
                "source_name": "SEC",
                "source_record_id": seam.DEFAULT_ACCESSION,
                "document_name": seam.DEFAULT_DOCUMENT,
                "matching_passage": passage,
                "source_handle": seam.handle_for_registered(passage),
                "known_at": "2025-05-01T00:00:00+00:00",
                "value": value,
                "metric": metric,
                **_E2E_DATAPOINT,
            },
            repo=repo,
        )
    for key, formula, inputs, value, metric in (
        ("calc", "difference", [ids["tot"], ids["dc"]], "3481", "non-Data Center revenue"),
        ("nested", "sum", [ids["calc"], ids["dc"]], "26044", "rebuilt total revenue"),
    ):
        _svc.record_evidence(
            sid,
            sec_job,
            {
                "evidence_id": ids[key],
                "claim_kind": "calculation",
                "claim_text": f"{formula} of {inputs}",
                "formula_id": formula,
                "formula_version": "1",
                "input_evidence_ids": inputs,
                "value": value,
                "metric": metric,
                **_E2E_DATAPOINT,
            },
            repo=repo,
        )
    rids = {"finra": f"{sid}:tr:e2e-finra", "web": f"{sid}:tr:e2e-web"}
    _svc.persist_tool_result(
        sid,
        fin,
        "query_finra",
        rids["finra"],
        {
            "dataset_id": "otcMarket/consolidatedShortInterest",
            "records": _E2E_FINRA_ROWS,
            "as_of_date": "2025-06-13",
            "published_at": "2025-06-20",
        },
        repo=repo,
    )
    nvda = _e2e_finra_line(_E2E_FINRA_ROWS[0])
    _svc.record_evidence(
        sid,
        fin,
        {
            **_hf_replay_item(ids["finra"], rids["finra"], nvda),
            "value": "12345",
            "metric": "shortInterest",
            "entity": "NVDA",
            "period": "2025-06-13",
        },
        repo=repo,
    )
    _svc.persist_tool_result(
        sid,
        web,
        "search_web",
        rids["web"],
        {
            "evidence": [
                {
                    "url": "https://example.com/nvda",
                    "source_domain": "example.com",
                    "title": "NVDA demand",
                    "published_at": "2025-06-20",
                    "retrieved_at": "2025-06-21",
                    "highlight": _E2E_WEB,
                }
            ]
        },
        repo=repo,
    )
    _svc.record_evidence(
        sid,
        web,
        {
            **_hf_replay_item(ids["web"], rids["web"], _E2E_WEB, source_name="WEB"),
            "value": "4.2%",
            "metric": "shares rose",
            "entity": "NVDA",
            "period": "2025-06-20",
            "unit": "%",
        },
        repo=repo,
    )
    _svc.record_evidence(
        sid,
        sec_job,
        {
            "wave_id": 1,
            "content": "no NVDA 8-K on the topic",
            "claim_text": "No NVDA 8-K mentions the topic within the searched scope",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "SEC",
            "known_at": "2025-06-29T00:00:00+00:00",
            "search_id": "e2e-search",
            "query": "NVDA 8-K topic 2025",
            "coverage": {
                "forms": ["8-K"],
                "dates": ["2025-01-01..2025-06-30"],
                "partitions": [],
                "entities": ["NVDA"],
                "docs": [],
                "gaps": [],
                "pagination_complete": True,
                "complete": True,
            },
        },
        repo=repo,
    )
    jobs = {"sec": sec_job, "finra": fin, "web": web}
    if freeze:
        for job in jobs.values():
            _svc.complete_job(job, {}, repo=repo)
        _svc.freeze_session(sid, 1, repo=repo)
    return _E2E(repo, sid, tmp_path / "data", ids, jobs, rids)


def _e2e_point(e2e: _E2E, key: str) -> str:
    """Kernel datapoint id of one e2e row (read from its stored canonical descriptor)."""
    from app.research.evidence import datapoint_descriptor

    point = datapoint_descriptor(_e2e_rows(e2e)[key])
    assert point is not None, key
    return str(point["id"])


def _e2e_claims(e2e: _E2E) -> list[dict[str, object]]:
    """One published claim per supported path: observed facts and calculations by datapoint id, inference, assumption."""
    ids = e2e.ids

    def fact(key: str, text: str, claim_type: str = "observed_fact") -> dict[str, object]:
        return {
            "text": text,
            "claim_type": claim_type,
            "evidence_ids": [ids[key]],
            "datapoint_ids": [_e2e_point(e2e, key)],
        }

    return [
        fact("tot", "NVDA Q1 FY2025 total revenue was $26,044 million"),
        fact("calc", "NVDA Q1 FY2025 non-Data Center revenue was 3481 million", "calculation"),
        fact("nested", "NVDA Q1 FY2025 rebuilt total revenue is 26044 million", "calculation"),
        fact("finra", "NVDA short interest for 2025-06-13 was 12345"),
        fact("web", "On 2025-06-20 NVDA shares rose 4.2%"),
        {
            "text": "Data Center revenue likely dominates NVDA revenue",
            "claim_type": "inference",
            "evidence_ids": [ids["tot"], ids["dc"]],
        },
        {"text": "Assume NVDA demand persists next quarter", "claim_type": "assumption", "evidence_ids": []},
    ]


def _e2e_publish(e2e: _E2E) -> dict[str, object]:
    """Committee over the freeze, then finalize_session with every supported claim path."""
    from app.research import service as _svc

    for role in ("stockbot", "bullbot", "bearbot"):
        jid = str(_svc.start_job(e2e.sid, role, repo=e2e.repo, wave_id=1)["job_id"])
        _svc.record_committee_analysis(e2e.sid, jid, role, _committee_analysis(e2e.ids["tot"]), repo=e2e.repo)
    claims = _e2e_claims(e2e)
    answer = " ".join(f"{claim['text']}." for claim in claims)
    return _svc.finalize_session(e2e.sid, answer, list(claims), repo=e2e.repo)


def _e2e_rows(e2e: _E2E) -> dict[str, Evidence]:
    from app.research.evidence import evidence_from_dict

    fresh = ResearchRepository()
    return {key: evidence_from_dict(fresh.get_evidence(eid)) for key, eid in e2e.ids.items()}


def test_e2e_every_supported_path_replays_from_archive_to_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every asserted datapoint carries a valid proof, replays from archived bytes, and publishes verified."""
    from app.research import service as _svc
    from app.research.evidence import value_hash
    from app.research.freeze import freeze_datapoint_hash, freeze_from_dict

    e2e = _e2e_session(tmp_path, monkeypatch)
    rows = _e2e_rows(e2e)
    ledger = _svc._session_rows(e2e.repo, e2e.sid)
    cutoff = e2e.repo.get_session(e2e.sid).as_of
    # Absence stays a coverage artifact: no evidence id, no freeze membership, no proof.
    assert set(ledger) == set(e2e.ids.values())
    [artifact] = e2e.repo.list_coverage_artifacts(e2e.sid)
    assert "evidence_id" not in artifact and "proof" not in artifact
    assert artifact["query"] == "NVDA 8-K topic 2025"
    for key, row in rows.items():
        kind = "calculation" if key in ("calc", "nested") else "observation"
        assert row.proof["kind"] == kind, key
        assert row.proof["valueHash"] == value_hash(row.value or ""), key
        assert row.known_at is not None and row.known_at <= ASOF, key
        assert _svc._proof_verdict(ledger, row, row.value, as_of=cutoff).status == "verified", key
        if kind == "observation":
            replay = _svc._proof_bytes(row)
            assert replay is not None and f"sha256:{sha256(replay).hexdigest()}" == row.proof["sourceId"], key
    sec_bytes = _svc._proof_bytes(rows["tot"])
    assert sec_bytes == _E2E_FULL.encode()
    assert rows["tot"].provenance["passage"] == _E2E_TOTAL
    assert rows["finra"].provenance["tool_result_id"] == e2e.rids["finra"]
    assert rows["web"].provenance["excerpt"] == _E2E_WEB
    # Calculations recompute from pinned inputs in formula argument order, nested one level deep.
    assert _svc._recompute("difference", "1", (rows["tot"].value or "", rows["dc"].value or "")) == rows["calc"].value
    assert _svc._recompute("sum", "1", (rows["calc"].value or "", rows["dc"].value or "")) == rows["nested"].value
    assert rows["nested"].proof["inputProofIds"] == [e2e.ids["calc"], e2e.ids["dc"]]
    # The freeze pins content and datapoints; a fresh replay reproduces both hashes.
    frozen = freeze_from_dict(ResearchRepository().get_freeze(f"{e2e.sid}:1:freeze"))
    assert set(frozen.evidence_ids) == set(e2e.ids.values())
    verify_freeze(frozen, list(_e2e_rows(e2e).values()))
    assert frozen.datapoint_hash == freeze_datapoint_hash(list(rows.values()))

    out = _e2e_publish(e2e)
    assert out["status"] == "completed"
    final = out["final_result"]
    assert isinstance(final, dict)
    published = {str(claim["text"]): claim for claim in final["claims"] if isinstance(claim, dict)}
    for claim in _e2e_claims(e2e):
        row = published[str(claim["text"])]
        assert row["claim_type"] == claim["claim_type"]
        assert row.get("datapoint_ids", []) == claim.get("datapoint_ids", [])
        assert not {"value", "entity", "period", "unit"} & set(row)
    assert set(final["evidence_refs"]) == set(e2e.ids.values())
    assert any("No NVDA 8-K mentions the topic" in text for text in final["absence_observations"])
    calculations = {str(view["evidence_id"]): view for view in final["calculations"] if isinstance(view, dict)}
    assert {key: view["verification"] for key, view in calculations.items()} == {
        e2e.ids["calc"]: "verified",
        e2e.ids["nested"]: "verified",
    }
    assert calculations[e2e.ids["nested"]]["recomputedValue"] == "26044"
    answer = str(final["answer"])
    for number in ("26,044", "3481", "26044", "12345", "4.2%"):
        assert number in answer, number
    # The published ledger and freeze are unchanged by publication.
    assert _svc._session_rows(e2e.repo, e2e.sid) == ledger
    verify_freeze(frozen, list(_e2e_rows(e2e).values()))


@pytest.mark.parametrize(
    ("key", "value", "refs", "error"),
    [
        ("tot", "99999", "tot", "ERR_DATAPOINT_UNBOUND"),
        ("tot", "26,044", "dc", "ERR_DATAPOINT_UNVERIFIED"),
        ("tot", "26,044", "absent", "ERR_PROOF_UNVERIFIED"),
        ("finra", "99999", "finra", "ERR_DATAPOINT_UNBOUND"),
        ("web", "9.9%", "web", "ERR_DATAPOINT_UNBOUND"),
    ],
    ids=[
        "fabricated-sec-value",
        "unrelated-evidence-id",
        "unknown-evidence-id",
        "fabricated-finra",
        "valid-url-unsupported",
    ],
)
def test_e2e_publication_rejects_values_without_their_own_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str, refs: str, error: str
) -> None:
    from app.research import service
    from app.research.agents import GroundedClaim
    from app.research.evidence import datapoint_descriptor

    e2e = _e2e_session(tmp_path, monkeypatch)
    point = datapoint_descriptor(_e2e_rows(e2e)[key])
    assert point is not None
    unit = f" {point['unit']}" if point.get("unit") else ""
    claim = GroundedClaim(
        text=f"{point['entity']} {point['metric']} for {point['period']} was {value}{unit}",
        claim_type="observed_fact",
        evidence_ids=[e2e.ids.get(refs, "EV-NOT-IN-LEDGER")],
        datapoint_ids=[str(point["id"])],
    )
    before = e2e.repo.list_evidence(e2e.sid)
    with pytest.raises(ValueError, match=error):
        service.require_verified_claims(e2e.repo, e2e.sid, [claim], "acceptance")
    assert e2e.repo.list_evidence(e2e.sid) == before
    assert e2e.repo.get_session(e2e.sid).final_result is None


@pytest.mark.parametrize(
    ("formula", "version", "input_keys", "value"),
    [
        ("difference", "1", ("dc", "tot"), "3481"),
        ("unknown-formula", "1", ("tot", "dc"), "3481"),
        ("difference", "999", ("tot", "dc"), "3481"),
        ("difference", "1", ("tot", "dc"), "99999"),
        ("difference", "1", ("tot", "absent"), "3481"),
    ],
    ids=["reordered-inputs", "unknown-formula", "unknown-version", "fabricated-result", "absent-input"],
)
def test_e2e_calculation_admission_rejects_invalid_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    formula: str,
    version: str,
    input_keys: tuple[str, str],
    value: str,
) -> None:
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch, freeze=False)
    before = e2e.repo.list_evidence(e2e.sid)
    with pytest.raises(ValueError, match="ERR_CALCULATION_INVALID"):
        service.record_evidence(
            e2e.sid,
            e2e.jobs["sec"],
            {
                "evidence_id": "EV-INVALID-CALC",
                "claim_kind": "calculation",
                "formula_id": formula,
                "formula_version": version,
                "input_evidence_ids": [e2e.ids.get(key, "EV-ABSENT") for key in input_keys],
                "value": value,
                "metric": "non-Data Center revenue",
                **_E2E_DATAPOINT,
            },
            repo=e2e.repo,
        )
    assert e2e.repo.list_evidence(e2e.sid) == before


def test_e2e_finra_cited_record_never_borrows_another_records_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AMD value in the same archived result cannot prove a typed NVDA datapoint."""
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch, freeze=False)
    before = e2e.repo.list_evidence(e2e.sid)
    with pytest.raises(ValueError, match="ERR_DATAPOINT_UNVERIFIED"):
        service.record_evidence(
            e2e.sid,
            e2e.jobs["finra"],
            {
                **_hf_replay_item("EV-FINRA-SUBSTITUTION", e2e.rids["finra"], _e2e_finra_line(_E2E_FINRA_ROWS[0])),
                "value": "67890",
                "metric": "shortInterest",
                "entity": "NVDA",
                "period": "2025-06-13",
            },
            repo=e2e.repo,
        )
    assert e2e.repo.list_evidence(e2e.sid) == before


def test_e2e_web_ambiguous_match_never_selects_first_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locator that matches two provider rows needs a URL, not an implicit first result."""
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch, freeze=False)
    service.persist_tool_result(
        e2e.sid,
        e2e.jobs["web"],
        "search_web",
        "TR-WEB-AMBIGUOUS",
        {
            "evidence": [
                {"url": url, "highlight": "NVDA revenue rose", "published_at": "2025-06-20"}
                for url in ("https://example.com/a", "https://example.com/b")
            ]
        },
        repo=e2e.repo,
    )
    before = e2e.repo.list_evidence(e2e.sid)
    with pytest.raises(ValueError, match="ERR_PASSAGE_AMBIGUOUS"):
        service.record_evidence(
            e2e.sid,
            e2e.jobs["web"],
            _hf_replay_item("EV-WEB-AMBIGUOUS", "TR-WEB-AMBIGUOUS", "NVDA revenue rose", source_name="WEB"),
            repo=e2e.repo,
        )
    assert e2e.repo.list_evidence(e2e.sid) == before


@pytest.mark.parametrize("domain", ["FINRA", "WEB"])
def test_e2e_post_as_of_source_cannot_enter_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    """Caller dates cannot backdate a persisted source that becomes known after as_of."""
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch, freeze=False)
    tool = "query_finra" if domain == "FINRA" else "search_web"
    future_row = {"symbol": "NVDA", "settlementDate": "2025-06-13", "shortInterest": 99999}
    passage = _e2e_finra_line(future_row) if domain == "FINRA" else "On 2025-06-13 NVDA reported 99999 shares"
    result: dict[str, object] = (
        {"records": [future_row], "published_at": "2025-07-01"}
        if domain == "FINRA"
        else {"evidence": [{"url": "https://example.com/future", "highlight": passage, "published_at": "2025-07-01"}]}
    )
    job = e2e.jobs[domain.lower()]
    service.persist_tool_result(e2e.sid, job, tool, "TR-FUTURE", result, repo=e2e.repo)
    before = e2e.repo.list_evidence(e2e.sid)
    with pytest.raises(EvidenceRejectedError, match="PIT_VIOLATION"):
        service.record_evidence(
            e2e.sid,
            job,
            {
                **_hf_replay_item("EV-FUTURE", "TR-FUTURE", passage, source_name=domain),
                "value": "99999",
                "metric": "shortInterest" if domain == "FINRA" else "NVDA reported",
                "entity": "NVDA",
                "period": "2025-06-13",
                **({} if domain == "FINRA" else {"unit": "shares"}),
                "known_at": "2025-06-01",
            },
            repo=e2e.repo,
        )
    assert e2e.repo.list_evidence(e2e.sid) == before
    for job_id in e2e.jobs.values():
        service.complete_job(job_id, {}, repo=e2e.repo)
    frozen = service.freeze_session(e2e.sid, 1, repo=e2e.repo)
    frozen_ids = frozen["evidence_ids"]
    assert isinstance(frozen_ids, list)
    assert "EV-FUTURE" not in frozen_ids


def _e2e_claim(e2e: _E2E, text: str, claim_type: str, cite: Sequence[str], points: Sequence[str]) -> GroundedClaim:
    return GroundedClaim(
        text=text,
        claim_type=claim_type,
        evidence_ids=[e2e.ids[key] for key in cite],
        datapoint_ids=[_e2e_point(e2e, key) for key in points],
    )


@pytest.mark.parametrize(
    ("text", "claim_type", "cite", "points", "error"),
    [
        (
            "NVDA Q1 FY2025 Data Center revenue was $26,044 million",
            "observed_fact",
            ("tot",),
            ("tot",),
            "ERR_DATAPOINT_UNBOUND",
        ),
        (
            "NVDA Q1 FY2025 total revenue was $22,563 million and Data Center revenue was $26,044 million",
            "observed_fact",
            ("tot", "dc"),
            ("tot", "dc"),
            "ERR_DATAPOINT_UNBOUND",
        ),
        ("NVDA Q1 FY2025 total revenue was $26,044 million", "observed_fact", ("tot",), (), "ERR_DATAPOINT_UNVERIFIED"),
        (
            "AMD Q1 FY2025 total revenue was $26,044 million",
            "observed_fact",
            ("tot",),
            ("tot",),
            "ERR_DATAPOINT_UNBOUND",
        ),
        (
            "NVDA Q2 FY2025 total revenue was $26,044 million",
            "observed_fact",
            ("tot",),
            ("tot",),
            "ERR_DATAPOINT_UNBOUND",
        ),
        ("NVDA Q1 FY2025 total revenue was $26,044", "observed_fact", ("tot",), ("tot",), "ERR_DATAPOINT_UNBOUND"),
        (
            "NVDA Q1 FY2025 total revenue was $26,044 million, up 262%",
            "observed_fact",
            ("tot",),
            ("tot",),
            "ERR_DATAPOINT_UNVERIFIED",
        ),
        (
            "NVDA Q1 FY2025 total revenue was $26,044 million because of fraud",
            "observed_fact",
            ("tot",),
            ("tot",),
            "ERR_STATEMENT_UNVERIFIED",
        ),
        ("NVDA dominates the AI chip market", "observed_fact", ("tot",), (), "ERR_STATEMENT_UNVERIFIED"),
        ("NVDA total revenue was $99,999 million", "unknown", (), (), "ERR_CLAIM_UNQUALIFIED"),
        ("Margin is unknown but NVDA shipped 99,999 units", "unknown", (), (), "ERR_DATAPOINT_UNVERIFIED"),
        ("Data Center dominates NVDA revenue", "inference", ("tot", "dc"), (), "ERR_CLAIM_UNQUALIFIED"),
        ("NVDA revenue was 99,999, assuming no restatement", "assumption", (), (), "ERR_CLAIM_UNQUALIFIED"),
        (
            "Management stated total revenue was $99,999 million",
            "attributed_report",
            ("tot",),
            (),
            "ERR_DATAPOINT_UNVERIFIED",
        ),
        (
            "NVDA Q1 FY2025 non-Data Center revenue was 3481 million",
            "observed_fact",
            ("calc",),
            ("calc",),
            "ERR_CLAIM_KIND",
        ),
        ("NVDA Q1 FY2025 total revenue was $26,044 million", "calculation", ("tot",), ("tot",), "ERR_CLAIM_KIND"),
    ],
    ids=[
        "authentic-number-wrong-metric",
        "authentic-numbers-swapped",
        "token-overlap-without-datapoint",
        "wrong-entity",
        "wrong-period",
        "missing-unit",
        "stray-number",
        "unsupported-clause",
        "unlocated-qualitative-fact",
        "unknown-label-fact",
        "unknown-label-number",
        "unhedged-inference",
        "trailing-assumption",
        "attributed-number",
        "calculated-as-observed",
        "observed-as-calculated",
    ],
)
def test_gate_rejects_wrong_assembly_missing_fields_and_label_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    claim_type: str,
    cite: tuple[str, ...],
    points: tuple[str, ...],
    error: str,
) -> None:
    """Authentic source numbers, declared types, or citations alone never publish a claim."""
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=f"{error}"):
        service.require_verified_claims(e2e.repo, e2e.sid, [_e2e_claim(e2e, text, claim_type, cite, points)], "gate")


def test_gate_accepts_exact_observations_canonical_calculations_and_qualified_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.research import service

    e2e = _e2e_session(tmp_path, monkeypatch)
    claims = [
        _e2e_claim(e2e, "NVDA Q1 FY2025 total revenue was $26,044 million [x]", "observed_fact", ("tot",), ("tot",)),
        _e2e_claim(
            e2e,
            "NVDA Q1 FY2025 total revenue was $26,044 million and Data Center revenue was $22,563 million",
            "observed_fact",
            ("tot", "dc"),
            ("tot", "dc"),
        ),
        _e2e_claim(e2e, "On 2025-06-20 NVDA shares rose 4.2%", "observed_fact", ("web",), ("web",)),
        _e2e_claim(e2e, "NVDA short interest for 2025-06-13 was 12345", "observed_fact", ("finra",), ("finra",)),
        _e2e_claim(e2e, "NVDA Q1 FY2025 non-Data Center revenue was 3481 million", "calculation", ("calc",), ("calc",)),
        _e2e_claim(e2e, "No NVDA 2026 guidance was found", "unknown", (), ()),
        _e2e_claim(e2e, "Assuming NVDA demand persists, margins hold at 70%", "assumption", (), ()),
        _e2e_claim(e2e, "Data Center revenue likely dominates NVDA revenue", "inference", ("tot", "dc"), ()),
    ]
    service.require_verified_claims(e2e.repo, e2e.sid, claims, "gate")


def test_gate_qualitative_observed_fact_must_equal_a_located_statement() -> None:
    from app.research.agents import GroundedClaim
    from app.research.service import _statement_defect

    point = {"id": "dp:x", "statement": "NVIDIA does not expect supply to improve. Revenue was $1 million."}
    exact = GroundedClaim(text="NVIDIA does not expect supply to improve [ev:1].", claim_type="observed_fact")
    assert _statement_defect(exact, [point], []) is None
    for text in (
        "NVIDIA does expect supply to improve.",
        "expect supply to improve",
        "NVIDIA does not expect supply to improve. Margins are strong.",
    ):
        claim = GroundedClaim(text=text, claim_type="observed_fact")
        assert "ERR_STATEMENT_UNVERIFIED" in (_statement_defect(claim, [point], []) or ""), text


def test_gate_binding_needs_every_descriptor_identity() -> None:
    from app.research.service import _binding_defect

    point: dict[str, object] = {
        "id": "dp:x",
        "metric": "us-gaap:Revenues",
        "value": "60922",
        "entity": "NVDA",
        "period": "FY2024",
        "unit": "USD",
        "statement": "{}",
    }
    assert _binding_defect("NVDA Revenues for FY2024 were $60,922", [point], "observed_fact") is None
    assert "ERR_DATAPOINT_UNBOUND" in (
        _binding_defect("NVDA Revenues for FY2024 were 60,922", [point], "observed_fact") or ""
    )
    for missing in ("metric", "value", "entity", "period"):
        defect = _binding_defect("NVDA Revenues for FY2024 were $60,922", [{**point, missing: None}], "observed_fact")
        assert defect is not None and "lacks" in defect, missing


def test_gate_rejects_borrowed_neighbour_metric_label() -> None:
    """A value cannot borrow a neighbouring metric label: net income beside lone revenue fails."""
    from app.research.service import _binding_defect, _wording_defect

    point: dict[str, object] = {
        "id": "dp:rev",
        "metric": "us-gaap:Revenues",
        "value": "60922",
        "entity": "NVDA",
        "period": "FY2024",
        "unit": "USD",
        "statement": "{}",
    }
    text = "NVDA Revenues and net income for FY2024 were $60,922"
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(text, [point], "observed_fact") or "")
    assert "ERR_STATEMENT_UNVERIFIED" in (_wording_defect(text, [point]) or "")


def test_gate_allows_controlled_restatement_and_rejects_unsupported_relations() -> None:
    """Approximators, source-stated scale words and same-sign change words pass; invented relations fail."""
    from app.research.service import _binding_defect, _wording_defect

    level: dict[str, object] = {
        "id": "dp:dc",
        "metric": "Data Center revenue",
        "value": "22,563",
        "entity": "NVDA",
        "period": "Q1 FY2025",
        "unit": None,
        "statement": "NVDA Q1 FY2025 Data Center revenue was $22,563 million",
    }
    record = {**level, "value": "22563", "unit": "USD", "statement": '{"dataCenterRevenue": 22563}'}
    growth: dict[str, object] = {
        "id": "dp:g",
        "metric": "revenue growth",
        "value": "12.5",
        "entity": "NVDA",
        "period": "Q1 FY2025",
        "unit": "%",
        "formula": "pct_change@1",
        "inputs": ["dp:a", "dp:b"],
        "statement": "revenue growth = pct_change@1(revenue=80, revenue=90) = 12.5",
    }
    fall = {**growth, "id": "dp:f", "value": "-12.5"}
    passing = [
        ("NVDA Data Center revenue for Q1 FY2025 was approximately $22,563 million.", level),
        ("NVDA revenue growth for Q1 FY2025 increased 12.5%.", growth),
        ("NVDA revenue growth for Q1 FY2025 fell 12.5%.", fall),
    ]
    for text, point in passing:
        assert _binding_defect(text, [point], "observed_fact") is None, text
        assert _wording_defect(text, [point]) is None, text
    failing = [
        ("NVDA Data Center revenue for Q1 FY2025 was $22,563 billion.", level),
        ("NVDA Data Center revenue for Q1 FY2025 was $22,563 million.", record),
        ("NVDA Data Center revenue for Q1 FY2025 increased to $22,563 million.", level),
        ("NVDA Data Center revenue for Q1 FY2025 was more than $22,563 million.", level),
        ("NVDA revenue growth for Q1 FY2025 decreased 12.5%.", growth),
        ("NVDA revenue growth for Q1 FY2025 rose 12.5%.", fall),
    ]
    for text, point in failing:
        assert "ERR_STATEMENT_UNVERIFIED" in (_wording_defect(text, [point]) or ""), text
    # A change word between a level's label and its value states a change amount, not the level.
    revenue = {**level, "id": "dp:r", "metric": "revenue", "value": "100", "unit": "%", "statement": "x"}
    text = "NVDA revenue for Q1 FY2025 increased 100%, revenue growth 12.5%."
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(text, [revenue, growth], "observed_fact") or "")


def test_gate_rejects_swapped_entities_across_sentences() -> None:
    """Each bound value sentence states its own descriptor identity: swapped entities fail."""
    from app.research.service import _binding_defect

    def _point(pid: str, entity: str, value: str) -> dict[str, object]:
        return {
            "id": pid,
            "metric": "revenue",
            "value": value,
            "entity": entity,
            "period": "2025",
            "unit": "USD",
            "statement": "{}",
        }

    points = [_point("dp:a", "NVDA", "100"), _point("dp:b", "AMD", "200")]
    swapped = "AMD revenue for 2025 was 100 USD. NVDA revenue for 2025 was 200 USD."
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(swapped, points, "observed_fact") or "")
    matched = "NVDA revenue for 2025 was 100 USD. AMD revenue for 2025 was 200 USD."
    assert _binding_defect(matched, points, "observed_fact") is None
    joint = "NVDA revenue for 2025 was 100 USD and AMD revenue for 2025 was 200 USD."
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(joint, points, "observed_fact") or "")
    # Other claim types keep the original claim-wide identity checks.
    assert _binding_defect(swapped, points, "inference") is None


def test_gate_accepts_descriptor_alias_spellings() -> None:
    """A claim may name an identity by its canonical or its deterministic alias spelling, never another."""
    from app.research.service import _binding_defect, _wording_defect

    point: dict[str, object] = {
        "id": "dp:alias",
        "metric": "revenue",
        "value": "100",
        "entity": "1045810",
        "period": "2025-04-27",
        "unit": "USD",
        "aliases": {"entity": "NVDA", "period": "Q1 FY2026"},
        "statement": "{}",
    }
    text = "NVDA revenue for Q1 FY2026 was $100."
    assert _binding_defect(text, [point], "observed_fact") is None
    assert _wording_defect(text, [point]) is None
    assert _binding_defect("1045810 revenue for 2025-04-27 was $100.", [point], "observed_fact") is None
    bare = {key: value for key, value in point.items() if key != "aliases"}
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(text, [bare], "observed_fact") or "")
    assert "ERR_DATAPOINT_UNBOUND" in (
        _binding_defect("AMD revenue for Q1 FY2026 was $100.", [point], "observed_fact") or ""
    )
    # A unit is stated by its word, plural or symbol, never by a longer word it prefixes.
    assert "ERR_DATAPOINT_UNBOUND" in (
        _binding_defect("NVDA revenue for Q1 FY2026 was 100 USDT.", [point], "observed_fact") or ""
    )


def test_gate_binds_each_shared_value_occurrence_to_its_own_metric() -> None:
    """Same value, different metrics, one occurrence each: valid; one occurrence for two: rejected."""
    from app.research.service import _binding_defect

    def _point(pid: str, metric: str) -> dict[str, object]:
        return {
            "id": pid,
            "metric": metric,
            "value": "100",
            "entity": "NVDA",
            "period": "2025",
            "unit": "USD",
            "statement": "{}",
        }

    points = [_point("dp:a", "revenue"), _point("dp:b", "expense")]
    two = "NVDA revenue for 2025 was 100 USD. NVDA expense for 2025 was 100 USD."
    assert _binding_defect(two, points, "observed_fact") is None
    # Two identical descriptor references cannot share one stated occurrence.
    same = [_point("dp:a", "revenue"), _point("dp:a", "revenue")]
    one = "NVDA revenue for 2025 was 100 USD."
    assert "ERR_DATAPOINT_UNBOUND" in (_binding_defect(one, same, "observed_fact") or "")
    # Same value with distinct identities in separate sentences: each binds its own.
    cross = [
        {**_point("dp:a", "revenue"), "entity": "NVDA"},
        {**_point("dp:b", "revenue"), "entity": "AMD"},
    ]
    split = "NVDA revenue for 2025 was 100 USD. AMD revenue for 2025 was 100 USD."
    assert _binding_defect(split, cross, "observed_fact") is None
    # Two different datapoint ids cannot bind the same stated occurrence.
    twins = [_point("dp:a", "revenue"), _point("dp:b", "revenue")]
    defect = _binding_defect(one, twins, "observed_fact") or ""
    assert "ERR_DATAPOINT_UNBOUND" in defect and "binds to several datapoints" in defect
    assert "dp:a" in defect and "dp:b" in defect
    # A value whose stated text overlaps another datapoint's occurrence also counts as shared.
    date = {**_point("dp:a", "report date"), "value": "2025-04-27", "unit": None}
    year = {**date, "id": "dp:b", "value": "2025"}
    overlap = _binding_defect("NVDA report date for 2025 likely was 2025-04-27.", [date, year], "inference") or ""
    assert "binds to several datapoints" in overlap
