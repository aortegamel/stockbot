"""Explicit repair of legacy evidence rows that carry no proof.

A repair never writes the legacy evidence row, a freeze, or an existing tool
result. It reads the persisted row and tool result, recovers the source identity
the row retained, resolves the historical SEC fact or persisted record under the
original ``as_of``, and records an immutable repair artifact that points at the
old row.

Without a target session a repair is audit-only: no publication gate reads
``evidence_repairs``, the legacy row stays unverified, and the source viewer
shows repairs as read-only history beside that verdict.

With an explicitly named open target session, each verified artifact of this
call is promoted to one new evidence row in that session. The kernel persists the
verified source row as a tool result of the target session (an SEC fact under
``get_xbrl_facts``, a cited record as a copy of its original persisted result)
and re-runs normal admission: replay, canonical datapoint, integrity, PIT
against the target ``as_of``, and proof replay before commit. A repaired status
alone never promotes. Calculation inputs promote first, and the calculation proof
names their new evidence ids. A row carries ``metadata.repair_id`` and
``metadata.repaired_evidence_id``, plus ``superseded_by`` when the legacy row is
in the target session. The identity key ``repair|<repair_id>`` makes a rerun
return the same row. Unresolved artifacts and failed admissions stay audit-only.
Legacy ids and repair ids never become citable.

Matching is deterministic. A retained accession or row pins one exact source.
When the identity is incomplete, every point-in-time candidate must agree, so a
first matching number never stands in for a source. A fact known only after
``as_of`` stays unresolved: a later restatement never substitutes for an older
observation. Every unresolved datapoint and every failed promotion logs its
reason at ERROR level.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from urllib.parse import parse_qs, urlparse

from app.storage import raw_archive

from .evidence import Evidence, _scalar_text, evidence_from_dict, validate_proof, value_hash
from .models import JSONValue, ResearchSession, validate_json_mapping
from .repository import ResearchRepository

logger = logging.getLogger(__name__)

__all__ = ["REPAIR_CATEGORIES", "REPAIR_REASONS", "repair_evidence", "repair_session"]

REPAIR_CATEGORIES = (
    "eps_without_accession",
    "balance_sheet_cik_only",
    "dividend_annual_ttm",
    "bridge_whole_list",
    "finra_structured",
    "derived_incomplete_inputs",
    "company_page_fallback",
)
"""Known legacy evidence shapes a repair recognizes."""

REPAIR_REASONS = (
    "SOURCE_IDENTITY_MISSING",
    "SOURCE_MISMATCH",
    "SOURCE_AMBIGUOUS",
    "SOURCE_UNAVAILABLE",
    "SOURCE_CORRUPT",
    "CALCULATION_INVALID",
)
"""Why a datapoint stays unresolved (the source-viewer vocabulary, minus PUBLIC_PAGE_UNAVAILABLE)."""

_REPAIRABLE_KINDS = (None, "none", "sec_record", "finra_record")
"""Provenance kinds a repair reads: none recorded, or a structured tool record."""

_EPS_KEYS = {"eps_diluted": "EarningsPerShareDiluted", "eps_basic": "EarningsPerShareBasic"}
_DPS_CONCEPT = "CommonStockDividendsPerShareDeclared"
_TTM_KEYS = {
    "ttm_eps_diluted": ("EarningsPerShareDiluted", "sec_eps_ttm"),
    "ttm_eps_basic": ("EarningsPerShareBasic", "sec_eps_ttm"),
    "ttm_dividend_per_share": (_DPS_CONCEPT, "sec_dividend_ttm"),
}
_ROW_LISTS = ("filings", "transactions", "quarterly_eps", "facts", "rows", "records", "documents", "events")
_SYMBOL_KEYS = ("symbolCode", "securitiesInformationProcessorSymbolIdentifier", "issueSymbolIdentifier", "ticker")
_DATE_KEYS = ("settlementDate", "tradeReportDate", "tradeDate", "date", "filing_date", "filed_at", "period_end")
_COMPANY_PAGE = re.compile(r"https?://(?:www\.)?sec\.gov/cgi-bin/browse-edgar", re.IGNORECASE)
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class _Unresolved(Exception):
    """One datapoint the repair cannot verify, with its classification."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class _Point:
    """One cited datapoint and the source identity the row retained for it."""

    key: str
    category: str
    value: str | None
    kind: str  # fact | ttm | derived | record
    spec: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _Run:
    """Per-row repair state: the original row, its as_of, and one companyfacts read per CIK."""

    store: ResearchRepository
    raw: Mapping[str, object]
    as_of: datetime
    facts: dict[int, list[Mapping[str, object]]] = field(default_factory=dict)

    @property
    def as_of_day(self) -> str:
        return self.as_of.date().isoformat()


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: object) -> str | None:
    """Canonical decimal text of one numeric value; None when it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return format(parsed.normalize(), "f") if parsed.is_finite() else None


def _same_number(left: object, right: object) -> bool:
    a = _number(left)
    return a is not None and a == _number(right)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _cik(*sources: object) -> int | None:
    """First retained CIK: a cik field, an SEC entity id, or the CIK of an SEC company page."""
    for source in sources:
        raw = (
            (source.get("cik") or source.get("issuer_cik") or source.get("entity_id"))
            if isinstance(source, Mapping)
            else source
        )
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            return raw
        text = _text(raw)
        if _COMPANY_PAGE.match(text):
            text = (parse_qs(urlparse(text).query).get("CIK") or [""])[0]
        match = re.fullmatch(r"(?:sec:cik:)?0*(\d+)", text)
        if match:
            return int(match[1])
    return None


def _nested(value: object, *keys: object) -> object:
    """value[k1][k2]...; None as soon as a level is not a mapping."""
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


# --------------------------------------------------------------------------- row context


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _as_of(store: ResearchRepository, evidence: Evidence) -> datetime:
    """The original constraint: the session cutoff when set, never later than the row's retrieval."""
    try:
        cutoff = store.get_session(evidence.session_id).as_of
    except KeyError:
        cutoff = None
    retrieved = _aware(evidence.retrieved_at)
    return min(_aware(cutoff), retrieved) if cutoff is not None else retrieved


def _raw_time(raw: Mapping[str, object]) -> datetime:
    """retrieved_at of an undecodable row; the epoch when it is unreadable (the row is corrupt anyway)."""
    try:
        return _aware(datetime.fromisoformat(_text(raw.get("retrieved_at"))))
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)


def _payload(result: Mapping[str, object] | None) -> Mapping[str, object]:
    """The tool's own payload inside one persisted result; empty when there is none."""
    if result is None:
        return {}
    inner = result.get("result")
    return inner if isinstance(inner, Mapping) else result


def _tool_results(run: _Run, evidence: Evidence) -> list[Mapping[str, object]]:
    """The persisted result the row names, else this job's results of the same tool (identity retained)."""
    prov = evidence.provenance
    ref = _text(prov.get("tool_result_id"))
    if ref:
        try:
            result = run.store.get_tool_result(ref)
        except KeyError:
            raise _Unresolved("SOURCE_UNAVAILABLE", f"persisted tool result {ref!r} is absent") from None
        if result.get("session_id") not in (None, "", evidence.session_id):
            raise _Unresolved("SOURCE_MISMATCH", f"tool result {ref!r} belongs to another session")
        return [result]
    tool = _text(prov.get("tool_name"))
    if not tool or not evidence.job_id:
        return []
    return [
        result
        for result in run.store.list_tool_results(evidence.session_id)
        if result.get("tool_name") == tool and result.get("job_id") == evidence.job_id
    ]


def _company_page(evidence: Evidence) -> bool:
    uris = (evidence.source_uri, evidence.provenance.get("source_uri"))
    return any(_COMPANY_PAGE.match(_text(uri)) for uri in uris)


def _row_category(evidence: Evidence | None, raw: Mapping[str, object]) -> str:
    """Category label for a row whose datapoints could not be extracted."""
    prov = raw.get("provenance")
    uris = (raw.get("source_uri"), prov.get("source_uri") if isinstance(prov, Mapping) else None)
    if any(_COMPANY_PAGE.match(_text(uri)) for uri in uris):
        return "company_page_fallback"
    if isinstance(prov, Mapping) and prov.get("kind") == "finra_record":
        return "finra_structured"
    if evidence is not None and evidence.claim_kind == "calculation":
        return "derived_incomplete_inputs"
    return "bridge_whole_list" if isinstance(prov, Mapping) and prov.get("kind") else "balance_sheet_cik_only"


# --------------------------------------------------------------------------- datapoint extraction


def _fact_spec(
    row: Mapping[str, object], cik: int | None, concept: str, *, accession: bool = True
) -> dict[str, object]:
    """Retained identity of one SEC fact; an accession only when the row kept one."""
    return {
        "cik": _cik(row) or cik,
        "concept": concept,
        "period_end": _text(row.get("period_end") or row.get("end")),
        "period_start": _text(row.get("period_start") or row.get("start")),
        "fiscal_period": _text(row.get("fiscal_period") or row.get("fp")),
        "unit": _text(row.get("unit")),
        "accession": _text(row.get("accession") or row.get("accession_no") or row.get("accn")) if accession else "",
    }


def _input_spec(row: Mapping[str, object], cik: int | None, *, accession: bool = True) -> dict[str, object]:
    """One calculation input: a nested derivation, else a fact with the identity it retained."""
    inputs = row.get("derivation_inputs")
    if row.get("derived") and isinstance(inputs, list):
        return {
            "formulaId": _text(row.get("formulaId")),
            "formulaVersion": _text(row.get("formulaVersion")) or "1",
            "value": _number(row.get("value")),
            "inputs": [_input_spec(item, cik, accession=accession) for item in inputs if isinstance(item, Mapping)],
        }
    return {**_fact_spec(row, cik, _text(row.get("concept")), accession=accession), "value": _number(row.get("value"))}


def _record_points(prefix: str, row: Mapping[str, object], cik: int | None) -> list[_Point]:
    """Datapoints one cited SEC record names: EPS values, an annual DPS, a derivation, or a concept value."""
    if row.get("derived") and isinstance(row.get("derivation_inputs"), list):
        key = next((k for k in (*_EPS_KEYS, "value") if _number(row.get(k)) is not None), "value")
        return [
            _Point(prefix + key, "derived_incomplete_inputs", _number(row.get(key)), "derived", _input_spec(row, cik))
        ]
    points: list[_Point] = []
    for key, concept in _EPS_KEYS.items():
        if _number(row.get(key)) is not None:
            fact = row.get("basic_fact") if key == "eps_basic" else None
            source = {**row, **fact} if isinstance(fact, Mapping) else row
            spec = {**_fact_spec(source, cik, concept), "unit": "USD/shares"}
            points.append(_Point(prefix + key, "eps_without_accession", _number(row.get(key)), "fact", spec))
    year = _number(row.get("fiscal_year"))
    if _number(row.get("dividend_per_share")) is not None and year is not None:
        spec = {"cik": cik, "concept": _DPS_CONCEPT, "year": year, "fiscal_period": "FY"}
        value = _number(row.get("dividend_per_share"))
        points.append(_Point(prefix + "dividend_per_share", "dividend_annual_ttm", value, "fact", spec))
    if not points and _text(row.get("concept")) and _number(row.get("value")) is not None:
        spec = _fact_spec(row, cik, _text(row.get("concept")))
        points.append(_Point(prefix + "value", "balance_sheet_cik_only", _number(row.get("value")), "fact", spec))
    return points


def _ttm_points(payload: Mapping[str, object], cik: int | None) -> list[_Point]:
    """TTM values: retained derivation inputs win, else the four quarters are reassembled as of as_of."""
    points: list[_Point] = []
    for key, (concept, formula) in _TTM_KEYS.items():
        value = _number(payload.get(key))
        if value is None:
            continue
        category = "dividend_annual_ttm" if concept == _DPS_CONCEPT else "eps_without_accession"
        retained = _nested(payload, "calculations", key)
        if isinstance(retained, Mapping) and isinstance(retained.get("derivation_inputs"), list):
            spec = _input_spec({**retained, "derived": True}, cik)
            points.append(_Point(key, "derived_incomplete_inputs", value, "derived", spec))
        else:
            spec = {"cik": cik, "concept": concept, "formulaId": formula, "formulaVersion": "1"}
            points.append(_Point(key, category, value, "ttm", spec))
    return points


def _envelope_points(payload: Mapping[str, object], cik: int | None) -> list[_Point]:
    points = _ttm_points(payload, cik)
    for name in ("quarterly_eps", "annual_history", "facts", "rows"):
        rows = payload.get(name)
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping):
                label = _text(row.get("period_end")) or str(row.get("fiscal_year") or "")
                points.extend(_record_points(f"{name}[{label}].", row, cik))
    return points


def _names_envelope(identity: str, payload: Mapping[str, object]) -> bool:
    """A truncated identity still names the whole envelope when it prefixes its compact JSON."""
    compact = " ".join(_canonical(dict(payload)).split())
    return bool(identity) and compact.startswith(" ".join(identity.split()))


def _sec_points(evidence: Evidence, cited: object, payload: Mapping[str, object]) -> list[_Point]:
    """SEC datapoints of the cited record or list, or of the whole envelope when the row cited all of it."""
    prov = evidence.provenance
    cik = _cik(cited, payload, evidence.metadata, evidence.entity, evidence.source_uri, prov.get("source_uri"))
    if isinstance(cited, Mapping):
        return _record_points("", cited, cik) or _envelope_points(cited, cik)
    if isinstance(cited, list):
        rows = [row for row in cited if isinstance(row, Mapping)]
        return [point for index, row in enumerate(rows) for point in _record_points(f"[{index}].", row, cik)]
    return _envelope_points(payload, cik) if _names_envelope(_text(prov.get("record_identity")), payload) else []


def _typed_record(evidence: Evidence) -> dict[str, object]:
    """A row with no tool record: its typed datapoint plus the identity its metadata kept."""
    record: dict[str, object] = dict(evidence.metadata)
    if evidence.value is not None:
        record["value"] = evidence.value
    if evidence.unit and not _text(record.get("unit")):
        record["unit"] = evidence.unit
    if evidence.period and _ISO_DAY.fullmatch(evidence.period) and not record.get("period_end"):
        record["period_end"] = evidence.period
    return record


def _select_typed(evidence: Evidence, points: list[_Point]) -> list[_Point]:
    """A row with a typed value cites exactly one datapoint: the unique one carrying that value."""
    if evidence.value is None:
        return points
    matches = [point for point in points if _same_number(point.value, evidence.value)]
    if len(matches) > 1:
        raise _Unresolved("SOURCE_AMBIGUOUS", f"value {evidence.value!r} matches {len(matches)} cited datapoints")
    if not matches:
        raise _Unresolved("SOURCE_MISMATCH", f"value {evidence.value!r} is not among the cited datapoints")
    return matches


# --------------------------------------------------------------------------- SEC facts


def _company_facts(run: _Run, cik: int) -> list[Mapping[str, object]]:
    """All normalized companyfacts rows for one CIK; the gateway archives the exact payload bytes."""
    if cik not in run.facts:
        from app.data_sources import SourceGateway

        try:
            rows = SourceGateway().company_facts(cik).get("financial_facts")
        except (OSError, ValueError, RuntimeError) as exc:
            raise _Unresolved("SOURCE_UNAVAILABLE", f"companyfacts for CIK {cik} unavailable: {exc}") from None
        run.facts[cik] = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    return run.facts[cik]


def _duration_ok(row: Mapping[str, object], spec: Mapping[str, object]) -> bool:
    from app.services.sec_facts import _FY_DAYS, _QUARTER_DAYS

    start, end = _text(row.get("period_start")), _text(row.get("period_end"))
    if _text(spec.get("period_start")):
        return start == spec["period_start"]
    fp = _text(spec.get("fiscal_period")).upper()
    bounds = _FY_DAYS if fp == "FY" else _QUARTER_DAYS if re.fullmatch(r"Q[1-4]", fp) else None
    if bounds is None:
        return True
    try:
        days = (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days
    except ValueError:
        return False
    return bounds[0] <= days <= bounds[1]


def _fact_matches(row: Mapping[str, object], spec: Mapping[str, object]) -> bool:
    end = _text(row.get("period_end"))
    return (
        row.get("concept") == spec.get("concept")
        and not row.get("dimensions")
        and (not _text(spec.get("accession")) or row.get("accession") == spec["accession"])
        and (not _text(spec.get("period_end")) or end == spec["period_end"])
        and (spec.get("year") is None or end[:4] == str(spec["year"]))
        and (not _text(spec.get("unit")) or row.get("unit") == spec["unit"])
        and _duration_ok(row, spec)
    )


def _pit_fact(run: _Run, spec: Mapping[str, object], value: str) -> Mapping[str, object]:
    """The one historical fact the identity names under as_of; ambiguity and late facts stay unresolved."""
    cik = spec.get("cik")
    if not isinstance(cik, int) or not _text(spec.get("concept")):
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "the row retained no CIK and concept for this datapoint")
    if not _text(spec.get("period_end")) and spec.get("year") is None:
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "the row retained no reporting period for this datapoint")
    matching = [row for row in _company_facts(run, cik) if _fact_matches(row, spec)]
    if not matching:
        raise _Unresolved("SOURCE_UNAVAILABLE", f"no reported {spec['concept']} fact matches the retained identity")
    known = [row for row in matching if _text(row.get("known_at"))[:10] <= run.as_of_day]
    if not known:
        raise _Unresolved(
            "SOURCE_UNAVAILABLE",
            f"every matching fact was filed after as_of {run.as_of_day}; a later restatement never substitutes",
        )
    variants = {(r.get("unit"), r.get("period_start"), r.get("period_end"), _number(r.get("value"))) for r in known}
    if len(variants) > 1:
        raise _Unresolved("SOURCE_AMBIGUOUS", f"{len(variants)} incompatible facts were known at as_of {run.as_of_day}")
    if not _same_number(known[0].get("value"), value):
        raise _Unresolved("SOURCE_MISMATCH", f"cited {value} but the filed fact is {known[0].get('value')}")
    return min(known, key=_filing_order)


def _filing_order(row: Mapping[str, object]) -> tuple[str, str]:
    """Original report first: earliest filing date, accession tie-break."""
    return _text(row.get("filed_at")), _text(row.get("accession"))


def _archived_fact(fact: Mapping[str, object]) -> str:
    """Replay the fact against the archived companyfacts bytes; return their sha256."""
    kind, digest = _text(fact.get("source_record_id")), _text(fact.get("content_hash"))
    payload = raw_archive.read_payload("sec", kind, "companyfacts", sha256=digest)
    if payload is None:
        reason = "SOURCE_CORRUPT" if raw_archive.find("sec", kind, "companyfacts") is not None else "SOURCE_UNAVAILABLE"
        raise _Unresolved(reason, f"companyfacts bytes {digest} unreadable")
    namespace, _, tag = _text(fact.get("original_concept")).partition(":")
    try:
        entries = _nested(json.loads(payload), "facts", namespace, tag, "units", fact.get("unit"))
    except ValueError:
        entries = None
    if not isinstance(entries, list) or fact.get("source_identity") not in entries:
        raise _Unresolved("SOURCE_CORRUPT", "the archived companyfacts bytes do not contain the resolved fact")
    return digest


def _fact_source(fact: Mapping[str, object]) -> dict[str, object]:
    """The verified fact as a record: the row a promotion persists and its datapoint binds to."""
    keys = (
        "entity_id",
        "cik",
        "concept",
        "original_concept",
        "unit",
        "period_start",
        "period_end",
        "fiscal_year",
        "fiscal_period",
    )
    return {
        **{key: fact.get(key) for key in keys},
        "accession": fact.get("accession"),
        "form": _nested(fact, "source_identity", "form"),
        "filed_at": fact.get("filed_at"),
        "known_at": fact.get("known_at"),
        "value": _number(fact.get("value")),
    }


# --------------------------------------------------------------------------- artifacts


def _artifact(run: _Run, point: _Point, *, status: str, **fields: object) -> dict[str, object]:
    """One repair artifact; its id hashes everything except the write time."""
    body: dict[str, object] = {
        "session_id": run.raw.get("session_id"),
        "evidence_id": run.raw.get("evidence_id"),
        "evidence_sha256": sha256(_canonical(dict(run.raw)).encode("utf-8")).hexdigest(),
        "category": point.category,
        "datapoint": point.key,
        "as_of": run.as_of.isoformat(),
        "status": status,
        "reason": "",
        "detail": "",
        "value": point.value,
        "entity": run.raw.get("entity"),
        "period": run.raw.get("period"),
        "unit": run.raw.get("unit"),
        "proof": {},
        "source": {},
        "inputs": [],
        **fields,
    }
    return {
        "repair_id": "repair:" + sha256(_canonical(body).encode("utf-8")).hexdigest(),
        **body,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _fact_artifact(run: _Run, point: _Point, spec: Mapping[str, object], value: str) -> dict[str, object]:
    fact = _pit_fact(run, spec, value)
    digest = _archived_fact(fact)
    locator = {"archive": f"sec/{fact.get('source_record_id')}/companyfacts", "fact": fact.get("source_identity")}
    proof = validate_proof(
        {"kind": "observation", "sourceId": f"sha256:{digest}", "locator": locator, "valueHash": value_hash(value)}
    )
    return _artifact(
        run,
        point,
        status="verified",
        value=value,
        entity=f"sec:cik:{fact.get('cik')}",
        period=fact.get("period_end"),
        unit=fact.get("unit"),
        proof=proof,
        source=_fact_source(fact),
    )


def _calculation_artifact(
    run: _Run, point: _Point, formula: tuple[str, str], specs: Sequence[Mapping[str, object]], value: str
) -> dict[str, object]:
    """Resolve every input, then recompute the versioned formula against the cited value."""
    from app.services.sec_facts import _has_contiguous_quarters

    from .service import _FORMULAS, _recompute

    if formula not in _FORMULAS:
        raise _Unresolved("CALCULATION_INVALID", f"formula {formula[0]!r} v{formula[1]} is unknown")
    if not specs:
        raise _Unresolved("CALCULATION_INVALID", "the row retained no calculation inputs")
    inputs = [_input_artifact(run, point, index, spec) for index, spec in enumerate(specs)]
    if len({item.get("entity") for item in inputs}) != 1 or len({item.get("unit") for item in inputs}) != 1:
        raise _Unresolved("CALCULATION_INVALID", "calculation inputs disagree on entity or unit")
    ends: list[object] = [str(item.get("period")) for item in inputs]
    if _FORMULAS[formula][1] == "ttm" and not _has_contiguous_quarters(ends):
        raise _Unresolved("CALCULATION_INVALID", "TTM inputs are not four contiguous quarters")
    try:
        recomputed = _recompute(formula[0], formula[1], tuple(str(item["value"]) for item in inputs))
    except (ArithmeticError, ValueError) as exc:
        raise _Unresolved("CALCULATION_INVALID", f"formula rejected its inputs: {exc}") from None
    if not _same_number(recomputed, value):
        raise _Unresolved("SOURCE_MISMATCH", f"cited {value} but {formula[0]} recomputes {recomputed}")
    proof = validate_proof(
        {
            "kind": "calculation",
            "formulaId": formula[0],
            "formulaVersion": formula[1],
            "inputProofIds": [str(item["repair_id"]) for item in inputs],
            "valueHash": value_hash(value),
        }
    )
    return _artifact(
        run,
        point,
        status="verified",
        value=value,
        entity=inputs[0].get("entity"),
        period=max(str(end) for end in ends),
        unit=inputs[0].get("unit"),
        proof=proof,
        source={"formulaId": formula[0], "formulaVersion": formula[1]},
        inputs=inputs,
    )


def _spec_artifact(run: _Run, point: _Point, spec: Mapping[str, object], value: str) -> dict[str, object]:
    """A fact spec resolves to an observation; a derivation spec recomputes over its own inputs."""
    if "formulaId" not in spec:
        return _fact_artifact(run, point, spec, value)
    nested = spec.get("inputs")
    formula = (_text(spec.get("formulaId")), _text(spec.get("formulaVersion")) or "1")
    return _calculation_artifact(run, point, formula, nested if isinstance(nested, list) else [], value)


def _input_artifact(run: _Run, parent: _Point, index: int, spec: Mapping[str, object]) -> dict[str, object]:
    """One verified calculation input; any input that fails leaves the whole calculation invalid."""
    value = _number(spec.get("value"))
    point = _Point(f"{parent.key}.inputs[{index}]", parent.category, value, "fact", spec)
    try:
        if value is None:
            raise _Unresolved("SOURCE_IDENTITY_MISSING", "no value retained")
        artifact = _spec_artifact(run, point, spec, value)
        # Nested write times are not source content: keep them out of the parent's hash and immutable body.
        return {key: item for key, item in artifact.items() if key != "created_at"}
    except _Unresolved as exc:
        raise _Unresolved("CALCULATION_INVALID", f"input {index}: {exc.reason} ({exc.detail})") from None


def _ttm_specs(run: _Run, spec: Mapping[str, object]) -> list[dict[str, object]]:
    """Reassemble the four latest quarters known at as_of by the tool's own rule; identities stay incomplete."""
    from app.services.sec_facts import _recent_concept_quarters, _validated_fact_row

    cik = spec.get("cik")
    if not isinstance(cik, int):
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "the row retained no CIK for this TTM value")
    validated = (_validated_fact_row(row) for row in _company_facts(run, cik))
    known = [fact for fact in validated if fact is not None and _text(fact.get("known_at"))[:10] <= run.as_of_day]
    quarters = _recent_concept_quarters(known, _text(spec.get("concept")))
    if len(quarters) != 4:
        raise _Unresolved("SOURCE_UNAVAILABLE", f"{len(quarters)} quarters were known at as_of {run.as_of_day}")
    return [_input_spec(quarter, cik, accession=False) for quarter in quarters]


def _sec_artifact(run: _Run, point: _Point) -> dict[str, object]:
    if point.value is None:
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "the row retained no cited value")
    if point.kind == "ttm":
        formula = (_text(point.spec.get("formulaId")), _text(point.spec.get("formulaVersion")))
        return _calculation_artifact(run, point, formula, _ttm_specs(run, point.spec), point.value)
    return _spec_artifact(run, point, point.spec, point.value)


# --------------------------------------------------------------------------- persisted records (FINRA, bridge lists)


def _result_rows(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for name in _ROW_LISTS:
        items = payload.get(name)
        if isinstance(items, list):
            rows.extend(item for item in items if isinstance(item, Mapping))
    return rows


def _row_has(row: Mapping[str, object], value: str) -> bool:
    return any(_same_number(item, value) or _text(item) == value for item in row.values())


def _row_agrees(row: Mapping[str, object], evidence: Evidence) -> bool:
    """Subject and date agree with the row's typed entity and period wherever both exist."""
    symbols = {_text(row.get(key)).casefold() for key in _SYMBOL_KEYS} - {""}
    dates = {_text(row.get(key))[:10] for key in _DATE_KEYS} - {""}
    entity_ok = not evidence.entity or not symbols or evidence.entity.casefold() in symbols
    period_ok = not evidence.period or not dates or evidence.period[:10] in dates
    return entity_ok and period_ok


def _cited_rows(evidence: Evidence, cited: object, result: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Rows of one result the evidence names: its exact cited row, else every row carrying its typed value."""
    rows = _result_rows(_payload(result))
    if isinstance(cited, Mapping) and any(_canonical(row) == _canonical(cited) for row in rows):
        return [cited] if evidence.value is None or _row_has(cited, evidence.value) else []
    if evidence.value is None:
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "a whole-list citation names no row and no typed value")
    return [row for row in rows if _row_has(row, evidence.value) and _row_agrees(row, evidence)]


def _row_known(run: _Run, row: Mapping[str, object], result: Mapping[str, object]) -> bool:
    payload = _payload(result)
    stamps = [_text(row.get(key))[:10] for key in _DATE_KEYS]
    stamps += [_text(payload.get(key))[:10] for key in ("published_at", "publication_date", "available_at")]
    return all(stamp <= run.as_of_day for stamp in stamps if _ISO_DAY.fullmatch(stamp))


def _archived_result(result: Mapping[str, object]) -> str:
    """sha256 of the canonical persisted result bytes; archived from the immutable row when absent."""
    payload = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    digest = raw_archive.content_hash(payload)
    if raw_archive.read_payload("artifacts", "sha256", digest, sha256=digest) is None:
        raw_archive.store_evidence_artifact(payload, metadata={"tool_result_id": result.get("tool_result_id")})
    if raw_archive.read_payload("artifacts", "sha256", digest, sha256=digest) is None:
        raise _Unresolved("SOURCE_CORRUPT", f"persisted tool result bytes {digest} did not archive")
    return digest


def _record_artifact(
    run: _Run, evidence: Evidence, point: _Point, results: list[Mapping[str, object]]
) -> dict[str, object]:
    if not results:
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "the row retained no persisted tool result to replay")
    found = [(result, row) for result in results for row in _cited_rows(evidence, point.spec.get("cited"), result)]
    distinct = {(_canonical(row), str(result.get("tool_result_id"))) for result, row in found}
    if not found:
        raise _Unresolved("SOURCE_MISMATCH", "no persisted row carries the cited record")
    if len(distinct) > 1:
        raise _Unresolved("SOURCE_AMBIGUOUS", f"{len(distinct)} persisted rows carry the cited value")
    result, row = found[0]
    if not _row_known(run, row, result):
        raise _Unresolved("SOURCE_UNAVAILABLE", f"the cited row is dated after as_of {run.as_of_day}")
    value = evidence.value or _canonical(row)
    locator = {"tool_result_id": result.get("tool_result_id"), "tool_name": result.get("tool_name"), "row": dict(row)}
    digest = _archived_result(result)
    proof = validate_proof(
        {"kind": "observation", "sourceId": f"sha256:{digest}", "locator": locator, "valueHash": value_hash(value)}
    )
    return _artifact(run, point, status="verified", value=value, proof=proof, source=locator)


# --------------------------------------------------------------------------- entry points


def _points(run: _Run, evidence: Evidence) -> tuple[list[_Point], list[Mapping[str, object]]]:
    """Cited datapoints plus the persisted results they replay against."""
    from .service import _FINRA_EVIDENCE_TOOLS

    prov = evidence.provenance
    structured = prov.get("kind") in ("sec_record", "finra_record")
    results: list[Mapping[str, object]] = _tool_results(run, evidence) if structured else []
    try:
        cited = json.loads(_text(prov.get("record_identity")))
    except ValueError:
        cited = None
    finra = prov.get("kind") == "finra_record" or prov.get("tool_name") in _FINRA_EVIDENCE_TOOLS
    if not finra:
        source = cited if structured else _typed_record(evidence)
        points = _sec_points(evidence, source, _payload(results[0] if len(results) == 1 else None))
        if points:
            page = _company_page(evidence)
            selected = _select_typed(evidence, points)
            return [
                _Point(p.key, "company_page_fallback", p.value, p.kind, p.spec) if page else p for p in selected
            ], results
    if not structured:
        raise _Unresolved("SOURCE_IDENTITY_MISSING", "no fact identity was retained for the cited value")
    envelope = cited if isinstance(cited, Mapping) else _payload(results[0] if len(results) == 1 else None)
    listed = {_canonical(row) for result in results for row in _result_rows(_payload(result))}
    cik_only = _canonical(envelope) not in listed and _cik(envelope) and not _result_rows(envelope)
    if not finra and ("balance_sheet" in envelope or cik_only):
        # A CIK-only envelope keeps no concept or period: the fact path reports exactly what is missing.
        return [_Point("record", "balance_sheet_cik_only", evidence.value, "fact", {"cik": _cik(envelope)})], results
    category = "finra_structured" if finra else "bridge_whole_list"
    return [_Point("record", category, evidence.value, "record", {"cited": cited})], results


def _outcome(run: _Run, evidence: Evidence, point: _Point, results: list[Mapping[str, object]]) -> dict[str, object]:
    try:
        if point.kind == "record":
            return _record_artifact(run, evidence, point, results)
        return _sec_artifact(run, point)
    except _Unresolved as exc:
        return _unresolved(run, point, exc)


def _unresolved(run: _Run, point: _Point, exc: _Unresolved) -> dict[str, object]:
    logger.error(
        "evidence repair unresolved session=%s evidence=%s category=%s datapoint=%s reason=%s detail=%s",
        run.raw.get("session_id"),
        run.raw.get("evidence_id"),
        point.category,
        point.key,
        exc.reason,
        exc.detail,
    )
    return _artifact(run, point, status="unresolved", reason=exc.reason, detail=exc.detail)


def _repairable(raw: Mapping[str, object]) -> bool:
    """A substantive row with no proof whose provenance a repair can read."""
    meta = raw.get("metadata")
    kind = raw.get("record_kind") or (meta.get("record_kind") if isinstance(meta, Mapping) else None)
    prov = raw.get("provenance")
    prov_kind = prov.get("kind") if isinstance(prov, Mapping) else None
    return kind != "discovery" and not raw.get("proof") and prov_kind in _REPAIRABLE_KINDS


def _repairs(store: ResearchRepository, raw: Mapping[str, object]) -> list[dict[str, object]]:
    try:
        evidence = evidence_from_dict(raw)
    except ValueError as exc:
        run = _Run(store, raw, _raw_time(raw))
        point = _Point("record", _row_category(None, raw), None, "record")
        return [_unresolved(run, point, _Unresolved("SOURCE_CORRUPT", f"stored row fails integrity: {exc}"))]
    run = _Run(store, raw, _as_of(store, evidence))
    try:
        points, results = _points(run, evidence)
    except _Unresolved as exc:
        return [_unresolved(run, _Point("record", _row_category(evidence, raw), evidence.value, "record"), exc)]
    return [_outcome(run, evidence, point, results) for point in points]


# --------------------------------------------------------------------------- promotion into a target session

_REPAIR_LINEAGE = "repair"
"""Job and agent id of promoted rows and their tool results: an explicit operator repair, never a model job."""

_REPAIR_SEC_TOOL = "get_xbrl_facts"
"""SEC replay tool a promoted companyfacts fact persists under."""


def _open_session(store: ResearchRepository, session_id: str) -> ResearchSession:
    """The target session as stored now; KeyError when absent, ValueError when terminal."""
    from .session import TERMINAL_STATUSES

    found = store.get_session(session_id)
    if found.status in TERMINAL_STATUSES:
        raise ValueError(f"target session {session_id!r} status is {found.status!r} (terminal)")
    return found


def _live_wave(found: ResearchSession) -> int:
    """The wave new evidence joins (the service live-wave rule); a frozen wave fails closed."""
    wave = max(found.current_wave, len(found.freeze_ids) + 1, 1)
    if f"{found.session_id}:{wave}:freeze" in found.freeze_ids:
        raise ValueError(f"target session {found.session_id!r} wave {wave} is frozen")
    return wave


def _source_record(
    store: ResearchRepository, artifact: Mapping[str, object]
) -> tuple[str, str, dict[str, object], Mapping[str, object], dict[str, str | None]]:
    """(domain, tool, payload, row, datapoint) one verified observation replays in the target session.

    An SEC fact persists as one XBRL fact row beside its companyfacts proof. A
    cited record copies its original persisted result, which must still hash to
    the verified source, and binds the one row field that holds the value.
    """
    source = artifact.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("the repair artifact names no source")  # noqa: TRY004 - promotion failures are ValueError
    if "tool_result_id" not in source:
        row = dict(source)
        datapoint = {
            "metric": _text(row.get("concept")) or None,
            "value": _scalar_text(row.get("value")),
            "entity": _scalar_text(row.get("entity_id")),
            "period": _scalar_text(row.get("period_end")),
            "unit": _scalar_text(row.get("unit")),
        }
        return "SEC", _REPAIR_SEC_TOOL, {"facts": [row], "companyfacts": artifact.get("proof")}, row, datapoint
    original = store.get_tool_result(str(source.get("tool_result_id")))
    digest = raw_archive.content_hash(_canonical(original).encode("utf-8"))
    row = source.get("row")
    if _nested(artifact, "proof", "sourceId") != f"sha256:{digest}" or not isinstance(row, Mapping):
        raise ValueError(f"tool result {source.get('tool_result_id')!r} no longer matches the verified source")
    value = artifact.get("value")
    fields = [key for key, item in row.items() if _same_number(item, value) or _text(item) == value]
    if len(fields) != 1:
        raise ValueError(f"the cited row holds the value in {len(fields)} fields, not exactly one")
    datapoint = {
        "metric": str(fields[0]),
        "value": _scalar_text(row[fields[0]]),
        "entity": _scalar_text(artifact.get("entity")),
        "period": _scalar_text(artifact.get("period")),
        "unit": _scalar_text(artifact.get("unit")),
    }
    domain = "FINRA" if artifact.get("category") == "finra_structured" else "SEC"
    return domain, str(original.get("tool_name")), dict(_payload(original)), row, datapoint


def _target_result(
    store: ResearchRepository,
    session_id: str,
    result_id: str,
    tool: str,
    payload: Mapping[str, object],
    artifact: Mapping[str, object],
) -> None:
    """Persist the verified source as a tool result of the target session once; a rerun reuses it."""
    try:
        store.get_tool_result(result_id)
    except KeyError:
        store.save_tool_result(
            {
                "tool_result_id": result_id,
                "session_id": session_id,
                "job_id": _REPAIR_LINEAGE,
                "tool_name": tool,
                "created_at": datetime.now(UTC).isoformat(),
                "arguments": {"repair_id": str(artifact["repair_id"]), "evidence_id": str(artifact["evidence_id"])},
                "result": validate_json_mapping(dict(payload), "<repair>: 'result'"),
            }
        )


def _lineage(artifact: Mapping[str, object], interval: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Promoted-row metadata: the repair and legacy row it came from, plus any period interval."""
    return {
        "repair_id": str(artifact["repair_id"]),
        "repaired_evidence_id": str(artifact["evidence_id"]),
        "repaired_session_id": str(artifact["session_id"]),
        **interval,
    }


def _promotion_key(session_id: str, artifact: Mapping[str, object]) -> str:
    return sha256(f"{session_id}|{artifact['repair_id']}".encode()).hexdigest()


def _promote_observation(
    store: ResearchRepository,
    session_id: str,
    raw: Mapping[str, object],
    artifact: Mapping[str, object],
    supersedes: str | None,
) -> str:
    """Admit one verified observation through the normal replay, datapoint, PIT, and proof path."""
    from .service import (
        _admission_observation_proof,
        _build_evidence_record,
        _finra_provenance,
        _persist_finra_web_record,
        _sec_record_provenance,
    )

    domain, tool, payload, row, datapoint = _source_record(store, artifact)
    key = _promotion_key(session_id, artifact)
    result_id = f"{session_id}:tr:repair:{key[:16]}"
    _target_result(store, session_id, result_id, tool, payload, artifact)
    data: dict[str, object] = {
        "tool_result_id": result_id,
        "record_identity": _canonical(row),
        "known_at": row.get("known_at") if domain == "SEC" else None,
        "source_type": domain.lower(),
        "agent_id": _REPAIR_LINEAGE,
        "superseded_by": supersedes,
    }
    replay = _finra_provenance if domain == "FINRA" else _sec_record_provenance
    provenance = replay(data, None, store=store, session_id=session_id)
    found = _open_session(store, session_id)
    proof, source_bytes, descriptor = _admission_observation_proof(store, provenance, datapoint, found.as_of)
    interval: dict[str, JSONValue] = {
        name: _text(row.get(name)) for name in ("period_start", "period_end") if _text(row.get(name))
    }
    content = (
        f"{datapoint['metric']} = {datapoint['value']} (repair {artifact['repair_id']} of {artifact['evidence_id']})"
    )
    record = _build_evidence_record(
        data,
        session_id=session_id,
        job_id=_REPAIR_LINEAGE,
        wave=_live_wave(found),
        evidence_id=f"ev:{key[:12]}",
        source_name=domain,
        subject=_text(raw.get("subject")),
        claim=content,
        content=content,
        supports=(),
        contradicts=(),
        metadata=_lineage(artifact, interval),
        identity_key=f"repair|{artifact['repair_id']}",
        claim_kind="observed_fact",
        provenance=provenance,
        datapoint={name: _scalar_text(descriptor.get(name)) for name in ("value", "entity", "period", "unit")},
        proof=proof,
    )
    stored = _persist_finra_web_record(store, session_id, _REPAIR_LINEAGE, found, record, proof_bytes=source_bytes)
    return str(stored["evidence_id"])


def _interval(semantics: str, rows: Sequence[Evidence]) -> dict[str, JSONValue]:
    """The calculation's own period interval for a temporal formula; the verifier checks it against its inputs."""
    starts = [_text(row.metadata.get("period_start")) for row in rows]
    ends = [_text(row.metadata.get("period_end")) for row in rows]
    if semantics == "ttm":
        return {"period_start": starts[0], "period_end": ends[-1]}
    if semantics == "quarter" and len(ends) == 2:
        return {"period_start": (date.fromisoformat(ends[1]) + timedelta(days=1)).isoformat(), "period_end": ends[0]}
    return {}


def _promote_calculation(
    store: ResearchRepository,
    session_id: str,
    raw: Mapping[str, object],
    artifact: Mapping[str, object],
    supersedes: str | None,
) -> str:
    """Promote every input first, then admit the calculation over those new rows, recomputed before commit."""
    from .evidence import calculation_datapoint, calculation_proof, evidence_content_hash
    from .service import _FORMULAS, _calculation_known_at, _persist_finra_web_record

    items = artifact.get("inputs")
    if not isinstance(items, list) or not items:
        raise ValueError("the calculation repair names no inputs")
    ids = [_promote_point(store, session_id, raw, item, None) for item in items]
    rows = [evidence_from_dict(store.get_evidence(eid)) for eid in ids]
    formula_id, version = (
        str(_nested(artifact, "proof", "formulaId")),
        str(_nested(artifact, "proof", "formulaVersion")),
    )
    value, period = str(artifact["value"]), str(artifact["period"])
    entity, unit = rows[0].entity, rows[0].unit
    datapoint = calculation_datapoint(
        formula_id=formula_id,
        formula_version=version,
        inputs=rows,
        metric=str(artifact["datapoint"]),
        value=value,
        entity=entity,
        period=period,
        unit=unit,
    )
    proof = calculation_proof(
        formula_id=formula_id, formula_version=version, input_proof_ids=ids, value=value, datapoint=datapoint
    )
    formula = f"{formula_id}@{version}"
    content = f"{formula}({', '.join(f'{row.evidence_id}={row.value}' for row in rows)}) = {value}"
    identity_key = f"repair|{artifact['repair_id']}"
    found = _open_session(store, session_id)
    record = Evidence(
        evidence_id=f"ev:{_promotion_key(session_id, artifact)[:12]}",
        session_id=session_id,
        wave_id=_live_wave(found),
        source_type="calculation",
        source_name="calculation",
        subject=_text(raw.get("subject")),
        claim_text=content[:500],
        content=content,
        content_hash=evidence_content_hash(content),
        retrieved_at=datetime.now(UTC),
        source_record_id=f"calculation:{formula}",
        known_at=_calculation_known_at(rows),
        job_id=_REPAIR_LINEAGE,
        agent_id=_REPAIR_LINEAGE,
        metadata={
            **_lineage(artifact, _interval(_FORMULAS[(formula_id, version)][1], rows)),
            "identity_key": identity_key,
            "claim_kind": "calculation",
        },
        superseded_by=supersedes,
        claim_kind="calculation",
        value=value,
        entity=entity,
        period=period,
        unit=unit,
        proof=proof,
    )
    stored = _persist_finra_web_record(store, session_id, _REPAIR_LINEAGE, found, record, proof_bytes=None)
    return str(stored["evidence_id"])


def _promote_point(
    store: ResearchRepository,
    session_id: str,
    raw: Mapping[str, object],
    artifact: object,
    supersedes: str | None,
) -> str:
    """The target-session row of one verified artifact: the existing one on a rerun, else a newly admitted one."""
    if not isinstance(artifact, Mapping) or artifact.get("status") != "verified":
        raise ValueError("only a verified repair artifact promotes")
    hit = store.find_evidence_by_identity(session_id, f"repair|{artifact['repair_id']}")
    if hit is not None:
        return str(hit["evidence_id"])
    promote = _promote_calculation if _nested(artifact, "proof", "kind") == "calculation" else _promote_observation
    return promote(store, session_id, raw, artifact, supersedes)


def _promote(
    store: ResearchRepository, session_id: str, raw: Mapping[str, object], artifact: Mapping[str, object]
) -> str | None:
    """Promoted evidence id of one artifact; an unresolved or refused artifact stays audit-only (None)."""
    if artifact.get("status") != "verified":
        return None
    # The correction chain links only inside one ledger: the legacy row must sit in the target session.
    supersedes = str(raw.get("evidence_id")) if raw.get("session_id") == session_id else None
    try:
        return _promote_point(store, session_id, raw, artifact, supersedes)
    except (KeyError, ValueError) as exc:
        logger.error(
            "evidence repair not promoted target=%s evidence=%s repair=%s: %s",
            session_id,
            raw.get("evidence_id"),
            artifact.get("repair_id"),
            exc,
        )
        return None


def repair_evidence(
    store: ResearchRepository, evidence_id: str, target_session_id: str | None = None
) -> list[dict[str, object]]:
    """Record one repair artifact per datapoint of one legacy row; the row itself is never written.

    With ``target_session_id`` every verified artifact of this call is also
    promoted into that open session (see the module docstring). Each returned
    artifact then carries ``promoted_evidence_id``: the new row id, or None when
    it stays audit-only. Persisted artifacts never carry that key.

    Raises KeyError for an unknown id or target session. Raises ValueError for a
    terminal or frozen target, or for a row a repair does not cover: a discovery
    row, a row that already carries proof, or a passage/web row.
    """
    raw = store.get_evidence(evidence_id)
    if not _repairable(raw):
        raise ValueError(f"evidence {evidence_id!r} is not a repairable legacy row")
    if target_session_id is not None:
        _live_wave(_open_session(store, target_session_id))
    artifacts = _repairs(store, raw)
    for artifact in artifacts:
        store.save_evidence_repair(artifact)
    if target_session_id is None:
        return artifacts
    return [
        {**artifact, "promoted_evidence_id": _promote(store, target_session_id, raw, artifact)}
        for artifact in artifacts
    ]


def repair_session(
    store: ResearchRepository, session_id: str, target_session_id: str | None = None
) -> list[dict[str, object]]:
    """Repair every repairable legacy row of one session, promoting into ``target_session_id`` when named."""
    rows = [row for row in store.list_evidence(session_id) if _repairable(row)]
    return [
        artifact for row in rows for artifact in repair_evidence(store, str(row.get("evidence_id")), target_session_id)
    ]
