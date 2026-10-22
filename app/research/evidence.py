"""Append-only evidence ledger with provenance + PIT ingest gate. stdlib only."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from itertools import pairwise
from typing import NamedTuple

from .models import JSONValue, pit_unverified, pit_violated, validate_json_mapping

__all__ = [
    "ACCESSION_RE",
    "CLAIM_KINDS",
    "PROOF_KINDS",
    "PROVENANCE_KINDS",
    "RECORD_KINDS",
    "VERIFICATION_STATUSES",
    "DiscoveryRecord",
    "Evidence",
    "EvidenceIntegrityError",
    "EvidenceLedger",
    "EvidenceNotFoundError",
    "EvidenceRecord",
    "EvidenceRejectedError",
    "ProofVerdict",
    "assumption_proof",
    "calculation_datapoint",
    "calculation_proof",
    "canonical_datapoint",
    "cited_record",
    "datapoint_descriptor",
    "discovery_only",
    "edgartools_version",
    "evidence_bundle_entry",
    "evidence_content_hash",
    "evidence_domain",
    "evidence_from_dict",
    "evidence_integrity",
    "evidence_to_dict",
    "finra_record_ref",
    "ingest_evidence",
    "normalize_accession",
    "observation_proof",
    "search_run_ref",
    "sec_record_ref",
    "sec_source_ref",
    "validate_proof",
    "validate_provenance",
    "value_hash",
    "verify_calculation",
    "verify_observation",
    "web_source_ref",
]

ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
"""Canonical SEC accession; a bare 18-digit run normalizes into it (``normalize_accession``)."""

CLAIM_KINDS = ("observed_fact", "calculation", "inference", "assumption")
"""Closed claim vocabulary: what the record asserts, stated by the caller, never inferred from wording.

An observed fact is read off a source, a calculation is derived from proven
inputs by a named formula, an inference reasons over cited rows (it carries no
proof of its own), and an assumption is a declared hypothetical. A
search-derived absence observation is a session coverage artifact
(``ResearchRepository.save_coverage_artifact``), never a ledger row: evidence is
what a source states or code derives, coverage is what a search did or did not reach.
"""

PROOF_KINDS = ("observation", "calculation", "assumption")
"""Closed proof vocabulary: what backs one datapoint row (``Evidence.proof['kind']``)."""

_CLAIM_PROOF_KIND: dict[str, str | None] = {
    "observed_fact": "observation",
    "calculation": "calculation",
    "assumption": "assumption",
    "inference": None,
}
"""The one proof kind each claim kind may carry; an inference carries none."""

VERIFICATION_STATUSES = ("verified", "unverified", "conflicted")
"""Closed verification vocabulary, computed at publication and never stored.

It stays separate from the public link status (``app.research.sources``).
Missing or ambiguous identity is ``unverified``; a value, hash, or byte
disagreement is ``conflicted``. Only ``verified`` may publish.
"""

PROVENANCE_KINDS = ("sec_source", "sec_record", "finra_record", "web_source", "search_run", "none")
"""Closed provenance vocabulary: a reloaded SEC passage, a persisted SEC tool
result, a persisted FINRA tool result, a persisted web-search result, the
executed search, or nothing recorded."""

PROVENANCE_DOMAINS: dict[str, str] = {
    "sec_source": "SEC",
    "sec_record": "SEC",
    "finra_record": "FINRA",
    "web_source": "WEB",
    "search_run": "WEB",
    "none": "SOURCE",
}
"""Kernel-owned provenance -> source-domain mapping (committee labels read this, never re-derive)."""

PROVENANCE_INTEGRITY: dict[str, str] = {
    "sec_source": "PRIMARY_DOCUMENT",
    "sec_record": "CANONICAL_STRUCTURED",
    "finra_record": "CANONICAL_STRUCTURED",
    "web_source": "EXTERNAL_SOURCE",
    "search_run": "EXTERNAL_SOURCE",
    "none": "EXTERNAL_SOURCE",
}
"""Kernel-owned provenance -> integrity-class mapping (same owner, same rule)."""


def evidence_domain(provenance: Mapping[str, object] | None) -> str:
    """Source domain for one provenance mapping (SEC|FINRA|WEB|SOURCE; unknown kinds stay SOURCE)."""
    kind = provenance.get("kind") if isinstance(provenance, Mapping) else None
    return PROVENANCE_DOMAINS.get(kind, "SOURCE") if isinstance(kind, str) else "SOURCE"


def evidence_integrity(provenance: Mapping[str, object] | None) -> str:
    """Integrity class for one provenance mapping (closed vocabulary, never model-assigned)."""
    kind = provenance.get("kind") if isinstance(provenance, Mapping) else None
    return PROVENANCE_INTEGRITY.get(kind, "EXTERNAL_SOURCE") if isinstance(kind, str) else "EXTERNAL_SOURCE"


class EvidenceIntegrityError(ValueError):
    """Hash mismatch, bad confidence, duplicate id, or broken supersede link."""


class EvidenceRejectedError(ValueError):
    """Ingest refusal; carries the journal payload reason."""

    def __init__(self, evidence_id: str, reason: str, detail: str = "") -> None:
        self.evidence_id = evidence_id
        self.reason = reason
        self.detail = detail
        message = f"evidence {evidence_id} rejected: {reason}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


class EvidenceNotFoundError(KeyError):
    """Ledger holds no record for the id."""


def evidence_content_hash(content: str) -> str:
    """Canonical content hash (sha256 hex over utf-8)."""
    return sha256(content.encode("utf-8")).hexdigest()


RECORD_KINDS = frozenset({"discovery", "evidence"})


def normalize_accession(value: object) -> str:
    """Canonical dashed accession; a bare 18-digit run takes the 10-2-6 dashes. ValueError otherwise."""
    text = value.strip() if isinstance(value, str) else ""
    if text.isdigit() and len(text) == 18:
        text = f"{text[:10]}-{text[10:12]}-{text[12:]}"
    if not ACCESSION_RE.match(text):
        raise ValueError(f"invalid accession number: {value!r}")
    return text


BASIS_KINDS = ("raw", "rendered")
"""Closed basis vocabulary for a materialized passage: the stored document text, or its rendered view."""


def _canonical_offset(value: object, key: str) -> int:
    """Non-negative int offset/end of a materialized passage."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sec_source_ref: {key} must be an int >= 0, got {value!r}")
    return value


def _canonical_hash(value: object) -> str:
    """Non-empty sha256 hex of the window the passage was sliced from."""
    text = value.strip() if isinstance(value, str) else ""
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"sec_source_ref: text_hash must be a sha256 hex digest, got {value!r}")
    return text


def sec_source_ref(
    *,
    accession_no: object,
    document_name: object,
    passage: object,
    source_uri: object = None,
    offset: object = None,
    end: object = None,
    basis: object = None,
    text_hash: object = None,
) -> dict[str, JSONValue]:
    """SECSourceRef: the filing document + the kernel-materialized passage an observed fact is read off.

    ``offset``/``end`` are the passage's coordinates in the read ``basis`` (``raw``
    stored text or ``rendered`` view), and ``text_hash`` pins the window the
    kernel reloaded from the archive. They are required: the kernel, never the
    model, materializes the source text.
    """
    document, quoted = _ref_texts(document_name, passage)
    if basis not in BASIS_KINDS:
        raise ValueError(f"sec_source_ref: basis must be one of {list(BASIS_KINDS)}, got {basis!r}")
    start, stop = _ref_window(offset, end)
    return {
        "kind": "sec_source",
        "accession_no": normalize_accession(accession_no),
        "document_name": document,
        "passage": quoted,
        "offset": start,
        "end": stop,
        "basis": basis,
        "text_hash": _canonical_hash(text_hash),
        "source_uri": source_uri.strip() if isinstance(source_uri, str) and source_uri.strip() else None,
    }


def sec_record_ref(
    *,
    tool_name: object,
    record_identity: object,
    dataset: object = None,
    source_uri: object = None,
    known_at: object = None,
    tool_result_id: object = None,
) -> dict[str, JSONValue]:
    """SecRecordRef: one persisted SEC structured-tool result an observed fact is read off."""
    tool = tool_name.strip() if isinstance(tool_name, str) else ""
    identity = record_identity.strip() if isinstance(record_identity, str) else ""
    if not tool or not identity:
        raise ValueError("sec_record_ref: tool_name and record_identity must be non-empty strings")
    ref: dict[str, JSONValue] = {"kind": "sec_record", "tool_name": tool, "record_identity": identity}
    for key, value in (
        ("dataset", dataset),
        ("source_uri", source_uri),
        ("known_at", known_at),
        ("tool_result_id", tool_result_id),
    ):
        text = _provenance_opt({key: value}, key)
        if text is not None:
            ref[key] = text
    return ref


def _ref_texts(document_name: object, passage: object) -> tuple[str, str]:
    """(document, quoted passage) of one source ref; both must be non-empty strings."""
    document = document_name.strip() if isinstance(document_name, str) else ""
    quoted = passage.strip() if isinstance(passage, str) else ""
    if not document or not quoted:
        raise ValueError("sec_source_ref: document_name and passage must be non-empty strings")
    return document, quoted


def _ref_window(offset: object, end: object) -> tuple[int, int]:
    """(offset, end) of the passage inside the read window; the window must be non-empty."""
    start = _canonical_offset(offset, "offset")
    stop = _canonical_offset(end, "end")
    if stop <= start:
        raise ValueError(f"sec_source_ref: end ({stop}) must be greater than offset ({start})")
    return start, stop


def search_run_ref(*, search_id: object, query: object) -> dict[str, JSONValue]:
    """SearchRunRef: the executed search a navigation (discovery) row records; never evidence of a claim."""
    sid = search_id.strip() if isinstance(search_id, str) else ""
    text = query.strip() if isinstance(query, str) else ""
    if not sid or not text:
        raise ValueError("search_run_ref: search_id and query must be non-empty strings")
    return {"kind": "search_run", "search_id": sid, "query": text}


def _provenance_opt(prov: object, key: str) -> str | None:
    """Optional non-blank string of one tool-result ref; absent stays None."""
    value = prov.get(key) if isinstance(prov, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def finra_record_ref(
    *,
    tool_name: object,
    record_identity: object,
    dataset: object = None,
    source_uri: object = None,
    known_at: object = None,
    tool_result_id: object = None,
) -> dict[str, JSONValue]:
    """FinraRecordRef: one persisted FINRA tool result an observed fact is read off.

    ``tool_name`` is the FINRA tool that produced it, ``record_identity`` the
    dataset row/briefing it names. The kernel, never the model, reloads that
    persisted result and stores its bytes; a model-authored row never qualifies.
    ``tool_result_id`` is the persisted kernel tool result this ref replays (None for legacy rows).
    """
    tool = tool_name.strip() if isinstance(tool_name, str) else ""
    identity = record_identity.strip() if isinstance(record_identity, str) else ""
    if not tool or not identity:
        raise ValueError("finra_record_ref: tool_name and record_identity must be non-empty strings")
    ref: dict[str, JSONValue] = {"kind": "finra_record", "tool_name": tool, "record_identity": identity}
    for key, value in (
        ("dataset", dataset),
        ("source_uri", source_uri),
        ("known_at", known_at),
        ("tool_result_id", tool_result_id),
    ):
        text = _provenance_opt({key: value}, key)
        if text is not None:
            ref[key] = text
    return ref


def web_source_ref(
    *,
    url: object,
    excerpt: object,
    title: object = None,
    domain: object = None,
    published_at: object = None,
    retrieved_at: object = None,
    tool_result_id: object = None,
    source_content: object = None,
    source_representation: object = None,
) -> dict[str, JSONValue]:
    """WebSourceRef: one persisted search_web result an observed fact is read off.

    ``url`` is the result URL, ``excerpt`` the highlight text behind the claim.
    The kernel, never the model, reloads that persisted result and stores its
    bytes; a model-authored URL/quote never qualifies.
    ``tool_result_id`` is the persisted kernel tool result this ref replays (None for legacy rows).
    ``source_content`` is the bounded provider extraction the viewer shows (absent on excerpt-only
    rows); the cited claim verifies against ``excerpt`` only, never the full extraction.
    """
    link = url.strip() if isinstance(url, str) else ""
    quote = excerpt.strip() if isinstance(excerpt, str) else ""
    if not link or not quote:
        raise ValueError("web_source_ref: url and excerpt must be non-empty strings")
    ref: dict[str, JSONValue] = {"kind": "web_source", "url": link, "excerpt": quote}
    for key, value in (
        ("title", title),
        ("domain", domain),
        ("published_at", published_at),
        ("retrieved_at", retrieved_at),
        ("tool_result_id", tool_result_id),
    ):
        text = _provenance_opt({key: value}, key)
        if text is not None:
            ref[key] = text
    content = _provenance_opt({"source_content": source_content}, "source_content")
    if content is not None:
        ref["source_content"] = content[:12000]
        representation = (
            _provenance_opt({"source_representation": source_representation}, "source_representation") or "text"
        )
        ref["source_representation"] = representation
    return ref


def _provenance_str(prov: Mapping[str, object], key: str, where: str) -> str:
    value = prov.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceIntegrityError(f"{where}: provenance[{key!r}] must be a non-empty string")
    return value


def _uncanonical_sec_source_ref(value: Mapping[str, object], where: str) -> dict[str, JSONValue]:
    """Pre-canonical SECSourceRef (no window coordinates): read tolerance for rows persisted earlier.

    It preserves exactly what the row recorded — no revision can be re-verified
    for such a row, so it is never written by this version.
    """
    return {
        "kind": "sec_source",
        "accession_no": normalize_accession(value.get("accession_no")),
        "document_name": _provenance_str(value, "document_name", where),
        "passage": _provenance_str(value, "passage", where),
        "source_uri": str(value["source_uri"]) if isinstance(value.get("source_uri"), str) else None,
    }


def validate_provenance(value: object, where: str = "<evidence>: 'provenance'") -> dict[str, JSONValue]:
    """Validate one persisted provenance mapping; empty means 'not recorded' (legacy rows)."""
    if not isinstance(value, Mapping):
        raise EvidenceIntegrityError(f"{where} must be an object")
    if not value:
        return {}
    kind = value.get("kind")
    if kind not in PROVENANCE_KINDS:
        raise EvidenceIntegrityError(f"{where}: 'kind' must be one of {list(PROVENANCE_KINDS)}, got {kind!r}")
    if kind == "none":
        return {"kind": "none"}
    if kind == "search_run":
        return search_run_ref(
            search_id=_provenance_str(value, "search_id", where),
            query=_provenance_str(value, "query", where),
        )
    if kind == "finra_record":
        return finra_record_ref(
            tool_name=_provenance_str(value, "tool_name", where),
            record_identity=_provenance_str(value, "record_identity", where),
            dataset=value.get("dataset"),
            source_uri=value.get("source_uri"),
            known_at=value.get("known_at"),
            tool_result_id=value.get("tool_result_id"),
        )
    if kind == "sec_record":
        return sec_record_ref(
            tool_name=_provenance_str(value, "tool_name", where),
            record_identity=_provenance_str(value, "record_identity", where),
            dataset=value.get("dataset"),
            source_uri=value.get("source_uri"),
            known_at=value.get("known_at"),
            tool_result_id=value.get("tool_result_id"),
        )
    if kind == "web_source":
        return web_source_ref(
            url=_provenance_str(value, "url", where),
            excerpt=_provenance_str(value, "excerpt", where),
            title=value.get("title"),
            domain=value.get("domain"),
            published_at=value.get("published_at"),
            retrieved_at=value.get("retrieved_at"),
            tool_result_id=value.get("tool_result_id"),
            source_content=value.get("source_content"),
            source_representation=value.get("source_representation"),
        )
    return _sec_source_provenance(value, where)


def _sec_source_provenance(value: Mapping[str, object], where: str) -> dict[str, JSONValue]:
    """Canonical SEC ref when the row holds kernel coordinates, else the legacy un-canonical ref."""
    if "text_hash" in value or "offset" in value or "basis" in value:
        try:
            return sec_source_ref(
                accession_no=value.get("accession_no"),
                document_name=_provenance_str(value, "document_name", where),
                passage=_provenance_str(value, "passage", where),
                source_uri=value.get("source_uri"),
                offset=value.get("offset"),
                end=value.get("end"),
                basis=value.get("basis"),
                text_hash=value.get("text_hash"),
            )
        except ValueError as exc:
            raise EvidenceIntegrityError(f"{where}: {exc}") from None
    try:
        return _uncanonical_sec_source_ref(value, where)
    except ValueError as exc:
        raise EvidenceIntegrityError(f"{where}: {exc}") from None


_HEX64 = re.compile(r"[0-9a-f]{64}")

_PROOF_FIELDS: dict[str, tuple[str, ...]] = {
    "observation": ("sourceId", "locator", "valueHash"),
    "calculation": ("formulaId", "formulaVersion", "inputProofIds", "valueHash"),
    "assumption": ("origin", "description"),
}
"""Exact keys per proof kind (besides ``kind``); anything else fails closed."""

_PROOF_OPTIONAL: dict[str, tuple[str, ...]] = {"calculation": ("datapoint",)}
"""Optional keys per proof kind: a calculation row persisted before canonical datapoints lacks one."""


def value_hash(value: str) -> str:
    """Canonical datapoint value hash (sha256 hex over the exact utf-8 text)."""
    return sha256(value.encode("utf-8")).hexdigest()


def _proof_field(key: str, raw: object, where: str) -> JSONValue:
    """One validated proof field; the shape is fixed per key."""
    if key in ("locator", "datapoint"):
        if not isinstance(raw, Mapping) or not raw:
            raise EvidenceIntegrityError(f"{where}: {key!r} must be a non-empty object")
        return validate_json_mapping(dict(raw), where)
    if key == "inputProofIds":
        if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(i, str) or not i.strip() for i in raw):
            raise EvidenceIntegrityError(f"{where}: 'inputProofIds' must be a non-empty list of evidence ids")
        return list(raw)
    if not isinstance(raw, str) or not raw.strip():
        raise EvidenceIntegrityError(f"{where}: {key!r} must be a non-empty string")
    if key == "valueHash" and not _HEX64.fullmatch(raw):
        raise EvidenceIntegrityError(f"{where}: 'valueHash' must be a sha256 hex digest")
    if key == "sourceId" and not (raw.startswith("sha256:") and _HEX64.fullmatch(raw[7:])):
        raise EvidenceIntegrityError(f"{where}: 'sourceId' must be sha256:<hex> of the archived source bytes")
    return raw


def validate_proof(value: object, where: str = "<evidence>: 'proof'") -> dict[str, JSONValue]:
    """Validate one proof mapping; empty means no proof recorded (legacy rows)."""
    if not isinstance(value, Mapping):
        raise EvidenceIntegrityError(f"{where} must be an object")
    if not value:
        return {}
    kind = value.get("kind")
    fields = _PROOF_FIELDS.get(kind) if isinstance(kind, str) else None
    if fields is None:
        raise EvidenceIntegrityError(f"{where}: 'kind' must be one of {list(PROOF_KINDS)}, got {kind!r}")
    optional = _PROOF_OPTIONAL.get(str(kind), ())
    if not {"kind", *fields} <= set(value) <= {"kind", *fields, *optional}:
        raise EvidenceIntegrityError(
            f"{where}: {kind} proof keys must be exactly {['kind', *fields]} (+{list(optional)})"
        )
    proof: dict[str, JSONValue] = {"kind": kind}
    for key in (*fields, *(key for key in optional if key in value)):
        proof[key] = _proof_field(key, value[key], where)
    return proof


def _proof_locator(provenance: Mapping[str, object]) -> dict[str, JSONValue] | None:
    """Identifying locator of one validated provenance; None when the source revision is ambiguous.

    A legacy SEC ref without window coordinates, or a tool-result ref without the
    persisted ``tool_result_id``, cannot pin one exact source, so it locates nothing.
    """
    kind = provenance.get("kind")
    if kind == "sec_source" and "text_hash" in provenance:
        keys: tuple[str, ...] = ("kind", "accession_no", "document_name", "offset", "end", "basis", "text_hash")
    elif kind in ("sec_record", "finra_record") and provenance.get("tool_result_id"):
        keys = ("kind", "tool_name", "record_identity", "tool_result_id")
    elif kind == "web_source" and provenance.get("tool_result_id"):
        keys = ("kind", "url", "tool_result_id")
    else:
        return None
    return validate_json_mapping({key: provenance.get(key) for key in keys}, "<proof>: 'locator'")


_RECORD_LISTS = (
    "filings",
    "transactions",
    "quarterly_eps",
    "facts",
    "matching_concepts",
    "rows",
    "records",
    "documents",
    "events",
)
"""Row lists of a persisted structured tool result (the same keys the replay gate cites)."""

_ENTITY_KEYS = (
    "symbol",
    "symbolCode",
    "securitiesInformationProcessorSymbolIdentifier",
    "ticker",
    "issueSymbolIdentifier",
    "issuerSymbol",
    "entity",
    "entity_id",
    "entity_name",
    "issuer",
    "issuer_cik",
    "subject_cik",
    "cik",
    "company",
    "subject",
)
_PERIOD_KEYS = (
    "period",
    "fiscal_period",
    "frame",
    "period_end",
    "end",
    "settlementDate",
    "settlement_date",
    "tradeReportDate",
    "trade_date",
    "weekStartDate",
    "date",
    "as_of_date",
    "report_date",
    "period_of_report",
)
_UNIT_KEYS = ("unit", "units", "uom", "currency")
_CONCEPT_KEYS = ("concept", "tag")
_VALUE_KEYS = ("value", "val")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|\n")


def _datapoint_text(value: object, key: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"proof: {key} must be a non-empty string")
    return text


def _scalar_text(value: object) -> str | None:
    """Exact text of one JSON scalar field: a string as stored, a number as JSON; bool/null state no value."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    return None


def _record_payload(result: Mapping[str, object]) -> Mapping[str, object]:
    payload = result.get("result")
    return payload if isinstance(payload, Mapping) else result


def _record_text(row: Mapping[str, object]) -> str:
    return json.dumps(dict(row), sort_keys=True, default=str)


def cited_record(result: Mapping[str, object], locator: object) -> Mapping[str, object] | None:
    """Find the persisted row with exactly the cited canonical JSON, else return None.

    A locator must contain the complete row or the complete flat payload.
    JSON key order and formatting do not affect equality.
    String content remains exact.
    """
    text = locator.strip() if isinstance(locator, str) else ""
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    text = _record_text(parsed)
    payload = _record_payload(result)
    rows: list[Mapping[str, object]] = []
    for key in _RECORD_LISTS:
        listed = payload.get(key)
        rows.extend(row for row in (listed if isinstance(listed, list) else []) if isinstance(row, Mapping))
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        rows.append(metrics)
    hits = [row for row in rows if text == _record_text(row)]
    if hits:
        return hits[0]
    return payload if text == _record_text(payload) else None


def _norm_key(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def _record_field(row: Mapping[str, object], metric: str) -> tuple[str, str]:
    """(field key, source metric spelling) the metric names in the row; none or several fail closed."""
    want = _norm_key(metric)
    if not want:
        raise ValueError(f"metric {metric!r} names no field")
    reserved = {_norm_key(key) for key in (*_ENTITY_KEYS, *_PERIOD_KEYS, *_UNIT_KEYS, *_CONCEPT_KEYS, *_VALUE_KEYS)}
    hits = [
        (key, key)
        for key, value in row.items()
        if _norm_key(key) == want and _norm_key(key) not in reserved and _scalar_text(value) is not None
    ]
    for concept_key in _CONCEPT_KEYS:
        concept = row.get(concept_key)
        if isinstance(concept, str) and want in (_norm_key(concept), _norm_key(concept.rsplit(":", 1)[-1])):
            hits.extend((value_key, concept) for value_key in _VALUE_KEYS if value_key in row)
    if len(hits) != 1:
        reason = "is ambiguous in" if hits else "names no field of"
        raise ValueError(f"metric {metric!r} {reason} the cited record")
    return hits[0]


def _record_label(
    row: Mapping[str, object], payload: Mapping[str, object], keys: tuple[str, ...], claimed: str | None, label: str
) -> str | None:
    """Source spelling of one identity field (row first, else payload); a disagreeing claim fails closed.

    Entity and period are required. A unit absent from the source stays None,
    and a claimed unit the source never states fails closed.
    """
    for source in (row, payload):
        values = [text for key in keys if (text := _scalar_text(source.get(key))) is not None]
        if not values:
            continue
        if claimed is None:
            if label != "unit" or len(set(values)) > 1:
                raise ValueError(f"datapoint {label} is required")
            return values[0]
        for text in values:
            if text.casefold() == claimed.casefold():
                return text
        raise ValueError(f"{label} {claimed!r} is not the cited record's {label}")
    if label == "unit" and claimed is None:
        return None
    raise ValueError(f"the cited record states no {label}")


def _record_datapoint(
    provenance: Mapping[str, object], source_bytes: bytes, claimed: Mapping[str, str | None]
) -> dict[str, JSONValue]:
    """Descriptor body of one structured-record datapoint: the exact field of the one cited row."""
    try:
        result = json.loads(source_bytes)
    except ValueError:
        raise ValueError("archived tool result is not JSON") from None
    if not isinstance(result, Mapping):
        raise ValueError("archived tool result is not a JSON object")  # noqa: TRY004 - proof builders raise ValueError
    payload = _record_payload(result)
    row = cited_record(result, provenance.get("record_identity"))
    if row is None:
        raise ValueError("the cited record is absent or ambiguous in the persisted result")
    field_key, metric = _record_field(row, str(claimed["metric"]))
    value = _scalar_text(row.get(field_key))
    if value is None or value != claimed["value"]:
        raise ValueError(f"value {claimed['value']!r} is not the exact {field_key!r} field value")
    return {
        "metric": metric,
        "field": field_key,
        "value": value,
        "entity": _record_label(row, payload, _ENTITY_KEYS, claimed["entity"], "entity"),
        "period": _record_label(row, payload, _PERIOD_KEYS, claimed["period"], "period"),
        "unit": _record_label(row, payload, _UNIT_KEYS, claimed["unit"], "unit"),
        "statement": json.dumps(dict(row), sort_keys=True, default=str),
    }


def _phrase(text: str) -> re.Pattern[str]:
    """Case-insensitive whole-phrase pattern; whitespace runs match any whitespace.

    A word boundary applies only at an alphanumeric edge, so ``%`` still matches in ``4.2%``.
    """
    body = r"\s+".join(re.escape(part) for part in text.split())
    head = r"(?<![0-9A-Za-z])" if text.strip()[:1].isalnum() else ""
    tail = r"(?![0-9A-Za-z])" if text.strip()[-1:].isalnum() else ""
    return re.compile(f"{head}{body}{tail}", re.IGNORECASE)


def _passage_label(passage: str, claimed: str | None, label: str) -> str | None:
    """Source spelling of an entity/period/unit the labeled passage states; absent fails closed."""
    if claimed is None:
        if label == "unit":
            return None
        raise ValueError(f"datapoint {label} is required")
    found = _phrase(claimed).search(passage)
    if found is None:
        raise ValueError(f"{label} {claimed!r} is not stated in the located passage")
    return found.group(0)


def _passage_datapoint(provenance: Mapping[str, object], claimed: Mapping[str, str | None]) -> dict[str, JSONValue]:
    """Bind one metric, value and identity within one sentence of a located passage.

    Reject additional metrics, values and identity from other sentences.
    """
    field_key = "passage" if provenance.get("kind") == "sec_source" else "excerpt"
    passage = provenance.get(field_key)
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError(f"the located {field_key} is empty")
    value = str(claimed["value"])
    hits = list(re.finditer(rf"(?<![0-9A-Za-z.,]){re.escape(value)}(?![0-9A-Za-z]|[.,][0-9])", passage))
    if len(hits) != 1:
        reason = "appears more than once in" if hits else "is not a whole token of"
        raise ValueError(f"value {value!r} {reason} the located {field_key}")
    start, end = hits[0].span()
    sentence_start = max((m.end() for m in _SENTENCE_BREAK.finditer(passage, 0, start)), default=0)
    after = _SENTENCE_BREAK.search(passage, end)
    statement = passage[sentence_start : after.start() if after else len(passage)].strip()
    entity = _passage_label(statement, claimed["entity"], "entity")
    period = _passage_label(statement, claimed["period"], "period")
    unit = _passage_label(statement, claimed["unit"], "unit")
    labels = list(_phrase(str(claimed["metric"])).finditer(passage, sentence_start, start))
    if not labels:
        raise ValueError(f"metric {claimed['metric']!r} does not label the value's source statement")
    between = passage[labels[-1].end() : start]
    for stated in (entity, period, unit):
        between = _phrase(stated).sub(" ", between) if stated else between
    if re.search(r"[0-9]", between):
        raise ValueError(f"metric {claimed['metric']!r} labels another number before value {value!r}")
    remainder = statement
    for selected in (str(claimed["metric"]), value, entity, period, unit):
        remainder = _phrase(selected).sub(" ", remainder) if selected else remainder
    connectors = {"a", "an", "the", "of", "in", "for", "at", "on", "as", "is", "are", "was", "were", "s"}
    if set(re.findall(r"[0-9A-Za-z]+", remainder.casefold())) - connectors:
        raise ValueError("the source statement contains an unrelated metric or value")
    return {
        "metric": labels[-1].group(0),
        "field": field_key,
        "value": value,
        "entity": entity,
        "period": period,
        "unit": unit,
        "statement": statement,
    }


def _datapoint_id(anchor: Mapping[str, object]) -> str:
    """Deterministic datapoint id: sha256 over the canonical JSON of its source anchor (never model prose)."""
    body = json.dumps(anchor, sort_keys=True, separators=(",", ":"), default=str)
    return f"dp:{sha256(body.encode('utf-8')).hexdigest()}"


def canonical_datapoint(
    *,
    provenance: Mapping[str, object],
    source_bytes: bytes,
    metric: object,
    value: object,
    entity: object,
    period: object,
    unit: object = None,
) -> dict[str, JSONValue]:
    """Kernel-built canonical datapoint descriptor of one observation; ValueError when the source refuses it.

    The caller's metric/value/entity/period/unit only select. Every descriptor
    field is read off the source: a structured record's exact field (record rows),
    or a labeled passage sentence (SEC passages, web excerpts). The ``id`` hashes
    the source revision, the locator, and the descriptor body.
    """
    prov = validate_provenance(provenance, "<proof>: 'provenance'")
    locator = _proof_locator(prov)
    if locator is None:
        raise ValueError("provenance does not identify one exact source revision")
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("source_bytes must be the non-empty archived bytes")
    claimed: dict[str, str | None] = {
        "metric": _datapoint_text(metric, "metric"),
        "value": _datapoint_text(value, "value"),
        "entity": _datapoint_text(entity, "entity") if entity is not None else None,
        "period": _datapoint_text(period, "period") if period is not None else None,
        "unit": _datapoint_text(unit, "unit") if unit is not None else None,
    }
    kind = prov.get("kind")
    if kind in ("sec_record", "finra_record"):
        body = _record_datapoint(prov, source_bytes, claimed)
    elif kind in ("sec_source", "web_source"):
        body = _passage_datapoint(prov, claimed)
    else:
        raise ValueError(f"{kind!r} provenance carries no datapoint")
    anchor = {"source": f"sha256:{sha256(source_bytes).hexdigest()}", "locator": locator, "datapoint": body}
    return {"id": _datapoint_id(anchor), **body}


def observation_proof(
    *,
    provenance: Mapping[str, object],
    source_bytes: bytes,
    value: object,
    metric: object,
    entity: object,
    period: object,
    unit: object = None,
) -> dict[str, JSONValue]:
    """Observation proof: archived source bytes, the locator with its canonical datapoint, and the value hash.

    Refuses an ambiguous source identity, absent bytes, and any datapoint the
    source does not state (``canonical_datapoint``). Store the descriptor's
    value/entity/period/unit on the row: they are the source spelling.
    """
    prov = validate_provenance(provenance, "<proof>: 'provenance'")
    datapoint = canonical_datapoint(
        provenance=prov, source_bytes=source_bytes, metric=metric, value=value, entity=entity, period=period, unit=unit
    )
    locator = _proof_locator(prov)
    assert locator is not None  # canonical_datapoint refuses an unlocated provenance
    return {
        "kind": "observation",
        "sourceId": f"sha256:{sha256(source_bytes).hexdigest()}",
        "locator": {**locator, "datapoint": datapoint},
        "valueHash": value_hash(str(datapoint["value"])),
    }


def datapoint_descriptor(evidence: Evidence) -> dict[str, JSONValue] | None:
    """Stored canonical datapoint of one row; None for legacy rows and proof-less kinds.

    It is authentic only when the row's proof verifies (``verify_observation`` /
    ``verify_calculation`` rebuild it from the source and compare).
    """
    kind = evidence.proof.get("kind")
    if kind == "observation":
        locator = evidence.proof.get("locator")
        raw = locator.get("datapoint") if isinstance(locator, Mapping) else None
    elif kind == "calculation":
        raw = evidence.proof.get("datapoint")
    else:
        raw = None
    return dict(raw) if isinstance(raw, Mapping) else None


def calculation_datapoint(
    *,
    formula_id: object,
    formula_version: object,
    inputs: Sequence[Evidence],
    metric: object,
    value: object,
    entity: object,
    period: object,
    unit: object,
) -> dict[str, JSONValue]:
    """Canonical datapoint of one calculation: formula over its inputs' canonical datapoints (argument order).

    ``metric`` is the declared label of the derived quantity (no source states
    it). Entity/period/unit are the calculation row's own fields;
    ``verify_calculation`` checks them against the inputs. Formula, input ids,
    and value are verified. Any input without a canonical datapoint fails.
    """
    formula = f"{_datapoint_text(formula_id, 'formula_id')}@{_datapoint_text(formula_version, 'formula_version')}"
    points = [datapoint_descriptor(row) for row in inputs]
    if not points or any(point is None for point in points):
        raise ValueError("every calculation input needs a canonical datapoint")
    named = [point for point in points if point is not None]
    label = _datapoint_text(metric, "metric")
    text = _datapoint_text(value, "value")
    operands = ", ".join(f"{point['metric']}={point['value']}" for point in named)
    body: dict[str, JSONValue] = {
        "metric": label,
        "field": "formula",
        "formula": formula,
        "value": text,
        "entity": _datapoint_text(entity, "entity"),
        "period": _datapoint_text(period, "period"),
        "unit": _datapoint_text(unit, "unit") if unit is not None else None,
        "statement": f"{label} = {formula}({operands}) = {text}",
        "inputs": [point["id"] for point in named],
    }
    return {"id": _datapoint_id({"formula": formula, "datapoint": body}), **body}


def calculation_proof(
    *,
    formula_id: object,
    formula_version: object,
    input_proof_ids: object,
    value: object,
    datapoint: Mapping[str, object] | None = None,
) -> dict[str, JSONValue]:
    """Calculation proof: formula id + version over input evidence ids (formula argument order).

    ``datapoint`` is the ``calculation_datapoint`` descriptor; the admission path
    adds it once the input rows are loaded.
    """
    proof: dict[str, object] = {
        "kind": "calculation",
        "formulaId": _datapoint_text(formula_id, "formula_id"),
        "formulaVersion": _datapoint_text(formula_version, "formula_version"),
        "inputProofIds": list(input_proof_ids) if isinstance(input_proof_ids, (list, tuple)) else input_proof_ids,
        "valueHash": value_hash(_datapoint_text(value, "value")),
    }
    if datapoint is not None:
        proof["datapoint"] = dict(datapoint)
    return validate_proof(proof, "<proof>")


def assumption_proof(*, origin: object, description: object) -> dict[str, JSONValue]:
    """Assumption proof: who declared it and what it assumes; never a sourced fact."""
    return {
        "kind": "assumption",
        "origin": _datapoint_text(origin, "origin"),
        "description": _datapoint_text(description, "description"),
    }


@dataclass(frozen=True)
class DiscoveryRecord:
    """One catalog/tool-discovery hit: provenance of the search, never substantive coverage."""

    record_id: str
    session_id: str
    tool: str
    query: str
    search_id: str | None = None
    retrieved_at: datetime | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    """One substantive sourced claim over frozen SEC evidence (the coverage unit)."""

    record_id: str
    session_id: str
    evidence_id: str
    claim_text: str


@dataclass(frozen=True)
class Evidence:
    """One sourced claim. Missing timestamps stay None; never invented."""

    evidence_id: str
    session_id: str
    wave_id: int
    source_type: str
    source_name: str
    subject: str
    claim_text: str
    content: str
    content_hash: str
    retrieved_at: datetime
    source_uri: str | None = None
    source_record_id: str | None = None
    published_at: datetime | None = None
    known_at: datetime | None = None
    effective_at: datetime | None = None
    job_id: str | None = None
    agent_id: str | None = None
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    confidence: float | None = None
    quality: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    # Correction chain: id of the prior record this record corrects; None for originals.
    superseded_by: str | None = None
    # discovery = catalog hit (never substantive coverage alone); evidence = sourced claim.
    record_kind: str = "evidence"
    # What the record asserts (never inferred from wording) and what backs it.
    claim_kind: str = "observed_fact"
    provenance: dict[str, JSONValue] = field(default_factory=dict)
    # Typed datapoint (verbatim value text) and what proves it; None/{} on legacy rows.
    value: str | None = None
    entity: str | None = None
    period: str | None = None
    unit: str | None = None
    proof: dict[str, JSONValue] = field(default_factory=dict)

    def _check_record_kind(self) -> None:
        if self.record_kind not in RECORD_KINDS:
            raise EvidenceIntegrityError(
                f"evidence {self.evidence_id}: 'record_kind' must be discovery|evidence, got {self.record_kind!r}"
            )

    def _check_claim_kind(self) -> None:
        if self.claim_kind not in CLAIM_KINDS:
            raise EvidenceIntegrityError(
                f"evidence {self.evidence_id}: 'claim_kind' must be one of {list(CLAIM_KINDS)}, got {self.claim_kind!r}"
            )

    def _check_wave_hash(self) -> None:
        if isinstance(self.wave_id, bool) or not isinstance(self.wave_id, int) or self.wave_id < 1:
            raise EvidenceIntegrityError(f"evidence {self.evidence_id}: 'wave_id' must be an int >= 1")
        if evidence_content_hash(self.content) != self.content_hash:
            raise EvidenceIntegrityError(f"evidence {self.evidence_id}: content_hash mismatch")

    def _check_proof(self) -> dict[str, JSONValue]:
        """Proof kind matches claim kind; valueHash pins the row value; datapoint fields are text."""
        where = f"evidence {self.evidence_id}"
        proof = validate_proof(self.proof, f"{where}: 'proof'")
        if proof and (self.record_kind == "discovery" or proof["kind"] != _CLAIM_PROOF_KIND[self.claim_kind]):
            raise EvidenceIntegrityError(
                f"{where}: {proof['kind']} proof cannot back a {self.record_kind} {self.claim_kind} row"
            )
        for key in ("value", "entity", "period", "unit"):
            text = getattr(self, key)
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise EvidenceIntegrityError(f"{where}: {key!r} must be a non-empty string or null")
        if "valueHash" in proof and (self.value is None or value_hash(self.value) != proof["valueHash"]):
            raise EvidenceIntegrityError(f"{where}: 'value' does not match proof valueHash")
        return proof

    def __post_init__(self) -> None:
        self._check_record_kind()
        self._check_claim_kind()
        self._check_wave_hash()
        proof = self._check_proof()
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EvidenceIntegrityError(f"evidence {self.evidence_id}: 'confidence' must be within 0..1")
        object.__setattr__(self, "supports", tuple(self.supports))
        object.__setattr__(self, "contradicts", tuple(self.contradicts))
        object.__setattr__(
            self,
            "provenance",
            validate_provenance(self.provenance, f"evidence {self.evidence_id}: 'provenance'"),
        )
        object.__setattr__(self, "proof", proof)


class ProofVerdict(NamedTuple):
    """Publication-time verification result: one VERIFICATION_STATUSES value plus why."""

    status: str
    reason: str


_VERIFIED = ProofVerdict("verified", "")


def _datapoint_verdict(evidence: Evidence, kind: str, value: object) -> ProofVerdict | None:
    """Shared preconditions: proof kind, datapoint identity, claimed value vs proven hash."""
    if evidence.proof.get("kind") != kind:
        return ProofVerdict("unverified", f"row carries no {kind} proof")
    missing = [key for key in ("value", "entity", "period") if getattr(evidence, key) is None]
    if missing:
        return ProofVerdict("unverified", f"datapoint identity missing: {', '.join(missing)}")
    claimed = value.strip() if isinstance(value, str) else ""
    if not claimed:
        return ProofVerdict("unverified", "claim carries no value")
    if value_hash(claimed) != evidence.proof["valueHash"]:
        return ProofVerdict("conflicted", f"claimed value {claimed!r} does not match the proven value hash")
    return None


def verify_observation(evidence: Evidence, *, value: object, source_bytes: bytes | None) -> ProofVerdict:
    """Check one claimed value against an observation row: value hash, locator, archived bytes, datapoint.

    ``source_bytes`` are the raw_archive bytes the caller loaded for ``proof.sourceId``;
    None (bytes absent) fails closed as unverified. The stored canonical datapoint
    must rebuild from those bytes exactly; a legacy proof without one stays
    unverified. Never raises.
    """
    early = _datapoint_verdict(evidence, "observation", value)
    if early is not None:
        return early
    locator = _proof_locator(evidence.provenance)
    if locator is None:
        return ProofVerdict("unverified", "provenance does not identify one exact source revision")
    stored = dict(evidence.proof["locator"]) if isinstance(evidence.proof["locator"], Mapping) else {}
    datapoint = stored.pop("datapoint", None)
    if stored != locator:
        return ProofVerdict("conflicted", "proof locator differs from the row provenance")
    if not isinstance(datapoint, Mapping):
        return ProofVerdict("unverified", "legacy observation proof carries no canonical datapoint")
    if source_bytes is None:
        return ProofVerdict("unverified", "archived source bytes absent")
    if f"sha256:{sha256(source_bytes).hexdigest()}" != evidence.proof["sourceId"]:
        return ProofVerdict("conflicted", "archived source bytes do not match proof sourceId")
    try:
        rebuilt = canonical_datapoint(
            provenance=evidence.provenance,
            source_bytes=source_bytes,
            metric=datapoint.get("metric"),
            value=evidence.value,
            entity=evidence.entity,
            period=evidence.period,
            unit=evidence.unit,
        )
    except ValueError as exc:
        return ProofVerdict("conflicted", f"canonical datapoint does not rebuild from the source: {exc}")
    if rebuilt != dict(datapoint):
        return ProofVerdict("conflicted", "stored canonical datapoint differs from the source rebuild")
    return _VERIFIED


def _calculation_metadata(row: Evidence, key: str, alias: str = "") -> object:
    """Read a financial compatibility field from the row's metadata."""
    return row.metadata.get(key, row.metadata.get(alias)) if alias else row.metadata.get(key)


def _calculation_period(row: Evidence) -> tuple[date, date]:
    """Read the explicit interval for a temporal formula."""
    start = date.fromisoformat(str(_calculation_metadata(row, "period_start", "periodStart")))
    end = date.fromisoformat(str(_calculation_metadata(row, "period_end", "periodEnd")))
    if start > end:
        raise ValueError("period_start follows period_end")
    return start, end


def _temporal_input_defect(evidence: Evidence, rows: Sequence[Evidence], semantics: str) -> str | None:
    """Check contiguous quarters and FY/YTD boundaries without inferring dates from labels."""
    try:
        periods = [_calculation_period(row) for row in rows]
        output = _calculation_period(evidence)
    except ValueError:
        return "temporal formulas require valid period_start and period_end"
    if semantics == "quarter":
        if len(periods) != 2:
            return "Q4 requires FY and YTD Q3 inputs"
        (fy_start, fy_end), (ytd_start, ytd_end) = periods
        if (
            fy_start != ytd_start
            or not 330 <= (fy_end - fy_start).days <= 380
            or not 250 <= (ytd_end - ytd_start).days <= 290
            or not 60 <= (fy_end - ytd_end).days <= 110
            or output != (ytd_end + timedelta(days=1), fy_end)
        ):
            return "Q4 requires matching FY and YTD Q3 periods"
        return None
    if len(periods) != 4 or len({row.evidence_id for row in rows}) != 4:
        return "TTM requires four distinct quarters"
    if (
        any(not 60 <= (end - start).days <= 110 for start, end in periods)
        or any(left[1] + timedelta(days=1) != right[0] for left, right in pairwise(periods))
        or output != (periods[0][0], periods[-1][1])
    ):
        return "TTM requires four contiguous quarters in formula argument order"
    return None


def _calculation_input_defect(evidence: Evidence, rows: Sequence[Evidence]) -> str | None:
    """Reject incompatible typed operands before arithmetic or ledger persistence."""
    from .service import _FORMULAS

    formula = _FORMULAS.get((str(evidence.proof["formulaId"]), str(evidence.proof["formulaVersion"])))
    if formula is None:
        return "formula is unknown"
    semantics = formula[1]
    if not evidence.entity or any(
        not row.entity or row.entity.casefold() != evidence.entity.casefold() for row in rows
    ):
        return "calculation entities are incompatible"
    if not evidence.period or any(not row.period for row in rows):
        return "calculation periods are missing"
    row_periods = [row.period.casefold() for row in rows if row.period]
    output_period = evidence.period.casefold() if evidence.period else ""
    same_period = len(set(row_periods)) == 1 and row_periods[0] == output_period
    if semantics in ("growth_pct", "growth_ratio", "change"):
        period_ok = len(rows) == 2 and len(set(row_periods)) == 2 and output_period not in row_periods
    elif semantics == "percentage":
        period_ok = same_period and len(rows) == 2
    elif semantics == "multiple":
        period_ok = len(rows) == 2 and (same_period or (len(set(row_periods)) == 2 and output_period in row_periods))
    elif semantics in ("quarter", "ttm"):
        period_ok = True  # source periods differ per input; _temporal_input_defect checks the intervals
    else:
        period_ok = same_period
    if not period_ok:
        return "calculation periods are incompatible"
    if semantics in ("quarter", "ttm"):
        defect = _temporal_input_defect(evidence, rows, semantics)
        if defect:
            return defect
    units = [row.unit.casefold().replace(" ", "") if row.unit else "" for row in rows]
    output_unit = evidence.unit.casefold().replace(" ", "") if evidence.unit else ""
    if not units or not all(units) or not output_unit:
        return "calculation units are missing"
    same = len(set(units)) == 1
    if semantics in ("add", "quarter", "ttm", "change"):
        valid = same and output_unit == units[0]
    elif semantics in ("growth_pct", "percentage"):
        valid = same and output_unit in ("%", "percent", "percentage")
    elif semantics == "growth_ratio":
        valid = same and output_unit == "ratio"
    elif semantics in ("ratio", "multiple"):
        valid = same and output_unit in ("ratio", "dimensionless", "multiple", "x", "1")
        if len(units) == 2:
            valid = valid or output_unit == f"{units[0]}/{units[1]}"
    elif semantics == "product":
        valid = len(units) == 2 and (
            output_unit == f"{units[0]}*{units[1]}"
            or (units[0] in ("ratio", "dimensionless", "multiple", "x", "1") and output_unit == units[1])
            or (units[1] in ("ratio", "dimensionless", "multiple", "x", "1") and output_unit == units[0])
        )
    elif semantics == "scale":
        valid = len(units) == 2 and output_unit == units[0]
    elif semantics == "per_share":
        valid = len(units) == 2 and output_unit in ("usd/share", "currency/share", "pershare", "$/share")
    else:
        valid = False
    return None if valid else "calculation units are incompatible"


def verify_calculation(
    evidence: Evidence,
    *,
    value: object,
    inputs: Sequence[tuple[Evidence, ProofVerdict]],
    recompute: Callable[[str, str, tuple[str, ...]], str | None],
) -> ProofVerdict:
    """Check one claimed value against a calculation row: verified inputs, formula recompute, value hash.

    ``inputs`` are the rows named by ``proof.inputProofIds`` in that order, each with
    the verdict the caller computed for it. ``recompute(formula_id, formula_version,
    input_values)`` returns the exact value text, or None for an unknown formula.
    Never raises for arithmetic/value errors inside ``recompute``.
    """
    early = _datapoint_verdict(evidence, "calculation", value)
    if early is not None:
        return early
    if [row.evidence_id for row, _ in inputs] != evidence.proof["inputProofIds"]:
        return ProofVerdict("unverified", "input proofs do not resolve to inputProofIds")
    statuses = {verdict.status for _, verdict in inputs}
    if "conflicted" in statuses:
        return ProofVerdict("conflicted", "an input proof is conflicted")
    if statuses != {"verified"}:
        return ProofVerdict("unverified", "an input proof is not verified")
    defect = _calculation_input_defect(evidence, [row for row, _ in inputs])
    if defect is not None:
        return ProofVerdict("unverified", defect)
    formula_id, version = str(evidence.proof["formulaId"]), str(evidence.proof["formulaVersion"])
    try:
        recomputed = recompute(formula_id, version, tuple(row.value or "" for row, _ in inputs))
    except (ArithmeticError, ValueError) as exc:
        return ProofVerdict("unverified", f"formula {formula_id}@{version} failed: {exc}")
    if recomputed is None:
        return ProofVerdict("unverified", f"formula {formula_id}@{version} is unknown")
    if recomputed.strip() != evidence.value:
        return ProofVerdict("conflicted", f"formula recomputes {recomputed.strip()!r}, row states {evidence.value!r}")
    return _calculation_datapoint_verdict(evidence, [row for row, _ in inputs])


def _calculation_datapoint_verdict(evidence: Evidence, rows: Sequence[Evidence]) -> ProofVerdict:
    """The stored calculation datapoint must rebuild from the verified inputs; a legacy row lacks one."""
    stored = evidence.proof.get("datapoint")
    if not isinstance(stored, Mapping):
        return ProofVerdict("unverified", "legacy calculation proof carries no canonical datapoint")
    try:
        rebuilt = calculation_datapoint(
            formula_id=evidence.proof["formulaId"],
            formula_version=evidence.proof["formulaVersion"],
            inputs=rows,
            metric=stored.get("metric"),
            value=evidence.value,
            entity=evidence.entity,
            period=evidence.period,
            unit=evidence.unit,
        )
    except ValueError as exc:
        return ProofVerdict("unverified", f"canonical datapoint does not rebuild: {exc}")
    if rebuilt != dict(stored):
        return ProofVerdict("conflicted", "stored canonical datapoint differs from the input rebuild")
    return _VERIFIED


class EvidenceLedger:
    """In-memory append-only ledger; the only evidence holder, never mutated."""

    def __init__(self) -> None:
        self._records: dict[str, Evidence] = {}

    def append(self, evidence: Evidence) -> Evidence:
        """Append one record; duplicates and dangling supersede links fail."""
        if evidence.evidence_id in self._records:
            raise EvidenceIntegrityError(f"duplicate evidence_id {evidence.evidence_id!r}")
        if evidence.superseded_by is not None and evidence.superseded_by not in self._records:
            raise EvidenceIntegrityError(
                f"evidence {evidence.evidence_id}: superseded_by {evidence.superseded_by!r} not in ledger"
            )
        self._records[evidence.evidence_id] = evidence
        return evidence

    def supersede(self, corrected: Evidence) -> Evidence:
        """Append a correction record; the original stays untouched."""
        if not corrected.superseded_by:
            raise EvidenceIntegrityError("supersede needs corrected.superseded_by set to the prior id")
        return self.append(corrected)

    def get(self, evidence_id: str) -> Evidence:
        """Return the record; raise EvidenceNotFoundError when absent."""
        try:
            return self._records[evidence_id]
        except KeyError:
            raise EvidenceNotFoundError(evidence_id) from None

    def current(self, evidence_id: str) -> Evidence:
        """Follow the supersede chain to the latest correction."""
        # ponytail: O(n) forward-index rebuild per call; index if ledgers grow large.
        forward = {e.superseded_by: e.evidence_id for e in self._records.values() if e.superseded_by is not None}
        seen = self.get(evidence_id)
        while seen.evidence_id in forward:
            seen = self.get(forward[seen.evidence_id])
        return seen

    def list_session(self, session_id: str) -> list[Evidence]:
        """Records for one session, in append order."""
        return [e for e in self._records.values() if e.session_id == session_id]

    def ids(self) -> tuple[str, ...]:
        """All record ids, in append order."""
        return tuple(self._records)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._records

    def __len__(self) -> int:
        return len(self._records)


def _record_kind_of(item: object) -> str:
    """Kind tag for one record: attr wins, then mapping keys, absent means evidence."""
    kind = getattr(item, "record_kind", None)
    if kind is None and isinstance(item, Mapping):
        meta = item.get("metadata")
        meta_kind = meta.get("record_kind") if isinstance(meta, dict) else None
        kind = item.get("record_kind", meta_kind)
    return str(kind) if kind is not None else "evidence"


def _record_items(records: object) -> list[object]:
    """Coerce list/tuple input to a list (anything else means no records)."""
    return list(records) if isinstance(records, (list, tuple)) else []


def discovery_only(records: object) -> bool:
    """True when every record is discovery-kind (no substantive evidence)."""
    items = _record_items(records)
    return bool(items) and all(_record_kind_of(item) == "discovery" for item in items)


def substantive_records(records: object) -> list[object]:
    """Filter to evidence-kind records (discovery never satisfies coverage alone)."""
    return [item for item in _record_items(records) if _record_kind_of(item) == "evidence"]


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else value


def _ingest_required(evidence: Evidence) -> tuple[tuple[str, object], ...]:
    """Required fields. A discovery row is a navigation artifact: it carries no source document.

    Only evidential rows need a source document/ref; a search run is never evidence,
    so a SearchRunRef can satisfy neither ingest nor a citation.
    """
    if evidence.record_kind == "discovery":
        return (
            ("session_id", evidence.session_id),
            ("retrieved_at", evidence.retrieved_at),
            ("lineage", evidence.job_id or evidence.agent_id),
        )
    return (
        ("session_id", evidence.session_id),
        ("source_name", evidence.source_name),
        ("source_ref", evidence.source_uri or evidence.source_record_id),
        ("retrieved_at", evidence.retrieved_at),
        ("lineage", evidence.job_id or evidence.agent_id),
    )


def _ingest_missing(evidence: Evidence) -> list[str]:
    missing = [key for key, value in _ingest_required(evidence) if not value]
    if isinstance(evidence.wave_id, bool) or not isinstance(evidence.wave_id, int):
        missing.append("wave_id")
    return missing


def _ingest_pit_gate(evidence: Evidence, as_of: datetime | str | None) -> tuple[str, str]:
    """PIT eligibility for an evidential row; a navigation artifact has no known_at to check."""
    if evidence.record_kind == "discovery":
        return "", ""
    try:
        unverified = pit_unverified(as_of, evidence.known_at)
        violated = False if unverified else pit_violated(as_of, evidence.known_at)
    except ValueError as exc:
        return "PROVENANCE_FAILURE", f"bad timestamp: {exc}"
    if unverified:
        return "PIT_UNVERIFIED", f"known_at unknown for historical as_of {_iso(as_of)}"
    if violated:
        return "PIT_VIOLATION", f"known_at {_iso(evidence.known_at)} > as_of {_iso(as_of)}"
    return "", ""


def _ingest_reject(
    evidence: Evidence,
    as_of: datetime | str | None,
    reason: str,
    detail: str,
    on_reject: Callable[[str, dict[str, object]], None] | None,
) -> None:
    payload: dict[str, object] = {
        "evidence_id": evidence.evidence_id,
        "reason": reason,
        "known_at": _iso(evidence.known_at),
        "as_of": _iso(as_of),
    }
    if detail:
        payload["detail"] = detail
    if on_reject is not None:
        try:
            on_reject("evidence.rejected", payload)
        except Exception as exc:
            raise EvidenceRejectedError(evidence.evidence_id, reason, detail) from exc
    raise EvidenceRejectedError(evidence.evidence_id, reason, detail)


def ingest_evidence(
    ledger: EvidenceLedger,
    evidence: Evidence,
    *,
    as_of: datetime | str | None,
    on_reject: Callable[[str, dict[str, object]], None] | None = None,
) -> Evidence:
    """Provenance + PIT gate: source/ref/retrieved_at/session/wave/lineage present, known_at <= as_of.

    Discovery rows are navigation artifacts (search/navigation tool results): they
    carry no source document, can never enter a freeze, and are never citable, so
    only their session/retrieval/lineage fields are required and PIT eligibility
    does not apply. Every evidential row keeps the full gate.

    Refusals journal ``evidence.rejected`` ({evidence_id, reason, known_at, as_of})
    via ``on_reject`` (wire KernelCore's append_event with functools.partial) then raise.
    """
    inputs = evidence.proof.get("inputProofIds")
    missing = _ingest_missing(evidence) + [
        f"input proof {eid}" for eid in (inputs if isinstance(inputs, list) else []) if eid not in ledger
    ]
    if missing:
        reason, detail = "PROVENANCE_FAILURE", f"missing: {', '.join(missing)}"
    else:
        reason, detail = _ingest_pit_gate(evidence, as_of)
    if reason:
        _ingest_reject(evidence, as_of, reason, detail, on_reject)
    return ledger.append(evidence)


def evidence_to_dict(evidence: Evidence) -> dict[str, JSONValue]:
    """Evidence -> JSON-able dict (datetimes as ISO); the JSON blob is source of truth."""
    return {
        "evidence_id": evidence.evidence_id,
        "session_id": evidence.session_id,
        "wave_id": evidence.wave_id,
        "source_type": evidence.source_type,
        "source_name": evidence.source_name,
        "source_uri": evidence.source_uri,
        "source_record_id": evidence.source_record_id,
        "subject": evidence.subject,
        "claim_text": evidence.claim_text,
        "content": evidence.content,
        "content_hash": evidence.content_hash,
        "published_at": evidence.published_at.isoformat() if evidence.published_at else None,
        "known_at": evidence.known_at.isoformat() if evidence.known_at else None,
        "effective_at": evidence.effective_at.isoformat() if evidence.effective_at else None,
        "retrieved_at": evidence.retrieved_at.isoformat(),
        "job_id": evidence.job_id,
        "agent_id": evidence.agent_id,
        "supports": _json_str_list(list(evidence.supports)),
        "contradicts": _json_str_list(list(evidence.contradicts)),
        "confidence": evidence.confidence,
        "quality": evidence.quality,
        "metadata": dict(evidence.metadata),
        "superseded_by": evidence.superseded_by,
        "record_kind": evidence.record_kind,
        "claim_kind": evidence.claim_kind,
        "provenance": dict(evidence.provenance),
        "value": evidence.value,
        "entity": evidence.entity,
        "period": evidence.period,
        "unit": evidence.unit,
        "proof": dict(evidence.proof),
        "source_domain": evidence_domain(evidence.provenance),
        "integrity_class": evidence_integrity(evidence.provenance),
    }


def _req_str(d: dict[str, object], key: str) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise EvidenceIntegrityError(f"evidence: '{key}' must be a non-empty string")
    return value


def _opt_str(d: dict[str, object], key: str) -> str | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceIntegrityError(f"evidence: '{key}' must be a string or null")
    return value or None


def _req_dt(d: dict[str, object], key: str) -> datetime:
    value = d.get(key)
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            pass
        else:
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    raise EvidenceIntegrityError(f"evidence: '{key}' must be an ISO-8601 datetime")


def _opt_dt(d: dict[str, object], key: str) -> datetime | None:
    if d.get(key) is None:
        return None
    return _req_dt(d, key)


def _req_int(d: dict[str, object], key: str) -> int:
    value = d.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceIntegrityError(f"evidence: '{key}' must be an int")
    return value


def _str_list(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
        raise EvidenceIntegrityError(f"evidence: '{key}' must be a list of strings")
    return tuple(value)


def _json_str_list(values: list[str]) -> list[JSONValue]:
    out: list[JSONValue] = []
    out.extend(values)
    return out


def _evidence_confidence(d: dict[str, object]) -> float | None:
    confidence = d.get("confidence")
    if confidence is None:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise EvidenceIntegrityError("evidence: 'confidence' must be a number or null")
    return float(confidence)


def _evidence_metadata(d: dict[str, object]) -> dict[str, object]:
    metadata = d.get("metadata", {})
    if not isinstance(metadata, dict):
        raise EvidenceIntegrityError("evidence: 'metadata' must be an object")
    return dict(metadata)


def _evidence_record_kind(d: dict[str, object]) -> str:
    """discovery|evidence; metadata.record_kind fallback; absent means evidence (back-compat)."""
    raw = d.get("record_kind")
    if raw is None:
        meta = d.get("metadata")
        raw = meta.get("record_kind") if isinstance(meta, dict) else None
    if raw is None:
        return "evidence"
    if isinstance(raw, str) and raw in RECORD_KINDS:
        return raw
    raise EvidenceIntegrityError(f"evidence: 'record_kind' must be discovery|evidence, got {raw!r}")


def _evidence_claim_kind(d: dict[str, object]) -> str:
    """claim_kind; absent means observed_fact so persisted history stays readable."""
    raw = d.get("claim_kind")
    if raw is None:
        return "observed_fact"
    if isinstance(raw, str) and raw in CLAIM_KINDS:
        return raw
    raise EvidenceIntegrityError(f"evidence: 'claim_kind' must be one of {list(CLAIM_KINDS)}, got {raw!r}")


def _evidence_proof(d: dict[str, object]) -> dict[str, JSONValue]:
    """proof; absent or null means no proof recorded so persisted history stays readable."""
    raw = d.get("proof")
    return {} if raw is None else validate_proof(raw)


def evidence_from_dict(data: Mapping[str, object]) -> Evidence:
    """Rebuild validated Evidence (constructor re-checks hash/confidence)."""
    d = dict(data)
    provenance = validate_provenance(d.get("provenance", {}), "<evidence>: 'provenance'")
    for key, expected in (
        ("source_domain", evidence_domain(provenance)),
        ("integrity_class", evidence_integrity(provenance)),
    ):
        raw = d.get(key)
        if raw is not None and raw != expected:
            raise EvidenceIntegrityError(f"evidence: '{key}' must match provenance-derived {expected!r}, got {raw!r}")
    return Evidence(
        evidence_id=_req_str(d, "evidence_id"),
        session_id=_req_str(d, "session_id"),
        wave_id=_req_int(d, "wave_id"),
        source_type=_req_str(d, "source_type"),
        source_name=_req_str(d, "source_name"),
        subject=_req_str(d, "subject"),
        claim_text=_req_str(d, "claim_text"),
        content=_req_str(d, "content"),
        content_hash=_req_str(d, "content_hash"),
        retrieved_at=_req_dt(d, "retrieved_at"),
        source_uri=_opt_str(d, "source_uri"),
        source_record_id=_opt_str(d, "source_record_id"),
        published_at=_opt_dt(d, "published_at"),
        known_at=_opt_dt(d, "known_at"),
        effective_at=_opt_dt(d, "effective_at"),
        job_id=_opt_str(d, "job_id"),
        agent_id=_opt_str(d, "agent_id"),
        supports=_str_list(d.get("supports", []), "supports"),
        contradicts=_str_list(d.get("contradicts", []), "contradicts"),
        confidence=_evidence_confidence(d),
        metadata=validate_json_mapping(_evidence_metadata(d), "<evidence>: 'metadata'"),
        superseded_by=_opt_str(d, "superseded_by"),
        record_kind=_evidence_record_kind(d),
        claim_kind=_evidence_claim_kind(d),
        provenance=provenance,
        value=_opt_str(d, "value"),
        entity=_opt_str(d, "entity"),
        period=_opt_str(d, "period"),
        unit=_opt_str(d, "unit"),
        proof=_evidence_proof(d),
    )


def edgartools_version() -> str | None:
    """Installed edgartools version, or None when unresolvable (provenance metadata, never a gate)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("edgartools")
    except PackageNotFoundError:
        return None


def _bundle_accession(evidence: Evidence, prov: Mapping[str, object]) -> str:
    """Prefer the provenance accession to the record identifier, without normalizing either."""
    accession = evidence.source_record_id or ""
    raw_accession = prov.get("accession_no")
    if isinstance(raw_accession, str) and raw_accession:
        accession = raw_accession
    return accession


def _bundle_document_name(evidence: Evidence, prov: Mapping[str, object]) -> str:
    """Prefer the provenance document name to the metadata fallback."""
    document = prov.get("document_name")
    if not isinstance(document, str) or not document:
        document = str(evidence.metadata.get("document_name", "") or "")
    return document


def _bundle_source_url(evidence: Evidence) -> str:
    """Canonical source URL: the explicit URI, else the SEC archive URL for the accession."""
    if isinstance(evidence.source_uri, str) and evidence.source_uri.strip():
        return evidence.source_uri.strip()
    prov = evidence.provenance if isinstance(evidence.provenance, Mapping) else {}
    uri = prov.get("source_uri")
    if isinstance(uri, str) and uri.strip():
        return uri.strip()
    bare = _bundle_accession(evidence, prov).replace("-", "")
    if bare:
        return f"https://www.sec.gov/Archives/edgar/data/{bare}"
    return ""


def _bundle_locator(evidence: Evidence) -> dict[str, JSONValue]:
    """Source locator: document + kernel-materialized window coordinates (the reload recipe)."""
    prov = evidence.provenance if isinstance(evidence.provenance, Mapping) else {}
    document = _bundle_document_name(evidence, prov)
    offset = prov.get("offset")
    end = prov.get("end")
    locator: dict[str, JSONValue] = {"document": document}
    locator["section"] = prov.get("section") if isinstance(prov.get("section"), str) else None
    locator["start"] = offset if isinstance(offset, int) and not isinstance(offset, bool) else None
    locator["end"] = end if isinstance(end, int) and not isinstance(end, bool) else None
    return locator


def evidence_bundle_entry(evidence: Evidence, *, accepted_at: str | None = None) -> dict[str, JSONValue]:
    """Per-session bundle evidence entry: the 12-file Contract fields with exact model-visible text."""
    prov = evidence.provenance if isinstance(evidence.provenance, Mapping) else {}
    accession = _bundle_accession(evidence, prov)
    form = evidence.metadata.get("form") or evidence.metadata.get("filing_form")
    retrieved = evidence.retrieved_at.isoformat()
    document_name = _bundle_document_name(evidence, prov)
    return {
        "evidence_id": evidence.evidence_id,
        "session_id": evidence.session_id,
        "source": evidence.source_type or "sec",
        "accession": accession,
        "form": form if isinstance(form, str) else None,
        "source_url": _bundle_source_url(evidence),
        "accepted_at": accepted_at or retrieved,
        "retrieved_at": retrieved,
        "edgartools_version": edgartools_version(),
        "rendered_to_model": evidence.content,
        "source_locator": _bundle_locator(evidence),
        # source_hash pins model-visible text, not filing bytes; exact bytes live in raw_archive on acceptance.
        "source_hash": f"sha256:{evidence.content_hash}",
        "source_artifact": (
            f"source://sec/{accession}/{document_name}"
            if accession and document_name and evidence.metadata.get("source_bytes") == "archived"
            else None
        ),
    }
