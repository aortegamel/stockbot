"""Regression fixtures: promote sessions into human-readable deterministic fixtures (s35).

Deterministic validators run first; a live regression run is only warranted
when a failure needs model reasoning to reproduce. Fixtures live under
evals/fixtures/agent_scenarios/<scenario-name>.json (indent=2 JSON).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from app.research.evals.scenarios import get_scenario
from app.research.models import JSONValue

FIXTURE_FORMAT = "agent-scenario-fixture/v1"

# s35 deterministic-first mapping: each validator replaces a live reproduction.
# future-accepted -> PIT validator, dangling-dossier -> dossier validator,
# over-budget -> scheduler check, bull-new-research -> capability policy,
# freeze-mutate -> immutable-freeze constraint.
VALIDATORS: tuple[str, ...] = (
    "pit",
    "dossier",
    "scheduler-budget",
    "capability-policy",
    "immutable-freeze",
    "timeout_resume_requires_failed",
)

# Timeout closure (failure-closure invariant timeout-requires-failed-job): a Pi
# TimeoutExpired must persist session FAILED + job FAILED(TIMEOUT) +
# model.failed/wave.stopped with zero RUNNING jobs before LiveModelError
# surfaces. The promoted fixture carries that closure in answer_excerpt, so
# this stays deterministic over fixture JSON alone; a pre-fix "Pi timed out"
# excerpt with no closure markers fails.
_TIMEOUT_CLOSURE_MARKERS: tuple[str, ...] = (
    "LiveModelError",
    "session=failed",
    "job=failed",
    "model.failed",
    "wave.stopped",
    "running_jobs=0",
)


class ResearchTelemetry(TypedDict, total=False):
    """Research-quality telemetry: cheap counters recorded alongside a fixture."""

    searches: int
    queries: list[str]
    forms: list[str]
    entities: list[str]
    exhibits: int
    relationships_found: int
    relationships_skipped: int
    coverage: str
    unresolved: list[str]
    stop_reason: str


class FixtureClaim(TypedDict):
    """One persisted claim: declared type plus how the answer rendered it ("" = undeclared)."""

    text: str
    claim_type: str
    rendered_as: str


class FixtureValidator(TypedDict):
    pit_as_of: str | None
    expected_tools: list[str]
    requires_evidence: bool
    validators: list[str]


class FixtureTrace(TypedDict, total=False):
    """Structural run trace (plan Phase 17): what the run actually opened, cited and froze.

    Optional fixture section; a fixture without it stays readable, but a
    trace-gated scenario cannot pass its evaluation without it.
    """

    filings_opened: list[str]
    documents_opened: list[str]
    passages_opened: list[str]
    raw_evidence_ids: list[str]
    navigation_evidence_ids: list[str]
    claims: list[FixtureClaim]
    claims_by_type: dict[str, int]
    waves: list[str]
    searches: list[str]
    committee_freeze_ids: list[str]
    roles_completed: list[str]
    limitations: list[str]
    coverage_complete: bool
    universal_absence_claims: list[str]
    material_channels: list[str]
    branches_covered: list[str]


class AgentFixture(TypedDict):
    format: str
    scenario_name: str
    family: str
    session_id: str
    question: str
    as_of: str | None
    tool_calls: list[str]
    evidence_ids: list[str]
    known_ats: list[str]
    answer_excerpt: str
    budget_used: int | None
    budget_cap: int | None
    freeze_before: str | None
    freeze_after: str | None
    validator: FixtureValidator
    telemetry: NotRequired[ResearchTelemetry]
    trace: NotRequired[FixtureTrace]


# OpenAI-bankruptcy MSFT regression gate (fixture cutoff 2026-08-10): a run
# counts as covering Microsoft material exposure only when the excerpt names
# at least one exposure channel and at least one non-MSFT branch, so a
# Microsoft-only answer still fails. No facts past the cutoff are asserted.
_MSFT_CHANNELS: tuple[str, ...] = (
    "investment",
    "ownership",
    "commercial",
    "revenue",
    "receivable",
    "credit",
    "azure",
    "purchase commitment",
)

_MSFT_NON_MSFT_BRANCHES: tuple[str, ...] = ("amzn", "amazon", "coreweave", "amd", "cerebras", "orcl", "oracle")


def _v_fixture_msft_openai(fixture: AgentFixture) -> str | None:
    """MSFT OpenAI-bankruptcy gate: material MSFT channel + one non-MSFT branch.

    Trace channel/branch coverage counts alongside the answer excerpt, so a
    structurally covered run never depends on prose wording alone.
    """
    if fixture["scenario_name"] != "msft-openai-bankruptcy-sec-only":
        return None
    trace = fixture.get("trace") or {}
    channel_text = " ".join((*trace.get("material_channels", ()), fixture["answer_excerpt"])).lower()
    if not any(channel in channel_text for channel in _MSFT_CHANNELS):
        return "msft-openai-no-material-msft-exposure"
    branch_text = " ".join((*trace.get("branches_covered", ()), fixture["answer_excerpt"])).lower()
    if not any(branch in branch_text for branch in _MSFT_NON_MSFT_BRANCHES):
        return "msft-openai-no-branch"
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def default_fixtures_dir() -> Path:
    """Repo fixtures dir (overridable per call for tests)."""
    return _repo_root() / "evals" / "fixtures" / "agent_scenarios"


def _ledger_strs(rows: object, key: str) -> list[str]:
    """Non-empty strings under one final/synthesis key (missing sections stay empty)."""
    if not isinstance(rows, dict):
        return []
    raw = rows.get(key)
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _ledger_claims(final: object) -> list[FixtureClaim]:
    """Declared claims from persisted final claims (explicit type only, never inferred)."""
    if not isinstance(final, dict):
        return []
    raw = final.get("claims")
    if not isinstance(raw, list):
        return []
    claims: list[FixtureClaim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        declared = item.get("claim_type")
        rendered = item.get("rendered_as")
        claims.append(
            FixtureClaim(
                text=text,
                claim_type=declared if isinstance(declared, str) else "",
                rendered_as=rendered if isinstance(rendered, str) else "",
            )
        )
    return claims


def _ledger_counts(claims: list[FixtureClaim]) -> dict[str, int]:
    """Declared-type counts from persisted claims (undeclared entries stay out of the map)."""
    counts: dict[str, int] = {}
    for claim in claims:
        if claim["claim_type"]:
            counts[claim["claim_type"]] = counts.get(claim["claim_type"], 0) + 1
    return counts


def _ledger_eids(rows: list[dict[str, object]], kind: str) -> list[str]:
    """Ledger evidence ids of one record kind (discovery rows are navigation)."""
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("record_kind") != kind:
            continue
        eid = row.get("evidence_id")
        if isinstance(eid, str) and eid and eid not in ids:
            ids.append(eid)
    return ids


def _sec_provenance_ids(rows: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    """(accessions, accession|document entries) of sec_source rows, in first-seen order."""
    accessions: list[str] = []
    documents: list[str] = []
    for row in rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("kind") != "sec_source":
            continue
        accession = provenance.get("accession_no")
        if isinstance(accession, str) and accession and accession not in accessions:
            accessions.append(accession)
        document = provenance.get("document_name")
        if isinstance(document, str) and isinstance(accession, str) and accession:
            entry = f"{accession}|{document}"
            if entry not in documents:
                documents.append(entry)
    return accessions, documents


def _cited_present_ids(final_dict: Mapping[str, object], rows: list[dict[str, object]]) -> list[str]:
    """Final-record filing references that the ledger actually carries."""
    refs = final_dict.get("filing_references")
    cited: list[str] = [item for item in refs if isinstance(item, str) and item] if isinstance(refs, list) else []
    present = {row.get("evidence_id") for row in rows}
    return [item for item in cited if item in present]


def _freeze_ids_of(freeze_ids: tuple[str, ...], freeze_id: str | None) -> list[str]:
    """Committee freeze ids when given, else the single freeze id, else none."""
    if freeze_ids:
        return [fid for fid in freeze_ids if isinstance(fid, str) and fid]
    return [freeze_id] if freeze_id else []


def _session_trace(
    final: object,
    evidence_rows: object,
    *,
    freeze_ids: tuple[str, ...] = (),
    freeze_id: str | None = None,
    committee_ids: tuple[str, ...] = (),
    roles: tuple[str, ...] = (),
) -> FixtureTrace:
    """Trace fields derivable from persisted final/ledger rows (missing sections stay empty)."""
    rows: list[dict[str, object]] = []
    if isinstance(evidence_rows, list):
        rows = [row for row in evidence_rows if isinstance(row, dict)]
    final_dict = final if isinstance(final, dict) else {}
    accessions, documents = _sec_provenance_ids(rows)
    raw_ids = _cited_present_ids(final_dict, rows)
    claims = _ledger_claims(final)
    navigation = [eid for eid in _ledger_eids(rows, "discovery") if eid not in raw_ids]
    return FixtureTrace(
        filings_opened=accessions,
        documents_opened=documents,
        passages_opened=[],
        raw_evidence_ids=raw_ids,
        navigation_evidence_ids=navigation,
        claims=claims,
        claims_by_type=_ledger_counts(claims),
        waves=_freeze_ids_of(freeze_ids, freeze_id),
        searches=[],
        committee_freeze_ids=list(committee_ids),
        roles_completed=list(roles),
        limitations=_ledger_strs(final_dict, "limitations"),
        coverage_complete=False,
        universal_absence_claims=[],
        material_channels=[],
        branches_covered=[],
    )


def build_fixture(
    *,
    session_id: str,
    scenario_name: str,
    question: str | None = None,
    as_of: str | None = None,
    tool_calls: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    known_ats: tuple[str, ...] = (),
    answer_excerpt: str = "",
    budget_used: int | None = None,
    budget_cap: int | None = None,
    freeze_before: str | None = None,
    freeze_after: str | None = None,
    telemetry: ResearchTelemetry | None = None,
    trace: FixtureTrace | None = None,
) -> AgentFixture:
    """Assemble a fixture; question/as_of default to the scenario definition."""
    scenario = get_scenario(scenario_name)
    resolved_q = question if question is not None else scenario.question
    resolved_as_of = as_of if as_of is not None else scenario.as_of
    fixture: AgentFixture = {
        "format": FIXTURE_FORMAT,
        "scenario_name": scenario.name,
        "family": scenario.family.value,
        "session_id": session_id,
        "question": resolved_q,
        "as_of": resolved_as_of,
        "tool_calls": list(tool_calls),
        "evidence_ids": list(evidence_ids),
        "known_ats": list(known_ats),
        "answer_excerpt": answer_excerpt,
        "budget_used": budget_used,
        "budget_cap": budget_cap,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "validator": {
            "pit_as_of": resolved_as_of,
            "expected_tools": list(scenario.expected_tools),
            "requires_evidence": scenario.requires_evidence,
            "validators": list(VALIDATORS),
        },
    }
    if telemetry is not None:
        fixture["telemetry"] = telemetry
    if trace is not None:
        fixture["trace"] = trace
    return fixture


def save_fixture(fixture: AgentFixture, fixtures_dir: Path | None = None) -> Path:
    """Write a human-readable fixture; returns its path."""
    dest_dir = fixtures_dir if fixtures_dir is not None else default_fixtures_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{fixture['scenario_name']}.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def promote_to_fixture(
    *,
    session_id: str,
    scenario_name: str,
    question: str | None = None,
    as_of: str | None = None,
    tool_calls: tuple[str, ...] | None = None,
    evidence_ids: tuple[str, ...] | None = None,
    known_ats: tuple[str, ...] | None = None,
    answer_excerpt: str | None = None,
    fixtures_dir: Path | None = None,
    repo: object | None = None,
) -> Path:
    """Promote one durable session: carry its ledger trace into the fixture."""
    from app.research.repository import ResearchRepository

    store = repo if isinstance(repo, ResearchRepository) else ResearchRepository()
    try:
        found = store.get_session(session_id)
    except KeyError:
        raise ValueError(f"promote: unknown session_id: {session_id!r}") from None
    final: dict[str, JSONValue] = found.final_result if isinstance(found.final_result, dict) else {}
    rows = store.list_evidence(session_id)
    jobs = store.list_jobs(session_id)
    freeze_id = final.get("freeze_id") if isinstance(final, dict) else None
    freeze_ids = tuple(found.freeze_ids)
    committee: list[str] = []
    for entry in found.committee_runs:
        if isinstance(entry, dict):
            fid = entry.get("freeze_id")
            got = entry.get("jobs")
            if isinstance(fid, str) and isinstance(got, list):
                committee.extend([fid] * sum(1 for jid in got if isinstance(jid, str)))
    known = [str(row.get("known_at")) for row in rows if isinstance(row, dict) and isinstance(row.get("known_at"), str)]
    completed = [job.job_type for job in jobs if job.status == "completed"]
    roles = tuple(role for role in ("stockbot", "bullbot", "bearbot") if role in completed)
    seen_tools: list[str] = []
    for job in jobs:
        tool = (job.diagnostics or {}).get("tool_name") if isinstance(job.diagnostics, dict) else None
        name = tool if isinstance(tool, str) and tool else job.job_type
        if name not in seen_tools:
            seen_tools.append(name)
    final_answer = final.get("answer") if isinstance(final, dict) else None
    derived = _session_trace(
        final,
        rows,
        freeze_ids=freeze_ids,
        freeze_id=freeze_id if isinstance(freeze_id, str) else None,
        committee_ids=tuple(committee) or freeze_ids,
        roles=roles,
    )
    fixture = build_fixture(
        session_id=session_id,
        scenario_name=scenario_name,
        question=question if question is not None else (found.query or None),
        as_of=as_of if as_of is not None else None,
        tool_calls=tuple(tool_calls) if tool_calls is not None else tuple(seen_tools),
        evidence_ids=tuple(evidence_ids) if evidence_ids is not None else tuple(derived["raw_evidence_ids"]),
        known_ats=tuple(known_ats) if known_ats is not None else tuple(known),
        answer_excerpt=answer_excerpt
        if answer_excerpt is not None
        else (final_answer if isinstance(final_answer, str) else ""),
        freeze_before=freeze_ids[-1] if freeze_ids else None,
        freeze_after=freeze_ids[-1] if freeze_ids else None,
        trace=derived,
    )
    return save_fixture(fixture, fixtures_dir)


def _req_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"fixture: {key!r} must be a string")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return value


def _opt_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"fixture: {key!r} must be a string or null")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return value


def _opt_int(raw: dict[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fixture: {key!r} must be an int or null")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return value


def _req_str_list(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"fixture: {key!r} must be a list of strings")
    return [item for item in value if isinstance(item, str)]


_TelemetryIntKey = Literal["searches", "exhibits", "relationships_found", "relationships_skipped"]
_TelemetryListKey = Literal["queries", "forms", "entities", "unresolved"]
_TelemetryStrKey = Literal["coverage", "stop_reason"]
_TELEMETRY_INT_KEYS: tuple[_TelemetryIntKey, ...] = (
    "searches",
    "exhibits",
    "relationships_found",
    "relationships_skipped",
)
_TELEMETRY_LIST_KEYS: tuple[_TelemetryListKey, ...] = ("queries", "forms", "entities", "unresolved")
_TELEMETRY_STR_KEYS: tuple[_TelemetryStrKey, ...] = ("coverage", "stop_reason")


def _valid_telemetry_int(value: object, key: _TelemetryIntKey) -> int:
    """Validated telemetry int (ValueError on bool/mistyped)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fixture: 'telemetry.{key}' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return value


def _valid_telemetry_strs(value: object, key: _TelemetryListKey) -> list[str]:
    """Validated telemetry string list (ValueError on mistyped entries)."""
    if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
        raise ValueError(f"fixture: 'telemetry.{key}' must be a list of strings")
    return list(value)


def _copy_telemetry_int(out: ResearchTelemetry, value: dict[str, object], key: _TelemetryIntKey) -> None:
    """Copy one validated telemetry int (ValueError on bool/mistyped)."""
    item = value.get(key)
    if item is not None:
        out[key] = _valid_telemetry_int(item, key)


def _copy_telemetry_strs(out: ResearchTelemetry, value: dict[str, object], key: _TelemetryListKey) -> None:
    """Copy one validated telemetry string list (ValueError on mistyped entries)."""
    item = value.get(key)
    if item is not None:
        out[key] = _valid_telemetry_strs(item, key)


def _copy_telemetry_str(out: ResearchTelemetry, value: dict[str, object], key: _TelemetryStrKey) -> None:
    """Copy one validated telemetry string field (ValueError on mistyped)."""
    item = value.get(key)
    if item is not None:
        if not isinstance(item, str):
            raise ValueError(f"fixture: 'telemetry.{key}' must be a string")
        out[key] = item


def _opt_telemetry(raw: dict[str, object]) -> ResearchTelemetry | None:
    value = raw.get("telemetry")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("fixture: 'telemetry' must be an object")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    out: ResearchTelemetry = {}
    for key in _TELEMETRY_INT_KEYS:
        _copy_telemetry_int(out, value, key)
    for key in _TELEMETRY_LIST_KEYS:
        _copy_telemetry_strs(out, value, key)
    for key in _TELEMETRY_STR_KEYS:
        _copy_telemetry_str(out, value, key)
    return out


def _trace_claim(raw: object) -> FixtureClaim:
    """One persisted claim; an undeclared type stays "", never invented."""
    if not isinstance(raw, dict):
        raise ValueError("fixture: 'trace.claims' entries must be objects")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    text = raw.get("text")
    if not isinstance(text, str):
        raise ValueError("fixture: 'trace.claims[].text' must be a string")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    declared = raw.get("claim_type")
    rendered = raw.get("rendered_as")
    return FixtureClaim(
        text=text,
        claim_type=declared if isinstance(declared, str) else "",
        rendered_as=rendered if isinstance(rendered, str) else "",
    )


def _opt_claims_by_type(value: dict[str, object]) -> dict[str, int]:
    """Declared claim-type counts for one trace section ({} when absent)."""
    raw = value.get("claims_by_type")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("fixture: 'trace.claims_by_type' must be an object")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    out: dict[str, int] = {}
    for key, item in raw.items():
        if isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool):
            out[key] = item
    return out


def _opt_str_list(value: dict[str, object], key: str) -> list[str]:
    """One optional string list inside a section ([] when absent)."""
    raw = value.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"fixture: 'trace.{key}' must be a list of strings")
    return [item for item in raw if isinstance(item, str)]


def _trace_claims(value: dict[str, object]) -> list[FixtureClaim]:
    """One trace section's persisted claims ([] when the section has none)."""
    claims = value.get("claims")
    if claims is None:
        return []
    if not isinstance(claims, list):
        raise ValueError("fixture: 'trace.claims' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return [_trace_claim(item) for item in claims]


def _opt_bool(value: dict[str, object], key: str) -> bool | None:
    """One optional bool field (None when absent, ValueError when mistyped)."""
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ValueError(f"fixture: 'trace.{key}' must be a bool")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return raw


def _opt_trace(raw: dict[str, object]) -> FixtureTrace | None:
    """Read the optional trace section; None keeps pre-Phase-17 fixtures readable."""
    value = raw.get("trace")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("fixture: 'trace' must be an object")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    trace: FixtureTrace = {
        "filings_opened": _opt_str_list(value, "filings_opened"),
        "documents_opened": _opt_str_list(value, "documents_opened"),
        "passages_opened": _opt_str_list(value, "passages_opened"),
        "raw_evidence_ids": _opt_str_list(value, "raw_evidence_ids"),
        "navigation_evidence_ids": _opt_str_list(value, "navigation_evidence_ids"),
        "claims": _trace_claims(value),
        "claims_by_type": _opt_claims_by_type(value),
        "waves": _opt_str_list(value, "waves"),
        "searches": _opt_str_list(value, "searches"),
        "committee_freeze_ids": _opt_str_list(value, "committee_freeze_ids"),
        "roles_completed": _opt_str_list(value, "roles_completed"),
        "limitations": _opt_str_list(value, "limitations"),
        "universal_absence_claims": _opt_str_list(value, "universal_absence_claims"),
        "material_channels": _opt_str_list(value, "material_channels"),
        "branches_covered": _opt_str_list(value, "branches_covered"),
    }
    complete = _opt_bool(value, "coverage_complete")
    if complete is not None:
        trace["coverage_complete"] = complete
    return trace


def _fixture_raw(path: Path) -> tuple[dict[str, object], dict[str, object], bool]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"fixture {path}: top level must be an object")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    raw: dict[str, object] = {k: v for k, v in decoded.items() if isinstance(k, str)}
    validator_raw = raw.get("validator")
    if not isinstance(validator_raw, dict):
        raise ValueError(f"fixture {path}: validator must be an object")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    vraw: dict[str, object] = {k: v for k, v in validator_raw.items() if isinstance(k, str)}
    flag = vraw.get("requires_evidence")
    if not isinstance(flag, bool):
        raise ValueError(f"fixture {path}: requires_evidence must be bool")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return raw, vraw, flag


def _fixture_body(raw: dict[str, object], vraw: dict[str, object], flag: bool) -> AgentFixture:
    fixture: AgentFixture = {
        "format": _req_str(raw, "format"),
        "scenario_name": _req_str(raw, "scenario_name"),
        "family": _req_str(raw, "family"),
        "session_id": _req_str(raw, "session_id"),
        "question": _req_str(raw, "question"),
        "as_of": _opt_str(raw, "as_of"),
        "tool_calls": _req_str_list(raw, "tool_calls"),
        "evidence_ids": _req_str_list(raw, "evidence_ids"),
        "known_ats": _req_str_list(raw, "known_ats"),
        "answer_excerpt": _req_str(raw, "answer_excerpt"),
        "budget_used": _opt_int(raw, "budget_used"),
        "budget_cap": _opt_int(raw, "budget_cap"),
        "freeze_before": _opt_str(raw, "freeze_before"),
        "freeze_after": _opt_str(raw, "freeze_after"),
        "validator": {
            "pit_as_of": _opt_str(vraw, "pit_as_of"),
            "expected_tools": _req_str_list(vraw, "expected_tools"),
            "requires_evidence": flag,
            "validators": _req_str_list(vraw, "validators"),
        },
    }
    telemetry = _opt_telemetry(raw)
    if telemetry is not None:
        fixture["telemetry"] = telemetry
    trace = _opt_trace(raw)
    if trace is not None:
        fixture["trace"] = trace
    return fixture


def load_fixture(scenario_name: str, fixtures_dir: Path | None = None) -> AgentFixture:
    """Load and validate one fixture; ValueError on malformed content."""
    dest_dir = fixtures_dir if fixtures_dir is not None else default_fixtures_dir()
    path = dest_dir / f"{scenario_name}.json"
    raw, vraw, flag = _fixture_raw(path)
    return _fixture_body(raw, vraw, flag)


def list_fixtures(fixtures_dir: Path | None = None) -> list[str]:
    """Sorted scenario names with a saved fixture."""
    dest_dir = fixtures_dir if fixtures_dir is not None else default_fixtures_dir()
    if not dest_dir.is_dir():
        return []
    return sorted(p.stem for p in dest_dir.glob("*.json") if p.is_file())


def _v_fixture_pit(fixture: AgentFixture) -> str | None:
    """PIT gate: any known_at after pit_as_of is a future crossing."""
    as_of = fixture["validator"]["pit_as_of"]
    if as_of is None:
        return None
    for known_at in fixture["known_ats"]:
        if known_at > as_of:
            return "future-crossing-as_of"
    return None


def _v_fixture_evidence(fixture: AgentFixture) -> str | None:
    """Evidence-required fixtures must cite at least one id."""
    if fixture["validator"]["requires_evidence"] and not fixture["evidence_ids"]:
        return "untraceable-dossier-claim"
    return None


def _v_fixture_budget(fixture: AgentFixture) -> str | None:
    """Over-budget fixtures fail the scheduler check."""
    used = fixture["budget_used"]
    cap = fixture["budget_cap"]
    if used is not None and cap is not None and used > cap:
        return "budget-violation"
    return None


def _v_fixture_capability(fixture: AgentFixture) -> str | None:
    """Unsupported families must not drive tools (capability policy)."""
    if fixture["family"] == "unsupported" and fixture["tool_calls"]:
        return "capability-policy-violation"
    return None


def _v_fixture_freeze(fixture: AgentFixture) -> str | None:
    """Freeze ids are immutable across the fixture."""
    before = fixture["freeze_before"]
    after = fixture["freeze_after"]
    if before is not None and after is not None and before != after:
        return "frozen-mutation"
    return None


def _v_fixture_timeout(fixture: AgentFixture) -> str | None:
    """Timeout fixtures must carry the failed-closure markers."""
    if fixture["scenario_name"] != "timeout-model-call-failed-resume":
        return None
    excerpt = fixture["answer_excerpt"]
    if any(marker not in excerpt for marker in _TIMEOUT_CLOSURE_MARKERS):
        return "timeout-without-failed-closure"
    return None


_FIXTURE_CHECKS = (
    _v_fixture_pit,
    _v_fixture_evidence,
    _v_fixture_budget,
    _v_fixture_capability,
    _v_fixture_freeze,
    _v_fixture_timeout,
    _v_fixture_msft_openai,
)


def run_deterministic_validators(fixture: AgentFixture) -> list[str]:
    """s35 validators over one fixture; returns violation codes (empty = pass)."""
    return [code for code in (check(fixture) for check in _FIXTURE_CHECKS) if code is not None]


def _trace_line(header: object, kinds: dict[str, int], count: int) -> str:
    trace_id = getattr(header, "trace_id", "?")
    wave_id = getattr(header, "wave_id", "?")
    model = getattr(header, "model", "?")
    status = getattr(header, "status", "?")
    breakdown = ", ".join(f"{kind}={n}" for kind, n in sorted(kinds.items()))
    return f"  trace {trace_id} wave={wave_id} model={model} status={status} events={count}" + (
        f" [{breakdown}]" if breakdown else ""
    )


def _trace_lines(session_id: str, data_root: Path | None) -> list[str]:
    from app.research.evals.traces import get_trace_events, list_traces

    headers = list_traces(session_id, data_root)
    if not headers:
        return [f"session {session_id}: no traces recorded yet"]
    lines = [f"session {session_id}: {len(headers)} trace(s)"]
    for header in headers:
        kinds: dict[str, int] = {}
        for event in get_trace_events(header.trace_id, data_root):
            kinds[event.event_type] = kinds.get(event.event_type, 0) + 1
        lines.append(_trace_line(header, kinds, sum(kinds.values())))
    return lines


def inspect_session(session_id: str, data_root: Path | None = None) -> str:
    """Human-readable trace summary for one session; never raises."""
    try:
        return "\n".join(_trace_lines(session_id, data_root))
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return f"session {session_id}: trace inspect unavailable ({exc})"
