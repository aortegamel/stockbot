"""Research kernel domain models: frozen dataclasses, stdlib only.

One authoritative ResearchSession; no private state copies. All datetimes
are UTC (naive inputs normalize to UTC). Missing timestamps stay None,
never invented. JSON helpers mirror app.thesis.models locally so the
kernel stands alone.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


def _json_scalars(value: object, where: str) -> JSONValue | None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: non-finite float not allowed, got {value!r}")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return None


def _json_dict(value: dict[object, object], where: str) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f"{where}: dict key must be a string, got {type(k).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        out[k] = validate_json_value(v, where)
    return out


def validate_json_value(value: object, where: str = "<dict>") -> JSONValue:
    """Recursively normalize an object into a JSONValue (deep copy)."""
    scalar = _json_scalars(value, where)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, (list, tuple)):
        return [validate_json_value(v, where) for v in value]
    if isinstance(value, dict):
        return _json_dict(value, where)
    raise ValueError(f"{where}: not a JSON value, got {type(value).__name__}")


def validate_json_mapping(value: object, where: str = "<dict>") -> dict[str, JSONValue]:
    """Validate untrusted payload as a JSON object."""
    validated = validate_json_value(value, where)
    if not isinstance(validated, dict):
        raise ValueError(f"{where}: must be a mapping, got {type(value).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return validated


def utcnow() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(UTC)


def query_with_today_utc(text: str, today: date | datetime | str | None = None) -> str:
    """Prefix a query/objective with `[Today UTC YYYY-MM-DD]` (idempotent)."""
    if not isinstance(text, str) or not text.strip():
        return text
    if "[Today UTC " in text:
        return text
    if today is None:
        day = utcnow().date().isoformat()
    elif isinstance(today, datetime):
        day = today.date().isoformat()
    elif isinstance(today, date):
        day = today.isoformat()
    else:
        day = today.strip() or utcnow().date().isoformat()
    return f"[Today UTC {day}] {text}"


def normalize_time(value: datetime) -> datetime:
    """Naive datetimes attach UTC; aware values convert to UTC (same instant, never relabeled)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def new_session_id() -> str:
    """Mint a Stockbot-owned research session id."""
    return f"rs:{uuid.uuid4()}"


def new_job_id() -> str:
    """Mint a Stockbot-owned research job id."""
    return f"job:{uuid.uuid4()}"


def new_event_id() -> str:
    """Mint a Stockbot-owned journal event id."""
    return f"jev:{uuid.uuid4()}"


class SessionStatus(StrEnum):
    """ResearchSession lifecycle states (forward flow + terminal states)."""

    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    FREEZING = "freezing"
    ANALYZING = "analyzing"
    TARGETED_RESEARCH = "targeted_research"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    """Worker roles; models decide relevance, infra decides budgets."""

    SOURCE_AGENT = "source_agent"
    SCOUT = "scout"
    STOCKBOT = "stockbot"
    BULLBOT = "bullbot"
    BEARBOT = "bearbot"
    SYNTHESIS = "synthesis"


class JobStatus(StrEnum):
    """Job execution states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class FailureCategory(StrEnum):
    """Closed failure vocabulary (25 values); free text lives in message."""

    TIMEOUT = "timeout"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    JOB_BUDGET_EXHAUSTED = "job_budget_exhausted"
    WAVE_BUDGET_EXHAUSTED = "wave_budget_exhausted"
    PARALLELISM_EXCEEDED = "parallelism_exceeded"
    DEPTH_EXCEEDED = "depth_exceeded"
    POLICY_DENIED = "policy_denied"
    TOOL_ERROR = "tool_error"
    MODEL_ERROR = "model_error"
    MODEL_OUTPUT_FAILURE = "model_output_failure"
    NO_EVIDENCE = "no_evidence"
    PIT_VIOLATION = "pit_violation"
    FREEZE_MISMATCH = "freeze_mismatch"
    COMMITTEE_DEADLOCK = "committee_deadlock"
    SYNTHESIS_FAILED = "synthesis_failed"
    CONTEXT_CAPACITY_EXCEEDED = "context_capacity_exceeded"
    RESPONSE_TOO_LARGE = "response_too_large"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    PROVIDER_ERROR = "provider_error"
    TOOL_OUTPUT_LIMIT = "tool_output_limit"
    STORAGE_ERROR = "storage_error"
    POLICY_REJECTION = "policy_rejection"
    DUPLICATE_RESEARCH_ACTION = "duplicate_research_action"
    RESEARCH_LOOP_DETECTED = "research_loop_detected"


# Control-plane bounds: single authority for source runtime and heartbeat staleness.
# Budgets are session/tool/token/cost totals only; no per-source numeric gate.
# 1800s headroom: the 61s evidence-denial dispatch (rs:7506fb7f) must never hit a job wall clock.
SOURCE_RUNTIME_BUDGET_S = 1800
HEARTBEAT_STALE_S = 120
# ponytail: single nested defaults dict; per-job overrides only via explicit
# create_job kwargs. Add sections when a new job type needs children/tools.
# None = no limit; only an explicit int (policy override) is ever a ceiling.
DEFAULT_BUDGETS: dict[str, JSONValue] = {
    "research": {
        "max_runtime": None,
        "max_total_jobs": None,
        "max_parallel": 6,
        "max_waves": None,
        "max_tool_calls": None,
    },
    "source": {"max_children": None, "max_tool": None},
    "scout": {"max_children": 0, "max_tool": None, "max_runtime": None},
    "committee": {"max_parallel": 3},
}


def default_policy() -> dict[str, JSONValue]:
    """Fresh copy of the full nested policy (research/source/scout/committee)."""
    out = validate_json_value(DEFAULT_BUDGETS, "<defaults>")
    assert isinstance(out, dict)
    return out


def default_budget() -> dict[str, JSONValue]:
    """Fresh copy of the session budget (research section + dispatch totals)."""
    out = validate_json_value(DEFAULT_BUDGETS["research"], "<defaults>")
    assert isinstance(out, dict)
    out["deadline_seconds"] = None  # unlimited unless policy configures an int
    out["total_tool_budget"] = None  # unlimited by default; explicit int only
    out["total_token_budget"] = None
    out["total_cost_budget"] = None
    return out


DEFAULT_POLICY: dict[str, JSONValue] = default_policy()
DEFAULT_BUDGET: dict[str, JSONValue] = default_budget()
SOURCE_POLICY_MODES = frozenset({"all", "allowlist"})
TEMPORAL_SCOPE_MODES = frozenset({"latest-available", "latest", "unbounded", "as_of", "range"})


def default_source_policy() -> dict[str, JSONValue]:
    """Fresh SEC-only allowlist (kernel default; mode all only when requested)."""
    return {"allowed": ["SEC"], "denied": [], "mode": "allowlist"}


def default_temporal_scope() -> dict[str, JSONValue]:
    """Fresh latest-available scope (cutoff set at resolve time, never invented)."""
    return {"as_of": None, "start": None, "end": None, "mode": "latest-available", "raw": None}


def validate_source_policy(value: object, where: str = "<dict>") -> dict[str, JSONValue]:
    """Validate a source_policy mapping ({allowed, denied, mode}); returns a copy."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: 'source_policy' must be a mapping, got {type(value).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    mode = value.get("mode", "all")
    if not isinstance(mode, str) or mode.strip().lower() not in SOURCE_POLICY_MODES:
        raise ValueError(
            f"{where}: 'source_policy.mode' must be one of {sorted(SOURCE_POLICY_MODES)}, got {value.get('mode')!r}"
        )
    allowed: list[JSONValue] = _json_str_list(_req_list_str(value, "allowed", where))
    denied: list[JSONValue] = _json_str_list(_req_list_str(value, "denied", where))
    return {
        "allowed": allowed,
        "denied": denied,
        "mode": mode.strip().lower(),
    }


def _source_container(policy: Mapping[str, object]) -> object:
    """research_sources container: nested key wins, else the bare mapping itself."""
    nested = policy.get("research_sources", None)
    return nested if nested is not None else (policy if "mode" in policy else None)


def _source_norm_mode(raw: Mapping[str, object], where: str) -> str:
    """Validate research_sources.mode; the fail-closed kernel vocabulary."""
    mode = raw.get("mode", "all")
    if not isinstance(mode, str) or mode.strip().lower() not in SOURCE_POLICY_MODES:
        raise ValueError(f"{where}: 'research_sources.mode' must be one of {sorted(SOURCE_POLICY_MODES)}, got {mode!r}")
    return mode.strip().lower()


def _source_cleaned_sources(raw: Mapping[str, object], norm: str, where: str) -> list[str]:
    """Validate + strip research_sources.sources (allowlist mode requires non-empty)."""
    sources = raw.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(s, str) or not s.strip() for s in sources):
        raise ValueError(f"{where}: 'research_sources.sources' must be a list of non-empty strings")
    cleaned = [s.strip() for s in sources]
    if norm == "allowlist" and not cleaned:
        raise ValueError(f"{where}: 'research_sources.sources' must be non-empty for allowlist mode")
    return cleaned


def resolve_source_policy(policy: object, where: str = "<session>") -> dict[str, JSONValue]:
    """Build source_policy from a full policy or a bare research_sources mapping.

    Accepts policy['research_sources'] ({mode: all|allowlist, sources}) or the
    research_sources mapping itself; absent key means the SEC-only default.
    """
    if policy is None:
        return default_source_policy()
    if not isinstance(policy, Mapping):
        raise ValueError(f"{where}: 'policy' must be a mapping, got {type(policy).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    raw = _source_container(policy)
    if raw is None:
        return default_source_policy()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: 'research_sources' must be a mapping, got {type(raw).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    norm = _source_norm_mode(raw, where)
    cleaned = _source_cleaned_sources(raw, norm, where)
    allowed_out: list[JSONValue] = list(cleaned)
    return {"allowed": allowed_out, "denied": [], "mode": norm}


def _policy_domain_set(checked: Mapping[str, object], key: str) -> set[str]:
    """Lowercased domain set for an allow/deny list (validated policy always holds lists)."""
    raw = checked.get(key, [])
    if not isinstance(raw, list):
        return set()
    return {s.strip().lower() for s in raw if isinstance(s, str)}


def source_domain_allowed(
    source_policy: Mapping[str, object] | None,
    source_domain: str | None,
    where: str = "<session>",
) -> bool:
    """True when a job source_domain may run under source_policy (None is SEC-default)."""
    if source_domain is None:
        return True
    if not isinstance(source_domain, str) or not source_domain.strip():
        raise ValueError(f"{where}: 'source_domain' must be a non-empty string or null, got {source_domain!r}")
    checked = validate_source_policy(source_policy if source_policy is not None else default_source_policy(), where)
    want = source_domain.strip().lower()
    if want in _policy_domain_set(checked, "denied"):
        return False
    if checked.get("mode") == "all":
        return True
    return want in _policy_domain_set(checked, "allowed")


_TEMPORAL_UNBOUNDED_RE = re.compile(
    r"\b(unbounded|all\s+history|entire\s+history|no\s+(time\s+)?cut-?off|without\s+time\s+limit)\b",
    re.IGNORECASE,
)
_TEMPORAL_EARNINGS_RE = re.compile(r"\bbefore\s+earnings\b", re.IGNORECASE)
_TEMPORAL_BETWEEN_RE = re.compile(r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_TEMPORAL_AS_OF_RE = re.compile(r"\bas\s+of\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_TEMPORAL_LAST_N_RE = re.compile(r"\blast\s+(\d+)\s+years?\b", re.IGNORECASE)
_TEMPORAL_LAST_YEAR_RE = re.compile(r"\blast\s+year\b", re.IGNORECASE)
_TEMPORAL_LAST_QUARTER_RE = re.compile(r"\blast\s+quarter\b", re.IGNORECASE)
_TEMPORAL_QUARTER_RE = re.compile(r"\bthis\s+quarter\b", re.IGNORECASE)
_TEMPORAL_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_TEMPORAL_LAST_WEEK_RE = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_TEMPORAL_LATEST_RE = re.compile(r"\blatest[\s-]*available\b", re.IGNORECASE)
_TEMPORAL_NOW_RE = re.compile(r"\b(today|right\s+now|as\s+of\s+now|most\s+recent|current|latest)\b", re.IGNORECASE)


def _temporal_day(text: str, where: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"{where}: invalid calendar date {text!r}") from None


def _shift_years(moment: datetime, years: int) -> datetime:
    try:
        return moment.replace(year=moment.year - years)
    except ValueError:  # Feb 29 -> Feb 28
        return moment.replace(year=moment.year - years, day=28)


def _temporal_text(temporal: str | None, query: str, where: str) -> tuple[str | None, str]:
    """Validated (raw, text) pair: explicit temporal wins, else the query string."""
    if temporal is not None and (not isinstance(temporal, str) or not temporal.strip()):
        raise ValueError(f"{where}: 'temporal' must be a non-empty string or null, got {temporal!r}")
    raw = temporal.strip() if isinstance(temporal, str) else None
    text = raw if raw is not None else (query if isinstance(query, str) else "")
    return raw, text


def _temporal_last_year_range(moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Range over the prior calendar year."""
    prev = moment.year - 1
    return {"as_of": f"{prev}-12-31", "start": f"{prev}-01-01", "end": f"{prev}-12-31", "mode": "range", "raw": raw}


def _temporal_quarter_range(moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Range from the current quarter start through now."""
    q0 = date(moment.year, 3 * ((moment.month - 1) // 3) + 1, 1).isoformat()
    return {"as_of": moment.isoformat(), "start": q0, "end": moment.isoformat(), "mode": "range", "raw": raw}


def _temporal_last_quarter_range(moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Range over the prior calendar quarter (UTC dates)."""
    q_start = date(moment.year, 3 * ((moment.month - 1) // 3) + 1, 1)
    prev_end = q_start - timedelta(days=1)
    prev_start = date(prev_end.year, 3 * ((prev_end.month - 1) // 3) + 1, 1)
    return {
        "as_of": prev_end.isoformat(),
        "start": prev_start.isoformat(),
        "end": prev_end.isoformat(),
        "mode": "range",
        "raw": raw,
    }


_NYC = ZoneInfo("America/New_York")


def _nyc_monday(moment: datetime) -> date:
    """Monday of the NYC calendar week containing moment (exchanges run on NYC dates)."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    nyc_day = aware.astimezone(_NYC).date()
    return nyc_day - timedelta(days=nyc_day.weekday())


def _temporal_this_week_range(moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Range from Monday 00:00 NYC through now."""
    monday = _nyc_monday(moment)
    return {
        "as_of": moment.isoformat(),
        "start": monday.isoformat(),
        "end": moment.isoformat(),
        "mode": "range",
        "raw": raw,
    }


def _temporal_last_week_range(moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Range over the prior calendar week (Mon-Sun, NYC dates)."""
    monday_this = _nyc_monday(moment)
    start = monday_this - timedelta(days=7)
    end = monday_this - timedelta(days=1)
    return {"as_of": end.isoformat(), "start": start.isoformat(), "end": end.isoformat(), "mode": "range", "raw": raw}


def _temporal_moment_asof(moment: datetime, raw: str | None, mode: str) -> dict[str, JSONValue]:
    """Cutoff pinned at now for pre-earnings/today/latest wording (no range)."""
    return {"as_of": moment.isoformat(), "start": None, "end": None, "mode": mode, "raw": raw}


_TEMPORAL_SIMPLE_TABLE = (
    ("last_n", _TEMPORAL_LAST_N_RE),
    ("last_year", _TEMPORAL_LAST_YEAR_RE),
    ("last_quarter", _TEMPORAL_LAST_QUARTER_RE),
    ("quarter", _TEMPORAL_QUARTER_RE),
    ("this_week", _TEMPORAL_THIS_WEEK_RE),
    ("last_week", _TEMPORAL_LAST_WEEK_RE),
    ("earnings", _TEMPORAL_EARNINGS_RE),
    ("latest", _TEMPORAL_LATEST_RE),
    ("now", _TEMPORAL_NOW_RE),
)


def _temporal_simple_hit(name: str, hit: re.Match[str], moment: datetime, raw: str | None) -> dict[str, JSONValue]:
    """Build the scope for a simple (non between/as-of) pattern hit by table name."""
    if name == "last_n":
        start_dt = _shift_years(moment, max(int(hit.group(1)), 1))
        return {
            "as_of": moment.isoformat(),
            "start": start_dt.isoformat(),
            "end": moment.isoformat(),
            "mode": "range",
            "raw": raw,
        }
    if name == "last_year":
        return _temporal_last_year_range(moment, raw)
    if name == "last_quarter":
        return _temporal_last_quarter_range(moment, raw)
    if name == "quarter":
        return _temporal_quarter_range(moment, raw)
    if name == "this_week":
        return _temporal_this_week_range(moment, raw)
    if name == "last_week":
        return _temporal_last_week_range(moment, raw)
    if name == "latest":
        return _temporal_moment_asof(moment, raw, "latest-available")
    return _temporal_moment_asof(moment, raw, "as_of")


def _temporal_match(text: str, moment: datetime, raw: str | None, where: str) -> dict[str, JSONValue] | None:
    """First temporal-pattern hit in kernel precedence order (None when no wording matches)."""
    if _TEMPORAL_UNBOUNDED_RE.search(text):
        return {"as_of": None, "start": None, "end": None, "mode": "unbounded", "raw": raw}
    between = _TEMPORAL_BETWEEN_RE.search(text)
    if between is not None:
        start = _temporal_day(between.group(1), where)
        end = _temporal_day(between.group(2), where)
        if start > end:
            raise ValueError(f"{where}: temporal range start {start!r} is after end {end!r}")
        return {"as_of": end, "start": start, "end": end, "mode": "range", "raw": raw}
    asof_hit = _TEMPORAL_AS_OF_RE.search(text)
    if asof_hit is not None:
        day = _temporal_day(asof_hit.group(1), where)
        return {"as_of": day, "start": None, "end": None, "mode": "as_of", "raw": raw}
    for name, regex in _TEMPORAL_SIMPLE_TABLE:
        hit = regex.search(text)
        if hit is not None:
            return _temporal_simple_hit(name, hit, moment, raw)
    return None


def resolve_temporal_scope(
    *,
    as_of: datetime | str | None = None,
    temporal: str | None = None,
    query: str = "",
    now: datetime | None = None,
    where: str = "<session>",
) -> dict[str, JSONValue]:
    """Resolve persisted temporal_scope ({as_of, start, end, mode, raw}).

    Explicit unbounded wording is the only path to mode 'unbounded'; no time
    info means latest-available with the cutoff set (never invented, never None).
    Raw wording is always kept.
    """
    moment = normalize_time(now) if isinstance(now, datetime) else utcnow()
    raw, text = _temporal_text(temporal, query, where)
    cut = _coerce_time(as_of, "as_of", where)
    if text:
        hit_scope = _temporal_match(text, moment, raw, where)
        if hit_scope is not None:
            return hit_scope
    if cut is not None:
        return {"as_of": cut.isoformat(), "start": None, "end": None, "mode": "as_of", "raw": raw}
    return {"as_of": moment.isoformat(), "start": None, "end": None, "mode": "latest-available", "raw": raw}


def validate_temporal_scope(value: object, where: str = "<dict>") -> dict[str, JSONValue]:
    """Validate a temporal_scope mapping ({as_of, start, end, mode, raw}); datetimes coerce to ISO strings."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: 'temporal_scope' must be a mapping, got {type(value).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    mode = value.get("mode", "latest-available")
    if not isinstance(mode, str) or mode.strip().lower() not in {*TEMPORAL_SCOPE_MODES, "latest"}:
        raise ValueError(
            f"{where}: 'temporal_scope.mode' must be one of {sorted({*TEMPORAL_SCOPE_MODES, 'latest'})}, got {value.get('mode')!r}"
        )
    norm = "latest-available" if mode.strip().lower() == "latest" else mode.strip().lower()
    out: dict[str, JSONValue] = {}
    for key in ("as_of", "start", "end"):
        parsed = _coerce_time(value.get(key), key, f"{where}: 'temporal_scope'")
        out[key] = parsed.isoformat() if parsed is not None else None
    out["mode"] = norm
    raw = value.get("raw", None)
    if raw is not None and not isinstance(raw, str):
        raise ValueError(f"{where}: 'temporal_scope.raw' must be a string or null, got {type(raw).__name__}")
    out["raw"] = raw
    return out


def _baseline_pit(filing: object) -> tuple[str | None, str | None]:
    """Local PIT read (known_at > accepted_at > filed_at); mirrors app.sec.models.pit_of."""

    def _get(name: str) -> object:
        if isinstance(filing, Mapping):
            return filing.get(name)
        return getattr(filing, name, None)

    for basis in ("known_at", "accepted_at", "filed_at"):
        try:
            value = _get(basis)
        except Exception:  # noqa: BLE001 - duck-typed PIT read coerces faulty attrs to None, never raises
            value = None
        if value is None:
            continue
        text = value.isoformat() if isinstance(value, datetime) else str(value)
        if text.strip():
            return text, basis
    return None, None


def _baseline_day(pair: tuple[str, object]) -> str:
    """Sort key for (day, filing) baseline pairs."""
    return pair[0]


_BASELINE_ANNUAL = frozenset({"10-K", "10-K/A"})
_BASELINE_QUARTERLY = frozenset({"10-Q", "10-Q/A"})
_BASELINE_CURRENT = frozenset({"8-K", "8-K/A"})


def _baseline_form(filing: object) -> str:
    """Filing form text from a Filing dataclass or mapping (empty when absent)."""
    if isinstance(filing, Mapping):
        raw = filing.get("form")
    else:
        raw = getattr(filing, "form", None)
    return raw.strip().upper() if isinstance(raw, str) else ""


def _baseline_accession(filing: object) -> str | None:
    """Accession id from a Filing dataclass or mapping (None when absent)."""
    if isinstance(filing, Mapping):
        raw = filing.get("accession_no", filing.get("accession"))
    else:
        raw = getattr(filing, "accession_no", getattr(filing, "accession", None))
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _baseline_superseded_by(filing: object) -> str | None:
    """Replacement accession when this filing declares itself superseded (None otherwise)."""
    if isinstance(filing, Mapping):
        raw = filing.get("superseded_by")
    else:
        raw = getattr(filing, "superseded_by", None)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _baseline_skipped(best: object, day: str, best_day: str) -> bool:
    """True when the candidate loses to the current pick (older day, or tie kept first)."""
    return best is not None and day <= best_day


def _baseline_eligible_day(filing: object, bound_day: str | None) -> str | None:
    """PIT day for one filing (None when missing PIT or past the bound)."""
    value, _basis = _baseline_pit(filing)
    if value is None:
        return None
    day = value[:10]
    if bound_day is not None and day > bound_day:
        return None
    return day


def _baseline_eligible(filings: list[object], bound_day: str | None) -> list[tuple[str, object]]:
    """Filings with PIT proof on/before the bound, as (day, filing) pairs."""
    out: list[tuple[str, object]] = []
    for filing in filings:
        day = _baseline_eligible_day(filing, bound_day)
        if day is None:
            continue
        out.append((day, filing))
    return out


def _baseline_accession_index(eligible: list[tuple[str, object]]) -> dict[str, object]:
    """Eligible filings keyed by accession (latest write wins on duplicates)."""
    index: dict[str, object] = {}
    for _, filing in eligible:
        acc = _baseline_accession(filing)
        if acc is None:
            continue
        index[acc] = filing
    return index


def _baseline_replacement_eligible(filing: object, by_accession: Mapping[str, object]) -> bool:
    """True when the filing names a PIT-eligible replacement (never wins)."""
    nxt = _baseline_superseded_by(filing)
    return nxt is not None and nxt in by_accession


def _baseline_pick_latest(
    eligible: list[tuple[str, object]],
    forms: frozenset[str],
    by_accession: Mapping[str, object],
) -> object:
    """Latest non-superseded filing among forms (ties keep the first seen)."""
    best: object = None
    best_day = ""
    for day, filing in eligible:
        if _baseline_form(filing) not in forms:
            continue
        if _baseline_replacement_eligible(filing, by_accession):
            continue
        if _baseline_skipped(best, day, best_day):
            continue
        best, best_day = filing, day
    return best


def select_latest_baseline(
    filings: list[object],
    as_of: datetime | str | None = None,
) -> dict[str, object]:
    """Pure PIT baseline pick: latest 10-K, latest 10-Q, and PIT-eligible 8-Ks.

    A filing is eligible only when its PIT timestamp (known_at, else
    accepted_at, else filed_at) is on/before as_of; None as_of is unbounded.
    A 10-K/10-Q that names a PIT-eligible replacement via ``superseded_by``
    never wins: the current-position read must pin the replacement, while
    historical as_of/range questions may still use the older filing directly.
    Every eligible 8-K/8-K/A counts as material, uncapped; newest first.
    Consumed by the Context slice; pair with list_sec_filings(..., as_of=...)
    which already PIT-filters at discovery.
    """
    bound = _coerce_time(as_of, "as_of", "<baseline>")
    bound_day = bound.date().isoformat() if bound is not None else None
    eligible = _baseline_eligible(filings, bound_day)
    by_accession = _baseline_accession_index(eligible)
    annual = _baseline_pick_latest(eligible, _BASELINE_ANNUAL, by_accession)
    quarterly = _baseline_pick_latest(eligible, _BASELINE_QUARTERLY, by_accession)
    currents = [(day, filing) for day, filing in eligible if _baseline_form(filing) in _BASELINE_CURRENT]
    currents.sort(key=_baseline_day, reverse=True)
    return {
        "annual_10k": annual,
        "quarterly_10q": quarterly,
        "material_8k": [filing for _, filing in currents],
        "as_of": bound.isoformat() if bound is not None else None,
    }


def _cited_accessions(cited: object) -> list[str]:
    """Normalize cited accession(s) to stripped strings (non-strings dropped)."""
    raw: list[object] = [cited] if isinstance(cited, str) else (list(cited) if isinstance(cited, (list, tuple)) else [])
    return [c.strip() for c in raw if isinstance(c, str) and c.strip()]


def superseded_current_violation(
    filings: list[object],
    cited: object,
    *,
    as_of: datetime | str | None = None,
) -> str | None:
    """Current-position staleness check: latest 10-K pinned, superseded-only current claim rejected.

    Returns an error string when ``cited`` rests solely on a superseded annual
    while the PIT-eligible replacement is available; None when the citation
    set is fresh (or the question is historical: as_of/range callers may use
    older filings and should skip this gate). ``cited`` is accession(s).
    """
    base = select_latest_baseline(filings, as_of=as_of)
    latest = base.get("annual_10k")
    latest_acc = _baseline_accession(latest) if latest is not None else None
    if latest_acc is None:
        return None
    cited_accs = _cited_accessions(cited)
    if not cited_accs or latest_acc in cited_accs:
        return None
    return f"current-position claim cites superseded annual {cited_accs[0]!r} while latest 10-K {latest_acc!r} is PIT-eligible"


STATUS_VALUES = frozenset(e.value for e in SessionStatus)
JOB_TYPE_VALUES = frozenset(e.value for e in JobType)
JOB_STATUS_VALUES = frozenset(e.value for e in JobStatus)
FAILURE_CATEGORY_VALUES = frozenset(e.value for e in FailureCategory)


def _json_str_list(values: list[str]) -> list[JSONValue]:
    """Copy a string list into a JSON list (works around list invariance)."""
    out: list[JSONValue] = []
    out.extend(values)
    return out


def _req_str(d: Mapping[str, object], key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return v


def _opt_str(v: object, key: str, where: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{where}: '{key}' must be a string or null, got {type(v).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return v


def _opt_int(v: object, key: str, where: str) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{where}: '{key}' must be an int or null, got {v!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return v


def _coerce_enum(allowed: frozenset[str], v: object, key: str, where: str) -> str:
    if isinstance(v, StrEnum):
        v = v.value
    if isinstance(v, str) and v in allowed:
        return v
    raise ValueError(f"{where}: '{key}' must be one of {sorted(allowed)}, got {v!r}")


def _parse_time_str(v: str, key: str, where: str) -> datetime:
    try:
        return normalize_time(datetime.fromisoformat(v.strip()))
    except ValueError:
        raise ValueError(f"{where}: '{key}' must be ISO-8601, got {v!r}") from None


def _coerce_time(v: object, key: str, where: str) -> datetime | None:
    """Parse an ISO-8601 string or datetime (naive normalizes to UTC); None stays None."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return normalize_time(v)
    if isinstance(v, str) and v.strip():
        return _parse_time_str(v, key, where)
    raise ValueError(f"{where}: '{key}' must be an ISO-8601 string, datetime, or null")


def _req_time(d: Mapping[str, object], key: str, where: str) -> datetime:
    out = _coerce_time(d.get(key), key, where)
    if out is None:
        raise ValueError(f"{where}: '{key}' must be a timestamp, got null")
    return out


def _req_list_str(d: Mapping[str, object], key: str, where: str) -> list[str]:
    v = d.get(key, [])
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise ValueError(f"{where}: '{key}' must be a list of strings")
    return list(v)


def pit_violated(as_of: datetime | str | None, known_at: datetime | str | None) -> bool:
    """True when known_at > as_of. None on either side never violates."""
    if isinstance(as_of, str) and as_of.strip().lower() in NO_CUTOFF_AS_OF:
        return False
    start = _coerce_time(as_of, "as_of", "<pit>")
    known = _coerce_time(known_at, "known_at", "<pit>")
    if start is None or known is None:
        return False
    return known > start


# The sentinel an unbounded session carries: no PIT cutoff exists, so no known_at
# can be after it and no known_at is unverified. One definition, used by every gate.
NO_CUTOFF_AS_OF: frozenset[str] = frozenset({"unbounded"})


def _as_of_bounded(as_of: datetime | str | None) -> bool:
    if as_of is None:
        return False
    if isinstance(as_of, str):
        text = as_of.strip().lower()
        return bool(text) and text not in NO_CUTOFF_AS_OF
    return isinstance(as_of, datetime)


def pit_unverified(as_of: datetime | str | None, known_at: datetime | str | None) -> bool:
    """True when historical as_of requires PIT proof but known_at is missing."""
    if known_at is not None:
        return False
    return _as_of_bounded(as_of)


@dataclass(frozen=True)
class Failure:
    """Categorized failure; category is closed vocabulary, detail is free text."""

    category: str
    message: str

    def validate(self, where: str = "<failure>") -> None:
        """Raise ValueError unless category is known and message non-empty."""
        _coerce_enum(FAILURE_CATEGORY_VALUES, self.category, "category", where)
        if not self.message:
            raise ValueError(f"{where}: 'message' must be a non-empty string")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize to plain JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> Failure:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: failure must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: failure"
        out = cls(
            category=_coerce_enum(FAILURE_CATEGORY_VALUES, d.get("category"), "category", where),
            message=_req_str(d, "message", where),
        )
        out.validate(where)
        return out


@dataclass(frozen=True)
class ResearchSession:
    """Authoritative session record; the only state holder, never copied privately."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    query: str
    objective: str
    as_of: datetime | None = None
    status: str = SessionStatus.CREATED.value
    current_wave: int = 0
    policy: dict[str, JSONValue] = field(default_factory=default_policy)
    budget: dict[str, JSONValue] = field(default_factory=default_budget)
    source_policy: dict[str, JSONValue] = field(default_factory=default_source_policy)
    temporal_scope: dict[str, JSONValue] = field(default_factory=default_temporal_scope)
    job_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    freeze_ids: list[str] = field(default_factory=list)
    dossier_ids: list[str] = field(default_factory=list)
    committee_runs: list[JSONValue] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    targeted_question: str | None = None
    targeted_domain: str | None = None
    final_result: dict[str, JSONValue] | None = None
    failure: Failure | None = None

    def _validate_ids(self, where: str) -> None:
        if not self.session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        if not self.query:
            raise ValueError(f"{where}: 'query' must be a non-empty string")
        if not self.objective:
            raise ValueError(f"{where}: 'objective' must be a non-empty string")

    def _validate_wave(self, where: str) -> None:
        _coerce_enum(STATUS_VALUES, self.status, "status", where)
        if isinstance(self.current_wave, bool) or not isinstance(self.current_wave, int):
            raise ValueError(f"{where}: 'current_wave' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        if self.current_wave < 0:
            raise ValueError(f"{where}: 'current_wave' must be >= 0, got {self.current_wave}")

    def _validate_targeting(self, where: str) -> None:
        if self.targeted_question is not None and not isinstance(self.targeted_question, str):
            raise ValueError(f"{where}: 'targeted_question' must be a string or null")
        if self.targeted_domain is not None and not isinstance(self.targeted_domain, str):
            raise ValueError(f"{where}: 'targeted_domain' must be a string or null")
        if self.failure is not None:
            self.failure.validate(f"{where}: failure")

    def validate(self, where: str = "<session>") -> None:
        """Raise ValueError on any contract violation."""
        self._validate_ids(where)
        self._validate_wave(where)
        self._validate_targeting(where)
        validate_source_policy(self.source_policy, f"{where}: 'source_policy'")
        validate_temporal_scope(self.temporal_scope, f"{where}: 'temporal_scope'")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize (datetimes as ISO-8601, failure nested as dict or None)."""
        d: dict[str, JSONValue] = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "query": self.query,
            "objective": self.objective,
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
            "status": self.status,
            "current_wave": self.current_wave,
            "policy": validate_json_mapping(self.policy, "<session>: 'policy'"),
            "budget": validate_json_mapping(self.budget, "<session>: 'budget'"),
            "source_policy": validate_json_mapping(self.source_policy, "<session>: 'source_policy'"),
            "temporal_scope": validate_json_mapping(self.temporal_scope, "<session>: 'temporal_scope'"),
            "job_ids": _json_str_list(self.job_ids),
            "evidence_ids": _json_str_list(self.evidence_ids),
            "freeze_ids": _json_str_list(self.freeze_ids),
            "dossier_ids": _json_str_list(self.dossier_ids),
            "committee_runs": [validate_json_value(x, "<session>: 'committee_runs'") for x in self.committee_runs],
            "unresolved_questions": _json_str_list(self.unresolved_questions),
            "targeted_question": self.targeted_question,
            "targeted_domain": self.targeted_domain,
            "final_result": validate_json_mapping(self.final_result, "<session>: 'final_result'")
            if self.final_result is not None
            else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }
        return d

    @classmethod
    def _parse_failure(cls, d: Mapping[str, object], where: str) -> Failure | None:
        raw_failure = d.get("failure")
        if raw_failure is None:
            return None
        if not isinstance(raw_failure, dict):
            raise ValueError(f"{where}: 'failure' must be a mapping or null")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return Failure.from_dict(raw_failure, where)

    @classmethod
    def _parse_result(cls, d: Mapping[str, object], where: str) -> dict[str, JSONValue] | None:
        raw_result = d.get("result", d.get("final_result"))
        if raw_result is None:
            return None
        return validate_json_mapping(raw_result, f"{where}: 'final_result'")

    @classmethod
    def _parse_wave(cls, d: Mapping[str, object], where: str) -> int:
        wave = d.get("current_wave", 0)
        if isinstance(wave, bool) or not isinstance(wave, int):
            raise ValueError(f"{where}: 'current_wave' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return wave

    @classmethod
    def _parse_runs(cls, d: Mapping[str, object], where: str) -> list[JSONValue]:
        raw_runs = d.get("committee_runs", [])
        if not isinstance(raw_runs, list):
            raise ValueError(f"{where}: 'committee_runs' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return [validate_json_value(x, f"{where}: 'committee_runs'") for x in raw_runs]

    @classmethod
    def _parse_source_policy(cls, d: Mapping[str, object], where: str) -> dict[str, JSONValue]:
        raw = d.get("source_policy", None)
        if raw is None:
            return default_source_policy()
        return validate_source_policy(raw, f"{where}: 'source_policy'")

    @classmethod
    def _parse_temporal_scope(cls, d: Mapping[str, object], where: str) -> dict[str, JSONValue]:
        raw = d.get("temporal_scope", None)
        if raw is None:
            return default_temporal_scope()
        return validate_temporal_scope(raw, f"{where}: 'temporal_scope'")

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> ResearchSession:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: session must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: session {d.get('session_id', '?')}"
        out = cls(
            session_id=_req_str(d, "session_id", where),
            created_at=_req_time(d, "created_at", where),
            updated_at=_req_time(d, "updated_at", where),
            query=_req_str(d, "query", where),
            objective=_req_str(d, "objective", where),
            as_of=_coerce_time(d.get("as_of"), "as_of", where),
            status=_coerce_enum(STATUS_VALUES, d.get("status", SessionStatus.CREATED.value), "status", where),
            current_wave=cls._parse_wave(d, where),
            policy=validate_json_mapping(d.get("policy", {}), f"{where}: 'policy'"),
            budget=validate_json_mapping(d.get("budget", {}), f"{where}: 'budget'"),
            source_policy=cls._parse_source_policy(d, where),
            temporal_scope=cls._parse_temporal_scope(d, where),
            job_ids=_req_list_str(d, "job_ids", where),
            evidence_ids=_req_list_str(d, "evidence_ids", where),
            freeze_ids=_req_list_str(d, "freeze_ids", where),
            dossier_ids=_req_list_str(d, "dossier_ids", where),
            committee_runs=cls._parse_runs(d, where),
            unresolved_questions=_req_list_str(d, "unresolved_questions", where),
            targeted_question=_opt_str(d.get("targeted_question"), "targeted_question", where),
            targeted_domain=_opt_str(d.get("targeted_domain"), "targeted_domain", where),
            final_result=cls._parse_result(d, where),
            failure=cls._parse_failure(d, where),
        )
        out.validate(where)
        return out


@dataclass(frozen=True)
class Job:
    """One unit of delegated work; budgets enforced at creation from policy."""

    job_id: str
    session_id: str
    wave_id: int
    parent_job_id: str | None
    job_type: str
    owner: str
    source_domain: str | None = None
    status: str = JobStatus.QUEUED.value
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    deadline: datetime | None = None
    last_heartbeat_at: datetime | None = None
    model: str | None = None
    token_budget: int | None = None
    tool_budget: int | None = None
    child_budget: int = 0
    result: dict[str, JSONValue] | None = None
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)
    failure: Failure | None = None

    def _validate_ids(self, where: str) -> None:
        if not self.job_id:
            raise ValueError(f"{where}: 'job_id' must be a non-empty string")
        if not self.session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        if isinstance(self.wave_id, bool) or not isinstance(self.wave_id, int):
            raise ValueError(f"{where}: 'wave_id' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        if self.wave_id < 1:
            raise ValueError(f"{where}: 'wave_id' must be >= 1, got {self.wave_id}")

    def _validate_budgets(self, where: str) -> None:
        if not self.owner:
            raise ValueError(f"{where}: 'owner' must be a non-empty string")
        if isinstance(self.child_budget, bool) or not isinstance(self.child_budget, int):
            raise ValueError(f"{where}: 'child_budget' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        if self.child_budget < 0:
            raise ValueError(f"{where}: 'child_budget' must be >= 0, got {self.child_budget}")
        if self.failure is not None:
            self.failure.validate(f"{where}: failure")

    def validate(self, where: str = "<job>") -> None:
        """Raise ValueError on any contract violation."""
        self._validate_ids(where)
        _coerce_enum(JOB_TYPE_VALUES, self.job_type, "job_type", where)
        _coerce_enum(JOB_STATUS_VALUES, self.status, "status", where)
        self._validate_budgets(where)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize (datetimes as ISO-8601, failure/result nested or None)."""
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "wave_id": self.wave_id,
            "parent_job_id": self.parent_job_id,
            "job_type": self.job_type,
            "owner": self.owner,
            "source_domain": self.source_domain,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at is not None else None,
            "deadline": self.deadline.isoformat() if self.deadline is not None else None,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at is not None else None,
            "model": self.model,
            "token_budget": self.token_budget,
            "tool_budget": self.tool_budget,
            "child_budget": self.child_budget,
            "result": validate_json_mapping(self.result, "<job>: 'result'") if self.result is not None else None,
            "diagnostics": validate_json_mapping(self.diagnostics, "<job>: 'diagnostics'"),
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    @classmethod
    def _parse_failure(cls, d: Mapping[str, object], where: str) -> Failure | None:
        raw_failure = d.get("failure")
        if raw_failure is None:
            return None
        if not isinstance(raw_failure, dict):
            raise ValueError(f"{where}: 'failure' must be a mapping or null")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return Failure.from_dict(raw_failure, where)

    @classmethod
    def _parse_result(cls, d: Mapping[str, object], where: str) -> dict[str, JSONValue] | None:
        raw_result = d.get("result")
        if raw_result is None:
            return None
        return validate_json_mapping(raw_result, f"{where}: 'result'")

    @classmethod
    def _parse_children(cls, d: Mapping[str, object], where: str) -> int:
        children = d.get("child_budget", 0)
        if isinstance(children, bool) or not isinstance(children, int):
            raise ValueError(f"{where}: 'child_budget' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return children

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> Job:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: job must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: job {d.get('job_id', '?')}"
        out = cls(
            job_id=_req_str(d, "job_id", where),
            session_id=_req_str(d, "session_id", where),
            wave_id=_req_time_wave(d, where),
            parent_job_id=_opt_str(d.get("parent_job_id"), "parent_job_id", where),
            job_type=_coerce_enum(JOB_TYPE_VALUES, d.get("job_type"), "job_type", where),
            owner=_req_str(d, "owner", where),
            source_domain=_opt_str(d.get("source_domain"), "source_domain", where),
            status=_coerce_enum(JOB_STATUS_VALUES, d.get("status", JobStatus.QUEUED.value), "status", where),
            created_at=_req_time(d, "created_at", where),
            started_at=_coerce_time(d.get("started_at"), "started_at", where),
            completed_at=_coerce_time(d.get("completed_at"), "completed_at", where),
            deadline=_coerce_time(d.get("deadline"), "deadline", where),
            last_heartbeat_at=_coerce_time(d.get("last_heartbeat_at"), "last_heartbeat_at", where),
            model=_opt_str(d.get("model"), "model", where),
            token_budget=_opt_int(d.get("token_budget"), "token_budget", where),
            tool_budget=_opt_int(d.get("tool_budget"), "tool_budget", where),
            child_budget=cls._parse_children(d, where),
            result=cls._parse_result(d, where),
            diagnostics=validate_json_mapping(d.get("diagnostics", {}), f"{where}: 'diagnostics'"),
            failure=cls._parse_failure(d, where),
        )
        out.validate(where)
        return out


def _req_time_wave(d: Mapping[str, object], where: str) -> int:
    wave = d.get("wave_id")
    if isinstance(wave, bool) or not isinstance(wave, int):
        raise ValueError(f"{where}: 'wave_id' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return wave


@dataclass(frozen=True)
class JournalEvent:
    """One append-only journal record; sequence is per-session, 1-based."""

    event_id: str
    session_id: str
    sequence: int
    event_type: str
    timestamp: datetime
    actor_type: str
    actor_id: str
    payload: dict[str, JSONValue] = field(default_factory=dict)
    previous_state: str | None = None
    new_state: str | None = None

    def _validate_ids_seq(self, where: str) -> None:
        if not self.event_id:
            raise ValueError(f"{where}: 'event_id' must be a non-empty string")
        if not self.session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError(f"{where}: 'sequence' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        if self.sequence < 1:
            raise ValueError(f"{where}: 'sequence' must be >= 1, got {self.sequence}")

    def validate(self, where: str = "<journal>") -> None:
        """Raise ValueError on any contract violation."""
        self._validate_ids_seq(where)
        if not self.event_type:
            raise ValueError(f"{where}: 'event_type' must be a non-empty string")
        if not self.actor_type:
            raise ValueError(f"{where}: 'actor_type' must be a non-empty string")
        if not self.actor_id:
            raise ValueError(f"{where}: 'actor_id' must be a non-empty string")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize to plain JSON-compatible dict."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "payload": validate_json_mapping(self.payload, "<journal>: 'payload'"),
            "previous_state": self.previous_state,
            "new_state": self.new_state,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> JournalEvent:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: journal event must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: event {d.get('event_id', '?')}"
        seq = d.get("sequence")
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError(f"{where}: 'sequence' must be an int")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        out = cls(
            event_id=_req_str(d, "event_id", where),
            session_id=_req_str(d, "session_id", where),
            sequence=seq,
            event_type=_req_str(d, "event_type", where),
            timestamp=_req_time(d, "timestamp", where),
            actor_type=_req_str(d, "actor_type", where),
            actor_id=_req_str(d, "actor_id", where),
            payload=validate_json_mapping(d.get("payload", {}), f"{where}: 'payload'"),
            previous_state=_opt_str(d.get("previous_state"), "previous_state", where),
            new_state=_opt_str(d.get("new_state"), "new_state", where),
        )
        out.validate(where)
        return out


def new_node_id() -> str:
    """Mint a Stockbot-owned research node id (what needs knowing)."""
    return f"rn:{uuid.uuid4()}"


def new_decision_id() -> str:
    """Mint a Stockbot-owned decision id."""
    return f"dec:{uuid.uuid4()}"


class ResearchNodeStatus(StrEnum):
    """ResearchNode lifecycle: what needs knowing (node) vs execution attempt (job)."""

    PROPOSED = "proposed"
    GATHERING = "gathering"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    RESOLVED = "resolved"
    BLOCKED = "blocked"
    REJECTED = "rejected"


NODE_STATUS_VALUES = frozenset(e.value for e in ResearchNodeStatus)
TOOL_DECISION_ACTIONS = frozenset({"invoke", "reason", "resolved"})


def _opt_tuple_str(v: object, key: str, where: str) -> tuple[str, ...]:
    if v is None:
        return ()
    if not isinstance(v, (list, tuple)) or any(not isinstance(x, str) for x in v):
        raise ValueError(f"{where}: '{key}' must be a list of strings or null")
    return tuple(v)


def _tool_names_from(d: Mapping[str, object], where: str) -> tuple[str, ...]:
    """tool_names list; single-tool compat: bare tool_name implies a one-tool set on invoke."""
    names = _opt_tuple_str(d.get("tool_names", []), "tool_names", where)
    if not names and d.get("action") == "invoke":
        single = d.get("tool_name")
        if isinstance(single, str) and single:
            return (single,)
    return names


def _opt_confidence(v: object, key: str, where: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: '{key}' must be a number or null, got {v!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    out = float(v)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{where}: '{key}' must be in [0, 1], got {v!r}")
    return out


@dataclass(frozen=True)
class ResearchNode:
    """One thing that needs knowing; a Job is an execution attempt against it."""

    node_id: str
    session_id: str
    question: str
    why_it_matters: str
    depends_on: tuple[str, ...] = ()
    status: str = ResearchNodeStatus.PROPOSED.value
    evidence_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    def validate(self, where: str = "<node>") -> None:
        """Raise ValueError on any contract violation."""
        if not self.node_id:
            raise ValueError(f"{where}: 'node_id' must be a non-empty string")
        if not self.session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        if not self.question:
            raise ValueError(f"{where}: 'question' must be a non-empty string")
        if not self.why_it_matters:
            raise ValueError(f"{where}: 'why_it_matters' must be a non-empty string")
        _coerce_enum(NODE_STATUS_VALUES, self.status, "status", where)

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize (tuples as lists)."""
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "question": self.question,
            "why_it_matters": self.why_it_matters,
            "depends_on": _json_str_list(list(self.depends_on)),
            "status": self.status,
            "evidence_ids": _json_str_list(list(self.evidence_ids)),
            "missing_evidence": _json_str_list(list(self.missing_evidence)),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> ResearchNode:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: node must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: node {d.get('node_id', '?')}"
        out = cls(
            node_id=_req_str(d, "node_id", where),
            session_id=_req_str(d, "session_id", where),
            question=_req_str(d, "question", where),
            why_it_matters=_req_str(d, "why_it_matters", where),
            depends_on=_opt_tuple_str(d.get("depends_on", []), "depends_on", where),
            status=_coerce_enum(
                NODE_STATUS_VALUES, d.get("status", ResearchNodeStatus.PROPOSED.value), "status", where
            ),
            evidence_ids=_opt_tuple_str(d.get("evidence_ids", []), "evidence_ids", where),
            missing_evidence=_opt_tuple_str(d.get("missing_evidence", []), "missing_evidence", where),
        )
        out.validate(where)
        return out


@dataclass(frozen=True)
class DecisionRecord:
    """Persisted JEV disposition: full candidate registry + probabilities + selected."""

    decision_id: str
    session_id: str
    node_id: str | None
    job_id: str | None
    decision_type: str
    candidates: dict[str, JSONValue]
    probabilities: dict[str, JSONValue]
    selected: JSONValue
    confidence: float | None = None
    created_at: datetime = field(default_factory=utcnow)

    def validate(self, where: str = "<decision>") -> None:
        """Raise ValueError on any contract violation."""
        if not self.decision_id:
            raise ValueError(f"{where}: 'decision_id' must be a non-empty string")
        if not self.session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        if not self.decision_type:
            raise ValueError(f"{where}: 'decision_type' must be a non-empty string")
        if self.confidence is not None:
            out = self.confidence
            if (
                isinstance(out, bool)
                or not isinstance(out, (int, float))
                or not math.isfinite(float(out))
                or not 0.0 <= float(out) <= 1.0
            ):
                raise ValueError(f"{where}: 'confidence' must be in [0, 1] or null, got {out!r}")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize (datetimes as ISO-8601)."""
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "job_id": self.job_id,
            "decision_type": self.decision_type,
            "candidates": validate_json_mapping(self.candidates, "<decision>: 'candidates'"),
            "probabilities": validate_json_mapping(self.probabilities, "<decision>: 'probabilities'"),
            "selected": validate_json_value(self.selected, "<decision>: 'selected'"),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> DecisionRecord:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: decision must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: decision {d.get('decision_id', '?')}"
        out = cls(
            decision_id=_req_str(d, "decision_id", where),
            session_id=_req_str(d, "session_id", where),
            node_id=_opt_str(d.get("node_id"), "node_id", where),
            job_id=_opt_str(d.get("job_id"), "job_id", where),
            decision_type=_req_str(d, "decision_type", where),
            candidates=validate_json_mapping(d.get("candidates", {}), f"{where}: 'candidates'"),
            probabilities=validate_json_mapping(d.get("probabilities", {}), f"{where}: 'probabilities'"),
            selected=validate_json_value(d.get("selected"), f"{where}: 'selected'"),
            confidence=_opt_confidence(d.get("confidence"), "confidence", where),
            created_at=_req_time(d, "created_at", where),
        )
        out.validate(where)
        return out


@dataclass(frozen=True)
class ToolDecision:
    """JEV tool disposition: one tool, a parallel set, reasoning escalation, or node-resolved."""

    action: str
    tool_name: str | None = None
    tool_names: tuple[str, ...] = ()
    probabilities: dict[str, JSONValue] = field(default_factory=dict)
    confidence: float | None = None

    def validate(self, where: str = "<tool_decision>") -> None:
        """Raise ValueError on any contract violation."""
        _coerce_enum(TOOL_DECISION_ACTIONS, self.action, "action", where)
        if self.action == "invoke":
            if not self.tool_names:
                raise ValueError(f"{where}: 'tool_names' must be non-empty when action is 'invoke'")
            if self.tool_name is not None and self.tool_name not in self.tool_names:
                raise ValueError(f"{where}: 'tool_name' must be a member of 'tool_names'")
        else:
            if self.tool_names:
                raise ValueError(f"{where}: 'tool_names' must be empty unless action is 'invoke'")
            if self.tool_name is not None:
                raise ValueError(f"{where}: 'tool_name' must be null unless action is 'invoke'")

    @property
    def selected_tools(self) -> tuple[str, ...]:
        """Effective parallel set (tool_names; single-tool compat falls back to tool_name)."""
        if self.tool_names:
            return self.tool_names
        return (self.tool_name,) if self.tool_name else ()

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize (tuples as lists)."""
        return {
            "action": self.action,
            "tool_name": self.tool_name,
            "tool_names": _json_str_list(list(self.tool_names)),
            "probabilities": validate_json_mapping(self.probabilities, "<tool_decision>: 'probabilities'"),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object], _path: str = "<dict>") -> ToolDecision:
        """Parse and validate; raises ValueError on malformed input."""
        if not isinstance(d, dict):
            raise ValueError(f"{_path}: tool decision must be a mapping, got {type(d).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = f"{_path}: tool decision"
        out = cls(
            action=_coerce_enum(TOOL_DECISION_ACTIONS, d.get("action"), "action", where),
            tool_name=_opt_str(d.get("tool_name"), "tool_name", where),
            tool_names=_tool_names_from(d, where),
            probabilities=validate_json_mapping(d.get("probabilities", {}), f"{where}: 'probabilities'"),
            confidence=_opt_confidence(d.get("confidence"), "confidence", where),
        )
        out.validate(where)
        return out
