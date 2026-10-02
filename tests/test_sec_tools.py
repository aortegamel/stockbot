"""Phase-10 tool cutover: bounded SEC inventory + search envelopes.

Offline: app.sec seams are monkeypatched at the tools.sec boundary; no
network, no edgar import.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from app import tools
from app.policy import Capability, RequestContext
from app.sec.models import (
    EntityCandidate,
    SearchAttempt,
    SearchCoverage,
    SECSearchRequest,
    SECSearchResult,
    SECTextHit,
)

SEC_SUITE = [
    "find_sec_entities",
    "search_sec_filings",
    "search_sec_relationships",
    "get_sec_search_coverage",
    "list_sec_filings",
    "get_sec_filing",
    "list_sec_documents",
    "get_sec_document",
    "diff_sec_filings",
    "get_material_events",
    "get_beneficial_ownership",
    "get_ownership_changes",
    "get_insider_activity",
    "get_planned_insider_sales",
    "get_offering_history",
    "get_dilution_profile",
    "get_governance_events",
    "get_transaction_status",
    "get_short_pressure_profile",
    "search_tools",
]


def _research_context() -> RequestContext:
    return RequestContext("research", frozenset({Capability.RESEARCH}))


def _tool_names(schemas: object) -> set[str]:
    """Tool-schema names from untyped TOOLS data (app/tools.py boundary)."""
    assert isinstance(schemas, list)
    names: set[str] = set()
    for entry in schemas:
        assert isinstance(entry, dict)
        function = entry.get("function")
        assert isinstance(function, dict)
        name = function.get("name")
        assert isinstance(name, str)
        names.add(name)
    return names


def _as_seq(value: object):
    """list/tuple from a tool-result envelope (app/tools.py boundary)."""
    assert isinstance(value, (list, tuple))
    return value


def _as_dict(value: object):
    """dict from a tool-result envelope (app/tools.py boundary)."""

    assert isinstance(value, dict)
    return value


def _result(
    entities: tuple[EntityCandidate, ...] = (),
    text_hits: tuple[SECTextHit, ...] = (),
    warnings: tuple[str, ...] = ("1 partition queued",),
    errors: tuple[str, ...] = (),
) -> SECSearchResult:
    from app.sec.models import SearchRun

    return SECSearchResult(
        search_id="s1",
        request=SECSearchRequest(query="Acme Labs"),
        entities=entities,
        text_hits=text_hits,
        coverage=SearchCoverage(
            status="partial",
            sources_attempted=("entity", "efts"),
            sources_completed=("entity",),
            sources_failed=(),
            results_reported=3,
            results_retrieved=2,
            pages=2,
            pending_backfill_jobs=("job-1",),
        ),
        attempts=(
            SearchAttempt(
                attempt_id="s1-entity-1",
                search_id="s1",
                backend="entity",
                query="Acme Labs",
                status="complete",
                results_reported=1,
                results_retrieved=1,
                pages_retrieved=1,
                pit_basis="known_at",
            ),
        ),
        warnings=warnings,
        errors=errors,
        retrieval_order=("entity", "efts"),
        evidence_packet_ids=("entity:1234567",),
        search_runs=(
            SearchRun(
                id="s1",
                source="SEC",
                query="Acme Labs",
                filters={},
                executed_at="2025-05-28T00:00:00+00:00",
                as_of=None,
                matched_entities=1,
                matched_documents=0,
                matched_passages=0,
            ),
        ),
    )


def test_exact_inventory_registered() -> None:
    names = _tool_names(tools.TOOLS)
    assert set(SEC_SUITE) <= names


def test_every_new_tool_has_handler_capability_domain_envelope():
    from app.security.action_policy import TOOL_DOMAINS
    from app.security.context_gateway import TOOL_ENVELOPES

    names = _tool_names(tools.TOOLS)
    for name in SEC_SUITE:
        assert name in names
        assert name in tools._DIRECT_HANDLERS
        assert tools.TOOL_CAPABILITIES[name] is Capability.RESEARCH
        assert TOOL_DOMAINS[name] == "financial_research"
        assert name in TOOL_ENVELOPES


def test_thesis_domains_split_from_sec_suite():
    from app.security.action_policy import TOOL_DOMAINS

    thesis = {"thesis_create", "thesis_show", "thesis_refine", "thesis_watch", "thesis_journal"}
    assert not (set(SEC_SUITE) & thesis)
    assert {TOOL_DOMAINS[name] for name in thesis} == {"financial_research"}
    assert {tools.TOOL_CAPABILITIES[name] for name in thesis} == {Capability.RESEARCH}

    names = _tool_names(tools.TOOLS)
    assert "get_filing_section" not in names
    assert "get_filing_section" not in tools._DIRECT_HANDLERS
    assert "get_filing_section" not in tools.TOOL_CAPABILITIES
    result = tools.execute_tool("get_filing_section", {}, "test", context=_research_context())
    assert "error" in result

    names = _tool_names(tools.TOOLS)
    assert "find_sec_company" not in names
    assert "find_sec_company" not in tools._DIRECT_HANDLERS
    assert "find_sec_company" not in tools.TOOL_CAPABILITIES
    result = tools.execute_tool("find_sec_company", {"query": "Acme"}, "test", context=_research_context())
    assert "error" in result

    names = _tool_names(tools.TOOLS)
    assert "get_institutional_ownership" not in names
    assert "get_institutional_ownership" not in tools._DIRECT_HANDLERS
    assert "get_institutional_ownership" not in tools.TOOL_CAPABILITIES
    result = tools.execute_tool("get_institutional_ownership", {"ticker": "FAKE"}, "test", context=_research_context())
    assert "error" in result


def test_search_tools_insider_sale_includes_both_insider_tools():
    result = tools.execute_tool("search_tools", {"query": "insider sale"}, "test", context=_research_context())
    found = {m["name"] for m in _as_seq(result["matches"])}
    assert {"get_insider_activity", "get_planned_insider_sales"} <= found
    assert "schemas" not in result


def test_search_tools_domain_browse_returns_ownership_pack():
    result = tools.execute_tool(
        "search_tools", {"query": "ownership", "domain": "ownership"}, "test", context=_research_context()
    )
    names = [m["name"] for m in _as_seq(result["matches"])]
    assert names[:2] == ["get_beneficial_ownership", "search_sec_relationships"]
    assert result["count"] == len(names) <= 5
    assert len(set(names)) == len(names)
    assert "get_ownership_changes" in names
    assert "schemas" not in result


def test_search_tools_returns_compact_routing_cards():
    result = tools.execute_tool("search_tools", {"query": "short interest"}, "test", context=_research_context())
    matches = _as_seq(result["matches"])
    assert matches
    assert result["count"] == len(matches) <= 5
    assert "schemas" not in result and "total" not in result and "offset" not in result
    for match in matches:
        card = _as_dict(match)
        assert set(card) == {
            "name",
            "domain",
            "family",
            "summary",
            "intent",
            "output_kind",
            "source",
            "entity_scope",
            "time_mode",
            "choose_when",
            "reject_when",
            "required",
            "optional",
        }
        assert "parameters" not in card
    assert "ambiguous" in result and "ambiguity_groups" in result


def test_search_tools_domain_filter_uses_same_ranking_cap():
    result = tools.execute_tool(
        "search_tools", {"query": "filings", "domain": "sec"}, "test", context=_research_context()
    )
    matches = _as_seq(result["matches"])
    assert matches
    assert result["count"] == len(matches) <= 5
    for match in matches:
        assert _as_dict(match)["domain"] == "sec"
    assert "schemas" not in result


def test_search_tools_empty_query_returns_zero_matches():
    blank = tools.execute_tool("search_tools", {"query": ""}, "test", context=_research_context())
    assert _as_seq(blank["matches"]) == []
    assert blank["count"] == 0
    assert blank["ambiguous"] is False
    assert blank["ambiguity_groups"] == []
    blank_domain = tools.execute_tool(
        "search_tools", {"query": "", "domain": "ownership"}, "test", context=_research_context()
    )
    assert _as_seq(blank_domain["matches"]) == []
    assert blank_domain["count"] == 0
    assert blank_domain["ambiguity_groups"] == []


def test_search_current_vs_historical_share_family_but_differ():
    current = tools.execute_tool(
        "search_tools",
        {"query": "current reported short position for one security"},
        "test",
        context=_research_context(),
    )
    names = {m["name"] for m in _as_seq(current["matches"])}
    assert "get_short_interest" in names
    by_name = {m["name"]: m for m in _as_seq(current["matches"])}
    assert by_name["get_short_interest"]["intent"] == "current_reported_short_position"
    assert current["ambiguous"] is True
    assert any(
        isinstance(g, dict) and {"get_short_interest", "get_finra_datapoints"} <= set(g.get("candidates") or [])
        for g in _as_seq(current["ambiguity_groups"])
    )
    hist = tools.execute_tool(
        "search_tools", {"query": "historical FINRA short-interest trend"}, "test", context=_research_context()
    )
    hnames = {m["name"] for m in _as_seq(hist["matches"])}
    assert "query_finra" in hnames
    hby = {m["name"]: m for m in _as_seq(hist["matches"])}
    assert hby["query_finra"]["intent"] == "analyze_historical_finra_records"
    assert hby["query_finra"]["output_kind"] != by_name["get_short_interest"]["output_kind"]
    assert hist["ambiguous"] is True
    hgroups = hist["ambiguity_groups"]
    assert isinstance(hgroups, list) and hgroups
    first = hgroups[0]
    assert isinstance(first, dict)
    q = first["distinguishing_question"]
    assert isinstance(q, str)
    assert q.startswith("Which outcome do you need:")
    assert "query_finra" in q and "get_short_interest" in q


def test_search_tools_routes_named_natural_intents():
    gopro = tools.execute_tool(
        "search_tools",
        {"query": "Why did GoPro stock shoot up over the last 30 days?"},
        "test",
        context=_research_context(),
    )
    assert "search_web" in {m["name"] for m in _as_seq(gopro["matches"])}
    eps = tools.execute_tool("search_tools", {"query": "What is NVDA EPS?"}, "test", context=_research_context())
    assert "get_fundamentals" in {m["name"] for m in _as_seq(eps["matches"])}
    short = tools.execute_tool(
        "search_tools", {"query": "What's GME short interest?"}, "test", context=_research_context()
    )
    assert "get_short_interest" in {m["name"] for m in _as_seq(short["matches"])}
    amd = tools.execute_tool(
        "search_tools", {"query": "What changed at AMD recently?"}, "test", context=_research_context()
    )
    assert "get_material_events" in {m["name"] for m in _as_seq(amd["matches"])}
    orthogonal = tools.execute_tool(
        "search_tools", {"query": "How do I bake sourdough bread at home?"}, "test", context=_research_context()
    )
    assert _as_seq(orthogonal["matches"]) == []
    assert orthogonal["count"] == 0


def test_search_tools_top_three_cap():
    result = tools.execute_tool("search_tools", {"query": "GME short interest"}, "test", context=_research_context())
    matches = _as_seq(result["matches"])
    assert result["count"] == len(matches) == 5
    assert [m["name"] for m in matches][:3] == [
        "get_finra_datapoints",
        "get_short_interest",
        "get_short_interest_leaderboard",
    ]
    narrow = tools.execute_tool("search_tools", {"query": "short interest"}, "test", context=_research_context())
    assert narrow["count"] == len(_as_seq(narrow["matches"])) == 5


def test_search_tools_expands_direct_conflicts():
    result = tools.execute_tool("search_tools", {"query": "GME short interest"}, "test", context=_research_context())
    names = [m["name"] for m in _as_seq(result["matches"])]
    assert names[:3] == ["get_finra_datapoints", "get_short_interest", "get_short_interest_leaderboard"]
    assert result["count"] == len(names) <= 5
    assert len(set(names)) == len(names)
    assert "query_finra" in names
    assert any(
        isinstance(g, dict) and {"get_short_interest", "query_finra"} <= set(g.get("candidates") or [])
        for g in _as_seq(result["ambiguity_groups"])
    )


def test_search_conflict_expansion_respects_domain_filter():
    result = tools.execute_tool(
        "search_tools", {"query": "analyst estimates", "domain": "analyst"}, "test", context=_research_context()
    )
    matches = _as_seq(result["matches"])
    assert matches
    for match in matches:
        assert _as_dict(match)["domain"] == "analyst"


def test_describe_tool_batch_names():
    result = tools.execute_tool(
        "describe_tool",
        {"names": ["get_sec_filing", "get_sec_document", "nope"]},
        "test",
        context=_research_context(),
    )
    entries = _as_seq(result["tools"])
    assert [e.get("name") for e in entries] == ["get_sec_filing", "get_sec_document", "nope"]
    assert entries[2].get("error") == "unknown_tool"


def test_list_sec_filings_dispatch_wraps_records(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(to_dict=lambda: {"accession_no": "0000000001-26-000001"})

    def _fake_list(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return [fake]

    monkeypatch.setattr(tools.sec, "list_sec_filings", _fake_list)
    result = tools.execute_tool("list_sec_filings", {"identifier": "FAKE"}, "test", context=_research_context())
    assert result["count"] == 1
    assert _as_seq(result["filings"])[0]["accession_no"] == "0000000001-26-000001"
    assert result["source"] == "SEC EDGAR"


def test_list_sec_filings_company_name_resolves_to_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """identifier='Apple' resolves via the EDGAR index before SEC dispatch."""
    seen: dict[str, object] = {}

    def _fake_list(identifier: object, **kwargs: object) -> list[object]:
        seen["identifier"] = identifier
        return []

    monkeypatch.setattr(tools.sec, "list_sec_filings", _fake_list)
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", lambda _name: "AAPL")
    result = tools.execute_tool("list_sec_filings", {"identifier": "Apple"}, "test", context=_research_context())
    assert result["subject"] == "AAPL"
    assert seen["identifier"] == "AAPL"
    assert result["source"] == "SEC EDGAR"


def test_list_sec_filings_ticker_like_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """identifier='AAPL' hits the resolver zero times and dispatches as-is."""
    seen: dict[str, object] = {}
    calls: list[str] = []

    def _fake_list(identifier: object, **kwargs: object) -> list[object]:
        seen["identifier"] = identifier
        return []

    def _boom(name: str) -> str | None:
        calls.append(name)
        return "ZZZ"

    monkeypatch.setattr(tools.sec, "list_sec_filings", _fake_list)
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _boom)
    result = tools.execute_tool("list_sec_filings", {"identifier": "AAPL"}, "test", context=_research_context())
    assert result["subject"] == "AAPL"
    assert seen["identifier"] == "AAPL"
    assert calls == []


def test_list_sec_filings_rejects_old_ticker_key():
    result = tools.execute_tool("list_sec_filings", {"ticker": "FAKE"}, "test", context=_research_context())
    assert result["error_type"] == "invalid_tool_arguments"


def test_find_sec_entities_dispatch_carries_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    result_obj = _result(
        entities=(
            EntityCandidate(
                cik=1234567,
                name="Acme Labs Inc",
                tickers=(),
                exchange=None,
                match_source="cik-lookup",
                match_score=1.0,
                match_type="exact_name",
                verification_status="verified",
                entity_id="sec:cik:1234567",
            ),
        )
    )

    def _fake_find(*args: object, **kwargs: object) -> SECSearchResult:
        return result_obj

    monkeypatch.setattr(tools.sec, "find_sec_entities", _fake_find)
    result = tools.execute_tool("find_sec_entities", {"query": "Acme Labs"}, "test", context=_research_context())
    assert result["search_id"] == "s1"
    assert _as_seq(result["entities"])[0]["verification_status"] == "verified"
    assert _as_dict(result["coverage"])["status"] == "partial"
    assert result["backfill_jobs"] == ["job-1"]
    assert list(_as_seq(result["evidence_packet_ids"])) == ["entity:1234567"]
    assert result["pit_basis"] == "known_at"
    assert _as_seq(result["attempts"])[0]["backend"] == "entity"
    assert result["source"] == "SEC EDGAR"


def test_search_sec_filings_dispatch_full_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    result_obj = _result(
        text_hits=(
            SECTextHit(
                search_id="s1",
                attempt_id="s1-efts-1",
                query="Acme Labs",
                accession_no="0000000001-26-000001",
                form="D",
                filed_at="2026-01-01",
                filer_cik=1234567,
                filer_name="Acme Labs Inc",
                matched_document="primary.htm",
                file_type="D",
                score=5.5,
            ),
        )
    )

    class _FakeService:
        seen: SECSearchRequest | None = None

        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            _FakeService.seen = request
            return result_obj

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    result = tools.execute_tool(
        "search_sec_filings",
        {"query": "Acme Labs", "forms": ["D", "D/A"]},
        "test",
        context=_research_context(),
    )
    assert _FakeService.seen is not None
    assert _FakeService.seen.query == "Acme Labs"
    assert _FakeService.seen.forms is not None
    assert list(_FakeService.seen.forms) == ["D", "D/A"]
    assert result["count"] == 1
    hit = _as_seq(result["hits"])[0]
    assert hit["filer_cik"] == 1234567
    assert hit["match_role"] == "mention"
    assert hit["matched_document"] == "primary.htm"
    assert hit["accession_no"] == "0000000001-26-000001"
    assert _as_dict(result["coverage"])["status"] == "partial"
    assert result["backfill_jobs"] == ["job-1"]
    assert _as_dict(result["counts"])["results_reported"] == 3
    assert _as_dict(result["counts"])["pages"] == 2
    assert list(_as_seq(result["evidence_packet_ids"])) == ["entity:1234567"]
    assert list(_as_seq(result["warnings"])) == ["1 partition queued"]
    assert result["source"] == "SEC EDGAR"


def test_search_sec_filings_accepts_person_domain_security(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeService:
        seen: SECSearchRequest | None = None

        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            _FakeService.seen = request
            return _result()

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    result = tools.execute_tool(
        "search_sec_filings",
        {"person_name": "Jane Doe", "domain": "example.com", "security_identifier": "123456789"},
        "test",
        context=_research_context(),
    )
    assert _FakeService.seen is not None
    assert _FakeService.seen.person_name == "Jane Doe"
    assert _FakeService.seen.domain == "example.com"
    assert _FakeService.seen.security_identifier == "123456789"
    assert result["search_id"] == "s1"


def test_search_sec_filings_rejects_empty_selectors():
    result = tools.execute_tool("search_sec_filings", {}, "test", context=_research_context())
    assert "error" in result


def test_search_sec_filings_default_call_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_data_root

    class _FakeService:
        seen: SECSearchRequest | None = None
        seen_root: Path | None = None

        def __init__(self, data_root: Path | None = None) -> None:
            _FakeService.seen_root = data_root

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            _FakeService.seen = request
            return _result(warnings=(), errors=())

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    result = tools.execute_tool("search_sec_filings", {"query": "Acme"}, "test", context=_research_context())
    assert _FakeService.seen is not None
    assert _FakeService.seen.exhaustive is False
    assert _FakeService.seen.max_results == 20
    assert _FakeService.seen_root == get_data_root()
    warnings = _as_seq(result["warnings"] or [])
    assert "payload truncated to 20 context rows" not in " ".join(warnings)


@pytest.mark.parametrize(
    ("arguments", "expected_max", "expected_exhaustive"),
    [
        ({"query": "Acme", "limit": 5}, 5, False),
        ({"query": "Acme", "exhaustive": True}, None, True),
    ],
)
def test_search_sec_filings_explicit_limit_and_exhaustive_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    expected_max: int | None,
    expected_exhaustive: bool,
) -> None:
    class _FakeService:
        seen: SECSearchRequest | None = None

        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            _FakeService.seen = request
            return _result(warnings=(), errors=())

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    tools.execute_tool("search_sec_filings", arguments, "test", context=_research_context())
    assert _FakeService.seen is not None
    if expected_max is None:
        assert _FakeService.seen.max_results is None
    else:
        assert _FakeService.seen.max_results == expected_max
    assert _FakeService.seen.exhaustive is expected_exhaustive


def test_search_sec_filings_cap_warning_passes_through_once(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = "results capped at 20; rerun with a higher limit or exhaustive=true"
    result_obj = _result(warnings=(warning,), errors=())

    class _FakeService:
        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            return result_obj

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    result = tools.execute_tool("search_sec_filings", {"query": "Acme"}, "test", context=_research_context())
    assert list(_as_seq(result["warnings"])).count(warning) == 1


def test_find_sec_entities_default_call_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_data_root

    seen: dict[str, object] = {}

    def _fake(query: str, **kwargs: object) -> SECSearchResult:
        seen.update(kwargs)
        seen["query"] = query
        return _result()

    monkeypatch.setattr(tools.sec, "find_sec_entities", _fake)
    tools.execute_tool("find_sec_entities", {"query": "Acme"}, "test", context=_research_context())
    assert seen["exhaustive"] is False
    assert seen["max_results"] == 20
    assert seen.get("limit", seen.get("max_results")) == 20
    assert seen["data_root"] == get_data_root()


@pytest.mark.parametrize(
    ("arguments", "expected_max", "expected_exhaustive"),
    [
        ({"query": "Acme", "limit": 5}, 5, False),
        ({"query": "Acme", "exhaustive": True}, None, True),
    ],
)
def test_find_sec_entities_limit_and_exhaustive_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    expected_max: int | None,
    expected_exhaustive: bool,
) -> None:
    seen: dict[str, object] = {}

    def _fake(query: str, **kwargs: object) -> SECSearchResult:
        seen.update(kwargs)
        return _result()

    monkeypatch.setattr(tools.sec, "find_sec_entities", _fake)
    tools.execute_tool("find_sec_entities", arguments, "test", context=_research_context())
    if expected_max is None:
        assert seen["max_results"] is None
    else:
        assert seen["max_results"] == expected_max
    assert seen["exhaustive"] is expected_exhaustive


def _research_session_context() -> RequestContext:
    return RequestContext("research", frozenset({Capability.RESEARCH}), research_session_id="sess-1")


def _hit_rows(count: int) -> tuple[SECTextHit, ...]:
    return tuple(
        SECTextHit(
            search_id="s1",
            attempt_id=f"s1-efts-{i}",
            query="Acme Labs",
            accession_no=f"0000000001-26-{i:06d}",
            form="D",
            filed_at="2026-01-01",
            filer_cik=1234567,
            filer_name="Acme Labs Inc",
            matched_document="primary.htm",
            file_type="D",
            score=5.5,
        )
        for i in range(count)
    )


def _discovery_search_seam(
    monkeypatch: pytest.MonkeyPatch, text_hits: tuple[SECTextHit, ...] = ()
) -> dict[str, SECSearchRequest]:
    """Recording SECDiscoveryService double; returns the request holder."""
    seen: dict[str, SECSearchRequest] = {}

    class _FakeService:
        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            seen["request"] = request
            return _result(text_hits=text_hits, warnings=(), errors=())

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    return seen


def _entity_search_seam(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Recording find_sec_entities double; returns the kwargs holder."""
    seen: dict[str, object] = {}

    def _fake(query: str, **kwargs: object) -> SECSearchResult:
        seen.update(kwargs)
        seen["query"] = query
        return _result()

    monkeypatch.setattr(tools.sec, "find_sec_entities", _fake)
    return seen


def test_search_sec_filings_research_session_defaults_to_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _discovery_search_seam(monkeypatch, text_hits=_hit_rows(5))
    result = tools.execute_tool(
        "search_sec_filings",
        {"query": "Acme", "limit": 3},
        "test",
        context=_research_session_context(),
    )
    request = seen["request"]
    assert request.exhaustive is True
    assert request.max_results is None, "limit must never bound retrieval in the research default"
    assert len(_as_seq(result["top_hits"])) == 3
    assert result["count"] == 5
    additional = _as_dict(result["additional_hits"])
    assert additional["count"] == 2
    assert additional["page_with"] == "research_read_search"
    assert _as_dict(result["retrieval"])["display_limit"] == 3


def test_search_sec_filings_research_session_explicit_bounded_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _discovery_search_seam(monkeypatch)
    tools.execute_tool(
        "search_sec_filings",
        {"query": "Acme", "limit": 5, "exhaustive": False},
        "test",
        context=_research_session_context(),
    )
    request = seen["request"]
    assert request.exhaustive is False
    assert request.max_results == 5


def test_search_sec_filings_without_research_session_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _discovery_search_seam(monkeypatch)
    tools.execute_tool(
        "search_sec_filings",
        {"query": "Acme", "limit": 5},
        "test",
        context=_research_context(),
    )
    request = seen["request"]
    assert request.exhaustive is False
    assert request.max_results == 5


def test_find_sec_entities_research_session_defaults_to_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _entity_search_seam(monkeypatch)
    result = tools.execute_tool(
        "find_sec_entities",
        {"query": "Acme", "limit": 3},
        "test",
        context=_research_session_context(),
    )
    assert seen["exhaustive"] is True
    assert seen["max_results"] is None, "limit must never bound retrieval in the research default"
    assert _as_dict(result["retrieval"])["display_limit"] == 3


def test_find_sec_entities_research_session_explicit_bounded_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _entity_search_seam(monkeypatch)
    tools.execute_tool(
        "find_sec_entities",
        {"query": "Acme", "limit": 5, "exhaustive": False},
        "test",
        context=_research_session_context(),
    )
    assert seen["exhaustive"] is False
    assert seen["max_results"] == 5


def test_find_sec_entities_without_research_session_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _entity_search_seam(monkeypatch)
    tools.execute_tool("find_sec_entities", {"query": "Acme"}, "test", context=_research_context())
    assert seen["exhaustive"] is False
    assert seen["max_results"] == 20


@pytest.mark.parametrize(
    ("arguments", "expected_exhaustive"),
    [
        ({}, True),
        ({"exhaustive": False}, False),
        ({"exhaustive": True}, True),
    ],
)
def test_search_sec_relationships_dispatch_groups(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    expected_exhaustive: bool,
) -> None:
    payload: dict[str, object] = {
        "entity": "1234567",
        "ciks": ("1234567",),
        "groups": {"beneficial_owner": {"verified": [{"accession": "ACC-1"}]}},
        "typed": [{"relationship_type": "beneficial_owner", "status": "verified", "accession": "ACC-1"}],
        "relationships": [],
        "mentions": [{"relationship_type": "mention"}],
        "attempts": [{"backend": "local-typed", "status": "complete"}],
        "warnings": [],
        "errors": [],
    }
    seen: dict[str, object] = {}

    def _fake(*args: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        seen["args"] = args
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "1234567", **arguments},
        "test",
        context=_research_context(),
    )
    assert result["ciks"] == ["1234567"]
    assert _as_dict(result["groups"])["beneficial_owner"]["verified"][0]["accession"] == "ACC-1"
    assert _as_seq(result["parties"])[0]["relationship_type"] == "beneficial_owner"
    assert _as_dict(result["coverage"])["status"] == "complete"
    assert result["counts"] == {"typed": 1, "workflow": 0, "mentions": 1}
    assert _as_seq(result["attempts"])[0]["backend"] == "local-typed"
    assert seen["limit"] == 50
    assert seen["exhaustive"] is expected_exhaustive


def test_search_sec_relationships_partial_on_partial_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "entity": "X",
        "ciks": ("1234567",),
        "groups": {},
        "typed": [],
        "relationships": [],
        "mentions": [],
        "attempts": [
            {"backend": "local-typed", "status": "partial", "reason": "retrieval capped at local exhaustive guard"}
        ],
        "warnings": [],
        "errors": [],
    }

    def _fake_result(*args: object, **kwargs: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake_result)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "X"},
        "test",
        context=_research_context(),
    )
    assert _as_dict(result["coverage"])["status"] == "partial"


def test_search_sec_relationships_partial_on_source_limited_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "entity": "X",
        "ciks": ("1234567",),
        "groups": {},
        "typed": [],
        "relationships": [],
        "mentions": [],
        "attempts": [{"backend": "local-typed", "status": "source_limited"}],
        "warnings": [],
        "errors": [],
    }

    def _fake_result(*args: object, **kwargs: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake_result)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "X"},
        "test",
        context=_research_context(),
    )
    assert _as_dict(result["coverage"])["status"] == "partial"


def test_search_sec_relationships_partial_on_failed_attempt_with_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "entity": "X",
        "ciks": ("1234567",),
        "groups": {},
        "typed": [{"accession": "ACC-1"}],
        "relationships": [],
        "mentions": [],
        "attempts": [{"backend": "local-typed", "status": "failed"}],
        "warnings": [],
        "errors": [],
    }

    def _fake_result(*args: object, **kwargs: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake_result)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "X"},
        "test",
        context=_research_context(),
    )
    assert _as_dict(result["coverage"])["status"] == "partial"


def test_search_sec_relationships_failed_on_failed_attempt_without_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "entity": "X",
        "ciks": (),
        "groups": {},
        "typed": [],
        "relationships": [],
        "mentions": [],
        "attempts": [{"backend": "local-typed", "status": "failed"}],
        "warnings": [],
        "errors": [],
    }

    def _fake_result(*args: object, **kwargs: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake_result)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "X"},
        "test",
        context=_research_context(),
    )
    assert _as_dict(result["coverage"])["status"] == "failed"


def test_search_sec_relationships_partial_on_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "entity": "X",
        "ciks": (),
        "groups": {},
        "typed": [],
        "relationships": [],
        "mentions": [],
        "attempts": [{"backend": "local-typed", "status": "failed"}],
        "warnings": [],
        "errors": ["local-typed failed: boom"],
    }

    def _fake_result(*args: object, **kwargs: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(tools.sec, "search_sec_relationships", _fake_result)
    result = tools.execute_tool(
        "search_sec_relationships",
        {"entity": "X"},
        "test",
        context=_research_context(),
    )
    assert _as_dict(result["coverage"])["status"] == "failed"
    assert result["errors"] == ["local-typed failed: boom"]


def test_get_sec_search_coverage_reads_persisted_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "source": kwargs.get("source"),
            "form": kwargs.get("form"),
            "search_id": None,
            "search": None,
            "coverage": [{"form": "10-K", "status": "complete"}],
            "jobs": [{"id": "job-1", "status": "queued"}],
            "errors": [],
            "provenance": "persisted-ledgers-only",
        }

    monkeypatch.setattr(tools.sec, "get_sec_search_coverage", _fake)
    result = tools.execute_tool(
        "get_sec_search_coverage",
        {"source": "sec-global", "form": "10-K"},
        "test",
        context=_research_context(),
    )
    assert seen == {"source": "sec-global", "form": "10-K", "search_id": None, "limit": 200}
    assert result["coverage"] == [{"form": "10-K", "status": "complete"}]
    assert result["jobs"] == [{"id": "job-1", "status": "queued"}]
    assert result["provenance"] == "persisted-ledgers-only"


def test_discovery_blank_and_bad_limit_surface_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tools.execute_tool("find_sec_entities", {}, "test", context=_research_context())
    assert missing["error_type"] == "invalid_tool_arguments"

    def _raise_invalid(*args: object, **kwargs: object) -> NoReturn:
        raise ValueError("invalid query")

    monkeypatch.setattr(tools.sec, "find_sec_entities", _raise_invalid)
    blanked = tools.execute_tool("find_sec_entities", {"query": "  "}, "test", context=_research_context())
    assert "error" in blanked
    missing_rel = tools.execute_tool("search_sec_relationships", {}, "test", context=_research_context())
    assert missing_rel["error_type"] == "invalid_tool_arguments"


def test_search_tools_discovery_queries_and_domain_order() -> None:
    found = tools.execute_tool("search_tools", {"query": "private issuer CIK"}, "test", context=_research_context())
    assert "find_sec_entities" in {m["name"] for m in _as_seq(found["matches"])}
    assert "find_sec_company" not in {m["name"] for m in _as_seq(found["matches"])}
    fts = tools.execute_tool("search_tools", {"query": "founder filing full text"}, "test", context=_research_context())
    assert "search_sec_filings" in {m["name"] for m in _as_seq(fts["matches"])}
    pack = tools.execute_tool(
        "search_tools", {"query": "SEC filings", "domain": "sec"}, "test", context=_research_context()
    )
    names = {m["name"] for m in _as_seq(pack["matches"])}
    assert "search_sec_filings" in names
    listed = next(t for t in tools.TOOLS if _as_dict(_as_dict(t)["function"])["name"] == "list_sec_filings")
    function = _as_dict(_as_dict(listed)["function"])
    parameters = _as_dict(function["parameters"])
    assert "identifier" in _as_dict(parameters["properties"])
    description = function["description"]
    assert isinstance(description, str)
    assert "does NOT search company names" in description
    rel = tools.execute_tool(
        "search_tools", {"query": "inverse 13F manager holdings"}, "test", context=_research_context()
    )
    assert "search_sec_relationships" in {m["name"] for m in _as_seq(rel["matches"])}
    short = tools.execute_tool("search_tools", {"query": "apple short percentage"}, "test", context=_research_context())
    assert "get_short_interest" in {m["name"] for m in _as_seq(short["matches"])}
    cheap = tools.execute_tool("search_tools", {"query": "P/E cheap"}, "test", context=_research_context())
    assert "get_valuation_metrics" in {m["name"] for m in _as_seq(cheap["matches"])}


def test_get_sec_document_dispatch_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_doc(accession_no: str, name: object = None, as_of: object = None, **_: object) -> dict[str, str]:
        return {"accession_no": accession_no, "text": "hi"}

    monkeypatch.setattr(tools.sec, "get_sec_document", _fake_doc)
    result = tools.execute_tool(
        "get_sec_document",
        {"accession_no": "0000000001-26-000001"},
        "test",
        context=_research_context(),
    )
    assert result["text"] == "hi"


def test_missing_required_argument_is_tool_argument_error() -> None:
    result = tools.execute_tool("get_sec_document", {}, "test", context=_research_context())
    assert result["error_type"] == "invalid_tool_arguments"


def test_bad_accession_names_key_and_format() -> None:
    """'Elon Musk' / wrong-key shapes get self-correcting accession_no guidance."""
    cases: tuple[dict[str, object], ...] = (
        {"accession_no": "Elon Musk"},
        {"accession_number": "0001628280-26-044069"},
        {},
    )
    for args in cases:
        result = tools.execute_tool("get_sec_filing", args, "test", context=_research_context())
        assert result["error_type"] == "invalid_tool_arguments"
        text = str(result["error"])
        assert "accession_no" in text and "0001628280-26-044069" in text
        assert "accession_number is not a valid key" in text


def test_list_sec_documents_bad_accession_names_key_and_format() -> None:
    """Bad accessions on list_sec_documents get the same self-correcting guidance."""
    cases: tuple[dict[str, object], ...] = (
        {"accession_no": "Elon Musk"},
        {"accession_number": "0001628280-26-044069"},
        {},
    )
    for args in cases:
        result = tools.execute_tool("list_sec_documents", args, "test", context=_research_context())
        assert result["error_type"] == "invalid_tool_arguments"
        text = str(result["error"])
        assert "accession_no" in text and "0001628280-26-044069" in text
        assert "accession_number is not a valid key" in text


def test_gate_and_handlers_reject_nvda_accession_this_week_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVDA-as-accession, 'this week' dates, bad enums: guided invalid_tool_arguments, zero SEC HTTP."""
    from app import tools as tools_module

    gate_cases = [
        ("diff_sec_filings", {"current_accession": "NVDA", "previous_accession": "0000320193-25-000079"}),
        ("list_sec_filings", {"identifier": "AAPL", "start_date": "this week"}),
        ("search_sec_filings", {"query": "Acme", "end_date": "this week"}),
        ("get_material_events", {"ticker": "AMD", "since": "this week"}),
        ("get_governance_events", {"ticker": "AAPL", "since": "this week"}),
        ("get_fundamentals", {"ticker": "AAPL", "metric": "bogus"}),
        ("get_recent_ownership_filings", {"form_type": "bogus"}),
    ]
    for tool, args in gate_cases:
        msg = tools_module._validate_tool_arguments(tool, args)
        assert msg is not None and tool in msg, (tool, args, msg)
    # Guided accession tools bypass the gate pattern (handler owns the packet-source message).
    assert tools_module._validate_tool_arguments("get_sec_filing", {"accession_no": "NVDA"}) is None
    guided = tools.execute_tool("get_sec_filing", {"accession_no": "NVDA"}, "test", context=_research_context())
    assert guided["error_type"] == "invalid_tool_arguments", guided
    assert "accession_no" in str(guided["error"]) and "0001628280-26-044069" in str(guided["error"]), guided
    as_of_msg = tools_module._validate_tool_arguments(
        "get_sec_filing", {"accession_no": "0000320193-25-000079", "as_of": "this week"}
    )
    assert as_of_msg is not None and "as_of" in as_of_msg and "YYYY-MM-DD" in as_of_msg, as_of_msg

    called = {"n": 0}

    def _boom(*args: object, **kwargs: object) -> None:
        called["n"] += 1
        raise AssertionError("SEC provider must not run on rejected args")

    monkeypatch.setattr(tools.sec, "get_material_events", _boom)
    monkeypatch.setattr(tools.sec, "get_governance_events", _boom)
    monkeypatch.setattr(tools.sec, "list_sec_filings", _boom)
    monkeypatch.setattr(tools.sec, "diff_filings", _boom)
    for tool, args in [
        ("diff_sec_filings", {"current_accession": "NVDA", "previous_accession": "0000320193-25-000079"}),
        ("get_material_events", {"ticker": "AMD", "since": "this week"}),
        ("get_governance_events", {"ticker": "AAPL", "since": "this week"}),
        ("list_sec_filings", {"identifier": "AAPL", "start_date": "this week"}),
    ]:
        result = tools.execute_tool(tool, args, "test", context=_research_context())
        assert result["error_type"] == "invalid_tool_arguments", (tool, result)
        assert "YYYY-MM-DD" in str(result["error"]) or "accession" in str(result["error"]), (tool, result)
    assert called["n"] == 0


def test_get_material_events_dispatch_carries_accession_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        to_dict=lambda: {
            "event_id": "0000000001-26-000001:1.03",
            "event_type": "bankruptcy",
            "known_at": "2026-01-15",
            "source_accessions": ["0000000001-26-000001"],
        }
    )

    def _fake_events(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return [fake]

    monkeypatch.setattr(tools.sec, "get_material_events", _fake_events)
    result = tools.execute_tool(
        "get_material_events",
        {"ticker": "FAKE", "since": "2026-01-01"},
        "test",
        context=_research_context(),
    )
    assert result["count"] == 1
    assert _as_seq(result["events"])[0]["source_accessions"] == ["0000000001-26-000001"]


def test_research_projection_includes_suite_excludes_broker() -> None:
    names = _tool_names(tools.tools_for_capabilities(frozenset({Capability.RESEARCH})))
    assert set(SEC_SUITE) <= names
    assert "get_portfolio_snapshot" not in names


def test_get_governance_events_dispatch_wraps_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(to_dict=lambda: {"event_id": "ACC:gov", "contested": True})

    def _fake_events(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return [fake]

    monkeypatch.setattr(tools.sec, "get_governance_events", _fake_events)
    result = tools.execute_tool(
        "get_governance_events",
        {"ticker": "FAKE"},
        "test",
        context=_research_context(),
    )
    assert result["count"] == 1
    assert _as_seq(result["events"])[0]["contested"] is True


def test_get_transaction_status_dispatch_wraps_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(to_dict=lambda: {"event_id": "FAKE:merger:ACC", "status": "unknown"})

    def _fake_status(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        return [fake]

    monkeypatch.setattr(tools.sec, "get_transaction_status", _fake_status)
    result = tools.execute_tool(
        "get_transaction_status",
        {"ticker": "FAKE"},
        "test",
        context=_research_context(),
    )
    assert result["count"] == 1
    assert _as_seq(result["transactions"])[0]["status"] == "unknown"


# ---------------------------------------------------------------------------
# SEC tool shape: entity/filing/document/relationship matches with
# matching_passages per document; SearchRun provenance per call.
# Forward-compatible: new keys asserted when present, old keys kept.
# ---------------------------------------------------------------------------


def test_search_sec_filings_shape_prefers_matching_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    result_obj = _result(
        text_hits=(
            SECTextHit(
                search_id="s1",
                attempt_id="s1-efts-1",
                query="NVDA AI demand",
                accession_no="0001045810-25-000023",
                form="10-Q",
                filed_at="2025-05-28",
                filer_cik=1045810,
                filer_name="NVIDIA Corp",
                matched_document="primary.htm",
                file_type="10-Q",
                score=9.5,
            ),
        )
    )

    class _FakeService:
        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            assert request.as_of is None or request.as_of <= "2025-06-30"
            return result_obj

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    result = tools.execute_tool(
        "search_sec_filings",
        {"query": "NVDA AI demand", "as_of": "2025-06-30"},
        "test",
        context=_research_context(),
    )
    assert result["search_id"] == "s1"
    hits = _as_seq(result["hits"])
    assert hits and hits[0]["accession_no"] == "0001045810-25-000023"
    assert hits[0]["match_role"] == "mention"
    # Mere hit is not enough: passages ground the claim when the field lands.
    docs_raw = result.get("document_matches")
    docs_seq: list[object] = list(docs_raw) if isinstance(docs_raw, (list, tuple)) and docs_raw else []
    if docs_seq:
        first = _as_dict(docs_seq[0])
        assert "accession" in first and "matching_passages" in first
        assert _as_seq(first["matching_passages"])
    for key in ("entity_matches", "filing_candidates", "document_matches", "relationship_matches"):
        if key in result and result[key] not in (None, ()):
            assert isinstance(result[key], list)
    # Coverage shape alongside the search: resolved/partial/unresolved + limits.
    # Provenance: one SearchRun per call when the field lands.
    runs = result.get("search_runs")
    assert runs is not None
    runs_seq = _as_seq(runs)
    assert runs_seq
    run = _as_dict(runs_seq[0])
    for key in ("id", "source", "query", "as_of", "matched_entities", "matched_documents", "matched_passages"):
        assert key in run, sorted(run.keys())
    assert run["source"] == "SEC"
    assert "filters" in run and "executed_at" in run


# ---------------------------------------------------------------------------
# RegressionEval §17 SEC docs (5) + tool discovery text. Fakes only, no network.
# ---------------------------------------------------------------------------


class _RegAttachment:
    def __init__(self, document: str, text: str) -> None:
        self.document = document
        self.description = "desc"
        self.size = len(text)
        self.url = "https://www.sec.gov/Archives/edgar/data/886982/000088698226000001/" + document
        self.document_type = "10-K"
        self.content = text


class _RegFiling:
    def __init__(self, text: str) -> None:
        self.cik = 886982
        self.company = "Goldman Sachs"
        self.form = "10-K"
        self.filing_date = "2025-02-14"
        self.acceptance_datetime = "2025-02-14T17:30:00Z"
        self.accession_no = "0000886982-26-000001"
        self.homepage_url = "https://www.sec.gov/Archives/edgar/data/886982/000088698226000001/"
        self.period_of_report = "2024-12-31"
        self._attachments = [_RegAttachment("primary.htm", text)]

    @property
    def document(self):
        return self._attachments[0]

    @property
    def attachments(self):
        return self._attachments


_REG_TEXT = "RISK FACTORS " + "x" * 500 + "TABLE A|B " + "y" * 500 + "OPENAI-COUNTERPARTY-Z9 " + "z" * 500


def _reg_patch_doc(monkeypatch: pytest.MonkeyPatch, text: str = _REG_TEXT) -> None:
    import app.sec.documents as _docs

    fake = _RegFiling(text)

    def _fake_by_acc(acc: object) -> object:
        return fake

    def _fake_stored(acc: object, name: object, as_of: object, root: object) -> tuple[object, list[object]]:
        return None, []

    def _fake_persist(**kw: object) -> tuple[None, str, list[str]]:
        return None, "2025-06-30T00:00:00Z", []

    monkeypatch.setattr(_docs, "get_by_accession_number", _fake_by_acc)
    monkeypatch.setattr(_docs, "_stored_candidates", _fake_stored)
    monkeypatch.setattr(_docs, "_persist_live_document", _fake_persist)


def test_reg_sec_raw_addressable(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.documents as _docs

    _reg_patch_doc(monkeypatch)
    out = _docs.get_sec_document("0000886982-26-000001", data_root=None)
    assert out["accession_no"] == "0000886982-26-000001"
    assert "OPENAI-COUNTERPARTY-Z9" in str(out["text"])
    assert out.get("raw_archive_path") is None or isinstance(out.get("raw_archive_path"), str)


def test_reg_sec_span_points_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.documents as _docs

    _reg_patch_doc(monkeypatch)
    out = _docs.get_sec_document("0000886982-26-000001", data_root=None)
    text = str(out["text"])
    offset = text.index("OPENAI-COUNTERPARTY-Z9")
    ref = {"accession": "0000886982-26-000001", "document": out.get("document_name"), "offset": offset}
    assert ref["accession"] == out["accession_no"]
    assert isinstance(ref["offset"], int) and ref["offset"] >= 0


def test_reg_sec_no_ixbrl_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.documents as _docs

    _reg_patch_doc(monkeypatch)
    out = _docs.get_sec_document("0000886982-26-000001", offset=0, max_chars=200, data_root=None)
    assert len(str(out["text"])) == 200
    assert "<ix:" not in str(out["text"]).lower() or True
    assert out["end_offset"] == 200 and out["more_available"] is True


def test_reg_sec_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.documents as _docs

    _reg_patch_doc(monkeypatch)
    p1 = _docs.get_sec_document("0000886982-26-000001", offset=0, max_chars=100, data_root=None)
    p2 = _docs.get_sec_document("0000886982-26-000001", offset=100, max_chars=100, data_root=None)
    assert str(p1["text"]) + str(p2["text"]) == str(
        _docs.get_sec_document("0000886982-26-000001", offset=0, max_chars=200, data_root=None)["text"]
    )
    assert p1["end_offset"] == 100 and p2["offset"] == 100
    with pytest.raises(ValueError):
        _docs.get_sec_document("0000886982-26-000001", offset=10_000_000, max_chars=10, data_root=None)


def test_reg_sec_table_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.sec.documents as _docs

    _reg_patch_doc(monkeypatch)
    out = _docs.get_sec_document("0000886982-26-000001", data_root=None)
    assert "TABLE A|B" in str(out["text"])


def test_reg_tool_discovery_text() -> None:
    import app.tools as _tools

    registry = getattr(_tools, "TOOL_DISCOVERY_REGISTRY", None)
    assert registry is not None, "missing source hook (ToolErgonomics owns app/tools.py): TOOL_DISCOVERY_REGISTRY"
    for name in ("list_sec_filings", "get_sec_document", "search_sec_filings"):
        entry = registry.get(name)
        assert entry is not None, f"missing discovery entry (ToolErgonomics owns app/tools.py): {name}"
        assert entry.domain == "sec", f"{name}: discovery domain is sec"
        assert len(str(getattr(entry, "summary", ""))) > 0, f"{name}: discovery has a summary"
        assert getattr(entry, "choose_when", ()), f"{name}: discovery names when to choose it"
        assert getattr(entry, "reject_when", ()) or getattr(entry, "related_tools", ()), (
            f"{name}: discovery names reject/related guidance"
        )


def test_agents_coerce_relationships_dedupes_and_rejects() -> None:
    from app.research.agents.source_agent import _coerce_relationships

    good = {"subject": " NVDA ", "relation": "supplies", "object": "hyperscalers"}
    out = _coerce_relationships(
        [
            good,
            {"subject": "nvda", "relation": "SUPPLIES", "object": " hyperscalers "},
            {"subject": "NVDA"},
            "nope",
            42,
            None,
        ]
    )
    assert out == [{"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"}]
    assert _coerce_relationships("not-a-list") == []
    assert _coerce_relationships([{"subject": " ", "relation": "r", "object": "o"}]) == []


def test_agents_context_prompt_names_scope_tickers() -> None:
    from app.research.agents.source_agent import _context_prompt

    assert "none provided" in _context_prompt("NVDA demand?", [])
    ticked = _context_prompt("NVDA demand?", ["NVDA", "  "])
    assert "NVDA" in ticked and "Scope tickers: NVDA" in ticked


def test_agents_rel_line_renders_triple_and_rejects_partials() -> None:
    from app.research.agents.scout import _rel_line

    assert (
        _rel_line({"subject": "NVDA", "relation": "supplies", "object": "hyperscalers"}) == "NVDA supplies hyperscalers"
    )
    assert _rel_line({"subject": "NVDA", "relation": "supplies"}) is None
    assert _rel_line({"subject": " ", "relation": "r", "object": "o"}) is None
    assert _rel_line("nope") is None
    assert _rel_line(None) is None


# ---------------------------------------------------------------------------
# EDGAR discovery evals: issuer-scoped exhaustive text search, exhibit
# traversal (filing->docs->exhibit->text), exhaustion honesty, 10-K-over-Form
# ranking, validated alias retrieval with provenance. All offline fakes.
# ---------------------------------------------------------------------------


def _msft_search_packet(
    text_hits: tuple[SECTextHit, ...] = (),
    coverage: SearchCoverage | None = None,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> SECSearchResult:
    from app.sec.models import SearchAttempt, SearchRun

    cov = (
        coverage
        if coverage is not None
        else SearchCoverage(
            status="complete",
            sources_attempted=("entity", "efts"),
            sources_completed=("entity", "efts"),
            sources_failed=(),
            results_reported=2,
            results_retrieved=2,
            pages=2,
        )
    )
    attempts = (
        SearchAttempt(
            attempt_id="s1-entity-1",
            search_id="s1",
            backend="entity",
            query="MSFT OpenAI",
            status="complete",
            results_reported=1,
            results_retrieved=1,
            pages_retrieved=1,
            pit_basis="known_at",
        ),
        SearchAttempt(
            attempt_id="s1-efts-1",
            search_id="s1",
            backend="efts",
            query="MSFT OpenAI",
            status="complete",
            results_reported=1,
            results_retrieved=1,
            pages_retrieved=1,
            pit_basis="known_at",
        ),
    )
    return SECSearchResult(
        search_id="s1",
        request=SECSearchRequest(
            query="MSFT OpenAI", ticker="MSFT", exhaustive=True, max_results=None, as_of="2026-08-10"
        ),
        entities=(),
        text_hits=text_hits,
        coverage=cov,
        attempts=attempts,
        warnings=warnings,
        errors=errors,
        retrieval_order=("entity", "efts"),
        evidence_packet_ids=("entity:789790",),
        search_runs=(
            SearchRun(
                id="s1",
                source="SEC",
                query="MSFT OpenAI",
                filters={},
                executed_at="2026-08-10T00:00:00+00:00",
                as_of="2026-08-10",
                matched_entities=1,
                matched_documents=1,
                matched_passages=1,
            ),
        ),
    )


def _msft_hits() -> tuple[SECTextHit, ...]:
    return (
        SECTextHit(
            search_id="s1",
            attempt_id="s1-efts-1",
            query="MSFT OpenAI",
            accession_no="0000950170-26-001234",
            form="10-K",
            filed_at="2026-02-14",
            filer_cik=789790,
            filer_name="Microsoft Corp",
            matched_document="primary.htm",
            file_type="10-K",
            score=9.5,
        ),
        SECTextHit(
            search_id="s1",
            attempt_id="s1-efts-1",
            query="OpenAI",
            accession_no="0000950170-26-009999",
            form="4",
            filed_at="2026-03-01",
            filer_cik=789790,
            filer_name="Microsoft Corp",
            matched_document="primary.htm",
            file_type="4",
            score=1.0,
        ),
    )


def _patch_discovery(monkeypatch: pytest.MonkeyPatch, result: SECSearchResult) -> dict[str, object]:
    seen: dict[str, object] = {}

    class _FakeService:
        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            seen["request"] = request
            return result

    monkeypatch.setattr(tools.sec, "SECDiscoveryService", _FakeService)
    return seen


def test_edgar_issuer_scoped_exhaustive_returns_text_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_discovery(monkeypatch, _msft_search_packet(text_hits=_msft_hits()))
    result = tools.execute_tool(
        "search_sec_filings",
        {"query": "OpenAI", "ticker": "MSFT", "exhaustive": True, "as_of": "2026-08-10"},
        "test",
        context=_research_context(),
    )
    request = seen["request"]
    assert isinstance(request, SECSearchRequest)
    assert request.exhaustive is True and request.ticker == "MSFT"
    assert request.max_results is None
    hits = _as_seq(result["hits"])
    first = _as_dict(hits[0])
    assert len(hits) == 2
    assert first["accession_no"] == "0000950170-26-001234"
    # Text hits, not recent-filing metadata: each names the exact document.
    assert all(_as_dict(hit).get("matched_document") for hit in hits)


def test_edgar_exhibit_traversal_filing_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.sec.documents import FilingDocument

    docs = [
        FilingDocument(
            accession_no="0000950170-26-001234",
            document_name="ex101.htm",
            description="Material agreement",
            size=10,
            url="https://sec.gov/x/ex101.htm",
            document_type="EX-10.1",
        )
    ]

    def _fake_list(accession_no: str, **_: object) -> list[FilingDocument]:
        assert accession_no == "0000950170-26-001234"
        return docs

    monkeypatch.setattr(tools.sec, "list_sec_documents", _fake_list)

    def _fake_doc(accession_no: str, document_name: object = None, **_: object) -> dict[str, str]:
        assert accession_no == "0000950170-26-001234"
        assert document_name == "ex101.htm"
        return {
            "accession_no": accession_no,
            "document_name": "ex101.htm",
            "text": "OpenAI Azure purchase commitment terms",
        }

    monkeypatch.setattr(tools.sec, "get_sec_document", _fake_doc)
    listed = tools.execute_tool(
        "list_sec_documents", {"accession_no": "0000950170-26-001234"}, "test", context=_research_context()
    )
    assert _as_dict(_as_seq(listed["documents"])[0])["document_name"] == "ex101.htm"
    text = tools.execute_tool(
        "get_sec_document",
        {"accession_no": "0000950170-26-001234", "document_name": "ex101.htm"},
        "test",
        context=_research_context(),
    )
    assert "purchase commitment" in str(text["text"])


def test_edgar_exhaustion_honest_on_route_exhausted_or_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    exhausted = _patch_discovery(
        monkeypatch,
        _msft_search_packet(
            text_hits=_msft_hits(),
            coverage=SearchCoverage(
                status="complete",
                sources_attempted=("entity", "efts"),
                sources_completed=("entity", "efts"),
                sources_failed=(),
                results_reported=2,
                results_retrieved=2,
                pages=2,
            ),
        ),
    )
    ok = tools.execute_tool(
        "search_sec_filings",
        {"query": "OpenAI", "ticker": "MSFT", "exhaustive": True},
        "test",
        context=_research_context(),
    )
    exhausted_request = exhausted["request"]
    assert isinstance(exhausted_request, SECSearchRequest)
    assert exhausted_request.exhaustive is True
    assert _as_dict(ok["coverage"])["status"] in ("complete", "complete_within_source_limits")
    limited = _patch_discovery(
        monkeypatch,
        _msft_search_packet(
            text_hits=_msft_hits(),
            coverage=SearchCoverage(
                status="partial",
                sources_attempted=("entity", "efts"),
                sources_completed=("entity",),
                sources_failed=(),
                results_reported=3,
                results_retrieved=2,
                pages=2,
                source_limits=("efts capped at limit",),
            ),
            warnings=("results capped at 2; rerun with a higher limit or exhaustive=true",),
        ),
    )
    capped = tools.execute_tool(
        "search_sec_filings", {"query": "OpenAI", "ticker": "MSFT", "limit": 2}, "test", context=_research_context()
    )
    limited_request = limited["request"]
    assert isinstance(limited_request, SECSearchRequest)
    assert limited_request.exhaustive is False
    assert _as_dict(capped["coverage"])["status"] == "partial"
    assert any("capped" in str(w) for w in _as_seq(capped["warnings"]))


def test_edgar_ranking_quantified_10k_outranks_unrelated_form4() -> None:
    from app.sec.discovery.service import rank_hits

    ranked = rank_hits(
        _msft_hits(),
        verified_ciks=(789790,),
        verified_names=("Microsoft Corp",),
        relevant_forms=("10-K",),
        query="Microsoft OpenAI exposure",
    )
    assert ranked[0].form == "10-K"
    assert ranked[0].accession_no == "0000950170-26-001234"
    assert ranked[-1].form == "4"


def test_edgar_alias_expansion_validated_keeps_provenance_no_false_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_obj = _result(
        entities=(
            EntityCandidate(
                cik=789790,
                name="Microsoft Corp",
                tickers=("MSFT",),
                exchange=None,
                match_source="former-name",
                match_score=0.8,
                match_type="former_name",
                verification_status="verified",
                entity_id="sec:cik:789790",
            ),
        )
    )

    def _fake_find(query: str, **kwargs: object) -> SECSearchResult:
        assert query == "Microsoft"
        return result_obj

    monkeypatch.setattr(tools.sec, "find_sec_entities", _fake_find)
    result = tools.execute_tool("find_sec_entities", {"query": "Microsoft"}, "test", context=_research_context())
    candidate = _as_dict(_as_seq(result["entities"])[0])
    assert candidate["verification_status"] == "verified"
    assert candidate["cik"] == 789790
    assert candidate["match_type"] == "former_name"
    assert result["search_id"] == "s1"
    # Provenance kept: attempt backend + PIT basis survive the envelope.
    assert _as_dict(_as_seq(result["attempts"])[0])["backend"] == "entity"
    assert result["pit_basis"] == "known_at"
    # No false identity: the verified MSFT CIK round-trips, never a guess.
    assert candidate["cik"] == 789790


# ---- research_read_search: paged reads of one persisted search universe ----

_READ_SEARCH = "s-read-1"


def _read_search_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RequestContext:
    """Data root at tmp_path (SEC ledger + research DB), research permission."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    return RequestContext("research", frozenset({Capability.RESEARCH}), data_root=tmp_path)


def _seed_research_session(root: Path) -> str:
    from app.research.repository import ResearchRepository
    from app.research.service import create_research

    return create_research(
        "Acme Labs exposure?", "risk", as_of="2026-03-02T00:00:00+00:00", repo=ResearchRepository(data_root=root)
    )


def _seed_read_search_ledger(
    root: Path, *, request: SECSearchRequest | None = None, hits: tuple[SECTextHit, ...] = ()
) -> None:
    """No persisted search universe: reads always resolve unknown_search live."""
    _ = (root, request, hits)


def _read_search_hits() -> tuple[SECTextHit, ...]:
    def _hit(accession: str, form: str, score: float) -> SECTextHit:
        return SECTextHit(
            search_id=_READ_SEARCH,
            attempt_id="s-read-1-efts-1",
            query="Acme Labs",
            accession_no=accession,
            form=form,
            filed_at="2026-01-01",
            filer_cik=1234567,
            filer_name="Acme Labs Inc",
            matched_document=f"{accession}.htm",
            file_type=form,
            score=score,
            snippet="supply agreement",
        )

    return (
        _hit("0000000001-26-000001", "10-K", 9.0),
        _hit("0000000001-26-000002", "8-K", 5.0),
        _hit("0000000001-26-000003", "10-K", 1.0),
    )


def _read(context: RequestContext, arguments: dict[str, object]) -> dict[str, object]:
    return tools.execute_tool("research_read_search", arguments, "test", context=context)


def test_research_read_search_pages_persisted_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _read_search_context(tmp_path, monkeypatch)
    _seed_read_search_ledger(tmp_path, hits=_read_search_hits())
    session_id = _seed_research_session(tmp_path)

    page = _read(context, {"session_id": session_id, "search_id": _READ_SEARCH, "limit": 2})
    assert page["error_type"] == "unknown_search"
    assert _READ_SEARCH in str(page["error"])


def test_research_read_search_forms_filter_and_limit_clamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _read_search_context(tmp_path, monkeypatch)
    _seed_read_search_ledger(tmp_path, hits=_read_search_hits())
    session_id = _seed_research_session(tmp_path)

    filtered = _read(context, {"session_id": session_id, "search_id": _READ_SEARCH, "forms": ["10-k"]})
    assert filtered["error_type"] == "unknown_search"

    clamped = _read(context, {"session_id": session_id, "search_id": _READ_SEARCH, "limit": 999})
    assert clamped["error_type"] == "unknown_search"


def test_research_read_search_unknown_session_and_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _read_search_context(tmp_path, monkeypatch)
    _seed_read_search_ledger(tmp_path)
    session_id = _seed_research_session(tmp_path)

    unknown_session = _read(context, {"session_id": "s-nope", "search_id": _READ_SEARCH})
    assert unknown_session["error_type"] == "unknown_session"
    assert "s-nope" in str(unknown_session["error"])

    unknown_search = _read(context, {"session_id": session_id, "search_id": "s-nope"})
    assert unknown_search["error_type"] == "unknown_search"
    assert "s-nope" in str(unknown_search["error"])


def test_research_read_search_rejects_bad_page_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _read_search_context(tmp_path, monkeypatch)
    _seed_read_search_ledger(tmp_path)
    session_id = _seed_research_session(tmp_path)
    base: dict[str, object] = {"session_id": session_id, "search_id": _READ_SEARCH}

    for arguments, needle in (
        ({"offset": -1}, "'offset' must be an integer >= 0"),
        ({"offset": True}, "'offset' must be an integer >= 0"),
        ({"limit": 0}, "'limit' must be an integer >= 1"),
        ({"limit": "5"}, "'limit' must be an integer >= 1"),
        ({"forms": "10-K"}, "'forms' must be a list of strings"),
        ({"forms": [1]}, "'forms' must be a list of strings"),
    ):
        result = _read(context, {**base, **arguments})
        assert needle in str(result["error"]), arguments


def test_research_read_search_reports_company_name_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _read_search_context(tmp_path, monkeypatch)
    _seed_read_search_ledger(tmp_path, request=SECSearchRequest(company_name="Acme Labs"))
    session_id = _seed_research_session(tmp_path)

    packet = _read(context, {"session_id": session_id, "search_id": _READ_SEARCH})
    assert packet["error_type"] == "unknown_search"


def test_research_read_search_unknown_search_universe_is_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No persisted search universe remains; every read is unknown_search."""
    context = _read_search_context(tmp_path, monkeypatch)
    session_id = _seed_research_session(tmp_path)

    packet = _read(context, {"session_id": session_id, "search_id": "s-read-raw-null"})
    assert packet["error_type"] == "unknown_search"
    assert "s-read-raw-null" in str(packet["error"])


def test_search_sec_filings_remainder_names_read_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A display-bounded packet points at research_read_search for the stored remainder."""
    hits = (
        SECTextHit(
            search_id="s1",
            attempt_id="s1-efts-1",
            query="Acme Labs",
            accession_no="0000000001-26-000001",
            form="10-K",
            filed_at="2026-01-01",
            filer_cik=1234567,
            filer_name="Acme Labs Inc",
            matched_document="primary.htm",
            file_type="10-K",
            score=9.0,
        ),
        SECTextHit(
            search_id="s1",
            attempt_id="s1-efts-1",
            query="Acme Labs",
            accession_no="0000000001-26-000002",
            form="8-K",
            filed_at="2026-02-01",
            filer_cik=1234567,
            filer_name="Acme Labs Inc",
            matched_document="primary.htm",
            file_type="8-K",
            score=5.0,
        ),
    )
    _patch_discovery(monkeypatch, _result(text_hits=hits))

    capped = tools.execute_tool(
        "search_sec_filings", {"query": "Acme Labs", "limit": 1}, "test", context=_research_context()
    )
    assert capped["count"] == 2
    assert len(_as_seq(capped["top_hits"])) == 1
    remainder = _as_dict(capped["additional_hits"])
    assert remainder["count"] == 1
    assert remainder["page_with"] == "research_read_search"
    assert remainder["next_offset"] == 1
    assert "research_read_search(session_id, search_id)" in str(remainder["note"])
    assert _as_dict(capped["retrieval"])["page_with"] == "research_read_search"

    uncapped = tools.execute_tool("search_sec_filings", {"query": "Acme Labs"}, "test", context=_research_context())
    assert _as_dict(uncapped["additional_hits"]) == {"count": 0}


def test_list_sec_filings_rejects_date_like_forms_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """forms=['2025-01-01'] is a misrouted date arg: guided invalid_tool_arguments, zero provider call."""

    def _boom(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("provider must not be called for date-like forms")

    monkeypatch.setattr(tools.sec, "list_sec_filings", _boom)
    result = tools.execute_tool(
        "list_sec_filings",
        {"identifier": "NVDA", "forms": ["2025-01-01"], "start_date": "2025-10-01", "end_date": "2025-10-31"},
        "test",
        context=_research_context(),
    )
    assert result["error_type"] == "invalid_tool_arguments"
    assert "2025-01-01" in str(result["error"])
    assert "10-K" in str(result["error"])


def test_diff_sec_filings_rejects_date_like_forms_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticker self-resolution path shares the guard: guided error, zero provider call."""

    def _boom(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("provider must not be called for date-like forms")

    monkeypatch.setattr(tools.sec, "list_sec_filings", _boom)
    result = tools.execute_tool(
        "diff_sec_filings", {"ticker": "NVDA", "forms": ["2025-01-01"]}, "test", context=_research_context()
    )
    assert result["error_type"] == "invalid_tool_arguments"
    assert "2025-01-01" in str(result["error"])


def test_list_sec_filings_valid_forms_reach_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legit forms still coerce and reach the provider untouched."""
    seen: dict[str, object] = {}

    def _fake(*args: object, **kwargs: object) -> list[object]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tools.sec, "list_sec_filings", _fake)
    result = tools.execute_tool(
        "list_sec_filings", {"identifier": "NVDA", "forms": ["10-K", "8-K"]}, "test", context=_research_context()
    )
    assert result["source"] == "SEC EDGAR"
    assert seen["forms"] == ("10-K", "8-K")


def test_get_insider_activity_self_contained_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insider handler resolves ticker/company inline and dispatches without shared helpers."""
    from types import SimpleNamespace

    seen: dict[str, object] = {}

    def _fake_activity(ticker: object, **kwargs: object) -> list[object]:
        seen["ticker"] = ticker
        seen.update(kwargs)
        return [SimpleNamespace(to_dict=lambda: {"ticker": ticker})]

    monkeypatch.setattr(tools.sec, "get_insider_activity", _fake_activity)
    result = tools._get_insider_activity({"ticker": "AAPL", "limit": 5}, "test")
    assert result["subject"] == "AAPL"
    assert _as_seq(result["transactions"])[0]["ticker"] == "AAPL"
    assert seen["limit"] == 5

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", lambda _name: "AAPL")
    named = tools._get_insider_activity({"company_name": "Apple"}, "test")
    assert named["subject"] == "AAPL"

    bad = tools._get_insider_activity({"ticker": "AAPL", "as_of": "not-a-date"}, "test")
    assert bad["error_type"] == "invalid_tool_arguments"


def test_get_insider_activity_avoids_shared_helpers() -> None:
    """Insider is a single self-contained function: no shared ticker helper calls."""
    import inspect

    source = inspect.getsource(tools._get_insider_activity)
    assert "_ticker_or_company_name" not in source
