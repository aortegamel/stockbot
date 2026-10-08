"""Global static registry gate: every schema has handler/capability/domain/envelope."""

from collections.abc import Mapping

import pytest

from app import tools
from app.security.action_policy import TOOL_DOMAINS
from app.security.context_gateway import TOOL_ENVELOPES


def _schema_functions() -> dict[str, Mapping[str, object]]:
    functions: dict[str, Mapping[str, object]] = {}
    for tool in tools.TOOLS:
        function = tool.get("function")
        assert isinstance(function, dict)
        name = function.get("name")
        assert isinstance(name, str)
        functions[name] = function
    return functions


def test_tool_registry_five_way_parity():
    schemas = set(_schema_functions())
    handlers = (
        set(tools._DIRECT_HANDLERS) | set(tools._FINRA_HANDLERS) | set(tools._ROBINHOOD_HANDLERS) | {"call_tool"}
    )  # gateway dispatch is call_tool's canonical handler; it has no _MODEL_HANDLERS entry by design
    assert schemas == handlers, f"Schemas without handlers: {sorted(schemas - handlers)}"
    assert schemas == set(tools.TOOL_CAPABILITIES), (
        f"Missing capability: {sorted(schemas - set(tools.TOOL_CAPABILITIES))}"
    )
    assert schemas == set(TOOL_DOMAINS), f"Missing security domain: {sorted(schemas - set(TOOL_DOMAINS))}"
    assert schemas == set(TOOL_ENVELOPES), f"Missing context envelope: {sorted(schemas - set(TOOL_ENVELOPES))}"


THESIS_TOOLS = frozenset(
    {"thesis_create", "thesis_show", "thesis_refine", "thesis_watch", "thesis_journal", "thesis_status"}
)


def test_thesis_domains_and_structured_schemas():
    assert {TOOL_DOMAINS[name] for name in THESIS_TOOLS} == {"financial_research"}

    schemas = _schema_functions()
    create = schemas["thesis_create"].get("parameters")
    assert isinstance(create, dict)
    assert create["required"] == ["user_thesis"]
    assert "user_thesis" in create["properties"]
    assert {"scope", "claims", "questions"} <= set(create["properties"])
    assert not ({"idea", "answers", "offline"} & set(create["properties"]))
    refine = schemas["thesis_refine"].get("parameters")
    assert isinstance(refine, dict)
    assert set(refine["required"]) == {"id", "clarification"}
    assert {"scope", "claims", "questions"} <= set(refine["properties"])
    assert not ({"idea", "answers", "offline"} & set(refine["properties"]))


def test_routing_cards_have_exact_shape():
    for name in tools.TOOL_DISCOVERY_REGISTRY:
        card = tools._routing_card(name)
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
        }, name
        assert card["name"] == name
        assert "parameters" not in card


def test_routing_card_required_optional_derive_from_schema():
    for name in tools.TOOL_DISCOVERY_REGISTRY:
        card = tools._routing_card(name)
        _, required, optional = tools._canonical_tool_schema(name)
        assert card["required"] == required, name
        assert card["optional"] == optional, name


def test_discovery_coordinates_complete():
    import re

    kebab = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    snake = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    assert len(tools.TOOL_DISCOVERY_REGISTRY) == 61
    assert "research_read_search" in tools.TOOL_DISCOVERY_REGISTRY
    for name, meta in tools.TOOL_DISCOVERY_REGISTRY.items():
        assert meta.domain and kebab.match(meta.domain), name
        assert meta.family and kebab.match(meta.family), name
        assert meta.source and kebab.match(meta.source), name
        assert meta.intent and snake.match(meta.intent), name
        assert meta.output_kind and snake.match(meta.output_kind), name
        assert meta.entity_scope and snake.match(meta.entity_scope), name
        assert meta.time_mode and snake.match(meta.time_mode), name
        assert meta.summary and len(meta.summary) <= 200, name
        assert meta.choose_when and meta.reject_when, name
    # filings renamed to sec; pressure profile lives under finra
    assert "filings" not in {m.domain for m in tools.TOOL_DISCOVERY_REGISTRY.values()}
    assert tools.TOOL_DISCOVERY_REGISTRY["get_short_pressure_profile"].domain == "finra"
    assert {m.domain for m in tools.TOOL_DISCOVERY_REGISTRY.values()} <= set(tools.DOMAIN_DESCRIPTIONS)


def test_conflicts_reciprocal_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = tools.TOOL_DISCOVERY_REGISTRY
    assert reg["get_short_interest"].conflicts_with
    for name, meta in reg.items():
        assert name not in meta.conflicts_with, name
        assert len(set(meta.conflicts_with)) == len(meta.conflicts_with), name
        for peer in meta.conflicts_with:
            assert peer in reg, (name, peer)
            assert name in reg[peer].conflicts_with, (name, peer)
            assert peer in " ".join(meta.reject_when), (name, peer)
    # one-way probe must fail validation
    import copy

    import app.tools as mod

    probe: dict[str, tools.ToolDiscovery] = {k: copy.deepcopy(v) for k, v in reg.items()}
    victim = probe["get_fundamentals"]
    object.__setattr__(victim, "conflicts_with", tuple(c for c in victim.conflicts_with if c != "get_xbrl_facts"))
    monkeypatch.setattr(mod, "TOOL_DISCOVERY_REGISTRY", probe)
    try:
        mod.validate_tool_discovery_registry()
    except AssertionError as exc:
        assert "get_fundamentals" in str(exc) and "get_xbrl_facts" in str(exc)
    else:
        raise AssertionError("one-way conflict should fail")
    # unknown peer must fail with both names
    probe2: dict[str, tools.ToolDiscovery] = {k: copy.deepcopy(v) for k, v in reg.items()}
    object.__setattr__(
        probe2["get_fundamentals"], "conflicts_with", (*probe2["get_fundamentals"].conflicts_with, "no_such_tool")
    )
    monkeypatch.setattr(mod, "TOOL_DISCOVERY_REGISTRY", probe2)
    try:
        mod.validate_tool_discovery_registry()
    except AssertionError as exc:
        assert "get_fundamentals" in str(exc) and "no_such_tool" in str(exc)
    else:
        raise AssertionError("unknown conflict should fail")
    # self conflict must fail
    probe3: dict[str, tools.ToolDiscovery] = {k: copy.deepcopy(v) for k, v in reg.items()}
    object.__setattr__(
        probe3["get_fundamentals"], "conflicts_with", (*probe3["get_fundamentals"].conflicts_with, "get_fundamentals")
    )
    monkeypatch.setattr(mod, "TOOL_DISCOVERY_REGISTRY", probe3)
    try:
        mod.validate_tool_discovery_registry()
    except AssertionError as exc:
        assert "get_fundamentals" in str(exc)
    else:
        raise AssertionError("self conflict should fail")


def test_registry_version_covers_routing_metadata():
    import hashlib
    import json

    v1 = tools.TOOL_REGISTRY_VERSION
    assert isinstance(v1, str) and len(v1) == 12
    # changing routing metadata must change the version (schemas alone are not enough)
    import copy

    import app.tools as mod

    probe = {k: copy.deepcopy(v) for k, v in mod.TOOL_DISCOVERY_REGISTRY.items()}
    object.__setattr__(probe["get_short_interest"], "intent", "mutated_intent")
    fp = json.dumps({n: [m.domain, m.family, m.intent] for n, m in sorted(probe.items())}, sort_keys=True)
    assert hashlib.sha256((json.dumps(mod.TOOLS, sort_keys=True) + fp).encode()).hexdigest()[:12] != v1
