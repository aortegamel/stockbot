"""Registration and dispatch tests for the search_web tool."""

import pytest

from app import tools
from app.policy import Capability, RequestContext

RESEARCH_CONTEXT = RequestContext("test", frozenset({Capability.RESEARCH}))

_APPROVED_SEARCH_TYPES = {"auto", "fast", "deep-lite"}
_APPROVED_CATEGORIES = {"news", "company", "publication", "financial report"}


def _search_web_schema():
    for entry in tools.TOOLS:
        fn = entry.get("function")
        assert isinstance(fn, dict)
        if fn.get("name") == "search_web":
            return fn
    raise AssertionError("search_web not registered")


def test_search_web_registered_everywhere() -> None:
    names: set[object] = set()
    for entry in tools.TOOLS:
        fn = entry.get("function")
        assert isinstance(fn, dict)
        names.add(fn.get("name"))
    assert "search_web" in names
    assert "search_web" in tools._DIRECT_HANDLERS
    assert tools.TOOL_CAPABILITIES["search_web"] == Capability.RESEARCH


def test_search_web_schema_shape() -> None:
    schema = _search_web_schema()
    params = schema["parameters"]
    assert params["required"] == ["query"]
    props = params["properties"]
    assert props["query"]["type"] == "string"
    assert set(props["category"]["enum"]) == _APPROVED_CATEGORIES
    assert set(props["search_type"]["enum"]) == _APPROVED_SEARCH_TYPES
    assert props["include_domains"]["type"] == "array"
    assert props["exclude_domains"]["type"] == "array"
    assert "YYYY-MM-DD" in props["start_published_date"]["description"]
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 25
    # Optional fields are plain types absent from `required` (repo style).
    for key in ("category", "search_type", "limit", "include_domains"):
        assert key not in params["required"]


def test_search_web_dispatcher_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_search(query: str, **kwargs: object) -> dict[str, object]:
        calls.append((query, kwargs))
        return {"result_type": "web_search", "query": query, "evidence": list[dict[str, object]]()}

    monkeypatch.setattr(tools.exa_client, "search", fake_search)
    result = tools.execute_tool("search_web", {"query": "AMD"}, model="test", context=RESEARCH_CONTEXT)
    assert result["result_type"] == "web_search"
    query, kwargs = calls[0]
    assert query == "AMD"
    assert kwargs == {
        "category": None,
        "include_domains": None,
        "exclude_domains": None,
        "start_published_date": None,
        "end_published_date": None,
        "search_type": "auto",
        "limit": 5,
    }


def test_search_web_dispatcher_passes_optional_args(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_search(query: str, **kwargs: object) -> dict[str, object]:
        calls.append((query, kwargs))
        return {"result_type": "web_search", "query": query, "evidence": list[dict[str, object]]()}

    monkeypatch.setattr(tools.exa_client, "search", fake_search)
    tools.execute_tool(
        "search_web",
        {
            "query": "AMD competition",
            "category": "news",
            "search_type": "fast",
            "limit": 3,
            "include_domains": ["amd.com"],
            "start_published_date": "2026-07-01",
        },
        model="test",
        context=RESEARCH_CONTEXT,
    )
    query, kwargs = calls[0]
    assert query == "AMD competition"
    assert kwargs["category"] == "news"
    assert kwargs["search_type"] == "fast"
    assert kwargs["limit"] == 3
    assert kwargs["include_domains"] == ["amd.com"]
    assert kwargs["start_published_date"] == "2026-07-01"


def test_search_web_disabled_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_ENABLED", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    result = tools.execute_tool("search_web", {"query": "AMD news"}, model="test", context=RESEARCH_CONTEXT)
    assert result["error"] == "Exa search unavailable"
    assert result["source"] == "exa"
    assert result["soft"] is True


def test_search_web_invalid_args_rejected_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema gate rejects bad enums before any Exa call (invalid_tool_arguments, not soft)."""
    monkeypatch.setenv("EXA_ENABLED", "true")
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    def _boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("provider must not be called for invalid args")

    monkeypatch.setattr(tools.exa_client, "search", _boom)
    result = tools.execute_tool(
        "search_web",
        {"query": "AMD news", "category": "gossip"},
        model="test",
        context=RESEARCH_CONTEXT,
    )
    assert result["error_type"] == "invalid_tool_arguments"
    assert "'category' 'gossip'" in str(result["error"])
    assert "is not one of" in str(result["error"])
    assert result.get("soft") is not True

    result = tools.execute_tool(
        "search_web",
        {"query": "AMD news", "search_type": "deep"},
        model="test",
        context=RESEARCH_CONTEXT,
    )
    assert result["error_type"] == "invalid_tool_arguments"
    assert "'search_type' 'deep'" in str(result["error"])
    assert "is not one of" in str(result["error"])
    assert result.get("soft") is not True
