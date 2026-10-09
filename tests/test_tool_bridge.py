from __future__ import annotations

import pytest

from app.tool_runtime import AgentToolSession
from scripts import tool_bridge


def test_describe_exposes_canonical_research_tools() -> None:
    response = tool_bridge.handle({"id": "d1", "op": "describe"})
    tools = response.get("tools")
    assert isinstance(tools, list)
    names = {
        fn["name"]
        for item in tools
        if isinstance(item, dict) and isinstance((fn := item.get("function")), dict) and isinstance(fn.get("name"), str)
    }
    assert "search_web" in names
    assert "search_sec_filings" in names


def test_invoke_reuses_generic_session_and_end_drops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[AgentToolSession] = []

    def fake_execute(
        name: str,
        arguments: dict[str, object],
        session: AgentToolSession,
        **kwargs: object,
    ) -> dict[str, object]:
        seen.append(session)
        return {"content": f"{name}:ok"}

    monkeypatch.setattr(tool_bridge, "execute_agent_tool", fake_execute)
    tool_bridge._sessions.pop("generic-1", None)
    try:
        first = tool_bridge.handle(
            {
                "id": "i1",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA Anthropic"},
                "session_id": "generic-1",
            }
        )
        second = tool_bridge.handle(
            {
                "id": "i2",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA exposure"},
                "session_id": "generic-1",
            }
        )
        assert first["result"] == {"content": "search_web:ok"}
        assert second["result"] == {"content": "search_web:ok"}
        assert len(seen) == 2
        assert seen[0] is seen[1]

        ended = tool_bridge.handle({"id": "e1", "op": "tool.session.end", "session_id": "generic-1"})
        assert ended == {"id": "e1", "result": {"ended": True}}

        tool_bridge.handle(
            {
                "id": "i3",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA"},
                "session_id": "generic-1",
            }
        )
        assert seen[2] is not seen[0]
    finally:
        tool_bridge._sessions.pop("generic-1", None)


def test_invoke_requires_explicit_session() -> None:
    assert tool_bridge.handle({"id": "bad", "op": "tool.invoke", "name": "search_web", "arguments": {}}) == {
        "id": "bad",
        "error": "missing_arg",
    }


def test_invoke_failure_reports_bridge_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(
        name: str,
        arguments: dict[str, object],
        session: AgentToolSession,
        **kwargs: object,
    ) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(tool_bridge, "execute_agent_tool", _boom)
    tool_bridge._sessions.pop("boom-1", None)
    try:
        assert tool_bridge.handle(
            {
                "id": "f1",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA"},
                "session_id": "boom-1",
            }
        ) == {"id": "f1", "error": "bridge_failed"}
    finally:
        tool_bridge._sessions.pop("boom-1", None)
