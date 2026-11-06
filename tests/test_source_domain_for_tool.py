"""source_domain_for_tool: every replayable tool lands in a persisting domain."""

from app.research.agents.source_agent import FINRA_TOOLS, source_domain_for_tool
from app.research.service import _FINRA_EVIDENCE_TOOLS


def test_web_tool_maps_web() -> None:
    assert source_domain_for_tool("search_web") == "WEB"


def test_every_finra_evidence_tool_maps_finra() -> None:
    assert _FINRA_EVIDENCE_TOOLS <= FINRA_TOOLS
    for tool in _FINRA_EVIDENCE_TOOLS:
        assert source_domain_for_tool(tool) == "FINRA"


def test_sec_tool_maps_sec() -> None:
    assert source_domain_for_tool("get_sec_document") == "SEC"


def test_every_alt_and_macro_tool_maps_its_lane() -> None:
    """Alternative-data and macro tools own their lane, so their results persist and admit."""
    from app.research.agents.source_agent import ALT_TOOLS, MACRO_TOOLS

    assert ALT_TOOLS and MACRO_TOOLS
    for tool in ALT_TOOLS:
        assert source_domain_for_tool(tool) == "ALT"
    for tool in MACRO_TOOLS:
        assert source_domain_for_tool(tool) == "MACRO"


def test_unknown_maps_other_not_sec() -> None:
    assert source_domain_for_tool("not_a_tool") == "OTHER"
    assert source_domain_for_tool("draft_thesis") == "OTHER"
    assert source_domain_for_tool("run_macro_analysis") == "OTHER"
