"""Grounded hypothetical reasoning: refs, numbers, assumptions, prompt bytes."""

from types import SimpleNamespace

import pytest

from app.reasoner_client import check_analyses
from app.research import grounding
from app.research.grounding import (
    check_grounded_analysis,
    compute_scenario_impact,
    format_grounded_block,
    split_grounded_context,
)
from app.research.scheduler import _analyze_prompt

# Exact tool bytes the live run must reason on (verbatim, never summarized).
NVDA_EXPOSURE = "NVDA OpenAI exposure 12.5% per 10-K accession 0001045810-26-000001"
ASSUMPTION_TEXT = "OpenAI IPO does terrible: apply 30% haircut to OpenAI-linked exposure"


def _node():
    return SimpleNamespace(node_id="n1", session_id="s1", question="q?", why_it_matters="w")


def _evidence():
    return [{"evidence_id": "ev-nvda-10k", "content": NVDA_EXPOSURE, "content_hash": "h1", "provenance": "SEC"}]


def _session():
    return {"session_id": "s1", "objective": "If OpenAI IPO does terrible, how does NVDA move?"}


def test_bad_evidence_ref_rejected():
    """Unknown evidenceRefs fail closed (caller drops, never accepts)."""
    bad = [{"nodeId": "n1", "objectiveId": "o1", "interpretation": "i", "evidenceRefs": ["ev-nope"]}]
    with pytest.raises(ValueError):
        check_analyses("analyze", bad, "o1", {"n1"}, {"ev-nvda-10k"})
    with pytest.raises(ValueError):
        check_grounded_analysis({"evidenceRefs": ["ev-nope"]}, {"ev-nvda-10k"})


def test_numbers_without_matching_ref_rejected():
    """numbers[].evidenceId must itself sit in evidenceRefs, with a verbatim quote."""
    base = {"nodeId": "n1", "objectiveId": "o1", "interpretation": "i", "evidenceRefs": ["ev-nvda-10k"]}
    orphan = dict(base, numbers=[{"value": "12.5%", "evidenceId": "ev-other", "quote": NVDA_EXPOSURE}])
    with pytest.raises(ValueError):
        check_analyses("analyze", [orphan], "o1", {"n1"}, {"ev-nvda-10k", "ev-other"})
    empty_quote = dict(base, numbers=[{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": ""}])
    with pytest.raises(ValueError):
        check_analyses("analyze", [empty_quote], "o1", {"n1"}, {"ev-nvda-10k"})
    with pytest.raises(ValueError):
        check_grounded_analysis(
            dict(base, numbers=[{"value": "x", "evidenceId": "ev-other", "quote": "q"}]), {"ev-nvda-10k", "ev-other"}
        )


def test_nonverbatim_quote_rejected_when_evidence_supplied():
    """A quote not found in the cited evidence content fails closed (paraphrase is not grounding)."""
    base = {"nodeId": "n1", "objectiveId": "o1", "interpretation": "i", "evidenceRefs": ["ev-nvda-10k"]}
    paraphrase = dict(
        base, numbers=[{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": "twelve point five percent"}]
    )
    with pytest.raises(ValueError):
        check_grounded_analysis(paraphrase, {"ev-nvda-10k"}, _evidence())
    # Without evidence bytes the gate only checks non-empty (back-compat for id-only callers).
    assert check_grounded_analysis(paraphrase, {"ev-nvda-10k"}) is paraphrase


def test_attach_scenario_impact_exact_and_fail_open():
    """First number x first assumption percents attach computed_impact; unparseable rides unchanged."""
    from app.research.grounding import attach_scenario_impact, parse_percent

    assert parse_percent("12.5% of revenue") == 12.5
    assert parse_percent("no number here") is None
    fake = {
        "nodeId": "n1",
        "objectiveId": "s1",
        "interpretation": "i",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}],
        "assumptions": [{"assumptionId": "openai-ipo-terrible", "text": ASSUMPTION_TEXT}],
    }
    out = attach_scenario_impact(fake)
    assert out["computed_impact"] == {
        "exposure_pct": 12.5,
        "haircut_pct": 30.0,
        "impact_pct": 3.75,
        "exposure_evidence": "ev-nvda-10k",
        "assumption_id": "openai-ipo-terrible",
    }
    no_pair = {"nodeId": "n1", "interpretation": "i", "evidenceRefs": []}
    assert attach_scenario_impact(no_pair) is no_pair
    unparseable = {
        "nodeId": "n1",
        "interpretation": "i",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "big", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}],
        "assumptions": [{"assumptionId": "a", "text": "no percent here"}],
    }
    assert attach_scenario_impact(unparseable) is unparseable


def test_reason_path_attaches_impact_and_persists_refs():
    """Offline reason path: kept analysis carries computed_impact; drop gate + adjudication refs persist."""
    import asyncio
    from types import SimpleNamespace

    from app.research import scheduler as sched

    recorded: list[dict[str, object]] = []

    class _K:
        def record_decision(self, sid: str, dtype: str, **kw: object) -> None:
            recorded.append({"dtype": dtype, **kw})

    good = {
        "nodeId": "n1",
        "objectiveId": "s1",
        "interpretation": "NVDA 12.5% exposure x 30% haircut; formula in words, code computes",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}],
        "assumptions": [{"assumptionId": "openai-ipo-terrible", "text": ASSUMPTION_TEXT}],
    }
    bad = {
        "nodeId": "n1",
        "objectiveId": "s1",
        "interpretation": "paraphrased number with no bytes",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "9%", "evidenceId": "ev-nvda-10k", "quote": "nine percent somewhere"}],
    }

    class _R:
        async def analyze(self, prompt: str) -> dict[str, object]:
            assert NVDA_EXPOSURE in prompt  # exact bytes in, not a summary
            return {"analyses": [bad, good], "evidenceRequests": []}

    class _J:
        async def adjudicate(self, analysis: object, node: object, **kw: object) -> SimpleNamespace:
            assert isinstance(analysis, dict)
            kept = analysis["analyses"]
            assert len(kept) == 1 and kept[0]["evidenceRefs"] == ["ev-nvda-10k"]
            assert kept[0]["computed_impact"]["impact_pct"] == 3.75
            return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)

    out = asyncio.run(sched._reasoner_analyze(_R(), _session(), _node(), _evidence(), []))
    assert len(out["analyses"]) == 1  # paraphrase dropped before adjudication
    verdict = asyncio.run(sched._adjudicate_analysis(_J(), _K(), out, _node(), "s1", "n1"))
    assert verdict.tool_name == "query_finra"
    assert recorded and recorded[0]["dtype"] == "reason_adjudication"
    selected = recorded[0]["selected"]
    assert isinstance(selected, dict)
    assert selected["evidence_refs"] == ["ev-nvda-10k"]
    assert selected["computed_impact"]["impact_pct"] == 3.75
    assert selected["computed_impact"]["exposure_evidence"] == "ev-nvda-10k"
    assert selected["computed_impact"]["assumption_id"] == "openai-ipo-terrible"


def test_assumption_hypothetical_needs_no_evidence_ref():
    """Invented assumptions carry assumptionId; the assumption itself needs no evidence."""
    no_refs: list[str] = []
    analysis: dict[str, object] = {
        "nodeId": "n1",
        "objectiveId": "o1",
        "interpretation": "terrible IPO -> haircut; formula in words, code computes",
        "evidenceRefs": no_refs,
        "assumptions": [{"assumptionId": "openai-ipo-terrible", "text": ASSUMPTION_TEXT}],
    }
    out = check_analyses("analyze", [analysis], "o1", {"n1"}, set())
    assert out[0]["assumptions"] == [{"assumptionId": "openai-ipo-terrible", "text": ASSUMPTION_TEXT}]
    assert check_grounded_analysis(analysis, set()) is analysis
    dup = dict(analysis, assumptions=[{"assumptionId": "a", "text": "x"}, {"assumptionId": "a", "text": "y"}])
    with pytest.raises(ValueError):
        check_analyses("analyze", [dup], "o1", {"n1"}, set())


def test_compute_scenario_impact_exact():
    """Code multiplies exposure x haircut; the model never does arithmetic."""
    assert compute_scenario_impact(12.5, 30.0) == 3.75
    assert compute_scenario_impact(12.5, 30) == pytest.approx(3.75)
    for bad in ("12.5", None, True, [12.5]):
        with pytest.raises(ValueError):
            compute_scenario_impact(bad, 30.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            compute_scenario_impact(12.5, bad)  # type: ignore[arg-type]


def test_analyze_prompt_grounded_assumptions_sections():
    """Prompt carries GROUNDED exact bytes + ASSUMPTIONS + numbers/assumptions shape."""
    attempts = [{"job_id": "job-0", "assumption": ASSUMPTION_TEXT}]
    prompt = _analyze_prompt(_session(), _node(), _evidence(), attempts)
    assert "GROUNDED" in prompt
    assert "ASSUMPTIONS" in prompt
    assert "numbers" in prompt
    assert "assumptions" in prompt
    assert "assumptionId" in prompt
    assert "evidenceRefs" in prompt


def test_verbatim_bytes_in_prompt_context():
    """Exact evidence content appears verbatim in the prompt context (not a summary)."""
    prompt = _analyze_prompt(_session(), _node(), _evidence(), [])
    assert NVDA_EXPOSURE in prompt
    assert NVDA_EXPOSURE in format_grounded_block(_evidence()[0])
    split = split_grounded_context(_evidence(), [])
    assert split["grounded"] == _evidence()
    # End-to-end: fake reasoner cites the exact bytes -> check path accepts.
    grounded_ids = {str(r.get("evidence_id")) for r in split["grounded"] if isinstance(r, dict)}
    fake: dict[str, object] = {
        "nodeId": "n1",
        "objectiveId": "s1",
        "interpretation": "NVDA 12.5% exposure x 30% haircut; formula in words, code computes",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}],
        "assumptions": [{"assumptionId": "openai-ipo-terrible", "text": ASSUMPTION_TEXT}],
    }
    assert grounding.check_grounded_analysis(fake, grounded_ids) is fake
    out = check_analyses("analyze", [fake], "s1", {"n1"}, grounded_ids)
    assert out[0]["evidenceRefs"] == ["ev-nvda-10k"]
