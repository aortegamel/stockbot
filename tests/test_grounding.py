"""Grounded hypothetical reasoning: refs, numbers, assumptions, prompt bytes."""

import json
from collections.abc import Callable

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from app.reasoner_client import ReasonerClient, check_analyses, check_evidence_requests
from app.research import grounding
from app.research.grounding import check_grounded_analysis, compute_scenario_impact, split_grounded_context
from app.research.models import JSONValue

# Exact tool bytes the live run must reason on (verbatim, never summarized).
NVDA_EXPOSURE = "NVDA OpenAI exposure 12.5% per 10-K accession 0001045810-26-000001"
ASSUMPTION_TEXT = "OpenAI IPO does terrible: apply 30% haircut to OpenAI-linked exposure"


def _evidence() -> list[JSONValue]:
    return [{"evidence_id": "ev-nvda-10k", "content": NVDA_EXPOSURE, "content_hash": "h1", "provenance": "SEC"}]


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
    base: dict[str, object] = {
        "nodeId": "n1",
        "objectiveId": "o1",
        "interpretation": "i",
        "evidenceRefs": ["ev-nvda-10k"],
    }
    paraphrase = dict(base, numbers=[{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": "exposure of 12.5%"}])
    with pytest.raises(ValueError, match="not found in evidence content"):
        check_grounded_analysis(paraphrase, {"ev-nvda-10k"}, _evidence())
    # Without evidence bytes the gate fails closed: no id-only publication path remains.
    with pytest.raises(ValueError, match="ERR_SOURCE_BYTES_ABSENT"):
        check_grounded_analysis(paraphrase, {"ev-nvda-10k"})
    # A real quote next to a fabricated number is still a fabricated number.
    fabricated = dict(base, numbers=[{"value": "13.5%", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}])
    with pytest.raises(ValueError, match=r"value '13\.5%' not in quote"):
        check_grounded_analysis(fabricated, {"ev-nvda-10k"}, _evidence())
    verbatim = dict(base, numbers=[{"value": "12.5%", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}])
    assert check_grounded_analysis(verbatim, {"ev-nvda-10k"}, _evidence()) is verbatim


def test_attach_scenario_impact_exact_and_fail_open():
    """First number x first assumption percents attach computed_impact; unparseable rides unchanged."""
    from app.research.grounding import attach_scenario_impact, parse_percent

    assert parse_percent("12.5% of revenue") == 12.5
    assert parse_percent("no number here") is None
    fake: dict[str, object] = {
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
    no_refs: list[str] = []
    no_pair: dict[str, object] = {"nodeId": "n1", "interpretation": "i", "evidenceRefs": no_refs}
    assert attach_scenario_impact(no_pair) is no_pair
    unparseable: dict[str, object] = {
        "nodeId": "n1",
        "interpretation": "i",
        "evidenceRefs": ["ev-nvda-10k"],
        "numbers": [{"value": "big", "evidenceId": "ev-nvda-10k", "quote": NVDA_EXPOSURE}],
        "assumptions": [{"assumptionId": "a", "text": "no percent here"}],
    }
    assert attach_scenario_impact(unparseable) is unparseable


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
    # Malformed inputs sit outside the float signature on purpose; the untyped alias keeps the checker green.
    compute_untyped: Callable[..., object] = compute_scenario_impact
    for bad in ("12.5", None, True, [12.5]):
        with pytest.raises(ValueError):
            compute_untyped(bad, 30.0)
        with pytest.raises(ValueError):
            compute_untyped(12.5, bad)


def test_fabricated_identifier_and_quote_rejected() -> None:
    """Needle must cite admitted ids with verbatim quotes; invented ids/quotes fail closed."""
    record: JSONValue = {
        "evidence_id": "ev-1",
        "content": "Total revenue $10",
        "content_hash": "h",
        "provenance": "SEC",
    }
    forged_id: dict[str, object] = {"evidenceRefs": ["ev-9"], "numbers": []}
    with pytest.raises(ValueError):
        check_grounded_analysis(forged_id, {"ev-1"}, [record])
    paraphrase: dict[str, object] = {
        "evidenceRefs": ["ev-1"],
        "numbers": [{"value": "$10", "evidenceId": "ev-1", "quote": "ten dollars somewhere"}],
    }
    with pytest.raises(ValueError):
        check_grounded_analysis(paraphrase, {"ev-1"}, [record])


def test_split_assumptions_preserves_list_precedence_and_stable_ids() -> None:
    evidence = ({"evidence_id": "ev-1", "content": "exact\nbytes"},)
    attempts = [
        None,
        {
            "assumption": "ignored single",
            "assumptions": [
                "  bare text  ",
                {"assumptionId": " declared ", "text": "  explicit text "},
                "",
                {"assumptionId": "", "text": "missing id"},
                {"assumptionId": "a", "text": " "},
                5,
            ],
        },
        {"assumptionId": " chosen ", "job_id": "unused", "assumption": " single "},
        {"job_id": " job-7 ", "assumption": " from job "},
        {"assumption": " positional "},
        {"assumption": " "},
    ]
    assert split_grounded_context(evidence, attempts) == {
        "grounded": list(evidence),
        "assumptions": [
            {"assumptionId": "attempt-1-0", "text": "bare text"},
            {"assumptionId": "declared", "text": "explicit text"},
            {"assumptionId": "chosen", "text": "single"},
            {"assumptionId": "job-7-assumption", "text": "from job"},
            {"assumptionId": "attempt-4-assumption", "text": "positional"},
        ],
    }
    assert split_grounded_context(None, None) == {"grounded": [], "assumptions": []}


@pytest.mark.parametrize("key", ["evidenceRefs", "numbers", "assumptions"])
def test_grounded_list_fields_accept_none_and_reject_non_lists(key: str) -> None:
    analysis: dict[str, object] = {key: None}
    assert check_grounded_analysis(analysis, set()) is analysis
    analysis[key] = "not a list"
    with pytest.raises(ValueError, match=f"analysis {key} must be a list"):
        check_grounded_analysis(analysis, set())


@pytest.mark.parametrize("ref", ["", None, 2])
def test_grounded_references_require_nonempty_strings(ref: object) -> None:
    with pytest.raises(ValueError, match="evidenceRefs must be non-empty strings"):
        check_grounded_analysis({"evidenceRefs": [ref]}, {"ev-1"})


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        (None, r"numbers\[\] entries must be mappings"),
        ({"value": "", "evidenceId": "ev-1", "quote": "source"}, r"numbers\[\] value must be a non-empty string"),
        ({"value": 12, "evidenceId": "ev-1", "quote": "source"}, r"numbers\[\] value must be a non-empty string"),
        ({"value": "12%", "evidenceId": "ev-1", "quote": " "}, r"numbers\[\] quote must be a non-empty string"),
        ({"value": "12%", "evidenceId": "ev-1", "quote": None}, r"numbers\[\] quote must be a non-empty string"),
    ],
)
def test_grounded_numbers_reject_malformed_entries(entry: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        check_grounded_analysis({"evidenceRefs": ["ev-1"], "numbers": [entry]}, {"ev-1"})


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        (None, r"assumptions\[\] entries must be mappings"),
        ({"assumptionId": "", "text": "scenario"}, "assumptionId must be a non-empty string"),
        ({"assumptionId": 1, "text": "scenario"}, "assumptionId must be a non-empty string"),
        ({"assumptionId": "a", "text": " "}, r"assumptions\[\] text must be a non-empty string"),
        ({"assumptionId": "a", "text": None}, r"assumptions\[\] text must be a non-empty string"),
    ],
)
def test_grounded_assumptions_reject_malformed_entries(entry: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        check_grounded_analysis({"assumptions": [entry]}, set())


def test_grounding_keeps_validation_precedence_and_missing_source_contract() -> None:
    number = {"value": "12%", "evidenceId": "ev-1", "quote": "source 12%"}
    analysis: dict[str, object] = {"evidenceRefs": ["ev-1", "ev-1"], "numbers": [number]}
    stored: list[JSONValue] = [{"evidence_id": "ev-1", "content": "the source 12% row"}]
    assert check_grounded_analysis(analysis, {"ev-1"}, stored) is analysis
    # Missing bytes fail closed: no evidence list, or a cited row without stored content.
    absent_cases: list[list[JSONValue]] = [[], [{"evidence_id": "ev-1"}]]
    for absent in absent_cases:
        with pytest.raises(ValueError, match="ERR_SOURCE_BYTES_ABSENT"):
            check_grounded_analysis(analysis, {"ev-1"}, absent)
    # The refs gate runs first, so admit "ev-1" to reach the numbers-before-assumptions precedence.
    with pytest.raises(ValueError, match="analysis numbers must be a list"):
        check_grounded_analysis(dict(analysis, numbers="bad", assumptions="bad"), {"ev-1"})
    unchecked: Callable[..., object] = check_grounded_analysis
    with pytest.raises(ValueError, match="analysis must be a mapping"):
        unchecked([], set())
    # Unsupported containers admit nothing. Valid containers ignore non-string members.
    with pytest.raises(ValueError, match="unknown evidence"):
        unchecked(analysis, "ev-1")
    assert unchecked(analysis, ["ev-1", None, 5], stored) is analysis


@pytest.mark.parametrize(
    "analysis",
    [
        {"numbers": [], "assumptions": []},
        {"numbers": [None], "assumptions": [{}]},
        {"numbers": [{}], "assumptions": [None]},
        {"numbers": [{"value": "12%"}], "assumptions": [{"text": "no percentage"}]},
        {"numbers": [{"value": "no percentage"}], "assumptions": [{"text": "30%"}]},
    ],
)
def test_scenario_impact_leaves_invalid_pairs_unchanged(analysis: dict[str, object]) -> None:
    assert grounding.attach_scenario_impact(analysis) is analysis


def test_scenario_impact_uses_first_pair_and_keeps_missing_ids_null() -> None:
    analysis: dict[str, object] = {
        "numbers": [{"value": "12.5%", "evidenceId": 9}, {"value": "99%"}],
        "assumptions": [{"text": "-30%", "assumptionId": 7}, {"text": "99%"}],
    }
    assert grounding.attach_scenario_impact(analysis)["computed_impact"] == {
        "exposure_pct": 12.5,
        "haircut_pct": -30.0,
        "impact_pct": -3.75,
        "exposure_evidence": None,
        "assumption_id": None,
    }
    unchecked: Callable[..., object] = grounding.attach_scenario_impact
    invalid: object = []
    assert unchecked(invalid) is invalid


@settings(max_examples=100, derandomize=True)
@given(st.lists(st.text(max_size=40), max_size=12))
def test_prop_bare_assumptions_keep_order_and_positional_identity(texts: list[str]) -> None:
    result = split_grounded_context([], [{"assumptions": texts, "assumption": "ignored"}])
    expected = [
        {"assumptionId": f"attempt-0-{index}", "text": text.strip()} for index, text in enumerate(texts) if text.strip()
    ]
    assert result == {"grounded": [], "assumptions": expected}


# --- reasoner analysis validation: boundaries, order, roundtrip ------------
def _analysis_candidate(**changes: object) -> dict[str, object]:
    return {
        "nodeId": "n1",
        "objectiveId": "o1",
        "interpretation": "i",
        "evidenceRefs": ["e1"],
        **changes,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extra": True}, "unexpected fields"),
        ({"nodeId": "unknown"}, "analysis references unknown proposal unknown"),
        ({"objectiveId": "wrong"}, "analysis n1 references wrong objective"),
        ({"interpretation": ""}, "analysis.interpretation must be a nonempty string"),
        ({"evidenceRefs": "e1"}, "evidenceRefs must be string"),
        ({"evidenceRefs": [1]}, "evidenceRefs must be string"),
        ({"evidenceRefs": ["unknown"]}, "references unknown evidence unknown"),
        ({"numbers": None}, "numbers must be an array"),
        ({"numbers": [None]}, "number must be an object"),
        ({"numbers": [{"value": "1", "evidenceId": "e1", "quote": "q", "extra": 1}]}, "unexpected fields"),
        ({"numbers": [{"value": "", "evidenceId": "e1", "quote": "q"}]}, "analysis.number.value must be"),
        ({"numbers": [{"value": "1", "evidenceId": "", "quote": "q"}]}, "analysis.number.evidenceId must be"),
        ({"numbers": [{"value": "1", "evidenceId": "e1", "quote": ""}]}, "analysis.number.quote must be"),
        ({"numbers": [{"value": "1", "evidenceId": "e2", "quote": "q"}]}, "number references unknown evidence e2"),
        ({"assumptions": None}, "assumptions must be an array"),
        ({"assumptions": [None]}, "assumption must be an object"),
        ({"assumptions": [{"assumptionId": "a", "text": "x", "extra": 1}]}, "unexpected fields"),
        ({"assumptions": [{"assumptionId": "", "text": "x"}]}, "analysis.assumption.assumptionId must be"),
        ({"assumptions": [{"assumptionId": "a", "text": ""}]}, "analysis.assumption.text must be"),
        (
            {"assumptions": [{"assumptionId": "a", "text": "x"}, {"assumptionId": "a", "text": "y"}]},
            "duplicate assumptionId a",
        ),
    ],
)
def test_reasoner_analysis_validation_boundaries(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_analyses("analyze", [_analysis_candidate(**changes)], "o1", {"n1"}, {"e1", "e2"})


def test_reasoner_analysis_validation_order_and_container_boundaries() -> None:
    with pytest.raises(TypeError, match="analyses must be an array"):
        check_analyses("analyze", {}, "o1", {"n1"}, {"e1"})
    with pytest.raises(ValueError, match="analysis must be an object"):
        check_analyses("analyze", [None], "o1", {"n1"}, {"e1"})
    missing = _analysis_candidate()
    del missing["evidenceRefs"]
    with pytest.raises(ValueError, match="unexpected fields"):
        check_analyses("analyze", [missing], "o1", {"n1"}, {"e1"})
    good = _analysis_candidate()
    with pytest.raises(ValueError, match="duplicate analysis for n1"):
        check_analyses("analyze", [good, good], "o1", {"n1"}, {"e1"})
    # Annotation validation runs before the duplicate-node gate: numbers beat assumptions beat the repeat.
    with pytest.raises(ValueError, match="numbers must be an array"):
        check_analyses("analyze", [good, _analysis_candidate(numbers=None, assumptions=None)], "o1", {"n1"}, {"e1"})
    with pytest.raises(ValueError, match="assumptions must be an array"):
        check_analyses("analyze", [good, _analysis_candidate(assumptions=None)], "o1", {"n1"}, {"e1"})
    assert check_analyses("analyze", [], "o1", set(), set()) == []


@seed(431208)
@settings(max_examples=40, deadline=None)
@given(
    st.text(min_size=1, max_size=32),
    st.text(min_size=1, max_size=32),
    st.lists(st.sampled_from(["e1", "e2"]), min_size=1, max_size=6),
)
def test_reasoner_analysis_roundtrip_preserves_annotation_text_and_refs(
    interpretation: str, quote: str, refs: list[str]
) -> None:
    candidate = _analysis_candidate(
        interpretation=interpretation,
        evidenceRefs=refs,
        numbers=[{"value": "12.5%", "evidenceId": refs[0], "quote": quote}],
        assumptions=[{"assumptionId": "a", "text": interpretation}],
    )
    result = check_analyses("analyze", [candidate], "o1", {"n1"}, {"e1", "e2"})
    assert result == [candidate]
    assert result[0] is not candidate


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({}, TypeError, "evidenceRequests must be an array"),
        ([None], ValueError, "evidenceRequest must be an object"),
        ([{}], ValueError, "unexpected fields"),
        (
            [{"nodeId": "", "objectiveId": "o1", "missingEvidence": "q"}],
            ValueError,
            "evidenceRequest.nodeId must be",
        ),
        (
            [{"nodeId": "unknown", "objectiveId": "o1", "missingEvidence": "q"}],
            ValueError,
            "references unknown node unknown",
        ),
        (
            [{"nodeId": "n1", "objectiveId": "wrong", "missingEvidence": "q"}],
            ValueError,
            "references wrong objective",
        ),
        (
            [{"nodeId": "n1", "objectiveId": "o1", "missingEvidence": ""}],
            ValueError,
            "evidenceRequest.missingEvidence must be",
        ),
    ],
)
def test_reasoner_evidence_request_validation(value: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        check_evidence_requests("expand", value, "o1", {"n1"})


def test_reasoner_evidence_request_valid_lists() -> None:
    request = {"nodeId": "n1", "objectiveId": "o1", "missingEvidence": "raw\nmissing evidence"}
    assert check_evidence_requests("expand", [request], "o1", {"n1"}) == [request]
    assert check_evidence_requests("expand", [], "o1", set()) == []


def test_reasoner_analyze_filters_only_envelope_items() -> None:
    analysis = _analysis_candidate()
    request = {"nodeId": "n1"}
    body: dict[str, object] = {"analyses": [analysis, 1], "evidenceRequests": [None, request]}

    def post(prompt: str, model: str) -> dict[str, object]:
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(body)}]},
            ]
        }

    client = ReasonerClient(model="m", url="u", post=post)
    assert client.analyze("prompt") == {"analyses": [analysis], "evidenceRequests": [request]}
    body.update(analyses=None, evidenceRequests={})
    assert client.analyze("prompt") == {"analyses": [], "evidenceRequests": []}


@pytest.mark.parametrize(
    "changes",
    [
        {"relevant": 1},
        {"headline": "x" * 91},
        {"bottom_line": "word " * 31},
        {"key_points": ["only one"]},
        {"key_points": ["word " * 26, "fact"]},
        {"sentiment": "positive"},
        {"sentiment": []},
        {"impact": "huge"},
        {"impact": None},
        {"watch": 123},
        {"limits": ""},
        {"extra": "field"},
    ],
)
def test_reasoner_story_validation_rejects_malformed_fields(changes: dict[str, object]) -> None:
    candidate = {
        "relevant": True,
        "headline": "Guidance rises",
        "bottom_line": "Revenue guidance increased.",
        "key_points": ["Revenue increased 10%.", "Guidance reached $10 billion."],
        "watch": None,
        "sentiment": "bullish",
        "impact": "medium",
        "limits": None,
        **changes,
    }

    def post(prompt: str, model: str) -> dict[str, object]:
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(candidate)}]}]}

    with pytest.raises(ValueError, match="^summarize_story"):
        ReasonerClient(model="m", url="u", post=post).summarize_story("AMD", "Title", "Article")
