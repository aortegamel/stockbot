"""Kernel entry surface: session scope, graph fan-out, watchlist page load, reasoner contracts."""

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from app.decision_client import JevClient
from app.reasoner_client import ReasonerClient, check_decompose_hints, check_proposals
from app.research import kernel_worker as kw
from app.research import scheduler as sched
from app.research.models import JSONValue
from app.research.repository import ResearchRepository


def test_hints_validator_accepts_optional_rejects_unknown() -> None:
    good = {
        "proposals": [{"id": "x", "v": 1}],
        "tickers": ["nvda"],
        "corrected_query": "fixed query",
    }
    out = check_decompose_hints("decompose", good)
    assert out["tickers"] == ["NVDA"] and out["corrected_query"] == "fixed query"
    try:
        check_decompose_hints("decompose", {"proposals": [], "bogus": 1})
    except ValueError as exc:
        assert "unexpected fields" in str(exc)
    else:
        raise AssertionError("unknown fields must reject")


def test_decompose_end_to_end_carries_all_three_keys() -> None:
    """The transport gate must let tickers/corrected_query reach the hints validator."""
    body: dict[str, object] = {
        "proposals": [
            {
                "id": "rs:t-q1",
                "objectiveId": "rs:t",
                "question": "Q?",
                "dependsOn": [],
                "whyItMatters": "W",
            }
        ],
        "tickers": ["nvda"],
        "corrected_query": "fixed",
    }

    def post(prompt: str, model: str) -> dict[str, object]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(body)}],
                }
            ]
        }

    client = ReasonerClient(api_key="x", model="m", url="u", post=post)
    out = client.decompose("prompt", "rs:t")
    assert out["tickers"] == ["NVDA"] and out["corrected_query"] == "fixed"
    assert isinstance(out["proposals"], list) and len(out["proposals"]) == 1


def _proposal(pid: str, objective_id: str) -> dict[str, object]:
    deps: list[str] = []
    return {
        "id": pid,
        "objectiveId": objective_id,
        "question": "Q?",
        "dependsOn": deps,
        "whyItMatters": "W",
    }


def test_decompose_rejects_prefix_valid_id_with_wrong_objective_id() -> None:
    body = {"proposals": [_proposal("rs:t-q1", "rs:other")]}  # prefix-valid id, wrong objectiveId

    def post(prompt: str, model: str) -> dict[str, object]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(body)}],
                }
            ]
        }

    with pytest.raises(ValueError, match="proposal rs:t-q1 references wrong objective"):
        ReasonerClient(api_key="x", model="m", url="u", post=post).decompose("prompt", "rs:t")


@pytest.mark.parametrize(
    ("objective", "families"),
    [
        (
            "NVDA pull-together: reported EPS, short interest, insider activity, and web sentiment.",
            ["fundamentals", "finra", "insider", "web"],
        ),
        ("What do we know about Rivian across SEC filings, fundamentals, and the web?", ["sec", "fundamentals", "web"]),
        (
            "Compare AMD's fundamentals, valuation multiples, and recent material events.",
            ["fundamentals", "valuation", "events"],
        ),
        (
            "Nvidia stock is trending — pull the trend evidence, then drill into the fundamental numbers behind it.",
            ["alternative", "fundamentals"],
        ),
        ("What is NVDA's latest reported EPS?", ["fundamentals"]),
        ("How do I bake sourdough bread at home?", []),
    ],
)
def test_graph_entry_splits_named_data_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, objective: str, families: list[str]
) -> None:
    """Several named data families get one node per family; one family or none keeps the single objective node."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr(
        "app.tools._resolve_company_to_ticker",
        {"NVDA": "NVDA", "Nvidia": "NVDA", "Rivian": "RIVN", "AMD": "AMD"}.get,
    )
    sid = kw.run_graph_prompt(objective, jev=JevClient())
    nodes = ResearchRepository().list_nodes(sid)
    expected = [f"{objective} ({family} only)" for family in families] if len(families) >= 2 else [objective]
    assert sched.question_source_families(objective) == families
    assert sorted(node.question for node in nodes) == sorted(expected)
    assert all(list(node.depends_on) == [] for node in nodes)
    if len(families) < 2:
        assert [node.why_it_matters for node in nodes] == ["Route question."]


def test_graph_entry_keeps_the_company_split_over_the_family_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A comparison splits by company first: two companies and two families make two nodes, not four."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr("app.tools._resolve_company_to_ticker", {"NVDA": "NVDA", "AMD": "AMD"}.get)
    objective = "Compare NVDA and AMD fundamentals and short interest."
    sid = kw.run_graph_prompt(objective, jev=JevClient())
    assert sched.question_source_families(objective) == ["fundamentals", "finra"]
    assert sorted(node.question for node in ResearchRepository().list_nodes(sid)) == sorted(
        [f"{objective} (NVDA only)", f"{objective} (AMD only)"]
    )


def _no_resolution(label: str) -> str | None:
    return None


def _sec_index(hits: Mapping[str, Mapping[str, object]]) -> Callable[..., list[dict[str, object]]]:
    """Fake company index: an exact-query hit or no rows."""

    def _top_hit(query: str, limit: int = 10) -> list[dict[str, object]]:
        hit = hits.get(query)
        return [dict(hit)] if hit is not None else []

    return _top_hit


def _graph_jev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probability: float
) -> tuple[JevClient, list[dict[str, JSONValue]]]:
    """JEV client for graph setup whose noul answers carry one probability; persistence stays off disk."""
    seen: list[dict[str, JSONValue]] = []

    def transport(state: object, questions: dict[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append(questions)
        return {"answers": {ticker: {"type": "noul", "noul": probability} for ticker in questions}}

    def no_persist(**kwargs: object) -> None:
        return None

    client = JevClient(transport=transport, data_root=tmp_path)
    monkeypatch.setattr(client, "_persist", no_persist)
    return client, seen


_ALPHABET_HIT = {"Google": {"name": "Alphabet Inc.", "tickers": ["GOOGL"]}}
_BEST_HIT = {"Best": {"name": "BestGofer Inc.", "tickers": ["BGFR"]}}


def test_graph_entry_asks_jev_for_a_subject_rules_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A company the lexical rules miss reaches JEV as one yes/no; the kept ticker becomes the session subject."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr("app.sec.client.find_sec_company", _sec_index(_ALPHABET_HIT))
    jev, seen = _graph_jev(monkeypatch, tmp_path, 0.8)
    prompt = "What does Google's cloud revenue look like?"

    sid = kw.run_graph_prompt(prompt, jev=jev)

    session = ResearchRepository().get_session(sid)
    assert session.ticker == "GOOGL"
    assert session.objective == prompt
    assert len(seen) == 1
    question = seen[0]["GOOGL"]
    assert isinstance(question, dict)
    text = question["instructions"]
    assert isinstance(text, str)
    assert text.startswith(f"Request: {prompt} Does this request ask about Alphabet Inc. (ticker GOOGL)?")
    assert "The request names it as 'Google'." in text


def test_graph_entry_skips_jev_when_a_rule_resolves_the_subject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit rule hit never spends a JEV round, and the session keeps no stamped ticker."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)

    def _oracle(label: str) -> str | None:
        return "ORCL" if label.casefold() == "oracle" else None

    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _oracle)
    monkeypatch.setattr("app.sec.client.find_sec_company", _sec_index({}))
    jev, seen = _graph_jev(monkeypatch, tmp_path, 0.8)

    sid = kw.run_graph_prompt("What is happening with Oracle now", jev=jev)

    assert ResearchRepository().get_session(sid).ticker is None
    assert seen == []


def test_graph_entry_keeps_no_subject_when_jev_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing JEV subject round leaves the run intact with no session subject."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr("app.sec.client.find_sec_company", _sec_index(_ALPHABET_HIT))

    def transport(state: object, questions: dict[str, JSONValue]) -> dict[str, JSONValue]:
        raise RuntimeError("decide down")

    jev = JevClient(transport=transport, data_root=tmp_path)
    with caplog.at_level("WARNING"):
        sid = kw.run_graph_prompt("What does Google's cloud revenue look like?", jev=jev)

    store = ResearchRepository()
    assert store.get_session(sid).ticker is None
    assert [node.question for node in store.list_nodes(sid)] == ["What does Google's cloud revenue look like?"]
    assert "subject selection failed" in caplog.text


def test_graph_entry_without_a_jev_client_asks_nobody(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A caller without a JEV client (thesis/script path) creates the run with no subject."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr("app.sec.client.find_sec_company", _sec_index(_ALPHABET_HIT))

    sid = kw.run_graph_prompt("What does Google's cloud revenue look like?")

    assert ResearchRepository().get_session(sid).ticker is None


def test_graph_entry_rejects_a_plain_word_misfire_when_jev_says_no(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A capitalized ordinary word that matches an issuer never grounds the run without JEV assent."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _no_resolution)
    monkeypatch.setattr("app.sec.client.find_sec_company", _sec_index(_BEST_HIT))
    jev, seen = _graph_jev(monkeypatch, tmp_path, 0.1)

    sid = kw.run_graph_prompt("Best AI stock right now?", jev=jev)

    assert ResearchRepository().get_session(sid).ticker is None
    assert len(seen) == 1
    question = seen[0]["BGFR"]
    assert isinstance(question, dict)
    assert "BestGofer Inc. (ticker BGFR)" in str(question["instructions"])


# Newest-first EDGAR rows: five newer 4/A filings would fill a mixed limit=5 window and bury the 8-K, 10-Q and 10-K.


_UNAVAILABLE = (
    "The requested data is unavailable: one or more tool calls failed or returned no data, so the exact "
    "values cannot be provided. No values are estimated, derived, or substituted.\n\nTool: {tool}\n  error: {msg}\n\n"
    "Next step: correct the request (dataset, fields, filters, or credentials) and retry."
)


def _watch_symbol(args: Mapping[str, object]) -> str:
    return str(args.get("identifier") or args.get("ticker"))


def _rows(value: object) -> list[dict[str, JSONValue]]:
    """Narrow a worker watchlist reply to its row dicts."""
    assert isinstance(value, list) and all(isinstance(r, dict) for r in value), value
    return [r for r in value if isinstance(r, dict)]


def _watch_success(tool: str, symbol: str) -> dict[str, object]:
    """Gateway success shape (content + safe meta) with per-issuer provenance."""
    return {
        "content": f"{tool} rows for {symbol}",
        "meta": {
            "as_of": f"2026-10-0{len(symbol)}",
            "returned_count": 3,
            "source_refs": {"record_id": f"{symbol}:{tool}", "uri": f"https://example.test/{symbol}", "known_at": "x"},
        },
    }


def test_watchlist_page_load_overlaps_read_only_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """op:watchlist fans out on the existing pools with no research session, job, or evidence."""
    import threading

    from app.research import tool_runner
    from app.tool_runtime import RuntimeToolSession

    symbols = ["NVDA", "QQQ"]
    # 2 SEC calls fit the 4-worker SEC pool and 6 FINRA calls fit the 8-worker tool pool, so all can overlap.
    gate = threading.Barrier(len(symbols) * len(kw._WATCHLIST_SOURCES), timeout=10)
    lock = threading.Lock()
    seen: list[tuple[str, dict[str, object], object, object, str]] = []

    def _gateway(tool: str, args: dict[str, object], session: RuntimeToolSession, **k: object) -> dict[str, object]:
        with lock:
            seen.append(
                (
                    tool,
                    dict(args),
                    k.get("active_research_session_id"),
                    k.get("active_research_job_id"),
                    session.session_id,
                )
            )
        gate.wait()  # BrokenBarrierError (an error row) unless every call is in flight at once
        if (tool, args.get("ticker")) == ("get_threshold_securities", "QQQ"):
            return {"error": _UNAVAILABLE.format(tool=tool, msg="No data found for QQQ: thresholdList")}
        return _watch_success(tool, _watch_symbol(args))

    monkeypatch.setattr(tool_runner, "execute_agent_tool", _gateway)
    out = kw._watchlist({"id": "w1", "watchlist": ["nvda", "QQQ", "nvda"]})
    assert out["id"] == "w1"
    rows = _rows(out["watchlist"])
    assert len(rows) == 8
    assert [(r["symbol"], r["tool"]) for r in rows] == [(s, t) for s in symbols for t in kw._WATCHLIST_SOURCES]
    assert len(seen) == 8
    # Read-only: no research session, no job, so the gateway stages and persists nothing.
    assert {(sid, jid, rsid) for _t, _a, sid, jid, rsid in seen} == {("", "", "watchlist")}
    assert {(t, _watch_symbol(a)) for t, a, *_ in seen} == {(t, s) for s in symbols for t in kw._WATCHLIST_SOURCES}
    statuses = {(r["symbol"], r["tool"]): r["status"] for r in rows}
    assert statuses.pop(("QQQ", "get_threshold_securities")) == "no_records"
    assert set(statuses.values()) == {"ok"}
    assert all("evidence_id" not in r for r in rows)


def test_watchlist_page_load_past_deadline_reports_every_call_as_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spent deadline cancels outstanding calls and still reports one timeout row per (symbol, source)."""
    import threading
    import time as _t

    from app.research import tool_runner

    release = threading.Event()

    def _held(tool: str, args: dict[str, object], session: object, **k: object) -> dict[str, object]:
        release.wait(10)
        return _watch_success(tool, _watch_symbol(args))

    monkeypatch.setattr(tool_runner, "execute_agent_tool", _held)
    try:
        out = kw._watchlist({"id": "w2", "watchlist": ["SPY"], "deadlineAt": (_t.time() - 1) * 1000})
    finally:
        release.set()
    rows = _rows(out["watchlist"])
    assert [(r["symbol"], r["source"], r["status"]) for r in rows] == [
        ("SPY", source, "timeout") for source in kw._WATCHLIST_SOURCES.values()
    ]
    assert all(r["error"] == "no result within 0s" for r in rows)


def test_watchlist_page_load_reuses_finra_cache_until_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page load fills the existing FINRA cache; a fresh hit skips the source, an expired one refreshes it."""
    from datetime import UTC, datetime, timedelta
    from unittest.mock import MagicMock

    from app import finra_client
    from tests.test_finra import _mock_get, _response, _token_response

    clock = {"now": 1_000.0}

    class _TtlCache:
        def __init__(self) -> None:
            self.rows: dict[str, tuple[object, float]] = {}

        def get(self, key: str, ttl: float | None = None) -> object | None:
            hit = self.rows.get(key)
            if hit is None or (ttl is not None and clock["now"] - hit[1] > ttl):
                return None
            return hit[0]

        def set(self, key: str, value: object) -> None:
            self.rows[key] = (json.loads(json.dumps(value)), clock["now"])

    settled = (datetime.now(UTC).date() - timedelta(days=3)).isoformat()
    record = {"symbolCode": "AAPL", "settlementDate": settled, "currentShortPositionQuantity": 100}
    post = MagicMock(side_effect=[_token_response(), _response([record]), _response([record])])
    monkeypatch.setenv("FINRA_USE_MOCK", "")
    monkeypatch.setenv("FINRA_ANALYSIS_MODEL", "")
    monkeypatch.setattr("app.finra_client.requests.get", MagicMock(side_effect=_mock_get))
    monkeypatch.setattr("app.finra_client.requests.post", post)
    monkeypatch.setattr("app.finra_client.get_finra_client_id", lambda: "client")
    monkeypatch.setattr("app.finra_client.get_finra_client_secret", lambda: "secret")
    monkeypatch.setattr(finra_client, "cache", _TtlCache())
    finra_client.reset_token_cache()
    finra_client.reset_discovery_cache()
    finra_client.reset_partitions_cache()

    # The handler's exact-ticker check stays offline: AAPL resolves without EDGAR.
    def _cik(value: object) -> int:
        return 320193

    monkeypatch.setattr("app.sec.client.resolve_cik", _cik)
    monkeypatch.setattr(kw, "_WATCHLIST_SOURCES", {"get_short_interest": "finra_short_interest"})

    def data_posts() -> int:
        return sum(1 for c in post.call_args_list if "oauth2" not in str(c.args[0]))

    try:
        page = asyncio.run(kw._watchlist_fetch(["AAPL"], 10.0))
        assert page[0]["status"] == "ok" and page[0]["symbol"] == "AAPL", page
        assert data_posts() == 1
        clock["now"] += finra_client.CACHE_TTL_SECONDS - 1
        assert asyncio.run(kw._watchlist_fetch(["AAPL"], 10.0)) == page
        assert data_posts() == 1  # fresh cache: no duplicate source work
        clock["now"] += 2
        assert asyncio.run(kw._watchlist_fetch(["AAPL"], 10.0))[0]["status"] == "ok"
        assert data_posts() == 2  # expired under the existing TTL: one refresh through the same path
    finally:
        finra_client.reset_token_cache()
        finra_client.reset_discovery_cache()
        finra_client.reset_partitions_cache()


# --- reasoner intake: hint/proposal/response boundaries + real transport ----
def _reasoner_response(text: str) -> dict[str, object]:
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], TypeError, "malformed_opencode_json"),
        ({}, ValueError, "unexpected fields"),
        ({"proposals": [], "extra": 1}, ValueError, "unexpected fields"),
        ({"proposals": [], "tickers": "NVDA"}, ValueError, "tickers must be string"),
        ({"proposals": [], "tickers": [1]}, ValueError, "tickers must be string"),
        ({"proposals": [], "tickers": [" "]}, ValueError, "tickers must be string"),
        ({"proposals": [], "corrected_query": 1}, ValueError, "corrected_query must be"),
        ({"proposals": [], "corrected_query": " "}, ValueError, "corrected_query must be"),
    ],
)
def test_reasoner_hint_boundaries(value: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        check_decompose_hints("decompose", value)


def test_reasoner_null_hints_omit_optional_fields() -> None:
    assert check_decompose_hints("decompose", {"proposals": [], "tickers": None, "corrected_query": None}) == {
        "proposals": []
    }
    hints = check_decompose_hints(
        "decompose", {"proposals": [], "tickers": [" nvda ", "aapl"], "corrected_query": " q "}
    )
    assert hints == {"proposals": [], "tickers": ["NVDA", "AAPL"], "corrected_query": "q"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "wrong"}, "must start with"),
        ({"objectiveId": "other"}, "references wrong objective"),
        ({"question": ""}, "proposal.question must be"),
        ({"whyItMatters": ""}, "proposal.whyItMatters must be"),
        ({"dependsOn": "s-a"}, "dependsOn must be string"),
        ({"dependsOn": [1]}, "dependsOn must be string"),
        ({"dependsOn": ["s-a"]}, "depends on itself"),
        ({"dependsOn": ["unknown"]}, "references unknown id"),
        ({"extra": True}, "unexpected fields"),
    ],
)
def test_reasoner_proposal_boundaries(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_proposals("decompose", [{**_proposal("s-a", "s"), **change}], "s", set())


def test_reasoner_proposal_batch_validation_precedes_dependencies() -> None:
    bad_dependency = {**_proposal("s-a", "s"), "dependsOn": ["unknown"]}
    with pytest.raises(ValueError, match="proposal must be an object"):
        check_proposals("decompose", [bad_dependency, None], "s", set())
    proposal = _proposal("s-a", "s")
    with pytest.raises(ValueError, match="duplicate proposal id"):
        check_proposals("decompose", [proposal, proposal], "s", set())
    with pytest.raises(ValueError, match="reused proposal id"):
        check_proposals("decompose", [proposal], "s", {"s-a"})
    with pytest.raises(TypeError, match="proposals must be an array"):
        check_proposals("decompose", {}, "s", set())
    forward = [{**_proposal("s-b", "s"), "dependsOn": ["s-a"]}, _proposal("s-a", "s")]
    assert [p["id"] for p in check_proposals("decompose", forward, "s", set())] == ["s-b", "s-a"]
    prior = {**_proposal("s-b", "s"), "dependsOn": ["s-prior"]}
    assert check_proposals("decompose", [prior], "s", {"s-prior"}) == [prior]


@seed(431207)
@settings(max_examples=40, deadline=None)
@given(st.lists(st.text(min_size=1, max_size=16), max_size=5))
def test_reasoner_split_text_roundtrips_valid_proposals(questions: list[str]) -> None:
    proposals = [{**_proposal(f"s-{i}", "s"), "question": question} for i, question in enumerate(questions)]
    text = json.dumps({"proposals": proposals}, ensure_ascii=True)
    parts: list[dict[str, object]] = [
        {"type": "output_text", "text": text[: len(text) // 2]},
        {"type": "ignored", "text": "not JSON"},
        {"type": "output_text", "text": 17},
        {"type": "output_text", "text": text[len(text) // 2 :]},
    ]
    response: dict[str, object] = {
        "output": [
            None,
            {"type": "reasoning"},
            {"type": "message", "content": None},
            {"type": "message", "content": [None, *parts]},
        ]
    }
    calls: list[tuple[str, str]] = []

    def post(prompt: str, model: str) -> dict[str, object]:
        calls.append((prompt, model))
        return response

    assert ReasonerClient(model="m", url="u", post=post).decompose("prompt\nbytes", "s") == {"proposals": proposals}
    assert calls == [("prompt\nbytes", "m")]


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        ({}, TypeError, "malformed_opencode_response"),
        ({"output": {}}, TypeError, "malformed_opencode_response"),
        ({"output": []}, ValueError, "malformed_opencode_response"),
        (_reasoner_response("{"), ValueError, "malformed_opencode_json"),
        (_reasoner_response("[]"), TypeError, "malformed_opencode_json"),
        (_reasoner_response('{"extra":1}'), ValueError, "unexpected fields"),
    ],
)
def test_reasoner_response_boundaries(response: dict[str, object], error: type[Exception], message: str) -> None:
    def post(prompt: str, model: str) -> dict[str, object]:
        return response

    with pytest.raises(error, match=message):
        ReasonerClient(model="m", url="u", post=post).decompose("prompt", "s")


def test_reasoner_expand_preserves_prior_dependencies_and_filters_requests() -> None:
    proposal = {**_proposal("s-new", "s"), "dependsOn": ["s-prior"]}
    body: dict[str, object] = {"proposals": [proposal], "evidenceRequests": [{"nodeId": "s-new"}, 4]}

    def post(prompt: str, model: str) -> dict[str, object]:
        return _reasoner_response(json.dumps(body))

    client = ReasonerClient(model="m", url="u", post=post)
    assert client.expand("prompt", "s", {"s-prior"}) == {
        "proposals": [proposal],
        "evidenceRequests": [{"nodeId": "s-new"}],
    }
    body["evidenceRequests"] = None
    assert client.expand("prompt", "s", {"s-prior"})["evidenceRequests"] == []
    with pytest.raises(ValueError, match="reused proposal id"):
        client.expand("prompt", "s", {"s-prior", "s-new"})


def test_reasoner_real_transport_sends_exact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    from urllib.request import Request

    from app import reasoner_client as rc

    requests: list[tuple[Request, float]] = []

    def urlopen(request: Request, timeout: float) -> io.BytesIO:
        requests.append((request, timeout))
        return io.BytesIO(json.dumps(_reasoner_response('{"proposals":[]}')).encode())

    monkeypatch.setattr(rc._urlrequest, "urlopen", urlopen)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    client = ReasonerClient(model="m", url="http://opencode.invalid", api_key="secret", timeout_s=7)
    assert client.decompose("raw\nprompt", "s") == {"proposals": []}
    request, timeout = requests[0]
    assert request.full_url == "http://opencode.invalid"
    data = request.data
    assert isinstance(data, bytes)
    assert json.loads(data) == {"model": "m", "input": "raw\nprompt"}
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 7
    # REASONING_EFFORT passes through unchanged as Responses reasoning.effort.
    monkeypatch.setenv("REASONING_EFFORT", "xhigh")
    client.decompose("raw\nprompt", "s")
    data = requests[1][0].data
    assert isinstance(data, bytes)
    assert json.loads(data) == {"model": "m", "input": "raw\nprompt", "reasoning": {"effort": "xhigh"}}
    # A bad shape is a config failure (terminal for the retry loop) before any request.
    monkeypatch.setenv("REASONING_EFFORT", "X High")
    with pytest.raises(RuntimeError, match=r"^opencode_unavailable: REASONING_EFFORT must match"):
        client.decompose("raw\nprompt", "s")
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"detail", "opencode_request_failed: 429 detail"),
        (b"", "opencode_request_failed: 429"),
    ],
)
def test_reasoner_http_error_detail(monkeypatch: pytest.MonkeyPatch, body: bytes, expected: str) -> None:
    import io
    from email.message import Message
    from urllib.error import HTTPError

    from app import reasoner_client as rc

    def urlopen(request: object, timeout: float) -> object:
        raise HTTPError("http://opencode.invalid", 429, "limited", Message(), io.BytesIO(body))

    monkeypatch.setattr(rc._urlrequest, "urlopen", urlopen)
    with pytest.raises(RuntimeError) as exc:
        ReasonerClient(model="m", url="http://opencode.invalid", api_key="k").decompose("prompt", "s")
    assert str(exc.value) == expected


@pytest.mark.parametrize(
    ("raw", "error", "message"),
    [
        (b"[]", TypeError, "non-object response"),
        (b"{", RuntimeError, "opencode_request_failed"),
    ],
)
def test_reasoner_transport_response_shape(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, error: type[Exception], message: str
) -> None:
    import io

    from app import reasoner_client as rc

    def urlopen(request: object, timeout: float) -> io.BytesIO:
        return io.BytesIO(raw)

    monkeypatch.setattr(rc._urlrequest, "urlopen", urlopen)
    with pytest.raises(error, match=message):
        ReasonerClient(model="m", url="http://opencode.invalid", api_key="k").decompose("prompt", "s")
