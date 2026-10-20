"""Trigger runs plus deterministic monitoring (monitor.tick)."""

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

import app.thesis.runner as runner_mod
from app.thesis.context import ContextBudgetExceeded, build_context, build_live_context
from app.thesis.models import Thesis
from app.thesis.monitor import CanonicalEvent, tick
from app.thesis.repository import ThesisRepository
from app.thesis.runner import run_trigger
from app.thesis.yaml import atomic_write_yaml, load_raw_yaml

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T00:00:00+00:00"
T2 = "2026-01-03T00:00:00+00:00"
T4 = "2026-01-05T00:00:00+00:00"


class _Src:
    name = "sec_filings"

    def __init__(self, events: Sequence[CanonicalEvent] = (), fail: bool = False, key: str = "sec_filings") -> None:
        self.events = list(events)
        self.fail = fail
        self.key = key
        self.calls = 0
        self.cutoffs: list[str] = []

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
        self.calls += 1
        self.cutoffs.append(known_at)
        if self.fail:
            raise RuntimeError("source boom")
        return [e for e in self.events if e.known_at <= known_at]


class _Kernel:
    """Fake _RUN_KERNEL: counts launches; ok-runs apply repo writes like kernel tool calls."""

    def __init__(self, fail: bool = False, write: Callable[..., None] | None = None) -> None:
        self.fail = fail
        self.write = write
        self.calls = 0
        self.prompts: list[str] = []
        self.run_ids: list[str] = []
        self.last_as_of: str | None = None
        self.last_run_id: str = ""

    def __call__(self, prompt: str, **kwargs: object) -> None:
        self.calls += 1
        self.prompts.append(prompt)
        tid = str(kwargs.get("thesis_id", ""))
        trig = str(kwargs.get("trigger_id", ""))
        rid = str(kwargs.get("run_id", ""))
        self.run_ids.append(rid)
        self.last_run_id = rid
        self.last_as_of = None
        if self.fail:
            raise RuntimeError("kernel boom")
        if self.write is not None:
            try:
                self.write(tid, trig, rid)
            except TypeError:
                self.write(tid, trig)


def _kernel(monkeypatch: pytest.MonkeyPatch, fail: bool = False, write: Callable[..., None] | None = None) -> _Kernel:
    fake = _Kernel(fail=fail, write=write)
    monkeypatch.setattr(runner_mod, "_RUN_KERNEL", fake)
    return fake


def _make(
    tmp_path: Path,
    scope: str = "NVDA",
    rule: str = "new_filing",
    exprs: Sequence[dict[str, object]] = (),
    invalidators: Sequence[str] = (),
) -> tuple[ThesisRepository, Thesis]:
    r = ThesisRepository(tmp_path / "theses")
    t = r.create_thesis(
        f"{scope} thesis",
        scope=scope,
        claims=[f"{scope} demand grows"],
        expressions=list(exprs),
        invalidators=list(invalidators),
        effective_at=T0,
    )
    full = r.load_thesis(t.thesis_id)
    cid = full.claims[0].claim_id
    eids = [e.expression_id for e in full.expressions]
    r.apply_research_result(
        t.thesis_id,
        {
            "watch_add": [
                {
                    "rule_id": "rule:1",
                    "rule_type": rule,
                    "enabled": True,
                    "support_status": "supported",
                    "support_reason": "",
                    "claim_ids": [cid],
                    "expression_ids": list(eids),
                }
            ]
        },
        "",
        effective_at=T0,
    )
    return r, t


def _ev(
    ref: str, known_at: str = T1, entity: str = "NVDA", summary: str = "NVDA files 10-K noting steady demand"
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=ref, canonical_ref=ref, source="sec_filings", known_at=known_at, entity=entity, summary=summary
    )


def _journal_text(tmp_path: Path, slug: str) -> str:
    return " ".join(
        p.read_text(encoding="utf-8") for p in sorted((tmp_path / "theses" / slug / "journal").glob("*.md"))
    )


def _journal_count(tmp_path: Path, slug: str) -> int:
    return len(list((tmp_path / "theses" / slug / "journal").glob("*.md")))


# -- expression assessment via fake-kernel run_trigger outcomes ---------------------


def test_timing_mismatch_flagged_without_changing_thesis_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(
        tmp_path,
        exprs=[
            {
                "instrument": "option",
                "direction": "long",
                "structure": "long puts",
                "horizon": "short-term",
                "status": "active",
            }
        ],
    )
    full = r.load_thesis(t.thesis_id)
    eid = full.expressions[0].expression_id
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        expression_ids=[eid],
        canonical_refs=["ev:m"],
        summary="s",
        summary_origin="deterministic",
    )

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "expression_updates": [{"expression_id": eid, "status": "flagged"}],
                "evidence_refs": [{"canonical_ref": "ev:m", "summary": "30-day puts noted", "known_at": T1}],
                "journal_entry": {
                    "entry_id": "journal:t1",
                    "title": "t",
                    "body": "timing mismatch: 30-day puts vs 2-year thesis; near-term filing cuts against puts timing",
                    "known_at": T2,
                },
            },
            run_id,
        )

    fake = _kernel(monkeypatch, write=_write)
    out = run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert out.processed and fake.calls == 1
    assert r.load_thesis(t.thesis_id).expressions[0].status == "flagged"
    assert r.load_state(t.thesis_id).assessment == "unresolved"
    body = _journal_text(tmp_path, t.slug)
    assert "timing mismatch" in body and "cuts against" in body


def test_bullish_equity_asks_no_options_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(
        tmp_path,
        exprs=[
            {
                "instrument": "equity",
                "direction": "long",
                "structure": "equity",
                "horizon": "long-term",
                "status": "active",
            }
        ],
    )
    full = r.load_thesis(t.thesis_id)
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        canonical_refs=["ev:e"],
        summary="s",
        summary_origin="deterministic",
    )

    def _journal_ok(tid: str, trig_id: str, run_id: str) -> None:
        r.append_journal_entry(tid, {"title": "t", "body": "ok", "trigger_id": trig_id, "run_id": run_id})

    fake = _kernel(monkeypatch, write=_journal_ok)
    out = run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert out.processed and fake.calls == 1
    questions = r.load_questions(t.thesis_id)
    assert questions == []
    assert r.load_state(t.thesis_id).assessment == "unresolved"


def test_covered_call_without_portfolio_leaves_ownership_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, t = _make(
        tmp_path, exprs=[{"instrument": "option", "direction": "long", "structure": "covered call", "status": "active"}]
    )
    full = r.load_thesis(t.thesis_id)
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        canonical_refs=["ev:c"],
        summary="s",
        summary_origin="deterministic",
    )

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [{"canonical_ref": "ev:c", "summary": "call overlay noted", "known_at": T1}],
                "journal_entry": {
                    "entry_id": "journal:t1",
                    "title": "t",
                    "body": "ownership unresolved without portfolio read",
                    "known_at": T2,
                },
            },
            run_id,
        )

    fake = _kernel(monkeypatch, write=_write)
    out = run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert out.processed and fake.calls == 1
    assert r.load_thesis(t.thesis_id).expressions[0].deterministic_support == "unknown"
    assert "unresolved" in _journal_text(tmp_path, t.slug)


def test_long_puts_without_market_has_no_invented_prices_or_greeks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, t = _make(
        tmp_path, exprs=[{"instrument": "option", "direction": "long", "structure": "long puts", "status": "active"}]
    )
    full = r.load_thesis(t.thesis_id)
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        canonical_refs=["ev:p"],
        summary="s",
        summary_origin="deterministic",
    )

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [
                    {"canonical_ref": "ev:p", "summary": "puts noted, no quote available", "known_at": T1}
                ],
                "journal_entry": {
                    "entry_id": "journal:t1",
                    "title": "t",
                    "body": "not evaluable without market read; no quotes fetched",
                    "known_at": T2,
                },
            },
            run_id,
        )

    _kernel(monkeypatch, write=_write)
    run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    body = _journal_text(tmp_path, t.slug)
    assert not re.search(r"\$\s*\d", body)
    assert not re.search(r"(?i)\b(delta|gamma|theta|vega|implied volatility)\b", body)


def test_unknown_and_processed_triggers_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    fake = _kernel(monkeypatch)
    with pytest.raises(KeyError):
        run_trigger(r, t.thesis_id, "trigger:nope", known_at=T2)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["ev:1"], summary="s", summary_origin="deterministic")

    def _journal_ok2(tid: str, trig_id: str, run_id: str) -> None:
        r.append_journal_entry(tid, {"title": "t", "body": "ok", "trigger_id": trig_id, "run_id": run_id})

    fake = _kernel(monkeypatch, write=_journal_ok2)
    out = run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert out.processed and fake.calls == 1
    with pytest.raises(ValueError):
        run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert fake.calls == 1


# -- monitoring ----------------------------------------------------------------


def test_irrelevant_event_produces_zero_kernel_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    fake = _kernel(monkeypatch)
    src = _Src([_ev("ev:irr", entity="UNRELATEDCORPXYZ", summary="unrelated corp files")])
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert fake.calls == 0 and res.triggers_created == [] and res.no_op


def test_duplicate_event_produces_zero_second_kernel_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [{"canonical_ref": "ev:d", "summary": "NVDA files", "known_at": T1}],
                "journal_entry": {"entry_id": "journal:t1", "title": "t", "body": "ok", "known_at": T2},
            },
            run_id,
        )

    fake = _kernel(monkeypatch, write=_write)
    src = _Src([_ev("ev:d")])
    first = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert fake.calls == 1 and len(first.triggers_created) == 1
    second = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert fake.calls == 1 and second.triggers_created == []


def test_identical_events_within_one_tick_coalesce_to_one_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    fake = _kernel(monkeypatch)
    src = _Src([_ev("ev:same"), _ev("ev:same")])
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert len(res.triggers_created) == 1 and fake.calls == 1


def test_new_trigger_pushes_ntfy_only_with_http_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    posts: list[urllib.request.Request] = []

    class _Resp:
        def close(self) -> None:
            return None

    def _urlopen(req: urllib.request.Request, timeout: float) -> _Resp:
        assert timeout == 5
        posts.append(req)
        if req.full_url.endswith("/down"):
            raise OSError("ntfy down")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    _kernel(monkeypatch)
    r, t = _make(tmp_path)
    link = "https://www.sec.gov/Archives/edgar/data/1045810/x-index.htm"
    filing = CanonicalEvent(
        event_id="sec:1",
        canonical_ref="sec:1",
        source="sec_filings",
        known_at=T1,
        entity="NVDA",
        summary="8-K filed by NVDA",
        metadata={"link": link},
    )
    monkeypatch.delenv("NTFY_TOPIC_URL", raising=False)
    assert len(tick(r, t.thesis_id, {"sec_filings": _Src([_ev("ev:a")])}, known_at=T2).triggers_created) == 1
    monkeypatch.setenv("NTFY_TOPIC_URL", "file:///etc/passwd")
    assert len(tick(r, t.thesis_id, {"sec_filings": _Src([_ev("ev:b")])}, known_at=T2).triggers_created) == 1
    assert posts == []
    monkeypatch.setenv("NTFY_TOPIC_URL", "https://ntfy.example/alerts")
    assert len(tick(r, t.thesis_id, {"sec_filings": _Src([filing])}, known_at=T2).triggers_created) == 1
    [req] = posts
    assert req.full_url == "https://ntfy.example/alerts" and req.get_method() == "POST"
    assert req.get_header("Title") == f"{t.slug}: new_filing" and req.get_header("Click") == link
    assert req.data == f"8-K filed by NVDA\n{link}".encode()
    monkeypatch.setenv("NTFY_TOPIC_URL", "https://ntfy.example/down")
    assert len(tick(r, t.thesis_id, {"sec_filings": _Src([_ev("ev:c")])}, known_at=T2).triggers_created) == 1
    assert posts[-1].data == b"NVDA files 10-K noting steady demand\nev:c" and posts[-1].get_header("Click") is None


def test_relevant_filing_produces_one_bounded_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = ThesisRepository(tmp_path / "theses")
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    cid = r.load_thesis(t.thesis_id).claims[0].claim_id
    r.apply_research_result(
        t.thesis_id,
        {
            "watch_add": [
                {
                    "rule_id": "rule:f",
                    "rule_type": "new_filing",
                    "enabled": True,
                    "support_status": "supported",
                    "support_reason": "",
                    "claim_ids": [cid],
                    "expression_ids": [],
                }
            ]
        },
        "",
        effective_at=T0,
    )

    class FilingSrc:
        name = "sec_filings"
        calls = 0

        def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
            type(self).calls += 1
            return [
                CanonicalEvent(
                    event_id="f1",
                    canonical_ref="edgar:NVDA:10-K:f1",
                    source="sec_filings",
                    known_at=T1,
                    entity="NVDA",
                    summary="NVDA files 10-K",
                )
            ]

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [
                    {"canonical_ref": "edgar:NVDA:10-K:f1", "summary": "10-K notes steady demand", "known_at": T1}
                ],
                "journal_entry": {
                    "entry_id": "journal:t1",
                    "title": "t",
                    "body": "reviewed filing; risk factors mention cyclicality",
                    "known_at": T2,
                },
            },
            run_id,
        )

    fake = _kernel(monkeypatch, write=_write)
    res = tick(r, t.thesis_id, {"sec_filings": FilingSrc()}, known_at=T2)
    assert fake.calls == 1 and len(res.triggers_created) == 1
    assert "cyclicality" in _journal_text(tmp_path, t.slug)


def test_simultaneous_meaningful_events_produce_bounded_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)

    def _journal_ok3(tid: str, trig_id: str, run_id: str) -> None:
        r.append_journal_entry(tid, {"title": "t", "body": "ok", "trigger_id": trig_id, "run_id": run_id})

    fake = _kernel(monkeypatch, write=_journal_ok3)
    src = _Src([_ev("ev:a", summary="NVDA files 10-K"), _ev("ev:b", summary="NVDA 8-K event")])
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert len(res.triggers_created) == 2 and fake.calls == 2


def test_possible_invalidator_produces_one_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path, rule="explicit_thesis_invalidator", invalidators=["demand collapse scenario"])
    r.apply_research_result(
        t.thesis_id,
        {
            "evidence_refs": [
                {
                    "evidence_id": "ev:seed",
                    "thesis_id": t.thesis_id,
                    "canonical_ref": "seed:1",
                    "summary": "analyst warns of demand collapse scenario for NVDA",
                    "known_at": T1,
                }
            ]
        },
        "run:seed",
    )
    fake = _kernel(monkeypatch)
    res = tick(r, t.thesis_id, {}, known_at=T2)
    assert fake.calls == 1 and len(res.triggers_created) == 1


def test_expression_impact_event_tags_expression_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path, exprs=[{"structure": "long puts", "status": "active"}])
    fake = _kernel(monkeypatch)
    src = _Src([_ev("ev:x", summary="NVDA volatility event affects puts")])
    tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert fake.calls == 1
    trig = r.load_triggers(t.thesis_id)[0]
    assert trig.expression_ids != ()


def test_source_failure_leaves_checkpoint_unadvanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    before = r.load_checkpoint(t.thesis_id).to_dict()
    fake = _kernel(monkeypatch)
    res = tick(r, t.thesis_id, {"sec_filings": _Src(fail=True)}, known_at=T2)
    assert fake.calls == 0 and res.triggers_created == []
    assert r.load_checkpoint(t.thesis_id).to_dict() == before


def test_kernel_failure_leaves_pending_and_halts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    src = _Src([_ev("ev:1"), _ev("ev:2")])
    fake = _kernel(monkeypatch, fail=True)
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert res.runs == [] and len(res.triggers_created) == 2
    pending = [x for x in r.load_triggers(t.thesis_id) if x.status == "pending"]
    assert len(pending) == 2 and fake.calls == 1


def test_restart_processes_pending_once_without_dupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    src = _Src([_ev("ev:1")])
    bad = _kernel(monkeypatch, fail=True)
    tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert bad.calls == 1
    journals_before = _journal_count(tmp_path, t.slug)

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [{"canonical_ref": "ev:1", "summary": "NVDA files", "known_at": T1}],
                "journal_entry": {"entry_id": "journal:t1", "title": "t", "body": "recovered", "known_at": T2},
            },
            run_id,
        )

    monkeypatch.setattr(runner_mod, "_RUN_KERNEL", _Kernel(write=_write))
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert res.triggers_created == []
    assert all(x.status == "processed" for x in r.load_triggers(t.thesis_id))
    assert _journal_count(tmp_path, t.slug) == journals_before + 1


def test_paused_and_closed_query_no_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    fake = _kernel(monkeypatch)
    r.pause_thesis(t.thesis_id)
    src = _Src([_ev("ev:1")])
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    assert src.calls == 0 and fake.calls == 0 and res.no_op
    r.resume_thesis(t.thesis_id)
    r.close_thesis(t.thesis_id)
    src2 = _Src([_ev("ev:1")])
    with pytest.raises(ValueError):
        tick(r, t.thesis_id, {"sec_filings": src2}, known_at=T2)
    assert src2.calls == 0


def test_two_theses_stay_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = ThesisRepository(tmp_path / "theses")
    a = r.create_thesis("NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    b = r.create_thesis("NVDA second thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    for th in (a, b):
        cid = r.load_thesis(th.thesis_id).claims[0].claim_id
        r.apply_research_result(
            th.thesis_id,
            {
                "watch_add": [
                    {
                        "rule_id": "rule:1",
                        "rule_type": "new_filing",
                        "enabled": True,
                        "support_status": "supported",
                        "support_reason": "",
                        "claim_ids": [cid],
                        "expression_ids": [],
                    }
                ]
            },
            "",
            effective_at=T0,
        )
    _kernel(monkeypatch)
    tick(r, a.thesis_id, {"sec_filings": _Src([_ev("ev:only-a")])}, known_at=T2)
    assert len(r.load_triggers(a.thesis_id)) == 1
    assert r.load_triggers(b.thesis_id) == []


def test_tight_budget_fails_explicitly(tmp_path: Path) -> None:
    r, t = _make(tmp_path)
    trig = r.create_trigger(t.thesis_id, canonical_refs=["ev:1"], summary="s", summary_origin="deterministic")
    with pytest.raises(ContextBudgetExceeded):
        build_context(r, t.thesis_id, trig, known_at=T2, max_tokens=5)
    ok = build_context(r, t.thesis_id, trig, known_at=T2)
    assert ok.known_at == T2 and isinstance(ok.omitted_ids, list)


def test_large_monitor_checkpoint_not_in_pi_context(tmp_path: Path) -> None:
    r, t = _make(tmp_path)
    r.save_checkpoint(
        t.thesis_id, {"thesis_id": t.thesis_id, "sources": {}, "recent_hashes": ["a" * 64 for _ in range(1000)]}
    )
    trig = r.create_trigger(t.thesis_id, canonical_refs=["ev:1"], summary="s", summary_origin="deterministic")
    ctx = build_live_context(r, t.thesis_id, trig, data_cutoff=T2)
    assert "checkpoint" not in ctx.thesis_packet
    assert ctx.estimated_tokens < 8000


def test_evidence_refs_stay_compact_and_reject_bodies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r, t = _make(tmp_path)
    full = r.load_thesis(t.thesis_id)
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        canonical_refs=["ev:b"],
        summary="s",
        summary_origin="deterministic",
    )

    def _bad(tid: str, trig_id: str) -> None:
        r.apply_research_result(
            tid,
            {
                "trigger_id": trig_id,
                "evidence_refs": [
                    {"canonical_ref": "ev:b", "summary": "note", "known_at": T1, "body": "FULL FILING BODY SMUGGLED"}
                ],
                "journal_entry": {"entry_id": "journal:bad", "title": "t", "body": "b", "known_at": T1},
            },
            "run:bad",
        )

    _kernel(monkeypatch, write=_bad)
    with pytest.raises(ValueError):
        run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert r.load_triggers(t.thesis_id)[0].status == "pending"
    assert list((tmp_path / "theses" / t.slug / "evidence").glob("*.yaml")) == []


def test_stored_pool_skips_missing_and_future_known_at(tmp_path: Path) -> None:
    r, t = _make(tmp_path, rule="explicit_thesis_invalidator", invalidators=["demand collapse scenario"])
    evdir = tmp_path / "theses" / t.slug / "evidence"
    for eid, ka in (("ev:noka", ""), ("ev:future", "2027-01-01T00:00:00+00:00")):
        atomic_write_yaml(
            evdir / f"{eid}.yaml",
            {
                "schema_version": 1,
                "evidence_id": eid,
                "thesis_id": t.thesis_id,
                "canonical_ref": "seed:1",
                "summary": "analyst warns of demand collapse scenario for NVDA",
                "known_at": ka,
            },
            tmp_path / "theses",
        )
    trig = r.create_trigger(
        t.thesis_id,
        canonical_refs=["seed:2"],
        summary="demand collapse scenario unfolding for NVDA",
        summary_origin="deterministic",
    )
    r.mark_trigger_processed(t.thesis_id, trig.trigger_id, "run:seed")
    res = tick(r, t.thesis_id, {}, known_at="2020-01-01T00:00:00+00:00")
    assert res.triggers_created == [] and res.no_op


def test_external_evidence_rule_loads_unsupported_and_never_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, t = _make(tmp_path, rule="new_external_evidence")
    fake = _kernel(monkeypatch)
    src = _Src([_ev("ev:ext", summary="NVDA external note")])
    res = tick(r, t.thesis_id, {"sec_filings": src}, known_at=T2)
    rules = r.load_watch_rules(t.thesis_id)
    assert rules[0].enabled is False and rules[0].support_status == "unsupported"
    assert "no production source" in rules[0].support_reason
    assert res.triggers_created == [] and fake.calls == 0 and src.calls == 0


def test_targetless_watch_add_rejected(tmp_path: Path) -> None:
    r, t = _make(tmp_path)
    with pytest.raises(ValueError):
        r.apply_research_result(
            t.thesis_id,
            {
                "watch_add": [
                    {
                        "rule_id": "rule:targetless",
                        "rule_type": "new_filing",
                        "enabled": True,
                        "support_status": "supported",
                        "support_reason": "",
                        "claim_ids": [],
                        "expression_ids": [],
                    }
                ]
            },
            "run:targetless",
        )


def test_watch_add_rejects_unknown_claim_and_expression_ids(tmp_path: Path) -> None:
    r, t = _make(tmp_path)
    cid = r.load_thesis(t.thesis_id).claims[0].claim_id
    with pytest.raises(ValueError):
        r.apply_research_result(
            t.thesis_id,
            {
                "watch_add": [
                    {
                        "rule_id": "rule:bad-claim",
                        "rule_type": "new_filing",
                        "enabled": True,
                        "support_status": "supported",
                        "support_reason": "",
                        "claim_ids": ["claim:nope"],
                        "expression_ids": [],
                    }
                ]
            },
            "run:bad",
        )
    with pytest.raises(ValueError):
        r.apply_research_result(
            t.thesis_id,
            {
                "watch_add": [
                    {
                        "rule_id": "rule:bad-expr",
                        "rule_type": "new_filing",
                        "enabled": True,
                        "support_status": "supported",
                        "support_reason": "",
                        "claim_ids": [cid],
                        "expression_ids": ["expr:nope"],
                    }
                ]
            },
            "run:bad",
        )


def test_refinement_covers_only_new_targets_and_ignores_disabled(tmp_path: Path) -> None:
    from app.thesis.intake import IntakeProposal, apply_refinement, plan_refinement

    r, t = _make(tmp_path)
    thesis = r.load_thesis(t.thesis_id)
    old_cid = thesis.claims[0].claim_id
    proposal = IntakeProposal.from_dict(
        {"user_thesis": thesis.user_thesis, "scope": "NVDA", "claims": [{"statement": "NVDA networking demand grows"}]},
        "<test>",
    )
    plan = plan_refinement(thesis, proposal)
    out = apply_refinement(r, t.thesis_id, plan, proposal)
    added = plan["added_claims"]
    assert isinstance(added, list)
    new_cid = added[0]["claim_id"]
    rules_added = out["rules_added"]
    assert isinstance(rules_added, list)
    assert len(rules_added) == 1
    assert set(rules_added[0]["claim_ids"]) == {new_cid}
    assert old_cid not in rules_added[0]["claim_ids"]
    # Repeat refinement adds nothing.
    plan2 = plan_refinement(r.load_thesis(t.thesis_id), proposal)
    out2 = apply_refinement(r, t.thesis_id, plan2, proposal)
    rules_added2 = out2["rules_added"]
    assert isinstance(rules_added2, list)
    assert rules_added2 == []
    # A disabled rule does not count as coverage and stays untouched.
    added_id = rules_added[0]["rule_id"]
    raw = load_raw_yaml(tmp_path / "theses" / t.slug / "watch.yaml")
    rules = raw["rules"]
    assert isinstance(rules, list)
    for rule in rules:
        assert isinstance(rule, dict)
        if rule["rule_id"] == added_id:
            rule["enabled"] = False
    atomic_write_yaml(tmp_path / "theses" / t.slug / "watch.yaml", raw, tmp_path / "theses")
    thesis2 = r.load_thesis(t.thesis_id)
    proposal3 = IntakeProposal.from_dict(
        {
            "user_thesis": thesis2.user_thesis,
            "scope": "NVDA",
            "claims": [{"statement": "NVDA automotive demand grows"}],
        },
        "<test>",
    )
    plan3 = plan_refinement(thesis2, proposal3)
    out3 = apply_refinement(r, t.thesis_id, plan3, proposal3)
    added3 = plan3["added_claims"]
    assert isinstance(added3, list)
    newest_cid = added3[0]["claim_id"]
    rules_added3 = out3["rules_added"]
    assert isinstance(rules_added3, list)
    assert len(rules_added3) == 1
    assert {new_cid, newest_cid} <= set(rules_added3[0]["claim_ids"])
    disabled = next(x for x in r.load_watch_rules(t.thesis_id) if x.rule_id == added_id)
    assert disabled.enabled is False and set(disabled.claim_ids) == {new_cid}


def test_live_trigger_pi_reaches_search_web_while_historical_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import tools as tools_mod
    from app.policy import Capability, RequestContext

    r, t = _make(tmp_path)
    full = r.load_thesis(t.thesis_id)
    trig = r.create_trigger(
        t.thesis_id,
        claim_ids=[full.claims[0].claim_id],
        canonical_refs=["ev:m"],
        summary="s",
        summary_origin="deterministic",
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_search(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, dict(kwargs)))
        return {"results": list[object]()}

    monkeypatch.setattr(tools_mod.exa_client, "search", _fake_search)

    def _write(tid: str, trig_id: str, run_id: str) -> None:
        ctx = RequestContext(
            principal_id="test", capabilities=frozenset({Capability.RESEARCH}), data_root=tmp_path, as_of=None
        )
        got = tools_mod.execute_tool("search_web", {"query": "NVDA news"}, "test-model", context=ctx)
        assert got.get("error_type") != "pit_unsafe_tool"
        assert calls != []
        r.append_journal_entry(tid, {"title": "t", "body": "ok", "trigger_id": trig_id, "run_id": run_id})

    fake = _kernel(monkeypatch, write=_write)
    out = run_trigger(r, t.thesis_id, trig.trigger_id, known_at=T2)
    assert out.processed and fake.calls == 1
    assert fake.last_as_of is None
    n = len(calls)
    hist = RequestContext(
        principal_id="test", capabilities=frozenset({Capability.RESEARCH}), data_root=tmp_path, as_of=T2
    )
    blocked = tools_mod.execute_tool("search_web", {"query": "NVDA news"}, "test-model", context=hist)
    assert blocked.get("error_type") == "pit_unsafe_tool"
    assert len(calls) == n


def test_live_tick_uses_current_state_with_cutoff_bounded_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import tools as tools_mod
    from app.policy import Capability, RequestContext

    r = ThesisRepository(tmp_path / "thesis")
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at=T0)
    tid = t.thesis_id
    cid = r.load_thesis(tid).claims[0].claim_id
    r.apply_research_result(
        tid,
        {
            "watch_add": [
                {
                    "rule_id": "rule:1",
                    "rule_type": "new_filing",
                    "enabled": True,
                    "support_status": "supported",
                    "support_reason": "",
                    "claim_ids": [cid],
                    "expression_ids": [],
                }
            ]
        },
        "",
        effective_at=T0,
    )
    r.apply_research_result(
        tid,
        {
            "evidence_refs": [
                {"canonical_ref": "ev:seed:T1", "summary": "NVDA 10-K notes steady demand", "known_at": T1}
            ]
        },
        "run:seed",
        effective_at=T1,
    )
    r.apply_research_result(
        tid, {"claim_updates": [{"claim_id": cid, "status": "challenged"}]}, "run:challenge", effective_at=T4
    )
    r.apply_research_result(
        tid,
        {
            "watch_add": [
                {
                    "rule_id": "rule:current",
                    "rule_type": "new_filing",
                    "enabled": True,
                    "support_status": "supported",
                    "support_reason": "",
                    "claim_ids": [cid],
                    "expression_ids": [],
                }
            ]
        },
        "run:watch",
        effective_at=T4,
    )
    src = _Src([_ev("ev:live", known_at=T1)])
    web_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_search2(*args: object, **kwargs: object) -> dict[str, object]:
        web_calls.append((args, dict(kwargs)))
        return {"results": list[object]()}

    monkeypatch.setattr(tools_mod.exa_client, "search", _fake_search2)

    def _write(live_tid: str, live_trig: str, run_id: str) -> None:
        live_ctx = RequestContext(
            principal_id="test", capabilities=frozenset({Capability.RESEARCH}), data_root=tmp_path, as_of=None
        )
        got = tools_mod.execute_tool("search_web", {"query": "NVDA news"}, "test-model", context=live_ctx)
        assert got.get("error_type") != "pit_unsafe_tool"
        res = tools_mod.execute_tool(
            "thesis_journal",
            {"id": live_tid, "body": "live review notes challenged claim", "trigger_id": live_trig, "run_id": run_id},
            "test-model",
            context=live_ctx,
        )
        assert "error" not in res

    fake = _kernel(monkeypatch, write=_write)
    res = tick(r, tid, {"sec_filings": src}, known_at=T1)
    assert fake.calls == 1 and len(res.triggers_created) == 1
    assert T1 in src.cutoffs
    prompt = fake.prompts[0]
    assert "challenged" in prompt and "rule:current" in prompt
    assert "CURRENT THESIS STATE" in prompt
    assert f"TRIGGER DATA CUTOFF: {T1}" in prompt
    assert "THESIS STATE AS OF" not in prompt
    assert "All sections above are point-in-time as of known_at" not in prompt
    assert fake.last_as_of is None
    assert web_calls != []
    trig = next(t for t in r.load_triggers(tid) if t.trigger_id == res.triggers_created[0])
    event_known_at = (trig.metadata or {}).get("event_known_at", T1)
    assert isinstance(event_known_at, str)
    assert event_known_at <= T1
    run_id = res.runs[0].run_id
    assert run_id and run_id in prompt
    jdir = r.dir_for_thesis(tid) / "journal"
    texts = [p.read_text(encoding="utf-8") for p in sorted(jdir.glob("*.md"))]
    assert texts != [] and any(run_id in b for b in texts)
    assert any("created_at:" in b for b in texts)
    assert not any(f"known_at: {T1}" in b for b in texts)
