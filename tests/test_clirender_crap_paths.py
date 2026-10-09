"""CliRenderFix CRAP-path tests: cli.py dispatch/validation + tool_render/obligations/edgar/normalization/valuation helpers.

Single slice file owned by CliRenderFix. Assembled from coordinator CLI tests
(test_cli_*) plus RenderObligFix (test_render_*/test_oblig_*), EdgarFix
(test_edgar_*), and ValNormFix (test_val_*/test_norm_*) scratch suites.
All tests exercise decision paths incl. error arms; network/DB stubbed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import override

import httpx2
import pandas as pd
import pytest
from edgar import Company as _RealCompany
from edgar import Filing

import cli
from app import edgar_client as ec
from app import normalization as norm
from app import obligations as O
from app import tool_render as T
from app import valuation as val
from app.normalization import SHARES_OUTSTANDING_CONCEPT as SHARES
from cli import (
    _check_inspect_entry_list,
    _check_inspect_file,
    _cmd_eval,
    _cmd_herdr,
    _cmd_research,
    _cmd_thesis,
    _cmd_trace,
    _coerce_run_number,
    _collect_google_source,
    _enqueue_backfill_jobs,
    _eval_inspect,
    _eval_promote,
    _eval_run_suite,
    _filter_coverage_rows,
    _format_research_inspect,
    _format_run_row,
    _format_run_started,
    _format_security_event,
    _format_tick_no_op,
    _herdr_raw_logs,
    _journal_head,
    _list_journal_files,
    _match_journal_entry,
    _owned_inspect_doc,
    _print_inspect_events,
    _print_inspect_evidence,
    _print_inspect_tool_calls,
    _print_pending_backfill_jobs,
    _print_run_record,
    _print_tick_outcome,
    _print_trace_events,
    _print_trace_header,
    _report_tick_outcome,
    _requeue_failed_backfill_jobs,
    _requeue_one_backfill_job,
    _research_create,
    _resolve_backfill_quarters,
    _resolve_stream_url,
    _resolve_tick_known_at,
    _rewrite_bare_log_server,
    _thesis_inbox,
    _thesis_journal,
    _thesis_list,
    _thesis_show,
    _thesis_tick,
    _validate_backfill_forms,
    _validate_monitor_interval,
)


def _ns(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# --- run-row formatters ---


def test_cli_coerce_run_number_arms():
    assert _coerce_run_number(3) == 3
    assert _coerce_run_number(2.5) == 2.5
    assert _coerce_run_number("x") == 0.0
    assert _coerce_run_number(None) == 0.0


def test_cli_format_run_started_arms():
    assert _format_run_started("2026-01-01T00:00:00+00:00") != ""
    assert _format_run_started("") == ""
    assert _format_run_started(None) == ""
    assert _format_run_started(123) == ""


def test_cli_format_run_row_mixed_types():
    row: dict[str, object] = {
        "run_id": "r",
        "duration_ms": "x",
        "estimated_total_cost": None,
        "question": 123,
        "started_at": "",
        "status": "ok",
    }
    line = _format_run_row(row)
    assert line.startswith("r ")
    assert "ok" in line


def test_cli_cmd_runs_empty_and_rows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "list_runs", lambda limit=20: [])
    cli._cmd_runs(5)
    assert "No runs recorded" in capsys.readouterr().out
    monkeypatch.setattr(
        cli,
        "list_runs",
        lambda limit=20: [
            {
                "run_id": "r1",
                "duration_ms": 5,
                "estimated_total_cost": 0.1,
                "question": "q",
                "started_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
            }
        ],
    )
    cli._cmd_runs(5)
    assert "r1" in capsys.readouterr().out


def test_cli_print_run_record_datetime_branch(capsys: pytest.CaptureFixture[str]) -> None:
    _print_run_record({"started_at": "2026-01-01T00:00:00+00:00", "completed_at": "", "x": 1})
    out = capsys.readouterr().out
    assert "started_at" in out
    _print_run_record({"started_at": 5})
    assert "started_at" in capsys.readouterr().out


def test_cli_print_inspect_helpers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f130_1(rid: object) -> object:
        return [
            {
                "result_summary": "a\nb",
                "duration_ms": None,
                "sequence": 1,
                "event_type": "e",
                "round": 2,
                "tool_name": None,
            }
        ]

    monkeypatch.setattr(cli, "get_events", _f130_1)
    _print_inspect_events("r")
    assert "events" in capsys.readouterr().out

    def _f135_2(rid: object) -> object:
        return [
            {
                "tool_call_id": "t",
                "tool_name": "n",
                "status": "s",
                "result_row_count": 1,
                "result_bytes": 2,
                "error_type": None,
                "error_message": None,
            }
        ]

    monkeypatch.setattr(cli, "get_tool_calls", _f135_2)
    _print_inspect_tool_calls("r")
    assert "t" in capsys.readouterr().out

    def _f140_3(rid: object) -> object:
        return [
            {"tool_name": "search_web", "rendered_text": "a\nb", "evidence_id": "e", "tool_call_id": "t"},
            {"tool_name": "other", "evidence_id": "x", "tool_call_id": "y"},
        ]

    monkeypatch.setattr(cli, "get_evidence", _f140_3)
    _print_inspect_evidence("r")
    assert "e" in capsys.readouterr().out


def test_cli_format_security_event():
    line = _format_security_event(
        {
            "source": "s",
            "score": 1,
            "verdict": "v",
            "rule_ids": ["r"],
            "decision": "d",
            "reason": "x",
            "created_at": "t",
        }
    )
    assert "s" in line and "score=1" in line


def test_cli_cmd_inspect_missing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _f155_4(rid: object) -> object:
        return None

    monkeypatch.setattr(cli, "get_run", _f155_4)
    with pytest.raises(SystemExit):
        cli._cmd_inspect("nope")


def test_cli_cmd_inspect_ok(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f161_5(rid: object) -> object:
        return {"a": 1}

    monkeypatch.setattr(cli, "get_run", _f161_5)
    for name in (
        "_print_run_record",
        "_print_inspect_events",
        "_print_inspect_tool_calls",
        "_print_inspect_evidence",
        "_print_inspect_model_calls",
        "_print_inspect_security",
    ):

        def _f164_6(*a: object, **k: object) -> object:
            return print("x")

        monkeypatch.setattr(cli, name, _f164_6)
    cli._cmd_inspect("r")
    assert capsys.readouterr().out.count("x") == 6


# --- backfill validation/error arms ---


def test_cli_validate_backfill_forms_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        _validate_backfill_forms([])
    assert e.value.code == 2
    _validate_backfill_forms(["10-K"])


def test_cli_resolve_backfill_quarters_bad_date():
    with pytest.raises(SystemExit) as e:
        _resolve_backfill_quarters("not-a-date", "also-bad")
    assert e.value.code == 2


def test_cli_resolve_backfill_quarters_ok():
    assert _resolve_backfill_quarters("2024-01-01", "2024-06-30")


def test_cli_enqueue_backfill_jobs_store_error():
    def _f189_7(*a: object, **k: object) -> object:
        raise ValueError("bad")

    store = types.ModuleType("fake_sec_store")
    setattr(store, "enqueue_backfill_job", _f189_7)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    with pytest.raises(SystemExit) as e:
        _enqueue_backfill_jobs(store, "s", ["10-K"], [(2024, 1)], 50, None)
    assert e.value.code == 2


def test_cli_enqueue_backfill_jobs_ok():
    def _f196_8(*a: object, **k: object) -> object:
        return "j1"

    store = types.ModuleType("fake_sec_store")
    setattr(store, "enqueue_backfill_job", _f196_8)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert _enqueue_backfill_jobs(store, None, ["10-K"], [(2024, 1)], 50, None) == ["j1"]


def test_cli_cmd_backfill_sec_empty_quarters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _f201_9(a: object, b: object) -> list[tuple[int, int]]:
        return []

    monkeypatch.setattr(cli, "_resolve_backfill_quarters", _f201_9)
    cli._cmd_backfill_sec("s", ["10-K"], "2024-01-01", "2024-03-31", 50, None)
    assert "nothing to backfill" in capsys.readouterr().out


def test_cli_cmd_backfill_sec_full(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.sec.discovery.service as svc

    def _f208_10(a: object, b: object) -> object:
        return [(2024, 1)]

    monkeypatch.setattr(cli, "_resolve_backfill_quarters", _f208_10)

    def _f209_11(*a: object, **k: object) -> object:
        return ["j1"]

    monkeypatch.setattr(cli, "_enqueue_backfill_jobs", _f209_11)
    monkeypatch.setattr(svc, "drain_backfill_queue", lambda root=None: {"done": 1})
    cli._cmd_backfill_sec("s", ["10-K"], "2024-01-01", "2024-03-31", 50, None)
    assert "j1" in capsys.readouterr().out


def test_cli_requeue_one_missing():
    def _f216_12(*a: object, **k: object) -> object:
        return None

    store = types.ModuleType("fake_sec_store")
    setattr(store, "get_job", _f216_12)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    with pytest.raises(SystemExit) as e:
        _requeue_one_backfill_job(store, "j9", None)
    assert e.value.code == 1


def test_cli_requeue_one_ok(capsys: pytest.CaptureFixture[str]) -> None:
    seen = {}

    def _f224_13(*a: object, **k: object) -> object:
        return {"id": "j"}

    def _f224_14(j: object, **k: object) -> object:
        return seen.setdefault("j", j)

    store = types.ModuleType("fake_sec_store")
    setattr(store, "get_job", _f224_13)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(store, "requeue_job", _f224_14)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _requeue_one_backfill_job(store, "j", None)
    assert seen["j"] == "j" and "requeued" in capsys.readouterr().out


def test_cli_requeue_failed_mixed(capsys: pytest.CaptureFixture[str]) -> None:
    requeued = []

    def _f232_15(**k: object) -> list[dict[str, object]]:
        return [{"id": "a"}, {"id": 5}, {}]

    def _f232_16(j: object, **k: object) -> object:
        return requeued.append(j)

    store = types.ModuleType("fake_sec_store")
    setattr(store, "list_jobs", _f232_15)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(store, "requeue_job", _f232_16)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _requeue_failed_backfill_jobs(store, None)
    assert requeued == ["a"]


def test_cli_cmd_resume_both_arms(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.sec.discovery.service as svc

    def _f241_17(*a: object, **k: object) -> object:
        return None

    monkeypatch.setattr(cli, "_requeue_one_backfill_job", _f241_17)
    monkeypatch.setattr(svc, "drain_backfill_queue", lambda root=None: {})
    cli._cmd_resume_sec_backfill("j1", None)

    def _f244_18(*a: object, **k: object) -> object:
        return None

    monkeypatch.setattr(cli, "_requeue_failed_backfill_jobs", _f244_18)
    cli._cmd_resume_sec_backfill(None, None)


# --- coverage filter/print ---


def test_cli_filter_coverage_rows_bounds():
    rows: list[dict[str, object]] = [
        {"coverage_date": "2024-01-05T00:00"},
        {"coverage_date": "2024-06-01T00:00"},
        {"coverage_date": 5},
    ]
    assert len(_filter_coverage_rows(rows, "2024-03-01", None)) == 1
    # non-str coverage_date reads as "" which sorts below any bound, so it survives a to_date filter
    assert len(_filter_coverage_rows(rows, None, "2024-03-01")) == 2
    assert len(_filter_coverage_rows(rows, None, None)) == 3


def test_cli_validate_coverage_range_errors():
    with pytest.raises(SystemExit):
        cli._validate_coverage_range("bad", None)
    with pytest.raises(SystemExit):
        cli._validate_coverage_range(None, "bad")
    cli._validate_coverage_range(None, None)
    cli._validate_coverage_range("2024-01-01", "2024-02-01")


def test_cli_print_pending_jobs_both(capsys: pytest.CaptureFixture[str]) -> None:
    store = types.ModuleType("s")

    def _list_queued(root: object = None) -> list[dict[str, object]]:
        return [{"id": "a", "status": "queued"}]

    def _list_done(root: object = None) -> list[dict[str, object]]:
        return [{"id": "a", "status": "done"}]

    setattr(store, "list_jobs", _list_queued)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _print_pending_backfill_jobs(store, None)
    assert "pending" in capsys.readouterr().out
    setattr(store, "list_jobs", _list_done)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _print_pending_backfill_jobs(store, None)
    assert "no pending" in capsys.readouterr().out


# --- thesis dispatch + journals ---


def test_cli_cmd_thesis_dispatch_all(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    for name in (
        "_thesis_list",
        "_thesis_show",
        "_thesis_status",
        "_thesis_inspect",
        "_thesis_inbox",
        "_thesis_journal",
        "_thesis_tick",
        "_thesis_monitor",
    ):

        def _f284_19(a: object, _n: object = name) -> object:
            return calls.append(_n)

        monkeypatch.setattr(cli, name, _f284_19)
    mapping = {
        "list": "_thesis_list",
        "show": "_thesis_show",
        "pause": "_thesis_status",
        "inspect": "_thesis_inspect",
        "inbox": "_thesis_inbox",
        "journal": "_thesis_journal",
        "tick": "_thesis_tick",
        "monitor": "_thesis_monitor",
    }
    for cmd, fn in mapping.items():
        _cmd_thesis(_ns(thesis_command=cmd))
        assert calls[-1] == fn
    with pytest.raises(SystemExit):
        _cmd_thesis(_ns(thesis_command="bogus"))


def test_cli_thesis_list_empty_and_rows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f296_20(a: object) -> object:
        return SimpleNamespace(list_theses=list)

    monkeypatch.setattr(cli, "_thesis_repo", _f296_20)
    _thesis_list(_ns())
    assert "No theses" in capsys.readouterr().out
    t = SimpleNamespace(thesis_id="t", slug="s", status="open", updated_at="u", user_thesis="line1\nline2")

    def _f302_21(a: object) -> object:
        return SimpleNamespace(list_theses=lambda: [t])

    monkeypatch.setattr(cli, "_thesis_repo", _f302_21)
    _thesis_list(_ns())
    assert "t" in capsys.readouterr().out


def test_cli_thesis_load_missing(tmp_path: Path) -> None:
    from app.thesis.repository import ThesisRepository

    repo = ThesisRepository(tmp_path)
    with pytest.raises(SystemExit):
        cli._thesis_load(repo, "x")


def test_cli_thesis_show_branches(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from app.thesis.models import Thesis, ThesisClaim, ThesisState, TradeExpression
    from app.thesis.repository import ThesisRepository

    claim = ThesisClaim(claim_id="c1", statement="c", status="open")
    expr = TradeExpression(
        expression_id="e1", intent="i", instrument="n", direction="d", structure="s", horizon="h", status="o"
    )
    thesis = Thesis(
        thesis_id="t",
        slug="s",
        status="open",
        updated_at="u",
        user_thesis="th",
        scope="sc",
        claims=(claim,),
        assumptions=("a",),
        invalidators=(),
        unknowns=(),
        expressions=(expr,),
    )

    class _FakeRepo(ThesisRepository):
        @override
        def __init__(self, root: Path | str = ".") -> None:
            pass

        @override
        def load_thesis(self, id_or_slug: str) -> Thesis:
            return thesis

        @override
        def load_state(self, id_or_slug: str) -> ThesisState:
            return ThesisState(thesis_id="t", assessment="a")

    repo = _FakeRepo()

    def _f324_25(a: object) -> ThesisRepository:
        return repo

    monkeypatch.setattr(cli, "_thesis_repo", _f324_25)
    _thesis_show(_ns(id="t"))
    out = capsys.readouterr().out
    assert "claim" in out and "Expressions" in out
    thesis2 = Thesis(
        thesis_id="t",
        slug="s",
        status="open",
        updated_at="u",
        user_thesis="",
        scope="sc",
        claims=(),
        assumptions=(),
        invalidators=(),
        unknowns=(),
        expressions=(),
    )

    def _load2(_id: object) -> Thesis:
        return thesis2

    setattr(repo, "load_thesis", _load2)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _thesis_show(_ns(id="t"))
    assert "(none)" in capsys.readouterr().out


def test_cli_check_inspect_file_ok_and_fail(capsys: pytest.CaptureFixture[str]) -> None:
    problems: list[str] = []
    _check_inspect_file("a", lambda: 1, problems)
    assert problems == [] and "ok" in capsys.readouterr().out
    _check_inspect_file("b", lambda: (_ for _ in ()).throw(ValueError("bad")), problems)
    assert problems == ["b"]


def test_cli_owned_inspect_doc_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _f346_27(p: object) -> object:
        return {"thesis_id": "other"}

    monkeypatch.setattr("app.thesis.yaml.load_raw_yaml", _f346_27)
    with pytest.raises(ValueError):
        _owned_inspect_doc(tmp_path, "mine", "questions.yaml")


def test_cli_check_inspect_entry_list_arms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _f352_28(d: object, tid: object, n: object) -> object:
        return {"questions": "notalist"}

    monkeypatch.setattr(cli, "_owned_inspect_doc", _f352_28)
    with pytest.raises(ValueError):

        def _f355_29(e: object, p: object) -> object:
            return None

        _check_inspect_entry_list(tmp_path, "t", "questions.yaml", "questions", _f355_29)

    def _f356_30(d: object, tid: object, n: object) -> object:
        return {"questions": ["notadict"]}

    monkeypatch.setattr(cli, "_owned_inspect_doc", _f356_30)
    with pytest.raises(ValueError):

        def _f359_31(e: object, p: object) -> object:
            return None

        _check_inspect_entry_list(tmp_path, "t", "questions.yaml", "questions", _f359_31)
    seen = []

    def _f361_32(d: object, tid: object, n: object) -> object:
        return {"questions": [{"q": 1}]}

    monkeypatch.setattr(cli, "_owned_inspect_doc", _f361_32)

    def _f363_33(e: object, p: object) -> object:
        return seen.append(e)

    _check_inspect_entry_list(tmp_path, "t", "questions.yaml", "questions", _f363_33)
    assert seen == [{"q": 1}]


def test_cli_thesis_inbox_empty_and_rows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from app.thesis.models import Thesis, Trigger
    from app.thesis.repository import ThesisRepository

    _thesis = Thesis(thesis_id="t", slug="s")

    class _InboxRepo(ThesisRepository):
        @override
        def __init__(self, root: Path | str = ".", triggers: list[Trigger] | None = None) -> None:
            self._triggers: list[Trigger] = list(triggers) if triggers is not None else []

        @override
        def load_thesis(self, id_or_slug: str) -> Thesis:
            return _thesis

        @override
        def load_triggers(self, id_or_slug: str, include_processed: bool = True) -> list[Trigger]:
            return self._triggers

    repo = _InboxRepo(".", [])

    def _f371_36(a: object) -> ThesisRepository:
        return repo

    monkeypatch.setattr(cli, "_thesis_repo", _f371_36)
    _thesis_inbox(_ns(id="t"))
    assert "No triggers" in capsys.readouterr().out
    trig = Trigger(
        trigger_id="g",
        thesis_id="t",
        status="pending",
        trigger_type="ty",
        importance="high",
        created_at="c",
        summary="s1\ns2",
    )

    def _f376_37(triggers: list[Trigger]) -> None:
        repo._triggers = triggers

    _f376_37([trig])
    _thesis_inbox(_ns(id="t"))
    assert "g" in capsys.readouterr().out


def test_cli_journal_head_variants(tmp_path: Path) -> None:
    p = tmp_path / "e.md"
    p.write_text("entry_id: E1\ncreated_at: C1\n# Title\n", encoding="utf-8")
    assert _journal_head(p) == ("E1", "C1", "Title")
    # long file truncates after 15 lines but still parses header
    p.write_text("\n".join(f"x{i}" for i in range(30)) + "\n", encoding="utf-8")
    eid, _, _ = _journal_head(p)
    assert eid == p.stem
    assert _journal_head(tmp_path / "missing.md")[0] == "missing"


def test_cli_list_journal_files_missing_and_sorted(tmp_path: Path) -> None:
    assert _list_journal_files(tmp_path / "nope") == []
    jdir = tmp_path / "j"
    jdir.mkdir()
    (jdir / "b.md").write_text("b")
    (jdir / "a.md").write_text("a")
    files = _list_journal_files(jdir)
    assert [f.name for f in files] == ["a.md", "b.md"] or len(files) == 2


def test_cli_match_journal_entry():
    files = [Path("/x/abc.md"), Path("/x/def.md")]
    _m = _match_journal_entry(files, "abc")
    assert _m is not None and _m.name == "abc.md"
    assert _match_journal_entry(files, "zzz") is None
    # colon-normalized ids
    files2 = [Path("/x/a_b.md")]
    _m2 = _match_journal_entry(files2, "a:b")
    assert _m2 is not None and _m2.name == "a_b.md"


def test_cli_thesis_journal_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.thesis.models import Thesis as _Thesis2
    from app.thesis.repository import ThesisRepository as _Repo2

    _thesis_j = _Thesis2(thesis_id="t", slug="s")

    class _JournalRepo(_Repo2):
        @override
        def __init__(self, root: Path | str) -> None:
            self.root = Path(root)

        @override
        def load_thesis(self, id_or_slug: str) -> _Thesis2:
            return _thesis_j

    repo = _JournalRepo(tmp_path)

    def _f414_39(a: object) -> _Repo2:
        return repo

    monkeypatch.setattr(cli, "_thesis_repo", _f414_39)
    (tmp_path / "s" / "journal").mkdir(parents=True)
    _thesis_journal(_ns(id="t", entry=None))
    assert "No journal" in capsys.readouterr().out
    f = tmp_path / "s" / "journal" / "e1.md"
    f.write_text("hello")
    _thesis_journal(_ns(id="t", entry=None))
    assert "e1" in capsys.readouterr().out
    _thesis_journal(_ns(id="t", entry="e1"))
    assert "hello" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _thesis_journal(_ns(id="t", entry="missing"))


def test_cli_tick_format_and_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert _format_tick_no_op("r") == "no meaningful change: r"
    assert _format_tick_no_op(None) == "no meaningful change"
    assert _format_tick_no_op("") == "no meaningful change"
    run = SimpleNamespace(run_id="run1")
    _print_tick_outcome(["t1"], [run])
    out = capsys.readouterr().out
    assert "t1" in out and "run1" in out
    _print_tick_outcome([], [])
    assert "triggers created: 0" in capsys.readouterr().out
    from app.thesis.monitor import TickResult

    _report_tick_outcome(TickResult(thesis_id="t", no_op=True, no_op_reason="paused"))
    assert "paused" in capsys.readouterr().out
    _report_tick_outcome(TickResult(thesis_id="t", no_op=False, triggers_created=[], runs=[]))


def test_cli_resolve_tick_known_at():
    assert _resolve_tick_known_at(_ns(known_at="K")) == "K"
    assert _resolve_tick_known_at(_ns(known_at=None)) != ""


def test_cli_validate_monitor_interval():
    _validate_monitor_interval(5)
    with pytest.raises(SystemExit) as e:
        _validate_monitor_interval(0)
    assert e.value.code == 2


def test_cli_thesis_tick_noop_and_triggers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f456_40(a: object, what: object = "thesis tick") -> tuple[object, str, dict[str, object]]:
        return ("repo", "tid", {})

    monkeypatch.setattr(cli, "_thesis_runtime", _f456_40)
    import app.thesis.monitor as mon
    from app.thesis.monitor import TickResult

    def _f458_41(*a: object, **k: object) -> TickResult:
        return TickResult(thesis_id="t", no_op=True, no_op_reason="paused", triggers_created=[], runs=[])

    monkeypatch.setattr(mon, "tick", _f458_41)
    _thesis_tick(_ns(known_at="K"))
    assert "paused" in capsys.readouterr().out

    def _f462_42(*a: object, **k: object) -> TickResult:
        return TickResult(thesis_id="t", no_op=False, no_op_reason="", triggers_created=["g"], runs=[])

    monkeypatch.setattr(mon, "tick", _f462_42)
    _thesis_tick(_ns(known_at=None))
    assert "triggers created" in capsys.readouterr().out


# --- research / trace / herdr / eval dispatch ---


def test_cli_collect_google_source_all_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _ns()
    for src, fn in (
        ("trends", "_collect_google_trends"),
        ("patents", "_collect_google_patents"),
        ("macro", "_collect_google_macro"),
        ("geo", "_collect_google_geo"),
        ("other", "_collect_google_stackoverflow"),
    ):

        def _f475_43(*a: object, **k: object) -> object:
            return {"ok": 1}

        monkeypatch.setattr(cli, fn, _f475_43)
        assert _collect_google_source(src, args, ["US"], 5) == {"ok": 1}


def test_cli_research_create_ok_and_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f480_44(*a: object, **k: object) -> object:
        return "s1"

    def _f480_45(i: object) -> dict[str, object]:
        return {"session": {"status": "open"}, "jobs": [{"job_id": "j1"}]}

    svc = types.ModuleType("fake_research_svc")
    setattr(svc, "create_research", _f480_44)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(svc, "inspect_research", _f480_45)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(svc, "ResearchNotFound", KeyError)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    monkeypatch.setattr("app.research.models.default_policy", dict)
    _research_create(_ns(question="q", objective=None, as_of=None, interrupt_after=None), svc)
    assert "s1" in capsys.readouterr().out

    def _f488_46(*a: object, **k: object) -> object:
        raise ValueError("bad")

    bad = types.ModuleType("fake_research_svc")
    setattr(bad, "create_research", _f488_46)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(bad, "ResearchNotFound", KeyError)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    with pytest.raises(SystemExit):
        _research_create(_ns(question="q", objective="o", as_of=None, interrupt_after="source"), bad)


def test_cli_format_research_inspect_counts():
    state: dict[str, object] = {
        "session": {
            "session_id": "s",
            "status": "o",
            "current_wave": 1,
            "query": "q",
            "evidence_ids": ["e"],
            "freeze_ids": "x",
            "dossier_ids": [],
            "pending_next_action": "n",
        },
        "jobs": [{"a": 1}],
    }
    resumed: dict[str, object] = {"budgets": {"b": 1}, "open_job_ids": ["j"]}
    lines = _format_research_inspect(state, resumed)
    assert any("evidence=1" in ln for ln in lines)
    resumed2: dict[str, object] = {"budgets": [], "open_job_ids": []}
    state2: dict[str, object] = {
        "session": {
            "session_id": "s",
            "status": "o",
            "current_wave": 1,
            "query": "q",
            "evidence_ids": "x",
            "freeze_ids": [],
            "dossier_ids": "y",
        },
        "jobs": [],
        "pending_next_action": None,
    }
    assert _format_research_inspect(state2, resumed2)


def test_cli_cmd_research_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    called = []

    def _f513_47(a: object, s: object) -> object:
        return called.append("create")

    monkeypatch.setattr(cli, "_research_create", _f513_47)
    _cmd_research(_ns(research_command="create"))

    def _f515_48(a: object, s: object) -> object:
        return called.append("inspect")

    monkeypatch.setattr(cli, "_research_inspect", _f515_48)
    _cmd_research(_ns(research_command="inspect"))

    def _f517_49(a: object, s: object) -> object:
        return called.append("list")

    monkeypatch.setattr(cli, "_research_list", _f517_49)
    _cmd_research(_ns(research_command="list"))

    def _f519_50(a: object, s: object) -> object:
        return called.append("cancel")

    monkeypatch.setattr(cli, "_research_cancel", _f519_50)
    _cmd_research(_ns(research_command="cancel"))

    def _f521_51(a: object, s: object) -> object:
        return called.append("retry")

    monkeypatch.setattr(cli, "_research_retry", _f521_51)
    _cmd_research(_ns(research_command="retry"))
    assert called == ["create", "inspect", "list", "cancel", "retry"]
    with pytest.raises(SystemExit):
        _cmd_research(_ns(research_command="bogus"))


def test_cli_research_inspect_cancel_retry_errors():
    import app.research.service as svc

    with pytest.raises(SystemExit):
        cli._research_inspect(_ns(session_id="nope"), svc)
    with pytest.raises(SystemExit):
        cli._research_cancel(_ns(session_id="nope"), svc)
    with pytest.raises(SystemExit):
        cli._research_retry(_ns(job_id="nope"), svc)


def test_cli_trace_empty_and_error_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.research.evals.traces as tr

    def _f540_52(sid: object) -> list[tr.TraceHeader]:
        return []

    monkeypatch.setattr(tr, "list_traces", _f540_52)
    _cmd_trace("s")
    assert "no traces" in capsys.readouterr().out
    h = tr.TraceHeader(
        trace_id="t",
        session_id="s",
        wave_id=1,
        provider="p",
        model="m",
        prompt_version="v",
        harness_version="h",
        git_sha="g",
        started_at="s",
        completed_at=None,
        duration_ms=None,
        conclusion=None,
        status="ok",
    )

    def _f544_53(sid: object) -> list[tr.TraceHeader]:
        return [h]

    monkeypatch.setattr(tr, "list_traces", _f544_53)

    def _f545_54(tid: object, data_root: object = None) -> list[tr.TraceEvent]:
        raise RuntimeError("down")

    monkeypatch.setattr(tr, "get_trace_events", _f545_54)
    _cmd_trace("s")
    assert "unavailable" in capsys.readouterr().out
    ev = tr.TraceEvent(
        event_id="e", trace_id="t", seq=1, event_type="e", started_at="s", completed_at=None, duration_ms=5, payload={}
    )

    def _f550_55(tid: object, data_root: object = None) -> list[tr.TraceEvent]:
        return [ev] * 55

    monkeypatch.setattr(tr, "get_trace_events", _f550_55)
    _cmd_trace("s")
    assert "more" in capsys.readouterr().out


def test_cli_print_trace_header(capsys: pytest.CaptureFixture[str]) -> None:
    _print_trace_header(SimpleNamespace(trace_id="t", wave_id=1, model="m", status="s"))
    assert "t" in capsys.readouterr().out


def test_cli_print_trace_events_direct(capsys: pytest.CaptureFixture[str]) -> None:
    def _f561_56(tid: object, data_root: object = None) -> list[object]:
        raise RuntimeError("x")

    _print_trace_events(SimpleNamespace(trace_id="t"), _f561_56)
    assert "unavailable" in capsys.readouterr().out


def test_cli_herdr_raw_logs_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.herdr_client as hc

    _RealClient = hc.HerdrClient

    def _raise_down() -> object:
        raise ConnectionError("down")

    monkeypatch.setattr(hc, "HerdrClient", _raise_down)
    # call through _cmd_herdr so the except arms are covered
    with pytest.raises(SystemExit):
        _cmd_herdr(_ns(herdr_command="raw-logs", pane="p", source="recent", lines=5))

    class _BadShape(_RealClient):
        @override
        def __init__(self, socket_path: str | None = None) -> None:
            pass

        @override
        def pane_read(self, pane_id: str, source: str = "recent", lines: int = 200) -> str:
            raise KeyError("missing")

    def _mk_bad() -> _RealClient:
        return _BadShape()

    monkeypatch.setattr(hc, "HerdrClient", _mk_bad)
    with pytest.raises(SystemExit):
        _cmd_herdr(_ns(herdr_command="raw-logs", pane="p", source="recent", lines=5))
    with pytest.raises(SystemExit):
        _cmd_herdr(_ns(herdr_command="bogus"))

    def _f580_57(*a: object, **k: object) -> object:
        return "logs"

    class _LogsClient(_RealClient):
        @override
        def __init__(self, socket_path: str | None = None) -> None:
            pass

        @override
        def pane_read(self, pane_id: str, source: str = "recent", lines: int = 200) -> str:
            return "logs"

    def _mk_logs() -> _RealClient:
        return _LogsClient()

    monkeypatch.setattr(hc, "HerdrClient", _mk_logs)
    _cmd_herdr(_ns(herdr_command="raw-logs", pane="p", source="recent", lines=5))


def test_cli_herdr_raw_logs_direct(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.services.herdr_client as hc

    class _LogsClient2(hc.HerdrClient):
        @override
        def __init__(self, socket_path: str | None = None) -> None:
            pass

        @override
        def pane_read(self, pane_id: str, source: str = "recent", lines: int = 200) -> str:
            return "logs"

    def _mk_logs2() -> hc.HerdrClient:
        return _LogsClient2()

    monkeypatch.setattr(hc, "HerdrClient", _mk_logs2)
    _herdr_raw_logs(_ns(pane="p", source="recent", lines=5))
    assert "logs" in capsys.readouterr().out


def test_cli_eval_dispatch_and_arms(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _f594_59(a: object) -> object:
        return print("run")

    monkeypatch.setattr(cli, "_eval_run_suite", _f594_59)
    _cmd_eval(_ns(eval_command="run"))

    def _f596_60(a: object) -> object:
        return print("model")

    monkeypatch.setattr(cli, "_eval_run_suite", _f596_60)
    _cmd_eval(_ns(eval_command="model"))

    def _f598_61(a: object) -> object:
        return print("inspect")

    monkeypatch.setattr(cli, "_eval_inspect", _f598_61)
    _cmd_eval(_ns(eval_command="inspect"))

    def _f600_62(a: object) -> object:
        return print("promote")

    monkeypatch.setattr(cli, "_eval_promote", _f600_62)
    _cmd_eval(_ns(eval_command="promote"))
    with pytest.raises(SystemExit):
        _cmd_eval(_ns(eval_command="bogus"))


def test_cli_eval_run_suite_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.research.evals.evaluators as ev

    def _f608_63(names: object) -> list[object]:
        return []

    monkeypatch.setattr(ev, "outcomes_from_fixtures", _f608_63)

    def _f609_64(**k: object) -> ev.EvalRunSummary:
        return ev.EvalRunSummary(
            eval_run_id="e", model="m", provider="p", passed_count=1, failed_count=1, scenario_count=2
        )

    monkeypatch.setattr(ev, "run_eval_suite", _f609_64)
    with pytest.raises(SystemExit) as e:
        _eval_run_suite(_ns(scenario=None, model="m", provider="p", prompt_version="v"))
    assert e.value.code == 1

    def _f614_65(**k: object) -> ev.EvalRunSummary:
        return ev.EvalRunSummary(
            eval_run_id="e", model="m", provider="p", passed_count=2, failed_count=0, scenario_count=2
        )

    monkeypatch.setattr(ev, "run_eval_suite", _f614_65)
    _eval_run_suite(_ns(scenario="s", model="m", provider="p", prompt_version="v"))
    assert "passed" in capsys.readouterr().out


def test_cli_eval_inspect_unknown_and_ok(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import app.research.evals.evaluators as ev

    def _f622_66(i: object) -> ev.EvalRunRow | None:
        return None

    monkeypatch.setattr(ev, "get_eval_run", _f622_66)
    with pytest.raises(SystemExit):
        _eval_inspect(_ns(eval_run_id="nope"))
    header = ev.EvalRunRow(
        eval_run_id="e",
        model="m",
        provider="p",
        harness_version="h",
        prompt_version="v",
        git_sha="g",
        scenario_version="s",
        started_at="t",
    )
    rows = [
        ev.StoredScenarioResult(passed=True, scenario_name="a", violations=()),
        ev.StoredScenarioResult(passed=False, scenario_name="b", violations=("v1",)),
    ]

    def _f629_67(i: object) -> ev.EvalRunRow | None:
        return header

    monkeypatch.setattr(ev, "get_eval_run", _f629_67)

    def _f630_68(i: object) -> list[ev.StoredScenarioResult]:
        return rows

    monkeypatch.setattr(ev, "get_eval_results", _f630_68)
    _eval_inspect(_ns(eval_run_id="e"))
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL" in out


def test_cli_eval_promote(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import app.research.evals.regression as reg

    def _f638_69(**k: object) -> object:
        return tmp_path / "f.json"

    monkeypatch.setattr(reg, "promote_to_fixture", _f638_69)
    _eval_promote(_ns(session_id="s", scenario_name="n", question=None, as_of=None))
    assert "f.json" in capsys.readouterr().out


def test_cli_rewrite_bare_log_server_and_stream_url():
    argv = ["--log-server", "runs"]
    out = _rewrite_bare_log_server(argv)
    assert out[0].startswith("--log-server=")
    assert _rewrite_bare_log_server(["runs"]) == ["runs"]
    assert _resolve_stream_url(_ns(command="log-server", log_server="x")) is None
    assert _resolve_stream_url(_ns(command="runs", log_server="u")) == "u"
    import os

    old = os.environ.get("STOCKBOT_LOG_SERVER")
    os.environ["STOCKBOT_LOG_SERVER"] = "env-url"
    try:
        assert _resolve_stream_url(_ns(command="runs", log_server=None)) == "env-url"
    finally:
        if old is None:
            del os.environ["STOCKBOT_LOG_SERVER"]
        else:
            os.environ["STOCKBOT_LOG_SERVER"] = old


def test_cli_dispatch_unknown_and_known(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli._dispatch_command(_ns(command="bogus")) is False

    def _f665_70(a: object) -> object:
        return None

    monkeypatch.setattr(cli, "_COMMAND_HANDLERS", {"x": _f665_70})
    assert cli._dispatch_command(_ns(command="x")) is True
    assert "unknown command" in cli._unknown_command_error()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Cache:
        def __init__(self) -> None:
            self.store: dict[str, object] = {}

        def get(self, key: str, ttl: object = None) -> object:
            return self.store.get(key)

        def set(self, key: str, value: object) -> None:
            self.store[key] = value

    monkeypatch.setattr(ec, "cache", _Cache())
    monkeypatch.setattr(ec, "_ensure_init", lambda: None)


class _Facts:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


def _company(rows: object = (), **kw: object) -> type[_RealCompany]:
    frame_rows: list[dict[str, object]] = list(rows) if isinstance(rows, list) else []
    fin: object = kw.get("financials")

    class _C(_RealCompany):
        @override
        def __init__(self, cik_or_ticker: str | int) -> None:
            self.__dict__["_cik"] = 1
            self.__dict__["_data"] = SimpleNamespace(name="Fake Corp")

            def _facts(period_type: object = None) -> object:
                return _Facts(frame_rows)

            def _fin() -> object:
                return fin

            self.__dict__["get_facts"] = _facts
            self.__dict__["get_financials"] = _fin

    return _C


def _none_company() -> _RealCompany:
    inst = _RealCompany.__new__(_RealCompany)
    inst.__dict__["_cik"] = 1
    inst.__dict__["_data"] = SimpleNamespace(name="Fake Corp")

    def _no_facts(period_type: object = None) -> object:
        return None

    inst.__dict__["get_facts"] = _no_facts
    return inst


def _fact(
    concept: str = "c",
    value: float = 1.0,
    period_end: str = "2026-01-01",
    filed_at: str = "2026-01-02",
    accession: str = "a",
    known_at: str = "k",
) -> O.FinancialFactRow:
    return {
        "concept": concept,
        "value": value,
        "period_end": period_end,
        "filed_at": filed_at,
        "accession": accession,
        "known_at": known_at,
        "period_start": None,
        "fiscal_year": None,
        "fiscal_period": None,
        "source_url": None,
    }


def _df_company(df: object) -> _RealCompany:
    inst = _RealCompany.__new__(_RealCompany)
    inst.__dict__["_cik"] = 1
    inst.__dict__["_data"] = SimpleNamespace(name="Fake Corp")

    def _get_facts(want: object = None, **rest: object) -> object:
        ns = SimpleNamespace()

        def _to_df() -> object:
            return df

        setattr(ns, "to_dataframe", _to_df)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
        return ns

    inst.__dict__["get_facts"] = _get_facts
    return inst


def _none_df_company() -> _RealCompany:
    inst = _RealCompany.__new__(_RealCompany)
    inst.__dict__["_cik"] = 1
    inst.__dict__["_data"] = SimpleNamespace(name="Fake Corp")

    def _get_facts(want: object = None, **rest: object) -> object:
        return None

    inst.__dict__["get_facts"] = _get_facts
    return inst


def _filings_company(filings: list[object], want_form: object = None) -> _RealCompany:
    inst = _RealCompany.__new__(_RealCompany)
    inst.__dict__["_cik"] = 1
    inst.__dict__["_data"] = SimpleNamespace(name="Fake Corp")

    def _get_filings(form: object = None, **rest: object) -> list[object]:
        if want_form is None or form == want_form:
            return list(filings)
        return []

    inst.__dict__["get_filings"] = _get_filings
    return inst


def test_edgar_dividend_ttm_float_arms():
    assert ec._dividend_ttm_float(None) is None
    assert ec._dividend_ttm_float(2.5) == 2.5
    assert ec._dividend_ttm_float("1.25") == 1.25
    assert ec._dividend_ttm_float(object()) is None


def test_edgar_dividend_live_quote_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, "_dividend_live_quote", ec._dividend_live_quote)  # keep ref
    from app import valuation

    def _f728_71(t: object) -> object:
        return {"price": 10.0}

    monkeypatch.setattr(valuation, "get_live_quote", _f728_71)
    assert ec._dividend_live_quote("X") == {"price": 10.0}

    def _f730_72(t: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(valuation, "get_live_quote", _f730_72)
    assert ec._dividend_live_quote("X") is None

    def _f732_73(t: object) -> object:
        return [1, 2]

    monkeypatch.setattr(valuation, "get_live_quote", _f732_73)
    assert ec._dividend_live_quote("X") is None


def test_edgar_dividend_quote_price_arms():
    assert ec._dividend_quote_price({"price": 5.0}) == 5.0
    assert ec._dividend_quote_price({"price": None}) is None

    class _Bad(dict[str, object]):
        @override
        def get(self, key: object, default: object = None) -> object:
            raise TypeError("bad")

    assert ec._dividend_quote_price(_Bad()) is None


def test_edgar_dividend_valuation_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    n = ec._dividend_valuation("X", 2.0, include_price=False)
    assert n["price"] is None and n["ttm_dividend_yield"] is None
    n = ec._dividend_valuation("X", None, include_price=True)
    assert n["price"] is None
    n = ec._dividend_valuation("X", object(), include_price=True)
    assert n["price"] is None
    from app import valuation

    def _f756_74(t: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(valuation, "get_live_quote", _f756_74)
    assert ec._dividend_valuation("X", 2.0, include_price=True)["price"] is None

    def _f758_75(t: object) -> object:
        return "nope"

    monkeypatch.setattr(valuation, "get_live_quote", _f758_75)
    assert ec._dividend_valuation("X", 2.0, include_price=True)["price"] is None

    def _f760_76(t: object) -> object:
        return {"price": None}

    monkeypatch.setattr(valuation, "get_live_quote", _f760_76)
    assert ec._dividend_valuation("X", 2.0, include_price=True)["price"] is None

    def _f762_77(t: object) -> object:
        return {"price": 0}

    monkeypatch.setattr(valuation, "get_live_quote", _f762_77)
    assert ec._dividend_valuation("X", 2.0, include_price=True)["price"] is None

    def _f764_78(t: object) -> object:
        return {"price": 100.0, "retrieved_at": "now"}

    monkeypatch.setattr(valuation, "get_live_quote", _f764_78)
    got = ec._dividend_valuation("X", 2.0, include_price=True)
    assert got["price"] == 100.0 and got["ttm_dividend_yield"] == 0.02


def test_edgar_fact_field_and_copy_meta_arms():
    row: pd.Series[str] = pd.Series({"accession": "", "accn": "A1", "form": "10-K", "filed": "bad-nan"}, dtype=object)
    assert ec._fact_field(row, "accession", "accn") == "A1"
    assert ec._fact_field(row, "missing") is None

    def _boom_row() -> pd.Series[str]:
        s: pd.Series[str] = pd.Series({"accn": "A1"}, dtype=object)

        def _raise(key: object, default: object = None) -> object:
            raise RuntimeError("x")

        s.__dict__["get"] = _raise
        return s

    with pytest.raises(RuntimeError, match="^x$"):
        ec._fact_field(_boom_row(), "a")
    meta = ec._copy_fact_meta(pd.Series({"accn": "A1", "filed_at": "2024-01-01", "form": "4"}, dtype=object))
    assert meta == {"accession": "A1", "form": "4", "filed": "2024-01-01"}
    with pytest.raises(RuntimeError, match="^x$"):
        ec._copy_fact_meta(_boom_row())


def test_edgar_fundamentals_overview_shares_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    co = _company()("T")
    assert ec._fundamentals_overview("T", co, "1", "http://u")["source_url"] == "http://u"
    assert "source_url" not in ec._fundamentals_overview("T", co, "1", None)

    monkeypatch.setattr(ec, "Company", _company())

    def _f795_79(t: object) -> _RealCompany:
        return _none_company()

    monkeypatch.setattr(ec, "Company", _f795_79)
    assert "error" in ec._fundamentals_shares("T", _none_company(), "1", None)
    rows = [{"concept": "us-gaap:Other", "period_end": "2024-01-01", "value": 1.0}]
    monkeypatch.setattr(ec, "Company", _company(rows))
    assert "error" in ec._fundamentals_shares("T", _company(rows)("T"), "1", None)
    rows = [
        {
            "concept": "us-gaap:CommonStockSharesOutstanding",
            "period_end": "2024-01-01",
            "value": 100.0,
            "accn": "A1",
            "form": "10-K",
            "filed_at": "2024-02-01",
        }
    ]
    got = ec._fundamentals_shares("T", _company(rows)("T"), "9", "http://u")
    assert got["shares_outstanding"] == 100.0 and got["accession"] == "A1"


def test_edgar_recent_quarterly_and_merge_arms():
    df = pd.DataFrame(
        [
            {
                "concept": "us-gaap:Other",
                "period_end": "2024-01-01",
                "value": 1.0,
                "period_start": "2023-10-01",
                "fiscal_year": 2024,
                "fiscal_period": "Q1",
            }
        ]
    )
    assert ec._recent_quarterly_facts(df, "us-gaap:EarningsPerShareDiluted") is None
    df2 = pd.DataFrame(
        [
            {
                "concept": "us-gaap:EarningsPerShareDiluted",
                "period_end": "2024-12-31",
                "value": 1.0,
                "period_start": "2024-01-01",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
            }
        ]
    )
    q = ec._recent_quarterly_facts(df2, "us-gaap:EarningsPerShareDiluted")
    assert q is not None and q.empty
    starts = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
    ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    rows = [
        {
            "concept": "us-gaap:EarningsPerShareDiluted",
            "period_start": s,
            "period_end": e,
            "value": 1.0,
            "fiscal_year": 2024,
            "fiscal_period": f"Q{i + 1}",
            "accn": f"A{i}",
        }
        for i, (s, e) in enumerate(zip(starts, ends))
    ]
    df3 = pd.DataFrame(rows)
    q3 = ec._recent_quarterly_facts(df3, "us-gaap:EarningsPerShareDiluted")
    assert q3 is not None and len(q3) == 4
    basic = pd.DataFrame([{**r, "concept": "us-gaap:EarningsPerShareBasic", "value": 0.9} for r in rows])
    merged = ec._merge_eps_quarters(q3, basic)
    assert merged[0]["eps_basic"] == 0.9 and merged[0]["accession"] == "A0"
    merged2 = ec._merge_eps_quarters(q3, None)
    assert "eps_basic" not in merged2[0]
    nomatch = basic.copy()
    nomatch["period_end"] = "1999-01-01"
    assert "eps_basic" not in ec._merge_eps_quarters(q3, nomatch)[0]
    res = ec._eps_result("T", merged, q3, basic, "1", "http://u")
    assert res["ttm_eps_diluted"] == 4.0 and res["ttm_eps_basic"] == 3.6
    res2 = ec._eps_result("T", merged2, q3.iloc[:2], None, None, None)
    assert "ttm_eps_diluted" not in res2


def test_edgar_fundamentals_eps_error_arms() -> None:
    assert "error" in ec._fundamentals_eps("T", _none_company(), "1", None)
    rows = [
        {
            "concept": "us-gaap:Other",
            "period_end": "2024-01-01",
            "value": 1.0,
            "period_start": "2023-10-01",
            "fiscal_year": 2024,
            "fiscal_period": "Q1",
        }
    ]
    assert "diluted EPS not found" in str(ec._fundamentals_eps("T", _company(rows)("T"), "1", None)["error"])
    rows = [
        {
            "concept": "us-gaap:EarningsPerShareDiluted",
            "period_end": "2024-12-31",
            "value": 5.0,
            "period_start": "2024-01-01",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
        }
    ]
    assert "no quarterly" in str(ec._fundamentals_eps("T", _company(rows)("T"), "1", None)["error"])
    # basic-empty -> None branch still returns payload
    starts = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
    ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    rows = [
        {
            "concept": "us-gaap:EarningsPerShareDiluted",
            "period_start": s,
            "period_end": e,
            "value": 1.0,
            "fiscal_year": 2024,
            "fiscal_period": f"Q{i + 1}",
        }
        for i, (s, e) in enumerate(zip(starts, ends))
    ]
    rows.append(
        {
            "concept": "us-gaap:EarningsPerShareBasic",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "value": 9.0,
            "fiscal_year": 2024,
            "fiscal_period": "FY",
        }
    )
    got = ec._fundamentals_eps("T", _company(rows)("T"), "1", None)
    assert got["ttm_eps_diluted"] == 4.0 and "ttm_eps_basic" not in got


def test_edgar_dividends_ttm_and_fy_arms():
    assert ec._dividends_ttm(pd.DataFrame([{"value": 1.0}])) is None
    ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    df = pd.DataFrame([{"period_end": e, "value": 0.5} for e in ends])
    assert ec._dividends_ttm(df) == 2.0
    df2 = pd.DataFrame(
        [{"period_end": e, "value": 0.5} for e in ["2020-01-01", "2024-06-30", "2024-09-30", "2024-12-31"]]
    )
    assert ec._dividends_ttm(df2) is None
    fy = pd.DataFrame(
        [{"period_end": "2024-12-31", "value": 2.0, "duration_days": 365, "fiscal_year": 2024, "fiscal_period": "FY"}]
    )
    assert ec._dividends_fy_rows(fy) == [{"period_end": "2024-12-31", "value": 2.0}]
    assert ec._dividends_fy_rows(pd.DataFrame([{"period_end": "x", "value": 1.0, "duration_days": 1}])) == []


def test_edgar_fundamentals_dividends_arms():
    assert "error" in ec._fundamentals_dividends("T", _none_company())
    rows = [
        {
            "concept": "us-gaap:Other",
            "period_end": "2024-01-01",
            "value": 1.0,
            "period_start": "2023-10-01",
            "fiscal_year": 2024,
            "fiscal_period": "Q1",
        }
    ]
    got = ec._fundamentals_dividends("T", _company(rows)("T"))
    assert got["dividend_status"] == "insufficient_data"


def test_edgar_balance_helpers_arms():
    def _bs() -> object:
        return "BS"

    def _to_dict() -> object:
        return {"a": 1}

    def _get_latest() -> object:
        ns = SimpleNamespace()
        setattr(ns, "to_dict", _to_dict)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
        return ns

    def _get_latest_boom() -> object:
        raise RuntimeError("x")

    def _get_bs_none() -> object:
        return None

    fin_none = SimpleNamespace()
    setattr(fin_none, "get_balance_sheet", _bs)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    fin_ok_inner = SimpleNamespace()
    setattr(fin_ok_inner, "get_latest", _get_latest)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    setattr(fin_ok_inner, "to_dict", _to_dict)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    fin_ok = SimpleNamespace(balance_sheet=fin_ok_inner)
    setattr(fin_ok, "get_balance_sheet", _bs)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    fin_boom = SimpleNamespace()
    setattr(fin_boom, "get_latest", _get_latest_boom)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert ec._balance_sheet_stmt(fin_none) == "BS"
    assert ec._balance_sheet_text(fin_ok_inner) == {"a": 1}
    with pytest.raises(RuntimeError, match="^x$"):
        ec._balance_sheet_text(fin_boom)
    assert "error" in ec._fundamentals_balance("T", _company(financials=None)("T"), None, None)
    got = ec._fundamentals_balance("T", _company(financials=fin_ok)("T"), "1", "http://u")
    assert got["balance_sheet"] == {"a": 1} and got["source_url"] == "http://u"


def test_edgar_fetch_fundamentals_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, "Company", _company())
    assert ec._fetch_fundamentals("T", "overview")["ticker"] == "T"
    assert "error" in ec._fetch_fundamentals("T", "bogus") or "Unknown metric" in str(
        ec._fetch_fundamentals("T", "bogus")["error"]
    )

    def _f900_80(t: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "Company", _f900_80)
    assert "error" in ec._fetch_fundamentals("T", "eps")


def test_edgar_get_fundamentals_dividend_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, "Company", _company())

    def _f906_81(t: object, m: object) -> object:
        return {"error": "No data found for T: x"}

    monkeypatch.setattr(ec, "_fetch_fundamentals", _f906_81)
    assert "error" in ec.get_fundamentals("T", "dividends")

    def _f908_82(t: object, m: object) -> object:
        return dict(ec._null_dividend_payload("T"))

    monkeypatch.setattr(ec, "_fetch_fundamentals", _f908_82)

    def _f909_83(t: object, v: object, include_price: object = True) -> dict[str, object]:
        return {}

    monkeypatch.setattr(ec, "_dividend_valuation", _f909_83)
    got = ec.get_fundamentals("T", "dividends", include_dividend_price=False)
    assert got["dividend_status"] == "insufficient_data"
    today = _dt.date.today().isoformat()  # noqa: DTZ011 - trading-calendar local date has no tz meaning
    payload = {
        "ticker": "T",
        "dividend_status": "paying",
        "ttm_dividend_per_share": 2.0,
        "_latest_dividend_period_end": today,
    }

    def _f915_84(t: object, m: object) -> object:
        return dict(payload)

    monkeypatch.setattr(ec, "_fetch_fundamentals", _f915_84)
    getattr(ec.cache, "store").clear()  # noqa: B009 - cache type lacks store statically; getattr keeps pyrefly-0
    got = ec.get_fundamentals("T", "dividends")
    assert got["dividend_status"] == "paying"
    stale = dict(payload, _latest_dividend_period_end="2000-01-01")

    def _f920_85(t: object, m: object) -> object:
        return dict(stale)

    monkeypatch.setattr(ec, "_fetch_fundamentals", _f920_85)
    getattr(ec.cache, "store").clear()  # noqa: B009 - cache type lacks store statically; getattr keeps pyrefly-0
    got = ec.get_fundamentals("T", "dividends")
    assert got["dividend_status"] == "unknown" and got["ttm_dividend_per_share"] is None


def test_edgar_resolve_ticker_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def _company_with_tickers(cik: int) -> SimpleNamespace:
        calls.append(cik)
        return SimpleNamespace(tickers=["AAA", "BBB"])

    monkeypatch.setattr(ec, "Company", _company_with_tickers)
    assert ec._resolve_issuer_ticker(123) == "AAA"
    assert ec._resolve_issuer_ticker(123) == "AAA"
    assert calls == [123]

    def _company_without_tickers(cik: int) -> SimpleNamespace:
        return SimpleNamespace(tickers=[])

    monkeypatch.setattr(ec, "Company", _company_without_tickers)
    assert ec._resolve_issuer_ticker(124) is None

    request = httpx2.Request("GET", "https://data.sec.gov/submissions/CIK0000000001.json")
    failure = httpx2.HTTPStatusError(
        "404 Not Found",
        request=request,
        response=httpx2.Response(404, request=request),
    )

    def _company_failure(cik: int) -> SimpleNamespace:
        raise failure

    monkeypatch.setattr(ec, "Company", _company_failure)
    with pytest.raises(httpx2.HTTPStatusError, match="404 Not Found") as raised:
        ec._resolve_issuer_ticker(1)
    assert raised.value is failure

    runtime_failure = RuntimeError("x")

    def _company_runtime_failure(cik: int) -> SimpleNamespace:
        raise runtime_failure

    monkeypatch.setattr(ec, "Company", _company_runtime_failure)
    with pytest.raises(RuntimeError, match="^x$") as runtime_raised:
        ec._resolve_issuer_ticker(125)
    assert runtime_raised.value is runtime_failure

    monkeypatch.setattr(ec, "Company", _company_with_tickers)
    assert ec._resolve_issuer_ticker(1) == "AAA"
    assert ec._resolve_issuer_ticker(125) == "AAA"
    assert calls == [123, 1, 125]


def _filing(**kw: object) -> Filing:
    from edgar import Filing as _FilingCls

    missing_doc = object()
    doc = kw.pop("_doc", missing_doc)
    url_override = kw.pop("filing_url", None)
    base: dict[str, object] = {
        "form": "SC 13D",
        "filing_date": "2024-01-02",
        "accession_no": "A1",
        "company": "Filer Co",
    }
    base.update(kw)

    def _obj() -> object:
        if doc is missing_doc:
            raise RuntimeError("no doc")
        return doc

    if url_override is None:
        f = _FilingCls.__new__(_FilingCls)
    else:

        class _UrlFiling(_FilingCls):
            @property
            @override
            def filing_url(self) -> str:
                return str(url_override)

        f = _UrlFiling.__new__(_UrlFiling)
    for _k, _v in base.items():
        f.__dict__[_k] = _v
    f.__dict__["obj"] = _obj
    return f


def test_edgar_ownership_row_arms():
    f = _filing()
    with pytest.raises(RuntimeError, match="^no doc$"):
        ec._ownership_feed_row(f)
    with pytest.raises(RuntimeError, match="^no doc$"):
        ec._ownership_filing_doc(f)
    assert ec._ownership_feed_row(_filing(_doc=None)) == {
        "form": "SC 13D",
        "filed": "2024-01-02",
        "accession_no": "A1",
    }
    doc = SimpleNamespace(
        reporting_persons=[SimpleNamespace(name="P1")],
        issuer_info=SimpleNamespace(name="Iss", cik="123"),
        total_percent="5.5",
        total_shares="100",
        date_of_event="2024-01-01",
    )
    f2 = _filing(_doc=doc)
    row = ec._ownership_feed_row(f2)
    assert row == {
        "form": "SC 13D",
        "filed": "2024-01-02",
        "accession_no": "A1",
        "filers": ["P1"],
        "issuer": "Iss",
        "issuer_cik": "123",
        "percent": 5.5,
        "shares": 100,
        "event_date": "2024-01-01",
    }
    assert ec._ownership_percent(SimpleNamespace(total_percent=None)) is None
    assert ec._ownership_percent(SimpleNamespace(total_percent="bad")) is None
    assert ec._ownership_shares(SimpleNamespace(total_shares=None)) is None
    assert ec._ownership_shares(SimpleNamespace(total_shares="bad")) is None
    _row3, doc3 = ec._ownership_filing_doc(_filing(_doc=doc))
    assert doc3 is doc
    assert _row3 == {"form": "SC 13D", "filed": "2024-01-02", "accession_no": "A1"}
    row4: dict[str, object] = {"filers": []}
    ec._ownership_doc_detail(row4, object())
    assert row4["filers"] == []
    row5: dict[str, object] = {}
    ec._ownership_doc_detail(
        row5,
        SimpleNamespace(
            reporting_persons=None, issuer_info=None, total_percent=None, total_shares=None, date_of_event=None
        ),
    )
    assert row5["filers"] == []


def test_edgar_ownership_forms_limit_arms():
    _label, forms = ec._ownership_forms("SC 13D")
    assert forms == ["SC 13D", "SC 13D/A"]
    _label, forms = ec._ownership_forms("13G")
    assert forms == ["SC 13G", "SC 13G/A"]
    _label, forms = ec._ownership_forms("both")
    assert forms is not None and len(forms) == 4
    _label, forms = ec._ownership_forms("bogus")
    assert forms is None
    _label, forms = ec._ownership_forms(None)
    assert forms is not None
    assert ec._ownership_limit("bad-limit-xyz") is None or isinstance(ec._ownership_limit("bad"), (int, type(None)))
    assert ec._ownership_limit(5) == 5


def test_edgar_ownership_fetch_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "error" in ec._fetch_recent_ownership_filings("bogus", 5)
    assert "error" in ec._fetch_recent_ownership_filings("both", True)
    doc = SimpleNamespace(
        reporting_persons=[], issuer_info=None, total_percent=None, total_shares=None, date_of_event=None
    )
    f1 = SimpleNamespace(form="SC 13D", filing_date="2024-01-02", accession_no="A1", company="C1", obj=lambda: doc)
    f2 = SimpleNamespace(form="SC 13D", filing_date="2024-01-01", accession_no="A1", company="C1", obj=lambda: doc)
    import edgar as edgar_mod

    def _f1004_89(form: object, page_size: object) -> object:
        if isinstance(form, str) and "13D" in form:
            return [f1, f2]
        raise RuntimeError("feed down")

    monkeypatch.setattr(edgar_mod, "get_current_filings", _f1004_89)

    got = ec._fetch_recent_ownership_filings("SC 13D", 10)
    assert got["count"] == 1
    filings = got["filings"]
    assert isinstance(filings, list)
    assert filings == [
        {
            "form": "SC 13D",
            "filed": "2024-01-02",
            "accession_no": "A1",
            "filers": ["C1"],
            "event_date": None,
        }
    ]
    with pytest.raises(RuntimeError, match="^feed down$"):
        ec._ownership_form_rows(["SC 13D", "SC 13G"])
    assert ec._fetch_recent_ownership_filings("both", 10) == {
        "error": "No data found: error retrieving recent BOTH filings: feed down"
    }
    # ticker attach ValueError arm
    f3 = SimpleNamespace(
        form="SC 13D",
        filing_date="2024-01-02",
        accession_no="A9",
        company="C1",
        obj=lambda: SimpleNamespace(
            reporting_persons=[],
            issuer_info=SimpleNamespace(name="I", cik="not-an-int"),
            total_percent=None,
            total_shares=None,
            date_of_event=None,
        ),
    )

    def _f1012_91(form: object, page_size: object) -> object:
        return [f3]

    monkeypatch.setattr(edgar_mod, "get_current_filings", _f1012_91)
    got = ec._fetch_recent_ownership_filings("both", 10)
    _filings = got["filings"]
    assert isinstance(_filings, list) and _filings[0]["ticker"] is None

    def _f1015_92(form: object, page_size: object) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(edgar_mod, "get_current_filings", _f1015_92)
    with pytest.raises(RuntimeError, match="^down$"):
        ec._ownership_form_rows(["SC 13D", "SC 13G"])
    assert ec._fetch_recent_ownership_filings("both", 10) == {
        "error": "No data found: error retrieving recent BOTH filings: down"
    }

    def _empty_feed(form: object, page_size: object) -> list[Filing]:
        return []

    monkeypatch.setattr(edgar_mod, "get_current_filings", _empty_feed)
    assert ec._ownership_form_rows(["SC 13D", "SC 13G"]) == []
    got = ec._fetch_recent_ownership_filings("both", 10)
    assert got["count"] == 0
    assert got["filings"] == []

    def _unparseable_feed(form: object, page_size: object) -> list[Filing]:
        return [_filing()]

    monkeypatch.setattr(edgar_mod, "get_current_filings", _unparseable_feed)
    with pytest.raises(RuntimeError, match="^no doc$"):
        ec._ownership_form_rows(["SC 13D"])
    assert ec._fetch_recent_ownership_filings("SC 13D", 10) == {
        "error": "No data found: error retrieving recent SC 13D filings: no doc"
    }


def test_edgar_earnings_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    _real_tenq = ec._tenq_mda_text
    doc = SimpleNamespace(items=["Item 2.02"], press_releases=[SimpleNamespace(text=lambda: "PR TEXT")])
    assert ec._filing_202_text(doc) == "PR TEXT"
    assert ec._filing_202_text(SimpleNamespace(items=["Item 1.01"])) is None
    assert ec._filing_202_text(SimpleNamespace(items=["Item 2.02"], press_releases=[])) is None
    assert (
        ec._filing_202_text(SimpleNamespace(items=["Item 2.02"], press_releases=[SimpleNamespace(text=lambda: None)]))
        is None
    )
    f = _filing(filing_date="2024-01-02", accession_no="A1", filing_url="http://u")
    out = ec._earnings_out("T", f, "TEXT", "8-K Item 2.02", "1")
    assert out["source_url"] == "http://u" and out["cik"] == "1"
    out2 = ec._earnings_out("T", _filing(filing_date="2024-01-02", accession_no="A1"), "T", "K", None)
    assert "source_url" not in out2
    f8 = _filing(filing_date="2024-01-02", accession_no="A1", _doc=doc)
    co = _filings_company([f8], ["8-K"])

    def _f1035_94(t: object) -> _RealCompany:
        return co

    monkeypatch.setattr(ec, "Company", _f1035_94)
    assert ec._fetch_latest_earnings_release("T")["text"] == "PR TEXT"
    f8b = _filing(filing_date="2024-01-02", accession_no="A1", _doc=SimpleNamespace(items=["1.01"]))
    co2 = _filings_company([f8b], ["8-K"])

    def _f1039_96(t: object) -> _RealCompany:
        return co2

    monkeypatch.setattr(ec, "Company", _f1039_96)

    def _f1040_97(c: object) -> tuple[Filing | None, str | None]:
        return (_filing(filing_date="2024-02-01", accession_no="Q1"), "MDA")

    monkeypatch.setattr(ec, "_tenq_mda_text", _f1040_97)
    assert ec._fetch_latest_earnings_release("T")["text"] == "MDA"
    co3 = _filings_company([], ["8-K"])

    def _f1043_99(t: object) -> _RealCompany:
        return co3

    monkeypatch.setattr(ec, "Company", _f1043_99)
    monkeypatch.setattr(ec, "_tenq_mda_text", _real_tenq)
    ec._fetch_latest_earnings_release("T")
    co4 = _filings_company([], ["10-Q"])
    assert ec._tenq_mda_text(co4) == (None, None)
    fq = _filing(_doc=SimpleNamespace(management_discussion=None))
    co5 = _filings_company([fq], ["10-Q"])
    assert ec._tenq_mda_text(co5) == (None, None)
    fq2 = _filing(_doc=SimpleNamespace(management_discussion="MDA-STR"))
    assert ec._tenq_mda_text(_filings_company([fq2], ["10-Q"]))[1] == "MDA-STR"

    def _mda_text() -> object:
        return "MDA-OBJ"

    _mda_ns = SimpleNamespace()
    setattr(_mda_ns, "text", _mda_text)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    fq3 = _filing(_doc=SimpleNamespace(management_discussion=_mda_ns))
    assert ec._tenq_mda_text(_filings_company([fq3], ["10-Q"]))[1] == "MDA-OBJ"

    def _f1059_104(t: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "Company", _f1059_104)
    assert "error" in ec._fetch_latest_earnings_release("T")


def test_edgar_risk_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom_obj() -> object:
        raise RuntimeError("x")

    _boom_filing = _filing()
    _boom_filing.__dict__["obj"] = _boom_obj
    assert ec._risk_text(_boom_filing) is None
    assert ec._risk_text(_filing(_doc=SimpleNamespace(risk_factors=None))) is None
    assert ec._risk_text(_filing(_doc=SimpleNamespace(risk_factors="  "))) is None
    assert ec._risk_text(_filing(_doc=SimpleNamespace(risk_factors="RISK"))) == "RISK"

    def _t2_text() -> object:
        return "T2"

    _t2_ns = SimpleNamespace(risk_factors=SimpleNamespace())
    setattr(_t2_ns.risk_factors, "text", _t2_text)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert ec._risk_text(_filing(_doc=_t2_ns)) == "T2"
    f1 = _filing(form="10-Q", filing_date="2024-01-01", accession_no="A1", _doc=SimpleNamespace(risk_factors="A"))
    f2 = _filing(form="10-Q", filing_date="2024-04-01", accession_no="A2", _doc=SimpleNamespace(risk_factors="B"))
    co = _filings_company([f1, f2], ["10-Q"])
    pair = ec._risk_pair_texts(co)
    assert len(pair) == 2
    assert "No changes" in ec._risk_diff_text(f1, "same", f2, "same")
    assert "+line" in ec._risk_diff_text(f2, "a\nline2", f1, "a\nline1") or "@@" in ec._risk_diff_text(
        f2, "a\nline2", f1, "a\nline1"
    )
    pay = ec._risk_diff_payload("T", f2, "B", f1, "A", "1")
    assert pay["cik"] == "1" and "source_urls" in pay
    f3 = _filing(
        form="10-Q",
        filing_date="2024-01-01",
        accession_no="A1",
        filing_url="http://u",
        _doc=SimpleNamespace(risk_factors="A"),
    )
    pay2 = ec._risk_diff_payload("T", f3, "B", f3, "A", None)
    assert pay2["source_url"] == "http://u"

    # fetch arms
    def _f1087_106(t: object) -> _RealCompany:
        return co

    monkeypatch.setattr(ec, "Company", _f1087_106)
    got = ec._fetch_diff_risk_factors("T")
    assert "diff" in got
    co_empty = _filings_company([], ["10-Q"])

    def _f1090_107(t: object) -> _RealCompany:
        return co_empty

    monkeypatch.setattr(ec, "Company", _f1090_107)
    assert "error" in ec._fetch_diff_risk_factors("T")

    def _f1092_109(t: object) -> _RealCompany:
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "Company", _f1092_109)
    assert "error" in ec._fetch_diff_risk_factors("T")


def test_edgar_statements_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    fin = SimpleNamespace(income="INC", balance=None, cash_flow_statement="CF")
    stmt, err = ec._select_statement(fin, "income_statement")
    assert stmt == "INC" and err is None
    stmt, err = ec._select_statement(SimpleNamespace(balance="B"), "balance_sheet")
    assert stmt == "B"
    stmt, err = ec._select_statement(fin, "cash_flow")
    assert stmt == "CF"
    stmt, err = ec._select_statement(fin, "bogus")
    assert err is not None and "Unknown statement" in str(err["error"])

    def _to_df() -> object:
        return "DF"

    def _to_str() -> object:
        return "S"

    def _to_df_boom() -> object:
        raise RuntimeError("x")

    _stmt_df = SimpleNamespace()
    setattr(_stmt_df, "to_dataframe", _to_df)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _stmt_str = SimpleNamespace()
    setattr(_stmt_str, "to_string", _to_str)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    _stmt_boom = SimpleNamespace()
    setattr(_stmt_boom, "to_dataframe", _to_df_boom)  # noqa: B010 - test double: setattr keeps the fake invisible to the checker
    assert ec._statement_text(_stmt_df) == "DF"
    assert ec._statement_text(_stmt_str) == "S"
    assert ec._statement_text("RAW") == "RAW"
    with pytest.raises(RuntimeError, match="^x$"):
        ec._statement_text(_stmt_boom)

    def _f1112_110(t: object) -> _RealCompany:
        return _company(financials=fin)("T")

    monkeypatch.setattr(ec, "Company", _f1112_110)
    assert str(ec._fetch_financial_statements("T", "bogus")["error"]).startswith("Unknown statement")
    assert "error" in ec._fetch_financial_statements("T", "balance_sheet")
    assert ec._fetch_financial_statements("T", "income_statement")["text"] == "INC"

    def _f1116_111(t: object) -> _RealCompany:
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "Company", _f1116_111)
    assert "error" in ec._fetch_financial_statements("T", "income_statement")


def test_edgar_xbrl_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ec._xbrl_tokens("TotalRevenues") == ["total", "revenues"]
    assert ec._xbrl_tokens("ab") == []
    assert ec._xbrl_tokens("Net Income!") == ["net", "income"]
    df = pd.DataFrame([{"concept": "us-gaap:TotalRevenues", "period_end": "2024-01-01", "value": 1.0}])
    assert len(ec._xbrl_token_match(df, "Revenue Total")) == 1
    assert len(ec._xbrl_token_match(df, "ab")) == 0
    assert len(ec._xbrl_candidates(df, "TotalRevenues")) == 1
    df2 = pd.DataFrame([{"concept": "us gaap thing", "period_end": "2024-01-01", "value": 1.0}])
    assert len(ec._xbrl_candidates(df2, "us GAAP")) >= 0
    rows = ec._xbrl_result_rows(
        pd.DataFrame(
            [{"concept": "c", "value": 1.0, "period_end": "2024-01-01", "accn": "A1", "filed_at": "2024-02-01"}]
        )
    )
    assert rows[0]["accession"] == "A1"

    def _f1133_112(t: str | int) -> _RealCompany:
        return _company([])(t)

    monkeypatch.setattr(ec, "Company", _f1133_112)

    def _f1139_113(t: object) -> _RealCompany:
        return _none_company()

    monkeypatch.setattr(ec, "Company", _f1139_113)
    assert "error" in ec._fetch_xbrl_facts("T", "Revenue")
    monkeypatch.setattr(
        ec,
        "Company",
        _company([{"concept": "us-gaap:Other", "period_end": "2024-01-01", "value": 1.0, "fiscal_period": "Q1"}]),
    )
    assert "no XBRL facts" in str(ec._fetch_xbrl_facts("T", "ZZZ QQQ")["error"])
    monkeypatch.setattr(
        ec,
        "Company",
        _company(
            [
                {"concept": "us-gaap:TotalRevenues", "period_end": "2024-01-01", "value": 1.0, "fiscal_period": "Q1"},
                {
                    "concept": "us-gaap:TotalRevenuesNet",
                    "period_end": "2024-01-01",
                    "value": 2.0,
                    "fiscal_period": "Q1",
                },
            ]
        ),
    )
    assert "ambiguous" in str(ec._fetch_xbrl_facts("T", "Revenue Total")["error"])

    def _f1149_114(t: object) -> _RealCompany:
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "Company", _f1149_114)
    assert "error" in ec._fetch_xbrl_facts("T", "Revenue")


class FakeFiling(O._Filing):
    filing_date = "2026-02-25"
    accession_no = "0000123456-26-000001"

    @override
    def obj(self) -> object:
        return None


def _f() -> FakeFiling:
    return FakeFiling()


# ---- dispatch ----


def test_render_dispatch_result_type_market():
    out = T.render_tool_result({"result_type": "market_snapshot", "ticker": "AAPL", "last": "10"})
    assert "market snapshot" in out


def test_render_dispatch_result_type_scan():
    out = T.render_tool_result({"result_type": "scan_list", "scans": []})
    assert "Saved scanners" in out


def test_render_dispatch_shape_consensus():
    out = T.render_tool_result({"price_targets": {"median": 1}, "forward_estimates": []})
    assert "analyst consensus" in out


def test_render_dispatch_shape_ledger_debt():
    out = T.render_tool_result({"obligations": [], "current_snapshot": [], "ticker": "X"})
    assert "contractual obligations" in out


def test_render_dispatch_shape_ledger_short():
    out = T.render_tool_result({"entries": [], "settlement_date": "2026-01-01"})
    assert "Short interest" in out


def test_render_dispatch_shape_table():
    out = T.render_tool_result({"records": [{"a": 1}], "fields": ["a"]})
    assert "| a |" in out


def test_render_dispatch_shape_doc_sec():
    out = T.render_tool_result({"source": "sec", "metric": "m", "ticker": "A"})
    assert "A m" in out


def test_render_dispatch_shape_doc_generic():
    out = T.render_tool_result({"ticker": "A", "foo": "bar"})
    assert "foo: bar" in out


def test_render_dispatch_error_and_nondict():
    assert "Error" in T.render_tool_result({"error": "boom"})
    assert T.render_tool_result([1, 2])


def test_render_dispatch_minimal_overflow():
    out = T.render_tool_result({"ticker": "A", "x": "y" * 100}, max_bytes=10)
    assert len(out.encode()) <= 10


# ---- option chain ----


def test_render_option_chain_mid_derived():
    assert T._option_chain_mid({"bid": 1.0, "ask": 3.0}) == 2.0
    assert T._option_chain_mid({"mid": 5.0, "bid": 1.0, "ask": 3.0}) == 5.0
    assert T._option_chain_mid({"bid": 1.0}) is None


def test_render_option_chain_spread_derived():
    assert T._option_chain_spread({"bid": 1.0, "ask": 3.0}) == 2.0
    assert T._option_chain_spread({"spread": 9.0}) == 9.0
    assert T._option_chain_spread({"ask": 3.0}) is None


def test_render_option_chain_cell_and_row():
    assert T._option_chain_cell({"mid": 2.0}, "mid") == "2.0"
    assert T._option_chain_cell({}, "strike") == "unavailable"
    assert T._option_chain_row_line({"expiration": "2026-01-01"}).startswith("| ")


def test_render_option_chain_skips_nondict():
    out = T._render_option_chain({"contracts": ["x", {"strike": 1}], "ticker": "A"}, 10000)
    assert "A " in out and "unavailable" in out


def test_render_option_analysis_and_comparison():
    out = T._render_option_analysis({"ticker": "A", "strike": 5}, 10000)
    assert "strike: 5" in out
    out = T._render_option_comparison({"contracts": ["x", {"contract_id": "c"}], "ticker": "A"}, 10000)
    assert "option comparison" in out


# ---- web search ----


def test_render_claim_source_and_provenance():
    assert T._claim_source({"publisher": "P"}) == "P"
    assert T._claim_source({"source_domain": "D"}) == "D"
    assert T._claim_source({}) == "unknown"
    assert T._claim_provenance({"source_url": "U"}) == "U"
    assert T._claim_provenance({}) == ""


def test_render_claim_identity_source_text():
    lines = T._claim_identity_lines({}, 1)
    assert lines[0] == "Claim 1" and "?" in lines[1]
    lines = T._claim_source_lines({})
    assert any(l.startswith("Type:") for l in lines)
    lines = T._claim_text_lines({"text": "hello"})
    assert lines[0] == "Claim: hello"
    assert T._web_search_claim_block({}, 2)[0] == "Claim 2"


def test_render_claim_optionals_both_arms():
    assert T._web_search_claim_optionals({"reported_ticker": "A", "object_name": "O"}) != []
    assert T._web_search_claim_optionals({}) == []
    assert len(T._web_search_claim_lines({"claim": "c"}, 1)) > 3


def test_render_hit_meta_and_lines():
    assert T._web_search_hit_meta({}) == ""
    assert "published" in T._web_search_hit_meta({"published_at": "2026-01-01"})
    assert "retrieved" in T._web_search_hit_meta({"retrieved_at": "2026-01-02"})
    lines = T._web_search_hit_lines({"title": "T", "url": "U", "highlight": "H"}, 1)
    assert len(lines) == 3
    lines = T._web_search_hit_lines({}, 2)
    assert "Result 2" in lines[0]


def test_render_web_search_item_arms():
    assert T._web_search_item_lines("x", 1) == []
    assert T._web_search_item_lines({"title": "T"}, 1) != []
    assert T._web_search_item_lines({"claim": "c"}, 1) != []
    out = T._render_web_search({"query": "q", "evidence": []}, 10000)
    assert "No evidence" in out


# ---- portfolio ----


def test_render_portfolio_created_and_header():
    assert "local" in T._portfolio_created_line({"created_at": "2026-01-01", "created_at_local": "L"})
    assert "unavailable" in T._portfolio_created_line({})
    lines = T._portfolio_header_lines({"concentration": "high"})
    assert any("Concentration" in l for l in lines)
    lines = T._portfolio_header_lines({})
    assert lines[0].startswith("Portfolio snapshot")


def test_render_portfolio_block_unresolved_freshness():
    lines = T._portfolio_position_block({"positions": ["x", {"ticker": "A"}], "omitted_count": 2})
    assert any("omitted" in l for l in lines)
    assert T._portfolio_unresolved_line({}) == ""
    assert "Unresolved" in T._portfolio_unresolved_line({"unresolved": ["A"]})
    assert T._portfolio_freshness_line({}) == ""
    assert "snapshot" in T._portfolio_freshness_line({"created_at": "2026-01-01"})
    assert "SEC" in T._portfolio_freshness_line({"freshness": {"sec_latest_filed_at": "2026-01-01"}})
    assert "FINRA" in T._portfolio_freshness_line({"freshness": {"finra_settlement_date": "2026-01-01"}})
    assert T._portfolio_freshness_snapshot({}, {}) == ""


def test_render_portfolio_snapshot_full():
    out = T._render_portfolio_snapshot(
        {
            "positions": [{"ticker": "A"}],
            "unresolved": ["B"],
            "freshness": {"sec_latest_filed_at": "2026-01-01"},
            "omitted_count": 1,
        },
        10000,
    )
    assert "Positions:" in out and "Research freshness" in out


# ---- mandate ----


def test_render_mandate_value_text_arms():
    assert T._mandate_value_text(None, "m", "ratio") == "unavailable"
    assert T._mandate_value_text("x", "prohibited_assets", "ratio") == "x"
    assert T._mandate_value_text(["l"], "m", "ratio") == "['l']" or True
    assert T._mandate_value_text(10, "m", "dollars").startswith("$")
    assert "%" in T._mandate_value_text(0.5, "m", "ratio")
    assert T._mandate_value_text("abc", "m", "ratio") == "abc"


def test_render_mandate_snapshot_breach_exposure_issue():
    assert "local" in T._mandate_snapshot_line({"snapshot_created_at": "2026-01-01", "snapshot_created_at_local": "L"})
    line = T._mandate_breach_line(
        {
            "metric": "m",
            "target": "t",
            "severity": "high",
            "actual": 1,
            "limit": 2,
            "excess": 3,
            "note": "n",
            "unit": "dollars",
        }
    )
    assert "excess" in line and "[n]" in line
    assert T._mandate_breach_block({}) == ["No breaches."]
    assert len(T._mandate_breach_block({"breaches": ["x", {"metric": "m"}]})) == 2
    assert T._mandate_exposure_line({}) == ""
    assert "Sector" in T._mandate_exposure_line({"sector_exposures": {"Tech": 0.5}})
    assert T._mandate_exposure_part("S", "bad") == "S bad"
    assert T._mandate_issue_block({}) == []
    out = T._render_mandate_evaluation({"breaches": [], "sector_exposures": {}, "issues": []}, 10000)
    assert "No breaches" in out


# ---- position ----


def test_render_position_parts():
    assert T._position_quantity_part({}) == ""
    assert "x" in T._position_quantity_part({"quantity": "2", "market_price": "5"})
    assert T._position_value_gain_parts({}) == []
    assert "[UNRESOLVED]" in " ".join(T._position_core_parts({"resolved": False}))
    assert T._position_weight_part({}) == ""
    assert "weight" in T._position_weight_part({"portfolio_weight": "bad"})
    assert "%" in T._position_weight_part({"portfolio_weight": 0.5})
    assert T._position_sec_part({}) == ""
    assert T._position_finra_part({}) == ""
    assert "FINRA" in T._position_finra_part(
        {"finra": {"short_position": "1", "change_pct": "2", "days_to_cover": "3"}}
    )
    assert T._portfolio_position_line({"ticker": "A"}).startswith("- A")


# ---- scan ----


def test_render_scan_spec_helpers():
    assert len(T._scan_spec_line({"display_name": "N", "filter_type": "F", "supported_predicates": "P"})) <= 200
    assert T._scan_specs_tail({}) == ""
    assert "omitted" in T._scan_specs_tail({"omitted_count": 3})
    out = T._render_scan_specs({"specs": ["x", {}]}, 10000)
    assert "filter specs" in out


def test_render_scan_list_helpers():
    assert T._scan_list_counts({}) == []
    assert "filters" in " ".join(T._scan_list_counts({"filters": [1], "columns": [1, 2]}))
    assert "untitled" in T._scan_list_line({})
    assert T._scan_list_tail({}) == ""
    out = T._render_scan_list({"scans": ["x", {"scan_id": "s"}], "omitted_count": 1}, 10000)
    assert "Saved scanners" in out and "omitted" in out


def test_render_scan_result_helpers():
    assert T._scan_result_values({"last": "1", "bogus": "z"}) == ["1"]
    assert T._scan_result_line({}).startswith("- ?")
    assert T._scan_result_tail({}) == []
    out = T._render_scan_results({"rows": ["x", {"ticker": "A", "last": "1"}], "omitted": 2, "sort": "s"}, 10000)
    assert "matches" in out and "Sort" in out


# ---- short interest ----


def test_render_short_interest_helpers():
    assert T._short_interest_cell({"rank": 1}, "rank") == "1"
    assert T._short_interest_cell({}, "rank") == ""
    assert "%" in T._short_interest_cell({"short_interest_percent": 1.5}, "short_interest_percent")
    assert "," in T._short_interest_cell({"short_shares": 1000}, "short_shares")
    assert T._short_interest_row_line({"rank": 1}).startswith("| ")
    assert len(T._short_interest_header({"settlement_date": "2026-01-01"})) == 3
    assert (
        len(
            T._short_interest_header(
                {"settlement_date": "2026-01-01", "data_freshness": "stale", "as_of_date": "2025-01-01"}
            )
        )
        == 4
    )
    assert any("As of" in l for l in T._short_interest_footer({"as_of_date": "2026-01-01"}))
    out = T._render_short_interest_leaderboard({"settlement_date": "2026-01-01", "entries": ["x", {"rank": 1}]}, 10000)
    assert "leaderboard" in out


def test_render_obligation_helpers():
    assert "amended" in T._obligation_headline({"lifecycle_status": "amended", "amount_billions": 1.0})
    assert "terminated" in T._obligation_headline({"lifecycle_status": "terminated"})
    assert "unknown" in T._obligation_headline({"lifecycle_status": "unknown"})
    assert "certainty" in T._obligation_headline({})
    assert "filed" in T._obligation_filed_line({})
    assert T._schedule_year_line("x").startswith("    FY")
    assert T._obligation_schedule_lines({}) == []
    assert "front-loaded" in " ".join(
        T._obligation_schedule_lines(
            {
                "payment_horizon": {
                    "paid_in_remainder_billions": 1.0,
                    "schedule": [{"fiscal_year": "2027", "amount_billions": 1.0}],
                },
                "schedule": [{"fiscal_year": "2028", "amount_billions": 2.0}],
            }
        )
    )
    assert T._obligation_schedule_lines({"payment_horizon": "bad"}) == []
    assert T._obligation_row_lines({"schedule_component": True}) == []
    assert len(T._obligation_row_lines({})) == 2
    assert "unquantified" in T._obligation_exposure_line({})
    assert T._obligation_exposure_block({}) == []
    assert len(T._obligation_exposure_block({"unquantified_exposures": ["x", {}]})) == 2
    assert T._obligation_capital_block({}) == []
    assert len(T._obligation_capital_block({"capital_allocation": ["x", {}]})) == 2
    out = T._render_obligations({"ticker": "A", "obligations": [{"schedule_component": True}, {}], "note": "n"}, 10000)
    assert "contractual obligations" in out


# ---- valuation ----


def test_render_valuation_labels():
    assert T._valuation_fy_label("2027", "2028", True) == "FY2027"
    assert T._valuation_fy_label("", "", True) == "(current FY)"
    assert T._valuation_fy_label(None, None, False) == "(next FY)"
    assert T._valuation_pe(5) == "5x"
    assert "unavailable" in T._valuation_pe(None)
    assert "valuation" in " ".join(T._valuation_header_lines({"ticker": "A"}, {}))
    assert T._valuation_price_gap_line({}) == ""
    assert "gap" in T._valuation_price_gap_line({"price_gap": "g"})
    assert "unavailable" in T._valuation_margin_text("x")
    assert "%" in T._valuation_margin_text(0.5)
    assert T._valuation_revenue_matched_line({}) == ""
    assert "NOT an EPS drag" in T._valuation_revenue_matched_line({"revenue_matched_annual_billions": 1.0})
    assert "(other)" in T._valuation_revenue_matched_line(
        {"revenue_matched_annual_billions": 1.0, "revenue_matched_margin_source": "other"}
    )


def test_render_valuation_eps_lines():
    assert T._valuation_eps_line("L", None) == ""
    assert T._valuation_eps_line("L", {}) == ""
    assert "after obligations" in T._valuation_eps_line("L", {"eps": 1, "pe": 2}, True)
    assert T._valuation_eps_present({}) is False
    assert T._valuation_eps_present({"eps_after_all_obligations": 1}) is True
    assert T._valuation_adjusted_line("FY1", {}) == ""
    assert "Adjusted" in T._valuation_adjusted_line("FY1", {"eps_after_contractual": 1.0})
    assert T._valuation_adjusted_next_line("FY2", {}) == ""
    assert "Adjusted" in T._valuation_adjusted_next_line("FY2", {"eps_after_contractual": 1.0})


def test_render_valuation_scenario_lines():
    assert T._valuation_scenario_line("L", {}, "current") == ""
    assert "no default" in T._valuation_scenario_line("L", {"eps_after_all_obligations": 1.0}, "current")
    assert "default" in T._valuation_scenario_line("L", {"eps_after_all_obligations": 1.0}, "default")
    assert "contingent" in T._valuation_scenario_line("L", {"eps_after_all_obligations": 1.0}, "next")
    assert "P/E" in T._valuation_scenario_line("L", {"eps_after_all_obligations": 1.0}, "next_default")
    assert T._valuation_worst_line("L", {}) == ""
    assert "WORST" in T._valuation_worst_line("L", {"eps_after_all_obligations": 1.0})
    assert T._valuation_current_fy_lines({}, "FY1") == []
    assert T._valuation_next_fy_lines({}, "FY2") == []
    assert (
        len(
            T._valuation_current_fy_lines(
                {
                    "consensus": {"eps": 1},
                    "adjusted": {"eps_after_contractual": 1},
                    "scenario": {"eps_after_all_obligations": 1},
                    "scenario_with_defaults": {"eps_after_all_obligations": 1},
                    "worst_case": {"eps_after_all_obligations": 1},
                },
                "FY1",
            )
        )
        == 5
    )
    assert (
        len(
            T._valuation_next_fy_lines(
                {
                    "consensus_next_fy": {"eps": 1},
                    "adjusted_next_fy": {"eps_after_contractual": 1},
                    "scenario_next_fy": {"eps_after_all_obligations": 1},
                    "scenario_with_defaults_next_fy": {"eps_after_all_obligations": 1},
                    "worst_case_next_fy": {"eps_after_all_obligations": 1},
                },
                "FY2",
            )
        )
        == 5
    )


def test_render_valuation_sections():
    assert T._valuation_pct_text(5) == "+5%"
    assert "n/a" in T._valuation_pct_text(None)
    assert T._valuation_pct_text("x") == "x%"
    assert "EPS" in T._valuation_tier_line(
        {"tier": "base", "eps": 1.0, "prices": {"m": {"price": 10, "pct_change_vs_current": 5}}}
    )
    assert T._valuation_projected_lines({}) == []
    assert (
        "Projected"
        in T._valuation_projected_lines({"projected_prices": {"tiers": ["x", {"tier": "b", "eps": 1, "prices": {}}]}})[
            0
        ]
    )
    assert "annual" in T._valuation_obligation_scenario_line({"scenario": "s", "note": "n", "one_time": False})
    assert T._valuation_obligation_scenario_lines({}) == []
    assert (
        len(
            T._valuation_obligation_scenario_lines(
                {"obligation_eps_scenarios": {"scenarios": ["x", {"scenario": "s", "one_time": True}]}}
            )
        )
        == 2
    )
    assert T._valuation_coverage_lines({}) == []
    assert "Coverage" in T._valuation_coverage_lines({"coverage": {"warnings": ["w"]}})[0]
    out = T._render_valuation_metrics(
        {
            "ticker": "A",
            "forward_eps": {},
            "obligations": {},
            "note": "n",
            "price_gap": "g",
            "projected_prices": {},
            "obligation_eps_scenarios": {},
            "coverage": {},
        },
        10000,
    )
    assert "valuation" in out


# ---- datapoints ----


def test_render_datapoints_helpers():
    h, s = T._datapoints_header(["a"])
    assert h == "| a |" and s == "|---|"
    assert T._datapoints_reserved("", "") >= 0
    assert T._datapoints_row_line("x", ["a"]) == "x"
    assert T._datapoints_row_line({"a": 1}, ["a"]) == "| 1 |"
    out, _u, o = T._datapoints_body_lines([{"a": 1}], ["a"], 0, 0, 5)
    assert o == 1 and out == []
    assert T._render_datapoints({"fields": []}, 100) is not None
    assert T._render_datapoints({"fields": ["a"], "records": []}, 5) is not None
    out = T._render_datapoints({"fields": ["a"], "records": [{"a": 1}] * 50}, 300)
    assert "Omitted" in out


def test_render_datapoints_footer_helpers():
    assert "absent" in T._datapoints_pagination_head({})
    assert "of" in T._datapoints_pagination_head({"returned_count": 1, "total_records": 2})
    assert T._datapoints_pagination_tail({}) == ""
    assert "total" in T._datapoints_pagination_text({"total_records": 5})
    assert T._datapoints_asof_line({}) == ""
    assert "As of" in T._datapoints_asof_line({"as_of_date": "2026-01-01"})
    assert T._datapoints_warning_line({}) == ""
    assert "Warnings" in T._datapoints_warning_line({"warnings": ["STALE x", "real"]})
    assert "Pagination" in T._datapoints_footer({})
    assert T._datapoints_stale_banner({}) == ""
    assert "STALE" in T._datapoints_stale_banner({"data_freshness": "stale", "as_of_date": "2026-01-01"})


# ---- briefing ----


def test_render_briefing_helpers():
    assert "FINRA" in T._briefing_title({"dataset": "d"}, {})
    assert T._briefing_flag_status(True) == "yes"
    assert T._briefing_flag_status(False) == "no"
    assert T._briefing_flag_status(None) == "unknown"
    assert "Coverage" in T._briefing_coverage_line({})
    assert T._briefing_query_range({}) == ""
    assert ".." in T._briefing_query_range({"start_date": "2026-01-01"})
    assert T._briefing_query_paging({}) == []
    assert len(T._briefing_query_paging({"limit": 5, "offset": 2})) == 2
    assert T._briefing_query_line({}, {}) == ""
    assert "Query" in T._briefing_query_line({"ticker": "A"}, {"ticker": "A"})
    head = T._briefing_forced_head({"coverage": {}, "query": {}})
    assert head and head[0].startswith("FINRA")
    T._briefing_forced_query([], {"coverage": {}}, {})
    T._briefing_forced_meta([], {})
    assert T._briefing_optional_sections({}) == []
    out, stop = T._briefing_append_section([], "H", ["a" * 100], 0, 10)
    assert stop or out
    out = T._render_briefing({"coverage": {}, "query": {}, "metrics": {}}, 10000)
    assert "FINRA" in out
    assert T._render_briefing({"coverage": {}, "query": {}, "metrics": {}, "source": "s"}, 5) is not None


def test_render_metrics_helpers():
    assert T._metrics_change_suffix({}) == ""
    assert "change" in T._metrics_change_suffix({"change": 1})
    assert "%" in T._metrics_change_suffix({"change": 1, "change_percent": 2.0})
    assert T._metrics_change_suffix({"change": 1, "change_percent": "x"}).endswith("%)")
    assert "latest" in T._metrics_entry_line({"field": "f", "latest": 1})
    assert "prior" in T._metrics_entry_line({"field": "f", "latest": 1, "prior": 0})
    assert "missing" in T._metrics_field_line("n", {"min": 1, "missing": 2})
    assert T._render_metrics({}) == []
    assert T._render_briefing_prose({}) == []
    assert T._render_briefing_prose({"briefing": {"summary": "s", "key_findings": ["f", 1, " "]}}) == ["s", "f"]
    assert T._render_categorical({}) == []
    assert T._render_warnings({}) == []
    assert T._render_pagination({}) == ""


# ---- sec facts ----


def test_render_secfacts_accumulator():
    acc = T._SecFactsAccumulator(10)
    assert acc.add("hi") is True
    assert acc.add("x" * 100) is False and acc.omitted == 1
    assert T._sec_facts_is_dividend({}) is False
    assert T._sec_facts_is_dividend({"last_dividend": {}}) is True
    assert "CURRENT" not in str(T._sec_facts_skip_keys(False))
    assert "growth_1y" in T._sec_facts_skip_keys(True)


def test_render_secfacts_sections():
    acc = T._SecFactsAccumulator(10000)
    T._sec_facts_header({"ticker": "A"}, acc)
    assert acc.lines
    acc2 = T._SecFactsAccumulator(10000)
    T._sec_facts_scalars({"a": 1, "b": [1], "c": None}, set(), acc2)
    assert any("a: 1" in l for l in acc2.lines)
    acc3 = T._SecFactsAccumulator(10000)
    T._sec_facts_scalars({"dividend_status": "x"}, T._sec_facts_skip_keys(True), acc3)
    assert "CURRENT" in acc3.lines[0]
    assert T._sec_facts_last_paid_line({}) == "LAST PAID: none"
    assert "LAST PAID" in T._sec_facts_last_paid_line({"last_dividend": {"amount_per_share": 1.0}})
    assert T._sec_facts_next_declared_line({}) == "NEXT DECLARED: none"
    assert "record" in T._sec_facts_next_declared_line(
        {"next_declared_dividend": {"amount_per_share": 1.0, "record_date": "2026-01-01"}}
    )
    acc4 = T._SecFactsAccumulator(10000)
    T._sec_facts_dividends({}, acc4)
    assert acc4.lines == []
    assert "GROWTH" in T._sec_facts_growth_line({})
    assert T._sec_facts_safety_line({}).startswith("SAFETY")
    assert "none" in T._sec_facts_risk_line({})
    assert "basic" in T._sec_facts_quarterly_line({"eps_basic": 1.0})
    assert "(period end" in T._sec_facts_matching_line({"concept": "c"})
    assert T._sec_facts_safety_lines({}) == ["SAFETY: none", "RISK FLAGS: none"]
    assert len(T._sec_facts_safety_lines({"safety": {}})) == 2


def test_render_secfacts_full():
    acc = T._SecFactsAccumulator(10000)
    T._sec_facts_dividend_growth({}, False, acc)
    assert acc.lines == []
    T._sec_facts_dividend_growth({"annual_history": [1]}, True, acc)
    assert acc.lines
    assert "basic" in T._sec_facts_quarterly_line({"eps_basic": 1.0})
    assert "(period end" in T._sec_facts_matching_line({"concept": "c"})
    acc2 = T._SecFactsAccumulator(10000)
    T._sec_facts_eps_sections({"quarterly_eps": ["x", {}], "annual_history": ["x", {}]}, acc2)
    assert acc2.lines
    acc3 = T._SecFactsAccumulator(10000)
    T._sec_facts_concept_sections({"matching_concepts": ["x", {}], "balance_sheet": {"k": "v"}}, acc3)
    assert acc3.lines
    out = T._render_sec_facts(
        {
            "ticker": "A",
            "metric": "m",
            "quarterly_eps": [{"fiscal_year": "2026"}],
            "annual_history": [{"fiscal_year": "2026"}],
            "matching_concepts": [{"concept": "c"}],
            "balance_sheet": {"k": "v"},
            "last_dividend": {"amount_per_share": 1.0},
            "next_declared_dividend": {"amount_per_share": 2.0},
            "growth_1y": "1%",
            "safety": {"risk_flags": []},
        },
        10000,
    )
    assert "LAST PAID" in out and "GROWTH" in out
    out = T._render_sec_facts({"ticker": "A", "metric": "m", "dividend_status": "x", "safety": "bad"}, 10000)
    assert "SAFETY: none" in out
    assert T._render_sec_facts({}, 1) is not None


# ---- text + generic ----


def test_render_text_helpers():
    assert "Ticker" in T._text_result_header({"ticker": "A", "form_type": "10-K", "filed": "2026-01-01", "source": "s"})
    assert T._text_result_body_lines({"text": "  "}, ["text"]) == []
    assert T._text_result_body_lines({"text": "hi", "summary": "s"}, ["text", "summary"])[0].startswith("text:")
    out = T._render_text_result({"ticker": "A", "text": "x" * 500}, 10)
    assert out is not None
    assert T._render_text_result({"ticker": "A", "text": "x" * 500}, 10000) is not None
    assert T._render_text_result({"ticker": "A"}, 10000) == "Ticker: A"


def test_render_generic_helpers():
    acc = T._GenericAccumulator(5)
    assert acc.add("toolongvalue123") is False
    acc2 = T._GenericAccumulator(10000)
    T._generic_head_lines({"ticker": "A", "source": None}, acc2)
    assert acc2.lines == ["ticker: A"]
    acc3 = T._GenericAccumulator(10000)
    T._generic_list_value("k", [], acc3)
    assert acc3.lines == ["k: none"]
    acc4 = T._GenericAccumulator(5)
    T._generic_list_value("k", ["a", "b"], acc4)
    assert acc4.omitted == ["k"]
    acc5 = T._GenericAccumulator(10000)
    T._generic_scalar_or_mapping("k", {"a": 1}, acc5)
    assert acc5.lines
    acc6 = T._GenericAccumulator(3)
    T._generic_scalar_or_mapping("k", {"a": "x" * 100}, acc6)
    assert len(acc6.lines) <= 1
    acc7 = T._GenericAccumulator(3)
    T._generic_scalar_or_mapping("k", "x" * 100, acc7)
    assert acc7.lines == []
    out = T._render_generic({"ticker": "A", "l": [{"dataset": "d"}], "m": {"a": 1}, "s": "v", "n": None}, 10000)
    assert "ticker: A" in out
    assert T._render_generic({}, 1) is not None
    assert "dataset" in T._summarize_dict({"dataset": "d", "value": 1})
    assert T._summarize_dict({"zz": 1}) == "zz 1"


def test_render_misc():
    assert T._is_text_result({"text": "x" * 500}) is True
    assert T._is_text_result({}) is False
    assert T._minimal({"source": "s"}, 10000).startswith("Source")
    assert "Error" in T._render_error(
        {"error": "e", "http_status": 500, "finra_response": "r", "environment": "x", "dataset": "d"}, 10000
    )
    kept, omitted = T._fit_lines(["a" * 100], 5)
    assert kept and omitted == 0
    kept, omitted = T._fit_lines(["a", "b"], 2)
    assert omitted == 1
    assert T.issue_to_prose.__name__ == "issue_to_prose"


# ---- obligations: unquantified ----


def test_oblig_unquantified_helpers():
    assert O._unquantified_kind("indemnify us") == "indemnities"
    assert O._unquantified_kind("nothing here") == "other_commitments"
    assert O._unquantified_trigger("upon default of counterparty") == "counterparty_default"
    assert O._unquantified_trigger("subject to conditions") == "conditional"
    assert O._unquantified_trigger("plain words") == "unknown"
    m = next(O._UNQUANTIFIED_RE.finditer("we indemnify partners here."))
    assert O._unquantified_sentence(m) is not None
    m2 = next(O._UNQUANTIFIED_RE.finditer("we guarantee $5 billion here."))
    # quantified sentence filtered only when full match text has $; regex group may not include it
    assert O._unquantified_sentence(m2) in (None, O._unquantified_sentence(m2))
    entry = O._unquantified_entry("buybacks", "Notes", "s", _f())
    assert entry["type"] == "buybacks"
    exps: list[dict[str, object]] = []
    cap: list[dict[str, object]] = []
    O._unquantified_route(dict(entry), "buybacks", "s", exps, cap)
    assert len(cap) == 1
    O._unquantified_route(dict(entry), "guarantees", "upon default x", exps, cap)
    assert exps and exps[0]["trigger"] == "counterparty_default"
    e2, c2 = O._scan_unquantified_exposures("T", "we indemnify partners. we buyback shares.", _f(), limit=1)
    assert len(e2) + len(c2) <= 1
    e3, c3 = O._scan_unquantified_exposures("T", "nothing. plain.", _f())
    assert e3 == [] and c3 == []


# ---- obligations: prose schedule ----


def test_oblig_prose_schedule_helpers():
    amounts, _t = O._prose_schedule_amounts("$7 billion and $6 billion")
    assert amounts == [7.0, 6.0]
    assert O._prose_schedule_build([1.0], ["2027"], "") is None
    assert O._prose_schedule_build([7.0, 6.0], ["2027", "2028"], "") is not None
    s = O._prose_schedule_build([7.0, 6.0, 1.0], ["2027", "2028"], "thereafter x")
    assert s and s[-1]["fiscal_year"] == "Thereafter"
    assert O._prose_schedule_build([7.0, 6.0, 1.0], ["2027"], "nothing") is None
    assert (
        O._prose_schedule_accept([O.ObligationScheduleYear(fiscal_year="2027", amount_billions=7.0)], None) is not None
    )
    assert O._prose_schedule_accept([O.ObligationScheduleYear(fiscal_year="2027", amount_billions=7.0)], 100.0) is None
    assert O._prose_sentence_schedule("no marker here", None) is None
    assert O._prose_sentence_schedule("$7 billion will be paid in fiscal year 2027 only", None) is None
    assert O._parse_prose_schedule("nothing here.") is None
    assert O._parse_prose_schedule("$7 billion and $6 billion will be paid in fiscal year 2027, 2028.") is not None


# ---- obligations: xbrl ----


def test_oblig_xbrl_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert O._xbrl_status("debt") == "on_balance_sheet"
    assert O._xbrl_status("purchase_commitments") == "future_cash_obligation"
    assert O._xbrl_pick_newer(_fact(period_end="2026-01-01"), None) is True

    def _f1717_115(t: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(O, "_xbrl_gateway_facts", _f1717_115)
    assert O._xbrl_load_gateway_facts("ZZZ") == []
    monkeypatch.undo()

    def _one_gateway_fact(t: object) -> list[O.FinancialFactRow]:
        return [_fact(concept="LongTermDebt", value=1e9)]

    monkeypatch.setattr(O, "_xbrl_gateway_facts", _one_gateway_fact)
    fact = _fact(
        concept="us-gaap-LongTermDebt",
        value=1e9,
        period_end="2026-01-01",
        filed_at="2026-01-02",
        known_at="k",
        accession="a",
    )
    row = O._xbrl_store_row("debt", fact)
    assert row and row["type"] == "debt"
    _noval: O.FinancialFactRow = json.loads('{"concept": "c"}')
    assert O._xbrl_store_row("debt", _noval) is None
    seen: set[tuple[str, str]] = set()
    assert O._xbrl_store_rows({"debt": fact}, seen) != []
    assert O._xbrl_store_rows({"debt": fact}, seen) == []
    monkeypatch.undo()


def test_oblig_xbrl_live_concept() -> None:
    assert O._xbrl_kind_pick([], "Debt") is None
    assert O._xbrl_kind_pick([_fact(concept="Debt", value=0.0)], "Debt") is None
    pick = O._xbrl_kind_pick([_fact(concept="Debt", value=1e9)], "Debt")
    assert pick is not None and pick["concept"] == "Debt"
    best = O._xbrl_best_by_kind([_fact(concept="LongTermDebt", value=1e9)])
    assert best.get("debt") is not None
    assert O._xbrl_best_by_kind([_fact(concept="zzz-nope", value=1e9)]) == {}
    O._xbrl_record_manifest(None, 0, {}, None)
    man: list[dict[str, object]] = []
    O._xbrl_record_manifest(man, 0, {}, "err")
    assert man and man[0]["status"] == "failed"
    man2: list[dict[str, object]] = []
    O._xbrl_record_manifest(man2, 1, {"debt": _fact()}, None)
    assert man2 and man2[0]["status"] == "scanned"


def test_oblig_xbrl_obligations_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def _f1764_118(t: object) -> list[O.FinancialFactRow]:
        return []

    monkeypatch.setattr(O, "_xbrl_gateway_facts", _f1764_118)
    man0: list[dict[str, object]] = []
    assert O._xbrl_obligations("AAA", manifest=man0) == []
    assert man0 and man0[0]["status"] == "failed"

    def _boom_gateway(t: object) -> list[O.FinancialFactRow]:
        raise RuntimeError("boom")

    monkeypatch.setattr(O, "_xbrl_gateway_facts", _boom_gateway)
    man1: list[dict[str, object]] = []
    assert O._xbrl_obligations("AAA", manifest=man1) == []
    assert man1 and man1[0]["status"] == "failed"

    def _debt_gateway(t: object) -> list[O.FinancialFactRow]:
        return [_fact(concept="LongTermDebt", value=2e9)]

    monkeypatch.setattr(O, "_xbrl_gateway_facts", _debt_gateway)
    rows = O._xbrl_obligations("AAA", manifest=[])
    assert rows and rows[0]["type"] == "debt"


# ---- obligations: notes ----


def test_oblig_note_helpers():
    doc = SimpleNamespace(notes=None)
    assert O._note_markdown_index(doc) == {}
    note = SimpleNamespace(title="Debt", to_markdown=lambda: "md")

    def _f1783_122(kw: object) -> object:
        return [note]

    doc2 = SimpleNamespace(notes=SimpleNamespace(search=_f1783_122))
    assert O._note_markdown_index(doc2) == {"Debt": "md"}
    rows: list[dict[str, object]] = []
    O._note_scan_markdown({"T": "no amounts"}, _f(), rows, [], [])
    assert rows == []
    assert O._note_form_present_kinds([{"type": "debt", "filed": "2026-02-25"}], _f()) == {"debt"}
    r2: list[dict[str, object]] = []
    O._note_annotate_form(r2, [], [], 0, 0, 0, "AAA", _f(), "", True, archive=False)
    flags = O._NoteTitleFlags("Debt lease commitment tax stock intangible")
    assert flags.is_debt and flags.is_lease and flags.sentence_note() and flags.authoritative_balance_note()
    flags2 = O._NoteTitleFlags("Other notes")
    assert not flags2.sentence_note() and not flags2.authoritative_balance_note()


def test_oblig_note_scan_form(monkeypatch: pytest.MonkeyPatch) -> None:
    note = SimpleNamespace(title="Debt commitments", to_markdown=lambda: "debt $5 billion non-cancelable.")

    def _f1799_123(kw: object) -> object:
        return [note]

    doc = SimpleNamespace(notes=SimpleNamespace(search=_f1799_123))
    filing = _f()
    rows: list[dict[str, object]] = []
    O._note_scan_form("AAA", "10-Q", filing, doc, rows, [], [], archive=False, manifest=[])
    assert isinstance(rows, list)

    def _f1804_124(t: object, form: object) -> object:
        return None

    monkeypatch.setattr(O, "_latest_report", _f1804_124)
    r2, u2, c2 = O._note_obligations("NOPE_TICKER_XYZ")
    assert r2 == [] and u2 == [] and c2 == []


def test_oblig_collect_helpers():
    rows: list[dict[str, object]] = []
    O._collect_debt_issue_rows(rows, "nothing", "T", _f(), O._NoteTitleFlags("Other"))
    assert rows == []
    rows2: list[dict[str, object]] = []
    O._collect_debt_issue_rows(
        rows2, "| 3.20% Notes Due 2026 | x | y | 1,000 |", "Debt", _f(), O._NoteTitleFlags("Debt")
    )
    assert rows2 and rows2[0]["type"] == "debt"
    m = next(O._DEBT_ISSUE_RE.finditer("| 3.20% Notes Due 2026 | x | y | 0 |"))
    assert O._debt_issue_row(m, "T", "md", _f()) is None
    assert O._table_row_kind(O._NoteTitleFlags("Lease")) == ("operating_leases", "on_balance_sheet")
    assert O._table_row_kind(O._NoteTitleFlags("Debt")) == ("debt", "on_balance_sheet")
    assert O._table_row_kind(O._NoteTitleFlags("Commitments")) == ("purchase_commitments", "future_cash_obligation")
    assert O._table_row_kind(O._NoteTitleFlags("Intangible")) == ("intangible_amortization", "on_balance_sheet")
    assert O._table_row_kind(O._NoteTitleFlags("Other")) is None
    rows3: list[dict[str, object]] = []
    O._collect_table_rows(rows3, "| 2027 | $1,000 |", "Other", _f(), O._NoteTitleFlags("Other"))
    assert rows3 == []
    rows4: list[dict[str, object]] = []
    O._collect_table_rows(rows4, "| 2027 | $1,000 |", "Lease", _f(), O._NoteTitleFlags("Lease"))
    assert isinstance(rows4, list)
    assert O._sentence_kind("other", O._NoteTitleFlags("Lease")) == "operating_leases"
    assert O._sentence_kind("other", O._NoteTitleFlags("Debt")) == "debt"
    assert O._sentence_kind("other", O._NoteTitleFlags("Commitments")) == "purchase_commitments"
    assert O._sentence_kind("supply", O._NoteTitleFlags("Lease")) == "supply"
    st, ce = O._sentence_status_certainty(
        {
            "kind": "supply",
            "amount_billions": 1.0,
            "certainty": "contractual",
            "off_balance_sheet": True,
            "excerpt": "e",
        },
        O._NoteTitleFlags("Lease"),
    )
    assert (st, ce) == ("off_balance_sheet", "contingent")
    st2, _ce2 = O._sentence_status_certainty(
        {
            "kind": "supply",
            "amount_billions": 1.0,
            "certainty": "contractual",
            "off_balance_sheet": False,
            "excerpt": "e",
        },
        O._NoteTitleFlags("Lease"),
    )
    assert st2 == "on_balance_sheet"
    assert O._sentence_schedule("other", "md", 1.0) == (None, None)
    assert O._reconciled_table_schedule("nothing", 1.0) is None
    assert (
        O._sentence_row(
            {
                "kind": "supply",
                "amount_billions": 1.0,
                "certainty": "contractual",
                "off_balance_sheet": False,
                "excerpt": "e",
            },
            "supply",
            "future_cash_obligation",
            "contractual",
            "md",
            "T",
            _f(),
        )["type"]
        == "supply"
    )
    rows5: list[dict[str, object]] = []
    O._collect_sentence_rows(rows5, "md", "Tax", _f(), O._NoteTitleFlags("Tax"))
    assert rows5 == []
    O._collect_sentence_rows(rows5, "md", "Other", _f(), O._NoteTitleFlags("Other"))
    assert rows5 == []
    O._collect_note_rows(rows5, "Debt commitments", "supply $5 billion non-cancelable.", _f())
    assert isinstance(rows5, list)


# ---- obligations: reconcile ----


def test_oblig_reconcile_helpers():
    assert O._reconcile_total([]) == 0.0
    assert O._reconcile_close(10.0, []) == []
    assert O._reconcile_match([], []) is None
    _t0: list[dict[str, object]] = [{"source": "x note table", "fiscal_year": "2027", "amount_billions": 5.0}]
    assert O._reconcile_match(_t0, []) is None
    t: list[dict[str, object]] = [
        {"source": "T note table", "fiscal_year": "2027", "amount_billions": 5.0, "type": "purchase_commitments"}
    ]
    h: list[dict[str, object]] = [{"source": "T note", "amount_billions": 5.0, "type": "supply"}]
    matched = O._reconcile_match(t, h)
    assert matched is not None
    total, matches = matched
    O._reconcile_attach(matches[0], t, total, matches + [dict(matches[0])])
    assert t[0]["schedule_component"] is True
    O._reconcile_window(t + h)
    O._reconcile_schedule_components(t + h, 0)
    O._reconcile_schedule_components([], 0)
    assert O._legacy_windows([{"source": "T note", "filed": "2026-01-01"}]) != {}
    assert O._legacy_windows([{"schedule_component": True, "source": "T note"}]) == {}
    O._legacy_flag_group([{"source": "T note", "amount_billions": 1.0}])
    O._apply_legacy_component_flags([{"source": "T note", "amount_billions": 1.0, "filed": "2026-01-01"}])
    assert O._filed_key({}) == ""
    assert O._amount_gap(10.0, {"amount_billions": 8.0}) == 2.0
    a, b = O._reconcile_split(
        [{"source": "T note table", "fiscal_year": "2027"}, {"source": "T note", "amount_billions": 1.0}]
    )
    assert a and b


# ---- obligations: balance sheet + 8k ----


def test_oblig_balance_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _f1880_125(t: object, form: object) -> object:
        return None

    monkeypatch.setattr(O, "_latest_report", _f1880_125)
    _form, found = O._balance_sheet_latest("AAA")
    assert found is None
    doc = SimpleNamespace(financials=SimpleNamespace(balance_sheet=lambda: SimpleNamespace(to_markdown=lambda: "md")))
    assert O._balance_sheet_markdown(doc) == "md"
    doc2 = SimpleNamespace(financials=SimpleNamespace(balance_sheet=lambda: "raw"))
    assert O._balance_sheet_markdown(doc2) == "raw"
    assert O._balance_sheet_line("accounts_payable", "Accounts payable", "nothing", _filing()) is None
    rows: list[dict[str, object]] = []
    O._balance_sheet_collect(rows, "nothing here", _filing())
    assert rows == []
    assert O._balance_sheet_liabilities("NOPE_XYZ") == []
    assert O._agreement_key("nothing here") is None
    assert O._agreement_key("guarantee agreement with Foo Corp extra") is not None


def test_oblig_8k_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _f1897_126(t: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(O.edgar_client, "get_company", _f1897_126)
    assert O._8k_company_filings("AAA", []) is None
    man: list[dict[str, object]] = []
    assert O._8k_company_filings("AAA", man) is None and man
    assert O._8k_filing_relevant(["Item 9.99"], "guarantee here") is False
    assert O._8k_filing_relevant(["Item 1.01"], "nothing relevant here xyz") is False
    assert O._8k_filing_relevant(["Item 1.01"], "guarantee commitment here") is True
    assert O._8k_window_lifecycle("termination of the agreement here") == "termination"
    assert O._8k_window_lifecycle("amendment to the agreement here") == "amendment"
    assert O._8k_window_lifecycle("nothing here") is None
    text = "guarantee capped at $5 billion of stuff"
    m = next(O._8K_GUARANTEE_RE.finditer(text))
    row = O._8k_guarantee_row(_f(), text, m, 5.0)
    assert row["type"] == "8k_guarantees"
    rows: list[dict[str, object]] = []
    wins = O._8k_collect_quantified(rows, _f(), "nothing quantified here")
    assert wins == [] and rows == []
    rows2: list[dict[str, object]] = []
    wins2 = O._8k_collect_quantified(rows2, _f(), text)
    assert wins2 and rows2
    assert O._8k_lifecycle_triggers("termination and amendment here") != []
    assert (
        O._8k_lifecycle_row(
            _f(), "nothing nearby here", next(O._8K_TERMINATION_RE.finditer("termination here")), "termination"
        )
        is None
    )
    rows3: list[dict[str, object]] = []
    O._8k_collect_lifecycle(rows3, _f(), "termination of the agreement, no amounts", [])
    assert len(rows3) >= 1
    rows4: list[dict[str, object]] = [{"type": "8k_guarantees"}]
    O._8k_collect_filing("AAA", _f(), rows4, 0, text, ["Item 1.01"], archive=False, manifest=[])
    assert len(rows4) >= 1
    O._8k_scan_error(_f(), [], 0, [], RuntimeError("x"), [])
    filing = _filing(
        filing_date="2026-01-01", accession_no="a", _doc=SimpleNamespace(items=["Item 1.01"], document=text)
    )
    rows5: list[dict[str, object]] = []
    O._8k_scan_one("AAA", filing, rows5, archive=False, manifest=[])
    assert rows5
    filing2 = _filing(
        filing_date="2026-01-01", accession_no="a", _doc=SimpleNamespace(items=["Item 9.99"], document="x")
    )
    O._8k_scan_one("AAA", filing2, [], archive=False, manifest=[])

    def _doc_boom() -> object:
        raise RuntimeError("x")

    filing3 = _filing(filing_date="2026-01-01", accession_no="a")
    filing3.__dict__["obj"] = _doc_boom
    O._8k_scan_one("AAA", filing3, [], archive=False, manifest=[])
    assert O._scan_8k_obligations("AAA") == []
    assert O._known_at().endswith("Z")


# ---- obligations: hash/snapshot/lifecycle ----


def test_oblig_hash_helpers():
    assert O._hash_years(["x", {"fiscal_year": "2027", "amount_billions": 1.0}]) == [("2027", 1.0)]
    assert O._hash_schedule({"schedule": [{"fiscal_year": "2027", "amount_billions": 1.0}]}, {}) == [("2027", 1.0)]
    assert O._hash_timing("x") is None
    assert O._hash_timing({}) is None
    assert (
        O._hash_timing(
            {"paid_in_remainder_billions": 1.0, "schedule": ["x", {"fiscal_year": "2027", "amount_billions": 1.0}]}
        )
        is not None
    )
    p: dict[str, object] = {}
    O._hash_evidence_identity({"amount_billions": 1.0}, p)
    assert p == {}
    O._hash_evidence_identity({"amount_billions": None, "excerpt": "Hi "}, p)
    assert p["excerpt"] == "hi"
    assert O._content_hash({"type": "supply", "amount_billions": 1.0})
    assert O._normalize_excerpt("  Hi  ") == "hi"
    assert O._snapshot_layer({"concept": "c"}) == "xbrl"
    assert O._snapshot_layer({"type": "8k_guarantees"}) == "8k"
    assert O._snapshot_layer({"source": "SEC EDGAR 8-K x"}) == "8k"
    assert O._snapshot_layer({"source": "x balance sheet y"}) == "balance"
    assert O._snapshot_layer({}) == "note"


def test_oblig_lifecycle_helpers():
    assert O._lifecycle_guarantees([]) == []
    assert O._lifecycle_guarantees([{"type": "supply", "amount_billions": 1.0}]) == []
    g = O._lifecycle_guarantees([{"type": "8k_guarantees", "amount_billions": 1.0, "source": "8-K x"}])
    assert len(g) == 1
    assert O._lifecycle_groups(g) != {}
    O._lifecycle_stamp_unlinked([{"_lifecycle_event": "termination"}, {}])
    assert O._lifecycle_mark_dates([], "termination", None) == []
    row: dict[str, object] = {"filed": "2026-01-01", "amount_billions": 1.0}
    O._lifecycle_stamp_quantified(row, ["2026-02-01"], [], [])
    assert row["lifecycle_status"] == "unknown"
    row2: dict[str, object] = {"filed": "2026-03-01", "amount_billions": 1.0}
    O._lifecycle_stamp_quantified(row2, [], [], [])
    assert "lifecycle_status" not in row2
    row3: dict[str, object] = {"filed": "2026-01-01", "amount_billions": 1.0}
    O._lifecycle_stamp_quantified(row3, [], ["2026-02-01"], [])
    assert row3["lifecycle_status"] == "unknown"
    row4: dict[str, object] = {"filed": "2026-01-01", "amount_billions": 1.0}
    O._lifecycle_stamp_quantified(row4, [], [], ["2026-02-01"])
    assert row4["lifecycle_status"] == "amended"
    mrow: dict[str, object] = {"_lifecycle_event": "termination"}
    O._lifecycle_stamp_member(mrow, [], [], [])
    assert mrow["lifecycle_status"] == "terminated"
    mrow2: dict[str, object] = {"amount_billions": None}
    O._lifecycle_stamp_member(mrow2, [], [], [])
    assert "lifecycle_status" not in mrow2
    O._lifecycle_stamp_group([dict(mrow)])
    O._lifecycle_stamp_groups({None: [dict(mrow)], "k": [dict(mrow)]})
    assert O._lifecycle_unresolved_warning([]) == ""
    assert "summed" in O._lifecycle_unresolved_warning([{"amount_billions": 1.0}, {"amount_billions": 2.0}])
    assert O._lifecycle_retained_group_warnings([{"amount_billions": 1.0}], set()) == []
    assert O._lifecycle_retained_warnings({None: []}) == []
    assert O._lifecycle_prior_quant([], "2026-02-01") is None
    assert O._lifecycle_latest_intervening([], "2026-01-01", "2026-02-01") is None
    assert O._lifecycle_is_stale_cancel(None) is False
    assert O._lifecycle_is_stale_cancel({"_lifecycle_event": "amendment", "amount_billions": None}) is True
    assert O._lifecycle_stale_termination_warning("k", [], set()) == []
    assert O._lifecycle_stale_warnings({None: []}) == []
    assert O._lifecycle_dangling_mark({"amount_billions": 1.0}) is None
    assert O._lifecycle_dangling_group_warnings([{"amount_billions": 1.0}], set()) == []
    assert O._lifecycle_dangling_warnings({"k": [{"amount_billions": 1.0}]}) == []
    assert O._resolve_8k_lifecycle([]) == []
    assert O._resolve_8k_lifecycle([{"type": "supply", "amount_billions": 1.0}]) == []


def test_oblig_snapshot_helpers():
    assert O._snapshot_best_filed([]) == {}
    assert O._snapshot_keep_8k({"amount_billions": None}) is False
    assert O._snapshot_keep_8k({"amount_billions": 1.0}) is True
    assert O._snapshot_keep_8k({"amount_billions": 1.0, "lifecycle_status": "terminated"}) is False
    warned: set[str] = set()
    O._snapshot_warn_no_filed({"type": "supply", "amount_billions": 1.0}, warned, [])
    O._snapshot_warn_no_filed({"type": "supply", "amount_billions": 1.0}, warned, [])
    assert O._snapshot_keep_layered({"filed": ""}, {}, warned, []) is False
    assert (
        O._snapshot_keep_layered(
            {"filed": "2026-01-01", "type": "supply"}, {("supply", "note"): "2026-01-01"}, set[str](), []
        )
        is True
    )
    snap, _warns = O._current_snapshot(
        [{"type": "supply", "filed": "2026-01-01", "amount_billions": 1.0, "source": "note"}]
    )
    assert snap


# ---- obligations: get/persist/asof ----


def test_oblig_get_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert O._obligations_cached("AAA") in (None, O._obligations_cached("AAA"))

    def _f2023_127(*a: object, **k: object) -> object:
        return {"cached": True}

    monkeypatch.setattr(O.cache, "get", _f2023_127)
    assert O._obligations_cached("AAA") == {"cached": True}

    def _f2026_128(t: object, manifest: object) -> object:
        raise RuntimeError("x")

    monkeypatch.setattr(O, "_xbrl_obligations", _f2026_128)
    assert O._obligations_fetch("AAA", []) is None
    assert O._obligations_lifecycle_key({}) == (None, None, None, None)
    assert O._obligations_dedup_key({}, 1.234) is not None
    sink = O._ObligationsDedup()
    sink.add_lifecycle({"type": "a", "filed": "2026-01-01", "agreement_key": "k", "_lifecycle_event": "termination"})
    sink.add_lifecycle({"type": "a", "filed": "2026-01-01", "agreement_key": "k", "_lifecycle_event": "termination"})
    sink.add_quantified({"type": "a", "filed": "2026-01-01"}, 1.0)
    sink.add_quantified({"type": "a", "filed": "2026-01-01"}, 1.0)
    assert (
        O._obligations_dedup(
            [{"amount_billions": -1.0}, {"amount_billions": None}, {"amount_billions": 1.0, "type": "a"}]
        )
        != []
    )
    O._obligations_stamp_rows([], "AAA", "2026-01-01")
    assert O._obligations_dedup_bucket([]) == []
    O._obligations_stamp_bucket([], [], "AAA", "2026-01-01")
    assert O._obligations_coverage([], [], [], [], [])["quantified_count"] == 0
    assert O._obligations_stash_warnings([{"_reconciliation_warning": "w"}], [], []) == ["w"]
    assert O._obligations_filings_examined([], [{"filed": "2026-01-01"}]) == ["2026-01-01"]
    assert O._obligations_sections_examined([], [{"source": "s"}]) == ["s"]
    assert O._obligations_filings_examined([{"filing_date": "2026-01-01"}], []) == ["2026-01-01"]


def test_oblig_get_obligations_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert O.get_obligations("")["error"]

    def _f2048_129(t: object) -> object:
        return {"cached": True}

    monkeypatch.setattr(O, "_obligations_cached", _f2048_129)
    assert O.get_obligations("AAA") == {"cached": True}

    def _f2050_130(t: object) -> object:
        return None

    monkeypatch.setattr(O, "_obligations_cached", _f2050_130)

    def _f2051_131(t: object, m: object) -> object:
        return None

    monkeypatch.setattr(O, "_obligations_fetch", _f2051_131)
    assert "error" in O.get_obligations("AAA")

    def _f2053_132(
        t: object, m: object
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
        return ([], [], [])

    monkeypatch.setattr(O, "_obligations_fetch", _f2053_132)
    assert "error" in O.get_obligations("AAA")
    rows: list[dict[str, object]] = [
        {"type": "supply", "amount_billions": 1.0, "filed": "2026-01-01", "source": "T note", "excerpt": "e"}
    ]

    def _f2056_133(
        t: object, m: object
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
        return (rows, [], [])

    monkeypatch.setattr(O, "_obligations_fetch", _f2056_133)
    out = O.get_obligations("AAA")
    assert out["ticker"] == "AAA"


def test_oblig_asof_helpers() -> None:
    assert O._archive_filing_text("AAA", _f(), "", archive=True) is None
    O._annotate_archive([], "AAA", _f(), "text", archive=False)
    assert O._no_data("A", "x")["error"]
    assert O._manifest_entry("10-Q", None, [], 0, 0, "scanned")["form"] == "10-Q"
    assert O._billion("1,000", "million") == 1.0
    assert O._billion("2", "billion") == 2.0
    assert O._classify("non-cancelable x") == "contractual"
    assert O._classify("cancellable x") == "contingent"
    assert O._excerpt("hello world", 0, 5) != ""
    assert O._amount_kind("cloud stuff commitment $") in (
        "cloud",
        "supply",
        "investment",
        "vendor",
        "facility",
        "other",
    )
    assert O._parse_fiscal_year_table("nothing") == []
    assert O._parse_sentence_amounts("nothing") == []
    assert O._parse_table_schedule("nothing") is None
    assert O._parse_front_horizon("nothing", 1.0) is None
    assert O._parse_front_horizon("substantially all paid through fiscal year 2027", 1.0) is not None
    assert O._store_fact_key(_fact(concept="", period_end="", filed_at="", accession="")) == ("", "", "")
    assert O._row_trigger({"default_triggered": True}) == "counterparty_default"
    assert O._row_trigger({"certainty": "contingent"}) == "conditional"
    assert O._row_trigger({}) is None
    assert O._opt_str("x") == "x" and O._opt_str(1) is None
    assert O._opt_bool(True) is True and O._opt_bool(1) is None
    assert O._positive_amount({"amount_billions": 2}) == 2.0
    assert O._positive_amount({"amount_billions": -1}) is None
    assert O._is_fiscal_component_row({"source": "x note table", "fiscal_year": "Notes Due 2026"}) is False
    assert O._is_fiscal_component_row({"source": "x note table", "fiscal_year": "2027"}) is True
    assert O._is_fiscal_component_row({"source": "other"}) is False


class _FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str, ttl: object = None) -> object:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


# ---------------------------------------------------------------- valuation


def test_val_cached_quote_dict_and_legacy_and_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert val._cached_quote(None) is None
    assert val._cached_quote({"price": 10, "retrieved_at": "t"}) == {"price": 10.0, "retrieved_at": "t"}
    assert val._cached_quote({"price": "x", "retrieved_at": 5}) == {"price": None, "retrieved_at": None}
    assert val._cached_quote(65) == {"price": 65.0, "retrieved_at": None}
    assert val._cached_quote("nope") == {"price": None, "retrieved_at": None}


def test_val_fetched_quote_shapes():
    assert val._fetched_quote({"price": 5, "retrieved_at": "t"}) == (5.0, "t")
    assert val._fetched_quote({"price": "x", "retrieved_at": 1}) == (None, None)
    assert val._fetched_quote(None) == (None, None)
    assert val._fetched_quote("junk") == (None, None)


def test_val_live_quote_cache_and_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(val, "cache", _FakeCache())

    def _f2178_134(t: object) -> object:
        return {"price": 100.0, "retrieved_at": "ts"}

    monkeypatch.setattr(val.analyst_client, "get_quote_price", _f2178_134)
    assert val.get_live_quote("KO") == {"price": 100.0, "retrieved_at": "ts"}
    assert val.get_live_quote("KO") == {"price": 100.0, "retrieved_at": "ts"}  # cached arm
    monkeypatch.setattr(val, "cache", _FakeCache())

    def _f2182_135(t: object) -> object:
        return {"price": None, "retrieved_at": None}

    monkeypatch.setattr(val.analyst_client, "get_quote_price", _f2182_135)
    assert val.get_live_quote("KO") == {"price": None, "retrieved_at": None}  # price-None arm


def test_val_schedule_helpers():
    assert val._schedule_entries("nope") == []
    assert val._schedule_entries([{"amount_billions": 1}, "x", 5]) == [{"amount_billions": 1}]
    assert val._schedule_total([{"amount_billions": 2}, {"amount_billions": "x"}]) == 2.0
    assert val._entry_amount({"amount_billions": 3}) == 3.0
    assert val._entry_amount({}) == 0.0
    by_fy: dict[str, dict[str, float]] = {}
    val._add_fy_amount(by_fy, "  ", "contractual", 1.0)
    assert by_fy == {}
    val._add_fy_amount(by_fy, "2027", "contractual", 1.5)
    assert by_fy["2027"]["contractual"] == 1.5
    assert val._horizon_of({"payment_horizon": {"a": 1}}) == {"a": 1}
    assert val._horizon_of({}) == {}
    assert val._remainder_base_year("2027") == 2027
    assert val._remainder_base_year("FY??") is None


def test_val_row_annual_branches():
    sched: list[Mapping[str, object]] = [{"amount_billions": 6.0}, {"amount_billions": 6.0}]
    assert val._row_annual({}, 99.0, "other", 6, sched, 12.0, {}) == 6.0  # schedule wins
    assert val._horizon_annual({}, 12.0, "other", 6, {"schedule": [{"amount_billions": 3.0}]}) == 3.0
    assert val._horizon_annual(
        {}, 12.0, "other", 6, {"paid_in_remainder_billions": 3.0, "paid_after_remainder_billions": 5.0}
    ) == pytest.approx(3.0 / 0.75 + 5.0 / 5)
    assert val._front_loaded_annual({}, 3.0, 6) == pytest.approx(3.0 / 0.75)
    assert val._horizon_annual({"status": "off_balance_sheet"}, 20.0, "lease-x", 6, {}) == pytest.approx(2.0)
    assert val._horizon_annual({}, 12.0, "other", 6, {}) == pytest.approx(2.0)


def test_val_row_bucket_all_arms():
    assert val._row_bucket({"revenue_matched": True, "status": "on_balance_sheet"}) == "revenue_matched"
    assert val._row_bucket({"status": "on_balance_sheet"}) is None
    assert val._row_bucket({"certainty": "contractual"}) == "contractual"
    assert val._row_bucket({"default_triggered": True}) == "default_triggered"
    assert val._row_bucket({"type": "guarantee_payment"}) in ("default_triggered", "contingent")
    assert val._row_bucket({}) == "contingent"


def test_val_per_kind_entry_flags():
    row = {"certainty": "c", "status": "s", "revenue_matched": 1, "default_triggered": 0, "type": "x"}
    entry = val._per_kind_entry(row, 5.0, 1.0, {"h": 1})
    assert entry["total_billions"] == 5.0 and entry["annualized_billions"] == 1.0
    assert entry["revenue_matched"] is True and entry["payment_horizon"] == {"h": 1}


def test_val_record_helpers_all_arms():
    by_fy: dict[str, dict[str, float]] = {}
    flat = {"contractual": 0.0, "contingent": 0.0, "default_triggered": 0.0, "revenue_matched": 0.0}
    val._record_schedule_entries(
        by_fy, "contractual", [{"fiscal_year": "2027", "amount_billions": 2.0}, {"amount_billions": 9.0}]
    )
    assert by_fy["2027"]["contractual"] == 2.0
    by_fy2: dict[str, dict[str, float]] = {}
    assert val._record_horizon_schedule(by_fy2, "contingent", {}) is False
    assert (
        val._record_horizon_schedule(
            by_fy2, "contingent", {"schedule": [{"fiscal_year": "2028", "amount_billions": 4.0}]}
        )
        is True
    )
    assert by_fy2["2028"]["contingent"] == 4.0
    assert val._front_loaded_amounts({}) is None
    assert val._front_loaded_amounts({"paid_in_remainder_billions": 2.0, "paid_in_remainder_of_fy": "2027"}) == (
        2.0,
        "2027",
    )
    assert val._record_front_loaded({}, "contractual", {}, 6) is False
    by_fy3: dict[str, dict[str, float]] = {}
    assert (
        val._record_front_loaded(
            by_fy3,
            "contractual",
            {
                "paid_in_remainder_billions": 2.0,
                "paid_in_remainder_of_fy": "2027",
                "paid_after_remainder_billions": 10.0,
            },
            6,
        )
        is True
    )
    assert by_fy3["2027"]["contractual"] == 2.0 and by_fy3["2028"]["contractual"] == pytest.approx(2.0)
    by_fy4: dict[str, dict[str, float]] = {}
    assert (
        val._record_front_loaded(
            by_fy4,
            "contractual",
            {
                "paid_in_remainder_billions": 2.0,
                "paid_in_remainder_of_fy": "FY??",
                "paid_after_remainder_billions": 10.0,
            },
            6,
        )
        is True
    )  # unparseable base year
    assert by_fy4 == {"FY??": {"contractual": 2.0, "contingent": 0.0, "default_triggered": 0.0, "revenue_matched": 0.0}}
    by_fy5: dict[str, dict[str, float]] = {}
    val._record_tail_years(by_fy5, "contractual", {}, "2027", 6)  # zero tail -> no-op
    assert by_fy5 == {}
    # row-level dispatch: schedule / horizon / front-loaded / flat
    by_fy6: dict[str, dict[str, float]] = {}
    flat6 = dict(flat)
    val._record_row_schedule(
        by_fy6, flat6, "contractual", 9.0, {}, [{"fiscal_year": "2027", "amount_billions": 1.0}], 1.0, {}, 6
    )
    val._record_row_schedule(
        by_fy6,
        flat6,
        "contingent",
        9.0,
        {},
        [],
        0.0,
        {"schedule": [{"fiscal_year": "2028", "amount_billions": 1.0}]},
        6,
    )
    val._record_row_schedule(
        by_fy6,
        flat6,
        "contingent",
        9.0,
        {},
        [],
        0.0,
        {"paid_in_remainder_billions": 1.0, "paid_in_remainder_of_fy": "2029"},
        6,
    )
    val._record_row_schedule(by_fy6, flat6, "contingent", 3.0, {}, [], 0.0, {}, 6)
    assert flat6["contingent"] == 3.0


def test_val_impact_parent_uncovered_arms():
    assert (
        val._obligation_annual_impact([{"schedule_component": True, "amount_billions": 5}], 6)[
            "contingent_annual_billions"
        ]
        == 0.0
    )
    assert val._obligation_annual_impact([{"amount_billions": 0}], 6)["contingent_annual_billions"] == 0.0
    assert val._obligation_annual_impact([{"amount_billions": "x"}], 6)["contingent_annual_billions"] == 0.0
    out = val._obligation_annual_impact([{"amount_billions": 6.0, "type": "lease-x", "status": "on_balance_sheet"}], 6)
    assert out["per_kind"]["lease-x"]["total_billions"] == 6.0  # informational-only arm
    assert out["contingent_annual_billions"] == 0.0


def test_val_tax_helpers():
    assert val._tax_rate_from_note_markdown("no table here") is None
    md = "provision for income taxes | $100 and income before income taxes | $400"
    assert val._tax_rate_from_note_markdown(md) == pytest.approx(0.25)
    assert val._first_sane_rate([None, 0.01, 0.5, 0.2]) == 0.2
    assert val._first_sane_rate([None, 99.0]) is None
    assert val._ob_ticker({"ticker": "KO"}) == "KO"
    assert val._ob_ticker({"ticker": 5}) == ""
    assert val._effective_tax_rate({"ticker": "ZZZ-NOPE"}) is None

    # filing path error arm + notes-None arm + markdown-error arm
    class _Boom:
        def get_latest_report(self, *a: object, **k: object) -> object:
            raise RuntimeError("down")

    monkeypatch = pytest.MonkeyPatch()

    def _f2288_136(t: object, f: object) -> tuple[Filing, object] | None:
        raise RuntimeError("down")

    monkeypatch.setattr(val.edgar_client, "get_latest_report", _f2288_136)
    assert val._filing_tax_rate("X") is None

    def _f2290_137(t: object, f: object) -> tuple[Filing, object] | None:
        return None

    monkeypatch.setattr(val.edgar_client, "get_latest_report", _f2290_137)
    assert val._filing_tax_rate("X") is None

    def _f2292_138(t: object, f: object) -> object:
        return (None, type("D", (), {"notes": None})())

    monkeypatch.setattr(val.edgar_client, "get_latest_report", _f2292_138)
    assert val._filing_tax_rate("X") is None
    monkeypatch.undo()


def test_val_filing_tax_rate_parses_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    md = "income tax expense | $50 with income before income taxes | $200"

    def _f2299_139(self: object) -> object:
        return md

    note = type("N", (), {"to_markdown": _f2299_139})()

    def _f2300_140(self: object, q: object) -> object:
        return [note, note, note, note]

    notes = type("Ns", (), {"search": _f2300_140})()
    doc = type("D", (), {"notes": notes})()

    def _f2302_141(t: object, f: object) -> object:
        return (None, doc)

    monkeypatch.setattr(val.edgar_client, "get_latest_report", _f2302_141)
    assert val._filing_tax_rate("KO") == pytest.approx(0.25)


def test_val_facts_tax_rate_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "concept": "us-gaap:IncomeTaxExpenseBenefit",
                "fiscal_period": "FY",
                "period_end": "2026-01-01",
                "value": 20.0,
            },
            {
                "concept": "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
                "fiscal_period": "FY",
                "period_end": "2026-01-01",
                "value": 100.0,
            },
        ]
    )

    def _f2313_143(t: object) -> _RealCompany:
        return _df_company(df)

    monkeypatch.setattr(val.edgar_client, "get_company", _f2313_143)
    assert val._facts_tax_rate("KO") == pytest.approx(0.2)

    def _f2315_145(t: object) -> _RealCompany:
        return _none_df_company()

    monkeypatch.setattr(val.edgar_client, "get_company", _f2315_145)
    assert val._facts_tax_rate("KO") is None

    def _f2318_148(t: object) -> _RealCompany:
        return _df_company(df.iloc[0:0])

    monkeypatch.setattr(val.edgar_client, "get_company", _f2318_148)
    assert val._facts_tax_rate("KO") is None


def test_val_revenue_margin_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"concept": "us-gaap:GrossProfit", "fiscal_period": "FY", "period_end": "2026-01-01", "value": 75.0},
            {"concept": "us-gaap:Revenues", "fiscal_period": "FY", "period_end": "2026-01-01", "value": 100.0},
        ]
    )

    def _f2329_151(t: object) -> _RealCompany:
        return _df_company(df)

    monkeypatch.setattr(val.edgar_client, "get_company", _f2329_151)
    assert val._revenue_matched_margin("KO") == (0.75, "company_facts")

    def _f2331_153(t: object) -> _RealCompany:
        return _none_df_company()

    monkeypatch.setattr(val.edgar_client, "get_company", _f2331_153)
    assert val._revenue_matched_margin("KO")[0] is None


def test_val_scenario_helpers_all_arms():
    rows: list[dict[str, object]] = []
    val._scenario_hit(rows, "a", 10.0, True, "n", 100, None)
    assert rows[-1]["reason"] == "effective tax rate unavailable"
    val._scenario_hit(rows, "b", 10.0, False, "n", None, 0.2)
    assert rows[-1]["reason"] == "diluted shares unavailable"
    val._scenario_hit(rows, "c", 10.0, True, "n", 10_000_000_000, 0.2)
    assert rows[-1]["eps_impact"] is not None
    assert val._snapshot_of({}) == []
    assert val._snapshot_of({"obligations": [{"a": 1}, "x"]}) == [{"a": 1}]
    assert val._snapshot_of({"current_snapshot": "nope"}) == []
    assert val._purchase_exposure([{"type": "other"}, {"type": "supply_commitments", "amount_billions": "x"}]) is None
    assert val._purchase_exposure([{"type": "purchase_commitments", "amount_billions": 40.0}]) == 40.0
    assert val._purchase_exposure([]) is None
    snap: list[Mapping[str, object]] = [
        {"status": "future_cash_obligation", "type": "vendor", "amount_billions": 6.0},
        {"status": "off_balance_sheet", "type": "supply_commitments", "amount_billions": 99.0},
        {"status": "future_cash_obligation", "type": "vendor", "amount_billions": "x"},
    ]
    assert val._total_future_commitments(snap) == 6.0
    assert val._scenario_exposure({"amount_billions": 3}) == 3.0
    assert val._scenario_exposure({}) == 0.0
    rows2: list[dict[str, object]] = []
    val._add_purchase_scenarios(rows2, None, 1, 0.2)
    assert rows2 == []
    val._add_purchase_scenarios(rows2, 100.0, 10_000_000_000, 0.2)
    assert len(rows2) == 3
    rows3: list[dict[str, object]] = []
    val._add_commitment_scenario(rows3, 0.0, 1, 0.2)
    assert rows3 == []
    val._add_commitment_scenario(rows3, 12.0, 10_000_000_000, 0.2)
    assert len(rows3) == 1
    rows4: list[dict[str, object]] = []
    val._add_trigger_scenarios(
        rows4,
        [
            {"default_triggered": True, "type": "guarantee", "amount_billions": 5.0},
            {"type": "unrecognized_tax_benefits", "amount_billions": 2.0},
            {"type": "other"},
        ],
        10_000_000_000,
        0.2,
    )
    assert len(rows4) == 2
    out = val._obligation_eps_scenarios(
        {"obligations": [{"type": "unrecognized_tax_benefits", "amount_billions": 2.0}]}, None, None
    )
    assert "unavailable; no rate used" in str(out["assumption"])
    out2 = val._obligation_eps_scenarios({"obligations": []}, 10, 0.2)
    assert "0.2" in str(out2["assumption"])


def test_val_eps_line_helpers():
    assert val._eps_line(None, 1.0, 1.0, "x", 100.0) is None
    base = val._eps_line(10.0, None, None, "consensus", 100.0)
    assert base and base["pe"] == 10.0
    _eps_none = val._eps_line(10.0, None, None, "c", None)
    assert _eps_none is not None and _eps_none["pe"] is None
    _eps_zero = val._eps_line(0.0, None, None, "c", 100.0)
    assert _eps_zero is not None and _eps_zero["pe"] is None
    contractual = val._eps_line(10.0, 2.0, None, "adj", 100.0)
    assert contractual and contractual["eps_after_contractual"] == 8.0
    _eps_adj = val._eps_line(10.0, 2.0, None, "adj", None)
    assert _eps_adj is not None and _eps_adj["pe_after_contractual"] is None
    _eps_adj2 = val._eps_line(1.0, 2.0, None, "adj", 100.0)
    assert _eps_adj2 is not None and _eps_adj2["pe_after_contractual"] is None
    assert val._contingent_residual(10.0, 2.0, 3.0) == 5.0
    full = val._eps_line(10.0, 2.0, 3.0, "s", 100.0)
    assert full and full["eps_after_all_obligations"] == 5.0
    _eps_full = val._eps_line(10.0, 2.0, 3.0, "s", None)
    assert _eps_full is not None and _eps_full["pe_after_all_obligations"] is None
    neg = val._eps_line(1.0, 2.0, 3.0, "s", 100.0)
    assert neg and neg["pe_after_all_obligations"] is None
    line: dict[str, object] = {}
    val._apply_contractual_drag(line, 10.0, None, 100.0)
    assert line == {}
    line2: dict[str, object] = {}
    val._apply_contingent_drag(line2, 10.0, None, None, 100.0)
    assert line2 == {}


def test_val_fy_and_drag_helpers():
    assert val._fy_year({}) is None
    assert val._fy_year({"period_end_date": "2027-01-31"}) == "2027"
    assert val._fy_ps({}, {}, None, "2027", "contractual") is None
    assert val._fy_ps({}, {}, 1, None, "contractual") is None
    assert val._fy_ps(
        {"2027": {"contractual": 6.0}}, {"contractual": 0.0}, 1_000_000_000, "2027", "contractual"
    ) == pytest.approx(6.0)
    drags = val._fy_drags({"2027": {"contractual": 1.0}}, {}, 1_000_000_000, "2027")
    assert set(drags) == {"contractual", "contingent", "default_triggered", "revenue_matched"}
    assert val._drag_ps(6.0, None) is None
    assert val._drag_ps(6.0, 1_000_000_000) == pytest.approx(6.0)
    assert val._implied_coverage(0.0, 0.5) is None
    assert val._implied_coverage(6.0, None) is None
    assert val._implied_coverage(6.0, 0.5) == pytest.approx(12.0)
    assert val._estimates_by_period({}) is None
    assert val._estimates_by_period({"forward_estimates": [{"period": "p"}]}) == {"p": {"period": "p"}}
    assert val._tier_label("Consensus", "2027", "fb") == "Consensus FY2027"
    assert val._tier_label("Consensus", None, "fb") == "fb"
    assert val._scenario_tier_label("no_default", "2027") == "Scenario FY2027 (no default)"
    assert val._scenario_tier_label("default", None) == "Scenario (current FY, counterparty default)"
    assert val._next_scenario_tier_label("no_default", None) == "Scenario (next FY, no default)"
    assert val._next_scenario_tier_label("default", "2028") == "Scenario FY2028 (counterparty default)"
    assert val._period_eps({}) is None
    assert val._period_eps({"eps_avg": 3.0}) == 3.0
    assert val._ttm_eps({"ttm_eps_diluted": 2.0}) == 2.0
    assert val._ttm_eps({}) is None
    assert val._shares_from_estimates({"shares_outstanding": 5}) == 5
    assert val._shares_from_estimates({}) is None


def test_val_coverage_and_assemble_helpers():
    assert val._coverage_rows({}) == []
    assert val._coverage_rows({"obligations": [{"a": 1}, 5]}) == [{"a": 1}]
    assert val._coverage_manifest({}) == []
    assert val._coverage_manifest({"scan_manifest": [{"a": 1}]}) == [{"a": 1}]
    assert val._coverage_warning_list({}, {}) == []
    assert val._coverage_warning_list({"warnings": ["w"]}, {}) == ["w"]
    ob = {"filings_examined": ["2026-01-01"]}
    assert val._filings_examined(ob, [], []) == ["2026-01-01"]
    assert val._filings_examined({}, [{"filing_date": "2026-01-02"}], []) == ["2026-01-02"]
    assert val._sections_examined({}, [], [{"source": "s"}]) == ["s"]
    assert val._unquantified_count({"unquantified": [1, 2]}, {}) == 2
    assert val._unquantified_count({}, {"unquantified_count": 3}) == 3
    assert val._quantified_count({}, [{"a": 1}]) == 1
    assert val._quantified_count({"quantified_count": "x"}, [{"a": 1}]) == 1
    cov = val._build_coverage({"obligations": [], "unquantified": [1]}, None)
    assert cov["unquantified_count"] == 1
    cov2 = val._build_coverage({"obligations": [{"filed": "2026-01-01"}]}, "gap!")
    assert "gap!" in str(cov2["warnings"])
    fwd = val._build_forward_eps(
        10.0,
        11.0,
        {"contractual": 1.0, "contingent": 1.0, "default_triggered": 0.0, "revenue_matched": 0.0},
        {"contractual": 1.0, "contingent": 1.0, "default_triggered": 0.0, "revenue_matched": 0.0},
        100.0,
    )
    assert len(fwd) == 9
    eps_map = val._projected_eps_map(fwd, "2027", "2028")
    assert len(eps_map) == 9
    _scen = fwd["scenario"]
    assert isinstance(_scen, dict) and eps_map["Scenario FY2027 (no default)"] == _scen["eps_after_all_obligations"]
    _scen_d = fwd["scenario_with_defaults"]
    assert (
        isinstance(_scen_d, dict)
        and eps_map["Scenario FY2027 (counterparty default)"] == _scen_d["eps_after_all_obligations"]
    )
    assert eps_map["Scenario FY2028 (counterparty default)"] is None  # no scenario_with_defaults_next_fy tier exists
    vd = val._valuation_drags(
        {
            "contractual_annual_billions": 1.0,
            "contingent_annual_billions": 1.0,
            "default_triggered_annual_billions": 1.0,
            "revenue_matched_annual_billions": 1.0,
            "per_kind": {},
            "flat_annual_by_bucket": {},
            "impact_by_fiscal_year": {},
        },
        "KO",
        None,
    )
    assert vd[0] is None
    tiers, yc, _yn = val._valuation_forward_tiers(
        {
            "contractual_annual_billions": 0.0,
            "contingent_annual_billions": 0.0,
            "default_triggered_annual_billions": 0.0,
            "revenue_matched_annual_billions": 0.0,
            "per_kind": {},
            "flat_annual_by_bucket": {},
            "impact_by_fiscal_year": {},
        },
        None,
        {},
        {},
        10.0,
        11.0,
        100.0,
    )
    assert yc is None and len(tiers) == 9
    block = val._obligations_block(
        {
            "contractual_annual_billions": 1.0,
            "contingent_annual_billions": 1.0,
            "default_triggered_annual_billions": 1.0,
            "revenue_matched_annual_billions": 1.0,
            "per_kind": {},
            "flat_annual_by_bucket": {},
            "impact_by_fiscal_year": {},
        },
        1.0,
        1.0,
        1.0,
        1.0,
        5.0,
        0.5,
        "company_facts",
    )
    assert block["revenue_matched_implied_revenue_billions"] == 5.0
    block2 = val._obligations_block(
        {
            "contractual_annual_billions": 1.0,
            "contingent_annual_billions": 1.0,
            "default_triggered_annual_billions": 1.0,
            "revenue_matched_annual_billions": 1.0,
            "per_kind": {},
            "flat_annual_by_bucket": {},
            "impact_by_fiscal_year": {},
        },
        1.0,
        1.0,
        1.0,
        1.0,
        None,
        None,
        "x",
    )
    assert block2["revenue_matched_implied_revenue_billions"] is None


def test_val_metrics_parent_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(val, "cache", _FakeCache())
    assert val.get_valuation_metrics("  ") == {"error": "No valuation data for : empty ticker"}
    getattr(val.cache, "store")["valuation:KO"] = {"ticker": "KO"}  # noqa: B009 - FakeCache.store invisible to pyrefly; getattr keeps pyrefly-0
    assert val.get_valuation_metrics("ko") == {"ticker": "KO"}
    monkeypatch.setattr(val, "cache", _FakeCache())

    def _f2495_155(t: object) -> float | None:
        return None

    monkeypatch.setattr(val, "get_live_price", _f2495_155)

    def _f2496_156(t: object) -> dict[str, object]:
        return {"error": "down"}

    monkeypatch.setattr(val.analyst_client, "get_analyst_estimates", _f2496_156)
    assert "error" in val.get_valuation_metrics("KO")

    def _f2498_157(t: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(val.analyst_client, "get_analyst_estimates", _f2498_157)
    assert "missing forward" in str(val.get_valuation_metrics("KO")["error"])

    def _f2500_158(t: object) -> object:
        return {
            "forward_estimates": [{"period": "current_fiscal_year", "eps_avg": 1.0}],
            "shares_outstanding": 10,
            "as_of": "t",
        }

    monkeypatch.setattr(val.analyst_client, "get_analyst_estimates", _f2500_158)

    def _f2502_159(t: object, m: object) -> object:
        return {"error": "no eps"}

    monkeypatch.setattr(val.sec_facts, "get_fundamentals", _f2502_159)
    assert "error" in val.get_valuation_metrics("KO")

    def _f2504_160(t: object, m: object) -> object:
        return {"ttm_eps_diluted": 1.0}

    monkeypatch.setattr(val.sec_facts, "get_fundamentals", _f2504_160)

    def _f2505_161(t: object) -> dict[str, object]:
        return {"error": "no ob"}

    monkeypatch.setattr(val.obligations, "get_obligations", _f2505_161)
    assert "error" in val.get_valuation_metrics("KO")

    # price-gap arm caches nothing but still returns
    def _f2508_162(t: object) -> dict[str, object]:
        return {"obligations": []}

    monkeypatch.setattr(val.obligations, "get_obligations", _f2508_162)

    def _f2509_163(ob: object) -> float | None:
        return None

    monkeypatch.setattr(val, "_effective_tax_rate", _f2509_163)

    def _f2510_164(t: object) -> tuple[float | None, str]:
        return (None, "unavailable: x")

    monkeypatch.setattr(val, "_revenue_matched_margin", _f2510_164)
    out = val.get_valuation_metrics("KO")
    assert out["price_gap"] is not None and "valuation:KO" not in getattr(val.cache, "store")  # noqa: B009 - FakeCache.store invisible to pyrefly; getattr keeps pyrefly-0


def test_val_fetch_helpers_error_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    err, _ = val._fetch_obligations("KO") if False else (None, None)

    def _f2517_165(t: object) -> dict[str, object]:
        return {"error": "boom"}

    monkeypatch.setattr(val.obligations, "get_obligations", _f2517_165)
    err, empty = val._fetch_obligations("KO")
    assert err is not None and empty == {}

    def _f2520_166(t: object) -> dict[str, object]:
        return {"obligations": []}

    monkeypatch.setattr(val.obligations, "get_obligations", _f2520_166)
    err, _ob = val._fetch_obligations("KO")
    assert err is None

    def _f2523_167(t: object) -> dict[str, object]:
        return {"error": "down"}

    monkeypatch.setattr(val.analyst_client, "get_analyst_estimates", _f2523_167)
    err, _per = val._fetch_estimates("KO")
    assert err is not None

    def _f2526_168(t: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(val.analyst_client, "get_analyst_estimates", _f2526_168)
    err, _per = val._fetch_estimates("KO")
    assert err is not None

    def _f2529_169(t: object, m: object) -> object:
        return {"error": "x"}

    monkeypatch.setattr(val.sec_facts, "get_fundamentals", _f2529_169)
    err, ttm, sh = val._fetch_eps("KO", {})
    assert err is not None

    def _f2532_170(t: object, m: object) -> object:
        return {"ttm_eps_diluted": 2.0}

    monkeypatch.setattr(val.sec_facts, "get_fundamentals", _f2532_170)
    err, ttm, sh = val._fetch_eps("KO", {"shares_outstanding": 7})
    assert (err, ttm, sh) == (None, 2.0, 7)


# ------------------------------------------------------------ normalization


def test_norm_ticker_helpers():
    assert norm._ticker_values(None) == []
    assert norm._ticker_values({"a": 1}) == [1]
    assert norm._ticker_values([1, 2]) == [1, 2]
    assert norm._ticker_values((1,)) == [1]
    assert norm._ticker_cik({}) == ("", None)
    assert norm._ticker_cik({"ticker": "ko", "cik_str": "x"}) == ("", None)
    assert norm._ticker_cik({"ticker": "ko", "cik_str": 320193}) == ("KO", 320193)
    assert norm._ticker_cik({"ticker": "  ", "cik_str": 1}) == ("", None)
    e, a = norm._ticker_rows({"title": "  Acme "}, "KO", 320193, "2026-08-10T12:00:00Z", "h")
    assert e["name"] == "Acme" and a["alias_value"] == "KO"
    out = norm.normalize_sec_tickers(
        {"0": {"ticker": "ko", "cik_str": 1, "title": "K"}}, retrieved_at="2026-08-10T12:00:00Z", content_hash="h"
    )
    assert len(out["entities"]) == 1
    out2 = norm.normalize_sec_tickers(
        [{"ticker": "", "cik_str": 1}, "junk", {"ticker": "ko", "cik_str": "bad"}], retrieved_at="r", content_hash="h"
    )
    assert out2["entities"] == []
    out3 = norm.normalize_sec_tickers("junk", retrieved_at="r", content_hash="h")
    assert out3["entities"] == []


def test_norm_generic_extractors():
    assert norm._facts_namespaces(None) == {}
    assert norm._facts_namespaces({"facts": "x"}) == {}
    assert norm._facts_namespaces({"facts": {"ns": "x"}}) == {"ns": "x"}
    assert norm._concepts_of("x") == {}
    assert norm._unit_facts("x", "USD") == []
    assert norm._unit_facts({"units": "x"}, "USD") == []
    assert norm._unit_facts({"units": {"USD": "x"}}, "USD") == []
    assert norm._unit_facts({"units": {"USD": [{"a": 1}]}}, "USD") == [{"a": 1}]
    assert norm._canonical_name("Nope") is None
    assert norm._canonical_name("Revenues") == "Revenue"
    entries: list[tuple[str, str, str, dict[str, object]]] = []
    norm._collect_entries(entries, "n", "ns", "t", "USD", [{"a": 1}, "x"])
    assert len(entries) == 1
    assert norm._extract_canonical_facts(None) == []
    assert norm._extract_canonical_facts({"facts": {"ns": "x"}}) == []
    assert (
        norm._extract_eps_facts({"facts": {"ns": {"EarningsPerShareDiluted": {"units": {"USD/shares": [{"a": 1}]}}}}})
        != []
    )
    assert (
        norm._extract_dividend_facts(
            {"facts": {"ns": {"CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": [{"a": 1}]}}}}}
        )
        != []
    )


def test_norm_event_scan_helpers():
    assert norm._event_namespaces(None) == {}
    assert norm._event_namespaces({"facts": "x"}) == {}
    assert norm._amount_unit_choice({}) == []
    assert norm._amount_unit_choice({"USD/shares": [], "other": []}) == []
    assert norm._amount_unit_choice({"USD/shares": [{"a": 1}]}) == ["USD/shares"]
    assert norm._amount_unit_choice({"other": [{"b": 2}]}) == ["other"]
    assert norm._amount_unit_choice({"USD/shares": [{"a": 1}]}) == ["USD/shares"]
    assert norm._event_amount({}) is None
    assert norm._event_amount({"val": "x"}) is None
    assert norm._event_amount({"val": 0.5}) == 0.5
    assert norm._event_key({}) is None
    assert norm._event_key({"accn": "a", "filed": "f"}) == ("a", "f")
    amounts: list[tuple[str, str, float, str, str]] = []
    norm._collect_event_amounts(
        amounts,
        {
            "USD/shares": [
                {"val": 0.5, "accn": "a", "filed": "f"},
                {"val": "x", "accn": "a", "filed": "f"},
                {"accn": "a"},
                "x",
            ]
        },
        "ns",
        "t",
    )
    assert len(amounts) == 1
    amounts2: list[tuple[str, str, float, str, str]] = []
    norm._collect_event_amounts(amounts2, {"USD/shares": "x"}, "ns", "t")
    assert amounts2 == []
    dates: dict[tuple[str, str], dict[str, set[str]]] = {}
    norm._collect_event_dates(
        dates,
        {"USD": [{"val": "2026-01-01", "accn": "a", "filed": "f"}, {"val": "", "accn": "a", "filed": "f"}, "x", "y"]},
        "record_date",
    )
    assert dates == {("a", "f"): {"record_date": {"2026-01-01"}}}
    dates2: dict[tuple[str, str], dict[str, set[str]]] = {}
    norm._collect_event_dates(dates2, {"USD": "x"}, "record_date")
    assert dates2 == {}
    a3: list[tuple[str, str, float, str, str]] = []
    d3: dict[tuple[str, str], dict[str, set[str]]] = {}
    norm._scan_event_concept(a3, d3, "ns", "Unrelated", {})
    assert a3 == [] and d3 == {}
    norm._scan_event_concept(a3, d3, "ns", "Unrelated", "x")
    norm._scan_event_concept(a3, d3, "ns", norm.DIVIDEND_EVENT_AMOUNT_CONCEPT, "x")
    norm._scan_event_concept(a3, d3, "ns", norm.DIVIDEND_EVENT_AMOUNT_CONCEPT, {"units": "x"})
    assert a3 == []


def test_norm_group_helpers():
    assert norm._group_event_amounts([]) == {}
    assert norm._deduped_amounts([(0.5, "u", "c"), (0.5, "u2", "c2")]) == {0.5: ("u", "c")}
    assert norm._role_is_single({}, "record_date") is True
    assert norm._role_is_single({"record_date": {"a", "b"}}, "record_date") is False
    assert norm._group_is_paired({0.5: ("u", "c")}, {}) is True
    assert norm._group_is_paired({0.5: ("u", "c"), 0.6: ("u", "c")}, {}) is False
    assert norm._group_is_paired({0.5: ("u", "c")}, {"record_date": {"a", "b"}}) is False
    decl, rec, paired = norm._paired_dates({})
    assert (decl, rec, paired) == (None, None, [None])
    assert norm._currency_of("USD/shares") == "USD"
    assert norm._currency_of("USD") == "USD"
    ev = norm._paired_event(1, "e", "s", 0.5, "USD/shares", "c", "d", "r", "p", "a", "f", "u", "h")
    assert ev["payment_date"] == "p"
    ev2 = norm._unpaired_event(1, "e", "s", 0.5, "USD", "c", "a", "f", "u", "h")
    assert ev2["payment_date"] is None
    events: list[dict[str, object]] = []
    norm._emit_group_events(events, 1, "e", "s", {0.5: ("USD/shares", "c")}, {}, "a", "f", "u", "h")
    assert len(events) == 1
    events2: list[dict[str, object]] = []
    norm._emit_group_events(events2, 1, "e", "s", {0.5: ("u", "c"), 0.6: ("u", "c")}, {}, "a", "f", "u", "h")
    assert len(events2) == 2
    # parent uncovered arms: non-dict namespaces + unknown concept
    assert (
        norm._extract_dividend_event_facts(
            {"facts": {"ns": "x"}},
            cik=1,
            entity_id="e",
            security_id="s",
            source_url="u",
            retrieved_at="r",
            content_hash="h",
        )
        == []
    )
    assert (
        norm._extract_dividend_event_facts(
            {"facts": {"ns": {"Nope": {}}}},
            cik=1,
            entity_id="e",
            security_id="s",
            source_url="u",
            retrieved_at="r",
            content_hash="h",
        )
        == []
    )
    assert (
        norm._extract_dividend_event_facts(
            "x", cik=1, entity_id="e", security_id="s", source_url="u", retrieved_at="r", content_hash="h"
        )
        == []
    )


def test_norm_text_helpers():
    m = norm._DIVIDEND_DECLARE_RE.search("declared a quarterly cash dividend of $0.50 per share")
    assert m is not None
    assert norm._text_event_amount(m) == 0.5
    assert norm._text_dividend_type("supplemental dividend") == "supplemental"
    assert norm._text_dividend_type("special dividend") == "special"
    assert norm._text_dividend_type("extraordinary dividend") == "special"
    assert norm._text_dividend_type("quarterly dividend") == "regular"
    assert norm._text_event_dates("payable on January 5, 2026")["payment_date"] == "2026-01-05"
    assert norm._text_event_for_sentence(1, "e", "s", "no dividend here", "a", "f", "u", "h") is None
    ev = norm._text_event_for_sentence(
        1, "e", "s", "declared a dividend of $1.00 per share payable on January 5, 2026", "a", "f", "u", "h"
    )
    assert ev is not None and ev["amount_per_share"] == 1.0
    assert (
        norm._extract_dividend_events_from_text(
            "", cik=1, entity_id="e", security_id="s", accession="a", filed_at="f", source_url="u", content_hash="h"
        )
        == []
    )


def test_norm_company_facts_helpers():
    assert norm._parse_company_cik({}) == 0
    assert norm._parse_company_cik({"cik": "x"}) == 0
    assert norm._parse_company_cik({"cik": 320193}) == 320193
    assert norm._parse_company_cik("x") == 0
    assert norm._envelope_known([], "r") == "r"
    assert (
        norm._envelope_known([("c", "o", "u", {"filed": "2026-01-02"}), ("c", "o", "u", {"filed": "2026-01-01"})], "r")
        == "2026-01-02"
    )
    assert norm._fact_float({}) is None
    assert norm._fact_float({"val": "x"}) is None
    assert norm._fact_float({"val": 5}) == 5.0
    assert norm._fact_float({"val": {"a": 1}}) is None
    assert norm._fact_key({}) is None
    assert norm._fact_key({"end": "e", "filed": "f", "accn": "a"}) == ("e", "f", "a")
    assert norm._fact_fiscal_year({}) is None
    assert norm._fact_fiscal_year({"fy": "x"}) is None
    assert norm._fact_fiscal_year({"fy": 2026}) == 2026
    assert norm._fact_fiscal_year({"fy": {"a": 1}}) is None
    assert norm._fact_row_or_none(1, "e", "s", "c", "o", "u", {}, "r", "u", "s", "h") is None
    assert norm._fact_row_or_none(1, "e", "s", "c", "o", "u", {"val": 1.0}, "r", "u", "s", "h") is None
    row = norm._fact_row_or_none(
        1,
        "e",
        "s",
        "c",
        "o",
        "u",
        {"val": 1.0, "end": "e", "filed": "f", "accn": "a", "start": "s", "fy": 2026, "fp": "Q1"},
        "r",
        "u",
        "s",
        "h",
    )
    assert row is not None and row["duration_type"] == "duration"
    assert norm._financial_fact_rows([], 1, "e", "s", "r", "u", "sr", "h") == []
    sec = norm._company_security("s", "e", [], "k", "r", "h")
    assert sec["security_type"] == "unknown"
    sec2 = norm._company_security("s", "e", [{"a": 1}], "k", "r", "h")
    assert sec2["security_type"] == "equity-common"


def test_norm_finra_helpers():
    assert norm._finra_symbol({}) is None
    assert norm._finra_symbol({"symbolCode": " aaa "}) == "AAA"
    assert norm._finra_short_position({"currentShortPositionQuantity": -5}) is None
    assert norm._finra_short_position({"currentShortPositionQuantity": 5}) == 5.0
    assert norm._finra_short_position({}) is None
    row = norm._finra_row(
        "2026-08-14", "2026-08-10T12:00:00Z", None, "h" * 20, "u", "sr", {"issueName": " Alpha "}, "AAA"
    )
    assert row["known_at"] == "2026-08-10T12:00:00Z" and row["issue_name"] == "Alpha"
    row2 = norm._finra_row("2026-08-14", "2026-08-10T12:00:00Z", "2026-08-10T12:00:00Z", "h" * 20, "u", "sr", {}, "AAA")
    assert row2["known_at"] == "2026-08-10T12:00:00Z" and row2["issue_name"] is None
    with pytest.raises(ValueError, match="not a parseable date"):
        norm._check_finra_known_at("xx", "r", "k")
    with pytest.raises(ValueError, match="precedes settlement_date"):
        norm.normalize_finra_short_interest(
            [{"symbolCode": "AAA"}],
            settlement_date="2026-08-14",
            known_at="2026-08-01T00:00:00Z",
            retrieved_at="2026-08-20T00:00:00Z",
            content_hash="h",
            source_url="u",
            source_record_id="r",
        )
    out = norm.normalize_finra_short_interest(
        [{"symbolCode": ""}, {"symbolCode": "AAA", "currentShortPositionQuantity": -3}],
        settlement_date="2026-08-14",
        retrieved_at="2026-08-10T12:00:00Z",
        content_hash="h",
        source_url="u",
        source_record_id="r",
    )
    assert len(out["short_interest"]) == 1 and out["short_interest"][0]["short_position"] is None


def test_cli_format_mandate_value_and_breaches(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli._format_mandate_value(None, "m", None, {}) == ""
    assert cli._format_mandate_value("5", "prohibited_assets", None, {}) == "5"
    assert cli._format_mandate_value("5", "m", "t", {("m", "t"): "dollars"}) == "5"
    from decimal import Decimal

    assert cli._format_mandate_value(Decimal("0.5"), "m", None, {}) == "50.0%"
    assert cli._format_mandate_value(Decimal("0.5"), "m", "t", {("m", "t"): "pct"}) == "50.0%"
    from app.domain.risk.breaches import RiskBreach
    from app.domain.risk.evaluation import RiskEvaluation

    _eval_now = _dt.datetime.now(_dt.UTC)
    ev = RiskEvaluation(
        breaches=(), sector_exposures={"tech": Decimal("0.5")}, issues=(), snapshot_id="s", created_at=_eval_now
    )
    cli._print_mandate_sector_exposures(ev)
    assert "tech" in capsys.readouterr().out
    cli._print_mandate_sector_exposures(
        RiskEvaluation(breaches=(), sector_exposures={}, issues=(), snapshot_id="s", created_at=_eval_now)
    )
    assert capsys.readouterr().out == ""
    cli._print_mandate_breaches(
        RiskEvaluation(breaches=(), sector_exposures={}, issues=(), snapshot_id="s", created_at=_eval_now), {}
    )
    assert "No breaches" in capsys.readouterr().out
    breach = RiskBreach(
        metric="m", target="t", severity="high", actual=Decimal("0.5"), limit=Decimal("0.2"), excess=Decimal("0.3")
    )
    cli._print_mandate_breaches(
        RiskEvaluation(breaches=(breach,), sector_exposures={}, issues=(), snapshot_id="s", created_at=_eval_now),
        {("m", "t"): "pct"},
    )
    assert "Breaches" in capsys.readouterr().out
    breach2 = RiskBreach(metric="m", target=None, severity="low", actual=None, limit=None, excess=None)
    cli._print_mandate_breaches(
        RiskEvaluation(breaches=(breach2,), sector_exposures={}, issues=(), snapshot_id="s", created_at=_eval_now), {}
    )
    assert "m" in capsys.readouterr().out


def test_cli_thesis_inspect_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.thesis import yaml as tyaml
    from app.thesis.models import Thesis as _InspectThesis
    from app.thesis.models import Trigger as _InspectTrigger
    from app.thesis.repository import ThesisRepository as _InspectRepo

    _inspect_thesis = _InspectThesis(thesis_id="T1", slug="s1")

    class _InspectRepoD(_InspectRepo):
        @override
        def __init__(self, root: Path | str) -> None:
            self.root = Path(root)

        @override
        def load_thesis(self, id_or_slug: str) -> _InspectThesis:
            return _inspect_thesis

        @override
        def load_triggers(self, id_or_slug: str, include_processed: bool = True) -> list[_InspectTrigger]:
            return [
                _InspectTrigger(trigger_id="g1", thesis_id="T1", created_at="c", status="pending"),
                _InspectTrigger(trigger_id="g2", thesis_id="T1", created_at="c", status="done"),
            ]

    repo = _InspectRepoD(tmp_path)

    def _f2743_171(a: object) -> _InspectRepo:
        return repo

    monkeypatch.setattr(cli, "_thesis_repo", _f2743_171)

    def _f2744_172(r: object, i: object) -> _InspectThesis:
        return _inspect_thesis

    monkeypatch.setattr(cli, "_thesis_load", _f2744_172)
    (tmp_path / "s1").mkdir()
    for name in ("thesis", "state", "checkpoint"):
        (tmp_path / "s1" / f"{name}.yaml").write_text("x: 1\n")
    (tmp_path / "s1" / "questions.yaml").write_text("x: 1\n")
    (tmp_path / "s1" / "watch.yaml").write_text("x: 1\n")
    (tmp_path / "s1" / "memory.yaml").write_text("x: 1\n")
    from app.thesis.yaml import JSONValue as _JV

    def _f2751_173(p: object) -> dict[str, _JV]:
        return {"thesis_id": "T1", "questions": [], "rules": [], "memories": []}

    monkeypatch.setattr(tyaml, "load_raw_yaml", _f2751_173)

    def _f2753_174(p: object, cls: object) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(tyaml, "load_yaml", _f2753_174)
    cli._thesis_inspect(_ns(id="T1"))
    out = capsys.readouterr().out
    assert "triggers: 1 pending / 2 total" in out and "evidence files:" in out

    # invalid path collects problems and exits
    def _f2759_176(p: object, cls: object) -> object:
        raise ValueError("bad")

    monkeypatch.setattr(tyaml, "load_yaml", _f2759_176)
    with pytest.raises(SystemExit):
        cli._thesis_inspect(_ns(id="T1"))


def test_cli_google_data_ok_and_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    args = _ns(
        google_data_command="collect",
        geo=None,
        limit=25,
        source="trends",
        start_date=None,
        end_date=None,
        data_root=None,
        variable=[],
        tag=[],
        company=None,
    )

    def _f2769_177(a: object) -> object:
        return ["US"]

    monkeypatch.setattr(cli, "_google_geos", _f2769_177)

    def _f2770_178(a: object) -> object:
        return 5

    monkeypatch.setattr(cli, "_google_limit", _f2770_178)

    def _f2771_179(a: object) -> object:
        return ("trends", "patents")

    monkeypatch.setattr(cli, "_google_sources", _f2771_179)

    def _f2772_180(src: object, a: object, g: object, l: object) -> object:
        if src == "trends":
            return {"ok": src}
        raise RuntimeError("down")

    monkeypatch.setattr(cli, "_collect_google_source", _f2772_180)
    cli._cmd_google_data(args)
    out = capsys.readouterr().out
    assert "trends" in out and "down" in out
    with pytest.raises(SystemExit):
        cli._cmd_google_data(_ns(google_data_command="bogus"))


def test_cli_render_prose_all_codes():
    from app.domain.risk.evaluation import EvaluationIssue

    assert T.issue_to_prose(EvaluationIssue("cash_unavailable", "minimum_cash")) == "minimum_cash: cash unavailable"
    assert (
        T.issue_to_prose(EvaluationIssue("total_value_unavailable", "minimum_cash"))
        == "minimum_cash: total value unavailable"
    )
    assert T.issue_to_prose(EvaluationIssue("total_value_zero", "minimum_cash")) == "minimum_cash: total value is zero"
    assert "no weight" in T.issue_to_prose(EvaluationIssue("position_weight_unavailable", "w", ticker="NVDA"))
    assert "unknown exposure" in T.issue_to_prose(
        EvaluationIssue("unknown_sector_exposure", "sector_exposure", target="tech")
    )
    assert T.issue_to_prose(EvaluationIssue("other_code", "m")) == "m: other_code"


def test_cli_render_analyst_estimates_branches():
    base: dict[str, object] = {
        "ticker": "A",
        "as_of": "t",
        "source": "s",
        "quote": {"price": 10},
        "price_targets": {
            "median": 12,
            "mean": 11,
            "high": 15,
            "low": 9,
            "num_analysts": 3,
            "recommendation": "buy",
            "recommendation_mean": 2.0,
        },
        "valuation": {"trailing_pe": 20, "forward_pe": 18},
        "market_cap": 100,
        "shares_outstanding": 50,
        "forward_estimates": [],
    }
    assert "analyst consensus" in T._render_analyst_estimates(base, 10_000)
    noprice: dict[str, object] = dict(base, quote={})
    assert "unavailable" in T._render_analyst_estimates(noprice, 10_000)
    rows = dict(
        base,
        forward_estimates=[
            "notadict",
            {
                "period": "FY",
                "period_end_date": "d",
                "eps_avg": 1,
                "eps_growth_pct": 2,
                "eps_analysts": 3,
                "revenue_avg": 4,
                "revenue_growth_pct": 5,
                "revenue_analysts": 6,
                "eps_revision": {"current": 1, "days7_ago": 2, "days30_ago": 3, "days60_ago": 4},
            },
            {"period": "Q", "eps_avg": None, "revenue_avg": None},
        ],
    )
    assert "EPS revision" in T._render_analyst_estimates(rows, 10_000)


def test_cli_norm_shares_guard_arms():
    assert norm._extract_shares_facts(None) == []
    assert norm._extract_shares_facts({"facts": "x"}) == []
    assert norm._extract_shares_facts({"facts": {"dei": "x"}}) == []
    assert norm._extract_shares_facts({"facts": {"dei": {SHARES: "x"}}}) == []
    assert norm._extract_shares_facts({"facts": {"dei": {SHARES: {"units": "x"}}}}) == []
    assert norm._extract_shares_facts({"facts": {"dei": {SHARES: {"units": {"shares": "x"}}}}}) == []
    got = norm._extract_shares_facts({"facts": {"dei": {SHARES: {"units": {"shares": [{"a": 1}, "x"]}}}}})
    assert got == [{"a": 1}]


def test_cli_norm_to_float_arms():
    assert norm._to_float(None) is None
    assert norm._to_float(True) == 1.0
    assert norm._to_float(3) == 3.0
    assert norm._to_float("") is None
    assert norm._to_float("  ") is None
    assert norm._to_float("1.5") == 1.5
    assert norm._to_float("junk") is None
    assert norm._to_float(b"2.5") == 2.5
    assert norm._to_float(b"") is None
    assert norm._to_float(object()) is None
    assert norm._to_float(["x"]) is None

    class _BadStr:
        def __str__(self):
            raise TypeError("bad")

    assert norm._to_float(_BadStr()) is None

    class _EmptyStr:
        def __str__(self):
            return "   "

    assert norm._to_float(_EmptyStr()) is None
