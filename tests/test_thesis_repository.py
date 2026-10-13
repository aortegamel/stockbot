"""Filesystem/model round trip for the thesis repository (no network)."""

import threading
from pathlib import Path

import pytest
import yaml

from app.thesis.models import Thesis
from app.thesis.repository import ThesisRepository
from app.thesis.yaml import atomic_write_yaml, load_raw_yaml


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _repo(tmp_path: Path) -> ThesisRepository:
    return ThesisRepository(tmp_path / "theses")


def test_round_trip_create_load_list_update(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA demand stays strong", scope="NVDA", claims=["demand holds"])
    assert r.load_thesis(t.thesis_id).slug == t.slug
    assert r.load_thesis(t.slug).thesis_id == t.thesis_id
    assert [x.thesis_id for x in r.list_theses()] == [t.thesis_id]
    updated = r.update_thesis(t.thesis_id, scope="NVDA datacenter")
    assert updated.scope == "NVDA datacenter" and updated.thesis_id == t.thesis_id


def test_relative_repository_root_writes_to_expected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repo = ThesisRepository("data/thesis")
    thesis = repo.create_thesis("NVDA thesis", scope="NVDA", claims=["demand remains strong"])
    assert repo.root == (tmp_path / "data/thesis").resolve()
    assert (tmp_path / "data/thesis" / thesis.slug / "thesis.yaml").is_file()
    assert not (tmp_path / "data/thesis/data/thesis").exists()


def test_zero_and_multiple_expressions(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t0 = r.create_thesis("undecided thesis", scope="NVDA", claims=["c"])
    assert r.load_thesis(t0.thesis_id).expressions == ()
    t1 = r.create_thesis(
        "two ways to play NVDA",
        scope="NVDA",
        claims=["c"],
        expressions=[
            {"intent": "bearish", "instrument": "option", "direction": "long", "structure": "long puts"},
            {"intent": "bullish", "instrument": "equity", "direction": "long", "structure": "equity"},
        ],
    )
    got = r.load_thesis(t1.thesis_id).expressions
    assert len(got) == 2 and {e.instrument for e in got} == {"option", "equity"}


def test_unknown_open_vocab_structure_retained_verbatim(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    weird = "diagonalized quantum butterfly 7:11!!"
    t = r.create_thesis("weird structure", scope="NVDA", claims=["c"], expressions=[{"structure": weird}])
    assert r.load_thesis(t.thesis_id).expressions[0].structure == weird


def test_unknown_literals_retained(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("unknown thesis", claims=["c"])
    loaded = r.load_thesis(t.thesis_id)
    assert loaded.scope == "unknown"
    assert loaded.expressions == () and loaded.unknowns == ()


def test_thesis_vs_expression_assessments_stored_separately(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    r.apply_research_result(
        t.thesis_id,
        {
            "state": {
                "thesis_id": t.thesis_id,
                "assessment": "supported",
                "claim_assessments": {"c1": "supported"},
                "expression_assessments": {"e1": "flagged"},
            }
        },
        "run:x",
    )
    state = r.load_state(t.thesis_id)
    assert state.assessment == "supported"
    assert state.claim_assessments == {"c1": "supported"}
    assert state.expression_assessments == {"e1": "flagged"}
    assert r.load_thesis(t.thesis_id).claims[0].status == "unvalidated"


def test_malformed_and_schema_mismatch_yaml_rejected(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    thesis_file = tmp_path / "theses" / t.slug / "thesis.yaml"
    thesis_file.write_text("{ unclosed: [,,,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        r.load_thesis(t.thesis_id)
    thesis_file.write_text(yaml.safe_dump({"schema_version": 99, "thesis_id": t.thesis_id}), encoding="utf-8")
    with pytest.raises(ValueError):
        r.load_thesis(t.thesis_id)


def test_atomic_write_failure_leaves_old_valid_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    thesis_file = tmp_path / "theses" / t.slug / "thesis.yaml"
    before = thesis_file.read_text(encoding="utf-8")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(yaml, "safe_dump", _boom)
    with pytest.raises(RuntimeError):
        r.update_thesis(t.thesis_id, scope="CHANGED")
    assert thesis_file.read_text(encoding="utf-8") == before
    assert r.load_thesis(t.thesis_id).scope == "NVDA"


def test_traversal_and_symlink_escape_rejected(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    with pytest.raises(ValueError):
        atomic_write_yaml(tmp_path / "theses" / ".." / "escape.yaml", {"schema_version": 1}, tmp_path / "theses")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "theses" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        atomic_write_yaml(tmp_path / "theses" / "link" / "evil.yaml", {"schema_version": 1}, tmp_path / "theses")
    assert not (outside / "evil.yaml").exists()


def test_symlinked_thesis_dir_quarantined_no_lock_outside(tmp_path: Path) -> None:
    import shutil

    r = ThesisRepository(tmp_path / "theses")
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["demand remains strong"])
    real = tmp_path / "theses" / t.slug
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.move(str(real), str(outside / "real"))
    (outside / "real" / ".lock").unlink(missing_ok=True)
    real.symlink_to(outside / "real", target_is_directory=True)
    with pytest.raises(ValueError):
        r.update_thesis(t.thesis_id, scope="CHANGED")
    assert t.slug in r.list_quarantine()
    assert not (outside / "real" / ".lock").exists()
    assert not (outside / ".lock").exists()
    assert list((outside / "real" / "journal").glob("*.md")) == []


def test_concurrent_writes_serialized(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("concurrent thesis", scope="NVDA", claims=["c"])
    threads = [
        threading.Thread(
            target=r.append_journal_entry, args=(t.thesis_id, {"entry_id": f"entry:{i}", "title": f"t{i}", "body": "b"})
        )
        for i in range(10)
    ]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert len(list((tmp_path / "theses" / t.slug / "journal").glob("*.md"))) == 10


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Thesis.from_dict(
            {
                "schema_version": 1,
                "thesis_id": "thesis:x",
                "slug": "s",
                "status": "active",
                "created_at": "",
                "updated_at": "",
                "user_thesis": "u",
                "scope": "unknown",
                "claims": [
                    {"claim_id": "dup", "statement": "a", "status": "unvalidated"},
                    {"claim_id": "dup", "statement": "b", "status": "unvalidated"},
                ],
                "assumptions": [],
                "invalidators": [],
                "unknowns": [],
                "expressions": [],
                "requirements": [],
            },
            "<t>",
        )
    r = _repo(tmp_path)
    r.create_thesis("t", scope="NVDA", claims=["c"])
    with pytest.raises(ValueError):
        Thesis.from_dict(
            {
                "schema_version": 1,
                "thesis_id": "thesis:y",
                "slug": "s",
                "status": "active",
                "created_at": "",
                "updated_at": "",
                "user_thesis": "u",
                "scope": "unknown",
                "claims": [],
                "assumptions": [],
                "invalidators": [],
                "unknowns": [],
                "expressions": [{"expression_id": "dup", "structure": "a"}, {"expression_id": "dup", "structure": "b"}],
                "requirements": [],
            },
            "<t>",
        )


def test_cross_thesis_refs_rejected(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    with pytest.raises(ValueError):
        r.create_trigger(
            t.thesis_id, claim_ids=["claim:absent"], canonical_refs=["x"], summary="s", summary_origin="deterministic"
        )
    with pytest.raises(ValueError):
        r.apply_research_result(
            t.thesis_id,
            {"watch_add": [{"rule_id": "rule:x", "rule_type": "new_external_evidence", "claim_ids": ["claim:absent"]}]},
            "run:x",
        )


@pytest.mark.parametrize("use_slug", [False, True], ids=["canonical-id", "slug"])
def test_pause_resume_close_transitions_enforced(tmp_path: Path, use_slug: bool) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"], effective_at="2026-01-01T00:00:00+00:00")
    identifier = t.slug if use_slug else t.thesis_id
    thesis_dir = r.dir_for_thesis(t.thesis_id)
    for operation, status, effective_at in (
        (r.pause_thesis, "paused", "2026-01-02T00:00:00+00:00"),
        (r.resume_thesis, "active", "2026-01-03T00:00:00+00:00"),
        (r.close_thesis, "closed", "2026-01-04T00:00:00+00:00"),
    ):
        updated = operation(identifier, effective_at=effective_at)
        assert (updated.thesis_id, updated.slug, updated.status) == (t.thesis_id, t.slug, status)
        persisted = load_raw_yaml(thesis_dir / "thesis.yaml")
        assert (persisted["thesis_id"], persisted["status"]) == (t.thesis_id, status)
        snapshot = r.load_state_as_of(t.thesis_id, effective_at)
        assert (snapshot.thesis_id, snapshot.effective_at, snapshot.reason) == (
            t.thesis_id,
            effective_at,
            "status_changed",
        )
        assert (snapshot.thesis["thesis_id"], snapshot.thesis["status"]) == (t.thesis_id, status)
        for section in (snapshot.state, snapshot.questions, snapshot.watch, snapshot.memory):
            assert section["thesis_id"] == t.thesis_id
    assert r.close_thesis(identifier).status == "closed"
    with pytest.raises(ValueError, match="closed"):
        r.pause_thesis(identifier)
    with pytest.raises(ValueError, match="closed"):
        r.resume_thesis(identifier)
    with pytest.raises(ValueError):
        r.create_trigger(t.thesis_id, canonical_refs=["x"], summary="s", summary_origin="deterministic")


@pytest.mark.parametrize("action", ["pause", "resume", "close"])
@pytest.mark.parametrize("identifier", ["thesis:absent", "absent-slug"])
def test_status_unknown_identifier_does_not_write(tmp_path: Path, action: str, identifier: str) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    thesis_file = r.dir_for_thesis(t.thesis_id) / "thesis.yaml"
    before = thesis_file.read_text(encoding="utf-8")
    versions = r.list_state_versions(t.thesis_id)
    operation = {"pause": r.pause_thesis, "resume": r.resume_thesis, "close": r.close_thesis}[action]
    with pytest.raises(KeyError, match="unknown thesis"):
        operation(identifier)
    assert thesis_file.read_text(encoding="utf-8") == before
    assert r.list_state_versions(t.thesis_id) == versions


@pytest.mark.parametrize("action,status", [("pause", "paused"), ("resume", "active"), ("close", "closed")])
def test_slug_status_backdated_preserves_live_state(tmp_path: Path, action: str, status: str) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"], effective_at="2026-01-01T00:00:00+00:00")
    r.pause_thesis(t.thesis_id, effective_at="2026-01-01T12:00:00+00:00")
    r.resume_thesis(t.thesis_id, effective_at="2026-01-04T00:00:00+00:00")
    thesis_file = r.dir_for_thesis(t.thesis_id) / "thesis.yaml"
    before = thesis_file.read_text(encoding="utf-8")
    effective_at = "2026-01-02T00:00:00+00:00"
    operation = {"pause": r.pause_thesis, "resume": r.resume_thesis, "close": r.close_thesis}[action]
    updated = operation(t.slug, effective_at=effective_at)
    assert (updated.thesis_id, updated.status, updated.updated_at) == (t.thesis_id, status, effective_at)
    assert thesis_file.read_text(encoding="utf-8") == before
    snapshot = r.load_state_as_of(t.thesis_id, effective_at)
    assert (snapshot.thesis_id, snapshot.effective_at, snapshot.thesis["status"]) == (t.thesis_id, effective_at, status)
    assert snapshot.thesis["thesis_id"] == t.thesis_id
    assert r.load_state_as_of(t.thesis_id, "2026-01-04T00:00:00+00:00").thesis["status"] == "active"


@pytest.mark.parametrize("action", ["pause", "resume", "close"])
def test_slug_status_rechecks_owner_after_directory_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    foreign = r.create_thesis("other thesis", scope="AMD", claims=["c"])
    thesis_dir = r.dir_for_thesis(t.thesis_id)
    thesis_file = thesis_dir / "thesis.yaml"
    versions = r.list_state_versions(t.thesis_id)
    history = {p.name: p.read_text(encoding="utf-8") for p in (thesis_dir / "history").glob("*.yaml")}
    dir_for = r._dir_for
    swapped: list[str] = []

    def replace_owner(thesis_id: str) -> Path:
        resolved = dir_for(thesis_id)
        assert thesis_id == t.thesis_id
        raw = load_raw_yaml(thesis_file)
        raw["thesis_id"] = foreign.thesis_id
        atomic_write_yaml(thesis_file, raw, r.root)
        swapped.append(thesis_file.read_text(encoding="utf-8"))
        return resolved

    monkeypatch.setattr(r, "_dir_for", replace_owner)
    operation = {"pause": r.pause_thesis, "resume": r.resume_thesis, "close": r.close_thesis}[action]
    with pytest.raises(ValueError, match="thesis_id mismatch") as error:
        operation(t.slug)
    assert repr(t.thesis_id) in str(error.value)
    assert repr(foreign.thesis_id) in str(error.value)
    assert len(swapped) == 1
    assert thesis_file.read_text(encoding="utf-8") == swapped[0]
    assert {p.name: p.read_text(encoding="utf-8") for p in (thesis_dir / "history").glob("*.yaml")} == history
    assert len(history) == len(versions)


def test_cli_slug_status_uses_canonical_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import cli

    r = ThesisRepository(tmp_path / "thesis")
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    parser = cli._build_parser()
    for action, status in (("pause", "paused"), ("resume", "active"), ("close", "closed")):
        args = parser.parse_args(["thesis", action, t.slug, "--data-root", str(tmp_path)])
        assert cli._dispatch_command(args) is True
        assert capsys.readouterr().out == f"{t.thesis_id} ({t.slug}) [{status}]\n"
        assert r.load_thesis(t.thesis_id).status == status
    args = parser.parse_args(["thesis", "pause", "absent-slug", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="unknown thesis"):
        cli._dispatch_command(args)
    assert r.load_thesis(t.thesis_id).status == "closed"


def test_tool_slug_status_uses_canonical_id(tmp_path: Path) -> None:
    from app.policy import Capability, RequestContext
    from app.tools import execute_tool

    r = ThesisRepository(tmp_path / "thesis")
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    context = RequestContext("test", frozenset({Capability.RESEARCH}), data_root=tmp_path)
    for action, status in (("pause", "paused"), ("resume", "active"), ("close", "closed")):
        result = execute_tool("thesis_status", {"id": t.slug, "action": action}, "m", context=context)
        assert result == {"thesis_id": t.thesis_id, "slug": t.slug, "status": status, "action": action}
        assert r.load_thesis(t.thesis_id).status == status
    result = execute_tool("thesis_status", {"id": "absent-slug", "action": "pause"}, "m", context=context)
    assert "unknown thesis" in str(result["error"])
    assert r.load_thesis(t.thesis_id).status == "closed"


def test_quarantine_lists_corrupt_child_while_healthy_ids_proceed(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    a = r.create_thesis("healthy thesis", scope="NVDA", claims=["c"])
    b = r.create_thesis("doomed thesis", scope="NVDA", claims=["c"])
    bfile = tmp_path / "theses" / b.slug / "thesis.yaml"
    bfile.write_text("{ unclosed: [,,,\n", encoding="utf-8")
    assert [x.thesis_id for x in r.list_theses()] == [a.thesis_id]
    assert r.load_thesis(a.thesis_id).slug == a.slug
    assert r.load_thesis(a.slug).thesis_id == a.thesis_id
    with pytest.raises(ValueError):
        r.load_thesis(b.slug)
    with pytest.raises(ValueError, match="quarantined"):
        r.load_thesis(b.thesis_id)
    q = r.list_quarantine()
    assert set(q) == {b.slug} and q[b.slug]
    bfile.write_text(yaml.safe_dump({"schema_version": 99, "thesis_id": b.thesis_id}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        r.load_thesis(b.thesis_id)


def test_duplicate_thesis_ids_stay_loud(tmp_path: Path) -> None:
    import shutil

    r = _repo(tmp_path)
    a = r.create_thesis("first thesis", scope="NVDA", claims=["c"])
    shutil.copytree(tmp_path / "theses" / a.slug, tmp_path / "theses" / "clone-dir")
    with pytest.raises(ValueError, match="duplicate thesis ID"):
        r.list_theses()


def test_answer_questions_marks_answered(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("q thesis", scope="NVDA", claims=["c"])
    r.answer_questions(t.thesis_id, [])  # empty is a no-op
    r.apply_research_result(t.thesis_id, {"questions_add": [{"question_id": "q:1", "text": "why?"}]}, "run:x")
    r.answer_questions(t.thesis_id, [{"question_id": "q:1", "answer": "because"}])
    (q,) = r.load_questions(t.thesis_id)
    assert (q.status, q.answer) == ("answered", "because")
    r.apply_research_result(t.thesis_id, {"questions_add": [{"question_id": "q:2", "text": "when?"}]}, "run:x")
    r.answer_questions(t.thesis_id, [{"question_id": "q:2", "answer": "soon"}])
    assert {q.question_id: (q.status, q.answer) for q in r.load_questions(t.thesis_id)} == {
        "q:1": ("answered", "because"),
        "q:2": ("answered", "soon"),
    }
    with pytest.raises(ValueError):
        r.answer_questions(t.thesis_id, [{"question_id": "q:absent", "answer": "x"}])


def test_normalize_watch_heals_and_load_watch_rules(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("w thesis", scope="NVDA", claims=["c"])
    cid = r.load_thesis(t.thesis_id).claims[0].claim_id
    watch = tmp_path / "theses" / t.slug / "watch.yaml"
    raw = load_raw_yaml(watch)
    rules = raw["rules"]
    assert isinstance(rules, list)
    rules.append(
        {
            "rule_id": "rule:odd",
            "rule_type": "price_moon",
            "enabled": True,
            "support_status": "supported",
            "support_reason": "",
            "claim_ids": [cid],
            "expression_ids": [],
        }
    )
    atomic_write_yaml(watch, raw, tmp_path / "theses")
    r.normalize_watch(t.thesis_id)
    (rule,) = r.load_watch_rules(t.thesis_id)
    assert rule.rule_id == "rule:odd" and rule.enabled is False
    assert rule.support_status == "unsupported" and rule.support_reason


def _evidence_files(repo: ThesisRepository, thesis_id: str) -> dict[str, object]:
    thesis = repo.load_thesis(thesis_id)
    evdir = repo.dir_for_thesis(thesis.thesis_id) / "evidence"
    out: dict[str, object] = {}
    for f in sorted(evdir.glob("*.yaml")):
        raw = load_raw_yaml(f)
        if isinstance(raw, dict) and raw.get("thesis_id") == thesis_id:
            eid = raw.get("evidence_id")
            assert isinstance(eid, str)
            out[eid] = raw
    return out


def test_provenance_bound_rejects_forged_future_ref_but_keeps_visible(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    cutoff = "2026-01-01T10:00:00+00:00"
    journal = {"entry_id": "journal:hist", "title": "t", "body": "b", "known_at": cutoff}
    r.apply_research_result(
        t.thesis_id,
        {
            "evidence_refs": [
                {
                    "evidence_id": "ev:visible",
                    "canonical_ref": "V",
                    "summary": "v",
                    "known_at": "2026-01-01T09:00:00+00:00",
                },
                {
                    "evidence_id": "ev:future",
                    "canonical_ref": "F",
                    "summary": "f",
                    "known_at": "2026-02-01T00:00:00+00:00",
                },
            ]
        },
        "run:seed",
    )
    trig = r.create_trigger(t.thesis_id, canonical_refs=["V"], summary="s", summary_origin="deterministic")
    with pytest.raises(ValueError, match="foreign canonical_ref"):
        r.apply_research_result(
            t.thesis_id,
            {
                "trigger_id": trig.trigger_id,
                "evidence_refs": [
                    {
                        "evidence_id": "ev:forged",
                        "canonical_ref": "F",
                        "summary": "forged",
                        "known_at": "2026-01-01T09:30:00+00:00",
                    }
                ],
                "journal_entry": dict(journal),
            },
            "run:x",
        )
    stored = _evidence_files(r, t.thesis_id)
    assert "ev:forged" not in stored and _as_dict(stored["ev:future"])["canonical_ref"] == "F"
    out = r.apply_research_result(
        t.thesis_id,
        {
            "trigger_id": trig.trigger_id,
            "evidence_refs": [{"evidence_id": "ev:ok", "canonical_ref": "V", "summary": "v2", "known_at": cutoff}],
            "journal_entry": dict(journal, entry_id="journal:hist-ok"),
        },
        "run:y",
    )
    assert out["evidence"] == 1
    stored = _evidence_files(r, t.thesis_id)
    assert _as_dict(stored["ev:ok"])["canonical_ref"] == "V"


def test_file_symlink_ignored_not_quarantined(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    t = r.create_thesis("NVDA thesis", scope="NVDA", claims=["c"])
    (tmp_path / "theses").mkdir(exist_ok=True)
    target = tmp_path / "theses" / t.slug / "thesis.yaml"
    (tmp_path / "theses" / "README.link").symlink_to(target)
    assert "README.link" not in r.list_quarantine()
    assert r.load_thesis(t.thesis_id).slug == t.slug
    with pytest.raises(KeyError):
        r.load_thesis("thesis:absent")
