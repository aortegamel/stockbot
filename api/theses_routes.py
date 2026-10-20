"""Thesis monitor API (mounted by api.main): list, detail, check now, pause/resume.

Routes:
- GET /api/theses — summaries with alert counts (triggers fired / still pending).
- GET /api/theses/{id} — claims, assessment, watch rules, triggers, journal, last tick.
- POST /api/theses/{id}/check — one ``run_monitor_once`` against the live sources.
- POST /api/theses/{id}/pause, /resume — status changes.

No push: the UI polls. Unknown thesis -> 404; repository refusals -> 409.
"""

from operator import itemgetter
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_data_root
from app.thesis import monitor
from app.thesis.models import Checkpoint, Thesis
from app.thesis.repository import ThesisRepository
from app.thesis.worker import run_monitor_once

router = APIRouter(prefix="/api/theses")

_JOURNAL_LIMIT = 20


def _repo() -> ThesisRepository:
    """Thesis root under the shared data root (same as the CLI and tools)."""
    return ThesisRepository(get_data_root() / "thesis")


def _load(repo: ThesisRepository, thesis_id: str) -> Thesis:
    try:
        return repo.load_thesis(thesis_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown thesis: {thesis_id}") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _services(thesis: Thesis) -> dict[str, monitor.SourceService]:
    # ponytail: mirrors cli._thesis_runtime; hoist into monitor when a third caller appears.
    targets = monitor.targets_for_thesis(thesis)
    return {
        "sec_filings": monitor.SecFilingsService(targets, since_default=thesis.created_at),
        "material_events": monitor.MaterialEventsService(targets, since_default=thesis.created_at),
        "finra_short_interest": monitor.FinraShortInterestService(targets),
    }


def _summary(repo: ThesisRepository, thesis: Thesis) -> dict[str, object]:
    triggers = repo.load_triggers(thesis.thesis_id)
    return {
        "thesis_id": thesis.thesis_id,
        "slug": thesis.slug,
        "status": thesis.status,
        "scope": thesis.scope,
        "user_thesis": thesis.user_thesis,
        "updated_at": thesis.updated_at,
        "alerts": len(triggers),
        "pending": sum(t.status == "pending" for t in triggers),
    }


def _journal_entry(path: Path) -> dict[str, str]:
    """One journal file (front matter + "# title" + body) as a flat row."""
    head, _, rest = path.read_text(encoding="utf-8").removeprefix("---\n").partition("\n---\n")
    fields = {k: v for k, sep, v in (ln.partition(": ") for ln in head.splitlines()) if sep}
    title, _, body = rest.partition("\n")
    return {
        "entry_id": fields.get("entry_id", path.stem),
        "created_at": fields.get("created_at", ""),
        "trigger_id": fields.get("trigger_id", ""),
        "title": title.removeprefix("# "),
        "body": body.strip(),
    }


def _journal(jdir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [_journal_entry(p) for p in jdir.glob("*.md")] if jdir.is_dir() else []
    rows.sort(key=itemgetter("created_at"), reverse=True)
    return rows[:_JOURNAL_LIMIT]


def _last_tick(checkpoint: Checkpoint) -> str | None:
    # ponytail: ticks persist no own timestamp; newest source advance is the durable proxy.
    stamps = [str(s["updated_at"]) for s in checkpoint.sources.values() if isinstance(s, dict) and s.get("updated_at")]
    return max(stamps, default=None)


@router.get("")
def list_theses() -> list[dict[str, object]]:
    repo = _repo()
    return [_summary(repo, t) for t in repo.list_theses()]


@router.get("/{thesis_id}")
def get_thesis(thesis_id: str) -> dict[str, object]:
    repo = _repo()
    thesis = _load(repo, thesis_id)
    tid = thesis.thesis_id
    return {
        **_summary(repo, thesis),
        "claims": [c.to_dict() for c in thesis.claims],
        "assessment": repo.load_state(tid).assessment,
        "watch_rules": [r.to_dict() for r in repo.load_watch_rules(tid)],
        "triggers": [t.to_dict() for t in reversed(repo.load_triggers(tid))],
        "journal": _journal(repo.dir_for_thesis(tid) / "journal"),
        "last_tick": _last_tick(repo.load_checkpoint(tid)),
    }


@router.post("/{thesis_id}/check")
def check_thesis(thesis_id: str) -> dict[str, object]:
    repo = _repo()
    thesis = _load(repo, thesis_id)
    try:
        out = run_monitor_once(repo, thesis.thesis_id, _services(thesis))
    except ValueError as exc:  # closed thesis refuses to tick
        raise HTTPException(409, str(exc)) from exc
    return {
        "thesis_id": out.thesis_id,
        "triggers_created": out.triggers_created,
        "runs": [r.run_id for r in out.runs],
        "no_op": out.no_op,
        "no_op_reason": out.no_op_reason,
    }


def _set_status(thesis_id: str, action: str) -> dict[str, object]:
    repo = _repo()
    thesis = _load(repo, thesis_id)
    op = repo.pause_thesis if action == "pause" else repo.resume_thesis
    try:
        updated = op(thesis.thesis_id)
    except ValueError as exc:  # closed theses refuse status changes
        raise HTTPException(409, str(exc)) from exc
    return {"thesis_id": updated.thesis_id, "status": updated.status}


@router.post("/{thesis_id}/pause")
def pause_thesis(thesis_id: str) -> dict[str, object]:
    return _set_status(thesis_id, "pause")


@router.post("/{thesis_id}/resume")
def resume_thesis(thesis_id: str) -> dict[str, object]:
    return _set_status(thesis_id, "resume")
