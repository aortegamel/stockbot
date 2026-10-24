"""ArtifactStore: persist-on-acceptance raw payloads for admitted evidence.

Callers persist the exact official HTML/text bytes under artifacts/sha256_<hex>
only when the document became evidence or the upstream response is
non-reproducible (live quotes/trends/news ranking). SEC search lists,
opened-but-unused filings, PDFs/screenshots/attachments-never-shown stay
trace-metadata-only and are never archived here. Content-hash dedup keeps one
copy per payload; payload and manifest files are write-once and never mutated.
Opted-in callers also append each retrieval to a per-key retrievals.jsonl log.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import get_data_root

DEFAULT_RAW_ROOT = get_data_root() / "raw"

MANIFEST_SUFFIX = ".manifest.json"
PAYLOAD_SUFFIX = ".json"
# ".jsonl" never matches the "*.json" payload glob, so find()/iter_archive() skip the log.
RETRIEVAL_LOG = "retrievals.jsonl"
_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def content_hash(data: bytes) -> str:
    """sha256 of the exact payload bytes; the archive's dedup key."""
    return hashlib.sha256(data).hexdigest()


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


@dataclass(frozen=True)
class ArchiveRecord:
    """One archived payload plus its retrieval metadata."""

    source: str
    kind: str
    key: str
    sha256: str
    payload_path: Path
    manifest_path: Path
    size: int
    retrieved_at: str
    url: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalEvent:
    """One successful retrieval of an archived payload revision."""

    sha256: str
    retrieved_at: str
    url: str


class RetrievalLogError(ValueError):
    """The retrieval log is unreadable or holds a malformed event."""


def _retrieval_event(raw: object) -> RetrievalEvent:
    """Validate one log object; raise TypeError/ValueError instead of trusting arbitrary JSON."""
    if not isinstance(raw, dict):
        raise TypeError("retrieval event must be a JSON object")
    if set(raw) != {"sha256", "retrieved_at", "url"}:
        raise ValueError("retrieval event must have exactly sha256, retrieved_at, url")
    sha, stamp, url = raw["sha256"], raw["retrieved_at"], raw["url"]
    if not isinstance(sha, str) or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("retrieval event sha256 must be 64 lowercase hex characters")
    # Readers compare stamps as strings, so only the canonical UTC form is valid.
    if not isinstance(stamp, str) or time.strftime(_STAMP_FORMAT, time.strptime(stamp, _STAMP_FORMAT)) != stamp:
        raise ValueError(f"retrieval event retrieved_at must match {_STAMP_FORMAT}")
    if not isinstance(url, str):
        raise TypeError("retrieval event url must be a string")
    return RetrievalEvent(sha256=sha, retrieved_at=stamp, url=url)


def _manifest_path(payload_path: Path) -> Path:
    return payload_path.with_suffix(MANIFEST_SUFFIX)


def archive(
    source: str,
    kind: str,
    key: str,
    payload: bytes,
    *,
    url: str,
    retrieved_at: str | None = None,
    metadata: dict[str, object] | None = None,
    root: Path | None = None,
    record_retrieval: bool = False,
) -> ArchiveRecord:
    """Persist one accepted payload and its manifest (persist-on-acceptance).

    Archive only evidence-bearing docs or non-reproducible live responses;
    search lists, opened-but-unused filings, and PDFs/mirrors stay
    trace-metadata-only (do not call this for them). Idempotent: archiving the
    same bytes for the same (source, kind, key) returns the existing record
    without rewriting anything; content-hash dedup means no per-run copies.
    With ``record_retrieval``, every call (dedup hits too) then appends this
    retrieval's sha256, timestamp, and url to the key's retrieval log.
    """
    root = Path(root) if root else get_data_root() / "raw"
    digest = content_hash(payload)
    stamp = retrieved_at or _utc_now()
    event = _retrieval_event({"sha256": digest, "retrieved_at": stamp, "url": url}) if record_retrieval else None
    directory = root / _safe_component(source) / _safe_component(kind) / _safe_component(key)
    payload_path = directory / f"{digest[:16]}{PAYLOAD_SUFFIX}"
    manifest_path = _manifest_path(payload_path)
    directory.mkdir(parents=True, exist_ok=True)
    if not payload_path.exists():
        payload_path.write_bytes(payload)
    if not manifest_path.exists():
        manifest = {
            "source": source,
            "kind": kind,
            "key": key,
            "sha256": digest,
            "size": len(payload),
            "retrieved_at": stamp,
            "url": url,
            "metadata": metadata or {},
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    record = _load_record(payload_path, manifest_path)
    if event is not None:
        with (directory / RETRIEVAL_LOG).open("a", encoding="utf-8") as log:
            log.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    return record


def _utc_now() -> str:
    return time.strftime(_STAMP_FORMAT, time.gmtime())


def _load_record(payload_path: Path, manifest_path: Path) -> ArchiveRecord:
    raw: object = json.loads(manifest_path.read_text())
    manifest: dict[str, object] = raw if isinstance(raw, dict) else {}
    source_value = manifest.get("source", "")
    kind_value = manifest.get("kind", "")
    key_value = manifest.get("key", "")
    sha_value = manifest.get("sha256", "")
    retrieved_value = manifest.get("retrieved_at", "")
    url_value = manifest.get("url", "")
    size_value = manifest.get("size", payload_path.stat().st_size)
    metadata_value = manifest.get("metadata")
    return ArchiveRecord(
        source=source_value if isinstance(source_value, str) else "",
        kind=kind_value if isinstance(kind_value, str) else "",
        key=key_value if isinstance(key_value, str) else "",
        sha256=sha_value if isinstance(sha_value, str) else "",
        payload_path=payload_path,
        manifest_path=manifest_path,
        size=size_value if isinstance(size_value, int) else payload_path.stat().st_size,
        retrieved_at=retrieved_value if isinstance(retrieved_value, str) else "",
        url=url_value if isinstance(url_value, str) else "",
        metadata=metadata_value if isinstance(metadata_value, dict) else {},
    )


def find(
    source: str,
    kind: str,
    key: str,
    *,
    sha256: str | None = None,
    root: Path | None = None,
) -> ArchiveRecord | None:
    """Return the archived record for a (source, kind, key), optionally
    narrowed by content hash, or None."""
    root = Path(root) if root else get_data_root() / "raw"
    directory = root / _safe_component(source) / _safe_component(kind) / _safe_component(key)
    if not directory.is_dir():
        return None
    for payload_path in sorted(directory.glob(f"*{PAYLOAD_SUFFIX}")):
        if sha256 is not None and payload_path.stem != sha256[:16]:
            continue
        manifest_path = _manifest_path(payload_path)
        if manifest_path.is_file():
            return _load_record(payload_path, manifest_path)
    return None


def iter_archive(
    source: str,
    kind: str,
    key: str,
    *,
    root: Path | None = None,
) -> Iterable[ArchiveRecord]:
    """All archived payload revisions for one source key, oldest first."""
    root = Path(root) if root else get_data_root() / "raw"
    directory = root / _safe_component(source) / _safe_component(kind) / _safe_component(key)
    if not directory.is_dir():
        return
    for payload_path in sorted(directory.glob(f"*{PAYLOAD_SUFFIX}")):
        manifest_path = _manifest_path(payload_path)
        if manifest_path.is_file():
            yield _load_record(payload_path, manifest_path)


def iter_retrievals(
    source: str,
    kind: str,
    key: str,
    *,
    root: Path | None = None,
) -> Iterable[RetrievalEvent]:
    """Every retrieval of one source key: manifest first writes (legacy seeds), then the log in append order.

    A malformed log raises RetrievalLogError, so callers fail closed instead of
    skipping the newest event and reviving an older mapping.
    """
    for record in iter_archive(source, kind, key, root=root):
        yield RetrievalEvent(sha256=record.sha256, retrieved_at=record.retrieved_at, url=record.url)
    root = Path(root) if root else get_data_root() / "raw"
    log_path = root / _safe_component(source) / _safe_component(kind) / _safe_component(key) / RETRIEVAL_LOG
    if not log_path.exists():
        return
    try:
        lines = log_path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RetrievalLogError(f"{log_path}: not UTF-8") from exc
    for number, line in enumerate(lines, start=1):
        try:
            event = _retrieval_event(json.loads(line))
        except (TypeError, ValueError) as exc:
            raise RetrievalLogError(f"{log_path}:{number}: {exc}") from exc
        yield event


def has_payload(
    source: str,
    kind: str,
    key: str,
    sha256: str,
    *,
    root: Path | None = None,
) -> bool:
    """True when a payload with this content hash is already archived."""
    return find(source, kind, key, sha256=sha256, root=root) is not None


def read_payload(
    source: str,
    kind: str,
    key: str,
    *,
    sha256: str,
    root: Path | None = None,
) -> bytes | None:
    """Exact archived bytes of one pinned revision; None when absent or no longer hashing to the pin.

    Publication verification reads proof bytes here, so a missing or tampered
    payload fails closed instead of verifying.
    """
    record = find(source, kind, key, sha256=sha256, root=root)
    if record is None:
        return None
    payload = record.payload_path.read_bytes()
    return payload if content_hash(payload) == sha256 else None


def store_evidence_artifact(
    payload: bytes,
    *,
    url: str = "",
    retrieved_at: str | None = None,
    metadata: dict[str, object] | None = None,
    root: Path | None = None,
) -> ArchiveRecord:
    """ArtifactStore write: one evidence-bearing (or non-reproducible live) payload.

    Content-addressed under artifacts/sha256_<hex> with dedup; no per-run copies.
    Search lists, opened-but-unused filings, and PDFs/mirrors are never stored.
    """
    digest = content_hash(payload)
    return archive(
        "artifacts",
        "sha256",
        digest,
        payload,
        url=url,
        retrieved_at=retrieved_at,
        metadata=metadata,
        root=root,
    )


def should_archive(*, became_evidence: bool, non_reproducible: bool, is_pdf_or_mirror: bool = False) -> bool:
    """Persist-on-acceptance gate: evidence or non-reproducible live only; never PDFs/mirrors."""
    if is_pdf_or_mirror:
        return False
    return became_evidence or non_reproducible
