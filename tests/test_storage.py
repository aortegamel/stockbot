"""Tests for the raw archive store and warehouse-removal gates."""

import importlib
import json
from pathlib import Path

import pytest

from app.storage import raw_archive


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def _payload(text: str) -> bytes:
    return json.dumps({"content": text}).encode()


# ---------------------------------------------------------------------------
# Raw archive
# ---------------------------------------------------------------------------


def test_archive_stores_payload_and_manifest(archive_root: Path):
    record = raw_archive.archive(
        "sec",
        "companyfacts",
        "cik0000320193",
        _payload("hello"),
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at="2026-08-21T12:00:00Z",
        metadata={"cik": "0000320193"},
        root=archive_root,
    )
    assert record.payload_path.is_file()
    assert record.manifest_path.is_file()
    assert record.sha256 == raw_archive.content_hash(_payload("hello"))
    assert record.size == len(_payload("hello"))
    assert record.url.endswith("CIK0000320193.json")
    assert record.metadata["cik"] == "0000320193"
    manifest = json.loads(record.manifest_path.read_text())
    assert manifest["sha256"] == record.sha256


def test_archive_is_immutable_and_idempotent(archive_root: Path):
    first = raw_archive.archive("finra", "data", "otc/cycle", _payload("x"), url="u", root=archive_root)
    second = raw_archive.archive("finra", "data", "otc/cycle", _payload("x"), url="u", root=archive_root)
    assert first.payload_path == second.payload_path
    assert first.manifest_path == second.manifest_path
    assert {p for p in archive_root.rglob("*.json")} == {second.payload_path, second.manifest_path}


def test_archive_keeps_distinct_payload_revisions(archive_root: Path):
    first = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v1"), url="u", root=archive_root)
    second = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v2"), url="u", root=archive_root)
    assert first.payload_path != second.payload_path
    revisions = list(raw_archive.iter_archive("sec", "companyfacts", "cik1", root=archive_root))
    assert [r.sha256 for r in revisions] == sorted(r.sha256 for r in revisions)


def test_find_and_has_payload(archive_root: Path):
    record = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v1"), url="u", root=archive_root)
    assert raw_archive.find("sec", "companyfacts", "cik1", root=archive_root) == record
    assert raw_archive.find("sec", "companyfacts", "cik1", sha256=record.sha256, root=archive_root) == record
    assert raw_archive.find("sec", "companyfacts", "cik1", sha256="0" * 64, root=archive_root) is None
    assert raw_archive.find("sec", "companyfacts", "nope", root=archive_root) is None
    assert raw_archive.has_payload("sec", "companyfacts", "cik1", record.sha256, root=archive_root)
    assert not raw_archive.has_payload("sec", "companyfacts", "cik1", "0" * 64, root=archive_root)


def _events(archive_root: Path, key: str = "k") -> list[tuple[str, str, str]]:
    return [(e.sha256, e.retrieved_at, e.url) for e in raw_archive.iter_retrievals("sec", "t", key, root=archive_root)]


def test_repeat_retrieval_keeps_original_record_and_logs_each_time(archive_root: Path):
    first = raw_archive.archive(
        "sec",
        "t",
        "k",
        _payload("a"),
        url="u1",
        retrieved_at="2026-01-01T00:00:00Z",
        record_retrieval=True,
        root=archive_root,
    )
    payload_bytes, manifest_bytes = first.payload_path.read_bytes(), first.manifest_path.read_bytes()
    second = raw_archive.archive(
        "sec",
        "t",
        "k",
        _payload("a"),
        url="u2",
        retrieved_at="2026-01-20T00:00:00Z",
        record_retrieval=True,
        root=archive_root,
    )
    assert second == first
    assert first.payload_path.read_bytes() == payload_bytes
    assert first.manifest_path.read_bytes() == manifest_bytes
    assert list(raw_archive.iter_archive("sec", "t", "k", root=archive_root)) == [first]
    assert _events(archive_root) == [
        (first.sha256, "2026-01-01T00:00:00Z", "u1"),
        (first.sha256, "2026-01-01T00:00:00Z", "u1"),
        (first.sha256, "2026-01-20T00:00:00Z", "u2"),
    ]


def test_dedup_hit_logs_invocation_time_not_manifest_time(archive_root: Path, monkeypatch: pytest.MonkeyPatch):
    stamps = iter(["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"])
    monkeypatch.setattr(raw_archive, "_utc_now", lambda: next(stamps))
    first = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    assert first.retrieved_at == "2026-01-01T00:00:00Z"
    assert [stamp for _, stamp, _ in _events(archive_root)] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
    ]


def test_a_b_a_retrievals_keep_every_hash_in_order(archive_root: Path):
    a, b = _payload("a"), _payload("b")
    for payload, stamp in ((a, "2026-01-01T00:00:00Z"), (b, "2026-01-02T00:00:00Z"), (a, "2026-01-03T00:00:00Z")):
        raw_archive.archive(
            "sec", "t", "k", payload, url="u", retrieved_at=stamp, record_retrieval=True, root=archive_root
        )
    sha_a, sha_b = raw_archive.content_hash(a), raw_archive.content_hash(b)
    assert {r.sha256 for r in raw_archive.iter_archive("sec", "t", "k", root=archive_root)} == {sha_a, sha_b}
    logged = _events(archive_root)[2:]
    assert [(sha, stamp) for sha, stamp, _ in logged] == [
        (sha_a, "2026-01-01T00:00:00Z"),
        (sha_b, "2026-01-02T00:00:00Z"),
        (sha_a, "2026-01-03T00:00:00Z"),
    ]


def test_same_timestamp_retrievals_stay_separate(archive_root: Path):
    for _ in range(2):
        raw_archive.archive(
            "sec",
            "t",
            "k",
            _payload("a"),
            url="u",
            retrieved_at="2026-01-01T00:00:00Z",
            record_retrieval=True,
            root=archive_root,
        )
    assert len(_events(archive_root)) == 3


def test_legacy_archive_without_log_yields_manifest_observation(archive_root: Path):
    record = raw_archive.archive(
        "sec", "t", "k", _payload("a"), url="u", retrieved_at="2026-01-01T00:00:00Z", root=archive_root
    )
    assert not (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).exists()
    assert _events(archive_root) == [(record.sha256, "2026-01-01T00:00:00Z", "u")]


def test_missing_archive_yields_no_retrievals(archive_root: Path):
    assert _events(archive_root, key="absent") == []


def test_record_retrieval_false_leaves_other_logs_alone(archive_root: Path):
    logged = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    log_path = logged.payload_path.parent / raw_archive.RETRIEVAL_LOG
    before = log_path.read_bytes()
    raw_archive.archive("sec", "t", "k", _payload("a"), url="u", root=archive_root)
    other = raw_archive.archive("sec", "t", "other", _payload("a"), url="u", root=archive_root)
    assert log_path.read_bytes() == before
    assert not (other.payload_path.parent / raw_archive.RETRIEVAL_LOG).exists()
    assert raw_archive.find("sec", "t", "k", root=archive_root) == logged


_VALID_EVENT = json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"})


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "",
        "[]",
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u", "extra": 1}),
        json.dumps({"sha256": "a" * 16, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"}),
        json.dumps({"sha256": "A" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-1-1T0:0:0Z", "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": 1, "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": None}),
    ],
)
def test_malformed_retrieval_log_fails_closed(archive_root: Path, line: str):
    record = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    with (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).open("a") as log:
        log.write(line + "\n" + _VALID_EVENT + "\n")
    with pytest.raises(raw_archive.RetrievalLogError):
        _events(archive_root)


def test_non_utf8_retrieval_log_fails_closed(archive_root: Path):
    record = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    with (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).open("ab") as log:
        log.write(b"\xff\n")
    with pytest.raises(raw_archive.RetrievalLogError):
        _events(archive_root)


def test_failed_retrieval_append_raises(archive_root: Path):
    (archive_root / "sec" / "t" / "k" / raw_archive.RETRIEVAL_LOG).mkdir(parents=True)
    with pytest.raises(OSError):
        raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)


def test_bad_retrieval_stamp_writes_nothing(archive_root: Path):
    with pytest.raises(ValueError):
        raw_archive.archive(
            "sec", "t", "k", _payload("a"), url="u", retrieved_at="yesterday", record_retrieval=True, root=archive_root
        )
    assert not archive_root.exists()


# ---------------------------------------------------------------------------
# Warehouse-removal gates
# ---------------------------------------------------------------------------


def test_migration_warehouse_modules_removed() -> None:
    """The warehouse query layer is gone; live providers serve reads now."""
    with pytest.raises(ImportError):
        importlib.import_module("app.storage.parquet")
    with pytest.raises(ImportError):
        importlib.import_module("app.storage.duckdb")


def test_migration_sec_facts_live_assembly() -> None:
    """sec_facts still exposes its live assembly helpers."""
    from app.services import sec_facts as _sec_facts

    assert _sec_facts._assemble_eps_payload is not None
    assert _sec_facts._store_rows is not None
