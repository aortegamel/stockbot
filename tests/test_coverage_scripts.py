"""Exercise coverage commands without running language test suites."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.normalize_lcov import normalize_lcov

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
REPORTS = [f"coverage-ts/{workspace}/lcov.info" for workspace in ("decision", "harness", "web")]


def _executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _run_script(name: str, root: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "-c", SCRIPTS[name]],
        cwd=root,
        env={**os.environ, "PATH": f"{root / 'tools'}:{os.environ['PATH']}", "TEST_ROOT": str(root), **environment},
        text=True,
        capture_output=True,
        check=False,
    )


def test_normalizer_maps_sources_without_changing_coverage_records(tmp_path: Path) -> None:
    for name in ("decision/client.ts", "needle-harness/lib/client.ts", "web/src/view.ts"):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    records = "FN:1,render\nFNDA:2,render\nDA:1,2\nDA:2,0\nBRDA:2,0,0,0\nLF:2\nLH:1\nend_of_record\n"
    for workspace, sources in (
        (".", ["decision/client.ts"]),
        ("needle-harness", ["lib/client.ts"]),
        ("web", ["src/view.ts", "../needle-harness/lib/client.ts", str(tmp_path / "decision/client.ts")]),
    ):
        report = tmp_path / "lcov.info"
        report.write_text("TN:\n" + "".join(f"SF:{source}\n{records}" for source in sources), encoding="utf-8")
        normalize_lcov(report, Path(workspace), tmp_path)
        expected = [(tmp_path / workspace / source).resolve().relative_to(tmp_path).as_posix() for source in sources]
        assert report.read_text(encoding="utf-8") == "TN:\n" + "".join(f"SF:{source}\n{records}" for source in expected)


@pytest.mark.parametrize("invalid", ["", "TN:\n", "SF:\n", "SF:missing.ts\n", "SF:.\n", "SF:../outside.ts\n"])
def test_normalizer_rejects_invalid_sources_without_rewriting_report(tmp_path: Path, invalid: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "valid.ts").touch()
    (tmp_path / "outside.ts").touch()
    report = root / "lcov.info"
    raw = invalid if not invalid.startswith("SF:") else "SF:valid.ts\nDA:1,1\nend_of_record\n" + invalid
    report.write_text(raw, encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        normalize_lcov(report, Path("."), root)
    assert report.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("status", [0, 23])
def test_python_coverage_keeps_branch_scope_and_propagates_test_failure(tmp_path: Path, status: int) -> None:
    report = tmp_path / "coverage-python.lcov"
    report.write_text("stale", encoding="utf-8")
    _executable(
        tmp_path / "venv/bin/coverage",
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path('calls.jsonl').open('a') as calls:\n"
        "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'run':\n"
        "    sys.exit(int(os.environ['TEST_STATUS']))\n"
        "if sys.argv[1] == 'lcov':\n"
        "    Path(sys.argv[-1]).write_text('SF:app/example.py\\nDA:1,1\\nend_of_record\\n')\n",
    )
    result = _run_script("coverage:py", tmp_path, TEST_STATUS=str(status))
    assert result.returncode == status, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls[:2] == [["erase"], ["run", "--branch", "-m", "pytest"]]
    if status:
        assert len(calls) == 2
        assert not report.exists()
    else:
        assert calls[2] == ["lcov", "-o", "coverage-python.lcov"]
        assert report.read_text().startswith("SF:app/example.py\n")


@pytest.fixture
def typescript_root(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/normalize_lcov.py", tmp_path / "scripts/normalize_lcov.py")
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/python").symlink_to(sys.executable)
    for name in ("decision/client.ts", "needle-harness/lib/client.ts", "web/src/view.ts"):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    for name in REPORTS:
        report = tmp_path / name
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("stale", encoding="utf-8")
    _executable(
        tmp_path / "tools/bun",
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['TEST_ROOT'])\n"
        "workspace = Path.cwd().relative_to(root).as_posix()\n"
        "with (root / 'calls.jsonl').open('a') as calls:\n"
        "    calls.write(json.dumps([workspace, sys.argv[1:]]) + '\\n')\n"
        "if os.environ.get('FAIL_WORKSPACE') == workspace:\n"
        "    sys.exit(29)\n"
        "if os.environ.get('OMIT_REPORT') == workspace:\n"
        "    sys.exit(0)\n"
        "directory = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--coverage-dir='))\n"
        "report = Path(directory) / 'lcov.info'\n"
        "report.parent.mkdir(parents=True, exist_ok=True)\n"
        "source = {'.': 'decision/client.ts', 'needle-harness': 'lib/client.ts', 'web': 'src/view.ts'}[workspace]\n"
        "raw = f'TN:\\nSF:{source}\\nDA:1,2\\nend_of_record\\n'\n"
        "if workspace == 'web':\n"
        "    raw += 'SF:../needle-harness/lib/client.ts\\nDA:1,0\\nend_of_record\\n'\n"
        "report.write_text(raw)\n",
    )
    return tmp_path


def test_typescript_coverage_produces_three_distinct_root_relative_reports(typescript_root: Path) -> None:
    result = _run_script("coverage:ts", typescript_root)
    assert result.returncode == 0, result.stderr
    expected = [
        ["decision/client.ts"],
        ["needle-harness/lib/client.ts"],
        ["web/src/view.ts", "needle-harness/lib/client.ts"],
    ]
    for name, sources in zip(REPORTS, expected, strict=True):
        raw = (typescript_root / name).read_text()
        assert [line[3:] for line in raw.splitlines() if line.startswith("SF:")] == sources
        assert "DA:1,2\n" in raw
        assert "stale" not in raw
    calls = [json.loads(line) for line in (typescript_root / "calls.jsonl").read_text().splitlines()]
    assert [workspace for workspace, _ in calls] == [".", "needle-harness", "web"]
    assert calls[0][1][:2] == ["test", "./decision/"]
    for _, args in calls[1:]:
        assert args[:3] == ["test", "--coverage", "--coverage-reporter=lcov"]


@pytest.mark.parametrize("workspace,index", [(".", 0), ("needle-harness", 1), ("web", 2)])
@pytest.mark.parametrize("failure", ["FAIL_WORKSPACE", "OMIT_REPORT"])
def test_typescript_coverage_never_uses_stale_reports(
    typescript_root: Path, workspace: str, index: int, failure: str
) -> None:
    result = _run_script("coverage:ts", typescript_root, **{failure: workspace})
    assert result.returncode != 0
    if failure == "FAIL_WORKSPACE":
        assert result.returncode == 29
    for name in REPORTS[index:]:
        assert not (typescript_root / name).exists()
    calls = [json.loads(line) for line in (typescript_root / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == index + 1


@pytest.mark.parametrize("command", ["crap", "crap:changed"])
def test_crap_reads_every_report_that_coverage_generates(typescript_root: Path, command: str) -> None:
    result = _run_script("coverage:ts", typescript_root)
    assert result.returncode == 0, result.stderr
    (typescript_root / "coverage-python.lcov").write_text("SF:app/example.py\nDA:1,1\nend_of_record\n")
    _executable(
        typescript_root / "tools/poly-crap",
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "reports = [args[i + 1] for i, arg in enumerate(args) if arg == '--coverage']\n"
        "print(json.dumps({'args': args, 'reports': {name: Path(name).read_text() for name in reports}}))\n",
    )
    result = _run_script(command, typescript_root)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert list(observed["reports"]) == ["coverage-python.lcov", *REPORTS]
    assert "SF:decision/client.ts\n" in observed["reports"][REPORTS[0]]
    assert "SF:needle-harness/lib/client.ts\n" in observed["reports"][REPORTS[1]]
    assert "SF:web/src/view.ts\n" in observed["reports"][REPORTS[2]]
    if command == "crap:changed":
        assert observed["args"][:3] == ["--diff-base", "main", "--fail-above"]


@pytest.mark.parametrize("manifest,status", [(False, 0), (True, 0), (True, 17)])
def test_optional_rust_coverage_removes_stale_report_and_preserves_failures(
    tmp_path: Path, manifest: bool, status: int
) -> None:
    report = tmp_path / "coverage-rust.lcov"
    report.write_text("stale", encoding="utf-8")
    if manifest:
        (tmp_path / "Cargo.toml").touch()
    _executable(
        tmp_path / "tools/cargo",
        "import os, sys\n"
        "from pathlib import Path\n"
        "status = int(os.environ['TEST_STATUS'])\n"
        "if status:\n"
        "    sys.exit(status)\n"
        "Path(sys.argv[-1]).write_text('SF:src/lib.rs\\nDA:1,1\\nend_of_record\\n')\n",
    )
    result = _run_script("coverage:rs", tmp_path, TEST_STATUS=str(status))
    assert result.returncode == status, result.stderr
    if manifest and not status:
        assert report.read_text() == "SF:src/lib.rs\nDA:1,1\nend_of_record\n"
    else:
        assert not report.exists()
