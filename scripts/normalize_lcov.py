"""Rewrite Bun LCOV source paths relative to the repository root."""

import sys
from pathlib import Path


def normalize_lcov(report: Path, workspace: Path, root: Path) -> None:
    root = root.resolve()
    workspace = (root / workspace).resolve()
    lines = report.read_text(encoding="utf-8").splitlines(keepends=True)
    sources = 0
    for index, line in enumerate(lines):
        if not line.startswith("SF:"):
            continue
        name = line[3:].rstrip("\r\n")
        if not name:
            raise ValueError(f"{report}: empty LCOV source path")
        source = (workspace / name).resolve(strict=True)
        relative = source.relative_to(root)
        if not source.is_file():
            raise ValueError(f"{report}: LCOV source is not a file: {name}")
        lines[index] = f"SF:{relative.as_posix()}" + line[3 + len(name) :]
        sources += 1
    if not sources:
        raise ValueError(f"{report}: LCOV report contains no source paths")
    report.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    workspace_arg, report_arg = sys.argv[1:]
    normalize_lcov(Path(report_arg), Path(workspace_arg), Path(__file__).resolve().parents[1])
