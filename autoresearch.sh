#!/usr/bin/env bash
# Registry/static checks + generated confusion benchmark (no frozen holdout tuning).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
export FINRA_USE_MOCK=1
export BROKER_ENABLED=0
TMPDIR_HARNESS="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_HARNESS"' EXIT
export RUNS_DB_PATH="$TMPDIR_HARNESS/runs.sqlite"
venv/bin/python scripts/verify_tool_registry.py
venv/bin/python scripts/strict_routing_harness.py
