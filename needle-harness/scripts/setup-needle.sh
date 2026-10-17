#!/usr/bin/env bash
set -euo pipefail
VENV_DIR="$HOME/.cache/needle-harness/.needle"
python3.14 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install cactus-needle
NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 "$VENV_DIR/bin/needle" download needle3
