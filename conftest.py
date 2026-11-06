"""Root test configuration ensuring the project root is in the Python path."""

import os
import sys

# Add project root directory to sys.path so tests can import `app`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_runs_db(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point RUNS_DB_PATH at a temp DB so offline tests never touch data/runs.sqlite."""
    os.environ["RUNS_DB_PATH"] = str(tmp_path_factory.mktemp("runs") / "runs.sqlite")
    os.environ["STOCKBOT_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    # app.config loads the repo .env; without these the reasoner would answer tool-step decisions live.
    for key in ("OPENCODE_API_KEY", "OPENCODE_URL", "OPENCODE_MODEL"):
        os.environ.pop(key, None)
