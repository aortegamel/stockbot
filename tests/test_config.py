"""Configuration validation stays aligned with the documented .env template."""

import importlib.util
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from app import config
from app.robinhood.auth import OAuthStoreError


@pytest.mark.parametrize(
    "value",
    ("Your Name your.email@example.com", "YourName [EMAIL]", "  sk-or-...  ", "your_finra_client_id"),
)
def test_require_env_rejects_documented_placeholder_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("TEST_SETTING", value)

    with pytest.raises(ValueError, match="TEST_SETTING is not properly set"):
        config._require_env("TEST_SETTING")


def test_require_env_strips_and_returns_configured_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_SETTING", "  configured-value  ")

    assert config._require_env("TEST_SETTING") == "configured-value"


@pytest.mark.parametrize(
    "url",
    (
        "http://agent.robinhood.com/mcp/trading",
        "https://agent.robinhood.com:8443/mcp/trading",
        "https://agent.robinhood.com/other",
        "https://user@agent.robinhood.com/mcp/trading",
        "https://agent.robinhood.com/mcp/trading?x=1",
    ),
)
def test_robinhood_endpoint_is_pinned(monkeypatch: pytest.MonkeyPatch, url: str):
    monkeypatch.setenv("ROBINHOOD_MCP_URL", url)
    with pytest.raises(OAuthStoreError, match="Robinhood MCP URL"):
        config.get_robinhood_mcp_url()


def test_broker_enabled_prefers_new_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BROKER_ENABLED", "true")
    monkeypatch.setenv("ROBINHOOD_ENABLED", "false")
    assert config.broker_enabled() is True


def test_broker_enabled_explicit_new_false_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BROKER_ENABLED", "false")
    monkeypatch.setenv("ROBINHOOD_ENABLED", "true")
    assert config.broker_enabled() is False


def test_broker_enabled_legacy_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BROKER_ENABLED", raising=False)
    monkeypatch.setenv("ROBINHOOD_ENABLED", "true")
    assert config.broker_enabled() is True


def test_broker_enabled_defaults_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BROKER_ENABLED", raising=False)
    monkeypatch.delenv("ROBINHOOD_ENABLED", raising=False)
    assert config.broker_enabled() is False


def _root_and_foreign_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_env: str) -> None:
    """Point config at a fixture repo root and run from a foreign cwd with its own .env."""
    root = tmp_path / "repo"
    foreign = tmp_path / "elsewhere"
    root.mkdir()
    foreign.mkdir()
    (root / ".env").write_text(root_env)
    (foreign / ".env").write_text("OPENCODE_MODEL=cwd-model\nSTOCKBOT_ENVTEST_CWD_ONLY=leaked\nPORT=9999\n")
    monkeypatch.setattr(config, "REPO_ROOT", root)
    monkeypatch.chdir(foreign)


def _absent(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    # setenv first so monkeypatch restores the original state after the loader writes it.
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)


def test_root_dotenv_replaces_stale_inherited_opencode_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _root_and_foreign_cwd(
        tmp_path,
        monkeypatch,
        "OPENCODE_API_KEY=file-key\nOPENCODE_URL=https://file.invalid/v1\nOPENCODE_MODEL=file-model\n",
    )
    monkeypatch.setenv("OPENCODE_API_KEY", "stale-key")
    monkeypatch.setenv("OPENCODE_URL", "https://stale.invalid/v1")
    monkeypatch.setenv("OPENCODE_MODEL", "stale-model")
    monkeypatch.setenv("STOCKBOT_ENVTEST_INHERITED", "kept")
    _absent(monkeypatch, "STOCKBOT_ENVTEST_CWD_ONLY")

    config._load_root_dotenv()

    assert os.environ["OPENCODE_API_KEY"] == "file-key"
    assert os.environ["OPENCODE_URL"] == "https://file.invalid/v1"
    assert os.environ["OPENCODE_MODEL"] == "file-model"
    assert os.environ["STOCKBOT_ENVTEST_INHERITED"] == "kept"
    assert "STOCKBOT_ENVTEST_CWD_ONLY" not in os.environ


def test_root_dotenv_keeps_inherited_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _root_and_foreign_cwd(tmp_path, monkeypatch, "PORT=3000\n")
    monkeypatch.setenv("PORT", "3101")

    config._load_root_dotenv()

    assert os.environ["PORT"] == "3101"


def test_root_dotenv_sets_port_when_not_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _root_and_foreign_cwd(tmp_path, monkeypatch, "PORT=3000\n")
    _absent(monkeypatch, "PORT")

    config._load_root_dotenv()

    assert os.environ["PORT"] == "3000"


@pytest.mark.parametrize(("line", "expected"), (("OPENCODE_API_KEY=\n", ""), ("OPENCODE_API_KEY\n", "stale-key")))
def test_root_dotenv_empty_value_clears_and_bare_name_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, expected: str
):
    _root_and_foreign_cwd(tmp_path, monkeypatch, line)
    monkeypatch.setenv("OPENCODE_API_KEY", "stale-key")

    config._load_root_dotenv()

    assert os.environ["OPENCODE_API_KEY"] == expected


def test_fresh_config_import_loads_root_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Executing app/config.py as if it lived under a fixture repo loads that repo's .env at import."""
    _root_and_foreign_cwd(tmp_path, monkeypatch, "OPENCODE_MODEL=file-model\n")
    monkeypatch.setenv("OPENCODE_MODEL", "stale-model")
    _absent(monkeypatch, "STOCKBOT_ENVTEST_CWD_ONLY")
    name = "app._config_import_probe"
    # Real source, fixture location: __file__ (and so REPO_ROOT) points at tmp_path/repo.
    loader = SourceFileLoader(name, config.__file__)
    spec = importlib.util.spec_from_file_location(name, tmp_path / "repo" / "app" / "config.py", loader=loader)
    assert spec is not None
    probe = importlib.util.module_from_spec(spec)

    loader.exec_module(probe)

    assert probe.REPO_ROOT == (tmp_path / "repo").resolve()
    assert os.environ["OPENCODE_MODEL"] == "file-model"
    assert "STOCKBOT_ENVTEST_CWD_ONLY" not in os.environ


@pytest.mark.parametrize("value", [None, "", "   ", "auto", " auto "])
@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "decisions"])
def test_reasoning_fields_omits_automatic_effort(
    monkeypatch: pytest.MonkeyPatch, value: str | None, protocol: str
) -> None:
    if value is None:
        monkeypatch.delenv("REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("REASONING_EFFORT", value)
    if protocol == "responses":
        assert config.reasoning_fields("responses", "test") == {}
    elif protocol == "chat_completions":
        assert config.reasoning_fields("chat_completions", "test") == {}
    else:
        assert config.reasoning_fields("decisions", "test") == {}


@pytest.mark.parametrize("value", ["low", "medium", "high", "xhigh", "provider-level"])
def test_reasoning_fields_preserves_concrete_effort(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("REASONING_EFFORT", f" {value} ")
    assert config.reasoning_fields("responses", "test") == {"reasoning": {"effort": value}}
    assert config.reasoning_fields("chat_completions", "test") == {"reasoning_effort": value}
    assert config.reasoning_fields("decisions", "test") == {}
