"""scripts/doctor.py: .env parsing and env-only feature checks (no network, no subprocess)."""

from pathlib import Path

import pytest

from scripts import doctor

FULL = {
    "SEC_EDGAR_IDENTITY": "Ann Lee ann@example.org",
    "DECISION_API_FORMAT": "decisions",
    "DECISION_API_URL": "https://openrouter.ai/api/alpha/decisions",
    "DECISION_API_KEY": "k",
    "DECISION_MODEL": "typesafe/jev-1.13",
    "OPENCODE_API_KEY": "k",
    "OPENCODE_URL": "https://opencode.ai/zen/go/v1/responses",
    "OPENCODE_MODEL": "m",
}


def test_parse_dotenv_quotes_comments_and_export() -> None:
    text = "# c\nA=\"x # kept\"\nexport B=y # dropped\nC='z'\nD=\nnoequals\n"
    assert doctor.parse_dotenv(text) == {"A": "x # kept", "B": "y", "C": "z", "D": ""}


def test_load_env_dotenv_wins_except_inherited_port(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('X="file"\nPORT="3000"\n', encoding="utf-8")
    assert doctor.load_env({"X": "shell", "PORT": "3101"}, path) == {"X": "file", "PORT": "3101"}
    assert doctor.load_env({"X": "shell"}, tmp_path / "absent") == {"X": "shell"}


def test_required_features_pass_with_full_env_and_fail_on_placeholders() -> None:
    for check in (doctor.check_sec, doctor.check_decision, doctor.check_opencode):
        assert check(FULL)[0] == doctor.OK
    placeholder = {**FULL, "SEC_EDGAR_IDENTITY": "Your Name your.email@example.com"}
    assert doctor.check_sec(placeholder)[0] == doctor.FAIL
    assert "DECISION_MODEL" in doctor.check_decision({**FULL, "DECISION_MODEL": " "})[2]
    assert doctor.check_decision({**FULL, "DECISION_API_FORMAT": "auto"})[0] == doctor.FAIL
    assert doctor.check_decision({**FULL, "DECISION_API_URL": "http://x/d"})[0] == doctor.FAIL
    assert doctor.check_opencode({**FULL, "OPENCODE_URL": "https://x/v1"})[0] == doctor.FAIL


def test_optional_features_skip_when_off_and_fail_when_half_configured() -> None:
    assert doctor.check_exa({"EXA_ENABLED": "false"})[0] == doctor.SKIP
    assert doctor.check_exa({"EXA_ENABLED": "true", "EXA_API_KEY": "your_exa_api_key"})[0] == doctor.FAIL
    assert doctor.check_exa({"EXA_ENABLED": "1", "EXA_API_KEY": "k"})[0] == doctor.OK
    assert doctor.check_finra({"FINRA_CLIENT_ID": "your_finra_client_id"})[0] == doctor.SKIP
    assert doctor.check_finra({"FINRA_CLIENT_ID": "id"})[0] == doctor.FAIL
    assert doctor.check_finra({"FINRA_CLIENT_ID": "id", "FINRA_CLIENT_SECRET": "s"})[0] == doctor.OK
    assert doctor.check_broker({"BROKER_ENABLED": "false", "ROBINHOOD_ENABLED": "true"})[0] == doctor.SKIP


def test_api_hosts_follow_enabled_features() -> None:
    base = [("SEC EDGAR", doctor.SEC_HOST), ("Yahoo market data", doctor.YAHOO_HOST)]
    assert doctor.api_hosts({}) == base
    env = {**FULL, "EXA_ENABLED": "yes", "FINRA_CLIENT_ID": "id", "BROKER_ENABLED": "true"}
    assert doctor.api_hosts(env) == [
        *base,
        ("OpenCode", "opencode.ai"),
        ("JEV decision API", "openrouter.ai"),
        ("Exa", doctor.EXA_HOST),
        ("FINRA", doctor.FINRA_HOST),
        ("Robinhood", doctor.ROBINHOOD_HOST),
    ]


def test_ports_reject_non_integer() -> None:
    assert doctor.check_ports({"PORT": "abc"})[0] == doctor.FAIL


def test_bun_checks_path_dependencies_and_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subprocess import CompletedProcess

    def absent(command: str) -> None:
        return None

    def installed(command: str) -> str:
        return "/bin/bun"

    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(doctor.shutil, "which", absent)
    assert doctor.check_bun({}) == (
        doctor.FAIL,
        "bun",
        "not on PATH. Fix: curl -fsSL https://bun.sh/install | bash",
    )
    monkeypatch.setattr(doctor.shutil, "which", installed)
    assert doctor.check_bun({}) == (
        doctor.FAIL,
        "bun",
        "web, needle-harness dependencies missing. Fix: bun install --cwd web && bun install --cwd needle-harness",
    )
    (tmp_path / "web" / "node_modules").mkdir(parents=True)
    assert doctor.check_bun({}) == (
        doctor.FAIL,
        "bun",
        "needle-harness dependencies missing. Fix: bun install --cwd needle-harness",
    )
    (tmp_path / "needle-harness" / "node_modules").mkdir(parents=True)

    def run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert command == ["/bin/bun", "--version"]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 30, "check": False}
        return CompletedProcess(command, 0, stdout="1.3.0\n")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    assert doctor.check_bun({}) == (
        doctor.OK,
        "bun",
        "bun 1.3.0, web + needle-harness dependencies installed",
    )


def test_main_reports_all_checks_and_failure_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(doctor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(doctor.os, "environ", {"MARKER": "value"})
    statuses = [doctor.OK, doctor.WARN, doctor.SKIP]

    def check(env: doctor.Env) -> doctor.Result:
        assert env == {"MARKER": "value"}
        return statuses[0], "local", "local detail"

    def reach(label: str, host: str) -> doctor.Result:
        assert (label, host) == ("fake", "fake.invalid")
        return statuses[-1], "network fake", "remote detail"

    monkeypatch.setattr(doctor, "CHECKS", (check,))

    def hosts(env: doctor.Env) -> list[tuple[str, str]]:
        return [("fake", "fake.invalid")]

    monkeypatch.setattr(doctor, "api_hosts", hosts)
    monkeypatch.setattr(doctor, "check_reach", reach)
    assert doctor.main() == 0
    assert capsys.readouterr().out == (
        f"{doctor.OK} local: local detail\n"
        f"{doctor.SKIP} network fake: remote detail\n"
        "doctor: ready, run bun run stockbot\n"
    )
    statuses[:] = [doctor.FAIL, doctor.FAIL]
    assert doctor.main() == 1
    assert capsys.readouterr().out == (
        f"{doctor.FAIL} local: local detail\n{doctor.FAIL} network fake: remote detail\ndoctor: 2 failed\n"
    )
