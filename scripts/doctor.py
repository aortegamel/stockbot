#!/usr/bin/env python3
"""Stockbot doctor: fresh-machine preflight for `bun run stockbot`.

Stdlib only. Runs every check and prints one line per check:
`✓` ok, `!` warning, `-` skipped (optional feature off), `✗` failure plus its fix.
Exits 1 when any check fails. Prints key names and hosts, never secret values.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "needle-harness"
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
OAUTH_PATH = Path.home() / ".stockbot" / "robinhood" / "oauth.json"
OK, WARN, SKIP, FAIL = "\u2713", "!", "-", "\u2717"
TRUTHY = ("1", "true", "yes")
DECISION_KEYS = ("DECISION_API_FORMAT", "DECISION_API_URL", "DECISION_API_KEY", "DECISION_MODEL")
OPENCODE_KEYS = ("OPENCODE_API_KEY", "OPENCODE_URL", "OPENCODE_MODEL")
SEC_HOST = "www.sec.gov"
YAHOO_HOST = "query2.finance.yahoo.com"
EXA_HOST = "api.exa.ai"
FINRA_HOST = "api.finra.org"
ROBINHOOD_HOST = "agent.robinhood.com"

Result = tuple[str, str, str]
Env = Mapping[str, str]


def parse_dotenv(text: str) -> dict[str, str]:
    """KEY=VALUE lines. Quoted values keep '#'. Unquoted values drop a trailing ' #' comment."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if value[:1] in ('"', "'"):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = value.split(" #", 1)[0].strip()
        out[key.strip()] = value
    return out


def load_env(environ: Env, dotenv_path: Path) -> dict[str, str]:
    """Mirror app/config.py: root .env replaces inherited values, except an inherited PORT."""
    env = dict(environ)
    if dotenv_path.exists():
        env.update(parse_dotenv(dotenv_path.read_text(encoding="utf-8")))
    if "PORT" in environ:
        env["PORT"] = environ["PORT"]
    return env


def val(env: Env, key: str) -> str:
    """Stripped value. An unchanged .env.example placeholder ("your...") counts as unset."""
    value = (env.get(key) or "").strip()
    return "" if value.lower().startswith("your") else value


def on(env: Env, key: str) -> bool:
    return val(env, key).lower() in TRUTHY


def check_python(env: Env) -> Result:
    name = "python venv"
    if not VENV_PYTHON.exists():
        return FAIL, name, "venv/bin/python missing. Fix: bun run setup"
    want = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    probe = "import sys, fastapi, uvicorn; print('%d.%d' % sys.version_info[:2])"
    proc = subprocess.run([str(VENV_PYTHON), "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        return FAIL, name, "venv lacks fastapi/uvicorn. Fix: bun run setup"
    got = proc.stdout.strip()
    if got != want:
        return (
            FAIL,
            name,
            f"venv is Python {got}, .python-version wants {want}. Fix: bun run clean-env && bun run setup",
        )
    return OK, name, f"Python {got} with requirements"


def check_bun(env: Env) -> Result:
    name = "bun"
    bun = shutil.which("bun")
    if bun is None:
        return FAIL, name, "not on PATH. Fix: curl -fsSL https://bun.sh/install | bash"
    missing = [d for d in ("web", "needle-harness") if not (REPO_ROOT / d / "node_modules").is_dir()]
    if missing:
        fix = " && ".join(f"bun install --cwd {d}" for d in missing)
        return FAIL, name, f"{', '.join(missing)} dependencies missing. Fix: {fix}"
    version = subprocess.run([bun, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    return OK, name, f"bun {version}, web + needle-harness dependencies installed"


def check_dotenv(env: Env) -> Result:
    if (REPO_ROOT / ".env").exists():
        return OK, ".env", "present"
    return FAIL, ".env", "missing. Fix: cp .env.example .env, then fill the MVP-required keys"


def check_sec(env: Env) -> Result:
    name = "SEC identity (MVP)"
    if "@" not in val(env, "SEC_EDGAR_IDENTITY"):
        return FAIL, name, 'SEC_EDGAR_IDENTITY unset or lacks an email. Fix: SEC_EDGAR_IDENTITY="Name you@host" in .env'
    return OK, name, "SEC_EDGAR_IDENTITY set"


def check_decision(env: Env) -> Result:
    name = "JEV decision API (MVP)"
    missing = [k for k in DECISION_KEYS if not val(env, k)]
    if missing:
        return FAIL, name, f"unset: {', '.join(missing)}. Fix: set them in .env (see docs/runtime.md)"
    fmt = val(env, "DECISION_API_FORMAT")
    if fmt not in ("chat_completions", "decisions"):
        return FAIL, name, 'DECISION_API_FORMAT is invalid. Fix: use "chat_completions" or "decisions"'
    if urlsplit(val(env, "DECISION_API_URL")).scheme != "https":
        return FAIL, name, "DECISION_API_URL is not HTTPS. Fix: use the provider's https:// endpoint"
    return OK, name, f"{fmt} model {val(env, 'DECISION_MODEL')}"


def check_opencode(env: Env) -> Result:
    name = "OpenCode reasoner (MVP)"
    missing = [k for k in OPENCODE_KEYS if not val(env, k)]
    if missing:
        return FAIL, name, f"unset: {', '.join(missing)}. Fix: set them in .env"
    if not val(env, "OPENCODE_URL").endswith("/responses"):
        return (
            FAIL,
            name,
            "OPENCODE_URL must end in /responses. Fix: OPENCODE_URL=https://opencode.ai/zen/go/v1/responses",
        )
    return OK, name, f"model {val(env, 'OPENCODE_MODEL')}"


def check_exa(env: Env) -> Result:
    name = "Exa web research (optional)"
    if not on(env, "EXA_ENABLED"):
        return SKIP, name, "EXA_ENABLED off"
    if not val(env, "EXA_API_KEY"):
        return FAIL, name, "EXA_ENABLED but EXA_API_KEY unset. Fix: set EXA_API_KEY or EXA_ENABLED=false"
    return OK, name, "EXA_API_KEY set"


def check_finra(env: Env) -> Result:
    name = "FINRA Query API (optional)"
    have = [k for k in ("FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET") if val(env, k)]
    if not have:
        return SKIP, name, "FINRA_CLIENT_ID/FINRA_CLIENT_SECRET unset"
    if len(have) == 1:
        return FAIL, name, f"only {have[0]} set. Fix: set both FINRA_CLIENT_ID and FINRA_CLIENT_SECRET"
    return OK, name, "client credentials set"


def broker_on(env: Env) -> bool:
    """Mirror app/config.broker_enabled: BROKER_ENABLED wins, else ROBINHOOD_ENABLED."""
    return on(env, "BROKER_ENABLED" if "BROKER_ENABLED" in env else "ROBINHOOD_ENABLED")


def check_broker(env: Env) -> Result:
    name = "Robinhood broker (optional)"
    if not broker_on(env):
        return SKIP, name, "BROKER_ENABLED off"
    if not OAUTH_PATH.exists():
        return FAIL, name, f"no OAuth state at {OAUTH_PATH}. Fix: venv/bin/python cli.py robinhood-login"
    return OK, name, "OAuth state present"


def check_needle(env: Env) -> Result:
    """Mirror needle-harness/lib/needle/server.py resolve_weights()."""
    name = "Needle model"
    pinned = val(env, "NEEDLE_WEIGHTS")
    path = HARNESS_DIR / pinned if pinned else REPO_ROOT / "needle3.cact"
    path = path.resolve()
    if path.is_file():
        return OK, name, f"weights {path}"
    if pinned:
        return FAIL, name, f"NEEDLE_WEIGHTS={pinned} resolves to missing {path}. Fix: git checkout -- needle3.cact"
    return WARN, name, "needle3.cact missing, launcher falls back to the HF cache. Fix: git checkout -- needle3.cact"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def check_ports(env: Env) -> Result:
    """Preferred ports as needle.ts derives them. A busy port only warns: the launcher picks the next free one."""
    name = "ports"
    try:
        web = int(val(env, "PORT") or "3000")
        api = int(val(env, "STOCKBOT_API_PORT") or str(web + 1))
        market = int(val(env, "STOCKBOT_MARKET_PORT") or str(api + 1))
    except ValueError:
        return FAIL, name, "PORT/STOCKBOT_API_PORT/STOCKBOT_MARKET_PORT is not an integer. Fix: correct it in .env"
    busy = [str(p) for p in (web, api, market) if not port_free(p)]
    if busy:
        return (
            WARN,
            name,
            f"busy: {', '.join(busy)}, launcher picks the next free port. Fix: stop the old bun run stockbot",
        )
    return OK, name, f"web {web}, api {api}, market {market} free"


def api_hosts(env: Env) -> list[tuple[str, str]]:
    """(label, host) per external API the configured features call."""
    hosts = [("SEC EDGAR", SEC_HOST), ("Yahoo market data", YAHOO_HOST)]
    for label, key in (("OpenCode", "OPENCODE_URL"), ("JEV decision API", "DECISION_API_URL")):
        host = urlsplit(val(env, key)).hostname
        if host:
            hosts.append((label, host))
    if on(env, "EXA_ENABLED"):
        hosts.append(("Exa", EXA_HOST))
    if val(env, "FINRA_CLIENT_ID"):
        hosts.append(("FINRA", FINRA_HOST))
    if broker_on(env):
        hosts.append(("Robinhood", ROBINHOOD_HOST))
    return hosts


def check_reach(label: str, host: str) -> Result:
    """TCP connect to host:443 only: no request, no credentials, no quota."""
    name = f"network {label}"
    try:
        socket.create_connection((host, 443), timeout=3).close()
    except OSError as exc:
        return FAIL, name, f"{host}:443 unreachable ({exc}). Fix: check DNS, firewall, or proxy"
    return OK, name, f"{host}:443 reachable"


CHECKS: tuple[Callable[[Env], Result], ...] = (
    check_python,
    check_bun,
    check_dotenv,
    check_sec,
    check_decision,
    check_opencode,
    check_exa,
    check_finra,
    check_broker,
    check_needle,
    check_ports,
)


def main() -> int:
    env = load_env(os.environ, REPO_ROOT / ".env")
    results = [check(env) for check in CHECKS] + [check_reach(label, host) for label, host in api_hosts(env)]
    for status, name, detail in results:
        print(f"{status} {name}: {detail}")
    failed = sum(status == FAIL for status, _, _ in results)
    print(f"doctor: {failed} failed" if failed else "doctor: ready, run bun run stockbot")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
