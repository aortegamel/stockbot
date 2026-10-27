"""Configuration: loads the repository root .env, fails fast on missing keys.

edgartools identity configuration lives behind the edgar boundary in
``app/edgar_client.py``; this module only validates the env values it needs.
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dotenv import dotenv_values

from .log_stream import LogStreamHandler

if TYPE_CHECKING:
    from .research.models import JSONValue

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_data_root() -> Path:
    """Single durable-data root: STOCKBOT_DATA_DIR else <repo>/data."""
    raw = (os.getenv("STOCKBOT_DATA_DIR") or "").strip()
    if not raw:
        return REPO_ROOT / "data"
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


FINRA_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
FINRA_API_BASE = "https://api.finra.org"
EXA_API_BASE = "https://api.exa.ai"

# Analyst consensus (unofficial Yahoo endpoint) + index-weight data sources.
YAHOO_QUERY_BASE = "https://query2.finance.yahoo.com"
YAHOO_CRUMB_URL = "https://fc.yahoo.com"
SLICKCHARTS_SP500_URL = "https://www.slickcharts.com/sp500"


def _load_root_dotenv() -> None:
    """Root .env values replace inherited ones. Inherited PORT wins, as in needle-harness/lib/env.ts."""
    for key, value in dotenv_values(REPO_ROOT / ".env").items():
        if value is not None and not (key == "PORT" and key in os.environ):
            os.environ[key] = value


_load_root_dotenv()


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    placeholders = ("Your Name", "YourName", "sk-or-...", "your_finra_")
    if not value or value.startswith(placeholders):
        raise ValueError(
            f"{name} is not properly set in your environment or .env file. "
            "See .env.example and configure this value before using the "
            "integration that requires it."
        )
    return value


def _env_bool(name: str) -> bool:
    """True when the env var is set to a truthy value (1/true/yes)."""
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(*, stream_url: str | None = None) -> None:
    """Configure root logging: WARNING on stderr by default; when stream_url
    is given, attach a LogStreamHandler and log everything (DEBUG)."""
    level = logging.DEBUG if stream_url else logging.WARNING
    logging.basicConfig(force=True, level=level, format=LOG_FORMAT)
    if stream_url:
        handler = LogStreamHandler(stream_url)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)


def init_config() -> None:
    """Validate env vars required by the SEC research path.

    Raises ValueError if SEC_EDGAR_IDENTITY is missing.
    """
    _require_env("SEC_EDGAR_IDENTITY")


def get_sec_edgar_identity() -> str:
    """The SEC EDGAR identity string (validated, placeholder-rejected)."""
    return _require_env("SEC_EDGAR_IDENTITY")


def get_finra_client_id() -> str:
    return _require_env("FINRA_CLIENT_ID")


def get_finra_client_secret() -> str:
    return _require_env("FINRA_CLIENT_SECRET")


def finra_use_mock() -> bool:
    return _env_bool("FINRA_USE_MOCK")


def get_robinhood_mcp_url() -> str:
    # Validate here as well as in OAuthConfig so environment and --server-url
    # paths cannot accidentally direct Robinhood credentials to another host.
    from .robinhood.auth import validate_robinhood_server_url

    return validate_robinhood_server_url(os.getenv("ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading"))


def broker_enabled() -> bool:
    if os.getenv("BROKER_ENABLED") is not None:
        return _env_bool("BROKER_ENABLED")
    return _env_bool("ROBINHOOD_ENABLED")


def exa_enabled() -> bool:
    """True when EXA_ENABLED is set to a truthy value."""
    return _env_bool("EXA_ENABLED")


def get_exa_api_key() -> str | None:
    """Exa API key, or None when unset (integration is optional)."""
    value = (os.getenv("EXA_API_KEY") or "").strip()
    return value or None


def _env_optional(name: str) -> str | None:
    """Optional env value: stripped string, or None when unset/blank."""
    value = (os.getenv(name) or "").strip()
    return value or None


_REASONING_EFFORT_SHAPE = re.compile(r"[a-z][a-z0-9_-]{0,31}")


def reasoning_fields(
    protocol: Literal["responses", "chat_completions", "decisions"], error_prefix: str
) -> dict[str, JSONValue]:
    """REASONING_EFFORT in the protocol's own request field; TS twin: decision/reasoning.ts.

    A concrete value passes through unchanged, so each provider's own levels work.
    Unset, empty, or auto sends no field. A bad shape raises before any request.
    """
    effort = _env_optional("REASONING_EFFORT")
    if effort is None or effort == "auto":
        return {}
    if not _REASONING_EFFORT_SHAPE.fullmatch(effort):
        raise RuntimeError(f"{error_prefix}: REASONING_EFFORT must match ^[a-z][a-z0-9_-]{{0,31}}$")
    if protocol == "responses":
        return {"reasoning": {"effort": effort}}
    if protocol == "chat_completions":
        return {"reasoning_effort": effort}
    # The Decisions API request schema (DecisionsRequest) documents no reasoning field: send none.
    return {}


def _env_int(name: str, default: int) -> int:
    """Optional int env value; default on missing/unparseable, never raises."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def google_data_enabled() -> bool:
    """True when GOOGLE_DATA_ENABLED is set to a truthy value (default off)."""
    return _env_bool("GOOGLE_DATA_ENABLED")


def get_google_cloud_project() -> str | None:
    """GCP project for BigQuery, or None when unset (source stays disabled)."""
    return _env_optional("GOOGLE_CLOUD_PROJECT")


def get_datacommons_api_key() -> str | None:
    """Data Commons API key, or None when unset (required by Data Commons)."""
    return _env_optional("DATACOMMONS_API_KEY")


def get_google_cloud_api_key() -> str | None:
    """Google Cloud API key (YouTube Data API v3), or None when unset."""
    return _env_optional("GOOGLE_CLOUD_API_KEY")


def get_bq_max_bytes_per_query() -> int:
    """Per-query billing cap (bytes); default 1 GiB."""
    return _env_int("BIGQUERY_MAX_BYTES_PER_QUERY", 1073741824)


def get_bq_monthly_bytes_limit() -> int:
    """Monthly reservation ceiling (bytes); default 500 GiB."""
    return _env_int("BIGQUERY_MONTHLY_BYTES_LIMIT", 536870912000)


def get_bq_daily_bytes_limit() -> int:
    """Daily reservation ceiling (bytes); default 10 GiB."""
    return _env_int("BIGQUERY_DAILY_BYTES_LIMIT", 10737418240)


def google_trends_api_enabled() -> bool:
    """True when GOOGLE_TRENDS_API_ENABLED is set (default off, pending alpha)."""
    return _env_bool("GOOGLE_TRENDS_API_ENABLED")


def bigquery_enabled() -> bool:
    """True when GOOGLE_DATA_ENABLED and a BigQuery project is configured."""
    return google_data_enabled() and bool(get_google_cloud_project())


def datacommons_enabled() -> bool:
    """True when GOOGLE_DATA_ENABLED (key required separately at call time)."""
    return google_data_enabled()


def youtube_enabled() -> bool:
    """True when GOOGLE_DATA_ENABLED and the YouTube source is configured."""
    return google_source_enabled("youtube")


def get_youtube_search_daily_limit() -> int:
    """Daily YouTube search reservation cap; default 80.

    A present-but-unusable value raises ValueError (fail closed at use).
    """
    raw = (os.getenv("YOUTUBE_SEARCH_DAILY_LIMIT") or "").strip()
    if not raw:
        return 80
    try:
        value = int(raw)
    except TypeError, ValueError:
        raise ValueError(f"invalid YOUTUBE_SEARCH_DAILY_LIMIT: {raw!r}") from None
    if value <= 0:
        raise ValueError(f"non-positive YOUTUBE_SEARCH_DAILY_LIMIT: {raw!r}")
    return value


def _youtube_gate() -> bool:
    try:
        limit = get_youtube_search_daily_limit()
    except ValueError:
        return False
    return bool(get_google_cloud_api_key()) and limit > 0


def _bigquery_gate() -> bool:
    return bool(get_google_cloud_project()) and get_bq_max_bytes_per_query() > 0 and get_bq_monthly_bytes_limit() > 0


def google_source_enabled(name: str) -> bool:
    """True when GOOGLE_DATA_ENABLED and the named source is configured.

    youtube needs GOOGLE_CLOUD_API_KEY (+ positive search limit);
    datacommons/macro needs DATACOMMONS_API_KEY; all other
    (BigQuery-backed) names need GOOGLE_CLOUD_PROJECT and positive
    byte limits. Unknown names fail closed as BigQuery-backed.
    """
    if not google_data_enabled():
        return False
    n = (name or "").strip().lower()
    if n == "youtube":
        return _youtube_gate()
    if n in ("datacommons", "data_commons", "data-commons", "macro", "macro_context"):
        return bool(get_datacommons_api_key())
    return _bigquery_gate()
