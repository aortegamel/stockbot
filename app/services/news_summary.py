"""Collect article text and write validated briefs.

Each pass takes one bounded snapshot of pending rows. Failed rows keep NULL status.
The refresh log records each error. The next collector pass retries those rows.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from app import exa_client
from app.config import get_data_root
from app.reasoner_client import ReasonerClient
from app.storage import market
from app.thesis.yaml import atomic_write_text

logger = logging.getLogger(__name__)
_missing_config_logged = False


def _published(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=parsed.tzinfo or UTC).timestamp()
    except ValueError:
        return None


def collect_exa_news(ticker: str, *, now: float) -> int:
    """Search the past two days and store headlines plus returned article text."""
    result = exa_client.search(
        f"{ticker} stock company news",
        category="news",
        start_published_date=(datetime.fromtimestamp(now, UTC) - timedelta(days=2)).date().isoformat(),
        caller="collector",
    )
    if "error" in result:
        market.mark_error("exa_news", ticker, now=now, error=str(result["error"]))
        return 0
    count = 0
    evidence = result.get("evidence")
    for row in evidence if isinstance(evidence, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            count += _store_row(ticker, row, now=now)
        except (OSError, ValueError, sqlite3.Error) as exc:
            market.mark_error("exa_news", ticker, now=now, error=str(exc))
    market.mark_ok("exa_news", ticker, now=now)
    return count


def _store_row(ticker: str, row: dict[str, object], *, now: float) -> int:
    """Store one search result: the headline row plus any returned article text."""
    url = str(row.get("url") or "")
    title = str(row.get("title") or "")
    count = market.save_news(
        ticker,
        [
            {
                "title": title,
                "url": url,
                "source": str(row.get("source_domain") or "Exa"),
                "published_at": _published(row.get("published_at")),
            }
        ],
        now=now,
    )
    body = row.get("source_content")
    if isinstance(body, str) and body.strip():
        market.set_news_body(ticker, market.url_hash(url or title), body)
    return count


def fetch_bodies(limit: int, *, now: float) -> int:
    """Fetch missing HTTP(S) article bodies in batches of 25. Each URL is requested at most once."""
    rows = [
        row
        for row in market.pending_summaries(limit)
        if row["body"] is None and exa_client.is_http_url(str(row["url"]))
    ]
    count = 0
    for offset in range(0, len(rows), 25):
        batch = rows[offset : offset + 25]
        bodies = _article_text(list(dict.fromkeys(str(row["url"]) for row in batch)), now=now)
        if bodies is None:
            break
        for row in batch:
            count += _store_body(row, bodies.get(str(row["url"])), now=now)
    return count


def _store_body(row: dict[str, object], body: object, *, now: float) -> bool:
    """Store one fetched body; '' marks "requested, no text" so the story is never re-bought."""
    text = body if isinstance(body, str) and body.strip() else ""
    try:
        market.set_news_body(str(row["symbol"]), str(row["url_hash"]), text)
    except (OSError, ValueError, sqlite3.Error) as exc:
        market.mark_error("news_bodies", str(row["symbol"]), str(row["url_hash"]), now=now, error=str(exc))
    return bool(text)


def _article_text(urls: list[str], *, now: float) -> dict[str, object] | None:
    """Request one batch of article text. None ends the pass without a retry."""
    result = exa_client.contents(urls, caller="collector")
    if "error" in result:
        # A refused budget ends this pass without repeated requests or log spam.
        if "budget" not in str(result["error"]).lower():
            market.mark_error("news_bodies", "*", now=now, error=str(result["error"]))
        return None
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        return {}
    return {str(item.get("url")): item.get("source_content") for item in evidence if isinstance(item, dict)}


def _text(value: object) -> str:
    """Render model fields as plain inline text, not executable Markdown or HTML.

    Every rendered line starts with fixed markup, so only inline syntax needs escaping:
    emphasis, code, links/images (brackets), tables, strikethrough. html.escape covers tags.
    """
    text = html.escape(" ".join(str(value).split()), quote=False)
    return re.sub(r"([\\`*_\[\]|~])", r"\\\1", text)


def _markdown(row: dict[str, object], summary: dict[str, object], *, now: float, model: str) -> str:
    published = row["published_at"]
    timestamp = datetime.fromtimestamp(float(published), UTC) if isinstance(published, (int, float)) else None
    metadata = {
        "ticker": row["symbol"],
        "published_at": timestamp.isoformat() if timestamp else None,
        "source": row["source"],
        "url": row["url"],
        "sentiment": summary["sentiment"],
        "impact": summary["impact"],
        "summarized_at": datetime.fromtimestamp(now, UTC).isoformat(),
        "model": model,
    }
    # JSON scalar values also form valid YAML scalars and prevent front-matter injection.
    lines = [
        "---",
        *(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()),
        "---",
        "",
        f"# {_text(summary['headline'])}",
        "",
    ]
    url = str(row["url"])
    if exa_client.is_http_url(url):
        link = quote(url, safe=":/?#@!$&'*+,;=%~-")
        time = timestamp.strftime("%H:%M UTC") if timestamp else "Time unavailable"
        lines.extend([f"**[Read article]({link})** · {_text(row['source'])} · {time}", ""])
    lines.extend([f"**Bottom line:** {_text(summary['bottom_line'])}", ""])
    points = summary["key_points"]
    if isinstance(points, list):
        lines.extend(f"- {_text(point)}" for point in points)
    if summary["watch"]:
        lines.extend(["", f"**Watch:** {_text(summary['watch'])}"])
    if summary["limits"]:
        lines.extend(["", f"*Limits: {_text(summary['limits'])}*"])
    return "\n".join(lines) + "\n"


def summarize_pending(limit: int, *, now: float) -> int:
    """Write relevant high/medium briefs. A story gets two attempts; then it is parked as 'error'."""
    global _missing_config_logged
    model = os.getenv("OPENCODE_MODEL", "").strip()
    url = os.getenv("OPENCODE_URL", "").strip()
    key = os.getenv("OPENCODE_API_KEY", "").strip()
    if not all((model, url, key)):
        if not _missing_config_logged:
            logger.info("News summaries require OPENCODE_MODEL, OPENCODE_URL, and OPENCODE_API_KEY")
            _missing_config_logged = True
        return 0
    _missing_config_logged = False
    client = ReasonerClient(model=model, url=url, api_key=key)
    root = get_data_root() / "news"
    count = 0
    for row in market.pending_summaries(limit):
        ticker, hash_ = str(row["symbol"]), str(row["url_hash"])
        try:
            count += _write_brief(row, client, root, model=model, now=now)
        except (RuntimeError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
            # Two attempts per story: a second failure parks it as 'error' so it never bills again.
            if (market.refresh_state("news_summary", ticker, hash_) or {}).get("last_error"):
                market.set_summary(ticker, hash_, "error")
            market.mark_error("news_summary", ticker, hash_, now=now, error=str(exc))
    return count


def _write_brief(row: dict[str, object], client: ReasonerClient, root: Path, *, model: str, now: float) -> bool:
    """Write one brief. False when the story is irrelevant or low impact."""
    ticker, hash_ = str(row["symbol"]), str(row["url_hash"])
    summary = client.summarize_story(ticker, str(row["title"]), str(row["body"] or ""))
    if not summary["relevant"] or summary["impact"] == "low":
        market.set_summary(ticker, hash_, "skipped")
        return False
    published = row["published_at"]
    day = datetime.fromtimestamp(float(published) if isinstance(published, (int, float)) else now, UTC)
    slug = re.sub(r"[^a-z0-9]+", "-", str(summary["headline"]).lower()).strip("-")[:70] or "story"
    safe_ticker = re.sub(r"[^A-Z0-9.-]", "-", ticker.upper())
    path = root / day.strftime("%Y-%m-%d") / f"{safe_ticker}-{slug}-{hash_[:8]}.md"
    atomic_write_text(path, _markdown(row, summary, now=now, model=model), root)
    market.set_summary(ticker, hash_, "written", str(path))
    market.mark_ok("news_summary", ticker, hash_, now=now)
    return True
