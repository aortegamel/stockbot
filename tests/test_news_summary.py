"""Offline article collection, Markdown, and retry checks."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from app.reasoner_client import STORY_SYSTEM_PROMPT, ReasonerClient, check_story
from app.services import news_summary as news
from app.storage import market

NOW = 1793352600.0


def summary(**changes: object) -> dict[str, object]:
    return {
        "relevant": True,
        "headline": "Company raises guidance",
        "bottom_line": "The company raised its revenue guidance.",
        "key_points": ["Revenue increased 10%.", "Guidance increased to $10 billion."],
        "watch": "Earnings on November 1",
        "sentiment": "bullish",
        "impact": "high",
        "limits": None,
        **changes,
    }


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENCODE_MODEL", "test-model")
    monkeypatch.setenv("OPENCODE_URL", "https://reasoner.invalid/responses")
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")


def add_story(url: str = "https://example.com/story", title: str = "Title") -> None:
    market.save_news("AMD", [{"title": title, "url": url, "source": "Example", "published_at": NOW}], now=NOW)


def test_collection_saves_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(query: str, **kwargs: object) -> dict[str, object]:
        assert "AMD" in query
        assert kwargs == {"category": "news", "start_published_date": "2026-10-28", "caller": "collector"}
        return {
            "evidence": [
                {
                    "title": "Title",
                    "url": "https://example.com/story",
                    "source_domain": "example.com",
                    "published_at": "2026-10-30T10:50:00Z",
                    "source_content": "Full article",
                }
            ]
        }

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 1
    assert news.collect_exa_news("AMD", now=NOW) == 0
    row = market.pending_summaries(5)[0]
    assert row["body"] == "Full article"
    assert (row["title"], row["url"], row["source"]) == ("Title", "https://example.com/story", "example.com")
    assert row["published_at"] == datetime(2026, 10, 30, 10, 50, tzinfo=UTC).timestamp()


def test_collection_defaults_missing_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [{"source_content": "Text"}]}

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 1
    row = market.pending_summaries(5)[0]
    assert (row["title"], row["url"], row["source"]) == ("", "", "Exa")
    assert row["body"] == "Text"


@pytest.mark.parametrize("published", ["not a date", 20261030, None])
def test_collection_ignores_unparseable_published_date(monkeypatch: pytest.MonkeyPatch, published: object) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [{"title": "Title", "url": "https://example.com/story", "published_at": published}]}

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 1
    assert market.pending_summaries(5)[0]["published_at"] is None


@pytest.mark.parametrize("published", ["2026-10-30T10:50:00Z", "2026-10-30T12:50:00+02:00"])
def test_collection_parses_published_date(monkeypatch: pytest.MonkeyPatch, published: str) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [{"title": "Title", "url": "https://example.com/story", "published_at": published}]}

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 1
    assert market.pending_summaries(5)[0]["published_at"] == datetime(2026, 10, 30, 10, 50, tzinfo=UTC).timestamp()


def test_fetch_bodies_filters_urls_and_stops_on_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()
    add_story("javascript:alert(1)", "Bad link")
    calls: list[list[str]] = []

    def contents(urls: list[str], *, caller: str) -> dict[str, object]:
        assert caller == "collector"
        calls.append(urls)
        return {"evidence": [{"url": urls[0], "source_content": "Article body"}]}

    monkeypatch.setattr(news.exa_client, "contents", contents)
    assert news.fetch_bodies(10, now=NOW) == 1
    assert calls == [["https://example.com/story"]]
    assert news.fetch_bodies(10, now=NOW) == 0
    for index in range(26):
        add_story(f"https://example.com/{index}", str(index))
    calls.clear()

    def refused(urls: list[str], *, caller: str) -> dict[str, object]:
        calls.append(urls)
        return {"error": "Exa monthly budget exhausted"}

    monkeypatch.setattr(news.exa_client, "contents", refused)
    assert news.fetch_bodies(50, now=NOW) == 0
    assert len(calls) == 1
    # A refused budget is not logged as an error.
    assert market.refresh_state("news_bodies", "*") is None


@pytest.mark.parametrize(
    "url,has_link", [("https://example.com/story", True), ("javascript:alert(1)", False), ("", False)]
)
def test_markdown_writes_metadata_and_safe_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str, has_link: bool
) -> None:
    add_story(url)

    def summarize(*args: object) -> dict[str, object]:
        return summary(headline="Facts <script> [bad](javascript:x)")

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW) == 1
    path = next((tmp_path / "news").rglob("*.md"))
    content = path.read_text()
    metadata = yaml.safe_load(content.split("---")[1])
    assert metadata["ticker"] == "AMD"
    assert metadata["model"] == "test-model"
    assert metadata["published_at"]
    assert ("**[Read article](" in content) == has_link
    assert "**Bottom line:**" in content and "**Watch:**" in content
    assert "<script>" not in content and "[bad](javascript:x)" not in content
    assert not market.pending_summaries(5)
    assert market.read_news("AMD", 1)[0]["summary_path"] == str(path)


@pytest.mark.parametrize("changes", [{"impact": "low"}, {"relevant": False}])
def test_low_or_irrelevant_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changes: dict[str, object]
) -> None:
    add_story()
    statuses: list[tuple[str, str | None]] = []
    set_summary = market.set_summary

    def record_status(symbol: str, hash_: str, status: str, path: str | None = None) -> None:
        statuses.append((status, path))
        set_summary(symbol, hash_, status, path)

    def summarize(*args: object) -> dict[str, object]:
        return summary(**changes)

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    monkeypatch.setattr(market, "set_summary", record_status)
    assert news.summarize_pending(5, now=NOW) == 0
    assert statuses == [("skipped", None)]
    assert not list(tmp_path.rglob("*.md"))
    assert not market.pending_summaries(5)


def test_failed_summary_retries_next_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()
    calls = []

    def fail(*args: object) -> dict[str, object]:
        calls.append(args)
        raise ValueError("bad model response")

    monkeypatch.setattr(ReasonerClient, "summarize_story", fail)
    assert news.summarize_pending(5, now=NOW) == 0
    assert len(calls) == 1
    assert len(market.pending_summaries(5)) == 1
    state = market.refresh_state("news_summary", "AMD", market.url_hash("https://example.com/story"))
    assert state and state["last_error"] == "bad model response"
    assert state["last_error_at"] == NOW

    def summarize(*args: object) -> dict[str, object]:
        return summary()

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW + 300) == 1


@pytest.mark.parametrize("missing", ["OPENCODE_MODEL", "OPENCODE_URL", "OPENCODE_API_KEY"])
def test_missing_reasoner_config_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, missing: str
) -> None:
    add_story()
    monkeypatch.delenv(missing)
    monkeypatch.setattr(news, "_missing_config_logged", False)
    with caplog.at_level("INFO"):
        assert news.summarize_pending(5, now=NOW) == 0
        assert news.summarize_pending(5, now=NOW) == 0
    assert [m for m in caplog.messages if "News summaries require" in m] == [
        "News summaries require OPENCODE_MODEL, OPENCODE_URL, and OPENCODE_API_KEY"
    ]
    assert len(market.pending_summaries(5)) == 1


def test_successful_pass_resets_missing_config_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert news.summarize_pending(0, now=NOW) == 0
    monkeypatch.delenv("OPENCODE_MODEL")
    with caplog.at_level("INFO"):
        assert news.summarize_pending(0, now=NOW) == 0
    assert [m for m in caplog.messages if "News summaries require" in m] == [
        "News summaries require OPENCODE_MODEL, OPENCODE_URL, and OPENCODE_API_KEY"
    ]


def test_summary_prompt_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def post(prompt: str, model: str) -> dict[str, object]:
        prompts.append(prompt)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(summary(limits="Headline only"))}],
                }
            ]
        }

    client = ReasonerClient(model="m", url="u", post=post)
    assert client.summarize_story("AMD", "Headline")["limits"] == "Headline only"
    assert prompts[0].startswith(STORY_SYSTEM_PROMPT)
    assert '"body": ""' in prompts[0]


@pytest.mark.parametrize("candidate", [None, [], "brief", 3])
def test_story_requires_object(candidate: object) -> None:
    with pytest.raises(ValueError, match="summary must be an object"):
        check_story(candidate)


@pytest.mark.parametrize("field", ["watch", "limits"])
def test_story_rejects_whitespace_optional_fields(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must contain text or null"):
        check_story(summary(**{field: " \t\n"}))


def test_summary_second_failure_parks_story(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()
    calls = []
    statuses: list[tuple[str, str, str, str | None]] = []
    set_summary = market.set_summary

    def record_status(symbol: str, hash_: str, status: str, path: str | None = None) -> None:
        statuses.append((symbol, hash_, status, path))
        set_summary(symbol, hash_, status, path)

    def fail(*args: object) -> dict[str, object]:
        calls.append(args)
        raise RuntimeError("reasoner unavailable")

    monkeypatch.setattr(ReasonerClient, "summarize_story", fail)
    monkeypatch.setattr(market, "set_summary", record_status)
    assert news.summarize_pending(5, now=NOW) == 0
    assert len(market.pending_summaries(5)) == 1
    assert news.summarize_pending(5, now=NOW + 300) == 0
    assert statuses == [("AMD", market.url_hash("https://example.com/story"), "error", None)]
    assert not market.pending_summaries(5)
    assert news.summarize_pending(5, now=NOW + 600) == 0
    assert len(calls) == 2


def test_markdown_renders_limits_without_publication_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    market.save_news("AMD", [{"title": "Title", "url": "https://example.com/a (b)"}], now=NOW)

    def summarize(*args: object) -> dict[str, object]:
        return summary(limits="Headline <only>", watch=None)

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW) == 1
    content = next((tmp_path / "news").rglob("*.md")).read_text()
    assert "published_at: null" in content
    assert "Time unavailable" in content
    assert "https://example.com/a%20%28b%29" in content
    assert "*Limits: Headline &lt;only&gt;*" in content
    assert "**Watch:**" not in content


@pytest.mark.parametrize("evidence", [None, [None, {"url": "https://example.com/story", "source_content": "  "}], []])
def test_body_without_text_is_not_requested_again(monkeypatch: pytest.MonkeyPatch, evidence: object) -> None:
    add_story()
    calls = []

    def contents(urls: list[str], *, caller: str) -> dict[str, object]:
        calls.append(urls)
        return {"evidence": evidence}

    monkeypatch.setattr(news.exa_client, "contents", contents)
    assert news.fetch_bodies(5, now=NOW) == 0
    assert market.pending_summaries(5)[0]["body"] == ""
    assert news.fetch_bodies(5, now=NOW) == 0
    assert calls == [["https://example.com/story"]]


def test_body_fetch_reports_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()

    def contents(*args: object, **kwargs: object) -> dict[str, object]:
        return {"error": "offline"}

    monkeypatch.setattr(news.exa_client, "contents", contents)
    assert news.fetch_bodies(5, now=NOW) == 0
    assert market.pending_summaries(5)[0]["body"] is None
    state = market.refresh_state("news_bodies", "*")
    assert state and state["last_error"] == "offline"
    assert state["last_error_at"] == NOW


def test_body_write_failure_keeps_story_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()

    def contents(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [{"url": "https://example.com/story", "source_content": "Article"}]}

    monkeypatch.setattr(news.exa_client, "contents", contents)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(market, "set_news_body", fail)
    assert news.fetch_bodies(5, now=NOW) == 1
    assert market.pending_summaries(5)[0]["body"] is None
    state = market.refresh_state("news_bodies", "AMD", market.url_hash("https://example.com/story"))
    assert state and state["last_error"] == "disk full"
    assert state["last_error_at"] == NOW


def test_collection_logs_search_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"error": "offline"}

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 0
    state = market.refresh_state("exa_news", "AMD")
    assert state and state["last_error"] == "offline"
    assert not market.pending_summaries(5)


def test_collection_ignores_malformed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [None]}

    monkeypatch.setattr(news.exa_client, "search", search)
    assert news.collect_exa_news("AMD", now=NOW) == 0
    assert not market.pending_summaries(5)


def test_collection_records_storage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(*args: object, **kwargs: object) -> dict[str, object]:
        return {"evidence": [{"title": "Title"}]}

    monkeypatch.setattr(news.exa_client, "search", search)

    def fail(*args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(market, "save_news", fail)
    assert news.collect_exa_news("AMD", now=NOW) == 0
    state = market.refresh_state("exa_news", "AMD")
    # The pass finishes normally after it records the row error.
    assert state and state["last_ok_at"] == NOW
    assert state["last_error"] == "disk full"
    assert not market.pending_summaries(5)


@pytest.mark.parametrize(
    "changes",
    [
        {"headline": " "},
        {"headline": ""},
        {"bottom_line": "\n"},
        {"bottom_line": ""},
        {"key_points": ["fact"] * 5},
        {"key_points": [" ", "fact"]},
        {"key_points": [1, "fact"]},
        {"key_points": None},
    ],
)
def test_story_rejects_empty_or_invalid_text(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="summarize_story"):
        check_story(summary(**changes))


@pytest.mark.parametrize("sentiment", ["bullish", "bearish", "neutral", "mixed"])
@pytest.mark.parametrize("impact", ["high", "medium", "low"])
def test_story_accepts_contract_boundaries(sentiment: str, impact: str) -> None:
    candidate = summary(
        relevant=False,
        headline="h" * 90,
        bottom_line=" ".join(["word"] * 30),
        key_points=[" ".join(["word"] * 25)] * 4,
        watch=None,
        limits="Headline only",
        sentiment=sentiment,
        impact=impact,
    )
    checked = check_story(candidate)
    assert checked == candidate
    assert checked is not candidate
    assert check_story(summary(headline="h", bottom_line="b", key_points=["a", "b"])) == summary(
        headline="h", bottom_line="b", key_points=["a", "b"]
    )


def test_body_fetch_batches_and_deduplicates_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(26):
        add_story(f"https://example.com/{index}", str(index))
    market.save_news("NVDA", [{"title": "Same story", "url": "https://example.com/0"}], now=NOW + 1)
    calls = []

    def contents(urls: list[str], *, caller: str) -> dict[str, object]:
        assert caller == "collector"
        calls.append(urls)
        return {"evidence": [{"url": url, "source_content": f"Body {url}"} for url in urls]}

    monkeypatch.setattr(news.exa_client, "contents", contents)
    assert news.fetch_bodies(30, now=NOW) == 27
    assert [len(urls) for urls in calls] == [24, 2]
    assert len({url for urls in calls for url in urls}) == 26
    assert all(row["body"] == f"Body {row['url']}" for row in market.pending_summaries(30))
    assert news.fetch_bodies(30, now=NOW) == 0
    assert len(calls) == 2


def test_summary_preserves_inputs_and_records_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    add_story()
    hash_ = market.url_hash("https://example.com/story")
    market.set_news_body("AMD", hash_, "Article body")
    statuses: list[tuple[str, str | None]] = []
    set_summary = market.set_summary

    def record_status(symbol: str, hash_: str, status: str, path: str | None = None) -> None:
        statuses.append((status, path))
        set_summary(symbol, hash_, status, path)

    def summarize(self: ReasonerClient, ticker: str, title: str, body: str) -> dict[str, object]:
        assert self.model == "test-model"
        assert self.url == "https://reasoner.invalid/responses"
        assert self.api_key == "test-key"
        assert (ticker, title, body) == ("AMD", "Title", "Article body")
        return summary()

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    monkeypatch.setattr(market, "set_summary", record_status)
    assert news.summarize_pending(1, now=NOW) == 1
    path = tmp_path / "news" / "2026-10-30" / f"AMD-company-raises-guidance-{hash_[:8]}.md"
    assert path.is_file()
    assert statuses == [("written", str(path))]
    assert market.read_news("AMD", 1)[0]["summary_path"] == str(path)
    state = market.refresh_state("news_summary", "AMD", hash_)
    assert state and state["last_ok_at"] == NOW and state["last_error"] is None


def test_brief_path_uses_published_day_and_bounded_slug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    published = datetime(2026, 10, 29, 23, 0, tzinfo=UTC).timestamp()
    market.save_news("AB CD", [{"title": "T", "url": "https://example.com/x", "published_at": published}], now=NOW)

    def summarize(*args: object) -> dict[str, object]:
        return summary(headline="A" * 100)

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW) == 1
    path = next((tmp_path / "news").rglob("*.md"))
    assert path.parent.name == "2026-10-29"
    assert path.name.startswith(f"AB-CD-{'a' * 70}-")


def test_two_stories_are_both_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()
    add_story("https://example.com/second", "Second")

    def summarize(*args: object) -> dict[str, object]:
        return summary()

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW) == 2


def test_missing_body_is_summarized_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()
    bodies: list[str] = []

    def summarize(*args: object) -> dict[str, object]:
        bodies.append(str(args[3]))
        return summary()

    monkeypatch.setattr(ReasonerClient, "summarize_story", summarize)
    assert news.summarize_pending(5, now=NOW) == 1
    assert bodies == [""]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"extra": "x"}, "summarize_story: unexpected fields"),
        ({"headline": 1}, "summarize_story: headline must be a nonempty string"),
        ({"headline": "x" * 91}, "summarize_story: headline must contain 1 to 90 characters"),
        ({"bottom_line": "word " * 31}, "summarize_story: bottom_line must contain 1 to 30 words"),
        ({"key_points": None}, "summarize_story: key_points must be string[]"),
        ({"watch": ""}, "summarize_story: watch must be a nonempty string"),
        ({"limits": 3}, "summarize_story: limits must be a nonempty string"),
    ],
)
def test_story_validation_messages(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError) as err:
        check_story(summary(**changes))
    assert str(err.value).startswith(message)
