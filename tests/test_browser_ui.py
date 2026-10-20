"""Offline Selenium checks use the actual Vite UI and local provider fixtures."""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import override
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import urlopen

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
# The header's one search field is the closed palette trigger; ticker hits are palette rows.
SEARCH = (By.CSS_SELECTOR, "header button.cmd-trigger")
PALETTE = (By.CSS_SELECTOR, 'input[role="combobox"]')
FIRST_HIT = (By.XPATH, "//*[@id='cmd-list']//*[@role='option'][b='ORCL']")
AGENT_INPUT = (By.CSS_SELECTOR, 'input[aria-label="Ask the research agent"]')
SEND = (By.CSS_SELECTOR, 'button[type="submit"]')
SOURCE_ID = "capture:fixture-long"
SOURCE: dict[str, object] = {
    "id": SOURCE_ID,
    "title": "10-K",
    "representation": "text",
    "captureKind": "original",
    "quote": "Revenue grew",
    "content": "\n".join("Revenue grew in the fixture year." if i == 5 else f"Capture line {i}" for i in range(1, 201)),
}
METRICS: dict[str, object] = {
    "totalMs": 1,
    "needle": {"calls": 0, "totalMs": 0, "escalations": 0},
    "tools": {"calls": 0, "totalMs": 0},
    "muse": {"calls": 1, "totalMs": 1},
    "evidence": {"count": 1, "characters": 0},
    "failures": {},
}
# The agent fixture streams a long answer. The answer always overflows the transcript.
STREAM = b"".join(
    b"data: " + json.dumps(frame).encode() + b"\n\n"
    for frame in [
        {"type": "reasoning_start", "model": "stockbot"},
        {
            "type": "evidence",
            "evidence": [
                {
                    "id": "ev:fixture",
                    "source": "sec",
                    "retrievedAt": "2026-10-10T00:00:00Z",
                    "content": "",
                    "sources": [{"id": SOURCE_ID, "title": "10-K"}],
                }
            ],
        },
        *(
            {"type": "answer_delta", "text": f"Line {i}: delayed research output for the layout check.\n"}
            for i in range(1, 121)
        ),
        {"type": "done", "metrics": METRICS},
    ]
)
# SPY is a fund. Its official filing date differs from the Atom updated timestamp.
FILINGS: dict[str, dict[str, object]] = {
    "SPY": {
        "sym": "SPY",
        "cik": 884394,
        "items": [
            {
                "title": "N-CSR - Certified Shareholder Report",
                "link": "https://www.sec.gov/Archives/edgar/data/884394/x-index.htm",
                "filed": "2026-03-13",
                "updated": "2026-03-12T17:20:00-04:00",
                "summary": "",
            }
        ],
    },
    "ORCL": {"sym": "ORCL", "cik": 1341439, "items": [], "error": "filings unavailable: HTTP Error 503"},
}
WATCH_DEFAULTS = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG"]
SOURCES = ["sec_filings", "finra_short_interest", "finra_short_sale_volume", "finra_threshold_list"]
CHART_PROMPT = "Show the AMD chart"
CITATION_PROMPT = "Cite the fixture source inline"
WATCHLIST_PROMPT = "Add AMD to my watchlist"
ALERT_PROMPT = "Alert me when NVDA short interest changes"
FIXTURE_SOURCE_PATH = "/fixture/source-doc"


def sse(*frames: dict[str, object]) -> bytes:
    return b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)


def watchlist_rows(symbols: list[str]) -> list[dict[str, object]]:
    """One row per (symbol, source): AAPL volume times out, QQQ has no threshold-list records."""
    rows: list[dict[str, object]] = []
    for sym in symbols:
        for source in SOURCES:
            row: dict[str, object] = {
                "symbol": sym,
                "source": source,
                "tool": source,
                "status": "ok",
                "as_of": "2026-10-09",
            }
            if (sym, source) == ("AAPL", "finra_short_sale_volume"):
                row = {"symbol": sym, "source": source, "tool": source, "status": "timeout", "error": "intake_timeout"}
            if (sym, source) == ("QQQ", "finra_threshold_list"):
                row = {
                    "symbol": sym,
                    "source": source,
                    "tool": source,
                    "status": "no_records",
                    "error": "No data found",
                }
            rows.append(row)
    return rows


UI = tuple[str, Queue[object], Queue[str]]


@pytest.fixture(scope="module")
def local_ui(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UI]:
    requests: Queue[object] = Queue()
    watch: Queue[str] = Queue()
    unknown: Queue[str] = Queue()

    class Provider(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlsplit(self.path).path == FIXTURE_SOURCE_PATH:
                page = b'<html><head><link rel="icon" href="data:,"></head><body><h1>Fixture source document</h1><p>Revenue grew in the fixture year.</p></body></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            sym = query.get("sym", ["NVDA"])[0]
            bodies: dict[str, object] = {
                "/api/health": {"debug": False},
                "/api/research/sessions": {"sessions": []},
                "/api/search": {
                    "query": query.get("q", [""])[0],
                    "hits": [
                        {"sym": "ORCL", "name": "Oracle Corporation"},
                        {"sym": "ORCLX", "name": "Second fixture result"},
                    ],
                },
                "/api/quote": {"sym": sym, "price": 150.0},
                "/api/candles": {
                    "sym": sym,
                    "range": query.get("range", ["6mo"])[0],
                    "interval": query.get("interval", ["1d"])[0],
                    "candles": [
                        {
                            "t": 1700000000 + i * 86400,
                            "open": 150 + i,
                            "high": 152 + i,
                            "low": 149 + i,
                            "close": 151 + i,
                            "vol": 1000,
                        }
                        for i in range(60)
                    ],
                },
                "/api/news": {"sym": sym, "items": []},
                "/api/filings": FILINGS.get(sym, {"sym": sym, "cik": None, "items": []}),
                "/api/watchlist": {"watchlist": watchlist_rows(query.get("symbols", [""])[0].split(","))},
                "/api/theses": [],
                "/api/scans": {
                    "scans": [
                        {"id": "watchlist", "name": "Watchlist", "kind": "preset", "filter": {}, "created_at": None}
                    ]
                },
                "/api/sparklines": {
                    "syms": {
                        ticker: {"t": [1, 2, 3], "closes": [100, 102, 101], "as_of": None}
                        for ticker in query.get("syms", [""])[0].split(",")
                    }
                },
                "/api/briefing": {
                    "date": "2026-10-20",
                    "syms": ["NVDA"],
                    "items": [],
                    "generated_at": "",
                    "cached": True,
                    "errors": [],
                },
                "/api/markets": {
                    "scan": query.get("scan", ["watchlist"])[0],
                    "sort": query.get("sort", ["chg_pct"])[0],
                    "universe": ["NVDA", "AAPL"],
                    "rows": [
                        {
                            "sym": ticker,
                            "last": 150.0,
                            "chg_pct": 1.0,
                            "volume": 1000,
                            "short_pct": None,
                            "last_filing": None,
                        }
                        for ticker in ["NVDA", "AAPL"]
                    ],
                    "errors": [],
                },
                "/api/positioning": {
                    "sym": sym,
                    "short_pct": None,
                    "short_as_of": None,
                    "insider_net_90d": None,
                    "threshold_listed": None,
                    "threshold_as_of": None,
                    "errors": {},
                },
                "/api/identity": {
                    "sym": sym,
                    "resolved": True,
                    "entity_id": "fixture-company",
                    "security_id": "fixture-security",
                },
                "/api/feed": {
                    "items": [
                        {
                            "id": "fixture-news",
                            "kind": "news",
                            "sym": "NVDA",
                            "title": "Fixture market update",
                            "at": "2026-10-20T12:00:00Z",
                            "source": "Fixture",
                        }
                    ],
                    "as_of": "2026-10-20T12:00:00Z",
                    "errors": [],
                },
            }
            port = server.server_port
            if url.path.startswith("/api/workspaces/"):
                bodies[url.path] = {"sym": url.path.rsplit("/", 1)[-1], "widgets": [], "updated_at": None}
            if url.path == f"/api/research/sources/{quote(SOURCE_ID, safe='')}":
                bodies[url.path] = {**SOURCE, "url": f"http://127.0.0.1:{port}{FIXTURE_SOURCE_PATH}"}
            if url.path == "/api/portfolio":
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                missing = json.dumps({"detail": "Robinhood portfolio unavailable."}).encode()
                self.send_header("Content-Length", str(len(missing)))
                self.end_headers()
                self.wfile.write(missing)
                return
            if url.path == "/api/watchlist":
                watch.put(query.get("symbols", [""])[0])
            if url.path not in bodies:
                unknown.put(self.path)
                self.send_error(500, "The browser test has no fixture for this route.")
                return
            body = json.dumps(bodies[url.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert self.headers.get("Content-Type") == "application/json"
            if self.path != "/api/agent":
                unknown.put(self.path)
                self.send_error(500, "The browser test has no fixture for this route.")
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.put(payload)
            prompt = payload.get("prompt") if isinstance(payload, dict) else None
            if prompt == CHART_PROMPT:
                body = sse(
                    {"type": "reasoning_start", "model": "stockbot"},
                    {"type": "chart_update", "ticker": "AMD"},
                    {"type": "answer_delta", "text": "Chart moved to AMD."},
                    {"type": "done", "metrics": METRICS},
                )
            elif prompt == CITATION_PROMPT:
                origin = f"http://127.0.0.1:{server.server_port}"
                body = sse(
                    {"type": "reasoning_start", "model": "stockbot"},
                    {
                        "type": "evidence",
                        "evidence": [
                            {
                                "id": "ev:fixture",
                                "source": "sec",
                                "retrievedAt": "2026-10-10T00:00:00Z",
                                "content": "",
                                "sources": [{"id": SOURCE_ID, "title": "10-K", "url": origin + FIXTURE_SOURCE_PATH}],
                            }
                        ],
                    },
                    {"type": "answer_delta", "text": "Revenue grew [ev:fixture] this year."},
                    {"type": "done", "metrics": METRICS},
                )
            elif prompt == WATCHLIST_PROMPT:
                body = sse(
                    {"type": "watchlist_update", "action": "add", "ticker": "AMD"},
                    {"type": "answer_delta", "text": "Added AMD to your watchlist."},
                    {"type": "done", "metrics": METRICS},
                )
            elif prompt == ALERT_PROMPT:
                body = sse(
                    {
                        "type": "tool_result",
                        "tool": "thesis_watch",
                        "evidenceId": "ev:alert",
                        "preview": "thesis_id: thesis:fixture\nadded: rule_id: rule:fixture, rule_type: material_short_interest_change, enabled: True",
                    },
                    {"type": "answer_delta", "text": "The alert is set."},
                    {"type": "done", "metrics": METRICS},
                )
            else:
                body = STREAM
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    bun = shutil.which("bun")
    assert bun, "Install Bun and run `bun install --cwd web` before the browser regression."
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "STOCKBOT_API_PORT": str(server.server_port),
        "STOCKBOT_MARKET_PORT": str(server.server_port),
        "VITE_API_URL": "",
    }
    log = tmp_path_factory.mktemp("browser-ui") / "vite.log"
    url = f"http://127.0.0.1:{port}"
    try:
        with log.open("w") as output:
            process = subprocess.Popen(
                [bun, "x", "vite", "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
                cwd=ROOT / "web",
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    assert process.poll() is None, log.read_text()
                    try:
                        with urlopen(url, timeout=1) as response:
                            if response.status == 200:
                                break
                    except URLError:
                        time.sleep(0.05)
                    except TimeoutError:
                        time.sleep(0.05)
                else:
                    pytest.fail(f"Vite did not start. {log.read_text()}")
                yield url, requests, watch
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert unknown.empty(), f"The browser requested an unconfigured route: {list(unknown.queue)}"


@pytest.fixture
def browser(local_ui: UI) -> Iterator[webdriver.Chrome]:
    binary = os.environ.get("STOCKBOT_CHROMIUM_BINARY") or next(
        (path for name in ("chromium", "chromium-browser", "google-chrome") if (path := shutil.which(name))), None
    )
    driver = os.environ.get("STOCKBOT_CHROMEDRIVER") or shutil.which("chromedriver")
    assert binary and os.access(binary, os.X_OK), "Install Chromium or set STOCKBOT_CHROMIUM_BINARY to its executable."
    assert driver and os.access(driver, os.X_OK), "Install matching ChromeDriver or set STOCKBOT_CHROMEDRIVER."
    options = webdriver.ChromeOptions()
    options.binary_location = binary
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1000")
    options.add_argument("--disable-background-networking")
    options.add_argument("--no-proxy-server")
    options.add_argument("--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1")
    with webdriver.Chrome(service=Service(executable_path=driver), options=options) as client:
        client.get(local_ui[0])
        WebDriverWait(client, 20).until(EC.visibility_of_element_located(SEARCH))
        yield client


def search_orcl(browser: webdriver.Chrome) -> WebElement:
    wait = WebDriverWait(browser, 10)
    browser.find_element(*SEARCH).click()
    wait.until(EC.visibility_of_element_located(PALETTE)).send_keys("ORCL")
    return wait.until(EC.visibility_of_element_located(FIRST_HIT))


def test_first_search_result_receives_pointer_click(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    first = search_orcl(browser)
    # The flex row lays out the symbol and name as separate items, so Selenium reads a line break between them.
    assert " ".join(first.text.split()) == "ORCL Oracle Corporation"
    # The quote strip lives in the footer, so the header dropdown never overlaps it.
    assert browser.execute_script(
        "const hit = arguments[0].getBoundingClientRect();"
        "const strip = document.querySelector('footer .strip').getBoundingClientRect();"
        "return hit.bottom <= strip.top;",
        first,
    )
    browser.execute_script(
        "arguments[0].addEventListener('pointerdown', event => {"
        "window.searchPointerTrusted = event.isTrusted; }, {once: true});",
        first,
    )
    first.click()  # WebDriver sends a native pointer click, not a synthetic JavaScript click.
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "ORCL"))
    assert browser.execute_script("return window.searchPointerTrusted") is True
    # A palette pick always opens the ticker workspace and closes the palette.
    wait.until(EC.url_contains("#/workspace/ORCL"))
    assert not browser.find_elements(*PALETTE)
    assert not browser.find_elements(*FIRST_HIT)


def test_agent_request_keeps_question_separate_from_selected_ticker(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    search_orcl(browser).click()
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "ORCL"))
    agent_input = browser.find_element(*AGENT_INPUT)
    wait.until(EC.element_to_be_clickable(AGENT_INPUT))
    agent_input.send_keys("What is the latest revenue?")
    browser.find_element(*SEND).click()
    # The actual watchlist rides beside the verbatim prompt; the server decides scope, the UI never rewrites the prompt.
    assert local_ui[1].get(timeout=10) == {
        "prompt": "What is the latest revenue?",
        "ticker": "ORCL",
        "currentChartTicker": "ORCL",
        "watchlist": ["ORCL", *WATCH_DEFAULTS],
    }


def drain(queue: Queue[str]) -> None:
    while not queue.empty():
        queue.get_nowait()


def sources_text(browser: webdriver.Chrome, sym: str) -> str:
    return browser.find_element(By.CSS_SELECTOR, f'[aria-label="{sym} sources"]').text


def test_page_load_fetches_every_watchlist_source_once(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    # Let this browser's first load finish, so its request can never land after the drain.
    wait.until(lambda b: "SEC" in sources_text(b, "GOOG"))
    drain(local_ui[2])
    browser.refresh()
    wait.until(lambda b: "VOL timeout" in sources_text(b, "AAPL"))
    # One page load: one fan-out request carrying the actual watchlist, nothing else.
    assert local_ui[2].get(timeout=5) == ",".join(WATCH_DEFAULTS)
    assert local_ui[2].empty()
    assert sources_text(browser, "NVDA") == "SEC 2026-10-09 SI 2026-10-09 VOL 2026-10-09 THR 2026-10-09"
    assert "THR none" in sources_text(browser, "QQQ")
    # Every symbol shows all four sources, so no slow issuer or failed source disappears.
    for sym in WATCH_DEFAULTS:
        assert sources_text(browser, sym).split()[::2] == ["SEC", "SI", "VOL", "THR"], sym


def test_watchlist_question_sends_actual_watchlist_with_verbatim_prompt(
    browser: webdriver.Chrome, local_ui: UI
) -> None:
    wait = WebDriverWait(browser, 10)
    wait.until(lambda b: "SEC" in sources_text(b, "GOOG"))
    drain(local_ui[2])
    search_orcl(browser).click()
    # Selecting a ticker outside the defaults changes the actual watchlist, so its sources load once more.
    assert local_ui[2].get(timeout=10) == ",".join(["ORCL", *WATCH_DEFAULTS])
    wait.until(EC.element_to_be_clickable(AGENT_INPUT)).send_keys("Summarize the latest filings for my watchlist")
    browser.find_element(*SEND).click()
    assert local_ui[1].get(timeout=10) == {
        "prompt": "Summarize the latest filings for my watchlist",
        "ticker": "ORCL",
        "currentChartTicker": "ORCL",
        "watchlist": ["ORCL", *WATCH_DEFAULTS],
    }


# Hit tests at the top and bottom edges. A clipped or covered element fails without any scrolling.
VISIBLE = """
const element = arguments[0];
const box = element.getBoundingClientRect();
const x = box.left + box.width / 2;
return [box.top + 1, box.bottom - 1].every((y) => {
  const hit = document.elementFromPoint(x, y);
  return hit !== null && element.contains(hit);
});
"""


def visible(browser: webdriver.Chrome, element: WebElement) -> bool:
    return bool(browser.execute_script(VISIBLE, element))


def assert_prompt_reachable(browser: webdriver.Chrome) -> None:
    assert visible(browser, browser.find_element(*AGENT_INPUT)), "The agent input is clipped or outside the viewport."
    assert visible(browser, browser.find_element(*SEND)), "The Send button is clipped or outside the viewport."


def wheel_until(browser: webdriver.Chrome, origin: tuple[int, int], reached: Callable[[], object]) -> None:
    """Send native wheel input at one viewport point until `reached()` is true."""
    for _ in range(40):
        try:
            WebDriverWait(browser, 1, poll_frequency=0.05).until(lambda _: reached())
            return
        except TimeoutException:
            ActionChains(browser).scroll_from_origin(ScrollOrigin.from_viewport(*origin), 0, 240).perform()
    pytest.fail("The target stayed out of reach after 40 wheel steps.")


def transcript_point(browser: webdriver.Chrome) -> tuple[int, int]:
    """Return a viewport point inside the transcript, directly above the prompt form."""
    box = browser.find_element(*AGENT_INPUT).rect
    return int(box["x"] + box["width"] / 2), int(box["y"] - 80)


def settle(browser: webdriver.Chrome, element: WebElement) -> None:
    """Wait until smooth scrolling for the element and its scroll ancestors stops."""
    last: list[object] = []

    def still(_: object) -> bool:
        now = browser.execute_script(
            "const box = arguments[0].getBoundingClientRect(); return [box.top, arguments[0].scrollTop];", element
        )
        same = last == [now]
        last[:] = [now]
        return same

    WebDriverWait(browser, 5, poll_frequency=0.2).until(still)


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 720)])
def test_agent_transcript_scrolls_inside_panel(
    browser: webdriver.Chrome, local_ui: UI, width: int, height: int
) -> None:
    browser.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )
    WebDriverWait(browser, 5).until(
        lambda client: client.execute_script("return [innerWidth, innerHeight]") == [width, height]
    )

    # Empty introduction: the prompt is reachable and the last intro line is reachable by wheel alone.
    assert_prompt_reachable(browser)
    intro_end = browser.find_element(By.XPATH, "//div[starts-with(normalize-space(), 'Try:')]")
    wheel_until(browser, transcript_point(browser), lambda: visible(browser, intro_end))
    assert_prompt_reachable(browser)

    # Long streamed answer: the answer stays inside the transcript and the prompt stays reachable.
    browser.find_element(*AGENT_INPUT).send_keys("Show the long answer.")
    browser.find_element(*SEND).click()
    assert local_ui[1].get(timeout=10) == {
        "prompt": "Show the long answer.",
        "ticker": "NVDA",
        "currentChartTicker": "NVDA",
        "watchlist": WATCH_DEFAULTS,
    }
    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, "//div[text()='── metrics ──']")))
    assert_prompt_reachable(browser)
    sources = browser.find_element(By.CSS_SELECTOR, 'button[aria-label="Open captured sources for ev:fixture"]')
    assert not visible(browser, sources), "The long answer must overflow the transcript for this check."
    wheel_until(browser, transcript_point(browser), lambda: visible(browser, sources))
    assert_prompt_reachable(browser)

    # Captured source: the content scrolls on its own until the final line.
    sources.click()
    WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.XPATH, "//aside//p[@role='status'][contains(., 'quote matches')]"))
    )
    content = browser.find_element(By.CSS_SELECTOR, '[aria-label="Captured source content"]')
    mark = content.find_element(By.TAG_NAME, "mark")
    settle(browser, content)
    # activateQuote scrolls smoothly; a loaded machine can delay the animation past settle's window.
    WebDriverWait(browser, 10).until(lambda _: visible(browser, mark), "The cited passage is not visible.")
    assert browser.execute_script("return arguments[0].scrollHeight > arguments[0].clientHeight + 1", content)
    box = mark.rect
    wheel_until(
        browser,
        (int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)),
        lambda: browser.execute_script(
            "const c = arguments[0]; return c.scrollTop > 0 && c.scrollTop + c.clientHeight >= c.scrollHeight - 1;",
            content,
        ),
    )
    assert_prompt_reachable(browser)


def test_filings_panel_shows_fund_filing_date_and_provider_error(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(By.XPATH, "//button[b='SPY']").click()
    spy = (By.XPATH, "//section[@aria-label='Financials · SPY']")
    wait.until(EC.text_to_be_present_in_element(spy, "filed 2026-03-13"))
    shown = browser.find_element(*spy).text
    assert "N-CSR - Certified Shareholder Report" in shown
    assert "2026-03-12" not in shown

    search_orcl(browser).click()
    orcl = (By.XPATH, "//section[@aria-label='Financials · ORCL']")
    wait.until(EC.text_to_be_present_in_element(orcl, "We could not load this data."))
    panel = browser.find_element(*orcl)
    assert "filings unavailable: HTTP Error 503" not in panel.text
    assert "No filings for ORCL" not in panel.text
    retry = panel.find_element(By.XPATH, ".//button[normalize-space()='Retry']")
    retry.click()
    wait.until(EC.text_to_be_present_in_element(orcl, "We could not load this data."))


def send_chat(browser: webdriver.Chrome, local_ui: UI, prompt: str) -> None:
    while not local_ui[1].empty():
        local_ui[1].get_nowait()
    WebDriverWait(browser, 10).until(EC.element_to_be_clickable(AGENT_INPUT)).send_keys(prompt)
    browser.find_element(*SEND).click()
    request = local_ui[1].get(timeout=10)
    assert isinstance(request, dict) and request["prompt"] == prompt


def test_chat_chart_update_moves_chart_to_amd(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    send_chat(browser, local_ui, CHART_PROMPT)
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "AMD"))
    assert browser.find_element(By.CSS_SELECTOR, 'iframe[title="AMD TradingView chart"]')
    assert "AGENT · viewing AMD" in browser.find_element(By.CSS_SELECTOR, "main").text


def test_chat_citation_opens_fixture_source_in_new_tab(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    send_chat(browser, local_ui, CITATION_PROMPT)
    citation = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[aria-label="Open source for ev:fixture"]')))
    assert citation.text == "[ev:fixture]"
    href = citation.get_attribute("href") or ""
    assert FIXTURE_SOURCE_PATH in href and href.startswith("http://127.0.0.1:")
    original = browser.current_window_handle
    tabs_before = set(browser.window_handles)
    citation.click()
    wait.until(lambda _: len(set(browser.window_handles) - tabs_before) == 1)
    new_tab = (set(browser.window_handles) - tabs_before).pop()
    browser.switch_to.window(new_tab)
    try:
        wait.until(EC.url_contains(FIXTURE_SOURCE_PATH))
        assert "Fixture source document" in browser.page_source
    finally:
        browser.close()
        browser.switch_to.window(original)
    assert browser.current_window_handle == original
    assert FIXTURE_SOURCE_PATH not in browser.current_url


def test_chat_watchlist_update_shows_amd_without_moving_chart(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    wait.until(lambda b: "SEC" in sources_text(b, "GOOG"))
    send_chat(browser, local_ui, WATCHLIST_PROMPT)
    wait.until(lambda b: "AMD" in b.find_element(By.XPATH, "//div[normalize-space()='WATCHLIST · DELAYED']/..").text)
    assert browser.find_element(By.XPATH, "//button[b='AMD']")
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "NVDA"))


def test_page_navigation_uses_nav_links(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    for label, route, content in [
        ("Home", "#/home", ".home-table"),
        ("Workspaces", "#/workspace/NVDA", ".ws-grid"),
        ("Feed", "#/feed", ".feed-row"),
    ]:
        link = browser.find_element(By.LINK_TEXT, label)
        browser.execute_script(
            "arguments[0].addEventListener('click', event => { window.navTrusted = event.isTrusted; }, {once: true});",
            link,
        )
        link.click()
        wait.until(EC.url_contains(route))
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, content)))
        assert browser.execute_script("return window.navTrusted") is True
        assert browser.find_element(By.LINK_TEXT, label).get_attribute("aria-current") == "page"


def test_palette_keyboard_opens_ticker_workspace(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    ActionChains(browser).send_keys("/").perform()
    palette = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[role="combobox"]')))
    palette.send_keys("ORCL")
    wait.until(EC.text_to_be_present_in_element((By.ID, "cmd-list"), "Oracle Corporation"))
    palette.send_keys(Keys.ENTER)
    wait.until(EC.url_contains("#/workspace/ORCL"))
    assert not browser.find_elements(By.CSS_SELECTOR, 'input[role="combobox"]')

    agent_input = wait.until(EC.element_to_be_clickable(AGENT_INPUT))
    agent_input.send_keys("/")
    assert agent_input.get_attribute("value") == "/"
    assert not browser.find_elements(By.CSS_SELECTOR, 'input[role="combobox"]')
    agent_input.clear()
    browser.find_element(By.LINK_TEXT, "Home").click()
    ActionChains(browser).send_keys("/").perform()
    palette = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[role="combobox"]')))
    palette.send_keys(Keys.ESCAPE)
    wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'input[role="combobox"]')))


def test_add_widget_increases_workspace_grid(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(By.LINK_TEXT, "Workspaces").click()
    menu = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'select[aria-label="Add widget"]')))
    before = len(browser.find_elements(By.CSS_SELECTOR, ".ws-grid > section"))
    Select(menu).select_by_visible_text("Monthly returns")
    wait.until(lambda client: len(client.find_elements(By.CSS_SELECTOR, ".ws-grid > section")) == before + 1)
    widget = wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, 'section[aria-label="Monthly returns · NVDA"]'))
    )
    assert widget.find_element(By.CSS_SELECTOR, 'table[aria-label="NVDA monthly returns"]')


def test_offline_alert_receipt_renders_card(browser: webdriver.Chrome, local_ui: UI) -> None:
    send_chat(browser, local_ui, ALERT_PROMPT)
    result = WebDriverWait(browser, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, '[aria-label="Results"]'))
    )
    assert "Alert set: material short interest change" in result.text
    assert "thesis_watch" not in result.text
    assert "rule:fixture" not in result.text


def test_markets_rows_support_arrow_navigation(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(By.LINK_TEXT, "Home").click()
    first = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".home-table tbody tr[tabindex='0']")))
    assert "NVDA" in first.text
    ActionChains(browser).click(first).send_keys(Keys.ARROW_DOWN).perform()
    wait.until(lambda client: "AAPL" in client.switch_to.active_element.text)
    selected = browser.switch_to.active_element
    assert selected.tag_name == "tr"
    assert selected.get_attribute("tabindex") == "0"
    assert selected.get_attribute("aria-selected") == "true"
    selected.send_keys(Keys.ARROW_DOWN)
    assert browser.switch_to.active_element == selected
    selected.send_keys(Keys.ARROW_UP)
    wait.until(lambda client: client.switch_to.active_element == first)
    assert first.get_attribute("tabindex") == "0"
