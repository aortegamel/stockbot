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
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
SEARCH = (By.CSS_SELECTOR, 'input[placeholder="Search any US ticker…"]')
FIRST_HIT = (By.CSS_SELECTOR, 'header [role="option"]')
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
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            sym = query.get("sym", ["NVDA"])[0]
            bodies: dict[str, dict[str, object]] = {
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
                f"/api/research/sources/{quote(SOURCE_ID, safe='')}": SOURCE,
            }
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
            requests.put(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
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


def test_first_search_result_receives_pointer_click(browser: webdriver.Chrome) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(*SEARCH).send_keys("ORCL")
    first = wait.until(EC.visibility_of_element_located(FIRST_HIT))
    assert first.text == "ORCL Oracle Corporation"
    # The first result crosses the ticker strip. This reproduces the reported geometry.
    assert browser.execute_script(
        "const hit = arguments[0].getBoundingClientRect();"
        "const strip = document.querySelector('.strip').getBoundingClientRect();"
        "return hit.top < strip.bottom && hit.bottom > strip.top;",
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
    assert browser.find_element(*SEARCH).get_attribute("value") == ""
    assert not browser.find_elements(*FIRST_HIT)


def test_agent_request_keeps_question_separate_from_selected_ticker(browser: webdriver.Chrome, local_ui: UI) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(*SEARCH).send_keys("ORCL")
    wait.until(EC.visibility_of_element_located(FIRST_HIT)).click()
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "ORCL"))
    agent_input = browser.find_element(*AGENT_INPUT)
    wait.until(EC.element_to_be_clickable(AGENT_INPUT))
    agent_input.send_keys("What is the latest revenue?")
    browser.find_element(*SEND).click()
    # The actual watchlist rides beside the verbatim prompt; the server decides scope, the UI never rewrites the prompt.
    assert local_ui[1].get(timeout=10) == {
        "prompt": "What is the latest revenue?",
        "ticker": "ORCL",
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
    browser.find_element(*SEARCH).send_keys("ORCL")
    wait.until(EC.visibility_of_element_located(FIRST_HIT)).click()
    # Selecting a ticker outside the defaults changes the actual watchlist, so its sources load once more.
    assert local_ui[2].get(timeout=10) == ",".join(["ORCL", *WATCH_DEFAULTS])
    wait.until(EC.element_to_be_clickable(AGENT_INPUT)).send_keys("Summarize the latest filings for my watchlist")
    browser.find_element(*SEND).click()
    assert local_ui[1].get(timeout=10) == {
        "prompt": "Summarize the latest filings for my watchlist",
        "ticker": "ORCL",
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
    assert visible(browser, mark), "The cited passage is not visible."
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
    spy = (By.XPATH, "//div[normalize-space()='FILINGS · SPY']/..")
    wait.until(EC.text_to_be_present_in_element(spy, "filed 2026-03-13"))
    shown = browser.find_element(*spy).text
    assert "N-CSR - Certified Shareholder Report" in shown
    assert "2026-03-12" not in shown

    browser.find_element(*SEARCH).send_keys("ORCL")
    wait.until(EC.visibility_of_element_located(FIRST_HIT)).click()
    orcl = (By.XPATH, "//div[normalize-space()='FILINGS · ORCL']/..")
    wait.until(EC.text_to_be_present_in_element(orcl, "filings unavailable: HTTP Error 503"))
    assert "no SEC filings found" not in browser.find_element(*orcl).text
