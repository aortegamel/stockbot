"""Offline Selenium checks use the actual Vite UI and local provider fixtures."""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import override
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
SEARCH = (By.CSS_SELECTOR, 'input[placeholder="Search any US ticker…"]')
FIRST_HIT = (By.CSS_SELECTOR, 'header [role="option"]')


@pytest.fixture(scope="module")
def local_ui(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Queue[object]]]:
    requests: Queue[object] = Queue()
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
                "/api/filings": {"sym": sym, "cik": None, "items": []},
            }
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
            body = b'data: {"type":"done"}\n\n'
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
                yield url, requests
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
def browser(local_ui: tuple[str, Queue[object]]) -> Iterator[webdriver.Chrome]:
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


def test_agent_request_keeps_question_separate_from_selected_ticker(
    browser: webdriver.Chrome, local_ui: tuple[str, Queue[object]]
) -> None:
    wait = WebDriverWait(browser, 10)
    browser.find_element(*SEARCH).send_keys("ORCL")
    wait.until(EC.visibility_of_element_located(FIRST_HIT)).click()
    wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "footer > span:last-child"), "ORCL"))
    agent_input = browser.find_element(By.CSS_SELECTOR, 'input[aria-label="Ask the research agent"]')
    browser.execute_script("arguments[0].scrollIntoView({block: 'nearest'});", agent_input)
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[aria-label="Ask the research agent"]')))
    agent_input.send_keys("What is the latest revenue?")
    browser.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
    assert local_ui[1].get(timeout=10) == {"prompt": "What is the latest revenue?", "ticker": "ORCL"}
