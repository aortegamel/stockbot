#!/usr/bin/env python3
"""Opt-in live end-to-end runner: evals/golden_live.json through POST /api/agent SSE (never CI).

Requires a running harness (`bun run stockbot`) and the credentials of
scripts/verify_hedgefund_live.py. This script performs real network calls. Invoke only:
  STOCKBOT_E2E=1 venv/bin/python scripts/verify_e2e.py [--base-url URL] [--only ID] [--runs N]
Without STOCKBOT_E2E=1 it exits 2 with an explicit opt-in message.

Each case selects chat or research mode. Research requests explicitly set mode in the body.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_hedgefund_live import _missing_env

GOLDEN = REPO_ROOT / "evals" / "golden_live.json"
DEFAULT_BASE_URL = "http://127.0.0.1:3000"  # needle.ts PORT default; Vite proxies /api/agent to the Bun API.
TIMEOUT_S = 400  # two 120s server walls + persona time; per-read idle and total guard share it.


class Case(TypedDict):
    id: str
    prompt: str
    chart_ticker: str | None
    mode: Literal["chat", "research"]
    ticker: str | None
    min_citations: int
    max_first_text_ms: int
    max_total_ms: int


@dataclass
class Run:
    events: list[str] = field(default_factory=list)
    first_text_ms: float | None = None
    first_progress_ms: float | None = None  # First readable status line, before any answer text.
    total_ms: float | None = None
    mode: Literal["chat", "research"] | None = None
    citation_count: int = 0
    answer: str = ""
    known: list[str] = field(default_factory=list)
    claims: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings: object = None
    chart_updates: list[str] = field(default_factory=list)


def _mode(types: list[str], metrics: object) -> Literal["chat", "research"] | None:
    """Infer research from worker progress or chat from a completed, single model call."""
    if "progress" in types:
        return "research"
    if "done" not in types or "error" in types or "failed" in types:
        return None
    tool_metrics = metrics.get("tools") if isinstance(metrics, dict) else None
    muse_metrics = metrics.get("muse") if isinstance(metrics, dict) else None
    if (
        isinstance(tool_metrics, dict)
        and tool_metrics.get("calls") == 0
        and isinstance(muse_metrics, dict)
        and muse_metrics.get("calls") == 1
    ):
        return "chat"
    return None


def parse_stream(lines: Iterable[tuple[float, bytes]]) -> Run:
    """Fold timed SSE lines (elapsed ms, raw line) into one run record. No clock, no I/O."""
    run = Run()
    answer: list[str] = []
    known: set[str] = set()
    claims: list[dict[str, object]] = []
    last_ms: float | None = None
    for ms, raw in lines:
        last_ms = ms
        if not raw.startswith(b"data:"):
            continue
        try:
            event = json.loads(raw[5:])
        except ValueError:
            run.errors.append(f"malformed SSE data: {raw[:80]!r}")
            continue
        kind = str(event.get("type")) if isinstance(event, dict) else "non-object"
        run.events.append(kind)
        if kind == "status":
            run.first_progress_ms = ms if run.first_progress_ms is None else run.first_progress_ms
        elif kind == "answer_delta":
            run.first_text_ms = ms if run.first_text_ms is None else run.first_text_ms
            answer.append(str(event.get("text", "")))
        elif kind == "evidence":
            # Each evidence event carries the full merged list, so the last one wins (agent-console.tsx).
            known = {str(e.get("id")) for e in event.get("evidence") or [] if isinstance(e, dict)}
            run.known = sorted(known)
        elif kind == "claims":
            # The gated claim inventory: the console chips each claim whose text appears in the answer.
            claims = [c for c in event.get("claims") or [] if isinstance(c, dict)]
        elif kind in ("failed", "error"):
            run.errors.append(f"{kind}: {event.get('message')}")
        elif kind == "chart_update":
            run.chart_updates.append(str(event.get("ticker")))
        elif kind == "done":
            run.total_ms, run.timings = ms, event.get("metrics")
    if run.total_ms is None:
        run.total_ms = last_ms
    text = "".join(answer)
    cited = {i for i in known if f"[{i}]" in text}
    for c in claims:
        ids = c.get("evidenceIds")
        if c.get("text") and str(c["text"]) in text and isinstance(ids, list):
            cited |= {str(i) for i in ids if str(i) in known}
    run.citation_count = len(cited)
    run.answer = text
    run.claims = claims
    run.mode = _mode(run.events, run.timings)
    return run


def evaluate(case: Case, run: Run) -> list[str]:
    """Names of the limits this run breaks; empty means pass."""
    fails = []
    if run.mode != case["mode"]:
        fails.append("mode")
    if run.first_text_ms is None or run.first_text_ms > case["max_first_text_ms"]:
        fails.append("first_text_ms")
    if run.total_ms is None or run.total_ms > case["max_total_ms"]:
        fails.append("total_ms")
    if run.citation_count < case["min_citations"]:
        fails.append("citations")
    if run.errors or "done" not in run.events:
        fails.append("terminal")
    if case["ticker"] and any(t != case["ticker"] for t in run.chart_updates):
        fails.append("ticker")
    return fails


def _post(base_url: str, case: Case) -> Run:
    body: dict[str, str] = {"prompt": case["prompt"]}
    if case["mode"] == "research":
        body["mode"] = "research"
    if case["chart_ticker"]:
        body["ticker"] = case["chart_ticker"]  # App.tsx sends the selected chart symbol as `ticker`.
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/agent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - t0) * 1000

    def timed(resp: Iterable[bytes]) -> Iterator[tuple[float, bytes]]:
        for line in resp:
            now = elapsed()
            if now > TIMEOUT_S * 1000:
                raise TimeoutError(f"request exceeded {TIMEOUT_S}s")
            yield now, line

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return parse_stream(timed(resp))
    except (OSError, ValueError) as exc:  # URLError, HTTPError, and socket timeouts are OSError.
        return Run(total_ms=elapsed(), errors=[f"transport: {exc}"])


def _pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    v = sorted(values)
    return v[max(0, math.ceil(p * len(v)) - 1)] if v else None


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="harness URL serving /api/agent")
    parser.add_argument("--only", default=None, help="run one golden prompt id")
    parser.add_argument("--runs", type=int, default=1, help="runs per prompt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (os.getenv("STOCKBOT_E2E") or "").strip() != "1":
        print("SKIP e2e: set STOCKBOT_E2E=1 to opt in (live harness, never CI).", file=sys.stderr)
        return 2
    missing = _missing_env()
    if missing:
        print(f"SKIP e2e: missing {', '.join(missing)}.", file=sys.stderr)
        return 2
    golden: list[Case] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = [c for c in golden if args.only in (None, c["id"])]
    if not cases:
        print(f"unknown golden id: {args.only!r}", file=sys.stderr)
        return 2
    results: list[tuple[Case, int, Run, list[str]]] = []
    for case in cases:
        for n in range(1, args.runs + 1):
            run = _post(args.base_url, case)
            results.append((case, n, run, evaluate(case, run)))
    research = [run for case, _, run, _ in results if case["mode"] == "research"]
    first = [r.first_text_ms for r in research if r.first_text_ms is not None]
    progress = [r.first_progress_ms for r in research if r.first_progress_ms is not None]
    total = [r.total_ms for r in research if r.total_ms is not None]
    passed = sum(not fails for *_, fails in results)
    summary = {
        "runs": len(results),
        "passed": passed,
        "research": {
            "first_progress_ms": {"p50": _pct(progress, 0.5), "p90": _pct(progress, 0.9)},
            "first_text_ms": {"p50": _pct(first, 0.5), "p90": _pct(first, 0.9)},
            "total_ms": {"p50": _pct(total, 0.5), "p90": _pct(total, 0.9)},
        },
    }
    rows = [{"id": c["id"], "run": n, **asdict(run), "failures": f, "pass": not f} for c, n, run, f in results]
    out = REPO_ROOT / "data" / "verify" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") / "e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"base_url": args.base_url, "runs": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    print("| id | mode | first_progress_ms | first_text_ms | total_ms | citations | pass |")
    print("|---|---|---|---|---|---|---|")
    for case, n, run, fails in results:
        label = f"{case['id']}#{n}" if args.runs > 1 else case["id"]
        verdict = "PASS" if not fails else f"FAIL {','.join(fails)}"
        print(
            f"| {label} | {run.mode} | {_ms(run.first_progress_ms)} | {_ms(run.first_text_ms)} | {_ms(run.total_ms)} | {run.citation_count} | {verdict} |"
        )
    print(f"\n{passed}/{len(results)} passed; research p50/p90 ms: {json.dumps(summary['research'])}")
    print(f"wrote {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
