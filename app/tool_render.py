"""Render tool results as compact plain text/Markdown for the main model.

Internal results stay structured Python dicts/JSON (caching, validation,
deterministic metrics, the small FINRA analysis model's validated prose).
Only the tool message sent to the main chat model is rendered to compact
text. Rendering enforces a fixed UTF-8 byte budget and reduces the structure
(rows first, then verbose sections, then individual text values) before
rendering — it never slices serialized JSON or rendered text blindly.
"""

from __future__ import annotations

from decimal import Decimal

from app.domain.risk.evaluation import EvaluationIssue
from app.services.portfolio_research import SEC_CONCEPTS

MAX_TOOL_MESSAGE_BYTES = 64 * 1024

TRUNCATED_MARKER = "... [Tool output truncated]"

_TEXT_KEYS = ("text", "diff", "summary")


def _as_dict(value: object) -> dict[str, object]:
    """Narrow untrusted render JSON to a mapping ({} when absent/mistyped)."""
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    """Narrow untrusted render JSON to a list ([] when absent/mistyped)."""
    return value if isinstance(value, list) else []


def _dispatch_result_type_render_a(result: dict[str, object], max_bytes: int) -> str | None:
    """Renderers for market/option result_type tags, else None."""
    tag = result.get("result_type")
    if tag == "market_snapshot":
        return _render_market_snapshot(result, max_bytes)
    if tag == "option_chain":
        return _render_option_chain(result, max_bytes)
    if tag == "option_analysis":
        return _render_option_analysis(result, max_bytes)
    if tag == "option_comparison":
        return _render_option_comparison(result, max_bytes)
    return None


def _dispatch_result_type_render_b(result: dict[str, object], max_bytes: int) -> str | None:
    """Renderers for portfolio/mandate/scan result_type tags, else None."""
    tag = result.get("result_type")
    if tag == "portfolio_snapshot":
        return _render_portfolio_snapshot(result, max_bytes)
    if tag == "mandate_evaluation":
        return _render_mandate_evaluation(result, max_bytes)
    if tag == "scan_specs":
        return _render_scan_specs(result, max_bytes)
    if tag == "scan_list":
        return _render_scan_list(result, max_bytes)
    if tag == "scan_results":
        return _render_scan_results(result, max_bytes)
    return None


def _dispatch_shape_render_a(result: dict[str, object], max_bytes: int) -> str | None:
    """Shape-dispatched renderers for analyst/market-structure envelopes."""
    text = _dispatch_shape_render_consensus(result, max_bytes)
    if text is not None:
        return text
    return _dispatch_shape_render_ledger(result, max_bytes)


def _dispatch_shape_render_consensus(result: dict[str, object], max_bytes: int) -> str | None:
    """Analyst-consensus and index-weight envelopes, else None."""
    if "price_targets" in result and "forward_estimates" in result:
        return _render_analyst_estimates(result, max_bytes)
    if "weight_pct" in result and "rank" in result:
        return _render_sp500_weight(result, max_bytes)
    return None


def _dispatch_shape_render_ledger_debt(result: dict[str, object], max_bytes: int) -> str | None:
    """Obligations/valuation envelopes, else None."""
    if "obligations" in result and "current_snapshot" in result and "forward_eps" not in result:
        return _render_obligations(result, max_bytes)
    if "forward_eps" in result and "obligations" in result:
        return _render_valuation_metrics(result, max_bytes)
    return None


def _dispatch_shape_render_ledger(result: dict[str, object], max_bytes: int) -> str | None:
    """Obligations/valuation/short-interest envelopes, else None."""
    text = _dispatch_shape_render_ledger_debt(result, max_bytes)
    if text is not None:
        return text
    if "entries" in result and "settlement_date" in result:
        return _render_short_interest_leaderboard(result, max_bytes)
    return None


def _dispatch_shape_render_table(result: dict[str, object], max_bytes: int) -> str | None:
    """Table/briefing envelopes, else None."""
    if "records" in result and "fields" in result:
        return _render_datapoints(result, max_bytes)
    if "coverage" in result and "metrics" in result:
        return _render_briefing(result, max_bytes)
    return None


def _dispatch_shape_render_doc(result: dict[str, object], max_bytes: int) -> str:
    """SEC/web-search/text/generic envelopes (total dispatch fallback)."""
    if result.get("source") == "sec" and "metric" in result:
        return _render_sec_facts(result, max_bytes)
    if result.get("result_type") == "web_search":
        return _render_web_search(result, max_bytes)
    if _is_insider_envelope(result):
        return _render_insider_activity(result, max_bytes)
    if _is_text_result(result):
        return _render_text_result(result, max_bytes)
    return _render_generic(result, max_bytes)


def _final_ids(refs: object, cap: int = 6) -> str:
    """Evidence-refs suffix from evidence ids (capped, empty when none)."""
    ids: list[str] = [e for e in refs if isinstance(e, str) and e.strip()] if isinstance(refs, list) else []
    return f" [{', '.join(ids[:cap])}]" if ids else ""


def _final_type_suffix(row: dict[str, object]) -> str:
    """Declared claim-type suffix (empty when the row states none)."""
    ctype = _cell(row.get("claim_type")).strip()
    return f" ({ctype})" if ctype else ""


def _final_claims_block(claims: object) -> list[str]:
    """Grounded-claim lines with declared type + evidence refs ([] when none)."""
    items = _as_list(claims)
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _cell(item.get("text")).strip()
        if not text:
            continue
        lines.append(f"- {text}{_final_type_suffix(item)}{_final_ids(item.get('evidence_ids'))}")
    return lines


def _final_channel_line(item: object) -> str:
    """One impact-channel line with severity + evidence refs."""
    row = _as_dict(item)
    name = _cell(row.get("name")).strip() or "Exposure"
    severity = _cell(row.get("severity")).strip() or "direct"
    explanation = _cell(row.get("explanation")).strip()
    detail = "" if not explanation or explanation == name or explanation == severity else f": {explanation}"
    return f"- {name} ({severity}){detail}{_final_ids(row.get('evidence_ids'))}"


def _final_side_block(side: object) -> list[str]:
    """Bull/bear summary line with its evidence refs ([] when no summary)."""
    row = _as_dict(side) if not isinstance(side, str) else {"summary": side}
    summary = _cell(row.get("summary")).strip()
    if not summary:
        return []
    return [f"- {summary}{_final_ids(row.get('evidence_ids'))}"]


def _final_str_block(items: object) -> list[str]:
    """Bulleted lines for plain string lists ([] when none)."""
    out: list[str] = []
    for item in _as_list(items):
        text = _cell(item).strip()
        if text and text not in out:
            out.append(f"- {text}")
    return out


def _final_effect_line(item: object) -> str:
    """One first/second-order effect line with declared type + evidence refs."""
    row = _as_dict(item)
    text = _cell(row.get("text", row.get("summary", row.get("name")))).strip()
    return f"- {text}{_final_type_suffix(row)}{_final_ids(row.get('evidence_ids'))}" if text else ""


def _final_effect_block(items: object) -> list[str]:
    """Effect lines for one order ([] when none)."""
    out: list[str] = []
    for item in _as_list(items):
        line = _final_effect_line(item)
        if line:
            out.append(line)
    return out


_COVERAGE_VERDICT_KEYS = frozenset({"source_domain", "source_sufficiency", "useful_for_question", "complete"})


def _final_coverage_block(coverage: object) -> list[str]:
    """Per-source coverage lines: every surviving scope key under its own name."""
    cov = _as_dict(coverage)
    lines: list[str] = []
    for key in ("sec", "finra", "web"):
        section = _as_dict(cov.get(key))
        if not section:
            continue
        parts: list[str] = []
        for field in sorted(section):
            if field in _COVERAGE_VERDICT_KEYS or field in ("detail", "summary"):
                continue
            items = [str(v).strip() for v in _as_list(section.get(field)) if str(v).strip()]
            if items:
                parts.append(f"{field}: {', '.join(items)}")
        detail = _cell(section.get("detail") or section.get("summary")).strip()
        head = key.upper()
        if parts or detail:
            lines.append(f"- {head} — {'; '.join(parts)}{(' — ' + detail) if detail and parts else detail}")
        else:
            lines.append(f"- {head}")
    return lines


def _final_source_block(sources: object) -> list[str]:
    """One line per source row: domain + integrity class + document + evidence id."""
    lines: list[str] = []
    for item in _as_list(sources):
        row = _as_dict(item)
        evidence_id = _cell(row.get("evidence_id")).strip()
        domain = _cell(row.get("domain")).strip().upper() or "SOURCE"
        integrity = _cell(row.get("integrity_class") or row.get("integrity")).strip().upper()
        document = _cell(row.get("document") or row.get("source_name")).strip()
        label = f"{domain} [{integrity}]" if integrity else domain
        head = " ".join(part for part in (label, document) if part) or evidence_id or "source"
        lines.append(f"- {head} [{evidence_id}]" if evidence_id and head != evidence_id else f"- {head}")
    return lines


def _final_scope_line(scope: object) -> str:
    """Searched-source scope boundary line (always present)."""
    allowed = _as_list(_as_dict(scope).get("allowed_sources"))
    names = [s for s in (_cell(a).strip() for a in allowed) if s] or ["SEC"]
    joined = ", ".join(names)
    if len(names) == 1 and names[0].upper() == "SEC":
        return "Scope: SEC sources only."
    return f"Scope: {joined} sources only."


def _final_core_lines(result: dict[str, object]) -> list[str]:
    """Bottom line + consensus head lines."""
    summary = _cell(result.get("executive_summary", result.get("answer"))).strip()
    lines = [f"Bottom line: {summary}"] if summary else []
    consensus = _cell(result.get("consensus")).strip()
    if consensus and consensus.lower() != "none stated":
        lines.append(f"Consensus: {consensus}")
    return lines


def _final_section(out: list[str], header: str, lines: list[str]) -> None:
    """Append one headed section (no-op when empty)."""
    if lines:
        out.append(f"## {header}")
        out.extend(lines)


def render_final_result(result: dict[str, object], max_bytes: int = MAX_TOOL_MESSAGE_BYTES) -> str:
    """Substantive structured answer for a rich FinalResearchResult."""
    out = _final_core_lines(result)
    _final_section(
        out,
        "Base case",
        [f"- {_cell(result.get('base_case')).strip()}"] if _cell(result.get("base_case")).strip() else [],
    )
    _final_section(
        out,
        "Major direct exposures",
        [_final_channel_line(c) for c in _as_list(result.get("impact_channels")) if _final_channel_line(c)],
    )
    _final_section(out, "First-order effects", _final_effect_block(result.get("first_order_effects")))
    _final_section(out, "Second-order effects", _final_effect_block(result.get("second_order_effects")))
    _final_section(out, "Bull case", _final_side_block(result.get("bull_case")))
    _final_section(out, "Bear case", _final_side_block(result.get("bear_case")))
    _final_section(out, "Positioning", _final_str_block(result.get("positioning")))
    _final_section(out, "Catalysts", _final_str_block(result.get("catalysts")))
    _final_section(out, "Critical disagreements", _final_str_block(result.get("critical_disagreements")))
    _final_section(out, "Uncertainties", _final_str_block(result.get("uncertainties")))
    _final_section(out, "What would change the view", _final_str_block(result.get("what_would_change")))
    _final_section(out, "Limitations", _final_str_block(result.get("evidence_limitations")))
    _final_section(out, "Coverage", _final_coverage_block(result.get("coverage")))
    _final_section(out, "Sources", _final_source_block(result.get("sources")))
    _final_section(out, "Evidence refs", _final_claims_block(result.get("grounded_claims", result.get("claims"))))
    out.append(_final_scope_line(result.get("research_scope")))
    text = "\n\n".join(line for line in out if line.strip())
    return _truncate_bytes(text if text.strip() else _cell(result.get("answer")) or "No grounded findings.", max_bytes)


def _finalize_counts(final: dict[str, object]) -> tuple[int, int]:
    """(grounded claims, impact channels) of a persisted final_result."""
    claims = _as_list(final.get("claims")) or _as_list(final.get("grounded_claims"))
    return len(claims), len(_as_list(final.get("impact_channels")))


def _render_finalize_confirmation(result: dict[str, object], max_bytes: int) -> str:
    """Short research_finalize card: freeze id + counts, never the answer itself."""
    final = _as_dict(result.get("final_result"))
    claims, channels = _finalize_counts(final)
    freeze = _cell(result.get("freeze_id")) or _cell(final.get("freeze_id")) or "unknown"
    lines = [
        "Research finalized",
        f"Freeze: {freeze}",
        f"Claims: {claims} | impact channels: {channels}",
        "Answer delivered separately as this session's final response.",
    ]
    return _truncate_bytes("\n".join(lines), max_bytes)


def _dispatch_shape_render_b(result: dict[str, object], max_bytes: int) -> str:
    """Shape-dispatched renderers for table/briefing/SEC/text/final envelopes."""
    if "impact_channels" in result or "grounded_claims" in result or "executive_summary" in result:
        return render_final_result(result, max_bytes)
    text = _dispatch_shape_render_table(result, max_bytes)
    if text is not None:
        return text
    return _dispatch_shape_render_doc(result, max_bytes)


def _dispatch_tool_render(result: dict[str, object], max_bytes: int) -> str:
    """First matching tool renderer (final table/briefing fallback never misses)."""
    research = _dispatch_research_render(result, max_bytes)
    if research is not None:
        return research
    for render in (_dispatch_result_type_render_a, _dispatch_result_type_render_b, _dispatch_shape_render_a):
        text = render(result, max_bytes)
        if text is not None:
            return text
    return _dispatch_shape_render_b(result, max_bytes)


def _dispatch_research_render(result: dict[str, object], max_bytes: int) -> str | None:
    """Research evidence/search cards (scoped absence, paged hits), else None."""
    evidence = _evidence_record(result)
    if evidence is not None and evidence.get("claim_kind") == "absence_observation":
        return _render_absence_observation(evidence, max_bytes)
    return _research_search_card(result, max_bytes)


def render_tool_result(result: object, max_bytes: int = MAX_TOOL_MESSAGE_BYTES) -> str:
    """Render a tool result as compact text within the byte budget.

    Always returns a non-empty string of at most max_bytes UTF-8 bytes. The
    research_finalize card stays a short confirmation: the substantive answer
    is delivered once as the session's final response, never duplicated here.
    """
    result = result if isinstance(result, dict) else {"result": result}
    if "error" in result:
        return _render_error(result, max_bytes)
    if isinstance(result.get("final_result"), dict) and result["final_result"]:
        return _render_finalize_confirmation(result, max_bytes)
    content = result.get("content")
    if isinstance(content, str) and content.lstrip().startswith("Bottom line:"):
        return _truncate_bytes(content.strip(), max_bytes)
    text = _dispatch_tool_render(result, max_bytes)
    return text if _utf8_size(text) <= max_bytes else _minimal(result, max_bytes)


_ABSENCE_SCOPE_LABEL = "Absence observation — searched SEC scope only"


def _evidence_record(result: dict[str, object]) -> dict[str, object] | None:
    """Evidence record in a tool result (research_add_evidence, or research_read)."""
    for candidate in (result, _as_dict(result.get("record"))):
        if candidate.get("claim_kind") and (
            candidate.get("evidence_id") or candidate.get("claim_text") or candidate.get("content")
        ):
            return candidate
    return None


def _absence_scope_line(record: dict[str, object]) -> str:
    """Search-run scope of an absence observation ('' when unknown)."""
    prov = _as_dict(record.get("provenance"))
    search_id = _cell(prov.get("search_id")) or _cell(record.get("search_id"))
    query = _cell(prov.get("query")) or _cell(record.get("query"))
    parts = [
        part
        for part in (
            f"search {search_id}" if search_id else "",
            f"query {query!r}" if query else "",
        )
        if part
    ]
    return "Searched scope: " + " | ".join(parts) if parts else ""


def _render_absence_observation(record: dict[str, object], max_bytes: int) -> str:
    """Absence-observation card: scoped language, never a categorical denial.

    A search-derived absence is coverage state, not evidence: the card names the
    coverage artifact and says so instead of printing an evidence id.
    """
    finding = _cell(record.get("claim_text")) or _cell(record.get("content"))
    scoped = "No disclosure was located within the searched SEC scope"
    lines = [_ABSENCE_SCOPE_LABEL]
    scope = _absence_scope_line(record)
    if scope:
        lines.append(scope)
    lines.append(f"{scoped}: {finding}" if finding else scoped + ".")
    artifact_id = _cell(record.get("artifact_id"))
    if artifact_id:
        lines.append(f"Coverage artifact: {artifact_id} (search scope, never citable evidence)")
    evidence_id = _cell(record.get("evidence_id"))
    if evidence_id:
        lines.append(f"Evidence: {evidence_id}")
    return _truncate_bytes("\n".join(lines), max_bytes)


def _research_search_hit_line(hit: object) -> str:
    """One paged search-hit line (document identity + snippet, navigation only)."""
    row = _as_dict(hit)
    head = " ".join(
        part
        for part in (
            _cell(row.get("form")),
            _cell(row.get("filed_at")),
            _cell(row.get("accession")),
            _cell(row.get("document")),
        )
        if part
    )
    score = _cell(row.get("score"))
    if score:
        head = f"{head} (score {score})" if head else f"(score {score})"
    snippet = _cell(row.get("snippet"))
    if not head:
        return ""
    return f"- {head}: {snippet}" if snippet else f"- {head}"


def _research_search_head(result: dict[str, object], shown: list[object]) -> list[str]:
    """Retrieval-truth header of a paged search card (coverage, never the display bound)."""
    cov = _as_dict(result.get("coverage"))
    return [
        f"SEC search {_cell(result.get('search_id'))} — {len(shown)} hit(s) at offset {_cell(result.get('offset'))} of {_cell(result.get('total'))} persisted",
        f"query: {_cell(result.get('query')) or 'none'}",
        (
            "coverage: "
            + (_cell(cov.get("status")) or "unknown")
            + f"; pagination complete: {_briefing_flag_status(cov.get('pagination_complete'))}"
            + f"; source exhausted: {_briefing_flag_status(cov.get('source_exhausted'))}"
        ),
    ]


def _research_search_hit_lines(shown: list[object]) -> list[str]:
    """Rendered hit lines, dropping rows that carry no document identity."""
    return [line for line in (_research_search_hit_line(hit) for hit in shown) if line]


def _research_search_card(result: dict[str, object], max_bytes: int) -> str | None:
    """Paged research_read_search card (retrieval truth + hits), else None."""
    if not all(key in result for key in ("search_id", "total", "offset", "hits", "more", "coverage")):
        return None
    shown = _as_list(result.get("hits"))
    offset = result.get("offset")
    next_offset = offset + len(shown) if isinstance(offset, int) else None
    head = _research_search_head(result, shown)
    tail = [
        "Hits are navigation artifacts: open the filing and cite a raw passage before recording evidence.",
        (
            f"More hits: call research_read_search with offset={next_offset}"
            if result.get("more") and next_offset is not None
            else "More hits: none (every persisted hit is shown)"
        ),
    ]
    budget = max_bytes - _utf8_size("\n".join(head + tail)) - 2
    kept, omitted = _fit_lines(_research_search_hit_lines(shown), max(budget, 1))
    body = kept + ([f"{TRUNCATED_MARKER} (Omitted hits: {omitted})"] if omitted else [])
    return "\n".join(head + body + tail)


def _render_market_snapshot(result: dict[str, object], max_bytes: int) -> str:
    retrieved = f"Retrieved: {_cell(result.get('retrieved_at')) or 'unavailable'}"
    if result.get("retrieved_at_local"):
        retrieved += f" (local {result['retrieved_at_local']})"
    lines = [
        f"{result.get('ticker', '?')} market snapshot",
        f"Last: {_cell(result.get('last')) or 'unavailable'}",
        f"Bid: {_cell(result.get('bid')) or 'unavailable'}",
        f"Ask: {_cell(result.get('ask')) or 'unavailable'}",
        retrieved,
        f"Source: {_cell(result.get('source')) or 'robinhood_mcp'}",
    ]
    return _truncate_bytes("\n".join(lines), max_bytes)


def _option_chain_mid(row: dict[str, object]) -> object:
    """Mid from bid/ask when unstated (None only when both legs missing)."""
    if row.get("mid") is not None:
        return row.get("mid")
    bid, ask = row.get("bid"), row.get("ask")
    if bid is None or ask is None:
        return None
    return (float(str(bid)) + float(str(ask))) / 2


def _option_chain_spread(row: dict[str, object]) -> object:
    """Spread (ask minus bid) when unstated (None only when a leg missing)."""
    if row.get("spread") is not None:
        return row.get("spread")
    bid, ask = row.get("bid"), row.get("ask")
    if bid is None or ask is None:
        return None
    return float(str(ask)) - float(str(bid))


def _option_chain_cell(row: dict[str, object], field: str) -> str:
    """One derived option-chain cell (mid/spread derive, rest direct)."""
    if field == "mid":
        value = _option_chain_mid(row)
    elif field == "spread":
        value = _option_chain_spread(row)
    else:
        value = row.get(field)
    return _table_cell(value if value is not None else "unavailable")


_OPTION_CHAIN_FIELDS = (
    "expiration",
    "dte",
    "strike",
    "bid",
    "ask",
    "mark",
    "mid",
    "spread",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
)
_OPTION_CHAIN_LABELS = (
    "Expiration",
    "DTE",
    "Strike",
    "Bid",
    "Ask",
    "Mark",
    "Mid",
    "Spread",
    "IV",
    "Delta",
    "Gamma",
    "Theta",
    "Vega",
)


def _option_chain_row_line(row: dict[str, object]) -> str:
    """One option-chain table line from a contract mapping."""
    return "| " + " | ".join(_option_chain_cell(row, field) for field in _OPTION_CHAIN_FIELDS) + " |"


def _render_option_chain(result: dict[str, object], max_bytes: int) -> str:
    lines = [
        f"{result.get('ticker', '?')} {str(result.get('option_type', '')).upper()} OPTIONS",
        "| " + " | ".join(_OPTION_CHAIN_LABELS) + " |",
        "|" + "|".join("---" for _ in _OPTION_CHAIN_FIELDS) + "|",
    ]
    for row in _as_list(result.get("contracts")):
        if isinstance(row, dict):
            lines.append(_option_chain_row_line(row))
    lines.append(f"Returned: {result.get('returned', 0)} of {result.get('matched', 0)} matched")
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _render_option_analysis(result: dict[str, object], max_bytes: int) -> str:
    fields = (
        "ticker",
        "expiration",
        "dte",
        "strike",
        "bid",
        "ask",
        "mid",
        "spread",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "target_price",
        "target_pnl",
        "target_return_pct",
    )
    lines = ["Option contract analysis"]
    for field in fields:
        value = result.get(field)
        if value is not None:
            lines.append(f"{field}: {_cell(value)}")
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _render_option_comparison(result: dict[str, object], max_bytes: int) -> str:
    lines = [
        f"{result.get('ticker', '?')} option comparison",
        "| Contract | Expiration | Strike | Mid | Spread % | IV | Delta | Theta | Vega | Target P/L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _as_list(result.get("contracts")):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                _table_cell(row.get(field, "unavailable"))
                for field in (
                    "contract_id",
                    "expiration",
                    "strike",
                    "mid",
                    "spread_pct",
                    "implied_volatility",
                    "delta",
                    "theta",
                    "vega",
                    "target_pnl",
                )
            )
            + " |"
        )
    lines.append(
        f"Returned: {result.get('returned', 0)} of {result.get('matched', 0)} matched; ranking: {result.get('ranking', 'unknown')}"
    )
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _claim_source(item: dict[str, object]) -> str:
    """Publisher-or-domain source name, unknown when neither present."""
    return _cell(item.get("publisher")) or _cell(item.get("source_domain")) or "unknown"


def _claim_provenance(item: dict[str, object]) -> str:
    """Source-or-page URL provenance for one claim item."""
    return _cell(item.get("source_url")) or _cell(item.get("url"))


def _claim_identity_lines(item: dict[str, object], index: int) -> list[str]:
    """Claim number plus entity/resolution identity lines."""
    subject = _cell(item.get("subject_name")) or "?"
    entity = _cell(item.get("entity_id")) or "unresolved"
    return [
        f"Claim {index}",
        f"Entity: {subject} [{entity}]",
        f"Resolution: {_cell(item.get('subject_resolution')) or 'unresolved'}",
    ]


def _claim_source_lines(item: dict[str, object]) -> list[str]:
    """Claim type/source/timing lines from publisher metadata."""
    ctype = _cell(item.get("claim_type")) or "other"
    tier = _cell(item.get("source_tier")) or "unknown"
    integrity = _cell(item.get("integrity")) or "external"
    published = _cell(item.get("published_at")) or "unknown"
    retrieved = _cell(item.get("retrieved_at")) or "unknown"
    return [
        f"Type: {ctype}",
        f"Source: {_claim_source(item)} [{tier}/{integrity}]",
        f"Published: {published} | Retrieved: {retrieved}",
    ]


def _claim_text_lines(item: dict[str, object]) -> list[str]:
    """Claim/evidence/provenance text lines."""
    text = _cell(item.get("text")) or _cell(item.get("claim"))
    return [
        f"Claim: {text}",
        f"Evidence: {_cell(item.get('evidence_summary'))}",
        f"Provenance: {_claim_provenance(item)}",
    ]


def _web_search_claim_block(item: dict[str, object], index: int) -> list[str]:
    """Typed EvidenceClaim block; raw title/highlight never rendered here."""
    return _claim_identity_lines(item, index) + _claim_source_lines(item) + _claim_text_lines(item)


def _web_search_claim_optionals(item: dict[str, object]) -> list[str]:
    """Reported-ticker/object extras present only on claim blocks."""
    extra = []
    reported = _cell(item.get("reported_ticker"))
    if reported:
        extra.append(f"Reported as: {reported}")
    obj = _cell(item.get("object_name"))
    if obj:
        extra.append(f"Object: {obj}")
    return extra


def _web_search_claim_lines(item: dict[str, object], index: int) -> list[str]:
    """Claim block with its optional extras spliced after Resolution."""
    block = _web_search_claim_block(item, index)
    return block[:3] + _web_search_claim_optionals(item) + block[3:]


def _web_search_hit_meta(item: dict[str, object]) -> str:
    """Parenthesized published/retrieved suffix, empty when neither."""
    meta = []
    if item.get("published_at"):
        meta.append(f"published {item['published_at']}")
    if item.get("retrieved_at"):
        meta.append(f"retrieved {item['retrieved_at']}")
    return f" ({'; '.join(meta)})" if meta else ""


def _web_search_hit_lines(item: dict[str, object], index: int) -> list[str]:
    """Untyped search-hit block: title line plus url/highlight extras."""
    title = _cell(item.get("title")) or item.get("url") or f"Result {index}"
    out = [f"{index}. {title} — {item.get('source_domain') or 'unknown'}{_web_search_hit_meta(item)}"]
    if item.get("url"):
        out.append(f"   {item['url']}")
    if item.get("highlight"):
        out.append(f"   {item['highlight']}")
    return out


def _web_search_item_lines(item: object, index: int) -> list[str]:
    """One evidence item as claim or hit lines ([] for mistyped items)."""
    if not isinstance(item, dict):
        return []
    if item.get("claim") or item.get("text"):
        return _web_search_claim_lines(item, index)
    return _web_search_hit_lines(item, index)


def _render_web_search(result: dict[str, object], max_bytes: int) -> str:
    lines = [f"CURRENT EXTERNAL EVIDENCE (search: {result.get('query') or '?'})"]
    evidence = _as_list(result.get("evidence"))
    if not evidence:
        lines.append("No evidence returned.")
    for index, item in enumerate(evidence, start=1):
        lines.extend(_web_search_item_lines(item, index))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _portfolio_created_line(result: dict[str, object]) -> str:
    """Created line with optional local-time suffix."""
    created = f"Created: {_cell(result.get('created_at')) or 'unavailable'}"
    if result.get("created_at_local"):
        created += f" (local {result['created_at_local']})"
    return created


def _portfolio_header_lines(result: dict[str, object]) -> list[str]:
    """Header plus optional concentration line, before the Positions block."""
    lines = [
        f"Portfolio snapshot — {_cell(result.get('broker')) or 'robinhood'}",
        _portfolio_created_line(result),
        f"Total value: {_cell(result.get('total_value')) or 'unavailable'}",
        f"Cash: {_cell(result.get('cash')) or 'unavailable'}",
        f"Invested: {_cell(result.get('invested_value')) or 'unavailable'}",
        f"Positions: {result.get('position_count', 0)} ({result.get('priced_position_count', 0)} priced, {result.get('unresolved_position_count', 0)} unresolved)",
    ]
    if result.get("concentration"):
        lines.append(f"Concentration: {_cell(result['concentration'])}")
    return lines


def _portfolio_position_block(result: dict[str, object]) -> list[str]:
    """Rendered position rows plus omitted-count tail."""
    lines = ["Positions:"]
    for row in _as_list(result.get("positions")):
        if isinstance(row, dict):
            lines.append(_portfolio_position_line(row))
    if result.get("omitted_count"):
        lines.append(f"... {result['omitted_count']} smaller positions omitted")
    return lines


def _portfolio_unresolved_line(result: dict[str, object]) -> str:
    """Unresolved-securities line, empty when none listed."""
    unresolved = _as_list(result.get("unresolved"))
    if not unresolved:
        return ""
    return "Unresolved securities: " + ", ".join(_cell(t) for t in unresolved)


def _portfolio_freshness_snapshot(result: dict[str, object], freshness: dict[str, object]) -> str:
    """Snapshot-created freshness part, empty when no snapshot date."""
    created = freshness.get("snapshot_created_at") or result.get("created_at")
    local = freshness.get("snapshot_created_at_local") or result.get("created_at_local")
    if not created:
        return ""
    return f"snapshot {created}" + (f" (local {local})" if local else "")


def _portfolio_freshness_line(result: dict[str, object]) -> str:
    """Research-freshness line, empty when no freshness parts present."""
    freshness = _as_dict(result.get("freshness"))
    parts = []
    snapshot = _portfolio_freshness_snapshot(result, freshness)
    if snapshot:
        parts.append(snapshot)
    if freshness.get("sec_latest_filed_at"):
        parts.append(f"SEC latest filing {freshness['sec_latest_filed_at']}")
    if freshness.get("finra_settlement_date"):
        parts.append(f"FINRA settlement {freshness['finra_settlement_date']}")
    if not parts:
        return ""
    return "Research freshness: " + "; ".join(parts)


def _render_portfolio_snapshot(result: dict[str, object], max_bytes: int) -> str:
    lines = _portfolio_header_lines(result)
    lines.extend(_portfolio_position_block(result))
    unresolved_line = _portfolio_unresolved_line(result)
    if unresolved_line:
        lines.append(unresolved_line)
    freshness_line = _portfolio_freshness_line(result)
    if freshness_line:
        lines.append(freshness_line)
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def issue_to_prose(issue: EvaluationIssue) -> str:
    """Render an evaluation issue to its legacy prose form."""
    if issue.code == "cash_unavailable":
        return "minimum_cash: cash unavailable"
    if issue.code == "total_value_unavailable":
        return "minimum_cash: total value unavailable"
    if issue.code == "total_value_zero":
        return "minimum_cash: total value is zero"
    if issue.code == "position_weight_unavailable":
        return f"{issue.metric}: {issue.ticker} (no weight)"
    if issue.code == "unknown_sector_exposure":
        return f"sector_exposure: {issue.target} (unknown exposure)"
    return f"{issue.metric}: {issue.code}"


def _mandate_value_text(value: object, metric: str, unit: str) -> str:
    """Breach value prose: dollars, percent, or cell fallback."""
    if value is None:
        return "unavailable"
    if metric == "prohibited_assets":
        return _cell(value)
    if not isinstance(value, (int, float, str, Decimal)):
        return _cell(value)
    if unit == "dollars":
        return f"${float(str(value)):,.2f}"
    try:
        return f"{float(str(value)) * 100:.1f}%"
    except TypeError, ValueError:
        return _cell(value)


def _mandate_snapshot_line(result: dict[str, object]) -> str:
    """Snapshot-created line with optional local-time suffix."""
    line = f"Snapshot created: {_cell(result.get('snapshot_created_at')) or 'unavailable'}"
    if result.get("snapshot_created_at_local"):
        line += f" (local {result['snapshot_created_at_local']})"
    return line


def _mandate_breach_line(breach: dict[str, object]) -> str:
    """One breach line with actual/limit plus excess/note extras."""
    metric = _cell(breach.get("metric")) or "?"
    unit = str(breach.get("unit") or "ratio")
    target = f" {breach['target']}" if breach.get("target") else ""
    severity = _cell(breach.get("severity")) or "warning"
    line = (
        f"  [{severity}] {metric}{target}: "
        f"actual {_mandate_value_text(breach.get('actual'), metric, unit)}, "
        f"limit {_mandate_value_text(breach.get('limit'), metric, unit)}"
    )
    if breach.get("excess") is not None:
        line += f", excess {_mandate_value_text(breach.get('excess'), metric, unit)}"
    if breach.get("note"):
        line += f" [{_cell(breach['note'])}]"
    return line


def _mandate_breach_block(result: dict[str, object]) -> list[str]:
    """Breaches section (header plus one line per breach mapping)."""
    breaches = _as_list(result.get("breaches"))
    if not breaches:
        return ["No breaches."]
    lines = ["Breaches:"]
    for breach in breaches:
        if isinstance(breach, dict):
            lines.append(_mandate_breach_line(breach))
    return lines


def _mandate_exposure_line(result: dict[str, object]) -> str:
    """Sector-exposures line, empty when no exposures mapped."""
    exposures = _as_dict(result.get("sector_exposures"))
    if not exposures:
        return ""
    return "Sector exposures: " + ", ".join(_mandate_exposure_part(s, w) for s, w in exposures.items())


def _mandate_exposure_part(sector: object, weight: object) -> str:
    """One sector exposure part (percent when numeric, raw otherwise)."""
    try:
        return f"{_cell(sector)} {float(str(weight)) * 100:.1f}%"
    except TypeError, ValueError:
        return f"{_cell(sector)} {_cell(weight)}"


def _mandate_issue_block(result: dict[str, object]) -> list[str]:
    """Not-evaluable issues section ([] when no issue mappings)."""
    issues = [i for i in _as_list(result.get("issues")) if isinstance(i, dict)]
    if not issues:
        return []
    lines = ["Not evaluable:"]
    for issue in issues:
        assert isinstance(issue, dict)
        lines.append(f"  - {_cell(issue_to_prose(EvaluationIssue(**issue)))}")
    return lines


def _render_mandate_evaluation(result: dict[str, object], max_bytes: int) -> str:
    """Compact mandate report: breaches, sector exposures, issues."""
    lines = ["Mandate evaluation", _mandate_snapshot_line(result)]
    lines.extend(_mandate_breach_block(result))
    exposure_line = _mandate_exposure_line(result)
    if exposure_line:
        lines.append(exposure_line)
    lines.extend(_mandate_issue_block(result))
    lines.append("Source: " + str(result.get("source", "mandate")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _position_quantity_part(row: dict[str, object]) -> str:
    """Quantity-times-price part, empty when no quantity present."""
    quantity = _cell(row.get("quantity"))
    if not quantity:
        return ""
    price = _cell(row.get("market_price")) or "unavailable"
    return f"{quantity} x {price}"


def _position_value_gain_parts(row: dict[str, object]) -> list[str]:
    """Value/weight/gain parts in order (absent fields skipped)."""
    parts = []
    value = _cell(row.get("market_value"))
    if value:
        parts.append(f"value {value}")
    weight = _position_weight_part(row)
    if weight:
        parts.append(weight)
    gain = _cell(row.get("unrealized_gain"))
    if gain:
        parts.append(f"gain {gain}")
    return parts


def _position_core_parts(row: dict[str, object]) -> list[str]:
    """Ticker/quantity/value/weight/gain parts (never SEC/FINRA extras)."""
    parts = [f"- {_cell(row.get('ticker')) or '?'}"]
    quantity = _position_quantity_part(row)
    if quantity:
        parts.append(quantity)
    parts.extend(_position_value_gain_parts(row))
    if not row.get("resolved", True):
        parts.append("[UNRESOLVED]")
    return parts


def _position_weight_part(row: dict[str, object]) -> str:
    """Weight part (percent when numeric, raw otherwise, empty when absent)."""
    weight = _cell(row.get("portfolio_weight"))
    if not weight:
        return ""
    try:
        weight = f"{float(weight) * 100:.2f}%"
    except TypeError, ValueError:
        pass
    return f"weight {weight}"


def _position_sec_part(row: dict[str, object]) -> str:
    """SEC fact extras, empty when no valued concepts present."""
    sec = _as_dict(row.get("sec"))
    sec_parts = []
    for concept in SEC_CONCEPTS:
        fact = sec.get(concept)
        if isinstance(fact, dict) and _cell(fact.get("value")):
            label = concept[:3].lower()
            sec_parts.append(f"{label} {_cell(fact['value'])}")
    if not sec_parts:
        return ""
    return "SEC " + " ".join(sec_parts)


def _position_finra_part(row: dict[str, object]) -> str:
    """FINRA extras, empty when no short-position fields present."""
    finra = _as_dict(row.get("finra"))
    finra_parts = []
    if _cell(finra.get("short_position")):
        finra_parts.append(f"short {_cell(finra['short_position'])}")
    if _cell(finra.get("change_pct")):
        finra_parts.append(f"d {_cell(finra['change_pct'])}")
    if _cell(finra.get("days_to_cover")):
        finra_parts.append(f"dtc {_cell(finra['days_to_cover'])}")
    if not finra_parts:
        return ""
    return "FINRA " + " ".join(finra_parts)


def _portfolio_position_line(row: dict[str, object]) -> str:
    parts = _position_core_parts(row)
    sec_part = _position_sec_part(row)
    if sec_part:
        parts.append(sec_part)
    finra_part = _position_finra_part(row)
    if finra_part:
        parts.append(finra_part)
    return "  ".join(parts)


def _scan_spec_line(spec: dict[str, object]) -> str:
    """One filter-spec line (truncated to 200 chars like the legacy row)."""
    name = _cell(spec.get("display_name") or spec.get("filter_type") or spec.get("name") or "?")
    filter_type = _cell(spec.get("filter_type")) or ""
    predicates = _cell(spec.get("supported_predicates")) or ""
    return f"- {name}{f' ({filter_type})' if filter_type else ''}: {predicates}"[:200]


def _scan_specs_tail(result: dict[str, object]) -> str:
    """Omitted-filter-types tail, empty when nothing omitted."""
    if not result.get("omitted_count"):
        return ""
    return f"... {result['omitted_count']} more filter types omitted"


def _render_scan_specs(result: dict[str, object], max_bytes: int) -> str:
    specs = _as_list(result.get("specs"))
    lines = [
        f"Scanner filter specs ({result.get('count', len(specs))} filter types)",
        "Live data from Robinhood MCP; call this before constructing scan filters.",
    ]
    for spec in specs:
        if isinstance(spec, dict):
            lines.append(_scan_spec_line(spec))
    tail = _scan_specs_tail(result)
    if tail:
        lines.append(tail)
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _scan_list_counts(scan: dict[str, object]) -> list[str]:
    """Filter/column count extras for one saved scanner."""
    parts = []
    if isinstance(scan.get("filters"), list):
        parts.append(f"{len(scan['filters'])} filters")
    if isinstance(scan.get("columns"), list):
        parts.append(f"{len(scan['columns'])} columns")
    return parts


def _scan_list_line(scan: dict[str, object]) -> str:
    """One saved-scanner line with managed/filters/columns extras."""
    scan_id = _cell(scan.get("scan_id") or scan.get("id") or "?")
    title = _cell(scan.get("title") or scan.get("name") or "untitled")
    parts = [f"- {scan_id}: {title}"]
    if scan.get("cortex_managed"):
        parts.append("[Cortex-managed, read-only]")
    parts.extend(_scan_list_counts(scan))
    return "  ".join(parts)


def _scan_list_tail(result: dict[str, object]) -> str:
    """Omitted-scans tail, empty when nothing omitted."""
    if not result.get("omitted_count"):
        return ""
    return f"... {result['omitted_count']} more scans omitted"


def _render_scan_list(result: dict[str, object], max_bytes: int) -> str:
    scans = _as_list(result.get("scans"))
    lines = [f"Saved scanners ({result.get('count', len(scans))})"]
    for scan in scans:
        if isinstance(scan, dict):
            lines.append(_scan_list_line(scan))
    tail = _scan_list_tail(result)
    if tail:
        lines.append(tail)
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _scan_result_values(row: dict[str, object]) -> list[str]:
    """Priced-field values present on one scan-result row."""
    return [
        _cell(row[key])
        for key in ("last", "price", "market_cap", "volume", "change", "change_percent")
        if _cell(row.get(key))
    ]


def _scan_result_line(row: dict[str, object]) -> str:
    """One scan-result row line (ticker plus priced values)."""
    ticker = _cell(row.get("ticker") or row.get("symbol") or "?")
    values = _scan_result_values(row)
    return f"- {ticker}" + ("  " + "  ".join(values) if values else "")


def _scan_result_tail(result: dict[str, object]) -> list[str]:
    """Omitted/sort tail lines ([] when neither present)."""
    tail = []
    if result.get("omitted"):
        tail.append(f"... {result['omitted']} more matches omitted")
    if result.get("sort"):
        tail.append(f"Sort: {_cell(result['sort'])}")
    return tail


def _render_scan_results(result: dict[str, object], max_bytes: int) -> str:
    rows = _as_list(result.get("rows"))
    title = _cell(result.get("title")) or "Scan"
    lines = [
        f"{title} — live results ({result.get('total', len(rows))} matches)",
        "Live market data from Robinhood MCP, evaluated at request time.",
    ]
    for row in rows:
        if isinstance(row, dict):
            lines.append(_scan_result_line(row))
    lines.extend(_scan_result_tail(result))
    lines.append("Source: " + str(result.get("source", "robinhood_mcp")))
    return _truncate_bytes("\n".join(lines), max_bytes)


_SHORT_INTEREST_FIELDS = (
    "rank",
    "ticker",
    "short_interest_percent",
    "short_shares",
    "shares_outstanding",
    "sec_shares_as_of",
    "sec_filed_at",
)
_SHORT_INTEREST_LABELS = (
    "Rank",
    "Ticker",
    "Short %",
    "Short shares",
    "Shares outstanding",
    "SEC shares as of",
    "SEC filed",
)


def _short_interest_cell(entry: dict[str, object], field: str) -> str:
    """One leaderboard cell (percent/shares formatted, raw otherwise)."""
    value = entry.get(field, "")
    if value == "":
        return _table_cell(value)
    if field == "short_interest_percent":
        return _table_cell(f"{float(str(value)):.2f}%")
    if field in ("short_shares", "shares_outstanding"):
        return _table_cell(f"{float(str(value)):,.0f}")
    return _table_cell(value)


def _short_interest_row_line(entry: dict[str, object]) -> str:
    """One leaderboard table line from an entry mapping."""
    return "| " + " | ".join(_short_interest_cell(entry, f) for f in _SHORT_INTEREST_FIELDS) + " |"


def _short_interest_header(result: dict[str, object]) -> list[str]:
    """Title plus table header plus stale banner when stale."""
    lines = [
        "Short interest leaderboard — FINRA settlement " + str(result["settlement_date"]),
        "| " + " | ".join(_SHORT_INTEREST_LABELS) + " |",
        "|" + "|".join("---" for _ in _SHORT_INTEREST_FIELDS) + "|",
    ]
    stale_banner = _datapoints_stale_banner(result)
    if stale_banner:
        lines.append(stale_banner)
    return lines


def _short_interest_footer(result: dict[str, object]) -> list[str]:
    """Source/metric/coverage/as-of/environment footer lines."""
    coverage = _as_dict(result.get("coverage"))
    lines = [
        "Source: " + str(result.get("source", "FINRA + SEC EDGAR")),
        "Metric: " + str(result.get("metric", "")),
        "Coverage: "
        + f"{coverage.get('eligible_rows', 0)} eligible of {coverage.get('finra_rows', 0)} FINRA rows; exclusions {coverage.get('exclusions', {})}",
    ]
    if result.get("as_of_date"):
        lines.append(
            "As of: "
            + str(result["as_of_date"])
            + " (freshness: "
            + str(result.get("data_freshness") or "unknown")
            + ")"
        )
    lines.append("Environment: " + str(result.get("environment", "unknown")))
    return lines


def _render_short_interest_leaderboard(result: dict[str, object], max_bytes: int) -> str:
    lines = _short_interest_header(result)
    for entry in _as_list(result.get("entries")):
        if isinstance(entry, dict):
            lines.append(_short_interest_row_line(entry))
    lines.extend(_short_interest_footer(result))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _render_analyst_estimates(result: dict[str, object], max_bytes: int) -> str:
    ticker = result.get("ticker", "?")
    quote = _as_dict(result.get("quote"))
    targets = _as_dict(result.get("price_targets"))
    valuation = _as_dict(result.get("valuation"))
    lines = [
        f"{ticker} analyst consensus (as of {result.get('as_of', '?')})",
        f"Source: {result.get('source', 'Yahoo Finance')}",
    ]
    price = quote.get("price")
    lines.append(
        f"Last: {price if price is not None else 'unavailable'}"
        f"  |  Market cap: {_cell(result.get('market_cap'))}"
        f"  |  Shares out: {_cell(result.get('shares_outstanding'))}"
    )
    lines.append(
        "Targets (12-mo): "
        f"median {_cell(targets.get('median'))}"
        f" | mean {_cell(targets.get('mean'))}"
        f" | high {_cell(targets.get('high'))}"
        f" | low {_cell(targets.get('low'))}"
        f" | {targets.get('num_analysts') or '?'} analysts"
    )
    lines.append(
        f"Rating: {targets.get('recommendation')}"
        f" (mean {_cell(targets.get('recommendation_mean'))})"
        f"  |  P/E trailing {_cell(valuation.get('trailing_pe'))}"
        f" / forward {_cell(valuation.get('forward_pe'))}"
    )
    for row in _as_list(result.get("forward_estimates")):
        if not isinstance(row, dict):
            continue
        eps = row.get("eps_avg")
        rev = row.get("revenue_avg")
        lines.append(
            f"- {row.get('period', '?')} (ends {row.get('period_end_date', '?')}): "
            f"EPS est {_cell(eps)}"
            f" ({_cell(row.get('eps_growth_pct'))}% YoY, n={row.get('eps_analysts')})"
            f" | Revenue est {_cell(rev)}"
            f" ({_cell(row.get('revenue_growth_pct'))}% YoY, n={row.get('revenue_analysts')})"
        )
        rev_trend = row.get("eps_revision") or {}
        if rev_trend.get("current") is not None:
            lines.append(
                f"    EPS revision: now {_cell(rev_trend.get('current'))}"
                f" | 7d ago {_cell(rev_trend.get('days7_ago'))}"
                f" | 30d ago {_cell(rev_trend.get('days30_ago'))}"
                f" | 60d ago {_cell(rev_trend.get('days60_ago'))}"
            )
    return _truncate_bytes("\n".join(lines), max_bytes)


def _render_sp500_weight(result: dict[str, object], max_bytes: int) -> str:
    ticker = result.get("ticker", "?")
    lines = [
        f"{ticker} S&P 500 index weight (as of {result.get('as_of', '?')})",
        f"Source: {result.get('source', 'Slickcharts')}",
        (
            f"Rank: {_cell(result.get('rank'))}"
            f"  |  Company: {result.get('company', '?')}"
            f"  |  Weight: {_cell(result.get('weight_pct'))}% of index market cap"
        ),
    ]
    if result.get("note"):
        lines.append("Note: " + str(result["note"]))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _obligation_headline(row: dict[str, object]) -> str:
    """Headline line for one obligation row (lifecycle-aware, amount-kept)."""
    amount = row.get("amount_billions")
    amount_s = f"${amount}B" if amount is not None else "(schedule only)"
    matched = " | revenue-matched" if row.get("revenue_matched") else ""
    status = f" | status: {row.get('status', '?')}"
    trigger = f" | trigger: {row['trigger']}" if row.get("trigger") else ""
    lifecycle = row.get("lifecycle_status")
    if lifecycle == "amended":
        return (
            f"- {row.get('type', '?')}: {amount_s}"
            f" | lifecycle: amended — amount unresolved, last disclosed {amount_s} retained"
            f" (included in current exposure)"
            f"{status}{matched}{trigger}"
        )
    if lifecycle in ("terminated", "unknown"):
        label = "terminated" if lifecycle == "terminated" else "currently unknown"
        return (
            f"- {row.get('type', '?')}: {amount_s}"
            f" | lifecycle: {label} (excluded from current exposure)"
            f"{status}{matched}{trigger}"
        )
    return f"- {row.get('type', '?')}: {amount_s} | certainty: {row.get('certainty', '?')}{status}{matched}{trigger}"


def _obligation_filed_line(row: dict[str, object]) -> str:
    """Provenance line for one obligation row."""
    return (
        f"    filed {row.get('filed') or '?'} | acc {row.get('accession') or 'n/a'}"
        f" | {_table_cell(row.get('excerpt') or '')}"
    )


def _schedule_year_line(year: object) -> str:
    """One per-year schedule line from a fiscal-year mapping."""
    y = year if isinstance(year, dict) else {}
    return f"    FY{y.get('fiscal_year')}: ${y.get('amount_billions')}B"


def _obligation_schedule_lines(row: dict[str, object]) -> list[str]:
    """Front-loaded plus per-year schedule lines ([] when none)."""
    lines: list[str] = []
    horizon = row.get("payment_horizon")
    horizon_map: dict[str, object] = horizon if isinstance(horizon, dict) else {}
    if horizon_map.get("paid_in_remainder_billions"):
        lines.append(
            f"    front-loaded: ${horizon_map['paid_in_remainder_billions']}B"
            f" paid in remainder of FY{horizon_map.get('paid_in_remainder_of_fy')},"
            f" ${horizon_map.get('paid_after_remainder_billions')}B after"
        )
    sched = horizon_map.get("schedule")
    if isinstance(sched, list):
        for y in sched:
            lines.append(_schedule_year_line(y))
    row_sched = row.get("schedule")
    if isinstance(row_sched, list):
        for y in row_sched:
            lines.append(_schedule_year_line(y))
    return lines


def _obligation_row_lines(row: dict[str, object]) -> list[str]:
    """All lines for one ledger obligation row ([] for components)."""
    if not isinstance(row, dict) or row.get("schedule_component"):
        return []
    return [_obligation_headline(row), _obligation_filed_line(row)] + _obligation_schedule_lines(row)


def _obligation_exposure_line(exp: dict[str, object]) -> str:
    """One unquantified-exposure line."""
    return (
        f"- {exp.get('type', '?')}: unquantified"
        f" | trigger: {exp.get('trigger', '?')} | filed {exp.get('filed', '?')}"
        f" | {exp.get('reason', 'excluded from quantified totals')}"
        f" | {_table_cell(exp.get('excerpt') or '')}"
    )


def _obligation_exposure_block(result: dict[str, object]) -> list[str]:
    """Unquantified-exposures section ([] when none disclosed)."""
    exposures = [e for e in _as_list(result.get("unquantified_exposures")) if isinstance(e, dict)]
    if not exposures:
        return []
    return ["Unquantified exposures (excluded from totals):"] + [_obligation_exposure_line(e) for e in exposures]


def _obligation_capital_block(result: dict[str, object]) -> list[str]:
    """Capital-allocation section ([] when none disclosed)."""
    capital = [e for e in _as_list(result.get("capital_allocation")) if isinstance(e, dict)]
    if not capital:
        return []
    lines = ["Capital allocation (discretionary, not obligations):"]
    for entry in capital:
        lines.append(
            f"- {entry.get('type', '?')}: discretionary, not an obligation"
            f" | filed {entry.get('filed', '?')}"
            f" | {_table_cell(entry.get('excerpt') or '')}"
        )
    return lines


def _render_obligations(result: dict[str, object], max_bytes: int) -> str:
    ticker = result.get("ticker", "?")
    lines = [
        f"{ticker} contractual obligations & commitments (filed {result.get('filed', '?')})",
        f"Source: {result.get('source', 'SEC EDGAR notes')}",
    ]
    for row in _as_list(result.get("obligations")):
        if isinstance(row, dict):
            lines.extend(_obligation_row_lines(row))
    lines.extend(_obligation_exposure_block(result))
    lines.extend(_obligation_capital_block(result))
    if result.get("note"):
        lines.append("Note: " + str(result["note"]))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _valuation_fy_label(year_cur: object, year_next: object, cur: bool) -> str:
    """FY label from fiscal-year metadata (fallback to current/next prose)."""
    y = year_cur if cur else year_next
    if isinstance(y, str) and y.strip():
        return f"FY{y.strip()}"
    return "(current FY)" if cur else "(next FY)"


def _valuation_pe(value: object) -> str:
    """P/E prose (multiple when priced, gap message otherwise)."""
    return f"{value}x" if value is not None else "unavailable (no live price)"


def _valuation_header_lines(result: dict[str, object], ob: dict[str, object]) -> list[str]:
    """Ticker/valuation header plus trailing-P/E and obligation-drag lines."""
    ticker = result.get("ticker", "?")
    price = _as_dict(result.get("price"))
    last = price.get("last")
    price_s = f"${last}" if last is not None else "unavailable (live-quote gap)"
    return [
        f"{ticker} valuation (live price {price_s}, as of {result.get('as_of', '?')})",
        f"Source: {result.get('source', '')}",
        f"Trailing P/E (GAAP TTM EPS ${result.get('ttm_gaap_eps')}): {_valuation_pe(result.get('trailing_pe'))}",
        (
            f"Obligation drag per share: contractual ${ob.get('drag_per_share_contractual')}"
            f" | contingent ${ob.get('drag_per_share_contingent')}"
            f" | default-triggered ${ob.get('drag_per_share_default_triggered')}"
            f" (annual: ${ob.get('contractual_annual_billions')}B contractual,"
            f" ${ob.get('contingent_annual_billions')}B contingent,"
            f" ${ob.get('default_triggered_annual_billions')}B default-triggered)"
        ),
    ]


def _valuation_price_gap_line(result: dict[str, object]) -> str:
    """Live-quote-gap line, empty when no gap reported."""
    if not result.get("price_gap"):
        return ""
    return "Live-quote gap: " + str(result["price_gap"])


def _valuation_margin_text(margin: object) -> str:
    """Gross-margin prose (percent when numeric, unavailable otherwise)."""
    if isinstance(margin, (int, float)):
        return f"~{margin:.0%} gross margin"
    return "gross margin unavailable"


def _valuation_revenue_matched_line(ob: dict[str, object]) -> str:
    """Revenue-matched supply-commitment line, empty when none disclosed."""
    if not ob.get("revenue_matched_annual_billions"):
        return ""
    margin_s = _valuation_margin_text(ob.get("revenue_matched_gross_margin"))
    if ob.get("revenue_matched_margin_source") != "company_facts":
        margin_s += f" ({ob.get('revenue_matched_margin_source') or 'company margin undisclosed'})"
    return (
        f"Revenue-matched (supply) commitments: "
        f"${ob.get('revenue_matched_annual_billions')}B/yr"
        f" | NOT an EPS drag (inventory sold at {margin_s})"
        f" | implied revenue coverage ~${ob.get('revenue_matched_implied_revenue_billions')}B/yr"
    )


def _valuation_eps_line(label: str, row: object, scenario: bool = False) -> str:
    """One consensus EPS/P/E line (empty when the row mapping is absent)."""
    if not isinstance(row, dict) or not row:
        return ""
    pe = _valuation_pe(row.get("pe"))
    eps = row.get("eps")
    if scenario:
        return f"- {label}: EPS ${eps} | P/E {pe} | after obligations"
    return f"- {label}: EPS ${eps} | P/E {pe}"


def _valuation_eps_present(value: object) -> bool:
    """True when an EPS-scenario mapping discloses an after-obligation EPS."""
    return isinstance(value, dict) and bool(value.get("eps_after_all_obligations") is not None)


def _valuation_adjusted_line(label: str, adj: object) -> str:
    """Adjusted contractual EPS line, empty when undisclosed."""
    if not isinstance(adj, dict) or adj.get("eps_after_contractual") is None:
        return ""
    return (
        f"- Adjusted {label} (contractual incl.): EPS ${adj['eps_after_contractual']}"
        f" | P/E {_valuation_pe(adj.get('pe_after_contractual'))}"
        f" | drag ${adj.get('obligation_drag_per_share')}/sh"
    )


def _valuation_adjusted_next_line(label: str, adj: object) -> str:
    """Adjusted contractual EPS line for the next FY, empty when undisclosed."""
    if not isinstance(adj, dict) or adj.get("eps_after_contractual") is None:
        return ""
    return (
        f"- Adjusted {label} (contractual incl.): EPS ${adj['eps_after_contractual']}"
        f" | P/E {_valuation_pe(adj.get('pe_after_contractual'))}"
    )


def _valuation_scenario_line(label: str, scn: object, tag: str) -> str:
    """Contingent-scenario EPS line for one tag, empty when undisclosed."""
    if not _valuation_eps_present(scn):
        return ""
    assert isinstance(scn, dict)
    if tag == "default":
        return (
            f"- Scenario {label} (counterparty default): EPS ${scn['eps_after_all_obligations']}"
            f" | P/E {_valuation_pe(scn.get('pe_after_all_obligations'))}"
            f" | drag ${scn.get('contingent_drag_per_share')}/sh"
        )
    if tag == "next_default":
        return (
            f"- Scenario {label} (counterparty default): EPS ${scn['eps_after_all_obligations']}"
            f" | P/E {_valuation_pe(scn.get('pe_after_all_obligations'))}"
        )
    if tag == "next":
        return (
            f"- Scenario {label} (+contingent, no default): EPS ${scn['eps_after_all_obligations']}"
            f" | P/E {_valuation_pe(scn.get('pe_after_all_obligations'))}"
            f" | drag ${scn.get('contingent_drag_per_share')}/sh contingent"
        )
    return (
        f"- Scenario {label} (+contingent, no default): EPS ${scn['eps_after_all_obligations']}"
        f" | P/E {_valuation_pe(scn.get('pe_after_all_obligations'))}"
        f" | drag ${scn.get('contingent_drag_per_share')}/sh"
    )


def _valuation_worst_line(label: str, worst: object) -> str:
    """Worst-case stranded-supply line, empty when undisclosed."""
    if not _valuation_eps_present(worst):
        return ""
    assert isinstance(worst, dict)
    return (
        f"- WORST CASE {label} (all obligations incl. supply stranded):"
        f" EPS ${worst['eps_after_all_obligations']}"
        f" | P/E {_valuation_pe(worst.get('pe_after_all_obligations'))}"
    )


def _valuation_current_fy_lines(fe: dict[str, object], fy_cur: str) -> list[str]:
    """Consensus/adjusted/scenario/worst lines for the current fiscal year."""
    lines = []
    line = _valuation_eps_line(f"Consensus {fy_cur}", fe.get("consensus"))
    if line:
        lines.append(line)
    line = _valuation_adjusted_line(fy_cur, fe.get("adjusted"))
    if line:
        lines.append(line)
    line = _valuation_scenario_line(fy_cur, fe.get("scenario"), "current")
    if line:
        lines.append(line)
    line = _valuation_scenario_line(fy_cur, fe.get("scenario_with_defaults"), "default")
    if line:
        lines.append(line)
    line = _valuation_worst_line(fy_cur, fe.get("worst_case"))
    if line:
        lines.append(line)
    return lines


def _valuation_next_fy_lines(fe: dict[str, object], fy_next: str) -> list[str]:
    """Consensus/adjusted/scenario/worst lines for the next fiscal year."""
    lines = []
    line = _valuation_eps_line(f"Consensus {fy_next}", fe.get("consensus_next_fy"))
    if line:
        lines.append(line)
    line = _valuation_adjusted_next_line(fy_next, fe.get("adjusted_next_fy"))
    if line:
        lines.append(line)
    line = _valuation_scenario_line(fy_next, fe.get("scenario_next_fy"), "next")
    if line:
        lines.append(line)
    line = _valuation_scenario_line(fy_next, fe.get("scenario_with_defaults_next_fy"), "next_default")
    if line:
        lines.append(line)
    line = _valuation_worst_line(fy_next, fe.get("worst_case_next_fy"))
    if line:
        lines.append(line)
    return lines


def _render_valuation_metrics(result: dict[str, object], max_bytes: int) -> str:
    fe_raw = result.get("forward_eps")
    fe = fe_raw if isinstance(fe_raw, dict) else {}
    ob = _as_dict(result.get("obligations"))
    year_cur = result.get("fiscal_year_current")
    year_next = result.get("fiscal_year_next")
    fy_cur = _valuation_fy_label(year_cur, year_next, True)
    fy_next = _valuation_fy_label(year_cur, year_next, False)
    lines = _valuation_header_lines(result, ob)
    gap = _valuation_price_gap_line(result)
    if gap:
        lines.append(gap)
    matched = _valuation_revenue_matched_line(ob)
    if matched:
        lines.append(matched)
    lines.extend(_valuation_current_fy_lines(fe, fy_cur))
    lines.extend(_valuation_next_fy_lines(fe, fy_next))
    lines.extend(_valuation_projected_lines(result))
    lines.extend(_valuation_obligation_scenario_lines(result))
    lines.extend(_valuation_coverage_lines(result))
    if result.get("note"):
        lines.append("Note: " + str(result["note"]))
    return _truncate_bytes("\n".join(lines), max_bytes)


def _valuation_pct_text(pct: object) -> str:
    """Percent-change prose for one projected price cell."""
    if isinstance(pct, (int, float, Decimal)):
        return f"{pct:+}%"
    if pct is None:
        return "n/a (no live price)"
    return f"{pct}%"


def _valuation_tier_line(tier: dict[str, object]) -> str:
    """One projected-price tier line (all P/E multiples joined)."""
    cells = []
    for m, c in _as_dict(tier.get("prices")).items():
        price_info = _as_dict(c)
        pct_s = _valuation_pct_text(price_info.get("pct_change_vs_current"))
        cells.append(f"{m} ${price_info.get('price')} ({pct_s})")
    return f"  {tier['tier']} (EPS ${tier['eps']}): " + " | ".join(cells)


def _valuation_projected_lines(result: dict[str, object]) -> list[str]:
    """Projected-share-price section ([] when no tiers disclosed)."""
    projected = _as_dict(result.get("projected_prices"))
    tiers = [t for t in _as_list(projected.get("tiers")) if isinstance(t, dict)]
    if not tiers:
        return []
    cur = projected.get("current_price")
    vs = f"(vs live ${cur})" if cur is not None else "(live price unavailable; moves vs current n/a)"
    lines = ["Projected share price by assumed P/E " + vs + ":"]
    for tier in tiers:
        assert isinstance(tier, dict)
        lines.append(_valuation_tier_line(tier))
    return lines


def _valuation_obligation_scenario_line(s: dict[str, object]) -> str:
    """One after-tax obligation EPS-impact scenario line."""
    eps_impact = s.get("eps_impact")
    reason_s = f" ({s['reason']})" if s.get("reason") else ""
    return (
        f"  {s['scenario']}: EPS {eps_impact if eps_impact is not None else 'unavailable'}"
        f" ({'one-time' if s['one_time'] else 'annual'})"
        f" — {s.get('note', '')}{reason_s}"
    )


def _valuation_obligation_scenario_lines(result: dict[str, object]) -> list[str]:
    """Obligation EPS-impact scenario section ([] when none disclosed)."""
    scenarios = _as_dict(result.get("obligation_eps_scenarios"))
    rows = [s for s in _as_list(scenarios.get("scenarios")) if isinstance(s, dict)]
    if not rows:
        return []
    rate = scenarios.get("effective_tax_rate")
    lines = [(f"Obligation EPS-impact scenarios (after-tax, tax rate {rate if rate is not None else 'unavailable'}):")]
    for s in rows:
        assert isinstance(s, dict)
        lines.append(_valuation_obligation_scenario_line(s))
    return lines


def _valuation_coverage_lines(result: dict[str, object]) -> list[str]:
    """Coverage/warnings section ([] when no coverage mapping)."""
    cov = _as_dict(result.get("coverage"))
    if not cov:
        return []
    filings = [str(f) for f in _as_list(cov.get("filings_examined"))]
    sections = _as_list(cov.get("sections_examined"))
    lines = [
        (
            f"Coverage: {cov.get('quantified_count', '?')} quantified / "
            f"{cov.get('unquantified_count', '?')} unquantified"
            f" | filings: {', '.join(filings) if filings else 'none with quantified rows'}"
            f" | sections examined: {len(sections)}"
        )
    ]
    warnings = [str(w) for w in _as_list(cov.get("warnings")) if str(w).strip()]
    if warnings:
        lines.append("Warnings: " + "; ".join(warnings))
    return lines


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_bytes(text: str, max_bytes: int, marker: str = TRUNCATED_MARKER) -> str:
    """Byte-safe prefix truncation with an explicit marker appended."""
    if _utf8_size(text) <= max_bytes:
        return text
    room = max_bytes - _utf8_size(marker)
    if room < 0:
        return marker[: max(0, max_bytes)]
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _utf8_size(text[:mid]) <= room:
            lo = mid
        else:
            hi = mid - 1
    prefix = text[:lo].rstrip()
    return prefix + marker


def _fit_lines(lines: list[str], max_bytes: int) -> tuple[list[str], int]:
    """Keep as many complete lines as fit; returns (kept, omitted)."""
    kept: list[str] = []
    omitted = 0
    used = 0
    for line in lines:
        cost = _utf8_size(line) + 1
        if cost > max_bytes:
            kept.append(_truncate_bytes(line, max_bytes - used))
            return kept, omitted
        if used + cost > max_bytes:
            omitted += 1
            continue
        kept.append(line)
        used += cost
    return kept, omitted


def _cell(value: object) -> str:
    if value is None:
        return ""
    s = str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return s.strip()


_MAX_TABLE_CELL_CHARS = 200


def _table_cell(value: object) -> str:
    """Single table cell; oversized values are truncated with a marker so
    one huge field cannot balloon the whole table."""
    s = _cell(value)
    if len(s) <= _MAX_TABLE_CELL_CHARS:
        return s
    return s[:_MAX_TABLE_CELL_CHARS].rstrip() + f"... [{len(s)} chars]"


def _minimal(result: dict[str, object], max_bytes: int) -> str:
    source = result.get("source") or result.get("dataset_id") or "tool result"
    text = f"Source: {source} | {TRUNCATED_MARKER}"
    return _truncate_bytes(text, max_bytes, marker="")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _render_error(result: dict[str, object], max_bytes: int) -> str:
    msg = str(result.get("error") or "Unknown error").strip()
    lines = [f"Error: {msg}"]
    for key in ("dataset", "dataset_id", "source", "request_purpose"):
        value = result.get(key)
        if value:
            lines.append(f"{key}: {value}")
    if result.get("http_status") is not None:
        lines.append(f"http_status: {result['http_status']}")
    if result.get("finra_response"):
        body = _cell(result["finra_response"])
        lines.append(f"finra_response: {body}")
    if result.get("environment"):
        lines.append(f"environment: {result['environment']}")
    return _truncate_bytes("\n".join(lines), max_bytes, marker="")


# ---------------------------------------------------------------------------
# get_finra_datapoints: compact Markdown table of only the selected fields
# ---------------------------------------------------------------------------


def _datapoints_header(fields: list[str]) -> tuple[str, str]:
    """Table header plus separator for the selected fields."""
    return (
        "| " + " | ".join(_table_cell(f) for f in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    )


def _datapoints_reserved(footer: str, stale_banner: str) -> int:
    """Bytes reserved for footer plus worst-case omission notice."""
    reserved = _utf8_size(TRUNCATED_MARKER + "\n" + "Omitted rows: 99999\n" + footer) + 32
    if stale_banner:
        reserved += _utf8_size(stale_banner) + 2
    return reserved


def _datapoints_row_line(row: object, fields: list[str]) -> str:
    """One datapoints table line (scalar cell or per-field mapping)."""
    if not isinstance(row, dict):
        return _table_cell(row)
    return "| " + " | ".join(_table_cell(row.get(f)) for f in fields) + " |"


def _datapoints_body_lines(
    records: list[object], fields: list[str], used: int, reserved: int, max_bytes: int
) -> tuple[list[str], int, int]:
    """Fitted body rows plus updated byte usage and omission count."""
    out: list[str] = []
    omitted = 0
    for row in records:
        line = _datapoints_row_line(row, fields)
        if used + _utf8_size(line) + 1 + reserved > max_bytes:
            omitted += 1
            continue
        out.append(line)
        used += _utf8_size(line) + 1
    return out, used, omitted


def _render_datapoints(result: dict[str, object], max_bytes: int) -> str:
    fields = [str(f) for f in _as_list(result.get("fields"))]
    if not fields:
        return _render_generic(result, max_bytes)
    records = _as_list(result.get("records"))
    header, sep = _datapoints_header(fields)
    stale_banner = _datapoints_stale_banner(result)
    footer = _datapoints_footer(result)
    if footer:
        footer = "\n" + footer
    reserved = _datapoints_reserved(footer, stale_banner)
    if reserved > max_bytes:
        return _minimal(result, max_bytes)
    used = _utf8_size(header + "\n" + sep + "\n")
    out = [header, sep]
    if stale_banner:
        out.append(stale_banner)
        used += _utf8_size(stale_banner) + 1
    body, _used, omitted = _datapoints_body_lines(records, fields, used, reserved, max_bytes)
    out.extend(body)
    text = "\n".join(out) + footer
    if omitted:
        text = "\n".join(out) + f"\n{TRUNCATED_MARKER}\nOmitted rows: {omitted}" + footer
    return text


def _datapoints_pagination_head(result: dict[str, object]) -> str:
    """Returned/total head of the pagination prose."""
    returned = result.get("returned_count")
    total = result.get("total_records")
    if total is not None:
        return f"{returned if returned is not None else '?'} returned of {total} total"
    return f"{returned if returned is not None else '?'} returned (Record-Total absent)"


def _datapoints_pagination_tail(result: dict[str, object]) -> str:
    """Source/more/offset tail of the pagination prose (may be empty)."""
    tail = ""
    if result.get("pagination_source"):
        tail += f", {result['pagination_source']}"
    if result.get("may_have_more") is not None:
        tail += f", more pages: {'yes' if result['may_have_more'] else 'no'}"
    if result.get("next_offset") is not None:
        tail += f", next_offset {result['next_offset']}"
    return tail


def _datapoints_pagination_text(result: dict[str, object]) -> str:
    """Pagination prose from counts plus paging extras."""
    return _datapoints_pagination_head(result) + _datapoints_pagination_tail(result)


def _datapoints_asof_line(result: dict[str, object]) -> str:
    """As-of/freshness line, empty when no as-of date present."""
    if not result.get("as_of_date"):
        return ""
    return f"As of: {result['as_of_date']} (freshness: {result.get('data_freshness') or 'unknown'})"


def _datapoints_warning_line(result: dict[str, object]) -> str:
    """Non-stale warnings line, empty when none present."""
    warnings = [str(w) for w in _as_list(result.get("warnings")) if str(w).strip()]
    non_stale = [w for w in warnings if "STALE" not in w]
    if not non_stale:
        return ""
    return "Warnings: " + "; ".join(non_stale)


def _datapoints_footer(result: dict[str, object]) -> str:
    parts = []
    if result.get("source"):
        parts.append(f"Source: {result['source']}")
    parts.append("Pagination: " + _datapoints_pagination_text(result))
    asof = _datapoints_asof_line(result)
    if asof:
        parts.append(asof)
    if result.get("environment"):
        parts.append(f"Environment: {result['environment']}")
    warn = _datapoints_warning_line(result)
    if warn:
        parts.append(warn)
    return "\n".join(parts)


def _datapoints_stale_banner(result: dict[str, object]) -> str:
    if result.get("data_freshness") == "stale" and result.get("as_of_date"):
        return (
            f"!! STALE/HISTORICAL DATA !! Newest record is "
            f"{result['as_of_date']} (over 90 days old); this is historical "
            "data, NOT current market data."
        )
    return ""


# ---------------------------------------------------------------------------
# FINRA analysis briefing
# ---------------------------------------------------------------------------


def _briefing_title(result: dict[str, object], query: dict[str, object]) -> str:
    """FINRA title from dataset name plus query ticker when present."""
    name = result.get("name") or result.get("dataset") or result.get("dataset_id")
    ticker = query.get("ticker") or result.get("ticker")
    return f"FINRA: {name}" + (f" — {ticker}" if ticker else "")


def _briefing_flag_status(value: object) -> str:
    """Completeness flag prose (yes/no/unknown, never blank)."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _briefing_coverage_line(cov: dict[str, object]) -> str:
    """Coverage line from row counts/dates/flags, empty when none."""
    cover = []
    if cov.get("rows_analyzed") is not None:
        cover.append(f"{cov['rows_analyzed']} rows analyzed")
    if cov.get("rows_matched") is not None:
        cover.append(f"{cov['rows_matched']} rows returned")
    if cov.get("first_date") and cov.get("last_date"):
        cover.append(f"{cov['first_date']} to {cov['last_date']}")
    statuses = [
        f"{f.replace('_', ' ')}: {_briefing_flag_status(cov.get(f))}"
        for f in ("page_complete", "query_complete", "analysis_complete")
    ]
    cover.append("; ".join(statuses))
    return "Coverage: " + ", ".join(cover)


def _briefing_query_range(query: dict[str, object]) -> str:
    """Date-range query part, empty when neither bound present."""
    if not (query.get("start_date") or query.get("end_date")):
        return ""
    return f"{query.get('start_date') or '…'}..{query.get('end_date') or '…'}"


def _briefing_query_paging(query: dict[str, object]) -> list[str]:
    """Limit/offset query parts ([] when neither present)."""
    parts = []
    if query.get("limit") is not None:
        parts.append(f"limit {query['limit']}")
    if query.get("offset"):
        parts.append(f"offset {query['offset']}")
    return parts


def _briefing_query_line(result: dict[str, object], query: dict[str, object]) -> str:
    """Query line from ticker/range/limit/offset, empty when none."""
    ticker = query.get("ticker") or result.get("ticker")
    parts = []
    if ticker:
        parts.append(f"ticker {ticker}")
    date_range = _briefing_query_range(query)
    if date_range:
        parts.append(date_range)
    parts.extend(_briefing_query_paging(query))
    if not parts:
        return ""
    return "Query: " + ", ".join(parts)


def _briefing_forced_query(forced: list[str], result: dict[str, object], query: dict[str, object]) -> None:
    """Append coverage/query lines to the forced head."""
    forced.append(_briefing_coverage_line(_as_dict(result.get("coverage"))))
    query_line = _briefing_query_line(result, query)
    if query_line:
        forced.append(query_line)


def _briefing_forced_meta(forced: list[str], result: dict[str, object]) -> None:
    """Append warnings/pagination/as-of/environment lines to the forced head."""
    warnings = _render_warnings(result)
    if warnings:
        forced.append("Warnings:\n" + "\n".join("  - " + w for w in warnings))
    pagination = _render_pagination(result)
    if pagination:
        forced.append("Pagination: " + pagination)
    asof = _datapoints_asof_line(result)
    if asof:
        forced.append(asof)
    if result.get("environment"):
        forced.append(f"Environment: {result['environment']}")


def _briefing_forced_head(result: dict[str, object]) -> list[str]:
    """Always-rendered head: title/source/coverage/query/warnings/paging."""
    query = _as_dict(result.get("query"))
    forced = [_briefing_title(result, query)]
    if result.get("source"):
        forced.append(f"Source: {result['source']}")
    _briefing_forced_query(forced, result, query)
    _briefing_forced_meta(forced, result)
    return forced


def _briefing_optional_sections(result: dict[str, object]) -> list[tuple[str, list[str]]]:
    """Fitted optional sections in render order (metrics/briefing/trends)."""
    optional: list[tuple[str, list[str]]] = []
    metrics = _render_metrics(result)
    if metrics:
        optional.append(("Key metrics", metrics))
    briefing = _render_briefing_prose(result)
    if briefing:
        optional.append(("Briefing", briefing))
    trends = [str(t) for t in _as_list(result.get("trends"))]
    if trends:
        optional.append(("Trends", trends))
    categorical = _render_categorical(result)
    if categorical:
        optional.append(("Categorical", categorical))
    return optional


def _briefing_append_section(
    out: list[str], header: str, lines: list[str], used: int, max_bytes: int
) -> tuple[int, bool]:
    """Append one optional section whole/fitted/truncated; returns (used, stop)."""
    block = header + "\n" + "\n".join("  - " + line for line in lines)
    cost = _utf8_size(block) + 1
    if used + cost <= max_bytes:
        out.append(block)
        return used + cost, False
    kept, omitted = _fit_lines(["  - " + line for line in lines], max_bytes - used - _utf8_size(header) - 4)
    if kept:
        out.append(header)
        out.extend(kept)
        used += _utf8_size(header) + _utf8_size("\n".join(kept)) + 4
    if omitted:
        out.append(f"{TRUNCATED_MARKER} (Omitted rows: {omitted})")
        return used, True
    return used, False


def _render_briefing(result: dict[str, object], max_bytes: int) -> str:
    forced = _briefing_forced_head(result)
    used = _utf8_size("\n".join(forced))
    if used > max_bytes:
        return _minimal(result, max_bytes)
    out = list(forced)
    for header, lines in _briefing_optional_sections(result):
        used, stop = _briefing_append_section(out, header, lines, used, max_bytes)
        if stop:
            break
    return "\n".join(out)


def _metrics_change_suffix(entry: dict[str, object]) -> str:
    """Change/pct suffix for one latest-vs-prior entry (empty when no change)."""
    change = entry.get("change")
    if change is None:
        return ""
    pct = entry.get("change_percent")
    if pct is None:
        return f" (change {change:+,})"
    if isinstance(pct, (int, float, Decimal)):
        return f" (change {change:+,}, {pct:+.2f}%)"
    return f" (change {change:+,}, {pct}%)"


def _metrics_entry_line(entry: dict[str, object]) -> str:
    """One latest-vs-prior metric line."""
    base = f"{entry.get('field', '?')}: latest {entry.get('latest')}"
    if entry.get("prior") is not None:
        base += f" vs prior {entry.get('prior')}"
    return base + _metrics_change_suffix(entry)


def _metrics_field_line(name: object, stats: dict[str, object]) -> str:
    """One field-stats line (min/max/mean/median/sum plus missing)."""
    parts = [f"{key} {stats[key]}" for key in ("min", "max", "mean", "median", "sum") if key in stats]
    if "missing" in stats:
        parts.append(f"missing {stats['missing']}")
    return f"{name}: {', '.join(parts)}"


def _render_metrics(result: dict[str, object]) -> list[str]:
    metrics = _as_dict(result.get("metrics"))
    lines = [_metrics_entry_line(e) for e in _as_list(metrics.get("latest_vs_prior")) if isinstance(e, dict)]
    fields = _as_dict(metrics.get("fields"))
    for name in sorted(fields):
        stats = fields.get(name)
        if isinstance(stats, dict):
            lines.append(_metrics_field_line(name, stats))
    return lines


def _render_briefing_prose(result: dict[str, object]) -> list[str]:
    briefing = result.get("briefing")
    if not isinstance(briefing, dict) or not briefing.get("summary"):
        return []
    lines = [str(briefing["summary"])]
    for finding in briefing.get("key_findings") or []:
        if isinstance(finding, str) and finding.strip():
            lines.append(finding.strip())
    return lines


def _render_categorical(result: dict[str, object]) -> list[str]:
    breakdowns = _as_dict(_as_dict(result.get("metrics")).get("categorical"))
    lines: list[str] = []
    for field, counts in breakdowns.items():
        if not isinstance(counts, dict):
            continue
        top = ", ".join(f"{k} {v}" for k, v in list(counts.items())[:8])
        lines.append(f"{field}: {top}")
    return lines


def _render_warnings(result: dict[str, object]) -> list[str]:
    return [str(w) for w in _as_list(result.get("warnings")) if str(w).strip()]


def _render_pagination(result: dict[str, object]) -> str:
    total = result.get("total_records")
    source = result.get("pagination_source")
    parts: list[str] = []
    if total is not None:
        parts.append(f"{total} total records")
    if source:
        parts.append(str(source))
    if result.get("may_have_more") is not None:
        parts.append(f"more pages: {'yes' if result['may_have_more'] else 'no'}")
    if result.get("next_offset") is not None:
        parts.append(f"next_offset {result['next_offset']}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# SEC filing-style results: header + byte-safe truncated text
# ---------------------------------------------------------------------------
class _SecFactsAccumulator:
    """Budgeted line accumulator for SEC-facts rendering."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.lines: list[str] = []
        self.used = 0
        self.omitted = 0

    def add(self, text: str) -> bool:
        """Append one line when it fits (counting omissions otherwise)."""
        cost = _utf8_size(text) + 1
        if self.used + cost > self.max_bytes:
            self.omitted += 1
            return False
        self.lines.append(text)
        self.used += cost
        return True


_SEC_FACTS_SKIP = frozenset(
    {
        "source",
        "metric",
        "data_source",
        "as_of_date",
        "requested_as_of",
        "row_count",
        "returned_count",
        "truncated",
        "ticker",
        "quarterly_eps",
        "annual_history",
        "matching_concepts",
        "balance_sheet",
        "last_dividend",
        "next_declared_dividend",
        "past_events",
        "safety",
        "risk_flags",
    }
)
_SEC_DIV_GROWTH_KEYS = ("growth_1y", "growth_3y_cagr", "growth_5y_cagr", "growth_10y_cagr")


def _sec_facts_is_dividend(result: dict[str, object]) -> bool:
    """True when the envelope carries dividend payload fields."""
    return "dividend_status" in result or "last_dividend" in result or "next_declared_dividend" in result


def _sec_facts_header(result: dict[str, object], acc: _SecFactsAccumulator) -> None:
    """Provenance plus row-count lines at the head of the envelope."""
    ticker = result.get("ticker", "?")
    metric = result.get("metric", "fundamentals")
    data_source = result.get("data_source", "?")
    acc.add(f"{ticker} {metric} [{data_source}] as of {result.get('as_of_date', '?')}")
    if result.get("requested_as_of"):
        acc.add(f"requested_as_of: {result['requested_as_of']} (live result: store could not serve that date)")
    rows_info = []
    if result.get("row_count") is not None:
        rows_info.append(f"rows: {result['row_count']}")
    if result.get("returned_count") is not None and result.get("returned_count") != result.get("row_count"):
        rows_info.append(f"returned: {result['returned_count']}")
    if result.get("truncated"):
        rows_info.append("truncated")
    if rows_info:
        acc.add(" | ".join(rows_info))


def _sec_facts_skip_keys(is_div: bool) -> set[str]:
    """Envelope keys excluded from the scalar sweep (plus growth keys)."""
    skip = set(_SEC_FACTS_SKIP)
    if is_div:
        skip |= set(_SEC_DIV_GROWTH_KEYS) | {"growth_trend", "growth_basis"}
    return skip


def _sec_facts_scalars(result: dict[str, object], skip: set[str], acc: _SecFactsAccumulator) -> None:
    """Scalar payload fields (lists/dicts/None never rendered here)."""
    if _sec_facts_is_dividend(result):
        acc.add("CURRENT:")
    for key, value in result.items():
        if key in skip or value is None or isinstance(value, (list, dict)):
            continue
        acc.add(f"{key}: {_cell(value)}")


def _sec_facts_last_paid_line(result: dict[str, object]) -> str:
    """LAST PAID dividend line (none when no amount disclosed)."""
    last = _as_dict(result.get("last_dividend"))
    if last.get("amount_per_share") is not None:
        return f"LAST PAID: {last.get('amount_per_share')} on {last.get('payment_date', '?')}"
    return "LAST PAID: none"


def _sec_facts_next_declared_line(result: dict[str, object]) -> str:
    """NEXT DECLARED dividend line (none when no amount disclosed)."""
    nxt = _as_dict(result.get("next_declared_dividend"))
    if nxt.get("amount_per_share") is None:
        return "NEXT DECLARED: none"
    text = f"NEXT DECLARED: {nxt.get('amount_per_share')} payable {nxt.get('payment_date', '?')}"
    if nxt.get("record_date"):
        text += f" (record {nxt['record_date']})"
    return text


def _sec_facts_dividends(result: dict[str, object], acc: _SecFactsAccumulator) -> None:
    """Declared-dividend lines (no-op when no dividend payload present)."""
    if "last_dividend" not in result and "next_declared_dividend" not in result:
        return
    acc.add(_sec_facts_last_paid_line(result))
    acc.add(_sec_facts_next_declared_line(result))


def _sec_facts_growth_line(result: dict[str, object]) -> str:
    """Dividend-growth line from CAGR keys plus trend/cadence extras."""
    growth = [f"{k}: {_cell(result.get(k))}" for k in _SEC_DIV_GROWTH_KEYS if result.get(k) is not None]
    if result.get("growth_trend"):
        growth.append(f"trend: {_cell(result.get('growth_trend'))}")
    if result.get("payment_cadence"):
        growth.append(f"cadence: {_cell(result.get('payment_cadence'))} ({_cell(result.get('cadence_confidence'))})")
    return f"GROWTH: {' | '.join(growth) if growth else 'none'}"


def _sec_facts_safety_line(safety: dict[str, object]) -> str:
    """SAFETY coverage line with methodology suffix when disclosed."""
    cov = [
        f"{k} {safety[k]}"
        for k in ("earnings_payout_ratio", "fcf_payout_ratio", "fcf_coverage", "cash_to_annual_dividend")
        if safety.get(k) is not None
    ]
    method = f" [{safety['methodology']}]" if safety.get("methodology") else ""
    return f"SAFETY: {' | '.join(cov) if cov else 'none'}{method}"


def _sec_facts_risk_line(safety: dict[str, object]) -> str:
    """RISK FLAGS line from raised flags (none when none raised)."""
    flags: object = safety.get("risk_flags") or []
    items: list[object] = flags if isinstance(flags, list) else []
    raised = [f["flag"] for f in items if isinstance(f, dict) and f.get("status") is True]
    return f"RISK FLAGS: {', '.join(str(f) for f in raised) if raised else 'none'}"


def _sec_facts_safety_lines(result: dict[str, object]) -> list[str]:
    """Dividend-safety plus risk-flag lines (none-pair when no mapping)."""
    safety = result.get("safety")
    if not isinstance(safety, dict):
        return ["SAFETY: none", "RISK FLAGS: none"]
    return [_sec_facts_safety_line(safety), _sec_facts_risk_line(safety)]


def _sec_facts_dividend_growth(result: dict[str, object], is_div: bool, acc: _SecFactsAccumulator) -> None:
    """Growth/safety/history lines for dividend envelopes (no-op otherwise)."""
    if not is_div:
        return
    acc.add(_sec_facts_growth_line(result))
    for line in _sec_facts_safety_lines(result):
        acc.add(line)
    if result.get("annual_history"):
        acc.add("HISTORY:")


def _sec_facts_quarterly_line(row: dict[str, object]) -> str:
    """One quarterly-EPS line with optional basic-EPS suffix."""
    text = (
        f"- {row.get('fiscal_year', '?')} {row.get('fiscal_period', '?')}"
        f" (period end {row.get('period_end', '?')}):"
        f" diluted {row.get('eps_diluted', '?')}"
    )
    if row.get("eps_basic") is not None:
        text += f" | basic {row['eps_basic']}"
    return text


def _sec_facts_matching_line(row: dict[str, object]) -> str:
    """One matching-concept line with period provenance."""
    return (
        f"- {_cell(row.get('concept'))}: {row.get('value', '?')}"
        f" (period end {row.get('period_end', '?')}, {row.get('fiscal_period', '?')})"
    )


def _sec_facts_eps_sections(result: dict[str, object], acc: _SecFactsAccumulator) -> None:
    """Quarterly-EPS plus annual-history dividend row sections."""
    for row in _as_list(result.get("quarterly_eps")):
        if isinstance(row, dict):
            acc.add(_sec_facts_quarterly_line(row))
    for row in _as_list(result.get("annual_history")):
        if isinstance(row, dict):
            acc.add(f"- {row.get('fiscal_year', '?')}: dividend {row.get('dividend_per_share', '?')}")


def _sec_facts_concept_sections(result: dict[str, object], acc: _SecFactsAccumulator) -> None:
    """Matching-concept plus balance-sheet row sections."""
    for row in _as_list(result.get("matching_concepts")):
        if isinstance(row, dict):
            acc.add(_sec_facts_matching_line(row))
    sheet = result.get("balance_sheet")
    if isinstance(sheet, dict):
        for key, value in sheet.items():
            acc.add(f"- {_cell(key)}: {_table_cell(value)}")


def _sec_facts_row_sections(result: dict[str, object], acc: _SecFactsAccumulator) -> None:
    """Quarterly/annual/matching/balance-sheet row sections in order."""
    _sec_facts_eps_sections(result, acc)
    _sec_facts_concept_sections(result, acc)


def _render_sec_facts(result: dict[str, object], max_bytes: int) -> str:
    """Compact render for the sec_facts envelope (get_fundamentals +
    get_xbrl_facts): header with provenance, scalar payload fields, then
    quarterly/matching rows one per line within the byte budget."""
    acc = _SecFactsAccumulator(max_bytes)
    is_div = _sec_facts_is_dividend(result)
    _sec_facts_header(result, acc)
    _sec_facts_scalars(result, _sec_facts_skip_keys(is_div), acc)
    _sec_facts_dividends(result, acc)
    _sec_facts_dividend_growth(result, is_div, acc)
    _sec_facts_row_sections(result, acc)
    if acc.omitted > 0:
        acc.add(f"{TRUNCATED_MARKER} (Omitted rows: {acc.omitted})")
    if not acc.lines:
        return _minimal(result, max_bytes)
    return "\n".join(acc.lines)


def _is_text_result(result: dict[str, object]) -> bool:
    for key in _TEXT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and len(value) > 400:
            return True
    return False


def _text_result_header(result: dict[str, object]) -> str:
    """Provenance header for long-text envelopes (ticker/form/filed/source)."""
    header_lines = []
    ticker = result.get("ticker")
    if ticker:
        header_lines.append(f"Ticker: {ticker}")
    for key in ("form_type", "item", "statement_type", "concept_searched"):
        if result.get(key):
            header_lines.append(f"{key}: {result[key]}")
    if result.get("filed"):
        header_lines.append(f"Filed: {result['filed']}")
    if result.get("source"):
        header_lines.append(f"Source: {result['source']}")
    return "\n".join(header_lines)


def _text_result_body_lines(result: dict[str, object], body_keys: list[str]) -> list[str]:
    """Labeled body lines for present text keys (blank texts skipped)."""
    body_lines: list[str] = []
    for key in body_keys:
        text = result.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        label = "" if key == "text" and len(body_keys) == 1 else f"{key}: "
        body_lines.append(label + text)
    return body_lines


def _render_text_result(result: dict[str, object], max_bytes: int) -> str:
    header = _text_result_header(result)
    header_cost = _utf8_size(header) + 1
    if header_cost > max_bytes:
        return _minimal(result, max_bytes)
    body_keys = [k for k in _TEXT_KEYS if isinstance(result.get(k), str)]
    budget_for_text = max_bytes - header_cost - 2
    if budget_for_text <= 0:
        return header
    body = "\n\n".join(_text_result_body_lines(result, body_keys))
    if body:
        body = _truncate_bytes(body, budget_for_text)
    return header + "\n\n" + body if body else header


class _GenericAccumulator:
    """Budgeted line accumulator for generic structured rendering."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.lines: list[str] = []
        self.used = 0
        self.omitted: list[str] = []

    def add(self, text: str) -> bool:
        """Append one line when it fits (False, never counted, otherwise)."""
        cost = _utf8_size(text) + 1
        if self.used + cost > self.max_bytes:
            return False
        self.lines.append(text)
        self.used += cost
        return True


def _generic_head_lines(result: dict[str, object], acc: _GenericAccumulator) -> None:
    """Ticker/source/concept head lines (present values only)."""
    for key in ("ticker", "source", "concept_searched"):
        value = result.get(key)
        if value is not None:
            acc.add(f"{key}: {_cell(value)}")


_INSIDER_ROW_KEYS = (
    "insider_name",
    "transaction_code",
    "transaction_kind",
    "transaction_date",
    "acquired_disposed",
    "shares",
    "price",
    "filed_at",
    "accession_no",
)


def _is_insider_envelope(result: dict[str, object]) -> bool:
    """True for the get_insider_activity envelope: transactions list of dicts with codes."""
    rows = result.get("transactions")
    return isinstance(rows, list) and any(
        isinstance(r, dict) and ("transaction_code" in r or "transaction_kind" in r) for r in rows
    )


def _insider_row_line(item: dict[str, object]) -> str:
    """One insider row: identity + code/kind/date/A-D/shares/price (never the first-6 cut)."""
    return ", ".join(f"{k} {_cell(item[k])}" for k in _INSIDER_ROW_KEYS if item.get(k) not in (None, ""))


def _render_insider_activity(result: dict[str, object], max_bytes: int) -> str:
    """Narrow insider renderer: head lines + one line per transaction row within budget."""
    acc = _GenericAccumulator(max_bytes)
    _generic_head_lines(result, acc)
    for key, value in result.items():
        if value is None or key in ("ticker", "source", "concept_searched", "transactions"):
            continue
        if isinstance(value, list):
            _generic_list_value(key, value, acc)
        else:
            _generic_scalar_or_mapping(key, value, acc)
    rows = result.get("transactions")
    rows = rows if isinstance(rows, list) else []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if not acc.add("  - " + _insider_row_line(item)):
            acc.omitted.append("transactions")
            break
    if acc.omitted:
        acc.add(f"{TRUNCATED_MARKER} (Omitted rows: {len(acc.omitted)} in {', '.join(acc.omitted)})")
    if not acc.lines:
        return _minimal(result, max_bytes)
    return "\n".join(acc.lines)


def _generic_list_value(key: str, value: list[object], acc: _GenericAccumulator) -> None:
    """List value as bulleted summary lines (empty list renders none)."""
    kept = 0
    for item in value:
        line = "  - " + _cell(item if not isinstance(item, dict) else _summarize_dict(item))
        if not acc.add(line):
            acc.omitted.append(key)
            break
        kept += 1
    if kept == 0 and not value:
        acc.add(f"{key}: none")


def _generic_scalar_or_mapping(key: str, value: object, acc: _GenericAccumulator) -> None:
    """Mapping as joined cells, scalar as one cell line (truncated on overflow)."""
    if isinstance(value, dict):
        rendered = ", ".join(f"{k}: {_cell(v)}" for k, v in value.items())
        if not acc.add(f"{key}: {rendered}"):
            acc.add(f"{key}: {TRUNCATED_MARKER}")
    elif not acc.add(f"{key}: {_cell(value)}"):
        acc.add(f"{key}: {TRUNCATED_MARKER}")


def _render_generic(result: dict[str, object], max_bytes: int) -> str:
    acc = _GenericAccumulator(max_bytes)
    _generic_head_lines(result, acc)
    for key, value in result.items():
        if value is None or key in ("ticker", "source", "concept_searched"):
            continue
        if isinstance(value, list):
            _generic_list_value(key, value, acc)
        else:
            _generic_scalar_or_mapping(key, value, acc)
    if acc.omitted:
        acc.add(f"{TRUNCATED_MARKER} (Omitted rows: {len(acc.omitted)} in {', '.join(acc.omitted)})")
    if not acc.lines:
        return _minimal(result, max_bytes)
    return "\n".join(acc.lines)


def _summarize_dict(item: dict[str, object]) -> str:
    parts = []
    for key in ("dataset", "group", "name", "description", "concept", "value", "period_end"):
        if key in item and item[key] not in (None, ""):
            parts.append(f"{key} {_cell(item[key])}")
    if not parts:
        return ", ".join(f"{k} {_cell(v)}" for k, v in list(item.items())[:6])
    return ", ".join(parts)
