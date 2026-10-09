"""Reasoner transport over the existing OpenCode Responses logic.

Reasoner proposes, never authorizes — output always goes back through JEV.
This module makes no JEV calls and no authorization decisions; it only posts
prompts built by the caller (decision/prompts.ts builders) to the OpenCode
Responses endpoint and validates the typed candidate shapes (proposals /
analyses / evidence requests). JEV admission (choice/noul/score) happens
elsewhere; the candidates returned here are non-authoritative until admitted.

Thin transport: reuses the decision/run.ts postOpenCode contract
(POST {model, input} -> Responses payload with output/message/output_text
parts -> exact-keys JSON object). No new protocol, no new auth flow.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict
from urllib import error as _urlerror
from urllib import request as _urlrequest


class Proposal(TypedDict):
    id: str
    objectiveId: str
    question: str
    dependsOn: list[str]
    whyItMatters: str


class AnalysisNumber(TypedDict):
    value: str
    evidenceId: str
    quote: str


class AnalysisAssumption(TypedDict):
    assumptionId: str
    text: str


class Analysis(TypedDict):
    nodeId: str
    objectiveId: str
    interpretation: str
    evidenceRefs: list[str]
    numbers: NotRequired[list[AnalysisNumber]]
    assumptions: NotRequired[list[AnalysisAssumption]]


class EvidenceRequest(TypedDict):
    nodeId: str
    objectiveId: str
    missingEvidence: str


def _is_obj(v: object) -> bool:
    return isinstance(v, dict)


def _nonempty(stage: str, v: object, what: str) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{stage}: {what} must be a nonempty string")
    return v


def _exact_keys(stage: str, o: Mapping[str, object], keys: list[str]) -> None:
    actual = sorted(o)
    if actual != sorted(keys):
        raise ValueError(f"{stage}: unexpected fields [{','.join(actual)}]")


def _str_list(stage: str, v: object, what: str) -> list[str]:
    if not isinstance(v, list) or any(not isinstance(d, str) for d in v):
        raise ValueError(f"{stage}: {what} must be string[]")
    return list(v)


def check_decompose_hints(stage: str, value: object) -> dict[str, object]:
    """Optional intake hints: tickers + corrected_query. Unknown fields still reject."""
    if not isinstance(value, dict):
        raise TypeError(f"{stage}: malformed_opencode_json")
    allowed = {"proposals", "tickers", "corrected_query"}
    actual = set(value.keys())
    if not actual <= allowed or "proposals" not in actual:
        raise ValueError(f"{stage}: unexpected fields [{','.join(sorted(actual))}]")
    out: dict[str, object] = {"proposals": value["proposals"]}
    raw_tickers = value.get("tickers")
    if raw_tickers is not None:
        if not isinstance(raw_tickers, list) or any(not isinstance(t, str) or not t.strip() for t in raw_tickers):
            raise ValueError(f"{stage}: tickers must be string[]")
        out["tickers"] = [t.strip().upper() for t in raw_tickers if isinstance(t, str) and t.strip()]
    raw_query = value.get("corrected_query")
    if raw_query is not None:
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError(f"{stage}: corrected_query must be a nonempty string")
        out["corrected_query"] = raw_query.strip()
    return out


def check_proposals(stage: str, value: object, objective_id: str, prior_ids: set[str]) -> list[Proposal]:
    """Validate non-authoritative proposal candidates (mirrors run.ts checkProposals)."""
    if not isinstance(value, list):
        raise TypeError(f"{stage}: proposals must be an array")
    ids: set[str] = set()
    items: list[Mapping[str, object]] = []
    for item in value:
        if not _is_obj(item) or not isinstance(item, Mapping):
            raise ValueError(f"{stage}: proposal must be an object")
        _exact_keys(stage, item, ["dependsOn", "id", "objectiveId", "question", "whyItMatters"])
        pid = _nonempty(stage, item.get("id"), "proposal.id")
        if not pid.startswith(f"{objective_id}-"):
            raise ValueError(f"{stage}: proposal id {pid} must start with {objective_id}-")
        if item.get("objectiveId") != objective_id:
            raise ValueError(f"{stage}: proposal {pid} references wrong objective")
        _nonempty(stage, item.get("question"), "proposal.question")
        _nonempty(stage, item.get("whyItMatters"), "proposal.whyItMatters")
        if pid in ids:
            raise ValueError(f"{stage}: duplicate proposal id {pid}")
        if pid in prior_ids:
            raise ValueError(f"{stage}: reused proposal id {pid}")
        ids.add(pid)
        items.append(item)
    refs = prior_ids | ids
    for item in items:
        pid = str(item.get("id"))
        for dep in _str_list(stage, item.get("dependsOn"), f"proposal {pid} dependsOn"):
            if dep == pid:
                raise ValueError(f"{stage}: proposal {pid} depends on itself")
            if dep not in refs:
                raise ValueError(f"{stage}: proposal {pid} references unknown id {dep}")
    out: list[Proposal] = []
    for item in items:
        raw_depends = item.get("dependsOn")
        depends = [str(d) for d in raw_depends] if isinstance(raw_depends, list) else []
        out.append(
            Proposal(
                id=str(item.get("id")),
                objectiveId=str(item.get("objectiveId")),
                question=str(item.get("question")),
                dependsOn=depends,
                whyItMatters=str(item.get("whyItMatters")),
            )
        )
    return out


def check_analyses(
    stage: str,
    value: object,
    objective_id: str,
    proposal_ids: set[str],
    evidence_ids: set[str],
) -> list[Analysis]:
    """Validate analysis candidates (mirrors run.ts checkAnalyses)."""
    if not isinstance(value, list):
        raise TypeError(f"{stage}: analyses must be an array")
    seen: set[str] = set()
    for item in value:
        if not _is_obj(item) or not isinstance(item, Mapping):
            raise ValueError(f"{stage}: analysis must be an object")
        actual = set(item.keys())
        if not {"evidenceRefs", "interpretation", "nodeId", "objectiveId"} <= actual or not actual <= {
            "assumptions",
            "evidenceRefs",
            "interpretation",
            "nodeId",
            "numbers",
            "objectiveId",
        }:
            raise ValueError(f"{stage}: unexpected fields [{','.join(sorted(item))}]")
        node_id = _nonempty(stage, item.get("nodeId"), "analysis.nodeId")
        if node_id not in proposal_ids:
            raise ValueError(f"{stage}: analysis references unknown proposal {node_id}")
        if item.get("objectiveId") != objective_id:
            raise ValueError(f"{stage}: analysis {node_id} references wrong objective")
        _nonempty(stage, item.get("interpretation"), "analysis.interpretation")
        refs = _str_list(stage, item.get("evidenceRefs"), f"analysis {node_id} evidenceRefs")
        for ref in refs:
            if ref not in evidence_ids:
                raise ValueError(f"{stage}: analysis {node_id} references unknown evidence {ref}")
        ref_set = set(refs)
        if "numbers" in item:
            raw_numbers = item.get("numbers")
            if not isinstance(raw_numbers, list):
                raise ValueError(f"{stage}: analysis {node_id} numbers must be an array")
            for n in raw_numbers:
                if not _is_obj(n) or not isinstance(n, Mapping):
                    raise ValueError(f"{stage}: analysis {node_id} number must be an object")
                _exact_keys(stage, n, ["evidenceId", "quote", "value"])
                _nonempty(stage, n.get("value"), "analysis.number.value")
                eid = _nonempty(stage, n.get("evidenceId"), "analysis.number.evidenceId")
                _nonempty(stage, n.get("quote"), "analysis.number.quote")
                if eid not in ref_set:
                    raise ValueError(f"{stage}: analysis {node_id} number references unknown evidence {eid}")
        if "assumptions" in item:
            raw_assumptions = item.get("assumptions")
            if not isinstance(raw_assumptions, list):
                raise ValueError(f"{stage}: analysis {node_id} assumptions must be an array")
            seen_assumption: set[str] = set()
            for a in raw_assumptions:
                if not _is_obj(a) or not isinstance(a, Mapping):
                    raise ValueError(f"{stage}: analysis {node_id} assumption must be an object")
                _exact_keys(stage, a, ["assumptionId", "text"])
                aid = _nonempty(stage, a.get("assumptionId"), "analysis.assumption.assumptionId")
                _nonempty(stage, a.get("text"), "analysis.assumption.text")
                if aid in seen_assumption:
                    raise ValueError(f"{stage}: analysis {node_id} duplicate assumptionId {aid}")
                seen_assumption.add(aid)
        if node_id in seen:
            raise ValueError(f"{stage}: duplicate analysis for {node_id}")
        seen.add(node_id)
    out: list[Analysis] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw_refs = item.get("evidenceRefs")
        refs = [str(r) for r in raw_refs] if isinstance(raw_refs, list) else []
        entry = Analysis(
            nodeId=str(item.get("nodeId")),
            objectiveId=str(item.get("objectiveId")),
            interpretation=str(item.get("interpretation")),
            evidenceRefs=refs,
        )
        if isinstance(item.get("numbers"), list):
            entry["numbers"] = [
                AnalysisNumber(
                    value=str(n.get("value")), evidenceId=str(n.get("evidenceId")), quote=str(n.get("quote"))
                )
                for n in item["numbers"]
                if isinstance(n, Mapping)
            ]
        if isinstance(item.get("assumptions"), list):
            entry["assumptions"] = [
                AnalysisAssumption(assumptionId=str(a.get("assumptionId")), text=str(a.get("text")))
                for a in item["assumptions"]
                if isinstance(a, Mapping)
            ]
        out.append(entry)
    return out


def check_evidence_requests(stage: str, value: object, objective_id: str, node_ids: set[str]) -> list[EvidenceRequest]:
    """Validate evidence-request candidates (mirrors run.ts checkEvidenceRequests)."""
    if not isinstance(value, list):
        raise TypeError(f"{stage}: evidenceRequests must be an array")
    for item in value:
        if not _is_obj(item) or not isinstance(item, Mapping):
            raise ValueError(f"{stage}: evidenceRequest must be an object")
        _exact_keys(stage, item, ["missingEvidence", "nodeId", "objectiveId"])
        node_id = _nonempty(stage, item.get("nodeId"), "evidenceRequest.nodeId")
        if node_id not in node_ids:
            raise ValueError(f"{stage}: evidenceRequest references unknown node {node_id}")
        if item.get("objectiveId") != objective_id:
            raise ValueError(f"{stage}: evidenceRequest for {node_id} references wrong objective")
        _nonempty(stage, item.get("missingEvidence"), "evidenceRequest.missingEvidence")
    out: list[EvidenceRequest] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        out.append(
            EvidenceRequest(
                nodeId=str(item.get("nodeId")),
                objectiveId=str(item.get("objectiveId")),
                missingEvidence=str(item.get("missingEvidence")),
            )
        )
    return out


def _parse_opencode_output(
    stage: str, raw: object, keys: list[str], optional: list[str] | None = None
) -> dict[str, object]:
    """Extract concatenated output_text, parse JSON with required + optional keys (mirrors run.ts)."""
    output = raw.get("output") if isinstance(raw, dict) else None
    if not isinstance(output, list):
        raise TypeError(f"{stage}: malformed_opencode_response")
    text = ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            chunk = part.get("text")
            if isinstance(chunk, str):
                text += chunk
    if not text:
        raise ValueError(f"{stage}: malformed_opencode_response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"{stage}: malformed_opencode_json") from None
    if not isinstance(parsed, dict):
        raise TypeError(f"{stage}: malformed_opencode_json")
    allowed = set(keys) | set(optional or [])
    actual = set(parsed.keys())
    if not set(keys) <= actual <= allowed:
        raise ValueError(f"{stage}: unexpected fields [{','.join(sorted(actual))}]")
    return parsed


PostFn = Callable[[str, str], dict[str, object]]


def _default_post(url: str, api_key: str, model: str, prompt: str, timeout_s: float = 120.0) -> dict[str, object]:
    body = json.dumps({"model": model, "input": prompt}).encode()
    req = _urlrequest.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "stockbot-reasoner/0.1",
            "x-opencode-session": str(uuid.uuid4()),
        },
    )
    try:
        with _urlrequest.urlopen(req, timeout=timeout_s) as res:
            loaded = json.load(res)
    except _urlerror.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "ignore")
        except OSError, ValueError, AttributeError:
            detail = ""
        if detail:
            raise RuntimeError(f"opencode_request_failed: {e.code} {detail[:300]}") from None
        raise RuntimeError(f"opencode_request_failed: {e.code}") from None
    except (OSError, TimeoutError, ValueError) as e:
        raise RuntimeError(f"opencode_request_failed: {e}") from None
    if not isinstance(loaded, dict):
        raise TypeError("opencode_request_failed: non-object response")
    return loaded


@dataclass
class ReasonerClient:
    """Thin OpenCode transport: propose candidates, never authorize.

    decompose/analyze/expand take caller-built prompt text (from
    decision/prompts.ts builders), post it, and return validated candidate
    dicts. Every return value needs JEV admission before it means anything.
    analyze returns envelope-checked lists; apply check_analyses /
    check_evidence_requests caller-side with the admitted proposal and
    evidence id sets (mirrors run.ts staging).
    """

    model: str
    url: str
    api_key: str = ""
    post: PostFn | None = None
    timeout_s: float = 120.0

    def _call(self, stage: str, prompt: str, keys: list[str], optional: list[str] | None = None) -> dict[str, object]:
        if not prompt:
            raise ValueError(f"{stage}: malformed_prompt_builder")
        if self.post is not None:
            return _parse_opencode_output(stage, self.post(prompt, self.model), keys, optional)
        if not self.model.strip():
            raise RuntimeError("opencode_unavailable: missing OPENCODE_MODEL")
        if not self.api_key:
            raise RuntimeError("opencode_unavailable: missing OPENCODE_API_KEY")
        if not self.url.strip():
            raise RuntimeError("opencode_unavailable: missing OPENCODE_URL")
        return _parse_opencode_output(
            stage, _default_post(self.url, self.api_key, self.model, prompt, self.timeout_s), keys, optional
        )

    def decompose(self, prompt: str, objective_id: str) -> dict[str, object]:
        """Propose follow-up questions; non-authoritative until JEV admits."""
        raw = self._call("decompose", prompt, ["proposals"], ["tickers", "corrected_query"])
        hints = check_decompose_hints("decompose", raw)
        out: dict[str, object] = {"proposals": check_proposals("decompose", hints["proposals"], objective_id, set())}
        if "tickers" in hints:
            out["tickers"] = hints["tickers"]
        if "corrected_query" in hints:
            out["corrected_query"] = hints["corrected_query"]
        return out

    def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
        """Propose interpretations + evidence requests; non-authoritative until JEV adjudicates."""
        out = self._call("analyze", prompt, ["analyses", "evidenceRequests"])
        raw_analyses: list[object] = out["analyses"] if isinstance(out["analyses"], list) else []
        raw_requests: list[object] = out["evidenceRequests"] if isinstance(out["evidenceRequests"], list) else []
        return {
            "analyses": [a for a in raw_analyses if isinstance(a, dict)],
            "evidenceRequests": [r for r in raw_requests if isinstance(r, dict)],
        }

    def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
        """Propose follow-ups + evidence requests; non-authoritative until JEV admits."""
        out = self._call("expand", prompt, ["evidenceRequests", "proposals"])
        proposals = check_proposals("expand", out["proposals"], objective_id, prior_ids or set())
        raw_requests: list[object] = out["evidenceRequests"] if isinstance(out["evidenceRequests"], list) else []
        checked: list[object] = [r for r in raw_requests if isinstance(r, dict)]
        return {"proposals": list(proposals), "evidenceRequests": checked}


Stage = Literal["decompose", "analyze", "expand"]
