"""Application-level authorization policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .config import get_data_root


class Capability(StrEnum):
    RESEARCH = "research"
    BROKER_MARKET_READ = "broker_market_read"
    PORTFOLIO_READ = "portfolio_read"


@dataclass(frozen=True)
class RunLimits:
    """Run budget for a single research run.

    No limit by default: a research run continues while materially useful
    (the Director owns completion). An explicitly configured int/float still
    enforces. max_tool_result_bytes mirrors tool_render.MAX_TOOL_MESSAGE_BYTES.
    """

    max_tool_calls: int | None = None
    max_runtime: float | None = None  # seconds
    max_tool_result_bytes: int = 64 * 1024  # == tool_render.MAX_TOOL_MESSAGE_BYTES
    # Retrieval unbounded: ledger + paged reads own safety; counters stay telemetry.
    max_evidence_tokens: int | None = None


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: frozenset[str] | None = None  # None -> capability-derived registry
    max_arguments_bytes: int = 8 * 1024
    deny_unpermitted: bool = True


@dataclass(frozen=True)
class RequestContext:
    principal_id: str
    capabilities: frozenset[Capability]
    tool_policy: ToolPolicy = ToolPolicy()
    data_root: Path = field(default_factory=get_data_root)
    as_of: str | None = None
    run_limits: RunLimits = RunLimits()
    source_policy: dict[str, object] | None = None
    research_session_id: str | None = None  # set on research-bound dispatches; None outside a session


def scoped_context(context: RequestContext, source_policy: dict[str, object] | None) -> RequestContext:
    """Copy a context with the kernel-persisted session source_policy attached."""
    return RequestContext(
        principal_id=context.principal_id,
        capabilities=context.capabilities,
        tool_policy=context.tool_policy,
        data_root=context.data_root,
        as_of=context.as_of,
        run_limits=context.run_limits,
        source_policy=dict(source_policy) if isinstance(source_policy, dict) else None,
        research_session_id=context.research_session_id,
    )


def context_allows_tool(context: RequestContext, name: str) -> bool:
    """Kernel gate: capability permit + source_policy allowlist (denied wins).

    SEC-only (allowed [sec]) denies FINRA/Web/Market/Analyst tools even when
    the capability registry lists them; allowed discovery tools (browse_tools
    et al) stay listed but their inner call_tool target is gated the same way.
    """
    from .security.action_policy import source_denied_reason

    capability = _tool_capability(name)
    if capability is None or capability not in context.capabilities:
        return False
    if context.tool_policy.allowed_tools is not None and name not in context.tool_policy.allowed_tools:
        return not context.tool_policy.deny_unpermitted
    return source_denied_reason(name, context.source_policy) is None


def _tool_capability(name: str) -> Capability | None:
    """Application capability for one tool name (None when unregistered)."""
    try:
        from .tools import TOOL_CAPABILITIES
    except ImportError:
        return None
    capability = TOOL_CAPABILITIES.get(name)
    return capability if isinstance(capability, Capability) else None


LOCAL_CONTEXT = RequestContext(
    principal_id="local",
    capabilities=frozenset({Capability.RESEARCH}),
)

LOCAL_BROKER_CONTEXT = RequestContext(
    principal_id="local-broker",
    capabilities=frozenset({Capability.RESEARCH, Capability.BROKER_MARKET_READ, Capability.PORTFOLIO_READ}),
)
