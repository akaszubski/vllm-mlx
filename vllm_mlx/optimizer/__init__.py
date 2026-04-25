# SPDX-License-Identifier: Apache-2.0
"""
Prompt optimizer for the Anthropic /v1/messages endpoint.

Two transformations, both deterministic so vllm-mlx's prefix cache hits cleanly:
  - Tool allowlist filtering (drop tools not on the allowlist)
  - Tool description stubbing + JSON schema simplification

Off by default; enabled via --optimize-prompts.
"""

from __future__ import annotations

import dataclasses
import logging
from argparse import Namespace
from dataclasses import dataclass

from ..api.anthropic_models import AnthropicRequest, AnthropicToolDef
from .tool_allowlist import filter_tools_by_allowlist
from .tool_stubber import STUBS, simplify_schema, stub_tools

__all__ = [
    "OptimizerConfig",
    "OptimizerStats",
    "ToolChoiceMismatchError",
    "build_config_from_args",
    "filter_tools_by_allowlist",
    "optimize_request",
    "simplify_schema",
    "stub_tools",
    "validate_tool_choice",
]

logger = logging.getLogger(__name__)

# Tracks unknown tool names already warned about so we log each name once
# per process rather than spamming on every request.
_warned_unknown_tools: set[str] = set()


class ToolChoiceMismatchError(ValueError):
    """Raised when tool_choice references a tool removed by the allowlist."""


@dataclass(frozen=True)
class OptimizerConfig:
    """Runtime configuration for the request optimizer."""

    enabled: bool = False
    tool_allowlist: tuple[str, ...] | None = None
    stub_tools: bool = False

    @property
    def is_noop(self) -> bool:
        return not self.enabled or (
            self.tool_allowlist is None and not self.stub_tools
        )


@dataclasses.dataclass
class OptimizerStats:
    """Diagnostic counts produced by a single optimize_request call."""

    tools_before: int = 0
    tools_after: int = 0
    tools_stubbed: int = 0
    descriptions_chars_before: int = 0
    descriptions_chars_after: int = 0


def build_config_from_args(args: Namespace) -> OptimizerConfig:
    """Construct an OptimizerConfig from a parsed argparse Namespace.

    Used by both ``vllm_mlx.cli.serve_command`` and ``vllm_mlx.server.main``
    to keep flag parsing in one place. Returns a disabled (no-op) config
    when ``--optimize-prompts`` was not set.
    """
    if not getattr(args, "optimize_prompts", False):
        return OptimizerConfig()

    raw = getattr(args, "optimize_tool_allowlist", None)
    allowlist: tuple[str, ...] | None = None
    if raw:
        allowlist = tuple(name.strip() for name in raw.split(",") if name.strip())
    return OptimizerConfig(
        enabled=True,
        tool_allowlist=allowlist,
        stub_tools=getattr(args, "optimize_stub_tools", False),
    )


def validate_tool_choice(
    tool_choice: dict | None, tools: list[AnthropicToolDef]
) -> None:
    """Raise if ``tool_choice`` names a tool that isn't in ``tools``.

    Anthropic's tool_choice can be ``{"type": "tool", "name": "X"}``. If the
    optimizer's allowlist has dropped X, the model would be told to call a
    tool that isn't in the request — better to fail fast with a 400 than to
    confuse the model.

    No-op for tool_choice values without a ``name`` field (e.g. ``auto``,
    ``any``, ``none``) and when tool_choice is None.
    """
    if not tool_choice:
        return
    name = tool_choice.get("name")
    if not name:
        return
    if not any(t.name == name for t in tools):
        raise ToolChoiceMismatchError(
            f"tool_choice names tool {name!r} but no such tool is in the "
            f"request (post-allowlist tools: {[t.name for t in tools]})"
        )


def _warn_unknown_tools(tools: list[AnthropicToolDef]) -> None:
    """Log a one-time warning for each tool name not in the STUBS table.

    Surfaces drift when Claude Code ships a renamed/new tool that the
    hand-maintained STUBS table hasn't caught up to.
    """
    for tool in tools:
        if tool.name not in STUBS and tool.name not in _warned_unknown_tools:
            _warned_unknown_tools.add(tool.name)
            logger.warning(
                "[OPTIMIZER] no stub registered for tool %r — falling back "
                "to original description (schema simplification still runs)",
                tool.name,
            )


def optimize_request(
    request: AnthropicRequest,
    config: OptimizerConfig,
) -> tuple[AnthropicRequest, OptimizerStats]:
    """Apply optimizer transformations to an AnthropicRequest.

    Returns a new request (via Pydantic ``model_copy``) and a stats object
    suitable for logging. The original request is never mutated.

    Raises ToolChoiceMismatchError if tool_choice names a tool the allowlist
    drops.
    """
    stats = OptimizerStats()
    tools_in: list[AnthropicToolDef] = list(request.tools or [])
    stats.tools_before = stats.tools_after = len(tools_in)
    stats.descriptions_chars_before = stats.descriptions_chars_after = sum(
        len(t.description or "") for t in tools_in
    )

    if config.is_noop or not tools_in:
        return request, stats

    tools = tools_in
    if config.tool_allowlist is not None:
        tools, _ = filter_tools_by_allowlist(tools, config.tool_allowlist)
        validate_tool_choice(request.tool_choice, tools)

    if config.stub_tools and tools:
        _warn_unknown_tools(tools)
        tools, stubbed_count = stub_tools(tools)
        stats.tools_stubbed = stubbed_count

    stats.tools_after = len(tools)
    stats.descriptions_chars_after = sum(len(t.description or "") for t in tools)

    return request.model_copy(update={"tools": tools}), stats
