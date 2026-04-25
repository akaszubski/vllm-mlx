# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prompt optimizer."""

import dataclasses
import json
from argparse import Namespace

import pytest

from vllm_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDef,
)
from vllm_mlx.optimizer import (
    OptimizerConfig,
    ToolChoiceMismatchError,
    build_config_from_args,
    filter_tools_by_allowlist,
    optimize_request,
    simplify_schema,
    stub_tools,
    validate_tool_choice,
)
from vllm_mlx.optimizer.tool_stubber import STUBS


def _tool(name: str, description: str = "", schema: dict | None = None) -> AnthropicToolDef:
    return AnthropicToolDef(
        name=name, description=description, input_schema=schema or {}
    )


def _request(
    tools: list[AnthropicToolDef] | None = None,
    tool_choice: dict | None = None,
) -> AnthropicRequest:
    return AnthropicRequest(
        model="test",
        messages=[AnthropicMessage(role="user", content="hi")],
        max_tokens=16,
        tools=tools,
        tool_choice=tool_choice,
    )


class TestFilterToolsByAllowlist:
    def test_none_allowlist_passes_through(self):
        tools = [_tool("Bash"), _tool("Read")]
        out, removed = filter_tools_by_allowlist(tools, None)
        assert out == tools
        assert removed == 0

    def test_empty_allowlist_drops_everything(self):
        tools = [_tool("Bash"), _tool("Read")]
        out, removed = filter_tools_by_allowlist(tools, [])
        assert out == []
        assert removed == 2

    def test_case_insensitive_match(self):
        tools = [_tool("Bash"), _tool("Read"), _tool("Glob")]
        out, removed = filter_tools_by_allowlist(tools, ["bash", "READ"])
        assert {t.name for t in out} == {"Bash", "Read"}
        assert removed == 1

    def test_unknown_names_in_allowlist_are_ignored(self):
        tools = [_tool("Bash")]
        out, _ = filter_tools_by_allowlist(tools, ["Bash", "DoesNotExist"])
        assert len(out) == 1


class TestSimplifySchema:
    def test_strips_description_and_examples(self):
        schema = {
            "type": "object",
            "description": "A long description",
            "examples": [{"x": 1}],
            "properties": {"x": {"type": "integer", "description": "an int"}},
            "required": ["x"],
        }
        out = simplify_schema(schema)
        assert "description" not in out
        assert "examples" not in out
        assert out["type"] == "object"
        assert out["required"] == ["x"]
        assert out["properties"]["x"] == {"type": "integer"}

    def test_recurses_into_items(self):
        schema = {"type": "array", "items": {"type": "string", "description": "x"}}
        assert simplify_schema(schema) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_recurses_into_anyof(self):
        schema = {
            "anyOf": [
                {"type": "string", "description": "x"},
                {"type": "integer", "description": "y"},
            ]
        }
        assert simplify_schema(schema) == {
            "anyOf": [{"type": "string"}, {"type": "integer"}]
        }

    def test_non_dict_input_returns_unchanged(self):
        assert simplify_schema("not a schema") == "not a schema"
        assert simplify_schema(None) is None
        assert simplify_schema(42) == 42

    def test_unknown_keys_dropped(self):
        schema = {"type": "string", "title": "x", "$comment": "y"}
        assert simplify_schema(schema) == {"type": "string"}


class TestStubTools:
    def test_replaces_known_tool_description(self):
        tool = _tool(
            "Bash",
            description="A very long description that goes on and on and on",
            schema={"type": "object", "description": "a"},
        )
        out, stubbed = stub_tools([tool])
        assert stubbed == 1
        assert out[0].description == STUBS["Bash"]
        # schema should be simplified (description stripped)
        assert "description" not in out[0].input_schema

    def test_keeps_short_description_unchanged(self):
        # Description shorter than the stub is kept as-is (avoid making it worse).
        short = "x"
        tool = _tool("Bash", description=short)
        out, stubbed = stub_tools([tool])
        assert stubbed == 0
        assert out[0].description == short

    def test_unknown_tool_name_keeps_description(self):
        tool = _tool("MysteryTool", description="long description here")
        out, stubbed = stub_tools([tool])
        assert stubbed == 0
        assert out[0].description == "long description here"

    def test_simplifies_schema_even_when_no_stub(self):
        tool = _tool(
            "MysteryTool",
            description="x",
            schema={"type": "object", "description": "drop me"},
        )
        out, _ = stub_tools([tool])
        assert "description" not in out[0].input_schema


class TestOptimizeRequest:
    def test_disabled_is_passthrough(self):
        req = _request(tools=[_tool("Bash"), _tool("Read")])
        out, stats = optimize_request(req, OptimizerConfig(enabled=False))
        assert out is req
        assert stats.tools_after == 2

    def test_enabled_no_transforms_is_noop(self):
        req = _request(tools=[_tool("Bash")])
        cfg = OptimizerConfig(enabled=True, tool_allowlist=None, stub_tools=False)
        out, stats = optimize_request(req, cfg)
        assert out is req
        assert stats.tools_before == 1
        assert stats.tools_after == 1

    def test_no_tools_in_request_is_safe(self):
        req = _request(tools=None)
        out, stats = optimize_request(
            req, OptimizerConfig(enabled=True, stub_tools=True)
        )
        assert out is req
        assert stats.tools_before == 0

    def test_empty_tools_list_is_safe(self):
        req = _request(tools=[])
        out, stats = optimize_request(
            req, OptimizerConfig(enabled=True, stub_tools=True)
        )
        assert out is req
        assert stats.tools_before == 0
        assert stats.tools_after == 0

    def test_allowlist_filters(self):
        req = _request(tools=[_tool("Bash"), _tool("Read"), _tool("Glob")])
        cfg = OptimizerConfig(enabled=True, tool_allowlist=("Bash",))
        out, stats = optimize_request(req, cfg)
        assert out is not req
        assert [t.name for t in out.tools] == ["Bash"]
        assert stats.tools_before == 3
        assert stats.tools_after == 1

    def test_stub_tools_replaces_descriptions(self):
        long_desc = "x" * 500
        req = _request(tools=[_tool("Bash", description=long_desc)])
        cfg = OptimizerConfig(enabled=True, stub_tools=True)
        out, stats = optimize_request(req, cfg)
        assert out is not req
        assert out.tools[0].description != long_desc
        assert stats.tools_stubbed == 1
        assert stats.descriptions_chars_after < stats.descriptions_chars_before

    def test_allowlist_then_stub_chain(self):
        req = _request(
            tools=[
                _tool("Bash", description="a" * 200),
                _tool("Read", description="b" * 200),
                _tool("Glob", description="c" * 200),
            ]
        )
        cfg = OptimizerConfig(
            enabled=True, tool_allowlist=("Bash", "Read"), stub_tools=True
        )
        out, stats = optimize_request(req, cfg)
        assert {t.name for t in out.tools} == {"Bash", "Read"}
        assert stats.tools_stubbed == 2

    def test_determinism_in_process(self):
        # Same input -> same output; required for prefix-cache compatibility.
        tools = [
            _tool("Bash", description="a" * 200, schema={"type": "object", "description": "x"}),
            _tool("Read", description="b" * 200),
        ]
        req1 = _request(tools=tools)
        req2 = _request(tools=tools)
        cfg = OptimizerConfig(
            enabled=True, tool_allowlist=("Bash", "Read"), stub_tools=True
        )
        out1, _ = optimize_request(req1, cfg)
        out2, _ = optimize_request(req2, cfg)
        assert out1.model_dump() == out2.model_dump()

    def test_determinism_byte_stable_serialization(self):
        # Stronger determinism: a sorted-keys JSON dump must be byte-identical.
        # Catches regressions where pydantic minor-version changes alter
        # field ordering in a way that bypasses the prefix cache.
        tools = [_tool("Bash", description="a" * 200), _tool("Read", description="b" * 200)]
        cfg = OptimizerConfig(
            enabled=True, tool_allowlist=("Bash", "Read"), stub_tools=True
        )
        out1, _ = optimize_request(_request(tools=tools), cfg)
        out2, _ = optimize_request(_request(tools=tools), cfg)
        s1 = json.dumps(out1.model_dump(), sort_keys=True, ensure_ascii=False)
        s2 = json.dumps(out2.model_dump(), sort_keys=True, ensure_ascii=False)
        assert s1 == s2

    def test_does_not_mutate_original(self):
        original_desc = "x" * 500
        tools = [_tool("Bash", description=original_desc)]
        req = _request(tools=tools)
        cfg = OptimizerConfig(enabled=True, stub_tools=True)
        optimize_request(req, cfg)
        assert req.tools[0].description == original_desc


class TestToolChoiceValidation:
    def test_no_tool_choice_is_fine(self):
        validate_tool_choice(None, [_tool("Bash")])

    def test_auto_tool_choice_is_fine(self):
        validate_tool_choice({"type": "auto"}, [_tool("Bash")])

    def test_named_tool_choice_present_is_fine(self):
        validate_tool_choice(
            {"type": "tool", "name": "Bash"}, [_tool("Bash"), _tool("Read")]
        )

    def test_named_tool_choice_dropped_raises(self):
        with pytest.raises(ToolChoiceMismatchError) as ei:
            validate_tool_choice({"type": "tool", "name": "Glob"}, [_tool("Bash")])
        assert "Glob" in str(ei.value)
        assert "Bash" in str(ei.value)

    def test_optimize_request_raises_on_mismatch(self):
        # End-to-end: tool_choice points to a tool the allowlist removes.
        req = _request(
            tools=[_tool("Bash"), _tool("Glob")],
            tool_choice={"type": "tool", "name": "Glob"},
        )
        cfg = OptimizerConfig(enabled=True, tool_allowlist=("Bash",))
        with pytest.raises(ToolChoiceMismatchError):
            optimize_request(req, cfg)


class TestOptimizerConfig:
    def test_is_noop_when_disabled(self):
        assert OptimizerConfig(enabled=False).is_noop

    def test_is_noop_when_enabled_but_no_transforms(self):
        assert OptimizerConfig(enabled=True).is_noop

    def test_is_not_noop_with_allowlist(self):
        assert not OptimizerConfig(
            enabled=True, tool_allowlist=("Bash",)
        ).is_noop

    def test_is_not_noop_with_stub(self):
        assert not OptimizerConfig(enabled=True, stub_tools=True).is_noop

    def test_frozen(self):
        cfg = OptimizerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = True  # type: ignore[misc]


class TestBuildConfigFromArgs:
    def test_optimize_prompts_off_returns_disabled(self):
        cfg = build_config_from_args(Namespace(optimize_prompts=False))
        assert not cfg.enabled
        assert cfg.is_noop

    def test_missing_attrs_default_safely(self):
        # The CLI subcommand path uses getattr with defaults; mirror that.
        cfg = build_config_from_args(Namespace())
        assert not cfg.enabled

    def test_full_config(self):
        cfg = build_config_from_args(
            Namespace(
                optimize_prompts=True,
                optimize_tool_allowlist="Bash, Read , ,Glob",
                optimize_stub_tools=True,
            )
        )
        assert cfg.enabled
        assert cfg.tool_allowlist == ("Bash", "Read", "Glob")
        assert cfg.stub_tools is True

    def test_empty_allowlist_string_is_none(self):
        cfg = build_config_from_args(
            Namespace(
                optimize_prompts=True,
                optimize_tool_allowlist="",
                optimize_stub_tools=False,
            )
        )
        assert cfg.tool_allowlist is None


class TestStubsTableRegression:
    """Pin the STUBS keyset so renames break CI rather than silently rotting."""

    EXPECTED_STUBS_KEYS = frozenset({
        # Core file + shell
        "Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit",
        # Background processes + monitoring
        "BashOutput", "KillShell", "Monitor", "PushNotification",
        # Web
        "WebFetch", "WebSearch",
        # Subagents + tasks
        "Task", "TodoWrite",
        "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskOutput", "TaskStop",
        # Plan mode + user prompts
        "ExitPlanMode", "EnterPlanMode", "AskUserQuestion",
        # Skills + slash commands
        "Skill", "SlashCommand", "ToolSearch",
        # Worktrees + scheduling + remote
        "EnterWorktree", "ExitWorktree",
        "CronCreate", "CronDelete", "CronList", "RemoteTrigger",
        # MCP introspection
        "ListMcpResourcesTool", "ReadMcpResourceTool",
    })

    def test_keyset_matches_pinned(self):
        # If this fails, Claude Code added/renamed/removed a tool. Decide
        # whether to add a new stub, remove a stale one, or simply update
        # EXPECTED_STUBS_KEYS — but the change should be deliberate.
        actual = frozenset(STUBS.keys())
        added = actual - self.EXPECTED_STUBS_KEYS
        removed = self.EXPECTED_STUBS_KEYS - actual
        assert not added and not removed, (
            f"STUBS keyset drifted. Added: {sorted(added)}; "
            f"Removed: {sorted(removed)}. Update EXPECTED_STUBS_KEYS deliberately."
        )

    def test_every_stub_is_shorter_than_a_typical_description(self):
        # Sanity check: stubs should actually be stubs (under ~80 chars).
        # Keeps the table from being accidentally re-bloated.
        for name, stub in STUBS.items():
            assert len(stub) <= 80, f"stub for {name!r} is too long: {stub!r}"
