# SPDX-License-Identifier: Apache-2.0
"""Tests for the auto thinking-gate, parser-mode hint, and thinking-leak warning.

These three layers work together to make tool-using requests reliably produce
committed answers (text or tool_use) instead of `<think>`-only responses that
break agent loops:

  - ``_apply_thinking_gate_for_tools``: when request.tools is non-empty AND
    enable_thinking isn't already set, default it to False before the engine
    sees the request.
  - ``_reasoning_parser_starts_in_content``: when enable_thinking=False has
    been resolved, tell the streaming parser to start in content mode so plain
    text deltas aren't misclassified as reasoning.
  - ``_warn_thinking_leak_once``: surface a one-time-per-process warning when
    a model emits reasoning but no committed answer (drift detector for
    configurations that bypass the gate).
"""

import logging

import vllm_mlx.server as srv
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser


# =============================================================================
# Layer A: thinking-gate helper
# =============================================================================


class TestThinkingGateForTools:
    def test_tools_present_and_kwarg_unset_sets_false(self, caplog):
        with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
            out = srv._apply_thinking_gate_for_tools(
                request_tools=[{"name": "Bash"}], resolved_kwargs={}
            )
        assert out == {"enable_thinking": False}
        assert any("[thinking-gate]" in rec.message for rec in caplog.records)

    def test_tools_present_explicit_kwarg_wins(self, caplog):
        with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
            out = srv._apply_thinking_gate_for_tools(
                request_tools=[{"name": "Bash"}],
                resolved_kwargs={"enable_thinking": True},
            )
        assert out == {"enable_thinking": True}
        assert not any("[thinking-gate]" in rec.message for rec in caplog.records)

    def test_server_default_in_resolved_kwargs_wins(self, caplog):
        # Server default --default-chat-template-kwargs '{"enable_thinking": true}'
        # already merged the key. Gate must not override it.
        with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
            out = srv._apply_thinking_gate_for_tools(
                request_tools=[{"name": "Bash"}],
                resolved_kwargs={"enable_thinking": True, "other_kw": 42},
            )
        assert out == {"enable_thinking": True, "other_kw": 42}
        assert not any("[thinking-gate]" in rec.message for rec in caplog.records)

    def test_no_tools_no_op(self, caplog):
        with caplog.at_level(logging.INFO, logger="vllm_mlx.server"):
            out = srv._apply_thinking_gate_for_tools(
                request_tools=None, resolved_kwargs={}
            )
        assert out == {}
        assert not any("[thinking-gate]" in rec.message for rec in caplog.records)

    def test_empty_tools_list_no_op(self):
        out = srv._apply_thinking_gate_for_tools(
            request_tools=[], resolved_kwargs={}
        )
        assert out == {}

    def test_does_not_mutate_input_dict(self):
        original = {"foo": "bar"}
        out = srv._apply_thinking_gate_for_tools(
            request_tools=[{"name": "Bash"}], resolved_kwargs=original
        )
        assert original == {"foo": "bar"}  # untouched
        assert out == {"foo": "bar", "enable_thinking": False}


# =============================================================================
# Layer A.5: parser-mode hint helper
# =============================================================================


class TestReasoningParserStartsInContent:
    def test_returns_true_when_enable_thinking_false(self):
        assert srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_returns_false_when_enable_thinking_true(self):
        assert not srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": {"enable_thinking": True}}
        )

    def test_returns_false_when_kwarg_absent(self):
        assert not srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": {"other_kw": "x"}}
        )

    def test_returns_false_when_chat_template_kwargs_absent(self):
        assert not srv._reasoning_parser_starts_in_content({})

    def test_returns_false_for_truthy_non_false_values(self):
        # Strict identity check on `is False`; truthy values that aren't
        # exactly False shouldn't trigger content-mode start.
        assert not srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": {"enable_thinking": 0}}
        )
        assert not srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": {"enable_thinking": None}}
        )

    def test_handles_non_dict_kwargs_gracefully(self):
        # Defensive: shouldn't blow up if something puts a non-dict here.
        assert not srv._reasoning_parser_starts_in_content(
            {"chat_template_kwargs": "garbage"}
        )


# =============================================================================
# Layer A.5: parser respects start_in_content_mode
# =============================================================================


class TestQwen3ParserStartInContentMode:
    """The streaming parser's pre_think phase classifies untagged deltas as
    reasoning. With start_in_content_mode=True, it must classify them as
    content from the first delta. Without it, existing behavior preserved."""

    def test_default_classifies_untagged_delta_as_reasoning(self):
        parser = Qwen3ReasoningParser()
        parser.reset_state()  # default: start_in_content_mode=False
        msg = parser.extract_reasoning_streaming("", "Hello", "Hello")
        assert msg is not None
        assert msg.reasoning == "Hello"
        assert msg.content is None

    def test_start_in_content_mode_classifies_first_delta_as_content(self):
        parser = Qwen3ReasoningParser()
        parser.reset_state(start_in_content_mode=True)
        msg = parser.extract_reasoning_streaming("", "Hello", "Hello")
        assert msg is not None
        assert msg.content == "Hello"
        assert msg.reasoning is None

    def test_start_in_content_mode_full_stream_all_content(self):
        parser = Qwen3ReasoningParser()
        parser.reset_state(start_in_content_mode=True)
        accumulated = ""
        deltas = ["Hello", "!", " How", " are", " you", "?"]
        for d in deltas:
            previous = accumulated
            accumulated += d
            msg = parser.extract_reasoning_streaming(previous, accumulated, d)
            assert msg is not None
            assert msg.content == d
            assert msg.reasoning is None

    def test_default_mode_handles_explicit_think_tags(self):
        # Existing behavior: when start_in_content_mode=False and the model
        # emits explicit <think>X</think>content, parser correctly extracts.
        parser = Qwen3ReasoningParser()
        parser.reset_state()
        # Feed full output as one delta to keep the test simple.
        full = "<think>reasoning here</think>final answer"
        msg = parser.extract_reasoning_streaming("", full, full)
        assert msg is not None
        assert msg.reasoning == "reasoning here"
        assert msg.content == "final answer"

    def test_start_in_content_mode_preserved_across_resets(self):
        # Each reset is independent; later defaults don't leak prior mode.
        parser = Qwen3ReasoningParser()
        parser.reset_state(start_in_content_mode=True)
        msg1 = parser.extract_reasoning_streaming("", "A", "A")
        assert msg1.content == "A"

        parser.reset_state()  # default
        msg2 = parser.extract_reasoning_streaming("", "B", "B")
        assert msg2.reasoning == "B"


# =============================================================================
# Layer C: thinking-leak warning fires once per model
# =============================================================================


class TestThinkingLeakWarning:
    def setup_method(self):
        # Clear the warned set so tests are independent.
        srv._thinking_leak_warned.clear()

    def test_fires_once_for_a_model(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
            srv._warn_thinking_leak_once("test-model", 42)
            srv._warn_thinking_leak_once("test-model", 99)
        leak_records = [r for r in caplog.records if "[thinking-leak]" in r.message]
        assert len(leak_records) == 1
        assert "test-model" in leak_records[0].message
        assert "42" in leak_records[0].message  # the first call's char count

    def test_fires_separately_per_model(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
            srv._warn_thinking_leak_once("model-a", 10)
            srv._warn_thinking_leak_once("model-b", 20)
        leak_records = [r for r in caplog.records if "[thinking-leak]" in r.message]
        assert len(leak_records) == 2

    def test_no_warning_for_empty_model_name(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
            srv._warn_thinking_leak_once("", 10)
            srv._warn_thinking_leak_once(None, 10)  # type: ignore[arg-type]
        leak_records = [r for r in caplog.records if "[thinking-leak]" in r.message]
        assert len(leak_records) == 0


# =============================================================================
# Composability: gate + parser hint compose for the canonical flow
# =============================================================================


class TestGateAndParserCompose:
    """End-to-end of the request-prep → parser-init flow without an engine."""

    def test_tool_request_propagates_through_gate_into_parser_hint(self):
        # Given: a tool-using request hits the gate
        resolved = srv._apply_thinking_gate_for_tools(
            request_tools=[{"name": "Bash"}], resolved_kwargs={}
        )
        # The kwargs that would land in chat_kwargs:
        chat_kwargs = {"chat_template_kwargs": resolved}
        # Then: the parser hint reads enable_thinking=False
        assert srv._reasoning_parser_starts_in_content(chat_kwargs) is True

    def test_chat_request_no_tools_does_not_force_content_mode(self):
        # Chat-style request (no tools): gate doesn't fire, parser stays in
        # default mode (preserves thinking-rendering behavior for chat).
        resolved = srv._apply_thinking_gate_for_tools(
            request_tools=None, resolved_kwargs={}
        )
        chat_kwargs = {"chat_template_kwargs": resolved} if resolved else {}
        assert srv._reasoning_parser_starts_in_content(chat_kwargs) is False

    def test_explicit_thinking_with_tools_skips_gate_and_parser_hint(self):
        # User explicitly opts into thinking with tools — no auto-gate, no
        # forced content mode. Their explicit choice wins.
        resolved = srv._apply_thinking_gate_for_tools(
            request_tools=[{"name": "Bash"}],
            resolved_kwargs={"enable_thinking": True},
        )
        chat_kwargs = {"chat_template_kwargs": resolved}
        assert srv._reasoning_parser_starts_in_content(chat_kwargs) is False
