# SPDX-License-Identifier: Apache-2.0
"""
Per-token logprobs surface (Plan-1.5 patch #2 / realign#1251).

Scope: schema + scheduler extraction helper + server payload builder. Full
HTTP-level integration coverage is intentionally out of scope per patch (see
``.claude/PROJECT.md`` Scope rules) -- this exercises every layer touched by
the patch in isolation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
)
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import RequestOutput, SamplingParams
from vllm_mlx.scheduler import _extract_logprob_entry
from vllm_mlx.server import _build_choice_logprobs


# ---------------------------------------------------------------------------
# Schema: ChatCompletionRequest validation
# ---------------------------------------------------------------------------


def _msg() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_request_defaults_have_no_logprobs():
    req = ChatCompletionRequest(model="m", messages=_msg())
    assert req.logprobs is None
    assert req.top_logprobs is None


def test_request_accepts_logprobs_true():
    req = ChatCompletionRequest(model="m", messages=_msg(), logprobs=True)
    assert req.logprobs is True
    assert req.top_logprobs is None


def test_request_accepts_top_logprobs_with_logprobs_true():
    req = ChatCompletionRequest(
        model="m", messages=_msg(), logprobs=True, top_logprobs=5
    )
    assert req.top_logprobs == 5


def test_request_rejects_top_logprobs_without_logprobs():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=_msg(), top_logprobs=5)


def test_request_rejects_top_logprobs_above_20():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m", messages=_msg(), logprobs=True, top_logprobs=25
        )


def test_request_allows_top_logprobs_zero_without_logprobs():
    """top_logprobs=0 with logprobs=False should not trigger the validator."""
    req = ChatCompletionRequest(model="m", messages=_msg(), top_logprobs=0)
    assert req.top_logprobs == 0


# ---------------------------------------------------------------------------
# Schema: ChoiceLogprobs nesting + ChatCompletionChoice integration
# ---------------------------------------------------------------------------


def test_choice_logprobs_round_trip():
    entry = ChatCompletionTokenLogprob(
        token="hello",
        logprob=-0.5,
        bytes=[104, 101, 108, 108, 111],
        top_logprobs=[
            ChatCompletionTokenLogprob(
                token="hi", logprob=-1.2, bytes=[104, 105]
            )
        ],
    )
    wrap = ChoiceLogprobs(content=[entry])
    dumped = wrap.model_dump()
    assert dumped["content"][0]["token"] == "hello"
    assert dumped["content"][0]["top_logprobs"][0]["token"] == "hi"
    # Round-trip via dict
    rebuilt = ChoiceLogprobs.model_validate(dumped)
    assert rebuilt.content[0].logprob == -0.5
    assert rebuilt.content[0].top_logprobs[0].logprob == -1.2


def test_chat_completion_choice_omits_logprobs_when_none():
    from vllm_mlx.api.models import AssistantMessage

    choice = ChatCompletionChoice(
        index=0,
        message=AssistantMessage(content="ok"),
        finish_reason="stop",
    )
    assert choice.logprobs is None
    assert choice.model_dump()["logprobs"] is None


# ---------------------------------------------------------------------------
# Engine layer: SamplingParams + RequestOutput + GenerationOutput
# ---------------------------------------------------------------------------


def test_sampling_params_default_off():
    sp = SamplingParams()
    assert sp.logprobs is False
    assert sp.top_logprobs == 0


def test_sampling_params_accepts_logprobs():
    sp = SamplingParams(logprobs=True, top_logprobs=4)
    assert sp.logprobs is True
    assert sp.top_logprobs == 4


def test_request_output_carries_logprobs():
    entries = [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []}]
    ro = RequestOutput(request_id="r1", new_logprobs=entries, logprobs=entries)
    assert ro.new_logprobs is entries
    assert ro.logprobs is entries


def test_generation_output_carries_logprobs():
    entries = [{"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []}]
    go = GenerationOutput(text="a", logprobs=entries, new_logprobs=entries)
    assert go.logprobs == entries


# ---------------------------------------------------------------------------
# Scheduler: _extract_logprob_entry (the MLX -> OpenAI shape boundary)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Decode helper used by the scheduler when building logprob entries."""

    _vocab = {0: "<pad>", 1: "hi", 2: " ", 3: "world"}

    def decode(self, ids):
        return "".join(self._vocab.get(int(i), "?") for i in ids)


def _log_softmax(values: list[float]) -> list[float]:
    lse = math.log(sum(math.exp(v) for v in values))
    return [v - lse for v in values]


def test_extract_logprob_entry_chosen_only():
    vals = _log_softmax([2.0, 3.0, 1.0, 0.0])
    dist = mx.array(vals)
    entry = _extract_logprob_entry(1, dist, _FakeTokenizer(), top_logprobs=0)
    assert entry["token"] == "hi"
    assert entry["bytes"] == [104, 105]
    assert entry["top_logprobs"] == []
    assert math.isclose(entry["logprob"], vals[1], abs_tol=1e-5)


def test_extract_logprob_entry_with_top_k():
    vals = _log_softmax([2.0, 3.0, 1.0, 0.0])
    dist = mx.array(vals)
    entry = _extract_logprob_entry(1, dist, _FakeTokenizer(), top_logprobs=3)
    # Sorted descending by logprob -> tokens 1, 0, 2
    assert [t["token"] for t in entry["top_logprobs"]] == ["hi", "<pad>", " "]
    # Logprobs descending
    lps = [t["logprob"] for t in entry["top_logprobs"]]
    assert lps[0] >= lps[1] >= lps[2]


def test_extract_logprob_entry_top_k_caps_at_vocab():
    """top_logprobs > vocab_size should not raise."""
    vals = _log_softmax([0.5, 0.5])
    dist = mx.array(vals)
    entry = _extract_logprob_entry(0, dist, _FakeTokenizer(), top_logprobs=50)
    assert len(entry["top_logprobs"]) == 2


# ---------------------------------------------------------------------------
# Server: _build_choice_logprobs (engine dict -> Pydantic wrapper)
# ---------------------------------------------------------------------------


def test_build_choice_logprobs_returns_none_when_empty():
    assert _build_choice_logprobs(None) is None
    assert _build_choice_logprobs([]) is None


def test_build_choice_logprobs_populates_content():
    raw = [
        {"token": "a", "logprob": -0.1, "bytes": [97], "top_logprobs": []},
        {
            "token": "b",
            "logprob": -0.2,
            "bytes": [98],
            "top_logprobs": [
                {"token": "c", "logprob": -1.0, "bytes": [99]},
            ],
        },
    ]
    wrap = _build_choice_logprobs(raw)
    assert wrap is not None
    assert len(wrap.content) == 2
    assert wrap.content[0].token == "a"
    assert wrap.content[1].top_logprobs[0].token == "c"
    # Acceptance shape: one entry per emitted token (issue#1251)
    assert all(isinstance(e, ChatCompletionTokenLogprob) for e in wrap.content)


def test_build_choice_logprobs_tolerates_missing_optional_fields():
    """Engine dict may omit ``bytes`` or ``top_logprobs`` defensively."""
    raw = [{"token": "a", "logprob": -0.1}]
    wrap = _build_choice_logprobs(raw)
    assert wrap is not None
    assert wrap.content[0].bytes is None
    assert wrap.content[0].top_logprobs == []


# ---------------------------------------------------------------------------
# Issue acceptance criterion: shape matches tokens (realign#1251)
# ---------------------------------------------------------------------------


def test_end_to_end_shape_matches_completion_tokens():
    """Per acceptance criterion: ``len(logprobs.content) == completion_tokens``.

    This composes the in-memory layers (scheduler -> engine -> server) without
    requiring a live model.
    """
    tokenizer = _FakeTokenizer()
    chosen_tokens = [1, 2, 3]  # "hi", " ", "world"
    raw_engine_entries = []
    for tid in chosen_tokens:
        vals = _log_softmax([0.1 * i for i in range(4)])
        dist = mx.array(vals)
        raw_engine_entries.append(
            _extract_logprob_entry(tid, dist, tokenizer, top_logprobs=2)
        )

    wrap = _build_choice_logprobs(raw_engine_entries)
    assert wrap is not None
    assert len(wrap.content) == len(chosen_tokens)
    # Reconstructed text aligns with token decoding
    reconstructed = "".join(e.token for e in wrap.content)
    assert reconstructed == "hi world"


# ---------------------------------------------------------------------------
# Regression: RequestOutputCollector merge must preserve per-token logprobs.
#
# Before the wire-level fix, ``_merge_outputs`` only copied tokens/text and
# dropped ``logprobs``/``new_logprobs``. On the unary chat path the engine
# loop produces one ``RequestOutput`` per token; consumers drain via
# ``await event.wait()`` and the collector aggregates the puts that landed
# before the wait returned. With merge silently stripping logprobs, the
# final drained output's ``logprobs`` field was None even though the
# scheduler had populated it on every step. The smoke test
# ``test_logprobs_populates_content_array`` caught it; these unit tests
# pin the merge contract so the regression cannot creep back.
# ---------------------------------------------------------------------------


def _lp_entry(token: str) -> dict:
    return {"token": token, "logprob": -0.5, "bytes": list(token.encode()), "top_logprobs": []}


def test_collector_merge_preserves_cumulative_logprobs():
    """Final ``logprobs`` after merge must equal the latest cumulative array."""
    collector = RequestOutputCollector(aggregate=True)

    first = RequestOutput(
        request_id="r1",
        new_token_ids=[1],
        new_text="a",
        output_token_ids=[1],
        output_text="a",
        new_logprobs=[_lp_entry("a")],
        logprobs=[_lp_entry("a")],
    )
    second = RequestOutput(
        request_id="r1",
        new_token_ids=[2],
        new_text="b",
        output_token_ids=[1, 2],
        output_text="ab",
        finished=True,
        finish_reason="stop",
        new_logprobs=[_lp_entry("b")],
        logprobs=[_lp_entry("a"), _lp_entry("b")],
    )

    collector.put(first)
    collector.put(second)  # triggers _merge_outputs

    merged = collector.get_nowait()
    assert merged is not None
    # Cumulative logprobs survive the merge
    assert merged.logprobs is not None
    assert [e["token"] for e in merged.logprobs] == ["a", "b"]
    # Per-step deltas concatenated
    assert merged.new_logprobs is not None
    assert [e["token"] for e in merged.new_logprobs] == ["a", "b"]
    # Tokens and finished flag still propagate
    assert merged.new_token_ids == [1, 2]
    assert merged.new_text == "ab"
    assert merged.finished is True


def test_collector_merge_leaves_logprobs_none_when_disabled():
    """Requests that did not opt in still see ``logprobs is None`` after merge."""
    collector = RequestOutputCollector(aggregate=True)

    first = RequestOutput(request_id="r1", new_token_ids=[1], new_text="a", output_text="a")
    second = RequestOutput(
        request_id="r1",
        new_token_ids=[2],
        new_text="b",
        output_text="ab",
        finished=True,
    )
    collector.put(first)
    collector.put(second)
    merged = collector.get_nowait()
    assert merged is not None
    assert merged.logprobs is None
    assert merged.new_logprobs is None
