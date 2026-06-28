# SPDX-License-Identifier: Apache-2.0
"""Spec-validator tests for SamplingParams.n (group sampling).

These tests are written BLIND to the implementation: they encode the
acceptance criteria from the feature spec only. They MUST pass against any
correct implementation, and MUST fail against a non-compliant one.
"""

from __future__ import annotations

import platform
import sys
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
)

# Server import (for AC6/AC7/AC8) is Apple-Silicon-only.
_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# Schema-level tests (AC1..AC5, AC12 partial) — no server required
# ---------------------------------------------------------------------------


def _chat_kwargs(**extra):
    return dict(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


def _completion_kwargs(**extra):
    return dict(model="m", prompt="hi", **extra)


# --- AC1 ---
def test_ac1_chat_default_n_is_1_and_best_of_none():
    req = ChatCompletionRequest(**_chat_kwargs())
    assert req.n == 1
    assert req.best_of is None


# --- AC2 ---
def test_ac2_chat_accepts_n_4():
    req = ChatCompletionRequest(**_chat_kwargs(n=4))
    assert req.n == 4


# --- AC3 ---
def test_ac3_chat_rejects_n_0():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_chat_kwargs(n=0))


# --- AC4 ---
def test_ac4_chat_rejects_n_65():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_chat_kwargs(n=65))


# --- AC5 (mirrored for CompletionRequest) ---
def test_ac5_completion_default_n_is_1_and_best_of_none():
    req = CompletionRequest(**_completion_kwargs())
    assert req.n == 1
    assert req.best_of is None


def test_ac5_completion_accepts_n_4():
    req = CompletionRequest(**_completion_kwargs(n=4))
    assert req.n == 4


def test_ac5_completion_rejects_n_0():
    with pytest.raises(ValidationError):
        CompletionRequest(**_completion_kwargs(n=0))


def test_ac5_completion_rejects_n_65():
    with pytest.raises(ValidationError):
        CompletionRequest(**_completion_kwargs(n=65))


# --- AC12: best_of bounds and no `seed` field added ---
def test_ac12_chat_best_of_bounds():
    # best_of accepts None (default), 1, 64; rejects 0 and 65.
    assert ChatCompletionRequest(**_chat_kwargs(best_of=1)).best_of == 1
    assert ChatCompletionRequest(**_chat_kwargs(best_of=64)).best_of == 64
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_chat_kwargs(best_of=0))
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_chat_kwargs(best_of=65))


def test_ac12_completion_best_of_bounds():
    assert CompletionRequest(**_completion_kwargs(best_of=1)).best_of == 1
    assert CompletionRequest(**_completion_kwargs(best_of=64)).best_of == 64
    with pytest.raises(ValidationError):
        CompletionRequest(**_completion_kwargs(best_of=0))
    with pytest.raises(ValidationError):
        CompletionRequest(**_completion_kwargs(best_of=65))


def test_ac12_no_seed_field_added():
    """Spec says no `seed` field added by this patch."""
    chat_fields = set(ChatCompletionRequest.model_fields.keys())
    comp_fields = set(CompletionRequest.model_fields.keys())
    assert "seed" not in chat_fields, "spec forbids new `seed` field on ChatCompletionRequest"
    assert "seed" not in comp_fields, "spec forbids new `seed` field on CompletionRequest"


def test_ac12_n_field_type_and_default():
    chat_field = ChatCompletionRequest.model_fields["n"]
    comp_field = CompletionRequest.model_fields["n"]
    assert chat_field.default == 1
    assert comp_field.default == 1
    # Type annotation should be int (not Optional[int]).
    assert chat_field.annotation is int
    assert comp_field.annotation is int


# ---------------------------------------------------------------------------
# Handler-level tests (AC6, AC7, AC8) — require server import
# ---------------------------------------------------------------------------

pytestmark_server = pytest.mark.skipif(
    not _IS_APPLE_SILICON,
    reason="Server import requires Apple Silicon (MLX dependency)",
)


@pytest.fixture()
def client():
    from vllm_mlx.server import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def server_state():
    """Reset module-level server state between tests."""
    if not _IS_APPLE_SILICON:
        yield
        return
    import vllm_mlx.server as srv

    saved = {
        "engine": srv._engine,
        "model_name": srv._model_name,
        "store": srv._responses_store,
        "store_max": srv._RESPONSES_STORE_MAX_SIZE,
        "api_key": srv._api_key,
        "default_ctk": getattr(srv, "_default_chat_template_kwargs", None),
    }
    srv._engine = None
    srv._model_name = "test-model"
    srv._responses_store = OrderedDict()
    srv._RESPONSES_STORE_MAX_SIZE = 1000
    srv._api_key = None
    srv._default_chat_template_kwargs = None
    try:
        yield
    finally:
        srv._engine = saved["engine"]
        srv._model_name = saved["model_name"]
        srv._responses_store = saved["store"]
        srv._RESPONSES_STORE_MAX_SIZE = saved["store_max"]
        srv._api_key = saved["api_key"]
        srv._default_chat_template_kwargs = saved["default_ctk"]


def _make_chat_output(text: str, prompt_tokens: int = 7, completion_tokens: int = 3):
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def _make_completion_output(text: str, prompt_tokens: int = 7, completion_tokens: int = 3):
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def _install_mock_chat_engine(monkeypatch, outputs):
    """Install a mocked engine for chat completions with the given outputs."""
    import vllm_mlx.server as srv

    engine = MagicMock()
    engine.model_name = "test-model"
    engine.preserve_native_tool_format = False
    engine.chat = AsyncMock(side_effect=list(outputs))
    srv._engine = engine

    async def _acquire(*args, **kwargs):
        return engine

    async def _release(*args, **kwargs):
        return None

    monkeypatch.setattr(srv, "_acquire_default_engine_for_request", _acquire)
    monkeypatch.setattr(srv, "_release_default_engine", _release)
    return engine


def _install_mock_completion_engine(monkeypatch, outputs):
    import vllm_mlx.server as srv

    engine = MagicMock()
    engine.model_name = "test-model"
    engine.generate = AsyncMock(side_effect=list(outputs))
    srv._engine = engine

    async def _acquire(*args, **kwargs):
        return engine

    async def _release(*args, **kwargs):
        return None

    monkeypatch.setattr(srv, "_acquire_default_engine_for_request", _acquire)
    monkeypatch.setattr(srv, "_release_default_engine", _release)
    return engine


# --- AC6: streaming + n>1 -> HTTP 400 with stream_n_unsupported code ---
@pytestmark_server
def test_ac6_chat_stream_with_n_gt_1_returns_400(client):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "n": 4,
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400, resp.text
    data = resp.json()
    assert "error" in data, data
    assert data["error"].get("code") == "stream_n_unsupported", data


@pytestmark_server
def test_ac6_completion_stream_with_n_gt_1_returns_400(client):
    body = {
        "model": "test-model",
        "prompt": "hi",
        "n": 4,
        "stream": True,
    }
    resp = client.post("/v1/completions", json=body)
    assert resp.status_code == 400, resp.text
    data = resp.json()
    assert "error" in data, data
    assert data["error"].get("code") == "stream_n_unsupported", data


# --- AC7 + AC8: n=4 non-stream returns 4 choices with distinct indices,
# and usage counts prompt_tokens once but completion_tokens summed.
@pytestmark_server
def test_ac7_ac8_chat_n_4_non_stream(client, monkeypatch):
    PT = 11  # prompt_tokens reported by mock engine
    CT_PER = 5  # completion_tokens per rollout
    outputs = [_make_chat_output(f"reply-{i}", PT, CT_PER) for i in range(4)]
    _install_mock_chat_engine(monkeypatch, outputs)

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "n": 4,
        "stream": False,
        "max_tokens": 8,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # AC7: 4 choices with distinct indices 0..3
    assert len(data["choices"]) == 4, data
    indices = sorted(c["index"] for c in data["choices"])
    assert indices == [0, 1, 2, 3], indices

    # AC8: prompt_tokens counted ONCE; completion_tokens summed; total consistent.
    usage = data["usage"]
    assert usage["prompt_tokens"] == PT, usage
    assert usage["completion_tokens"] == CT_PER * 4, usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytestmark_server
def test_ac7_ac8_completion_n_4_non_stream(client, monkeypatch):
    PT = 9
    CT_PER = 4
    outputs = [_make_completion_output(f"text-{i}", PT, CT_PER) for i in range(4)]
    _install_mock_completion_engine(monkeypatch, outputs)

    body = {
        "model": "test-model",
        "prompt": "hi",
        "n": 4,
        "stream": False,
        "max_tokens": 8,
    }
    resp = client.post("/v1/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["choices"]) == 4, data
    indices = sorted(c["index"] for c in data["choices"])
    assert indices == [0, 1, 2, 3], indices

    usage = data["usage"]
    assert usage["prompt_tokens"] == PT, usage
    assert usage["completion_tokens"] == CT_PER * 4, usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
