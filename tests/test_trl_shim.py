# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TRL-compatible HTTP shim.

These tests exercise the FastAPI route layer in isolation by stubbing
``server._engine`` with a minimal fake. They do NOT require a real MLX model
or running server.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server
from vllm_mlx.api.trl_shim import _extract_logprob_values


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Encodes by mapping each whitespace-separated word to ``hash(word) % 1000``."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [hash(w) % 1000 for w in text.split()]
        if add_special_tokens:
            return [101] + tokens  # 101 mimics a BOS-ish marker
        return tokens


@dataclass
class _FakeOutput:
    text: str = "answer"
    tokens: list[int] = field(default_factory=lambda: [1, 2, 3])
    prompt_tokens: int = 4
    completion_tokens: int = 3
    finish_reason: str = "stop"
    finished: bool = True
    logprobs: list[dict] | None = field(
        default_factory=lambda: [
            {"token": "a", "logprob": -0.5, "bytes": [], "top_logprobs": []},
            {"token": "b", "logprob": -1.5, "bytes": [], "top_logprobs": []},
            {"token": "c", "logprob": -0.1, "bytes": [], "top_logprobs": []},
        ]
    )


class _FakeEngine:
    def __init__(self, *, generate_logprobs: list[dict] | None = None) -> None:
        self._generate_logprobs = generate_logprobs
        self.calls: list[dict] = []

    @property
    def tokenizer(self) -> _FakeTokenizer:
        return _FakeTokenizer()

    async def generate(self, *, prompt: str, **kwargs) -> _FakeOutput:
        self.calls.append({"prompt": prompt, **kwargs})
        if self._generate_logprobs is not None:
            out = _FakeOutput()
            out.logprobs = list(self._generate_logprobs)
            return out
        return _FakeOutput()

    def clear_runtime_caches(self) -> dict:
        return {"prefix_cache_entries": 0}

    async def stop(self) -> None:
        """No-op shutdown hook — exercised by FastAPI's lifespan teardown."""
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def shim_client(monkeypatch):
    """TestClient bound to the real `server.app` with a fake engine."""
    fake_engine = _FakeEngine()
    monkeypatch.setattr(server, "_engine", fake_engine)
    monkeypatch.setattr(server, "_api_key", None)
    with TestClient(server.app) as client:
        yield client, fake_engine


# ---------------------------------------------------------------------------
# /generate/ contract
# ---------------------------------------------------------------------------


def test_generate_returns_trl_shape_for_single_prompt(shim_client):
    client, engine = shim_client
    resp = client.post(
        "/generate/",
        json={
            "prompts": ["hello world"],
            "n": 1,
            "max_tokens": 8,
            "temperature": 0.5,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"prompt_ids", "completion_ids", "logprobs"}
    assert len(body["prompt_ids"]) == 1
    assert len(body["completion_ids"]) == 1
    assert len(body["logprobs"]) == 1
    # Logprobs were flattened from dict-format to float-list.
    assert all(isinstance(v, float) for v in body["logprobs"][0])
    # Sampling kwargs reached the engine.
    assert engine.calls[0]["max_tokens"] == 8
    assert engine.calls[0]["temperature"] == 0.5
    assert engine.calls[0]["logprobs"] is True


def test_generate_fans_out_n_completions_per_prompt(shim_client):
    client, engine = shim_client
    resp = client.post(
        "/generate/",
        json={"prompts": ["p1", "p2"], "n": 3, "max_tokens": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 2 prompts, n=3 → 2 prompt_ids, 6 completion_ids/logprobs.
    assert len(body["prompt_ids"]) == 2
    assert len(body["completion_ids"]) == 2 * 3
    assert len(body["logprobs"]) == 2 * 3
    # Engine was called 2*3 = 6 times.
    assert len(engine.calls) == 6


def test_generate_503_when_engine_not_initialized(monkeypatch):
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_api_key", None)
    with TestClient(server.app) as client:
        resp = client.post("/generate/", json={"prompts": ["x"], "n": 1})
    assert resp.status_code == 503


def test_generate_accepts_structured_outputs_regex_field(shim_client):
    """TRL 0.27+ uses 'structured_outputs_regex' instead of 'guided_decoding_regex'."""
    client, _ = shim_client
    resp = client.post(
        "/generate/",
        json={
            "prompts": ["x"],
            "n": 1,
            "max_tokens": 2,
            "structured_outputs_regex": "[0-9]+",
        },
    )
    assert resp.status_code == 200, resp.text


def test_generate_accepts_guided_decoding_regex_field(shim_client):
    """TRL <0.27 used 'guided_decoding_regex'. Both must parse."""
    client, _ = shim_client
    resp = client.post(
        "/generate/",
        json={
            "prompts": ["x"],
            "n": 1,
            "max_tokens": 2,
            "guided_decoding_regex": "[0-9]+",
        },
    )
    assert resp.status_code == 200, resp.text


def test_generate_ignores_images_field(shim_client):
    """Phase 2 is text-only; images must be accepted but not crash."""
    client, _ = shim_client
    resp = client.post(
        "/generate/",
        json={"prompts": ["x"], "images": [None], "n": 1, "max_tokens": 2},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# /reset_prefix_cache/
# ---------------------------------------------------------------------------


def test_reset_prefix_cache_calls_engine(shim_client):
    client, _ = shim_client
    resp = client.post("/reset_prefix_cache/")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"


def test_reset_prefix_cache_no_engine(monkeypatch):
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_api_key", None)
    with TestClient(server.app) as client:
        resp = client.post("/reset_prefix_cache/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_engine"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def test_init_communicator_stub_returns_ok(shim_client):
    client, _ = shim_client
    resp = client.post(
        "/init_communicator/",
        json={"host": "0.0.0.0", "port": 51216, "world_size": 2, "client_device_uuid": "x"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_update_named_param_stub_returns_ok(shim_client):
    client, _ = shim_client
    resp = client.post(
        "/update_named_param/",
        json={"name": "model.embed.weight", "dtype": "torch.float32", "shape": [10, 4]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_close_communicator_stub_returns_ok(shim_client):
    client, _ = shim_client
    resp = client.post("/close_communicator/")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _extract_logprob_values helper
# ---------------------------------------------------------------------------


def test_extract_logprob_values_handles_none():
    assert _extract_logprob_values(None) == []


def test_extract_logprob_values_handles_empty_list():
    assert _extract_logprob_values([]) == []


def test_extract_logprob_values_handles_openai_dict_shape():
    entries = [
        {"token": "a", "logprob": -0.5, "bytes": [], "top_logprobs": []},
        {"token": "b", "logprob": -1.25, "bytes": [], "top_logprobs": []},
    ]
    assert _extract_logprob_values(entries) == [-0.5, -1.25]


def test_extract_logprob_values_skips_missing_logprob_keys():
    entries = [
        {"token": "a", "logprob": -0.5},
        {"token": "b"},  # malformed — no logprob field
        {"token": "c", "logprob": -0.25},
    ]
    assert _extract_logprob_values(entries) == [-0.5, -0.25]


def test_extract_logprob_values_handles_raw_floats():
    """Defensive — if a future engine emits floats directly, don't crash."""
    assert _extract_logprob_values([-0.5, -1.0, -2.0]) == [-0.5, -1.0, -2.0]


# ---------------------------------------------------------------------------
# Router registration regression
# ---------------------------------------------------------------------------


def test_all_trl_routes_registered_on_app():
    """Guard against the router being dropped from server.py during refactor."""
    paths = {getattr(r, "path", None) for r in server.app.routes}
    for required in (
        "/generate/",
        "/reset_prefix_cache/",
        "/init_communicator/",
        "/update_named_param/",
        "/close_communicator/",
    ):
        assert required in paths, f"missing route: {required}"


# ---------------------------------------------------------------------------
# Engine-lookup resilience
# ---------------------------------------------------------------------------


def test_resolve_engine_prefers_main_module(monkeypatch):
    """When ``python -m vllm_mlx.server`` runs, ``__main__`` holds the live
    engine — not ``vllm_mlx.server``. Confirm we read from __main__ first."""
    import sys

    from vllm_mlx.api import trl_shim

    sentinel_main = object()
    sentinel_server = object()
    monkeypatch.setattr(server, "_engine", sentinel_server)
    main_mod = sys.modules["__main__"]
    monkeypatch.setattr(main_mod, "_engine", sentinel_main, raising=False)
    # monkeypatch.undo() will clean both attrs.
    assert trl_shim._resolve_engine() is sentinel_main


def test_resolve_engine_falls_back_to_server_module(monkeypatch):
    """When __main__ has no _engine, fall back to vllm_mlx.server."""
    import sys

    from vllm_mlx.api import trl_shim

    sentinel = object()
    monkeypatch.setattr(server, "_engine", sentinel)
    main_mod = sys.modules["__main__"]
    # Force __main__._engine to None so trl_shim falls through to server.
    monkeypatch.setattr(main_mod, "_engine", None, raising=False)
    assert trl_shim._resolve_engine() is sentinel


# ---------------------------------------------------------------------------
# Wire fix: BatchedEngine proxies for weight-transfer methods
# ---------------------------------------------------------------------------


def test_batched_engine_proxies_weight_transfer_to_engine_core():
    """Regression for the wire bug discovered in Phase 2.

    The /init_weight_transfer_engine, /update_weights, /finish_weight_update,
    /start_weight_update, /get_world_size, /pause, /resume routes
    delegate to ``_engine.<method>``. ``_engine`` is a BatchedEngine, but
    the actual implementation lives on EngineCore — *two* layers under
    BatchedEngine. Without the proxies on BatchedEngine, the calls raised
    ``AttributeError: 'BatchedEngine' object has no attribute ...``.
    """
    from unittest.mock import MagicMock

    from vllm_mlx.engine.batched import BatchedEngine

    be = BatchedEngine.__new__(BatchedEngine)
    inner_core = MagicMock()
    inner_core.init_weight_transfer_engine.return_value = {"initialized": True}
    inner_core.start_weight_update.return_value = {"started": True}
    inner_core.update_weights.return_value = {"applied": True, "num_params": 1}
    inner_core.finish_weight_update.return_value = {"resumed": True}
    inner_core.get_world_size.return_value = 1
    inner_core.pause.return_value = {"paused": True}
    inner_core.resume.return_value = {"resumed": True}
    async_core = MagicMock()
    async_core.engine = inner_core
    be._engine = async_core
    be._mllm_scheduler = None

    assert be.init_weight_transfer_engine({"init_info": {"world_size": 1}}) == {
        "initialized": True
    }
    assert be.start_weight_update(is_checkpoint_format=True) == {"started": True}
    assert be.update_weights({"update_info": {}}) == {"applied": True, "num_params": 1}
    assert be.finish_weight_update() == {"resumed": True}
    assert be.get_world_size() == 1
    assert be.pause(mode="wait", clear_cache=True) == {"paused": True}
    assert be.resume() == {"resumed": True}

    # And confirm each call did reach the inner EngineCore.
    inner_core.init_weight_transfer_engine.assert_called_once()
    inner_core.start_weight_update.assert_called_once_with(True)
    inner_core.update_weights.assert_called_once()
    inner_core.finish_weight_update.assert_called_once()
    inner_core.get_world_size.assert_called_once()
    inner_core.pause.assert_called_once_with(mode="wait", clear_cache=True)
    inner_core.resume.assert_called_once()


# ---------------------------------------------------------------------------
# Wire fix: RequestOutputCollector._merge_outputs preserves logprobs
# ---------------------------------------------------------------------------


def test_request_output_collector_merge_preserves_logprobs():
    """Regression for the second Phase 2 wire bug: when the engine
    produces multiple chunks before the consumer drains, ``_merge_outputs``
    used to drop ``logprobs`` and ``new_logprobs``. This made
    ``output.logprobs`` come back empty on /generate/ even though
    ``sampling_params.logprobs=True``.
    """
    from vllm_mlx.output_collector import RequestOutputCollector
    from vllm_mlx.request import RequestOutput

    collector = RequestOutputCollector(aggregate=True)

    first = RequestOutput(
        request_id="r",
        new_token_ids=[1],
        new_text="a",
        output_token_ids=[1],
        output_text="a",
        finished=False,
        prompt_tokens=2,
        completion_tokens=1,
        logprobs=[{"token": "a", "logprob": -0.5, "bytes": [], "top_logprobs": []}],
        new_logprobs=[{"token": "a", "logprob": -0.5, "bytes": [], "top_logprobs": []}],
    )
    second = RequestOutput(
        request_id="r",
        new_token_ids=[2],
        new_text="b",
        output_token_ids=[1, 2],
        output_text="ab",
        finished=True,
        finish_reason="stop",
        prompt_tokens=2,
        completion_tokens=2,
        logprobs=[
            {"token": "a", "logprob": -0.5, "bytes": [], "top_logprobs": []},
            {"token": "b", "logprob": -1.0, "bytes": [], "top_logprobs": []},
        ],
        new_logprobs=[
            {"token": "b", "logprob": -1.0, "bytes": [], "top_logprobs": []}
        ],
    )

    collector.put(first)
    collector.put(second)  # triggers merge

    merged = collector.get_nowait()
    assert merged is not None
    # Cumulative logprobs are preserved from the latest chunk.
    assert merged.logprobs is not None
    assert len(merged.logprobs) == 2
    assert [lp["logprob"] for lp in merged.logprobs] == [-0.5, -1.0]
    # new_logprobs is concatenated across merged chunks.
    assert merged.new_logprobs is not None
    assert len(merged.new_logprobs) == 2


def test_batched_engine_proxy_raises_when_engine_not_started():
    """Calling the proxy without ``_engine`` (AsyncEngineCore) set raises
    a clear RuntimeError rather than ``AttributeError: 'NoneType' ...``.
    """
    import pytest

    from vllm_mlx.engine.batched import BatchedEngine

    be = BatchedEngine.__new__(BatchedEngine)
    be._engine = None
    be._mllm_scheduler = None

    with pytest.raises(RuntimeError, match="not started"):
        be.init_weight_transfer_engine({"init_info": {"world_size": 1}})
