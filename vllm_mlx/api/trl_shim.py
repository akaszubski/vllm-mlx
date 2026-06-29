# SPDX-License-Identifier: Apache-2.0
"""TRL-compatible HTTP surface for the vllm-mlx server.

This module exposes the `/generate/`, `/reset_prefix_cache/`, and stub
`/init_communicator/`, `/update_named_param/`, `/close_communicator/` routes
that TRL's ``trl.generation.vllm_client.VLLMClient`` expects.

It is deliberately thin: it translates TRL's request body into our existing
engine API (``engine.generate`` with ``n=`` and ``logprobs=True``) and
re-shapes the per-token logprob dicts into the ``list[list[float]]`` format
TRL consumes.

The init/update/close communicator routes are stubs. The
``MLXVLLMClient`` subclass (see ``examples/trl_grpo_smoke/mlx_vllm_client.py``)
overrides these on the client side and routes weight transfers through the
already-existing ``/init_weight_transfer_engine`` + ``/start_weight_update``
+ ``/update_weights`` + ``/finish_weight_update`` endpoints in
``vllm_mlx/server.py``. The stubs only exist to keep TRL's default code path
from raising 404 if someone forgets to subclass.

See `realign/docs/research/2026-06-29-trl-on-mps-viability.md` for design
rationale (Phase 1 report).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_engine():
    """Return the live ``_engine`` instance from whichever copy of
    ``vllm_mlx.server`` is hosting it.

    When the server is started via ``python -m vllm_mlx.server``, the server
    module is imported as ``__main__``. Anyone else importing
    ``vllm_mlx.server`` from a different entry point will see a freshly
    loaded module with ``_engine = None``. Check ``__main__`` first, then
    fall back to ``vllm_mlx.server``.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and getattr(main_mod, "_engine", None) is not None:
        return main_mod._engine
    server_mod = sys.modules.get("vllm_mlx.server")
    if server_mod is not None and getattr(server_mod, "_engine", None) is not None:
        return server_mod._engine
    return None


# ---------------------------------------------------------------------------
# Request / response schemas (kept inline — they're trivial and TRL-private)
# ---------------------------------------------------------------------------


class TRLGenerateRequest(BaseModel):
    """Matches `trl.generation.vllm_client.VLLMClient.generate` payload.

    Both ``guided_decoding_regex`` (TRL <0.27) and ``structured_outputs_regex``
    (TRL >=0.27) are accepted to insulate the shim from TRL minor-version
    drift.
    """

    model_config = {"extra": "allow"}

    prompts: list[str]
    images: list[str | None] | None = None
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    truncate_prompt_tokens: int | None = None
    guided_decoding_regex: str | None = None
    structured_outputs_regex: str | None = None
    generation_kwargs: dict = Field(default_factory=dict)


class TRLGenerateResponse(BaseModel):
    """Matches the vLLM-server response shape TRL expects."""

    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[float]]


# ---------------------------------------------------------------------------
# /generate/
# ---------------------------------------------------------------------------


def _extract_logprob_values(per_token: list[dict] | None) -> list[float]:
    """Pull the per-token chosen-token ``logprob`` field out of our engine's
    OpenAI-shaped logprobs list.

    Our engine emits ``output.logprobs`` as a list of dicts:
    ``{"token": "...", "logprob": -0.5, "bytes": [...], "top_logprobs": []}``.
    TRL expects a flat ``list[float]``.

    Args:
        per_token: ``output.logprobs`` from a single generation.

    Returns:
        List of floats, one per generated token. Empty when ``per_token`` is
        None.
    """
    if not per_token:
        return []
    values: list[float] = []
    for entry in per_token:
        # Be lenient — accept dicts (our format) or raw floats (defensive).
        if isinstance(entry, dict):
            lp = entry.get("logprob")
            # Some servers (incl. ours, when logprobs disabled) might emit
            # entries without 'logprob'. Skip silently rather than crash.
            if lp is None:
                continue
            values.append(float(lp))
        elif isinstance(entry, (int, float)):
            values.append(float(entry))
    return values


@router.post("/generate/", response_model=TRLGenerateResponse)
async def generate_trl(req: TRLGenerateRequest) -> dict:
    """TRL-compatible ``/generate/`` endpoint.

    For each prompt, fan out ``n`` parallel rollouts using the existing
    engine's ``generate`` method with ``logprobs=True``. Returns prompt token
    ids (one per prompt), completion token ids (n per prompt), and per-token
    logprob lists (n per prompt).

    Notes:
        - ``images`` is accepted but ignored — text-only smoke for Phase 2.
        - ``truncate_prompt_tokens`` is accepted but ignored — engine handles
          its own prompt-length policy.
        - ``guided_decoding_regex`` / ``structured_outputs_regex`` are
          accepted but ignored for Phase 2 (no constrained decoding hook in
          this shim).
    """
    engine = _resolve_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if req.images is not None and any(img is not None for img in req.images):
        logger.warning(
            "trl_shim /generate/: 'images' was provided but the Phase 2 shim "
            "ignores it (text-only smoke)."
        )

    tokenizer = engine.tokenizer
    if tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Tokenizer not available on engine"
        )

    # Build kwargs once.
    generate_kwargs: dict[str, Any] = {
        "max_tokens": int(req.max_tokens),
        "temperature": float(req.temperature),
        "top_p": float(req.top_p),
        "top_k": int(req.top_k) if req.top_k and req.top_k > 0 else 0,
        "min_p": float(req.min_p),
        "repetition_penalty": float(req.repetition_penalty),
        "logprobs": True,
        "top_logprobs": 0,
    }
    # Permit explicit overrides from generation_kwargs (seed, frequency_penalty, ...).
    for k, v in (req.generation_kwargs or {}).items():
        if k in {"n", "prompt", "max_tokens", "logprobs"}:
            # 'n' is honored via per-prompt fan-out; reject other reserved keys.
            continue
        generate_kwargs[k] = v

    prompt_ids_all: list[list[int]] = []
    completion_ids_all: list[list[int]] = []
    logprobs_all: list[list[float]] = []

    n = max(1, int(req.n))

    for prompt in req.prompts:
        # Tokenize the prompt. Most HF tokenizers prepend BOS when
        # add_special_tokens=True. TRL's reference server uses
        # `output.prompt_token_ids` directly from vLLM, which includes BOS by
        # default. Match that behavior — TRL slices these later for loss
        # computation and expects BOS-included token ids.
        try:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        except TypeError:
            # mlx-lm tokenizers sometimes don't accept add_special_tokens.
            prompt_ids = tokenizer.encode(prompt)
        prompt_ids_all.append(list(prompt_ids))

        # Fan out n parallel rollouts. Reuses the same pattern as
        # /v1/completions group sampling (server.py:4014–4060).
        if n == 1:
            outputs = [await engine.generate(prompt=prompt, **generate_kwargs)]
        else:
            outputs = await asyncio.gather(
                *(engine.generate(prompt=prompt, **generate_kwargs) for _ in range(n))
            )

        for output in outputs:
            completion_ids_all.append(list(output.tokens or []))
            logprobs_all.append(_extract_logprob_values(output.logprobs))

    return {
        "prompt_ids": prompt_ids_all,
        "completion_ids": completion_ids_all,
        "logprobs": logprobs_all,
    }


# ---------------------------------------------------------------------------
# /reset_prefix_cache/
# ---------------------------------------------------------------------------


@router.post("/reset_prefix_cache/")
async def reset_prefix_cache_trl() -> dict:
    """TRL-compatible prefix-cache reset.

    Delegates to ``engine.clear_runtime_caches()`` when available. Trailing
    slash matches TRL's URL convention.
    """
    engine = _resolve_engine()
    if engine is None:
        # Trainer may call this before the engine is up; treat as no-op.
        return {"status": "no_engine"}

    cleared = None
    if hasattr(engine, "clear_runtime_caches"):
        try:
            cleared = engine.clear_runtime_caches()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("trl_shim /reset_prefix_cache/: %s", e)
            cleared = None
    return {"status": "ok", "cleared": cleared}


# ---------------------------------------------------------------------------
# /init_communicator/, /update_named_param/, /close_communicator/  (stubs)
# ---------------------------------------------------------------------------
#
# Per the Phase 1 viability study (section 6, "Patches"), the TRL client
# subclass on the trainer side (``MLXVLLMClient``) overrides these methods so
# they never hit the network. We still expose stub HTTP routes so an
# unsubclassed VLLMClient gets a 200 instead of 404 and surfaces the
# "you forgot to subclass" path as a `ValueError` from the trainer-side flow
# rather than an opaque HTTP error.


class _PermissivePayload(BaseModel):
    model_config = {"extra": "allow"}


@router.post("/init_communicator/")
async def init_communicator_stub(payload: _PermissivePayload | None = None) -> dict:
    """Stub. MLX has no NCCL; the MLXVLLMClient subclass routes init via
    ``/init_weight_transfer_engine`` instead.
    """
    logger.info(
        "trl_shim /init_communicator/: stub hit. If you see this without a "
        "MLXVLLMClient subclass, weight sync will be a no-op."
    )
    return {"status": "ok", "note": "stub; use MLXVLLMClient for weight sync"}


@router.post("/update_named_param/")
async def update_named_param_stub(payload: _PermissivePayload | None = None) -> dict:
    """Stub. The MLXVLLMClient subclass routes weight updates via
    ``/update_weights`` (file-based, single tensor per call).
    """
    return {"status": "ok", "note": "stub; use MLXVLLMClient for weight sync"}


@router.post("/close_communicator/")
async def close_communicator_stub() -> dict:
    """Stub. Nothing to close on the MLX side."""
    return {"status": "ok"}
