# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0


class MLXLanguageModel:
    """
    Wrapper around mlx-lm for LLM inference.

    This class provides a unified interface for loading and running
    inference on language models using Apple's MLX framework.

    Example:
        >>> model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
        >>> output = model.generate("Hello, how are you?", max_tokens=100)
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        mtp: bool = False,
        mtp_num_draft_tokens: int = 1,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            mtp_num_draft_tokens: Draft tokens per speculative MTP step
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code
        self._mtp = mtp
        self._mtp_num_draft_tokens = mtp_num_draft_tokens

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from ..utils.tokenizer import load_model_with_fallback

            logger.info(f"Loading model: {self.model_name}")

            # Build tokenizer config
            tokenizer_config = {"trust_remote_code": self.trust_remote_code}

            # Qwen3 fix: eos_token changed from <|im_end|> to <|endoftext|>
            # but chat template still uses <|im_end|>, so we need to set it explicitly
            if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
                tokenizer_config["eos_token"] = "<|im_end|>"
                logger.info("Qwen3 detected: setting eos_token to <|im_end|>")

            self.model, self.tokenizer = load_model_with_fallback(
                self.model_name,
                tokenizer_config=tokenizer_config,
            )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError as err:
            raise ImportError(
                "mlx-lm is required for LLM inference. Install with: pip install mlx-lm"
            ) from err
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """Create a sampler for text generation."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def _create_logits_processors(
        self,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        """Create logits processors for penalty-based sampling."""
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        )
        return processors if processors else None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        logits_processors: list | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling (0 = disabled)
            min_p: Minimum probability threshold
            presence_penalty: Additive penalty for token presence
            repetition_penalty: Multiplicative penalty for repeating tokens
            stop: List of stop sequences
            logits_processors: Optional externally-supplied logits processors
                (e.g. JSON schema constrained decoding).  Merged with built-in
                penalty processors.

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        from mlx_lm import generate

        # Create sampler and logits processors with full Unsloth params
        sampler = self._create_sampler(temperature, top_p, top_k, min_p)
        penalty_processors = self._create_logits_processors(
            presence_penalty, repetition_penalty
        )
        # Merge any externally-provided logits_processors with penalty processors
        all_processors = penalty_processors or []
        if logits_processors:
            all_processors = list(logits_processors) + all_processors

        # Generate text
        output_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=all_processors if all_processors else None,
            verbose=False,
        )

        # Tokenize output to get token IDs
        tokens = self.tokenizer.encode(output_text)

        # Determine finish reason
        finish_reason = "length" if len(tokens) >= max_tokens else "stop"

        return GenerationOutput(
            text=output_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )

    def stream_generate(
        self,
        prompt: Union[str, "mx.array", list[int]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        logits_processors: list | None = None,
        prompt_cache=None,
        **kwargs,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text, token array, or token id list
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling (0 = disabled)
            min_p: Minimum probability threshold
            presence_penalty: Additive penalty for token presence
            repetition_penalty: Multiplicative penalty for repeating tokens
            stop: List of stop sequences
            prompt_cache: Pre-populated KV cache (e.g. from SpecPrefill)

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        from mlx_lm import stream_generate

        # Create sampler and logits processors with full Unsloth params
        sampler = self._create_sampler(temperature, top_p, top_k, min_p)
        penalty_processors = self._create_logits_processors(
            presence_penalty, repetition_penalty
        )
        # Merge any externally-provided logits_processors with penalty processors
        all_processors = None
        if penalty_processors or logits_processors:
            all_processors = (logits_processors or []) + (penalty_processors or [])

        # Count prompt tokens once upfront
        if isinstance(prompt, str):
            num_prompt_tokens = len(self.tokenizer.encode(prompt))
        else:
            num_prompt_tokens = len(prompt)

        accumulated_text = ""

        mtp_kwargs = {}
        if self._mtp:
            mtp_kwargs["mtp"] = True
            mtp_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
        if prompt_cache is not None:
            mtp_kwargs["prompt_cache"] = prompt_cache

        for token_count, response in enumerate(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=all_processors,
                **mtp_kwargs,
            ),
            start=1,
        ):
            # response.text is the new token text (not accumulated)
            new_text = response.text
            accumulated_text += new_text

            # Check for stop sequences
            should_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in accumulated_text:
                        should_stop = True
                        break

            finished = should_stop or token_count >= max_tokens
            finish_reason = None
            if finished:
                finish_reason = "stop" if should_stop else "length"

            yield StreamingOutput(
                text=new_text,
                token=response.token if hasattr(response, "token") else 0,
                finished=finished,
                finish_reason=finish_reason,
                prompt_tokens=num_prompt_tokens,
            )

            if finished:
                break

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list | None = None,
        chat_template_kwargs: dict | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a chat response.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: Optional list of tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with the assistant's response
        """
        if not self._loaded:
            self.load()

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Build kwargs for apply_chat_template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Add tools if provided and supported
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support all requested template kwargs
                template_kwargs.pop("tools", None)
                for key in (chat_template_kwargs or {}).keys():
                    template_kwargs.pop(key, None)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            prompt += "\nassistant:"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # score_completion — teacher-forced per-token logprobs (Phase A)
    #
    # Motivation: TRL/GRPO importance-sampling ratios require the log-prob
    # of every completion token under the *current* policy AND under the
    # frozen reference model. Sampling logprobs from ``/generate/`` only
    # cover the policy at rollout time; the trainer needs a way to re-score
    # arbitrary (prompt, completion) pairs without triggering the sampling
    # loop. Realign's PyTorch+MPS re-score path is ~200-250s/iter; one MLX
    # teacher-forced forward pass replaces it at ~40-50s. See
    # ``realign/docs/research/2026-07-04-vllm-mlx-score-endpoint.md``.
    # ------------------------------------------------------------------

    def _forward_full_sequence(self, full_ids: "mx.array") -> "mx.array":
        """Run the model forward on a full [1, T] sequence with L2-friendly chunking.

        Mirrors the chunking pattern from ``model_runner.py:_prefill_with_chunking``
        so long prompts do not spill L2 cache. Returns the full logits tensor
        ``[1, T, vocab_size]``. No sampling; no cache retention.

        Args:
            full_ids: Token ids as an mlx array with shape ``[1, T]``.

        Returns:
            Logits array of shape ``[1, T, vocab_size]``.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        import mlx.core as mx

        if not self._loaded or self.model is None:
            raise RuntimeError(
                "score_completion called before model.load(). "
                "Call MLXLanguageModel.load() first.\n"
                "See: docs/api.md"
            )

        if full_ids.ndim == 1:
            full_ids = full_ids.reshape(1, -1)

        # L2-friendly chunk size; falls back to a conservative default when
        # the optimizations module is unavailable.
        try:
            from vllm_mlx.optimizations import get_optimal_prefill_size
        except ImportError:  # pragma: no cover — defensive
            def get_optimal_prefill_size(seq_len: int) -> int:
                return min(512, seq_len)

        seq_len = full_ids.shape[-1]
        chunk_size = get_optimal_prefill_size(seq_len)

        # Single-pass fast path
        if seq_len <= chunk_size:
            return self.model(full_ids)

        # Chunked forward. Each chunk uses the running KV cache so later chunks
        # attend to earlier tokens (position-consistent). Only the final chunk's
        # logits are kept — we still want the full-sequence logits though, so
        # we accumulate.
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        parts: list[mx.array] = []
        for i in range(0, seq_len, chunk_size):
            chunk = full_ids[:, i : i + chunk_size]
            logits = self.model(chunk, cache=cache)
            parts.append(logits)
            mx.eval([c.state for c in cache])
        return mx.concatenate(parts, axis=1)

    def score_completion(
        self,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        temperature: float = 1.0,
        return_top_logprobs: int = 0,
    ) -> tuple[list[float], list[list[dict]] | None]:
        """Teacher-forced per-token log-probabilities of a completion.

        For a fixed ``(prompt, completion)`` pair, runs a single forward pass
        over ``prompt + completion`` and returns log ``P(completion_i | prompt,
        completion_<i)`` for every completion token. This is the primitive TRL
        needs for GRPO importance-sampling ratios and reference-model KL — one
        MLX forward per pair instead of two PyTorch/MPS forwards.

        Args:
            prompt_token_ids: The prompt token ids (BOS should already be
                included by the caller — matches ``/generate/`` behaviour).
            completion_token_ids: The completion token ids to score. Must be
                non-empty.
            temperature: Softmax temperature applied to logits *before*
                ``log_softmax`` (matches realign's ``chunked_log_softmax``
                convention). ``temperature=1.0`` = raw model log-probs.
            return_top_logprobs: When > 0, also return the top-k tokens and
                their logprobs at every completion position. Capped by the
                caller (shim rejects > 20).

        Returns:
            A tuple ``(logprobs, top_logprobs)``:
              - ``logprobs`` is a list of ``len(completion_token_ids)`` floats,
                one per completion position.
              - ``top_logprobs`` is ``None`` when ``return_top_logprobs == 0``,
                else a list of length ``len(completion_token_ids)`` where each
                element is a list of ``return_top_logprobs`` dicts with keys
                ``token_id`` (int) and ``logprob`` (float), sorted by logprob
                descending.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If either token list is empty, if the combined
                sequence exceeds ``max_position_embeddings``, or if
                ``temperature`` is not positive.
        """
        import mlx.core as mx

        if not self._loaded or self.model is None:
            raise RuntimeError(
                "score_completion called before model.load(). "
                "Call MLXLanguageModel.load() first."
            )

        if not prompt_token_ids:
            raise ValueError(
                "score_completion: prompt_token_ids is empty. "
                "Expected: at least one prompt token id."
            )
        if not completion_token_ids:
            raise ValueError(
                "score_completion: completion_token_ids is empty. "
                "Expected: at least one completion token id."
            )
        if not (temperature > 0):
            raise ValueError(
                f"score_completion: temperature must be > 0 (got {temperature}). "
                "Expected: positive float; use 1.0 for raw log-probs."
            )

        prompt_len = len(prompt_token_ids)
        completion_len = len(completion_token_ids)
        total_len = prompt_len + completion_len

        # Context-length guard. mlx-lm models expose their config via
        # ``model.args`` (Qwen3, Llama, etc.); fall back to ``model.config``.
        max_pos = None
        args = getattr(self.model, "args", None)
        if args is not None:
            max_pos = getattr(args, "max_position_embeddings", None)
        if max_pos is None:
            config = getattr(self.model, "config", None)
            if config is not None:
                max_pos = getattr(config, "max_position_embeddings", None)
        if max_pos is not None and total_len > max_pos:
            raise ValueError(
                f"score_completion: prompt+completion length {total_len} "
                f"exceeds model max_position_embeddings {max_pos}. "
                "Expected: shorter inputs, or a model with longer context."
            )

        # Build the full sequence tensor. The model needs the full context so
        # each completion position sees prompt + all prior completion tokens.
        full_ids = mx.array(
            list(prompt_token_ids) + list(completion_token_ids), dtype=mx.int32
        ).reshape(1, -1)

        # Forward pass. Returns [1, total_len, vocab_size].
        logits = self._forward_full_sequence(full_ids)

        # Slice to the positions that predict completion tokens. Position i
        # predicts token i+1. So completion token j (0-indexed within the
        # completion) is predicted at index (prompt_len - 1 + j). We need
        # ``completion_len`` positions starting at prompt_len - 1.
        start = prompt_len - 1
        end = start + completion_len
        completion_logits = logits[:, start:end, :]  # [1, completion_len, V]

        # Numerical safety per realign convention: cast to fp32 BEFORE
        # softmax, apply temperature BEFORE log_softmax, cast back on exit.
        model_dtype = completion_logits.dtype
        completion_logits_fp32 = completion_logits.astype(mx.float32)

        # Apply temperature scaling before log_softmax (chunked_log_softmax
        # invariant — larger temp flattens the distribution, smaller
        # sharpens it).
        scaled = completion_logits_fp32 / temperature

        # Numerically stable log_softmax.
        log_probs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)

        # Gather the chosen-token log-prob at each completion position.
        # Build an [1, completion_len] index array.
        completion_ids_arr = mx.array(list(completion_token_ids), dtype=mx.int32)
        # Fancy index via mx.take_along_axis for a batch-1 gather.
        gathered = mx.take_along_axis(
            log_probs,
            completion_ids_arr.reshape(1, -1, 1),
            axis=-1,
        ).squeeze(-1)  # [1, completion_len]

        # Evaluate only what we return (bandwidth-friendly per realign
        # numerical-safety rule).
        mx.eval(gathered)
        chosen_logprobs: list[float] = [float(x) for x in gathered[0].tolist()]

        top_logprobs_out: list[list[dict]] | None = None
        if return_top_logprobs > 0:
            k = int(return_top_logprobs)
            # Cap k by vocab size to avoid runtime errors.
            vocab_size = log_probs.shape[-1]
            k = min(k, vocab_size)

            # For each position, take the top-k tokens by log-prob.
            # log_probs has shape [1, completion_len, V]. Sort descending.
            # mx.argpartition is available; use argsort for simplicity.
            sorted_ids = mx.argsort(-log_probs, axis=-1)[:, :, :k]  # [1, L, k]
            sorted_lp = mx.take_along_axis(log_probs, sorted_ids, axis=-1)
            mx.eval(sorted_ids, sorted_lp)

            ids_py = sorted_ids[0].tolist()  # [L][k]
            lp_py = sorted_lp[0].tolist()  # [L][k]
            top_logprobs_out = []
            for pos_ids, pos_lps in zip(ids_py, lp_py):
                row = [
                    {"token_id": int(t), "logprob": float(lp)}
                    for t, lp in zip(pos_ids, pos_lps)
                ]
                top_logprobs_out.append(row)

        # Cast back to model dtype implicitly not needed — we return Python
        # floats. But we free the fp32 intermediates.
        del completion_logits_fp32, scaled, log_probs, model_dtype

        # Metal buffer hygiene (realign #981 / `feedback_mlx_deadlock_reboot`
        # class): campaign-scale scoring (5000 iters × N pairs) accumulates
        # Metal buffers if we never release the cache. Clear on the way out
        # so repeated scoring calls hold a bounded footprint.
        mx.clear_cache()

        return chosen_logprobs, top_logprobs_out

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXLanguageModel model={self.model_name} status={status}>"
