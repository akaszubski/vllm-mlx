# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for vllm-mlx inference.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # Per-token logprobs in OpenAI Chat Completions format (Plan-1.5 patch #2
    # / realign#1251). Each entry has keys ``token``, ``logprob``, ``bytes``,
    # ``top_logprobs``. None when the request did not opt in to logprobs.
    # For streaming, ``logprobs`` is the cumulative array up to and including
    # the current chunk; ``new_logprobs`` covers only the tokens emitted by
    # the current chunk (typically length 1).
    logprobs: list[dict] | None = None
    new_logprobs: list[dict] | None = None


@contextmanager
def suspend_cancellation():
    """Temporarily clear task cancellation so cleanup can finish deterministically."""
    task = asyncio.current_task()
    if task is None:
        yield
        return

    cancelling = getattr(task, "cancelling", None)
    uncancel = getattr(task, "uncancel", None)
    if cancelling is None or uncancel is None:
        yield
        return

    pending_cancels = cancelling()
    for _ in range(pending_cancels):
        uncancel()
    try:
        yield
    finally:
        for _ in range(pending_cancels):
            task.cancel()


async def run_blocking_startup_work(work: Callable[[], Any]) -> None:
    """Run blocking startup work off-loop without leaking cancellation races."""
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with suspend_cancellation():
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        raise


async def cleanup_startup_cancellation(cleanup: Callable[[], Awaitable[None]]) -> None:
    """Run startup cleanup without letting cleanup failures replace cancellation."""
    with suspend_cancellation():
        try:
            await cleanup()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error(
                "Engine startup cleanup failed while preserving cancellation",
                exc_info=(type(exc), exc, exc.__traceback__),
            )


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @property
    def preserve_native_tool_format(self) -> bool:
        """
        Whether to preserve native tool message format.

        When True, role="tool" messages and tool_calls fields are preserved
        instead of being converted to text. Set by server based on tool parser.
        """
        return getattr(self, "_preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self._preserve_native_tool_format = value

    def prepare_for_start(self) -> None:
        """Run blocking startup work before async engine start.

        Engines can override this to perform heavyweight synchronous model
        loads off the serving event loop. The default implementation is a
        no-op so lightweight engines do not need extra plumbing.
        """
        return None

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    async def score(
        self,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        temperature: float = 1.0,
        return_top_logprobs: int = 0,
    ) -> tuple[list[float], list[list[dict]] | None]:
        """Teacher-forced per-token log-probabilities of a completion.

        Text-only scoring primitive for RL importance ratios and reference-
        model KL. Runs a single forward pass over ``prompt + completion``
        and returns log ``P(completion_i | prompt, completion_<i)`` for every
        completion token.

        Args:
            prompt_token_ids: Prompt token ids (BOS included by the caller).
            completion_token_ids: Completion token ids to score (must be
                non-empty).
            temperature: Softmax temperature applied *before* log_softmax.
                Positive float. 1.0 = raw log-probs.
            return_top_logprobs: If > 0, also return top-k tokens/logprobs
                per position. 0 disables.

        Returns:
            Tuple ``(logprobs, top_logprobs)`` — see
            ``MLXLanguageModel.score_completion`` for exact schema.

        Raises:
            NotImplementedError: Default; subclasses must override.
            RuntimeError: If the engine is not started or does not support
                scoring (e.g. MLLM path).
            ValueError: If inputs are invalid.
        """
        raise NotImplementedError(
            "score() is not implemented by this engine. "
            "Only text-only engines wrapping MLXLanguageModel support scoring."
        )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        return {}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics. Override in subclasses."""
        return None

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches. Override in subclasses."""
        return None
