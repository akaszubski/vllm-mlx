# SPDX-License-Identifier: Apache-2.0
"""Abstract WeightTransferEngine + Factory mirroring vllm-project/vllm v0.23+ interface.

This file is the local copy of the upstream extension point. realign#1310 will
contribute the MLX backend back to vllm-project/vllm and (potentially) remove
this file in favor of importing from upstream — but until then, we own it here
to keep the dependency footprint minimal.

The interface intentionally mirrors upstream's `WeightTransferEngine` so the
MLX backend can be contributed without re-shaping its API.
"""

from __future__ import annotations

import importlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    Tuple,
    Type,
    TypeVar,
)


@dataclass
class WeightTransferInitInfo:
    """Base init info — backends extend with their own fields.

    Attributes:
        world_size: Number of ranks participating in the transfer. MLX runs
            single-host only; >1 is reserved for upstream backends.
        rank: This worker's rank in [0, world_size).
        backend: Backend identifier (e.g., "mlx", "nccl"). Not the same as the
            registry key, but typically matches.
    """

    world_size: int = 1
    rank: int = 0
    backend: str = ""


@dataclass
class WeightTransferUpdateInfo:
    """Base update info — backends extend with their own fields.

    Attributes:
        names: Parameter names in the model namespace.
        dtype_names: Per-parameter dtype names, parallel to `names`.
        shapes: Per-parameter shapes (lists of ints), parallel to `names`.
    """

    names: list[str] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)
    shapes: list[list[int]] = field(default_factory=list)


TInitInfo = TypeVar("TInitInfo", bound=WeightTransferInitInfo)
TUpdateInfo = TypeVar("TUpdateInfo", bound=WeightTransferUpdateInfo)


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """Abstract weight transfer engine.

    Subclasses MUST:
      - Set class attributes `init_info_cls` and `update_info_cls`.
      - Implement `init_transfer_engine`, `receive_weights`, `shutdown`.
      - Implement static `trainer_send_weights(iterator, trainer_args)`.

    Subclasses MAY:
      - Override `receive_sparse_weights` and `trainer_send_sparse_weights`
        (default raises NotImplementedError).
    """

    init_info_cls: ClassVar[Type[WeightTransferInitInfo]]
    update_info_cls: ClassVar[Type[WeightTransferUpdateInfo]]

    def __init__(self, config: Any, parallel_config: Any, model: Any) -> None:
        """Store engine context.

        Args:
            config: Engine-level config (vLLM EngineConfig or analog).
            parallel_config: Parallel config (may be None on single-host).
            model: The MLX/torch model whose parameters will be updated.
        """
        self._config = config
        self._parallel_config = parallel_config
        self._model = model

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        """Bring the transfer engine to ready state for a particular session."""

    @abstractmethod
    def receive_weights(
        self,
        update_info: TUpdateInfo,
        load_weights: Callable[[list[Tuple[str, Any]]], None],
    ) -> None:
        """Receive one weight update and apply it via `load_weights`.

        Args:
            update_info: Per-update metadata (names/dtypes/shapes + payload pointer).
            load_weights: Callable that does `model.update(dict)` + a barrier
                (e.g., mx.eval). The backend MUST call this exactly once
                with the materialized `[(name, array), ...]` pairs.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release any per-session resources (sockets, buffers, threads)."""

    @staticmethod
    @abstractmethod
    def trainer_send_weights(
        iterator: Iterator[Tuple[str, Any]],
        trainer_args: Dict[str, Any],
    ) -> None:
        """Trainer-side: serialize an iterator of (name, array) for transport."""

    def receive_sparse_weights(
        self,
        update_info: Any,
        apply_patches: Callable[..., None],
    ) -> None:
        """Optional: receive sparse weight patches (e.g., LoRA deltas)."""
        raise NotImplementedError(
            "Sparse weight transfer not implemented by this backend"
        )

    @staticmethod
    def trainer_send_sparse_weights(
        iterator: Iterator[Tuple[str, Any]],
        trainer_args: Dict[str, Any],
    ) -> None:
        """Optional trainer-side sparse send."""
        raise NotImplementedError(
            "Sparse weight transfer not implemented by this backend"
        )

    @classmethod
    def parse_init_info(cls, init_dict: Dict[str, Any]) -> TInitInfo:
        """Build init_info from a dict.

        Permissive: unknown keys are routed to an `extra` field if present;
        otherwise dropped silently. The intent is to remain forward-compatible
        with upstream NCCL/UCX fields without breaking MLX deployments.
        """
        fields = {f.name for f in cls.init_info_cls.__dataclass_fields__.values()}
        known = {k: v for k, v in init_dict.items() if k in fields}
        unknown = {k: v for k, v in init_dict.items() if k not in fields}
        if unknown and "extra" in fields:
            known["extra"] = unknown
        return cls.init_info_cls(**known)

    @classmethod
    def parse_update_info(cls, update_dict: Dict[str, Any]) -> TUpdateInfo:
        """Build update_info from a dict, dropping unknown keys."""
        fields = {f.name for f in cls.update_info_cls.__dataclass_fields__.values()}
        return cls.update_info_cls(
            **{k: v for k, v in update_dict.items() if k in fields}
        )


class WeightTransferEngineFactory:
    """Lazy backend registry.

    `register_engine` stores `(module_path, class_name)` strings — the first
    `create_engine(backend, ...)` triggers the import. This honors the
    constraint that code paths importing mlx MUST be guarded.

    For tests, `register_engine_class` skips the indirection.
    """

    _registry: Dict[str, Tuple[str, str]] = {}
    _classes: Dict[str, Type["WeightTransferEngine[Any, Any]"]] = {}
    _lock = threading.Lock()

    @classmethod
    def register_engine(cls, backend: str, module_path: str, class_name: str) -> None:
        """Register a backend by module path + class name (lazy)."""
        with cls._lock:
            cls._registry[backend] = (module_path, class_name)

    @classmethod
    def register_engine_class(
        cls,
        backend: str,
        engine_cls: Type["WeightTransferEngine[Any, Any]"],
    ) -> None:
        """Direct (eager) registration — primarily for tests."""
        with cls._lock:
            cls._classes[backend] = engine_cls

    @classmethod
    def get_engine_class(
        cls, backend: str
    ) -> Type["WeightTransferEngine[Any, Any]"]:
        """Resolve the backend class, importing on first access."""
        with cls._lock:
            if backend in cls._classes:
                return cls._classes[backend]
            if backend not in cls._registry:
                raise KeyError(
                    f"No weight-transfer backend registered for: {backend!r}.\n"
                    f"Registered backends: {list(cls._registry)}\n"
                    f"See: docs/weight_transfer.md for adding a backend."
                )
            module_path, class_name = cls._registry[backend]
        module = importlib.import_module(module_path)
        engine_cls = getattr(module, class_name)
        with cls._lock:
            cls._classes[backend] = engine_cls
        return engine_cls

    @classmethod
    def create_engine(
        cls,
        backend: str,
        config: Any,
        parallel_config: Any,
        model: Any,
    ) -> "WeightTransferEngine[Any, Any]":
        """Instantiate the backend engine."""
        engine_cls = cls.get_engine_class(backend)
        return engine_cls(config, parallel_config, model)

    @classmethod
    def reset_registry(cls) -> None:
        """Clear all registrations. Test-only."""
        with cls._lock:
            cls._registry.clear()
            cls._classes.clear()
