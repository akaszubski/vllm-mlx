# SPDX-License-Identifier: Apache-2.0
"""MLX backend for `WeightTransferEngine`.

This module imports `mlx.core` at module top, so it MUST be imported lazily
(via `WeightTransferEngineFactory.get_engine_class('mlx')`) on Apple-Silicon
hosts only.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Tuple

import mlx.core as mx

from .base import WeightTransferEngine
from .types import MLXWeightTransferInitInfo, MLXWeightTransferUpdateInfo

logger = logging.getLogger(__name__)


class MLXWeightTransferEngine(
    WeightTransferEngine[MLXWeightTransferInitInfo, MLXWeightTransferUpdateInfo]
):
    """MLX-backed weight transfer engine.

    Honors the upstream `WeightTransferEngine` contract while delegating the
    actual array materialization to MLX (`mx.load`, `mx.save_safetensors`,
    `mx.eval`).
    """

    init_info_cls = MLXWeightTransferInitInfo
    update_info_cls = MLXWeightTransferUpdateInfo

    def __init__(self, config: Any, parallel_config: Any, model: Any) -> None:
        super().__init__(config, parallel_config, model)
        self._initialized = False
        self._init_info: MLXWeightTransferInitInfo | None = None

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def init_transfer_engine(
        self, init_info: MLXWeightTransferInitInfo
    ) -> None:
        """Validate single-host invariants and stash session info.

        Args:
            init_info: Parsed init info. world_size MUST equal 1.

        Raises:
            ValueError: If world_size != 1 (MLX is single-host only).
        """
        if init_info.world_size != 1:
            raise ValueError(
                f"MLX weight-transfer backend requires world_size=1, "
                f"got world_size={init_info.world_size}.\n"
                f"MLX is single-host (Apple Silicon); use NCCL/UCX for multi-rank.\n"
                f"See: PROJECT.md scope notes."
            )
        if init_info.extra:
            logger.warning(
                "MLXWeightTransferEngine: ignoring upstream fields not "
                "applicable to single-host MLX: %s",
                sorted(init_info.extra.keys()),
            )
        self._init_info = init_info
        self._initialized = True
        logger.info(
            "MLXWeightTransferEngine initialized (rank=%d, backend=%r)",
            init_info.rank,
            init_info.backend or "mlx",
        )

    def shutdown(self) -> None:
        """Drop references and trigger gc."""
        self._init_info = None
        self._initialized = False
        gc.collect()
        logger.debug("MLXWeightTransferEngine shutdown complete")

    # ------------------------------------------------------------------
    # Receive path
    # ------------------------------------------------------------------

    def receive_weights(
        self,
        update_info: MLXWeightTransferUpdateInfo,
        load_weights: Callable[[List[Tuple[str, Any]]], None],
    ) -> None:
        """Materialize one weight update and apply it via `load_weights`.

        Args:
            update_info: Parsed update info. Exactly one of `path`/`packed`
                MUST be set.
            load_weights: Callable invoked once with
                `[(name, mx.array), ...]` for the parameters in
                `update_info.names`.

        Raises:
            ValueError: If neither/both of `path`/`packed` are set, or if
                lengths/shapes don't match.
            FileNotFoundError: If `path` is set but the file does not exist.
        """
        if not self._initialized:
            raise RuntimeError(
                "receive_weights called before init_transfer_engine; "
                "call init_transfer_engine first."
            )

        has_path = update_info.path is not None
        has_packed = update_info.packed is not None
        if has_path == has_packed:
            raise ValueError(
                "MLXWeightTransferUpdateInfo must set exactly one of "
                "'path' or 'packed'."
            )

        n = len(update_info.names)
        if len(update_info.dtype_names) != n or len(update_info.shapes) != n:
            raise ValueError(
                f"update_info length mismatch: names={n} "
                f"dtype_names={len(update_info.dtype_names)} "
                f"shapes={len(update_info.shapes)}"
            )

        arrays = self._load_arrays(update_info)

        params_list: List[Tuple[str, Any]] = []
        for name, expected_shape, declared_dtype in zip(
            update_info.names, update_info.shapes, update_info.dtype_names
        ):
            if name not in arrays:
                raise KeyError(
                    f"update_info.names references {name!r} but it was not "
                    f"found in the loaded safetensors payload."
                )
            arr = arrays[name]
            actual_shape = list(arr.shape)
            expected = list(expected_shape)
            if actual_shape != expected:
                raise ValueError(
                    f"shape mismatch for {name!r}: "
                    f"expected {expected}, got {actual_shape}"
                )
            # Security A08 / #1316: verify declared dtype matches loaded array.
            # dtype_names is structurally length-checked above but never compared
            # to actual data — a caller declaring float32 while shipping
            # bfloat16 would silently poison inference.
            actual_dtype = str(arr.dtype)
            if actual_dtype != declared_dtype:
                raise ValueError(
                    f"dtype mismatch for {name!r}: "
                    f"declared {declared_dtype!r}, actual {actual_dtype!r}. "
                    "This would silently corrupt inference — the transfer "
                    "engine refuses to load a param whose shipped dtype "
                    "diverges from the declaration."
                )
            params_list.append((name, arr))

        load_weights(params_list)

    def _load_arrays(
        self, update_info: MLXWeightTransferUpdateInfo
    ) -> Dict[str, Any]:
        """Load arrays from `path` or `packed` bytes.

        Security A03 / #1313: `path` mode is fail-closed. It is refused unless
        env var ``VLLM_MLX_WEIGHT_TRANSFER_ALLOWED_DIR`` declares a
        colon-separated allowlist of parent directories, and the resolved
        (symlink-expanded, `..`-collapsed) path lies under one of them.
        Prevents directory-traversal reads of arbitrary safetensors files
        (or the error-message leak from attempting to parse non-safetensors
        like ``~/.ssh/id_rsa``).
        """
        if update_info.path is not None:
            allowed_env = os.environ.get(
                "VLLM_MLX_WEIGHT_TRANSFER_ALLOWED_DIR", ""
            ).strip()
            if not allowed_env:
                raise PermissionError(
                    "weight-update path mode disabled: "
                    "VLLM_MLX_WEIGHT_TRANSFER_ALLOWED_DIR is unset. "
                    "Set it to a colon-separated list of directories the "
                    "transfer engine may load from, or use the packed bytes "
                    "path (update_info.packed) instead. See #1313."
                )
            requested = os.path.realpath(update_info.path)
            allowed_dirs = [
                os.path.realpath(d) for d in allowed_env.split(":") if d
            ]
            if not any(
                requested == d
                or requested.startswith(d.rstrip("/") + "/")
                for d in allowed_dirs
            ):
                raise PermissionError(
                    f"weight-update path {update_info.path!r} resolves to "
                    f"{requested!r}, which is not under any allowed dir in "
                    f"VLLM_MLX_WEIGHT_TRANSFER_ALLOWED_DIR ({allowed_env!r}). "
                    "Refusing to load — directory-traversal check. See #1313."
                )
            if not os.path.exists(requested):
                raise FileNotFoundError(
                    f"weight-update path does not exist: {update_info.path}"
                )
            loaded = mx.load(requested)
            return self._coerce_to_dict(loaded, requested)

        # packed bytes → tempfile → mx.load
        assert update_info.packed is not None
        with tempfile.NamedTemporaryFile(
            suffix=".safetensors", delete=False
        ) as tf:
            tf.write(update_info.packed)
            tmp_path = tf.name
        try:
            loaded = mx.load(tmp_path)
            return self._coerce_to_dict(loaded, tmp_path)
        finally:
            # Security A08 / #1315: log failed cleanups so a stuck temp file
            # (permissions, disk full, race with another process) is visible
            # in operator logs rather than silently accumulating multi-GB
            # weight payloads in /tmp until OS cleanup fires.
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning(
                    "Failed to unlink temp weight file %s: %s. "
                    "Payload may persist in /tmp until OS-level cleanup.",
                    tmp_path,
                    exc,
                )

    @staticmethod
    def _coerce_to_dict(loaded: Any, source: str) -> Dict[str, Any]:
        """mx.load returns a dict for safetensors; guard against array returns."""
        if isinstance(loaded, dict):
            return loaded
        raise ValueError(
            f"Expected dict from mx.load({source!r}), got {type(loaded).__name__}. "
            f"Only safetensors-format payloads are supported."
        )

    # ------------------------------------------------------------------
    # Trainer-side helpers
    # ------------------------------------------------------------------

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[Tuple[str, Any]],
        trainer_args: Dict[str, Any],
    ) -> None:
        """Serialize an iterator of (name, mx.array) to safetensors.

        Args:
            iterator: Yields `(name, mx.array)` pairs.
            trainer_args: Must include `'out_path'` (str) — destination
                safetensors file. In-process push (no `out_path`) is not
                implemented in this MLX-only backend.

        Raises:
            NotImplementedError: If `out_path` is missing.
        """
        out_path = trainer_args.get("out_path")
        if out_path is None:
            raise NotImplementedError(
                "MLXWeightTransferEngine.trainer_send_weights requires "
                "trainer_args['out_path']; in-process push not implemented."
            )

        materialized: Dict[str, Any] = {}
        for name, arr in iterator:
            # Idempotent barrier — eval() forces any pending lazy ops on arr.
            mx.eval(arr)
            materialized[name] = arr

        mx.save_safetensors(out_path, materialized)
        logger.info(
            "trainer_send_weights: wrote %d params to %s",
            len(materialized),
            out_path,
        )
