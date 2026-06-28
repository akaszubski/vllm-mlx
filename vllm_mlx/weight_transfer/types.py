# SPDX-License-Identifier: Apache-2.0
"""MLX-specific init/update info dataclasses for the weight-transfer engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import WeightTransferInitInfo, WeightTransferUpdateInfo


@dataclass
class MLXWeightTransferInitInfo(WeightTransferInitInfo):
    """MLX backend init info.

    Attributes:
        world_size: MUST be 1 — MLX runs single-host (PROJECT.md scope).
        rank: Always 0 for MLX.
        backend: Conventionally "mlx".
        extra: Forward-compat slot for unknown upstream fields
            (e.g., NCCL endpoint URIs that we silently ignore).
    """

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLXWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """MLX backend update info.

    Exactly one of `path` or `packed` MUST be set. Server-side Pydantic
    validation enforces this; backend asserts defensively.

    Attributes:
        path: Filesystem path to a safetensors file (preferred for large updates).
        packed: Raw packed safetensors bytes (preferred for low-latency in-process).
        extra: Forward-compat slot for backend-specific fields.
    """

    path: Optional[str] = None
    packed: Optional[bytes] = None
    extra: Dict[str, Any] = field(default_factory=dict)
