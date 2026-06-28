# SPDX-License-Identifier: Apache-2.0
"""Weight-transfer package: abstract base + MLX backend.

Importing this module registers the MLX backend with
`WeightTransferEngineFactory` lazily (the factory holds module/class strings;
the MLX backend module is only imported on first `create_engine('mlx', ...)`).
"""

from .base import (
    WeightTransferEngine,
    WeightTransferEngineFactory,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from .types import (
    MLXWeightTransferInitInfo,
    MLXWeightTransferUpdateInfo,
)

# Lazy registration — the mlx_backend module is NOT imported here.
WeightTransferEngineFactory.register_engine(
    "mlx",
    "vllm_mlx.weight_transfer.mlx_backend",
    "MLXWeightTransferEngine",
)


def _get_mlx_engine_class():
    """Helper to materialize the MLX backend class (forces lazy import)."""
    return WeightTransferEngineFactory.get_engine_class("mlx")


__all__ = [
    "WeightTransferEngine",
    "WeightTransferEngineFactory",
    "WeightTransferInitInfo",
    "WeightTransferUpdateInfo",
    "MLXWeightTransferInitInfo",
    "MLXWeightTransferUpdateInfo",
]
