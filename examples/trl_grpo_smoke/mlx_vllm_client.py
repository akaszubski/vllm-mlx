# SPDX-License-Identifier: Apache-2.0
"""TRL ``VLLMClient`` subclass that talks to a running vllm-mlx server.

This client replaces the NCCL data plane with file-based safetensors over
vllm-mlx's existing ``/init_weight_transfer_engine`` + ``/start_weight_update``
+ ``/update_weights`` + ``/finish_weight_update`` endpoints (Patch #5
in this fork). MLX has no broadcast primitive and PyTorch Gloo+MPS is broken
(pytorch#160732), so we cannot reuse TRL's stock NCCL transport on Apple
Silicon. See ``realign/docs/research/2026-06-29-trl-on-mps-viability.md`` for
the full design rationale (Phase 1 viability study).

Design notes
------------
- We do **not** call ``super().__init__``. The parent's ``__init__`` calls
  ``is_vllm_available()`` and raises ``ImportError`` if vLLM isn't installed.
  vLLM cannot be installed on MPS, so we reconstruct just the attributes we
  actually need (``session``, ``base_url``, ``host``, ``server_port``,
  ``group_port``, ``rank``, ``communicator``).
- ``init_communicator(device)`` is a no-op against TRL's NCCL machinery and
  instead bootstraps the vllm-mlx weight-transfer engine.
- ``update_named_param(name, weights)`` serializes one tensor to safetensors
  in an atomic-rename temp directory, then POSTs the path to ``/update_weights``.
  The vllm-mlx server's ``MLXWeightTransferEngine`` reads the file via
  ``mx.load`` (mmap-backed, zero-copy on Apple Silicon's unified memory).
- ``close_communicator()`` is a no-op aside from cleaning the temp dir; the
  vllm-mlx server's weight-transfer engine is process-scoped and dies with
  the subprocess.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from safetensors.numpy import save_file

# We import the parent class for subclass identity (so TRL ``isinstance``
# checks pass if any). The parent ``__init__`` is *not* called.
from trl.generation.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


# Mapping from torch dtype string → numpy dtype + mx-friendly dtype name.
# torch dtypes like 'torch.bfloat16' are not directly convertible via
# .numpy() — we cast to float32 first and let the server-side update_info
# carry the original torch dtype string so the trainer can decide.
_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def _torch_to_numpy(t: torch.Tensor) -> tuple[np.ndarray, str]:
    """Move a torch tensor to CPU and convert to a numpy array.

    bfloat16 and float16 are upcast to float32 because mlx-community 4-bit
    Llama-3.2-1B weights are stored as float16/4bit-quantized on the server
    side, and the trainer-side LoRA-adapted weights are typically bf16. We
    upcast to float32 for safe transport; the server's ``mx.load`` will
    accept it.

    Returns:
        ``(np_array, dtype_name)`` where ``dtype_name`` is an mx-friendly
        string like ``"float32"`` (not ``"torch.float32"``).
    """
    t = t.detach().contiguous()
    if t.device.type != "cpu":
        t = t.to("cpu")

    if t.dtype in (torch.bfloat16,):
        # numpy has no bfloat16 — upcast to float32 for transport.
        t = t.to(torch.float32)
    if t.dtype not in _TORCH_TO_NP:
        # Last resort — cast to float32.
        logger.warning(
            "Unmapped torch dtype %s in update_named_param; casting to float32.",
            t.dtype,
        )
        t = t.to(torch.float32)

    np_arr = t.numpy()
    # Strip the 'numpy.' prefix: 'float32' not 'numpy.float32'.
    dtype_name = np_arr.dtype.name
    return np_arr, dtype_name


class MLXVLLMClient(VLLMClient):
    """vllm-mlx-compatible TRL ``VLLMClient``.

    Args:
        base_url: Base URL of the vllm-mlx server (e.g., ``"http://localhost:8765"``).
            If provided, ``host`` and ``server_port`` are ignored.
        host: vllm-mlx server host. Default ``"0.0.0.0"``.
        server_port: vllm-mlx server port. Default ``8000``.
        group_port: Ignored — kept for API compatibility with the parent.
        connection_timeout: Total seconds to wait for the server to be
            healthy. Defaults to 0 (single attempt).
        weight_xfer_dir: Optional path for staging safetensors files. A
            tempdir is created when omitted; cleaned up by
            ``close_communicator``.
        api_key: Optional Bearer token if the server was started with
            ``--api-key``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
        weight_xfer_dir: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # We intentionally skip super().__init__ — see module docstring.
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        if base_url is not None:
            parsed_url = urlparse(base_url)
            try:
                self.host = socket.gethostbyname(parsed_url.hostname)
            except (socket.gaierror, TypeError):
                self.host = parsed_url.hostname or host
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}".rstrip("/")
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"

        self.group_port = group_port
        self.rank = 0
        self.communicator = None  # Sentinel — TRL's close_communicator checks this.

        # File-transfer staging directory.
        self.weight_xfer_dir = Path(
            weight_xfer_dir or tempfile.mkdtemp(prefix="trl-mlx-weights-")
        )
        self.weight_xfer_dir.mkdir(parents=True, exist_ok=True)
        self._weight_xfer_inited = False

        # Health check (parent does this).
        self.check_server(connection_timeout)
        logger.info(
            "MLXVLLMClient ready: base_url=%s, weight_xfer_dir=%s",
            self.base_url,
            self.weight_xfer_dir,
        )

    # ------------------------------------------------------------------
    # Communicator overrides (NCCL → file-based)
    # ------------------------------------------------------------------

    def init_communicator(self, device: Any = 0) -> None:
        """No-op on MPS — bootstrap vllm-mlx's weight transfer engine instead.

        The ``device`` argument is accepted to keep the TRL signature happy
        but is unused. MLX runs on a single Apple Silicon host (world_size=1,
        rank=0).
        """
        if self._weight_xfer_inited:
            return
        url = f"{self.base_url}/init_weight_transfer_engine"
        response = self.session.post(
            url,
            json={
                "init_info": {
                    "world_size": 1,
                    "rank": 0,
                    "backend": "mlx",
                },
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"init_weight_transfer_engine failed "
                f"({response.status_code}): {response.text}"
            )
        self._weight_xfer_inited = True
        logger.info("MLXVLLMClient: weight transfer engine initialized: %s", response.json())

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        """Push a single tensor to the vllm-mlx server via safetensors file.

        Sequence per call:
            1. Convert ``weights`` (torch → numpy, contiguous, dtype-normalized).
            2. Save to ``.tmp.safetensors`` then atomic-rename to final path
               (APFS rename is atomic on the same volume).
            3. POST ``/start_weight_update`` (idempotent for the first call
               in a window; the server tolerates re-entrancy).
            4. POST ``/update_weights`` with ``{"update_info": {...}}``.
            5. POST ``/finish_weight_update``.

        Note: TRL's ``sync_weights`` iterates all parameters one-by-one and
        wraps each call with the (start, update, finish) triplet. The server
        is re-entrant on ``start_weight_update`` — second/third calls during
        an in-progress window are no-ops returning ``{"started": False}``.
        We still call finish at the end of *each* parameter so that an
        external failure between params doesn't leave the scheduler paused.

        Args:
            name: Parameter name (TRL-fixed, e.g. ``model.embed_tokens.weight``).
            weights: Tensor on any device with any dtype. Will be moved to
                CPU and dtype-normalized to a numpy-compatible dtype.
        """
        if not self._weight_xfer_inited:
            # TRL sometimes calls update_named_param before init_communicator
            # if the trainer was constructed without device sync. Be lenient.
            self.init_communicator()

        np_arr, dtype_name = _torch_to_numpy(weights)
        shape = list(np_arr.shape)

        # Safetensors needs a key — use the param name. Slashes are valid in
        # safetensors keys, but we replace '/' in the filename only.
        safe_filename = name.replace("/", "__")
        tmp_path = self.weight_xfer_dir / f"{safe_filename}.tmp.safetensors"
        final_path = self.weight_xfer_dir / f"{safe_filename}.safetensors"

        save_file({name: np_arr}, str(tmp_path))
        os.replace(str(tmp_path), str(final_path))  # atomic on APFS

        # Begin the weight-update window (idempotent on the server side).
        start_resp = self.session.post(
            f"{self.base_url}/start_weight_update",
            json={"is_checkpoint_format": True},
            timeout=30,
        )
        if start_resp.status_code != 200:
            raise RuntimeError(
                f"start_weight_update failed for {name} "
                f"({start_resp.status_code}): {start_resp.text}"
            )

        update_resp = self.session.post(
            f"{self.base_url}/update_weights",
            json={
                "update_info": {
                    "names": [name],
                    "dtype_names": [dtype_name],
                    "shapes": [shape],
                    "path": str(final_path),
                },
            },
            timeout=120,
        )
        if update_resp.status_code != 200:
            # Always try to close the window even on failure.
            try:
                self.session.post(f"{self.base_url}/finish_weight_update", timeout=30)
            except Exception:
                pass
            raise RuntimeError(
                f"update_weights failed for {name} "
                f"({update_resp.status_code}): {update_resp.text}"
            )

        finish_resp = self.session.post(
            f"{self.base_url}/finish_weight_update",
            timeout=30,
        )
        if finish_resp.status_code != 200:
            raise RuntimeError(
                f"finish_weight_update failed for {name} "
                f"({finish_resp.status_code}): {finish_resp.text}"
            )

    def close_communicator(self) -> None:
        """Clean up temp dir. Server-side engine is process-scoped."""
        # TRL's parent close_communicator hits POST /close_communicator/ —
        # we keep that semantic (stub on server side returns 200).
        try:
            self.session.post(f"{self.base_url}/close_communicator/", timeout=5)
        except requests.exceptions.RequestException:
            pass

        # Wipe the staging dir.
        if self.weight_xfer_dir.exists():
            shutil.rmtree(self.weight_xfer_dir, ignore_errors=True)
        self.communicator = None
        self._weight_xfer_inited = False
