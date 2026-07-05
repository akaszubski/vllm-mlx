#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 smoke harness: TRL GRPOTrainer (PyTorch+MPS) → vllm-mlx server.

Boots a vllm-mlx server as a subprocess, monkey-patches TRL to use
``MLXVLLMClient`` instead of the stock NCCL-backed ``VLLMClient``, then runs
a 50-iteration GRPO smoke on a tiny GSM8K-style dataset.

The goal is to demonstrate the integration runs end-to-end on Apple Silicon:

  - Server starts cleanly
  - Trainer reaches iter 1 (generation visible in server logs)
  - Weights pushed at iter 1 (visible in /update_weights server logs)
  - 50 iters or a real blocker surfaced
  - Loss trajectory is non-degenerate

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/trl_grpo_smoke/run_smoke.py \\
        --iters 5 \\
        --server-model mlx-community/Llama-3.2-1B-Instruct-4bit \\
        --trainer-model meta-llama/Llama-3.2-1B-Instruct

Prereqs:
    - mlx-community/Llama-3.2-1B-Instruct-4bit cached in ~/Models (mlx_lm format)
    - meta-llama/Llama-3.2-1B-Instruct cached for the PyTorch-side trainer
    - transformers + trl + peft installed
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# MPS fallback for ops without Metal kernels (SDPA backward, etc.).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Avoid HF tokenizer fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Make 'examples.trl_grpo_smoke' importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("trl_smoke")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def boot_server(model: str, port: int, log_path: Path) -> subprocess.Popen:
    """Launch vllm-mlx server as a subprocess, redirecting stdout/stderr to log_path."""
    logger.info("Booting vllm-mlx server: model=%s port=%d", model, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_mlx.server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--continuous-batching",  # required for engine.generate to surface logprobs
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )
    logger.info("Server PID=%d, logs → %s", proc.pid, log_path)
    return proc


def wait_for_server(
    host: str, port: int, *, max_seconds: float = 240.0
) -> None:
    """Poll /health/ until ready or timeout."""
    import requests

    deadline = time.time() + max_seconds
    url = f"http://{host}:{port}/health/"
    last_err: str | None = None
    while time.time() < deadline:
        # Quick TCP probe first to avoid noisy HTTP errors during boot.
        if not _port_open(host, port):
            time.sleep(1.0)
            continue
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("Server healthy at %s", url)
                return
            last_err = f"status={resp.status_code} body={resp.text[:200]}"
        except requests.exceptions.RequestException as exc:
            last_err = str(exc)
        time.sleep(2.0)
    raise TimeoutError(
        f"Server at {url} did not become ready within {max_seconds}s. "
        f"Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------


def build_dataset():
    """Return a small in-memory GSM8K-style dataset of (prompt, answer) pairs."""
    from datasets import Dataset

    items = [
        ("If Tim has 3 apples and Jane gives him 2 more, how many apples does Tim have?", "5"),
        ("A train travels 60 miles in 2 hours. What is its average speed in mph?", "30"),
        ("What is 12 + 8?", "20"),
        ("What is 9 times 7?", "63"),
        ("A pizza has 8 slices. 3 are eaten. How many are left?", "5"),
        ("If a book has 240 pages and you read 60 a day, how many days to finish?", "4"),
        ("What is half of 50?", "25"),
        ("A bag of marbles has 15 red and 10 blue. How many marbles in total?", "25"),
        ("If you split 36 cookies evenly between 4 friends, how many does each get?", "9"),
        ("What is 100 minus 37?", "63"),
        ("Sally is 12. Her brother is 3 years younger. How old is her brother?", "9"),
        ("A square has side length 6. What is its area?", "36"),
        ("How many minutes in 3 hours?", "180"),
        ("If 7 apples cost $14, how much does one apple cost?", "2"),
        ("What is 8 squared?", "64"),
        ("If you have 5 quarters, how many cents do you have?", "125"),
    ]
    return Dataset.from_list(
        [
            {"prompt": f"Question: {q}\nAnswer:", "ground_truth": a}
            for q, a in items
        ]
    )


def accuracy_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """Reward = 1.0 if the ground-truth string appears in the completion, else 0.0.

    Trivially noisy on the smoke; the goal is to confirm the trainer loop runs
    and reward signal flows through GRPO, not to actually learn.
    """
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truth):
        rewards.append(1.0 if gt.strip() in completion else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# TRL monkey-patches
# ---------------------------------------------------------------------------


def install_mlx_patches() -> None:
    """Patch TRL+torch surfaces that hard-reference CUDA.

    Specifically:
      - ``trl.import_utils.is_vllm_available`` → ``lambda: True`` so
        ``VLLMGeneration._initialize_vllm`` doesn't bail on "vLLM not
        installed".
      - ``trl.generation.vllm_client.VLLMClient`` → ``MLXVLLMClient``
        (subclass with NCCL replaced by file-based weight transfer).
      - ``trl.generation.vllm_generation.VLLMClient`` → same.
      - ``torch.cuda.current_device`` → returns ``"mps"`` so the
        ``init_communicator(device=torch.cuda.current_device())`` call in
        ``VLLMGeneration`` doesn't raise on a no-CUDA box.

    Call this BEFORE constructing the GRPOTrainer.
    """
    import torch

    from examples.trl_grpo_smoke.mlx_vllm_client import MLXVLLMClient

    # IMPORTANT: import order matters. We must import vllm_generation FIRST,
    # while ``is_vllm_available()`` still returns False — its module-level
    # ``if is_vllm_available(): import vllm`` block then evaluates to False
    # and we don't crash. Only THEN do we patch the name back to True so
    # runtime calls (in __init__, lines around 264) see the right value.
    import trl.import_utils as _import_utils
    import trl.generation.vllm_generation as _vllm_gen
    import trl.generation.vllm_client as _vllm_client_mod

    # 1) is_vllm_available → True for runtime checks.
    _import_utils.is_vllm_available = lambda: True
    _vllm_gen.is_vllm_available = lambda: True

    # 2) Swap VLLMClient everywhere it was imported.
    _vllm_gen.VLLMClient = MLXVLLMClient
    _vllm_client_mod.VLLMClient = MLXVLLMClient

    # 3) torch.cuda.current_device shim so init_communicator(device=...) doesn't crash.
    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: torch.device("mps") if torch.backends.mps.is_available() else "cpu"  # type: ignore[assignment]
        # Some TRL paths also call torch.cuda.get_device_properties — stub if needed.
        try:
            torch.cuda.get_device_properties  # noqa: B018
        except AttributeError:
            pass

    logger.info("TRL monkey-patches installed (is_vllm_available=True, VLLMClient=MLXVLLMClient)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--server-model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="MLX-format model the vllm-mlx server loads",
    )
    p.add_argument(
        "--trainer-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF PyTorch model the GRPOTrainer trains (must match server's tokenizer)",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--iters", type=int, default=50, help="Max training steps")
    p.add_argument(
        "--no-server",
        action="store_true",
        help="Assume the server is already running on --port; don't spawn one",
    )
    p.add_argument(
        "--server-log",
        default=str(_REPO_ROOT / "logs" / "trl_smoke_server.log"),
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=2,
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=128,
    )
    args = p.parse_args()

    proc: subprocess.Popen | None = None
    if not args.no_server:
        proc = boot_server(args.server_model, args.port, Path(args.server_log))

    try:
        wait_for_server("127.0.0.1", args.port, max_seconds=240.0)

        install_mlx_patches()

        # Delay TRL imports until AFTER monkey-patches are in place.
        import torch
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        ds = build_dataset()

        # TRL requires `per_device_train_batch_size * gradient_accumulation_steps`
        # to be divisible by num_generations. Default num_generations=2 so we
        # need batch=2 OR grad_accum=2. Pick batch=num_generations to keep
        # the math symmetric.
        config = GRPOConfig(
            output_dir="outputs/trl_grpo_smoke",
            per_device_train_batch_size=args.num_generations,
            gradient_accumulation_steps=1,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            max_steps=args.iters,
            learning_rate=1e-6,
            beta=0.0,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_base_url=f"http://127.0.0.1:{args.port}",
            vllm_server_timeout=240.0,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            bf16=False,
            fp16=False,
            use_cpu=False,
        )

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(args.trainer_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Wrap reward to read ground_truth from the dataset row keyword.
        # Pop it from kwargs before forwarding so we don't pass it twice.
        def reward_fn(completions, **kwargs):
            gt = kwargs.pop("ground_truth", [])
            return accuracy_reward(completions, gt, **kwargs)

        trainer = GRPOTrainer(
            model=args.trainer_model,
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        logger.info("Starting GRPO training for %d iters", args.iters)
        result = trainer.train()
        logger.info("Training finished: %s", result)
        return 0
    finally:
        if proc is not None:
            logger.info("Terminating server PID=%d", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop in 15s; killing")
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
