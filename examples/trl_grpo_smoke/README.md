# TRL GRPO + vllm-mlx smoke

Phase 2 of the TRL-on-MPS reframe. Runs `huggingface/trl`'s `GRPOTrainer`
(PyTorch + MPS) against a `vllm-mlx` HTTP server for rollouts and file-based
weight transfer.

See `realign/docs/research/2026-06-29-trl-on-mps-viability.md` (Phase 1 viability study) for design rationale.

## What this does

1. Boots a `vllm-mlx` server as a subprocess on `localhost:8765` loading
   `mlx-community/Llama-3.2-1B-Instruct-4bit`.
2. Monkey-patches `trl.generation.vllm_client.VLLMClient` →
   `MLXVLLMClient` (subclass that replaces NCCL with file-based weight
   transfer via `/init_weight_transfer_engine` + `/update_weights`).
3. Patches `torch.cuda.current_device` so TRL's
   `init_communicator(device=torch.cuda.current_device())` call doesn't
   crash on a no-CUDA box.
4. Runs `GRPOTrainer` for up to 50 iterations on a tiny GSM8K-style
   in-memory dataset with LoRA-8.

## Prereqs

- macOS (Apple Silicon) with MPS-enabled PyTorch (`torch.backends.mps.is_available()`)
- `mlx-community/Llama-3.2-1B-Instruct-4bit` already downloaded (mlx_lm format).
  If missing, `vllm-mlx` will fetch it from HuggingFace on first server boot
  (~1 GB).
- `meta-llama/Llama-3.2-1B-Instruct` available for the PyTorch trainer.
  Gated on Hugging Face — you must accept the license at
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct first.
- Python packages: `trl>=0.28`, `peft`, `transformers`, `datasets`, `safetensors`,
  `torch`, `requests`, `numpy`.

## Run it

```bash
cd /path/to/vllm-mlx
PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/trl_grpo_smoke/run_smoke.py --iters 5
```

For the full 50-iter smoke:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/trl_grpo_smoke/run_smoke.py --iters 50
```

If you already have a `vllm-mlx` server running on port 8765, skip the
subprocess boot:

```bash
python examples/trl_grpo_smoke/run_smoke.py --no-server --iters 5
```

## Expected output

- `[trl_smoke] INFO: Booting vllm-mlx server: ...`
- `[trl_smoke] INFO: Server healthy at http://127.0.0.1:8765/health/`
- `[trl_smoke] INFO: TRL monkey-patches installed`
- `[trl_smoke] INFO: MLXVLLMClient ready: base_url=...`
- `[trl_smoke] INFO: MLXVLLMClient: weight transfer engine initialized`
- Trainer progress logs from `transformers.Trainer` showing per-step loss.
- Server log (`logs/trl_smoke_server.log`) shows POST `/generate/` requests
  arriving from the trainer, and POST `/update_weights` for each parameter
  push.

## Knobs

- `--iters N` — max training steps (default 50).
- `--num-generations 2` — group size (lower = less memory).
- `--max-completion-length 128` — completion budget per rollout.
- `--lora-rank 8` — LoRA rank.
- `--server-model PATH` — alternative MLX-format model.
- `--trainer-model PATH` — alternative HF PyTorch model (must share tokenizer
  with `--server-model`).

## Memory notes (M4 with 64 GB GPU-wired)

The Phase 1 study estimated 28–35 GB peak for 1B + LoRA8 + group=2 +
max_comp=128. Run with Activity Monitor open to confirm. If you OOM, drop
`--num-generations 1` and `--max-completion-length 64`.

## Known limitations

- Text-only: the `/generate/` shim ignores `images`.
- Single-host: `world_size=1`, no multi-GPU/multi-machine.
- No constrained decoding: `guided_decoding_regex` / `structured_outputs_regex`
  are accepted but ignored.
- Weight push is one-tensor-per-HTTP-call: TRL iterates `model.named_parameters()`
  and each tensor triggers `/start_weight_update` → `/update_weights` →
  `/finish_weight_update`. The server's re-entrancy is exercised heavily.
