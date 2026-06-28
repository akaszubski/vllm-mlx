# Changelog

All notable user-visible changes to vllm-mlx are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `SamplingParams.n` group sampling (1..64) and `best_of` field for `/v1/chat/completions` and `/v1/completions`. Enables GRPO/RLHF rollout group sampling against a single colocated MLX server. Streaming (`stream=True`) with `n>1` rejected with HTTP 400 (OpenAI-compatible error envelope). Prefix cache automatically dedups the shared prompt-KV across rollouts. Anthropic `/v1/messages` is unaffected (native Anthropic API does not expose `n`). E2E model-load integration tests deferred to a follow-up patch. (realign#1308 / Plan-1.5 patch #1)
- MLX-backend `WeightTransferEngine` for RLHF/GRPO weight hot-reload.
  Registers via `WeightTransferEngineFactory.register_engine('mlx', ...)`
  with lazy import so non-Apple-Silicon installs are unaffected. (realign#1307, realign#1309)
- 7 new HTTP routes mirroring upstream vLLM async-RL surface (all auth-gated):
  `POST /init_weight_transfer_engine`, `POST /start_weight_update`,
  `POST /update_weights`, `POST /finish_weight_update`, `POST /pause`,
  `POST /resume`, `GET /get_world_size`. (realign#1309)
- `Scheduler.pause(mode, clear_cache, timeout_s)` and `Scheduler.resume()`
  with `mode in {'abort', 'wait', 'keep'}` matching upstream semantics. (realign#1309)
- Pydantic request models: `WeightTransferInitRequest`,
  `StartWeightUpdateRequest`, `WeightTransferUpdateRequest`, `PauseRequest`
  in `vllm_mlx/api/models.py`. (realign#1309)

> **Note**: Trainer-level end-to-end (`realign train --reload-every 1`) and LoRA +
> grad-step deadlock validation are tracked separately (realign#1309 follow-up);
> out of scope for this patch.
