# Changelog

All notable user-visible changes to vllm-mlx are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
