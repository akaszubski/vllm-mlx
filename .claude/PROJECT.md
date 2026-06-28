# vllm-mlx (akaszubski fork)

A high-performance, OpenAI- and Anthropic-compatible inference server for Apple Silicon, built on MLX. Runs LLMs, vision, audio, embeddings, and reranking from a single process — including hosting Claude Code locally via the `/v1/messages` endpoint.

Fork of `waybarrios/vllm-mlx`. Upstream goal: contribute the MLX `WeightTransferEngine` backend back to `vllm-project/vllm`.

## GOALS

1. **OpenAI + Anthropic API surface** on Apple Silicon. Both `/v1/chat/completions` and `/v1/messages` are first-class, including the `ANTHROPIC_BASE_URL=http://localhost:8000 + claude` flow for running Claude Code locally against this server.
2. **Performance moat over `mlx-lm` / `Ollama`**: continuous batching, paged KV cache, prefix caching (trie), SSD-tiered KV spill, warm-prompt preload, sparse prefill. Throughput and TTFT are explicit success metrics.
3. **Multimodal serving** (text, vision, video, audio in + native TTS / STT out, embeddings, rerank, MCP tool calling) from one server, one process, no model conversion.
4. **RLHF/GRPO-ready feature surface** — group sampling (`SamplingParams.n`), per-token logprobs, hot weight reload — so the realign training pipeline can colocate trainer + rollouts on the same Mac. (One consumer among several; not the only purpose of the fork.)
5. **Minimal divergence from `vllm-project/vllm`** so the MLX backend can be upstreamed (`WeightTransferEngine` and sampler hooks land in mainline vLLM long-term).

## SCOPE — In Scope

- HTTP API schema extensions: OpenAI `/v1/*` and Anthropic `/v1/messages` (chat, completions, embeddings, rerank, responses).
- New API surface needed for RLHF (e.g., `SamplingParams.n`, `logprobs` field, weight-transfer endpoints).
- Performance work: KV-cache tuning, continuous-batching schedulers, prefix-cache eviction, sparse prefill, MoE top-k, speculative decoding glue.
- Multimodal endpoints and processors (vision, audio in/out, TTS voices, STT pipelines, image/video).
- MLX backend implementations of vLLM's documented extension interfaces (`WeightTransferEngine`, sampler hooks, etc.).
- Reasoning parsers (Qwen3, DeepSeek-R1, Gemma, GPT-OSS, Harmony) and tool-call parsers.
- Observability: Prometheus `/metrics`, `vllm-mlx bench-serve` harness, structured logs.
- Fork-local patches that will be submitted upstream (track via realign issue #1310 for the upstream-contribution thread).
- Schema + Pydantic model changes in `vllm_mlx/api/models.py`.
- Handler-level fan-out / dispatch changes in `vllm_mlx/server.py`.
- Tests in `tests/` covering schema fields, handler behavior, and perf regressions (schema-level unit tests in-scope per patch; full e2e integration tests OUT of scope per patch unless explicitly required).

## SCOPE — Out of Scope

- CUDA / non-Apple-Silicon backends (the fork is Apple-Silicon-only; upstream-bound code MUST stay portable).
- Model training itself (training lives in realign; vllm-mlx serves rollouts and general inference, never gradient steps).
- Multi-host / distributed inference (single-machine Apple Silicon only; NFS for shared model cache is fine, but no inter-host scheduling).
- Quantization research (use `mlx-lm` quantization utilities directly; the fork consumes already-quantized weights).
- `mlx_lm.server` architectural pivot — rejected. The fork stays on vLLM's HTTP API surface, not Apple's variant.
- Marketing site / standalone docs site (the README + `docs/benchmarks/` + CHANGELOG are sufficient).

## CONSTRAINTS

- **Architectural consistency with upstream vLLM**: every new public API (request schema field, HTTP endpoint, backend interface) MUST mirror upstream `vllm-project/vllm` naming and semantics. Same TRL config must work across MPS + CUDA.
- **OpenAI-compatible AND Anthropic-compatible** endpoints stay in lockstep — schema changes to one usually need a parallel update for the other (e.g., if `/v1/chat/completions` gains `n=`, the Anthropic shim either supports it or rejects it cleanly).
- **Optional MLX dependency at install time**: code paths that import `mlx`/`mlx-lm` MUST be guarded so non-Apple-Silicon installs don't break (matters for upstream contribution).
- **Continuous batching + paged KV + prefix cache** are load-bearing and MUST be preserved across all patches. Perf regressions are blocking findings, not stylistic notes.
- **Tests use the existing `pytest.ini` config**: don't add new pytest configs without a clear reason.
- **CHANGELOG.md MUST be updated for every user-visible schema or endpoint change**.

## Tracking

- Patches related to the **realign / Plan-1.5** workstream are tracked as issues in `akaszubski/realign`, NOT in this repo's issue tracker:
  - Plan-1.5 umbrella: `realign#1307`
  - Patch #1 SamplingParams.n: `realign#1308`
  - Patch #2 per-token logprobs: `realign#1251`
  - Patch #3 WeightTransferEngine MLX: `realign#1309`
  - Patch #4 upstream contribution: `realign#1310`
- General vllm-mlx work (perf, multimodal, Anthropic API, bug fixes) uses this repo's own issue tracker.
- Branch convention for realign patches: `feat/rlvr-{slug}` (e.g., `feat/rlvr-n-param`). General work: standard `feat/`, `fix/`, `perf/` prefixes.
- Push target: `akaszubski/vllm-mlx` (the user's fork, NOT `waybarrios/vllm-mlx` upstream). Upstream contributions go via a separate PR thread tracked under `realign#1310`.

## Backstory

This fork inherits the full vllm-mlx product surface (OpenAI + Anthropic APIs, continuous batching, multimodal, perf) from `waybarrios/vllm-mlx`. The primary local use case is running Claude Code against `ANTHROPIC_BASE_URL=http://localhost:8000` for low-latency, on-device inference on Apple Silicon — the **"localclaude"** workflow.

The `akaszubski/` fork additionally exists to add 3 features required by GRPO rollouts that upstream lacks:

1. `SamplingParams.n` for group sampling (Plan-1.5 patch #1)
2. Per-token logprobs for IS-ratio computation (Plan-1.5 patch #2)
3. Weight reload endpoint for after-each-step weight push from the trainer (Plan-1.5 patch #3)

Plan-1.5 patches these 3 gaps. After local validation in a 50-iter GRPO smoke, the MLX backend gets contributed upstream (Plan-1.5 patch #4) so the fork can eventually collapse back into mainline vLLM.

RLHF/GRPO is one consumer of this fork. Local Claude Code, general LLM serving, multimodal, and perf research are equally first-class.
