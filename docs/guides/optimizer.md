# Anthropic /v1/messages Prompt Optimizer

The optimizer shrinks Anthropic-style requests **before** they enter the
inference engine. It is most useful with Claude Code, whose system prompt
plus tool schemas can run to ~100K tokens per turn.

It does two things:

- **Tool allowlist**: drop tool definitions whose names aren't on the list
- **Tool stubbing**: replace verbose descriptions with short stubs and
  recursively strip non-structural metadata from `input_schema`

Both transformations are deterministic, so vllm-mlx's prefix cache hits
cleanly. The optimizer is **off by default**; enable it with
`--optimize-prompts`.

## Quick start

```bash
# 1. Start the server with optimization enabled
vllm-mlx serve mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
    --port 8000 \
    --continuous-batching \
    --optimize-prompts \
    --optimize-tool-allowlist Read,Edit,Bash,Grep,Glob,Write \
    --optimize-stub-tools

# 2. Point Claude Code at it
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--optimize-prompts` | off | Master switch. All other `--optimize-*` flags require this. |
| `--optimize-tool-allowlist NAME1,NAME2,…` | none | Case-insensitive. Empty allowlist drops all tools (rarely useful). |
| `--optimize-stub-tools` | off | Description stubs + JSON schema simplification. |

When the server boots with `--optimize-prompts`, you'll see a log line:

```
[OPTIMIZER] enabled allowlist=('Read', 'Edit', ...) stub_tools=True
```

Each request that hits `/v1/messages` then logs an `[OPTIMIZER]` summary line:

```
[OPTIMIZER] tools 47->6 stubbed=6 desc_chars 31200->180
```

If `tool_choice` names a specific tool that the allowlist drops, the
request fails fast with `400 Bad Request` rather than silently asking the
model to call a tool that isn't in the request.

If the optimizer encounters a tool name it has no stub for (e.g. Anthropic
ships a new tool), it logs a one-time warning and falls through with the
original description; the schema simplifier still runs.

## Measuring the impact on your model + hardware

**Token reduction is not the same as latency reduction.** The savings on a
captured Claude Code request can easily reach 90%+ on input-token count
(offline, measured with `tiktoken cl100k_base`), but turn-N-≥-2 TTFT depends
on how much of the prompt is already in the prefix cache. Run the bench
yourself before deciding whether the optimizer is worth enabling for your
combination of model, hardware, and Claude Code session shape.

### 1. Capture a real Claude Code request

Run vllm-mlx with verbose logging or use a reverse proxy (mitmproxy,
`tcpdump`-then-extract, or a small `httpx`/`requests` shim). Save the
JSON body of one POST `/v1/messages` to
`tests/fixtures/claude_code_request.json`. The repo includes
`scripts/capture_server.py` as a no-model FastAPI shim for this.

The minimum saved fields the bench script needs are `system`, `messages`,
`tools`, `model`, and `max_tokens`.

### 2. Count tokens offline

```bash
python scripts/bench_optimizer.py count \
    --request tests/fixtures/claude_code_request.json \
    --allowlist Read,Edit,Bash,Grep,Glob,Write \
    --stub-tools
```

Outputs the token count for `system`, `tools`, and `messages` before and
after optimization. Token counts use `tiktoken cl100k_base` if installed,
else a 4-chars-per-token heuristic.

### 3. Time TTFT against a running server

```bash
# Server WITHOUT optimizer
vllm-mlx serve <model> --continuous-batching &
python scripts/bench_optimizer.py ttft \
    --request tests/fixtures/claude_code_request.json --runs 3
# kill server

# Server WITH optimizer
vllm-mlx serve <model> --continuous-batching \
    --optimize-prompts --optimize-tool-allowlist Read,Edit,Bash --optimize-stub-tools &
python scripts/bench_optimizer.py ttft \
    --request tests/fixtures/claude_code_request.json --runs 3
```

The `ttft` subcommand prints cold (first run) and warm (median of remaining)
TTFT in milliseconds, plus the cold→warm speedup. Compare the two server
configurations side by side.

### Decision gate

- If turn-2 TTFT is already <2s without the optimizer, the incremental win
  is small. Consider focusing effort on a different bottleneck (model size,
  quantization, hardware).
- If turn-1 TTFT is >30s and the optimizer drops it meaningfully, ship it
  by default for your Claude Code workflow.

## What's not in scope here

- **System-prompt filtering.** Claude Code's system prompt has structured
  sections (instructions, environment, etc.); a section-aware filter could
  drop or compress sections the local model doesn't need. It's not in this
  module because the system prompt's structure is brittle to Claude Code
  revisions and the schema/length isn't documented as a stable contract.
- **Static prompt cache injection.** vllm-mlx's native trie-based prefix
  cache covers this at the token level once the prompt is deterministic.

## Implementation notes

- Module: `vllm_mlx/optimizer/`
- Entry point: `optimize_request(req, config) -> (new_req, stats)` —
  returns a `model_copy` of the request with `tools` rewritten; never
  mutates the original.
- Tests: `tests/test_optimizer.py` covers filter, simplify, stub,
  end-to-end, determinism, no-mutation, frozen-config, tool_choice
  validation, and a stub-table regression test.
- Insertion point in the server: `vllm_mlx/server.py`,
  `create_anthropic_message()`, immediately after
  `AnthropicRequest(**body)`.

## Caveats

- The `STUBS` table in `tool_stubber.py` is hand-maintained against current
  Claude Code tool names. If Anthropic ships a new tool, the optimizer
  logs a one-time warning and falls through with the original description.
- **Never rename tools.** vllm-mlx's tool-call response parsers extract
  the tool name from model output; if you rewrite `tool.name`, the model
  may emit a tool call to a name not in the request. Stubbing only touches
  `description` and `input_schema`.
- `cache_control` hints on tool definitions are dropped at request parsing
  (the `AnthropicToolDef` model doesn't carry them), independent of the
  optimizer. If you depend on tool-level cache control, you'll need to
  extend the model first.
- The optimizer is fully deterministic (no timestamps, no randomness), so
  it composes correctly with the prefix cache.

## Credit

The tool-stubbing approach was inspired by anyclaude's TypeScript proxy.
This implementation is a Python re-port that lives server-side so it
composes with vllm-mlx's prefix cache instead of running in a sidecar.
