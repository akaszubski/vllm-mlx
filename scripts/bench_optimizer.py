#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Token-accounting and TTFT benchmark harness for the prompt optimizer.

Usage:

  # Token accounting only (no server required)
  python scripts/bench_optimizer.py count \\
      --request tests/fixtures/claude_code_request.json

  # End-to-end TTFT against a running vllm-mlx server
  python scripts/bench_optimizer.py ttft \\
      --request tests/fixtures/claude_code_request.json \\
      --url http://localhost:8000 \\
      --runs 3

  # Compare optimizer on/off (you must restart the server with the desired flags
  # between the two runs; this script only times whichever server is responding)

The "count" subcommand simulates the optimizer locally without hitting the
server, so you can see token-reduction numbers before deciding whether to
restart the server with --optimize-prompts.

Token counting uses tiktoken's cl100k_base if available, else falls back to a
4-chars-per-token heuristic (logged so you can interpret results accordingly).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vllm_mlx.api.anthropic_models import AnthropicRequest  # noqa: E402
from vllm_mlx.optimizer import OptimizerConfig, optimize_request  # noqa: E402


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _serialize_for_count(req: AnthropicRequest) -> dict[str, str]:
    """Return a dict of section -> serialized text for token counting."""
    sections: dict[str, str] = {}
    sys_val = req.system
    if isinstance(sys_val, str):
        sections["system"] = sys_val
    elif isinstance(sys_val, list):
        sections["system"] = json.dumps(sys_val, ensure_ascii=False)
    else:
        sections["system"] = ""

    if req.tools:
        sections["tools"] = json.dumps(
            [t.model_dump() for t in req.tools], ensure_ascii=False
        )
    else:
        sections["tools"] = ""

    sections["messages"] = json.dumps(
        [m.model_dump() for m in req.messages], ensure_ascii=False
    )
    return sections


def _print_count_report(label: str, sections: dict[str, str]) -> dict[str, int]:
    counts = {k: _count_tokens(v) for k, v in sections.items()}
    counts["chars_total"] = sum(len(v) for v in sections.values())
    counts["tokens_total"] = sum(counts[k] for k in ("system", "tools", "messages"))
    print(f"--- {label} ---")
    for k in ("system", "tools", "messages"):
        print(f"  {k:<10} chars={len(sections[k]):>8d}  tokens={counts[k]:>6d}")
    print(f"  {'TOTAL':<10} chars={counts['chars_total']:>8d}  tokens={counts['tokens_total']:>6d}")
    return counts


def cmd_count(args: argparse.Namespace) -> int:
    body = json.loads(Path(args.request).read_text())
    req = AnthropicRequest(**body)

    print(f"Request: {args.request}")
    print(f"Tools in request: {len(req.tools or [])}")
    print()

    baseline = _serialize_for_count(req)
    base_counts = _print_count_report("BASELINE (no optimizer)", baseline)
    print()

    cfg = OptimizerConfig(
        enabled=True,
        tool_allowlist=(
            tuple(name.strip() for name in args.allowlist.split(",") if name.strip())
            if args.allowlist
            else None
        ),
        stub_tools=args.stub_tools,
    )
    optimized, stats = optimize_request(req, cfg)
    sections = _serialize_for_count(optimized)
    opt_counts = _print_count_report("OPTIMIZED", sections)
    print()

    delta_tokens = base_counts["tokens_total"] - opt_counts["tokens_total"]
    pct = (
        100.0 * delta_tokens / base_counts["tokens_total"]
        if base_counts["tokens_total"]
        else 0.0
    )
    print(
        f"Reduction: {delta_tokens} tokens ({pct:.1f}%) "
        f"| tools {stats.tools_before}->{stats.tools_after} "
        f"stubbed={stats.tools_stubbed}"
    )
    return 0


def _post_messages(url: str, body: dict[str, Any], timeout: float) -> tuple[float, int]:
    """POST /v1/messages, return (TTFT seconds, status_code).

    For streaming requests, TTFT is time-to-first-byte. For non-streaming, it
    is total request time.
    """
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/messages",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # First byte read = TTFT for streaming, full response for non-streaming
            resp.read(1)
            ttft = time.perf_counter() - start
            return ttft, resp.status
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  request failed after {elapsed:.2f}s: {e}", file=sys.stderr)
        return elapsed, -1


def cmd_ttft(args: argparse.Namespace) -> int:
    body = json.loads(Path(args.request).read_text())
    body.setdefault("stream", True)

    print(f"Server: {args.url}")
    print(f"Request: {args.request}")
    print(f"Stream: {body['stream']}  | runs: {args.runs}")
    print()

    timings: list[float] = []
    for i in range(args.runs):
        ttft, status = _post_messages(args.url, body, args.timeout)
        timings.append(ttft)
        label = "cold" if i == 0 else f"warm{i}"
        print(f"  run {i+1} ({label}): {ttft*1000:.0f} ms (status {status})")

    if timings:
        print()
        print(f"min={min(timings)*1000:.0f}ms  median={statistics.median(timings)*1000:.0f}ms  max={max(timings)*1000:.0f}ms")
        if len(timings) >= 2:
            cold = timings[0]
            warm_med = statistics.median(timings[1:])
            speedup = cold / warm_med if warm_med > 0 else float("inf")
            print(f"cold->warm speedup: {speedup:.2f}x  (cold={cold*1000:.0f}ms warm_median={warm_med*1000:.0f}ms)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("count", help="Token accounting (offline, no server)")
    p_count.add_argument("--request", required=True, help="Path to a captured /v1/messages request body (JSON)")
    p_count.add_argument("--allowlist", default=None, help="Comma-separated tool names")
    p_count.add_argument("--stub-tools", action="store_true", help="Apply tool stubbing")
    p_count.set_defaults(func=cmd_count)

    p_ttft = sub.add_parser("ttft", help="Time-to-first-byte against a running server")
    p_ttft.add_argument("--request", required=True, help="Path to a captured /v1/messages request body (JSON)")
    p_ttft.add_argument("--url", default=os.environ.get("VLLM_MLX_URL", "http://localhost:8000"))
    p_ttft.add_argument("--runs", type=int, default=3, help="Total runs (first is cold, rest are warm)")
    p_ttft.add_argument("--timeout", type=float, default=120.0)
    p_ttft.set_defaults(func=cmd_ttft)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
