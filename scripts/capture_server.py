#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tiny request-capture server.

Stands in for vllm-mlx so we can capture the exact /v1/messages body Claude
Code sends, without needing to download a 17GB model first. Dumps every
incoming request to a JSON file and returns a minimal valid Anthropic
response so Claude Code doesn't hang.

Usage:
    python scripts/capture_server.py --out tests/fixtures/claude_code_request.json --port 8765
    # in another terminal:
    ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_API_KEY=x \\
        claude --print "list files"
    # send Ctrl-C to the capture server when done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


def make_app(out_path: Path) -> FastAPI:
    app = FastAPI()
    state = {"count": 0}

    def stub_response(model: str) -> dict:
        return {
            "id": "msg_capture",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        }

    @app.post("/v1/messages")
    async def messages(req: Request):
        body = await req.body()
        try:
            payload = json.loads(body)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        state["count"] += 1
        idx = state["count"]
        # Save the first request to the requested path; later requests get suffixed.
        target = out_path if idx == 1 else out_path.with_name(
            f"{out_path.stem}_{idx}{out_path.suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        sys_field = payload.get("system")
        sys_chars = (
            len(sys_field) if isinstance(sys_field, str)
            else len(json.dumps(sys_field)) if sys_field else 0
        )
        n_tools = len(payload.get("tools") or [])
        n_msgs = len(payload.get("messages") or [])
        print(
            f"[CAPTURE #{idx}] -> {target}  "
            f"system_chars={sys_chars} tools={n_tools} messages={n_msgs} "
            f"stream={payload.get('stream', False)}"
        )

        model = payload.get("model", "capture-stub")
        if payload.get("stream"):
            # Minimal SSE stream that satisfies the Anthropic client.
            def gen():
                start = stub_response(model)
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {**start, 'content': []}})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'ok'}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 1}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return JSONResponse(stub_response(model))

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "capture-stub", "object": "model", "created": int(time.time())}
            ],
        }

    @app.get("/")
    async def root():
        return {"status": "capture-server", "captured": state["count"]}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="tests/fixtures/claude_code_request.json",
        help="Path to write the first captured request body",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    print(f"Capture server starting on http://{args.host}:{args.port}")
    print(f"First request -> {out_path}")
    print("Point Claude Code at this server with:")
    print(f"  ANTHROPIC_BASE_URL=http://{args.host}:{args.port}")
    print("  ANTHROPIC_API_KEY=x")
    print("  claude --print 'list files'")
    print()

    app = make_app(out_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
