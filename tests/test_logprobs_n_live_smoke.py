# SPDX-License-Identifier: Apache-2.0
"""
Live-HTTP smoke validation for Plan-1.5 patches #1 (n group sampling) and
#2 (per-token logprobs).

This module is opt-in: it is decorated with the ``live_server`` marker so
``pytest`` does not collect it by default (see ``pytest.ini``). To run it
explicitly::

    pytest -m live_server tests/test_logprobs_n_live_smoke.py -v

The fixture boots a real ``python -m vllm_mlx.server`` subprocess on an
ephemeral free port, waits for ``/health`` to report ``model_loaded: True``,
yields the base URL, and reaps the process on teardown (SIGTERM, then
SIGKILL after 10s). All four tests below issue real HTTP requests against
``/v1/chat/completions``.

Why this exists: unit coverage in ``test_logprobs_surface.py`` exercises the
schema and helpers in-memory, but does not exercise the actual end-to-end
wire format the trainer hits. This smoke catches integration bugs the unit
tests cannot.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("mlx")

# Hardcoded per plan: vetted server default, present on this machine.
MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
MODEL_CACHE_DIR = (
    Path.home() / "Models" / "hub" / "models--mlx-community--Llama-3.2-3B-Instruct-4bit"
)

# Module-level marker: applied to every test in this file. The
# ``enable_socket`` marker tells ``pytest-socket`` (a transitive test-dep
# that disables real sockets by default) to allow real TCP for every test
# here -- the entire module exists to talk to a real HTTP server.
pytestmark = [pytest.mark.live_server, pytest.mark.enable_socket]


def _pick_free_port() -> int:
    """Bind a socket to port 0 to ask the OS for an unused port, then close it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_healthy(base_url: str, timeout_s: float = 90.0) -> dict:
    """Poll ``GET /health`` until ``model_loaded`` is True.

    Backoff: 0.5s -> 2.0s (cap). Raises ``pytest.fail`` if the server does
    not report healthy within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.5
    last_payload: dict | str = "<no response>"
    last_exc: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5.0)
            last_payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            if resp.status_code == 200 and isinstance(last_payload, dict) and last_payload.get("model_loaded") is True:
                return last_payload
        except Exception as exc:  # noqa: BLE001 - intentional broad catch while polling
            last_exc = repr(exc)
        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)
    pytest.fail(
        f"server at {base_url} did not become healthy in {timeout_s:.0f}s. "
        f"last /health payload: {last_payload!r}; last exception: {last_exc}"
    )


@pytest.fixture(scope="session")
def live_server() -> Iterator[str]:
    """Boot a real vllm_mlx server subprocess on a free port for the test session."""
    if not MODEL_CACHE_DIR.exists():
        pytest.skip(
            f"model snapshot not on disk at {MODEL_CACHE_DIR}; cache the model first"
        )

    # ``pytest-socket`` (a transitive test-dep) disables real sockets by
    # default. This whole module's point is to talk to a real HTTP server,
    # so re-enable sockets for the duration of the session. Best-effort:
    # if the plugin isn't loaded the import will fail and we just continue.
    try:
        from pytest_socket import enable_socket  # type: ignore[import-not-found]

        enable_socket()
    except ImportError:
        pass

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    # Capture server logs to a temp file so debug breadcrumbs survive when
    # the test fails. Set ``VLLM_MLX_SMOKE_LOGS=1`` to print the tail on
    # failure / teardown.
    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"vllm_mlx_smoke_{port}.log"
    log_fh = open(log_path, "w", buffering=1)

    # ``--continuous-batching`` is required: it selects ``BatchedEngine``,
    # which is the only engine wired with per-token logprobs (per the
    # patch #2 CHANGELOG note). ``SimpleEngine`` does not populate them
    # and -- as of Python 3.14 + mlx_lm -- also crashes with a thread-bound
    # GPU stream error inside the ``asyncio.to_thread`` worker.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_mlx.server",
            "--model",
            MODEL_ID,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--continuous-batching",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # New process group so we can SIGTERM the entire tree if needed.
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    def _reap() -> None:
        """Best-effort process reap; safe to call multiple times."""
        if proc.poll() is not None:
            return
        try:
            try:
                # Signal the whole process group first, fall back to the leader.
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                proc.wait()
        except Exception:
            # We're in teardown; never propagate.
            pass

    # Belt-and-suspenders: ensure reap on interpreter exit even if pytest crashes.
    atexit.register(_reap)

    try:
        _wait_for_healthy(base_url, timeout_s=90.0)
        yield base_url
    finally:
        _reap()
        try:
            log_fh.close()
        except Exception:
            pass
        # Surface the tail when debugging.
        if os.environ.get("VLLM_MLX_SMOKE_LOGS") == "1":
            try:
                tail = log_path.read_text().splitlines()[-80:]
                print(f"\n=== vllm_mlx server log tail ({log_path}) ===")
                for line in tail:
                    print(line)
                print("=== end ===\n")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers for chat completion requests
# ---------------------------------------------------------------------------


_LOGPROB_KEYS = {"token", "logprob", "bytes", "top_logprobs"}


def _post_chat(base_url: str, payload: dict, timeout_s: float = 120.0) -> dict:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    return resp.json()


def _user_msg(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Test A: logprobs=true (no top_logprobs) -> populated content array
# ---------------------------------------------------------------------------


def test_logprobs_populates_content_array(live_server: str) -> None:
    """Patch #2 acceptance: ``choices[0].logprobs.content`` matches completion_tokens.

    Each entry MUST have the OpenAI-shaped keys ``{token, logprob, bytes, top_logprobs}``
    where ``top_logprobs`` is an empty list when the request did not ask for top-k
    (matches ``_build_choice_logprobs`` server-side behavior).
    """
    payload = {
        "model": MODEL_ID,
        "messages": _user_msg("Say the single word: hello"),
        "max_tokens": 16,
        "temperature": 0.0,
        "logprobs": True,
    }
    body = _post_chat(live_server, payload)

    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["logprobs"] is not None, f"logprobs missing on choice: {choice}"

    content = choice["logprobs"]["content"]
    assert isinstance(content, list) and len(content) > 0, (
        f"empty logprobs.content: {choice['logprobs']}"
    )

    completion_tokens = body["usage"]["completion_tokens"]
    assert len(content) == completion_tokens, (
        f"len(content)={len(content)} != completion_tokens={completion_tokens}"
    )

    for i, entry in enumerate(content):
        assert _LOGPROB_KEYS.issubset(entry.keys()), (
            f"entry[{i}] missing keys; got {set(entry.keys())}"
        )
        assert isinstance(entry["token"], str)
        assert isinstance(entry["logprob"], float)
        # bytes may be None for some BPE tokens (per OpenAI spec); when present
        # it must be a list of ints.
        if entry["bytes"] is not None:
            assert isinstance(entry["bytes"], list)
            assert all(isinstance(b, int) for b in entry["bytes"])
        # top_logprobs was not requested -> server emits an empty list
        # (see vllm_mlx.server._build_choice_logprobs).
        assert entry["top_logprobs"] == [], (
            f"entry[{i}].top_logprobs should be [] when not requested; "
            f"got {entry['top_logprobs']!r}"
        )


# ---------------------------------------------------------------------------
# Test B: n=4 -> exactly 4 choices covering indices {0,1,2,3}
# ---------------------------------------------------------------------------


def test_n_group_sampling_returns_four_choices(live_server: str) -> None:
    """Patch #1 acceptance: ``n=4`` yields 4 choices with index set ``{0,1,2,3}``."""
    payload = {
        "model": MODEL_ID,
        "messages": _user_msg("Write a haiku about the ocean."),
        "max_tokens": 32,
        "temperature": 0.8,
        "n": 4,
    }
    body = _post_chat(live_server, payload)

    choices = body["choices"]
    assert len(choices) == 4, f"expected 4 choices, got {len(choices)}"
    indices = {c["index"] for c in choices}
    assert indices == {0, 1, 2, 3}, f"unexpected index set: {indices}"


# ---------------------------------------------------------------------------
# Test C: n=4 + logprobs=True -> populated logprobs on all 4 choices
# ---------------------------------------------------------------------------


def test_n_with_logprobs_populates_every_choice(live_server: str) -> None:
    """Patches #1 + #2 composed: ``n=4`` + ``logprobs=true`` -> all 4 carry logprobs."""
    payload = {
        "model": MODEL_ID,
        "messages": _user_msg("List three colors."),
        "max_tokens": 24,
        "temperature": 0.8,
        "n": 4,
        "logprobs": True,
    }
    body = _post_chat(live_server, payload)

    choices = body["choices"]
    assert len(choices) == 4

    for choice in choices:
        assert choice["logprobs"] is not None, (
            f"choice index={choice['index']} missing logprobs"
        )
        content = choice["logprobs"]["content"]
        assert isinstance(content, list) and len(content) > 0, (
            f"choice index={choice['index']} has empty logprobs.content"
        )


# ---------------------------------------------------------------------------
# Test D: top_logprobs=5 -> 5 entries per slot, sorted non-increasing
# ---------------------------------------------------------------------------


def test_top_logprobs_returns_sorted_top_k(live_server: str) -> None:
    """Patch #2: ``top_logprobs=5`` yields exactly 5 entries per content slot,
    sorted by ``logprob`` in non-increasing order (ties allowed)."""
    payload = {
        "model": MODEL_ID,
        "messages": _user_msg("Say: yes"),
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    body = _post_chat(live_server, payload)

    content = body["choices"][0]["logprobs"]["content"]
    assert len(content) > 0

    for i, entry in enumerate(content):
        top = entry["top_logprobs"]
        assert len(top) == 5, (
            f"entry[{i}].top_logprobs length={len(top)} (expected 5)"
        )
        lps = [t["logprob"] for t in top]
        # Non-increasing: allow ties.
        for j in range(len(lps) - 1):
            assert lps[j] >= lps[j + 1], (
                f"entry[{i}].top_logprobs not sorted at j={j}: "
                f"{lps[j]} < {lps[j + 1]}"
            )
        for alt in top:
            assert isinstance(alt["token"], str)
            assert isinstance(alt["logprob"], float)
            if alt.get("bytes") is not None:
                assert isinstance(alt["bytes"], list)
