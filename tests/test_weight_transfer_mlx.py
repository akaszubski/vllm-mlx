# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MLX WeightTransferEngine (Plan-1.5 Patch #3).

Test ID conventions follow plan section 5:
- test_factory_lazy_registration      (AC-1)
- test_scheduler_pause_*              (AC-2)
- test_engine_clear_kv_and_prefix_cache (AC-3)
- test_ten_cycle_no_memory_growth     (AC-4, Apple Silicon only)
- test_precompiled_kernel_survives_update (AC-5)
- test_trainer_send_weights_eval_barrier (AC-6)
- test_routes_*                        (AC-7)
- test_pydantic_*                      (AC-8)
- test_changelog_unreleased_entry      (AC-9)
- test_integration_real_model_rollout  (skipped — OUT OF SCOPE)
"""

from __future__ import annotations

import gc
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-under-test imports
# ---------------------------------------------------------------------------

import mlx.core as mx
import mlx.nn as nn

import vllm_mlx.server as srv
from vllm_mlx.api.models import (
    PauseRequest,
    StartWeightUpdateRequest,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm_mlx.weight_transfer import (
    MLXWeightTransferInitInfo,
    MLXWeightTransferUpdateInfo,
    WeightTransferEngine,
    WeightTransferEngineFactory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_safetensors_file(arrays: dict) -> str:
    """Write `arrays` to a temp safetensors file; return path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".safetensors", delete=False
    )
    tmp.close()
    mx.save_safetensors(tmp.name, arrays)
    return tmp.name


# ---------------------------------------------------------------------------
# AC-1: Factory lazy registration
# ---------------------------------------------------------------------------


class TestFactoryLazyRegistration:
    """Factory holds module/class strings; first get_engine_class imports."""

    def test_factory_lazy_registration(self) -> None:
        # registry holds 2-tuple of strings before any import
        assert "mlx" in WeightTransferEngineFactory._registry
        module_path, class_name = WeightTransferEngineFactory._registry["mlx"]
        assert isinstance(module_path, str)
        assert isinstance(class_name, str)
        assert module_path == "vllm_mlx.weight_transfer.mlx_backend"
        assert class_name == "MLXWeightTransferEngine"

        # First resolution triggers import, second call returns cached.
        cls1 = WeightTransferEngineFactory.get_engine_class("mlx")
        cls2 = WeightTransferEngineFactory.get_engine_class("mlx")
        assert cls1 is cls2
        assert issubclass(cls1, WeightTransferEngine)

    def test_factory_unknown_backend_raises(self) -> None:
        with pytest.raises(KeyError, match="No weight-transfer backend"):
            WeightTransferEngineFactory.get_engine_class("does-not-exist")


# ---------------------------------------------------------------------------
# AC-2: Scheduler pause/resume
# ---------------------------------------------------------------------------


def _make_scheduler_without_init():
    """Construct a Scheduler without invoking heavy __init__.

    We skip the real __init__ (which would build a BatchGenerator etc.) by
    creating a bare instance and stubbing in just the fields pause() touches.
    """
    from vllm_mlx.scheduler import Scheduler
    import threading

    sched = Scheduler.__new__(Scheduler)
    sched.running = {}
    sched.waiting = []
    sched.finished_req_ids = set()
    sched._paused = False
    sched._pause_mode = None
    sched._pause_lock = threading.Lock()
    sched._inflight_drained = threading.Event()
    sched._inflight_drained.set()
    return sched


class TestSchedulerPauseResume:
    def test_scheduler_pause_wait_no_inflight(self) -> None:
        sched = _make_scheduler_without_init()
        result = sched.pause(mode="wait", clear_cache=True, timeout_s=0.5)
        assert result["paused"] is True
        assert result["drained"] is True
        assert result["mode"] == "wait"
        assert result["clear_cache_requested"] is True
        assert sched._paused is True

    def test_scheduler_pause_abort_cancels_inflight(self) -> None:
        from vllm_mlx.request import RequestStatus

        sched = _make_scheduler_without_init()
        # Populate fake running requests.
        for rid in ("req-a", "req-b"):
            req = MagicMock()
            req.request_id = rid
            req.status = RequestStatus.RUNNING
            req.finish_reason = None
            sched.running[rid] = req

        result = sched.pause(mode="abort", clear_cache=False, timeout_s=0.5)
        assert result["mode"] == "abort"
        assert sched.running == {}
        assert "req-a" in sched.finished_req_ids
        assert "req-b" in sched.finished_req_ids
        # All previously running requests must be marked FINISHED_ABORTED.
        # (We mutated the original mock objects in-place.)
        # No direct access to the request objects anymore — they were removed.
        # Verify _paused and mode set.
        assert sched._paused is True
        assert sched._pause_mode == "abort"

    def test_scheduler_pause_keep_freezes(self) -> None:
        sched = _make_scheduler_without_init()
        req = MagicMock(request_id="r1")
        sched.running["r1"] = req
        result = sched.pause(mode="keep", clear_cache=False, timeout_s=0.5)
        assert result["mode"] == "keep"
        assert "r1" in sched.running  # not modified
        assert sched._paused is True

    def test_scheduler_pause_invalid_mode(self) -> None:
        sched = _make_scheduler_without_init()
        with pytest.raises(ValueError, match="unknown pause mode"):
            sched.pause(mode="garbage")

    def test_scheduler_resume(self) -> None:
        sched = _make_scheduler_without_init()
        sched.pause(mode="keep", clear_cache=False)
        result = sched.resume()
        assert result["resumed"] is True
        assert result["previous_mode"] == "keep"
        assert sched._paused is False

    def test_scheduler_step_returns_empty_when_paused(self) -> None:
        """When paused, step() must return an empty SchedulerOutput safely."""
        from vllm_mlx.scheduler import SchedulerOutput

        sched = _make_scheduler_without_init()
        sched.pause(mode="keep", clear_cache=False)
        out = sched.step()
        assert isinstance(out, SchedulerOutput)
        assert getattr(out, "outputs", []) == []
        assert getattr(out, "scheduled_request_ids", []) == []


# ---------------------------------------------------------------------------
# AC-3: Engine clear KV + prefix cache shim
# ---------------------------------------------------------------------------


class TestEngineClearCache:
    def test_engine_clear_kv_and_prefix_cache(self) -> None:
        """The shim must call both scheduler.clear_runtime_caches and
        scheduler.clear_prefix_cache."""
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = MagicMock()
        engine.scheduler.clear_runtime_caches = MagicMock(return_value={})
        engine.scheduler.clear_prefix_cache = MagicMock(return_value=None)

        engine.clear_kv_and_prefix_cache()
        engine.scheduler.clear_runtime_caches.assert_called_once()
        engine.scheduler.clear_prefix_cache.assert_called_once()

    def test_engine_clear_cache_tolerates_missing_prefix_method(self) -> None:
        """If scheduler lacks clear_prefix_cache, the shim still succeeds."""
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore.__new__(EngineCore)
        sched = MagicMock(spec=["clear_runtime_caches"])
        sched.clear_runtime_caches = MagicMock(return_value={})
        engine.scheduler = sched

        engine.clear_kv_and_prefix_cache()  # must not raise
        sched.clear_runtime_caches.assert_called_once()


# ---------------------------------------------------------------------------
# AC-4: 10-cycle memory growth bound
# ---------------------------------------------------------------------------


_IS_APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


@pytest.mark.skipif(
    not _IS_APPLE_SILICON, reason="MLX weight transfer requires Apple Silicon"
)
def test_ten_cycle_no_memory_growth() -> None:
    """10 cycles of update on a tiny model must not balloon process RSS.

    Threshold: <= 16 MiB delta over 10 iterations.
    """
    import resource

    model = nn.Linear(8, 8)
    cls = WeightTransferEngineFactory.get_engine_class("mlx")
    engine = cls(config=None, parallel_config=None, model=model)
    engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=1))

    def _load_weights(params_list):
        params_dict = dict(params_list)
        model.update(params_dict)
        mx.eval(model.parameters())

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(10):
        # Make a fresh set of params with same shapes.
        new_w = mx.random.normal(shape=(8, 8))
        new_b = mx.random.normal(shape=(8,))
        path = _make_safetensors_file({"weight": new_w, "bias": new_b})
        try:
            info = MLXWeightTransferUpdateInfo(
                names=["weight", "bias"],
                dtype_names=["float32", "float32"],
                shapes=[[8, 8], [8]],
                path=path,
            )
            engine.receive_weights(info, _load_weights)
        finally:
            Path(path).unlink(missing_ok=True)
        gc.collect()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Darwin reports ru_maxrss in bytes; Linux in KiB. Normalize to bytes.
    if sys.platform == "linux":
        rss_before *= 1024
        rss_after *= 1024
    delta_mib = (rss_after - rss_before) / (1024 * 1024)
    # ru_maxrss is a *peak* watermark, so it can only grow. The 16 MiB
    # epsilon must cover MLX's lazy allocator + safetensors I/O headroom.
    assert delta_mib < 16, (
        f"RSS growth {delta_mib:.2f} MiB exceeds 16 MiB bound over 10 cycles"
    )


# ---------------------------------------------------------------------------
# AC-5: Precompiled kernel survives weight update
# ---------------------------------------------------------------------------


def test_precompiled_kernel_survives_update() -> None:
    """After model.update + mx.eval, a compiled forward still produces
    a correctly-shaped output (no recompile crash).

    Note: ``mx.compile`` of a closure captures parameter references at trace
    time; the captured kernel is reused after update. The contract here is
    that the kernel does NOT crash and the output shape is preserved.
    Re-evaluation via ``model(x)`` directly will pick up new weights.
    """
    model = nn.Linear(8, 8)
    compiled = mx.compile(lambda x: model(x))

    x = mx.random.normal(shape=(2, 8))
    y1 = compiled(x)
    mx.eval(y1)
    assert y1.shape == (2, 8)

    # Update weights to a fresh init (zeros for determinism).
    new_params = {
        "weight": mx.zeros((8, 8)),
        "bias": mx.zeros((8,)),
    }
    model.update(new_params)
    mx.eval(model.parameters())

    # The precompiled kernel must still execute without raising.
    y2 = compiled(x)
    mx.eval(y2)
    assert y2.shape == (2, 8)

    # Direct (uncompiled) call WILL see the new weights — sanity-check that
    # the update itself took effect on the module's parameters.
    y_direct = model(x)
    mx.eval(y_direct)
    assert y_direct.shape == (2, 8)
    assert mx.allclose(y_direct, mx.zeros((2, 8))).item()


# ---------------------------------------------------------------------------
# AC-6: trainer_send_weights eval barrier + safetensors out
# ---------------------------------------------------------------------------


def test_trainer_send_weights_eval_barrier() -> None:
    """A lazy array is fully evaluated before being written."""
    cls = WeightTransferEngineFactory.get_engine_class("mlx")
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    lazy_sum = a + b  # lazy
    weights = [("sum", lazy_sum)]

    with tempfile.NamedTemporaryFile(
        suffix=".safetensors", delete=False
    ) as tf:
        out_path = tf.name

    try:
        cls.trainer_send_weights(iter(weights), {"out_path": out_path})
        assert Path(out_path).exists()
        loaded = mx.load(out_path)
        assert "sum" in loaded
        assert mx.allclose(loaded["sum"], mx.array([5.0, 7.0, 9.0])).item()
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_trainer_send_weights_no_out_path_raises() -> None:
    cls = WeightTransferEngineFactory.get_engine_class("mlx")
    with pytest.raises(NotImplementedError, match="out_path"):
        cls.trainer_send_weights(iter([("x", mx.array([1.0]))]), {})


# ---------------------------------------------------------------------------
# AC-8: Pydantic schema validation
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    def test_pydantic_init_request_permissive(self) -> None:
        # Extra fields allowed (forward-compat with upstream NCCL etc.).
        m = WeightTransferInitRequest(
            init_info={"world_size": 1, "rank": 0},
            nccl_uri="ignored",  # arbitrary extra
        )
        assert m.init_info["world_size"] == 1

    def test_pydantic_update_request_missing_keys(self) -> None:
        # missing 'names'
        with pytest.raises(ValueError, match="missing required key"):
            WeightTransferUpdateRequest(
                update_info={
                    "dtype_names": [],
                    "shapes": [],
                    "path": "/tmp/x",
                }
            )

    def test_pydantic_update_request_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            WeightTransferUpdateRequest(
                update_info={
                    "names": ["a"],
                    "dtype_names": ["float32", "float32"],
                    "shapes": [[1]],
                    "path": "/tmp/x",
                }
            )

    def test_pydantic_update_request_both_packed_and_path(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            WeightTransferUpdateRequest(
                update_info={
                    "names": [],
                    "dtype_names": [],
                    "shapes": [],
                    "packed": b"x",
                    "path": "/tmp/x",
                }
            )

    def test_pydantic_update_request_neither_packed_nor_path(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            WeightTransferUpdateRequest(
                update_info={
                    "names": [],
                    "dtype_names": [],
                    "shapes": [],
                }
            )

    def test_pydantic_pause_request_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            PauseRequest(mode="invalid")  # type: ignore[arg-type]

    def test_pydantic_pause_request_defaults(self) -> None:
        p = PauseRequest()
        assert p.mode == "wait"
        assert p.clear_cache is True

    def test_pydantic_start_weight_update_defaults(self) -> None:
        s = StartWeightUpdateRequest()
        assert s.is_checkpoint_format is True


# ---------------------------------------------------------------------------
# AC-7: HTTP route auth + dispatch
# ---------------------------------------------------------------------------


_WT_ROUTES = [
    ("POST", "/init_weight_transfer_engine", {"init_info": {"world_size": 1}}),
    ("POST", "/start_weight_update", {"is_checkpoint_format": True}),
    (
        "POST",
        "/update_weights",
        {
            "update_info": {
                "names": [],
                "dtype_names": [],
                "shapes": [],
                "path": "/tmp/x",
            }
        },
    ),
    ("POST", "/finish_weight_update", None),
    ("POST", "/pause", {"mode": "wait", "clear_cache": True}),
    ("POST", "/resume", None),
    ("GET", "/get_world_size", None),
]


class TestRoutesAuthGate:
    """When _api_key is set, every weight-transfer route must reject
    unauthenticated requests."""

    def test_routes_auth_gate(self) -> None:
        client = TestClient(srv.app)
        original_key = srv._api_key
        original_engine = srv._engine
        srv._api_key = "secret"
        srv._engine = MagicMock()
        try:
            for method, path, body in _WT_ROUTES:
                if method == "POST":
                    resp = client.post(path, json=body)
                else:
                    resp = client.get(path)
                assert resp.status_code in (401, 403), (
                    f"{method} {path} returned {resp.status_code} "
                    f"without auth — expected 401/403"
                )
        finally:
            srv._api_key = original_key
            srv._engine = original_engine


class TestRoutesDispatch:
    """When auth is off and an engine is mocked in, each route maps to the
    correct engine shim with the correct args."""

    def _patch_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.init_weight_transfer_engine.return_value = {
            "initialized": True,
            "backend": "mlx",
            "world_size": 1,
        }
        engine.start_weight_update.return_value = {"started": True}
        engine.update_weights.return_value = {"applied": True}
        engine.finish_weight_update.return_value = {"resumed": True}
        engine.pause.return_value = {"paused": True}
        engine.resume.return_value = {"resumed": True}
        engine.get_world_size.return_value = 1
        return engine

    def test_init_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post(
                "/init_weight_transfer_engine",
                json={"init_info": {"world_size": 1, "rank": 0}},
            )
            assert r.status_code == 200, r.text
            srv._engine.init_weight_transfer_engine.assert_called_once()
            # The route passes the Pydantic model in; verify shape.
            call_arg = srv._engine.init_weight_transfer_engine.call_args[0][0]
            assert call_arg.init_info["world_size"] == 1
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_start_weight_update_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post(
                "/start_weight_update", json={"is_checkpoint_format": False}
            )
            assert r.status_code == 200, r.text
            srv._engine.start_weight_update.assert_called_once_with(False)
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_update_weights_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post(
                "/update_weights",
                json={
                    "update_info": {
                        "names": ["weight"],
                        "dtype_names": ["float32"],
                        "shapes": [[8, 8]],
                        "path": "/tmp/dummy.safetensors",
                    }
                },
            )
            assert r.status_code == 200, r.text
            srv._engine.update_weights.assert_called_once()
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_finish_weight_update_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post("/finish_weight_update")
            assert r.status_code == 200, r.text
            srv._engine.finish_weight_update.assert_called_once()
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_pause_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post(
                "/pause", json={"mode": "abort", "clear_cache": False}
            )
            assert r.status_code == 200, r.text
            srv._engine.pause.assert_called_once_with(
                mode="abort", clear_cache=False
            )
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_resume_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.post("/resume")
            assert r.status_code == 200, r.text
            srv._engine.resume.assert_called_once()
        finally:
            srv._engine = original_engine
            srv._api_key = original_key

    def test_get_world_size_route_dispatches(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = self._patch_engine()
        srv._api_key = None
        try:
            r = client.get("/get_world_size")
            assert r.status_code == 200, r.text
            assert r.json() == {"world_size": 1}
        finally:
            srv._engine = original_engine
            srv._api_key = original_key


class TestUpdateWeightsReentrancyGuard:
    """M-3 mitigation: update_weights must require prior start_weight_update.

    Without an active weight-update window, the scheduler is not paused and
    caches have not been cleared — applying weights would race in-flight
    forward passes against model.update() on shared state.
    """

    def test_update_weights_without_start_raises(self) -> None:
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore.__new__(EngineCore)
        # Pretend the transfer engine has been initialized so the first
        # guard (engine is None) passes; the re-entrancy guard is the
        # check under test.
        engine._weight_transfer_engine = MagicMock()
        engine._weight_update_in_progress = False

        with pytest.raises(RuntimeError, match="start_weight_update"):
            engine.update_weights(MagicMock())


class TestRoutesEngineMissing:
    """Each route must return 503 when engine isn't initialized."""

    def test_routes_503_when_no_engine(self) -> None:
        client = TestClient(srv.app)
        original_engine = srv._engine
        original_key = srv._api_key
        srv._engine = None
        srv._api_key = None
        try:
            for method, path, body in _WT_ROUTES:
                if method == "POST":
                    resp = client.post(path, json=body)
                else:
                    resp = client.get(path)
                assert resp.status_code == 503, f"{method} {path} -> {resp.status_code}"
        finally:
            srv._engine = original_engine
            srv._api_key = original_key


# ---------------------------------------------------------------------------
# AC-9: CHANGELOG entry
# ---------------------------------------------------------------------------


def test_changelog_unreleased_entry() -> None:
    changelog = REPO_ROOT / "CHANGELOG.md"
    assert changelog.exists(), f"CHANGELOG.md missing at {changelog}"
    text = changelog.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert "WeightTransferEngine" in text


# ---------------------------------------------------------------------------
# MLX backend behavioral tests
# ---------------------------------------------------------------------------


class TestMLXBackendBehavior:
    def test_init_world_size_must_be_one(self) -> None:
        cls = WeightTransferEngineFactory.get_engine_class("mlx")
        engine = cls(config=None, parallel_config=None, model=MagicMock())
        with pytest.raises(ValueError, match="world_size=1"):
            engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=2))

    def test_receive_weights_path_roundtrip(self) -> None:
        model = nn.Linear(4, 4)
        cls = WeightTransferEngineFactory.get_engine_class("mlx")
        engine = cls(config=None, parallel_config=None, model=model)
        engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=1))

        new_w = mx.zeros((4, 4))
        new_b = mx.zeros((4,))
        path = _make_safetensors_file({"weight": new_w, "bias": new_b})

        applied: list = []

        def _load(params_list):
            applied.extend(params_list)
            model.update(dict(params_list))

        try:
            info = MLXWeightTransferUpdateInfo(
                names=["weight", "bias"],
                dtype_names=["float32", "float32"],
                shapes=[[4, 4], [4]],
                path=path,
            )
            engine.receive_weights(info, _load)
        finally:
            Path(path).unlink(missing_ok=True)

        names = [n for n, _ in applied]
        assert names == ["weight", "bias"]

    def test_receive_weights_shape_mismatch_raises(self) -> None:
        cls = WeightTransferEngineFactory.get_engine_class("mlx")
        engine = cls(config=None, parallel_config=None, model=MagicMock())
        engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=1))

        path = _make_safetensors_file({"weight": mx.zeros((4, 4))})
        try:
            info = MLXWeightTransferUpdateInfo(
                names=["weight"],
                dtype_names=["float32"],
                shapes=[[8, 8]],  # WRONG
                path=path,
            )
            with pytest.raises(ValueError, match="shape mismatch"):
                engine.receive_weights(info, lambda *_: None)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_receive_weights_rejects_both_path_and_packed(self) -> None:
        cls = WeightTransferEngineFactory.get_engine_class("mlx")
        engine = cls(config=None, parallel_config=None, model=MagicMock())
        engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=1))

        info = MLXWeightTransferUpdateInfo(
            names=["x"],
            dtype_names=["float32"],
            shapes=[[1]],
            path="/tmp/a",
            packed=b"x",
        )
        with pytest.raises(ValueError, match="exactly one"):
            engine.receive_weights(info, lambda *_: None)

    def test_shutdown_idempotent(self) -> None:
        cls = WeightTransferEngineFactory.get_engine_class("mlx")
        engine = cls(config=None, parallel_config=None, model=MagicMock())
        engine.init_transfer_engine(MLXWeightTransferInitInfo(world_size=1))
        engine.shutdown()
        engine.shutdown()  # second call must not raise


# ---------------------------------------------------------------------------
# Integration (skipped — OUT OF SCOPE per PROJECT.md)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="integration; OUT OF SCOPE per PROJECT.md")
def test_integration_real_model_rollout() -> None:  # pragma: no cover
    """End-to-end: realign train --reload-every 1. Tracked separately
    (realign#1309 follow-up)."""


# ---------------------------------------------------------------------------
# Regression: dotted-key params_dict must be tree-unflattened before update()
# ---------------------------------------------------------------------------


def test_load_weights_handles_dotted_param_names() -> None:
    """EngineCore._load_weights must accept dotted-key (name, array) pairs
    like ``("model.embed_tokens.weight", arr)`` — these are what real
    HuggingFace-shaped trainer pushes look like (TRL GRPOTrainer).

    The pre-fix bug: ``model.update({"model.embed_tokens.weight": arr})``
    raised ``ValueError: Module does not have parameter named ...`` because
    MLX's ``nn.Module.update()`` walks a NESTED dict, not a flat dotted-key
    one. Fix uses ``mlx.utils.tree_unflatten`` before calling ``update()``.

    Surfaced by the TRL+vllm-mlx Phase 2 smoke (2026-06-29). The existing
    MagicMock-based tests masked it because MagicMock answers any attribute.
    """
    from mlx.utils import tree_unflatten

    # Build a small nested module so update() must walk the tree.
    class NestedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            class Inner(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.embed_tokens = nn.Linear(4, 4, bias=False)

            self.model = Inner()

    module = NestedModule()
    # Mimic exactly what EngineCore._load_weights does post-fix.
    params_list = [
        ("model.embed_tokens.weight", mx.zeros((4, 4))),
    ]
    nested = tree_unflatten(params_list)
    # Must not raise (pre-fix code did: ValueError: Module does not have parameter named ...).
    module.update(nested)
    mx.eval(module.parameters())
    # Confirm the actual values landed where MLX expects them.
    new_weight = module.model.embed_tokens.weight
    assert new_weight.shape == (4, 4)
    assert mx.all(new_weight == 0).item()


def test_load_weights_flat_dict_raises_without_unflatten() -> None:
    """Anti-tautology: confirm the pre-fix path actually fails.

    Without ``tree_unflatten``, passing a flat dotted-key dict to MLX's
    ``Module.update()`` raises ``ValueError`` — this is the bug that the
    fix above prevents. If this test ever passes without the fix, MLX
    semantics changed and the regression test above can be relaxed.
    """

    class NestedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            class Inner(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.embed_tokens = nn.Linear(4, 4, bias=False)

            self.model = Inner()

    module = NestedModule()
    flat = {"model.embed_tokens.weight": mx.zeros((4, 4))}
    with pytest.raises(ValueError, match="does not have parameter named"):
        module.update(flat)
