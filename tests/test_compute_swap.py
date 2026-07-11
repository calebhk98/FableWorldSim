"""Tests for hot-swapping the compute backend (CPU<->GPU) at runtime.

The swap is exposed on the API (``get_compute_backend`` /
``set_compute_backend``) and implemented on :class:`AppState` so it lands
between ticks via the single-writer lock and rejects a backend the host
cannot provide without disturbing the running one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.commands import apply_setting
from api.settings import Settings
from api.state import AppState, EventBus
from api.ws_events import EVENT_MODELS, ComputeBackendChangedEvent
from ports.array_backend import ComputeBackendUnavailableError

_HTTP_OK = 200
_HTTP_CONFLICT = 409
_HTTP_UNPROCESSABLE = 422


def _client() -> TestClient:
    return TestClient(create_app(Settings()))


def _state() -> AppState:
    return AppState(Settings(), EventBus())


def test_get_reports_the_live_backend_and_device() -> None:
    """The read command names the backend, its device, and device count."""
    result = _client().post("/commands/get_compute_backend").json()["result"]
    assert result["device"] in {"cpu", "gpu"}
    assert result["num_devices"] >= 1
    assert isinstance(result["backend"], str)


def test_swap_to_numpy_changes_backend_and_setting() -> None:
    """Swapping updates both the live backend and the stored setting."""
    client = _client()
    result = client.post("/commands/set_compute_backend", json={"backend": "numpy"})
    assert result.status_code == _HTTP_OK
    assert result.json()["result"] == {"backend": "numpy", "device": "cpu", "num_devices": 1}
    assert client.get("/settings").json()["compute"]["backend"] == "numpy"


def test_unavailable_gpu_backend_is_rejected_without_disturbing_the_running_one() -> None:
    """Asking for cupy on a CPU host is a clean 409, and nothing changes."""
    client = _client()
    client.post("/commands/set_compute_backend", json={"backend": "numpy"})

    rejected = client.post("/commands/set_compute_backend", json={"backend": "cupy"})
    assert rejected.status_code == _HTTP_CONFLICT

    # The running backend and the setting are untouched by the failed swap.
    assert client.get("/settings").json()["compute"]["backend"] == "numpy"
    still = client.post("/commands/get_compute_backend").json()["result"]
    assert still["backend"] == "numpy"


def test_unknown_backend_name_is_a_validation_error() -> None:
    """A name outside the schema is rejected as malformed input (422)."""
    rejected = _client().post("/commands/set_compute_backend", json={"backend": "banana"})
    assert rejected.status_code == _HTTP_UNPROCESSABLE


def test_setting_compute_backend_routes_through_the_swap() -> None:
    """The generic set-setting path swaps the live handle too, atomically."""
    state = _state()
    apply_setting(state, "compute.backend", "numpy")
    assert state.compute_backend.name == "numpy"
    assert state.settings.compute.backend == "numpy"


def test_swap_publishes_a_change_event() -> None:
    """Every swap broadcasts a compute_backend_changed event to viewers."""
    state = _state()
    queue = state.bus.subscribe()
    state.set_compute_backend("numpy")
    event = queue.get_nowait()
    assert isinstance(event, ComputeBackendChangedEvent)
    assert event.backend == "numpy"
    assert event.device == "cpu"


def test_failed_swap_preserves_the_running_backend() -> None:
    """A rejected swap leaves the previously selected backend in place."""
    state = _state()
    state.set_compute_backend("numpy")
    with pytest.raises(ComputeBackendUnavailableError):
        state.set_compute_backend("cupy")
    assert state.compute_backend.name == "numpy"
    assert state.settings.compute.backend == "numpy"


def test_swap_is_safe_to_call_while_holding_the_write_lock() -> None:
    """The swap reuses the single-writer lock (reentrant) — no deadlock.

    Mutating commands already run under ``write_lock``; the swap taking it
    again must not deadlock, which is what lets it land between ticks.
    """
    state = _state()
    with state.write_lock:
        state.set_compute_backend("numpy")
    assert state.compute_backend.name == "numpy"


def test_change_event_is_in_the_ws_vocabulary() -> None:
    """The event type is discoverable via the /ws/schema vocabulary."""
    assert EVENT_MODELS["compute_backend_changed"] is ComputeBackendChangedEvent
