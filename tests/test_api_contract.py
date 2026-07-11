"""API contract tests: registry/OpenAPI drift + light Hypothesis fuzzing.

Complements ``tests/test_api.py`` (happy-path REST surface) and
``tests/test_access.py`` (access tiers) with two kinds of coverage that
neither of those files exercises:

1. Drift — the self-describing ``/commands`` list and the OpenAPI
   ``/schema`` document must stay in sync with the commands actually
   registered in ``build_default_registry()``, and every self-described
   read command must actually be callable (not just describable).
2. Fuzzing — ``/commands/{name}`` and ``/settings/{path}`` must never
   surface a raw 500 for malformed/garbage client input; a clean 4xx
   (or 200, if the garbage happens to be valid) is the only acceptable
   outcome for *any* input, however weird.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("hypothesis")

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from api.app import create_app
from api.commands import NoParams, build_default_registry
from api.settings import Settings, setting_paths

_HTTP_OK = 200
_HTTP_SERVER_ERROR_FLOOR = 500
_MAX_EXAMPLES = 50


def _client() -> TestClient:
    """Same app/client construction as test_api.py / test_access.py."""
    return TestClient(create_app(Settings()))


def _fuzz_client() -> TestClient:
    """A client that turns unhandled server exceptions into real 500
    responses instead of re-raising them into the test process.

    FastAPI's TestClient defaults to ``raise_server_exceptions=True``
    (great for the happy-path tests), but a fuzz test wants to assert on
    the *status code* an uncaught exception produces in production, not
    have the exception blow up the test itself.
    """
    return TestClient(create_app(Settings()), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Contract / drift
# ---------------------------------------------------------------------------


def test_every_command_is_discoverable_and_self_consistent() -> None:
    """`/commands` must list exactly the registry's commands, each with a
    real description, a well-formed params JSON Schema, and a bool tier."""
    registry = build_default_registry()
    expected_names = {entry["name"] for entry in registry.describe()}

    live = _client().get("/commands").json()
    live_by_name = {entry["name"]: entry for entry in live}

    assert set(live_by_name) == expected_names

    for entry in live:
        assert isinstance(entry["description"], str)
        assert entry["description"].strip()
        assert isinstance(entry["mutates"], bool)
        params_schema = entry["params"]
        assert isinstance(params_schema, dict)
        assert params_schema.get("type") == "object"
        assert "properties" in params_schema


def test_openapi_schema_documents_command_execution() -> None:
    """The OpenAPI doc served at /schema must include the command-execution
    route, so external tooling generated from it can reach every command."""
    schema = _client().get("/schema").json()
    assert any("/commands/" in path for path in schema["paths"])


def test_self_described_read_commands_are_actually_executable() -> None:
    """Every non-mutating, no-argument command the registry describes must
    actually run and return JSON — catching a command that is correctly
    self-described but whose handler is broken or missing."""
    registry = build_default_registry()
    client = _client()
    checked = 0
    for entry in registry.describe():
        command = registry.get(entry["name"])
        if command.mutates or command.params is not NoParams:
            continue
        response = client.post(f"/commands/{entry['name']}")
        assert response.status_code == _HTTP_OK, entry["name"]
        body = response.json()
        assert body["command"] == entry["name"]
        assert "result" in body
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Fuzzing: malformed/garbage input must never produce a raw 500
# ---------------------------------------------------------------------------

_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)

# Arbitrary JSON-shaped values: scalars, or lists/dicts of them, nested a
# few levels deep. NaN/Infinity are excluded because they aren't valid JSON
# tokens; a client encoding them would fail before the request is even
# sent, which would test httpx, not the server.
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=10,
)

# Path text safe to interpolate into a URL segment: printable, non-control
# characters only. Control characters (e.g. "\t", "\x00") make httpx raise
# InvalidURL client-side, which would test the HTTP client, not the app.
_SAFE_PATH_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cc", "Cs")), max_size=40)

_KNOWN_COMMAND_NAMES = (
    "ping",
    "get_settings",
    "list_setting_paths",
    "set_setting",
    "list_grid_backends",
    "probe_hardware",
)
_VALID_SETTING_PATHS = setting_paths(Settings())

_FUZZ_CLIENT = _fuzz_client()


@given(path=st.text(max_size=30), value=_JSON_VALUES)
@hyp_settings(
    max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_set_setting_command_never_500s_on_garbage(path: str, value: object) -> None:
    """set_setting is the one mutating, richly-typed command; throw garbage
    dotted paths and arbitrary values at it as an authorized (admin)
    principal and require a clean status, never a 500."""
    response = _FUZZ_CLIENT.post("/commands/set_setting", json={"path": path, "value": value})
    assert response.status_code < _HTTP_SERVER_ERROR_FLOOR


@given(name=st.sampled_from(_KNOWN_COMMAND_NAMES), body=_JSON_VALUES)
@hyp_settings(
    max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_command_endpoint_never_500s_on_arbitrary_json_body(name: str, body: object) -> None:
    """Any known command, given an arbitrary JSON body (not necessarily
    shaped like its params model — could be a list, a string, a number,
    or a malformed dict), must fail cleanly rather than 500."""
    response = _FUZZ_CLIENT.post(f"/commands/{name}", json=body)
    assert response.status_code < _HTTP_SERVER_ERROR_FLOOR


@given(path=st.sampled_from(_VALID_SETTING_PATHS), value=_JSON_VALUES)
@hyp_settings(
    max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_settings_put_never_500s_on_garbage_value_for_valid_path(path: str, value: object) -> None:
    """PUT a real setting path with an arbitrary (likely schema-invalid)
    value; the settings schema should reject it with 4xx, never 500."""
    response = _FUZZ_CLIENT.put(f"/settings/{path}", json={"value": value})
    assert response.status_code < _HTTP_SERVER_ERROR_FLOOR


@given(path=_SAFE_PATH_TEXT, value=_JSON_VALUES)
@hyp_settings(
    max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_settings_put_never_500s_on_invalid_path(path: str, value: object) -> None:
    """A clearly-bogus dotted path (arbitrary text) must 404, never 500,
    regardless of how strange the accompanying value is."""
    response = _FUZZ_CLIENT.put(f"/settings/{path}", json={"value": value})
    assert response.status_code < _HTTP_SERVER_ERROR_FLOOR
