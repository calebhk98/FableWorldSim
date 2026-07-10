"""Tests for the adaptive-LOD refinement policy hook."""

from __future__ import annotations

from core.grid.refinement import GradientRefinementPolicy, refinement_targets
from tests.test_grid_port import FakeGrid


def test_gradient_policy_flags_sharp_edges_only() -> None:
    grid = FakeGrid(0, 1_000.0)
    fields = {"elevation_m": {"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 2_000.0}}
    policy = GradientRefinementPolicy(field_name="elevation_m", min_contrast=500.0)
    targets = set(refinement_targets(grid, policy, fields))
    assert "c3" in targets
    assert targets == {"c0", "c1", "c2", "c3"}  # tetrahedron: all touch c3

    flat = {"elevation_m": dict.fromkeys(fields["elevation_m"], 5.0)}
    assert set(refinement_targets(grid, policy, flat)) == set()
