#!/usr/bin/env python3
"""Tests for GEANT4-ready particle momentum metadata."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.science.run_powheg_pythia_geant4_resumable import attach_global_exit_position


def base_ready_row() -> dict[str, float]:
    return {
        "interaction_x_rg": 10.0,
        "interaction_y_rg": 0.0,
        "interaction_z_rg": 0.0,
        "interaction_r_rg": 10.0,
        "interaction_theta_rad": math.pi / 2.0,
        "interaction_phi_rad": 0.0,
        "geant4_exit_local_x_rg": 0.0,
        "geant4_exit_local_y_rg": 0.0,
        "geant4_exit_local_z_rg": 1.0e-6,
        "px": 0.0,
        "py": 0.0,
        "pz": 3.0,
    }


def test_zamo_transform_writes_explicit_momentum_contract() -> None:
    row = base_ready_row()
    attach_global_exit_position(row, spin=0.0, transform="zamo_tetrad")
    assert row["global_position_transform"] == "ZAMO_TETRAD_LOCAL_BOX"
    assert row["momentum_input_mode"] == "zamo_tetrad"
    assert row["ready_particle_momentum_frame"] == "ZAMO_TETRAD_LOCAL_BOX"
    assert row["momentum_input_mode_policy"] == "explicit_zamo_tetrad_components"
    assert row["global_momentum_status"] == "GLOBAL_MOMENTUM_ZAMO_SPATIAL_TRIAD"
    for field in ["n_zamo_r", "n_zamo_theta", "n_zamo_phi"]:
        assert field in row
        assert math.isfinite(float(row[field]))
    assert abs(float(row["n_zamo_r"]) - 1.0) < 1.0e-12
    assert abs(float(row["n_zamo_theta"])) < 1.0e-12
    assert abs(float(row["n_zamo_phi"])) < 1.0e-12


def test_debug_transform_marks_momentum_unknown() -> None:
    row = base_ready_row()
    attach_global_exit_position(row, spin=0.0, transform="local_cartesian")
    assert row["momentum_input_mode"] == "unknown"
    assert row["ready_particle_momentum_frame"] == "unknown"
    assert row["momentum_input_mode_policy"] == "explicit_mode_required_for_photon_observer_camera"


def test_missing_momentum_marks_unknown() -> None:
    row = base_ready_row()
    row.pop("px")
    attach_global_exit_position(row, spin=0.0, transform="zamo_tetrad")
    assert row["global_momentum_status"] == "GLOBAL_MOMENTUM_NOT_AVAILABLE"
    assert row["momentum_input_mode"] == "unknown"


if __name__ == "__main__":
    test_zamo_transform_writes_explicit_momentum_contract()
    test_debug_transform_marks_momentum_unknown()
    test_missing_momentum_marks_unknown()
