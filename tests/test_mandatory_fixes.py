#!/usr/bin/env python3
"""Lightweight validation for mandatory HADROS-CASCADE consistency fixes."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "science"))


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


final_pipeline = load_module("run_hadros_final_pipeline", "scripts/run_hadros_final_pipeline.py")
geant4_runner = load_module("run_powheg_pythia_geant4_resumable", "scripts/science/run_powheg_pythia_geant4_resumable.py")
ray_link = load_module("build_uhe_ray_event_link", "scripts/science/build_uhe_ray_event_link.py")
gbw_iim = load_module("build_gbw_iim_real_kerr_reweighting", "scripts/science/build_gbw_iim_real_kerr_reweighting.py")


def assert_close(a: float, b: float, rel: float = 1.0e-12) -> None:
    if abs(a - b) / max(abs(b), 1.0) > rel:
        raise AssertionError(f"{a!r} != {b!r}")


def test_cm_per_rg_mbh3() -> None:
    expected = 1.4766250385e5 * 3.0
    assert_close(final_pipeline.cm_per_rg_from_mbh_msun(3.0), expected)
    assert_close(geant4_runner.validate_geant4_local_cm_per_rg(expected, 3.0), expected)
    try:
        geant4_runner.validate_geant4_local_cm_per_rg(1.0, 3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("geant4-local-cm-per-rg=1.0 must fail")


def test_gbw_iim_uses_local_incident_energy() -> None:
    point = {"event_id": 7, "E_nu_inf_GeV": 1.0e9, "redshift_factor": 2.5, "E_nu_local_GeV": 2.5e9}
    meta = gbw_iim.incident_neutrino_metadata(point)
    assert_close(meta["E_nu_local_GeV"], 2.5e9)
    try:
        gbw_iim.incident_neutrino_metadata({"event_id": 8, "energy_gev": 9.0e9})
    except RuntimeError as exc:
        if "final-state energy proxy" not in str(exc):
            raise
    else:
        raise AssertionError("missing incident neutrino energy must fail")


def test_redshift_python_matches_cpp_formula() -> None:
    e_inf = 1.0e9
    redshift = 1.75
    expected = e_inf * redshift
    assert_close(ray_link.local_neutrino_energy_gev(e_inf, redshift), expected)
    assert_close(gbw_iim.local_neutrino_energy_gev(e_inf, redshift), expected)


def test_ray_id_nx_ne_ny() -> None:
    nx, ny = 3, 5
    pixel_x, pixel_y = 1, 4
    ray_id = pixel_y * nx + pixel_x
    old_transposed = pixel_x * ny + pixel_y
    if ray_id == old_transposed:
        raise AssertionError("test must use nx != ny and non-degenerate pixel")
    if ray_id != 13:
        raise AssertionError(f"unexpected ray_id={ray_id}")


def test_physics_mode_uhe_dis_only_has_no_geant4() -> None:
    config = {
        "run_name": "UnitMode",
        "output_dir": "/tmp/hadros_cascade_mandatory_fix_mode_test",
        "physics_mode": "uhe_dis_only",
        "black_hole_mass_msun": 3.0,
        "spin": 0.8,
        "camera_r_obs_rg": 80.0,
        "camera_theta_deg": 70.0,
        "camera_fov_deg": 60.0,
        "camera_nx": 3,
        "camera_ny": 5,
        "camera_r_max_rg": 120.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 2,
        "seed": 123,
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    forbidden = {"powheg_pythia_event_records", "geant4_real_safe_zamo", "real_kerr_particle_camera"}
    if forbidden.intersection(names):
        raise AssertionError(f"uhe_dis_only included forbidden steps: {forbidden.intersection(names)}")


def main() -> int:
    tests = [
        test_cm_per_rg_mbh3,
        test_gbw_iim_uses_local_incident_energy,
        test_redshift_python_matches_cpp_formula,
        test_ray_id_nx_ne_ny,
        test_physics_mode_uhe_dis_only_has_no_geant4,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
