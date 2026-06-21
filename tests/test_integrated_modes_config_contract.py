#!/usr/bin/env python3
"""Integrated lightweight checks for physics/association/naming config flow."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


final_pipeline = load_module("run_hadros_final_pipeline_integrated", "scripts/run_hadros_final_pipeline.py")


def base_config(output_dir: str, *, physics_mode: str = "uhe_particles_camera") -> dict[str, object]:
    return {
        "run_name": "IntegratedModeTest",
        "output_dir": output_dir,
        "physics_mode": physics_mode,
        "black_hole_mass_msun": 3.0,
        "spin": 0.8,
        "camera_nx": 3,
        "camera_ny": 5,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 80.0,
        "camera_r_max_rg": 120.0,
        "camera_step": 0.05,
        "association_mode": "spatial_plus_direction",
        "camera_naming_mode": "both",
        "spatial_tolerance_rg": 1.0,
        "angular_tolerance_deg": 1.0,
        "source_model": "funnel_wall",
        "neutrino_energy_gev": 1.0e9,
        "dis_model": "both",
        "n_events": 2,
        "seed": 123,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
    }


def step_names(config: dict[str, object]) -> list[str]:
    return [step.name for step in final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")]


def test_uhe_dis_only_has_no_cascade_or_camera() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
        names = step_names(base_config(tmp, physics_mode="uhe_dis_only"))
    forbidden = {"powheg_pythia_event_records", "geant4_real_safe_zamo", "real_kerr_particle_camera"}
    if forbidden.intersection(names):
        raise AssertionError(f"uhe_dis_only included forbidden steps: {forbidden.intersection(names)}")


def test_uhe_cascade_has_no_camera() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
        names = step_names(base_config(tmp, physics_mode="uhe_cascade"))
    if "powheg_pythia_event_records" not in names or "geant4_real_safe_zamo" not in names:
        raise AssertionError(f"uhe_cascade missing cascade steps: {names}")
    if "real_kerr_particle_camera" in names:
        raise AssertionError("uhe_cascade should not require particle-ray association camera")


def test_uhe_particles_camera_passes_modes() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
        config = base_config(tmp, physics_mode="uhe_particles_camera")
        config["association_mode"] = "spatial_only"
        config["camera_naming_mode"] = "semantic"
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    camera_step = next(step for step in steps if step.name == "real_kerr_particle_camera")
    command = camera_step.command
    if "--association-mode" not in command or command[command.index("--association-mode") + 1] != "spatial_only":
        raise AssertionError(f"association_mode not passed: {command}")
    if "--camera-naming-mode" not in command or command[command.index("--camera-naming-mode") + 1] != "semantic":
        raise AssertionError(f"camera_naming_mode not passed: {command}")
    required = [path.name for path in camera_step.required_outputs]
    if required != ["particle_ray_association_camera.csv"]:
        raise AssertionError(f"semantic mode expected semantic output, got {required}")


def test_full_transport_fails_explicitly() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
        config = base_config(tmp, physics_mode="uhe_particles_camera")
        config["association_mode"] = "full_transport"
        try:
            final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        except NotImplementedError as exc:
            if "full_transport is not implemented yet" not in str(exc):
                raise
        else:
            raise AssertionError("full_transport must fail explicitly")


def test_camera_naming_mode_required_outputs() -> None:
    expected = {
        "semantic": "particle_ray_association_camera.csv",
        "both": "particle_ray_association_camera.csv",
        "legacy": "observed_particles_by_pixel.csv",
    }
    for mode, filename in expected.items():
        with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
            config = base_config(tmp, physics_mode="uhe_particles_camera")
            config["camera_naming_mode"] = mode
            steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        camera_step = next(step for step in steps if step.name == "real_kerr_particle_camera")
        if [path.name for path in camera_step.required_outputs] != [filename]:
            raise AssertionError(f"{mode} expected {filename}, got {camera_step.required_outputs}")


def test_provenance_contains_camera_limitation() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_integrated_modes_") as tmp:
        config = base_config(tmp, physics_mode="uhe_particles_camera")
        path = final_pipeline.config_for_interaction_scripts(config, Path(tmp) / "cascade")
        payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    expected = {
        "physics_mode_effective": "uhe_particles_camera",
        "association_mode_effective": "spatial_plus_direction",
        "camera_naming_mode": "both",
        "spatial_tolerance_rg": 1.0,
        "angular_tolerance_deg": 1.0,
        "camera_physical_interpretation": "particle-ray association / cascade origin map",
        "camera_is_full_observational_transport": False,
        "full_transport_available": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise AssertionError(f"provenance[{key!r}]={provenance.get(key)!r}, expected {value!r}")
    if "not propagated to the distant observer" not in provenance.get("camera_limitation", ""):
        raise AssertionError("camera_limitation missing explicit transport limitation")


def test_config_web_has_operational_camera_parameters() -> None:
    text = (ROOT / "scripts" / "config_web_final.py").read_text(encoding="utf-8")
    for needle in [
        "physics_mode",
        "association_mode",
        "spatial_tolerance_rg",
        "angular_tolerance_deg",
        "camera_naming_mode",
        'options=["spatial_only", "spatial_plus_direction", "full_transport"]',
        'options=["both", "semantic", "legacy"]',
    ]:
        if needle not in text:
            raise AssertionError(f"config_web_final.py missing {needle}")


def test_wrapper_has_no_camera_operational_defaults() -> None:
    text = (ROOT / "scripts" / "science" / "run_real_kerr_particle_camera.py").read_text(encoding="utf-8")
    forbidden = [
        'parser.add_argument("--camera-nx", type=int, default=',
        'parser.add_argument("--camera-ny", type=int, default=',
        'parser.add_argument("--spatial-tolerance-rg", type=float, default=',
        'parser.add_argument("--angular-tolerance-deg", type=float, default=',
        'default="spatial_plus_direction"',
        'default="both"',
        'parser.add_argument("--aspin", type=float, default=',
    ]
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"hidden wrapper operational default remains: {needle}")


def main() -> int:
    tests = [
        test_uhe_dis_only_has_no_cascade_or_camera,
        test_uhe_cascade_has_no_camera,
        test_uhe_particles_camera_passes_modes,
        test_full_transport_fails_explicitly,
        test_camera_naming_mode_required_outputs,
        test_provenance_contains_camera_limitation,
        test_config_web_has_operational_camera_parameters,
        test_wrapper_has_no_camera_operational_defaults,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
