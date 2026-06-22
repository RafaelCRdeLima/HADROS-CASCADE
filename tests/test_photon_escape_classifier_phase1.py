#!/usr/bin/env python3
"""Lightweight tests for Photon Escape Classifier Phase 1."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
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


final_pipeline = load_module("run_hadros_final_pipeline_photon_escape", "scripts/run_hadros_final_pipeline.py")
phase2_hits = load_module("build_photon_observer_sphere_hits", "scripts/science/build_photon_observer_sphere_hits.py")
phase3_projection = load_module("build_photon_observer_camera_projection", "scripts/science/build_photon_observer_camera_projection.py")
phase4_redshift = load_module("build_photon_observer_camera_redshift", "scripts/science/build_photon_observer_camera_redshift.py")


P_MU_FIELDS = [
    "p_t_initial",
    "p_r_initial",
    "p_theta_initial",
    "p_phi_initial",
    "p_t_crossing",
    "p_r_crossing",
    "p_theta_crossing",
    "p_phi_crossing",
]

INITIAL_POSITION_FIELDS = [
    "initial_r_rg",
    "initial_theta_rad",
    "initial_phi_rad",
]


def assert_finite_p_mu(row: dict[str, object], fields: list[str] = P_MU_FIELDS) -> None:
    for field in fields:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            raise AssertionError(f"{field} is not finite in row: {row}")


def assert_finite_initial_position(row: dict[str, object]) -> None:
    for field in INITIAL_POSITION_FIELDS:
        value = row.get(field)
        if value is None or not math.isfinite(float(value)):
            raise AssertionError(f"{field} is not finite in row: {row}")


def compile_backend(tmp: Path) -> Path:
    binary = tmp / "compute_kerr_photon_escape_classifier"
    cmd = [
        "g++",
        "-std=c++17",
        "-Iinclude",
        "apps/compute_kerr_photon_escape_classifier.cpp",
        "src/photon_escape_classifier.cpp",
        "src/cascade/kerr_local_tetrad.cpp",
        "src/kerr_metric.cpp",
        "src/kerr_geodesic.cpp",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return binary


def classifier_command(
    tmp: Path,
    binary: Path,
    rows: list[dict[str, object]],
    *,
    invariant_tolerance: str = "1e-6",
    horizon_tolerance: str = "1e-6",
    fail_on_invariant: str = "true",
) -> tuple[list[str], Path, Path, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_path = tmp / "geant4_ready_particles.jsonl"
    output_jsonl = tmp / "photon_escape_classifier.jsonl"
    summary_csv = tmp / "photon_escape_summary.csv"
    summary_md = tmp / "photon_escape_summary.md"
    provenance = tmp / "photon_escape_provenance.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/science/run_kerr_photon_escape_classifier.py",
        "--input", str(input_path),
        "--output-jsonl", str(output_jsonl),
        "--summary-csv", str(summary_csv),
        "--summary-md", str(summary_md),
        "--provenance", str(provenance),
        "--backend", str(binary),
        "--spin", "0.0",
        "--observer-radius-rg", "20.0",
        "--max-radius-rg", "40.0",
        "--photon-geodesic-step-rg", "0.02",
        "--photon-max-geodesic-steps", "10000",
        "--photon-null-norm-tolerance", "1e-8",
        "--photon-invariant-tolerance", invariant_tolerance,
        "--photon-horizon-crossing-tolerance-rg", horizon_tolerance,
        "--photon-observer-crossing-tolerance-rg", "1e-8",
        "--photon-fail-on-invariant-violation", fail_on_invariant,
        "--photon-min-energy-gev", "0.0",
        "--photon-observer-frame", "ZAMO",
    ]
    return cmd, output_jsonl, summary_csv, provenance


def run_classifier(
    tmp: Path,
    binary: Path,
    rows: list[dict[str, object]],
    *,
    invariant_tolerance: str = "1e-6",
    horizon_tolerance: str = "1e-6",
    fail_on_invariant: str = "true",
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    cmd, output_jsonl, summary_csv, provenance = classifier_command(
        tmp,
        binary,
        rows,
        invariant_tolerance=invariant_tolerance,
        horizon_tolerance=horizon_tolerance,
        fail_on_invariant=fail_on_invariant,
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    out_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return out_rows, summary, prov


def photon_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": 1,
        "particle_id": 1,
        "pdg": 22,
        "energy_gev": 10.0,
        "global_exit_r_rg": 10.0,
        "global_exit_theta_rad": 1.5707963267948966,
        "global_exit_phi_rad": 0.0,
        "global_position_status": "GLOBAL_POSITION_VALID_ZAMO_TETRAD",
        "global_momentum_status": "GLOBAL_MOMENTUM_ZAMO_SPATIAL_TRIAD",
        "momentum_input_mode": "zamo_tetrad",
        "px": 0.0,
        "py": 0.0,
        "pz": 1.0,
    }
    row.update(updates)
    return row


def test_pdg_filtering(binary: Path, tmp: Path) -> None:
    rows = [photon_row(), photon_row(event_id=2, particle_id=2, pdg=11)]
    out_rows, summary, _ = run_classifier(tmp, binary, rows)
    if len(out_rows) != 1:
        raise AssertionError(f"expected one photon output row, got {out_rows}")
    if summary["n_input_particles"] != "2" or summary["n_photons"] != "1" or summary["n_non_photons"] != "1":
        raise AssertionError(f"bad filtering summary: {summary}")


def test_radial_outward_reaches_observer(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row()])
    if out_rows[0]["classification"] != "reaches_observer_sphere":
        raise AssertionError(f"outward photon did not reach observer: {out_rows[0]}")
    if out_rows[0]["observer_crossing_interpolated"] is not True:
        raise AssertionError(f"observer crossing was not interpolated: {out_rows[0]}")
    if abs(float(out_rows[0]["observer_crossing_r_rg"]) - 20.0) > 1.0e-8:
        raise AssertionError(f"bad observer crossing radius: {out_rows[0]}")
    if float(out_rows[0]["crossing_r_error_rg"]) > 1.0e-8:
        raise AssertionError(f"crossing radius was not refined to tolerance: {out_rows[0]}")
    if float(out_rows[0]["crossing_null_norm_abs_error"]) > 1.0e-6:
        raise AssertionError(f"crossing state is not null within tolerance: {out_rows[0]}")
    if out_rows[0].get("crossing_momentum_method") != "fractional_rk_crossing_state":
        raise AssertionError(f"crossing momentum method was not fractional RK: {out_rows[0]}")
    assert_finite_p_mu(out_rows[0])
    assert_finite_initial_position(out_rows[0])
    if out_rows[0].get("crossing_momentum_available") is not True:
        raise AssertionError(f"crossing momentum was not marked available: {out_rows[0]}")
    if "observed_energy_gev" in out_rows[0]:
        raise AssertionError(f"Phase 1 unexpectedly emitted observed_energy_gev: {out_rows[0]}")


def test_radial_inward_captured(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row(pz=-1.0)], fail_on_invariant="false")
    if out_rows[0]["classification"] != "captured_by_black_hole":
        raise AssertionError(f"inward photon was not captured: {out_rows[0]}")
    if int(out_rows[0]["geodesic_steps"]) <= 0:
        raise AssertionError(f"capture did not require an integrated horizon crossing: {out_rows[0]}")
    if out_rows[0].get("crossing_momentum_available") is not False:
        raise AssertionError(f"captured photon should not expose crossing momentum: {out_rows[0]}")
    for field in ["p_t_crossing", "p_r_crossing", "p_theta_crossing", "p_phi_crossing"]:
        if field in out_rows[0]:
            raise AssertionError(f"captured photon unexpectedly emitted {field}: {out_rows[0]}")


def test_ambiguous_generic_momentum_rejected(binary: Path, tmp: Path) -> None:
    row = photon_row()
    row.pop("momentum_input_mode")
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["classification"] != "integration_failed_ambiguous_momentum_input":
        raise AssertionError(f"ambiguous px/py/pz was not rejected: {out_rows[0]}")


def test_named_zamo_tetrad_direction_fields_accepted(binary: Path, tmp: Path) -> None:
    row = photon_row(n_zamo_r=1.0, n_zamo_theta=0.0, n_zamo_phi=0.0)
    row.pop("px")
    row.pop("py")
    row.pop("pz")
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["momentum_input_mode"] != "zamo_tetrad":
        raise AssertionError(f"momentum_input_mode was not preserved: {out_rows[0]}")
    if out_rows[0]["classification"] != "reaches_observer_sphere":
        raise AssertionError(f"named ZAMO tetrad fields were not accepted: {out_rows[0]}")


def test_unknown_momentum_input_mode_rejected(binary: Path, tmp: Path) -> None:
    row = photon_row(momentum_input_mode="unknown")
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["classification"] == "reaches_observer_sphere":
        raise AssertionError(f"unknown momentum_input_mode was accepted: {out_rows[0]}")
    if out_rows[0]["failure_reason"] != "integration_failed_invalid_or_missing_momentum_input_mode":
        raise AssertionError(f"unknown mode did not fail clearly: {out_rows[0]}")


def test_global_boyer_lindquist_mode_accepted(binary: Path, tmp: Path) -> None:
    row = photon_row(momentum_input_mode="global_boyer_lindquist")
    row.pop("px")
    row.pop("py")
    row.pop("pz")
    row.update({"global_px": 1.0, "global_py": 0.0, "global_pz": 0.0})
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["momentum_input_mode"] != "global_boyer_lindquist":
        raise AssertionError(f"momentum_input_mode was not preserved: {out_rows[0]}")
    if out_rows[0]["classification"] != "reaches_observer_sphere":
        raise AssertionError(f"global_boyer_lindquist mode was not accepted: {out_rows[0]}")


def test_invalid_null_momentum(binary: Path, tmp: Path) -> None:
    row = photon_row(momentum_input_mode="covariant_p_mu", p_t=-1.0, p_r=0.0, p_theta=0.0, p_phi=0.0)
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["classification"] != "integration_failed_invalid_null_momentum":
        raise AssertionError(f"invalid null momentum was not rejected: {out_rows[0]}")


def test_invariant_violation_can_fail(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row()], invariant_tolerance="1e-30")
    if out_rows[0]["classification"] != "integration_failed_invariant_violation":
        raise AssertionError(f"invariant violation did not fail: {out_rows[0]}")


def test_negative_tolerances_rejected(binary: Path, tmp: Path) -> None:
    cmd, _, _, _ = classifier_command(tmp, binary, [photon_row()], invariant_tolerance="-1.0")
    completed = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        raise AssertionError("negative photon_invariant_tolerance was accepted")
    if "Invalid photon escape classifier configuration" not in completed.stderr:
        raise AssertionError(f"negative tolerance did not fail clearly: {completed.stderr}")


def test_config_web_contains_all_parameters() -> None:
    text = (ROOT / "scripts/config_web_final.py").read_text(encoding="utf-8")
    for needle in [
        "enable_photon_observer_camera",
        "photon_observer_mode",
        "photon_observer_frame",
        "photon_null_norm_tolerance",
        "photon_invariant_tolerance",
        "photon_horizon_crossing_tolerance_rg",
        "photon_observer_crossing_tolerance_rg",
        "photon_fail_on_invariant_violation",
        "photon_max_geodesic_steps",
        "photon_geodesic_step_rg",
        "photon_min_energy_gev",
        "photon_camera_output_mode",
        "photon_redshift_mode",
        "photon_redshift_emitter_frame",
        "photon_redshift_observer_frame",
        "photon_redshift_energy_tolerance",
        "photon_redshift_fail_on_invalid",
        "enable_photon_validation_gate",
        "enable_photon_observer_science_products",
        "photon_observer_science_require_validation",
        "photon_camera_projection_mode",
        "photon_camera_fov_deg",
        "photon_camera_fov_definition",
        "photon_camera_resolution_mode",
        "photon_camera_center_theta_source",
        "photon_camera_center_phi_rad",
        "photon_camera_clipping_mode",
        "observer_sphere_hits",
        "observer_camera_projection",
    ]:
        if needle not in text:
            raise AssertionError(f"config_web_final.py missing {needle}")


def test_pipeline_passes_all_parameters(tmp: Path) -> None:
    config = {
        "run_name": "PhotonPipelinePass",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 3,
        "camera_ny": 3,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "escape_classifier",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_observer_crossing_tolerance_rg": 1.0e-8,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "disabled",
        "photon_redshift_emitter_frame": "ZAMO",
        "photon_redshift_observer_frame": "ZAMO",
        "photon_redshift_energy_tolerance": 1.0e-6,
        "photon_redshift_fail_on_invalid": True,
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": False,
        "photon_observer_science_require_validation": True,
        "photon_camera_projection_mode": "gnomonic_pinhole",
        "photon_camera_fov_deg": 60.0,
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_resolution_mode": "reuse_main_camera",
        "photon_camera_center_theta_source": "observer_inclination_deg",
        "photon_camera_center_phi_rad": 0.0,
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    step = next(item for item in steps if item.name == "photon_escape_classifier")
    command = step.command
    expected_pairs = {
        "--photon-max-geodesic-steps": "1234",
        "--photon-geodesic-step-rg": "0.03",
        "--photon-min-energy-gev": "2.0",
        "--photon-horizon-crossing-tolerance-rg": "1e-07",
        "--photon-observer-crossing-tolerance-rg": "1e-08",
        "--photon-observer-frame": "ZAMO",
        "--photon-fail-on-invariant-violation": "true",
    }
    for flag, value in expected_pairs.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise AssertionError(f"pipeline did not pass {flag}={value}: {command}")
    science_config = final_pipeline.config_for_interaction_scripts(config, tmp / "run" / "cascade")
    provenance = json.loads(science_config.read_text(encoding="utf-8"))["provenance"]
    if provenance.get("photon_camera_physical_interpretation") != "photon_escape_classifier":
        raise AssertionError(f"pipeline provenance missing photon interpretation: {provenance}")
    if provenance.get("photon_projected_to_pixels") is not False:
        raise AssertionError(f"pipeline provenance must record projected_to_pixels=false: {provenance}")


def test_pipeline_runs_phase1_then_phase2_for_observer_sphere_hits(tmp: Path) -> None:
    config = {
        "run_name": "PhotonSphereHitsPipeline",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 3,
        "camera_ny": 3,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_sphere_hits",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_observer_crossing_tolerance_rg": 1.0e-8,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "disabled",
        "photon_redshift_emitter_frame": "ZAMO",
        "photon_redshift_observer_frame": "ZAMO",
        "photon_redshift_energy_tolerance": 1.0e-6,
        "photon_redshift_fail_on_invalid": True,
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": False,
        "photon_observer_science_require_validation": True,
        "photon_camera_projection_mode": "gnomonic_pinhole",
        "photon_camera_fov_deg": 60.0,
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_resolution_mode": "reuse_main_camera",
        "photon_camera_center_theta_source": "observer_inclination_deg",
        "photon_camera_center_phi_rad": 0.0,
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    if "photon_escape_classifier" not in names or "photon_observer_sphere_hit_map" not in names:
        raise AssertionError(f"pipeline did not include Phase 1 and Phase 2: {names}")
    if names.index("photon_escape_classifier") > names.index("photon_observer_sphere_hit_map"):
        raise AssertionError(f"Phase 2 was not scheduled after Phase 1: {names}")
    phase2 = next(step for step in steps if step.name == "photon_observer_sphere_hit_map")
    command_text = " ".join(phase2.command)
    for expected in [
        "photon_escape_classifier.jsonl",
        "photon_observer_sphere_hits.jsonl",
        "photon_observer_sphere_summary.csv",
        "photon_observer_sphere_provenance.json",
    ]:
        if expected not in command_text:
            raise AssertionError(f"Phase 2 command missing {expected}: {phase2.command}")
    science_config = final_pipeline.config_for_interaction_scripts(config, tmp / "run" / "cascade")
    provenance = json.loads(science_config.read_text(encoding="utf-8"))["provenance"]
    if provenance.get("photon_observer_sphere_hit_map_enabled_effective") is not True:
        raise AssertionError(f"pipeline provenance missing Phase 2 enabled flag: {provenance}")
    if provenance.get("photon_observer_sphere_hits_camera_aperture") is not False:
        raise AssertionError(f"pipeline provenance must record no camera aperture hit: {provenance}")


def test_pipeline_runs_phase1_phase2_phase3_for_observer_camera_projection(tmp: Path) -> None:
    config = {
        "run_name": "PhotonCameraProjectionPipeline",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 7,
        "camera_ny": 5,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_camera_projection",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_observer_crossing_tolerance_rg": 1.0e-8,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "disabled",
        "photon_redshift_emitter_frame": "ZAMO",
        "photon_redshift_observer_frame": "ZAMO",
        "photon_redshift_energy_tolerance": 1.0e-6,
        "photon_redshift_fail_on_invalid": True,
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": False,
        "photon_observer_science_require_validation": True,
        "photon_camera_projection_mode": "gnomonic_pinhole",
        "photon_camera_fov_deg": 60.0,
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_resolution_mode": "reuse_main_camera",
        "photon_camera_center_theta_source": "observer_inclination_deg",
        "photon_camera_center_phi_rad": 0.25,
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    expected_order = [
        "photon_escape_classifier",
        "photon_observer_sphere_hit_map",
        "photon_observer_camera_projection",
    ]
    for name in expected_order:
        if name not in names:
            raise AssertionError(f"pipeline did not include {name}: {names}")
    if "photon_observer_camera_redshift" in names:
        raise AssertionError(f"disabled redshift mode should not schedule Phase 4: {names}")
    if [names.index(name) for name in expected_order] != sorted(names.index(name) for name in expected_order):
        raise AssertionError(f"Phase 3 was not scheduled after Phase 1 and Phase 2: {names}")
    phase3 = next(step for step in steps if step.name == "photon_observer_camera_projection")
    command_text = " ".join(phase3.command)
    for expected in [
        "photon_observer_sphere_hits.jsonl",
        "photon_observer_camera.csv",
        "photon_observer_camera_summary.csv",
        "photon_observer_camera_provenance.json",
        "--photon-camera-projection-mode gnomonic_pinhole",
        "--photon-camera-fov-definition square_half_angle",
        "--photon-camera-clipping-mode keep_outside_fov",
    ]:
        if expected not in command_text:
            raise AssertionError(f"Phase 3 command missing {expected}: {phase3.command}")
    science_config = final_pipeline.config_for_interaction_scripts(config, tmp / "run" / "cascade")
    provenance = json.loads(science_config.read_text(encoding="utf-8"))["provenance"]
    if provenance.get("photon_projected_to_pixels") is not True:
        raise AssertionError(f"pipeline provenance missing pixel projection flag: {provenance}")
    if provenance.get("photon_observer_camera_projection_enabled_effective") is not True:
        raise AssertionError(f"pipeline provenance missing Phase 3 enabled flag: {provenance}")
    if provenance.get("photon_observer_camera_aperture_acceptance_applied") is not False:
        raise AssertionError(f"pipeline provenance must record no aperture acceptance: {provenance}")


def test_pipeline_runs_phase4_only_for_validated_zamo(tmp: Path) -> None:
    config = {
        "run_name": "PhotonCameraRedshiftPipeline",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 7,
        "camera_ny": 5,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_camera_projection",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_observer_crossing_tolerance_rg": 1.0e-8,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "validated_zamo",
        "photon_redshift_emitter_frame": "ZAMO",
        "photon_redshift_observer_frame": "ZAMO",
        "photon_redshift_energy_tolerance": 1.0e-6,
        "photon_redshift_fail_on_invalid": False,
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": False,
        "photon_observer_science_require_validation": True,
        "photon_camera_projection_mode": "gnomonic_pinhole",
        "photon_camera_fov_deg": 60.0,
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_resolution_mode": "reuse_main_camera",
        "photon_camera_center_theta_source": "observer_inclination_deg",
        "photon_camera_center_phi_rad": 0.25,
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    expected_order = [
        "photon_escape_classifier",
        "photon_observer_sphere_hit_map",
        "photon_observer_camera_projection",
        "photon_observer_camera_redshift",
    ]
    for name in expected_order:
        if name not in names:
            raise AssertionError(f"pipeline did not include {name}: {names}")
    if [names.index(name) for name in expected_order] != sorted(names.index(name) for name in expected_order):
        raise AssertionError(f"Phase 4 was not scheduled after Phase 3: {names}")
    phase4 = next(step for step in steps if step.name == "photon_observer_camera_redshift")
    command_text = " ".join(phase4.command)
    for expected in [
        "photon_observer_camera.csv",
        "photon_observer_camera_redshift.csv",
        "photon_observer_camera_redshift_summary.csv",
        "photon_observer_camera_redshift_provenance.json",
        "--photon-redshift-mode validated_zamo",
        "--photon-redshift-emitter-frame ZAMO",
        "--photon-redshift-observer-frame ZAMO",
    ]:
        if expected not in command_text:
            raise AssertionError(f"Phase 4 command missing {expected}: {phase4.command}")


def test_pipeline_rejects_validated_zamo_without_camera_projection(tmp: Path) -> None:
    config = {
        "run_name": "PhotonInvalidRedshiftMode",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 7,
        "camera_ny": 5,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_sphere_hits",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_observer_crossing_tolerance_rg": 1.0e-8,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "validated_zamo",
        "photon_redshift_emitter_frame": "ZAMO",
        "photon_redshift_observer_frame": "ZAMO",
        "photon_redshift_energy_tolerance": 1.0e-6,
        "photon_redshift_fail_on_invalid": False,
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": False,
        "photon_observer_science_require_validation": True,
        "photon_camera_projection_mode": "gnomonic_pinhole",
        "photon_camera_fov_deg": 60.0,
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_resolution_mode": "reuse_main_camera",
        "photon_camera_center_theta_source": "observer_inclination_deg",
        "photon_camera_center_phi_rad": 0.25,
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    try:
        final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    except ValueError as exc:
        if "observer_camera_projection" not in str(exc):
            raise AssertionError(f"validated_zamo rejection error was unclear: {exc}") from exc
    else:
        raise AssertionError("validated_zamo without observer_camera_projection was accepted")


def test_wrapper_has_no_physical_defaults() -> None:
    text = (ROOT / "scripts/science/run_kerr_photon_escape_classifier.py").read_text(encoding="utf-8")
    forbidden = [
        'parser.add_argument("--spin", default=',
        'parser.add_argument("--observer-radius-rg", default=',
        'parser.add_argument("--max-radius-rg", default=',
        'parser.add_argument("--photon-geodesic-step-rg", default=',
        'parser.add_argument("--photon-max-geodesic-steps", default=',
        'parser.add_argument("--photon-null-norm-tolerance", default=',
        'parser.add_argument("--photon-invariant-tolerance", default=',
        'parser.add_argument("--photon-horizon-crossing-tolerance-rg", default=',
        'parser.add_argument("--photon-observer-crossing-tolerance-rg", default=',
        'parser.add_argument("--photon-min-energy-gev", default=',
        'default="ZAMO"',
        'default="true"',
    ]
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"hidden wrapper physical default remains: {needle}")


def test_pipeline_has_no_photon_physical_defaults() -> None:
    text = (ROOT / "scripts/run_hadros_final_pipeline.py").read_text(encoding="utf-8")
    forbidden = [
        'config.get("photon_observer_mode"',
        'config.get("photon_observer_frame"',
        'config.get("photon_null_norm_tolerance"',
        'config.get("photon_invariant_tolerance"',
        'config.get("photon_horizon_crossing_tolerance_rg"',
        'config.get("photon_observer_crossing_tolerance_rg"',
        'config.get("photon_fail_on_invariant_violation"',
        'config.get("photon_max_geodesic_steps"',
        'config.get("photon_geodesic_step_rg"',
        'config.get("photon_min_energy_gev"',
        'config.get("photon_camera_output_mode"',
        'config.get("photon_redshift_mode"',
        'config.get("photon_redshift_emitter_frame"',
        'config.get("photon_redshift_observer_frame"',
        'config.get("photon_redshift_energy_tolerance"',
        'config.get("photon_redshift_fail_on_invalid"',
        'config.get("enable_photon_validation_gate"',
        'config.get("enable_photon_observer_science_products"',
        'config.get("photon_observer_science_require_validation"',
        'config.get("photon_camera_projection_mode"',
        'config.get("photon_camera_fov_deg"',
        'config.get("photon_camera_fov_definition"',
        'config.get("photon_camera_resolution_mode"',
        'config.get("photon_camera_center_theta_source"',
        'config.get("photon_camera_center_phi_rad"',
        'config.get("photon_camera_clipping_mode"',
        'photon.get("photon_observer_mode"',
        'photon.get("photon_observer_frame"',
        'photon.get("photon_null_norm_tolerance"',
        'photon.get("photon_invariant_tolerance"',
        'photon.get("photon_horizon_crossing_tolerance_rg"',
        'photon.get("photon_observer_crossing_tolerance_rg"',
        'photon.get("photon_fail_on_invariant_violation"',
        'photon.get("photon_max_geodesic_steps"',
        'photon.get("photon_geodesic_step_rg"',
        'photon.get("photon_min_energy_gev"',
        'photon.get("photon_camera_output_mode"',
        'photon.get("photon_redshift_mode"',
        'photon.get("photon_redshift_emitter_frame"',
        'photon.get("photon_redshift_observer_frame"',
        'photon.get("photon_redshift_energy_tolerance"',
        'photon.get("photon_redshift_fail_on_invalid"',
        'photon.get("enable_photon_validation_gate"',
        'photon.get("enable_photon_observer_science_products"',
        'photon.get("photon_observer_science_require_validation"',
        'photon.get("photon_camera_projection_mode"',
        'photon.get("photon_camera_fov_deg"',
        'photon.get("photon_camera_fov_definition"',
        'photon.get("photon_camera_resolution_mode"',
        'photon.get("photon_camera_center_theta_source"',
        'photon.get("photon_camera_center_phi_rad"',
        'photon.get("photon_camera_clipping_mode"',
    ]
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"hidden pipeline photon default remains: {needle}")


def test_provenance_contains_phase1_limitations(binary: Path, tmp: Path) -> None:
    _, _, prov = run_classifier(tmp, binary, [photon_row()])
    expected = {
        "camera_physical_interpretation": "photon_escape_classifier",
        "camera_is_full_observational_transport": False,
        "projected_to_pixels": False,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise AssertionError(f"provenance[{key!r}]={prov.get(key)!r}, expected {value!r}")
    if prov.get("momentum_input_mode") != "per_record_required":
        raise AssertionError(f"provenance missing momentum_input_mode policy: {prov}")
    if float(prov.get("photon_horizon_crossing_tolerance_rg", -1.0)) < 0.0:
        raise AssertionError(f"provenance missing horizon tolerance: {prov}")
    if float(prov.get("photon_observer_crossing_tolerance_rg", -1.0)) <= 0.0:
        raise AssertionError(f"provenance missing observer crossing tolerance: {prov}")
    if prov.get("crossing_momentum_interpolation") != "none":
        raise AssertionError(f"provenance missing crossing momentum interpolation policy: {prov}")
    if prov.get("crossing_momentum_method") != "fractional_rk_crossing_state":
        raise AssertionError(f"provenance missing fractional RK crossing method: {prov}")


def phase1_hit_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": 7,
        "particle_id": 3,
        "pdg": 22,
        "input_energy_gev": 10.0,
        "classification": "reaches_observer_sphere",
        "observer_crossing_r_rg": 20.0,
        "observer_crossing_theta_rad": 1.2,
        "observer_crossing_phi_rad": 0.4,
        "observer_crossing_interpolated": True,
        "geodesic_steps": 42,
        "E_killing_initial": 0.9,
        "E_killing_final": 0.900000001,
        "Lz_initial": 0.1,
        "Lz_final": 0.100000001,
        "null_norm_max_abs_error": 1.0e-12,
        "relative_E_error": 1.0e-9,
        "relative_Lz_error": 1.0e-8,
        "momentum_input_mode": "zamo_tetrad",
        "initial_r_rg": 10.0,
        "initial_theta_rad": 1.2,
        "initial_phi_rad": 0.4,
        "p_t_initial": -10.0,
        "p_r_initial": 9.0,
        "p_theta_initial": 0.1,
        "p_phi_initial": 0.2,
        "p_t_crossing": -10.0,
        "p_r_crossing": 8.5,
        "p_theta_crossing": 0.15,
        "p_phi_crossing": 0.2,
        "crossing_momentum_available": True,
        "crossing_momentum_method": "fractional_rk_crossing_state",
        "crossing_r_error_rg": 1.0e-10,
        "crossing_null_norm_abs_error": 1.0e-12,
    }
    row.update(updates)
    return row


def run_phase2_hits(tmp: Path, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object], str]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_path = tmp / "photon_escape_classifier.jsonl"
    output_jsonl = tmp / "photon_observer_sphere_hits.jsonl"
    summary_csv = tmp / "photon_observer_sphere_summary.csv"
    summary_md = tmp / "photon_observer_sphere_summary.md"
    provenance = tmp / "photon_observer_sphere_provenance.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    args = type("Args", (), {})()
    args.input = input_path
    args.output_jsonl = output_jsonl
    args.summary_csv = summary_csv
    args.summary_md = summary_md
    args.provenance = provenance
    phase1_rows = phase2_hits.read_jsonl(args.input)
    hits = [phase2_hits.hit_from_row(row) for row in phase1_rows if row.get("classification") == "reaches_observer_sphere"]
    summary = phase2_hits.build_summary(phase1_rows, hits)
    phase2_hits.write_jsonl(output_jsonl, hits)
    phase2_hits.write_summary_csv(summary_csv, summary)
    phase2_hits.write_summary_md(summary_md, summary)
    phase2_hits.write_provenance(provenance, args, summary)
    out_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_row = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return out_rows, summary_row, prov, summary_md.read_text(encoding="utf-8")


def test_phase2_excludes_non_observer_sphere_classifications(tmp: Path) -> None:
    rows = [
        phase1_hit_row(event_id=1),
        phase1_hit_row(event_id=2, classification="captured_by_black_hole"),
        phase1_hit_row(event_id=3, classification="escapes_but_misses_observer"),
    ]
    hits, summary, _, _ = run_phase2_hits(tmp, rows)
    if [hit["event_id"] for hit in hits] != [1]:
        raise AssertionError(f"Phase 2 did not filter classifications correctly: {hits}")
    if summary["n_input_particles"] != "3" or summary["n_reached_observer_sphere"] != "1":
        raise AssertionError(f"bad Phase 2 filtering summary: {summary}")


def test_phase2_uses_interpolated_crossing_coordinates(tmp: Path) -> None:
    row = phase1_hit_row(observer_crossing_theta_rad=1.2345, observer_crossing_phi_rad=2.3456, geodesic_steps=77)
    hits, _, _, _ = run_phase2_hits(tmp, [row])
    hit = hits[0]
    if hit["observer_crossing_interpolated"] is not True:
        raise AssertionError(f"Phase 2 lost interpolation flag: {hit}")
    if hit["observer_crossing_theta_rad"] != 1.2345 or hit["observer_crossing_phi_rad"] != 2.3456:
        raise AssertionError(f"Phase 2 did not preserve crossing coordinates: {hit}")
    if hit["crossing_step_index"] != 77:
        raise AssertionError(f"Phase 2 did not preserve crossing step index: {hit}")


def test_phase2_preserves_covariant_momentum_fields(tmp: Path) -> None:
    source = phase1_hit_row()
    hits, _, _, _ = run_phase2_hits(tmp, [source])
    hit = hits[0]
    assert_finite_p_mu(hit)
    assert_finite_initial_position(hit)
    for field in P_MU_FIELDS:
        if float(hit[field]) != float(source[field]):
            raise AssertionError(f"Phase 2 did not preserve {field}: {hit}")
    for field in INITIAL_POSITION_FIELDS:
        if float(hit[field]) != float(source[field]):
            raise AssertionError(f"Phase 2 did not preserve {field}: {hit}")
    if hit.get("crossing_momentum_available") is not True:
        raise AssertionError(f"Phase 2 did not preserve crossing_momentum_available: {hit}")
    if hit.get("crossing_momentum_method") != "fractional_rk_crossing_state":
        raise AssertionError(f"Phase 2 did not preserve crossing_momentum_method: {hit}")
    for field in ["crossing_r_error_rg", "crossing_null_norm_abs_error"]:
        if float(hit[field]) != float(source[field]):
            raise AssertionError(f"Phase 2 did not preserve {field}: {hit}")


def test_phase2_summary_accumulates_reached_energy(tmp: Path) -> None:
    rows = [
        phase1_hit_row(event_id=1, input_energy_gev=10.0),
        phase1_hit_row(event_id=2, input_energy_gev=5.0, observer_crossing_theta_rad=1.4, observer_crossing_phi_rad=0.8),
        phase1_hit_row(event_id=3, input_energy_gev=100.0, classification="captured_by_black_hole"),
    ]
    _, summary, _, _ = run_phase2_hits(tmp, rows)
    if summary["n_reached_observer_sphere"] != "2":
        raise AssertionError(f"bad Phase 2 reached count: {summary}")
    if abs(float(summary["total_input_energy_reached_observer_sphere_gev"]) - 15.0) > 1.0e-12:
        raise AssertionError(f"bad Phase 2 total energy: {summary}")
    if abs(float(summary["mean_input_energy_reached_observer_sphere_gev"]) - 7.5) > 1.0e-12:
        raise AssertionError(f"bad Phase 2 mean energy: {summary}")


def test_phase2_outputs_avoid_pixel_detector_and_observed_energy_fields(tmp: Path) -> None:
    hits, _, _, summary_md = run_phase2_hits(tmp, [phase1_hit_row()])
    serialized_hits = json.dumps(hits, sort_keys=True)
    for forbidden in ["pixel_x", "pixel_y", "observed_energy_gev", "detector", "aperture"]:
        if forbidden in serialized_hits:
            raise AssertionError(f"Phase 2 hit output contains forbidden field {forbidden}: {serialized_hits}")
    for forbidden in ["pixel_x", "pixel_y", "observed_energy_gev"]:
        if forbidden in summary_md:
            raise AssertionError(f"Phase 2 summary contains forbidden field {forbidden}: {summary_md}")


def test_phase2_provenance_records_limitations(tmp: Path) -> None:
    _, _, prov, _ = run_phase2_hits(tmp, [phase1_hit_row()])
    expected = {
        "phase": "photon_observer_sphere_hit_map",
        "photon_observer_mode": "observer_sphere_hits",
        "projected_to_pixels": False,
        "hits_camera_aperture": False,
        "observer_sphere_crossing_is_detection": False,
        "observed_energy_available": False,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise AssertionError(f"Phase 2 provenance[{key!r}]={prov.get(key)!r}, expected {value!r}")


def direction_from_camera_coords(theta0_deg: float, phi0: float, camera_x: float, camera_y: float) -> tuple[float, float]:
    theta0 = math.radians(theta0_deg)
    c, e_x, e_y = phase3_projection.camera_basis(theta0, phi0)
    raw = (
        c[0] + camera_x * e_x[0] + camera_y * e_y[0],
        c[1] + camera_x * e_x[1] + camera_y * e_y[1],
        c[2] + camera_x * e_x[2] + camera_y * e_y[2],
    )
    norm = math.sqrt(sum(component * component for component in raw))
    unit = tuple(component / norm for component in raw)
    theta = math.acos(max(-1.0, min(1.0, unit[2])))
    phi = math.atan2(unit[1], unit[0]) % (2.0 * math.pi)
    return theta, phi


def run_phase3_projection(
    tmp: Path,
    rows: list[dict[str, object]],
    *,
    nx: int = 5,
    ny: int = 5,
    fov_deg: float = 60.0,
    theta0_deg: float = 70.0,
    phi0_rad: float = 0.0,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, object], str]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_path = tmp / "photon_observer_sphere_hits.jsonl"
    output_csv = tmp / "photon_observer_camera.csv"
    summary_csv = tmp / "photon_observer_camera_summary.csv"
    provenance = tmp / "photon_observer_camera_provenance.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    args = type("Args", (), {})()
    args.input = input_path
    args.output_csv = output_csv
    args.summary_csv = summary_csv
    args.provenance = provenance
    args.camera_nx = nx
    args.camera_ny = ny
    args.photon_camera_fov_deg = fov_deg
    args.photon_camera_projection_mode = "gnomonic_pinhole"
    args.photon_camera_fov_definition = "square_half_angle"
    args.photon_camera_resolution_mode = "reuse_main_camera"
    args.photon_camera_center_theta_source = "observer_inclination_deg"
    args.photon_camera_center_theta_deg = theta0_deg
    args.photon_camera_center_phi_rad = phi0_rad
    args.photon_camera_clipping_mode = "keep_outside_fov"
    phase3_projection.validate_config(args)
    theta0 = math.radians(args.photon_camera_center_theta_deg)
    c, e_x, e_y = phase3_projection.camera_basis(theta0, args.photon_camera_center_phi_rad)
    extent = math.tan(0.5 * math.radians(args.photon_camera_fov_deg))
    input_rows = phase3_projection.read_jsonl(args.input)
    out = [
        phase3_projection.project_hit(
            row,
            c=c,
            e_x=e_x,
            e_y=e_y,
            extent=extent,
            nx=args.camera_nx,
            ny=args.camera_ny,
            projection_mode=args.photon_camera_projection_mode,
        )
        for row in input_rows
    ]
    summary = phase3_projection.build_summary(out, args.photon_camera_projection_mode)
    phase3_projection.write_camera_csv(output_csv, out)
    phase3_projection.write_summary_csv(summary_csv, summary)
    phase3_projection.write_provenance(provenance, args, summary)
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        camera_rows = list(csv.DictReader(handle))
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_row = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return camera_rows, summary_row, prov, output_csv.read_text(encoding="utf-8")


def phase3_hit_at(theta: float, phi: float, **updates: object) -> dict[str, object]:
    row = phase1_hit_row(
        observer_crossing_theta_rad=theta,
        observer_crossing_phi_rad=phi,
        crossing_step_index=12,
    )
    row.pop("classification", None)
    row.update(updates)
    return row


def test_phase3_optical_center_maps_to_center_pixel(tmp: Path) -> None:
    theta, phi = math.radians(70.0), 0.0
    rows, summary, _, _ = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)])
    row = rows[0]
    if row["inside_fov"] != "true" or row["pixel_x"] != "2" or row["pixel_y"] != "2":
        raise AssertionError(f"optical center did not map to central pixel: {row}")
    if summary["n_input_hits"] != "1" or summary["n_inside_fov"] != "1":
        raise AssertionError(f"bad Phase 3 center summary: {summary}")


def test_phase3_camera_axis_signs_match_pixel_convention(tmp: Path) -> None:
    center_theta, center_phi = math.radians(70.0), 0.0
    x_theta, x_phi = direction_from_camera_coords(70.0, 0.0, 0.2, 0.0)
    y_theta, y_phi = direction_from_camera_coords(70.0, 0.0, 0.0, 0.2)
    rows, _, _, _ = run_phase3_projection(
        tmp,
        [
            phase3_hit_at(center_theta, center_phi, particle_id=1),
            phase3_hit_at(x_theta, x_phi, particle_id=2),
            phase3_hit_at(y_theta, y_phi, particle_id=3),
        ],
    )
    center, positive_x, positive_y = rows
    if int(positive_x["pixel_x"]) <= int(center["pixel_x"]):
        raise AssertionError(f"positive camera_x did not increase pixel_x: {rows}")
    if int(positive_y["pixel_y"]) >= int(center["pixel_y"]):
        raise AssertionError(f"positive camera_y did not decrease pixel_y: {rows}")


def test_phase3_fov_edge_clamps_to_edge_pixel(tmp: Path) -> None:
    extent = math.tan(0.5 * math.radians(60.0))
    theta, phi = direction_from_camera_coords(70.0, 0.0, extent, 0.0)
    rows, _, _, _ = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)])
    row = rows[0]
    if row["inside_fov"] != "true" or row["pixel_x"] != "4":
        raise AssertionError(f"FOV upper edge did not clamp to edge pixel: {row}")


def test_phase3_outside_fov_keeps_row_with_null_pixels(tmp: Path) -> None:
    extent = math.tan(0.5 * math.radians(60.0))
    theta, phi = direction_from_camera_coords(70.0, 0.0, 1.1 * extent, 0.0)
    rows, summary, _, _ = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)])
    row = rows[0]
    if row["inside_fov"] != "false" or row["projection_status"] != "outside_fov":
        raise AssertionError(f"outside-FOV row misclassified: {row}")
    if row["pixel_x"] != "" or row["pixel_y"] != "":
        raise AssertionError(f"outside-FOV row should keep null pixels: {row}")
    if summary["n_outside_fov"] != "1":
        raise AssertionError(f"bad outside-FOV summary: {summary}")


def test_phase3_nx_ne_ny_changes_only_pixel_sampling(tmp: Path) -> None:
    theta, phi = direction_from_camera_coords(70.0, 0.0, 0.15, -0.1)
    rows_a, _, _, _ = run_phase3_projection(tmp / "a", [phase3_hit_at(theta, phi)], nx=5, ny=5)
    rows_b, _, _, _ = run_phase3_projection(tmp / "b", [phase3_hit_at(theta, phi)], nx=9, ny=3)
    row_a, row_b = rows_a[0], rows_b[0]
    if abs(float(row_a["camera_x"]) - float(row_b["camera_x"])) > 1.0e-12:
        raise AssertionError(f"camera_x changed with resolution: {row_a} {row_b}")
    if abs(float(row_a["camera_y"]) - float(row_b["camera_y"])) > 1.0e-12:
        raise AssertionError(f"camera_y changed with resolution: {row_a} {row_b}")
    if (row_a["pixel_x"], row_a["pixel_y"]) == (row_b["pixel_x"], row_b["pixel_y"]):
        raise AssertionError(f"pixel sampling did not change for nx != ny: {row_a} {row_b}")


def test_phase3_phi_wraparound_projects_continuously(tmp: Path) -> None:
    phi0 = 2.0 * math.pi - 0.01
    theta, phi = direction_from_camera_coords(70.0, phi0, 0.05, 0.0)
    if not (phi < 0.1 or phi > 2.0 * math.pi - 0.1):
        raise AssertionError(f"test setup did not exercise phi wraparound: {phi}")
    rows, _, _, _ = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)], phi0_rad=phi0)
    row = rows[0]
    if abs(float(row["camera_x"]) - 0.05) > 1.0e-12:
        raise AssertionError(f"phi wraparound projection discontinuity: {row}")


def test_phase3_invalid_fov_and_resolution_rejected() -> None:
    args = type("Args", (), {})()
    args.camera_nx = 5
    args.camera_ny = 5
    args.photon_camera_fov_deg = 60.0
    args.photon_camera_projection_mode = "gnomonic_pinhole"
    args.photon_camera_fov_definition = "square_half_angle"
    args.photon_camera_resolution_mode = "reuse_main_camera"
    args.photon_camera_center_theta_source = "observer_inclination_deg"
    args.photon_camera_center_theta_deg = 70.0
    args.photon_camera_center_phi_rad = 0.0
    args.photon_camera_clipping_mode = "keep_outside_fov"
    for key, value in [
        ("photon_camera_fov_deg", 0.0),
        ("photon_camera_fov_deg", 180.0),
        ("camera_nx", 0),
        ("camera_ny", 0),
        ("photon_camera_center_theta_deg", 0.0),
    ]:
        old = getattr(args, key)
        setattr(args, key, value)
        try:
            phase3_projection.validate_config(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Phase 3 config was accepted: {key}={value}")
        setattr(args, key, old)


def test_phase3_outputs_avoid_observed_energy_detector_and_aperture_fields(tmp: Path) -> None:
    theta, phi = math.radians(70.0), 0.0
    rows, _, _, csv_text = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)])
    forbidden_text = ["observed_energy_gev", "detector", "aperture"]
    for forbidden in forbidden_text:
        if forbidden in csv_text:
            raise AssertionError(f"Phase 3 camera CSV contains forbidden field {forbidden}: {csv_text}")
    for forbidden in forbidden_text:
        if forbidden in rows[0]:
            raise AssertionError(f"Phase 3 camera row contains forbidden field {forbidden}: {rows[0]}")


def test_phase3_provenance_records_projection_contract(tmp: Path) -> None:
    theta, phi = math.radians(70.0), 0.0
    _, _, prov, _ = run_phase3_projection(tmp, [phase3_hit_at(theta, phi)])
    expected = {
        "phase": "photon_observer_camera_projection",
        "projected_to_pixels": True,
        "observer_sphere_crossing_is_detection": False,
        "observed_energy_available": False,
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "projection_mode": "gnomonic_pinhole",
        "photon_observer_mode": "observer_camera_projection",
        "photon_camera_fov_definition": "square_half_angle",
        "photon_camera_clipping_mode": "keep_outside_fov",
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise AssertionError(f"Phase 3 provenance[{key!r}]={prov.get(key)!r}, expected {value!r}")


def camera_row_for_redshift(**updates: object) -> dict[str, object]:
    initial_r = float(updates.get("initial_r_rg", 100.0))
    crossing_r = float(updates.get("observer_crossing_r_rg", 200.0))
    theta = float(updates.get("initial_theta_rad", 1.2))
    input_energy = float(updates.get("input_energy_gev", 10.0))
    alpha_initial = math.sqrt(1.0 - 2.0 / initial_r)
    p_t = -input_energy * alpha_initial
    row = phase1_hit_row(
        initial_r_rg=initial_r,
        initial_theta_rad=theta,
        initial_phi_rad=0.4,
        observer_crossing_r_rg=crossing_r,
        observer_crossing_theta_rad=theta,
        observer_crossing_phi_rad=0.4,
        p_t_initial=p_t,
        p_r_initial=0.0,
        p_theta_initial=0.0,
        p_phi_initial=0.0,
        p_t_crossing=p_t,
        p_r_crossing=0.0,
        p_theta_crossing=0.0,
        p_phi_crossing=0.0,
        input_energy_gev=input_energy,
        relative_E_error=1.0e-12,
        relative_Lz_error=1.0e-12,
        null_norm_max_abs_error=1.0e-12,
    )
    row.pop("classification", None)
    row.update(
        {
            "pixel_x": 2,
            "pixel_y": 2,
            "camera_x": 0.0,
            "camera_y": 0.0,
            "inside_fov": True,
            "projection_status": "inside_fov",
            "projection_mode": "gnomonic_pinhole",
        }
    )
    row.update(updates)
    return row


def run_phase4_redshift(
    tmp: Path,
    rows: list[dict[str, object]],
    *,
    mode: str = "validated_zamo",
    fail_on_invalid: bool = False,
    energy_tolerance: float = 1.0e-6,
    invariant_tolerance: float = 1.0e-6,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, object], str]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_csv = tmp / "photon_observer_camera.csv"
    output_csv = tmp / "photon_observer_camera_redshift.csv"
    summary_csv = tmp / "photon_observer_camera_redshift_summary.csv"
    provenance = tmp / "photon_observer_camera_redshift_provenance.json"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    args = type("Args", (), {})()
    args.input = input_csv
    args.output_csv = output_csv
    args.summary_csv = summary_csv
    args.provenance = provenance
    args.spin = 0.0
    args.photon_redshift_mode = mode
    args.photon_redshift_emitter_frame = "ZAMO"
    args.photon_redshift_observer_frame = "ZAMO"
    args.photon_redshift_energy_tolerance = energy_tolerance
    args.photon_redshift_fail_on_invalid = "true" if fail_on_invalid else "false"
    args.photon_invariant_tolerance = invariant_tolerance
    phase4_redshift.validate_config(args)
    input_fields, input_rows = phase4_redshift.read_camera_csv(args.input)
    processed = phase4_redshift.process_rows(input_rows, args)
    summary = phase4_redshift.build_summary(processed, args.photon_redshift_mode)
    phase4_redshift.write_rows(output_csv, phase4_redshift.output_fieldnames(input_fields, args.photon_redshift_mode), processed)
    phase4_redshift.write_summary_csv(summary_csv, summary)
    phase4_redshift.write_provenance(provenance, args, summary)
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        out_rows = list(csv.DictReader(handle))
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_row = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return out_rows, summary_row, prov, output_csv.read_text(encoding="utf-8")


def test_phase4_disabled_does_not_create_observed_energy(tmp: Path) -> None:
    rows, _, prov, csv_text = run_phase4_redshift(tmp, [camera_row_for_redshift()], mode="disabled")
    if "observed_energy_gev" in csv_text or "redshift_factor" in csv_text:
        raise AssertionError(f"disabled redshift mode created observed-energy fields: {csv_text}")
    if "observed_energy_gev" in rows[0]:
        raise AssertionError(f"disabled redshift row contains observed energy: {rows[0]}")
    if prov.get("observed_energy_available") is not False:
        raise AssertionError(f"disabled redshift provenance claims observed energy: {prov}")


def test_phase4_validated_zamo_produces_finite_observed_energy(tmp: Path) -> None:
    rows, summary, prov, _ = run_phase4_redshift(tmp, [camera_row_for_redshift()])
    row = rows[0]
    if row["redshift_status"] != "valid":
        raise AssertionError(f"valid redshift row was not marked valid: {row}")
    for field in ["emit_energy_zamo_gev", "observed_energy_gev", "redshift_factor"]:
        if not math.isfinite(float(row[field])) or float(row[field]) <= 0.0:
            raise AssertionError(f"invalid finite redshift field {field}: {row}")
    if summary["n_redshift_valid"] != "1" or summary["n_redshift_invalid"] != "0":
        raise AssertionError(f"bad redshift summary for valid row: {summary}")
    if prov.get("photon_redshift_emitter_frame") != "ZAMO" or prov.get("photon_redshift_observer_frame") != "ZAMO":
        raise AssertionError(f"redshift provenance missing frames: {prov}")
    if prov.get("observed_energy_available") is not True:
        raise AssertionError(f"redshift provenance did not expose observed energy availability: {prov}")


def test_phase4_missing_crossing_momentum_blocks_observed_energy(tmp: Path) -> None:
    row = camera_row_for_redshift()
    row.pop("p_t_crossing")
    rows, summary, _, _ = run_phase4_redshift(tmp, [row])
    out = rows[0]
    if out["redshift_status"] != "missing_required_momentum" or out["observed_energy_gev"] != "":
        raise AssertionError(f"missing crossing momentum did not block redshift: {out}")
    if summary["n_redshift_invalid"] != "1":
        raise AssertionError(f"bad missing-momentum summary: {summary}")


def test_phase4_emit_energy_mismatch_blocks_observed_energy(tmp: Path) -> None:
    row = camera_row_for_redshift(input_energy_gev=20.0)
    row["p_t_initial"] = -10.0 * math.sqrt(1.0 - 2.0 / float(row["initial_r_rg"]))
    row["p_t_crossing"] = row["p_t_initial"]
    rows, _, _, _ = run_phase4_redshift(tmp, [row])
    out = rows[0]
    if out["redshift_status"] != "invalid_emit_energy_mismatch" or out["observed_energy_gev"] != "":
        raise AssertionError(f"energy mismatch did not block observed energy: {out}")


def test_phase4_invalid_invariants_block_observed_energy(tmp: Path) -> None:
    rows, _, _, _ = run_phase4_redshift(tmp, [camera_row_for_redshift(null_norm_max_abs_error=1.0e-2)])
    out = rows[0]
    if out["redshift_status"] != "invalid_invariants" or out["observed_energy_gev"] != "":
        raise AssertionError(f"invalid invariants did not block observed energy: {out}")


def test_phase4_fail_on_invalid_raises(tmp: Path) -> None:
    try:
        run_phase4_redshift(
            tmp,
            [camera_row_for_redshift(null_norm_max_abs_error=1.0e-2)],
            fail_on_invalid=True,
        )
    except ValueError as exc:
        if "invalid_invariants" not in str(exc):
            raise AssertionError(f"fail_on_invalid raised unclear error: {exc}") from exc
    else:
        raise AssertionError("fail_on_invalid accepted an invalid redshift row")


def test_phase4_invalid_frames_and_tolerances_rejected(tmp: Path) -> None:
    args = type("Args", (), {})()
    args.input = tmp / "photon_observer_camera.csv"
    args.output_csv = tmp / "photon_observer_camera_redshift.csv"
    args.summary_csv = tmp / "photon_observer_camera_redshift_summary.csv"
    args.provenance = tmp / "photon_observer_camera_redshift_provenance.json"
    args.spin = 0.0
    args.photon_redshift_mode = "validated_zamo"
    args.photon_redshift_emitter_frame = "ZAMO"
    args.photon_redshift_observer_frame = "ZAMO"
    args.photon_redshift_energy_tolerance = 1.0e-6
    args.photon_redshift_fail_on_invalid = "true"
    args.photon_invariant_tolerance = 1.0e-6
    for key, value in [
        ("photon_redshift_emitter_frame", "STATIC"),
        ("photon_redshift_observer_frame", "STATIC"),
        ("photon_redshift_energy_tolerance", 0.0),
        ("photon_invariant_tolerance", 0.0),
    ]:
        old = getattr(args, key)
        setattr(args, key, value)
        try:
            phase4_redshift.validate_config(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Phase 4 config was accepted: {key}={value}")
        setattr(args, key, old)


def test_phase4_rows_do_not_add_detector_or_aperture_fields(tmp: Path) -> None:
    _, _, prov, csv_text = run_phase4_redshift(tmp, [camera_row_for_redshift()])
    for forbidden in ["detector", "aperture", "instrument_response"]:
        if forbidden in csv_text:
            raise AssertionError(f"Phase 4 CSV row contains forbidden field {forbidden}: {csv_text}")
    for key in ["detector_model_applied", "aperture_acceptance_applied", "instrument_response_applied"]:
        if prov.get(key) is not False:
            raise AssertionError(f"Phase 4 provenance missing false physical flag {key}: {prov}")


def test_phase3_preserves_covariant_momentum_fields(tmp: Path) -> None:
    theta, phi = math.radians(70.0), 0.0
    source = phase3_hit_at(theta, phi)
    rows, _, _, csv_text = run_phase3_projection(tmp, [source])
    row = rows[0]
    for field in INITIAL_POSITION_FIELDS + P_MU_FIELDS:
        if field not in row:
            raise AssertionError(f"Phase 3 camera CSV missing {field}: {csv_text}")
        if float(row[field]) != float(source[field]):
            raise AssertionError(f"Phase 3 did not preserve {field}: {row}")
    if row.get("crossing_momentum_available") != "true":
        raise AssertionError(f"Phase 3 did not preserve crossing_momentum_available: {row}")
    if row.get("crossing_momentum_method") != "fractional_rk_crossing_state":
        raise AssertionError(f"Phase 3 did not preserve crossing_momentum_method: {row}")
    for field in ["crossing_r_error_rg", "crossing_null_norm_abs_error"]:
        if field not in row:
            raise AssertionError(f"Phase 3 camera CSV missing {field}: {csv_text}")
        if float(row[field]) != float(source[field]):
            raise AssertionError(f"Phase 3 did not preserve {field}: {row}")
    if "observed_energy_gev" in csv_text:
        raise AssertionError(f"Phase 3 unexpectedly emitted observed_energy_gev: {csv_text}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_escape_tests_") as tmp_name:
        base = Path(tmp_name)
        binary = compile_backend(base)
        tests = [
            lambda: test_pdg_filtering(binary, base / "filter"),
            lambda: test_radial_outward_reaches_observer(binary, base / "outward"),
            lambda: test_radial_inward_captured(binary, base / "inward"),
            lambda: test_ambiguous_generic_momentum_rejected(binary, base / "ambiguous"),
            lambda: test_named_zamo_tetrad_direction_fields_accepted(binary, base / "named_zamo"),
            lambda: test_unknown_momentum_input_mode_rejected(binary, base / "unknown_mode"),
            lambda: test_global_boyer_lindquist_mode_accepted(binary, base / "global_bl"),
            lambda: test_invalid_null_momentum(binary, base / "invalid_null"),
            lambda: test_invariant_violation_can_fail(binary, base / "invariant"),
            lambda: test_negative_tolerances_rejected(binary, base / "negative_tolerance"),
            test_config_web_contains_all_parameters,
            lambda: test_pipeline_passes_all_parameters(base / "pipeline"),
            lambda: test_pipeline_runs_phase1_then_phase2_for_observer_sphere_hits(base / "pipeline_phase2"),
            lambda: test_pipeline_runs_phase1_phase2_phase3_for_observer_camera_projection(base / "pipeline_phase3"),
            lambda: test_pipeline_runs_phase4_only_for_validated_zamo(base / "pipeline_phase4"),
            lambda: test_pipeline_rejects_validated_zamo_without_camera_projection(base / "pipeline_phase4_reject"),
            test_wrapper_has_no_physical_defaults,
            test_pipeline_has_no_photon_physical_defaults,
            lambda: test_provenance_contains_phase1_limitations(binary, base / "provenance"),
            lambda: test_phase2_excludes_non_observer_sphere_classifications(base / "phase2_filter"),
            lambda: test_phase2_uses_interpolated_crossing_coordinates(base / "phase2_crossing"),
            lambda: test_phase2_preserves_covariant_momentum_fields(base / "phase2_momentum"),
            lambda: test_phase2_summary_accumulates_reached_energy(base / "phase2_summary"),
            lambda: test_phase2_outputs_avoid_pixel_detector_and_observed_energy_fields(base / "phase2_forbidden"),
            lambda: test_phase2_provenance_records_limitations(base / "phase2_provenance"),
            lambda: test_phase3_optical_center_maps_to_center_pixel(base / "phase3_center"),
            lambda: test_phase3_camera_axis_signs_match_pixel_convention(base / "phase3_axis"),
            lambda: test_phase3_fov_edge_clamps_to_edge_pixel(base / "phase3_edge"),
            lambda: test_phase3_outside_fov_keeps_row_with_null_pixels(base / "phase3_outside"),
            lambda: test_phase3_nx_ne_ny_changes_only_pixel_sampling(base / "phase3_aspect"),
            lambda: test_phase3_phi_wraparound_projects_continuously(base / "phase3_wrap"),
            test_phase3_invalid_fov_and_resolution_rejected,
            lambda: test_phase3_outputs_avoid_observed_energy_detector_and_aperture_fields(base / "phase3_forbidden"),
            lambda: test_phase3_provenance_records_projection_contract(base / "phase3_provenance"),
            lambda: test_phase4_disabled_does_not_create_observed_energy(base / "phase4_disabled"),
            lambda: test_phase4_validated_zamo_produces_finite_observed_energy(base / "phase4_valid"),
            lambda: test_phase4_missing_crossing_momentum_blocks_observed_energy(base / "phase4_missing"),
            lambda: test_phase4_emit_energy_mismatch_blocks_observed_energy(base / "phase4_mismatch"),
            lambda: test_phase4_invalid_invariants_block_observed_energy(base / "phase4_invariants"),
            lambda: test_phase4_fail_on_invalid_raises(base / "phase4_fail_on_invalid"),
            lambda: test_phase4_invalid_frames_and_tolerances_rejected(base / "phase4_invalid_config"),
            lambda: test_phase4_rows_do_not_add_detector_or_aperture_fields(base / "phase4_forbidden"),
            lambda: test_phase3_preserves_covariant_momentum_fields(base / "phase3_momentum"),
        ]
        for test in tests:
            test()
            name = getattr(test, "__name__", "lambda_test")
            print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
