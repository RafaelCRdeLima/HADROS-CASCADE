#!/usr/bin/env python3
"""Tests for the photon observer-camera validation gate."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "science" / "validate_photon_observer_camera.py"

from scripts import config_web_final
from scripts import run_hadros_final_pipeline as final_pipeline


CAMERA_FIELDS = [
    "event_id",
    "particle_id",
    "pixel_x",
    "pixel_y",
    "camera_x",
    "camera_y",
    "inside_fov",
    "projection_status",
    "input_energy_gev",
]

REDSHIFT_FIELDS = CAMERA_FIELDS + [
    "initial_r_rg",
    "initial_theta_rad",
    "initial_phi_rad",
    "observer_crossing_r_rg",
    "observer_crossing_theta_rad",
    "observer_crossing_phi_rad",
    "p_t_initial",
    "p_r_initial",
    "p_theta_initial",
    "p_phi_initial",
    "p_t_crossing",
    "p_r_crossing",
    "p_theta_crossing",
    "p_phi_crossing",
    "crossing_momentum_available",
    "crossing_momentum_method",
    "crossing_r_error_rg",
    "crossing_null_norm_abs_error",
    "null_norm_initial",
    "null_norm_max_abs_error",
    "relative_E_error",
    "relative_Lz_error",
    "emit_energy_zamo_gev",
    "observed_energy_gev",
    "redshift_factor",
    "redshift_status",
    "energy_emit_input_relative_error",
    "photon_redshift_mode",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def photon_row(*, bad_null_norm: bool = False) -> dict[str, object]:
    r_emit = 10.0
    r_obs = 80.0
    theta = math.pi / 2.0
    p_t = -1.0
    alpha_emit = math.sqrt(1.0 - 2.0 / r_emit)
    alpha_obs = math.sqrt(1.0 - 2.0 / r_obs)
    p_r_emit = 1.0 / (1.0 - 2.0 / r_emit)
    p_r_obs = 1.0 / (1.0 - 2.0 / r_obs)
    emit_energy = -p_t / alpha_emit
    observed_energy = -p_t / alpha_obs
    return {
        "event_id": 1,
        "particle_id": 1,
        "pixel_x": 4,
        "pixel_y": 4,
        "camera_x": 0.0,
        "camera_y": 0.0,
        "inside_fov": "true",
        "projection_status": "inside_fov",
        "input_energy_gev": emit_energy,
        "initial_r_rg": r_emit,
        "initial_theta_rad": theta,
        "initial_phi_rad": 0.0,
        "observer_crossing_r_rg": r_obs,
        "observer_crossing_theta_rad": theta,
        "observer_crossing_phi_rad": 0.0,
        "p_t_initial": p_t,
        "p_r_initial": 0.5 if bad_null_norm else p_r_emit,
        "p_theta_initial": 0.0,
        "p_phi_initial": 0.0,
        "p_t_crossing": p_t,
        "p_r_crossing": p_r_obs,
        "p_theta_crossing": 0.0,
        "p_phi_crossing": 0.0,
        "crossing_momentum_available": "true",
        "crossing_momentum_method": "fractional_rk_crossing_state",
        "crossing_r_error_rg": 0.0,
        "crossing_null_norm_abs_error": 1.0 if bad_null_norm else 0.0,
        "null_norm_initial": 1.0 if bad_null_norm else 0.0,
        "null_norm_max_abs_error": 1.0 if bad_null_norm else 0.0,
        "relative_E_error": 0.0,
        "relative_Lz_error": 0.0,
        "emit_energy_zamo_gev": emit_energy,
        "observed_energy_gev": observed_energy,
        "redshift_factor": observed_energy / emit_energy,
        "redshift_status": "valid",
        "energy_emit_input_relative_error": 0.0,
        "photon_redshift_mode": "validated_zamo",
    }


def write_inputs(
    tmp: Path,
    *,
    bad_null_norm: bool = False,
    legacy_missing_recoverable_config: bool = False,
    missing_essential_config: bool = False,
) -> dict[str, Path]:
    cascade = tmp / "cascade"
    camera_csv = cascade / "photon_observer_camera.csv"
    redshift_csv = cascade / "photon_observer_camera_redshift.csv"
    row = photon_row(bad_null_norm=bad_null_norm)
    write_csv(camera_csv, CAMERA_FIELDS, [{key: row[key] for key in CAMERA_FIELDS}])
    write_csv(redshift_csv, REDSHIFT_FIELDS, [row])
    camera_prov = cascade / "photon_observer_camera_provenance.json"
    camera_prov.write_text(
        json.dumps(
            {
                "camera_nx": 8,
                "camera_ny": 8,
                "photon_camera_fov_deg": 60.0,
                "photon_camera_center_theta_deg": 70.0,
                "photon_camera_center_phi_rad": 0.0,
                "detector_model_applied": False,
                "instrument_response_applied": False,
                "aperture_acceptance_applied": False,
            }
        ),
        encoding="utf-8",
    )
    redshift_prov = cascade / "photon_observer_camera_redshift_provenance.json"
    redshift_prov.write_text(
        json.dumps(
            {
                "phase": "photon_observer_camera_redshift",
                "detector_model_applied": False,
                "instrument_response_applied": False,
                "aperture_acceptance_applied": False,
            }
        ),
        encoding="utf-8",
    )
    values = config_web_final.defaults()
    values["photon_escape_classifier"]["enable_photon_observer_camera"] = True
    values["photon_escape_classifier"]["photon_observer_mode"] = "observer_camera_projection"
    values["photon_escape_classifier"]["photon_redshift_mode"] = "validated_zamo"
    pipeline_config = config_web_final.final_pipeline_config(values)
    if legacy_missing_recoverable_config:
        pipeline_config.pop("enable_photon_validation_gate", None)
        pipeline_config.pop("photon_observer_crossing_tolerance_rg", None)
        pipeline_config.get("photon_escape_classifier", {}).pop("enable_photon_validation_gate", None)
        pipeline_config.get("photon_escape_classifier", {}).pop("photon_observer_crossing_tolerance_rg", None)
    if missing_essential_config:
        pipeline_config.pop("photon_redshift_mode", None)
        pipeline_config.get("photon_escape_classifier", {}).pop("photon_redshift_mode", None)
    pipeline_path = cascade / "final_pipeline_science_config.json"
    pipeline_path.write_text(json.dumps(pipeline_config), encoding="utf-8")
    return {
        "camera_csv": camera_csv,
        "redshift_csv": redshift_csv,
        "camera_prov": camera_prov,
        "redshift_prov": redshift_prov,
        "pipeline_config": pipeline_path,
        "report": cascade / "photon_observer_camera_validation_report.md",
        "summary": cascade / "photon_observer_camera_validation_summary.csv",
        "provenance": cascade / "photon_observer_camera_validation_provenance.json",
    }


def run_gate(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--camera-csv",
            str(paths["camera_csv"]),
            "--redshift-csv",
            str(paths["redshift_csv"]),
            "--camera-provenance",
            str(paths["camera_prov"]),
            "--redshift-provenance",
            str(paths["redshift_prov"]),
            "--pipeline-config",
            str(paths["pipeline_config"]),
            "--report-md",
            str(paths["report"]),
            "--summary-csv",
            str(paths["summary"]),
            "--provenance",
            str(paths["provenance"]),
            "--spin",
            "0.0",
            "--photon-invariant-tolerance",
            "1e-6",
            "--photon-redshift-energy-tolerance",
            "1e-6",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_validation_gate_creates_report_and_flat_schwarzschild_pass() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_validation_") as tmp_name:
        paths = write_inputs(Path(tmp_name))
        completed = run_gate(paths)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        for key in ["report", "summary", "provenance"]:
            if not paths[key].exists() or paths[key].stat().st_size <= 0:
                raise AssertionError(f"missing validation output: {paths[key]}")
        rows = list(csv.DictReader(paths["summary"].open()))
        statuses = {row["test_name"]: row["status"] for row in rows}
        for name in ["flat_limit_redshift", "schwarzschild_radial_redshift", "zamo_redshift_consistency"]:
            if statuses.get(name) != "PASS":
                raise AssertionError(f"{name} did not pass: {rows}")
        for name in [
            "null_norm_initial",
            "null_norm_max_along_path",
            "null_norm_at_crossing",
            "crossing_r_error_rg",
            "null_norm_recomputed_from_output_initial",
            "null_norm_recomputed_from_output_crossing",
        ]:
            if statuses.get(name) != "WARNING":
                raise AssertionError(f"missing null-norm diagnostic {name}: {rows}")
        if statuses.get("crossing_momentum_method") != "PASS":
            raise AssertionError(rows)
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if provenance.get("physics_status") != "PASS" or provenance.get("config_status") != "PASS":
            raise AssertionError(provenance)
        if provenance.get("overall_status") != "PASS":
            raise AssertionError(provenance)
        for name in ["null_norm_kerr", "null_norm_recomputed_from_output_crossing"]:
            if name not in provenance.get("null_norm_diagnostics", {}):
                raise AssertionError(provenance)


def test_validation_gate_detects_synthetic_null_norm_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_validation_fail_") as tmp_name:
        paths = write_inputs(Path(tmp_name), bad_null_norm=True)
        completed = run_gate(paths)
        if completed.returncode == 0:
            raise AssertionError("validation gate accepted bad null norm")
        rows = list(csv.DictReader(paths["summary"].open()))
        null_row = next(row for row in rows if row["test_name"] == "null_norm_kerr")
        if null_row["status"] != "FAIL":
            raise AssertionError(rows)
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if provenance.get("overall_status") != "VALIDATION_FAILED":
            raise AssertionError(provenance)
        if provenance.get("physics_status") != "FAIL":
            raise AssertionError(provenance)


def test_validation_gate_warns_for_legacy_recoverable_config() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_validation_legacy_") as tmp_name:
        paths = write_inputs(Path(tmp_name), legacy_missing_recoverable_config=True)
        completed = run_gate(paths)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = list(csv.DictReader(paths["summary"].open()))
        config_row = next(row for row in rows if row["test_name"] == "config_web_final_contract")
        if config_row["status"] != "WARNING":
            raise AssertionError(rows)
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if provenance.get("physics_status") != "PASS":
            raise AssertionError(provenance)
        if provenance.get("config_status") != "WARNING":
            raise AssertionError(provenance)
        if provenance.get("overall_status") != "VALIDATION_WARNING":
            raise AssertionError(provenance)


def test_validation_gate_fails_for_missing_essential_config() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_validation_bad_config_") as tmp_name:
        paths = write_inputs(Path(tmp_name), missing_essential_config=True)
        completed = run_gate(paths)
        if completed.returncode == 0:
            raise AssertionError("validation gate accepted missing essential config")
        rows = list(csv.DictReader(paths["summary"].open()))
        config_row = next(row for row in rows if row["test_name"] == "config_web_final_contract")
        if config_row["status"] != "FAIL":
            raise AssertionError(rows)
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if provenance.get("physics_status") != "PASS":
            raise AssertionError(provenance)
        if provenance.get("config_status") != "FAIL":
            raise AssertionError(provenance)
        if provenance.get("overall_status") != "VALIDATION_FAILED":
            raise AssertionError(provenance)


def test_config_web_contains_validation_gate_parameter() -> None:
    values = config_web_final.defaults()
    if "enable_photon_validation_gate" not in values["photon_escape_classifier"]:
        raise AssertionError(values["photon_escape_classifier"])
    if values["photon_escape_classifier"]["enable_photon_validation_gate"] is not True:
        raise AssertionError(values["photon_escape_classifier"])
    pipeline = config_web_final.final_pipeline_config(values)
    if "enable_photon_validation_gate" not in pipeline:
        raise AssertionError(pipeline)
    if "photon_observer_crossing_tolerance_rg" not in pipeline:
        raise AssertionError(pipeline)


def test_pipeline_schedules_validation_gate_after_phase4() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_validation_pipeline_") as tmp_name:
        config = config_web_final.final_pipeline_config(config_web_final.defaults())
        config.update(
            {
                "output_dir": str(Path(tmp_name) / "run"),
                "physics_mode": "uhe_cascade",
                "enable_photon_observer_camera": True,
                "photon_observer_mode": "observer_camera_projection",
                "photon_redshift_mode": "validated_zamo",
                "enable_photon_validation_gate": True,
                "spin": 0.0,
                "black_hole_mass_msun": 3.0,
            }
        )
        config["photon_escape_classifier"] = dict(config["photon_escape_classifier"])
        config["photon_escape_classifier"].update(
            {
                "enable_photon_observer_camera": True,
                "photon_observer_mode": "observer_camera_projection",
                "photon_redshift_mode": "validated_zamo",
                "enable_photon_validation_gate": True,
            }
        )
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        if "photon_observer_camera_validation_gate" not in names:
            raise AssertionError(names)
        if names.index("photon_observer_camera_validation_gate") <= names.index("photon_observer_camera_redshift"):
            raise AssertionError(names)
        command = " ".join(next(step.command for step in steps if step.name == "photon_observer_camera_validation_gate"))
        for expected in [
            "validate_photon_observer_camera.py",
            "photon_observer_camera_validation_report.md",
            "photon_observer_camera_validation_summary.csv",
            "photon_observer_camera_validation_provenance.json",
        ]:
            if expected not in command:
                raise AssertionError(command)


if __name__ == "__main__":
    test_validation_gate_creates_report_and_flat_schwarzschild_pass()
    test_validation_gate_detects_synthetic_null_norm_failure()
    test_validation_gate_warns_for_legacy_recoverable_config()
    test_validation_gate_fails_for_missing_essential_config()
    test_config_web_contains_validation_gate_parameter()
    test_pipeline_schedules_validation_gate_after_phase4()
