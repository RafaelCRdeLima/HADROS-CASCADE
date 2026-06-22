#!/usr/bin/env python3
"""Lightweight tests for ideal photon observer science products."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/build_photon_observer_science_products.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import config_web_final  # noqa: E402
import run_hadros_final_pipeline as final_pipeline  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_validation(tmp: Path, *, physics: str = "PASS", config: str = "PASS", overall: str = "PASS") -> tuple[Path, Path]:
    summary = tmp / "photon_observer_camera_validation_summary.csv"
    provenance = tmp / "photon_observer_camera_validation_provenance.json"
    write_csv(
        summary,
        ["test_name", "physics_validated", "equation", "measured_error", "tolerance", "status", "notes"],
        [
            {
                "test_name": "null_norm_kerr",
                "physics_validated": "null geodesic",
                "equation": "g^{mu nu} p_mu p_nu = 0",
                "measured_error": "1e-12",
                "tolerance": "1e-6",
                "status": "PASS" if physics == "PASS" else "FAIL",
                "notes": "",
            }
        ],
    )
    provenance.write_text(
        json.dumps(
            {
                "phase": "photon_observer_camera_validation_gate",
                "physics_status": physics,
                "config_status": config,
                "overall_status": overall,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, provenance


def write_redshift_csv(path: Path) -> None:
    fields = [
        "pixel_x",
        "pixel_y",
        "inside_fov",
        "redshift_status",
        "input_energy_gev",
        "observed_energy_gev",
        "redshift_factor",
    ]
    write_csv(
        path,
        fields,
        [
            {
                "pixel_x": 0,
                "pixel_y": 0,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 10.0,
                "observed_energy_gev": 8.0,
                "redshift_factor": 0.8,
            },
            {
                "pixel_x": 0,
                "pixel_y": 0,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 5.0,
                "observed_energy_gev": 4.0,
                "redshift_factor": 0.8,
            },
            {
                "pixel_x": 1,
                "pixel_y": 0,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 2.0,
                "observed_energy_gev": 1.0,
                "redshift_factor": 0.5,
            },
            {
                "pixel_x": 1,
                "pixel_y": 1,
                "inside_fov": "true",
                "redshift_status": "invalid_null_norm",
                "input_energy_gev": 99.0,
                "observed_energy_gev": 99.0,
                "redshift_factor": 1.0,
            },
            {
                "pixel_x": 1,
                "pixel_y": 1,
                "inside_fov": "false",
                "redshift_status": "valid",
                "input_energy_gev": 77.0,
                "observed_energy_gev": 77.0,
                "redshift_factor": 1.0,
            },
            {
                "pixel_x": 1,
                "pixel_y": 1,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 3.0,
                "observed_energy_gev": "",
                "redshift_factor": 1.0,
            },
        ],
    )


def run_products(tmp: Path, *, physics: str = "PASS", config: str = "PASS", overall: str = "PASS") -> subprocess.CompletedProcess[str]:
    redshift = tmp / "photon_observer_camera_redshift.csv"
    write_redshift_csv(redshift)
    summary, provenance = write_validation(tmp, physics=physics, config=config, overall=overall)
    out = tmp / "science"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--redshift-csv",
            str(redshift),
            "--validation-summary-csv",
            str(summary),
            "--validation-provenance",
            str(provenance),
            "--output-dir",
            str(out),
            "--camera-nx",
            "2",
            "--camera-ny",
            "2",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_science_products_generate_validated_maps_and_histograms() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_science_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_products(tmp)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        out = tmp / "science"
        counts = {(int(row["pixel_x"]), int(row["pixel_y"])): int(row["n_photons"]) for row in read_csv(out / "photon_observer_counts_map.csv")}
        observed = {
            (int(row["pixel_x"]), int(row["pixel_y"])): float(row["sum_observed_energy_gev"])
            for row in read_csv(out / "photon_observer_observed_energy_map.csv")
        }
        redshift = {
            (int(row["pixel_x"]), int(row["pixel_y"])): row["mean_redshift_factor"]
            for row in read_csv(out / "photon_observer_mean_redshift_map.csv")
        }
        if counts != {(0, 0): 2, (1, 0): 1, (0, 1): 0, (1, 1): 0}:
            raise AssertionError(counts)
        if observed[(0, 0)] != 12.0 or observed[(1, 0)] != 1.0:
            raise AssertionError(observed)
        if float(redshift[(0, 0)]) != 0.8 or float(redshift[(1, 0)]) != 0.5 or redshift[(0, 1)] != "":
            raise AssertionError(redshift)
        hist_count = sum(int(row["count"]) for row in read_csv(out / "photon_observer_spectrum_observed.csv"))
        if hist_count != 3:
            raise AssertionError(hist_count)
        provenance = json.loads((out / "photon_observer_science_provenance.json").read_text(encoding="utf-8"))
        if provenance["detector_model_applied"] is not False or provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)
        if provenance["n_valid_photons"] != 3:
            raise AssertionError(provenance)


def test_validation_warning_is_allowed_but_physics_fail_is_blocked() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_science_warning_") as tmp_name:
        tmp = Path(tmp_name)
        warning = run_products(tmp, config="WARNING", overall="VALIDATION_WARNING")
        if warning.returncode != 0:
            raise AssertionError(warning.stderr)
    with tempfile.TemporaryDirectory(prefix="hadros_photon_science_fail_") as tmp_name:
        tmp = Path(tmp_name)
        failed = run_products(tmp, physics="FAIL", config="PASS", overall="VALIDATION_FAILED")
        if failed.returncode == 0:
            raise AssertionError("physics FAIL validation was accepted")
        if (tmp / "science" / "photon_observer_science_provenance.json").exists():
            raise AssertionError("science provenance was written for failed physics validation")


def test_config_web_and_preset_contain_science_parameters() -> None:
    values = config_web_final.defaults()
    photon = values["photon_escape_classifier"]
    for key in ["enable_photon_observer_science_products", "photon_observer_science_require_validation"]:
        if key not in photon:
            raise AssertionError(key)
    pipeline = config_web_final.final_pipeline_config(values)
    for key in ["enable_photon_observer_science_products", "photon_observer_science_require_validation"]:
        if key not in pipeline or key not in pipeline["photon_escape_classifier"]:
            raise AssertionError(key)
    preset = json.loads((ROOT / "presets/config_web/final_pipeline_config.json").read_text(encoding="utf-8"))
    if preset.get("enable_photon_observer_science_products") is not False:
        raise AssertionError(preset)
    if preset.get("photon_observer_science_require_validation") is not True:
        raise AssertionError(preset)


def test_pipeline_schedules_science_products_after_validation_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_science_pipeline_") as tmp_name:
        config = config_web_final.final_pipeline_config(config_web_final.defaults())
        config.update(
            {
                "output_dir": str(Path(tmp_name) / "run"),
                "physics_mode": "uhe_cascade",
                "enable_photon_observer_camera": True,
                "photon_observer_mode": "observer_camera_projection",
                "photon_redshift_mode": "validated_zamo",
                "enable_photon_validation_gate": True,
                "enable_photon_observer_science_products": True,
                "photon_observer_science_require_validation": True,
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
                "enable_photon_observer_science_products": True,
                "photon_observer_science_require_validation": True,
            }
        )
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        if "photon_observer_science_products" not in names:
            raise AssertionError(names)
        if names.index("photon_observer_science_products") <= names.index("photon_observer_camera_validation_gate"):
            raise AssertionError(names)
        command = " ".join(next(step.command for step in steps if step.name == "photon_observer_science_products"))
        for expected in [
            "build_photon_observer_science_products.py",
            "photon_observer_camera_redshift.csv",
            "photon_observer_camera_validation_summary.csv",
            "photon_observer_science_products",
        ]:
            if expected not in command:
                raise AssertionError(command)


def test_pipeline_rejects_science_products_without_validation_gate() -> None:
    config = config_web_final.final_pipeline_config(config_web_final.defaults())
    config.update(
        {
            "enable_photon_observer_camera": True,
            "photon_observer_mode": "observer_camera_projection",
            "photon_redshift_mode": "validated_zamo",
            "enable_photon_validation_gate": False,
            "enable_photon_observer_science_products": True,
            "photon_observer_science_require_validation": True,
        }
    )
    config["photon_escape_classifier"] = dict(config["photon_escape_classifier"])
    config["photon_escape_classifier"].update(
        {
            "enable_photon_observer_camera": True,
            "photon_observer_mode": "observer_camera_projection",
            "photon_redshift_mode": "validated_zamo",
            "enable_photon_validation_gate": False,
            "enable_photon_observer_science_products": True,
            "photon_observer_science_require_validation": True,
        }
    )
    try:
        final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    except ValueError as exc:
        if "validation_gate" not in str(exc):
            raise AssertionError(exc) from exc
    else:
        raise AssertionError("science products were accepted without validation gate")


if __name__ == "__main__":
    test_science_products_generate_validated_maps_and_histograms()
    test_validation_warning_is_allowed_but_physics_fail_is_blocked()
    test_config_web_and_preset_contain_science_parameters()
    test_pipeline_schedules_science_products_after_validation_gate()
    test_pipeline_rejects_science_products_without_validation_gate()
