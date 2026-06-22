#!/usr/bin/env python3
"""Lightweight tests for photon observer spectra products."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/build_photon_observer_spectra.py"
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


def write_redshift(path: Path) -> None:
    fields = [
        "event_id",
        "particle_id",
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
            {"event_id": 1, "particle_id": 1, "pixel_x": 0, "pixel_y": 0, "inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 10.0, "observed_energy_gev": 1.0, "redshift_factor": 0.1},
            {"event_id": 2, "particle_id": 2, "pixel_x": 0, "pixel_y": 0, "inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 20.0, "observed_energy_gev": 2.0, "redshift_factor": 0.1},
            {"event_id": 3, "particle_id": 3, "pixel_x": 1, "pixel_y": 0, "inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 40.0, "observed_energy_gev": 4.0, "redshift_factor": 0.1},
            {"event_id": 4, "particle_id": 4, "pixel_x": "", "pixel_y": "", "inside_fov": "false", "redshift_status": "valid", "input_energy_gev": 80.0, "observed_energy_gev": 8.0, "redshift_factor": 0.1},
            {"event_id": 5, "particle_id": 5, "pixel_x": 1, "pixel_y": 1, "inside_fov": "true", "redshift_status": "invalid_null_norm", "input_energy_gev": 999.0, "observed_energy_gev": 999.0, "redshift_factor": 1.0},
            {"event_id": 6, "particle_id": 6, "pixel_x": 1, "pixel_y": 1, "inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 5.0, "observed_energy_gev": "", "redshift_factor": 1.0},
            {"event_id": 7, "particle_id": 7, "pixel_x": 1, "pixel_y": 1, "inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 5.0, "observed_energy_gev": -1.0, "redshift_factor": 1.0},
        ],
    )


def write_attenuated(path: Path) -> None:
    fields = [
        "inside_fov",
        "redshift_status",
        "observed_energy_gev",
        "attenuated_observed_energy_gev",
        "redshift_factor",
        "input_energy_gev",
        "photon_opacity_status",
        "photon_opacity_mode",
    ]
    write_csv(
        path,
        fields,
        [
            {"inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 10.0, "observed_energy_gev": 1.0, "attenuated_observed_energy_gev": 1.0, "redshift_factor": 0.1, "photon_opacity_status": "valid_vacuum", "photon_opacity_mode": "vacuum"},
            {"inside_fov": "true", "redshift_status": "valid", "input_energy_gev": 20.0, "observed_energy_gev": 2.0, "attenuated_observed_energy_gev": 2.0, "redshift_factor": 0.1, "photon_opacity_status": "valid_vacuum", "photon_opacity_mode": "vacuum"},
        ],
    )


def run_spectra(
    tmp: Path,
    *,
    selection: str = "inside_fov_only",
    binning: str = "log",
    n_bins: int = 4,
    energy_min: str = "1",
    energy_max: str = "16",
    plots: bool = False,
    frequency: bool = True,
    physics: str = "PASS",
    config: str = "PASS",
    overall: str = "PASS",
    attenuated: bool = False,
) -> subprocess.CompletedProcess[str]:
    redshift = tmp / "photon_observer_camera_redshift.csv"
    write_redshift(redshift)
    summary, provenance = write_validation(tmp, physics=physics, config=config, overall=overall)
    attenuated_csv = tmp / "photon_observer_camera_attenuated.csv"
    if attenuated:
        write_attenuated(attenuated_csv)
    command = [
        sys.executable,
        str(SCRIPT),
        "--redshift-csv",
        str(redshift),
        "--validation-summary-csv",
        str(summary),
        "--validation-provenance",
        str(provenance),
        "--output-dir",
        str(tmp / "spectra"),
        "--photon-spectrum-selection",
        selection,
        "--photon-spectrum-binning",
        binning,
        "--photon-spectrum-n-bins",
        str(n_bins),
        "--photon-spectrum-energy-min-gev",
        energy_min,
        "--photon-spectrum-energy-max-gev",
        energy_max,
        "--photon-spectrum-generate-plots",
        "true" if plots else "false",
        "--photon-spectrum-include-frequency",
        "true" if frequency else "false",
        "--photon-spectrum-require-validation",
        "true",
    ]
    if attenuated:
        command.extend(["--attenuated-csv", str(attenuated_csv)])
    return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)


def test_validation_pass_and_warning_generate_spectra() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_pass_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(tmp)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        if not (tmp / "spectra" / "photon_observer_spectrum_observed.csv").exists():
            raise AssertionError("observed spectrum was not written")
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_warning_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(tmp, config="WARNING", overall="VALIDATION_WARNING")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)


def test_validation_failed_blocks_spectra() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_fail_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(tmp, physics="FAIL", overall="VALIDATION_FAILED")
        if completed.returncode == 0:
            raise AssertionError("validation failed but spectra were generated")
        if (tmp / "spectra" / "photon_observer_spectra_provenance.json").exists():
            raise AssertionError("provenance was written despite validation failure")


def test_selection_modes_and_invalid_energy_filtering() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_selection_") as tmp_name:
        tmp = Path(tmp_name)
        inside = run_spectra(tmp / "inside", selection="inside_fov_only", binning="linear", energy_min="1", energy_max="9")
        if inside.returncode != 0:
            raise AssertionError(inside.stderr)
        inside_count = sum(int(row["counts"]) for row in read_csv(tmp / "inside" / "spectra" / "photon_observer_spectrum_observed.csv"))
        if inside_count != 3:
            raise AssertionError(inside_count)
        all_rows = run_spectra(tmp / "all", selection="all_reached_observer_sphere", binning="linear", energy_min="1", energy_max="9")
        if all_rows.returncode != 0:
            raise AssertionError(all_rows.stderr)
        all_count = sum(int(row["counts"]) for row in read_csv(tmp / "all" / "spectra" / "photon_observer_spectrum_observed.csv"))
        if all_count != 4:
            raise AssertionError(all_count)


def test_linear_bins_derivatives_and_frequency_constant() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_linear_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(tmp, selection="inside_fov_only", binning="linear", n_bins=4, energy_min="1", energy_max="5")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = read_csv(tmp / "spectra" / "photon_observer_spectrum_observed.csv")
        first = rows[0]
        if int(first["counts"]) != 1:
            raise AssertionError(rows)
        if abs(float(first["dN_dE"]) - 1.0) > 1.0e-12:
            raise AssertionError(first)
        center = float(first["energy_center_gev"])
        if abs(float(first["E_dN_dE"]) - center * float(first["dN_dE"])) > 1.0e-12:
            raise AssertionError(first)
        if abs(float(first["E2_dN_dE"]) - center * center * float(first["dN_dE"])) > 1.0e-12:
            raise AssertionError(first)
        frequency_rows = read_csv(tmp / "spectra" / "photon_observer_spectrum_frequency.csv")
        provenance = json.loads((tmp / "spectra" / "photon_observer_spectra_provenance.json").read_text(encoding="utf-8"))
        h_gev_s = float(provenance["planck_constant_gev_s"])
        if abs(float(frequency_rows[0]["frequency_min_hz"]) - 1.0 / h_gev_s) / (1.0 / h_gev_s) > 1.0e-12:
            raise AssertionError(frequency_rows[0])


def test_log_bins_plots_attenuated_and_no_paper_ready_names() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_log_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(tmp, plots=True, attenuated=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        out = tmp / "spectra"
        for name in [
            "photon_observer_spectrum_observed_counts.png",
            "photon_observer_spectrum_observed_E2dNdE.png",
            "photon_observer_spectrum_input_vs_observed.png",
            "photon_observer_spectrum_frequency_counts.png",
            "photon_observer_spectrum_attenuated.csv",
        ]:
            if not (out / name).exists():
                raise AssertionError(f"missing {name}")
        for path in out.iterdir():
            if "paper" in path.name.lower() or "ready" in path.name.lower():
                raise AssertionError(f"paper-ready naming leaked into spectra output: {path.name}")
        provenance = json.loads((out / "photon_observer_spectra_provenance.json").read_text(encoding="utf-8"))
        if provenance["detector_model_applied"] is not False or provenance["instrument_response_applied"] is not False:
            raise AssertionError(provenance)
        if provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)
        if "not paper-ready" not in "\n".join(provenance["physical_limitations"]):
            raise AssertionError(provenance)


def test_histogram_count_conservation_and_no_unexpected_flow() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_counts_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_spectra(
            tmp,
            binning="log",
            n_bins=8,
            energy_min="auto",
            energy_max="auto",
            attenuated=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        out = tmp / "spectra"
        expected_counts = {
            "photon_observer_spectrum_observed.csv": 3,
            "photon_observer_spectrum_input.csv": 3,
            "photon_observer_spectrum_frequency.csv": 3,
            "photon_observer_spectrum_attenuated.csv": 2,
        }
        for filename, expected in expected_counts.items():
            rows = read_csv(out / filename)
            actual = sum(int(row["counts"]) for row in rows)
            if actual != expected:
                raise AssertionError(f"{filename}: counts={actual} expected={expected}")
        summary = (out / "photon_observer_spectra_summary.md").read_text(encoding="utf-8")
        for expected in [
            "| observed_histogram_counts | 3 |",
            "| input_histogram_counts | 3 |",
            "| frequency_histogram_counts | 3 |",
            "| attenuated_histogram_counts | 2 |",
            "| histogram_count_conservation_error_total | 0 |",
            "| histogram_underflow_count_total | 0 |",
            "| histogram_overflow_count_total | 0 |",
        ]:
            if expected not in summary:
                raise AssertionError(summary)
        if "underflow" not in summary.lower() or "overflow" not in summary.lower():
            raise AssertionError(summary)
        provenance = json.loads((out / "photon_observer_spectra_provenance.json").read_text(encoding="utf-8"))
        if "photon_absorption_applied" not in provenance:
            raise AssertionError(provenance)
        if provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)
        for name, expected in {
            "observed": 3,
            "input": 3,
            "frequency": 3,
            "attenuated": 2,
        }.items():
            audit = provenance["histogram_audit"][name]
            if audit["histogram_counts"] != expected:
                raise AssertionError(provenance["histogram_audit"])
            if audit["underflow_count"] != 0 or audit["overflow_count"] != 0:
                raise AssertionError(provenance["histogram_audit"])


def test_config_web_preset_and_pipeline_contract() -> None:
    values = config_web_final.defaults()
    photon = values["photon_escape_classifier"]
    keys = [
        "enable_photon_observer_spectra",
        "photon_spectrum_selection",
        "photon_spectrum_binning",
        "photon_spectrum_n_bins",
        "photon_spectrum_energy_min_gev",
        "photon_spectrum_energy_max_gev",
        "photon_spectrum_generate_plots",
        "photon_spectrum_include_frequency",
        "photon_spectrum_require_validation",
    ]
    for key in keys:
        if key not in photon:
            raise AssertionError(key)
    pipeline = config_web_final.final_pipeline_config(values)
    for key in keys:
        if key not in pipeline or key not in pipeline["photon_escape_classifier"]:
            raise AssertionError(key)
    preset = json.loads((ROOT / "presets/config_web/final_pipeline_config.json").read_text(encoding="utf-8"))
    if preset.get("enable_photon_observer_spectra") is not False:
        raise AssertionError(preset)


def test_photon_observer_full_validated_preset_contract() -> None:
    preset_path = ROOT / "presets/config_web/photon_observer_full_validated.json"
    if not preset_path.exists():
        raise AssertionError("missing photon_observer_full_validated preset")
    values = config_web_final.load_values(preset_path)
    pipeline = config_web_final.final_pipeline_config(values)
    expected = {
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_camera_projection",
        "photon_redshift_mode": "validated_zamo",
        "enable_photon_validation_gate": True,
        "enable_photon_observer_science_products": True,
        "enable_photon_observer_spectra": True,
        "enable_photon_opacity": True,
        "photon_opacity_mode": "vacuum",
        "enable_photon_path_sampling": False,
    }
    for key, value in expected.items():
        if pipeline.get(key) != value:
            raise AssertionError((key, pipeline.get(key), value))
        if pipeline["photon_escape_classifier"].get(key) != value:
            raise AssertionError((key, pipeline["photon_escape_classifier"].get(key), value))
    if pipeline.get("photon_opacity_mode") == "tabulated_gray":
        raise AssertionError(pipeline)
    if pipeline.get("enable_neutrino_camera"):
        raise AssertionError(pipeline)
    steps = final_pipeline.build_steps(pipeline, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    for name in [
        "photon_observer_camera_validation_gate",
        "photon_observer_opacity_vacuum",
        "photon_observer_science_products",
        "photon_observer_spectra",
    ]:
        if name not in names:
            raise AssertionError(names)


def test_pipeline_schedules_spectra_after_validation_opacity_and_science_products() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_spectra_pipeline_") as tmp_name:
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
                "enable_photon_observer_spectra": True,
                "enable_photon_opacity": True,
                "photon_opacity_mode": "vacuum",
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
                "enable_photon_observer_spectra": True,
                "enable_photon_opacity": True,
                "photon_opacity_mode": "vacuum",
            }
        )
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        for name in [
            "photon_observer_camera_validation_gate",
            "photon_observer_opacity_vacuum",
            "photon_observer_science_products",
            "photon_observer_spectra",
        ]:
            if name not in names:
                raise AssertionError(names)
        if names.index("photon_observer_spectra") <= names.index("photon_observer_science_products"):
            raise AssertionError(names)
        if names.index("photon_observer_spectra") <= names.index("photon_observer_opacity_vacuum"):
            raise AssertionError(names)
        command = " ".join(next(step.command for step in steps if step.name == "photon_observer_spectra"))
        for expected in [
            "build_photon_observer_spectra.py",
            "--photon-spectrum-selection inside_fov_only",
            "--photon-spectrum-binning log",
            "--photon-spectrum-n-bins 32",
            "photon_observer_camera_attenuated.csv",
        ]:
            if expected not in command:
                raise AssertionError(command)


def main() -> int:
    tests = [
        test_validation_pass_and_warning_generate_spectra,
        test_validation_failed_blocks_spectra,
        test_selection_modes_and_invalid_energy_filtering,
        test_linear_bins_derivatives_and_frequency_constant,
        test_log_bins_plots_attenuated_and_no_paper_ready_names,
        test_histogram_count_conservation_and_no_unexpected_flow,
        test_config_web_preset_and_pipeline_contract,
        test_photon_observer_full_validated_preset_contract,
        test_pipeline_schedules_spectra_after_validation_opacity_and_science_products,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
