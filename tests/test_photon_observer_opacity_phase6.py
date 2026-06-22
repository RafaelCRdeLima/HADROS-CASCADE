#!/usr/bin/env python3
"""Lightweight tests for Photon Observer Phase 6 opacity infrastructure."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/build_photon_observer_opacity.py"
SCIENCE_SCRIPT = ROOT / "scripts/science/build_photon_observer_science_products.py"
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


def write_redshift(path: Path) -> None:
    fields = [
        "event_id",
        "photon_path_id",
        "particle_id",
        "pixel_x",
        "pixel_y",
        "inside_fov",
        "redshift_status",
        "input_energy_gev",
        "observed_energy_gev",
        "redshift_factor",
        "total_path_length_rg",
    ]
    write_csv(
        path,
        fields,
        [
            {
                "event_id": 1,
                "photon_path_id": 101,
                "particle_id": 1,
                "pixel_x": 0,
                "pixel_y": 0,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 10.0,
                "observed_energy_gev": 8.0,
                "redshift_factor": 0.8,
                "total_path_length_rg": 10.0,
            },
            {
                "event_id": 1,
                "photon_path_id": 102,
                "particle_id": 2,
                "pixel_x": 1,
                "pixel_y": 0,
                "inside_fov": "true",
                "redshift_status": "valid",
                "input_energy_gev": 5.0,
                "observed_energy_gev": 4.0,
                "redshift_factor": 0.8,
                "total_path_length_rg": 20.0,
            },
            {
                "event_id": 1,
                "photon_path_id": 103,
                "particle_id": 3,
                "pixel_x": 1,
                "pixel_y": 1,
                "inside_fov": "false",
                "redshift_status": "valid",
                "input_energy_gev": 3.0,
                "observed_energy_gev": 2.0,
                "redshift_factor": 0.67,
                "total_path_length_rg": 30.0,
            },
        ],
    )


def write_validation(tmp: Path) -> tuple[Path, Path]:
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
                "measured_error": "0",
                "tolerance": "1e-6",
                "status": "PASS",
                "notes": "",
            }
        ],
    )
    provenance.write_text(
        json.dumps({"physics_status": "PASS", "config_status": "PASS", "overall_status": "PASS"}) + "\n",
        encoding="utf-8",
    )
    return summary, provenance


def write_path_summary(path: Path) -> None:
    write_csv(
        path,
        ["photon_path_id", "event_id", "particle_id", "total_path_length_rg", "truncated", "truncation_status"],
        [
            {"photon_path_id": 101, "event_id": 1, "particle_id": 1, "total_path_length_rg": 10.0, "truncated": "false", "truncation_status": "complete"},
            {"photon_path_id": 102, "event_id": 1, "particle_id": 2, "total_path_length_rg": 20.0, "truncated": "false", "truncation_status": "complete"},
            {"photon_path_id": 103, "event_id": 1, "particle_id": 3, "total_path_length_rg": 30.0, "truncated": "false", "truncation_status": "complete"},
        ],
    )


def write_medium_compressed(
    tmp: Path,
    *,
    include_truncated: bool = True,
    e_gamma_second: float = 2.0,
    medium_backend: str = "analytic_torus",
) -> tuple[Path, Path, Path]:
    medium = tmp / "photon_observer_medium_compressed_segments.jsonl"
    summary = tmp / "photon_observer_medium_compressed_summary.csv"
    provenance = tmp / "photon_observer_medium_compressed_provenance.json"
    rows = [
        {
            "event_id": 1,
            "particle_id": 1,
            "photon_path_id": 101,
            "segment_index": 0,
            "r_mid_rg": 10.0,
            "theta_mid_rad": 1.0,
            "phi_mid_rad": 0.0,
            "p_t_mid": -1.0,
            "p_r_mid": 0.0,
            "p_theta_mid": 0.0,
            "p_phi_mid": 0.0,
            "dl_segment_rg": 2.0,
            "rho_g_cm3": 3.0,
            "temperature_mev": 1.0,
            "Ye": 0.5,
            "u_fluid_t": 1.0,
            "u_fluid_r": 0.0,
            "u_fluid_theta": 0.0,
            "u_fluid_phi": 0.0,
            "E_gamma_fluid_gev": 1.0,
            "medium_status": "valid",
            "compression_complete": True,
        },
        {
            "event_id": 1,
            "particle_id": 2,
            "photon_path_id": 102,
            "segment_index": 0,
            "r_mid_rg": 10.0,
            "theta_mid_rad": 1.0,
            "phi_mid_rad": 0.0,
            "p_t_mid": -1.0,
            "p_r_mid": 0.0,
            "p_theta_mid": 0.0,
            "p_phi_mid": 0.0,
            "dl_segment_rg": 5.0,
            "rho_g_cm3": 3.0,
            "temperature_mev": 1.0,
            "Ye": 0.5,
            "u_fluid_t": 1.0,
            "u_fluid_r": 0.0,
            "u_fluid_theta": 0.0,
            "u_fluid_phi": 0.0,
            "E_gamma_fluid_gev": e_gamma_second,
            "medium_status": "valid",
            "compression_complete": True,
        },
    ]
    if include_truncated:
        rows.append(
            {
                "event_id": 1,
                "particle_id": 3,
                "photon_path_id": 103,
                "segment_index": 0,
                "r_mid_rg": 10.0,
                "theta_mid_rad": 1.0,
                "phi_mid_rad": 0.0,
                "p_t_mid": -1.0,
                "p_r_mid": 0.0,
                "p_theta_mid": 0.0,
                "p_phi_mid": 0.0,
                "dl_segment_rg": 7.0,
                "rho_g_cm3": 3.0,
                "temperature_mev": 1.0,
                "Ye": 0.5,
                "u_fluid_t": 1.0,
                "u_fluid_r": 0.0,
                "u_fluid_theta": 0.0,
                "u_fluid_phi": 0.0,
                "E_gamma_fluid_gev": 1.0,
                "medium_status": "valid",
                "compression_complete": False,
            }
        )
    medium.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_csv(
        summary,
        [
            "photon_medium_model",
            "medium_input_path_mode",
            "truncated_path_policy",
            "n_medium_queries",
            "n_medium_valid",
            "n_medium_invalid",
            "n_compressed_paths_total",
            "n_compressed_paths_used",
            "n_compressed_paths_excluded_truncated",
        ],
        [
            {
                "photon_medium_model": medium_backend,
                "medium_input_path_mode": "compressed_complete_paths",
                "truncated_path_policy": "exclude" if include_truncated else "fail",
                "n_medium_queries": len(rows),
                "n_medium_valid": len(rows),
                "n_medium_invalid": 0,
                "n_compressed_paths_total": 3 if include_truncated else 2,
                "n_compressed_paths_used": 2,
                "n_compressed_paths_excluded_truncated": 1 if include_truncated else 0,
            }
        ],
    )
    provenance.write_text(
        json.dumps(
            {
                "medium_lookup_applied": True,
                "medium_backend": medium_backend,
                "photon_medium_model": medium_backend,
                "medium_input_path_mode": "compressed_complete_paths",
                "truncated_path_policy": "exclude" if include_truncated else "fail",
                "no_opacity_applied": True,
                "opacity_applied": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return medium, summary, provenance


def run_opacity(
    tmp: Path,
    mode: str = "vacuum",
    fail_on_invalid: str = "true",
    alpha: float = 0.0,
    write_paths: bool = True,
    kappa: float = 0.0,
    energy_exponent: float = 0.0,
    reference_energy: float = 1.0,
    truncated_policy: str = "fail",
    include_medium: bool = False,
    include_truncated_medium: bool = True,
    medium_backend: str = "analytic_torus",
) -> subprocess.CompletedProcess[str]:
    redshift = tmp / "photon_observer_camera_redshift.csv"
    write_redshift(redshift)
    path_summary = tmp / "photon_observer_geodesic_path_samples_per_photon_summary.csv"
    if write_paths:
        write_path_summary(path_summary)
    command = [
            sys.executable,
            str(SCRIPT),
            "--redshift-csv",
            str(redshift),
            "--output-csv",
            str(tmp / "photon_observer_camera_attenuated.csv"),
            "--summary-csv",
            str(tmp / "photon_observer_opacity_summary.csv"),
            "--provenance",
            str(tmp / "photon_observer_opacity_provenance.json"),
            "--path-summary-csv",
            str(path_summary),
            "--photon-opacity-mode",
            mode,
            "--photon-constant-alpha-per-rg",
            str(alpha),
            "--photon-density-gray-kappa-per-rg-per-gcm3",
            str(kappa),
            "--photon-density-gray-energy-exponent",
            str(energy_exponent),
            "--photon-density-gray-reference-energy-gev",
            str(reference_energy),
            "--photon-opacity-truncated-path-policy",
            truncated_policy,
            "--photon-opacity-fail-on-invalid",
            fail_on_invalid,
            "--photon-opacity-output-mode",
            "separate_file",
        ]
    if include_medium:
        medium, medium_summary, medium_provenance = write_medium_compressed(
            tmp,
            include_truncated=include_truncated_medium,
            medium_backend=medium_backend,
        )
        command.extend(
            [
                "--medium-compressed-jsonl",
                str(medium),
                "--medium-compressed-summary-csv",
                str(medium_summary),
                "--medium-compressed-provenance",
                str(medium_provenance),
            ]
        )
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_disabled_does_not_create_attenuated_file() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_disabled_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(tmp, mode="disabled")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        if (tmp / "photon_observer_camera_attenuated.csv").exists():
            raise AssertionError("disabled mode created attenuated camera")
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["created_outputs"] is not False or provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)


def test_vacuum_creates_identity_attenuation() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_vacuum_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(tmp, mode="vacuum")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = read_csv(tmp / "photon_observer_camera_attenuated.csv")
        for row in rows:
            if row["photon_opacity_status"] != "valid_vacuum":
                raise AssertionError(row)
            if float(row["photon_path_optical_depth"]) != 0.0:
                raise AssertionError(row)
            if float(row["photon_survival_probability"]) != 1.0:
                raise AssertionError(row)
            if float(row["attenuated_observed_energy_gev"]) != float(row["observed_energy_gev"]):
                raise AssertionError(row)
        summary = read_csv(tmp / "photon_observer_opacity_summary.csv")[0]
        if int(summary["n_opacity_valid"]) != 3 or int(summary["n_opacity_invalid"]) != 0:
            raise AssertionError(summary)
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["photon_opacity_model"] != "vacuum_no_absorption":
            raise AssertionError(provenance)
        if provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)
        if provenance["path_sampling_used"] is not False:
            raise AssertionError(provenance)


def test_constant_alpha_zero_matches_vacuum_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_alpha0_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(tmp, mode="constant_alpha_path", alpha=0.0)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = read_csv(tmp / "photon_observer_camera_attenuated.csv")
        for row in rows:
            if row["photon_opacity_status"] != "valid_constant_alpha_path":
                raise AssertionError(row)
            if float(row["tau_path"]) != 0.0 or float(row["photon_path_optical_depth"]) != 0.0:
                raise AssertionError(row)
            if float(row["survival_probability"]) != 1.0 or float(row["photon_survival_probability"]) != 1.0:
                raise AssertionError(row)
            if float(row["attenuated_observed_energy_gev"]) != float(row["observed_energy_gev"]):
                raise AssertionError(row)
            if row["opacity_integration_method"] != "constant_alpha_path":
                raise AssertionError(row)
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)
        if provenance["path_sampling_used"] is not False or provenance["path_sampling_audit_available"] is not True:
            raise AssertionError(provenance)


def test_constant_alpha_path_uses_path_length() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_alpha_") as tmp_name:
        tmp = Path(tmp_name)
        alpha = 0.05
        completed = run_opacity(tmp, mode="constant_alpha_path", alpha=alpha)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = read_csv(tmp / "photon_observer_camera_attenuated.csv")
        lengths = {"101": 10.0, "102": 20.0, "103": 30.0}
        for row in rows:
            expected_tau = alpha * lengths[row["photon_path_id"]]
            expected_survival = math.exp(-expected_tau)
            if abs(float(row["tau_path"]) - expected_tau) > 1e-14:
                raise AssertionError(row)
            if abs(float(row["survival_probability"]) - expected_survival) > 1e-14:
                raise AssertionError(row)
            expected_attenuated = float(row["observed_energy_gev"]) * expected_survival
            if abs(float(row["attenuated_observed_energy_gev"]) - expected_attenuated) > 1e-14:
                raise AssertionError(row)
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["photon_absorption_applied"] is not True:
            raise AssertionError(provenance)
        if provenance["alpha_const_per_rg"] != alpha:
            raise AssertionError(provenance)


def test_negative_constant_alpha_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_bad_alpha_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(tmp, mode="constant_alpha_path", alpha=-0.1)
        if completed.returncode == 0:
            raise AssertionError("negative alpha was accepted")


def test_constant_alpha_requires_path_length() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_missing_length_") as tmp_name:
        tmp = Path(tmp_name)
        redshift = tmp / "photon_observer_camera_redshift.csv"
        write_csv(
            redshift,
            ["event_id", "photon_path_id", "particle_id", "redshift_status", "observed_energy_gev"],
            [{"event_id": 1, "photon_path_id": 1, "particle_id": 1, "redshift_status": "valid", "observed_energy_gev": 1.0}],
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--redshift-csv",
                str(redshift),
                "--output-csv",
                str(tmp / "photon_observer_camera_attenuated.csv"),
                "--summary-csv",
                str(tmp / "photon_observer_opacity_summary.csv"),
                "--provenance",
                str(tmp / "photon_observer_opacity_provenance.json"),
                "--photon-opacity-mode",
                "constant_alpha_path",
                "--photon-constant-alpha-per-rg",
                "0.1",
                "--photon-density-gray-kappa-per-rg-per-gcm3",
                "0.0",
                "--photon-density-gray-energy-exponent",
                "0.0",
                "--photon-density-gray-reference-energy-gev",
                "1.0",
                "--photon-opacity-truncated-path-policy",
                "fail",
                "--photon-opacity-fail-on-invalid",
                "true",
                "--photon-opacity-output-mode",
                "separate_file",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            raise AssertionError("constant_alpha_path accepted missing total_path_length_rg")


def test_unsupported_opacity_mode_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_bad_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(tmp, mode="tabulated_gray")
        if completed.returncode == 0:
            raise AssertionError("tabulated_gray was accepted by initial Phase 6")


def test_density_gray_toy_kappa_zero_matches_vacuum_for_complete_subset() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_density0_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(
            tmp,
            mode="density_gray_toy",
            kappa=0.0,
            truncated_policy="exclude",
            include_medium=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = read_csv(tmp / "photon_observer_camera_attenuated.csv")
        valid = [row for row in rows if row["photon_opacity_status"] == "valid_density_gray_toy"]
        if len(valid) != 2:
            raise AssertionError(rows)
        for row in valid:
            if float(row["tau_path"]) != 0.0 or float(row["survival_probability"]) != 1.0:
                raise AssertionError(row)
            if float(row["attenuated_observed_energy_gev"]) != float(row["observed_energy_gev"]):
                raise AssertionError(row)
            if row["path_subset_status"] != "complete_path_subset_only":
                raise AssertionError(row)
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["opacity_is_toy_model"] is not True or provenance["photon_absorption_applied"] is not False:
            raise AssertionError(provenance)


def test_density_gray_toy_constant_rho_tau() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_density_") as tmp_name:
        tmp = Path(tmp_name)
        kappa = 0.1
        completed = run_opacity(
            tmp,
            mode="density_gray_toy",
            kappa=kappa,
            truncated_policy="exclude",
            include_medium=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = {row["photon_path_id"]: row for row in read_csv(tmp / "photon_observer_camera_attenuated.csv")}
        expected = {"101": kappa * 3.0 * 2.0, "102": kappa * 3.0 * 5.0}
        for path_id, tau in expected.items():
            row = rows[path_id]
            if abs(float(row["tau_path"]) - tau) > 1.0e-14:
                raise AssertionError(row)
            if int(row["n_medium_segments_used"]) != 1:
                raise AssertionError(row)
        if rows["103"]["photon_opacity_status"] != "excluded_or_missing_complete_medium_path":
            raise AssertionError(rows["103"])
        summary = read_csv(tmp / "photon_observer_opacity_summary.csv")[0]
        if int(summary["n_paths_used"]) != 2 or int(summary["n_paths_excluded_truncated"]) != 1:
            raise AssertionError(summary)


def test_density_gray_toy_model_label_tracks_medium_backend() -> None:
    cases = {
        "analytic_torus": "density_gray_toy_analytic_torus",
        "uribe_radial_scalar": "density_gray_toy_uribe_radial_scalar",
    }
    for backend, expected_model in cases.items():
        with tempfile.TemporaryDirectory(prefix=f"hadros_photon_opacity_label_{backend}_") as tmp_name:
            tmp = Path(tmp_name)
            completed = run_opacity(
                tmp,
                mode="density_gray_toy",
                kappa=0.1,
                truncated_policy="exclude",
                include_medium=True,
                medium_backend=backend,
            )
            if completed.returncode != 0:
                raise AssertionError(completed.stderr)
            rows = read_csv(tmp / "photon_observer_camera_attenuated.csv")
            summary = read_csv(tmp / "photon_observer_opacity_summary.csv")[0]
            provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        valid = [row for row in rows if row["photon_opacity_status"] == "valid_density_gray_toy"]
        if not valid:
            raise AssertionError(rows)
        if any(row["photon_opacity_model"] != expected_model for row in valid):
            raise AssertionError(valid)
        if summary["photon_opacity_model"] != expected_model:
            raise AssertionError(summary)
        if provenance["photon_opacity_model"] != expected_model:
            raise AssertionError(provenance)
        if provenance["medium_backend"] != backend:
            raise AssertionError(provenance)


def test_density_gray_toy_energy_exponent_uses_fluid_energy() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_density_e_") as tmp_name:
        tmp = Path(tmp_name)
        kappa = 0.1
        completed = run_opacity(
            tmp,
            mode="density_gray_toy",
            kappa=kappa,
            energy_exponent=1.0,
            reference_energy=1.0,
            truncated_policy="exclude",
            include_medium=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        rows = {row["photon_path_id"]: row for row in read_csv(tmp / "photon_observer_camera_attenuated.csv")}
        if abs(float(rows["101"]["tau_path"]) - 0.6) > 1.0e-14:
            raise AssertionError(rows["101"])
        if abs(float(rows["102"]["tau_path"]) - 3.0) > 1.0e-14:
            raise AssertionError(rows["102"])


def test_density_gray_toy_truncated_fail_policy_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_density_fail_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(
            tmp,
            mode="density_gray_toy",
            kappa=0.1,
            truncated_policy="fail",
            include_medium=True,
        )
        if completed.returncode == 0:
            raise AssertionError("density_gray_toy fail policy accepted truncated paths")


def test_density_gray_toy_requires_complete_paths() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_density_none_") as tmp_name:
        tmp = Path(tmp_name)
        completed = run_opacity(
            tmp,
            mode="density_gray_toy",
            kappa=0.1,
            truncated_policy="exclude",
            include_medium=True,
            include_truncated_medium=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        provenance = json.loads((tmp / "photon_observer_opacity_provenance.json").read_text(encoding="utf-8"))
        if provenance["medium_lookup_used"] is not True or provenance["not_physical_radiative_transfer"] is not True:
            raise AssertionError(provenance)


def test_science_products_write_separate_attenuated_products() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_science_") as tmp_name:
        tmp = Path(tmp_name)
        opacity = run_opacity(tmp, mode="vacuum")
        if opacity.returncode != 0:
            raise AssertionError(opacity.stderr)
        summary, provenance = write_validation(tmp)
        completed = subprocess.run(
            [
                sys.executable,
                str(SCIENCE_SCRIPT),
                "--redshift-csv",
                str(tmp / "photon_observer_camera_redshift.csv"),
                "--attenuated-csv",
                str(tmp / "photon_observer_camera_attenuated.csv"),
                "--validation-summary-csv",
                str(summary),
                "--validation-provenance",
                str(provenance),
                "--output-dir",
                str(tmp / "science"),
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
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        ideal = read_csv(tmp / "science" / "photon_observer_observed_energy_map.csv")
        attenuated = read_csv(tmp / "science" / "photon_observer_attenuated_energy_map.csv")
        if not (tmp / "science" / "photon_observer_survival_map.csv").exists():
            raise AssertionError("missing survival map")
        if not (tmp / "science" / "photon_observer_attenuated_spectrum.csv").exists():
            raise AssertionError("missing attenuated spectrum")
        ideal_by_pixel = {(row["pixel_x"], row["pixel_y"]): float(row["sum_observed_energy_gev"]) for row in ideal}
        att_by_pixel = {
            (row["pixel_x"], row["pixel_y"]): float(row["sum_attenuated_observed_energy_gev"])
            for row in attenuated
        }
        if ideal_by_pixel != att_by_pixel:
            raise AssertionError((ideal_by_pixel, att_by_pixel))
        science_prov = json.loads((tmp / "science" / "photon_observer_science_provenance.json").read_text(encoding="utf-8"))
        if science_prov["attenuated_products"]["attenuated_products_available"] is not True:
            raise AssertionError(science_prov)
        if science_prov["attenuated_products"]["photon_absorption_applied"] is not False:
            raise AssertionError(science_prov)


def test_config_web_and_pipeline_schedule_opacity() -> None:
    values = config_web_final.defaults()
    photon = values["photon_escape_classifier"]
    opacity_keys = [
        "enable_photon_opacity",
        "photon_opacity_mode",
        "photon_constant_alpha_per_rg",
        "photon_density_gray_kappa_per_rg_per_gcm3",
        "photon_density_gray_energy_exponent",
        "photon_density_gray_reference_energy_gev",
        "photon_opacity_truncated_path_policy",
        "photon_opacity_fail_on_invalid",
        "photon_opacity_output_mode",
    ]
    for key in opacity_keys:
        if key not in photon:
            raise AssertionError(key)
    pipeline = config_web_final.final_pipeline_config(values)
    for key in opacity_keys:
        if key not in pipeline or key not in pipeline["photon_escape_classifier"]:
            raise AssertionError(key)

    with tempfile.TemporaryDirectory(prefix="hadros_photon_opacity_pipeline_") as tmp_name:
        config = config_web_final.final_pipeline_config(config_web_final.defaults())
        config.update(
            {
                "output_dir": str(Path(tmp_name) / "run"),
                "physics_mode": "uhe_cascade",
                "enable_photon_observer_camera": True,
                "photon_observer_mode": "observer_camera_projection",
                "photon_redshift_mode": "validated_zamo",
                "enable_photon_validation_gate": True,
                "enable_photon_opacity": True,
                "photon_opacity_mode": "vacuum",
                "photon_constant_alpha_per_rg": 0.0,
                "photon_density_gray_kappa_per_rg_per_gcm3": 0.0,
                "photon_density_gray_energy_exponent": 0.0,
                "photon_density_gray_reference_energy_gev": 1.0,
                "photon_opacity_truncated_path_policy": "fail",
                "photon_opacity_fail_on_invalid": True,
                "photon_opacity_output_mode": "separate_file",
                "enable_photon_observer_science_products": True,
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
                "enable_photon_opacity": True,
                "photon_opacity_mode": "vacuum",
                "photon_constant_alpha_per_rg": 0.0,
                "photon_density_gray_kappa_per_rg_per_gcm3": 0.0,
                "photon_density_gray_energy_exponent": 0.0,
                "photon_density_gray_reference_energy_gev": 1.0,
                "photon_opacity_truncated_path_policy": "fail",
                "photon_opacity_fail_on_invalid": True,
                "photon_opacity_output_mode": "separate_file",
                "enable_photon_observer_science_products": True,
            }
        )
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        if "photon_observer_opacity_vacuum" not in names:
            raise AssertionError(names)
        if names.index("photon_observer_opacity_vacuum") <= names.index("photon_observer_camera_validation_gate"):
            raise AssertionError(names)
        if names.index("photon_observer_science_products") <= names.index("photon_observer_opacity_vacuum"):
            raise AssertionError(names)
        science_command = " ".join(next(step.command for step in steps if step.name == "photon_observer_science_products"))
        if "--attenuated-csv" not in science_command:
            raise AssertionError(science_command)

        config["photon_opacity_mode"] = "constant_alpha_path"
        config["photon_constant_alpha_per_rg"] = 0.01
        config["photon_escape_classifier"]["photon_opacity_mode"] = "constant_alpha_path"
        config["photon_escape_classifier"]["photon_constant_alpha_per_rg"] = 0.01
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        if "photon_observer_opacity_constant_alpha_path" not in names:
            raise AssertionError(names)
        opacity_command = " ".join(next(step.command for step in steps if step.name == "photon_observer_opacity_constant_alpha_path"))
        if "--path-summary-csv" not in opacity_command or "--photon-constant-alpha-per-rg 0.01" not in opacity_command:
            raise AssertionError(opacity_command)
        if names.index("photon_observer_opacity_constant_alpha_path") <= names.index("photon_observer_camera_validation_gate"):
            raise AssertionError(names)

        config.update(
            {
                "photon_opacity_mode": "density_gray_toy",
                "photon_density_gray_kappa_per_rg_per_gcm3": 0.01,
                "photon_opacity_truncated_path_policy": "exclude",
                "enable_photon_path_sampling": True,
                "enable_photon_path_compression": True,
                "photon_path_compression_mode": "fixed_stride_segments",
                "photon_medium_model": "analytic_torus",
                "photon_medium_input_path_mode": "compressed_complete_paths",
                "photon_medium_truncated_path_policy": "exclude",
            }
        )
        config["photon_escape_classifier"].update(
            {
                "photon_opacity_mode": "density_gray_toy",
                "photon_density_gray_kappa_per_rg_per_gcm3": 0.01,
                "photon_opacity_truncated_path_policy": "exclude",
                "enable_photon_path_sampling": True,
                "enable_photon_path_compression": True,
                "photon_path_compression_mode": "fixed_stride_segments",
                "photon_medium_model": "analytic_torus",
                "photon_medium_input_path_mode": "compressed_complete_paths",
                "photon_medium_truncated_path_policy": "exclude",
            }
        )
        steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
        names = [step.name for step in steps]
        if "photon_observer_opacity_density_gray_toy" not in names:
            raise AssertionError(names)
        density_command = " ".join(next(step.command for step in steps if step.name == "photon_observer_opacity_density_gray_toy"))
        for required in [
            "--medium-compressed-jsonl",
            "--photon-density-gray-kappa-per-rg-per-gcm3 0.01",
            "--photon-opacity-truncated-path-policy exclude",
        ]:
            if required not in density_command:
                raise AssertionError(density_command)


if __name__ == "__main__":
    test_disabled_does_not_create_attenuated_file()
    test_vacuum_creates_identity_attenuation()
    test_constant_alpha_zero_matches_vacuum_identity()
    test_constant_alpha_path_uses_path_length()
    test_negative_constant_alpha_fails()
    test_constant_alpha_requires_path_length()
    test_unsupported_opacity_mode_fails()
    test_density_gray_toy_kappa_zero_matches_vacuum_for_complete_subset()
    test_density_gray_toy_constant_rho_tau()
    test_density_gray_toy_model_label_tracks_medium_backend()
    test_density_gray_toy_energy_exponent_uses_fluid_energy()
    test_density_gray_toy_truncated_fail_policy_fails()
    test_density_gray_toy_requires_complete_paths()
    test_science_products_write_separate_attenuated_products()
    test_config_web_and_pipeline_schedule_opacity()
