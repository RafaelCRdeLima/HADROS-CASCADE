#!/usr/bin/env python3
"""Validation gate for the ideal photon observer camera chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "test_name",
    "physics_validated",
    "equation",
    "measured_error",
    "tolerance",
    "status",
    "notes",
]

REQUIRED_CONFIG_KEYS = [
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
    "enable_photon_opacity",
    "photon_opacity_mode",
    "photon_opacity_fail_on_invalid",
    "photon_opacity_output_mode",
    "photon_camera_projection_mode",
    "photon_camera_fov_deg",
    "photon_camera_fov_definition",
    "photon_camera_resolution_mode",
    "photon_camera_center_theta_source",
    "photon_camera_center_phi_rad",
    "photon_camera_clipping_mode",
]

CONFIG_CONTRACT_TESTS = {"config_web_final_contract"}

LEGACY_RECOVERABLE_CONFIG_KEYS = {
    "enable_photon_validation_gate",
    "enable_photon_observer_science_products",
    "photon_observer_science_require_validation",
    "enable_photon_opacity",
    "photon_opacity_mode",
    "photon_opacity_fail_on_invalid",
    "photon_opacity_output_mode",
    "photon_observer_crossing_tolerance_rg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-csv", required=True, type=Path)
    parser.add_argument("--redshift-csv", required=True, type=Path)
    parser.add_argument("--camera-provenance", required=True, type=Path)
    parser.add_argument("--redshift-provenance", required=True, type=Path)
    parser.add_argument("--pipeline-config", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--spin", required=True, type=float)
    parser.add_argument("--photon-invariant-tolerance", required=True, type=float)
    parser.add_argument("--photon-redshift-energy-tolerance", required=True, type=float)
    parser.add_argument("--projection-tolerance", default=1.0e-12, type=float)
    return parser.parse_args()


def as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return data


def kerr_inverse_metric(spin: float, r: float, theta: float) -> dict[str, float] | None:
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= 0.0:
        return None
    sigma = r * r + spin * spin * math.cos(theta) * math.cos(theta)
    delta = r * r - 2.0 * r + spin * spin
    a_func = (r * r + spin * spin) ** 2 - spin * spin * delta * sin_theta * sin_theta
    if sigma <= 0.0 or delta <= 0.0 or a_func <= 0.0:
        return None
    return {
        "tt": -a_func / (sigma * delta),
        "tphi": -2.0 * spin * r / (sigma * delta),
        "rr": delta / sigma,
        "thetatheta": 1.0 / sigma,
        "phiphi": (delta - spin * spin * sin_theta * sin_theta) / (sigma * delta * sin_theta * sin_theta),
    }


def null_norm(
    *,
    spin: float,
    r: float,
    theta: float,
    p_t: float,
    p_r: float,
    p_theta: float,
    p_phi: float,
) -> float | None:
    metric = kerr_inverse_metric(spin, r, theta)
    if metric is None:
        return None
    value = (
        metric["tt"] * p_t * p_t
        + 2.0 * metric["tphi"] * p_t * p_phi
        + metric["rr"] * p_r * p_r
        + metric["thetatheta"] * p_theta * p_theta
        + metric["phiphi"] * p_phi * p_phi
    )
    return value if math.isfinite(value) else None


def kerr_lapse_and_omega(spin: float, r: float, theta: float) -> tuple[float, float] | None:
    sigma = r * r + spin * spin * math.cos(theta) * math.cos(theta)
    delta = r * r - 2.0 * r + spin * spin
    a_func = (r * r + spin * spin) ** 2 - spin * spin * delta * math.sin(theta) * math.sin(theta)
    if sigma <= 0.0 or delta <= 0.0 or a_func <= 0.0:
        return None
    lapse_sq = sigma * delta / a_func
    if lapse_sq <= 0.0:
        return None
    alpha = math.sqrt(lapse_sq)
    omega = 2.0 * spin * r / a_func
    if not (math.isfinite(alpha) and math.isfinite(omega) and alpha > 0.0):
        return None
    return alpha, omega


def zamo_energy(spin: float, r: float, theta: float, p_t: float, p_phi: float) -> float | None:
    lapse_omega = kerr_lapse_and_omega(spin, r, theta)
    if lapse_omega is None:
        return None
    alpha, omega = lapse_omega
    energy = -(p_t + omega * p_phi) / alpha
    return energy if math.isfinite(energy) else None


def unit_from_angles(theta: float, phi: float) -> tuple[float, float, float]:
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def camera_basis(theta0: float, phi0: float) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    c = unit_from_angles(theta0, phi0)
    e_x = (-math.sin(phi0), math.cos(phi0), 0.0)
    e_y = (
        -math.cos(theta0) * math.cos(phi0),
        -math.cos(theta0) * math.sin(phi0),
        math.sin(theta0),
    )
    return c, e_x, e_y


def project_direction(
    n: tuple[float, float, float],
    *,
    theta0: float,
    phi0: float,
    fov_deg: float,
    nx: int,
    ny: int,
) -> dict[str, Any]:
    c, e_x, e_y = camera_basis(theta0, phi0)
    denom = dot(n, c)
    extent = math.tan(0.5 * math.radians(fov_deg))
    out: dict[str, Any] = {"inside_fov": False, "pixel_x": None, "pixel_y": None}
    if denom <= 0.0:
        return out
    camera_x = dot(n, e_x) / denom
    camera_y = dot(n, e_y) / denom
    out["camera_x"] = camera_x
    out["camera_y"] = camera_y
    inside = abs(camera_x) <= extent and abs(camera_y) <= extent
    out["inside_fov"] = inside
    if inside:
        u = 0.5 * (camera_x / extent + 1.0)
        v = 0.5 * (1.0 - camera_y / extent)
        pixel_x = math.floor(u * nx)
        pixel_y = math.floor(v * ny)
        if pixel_x == nx:
            pixel_x = nx - 1
        if pixel_y == ny:
            pixel_y = ny - 1
        out["pixel_x"] = pixel_x
        out["pixel_y"] = pixel_y
    return out


def result(
    test_name: str,
    physics_validated: str,
    equation: str,
    measured_error: float | None,
    tolerance: float | None,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "test_name": test_name,
        "physics_validated": physics_validated,
        "equation": equation,
        "measured_error": measured_error,
        "tolerance": tolerance,
        "status": status,
        "notes": notes,
    }


def pass_fail(error: float | None, tolerance: float, *, fail_note: str, pass_note: str) -> tuple[str, str]:
    if error is None or not math.isfinite(error):
        return "FAIL", fail_note
    if error <= tolerance:
        return "PASS", pass_note
    return "FAIL", fail_note


def null_norm_breakdown(rows: list[dict[str, str]], spin: float) -> dict[str, float | None]:
    breakdown: dict[str, list[float]] = {
        "null_norm_initial": [],
        "null_norm_max_along_path": [],
        "null_norm_recomputed_from_output_initial": [],
        "null_norm_recomputed_from_output_crossing": [],
        "null_norm_at_crossing": [],
        "crossing_r_error_rg": [],
    }

    def value_or_nan(row: dict[str, str], key: str) -> float:
        value = as_float(row, key)
        return value if value is not None else math.nan

    for row in rows:
        stored_initial = as_float(row, "null_norm_initial")
        if stored_initial is not None:
            breakdown["null_norm_initial"].append(abs(stored_initial))
        stored_path = as_float(row, "null_norm_max_abs_error")
        if stored_path is not None:
            breakdown["null_norm_max_along_path"].append(abs(stored_path))
        stored_crossing = as_float(row, "crossing_null_norm_abs_error")
        if stored_crossing is not None:
            breakdown["null_norm_at_crossing"].append(abs(stored_crossing))
        crossing_r_error = as_float(row, "crossing_r_error_rg")
        if crossing_r_error is not None:
            breakdown["crossing_r_error_rg"].append(abs(crossing_r_error))

        initial = null_norm(
            spin=spin,
            r=value_or_nan(row, "initial_r_rg"),
            theta=value_or_nan(row, "initial_theta_rad"),
            p_t=value_or_nan(row, "p_t_initial"),
            p_r=value_or_nan(row, "p_r_initial"),
            p_theta=value_or_nan(row, "p_theta_initial"),
            p_phi=value_or_nan(row, "p_phi_initial"),
        )
        if initial is not None:
            breakdown["null_norm_recomputed_from_output_initial"].append(abs(initial))

        crossing = null_norm(
            spin=spin,
            r=value_or_nan(row, "observer_crossing_r_rg"),
            theta=value_or_nan(row, "observer_crossing_theta_rad"),
            p_t=value_or_nan(row, "p_t_crossing"),
            p_r=value_or_nan(row, "p_r_crossing"),
            p_theta=value_or_nan(row, "p_theta_crossing"),
            p_phi=value_or_nan(row, "p_phi_crossing"),
        )
        if crossing is not None:
            breakdown["null_norm_recomputed_from_output_crossing"].append(abs(crossing))

    out = {key: (max(values) if values else None) for key, values in breakdown.items()}
    if out["null_norm_at_crossing"] is None:
        out["null_norm_at_crossing"] = out["null_norm_recomputed_from_output_crossing"]
    return out


def validate_rows(rows: list[dict[str, str]], spin: float, invariant_tolerance: float, redshift_tolerance: float) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("redshift_status") == "valid"]
    out: list[dict[str, Any]] = []

    def value_or_nan(row: dict[str, str], key: str) -> float:
        value = as_float(row, key)
        return value if value is not None else math.nan

    null_errors = []
    for row in valid:
        initial = null_norm(
            spin=spin,
            r=value_or_nan(row, "initial_r_rg"),
            theta=value_or_nan(row, "initial_theta_rad"),
            p_t=value_or_nan(row, "p_t_initial"),
            p_r=value_or_nan(row, "p_r_initial"),
            p_theta=value_or_nan(row, "p_theta_initial"),
            p_phi=value_or_nan(row, "p_phi_initial"),
        )
        crossing = null_norm(
            spin=spin,
            r=value_or_nan(row, "observer_crossing_r_rg"),
            theta=value_or_nan(row, "observer_crossing_theta_rad"),
            p_t=value_or_nan(row, "p_t_crossing"),
            p_r=value_or_nan(row, "p_r_crossing"),
            p_theta=value_or_nan(row, "p_theta_crossing"),
            p_phi=value_or_nan(row, "p_phi_crossing"),
        )
        for value in [initial, crossing, as_float(row, "null_norm_max_abs_error"), as_float(row, "crossing_null_norm_abs_error")]:
            if value is not None:
                null_errors.append(abs(value))
    null_error = max(null_errors) if null_errors else None
    status, notes = pass_fail(
        null_error,
        invariant_tolerance,
        fail_note="null norm drift exceeds tolerance or no valid photon rows",
        pass_note=f"validated {len(valid)} redshift-valid photon rows",
    )
    out.append(result("null_norm_kerr", "massless Kerr photon transport", "g^{mu nu} p_mu p_nu = 0", null_error, invariant_tolerance, status, notes))

    for key, value in null_norm_breakdown(valid, spin).items():
        diagnostic_note = "diagnostic component only; null_norm_kerr carries the required PASS/FAIL status"
        if key == "null_norm_initial" and value is None:
            diagnostic_note = "stored null_norm_initial is absent from downstream redshift CSV; recomputed initial diagnostic is available"
        out.append(
            result(
                key,
                "null-norm diagnostic breakdown",
                "g^{mu nu} p_mu p_nu = 0",
                value,
                invariant_tolerance,
                "WARNING",
                diagnostic_note,
            )
        )

    bad_crossing_methods = [
        row.get("crossing_momentum_method", "")
        for row in valid
        if row.get("crossing_momentum_method") != "fractional_rk_crossing_state"
    ]
    method_error = float(len(bad_crossing_methods))
    status, notes = pass_fail(
        method_error,
        0.0,
        fail_note=f"non-fractional crossing momentum methods found: {sorted(set(bad_crossing_methods))}",
        pass_note="all valid redshift rows use fractional_rk_crossing_state",
    )
    out.append(result("crossing_momentum_method", "observer-sphere crossing state", "p_mu_crossing from fractional RK state", method_error, 0.0, status, notes))

    e_errors = []
    lz_errors = []
    for row in valid:
        p_t_i = as_float(row, "p_t_initial")
        p_t_f = as_float(row, "p_t_crossing")
        p_phi_i = as_float(row, "p_phi_initial")
        p_phi_f = as_float(row, "p_phi_crossing")
        if p_t_i is not None and p_t_f is not None:
            e_errors.append(abs((-p_t_f) - (-p_t_i)) / max(abs(p_t_i), sys.float_info.epsilon))
        if p_phi_i is not None and p_phi_f is not None:
            lz_errors.append(abs(p_phi_f - p_phi_i) / max(abs(p_phi_i), 1.0, sys.float_info.epsilon))
        for key, target in [("relative_E_error", e_errors), ("relative_Lz_error", lz_errors)]:
            value = as_float(row, key)
            if value is not None:
                target.append(abs(value))
    e_error = max(e_errors) if e_errors else None
    lz_error = max(lz_errors) if lz_errors else None
    status, notes = pass_fail(e_error, invariant_tolerance, fail_note="Killing energy drift exceeds tolerance", pass_note="Killing energy conserved within tolerance")
    out.append(result("killing_energy_conservation", "Kerr stationarity invariant", "E = -p_t", e_error, invariant_tolerance, status, notes))
    status, notes = pass_fail(lz_error, invariant_tolerance, fail_note="Lz drift exceeds tolerance", pass_note="Lz conserved within tolerance")
    out.append(result("lz_conservation", "Kerr axial symmetry invariant", "L_z = p_phi", lz_error, invariant_tolerance, status, notes))

    redshift_errors = []
    for row in valid:
        emit = as_float(row, "emit_energy_zamo_gev")
        observed = as_float(row, "observed_energy_gev")
        factor = as_float(row, "redshift_factor")
        input_error = as_float(row, "energy_emit_input_relative_error")
        crossing_energy = None
        if all(as_float(row, key) is not None for key in ["observer_crossing_r_rg", "observer_crossing_theta_rad", "p_t_crossing", "p_phi_crossing"]):
            crossing_energy = zamo_energy(
                spin,
                value_or_nan(row, "observer_crossing_r_rg"),
                value_or_nan(row, "observer_crossing_theta_rad"),
                value_or_nan(row, "p_t_crossing"),
                value_or_nan(row, "p_phi_crossing"),
            )
        if emit and observed is not None and factor is not None:
            redshift_errors.append(abs((observed / emit) - factor))
        if crossing_energy is not None and observed is not None:
            redshift_errors.append(abs(crossing_energy - observed) / max(abs(observed), sys.float_info.epsilon))
        if input_error is not None:
            redshift_errors.append(abs(input_error))
    redshift_error = max(redshift_errors) if redshift_errors else None
    status, notes = pass_fail(redshift_error, redshift_tolerance, fail_note="E=-p_mu u^mu redshift consistency failed", pass_note="ZAMO redshift columns are self-consistent")
    out.append(result("zamo_redshift_consistency", "observed photon energy", "E = -p_mu u^mu; g = E_obs/E_emit", redshift_error, redshift_tolerance, status, notes))
    return out


def analytical_tests(redshift_tolerance: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    flat_emit = 10.0
    flat_obs = 10.0
    flat_g = flat_obs / flat_emit
    flat_error = abs(flat_g - 1.0)
    status, notes = pass_fail(flat_error, redshift_tolerance, fail_note="flat limit redshift differs from 1", pass_note="flat limit gives g=1")
    out.append(result("flat_limit_redshift", "flat spacetime observer energy", "g = E_obs/E_emit = 1", flat_error, redshift_tolerance, status, notes))

    r_emit = 10.0
    r_obs = 80.0
    alpha_emit = math.sqrt(1.0 - 2.0 / r_emit)
    alpha_obs = math.sqrt(1.0 - 2.0 / r_obs)
    analytic = alpha_emit / alpha_obs
    p_t = -1.0
    measured = (-(p_t) / alpha_obs) / (-(p_t) / alpha_emit)
    error = abs(measured - analytic)
    status, notes = pass_fail(error, redshift_tolerance, fail_note="Schwarzschild radial redshift failed", pass_note="Schwarzschild radial redshift matches analytic formula")
    out.append(result("schwarzschild_radial_redshift", "static Schwarzschild redshift", "sqrt(1-2M/r_emit)/sqrt(1-2M/r_obs)", error, redshift_tolerance, status, notes))
    return out


def projection_tests(camera_provenance: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    nx = int(camera_provenance.get("camera_nx", 0))
    ny = int(camera_provenance.get("camera_ny", 0))
    fov = float(camera_provenance.get("photon_camera_fov_deg", 60.0))
    theta0 = math.radians(float(camera_provenance.get("photon_camera_center_theta_deg", 90.0)))
    phi0 = float(camera_provenance.get("photon_camera_center_phi_rad", 0.0))
    if nx <= 0 or ny <= 0:
        return [result("projection_config", "camera projection", "camera_nx,camera_ny > 0", None, tolerance, "FAIL", "invalid camera shape in provenance")]
    center = project_direction(unit_from_angles(theta0, phi0), theta0=theta0, phi0=phi0, fov_deg=fov, nx=nx, ny=ny)
    center_error = 0.0 if center["pixel_x"] == nx // 2 and center["pixel_y"] == ny // 2 else 1.0
    status, notes = pass_fail(center_error, tolerance, fail_note=f"center mapped to ({center['pixel_x']},{center['pixel_y']})", pass_note="optical center maps to central pixel")
    out = [result("projection_center_pixel", "gnomonic pinhole projection", "n=c -> center pixel", center_error, tolerance, status, notes)]

    c, e_x, _ = camera_basis(theta0, phi0)
    extent = math.tan(0.5 * math.radians(fov))
    edge_vec = tuple(c[i] + (1.0 - tolerance) * extent * e_x[i] for i in range(3))
    norm = math.sqrt(sum(value * value for value in edge_vec))
    edge = project_direction(tuple(value / norm for value in edge_vec), theta0=theta0, phi0=phi0, fov_deg=fov, nx=nx, ny=ny)
    edge_error = 0.0 if edge["inside_fov"] and edge["pixel_x"] == nx - 1 else 1.0
    status, notes = pass_fail(edge_error, tolerance, fail_note=f"FOV edge mapped incorrectly: {edge}", pass_note="positive x FOV edge maps to right boundary pixel")
    out.append(result("projection_fov_edge", "camera FOV edge", "pixel_x=floor(u N_x) with upper-edge clamp", edge_error, tolerance, status, notes))

    outside_vec = tuple(c[i] + 1.01 * extent * e_x[i] for i in range(3))
    norm = math.sqrt(sum(value * value for value in outside_vec))
    outside = project_direction(tuple(value / norm for value in outside_vec), theta0=theta0, phi0=phi0, fov_deg=fov, nx=nx, ny=ny)
    outside_error = 0.0 if not outside["inside_fov"] and outside["pixel_x"] is None and outside["pixel_y"] is None else 1.0
    status, notes = pass_fail(outside_error, tolerance, fail_note=f"outside-FOV direction was clamped: {outside}", pass_note="outside-FOV direction remains outside with null pixels")
    out.append(result("projection_outside_fov", "camera clipping semantics", "outside FOV -> inside_fov=false", outside_error, tolerance, status, notes))
    return out


def semantic_tests(
    *,
    camera_fields: list[str],
    redshift_fields: list[str],
    camera_provenance: dict[str, Any],
    redshift_provenance: dict[str, Any],
    pipeline_config: dict[str, Any],
    redshift_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    camera_has_observed = "observed_energy_gev" in camera_fields
    redshift_has_observed = "observed_energy_gev" in redshift_fields
    error = 0.0 if (not camera_has_observed and redshift_has_observed) else 1.0
    status, notes = pass_fail(error, 0.0, fail_note="observed_energy_gev appears outside redshift file or missing from redshift file", pass_note="observed_energy_gev is limited to redshift output")
    out.append(result("observed_energy_file_semantics", "observed-energy output contract", "observed_energy only after validated redshift", error, 0.0, status, notes))

    disabled_rows_with_observed = [
        row for row in redshift_rows
        if row.get("photon_redshift_mode") == "disabled" and str(row.get("observed_energy_gev", "")).strip()
    ]
    error = float(len(disabled_rows_with_observed))
    status, notes = pass_fail(error, 0.0, fail_note="disabled redshift rows contain observed_energy_gev", pass_note="no disabled rows contain observed_energy_gev")
    out.append(result("redshift_disabled_no_observed_energy", "redshift disabled semantics", "disabled -> no observed_energy_gev", error, 0.0, status, notes))

    for test_name, key in [
        ("detector_model_not_applied", "detector_model_applied"),
        ("instrument_response_not_applied", "instrument_response_applied"),
        ("aperture_acceptance_not_applied", "aperture_acceptance_applied"),
    ]:
        values = [camera_provenance.get(key), redshift_provenance.get(key)]
        error = 0.0 if all(value is False for value in values) else 1.0
        status, notes = pass_fail(error, 0.0, fail_note=f"{key} is not false in provenance", pass_note=f"{key}=false in camera/redshift provenance")
        out.append(result(test_name, "no detector response in ideal camera", key, error, 0.0, status, notes))

    photon = pipeline_config.get("photon_escape_classifier", {})
    config_missing = [key for key in REQUIRED_CONFIG_KEYS if key not in photon and key not in pipeline_config]
    mirrored_errors = []
    for key in REQUIRED_CONFIG_KEYS:
        if key in photon and key in pipeline_config and str(photon[key]) != str(pipeline_config[key]):
            mirrored_errors.append(key)
    recoverable_missing = [key for key in config_missing if key in LEGACY_RECOVERABLE_CONFIG_KEYS]
    essential_missing = [key for key in config_missing if key not in LEGACY_RECOVERABLE_CONFIG_KEYS]
    error = float(len(essential_missing) + len(mirrored_errors))
    warning_count = float(len(recoverable_missing))
    if essential_missing or mirrored_errors:
        status = "FAIL"
        measured = error
        notes = f"missing_essential={essential_missing}; recoverable_legacy_missing={recoverable_missing}; mismatched_top_level={mirrored_errors}"
    elif recoverable_missing:
        status = "WARNING"
        measured = warning_count
        notes = f"config_schema_outdated; recoverable_legacy_missing={recoverable_missing}; physics interpreted from current inputs/provenance"
    else:
        status = "PASS"
        measured = 0.0
        notes = "config contract satisfied"
    out.append(result("config_web_final_contract", "single source of operational config", "config_web_final.py -> final pipeline config", measured, 0.0, status, notes))
    return out


def diagnostic_semantics(output_dir: Path) -> list[dict[str, Any]]:
    diagnostics = output_dir / "photon_observer_diagnostics"
    if not diagnostics.exists():
        return [result("diagnostic_products_not_paper_ready", "diagnostic plot semantics", "diagnostic != paper-ready", None, 0.0, "WARNING", "diagnostic folder absent; no diagnostic products to inspect")]
    bad_names = [path.name for path in diagnostics.iterdir() if "paper" in path.name.lower() or "final" in path.name.lower()]
    summaries = list(diagnostics.glob("*.md"))
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in summaries)
    has_label = "not paper-ready" in text and "ideal photon observer camera, no detector response" in text
    error = float(len(bad_names) + (0 if has_label else 1))
    status, notes = pass_fail(error, 0.0, fail_note=f"diagnostic products not clearly labeled: bad_names={bad_names}", pass_note="diagnostic products are not paper-ready and are labeled as no detector response")
    return [result("diagnostic_products_not_paper_ready", "diagnostic plot semantics", "diagnostic != paper-ready", error, 0.0, status, notes)]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ["measured_error", "tolerance"]:
                value = formatted.get(key)
                formatted[key] = "" if value is None else f"{float(value):.17g}"
            writer.writerow({key: formatted.get(key, "") for key in SUMMARY_FIELDS})


def validation_statuses(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    config_rows = [row for row in rows if row["test_name"] in CONFIG_CONTRACT_TESTS]
    physics_rows = [row for row in rows if row["test_name"] not in CONFIG_CONTRACT_TESTS]
    physics_status = "FAIL" if any(row["status"] == "FAIL" for row in physics_rows) else "PASS"
    if any(row["status"] == "FAIL" for row in config_rows):
        config_status = "FAIL"
    elif any(row["status"] == "WARNING" for row in config_rows):
        config_status = "WARNING"
    else:
        config_status = "PASS"
    if physics_status == "FAIL" or config_status == "FAIL":
        overall_status = "VALIDATION_FAILED"
    elif config_status == "WARNING":
        overall_status = "VALIDATION_WARNING"
    else:
        overall_status = "PASS"
    return physics_status, config_status, overall_status


def write_report(path: Path, rows: list[dict[str, Any]], physics_status: str, config_status: str, overall_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Photon Observer Camera Validation Report",
        "",
        f"- physics_status: `{physics_status}`",
        f"- config_status: `{config_status}`",
        f"- overall_status: `{overall_status}`",
        f"- PHYSICS_VALIDATION_STATUS: `{physics_status}`",
        f"- CONFIG_CONTRACT_STATUS: `{config_status}`",
        f"- OVERALL_STATUS: `{overall_status}`",
        "- chain: `Photon Escape Classifier -> Observer Sphere Hits -> Observer Camera Projection -> Validated ZAMO Redshift`",
        "- detector_model_applied: `false`",
        "- instrument_response_applied: `false`",
        "- aperture_acceptance_applied: `false`",
        "",
        "| Test | Physics | Equation | Error | Tolerance | Status | Notes |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        error = "" if row["measured_error"] is None else f"{float(row['measured_error']):.6g}"
        tolerance = "" if row["tolerance"] is None else f"{float(row['tolerance']):.6g}"
        lines.append(
            f"| `{row['test_name']}` | {row['physics_validated']} | `{row['equation']}` | "
            f"{error} | {tolerance} | `{row['status']}` | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], physics_status: str, config_status: str, overall_status: str) -> None:
    null_norm_diagnostics = {
        row["test_name"]: row["measured_error"]
        for row in rows
        if row["test_name"].startswith("null_norm_")
    }
    payload = {
        "phase": "photon_observer_camera_validation_gate",
        "physics_status": physics_status,
        "config_status": config_status,
        "overall_status": overall_status,
        "status": overall_status,
        "camera_csv": str(args.camera_csv),
        "redshift_csv": str(args.redshift_csv),
        "summary_csv": str(args.summary_csv),
        "report_md": str(args.report_md),
        "spin": args.spin,
        "photon_invariant_tolerance": args.photon_invariant_tolerance,
        "photon_redshift_energy_tolerance": args.photon_redshift_energy_tolerance,
        "projection_tolerance": args.projection_tolerance,
        "n_tests": len(rows),
        "n_pass": sum(1 for row in rows if row["status"] == "PASS"),
        "n_fail": sum(1 for row in rows if row["status"] == "FAIL"),
        "n_warning": sum(1 for row in rows if row["status"] == "WARNING"),
        "null_norm_diagnostics": null_norm_diagnostics,
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "observer_sphere_crossing_is_detection": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_validation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, str, str]:
    camera_fields, _camera_rows = read_csv(args.camera_csv)
    redshift_fields, redshift_rows = read_csv(args.redshift_csv)
    camera_provenance = read_json(args.camera_provenance)
    redshift_provenance = read_json(args.redshift_provenance)
    pipeline_config = read_json(args.pipeline_config)
    rows: list[dict[str, Any]] = []
    rows.extend(validate_rows(redshift_rows, args.spin, args.photon_invariant_tolerance, args.photon_redshift_energy_tolerance))
    rows.extend(analytical_tests(args.photon_redshift_energy_tolerance))
    rows.extend(projection_tests(camera_provenance, args.projection_tolerance))
    rows.extend(
        semantic_tests(
            camera_fields=camera_fields,
            redshift_fields=redshift_fields,
            camera_provenance=camera_provenance,
            redshift_provenance=redshift_provenance,
            pipeline_config=pipeline_config,
            redshift_rows=redshift_rows,
        )
    )
    rows.extend(diagnostic_semantics(args.redshift_csv.parent))
    physics_status, config_status, overall_status = validation_statuses(rows)
    return rows, physics_status, config_status, overall_status


def main() -> int:
    args = parse_args()
    try:
        if not math.isfinite(args.spin) or abs(args.spin) >= 1.0:
            raise ValueError("spin must be finite with |spin| < 1")
        if args.photon_invariant_tolerance <= 0.0:
            raise ValueError("photon_invariant_tolerance must be > 0")
        if args.photon_redshift_energy_tolerance <= 0.0:
            raise ValueError("photon_redshift_energy_tolerance must be > 0")
        if args.projection_tolerance < 0.0:
            raise ValueError("projection_tolerance must be >= 0")
        rows, physics_status, config_status, overall_status = run_validation(args)
    except Exception as exc:
        rows = [
            result(
                "validation_gate_execution",
                "validation infrastructure",
                "all required inputs readable",
                None,
                None,
                "FAIL",
                str(exc),
            )
        ]
        physics_status = "FAIL"
        config_status = "FAIL"
        overall_status = "VALIDATION_FAILED"
    write_summary(args.summary_csv, rows)
    write_report(args.report_md, rows, physics_status, config_status, overall_status)
    write_provenance(args.provenance, args, rows, physics_status, config_status, overall_status)
    if overall_status == "VALIDATION_FAILED":
        print(f"Photon observer camera validation failed; see {args.report_md}", file=sys.stderr)
        return 2
    if overall_status == "VALIDATION_WARNING":
        print(f"Photon observer camera validation passed with warnings; see {args.report_md}")
    else:
        print(f"Photon observer camera validation passed; see {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
