#!/usr/bin/env python3
"""Add validated ZAMO redshift to projected photon observer-camera rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REDSHIFT_FIELDS = [
    "emit_energy_zamo_gev",
    "observed_energy_gev",
    "redshift_factor",
    "redshift_status",
    "energy_emit_input_relative_error",
    "photon_redshift_mode",
    "photon_redshift_emitter_frame",
    "photon_redshift_observer_frame",
]

SUMMARY_FIELDS = [
    "photon_redshift_mode",
    "n_input_rows",
    "n_redshift_valid",
    "n_redshift_invalid",
    "total_input_energy_gev",
    "total_observed_energy_gev",
    "mean_redshift_factor",
    "min_redshift_factor",
    "max_redshift_factor",
]

REQUIRED_MOMENTUM_FIELDS = [
    "initial_r_rg",
    "initial_theta_rad",
    "initial_phi_rad",
    "p_t_initial",
    "p_r_initial",
    "p_theta_initial",
    "p_phi_initial",
    "observer_crossing_r_rg",
    "observer_crossing_theta_rad",
    "observer_crossing_phi_rad",
    "p_t_crossing",
    "p_r_crossing",
    "p_theta_crossing",
    "p_phi_crossing",
    "input_energy_gev",
]

INVARIANT_FIELDS = [
    "null_norm_max_abs_error",
    "relative_E_error",
    "relative_Lz_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--spin", required=True, type=float)
    parser.add_argument("--photon-redshift-mode", required=True)
    parser.add_argument("--photon-redshift-emitter-frame", required=True)
    parser.add_argument("--photon-redshift-observer-frame", required=True)
    parser.add_argument("--photon-redshift-energy-tolerance", required=True, type=float)
    parser.add_argument("--photon-redshift-fail-on-invalid", required=True)
    parser.add_argument("--photon-invariant-tolerance", required=True, type=float)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_config(args: argparse.Namespace) -> None:
    if args.photon_redshift_mode not in {"disabled", "validated_zamo"}:
        raise ValueError("photon_redshift_mode must be disabled or validated_zamo")
    if args.photon_redshift_emitter_frame != "ZAMO":
        raise ValueError("photon_redshift_emitter_frame must be ZAMO")
    if args.photon_redshift_observer_frame != "ZAMO":
        raise ValueError("photon_redshift_observer_frame must be ZAMO")
    if not math.isfinite(args.spin) or abs(args.spin) >= 1.0:
        raise ValueError("spin must be finite with |spin| < 1")
    if not math.isfinite(args.photon_redshift_energy_tolerance) or args.photon_redshift_energy_tolerance <= 0.0:
        raise ValueError("photon_redshift_energy_tolerance must be > 0")
    if not math.isfinite(args.photon_invariant_tolerance) or args.photon_invariant_tolerance <= 0.0:
        raise ValueError("photon_invariant_tolerance must be > 0")
    parse_bool(args.photon_redshift_fail_on_invalid)


def read_camera_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 photon observer-camera file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Phase 3 photon observer-camera file has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def row_bool(row: dict[str, Any], key: str) -> bool:
    try:
        return parse_bool(row.get(key, False))
    except ValueError:
        return False


def kerr_lapse_and_omega(spin: float, r: float, theta: float) -> tuple[float, float] | None:
    sigma = r * r + spin * spin * math.cos(theta) * math.cos(theta)
    delta = r * r - 2.0 * r + spin * spin
    a_func = (r * r + spin * spin) * (r * r + spin * spin) - spin * spin * delta * math.sin(theta) * math.sin(theta)
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


def zamo_energy_gev(spin: float, r: float, theta: float, p_t: float, p_phi: float) -> float | None:
    lapse_omega = kerr_lapse_and_omega(spin, r, theta)
    if lapse_omega is None:
        return None
    alpha, omega = lapse_omega
    energy = -(p_t + omega * p_phi) / alpha
    return energy if math.isfinite(energy) else None


def missing_required_fields(row: dict[str, Any]) -> bool:
    if not row_bool(row, "crossing_momentum_available"):
        return True
    return any(as_float(row, field) is None for field in REQUIRED_MOMENTUM_FIELDS)


def invariant_invalid(row: dict[str, Any], tolerance: float) -> bool:
    for field in INVARIANT_FIELDS:
        value = as_float(row, field)
        if value is None or abs(value) > tolerance:
            return True
    return False


def redshift_columns(
    row: dict[str, Any],
    *,
    spin: float,
    energy_tolerance: float,
    invariant_tolerance: float,
    mode: str,
    emitter_frame: str,
    observer_frame: str,
) -> dict[str, Any]:
    out = {
        "emit_energy_zamo_gev": None,
        "observed_energy_gev": None,
        "redshift_factor": None,
        "redshift_status": "valid",
        "energy_emit_input_relative_error": None,
        "photon_redshift_mode": mode,
        "photon_redshift_emitter_frame": emitter_frame,
        "photon_redshift_observer_frame": observer_frame,
    }
    if missing_required_fields(row):
        out["redshift_status"] = "missing_required_momentum"
        return out
    if invariant_invalid(row, invariant_tolerance):
        out["redshift_status"] = "invalid_invariants"
        return out

    initial_r = as_float(row, "initial_r_rg")
    initial_theta = as_float(row, "initial_theta_rad")
    p_t_initial = as_float(row, "p_t_initial")
    p_phi_initial = as_float(row, "p_phi_initial")
    crossing_r = as_float(row, "observer_crossing_r_rg")
    crossing_theta = as_float(row, "observer_crossing_theta_rad")
    p_t_crossing = as_float(row, "p_t_crossing")
    p_phi_crossing = as_float(row, "p_phi_crossing")
    input_energy = as_float(row, "input_energy_gev")
    assert initial_r is not None
    assert initial_theta is not None
    assert p_t_initial is not None
    assert p_phi_initial is not None
    assert crossing_r is not None
    assert crossing_theta is not None
    assert p_t_crossing is not None
    assert p_phi_crossing is not None
    assert input_energy is not None

    emit_energy = zamo_energy_gev(spin, initial_r, initial_theta, p_t_initial, p_phi_initial)
    observed_energy = zamo_energy_gev(spin, crossing_r, crossing_theta, p_t_crossing, p_phi_crossing)
    if emit_energy is None or observed_energy is None or emit_energy <= 0.0 or observed_energy <= 0.0:
        out["redshift_status"] = "invalid_nonpositive_energy"
        return out

    out["emit_energy_zamo_gev"] = emit_energy
    energy_error = abs(emit_energy - input_energy) / max(abs(input_energy), sys.float_info.epsilon)
    out["energy_emit_input_relative_error"] = energy_error
    if not math.isfinite(energy_error) or energy_error > energy_tolerance:
        out["redshift_status"] = "invalid_emit_energy_mismatch"
        return out

    redshift_factor = observed_energy / emit_energy
    if not (math.isfinite(redshift_factor) and redshift_factor > 0.0):
        out["redshift_status"] = "invalid_nonpositive_energy"
        out["observed_energy_gev"] = None
        out["redshift_factor"] = None
        return out
    out["observed_energy_gev"] = observed_energy
    out["redshift_factor"] = redshift_factor
    return out


def process_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    mode = args.photon_redshift_mode
    if mode == "disabled":
        return [dict(row) for row in rows]
    processed = []
    for row in rows:
        out = dict(row)
        out.update(
            redshift_columns(
                out,
                spin=args.spin,
                energy_tolerance=args.photon_redshift_energy_tolerance,
                invariant_tolerance=args.photon_invariant_tolerance,
                mode=mode,
                emitter_frame=args.photon_redshift_emitter_frame,
                observer_frame=args.photon_redshift_observer_frame,
            )
        )
        processed.append(out)
    if parse_bool(args.photon_redshift_fail_on_invalid):
        invalid = [row for row in processed if row.get("redshift_status") != "valid"]
        if invalid:
            status = invalid[0].get("redshift_status", "invalid")
            raise ValueError(f"validated_zamo redshift failed for {len(invalid)} rows; first status={status}")
    return processed


def build_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get("redshift_status") == "valid"]
    redshifts = [float(row["redshift_factor"]) for row in valid]
    observed = [float(row["observed_energy_gev"]) for row in valid]
    total_input = sum(float(row.get("input_energy_gev", 0.0) or 0.0) for row in rows)
    return {
        "photon_redshift_mode": mode,
        "n_input_rows": len(rows),
        "n_redshift_valid": len(valid),
        "n_redshift_invalid": len(rows) - len(valid) if mode == "validated_zamo" else 0,
        "total_input_energy_gev": total_input,
        "total_observed_energy_gev": sum(observed),
        "mean_redshift_factor": sum(redshifts) / len(redshifts) if redshifts else 0.0,
        "min_redshift_factor": min(redshifts) if redshifts else 0.0,
        "max_redshift_factor": max(redshifts) if redshifts else 0.0,
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in SUMMARY_FIELDS})


def write_provenance(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "phase": "photon_observer_camera_redshift",
        "input": str(args.input),
        "output_csv": str(args.output_csv),
        "observed_energy_available": (
            args.photon_redshift_mode == "validated_zamo"
            and int(summary["n_redshift_valid"]) > 0
        ),
        "photon_redshift_mode": args.photon_redshift_mode,
        "photon_redshift_emitter_frame": args.photon_redshift_emitter_frame,
        "photon_redshift_observer_frame": args.photon_redshift_observer_frame,
        "photon_redshift_energy_tolerance": args.photon_redshift_energy_tolerance,
        "photon_redshift_fail_on_invalid": parse_bool(args.photon_redshift_fail_on_invalid),
        "photon_invariant_tolerance": args.photon_invariant_tolerance,
        "requires_p_mu_initial": True,
        "requires_p_mu_crossing": True,
        "redshift_formula": "E(u) = -p_mu u^mu",
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "observer_sphere_crossing_is_detection": False,
        "limitations": [
            "ideal local ZAMO photon energy at observer sphere",
            "no detector model",
            "no aperture acceptance",
            "no instrument response",
        ],
        **summary,
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_fieldnames(input_fields: list[str], mode: str) -> list[str]:
    if mode == "disabled":
        return list(input_fields)
    fields = list(input_fields)
    for field in REDSHIFT_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def main() -> int:
    args = parse_args()
    try:
        validate_config(args)
        input_fields, rows = read_camera_csv(args.input)
        processed = process_rows(rows, args)
        summary = build_summary(processed, args.photon_redshift_mode)
        write_rows(args.output_csv, output_fieldnames(input_fields, args.photon_redshift_mode), processed)
        write_summary_csv(args.summary_csv, summary)
        write_provenance(args.provenance, args, summary)
    except Exception as exc:
        print(f"Failed to build photon observer camera redshift: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
