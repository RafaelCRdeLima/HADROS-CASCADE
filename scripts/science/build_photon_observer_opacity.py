#!/usr/bin/env python3
"""Build photon observer opacity products for disabled/vacuum/constant-alpha modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


OPACITY_FIELDS = [
    "photon_opacity_mode",
    "photon_opacity_model",
    "photon_path_optical_depth",
    "tau_path",
    "photon_survival_probability",
    "survival_probability",
    "attenuated_observed_energy_gev",
    "opacity_integration_method",
    "photon_opacity_status",
]

SUMMARY_FIELDS = [
    "photon_opacity_mode",
    "photon_opacity_model",
    "n_input_photons",
    "n_opacity_valid",
    "n_opacity_invalid",
    "mean_tau_path",
    "max_tau_path",
    "mean_survival_probability",
    "total_observed_energy_gev",
    "total_attenuated_observed_energy_gev",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redshift-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--path-summary-csv", type=Path)
    parser.add_argument("--photon-opacity-mode", required=True, choices=["disabled", "vacuum", "constant_alpha_path"])
    parser.add_argument("--photon-constant-alpha-per-rg", required=True, type=float)
    parser.add_argument("--photon-opacity-fail-on-invalid", required=True, choices=["true", "false"])
    parser.add_argument("--photon-opacity-output-mode", required=True, choices=["separate_file"])
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def path_length_by_photon_id(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    fields, rows = read_csv(path)
    required = {"photon_path_id", "total_path_length_rg"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"path summary missing required fields: {sorted(missing)}")
    out: dict[str, float] = {}
    for index, row in enumerate(rows):
        photon_path_id = str(row.get("photon_path_id", "")).strip()
        if not photon_path_id:
            raise ValueError(f"path summary row {index} is missing photon_path_id")
        length = as_float(row.get("total_path_length_rg"))
        if length is None or length < 0.0:
            raise ValueError(f"path summary row {index} has invalid total_path_length_rg={row.get('total_path_length_rg')!r}")
        if photon_path_id in out:
            raise ValueError(f"duplicate photon_path_id in path summary: {photon_path_id}")
        out[photon_path_id] = length
    return out


def git_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def opacity_row(
    row: dict[str, str],
    *,
    mode: str,
    model: str,
    tau: float,
    observed: float,
    integration_method: str,
    status: str,
) -> dict[str, Any]:
    survival = math.exp(-tau)
    attenuated = observed * survival
    out = dict(row)
    out.update(
        {
            "photon_opacity_mode": mode,
            "photon_opacity_model": model,
            "photon_path_optical_depth": tau,
            "tau_path": tau,
            "photon_survival_probability": survival,
            "survival_probability": survival,
            "attenuated_observed_energy_gev": attenuated,
            "opacity_integration_method": integration_method,
            "photon_opacity_status": status,
        }
    )
    return out


def invalid_row(row: dict[str, str], *, mode: str, model: str, status: str) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "photon_opacity_mode": mode,
            "photon_opacity_model": model,
            "photon_path_optical_depth": "",
            "tau_path": "",
            "photon_survival_probability": "",
            "survival_probability": "",
            "attenuated_observed_energy_gev": "",
            "opacity_integration_method": "",
            "photon_opacity_status": status,
        }
    )
    return out


def build_opacity_rows(
    rows: list[dict[str, str]],
    *,
    mode: str,
    alpha_const_per_rg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    valid = 0
    invalid = 0
    taus: list[float] = []
    survivals: list[float] = []
    total_observed = 0.0
    total_attenuated = 0.0
    model = "vacuum_no_absorption" if mode == "vacuum" else "constant_alpha_per_rg"
    integration_method = "vacuum_identity" if mode == "vacuum" else "constant_alpha_path"

    for row in rows:
        observed = as_float(row.get("observed_energy_gev"))
        if row.get("redshift_status") == "valid" and observed is not None:
            if mode == "vacuum":
                tau = 0.0
            else:
                path_length = as_float(row.get("total_path_length_rg"))
                if path_length is None or path_length < 0.0:
                    output_rows.append(invalid_row(row, mode=mode, model=model, status="invalid_missing_path_length"))
                    invalid += 1
                    continue
                tau = alpha_const_per_rg * path_length
            out = opacity_row(
                row,
                mode=mode,
                model=model,
                tau=tau,
                observed=observed,
                integration_method=integration_method,
                status=f"valid_{mode}",
            )
            survival = float(out["photon_survival_probability"])
            attenuated = float(out["attenuated_observed_energy_gev"])
            valid += 1
            taus.append(tau)
            survivals.append(survival)
            total_observed += observed
            total_attenuated += attenuated
        else:
            out = invalid_row(row, mode=mode, model=model, status="invalid_missing_observed_energy")
            invalid += 1
        output_rows.append(out)

    summary = {
        "photon_opacity_mode": mode,
        "photon_opacity_model": model,
        "n_input_photons": len(rows),
        "n_opacity_valid": valid,
        "n_opacity_invalid": invalid,
        "mean_tau_path": sum(taus) / len(taus) if taus else "",
        "max_tau_path": max(taus) if taus else "",
        "mean_survival_probability": sum(survivals) / len(survivals) if survivals else "",
        "total_observed_energy_gev": total_observed,
        "total_attenuated_observed_energy_gev": total_attenuated,
    }
    return output_rows, summary


def validate_opacity_outputs(rows: list[dict[str, Any]], *, mode: str, alpha_const_per_rg: float) -> None:
    if alpha_const_per_rg < 0.0 or not math.isfinite(alpha_const_per_rg):
        raise ValueError(f"photon_constant_alpha_per_rg must be finite and non-negative: {alpha_const_per_rg}")
    for index, row in enumerate(rows):
        if not str(row.get("photon_opacity_status", "")).startswith("valid"):
            continue
        tau = as_float(row.get("photon_path_optical_depth"))
        tau_alias = as_float(row.get("tau_path"))
        survival = as_float(row.get("photon_survival_probability"))
        survival_alias = as_float(row.get("survival_probability"))
        observed = as_float(row.get("observed_energy_gev"))
        attenuated = as_float(row.get("attenuated_observed_energy_gev"))
        if tau is None or tau < 0.0:
            raise ValueError(f"invalid tau at row {index}: {tau}")
        if tau_alias is None or abs(tau - tau_alias) > 0.0:
            raise ValueError(f"tau aliases disagree at row {index}: {tau} != {tau_alias}")
        if survival is None or not (0.0 <= survival <= 1.0):
            raise ValueError(f"invalid survival at row {index}: {survival}")
        if survival_alias is None or abs(survival - survival_alias) > 0.0:
            raise ValueError(f"survival aliases disagree at row {index}: {survival} != {survival_alias}")
        expected_survival = math.exp(-tau)
        if abs(survival - expected_survival) > 1.0e-14:
            raise ValueError(f"survival != exp(-tau) at row {index}: {survival} != {expected_survival}")
        if observed is None or attenuated is None or abs(attenuated - observed * survival) > 1.0e-14 * max(1.0, abs(observed)):
            raise ValueError(f"attenuated energy mismatch at row {index}")
        if mode == "vacuum" and (tau != 0.0 or survival != 1.0 or abs(observed - attenuated) > 0.0):
            raise ValueError(f"vacuum identity failed at row {index}")


def write_provenance(args: argparse.Namespace, summary: dict[str, Any], *, created_outputs: bool) -> None:
    absorption_applied = args.photon_opacity_mode == "constant_alpha_path" and float(args.photon_constant_alpha_per_rg) > 0.0
    model = (
        "vacuum_no_absorption"
        if args.photon_opacity_mode == "vacuum"
        else "constant_alpha_per_rg"
        if args.photon_opacity_mode == "constant_alpha_path"
        else "none"
    )
    provenance = {
        "phase": "photon_observer_camera_opacity",
        "input": str(args.redshift_csv),
        "output": str(args.output_csv) if created_outputs else None,
        "summary": str(args.summary_csv) if created_outputs else None,
        "photon_opacity_mode": args.photon_opacity_mode,
        "photon_opacity_model": model,
        "alpha_const_per_rg": float(args.photon_constant_alpha_per_rg),
        "tau_integration_method": "constant_alpha_path" if args.photon_opacity_mode == "constant_alpha_path" else "vacuum_identity" if args.photon_opacity_mode == "vacuum" else "none",
        "path_summary_csv": str(args.path_summary_csv) if args.path_summary_csv else None,
        "path_sampling_used": False,
        "path_sampling_audit_available": bool(args.path_summary_csv and args.path_summary_csv.exists()),
        "photon_opacity_output_mode": args.photon_opacity_output_mode,
        "photon_opacity_fail_on_invalid": as_bool(args.photon_opacity_fail_on_invalid),
        "photon_absorption_applied": absorption_applied,
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "observer_sphere_crossing_is_detection": False,
        "created_outputs": created_outputs,
        "config_snapshot": read_json_optional(args.pipeline_config),
        "git_hash": git_hash(),
        "limitations": [
            "constant_alpha_path is a synthetic opacity accumulator contract when enabled",
            "path sampling is an audit artifact and is not required for production opacity integration",
            "no pair production",
            "no Compton scattering",
            "no medium lookup",
            "no rho/T/Ye/u_fluid",
            "no full radiative transfer",
            "no detector response",
        ],
        **summary,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        if not math.isfinite(float(args.photon_constant_alpha_per_rg)) or float(args.photon_constant_alpha_per_rg) < 0.0:
            raise ValueError("photon_constant_alpha_per_rg must be finite and non-negative")
        if args.photon_opacity_mode == "disabled":
            write_provenance(
                args,
                {
                    "n_input_photons": "",
                    "n_opacity_valid": 0,
                    "n_opacity_invalid": 0,
                    "mean_tau_path": "",
                    "max_tau_path": "",
                    "mean_survival_probability": "",
                    "total_observed_energy_gev": "",
                    "total_attenuated_observed_energy_gev": "",
                },
                created_outputs=False,
            )
            print("Photon observer opacity disabled; no attenuated camera file written")
            return 0

        fields, rows = read_csv(args.redshift_csv)
        if "observed_energy_gev" not in fields:
            raise ValueError("photon opacity mode requires observed_energy_gev")
        if args.photon_opacity_mode == "constant_alpha_path" and "total_path_length_rg" not in fields:
            raise ValueError("constant_alpha_path requires total_path_length_rg from photon geodesic integration")
        output_rows, summary = build_opacity_rows(
            rows,
            mode=args.photon_opacity_mode,
            alpha_const_per_rg=float(args.photon_constant_alpha_per_rg),
        )
        validate_opacity_outputs(
            output_rows,
            mode=args.photon_opacity_mode,
            alpha_const_per_rg=float(args.photon_constant_alpha_per_rg),
        )
        if as_bool(args.photon_opacity_fail_on_invalid) and int(summary["n_opacity_invalid"]) > 0:
            raise ValueError(f"invalid photon opacity rows: {summary['n_opacity_invalid']}")
        output_fields = list(fields)
        for field in OPACITY_FIELDS:
            if field not in output_fields:
                output_fields.append(field)
        write_csv_rows(args.output_csv, output_fields, output_rows)
        write_csv_rows(args.summary_csv, SUMMARY_FIELDS, [summary])
        write_provenance(args, summary, created_outputs=True)
    except Exception as exc:
        print(f"Photon observer opacity failed: {exc}", file=sys.stderr)
        return 2
    print(f"Photon observer opacity products written to {args.output_csv.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
