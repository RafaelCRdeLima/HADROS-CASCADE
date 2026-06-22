#!/usr/bin/env python3
"""Build initial photon observer opacity products for disabled/vacuum modes."""

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
    "photon_survival_probability",
    "attenuated_observed_energy_gev",
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
    parser.add_argument("--photon-opacity-mode", required=True, choices=["disabled", "vacuum"])
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


def build_vacuum_rows(rows: list[dict[str, str]], input_fields: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    valid = 0
    invalid = 0
    taus: list[float] = []
    survivals: list[float] = []
    total_observed = 0.0
    total_attenuated = 0.0

    for row in rows:
        out = dict(row)
        observed = as_float(row.get("observed_energy_gev"))
        if row.get("redshift_status") == "valid" and observed is not None:
            tau = 0.0
            survival = 1.0
            attenuated = observed
            out.update(
                {
                    "photon_opacity_mode": "vacuum",
                    "photon_opacity_model": "vacuum_no_absorption",
                    "photon_path_optical_depth": tau,
                    "photon_survival_probability": survival,
                    "attenuated_observed_energy_gev": attenuated,
                    "photon_opacity_status": "valid_vacuum",
                }
            )
            valid += 1
            taus.append(tau)
            survivals.append(survival)
            total_observed += observed
            total_attenuated += attenuated
        else:
            out.update(
                {
                    "photon_opacity_mode": "vacuum",
                    "photon_opacity_model": "vacuum_no_absorption",
                    "photon_path_optical_depth": "",
                    "photon_survival_probability": "",
                    "attenuated_observed_energy_gev": "",
                    "photon_opacity_status": "invalid_missing_observed_energy",
                }
            )
            invalid += 1
        output_rows.append(out)

    summary = {
        "photon_opacity_mode": "vacuum",
        "photon_opacity_model": "vacuum_no_absorption",
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


def validate_vacuum_outputs(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row.get("photon_opacity_status") != "valid_vacuum":
            continue
        tau = as_float(row.get("photon_path_optical_depth"))
        survival = as_float(row.get("photon_survival_probability"))
        observed = as_float(row.get("observed_energy_gev"))
        attenuated = as_float(row.get("attenuated_observed_energy_gev"))
        if tau is None or tau < 0.0:
            raise ValueError(f"invalid vacuum tau at row {index}: {tau}")
        if survival is None or not (0.0 <= survival <= 1.0):
            raise ValueError(f"invalid vacuum survival at row {index}: {survival}")
        if tau != 0.0:
            raise ValueError(f"vacuum tau must be 0 at row {index}: {tau}")
        if survival != 1.0:
            raise ValueError(f"vacuum survival must be 1 at row {index}: {survival}")
        if observed is None or attenuated is None or abs(observed - attenuated) > 0.0:
            raise ValueError(f"vacuum attenuation must preserve observed energy at row {index}")


def write_provenance(args: argparse.Namespace, summary: dict[str, Any], *, created_outputs: bool) -> None:
    provenance = {
        "phase": "photon_observer_camera_opacity",
        "input": str(args.redshift_csv),
        "output": str(args.output_csv) if created_outputs else None,
        "summary": str(args.summary_csv) if created_outputs else None,
        "photon_opacity_mode": args.photon_opacity_mode,
        "photon_opacity_model": "vacuum_no_absorption" if args.photon_opacity_mode == "vacuum" else "none",
        "photon_opacity_output_mode": args.photon_opacity_output_mode,
        "photon_opacity_fail_on_invalid": as_bool(args.photon_opacity_fail_on_invalid),
        "photon_absorption_applied": False,
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "observer_sphere_crossing_is_detection": False,
        "created_outputs": created_outputs,
        "config_snapshot": read_json_optional(args.pipeline_config),
        "git_hash": git_hash(),
        "limitations": [
            "vacuum opacity infrastructure only",
            "no astrophysical absorption",
            "no pair production",
            "no Compton scattering",
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
            raise ValueError("photon opacity vacuum mode requires observed_energy_gev")
        output_rows, summary = build_vacuum_rows(rows, fields)
        validate_vacuum_outputs(output_rows)
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
