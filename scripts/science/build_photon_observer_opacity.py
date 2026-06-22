#!/usr/bin/env python3
"""Build photon observer opacity products for validated toy/contract modes."""

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
    "opacity_status",
    "n_medium_segments_used",
    "path_subset_status",
]

SUMMARY_FIELDS = [
    "photon_opacity_mode",
    "photon_opacity_model",
    "n_input_photons",
    "n_opacity_valid",
    "n_opacity_invalid",
    "n_paths_used",
    "n_paths_excluded_truncated",
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
    parser.add_argument("--medium-compressed-jsonl", type=Path)
    parser.add_argument("--medium-compressed-summary-csv", type=Path)
    parser.add_argument("--medium-compressed-provenance", type=Path)
    parser.add_argument("--photon-opacity-mode", required=True, choices=["disabled", "vacuum", "constant_alpha_path", "density_gray_toy"])
    parser.add_argument("--photon-constant-alpha-per-rg", required=True, type=float)
    parser.add_argument("--photon-density-gray-kappa-per-rg-per-gcm3", default=0.0, type=float)
    parser.add_argument("--photon-density-gray-energy-exponent", default=0.0, type=float)
    parser.add_argument("--photon-density-gray-reference-energy-gev", default=1.0, type=float)
    parser.add_argument("--photon-opacity-truncated-path-policy", default="fail", choices=["fail", "exclude", "diagnostic_only"])
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


def medium_summary_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    _, rows = read_csv(path)
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, int] = {}
    for key in ["n_compressed_paths_total", "n_compressed_paths_used", "n_compressed_paths_excluded_truncated"]:
        try:
            out[key] = int(row.get(key, 0) or 0)
        except ValueError:
            out[key] = 0
    return out


def density_gray_medium_backend(provenance_path: Path | None, summary_path: Path | None) -> tuple[str, list[str]]:
    warnings: list[str] = []
    provenance = read_json_optional(provenance_path)
    if provenance:
        backend = str(provenance.get("medium_backend") or provenance.get("photon_medium_model") or "").strip()
        if backend:
            return backend, warnings
    if summary_path is not None and summary_path.exists():
        _, rows = read_csv(summary_path)
        if rows:
            backend = str(rows[0].get("photon_medium_model") or "").strip()
            if backend:
                warnings.append("medium backend inferred from medium summary because provenance did not provide medium_backend")
                return backend, warnings
    warnings.append("medium backend could not be determined")
    return "unknown_medium", warnings


def opacity_model_for_mode(mode: str, medium_backend: str = "unknown_medium") -> str:
    if mode == "vacuum":
        return "vacuum_no_absorption"
    if mode == "constant_alpha_path":
        return "constant_alpha_per_rg"
    if mode == "density_gray_toy":
        safe_backend = medium_backend.strip() or "unknown_medium"
        if safe_backend in {"none", "unknown", "unknown_medium"}:
            return "density_gray_toy_unknown_medium"
        return f"density_gray_toy_{safe_backend}"
    return "none"


def medium_tau_by_photon_id(
    path: Path | None,
    *,
    kappa: float,
    energy_exponent: float,
    reference_energy: float,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None or not path.exists():
        raise FileNotFoundError("density_gray_toy requires photon_observer_medium_compressed_segments.jsonl")
    if kappa < 0.0 or not math.isfinite(kappa):
        raise ValueError("photon_density_gray_kappa_per_rg_per_gcm3 must be finite and non-negative")
    if reference_energy <= 0.0 or not math.isfinite(reference_energy):
        raise ValueError("photon_density_gray_reference_energy_gev must be finite and > 0")
    if not math.isfinite(energy_exponent):
        raise ValueError("photon_density_gray_energy_exponent must be finite")

    by_path: dict[str, dict[str, float]] = {}
    total_segments = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"invalid medium segment row at {line_number}")
            path_id = str(row.get("photon_path_id", "")).strip()
            if not path_id:
                raise ValueError(f"medium segment row {line_number} missing photon_path_id")
            if not as_bool(row.get("compression_complete", False)):
                continue
            if row.get("medium_status") != "valid":
                continue
            rho = as_float(row.get("rho_g_cm3"))
            e_gamma = as_float(row.get("E_gamma_fluid_gev"))
            dl = as_float(row.get("dl_segment_rg"))
            if rho is None or rho < 0.0:
                raise ValueError(f"invalid rho_g_cm3 at medium segment row {line_number}")
            if e_gamma is None or e_gamma <= 0.0:
                raise ValueError(f"invalid E_gamma_fluid_gev at medium segment row {line_number}")
            if dl is None or dl < 0.0:
                raise ValueError(f"invalid dl_segment_rg at medium segment row {line_number}")
            alpha = kappa * rho * ((e_gamma / reference_energy) ** energy_exponent)
            if not math.isfinite(alpha) or alpha < 0.0:
                raise ValueError(f"invalid alpha_gamma_per_rg at medium segment row {line_number}")
            tau_increment = alpha * dl
            if not math.isfinite(tau_increment) or tau_increment < 0.0:
                raise ValueError(f"invalid tau increment at medium segment row {line_number}")
            item = by_path.setdefault(path_id, {"tau": 0.0, "n_segments": 0.0, "path_length": 0.0})
            item["tau"] += tau_increment
            item["n_segments"] += 1.0
            item["path_length"] += dl
            total_segments += 1
    return by_path, {"n_medium_segments_integrated": total_segments}


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
    n_medium_segments_used: int | str = "",
    path_subset_status: str = "",
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
            "opacity_status": status,
            "n_medium_segments_used": n_medium_segments_used,
            "path_subset_status": path_subset_status,
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
            "opacity_status": status,
            "n_medium_segments_used": "",
            "path_subset_status": "",
        }
    )
    return out


def build_opacity_rows(
    rows: list[dict[str, str]],
    *,
    mode: str,
    alpha_const_per_rg: float,
    model: str,
    density_tau_by_path: dict[str, dict[str, float]] | None = None,
    n_paths_excluded_truncated: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    valid = 0
    invalid = 0
    taus: list[float] = []
    survivals: list[float] = []
    total_observed = 0.0
    total_attenuated = 0.0
    integration_method = (
        "vacuum_identity"
        if mode == "vacuum"
        else "constant_alpha_path"
        if mode == "constant_alpha_path"
        else "density_gray_toy_compressed_segments"
    )
    used_paths: set[str] = set()

    for row in rows:
        observed = as_float(row.get("observed_energy_gev"))
        if row.get("redshift_status") == "valid" and observed is not None:
            if mode == "vacuum":
                tau = 0.0
                n_segments: int | str = ""
                subset_status = ""
            elif mode == "constant_alpha_path":
                path_length = as_float(row.get("total_path_length_rg"))
                if path_length is None or path_length < 0.0:
                    output_rows.append(invalid_row(row, mode=mode, model=model, status="invalid_missing_path_length"))
                    invalid += 1
                    continue
                tau = alpha_const_per_rg * path_length
                n_segments = ""
                subset_status = ""
            else:
                path_id = str(row.get("photon_path_id", "")).strip()
                item = (density_tau_by_path or {}).get(path_id)
                if item is None:
                    out = invalid_row(row, mode=mode, model=model, status="excluded_or_missing_complete_medium_path")
                    out["path_subset_status"] = "excluded_or_missing_complete_path"
                    output_rows.append(out)
                    continue
                tau = float(item["tau"])
                n_segments = int(item["n_segments"])
                subset_status = "complete_path_subset_only"
                used_paths.add(path_id)
            out = opacity_row(
                row,
                mode=mode,
                model=model,
                tau=tau,
                observed=observed,
                integration_method=integration_method,
                status=f"valid_{mode}",
                n_medium_segments_used=n_segments,
                path_subset_status=subset_status,
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
        "n_paths_used": len(used_paths) if mode == "density_gray_toy" else valid,
        "n_paths_excluded_truncated": n_paths_excluded_truncated if mode == "density_gray_toy" else 0,
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


def write_provenance(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    created_outputs: bool,
    model: str | None = None,
    medium_backend: str = "none",
    medium_backend_warnings: list[str] | None = None,
) -> None:
    absorption_applied = (
        (args.photon_opacity_mode == "constant_alpha_path" and float(args.photon_constant_alpha_per_rg) > 0.0)
        or (args.photon_opacity_mode == "density_gray_toy" and float(args.photon_density_gray_kappa_per_rg_per_gcm3) > 0.0)
    )
    model = model or opacity_model_for_mode(args.photon_opacity_mode, medium_backend)
    toy_mode = args.photon_opacity_mode == "density_gray_toy"
    provenance = {
        "phase": "photon_observer_camera_opacity",
        "input": str(args.redshift_csv),
        "output": str(args.output_csv) if created_outputs else None,
        "summary": str(args.summary_csv) if created_outputs else None,
        "photon_opacity_mode": args.photon_opacity_mode,
        "photon_opacity_model": model,
        "alpha_const_per_rg": float(args.photon_constant_alpha_per_rg),
        "density_gray_kappa_per_rg_per_gcm3": float(args.photon_density_gray_kappa_per_rg_per_gcm3),
        "density_gray_energy_exponent": float(args.photon_density_gray_energy_exponent),
        "density_gray_reference_energy_gev": float(args.photon_density_gray_reference_energy_gev),
        "tau_integration_method": "density_gray_toy_compressed_segments" if toy_mode else "constant_alpha_path" if args.photon_opacity_mode == "constant_alpha_path" else "vacuum_identity" if args.photon_opacity_mode == "vacuum" else "none",
        "path_summary_csv": str(args.path_summary_csv) if args.path_summary_csv else None,
        "medium_compressed_jsonl": str(args.medium_compressed_jsonl) if args.medium_compressed_jsonl else None,
        "medium_compressed_summary_csv": str(args.medium_compressed_summary_csv) if args.medium_compressed_summary_csv else None,
        "medium_compressed_provenance": str(args.medium_compressed_provenance) if args.medium_compressed_provenance else None,
        "path_sampling_used": False,
        "path_sampling_audit_available": bool(args.path_summary_csv and args.path_summary_csv.exists()),
        "medium_lookup_used": toy_mode,
        "medium_backend": medium_backend if toy_mode else "none",
        "medium_backend_warnings": medium_backend_warnings or [],
        "medium_input_path_mode": "compressed_complete_paths" if toy_mode else "none",
        "truncated_path_policy": args.photon_opacity_truncated_path_policy,
        "n_paths_excluded_truncated": summary.get("n_paths_excluded_truncated", 0),
        "opacity_is_toy_model": toy_mode,
        "not_pair_production": True,
        "not_compton": True,
        "not_physical_radiative_transfer": toy_mode,
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
            f"density_gray_toy uses {medium_backend} medium lookup" if toy_mode else "no medium lookup",
            "density_gray_toy is not physical radiative transfer" if toy_mode else "no rho/T/Ye/u_fluid",
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
        medium_backend = "none"
        medium_backend_warnings: list[str] = []
        if args.photon_opacity_mode == "density_gray_toy":
            medium_backend, medium_backend_warnings = density_gray_medium_backend(
                args.medium_compressed_provenance,
                args.medium_compressed_summary_csv,
            )
        model = opacity_model_for_mode(args.photon_opacity_mode, medium_backend)
        if not math.isfinite(float(args.photon_constant_alpha_per_rg)) or float(args.photon_constant_alpha_per_rg) < 0.0:
            raise ValueError("photon_constant_alpha_per_rg must be finite and non-negative")
        if not math.isfinite(float(args.photon_density_gray_kappa_per_rg_per_gcm3)) or float(args.photon_density_gray_kappa_per_rg_per_gcm3) < 0.0:
            raise ValueError("photon_density_gray_kappa_per_rg_per_gcm3 must be finite and non-negative")
        if not math.isfinite(float(args.photon_density_gray_energy_exponent)):
            raise ValueError("photon_density_gray_energy_exponent must be finite")
        if not math.isfinite(float(args.photon_density_gray_reference_energy_gev)) or float(args.photon_density_gray_reference_energy_gev) <= 0.0:
            raise ValueError("photon_density_gray_reference_energy_gev must be finite and > 0")
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
                model=model,
                medium_backend=medium_backend,
                medium_backend_warnings=medium_backend_warnings,
            )
            print("Photon observer opacity disabled; no attenuated camera file written")
            return 0

        fields, rows = read_csv(args.redshift_csv)
        if "observed_energy_gev" not in fields:
            raise ValueError("photon opacity mode requires observed_energy_gev")
        if args.photon_opacity_mode == "constant_alpha_path" and "total_path_length_rg" not in fields:
            raise ValueError("constant_alpha_path requires total_path_length_rg from photon geodesic integration")
        density_tau: dict[str, dict[str, float]] | None = None
        n_excluded = 0
        if args.photon_opacity_mode == "density_gray_toy":
            medium_counts = medium_summary_counts(args.medium_compressed_summary_csv)
            n_excluded = int(medium_counts.get("n_compressed_paths_excluded_truncated", 0))
            if n_excluded > 0 and args.photon_opacity_truncated_path_policy == "fail":
                raise ValueError("density_gray_toy found truncated paths and photon_opacity_truncated_path_policy=fail")
            density_tau, _ = medium_tau_by_photon_id(
                args.medium_compressed_jsonl,
                kappa=float(args.photon_density_gray_kappa_per_rg_per_gcm3),
                energy_exponent=float(args.photon_density_gray_energy_exponent),
                reference_energy=float(args.photon_density_gray_reference_energy_gev),
            )
            if not density_tau:
                raise ValueError("density_gray_toy found no complete valid medium paths")
        output_rows, summary = build_opacity_rows(
            rows,
            mode=args.photon_opacity_mode,
            alpha_const_per_rg=float(args.photon_constant_alpha_per_rg),
            model=model,
            density_tau_by_path=density_tau,
            n_paths_excluded_truncated=n_excluded,
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
        write_provenance(
            args,
            summary,
            created_outputs=True,
            model=model,
            medium_backend=medium_backend,
            medium_backend_warnings=medium_backend_warnings,
        )
    except Exception as exc:
        print(f"Photon observer opacity failed: {exc}", file=sys.stderr)
        return 2
    print(f"Photon observer opacity products written to {args.output_csv.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
