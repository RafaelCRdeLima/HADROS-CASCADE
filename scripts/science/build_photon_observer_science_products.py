#!/usr/bin/env python3
"""Build ideal photon observer-camera science products from validated redshift rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


MAP_COUNT_FIELDS = ["pixel_x", "pixel_y", "n_photons"]
MAP_INPUT_FIELDS = ["pixel_x", "pixel_y", "sum_input_energy_gev"]
MAP_OBSERVED_FIELDS = ["pixel_x", "pixel_y", "sum_observed_energy_gev"]
MAP_REDSHIFT_FIELDS = ["pixel_x", "pixel_y", "mean_redshift_factor"]
MAP_ATTENUATED_FIELDS = ["pixel_x", "pixel_y", "sum_attenuated_observed_energy_gev"]
MAP_SURVIVAL_FIELDS = ["pixel_x", "pixel_y", "mean_photon_survival_probability"]
HIST_FIELDS = ["bin_index", "bin_min", "bin_max", "count"]
N_HIST_BINS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redshift-csv", required=True, type=Path)
    parser.add_argument("--attenuated-csv", type=Path)
    parser.add_argument("--validation-summary-csv", required=True, type=Path)
    parser.add_argument("--validation-provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--camera-nx", type=int)
    parser.add_argument("--camera-ny", type=int)
    parser.add_argument("--allow-unvalidated-diagnostic", action="store_true")
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


def validation_status(validation_provenance: Path, allow_unvalidated_diagnostic: bool) -> dict[str, str]:
    if not validation_provenance.exists():
        if allow_unvalidated_diagnostic:
            return {
                "physics_status": "UNKNOWN",
                "config_status": "UNKNOWN",
                "overall_status": "UNVALIDATED_DIAGNOSTIC",
                "product_class": "diagnostic_only",
            }
        raise FileNotFoundError(f"validation provenance is required for science products: {validation_provenance}")
    data = read_json(validation_provenance)
    physics_status = str(data.get("physics_status", data.get("PHYSICS_VALIDATION_STATUS", "")))
    config_status = str(data.get("config_status", data.get("CONFIG_CONTRACT_STATUS", "")))
    overall_status = str(data.get("overall_status", data.get("OVERALL_STATUS", data.get("status", ""))))
    if physics_status != "PASS" or overall_status == "VALIDATION_FAILED":
        raise ValueError(
            "photon observer science products require physics_status=PASS and overall_status not VALIDATION_FAILED; "
            f"got physics_status={physics_status!r}, overall_status={overall_status!r}"
        )
    if overall_status not in {"PASS", "VALIDATION_WARNING"}:
        raise ValueError(f"unsupported validation overall_status for science products: {overall_status!r}")
    return {
        "physics_status": physics_status,
        "config_status": config_status or "UNKNOWN",
        "overall_status": overall_status,
        "product_class": "ideal_observer_science",
    }


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("redshift_status") != "valid":
            continue
        if not as_bool(row.get("inside_fov")):
            continue
        pixel_x = as_float(row, "pixel_x")
        pixel_y = as_float(row, "pixel_y")
        input_energy = as_float(row, "input_energy_gev")
        observed_energy = as_float(row, "observed_energy_gev")
        redshift = as_float(row, "redshift_factor")
        if None in {pixel_x, pixel_y, input_energy, observed_energy, redshift}:
            continue
        selected.append(
            {
                "pixel_x": int(pixel_x),
                "pixel_y": int(pixel_y),
                "input_energy_gev": float(input_energy),
                "observed_energy_gev": float(observed_energy),
                "redshift_factor": float(redshift),
            }
        )
    return selected


def selected_attenuated_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
    selected: list[dict[str, Any]] = []
    mode: str | None = None
    for row in rows:
        if row.get("redshift_status") != "valid":
            continue
        if row.get("photon_opacity_status") not in {"valid_vacuum"}:
            continue
        if not as_bool(row.get("inside_fov")):
            continue
        pixel_x = as_float(row, "pixel_x")
        pixel_y = as_float(row, "pixel_y")
        attenuated = as_float(row, "attenuated_observed_energy_gev")
        survival = as_float(row, "photon_survival_probability")
        if None in {pixel_x, pixel_y, attenuated, survival}:
            continue
        row_mode = str(row.get("photon_opacity_mode", ""))
        if mode is None:
            mode = row_mode
        elif row_mode != mode:
            raise ValueError("attenuated CSV mixes photon_opacity_mode values")
        selected.append(
            {
                "pixel_x": int(pixel_x),
                "pixel_y": int(pixel_y),
                "attenuated_observed_energy_gev": float(attenuated),
                "photon_survival_probability": float(survival),
            }
        )
    return selected, mode


def infer_shape(rows: list[dict[str, Any]], nx: int | None, ny: int | None) -> tuple[int, int]:
    if nx is not None and nx <= 0:
        raise ValueError("camera_nx must be > 0")
    if ny is not None and ny <= 0:
        raise ValueError("camera_ny must be > 0")
    if nx is not None and ny is not None:
        return nx, ny
    max_x = max((int(row["pixel_x"]) for row in rows), default=-1)
    max_y = max((int(row["pixel_y"]) for row in rows), default=-1)
    return nx or max_x + 1, ny or max_y + 1


def pixel_maps(rows: list[dict[str, Any]], nx: int, ny: int) -> dict[str, list[dict[str, Any]]]:
    accum: dict[tuple[int, int], dict[str, float]] = {}
    for y in range(ny):
        for x in range(nx):
            accum[(x, y)] = {"count": 0.0, "input": 0.0, "observed": 0.0, "redshift_sum": 0.0}
    for row in rows:
        key = (int(row["pixel_x"]), int(row["pixel_y"]))
        if key not in accum:
            continue
        accum[key]["count"] += 1.0
        accum[key]["input"] += float(row["input_energy_gev"])
        accum[key]["observed"] += float(row["observed_energy_gev"])
        accum[key]["redshift_sum"] += float(row["redshift_factor"])

    count_rows = []
    input_rows = []
    observed_rows = []
    redshift_rows = []
    for y in range(ny):
        for x in range(nx):
            item = accum[(x, y)]
            n = int(item["count"])
            count_rows.append({"pixel_x": x, "pixel_y": y, "n_photons": n})
            input_rows.append({"pixel_x": x, "pixel_y": y, "sum_input_energy_gev": item["input"]})
            observed_rows.append({"pixel_x": x, "pixel_y": y, "sum_observed_energy_gev": item["observed"]})
            mean = item["redshift_sum"] / n if n > 0 else ""
            redshift_rows.append({"pixel_x": x, "pixel_y": y, "mean_redshift_factor": mean})
    return {
        "counts": count_rows,
        "input_energy": input_rows,
        "observed_energy": observed_rows,
        "mean_redshift": redshift_rows,
    }


def attenuated_maps(rows: list[dict[str, Any]], nx: int, ny: int) -> dict[str, list[dict[str, Any]]]:
    accum: dict[tuple[int, int], dict[str, float]] = {}
    for y in range(ny):
        for x in range(nx):
            accum[(x, y)] = {"count": 0.0, "attenuated": 0.0, "survival_sum": 0.0}
    for row in rows:
        key = (int(row["pixel_x"]), int(row["pixel_y"]))
        if key not in accum:
            continue
        accum[key]["count"] += 1.0
        accum[key]["attenuated"] += float(row["attenuated_observed_energy_gev"])
        accum[key]["survival_sum"] += float(row["photon_survival_probability"])

    energy_rows = []
    survival_rows = []
    for y in range(ny):
        for x in range(nx):
            item = accum[(x, y)]
            n = int(item["count"])
            energy_rows.append(
                {
                    "pixel_x": x,
                    "pixel_y": y,
                    "sum_attenuated_observed_energy_gev": item["attenuated"],
                }
            )
            survival_rows.append(
                {
                    "pixel_x": x,
                    "pixel_y": y,
                    "mean_photon_survival_probability": item["survival_sum"] / n if n > 0 else "",
                }
            )
    return {"attenuated_energy": energy_rows, "survival": survival_rows}


def histogram(values: list[float], n_bins: int = N_HIST_BINS) -> list[dict[str, Any]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [{"bin_index": 0, "bin_min": lo, "bin_max": hi, "count": len(values)}]
    width = (hi - lo) / n_bins
    counts = [0 for _ in range(n_bins)]
    for value in values:
        index = min(n_bins - 1, max(0, int((value - lo) / width)))
        counts[index] += 1
    return [
        {"bin_index": i, "bin_min": lo + i * width, "bin_max": lo + (i + 1) * width, "count": counts[i]}
        for i in range(n_bins)
    ]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def load_config_snapshot(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def build_summary(
    *,
    rows: list[dict[str, Any]],
    active_pixels: int,
    validation: dict[str, str],
    product_class: str,
) -> dict[str, Any]:
    observed = [float(row["observed_energy_gev"]) for row in rows]
    redshifts = [float(row["redshift_factor"]) for row in rows]
    return {
        "product_class": product_class,
        "n_valid_photons": len(rows),
        "n_active_pixels": active_pixels,
        "total_input_energy_gev": sum(float(row["input_energy_gev"]) for row in rows),
        "total_observed_energy_gev": sum(observed),
        "mean_redshift_factor": sum(redshifts) / len(redshifts) if redshifts else 0.0,
        "min_observed_energy_gev": min(observed) if observed else 0.0,
        "max_observed_energy_gev": max(observed) if observed else 0.0,
        "physics_status": validation["physics_status"],
        "config_status": validation["config_status"],
        "overall_status": validation["overall_status"],
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "photon_absorption_applied": False,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Photon Observer Science Products Summary",
        "",
        "- product_class: `{}`".format(summary["product_class"]),
        "- camera_class: `ideal_observational`",
        "- detector_model_applied: `false`",
        "- instrument_response_applied: `false`",
        "- aperture_acceptance_applied: `false`",
        "- photon_absorption_applied: `false`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for key in [
        "n_valid_photons",
        "n_active_pixels",
        "total_input_energy_gev",
        "total_observed_energy_gev",
        "mean_redshift_factor",
        "min_observed_energy_gev",
        "max_observed_energy_gev",
        "physics_status",
        "config_status",
        "overall_status",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        validation = validation_status(args.validation_provenance, args.allow_unvalidated_diagnostic)
        if not args.validation_summary_csv.exists() and not args.allow_unvalidated_diagnostic:
            raise FileNotFoundError(f"validation summary is required for science products: {args.validation_summary_csv}")
        if args.validation_summary_csv.exists():
            read_csv(args.validation_summary_csv)
        redshift_fields, redshift_rows = read_csv(args.redshift_csv)
        rows = selected_rows(redshift_rows)
        nx, ny = infer_shape(rows, args.camera_nx, args.camera_ny)
        maps = pixel_maps(rows, nx, ny)
        active_pixels = sum(1 for row in maps["counts"] if int(row["n_photons"]) > 0)
        summary = build_summary(
            rows=rows,
            active_pixels=active_pixels,
            validation=validation,
            product_class=validation["product_class"],
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(args.output_dir / "photon_observer_counts_map.csv", MAP_COUNT_FIELDS, maps["counts"])
        write_csv_rows(args.output_dir / "photon_observer_input_energy_map.csv", MAP_INPUT_FIELDS, maps["input_energy"])
        write_csv_rows(args.output_dir / "photon_observer_observed_energy_map.csv", MAP_OBSERVED_FIELDS, maps["observed_energy"])
        write_csv_rows(args.output_dir / "photon_observer_mean_redshift_map.csv", MAP_REDSHIFT_FIELDS, maps["mean_redshift"])
        write_csv_rows(
            args.output_dir / "photon_observer_spectrum_input.csv",
            HIST_FIELDS,
            histogram([float(row["input_energy_gev"]) for row in rows]),
        )
        write_csv_rows(
            args.output_dir / "photon_observer_spectrum_observed.csv",
            HIST_FIELDS,
            histogram([float(row["observed_energy_gev"]) for row in rows]),
        )
        write_csv_rows(
            args.output_dir / "photon_observer_redshift_distribution.csv",
            HIST_FIELDS,
            histogram([float(row["redshift_factor"]) for row in rows]),
        )
        attenuated_summary: dict[str, Any] = {
            "attenuated_products_available": False,
            "photon_absorption_applied": False,
            "photon_opacity_mode": "none",
        }
        if args.attenuated_csv is not None and args.attenuated_csv.exists():
            attenuated_fields, attenuated_input_rows = read_csv(args.attenuated_csv)
            attenuated_selected, opacity_mode = selected_attenuated_rows(attenuated_input_rows)
            if opacity_mode != "vacuum":
                raise ValueError(f"only photon_opacity_mode='vacuum' is supported for attenuated science products, got {opacity_mode!r}")
            att_maps = attenuated_maps(attenuated_selected, nx, ny)
            write_csv_rows(
                args.output_dir / "photon_observer_attenuated_energy_map.csv",
                MAP_ATTENUATED_FIELDS,
                att_maps["attenuated_energy"],
            )
            write_csv_rows(
                args.output_dir / "photon_observer_survival_map.csv",
                MAP_SURVIVAL_FIELDS,
                att_maps["survival"],
            )
            write_csv_rows(
                args.output_dir / "photon_observer_attenuated_spectrum.csv",
                HIST_FIELDS,
                histogram([float(row["attenuated_observed_energy_gev"]) for row in attenuated_selected]),
            )
            attenuated_summary = {
                "attenuated_products_available": True,
                "photon_absorption_applied": False,
                "photon_opacity_mode": opacity_mode,
                "n_attenuated_photons": len(attenuated_selected),
                "total_attenuated_observed_energy_gev": sum(
                    float(row["attenuated_observed_energy_gev"]) for row in attenuated_selected
                ),
            }
        write_summary_md(args.output_dir / "photon_observer_science_summary.md", summary)
        provenance = {
            "phase": "photon_observer_science_products",
            "product_class": validation["product_class"],
            "input_files": {
                "redshift_csv": str(args.redshift_csv),
                "attenuated_csv": str(args.attenuated_csv) if args.attenuated_csv else None,
                "validation_summary_csv": str(args.validation_summary_csv),
                "validation_provenance": str(args.validation_provenance),
                "pipeline_config": str(args.pipeline_config) if args.pipeline_config else None,
            },
            "validation_statuses": validation,
            "selection_rules": [
                "redshift_status == valid",
                "inside_fov == true",
                "observed_energy_gev finite",
                "redshift_factor finite",
            ],
            "binning_rules": {
                "pixel_maps": "camera pixel grid",
                "histograms": f"{N_HIST_BINS} uniform bins over selected finite values",
            },
            "camera_nx": nx,
            "camera_ny": ny,
            "git_hash": git_hash(),
            "config_snapshot": load_config_snapshot(args.pipeline_config),
            "detector_model_applied": False,
            "instrument_response_applied": False,
            "aperture_acceptance_applied": False,
            "photon_absorption_applied": False,
            "attenuated_products": attenuated_summary,
            "physical_limitations": [
                "ideal photon observer camera",
                "no detector response",
                "no aperture acceptance",
                "no instrument response",
                "no photon absorption or scattering",
                "not particle_ray_association_camera",
            ],
            **summary,
        }
        (args.output_dir / "photon_observer_science_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Photon observer science products failed: {exc}", file=sys.stderr)
        return 2
    print(f"Photon observer science products written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
