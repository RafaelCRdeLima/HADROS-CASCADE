#!/usr/bin/env python3
"""Build Phase 2 photon observer-sphere hit records from Phase 1 crossings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


HIT_FIELDS = [
    "photon_path_id",
    "event_id",
    "particle_id",
    "pdg",
    "input_energy_gev",
    "observer_crossing_r_rg",
    "observer_crossing_theta_rad",
    "observer_crossing_phi_rad",
    "observer_crossing_interpolated",
    "crossing_step_index",
    "total_path_length_rg",
    "E_killing_initial",
    "E_killing_final",
    "Lz_initial",
    "Lz_final",
    "null_norm_max_abs_error",
    "relative_E_error",
    "relative_Lz_error",
    "momentum_input_mode",
    "initial_r_rg",
    "initial_theta_rad",
    "initial_phi_rad",
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
]

SUMMARY_FIELDS = [
    "n_input_particles",
    "n_photons",
    "n_reached_observer_sphere",
    "total_input_energy_reached_observer_sphere_gev",
    "mean_input_energy_reached_observer_sphere_gev",
    "min_theta_rad",
    "max_theta_rad",
    "min_phi_rad",
    "max_phi_rad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 photon escape classifier output not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL row at {path}:{line_number}: expected object")
            rows.append(row)
    return rows


def as_float(row: dict[str, Any], key: str, *, required: bool = True) -> float:
    value = row.get(key)
    if value is None:
        if required:
            raise ValueError(f"Phase 1 reached-observer row missing required field {key!r}: {row}")
        return math.nan
    return float(value)


def hit_from_row(row: dict[str, Any]) -> dict[str, Any]:
    energy = row.get("input_energy_gev", row.get("energy_gev"))
    if energy is None:
        raise ValueError(f"Phase 1 reached-observer row missing input energy field: {row}")
    hit = {
        "photon_path_id": row.get("photon_path_id"),
        "event_id": row.get("event_id"),
        "particle_id": row.get("particle_id"),
        "pdg": int(row.get("pdg", 22)),
        "input_energy_gev": float(energy),
        "observer_crossing_r_rg": as_float(row, "observer_crossing_r_rg"),
        "observer_crossing_theta_rad": as_float(row, "observer_crossing_theta_rad"),
        "observer_crossing_phi_rad": as_float(row, "observer_crossing_phi_rad"),
        "observer_crossing_interpolated": bool(row.get("observer_crossing_interpolated")),
        "crossing_step_index": int(row.get("crossing_step_index", row.get("geodesic_steps", -1))),
        "E_killing_initial": as_float(row, "E_killing_initial"),
        "E_killing_final": as_float(row, "E_killing_final"),
        "Lz_initial": as_float(row, "Lz_initial"),
        "Lz_final": as_float(row, "Lz_final"),
        "null_norm_max_abs_error": as_float(row, "null_norm_max_abs_error"),
        "relative_E_error": as_float(row, "relative_E_error"),
        "relative_Lz_error": as_float(row, "relative_Lz_error"),
        "momentum_input_mode": row.get("momentum_input_mode"),
        "initial_r_rg": as_float(row, "initial_r_rg"),
        "initial_theta_rad": as_float(row, "initial_theta_rad"),
        "initial_phi_rad": as_float(row, "initial_phi_rad"),
        "p_t_initial": as_float(row, "p_t_initial"),
        "p_r_initial": as_float(row, "p_r_initial"),
        "p_theta_initial": as_float(row, "p_theta_initial"),
        "p_phi_initial": as_float(row, "p_phi_initial"),
        "p_t_crossing": as_float(row, "p_t_crossing"),
        "p_r_crossing": as_float(row, "p_r_crossing"),
        "p_theta_crossing": as_float(row, "p_theta_crossing"),
        "p_phi_crossing": as_float(row, "p_phi_crossing"),
        "crossing_momentum_available": bool(row.get("crossing_momentum_available")),
        "crossing_momentum_method": row.get("crossing_momentum_method"),
        "crossing_r_error_rg": as_float(row, "crossing_r_error_rg"),
        "crossing_null_norm_abs_error": as_float(row, "crossing_null_norm_abs_error"),
    }
    optional_fields = {"photon_path_id", "total_path_length_rg"}
    missing = [key for key in HIT_FIELDS if key not in optional_fields and hit.get(key) is None]
    if missing:
        raise ValueError(f"Phase 1 reached-observer row missing required fields {missing}: {row}")
    return hit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in SUMMARY_FIELDS})


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Photon Observer Sphere Hit Map

Phase: photon_observer_sphere_hit_map

This Phase 2 product contains only interpolated photon crossings of the observer sphere from the Phase 1 escape classifier. It does not project to pixels, does not model camera acceptance, does not represent detection, and does not report observed photon energy.

| Quantity | Value |
|---|---:|
| Input Phase 1 rows | {summary["n_input_particles"]} |
| Photon rows | {summary["n_photons"]} |
| Reached observer sphere | {summary["n_reached_observer_sphere"]} |
| Total input energy at reached sphere (GeV) | {summary["total_input_energy_reached_observer_sphere_gev"]:.17g} |
| Mean input energy at reached sphere (GeV) | {summary["mean_input_energy_reached_observer_sphere_gev"]:.17g} |
| Min theta (rad) | {summary["min_theta_rad"]} |
| Max theta (rad) | {summary["max_theta_rad"]} |
| Min phi (rad) | {summary["min_phi_rad"]} |
| Max phi (rad) | {summary["max_phi_rad"]} |
"""
    path.write_text(text, encoding="utf-8")


def build_summary(rows: list[dict[str, Any]], hits: list[dict[str, Any]]) -> dict[str, Any]:
    energies = [float(hit["input_energy_gev"]) for hit in hits]
    thetas = [float(hit["observer_crossing_theta_rad"]) for hit in hits]
    phis = [float(hit["observer_crossing_phi_rad"]) for hit in hits]
    total_energy = sum(energies)
    return {
        "n_input_particles": len(rows),
        "n_photons": sum(1 for row in rows if int(row.get("pdg", 0)) == 22),
        "n_reached_observer_sphere": len(hits),
        "total_input_energy_reached_observer_sphere_gev": total_energy,
        "mean_input_energy_reached_observer_sphere_gev": total_energy / len(hits) if hits else 0.0,
        "min_theta_rad": min(thetas) if thetas else math.nan,
        "max_theta_rad": max(thetas) if thetas else math.nan,
        "min_phi_rad": min(phis) if phis else math.nan,
        "max_phi_rad": max(phis) if phis else math.nan,
    }


def write_provenance(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "phase": "photon_observer_sphere_hit_map",
        "photon_observer_mode": "observer_sphere_hits",
        "input": str(args.input),
        "output_jsonl": str(args.output_jsonl),
        "summary_csv": str(args.summary_csv),
        "summary_md": str(args.summary_md),
        "projected_to_pixels": False,
        "hits_camera_aperture": False,
        "observer_sphere_crossing_is_detection": False,
        "observed_energy_available": False,
        "redshift_observational_final_applied": False,
        "physical_interpretation": "interpolated observer-sphere crossings of photons classified by Phase 1",
        "limitation": "No pixel projection, no camera-acceptance model, and no final observed-energy/redshift product are applied in Phase 2.",
        **summary,
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        rows = read_jsonl(args.input)
        hits = [hit_from_row(row) for row in rows if row.get("classification") == "reaches_observer_sphere"]
        summary = build_summary(rows, hits)
        write_jsonl(args.output_jsonl, hits)
        write_summary_csv(args.summary_csv, summary)
        write_summary_md(args.summary_md, summary)
        write_provenance(args.provenance, args, summary)
    except Exception as exc:
        print(f"Failed to build photon observer sphere hits: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
