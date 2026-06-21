#!/usr/bin/env python3
"""Project Phase 2 photon observer-sphere hits onto an ideal camera plane."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


CAMERA_FIELDS = [
    "event_id",
    "particle_id",
    "pdg",
    "pixel_x",
    "pixel_y",
    "camera_x",
    "camera_y",
    "inside_fov",
    "projection_status",
    "input_energy_gev",
    "observer_crossing_r_rg",
    "observer_crossing_theta_rad",
    "observer_crossing_phi_rad",
    "observer_crossing_interpolated",
    "crossing_step_index",
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
    "E_killing_initial",
    "E_killing_final",
    "Lz_initial",
    "Lz_final",
    "null_norm_max_abs_error",
    "relative_E_error",
    "relative_Lz_error",
    "projection_mode",
]

SUMMARY_FIELDS = [
    "n_input_hits",
    "n_inside_fov",
    "n_outside_fov",
    "total_input_energy_inside_fov_gev",
    "mean_input_energy_inside_fov_gev",
    "min_pixel_x",
    "max_pixel_x",
    "min_pixel_y",
    "max_pixel_y",
    "projection_mode",
]

POLE_TOLERANCE = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--camera-nx", required=True, type=int)
    parser.add_argument("--camera-ny", required=True, type=int)
    parser.add_argument("--photon-camera-fov-deg", required=True, type=float)
    parser.add_argument("--photon-camera-projection-mode", required=True)
    parser.add_argument("--photon-camera-fov-definition", required=True)
    parser.add_argument("--photon-camera-resolution-mode", required=True)
    parser.add_argument("--photon-camera-center-theta-source", required=True)
    parser.add_argument("--photon-camera-center-theta-deg", required=True, type=float)
    parser.add_argument("--photon-camera-center-phi-rad", required=True, type=float)
    parser.add_argument("--photon-camera-clipping-mode", required=True)
    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> None:
    if args.photon_camera_projection_mode != "gnomonic_pinhole":
        raise ValueError("photon_camera_projection_mode must be gnomonic_pinhole")
    if args.photon_camera_fov_definition != "square_half_angle":
        raise ValueError("photon_camera_fov_definition must be square_half_angle")
    if args.photon_camera_resolution_mode != "reuse_main_camera":
        raise ValueError("photon_camera_resolution_mode must be reuse_main_camera")
    if args.photon_camera_center_theta_source != "observer_inclination_deg":
        raise ValueError("photon_camera_center_theta_source must be observer_inclination_deg")
    if args.photon_camera_clipping_mode != "keep_outside_fov":
        raise ValueError("photon_camera_clipping_mode must be keep_outside_fov")
    if not (args.photon_camera_fov_deg > 0.0 and args.photon_camera_fov_deg < 180.0):
        raise ValueError("photon_camera_fov_deg must satisfy 0 < value < 180")
    if args.camera_nx <= 0:
        raise ValueError("camera_nx must be > 0")
    if args.camera_ny <= 0:
        raise ValueError("camera_ny must be > 0")
    theta0 = math.radians(args.photon_camera_center_theta_deg)
    if abs(math.sin(theta0)) < POLE_TOLERANCE:
        raise ValueError("photon camera optical center is too close to a spherical pole")
    if not math.isfinite(args.photon_camera_center_phi_rad):
        raise ValueError("photon_camera_center_phi_rad must be finite")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 photon observer-sphere hit file not found: {path}")
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


def require_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Phase 2 hit missing required field {key!r}: {row}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Phase 2 hit field {key!r} is not finite: {row}")
    return out


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit_from_angles(theta: float, phi: float) -> tuple[float, float, float]:
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def camera_basis(theta0: float, phi0: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    c = unit_from_angles(theta0, phi0)
    e_x = (-math.sin(phi0), math.cos(phi0), 0.0)
    e_y = (
        -math.cos(theta0) * math.cos(phi0),
        -math.cos(theta0) * math.sin(phi0),
        math.sin(theta0),
    )
    return c, e_x, e_y


def project_hit(
    row: dict[str, Any],
    *,
    c: tuple[float, float, float],
    e_x: tuple[float, float, float],
    e_y: tuple[float, float, float],
    extent: float,
    nx: int,
    ny: int,
    projection_mode: str,
) -> dict[str, Any]:
    theta = require_float(row, "observer_crossing_theta_rad")
    phi = require_float(row, "observer_crossing_phi_rad")
    n = unit_from_angles(theta, phi)
    denom = dot(n, c)

    camera_x: float | None = None
    camera_y: float | None = None
    inside_fov = False
    projection_status = "behind_camera_plane"
    pixel_x: int | None = None
    pixel_y: int | None = None

    if denom > 0.0:
        camera_x = dot(n, e_x) / denom
        camera_y = dot(n, e_y) / denom
        inside_fov = abs(camera_x) <= extent and abs(camera_y) <= extent
        projection_status = "inside_fov" if inside_fov else "outside_fov"
        if inside_fov:
            u = 0.5 * (camera_x / extent + 1.0)
            v = 0.5 * (1.0 - camera_y / extent)
            pixel_x = math.floor(u * nx)
            pixel_y = math.floor(v * ny)
            if pixel_x == nx:
                pixel_x = nx - 1
            if pixel_y == ny:
                pixel_y = ny - 1

    return {
        "event_id": row.get("event_id"),
        "particle_id": row.get("particle_id"),
        "pdg": int(row.get("pdg", 22)),
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "camera_x": camera_x,
        "camera_y": camera_y,
        "inside_fov": inside_fov,
        "projection_status": projection_status,
        "input_energy_gev": require_float(row, "input_energy_gev"),
        "observer_crossing_r_rg": require_float(row, "observer_crossing_r_rg"),
        "observer_crossing_theta_rad": theta,
        "observer_crossing_phi_rad": phi,
        "observer_crossing_interpolated": bool(row.get("observer_crossing_interpolated")),
        "crossing_step_index": int(row.get("crossing_step_index", -1)),
        "momentum_input_mode": row.get("momentum_input_mode"),
        "initial_r_rg": require_float(row, "initial_r_rg"),
        "initial_theta_rad": require_float(row, "initial_theta_rad"),
        "initial_phi_rad": require_float(row, "initial_phi_rad"),
        "p_t_initial": require_float(row, "p_t_initial"),
        "p_r_initial": require_float(row, "p_r_initial"),
        "p_theta_initial": require_float(row, "p_theta_initial"),
        "p_phi_initial": require_float(row, "p_phi_initial"),
        "p_t_crossing": require_float(row, "p_t_crossing"),
        "p_r_crossing": require_float(row, "p_r_crossing"),
        "p_theta_crossing": require_float(row, "p_theta_crossing"),
        "p_phi_crossing": require_float(row, "p_phi_crossing"),
        "crossing_momentum_available": bool(row.get("crossing_momentum_available")),
        "E_killing_initial": require_float(row, "E_killing_initial"),
        "E_killing_final": require_float(row, "E_killing_final"),
        "Lz_initial": require_float(row, "Lz_initial"),
        "Lz_final": require_float(row, "Lz_final"),
        "null_norm_max_abs_error": require_float(row, "null_norm_max_abs_error"),
        "relative_E_error": require_float(row, "relative_E_error"),
        "relative_Lz_error": require_float(row, "relative_Lz_error"),
        "projection_mode": projection_mode,
    }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_camera_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in CAMERA_FIELDS})


def build_summary(rows: list[dict[str, Any]], projection_mode: str) -> dict[str, Any]:
    inside = [row for row in rows if row["inside_fov"]]
    energies = [float(row["input_energy_gev"]) for row in inside]
    pixel_xs = [int(row["pixel_x"]) for row in inside]
    pixel_ys = [int(row["pixel_y"]) for row in inside]
    total_energy = sum(energies)
    return {
        "n_input_hits": len(rows),
        "n_inside_fov": len(inside),
        "n_outside_fov": len(rows) - len(inside),
        "total_input_energy_inside_fov_gev": total_energy,
        "mean_input_energy_inside_fov_gev": total_energy / len(inside) if inside else 0.0,
        "min_pixel_x": min(pixel_xs) if pixel_xs else math.nan,
        "max_pixel_x": max(pixel_xs) if pixel_xs else math.nan,
        "min_pixel_y": min(pixel_ys) if pixel_ys else math.nan,
        "max_pixel_y": max(pixel_ys) if pixel_ys else math.nan,
        "projection_mode": projection_mode,
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
        "phase": "photon_observer_camera_projection",
        "input": str(args.input),
        "projected_to_pixels": True,
        "observer_sphere_crossing_is_detection": False,
        "observed_energy_available": False,
        "detector_model_applied": False,
        "instrument_response_applied": False,
        "aperture_acceptance_applied": False,
        "projection_mode": args.photon_camera_projection_mode,
        "photon_observer_mode": "observer_camera_projection",
        "photon_camera_fov_deg": args.photon_camera_fov_deg,
        "photon_camera_fov_definition": args.photon_camera_fov_definition,
        "photon_camera_resolution_mode": args.photon_camera_resolution_mode,
        "photon_camera_center_theta_source": args.photon_camera_center_theta_source,
        "photon_camera_center_theta_deg": args.photon_camera_center_theta_deg,
        "photon_camera_center_phi_rad": args.photon_camera_center_phi_rad,
        "photon_camera_clipping_mode": args.photon_camera_clipping_mode,
        "camera_nx": args.camera_nx,
        "camera_ny": args.camera_ny,
        "limitations": [
            "no observed photon energy",
            "no detector model",
            "no aperture acceptance",
            "no instrument response",
            "no geodesic reintegration",
        ],
        **summary,
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        validate_config(args)
        theta0 = math.radians(args.photon_camera_center_theta_deg)
        phi0 = args.photon_camera_center_phi_rad
        c, e_x, e_y = camera_basis(theta0, phi0)
        extent = math.tan(0.5 * math.radians(args.photon_camera_fov_deg))
        input_rows = read_jsonl(args.input)
        rows = [
            project_hit(
                row,
                c=c,
                e_x=e_x,
                e_y=e_y,
                extent=extent,
                nx=args.camera_nx,
                ny=args.camera_ny,
                projection_mode=args.photon_camera_projection_mode,
            )
            for row in input_rows
        ]
        summary = build_summary(rows, args.photon_camera_projection_mode)
        write_camera_csv(args.output_csv, rows)
        write_summary_csv(args.summary_csv, summary)
        write_provenance(args.provenance, args, summary)
    except Exception as exc:
        print(f"Failed to build photon observer camera projection: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
