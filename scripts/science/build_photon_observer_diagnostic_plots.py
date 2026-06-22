#!/usr/bin/env python3
"""Build diagnostic plots for the photon observer camera.

These plots are lightweight diagnostics for the ideal photon observer camera.
They are not paper-ready figures and do not model detector response.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PHYSICS_LABEL = "ideal photon observer camera, no detector response"

OUTPUT_FILENAMES = [
    "photon_diagnostic_input_energy_map.png",
    "photon_diagnostic_observed_energy_map.png",
    "photon_diagnostic_counts_map.png",
    "photon_diagnostic_mean_redshift_map.png",
    "photon_diagnostic_valid_photon_density_map.png",
    "photon_diagnostic_mean_observed_energy_map.png",
    "photon_diagnostic_redshift_histogram.png",
    "photon_diagnostic_input_vs_observed_energy.png",
]

RECENTERED_OUTPUT_FILENAMES = [
    "photon_diagnostic_recentered_counts_map.png",
    "photon_diagnostic_recentered_observed_energy_map.png",
    "photon_diagnostic_recentered_mean_redshift_map.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--diagnostic-recenter",
        action="store_true",
        help="also build non-default diagnostic maps recentered on the photon angular centroid",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"photon observer camera redshift CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(row: dict[str, Any], key: str) -> int | None:
    value = as_float(row, key)
    if value is None:
        return None
    out = int(value)
    return out if abs(value - out) < 1.0e-9 else None


def as_bool(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def require_fields(fieldnames: list[str]) -> None:
    required = {
        "input_energy_gev",
        "observed_energy_gev",
        "redshift_factor",
        "redshift_status",
        "inside_fov",
        "pixel_x",
        "pixel_y",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"missing required photon diagnostic field(s): {', '.join(missing)}")


def require_recenter_fields(fieldnames: list[str]) -> None:
    required = {
        "observer_crossing_theta_rad",
        "observer_crossing_phi_rad",
        "observed_energy_gev",
        "input_energy_gev",
        "redshift_factor",
        "redshift_status",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"missing required photon recentering field(s): {', '.join(missing)}")


def valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("redshift_status", "")).strip().lower() == "valid"]


def valid_inside_fov_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in valid_rows(rows)
        if as_bool(row, "inside_fov")
        and as_int(row, "pixel_x") is not None
        and as_int(row, "pixel_y") is not None
    ]


def infer_shape_from_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    max_x = max((as_int(row, "pixel_x") or 0 for row in rows), default=0)
    max_y = max((as_int(row, "pixel_y") or 0 for row in rows), default=0)
    return max_x + 1, max_y + 1


def read_camera_shape(input_path: Path, rows: list[dict[str, str]]) -> tuple[int, int]:
    provenance = input_path.with_name("photon_observer_camera_provenance.json")
    if provenance.exists():
        try:
            data = json.loads(provenance.read_text(encoding="utf-8"))
            nx = int(data.get("camera_nx", 0))
            ny = int(data.get("camera_ny", 0))
            if nx > 0 and ny > 0:
                return nx, ny
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return infer_shape_from_rows(rows)


def read_camera_projection_config(input_path: Path) -> dict[str, float]:
    provenance = input_path.with_name("photon_observer_camera_provenance.json")
    defaults = {
        "camera_nx": 0.0,
        "camera_ny": 0.0,
        "photon_camera_fov_deg": 60.0,
        "photon_camera_center_theta_deg": 90.0,
        "photon_camera_center_phi_rad": 0.0,
    }
    if not provenance.exists():
        return defaults
    try:
        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    for key in defaults:
        try:
            defaults[key] = float(data.get(key, defaults[key]))
        except (TypeError, ValueError):
            pass
    return defaults


def zero_map(nx: int, ny: int) -> list[list[float]]:
    return [[0.0 for _ in range(nx)] for _ in range(ny)]


def divide_maps(numerator: list[list[float]], denominator: list[list[float]]) -> list[list[float]]:
    ny = len(numerator)
    nx = len(numerator[0]) if ny else 0
    out = zero_map(nx, ny)
    for y, line in enumerate(numerator):
        for x, value in enumerate(line):
            count = denominator[y][x]
            out[y][x] = value / count if count > 0.0 else 0.0
    return out


def build_maps(rows: list[dict[str, str]], nx: int, ny: int) -> dict[str, list[list[float]]]:
    input_energy = zero_map(nx, ny)
    observed_energy = zero_map(nx, ny)
    redshift_sum = zero_map(nx, ny)
    counts = zero_map(nx, ny)
    negative_energy = zero_map(nx, ny)
    for row in rows:
        x = as_int(row, "pixel_x")
        y = as_int(row, "pixel_y")
        input_value = as_float(row, "input_energy_gev")
        observed_value = as_float(row, "observed_energy_gev")
        redshift_value = as_float(row, "redshift_factor")
        if x is None or y is None or input_value is None or observed_value is None or redshift_value is None:
            continue
        if not (0 <= y < ny and 0 <= x < nx):
            continue
        input_energy[y][x] += input_value
        observed_energy[y][x] += observed_value
        redshift_sum[y][x] += redshift_value
        counts[y][x] += 1.0
        if input_value < 0.0 or observed_value < 0.0:
            negative_energy[y][x] += 1.0
    return {
        "input_energy": input_energy,
        "observed_energy": observed_energy,
        "counts": counts,
        "valid_density": counts,
        "mean_redshift": divide_maps(redshift_sum, counts),
        "mean_observed_energy": divide_maps(observed_energy, counts),
        "negative_energy": negative_energy,
    }


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit_from_angles(theta: float, phi: float) -> tuple[float, float, float]:
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def angles_from_unit(vector: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = vector
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("cannot compute diagnostic recentering from a zero angular centroid")
    z_unit = max(-1.0, min(1.0, z / norm))
    return math.acos(z_unit), math.atan2(y, x)


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


def angular_centroid(rows: list[dict[str, str]], *, weight_key: str | None = None) -> tuple[float, float]:
    sx = sy = sz = total_weight = 0.0
    for row in rows:
        theta = as_float(row, "observer_crossing_theta_rad")
        phi = as_float(row, "observer_crossing_phi_rad")
        if theta is None or phi is None:
            continue
        weight = 1.0
        if weight_key is not None:
            weight_value = as_float(row, weight_key)
            if weight_value is None or weight_value <= 0.0:
                continue
            weight = weight_value
        x, y, z = unit_from_angles(theta, phi)
        sx += weight * x
        sy += weight * y
        sz += weight * z
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError("no valid rows available for diagnostic recentering centroid")
    return angles_from_unit((sx / total_weight, sy / total_weight, sz / total_weight))


def project_rows_for_diagnostic_center(
    rows: list[dict[str, str]],
    *,
    theta0: float,
    phi0: float,
    fov_deg: float,
    nx: int,
    ny: int,
) -> list[dict[str, str]]:
    c, e_x, e_y = camera_basis(theta0, phi0)
    extent = math.tan(0.5 * math.radians(fov_deg))
    projected: list[dict[str, str]] = []
    for row in rows:
        theta = as_float(row, "observer_crossing_theta_rad")
        phi = as_float(row, "observer_crossing_phi_rad")
        if theta is None or phi is None:
            continue
        n = unit_from_angles(theta, phi)
        denom = dot(n, c)
        out = dict(row)
        out["inside_fov"] = "false"
        out["pixel_x"] = ""
        out["pixel_y"] = ""
        out["camera_x"] = ""
        out["camera_y"] = ""
        out["projection_status"] = "behind_camera_plane"
        if denom > 0.0:
            camera_x = dot(n, e_x) / denom
            camera_y = dot(n, e_y) / denom
            out["camera_x"] = f"{camera_x:.17g}"
            out["camera_y"] = f"{camera_y:.17g}"
            inside = abs(camera_x) <= extent and abs(camera_y) <= extent
            out["inside_fov"] = "true" if inside else "false"
            out["projection_status"] = "inside_fov" if inside else "outside_fov"
            if inside:
                u = 0.5 * (camera_x / extent + 1.0)
                v = 0.5 * (1.0 - camera_y / extent)
                pixel_x = math.floor(u * nx)
                pixel_y = math.floor(v * ny)
                if pixel_x == nx:
                    pixel_x = nx - 1
                if pixel_y == ny:
                    pixel_y = ny - 1
                out["pixel_x"] = str(pixel_x)
                out["pixel_y"] = str(pixel_y)
        projected.append(out)
    return projected


def flatten_map(data: list[list[float]]) -> list[float]:
    return [value for line in data for value in line]


def morphology_metrics(
    maps: dict[str, list[list[float]]],
    *,
    nx: int,
    ny: int,
    inside: list[dict[str, str]],
    valid: list[dict[str, str]],
) -> dict[str, Any]:
    counts = maps["counts"]
    input_energy = maps["input_energy"]
    observed_energy = maps["observed_energy"]
    active = [(x, y) for y, line in enumerate(counts) for x, value in enumerate(line) if value > 0.0]
    total_count = sum(flatten_map(counts))
    brightest_count = max(flatten_map(counts), default=0.0)
    observed_total = sum(flatten_map(observed_energy))
    center_x = 0.0
    center_y = 0.0
    if observed_total > 0.0:
        center_x = sum(x * observed_energy[y][x] for y in range(ny) for x in range(nx)) / observed_total
        center_y = sum(y * observed_energy[y][x] for y in range(ny) for x in range(nx)) / observed_total

    redshifts = [value for row in valid if (value := as_float(row, "redshift_factor")) is not None]
    mean_redshift = sum(redshifts) / len(redshifts) if redshifts else 0.0
    redshift_std = (
        math.sqrt(sum((value - mean_redshift) ** 2 for value in redshifts) / len(redshifts))
        if redshifts else 0.0
    )

    warnings: list[str] = []
    if not active:
        warnings.append("no_active_pixels")
    if len(active) <= 1 and inside:
        warnings.append("distribution_does_not_occupy_multiple_pixels")
    if total_count > 0.0 and brightest_count / total_count > 0.5:
        warnings.append("single_pixel_contains_more_than_half_of_valid_inside_fov_photons")
    if any(as_float(row, key) is None for row in inside for key in ["input_energy_gev", "observed_energy_gev", "redshift_factor"]):
        warnings.append("nan_or_nonfinite_values_present")
    if any(value > 0.0 for value in flatten_map(maps["negative_energy"])):
        warnings.append("negative_energy_values_present")

    return {
        "n_pixels_active": len(active),
        "n_pixels_total": nx * ny,
        "active_fraction": len(active) / max(nx * ny, 1),
        "brightest_pixel_count": brightest_count,
        "brightest_pixel_input_energy": max(flatten_map(input_energy), default=0.0),
        "brightest_pixel_observed_energy": max(flatten_map(observed_energy), default=0.0),
        "mean_redshift": mean_redshift,
        "redshift_std": redshift_std,
        "center_of_light_x": center_x,
        "center_of_light_y": center_y,
        "warnings": warnings,
    }


def setup_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_map(
    plt,
    data: list[list[float]],
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    diagnostic_context: str = "Diagnostic only",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(data, origin="upper", interpolation="nearest", aspect="equal")
    ax.set_title(f"{diagnostic_context}\n{title}\n{PHYSICS_LABEL}")
    ax.set_xlabel("pixel_x")
    ax.set_ylabel("pixel_y")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_histogram(plt, rows: list[dict[str, str]], path: Path) -> None:
    redshifts = [value for row in rows if (value := as_float(row, "redshift_factor")) is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(redshifts, bins=min(40, max(5, int(math.sqrt(max(len(redshifts), 1))))), color="#2f6f6d", edgecolor="black")
    ax.set_title(f"Diagnostic only\nRedshift factor histogram\n{PHYSICS_LABEL}")
    ax.set_xlabel("redshift_factor")
    ax.set_ylabel("valid photon count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scatter(plt, rows: list[dict[str, str]], path: Path) -> None:
    points = [
        (input_energy, observed_energy)
        for row in rows
        if (input_energy := as_float(row, "input_energy_gev")) is not None
        and (observed_energy := as_float(row, "observed_energy_gev")) is not None
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(xs, ys, s=10, alpha=0.45, color="#5b4b8a")
    if xs and ys:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="1:1 reference")
        ax.legend(loc="best")
    ax.set_title(f"Diagnostic only\nInput vs observed photon energy\n{PHYSICS_LABEL}")
    ax.set_xlabel("input_energy_gev")
    ax.set_ylabel("observed_energy_gev")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_lines(
    *,
    input_path: Path,
    rows: list[dict[str, str]],
    valid: list[dict[str, str]],
    inside: list[dict[str, str]],
    metrics: dict[str, Any],
) -> list[str]:
    input_total = sum(as_float(row, "input_energy_gev") or 0.0 for row in inside)
    observed_total = sum(as_float(row, "observed_energy_gev") or 0.0 for row in inside)
    lines = [
        "# Photon Observer Camera Diagnostic Plots",
        "",
        "These are diagnostic plots only; they are not paper-ready figures.",
        "",
        f"Physical label: `{PHYSICS_LABEL}`.",
        "",
        f"- input_csv: `{input_path}`",
        f"- n_rows: `{len(rows)}`",
        f"- n_valid_redshift_rows: `{len(valid)}`",
        f"- n_valid_inside_fov_rows: `{len(inside)}`",
        f"- total_input_energy_inside_fov_gev: `{input_total:.12g}`",
        f"- total_observed_energy_inside_fov_gev: `{observed_total:.12g}`",
        f"- n_pixels_active: `{metrics['n_pixels_active']}`",
        f"- n_pixels_total: `{metrics['n_pixels_total']}`",
        f"- active_fraction: `{metrics['active_fraction']:.12g}`",
        f"- brightest_pixel_count: `{metrics['brightest_pixel_count']:.12g}`",
        f"- brightest_pixel_input_energy: `{metrics['brightest_pixel_input_energy']:.12g}`",
        f"- brightest_pixel_observed_energy: `{metrics['brightest_pixel_observed_energy']:.12g}`",
        f"- mean_redshift: `{metrics['mean_redshift']:.12g}`",
        f"- redshift_std: `{metrics['redshift_std']:.12g}`",
        f"- center_of_light_x: `{metrics['center_of_light_x']:.12g}`",
        f"- center_of_light_y: `{metrics['center_of_light_y']:.12g}`",
        "",
        "Morphology warnings:",
    ]
    if metrics["warnings"]:
        lines.extend(f"- `{warning}`" for warning in metrics["warnings"])
    else:
        lines.append("- `none`")
    lines.extend(["", "Generated diagnostic products:"])
    lines.extend(f"- `{name}`" for name in OUTPUT_FILENAMES)
    return lines


def write_summary(
    path: Path,
    *,
    input_path: Path,
    rows: list[dict[str, str]],
    valid: list[dict[str, str]],
    inside: list[dict[str, str]],
    metrics: dict[str, Any],
) -> None:
    path.write_text(
        "\n".join(summary_lines(input_path=input_path, rows=rows, valid=valid, inside=inside, metrics=metrics)) + "\n",
        encoding="utf-8",
    )


def brightest_fraction(metrics: dict[str, Any], rows: list[dict[str, str]]) -> float:
    total = max(float(len(rows)), 1.0)
    return float(metrics["brightest_pixel_count"]) / total


def write_projection_stats(path: Path, rows: list[dict[str, str]], *, fov_deg: float) -> None:
    extent = math.tan(0.5 * math.radians(fov_deg))
    inside = valid_inside_fov_rows(rows)
    camera_x = [value for row in inside if (value := as_float(row, "camera_x")) is not None]
    camera_y = [value for row in inside if (value := as_float(row, "camera_y")) is not None]
    u_values = [0.5 * (value / extent + 1.0) for value in camera_x]
    v_values = [0.5 * (1.0 - value / extent) for value in camera_y]

    def percentile(values: list[float], q: float) -> float | None:
        values = sorted(values)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * q / 100.0
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return values[low]
        return values[low] * (high - position) + values[high] * (position - low)

    metrics: list[tuple[str, float | int | None]] = [
        ("n_recentered_inside_fov_rows", len(inside)),
        ("camera_x_min", min(camera_x) if camera_x else None),
        ("camera_x_p05", percentile(camera_x, 5)),
        ("camera_x_p50", percentile(camera_x, 50)),
        ("camera_x_mean", sum(camera_x) / len(camera_x) if camera_x else None),
        ("camera_x_p95", percentile(camera_x, 95)),
        ("camera_x_max", max(camera_x) if camera_x else None),
        ("camera_y_min", min(camera_y) if camera_y else None),
        ("camera_y_p05", percentile(camera_y, 5)),
        ("camera_y_p50", percentile(camera_y, 50)),
        ("camera_y_mean", sum(camera_y) / len(camera_y) if camera_y else None),
        ("camera_y_p95", percentile(camera_y, 95)),
        ("camera_y_max", max(camera_y) if camera_y else None),
        ("u_min", min(u_values) if u_values else None),
        ("u_p50", percentile(u_values, 50)),
        ("u_max", max(u_values) if u_values else None),
        ("v_min", min(v_values) if v_values else None),
        ("v_p50", percentile(v_values, 50)),
        ("v_max", max(v_values) if v_values else None),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics:
            if value is None:
                writer.writerow([key, ""])
            elif isinstance(value, float):
                writer.writerow([key, f"{value:.17g}"])
            else:
                writer.writerow([key, value])


def write_recentered_summary(
    path: Path,
    *,
    default_theta: float,
    default_phi: float,
    unweighted_theta: float,
    unweighted_phi: float,
    weighted_theta: float,
    weighted_phi: float,
    default_metrics: dict[str, Any],
    default_inside: list[dict[str, str]],
    recentered_metrics: dict[str, Any],
    recentered_inside: list[dict[str, str]],
) -> None:
    lines = [
        "# Photon Observer Camera Recentered Diagnostic Projection",
        "",
        "Diagnostic recentered projection only; this is not the default camera pointing.",
        "",
        "These products are diagnostic only; they are not paper-ready figures.",
        "",
        f"Physical label: `{PHYSICS_LABEL}`.",
        "",
        "- Recenter source rows: `redshift_status = valid` with finite observer-sphere angles.",
        "- Recenter map rows: valid rows that are inside the diagnostic recentered FOV.",
        "- Official `photon_observer_camera.csv` and `photon_observer_camera_redshift.csv` are not modified.",
        "",
        f"- default_center_theta_rad: `{default_theta:.12g}`",
        f"- default_center_phi_rad: `{default_phi:.12g}`",
        f"- unweighted_centroid_theta_rad: `{unweighted_theta:.12g}`",
        f"- unweighted_centroid_phi_rad: `{unweighted_phi:.12g}`",
        f"- energy_weighted_centroid_theta_rad: `{weighted_theta:.12g}`",
        f"- energy_weighted_centroid_phi_rad: `{weighted_phi:.12g}`",
        "",
        "Before/after morphology:",
        f"- brightest_pixel_fraction_before: `{brightest_fraction(default_metrics, default_inside):.12g}`",
        f"- brightest_pixel_fraction_after: `{brightest_fraction(recentered_metrics, recentered_inside):.12g}`",
        f"- active_fraction_before: `{default_metrics['active_fraction']:.12g}`",
        f"- active_fraction_after: `{recentered_metrics['active_fraction']:.12g}`",
        f"- center_of_light_x_before: `{default_metrics['center_of_light_x']:.12g}`",
        f"- center_of_light_y_before: `{default_metrics['center_of_light_y']:.12g}`",
        f"- center_of_light_x_after: `{recentered_metrics['center_of_light_x']:.12g}`",
        f"- center_of_light_y_after: `{recentered_metrics['center_of_light_y']:.12g}`",
        "",
        "Generated recentered diagnostic products:",
    ]
    lines.extend(f"- `{name}`" for name in RECENTERED_OUTPUT_FILENAMES)
    lines.append("- `photon_diagnostic_recentered_projection_stats.csv`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_recentered_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    plt: Any,
    fieldnames: list[str],
    valid: list[dict[str, str]],
    default_inside: list[dict[str, str]],
    default_metrics: dict[str, Any],
    nx: int,
    ny: int,
) -> list[Path]:
    require_recenter_fields(fieldnames)
    config = read_camera_projection_config(input_path)
    fov_deg = float(config["photon_camera_fov_deg"])
    default_theta = math.radians(float(config["photon_camera_center_theta_deg"]))
    default_phi = float(config["photon_camera_center_phi_rad"])
    unweighted_theta, unweighted_phi = angular_centroid(valid)
    weighted_theta, weighted_phi = angular_centroid(valid, weight_key="observed_energy_gev")

    recentered_rows = project_rows_for_diagnostic_center(
        valid,
        theta0=weighted_theta,
        phi0=weighted_phi,
        fov_deg=fov_deg,
        nx=nx,
        ny=ny,
    )
    recentered_inside = valid_inside_fov_rows(recentered_rows)
    recentered_maps = build_maps(recentered_inside, nx, ny)
    recentered_metrics = morphology_metrics(
        recentered_maps,
        nx=nx,
        ny=ny,
        inside=recentered_inside,
        valid=valid_rows(recentered_rows),
    )
    context = "Diagnostic recentered projection\nnot default camera pointing"
    products = [
        output_dir / "photon_diagnostic_recentered_counts_map.png",
        output_dir / "photon_diagnostic_recentered_observed_energy_map.png",
        output_dir / "photon_diagnostic_recentered_mean_redshift_map.png",
    ]
    save_map(
        plt,
        recentered_maps["counts"],
        products[0],
        title="Recentered photon count by pixel",
        colorbar_label="N_photons(pixel)",
        diagnostic_context=context,
    )
    save_map(
        plt,
        recentered_maps["observed_energy"],
        products[1],
        title="Recentered observed photon energy by pixel",
        colorbar_label="sum observed_energy_gev",
        diagnostic_context=context,
    )
    save_map(
        plt,
        recentered_maps["mean_redshift"],
        products[2],
        title="Recentered mean redshift factor by pixel",
        colorbar_label="mean(redshift_factor)",
        diagnostic_context=context,
    )

    stats = output_dir / "photon_diagnostic_recentered_projection_stats.csv"
    write_projection_stats(stats, recentered_rows, fov_deg=fov_deg)
    summary = output_dir / "photon_diagnostic_recentered_summary.md"
    write_recentered_summary(
        summary,
        default_theta=default_theta,
        default_phi=default_phi,
        unweighted_theta=unweighted_theta,
        unweighted_phi=unweighted_phi,
        weighted_theta=weighted_theta,
        weighted_phi=weighted_phi,
        default_metrics=default_metrics,
        default_inside=default_inside,
        recentered_metrics=recentered_metrics,
        recentered_inside=recentered_inside,
    )
    products.extend([summary, stats])
    return products


def build_diagnostics(input_path: Path, output_dir: Path, *, diagnostic_recenter: bool = False) -> list[Path]:
    fieldnames, rows = read_csv_rows(input_path)
    require_fields(fieldnames)
    valid = valid_rows(rows)
    inside = valid_inside_fov_rows(rows)
    nx, ny = read_camera_shape(input_path, inside)
    maps = build_maps(inside, nx, ny)
    metrics = morphology_metrics(maps, nx=nx, ny=ny, inside=inside, valid=valid)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(output_dir)
    products = [
        output_dir / "photon_diagnostic_input_energy_map.png",
        output_dir / "photon_diagnostic_observed_energy_map.png",
        output_dir / "photon_diagnostic_counts_map.png",
        output_dir / "photon_diagnostic_mean_redshift_map.png",
        output_dir / "photon_diagnostic_valid_photon_density_map.png",
        output_dir / "photon_diagnostic_mean_observed_energy_map.png",
        output_dir / "photon_diagnostic_redshift_histogram.png",
        output_dir / "photon_diagnostic_input_vs_observed_energy.png",
    ]
    save_map(plt, maps["input_energy"], products[0], title="Input photon energy by pixel", colorbar_label="sum input_energy_gev")
    save_map(plt, maps["observed_energy"], products[1], title="Observed photon energy by pixel", colorbar_label="sum observed_energy_gev")
    save_map(plt, maps["counts"], products[2], title="Photon count by pixel", colorbar_label="N_photons(pixel)")
    save_map(plt, maps["mean_redshift"], products[3], title="Mean redshift factor by pixel", colorbar_label="mean(redshift_factor)")
    save_map(plt, maps["valid_density"], products[4], title="Valid photon density by pixel", colorbar_label="N_valid_redshift(pixel)")
    save_map(plt, maps["mean_observed_energy"], products[5], title="Mean observed photon energy by pixel", colorbar_label="mean observed_energy_gev")
    save_histogram(plt, valid, products[6])
    save_scatter(plt, valid, products[7])

    summary = output_dir / "photon_diagnostic_summary.md"
    morphology = output_dir / "photon_diagnostic_morphology_summary.md"
    write_summary(summary, input_path=input_path, rows=rows, valid=valid, inside=inside, metrics=metrics)
    write_summary(morphology, input_path=input_path, rows=rows, valid=valid, inside=inside, metrics=metrics)
    products.extend([summary, morphology])
    if diagnostic_recenter:
        products.extend(
            build_recentered_diagnostics(
                input_path=input_path,
                output_dir=output_dir,
                plt=plt,
                fieldnames=fieldnames,
                valid=valid,
                default_inside=inside,
                default_metrics=metrics,
                nx=nx,
                ny=ny,
            )
        )
    return products


def main() -> int:
    args = parse_args()
    forbidden = [name for name in OUTPUT_FILENAMES if "paper" in name.lower() or "final" in name.lower()]
    if forbidden:
        raise RuntimeError(f"diagnostic output filenames must not look paper/final: {forbidden}")
    try:
        products = build_diagnostics(args.input, args.output_dir, diagnostic_recenter=args.diagnostic_recenter)
    except Exception as exc:
        print(f"Failed to build photon observer diagnostic plots: {exc}", file=sys.stderr)
        return 2
    print("photon_observer_diagnostic_products")
    for product in products:
        print(product)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
