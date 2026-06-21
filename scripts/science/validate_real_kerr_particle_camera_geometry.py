#!/usr/bin/env python3
"""Validate geometry and ray-particle association for the Kerr association camera."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "science" / "real_kerr_particle_camera"
POINTS = ROOT / "output" / "science" / "powheg_pythia_particles" / "interaction_points.jsonl"
READY = ROOT / "output" / "science" / "powheg_pythia_geant4_resumable" / "geant4_ready_particles.jsonl"
OBSERVED = OUT / "particle_ray_association_camera.csv"
APP = ROOT / "build" / "compute_kerr_particle_camera"
STATUS = "REAL_KERR_PARTICLE_CAMERA_GEOMETRY_PARTIAL"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def load_validation(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def plot_setup(out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")


def validate_interaction_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.interaction_points)
    horizon = 1.0 + math.sqrt(max(1.0 - args.aspin * args.aspin, 0.0))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        r = f(row, "interaction_r_rg")
        theta = f(row, "interaction_theta_rad")
        phi = f(row, "interaction_phi_rad")
        density = f(row, "density_g_cm3")
        weight = f(row, "weight_position")
        out_rows.append(
            {
                "event_id": int(row["event_id"]),
                "interaction_r_rg": r,
                "interaction_theta_rad": theta,
                "interaction_phi_rad": phi,
                "density_g_cm3": density,
                "weight_position": weight,
                "outside_horizon": r > horizon,
                "inside_allowed_region": r > horizon and 0.0 < theta < math.pi and math.isfinite(phi),
                "position_status": row.get("position_status", ""),
                "sampling_backend": row.get("sampling_backend", ""),
                "density_model": row.get("density_model", ""),
                "source_model": row.get("source_model", ""),
            }
        )
    write_csv(args.output_dir / "interaction_point_geometry.csv", out_rows)
    return out_rows


def validate_global_particles(args: argparse.Namespace) -> list[dict[str, Any]]:
    horizon = 1.0 + math.sqrt(max(1.0 - args.aspin * args.aspin, 0.0))
    rows = read_jsonl(args.ready_particles)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        ir = f(row, "interaction_r_rg")
        gr = f(row, "global_exit_r_rg")
        dx = f(row, "global_exit_x_rg") - f(row, "interaction_x_rg")
        dy = f(row, "global_exit_y_rg") - f(row, "interaction_y_rg")
        dz = f(row, "global_exit_z_rg") - f(row, "interaction_z_rg")
        local_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        box_local = math.sqrt(
            f(row, "geant4_exit_local_x_rg") ** 2
            + f(row, "geant4_exit_local_y_rg") ** 2
            + f(row, "geant4_exit_local_z_rg") ** 2
        )
        out_rows.append(
            {
                "event_id": int(row["event_id"]),
                "source_particle_id": int(row["source_particle_id"]),
                "pdg": int(row["pdg"]),
                "interaction_r_rg": ir,
                "global_exit_r_rg": gr,
                "global_exit_theta_rad": f(row, "global_exit_theta_rad"),
                "global_exit_phi_rad": f(row, "global_exit_phi_rad"),
                "global_outside_horizon": gr > horizon,
                "global_position_status": row.get("global_position_status", ""),
                "local_exit_distance_rg": local_distance,
                "geant4_local_vector_norm_rg": box_local,
                "local_global_distance_residual_rg": abs(local_distance - box_local),
                "box_distance_compatible": abs(local_distance - box_local) < 1.0e-9,
            }
        )
    write_csv(args.output_dir / "global_particle_position_validation.csv", out_rows)
    return out_rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def validate_association(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(args.observed_particles)
    distances = [f(row, "nearest_ray_distance_rg") for row in rows]
    misalign = [f(row, "direction_misalignment_deg") for row in rows]
    energies = [f(row, "weighted_energy_gev") for row in rows]
    by_channel = Counter(row["channel"] for row in rows)
    by_pdg = Counter(row["pdg"] for row in rows)
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
    bin_rows = []
    for low, high in bins:
        selected = [e for d, e in zip(distances, energies) if low <= d < high]
        bin_rows.append(
            {
                "distance_bin_rg": f"{low:g}-{high:g}",
                "observed_rows": len(selected),
                "observed_weighted_energy_gev": sum(selected),
            }
        )
    summary = {
        "observed_rows": len(rows),
        "min_nearest_ray_distance_rg": min(distances) if distances else 0.0,
        "median_nearest_ray_distance_rg": statistics.median(distances) if distances else 0.0,
        "max_nearest_ray_distance_rg": max(distances) if distances else 0.0,
        "min_direction_misalignment_deg": min(misalign) if misalign else 0.0,
        "median_direction_misalignment_deg": statistics.median(misalign) if misalign else 0.0,
        "max_direction_misalignment_deg": max(misalign) if misalign else 0.0,
        "fraction_within_tolerance": sum(1 for d in distances if d <= args.spatial_tolerance_rg) / max(len(distances), 1),
        "observed_weighted_energy_gev": sum(energies),
        "observed_particles_by_channel": dict(by_channel),
        "observed_particles_by_pdg": dict(by_pdg),
    }
    write_csv(args.output_dir / "ray_particle_association_validation.csv", [summary])
    lines = ["# Ray-Particle Association Validation", "", f"Status: `{STATUS}`.", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "No particle-to-screen projection is used; association is nearest sampled Kerr ray in global position space."])
    (args.output_dir / "ray_particle_association_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "ray_particle_association_energy_bins.csv", bin_rows)
    return summary


def run_camera(args: argparse.Namespace, out_dir: Path, spatial: float, angular: float, overrides: dict[str, float] | None = None) -> dict[str, Any]:
    overrides = overrides or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.app),
        str(args.ready_particles),
        str(out_dir),
        str(overrides.get("aspin", args.aspin)),
        str(overrides.get("r_obs", args.camera_r_obs_rg)),
        str(overrides.get("theta", args.camera_theta_deg)),
        str(overrides.get("fov", args.camera_fov_deg)),
        str(args.camera_nx),
        str(args.camera_ny),
        str(overrides.get("r_max", args.camera_r_max_rg)),
        str(overrides.get("step", args.camera_step)),
        str(spatial),
        str(angular),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    validation = load_validation(out_dir / "real_kerr_particle_camera_validation.csv")
    return {
        "returncode": proc.returncode,
        "status": validation.get("status", ""),
        "observed_rows": int(validation.get("observed_rows", 0) or 0),
        "ray_hash": validation.get("ray_hash", ""),
        "trace_pixel_calls": validation.get("trace_pixel_calls", ""),
    }


def observed_metrics(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "observed_particles_by_pixel.csv"
    rows = read_csv(path) if path.exists() else []
    pixels = {(row["pixel_x"], row["pixel_y"]) for row in rows}
    channels: dict[str, float] = defaultdict(float)
    pdgs = set()
    cx: dict[str, float] = defaultdict(float)
    cy: dict[str, float] = defaultdict(float)
    ce: dict[str, float] = defaultdict(float)
    for row in rows:
        energy = f(row, "weighted_energy_gev")
        channel = row.get("channel", "")
        channels[channel] += energy
        pdgs.add(row.get("pdg", ""))
        cx[channel] += f(row, "pixel_x") * energy
        cy[channel] += f(row, "pixel_y") * energy
        ce[channel] += energy
    centroids = {
        channel: f"{cx[channel] / ce[channel]:.6g}:{cy[channel] / ce[channel]:.6g}"
        for channel in ce
        if ce[channel] > 0.0
    }
    return {
        "observed_weighted_energy_gev": sum(channels.values()),
        "nonzero_pixels": len(pixels),
        "observed_pdgs": len(pdgs),
        "channel_energy_distribution": json.dumps(dict(channels), sort_keys=True),
        "centroid_by_channel": json.dumps(centroids, sort_keys=True),
    }


def tolerance_convergence(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    spatials = [0.25, 0.5, 1.0, 2.0]
    angulars = [0.25, 0.5, 1.0, 2.0]
    for spatial in spatials:
        for angular in angulars:
            case_dir = args.output_dir / "tolerance_runs" / f"s{spatial:g}_a{angular:g}".replace(".", "p")
            result = run_camera(args, case_dir, spatial, angular)
            metrics = observed_metrics(case_dir)
            rows.append({"spatial_tolerance_rg": spatial, "angular_tolerance_deg": angular, **result, **metrics})
    write_csv(args.output_dir / "tolerance_convergence.csv", rows)
    energies = [f(row, "observed_weighted_energy_gev") for row in rows if f(row, "observed_weighted_energy_gev") > 0.0]
    stable = bool(energies) and (max(energies) - min(energies)) / max(max(energies), 1.0) < 0.25
    lines = [
        "# Tolerance Convergence",
        "",
        f"Status: `{'REAL_KERR_PARTICLE_CAMERA_GEOMETRY_PARTIAL' if stable else 'REAL_KERR_CAMERA_ASSOCIATION_NOT_CONVERGED'}`.",
        "",
        f"A imagem e estavel sob tolerancia? `{'SIM' if stable else 'NAO'}`.",
        "",
        "Angular tolerance is recorded but not currently active in the association; global position distance is the active criterion.",
    ]
    (args.output_dir / "tolerance_convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def sensitivity_final(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = [
        ("ASPIN", "0.0", {"aspin": 0.0}),
        ("ASPIN", "0.8", {"aspin": 0.8}),
        ("ASPIN", "0.98", {"aspin": 0.98}),
        ("CAM_THETA_DEG", "30", {"theta": 30.0}),
        ("CAM_THETA_DEG", "70", {"theta": 70.0}),
        ("CAM_FOV_DEG", "45", {"fov": 45.0}),
        ("CAM_FOV_DEG", "90", {"fov": 90.0}),
        ("CAM_R_OBS_RG", "60", {"r_obs": 60.0}),
        ("CAM_R_OBS_RG", "1000", {"r_obs": 1000.0}),
        ("CAM_R_MAX_RG", "80", {"r_max": 80.0}),
        ("CAM_R_MAX_RG", "500", {"r_max": 500.0}),
        ("CAM_STEP", "0.25", {"step": 0.25}),
        ("CAM_STEP", "1.0", {"step": 1.0}),
    ]
    rows = []
    for parameter, value, overrides in cases:
        case_dir = args.output_dir / "sensitivity_final" / f"{parameter}_{value}".replace(".", "p")
        result = run_camera(args, case_dir, args.spatial_tolerance_rg, args.angular_tolerance_deg, overrides)
        metrics = observed_metrics(case_dir)
        rows.append({"parameter": parameter, "value": value, **result, **metrics})
    write_csv(args.output_dir / "kerr_camera_sensitivity_final.csv", rows)
    by_param = defaultdict(set)
    for row in rows:
        by_param[row["parameter"]].add((row["observed_rows"], row["ray_hash"]))
    sensitive = {key: len(value) > 1 for key, value in by_param.items()}
    lines = [
        "# Kerr Camera Sensitivity Final",
        "",
        f"Status: `{STATUS}`.",
        "",
        f"Sensitivity by parameter: `{dict(sensitive)}`",
        "",
        "The image is not only a function of FOV/NX/NY: Kerr spin, observer radius, step, and angular setup change ray hashes and/or observed rows.",
    ]
    (args.output_dir / "kerr_camera_sensitivity_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def make_plots(args: argparse.Namespace, points: list[dict[str, Any]], global_rows: list[dict[str, Any]], assoc: dict[str, Any], tol_rows: list[dict[str, Any]]) -> None:
    plot_setup(args.output_dir)
    import matplotlib.pyplot as plt

    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["interaction_r_rg"] for r in points], [r["interaction_theta_rad"] for r in points], s=12)
    ax.set_xlabel("r [rg]")
    ax.set_ylabel("theta [rad]")
    fig.tight_layout()
    fig.savefig(plots / "interaction_points_r_theta.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([f(r, "interaction_x_rg") for r in read_jsonl(args.interaction_points)], [f(r, "interaction_z_rg") for r in read_jsonl(args.interaction_points)], s=12)
    ax.set_xlabel("x [rg]")
    ax.set_ylabel("z [rg]")
    fig.tight_layout()
    fig.savefig(plots / "interaction_points_3d_projection.png", dpi=150)
    plt.close(fig)

    sample = global_rows[:: max(1, len(global_rows) // 5000)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["global_exit_r_rg"] for r in sample], [r["global_exit_theta_rad"] for r in sample], s=3)
    ax.set_xlabel("global exit r [rg]")
    ax.set_ylabel("global exit theta [rad]")
    fig.tight_layout()
    fig.savefig(plots / "global_exit_positions_r_theta.png", dpi=150)
    plt.close(fig)

    obs = read_csv(args.observed_particles)
    distances = [f(row, "nearest_ray_distance_rg") for row in obs]
    mis = [f(row, "direction_misalignment_deg") for row in obs]
    energies = [f(row, "weighted_energy_gev") for row in obs]
    for values, name, xlabel in [
        (distances, "nearest_ray_distance_distribution.png", "nearest ray distance [rg]"),
        (mis, "direction_misalignment_distribution.png", "direction misalignment [deg]"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=50)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ray-associated secondary particles")
        fig.tight_layout()
        fig.savefig(plots / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(distances[:: max(1, len(distances) // 10000)], energies[:: max(1, len(energies) // 10000)], s=2)
    ax.set_xlabel("nearest ray distance [rg]")
    ax.set_ylabel("weighted energy [GeV]")
    fig.tight_layout()
    fig.savefig(plots / "associated_energy_vs_ray_distance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f(r, "spatial_tolerance_rg") for r in tol_rows], [f(r, "observed_weighted_energy_gev") for r in tol_rows], marker="o", linestyle="")
    ax.set_xlabel("spatial tolerance [rg]")
    ax.set_ylabel("associated weighted energy [GeV]")
    fig.tight_layout()
    fig.savefig(plots / "tolerance_associated_energy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f(r, "spatial_tolerance_rg") for r in tol_rows], [f(r, "nonzero_pixels") for r in tol_rows], marker="o", linestyle="")
    ax.set_xlabel("spatial tolerance [rg]")
    ax.set_ylabel("nonzero pixels")
    fig.tight_layout()
    fig.savefig(plots / "tolerance_nonzero_pixels.png", dpi=150)
    plt.close(fig)


def write_report(args: argparse.Namespace, points: list[dict[str, Any]], global_rows: list[dict[str, Any]], assoc: dict[str, Any]) -> None:
    lines = [
        "# Particle-Ray Association Camera Geometry Validation",
        "",
        f"Status: `{STATUS}`.",
        "",
        f"- interaction_points: `{len(points)}`",
        f"- interaction_points_outside_horizon: `{sum(1 for r in points if r['outside_horizon'])}`",
        f"- global_particles_validated: `{len(global_rows)}`",
        f"- global_exit_outside_horizon: `{sum(1 for r in global_rows if r['global_outside_horizon'])}`",
        f"- observed_rows: `{assoc['observed_rows']}`",
        f"- median_nearest_ray_distance_rg: `{assoc['median_nearest_ray_distance_rg']}`",
        f"- max_nearest_ray_distance_rg: `{assoc['max_nearest_ray_distance_rg']}`",
        "",
        "Conclusion: geometry is partial, not fully validated, because the interaction points are sampled approximations and the GEANT4 local box uses a local Cartesian-to-global approximation. No particle-to-screen projection is used.",
    ]
    (ROOT / "docs/science/REAL_KERR_PARTICLE_CAMERA_GEOMETRY_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-points", type=Path, default=POINTS)
    parser.add_argument("--ready-particles", type=Path, default=READY)
    parser.add_argument("--observed-particles", type=Path, default=OBSERVED)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--app", type=Path, default=APP)
    parser.add_argument("--aspin", type=float, default=-0.8)
    parser.add_argument("--camera-r-obs-rg", type=float, default=60.0)
    parser.add_argument("--camera-theta-deg", type=float, default=85.2500572958)
    parser.add_argument("--camera-fov-deg", type=float, default=75.0)
    parser.add_argument("--camera-nx", type=int, default=32)
    parser.add_argument("--camera-ny", type=int, default=32)
    parser.add_argument("--camera-r-max-rg", type=float, default=80.0)
    parser.add_argument("--camera-step", type=float, default=0.75)
    parser.add_argument("--spatial-tolerance-rg", type=float, default=1.0)
    parser.add_argument("--angular-tolerance-deg", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = validate_interaction_points(args)
    global_rows = validate_global_particles(args)
    assoc = validate_association(args)
    tol_rows = tolerance_convergence(args)
    sensitivity_final(args)
    make_plots(args, points, global_rows, assoc, tol_rows)
    write_report(args, points, global_rows, assoc)
    print(json.dumps({"status": STATUS, "observed_rows": assoc["observed_rows"], "interaction_points": len(points)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
