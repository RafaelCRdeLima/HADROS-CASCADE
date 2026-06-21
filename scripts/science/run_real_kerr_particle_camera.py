#!/usr/bin/env python3
"""Run the HADROS Kerr particle-ray association camera backend."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output/science/powheg_pythia_geant4_resumable/geant4_ready_particles.jsonl"
DEFAULT_OUTPUT = ROOT / "output/science/real_kerr_particle_camera"
APP = ROOT / "build/compute_kerr_particle_camera"
BLOCKED = "PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_GLOBAL_POSITION"
BLOCKED_ASSOCIATION = "PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_ASSOCIATION_CRITERIA"
PARTIAL = "PARTICLE_RAY_ASSOCIATION_CAMERA_PARTIAL_SAMPLED_INTERACTIONS"
VALIDATED = "PARTICLE_RAY_ASSOCIATION_CAMERA_VALIDATED"


def run_backend(args: argparse.Namespace, output_dir: Path, *, aspin: float | None = None, r_obs: float | None = None, r_max: float | None = None, step: float | None = None, theta: float | None = None, fov: float | None = None) -> tuple[int, dict[str, Any]]:
    command = [
        str(args.app),
        str(args.input),
        str(output_dir),
        str(args.aspin if aspin is None else aspin),
        str(args.camera_r_obs_rg if r_obs is None else r_obs),
        str(args.camera_theta_deg if theta is None else theta),
        str(args.camera_fov_deg if fov is None else fov),
        str(args.camera_nx),
        str(args.camera_ny),
        str(args.camera_r_max_rg if r_max is None else r_max),
        str(args.camera_step if step is None else step),
        str(args.spatial_tolerance_rg),
        str(args.angular_tolerance_deg),
        str(args.association_mode),
        str(args.camera_naming_mode),
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    payload: dict[str, Any] = {"stdout": proc.stdout, "stderr": proc.stderr, "command": command}
    try:
        payload.update(json.loads(proc.stdout))
    except json.JSONDecodeError:
        pass
    (output_dir / "backend_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "backend_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return proc.returncode, payload


def read_validation(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def validation_path(output_dir: Path, camera_naming_mode: str) -> Path:
    primary = (
        output_dir / "real_kerr_particle_camera_validation.csv"
        if camera_naming_mode == "legacy"
        else output_dir / "particle_ray_association_camera_validation.csv"
    )
    if primary.exists():
        return primary
    fallback = (
        output_dir / "particle_ray_association_camera_validation.csv"
        if camera_naming_mode == "legacy"
        else output_dir / "real_kerr_particle_camera_validation.csv"
    )
    return fallback


def write_sensitivity(args: argparse.Namespace) -> None:
    cases = [
        ("ASPIN", "0.0", {"aspin": 0.0}),
        ("ASPIN", "0.8", {"aspin": 0.8}),
        ("ASPIN", "0.98", {"aspin": 0.98}),
        ("CAM_R_OBS_RG", "60", {"r_obs": 60.0}),
        ("CAM_R_OBS_RG", "1000", {"r_obs": 1000.0}),
        ("CAM_R_MAX_RG", "80", {"r_max": 80.0}),
        ("CAM_R_MAX_RG", "500", {"r_max": 500.0}),
        ("CAM_STEP", "0.1", {"step": 0.1}),
        ("CAM_STEP", "1.0", {"step": 1.0}),
        ("CAM_THETA_DEG", "30", {"theta": 30.0}),
        ("CAM_THETA_DEG", "70", {"theta": 70.0}),
        ("CAM_FOV_DEG", "30", {"fov": 30.0}),
        ("CAM_FOV_DEG", "90", {"fov": 90.0}),
    ]
    rows = []
    for parameter, value, overrides in cases:
        case_dir = args.output_dir / "sensitivity" / f"{parameter}_{value}".replace(".", "p")
        case_dir.mkdir(parents=True, exist_ok=True)
        _code, payload = run_backend(args, case_dir, **overrides)
        validation = read_validation(validation_path(case_dir, args.camera_naming_mode))
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "status": validation.get("status", payload.get("status", "")),
                "trace_pixel_calls": validation.get("trace_pixel_calls", payload.get("trace_pixel_calls", "")),
                "ray_sample_count": validation.get("ray_sample_count", payload.get("ray_sample_count", "")),
                "ray_hash": validation.get("ray_hash", ""),
                "observed_rows": validation.get("observed_rows", "0"),
                "association_mode": validation.get("association_mode", ""),
                "camera_naming_mode": validation.get("camera_naming_mode", args.camera_naming_mode),
                "spatial_tolerance_rg": validation.get("spatial_tolerance_rg", ""),
                "angular_tolerance_deg": validation.get("angular_tolerance_deg", ""),
                "image_status": (
                    "SCIENTIFIC_IMAGE_PARTIAL_SAMPLED_INTERACTIONS"
                    if validation.get("status") == PARTIAL
                    else ("ASSOCIATION_CAMERA_VALIDATED" if validation.get("status") == VALIDATED else "NO_ASSOCIATION_CAMERA_ROWS")
                ),
            }
        )
    fields = [
        "parameter",
        "value",
        "status",
        "trace_pixel_calls",
        "ray_sample_count",
        "ray_hash",
        "observed_rows",
        "association_mode",
        "camera_naming_mode",
        "spatial_tolerance_rg",
        "angular_tolerance_deg",
        "image_status",
    ]
    with (args.output_dir / "camera_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "global_position_camera_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_wrapper_report(args: argparse.Namespace, payload: dict[str, Any], returncode: int) -> None:
    validation = read_validation(validation_path(args.output_dir, args.camera_naming_mode))
    status = validation.get("status", payload.get("status", BLOCKED))
    legacy_outputs = (
        "`observed_particles_by_pixel.csv`, `observed_particles_by_pixel.jsonl`"
        if args.camera_naming_mode in {"both", "legacy"}
        else "`not written in this camera_naming_mode`"
    )
    lines = [
        "# Particle Ray Association Camera Wrapper",
        "",
        f"Status: `{status}`.",
        "",
        "- particle_camera_backend: `particle_ray_association_camera`",
        "- camera_backend: `PARTICLE_RAY_ASSOCIATION_CAMERA`",
        f"- legacy_compatible_outputs: {legacy_outputs}",
        f"- backend_returncode: `{returncode}`",
        f"- input_particles: `{validation.get('input_particles', '')}`",
        f"- rows_with_position: `{validation.get('rows_with_position', '')}`",
        f"- trace_pixel_calls: `{validation.get('trace_pixel_calls', '')}`",
        f"- ray_sample_count: `{validation.get('ray_sample_count', '')}`",
        f"- association_mode: `{validation.get('association_mode', '')}`",
        f"- camera_naming_mode: `{validation.get('camera_naming_mode', '')}`",
        f"- camera_physical_interpretation: `{validation.get('camera_physical_interpretation', 'particle-ray association / cascade origin map')}`",
        f"- camera_is_full_observational_transport: `{validation.get('camera_is_full_observational_transport', 'false')}`",
        f"- camera_limitation: `{validation.get('camera_limitation', 'secondary particles are associated with Kerr rays by spatial/angular criteria; they are not propagated to the distant observer')}`",
        f"- full_transport_available: `{validation.get('full_transport_available', 'false')}`",
        f"- spatial_tolerance_rg: `{validation.get('spatial_tolerance_rg', '')}`",
        f"- angular_tolerance_deg: `{validation.get('angular_tolerance_deg', '')}`",
        f"- rejected_missing_direction: `{validation.get('rejected_missing_direction', '')}`",
        f"- rejected_angular_tolerance: `{validation.get('rejected_angular_tolerance', '')}`",
        f"- blocked_reason: `{validation.get('blocked_reason', '')}`",
        "",
        "`direction_misalignment_deg = NaN` means the direction was not calculable and is not accepted as aligned.",
        "This is a spatial+angular particle/ray association product, not full particle transport to a physical observer.",
        "Legacy `observed_particles_by_pixel.*` names are compatibility outputs only and do not imply full observation.",
    ]
    (args.output_dir / "real_kerr_particle_camera_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_maps(args: argparse.Namespace, status: str = BLOCKED) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros((args.camera_ny, args.camera_nx), dtype=float)
    names = [
        "particle_ray_association_gamma_map.png",
        "particle_ray_association_electromagnetic_map.png",
        "particle_ray_association_hadronic_map.png",
        "particle_ray_association_neutrino_map.png",
    ]
    title = (
        "PARTIAL: local-to-global approximation"
        if status == PARTIAL
        else ("VALIDATED: particle-ray association" if status == VALIDATED else "BLOCKED: no accepted association")
    )
    for name in names:
        fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
        im = ax.imshow(zeros, origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("pixel_x")
        ax.set_ylabel("pixel_y")
        fig.colorbar(im, ax=ax, label="weighted energy [GeV]")
        fig.savefig(plots / name, dpi=140)
        plt.close(fig)
    rgb = np.dstack([zeros, zeros, zeros])
    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("pixel_x")
    ax.set_ylabel("pixel_y")
    fig.savefig(plots / "particle_ray_association_rgb_channels.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(args.output_dir / "particle_ray_association_maps.npz", gamma=zeros, electromagnetic=zeros, hadronic=zeros, neutrino=zeros)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--app", type=Path, default=APP)
    parser.add_argument("--camera-nx", type=int, required=True)
    parser.add_argument("--camera-ny", type=int, required=True)
    parser.add_argument("--camera-fov-deg", type=float, required=True)
    parser.add_argument("--camera-theta-deg", type=float, required=True)
    parser.add_argument("--camera-r-obs-rg", type=float, required=True)
    parser.add_argument("--camera-r-max-rg", type=float, required=True)
    parser.add_argument("--camera-step", type=float, required=True)
    parser.add_argument("--spatial-tolerance-rg", type=float, required=True)
    parser.add_argument("--angular-tolerance-deg", type=float, required=True)
    parser.add_argument(
        "--association-mode",
        choices=["spatial_only", "spatial_plus_direction", "full_transport"],
        required=True,
    )
    parser.add_argument(
        "--camera-naming-mode",
        choices=["both", "semantic", "legacy"],
        required=True,
    )
    parser.add_argument("--aspin", type=float, required=True)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.association_mode == "full_transport":
        raise SystemExit("full_transport is not implemented yet")
    args.input = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        subprocess.run(["make", "compute_kerr_particle_camera"], cwd=ROOT, check=True)
    returncode, payload = run_backend(args, args.output_dir)
    write_sensitivity(args)
    write_wrapper_report(args, payload, returncode)
    status = str(payload.get("status", BLOCKED))
    write_blocked_maps(args, status)
    print(json.dumps({"status": status, "backend_returncode": returncode}, indent=2, sort_keys=True))
    return 0 if status in {VALIDATED, PARTIAL} else 2


if __name__ == "__main__":
    raise SystemExit(main())
