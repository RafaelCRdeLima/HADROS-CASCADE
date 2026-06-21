#!/usr/bin/env python3
"""DEPRECATED / DEBUG ONLY: backward-camera particle-channel proxy prototype.

This file is not part of the final scientific HADROS chain.
Do not use for scientific production.

This is camera-first: config-web/HADROS camera pixels launch backward Kerr
null rays, ray samples through the torus are weighted by DIS optical depth, and
an explicit local-response proxy is accumulated per pixel. It does not use
forward packet projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


CHANNEL_COLUMNS = {
    "gamma": "gamma_weighted_energy_proxy",
    "electromagnetic": "electromagnetic_weighted_energy_proxy",
    "hadronic": "hadronic_weighted_energy_proxy",
    "pion_charged": "pion_charged_weighted_energy_proxy",
    "deposited": "deposited_weighted_energy_proxy",
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grid(rows: list[dict[str, str]], column: str, ny: int, nx: int) -> np.ndarray:
    arr = np.zeros((ny, nx), dtype=float)
    for row in rows:
        i = int(finite(row.get("pixel_i")))
        j = int(finite(row.get("pixel_j")))
        if 0 <= i < nx and 0 <= j < ny:
            arr[j, i] = finite(row.get(column))
    return arr


def save_plot(path: Path, image: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.colorbar(im, ax=ax, label="weighted-energy proxy")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rgb(path: Path, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def norm(x: np.ndarray) -> np.ndarray:
        m = float(np.max(x))
        return x / m if m > 0 else x

    rgb = np.dstack([norm(red), norm(green), norm(blue)])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    ax.set_title("Backward camera RGB proxy: R=hadronic, G=EM, B=deposited")
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_app(args: argparse.Namespace, csv_path: Path, summary_path: Path) -> None:
    exe = Path("build/compute_backward_camera_particle_channels")
    if not exe.exists() or args.rebuild:
        subprocess.run(["make", "compute_backward_camera_particle_channels"], check=True)
    command = [
        str(exe),
        "--output-csv",
        str(csv_path),
        "--output-summary",
        str(summary_path),
        "--sigma-table",
        str(args.sigma_table),
        "--dis-model",
        args.dis_model,
        "--energy-gev",
        str(args.energy_gev),
        "--spin",
        str(args.spin),
        "--mbh-msun",
        str(args.mbh_msun),
        "--camera-r-obs-rg",
        str(args.camera_r_obs_rg),
        "--camera-theta-deg",
        str(args.camera_theta_deg),
        "--camera-fov-deg",
        str(args.camera_fov_deg),
        "--camera-nx",
        str(args.camera_nx),
        "--camera-ny",
        str(args.camera_ny),
        "--camera-r-max-rg",
        str(args.camera_r_max_rg),
        "--camera-step",
        str(args.camera_step),
        "--torus-rho0",
        str(args.torus_rho0),
        "--torus-r0-rg",
        str(args.torus_r0_rg),
        "--torus-sigma-rg",
        str(args.torus_sigma_rg),
        "--torus-h-over-r",
        str(args.torus_h_over_r),
        "--density-profile",
        args.density_profile,
        "--torus-radial-power",
        str(args.torus_radial_power),
        "--funnel-depletion",
        str(args.funnel_depletion),
        "--funnel-theta-deg",
        str(args.funnel_theta_deg),
        "--envelope-rho0",
        str(args.envelope_rho0),
        "--envelope-alpha",
        str(args.envelope_alpha),
        "--torus-r-min-rg",
        str(args.torus_r_min_rg),
        "--torus-r-max-rg",
        str(args.torus_r_max_rg),
        "--rho-floor",
        str(args.rho_floor),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/science/backward_camera_particle_channels"))
    parser.add_argument("--sigma-table", type=Path, default=Path("data/sigma/sigma_nuN_CC_GBW.dat"))
    parser.add_argument("--dis-model", default="GBW")
    parser.add_argument("--energy-gev", type=float, default=1.0e9)
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--mbh-msun", type=float, default=3.0)
    parser.add_argument("--camera-r-obs-rg", type=float, default=80.0)
    parser.add_argument("--camera-theta-deg", type=float, default=70.0)
    parser.add_argument("--camera-fov-deg", type=float, default=60.0)
    parser.add_argument("--camera-nx", type=int, default=32)
    parser.add_argument("--camera-ny", type=int, default=32)
    parser.add_argument("--camera-r-max-rg", type=float, default=120.0)
    parser.add_argument("--camera-step", type=float, default=0.02)
    parser.add_argument("--torus-rho0", type=float, default=1.0e-2)
    parser.add_argument("--torus-r0-rg", type=float, default=10.0)
    parser.add_argument("--torus-sigma-rg", type=float, default=5.0)
    parser.add_argument("--torus-h-over-r", type=float, default=0.25)
    parser.add_argument("--density-profile", default="gaussian")
    parser.add_argument("--torus-radial-power", type=float, default=2.0)
    parser.add_argument("--funnel-depletion", type=float, default=0.0)
    parser.add_argument("--funnel-theta-deg", type=float, default=15.0)
    parser.add_argument("--envelope-rho0", type=float, default=0.0)
    parser.add_argument("--envelope-alpha", type=float, default=2.5)
    parser.add_argument("--torus-r-min-rg", type=float, default=4.0)
    parser.add_argument("--torus-r-max-rg", type=float, default=60.0)
    parser.add_argument("--rho-floor", type=float, default=1.0e-99)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    csv_path = args.output_dir / "backward_camera_particle_channels.csv"
    summary_path = args.output_dir / "backward_camera_particle_summary.md"
    run_app(args, csv_path, summary_path)

    rows = read_rows(csv_path)
    nx = int(args.camera_nx)
    ny = int(args.camera_ny)
    images = {name: grid(rows, column, ny, nx) for name, column in CHANNEL_COLUMNS.items()}
    tau = grid(rows, "tau", ny, nx)
    inside = grid(rows, "n_inside_torus", ny, nx)
    metadata = {
        "status": "BACKWARD_CAMERA_LOCAL_RESPONSE_PROTOTYPE",
        "proxy_status": "backward_camera_local_response_proxy",
        "camera_pipeline": "backward_kerr_ray_tracing",
        "uses_forward_packets": False,
        "dis_model": args.dis_model,
        "sigma_table": str(args.sigma_table),
        "energy_gev": args.energy_gev,
        "camera_nx": nx,
        "camera_ny": ny,
        "camera_fov_deg": args.camera_fov_deg,
        "camera_theta_deg": args.camera_theta_deg,
        "camera_r_obs_rg": args.camera_r_obs_rg,
        "camera_r_max_rg": args.camera_r_max_rg,
        "camera_step": args.camera_step,
    }
    np.savez(
        args.output_dir / "backward_camera_particle_channels.npz",
        backward_gamma=images["gamma"],
        backward_electromagnetic=images["electromagnetic"],
        backward_hadronic=images["hadronic"],
        backward_pion_charged=images["pion_charged"],
        backward_deposited=images["deposited"],
        tau_map=tau,
        inside_torus_samples=inside,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    save_plot(args.output_dir / "backward_gamma.png", images["gamma"], "gamma weighted-energy proxy")
    save_plot(args.output_dir / "backward_electromagnetic.png", images["electromagnetic"], "electromagnetic weighted-energy proxy")
    save_plot(args.output_dir / "backward_hadronic.png", images["hadronic"], "hadronic weighted-energy proxy")
    save_plot(args.output_dir / "backward_deposited.png", images["deposited"], "deposited weighted-energy proxy")
    save_rgb(args.output_dir / "backward_rgb.png", images["hadronic"], images["electromagnetic"], images["deposited"])
    save_plot(plots / "ray_tau_map.png", tau, "tau accumulated per backward camera pixel")
    save_plot(plots / "ray_torus_intersections.png", inside, "samples inside torus per pixel")

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Python products\n\n")
        handle.write("- npz: `backward_camera_particle_channels.npz`\n")
        handle.write("- plots: `backward_gamma.png`, `backward_electromagnetic.png`, `backward_hadronic.png`, `backward_deposited.png`, `backward_rgb.png`\n")
        handle.write("\nThis remains a camera-selected weighted-energy proxy map, not a physical luminosity image.\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
