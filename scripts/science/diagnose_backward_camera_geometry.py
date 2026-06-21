#!/usr/bin/env python3
"""DEPRECATED / DEBUG ONLY: diagnose Phase 13 backward-camera prototype geometry.

This file is not part of the final scientific HADROS chain.
Do not use for scientific production.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grid(rows: list[dict[str, str]], column: str) -> np.ndarray:
    nx = max(int(finite(row["pixel_i"])) for row in rows) + 1
    ny = max(int(finite(row["pixel_j"])) for row in rows) + 1
    out = np.zeros((ny, nx), dtype=float)
    for row in rows:
        out[int(finite(row["pixel_j"])), int(finite(row["pixel_i"]))] = finite(row.get(column))
    return out


def save_plot(path: Path, image: np.ndarray, title: str, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.colorbar(im, ax=ax, label=label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("output/science/backward_camera_particle_channels/backward_camera_particle_channels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/science/backward_camera_geometry"))
    parser.add_argument("--generate-if-missing", action="store_true")
    args, extra = parser.parse_known_args()

    if args.generate_if_missing and not args.input_csv.exists():
        subprocess.run(
            [sys.executable, "scripts/science/run_backward_camera_particle_channels.py", *extra],
            check=True,
        )

    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = args.output_dir / "plots"
    inside = grid(rows, "n_inside_torus")
    path_inside = grid(rows, "path_inside_torus_rg")
    tau = grid(rows, "tau")
    density_proxy = np.divide(tau, np.maximum(path_inside, 1.0e-300))
    density_proxy[path_inside <= 0] = 0.0

    save_plot(args.output_dir / "ray_torus_intersections.png", inside, "Backward rays intersecting torus", "inside samples")
    save_plot(args.output_dir / "ray_density_samples.png", density_proxy, "Ray density/tau sample proxy", "tau per rg inside")
    save_plot(args.output_dir / "ray_tau_map.png", tau, "DIS tau per backward camera pixel", "tau")
    save_plot(args.output_dir / "ray_path_examples.png", path_inside, "Path length inside torus", "rg")
    save_plot(plots / "ray_torus_intersections.png", inside, "Backward rays intersecting torus", "inside samples")
    save_plot(plots / "ray_tau_map.png", tau, "DIS tau per backward camera pixel", "tau")

    total_pixels = len(rows)
    hit_pixels = sum(int(finite(row.get("n_inside_torus"))) > 0 for row in rows)
    total_tau = sum(finite(row.get("tau")) for row in rows)
    total_samples = sum(int(finite(row.get("n_samples"))) for row in rows)
    total_inside = sum(int(finite(row.get("n_inside_torus"))) for row in rows)

    summary = [
        "# Backward Camera Geometry Diagnostic",
        "",
        "Status: `BACKWARD_CAMERA_LOCAL_RESPONSE_PROTOTYPE` geometry diagnostic.",
        "",
        "- This diagnostic uses camera-first backward ray tracing products.",
        "- It does not use forward packet projection.",
        "- It does not compute luminosity, flux, redshift-calibrated energy, or radiative transfer.",
        "",
        f"- input_csv: `{args.input_csv}`",
        f"- total_pixels: `{total_pixels}`",
        f"- pixels_intersecting_torus: `{hit_pixels}`",
        f"- total_ray_samples: `{total_samples}`",
        f"- total_inside_torus_samples: `{total_inside}`",
        f"- total_tau_sum_over_pixels: `{total_tau:.12g}`",
        "",
        "Plots:",
        "",
        "- `ray_torus_intersections.png`",
        "- `ray_density_samples.png`",
        "- `ray_tau_map.png`",
        "- `ray_path_examples.png`",
    ]
    (args.output_dir / "backward_camera_geometry_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
