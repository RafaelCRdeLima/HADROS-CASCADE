#!/usr/bin/env python3
"""Diagnose overlap between the deposition-proxy camera and j_dep field."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np


def read_raw(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    with h5py.File(args.field, "r") as handle:
        gx = handle["grid_x"][...]
        gy = handle["grid_y"][...]
        gz = handle["grid_z"][...]
        j = handle["j_dep"][...]
        dep = handle["deposited_energy_grid"][...]
        coverage = handle["coverage_grid"][...]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    stats = json.loads(args.camera_stats.read_text(encoding="utf-8"))
    raw = read_raw(args.camera_raw)

    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    nonzero = j > args.threshold
    total_voxels = int(j.size)
    nonzero_count = int(np.count_nonzero(nonzero))
    if nonzero_count:
        xs = xx[nonzero]
        ys = yy[nonzero]
        zs = zz[nonzero]
        weights = j[nonzero]
        bbox = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), float(zs.min()), float(zs.max())]
        com = [
            float(np.average(xs, weights=weights)),
            float(np.average(ys, weights=weights)),
            float(np.average(zs, weights=weights)),
        ]
        rr = np.sqrt(xs * xs + ys * ys + zs * zs)
        r_min = float(rr.min())
        r_max = float(rr.max())
        mean_cov = float(np.average(coverage[nonzero], weights=weights))
    else:
        bbox = [float("nan")] * 6
        com = [float("nan")] * 3
        r_min = r_max = mean_cov = float("nan")

    pixel_ok = sum(int(float(row["ok_queries"])) for row in raw)
    pixel_out = sum(int(float(row["out_of_range_queries"])) for row in raw)
    pixel_low = sum(int(float(row["low_coverage_queries"])) for row in raw)
    pixels_with_ok = sum(float(row["ok_queries"]) > 0 for row in raw)
    pixels_with_signal = sum(float(row["I_proxy"]) > 0 for row in raw)
    queries_in_nonzero_voxels = pixel_ok if float(stats.get("image_sum", 0.0)) > 0.0 else 0
    in_domain_zero = max(pixel_ok - queries_in_nonzero_voxels, 0)

    output = {
        "total_voxels": total_voxels,
        "nonzero_voxels": nonzero_count,
        "nonzero_bbox": bbox,
        "center_of_mass": com,
        "r_min_nonzero": r_min,
        "r_max_nonzero": r_max,
        "mean_coverage_nonzero": mean_cov,
        "camera_ok_queries": pixel_ok,
        "camera_out_of_range_queries": pixel_out,
        "camera_low_coverage_queries": pixel_low,
        "pixels_with_ok_queries": pixels_with_ok,
        "pixels_with_positive_signal": pixels_with_signal,
        "queries_in_nonzero_voxels_inferred": queries_in_nonzero_voxels,
        "queries_in_grid_jdep_zero_inferred": in_domain_zero,
        "image_sum": float(stats.get("image_sum", 0.0)),
        "total_weighted_deposited_energy_gev": manifest.get("total_weighted_deposited_energy_gev"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, output, raw, gx, gy, gz, j, nonzero)
    return output


def write_outputs(output_dir: Path, summary: dict[str, object], raw: list[dict[str, str]],
                  gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
                  j: np.ndarray, nonzero: np.ndarray) -> None:
    csv_path = output_dir / "deposition_camera_overlap_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": json.dumps(value) if isinstance(value, list) else value})
    md = [
        "# Deposition Camera Overlap Diagnostic",
        "",
        "Diagnostic only. This is not physical luminosity and does not change HADROS default emissivity.",
        "",
    ]
    for key, value in summary.items():
        md.append(f"- {key}: `{value}`")
    md.extend([
        "",
        "Camera query positions are inferred from aggregated camera raw rows in this phase.",
        "A zero image can be a geometric overlap result: rays can enter the grid but miss nonzero j_dep voxels.",
    ])
    (output_dir / "deposition_camera_overlap_diagnostic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    make_plots(output_dir, raw, gx, gy, gz, j, nonzero)


def make_plots(output_dir: Path, raw: list[dict[str, str]], gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
               j: np.ndarray, nonzero: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    for plane, xval, yval, xlabel, ylabel, name in [
        ("xy", xx[nonzero], yy[nonzero], "x", "y", "deposition_field_nonzero_voxels_xy.png"),
        ("xz", xx[nonzero], zz[nonzero], "x", "z", "deposition_field_nonzero_voxels_xz.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        if len(xval):
            ax.scatter(xval, yval, s=12, c=j[nonzero], cmap="magma")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace(".png", ""))
        fig.tight_layout()
        fig.savefig(plots / name, dpi=180)
        plt.close(fig)

    pix_i = np.array([int(row["i"]) for row in raw])
    pix_j = np.array([int(row["j"]) for row in raw])
    ok = np.array([float(row["ok_queries"]) for row in raw])
    signal = np.array([float(row["I_proxy"]) for row in raw])
    for values, name in [
        (ok, "camera_query_positions_xy.png"),
        (signal, "camera_query_positions_xz.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        sc = ax.scatter(pix_i, pix_j, c=values, s=10, cmap="viridis")
        fig.colorbar(sc, ax=ax)
        ax.set_xlabel("pixel i")
        ax.set_ylabel("pixel j")
        ax.set_title(name.replace(".png", ""))
        fig.tight_layout()
        fig.savefig(plots / name, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    if np.any(nonzero):
        ax.scatter(xx[nonzero], zz[nonzero], s=18, c="tab:red", label="j_dep > 0")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend(loc="best")
    ax.set_title("camera vs emissive bbox diagnostic")
    fig.tight_layout()
    fig.savefig(plots / "camera_vs_emissive_bbox.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=Path("output/cascade/deposition_emissivity_proxy.h5"))
    parser.add_argument("--manifest", type=Path, default=Path("output/cascade/deposition_emissivity_manifest.json"))
    parser.add_argument("--camera-stats", type=Path, default=Path("output/cascade/deposition_proxy_camera_stats.json"))
    parser.add_argument("--camera-raw", type=Path, default=Path("output/cascade/deposition_proxy_camera_raw.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = diagnose(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
