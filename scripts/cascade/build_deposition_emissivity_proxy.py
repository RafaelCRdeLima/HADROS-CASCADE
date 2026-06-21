#!/usr/bin/env python3
"""Build a deposition-emissivity proxy from weighted local deposition rows.

This Phase 4 tool constructs a voxelized proxy field j_dep proportional to
weighted deposited energy. It does not perform radiative conversion, Kerr ray
tracing, camera synthesis, or observable image generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ENERGY_COLUMNS = {
    "deposited": "weighted_deposited_energy_gev",
    "escaped": "weighted_escaped_energy_gev",
    "invisible": "weighted_invisible_energy_gev",
    "untracked": "weighted_untracked_energy_gev",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def parse_bounds(values: list[str], xyz: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if len(values) == 1 and values[0] == "auto":
        bounds = []
        for axis in range(3):
            lo = float(np.min(xyz[:, axis])) if xyz.size else -0.5
            hi = float(np.max(xyz[:, axis])) if xyz.size else 0.5
            if math.isclose(lo, hi):
                lo -= 0.5
                hi += 0.5
            else:
                pad = 0.05 * (hi - lo)
                lo -= pad
                hi += pad
            bounds.extend([lo, hi])
        return tuple(bounds)  # type: ignore[return-value]
    if len(values) != 6:
        raise ValueError("--bounds must be 'auto' or six numbers: xmin xmax ymin ymax zmin zmax")
    parsed = tuple(float(item) for item in values)
    if not (parsed[0] < parsed[1] and parsed[2] < parsed[3] and parsed[4] < parsed[5]):
        raise ValueError("bounds must satisfy min < max for x, y, and z")
    return parsed  # type: ignore[return-value]


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def voxel_index(value: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def normalize_field(field: np.ndarray, normalization: str, voxel_volume: float) -> np.ndarray:
    if normalization == "none":
        return field.copy()
    if normalization == "max":
        maximum = float(np.max(field))
        return field / maximum if maximum > 0.0 else field.copy()
    if normalization == "total":
        total = float(np.sum(field))
        return field / total if total > 0.0 else field.copy()
    if normalization == "volume":
        return field / voxel_volume if voxel_volume > 0.0 else field.copy()
    raise ValueError(f"unknown normalization: {normalization}")


def build_proxy(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.input)
    if rows:
        xyz = np.array([[as_float(row, "x"), as_float(row, "y"), as_float(row, "z")] for row in rows], dtype=float)
    else:
        xyz = np.zeros((0, 3), dtype=float)

    xmin, xmax, ymin, ymax, zmin, zmax = parse_bounds(args.bounds, xyz)
    n = args.grid_size
    x_edges = np.linspace(xmin, xmax, n + 1)
    y_edges = np.linspace(ymin, ymax, n + 1)
    z_edges = np.linspace(zmin, zmax, n + 1)
    shape = (n, n, n)
    deposited = np.zeros(shape, dtype=float)
    escaped = np.zeros(shape, dtype=float)
    invisible = np.zeros(shape, dtype=float)
    untracked = np.zeros(shape, dtype=float)
    ok_counts = np.zeros(shape, dtype=float)
    total_counts = np.zeros(shape, dtype=float)
    event_count = np.zeros(shape, dtype=float)
    voxel_events: dict[tuple[int, int, int], set[int]] = {}

    status_counts: Counter[str] = Counter()
    weighted_input_ok = 0.0
    weighted_input_total = 0.0

    for row in rows:
        ix = voxel_index(as_float(row, "x"), x_edges)
        iy = voxel_index(as_float(row, "y"), y_edges)
        iz = voxel_index(as_float(row, "z"), z_edges)
        idx = (ix, iy, iz)
        status = row.get("query_status", "")
        is_ok = status.startswith("OK")
        status_counts[status] += 1
        total_counts[idx] += 1.0
        ok_counts[idx] += 1.0 if is_ok else 0.0
        voxel_events.setdefault(idx, set()).add(int(float(row.get("event_id", 0) or 0)))
        deposited[idx] += as_float(row, ENERGY_COLUMNS["deposited"])
        escaped[idx] += as_float(row, ENERGY_COLUMNS["escaped"])
        invisible[idx] += as_float(row, ENERGY_COLUMNS["invisible"])
        untracked[idx] += as_float(row, ENERGY_COLUMNS["untracked"])
        weighted_kinetic = as_float(row, "kinetic_energy_gev") * as_float(row, "weight", 1.0)
        weighted_input_total += weighted_kinetic
        if is_ok:
            weighted_input_ok += weighted_kinetic

    for idx, events in voxel_events.items():
        event_count[idx] = float(len(events))

    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_grid = np.divide(ok_counts, total_counts, out=np.zeros_like(ok_counts), where=total_counts > 0.0)

    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    dz = (zmax - zmin) / n
    voxel_volume = dx * dy * dz
    j_dep = normalize_field(deposited, args.normalization, voxel_volume)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(args.input),
        "grid_size": n,
        "bounds": [xmin, xmax, ymin, ymax, zmin, zmax],
        "normalization": args.normalization,
        "voxel_volume": voxel_volume,
        "total_rows": len(rows),
        "status_counts": dict(status_counts),
        "global_ok_fraction_count": (status_counts["OK_INTERPOLATED"] + status_counts["OK_NEAREST"]) / max(len(rows), 1),
        "global_ok_fraction_weighted_kinetic": weighted_input_ok / max(weighted_input_total, 1.0e-300),
        "scope": "proxy only; no radiative conversion, Kerr ray tracing, camera, or observable image",
    }

    npz_path = output_dir / "deposition_emissivity_proxy.npz"
    np.savez(
        npz_path,
        grid_x=centers(x_edges),
        grid_y=centers(y_edges),
        grid_z=centers(z_edges),
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        j_dep=j_dep,
        deposited_energy_grid=deposited,
        escaped_energy_grid=escaped,
        invisible_energy_grid=invisible,
        untracked_energy_grid=untracked,
        coverage_grid=coverage_grid,
        event_count=event_count,
        metadata=json.dumps(metadata, sort_keys=True),
    )

    csv_path = output_dir / "deposition_emissivity_proxy.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "ix", "iy", "iz", "x", "y", "z", "j_dep", "deposited_energy_gev",
            "escaped_energy_gev", "invisible_energy_gev", "untracked_energy_gev",
            "coverage_ok_fraction", "event_count", "row_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        xs, ys, zs = centers(x_edges), centers(y_edges), centers(z_edges)
        nonzero = np.argwhere(total_counts > 0.0)
        for ix, iy, iz in nonzero:
            writer.writerow({
                "ix": int(ix),
                "iy": int(iy),
                "iz": int(iz),
                "x": xs[ix],
                "y": ys[iy],
                "z": zs[iz],
                "j_dep": j_dep[ix, iy, iz],
                "deposited_energy_gev": deposited[ix, iy, iz],
                "escaped_energy_gev": escaped[ix, iy, iz],
                "invisible_energy_gev": invisible[ix, iy, iz],
                "untracked_energy_gev": untracked[ix, iy, iz],
                "coverage_ok_fraction": coverage_grid[ix, iy, iz],
                "event_count": event_count[ix, iy, iz],
                "row_count": total_counts[ix, iy, iz],
            })

    summary_path = output_dir / "deposition_emissivity_summary.md"
    nonempty = int(np.count_nonzero(total_counts))
    total_dep = float(np.sum(deposited))
    total_esc = float(np.sum(escaped))
    total_inv = float(np.sum(invisible))
    total_untracked = float(np.sum(untracked))
    out_fraction = status_counts["OUT_OF_RANGE"] / max(len(rows), 1)
    lines = [
        "# Deposition Emissivity Proxy",
        "",
        "This is a proxy field proportional to local weighted deposited energy.",
        "It is not a physical luminosity, radiative-transfer product, Kerr ray-traced image, or observable map.",
        "",
        f"- input: `{args.input}`",
        f"- grid size: `{n}`",
        f"- normalization: `{args.normalization}`",
        f"- weighted deposited energy [GeV]: `{total_dep:.12g}`",
        f"- weighted escaped energy [GeV]: `{total_esc:.12g}`",
        f"- weighted invisible energy [GeV]: `{total_inv:.12g}`",
        f"- weighted untracked energy [GeV]: `{total_untracked:.12g}`",
        f"- non-empty voxels: `{nonempty}`",
        f"- global OK fraction by count: `{metadata['global_ok_fraction_count']:.12g}`",
        f"- global OK fraction by weighted kinetic energy: `{metadata['global_ok_fraction_weighted_kinetic']:.12g}`",
        f"- OUT_OF_RANGE fraction by count: `{out_fraction:.12g}`",
        "",
        "Coverage is incomplete if any non-OK status is present. Non-OK rows remain represented in status counts and coverage fields.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    make_plots(output_dir, deposited, escaped, invisible, untracked, coverage_grid, total_counts, x_edges, y_edges, z_edges)
    return {
        "npz": str(npz_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "weighted_deposited_energy_gev": total_dep,
        "global_ok_fraction_count": metadata["global_ok_fraction_count"],
        "global_ok_fraction_weighted_kinetic": metadata["global_ok_fraction_weighted_kinetic"],
        "nonempty_voxels": nonempty,
    }


def make_plots(
    output_dir: Path,
    deposited: np.ndarray,
    escaped: np.ndarray,
    invisible: np.ndarray,
    untracked: np.ndarray,
    coverage: np.ndarray,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> None:
    mpl_cache = output_dir / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    x_centers = centers(x_edges)
    y_centers = centers(y_edges)
    z_centers = centers(z_edges)

    xy = np.sum(deposited, axis=2).T
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(xy, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect="auto")
    fig.colorbar(im, ax=ax, label="weighted deposited energy [GeV]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("deposition emissivity proxy xy")
    fig.tight_layout()
    fig.savefig(plots / "deposition_emissivity_xy.png", dpi=180)
    plt.close(fig)

    xz = np.sum(deposited, axis=1).T
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(xz, origin="lower", extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]], aspect="auto")
    fig.colorbar(im, ax=ax, label="weighted deposited energy [GeV]")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("deposition emissivity proxy xz")
    fig.tight_layout()
    fig.savefig(plots / "deposition_emissivity_xz.png", dpi=180)
    plt.close(fig)

    weighted_cov = np.divide(np.sum(coverage * counts, axis=2), np.sum(counts, axis=2), out=np.zeros_like(np.sum(coverage, axis=2)), where=np.sum(counts, axis=2) > 0.0).T
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(weighted_cov, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="OK coverage fraction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("deposition emissivity coverage xy")
    fig.tight_layout()
    fig.savefig(plots / "deposition_emissivity_coverage_xy.png", dpi=180)
    plt.close(fig)

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy + zz * zz)
    flat_r = rr.ravel()
    flat_dep = deposited.ravel()
    if flat_r.size:
        bins = min(32, max(4, deposited.shape[0]))
        profile, edges = np.histogram(flat_r, bins=bins, weights=flat_dep)
        rc = 0.5 * (edges[:-1] + edges[1:])
    else:
        rc = np.array([])
        profile = np.array([])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.step(rc, profile, where="mid")
    ax.set_xlabel("r")
    ax.set_ylabel("weighted deposited energy [GeV]")
    ax.set_title("deposition emissivity r profile")
    fig.tight_layout()
    fig.savefig(plots / "deposition_emissivity_r_profile.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/cascade/response_weighted_deposition_refined.csv"))
    parser.add_argument("--map", type=Path, default=Path("output/cascade/response_deposition_map_refined.npz"), help="Accepted for provenance; the CSV is the authoritative input.")
    parser.add_argument("--primary-interactions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--bounds", nargs="+", default=["auto"])
    parser.add_argument("--normalization", choices=["none", "max", "total", "volume"], default="none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid_size <= 0:
        print("--grid-size must be positive", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"missing required input: {args.input}", file=sys.stderr)
        return 2
    result = build_proxy(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
