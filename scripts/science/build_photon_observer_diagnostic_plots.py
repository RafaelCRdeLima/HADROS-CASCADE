#!/usr/bin/env python3
"""Build diagnostic plots for the photon observer camera.

These plots are lightweight diagnostics for the ideal photon observer camera.
They are not paper-ready figures and do not model detector response.
"""

from __future__ import annotations

import argparse
import csv
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
    "photon_diagnostic_redshift_histogram.png",
    "photon_diagnostic_input_vs_observed_energy.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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


def valid_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("redshift_status", "")).strip().lower() == "valid"]


def valid_inside_fov_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in valid_rows(rows)
        if as_bool(row, "inside_fov")
        and as_int(row, "pixel_x") is not None
        and as_int(row, "pixel_y") is not None
    ]


def infer_shape(rows: list[dict[str, str]]) -> tuple[int, int]:
    max_x = max((as_int(row, "pixel_x") or 0 for row in rows), default=0)
    max_y = max((as_int(row, "pixel_y") or 0 for row in rows), default=0)
    return max_x + 1, max_y + 1


def zero_map(nx: int, ny: int) -> list[list[float]]:
    return [[0.0 for _ in range(nx)] for _ in range(ny)]


def build_maps(rows: list[dict[str, str]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    nx, ny = infer_shape(rows)
    input_energy = zero_map(nx, ny)
    observed_energy = zero_map(nx, ny)
    counts = zero_map(nx, ny)
    for row in rows:
        x = as_int(row, "pixel_x")
        y = as_int(row, "pixel_y")
        input_value = as_float(row, "input_energy_gev")
        observed_value = as_float(row, "observed_energy_gev")
        if x is None or y is None or input_value is None or observed_value is None:
            continue
        if not (0 <= y < ny and 0 <= x < nx):
            continue
        input_energy[y][x] += input_value
        observed_energy[y][x] += observed_value
        counts[y][x] += 1.0
    return input_energy, observed_energy, counts


def setup_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_map(plt, data: list[list[float]], path: Path, *, title: str, colorbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(data, origin="upper", interpolation="nearest", aspect="equal")
    ax.set_title(f"{title}\n{PHYSICS_LABEL}")
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
    ax.set_title(f"Diagnostic redshift factor histogram\n{PHYSICS_LABEL}")
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
    ax.set_title(f"Diagnostic input vs observed photon energy\n{PHYSICS_LABEL}")
    ax.set_xlabel("input_energy_gev")
    ax.set_ylabel("observed_energy_gev")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, *, input_path: Path, rows: list[dict[str, str]], valid: list[dict[str, str]], inside: list[dict[str, str]]) -> None:
    input_total = sum(as_float(row, "input_energy_gev") or 0.0 for row in inside)
    observed_total = sum(as_float(row, "observed_energy_gev") or 0.0 for row in inside)
    redshifts = [value for row in valid if (value := as_float(row, "redshift_factor")) is not None]
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
        f"- mean_redshift_factor_valid: `{(sum(redshifts) / len(redshifts)) if redshifts else 0.0:.12g}`",
        "",
        "Generated diagnostic products:",
    ]
    lines.extend(f"- `{name}`" for name in OUTPUT_FILENAMES)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_diagnostics(input_path: Path, output_dir: Path) -> list[Path]:
    fieldnames, rows = read_csv_rows(input_path)
    require_fields(fieldnames)
    valid = valid_rows(rows)
    inside = valid_inside_fov_rows(rows)
    input_map, observed_map, count_map = build_maps(inside)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(output_dir)
    products = [
        output_dir / "photon_diagnostic_input_energy_map.png",
        output_dir / "photon_diagnostic_observed_energy_map.png",
        output_dir / "photon_diagnostic_counts_map.png",
        output_dir / "photon_diagnostic_redshift_histogram.png",
        output_dir / "photon_diagnostic_input_vs_observed_energy.png",
    ]
    save_map(plt, input_map, products[0], title="Diagnostic input photon energy by pixel", colorbar_label="sum input_energy_gev")
    save_map(plt, observed_map, products[1], title="Diagnostic observed photon energy by pixel", colorbar_label="sum observed_energy_gev")
    save_map(plt, count_map, products[2], title="Diagnostic photon counts by pixel", colorbar_label="valid inside-FOV photons")
    save_histogram(plt, valid, products[3])
    save_scatter(plt, valid, products[4])
    summary = output_dir / "photon_diagnostic_summary.md"
    write_summary(summary, input_path=input_path, rows=rows, valid=valid, inside=inside)
    products.append(summary)
    return products


def main() -> int:
    args = parse_args()
    forbidden = [name for name in OUTPUT_FILENAMES if "paper" in name.lower() or "final" in name.lower()]
    if forbidden:
        raise RuntimeError(f"diagnostic output filenames must not look paper/final: {forbidden}")
    try:
        products = build_diagnostics(args.input, args.output_dir)
    except Exception as exc:
        print(f"Failed to build photon observer diagnostic plots: {exc}", file=sys.stderr)
        return 2
    print("photon_observer_diagnostic_products")
    for product in products:
        print(product)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
