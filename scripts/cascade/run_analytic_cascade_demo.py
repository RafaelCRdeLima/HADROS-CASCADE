#!/usr/bin/env python3
"""Run the dependency-free HADROS-CASCADE analytic end-to-end demo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_interaction_points(path: Path, n_events: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = ["clean_funnel", "funnel_wall", "disk_atmosphere", "saturated_disk_body"]
    with path.open("w", encoding="utf-8") as handle:
        for event_id in range(1, n_events + 1):
            radius_cm = 1.0e7 + 3.0e7 * rng.random()
            phi = 2.0 * math.pi * rng.random()
            z_cm = rng.uniform(-2.0e7, 2.0e7)
            density = 10.0 ** rng.uniform(-8.0, 2.0)
            weight = 0.2 + 1.8 * rng.random()
            row = {
                "event_id": event_id,
                "x_cm": radius_cm * math.cos(phi),
                "y_cm": radius_cm * math.sin(phi),
                "z_cm": z_cm,
                "density_g_cm3": density,
                "electron_fraction": 0.45 + 0.1 * rng.random(),
                "column_before_cm2": 10.0 ** rng.uniform(24.0, 32.0),
                "tau_before": 10.0 ** rng.uniform(-3.0, 3.0),
                "weight": weight,
                "region_class": classes[(event_id - 1) % len(classes)],
            }
            handle.write(json.dumps(row, sort_keys=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_energy_budget(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def build_cpp_backend(root: Path) -> None:
    subprocess.run(["make", "cascade_analytic_demo"], cwd=root, check=True)


def run_cpp_pipeline(root: Path, interaction_path: Path, output_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        str(root / "build" / "cascade_analytic_demo"),
        str(interaction_path),
        str(output_dir),
        f"{args.energy_gev:.17g}",
        str(args.seed),
        f"{args.fixed_y:.17g}",
        "1" if args.sample_y else "0",
    ]
    completed = subprocess.run(cmd, cwd=root, check=True, text=True, capture_output=True)
    return completed.stdout


def write_deposition_npz(output_dir: Path, budget_rows: list[dict[str, float]]) -> Path:
    path = output_dir / "deposition_maps.npz"
    event_id = np.asarray([row["event_id"] for row in budget_rows], dtype=np.uint64)
    deposited_em = np.asarray([row["deposited_em_gev"] for row in budget_rows], dtype=np.float64)
    deposited_had = np.asarray([row["deposited_hadronic_gev"] for row in budget_rows], dtype=np.float64)
    escaped_muon = np.asarray([row["escaped_muon_gev"] for row in budget_rows], dtype=np.float64)
    escaped_neutrino = np.asarray([row["escaped_neutrino_gev"] for row in budget_rows], dtype=np.float64)
    np.savez(
        path,
        event_id=event_id,
        deposited_em_gev=deposited_em,
        deposited_hadronic_gev=deposited_had,
        escaped_muon_gev=escaped_muon,
        escaped_neutrino_gev=escaped_neutrino,
        deposited_total_gev=deposited_em + deposited_had,
        escaped_total_gev=escaped_muon + escaped_neutrino,
    )
    return path


def make_plots(output_dir: Path, primary_events: list[dict], budget_rows: list[dict[str, float]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    event_id = np.asarray([row["event_id"] for row in budget_rows], dtype=np.float64)
    input_energy = np.asarray([row["energy_gev"] for row in primary_events], dtype=np.float64)
    weights = np.asarray([row["weight"] for row in primary_events], dtype=np.float64)
    deposited = np.asarray([row["deposited_total_gev"] for row in budget_rows], dtype=np.float64)
    escaped = np.asarray([row["escaped_total_gev"] for row in budget_rows], dtype=np.float64)
    accounted = deposited + escaped
    closure_error = accounted - input_energy

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(event_id, closure_error / np.maximum(input_energy, 1.0), marker="o", linewidth=1.2)
    ax.set_xlabel("event id")
    ax.set_ylabel("(deposited + escaped - input) / input")
    ax.set_title("Analytic cascade energy closure")
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_closure.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(deposited / np.maximum(input_energy, 1.0), bins=20, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("deposited energy fraction")
    ax.set_ylabel("events")
    ax.set_title("Analytic deposited fraction")
    fig.tight_layout()
    fig.savefig(plots_dir / "deposited_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(weights, bins=20, color="#577590", edgecolor="white")
    ax.set_xlabel("statistical weight")
    ax.set_ylabel("events")
    ax.set_title("Event weight distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "event_weight_distribution.png", dpi=180)
    plt.close(fig)


def energy_summary(primary_events: list[dict], budget_rows: list[dict[str, float]]) -> dict[str, float]:
    total_input = sum(row["energy_gev"] for row in primary_events)
    total_deposited = sum(row["deposited_total_gev"] for row in budget_rows)
    total_escaped = sum(row["escaped_total_gev"] for row in budget_rows)
    closure_error = total_deposited + total_escaped - total_input
    max_relative_error = 0.0
    for primary, budget in zip(primary_events, budget_rows):
        input_energy = primary["energy_gev"]
        accounted = budget["deposited_total_gev"] + budget["escaped_total_gev"]
        max_relative_error = max(max_relative_error, abs(accounted - input_energy) / max(input_energy, 1.0))
    return {
        "events": float(len(primary_events)),
        "total_input_gev": total_input,
        "total_deposited_gev": total_deposited,
        "total_escaped_gev": total_escaped,
        "closure_error_gev": closure_error,
        "max_relative_error": max_relative_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--interaction-points", type=Path)
    parser.add_argument("--n-events", type=int, default=64)
    parser.add_argument("--energy-gev", type=float, default=1.0e9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fixed-y", type=float, default=0.25)
    parser.add_argument("--sample-y", action="store_true")
    parser.add_argument("--regenerate-interactions", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    interaction_path = args.interaction_points
    if interaction_path is None:
        interaction_path = output_dir / "interaction_points.jsonl"
    elif not interaction_path.is_absolute():
        interaction_path = (root / interaction_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.regenerate_interactions or not interaction_path.exists():
        write_interaction_points(interaction_path, args.n_events, args.seed)

    if not args.no_build:
        build_cpp_backend(root)

    cpp_summary = run_cpp_pipeline(root, interaction_path, output_dir, args)
    primary_events = read_jsonl(output_dir / "primary_events.jsonl")
    budget_rows = read_energy_budget(output_dir / "cascade_energy_budget.csv")
    npz_path = write_deposition_npz(output_dir, budget_rows)
    make_plots(output_dir, primary_events, budget_rows)
    summary = energy_summary(primary_events, budget_rows)

    print(cpp_summary.strip())
    print("analytic cascade demo summary")
    for key, value in summary.items():
        print(f"{key}={value:.12e}")
    print(f"deposition_maps={npz_path}")
    print(f"plots={output_dir / 'plots'}")
    print("NOTE: this is an architecture/audit demo, not a publishable physics simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
