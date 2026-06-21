#!/usr/bin/env python3
"""Build and run the Kerr tetrad initialization diagnostic."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True, text=True)


def make_plots(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = output_dir / "kerr_tetrad_initialization_diagnostic.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if row["valid"] == "1"]
    errors = [abs(float(row["null_norm"])) for row in valid]
    energies = [float(row["zamo_energy"]) for row in valid]
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.semilogy(range(len(errors)), errors, marker="o", linestyle="")
    ax.set_xlabel("valid diagnostic case")
    ax.set_ylabel("|g(p,p)|")
    ax.set_title("ZAMO tetrad null norm error")
    fig.tight_layout()
    fig.savefig(plots / "kerr_tetrad_null_norm_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(range(len(energies)), energies, marker="o", linestyle="")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("valid diagnostic case")
    ax.set_ylabel("ZAMO energy")
    ax.set_title("ZAMO energy positivity")
    fig.tight_layout()
    fig.savefig(plots / "kerr_tetrad_energy_positive.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--spin", type=float, default=0.8)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    app = ROOT / "build" / "test_kerr_tetrad_initialization"
    run_command([
        "g++",
        "-std=c++17",
        "-Iinclude",
        "apps/test_kerr_tetrad_initialization.cpp",
        "src/cascade/kerr_local_tetrad.cpp",
        "src/kerr_metric.cpp",
        "-o",
        str(app),
    ])
    run_command([str(app), "--output-dir", str(output), "--spin", str(args.spin)])
    make_plots(output)
    print(f"wrote {output / 'kerr_tetrad_initialization_diagnostic.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
