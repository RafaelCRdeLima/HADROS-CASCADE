#!/usr/bin/env python3
"""Run Phase 9.0 real-Kerr packet validation and plot the comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def angular_delta(a_theta: float, a_phi: float, b_theta: float, b_phi: float) -> float:
    ax = math.sin(a_theta) * math.cos(a_phi)
    ay = math.sin(a_theta) * math.sin(a_phi)
    az = math.cos(a_theta)
    bx = math.sin(b_theta) * math.cos(b_phi)
    by = math.sin(b_theta) * math.sin(b_phi)
    bz = math.cos(b_theta)
    dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
    return math.degrees(math.acos(dot))


def validate(rows: list[dict[str, str]]) -> dict[str, Any]:
    same_dir = [row for row in rows if row["case"] == "same_direction_different_origins"]
    same_origin = [row for row in rows if row["case"] == "same_origin_different_directions"]
    if len(same_dir) != 2 or len(same_origin) != 2:
        raise RuntimeError("expected two rows for each validation case")
    if not all(row["backend"] == "REAL_HADROS_KERR_GEODESIC" for row in rows):
        raise RuntimeError("real Kerr backend label missing")

    straight_delta_same_dir = angular_delta(
        finite(same_dir[0]["straight_observer_theta"]),
        finite(same_dir[0]["straight_observer_phi"]),
        finite(same_dir[1]["straight_observer_theta"]),
        finite(same_dir[1]["straight_observer_phi"]),
    )
    kerr_delta_same_dir = angular_delta(
        finite(same_dir[0]["real_kerr_observer_theta"]),
        finite(same_dir[0]["real_kerr_observer_phi"]),
        finite(same_dir[1]["real_kerr_observer_theta"]),
        finite(same_dir[1]["real_kerr_observer_phi"]),
    )
    straight_delta_same_origin = angular_delta(
        finite(same_origin[0]["straight_observer_theta"]),
        finite(same_origin[0]["straight_observer_phi"]),
        finite(same_origin[1]["straight_observer_theta"]),
        finite(same_origin[1]["straight_observer_phi"]),
    )
    kerr_delta_same_origin = angular_delta(
        finite(same_origin[0]["real_kerr_observer_theta"]),
        finite(same_origin[0]["real_kerr_observer_phi"]),
        finite(same_origin[1]["real_kerr_observer_theta"]),
        finite(same_origin[1]["real_kerr_observer_phi"]),
    )
    if straight_delta_same_dir > 1.0e-9:
        raise RuntimeError("straight-line projection changed for same direction")
    if kerr_delta_same_dir <= 1.0e-6:
        raise RuntimeError("real Kerr observer coordinates did not differ for different origins")
    return {
        "straight_delta_same_direction_deg": straight_delta_same_dir,
        "real_kerr_delta_same_direction_deg": kerr_delta_same_dir,
        "straight_delta_same_origin_deg": straight_delta_same_origin,
        "real_kerr_delta_same_origin_deg": kerr_delta_same_origin,
    }


def make_plot(output_dir: Path, rows: list[dict[str, str]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for case, marker in [("same_origin_different_directions", "o"), ("same_direction_different_origins", "s")]:
        subset = [row for row in rows if row["case"] == case]
        ax.scatter(
            [math.degrees(finite(row["straight_observer_phi"])) for row in subset],
            [math.degrees(finite(row["straight_observer_theta"])) for row in subset],
            marker=marker,
            s=70,
            label=f"{case} straight",
            alpha=0.55,
        )
        ax.scatter(
            [math.degrees(finite(row["real_kerr_observer_phi"])) for row in subset],
            [math.degrees(finite(row["real_kerr_observer_theta"])) for row in subset],
            marker=marker,
            s=70,
            edgecolor="black",
            label=f"{case} real Kerr",
        )
    ax.set_xlabel("observer phi [deg]")
    ax.set_ylabel("observer theta [deg]")
    ax.set_title("Escaping packets: real Kerr geodesic vs straight proxy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "packet_real_kerr_vs_straight.png", dpi=180)
    plt.close(fig)


def append_validation_to_markdown(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Numerical Validation\n\n")
        handle.write("| Metric | Value |\n")
        handle.write("|---|---:|\n")
        for key, value in metrics.items():
            handle.write(f"| {key} | {float(value):.12g} |\n")
        handle.write("\n")
        handle.write("This demonstrates that same-direction packets have identical straight-line angular coordinates, while the real Kerr integration returns different observer coordinates for different origins.\n")
        handle.write("The comparison distinguishes `REAL_HADROS_KERR_GEODESIC` from `PROXY_STRAIGHT_LINE`.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        subprocess.run(["make", "packet_real_kerr_vs_straight"], cwd=ROOT, check=True, text=True)
    exe = ROOT / "build/packet_real_kerr_vs_straight"
    subprocess.run([str(exe), "--output-dir", str(args.output_dir)], cwd=ROOT, check=True, text=True, capture_output=True)
    rows = read_csv(args.output_dir / "packet_real_kerr_vs_straight.csv")
    metrics = validate(rows)
    make_plot(args.output_dir, rows)
    append_validation_to_markdown(args.output_dir / "packet_real_kerr_vs_straight.md", metrics)
    print(json.dumps({"backend": "REAL_HADROS_KERR_GEODESIC", **metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
