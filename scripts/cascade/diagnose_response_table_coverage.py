#!/usr/bin/env python3
"""Diagnose local-response-table coverage for a secondary-particle file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from apply_local_response_to_events import LocalResponseTable, kinetic_energy_gev  # noqa: E402


PDG_LABELS = {
    22: "gamma",
    11: "e-",
    -11: "e+",
    13: "mu-",
    -13: "mu+",
    111: "pi0",
    211: "pi+",
    -211: "pi-",
    321: "K+",
    -321: "K-",
    130: "K0L",
    310: "K0S",
    2212: "proton",
    -2212: "anti_proton",
    2112: "neutron",
    -2112: "anti_neutron",
    12: "nu_e",
    -12: "anti_nu_e",
    14: "nu_mu",
    -14: "anti_nu_mu",
    16: "nu_tau",
    -16: "anti_nu_tau",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dominant_status(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return counter.most_common(1)[0][0]


def issue_from_status(status: str) -> str:
    if status.startswith("OK_"):
        return "covered"
    if status in {"MISSING_PDG", "MISSING_MATERIAL", "MISSING_PHYSICS_LIST", "OUT_OF_RANGE", "EMPTY_TABLE"}:
        return status
    return "uncovered"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["pdg_id"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# Response Table Coverage Diagnostic",
        "",
        "This is a coverage diagnostic for local homogeneous-box response tables.",
        "It is not a physics result and performs no GEANT4 transport.",
        "",
        f"- secondaries: `{summary['secondaries']}`",
        f"- table: `{summary['table']}`",
        f"- material: `{summary['material']}`",
        f"- box size cm: `{summary['box_size_cm']}`",
        f"- physics list: `{summary['physics_list']}`",
        f"- total secondaries: `{summary['total_secondaries']}`",
        f"- total energy GeV: `{summary['total_energy_gev']:.12g}`",
        f"- OK fraction by count: `{summary['ok_fraction_count']:.12g}`",
        f"- OK fraction by total energy: `{summary['ok_fraction_energy']:.12g}`",
        "",
        "| PDG | Label | Count | Energy fraction | Kinetic min [GeV] | Kinetic max [GeV] | Dominant status | Issue |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pdg_id']} | {row['label']} | {row['count']} | "
            f"{float(row['total_energy_fraction']):.6g} | {float(row['kinetic_energy_min_gev']):.6g} | "
            f"{float(row['kinetic_energy_max_gev']):.6g} | {row['dominant_status']} | {row['coverage_issue']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    mpl_cache = output_dir / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    ok_fractions = [float(row["ok_fraction_count"]) for row in rows]
    energy_fractions = [float(row["total_energy_fraction"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(labels, ok_fractions, color="#2a9d8f")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("covered fraction by count")
    ax.set_title("response coverage by PDG")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots / "response_coverage_by_pdg.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(labels, energy_fractions, color="#457b9d")
    ax.set_ylabel("fraction of secondary total energy")
    ax.set_title("response coverage energy by PDG")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots / "response_coverage_energy_by_pdg.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, counts, width=0.4, label="count", color="#8ecae6")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, ok_fractions, width=0.4, label="OK fraction", color="#ffb703")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("count")
    ax2.set_ylabel("OK fraction")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_title("local response expanded particle comparison")
    fig.tight_layout()
    fig.savefig(plots / "local_response_expanded_particle_comparison.png", dpi=180)
    plt.close(fig)


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    table = LocalResponseTable.from_csv(args.table)
    warnings: Counter[str] = Counter()
    grouped: dict[int, dict[str, object]] = defaultdict(lambda: {
        "count": 0,
        "total_energy": 0.0,
        "kinetic": [],
        "statuses": Counter(),
    })
    total_count = 0
    total_energy = 0.0
    ok_count = 0
    ok_energy = 0.0

    for row in read_jsonl(args.secondaries):
        pdg = int(row.get("pdg_id", row.get("pdg")))
        energy = float(row["energy_gev"])
        kinetic, _ = kinetic_energy_gev(row, allow_unknown_mass=True, warnings=warnings)
        result = table.query(
            pdg_id=pdg,
            energy_gev=kinetic,
            density_g_cm3=args.density_g_cm3,
            material=args.material,
            box_size_cm=args.box_size_cm,
            physics_list=args.physics_list,
            mode=args.mode,
        )
        bucket = grouped[pdg]
        bucket["count"] = int(bucket["count"]) + 1
        bucket["total_energy"] = float(bucket["total_energy"]) + energy
        bucket["kinetic"].append(kinetic)  # type: ignore[union-attr]
        bucket["statuses"][result.status] += 1  # type: ignore[index]
        total_count += 1
        total_energy += energy
        if result.status in {"OK_INTERPOLATED", "OK_NEAREST"}:
            ok_count += 1
            ok_energy += energy

    rows: list[dict[str, object]] = []
    for pdg, data in sorted(grouped.items(), key=lambda item: (-(float(item[1]["total_energy"])), item[0])):
        statuses: Counter[str] = data["statuses"]  # type: ignore[assignment]
        kinetic_values = [value for value in data["kinetic"] if math.isfinite(value)]  # type: ignore[index]
        dom = dominant_status(statuses)
        count = int(data["count"])
        ok = statuses["OK_INTERPOLATED"] + statuses["OK_NEAREST"]
        rows.append({
            "pdg_id": pdg,
            "label": PDG_LABELS.get(pdg, f"pdg{pdg}"),
            "count": count,
            "total_energy_gev": float(data["total_energy"]),
            "total_energy_fraction": float(data["total_energy"]) / max(total_energy, 1.0e-300),
            "kinetic_energy_min_gev": min(kinetic_values) if kinetic_values else math.nan,
            "kinetic_energy_max_gev": max(kinetic_values) if kinetic_values else math.nan,
            "ok_count": ok,
            "ok_fraction_count": ok / max(count, 1),
            "dominant_status": dom,
            "coverage_issue": issue_from_status(dom),
            "status_counts": ";".join(f"{key}:{statuses[key]}" for key in sorted(statuses)),
        })

    summary = {
        "secondaries": str(args.secondaries),
        "table": str(args.table),
        "material": args.material,
        "box_size_cm": args.box_size_cm,
        "physics_list": args.physics_list,
        "total_secondaries": total_count,
        "total_energy_gev": total_energy,
        "ok_fraction_count": ok_count / max(total_count, 1),
        "ok_fraction_energy": ok_energy / max(total_energy, 1.0e-300),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "response_table_coverage.csv", rows)
    write_markdown(args.output_dir / "response_table_coverage.md", rows, summary)
    make_plots(args.output_dir, rows)
    return summary | {"rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondaries", type=Path, default=Path("output/cascade/pythia_secondaries.jsonl"))
    parser.add_argument("--table", type=Path, default=Path("output/cascade/local_response_table.csv"))
    parser.add_argument("--material", default="water")
    parser.add_argument("--box-size-cm", type=float, default=10.0)
    parser.add_argument("--physics-list", default="FTFP_BERT")
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--mode", choices=["interpolated", "nearest"], default="interpolated")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.secondaries, args.table]:
        if not path.exists():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2
    summary = diagnose(args)
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
