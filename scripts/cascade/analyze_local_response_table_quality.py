#!/usr/bin/env python3
"""Analyze local response table quality and refinement priorities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from apply_local_response_to_events import LocalResponseTable, kinetic_energy_gev  # noqa: E402
from diagnose_response_table_coverage import PDG_LABELS  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_table_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["record_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, table_rows: list[dict[str, str]], priority_rows: list[dict[str, object]], coverage: dict[str, float]) -> None:
    mpl_cache = output_dir / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    by_pdg: dict[int, Counter[str]] = defaultdict(Counter)
    for row in table_rows:
        by_pdg[int(float(row["pdg_id"]))][row.get("status", "")] += 1
    labels = [PDG_LABELS.get(pdg, str(pdg)) for pdg in sorted(by_pdg)]
    pass_counts = [by_pdg[pdg]["PASS"] for pdg in sorted(by_pdg)]
    bad_counts = [by_pdg[pdg]["BAD_ENERGY"] for pdg in sorted(by_pdg)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(labels))
    ax.bar(x, pass_counts, label="PASS", color="#2a9d8f")
    ax.bar(x, bad_counts, bottom=pass_counts, label="BAD_ENERGY", color="#e76f51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("table cells")
    ax.set_title("local response quality status by PDG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "local_response_quality_status_by_pdg.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(["count", "energy"], [coverage["ok_fraction_count"], coverage["ok_fraction_energy"]], color=["#8ecae6", "#219ebc"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("covered fraction")
    ax.set_title("local response quality energy coverage")
    fig.tight_layout()
    fig.savefig(plots / "local_response_quality_energy_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    top = priority_rows[:12]
    ax.bar(
        [f"{PDG_LABELS.get(int(row['pdg_id']), row['pdg_id'])}\n{float(row['kinetic_energy_gev']):.3g}" for row in top],
        [float(row["weighted_kinetic_energy_gev"]) for row in top],
        color="#ffb703",
    )
    ax.set_ylabel("weighted missing kinetic energy [GeV]")
    ax.set_title("local response refinement priorities")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots / "local_response_refinement_before_after.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    table_rows = read_table_rows(args.table)
    table = LocalResponseTable.from_csv(args.table)
    secondaries = read_jsonl(args.secondaries)
    warnings: Counter[str] = Counter()

    status_by_pdg: dict[int, Counter[str]] = defaultdict(Counter)
    bad_cells: list[dict[str, object]] = []
    for row in table_rows:
        pdg = int(float(row["pdg_id"]))
        status = row.get("status", "")
        status_by_pdg[pdg][status] += 1
        if status == "BAD_ENERGY":
            bad_cells.append({
                "record_type": "bad_energy_cell",
                "pdg_id": pdg,
                "label": PDG_LABELS.get(pdg, f"pdg{pdg}"),
                "kinetic_energy_gev": float(row["energy_gev"]),
                "density_g_cm3": float(row["density_g_cm3"]),
                "material": row["material"],
                "box_size_cm": float(row["box_size_cm"]),
                "physics_list": row["physics_list"],
                "weighted_kinetic_energy_gev": 0.0,
                "query_count": 0,
                "status": "BAD_ENERGY",
                "diagnostic": row.get("diagnostic", ""),
                "stdout_path": row.get("stdout_path", ""),
                "stderr_path": row.get("stderr_path", ""),
            })

    proposed: dict[tuple[int, float, float, str, float, str], dict[str, object]] = {}
    total_count = 0
    ok_count = 0
    total_energy = 0.0
    ok_energy = 0.0
    for row in secondaries:
        pdg = int(row.get("pdg_id", row.get("pdg")))
        kinetic, _ = kinetic_energy_gev(row, allow_unknown_mass=True, warnings=warnings)
        total = float(row["energy_gev"])
        weight = float(row.get("weight", 1.0))
        result = table.query(pdg, kinetic, args.density_g_cm3, args.material, args.box_size_cm, args.physics_list, args.mode)
        total_count += 1
        total_energy += total
        if result.status in {"OK_INTERPOLATED", "OK_NEAREST"}:
            ok_count += 1
            ok_energy += total
            continue
        if result.status == "OUT_OF_RANGE" and math.isfinite(kinetic) and kinetic > 0.0:
            key = (pdg, kinetic, args.density_g_cm3, args.material, args.box_size_cm, args.physics_list)
            item = proposed.setdefault(key, {
                "record_type": "proposed_cell",
                "pdg_id": pdg,
                "label": PDG_LABELS.get(pdg, f"pdg{pdg}"),
                "kinetic_energy_gev": kinetic,
                "density_g_cm3": args.density_g_cm3,
                "material": args.material,
                "box_size_cm": args.box_size_cm,
                "physics_list": args.physics_list,
                "weighted_kinetic_energy_gev": 0.0,
                "query_count": 0,
                "status": "PROPOSED",
                "diagnostic": result.status,
                "stdout_path": "",
                "stderr_path": "",
            })
            item["weighted_kinetic_energy_gev"] = float(item["weighted_kinetic_energy_gev"]) + weight * kinetic
            item["query_count"] = int(item["query_count"]) + 1

    priority_rows = sorted(proposed.values(), key=lambda item: float(item["weighted_kinetic_energy_gev"]), reverse=True)
    summary_rows: list[dict[str, object]] = []
    for pdg in sorted(status_by_pdg):
        counts = status_by_pdg[pdg]
        summary_rows.append({
            "record_type": "pdg_summary",
            "pdg_id": pdg,
            "label": PDG_LABELS.get(pdg, f"pdg{pdg}"),
            "kinetic_energy_gev": math.nan,
            "density_g_cm3": args.density_g_cm3,
            "material": args.material,
            "box_size_cm": args.box_size_cm,
            "physics_list": args.physics_list,
            "weighted_kinetic_energy_gev": 0.0,
            "query_count": 0,
            "status": f"PASS:{counts['PASS']};BAD_ENERGY:{counts['BAD_ENERGY']};OTHER:{sum(counts.values()) - counts['PASS'] - counts['BAD_ENERGY']}",
            "diagnostic": "",
            "stdout_path": "",
            "stderr_path": "",
        })

    rows = summary_rows + bad_cells + priority_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "local_response_table_quality.csv", rows)

    coverage = {
        "ok_fraction_count": ok_count / max(total_count, 1),
        "ok_fraction_energy": ok_energy / max(total_energy, 1.0e-300),
    }
    lines = [
        "# Local Response Table Quality",
        "",
        "This is a quality-control report for local homogeneous-box response tables.",
        "It is not a physics result and does not run GEANT4.",
        "",
        f"- table: `{args.table}`",
        f"- secondaries: `{args.secondaries}`",
        f"- total secondaries: `{total_count}`",
        f"- OK fraction by count: `{coverage['ok_fraction_count']:.12g}`",
        f"- OK fraction by total energy: `{coverage['ok_fraction_energy']:.12g}`",
        f"- BAD_ENERGY cells: `{len(bad_cells)}`",
        f"- proposed OUT_OF_RANGE cells: `{len(priority_rows)}`",
        "",
        "## PASS/BAD_ENERGY By PDG",
        "",
        "| PDG | Label | PASS | BAD_ENERGY | Other |",
        "|---:|---|---:|---:|---:|",
    ]
    for pdg in sorted(status_by_pdg):
        counts = status_by_pdg[pdg]
        other = sum(counts.values()) - counts["PASS"] - counts["BAD_ENERGY"]
        lines.append(f"| {pdg} | {PDG_LABELS.get(pdg, f'pdg{pdg}')} | {counts['PASS']} | {counts['BAD_ENERGY']} | {other} |")
    lines.extend(["", "## Highest Priority New Cells", "", "| PDG | Label | Ekin [GeV] | Weighted missing Ekin [GeV] | Count |", "|---:|---|---:|---:|---:|"])
    for row in priority_rows[:20]:
        lines.append(
            f"| {row['pdg_id']} | {row['label']} | {float(row['kinetic_energy_gev']):.8g} | "
            f"{float(row['weighted_kinetic_energy_gev']):.8g} | {row['query_count']} |"
        )
    lines.extend(["", "OUT_OF_RANGE and BAD_ENERGY are diagnostics. They are not hidden or promoted to PASS."])
    (args.output_dir / "local_response_table_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_plots(args.output_dir, table_rows, priority_rows, coverage)
    return {"coverage": coverage, "bad_cells": len(bad_cells), "proposed_cells": len(priority_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("output/cascade/local_response_table_expanded.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("output/cascade/response_table_coverage.csv"))
    parser.add_argument("--secondaries", type=Path, default=Path("output/cascade/pythia_secondaries.jsonl"))
    parser.add_argument("--material", default="water")
    parser.add_argument("--box-size-cm", type=float, default=10.0)
    parser.add_argument("--physics-list", default="FTFP_BERT")
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--mode", choices=["interpolated", "nearest"], default="interpolated")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.table, args.secondaries]:
        if not path.exists():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2
    result = analyze(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
