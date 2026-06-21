#!/usr/bin/env python3
"""Adaptively refine a local response table for observed secondaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from apply_local_response_to_events import LocalResponseTable, kinetic_energy_gev  # noqa: E402
from build_local_response_table import PDG_LABELS, run_case, write_csv as write_table_csv, write_npz  # noqa: E402


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_table(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_markdown(path: Path, args: argparse.Namespace, existing: list[dict[str, object]], new_rows: list[dict[str, object]], before: dict[str, float], after: dict[str, float]) -> None:
    counts = Counter(str(row.get("status", "")) for row in new_rows)
    lines = [
        "# Local Response Table Refinement",
        "",
        "Adaptive refinement for a local homogeneous-box response table.",
        "This is not collapsar physics and does not perform ray tracing or camera synthesis.",
        "",
        f"- input table: `{args.table}`",
        f"- secondaries: `{args.secondaries}`",
        f"- existing rows: `{len(existing)}`",
        f"- attempted new cells: `{len(new_rows)}`",
        f"- new PASS: `{counts['PASS']}`",
        f"- new BAD_ENERGY: `{counts['BAD_ENERGY']}`",
        f"- new FAIL: `{counts['FAIL']}`",
        f"- before OK fraction by count: `{before['ok_fraction_count']:.12g}`",
        f"- after OK fraction by count: `{after['ok_fraction_count']:.12g}`",
        f"- before OK fraction by energy: `{before['ok_fraction_energy']:.12g}`",
        f"- after OK fraction by energy: `{after['ok_fraction_energy']:.12g}`",
        "",
        "BAD_ENERGY cells remain explicit and are not promoted to PASS.",
        "",
        "## New Cells",
        "",
        "| PDG | Particle | Ekin [GeV] | Status | Diagnostic |",
        "|---:|---|---:|---|---|",
    ]
    for row in new_rows:
        lines.append(
            f"| {row['pdg_id']} | {row['particle']} | {float(row['energy_gev']):.8g} | "
            f"{row['status']} | {row.get('diagnostic', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def coverage(table_path: Path, secondaries: Path, args: argparse.Namespace) -> dict[str, float]:
    table = LocalResponseTable.from_csv(table_path)
    total_count = ok_count = 0
    total_energy = ok_energy = 0.0
    warnings: Counter[str] = Counter()
    for row in read_jsonl(secondaries):
        pdg = int(row.get("pdg_id", row.get("pdg")))
        kinetic, _ = kinetic_energy_gev(row, allow_unknown_mass=True, warnings=warnings)
        total = float(row["energy_gev"])
        result = table.query(pdg, kinetic, args.density_g_cm3, args.material, args.box_size_cm, args.physics_list, "interpolated")
        total_count += 1
        total_energy += total
        if result.status in {"OK_INTERPOLATED", "OK_NEAREST"}:
            ok_count += 1
            ok_energy += total
    return {
        "ok_fraction_count": ok_count / max(total_count, 1),
        "ok_fraction_energy": ok_energy / max(total_energy, 1.0e-300),
    }


def priorities(table_path: Path, secondaries: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    table = LocalResponseTable.from_csv(table_path)
    warnings: Counter[str] = Counter()
    grouped: dict[tuple[int, float], dict[str, object]] = {}
    for row in read_jsonl(secondaries):
        pdg = int(row.get("pdg_id", row.get("pdg")))
        kinetic, _ = kinetic_energy_gev(row, allow_unknown_mass=True, warnings=warnings)
        if not math.isfinite(kinetic) or kinetic <= 0.0:
            continue
        result = table.query(pdg, kinetic, args.density_g_cm3, args.material, args.box_size_cm, args.physics_list, "interpolated")
        if result.status != "OUT_OF_RANGE":
            continue
        key = (pdg, kinetic)
        item = grouped.setdefault(key, {
            "pdg_id": pdg,
            "particle": PDG_LABELS.get(pdg, f"pdg{pdg}"),
            "energy_gev": kinetic,
            "weighted_kinetic_energy_gev": 0.0,
            "count": 0,
        })
        item["weighted_kinetic_energy_gev"] = float(item["weighted_kinetic_energy_gev"]) + float(row.get("weight", 1.0)) * kinetic
        item["count"] = int(item["count"]) + 1
    return sorted(grouped.values(), key=lambda item: float(item["weighted_kinetic_energy_gev"]), reverse=True)


def make_plots(output_dir: Path, before: dict[str, float], after: dict[str, float]) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(["count before", "count after", "energy before", "energy after"], [
        before["ok_fraction_count"],
        after["ok_fraction_count"],
        before["ok_fraction_energy"],
        after["ok_fraction_energy"],
    ], color=["#adb5bd", "#2a9d8f", "#adb5bd", "#219ebc"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("OK coverage fraction")
    ax.set_title("local response refinement before after")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots / "local_response_refinement_before_after.png", dpi=180)
    plt.close(fig)


def refine(args: argparse.Namespace) -> dict[str, object]:
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("geant4-config") is None:
        print("local response refinement skipped: geant4-config not found.")
        return {"skipped": True}
    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)

    existing = read_table(args.table)
    before = coverage(args.table, args.secondaries, args)
    selected = priorities(args.table, args.secondaries, args)[: args.max_new_cells]
    cases_dir = output_dir / "local_response_table_refinement_cases"
    new_rows: list[dict[str, object]] = []
    for index, item in enumerate(selected, start=1):
        row = run_case(
            root=root,
            cases_dir=cases_dir,
            case_id=index,
            particle=str(item["particle"]),
            pdg=int(item["pdg_id"]),
            energy=float(item["energy_gev"]),
            density=args.density_g_cm3,
            material=args.material,
            box_size=args.box_size_cm,
            physics_list=args.physics_list,
            n_events=args.n_events,
            seed=args.seed + index,
            timeout=args.timeout,
            closure_tolerance=args.closure_tolerance,
        )
        if args.retry_bad_energy and row["status"] == "BAD_ENERGY":
            retry = run_case(
                root=root,
                cases_dir=cases_dir,
                case_id=100000 + index,
                particle=str(item["particle"]),
                pdg=int(item["pdg_id"]),
                energy=float(item["energy_gev"]),
                density=args.density_g_cm3,
                material=args.material,
                box_size=args.box_size_cm,
                physics_list=args.physics_list,
                n_events=args.n_events,
                seed=args.seed + 100000 + index,
                timeout=args.timeout,
                closure_tolerance=args.closure_tolerance,
            )
            if retry["status"] == "PASS":
                row = retry
            else:
                row["diagnostic"] = f"{row.get('diagnostic', '')};retry_unresolved:{retry.get('status')}:{retry.get('diagnostic')}"
        new_rows.append(row)
        temp_rows = existing + new_rows
        temp_path = output_dir / "_local_response_table_refined_tmp.csv"
        write_table_csv(temp_path, temp_rows)
        current = coverage(temp_path, args.secondaries, args)
        if current["ok_fraction_count"] >= args.target_ok_fraction or current["ok_fraction_energy"] >= args.target_energy_coverage:
            break

    refined_rows = existing + new_rows
    refined_csv = output_dir / "local_response_table_refined.csv"
    write_table_csv(refined_csv, refined_rows)
    write_npz(output_dir / "local_response_table_refined.npz", refined_rows)
    after = coverage(refined_csv, args.secondaries, args)
    write_markdown(output_dir / "local_response_table_refined.md", args, existing, new_rows, before, after)
    make_plots(output_dir, before, after)
    tmp = output_dir / "_local_response_table_refined_tmp.csv"
    if tmp.exists():
        tmp.unlink()
    return {"skipped": False, "before": before, "after": after, "new_cells": len(new_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("output/cascade/local_response_table_expanded.csv"))
    parser.add_argument("--secondaries", type=Path, default=Path("output/cascade/pythia_secondaries.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--material", default="water")
    parser.add_argument("--box-size-cm", type=float, default=10.0)
    parser.add_argument("--physics-list", default="FTFP_BERT")
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--max-new-cells", type=int, default=12)
    parser.add_argument("--target-ok-fraction", type=float, default=0.95)
    parser.add_argument("--target-energy-coverage", type=float, default=0.98)
    parser.add_argument("--n-events", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--closure-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--retry-bad-energy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.table, args.secondaries]:
        if not path.exists():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2
    result = refine(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
