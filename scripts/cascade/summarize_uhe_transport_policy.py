#!/usr/bin/env python3
"""Summarize the UHE-aware GEANT4 transport policy bookkeeping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_budget(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def summarize(output_dir: Path) -> dict[str, Any]:
    budget = read_budget(output_dir / "geant4_energy_budget.csv")
    unsupported = read_jsonl(output_dir / "geant4_unsupported_uhe_particles.jsonl")
    total_input = sum(f(row, "input_energy_gev") for row in budget)
    deposited = sum(f(row, "deposited_energy_gev") for row in budget)
    escaped = sum(f(row, "escaped_energy_gev") for row in budget)
    invisible = sum(f(row, "invisible_energy_gev") for row in budget)
    untracked = sum(f(row, "untracked_energy_gev") for row in budget)
    escaped_unsupported = sum(f(row, "escaped_unsupported_uhe_energy_gev") for row in budget)
    transported = max(total_input - escaped_unsupported, 0.0)
    closure = deposited + escaped + invisible + untracked + escaped_unsupported - total_input

    pdg_energy: dict[int, float] = defaultdict(float)
    pdg_count: dict[int, int] = defaultdict(int)
    top_particles = []
    for row in unsupported:
        pdg = int(row.get("pdg_id", row.get("pdg", 0)))
        energy = float(row.get("energy_gev", 0.0))
        kinetic = float(row.get("kinetic_energy_gev", energy))
        pdg_energy[pdg] += energy
        pdg_count[pdg] += 1
        top_particles.append((energy, kinetic, pdg, row))

    return {
        "total_input_energy_gev": total_input,
        "transported_by_geant4_energy_gev": transported,
        "skipped_unsupported_uhe_energy_gev": escaped_unsupported,
        "deposited_energy_gev": deposited,
        "escaped_energy_gev": escaped,
        "invisible_energy_gev": invisible,
        "untracked_energy_gev": untracked,
        "closure_error_gev": closure,
        "n_unsupported_uhe_particles": len(unsupported),
        "top_skipped_pdgs": sorted(
            [
                {"pdg_id": pdg, "count": pdg_count[pdg], "energy_gev": energy}
                for pdg, energy in pdg_energy.items()
            ],
            key=lambda item: item["energy_gev"],
            reverse=True,
        ),
        "top_skipped_particles": [
            {"pdg_id": pdg, "energy_gev": energy, "kinetic_energy_gev": kinetic}
            for energy, kinetic, pdg, _ in sorted(top_particles, reverse=True)[:20]
        ],
    }


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "uhe_transport_policy_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "quantity",
            "value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in [
            "total_input_energy_gev",
            "transported_by_geant4_energy_gev",
            "skipped_unsupported_uhe_energy_gev",
            "deposited_energy_gev",
            "escaped_energy_gev",
            "invisible_energy_gev",
            "untracked_energy_gev",
            "closure_error_gev",
            "n_unsupported_uhe_particles",
        ]:
            writer.writerow({"quantity": key, "value": summary[key]})

    lines = [
        "# UHE Transport Policy Summary",
        "",
        "GEANT4 real transport is applied only below the configured model-support thresholds.",
        "Unsupported UHE particles are not injected into GEANT4 and are passed to the escaping-packet pipeline.",
        "",
    ]
    for key in [
        "total_input_energy_gev",
        "transported_by_geant4_energy_gev",
        "skipped_unsupported_uhe_energy_gev",
        "deposited_energy_gev",
        "escaped_energy_gev",
        "invisible_energy_gev",
        "untracked_energy_gev",
        "closure_error_gev",
        "n_unsupported_uhe_particles",
    ]:
        lines.append(f"- {key}: `{summary[key]:.12g}`")
    lines.extend(["", "## Top Skipped PDGs", "", "| PDG | Count | Energy [GeV] |", "|---:|---:|---:|"])
    for item in summary["top_skipped_pdgs"][:20]:
        lines.append(f"| {item['pdg_id']} | {item['count']} | {item['energy_gev']:.12g} |")
    lines.extend(["", "## Top Skipped Energies", "", "| PDG | Energy [GeV] | Kinetic [GeV] |", "|---:|---:|---:|"])
    for item in summary["top_skipped_particles"]:
        lines.append(f"| {item['pdg_id']} | {item['energy_gev']:.12g} | {item['kinetic_energy_gev']:.12g} |")
    (output_dir / "uhe_transport_policy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(args.output_dir)
    write_outputs(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
