#!/usr/bin/env python3
"""Run minimal homogeneous-material GEANT4 benchmarks for HADROS-CASCADE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PDG = {
    "gamma": 22,
    "electron": 11,
    "proton": 2212,
    "nu_mu": 14,
}

MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    2212: 0.93827208816,
    14: 0.0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_secondary(path: Path, particle: str, energy_gev: float) -> None:
    pdg = PDG[particle]
    mass = MASS_GEV[pdg]
    momentum = math.sqrt(max(energy_gev * energy_gev - mass * mass, 0.0))
    row = {
        "event_id": 1,
        "parent_event_id": 1,
        "pdg": pdg,
        "pdg_id": pdg,
        "energy_gev": energy_gev,
        "px_gev": 0.0,
        "py_gev": 0.0,
        "pz_gev": momentum,
        "mass_gev": mass,
        "weight": 1.0,
        "stable": 1,
        "origin": f"geant4_material_benchmark_{particle}",
        "origin_backend": "geant4_material_benchmark",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")


def read_budget(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {key: float(value) for key, value in rows[0].items() if key != "energy_convention"}


def classify(returncode: int, timed_out: bool, budget: dict[str, float] | None) -> tuple[str, float, str]:
    if timed_out:
        return "TIMEOUT", math.nan, "timeout"
    if returncode < 0 or returncode == 139:
        return "CRASH", math.nan, f"signal_{-returncode}" if returncode < 0 else "sigsegv"
    if returncode != 0:
        return "FAIL", math.nan, f"exit_{returncode}"
    if budget is None:
        return "FAIL", math.nan, "missing_energy_budget"
    values = [
        budget.get("input_energy_gev", math.nan),
        budget.get("deposited_energy_gev", math.nan),
        budget.get("escaped_energy_gev", math.nan),
        budget.get("invisible_energy_gev", math.nan),
        budget.get("untracked_energy_gev", 0.0),
    ]
    if any((not math.isfinite(value) or value < 0.0) for value in values):
        return "BAD_ENERGY", math.nan, "nan_or_negative_energy"
    input_energy, deposited, escaped, invisible, untracked = values
    relative = abs(deposited + escaped + invisible + untracked - input_energy) / max(input_energy, 1.0)
    if relative > 1.0e-5:
        return "BAD_ENERGY", relative, "closure_tolerance"
    return "PASS", relative, ""


def benchmark_cases(full: bool) -> list[dict[str, object]]:
    if not full:
        return [
            {"benchmark": "gamma_water", "particle": "gamma", "energy_gev": 0.1, "box_size_cm": 10.0, "material": "water"},
            {"benchmark": "electron_water", "particle": "electron", "energy_gev": 0.1, "box_size_cm": 10.0, "material": "water"},
            {"benchmark": "proton_hydrogen", "particle": "proton", "energy_gev": 1.0, "box_size_cm": 10.0, "material": "hydrogen"},
            {"benchmark": "neutrino_hydrogen", "particle": "nu_mu", "energy_gev": 1.0, "box_size_cm": 10.0, "material": "hydrogen"},
        ]

    cases: list[dict[str, object]] = []
    for material in ["water", "hydrogen"]:
        for particle, energies in [
            ("gamma", [0.01, 0.1, 1.0]),
            ("electron", [0.01, 0.1, 1.0]),
        ]:
            for energy in energies:
                for box_size in [1.0, 10.0, 100.0]:
                    cases.append({
                        "benchmark": f"{particle}_{material}",
                        "particle": particle,
                        "energy_gev": energy,
                        "box_size_cm": box_size,
                        "material": material,
                    })
    for energy in [0.1, 1.0, 10.0]:
        for box_size in [1.0, 10.0, 100.0]:
            cases.append({
                "benchmark": "proton_hydrogen",
                "particle": "proton",
                "energy_gev": energy,
                "box_size_cm": box_size,
                "material": "hydrogen",
            })
    for box_size in [1.0, 10.0, 100.0]:
        cases.append({
            "benchmark": "neutrino_hydrogen",
            "particle": "nu_mu",
            "energy_gev": 1.0,
            "box_size_cm": box_size,
            "material": "hydrogen",
        })
    return cases


def run_case(root: Path, cases_dir: Path, case_id: int, case: dict[str, object], timeout: float) -> dict[str, object]:
    particle = str(case["particle"])
    energy = float(case["energy_gev"])
    box_size = float(case["box_size_cm"])
    material = str(case["material"])
    case_dir = cases_dir / f"case_{case_id:04d}_{particle}_{energy:g}GeV_{box_size:g}cm_{material}"
    input_path = case_dir / "secondaries.jsonl"
    output_dir = case_dir / "out"
    write_secondary(input_path, particle, energy)

    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(input_path),
        str(output_dir),
        f"{box_size:.17g}",
        "1.0",
        "FTFP_BERT",
        material,
        "geant4",
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=timeout)
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {timeout} s"
    runtime = time.monotonic() - started

    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

    budget = read_budget(output_dir / "geant4_energy_budget.csv")
    status, relative_closure, diagnostic = classify(returncode, timed_out, budget)
    input_energy = budget.get("input_energy_gev", math.nan) if budget else math.nan
    deposited = budget.get("deposited_energy_gev", math.nan) if budget else math.nan
    escaped = budget.get("escaped_energy_gev", math.nan) if budget else math.nan
    invisible = budget.get("invisible_energy_gev", math.nan) if budget else math.nan
    denom = max(input_energy, 1.0e-300) if math.isfinite(input_energy) else math.nan

    return {
        "case_id": case_id,
        "benchmark": case["benchmark"],
        "particle": particle,
        "pdg_id": PDG[particle],
        "energy_gev": energy,
        "material": material,
        "box_size_cm": box_size,
        "density_g_cm3": 1.0,
        "physics_list": "FTFP_BERT",
        "status": status,
        "diagnostic": diagnostic,
        "exit_code": returncode,
        "runtime_s": runtime,
        "relative_closure_error": relative_closure,
        "input_energy_gev": input_energy,
        "deposited_energy_gev": deposited,
        "escaped_energy_gev": escaped,
        "invisible_energy_gev": invisible,
        "deposited_fraction": deposited / denom if math.isfinite(deposited) and math.isfinite(denom) else math.nan,
        "escaped_fraction": escaped / denom if math.isfinite(escaped) and math.isfinite(denom) else math.nan,
        "invisible_fraction": invisible / denom if math.isfinite(invisible) and math.isfinite(denom) else math.nan,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_dir": str(output_dir),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], full: bool) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    lines = [
        "# GEANT4 Material Benchmarks",
        "",
        "These are minimal homogeneous-material benchmarks for technical/physics sanity checks.",
        "They are not collapsar, Kerr, plasma, PeV/EeV, or publication-grade results.",
        "",
        f"- mode: `{'full' if full else 'quick'}`",
        f"- total cases: `{len(rows)}`",
    ]
    for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]:
        lines.append(f"- {status}: `{counts.get(status, 0)}`")
    lines.extend(["", "## Cases", ""])
    lines.append("| case | benchmark | particle | E [GeV] | material | box [cm] | status | deposited | escaped | invisible | closure |")
    lines.append("|---:|---|---|---:|---|---:|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['benchmark']} | {row['particle']} | {row['energy_gev']} | "
            f"{row['material']} | {row['box_size_cm']} | {row['status']} | "
            f"{float(row['deposited_fraction']):.6g} | {float(row['escaped_fraction']):.6g} | "
            f"{float(row['invisible_fraction']):.6g} | {float(row['relative_closure_error']):.3g} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append("- Gamma/electron/proton cases probe simple homogeneous-material response only.")
    lines.append("- Neutrino cases are sanity checks that energy is invisible/escaped unless explicit interactions are modeled.")
    lines.append("- Physical use requires comparison with known GEANT4 examples and material-response benchmarks.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    passing = [row for row in rows if row["status"] == "PASS"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for key, group in sorted(group_rows(passing, ["particle", "material", "energy_gev"]).items()):
        group = sorted(group, key=lambda row: float(row["box_size_cm"]))
        ax.plot(
            [float(row["box_size_cm"]) for row in group],
            [float(row["deposited_fraction"]) for row in group],
            marker="o",
            linewidth=1.2,
            label=f"{key[0]} {key[1]} {float(key[2]):g} GeV",
        )
    ax.set_xscale("log")
    ax.set_xlabel("box size [cm]")
    ax.set_ylabel("deposited fraction")
    ax.set_title("GEANT4 benchmark deposited vs box")
    if passing:
        ax.legend(fontsize=7, ncols=1)
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_benchmark_deposited_vs_box.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for key, group in sorted(group_rows(passing, ["particle", "material", "energy_gev"]).items()):
        group = sorted(group, key=lambda row: float(row["box_size_cm"]))
        ax.plot(
            [float(row["box_size_cm"]) for row in group],
            [float(row["escaped_fraction"]) for row in group],
            marker="o",
            linewidth=1.2,
            label=f"{key[0]} {key[1]} {float(key[2]):g} GeV",
        )
    ax.set_xscale("log")
    ax.set_xlabel("box size [cm]")
    ax.set_ylabel("escaped fraction")
    ax.set_title("GEANT4 benchmark escaped vs box")
    if passing:
        ax.legend(fontsize=7, ncols=1)
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_benchmark_escaped_vs_box.png", dpi=180)
    plt.close(fig)

    labels = [f"{row['particle']}\n{float(row['energy_gev']):g} GeV\n{row['box_size_cm']} cm" for row in rows]
    x = np.arange(len(rows))
    deposited = np.asarray([float(row["deposited_fraction"]) for row in rows])
    escaped = np.asarray([float(row["escaped_fraction"]) for row in rows])
    invisible = np.asarray([float(row["invisible_fraction"]) for row in rows])
    fig, ax = plt.subplots(figsize=(max(7.2, 0.6 * len(rows)), 4.8))
    ax.bar(x, deposited, label="deposited", color="#2a9d8f")
    ax.bar(x, escaped, bottom=deposited, label="escaped", color="#577590")
    ax.bar(x, invisible, bottom=deposited + escaped, label="invisible", color="#9d4edd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("energy fraction")
    ax.set_title("GEANT4 benchmark particle comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_benchmark_particle_comparison.png", dpi=180)
    plt.close(fig)


def group_rows(rows: list[dict[str, object]], fields: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    cases_dir = output_dir / "geant4_material_benchmark_cases"

    if shutil.which("geant4-config") is None:
        print("GEANT4 material benchmarks skipped: geant4-config not found.")
        return 0

    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)

    cases = benchmark_cases(full=args.full)
    rows = []
    for case_id, case in enumerate(cases, start=1):
        print(
            f"[{case_id}] {case['particle']} E={float(case['energy_gev']):g} "
            f"material={case['material']} box={float(case['box_size_cm']):g} cm"
        )
        rows.append(run_case(root, cases_dir, case_id, case, args.timeout))

    write_csv(output_dir / "geant4_material_benchmarks.csv", rows)
    write_markdown(output_dir / "geant4_material_benchmarks.md", rows, full=args.full)
    make_plots(output_dir, rows)

    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]}
    print("GEANT4 material benchmark summary:")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"CSV: {output_dir / 'geant4_material_benchmarks.csv'}")
    print(f"MD:  {output_dir / 'geant4_material_benchmarks.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
