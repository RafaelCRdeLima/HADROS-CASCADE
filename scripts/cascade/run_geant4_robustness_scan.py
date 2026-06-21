#!/usr/bin/env python3
"""Run a GEANT4 local-box robustness scan for HADROS-CASCADE diagnostics."""

from __future__ import annotations

import argparse
import csv
import itertools
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


PARTICLE_DEFAULTS = {
    "gamma": 22,
    "electron": 11,
    "positron": -11,
    "muon": 13,
    "proton": 2212,
    "neutron": 2112,
    "pi+": 211,
    "pi-": -211,
    "nu_mu": 14,
}

PDG_MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    -11: 0.00051099895,
    13: 0.1056583755,
    -13: 0.1056583755,
    2212: 0.93827208816,
    2112: 0.93956542052,
    211: 0.13957039,
    -211: 0.13957039,
    14: 0.0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_number_list(value: str | None, defaults: list[float]) -> list[float]:
    if not value:
        return defaults
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_string_list(value: str | None, defaults: list[str]) -> list[str]:
    if not value:
        return defaults
    return [item.strip() for item in value.split(",") if item.strip()]


def particles_from_arg(value: str | None, defaults: list[str]) -> list[tuple[str, int]]:
    labels = parse_string_list(value, defaults)
    particles = []
    for label in labels:
        if label in PARTICLE_DEFAULTS:
            particles.append((label, PARTICLE_DEFAULTS[label]))
        else:
            pdg = int(label)
            particles.append((f"pdg{pdg}", pdg))
    return particles


def signal_note(returncode: int) -> str:
    if returncode < 0:
        return f"signal_{-returncode}"
    if returncode == 139:
        return "sigsegv"
    return ""


def write_secondary(path: Path, event_id: int, particle_label: str, pdg: int, energy_gev: float) -> None:
    mass = PDG_MASS_GEV.get(pdg, 0.0)
    momentum = math.sqrt(max(energy_gev * energy_gev - mass * mass, 0.0))
    row = {
        "event_id": event_id,
        "parent_event_id": event_id,
        "pdg": pdg,
        "pdg_id": pdg,
        "energy_gev": energy_gev,
        "px_gev": 0.0,
        "py_gev": 0.0,
        "pz_gev": momentum,
        "mass_gev": mass,
        "weight": 1.0,
        "stable": 1,
        "origin": f"geant4_robustness_{particle_label}",
        "origin_backend": "geant4_robustness_scan",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")


def read_energy_budget(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {key: float(value) for key, value in rows[0].items() if key != "energy_convention"}


def classify_result(returncode: int, timed_out: bool, budget: dict[str, float] | None) -> tuple[str, float, str]:
    if timed_out:
        return "TIMEOUT", math.nan, "timeout"
    if returncode < 0 or returncode == 139:
        return "CRASH", math.nan, signal_note(returncode)
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
    closure_error = deposited + escaped + invisible + untracked - input_energy
    relative = abs(closure_error) / max(input_energy, 1.0)
    if relative > 1.0e-5:
        return "BAD_ENERGY", relative, "closure_tolerance"
    return "PASS", relative, ""


def run_case(root: Path, scan_dir: Path, case_id: int, particle_label: str, pdg: int,
             energy: float, density: float, box_size: float, physics_list: str,
             timeout: float) -> dict[str, object]:
    case_dir = scan_dir / f"case_{case_id:05d}_{particle_label}_{energy:g}GeV_{density:g}rho_{box_size:g}cm_{physics_list}"
    input_path = case_dir / "secondaries.jsonl"
    output_dir = case_dir / "out"
    write_secondary(input_path, 1, particle_label, pdg, energy)

    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(input_path),
        str(output_dir),
        f"{box_size:.17g}",
        f"{density:.17g}",
        physics_list,
        "hydrogen",
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

    budget = read_energy_budget(output_dir / "geant4_energy_budget.csv")
    status, relative_closure, diagnostic = classify_result(returncode, timed_out, budget)
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

    row = {
        "case_id": case_id,
        "particle": particle_label,
        "pdg_id": pdg,
        "energy_gev": energy,
        "density_g_cm3": density,
        "box_size_cm": box_size,
        "physics_list": physics_list,
        "status": status,
        "diagnostic": diagnostic,
        "exit_code": returncode,
        "runtime_s": runtime,
        "relative_closure_error": relative_closure,
        "input_energy_gev": budget.get("input_energy_gev", math.nan) if budget else math.nan,
        "deposited_energy_gev": budget.get("deposited_energy_gev", math.nan) if budget else math.nan,
        "escaped_energy_gev": budget.get("escaped_energy_gev", math.nan) if budget else math.nan,
        "invisible_energy_gev": budget.get("invisible_energy_gev", math.nan) if budget else math.nan,
        "output_dir": str(output_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], quick: bool) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1

    lines = [
        "# GEANT4 Local-Box Robustness Scan",
        "",
        "This is a technical stability diagnostic, not a physics validation.",
        "",
        f"- mode: `{'quick' if quick else 'custom/full'}`",
        f"- total cases: `{len(rows)}`",
    ]
    for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]:
        lines.append(f"- {status}: `{counts.get(status, 0)}`")
    lines.extend(["", "## Failed Or Non-Passing Cases", ""])
    nonpassing = [row for row in rows if row["status"] != "PASS"]
    if not nonpassing:
        lines.append("All scanned cases passed technical checks.")
    else:
        lines.append("| case | particle | E [GeV] | density | box [cm] | list | status | diagnostic | exit |")
        lines.append("|---:|---|---:|---:|---:|---|---|---|---:|")
        for row in nonpassing:
            lines.append(
                f"| {row['case_id']} | {row['particle']} | {row['energy_gev']} | "
                f"{row['density_g_cm3']} | {row['box_size_cm']} | {row['physics_list']} | "
                f"{row['status']} | {row['diagnostic']} | {row['exit_code']} |"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.append("- PASS means the command completed and energy bookkeeping closed within tolerance.")
    lines.append("- PASS does not mean the GEANT4 physics is validated for astrophysical use.")
    lines.append("- CRASH/TIMEOUT/BAD_ENERGY cases should be treated as blockers for direct transport in that region.")
    lines.append("- Physical use requires later comparison with known GEANT4 benchmarks and domain-specific validation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    statuses = ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]
    counts = [sum(1 for row in rows if row["status"] == status) for status in statuses]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(statuses, counts, color=["#2a9d8f", "#e9c46a", "#e76f51", "#577590", "#9d4edd"])
    ax.set_ylabel("cases")
    ax.set_title("GEANT4 local-box scan pass/fail")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_scan_pass_fail.png", dpi=180)
    plt.close(fig)

    closure = np.asarray([float(row["relative_closure_error"]) for row in rows], dtype=float)
    closure = closure[np.isfinite(closure)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if closure.size:
        ax.hist(closure, bins=20, color="#457b9d", edgecolor="white")
    ax.set_xlabel("relative closure error")
    ax.set_ylabel("cases")
    ax.set_title("GEANT4 local-box scan energy closure")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_scan_energy_closure.png", dpi=180)
    plt.close(fig)

    runtime = np.asarray([float(row["runtime_s"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(runtime[np.isfinite(runtime)], bins=20, color="#f4a261", edgecolor="white")
    ax.set_xlabel("runtime [s]")
    ax.set_ylabel("cases")
    ax.set_title("GEANT4 local-box scan runtime")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_scan_runtime.png", dpi=180)
    plt.close(fig)


def quick_defaults() -> tuple[list[str], list[float], list[float], list[float], list[str]]:
    return ["gamma", "electron", "proton", "nu_mu"], [1.0], [1.0], [10.0], ["FTFP_BERT"]


def full_defaults() -> tuple[list[str], list[float], list[float], list[float], list[str]]:
    return (
        ["gamma", "electron", "positron", "muon", "proton", "neutron", "pi+", "pi-", "nu_mu"],
        [1.0, 10.0, 50.0, 1000.0],
        [1.0e-6, 1.0e-3, 1.0, 10.0],
        [1.0, 10.0, 50.0, 100.0],
        ["FTFP_BERT", "QGSP_BERT"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--particles")
    parser.add_argument("--energies")
    parser.add_argument("--densities")
    parser.add_argument("--box-sizes")
    parser.add_argument("--physics-lists")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    scan_dir = output_dir / "geant4_robustness_cases"

    if shutil.which("geant4-config") is None:
        print("GEANT4 robustness scan skipped: geant4-config not found.")
        return 0

    defaults = full_defaults() if args.full else quick_defaults()
    particle_labels, energies, densities, box_sizes, physics_lists = defaults
    particles = particles_from_arg(args.particles, particle_labels)
    energies = parse_number_list(args.energies, energies)
    densities = parse_number_list(args.densities, densities)
    box_sizes = parse_number_list(args.box_sizes, box_sizes)
    physics_lists = parse_string_list(args.physics_lists, physics_lists)

    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)

    rows: list[dict[str, object]] = []
    combinations = itertools.product(particles, energies, densities, box_sizes, physics_lists)
    for case_id, ((particle_label, pdg), energy, density, box_size, physics_list) in enumerate(combinations, start=1):
        print(f"[{case_id}] {particle_label} E={energy:g} density={density:g} box={box_size:g} list={physics_list}")
        rows.append(run_case(root, scan_dir, case_id, particle_label, pdg, energy, density, box_size, physics_list, args.timeout))

    write_csv(output_dir / "geant4_robustness_scan.csv", rows)
    write_markdown(output_dir / "geant4_robustness_scan.md", rows, quick=not args.full)
    make_plots(output_dir, rows)

    summary = {status: sum(1 for row in rows if row["status"] == status) for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]}
    print("GEANT4 robustness scan summary:")
    for status, count in summary.items():
        print(f"  {status}: {count}")
    print(f"CSV: {output_dir / 'geant4_robustness_scan.csv'}")
    print(f"MD:  {output_dir / 'geant4_robustness_scan.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
