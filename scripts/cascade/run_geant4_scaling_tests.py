#!/usr/bin/env python3
"""Run GEANT4 local-box convergence and scaling diagnostics."""

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


CASES = [
    ("gamma_water", "gamma", 22, 0.1, "water"),
    ("electron_water", "electron", 11, 0.1, "water"),
    ("proton_hydrogen", "proton", 2212, 1.0, "hydrogen"),
]

MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    2212: 0.93827208816,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_secondary(path: Path, case_label: str, pdg: int, energy_gev: float, seed: int) -> None:
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
        "origin": f"geant4_scaling_{case_label}_seed_{seed}",
        "origin_backend": "geant4_scaling_tests",
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


def run_case(root: Path, cases_dir: Path, case_id: int, scan_type: str, case_label: str,
             particle: str, pdg: int, energy: float, material: str, density: float,
             box_size: float, physics_list: str, seed: int, timeout: float) -> dict[str, object]:
    case_dir = cases_dir / f"case_{case_id:04d}_{scan_type}_{case_label}_{density:g}rho_{box_size:g}cm_{physics_list}_seed{seed}"
    input_path = case_dir / "secondaries.jsonl"
    output_dir = case_dir / "out"
    write_secondary(input_path, case_label, pdg, energy, seed)
    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(input_path),
        str(output_dir),
        f"{box_size:.17g}",
        f"{density:.17g}",
        physics_list,
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
        "scan_type": scan_type,
        "case_label": case_label,
        "particle": particle,
        "pdg_id": pdg,
        "energy_gev": energy,
        "material": material,
        "density_g_cm3": density,
        "box_size_cm": box_size,
        "physics_list": physics_list,
        "seed": seed,
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


def build_matrix(full: bool) -> list[tuple[str, str, int, float, str, float, float, str, int]]:
    box_sizes = [1.0, 3.0, 10.0, 30.0, 100.0] if full else [1.0, 10.0, 100.0]
    densities = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0] if full else [1.0e-3, 1.0, 10.0]
    physics_lists = ["FTFP_BERT", "QGSP_BERT"]
    rows: list[tuple[str, str, int, float, str, float, float, str, int]] = []
    for case_label, particle, pdg, energy, material in CASES:
        for box_size in box_sizes:
            rows.append(("box_size", case_label, pdg, energy, material, 1.0, box_size, "FTFP_BERT", 12345))
        for density in densities:
            rows.append(("density", case_label, pdg, energy, material, density, 10.0, "FTFP_BERT", 12345))
        for physics_list in physics_lists:
            rows.append(("physics_list", case_label, pdg, energy, material, 1.0, 10.0, physics_list, 12345))
        for seed in [12345, 12345, 67890]:
            rows.append(("reproducibility", case_label, pdg, energy, material, 1.0, 10.0, "FTFP_BERT", seed))
    if not full:
        # Keep quick mode compact while preserving all diagnostic categories.
        return [row for row in rows if row[1] in {"gamma_water", "electron_water", "proton_hydrogen"}]
    return rows


def particle_from_case(case_label: str) -> str:
    return case_label.split("_")[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trend_text(rows: list[dict[str, object]], scan_type: str, x_field: str) -> list[str]:
    lines: list[str] = []
    groups = group_rows([row for row in rows if row["scan_type"] == scan_type and row["status"] == "PASS"], ["case_label"])
    for case_label, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: float(row[x_field]))
        values = [float(row["deposited_fraction"]) for row in ordered]
        if len(values) < 2:
            trend = "insufficient points"
        else:
            delta = values[-1] - values[0]
            trend = "increasing" if delta > 1.0e-6 else "decreasing" if delta < -1.0e-6 else "flat"
        lines.append(f"- `{case_label[0]}` deposited fraction trend vs `{x_field}`: {trend}")
    return lines


def write_markdown(path: Path, rows: list[dict[str, object]], full: bool) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    lines = [
        "# GEANT4 Local-Box Scaling Tests",
        "",
        "These are local homogeneous-material convergence/scaling diagnostics, not collapsar physics.",
        "",
        f"- mode: `{'full' if full else 'quick'}`",
        f"- total cases: `{len(rows)}`",
    ]
    for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]:
        lines.append(f"- {status}: `{counts.get(status, 0)}`")
    lines.extend(["", "## Trends", ""])
    lines.extend(trend_text(rows, "box_size", "box_size_cm"))
    lines.extend(trend_text(rows, "density", "density_g_cm3"))
    lines.extend(["", "## Physics List Comparison", ""])
    comparison_rows = [row for row in rows if row["scan_type"] == "physics_list" and row["status"] == "PASS"]
    comparison = group_rows(comparison_rows, ["case_label"])
    for case_label, group in sorted(comparison.items()):
        values = {row["physics_list"]: float(row["deposited_fraction"]) for row in group}
        if "FTFP_BERT" in values and "QGSP_BERT" in values:
            denom = max(abs(values["FTFP_BERT"]), 1.0e-300)
            rel = (values["QGSP_BERT"] - values["FTFP_BERT"]) / denom
            lines.append(f"- `{case_label[0]}` QGSP_BERT vs FTFP_BERT deposited fraction relative difference: `{rel:.6g}`")
    lines.extend(["", "## Interpretation", ""])
    lines.append("- Monotonicity is not required in this phase; trends are recorded for inspection.")
    lines.append("- Energy closure and finite fractions are the quick-test requirements.")
    lines.append("- Physical use requires later comparison with known benchmarks and domain validation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_rows(rows: list[dict[str, object]], fields: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def make_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    passing = [row for row in rows if row["status"] == "PASS"]

    def line_plot(scan_type: str, x_field: str, output_name: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        groups = group_rows([row for row in passing if row["scan_type"] == scan_type], ["case_label"])
        for key, group in sorted(groups.items()):
            group = sorted(group, key=lambda row: float(row[x_field]))
            ax.plot(
                [float(row[x_field]) for row in group],
                [float(row["deposited_fraction"]) for row in group],
                marker="o",
                linewidth=1.2,
                label=str(key[0]),
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("deposited fraction")
        ax.set_title(output_name.replace("_", " ").replace(".png", ""))
        if groups:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / output_name, dpi=180)
        plt.close(fig)

    line_plot("box_size", "box_size_cm", "geant4_scaling_box_size.png", "box size [cm]")
    line_plot("density", "density_g_cm3", "geant4_scaling_density.png", "density [g cm^-3]")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    comparison = [row for row in passing if row["scan_type"] == "physics_list"]
    labels = sorted({str(row["case_label"]) for row in comparison})
    x = np.arange(len(labels))
    width = 0.35
    for offset, physics_list in [(-0.5 * width, "FTFP_BERT"), (0.5 * width, "QGSP_BERT")]:
        values = []
        for label in labels:
            match = [row for row in comparison if row["case_label"] == label and row["physics_list"] == physics_list]
            values.append(float(match[0]["deposited_fraction"]) if match else np.nan)
        ax.bar(x + offset, values, width=width, label=physics_list)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("deposited fraction")
    ax.set_title("GEANT4 physics-list comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_scaling_physics_list_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    closure = np.asarray([float(row["relative_closure_error"]) for row in rows], dtype=float)
    closure = closure[np.isfinite(closure)]
    if closure.size:
        ax.hist(closure, bins=20, color="#457b9d", edgecolor="white")
    ax.set_xlabel("relative closure error")
    ax.set_ylabel("cases")
    ax.set_title("GEANT4 scaling energy closure")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_scaling_energy_closure.png", dpi=180)
    plt.close(fig)


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
    cases_dir = output_dir / "geant4_scaling_cases"

    if shutil.which("geant4-config") is None:
        print("GEANT4 scaling tests skipped: geant4-config not found.")
        return 0

    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)

    matrix = build_matrix(full=args.full)
    rows = []
    for case_id, (scan_type, case_label, pdg, energy, material, density, box_size, physics_list, seed) in enumerate(matrix, start=1):
        print(f"[{case_id}] {scan_type} {case_label} E={energy:g} density={density:g} box={box_size:g} list={physics_list} seed={seed}")
        rows.append(
            run_case(
                root,
                cases_dir,
                case_id,
                scan_type,
                case_label,
                particle_from_case(case_label),
                pdg,
                energy,
                material,
                density,
                box_size,
                physics_list,
                seed,
                args.timeout,
            )
        )

    write_csv(output_dir / "geant4_scaling_tests.csv", rows)
    write_markdown(output_dir / "geant4_scaling_tests.md", rows, full=args.full)
    make_plots(output_dir, rows)

    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]}
    print("GEANT4 scaling test summary:")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"CSV: {output_dir / 'geant4_scaling_tests.csv'}")
    print(f"MD:  {output_dir / 'geant4_scaling_tests.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
