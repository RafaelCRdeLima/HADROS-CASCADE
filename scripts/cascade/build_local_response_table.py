#!/usr/bin/env python3
"""Build optional GEANT4 local homogeneous-box response tables."""

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


PARTICLES = {
    "gamma": 22,
    "electron": 11,
    "positron": -11,
    "muon": 13,
    "anti_muon": -13,
    "proton": 2212,
    "anti_proton": -2212,
    "neutron": 2112,
    "anti_neutron": -2112,
    "pi0": 111,
    "pi+": 211,
    "pi-": -211,
    "kaon+": 321,
    "kaon-": -321,
    "kaon0L": 130,
    "kaon0S": 310,
    "nu_e": 12,
    "anti_nu_e": -12,
    "nu_mu": 14,
    "anti_nu_mu": -14,
    "nu_tau": 16,
    "anti_nu_tau": -16,
}

PDG_LABELS = {pdg: label for label, pdg in PARTICLES.items()}

MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    -11: 0.00051099895,
    13: 0.1056583755,
    -13: 0.1056583755,
    2212: 0.93827208816,
    -2212: 0.93827208816,
    2112: 0.93956542052,
    -2112: 0.93956542052,
    111: 0.1349768,
    211: 0.13957039,
    -211: 0.13957039,
    321: 0.493677,
    -321: 0.493677,
    130: 0.497611,
    310: 0.497611,
    12: 0.0,
    -12: 0.0,
    14: 0.0,
    -14: 0.0,
    16: 0.0,
    -16: 0.0,
}

BASE_SECONDARY_PDGS = [22, 11, -11, 13, -13, 211, -211, 111, 2212, 2112, 14]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_floats(value: str | None, defaults: list[float]) -> list[float]:
    if not value:
        return defaults
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str | None, defaults: list[str]) -> list[str]:
    if not value:
        return defaults
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_particles(value: str | None, defaults: list[str]) -> list[tuple[str, int]]:
    labels = parse_strings(value, defaults)
    parsed: list[tuple[str, int]] = []
    for label in labels:
        if label in PARTICLES:
            parsed.append((label, PARTICLES[label]))
        else:
            pdg = int(label)
            parsed.append((f"pdg{pdg}", pdg))
    return parsed


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def kinetic_energy_from_secondary(row: dict[str, object]) -> float | None:
    pdg = int(row.get("pdg_id", row.get("pdg", 0)))
    try:
        total = float(row["energy_gev"])
    except (KeyError, TypeError, ValueError):
        return None
    mass_value = row.get("mass_gev", MASS_GEV.get(pdg))
    try:
        mass = float(mass_value) if mass_value is not None else MASS_GEV.get(pdg)
    except (TypeError, ValueError):
        mass = MASS_GEV.get(pdg)
    if mass is None:
        return total
    if total + 1.0e-12 < mass:
        return None
    return max(total - mass, 0.0)


def energy_grid_for_observed(values: list[float]) -> list[float]:
    finite = sorted(value for value in values if math.isfinite(value) and value > 0.0)
    if not finite:
        return [0.1, 1.0, 10.0]
    emin = max(0.5 * finite[0], 1.0e-9)
    emax = max(2.0 * finite[-1], emin * 10.0)
    if math.isclose(emin, emax):
        return [emin, emax]
    mid = math.sqrt(emin * emax)
    return sorted({emin, mid, emax})


def particles_from_secondaries(path: Path) -> dict[int, list[float]]:
    observed: dict[int, list[float]] = {}
    for row in read_jsonl(path):
        pdg = int(row.get("pdg_id", row.get("pdg", 0)))
        kinetic = kinetic_energy_from_secondary(row)
        if kinetic is None:
            continue
        observed.setdefault(pdg, []).append(kinetic)
    return observed


def write_secondaries(path: Path, particle: str, pdg: int, kinetic_energy_gev: float, n_events: int, seed: int,
                      event_offset: int = 0) -> None:
    mass = MASS_GEV.get(pdg, 0.0)
    total_energy = kinetic_energy_gev + mass
    momentum = math.sqrt(max(total_energy * total_energy - mass * mass, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event_index in range(n_events):
            event_id = event_offset + event_index + 1
            row = {
                "event_id": event_id,
                "parent_event_id": event_id,
                "pdg": pdg,
                "pdg_id": pdg,
                "energy_gev": kinetic_energy_gev,
                "px_gev": 0.0,
                "py_gev": 0.0,
                "pz_gev": momentum,
                "mass_gev": mass,
                "weight": 1.0,
                "stable": 1,
                "origin": f"local_response_table_{particle}_seed_{seed + event_index}",
                "origin_backend": "local_response_table",
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_budget_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_nonnegative(values: list[float]) -> bool:
    return all(math.isfinite(value) and value >= 0.0 for value in values)


def classify_budget(rows: list[dict[str, str]], closure_tolerance: float) -> tuple[str, str, float]:
    if not rows:
        return "FAIL", "missing_energy_budget", math.nan
    max_closure = 0.0
    for row in rows:
        values = [
            float(row["input_energy_gev"]),
            float(row["deposited_energy_gev"]),
            float(row["escaped_energy_gev"]),
            float(row["invisible_energy_gev"]),
            float(row.get("untracked_energy_gev", 0.0)),
        ]
        if not finite_nonnegative(values):
            return "BAD_ENERGY", "nan_or_negative_energy", math.nan
        closure = abs(values[1] + values[2] + values[3] + values[4] - values[0]) / max(values[0], 1.0)
        max_closure = max(max_closure, closure)
    if max_closure > closure_tolerance:
        return "BAD_ENERGY", "closure_tolerance", max_closure
    return "PASS", "", max_closure


def aggregate_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    n = len(rows)
    sums = {
        "input_energy_gev": 0.0,
        "deposited_energy_gev": 0.0,
        "escaped_energy_gev": 0.0,
        "invisible_energy_gev": 0.0,
        "untracked_energy_gev": 0.0,
        "escaped_particle_count": 0.0,
    }
    for row in rows:
        for key in sums:
            sums[key] += float(row.get(key, 0.0))
    denom = max(sums["input_energy_gev"], 1.0e-300)
    return {
        "input_energy_gev_mean": sums["input_energy_gev"] / max(n, 1),
        "deposited_fraction": sums["deposited_energy_gev"] / denom,
        "escaped_fraction": sums["escaped_energy_gev"] / denom,
        "invisible_fraction": sums["invisible_energy_gev"] / denom,
        "untracked_fraction": sums["untracked_energy_gev"] / denom,
        "multiplicity_escaped": sums["escaped_particle_count"] / max(n, 1),
    }


def run_one_event(root: Path, event_dir: Path, particle: str, pdg: int, energy: float, density: float,
                  material: str, box_size: float, physics_list: str, seed: int, event_index: int,
                  timeout: float) -> tuple[int, bool, float, list[dict[str, str]], str, str]:
    input_path = event_dir / "secondaries.jsonl"
    output_dir = event_dir / "out"
    write_secondaries(input_path, particle, pdg, energy, 1, seed, event_offset=event_index)
    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(input_path),
        str(output_dir),
        f"{box_size:.17g}",
        f"{density:.17g}",
        physics_list,
        material,
        "geant4",
        "--energy-convention",
        "kinetic",
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
    budget_rows = read_budget_rows(output_dir / "geant4_energy_budget.csv")
    return returncode, timed_out, runtime, budget_rows, stdout, stderr


def run_case(root: Path, cases_dir: Path, case_id: int, particle: str, pdg: int, energy: float,
             density: float, material: str, box_size: float, physics_list: str, n_events: int,
             seed: int, timeout: float, closure_tolerance: float) -> dict[str, object]:
    case_dir = cases_dir / (
        f"case_{case_id:05d}_{particle}_{energy:g}GeV_{density:g}rho_"
        f"{material}_{box_size:g}cm_{physics_list}"
    )
    event_root = case_dir / "events"
    all_budget_rows: list[dict[str, str]] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    runtime = 0.0
    returncode = 0
    timed_out = False

    for event_index in range(n_events):
        event_dir = event_root / f"event_{event_index + 1:06d}"
        code, event_timed_out, event_runtime, budget_rows, stdout, stderr = run_one_event(
            root, event_dir, particle, pdg, energy, density, material, box_size,
            physics_list, seed + event_index, event_index, timeout,
        )
        runtime += event_runtime
        stdout_parts.append(f"=== event {event_index + 1} returncode={code} ===\n{stdout}")
        stderr_parts.append(f"=== event {event_index + 1} returncode={code} ===\n{stderr}")
        if code != 0:
            returncode = code
            timed_out = event_timed_out
            break
        all_budget_rows.extend(budget_rows)

    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("\n".join(stdout_parts), encoding="utf-8", errors="replace")
    stderr_path.write_text("\n".join(stderr_parts), encoding="utf-8", errors="replace")

    if timed_out:
        status, diagnostic, closure = "TIMEOUT", "timeout", math.nan
        aggregate = {}
    elif returncode < 0 or returncode == 139:
        status, diagnostic, closure = "CRASH", f"signal_{-returncode}" if returncode < 0 else "sigsegv", math.nan
        aggregate = {}
    elif returncode != 0:
        status, diagnostic, closure = "FAIL", f"exit_{returncode}", math.nan
        aggregate = {}
    else:
        status, diagnostic, closure = classify_budget(all_budget_rows, closure_tolerance)
        aggregate = aggregate_rows(all_budget_rows) if all_budget_rows else {}

    return {
        "case_id": case_id,
        "particle": particle,
        "pdg_id": pdg,
        "energy_gev": energy,
        "energy_convention": "kinetic",
        "density_g_cm3": density,
        "material": material,
        "box_size_cm": box_size,
        "physics_list": physics_list,
        "n_events": n_events,
        "seed": seed,
        "status": status,
        "diagnostic": diagnostic,
        "exit_code": returncode,
        "runtime_s": runtime,
        "input_energy_gev_mean": aggregate.get("input_energy_gev_mean", math.nan),
        "deposited_fraction": aggregate.get("deposited_fraction", math.nan),
        "escaped_fraction": aggregate.get("escaped_fraction", math.nan),
        "invisible_fraction": aggregate.get("invisible_fraction", math.nan),
        "untracked_fraction": aggregate.get("untracked_fraction", math.nan),
        "multiplicity_escaped": aggregate.get("multiplicity_escaped", math.nan),
        "energy_closure_error": closure,
        "closure_tolerance": closure_tolerance,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_dir": str(event_root),
    }


def build_matrix(args: argparse.Namespace) -> list[tuple[str, int, float, float, str, float, str]]:
    if args.from_secondaries is not None:
        return build_matrix_from_secondaries(args)
    particles = parse_particles(args.particles, ["gamma", "electron", "proton", "nu_mu"])
    energies = parse_floats(args.energies_gev, [0.1, 1.0, 10.0])
    densities = parse_floats(args.densities_g_cm3, [1.0e-3, 1.0, 10.0])
    materials = parse_strings(args.materials, ["hydrogen", "water"])
    box_sizes = parse_floats(args.box_sizes_cm, [10.0, 100.0])
    physics_lists = parse_strings(args.physics_lists, ["FTFP_BERT"] if not args.full else ["FTFP_BERT", "QGSP_BERT"])
    return [
        (particle, pdg, energy, density, material, box_size, physics_list)
        for particle, pdg in particles
        for energy in energies
        for density in densities
        for material in materials
        for box_size in box_sizes
        for physics_list in physics_lists
    ]


def build_matrix_from_secondaries(args: argparse.Namespace) -> list[tuple[str, int, float, float, str, float, str]]:
    observed = particles_from_secondaries(args.from_secondaries)
    pdgs = set(observed) | set(BASE_SECONDARY_PDGS)
    for kaon_pdg in [321, -321, 130, 310]:
        if kaon_pdg in observed:
            pdgs.add(kaon_pdg)
    for neutrino_pdg in [12, -12, 14, -14, 16, -16]:
        if neutrino_pdg in observed:
            pdgs.add(neutrino_pdg)
    if args.particles:
        pdgs.update(pdg for _, pdg in parse_particles(args.particles, []))

    densities = parse_floats(args.densities_g_cm3, [1.0e-3, 1.0, 10.0])
    materials = parse_strings(args.materials, ["hydrogen", "water"])
    box_sizes = parse_floats(args.box_sizes_cm, [10.0, 100.0])
    physics_lists = parse_strings(args.physics_lists, ["FTFP_BERT"] if not args.full else ["FTFP_BERT", "QGSP_BERT"])

    matrix: list[tuple[str, int, float, float, str, float, str]] = []
    for pdg in sorted(pdgs, key=lambda item: (abs(item), item)):
        label = PDG_LABELS.get(pdg, f"pdg{pdg}")
        energies = energy_grid_for_observed(observed.get(pdg, []))
        for energy in energies:
            for density in densities:
                for material in materials:
                    for box_size in box_sizes:
                        for physics_list in physics_lists:
                            matrix.append((label, pdg, energy, density, material, box_size, physics_list))
    return matrix


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_npz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_fields = [
        "case_id",
        "pdg_id",
        "energy_gev",
        "density_g_cm3",
        "box_size_cm",
        "n_events",
        "runtime_s",
        "deposited_fraction",
        "escaped_fraction",
        "invisible_fraction",
        "untracked_fraction",
        "multiplicity_escaped",
        "energy_closure_error",
        "closure_tolerance",
    ]
    arrays = {field: np.asarray([float(row[field]) for row in rows], dtype=np.float64) for field in numeric_fields}
    arrays.update({
        "particle": np.asarray([str(row["particle"]) for row in rows]),
        "material": np.asarray([str(row["material"]) for row in rows]),
        "physics_list": np.asarray([str(row["physics_list"]) for row in rows]),
        "status": np.asarray([str(row["status"]) for row in rows]),
        "energy_convention": np.asarray([str(row["energy_convention"]) for row in rows]),
    })
    np.savez(path, **arrays)


def trend(values: list[tuple[float, float]]) -> str:
    ordered = sorted(values)
    finite = [(x, y) for x, y in ordered if math.isfinite(y)]
    if len(finite) < 2:
        return "insufficient points"
    delta = finite[-1][1] - finite[0][1]
    if delta > 1.0e-6:
        return "increasing"
    if delta < -1.0e-6:
        return "decreasing"
    return "flat"


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    counts = {status: sum(str(row["status"]) == status for row in rows) for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]}
    passing = [row for row in rows if row["status"] == "PASS"]
    mean_dep = float(np.mean([float(row["deposited_fraction"]) for row in passing])) if passing else math.nan
    max_closure = max([float(row["energy_closure_error"]) for row in passing], default=math.nan)

    lines = [
        "# Local Response Table",
        "",
        "This is a GEANT4 local homogeneous-box response table, not collapsar physics.",
        "",
        f"- mode: `{'full' if args.full else 'quick'}`",
        f"- energy convention: `kinetic`",
        f"- from secondaries: `{args.from_secondaries if args.from_secondaries else 'none'}`",
        f"- total cases: `{len(rows)}`",
        f"- n_events per case: `{args.n_events}`",
        f"- closure tolerance: `{args.closure_tolerance:.12g}`",
        f"- mean deposited fraction over passing cases: `{mean_dep:.12g}`",
        f"- max closure error over passing cases: `{max_closure:.12e}`",
    ]
    for status, count in counts.items():
        lines.append(f"- {status}: `{count}`")

    lines.extend(["", "## Recorded Columns", ""])
    lines.append("`PDG id`, `energy_gev`, `density_g_cm3`, `material`, `box_size_cm`, `physics_list`,")
    lines.append("`deposited_fraction`, `escaped_fraction`, `invisible_fraction`, `untracked_fraction`,")
    lines.append("`multiplicity_escaped`, `energy_closure_error`, and `runtime_s`.")

    lines.extend(["", "## Simple Trends", ""])
    for particle in sorted({str(row["particle"]) for row in passing}):
        subset = [row for row in passing if row["particle"] == particle and float(row["density_g_cm3"]) == 1.0]
        values = [(float(row["energy_gev"]), float(row["deposited_fraction"])) for row in subset]
        lines.append(f"- `{particle}` deposited fraction vs energy at rho=1 g/cm3: {trend(values)}")
    for particle in sorted({str(row["particle"]) for row in passing}):
        subset = [row for row in passing if row["particle"] == particle and float(row["energy_gev"]) == 1.0]
        values = [(float(row["density_g_cm3"]), float(row["deposited_fraction"])) for row in subset]
        lines.append(f"- `{particle}` deposited fraction vs density at E=1 GeV: {trend(values)}")

    lines.extend(["", "## Scope", ""])
    lines.append("- GEANT4 is used only as a local response microscope.")
    lines.append("- The table is not a global torus, Kerr, collapsar, plasma, PeV/EeV, or radiative-transfer model.")
    lines.append("- Future HADROS stages may interpolate this table; this script only builds local homogeneous-box data.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_rows(rows: list[dict[str, object]], fields: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def make_plots(output_dir: Path, rows: list[dict[str, object]], expanded: bool = False) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    passing = [row for row in rows if row["status"] == "PASS"]

    def line_groups(scan_x: str, output_name: str, xlabel: str, fixed: dict[str, object]) -> None:
        selected = []
        for row in passing:
            keep = True
            for key, value in fixed.items():
                if str(row[key]) != str(value):
                    keep = False
                    break
            if keep:
                selected.append(row)
        groups = group_rows(selected, ["particle", "material", "box_size_cm"])
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for key, group in sorted(groups.items()):
            ordered = sorted(group, key=lambda row: float(row[scan_x]))
            ax.plot(
                [float(row[scan_x]) for row in ordered],
                [float(row["deposited_fraction"]) for row in ordered],
                marker="o",
                linewidth=1.1,
                label=f"{key[0]} {key[1]} {float(key[2]):g} cm",
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("deposited fraction")
        ax.set_title(output_name.replace("_", " ").replace(".png", ""))
        if groups:
            ax.legend(fontsize=7, ncols=2)
        fig.tight_layout()
        fig.savefig(plots_dir / output_name, dpi=180)
        plt.close(fig)

    line_groups("energy_gev", "local_response_deposited_vs_energy.png", "kinetic energy [GeV]", {"density_g_cm3": 1.0, "physics_list": "FTFP_BERT"})
    line_groups("density_g_cm3", "local_response_deposited_vs_density.png", "density [g cm^-3]", {"energy_gev": 1.0, "physics_list": "FTFP_BERT"})

    groups = group_rows([row for row in passing if str(row["physics_list"]) == "FTFP_BERT" and float(row["density_g_cm3"]) == 1.0], ["particle", "material", "box_size_cm"])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: float(row["energy_gev"]))
        ax.plot(
            [float(row["energy_gev"]) for row in ordered],
            [float(row["escaped_fraction"]) for row in ordered],
            marker="o",
            linewidth=1.1,
            label=f"{key[0]} {key[1]} {float(key[2]):g} cm",
        )
    ax.set_xscale("log")
    ax.set_xlabel("kinetic energy [GeV]")
    ax.set_ylabel("escaped fraction")
    ax.set_title("local response escaped vs energy")
    if groups:
        ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "local_response_escaped_vs_energy.png", dpi=180)
    plt.close(fig)

    particle_means = []
    for particle, group in sorted(group_rows(passing, ["particle"]).items()):
        particle_means.append((particle[0], float(np.mean([float(row["deposited_fraction"]) for row in group]))))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar([item[0] for item in particle_means], [item[1] for item in particle_means], color="#2a9d8f")
    ax.set_ylabel("mean deposited fraction")
    ax.set_title("local response particle comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "local_response_particle_comparison.png", dpi=180)
    plt.close(fig)

    if expanded:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar([item[0] for item in particle_means], [item[1] for item in particle_means], color="#457b9d")
        ax.set_ylabel("mean deposited fraction")
        ax.set_title("local response expanded particle comparison")
        fig.tight_layout()
        fig.savefig(plots_dir / "local_response_expanded_particle_comparison.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Use the default compact table grid.")
    mode.add_argument("--full", action="store_true", help="Also include QGSP_BERT unless overridden.")
    parser.add_argument("--particles")
    parser.add_argument("--energies-gev")
    parser.add_argument("--densities-g-cm3")
    parser.add_argument("--materials")
    parser.add_argument("--box-sizes-cm")
    parser.add_argument("--physics-lists")
    parser.add_argument("--n-events", type=int, default=3)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--from-secondaries", type=Path, help="Build an expanded grid from observed secondary PDG IDs and kinetic-energy ranges.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--closure-tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    cases_dir = output_dir / "local_response_table_cases"
    table_stem = "local_response_table_expanded" if args.from_secondaries is not None else "local_response_table"
    if args.from_secondaries is not None and not args.from_secondaries.is_absolute():
        args.from_secondaries = (root / args.from_secondaries).resolve()

    if shutil.which("geant4-config") is None:
        print("local response table skipped: geant4-config not found.")
        return 0

    if args.n_events <= 0:
        raise ValueError("--n-events must be positive")

    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)

    matrix = build_matrix(args)
    rows: list[dict[str, object]] = []
    for case_id, (particle, pdg, energy, density, material, box_size, physics_list) in enumerate(matrix, start=1):
        print(
            f"[{case_id}/{len(matrix)}] {particle} pdg={pdg} Ekin={energy:g} "
            f"rho={density:g} material={material} box={box_size:g} list={physics_list}"
        )
        rows.append(run_case(root, cases_dir, case_id, particle, pdg, energy, density, material,
                             box_size, physics_list, args.n_events, args.seed + case_id,
                             args.timeout, args.closure_tolerance))

    write_csv(output_dir / f"{table_stem}.csv", rows)
    write_markdown(output_dir / f"{table_stem}.md", rows, args)
    write_npz(output_dir / f"{table_stem}.npz", rows)
    make_plots(output_dir, rows, expanded=args.from_secondaries is not None)

    counts = {status: sum(str(row["status"]) == status for row in rows) for status in ["PASS", "FAIL", "CRASH", "TIMEOUT", "BAD_ENERGY"]}
    print("Local response table summary:")
    for status, count in counts.items():
        print(f"  {status}: {count}")
    print(f"CSV: {output_dir / f'{table_stem}.csv'}")
    print(f"MD:  {output_dir / f'{table_stem}.md'}")
    print(f"NPZ: {output_dir / f'{table_stem}.npz'}")
    print("NOTE: local homogeneous-box response table only; not collapsar, Kerr, plasma, or PeV/EeV validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
