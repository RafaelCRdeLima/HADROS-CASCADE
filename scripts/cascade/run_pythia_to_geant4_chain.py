#!/usr/bin/env python3
"""Run a controlled optional PYTHIA-proxy to GEANT4-local-box chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_checked(cmd: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True)


def write_primary_interactions(path: Path, n_events: int, energy_gev: float, seed: int,
                               density_g_cm3: float, weight: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n_events):
            event_id = i + 1
            row = {
                "event_id": event_id,
                "pdg_id": 14,
                "energy_gev": energy_gev,
                "px_gev": 0.0,
                "py_gev": 0.0,
                "pz_gev": energy_gev,
                "mass_gev": 0.0,
                "weight": weight,
                "seed": seed + event_id,
                "particle_label": "nu_mu_proxy_primary",
                "x_cm": 0.0,
                "y_cm": 0.0,
                "z_cm": float(event_id),
                "r_cm": float(event_id),
                "theta_rad": 0.0,
                "phi_rad": 0.0,
                "density_g_cm3": density_g_cm3,
                "temperature_mev": 0.0,
                "temperature_proxy": 0.0,
                "composition_proxy": 0.0,
                "electron_fraction": 0.5,
                "column_before_cm2": 0.0,
                "tau_before": 0.0,
                "region_label": "local_box_proxy",
                "region_class": "controlled_chain_demo",
                "interaction_model": "pythia_proxy_plumbing",
                "backend_name": "controlled_pythia_to_geant4_chain",
                "x_bjorken": -1.0,
                "q2_gev2": -1.0,
                "y_inelasticity": -1.0,
                "metadata": "technical integration only; not physical UHE neutrino DIS",
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_optional_executables(root: Path) -> None:
    run_checked(["make", "cascade_pythia_proxy", "HADROS_WITH_PYTHIA=ON"], root)
    run_checked(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], root)


def run_geant4_per_event(root: Path, output_dir: Path, args: argparse.Namespace,
                         primary_path: Path) -> None:
    secondaries = read_jsonl(output_dir / "pythia_secondaries.jsonl")
    by_event: dict[int, list[dict]] = defaultdict(list)
    for particle in secondaries:
        by_event[int(particle["event_id"])].append(particle)

    case_root = output_dir / "pythia_to_geant4_event_boxes"
    case_root.mkdir(parents=True, exist_ok=True)
    combined_results: list[str] = []
    combined_escaped: list[str] = []
    combined_budget_rows: list[dict[str, str]] = []

    for event_id in sorted(by_event):
        event_dir = case_root / f"event_{event_id:06d}"
        event_input = event_dir / "pythia_secondaries_event.jsonl"
        write_jsonl(event_input, by_event[event_id])
        run_checked(
            [
                str(root / "build" / "cascade_geant4_local_box"),
                str(event_input),
                str(event_dir),
                f"{args.box_size_cm:.17g}",
                f"{args.density_g_cm3:.17g}",
                args.physics_list,
                args.material,
                "geant4",
                str(primary_path),
                "--energy-convention",
                args.energy_convention,
            ],
            root,
        )
        combined_results.extend(
            line for line in (event_dir / "geant4_cascade_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        escaped_path = event_dir / "geant4_escaped_particles.jsonl"
        if escaped_path.exists():
            combined_escaped.extend(line for line in escaped_path.read_text(encoding="utf-8").splitlines() if line.strip())
        combined_budget_rows.extend(read_csv(event_dir / "geant4_energy_budget.csv"))

    (output_dir / "geant4_cascade_results.jsonl").write_text(
        "\n".join(combined_results) + ("\n" if combined_results else ""),
        encoding="utf-8",
    )
    (output_dir / "geant4_escaped_particles.jsonl").write_text(
        "\n".join(combined_escaped) + ("\n" if combined_escaped else ""),
        encoding="utf-8",
    )
    with (output_dir / "geant4_energy_budget.csv").open("w", encoding="utf-8", newline="") as handle:
        if combined_budget_rows:
            writer = csv.DictWriter(handle, fieldnames=list(combined_budget_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_budget_rows)


def run_geant4_multi_event_direct(root: Path, output_dir: Path, args: argparse.Namespace,
                                  primary_path: Path) -> None:
    print(
        "WARNING: multi-event direct GEANT4 mode is experimental and may crash "
        "with PYTHIA-rich secondary lists",
        flush=True,
    )
    run_checked(
        [
            str(root / "build" / "cascade_geant4_local_box"),
            str(output_dir / "pythia_secondaries.jsonl"),
            str(output_dir),
            f"{args.box_size_cm:.17g}",
            f"{args.density_g_cm3:.17g}",
            args.physics_list,
            args.material,
            "geant4",
            str(primary_path),
            "--energy-convention",
            args.energy_convention,
        ],
        root,
    )


def run_chain(root: Path, output_dir: Path, args: argparse.Namespace) -> None:
    primary_path = output_dir / "primary_interactions.jsonl"
    if args.regenerate_interactions or not primary_path.exists():
        write_primary_interactions(
            primary_path,
            args.n_events,
            args.energy_gev,
            args.seed,
            args.density_g_cm3,
            args.weight,
        )

    run_checked(
        [
            str(root / "build" / "cascade_pythia_proxy"),
            str(primary_path),
            str(output_dir),
            str(args.seed),
        ],
        root,
    )

    if args.per_event_geant4:
        run_geant4_per_event(root, output_dir, args, primary_path)
    else:
        run_geant4_multi_event_direct(root, output_dir, args, primary_path)


def build_summary(output_dir: Path) -> list[dict[str, float]]:
    primaries = {int(row["event_id"]): row for row in read_jsonl(output_dir / "primary_interactions.jsonl")}
    generator_rows = {int(float(row["event_id"])): row for row in read_csv(output_dir / "generator_summary.csv")}
    geant4_rows = {int(float(row["event_id"])): row for row in read_csv(output_dir / "geant4_energy_budget.csv")}

    secondaries_by_event: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(output_dir / "pythia_secondaries.jsonl"):
        secondaries_by_event[int(row["event_id"])].append(row)

    summary: list[dict[str, float]] = []
    for event_id in sorted(primaries):
        primary = primaries[event_id]
        generator = generator_rows[event_id]
        geant4 = geant4_rows[event_id]
        secondaries = secondaries_by_event[event_id]

        e_primary = float(primary["energy_gev"])
        e_secondaries = sum(float(particle["energy_gev"]) for particle in secondaries)
        multiplicity = float(len(secondaries))
        deposited = float(geant4["deposited_energy_gev"])
        escaped = float(geant4["escaped_energy_gev"])
        invisible = float(geant4["invisible_energy_gev"])
        untracked = float(geant4.get("untracked_energy_gev", 0.0))
        geant4_input = float(geant4["input_energy_gev"])
        accounted = deposited + escaped + invisible + untracked

        summary.append({
            "event_id": float(event_id),
            "weight": float(primary["weight"]),
            "E_primary": e_primary,
            "E_secondaries_total": e_secondaries,
            "E_generator_summary_total": float(generator["total_final_energy_gev"]),
            "E_geant4_input": geant4_input,
            "E_geant4_deposited": deposited,
            "E_geant4_escaped": escaped,
            "E_geant4_invisible": invisible,
            "E_geant4_untracked": untracked,
            "multiplicity": multiplicity,
            "closure_primary_to_secondaries": (e_secondaries - e_primary) / max(e_primary, 1.0),
            "closure_secondaries_to_geant4": (accounted - e_secondaries) / max(e_secondaries, 1.0),
            "deposited_fraction_of_secondaries": deposited / e_secondaries if e_secondaries > 0.0 else math.nan,
            "escaped_fraction_of_secondaries": escaped / e_secondaries if e_secondaries > 0.0 else math.nan,
            "invisible_fraction_of_secondaries": invisible / e_secondaries if e_secondaries > 0.0 else math.nan,
            "untracked_fraction_of_secondaries": untracked / e_secondaries if e_secondaries > 0.0 else math.nan,
        })
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError("No chain rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: list[dict[str, float]], args: argparse.Namespace) -> None:
    max_primary = max(abs(row["closure_primary_to_secondaries"]) for row in rows)
    max_geant4 = max(abs(row["closure_secondaries_to_geant4"]) for row in rows)
    mean_mult = float(np.mean([row["multiplicity"] for row in rows]))
    mean_dep = float(np.mean([row["deposited_fraction_of_secondaries"] for row in rows]))

    lines = [
        "# Controlled PYTHIA-to-GEANT4 Chain",
        "",
        "This is an optional technical integration diagnostic, not collapsar physics and not a publishable UHE neutrino-DIS calculation.",
        "",
        f"- events: `{len(rows)}`",
        f"- primary energy per event [GeV]: `{args.energy_gev:.12g}`",
        f"- material: `{args.material}`",
        f"- density [g cm^-3]: `{args.density_g_cm3:.12g}`",
        f"- box size [cm]: `{args.box_size_cm:.12g}`",
        f"- physics list: `{args.physics_list}`",
        f"- energy convention: `{args.energy_convention}`",
        f"- per-event GEANT4 execution: `{args.per_event_geant4}`",
        f"- mean PYTHIA final-state multiplicity: `{mean_mult:.12g}`",
        f"- mean GEANT4 deposited fraction of secondaries: `{mean_dep:.12g}`",
        f"- max |primary-to-secondaries closure|: `{max_primary:.12e}`",
        f"- max |secondaries-to-GEANT4 closure|: `{max_geant4:.12e}`",
        "",
        "PYTHIA is used here only as a standalone proxy/shower/hadronization plumbing layer.",
        "GEANT4 is used only as a local homogeneous-box material-response backend.",
        "This chain does not replace the HADROS GBW/IIM dipole treatment and does not model a full collapsar.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, float]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    event_id = np.asarray([row["event_id"] for row in rows], dtype=float)
    primary = np.asarray([row["E_primary"] for row in rows], dtype=float)
    secondaries = np.asarray([row["E_secondaries_total"] for row in rows], dtype=float)
    deposited = np.asarray([row["E_geant4_deposited"] for row in rows], dtype=float)
    escaped = np.asarray([row["E_geant4_escaped"] for row in rows], dtype=float)
    invisible = np.asarray([row["E_geant4_invisible"] for row in rows], dtype=float)
    untracked = np.asarray([row["E_geant4_untracked"] for row in rows], dtype=float)
    deposited_fraction = np.asarray([row["deposited_fraction_of_secondaries"] for row in rows], dtype=float)
    multiplicity = np.asarray([row["multiplicity"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    width = 0.18
    ax.bar(event_id - 1.5 * width, primary, width, label="primary", color="#2a9d8f")
    ax.bar(event_id - 0.5 * width, secondaries, width, label="PYTHIA secondaries", color="#457b9d")
    ax.bar(event_id + 0.5 * width, deposited + escaped + invisible + untracked, width, label="GEANT4 accounted", color="#e76f51")
    ax.set_xlabel("event id")
    ax.set_ylabel("energy [GeV]")
    ax.set_title("PYTHIA-to-GEANT4 chain energy flow")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "pythia_to_geant4_energy_flow.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(event_id, deposited_fraction, color="#6a4c93")
    ax.set_ylim(0.0, min(1.0, max(float(np.nanmax(deposited_fraction)) * 1.2, 0.02)))
    ax.set_xlabel("event id")
    ax.set_ylabel("deposited / secondary energy")
    ax.set_title("GEANT4 local-box deposited fraction")
    fig.tight_layout()
    fig.savefig(plots_dir / "pythia_to_geant4_deposited_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(multiplicity, deposited_fraction, s=42, color="#d62828")
    ax.set_xlabel("PYTHIA stable multiplicity")
    ax.set_ylabel("deposited / secondary energy")
    ax.set_title("Multiplicity versus local deposition")
    fig.tight_layout()
    fig.savefig(plots_dir / "pythia_to_geant4_multiplicity_vs_deposition.png", dpi=180)
    plt.close(fig)


def print_summary(output_dir: Path, rows: list[dict[str, float]]) -> None:
    print("Controlled PYTHIA-to-GEANT4 chain summary")
    print(f"events={len(rows)}")
    print(f"mean_multiplicity={np.mean([row['multiplicity'] for row in rows]):.12e}")
    print(f"mean_deposited_fraction={np.mean([row['deposited_fraction_of_secondaries'] for row in rows]):.12e}")
    print(f"max_primary_to_secondaries_closure={max(abs(row['closure_primary_to_secondaries']) for row in rows):.12e}")
    print(f"max_secondaries_to_geant4_closure={max(abs(row['closure_secondaries_to_geant4']) for row in rows):.12e}")
    print(f"summary_csv={output_dir / 'pythia_to_geant4_chain_summary.csv'}")
    print(f"summary_md={output_dir / 'pythia_to_geant4_chain_summary.md'}")
    print("NOTE: technical integration only; not physical UHE neutrino-DIS or collapsar output.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--n-events", type=int, default=3)
    parser.add_argument("--energy-gev", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--box-size-cm", type=float, default=10.0)
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--physics-list", default="FTFP_BERT")
    parser.add_argument("--material", default="water", choices=["water", "hydrogen"])
    parser.add_argument("--energy-convention", default="total", choices=["total", "kinetic"])
    parser.add_argument("--per-event-geant4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate-interactions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    missing = [name for name in ["pythia8-config", "geant4-config"] if not tool_available(name)]
    if missing:
        print("PYTHIA-to-GEANT4 chain skipped: missing " + ", ".join(missing) + ".")
        print("This is not fatal; external generators remain optional.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    build_optional_executables(root)
    run_chain(root, output_dir, args)
    rows = build_summary(output_dir)
    write_summary_csv(output_dir / "pythia_to_geant4_chain_summary.csv", rows)
    write_summary_md(output_dir / "pythia_to_geant4_chain_summary.md", rows, args)
    make_plots(output_dir, rows)
    print_summary(output_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
