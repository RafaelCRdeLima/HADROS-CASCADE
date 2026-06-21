#!/usr/bin/env python3
"""Run the optional GEANT4 local homogeneous-box cascade demo."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def geant4_available() -> bool:
    return shutil.which("geant4-config") is not None


def pythia_available() -> bool:
    return shutil.which("pythia8-config") is not None


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                key: float(value)
                for key, value in row.items()
                if key not in {"energy_convention", "uhe_transport_policy"}
            }
            for row in csv.DictReader(handle)
        ]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_analytic_demo(root: Path, output_dir: Path, args: argparse.Namespace) -> None:
    subprocess.run(
        [
            "python3",
            str(root / "scripts" / "cascade" / "run_analytic_cascade_demo.py"),
            "--output-dir",
            str(output_dir),
            "--n-events",
            str(args.n_events),
            "--energy-gev",
            f"{args.energy_gev:.17g}",
            "--seed",
            str(args.seed),
            "--fixed-y",
            "0.25",
            "--regenerate-interactions",
        ],
        cwd=root,
        check=True,
    )


def ensure_secondaries(root: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    pythia_secondaries = output_dir / "pythia_secondaries.jsonl"
    if args.reuse_existing and pythia_secondaries.exists():
        return pythia_secondaries

    if pythia_available():
        subprocess.run(
            [
                "python3",
                str(root / "scripts" / "cascade" / "run_pythia_proxy_demo.py"),
                "--output-dir",
                str(output_dir),
                "--n-events",
                str(args.n_events),
                "--energy-gev",
                f"{args.energy_gev:.17g}",
                "--seed",
                str(args.seed),
            ],
            cwd=root,
            check=True,
        )
        if pythia_secondaries.exists():
            return pythia_secondaries

    print("PYTHIA secondaries unavailable; using analytic secondaries for GEANT4 local-box plumbing.")
    run_analytic_demo(root, output_dir, args)
    return output_dir / "secondaries.jsonl"


def build_geant4_app(root: Path) -> None:
    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)


def run_geant4_app(root: Path, output_dir: Path, secondaries: Path, args: argparse.Namespace) -> None:
    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(secondaries),
        str(output_dir),
        f"{args.box_size_cm:.17g}",
        f"{args.density_g_cm3:.17g}",
        args.physics_list,
        args.material,
        args.transport_mode,
        "--geant4-safety-mode",
        args.geant4_safety_mode,
        "--uhe-transport-policy",
        args.uhe_transport_policy,
        "--geant4-hadron-max-kinetic-gev",
        f"{args.geant4_hadron_max_kinetic_gev:.17g}",
        "--geant4-lepton-max-kinetic-gev",
        f"{args.geant4_lepton_max_kinetic_gev:.17g}",
        "--geant4-photon-max-kinetic-gev",
        f"{args.geant4_photon_max_kinetic_gev:.17g}",
    ]
    if args.geant4_one_particle_per_run:
        cmd.append("--geant4-one-particle-per-run")
    if args.debug_single_particle:
        cmd.append("--debug-single-particle")
    if args.energy_convention:
        cmd.extend(["--energy-convention", args.energy_convention])
    interactions = output_dir / "primary_interactions.jsonl"
    if interactions.exists():
        cmd.append(str(interactions))
    subprocess.run(cmd, cwd=root, check=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_budget_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_geant4_one_particle_processes(root: Path, output_dir: Path, secondaries: Path, args: argparse.Namespace) -> None:
    particles = read_jsonl(secondaries)
    cases_dir = output_dir / "geant4_one_particle_processes"
    cases_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[int, dict[str, float]] = {}
    escaped_all: list[dict] = []
    unsupported_uhe_all: list[dict] = []
    crash_case: Path | None = None

    for index, particle in enumerate(particles):
        event_id = int(particle.get("event_id", 0))
        case_dir = cases_dir / f"case_{index:05d}_event_{event_id}_pdg_{int(particle.get('pdg_id', particle.get('pdg', 0)))}"
        case_input = case_dir / "input.jsonl"
        write_jsonl(case_input, [particle])
        cmd = [
            str(root / "build" / "cascade_geant4_local_box"),
            str(case_input),
            str(case_dir),
            f"{args.box_size_cm:.17g}",
            f"{args.density_g_cm3:.17g}",
            args.physics_list,
            args.material,
            "geant4",
            "--geant4-safety-mode",
            args.geant4_safety_mode,
            "--uhe-transport-policy",
            args.uhe_transport_policy,
            "--geant4-hadron-max-kinetic-gev",
            f"{args.geant4_hadron_max_kinetic_gev:.17g}",
            "--geant4-lepton-max-kinetic-gev",
            f"{args.geant4_lepton_max_kinetic_gev:.17g}",
            "--geant4-photon-max-kinetic-gev",
            f"{args.geant4_photon_max_kinetic_gev:.17g}",
            "--energy-convention",
            args.energy_convention,
        ]
        if args.debug_single_particle:
            cmd.append("--debug-single-particle")
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (case_dir / "stdout_stderr.txt").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            crash_case = output_dir / "minimal_geant4_crash_case.jsonl"
            write_jsonl(crash_case, [particle])
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)

        for row in read_budget_csv(case_dir / "geant4_energy_budget.csv"):
            bucket = combined.setdefault(event_id, {
                "event_id": float(event_id),
                "input_energy_gev": 0.0,
                "deposited_energy_gev": 0.0,
                "escaped_energy_gev": 0.0,
                "invisible_energy_gev": 0.0,
                "untracked_energy_gev": 0.0,
                "unsupported_uhe_energy_gev": 0.0,
                "escaped_unsupported_uhe_energy_gev": 0.0,
                "n_unsupported_uhe_particles": 0.0,
                "escaped_particle_count": 0.0,
            })
            for key in [
                "input_energy_gev",
                "deposited_energy_gev",
                "escaped_energy_gev",
                "invisible_energy_gev",
                "untracked_energy_gev",
                "unsupported_uhe_energy_gev",
                "escaped_unsupported_uhe_energy_gev",
                "n_unsupported_uhe_particles",
                "escaped_particle_count",
            ]:
                bucket[key] += float(row[key])
        escaped_all.extend(read_jsonl(case_dir / "geant4_escaped_particles.jsonl"))
        unsupported_path = case_dir / "geant4_unsupported_uhe_particles.jsonl"
        if unsupported_path.exists():
            unsupported_uhe_all.extend(read_jsonl(unsupported_path))

    results = [combined[event_id] for event_id in sorted(combined)]
    with (output_dir / "geant4_energy_budget.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "event_id",
            "input_energy_gev",
            "deposited_energy_gev",
            "escaped_energy_gev",
            "invisible_energy_gev",
            "untracked_energy_gev",
            "unsupported_uhe_energy_gev",
            "escaped_unsupported_uhe_energy_gev",
            "n_unsupported_uhe_particles",
            "accounted_energy_gev",
            "closure_error_gev",
            "escaped_particle_count",
            "energy_convention",
            "uhe_transport_policy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            accounted = (
                row["deposited_energy_gev"]
                + row["escaped_energy_gev"]
                + row["invisible_energy_gev"]
                + row["untracked_energy_gev"]
                + row["escaped_unsupported_uhe_energy_gev"]
            )
            writer.writerow({
                "event_id": int(row["event_id"]),
                "input_energy_gev": row["input_energy_gev"],
                "deposited_energy_gev": row["deposited_energy_gev"],
                "escaped_energy_gev": row["escaped_energy_gev"],
                "invisible_energy_gev": row["invisible_energy_gev"],
                "untracked_energy_gev": row["untracked_energy_gev"],
                "unsupported_uhe_energy_gev": row["unsupported_uhe_energy_gev"],
                "escaped_unsupported_uhe_energy_gev": row["escaped_unsupported_uhe_energy_gev"],
                "n_unsupported_uhe_particles": int(row["n_unsupported_uhe_particles"]),
                "accounted_energy_gev": accounted,
                "closure_error_gev": accounted - row["input_energy_gev"],
                "escaped_particle_count": int(row["escaped_particle_count"]),
                "energy_convention": args.energy_convention,
                "uhe_transport_policy": args.uhe_transport_policy,
            })

    with (output_dir / "geant4_cascade_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps({
                "event_id": int(row["event_id"]),
                "input_energy_gev": row["input_energy_gev"],
                "deposited_energy_gev": row["deposited_energy_gev"],
                "escaped_energy_gev": row["escaped_energy_gev"],
                "invisible_energy_gev": row["invisible_energy_gev"],
                "untracked_energy_gev": row["untracked_energy_gev"],
                "unsupported_uhe_energy_gev": row["unsupported_uhe_energy_gev"],
                "escaped_unsupported_uhe_energy_gev": row["escaped_unsupported_uhe_energy_gev"],
                "n_unsupported_uhe_particles": int(row["n_unsupported_uhe_particles"]),
                "escaped_particle_count": int(row["escaped_particle_count"]),
                "energy_convention": args.energy_convention,
                "uhe_transport_policy": args.uhe_transport_policy,
                "backend": "Geant4LocalBoxBackendOneParticlePerProcess",
            }, sort_keys=True) + "\n")
    write_jsonl(output_dir / "geant4_escaped_particles.jsonl", escaped_all)
    write_jsonl(output_dir / "geant4_unsupported_uhe_particles.jsonl", unsupported_uhe_all)
    write_isolated_safety_report(output_dir, particles, args)


def run_geant4_event_processes(root: Path, output_dir: Path, secondaries: Path, args: argparse.Namespace) -> None:
    particles = read_jsonl(secondaries)
    events: dict[int, list[dict]] = {}
    for particle in particles:
        events.setdefault(int(particle.get("event_id", 0)), []).append(particle)

    cases_dir = output_dir / "geant4_event_processes"
    cases_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, str]] = []
    escaped_all: list[dict] = []
    unsupported_uhe_all: list[dict] = []
    fallback_events: list[int] = []

    def single_particle_fallback(event_id: int, event_particles: list[dict], parent_case_dir: Path) -> dict[str, str]:
        fallback_events.append(event_id)
        fallback_dir = parent_case_dir / "particle_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        bucket = {
            "event_id": float(event_id),
            "input_energy_gev": 0.0,
            "deposited_energy_gev": 0.0,
            "escaped_energy_gev": 0.0,
            "invisible_energy_gev": 0.0,
            "untracked_energy_gev": 0.0,
            "unsupported_uhe_energy_gev": 0.0,
            "escaped_unsupported_uhe_energy_gev": 0.0,
            "n_unsupported_uhe_particles": 0.0,
            "escaped_particle_count": 0.0,
        }
        for index, particle in enumerate(event_particles):
            pdg = int(particle.get("pdg_id", particle.get("pdg", 0)))
            particle_dir = fallback_dir / f"particle_{index:05d}_pdg_{pdg}"
            particle_input = particle_dir / "input.jsonl"
            write_jsonl(particle_input, [particle])
            cmd = [
                str(root / "build" / "cascade_geant4_local_box"),
                str(particle_input),
                str(particle_dir),
                f"{args.box_size_cm:.17g}",
                f"{args.density_g_cm3:.17g}",
                args.physics_list,
                args.material,
                "geant4",
                "--geant4-safety-mode",
                args.geant4_safety_mode,
                "--uhe-transport-policy",
                args.uhe_transport_policy,
                "--geant4-hadron-max-kinetic-gev",
                f"{args.geant4_hadron_max_kinetic_gev:.17g}",
                "--geant4-lepton-max-kinetic-gev",
                f"{args.geant4_lepton_max_kinetic_gev:.17g}",
                "--geant4-photon-max-kinetic-gev",
                f"{args.geant4_photon_max_kinetic_gev:.17g}",
                "--energy-convention",
                args.energy_convention,
            ]
            proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            (particle_dir / "stdout_stderr.txt").write_text(proc.stdout, encoding="utf-8")
            if proc.returncode != 0:
                write_jsonl(output_dir / "minimal_geant4_crash_case.jsonl", [particle])
                raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
            for row in read_budget_csv(particle_dir / "geant4_energy_budget.csv"):
                for key in [
                    "input_energy_gev",
                    "deposited_energy_gev",
                    "escaped_energy_gev",
                    "invisible_energy_gev",
                    "untracked_energy_gev",
                    "unsupported_uhe_energy_gev",
                    "escaped_unsupported_uhe_energy_gev",
                    "n_unsupported_uhe_particles",
                    "escaped_particle_count",
                ]:
                    bucket[key] += float(row.get(key, 0.0) or 0.0)
            escaped_all.extend(read_jsonl(particle_dir / "geant4_escaped_particles.jsonl"))
            unsupported_path = particle_dir / "geant4_unsupported_uhe_particles.jsonl"
            if unsupported_path.exists():
                unsupported_uhe_all.extend(read_jsonl(unsupported_path))
        accounted = (
            bucket["deposited_energy_gev"]
            + bucket["escaped_energy_gev"]
            + bucket["invisible_energy_gev"]
            + bucket["untracked_energy_gev"]
            + bucket["escaped_unsupported_uhe_energy_gev"]
        )
        return {
            "event_id": str(int(bucket["event_id"])),
            "input_energy_gev": f"{bucket['input_energy_gev']:.17g}",
            "deposited_energy_gev": f"{bucket['deposited_energy_gev']:.17g}",
            "escaped_energy_gev": f"{bucket['escaped_energy_gev']:.17g}",
            "invisible_energy_gev": f"{bucket['invisible_energy_gev']:.17g}",
            "untracked_energy_gev": f"{bucket['untracked_energy_gev']:.17g}",
            "unsupported_uhe_energy_gev": f"{bucket['unsupported_uhe_energy_gev']:.17g}",
            "escaped_unsupported_uhe_energy_gev": f"{bucket['escaped_unsupported_uhe_energy_gev']:.17g}",
            "n_unsupported_uhe_particles": str(int(bucket["n_unsupported_uhe_particles"])),
            "accounted_energy_gev": f"{accounted:.17g}",
            "closure_error_gev": f"{accounted - bucket['input_energy_gev']:.17g}",
            "escaped_particle_count": str(int(bucket["escaped_particle_count"])),
            "energy_convention": args.energy_convention,
            "uhe_transport_policy": args.uhe_transport_policy,
        }

    for event_id in sorted(events):
        case_dir = cases_dir / f"event_{event_id}"
        case_input = case_dir / "input.jsonl"
        write_jsonl(case_input, events[event_id])
        cmd = [
            str(root / "build" / "cascade_geant4_local_box"),
            str(case_input),
            str(case_dir),
            f"{args.box_size_cm:.17g}",
            f"{args.density_g_cm3:.17g}",
            args.physics_list,
            args.material,
            "geant4",
            "--geant4-safety-mode",
            args.geant4_safety_mode,
            "--uhe-transport-policy",
            args.uhe_transport_policy,
            "--geant4-hadron-max-kinetic-gev",
            f"{args.geant4_hadron_max_kinetic_gev:.17g}",
            "--geant4-lepton-max-kinetic-gev",
            f"{args.geant4_lepton_max_kinetic_gev:.17g}",
            "--geant4-photon-max-kinetic-gev",
            f"{args.geant4_photon_max_kinetic_gev:.17g}",
            "--geant4-one-particle-per-run",
            "--energy-convention",
            args.energy_convention,
        ]
        if args.debug_single_particle:
            cmd.append("--debug-single-particle")
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (case_dir / "stdout_stderr.txt").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            combined_rows.append(single_particle_fallback(event_id, events[event_id], case_dir))
        else:
            combined_rows.extend(read_budget_csv(case_dir / "geant4_energy_budget.csv"))
            escaped_all.extend(read_jsonl(case_dir / "geant4_escaped_particles.jsonl"))
            unsupported_path = case_dir / "geant4_unsupported_uhe_particles.jsonl"
            if unsupported_path.exists():
                unsupported_uhe_all.extend(read_jsonl(unsupported_path))

    fieldnames = [
        "event_id",
        "input_energy_gev",
        "deposited_energy_gev",
        "escaped_energy_gev",
        "invisible_energy_gev",
        "untracked_energy_gev",
        "unsupported_uhe_energy_gev",
        "escaped_unsupported_uhe_energy_gev",
        "n_unsupported_uhe_particles",
        "accounted_energy_gev",
        "closure_error_gev",
        "escaped_particle_count",
        "energy_convention",
        "uhe_transport_policy",
    ]
    with (output_dir / "geant4_energy_budget.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with (output_dir / "geant4_cascade_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(json.dumps({
                "event_id": int(float(row["event_id"])),
                "input_energy_gev": float(row["input_energy_gev"]),
                "deposited_energy_gev": float(row["deposited_energy_gev"]),
                "escaped_energy_gev": float(row["escaped_energy_gev"]),
                "invisible_energy_gev": float(row["invisible_energy_gev"]),
                "untracked_energy_gev": float(row["untracked_energy_gev"]),
                "unsupported_uhe_energy_gev": float(row.get("unsupported_uhe_energy_gev", 0.0) or 0.0),
                "escaped_unsupported_uhe_energy_gev": float(row.get("escaped_unsupported_uhe_energy_gev", 0.0) or 0.0),
                "n_unsupported_uhe_particles": int(float(row.get("n_unsupported_uhe_particles", 0.0) or 0.0)),
                "escaped_particle_count": int(float(row["escaped_particle_count"])),
                "energy_convention": args.energy_convention,
                "uhe_transport_policy": args.uhe_transport_policy,
                "backend": "Geant4LocalBoxBackendEventProcess",
            }, sort_keys=True) + "\n")
    write_jsonl(output_dir / "geant4_escaped_particles.jsonl", escaped_all)
    write_jsonl(output_dir / "geant4_unsupported_uhe_particles.jsonl", unsupported_uhe_all)
    write_event_isolation_report(output_dir, particles, len(events), args, fallback_events)


def write_event_isolation_report(output_dir: Path, particles: list[dict], n_events: int, args: argparse.Namespace, fallback_events: list[int]) -> None:
    lines = [
        "# GEANT4 Safety Filter Report",
        "",
        "- safety_mode: `" + args.geant4_safety_mode + "`",
        "- energy_convention: `" + args.energy_convention + "`",
        "- geant4_one_particle_per_run: `true`",
        "- uhe_transport_policy: `" + args.uhe_transport_policy + "`",
        "- geant4_hadron_max_kinetic_gev: `" + f"{args.geant4_hadron_max_kinetic_gev:.12g}" + "`",
        "- geant4_lepton_max_kinetic_gev: `" + f"{args.geant4_lepton_max_kinetic_gev:.12g}" + "`",
        "- geant4_photon_max_kinetic_gev: `" + f"{args.geant4_photon_max_kinetic_gev:.12g}" + "`",
        "- isolation: `one_event_per_process_with_one_particle_per_run`",
        "- total_particles: `" + str(len(particles)) + "`",
        "- events: `" + str(n_events) + "`",
        "- fallback_events_particle_per_process: `" + ",".join(str(event_id) for event_id in fallback_events) + "`",
        "",
        "Each event was processed in an isolated subprocess; each transported secondary uses a separate RunManager inside the app.",
        "Events listed in fallback_events_particle_per_process crashed in event-process mode and were rerun particle-by-particle.",
    ]
    (output_dir / "geant4_safety_filter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_isolated_safety_report(output_dir: Path, particles: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# GEANT4 Safety Filter Report",
        "",
        "- safety_mode: `" + args.geant4_safety_mode + "`",
        "- energy_convention: `" + args.energy_convention + "`",
        "- geant4_one_particle_per_run: `true`",
        "- uhe_transport_policy: `" + args.uhe_transport_policy + "`",
        "- geant4_hadron_max_kinetic_gev: `" + f"{args.geant4_hadron_max_kinetic_gev:.12g}" + "`",
        "- geant4_lepton_max_kinetic_gev: `" + f"{args.geant4_lepton_max_kinetic_gev:.12g}" + "`",
        "- geant4_photon_max_kinetic_gev: `" + f"{args.geant4_photon_max_kinetic_gev:.12g}" + "`",
        "- isolation: `one_particle_per_process`",
        "- total_particles: `" + str(len(particles)) + "`",
        "",
        "Each secondary was transported in an isolated subprocess/RunManager. This is the validated stability route for PYTHIA-rich secondary lists.",
    ]
    (output_dir / "geant4_safety_filter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_npz(output_dir: Path, rows: list[dict[str, float]]) -> Path:
    path = output_dir / "geant4_deposition_maps.npz"
    np.savez(
        path,
        event_id=np.asarray([row["event_id"] for row in rows], dtype=np.uint64),
        input_energy_gev=np.asarray([row["input_energy_gev"] for row in rows], dtype=np.float64),
        deposited_energy_gev=np.asarray([row["deposited_energy_gev"] for row in rows], dtype=np.float64),
        escaped_energy_gev=np.asarray([row["escaped_energy_gev"] for row in rows], dtype=np.float64),
        invisible_energy_gev=np.asarray([row["invisible_energy_gev"] for row in rows], dtype=np.float64),
        untracked_energy_gev=np.asarray([row["untracked_energy_gev"] for row in rows], dtype=np.float64),
        unsupported_uhe_energy_gev=np.asarray([row.get("unsupported_uhe_energy_gev", 0.0) for row in rows], dtype=np.float64),
        escaped_unsupported_uhe_energy_gev=np.asarray([row.get("escaped_unsupported_uhe_energy_gev", 0.0) for row in rows], dtype=np.float64),
    )
    return path


def make_plots(output_dir: Path, rows: list[dict[str, float]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    event_id = np.asarray([row["event_id"] for row in rows], dtype=np.float64)
    input_energy = np.asarray([row["input_energy_gev"] for row in rows], dtype=np.float64)
    deposited = np.asarray([row["deposited_energy_gev"] for row in rows], dtype=np.float64)
    escaped = np.asarray([row["escaped_energy_gev"] for row in rows], dtype=np.float64)
    invisible = np.asarray([row["invisible_energy_gev"] for row in rows], dtype=np.float64)
    untracked = np.asarray([row["untracked_energy_gev"] for row in rows], dtype=np.float64)
    escaped_unsupported = np.asarray([row.get("escaped_unsupported_uhe_energy_gev", 0.0) for row in rows], dtype=np.float64)
    closure = deposited + escaped + invisible + untracked + escaped_unsupported - input_energy
    denom = np.maximum(input_energy, 1.0e-300)

    def finite_hist_range(values: np.ndarray) -> tuple[float, float] | None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo <= max(abs(lo), abs(hi), 1.0) * 1.0e-9:
            pad = max(abs(lo), abs(hi), 1.0) * 1.0e-6
            return (lo - pad, hi + pad)
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(deposited / denom, bins=20, range=finite_hist_range(deposited / denom), color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("deposited / input")
    ax.set_ylabel("events")
    ax.set_title("GEANT4 local-box deposited fraction")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_deposited_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(escaped / denom, bins=20, range=finite_hist_range(escaped / denom), color="#577590", edgecolor="white")
    ax.set_xlabel("escaped / input")
    ax.set_ylabel("events")
    ax.set_title("GEANT4 local-box escaped fraction")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_escaped_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(event_id, closure / denom, marker="o", linewidth=1.1)
    ax.set_xlabel("event id")
    ax.set_ylabel("(deposited + escaped + invisible + untracked + skipped UHE - input) / input")
    ax.set_title("GEANT4 local-box energy closure")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_energy_closure.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(event_id, deposited, color="#e76f51")
    ax.set_xlabel("event id")
    ax.set_ylabel("deposited energy [GeV]")
    ax.set_title("GEANT4 local-box deposition by event")
    fig.tight_layout()
    fig.savefig(plots_dir / "geant4_deposition_by_event.png", dpi=180)
    plt.close(fig)


def print_summary(output_dir: Path, rows: list[dict[str, float]], secondaries: Path, npz_path: Path) -> None:
    input_energy = sum(row["input_energy_gev"] for row in rows)
    deposited = sum(row["deposited_energy_gev"] for row in rows)
    escaped = sum(row["escaped_energy_gev"] for row in rows)
    invisible = sum(row["invisible_energy_gev"] for row in rows)
    untracked = sum(row["untracked_energy_gev"] for row in rows)
    unsupported = sum(row.get("unsupported_uhe_energy_gev", 0.0) for row in rows)
    escaped_unsupported = sum(row.get("escaped_unsupported_uhe_energy_gev", 0.0) for row in rows)
    closure = deposited + escaped + invisible + untracked + escaped_unsupported - input_energy
    escaped_particles = read_jsonl(output_dir / "geant4_escaped_particles.jsonl")
    print("GEANT4 local-box demo summary")
    print(f"events={len(rows)}")
    print(f"input_secondaries={secondaries}")
    print(f"escaped_particles={len(escaped_particles)}")
    print(f"input_energy_gev={input_energy:.12e}")
    print(f"deposited_energy_gev={deposited:.12e}")
    print(f"escaped_energy_gev={escaped:.12e}")
    print(f"invisible_energy_gev={invisible:.12e}")
    print(f"untracked_energy_gev={untracked:.12e}")
    print(f"unsupported_uhe_energy_gev={unsupported:.12e}")
    print(f"escaped_unsupported_uhe_energy_gev={escaped_unsupported:.12e}")
    print(f"closure_error_gev={closure:.12e}")
    print(f"deposition_maps={npz_path}")
    print(f"plots={output_dir / 'plots'}")
    print("NOTE: GEANT4 local homogeneous box only; not global collapsar transport or plasma physics.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--n-events", type=int, default=8)
    parser.add_argument("--energy-gev", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=86420)
    parser.add_argument("--box-size-cm", type=float, default=100.0)
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--physics-list", choices=["FTFP_BERT", "QGSP_BERT"], default="FTFP_BERT")
    parser.add_argument("--material", choices=["hydrogen", "water"], default="hydrogen")
    parser.add_argument("--transport-mode", choices=["proxy", "geant4"], default="proxy")
    parser.add_argument("--geant4-safety-mode", choices=["off", "strict"], default="off")
    parser.add_argument("--uhe-transport-policy", choices=["error", "skip_to_escaped", "split_energy_proxy"], default="error")
    parser.add_argument("--geant4-hadron-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-lepton-max-kinetic-gev", type=float, default=1.0e9)
    parser.add_argument("--geant4-photon-max-kinetic-gev", type=float, default=1.0e9)
    parser.add_argument("--geant4-one-particle-per-run", action="store_true")
    parser.add_argument(
        "--geant4-one-particle-per-process",
        action="store_true",
        help="Diagnostic slow path: launch a fresh executable process for each secondary.",
    )
    parser.add_argument("--debug-single-particle", action="store_true")
    parser.add_argument("--energy-convention", choices=["total", "kinetic"], default="total")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not geant4_available():
        print("GEANT4 local-box demo skipped: geant4-config was not found in PATH.")
        print("This is not fatal; HADROS-CASCADE remains usable without GEANT4.")
        return 0

    secondaries = ensure_secondaries(root, output_dir, args)
    build_geant4_app(root)
    if args.transport_mode == "geant4" and args.geant4_one_particle_per_process:
        run_geant4_one_particle_processes(root, output_dir, secondaries, args)
    elif args.transport_mode == "geant4" and args.geant4_one_particle_per_run:
        run_geant4_event_processes(root, output_dir, secondaries, args)
    else:
        run_geant4_app(root, output_dir, secondaries, args)
    rows = read_csv(output_dir / "geant4_energy_budget.csv")
    npz_path = write_npz(output_dir, rows)
    make_plots(output_dir, rows)
    print_summary(output_dir, rows, secondaries, npz_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
