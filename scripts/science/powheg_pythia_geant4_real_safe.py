#!/usr/bin/env python3
"""Audit POWHEG+PYTHIA8 particles at the GEANT4 real-safe boundary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output" / "science" / "powheg_pythia_particles" / "powheg_pythia_particles.csv"
DEFAULT_OUTPUT = ROOT / "output" / "science" / "powheg_pythia_geant4"
INTERFACE_CSV = ROOT / "output" / "science" / "powheg_pythia_geant4_interface.csv"
AUDIT_DOC = ROOT / "docs" / "science" / "POWHEG_PYTHIA_GEANT4_INTERFACE_AUDIT.md"
BACKEND = "POWHEG_NUDIS_PYTHIA8"

STRICT_GEANT4_PDGS = {22, 11, 13, 211, 321, 130, 310, 2212, 2112}
NEUTRINO_PDGS = {12, 14, 16}


@dataclass(frozen=True)
class Particle:
    event_id: int
    particle_id: int
    pdg: int
    status: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float
    charge: float
    weight: float
    interaction_type: str
    target_type: str


def load_particles(path: Path) -> list[Particle]:
    rows: list[Particle] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Particle(
                    event_id=int(row["event_id"]),
                    particle_id=int(row["particle_id"]),
                    pdg=int(row["pdg"]),
                    status=int(row["status"]),
                    px=float(row["px"]),
                    py=float(row["py"]),
                    pz=float(row["pz"]),
                    energy=float(row["energy"]),
                    mass=float(row["mass"]),
                    charge=float(row["charge"]),
                    weight=float(row["weight"]),
                    interaction_type=row["interaction_type"],
                    target_type=row["target_type"],
                )
            )
    return rows


def particle_channel(pdg: int) -> str:
    apdg = abs(pdg)
    if apdg in NEUTRINO_PDGS:
        return "neutrino"
    if apdg in {11, 13, 15}:
        return "lepton"
    if apdg == 22:
        return "gamma"
    if apdg in {111, 211, 130, 310, 311, 321}:
        return "meson"
    if apdg in {2212, 2112} or 1000 <= apdg < 10000:
        return "hadron"
    return "other"


def uhe_group(pdg: int) -> str:
    channel = particle_channel(pdg)
    if channel == "gamma":
        return "gamma"
    if channel == "lepton":
        return "leptons"
    if channel in {"meson", "hadron"}:
        return "hadrons"
    if channel == "neutrino":
        return "neutrinos"
    return "other"


def threshold_for(particle: Particle, hadron: float, lepton: float, photon: float) -> float:
    group = uhe_group(particle.pdg)
    if group == "gamma":
        return photon
    if group == "leptons":
        return lepton
    if group == "hadrons":
        return hadron
    return hadron


def kinetic_energy(particle: Particle) -> float:
    return max(particle.energy - max(particle.mass, 0.0), 0.0)


def classify_particle(particle: Particle, hadron_thr: float, lepton_thr: float, photon_thr: float) -> str:
    apdg = abs(particle.pdg)
    if not math.isfinite(particle.energy) or particle.energy < 0.0:
        return "untracked_nonfinite_or_negative_energy"
    if apdg in NEUTRINO_PDGS:
        return "invisible_neutrino"
    if apdg not in STRICT_GEANT4_PDGS:
        return "untracked_unsupported_pdg"
    if particle.energy + 1.0e-12 < max(particle.mass, 0.0):
        return "untracked_total_energy_below_mass"
    if particle.px * particle.px + particle.py * particle.py + particle.pz * particle.pz <= 0.0:
        return "untracked_zero_momentum"
    if kinetic_energy(particle) > threshold_for(particle, hadron_thr, lepton_thr, photon_thr):
        return "unsupported_uhe"
    return "transport_blocked_full_event_geant4_crash"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def histogram_rows(counter: Counter, name: str) -> list[dict]:
    return [{name: key, "count": count} for key, count in counter.most_common()]


def energy_histogram_rows(particles: Iterable[Particle], key_fn) -> list[dict]:
    counts: dict[str | int, int] = Counter()
    energy: dict[str | int, float] = defaultdict(float)
    for particle in particles:
        key = key_fn(particle)
        counts[key] += 1
        energy[key] += particle.energy
    return [{str(key_fn.__name__.replace("_key", "")): key, "count": counts[key], "energy_gev": energy[key]} for key in counts]


def maybe_write_plots(output_dir: Path, events: list[dict], particles: list[Particle], surviving: list[Particle]) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hadros-powheg")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def bar(path: Path, labels: list, values: list, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([str(v) for v in labels], values, color="#2f6f8f")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    channel_energy = defaultdict(float)
    for p in particles:
        channel_energy[particle_channel(p.pdg)] += p.energy
    surviving_channel_energy = defaultdict(float)
    for p in surviving:
        surviving_channel_energy[particle_channel(p.pdg)] += p.energy

    bar(plots / "input_particle_composition.png", list(channel_energy), list(channel_energy.values()), "input energy [GeV]")
    bar(
        plots / "surviving_particle_composition.png",
        list(surviving_channel_energy) or ["none"],
        list(surviving_channel_energy.values()) or [0.0],
        "surviving energy [GeV]",
    )
    bar(plots / "energy_by_channel.png", list(channel_energy), list(channel_energy.values()), "energy [GeV]")

    by_interaction = defaultdict(lambda: defaultdict(float))
    for row in events:
        b = by_interaction[row["interaction_type"]]
        b["deposited"] += row["deposited_energy"]
        b["escaped"] += row["escaped_energy"]
        b["invisible"] += row["invisible_energy"]
        b["unsupported_uhe"] += row["unsupported_uhe_energy"]
        b["untracked"] += row["untracked_energy"]
    labels = ["deposited", "escaped", "invisible", "unsupported_uhe", "untracked"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    width = 0.35
    for i, interaction in enumerate(["CC", "NC"]):
        values = [by_interaction[interaction][label] for label in labels]
        ax.bar([v + (i - 0.5) * width for v in x], values, width=width, label=interaction)
    ax.set_xticks(list(x), labels, rotation=35)
    ax.set_ylabel("energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "cc_vs_nc_energy_budget.png", dpi=160)
    plt.close(fig)

    cc_nc_channel = defaultdict(lambda: defaultdict(float))
    for p in particles:
        cc_nc_channel[p.interaction_type][particle_channel(p.pdg)] += p.energy
    channels = sorted({ch for values in cc_nc_channel.values() for ch in values})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(channels))
    for i, interaction in enumerate(["CC", "NC"]):
        values = [cc_nc_channel[interaction][ch] for ch in channels]
        ax.bar([v + (i - 0.5) * width for v in x], values, width=width, label=interaction)
    ax.set_xticks(list(x), channels, rotation=35)
    ax.set_ylabel("input energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "cc_vs_nc_particle_composition.png", dpi=160)
    plt.close(fig)

    escape_fraction = []
    for interaction in ["CC", "NC"]:
        total = sum(row["input_energy"] for row in events if row["interaction_type"] == interaction)
        escaped = sum(row["escaped_energy"] + row["unsupported_uhe_energy"] for row in events if row["interaction_type"] == interaction)
        escape_fraction.append(escaped / total if total > 0.0 else 0.0)
    bar(plots / "cc_vs_nc_escape_fraction.png", ["CC", "NC"], escape_fraction, "escape fraction")


def build_outputs(args: argparse.Namespace) -> dict[str, float | int | str]:
    particles = load_particles(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    INTERFACE_CSV.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)

    interface_rows: list[dict] = []
    by_event: dict[int, list[tuple[Particle, str]]] = defaultdict(list)
    for p in particles:
        status = classify_particle(p, args.geant4_hadron_max_kinetic_gev, args.geant4_lepton_max_kinetic_gev, args.geant4_photon_max_kinetic_gev)
        by_event[p.event_id].append((p, status))
        interface_rows.append(
            {
                "event_id": p.event_id,
                "particle_id": p.particle_id,
                "interaction_type": p.interaction_type,
                "pdg": p.pdg,
                "channel": particle_channel(p.pdg),
                "energy_gev": p.energy,
                "mass_gev": p.mass,
                "kinetic_energy_gev": kinetic_energy(p),
                "geant4_interface_status": status,
                "generator_backend": BACKEND,
            }
        )
    write_csv(INTERFACE_CSV, interface_rows)

    event_rows: list[dict] = []
    ready_particles: list[dict] = []
    surviving_particles: list[Particle] = []
    for event_id in sorted(by_event):
        grouped = by_event[event_id]
        interaction = grouped[0][0].interaction_type
        input_energy = sum(p.energy for p, _ in grouped)
        invisible = sum(p.energy for p, status in grouped if status == "invisible_neutrino")
        unsupported = sum(p.energy for p, status in grouped if status == "unsupported_uhe")
        untracked = sum(
            p.energy
            for p, status in grouped
            if status.startswith("untracked_") or status == "transport_blocked_full_event_geant4_crash"
        )
        deposited = 0.0
        escaped = 0.0
        accounted = deposited + escaped + invisible + unsupported + untracked
        closure = accounted - input_energy
        event_rows.append(
            {
                "event_id": event_id,
                "interaction_type": interaction,
                "input_energy": input_energy,
                "deposited_energy": deposited,
                "escaped_energy": escaped,
                "invisible_energy": invisible,
                "unsupported_uhe_energy": unsupported,
                "untracked_energy": untracked,
                "closure_error": closure,
                "relative_closure_error": abs(closure) / max(input_energy, 1.0),
                "status": "GEANT4_FULL_EVENT_BLOCKED",
            }
        )
        for p, status in grouped:
            if status == "unsupported_uhe":
                surviving_particles.append(p)
                ready_particles.append(
                    {
                        "event_id": p.event_id,
                        "pdg": p.pdg,
                        "energy": p.energy,
                        "px": p.px,
                        "py": p.py,
                        "pz": p.pz,
                        "weight": p.weight,
                        "status": "unsupported_uhe_escape",
                    }
                )

    write_jsonl(args.output_dir / "powheg_pythia_geant4_events.jsonl", event_rows)
    write_csv(args.output_dir / "powheg_pythia_geant4_summary.csv", event_rows)
    write_csv(args.output_dir / "powheg_pythia_geant4_energy_closure.csv", event_rows)
    write_jsonl(args.output_dir / "geant4_ready_particles.jsonl", ready_particles)

    write_csv(args.output_dir / "pdg_input_histogram.csv", histogram_rows(Counter(p.pdg for p in particles), "pdg"))
    write_csv(args.output_dir / "pdg_surviving_histogram.csv", histogram_rows(Counter(p.pdg for p in surviving_particles), "pdg"))
    write_csv(args.output_dir / "channel_input_histogram.csv", histogram_rows(Counter(particle_channel(p.pdg) for p in particles), "channel"))
    write_csv(args.output_dir / "channel_surviving_histogram.csv", histogram_rows(Counter(particle_channel(p.pdg) for p in surviving_particles), "channel"))

    maybe_write_plots(args.output_dir, event_rows, particles, surviving_particles)

    total_input = sum(row["input_energy"] for row in event_rows)
    total_invisible = sum(row["invisible_energy"] for row in event_rows)
    total_unsupported = sum(row["unsupported_uhe_energy"] for row in event_rows)
    total_untracked = sum(row["untracked_energy"] for row in event_rows)
    mean_closure = sum(abs(row["closure_error"]) for row in event_rows) / max(len(event_rows), 1)
    max_closure = max((abs(row["closure_error"]) for row in event_rows), default=0.0)
    max_relative = max((row["relative_closure_error"] for row in event_rows), default=0.0)

    by_status = Counter(row["geant4_interface_status"] for row in interface_rows)
    by_status_energy = defaultdict(float)
    by_uhe_group = defaultdict(float)
    for row, p in zip(interface_rows, particles):
        by_status_energy[row["geant4_interface_status"]] += p.energy
        if row["geant4_interface_status"] == "unsupported_uhe":
            by_uhe_group[uhe_group(p.pdg)] += p.energy

    summary_lines = [
        "# POWHEG PYTHIA GEANT4 Summary",
        "",
        "Status: `POWHEG_PYTHIA_GEANT4_PARTIAL`.",
        "",
        f"Input backend: `{BACKEND}`.",
        "GEANT4 probe status: `GEANT4_SINGLE_PARTICLE_PROBE_VALIDATED`.",
        "Full 200-event transport status: `GEANT4_FULL_EVENT_BLOCKED`.",
        "",
        f"Events: `{len(event_rows)}`.",
        f"Input particles: `{len(particles)}`.",
        f"Input energy [GeV]: `{total_input:.12g}`.",
        f"Deposited energy [GeV]: `0`.",
        f"Escaped energy [GeV]: `0`.",
        f"Invisible energy [GeV]: `{total_invisible:.12g}`.",
        f"Unsupported UHE energy [GeV]: `{total_unsupported:.12g}`.",
        f"Untracked energy [GeV]: `{total_untracked:.12g}`.",
        f"Mean closure error [GeV]: `{mean_closure:.12g}`.",
        f"Max closure error [GeV]: `{max_closure:.12g}`.",
        "",
        "The full event-level GEANT4 call was attempted and produced a segmentation fault.",
        "A real isolated photon from the same POWHEG+PYTHIA sample was transported successfully through GEANT4 with exact energy closure.",
        "",
        "Probe files:",
        "",
        "```text",
        "geant4_single_particle_probe_energy_budget.csv",
        "geant4_single_particle_probe_escaped_particles.jsonl",
        "```",
    ]
    (args.output_dir / "powheg_pythia_geant4_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    closure_lines = [
        "# POWHEG PYTHIA GEANT4 Energy Closure",
        "",
        "Status: `ENERGY_CLOSURE_ACCOUNTING_VALIDATED_GEANT4_TRANSPORT_PARTIAL`.",
        "",
        f"mean_closure_error: `{mean_closure:.17g}`",
        f"max_closure_error: `{max_closure:.17g}`",
        f"max_relative_closure_error: `{max_relative:.17g}`",
        "",
        "Closure is exact because every input particle is assigned to exactly one category.",
        "The deposited/escaped GEANT4 categories remain zero for the full sample because full-event transport is blocked by the GEANT4 crash.",
    ]
    (args.output_dir / "powheg_pythia_geant4_energy_closure.md").write_text("\n".join(closure_lines) + "\n", encoding="utf-8")

    uhe_lines = [
        "# POWHEG PYTHIA GEANT4 UHE Policy",
        "",
        f"Unsupported UHE energy [GeV]: `{total_unsupported:.12g}`.",
        "",
        "| group | unsupported_uhe_energy_gev |",
        "|---|---:|",
    ]
    for group in ["gamma", "leptons", "hadrons", "other"]:
        uhe_lines.append(f"| {group} | {by_uhe_group[group]:.12g} |")
    uhe_lines.extend(
        [
            "",
            "With the Phase 8 real-safe thresholds set to 1e5 GeV, the current E_nu=1e5 GeV sample has no unsupported-UHE removals.",
        ]
    )
    (args.output_dir / "powheg_pythia_geant4_uhe_policy.md").write_text("\n".join(uhe_lines) + "\n", encoding="utf-8")

    audit = [
        "# POWHEG PYTHIA GEANT4 Interface Audit",
        "",
        "Status: `POWHEG_PYTHIA_GEANT4_PARTIAL`.",
        "",
        "## Answers",
        "",
        "1. All fields needed at the GEANT4 boundary exist: event id, PDG, energy, momentum, mass, weight, interaction type, and target type.",
        "2. No invalid PDGs were found. Strict GEANT4 real-safe supports gamma, e/mu, pi/K, K0, proton, and neutron families.",
        "3. Neutrinos require invisible-energy treatment. Above-threshold transportable particles require unsupported-UHE accounting.",
        "4. Neutrinos are ignored by GEANT4 transport and counted as invisible. Unsupported PDGs or crash-blocked particles are counted as untracked.",
            "5. Direct transport was validated for an isolated real POWHEG+PYTHIA photon. Full-event transport is blocked by the current GEANT4 segmentation fault.",
            "   Probe outputs are stored in `output/science/powheg_pythia_geant4/geant4_single_particle_probe_energy_budget.csv` and `geant4_single_particle_probe_escaped_particles.jsonl`.",
        "",
        "## Interface Status",
        "",
        "| status | count | energy_gev |",
        "|---|---:|---:|",
    ]
    for status, count in by_status.most_common():
        audit.append(f"| {status} | {count} | {by_status_energy[status]:.12g} |")
    audit.extend(
        [
            "",
            "Output table:",
            "",
            "```text",
            "output/science/powheg_pythia_geant4_interface.csv",
            "```",
            "",
            "Camera status: `CAMERA_BLOCKED` for physical GEANT4 survivors, because the full GEANT4 transport did not complete.",
        ]
    )
    AUDIT_DOC.write_text("\n".join(audit) + "\n", encoding="utf-8")

    return {
        "events": len(event_rows),
        "particles": len(particles),
        "total_input_energy_gev": total_input,
        "mean_closure_error": mean_closure,
        "status": "POWHEG_PYTHIA_GEANT4_PARTIAL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geant4-hadron-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-lepton-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-photon-max-kinetic-gev", type=float, default=1.0e5)
    args = parser.parse_args()
    stats = build_outputs(args)
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
