#!/usr/bin/env python3
"""Classify escaping packets for future effective null-geodesic propagation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


PDG_MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    -11: 0.00051099895,
    13: 0.1056583755,
    -13: 0.1056583755,
    111: 0.1349768,
    211: 0.13957039,
    -211: 0.13957039,
    321: 0.493677,
    -321: 0.493677,
    130: 0.497611,
    310: 0.497611,
    2212: 0.93827208816,
    -2212: 0.93827208816,
    2112: 0.93956542052,
    -2112: 0.93956542052,
    12: 0.0,
    -12: 0.0,
    14: 0.0,
    -14: 0.0,
    16: 0.0,
    -16: 0.0,
}

PDG_LABELS = {
    22: "gamma",
    11: "electron",
    -11: "positron",
    13: "muon_minus",
    -13: "muon_plus",
    111: "pi0",
    211: "pi_plus",
    -211: "pi_minus",
    321: "kaon_plus",
    -321: "kaon_minus",
    130: "kaon0L",
    310: "kaon0S",
    2212: "proton",
    -2212: "anti_proton",
    2112: "neutron",
    -2112: "anti_neutron",
    12: "nu_e",
    -12: "anti_nu_e",
    14: "nu_mu",
    -14: "anti_nu_mu",
    16: "nu_tau",
    -16: "anti_nu_tau",
}

NEUTRINOS = {12, -12, 14, -14, 16, -16}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify_packet(packet: dict[str, Any]) -> dict[str, Any]:
    pdg = int(packet.get("pdg_id", 0))
    energy = float(packet.get("energy_gev", 0.0))
    weighted_energy = float(packet.get("weighted_energy_gev", energy * float(packet.get("weight", 1.0))))
    label = str(packet.get("particle_label", PDG_LABELS.get(pdg, f"pdg{pdg}")))
    mass = PDG_MASS_GEV.get(pdg)
    gamma = math.nan
    beta = math.nan
    error = math.nan

    if pdg in NEUTRINOS:
        classification = "INVISIBLE_SKIP"
        mass = 0.0 if mass is None else mass
    elif mass is None:
        classification = "UNKNOWN_MASS"
    elif mass == 0.0:
        classification = "MASSLESS_NULL"
        gamma = math.inf
        beta = 1.0
        error = 0.0
    elif energy <= 0.0:
        classification = "MASSIVE_PROPAGATION_REQUIRED"
    else:
        gamma = energy / mass
        beta = math.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0)) if gamma >= 1.0 else 0.0
        error = 0.5 * (mass / energy) ** 2
        if gamma >= 100.0:
            classification = "ULTRARELATIVISTIC_NULL_OK"
        elif gamma >= 10.0:
            classification = "MARGINAL_ULTRARELATIVISTIC"
        else:
            classification = "MASSIVE_PROPAGATION_REQUIRED"

    return {
        "event_id": int(packet.get("event_id", 0)),
        "pdg_id": pdg,
        "particle_label": label,
        "energy_gev": energy,
        "mass_gev": "" if mass is None else mass,
        "gamma": gamma,
        "beta": beta,
        "null_geodesic_error_estimate": error,
        "weighted_energy_gev": weighted_energy,
        "classification": classification,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id",
        "pdg_id",
        "particle_label",
        "energy_gev",
        "mass_gev",
        "gamma",
        "beta",
        "null_geodesic_error_estimate",
        "weighted_energy_gev",
        "classification",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy_by_class: dict[str, float] = defaultdict(float)
    energy_by_pdg: dict[int, float] = defaultdict(float)
    for row in rows:
        energy_by_class[str(row["classification"])] += float(row["weighted_energy_gev"])
        energy_by_pdg[int(row["pdg_id"])] += float(row["weighted_energy_gev"])
    total = sum(energy_by_class.values())
    null_energy = energy_by_class["MASSLESS_NULL"] + energy_by_class["ULTRARELATIVISTIC_NULL_OK"]
    marginal = energy_by_class["MARGINAL_ULTRARELATIVISTIC"]
    massive = energy_by_class["MASSIVE_PROPAGATION_REQUIRED"]
    return {
        "total_weighted_energy_gev": total,
        "energy_by_class": dict(sorted(energy_by_class.items())),
        "effective_null_fraction": null_energy / max(total, 1.0e-300),
        "marginal_fraction": marginal / max(total, 1.0e-300),
        "massive_required_fraction": massive / max(total, 1.0e-300),
        "ranking_by_pdg": [
            {
                "pdg_id": pdg,
                "particle_label": PDG_LABELS.get(pdg, f"pdg{pdg}"),
                "weighted_energy_gev": energy,
                "fraction": energy / max(total, 1.0e-300),
            }
            for pdg, energy in sorted(energy_by_pdg.items(), key=lambda item: item[1], reverse=True)
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Escaping Packet Classification",
        "",
        "Diagnostic classification for future effective null-geodesic packet propagation.",
        "No propagation is performed in this phase.",
        "",
        f"- total weighted packet energy [GeV]: `{summary['total_weighted_energy_gev']:.12g}`",
        f"- effective null-geodesic fraction: `{summary['effective_null_fraction']:.12g}`",
        f"- marginal ultrarelativistic fraction: `{summary['marginal_fraction']:.12g}`",
        f"- massive-propagation-required fraction: `{summary['massive_required_fraction']:.12g}`",
        "",
        "## Energy By Class",
        "",
        "| Class | Weighted energy [GeV] | Fraction |",
        "|---|---:|---:|",
    ]
    total = float(summary["total_weighted_energy_gev"])
    for cls, energy in summary["energy_by_class"].items():
        lines.append(f"| {cls} | {energy:.12g} | {energy / max(total, 1.0e-300):.12g} |")
    lines.extend(["", "## Ranking By PDG", "", "| PDG | Label | Weighted energy [GeV] | Fraction |", "|---:|---|---:|---:|"])
    for row in summary["ranking_by_pdg"]:
        lines.append(
            f"| {row['pdg_id']} | {row['particle_label']} | "
            f"{row['weighted_energy_gev']:.12g} | {row['fraction']:.12g} |"
        )
    lines.extend([
        "",
        "Criteria: photons are massless-null, neutrinos are skipped as invisible by default,",
        "`gamma >= 100` is null-ok, `10 <= gamma < 100` is marginal, and `gamma < 10`",
        "requires future massive-packet propagation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = list(summary["energy_by_class"].keys())
    energies = [summary["energy_by_class"][label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if sum(energies) > 0.0:
        ax.pie(energies, labels=labels, autopct="%1.1f%%")
    ax.set_title("escaping packet class energy fraction")
    fig.tight_layout()
    fig.savefig(plots / "escaping_packet_class_energy_fraction.png", dpi=180)
    plt.close(fig)

    by_pdg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pdg[int(row["pdg_id"])].append(row)
    pdgs = sorted(by_pdg, key=lambda pdg: sum(float(r["weighted_energy_gev"]) for r in by_pdg[pdg]), reverse=True)
    labels_pdg = [PDG_LABELS.get(pdg, f"pdg{pdg}") for pdg in pdgs]
    gamma_vals = []
    error_vals = []
    for pdg in pdgs:
        group = by_pdg[pdg]
        weights = [float(r["weighted_energy_gev"]) for r in group]
        gammas = [float(r["gamma"]) for r in group if math.isfinite(float(r["gamma"]))]
        errors = [float(r["null_geodesic_error_estimate"]) for r in group if math.isfinite(float(r["null_geodesic_error_estimate"]))]
        gamma_vals.append(sum(g * w for g, w in zip(gammas, weights[:len(gammas)])) / max(sum(weights[:len(gammas)]), 1.0e-300) if gammas else math.nan)
        error_vals.append(sum(e * w for e, w in zip(errors, weights[:len(errors)])) / max(sum(weights[:len(errors)]), 1.0e-300) if errors else math.nan)
    for vals, ylabel, filename, logy in [
        (gamma_vals, "energy-weighted gamma", "escaping_packet_gamma_by_pdg.png", True),
        (error_vals, "null error estimate", "escaping_packet_null_error_by_pdg.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.bar(labels_pdg, [0.0 if not math.isfinite(v) else v for v in vals])
        if logy:
            ax.set_yscale("symlog", linthresh=1.0e-12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)


def classify_file(args: argparse.Namespace) -> dict[str, Any]:
    packets = read_jsonl(args.input)
    rows = [classify_packet(packet) for packet in packets]
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "escaping_packet_classification.csv", rows)
    write_markdown(args.output_dir / "escaping_packet_classification.md", summary)
    make_plots(args.output_dir, rows, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = classify_file(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
