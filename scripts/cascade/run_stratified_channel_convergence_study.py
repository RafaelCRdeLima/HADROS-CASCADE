#!/usr/bin/env python3
"""Run Phase 8.5 stratified packet/channel convergence sampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_energy_fraction_convergence_study as ef


MODES = ["energy_desc", "stratified_pdg", "stratified_channel", "hybrid_energy_channel"]
NEUTRINOS = {12, -12, 14, -14, 16, -16}
GAMMA = {22}
ELECTRONS = {11, -11}
MUONS = {13, -13}
PIONS = {211, -211}
HADRONS = {111, 130, 310, 321, -321, 2112, -2112, 2212, -2212}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def energy(row: dict[str, Any]) -> float:
    return ef.finite(row.get("energy_gev"))


def pdg_id(row: dict[str, Any]) -> int:
    return int(row.get("pdg_id", row.get("pdg", 0)) or 0)


def channel_keys(row: dict[str, Any], uhe_threshold: float) -> set[str]:
    pdg = pdg_id(row)
    keys: set[str] = set()
    if pdg in NEUTRINOS:
        keys.add("neutrino_invisible")
    if pdg in GAMMA:
        keys.update({"gamma", "electromagnetic"})
    if pdg in ELECTRONS:
        keys.update({"electron_positron", "electromagnetic"})
    if pdg in MUONS:
        keys.add("muon")
    if pdg in PIONS:
        keys.update({"pion_charged", "hadronic"})
    if pdg in HADRONS or (abs(pdg) > 100 and pdg not in PIONS):
        keys.add("hadronic")
    if not keys:
        keys.add("other")
    if energy(row) > uhe_threshold:
        keys.add("unsupported_uhe")
    return keys


def add_unique(selected: list[dict[str, Any]], seen: set[int], indexed_row: tuple[int, dict[str, Any]]) -> bool:
    idx, row = indexed_row
    if idx in seen:
        return False
    selected.append(row)
    seen.add(idx)
    return True


def selected_energy(rows: list[dict[str, Any]]) -> float:
    return sum(energy(row) for row in rows)


def select_energy_desc(indexed: list[tuple[int, dict[str, Any]]], target_energy: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in sorted(indexed, key=lambda pair: energy(pair[1]), reverse=True):
        add_unique(selected, seen, item)
        if selected_energy(selected) >= target_energy:
            break
    return selected


def select_round_robin(groups: dict[str, list[tuple[int, dict[str, Any]]]], target_energy: float, min_jobs: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    sorted_groups = {
        key: sorted(items, key=lambda pair: energy(pair[1]), reverse=True)
        for key, items in groups.items()
        if items
    }
    for key in sorted(sorted_groups):
        for item in sorted_groups[key][:max(0, min_jobs)]:
            add_unique(selected, seen, item)
    cursor = {key: max(0, min_jobs) for key in sorted_groups}
    while selected_energy(selected) < target_energy:
        progressed = False
        for key in sorted(sorted_groups, key=lambda k: sum(energy(row) for _, row in sorted_groups[k]), reverse=True):
            items = sorted_groups[key]
            while cursor[key] < len(items) and items[cursor[key]][0] in seen:
                cursor[key] += 1
            if cursor[key] >= len(items):
                continue
            add_unique(selected, seen, items[cursor[key]])
            cursor[key] += 1
            progressed = True
            if selected_energy(selected) >= target_energy:
                break
        if not progressed:
            break
    return selected


def select_stratified_pdg(indexed: list[tuple[int, dict[str, Any]]], target_energy: float, min_jobs: int) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in indexed:
        groups[str(pdg_id(item[1]))].append(item)
    return select_round_robin(groups, target_energy, min_jobs)


def select_stratified_channel(indexed: list[tuple[int, dict[str, Any]]], target_energy: float, min_jobs: int, uhe_threshold: float) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in indexed:
        for key in channel_keys(item[1], uhe_threshold):
            groups[key].append(item)
    return select_round_robin(groups, target_energy, min_jobs)


def select_hybrid(indexed: list[tuple[int, dict[str, Any]]], target_energy: float, min_jobs: int, uhe_threshold: float) -> list[dict[str, Any]]:
    first = select_energy_desc(indexed, 0.5 * target_energy)
    seen_rows = {id(row) for row in first}
    seen_indices = {idx for idx, row in indexed if id(row) in seen_rows}
    selected = list(first)
    strat = select_stratified_channel(indexed, target_energy, min_jobs, uhe_threshold)
    for row in strat:
        for idx, original in indexed:
            if original is row and idx not in seen_indices:
                selected.append(row)
                seen_indices.add(idx)
                break
        if selected_energy(selected) >= target_energy:
            break
    return selected


def select_rows(rows: list[dict[str, Any]], mode: str, frac: float, args: argparse.Namespace) -> list[dict[str, Any]]:
    indexed = list(enumerate(rows))
    total = selected_energy(rows)
    target = frac * total
    if mode == "energy_desc":
        return select_energy_desc(indexed, target)
    if mode == "stratified_pdg":
        return select_stratified_pdg(indexed, target, args.min_jobs_per_pdg)
    if mode == "stratified_channel":
        return select_stratified_channel(indexed, target, args.min_jobs_per_channel, args.uhe_channel_threshold_gev)
    if mode == "hybrid_energy_channel":
        return select_hybrid(indexed, target, args.min_jobs_per_channel, args.uhe_channel_threshold_gev)
    raise ValueError(f"unknown sampling mode: {mode}")


def source_channels(rows: list[dict[str, Any]], uhe_threshold: float) -> set[str]:
    out: set[str] = set()
    for row in rows:
        out.update(channel_keys(row, uhe_threshold))
    return out


def prepare_selected_dir(args: argparse.Namespace, mode: str, frac: float, all_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_rows(all_rows, mode, frac, args)
    write_jsonl(out_dir / "pythia_secondaries.jsonl", selected)
    for name in ["interaction_points.jsonl", "primary_events.jsonl"]:
        src = args.source_run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    src_pdgs = {pdg_id(row) for row in all_rows}
    sel_pdgs = {pdg_id(row) for row in selected}
    src_channels = source_channels(all_rows, args.uhe_channel_threshold_gev)
    sel_channels = source_channels(selected, args.uhe_channel_threshold_gev)
    metadata = {
        "sampling_mode": mode,
        "fraction_target": frac,
        "source_energy_gev": selected_energy(all_rows),
        "selected_energy_gev": selected_energy(selected),
        "selected_energy_fraction": selected_energy(selected) / max(selected_energy(all_rows), 1.0e-300),
        "selected_jobs": len(selected),
        "source_pdgs": len(src_pdgs),
        "selected_pdgs": len(sel_pdgs),
        "pdg_coverage_fraction": len(sel_pdgs) / max(len(src_pdgs), 1),
        "source_channels": sorted(src_channels),
        "selected_channels": sorted(sel_channels),
        "channel_coverage_fraction": len(sel_channels) / max(len(src_channels), 1),
    }
    (out_dir / "sampling_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_products(root: Path, args: argparse.Namespace, out_dir: Path, mode: str, frac: float) -> None:
    required = [
        out_dir / "particle_channel_images_summary.md",
        out_dir / "particle_channel_image_audit.md",
        out_dir / "escaping_packet_classification.csv",
        out_dir / "kerr_packet_observer_scan.csv",
    ]
    if args.skip_existing_products and all(path.exists() for path in required):
        return
    py = args.python_executable
    log = out_dir / "stratified_convergence_run.log"
    commands = [
        [
            py,
            "scripts/cascade/run_geant4_real_resumable_batches.py",
            "--secondaries",
            str(out_dir / "pythia_secondaries.jsonl"),
            "--output-dir",
            str(out_dir),
            "--interaction-points",
            str(out_dir / "interaction_points.jsonl"),
            "--batch-mode",
            "one_particle_per_process",
            "--workers",
            str(args.workers),
            "--geant4-safety-mode",
            "strict",
            "--uhe-transport-policy",
            "skip_to_escaped",
            "--geant4-hadron-max-kinetic-gev",
            "1e5",
            "--geant4-lepton-max-kinetic-gev",
            "1e5",
            "--geant4-photon-max-kinetic-gev",
            "1e5",
            "--allow-partial-exit-zero",
        ],
        [py, "scripts/cascade/summarize_uhe_transport_policy.py", "--output-dir", str(out_dir)],
        [
            py,
            "scripts/cascade/build_escaping_particle_packets.py",
            "--input",
            str(out_dir / "geant4_escaped_particles.jsonl"),
            "--interaction-points",
            str(out_dir / "interaction_points.jsonl"),
            "--output-dir",
            str(out_dir),
            "--aggregate-by-pdg",
            "--require-physical-interaction-points",
            "--sample-label",
            f"{mode}-{frac:g}",
        ],
        [py, "scripts/cascade/audit_packet_origin.py", "--output-dir", str(out_dir)],
        [py, "scripts/cascade/classify_escaping_packets.py", "--input", str(out_dir / "escaping_particle_packets.jsonl"), "--output-dir", str(out_dir)],
        [
            py,
            "scripts/cascade/propagate_kerr_null_packets.py",
            "--packets",
            str(out_dir / "escaping_particle_packets.jsonl"),
            "--classification",
            str(out_dir / "escaping_packet_classification.csv"),
            "--straight-line",
            str(out_dir / "null_propagated_packets.csv"),
            "--output-dir",
            str(out_dir),
            "--kerr-init-mode",
            "zamo_tetrad",
            "--normalize-null-momentum",
        ],
        [
            py,
            "scripts/cascade/scan_kerr_packet_observers.py",
            "--packets",
            str(out_dir / "escaping_particle_packets.jsonl"),
            "--classification",
            str(out_dir / "escaping_packet_classification.csv"),
            "--straight-scan",
            str(out_dir / "packet_observer_scan.csv"),
            "--output-dir",
            str(out_dir),
            "--cones-deg",
            f"{args.cone_deg:g}",
            "--kerr-init-mode",
            "zamo_tetrad",
        ],
        [
            py,
            "scripts/cascade/build_particle_channel_images.py",
            "--output-dir",
            str(out_dir),
            "--packets",
            str(out_dir / "escaping_particle_packets.jsonl"),
            "--classification",
            str(out_dir / "escaping_packet_classification.csv"),
            "--kerr-scan",
            str(out_dir / "kerr_packet_observer_scan.csv"),
            "--observer-mode",
            "best_cone",
            "--cone-deg",
            f"{args.cone_deg:g}",
            "--sample-label",
            f"{mode}-{frac:g}",
        ],
        [
            py,
            "scripts/cascade/build_particle_channel_image_audit.py",
            "--output-dir",
            str(out_dir),
            "--channel-csv",
            str(out_dir / "particle_channel_images.csv"),
            "--channel-summary",
            str(out_dir / "particle_channel_images_summary.md"),
        ],
    ]
    for command in commands:
        ef.run_command(command, root, log)


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_metrics(mode: str, frac: float, out_dir: Path, cone_deg: float) -> dict[str, Any]:
    row = ef.parse_fraction_metrics(frac, out_dir, cone_deg)
    meta = load_metadata(out_dir / "sampling_metadata.json")
    row.update({
        "sampling_mode": mode,
        "processed_energy_fraction": meta.get("selected_energy_fraction", 0.0),
        "channel_coverage_fraction": meta.get("channel_coverage_fraction", 0.0),
        "number_of_pdgs_sampled": meta.get("selected_pdgs", 0),
        "number_of_channels_sampled": len(meta.get("selected_channels", [])),
        "selected_jobs": meta.get("selected_jobs", 0),
    })
    return row


def convergence_by_mode(rows: list[dict[str, Any]], stable_tol: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["sampling_mode"])].append(row)
    for mode, mode_rows in by_mode.items():
        mode_rows.sort(key=lambda row: float(row["fraction_target"]))
        for delta in ef.convergence_rows(mode_rows, stable_tol):
            delta["sampling_mode"] = mode
            out.append(delta)
    return out


def score_modes(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        by_mode[str(row["sampling_mode"])].append(row)
    scores = []
    for mode, rows in by_mode.items():
        late = [row for row in rows if row["pair"].endswith("->f030")]
        instability = sum(ef.finite(row["delta_relative"]) for row in late if row["metric"] in {"gamma_channel_energy", "electromagnetic_channel_energy", "hadronic_channel_energy", "captured_fraction"})
        observer_delta = sum(ef.finite(row["delta_relative"]) for row in late if row["metric"] in {"best_theta", "best_phi"})
        unstable_count = sum(1 for row in late if row["stable"] == "no")
        scores.append({
            "sampling_mode": mode,
            "late_instability_score": instability,
            "late_observer_delta_deg": observer_delta,
            "late_unstable_metrics": unstable_count,
        })
    scores.sort(key=lambda row: (row["late_unstable_metrics"], row["late_instability_score"], row["late_observer_delta_deg"]))
    return scores


def write_outputs(output_root: Path, rows: list[dict[str, Any]], deltas: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    fields = [
        "sampling_mode", "fraction_target", "fraction_label", "processed_energy_fraction",
        "channel_coverage_fraction", "number_of_pdgs_sampled", "number_of_channels_sampled",
        "selected_jobs", "gamma_channel_energy", "electromagnetic_channel_energy",
        "hadronic_channel_energy", "pion_channel_energy", "captured_fraction", "best_theta",
        "best_phi", "number_of_packets", "run_dir",
    ]
    ef.write_csv(output_root / "stratified_convergence.csv", rows, fields)
    ef.write_csv(output_root / "stratified_convergence_deltas.csv", deltas, ["sampling_mode", "pair", "metric", "delta_relative", "stable"])
    ef.write_csv(output_root / "stratified_convergence_mode_scores.csv", scores, ["sampling_mode", "late_instability_score", "late_observer_delta_deg", "late_unstable_metrics"])
    best = scores[0]["sampling_mode"] if scores else "none"
    lines = [
        "# Stratified Channel Convergence Sampling",
        "",
        "Phase 8.5 diagnostic comparison. No new physics is introduced; all products remain weighted-energy proxies, not luminosities.",
        "",
        f"- best mode by late 10%->30% stability score: `{best}`",
        "",
        "## Mode Scores",
        "",
        "| mode | late instability score | late observer delta [deg] | late unstable metrics |",
        "|---|---:|---:|---:|",
    ]
    for row in scores:
        lines.append(f"| {row['sampling_mode']} | {row['late_instability_score']:.6g} | {row['late_observer_delta_deg']:.6g} | {row['late_unstable_metrics']} |")
    lines.extend(["", "## Metrics", "", "| mode | fraction | processed fraction | channel coverage | PDGs | channels | gamma | EM | hadronic | best theta | captured fraction |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['sampling_mode']} | {row['fraction_target']:.3g} | {row['processed_energy_fraction']:.6g} | "
            f"{row['channel_coverage_fraction']:.6g} | {row['number_of_pdgs_sampled']} | {row['number_of_channels_sampled']} | "
            f"{row['gamma_channel_energy']:.6g} | {row['electromagnetic_channel_energy']:.6g} | "
            f"{row['hadronic_channel_energy']:.6g} | {row['best_theta']:.6g} | {row['captured_fraction']:.6g} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The preferred sampling mode is the one with the smallest 10%->30% channel-energy instability and observer drift.",
        "If all modes remain unstable, partial sampling should remain diagnostic only.",
    ])
    (output_root / "stratified_convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_root: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = sorted({str(row["sampling_mode"]) for row in rows})

    def plot_metric(metric: str, ylabel: str, filename: str, yscale: str = "linear") -> None:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for mode in modes:
            mode_rows = sorted([row for row in rows if row["sampling_mode"] == mode], key=lambda row: row["fraction_target"])
            ax.plot([row["fraction_target"] for row in mode_rows], [row[metric] for row in mode_rows], marker="o", label=mode)
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("target processed energy fraction")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    plot_metric("captured_fraction", "captured fraction", "convergence_by_sampling_mode.png")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for mode in modes:
        mode_rows = sorted([row for row in rows if row["sampling_mode"] == mode], key=lambda row: row["fraction_target"])
        ax.plot([row["fraction_target"] for row in mode_rows], [row["gamma_channel_energy"] for row in mode_rows], marker="o", label=f"{mode}: gamma")
        ax.plot([row["fraction_target"] for row in mode_rows], [row["hadronic_channel_energy"] for row in mode_rows], marker="s", linestyle="--", label=f"{mode}: hadronic")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("target processed energy fraction")
    ax.set_ylabel("channel energy [GeV]")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "channel_energy_by_sampling_mode.png", dpi=180)
    plt.close(fig)
    plot_metric("number_of_pdgs_sampled", "sampled PDG count", "pdg_coverage_by_sampling_mode.png")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for mode in modes:
        mode_rows = sorted([row for row in rows if row["sampling_mode"] == mode], key=lambda row: row["fraction_target"])
        ax.plot([row["fraction_target"] for row in mode_rows], [row["best_theta"] for row in mode_rows], marker="o", label=f"{mode} theta")
    ax.set_xscale("log")
    ax.set_xlabel("target processed energy fraction")
    ax.set_ylabel("best observer theta [deg]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "best_observer_by_sampling_mode.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"))
    parser.add_argument("--output-root", type=Path, default=Path("output/convergence_stratified"))
    parser.add_argument("--sampling-mode", choices=MODES, action="append", default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.30])
    parser.add_argument("--min-jobs-per-pdg", type=int, default=1)
    parser.add_argument("--min-jobs-per-channel", type=int, default=1)
    parser.add_argument("--uhe-channel-threshold-gev", type=float, default=1.0e5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cone-deg", type=float, default=30.0)
    parser.add_argument("--stable-tol", type=float, default=0.10)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-existing-products", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    modes = args.sampling_mode or MODES
    source_rows = read_jsonl(args.source_run_dir / "pythia_secondaries.jsonl")
    all_metrics: list[dict[str, Any]] = []
    for mode in modes:
        for frac in args.fractions:
            out_dir = args.output_root / mode / ef.fraction_label(frac)
            if not args.summarize_only:
                prepare_selected_dir(args, mode, frac, source_rows, out_dir)
                run_products(root, args, out_dir, mode, frac)
            all_metrics.append(parse_metrics(mode, frac, out_dir, args.cone_deg))
    deltas = convergence_by_mode(all_metrics, args.stable_tol)
    scores = score_modes(deltas)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_root, all_metrics, deltas, scores)
    make_plots(args.output_root, all_metrics)
    print(json.dumps({
        "output": str(args.output_root / "stratified_convergence_summary.md"),
        "modes": modes,
        "best_mode": scores[0]["sampling_mode"] if scores else None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
