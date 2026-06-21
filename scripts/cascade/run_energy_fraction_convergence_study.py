#!/usr/bin/env python3
"""Run and summarize the Phase 8.4 energy-fraction convergence study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CHANNELS = ["gamma", "electromagnetic", "hadronic", "pion_charged"]


def fraction_label(frac: float) -> str:
    return f"f{int(round(frac * 100)):03d}"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_md_value(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"- {re.escape(key)}:\s*`([^`]+)`", text)
    return finite(match.group(1), default) if match else default


def best_scan_row(path: Path, cone_deg: float) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        return {}
    key = f"captured_energy_cone_{cone_deg:g}_deg"
    if key not in rows[0]:
        key = "captured_energy_gev"
    return max(rows, key=lambda row: finite(row.get(key)))


def channel_map(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("channel", ""): row for row in read_csv(path)}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def copy_input(source: Path, dest: Path, name: str) -> None:
    src = source / name
    dst = dest / name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def run_command(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        handle.write(proc.stdout or "")
        handle.write(f"\n[returncode] {proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with {proc.returncode}: {' '.join(command)}")


def maybe_run_fraction(root: Path, args: argparse.Namespace, frac: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["pythia_secondaries.jsonl", "interaction_points.jsonl", "primary_events.jsonl"]:
        copy_input(args.source_run_dir, out_dir, name)
    required = [
        out_dir / "particle_channel_images_summary.md",
        out_dir / "particle_channel_image_audit.md",
        out_dir / "escaping_packet_classification.csv",
        out_dir / "kerr_packet_observer_scan.csv",
    ]
    if args.skip_existing_products and all(path.exists() for path in required):
        return
    py = args.python_executable
    log = out_dir / "energy_fraction_convergence_run.log"
    run_command(
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
            "--target-processed-energy-fraction",
            f"{frac:.17g}",
            "--prioritize-energy-desc",
            "--allow-partial-exit-zero",
        ],
        root,
        log,
    )
    commands = [
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
            f"energy-fraction-{frac:g}",
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
            f"energy-fraction-{frac:g}",
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
        run_command(command, root, log)


def parse_fraction_metrics(frac: float, out_dir: Path, cone_deg: float) -> dict[str, Any]:
    batch = out_dir / "geant4_batch_summary.md"
    classes = read_csv(out_dir / "escaping_packet_classification.csv")
    packets = read_csv(out_dir / "escaping_particle_packets.csv")
    scan = best_scan_row(out_dir / "kerr_packet_observer_scan.csv", cone_deg)
    channels = channel_map(out_dir / "particle_channel_images.csv")
    class_total = sum(finite(row.get("weighted_energy_gev")) for row in classes)
    null_ok = sum(
        finite(row.get("weighted_energy_gev"))
        for row in classes
        if row.get("classification") in {"MASSLESS_NULL", "ULTRARELATIVISTIC_NULL_OK"}
    )
    key_energy = f"captured_energy_cone_{cone_deg:g}_deg"
    key_fraction = f"captured_fraction_cone_{cone_deg:g}_deg"
    return {
        "fraction_target": frac,
        "fraction_label": fraction_label(frac),
        "processed_energy": read_md_value(batch, "processed_energy_gev"),
        "unsupported_uhe_energy": read_md_value(batch, "escaped_unsupported_uhe_energy_gev"),
        "deposited_energy": read_md_value(batch, "deposited_energy_gev"),
        "escaped_energy": read_md_value(batch, "escaped_energy_gev"),
        "null_compatible_fraction": null_ok / max(class_total, 1.0e-300),
        "best_theta": finite(scan.get("theta_obs_deg")),
        "best_phi": finite(scan.get("phi_obs_deg")),
        "best_cone_energy": finite(scan.get(key_energy, scan.get("captured_energy_gev"))),
        "captured_fraction": finite(scan.get(key_fraction, scan.get("captured_fraction"))),
        "gamma_channel_energy": finite(channels.get("gamma", {}).get("image_energy_gev")),
        "electromagnetic_channel_energy": finite(channels.get("electromagnetic", {}).get("image_energy_gev")),
        "hadronic_channel_energy": finite(channels.get("hadronic", {}).get("image_energy_gev")),
        "pion_channel_energy": finite(channels.get("pion_charged", {}).get("image_energy_gev")),
        "number_of_packets": len(packets),
        "run_dir": str(out_dir),
    }


def angle_delta_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def relative_delta(old: float, new: float) -> float:
    return abs(new - old) / max(abs(old), 1.0e-300)


def convergence_rows(rows: list[dict[str, Any]], stable_tol: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for prev, cur in zip(rows, rows[1:]):
        pair = f"{prev['fraction_label']}->{cur['fraction_label']}"
        checks = {
            "captured_fraction": relative_delta(prev["captured_fraction"], cur["captured_fraction"]),
            "best_theta": angle_delta_deg(prev["best_theta"], cur["best_theta"]),
            "best_phi": angle_delta_deg(prev["best_phi"], cur["best_phi"]),
            "gamma_channel_energy": relative_delta(prev["gamma_channel_energy"], cur["gamma_channel_energy"]),
            "electromagnetic_channel_energy": relative_delta(prev["electromagnetic_channel_energy"], cur["electromagnetic_channel_energy"]),
            "hadronic_channel_energy": relative_delta(prev["hadronic_channel_energy"], cur["hadronic_channel_energy"]),
        }
        for metric, delta in checks.items():
            if metric in {"best_theta", "best_phi"}:
                stable = delta <= 5.0
            else:
                stable = delta <= stable_tol
            out.append({"pair": pair, "metric": metric, "delta_relative": delta, "stable": "yes" if stable else "no"})
    return out


def first_stable_fraction(rows: list[dict[str, Any]], conv: list[dict[str, Any]]) -> str:
    if len(rows) < 2:
        return "insufficient data"
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in conv:
        by_pair.setdefault(row["pair"], []).append(row)
    labels = [row["fraction_label"] for row in rows]
    for label in labels[1:]:
        idx = labels.index(label)
        remaining_pairs = [f"{labels[i]}->{labels[i+1]}" for i in range(idx - 1, len(labels) - 1)]
        if remaining_pairs and all(all(item["stable"] == "yes" for item in by_pair.get(pair, [])) for pair in remaining_pairs):
            return label
    return "not converged over tested fractions"


def write_summary(output_root: Path, rows: list[dict[str, Any]], conv: list[dict[str, Any]]) -> None:
    stable_from = first_stable_fraction(rows, conv)
    dominant_channels = []
    for row in rows:
        energies = {
            "gamma": row["gamma_channel_energy"],
            "electromagnetic": row["electromagnetic_channel_energy"],
            "hadronic": row["hadronic_channel_energy"],
            "pion_charged": row["pion_channel_energy"],
        }
        dominant_channels.append(max(energies, key=energies.get))
    observer_changes = any(
        angle_delta_deg(rows[i]["best_theta"], rows[i + 1]["best_theta"]) > 5.0
        or angle_delta_deg(rows[i]["best_phi"], rows[i + 1]["best_phi"]) > 5.0
        for i in range(len(rows) - 1)
    )
    anisotropy_changes = observer_changes or any(row["stable"] == "no" for row in conv if row["metric"] in {"captured_fraction", "best_theta", "best_phi"})
    channel_changes = len(set(dominant_channels)) > 1
    lines = [
        "# Energy Fraction Convergence Study",
        "",
        "Diagnostic HADROS-CASCADE Phase 8.4 study. These are weighted-energy proxy images, not physical luminosities.",
        "",
        f"- tested fractions: `{', '.join(str(row['fraction_target']) for row in rows)}`",
        f"- apparent stable fraction: `{stable_from}`",
        f"- qualitative anisotropy change: `{'yes' if anisotropy_changes else 'no'}`",
        f"- dominant channel changes: `{'yes' if channel_changes else 'no'}`",
        f"- best-cone observer changes: `{'yes' if observer_changes else 'no'}`",
        "",
        "## Fraction Metrics",
        "",
        "| fraction | processed energy | captured fraction | best theta | best phi | gamma | EM | hadronic | packets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fraction_target']:.3g} | {row['processed_energy']:.6g} | {row['captured_fraction']:.6g} | "
            f"{row['best_theta']:.6g} | {row['best_phi']:.6g} | {row['gamma_channel_energy']:.6g} | "
            f"{row['electromagnetic_channel_energy']:.6g} | {row['hadronic_channel_energy']:.6g} | {row['number_of_packets']} |"
        )
    lines.extend(["", "## Successive Differences", "", "| pair | metric | delta | stable |", "|---|---|---:|---|"])
    for row in conv:
        lines.append(f"| {row['pair']} | {row['metric']} | {row['delta_relative']:.6g} | {row['stable']} |")
    lines.extend([
        "",
        "## Answers",
        "",
        f"- A partir de qual fração os resultados parecem estabilizar? `{stable_from}`.",
        f"- Existe mudança qualitativa na anisotropia? `{'yes' if anisotropy_changes else 'no'}`.",
        f"- Os canais dominantes mudam? `{'yes' if channel_changes else 'no'}`.",
        f"- O observador best-cone muda? `{'yes' if observer_changes else 'no'}`.",
        "",
        "If the answer is `not converged over tested fractions`, a full or higher-fraction run remains required before using the partial sample for rapid physical exploration.",
    ])
    (output_root / "energy_fraction_convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_root: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    x = [row["fraction_target"] for row in rows]

    def save_line(ykey: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.plot(x, [row[ykey] for row in rows], marker="o")
        ax.set_xlabel("processed energy fraction target")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    save_line("captured_fraction", "best-cone captured fraction", "convergence_captured_fraction.png")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(x, [row["best_theta"] for row in rows], marker="o", label="theta")
    ax.plot(x, [row["best_phi"] for row in rows], marker="s", label="phi")
    ax.set_xscale("log")
    ax.set_xlabel("processed energy fraction target")
    ax.set_ylabel("best observer angle [deg]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "convergence_best_observer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, label in [
        ("gamma_channel_energy", "gamma"),
        ("electromagnetic_channel_energy", "electromagnetic"),
        ("hadronic_channel_energy", "hadronic"),
        ("pion_channel_energy", "charged pion"),
    ]:
        ax.plot(x, [row[key] for row in rows], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("processed energy fraction target")
    ax.set_ylabel("captured channel energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "convergence_channel_energies.png", dpi=180)
    plt.close(fig)
    save_line("gamma_channel_energy", "gamma channel energy [GeV]", "convergence_gamma_channel.png")
    save_line("hadronic_channel_energy", "hadronic channel energy [GeV]", "convergence_hadronic_channel.png")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, label in [
        ("processed_energy", "processed"),
        ("unsupported_uhe_energy", "unsupported UHE"),
        ("deposited_energy", "deposited"),
        ("escaped_energy", "escaped"),
    ]:
        ax.plot(x, [row[key] for row in rows], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("processed energy fraction target")
    ax.set_ylabel("energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "convergence_energy_budget.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"))
    parser.add_argument("--output-root", type=Path, default=Path("output/convergence"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.30])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cone-deg", type=float, default=30.0)
    parser.add_argument("--stable-tol", type=float, default=0.10)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-existing-products", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_root.mkdir(parents=True, exist_ok=True)
    fraction_rows: list[dict[str, Any]] = []
    for frac in args.fractions:
        out_dir = args.output_root / fraction_label(frac)
        if not args.summarize_only:
            maybe_run_fraction(root, args, frac, out_dir)
        fraction_rows.append(parse_fraction_metrics(frac, out_dir, args.cone_deg))
    conv = convergence_rows(fraction_rows, args.stable_tol)
    fields = [
        "fraction_target", "fraction_label", "processed_energy", "unsupported_uhe_energy",
        "deposited_energy", "escaped_energy", "null_compatible_fraction", "best_theta",
        "best_phi", "best_cone_energy", "captured_fraction", "gamma_channel_energy",
        "electromagnetic_channel_energy", "hadronic_channel_energy", "pion_channel_energy",
        "number_of_packets", "run_dir",
    ]
    write_csv(args.output_root / "energy_fraction_convergence.csv", fraction_rows, fields)
    write_csv(args.output_root / "energy_fraction_convergence_deltas.csv", conv, ["pair", "metric", "delta_relative", "stable"])
    write_summary(args.output_root, fraction_rows, conv)
    make_plots(args.output_root, fraction_rows)
    print(json.dumps({
        "fractions": args.fractions,
        "summary": str(args.output_root / "convergence_summary.md"),
        "csv": str(args.output_root / "energy_fraction_convergence.csv"),
        "stable_from": first_stable_fraction(fraction_rows, conv),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
