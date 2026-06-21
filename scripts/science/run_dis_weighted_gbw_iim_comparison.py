#!/usr/bin/env python3
"""Run a conservative DIS-weighted GBW/IIM cascade comparison.

The downstream events are reused. Only event statistical weights are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"
DEFAULT_OUTPUT = ROOT / "output/science/dis_weighted"
DEFAULT_DOC = ROOT / "docs/science/DIS_WEIGHTED_GBW_IIM_CASCADE_STUDY.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def weighted_budget(source_dir: Path, weights_csv: Path, weight_column: str) -> dict[str, float]:
    weights = {
        int(float(row["event_id"])): fnum(row.get(weight_column), math.nan)
        for row in read_csv(weights_csv)
    }
    out = {
        "E_dep_weighted": 0.0,
        "E_esc_weighted": 0.0,
        "E_invisible_weighted": 0.0,
        "E_untracked_weighted": 0.0,
        "E_unsupported_UHE_weighted": 0.0,
        "input_energy_weighted": 0.0,
    }
    for row in read_csv(source_dir / "geant4_energy_budget.csv"):
        event_id = int(float(row.get("event_id", 0)))
        w = weights.get(event_id, math.nan)
        if not math.isfinite(w):
            continue
        out["input_energy_weighted"] += fnum(row.get("input_energy_gev")) * w
        out["E_dep_weighted"] += fnum(row.get("deposited_energy_gev")) * w
        out["E_esc_weighted"] += fnum(row.get("escaped_energy_gev")) * w
        out["E_invisible_weighted"] += fnum(row.get("invisible_energy_gev")) * w
        out["E_untracked_weighted"] += fnum(row.get("untracked_energy_gev")) * w
        out["E_unsupported_UHE_weighted"] += fnum(row.get("escaped_unsupported_uhe_energy_gev")) * w
    return out


def channel_summary(run_dir: Path) -> dict[str, float]:
    out = {}
    for row in read_csv(run_dir / "particle_channel_images.csv"):
        channel = row.get("channel", "")
        out[f"{channel}_energy_weighted"] = fnum(row.get("total_energy_gev"))
        out[f"{channel}_captured_weighted"] = fnum(row.get("image_energy_gev"))
    return out


def parse_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            out[key.strip().replace(" ", "_")] = value.strip().strip("`")
    return out


def build_model(source_dir: Path, weights_csv: Path, output_dir: Path, model: str, weight_column: str) -> dict[str, Any]:
    model_dir = output_dir / model.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "scripts/cascade/build_escaping_particle_packets.py",
        "--input",
        str(source_dir / "geant4_escaped_particles.jsonl"),
        "--output-dir",
        str(model_dir),
        "--interaction-points",
        str(source_dir / "interaction_points.jsonl"),
        "--event-weights",
        str(weights_csv),
        "--weight-column",
        weight_column,
        "--sample-label",
        f"DIS_weighted_{model}",
    ])
    run([
        sys.executable,
        "scripts/cascade/classify_escaping_packets.py",
        "--input",
        str(model_dir / "escaping_particle_packets.jsonl"),
        "--output-dir",
        str(model_dir),
    ])
    run([
        sys.executable,
        "scripts/cascade/propagate_kerr_null_packets.py",
        "--packets",
        str(model_dir / "escaping_particle_packets.jsonl"),
        "--classification",
        str(model_dir / "escaping_packet_classification.csv"),
        "--output-dir",
        str(model_dir),
        "--packet-propagation-backend",
        "real_kerr_geodesic",
        "--kerr-init-mode",
        "zamo_tetrad",
    ])
    run([
        sys.executable,
        "scripts/cascade/build_particle_channel_images.py",
        "--output-dir",
        str(model_dir),
        "--packets",
        str(model_dir / "escaping_particle_packets.jsonl"),
        "--classification",
        str(model_dir / "escaping_packet_classification.csv"),
        "--packet-propagation-backend",
        "real_kerr_geodesic",
        "--real-kerr-propagated",
        str(model_dir / "real_kerr_propagated_packets.csv"),
        "--observer-mode",
        "best_cone",
        "--cone-deg",
        "30",
        "--dis-weight-model",
        model,
    ])
    packet_summary = parse_summary(model_dir / "escaping_particle_packets_summary.md")
    image_summary = parse_summary(model_dir / "particle_channel_images_summary.md")
    row: dict[str, Any] = {
        "model": model,
        "weight_column": weight_column,
        "run_dir": str(model_dir),
        **weighted_budget(source_dir, weights_csv, weight_column),
        **channel_summary(model_dir),
        "packet_weighted_energy_gev": fnum(packet_summary.get("packet_weighted_energy_gev")),
        "null_ok_fraction": fnum(image_summary.get("null_ok_fraction")),
        "best_theta_deg": fnum(image_summary.get("theta_deg")),
        "best_phi_deg": fnum(image_summary.get("phi_deg")),
        "best_cone_deg": fnum(image_summary.get("cone_deg")),
        "captured_fraction": (
            fnum(channel_summary(model_dir).get("total_escaping_null_ok_captured_weighted"))
            / max(fnum(channel_summary(model_dir).get("total_escaping_null_ok_energy_weighted")), 1.0e-300)
        ),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in rows]
    def vals(key: str) -> list[float]:
        return [fnum(row.get(key)) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(rows)
    for key, label in [
        ("E_dep_weighted", "deposited"),
        ("E_esc_weighted", "escaped"),
        ("E_invisible_weighted", "invisible"),
        ("E_untracked_weighted", "untracked"),
        ("E_unsupported_UHE_weighted", "unsupported UHE"),
    ]:
        v = vals(key)
        ax.bar(labels, v, bottom=bottom, label=label)
        bottom = [bottom[i] + v[i] for i in range(len(v))]
    ax.set_yscale("symlog", linthresh=1e-40)
    ax.set_ylabel("DIS-weighted energy [GeV]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "weighted_energy_budget.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    xs = range(len(rows))
    for j, key in enumerate(["gamma_energy_weighted", "electromagnetic_energy_weighted", "hadronic_energy_weighted", "pion_charged_energy_weighted"]):
        ax.bar([x + (j - 1.5) * width for x in xs], vals(key), width=width, label=key.replace("_energy_weighted", ""))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_yscale("symlog", linthresh=1e-40)
    ax.set_ylabel("DIS-weighted channel energy [GeV]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "weighted_channel_composition.png", dpi=180)
    plt.close(fig)

    if len(rows) == 2:
        ratios = []
        names = []
        for key in ["E_dep_weighted", "E_esc_weighted", "gamma_energy_weighted", "electromagnetic_energy_weighted", "hadronic_energy_weighted"]:
            g = fnum(rows[0].get(key))
            i = fnum(rows[1].get(key))
            names.append(key.replace("_weighted", ""))
            ratios.append(i / g if g != 0.0 else math.nan)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, ratios)
        ax.axhline(1.0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("IIM / GBW")
        fig.tight_layout()
        fig.savefig(plots / "weighted_gbw_iim_ratio.png", dpi=180)
        plt.close(fig)


def write_report(doc_path: Path, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DIS-Weighted GBW/IIM Cascade Study",
        "",
        "This is the first conservative DIS-weighted cascade reweighting study.",
        "",
        "The events, PYTHIA secondaries, GEANT4 products, packet directions, and Kerr geodesic trajectories are reused. Only event statistical weights are changed according to `P_int_model = 1 - exp(-tau_model)`.",
        "",
        "This is not full GBW/IIM event generation.",
        "",
        "| model | E_dep weighted | E_esc weighted | E_unsupported_UHE weighted | gamma weighted | EM weighted | hadronic weighted | captured fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {fnum(row.get('E_dep_weighted')):.12g} | {fnum(row.get('E_esc_weighted')):.12g} | "
            f"{fnum(row.get('E_unsupported_UHE_weighted')):.12g} | {fnum(row.get('gamma_energy_weighted')):.12g} | "
            f"{fnum(row.get('electromagnetic_energy_weighted')):.12g} | {fnum(row.get('hadronic_energy_weighted')):.12g} | "
            f"{fnum(row.get('captured_fraction')):.12g} |"
        )
    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- Same events and secondaries are used for both models.",
        "- Differences come only from DIS-dependent event weights.",
        "- PYTHIA and GEANT4 are not regenerated by DIS model.",
        "- The channel images remain weighted-energy proxy maps, not luminosities.",
    ]
    text = "\n".join(lines) + "\n"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")
    (output_dir / "DIS_WEIGHTED_GBW_IIM_CASCADE_STUDY.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir
    run([
        sys.executable,
        "scripts/science/build_dis_event_weights.py",
        "--interaction-points",
        str(args.source_run_dir / "interaction_points.jsonl"),
        "--primary-events",
        str(args.source_run_dir / "primary_events.jsonl"),
        "--output-dir",
        str(weights_dir),
    ])
    weights_csv = weights_dir / "dis_event_weights.csv"
    rows = [
        build_model(args.source_run_dir, weights_csv, args.output_dir, "GBW", "weight_GBW"),
        build_model(args.source_run_dir, weights_csv, args.output_dir, "IIM", "weight_IIM"),
    ]
    write_csv(args.output_dir / "dis_weighted_gbw_iim_summary.csv", rows)
    make_plots(args.output_dir, rows)
    write_report(args.doc_path, args.output_dir, rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
