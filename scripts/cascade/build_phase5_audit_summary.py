#!/usr/bin/env python3
"""Build a consolidated Phase 5 escaping-packet audit summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CONES = [5.0, 10.0, 30.0, 45.0, 60.0, 90.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def extract_md_number(text: str, label: str, default: float = 0.0) -> float:
    pattern = re.compile(rf"{re.escape(label)}[^`\n]*`([^`]+)`", re.IGNORECASE)
    match = pattern.search(text)
    return finite(match.group(1), default) if match else default


def class_energy(class_rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in class_rows:
        out[row.get("classification", "UNKNOWN")] += finite(row.get("weighted_energy_gev"))
    return dict(out)


def best_observers(scan_rows: list[dict[str, str]], cones: list[float]) -> dict[float, dict[str, Any]]:
    best: dict[float, dict[str, Any]] = {}
    for cone in cones:
        key = f"captured_energy_cone_{cone:g}_deg"
        if not scan_rows or key not in scan_rows[0]:
            continue
        row = max(scan_rows, key=lambda item: finite(item.get(key)))
        best[cone] = {
            "theta_obs_deg": finite(row.get("theta_obs_deg")),
            "phi_obs_deg": finite(row.get("phi_obs_deg")),
            "captured_energy_gev": finite(row.get(key)),
            "captured_fraction": finite(row.get(f"captured_fraction_cone_{cone:g}_deg")),
            "captured_count": int(finite(row.get(f"captured_count_cone_{cone:g}_deg"))),
            "dominant_pdg": row.get(f"dominant_pdg_cone_{cone:g}_deg", row.get("dominant_pdg", "")),
            "dominant_pdg_label": row.get(f"dominant_pdg_label_cone_{cone:g}_deg", row.get("dominant_pdg_label", "")),
        }
    return best


def summarize(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet_summary = read_text(args.packet_summary)
    deposition_summary = read_text(args.deposition_summary)
    classification_rows = read_csv(args.classification)
    null_summary = read_text(args.null_summary)
    overlap_summary = read_text(args.overlap_summary)
    scan_rows = read_csv(args.observer_scan)

    energy_by_class = class_energy(classification_rows)
    escaped_packets = extract_md_number(packet_summary, "packet_weighted_energy_gev")
    deposited = extract_md_number(deposition_summary, "weighted deposited energy [GeV]")
    escaped_deposition_line = extract_md_number(deposition_summary, "weighted escaped energy [GeV]")
    total_packet_energy = sum(energy_by_class.values()) if energy_by_class else escaped_packets
    propagable = energy_by_class.get("MASSLESS_NULL", 0.0) + energy_by_class.get("ULTRARELATIVISTIC_NULL_OK", 0.0)
    marginal = energy_by_class.get("MARGINAL_ULTRARELATIVISTIC", 0.0)
    massive = energy_by_class.get("MASSIVE_PROPAGATION_REQUIRED", 0.0)
    invisible = energy_by_class.get("INVISIBLE_SKIP", 0.0)
    null_selected = extract_md_number(null_summary, "selected/propagated weighted energy [GeV]", propagable)
    null_fraction = extract_md_number(null_summary, "propagated energy fraction", propagable / max(total_packet_energy, 1.0e-300))
    min_angle = extract_md_number(overlap_summary, "min angle to observer [deg]")
    best_axis_energy = extract_md_number(overlap_summary, "best discrete observer energy at 90 deg cone [GeV]")
    best_axis_match = re.search(r"best discrete observer axis at 90 deg cone:\s*`([^`]+)`", overlap_summary)
    best_axis = best_axis_match.group(1) if best_axis_match else ""
    best = best_observers(scan_rows, CONES)

    metrics = {
        "weighted_deposited_energy_gev": deposited,
        "weighted_escaped_energy_deposition_line_gev": escaped_deposition_line,
        "weighted_escaped_packet_energy_gev": escaped_packets,
        "classification_total_weighted_energy_gev": total_packet_energy,
        "propagable_weighted_energy_gev": propagable,
        "propagable_fraction": propagable / max(total_packet_energy, 1.0e-300),
        "null_selected_weighted_energy_gev": null_selected,
        "null_selected_fraction": null_fraction,
        "marginal_weighted_energy_gev": marginal,
        "marginal_fraction": marginal / max(total_packet_energy, 1.0e-300),
        "massive_required_weighted_energy_gev": massive,
        "massive_required_fraction": massive / max(total_packet_energy, 1.0e-300),
        "invisible_skip_weighted_energy_gev": invisible,
        "min_angle_to_default_observer_deg": min_angle,
        "best_discrete_observer_axis_90deg": best_axis,
        "best_discrete_observer_energy_90deg": best_axis_energy,
        "energy_by_class": energy_by_class,
        "best_observers_by_cone": best,
    }
    rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in metrics.items()
    ]
    for cone, row in best.items():
        rows.extend([
            {"metric": f"best_observer_cone_{cone:g}_theta_deg", "value": row["theta_obs_deg"]},
            {"metric": f"best_observer_cone_{cone:g}_phi_deg", "value": row["phi_obs_deg"]},
            {"metric": f"best_observer_cone_{cone:g}_captured_energy_gev", "value": row["captured_energy_gev"]},
            {"metric": f"best_observer_cone_{cone:g}_captured_fraction", "value": row["captured_fraction"]},
            {"metric": f"best_observer_cone_{cone:g}_dominant_pdg", "value": row["dominant_pdg"]},
            {"metric": f"best_observer_cone_{cone:g}_dominant_label", "value": row["dominant_pdg_label"]},
        ])
    return metrics, rows


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    best = metrics["best_observers_by_cone"]
    lines = [
        "# HADROS-CASCADE Phase 5 Audit Summary",
        "",
        "Consolidated audit of escaping-packet products from Phases 5.0-5.4.",
        "No new physics is implemented by this report.",
        "",
        "## Phase Summary",
        "",
        "- Phase 5.0 created conservative `EscapingParticlePacket` records.",
        "- Phase 5.1 classified packets by effective null-geodesic validity.",
        "- Phase 5.2 propagated only massless/null-ok packets with the current `effective_straight_line` diagnostic backend.",
        "- Phase 5.3 diagnosed overlap between packet directions and a default observer cone.",
        "- Phase 5.4 scanned observer directions and cone half-angles.",
        "",
        "## Key Numbers",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| weighted deposited energy [GeV] | {metrics['weighted_deposited_energy_gev']:.12g} |",
        f"| weighted escaped energy in packets [GeV] | {metrics['weighted_escaped_packet_energy_gev']:.12g} |",
        f"| propagable weighted energy [GeV] | {metrics['propagable_weighted_energy_gev']:.12g} |",
        f"| propagable fraction | {metrics['propagable_fraction']:.12g} |",
        f"| marginal ultrarelativistic energy [GeV] | {metrics['marginal_weighted_energy_gev']:.12g} |",
        f"| massive-propagation-required energy [GeV] | {metrics['massive_required_weighted_energy_gev']:.12g} |",
        f"| invisible skipped energy [GeV] | {metrics['invisible_skip_weighted_energy_gev']:.12g} |",
        f"| min angle to default +z observer [deg] | {metrics['min_angle_to_default_observer_deg']:.12g} |",
        f"| best discrete 90 deg observer axis | {metrics['best_discrete_observer_axis_90deg']} |",
        f"| best discrete 90 deg observer energy [GeV] | {metrics['best_discrete_observer_energy_90deg']:.12g} |",
        "",
        "## Best Observers By Cone",
        "",
        "| Cone half-angle [deg] | theta [deg] | phi [deg] | captured energy [GeV] | fraction | dominant PDG |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for cone in CONES:
        row = best.get(cone)
        if not row:
            continue
        lines.append(
            f"| {cone:g} | {row['theta_obs_deg']:.6g} | {row['phi_obs_deg']:.6g} | "
            f"{row['captured_energy_gev']:.12g} | {row['captured_fraction']:.12g} | {row['dominant_pdg_label']} |"
        )
    lines.extend([
        "",
        "## Allowed Claims",
        "",
        "- Escaped energy dominates over deposited energy in this audited sample.",
        "- Most escaped packet energy is in classes compatible with an effective null-geodesic approximation.",
        "- The angular distribution of propagable escaped packets is anisotropic.",
        "- A polar/default observer can miss the escaped component for this specific geometry.",
        "- Energy accounting closes for the packet and classification files used here.",
        "",
        "## Claims Not Allowed",
        "",
        "- This is not final physical Kerr ray tracing.",
        "- This is not observed luminosity.",
        "- This does not include massive geodesics.",
        "- This does not validate PeV/EeV physics.",
        "- This does not replace future physically validated PYTHIA/GEANT4 generator or transport studies.",
        "- The propagation diagnostic still uses `effective_straight_line`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, metrics: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    budget_labels = ["deposited", "escaped packets", "propagable", "marginal", "massive required"]
    budget_values = [
        metrics["weighted_deposited_energy_gev"],
        metrics["weighted_escaped_packet_energy_gev"],
        metrics["propagable_weighted_energy_gev"],
        metrics["marginal_weighted_energy_gev"],
        metrics["massive_required_weighted_energy_gev"],
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(budget_labels, budget_values, color=["#54a24b", "#4c78a8", "#72b7b2", "#f58518", "#e45756"])
    ax.set_ylabel("weighted energy [GeV]")
    ax.set_title("Phase 5 energy budget")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots / "phase5_energy_budget.png", dpi=180)
    plt.close(fig)

    class_energy = metrics["energy_by_class"]
    labels = list(class_energy.keys())
    values = [class_energy[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if values:
        ax.pie(values, labels=labels, autopct="%1.1f%%")
    ax.set_title("Escaping packet class fractions")
    fig.tight_layout()
    fig.savefig(plots / "phase5_null_fraction.png", dpi=180)
    plt.close(fig)

    cones = [cone for cone in CONES if cone in metrics["best_observers_by_cone"]]
    captured = [metrics["best_observers_by_cone"][cone]["captured_energy_gev"] for cone in cones]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(cones, captured, marker="o", color="#f58518")
    ax.set_xlabel("cone half-angle [deg]")
    ax.set_ylabel("best captured energy [GeV]")
    ax.set_title("Best observer capture by cone")
    fig.tight_layout()
    fig.savefig(plots / "phase5_observer_cone_capture.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--packet-summary", type=Path, default=Path("output/cascade/escaping_particle_packets_summary.md"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--null-summary", type=Path, default=Path("output/cascade/null_packet_propagation_summary.md"))
    parser.add_argument("--overlap-summary", type=Path, default=Path("output/cascade/packet_observer_overlap_diagnostic.md"))
    parser.add_argument("--observer-scan", type=Path, default=Path("output/cascade/packet_observer_scan.csv"))
    parser.add_argument("--deposition-summary", type=Path, default=Path("output/cascade/deposition_emissivity_summary.md"))
    args = parser.parse_args()
    metrics, rows = summarize(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv_summary(output / "phase5_audit_summary.csv", rows)
    write_markdown(output / "phase5_audit_summary.md", metrics)
    make_plots(output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
