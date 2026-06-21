#!/usr/bin/env python3
"""Diagnose observer overlap for null-propagated escaping packets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


PROPAGABLE = {"MASSLESS_NULL", "ULTRARELATIVISTIC_NULL_OK"}
CONES_DEG = [1.0, 5.0, 10.0, 30.0, 60.0, 90.0]
AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(v * v for v in vec))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return tuple(v / norm for v in vec)  # type: ignore[return-value]


def angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, sum(a[i] * b[i] for i in range(3))))
    return math.degrees(math.acos(dot))


def packet_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("event_id", 0)),
        int(row.get("pdg_id", 0)),
        f"{finite(row.get('energy_gev')):.15g}",
    )


def pair_packets_with_classes(
    packets: list[dict[str, Any]],
    classes: list[dict[str, str]],
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    by_key: dict[tuple[int, int, str], deque[dict[str, str]]] = defaultdict(deque)
    for row in classes:
        by_key[packet_key(row)].append(row)
    paired = []
    for index, packet in enumerate(packets):
        key = packet_key(packet)
        if by_key[key]:
            cls = by_key[key].popleft()
        elif index < len(classes):
            cls = classes[index]
        else:
            cls = {"classification": "UNKNOWN_CLASSIFICATION"}
        paired.append((packet, cls))
    return paired


def observer_axis(name: str) -> tuple[float, float, float]:
    aliases = {"x": "+x", "y": "+y", "z": "+z"}
    return AXES[aliases.get(name, name)]


def build_rows(paired: list[tuple[dict[str, Any], dict[str, str]]], observer: tuple[float, float, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet, cls_row in paired:
        classification = str(cls_row.get("classification", "UNKNOWN_CLASSIFICATION"))
        direction = normalize((
            finite(packet.get("px_gev")),
            finite(packet.get("py_gev")),
            finite(packet.get("pz_gev")),
        ))
        energy = finite(cls_row.get("weighted_energy_gev"), finite(packet.get("weighted_energy_gev"), finite(packet.get("energy_gev"))))
        pdg = int(packet.get("pdg_id", 0))
        if direction is None:
            continue
        angles = {axis: angle_deg(direction, vec) for axis, vec in AXES.items()}
        rows.append({
            "event_id": int(packet.get("event_id", 0)),
            "pdg_id": pdg,
            "particle_label": str(packet.get("particle_label", PDG_LABELS.get(pdg, f"pdg{pdg}"))),
            "classification": classification,
            "propagable": classification in PROPAGABLE,
            "weighted_energy_gev": energy,
            "dir_x": direction[0],
            "dir_y": direction[1],
            "dir_z": direction[2],
            "angle_to_observer_deg": angle_deg(direction, observer),
            **{f"angle_to_{axis.replace('+', 'p').replace('-', 'm')}_deg": value for axis, value in angles.items()},
        })
    return rows


def summarize(rows: list[dict[str, Any]], observer_name: str, observer: tuple[float, float, float]) -> dict[str, Any]:
    prop_rows = [row for row in rows if row["propagable"]]
    total_prop_energy = sum(float(row["weighted_energy_gev"]) for row in prop_rows)
    total_all_energy = sum(float(row["weighted_energy_gev"]) for row in rows)
    cone_energy = {
        cone: sum(float(row["weighted_energy_gev"]) for row in prop_rows if float(row["angle_to_observer_deg"]) <= cone)
        for cone in CONES_DEG
    }
    axis_cone_energy: dict[str, dict[float, float]] = {}
    for axis, vec in AXES.items():
        axis_cone_energy[axis] = {
            cone: sum(float(row["weighted_energy_gev"]) for row in prop_rows if angle_deg((float(row["dir_x"]), float(row["dir_y"]), float(row["dir_z"])), vec) <= cone)
            for cone in CONES_DEG
        }
    best_axis = max(AXES, key=lambda axis: axis_cone_energy[axis][90.0])
    sx = sy = sz = 0.0
    for row in prop_rows:
        energy = float(row["weighted_energy_gev"])
        sx += float(row["dir_x"]) * energy
        sy += float(row["dir_y"]) * energy
        sz += float(row["dir_z"]) * energy
    mean_dir = normalize((sx, sy, sz)) or (0.0, 0.0, 0.0)
    by_pdg: dict[int, dict[str, Any]] = {}
    for row in prop_rows:
        pdg = int(row["pdg_id"])
        item = by_pdg.setdefault(pdg, {
            "pdg_id": pdg,
            "particle_label": row["particle_label"],
            "weighted_energy_gev": 0.0,
            "sx": 0.0,
            "sy": 0.0,
            "sz": 0.0,
        })
        energy = float(row["weighted_energy_gev"])
        item["weighted_energy_gev"] += energy
        item["sx"] += float(row["dir_x"]) * energy
        item["sy"] += float(row["dir_y"]) * energy
        item["sz"] += float(row["dir_z"]) * energy
    pdg_summary = []
    for item in by_pdg.values():
        direction = normalize((item.pop("sx"), item.pop("sy"), item.pop("sz"))) or (0.0, 0.0, 0.0)
        item["dominant_dir_x"] = direction[0]
        item["dominant_dir_y"] = direction[1]
        item["dominant_dir_z"] = direction[2]
        item["angle_to_observer_deg"] = angle_deg(direction, observer)
        pdg_summary.append(item)
    pdg_summary.sort(key=lambda item: item["weighted_energy_gev"], reverse=True)
    min_angle = min((float(row["angle_to_observer_deg"]) for row in prop_rows), default=math.nan)
    return {
        "observer_axis": observer_name,
        "observer_vector": list(observer),
        "total_packet_weighted_energy_gev": total_all_energy,
        "propagable_weighted_energy_gev": total_prop_energy,
        "propagable_count": len(prop_rows),
        "min_angle_to_observer_deg": min_angle,
        "mean_weighted_direction": list(mean_dir),
        "angle_mean_direction_to_observer_deg": angle_deg(mean_dir, observer) if total_prop_energy > 0.0 else math.nan,
        "cone_energy": cone_energy,
        "cone_fraction_of_propagable": {cone: value / max(total_prop_energy, 1.0e-300) for cone, value in cone_energy.items()},
        "best_discrete_observer_axis_90deg": best_axis,
        "best_discrete_observer_energy_90deg": axis_cone_energy[best_axis][90.0],
        "axis_cone_energy": axis_cone_energy,
        "pdg_summary": pdg_summary,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    total = float(summary["propagable_weighted_energy_gev"])
    lines = [
        "# Packet Observer Overlap Diagnostic",
        "",
        "Phase 5.3 geometric diagnostic for effective-null escaping packets.",
        "This is not physical luminosity and does not implement Kerr packet tracing.",
        "",
        f"- observer_axis: `{summary['observer_axis']}`",
        f"- observer_vector: `{summary['observer_vector']}`",
        f"- propagable packets: `{summary['propagable_count']}`",
        f"- propagable weighted energy [GeV]: `{total:.12g}`",
        f"- min angle to observer [deg]: `{summary['min_angle_to_observer_deg']:.12g}`",
        f"- mean weighted direction: `{summary['mean_weighted_direction']}`",
        f"- angle mean direction to observer [deg]: `{summary['angle_mean_direction_to_observer_deg']:.12g}`",
        f"- best discrete observer axis at 90 deg cone: `{summary['best_discrete_observer_axis_90deg']}`",
        f"- best discrete observer energy at 90 deg cone [GeV]: `{summary['best_discrete_observer_energy_90deg']:.12g}`",
        "",
        "## Energy Inside Observer Cones",
        "",
        "| Cone half-angle [deg] | Energy [GeV] | Fraction of propagable energy |",
        "|---:|---:|---:|",
    ]
    for cone, energy in summary["cone_energy"].items():
        lines.append(f"| {cone:g} | {energy:.12g} | {summary['cone_fraction_of_propagable'][cone]:.12g} |")
    lines.extend(["", "## Dominant Direction By PDG", "", "| PDG | Label | Energy [GeV] | dir_x | dir_y | dir_z | angle to observer [deg] |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in summary["pdg_summary"]:
        lines.append(
            f"| {row['pdg_id']} | {row['particle_label']} | {row['weighted_energy_gev']:.12g} | "
            f"{row['dominant_dir_x']:.6g} | {row['dominant_dir_y']:.6g} | {row['dominant_dir_z']:.6g} | "
            f"{row['angle_to_observer_deg']:.6g} |"
        )
    lines.extend([
        "",
        "A zero image in Phase 5.2 can be purely geometric: packets may leave the",
        "domain but miss the chosen observer cone/pixelization. Auto-observer modes",
        "are debug tools and do not define a physical observer inclination.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    prop_rows = [row for row in rows if row["propagable"]]
    weights = [float(row["weighted_energy_gev"]) for row in prop_rows]
    xs = [float(row["dir_x"]) for row in prop_rows]
    ys = [float(row["dir_y"]) for row in prop_rows]
    zs = [float(row["dir_z"]) for row in prop_rows]
    lon = [math.atan2(y, x) for x, y in zip(xs, ys)]
    lat = [math.asin(max(-1.0, min(1.0, z))) for z in zs]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), subplot_kw={"projection": "mollweide"})
    if lon:
        sc = ax.scatter(lon, lat, c=weights, s=55, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="weighted energy [GeV]")
    ax.grid(True, alpha=0.35)
    ax.set_title("Escaping packet directions")
    fig.tight_layout()
    fig.savefig(plots / "packet_direction_sphere.png", dpi=180)
    plt.close(fig)

    angles = [float(row["angle_to_observer_deg"]) for row in prop_rows]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    if angles:
        ax.hist(angles, weights=weights, bins=12, color="#4c78a8")
    ax.set_xlabel("angle to observer [deg]")
    ax.set_ylabel("weighted energy [GeV]")
    ax.set_title("Energy versus observer angle")
    fig.tight_layout()
    fig.savefig(plots / "packet_angle_to_observer.png", dpi=180)
    plt.close(fig)

    cones = list(summary["cone_energy"].keys())
    energies = [summary["cone_energy"][cone] for cone in cones]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(cones, energies, marker="o", color="#f58518")
    ax.set_xlabel("observer cone half-angle [deg]")
    ax.set_ylabel("weighted energy inside cone [GeV]")
    ax.set_title("Packet energy captured by cone")
    fig.tight_layout()
    fig.savefig(plots / "packet_energy_vs_observer_cone.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    if xs:
        labels = [str(row["pdg_id"]) for row in prop_rows]
        sc = ax.scatter(xs, ys, c=weights, s=65, cmap="plasma", edgecolor="black", linewidth=0.3)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=7)
        fig.colorbar(sc, ax=ax, label="weighted energy [GeV]")
    ax.set_xlabel("direction x")
    ax.set_ylabel("direction y")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Packet direction by PDG")
    fig.tight_layout()
    fig.savefig(plots / "packet_direction_by_pdg.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--null-propagated", type=Path, default=Path("output/cascade/null_propagated_packets.csv"))
    parser.add_argument("--summary", type=Path, default=Path("output/cascade/null_packet_propagation_summary.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--observer-axis", choices=["x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z"], default="+z")
    args = parser.parse_args()

    packets = read_jsonl(args.packets)
    classes = read_csv(args.classification)
    paired = pair_packets_with_classes(packets, classes)
    observer = observer_axis(args.observer_axis)
    rows = build_rows(paired, observer)
    summary = summarize(rows, args.observer_axis, observer)
    output = args.output_dir
    fields = [
        "event_id",
        "pdg_id",
        "particle_label",
        "classification",
        "propagable",
        "weighted_energy_gev",
        "dir_x",
        "dir_y",
        "dir_z",
        "angle_to_observer_deg",
        "angle_to_px_deg",
        "angle_to_mx_deg",
        "angle_to_py_deg",
        "angle_to_my_deg",
        "angle_to_pz_deg",
        "angle_to_mz_deg",
    ]
    write_csv(output / "packet_observer_overlap_diagnostic.csv", rows, fields)
    write_markdown(output / "packet_observer_overlap_diagnostic.md", summary)
    make_plots(output, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
