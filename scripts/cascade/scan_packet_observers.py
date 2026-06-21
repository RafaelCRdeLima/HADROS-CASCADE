#!/usr/bin/env python3
"""Scan angular observers for effective-null escaping packets."""

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


def direction_from_angles(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


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


def load_propagable(packets: Path, classification: Path) -> tuple[list[dict[str, Any]], dict[str, float], tuple[float, float, float]]:
    paired = pair_packets_with_classes(read_jsonl(packets), read_csv(classification))
    rows: list[dict[str, Any]] = []
    energy_by_class: dict[str, float] = defaultdict(float)
    sx = sy = sz = 0.0
    for packet, cls_row in paired:
        cls = str(cls_row.get("classification", "UNKNOWN_CLASSIFICATION"))
        energy = finite(cls_row.get("weighted_energy_gev"), finite(packet.get("weighted_energy_gev"), finite(packet.get("energy_gev"))))
        energy_by_class[cls] += energy
        direction = normalize((
            finite(packet.get("px_gev")),
            finite(packet.get("py_gev")),
            finite(packet.get("pz_gev")),
        ))
        if cls not in PROPAGABLE or direction is None:
            continue
        pdg = int(packet.get("pdg_id", 0))
        rows.append({
            "event_id": int(packet.get("event_id", 0)),
            "pdg_id": pdg,
            "particle_label": str(packet.get("particle_label", PDG_LABELS.get(pdg, f"pdg{pdg}"))),
            "classification": cls,
            "weighted_energy_gev": energy,
            "dir_x": direction[0],
            "dir_y": direction[1],
            "dir_z": direction[2],
        })
        sx += direction[0] * energy
        sy += direction[1] * energy
        sz += direction[2] * energy
    mean = normalize((sx, sy, sz)) or (0.0, 0.0, 1.0)
    return rows, dict(energy_by_class), mean


def scan(
    packets: list[dict[str, Any]],
    energy_by_class: dict[str, float],
    mean_dir: tuple[float, float, float],
    theta_values: list[float],
    phi_values: list[float],
    cones: list[float],
) -> list[dict[str, Any]]:
    total_prop = sum(float(row["weighted_energy_gev"]) for row in packets)
    rows: list[dict[str, Any]] = []
    for theta in theta_values:
        for phi in phi_values:
            axis = direction_from_angles(theta, phi)
            angles = [(packet, angle_deg((float(packet["dir_x"]), float(packet["dir_y"]), float(packet["dir_z"])), axis)) for packet in packets]
            by_pdg: dict[int, float] = defaultdict(float)
            by_class: dict[str, float] = defaultdict(float)
            captured_default = 0.0
            captured_count = 0
            row: dict[str, Any] = {
                "theta_obs_deg": theta,
                "phi_obs_deg": phi,
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "propagable_energy_gev": total_prop,
                "angle_to_mean_direction_deg": angle_deg(axis, mean_dir),
            }
            for cone in cones:
                selected = [(packet, ang) for packet, ang in angles if ang <= cone]
                energy = sum(float(packet["weighted_energy_gev"]) for packet, _ in selected)
                cone_by_pdg: dict[int, float] = defaultdict(float)
                cone_by_class: dict[str, float] = defaultdict(float)
                for packet, _ in selected:
                    cone_by_pdg[int(packet["pdg_id"])] += float(packet["weighted_energy_gev"])
                    cone_by_class[str(packet["classification"])] += float(packet["weighted_energy_gev"])
                cone_dominant_pdg = max(cone_by_pdg, key=cone_by_pdg.get) if cone_by_pdg else 0
                cone_dominant_class = max(cone_by_class, key=cone_by_class.get) if cone_by_class else ""
                row[f"captured_energy_cone_{cone:g}_deg"] = energy
                row[f"captured_fraction_cone_{cone:g}_deg"] = energy / max(total_prop, 1.0e-300)
                row[f"captured_count_cone_{cone:g}_deg"] = len(selected)
                row[f"dominant_pdg_cone_{cone:g}_deg"] = cone_dominant_pdg
                row[f"dominant_pdg_label_cone_{cone:g}_deg"] = PDG_LABELS.get(cone_dominant_pdg, f"pdg{cone_dominant_pdg}") if cone_dominant_pdg else ""
                row[f"dominant_class_cone_{cone:g}_deg"] = cone_dominant_class
                row[f"captured_energy_by_pdg_cone_{cone:g}_deg_json"] = json.dumps({str(k): v for k, v in sorted(cone_by_pdg.items())}, sort_keys=True)
                if cone == cones[0]:
                    captured_default = energy
                    captured_count = len(selected)
                    by_pdg = cone_by_pdg
                    by_class = cone_by_class
            dominant_pdg = max(by_pdg, key=by_pdg.get) if by_pdg else 0
            dominant_class = max(by_class, key=by_class.get) if by_class else ""
            row.update({
                "captured_energy_gev": captured_default,
                "captured_fraction": captured_default / max(total_prop, 1.0e-300),
                "captured_count": captured_count,
                "dominant_pdg": dominant_pdg,
                "dominant_pdg_label": PDG_LABELS.get(dominant_pdg, f"pdg{dominant_pdg}") if dominant_pdg else "",
                "dominant_class": dominant_class,
                "energy_by_class_json": json.dumps(energy_by_class, sort_keys=True),
                "captured_energy_by_pdg_json": json.dumps({str(k): v for k, v in sorted(by_pdg.items())}, sort_keys=True),
            })
            rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]], cones: list[float], mean_dir: tuple[float, float, float], energy_by_class: dict[str, float]) -> None:
    total = float(rows[0]["propagable_energy_gev"]) if rows else 0.0
    best_by_cone = {
        cone: max(rows, key=lambda row: float(row[f"captured_energy_cone_{cone:g}_deg"])) if rows else {}
        for cone in cones
    }
    lines = [
        "# Packet Observer Angular Scan",
        "",
        "Phase 5.4 angular anisotropy diagnostic for effective-straight-line escaping packets.",
        "This is not physical luminosity and not Kerr packet ray tracing.",
        "",
        f"- observer_count: `{len(rows)}`",
        f"- propagable_weighted_energy_gev: `{total:.12g}`",
        f"- mean_weighted_direction: `{list(mean_dir)}`",
        "",
        "## Energy By Class",
        "",
        "| Class | Weighted energy [GeV] |",
        "|---|---:|",
    ]
    for cls, energy in sorted(energy_by_class.items()):
        lines.append(f"| {cls} | {energy:.12g} |")
    lines.extend(["", "## Best Observers By Cone", "", "| Cone half-angle [deg] | theta [deg] | phi [deg] | captured energy [GeV] | fraction | captured packets | dominant PDG |", "|---:|---:|---:|---:|---:|---:|---|"])
    for cone in cones:
        row = best_by_cone[cone]
        lines.append(
            f"| {cone:g} | {float(row['theta_obs_deg']):.6g} | {float(row['phi_obs_deg']):.6g} | "
            f"{float(row[f'captured_energy_cone_{cone:g}_deg']):.12g} | "
            f"{float(row[f'captured_fraction_cone_{cone:g}_deg']):.12g} | "
            f"{int(row[f'captured_count_cone_{cone:g}_deg'])} | {row[f'dominant_pdg_label_cone_{cone:g}_deg']} |"
        )
    lines.extend([
        "",
        "The scan maps geometric anisotropy of escaped packet energy. It is a",
        "diagnostic for choosing observer directions before a future Kerr packet",
        "launcher exists; it should not be interpreted as observed luminosity.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]], cones: list[float]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    default_cone = cones[0]
    theta = [float(row["theta_obs_deg"]) for row in rows]
    phi = [float(row["phi_obs_deg"]) for row in rows]
    energy = [float(row[f"captured_energy_cone_{default_cone:g}_deg"]) for row in rows]
    lon = [math.radians(p if p <= 180.0 else p - 360.0) for p in phi]
    lat = [math.radians(90.0 - t) for t in theta]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), subplot_kw={"projection": "mollweide"})
    if rows:
        sc = ax.scatter(lon, lat, c=energy, s=36, cmap="magma")
        fig.colorbar(sc, ax=ax, label=f"captured energy, cone {default_cone:g} deg [GeV]")
    ax.grid(True, alpha=0.35)
    ax.set_title("Observer scan sky map")
    fig.tight_layout()
    fig.savefig(plots / "packet_observer_scan_sky_map.png", dpi=180)
    plt.close(fig)

    by_theta: dict[float, list[float]] = defaultdict(list)
    by_phi: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_theta[float(row["theta_obs_deg"])].append(float(row[f"captured_energy_cone_{default_cone:g}_deg"]))
        by_phi[float(row["phi_obs_deg"])].append(float(row[f"captured_energy_cone_{default_cone:g}_deg"]))
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(sorted(by_theta), [max(by_theta[t]) for t in sorted(by_theta)], marker="o")
    ax.set_xlabel("theta observer [deg]")
    ax.set_ylabel("max captured energy [GeV]")
    ax.set_title("Inclination profile")
    fig.tight_layout()
    fig.savefig(plots / "packet_observer_scan_inclination_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(sorted(by_phi), [max(by_phi[p]) for p in sorted(by_phi)], marker="o", color="#f58518")
    ax.set_xlabel("phi observer [deg]")
    ax.set_ylabel("max captured energy [GeV]")
    ax.set_title("Azimuth profile")
    fig.tight_layout()
    fig.savefig(plots / "packet_observer_scan_azimuth_profile.png", dpi=180)
    plt.close(fig)

    top = sorted(rows, key=lambda row: float(row[f"captured_energy_cone_{default_cone:g}_deg"]), reverse=True)[:10]
    labels = [f"{float(row['theta_obs_deg']):.0f}/{float(row['phi_obs_deg']):.0f}" for row in top]
    vals = [float(row[f"captured_energy_cone_{default_cone:g}_deg"]) for row in top]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(labels, vals, color="#54a24b")
    ax.set_xlabel("theta/phi [deg]")
    ax.set_ylabel("captured energy [GeV]")
    ax.set_title("Top observer directions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots / "packet_observer_scan_top_observers.png", dpi=180)
    plt.close(fig)


def parse_values(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--theta-step-deg", type=float, default=15.0)
    parser.add_argument("--phi-step-deg", type=float, default=15.0)
    parser.add_argument("--cone-deg", type=float, default=45.0)
    parser.add_argument("--cones-deg", default="", help="Optional comma/space list of cone half-angles. Overrides --cone-deg.")
    args = parser.parse_args()

    cones = parse_values(args.cones_deg) if args.cones_deg.strip() else [args.cone_deg]
    cones = sorted(set(cones))
    theta_values = [round(v, 10) for v in frange(0.0, 180.0, args.theta_step_deg, include_stop=True)]
    phi_values = [round(v, 10) for v in frange(0.0, 360.0, args.phi_step_deg, include_stop=False)]
    packets, energy_by_class, mean_dir = load_propagable(args.packets, args.classification)
    rows = scan(packets, energy_by_class, mean_dir, theta_values, phi_values, cones)
    fields = [
        "theta_obs_deg",
        "phi_obs_deg",
        "axis_x",
        "axis_y",
        "axis_z",
        "propagable_energy_gev",
        "captured_energy_gev",
        "captured_fraction",
        "captured_count",
        "dominant_pdg",
        "dominant_pdg_label",
        "dominant_class",
        "angle_to_mean_direction_deg",
        "energy_by_class_json",
        "captured_energy_by_pdg_json",
    ]
    for cone in cones:
        fields.extend([
            f"captured_energy_cone_{cone:g}_deg",
            f"captured_fraction_cone_{cone:g}_deg",
            f"captured_count_cone_{cone:g}_deg",
            f"dominant_pdg_cone_{cone:g}_deg",
            f"dominant_pdg_label_cone_{cone:g}_deg",
            f"dominant_class_cone_{cone:g}_deg",
            f"captured_energy_by_pdg_cone_{cone:g}_deg_json",
        ])
    output = args.output_dir
    write_csv(output / "packet_observer_scan.csv", rows, fields)
    write_summary(output / "packet_observer_scan_summary.md", rows, cones, mean_dir, energy_by_class)
    make_plots(output, rows, cones)
    best = max(rows, key=lambda row: float(row[f"captured_energy_cone_{cones[0]:g}_deg"])) if rows else {}
    print(json.dumps({
        "observer_count": len(rows),
        "cones_deg": cones,
        "propagable_energy_gev": sum(float(row["weighted_energy_gev"]) for row in packets),
        "mean_weighted_direction": list(mean_dir),
        "best_default_cone": best,
    }, indent=2, sort_keys=True))
    return 0


def frange(start: float, stop: float, step: float, include_stop: bool) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    values = []
    value = start
    eps = step * 1.0e-9
    while value < stop + (eps if include_stop else -eps):
        values.append(value)
        value += step
    if include_stop and abs(values[-1] - stop) > eps:
        values.append(stop)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
