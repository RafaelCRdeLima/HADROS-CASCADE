#!/usr/bin/env python3
"""Scan observers for experimental Kerr/ZAMO null escaping packets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from propagate_kerr_null_packets import (
    DEFAULT_ALLOWED,
    finite,
    normalize,
    pair_packets_with_classes,
    read_csv,
    read_jsonl,
    zamo_tetrad_diagnostics,
)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction_from_angles(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, sum(a[i] * b[i] for i in range(3))))
    return math.degrees(math.acos(dot))


def frange(start: float, stop: float, step: float, include_stop: bool) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    values = []
    value = start
    eps = step * 1.0e-9
    while value < stop + (eps if include_stop else -eps):
        values.append(round(value, 10))
        value += step
    if include_stop and abs(values[-1] - stop) > eps:
        values.append(stop)
    return values


def parse_values(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item.strip()]


def load_kerr_packets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float], tuple[float, float, float], dict[str, Any]]:
    allowed = set(DEFAULT_ALLOWED)
    if args.include_marginal:
        allowed.add("MARGINAL_ULTRARELATIVISTIC")
    paired = pair_packets_with_classes(read_jsonl(args.packets), read_csv(args.classification))
    packets: list[dict[str, Any]] = []
    energy_by_class: dict[str, float] = defaultdict(float)
    skipped_by_class: dict[str, float] = defaultdict(float)
    sx = sy = sz = 0.0
    max_abs_norm = 0.0
    min_zamo_energy = math.inf
    theta_clamp_count = 0
    status_changes = 0
    for packet, cls_row in paired:
        cls = str(cls_row.get("classification", "UNKNOWN_CLASSIFICATION"))
        energy = finite(cls_row.get("weighted_energy_gev"), finite(packet.get("weighted_energy_gev"), finite(packet.get("energy_gev"))))
        energy_by_class[cls] += energy
        direction = normalize((finite(packet.get("px_gev")), finite(packet.get("py_gev")), finite(packet.get("pz_gev"))))
        if cls not in allowed:
            skipped_by_class[cls] += energy
            continue
        if direction is None:
            skipped_by_class["FAILED_DIRECTION"] += energy
            continue
        pos = (finite(packet.get("x")), finite(packet.get("y")), finite(packet.get("z")))
        diag = {}
        if args.kerr_init_mode == "zamo_tetrad":
            diag = zamo_tetrad_diagnostics(pos, direction, args)
            if diag.get("theta_was_clamped"):
                theta_clamp_count += 1
            if diag.get("tetrad_status") != "OK":
                skipped_by_class[f"TETRAD_{diag.get('tetrad_status', 'BAD')}"] += energy
                continue
            max_abs_norm = max(max_abs_norm, abs(finite(diag.get("null_norm"))))
            min_zamo_energy = min(min_zamo_energy, finite(diag.get("zamo_energy"), math.inf))
        pdg = int(packet.get("pdg_id", 0))
        row = {
            "event_id": int(packet.get("event_id", 0)),
            "pdg_id": pdg,
            "particle_label": str(packet.get("particle_label", PDG_LABELS.get(pdg, f"pdg{pdg}"))),
            "classification": cls,
            "weighted_energy_gev": energy,
            "dir_x": direction[0],
            "dir_y": direction[1],
            "dir_z": direction[2],
            "theta_was_clamped": bool(diag.get("theta_was_clamped", False)),
            "null_norm": finite(diag.get("null_norm"), 0.0),
            "zamo_energy": finite(diag.get("zamo_energy"), 1.0),
            "tetrad_status": diag.get("tetrad_status", "NOT_USED"),
        }
        packets.append(row)
        sx += direction[0] * energy
        sy += direction[1] * energy
        sz += direction[2] * energy
    mean = normalize((sx, sy, sz)) or (0.0, 0.0, 1.0)
    meta = {
        "energy_by_class": dict(energy_by_class),
        "skipped_energy_by_class": dict(skipped_by_class),
        "max_abs_null_norm": max_abs_norm,
        "min_zamo_energy": min_zamo_energy if math.isfinite(min_zamo_energy) else 0.0,
        "theta_clamp_count": theta_clamp_count,
        "status_changes": status_changes,
    }
    return packets, dict(energy_by_class), mean, meta


def straight_lookup(path: Path) -> dict[tuple[float, float], dict[str, str]]:
    rows = read_csv(path) if path.exists() else []
    return {
        (finite(row.get("theta_obs_deg")), finite(row.get("phi_obs_deg"))): row
        for row in rows
    }


def scan(
    packets: list[dict[str, Any]],
    energy_by_class: dict[str, float],
    mean_dir: tuple[float, float, float],
    theta_values: list[float],
    phi_values: list[float],
    cones: list[float],
    straight: dict[tuple[float, float], dict[str, str]],
) -> list[dict[str, Any]]:
    total_prop = sum(float(row["weighted_energy_gev"]) for row in packets)
    rows: list[dict[str, Any]] = []
    for theta in theta_values:
        for phi in phi_values:
            axis = direction_from_angles(theta, phi)
            angles = [(packet, angle_deg((float(packet["dir_x"]), float(packet["dir_y"]), float(packet["dir_z"])), axis)) for packet in packets]
            row: dict[str, Any] = {
                "theta_obs_deg": theta,
                "phi_obs_deg": phi,
                "axis_x": axis[0],
                "axis_y": axis[1],
                "axis_z": axis[2],
                "kerr_init_mode": "zamo_tetrad",
                "propagable_energy_gev": total_prop,
                "angle_to_mean_direction_deg": angle_deg(axis, mean_dir),
                "energy_by_class_json": json.dumps(energy_by_class, sort_keys=True),
            }
            for cone in cones:
                selected = [(packet, ang) for packet, ang in angles if ang <= cone]
                energy = sum(float(packet["weighted_energy_gev"]) for packet, _ in selected)
                by_pdg: dict[int, float] = defaultdict(float)
                by_class: dict[str, float] = defaultdict(float)
                for packet, _ in selected:
                    by_pdg[int(packet["pdg_id"])] += float(packet["weighted_energy_gev"])
                    by_class[str(packet["classification"])] += float(packet["weighted_energy_gev"])
                dominant_pdg = max(by_pdg, key=by_pdg.get) if by_pdg else 0
                dominant_class = max(by_class, key=by_class.get) if by_class else ""
                straight_row = straight.get((theta, phi), {})
                straight_energy = finite(straight_row.get(f"captured_energy_cone_{cone:g}_deg"), math.nan)
                if not math.isfinite(straight_energy):
                    straight_energy = finite(straight_row.get("captured_energy_gev"), 0.0) if cone == cones[0] else 0.0
                delta = energy - straight_energy
                rel = delta / max(abs(straight_energy), 1.0e-300) if straight_row else 0.0
                row[f"captured_energy_cone_{cone:g}_deg"] = energy
                row[f"captured_fraction_cone_{cone:g}_deg"] = energy / max(total_prop, 1.0e-300)
                row[f"captured_count_cone_{cone:g}_deg"] = len(selected)
                row[f"dominant_pdg_cone_{cone:g}_deg"] = dominant_pdg
                row[f"dominant_pdg_label_cone_{cone:g}_deg"] = PDG_LABELS.get(dominant_pdg, f"pdg{dominant_pdg}") if dominant_pdg else ""
                row[f"dominant_class_cone_{cone:g}_deg"] = dominant_class
                row[f"straight_energy_cone_{cone:g}_deg"] = straight_energy if straight_row else ""
                row[f"delta_vs_straight_cone_{cone:g}_deg"] = delta if straight_row else ""
                row[f"relative_delta_vs_straight_cone_{cone:g}_deg"] = rel if straight_row else ""
                row[f"captured_energy_by_pdg_cone_{cone:g}_deg_json"] = json.dumps({str(k): v for k, v in sorted(by_pdg.items())}, sort_keys=True)
                if cone == cones[0]:
                    row["captured_energy_gev"] = energy
                    row["captured_fraction"] = energy / max(total_prop, 1.0e-300)
                    row["captured_count"] = len(selected)
                    row["dominant_pdg"] = dominant_pdg
                    row["dominant_pdg_label"] = row[f"dominant_pdg_label_cone_{cone:g}_deg"]
                    row["dominant_class"] = dominant_class
                    row["delta_vs_straight"] = delta if straight_row else ""
                    row["relative_delta_vs_straight"] = rel if straight_row else ""
            rows.append(row)
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]], cones: list[float], mean_dir: tuple[float, float, float], meta: dict[str, Any], straight_available: bool) -> None:
    total = float(rows[0]["propagable_energy_gev"]) if rows else 0.0
    lines = [
        "# Kerr/ZAMO Packet Observer Scan",
        "",
        "Phase 6.2 angular scan with experimental `zamo_tetrad` initialization.",
        "This is not observed luminosity and does not include massive geodesics.",
        "",
        f"- observer_count: `{len(rows)}`",
        f"- propagable_weighted_energy_gev: `{total:.12g}`",
        f"- mean_weighted_direction: `{list(mean_dir)}`",
        f"- max_abs_null_norm: `{meta['max_abs_null_norm']:.12g}`",
        f"- min_zamo_energy: `{meta['min_zamo_energy']:.12g}`",
        f"- theta_clamp_count: `{meta['theta_clamp_count']}`",
        f"- straight_line_comparison_available: `{straight_available}`",
        "",
        "## Best Observers By Cone",
        "",
        "| Cone half-angle [deg] | theta [deg] | phi [deg] | captured energy [GeV] | fraction | delta vs straight [GeV] | dominant PDG |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cone in cones:
        key = f"captured_energy_cone_{cone:g}_deg"
        best = max(rows, key=lambda row: float(row[key])) if rows else {}
        lines.append(
            f"| {cone:g} | {float(best['theta_obs_deg']):.6g} | {float(best['phi_obs_deg']):.6g} | "
            f"{float(best[key]):.12g} | {float(best[f'captured_fraction_cone_{cone:g}_deg']):.12g} | "
            f"{finite(best.get(f'delta_vs_straight_cone_{cone:g}_deg')):.12g} | "
            f"{best[f'dominant_pdg_label_cone_{cone:g}_deg']} |"
        )
    lines.extend([
        "",
        "## Skipped Energy",
        "",
        "| Class/status | Weighted energy [GeV] |",
        "|---|---:|",
    ])
    for cls, energy in sorted(meta["skipped_energy_by_class"].items()):
        lines.append(f"| {cls} | {energy:.12g} |")
    lines.extend([
        "",
        "`zamo_tetrad` remains experimental. `theta_was_clamped` indicates packets",
        "near the Boyer-Lindquist axis where a small coordinate clamp was applied",
        "to avoid the coordinate singularity.",
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
        fig.colorbar(sc, ax=ax, label=f"Kerr/ZAMO captured energy, cone {default_cone:g} deg [GeV]")
    ax.grid(True, alpha=0.35)
    ax.set_title("Kerr/ZAMO observer scan sky map")
    fig.tight_layout()
    fig.savefig(plots / "kerr_packet_observer_scan_sky_map.png", dpi=180)
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
    ax.set_title("Kerr/ZAMO inclination profile")
    fig.tight_layout()
    fig.savefig(plots / "kerr_packet_observer_scan_inclination_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(sorted(by_phi), [max(by_phi[p]) for p in sorted(by_phi)], marker="o", color="#f58518")
    ax.set_xlabel("phi observer [deg]")
    ax.set_ylabel("max captured energy [GeV]")
    ax.set_title("Kerr/ZAMO azimuth profile")
    fig.tight_layout()
    fig.savefig(plots / "kerr_packet_observer_scan_azimuth_profile.png", dpi=180)
    plt.close(fig)

    deltas = [finite(row.get(f"delta_vs_straight_cone_{default_cone:g}_deg")) for row in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(energy, deltas, s=30, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Kerr/ZAMO captured energy [GeV]")
    ax.set_ylabel("delta vs straight [GeV]")
    ax.set_title("Kerr/ZAMO vs straight capture")
    fig.tight_layout()
    fig.savefig(plots / "kerr_vs_straight_observer_capture.png", dpi=180)
    plt.close(fig)

    best_rows = [max(rows, key=lambda row: float(row[f"captured_energy_cone_{cone:g}_deg"])) for cone in cones]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(cones, [float(row[f"captured_energy_cone_{cone:g}_deg"]) for cone, row in zip(cones, best_rows)], marker="o", color="#54a24b")
    ax.set_xlabel("cone half-angle [deg]")
    ax.set_ylabel("best captured energy [GeV]")
    ax.set_title("Kerr/ZAMO best observer by cone")
    fig.tight_layout()
    fig.savefig(plots / "kerr_observer_top_cones.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--straight-scan", type=Path, default=Path("output/cascade/packet_observer_scan.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--theta-step-deg", type=float, default=15.0)
    parser.add_argument("--phi-step-deg", type=float, default=15.0)
    parser.add_argument("--cones-deg", default="45")
    parser.add_argument("--kerr-init-mode", choices=["zamo_tetrad"], default="zamo_tetrad")
    parser.add_argument("--include-marginal", action="store_true")
    args = parser.parse_args()

    cones = sorted(set(parse_values(args.cones_deg)))
    theta_values = frange(0.0, 180.0, args.theta_step_deg, include_stop=True)
    phi_values = frange(0.0, 360.0, args.phi_step_deg, include_stop=False)
    packets, energy_by_class, mean_dir, meta = load_kerr_packets(args)
    straight = straight_lookup(args.straight_scan)
    rows = scan(packets, energy_by_class, mean_dir, theta_values, phi_values, cones, straight)
    fields = [
        "theta_obs_deg",
        "phi_obs_deg",
        "axis_x",
        "axis_y",
        "axis_z",
        "kerr_init_mode",
        "propagable_energy_gev",
        "captured_energy_gev",
        "captured_fraction",
        "captured_count",
        "dominant_pdg",
        "dominant_pdg_label",
        "dominant_class",
        "delta_vs_straight",
        "relative_delta_vs_straight",
        "angle_to_mean_direction_deg",
        "energy_by_class_json",
    ]
    for cone in cones:
        fields.extend([
            f"captured_energy_cone_{cone:g}_deg",
            f"captured_fraction_cone_{cone:g}_deg",
            f"captured_count_cone_{cone:g}_deg",
            f"dominant_pdg_cone_{cone:g}_deg",
            f"dominant_pdg_label_cone_{cone:g}_deg",
            f"dominant_class_cone_{cone:g}_deg",
            f"straight_energy_cone_{cone:g}_deg",
            f"delta_vs_straight_cone_{cone:g}_deg",
            f"relative_delta_vs_straight_cone_{cone:g}_deg",
            f"captured_energy_by_pdg_cone_{cone:g}_deg_json",
        ])
    output = args.output_dir
    write_csv(output / "kerr_packet_observer_scan.csv", rows, fields)
    write_summary(output / "kerr_packet_observer_scan_summary.md", rows, cones, mean_dir, meta, bool(straight))
    make_plots(output, rows, cones)
    best = max(rows, key=lambda row: float(row[f"captured_energy_cone_{cones[0]:g}_deg"])) if rows else {}
    print(json.dumps({
        "observer_count": len(rows),
        "cones_deg": cones,
        "propagable_energy_gev": sum(float(row["weighted_energy_gev"]) for row in packets),
        "mean_weighted_direction": list(mean_dir),
        "max_abs_null_norm": meta["max_abs_null_norm"],
        "min_zamo_energy": meta["min_zamo_energy"],
        "theta_clamp_count": meta["theta_clamp_count"],
        "best_default_cone": best,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
