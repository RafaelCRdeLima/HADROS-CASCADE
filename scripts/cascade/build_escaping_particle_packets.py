#!/usr/bin/env python3
"""Build conservative effective packets for escaped cascade energy."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_event_weights(path: Path | None, weight_column: str) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    weights: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                event_id = int(float(row.get("event_id", "0")))
                factor = float(row.get(weight_column, "nan"))
            except (TypeError, ValueError):
                continue
            weights[event_id] = {
                "factor": factor,
                "status": row.get("weight_status", ""),
                "column": weight_column,
            }
    return weights


def apply_event_weights(rows: list[dict[str, Any]], weights: dict[int, dict[str, Any]]) -> None:
    if not weights:
        return
    for row in rows:
        event_id = int(row.get("event_id", 0))
        info = weights.get(event_id)
        raw_weight = float(row.get("weight", 1.0))
        row["raw_weight"] = raw_weight
        if info is None or not math.isfinite(float(info.get("factor", math.nan))):
            row["dis_weight_status"] = "MISSING_OR_INVALID_EVENT_WEIGHT"
            row["dis_weight_factor"] = math.nan
            row["weight"] = 0.0
            continue
        factor = float(info["factor"])
        row["dis_weight_factor"] = factor
        row["dis_weight_status"] = str(info.get("status", ""))
        row["dis_weight_column"] = str(info.get("column", ""))
        row["weight"] = raw_weight * factor


def rg_cm_from_mbh(mbh_msun: float) -> float:
    return 6.67430e-8 * mbh_msun * 1.98847e33 / (2.99792458e10 * 2.99792458e10)


def load_mbh_msun(config_path: Path) -> float:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path)
    for section in parser.sections():
        if parser.has_option(section, "MBH_MSUN"):
            try:
                return float(parser.get(section, "MBH_MSUN"))
            except ValueError:
                pass
    return 2.0


def spherical_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0 or not math.isfinite(r):
        return 0.0, 0.0, 0.0
    return r, math.acos(max(-1.0, min(1.0, z / r))), math.atan2(y, x)


def load_points(path: Path | None, rg_cm: float) -> dict[int, dict[str, float | str]]:
    if path is None or not path.exists():
        return {}
    points = {}
    for row in read_jsonl(path):
        event_id = int(row.get("event_id", row.get("primary", {}).get("event_id", 0)))
        point = row.get("point") if isinstance(row.get("point"), dict) else row
        if {"x", "y", "z"}.issubset(point):
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            z = float(point.get("z", 0.0))
        elif {"x_cm", "y_cm", "z_cm"}.issubset(point):
            x = float(point.get("x_cm", 0.0)) / rg_cm
            y = float(point.get("y_cm", 0.0)) / rg_cm
            z = float(point.get("z_cm", 0.0)) / rg_cm
        else:
            continue
        r, theta, phi = spherical_from_xyz(x, y, z)
        points[event_id] = {
            "x": x, "y": y, "z": z, "r": r, "theta": theta, "phi": phi,
            "origin_status": "INTERACTION_POINT_POSITION",
            "region_label": str(point.get("region_label", point.get("region_class", ""))),
        }
    return points


def direction_bin(row: dict[str, Any]) -> str:
    px = float(row.get("px_gev", 0.0))
    py = float(row.get("py_gev", 0.0))
    pz = float(row.get("pz_gev", 0.0))
    labels = ["px+" if px >= 0 else "px-", "py+" if py >= 0 else "py-", "pz+" if pz >= 0 else "pz-"]
    return "_".join(labels)


def canonical_aggregation_mode(mode: str) -> str:
    aliases = {
        "aggregate-by-pdg": "pdg",
        "aggregate-by-event": "event",
        "aggregate-by-direction": "direction_octant",
    }
    return aliases.get(mode, mode)


def momentum_vector(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("px_gev", 0.0)),
        float(row.get("py_gev", 0.0)),
        float(row.get("pz_gev", 0.0)),
    )


def norm(vec: tuple[float, float, float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def unit_vector(vec: tuple[float, float, float]) -> tuple[float, float, float] | None:
    size = norm(vec)
    if not math.isfinite(size) or size <= 0.0:
        return None
    return tuple(v / size for v in vec)  # type: ignore[return-value]


def direction_angles_deg(row: dict[str, Any]) -> tuple[float, float] | None:
    direction = unit_vector(momentum_vector(row))
    if direction is None:
        return None
    theta = math.degrees(math.acos(max(-1.0, min(1.0, direction[2]))))
    phi = (math.degrees(math.atan2(direction[1], direction[0])) + 360.0) % 360.0
    return theta, phi


def angular_bin(row: dict[str, Any], bin_deg: float) -> tuple[int, int]:
    angles = direction_angles_deg(row)
    if angles is None or bin_deg <= 0.0:
        return -1, -1
    theta, phi = angles
    return int(math.floor(theta / bin_deg)), int(math.floor(phi / bin_deg))


def angular_distance_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    au = unit_vector(a)
    bu = unit_vector(b)
    if au is None or bu is None:
        return math.nan
    dot = max(-1.0, min(1.0, sum(au[i] * bu[i] for i in range(3))))
    return math.degrees(math.acos(dot))


def aggregation_key(row: dict[str, Any], mode: str, angular_bin_deg: float) -> tuple[Any, ...]:
    mode = canonical_aggregation_mode(mode)
    pdg = int(row.get("pdg_id", row.get("pdg", 0)))
    event_id = int(row.get("event_id", 0))
    origin_backend = str(row.get("origin_backend", row.get("origin", "unknown")))
    if mode == "event":
        return (event_id,)
    if mode == "event_pdg":
        return (event_id, pdg, origin_backend)
    if mode == "event_pdg_angle":
        theta_bin, phi_bin = angular_bin(row, angular_bin_deg)
        return (event_id, pdg, theta_bin, phi_bin, origin_backend)
    if mode == "direction_octant":
        return (direction_bin(row), origin_backend)
    return (pdg, origin_backend)


def row_has_position(row: dict[str, Any]) -> bool:
    return all(key in row for key in ["x", "y", "z"]) or all(key in row for key in ["r", "theta", "phi"])


def row_position(row: dict[str, Any]) -> dict[str, Any] | None:
    if all(key in row for key in ["x", "y", "z"]):
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        z = float(row.get("z", 0.0))
        r, theta, phi = spherical_from_xyz(x, y, z)
        return {"x": x, "y": y, "z": z, "r": r, "theta": theta, "phi": phi}
    if all(key in row for key in ["r", "theta", "phi"]):
        r = float(row.get("r", 0.0))
        theta = float(row.get("theta", 0.0))
        phi = float(row.get("phi", 0.0))
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        return {"x": x, "y": y, "z": z, "r": r, "theta": theta, "phi": phi}
    return None


def resolve_row_position(row: dict[str, Any], points: dict[int, dict[str, float | str]]) -> dict[str, Any]:
    event_id = int(row.get("event_id", 0))
    if str(row.get("origin_status", "")) == "GEANT4_EXIT_POSITION" and row_has_position(row):
        pos = row_position(row) or {}
        return {**pos, "origin_status": "GEANT4_EXIT_POSITION", "region_label": row.get("region_label", "")}
    if event_id in points:
        return dict(points[event_id])
    if row_has_position(row):
        pos = row_position(row) or {}
        status = str(row.get("origin_status", "SECONDARY_POSITION"))
        if status not in {"SYNTHETIC_TEST_POSITION", "SECONDARY_POSITION", "GEANT4_EXIT_POSITION"}:
            status = "SECONDARY_POSITION"
        return {**pos, "origin_status": status, "region_label": row.get("region_label", "")}
    return {"x": 0.0, "y": 0.0, "z": 0.0, "r": 0.0, "theta": 0.0, "phi": 0.0, "origin_status": "MISSING_POSITION", "region_label": row.get("region_label", "")}


def weighted_position(items: list[dict[str, Any]], points: dict[int, dict[str, float | str]], horizon: float) -> dict[str, Any]:
    sums = {key: 0.0 for key in ["x", "y", "z", "r", "theta", "phi"]}
    den = 0.0
    statuses: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    num = 0.0
    for row in items:
        pos = resolve_row_position(row, points)
        weight = float(row.get("energy_gev", 0.0)) * float(row.get("weight", 1.0))
        for key in sums:
            sums[key] += float(pos.get(key, 0.0)) * weight
        den += weight
        statuses[str(pos.get("origin_status", "MISSING_POSITION"))] += 1
        if pos.get("region_label"):
            labels[str(pos.get("region_label"))] += 1
    if den > 0.0:
        for key in sums:
            sums[key] /= den
    status = statuses.most_common(1)[0][0] if statuses else "MISSING_POSITION"
    theta_defaulted = status == "MISSING_POSITION"
    inside = sums["r"] <= horizon if status != "MISSING_POSITION" else False
    return {
        **sums,
        "origin_status": status,
        "inside_horizon": inside,
        "theta_was_defaulted": theta_defaulted,
        "region_label": labels.most_common(1)[0][0] if labels else "",
    }


def packet_direction_metadata(items: list[dict[str, Any]], angular_bin_deg: float) -> dict[str, Any]:
    px = sum(float(row.get("px_gev", 0.0)) for row in items)
    py = sum(float(row.get("py_gev", 0.0)) for row in items)
    pz = sum(float(row.get("pz_gev", 0.0)) for row in items)
    total = (px, py, pz)
    total_norm = norm(total)
    energy = sum(float(row.get("energy_gev", 0.0)) for row in items)
    cancelled = total_norm <= max(1.0e-12 * max(energy, 1.0), 1.0e-30)
    if cancelled:
        theta_dir = math.nan
        phi_dir = math.nan
        theta_bin = -1
        phi_bin = -1
        spread = math.nan
    else:
        unit = unit_vector(total) or (1.0, 0.0, 0.0)
        theta_dir = math.degrees(math.acos(max(-1.0, min(1.0, unit[2]))))
        phi_dir = (math.degrees(math.atan2(unit[1], unit[0])) + 360.0) % 360.0
        theta_bin = int(math.floor(theta_dir / angular_bin_deg)) if angular_bin_deg > 0.0 else -1
        phi_bin = int(math.floor(phi_dir / angular_bin_deg)) if angular_bin_deg > 0.0 else -1
        distances = [angular_distance_deg(momentum_vector(row), total) for row in items]
        finite_distances = [value for value in distances if math.isfinite(value)]
        spread = max(finite_distances, default=0.0)
    return {
        "theta_dir_deg": theta_dir,
        "phi_dir_deg": phi_dir,
        "theta_bin": theta_bin,
        "phi_bin": phi_bin,
        "n_particles_in_packet": len(items),
        "angular_spread_deg": spread,
        "direction_status": "MOMENTUM_CANCELLED" if cancelled else "OK",
    }


def make_packet(
    items: list[dict[str, Any]],
    mode: str,
    points: dict[int, dict[str, float | str]],
    horizon: float,
    angular_bin_deg: float,
) -> dict[str, Any]:
    pdg_counts = Counter(int(row.get("pdg_id", row.get("pdg", 0))) for row in items)
    origin_counts = Counter(str(row.get("origin_backend", row.get("origin", "unknown"))) for row in items)
    dominant_pdg = pdg_counts.most_common(1)[0][0]
    dominant_origin = origin_counts.most_common(1)[0][0]
    event_ids = {int(row.get("event_id", 0)) for row in items}
    packet_event_id = next(iter(event_ids)) if len(event_ids) == 1 else 0
    energy = sum(float(row.get("energy_gev", 0.0)) for row in items)
    weight_sum = sum(float(row.get("weight", 1.0)) for row in items)
    weighted_energy = sum(float(row.get("energy_gev", 0.0)) * float(row.get("weight", 1.0)) for row in items)
    pos = weighted_position(items, points, horizon)
    direction_meta = packet_direction_metadata(items, angular_bin_deg)
    return {
        "event_id": packet_event_id,
        "pdg_id": dominant_pdg,
        "particle_label": PDG_LABELS.get(dominant_pdg, f"pdg{dominant_pdg}"),
        "energy_gev": energy,
        "weighted_energy_gev": weighted_energy,
        "px_gev": sum(float(row.get("px_gev", 0.0)) for row in items),
        "py_gev": sum(float(row.get("py_gev", 0.0)) for row in items),
        "pz_gev": sum(float(row.get("pz_gev", 0.0)) for row in items),
        "weight": weight_sum,
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
        "r": pos["r"],
        "theta": pos["theta"],
        "phi": pos["phi"],
        "origin_status": pos["origin_status"],
        "inside_horizon": bool(pos["inside_horizon"]),
        "theta_was_defaulted": bool(pos["theta_was_defaulted"]),
        "region_label": pos["region_label"],
        "origin_backend": dominant_origin if len(origin_counts) == 1 else "mixed",
        "aggregation_mode": mode,
        "source_particle_count": len(items),
        "angular_bin_deg": angular_bin_deg if canonical_aggregation_mode(mode) == "event_pdg_angle" else "",
        **direction_meta,
        "dominant_pdg_fraction": pdg_counts[dominant_pdg] / max(len(items), 1),
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ["energy_gev", "px_gev", "py_gev", "pz_gev", "weight"]:
            value = float(row.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"non-finite value for {key}: {row}")
        if float(row.get("energy_gev", 0.0)) < 0.0:
            raise ValueError(f"negative escaped energy: {row}")


def write_outputs(output_dir: Path, packets: list[dict[str, Any]], input_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "escaping_particle_packets.jsonl"
    csv_path = output_dir / "escaping_particle_packets.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, sort_keys=True) + "\n")
    fields = list(packets[0].keys()) if packets else [
        "event_id", "pdg_id", "energy_gev", "weighted_energy_gev", "px_gev", "py_gev", "pz_gev", "weight"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(packets)

    input_energy = sum(float(row.get("energy_gev", 0.0)) for row in input_rows)
    packet_energy = sum(float(row.get("energy_gev", 0.0)) for row in packets)
    input_weight = sum(float(row.get("weight", 1.0)) for row in input_rows)
    packet_weight = sum(float(row.get("weight", 0.0)) for row in packets)
    input_weighted = sum(float(row.get("energy_gev", 0.0)) * float(row.get("weight", 1.0)) for row in input_rows)
    packet_weighted = sum(float(row.get("weighted_energy_gev", 0.0)) for row in packets)
    closure_error = abs(packet_energy - input_energy) / max(input_energy, 1.0e-300)
    weighted_closure_error = abs(packet_weighted - input_weighted) / max(input_weighted, 1.0e-300)
    origin_counts = Counter(str(packet.get("origin_status", "MISSING_POSITION")) for packet in packets)
    inside_horizon_count = sum(1 for packet in packets if bool(packet.get("inside_horizon", False)))
    theta_default_count = sum(1 for packet in packets if bool(packet.get("theta_was_defaulted", False)))

    event_energy_in: dict[int, float] = defaultdict(float)
    for row in input_rows:
        event_energy_in[int(row.get("event_id", 0))] += float(row.get("energy_gev", 0.0))
    event_closure = {
        event_id: 0.0 for event_id in sorted(event_energy_in)
    }
    if canonical_aggregation_mode(args.aggregation_mode) == "event":
        packet_by_event: dict[int, float] = defaultdict(float)
        for packet in packets:
            packet_by_event[int(packet["event_id"])] += float(packet["energy_gev"])
        event_closure = {
            event_id: abs(packet_by_event[event_id] - energy) / max(energy, 1.0e-300)
            for event_id, energy in sorted(event_energy_in.items())
        }

    summary_path = output_dir / "escaping_particle_packets_summary.md"
    lines = [
        "# Escaping Particle Packets",
        "",
        "Effective conservative packets for escaped cascade energy.",
        "These are not individual particles and no propagation is performed in Phase 5.",
        "",
        f"- input: `{args.input}`",
        f"- sample_label: `{args.sample_label}`",
        f"- event_weights: `{args.event_weights}`",
        f"- weight_column: `{args.weight_column}`",
        f"- aggregation_mode: `{args.aggregation_mode}`",
        f"- angular_bin_deg: `{args.angular_bin_deg}`",
        f"- input_particles: `{len(input_rows)}`",
        f"- packets: `{len(packets)}`",
        f"- input_escaped_energy_gev: `{input_energy:.12g}`",
        f"- packet_energy_gev: `{packet_energy:.12g}`",
        f"- input_weighted_escaped_energy_gev: `{input_weighted:.12g}`",
        f"- packet_weighted_energy_gev: `{packet_weighted:.12g}`",
        f"- input_weight_sum: `{input_weight:.12g}`",
        f"- packet_weight_sum: `{packet_weight:.12g}`",
        f"- closure_error: `{closure_error:.12e}`",
        f"- weighted_closure_error: `{weighted_closure_error:.12e}`",
        f"- interaction_points: `{args.interaction_points}`",
        f"- require_physical_interaction_points: `{bool(args.require_physical_interaction_points)}`",
        f"- inside_horizon_packets: `{inside_horizon_count}`",
        f"- theta_defaulted_packets: `{theta_default_count}`",
        "",
        "## Origin Status",
        "",
        "| origin_status | packets |",
        "|---|---:|",
        *[f"| {status} | {count} |" for status, count in sorted(origin_counts.items())],
        "",
        "Packets represent total escaped energy, total escaped momentum, and dominant species.",
    ]
    if canonical_aggregation_mode(args.aggregation_mode) != "event":
        lines.append("Per-event closure is only exact in aggregate-by-event mode; global closure is used here.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_plots(output_dir, packets)
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "input_escaped_energy_gev": input_energy,
        "packet_energy_gev": packet_energy,
        "input_weighted_escaped_energy_gev": input_weighted,
        "packet_weighted_energy_gev": packet_weighted,
        "closure_error": closure_error,
        "weighted_closure_error": weighted_closure_error,
        "event_closure_error": event_closure,
        "origin_status_counts": dict(sorted(origin_counts.items())),
        "inside_horizon_packets": inside_horizon_count,
        "theta_defaulted_packets": theta_default_count,
    }


def make_plots(output_dir: Path, packets: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by_pdg_energy: dict[int, float] = defaultdict(float)
    by_pdg_weight: dict[int, float] = defaultdict(float)
    for packet in packets:
        pdg = int(packet["pdg_id"])
        by_pdg_energy[pdg] += float(packet["energy_gev"])
        by_pdg_weight[pdg] += float(packet["weight"])
    labels = [PDG_LABELS.get(pdg, str(pdg)) for pdg in sorted(by_pdg_energy)]
    energies = [by_pdg_energy[pdg] for pdg in sorted(by_pdg_energy)]
    weights = [by_pdg_weight[pdg] for pdg in sorted(by_pdg_energy)]
    for values, ylabel, filename in [
        (energies, "escaped packet energy [GeV]", "escaping_packet_energy_by_pdg.png"),
        (weights, "packet weight sum", "escaping_packet_weight_by_pdg.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)
    total = sum(energies)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if total > 0.0:
        ax.pie(energies, labels=labels, autopct="%1.1f%%")
    ax.set_title("escaping packet energy fraction")
    fig.tight_layout()
    fig.savefig(plots / "escaping_packet_energy_fraction.png", dpi=180)
    plt.close(fig)


def build_packets(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.input)
    extra_inputs = list(args.extra_input)
    if not extra_inputs and args.input.name == "geant4_escaped_particles.jsonl":
        sibling = args.input.with_name("geant4_unsupported_uhe_particles.jsonl")
        if sibling.exists():
            extra_inputs.append(sibling)
    for extra in extra_inputs:
        if extra.exists():
            rows.extend(read_jsonl(extra))
    event_weights = load_event_weights(args.event_weights, args.weight_column)
    apply_event_weights(rows, event_weights)
    validate_rows(rows)
    if args.interaction_points == Path("output/cascade/primary_interactions.jsonl") and not args.interaction_points.exists():
        sibling = args.input.with_name("interaction_points.jsonl")
        if sibling.exists():
            args.interaction_points = sibling
    mbh_msun = args.mbh_msun if args.mbh_msun is not None else load_mbh_msun(args.config)
    horizon = 1.0 + math.sqrt(max(1.0 - args.spin * args.spin, 0.0))
    points = load_points(args.interaction_points, rg_cm_from_mbh(mbh_msun))
    if args.require_physical_interaction_points and not points:
        raise RuntimeError(
            "physical interaction points are required, but no usable positions were found in "
            f"{args.interaction_points}"
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[aggregation_key(row, args.aggregation_mode, args.angular_bin_deg)].append(row)
    packets = [
        make_packet(items, args.aggregation_mode, points, horizon, args.angular_bin_deg)
        for _, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    return write_outputs(args.output_dir, packets, rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/cascade/geant4_escaped_particles.jsonl"))
    parser.add_argument("--extra-input", type=Path, action="append", default=[])
    parser.add_argument("--fallback-input", type=Path, default=Path("output/cascade/pythia_secondaries.jsonl"))
    parser.add_argument("--interaction-points", type=Path, default=Path("output/cascade/primary_interactions.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config.ini"))
    parser.add_argument("--mbh-msun", type=float, default=None)
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--require-physical-interaction-points", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--sample-label", default="")
    parser.add_argument("--event-weights", type=Path, default=None)
    parser.add_argument("--weight-column", default="weight_GBW")
    parser.add_argument(
        "--aggregation-mode",
        choices=[
            "pdg",
            "event_pdg",
            "event_pdg_angle",
            "aggregate-by-pdg",
            "aggregate-by-event",
            "aggregate-by-direction",
        ],
        default=None,
        help="Packet aggregation mode. Legacy aggregate-by-* values remain accepted.",
    )
    parser.add_argument("--angular-bin-deg", type=float, default=5.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--aggregate-by-pdg", dest="aggregation_mode", action="store_const", const="aggregate-by-pdg")
    group.add_argument("--aggregate-by-event", dest="aggregation_mode", action="store_const", const="aggregate-by-event")
    group.add_argument("--aggregate-by-direction", dest="aggregation_mode", action="store_const", const="aggregate-by-direction")
    parser.set_defaults(aggregation_mode="aggregate-by-pdg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        if args.fallback_input.exists():
            args.input = args.fallback_input
        else:
            print(f"missing input and fallback: {args.input}, {args.fallback_input}", file=sys.stderr)
            return 2
    result = build_packets(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
