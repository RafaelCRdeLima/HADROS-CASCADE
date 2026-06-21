#!/usr/bin/env python3
"""Experimental Kerr-null packet propagation audit.

Phase 6.0 connects escaping packets to an explicitly experimental Kerr-null
propagation path. The only implemented initialization mode is `flat_local`,
which uses the packet Cartesian momentum direction as a local null direction.
This is not a full physical tetrad initialization and not massive-particle
transport.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED = {"MASSLESS_NULL", "ULTRARELATIVISTIC_NULL_OK"}
THETA_EPS = 1.0e-6
ROOT = Path(__file__).resolve().parents[2]


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
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def packet_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row.get("event_id", 0)), int(row.get("pdg_id", 0)), f"{finite(row.get('energy_gev')):.15g}"


def pair_packets_with_classes(packets: list[dict[str, Any]], classes: list[dict[str, str]]) -> list[tuple[dict[str, Any], dict[str, str]]]:
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


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(v * v for v in vec))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return tuple(v / norm for v in vec)  # type: ignore[return-value]


def ray_sphere_distance(pos: tuple[float, float, float], direction: tuple[float, float, float], radius: float) -> float | None:
    b = sum(pos[i] * direction[i] for i in range(3))
    c = sum(p * p for p in pos) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    root = math.sqrt(max(disc, 0.0))
    positive = [t for t in (-b - root, -b + root) if t >= 0.0]
    return min(positive) if positive else None


def hits_horizon(pos: tuple[float, float, float], direction: tuple[float, float, float], horizon: float) -> bool:
    r2 = sum(p * p for p in pos)
    if math.sqrt(r2) <= horizon:
        return True
    approach = sum(pos[i] * direction[i] for i in range(3))
    if approach >= 0.0:
        return False
    closest2 = max(r2 - approach * approach, 0.0)
    return closest2 <= horizon * horizon


def spherical_from_cartesian(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pos
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        return 0.0, math.pi * 0.5, 0.0
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return r, theta, phi


def local_spherical_direction(
    pos: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    _, theta, phi = spherical_from_cartesian(pos)
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    e_r = (st * cp, st * sp, ct)
    e_theta = (ct * cp, ct * sp, -st)
    e_phi = (-sp, cp, 0.0)
    return (
        sum(direction[i] * e_r[i] for i in range(3)),
        sum(direction[i] * e_theta[i] for i in range(3)),
        sum(direction[i] * e_phi[i] for i in range(3)),
    )


def kerr_metric(spin: float, r: float, theta: float) -> list[list[float]]:
    sig = r * r + spin * spin * math.cos(theta) ** 2
    delta = r * r - 2.0 * r + spin * spin
    s2 = math.sin(theta) ** 2
    g = [[0.0 for _ in range(4)] for _ in range(4)]
    g[0][0] = -(1.0 - 2.0 * r / sig)
    g[0][3] = -2.0 * spin * r * s2 / sig
    g[3][0] = g[0][3]
    g[1][1] = sig / delta
    g[2][2] = sig
    g[3][3] = (r * r + spin * spin + 2.0 * spin * spin * r * s2 / sig) * s2
    return g


def kerr_delta(spin: float, r: float) -> float:
    return r * r - 2.0 * r + spin * spin


def kerr_lapse(spin: float, r: float, theta: float) -> float:
    sig = r * r + spin * spin * math.cos(theta) ** 2
    delta = kerr_delta(spin, r)
    big_a = (r * r + spin * spin) ** 2 - spin * spin * delta * math.sin(theta) ** 2
    return math.sqrt(sig * delta / big_a)


def kerr_omega(spin: float, r: float, theta: float) -> float:
    delta = kerr_delta(spin, r)
    big_a = (r * r + spin * spin) ** 2 - spin * spin * delta * math.sin(theta) ** 2
    return 2.0 * spin * r / big_a


def zamo_tetrad_diagnostics(
    pos: tuple[float, float, float],
    direction: tuple[float, float, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    r, theta_raw, phi = spherical_from_cartesian(pos)
    horizon = 1.0 + math.sqrt(max(1.0 - args.spin * args.spin, 0.0))
    theta = min(max(theta_raw, THETA_EPS), math.pi - THETA_EPS)
    theta_clamped = abs(theta - theta_raw) > 0.0
    if r <= horizon:
        return {
            "r_bl": r,
            "theta_bl": theta,
            "phi_bl": phi,
            "theta_was_clamped": theta_clamped,
            "tetrad_status": "INSIDE_HORIZON",
            "null_norm": math.nan,
            "zamo_energy": math.nan,
        }
    n_r, n_theta, n_phi = local_spherical_direction(pos, direction)
    n = normalize((n_r, n_theta, n_phi)) or (1.0, 0.0, 0.0)
    g = kerr_metric(args.spin, r, theta)
    alpha = kerr_lapse(args.spin, r, theta)
    omega = kerr_omega(args.spin, r, theta)
    p_contra = [
        1.0 / alpha,
        n[0] / math.sqrt(g[1][1]),
        n[1] / math.sqrt(g[2][2]),
        n[2] / math.sqrt(g[3][3]) + omega / alpha,
    ]
    p_cov = [sum(g[mu][nu] * p_contra[nu] for nu in range(4)) for mu in range(4)]
    null_norm = sum(g[mu][nu] * p_contra[mu] * p_contra[nu] for mu in range(4) for nu in range(4))
    zamo_energy = -(p_cov[0] + omega * p_cov[3]) / alpha
    return {
        "r_bl": r,
        "theta_bl": theta,
        "phi_bl": phi,
        "theta_was_clamped": theta_clamped,
        "tetrad_status": "OK" if math.isfinite(null_norm) and zamo_energy > 0.0 else "BAD_TETRAD",
        "null_norm": null_norm,
        "zamo_energy": zamo_energy,
        "p_t_cov": p_cov[0],
        "p_r_cov": p_cov[1],
        "p_theta_cov": p_cov[2],
        "p_phi_cov": p_cov[3],
    }


def axis_vector(name: str) -> tuple[float, float, float]:
    axes = {
        "x": (1.0, 0.0, 0.0),
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    return axes[name]


def camera_basis(axis: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = normalize((ref[1] * axis[2] - ref[2] * axis[1], ref[2] * axis[0] - ref[0] * axis[2], ref[0] * axis[1] - ref[1] * axis[0]))
    if u is None:
        u = (1.0, 0.0, 0.0)
    v = normalize((axis[1] * u[2] - axis[2] * u[1], axis[2] * u[0] - axis[0] * u[2], axis[0] * u[1] - axis[1] * u[0]))
    return u, v if v is not None else (0.0, 1.0, 0.0)


def observer_pixel(direction: tuple[float, float, float], nx: int, ny: int, fov_deg: float, observer_axis: str) -> tuple[int, int] | None:
    axis = axis_vector(observer_axis)
    cosang = sum(direction[i] * axis[i] for i in range(3))
    if cosang <= 0.0:
        return None
    scale = math.tan(0.5 * math.radians(fov_deg))
    u_axis, v_axis = camera_basis(axis)
    u = sum(direction[i] * u_axis[i] for i in range(3)) / max(cosang, 1.0e-300)
    v = sum(direction[i] * v_axis[i] for i in range(3)) / max(cosang, 1.0e-300)
    if abs(u) > scale or abs(v) > scale:
        return None
    i = max(0, min(nx - 1, int((u / scale + 1.0) * 0.5 * nx)))
    j = max(0, min(ny - 1, int((v / scale + 1.0) * 0.5 * ny)))
    return i, j


def propagate_one(packet: dict[str, Any], classification: str, args: argparse.Namespace, backend: str) -> dict[str, Any]:
    energy = finite(packet.get("energy_gev"))
    weight = finite(packet.get("weight"), 1.0)
    weighted = finite(packet.get("weighted_energy_gev"), energy * weight)
    origin_status = str(packet.get("origin_status", ""))
    direction_status = str(packet.get("direction_status", "OK"))
    allowed = set(DEFAULT_ALLOWED)
    if args.include_marginal:
        allowed.add("MARGINAL_ULTRARELATIVISTIC")
    pos = (finite(packet.get("x")), finite(packet.get("y")), finite(packet.get("z")))
    r_packet = finite(packet.get("r"), math.sqrt(sum(p * p for p in pos)))
    direction = normalize((finite(packet.get("px_gev")), finite(packet.get("py_gev")), finite(packet.get("pz_gev"))))
    row: dict[str, Any] = {
        "event_id": int(packet.get("event_id", 0)),
        "pdg_id": int(packet.get("pdg_id", 0)),
        "classification": classification,
        "energy_gev": energy,
        "weighted_energy_gev": weighted,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "r": r_packet,
        "theta": finite(packet.get("theta"), math.nan),
        "phi": finite(packet.get("phi"), math.nan),
        "origin_status": origin_status,
        "direction_status": direction_status,
        "inside_horizon": bool(packet.get("inside_horizon", False)),
        "theta_was_defaulted": bool(packet.get("theta_was_defaulted", False)),
        "dir_x": math.nan if direction is None else direction[0],
        "dir_y": math.nan if direction is None else direction[1],
        "dir_z": math.nan if direction is None else direction[2],
        "final_status": "FAILED_INTEGRATION",
        "failure_reason": "",
        "observer_pixel_i": "",
        "observer_pixel_j": "",
        "path_length": 0.0,
        "affine_steps": 0,
        "initial_step_size": args.step * args.initial_step_scale,
        "max_affine_step": args.max_affine_step,
        "final_x": pos[0],
        "final_y": pos[1],
        "final_z": pos[2],
        "redshift_factor": 1.0,
        "observed_energy_proxy_gev": energy,
        "weighted_observed_energy_proxy_gev": weighted,
        "normalize_null_momentum": bool(args.normalize_null_momentum),
        "normalization_factor": 1.0 / max(energy, 1.0e-300) if args.normalize_null_momentum else 1.0,
        "kerr_init_mode": args.kerr_init_mode,
        "r_bl": "",
        "theta_bl": "",
        "phi_bl": "",
        "theta_was_clamped": "",
        "tetrad_status": "",
        "null_norm": "",
        "zamo_energy": "",
        "p_t_cov": "",
        "p_r_cov": "",
        "p_theta_cov": "",
        "p_phi_cov": "",
        "backend": backend,
    }
    if classification not in allowed:
        row["final_status"] = "SKIPPED_CLASS"
        row["failure_reason"] = "class_not_selected"
        return row
    if direction_status == "MOMENTUM_CANCELLED":
        row["final_status"] = "SKIPPED_MOMENTUM_CANCELLED"
        row["failure_reason"] = "packet_total_momentum_cancelled"
        return row
    if origin_status == "MISSING_POSITION":
        row["final_status"] = "SKIPPED_MISSING_POSITION"
        row["failure_reason"] = "missing_physical_packet_origin"
        return row
    horizon = 1.0 + math.sqrt(max(1.0 - args.spin * args.spin, 0.0))
    if r_packet <= horizon:
        row["final_status"] = "SKIPPED_INSIDE_HORIZON"
        row["failure_reason"] = "packet_origin_inside_or_on_horizon"
        row["r_bl"] = r_packet
        row["tetrad_status"] = "INSIDE_HORIZON"
        return row
    if direction is None or energy < 0.0:
        row["failure_reason"] = "bad_direction_or_negative_energy"
        return row
    if args.kerr_init_mode == "zamo_tetrad":
        diag = zamo_tetrad_diagnostics(pos, direction, args)
        row.update(diag)
        if diag["tetrad_status"] != "OK":
            row["final_status"] = "FAILED_INTEGRATION"
            row["failure_reason"] = str(diag["tetrad_status"])
            return row
    if hits_horizon(pos, direction, args.horizon_radius):
        status = "HIT_HORIZON"
        distance = ray_sphere_distance(pos, direction, args.horizon_radius) or 0.0
    else:
        distance = ray_sphere_distance(pos, direction, args.domain_radius)
        if distance is None:
            status = "OUT_OF_RANGE"
            distance = 0.0
            row["failure_reason"] = "no_domain_intersection"
        else:
            pixel = observer_pixel(direction, args.nx, args.ny, args.fov_deg, args.observer_axis)
            status = "ESCAPED_TO_OBSERVER" if pixel is not None else "ESCAPED_DOMAIN"
            if pixel is not None:
                row["observer_pixel_i"], row["observer_pixel_j"] = pixel
    row["final_x"] = pos[0] + direction[0] * distance
    row["final_y"] = pos[1] + direction[1] * distance
    row["final_z"] = pos[2] + direction[2] * distance
    row["final_status"] = status
    row["path_length"] = distance
    effective_step = min(max(args.step * args.initial_step_scale, 1.0e-300), max(args.max_affine_step, 1.0e-300))
    row["affine_steps"] = int(distance / effective_step)
    return row


def summarize(rows: list[dict[str, Any]], straight_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    energy_by_status: dict[str, float] = defaultdict(float)
    count_by_status: Counter[str] = Counter()
    not_propagated: dict[str, float] = defaultdict(float)
    for row in rows:
        status = str(row["final_status"])
        count_by_status[status] += 1
        energy_by_status[status] += float(row["weighted_energy_gev"])
        if status == "SKIPPED_CLASS":
            not_propagated[str(row["classification"])] += float(row["weighted_energy_gev"])
    straight_by_key = {(row["event_id"], row["pdg_id"], f"{float(row['energy_gev']):.15g}"): row for row in straight_rows}
    changed = []
    for row in rows:
        other = straight_by_key.get((row["event_id"], row["pdg_id"], f"{float(row['energy_gev']):.15g}"))
        if other and other["final_status"] != row["final_status"]:
            changed.append({
                "event_id": row["event_id"],
                "pdg_id": row["pdg_id"],
                "energy_gev": row["energy_gev"],
                "straight_status": other["final_status"],
                "kerr_status": row["final_status"],
            })
    total = sum(float(row["weighted_energy_gev"]) for row in rows)
    selected = sum(float(row["weighted_energy_gev"]) for row in rows if row["final_status"] != "SKIPPED_CLASS")
    return {
        "backend": "experimental_kerr_null",
        "kerr_init_mode": args.kerr_init_mode,
        "init_mode_note": (
            "zamo_tetrad uses a local ZAMO/LNRF tetrad and validates g(p,p)=0; still experimental."
            if args.kerr_init_mode == "zamo_tetrad"
            else "flat_local uses packet Cartesian momentum as local null direction; not a full tetrad initialization."
        ),
        "total_weighted_energy_gev": total,
        "selected_weighted_energy_gev": selected,
        "selected_fraction": selected / max(total, 1.0e-300),
        "energy_by_status": dict(sorted(energy_by_status.items())),
        "count_by_status": dict(sorted(count_by_status.items())),
        "not_propagated_energy_by_class": dict(sorted(not_propagated.items())),
        "changed_status_packets": changed,
        "changed_status_count": len(changed),
        "observer_axis": args.observer_axis,
        "fov_deg": args.fov_deg,
        "max_abs_null_norm": max((abs(finite(row.get("null_norm"), 0.0)) for row in rows if row.get("tetrad_status") == "OK"), default=0.0),
        "min_zamo_energy": min((finite(row.get("zamo_energy"), math.inf) for row in rows if row.get("tetrad_status") == "OK"), default=math.inf),
    }


def make_image(rows: list[dict[str, Any]], nx: int, ny: int) -> Any:
    import numpy as np

    image = np.zeros((ny, nx), dtype=float)
    for row in rows:
        if row["final_status"] != "ESCAPED_TO_OBSERVER":
            continue
        if row["observer_pixel_i"] == "" or row["observer_pixel_j"] == "":
            continue
        image[int(row["observer_pixel_j"]), int(row["observer_pixel_i"])] += float(row["weighted_observed_energy_proxy_gev"])
    return image


def write_summary(path: Path, summary: dict[str, Any], straight_summary: dict[str, float]) -> None:
    lines = [
        "# Experimental Kerr Null Packet Propagation",
        "",
        (
            "Phase 6.1 tetrad-initialized audit output. This is experimental and not physical luminosity."
            if summary["kerr_init_mode"] == "zamo_tetrad"
            else "Phase 6.0 audit output. This is experimental and not physical luminosity."
        ),
        "",
        f"- backend: `{summary['backend']}`",
        f"- kerr_init_mode: `{summary['kerr_init_mode']}`",
        f"- init_mode_note: {summary['init_mode_note']}",
        f"- total weighted energy [GeV]: `{summary['total_weighted_energy_gev']:.12g}`",
        f"- selected weighted energy [GeV]: `{summary['selected_weighted_energy_gev']:.12g}`",
        f"- selected fraction: `{summary['selected_fraction']:.12g}`",
        f"- max_abs_null_norm: `{summary['max_abs_null_norm']:.12g}`",
        f"- min_zamo_energy: `{summary['min_zamo_energy']:.12g}`",
        f"- changed status packets vs straight-line: `{summary['changed_status_count']}`",
        "",
        "## Kerr Status Energy",
        "",
        "| Status | Count | Weighted energy [GeV] |",
        "|---|---:|---:|",
    ]
    for status, energy in summary["energy_by_status"].items():
        lines.append(f"| {status} | {summary['count_by_status'].get(status, 0)} | {energy:.12g} |")
    lines.extend(["", "## Straight-Line Status Energy", "", "| Status | Weighted energy [GeV] |", "|---|---:|"])
    for status, energy in sorted(straight_summary.items()):
        lines.append(f"| {status} | {energy:.12g} |")
    lines.extend([
        "",
        "## Scope",
        "",
        "- Only `MASSLESS_NULL` and `ULTRARELATIVISTIC_NULL_OK` are propagated by default.",
        "- `flat_local` is an initialization placeholder, not a complete local tetrad conversion.",
        "- `zamo_tetrad` is the first local tetrad initialization and remains experimental.",
        "- Massive geodesics are not implemented.",
        "- Packets are effective bundles, not individual particles.",
        "- This does not alter the default HADROS pipeline.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]], straight_rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    straight_energy = status_energy(straight_rows)
    kerr_energy = status_energy(rows)
    statuses = sorted(set(straight_energy) | set(kerr_energy))
    xs = range(len(statuses))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar([x - width / 2 for x in xs], [straight_energy.get(s, 0.0) for s in statuses], width=width, label="straight")
    ax.bar([x + width / 2 for x in xs], [kerr_energy.get(s, 0.0) for s in statuses], width=width, label="kerr_null")
    ax.set_xticks(list(xs), statuses, rotation=25)
    ax.set_ylabel("weighted energy [GeV]")
    ax.set_title("Packet status: Kerr-null vs straight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "kerr_vs_straight_packet_status.png", dpi=180)
    plt.close(fig)

    straight_by_key = {(str(row.get("event_id")), str(row.get("pdg_id")), f"{finite(row.get('energy_gev')):.15g}"): row for row in straight_rows}
    xs_energy = []
    ys_energy = []
    for row in rows:
        key = (str(row.get("event_id")), str(row.get("pdg_id")), f"{finite(row.get('energy_gev')):.15g}")
        if key not in straight_by_key:
            continue
        xs_energy.append(finite(straight_by_key[key].get("weighted_energy_gev")))
        ys_energy.append(finite(row.get("weighted_energy_gev")))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(xs_energy, ys_energy, s=50, color="#4c78a8")
    ax.set_xlabel("straight-line weighted energy [GeV]")
    ax.set_ylabel("kerr-null weighted energy [GeV]")
    ax.set_title("Packet energy comparison")
    fig.tight_layout()
    fig.savefig(plots / "kerr_vs_straight_packet_energy.png", dpi=180)
    plt.close(fig)


def make_flat_vs_zamo_plots(output_dir: Path, zamo_rows: list[dict[str, Any]], flat_rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    flat_energy = status_energy(flat_rows)
    zamo_energy = status_energy(zamo_rows)
    statuses = sorted(set(flat_energy) | set(zamo_energy))
    xs = range(len(statuses))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar([x - width / 2 for x in xs], [flat_energy.get(s, 0.0) for s in statuses], width=width, label="flat_local")
    ax.bar([x + width / 2 for x in xs], [zamo_energy.get(s, 0.0) for s in statuses], width=width, label="zamo_tetrad")
    ax.set_xticks(list(xs), statuses, rotation=25)
    ax.set_ylabel("weighted energy [GeV]")
    ax.set_title("Kerr packet status: flat vs ZAMO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "kerr_flat_vs_zamo_status.png", dpi=180)
    plt.close(fig)

    flat_by_key = {(str(row.get("event_id")), str(row.get("pdg_id")), f"{finite(row.get('energy_gev')):.15g}"): row for row in flat_rows}
    xs_energy = []
    ys_energy = []
    for row in zamo_rows:
        key = (str(row.get("event_id")), str(row.get("pdg_id")), f"{finite(row.get('energy_gev')):.15g}")
        if key not in flat_by_key:
            continue
        xs_energy.append(finite(flat_by_key[key].get("weighted_energy_gev")))
        ys_energy.append(finite(row.get("weighted_energy_gev")))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(xs_energy, ys_energy, s=50, color="#72b7b2")
    ax.set_xlabel("flat_local weighted energy [GeV]")
    ax.set_ylabel("zamo_tetrad weighted energy [GeV]")
    ax.set_title("Flat vs ZAMO packet energy")
    fig.tight_layout()
    fig.savefig(plots / "kerr_flat_vs_zamo_energy.png", dpi=180)
    plt.close(fig)


def status_energy(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows:
        out[str(row["final_status"])] += float(row["weighted_energy_gev"])
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--straight-line", type=Path, default=Path("output/cascade/null_propagated_packets.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--packet-propagation-backend", choices=["proxy_straight_line", "real_kerr_geodesic"], default="proxy_straight_line")
    parser.add_argument("--include-marginal", action="store_true")
    parser.add_argument("--kerr-init-mode", choices=["flat_local", "zamo_tetrad"], default="flat_local")
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--observer-axis", choices=["x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z"], default="+z")
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--domain-radius", type=float, default=200.0)
    parser.add_argument("--horizon-radius", type=float, default=1.2)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--normalize-null-momentum", action="store_true")
    parser.add_argument("--max-affine-step", type=float, default=0.05)
    parser.add_argument("--initial-step-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.packet_propagation_backend == "real_kerr_geodesic":
        exe = ROOT / "build" / "propagate_packets_real_kerr"
        if not exe.exists():
            subprocess.run(["make", "propagate_packets_real_kerr"], cwd=ROOT, check=True)
        command = [
            str(exe),
            "--packets", str(args.packets),
            "--classification", str(args.classification),
            "--output-dir", str(args.output_dir),
            "--kerr-init-mode", args.kerr_init_mode,
            "--spin", str(args.spin),
            "--domain-radius", str(args.domain_radius),
            "--step", str(args.step),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        return 0

    paired = pair_packets_with_classes(read_jsonl(args.packets), read_csv(args.classification))
    rows = [propagate_one(packet, str(cls.get("classification", "UNKNOWN_CLASSIFICATION")), args, "experimental_kerr_null") for packet, cls in paired]
    straight_rows = read_csv(args.straight_line)
    if not straight_rows:
        straight_rows = [propagate_one(packet, str(cls.get("classification", "UNKNOWN_CLASSIFICATION")), args, "effective_straight_line") for packet, cls in paired]
    summary = summarize(rows, straight_rows, args)
    output = args.output_dir
    fields = [
        "event_id", "pdg_id", "classification", "energy_gev", "weighted_energy_gev",
        "x", "y", "z", "r", "theta", "phi", "origin_status", "inside_horizon",
        "theta_was_defaulted", "direction_status", "dir_x", "dir_y", "dir_z", "final_status",
        "failure_reason",
        "observer_pixel_i", "observer_pixel_j", "path_length", "affine_steps",
        "initial_step_size", "max_affine_step", "final_x", "final_y", "final_z",
        "redshift_factor", "observed_energy_proxy_gev", "weighted_observed_energy_proxy_gev",
        "normalize_null_momentum", "normalization_factor",
        "r_bl", "theta_bl", "phi_bl", "theta_was_clamped", "tetrad_status",
        "null_norm", "zamo_energy", "p_t_cov", "p_r_cov", "p_theta_cov", "p_phi_cov",
        "kerr_init_mode", "backend",
    ]
    suffix = "_zamo" if args.kerr_init_mode == "zamo_tetrad" else ""
    write_jsonl(output / f"kerr_null_propagated_packets{suffix}.jsonl", rows)
    write_csv(output / f"kerr_null_propagated_packets{suffix}.csv", rows, fields)
    write_summary(output / f"kerr_null_packet_summary{suffix}.md", summary, status_energy(straight_rows))
    image = make_image(rows, args.nx, args.ny)
    import numpy as np

    np.savez(output / f"kerr_null_packet_camera_image{suffix}.npz", image=image, backend=np.array("experimental_kerr_null"), kerr_init_mode=np.array(args.kerr_init_mode))
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(image, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="weighted observed energy proxy [GeV]")
    ax.set_title("Experimental Kerr-null packet camera proxy")
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.tight_layout()
    fig.savefig(plots / f"kerr_null_packet_camera{suffix}.png", dpi=180)
    plt.close(fig)
    make_plots(output, rows, straight_rows)
    if args.kerr_init_mode == "zamo_tetrad":
        make_flat_vs_zamo_plots(output, rows, read_csv(output / "kerr_null_propagated_packets.csv"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
