#!/usr/bin/env python3
"""Build diagnostic particle-channel image products from existing cascade outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "science"))
from observed_particles_by_pixel import channels_for_pdg  # noqa: E402


NULL_OK_CLASSES = {"MASSLESS_NULL", "ULTRARELATIVISTIC_NULL_OK"}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return tuple(x / norm for x in vec)


def direction_from_angles(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def camera_basis(axis: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ref = (0.0, 0.0, 1.0)
    if abs(sum(axis[i] * ref[i] for i in range(3))) > 0.95:
        ref = (0.0, 1.0, 0.0)
    ux = ref[1] * axis[2] - ref[2] * axis[1]
    uy = ref[2] * axis[0] - ref[0] * axis[2]
    uz = ref[0] * axis[1] - ref[1] * axis[0]
    u = normalize((ux, uy, uz)) or (1.0, 0.0, 0.0)
    v = (
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    )
    return u, normalize(v) or (0.0, 1.0, 0.0)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def pair_packets_with_classes(packets: list[dict[str, Any]], class_rows: list[dict[str, str]]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    buckets: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in class_rows:
        buckets[(int(finite(row.get("event_id"))), int(finite(row.get("pdg_id"))))].append(row)
    used: dict[tuple[int, int], int] = defaultdict(int)
    paired = []
    for packet in packets:
        key = (int(finite(packet.get("event_id"))), int(finite(packet.get("pdg_id"))))
        index = used[key]
        used[key] += 1
        rows = buckets.get(key, [])
        cls = rows[index] if index < len(rows) else {}
        paired.append((packet, cls))
    return paired


def packet_channels(pdg_id: int, classification: str) -> set[str]:
    canonical = set(channels_for_pdg(pdg_id))
    channels: set[str] = set()
    if "gamma" in canonical:
        channels.add("gamma")
    if "electromagnetic" in canonical:
        channels.add("electromagnetic")
    if "muon" in canonical:
        channels.add("muon")
    if "pion_charged" in canonical:
        channels.add("pion_charged")
    if "hadronic" in canonical:
        channels.add("hadronic")
    if "neutrino" in canonical or classification not in NULL_OK_CLASSES:
        channels.add("invisible_or_skipped")
    if classification in NULL_OK_CLASSES:
        channels.add("total_escaping_null_ok")
    return channels


def load_deposited_image(args: argparse.Namespace) -> tuple[np.ndarray, Path | None, dict[str, Any]]:
    candidates = [args.deposited_image]
    if args.deposited_image is None:
        candidates = [
            args.output_dir / "deposition_proxy_camera_autoframe_image.npz",
            args.output_dir / "deposition_proxy_camera_image.npz",
        ]
    for candidate in candidates:
        if candidate and candidate.exists():
            data = np.load(candidate, allow_pickle=True)
            image = np.asarray(data["image"], dtype=float)
            stats: dict[str, Any] = {}
            if "stats" in data.files:
                try:
                    stats = json.loads(str(data["stats"]))
                except json.JSONDecodeError:
                    stats = {"raw_stats": str(data["stats"])}
            return image, candidate, stats
    return np.zeros((args.image_size, args.image_size), dtype=float), None, {}


def choose_observer(args: argparse.Namespace, paired: list[tuple[dict[str, Any], dict[str, str]]]) -> tuple[float, float, tuple[float, float, float], str]:
    if args.observer_mode == "manual":
        axis = normalize(direction_from_angles(args.theta_deg, args.phi_deg)) or (0.0, 0.0, 1.0)
        return args.theta_deg, args.phi_deg, axis, "manual"
    if args.observer_mode == "best_cone":
        scan_path = args.kerr_scan if args.kerr_scan.exists() else args.straight_scan
        rows = read_csv(scan_path)
        key = f"captured_energy_cone_{args.cone_deg:g}_deg"
        if rows and key in rows[0]:
            best = max(rows, key=lambda row: finite(row.get(key)))
            theta = finite(best.get("theta_obs_deg"))
            phi = finite(best.get("phi_obs_deg"))
            axis = normalize(direction_from_angles(theta, phi)) or (0.0, 0.0, 1.0)
            return theta, phi, axis, f"best_cone:{scan_path}"
    sx = sy = sz = sw = 0.0
    for packet, cls in paired:
        classification = str(cls.get("classification", ""))
        if classification not in NULL_OK_CLASSES:
            continue
        energy = finite(cls.get("weighted_energy_gev"), finite(packet.get("weighted_energy_gev"), finite(packet.get("energy_gev"))))
        direction = normalize((finite(packet.get("px_gev")), finite(packet.get("py_gev")), finite(packet.get("pz_gev"))))
        if direction is None:
            continue
        sx += energy * direction[0]
        sy += energy * direction[1]
        sz += energy * direction[2]
        sw += energy
    axis = normalize((sx, sy, sz)) or (0.0, 0.0, 1.0)
    theta = math.degrees(math.acos(max(-1.0, min(1.0, axis[2]))))
    phi = (math.degrees(math.atan2(axis[1], axis[0])) + 360.0) % 360.0
    return theta, phi, axis, "auto"


def project_direction(
    direction: tuple[float, float, float],
    axis: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    cone_deg: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    cosang = max(-1.0, min(1.0, dot(direction, axis)))
    angle = math.degrees(math.acos(cosang))
    if angle > cone_deg or cosang <= 0.0:
        return None
    scale = math.tan(math.radians(cone_deg))
    if scale <= 0.0:
        return None
    x = dot(direction, u) / cosang / scale
    y = dot(direction, v) / cosang / scale
    if abs(x) > 1.0 or abs(y) > 1.0:
        return None
    ix = int((x + 1.0) * 0.5 * image_width)
    iy = int((1.0 - (y + 1.0) * 0.5) * image_height)
    ix = max(0, min(image_width - 1, ix))
    iy = max(0, min(image_height - 1, iy))
    return iy, ix


def pixel_from_real_kerr_row(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    camera_source: str,
    particle_camera_mode: str,
    axis: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    fov_deg: float,
) -> tuple[int, int] | None:
    status = str(row.get("final_status", ""))
    if status not in {"ESCAPED_DOMAIN", "ESCAPED_TO_OBSERVER"}:
        return None
    theta = finite(row.get("observer_theta"), math.nan)
    phi = finite(row.get("observer_phi"), math.nan)
    if not math.isfinite(theta) or not math.isfinite(phi):
        return None
    theta = max(0.0, min(math.pi, theta))
    phi = (phi + 2.0 * math.pi) % (2.0 * math.pi)
    if particle_camera_mode == "hybrid_packet_screen":
        direction = normalize((
            finite(row.get("final_x")),
            finite(row.get("final_y")),
            finite(row.get("final_z")),
        ))
        if direction is None:
            direction = (
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            )
    else:
        direction = (
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        )
    if camera_source == "config_web" or particle_camera_mode == "hybrid_packet_screen":
        cosang = max(-1.0, min(1.0, dot(direction, axis)))
        if cosang <= 0.0:
            return None
        half_angle = max(1.0e-12, 0.5 * fov_deg)
        angle = math.degrees(math.acos(cosang))
        if angle > half_angle:
            return None
        scale = math.tan(math.radians(half_angle))
        if scale <= 0.0:
            return None
        x = dot(direction, u) / cosang / scale
        y = dot(direction, v) / cosang / scale
        if abs(x) > 1.0 or abs(y) > 1.0:
            return None
        ix = int((x + 1.0) * 0.5 * image_width)
        iy = int((1.0 - (y + 1.0) * 0.5) * image_height)
    else:
        ix = int(phi / (2.0 * math.pi) * image_width)
        iy = int((1.0 - theta / math.pi) * image_height)
    ix = max(0, min(image_width - 1, ix))
    iy = max(0, min(image_height - 1, iy))
    return iy, ix


def real_kerr_rows_by_packet(rows: list[dict[str, str]]) -> dict[tuple[int, int, str], list[dict[str, str]]]:
    buckets: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            int(finite(row.get("event_id"))),
            int(finite(row.get("pdg_id"))),
            f"{finite(row.get('energy_gev')):.15g}",
        )
        buckets[key].append(row)
    return buckets


def add_rgb(name: str, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack([red, green, blue], axis=-1).astype(float)
    maxv = float(np.nanmax(rgb)) if rgb.size else 0.0
    if maxv > 0.0:
        rgb /= maxv
    return rgb


def make_plots(output_dir: Path, images: dict[str, np.ndarray], rgb: np.ndarray, rgb_alt: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for channel in [
        "deposited",
        "gamma",
        "electromagnetic",
        "hadronic",
        "pion_charged",
        "total_escaping_null_ok",
    ]:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(images[channel], origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, label="weighted energy proxy [GeV]")
        ax.set_title(f"diagnostic channel: {channel}")
        ax.set_xlabel("camera x pixel")
        ax.set_ylabel("camera y pixel")
        fig.tight_layout()
        fig.savefig(plots / f"channel_{channel}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(rgb, origin="lower")
    ax.set_title("RGB: R hadronic, G electromagnetic, B deposited")
    ax.set_xlabel("camera x pixel")
    ax.set_ylabel("camera y pixel")
    fig.tight_layout()
    fig.savefig(plots / "channel_rgb_composite.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(rgb_alt, origin="lower")
    ax.set_title("RGB: R gamma, G charged pions, B deposited")
    ax.set_xlabel("camera x pixel")
    ax.set_ylabel("camera y pixel")
    fig.tight_layout()
    fig.savefig(plots / "channel_rgb_gamma_pion_deposited.png", dpi=180)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# Particle-Channel Diagnostic Images",
        "",
        "diagnostic channel image, not physical luminosity",
        "",
        f"- observer_mode: `{meta['observer_mode']}`",
        f"- sample_label: `{meta.get('sample_label', '')}`",
        f"- packet_propagation_backend: `{meta.get('packet_propagation_backend', 'proxy_straight_line')}`",
        f"- dis_weight_model: `{meta.get('dis_weight_model', '')}`",
        f"- observer_source: `{meta['observer_source']}`",
        f"- camera_source: `{meta.get('camera_source', 'cascade_defaults')}`",
        f"- particle_camera_mode: `{meta.get('particle_camera_mode', 'exit_sky')}`",
        f"- proxy_status: `{meta.get('proxy_status', 'weighted_energy_proxy_not_luminosity')}`",
        f"- requested_theta_deg: `{float(meta.get('requested_theta_deg', meta['theta_deg'])):.12g}`",
        f"- requested_phi_deg: `{float(meta.get('requested_phi_deg', meta['phi_deg'])):.12g}`",
        f"- requested_fov_deg: `{float(meta.get('requested_fov_deg', meta.get('camera_fov_deg', meta['cone_deg']))):.12g}`",
        f"- effective_theta_deg: `{float(meta.get('effective_theta_deg', meta['theta_deg'])):.12g}`",
        f"- effective_phi_deg: `{float(meta.get('effective_phi_deg', meta['phi_deg'])):.12g}`",
        f"- effective_fov_deg: `{float(meta.get('effective_fov_deg', meta.get('camera_fov_deg', meta['cone_deg']))):.12g}`",
        f"- auto_frame_capture_fraction: `{meta.get('auto_frame_capture_fraction', '')}`",
        f"- auto_frame_status: `{meta.get('auto_frame_status', 'NOT_REQUESTED')}`",
        f"- theta_deg: `{meta['theta_deg']:.12g}`",
        f"- phi_deg: `{meta['phi_deg']:.12g}`",
        f"- cone_deg: `{meta['cone_deg']:.12g}`",
        f"- camera_fov_deg: `{float(meta.get('camera_fov_deg', meta['cone_deg'])):.12g}`",
        f"- camera_r_obs_rg: `{meta.get('camera_r_obs_rg', '')}`",
        f"- camera_r_max_rg: `{meta.get('camera_r_max_rg', '')}`",
        f"- camera_step: `{meta.get('camera_step', '')}`",
        f"- deposited_image_source: `{meta['deposited_image_source']}`",
        f"- total_deposited_source_energy_gev: `{meta['total_deposited_source_energy_gev']:.12g}`",
        f"- total_deposited_image_energy_gev: `{meta['total_deposited_image_energy_gev']:.12g}`",
        f"- total_packet_weighted_energy_gev: `{meta['total_packet_weighted_energy_gev']:.12g}`",
        f"- total_packet_raw_energy_gev: `{meta['total_packet_raw_energy_gev']:.12g}`",
        f"- null_ok_weighted_energy_gev: `{meta['null_ok_weighted_energy_gev']:.12g}`",
        f"- non_propagated_weighted_energy_gev: `{meta['non_propagated_weighted_energy_gev']:.12g}`",
        f"- null_ok_fraction: `{meta['null_ok_fraction']:.12g}`",
        "",
        "The images are channelized weighted-energy proxies. They do not include",
        "radiative microphysics, physical luminosity calibration, or massive",
        "geodesic propagation.",
        "",
        "## Channel Energies",
        "",
        "| Channel | Total energy [GeV] | Captured image energy [GeV] |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['channel']} | {float(row['total_energy_gev']):.12g} | {float(row['image_energy_gev']):.12g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--packets", type=Path, default=Path("output/cascade/escaping_particle_packets.jsonl"))
    parser.add_argument("--classification", type=Path, default=Path("output/cascade/escaping_packet_classification.csv"))
    parser.add_argument("--kerr-scan", type=Path, default=Path("output/cascade/kerr_packet_observer_scan.csv"))
    parser.add_argument("--straight-scan", type=Path, default=Path("output/cascade/packet_observer_scan.csv"))
    parser.add_argument("--packet-propagation-backend", choices=["proxy_straight_line", "real_kerr_geodesic"], default="proxy_straight_line")
    parser.add_argument("--real-kerr-propagated", type=Path, default=None)
    parser.add_argument("--deposited-image", type=Path, default=None)
    parser.add_argument("--observer-mode", choices=["best_cone", "auto", "manual"], default="best_cone")
    parser.add_argument("--theta-deg", type=float, default=0.0)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    parser.add_argument("--cone-deg", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--camera-source", choices=["cascade_defaults", "config_web", "auto_frame_particle_packets"], default="cascade_defaults")
    parser.add_argument("--camera-fov-deg", type=float, default=None)
    parser.add_argument("--camera-theta-deg", type=float, default=None)
    parser.add_argument("--camera-phi-deg", type=float, default=None)
    parser.add_argument("--camera-r-obs-rg", default="")
    parser.add_argument("--camera-r-max-rg", default="")
    parser.add_argument("--camera-step", default="")
    parser.add_argument("--particle-camera-mode", choices=["exit_sky", "hybrid_packet_screen"], default="exit_sky")
    parser.add_argument("--auto-frame-json", type=Path, default=None)
    parser.add_argument("--auto-frame-capture-fraction", type=float, default=None)
    parser.add_argument("--sample-label", default="")
    parser.add_argument("--dis-weight-model", default="")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    packets = read_jsonl(args.packets)
    classifications = read_csv(args.classification)
    paired = pair_packets_with_classes(packets, classifications)
    real_kerr_path = args.real_kerr_propagated or (output / "real_kerr_propagated_packets.csv")
    real_kerr_rows = read_csv(real_kerr_path) if args.packet_propagation_backend == "real_kerr_geodesic" else []
    real_kerr_buckets = real_kerr_rows_by_packet(real_kerr_rows)
    real_kerr_used: dict[tuple[int, int, str], int] = defaultdict(int)
    deposited, deposited_source, deposited_stats = load_deposited_image(args)
    if args.image_width is None and args.image_height is None and deposited_source is not None and deposited.ndim == 2:
        image_height, image_width = int(deposited.shape[0]), int(deposited.shape[1])
    else:
        image_width = int(args.image_width or args.image_size)
        image_height = int(args.image_height or image_width)
    if deposited.shape != (image_height, image_width):
        deposited = np.zeros((image_height, image_width), dtype=float)
    theta, phi, axis, observer_source = choose_observer(args, paired)
    camera_fov = float(args.camera_fov_deg if args.camera_fov_deg is not None else 2.0 * args.cone_deg)
    requested_theta = float(args.camera_theta_deg if args.camera_theta_deg is not None else theta)
    requested_phi = float(args.camera_phi_deg if args.camera_phi_deg is not None else phi)
    requested_fov = camera_fov
    auto_frame_status = "NOT_REQUESTED"
    if args.auto_frame_json is not None:
        if args.auto_frame_json.exists():
            auto = json.loads(args.auto_frame_json.read_text(encoding="utf-8"))
            theta = finite(auto.get("theta_auto_deg"), theta)
            phi = finite(auto.get("phi_auto_deg"), phi)
            camera_fov = finite(auto.get("recommended_fov_deg"), camera_fov)
            args.camera_source = "auto_frame_particle_packets"
            auto_frame_status = "APPLIED"
            if args.auto_frame_capture_fraction is None:
                args.auto_frame_capture_fraction = finite(auto.get("auto_frame_capture_fraction"), math.nan)
        else:
            auto_frame_status = "MISSING_AUTO_FRAME_JSON"
    if args.camera_source == "config_web":
        theta = float(args.camera_theta_deg if args.camera_theta_deg is not None else theta)
        phi = float(args.camera_phi_deg if args.camera_phi_deg is not None else phi)
        axis = normalize(direction_from_angles(theta, phi)) or axis
        observer_source = "config_web_camera"
    elif args.camera_source == "auto_frame_particle_packets":
        axis = normalize(direction_from_angles(theta, phi)) or axis
        observer_source = "auto_frame_particle_packets"
    if args.packet_propagation_backend == "real_kerr_geodesic":
        observer_source = f"real_kerr_exit_sky:{real_kerr_path}"
        if args.camera_source == "config_web":
            observer_source = f"config_web_camera_projected_real_kerr_exit_sky:{real_kerr_path}"
        if args.camera_source == "auto_frame_particle_packets":
            observer_source = f"auto_frame_particle_packets_projected_real_kerr_exit_sky:{real_kerr_path}"
        if args.particle_camera_mode == "hybrid_packet_screen":
            observer_source = f"hybrid_packet_screen:{real_kerr_path}"
    u, v = camera_basis(axis)

    channels = [
        "deposited",
        "gamma",
        "electron_positron",
        "muon",
        "pion_charged",
        "hadronic",
        "electromagnetic",
        "invisible_or_skipped",
        "total_escaping_null_ok",
    ]
    images = {channel: np.zeros((image_height, image_width), dtype=float) for channel in channels}
    images["deposited"] = deposited.copy()
    totals = {channel: 0.0 for channel in channels}
    totals["deposited"] = float(np.nansum(deposited))
    captured = {channel: 0.0 for channel in channels}
    captured["deposited"] = totals["deposited"]

    total_packet_energy = 0.0
    total_packet_raw_energy = 0.0
    null_ok_energy = 0.0
    non_propagated_energy = 0.0
    packet_rows: list[dict[str, Any]] = []
    for packet, cls in paired:
        pdg_id = int(finite(packet.get("pdg_id")))
        classification = str(cls.get("classification", "UNKNOWN_CLASSIFICATION"))
        energy = finite(cls.get("weighted_energy_gev"), finite(packet.get("weighted_energy_gev"), finite(packet.get("energy_gev"))))
        raw_energy = finite(packet.get("energy_gev"))
        total_packet_energy += energy
        total_packet_raw_energy += raw_energy
        if classification in NULL_OK_CLASSES:
            null_ok_energy += energy
        else:
            non_propagated_energy += energy
        packet_ch = packet_channels(pdg_id, classification)
        for channel in packet_ch:
            totals[channel] += energy
        pixel = None
        final_status = ""
        if args.packet_propagation_backend == "real_kerr_geodesic":
            key = (int(finite(packet.get("event_id"))), pdg_id, f"{finite(packet.get('energy_gev')):.15g}")
            index = real_kerr_used[key]
            real_kerr_used[key] += 1
            rows_for_packet = real_kerr_buckets.get(key, [])
            real_row = rows_for_packet[index] if index < len(rows_for_packet) else {}
            final_status = str(real_row.get("final_status", "MISSING_REAL_KERR_ROW"))
            pixel = pixel_from_real_kerr_row(real_row, image_width, image_height, args.camera_source, args.particle_camera_mode, axis, u, v, camera_fov) if real_row else None
        else:
            direction = normalize((finite(packet.get("px_gev")), finite(packet.get("py_gev")), finite(packet.get("pz_gev"))))
            projection_cone = 0.5 * camera_fov if args.camera_source == "config_web" else args.cone_deg
            pixel = project_direction(direction, axis, u, v, projection_cone, image_width, image_height) if direction else None
        if pixel is not None:
            iy, ix = pixel
            for channel in packet_ch:
                images[channel][iy, ix] += energy
                captured[channel] += energy
        packet_rows.append({
            "event_id": int(finite(packet.get("event_id"))),
            "pdg_id": pdg_id,
            "particle_label": packet.get("particle_label", ""),
            "classification": classification,
            "weighted_energy_gev": energy,
            "channels": ";".join(sorted(packet_ch)),
            "captured": pixel is not None,
            "packet_propagation_backend": args.packet_propagation_backend,
            "final_status": final_status,
        })

    rows = [
        {
            "channel": channel,
            "total_energy_gev": totals[channel],
            "image_energy_gev": captured[channel],
            "fraction_of_packet_energy": totals[channel] / max(total_packet_energy, 1.0e-300) if channel != "deposited" else 0.0,
        }
        for channel in channels
    ]
    rgb = add_rgb("had_em_dep", images["hadronic"], images["electromagnetic"], images["deposited"])
    rgb_alt = add_rgb("gamma_pion_dep", images["gamma"], images["pion_charged"], images["deposited"])
    np.savez_compressed(
        output / "particle_channel_images.npz",
        **{f"channel_{key}": value for key, value in images.items()},
        rgb_hadronic_electromagnetic_deposited=rgb,
        rgb_gamma_pion_deposited=rgb_alt,
        observer_axis=np.asarray(axis, dtype=float),
        theta_deg=np.asarray(theta),
        phi_deg=np.asarray(phi),
        cone_deg=np.asarray(args.cone_deg),
        camera_fov_deg=np.asarray(camera_fov),
        camera_source=np.asarray(args.camera_source),
        particle_camera_mode=np.asarray(args.particle_camera_mode),
        image_width=np.asarray(image_width),
        image_height=np.asarray(image_height),
        metadata=np.asarray(json.dumps({
            "warning": "diagnostic channel image, not physical luminosity",
            "observer_mode": args.observer_mode,
            "sample_label": args.sample_label,
            "observer_source": observer_source,
            "packet_propagation_backend": args.packet_propagation_backend,
            "dis_weight_model": args.dis_weight_model,
            "real_kerr_propagated": str(real_kerr_path) if args.packet_propagation_backend == "real_kerr_geodesic" else "",
            "deposited_image_source": str(deposited_source) if deposited_source else "",
            "deposited_stats": deposited_stats,
            "camera_source": args.camera_source,
            "particle_camera_mode": args.particle_camera_mode,
            "proxy_status": "weighted_energy_proxy_not_luminosity",
            "requested_theta_deg": requested_theta,
            "requested_phi_deg": requested_phi,
            "requested_fov_deg": requested_fov,
            "effective_theta_deg": theta,
            "effective_phi_deg": phi,
            "effective_fov_deg": camera_fov,
            "auto_frame_capture_fraction": args.auto_frame_capture_fraction,
            "auto_frame_status": auto_frame_status,
            "camera_fov_deg": camera_fov,
            "camera_theta_deg": theta,
            "camera_phi_deg": phi,
            "camera_r_obs_rg": args.camera_r_obs_rg,
            "camera_r_max_rg": args.camera_r_max_rg,
            "camera_step": args.camera_step,
            "image_width": image_width,
            "image_height": image_height,
        }, sort_keys=True)),
    )
    write_csv(output / "particle_channel_images.csv", rows, ["channel", "total_energy_gev", "image_energy_gev", "fraction_of_packet_energy"])
    write_csv(output / "particle_channel_packet_assignments.csv", packet_rows, ["event_id", "pdg_id", "particle_label", "classification", "weighted_energy_gev", "channels", "captured", "packet_propagation_backend", "final_status"])
    meta = {
        "observer_mode": args.observer_mode,
        "sample_label": args.sample_label,
        "dis_weight_model": args.dis_weight_model,
        "observer_source": observer_source,
        "packet_propagation_backend": args.packet_propagation_backend,
        "theta_deg": theta,
        "phi_deg": phi,
        "cone_deg": args.cone_deg,
        "camera_source": args.camera_source,
        "particle_camera_mode": args.particle_camera_mode,
        "proxy_status": "weighted_energy_proxy_not_luminosity",
        "requested_theta_deg": requested_theta,
        "requested_phi_deg": requested_phi,
        "requested_fov_deg": requested_fov,
        "effective_theta_deg": theta,
        "effective_phi_deg": phi,
        "effective_fov_deg": camera_fov,
        "auto_frame_capture_fraction": args.auto_frame_capture_fraction,
        "auto_frame_status": auto_frame_status,
        "camera_fov_deg": camera_fov,
        "camera_r_obs_rg": args.camera_r_obs_rg,
        "camera_r_max_rg": args.camera_r_max_rg,
        "camera_step": args.camera_step,
        "deposited_image_source": str(deposited_source) if deposited_source else "",
        "total_deposited_source_energy_gev": finite(
            deposited_stats.get("total_weighted_deposited_energy_gev"),
            totals["deposited"],
        ),
        "total_deposited_image_energy_gev": totals["deposited"],
        "total_packet_weighted_energy_gev": total_packet_energy,
        "total_packet_raw_energy_gev": total_packet_raw_energy,
        "null_ok_weighted_energy_gev": null_ok_energy,
        "non_propagated_weighted_energy_gev": non_propagated_energy,
        "null_ok_fraction": null_ok_energy / max(total_packet_energy, 1.0e-300),
    }
    write_summary(output / "particle_channel_images_summary.md", rows, meta)
    make_plots(output, images, rgb, rgb_alt)
    if args.particle_camera_mode == "hybrid_packet_screen":
        np.savez_compressed(
            output / "hybrid_packet_screen_image.npz",
            **{f"channel_{key}": value for key, value in images.items()},
            rgb_hadronic_electromagnetic_deposited=rgb,
            metadata=np.asarray(json.dumps({
                "camera_source": "hadros_camera_geometry",
                "particle_camera_mode": "hybrid_packet_screen",
                "proxy_status": "weighted_energy_proxy_not_luminosity",
                "source": "REAL_HADROS_KERR_GEODESIC packet final positions projected onto camera screen",
                "camera_fov_deg": camera_fov,
                "theta_deg": theta,
                "phi_deg": phi,
                "image_width": image_width,
                "image_height": image_height,
            }, sort_keys=True)),
        )
        (output / "hybrid_packet_screen_summary.md").write_text(
            "\n".join([
                "# Hybrid Packet-Screen Camera Prototype",
                "",
                "Experimental prototype: weighted-energy proxy, not physical luminosity.",
                "",
                "- camera_source: `hadros_camera_geometry`",
                "- particle_camera_mode: `hybrid_packet_screen`",
                "- proxy_status: `weighted_energy_proxy_not_luminosity`",
                f"- packet_propagation_backend: `{args.packet_propagation_backend}`",
                f"- camera_fov_deg: `{camera_fov:.12g}`",
                f"- theta_deg: `{theta:.12g}`",
                f"- phi_deg: `{phi:.12g}`",
                f"- image_width: `{image_width}`",
                f"- image_height: `{image_height}`",
                "",
                "The prototype projects real-Kerr packet escape positions/directions onto",
                "a camera-defined screen. It does not implement redshift, radiative",
                "transfer, physical emissivity, flux calibration, or massive geodesics.",
                "",
            ]),
            encoding="utf-8",
        )
        plots = output / "plots"
        for src, dst in [
            ("channel_gamma.png", "hybrid_gamma.png"),
            ("channel_electromagnetic.png", "hybrid_electromagnetic.png"),
            ("channel_hadronic.png", "hybrid_hadronic.png"),
            ("channel_rgb_composite.png", "hybrid_rgb.png"),
        ]:
            src_path = plots / src
            if src_path.exists():
                (output / dst).write_bytes(src_path.read_bytes())
    print(json.dumps({"meta": meta, "channels": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
