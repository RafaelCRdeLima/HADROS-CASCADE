#!/usr/bin/env python3
"""DEPRECATED / DEBUG ONLY: legacy POWHEG/GEANT4 backward-camera projection.

This file is not part of the final scientific HADROS chain.
Do not use for scientific production.

Final production uses the Phase 15.8 ray-linked source-driven chain and
legacy ``observed_particles_by_pixel`` compatibility rows from the HADROS
particle-ray association camera.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from observed_particles_by_pixel import channels_for_pdg, particle_info


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output/science/powheg_pythia_geant4_resumable/geant4_ready_particles.jsonl"
DEFAULT_EVENT_SOURCE = ROOT / "output/science/powheg_pythia_particles/hadros_particle_events.jsonl"
DEFAULT_OUTPUT = ROOT / "output/science/powheg_geant4_backward_camera"
INPUT_BACKEND = "POWHEG_NUDIS_PYTHIA8_GEANT4_REAL_SAFE"
GENERATOR_BACKEND = "POWHEG_NUDIS_PYTHIA8"
CAMERA_BACKEND = "HADROS_BACKWARD_CAMERA_DIRECTIONAL_PIXEL_ASSOCIATION"

OBSERVED_FIELDS = [
    "pixel_x",
    "pixel_y",
    "event_id",
    "source_particle_id",
    "pdg",
    "particle_name",
    "channel",
    "energy_gev",
    "weighted_energy_gev",
    "px",
    "py",
    "pz",
    "weight",
    "interaction_type",
    "target_type",
    "generator_backend",
    "transport_backend",
    "camera_backend",
    "status",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def load_event_metadata(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        return {}
    counters: dict[int, int] = defaultdict(int)
    meta: dict[tuple[int, int], dict[str, str]] = {}
    for row in read_jsonl(path):
        if row.get("generator_backend") != GENERATOR_BACKEND:
            continue
        event_id = int(row["event_id"])
        counters[event_id] += 1
        particle_id = int(row.get("particle_id", counters[event_id]))
        meta[(event_id, particle_id)] = {
            "interaction_type": str(row.get("interaction_type", "")),
            "target_type": str(row.get("target_type", "")),
        }
    return meta


def project_to_pixel(row: dict[str, Any], args: argparse.Namespace) -> tuple[int | None, int | None, str]:
    px = fnum(row, "px")
    py = fnum(row, "py")
    pz = fnum(row, "pz")
    norm = math.sqrt(px * px + py * py + pz * pz)
    if norm <= 0.0:
        return None, None, "NO_DIRECTION"
    if pz <= 0.0:
        return None, None, "OUTSIDE_FOV"

    half = math.tan(math.radians(args.camera_fov_deg * 0.5))
    if half <= 0.0:
        return None, None, "OUTSIDE_FOV"
    sx = px / pz
    sy = py / pz
    if abs(sx) > half or abs(sy) > half:
        return None, None, "OUTSIDE_FOV"
    pixel_x = int(((sx / half) + 1.0) * 0.5 * args.camera_nx)
    pixel_y = int(((sy / half) + 1.0) * 0.5 * args.camera_ny)
    pixel_x = min(max(pixel_x, 0), args.camera_nx - 1)
    pixel_y = min(max(pixel_y, 0), args.camera_ny - 1)
    return pixel_x, pixel_y, "OBSERVED"


def base_channel(pdg: int) -> str:
    channels = channels_for_pdg(pdg)
    info = particle_info(pdg)
    if "neutrino" in channels:
        return "neutrino"
    if "gamma" in channels:
        return "gamma"
    if "electromagnetic" in channels:
        return "electromagnetic"
    if "hadronic" in channels:
        return "hadronic"
    return info["channel"]


def build_rows(particles: list[dict[str, Any]], event_meta: dict[tuple[int, int], dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for particle in particles:
        if particle.get("origin_backend") != INPUT_BACKEND:
            raise ValueError(f"unexpected origin_backend: {particle.get('origin_backend')}")
        event_id = int(particle["event_id"])
        source_particle_id = int(particle["source_particle_id"])
        pdg = int(particle["pdg"])
        info = particle_info(pdg)
        px = fnum(particle, "px")
        py = fnum(particle, "py")
        pz = fnum(particle, "pz")
        energy = fnum(particle, "energy_gev")
        weight = fnum(particle, "weight", 1.0)
        pixel_x, pixel_y, status = project_to_pixel(particle, args)
        meta = event_meta.get((event_id, source_particle_id), {})
        rows.append(
            {
                "pixel_x": "" if pixel_x is None else pixel_x,
                "pixel_y": "" if pixel_y is None else pixel_y,
                "event_id": event_id,
                "source_particle_id": source_particle_id,
                "pdg": pdg,
                "particle_name": info["particle_name"],
                "channel": base_channel(pdg),
                "energy_gev": energy,
                "weighted_energy_gev": energy * weight if status == "OBSERVED" else 0.0,
                "px": px,
                "py": py,
                "pz": pz,
                "weight": weight,
                "interaction_type": str(particle.get("interaction_type") or meta.get("interaction_type", "")),
                "target_type": str(particle.get("target_type") or meta.get("target_type", "")),
                "generator_backend": GENERATOR_BACKEND,
                "transport_backend": INPUT_BACKEND,
                "camera_backend": CAMERA_BACKEND,
                "status": status,
            }
        )
    return rows


def histograms(rows: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_pdg: dict[tuple[int, str, str], dict[str, Any]] = {}
    by_channel: dict[str, dict[str, Any]] = {}
    pixels_by_pdg: dict[tuple[int, str, str], set[tuple[int, int]]] = defaultdict(set)
    pixels_by_channel: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        pdg = int(row["pdg"])
        key = (pdg, str(row["particle_name"]), str(row["channel"]))
        weighted = fnum(row, "weighted_energy_gev")
        status = str(row["status"])
        pdg_entry = by_pdg.setdefault(
            key,
            {
                "pdg": pdg,
                "particle_name": row["particle_name"],
                "channel": row["channel"],
                "total_energy_gev": 0.0,
                "total_weighted_energy_gev": 0.0,
                "n_particles": 0,
                "n_observed": 0,
                "n_pixels": 0,
            },
        )
        ch_entry = by_channel.setdefault(
            str(row["channel"]),
            {
                "channel": row["channel"],
                "total_energy_gev": 0.0,
                "total_weighted_energy_gev": 0.0,
                "n_particles": 0,
                "n_observed": 0,
                "n_pixels": 0,
            },
        )
        for entry in (pdg_entry, ch_entry):
            entry["total_energy_gev"] += fnum(row, "energy_gev")
            entry["total_weighted_energy_gev"] += weighted
            entry["n_particles"] += 1
            entry["n_observed"] += 1 if status == "OBSERVED" else 0
        if status == "OBSERVED":
            pix = (int(row["pixel_x"]), int(row["pixel_y"]))
            pixels_by_pdg[key].add(pix)
            pixels_by_channel[str(row["channel"])].add(pix)
    for key, entry in by_pdg.items():
        entry["n_pixels"] = len(pixels_by_pdg[key])
    for channel, entry in by_channel.items():
        entry["n_pixels"] = len(pixels_by_channel[channel])
    pdg_rows = sorted(by_pdg.values(), key=lambda item: (-float(item["total_weighted_energy_gev"]), int(item["pdg"])))
    channel_rows = sorted(by_channel.values(), key=lambda item: (-float(item["total_weighted_energy_gev"]), str(item["channel"])))
    write_csv(
        output_dir / "observed_particle_pdg_histogram.csv",
        pdg_rows,
        ["pdg", "particle_name", "channel", "total_energy_gev", "total_weighted_energy_gev", "n_particles", "n_observed", "n_pixels"],
    )
    write_csv(
        output_dir / "observed_particle_channel_histogram.csv",
        channel_rows,
        ["channel", "total_energy_gev", "total_weighted_energy_gev", "n_particles", "n_observed", "n_pixels"],
    )
    return pdg_rows, channel_rows


def make_maps(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    maps = {
        "gamma": np.zeros((args.camera_ny, args.camera_nx), dtype=float),
        "electromagnetic": np.zeros((args.camera_ny, args.camera_nx), dtype=float),
        "hadronic": np.zeros((args.camera_ny, args.camera_nx), dtype=float),
        "neutrino": np.zeros((args.camera_ny, args.camera_nx), dtype=float),
    }
    for row in rows:
        if row["status"] != "OBSERVED":
            continue
        x = int(row["pixel_x"])
        y = int(row["pixel_y"])
        energy = fnum(row, "weighted_energy_gev")
        ch = str(row["channel"])
        if ch in maps:
            maps[ch][y, x] += energy
        if "hadronic" in channels_for_pdg(int(row["pdg"])):
            maps["hadronic"][y, x] += energy
    return maps


def save_image(path: Path, image: np.ndarray, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("pixel_x")
    ax.set_ylabel("pixel_y")
    fig.colorbar(im, ax=ax, label="weighted energy [GeV]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rgb(path: Path, maps: dict[str, np.ndarray]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.dstack([maps["hadronic"], maps["electromagnetic"], maps["neutrino"]])
    maxv = float(np.nanmax(rgb)) if rgb.size else 0.0
    if maxv > 0.0:
        rgb = rgb / maxv
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    ax.set_title("RGB: hadronic, electromagnetic, neutrino")
    ax.set_xlabel("pixel_x")
    ax.set_ylabel("pixel_y")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_maps(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    maps = make_maps(rows, args)
    plots = args.output_dir / "plots"
    save_image(plots / "observed_gamma_map.png", maps["gamma"], "legacy ray-associated gamma")
    save_image(plots / "observed_electromagnetic_map.png", maps["electromagnetic"], "observed electromagnetic")
    save_image(plots / "observed_hadronic_map.png", maps["hadronic"], "legacy ray-associated hadronic")
    save_image(plots / "observed_neutrino_map.png", maps["neutrino"], "observed neutrino")
    save_rgb(plots / "observed_rgb_channels.png", maps)
    np.savez_compressed(args.output_dir / "observed_particle_maps.npz", **maps)
    return maps


def validation(rows: list[dict[str, Any]], maps: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    total_input = sum(fnum(row, "energy_gev") * fnum(row, "weight", 1.0) for row in rows)
    observed = sum(fnum(row, "weighted_energy_gev") for row in rows)
    outside = sum(fnum(row, "energy_gev") * fnum(row, "weight", 1.0) for row in rows if row["status"] == "OUTSIDE_FOV")
    no_direction = sum(fnum(row, "energy_gev") * fnum(row, "weight", 1.0) for row in rows if row["status"] == "NO_DIRECTION")
    observed_pixels = {(int(row["pixel_x"]), int(row["pixel_y"])) for row in rows if row["status"] == "OBSERVED"}
    pdgs = {int(row["pdg"]) for row in rows if row["status"] == "OBSERVED"}
    return {
        "status": "POWHEG_GEANT4_BACKWARD_CAMERA_VALIDATED" if observed > 0.0 and len(observed_pixels) > 0 else "POWHEG_GEANT4_BACKWARD_CAMERA_BLOCKED",
        "input_rows": len(rows),
        "observed_rows": sum(1 for row in rows if row["status"] == "OBSERVED"),
        "input_weighted_energy_gev": total_input,
        "observed_weighted_energy_gev": observed,
        "outside_fov_weighted_energy_gev": outside,
        "no_direction_weighted_energy_gev": no_direction,
        "outside_fov_fraction": outside / total_input if total_input > 0.0 else 0.0,
        "no_direction_fraction": no_direction / total_input if total_input > 0.0 else 0.0,
        "nonzero_pixels": len(observed_pixels),
        "observed_pdgs": len(pdgs),
        "camera_nx": args.camera_nx,
        "camera_ny": args.camera_ny,
    }


def write_metadata(args: argparse.Namespace) -> None:
    payload = {
        "CAM_NX": args.camera_nx,
        "CAM_NY": args.camera_ny,
        "CAM_FOV_DEG": args.camera_fov_deg,
        "CAM_THETA_DEG": args.camera_theta_deg,
        "CAM_R_OBS_RG": args.camera_r_obs_rg,
        "CAM_R_MAX_RG": args.camera_r_max_rg,
        "CAM_STEP": args.camera_step,
        "camera_backend": CAMERA_BACKEND,
        "physics_control_notes": {
            "CAM_NX": "Controls output image width.",
            "CAM_NY": "Controls output image height.",
            "CAM_FOV_DEG": "Controls angular acceptance and pixel mapping.",
            "CAM_THETA_DEG": "Recorded as HADROS camera provenance; this directional association does not ray-trace through Kerr geometry.",
            "CAM_R_OBS_RG": "Recorded as HADROS camera provenance; not used in this local direction-to-pixel association.",
            "CAM_R_MAX_RG": "Recorded as HADROS camera provenance; not used because no geodesic integration is performed in Phase 15.4.",
            "CAM_STEP": "Recorded as HADROS camera provenance; not used because no geodesic integration is performed in Phase 15.4.",
        },
    }
    (args.output_dir / "camera_parameters.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reports(summary: dict[str, Any], pdg_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    write_csv(args.output_dir / "backward_camera_particle_validation.csv", [summary], list(summary.keys()))
    lines = [
        "# POWHEG GEANT4 Backward Camera Particle Validation",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"- input_backend: `{INPUT_BACKEND}`",
        f"- transport_backend: `{INPUT_BACKEND}`",
        f"- camera_backend: `{CAMERA_BACKEND}`",
        f"- input_rows: `{summary['input_rows']}`",
        f"- ray_associated_rows: `{summary['observed_rows']}`",
        f"- input_weighted_energy_gev: `{summary['input_weighted_energy_gev']:.12g}`",
        f"- ray_associated_weighted_energy_gev: `{summary['observed_weighted_energy_gev']:.12g}`",
        f"- outside_fov_fraction: `{summary['outside_fov_fraction']:.12g}`",
        f"- no_direction_fraction: `{summary['no_direction_fraction']:.12g}`",
        f"- nonzero_pixels: `{summary['nonzero_pixels']}`",
        f"- ray_associated_pdgs: `{summary['observed_pdgs']}`",
        "",
        "No PYTHIA e+e- proxy, local response proxy, forward packet projection, hybrid packet screen, or auto-frame packet diagnostic is used.",
    ]
    (args.output_dir / "backward_camera_particle_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Legacy Particle-Ray Association Rows By Pixel",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"- rows: `{summary['observed_rows']}` ray-associated / `{summary['input_rows']}` input",
        f"- input_weighted_energy_gev: `{summary['input_weighted_energy_gev']:.12g}`",
        f"- ray_associated_weighted_energy_gev: `{summary['observed_weighted_energy_gev']:.12g}`",
        f"- nonzero_pixels: `{summary['nonzero_pixels']}`",
        f"- ray_associated_pdgs: `{summary['observed_pdgs']}`",
        f"- camera_backend: `{CAMERA_BACKEND}`",
        "- legacy_name_notice: `observed_particles_by_pixel.* is a compatibility name; rows are particle-ray association records, not full particle transport to the observer`",
        "",
        "Rows are physical GEANT4 escaped particles from POWHEG nuDIS + PYTHIA8, associated to backward-camera pixels by their momentum direction. This is not a detector image.",
        "",
        "## PDG Histogram",
        "",
        "| PDG | particle | channel | ray-associated rows | pixels | weighted energy [GeV] |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in pdg_rows:
        summary_lines.append(
            f"| {row['pdg']} | {row['particle_name']} | {row['channel']} | {row['n_observed']} | {row['n_pixels']} | {float(row['total_weighted_energy_gev']):.12g} |"
        )
    summary_lines.extend(["", "## Channel Histogram", "", "| channel | ray-associated rows | pixels | weighted energy [GeV] |", "|---|---:|---:|---:|"])
    for row in channel_rows:
        summary_lines.append(f"| {row['channel']} | {row['n_observed']} | {row['n_pixels']} | {float(row['total_weighted_energy_gev']):.12g} |")
    (args.output_dir / "observed_particles_by_pixel_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--event-source", type=Path, default=DEFAULT_EVENT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-nx", type=int, default=int(os.environ.get("CAM_NX", "128")))
    parser.add_argument("--camera-ny", type=int, default=int(os.environ.get("CAM_NY", "128")))
    parser.add_argument("--camera-fov-deg", type=float, default=float(os.environ.get("CAM_FOV_DEG", "60.0")))
    parser.add_argument("--camera-theta-deg", type=float, default=float(os.environ.get("CAM_THETA_DEG", "70.0")))
    parser.add_argument("--camera-r-obs-rg", type=float, default=float(os.environ.get("CAM_R_OBS_RG", "80.0")))
    parser.add_argument("--camera-r-max-rg", type=float, default=float(os.environ.get("CAM_R_MAX_RG", "120.0")))
    parser.add_argument("--camera-step", type=float, default=float(os.environ.get("CAM_STEP", "0.02")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    particles = read_jsonl(args.input)
    event_meta = load_event_metadata(args.event_source)
    rows = build_rows(particles, event_meta, args)
    write_csv(args.output_dir / "observed_particles_by_pixel.csv", rows, OBSERVED_FIELDS)
    write_jsonl(args.output_dir / "observed_particles_by_pixel.jsonl", rows)
    pdg_rows, channel_rows = histograms(rows, args.output_dir)
    maps = write_maps(rows, args)
    summary = validation(rows, maps, args)
    write_metadata(args)
    write_reports(summary, pdg_rows, channel_rows, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "POWHEG_GEANT4_BACKWARD_CAMERA_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
