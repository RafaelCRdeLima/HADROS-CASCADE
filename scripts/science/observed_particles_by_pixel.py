#!/usr/bin/env python3
"""Utilities for pixel-level observed particle products."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PDG_MAP_PATH = ROOT / "data/particles/pdg_particle_channels.yaml"

OBSERVED_PARTICLE_FIELDS = [
    "run_name",
    "pixel_x",
    "pixel_y",
    "pdg_id",
    "particle_name",
    "channel",
    "parent_channel",
    "source_event_id",
    "source_particle_id",
    "production_chain",
    "origin_x_rg",
    "origin_y_rg",
    "origin_z_rg",
    "camera_theta_deg",
    "camera_phi_deg",
    "camera_fov_deg",
    "camera_r_obs_rg",
    "source_energy_gev",
    "observed_energy_proxy_gev",
    "weight",
    "weighted_energy_proxy_gev",
    "momentum_px_proxy",
    "momentum_py_proxy",
    "momentum_pz_proxy",
    "momentum_norm_proxy",
    "status",
    "notes",
]

TRACKING_STATUSES = {
    "OBSERVED",
    "OUTSIDE_FOV",
    "SKIPPED_UNSUPPORTED_UHE",
    "SKIPPED_INVISIBLE",
    "SKIPPED_MASSIVE_GEODESIC_NOT_IMPLEMENTED",
    "UNKNOWN_PDG",
    "NO_PARTICLE_PRODUCTION",
}

HADRONIC_CHANNELS = {"pion_charged", "pion_neutral", "kaon", "baryon"}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return int(value)
    except ValueError:
        return value


def load_pdg_channel_map(path: Path = PDG_MAP_PATH) -> tuple[dict[int, dict[str, str]], list[str]]:
    """Load the small repo-local YAML map without requiring PyYAML."""
    particles: dict[int, dict[str, str]] = {}
    aggregated: list[str] = []
    current: dict[str, Any] | None = None
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "particles:":
            section = "particles"
            continue
        if stripped == "aggregated_channels:":
            if current and "pdg_id" in current:
                particles[int(current["pdg_id"])] = {
                    "particle_name": str(current.get("particle_name", "unknown")),
                    "channel": str(current.get("channel", "unknown")),
                }
            current = None
            section = "aggregated_channels"
            continue
        if section == "particles":
            if stripped.startswith("- "):
                if current and "pdg_id" in current:
                    particles[int(current["pdg_id"])] = {
                        "particle_name": str(current.get("particle_name", "unknown")),
                        "channel": str(current.get("channel", "unknown")),
                    }
                current = {}
                stripped = stripped[2:].strip()
            if ":" in stripped and current is not None:
                key, value = stripped.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
        elif section == "aggregated_channels" and stripped.startswith("- "):
            aggregated.append(stripped[2:].strip())
    if current and "pdg_id" in current:
        particles[int(current["pdg_id"])] = {
            "particle_name": str(current.get("particle_name", "unknown")),
            "channel": str(current.get("channel", "unknown")),
        }
    return particles, aggregated


def particle_info(pdg_id: int) -> dict[str, str]:
    mapping, _channels = load_pdg_channel_map()
    info = mapping.get(int(pdg_id))
    if info:
        return dict(info)
    return {"particle_name": f"pdg_{int(pdg_id)}", "channel": "unknown"}


def channels_for_pdg(pdg_id: int) -> set[str]:
    channel = particle_info(pdg_id)["channel"]
    channels = {channel}
    if int(pdg_id) in {22, 11, -11}:
        channels.add("electromagnetic")
    if channel in HADRONIC_CHANNELS:
        channels.add("hadronic")
    if abs(int(pdg_id)) > 1000 and channel == "unknown":
        channels.add("hadronic")
    return channels


def normalize_momentum(px: float, py: float, pz: float) -> tuple[float, float, float, float]:
    norm = math.sqrt(px * px + py * py + pz * pz)
    if norm <= 0.0 or not math.isfinite(norm):
        return 0.0, 0.0, 0.0, 0.0
    return px / norm, py / norm, pz / norm, norm


def pixel_momentum_proxy(pixel_x: int, pixel_y: int, nx: int, ny: int, fov_deg: float) -> tuple[float, float, float, float]:
    half = math.tan(math.radians(max(fov_deg, 1.0e-12) * 0.5))
    x = (((pixel_x + 0.5) / max(nx, 1)) * 2.0 - 1.0) * half
    y = (((pixel_y + 0.5) / max(ny, 1)) * 2.0 - 1.0) * half
    return normalize_momentum(x, y, 1.0)


def row_matches_filters(row: dict[str, Any], *, particle_filter: str = "all", pdg_filter: str = "", channel_filter: str = "") -> bool:
    particle_filter = (particle_filter or "all").strip()
    pdg_filter = (pdg_filter or "").strip()
    channel_filter = (channel_filter or "").strip()
    if pdg_filter:
        allowed = {int(item.strip()) for item in pdg_filter.split(",") if item.strip()}
        if int(row["pdg_id"]) not in allowed:
            return False
    row_channels = channels_for_pdg(int(row["pdg_id"])) | {str(row.get("channel", "")), str(row.get("parent_channel", ""))}
    if channel_filter and channel_filter not in row_channels:
        return False
    if particle_filter and particle_filter != "all":
        if particle_filter in row_channels:
            return True
        if particle_filter == str(row.get("particle_name", "")):
            return True
        try:
            return int(particle_filter) == int(row["pdg_id"])
        except ValueError:
            return False
    return True


def write_observed_particle_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "observed_particles_by_pixel.csv"
    jsonl_path = output_dir / "observed_particles_by_pixel.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVED_PARTICLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OBSERVED_PARTICLE_FIELDS})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({field: row.get(field, "") for field in OBSERVED_PARTICLE_FIELDS}, sort_keys=True) + "\n")


def read_observed_particle_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_name(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"+", "-"}:
            out.append("_plus" if ch == "+" else "_minus")
        else:
            out.append("_")
    cleaned = "_".join(part for part in "".join(out).split("_") if part)
    return cleaned or "unknown"


def build_maps_and_histograms(
    output_dir: Path,
    rows: list[dict[str, Any]],
    nx: int,
    ny: int,
    *,
    observed_particle_filter: str = "all",
    observed_pdg_filter: str = "",
    observed_channel_filter: str = "",
) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    mapping, aggregate_channels = load_pdg_channel_map()
    image_channels = ["gamma", "neutrino", "hadronic", "electromagnetic"]
    images = {channel: np.zeros((ny, nx), dtype=float) for channel in image_channels}
    particle_images: dict[tuple[int, str, str], np.ndarray] = {}
    selected_image = np.zeros((ny, nx), dtype=float)
    has_selected_filter = any(str(value).strip() and str(value).strip() != "all" for value in [observed_particle_filter, observed_pdg_filter, observed_channel_filter])
    by_pdg: dict[tuple[int, str, str], dict[str, Any]] = {}
    by_channel: dict[str, dict[str, Any]] = {}
    pixels_by_pdg: dict[tuple[int, str, str], set[tuple[int, int]]] = defaultdict(set)
    pixels_by_channel: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for row in rows:
        pdg_id = int(finite(row.get("pdg_id")))
        info = mapping.get(pdg_id, {"particle_name": str(row.get("particle_name", f"pdg_{pdg_id}")), "channel": str(row.get("channel", "unknown"))})
        base_channel = str(row.get("channel") or info["channel"])
        weighted = finite(row.get("weighted_energy_proxy_gev"))
        px = int(finite(row.get("pixel_x"), -1))
        py = int(finite(row.get("pixel_y"), -1))
        channels = channels_for_pdg(pdg_id) | {base_channel}
        if base_channel in HADRONIC_CHANNELS:
            channels.add("hadronic")
        key = (pdg_id, info["particle_name"], base_channel)
        particle_image = particle_images.setdefault(key, np.zeros((ny, nx), dtype=float))
        if 0 <= px < nx and 0 <= py < ny:
            particle_image[py, px] += weighted
            if has_selected_filter and row_matches_filters(
                row,
                particle_filter=observed_particle_filter,
                pdg_filter=observed_pdg_filter,
                channel_filter=observed_channel_filter,
            ):
                selected_image[py, px] += weighted
            for channel in channels:
                if channel in images:
                    images[channel][py, px] += weighted
                pixels_by_channel[channel].add((px, py))
            pixels_by_pdg[key].add((px, py))
        entry = by_pdg.setdefault(key, {
            "pdg_id": pdg_id,
            "particle_name": info["particle_name"],
            "channel": base_channel,
            "total_weighted_energy_proxy_gev": 0.0,
            "number_of_contributing_pixels": 0,
            "number_of_contributions": 0,
            "max_pixel_energy_proxy_gev": 0.0,
        })
        entry["total_weighted_energy_proxy_gev"] += weighted
        entry["number_of_contributions"] += 1
        entry["max_pixel_energy_proxy_gev"] = max(entry["max_pixel_energy_proxy_gev"], weighted)
        for channel in channels:
            centry = by_channel.setdefault(channel, {
                "channel": channel,
                "total_weighted_energy_proxy_gev": 0.0,
                "number_of_contributing_pixels": 0,
                "number_of_contributions": 0,
                "max_pixel_energy_proxy_gev": 0.0,
            })
            centry["total_weighted_energy_proxy_gev"] += weighted
            centry["number_of_contributions"] += 1
            centry["max_pixel_energy_proxy_gev"] = max(centry["max_pixel_energy_proxy_gev"], weighted)

    for key, entry in by_pdg.items():
        entry["number_of_contributing_pixels"] = len(pixels_by_pdg[key])
    for channel, entry in by_channel.items():
        entry["number_of_contributing_pixels"] = len(pixels_by_channel[channel])

    pdg_rows = sorted(by_pdg.values(), key=lambda item: (-float(item["total_weighted_energy_proxy_gev"]), int(item["pdg_id"])))
    channel_rows = sorted(by_channel.values(), key=lambda item: (-float(item["total_weighted_energy_proxy_gev"]), str(item["channel"])))
    write_table(output_dir / "observed_particle_pdg_histogram.csv", pdg_rows, [
        "pdg_id",
        "particle_name",
        "channel",
        "total_weighted_energy_proxy_gev",
        "number_of_contributing_pixels",
        "number_of_contributions",
        "max_pixel_energy_proxy_gev",
    ])
    write_table(output_dir / "observed_particle_channel_histogram.csv", channel_rows, [
        "channel",
        "total_weighted_energy_proxy_gev",
        "number_of_contributing_pixels",
        "number_of_contributions",
        "max_pixel_energy_proxy_gev",
    ])
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "observed_particles_by_pixel_maps.npz",
        **{f"observed_{key}": value for key, value in images.items()},
        **{
            f"observed_particle_{safe_name(name)}_pdg_{pdg_id}_{safe_name(channel)}": image
            for (pdg_id, name, channel), image in particle_images.items()
        },
        observed_selected_particle_pixels=selected_image,
    )
    for channel, image in images.items():
        save_image(plots / f"observed_{channel}_map.png", image, f"observed {channel} weighted-energy proxy")
    for (pdg_id, name, channel), image in sorted(particle_images.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
        save_image(
            plots / f"observed_particle_{safe_name(name)}_pdg_{pdg_id}_{safe_name(channel)}_map.png",
            image,
            f"observed {name} (PDG {pdg_id}, {channel}) weighted-energy proxy",
        )
    if has_selected_filter:
        label_parts = [
            f"particle={observed_particle_filter or 'all'}",
            f"pdg={observed_pdg_filter or 'any'}",
            f"channel={observed_channel_filter or 'any'}",
        ]
        save_image(
            plots / "observed_selected_particle_pixels.png",
            selected_image,
            "selected observed particle pixels: " + ", ".join(label_parts),
        )
    rgb = rgb_image(images["hadronic"], images["electromagnetic"], images["neutrino"])
    save_rgb(plots / "observed_rgb_channels.png", rgb, "RGB: hadronic, electromagnetic, neutrino")
    save_bar(plots / "observed_particles_by_pdg.png", [str(row["pdg_id"]) for row in pdg_rows], [float(row["total_weighted_energy_proxy_gev"]) for row in pdg_rows], "PDG", "weighted energy proxy [GeV]")
    save_bar(plots / "observed_particles_by_channel.png", [str(row["channel"]) for row in channel_rows], [float(row["total_weighted_energy_proxy_gev"]) for row in channel_rows], "channel", "weighted energy proxy [GeV]")
    write_summary(
        output_dir / "observed_particles_by_pixel_summary.md",
        rows,
        pdg_rows,
        channel_rows,
        aggregate_channels,
        observed_particle_filter=observed_particle_filter,
        observed_pdg_filter=observed_pdg_filter,
        observed_channel_filter=observed_channel_filter,
    )
    return {"pdg_rows": pdg_rows, "channel_rows": channel_rows, "images": images, "particle_images": particle_images, "selected_image": selected_image}


def write_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_image(path: Path, image: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("camera x pixel")
    ax.set_ylabel("camera y pixel")
    fig.colorbar(im, ax=ax, label="weighted energy proxy [GeV]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rgb_image(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.dstack([red, green, blue]).astype(float)
    maxv = float(np.nanmax(rgb)) if rgb.size else 0.0
    if maxv > 0.0:
        rgb /= maxv
    return rgb


def save_rgb(path: Path, rgb: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("camera x pixel")
    ax.set_ylabel("camera y pixel")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(path: Path, labels: list[str], values: list[float], xlabel: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=40)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    pdg_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    aggregate_channels: list[str],
    *,
    observed_particle_filter: str = "all",
    observed_pdg_filter: str = "",
    observed_channel_filter: str = "",
) -> None:
    lines = [
        "# Legacy Particle-Ray Association Rows By Camera Pixel",
        "",
        "- particle_tracking_status: `PIXEL_PARTICLE_TRACKING_PARTIAL`" if rows else "- particle_tracking_status: `PIXEL_PARTICLE_TRACKING_NOT_AVAILABLE`",
        f"- observed_particle_filter: `{observed_particle_filter}`",
        f"- observed_pdg_filter: `{observed_pdg_filter}`",
        f"- observed_channel_filter: `{observed_channel_filter}`",
        "- proxy_status: `observed_energy_proxy_gev is not redshift-calibrated physical observed energy`",
        "- legacy_name_notice: `observed_particles_by_pixel.* is a compatibility name; rows are particle-ray association records, not proof of full transport to the observer`",
        "- uses_massive_geodesics: `false`",
        "- aggregate_channels: `" + ", ".join(aggregate_channels) + "`",
        "",
        "Each row in `observed_particles_by_pixel.csv` is one legacy-named particle/channel association with a camera pixel.",
        "Plots named `observed_particle_*_map.png` are legacy compatibility plots separated by individual PDG/particle.",
        "`observed_selected_particle_pixels.png` is generated when a particle, PDG, or channel filter is active.",
        "",
        "## PDG Histogram",
        "",
        "| PDG | particle | channel | total weighted energy proxy [GeV] | pixels | contributions |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in pdg_rows:
        lines.append(
            f"| {row['pdg_id']} | {row['particle_name']} | {row['channel']} | "
            f"{float(row['total_weighted_energy_proxy_gev']):.12g} | {row['number_of_contributing_pixels']} | {row['number_of_contributions']} |"
        )
    lines.extend(["", "## Channel Histogram", "", "| channel | total weighted energy proxy [GeV] | pixels | contributions |", "|---|---:|---:|---:|"])
    for row in channel_rows:
        lines.append(
            f"| {row['channel']} | {float(row['total_weighted_energy_proxy_gev']):.12g} | "
            f"{row['number_of_contributing_pixels']} | {row['number_of_contributions']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
