#!/usr/bin/env python3
"""Analyze real-Kerr weighted-energy proxy morphology by particle channel.

This script does not generate new cascade physics. It consumes existing
particle-channel image products produced with the REAL_HADROS_KERR_GEODESIC
backend and computes morphology diagnostics for proxy maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = {
    "gamma": "channel_gamma",
    "electromagnetic": "channel_electromagnetic",
    "hadronic": "channel_hadronic",
    "pion": "channel_pion_charged",
    "total": "channel_total_escaping_null_ok",
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.where(arr > 0.0, arr, 0.0)
    maxv = float(np.max(arr)) if arr.size else 0.0
    return arr / maxv if maxv > 0.0 else arr


def save_image_npz(path: Path, channel: str, image: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=np.asarray(image, dtype=float),
        channel=np.asarray(channel),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def morphology_metrics(channel: str, image: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(image, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.where(arr > 0.0, arr, 0.0)
    total = float(np.sum(arr))
    peak = float(np.max(arr)) if arr.size else 0.0
    nonzero = int(np.count_nonzero(arr > 0.0))
    height, width = arr.shape if arr.ndim == 2 else (0, 0)

    if total > 0.0 and height > 0 and width > 0:
        y, x = np.indices(arr.shape, dtype=float)
        p = arr / total
        cx = float(np.sum(p * x))
        cy = float(np.sum(p * y))
        second = float(np.sum(p * ((x - cx) ** 2 + (y - cy) ** 2)))
        effective_radius = math.sqrt(max(second, 0.0))
        positive = p[p > 0.0]
        entropy_raw = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
        entropy = entropy_raw / math.log(positive.size) if positive.size > 1 else 0.0
    else:
        cx = cy = math.nan
        second = math.nan
        effective_radius = math.nan
        entropy = 0.0

    return {
        "channel": channel,
        "total_weighted_energy_gev": total,
        "peak_pixel_energy_gev": peak,
        "fraction_in_brightest_pixel": peak / total if total > 0.0 else 0.0,
        "number_of_nonzero_pixels": nonzero,
        "centroid_x": cx,
        "centroid_y": cy,
        "image_entropy": entropy,
        "second_moment_pixel2": second,
        "effective_radius_pixel": effective_radius,
        "image_width_pixels": width,
        "image_height_pixels": height,
    }


def centroid_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = finite(a.get("centroid_x"), math.nan), finite(a.get("centroid_y"), math.nan)
    bx, by = finite(b.get("centroid_x"), math.nan), finite(b.get("centroid_y"), math.nan)
    if not all(math.isfinite(v) for v in (ax, ay, bx, by)):
        return math.nan
    return math.hypot(ax - bx, ay - by)


def image_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = np.where(np.isfinite(aa) & (aa > 0.0), aa, 0.0)
    bb = np.where(np.isfinite(bb) & (bb > 0.0), bb, 0.0)
    sa = float(np.sum(aa))
    sb = float(np.sum(bb))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    pa = aa / sa
    pb = bb / sb
    return float(np.sum(np.minimum(pa, pb)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_image(path: Path, image: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(image, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="weighted energy proxy [GeV]")
    ax.set_title(title)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparisons(output: Path, images: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for channel, image in images.items():
        plot_image(plots / f"{channel}_real_kerr.png", image, f"{channel} real-Kerr proxy map")

    gamma = images["gamma"]
    hadronic = images["hadronic"]
    em = images["electromagnetic"]
    eps = 1.0e-300

    plot_image(
        plots / "gamma_vs_hadronic_difference.png",
        gamma - hadronic,
        "gamma - hadronic weighted-energy proxy",
    )
    ratio = np.zeros_like(gamma, dtype=float)
    mask = hadronic > 0.0
    ratio[mask] = gamma[mask] / np.maximum(hadronic[mask], eps)
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    plot_image(plots / "gamma_vs_hadronic_ratio.png", ratio, "gamma / hadronic proxy ratio")

    rgb_gh = np.stack([normalize_image(gamma), normalize_image(hadronic), np.zeros_like(gamma)], axis=-1)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(rgb_gh, origin="lower")
    ax.set_title("overlay: R gamma, G hadronic")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(plots / "gamma_vs_hadronic_overlay.png", dpi=180)
    plt.close(fig)

    rgb_ge = np.stack([normalize_image(gamma), normalize_image(em), np.zeros_like(gamma)], axis=-1)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(rgb_ge, origin="lower")
    ax.set_title("overlay: R gamma, G electromagnetic")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(plots / "gamma_vs_em_overlay.png", dpi=180)
    plt.close(fig)


def answer_questions(metrics: dict[str, dict[str, Any]], images: dict[str, np.ndarray]) -> list[str]:
    gamma = metrics["gamma"]
    had = metrics["hadronic"]
    em = metrics["electromagnetic"]
    gamma_had_centroid = centroid_distance(gamma, had)
    gamma_em_centroid = centroid_distance(gamma, em)
    gamma_had_overlap = image_overlap(images["gamma"], images["hadronic"])
    gamma_em_overlap = image_overlap(images["gamma"], images["electromagnetic"])
    gamma_had_entropy_delta = finite(gamma["image_entropy"]) - finite(had["image_entropy"])
    one_pixel = 1.0

    energy_ratio = (
        finite(gamma["total_weighted_energy_gev"]) / finite(had["total_weighted_energy_gev"], math.nan)
        if finite(had["total_weighted_energy_gev"]) > 0.0
        else math.inf
    )
    answers = [
        "## Required Questions",
        "",
        "1. **O canal gamma domina apenas em energia ou tambem em morfologia?**",
        "",
        f"   Gamma/hadronic total weighted-energy ratio: `{energy_ratio:.12g}`. "
        f"Brightest-pixel fractions are gamma `{finite(gamma['fraction_in_brightest_pixel']):.12g}` "
        f"and hadronic `{finite(had['fraction_in_brightest_pixel']):.12g}`. "
        "This is a proxy-map morphology comparison, not a luminosity comparison.",
        "",
        "2. **Os centroides diferem?**",
        "",
        f"   Gamma-hadronic centroid distance: `{gamma_had_centroid:.12g}` pixels. "
        f"Gamma-electromagnetic centroid distance: `{gamma_em_centroid:.12g}` pixels.",
        "",
        "3. **Os canais ocupam regioes diferentes da tela?**",
        "",
        f"   Normalized gamma-hadronic spatial overlap: `{gamma_had_overlap:.12g}`. "
        f"Normalized gamma-electromagnetic overlap: `{gamma_em_overlap:.12g}`.",
        "",
        "4. **Os canais possuem entropias diferentes?**",
        "",
        f"   Entropy gamma: `{finite(gamma['image_entropy']):.12g}`; "
        f"hadronic: `{finite(had['image_entropy']):.12g}`; "
        f"delta gamma-hadronic: `{gamma_had_entropy_delta:.12g}`.",
        "",
        "5. **Ha evidencia de separacao angular entre canais?**",
        "",
        f"   The proxy-map evidence is encoded by centroid separation and overlap. "
        f"For gamma versus hadronic, separation is `{gamma_had_centroid:.12g}` pixels "
        f"and overlap is `{gamma_had_overlap:.12g}`.",
        "",
        "6. **As diferencas sao maiores que a resolucao da imagem?**",
        "",
        f"   A one-pixel diagnostic threshold is used. Gamma-hadronic centroid "
        f"separation is {'larger' if gamma_had_centroid > one_pixel else 'not larger'} "
        "than one pixel. This is not an angular-resolution or instrument claim.",
        "",
    ]
    return answers


def write_summary(output: Path, source: Path, metrics: dict[str, dict[str, Any]], images: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    gamma = metrics["gamma"]
    had = metrics["hadronic"]
    em = metrics["electromagnetic"]
    lines = [
        "# Real Kerr Channel Morphology Study",
        "",
        "These are weighted-energy proxy maps routed through `REAL_HADROS_KERR_GEODESIC`.",
        "They are not physical luminosity maps, flux maps, spectra, or calibrated observations.",
        "",
        f"- source_npz: `{source}`",
        f"- packet_propagation_backend: `{metadata.get('packet_propagation_backend', 'unknown')}`",
        f"- warning: `{metadata.get('warning', 'weighted-energy proxy maps only')}`",
        "",
        "## Channel Metrics",
        "",
        "| channel | total weighted energy [GeV] | peak pixel [GeV] | brightest fraction | nonzero pixels | centroid x | centroid y | entropy | effective radius [px] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in ["gamma", "electromagnetic", "hadronic", "pion", "total"]:
        row = metrics[channel]
        lines.append(
            f"| {channel} | {finite(row['total_weighted_energy_gev']):.12g} | "
            f"{finite(row['peak_pixel_energy_gev']):.12g} | "
            f"{finite(row['fraction_in_brightest_pixel']):.12g} | "
            f"{int(row['number_of_nonzero_pixels'])} | "
            f"{finite(row['centroid_x'], math.nan):.12g} | "
            f"{finite(row['centroid_y'], math.nan):.12g} | "
            f"{finite(row['image_entropy']):.12g} | "
            f"{finite(row['effective_radius_pixel'], math.nan):.12g} |"
        )
    lines.extend([
        "",
        "## Pairwise Comparisons",
        "",
        f"- gamma_vs_hadronic_centroid_distance_px: `{centroid_distance(gamma, had):.12g}`",
        f"- gamma_vs_electromagnetic_centroid_distance_px: `{centroid_distance(gamma, em):.12g}`",
        f"- gamma_vs_hadronic_overlap: `{image_overlap(images['gamma'], images['hadronic']):.12g}`",
        f"- gamma_vs_electromagnetic_overlap: `{image_overlap(images['gamma'], images['electromagnetic']):.12g}`",
        "",
    ])
    lines.extend(answer_questions(metrics, images))
    lines.extend([
        "## Allowed Interpretation",
        "",
        "The allowed conclusion is limited to morphology differences in weighted-energy",
        "proxy maps after real Kerr null geodesic propagation of null-compatible packets.",
        "",
        "## Prohibited Interpretation",
        "",
        "Do not interpret these maps as physical luminosity, observed flux, physical",
        "spectra, calibrated redshifted intensity, or radiative-transfer output.",
        "",
    ])
    (output / "channel_morphology_summary.md").write_text("\n".join(lines), encoding="utf-8")


def load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    try:
        return json.loads(str(data["metadata"]))
    except json.JSONDecodeError:
        return {"raw_metadata": str(data["metadata"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"),
    )
    parser.add_argument("--channel-images", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/science/channel_morphology"))
    args = parser.parse_args()

    source_npz = args.channel_images or (args.source_run_dir / "real_kerr_packet_images" / "particle_channel_images.npz")
    if not source_npz.exists():
        raise FileNotFoundError(f"Missing real-Kerr channel image NPZ: {source_npz}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))

    data = np.load(source_npz, allow_pickle=True)
    metadata = load_metadata(data)
    backend = metadata.get("packet_propagation_backend", "")
    if backend and backend != "real_kerr_geodesic":
        raise ValueError(f"Expected real_kerr_geodesic channel images, found {backend!r}")

    images: dict[str, np.ndarray] = {}
    for channel, key in CHANNELS.items():
        if key not in data.files:
            raise KeyError(f"Missing channel image key {key} in {source_npz}")
        images[channel] = np.asarray(data[key], dtype=float)

    npz_names = {
        "gamma": "gamma_real_kerr_image.npz",
        "electromagnetic": "electromagnetic_real_kerr_image.npz",
        "hadronic": "hadronic_real_kerr_image.npz",
        "pion": "pion_real_kerr_image.npz",
        "total": "total_real_kerr_image.npz",
    }
    for channel, name in npz_names.items():
        save_image_npz(output / name, channel, images[channel], metadata)

    metrics = {channel: morphology_metrics(channel, image) for channel, image in images.items()}
    rows = [metrics[channel] for channel in ["gamma", "electromagnetic", "hadronic", "pion", "total"]]
    write_csv(output / "channel_morphology_summary.csv", rows)
    plot_comparisons(output, images)
    write_summary(output, source_npz, metrics, images, metadata)
    print(json.dumps({"source_npz": str(source_npz), "metrics": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
