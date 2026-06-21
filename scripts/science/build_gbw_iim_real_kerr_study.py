#!/usr/bin/env python3
"""Build the Phase 11 GBW/IIM real-Kerr packet study.

This script is deliberately a reporting layer. It does not run new physics,
does not modify the cascade pipeline, and does not invent an IIM result when
the frozen cascade state has not produced one.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBW_RUN = ROOT / "output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"
DEFAULT_OUTPUT = ROOT / "output/science"
DEFAULT_DOC = ROOT / "docs/science/GBW_IIM_REAL_KERR_STUDY.md"

NULL_CLASSES = {"MASSLESS_NULL", "ULTRARELATIVISTIC_NULL_OK"}
REQUIRED_CHANNELS = ["gamma", "electromagnetic", "hadronic", "pion_charged"]


def parse_value(text: str) -> Any:
    value = text.strip().strip("`").strip()
    if value.lower() in {"", "none", "nan"}:
        return math.nan if value.lower() == "nan" else ""
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_md_key_values(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return data
    pattern = re.compile(r"^\s*[-*]\s+([^:]+):\s+`?([^`]+)`?\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            data[key] = parse_value(match.group(2))
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_channels(run_dir: Path) -> dict[str, float]:
    rows = read_csv(run_dir / "particle_channel_images.csv")
    out = {f"{channel}_energy_gev": math.nan for channel in REQUIRED_CHANNELS}
    out["total_escaping_null_ok_energy_gev"] = math.nan
    out["captured_total_escaping_null_ok_energy_gev"] = math.nan
    for row in rows:
        channel = row.get("channel", "")
        total = fnum(row.get("total_energy_gev"))
        image = fnum(row.get("image_energy_gev"))
        if channel in REQUIRED_CHANNELS:
            out[f"{channel}_energy_gev"] = total
            out[f"{channel}_captured_energy_gev"] = image
        if channel == "total_escaping_null_ok":
            out["total_escaping_null_ok_energy_gev"] = total
            out["captured_total_escaping_null_ok_energy_gev"] = image
    return out


def summarize_classification(run_dir: Path) -> dict[str, float]:
    rows = read_csv(run_dir / "escaping_packet_classification.csv")
    total = sum(fnum(row.get("weighted_energy_gev"), 0.0) for row in rows)
    null_total = sum(
        fnum(row.get("weighted_energy_gev"), 0.0)
        for row in rows
        if row.get("classification") in NULL_CLASSES
    )
    return {
        "number_of_packets": float(len(rows)),
        "packet_weighted_energy_gev": total,
        "null_compatible_energy_gev": null_total,
        "null_compatible_fraction": null_total / total if total > 0 else math.nan,
    }


def summarize_validation(output_dir: Path) -> dict[str, float]:
    rows = read_csv(output_dir / "real_kerr_packet_validation.csv")
    if rows:
        delta_values = [
            abs(fnum(row.get("delta_h", row.get("hamiltonian_error")), 0.0))
            for row in rows
        ]
        null_values = [
            abs(fnum(row.get("initial_null_norm", ""), math.nan))
            for row in rows
        ]
        if not any(math.isfinite(value) for value in null_values):
            null_values = [
                2.0 * abs(fnum(row.get("initial_hamiltonian"), 0.0))
                for row in rows
            ]
        return {
            "max_delta_h": max(delta_values) if delta_values else math.nan,
            "max_gpp": max(null_values) if null_values else math.nan,
        }
    md = parse_md_key_values(output_dir / "real_kerr_packet_validation.md")
    return {
        "max_delta_h": fnum(md.get("max_|delta_h|", md.get("max_delta_h"))),
        "max_gpp": fnum(md.get("max_|g(p,p)|", md.get("max_gpp"))),
    }


def summarize_run(model: str, run_dir: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "run_dir": str(run_dir) if run_dir else "",
        "status": "MISSING_RUN",
        "dis_model_operationally_coupled": "false",
        "note": "",
    }
    if run_dir is None or not run_dir.exists():
        row["note"] = "No run directory was provided for this DIS model."
        return row

    batch = parse_md_key_values(run_dir / "geant4_batch_summary.md")
    channels = summarize_channels(run_dir)
    packets = summarize_classification(run_dir)
    channel_summary = parse_md_key_values(run_dir / "particle_channel_images_summary.md")
    validation = summarize_validation(ROOT / "output/cascade")
    local_validation = summarize_validation(run_dir)
    for key, value in local_validation.items():
        if not math.isfinite(value):
            local_validation[key] = validation.get(key, math.nan)

    processed = fnum(batch.get("processed_energy_gev"))
    escaped = fnum(batch.get("escaped_energy_gev"), 0.0)
    unsupported = fnum(batch.get("escaped_unsupported_uhe_energy_gev"), 0.0)
    deposited = fnum(batch.get("deposited_energy_gev"), 0.0)
    invisible = fnum(batch.get("invisible_energy_gev"), 0.0)
    untracked = fnum(batch.get("untracked_energy_gev"), 0.0)
    captured = fnum(channels.get("captured_total_escaping_null_ok_energy_gev"), 0.0)
    total_packets = fnum(channels.get("total_escaping_null_ok_energy_gev"), packets["packet_weighted_energy_gev"])

    row.update(
        {
            "status": "AVAILABLE_REAL_KERR_PARTIAL" if "PARTIAL" in str(batch.get("status", "")) else "AVAILABLE_REAL_KERR",
            "batch_status": batch.get("status", ""),
            "processed_energy_gev": processed,
            "unprocessed_energy_gev": fnum(batch.get("unprocessed_energy_gev")),
            "E_dep_gev": deposited,
            "E_esc_gev": escaped,
            "E_invisible_gev": invisible,
            "E_untracked_gev": untracked,
            "E_unsupported_UHE_gev": unsupported,
            "escape_fraction_including_unsupported_uhe": (escaped + unsupported) / processed if processed > 0 else math.nan,
            "local_transport_escape_fraction": escaped / processed if processed > 0 else math.nan,
            "deposited_fraction": deposited / processed if processed > 0 else math.nan,
            "closure_error_gev": fnum(batch.get("closure_error_gev"), fnum(batch.get("closure_error"))),
            "number_of_packets": packets["number_of_packets"],
            "packet_weighted_energy_gev": packets["packet_weighted_energy_gev"],
            "null_compatible_fraction": packets["null_compatible_fraction"],
            "best_theta_deg": fnum(channel_summary.get("theta_deg")),
            "best_phi_deg": fnum(channel_summary.get("phi_deg")),
            "best_cone_deg": fnum(channel_summary.get("cone_deg")),
            "captured_fraction": captured / total_packets if total_packets > 0 else math.nan,
            "captured_energy_gev": captured,
            "max_delta_H": local_validation["max_delta_h"],
            "max_g_p_p": local_validation["max_gpp"],
            "note": "Real-Kerr packet/channel products are available for this run.",
        }
    )
    row.update(channels)
    return row


def make_missing_iim_row() -> dict[str, Any]:
    row = summarize_run("IIM", None)
    row["status"] = "NOT_OPERATIONALLY_COUPLED_IN_FROZEN_CASCADE"
    row["note"] = (
        "The frozen cascade/config-web path records dis_model as provenance, "
        "but the PYTHIA/GEANT4/packet/channel chain is not driven by an IIM "
        "cross-section table in this production layer."
    )
    return row


def ratio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in rows}
    gbw = by_model.get("GBW", {})
    iim = by_model.get("IIM", {})
    metrics = [
        "E_dep_gev",
        "E_esc_gev",
        "captured_fraction",
        "gamma_energy_gev",
        "electromagnetic_energy_gev",
        "hadronic_energy_gev",
    ]
    out = []
    for metric in metrics:
        g = fnum(gbw.get(metric))
        i = fnum(iim.get(metric))
        ratio = i / g if math.isfinite(i) and math.isfinite(g) and g != 0 else math.nan
        rel = (i - g) / g if math.isfinite(i) and math.isfinite(g) and g != 0 else math.nan
        out.append({"metric": metric, "GBW": g, "IIM": i, "IIM_over_GBW": ratio, "relative_difference": rel})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_zero(value: Any) -> float:
    x = fnum(value, 0.0)
    return x if math.isfinite(x) else 0.0


def annotate_missing(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in rows]

    def values(metric: str) -> list[float]:
        return [finite_or_zero(row.get(metric)) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(rows))
    for metric, name in [
        ("E_dep_gev", "deposited"),
        ("E_esc_gev", "escaped transported"),
        ("E_invisible_gev", "invisible"),
        ("E_untracked_gev", "untracked"),
        ("E_unsupported_UHE_gev", "unsupported UHE escaped"),
    ]:
        vals = np.array(values(metric))
        ax.bar(labels, vals, bottom=bottom, label=name)
        bottom += vals
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Energy [GeV]")
    ax.set_title("GBW vs IIM energy budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_energy_budget.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values("escape_fraction_including_unsupported_uhe"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Escape fraction including unsupported UHE")
    ax.set_title("Escape fraction")
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_escape_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values("null_compatible_fraction"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Energy fraction")
    ax.set_title("Null-compatible packet fraction")
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_packet_composition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    width = 0.2
    for idx, metric in enumerate(["gamma_energy_gev", "electromagnetic_energy_gev", "hadronic_energy_gev", "pion_charged_energy_gev"]):
        ax.bar(x + (idx - 1.5) * width, values(metric), width, label=metric.replace("_energy_gev", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Channel energy [GeV]")
    ax.set_title("Channel composition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_channel_composition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    theta = values("best_theta_deg")
    phi = values("best_phi_deg")
    frac = values("captured_fraction")
    sc = ax.scatter(phi, theta, s=[80 + 220 * v for v in frac], c=frac, vmin=0, vmax=1)
    for label, px, py in zip(labels, phi, theta):
        ax.annotate(label, (px, py), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("best phi [deg]")
    ax.set_ylabel("best theta [deg]")
    ax.set_title("Best-cone observer")
    fig.colorbar(sc, ax=ax, label="captured fraction")
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_best_observer.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, row in zip(axes, rows):
        npz_path = Path(str(row.get("run_dir", ""))) / "particle_channel_images.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                if "rgb_hadronic_em_deposited" in data:
                    img = data["rgb_hadronic_em_deposited"]
                elif "rgb" in data:
                    img = data["rgb"]
                else:
                    img = np.zeros((16, 16, 3))
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"{row['model']} RGB proxy")
        else:
            ax.set_title(f"{row['model']} unavailable")
            annotate_missing(ax, "No image\navailable")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Diagnostic RGB comparison, not physical luminosity")
    fig.tight_layout()
    fig.savefig(plot_dir / "gbw_vs_iim_rgb_comparison.png", dpi=180)
    plt.close(fig)


def format_table(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}" if math.isfinite(val) else "n/a")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_markdown(rows: list[dict[str, Any]], ratios: list[dict[str, Any]], output_dir: Path, doc_path: Path) -> None:
    gbw = next((row for row in rows if row["model"] == "GBW"), {})
    lines = [
        "# GBW vs IIM Real-Kerr Packet Study",
        "",
        "This is the Phase 11.0 scientific production audit using the frozen HADROS-CASCADE state.",
        "It uses `REAL_HADROS_KERR_GEODESIC` packet propagation where available.",
        "",
        "## Executive Result",
        "",
        "**The current frozen cascade production layer cannot yet answer a physical GBW-vs-IIM cascade question.**",
        "The available real-Kerr production run is GBW-labelled, while the config-web cascade path records `dis_model` as provenance and does not drive PYTHIA, GEANT4, packetization, or channel images with a GBW/IIM cross-section table.",
        "",
        "Therefore the honest scientific conclusion is a limiting result: downstream packet/channel differences between GBW and IIM are not scientifically defined in the current frozen chain. A full GBW/IIM comparison requires upstream interaction sampling or statistical weights that are operationally coupled to the selected DIS table before the cascade products are generated.",
        "",
        "## Current Pipeline Used",
        "",
        "```text",
        "interaction points -> DIS-labelled config provenance -> PYTHIA proxy",
        "-> GEANT4 real_safe / UHE skip policy -> escaping packets",
        "-> ultrarelativistic classification -> REAL_HADROS_KERR_GEODESIC",
        "-> weighted-energy proxy channel images",
        "```",
        "",
        "The channel images remain weighted-energy proxy maps, not physical luminosities.",
        "",
        "## Run Summary",
        "",
    ]
    keys = [
        "model",
        "status",
        "processed_energy_gev",
        "E_dep_gev",
        "E_esc_gev",
        "E_invisible_gev",
        "E_untracked_gev",
        "E_unsupported_UHE_gev",
        "escape_fraction_including_unsupported_uhe",
        "number_of_packets",
        "null_compatible_fraction",
        "captured_fraction",
    ]
    lines.extend(format_table(rows, keys))
    lines += [
        "",
        "## Channel Composition",
        "",
    ]
    channel_keys = ["model", "gamma_energy_gev", "electromagnetic_energy_gev", "hadronic_energy_gev", "pion_charged_energy_gev", "best_theta_deg", "best_phi_deg", "best_cone_deg"]
    lines.extend(format_table(rows, channel_keys))
    lines += [
        "",
        "## IIM / GBW Ratios",
        "",
        "Ratios are reported as `n/a` when the IIM quantity is not operationally available in the frozen pipeline.",
    ]
    lines.extend(format_table(ratios, ["metric", "GBW", "IIM", "IIM_over_GBW", "relative_difference"]))
    lines += [
        "",
        "## Geodesic Diagnostics",
        "",
        f"- max `|Delta H|`: `{gbw.get('max_delta_H', math.nan)}`",
        f"- max `|g(p,p)|`: `{gbw.get('max_g_p_p', math.nan)}`",
        "- Backend: `REAL_HADROS_KERR_GEODESIC` for the available packet image products.",
        "- Redshift remains uncalibrated; image values are weighted-energy proxies.",
        "",
        "## Scientific Interpretation",
        "",
        "### GBW e IIM produzem diferenças significativas?",
        "",
        "Not testable in this frozen cascade production state. The available downstream products do not carry an operational GBW/IIM difference beyond provenance.",
        "",
        "### Essas diferenças aparecem mais na deposição ou nos packets escapados?",
        "",
        "For the available GBW-labelled partial run, escaped energy dominates deposited energy, mostly through the explicit unsupported-UHE escape policy. A GBW/IIM contrast in deposition versus escaping packets is not defined until the DIS model changes the sampled interactions or weights.",
        "",
        "### A anisotropia muda?",
        "",
        "The available GBW-labelled run has real-Kerr packet anisotropy diagnostics, but there is no independent IIM real-Kerr run with operational DIS coupling to compare.",
        "",
        "### A composição por canal muda?",
        "",
        "The available run is gamma/electromagnetic dominated. A GBW/IIM channel-composition change is not established by the current pipeline.",
        "",
        "## Allowed Claims From This Study",
        "",
        "- The available E_nu=1e9 GeV partial real_safe run is escape dominated.",
        "- The available channel products were routed through `REAL_HADROS_KERR_GEODESIC`.",
        "- The current config-web cascade layer does not yet provide an operational GBW/IIM downstream comparison.",
        "- A future publishable GBW/IIM cascade study must couple the DIS table to interaction sampling or event weights before PYTHIA/GEANT4/packet products are compared.",
        "",
        "## Claims Not Allowed",
        "",
        "- Do not claim that GBW and IIM differ in packet composition from this run.",
        "- Do not claim physical luminosity, physical flux, or calibrated observed spectra.",
        "- Do not treat unsupported-UHE skipped energy as local GEANT4 deposition.",
        "- Do not treat PYTHIA proxy as a publishable UHE neutrino-DIS generator.",
    ]
    text = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gbw_iim_real_kerr_summary.md").write_text(text, encoding="utf-8")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbw-run-dir", type=Path, default=DEFAULT_GBW_RUN)
    parser.add_argument("--iim-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    rows = [summarize_run("GBW", args.gbw_run_dir)]
    rows.append(summarize_run("IIM", args.iim_run_dir) if args.iim_run_dir else make_missing_iim_row())
    ratios = ratio_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gbw_iim_real_kerr_summary.csv", rows)
    write_markdown(rows, ratios, args.output_dir, args.doc_path)
    make_plots(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
