#!/usr/bin/env python3
"""Build the standard scientific plot bundle for particle-production runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]

CHANNELS = ["gamma", "electromagnetic", "hadronic", "neutrino"]


def spec(plot_id: str, filename: str, group: str, source_data: list[str], description: str) -> dict[str, Any]:
    return {
        "plot_id": plot_id,
        "filename": filename,
        "group": group,
        "category": "CORE_SCIENCE",
        "importance": "A",
        "source_data": source_data,
        "description": description,
        "scientific_claim": "HADROS cascade-origin / particle-ray association product when generated from the listed source data; not full secondary-particle transport to the observer.",
        "visibility": "NORMAL",
    }


PLOT_SPECS = [
    spec("01", "01_particle_ray_association_rgb.png", "observed", ["particle_ray_association_camera.csv"], "RGB particle-ray association map."),
    spec("02", "02_particle_ray_association_gamma_map.png", "observed", ["particle_ray_association_camera.csv"], "Ray-associated gamma secondary map."),
    spec("03", "03_particle_ray_association_hadronic_map.png", "observed", ["particle_ray_association_camera.csv"], "Ray-associated hadronic secondary map."),
    spec("04", "04_produced_energy_by_channel.png", "production", ["hadros_particle_events.jsonl", "powheg_pythia_particles.csv"], "Produced energy by channel after POWHEG/PYTHIA."),
    spec("05", "05_surviving_energy_by_channel.png", "geant4", ["geant4_ready_particles.jsonl", "powheg_pythia_geant4_resumable_particles.csv"], "Surviving/escaped energy by channel after GEANT4."),
    spec("06", "06_particle_ray_association_energy_by_channel.png", "observed", ["particle_ray_association_camera.csv"], "Associated secondary energy by channel in Kerr-ray pixels."),
    spec("07", "07_particle_ray_association_energy_spectrum_by_channel.png", "observed", ["particle_ray_association_camera.csv"], "Associated secondary energy spectrum by channel."),
    spec("08", "08_geant4_energy_budget.png", "geant4", ["powheg_pythia_geant4_resumable_summary.csv"], "GEANT4 deposited/escaped/invisible/untracked/residual energy budget."),
    spec("09", "09_gbw_iim_energy_ratio_by_channel.png", "gbw_iim", ["gbw_iim_camera_summary.csv", "particle_ray_association_camera.csv"], "IIM/GBW weighted associated-energy ratio by channel."),
    spec("10", "10_gbw_iim_ratio_map.png", "gbw_iim", ["gbw_iim_camera_summary.csv", "particle_ray_association_camera.csv"], "Spatial IIM/GBW ratio map over ray-associated secondary particles."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        if not fields:
            fields = ["missing_reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        return out if math.isfinite(out) else default
    return default


def pdg(row: dict[str, Any]) -> int:
    return int(float(str(row.get("pdg_id", row.get("pdg", row.get("particle_pdg", 0))) or 0)))


def channel_for(row: dict[str, Any]) -> str:
    explicit = str(row.get("channel", row.get("parent_channel", ""))).strip()
    if explicit:
        if explicit in {"pion_charged", "pion_neutral", "kaon", "baryon", "hadron", "meson"}:
            return "hadronic"
        if explicit in {"lepton", "electron", "positron"}:
            return "electromagnetic"
        return explicit
    apdg = abs(pdg(row))
    if apdg in {12, 14, 16}:
        return "neutrino"
    if apdg in {22, 11}:
        return "gamma" if apdg == 22 else "electromagnetic"
    if apdg in {111, 211, 130, 310, 311, 321, 2212, 2112} or apdg > 1000:
        return "hadronic"
    return "other"


def energy(row: dict[str, Any]) -> float:
    return fnum(row, "weighted_energy_proxy_gev", "observed_energy_proxy_gev", "weighted_energy_gev", "energy_gev", "energy", "source_energy_gev")


def momentum(row: dict[str, Any]) -> float:
    explicit = fnum(row, "momentum_norm_proxy", "p", "momentum", default=math.nan)
    if math.isfinite(explicit):
        return explicit
    px = fnum(row, "momentum_px_proxy", "px")
    py = fnum(row, "momentum_py_proxy", "py")
    pz = fnum(row, "momentum_pz_proxy", "pz")
    return math.sqrt(px * px + py * py + pz * pz)


def load_sources(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    candidates = [run_dir, run_dir / "cascade", run_dir / "science"]

    def find(name: str) -> Path:
        for base in candidates:
            direct = base / name
            if direct.exists():
                return direct
            matches = sorted(base.rglob(name)) if base.exists() else []
            if matches:
                return matches[0]
        return run_dir / name

    sources = {
        "produced_jsonl": read_jsonl(find("hadros_particle_events.jsonl")),
        "produced_csv": read_csv(find("powheg_pythia_particles.csv")),
        "geant4_summary": read_csv(find("powheg_pythia_geant4_resumable_summary.csv")),
        "geant4_particles": read_csv(find("powheg_pythia_geant4_resumable_particles.csv")),
        "geant4_ready": read_jsonl(find("geant4_ready_particles.jsonl")),
        "observed": read_csv(find("particle_ray_association_camera.csv")) or read_csv(find("observed_particles_by_pixel.csv")),
        "gbw_iim": read_csv(find("gbw_iim_camera_summary.csv")),
    }
    return sources


def setup_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def bar_plot(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, ylabel: str) -> None:
    plt = setup_matplotlib(path.parent)
    labels = [str(row[x_key]) for row in rows]
    values = [float(row[y_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels or ["none"], values or [0.0], color="#2f6f8f")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def hist_plot(path: Path, values_by_channel: dict[str, list[float]], xlabel: str) -> None:
    plt = setup_matplotlib(path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for channel, values in sorted(values_by_channel.items()):
        clean = [v for v in values if math.isfinite(v) and v >= 0.0]
        if clean:
            ax.hist(clean, bins=min(20, max(4, len(clean))), alpha=0.55, label=channel)
            plotted = True
    if not plotted:
        ax.hist([0.0], bins=1, alpha=0.55, label="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def image_plot(path: Path, rows: list[dict[str, Any]], channel: str | None = None, rgb: bool = False, ratio: bool = False) -> None:
    import numpy as np

    plt = setup_matplotlib(path.parent)
    if not rows:
        arr = np.zeros((2, 2, 3 if rgb else 1))
    else:
        nx = max(int(fnum(row, "pixel_x")) for row in rows) + 1
        ny = max(int(fnum(row, "pixel_y")) for row in rows) + 1
        if rgb:
            arr = np.zeros((ny, nx, 3), dtype=float)
        else:
            arr = np.zeros((ny, nx), dtype=float)
        for row in rows:
            x = int(fnum(row, "pixel_x"))
            y = int(fnum(row, "pixel_y"))
            ch = channel_for(row)
            val = energy(row)
            if ratio:
                val = fnum(row, "weight_IIM", "iim_weight", default=1.0) / max(fnum(row, "weight_GBW", "gbw_weight", default=1.0), 1.0e-30)
            if rgb:
                idx = 0 if ch == "hadronic" else 1 if ch in {"gamma", "electromagnetic"} else 2 if ch == "neutrino" else None
                if idx is not None:
                    arr[y, x, idx] += val
            elif channel is None or ch == channel or (channel == "hadronic" and ch in {"hadron", "meson"}):
                arr[y, x] += val
        if rgb and arr.max() > 0:
            arr = arr / arr.max()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if rgb:
        ax.imshow(arr, origin="lower")
    else:
        im = ax.imshow(arr, origin="lower")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def group_rows(rows: list[dict[str, Any]], value_fn: Callable[[dict[str, Any]], float] | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (channel_for(row), pdg(row))
        item = grouped.setdefault(key, {"channel": key[0], "pdg_id": key[1], "count": 0, "energy_gev": 0.0})
        item["count"] += 1
        item["energy_gev"] += value_fn(row) if value_fn else energy(row)
    return sorted(grouped.values(), key=lambda row: (row["channel"], row["pdg_id"]))


def energy_budget_rows(summary: list[dict[str, Any]], ready: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = defaultdict(float)
    for row in summary:
        for key in ["deposited_energy", "deposited_energy_gev", "escaped_energy", "escaped_energy_gev", "invisible_energy", "invisible_energy_gev", "untracked_energy", "untracked_energy_gev", "unsupported_uhe_energy"]:
            value = fnum(row, key, default=math.nan)
            if math.isfinite(value):
                label = key.replace("_gev", "").replace("_energy", "")
                totals[label] += value
    if not totals:
        for row in ready:
            status = str(row.get("final_status", row.get("status", "escaped"))).lower()
            label = "escaped" if "escap" in status or "ready" in status else "untracked"
            totals[label] += energy(row)
    return [{"component": key, "energy_gev": value} for key, value in sorted(totals.items())]


def ratio_rows(gbw_iim: list[dict[str, Any]], observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in gbw_iim:
        channel = str(row.get("channel", row.get("model", "all")))
        gbw = fnum(row, "GBW", "gbw", "energy_GBW", "weighted_energy_GBW", "gbw_weighted_energy_gev", default=math.nan)
        iim = fnum(row, "IIM", "iim", "energy_IIM", "weighted_energy_IIM", "iim_weighted_energy_gev", default=math.nan)
        if math.isfinite(gbw) and math.isfinite(iim):
            rows.append({"channel": channel, "iim_over_gbw": iim / max(gbw, 1.0e-30)})
    if rows:
        return rows
    by_channel = defaultdict(lambda: {"gbw": 0.0, "iim": 0.0})
    for row in observed:
        ch = channel_for(row)
        base = energy(row)
        by_channel[ch]["gbw"] += base * fnum(row, "weight_GBW", "gbw_weight", default=1.0)
        by_channel[ch]["iim"] += base * fnum(row, "weight_IIM", "iim_weight", default=1.0)
    return [{"channel": ch, "iim_over_gbw": vals["iim"] / max(vals["gbw"], 1.0e-30)} for ch, vals in sorted(by_channel.items())]


def build_plot(spec_row: dict[str, Any], sources: dict[str, list[dict[str, Any]]], science_dir: Path) -> tuple[bool, str]:
    plot_path = science_dir / spec_row["filename"]
    data_path = science_dir / "data" / spec_row["filename"].replace(".png", ".csv")
    plot_id = spec_row["plot_id"]
    produced = sources["produced_csv"] or sources["produced_jsonl"]
    ready = sources["geant4_particles"] or sources["geant4_ready"]
    observed = sources["observed"]
    gbw_iim = sources["gbw_iim"]

    try:
        if plot_id in {"01", "02", "03"}:
            if not observed:
                write_csv(data_path, [{"missing_reason": "missing particle_ray_association_camera.csv or legacy observed_particles_by_pixel.csv"}])
                return False, "missing particle-ray association camera rows"
            write_csv(data_path, observed)
            channel = {"02": "gamma", "03": "hadronic"}.get(plot_id)
            image_plot(plot_path, observed, channel=channel, rgb=plot_id == "01")
        elif plot_id == "04":
            if not produced:
                write_csv(data_path, [{"missing_reason": "missing produced particle data"}])
                return False, "missing hadros_particle_events.jsonl or powheg_pythia_particles.csv"
            rows = group_rows(produced)
            write_csv(data_path, rows)
            bar_plot(plot_path, rows, "channel", "energy_gev", "energy [GeV]")
        elif plot_id == "05":
            if not ready:
                write_csv(data_path, [{"missing_reason": "missing GEANT4 survivor data"}])
                return False, "missing GEANT4 survivor data"
            rows = group_rows(ready)
            write_csv(data_path, rows)
            bar_plot(plot_path, rows, "channel", "energy_gev", "energy [GeV]")
        elif plot_id == "06":
            if not observed:
                write_csv(data_path, [{"missing_reason": "missing particle_ray_association_camera.csv or legacy observed_particles_by_pixel.csv"}])
                return False, "missing particle-ray association camera rows"
            rows = group_rows(observed)
            write_csv(data_path, rows)
            bar_plot(plot_path, rows, "channel", "energy_gev", "energy [GeV]")
        elif plot_id == "07":
            if not observed:
                write_csv(data_path, [{"missing_reason": "missing particle_ray_association_camera.csv or legacy observed_particles_by_pixel.csv"}])
                return False, "missing particle-ray association camera rows"
            values = defaultdict(list)
            for row in observed:
                values[channel_for(row)].append(energy(row))
            write_csv(data_path, [{"channel": ch, "energy_gev": v} for ch, vals in values.items() for v in vals])
            hist_plot(plot_path, values, "associated secondary energy [GeV]")
        elif plot_id == "08":
            rows = energy_budget_rows(sources["geant4_summary"], sources["geant4_ready"])
            if not rows:
                write_csv(data_path, [{"missing_reason": "missing GEANT4 summary data"}])
                return False, "missing powheg_pythia_geant4_resumable_summary.csv"
            write_csv(data_path, rows)
            bar_plot(plot_path, rows, "component", "energy_gev", "energy [GeV]")
        elif plot_id in {"09", "10"}:
            if not gbw_iim and not observed:
                write_csv(data_path, [{"missing_reason": "missing GBW/IIM weighted outputs"}])
                return False, "missing GBW/IIM weighted outputs"
            rows = ratio_rows(gbw_iim, observed)
            if not rows:
                write_csv(data_path, [{"missing_reason": "missing usable GBW/IIM ratio columns"}])
                return False, "missing usable GBW/IIM ratio columns"
            write_csv(data_path, rows)
            if plot_id == "09":
                bar_plot(plot_path, rows, "channel", "iim_over_gbw", "IIM / GBW")
            else:
                image_plot(plot_path, observed, ratio=True)
        else:
            return False, "unknown plot id"
    except Exception as exc:
        write_csv(data_path, [{"missing_reason": f"plot generation failed: {exc}"}])
        return False, f"plot generation failed: {exc}"
    return True, ""


def enabled_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = {
        "production": args.production_plots,
        "geant4": args.geant4_plots,
        "observed": args.observed_kerr_plots,
        "gbw_iim": args.gbw_iim_plots,
    }
    if args.dis_model.lower() != "both":
        groups["gbw_iim"] = False
    return [row for row in PLOT_SPECS if groups.get(row["group"], True)]


def build_bundle(args: argparse.Namespace) -> list[dict[str, Any]]:
    science_dir = args.run_dir / "plots" / "science"
    (science_dir / "data").mkdir(parents=True, exist_ok=True)
    sources = load_sources(args.run_dir)
    manifest = []
    for row in enabled_specs(args):
        generated, missing = build_plot(row, sources, science_dir)
        manifest.append(
            {
                "plot_id": row["plot_id"],
                "filename": row["filename"],
                "exists": str(bool(generated)).lower(),
                "generated": str(bool(generated)),
                "category": row["category"],
                "importance": row["importance"],
                "visibility": row["visibility"],
                "source_data": ";".join(row["source_data"]),
                "description": row["description"],
                "scientific_claim": row["scientific_claim"],
                "missing_reason": missing,
            }
        )
    write_csv(science_dir / "plot_bundle_manifest.csv", manifest)
    lines = [
        "# Scientific Plot Bundle Manifest",
        "",
        "| plot_id | filename | exists | missing_reason |",
        "|---|---|---|---|",
    ]
    for row in manifest:
        lines.append(f"| {row['plot_id']} | `{row['filename']}` | {row['exists']} | {row['missing_reason']} |")
    (science_dir / "plot_bundle_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing_rows = [row for row in manifest if row["exists"] != "true"]
    if missing_rows:
        missing_lines = [
            "# Missing Paper-Ready Science Figures",
            "",
            "| plot_id | filename | missing_reason |",
            "|---|---|---|",
        ]
        for row in missing_rows:
            missing_lines.append(f"| {row['plot_id']} | `{row['filename']}` | {row['missing_reason']} |")
    else:
        missing_lines = ["# Missing Paper-Ready Science Figures", "", "All mandatory paper-ready science figures were generated."]
    (science_dir / "missing_science_figures.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dis-model", default="both", choices=["GBW", "IIM", "both", "gbw", "iim"])
    parser.add_argument("--production-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geant4-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--particle-ray-association-plots", "--observed-kerr-plots", dest="observed_kerr_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gbw-iim-plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    manifest = build_bundle(args)
    generated = sum(row["generated"] == "True" for row in manifest)
    print(f"Scientific plot bundle manifest: {args.run_dir / 'plots/science/plot_bundle_manifest.csv'}")
    print(f"Generated plots: {generated}/{len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
