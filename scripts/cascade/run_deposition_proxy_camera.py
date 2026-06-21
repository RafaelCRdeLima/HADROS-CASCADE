#!/usr/bin/env python3
"""Run the experimental deposition-proxy camera mode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read_raw(path: Path, nx: int, ny: int) -> dict[str, np.ndarray]:
    image = np.zeros((nx, ny), dtype=float)
    ok = np.zeros((nx, ny), dtype=float)
    out = np.zeros((nx, ny), dtype=float)
    low = np.zeros((nx, ny), dtype=float)
    coverage = np.zeros((nx, ny), dtype=float)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            i = int(row["i"])
            j = int(row["j"])
            image[i, j] = float(row["I_proxy"])
            ok[i, j] = float(row["ok_queries"])
            out[i, j] = float(row["out_of_range_queries"])
            low[i, j] = float(row["low_coverage_queries"])
            coverage[i, j] = float(row["mean_coverage"])
    return {"image": image, "ok": ok, "out": out, "low": low, "coverage": coverage}


def write_outputs(output_dir: Path, arrays: dict[str, np.ndarray], stats: dict, manifest: dict | None, suffix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"deposition_proxy_camera{suffix}_image.npz",
        image=arrays["image"],
        ok_queries=arrays["ok"],
        out_of_range_queries=arrays["out"],
        low_coverage_queries=arrays["low"],
        mean_coverage=arrays["coverage"],
        stats=json.dumps(stats, sort_keys=True),
    )

    lines = [
        "# Deposition Proxy Camera Summary",
        "",
        "Experimental emissivity proxy, not physical luminosity.",
        "This is not radiative microphysics, not radiative transfer, and not a final observable image.",
        "",
        f"- emissivity_mode: `{stats.get('emissivity_mode')}`",
        f"- h5_path: `{stats.get('h5_path')}`",
        f"- manifest_path: `{stats.get('manifest_path')}`",
        f"- total_weighted_deposited_energy_gev: `{stats.get('total_weighted_deposited_energy_gev')}`",
        f"- normalization: `{(manifest or {}).get('normalization', 'unknown')}`",
        f"- ok_queries: `{stats.get('ok_queries')}`",
        f"- out_of_range_queries: `{stats.get('out_of_range_queries')}`",
        f"- low_coverage_queries: `{stats.get('low_coverage_queries')}`",
        f"- mean_coverage: `{stats.get('mean_coverage')}`",
        f"- image_sum: `{stats.get('image_sum')}`",
        "",
        "OUT_OF_RANGE samples contribute zero emissivity in this experimental mode.",
    ]
    (output_dir / f"deposition_proxy_camera{suffix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(arrays["image"].T, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="I_proxy")
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    ax.set_title("deposition proxy camera")
    fig.tight_layout()
    fig.savefig(plots / f"deposition_proxy_camera{suffix}.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissivity-mode", choices=["default", "deposition_proxy"], default="default")
    parser.add_argument("--deposition-emissivity-h5", type=Path)
    parser.add_argument("--deposition-emissivity-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--r-max", type=float, default=120.0)
    parser.add_argument("--auto-frame-deposition-field", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("h5c++") is None:
        print("skipped: h5c++ not available")
        return 0
    if args.emissivity_mode == "deposition_proxy":
        if args.deposition_emissivity_h5 is None or args.deposition_emissivity_manifest is None:
            print("deposition_proxy requires --deposition-emissivity-h5 and --deposition-emissivity-manifest", file=sys.stderr)
            return 2
    subprocess.run(["make", "compute_deposition_proxy_camera", "HADROS_WITH_HDF5=ON"], cwd=ROOT, check=True)
    cmd = [
        str(ROOT / "build" / "compute_deposition_proxy_camera"),
        "--emissivity-mode",
        args.emissivity_mode,
        "--output-dir",
        str(args.output_dir),
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--step",
        str(args.step),
        "--r-max",
        str(args.r_max),
    ]
    if args.emissivity_mode == "deposition_proxy":
        cmd.extend([
            "--deposition-emissivity-h5",
            str(args.deposition_emissivity_h5),
            "--deposition-emissivity-manifest",
            str(args.deposition_emissivity_manifest),
        ])
    if args.auto_frame_deposition_field:
        cmd.append("--auto-frame-deposition-field")
    subprocess.run(cmd, cwd=ROOT, check=True)
    raw = args.output_dir / "deposition_proxy_camera_raw.csv"
    stats = json.loads((args.output_dir / "deposition_proxy_camera_stats.json").read_text(encoding="utf-8"))
    manifest = None
    if args.deposition_emissivity_manifest and args.deposition_emissivity_manifest.exists():
        manifest = json.loads(args.deposition_emissivity_manifest.read_text(encoding="utf-8"))
    arrays = read_raw(raw, args.nx, args.ny)
    suffix = "_autoframe" if args.auto_frame_deposition_field else ""
    write_outputs(args.output_dir, arrays, stats, manifest, suffix=suffix)
    print(json.dumps({
        "image": str(args.output_dir / f"deposition_proxy_camera{suffix}_image.npz"),
        "summary": str(args.output_dir / f"deposition_proxy_camera{suffix}_summary.md"),
        "plot": str(args.output_dir / "plots" / f"deposition_proxy_camera{suffix}.png"),
        "ok_queries": stats.get("ok_queries"),
        "out_of_range_queries": stats.get("out_of_range_queries"),
        "image_sum": stats.get("image_sum"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
