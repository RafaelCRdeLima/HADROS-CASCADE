#!/usr/bin/env python3
"""Sample final-chain interaction points directly on HADROS Kerr rays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from build_gbw_iim_real_kerr_reweighting import RG_CM_PER_MSUN, load_hadros_config, read_sigma_table, spherical_from_xyz
from build_uhe_ray_event_link import load_ray_samples


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBW = ROOT / "data/sigma/sigma_nuN_CC_GBW.dat"
DEFAULT_IIM = ROOT / "data/sigma/sigma_nuN_CC_IIM.dat"
STATUS = "FINAL_CHAIN_INTERACTION_POINT_ON_KERR_GEODESIC_SAMPLE"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def finite_positive(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) and out > 0.0 else default


def select_samples(samples: list[dict[str, Any]], n_events: int, seed: int) -> list[dict[str, Any]]:
    usable = [row for row in samples if not int(row.get("captured", 0) or 0) and finite_positive(row.get("column_before_cm2"), 0.0) > 0.0]
    if len(usable) < n_events:
        usable = [row for row in samples if finite_positive(row.get("column_before_cm2"), 0.0) > 0.0]
    if not usable:
        raise RuntimeError("No usable Kerr-ray samples with positive incoming column were found.")
    stride = max(1, len(usable) // max(n_events, 1))
    start = seed % len(usable)
    return [usable[(start + idx * stride) % len(usable)] for idx in range(n_events)]


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_hadros_config(args.config)
    rg_cm = RG_CM_PER_MSUN * args.mbh_msun
    sigma_gbw = read_sigma_table(args.sigma_gbw)
    sigma_iim = read_sigma_table(args.sigma_iim)
    samples, cache_summary = load_ray_samples(args.geodesic_cache, cfg, rg_cm, sigma_gbw, sigma_iim, args.reference_energy_gev)
    selected = select_samples(samples, args.n_events, args.seed)
    rows: list[dict[str, Any]] = []
    for event_id, sample in enumerate(selected, start=1):
        x, y, z = sample["x"], sample["y"], sample["z"]
        r, theta, phi = spherical_from_xyz(x, y, z)
        rows.append(
            {
                "event_id": event_id,
                "incoming_ray_id": sample["incoming_ray_id"],
                "geodesic_cache_ray_id": sample["geodesic_cache_ray_id"],
                "pixel_x": sample["pixel_x"],
                "pixel_y": sample["pixel_y"],
                "nx": sample["nx"],
                "ny": sample["ny"],
                "ray_id": sample["ray_id"],
                "ray_id_convention": sample["ray_id_convention"],
                "incoming_ray_pixel_x": sample["pixel_x"],
                "incoming_ray_pixel_y": sample["pixel_y"],
                "source_pixel_x": sample["pixel_x"],
                "source_pixel_y": sample["pixel_y"],
                "ray_sample_index": sample["ray_sample_index"],
                "incoming_geodesic_sample_index": sample["ray_sample_index"],
                "lambda": sample["lambda"],
                "incoming_geodesic_lambda": sample["lambda"],
                "interaction_x_rg": x,
                "interaction_y_rg": y,
                "interaction_z_rg": z,
                "interaction_r_rg": r,
                "interaction_theta_rad": theta,
                "interaction_phi_rad": phi,
                "position_status": STATUS,
                "interaction_position_status": "GLOBAL_POSITION_VALID",
                "sampling_backend": "HADROS_KERR_GEODESIC_SAMPLE",
                "column_before_cm2": sample["column_before_cm2"],
                "redshift_factor": sample["redshift_factor"],
                "E_nu_inf_GeV": sample["E_nu_inf_GeV"],
                "E_nu_local_GeV": sample["E_nu_local_GeV"],
                "sigma_GBW_cm2": sample["sigma_GBW_cm2"],
                "sigma_IIM_cm2": sample["sigma_IIM_cm2"],
                "column_model": "INCOMING_KERR_GEODESIC_COLUMN",
                "column_integration_status": "INCOMING_KERR_GEODESIC_COLUMN_INTEGRATED",
                "tau_before_GBW": sample["tau_before_GBW"],
                "tau_before_IIM": sample["tau_before_IIM"],
                "Pint_GBW": sample["Pint_GBW"],
                "Pint_IIM": sample["Pint_IIM"],
                "interaction_weight": sample["Pint_GBW"],
                "interaction_weight_GBW": sample["Pint_GBW"],
                "interaction_weight_IIM": sample["Pint_IIM"],
                "ray_link_status": "REAL_HADROS_UHE_RAY_SAMPLE_LINKED",
                "n_samples": sample["ray_sample_index"] + 1,
                "dl_total_cm": sample["lambda"] * rg_cm,
                "density_g_cm3": sample["density_g_cm3"],
                "density_profile_used": sample["density_profile_used"],
            }
        )
    return rows, cache_summary


def write_summary(path: Path, rows: list[dict[str, Any]], cache_summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Final Geodesic Interaction Points",
        "",
        f"Status: `{STATUS}`.",
        "",
        f"- events: `{len(rows)}`",
        f"- geodesic_cache: `{args.geodesic_cache}`",
        f"- cache_samples: `{cache_summary['samples']}`",
        f"- reference_energy_gev: `{args.reference_energy_gev}`",
        "- column_model: `INCOMING_KERR_GEODESIC_COLUMN`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geodesic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linked-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mbh-msun", type=float, default=2.0)
    parser.add_argument("--reference-energy-gev", type=float, default=1.0e9)
    parser.add_argument("--sigma-gbw", type=Path, default=DEFAULT_GBW)
    parser.add_argument("--sigma-iim", type=Path, default=DEFAULT_IIM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, cache_summary = build_rows(args)
    write_jsonl(args.output, rows)
    write_jsonl(args.linked_output, rows)
    write_summary(args.summary, rows, cache_summary, args)
    print(json.dumps({"status": STATUS, "events": len(rows), "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
