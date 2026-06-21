#!/usr/bin/env python3
"""DEPRECATED / DEBUG ONLY: legacy sampled interaction-point approximation.

This file is not part of the final scientific HADROS chain.
Do not use for scientific production.

Final Phase 15.8+ production uses real HADROS UHE Kerr-ray samples through
``build_uhe_ray_event_link.py`` and ``INCOMING_KERR_GEODESIC_COLUMN``.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output" / "science" / "powheg_pythia_particles" / "hadros_particle_events.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "science" / "powheg_pythia_particles" / "interaction_points.jsonl"
DEFAULT_CONFIG = ROOT / "config.ini"
STATUS = "GLOBAL_INTERACTION_POSITION_SAMPLED_APPROXIMATION"
COLUMN_MODEL_FALLBACK = "SOURCE_TO_INTERACTION_RAY_APPROXIMATION"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fget(cfg: configparser.ConfigParser, section: str, key: str, default: float) -> float:
    try:
        value = cfg.get(section, key, fallback="")
        return float(value) if str(value).strip() else default
    except ValueError:
        return default


def sget(cfg: configparser.ConfigParser, section: str, key: str, default: str) -> str:
    value = cfg.get(section, key, fallback=default)
    return str(value).strip() or default


def _set_if_present(cfg: configparser.ConfigParser, section: str, key: str, value: Any) -> None:
    if value is None:
        return
    if not cfg.has_section(section):
        cfg.add_section(section)
    cfg.set(section, key, str(value))


def _load_json_config(path: Path) -> configparser.ConfigParser:
    """Map config-web JSON into the INI-shaped sections used by this sampler."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = configparser.ConfigParser()
    values = data.get("config_web_values", {})
    if isinstance(values, dict):
        for section, section_values in values.items():
            if not isinstance(section_values, dict):
                continue
            if not cfg.has_section(section):
                cfg.add_section(section)
            for key, value in section_values.items():
                if value is not None:
                    cfg.set(section, key, str(value))

    _set_if_present(cfg, "black_hole", "ASPIN", data.get("spin", data.get("ASPIN")))
    _set_if_present(cfg, "black_hole", "MBH_MSUN", data.get("mbh_msun", data.get("MBH_MSUN")))
    _set_if_present(cfg, "camera", "CAM_FOV_DEG", data.get("camera_fov_deg"))
    _set_if_present(cfg, "camera", "CAM_NX", data.get("camera_nx"))
    _set_if_present(cfg, "camera", "CAM_NY", data.get("camera_ny"))
    _set_if_present(cfg, "density_profile", "FUNNEL_THETA_DEG", data.get("funnel_theta_deg", data.get("cone_deg")))
    _set_if_present(cfg, "uhe_source", "SOURCE_R_RG", data.get("source_r_rg"))
    _set_if_present(cfg, "uhe_source", "SOURCE_SIGMA_RG", data.get("source_sigma_rg"))
    _set_if_present(cfg, "uhe_source", "SOURCE_FUNNEL_THETA_DEG", data.get("source_funnel_theta_deg"))
    return cfg


def load_config(path: Path) -> tuple[configparser.ConfigParser, str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return _load_json_config(path), "json"
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg, "ini"


def source_theta_rad(source_model: str, funnel_theta_deg: float) -> float:
    if source_model == "funnel_wall":
        # Existing config-web preview interprets funnel wall as polar angle
        # theta = 90 deg - half-opening angle.
        return math.radians(max(1.0, min(179.0, 90.0 - funnel_theta_deg)))
    return math.radians(max(1.0, min(179.0, funnel_theta_deg)))


def density_at_point(cfg: configparser.ConfigParser, r_rg: float, theta_rad: float) -> tuple[float, str]:
    if cfg.get("tabulated_funnel", "TABULATED_FUNNEL_ENABLED", fallback="0").strip() == "1":
        rho_wall = fget(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RHO_WALL", 1.0e3)
        r_in = max(fget(cfg, "tabulated_funnel", "TABULATED_FUNNEL_R_IN_RG", 1.5), 1.0e-6)
        power = fget(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RADIAL_POWER", 2.0)
        density = rho_wall * (max(r_rg, r_in) / r_in) ** (-power)
        return density, "tabulated_funnel_wall_powerlaw"
    rho0 = fget(cfg, "torus", "TORUS_RHO0", 1.0e8)
    r0 = max(fget(cfg, "torus", "TORUS_R0_RG", 10.0), 1.0e-6)
    power = fget(cfg, "torus", "TORUS_RADIAL_POWER", 2.0)
    return rho0 * (max(r_rg, r0) / r0) ** (-power), "analytic_torus_powerlaw"


def unit_vector(dx: float, dy: float, dz: float) -> tuple[float | None, float | None, float | None]:
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 0.0 or not math.isfinite(norm):
        return None, None, None
    return dx / norm, dy / norm, dz / norm


def event_ids_from_particles(path: Path) -> list[int]:
    events = sorted({int(row["event_id"]) for row in read_jsonl(path)})
    if not events:
        raise RuntimeError(f"No event_id values found in {path}")
    return events


def sample_points(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, config_format = load_config(args.config)
    events = event_ids_from_particles(args.input)

    spin = fget(cfg, "black_hole", "ASPIN", 0.8)
    source_model = sget(cfg, "uhe_source", "SOURCE_MODEL", "funnel_wall")
    source_r = fget(cfg, "uhe_source", "SOURCE_R_RG", 3.5)
    source_sigma = max(fget(cfg, "uhe_source", "SOURCE_SIGMA_RG", 1.0), 1.0e-6)
    funnel_theta = fget(cfg, "uhe_source", "SOURCE_FUNNEL_THETA_DEG", fget(cfg, "density_profile", "FUNNEL_THETA_DEG", 20.0))
    camera_fov = fget(cfg, "camera", "CAM_FOV_DEG", 60.0)
    horizon = 1.0 + math.sqrt(max(1.0 - spin * spin, 0.0))
    theta0 = source_theta_rad(source_model, funnel_theta)
    theta_sigma = math.radians(max(0.25, min(5.0, funnel_theta * 0.15)))
    # Visible-sector azimuth sampling: not a fixed origin, but an explicit
    # camera-visible approximation so the real Kerr camera can audit matches.
    phi_sigma = math.radians(max(2.0, min(20.0, camera_fov * 0.25)))
    r_min = max(horizon + 0.25, fget(cfg, "tabulated_funnel", "TABULATED_FUNNEL_R_IN_RG", 1.5))
    r_max = max(r_min + 1.0, fget(cfg, "tabulated_funnel", "TABULATED_FUNNEL_R_OUT_RG", 120.0))
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for event_id in events:
        local = random.Random(rng.randrange(2**63) ^ event_id)
        r = min(r_max, max(r_min, local.lognormvariate(math.log(max(source_r, r_min)), source_sigma / max(source_r, 1.0))))
        theta = min(math.pi - 1.0e-6, max(1.0e-6, local.gauss(theta0, theta_sigma)))
        phi = local.gauss(0.0, phi_sigma)
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        density, density_model = density_at_point(cfg, r, theta)
        weight_position = max(density, 0.0) * r * r * max(math.sin(theta), 0.0)
        source_x = source_r * math.sin(theta0)
        source_y = 0.0
        source_z = source_r * math.cos(theta0)
        dir_x, dir_y, dir_z = unit_vector(x - source_x, y - source_y, z - source_z)
        rows.append(
            {
                "event_id": event_id,
                "interaction_x_rg": x,
                "interaction_y_rg": y,
                "interaction_z_rg": z,
                "interaction_r_rg": r,
                "interaction_theta_rad": theta,
                "interaction_phi_rad": phi,
                "position_status": STATUS,
                "interaction_position_status": "GLOBAL_POSITION_VALID",
                "sampling_backend": "HADROS_FUNNEL_WALL_VISIBLE_SECTOR_SAMPLED_APPROXIMATION",
                "density_model": density_model,
                "source_model": source_model,
                "density_g_cm3": density,
                "weight_position": weight_position,
                "incoming_ray_id": None,
                "incoming_ray_pixel_x": None,
                "incoming_ray_pixel_y": None,
                "incoming_geodesic_sample_index": None,
                "incoming_geodesic_lambda": None,
                "incoming_neutrino_direction_global": None if dir_x is None else [dir_x, dir_y, dir_z],
                "column_before_cm2": None,
                "column_model": COLUMN_MODEL_FALLBACK,
                "tau_before_GBW": None,
                "tau_before_IIM": None,
                "Pint_GBW": None,
                "Pint_IIM": None,
                "column_integration_status": "FALLBACK_PENDING_REWEIGHTING",
                "n_samples": 0,
                "dl_total_cm": None,
                "density_profile_used": density_model,
            }
        )
    summary = {
        "events": len(events),
        "status": STATUS,
        "source_model": source_model,
        "density_models": dict(Counter(row["density_model"] for row in rows)),
        "spin": spin,
        "horizon_r_plus_rg": horizon,
        "r_min_rg": min(row["interaction_r_rg"] for row in rows),
        "r_max_rg": max(row["interaction_r_rg"] for row in rows),
        "theta_min_rad": min(row["interaction_theta_rad"] for row in rows),
        "theta_max_rad": max(row["interaction_theta_rad"] for row in rows),
        "seed": args.seed,
        "config_format": config_format,
    }
    return rows, summary


def write_summary(path: Path, summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# POWHEG Global Interaction Points",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"- output: `{output}`",
        f"- events: `{summary['events']}`",
        f"- source_model: `{summary['source_model']}`",
        f"- sampling_backend: `HADROS_FUNNEL_WALL_VISIBLE_SECTOR_SAMPLED_APPROXIMATION`",
        f"- density_models: `{summary['density_models']}`",
        f"- horizon_r_plus_rg: `{summary['horizon_r_plus_rg']}`",
        f"- r_range_rg: `{summary['r_min_rg']}` - `{summary['r_max_rg']}`",
        f"- theta_range_rad: `{summary['theta_min_rad']}` - `{summary['theta_max_rad']}`",
        f"- config_format: `{summary['config_format']}`",
        "",
        "No fixed origin, r=1/theta=0 fallback, or particle-to-screen projection is used.",
        "The sampler is an approximation and must not be promoted to full validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=ROOT / "output/science/powheg_pythia_particles/interaction_points_summary.md")
    parser.add_argument("--seed", type=int, default=154)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_absolute():
        args.input = (ROOT / args.input).resolve()
    if not args.output.is_absolute():
        args.output = (ROOT / args.output).resolve()
    if not args.config.is_absolute():
        args.config = (ROOT / args.config).resolve()
    rows, summary = sample_points(args)
    write_jsonl(args.output, rows)
    write_summary(args.summary, summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
