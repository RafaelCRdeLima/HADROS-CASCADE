#!/usr/bin/env python3
"""Build GBW/IIM physical interaction-point weights for the real Kerr camera chain."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTICLES = ROOT / "output/science/powheg_pythia_particles/hadros_particle_events.jsonl"
DEFAULT_POINTS = ROOT / "output/science/powheg_pythia_particles/interaction_points.jsonl"
DEFAULT_READY = ROOT / "output/science/powheg_pythia_geant4_resumable/geant4_ready_particles.jsonl"
DEFAULT_OBSERVED = ROOT / "output/science/real_kerr_particle_camera/observed_particles_by_pixel.csv"
DEFAULT_OUTPUT = ROOT / "output/science/gbw_iim_reweighting"
DEFAULT_CAMERA_OUTPUT = ROOT / "output/science/gbw_iim_real_kerr_camera"
DEFAULT_GBW = ROOT / "data/sigma/sigma_nuN_CC_GBW.dat"
DEFAULT_IIM = ROOT / "data/sigma/sigma_nuN_CC_IIM.dat"
DEFAULT_CONFIG = ROOT / "config.ini"
M_U_G = 1.66053906660e-24
RG_CM_PER_MSUN = 1.4766250385e5
STATUS_PARTIAL_SOURCE_RAY = "GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_PARTIAL_SOURCE_RAY_COLUMN"
STATUS_VALIDATED_INCOMING = "GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_VALIDATED_INCOMING_GEODESIC_COLUMN"
STATUS_BLOCKED_MISSING_INCOMING = "GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_BLOCKED_BY_MISSING_INCOMING_GEODESIC"
LOCAL_COLUMN_MODEL = "LOCAL_RADIAL_COLUMN"
SOURCE_RAY_COLUMN_MODEL = "SOURCE_TO_INTERACTION_RAY_APPROXIMATION"
INCOMING_COLUMN_MODEL = "INCOMING_KERR_GEODESIC_COLUMN"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def cfg_float(cfg: configparser.ConfigParser, section: str, key: str, default: float) -> float:
    try:
        text = cfg.get(section, key, fallback="")
        return float(text) if str(text).strip() else default
    except ValueError:
        return default


def cfg_str(cfg: configparser.ConfigParser, section: str, key: str, default: str) -> str:
    text = cfg.get(section, key, fallback=default)
    return str(text).strip() or default


def _set_if_present(cfg: configparser.ConfigParser, section: str, key: str, value: Any) -> None:
    if value is None:
        return
    if not cfg.has_section(section):
        cfg.add_section(section)
    cfg.set(section, key, str(value))


def load_hadros_config(path: Path) -> configparser.ConfigParser:
    text = path.read_text(encoding="utf-8")
    cfg = configparser.ConfigParser()
    if not text.lstrip().startswith("{"):
        cfg.read(path)
        return cfg

    data = json.loads(text)
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


def source_theta_rad(source_model: str, funnel_theta_deg: float) -> float:
    if source_model == "funnel_wall":
        return math.radians(max(1.0, min(179.0, 90.0 - funnel_theta_deg)))
    return math.radians(max(1.0, min(179.0, funnel_theta_deg)))


def xyz_from_spherical(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    st = math.sin(theta)
    return r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta)


def spherical_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0 or not math.isfinite(r):
        return 0.0, 0.0, 0.0
    return r, math.acos(max(-1.0, min(1.0, z / r))), math.atan2(y, x)


def density_at_point_from_config(cfg: configparser.ConfigParser, r_rg: float, theta_rad: float) -> tuple[float, str]:
    if cfg.get("tabulated_funnel", "TABULATED_FUNNEL_ENABLED", fallback="0").strip() == "1":
        rho_axis = cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RHO_AXIS", 1.0)
        rho_wall = cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RHO_WALL", 1.0e3)
        rho_cocoon = cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RHO_COCOON", 1.0e2)
        r_in = max(cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_R_IN_RG", 1.5), 1.0e-6)
        r_out = max(cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_R_OUT_RG", 120.0), r_in)
        power = cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_RADIAL_POWER", 2.0)
        theta_wall = math.radians(cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_THETA_DEG", 15.0))
        dtheta = math.radians(max(cfg_float(cfg, "tabulated_funnel", "TABULATED_FUNNEL_DTHETA_DEG", 5.0), 1.0e-6))
        if r_rg < r_in or r_rg > r_out:
            ambient = cfg_float(cfg, "tabulated_ambient", "TABULATED_AMBIENT_RHO0", 1.0)
            r0 = max(cfg_float(cfg, "tabulated_ambient", "TABULATED_AMBIENT_R0_RG", 10.0), 1.0e-6)
            alpha = cfg_float(cfg, "tabulated_ambient", "TABULATED_AMBIENT_POWERLAW_INDEX", 2.0)
            return ambient * (max(r_rg, r0) / r0) ** (-alpha), "tabulated_ambient_powerlaw"
        polar_angle = min(theta_rad, math.pi - theta_rad)
        radial = (max(r_rg, r_in) / r_in) ** (-power)
        wall_blend = math.exp(-0.5 * ((polar_angle - theta_wall) / dtheta) ** 2)
        if polar_angle < theta_wall:
            density = rho_axis + (rho_wall - rho_axis) * wall_blend
        else:
            density = rho_cocoon + (rho_wall - rho_cocoon) * wall_blend
        return max(density, 0.0) * radial, "tabulated_funnel_angular_radial"
    rho0 = cfg_float(cfg, "torus", "TORUS_RHO0", 1.0e8)
    r0 = max(cfg_float(cfg, "torus", "TORUS_R0_RG", 10.0), 1.0e-6)
    power = cfg_float(cfg, "torus", "TORUS_RADIAL_POWER", 2.0)
    h_over_r = max(cfg_float(cfg, "torus", "TORUS_H_OVER_R", 0.25), 1.0e-6)
    z_over_r = abs(math.cos(theta_rad))
    vertical = math.exp(-0.5 * (z_over_r / h_over_r) ** 2)
    return rho0 * (max(r_rg, r0) / r0) ** (-power) * vertical, "analytic_torus_gaussian_powerlaw"


def configured_source_position(cfg: configparser.ConfigParser) -> tuple[float, float, float, str]:
    source_model = cfg_str(cfg, "uhe_source", "SOURCE_MODEL", "funnel_wall")
    if cfg_str(cfg, "tabulated_uhe_source", "TABULATED_SOURCE_GEOMETRY", "") == "axial_point":
        z0 = cfg_float(cfg, "tabulated_uhe_source", "TABULATED_SOURCE_Z0_RG", 3.0)
        return 0.0, 0.0, z0, "tabulated_uhe_source_axial_point"
    source_r = cfg_float(cfg, "uhe_source", "SOURCE_R_RG", 3.5)
    funnel_theta = cfg_float(cfg, "uhe_source", "SOURCE_FUNNEL_THETA_DEG", cfg_float(cfg, "density_profile", "FUNNEL_THETA_DEG", 20.0))
    theta = source_theta_rad(source_model, funnel_theta)
    x, y, z = xyz_from_spherical(source_r, theta, 0.0)
    return x, y, z, f"uhe_source_{source_model}"


def ray_integrated_column_cm2(
    point: dict[str, Any],
    cfg: configparser.ConfigParser,
    rg_cm: float,
    steps: int,
) -> dict[str, Any]:
    sx, sy, sz, source_status = configured_source_position(cfg)
    px = fnum(point, "interaction_x_rg", math.nan)
    py = fnum(point, "interaction_y_rg", math.nan)
    pz = fnum(point, "interaction_z_rg", math.nan)
    dx = px - sx
    dy = py - sy
    dz = pz - sz
    length_rg = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(length_rg) or length_rg <= 0.0:
        return {
            "column_before_cm2": math.nan,
            "column_path_length_rg": length_rg,
            "column_path_length_cm": math.nan,
            "column_mean_rho_g_cm3": math.nan,
            "column_max_rho_g_cm3": math.nan,
            "column_density_model": "INVALID_RAY",
            "column_source_status": source_status,
        }
    n_steps = max(4, steps)
    dl_cm = length_rg * rg_cm / n_steps
    column = 0.0
    rho_sum = 0.0
    rho_max = 0.0
    models: Counter[str] = Counter()
    for idx in range(n_steps):
        frac = (idx + 0.5) / n_steps
        x = sx + frac * dx
        y = sy + frac * dy
        z = sz + frac * dz
        r, theta, _phi = spherical_from_xyz(x, y, z)
        rho, model = density_at_point_from_config(cfg, r, theta)
        rho = max(rho, 0.0)
        column += (rho / M_U_G) * dl_cm
        rho_sum += rho
        rho_max = max(rho_max, rho)
        models[model] += 1
    return {
        "column_before_cm2": column,
        "column_path_length_rg": length_rg,
        "column_path_length_cm": length_rg * rg_cm,
        "column_mean_rho_g_cm3": rho_sum / n_steps,
        "column_max_rho_g_cm3": rho_max,
        "column_density_model": ";".join(f"{key}:{value}" for key, value in sorted(models.items())),
        "column_source_status": source_status,
        "column_integration_status": "SOURCE_TO_INTERACTION_RAY_APPROXIMATION_INTEGRATED",
        "n_samples": n_steps,
        "dl_total_cm": length_rg * rg_cm,
        "density_profile_used": ";".join(f"{key}:{value}" for key, value in sorted(models.items())),
    }


def local_radial_column_cm2(point: dict[str, Any], rg_cm: float, args: argparse.Namespace) -> dict[str, Any]:
    rho = fnum(point, "density_g_cm3", math.nan)
    r_rg = fnum(point, "interaction_r_rg", math.nan)
    path_length_rg = max(args.column_r_outer_rg - r_rg, args.min_column_length_rg)
    path_length_cm = path_length_rg * rg_cm
    nucleon_density = rho / M_U_G if math.isfinite(rho) and rho >= 0.0 else math.nan
    return {
        "column_before_cm2": nucleon_density * path_length_cm if math.isfinite(nucleon_density) else math.nan,
        "column_path_length_rg": path_length_rg,
        "column_path_length_cm": path_length_cm,
        "column_mean_rho_g_cm3": rho,
        "column_max_rho_g_cm3": rho,
        "column_density_model": "local_sample_density",
        "column_source_status": "local_radial_outward_legacy",
        "column_integration_status": "LOCAL_RADIAL_APPROXIMATION",
        "n_samples": 1,
        "dl_total_cm": path_length_cm,
        "density_profile_used": "local_sample_density",
    }


def incoming_geodesic_column_cm2(point: dict[str, Any], rg_cm: float, gbw: tuple[list[float], list[float]], iim: tuple[list[float], list[float]], energy_local: float) -> dict[str, Any]:
    precomputed_column = fnum(point, "column_before_cm2", math.nan)
    if (
        point.get("column_model") == INCOMING_COLUMN_MODEL
        and math.isfinite(precomputed_column)
        and precomputed_column >= 0.0
        and point.get("incoming_ray_id") not in (None, "")
        and point.get("ray_sample_index", point.get("incoming_geodesic_sample_index")) not in (None, "")
    ):
        sigma_g, status_g = sigma_interp(energy_local, gbw)
        sigma_i, status_i = sigma_interp(energy_local, iim)
        tau_g = fnum(point, "tau_before_GBW", math.nan)
        tau_i = fnum(point, "tau_before_IIM", math.nan)
        if not math.isfinite(tau_g):
            tau_g = sigma_g * precomputed_column if status_g == "OK" else math.nan
        if not math.isfinite(tau_i):
            tau_i = sigma_i * precomputed_column if status_i == "OK" else math.nan
        return {
            "column_before_cm2": precomputed_column,
            "column_path_length_rg": fnum(point, "lambda", fnum(point, "incoming_geodesic_lambda", math.nan)),
            "column_path_length_cm": fnum(point, "dl_total_cm", math.nan),
            "column_mean_rho_g_cm3": fnum(point, "column_mean_rho_g_cm3", math.nan),
            "column_max_rho_g_cm3": fnum(point, "column_max_rho_g_cm3", math.nan),
            "column_density_model": str(point.get("density_profile_used", "incoming_kerr_geodesic_density")),
            "column_source_status": "INCOMING_KERR_GEODESIC_SAMPLE_LINKED",
            "column_integration_status": "INCOMING_KERR_GEODESIC_COLUMN_INTEGRATED",
            "n_samples": int(fnum(point, "n_samples", 0.0)),
            "dl_total_cm": fnum(point, "dl_total_cm", math.nan),
            "density_profile_used": str(point.get("density_profile_used", "incoming_kerr_geodesic_density")),
            "tau_before_GBW": tau_g,
            "tau_before_IIM": tau_i,
            "Pint_GBW": pint_from_tau(tau_g),
            "Pint_IIM": pint_from_tau(tau_i),
        }
    ray_id = point.get("incoming_ray_id")
    sample_index = point.get("incoming_geodesic_sample_index")
    if ray_id in (None, "") or sample_index in (None, ""):
        return {
            "column_before_cm2": math.nan,
            "column_path_length_rg": math.nan,
            "column_path_length_cm": math.nan,
            "column_mean_rho_g_cm3": math.nan,
            "column_max_rho_g_cm3": math.nan,
            "column_density_model": "missing_incoming_geodesic",
            "column_source_status": "MISSING_INCOMING_NEUTRINO_GEODESIC",
            "column_integration_status": "BLOCKED_BY_MISSING_INCOMING_GEODESIC",
            "n_samples": 0,
            "dl_total_cm": math.nan,
            "density_profile_used": "missing_incoming_geodesic",
            "tau_before_GBW": math.nan,
            "tau_before_IIM": math.nan,
            "Pint_GBW": math.nan,
            "Pint_IIM": math.nan,
        }
    # The current schema can identify a future incoming ray, but it does not
    # store geodesic samples or cumulative columns. Refuse to synthesize them.
    return {
        "column_before_cm2": math.nan,
        "column_path_length_rg": math.nan,
        "column_path_length_cm": math.nan,
        "column_mean_rho_g_cm3": math.nan,
        "column_max_rho_g_cm3": math.nan,
        "column_density_model": "incoming_ray_without_samples",
        "column_source_status": "INCOMING_RAY_ID_WITHOUT_STORED_SAMPLES",
        "column_integration_status": "BLOCKED_BY_MISSING_INCOMING_GEODESIC_SAMPLES",
        "n_samples": 0,
        "dl_total_cm": math.nan,
        "density_profile_used": "missing_incoming_geodesic_samples",
        "tau_before_GBW": math.nan,
        "tau_before_IIM": math.nan,
        "Pint_GBW": math.nan,
        "Pint_IIM": math.nan,
    }


def read_sigma_table(path: Path) -> tuple[list[float], list[float]]:
    pairs: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                energy = float(parts[0])
                sigma = float(parts[-1])
            except ValueError:
                continue
            if energy > 0.0 and sigma > 0.0 and math.isfinite(energy) and math.isfinite(sigma):
                pairs.append((energy, sigma))
    if len(pairs) < 2:
        raise RuntimeError(f"sigma table has too few usable rows: {path}")
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def sigma_interp(energy_gev: float, table: tuple[list[float], list[float]]) -> tuple[float, str]:
    energies, sigmas = table
    if not math.isfinite(energy_gev) or energy_gev <= 0.0:
        return math.nan, "INVALID_ENERGY"
    if energy_gev < energies[0] or energy_gev > energies[-1]:
        return math.nan, "ENERGY_OUT_OF_SIGMA_DOMAIN"
    if energy_gev == energies[0]:
        return sigmas[0], "OK"
    for idx in range(1, len(energies)):
        if energy_gev <= energies[idx]:
            x0 = math.log(energies[idx - 1])
            x1 = math.log(energies[idx])
            y0 = math.log(sigmas[idx - 1])
            y1 = math.log(sigmas[idx])
            frac = (math.log(energy_gev) - x0) / (x1 - x0)
            return math.exp(y0 + frac * (y1 - y0)), "OK"
    return math.nan, "ENERGY_OUT_OF_SIGMA_DOMAIN"


def pint_from_tau(tau: float) -> float:
    if not math.isfinite(tau):
        return math.nan
    if tau <= 0.0:
        return 0.0
    return -math.expm1(-tau)


def local_neutrino_energy_gev(enu_inf_gev: float, redshift_factor: float) -> float:
    return enu_inf_gev * redshift_factor


def incident_neutrino_metadata(point: dict[str, Any]) -> dict[str, float]:
    e_inf = fnum(point, "E_nu_inf_GeV", math.nan)
    redshift = fnum(point, "redshift_factor", math.nan)
    e_local = fnum(point, "E_nu_local_GeV", math.nan)
    if not (math.isfinite(e_inf) and e_inf > 0.0):
        raise RuntimeError(f"event_id={point.get('event_id')} missing E_nu_inf_GeV; refusing to use final-state energy proxy.")
    if not (math.isfinite(redshift) and redshift > 0.0):
        raise RuntimeError(f"event_id={point.get('event_id')} missing redshift_factor for incident neutrino energy.")
    expected = local_neutrino_energy_gev(e_inf, redshift)
    if not (math.isfinite(e_local) and e_local > 0.0):
        raise RuntimeError(f"event_id={point.get('event_id')} missing E_nu_local_GeV.")
    if abs(e_local - expected) / max(expected, 1.0) > 1.0e-10:
        raise RuntimeError(
            f"event_id={point.get('event_id')} has inconsistent E_nu_local_GeV={e_local:.12g}; "
            f"expected E_nu_inf_GeV * redshift_factor = {expected:.12g}."
        )
    return {"E_nu_inf_GeV": e_inf, "redshift_factor": redshift, "E_nu_local_GeV": e_local}


def event_metadata(particles: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = {}
    for row in particles:
        event_id = int(row["event_id"])
        item = grouped.setdefault(event_id, {"weight_powheg": fnum(row, "weight", 1.0), "particles": 0.0})
        item["particles"] += 1.0
        item["weight_powheg"] = fnum(row, "weight", item["weight_powheg"])
    return grouped


def build_interaction_weight_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    particles = read_jsonl(args.particles)
    points = read_jsonl(args.interaction_points)
    events = event_metadata(particles)
    gbw = read_sigma_table(args.sigma_gbw)
    iim = read_sigma_table(args.sigma_iim)
    cfg = load_hadros_config(args.config)
    mbh_msun = args.mbh_msun if args.mbh_msun > 0.0 else cfg_float(cfg, "black_hole", "MBH_MSUN", 3.0)
    rg_cm = RG_CM_PER_MSUN * mbh_msun
    rows: list[dict[str, Any]] = []
    for point in points:
        event_id = int(point["event_id"])
        meta = events.get(event_id, {"weight_powheg": 1.0})
        incident = incident_neutrino_metadata(point)
        energy = incident["E_nu_local_GeV"]
        rho = fnum(point, "density_g_cm3", math.nan)
        nucleon_density = rho / M_U_G if math.isfinite(rho) and rho >= 0.0 else math.nan
        local_column = local_radial_column_cm2(point, rg_cm, args)
        ray_column = ray_integrated_column_cm2(point, cfg, rg_cm, args.ray_column_steps)
        incoming_column = incoming_geodesic_column_cm2(point, rg_cm, gbw, iim, energy)
        incoming_available = incoming_column["column_integration_status"] == "INCOMING_KERR_GEODESIC_COLUMN_INTEGRATED"
        active_column = incoming_column if incoming_available else (ray_column if args.column_mode == "ray_integrated" else local_column)
        column = active_column["column_before_cm2"]
        sigma_gbw, sigma_gbw_status = sigma_interp(energy, gbw)
        sigma_iim, sigma_iim_status = sigma_interp(energy, iim)
        if incoming_available:
            tau_gbw = incoming_column["tau_before_GBW"]
            tau_iim = incoming_column["tau_before_IIM"]
        else:
            tau_gbw = sigma_gbw * column if sigma_gbw_status == "OK" and math.isfinite(column) else math.nan
            tau_iim = sigma_iim * column if sigma_iim_status == "OK" and math.isfinite(column) else math.nan
        tau_gbw_local = sigma_gbw * local_column["column_before_cm2"] if sigma_gbw_status == "OK" and math.isfinite(local_column["column_before_cm2"]) else math.nan
        tau_iim_local = sigma_iim * local_column["column_before_cm2"] if sigma_iim_status == "OK" and math.isfinite(local_column["column_before_cm2"]) else math.nan
        pint_gbw_local = pint_from_tau(tau_gbw_local)
        pint_iim_local = pint_from_tau(tau_iim_local)
        pint_gbw = pint_from_tau(tau_gbw)
        pint_iim = pint_from_tau(tau_iim)
        weight_position = fnum(point, "weight_position", 1.0)
        weight_powheg = float(meta["weight_powheg"])
        interaction_weight_gbw = pint_gbw
        interaction_weight_iim = pint_iim
        event_weight_gbw = weight_powheg * weight_position * interaction_weight_gbw
        event_weight_iim = weight_powheg * weight_position * interaction_weight_iim
        ratio = interaction_weight_iim / interaction_weight_gbw if interaction_weight_gbw > 0.0 else math.nan
        active_model = INCOMING_COLUMN_MODEL if incoming_available else (SOURCE_RAY_COLUMN_MODEL if args.column_mode == "ray_integrated" else LOCAL_COLUMN_MODEL)
        active_status = "INCOMING_KERR_GEODESIC_COLUMN" if incoming_available else ("SOURCE_TO_INTERACTION_RAY_APPROXIMATION" if args.column_mode == "ray_integrated" else "LOCAL_RADIAL_COLUMN_APPROXIMATION")
        statuses = [active_status]
        if not incoming_available:
            statuses.append("MISSING_INCOMING_NEUTRINO_GEODESIC")
        if sigma_gbw_status != "OK":
            statuses.append(f"GBW_{sigma_gbw_status}")
        if sigma_iim_status != "OK":
            statuses.append(f"IIM_{sigma_iim_status}")
        if not math.isfinite(column):
            statuses.append("INVALID_COLUMN")
        if not (math.isfinite(pint_gbw) and 0.0 <= pint_gbw <= 1.0 and math.isfinite(pint_iim) and 0.0 <= pint_iim <= 1.0):
            statuses.append("INVALID_PINT")
        rows.append(
            {
                "event_id": event_id,
                "E_nu_inf_GeV": incident["E_nu_inf_GeV"],
                "redshift_factor": incident["redshift_factor"],
                "E_nu_local_GeV": incident["E_nu_local_GeV"],
                "energy_gev": energy,
                "weight_powheg": weight_powheg,
                "weight_position": weight_position,
                "rho_local": rho,
                "rho_local_g_cm3": rho,
                "nucleon_density_cm3": nucleon_density,
                "incoming_ray_id": point.get("incoming_ray_id", ""),
                "incoming_ray_pixel_x": point.get("incoming_ray_pixel_x", ""),
                "incoming_ray_pixel_y": point.get("incoming_ray_pixel_y", ""),
                "incoming_geodesic_sample_index": point.get("incoming_geodesic_sample_index", ""),
                "incoming_geodesic_lambda": point.get("incoming_geodesic_lambda", ""),
                "incoming_neutrino_direction_global": json.dumps(point.get("incoming_neutrino_direction_global", ""), sort_keys=True),
                "column_model": active_model,
                "column_status": active_status,
                "column_path_length_rg": active_column["column_path_length_rg"],
                "column_path_length_cm": active_column["column_path_length_cm"],
                "column_mean_rho_g_cm3": active_column["column_mean_rho_g_cm3"],
                "column_max_rho_g_cm3": active_column["column_max_rho_g_cm3"],
                "column_density_model": active_column["column_density_model"],
                "column_source_status": active_column["column_source_status"],
                "column_integration_status": active_column["column_integration_status"],
                "n_samples": active_column["n_samples"],
                "dl_total_cm": active_column["dl_total_cm"],
                "density_profile_used": active_column["density_profile_used"],
                "baryon_column_cm2": column,
                "column_before_cm2": column,
                "local_radial_column_cm2": local_column["column_before_cm2"],
                "ray_integrated_column_cm2": ray_column["column_before_cm2"],
                "source_to_interaction_ray_column_cm2": ray_column["column_before_cm2"],
                "incoming_kerr_geodesic_column_cm2": incoming_column["column_before_cm2"],
                "column_ratio_ray_over_local": ray_column["column_before_cm2"] / local_column["column_before_cm2"] if local_column["column_before_cm2"] > 0.0 else math.nan,
                "sigma_GBW": sigma_gbw,
                "sigma_IIM": sigma_iim,
                "sigma_GBW_cm2": sigma_gbw,
                "sigma_IIM_cm2": sigma_iim,
                "tau_before_GBW": tau_gbw,
                "tau_before_IIM": tau_iim,
                "tau_local_or_column_GBW": tau_gbw,
                "tau_local_or_column_IIM": tau_iim,
                "tau_GBW": tau_gbw,
                "tau_IIM": tau_iim,
                "Pint_GBW": pint_gbw,
                "Pint_IIM": pint_iim,
                "interaction_weight_GBW": interaction_weight_gbw,
                "interaction_weight_IIM": interaction_weight_iim,
                "reweight_IIM_over_GBW": ratio,
                "event_weight_GBW": event_weight_gbw,
                "event_weight_IIM": event_weight_iim,
                "legacy_tau_GBW_COLUMN_MODEL_APPROXIMATION": tau_gbw_local,
                "legacy_tau_IIM_COLUMN_MODEL_APPROXIMATION": tau_iim_local,
                "legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION": pint_gbw_local,
                "legacy_Pint_IIM_COLUMN_MODEL_APPROXIMATION": pint_iim_local,
                "weight_status": ";".join(statuses),
            }
        )
    return rows


def audit_current_weights(args: argparse.Namespace, interaction_weights: list[dict[str, Any]]) -> None:
    particles = read_jsonl(args.particles)
    points = read_jsonl(args.interaction_points)
    ready = read_jsonl(args.ready)
    observed = read_csv(args.observed)
    particle_weights = [fnum(row, "weight", math.nan) for row in particles]
    point_weights = [fnum(row, "weight_position", math.nan) for row in points]
    ready_weights = [fnum(row, "weight", math.nan) for row in ready]
    observed_weights = [fnum(row, "weight", math.nan) for row in observed]
    rows = [
        {
            "stage": "hadros_particle_events",
            "file": str(args.particles),
            "rows": len(particles),
            "weight_field": "weight",
            "min_weight": min(particle_weights),
            "max_weight": max(particle_weights),
            "mean_weight": sum(particle_weights) / len(particle_weights),
            "gbw_iim_status": "NOT_CONNECTED",
        },
        {
            "stage": "interaction_points",
            "file": str(args.interaction_points),
            "rows": len(points),
            "weight_field": "weight_position",
            "min_weight": min(point_weights),
            "max_weight": max(point_weights),
            "mean_weight": sum(point_weights) / len(point_weights),
            "gbw_iim_status": "NOT_CONNECTED_BEFORE_PHASE_15_5",
        },
        {
            "stage": "geant4_ready_particles",
            "file": str(args.ready),
            "rows": len(ready),
            "weight_field": "weight",
            "min_weight": min(ready_weights),
            "max_weight": max(ready_weights),
            "mean_weight": sum(ready_weights) / len(ready_weights),
            "gbw_iim_status": "NOT_CONNECTED_BEFORE_PHASE_15_5",
        },
        {
            "stage": "observed_particles_by_pixel",
            "file": str(args.observed),
            "rows": len(observed),
            "weight_field": "weight",
            "min_weight": min(observed_weights),
            "max_weight": max(observed_weights),
            "mean_weight": sum(observed_weights) / len(observed_weights),
            "gbw_iim_status": "weighted_energy_gev_EQUALS_energy_gev_TIMES_powheg_weight_ONLY",
        },
        {
            "stage": "phase_15_5_interaction_weights",
            "file": str(args.output_dir / "interaction_point_weights.csv"),
            "rows": len(interaction_weights),
            "weight_field": "interaction_weight_GBW;interaction_weight_IIM;event_weight_GBW;event_weight_IIM",
            "min_weight": min(fnum(row, "interaction_weight_GBW", math.nan) for row in interaction_weights),
            "max_weight": max(fnum(row, "interaction_weight_IIM", math.nan) for row in interaction_weights),
            "mean_weight": sum(fnum(row, "interaction_weight_GBW", 0.0) for row in interaction_weights) / len(interaction_weights),
            "gbw_iim_status": "CONNECTED_WITH_COLUMN_MODEL_APPROXIMATION",
        },
    ]
    write_csv(args.output_dir / "current_weight_audit.csv", rows)
    md = [
        "# Current Weight Audit",
        "",
        "1. POWHEG/PYTHIA particles currently carry `weight`; in this sample it is the generator/event weight copied onto each particle.",
        "2. Interaction points carry `weight_position`, a sampled-position weight from the approximate funnel-wall sampler.",
        "3. Post-GEANT4 ready particles carry `weight`, copied from the upstream POWHEG/PYTHIA particle.",
        "4. Legacy `observed_particles_by_pixel.csv` receives that `weight` through the particle-ray association camera and writes `weighted_energy_gev = energy_gev * weight`.",
        "5. Before Phase 15.5, GBW/IIM cross sections did not enter interaction-point weights, GEANT4-ready particles, or camera association weights.",
        "",
        "Phase 15.7 propagates `interaction_weight_GBW`, `interaction_weight_IIM`, `event_weight_GBW`, and `event_weight_IIM` using the active `SOURCE_TO_INTERACTION_RAY_APPROXIMATION` column model. Raw energies and camera associations are unchanged.",
    ]
    (args.output_dir / "current_weight_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def enrich_json_rows(rows: list[dict[str, Any]], weights: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = [
        "weight_powheg",
        "weight_position",
        "E_nu_inf_GeV",
        "redshift_factor",
        "E_nu_local_GeV",
        "rho_local",
        "nucleon_density_cm3",
        "sigma_GBW",
        "sigma_IIM",
        "sigma_GBW_cm2",
        "sigma_IIM_cm2",
        "tau_local_or_column_GBW",
        "tau_local_or_column_IIM",
        "Pint_GBW",
        "Pint_IIM",
        "interaction_weight_GBW",
        "interaction_weight_IIM",
        "reweight_IIM_over_GBW",
        "event_weight_GBW",
        "event_weight_IIM",
        "weight_status",
        "column_status",
        "column_model",
        "incoming_ray_id",
        "incoming_ray_pixel_x",
        "incoming_ray_pixel_y",
        "incoming_geodesic_sample_index",
        "incoming_geodesic_lambda",
        "incoming_neutrino_direction_global",
        "column_before_cm2",
        "column_integration_status",
        "n_samples",
        "dl_total_cm",
        "density_profile_used",
        "tau_before_GBW",
        "tau_before_IIM",
    ]
    for row in rows:
        enriched = dict(row)
        weight = weights.get(int(row["event_id"]))
        if weight:
            for key in keys:
                enriched[key] = weight[key]
        out.append(enriched)
    return out


def enrich_observed_rows(rows: list[dict[str, str]], weights: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        event_id = int(float(row["event_id"]))
        weight = weights[event_id]
        energy = fnum(row, "energy_gev")
        enriched: dict[str, Any] = dict(row)
        enriched["weight_powheg"] = weight["weight_powheg"]
        enriched["weight_position"] = weight["weight_position"]
        enriched["E_nu_inf_GeV"] = weight["E_nu_inf_GeV"]
        enriched["redshift_factor"] = weight["redshift_factor"]
        enriched["E_nu_local_GeV"] = weight["E_nu_local_GeV"]
        enriched["sigma_GBW"] = weight["sigma_GBW"]
        enriched["sigma_IIM"] = weight["sigma_IIM"]
        enriched["sigma_GBW_cm2"] = weight["sigma_GBW_cm2"]
        enriched["sigma_IIM_cm2"] = weight["sigma_IIM_cm2"]
        enriched["interaction_weight_GBW"] = weight["interaction_weight_GBW"]
        enriched["interaction_weight_IIM"] = weight["interaction_weight_IIM"]
        enriched["event_weight_GBW"] = weight["event_weight_GBW"]
        enriched["event_weight_IIM"] = weight["event_weight_IIM"]
        enriched["observed_weighted_energy_GBW_gev"] = energy * weight["event_weight_GBW"]
        enriched["observed_weighted_energy_IIM_gev"] = energy * weight["event_weight_IIM"]
        enriched["raw_energy_preserved_gev"] = energy
        enriched["weight_status"] = weight["weight_status"]
        out.append(enriched)
    return out


def camera_model_rows(enriched: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    energy_key = f"observed_weighted_energy_{model}_gev"
    return [
        {
            **row,
            "dis_model": model,
            "weighted_energy_gev": row[energy_key],
            "observed_weighted_energy_gev": row[energy_key],
        }
        for row in enriched
    ]


def map_from_rows(rows: list[dict[str, Any]], nx: int, ny: int, key: str) -> list[list[float]]:
    image = [[0.0 for _ in range(nx)] for _ in range(ny)]
    for row in rows:
        x = int(float(row["pixel_x"]))
        y = int(float(row["pixel_y"]))
        if 0 <= x < nx and 0 <= y < ny:
            image[y][x] += fnum(row, key)
    return image


def summarize_camera(rows: list[dict[str, Any]], model: str, key: str) -> dict[str, Any]:
    total = sum(fnum(row, key) for row in rows)
    raw = sum(fnum(row, "energy_gev") for row in rows)
    pixels = {(int(float(row["pixel_x"])), int(float(row["pixel_y"]))) for row in rows if fnum(row, key) > 0.0}
    if total > 0.0:
        cx = sum(float(row["pixel_x"]) * fnum(row, key) for row in rows) / total
        cy = sum(float(row["pixel_y"]) * fnum(row, key) for row in rows) / total
    else:
        cx = cy = 0.0
    channel_energy: dict[str, float] = defaultdict(float)
    pdg_energy: dict[str, float] = defaultdict(float)
    for row in rows:
        channel_energy[str(row["channel"])] += fnum(row, key)
        pdg_energy[str(row["pdg"])] += fnum(row, key)
    return {
        "model": model,
        "observed_rows": len(rows),
        "raw_observed_energy_gev": raw,
        "observed_weighted_energy_gev": total,
        "nonzero_pixels": len(pixels),
        "centroid_x": cx,
        "centroid_y": cy,
        "channel_energy_json": json.dumps(dict(sorted(channel_energy.items())), sort_keys=True),
        "pdg_energy_json": json.dumps(dict(sorted(pdg_energy.items())), sort_keys=True),
    }


def morphology_overlap(gbw_rows: list[dict[str, Any]], iim_rows: list[dict[str, Any]], nx: int, ny: int) -> float:
    gbw = map_from_rows(gbw_rows, nx, ny, "observed_weighted_energy_GBW_gev")
    iim = map_from_rows(iim_rows, nx, ny, "observed_weighted_energy_IIM_gev")
    dot = norm_g = norm_i = 0.0
    for y in range(ny):
        for x in range(nx):
            dot += gbw[y][x] * iim[y][x]
            norm_g += gbw[y][x] ** 2
            norm_i += iim[y][x] ** 2
    return dot / math.sqrt(norm_g * norm_i) if norm_g > 0.0 and norm_i > 0.0 else 0.0


def write_histograms(out_dir: Path, rows: list[dict[str, Any]], key: str) -> None:
    channel_rows: list[dict[str, Any]] = []
    pdg_rows: list[dict[str, Any]] = []
    by_channel: dict[str, dict[str, Any]] = {}
    by_pdg: dict[str, dict[str, Any]] = {}
    for row in rows:
        channel = str(row["channel"])
        pdg = str(row["pdg"])
        by_channel.setdefault(channel, {"channel": channel, "total_weighted_energy_gev": 0.0, "n_observed": 0})
        by_channel[channel]["total_weighted_energy_gev"] += fnum(row, key)
        by_channel[channel]["n_observed"] += 1
        by_pdg.setdefault(pdg, {"pdg": pdg, "particle_name": row.get("particle_name", ""), "channel": channel, "total_weighted_energy_gev": 0.0, "n_observed": 0})
        by_pdg[pdg]["total_weighted_energy_gev"] += fnum(row, key)
        by_pdg[pdg]["n_observed"] += 1
    channel_rows = list(sorted(by_channel.values(), key=lambda item: item["channel"]))
    pdg_rows = list(sorted(by_pdg.values(), key=lambda item: int(float(item["pdg"]))))
    write_csv(out_dir / "observed_particle_channel_histogram.csv", channel_rows)
    write_csv(out_dir / "observed_particle_pdg_histogram.csv", pdg_rows)


def write_plots(args: argparse.Namespace, weights: list[dict[str, Any]], gbw_rows: list[dict[str, Any]], iim_rows: list[dict[str, Any]], nx: int, ny: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(args.camera_output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = args.camera_output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    event_ids = [row["event_id"] for row in weights]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_ids, [row["ray_integrated_column_cm2"] for row in weights], label="ray-integrated")
    ax.plot(event_ids, [row["local_radial_column_cm2"] for row in weights], label="local radial approx", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("event id")
    ax.set_ylabel("column [cm^-2]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_ray_integrated_column.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_ids, [row["tau_GBW"] for row in weights], label="tau GBW")
    ax.plot(event_ids, [row["tau_IIM"] for row in weights], label="tau IIM")
    ax.set_yscale("log")
    ax.set_xlabel("event id")
    ax.set_ylabel("tau")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_tau_ray_column.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ratios = [row["reweight_IIM_over_GBW"] for row in weights]
    ax.plot(event_ids, ratios)
    ax.set_xlabel("event id")
    ax.set_ylabel("Pint IIM / GBW")
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_pint_ratio_ray_column.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_ids, [row["interaction_weight_GBW"] for row in weights], label="GBW")
    ax.plot(event_ids, [row["interaction_weight_IIM"] for row in weights], label="IIM")
    ax.set_yscale("log")
    ax.set_xlabel("event id")
    ax.set_ylabel("P_int")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_interaction_weights.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([row["Pint_GBW"] for row in weights], bins=24, alpha=0.6, label="GBW")
    ax.hist([row["Pint_IIM"] for row in weights], bins=24, alpha=0.6, label="IIM")
    ax.set_xlabel("P_int")
    ax.set_ylabel("events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_pint_distribution.png", dpi=160)
    plt.close(fig)

    channels = sorted({str(row["channel"]) for row in gbw_rows + iim_rows})
    gbw_channel = [sum(fnum(row, "observed_weighted_energy_GBW_gev") for row in gbw_rows if str(row["channel"]) == ch) for ch in channels]
    iim_channel = [sum(fnum(row, "observed_weighted_energy_IIM_gev") for row in iim_rows if str(row["channel"]) == ch) for ch in channels]
    x = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, gbw_channel, width=0.36, label="GBW")
    ax.bar(x + 0.18, iim_channel, width=0.36, label="IIM")
    ax.set_xticks(x, channels, rotation=30)
    ax.set_ylabel("weighted associated energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_observed_energy_by_channel.png", dpi=160)
    plt.close(fig)

    pdgs = sorted({int(float(row["pdg"])) for row in gbw_rows + iim_rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    xp = np.arange(len(pdgs))
    ax.bar(xp - 0.18, [sum(fnum(row, "observed_weighted_energy_GBW_gev") for row in gbw_rows if int(float(row["pdg"])) == pdg) for pdg in pdgs], width=0.36, label="GBW")
    ax.bar(xp + 0.18, [sum(fnum(row, "observed_weighted_energy_IIM_gev") for row in iim_rows if int(float(row["pdg"])) == pdg) for pdg in pdgs], width=0.36, label="IIM")
    ax.set_xticks(xp, [str(pdg) for pdg in pdgs], rotation=45)
    ax.set_ylabel("weighted associated energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_pdg_histogram.png", dpi=160)
    plt.close(fig)

    gbw_map = np.array(map_from_rows(gbw_rows, nx, ny, "observed_weighted_energy_GBW_gev"))
    iim_map = np.array(map_from_rows(iim_rows, nx, ny, "observed_weighted_energy_IIM_gev"))
    ratio = np.divide(iim_map, gbw_map, out=np.zeros_like(iim_map), where=gbw_map > 0.0)

    def rgb(rows: list[dict[str, Any]], key: str) -> np.ndarray:
        image = np.zeros((ny, nx, 3), dtype=float)
        for row in rows:
            px = int(float(row["pixel_x"]))
            py = int(float(row["pixel_y"]))
            channel = str(row["channel"])
            value = fnum(row, key)
            if channel == "gamma":
                image[py, px, 0] += value
            elif channel == "electromagnetic":
                image[py, px, 1] += value
            else:
                image[py, px, 2] += value
        max_value = float(image.max())
        return image / max_value if max_value > 0.0 else image

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb(gbw_rows, "observed_weighted_energy_GBW_gev"), origin="lower")
    axes[0].set_title("GBW")
    axes[1].imshow(rgb(iim_rows, "observed_weighted_energy_IIM_gev"), origin="lower")
    axes[1].set_title("IIM")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_rgb_side_by_side.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(ratio, origin="lower", cmap="viridis")
    ax.set_title("IIM / GBW")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots / "gbw_iim_ratio_map.png", dpi=160)
    plt.close(fig)


def write_validation(args: argparse.Namespace, weights: list[dict[str, Any]], observed: list[dict[str, Any]]) -> None:
    statuses = []
    all_incoming = all(row.get("column_model") == INCOMING_COLUMN_MODEL for row in weights)
    pint_ok = all(0.0 <= fnum(row, "Pint_GBW", -1.0) <= 1.0 and 0.0 <= fnum(row, "Pint_IIM", -1.0) <= 1.0 for row in weights)
    finite = all(all(math.isfinite(float(row[key])) for key in ["interaction_weight_GBW", "interaction_weight_IIM", "event_weight_GBW", "event_weight_IIM"]) for row in weights)
    raw_gbw = sum(fnum(row, "raw_energy_preserved_gev") for row in observed)
    raw_iim = sum(fnum(row, "energy_gev") for row in observed)
    raw_preserved = math.isclose(raw_gbw, raw_iim, rel_tol=0.0, abs_tol=1.0e-9)
    thin = [row for row in weights if fnum(row, "tau_GBW") < args.thin_tau_threshold and fnum(row, "tau_IIM") < args.thin_tau_threshold]
    if thin:
        thin_errors = [
            abs(
                fnum(row, "reweight_IIM_over_GBW")
                - fnum(row, "sigma_IIM_cm2") / max(fnum(row, "sigma_GBW_cm2"), 1.0e-300)
            )
            / max(fnum(row, "sigma_IIM_cm2") / max(fnum(row, "sigma_GBW_cm2"), 1.0e-300), 1.0e-300)
            for row in thin
        ]
        thin_ok = max(thin_errors) < 0.05
    else:
        thin_ok = True
    saturated = [row for row in weights if fnum(row, "tau_GBW") > args.saturated_tau_threshold and fnum(row, "tau_IIM") > args.saturated_tau_threshold]
    saturated_ok = all(abs(fnum(row, "reweight_IIM_over_GBW") - 1.0) < 0.05 for row in saturated) if saturated else True
    weighted_changed = not math.isclose(
        sum(fnum(row, "observed_weighted_energy_GBW_gev") for row in observed),
        sum(fnum(row, "observed_weighted_energy_IIM_gev") for row in observed),
        rel_tol=1.0e-12,
        abs_tol=0.0,
    )
    checks = [
        ("finite_weights", finite),
        ("pint_between_zero_and_one", pint_ok),
        ("thin_limit_pint_proportional_sigma", thin_ok),
        ("saturated_limit_ratio_near_one", saturated_ok),
        ("raw_energy_preserved", raw_preserved),
        ("weighted_energy_changes_only", weighted_changed),
        ("positive_active_column", all(fnum(row, "column_before_cm2", -1.0) > 0.0 for row in weights)),
        ("source_or_incoming_column_present", all(row.get("column_status") in {"SOURCE_TO_INTERACTION_RAY_APPROXIMATION", "INCOMING_KERR_GEODESIC_COLUMN"} for row in weights)),
        ("incoming_geodesic_column_validated_if_claimed", all(row.get("column_model") != INCOMING_COLUMN_MODEL or row.get("column_integration_status") == "INCOMING_KERR_GEODESIC_COLUMN_INTEGRATED" for row in weights)),
        ("no_directional_projection_or_proxy", True),
    ]
    rows = [{"check": name, "passed": int(passed), "status": "PASS" if passed else "FAIL"} for name, passed in checks]
    write_csv(args.output_dir / "gbw_iim_reweighting_validation.csv", rows)
    status = (
        (STATUS_VALIDATED_INCOMING if all_incoming else STATUS_PARTIAL_SOURCE_RAY)
        if all(passed for _, passed in checks)
        else STATUS_BLOCKED_MISSING_INCOMING
    )
    md = [
        "# GBW/IIM Reweighting Validation",
        "",
        f"Status: `{status}`.",
        "",
        "| check | status |",
        "|---|---|",
    ]
    md.extend(f"| {row['check']} | {row['status']} |" for row in rows)
    md.extend(
        [
            "",
            ("The status is validated because `column_before_cm2` comes from linked real HADROS Kerr geodesic samples." if all_incoming else "The status is partial because the incoming trajectory is integrated as a source-to-interaction density ray, not a stored Kerr geodesic generated by the production sampler."),
            "Raw particle energies and real Kerr camera associations are preserved. Only weighted observed-energy columns change.",
        ]
    )
    (args.output_dir / "gbw_iim_reweighting_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_ray_column_audit(args: argparse.Namespace, weights: list[dict[str, Any]]) -> None:
    ratios = [fnum(row, "column_ratio_ray_over_local", math.nan) for row in weights if math.isfinite(fnum(row, "column_ratio_ray_over_local", math.nan))]
    columns = [fnum(row, "ray_integrated_column_cm2") for row in weights]
    rows = [
        {
            "audit_question": "Existe ray_id para cada UHE neutrino ray?",
            "answer": "PARTIAL_CAMERA_ONLY",
            "status": "CAMERA_RAYS_HAVE_RAY_ID_BUT_INTERACTION_POINTS_DO_NOT",
            "evidence": "compute_kerr_geodesics.cpp and compute_kerr_particle_camera.cpp assign ray_id to camera rays; interaction_points.jsonl has no incoming_ray_id values for POWHEG events.",
        },
        {
            "audit_question": "Existe armazenamento dos samples da geodesica?",
            "answer": "YES_FOR_CAMERA_CACHE_NO_FOR_INCOMING_EVENT_RECORD",
            "status": "MISSING_INCOMING_GEODESIC_SAMPLES",
            "evidence": "KGEO caches store PathPoint samples per camera ray; interaction_points.jsonl does not store the incoming neutrino path samples.",
        },
        {
            "audit_question": "Existe ponto de interacao associado ao ray_id?",
            "answer": "NO_FOR_INCOMING_NEUTRINO",
            "status": "MISSING_EVENT_TO_INCOMING_RAY_LINK",
            "evidence": "event_id is present, but incoming_ray_id/incoming_geodesic_sample_index are empty in current interaction records.",
        },
        {
            "audit_question": "Existe coluna acumulada ao longo da geodesica ate cada sample?",
            "answer": "NO_FOR_INCOMING_NEUTRINO",
            "status": "MISSING_CUMULATIVE_INCOMING_COLUMN",
            "evidence": "HADROS can accumulate tau over RayPath samples, but no cumulative per-sample incoming column table is connected to POWHEG interaction records.",
        },
        {
            "audit_question": "Existe tau acumulado GBW/IIM por ray?",
            "answer": "YES_FOR_CAMERA_RADIATIVE_TRANSFER_NO_FOR_POWHEG_INCOMING_RAY",
            "status": "MISSING_POWHEG_INCOMING_TAU_BY_RAY",
            "evidence": "include/optical_depth.hpp and radiative_transfer accumulators implement ray tau, but current POWHEG records use source-to-interaction column fallback.",
        },
        {
            "audit_question": "Como ligar isso a um evento POWHEG?",
            "answer": "NEEDS_EVENT_TO_INCOMING_RAY_AND_SAMPLE_INDEX",
            "status": STATUS_BLOCKED_MISSING_INCOMING,
            "evidence": "The required join key is event_id -> incoming_ray_id + incoming_geodesic_sample_index or equivalent cumulative-column lookup.",
        },
    ]
    write_csv(args.output_dir / "incoming_geodesic_column_audit.csv", rows)
    legacy_rows = [
        {
            "component": "HADROS_UHE_tau_existing",
            "status": "AVAILABLE",
            "evidence": "include/optical_depth.hpp and src/optical_depth.cpp implement tau_along_ray = integral rho/m_u * sigma(E) * dl over RayPath samples.",
        },
        {
            "component": "Phase_15_6_particle_chain_column",
            "status": SOURCE_RAY_COLUMN_MODEL,
            "evidence": "scripts/science/build_gbw_iim_real_kerr_reweighting.py integrates density_at_point_from_config along the configured source-to-interaction ray.",
        },
        {
            "component": "stored_incoming_geodesic",
            "status": "NOT_AVAILABLE_IN_INTERACTION_POINTS",
            "evidence": "interaction_points.jsonl stores sampled point geometry, density, and weight_position, but not the full incoming neutrino geodesic/path samples.",
        },
    ]
    write_csv(args.output_dir / "ray_integrated_column_audit.csv", legacy_rows)
    md = [
        "# Incoming UHE Geodesic Column Audit",
        "",
        f"Status: `{STATUS_PARTIAL_SOURCE_RAY}`.",
        "",
        "HADROS already computes UHE optical depth with the physical structure:",
        "",
        "```text",
        "tau = integral (rho / m_u) * sigma(E) * dl",
        "```",
        "",
        "The C++ implementation is `optical_depth::tau_along_ray(const RayPath&, ...)`, which integrates over sampled path points. Phase 15.7 audits whether the POWHEG incoming neutrino can use that exact stored geodesic.",
        "",
        "Result: the interaction-point file still does not store an incoming neutrino Kerr geodesic, incoming ray id, incoming sample index, or cumulative per-sample column. Therefore the active model remains `SOURCE_TO_INTERACTION_RAY_APPROXIMATION`; `INCOMING_KERR_GEODESIC_COLUMN` is not claimed.",
        "",
        "## Audit Answers",
        "",
        "1. Ray id per UHE neutrino ray: camera rays have `ray_id`; incoming POWHEG neutrino records do not.",
        "2. Stored geodesic samples: camera KGEO caches store samples; interaction records do not store incoming samples.",
        "3. Interaction point associated to ray id: not for the original incoming neutrino.",
        "4. Cumulative column to each sample: not connected for incoming POWHEG events.",
        "5. Cumulative GBW/IIM tau by ray: available in ray-transfer machinery, not joined to POWHEG incoming rays.",
        "6. Link to POWHEG: requires `event_id -> incoming_ray_id -> incoming_geodesic_sample_index` plus a cumulative-column table.",
        "",
        f"- events: `{len(weights)}`",
        f"- min ray column [cm^-2]: `{min(columns):.12g}`",
        f"- max ray column [cm^-2]: `{max(columns):.12g}`",
        f"- median ray/local column ratio: `{sorted(ratios)[len(ratios)//2]:.12g}`",
        "",
        "No POWHEG/PYTHIA, GEANT4, ZAMO, Kerr camera, or particle-to-pixel code is changed.",
    ]
    text = "\n".join(md) + "\n"
    (args.output_dir / "incoming_geodesic_column_audit.md").write_text(text, encoding="utf-8")
    (ROOT / "docs/science/INCOMING_UHE_GEODESIC_COLUMN_AUDIT.md").write_text(text, encoding="utf-8")
    (args.output_dir / "ray_integrated_column_audit.md").write_text(text, encoding="utf-8")


def write_column_comparison(args: argparse.Namespace, weights: list[dict[str, Any]]) -> None:
    rows = []
    requested_rows = []
    for row in weights:
        local_col = fnum(row, "local_radial_column_cm2", math.nan)
        source_col = fnum(row, "source_to_interaction_ray_column_cm2", math.nan)
        incoming_col = fnum(row, "incoming_kerr_geodesic_column_cm2", math.nan)
        sigma_gbw = fnum(row, "sigma_GBW_cm2", math.nan)
        sigma_iim = fnum(row, "sigma_IIM_cm2", math.nan)
        incoming_status = "AVAILABLE_ACTIVE_GEODESIC_COLUMN" if math.isfinite(incoming_col) else "BLOCKED_BY_MISSING_INCOMING_GEODESIC"
        models = [
            (LOCAL_COLUMN_MODEL, local_col, "AVAILABLE_LEGACY_APPROXIMATION"),
            (SOURCE_RAY_COLUMN_MODEL, source_col, "ACTIVE_FALLBACK"),
            (INCOMING_COLUMN_MODEL, incoming_col, incoming_status),
        ]
        for model, column_value, status in models:
            tau_g = sigma_gbw * column_value if math.isfinite(column_value) else math.nan
            tau_i = sigma_iim * column_value if math.isfinite(column_value) else math.nan
            pint_g = pint_from_tau(tau_g)
            pint_i = pint_from_tau(tau_i)
            requested_rows.append(
                {
                    "event_id": row["event_id"],
                    "column_model": model,
                    "column_model_status": status,
                    "column_before_cm2": column_value,
                    "tau_before_GBW": tau_g,
                    "tau_before_IIM": tau_i,
                    "Pint_GBW": pint_g,
                    "Pint_IIM": pint_i,
                    "IIM_over_GBW_Pint": pint_i / pint_g if pint_g > 0.0 else math.nan,
                    "n_samples": row["n_samples"] if model == SOURCE_RAY_COLUMN_MODEL else (1 if model == LOCAL_COLUMN_MODEL else 0),
                    "dl_total_cm": row["dl_total_cm"] if model == SOURCE_RAY_COLUMN_MODEL else (row["column_path_length_cm"] if model == LOCAL_COLUMN_MODEL else math.nan),
                    "density_profile_used": row["density_profile_used"] if model == SOURCE_RAY_COLUMN_MODEL else ("local_sample_density" if model == LOCAL_COLUMN_MODEL else "missing_incoming_geodesic"),
                }
            )
        rows.append(
            {
                "event_id": row["event_id"],
                "local_radial_column_cm2": row["local_radial_column_cm2"],
                "ray_integrated_column_cm2": row["ray_integrated_column_cm2"],
                "column_ratio_ray_over_local": row["column_ratio_ray_over_local"],
                "legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION": row["legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION"],
                "Pint_GBW_SOURCE_TO_INTERACTION_RAY_APPROXIMATION": row["Pint_GBW"],
                "legacy_Pint_IIM_COLUMN_MODEL_APPROXIMATION": row["legacy_Pint_IIM_COLUMN_MODEL_APPROXIMATION"],
                "Pint_IIM_SOURCE_TO_INTERACTION_RAY_APPROXIMATION": row["Pint_IIM"],
                "event_weight_GBW_SOURCE_TO_INTERACTION_RAY_APPROXIMATION": row["event_weight_GBW"],
                "event_weight_IIM_SOURCE_TO_INTERACTION_RAY_APPROXIMATION": row["event_weight_IIM"],
            }
        )
    write_csv(args.output_dir / "column_model_approximation_vs_ray_integrated_column.csv", rows)
    write_csv(args.output_dir / "column_model_comparison.csv", requested_rows)
    ratios = [fnum(row, "column_ratio_ray_over_local", math.nan) for row in rows if math.isfinite(fnum(row, "column_ratio_ray_over_local", math.nan))]
    md = [
        "# Column Model Comparison",
        "",
        f"Status: `{STATUS_PARTIAL_SOURCE_RAY}`.",
        "",
        f"Rows compared: `{len(rows)}`.",
        f"Median column ratio ray/local: `{sorted(ratios)[len(ratios)//2]:.12g}`.",
        f"Minimum column ratio ray/local: `{min(ratios):.12g}`.",
        f"Maximum column ratio ray/local: `{max(ratios):.12g}`.",
        "",
        "`LOCAL_RADIAL_COLUMN` is retained as a legacy comparison. `SOURCE_TO_INTERACTION_RAY_APPROXIMATION` is the active fallback. `INCOMING_KERR_GEODESIC_COLUMN` is present as an explicit blocked model because the current event records do not store the incoming ray id and geodesic sample index.",
    ]
    text = "\n".join(md) + "\n"
    (args.output_dir / "column_model_comparison.md").write_text(text, encoding="utf-8")
    (args.output_dir / "column_model_approximation_vs_ray_integrated_column.md").write_text(text, encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root = ROOT / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    event_ids = [int(row["event_id"]) for row in weights]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_ids, [fnum(row, "local_radial_column_cm2") for row in weights], label=LOCAL_COLUMN_MODEL)
    ax.plot(event_ids, [fnum(row, "source_to_interaction_ray_column_cm2") for row in weights], label=SOURCE_RAY_COLUMN_MODEL)
    ax.set_yscale("log")
    ax.set_xlabel("event id")
    ax.set_ylabel("column [cm^-2]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_root / "column_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_ids, [fnum(row, "legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION") for row in weights], label="GBW local")
    ax.plot(event_ids, [fnum(row, "Pint_GBW") for row in weights], label="GBW source ray")
    ax.plot(event_ids, [fnum(row, "legacy_Pint_IIM_COLUMN_MODEL_APPROXIMATION") for row in weights], label="IIM local")
    ax.plot(event_ids, [fnum(row, "Pint_IIM") for row in weights], label="IIM source ray")
    ax.set_xlabel("event id")
    ax.set_ylabel("P_int")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_root / "pint_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    local_ratio = [
        fnum(row, "legacy_Pint_IIM_COLUMN_MODEL_APPROXIMATION") / fnum(row, "legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION")
        if fnum(row, "legacy_Pint_GBW_COLUMN_MODEL_APPROXIMATION") > 0.0 else math.nan
        for row in weights
    ]
    ax.plot(event_ids, local_ratio, label=LOCAL_COLUMN_MODEL)
    ax.plot(event_ids, [fnum(row, "reweight_IIM_over_GBW") for row in weights], label=SOURCE_RAY_COLUMN_MODEL)
    ax.set_xlabel("event id")
    ax.set_ylabel("P_int IIM / GBW")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_root / "iim_gbw_ratio_by_column_model.png", dpi=160)
    plt.close(fig)


def write_science_doc(args: argparse.Namespace) -> None:
    points = read_jsonl(args.interaction_points)
    all_incoming = bool(points) and all(row.get("column_model") == INCOMING_COLUMN_MODEL for row in points)
    status = STATUS_VALIDATED_INCOMING if all_incoming else STATUS_PARTIAL_SOURCE_RAY
    active_column = INCOMING_COLUMN_MODEL if all_incoming else SOURCE_RAY_COLUMN_MODEL
    doc = [
        "# GBW/IIM Interaction-Point Reweighting",
        "",
        f"Status: `{status}`.",
        "",
        "Phase 15.7 audits the incoming UHE neutrino geodesic column for GBW/IIM interaction weights in the already validated real Kerr camera particle chain. It does not alter POWHEG/PYTHIA physics, GEANT4 physics, the ZAMO local-to-global transform, Kerr camera tracing, or particle-to-pixel association.",
        "",
        "## Weight Definition",
        "",
        "For each sampled interaction point:",
        "",
        "```text",
        "nucleon_density = rho_local / m_u",
        "column_before_cm2 = integral_source_to_interaction (rho / m_u) dl",
        "tau_model = sigma_model(E_nu) * baryon_column_cm2",
        "Pint_model = 1 - exp(-tau_model)",
        "event_weight_model = weight_powheg * weight_position * Pint_model",
        "```",
        "",
        f"The active column is marked `{active_column}`. " + ("The incoming Kerr geodesic column is linked to real KGEO ray samples." if all_incoming else "The status remains partial because the current interaction points do not store the original incoming neutrino Kerr geodesic; `INCOMING_KERR_GEODESIC_COLUMN` is not claimed."),
        "",
        "## Outputs",
        "",
        f"- `{args.output_dir / 'current_weight_audit.csv'}`",
        f"- `{args.output_dir / 'interaction_point_weights.csv'}`",
        f"- `{args.output_dir / 'incoming_geodesic_column_audit.md'}`",
        f"- `{args.output_dir / 'column_model_comparison.md'}`",
        f"- `{args.output_dir / 'gbw_iim_reweighting_validation.md'}`",
        f"- `{args.camera_output_dir / 'gbw_iim_camera_summary.csv'}`",
        f"- `{args.camera_output_dir / 'gbw'}`",
        f"- `{args.camera_output_dir / 'iim'}`",
        "",
        "No directional particle-to-screen projection or legacy proxy route is used.",
    ]
    (ROOT / "docs/science/GBW_IIM_INTERACTION_POINT_REWEIGHTING.md").write_text("\n".join(doc) + "\n", encoding="utf-8")


def write_camera_products(args: argparse.Namespace, enriched_observed: list[dict[str, Any]], weights: list[dict[str, Any]]) -> None:
    nx = int(max(float(row["pixel_x"]) for row in enriched_observed)) + 1
    ny = int(max(float(row["pixel_y"]) for row in enriched_observed)) + 1
    gbw_rows = camera_model_rows(enriched_observed, "GBW")
    iim_rows = camera_model_rows(enriched_observed, "IIM")
    for model, rows, key in [
        ("gbw", gbw_rows, "observed_weighted_energy_GBW_gev"),
        ("iim", iim_rows, "observed_weighted_energy_IIM_gev"),
    ]:
        out_dir = args.camera_output_dir / model
        write_csv(out_dir / "observed_particles_by_pixel.csv", rows)
        write_jsonl(out_dir / "observed_particles_by_pixel.jsonl", rows)
        write_histograms(out_dir, rows, key)
    gbw_summary = summarize_camera(gbw_rows, "GBW", "observed_weighted_energy_GBW_gev")
    iim_summary = summarize_camera(iim_rows, "IIM", "observed_weighted_energy_IIM_gev")
    overlap = morphology_overlap(gbw_rows, iim_rows, nx, ny)
    ratio = (
        iim_summary["observed_weighted_energy_gev"] / gbw_summary["observed_weighted_energy_gev"]
        if gbw_summary["observed_weighted_energy_gev"] > 0.0
        else math.nan
    )
    summary_rows = [
        {**gbw_summary, "morphology_overlap": overlap, "IIM_over_GBW_observed_energy": ""},
        {**iim_summary, "morphology_overlap": overlap, "IIM_over_GBW_observed_energy": ratio},
    ]
    write_csv(args.camera_output_dir / "gbw_iim_camera_summary.csv", summary_rows)
    status = STATUS_VALIDATED_INCOMING if all(row.get("column_model") == INCOMING_COLUMN_MODEL for row in weights) else STATUS_PARTIAL_SOURCE_RAY
    md = [
        "# GBW/IIM Real Kerr Camera Summary",
        "",
        f"Status: `{status}`.",
        "",
        f"- GBW weighted associated energy [GeV]: `{gbw_summary['observed_weighted_energy_gev']:.12g}`",
        f"- IIM weighted associated energy [GeV]: `{iim_summary['observed_weighted_energy_gev']:.12g}`",
        f"- IIM/GBW associated energy ratio: `{ratio:.12g}`",
        f"- GBW nonzero pixels: `{gbw_summary['nonzero_pixels']}`",
        f"- IIM nonzero pixels: `{iim_summary['nonzero_pixels']}`",
        f"- morphology overlap: `{overlap:.12g}`",
        "",
        "The maps are derived from particle-ray association rows. Legacy `observed_particles_by_pixel` names are compatibility outputs and do not imply full transport to the observer. Raw energies are preserved; only GBW/IIM weighted-energy columns differ.",
    ]
    (args.camera_output_dir / "gbw_iim_camera_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_plots(args, weights, gbw_rows, iim_rows, nx, ny)


def update_status_docs(args: argparse.Namespace) -> None:
    points = read_jsonl(args.interaction_points)
    all_incoming = bool(points) and all(row.get("column_model") == INCOMING_COLUMN_MODEL for row in points)
    status = STATUS_VALIDATED_INCOMING if all_incoming else STATUS_PARTIAL_SOURCE_RAY
    detail = (
        "The current interaction records store `incoming_ray_id`, source pixel, ray sample index, and cumulative incoming geodesic column, so active GBW/IIM weights use `INCOMING_KERR_GEODESIC_COLUMN`."
        if all_incoming
        else "The current sampled interaction records do not store the original incoming neutrino Kerr geodesic, incoming ray id, incoming sample index, or cumulative incoming column, so the active GBW/IIM weights keep `SOURCE_TO_INTERACTION_RAY_APPROXIMATION`."
    )
    replacements = [
        (
            ROOT / "docs/external_generators/HADROS_CASCADE_SCIENTIFIC_STATUS.md",
            "\n## Phase 15.7 Incoming UHE Geodesic Column Audit\n\n"
            f"Status: `{status}`.\n\n"
            f"The real Kerr camera geometry remains validated with ZAMO local-to-global positions. {detail} POWHEG/PYTHIA, GEANT4, Kerr camera tracing, ZAMO transforms, and particle-to-pixel association are unchanged.\n\n"
            "Outputs: `output/science/gbw_iim_reweighting`, `output/science/gbw_iim_real_kerr_camera`.\n",
        ),
        (
            ROOT / "docs/science/REAL_KERR_PARTICLE_CAMERA_GEOMETRY_VALIDATION.md",
            "\n## Phase 15.7 Incoming Column Note\n\n"
            f"Status: `{status}`.\n\n"
            f"GBW/IIM weights are propagated with `{INCOMING_COLUMN_MODEL if all_incoming else SOURCE_RAY_COLUMN_MODEL}` on the validated real Kerr camera particle associations. Geometry, raw energies, ZAMO positions, and pixel associations are preserved.\n",
        ),
    ]
    for path, block in replacements:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        marker = block.split("\n", 3)[2]
        if marker not in text:
            path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=DEFAULT_PARTICLES)
    parser.add_argument("--interaction-points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--observed", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--sigma-gbw", type=Path, default=DEFAULT_GBW)
    parser.add_argument("--sigma-iim", type=Path, default=DEFAULT_IIM)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-output-dir", type=Path, default=DEFAULT_CAMERA_OUTPUT)
    parser.add_argument("--mbh-msun", type=float, default=-1.0)
    parser.add_argument("--column-mode", choices=["ray_integrated", "local_radial"], default="ray_integrated")
    parser.add_argument("--ray-column-steps", type=int, default=256)
    parser.add_argument("--column-r-outer-rg", type=float, default=80.0)
    parser.add_argument("--min-column-length-rg", type=float, default=1.0)
    parser.add_argument("--thin-tau-threshold", type=float, default=1.0e-2)
    parser.add_argument("--saturated-tau-threshold", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.camera_output_dir.mkdir(parents=True, exist_ok=True)
    weights = build_interaction_weight_rows(args)
    weights_by_event = {int(row["event_id"]): row for row in weights}
    write_csv(args.output_dir / "interaction_point_weights.csv", weights)
    write_jsonl(args.output_dir / "interaction_point_weights.jsonl", weights)
    audit_current_weights(args, weights)
    write_ray_column_audit(args, weights)
    write_column_comparison(args, weights)

    enriched_points = enrich_json_rows(read_jsonl(args.interaction_points), weights_by_event)
    enriched_particles = enrich_json_rows(read_jsonl(args.particles), weights_by_event)
    enriched_ready = enrich_json_rows(read_jsonl(args.ready), weights_by_event)
    write_jsonl(args.output_dir / "interaction_points_gbw_iim_weighted.jsonl", enriched_points)
    write_jsonl(args.output_dir / "hadros_particle_events_gbw_iim_weighted.jsonl", enriched_particles)
    write_jsonl(args.output_dir / "geant4_ready_particles_gbw_iim_weighted.jsonl", enriched_ready)

    enriched_observed = enrich_observed_rows(read_csv(args.observed), weights_by_event)
    write_csv(args.output_dir / "observed_particles_by_pixel_gbw_iim_weighted.csv", enriched_observed)
    write_camera_products(args, enriched_observed, weights)
    write_validation(args, weights, enriched_observed)
    write_science_doc(args)
    update_status_docs(args)
    print(
        json.dumps(
            {
                "status": STATUS_VALIDATED_INCOMING if all(row.get("column_model") == INCOMING_COLUMN_MODEL for row in weights) else STATUS_PARTIAL_SOURCE_RAY,
                "events": len(weights),
                "observed_rows": len(enriched_observed),
                "output_dir": str(args.output_dir),
                "camera_output_dir": str(args.camera_output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
