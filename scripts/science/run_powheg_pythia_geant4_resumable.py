#!/usr/bin/env python3
"""Run POWHEG+PYTHIA8 particles through GEANT4 real-safe resumable jobs."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output" / "science" / "powheg_pythia_particles" / "hadros_particle_events.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "science" / "powheg_pythia_geant4_resumable"
GEANT4_APP = ROOT / "build" / "cascade_geant4_local_box"
BACKEND = "POWHEG_NUDIS_PYTHIA8"
OUTPUT_BACKEND = "POWHEG_NUDIS_PYTHIA8_GEANT4_REAL_SAFE"

STRICT_GEANT4_PDGS = {22, 11, 13, 211, 321, 130, 310, 2212, 2112}
NEUTRINO_PDGS = {12, 14, 16}
POSITION_FIELDS = [
    "interaction_x_rg",
    "interaction_y_rg",
    "interaction_z_rg",
    "interaction_r_rg",
    "interaction_theta_rad",
    "interaction_phi_rad",
    "interaction_position_status",
    "geant4_box_origin_x_rg",
    "geant4_box_origin_y_rg",
    "geant4_box_origin_z_rg",
    "exit_x_rg",
    "exit_y_rg",
    "exit_z_rg",
    "geant4_exit_local_x_rg",
    "geant4_exit_local_y_rg",
    "geant4_exit_local_z_rg",
    "global_exit_x_rg",
    "global_exit_y_rg",
    "global_exit_z_rg",
    "global_exit_r_rg",
    "global_exit_theta_rad",
    "global_exit_phi_rad",
    "global_position_status",
    "global_position_transform",
    "local_to_global_transform",
    "tetrad_status",
    "cartesian_global_exit_x_rg",
    "cartesian_global_exit_y_rg",
    "cartesian_global_exit_z_rg",
    "cartesian_global_exit_r_rg",
    "cartesian_global_exit_theta_rad",
    "cartesian_global_exit_phi_rad",
    "global_px",
    "global_py",
    "global_pz",
    "global_momentum_status",
    "momentum_input_mode",
    "ready_particle_momentum_frame",
    "momentum_input_mode_policy",
    "n_zamo_r",
    "n_zamo_theta",
    "n_zamo_phi",
    "geant4_box_origin_x_cm",
    "geant4_box_origin_y_cm",
    "geant4_box_origin_z_cm",
    "geant4_local_exit_x_cm",
    "geant4_local_exit_y_cm",
    "geant4_local_exit_z_cm",
    "geant4_local_cm_per_rg",
]

STRING_POSITION_FIELDS = {
    "interaction_position_status",
    "global_position_status",
    "global_position_transform",
    "local_to_global_transform",
    "tetrad_status",
    "global_momentum_status",
    "momentum_input_mode",
    "ready_particle_momentum_frame",
    "momentum_input_mode_policy",
}


def rg_cm_from_mbh(mbh_msun: float) -> float:
    # Gravitational radius GM/c^2 in cm.
    return 1.4766250385e5 * mbh_msun


def validate_geant4_local_cm_per_rg(cm_per_rg: float, mbh_msun: float) -> float:
    expected = rg_cm_from_mbh(mbh_msun)
    if not math.isfinite(cm_per_rg) or cm_per_rg <= 10.0:
        raise ValueError(
            f"Invalid geant4-local-cm-per-rg={cm_per_rg!r}; expected GM/c^2 scale "
            f"({expected:.12g} cm for MBH_MSUN={mbh_msun:g})."
        )
    if abs(cm_per_rg - 1.0) <= 1.0e-9:
        raise ValueError(
            "Invalid geant4-local-cm-per-rg=1.0 for astrophysical HADROS-CASCADE run; "
            f"use GM/c^2 ({expected:.12g} cm for MBH_MSUN={mbh_msun:g})."
        )
    rel = abs(cm_per_rg - expected) / max(expected, 1.0)
    if rel > 1.0e-6:
        raise ValueError(
            f"geant4-local-cm-per-rg={cm_per_rg:.12g} is inconsistent with GM/c^2="
            f"{expected:.12g} for MBH_MSUN={mbh_msun:g}."
        )
    return cm_per_rg


def geant4_env() -> dict[str, str]:
    env = os.environ.copy()
    data = Path("/home/rafael/micromamba/envs/hadros-cascade/share/Geant4/data")
    mapping = {
        "G4ENSDFSTATEDATA": data / "ENSDFSTATE3.0",
        "G4LEVELGAMMADATA": data / "PhotonEvaporation6.1.2",
        "G4RADIOACTIVEDATA": data / "RadioactiveDecay6.1.2",
        "G4LEDATA": data / "EMLOW8.8",
        "G4PARTICLEXSDATA": data / "PARTICLEXS4.2",
        "G4NEUTRONHPDATA": data / "NDL4.7.1",
        "G4INCLDATA": data / "INCL1.3",
        "G4ABLADATA": data / "ABLA3.3",
        "G4PIIDATA": data / "PII1.3",
        "G4REALSURFACEDATA": data / "RealSurface2.2",
        "G4SAIDXSDATA": data / "SAIDDATA2.0",
    }
    for key, path in mapping.items():
        if path.exists():
            env.setdefault(key, str(path))
    return env


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def spherical_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(r) or r <= 0.0:
        return 0.0, 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return r, theta, phi


def xyz_from_spherical(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    sin_theta = math.sin(theta)
    return (
        r * sin_theta * math.cos(phi),
        r * sin_theta * math.sin(phi),
        r * math.cos(theta),
    )


def kerr_spatial_metric(spin: float, r: float, theta: float) -> tuple[float, float, float] | None:
    sigma = r * r + spin * spin * math.cos(theta) ** 2
    delta = r * r - 2.0 * r + spin * spin
    sin_theta = math.sin(theta)
    a_term = (r * r + spin * spin) ** 2 - spin * spin * delta * sin_theta * sin_theta
    if sigma <= 0.0 or delta <= 0.0 or a_term <= 0.0:
        return None
    grr = sigma / delta
    gthth = sigma
    gphph = a_term * sin_theta * sin_theta / sigma
    if grr <= 0.0 or gthth <= 0.0 or gphph <= 0.0:
        return None
    return grr, gthth, gphph


def finite_value(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def copy_position_fields(src: dict[str, Any], dst: dict[str, Any]) -> None:
    for field in POSITION_FIELDS:
        value = finite_value(src, field)
        if value is not None:
            dst[field] = value
        elif field in STRING_POSITION_FIELDS and src.get(field):
            dst[field] = str(src[field])
    if src.get("position_status"):
        dst["position_status"] = str(src["position_status"])


def load_interaction_positions(path: Path | None, mbh_msun: float) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rg_cm = rg_cm_from_mbh(mbh_msun)
    positions: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        event_id = int(row.get("event_id", row.get("primary", {}).get("event_id", 0)) or 0)
        point = row.get("point") if isinstance(row.get("point"), dict) else row
        if {"interaction_x_rg", "interaction_y_rg", "interaction_z_rg"}.issubset(point):
            x = float(point["interaction_x_rg"])
            y = float(point["interaction_y_rg"])
            z = float(point["interaction_z_rg"])
        elif {"x", "y", "z"}.issubset(point):
            x = float(point["x"])
            y = float(point["y"])
            z = float(point["z"])
        elif {"x_cm", "y_cm", "z_cm"}.issubset(point):
            x = float(point["x_cm"]) / rg_cm
            y = float(point["y_cm"]) / rg_cm
            z = float(point["z_cm"]) / rg_cm
        else:
            continue
        r, theta, phi = spherical_from_xyz(x, y, z)
        if event_id > 0 and r > 0.0:
            positions[event_id] = {
                "interaction_x_rg": x,
                "interaction_y_rg": y,
                "interaction_z_rg": z,
                "interaction_r_rg": r,
                "interaction_theta_rad": theta,
                "interaction_phi_rad": phi,
                "interaction_position_status": "GLOBAL_POSITION_VALID",
            }
    return positions


def attach_global_exit_position(row: dict[str, Any], *, spin: float, transform: str = "zamo_tetrad") -> None:
    for src, dst in [
        ("exit_x_rg", "geant4_exit_local_x_rg"),
        ("exit_y_rg", "geant4_exit_local_y_rg"),
        ("exit_z_rg", "geant4_exit_local_z_rg"),
    ]:
        value = finite_value(row, src)
        if value is not None:
            row.setdefault(dst, value)
    ix = finite_value(row, "interaction_x_rg")
    iy = finite_value(row, "interaction_y_rg")
    iz = finite_value(row, "interaction_z_rg")
    lx = finite_value(row, "geant4_exit_local_x_rg")
    ly = finite_value(row, "geant4_exit_local_y_rg")
    lz = finite_value(row, "geant4_exit_local_z_rg")
    if ix is None or iy is None or iz is None:
        row["interaction_position_status"] = "MISSING_INTERACTION_POSITION"
        row["global_position_status"] = "MISSING_INTERACTION_POSITION"
        return
    if lx is None or ly is None or lz is None:
        row["global_position_status"] = "MISSING_GEANT4_EXIT_POSITION"
        return
    cgx = ix + lx
    cgy = iy + ly
    cgz = iz + lz
    cr, ctheta, cphi = spherical_from_xyz(cgx, cgy, cgz)
    row["cartesian_global_exit_x_rg"] = cgx
    row["cartesian_global_exit_y_rg"] = cgy
    row["cartesian_global_exit_z_rg"] = cgz
    row["cartesian_global_exit_r_rg"] = cr
    row["cartesian_global_exit_theta_rad"] = ctheta
    row["cartesian_global_exit_phi_rad"] = cphi
    horizon = 1.0 + math.sqrt(max(1.0 - spin * spin, 0.0))
    if transform == "local_cartesian":
        gx, gy, gz = cgx, cgy, cgz
        r, theta, phi = cr, ctheta, cphi
        transform_name = "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION"
        status_name = "GLOBAL_POSITION_VALID"
        tetrad_status = "NOT_USED_LOCAL_CARTESIAN_DEBUG"
    else:
        ir = finite_value(row, "interaction_r_rg")
        itheta = finite_value(row, "interaction_theta_rad")
        iphi = finite_value(row, "interaction_phi_rad")
        if ir is None or itheta is None or iphi is None or ir <= horizon:
            row["global_position_status"] = "INVALID_INTERACTION_POSITION_FOR_ZAMO_TETRAD"
            row["tetrad_status"] = "ZAMO_TETRAD_BLOCKED_BY_INTERACTION_POSITION"
            return
        metric = kerr_spatial_metric(spin, ir, itheta)
        if metric is None:
            row["global_position_status"] = "BAD_KERR_SPATIAL_METRIC"
            row["tetrad_status"] = "ZAMO_TETRAD_BAD_METRIC"
            return
        grr, gthth, gphph = metric
        # GEANT4 local-box convention for escaped particles:
        # local z is the outward ZAMO radial axis, local x is +theta,
        # and local y is +phi. The values are local orthonormal lengths in rg.
        dr = lz / math.sqrt(grr)
        dtheta = lx / math.sqrt(gthth)
        dphi = ly / math.sqrt(gphph)
        r = ir + dr
        theta = max(1.0e-9, min(math.pi - 1.0e-9, itheta + dtheta))
        phi = iphi + dphi
        gx, gy, gz = xyz_from_spherical(r, theta, phi)
        transform_name = "ZAMO_TETRAD_LOCAL_BOX"
        status_name = "GLOBAL_POSITION_VALID_ZAMO_TETRAD"
        tetrad_status = "ZAMO_TETRAD_LOCAL_DISPLACEMENT_VALID"
    if r <= 0.0:
        row["global_position_status"] = "INVALID_GLOBAL_POSITION"
        return
    if r <= horizon:
        row["global_position_status"] = "INSIDE_HORIZON"
        return
    row["global_exit_x_rg"] = gx
    row["global_exit_y_rg"] = gy
    row["global_exit_z_rg"] = gz
    row["global_exit_r_rg"] = r
    row["global_exit_theta_rad"] = theta
    row["global_exit_phi_rad"] = phi
    row["global_position_status"] = status_name
    row["global_position_transform"] = transform_name
    row["local_to_global_transform"] = transform_name
    row["tetrad_status"] = tetrad_status

    px = finite_value(row, "px")
    py = finite_value(row, "py")
    pz = finite_value(row, "pz")
    if px is None or py is None or pz is None:
        row["global_momentum_status"] = "GLOBAL_MOMENTUM_NOT_AVAILABLE"
        row["momentum_input_mode"] = "unknown"
        row["ready_particle_momentum_frame"] = "unknown"
        row["momentum_input_mode_policy"] = "explicit_mode_required_for_photon_observer_camera"
        return
    momentum_norm = math.sqrt(px * px + py * py + pz * pz)
    if not math.isfinite(momentum_norm) or momentum_norm <= 0.0:
        row["global_momentum_status"] = "GLOBAL_MOMENTUM_INVALID"
        row["momentum_input_mode"] = "unknown"
        row["ready_particle_momentum_frame"] = "unknown"
        row["momentum_input_mode_policy"] = "explicit_mode_required_for_photon_observer_camera"
        return
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    sin_p = math.sin(phi)
    cos_p = math.cos(phi)
    er = (sin_t * cos_p, sin_t * sin_p, cos_t)
    etheta = (cos_t * cos_p, cos_t * sin_p, -sin_t)
    ephi = (-sin_p, cos_p, 0.0)
    p_radial = pz
    p_theta = px
    p_phi = py
    row["global_px"] = p_radial * er[0] + p_theta * etheta[0] + p_phi * ephi[0]
    row["global_py"] = p_radial * er[1] + p_theta * etheta[1] + p_phi * ephi[1]
    row["global_pz"] = p_radial * er[2] + p_theta * etheta[2] + p_phi * ephi[2]
    if transform_name == "ZAMO_TETRAD_LOCAL_BOX":
        row["global_momentum_status"] = "GLOBAL_MOMENTUM_ZAMO_SPATIAL_TRIAD"
        row["momentum_input_mode"] = "zamo_tetrad"
        row["ready_particle_momentum_frame"] = "ZAMO_TETRAD_LOCAL_BOX"
        row["momentum_input_mode_policy"] = "explicit_zamo_tetrad_components"
        # GEANT4 local-box convention: local z is +radial ZAMO,
        # local x is +theta, and local y is +phi.
        row["n_zamo_r"] = pz / momentum_norm
        row["n_zamo_theta"] = px / momentum_norm
        row["n_zamo_phi"] = py / momentum_norm
    else:
        row["global_momentum_status"] = "GLOBAL_MOMENTUM_LOCAL_CARTESIAN_DEBUG"
        row["momentum_input_mode"] = "unknown"
        row["ready_particle_momentum_frame"] = "unknown"
        row["momentum_input_mode_policy"] = "explicit_mode_required_for_photon_observer_camera"


def apply_geant4_local_rg_scale(row: dict[str, Any], cm_per_rg: float) -> None:
    if not math.isfinite(cm_per_rg) or cm_per_rg <= 0.0:
        return
    for axis in ["x", "y", "z"]:
        cm_value = finite_value(row, f"geant4_local_exit_{axis}_cm")
        if cm_value is None:
            continue
        rg_value = cm_value / cm_per_rg
        row[f"geant4_exit_local_{axis}_rg"] = rg_value
        row[f"exit_{axis}_rg"] = rg_value
    row["geant4_local_cm_per_rg"] = cm_per_rg


def channel(pdg: int) -> str:
    apdg = abs(pdg)
    if apdg in NEUTRINO_PDGS:
        return "neutrino"
    if apdg in {11, 13, 15}:
        return "lepton"
    if apdg == 22:
        return "gamma"
    if apdg in {111, 211, 130, 310, 311, 321}:
        return "meson"
    if apdg in {2212, 2112} or 1000 <= apdg < 10000:
        return "hadron"
    return "other"


def supported_transport_pdg(pdg: int) -> bool:
    return abs(pdg) in STRICT_GEANT4_PDGS


def invariant_mass(row: dict[str, Any]) -> float:
    e = fnum(row, "energy_gev")
    px = fnum(row, "px")
    py = fnum(row, "py")
    pz = fnum(row, "pz")
    m2 = e * e - px * px - py * py - pz * pz
    if not math.isfinite(m2):
        return 0.0
    return math.sqrt(max(m2, 0.0))


def kinetic_energy(row: dict[str, Any]) -> float:
    return max(fnum(row, "energy_gev") - max(fnum(row, "mass_gev", invariant_mass(row)), 0.0), 0.0)


def threshold_for(row: dict[str, Any], args: argparse.Namespace) -> float:
    ch = channel(int(row["pdg"]))
    if ch == "gamma":
        return args.geant4_photon_max_kinetic_gev
    if ch == "lepton":
        return args.geant4_lepton_max_kinetic_gev
    return args.geant4_hadron_max_kinetic_gev


def normalize_input(rows: list[dict[str, Any]], interaction_positions: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[int, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("generator_backend") != BACKEND:
            raise ValueError(f"unexpected generator_backend: {row.get('generator_backend')}")
        event_id = int(row["event_id"])
        counters[event_id] += 1
        particle_id = int(row.get("particle_id", counters[event_id]))
        energy = fnum(row, "energy_gev", fnum(row, "energy", 0.0))
        norm = {
            "event_id": event_id,
            "particle_id": particle_id,
            "source_particle_id": particle_id,
            "pdg": int(row["pdg"]),
            "pdg_id": int(row["pdg"]),
            "energy_gev": energy,
            "px_gev": fnum(row, "px"),
            "py_gev": fnum(row, "py"),
            "pz_gev": fnum(row, "pz"),
            "mass_gev": fnum(row, "mass_gev", 0.0),
            "weight": fnum(row, "weight", 1.0),
            "stable": 1,
            "interaction_type": str(row.get("interaction_type", "")),
            "target_type": str(row.get("target_type", "")),
            "generator_backend": str(row.get("generator_backend", "")),
            "origin": BACKEND,
            "origin_backend": BACKEND,
        }
        copy_position_fields(row, norm)
        norm.setdefault("position_status", "MISSING_PARTICLE_POSITION")
        if norm["mass_gev"] <= 0.0:
            norm["mass_gev"] = invariant_mass({**norm, "px": norm["px_gev"], "py": norm["py_gev"], "pz": norm["pz_gev"]})
        if event_id in interaction_positions:
            norm.update(interaction_positions[event_id])
        else:
            norm.setdefault("interaction_position_status", "MISSING_INTERACTION_POSITION")
        out.append(norm)
    return out


def classify(row: dict[str, Any], args: argparse.Namespace) -> str:
    pdg = int(row["pdg"])
    energy = fnum(row, "energy_gev")
    if not math.isfinite(energy) or energy < 0.0:
        return "untracked"
    if abs(pdg) in NEUTRINO_PDGS:
        return "invisible"
    if not supported_transport_pdg(pdg):
        return "untracked"
    if energy + 1.0e-12 < max(fnum(row, "mass_gev"), 0.0):
        return "untracked"
    if fnum(row, "px_gev") ** 2 + fnum(row, "py_gev") ** 2 + fnum(row, "pz_gev") ** 2 <= 0.0:
        return "untracked"
    if kinetic_energy(row) > threshold_for(row, args):
        return "unsupported_uhe"
    return "transportable"


def job_name(row: dict[str, Any]) -> str:
    return f"job_{int(row['event_id'])}_{int(row['source_particle_id'])}"


def status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def load_status(job_dir: Path) -> dict[str, Any] | None:
    path = status_path(job_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def geant4_output_has_position_status(job_dir: Path) -> bool:
    path = job_dir / "outputs" / "geant4_escaped_particles.jsonl"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return not text.strip() or "position_status" in text


def save_status(job_dir: Path, status: dict[str, Any]) -> None:
    status_path(job_dir).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def geant4_input(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "event_id": row["event_id"],
        "parent_event_id": row["event_id"],
        "pdg": row["pdg"],
        "pdg_id": row["pdg_id"],
        "energy_gev": row["energy_gev"],
        "px_gev": row["px_gev"],
        "py_gev": row["py_gev"],
        "pz_gev": row["pz_gev"],
        "mass_gev": row["mass_gev"],
        "weight": row["weight"],
        "stable": 1,
        "origin": BACKEND,
        "origin_backend": BACKEND,
    }
    copy_position_fields(row, payload)
    return payload


def run_job(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    name = job_name(row)
    job_dir = args.output_dir / "jobs" / name
    outputs = job_dir / "outputs"
    job_dir.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    existing = load_status(job_dir)
    if (
        existing
        and existing.get("status") == "PASS"
        and args.rerun_missing_positions
        and geant4_output_has_position_status(job_dir)
    ):
        return {**existing, "skipped_existing_positioned_pass": True}
    if existing and existing.get("status") == "PASS" and not args.rerun_pass and not args.rerun_missing_positions:
        return {**existing, "skipped_existing_pass": True}
    if existing and existing.get("status") in {"FAILED", "TIMEOUT"} and not args.retry_failed:
        return {**existing, "skipped_existing_failed": True}

    input_path = job_dir / "input.jsonl"
    write_jsonl(input_path, [geant4_input(row)])
    cmd = [
        str(args.geant4_app),
        str(input_path),
        str(outputs),
        f"{args.box_size_cm:.17g}",
        f"{args.density_g_cm3:.17g}",
        args.physics_list,
        args.material,
        "geant4",
        "--geant4-safety-mode",
        "strict",
        "--uhe-transport-policy",
        "skip_to_escaped",
        "--geant4-hadron-max-kinetic-gev",
        f"{args.geant4_hadron_max_kinetic_gev:.17g}",
        "--geant4-lepton-max-kinetic-gev",
        f"{args.geant4_lepton_max_kinetic_gev:.17g}",
        "--geant4-photon-max-kinetic-gev",
        f"{args.geant4_photon_max_kinetic_gev:.17g}",
        "--geant4-local-cm-per-rg",
        f"{args.geant4_local_cm_per_rg:.17g}",
        "--energy-convention",
        "total",
    ]
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=geant4_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_s,
            check=False,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    runtime = time.monotonic() - start

    (job_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (job_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    status = "TIMEOUT" if timed_out else ("PASS" if returncode == 0 else "FAILED")
    result = {
        "job": name,
        "status": status,
        "returncode": returncode,
        "runtime_s": runtime,
        "event_id": row["event_id"],
        "particle_id": row["particle_id"],
        "source_particle_id": row["source_particle_id"],
        "pdg": row["pdg"],
        "energy_gev": row["energy_gev"],
        "interaction_type": row["interaction_type"],
        "command": cmd,
    }
    save_status(job_dir, result)
    return result


def aggregate(rows: list[dict[str, Any]], statuses: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    status_by_name = {row["job"]: row for row in statuses}
    event_buckets: dict[int, dict[str, Any]] = {}
    particle_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []

    def bucket_for(row: dict[str, Any]) -> dict[str, Any]:
        event_id = int(row["event_id"])
        return event_buckets.setdefault(
            event_id,
            {
                "event_id": event_id,
                "interaction_type": row["interaction_type"],
                "input_energy": 0.0,
                "deposited_energy": 0.0,
                "escaped_energy": 0.0,
                "invisible_energy": 0.0,
                "unsupported_uhe_energy": 0.0,
                "untracked_energy": 0.0,
                "geant4_failed_energy": 0.0,
                "geant4_budget_residual_energy": 0.0,
                "n_particles_input": 0,
                "n_particles_transportable": 0,
                "n_particles_pass": 0,
                "n_particles_failed": 0,
            },
        )

    for row in rows:
        b = bucket_for(row)
        energy = fnum(row, "energy_gev")
        b["input_energy"] += energy
        b["n_particles_input"] += 1
        cls = row["pre_geant4_classification"]
        transport_status = cls
        deposited = escaped = invisible = unsupported = untracked = failed = 0.0
        residual = 0.0
        escaped_particles = []
        if cls == "invisible":
            invisible = energy
            b["invisible_energy"] += energy
        elif cls == "unsupported_uhe":
            unsupported = energy
            b["unsupported_uhe_energy"] += energy
        elif cls == "untracked":
            untracked = energy
            b["untracked_energy"] += energy
        elif cls == "transportable":
            b["n_particles_transportable"] += 1
            status = status_by_name.get(job_name(row), {"status": "NOT_RUN"})
            transport_status = str(status.get("status", "NOT_RUN"))
            if status.get("status") != "PASS":
                failed = energy
                b["geant4_failed_energy"] += energy
                b["n_particles_failed"] += 1
            else:
                b["n_particles_pass"] += 1
                budget_rows = read_csv(args.output_dir / "jobs" / job_name(row) / "outputs" / "geant4_energy_budget.csv")
                if budget_rows:
                    budget = budget_rows[0]
                    deposited = fnum(budget, "deposited_energy_gev")
                    escaped = fnum(budget, "escaped_energy_gev") + fnum(budget, "escaped_unsupported_uhe_energy_gev")
                    invisible = fnum(budget, "invisible_energy_gev")
                    untracked = fnum(budget, "untracked_energy_gev")
                    unsupported = fnum(budget, "unsupported_uhe_energy_gev")
                    residual = energy - (deposited + escaped + invisible + untracked + unsupported)
                b["deposited_energy"] += deposited
                b["escaped_energy"] += escaped
                b["invisible_energy"] += invisible
                b["untracked_energy"] += untracked
                b["unsupported_uhe_energy"] += unsupported
                b["geant4_budget_residual_energy"] += residual
                escaped_particles = read_jsonl_lenient(args.output_dir / "jobs" / job_name(row) / "outputs" / "geant4_escaped_particles.jsonl")
                for escaped_index, particle in enumerate(escaped_particles, start=1):
                    track_id = int(particle.get("track_id", 0) or 0)
                    if track_id <= 0:
                        track_id = int(row["source_particle_id"]) * 1_000_000 + escaped_index
                    parent_track_id = int(particle.get("parent_track_id", 0) or 0)
                    if parent_track_id <= 0:
                        parent_track_id = int(row["source_particle_id"])
                    ready = {
                        "event_id": row["event_id"],
                        "particle_id": row["particle_id"],
                        "source_particle_id": row["source_particle_id"],
                        "track_id": track_id,
                        "parent_track_id": parent_track_id,
                        "pdg": int(particle.get("pdg_id", particle.get("pdg", row["pdg"]))),
                        "energy_gev": fnum(particle, "energy_gev"),
                        "px": fnum(particle, "px_gev"),
                        "py": fnum(particle, "py_gev"),
                        "pz": fnum(particle, "pz_gev"),
                        "weight": row["weight"],
                        "interaction_type": row["interaction_type"],
                        "target_type": row["target_type"],
                        "origin_backend": OUTPUT_BACKEND,
                        "generator_backend": BACKEND,
                        "transport_backend": "GEANT4_LOCAL_BOX_REAL_SAFE",
                        "transport_status": "GEANT4_ESCAPED",
                    }
                    copy_position_fields(row, ready)
                    copy_position_fields(particle, ready)
                    apply_geant4_local_rg_scale(ready, args.geant4_local_cm_per_rg)
                    attach_global_exit_position(ready, spin=args.spin, transform=args.local_to_global_transform)
                    ready.setdefault("position_status", "MISSING_PARTICLE_POSITION")
                    ready_rows.append(ready)
        particle_rows.append(
            {
                "event_id": row["event_id"],
                "particle_id": row["particle_id"],
                "source_particle_id": row["source_particle_id"],
                "interaction_type": row["interaction_type"],
                "pdg": row["pdg"],
                "channel": channel(row["pdg"]),
                "energy_gev": energy,
                "pre_geant4_classification": cls,
                "transport_status": transport_status,
                "deposited_energy": deposited,
                "escaped_energy": escaped,
                "invisible_energy": invisible,
                "unsupported_uhe_energy": unsupported,
                "untracked_energy": untracked,
                "geant4_failed_energy": failed,
                "geant4_budget_residual_energy": residual,
            }
        )

    event_rows = []
    for event_id in sorted(event_buckets):
        b = event_buckets[event_id]
        accounted = (
            b["deposited_energy"]
            + b["escaped_energy"]
            + b["invisible_energy"]
            + b["unsupported_uhe_energy"]
            + b["untracked_energy"]
            + b["geant4_failed_energy"]
            + b["geant4_budget_residual_energy"]
        )
        event_rows.append(
            {
                **b,
                "closure_error": accounted - b["input_energy"],
                "relative_closure_error": abs(accounted - b["input_energy"]) / max(b["input_energy"], 1.0),
            }
        )

    write_csv(args.output_dir / "powheg_pythia_geant4_resumable_events.csv", event_rows)
    write_csv(args.output_dir / "powheg_pythia_geant4_resumable_summary.csv", event_rows)
    write_csv(args.output_dir / "powheg_pythia_geant4_resumable_particles.csv", particle_rows)
    write_csv(args.output_dir / "powheg_pythia_geant4_resumable_energy_closure.csv", event_rows)
    write_jsonl(args.output_dir / "geant4_ready_particles.jsonl", ready_rows)

    summary = {
        "events": len(event_rows),
        "particles": len(rows),
        "transportable_particles": sum(1 for row in rows if row["pre_geant4_classification"] == "transportable"),
        "jobs_pass": sum(1 for row in statuses if row.get("status") == "PASS"),
        "jobs_failed": sum(1 for row in statuses if row.get("status") in {"FAILED", "TIMEOUT"}),
        "jobs_recorded": len(statuses),
        "input_energy": sum(r["input_energy"] for r in event_rows),
        "deposited_energy": sum(r["deposited_energy"] for r in event_rows),
        "escaped_energy": sum(r["escaped_energy"] for r in event_rows),
        "invisible_energy": sum(r["invisible_energy"] for r in event_rows),
        "unsupported_uhe_energy": sum(r["unsupported_uhe_energy"] for r in event_rows),
        "untracked_energy": sum(r["untracked_energy"] for r in event_rows),
        "geant4_failed_energy": sum(r["geant4_failed_energy"] for r in event_rows),
        "geant4_budget_residual_energy": sum(r["geant4_budget_residual_energy"] for r in event_rows),
        "max_closure_error": max((abs(r["closure_error"]) for r in event_rows), default=0.0),
        "max_abs_geant4_budget_residual_energy": max((abs(r["geant4_budget_residual_energy"]) for r in event_rows), default=0.0),
        "ready_particles": len(ready_rows),
    }
    summary["camera_status"] = "CAMERA_READY" if ready_rows else "CAMERA_BLOCKED_NO_GEANT4_ESCAPED_PARTICLES"
    summary["ready_particles_with_global_position"] = sum(
        1 for row in ready_rows if str(row.get("global_position_status", "")).startswith("GLOBAL_POSITION_VALID")
    )
    summary["ready_particles_with_zamo_tetrad_position"] = sum(
        1 for row in ready_rows if row.get("global_position_transform") == "ZAMO_TETRAD_LOCAL_BOX"
    )
    summary["ready_particles_with_explicit_zamo_momentum"] = sum(
        1 for row in ready_rows if row.get("momentum_input_mode") == "zamo_tetrad"
    )
    summary["ready_particle_momentum_frame"] = (
        "ZAMO_TETRAD_LOCAL_BOX"
        if summary["ready_particles_with_explicit_zamo_momentum"] == len(ready_rows) and ready_rows
        else "mixed_or_unavailable"
    )
    summary["momentum_input_mode_policy"] = "explicit_per_ready_particle"
    summary["local_to_global_transform"] = args.local_to_global_transform
    summary["geant4_local_cm_per_rg"] = args.geant4_local_cm_per_rg
    summary["mbh_msun"] = args.mbh_msun
    # Threshold for VALIDATED status: at least as many events as were requested,
    # or 10 if not specified — avoids permanent PARTIAL for small scientific runs.
    validation_min_events = getattr(args, "validation_event_threshold", None) or max(10, summary["transportable_particles"])
    summary["status"] = (
        "POWHEG_PYTHIA_GEANT4_RESUMABLE_VALIDATED"
        if summary["events"] >= validation_min_events
        and summary["jobs_recorded"] >= summary["transportable_particles"]
        and summary["jobs_pass"] + summary["jobs_failed"] >= summary["transportable_particles"]
        and summary["jobs_pass"] > 0
        and summary["max_closure_error"] < 1.0e-9
        and summary["max_abs_geant4_budget_residual_energy"] < 1.0
        and ready_rows
        else "POWHEG_PYTHIA_GEANT4_RESUMABLE_PARTIAL"
    )
    return summary


def write_reports(summary: dict[str, Any], statuses: list[dict[str, Any]], args: argparse.Namespace) -> None:
    status_counts = Counter(str(row.get("status", "")) for row in statuses)
    lines = [
        "# POWHEG PYTHIA GEANT4 Resumable Summary",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"Input backend: `{BACKEND}`.",
        "Execution: `one-particle-per-process resumable batches`.",
        "",
        f"Events: `{summary['events']}`.",
        f"Input particles: `{summary['particles']}`.",
        f"Transportable particles: `{summary['transportable_particles']}`.",
        f"Jobs PASS: `{summary['jobs_pass']}`.",
        f"Jobs failed: `{summary['jobs_failed']}`.",
        f"Input energy [GeV]: `{summary['input_energy']:.12g}`.",
        f"Deposited energy [GeV]: `{summary['deposited_energy']:.12g}`.",
        f"Escaped energy [GeV]: `{summary['escaped_energy']:.12g}`.",
        f"Invisible energy [GeV]: `{summary['invisible_energy']:.12g}`.",
        f"Unsupported UHE energy [GeV]: `{summary['unsupported_uhe_energy']:.12g}`.",
        f"Untracked energy [GeV]: `{summary['untracked_energy']:.12g}`.",
        f"GEANT4 failed energy [GeV]: `{summary['geant4_failed_energy']:.12g}`.",
        f"GEANT4 budget residual energy [GeV]: `{summary['geant4_budget_residual_energy']:.12g}`.",
        f"Max closure error [GeV]: `{summary['max_closure_error']:.12g}`.",
        f"Max abs GEANT4 budget residual [GeV]: `{summary['max_abs_geant4_budget_residual_energy']:.12g}`.",
        f"Camera status: `{summary['camera_status']}`.",
        f"Local-to-global transform: `{summary.get('local_to_global_transform', '')}`.",
        f"Ready-particle momentum frame: `{summary.get('ready_particle_momentum_frame', '')}`.",
        f"Momentum input mode policy: `{summary.get('momentum_input_mode_policy', '')}`.",
        f"GEANT4 local cm per rg: `{summary.get('geant4_local_cm_per_rg', '')}`.",
        f"MBH_MSUN: `{summary.get('mbh_msun', '')}`.",
        f"Ready particles with ZAMO tetrad positions: `{summary.get('ready_particles_with_zamo_tetrad_position', 0)}`.",
        f"Ready particles with explicit ZAMO momentum: `{summary.get('ready_particles_with_explicit_zamo_momentum', 0)}`.",
        "",
        "## Job Status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "No PYTHIA e+e- proxy, local response proxy, synthetic particle source, or camera projection is used.",
        ]
    )
    (args.output_dir / "powheg_pythia_geant4_resumable_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    closure = [
        "# POWHEG PYTHIA GEANT4 Resumable Energy Closure",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"max_closure_error: `{summary['max_closure_error']:.17g}`",
        f"max_abs_geant4_budget_residual_energy: `{summary['max_abs_geant4_budget_residual_energy']:.17g}`",
        "",
        "Energy is closed by construction per event:",
        "",
        "```text",
        "E_in = E_dep + E_esc + E_invisible + E_unsupported_uhe + E_untracked + E_geant4_failed + E_geant4_budget_residual",
        "```",
        "",
        "`E_geant4_budget_residual` is the signed residual between the one-particle GEANT4 budget",
        "and the original POWHEG+PYTHIA particle energy. It is reported explicitly instead of being hidden.",
    ]
    (args.output_dir / "powheg_pythia_geant4_resumable_energy_closure.md").write_text("\n".join(closure) + "\n", encoding="utf-8")


def maybe_plot(args: argparse.Namespace) -> None:
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hadros-powheg")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    events = read_csv(args.output_dir / "powheg_pythia_geant4_resumable_events.csv")
    particles = read_csv(args.output_dir / "powheg_pythia_geant4_resumable_particles.csv")

    def bar(path: Path, labels: list[str], values: list[float], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#2f6f8f")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    fields = [
        "deposited_energy",
        "escaped_energy",
        "invisible_energy",
        "unsupported_uhe_energy",
        "untracked_energy",
        "geant4_failed_energy",
        "geant4_budget_residual_energy",
    ]
    totals = [sum(fnum(row, field) for row in events) for field in fields]
    bar(plots / "energy_budget_resumable.png", fields, totals, "energy [GeV]")

    status_energy = defaultdict(float)
    pdg_status = defaultdict(Counter)
    escaped_channel = defaultdict(float)
    for row in particles:
        status_energy[str(row["transport_status"])] += fnum(row, "energy_gev")
        pdg_status[int(row["pdg"])][str(row["transport_status"])] += 1
        escaped_channel[str(row["channel"])] += fnum(row, "escaped_energy")
    bar(plots / "particle_transport_status.png", list(status_energy), list(status_energy.values()), "input energy [GeV]")
    bar(plots / "escaped_energy_by_channel.png", list(escaped_channel), list(escaped_channel.values()), "escaped energy [GeV]")

    pass_fail = defaultdict(lambda: {"PASS": 0, "FAILED": 0})
    for row in particles:
        status = str(row["transport_status"])
        if status in {"PASS", "FAILED", "TIMEOUT"}:
            pass_fail[int(row["pdg"])]["PASS" if status == "PASS" else "FAILED"] += 1
    labels = [str(pdg) for pdg in sorted(pass_fail)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(labels)))
    ax.bar([v - 0.2 for v in x], [pass_fail[int(label)]["PASS"] for label in labels], width=0.4, label="PASS")
    ax.bar([v + 0.2 for v in x], [pass_fail[int(label)]["FAILED"] for label in labels], width=0.4, label="FAILED")
    ax.set_xticks(x, labels, rotation=45)
    ax.set_ylabel("jobs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "geant4_pass_fail_by_pdg.png", dpi=160)
    plt.close(fig)

    by_interaction = defaultdict(lambda: defaultdict(float))
    for row in events:
        for field in fields:
            by_interaction[str(row["interaction_type"])][field] += fnum(row, field)
    labels2 = fields
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(labels2)))
    for idx, interaction in enumerate(["CC", "NC"]):
        ax.bar([v + (idx - 0.5) * 0.35 for v in x], [by_interaction[interaction][field] for field in labels2], width=0.35, label=interaction)
    ax.set_xticks(x, labels2, rotation=35)
    ax.set_ylabel("energy [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "cc_vs_nc_energy_budget_resumable.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geant4-app", type=Path, default=GEANT4_APP)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun-pass", action="store_true")
    parser.add_argument("--rerun-missing-positions", action="store_true")
    parser.add_argument("--box-size-cm", type=float, default=100.0)
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--physics-list", choices=["FTFP_BERT", "QGSP_BERT"], default="FTFP_BERT")
    parser.add_argument("--material", choices=["hydrogen", "water"], default="hydrogen")
    parser.add_argument("--geant4-hadron-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-lepton-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-photon-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-local-cm-per-rg", type=float, required=True)
    parser.add_argument("--interaction-points", type=Path, default=None)
    parser.add_argument("--mbh-msun", type=float, default=3.0)
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument(
        "--local-to-global-transform",
        choices=["zamo_tetrad", "local_cartesian"],
        default="zamo_tetrad",
        help="Transform GEANT4 local-box exits to global Kerr positions. Scientific default: zamo_tetrad.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.geant4_app.exists():
        raise SystemExit(f"GEANT4 app not found: {args.geant4_app}")
    try:
        args.geant4_local_cm_per_rg = validate_geant4_local_cm_per_rg(args.geant4_local_cm_per_rg, args.mbh_msun)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.interaction_points is not None and not args.interaction_points.is_absolute():
        args.interaction_points = (ROOT / args.interaction_points).resolve()
    interaction_positions = load_interaction_positions(args.interaction_points, args.mbh_msun)
    rows = normalize_input(read_jsonl(args.input), interaction_positions)
    for row in rows:
        row["pre_geant4_classification"] = classify(row, args)

    transportable = [row for row in rows if row["pre_geant4_classification"] == "transportable"]
    jobs = transportable if args.max_jobs <= 0 else transportable[: args.max_jobs]
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(run_job, row, args): row for row in jobs}
        for future in as_completed(future_map):
            statuses.append(future.result())
    for row in transportable:
        if row not in jobs:
            status = load_status(args.output_dir / "jobs" / job_name(row))
            if status:
                statuses.append(status)

    statuses.sort(key=lambda item: str(item.get("job", "")))
    write_csv(
        args.output_dir / "geant4_job_status.csv",
        statuses,
        ["job", "status", "returncode", "runtime_s", "event_id", "particle_id", "source_particle_id", "pdg", "energy_gev", "interaction_type"],
    )
    summary = aggregate(rows, statuses, args)
    write_reports(summary, statuses, args)
    maybe_plot(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "POWHEG_PYTHIA_GEANT4_RESUMABLE_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
