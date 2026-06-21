#!/usr/bin/env python3
"""Run the clean final HADROS scientific pipeline from a compact JSON config."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RG_CM_PER_MSUN = 1.4766250385e5


@dataclass
class FinalStep:
    name: str
    command: list[str]
    required_outputs: list[Path]
    continue_on_geant4_partial: bool = False


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def effective_physics_mode(config: dict[str, Any]) -> str:
    raw = str(config.get("physics_mode", "uhe_particles_camera")).strip().lower()
    aliases = {
        "uhe_particles": "uhe_particles_camera",
        "uhe_particles_mev": "mev_torus",
        "uhe_dis": "uhe_dis_only",
    }
    mode = aliases.get(raw, raw)
    allowed = {"uhe_dis_only", "uhe_cascade", "uhe_particles_camera", "mev_torus"}
    if mode not in allowed:
        raise ValueError(f"Unsupported physics_mode={raw!r}; expected one of {sorted(allowed)}")
    return mode


def cm_per_rg_from_mbh_msun(mbh_msun: Any) -> float:
    value = float(mbh_msun)
    cm_per_rg = RG_CM_PER_MSUN * value
    if not (cm_per_rg > 10.0) or abs(cm_per_rg - 1.0) <= 1.0e-9:
        raise ValueError(f"Invalid cm_per_rg={cm_per_rg!r} for MBH_MSUN={value:g}")
    return cm_per_rg


def effective_association_mode(config: dict[str, Any]) -> str:
    raw = str(config.get("association_mode", "spatial_plus_direction")).strip().lower()
    allowed = {"spatial_only", "spatial_plus_direction", "full_transport"}
    if raw not in allowed:
        raise ValueError(f"Unsupported association_mode={raw!r}; expected one of {sorted(allowed)}")
    return raw


def effective_camera_naming_mode(config: dict[str, Any]) -> str:
    raw = str(config.get("camera_naming_mode", "both")).strip().lower()
    allowed = {"both", "semantic", "legacy"}
    if raw not in allowed:
        raise ValueError(f"Unsupported camera_naming_mode={raw!r}; expected one of {sorted(allowed)}")
    return raw


def photon_escape_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("photon_escape_classifier", {})
    if not isinstance(nested, dict):
        nested = {}
    required_keys = [
        "enable_photon_observer_camera",
        "photon_observer_mode",
        "photon_observer_frame",
        "photon_null_norm_tolerance",
        "photon_invariant_tolerance",
        "photon_horizon_crossing_tolerance_rg",
        "photon_fail_on_invariant_violation",
        "photon_max_geodesic_steps",
        "photon_geodesic_step_rg",
        "photon_min_energy_gev",
        "photon_camera_output_mode",
        "photon_redshift_mode",
        "photon_redshift_emitter_frame",
        "photon_redshift_observer_frame",
        "photon_redshift_energy_tolerance",
        "photon_redshift_fail_on_invalid",
        "photon_camera_projection_mode",
        "photon_camera_fov_deg",
        "photon_camera_fov_definition",
        "photon_camera_resolution_mode",
        "photon_camera_center_theta_source",
        "photon_camera_center_phi_rad",
        "photon_camera_clipping_mode",
    ]
    values = {}
    for key in required_keys:
        if key in nested:
            values[key] = nested[key]
        elif key in config:
            values[key] = config[key]
        else:
            raise ValueError(f"Missing required photon_escape_classifier parameter: {key}")
    if str(values["photon_observer_mode"]) not in {"escape_classifier", "observer_sphere_hits", "observer_camera_projection"}:
        raise ValueError("Unsupported photon_observer_mode; expected 'escape_classifier', 'observer_sphere_hits', or 'observer_camera_projection'")
    if str(values["photon_observer_frame"]) != "ZAMO":
        raise ValueError("Only photon_observer_frame='ZAMO' is implemented in Phase 1")
    if str(values["photon_redshift_mode"]) not in {"disabled", "validated_zamo"}:
        raise ValueError("Unsupported photon_redshift_mode; expected 'disabled' or 'validated_zamo'")
    if str(values["photon_redshift_emitter_frame"]) != "ZAMO":
        raise ValueError("Only photon_redshift_emitter_frame='ZAMO' is implemented")
    if str(values["photon_redshift_observer_frame"]) != "ZAMO":
        raise ValueError("Only photon_redshift_observer_frame='ZAMO' is implemented")
    if not math.isfinite(float(values["photon_redshift_energy_tolerance"])) or float(values["photon_redshift_energy_tolerance"]) <= 0.0:
        raise ValueError("photon_redshift_energy_tolerance must be > 0")
    as_bool(values["photon_redshift_fail_on_invalid"])
    if str(values["photon_camera_output_mode"]) not in {"summary_only", "arrivals"}:
        raise ValueError("Unsupported photon_camera_output_mode for Phase 1")
    if str(values["photon_camera_projection_mode"]) != "gnomonic_pinhole":
        raise ValueError("photon_camera_projection_mode must be gnomonic_pinhole")
    if str(values["photon_camera_fov_definition"]) != "square_half_angle":
        raise ValueError("photon_camera_fov_definition must be square_half_angle")
    if str(values["photon_camera_resolution_mode"]) != "reuse_main_camera":
        raise ValueError("photon_camera_resolution_mode must be reuse_main_camera")
    if str(values["photon_camera_center_theta_source"]) != "observer_inclination_deg":
        raise ValueError("photon_camera_center_theta_source must be observer_inclination_deg")
    if str(values["photon_camera_clipping_mode"]) != "keep_outside_fov":
        raise ValueError("photon_camera_clipping_mode must be keep_outside_fov")
    if not (0.0 < float(values["photon_camera_fov_deg"]) < 180.0):
        raise ValueError("photon_camera_fov_deg must satisfy 0 < value < 180")
    if not math.isfinite(float(values["photon_camera_center_phi_rad"])):
        raise ValueError("photon_camera_center_phi_rad must be finite")
    if float(values["photon_null_norm_tolerance"]) <= 0.0:
        raise ValueError("photon_null_norm_tolerance must be > 0")
    if float(values["photon_invariant_tolerance"]) <= 0.0:
        raise ValueError("photon_invariant_tolerance must be > 0")
    if float(values["photon_horizon_crossing_tolerance_rg"]) < 0.0:
        raise ValueError("photon_horizon_crossing_tolerance_rg must be >= 0")
    if float(values["photon_geodesic_step_rg"]) <= 0.0:
        raise ValueError("photon_geodesic_step_rg must be > 0")
    if int(values["photon_max_geodesic_steps"]) <= 0:
        raise ValueError("photon_max_geodesic_steps must be > 0")
    return values


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("run"), dict):
        data.setdefault("run_name", data["run"].get("run_name", "Run_Final_PaperFigures"))
        data.setdefault("output_dir", data["run"].get("output_dir", f"output/{data['run_name']}"))
        data.setdefault("physics_mode", data["run"].get("physics_mode", "uhe_particles"))
    if isinstance(data.get("black_hole"), dict):
        data.setdefault("black_hole_mass_msun", data["black_hole"].get("black_hole_mass_msun", 2.0))
        data.setdefault("spin", data["black_hole"].get("spin", 0.8))
    if isinstance(data.get("camera"), dict):
        camera = data["camera"]
        data.setdefault("camera_theta_deg", camera.get("observer_inclination_deg", camera.get("camera_theta_deg", 70.0)))
        data.setdefault("camera_fov_deg", camera.get("field_of_view_deg", camera.get("camera_fov_deg", 60.0)))
        data.setdefault("camera_nx", camera.get("resolution", camera.get("camera_nx", 8)))
        data.setdefault("camera_ny", camera.get("resolution", camera.get("camera_ny", data.get("camera_nx", 8))))
        data.setdefault("camera_r_obs_rg", camera.get("observer_radius_rg", camera.get("camera_r_obs_rg", 80.0)))
        data.setdefault("camera_r_max_rg", camera.get("ray_max_radius_rg", camera.get("camera_r_max_rg", 120.0)))
        data.setdefault("camera_step", camera.get("ray_step", camera.get("camera_step", 0.05)))
    if isinstance(data.get("particle_ray_association_camera"), dict):
        assoc = data["particle_ray_association_camera"]
        data.setdefault("association_mode", assoc.get("association_mode", "spatial_plus_direction"))
        data.setdefault("spatial_tolerance_rg", assoc.get("spatial_tolerance_rg", 1.0))
        data.setdefault("angular_tolerance_deg", assoc.get("angular_tolerance_deg", 1.0))
        data.setdefault("camera_naming_mode", assoc.get("camera_naming_mode", "both"))
    if isinstance(data.get("photon_escape_classifier"), dict):
        photon = data["photon_escape_classifier"]
        for key, value in photon.items():
            data.setdefault(key, value)
    if isinstance(data.get("uhe_dis"), dict):
        uhe = data["uhe_dis"]
        data.setdefault("source_model", uhe.get("source_model", "funnel_wall"))
        data.setdefault("neutrino_energy_gev", uhe.get("energy_gev", 1.0e9))
        data.setdefault("dis_model", uhe.get("dis_model", "both"))
        data.setdefault("spectral_bins", uhe.get("spectral_bins", 1))
        data.setdefault("sigma_table_path", uhe.get("sigma_table", "data/sigma/sigma_nuN_CC_GBW.dat"))
        data.setdefault("powheg_executable_path", uhe.get("powheg_executable", ""))
        data.setdefault("pythia8_config_path", uhe.get("pythia8_config", ""))
        data.setdefault("n_events", uhe.get("n_events", 10))
        data.setdefault("seed", uhe.get("seed", 12345))
    if isinstance(data.get("outputs"), dict):
        outputs = data["outputs"]
        data.setdefault("generate_standard_scientific_plots", outputs.get("science_plots", True))
        data.setdefault("generate_dashboard", outputs.get("dashboard", True))
        data.setdefault("validation_plots", outputs.get("validation_plots", True))
        data.setdefault("diagnostic_plots", outputs.get("diagnostic_plots", False))
    data.setdefault("run_name", "Run_Final_PaperFigures")
    data.setdefault("output_dir", f"output/{data['run_name']}")
    mode = effective_physics_mode(data)
    data["physics_mode_effective"] = mode
    data["association_mode_effective"] = effective_association_mode(data)
    data["camera_naming_mode_effective"] = effective_camera_naming_mode(data)
    data["photon_escape_classifier_effective"] = photon_escape_config(data)
    data["produce_uhe_collision_particles"] = mode in {"uhe_cascade", "uhe_particles_camera"}
    data["run_mev_torus_neutrinos"] = mode == "mev_torus"
    data.setdefault("generate_standard_scientific_plots", True)
    data.setdefault("generate_dashboard", True)
    return data


def config_for_interaction_scripts(config: dict[str, Any], output_dir: Path) -> Path:
    cm_per_rg = cm_per_rg_from_mbh_msun(config.get("black_hole_mass_msun", 2.0))
    values = {
        "black_hole": {
            "ASPIN": str(config.get("spin", 0.8)),
            "MBH_MSUN": str(config.get("black_hole_mass_msun", 2.0)),
        },
        "camera": {
            "CAM_FOV_DEG": str(config.get("camera_fov_deg", 60.0)),
            "CAM_NX": str(config.get("camera_nx", 8)),
            "CAM_NY": str(config.get("camera_ny", config.get("camera_nx", 8))),
            "CAM_R_OBS_RG": str(config.get("camera_r_obs_rg", 80.0)),
            "CAM_R_MAX_RG": str(config.get("camera_r_max_rg", 120.0)),
            "CAM_STEP": str(config.get("camera_step", 0.05)),
            "CAM_THETA_DEG": str(config.get("camera_theta_deg", 70.0)),
        },
        "particle_ray_association_camera": {
            "ASSOCIATION_MODE": str(effective_association_mode(config)),
            "SPATIAL_TOLERANCE_RG": str(config.get("spatial_tolerance_rg", 1.0)),
            "ANGULAR_TOLERANCE_DEG": str(config.get("angular_tolerance_deg", 1.0)),
            "CAMERA_NAMING_MODE": str(effective_camera_naming_mode(config)),
        },
        "density_profile": {"FUNNEL_THETA_DEG": "20.0"},
        "tabulated_funnel": {
            "TABULATED_FUNNEL_ENABLED": "1" if as_bool(config.get("torus", {}).get("funnel_enabled"), True) else "0",
            "TABULATED_FUNNEL_R_IN_RG": "1.5",
            "TABULATED_FUNNEL_R_OUT_RG": "120.0",
            "TABULATED_FUNNEL_RHO_AXIS": "1.0",
            "TABULATED_FUNNEL_RHO_WALL": "1.0e3",
            "TABULATED_FUNNEL_RHO_COCOON": "1.0e2",
            "TABULATED_FUNNEL_THETA_DEG": "15.0",
            "TABULATED_FUNNEL_DTHETA_DEG": "5.0",
            "TABULATED_FUNNEL_RADIAL_POWER": "2.0",
        },
        "tabulated_ambient": {
            "TABULATED_AMBIENT_RHO0": str(config.get("torus", {}).get("ambient_density", 1.0)),
            "TABULATED_AMBIENT_R0_RG": "10.0",
            "TABULATED_AMBIENT_POWERLAW_INDEX": "2.0",
        },
        "uhe_source": {
            "SOURCE_MODEL": str(config.get("source_model", "funnel_wall")),
            "SOURCE_R_RG": "3.5",
            "SOURCE_SIGMA_RG": "1.0",
            "SOURCE_FUNNEL_THETA_DEG": "20.0",
        },
    }
    provenance = {
        "physics_mode_effective": effective_physics_mode(config),
        "cm_per_rg": cm_per_rg,
        "cm_per_rg_formula": "G * MBH / c^2",
        "MBH_MSUN": float(config.get("black_hole_mass_msun", 2.0)),
        "association_mode_effective": effective_association_mode(config),
        "spatial_tolerance_rg": float(config.get("spatial_tolerance_rg", 1.0)),
        "angular_tolerance_deg": float(config.get("angular_tolerance_deg", 1.0)),
        "camera_naming_mode": effective_camera_naming_mode(config),
        "camera_physical_interpretation": "particle-ray association / cascade origin map",
        "camera_is_full_observational_transport": False,
        "camera_limitation": "secondary particles are associated with Kerr rays by spatial/angular criteria; they are not propagated to the distant observer",
        "full_transport_available": False,
    }
    photon = photon_escape_config(config)
    photon_mode = str(photon["photon_observer_mode"])
    provenance.update({
        "photon_escape_classifier_enabled_effective": as_bool(photon["enable_photon_observer_camera"]),
        "photon_observer_mode": photon["photon_observer_mode"],
        "photon_observer_frame": photon["photon_observer_frame"],
        "photon_null_norm_tolerance": float(photon["photon_null_norm_tolerance"]),
        "photon_invariant_tolerance": float(photon["photon_invariant_tolerance"]),
        "photon_horizon_crossing_tolerance_rg": float(photon["photon_horizon_crossing_tolerance_rg"]),
        "photon_fail_on_invariant_violation": as_bool(photon["photon_fail_on_invariant_violation"]),
        "photon_max_geodesic_steps": int(photon["photon_max_geodesic_steps"]),
        "photon_geodesic_step_rg": float(photon["photon_geodesic_step_rg"]),
        "photon_min_energy_gev": float(photon["photon_min_energy_gev"]),
        "photon_camera_output_mode": photon["photon_camera_output_mode"],
        "photon_redshift_mode": photon["photon_redshift_mode"],
        "photon_redshift_emitter_frame": photon["photon_redshift_emitter_frame"],
        "photon_redshift_observer_frame": photon["photon_redshift_observer_frame"],
        "photon_redshift_energy_tolerance": float(photon["photon_redshift_energy_tolerance"]),
        "photon_redshift_fail_on_invalid": as_bool(photon["photon_redshift_fail_on_invalid"]),
        "photon_camera_projection_mode": photon["photon_camera_projection_mode"],
        "photon_camera_fov_deg": float(photon["photon_camera_fov_deg"]),
        "photon_camera_fov_definition": photon["photon_camera_fov_definition"],
        "photon_camera_resolution_mode": photon["photon_camera_resolution_mode"],
        "photon_camera_center_theta_source": photon["photon_camera_center_theta_source"],
        "photon_camera_center_phi_rad": float(photon["photon_camera_center_phi_rad"]),
        "photon_camera_clipping_mode": photon["photon_camera_clipping_mode"],
        "photon_camera_physical_interpretation": "photon_escape_classifier",
        "photon_camera_is_full_observational_transport": False,
        "photon_projected_to_pixels": photon_mode == "observer_camera_projection",
        "photon_observer_sphere_crossing_is_detection": False,
        "photon_observer_sphere_hit_map_enabled_effective": (
            as_bool(photon["enable_photon_observer_camera"])
            and photon_mode in {"observer_sphere_hits", "observer_camera_projection"}
        ),
        "photon_observer_sphere_phase": (
            "photon_observer_sphere_hit_map"
            if photon_mode in {"observer_sphere_hits", "observer_camera_projection"}
            else "not_run"
        ),
        "photon_observer_sphere_projected_to_pixels": False,
        "photon_observer_sphere_hits_camera_aperture": False,
        "photon_observer_sphere_observed_energy_available": False,
        "photon_observer_camera_projection_enabled_effective": (
            as_bool(photon["enable_photon_observer_camera"])
            and photon_mode == "observer_camera_projection"
        ),
        "photon_observer_camera_phase": (
            "photon_observer_camera_projection"
            if photon_mode == "observer_camera_projection"
            else "not_run"
        ),
        "photon_observer_camera_observed_energy_available": False,
        "photon_observer_camera_redshift_phase": (
            "photon_observer_camera_redshift"
            if photon_mode == "observer_camera_projection" and str(photon["photon_redshift_mode"]) == "validated_zamo"
            else "not_run"
        ),
        "photon_observer_camera_detector_model_applied": False,
        "photon_observer_camera_instrument_response_applied": False,
        "photon_observer_camera_aperture_acceptance_applied": False,
    })
    path = output_dir / "final_pipeline_science_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config_web_values": values, "provenance": provenance, **config}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def find_executable(name: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        ROOT / "output" / "Run_Final_PaperFigures" / "cascade" / "powheg" / name,
        ROOT / "output" / "Run_E2E_PaperFigures" / "cascade" / "powheg" / name,
        Path("/tmp/hadros_powheg_box_res_build_copy/DIS") / name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return ""


def write_final_powheg_card(path: Path, config: dict[str, Any]) -> None:
    import math
    path.parent.mkdir(parents=True, exist_ok=True)
    n_events = int(config.get("n_events", 10) or 10)
    energy = float(config.get("neutrino_energy_gev", 1.0e9) or 1.0e9)
    energy_fortran = f"{energy:.10E}".replace("E", "D")
    seed = int(config.get("seed", 12345) or 12345)
    # CC channel (charged current: nu + q -> l + q') — the interaction
    # that produces a charged lepton and changes quark flavor, giving the
    # main observable hadronic + leptonic signal.
    # channel_type=3 → CC DIS; vtype=2 → CC virtual corrections.
    channel_type = int(config.get("powheg_channel_type", 3))
    vtype = int(config.get("powheg_vtype", 2))
    # PDF: NNPDF31_nlo_as_0118 (set 303400) — consistent with audit records.
    lhapdf_set = int(config.get("powheg_lhapdf_set", 303400))
    lhapdf_fortran = str(lhapdf_set)
    # Qmax: kinematic maximum is sqrt(2 * m_N * E_nu).  Cap at 1e5 GeV
    # (POWHEG-DIS upper limit) to avoid phase-space extrapolation errors.
    qmax = min(2.0 * math.sqrt(0.938272 * energy), 1.0e5)
    qmax_fortran = f"{qmax:.10E}".replace("E", "D")
    text = f"""! HADROS final-pipeline POWHEG DIS card.
! Generated by scripts/run_hadros_final_pipeline.py.
! channel_type={channel_type} (3=CC, 4=NC)  vtype={vtype}
! PDF: NNPDF31_nlo_as_0118 set {lhapdf_set}
! Qmax={qmax:.6g} GeV (kinematic limit for E_nu={energy:.6g} GeV)
LOevents 1
numevts {n_events}
ih1 12
ih2 1
ebeam1 {energy_fortran}
ebeam2 0.938272d0
bornktmin 0d0
bornsuppfact 0d0
Qmin 10d0
Qmax {qmax_fortran}
xmin 0d0
xmax 1d0
ymin 0d0
ymax 1d0
q2suppr 200d0
lhans1 {lhapdf_fortran}
lhans2 {lhapdf_fortran}
alphas_from_pdf 1
renscfact 1d0
facscfact 1d0
use-old-grid 0
use-old-ubound 0
ncall1 1000
itmx1 1
ncall2 2000
itmx2 1
foldcsi 1
foldy 1
foldphi 1
nubound 1000
iupperfsr 1
fastbtlbound 1
storemintupb 1
ubexcess_correct 1
storeinfo_rwgt 1
hdamp 0
bornzerodamp 1
withnegweights 1
flg_jacsing 1
testplots 0
xupbound 2d0
iseed {seed}
manyseeds 0
doublefsr 0
runningscales 1
olddij 0
channel_type {channel_type}
vtype {vtype}
smartsig 1
nores 1
parallelstage 0
xgriditeration 1
py8QED 0
py8MPI 1
py8had 2
py8shower 1
colltest 0
softtest 0
"""
    path.write_text(text, encoding="utf-8")


def build_steps(config: dict[str, Any], config_path: Path) -> list[FinalStep]:
    del config_path
    output_dir = Path(config["output_dir"])
    cascade = output_dir / "cascade"
    run_dir = output_dir
    mode = effective_physics_mode(config)
    if mode == "mev_torus":
        raise NotImplementedError("mev_torus not implemented in HADROS-CASCADE final pipeline")
    science_config = config_for_interaction_scripts(config, cascade)
    ray_cache = cascade / "rays" / "kerr_geodesics_e2e.bin"
    powheg_workdir = cascade / "powheg"
    n_events = str(int(config.get("n_events", 10) or 10))
    seed = str(int(config.get("seed", 12345) or 12345))
    dis_model = str(config.get("dis_model", "both")).lower()
    cm_per_rg = cm_per_rg_from_mbh_msun(config.get("black_hole_mass_msun", 2.0))
    association_mode = effective_association_mode(config)
    steps = [
        FinalStep(
            "uhe_kerr_rays",
            [
                "./compute_kerr_geodesics",
                str(config.get("spin", 0.8)),
                str(config.get("camera_r_obs_rg", 80.0)),
                str(config.get("camera_theta_deg", 70.0)),
                str(config.get("camera_fov_deg", 60.0)),
                str(int(config.get("camera_nx", 8) or 8)),
                str(int(config.get("camera_ny", config.get("camera_nx", 8)) or 8)),
                str(config.get("camera_r_max_rg", 120.0)),
                str(config.get("camera_step", 0.05)),
                str(ray_cache),
            ],
            [ray_cache],
        ),
        FinalStep(
            "interaction_points_on_geodesic_samples",
            [
                sys.executable,
                "scripts/science/sample_final_geodesic_interaction_points.py",
                "--geodesic-cache",
                str(ray_cache),
                "--output",
                str(cascade / "interaction_points.jsonl"),
                "--linked-output",
                str(cascade / "interaction_points_ray_linked.jsonl"),
                "--summary",
                str(cascade / "interaction_points_summary.md"),
                "--config",
                str(science_config),
                "--n-events",
                n_events,
                "--seed",
                seed,
                "--mbh-msun",
                str(config.get("black_hole_mass_msun", 2.0)),
                "--reference-energy-gev",
                str(config.get("neutrino_energy_gev", 1.0e9)),
            ],
            [cascade / "interaction_points.jsonl", cascade / "interaction_points_ray_linked.jsonl"],
        ),
    ]
    if mode == "uhe_dis_only":
        return steps

    write_final_powheg_card(powheg_workdir / "powheg.input", config)
    pwhg_main = find_executable("pwhg_main", str(config.get("powheg_executable_path", "")).strip())
    steps.extend([
        FinalStep(
            "powheg_pythia_event_records",
            [
                sys.executable,
                "scripts/science/run_powheg_pythia_event_records.py",
                "--output-dir",
                str(cascade),
                "--run-name",
                str(config.get("run_name", "Run_Final_PaperFigures")),
                "--n-events",
                n_events,
                "--interaction-points",
                str(cascade / "interaction_points_ray_linked.jsonl"),
                "--dis-mode",
                dis_model,
                "--mode",
                "both",
                "--seed",
                seed,
                "--powheg-workdir",
                str(powheg_workdir),
                *(["--pwhg-main", pwhg_main] if pwhg_main else []),
                *(["--pythia8-config", str(config.get("pythia8_config_path"))] if str(config.get("pythia8_config_path", "")).strip() else []),
            ],
            [cascade / "hadros_particle_events.jsonl", cascade / "powheg_pythia_particles.csv"],
        ),
        FinalStep(
            "geant4_real_safe_zamo",
            [
                sys.executable,
                "scripts/science/run_powheg_pythia_geant4_resumable.py",
                "--input",
                str(cascade / "hadros_particle_events.jsonl"),
                "--output-dir",
                str(cascade),
                "--interaction-points",
                str(cascade / "interaction_points.jsonl"),
                "--workers",
                str(int(config.get("geant4_batch_workers", 1) or 1)),
                "--geant4-app",
                "build/cascade_geant4_local_box",
                "--mbh-msun",
                str(config.get("black_hole_mass_msun", 2.0)),
                "--spin",
                str(config.get("spin", 0.8)),
                "--geant4-local-cm-per-rg",
                f"{cm_per_rg:.17g}",
            ],
            [cascade / "geant4_ready_particles.jsonl"],
            continue_on_geant4_partial=True,
        ),
    ])
    photon = photon_escape_config(config)
    if as_bool(photon["enable_photon_observer_camera"]):
        photon_mode = str(photon["photon_observer_mode"])
        fail_on_invariant = "true" if as_bool(photon["photon_fail_on_invariant_violation"]) else "false"
        steps.append(
            FinalStep(
                "photon_escape_classifier",
                [
                    sys.executable,
                    "scripts/science/run_kerr_photon_escape_classifier.py",
                    "--input",
                    str(cascade / "geant4_ready_particles.jsonl"),
                    "--output-jsonl",
                    str(cascade / "photon_escape_classifier.jsonl"),
                    "--summary-csv",
                    str(cascade / "photon_escape_summary.csv"),
                    "--summary-md",
                    str(cascade / "photon_escape_summary.md"),
                    "--provenance",
                    str(cascade / "photon_escape_provenance.json"),
                    "--backend",
                    "build/compute_kerr_photon_escape_classifier",
                    "--spin",
                    str(config.get("spin", 0.8)),
                    "--observer-radius-rg",
                    str(config.get("camera_r_obs_rg", 80.0)),
                    "--max-radius-rg",
                    str(config.get("camera_r_max_rg", 120.0)),
                    "--photon-geodesic-step-rg",
                    str(photon["photon_geodesic_step_rg"]),
                    "--photon-max-geodesic-steps",
                    str(int(photon["photon_max_geodesic_steps"])),
                    "--photon-null-norm-tolerance",
                    str(photon["photon_null_norm_tolerance"]),
                    "--photon-invariant-tolerance",
                    str(photon["photon_invariant_tolerance"]),
                    "--photon-horizon-crossing-tolerance-rg",
                    str(photon["photon_horizon_crossing_tolerance_rg"]),
                    "--photon-fail-on-invariant-violation",
                    fail_on_invariant,
                    "--photon-min-energy-gev",
                    str(photon["photon_min_energy_gev"]),
                    "--photon-observer-frame",
                    str(photon["photon_observer_frame"]),
                ],
                [
                    cascade / "photon_escape_classifier.jsonl",
                    cascade / "photon_escape_summary.csv",
                    cascade / "photon_escape_provenance.json",
                ],
            )
        )
        if photon_mode in {"observer_sphere_hits", "observer_camera_projection"}:
            steps.append(
                FinalStep(
                    "photon_observer_sphere_hit_map",
                    [
                        sys.executable,
                        "scripts/science/build_photon_observer_sphere_hits.py",
                        "--input",
                        str(cascade / "photon_escape_classifier.jsonl"),
                        "--output-jsonl",
                        str(cascade / "photon_observer_sphere_hits.jsonl"),
                        "--summary-csv",
                        str(cascade / "photon_observer_sphere_summary.csv"),
                        "--summary-md",
                        str(cascade / "photon_observer_sphere_summary.md"),
                        "--provenance",
                        str(cascade / "photon_observer_sphere_provenance.json"),
                    ],
                    [
                        cascade / "photon_observer_sphere_hits.jsonl",
                        cascade / "photon_observer_sphere_summary.csv",
                        cascade / "photon_observer_sphere_provenance.json",
                    ],
                )
            )
        if photon_mode == "observer_camera_projection":
            camera_nx = int(config.get("camera_nx", 0))
            camera_ny = int(config.get("camera_ny", 0))
            camera_theta_deg = float(config.get("camera_theta_deg", float("nan")))
            if camera_nx <= 0:
                raise ValueError("camera_nx must be > 0 for photon observer camera projection")
            if camera_ny <= 0:
                raise ValueError("camera_ny must be > 0 for photon observer camera projection")
            if not math.isfinite(camera_theta_deg):
                raise ValueError("camera_theta_deg must be finite for photon observer camera projection")
            if abs(math.sin(math.radians(camera_theta_deg))) < 1.0e-8:
                raise ValueError("photon camera optical center is too close to a spherical pole")
            steps.append(
                FinalStep(
                    "photon_observer_camera_projection",
                    [
                        sys.executable,
                        "scripts/science/build_photon_observer_camera_projection.py",
                        "--input",
                        str(cascade / "photon_observer_sphere_hits.jsonl"),
                        "--output-csv",
                        str(cascade / "photon_observer_camera.csv"),
                        "--summary-csv",
                        str(cascade / "photon_observer_camera_summary.csv"),
                        "--provenance",
                        str(cascade / "photon_observer_camera_provenance.json"),
                        "--camera-nx",
                        str(camera_nx),
                        "--camera-ny",
                        str(camera_ny),
                        "--photon-camera-fov-deg",
                        str(photon["photon_camera_fov_deg"]),
                        "--photon-camera-projection-mode",
                        str(photon["photon_camera_projection_mode"]),
                        "--photon-camera-fov-definition",
                        str(photon["photon_camera_fov_definition"]),
                        "--photon-camera-resolution-mode",
                        str(photon["photon_camera_resolution_mode"]),
                        "--photon-camera-center-theta-source",
                        str(photon["photon_camera_center_theta_source"]),
                        "--photon-camera-center-theta-deg",
                        str(camera_theta_deg),
                        "--photon-camera-center-phi-rad",
                        str(photon["photon_camera_center_phi_rad"]),
                        "--photon-camera-clipping-mode",
                        str(photon["photon_camera_clipping_mode"]),
                    ],
                    [
                        cascade / "photon_observer_camera.csv",
                        cascade / "photon_observer_camera_summary.csv",
                        cascade / "photon_observer_camera_provenance.json",
                    ],
                )
            )
            if str(photon["photon_redshift_mode"]) == "validated_zamo":
                if "spin" not in config:
                    raise ValueError("spin is required for photon observer camera redshift")
                fail_on_redshift_invalid = "true" if as_bool(photon["photon_redshift_fail_on_invalid"]) else "false"
                steps.append(
                    FinalStep(
                        "photon_observer_camera_redshift",
                        [
                            sys.executable,
                            "scripts/science/build_photon_observer_camera_redshift.py",
                            "--input",
                            str(cascade / "photon_observer_camera.csv"),
                            "--output-csv",
                            str(cascade / "photon_observer_camera_redshift.csv"),
                            "--summary-csv",
                            str(cascade / "photon_observer_camera_redshift_summary.csv"),
                            "--provenance",
                            str(cascade / "photon_observer_camera_redshift_provenance.json"),
                            "--spin",
                            str(config["spin"]),
                            "--photon-redshift-mode",
                            str(photon["photon_redshift_mode"]),
                            "--photon-redshift-emitter-frame",
                            str(photon["photon_redshift_emitter_frame"]),
                            "--photon-redshift-observer-frame",
                            str(photon["photon_redshift_observer_frame"]),
                            "--photon-redshift-energy-tolerance",
                            str(photon["photon_redshift_energy_tolerance"]),
                            "--photon-redshift-fail-on-invalid",
                            fail_on_redshift_invalid,
                            "--photon-invariant-tolerance",
                            str(photon["photon_invariant_tolerance"]),
                        ],
                        [
                            cascade / "photon_observer_camera_redshift.csv",
                            cascade / "photon_observer_camera_redshift_summary.csv",
                            cascade / "photon_observer_camera_redshift_provenance.json",
                        ],
                    )
                )
    if mode == "uhe_cascade":
        return steps
    if association_mode == "full_transport":
        raise NotImplementedError("full_transport is not implemented yet")
    camera_naming_mode = effective_camera_naming_mode(config)
    camera_csv = cascade / (
        "observed_particles_by_pixel.csv"
        if camera_naming_mode == "legacy"
        else "particle_ray_association_camera.csv"
    )
    camera_jsonl = cascade / (
        "observed_particles_by_pixel.jsonl"
        if camera_naming_mode == "legacy"
        else "particle_ray_association_camera.jsonl"
    )

    steps.extend([
        FinalStep(
            "real_kerr_particle_camera",
            [
                sys.executable,
                "scripts/science/run_real_kerr_particle_camera.py",
                "--input",
                str(cascade / "geant4_ready_particles.jsonl"),
                "--output-dir",
                str(cascade),
                "--camera-nx",
                str(int(config.get("camera_nx", 8) or 8)),
                "--camera-ny",
                str(int(config.get("camera_ny", config.get("camera_nx", 8)) or 8)),
                "--camera-fov-deg",
                str(config.get("camera_fov_deg", 60.0)),
                "--camera-theta-deg",
                str(config.get("camera_theta_deg", 70.0)),
                "--camera-r-obs-rg",
                str(config.get("camera_r_obs_rg", 80.0)),
                "--camera-r-max-rg",
                str(config.get("camera_r_max_rg", 120.0)),
                "--camera-step",
                str(config.get("camera_step", 0.05)),
                "--spatial-tolerance-rg",
                str(config.get("spatial_tolerance_rg", 1.0)),
                "--angular-tolerance-deg",
                str(config.get("angular_tolerance_deg", 1.0)),
                "--association-mode",
                association_mode,
                "--camera-naming-mode",
                camera_naming_mode,
                "--aspin",
                str(config.get("spin", 0.8)),
                "--skip-build",
            ],
            [camera_csv],
        ),
        FinalStep(
            "incoming_ray_event_link",
            [
                sys.executable,
                "scripts/science/build_uhe_ray_event_link.py",
                "--geodesic-cache",
                str(ray_cache),
                "--interaction-points",
                str(cascade / "interaction_points.jsonl"),
                "--particles",
                str(cascade / "hadros_particle_events.jsonl"),
                "--ready",
                str(cascade / "geant4_ready_particles.jsonl"),
                "--observed-csv",
                str(camera_csv),
                "--observed-jsonl",
                str(camera_jsonl),
                "--output-dir",
                str(cascade),
                "--config",
                str(science_config),
                "--reference-energy-gev",
                str(config.get("neutrino_energy_gev", 1.0e9)),
            ],
            [cascade / "interaction_points_ray_linked.jsonl"],
        ),
        FinalStep(
            "gbw_iim_incoming_weighting",
            [
                sys.executable,
                "scripts/science/build_gbw_iim_real_kerr_reweighting.py",
                "--particles",
                str(cascade / "hadros_particle_events.jsonl"),
                "--interaction-points",
                str(cascade / "interaction_points_ray_linked.jsonl"),
                "--ready",
                str(cascade / "geant4_ready_particles.jsonl"),
                "--observed",
                str(camera_csv),
                "--output-dir",
                str(cascade / "gbw_iim_reweighting"),
                "--camera-output-dir",
                str(cascade),
                "--config",
                str(science_config),
            ],
            [cascade / "gbw_iim_camera_summary.csv"],
        ),
    ])
    if as_bool(config.get("generate_standard_scientific_plots"), True):
        steps.append(
            FinalStep(
                "paper_ready_science_figures",
                [
                    sys.executable,
                    "scripts/science/build_scientific_plot_bundle.py",
                    "--run-dir",
                    str(run_dir),
                    "--dis-model",
                    "both",
                    "--production-plots",
                    "--geant4-plots",
                    "--particle-ray-association-plots",
                    "--gbw-iim-plots",
                ],
                [run_dir / "plots" / "science" / "01_particle_ray_association_rgb.png"],
            )
        )
    if as_bool(config.get("generate_dashboard"), True):
        steps.append(
            FinalStep(
                "dashboard",
                [sys.executable, "scripts/build_run_plot_dashboard.py", "--run-dir", str(run_dir)],
                [run_dir / "dashboard" / "index.html"],
            )
        )
    return steps


def command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def write_plan(path: Path, steps: list[FinalStep]) -> None:
    lines = ["# HADROS Final Pipeline Plan", "", "| order | step | command |", "|---:|---|---|"]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"| {idx} | `{step.name}` | `{command_text(step.command)}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def outputs_ready(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def prepare_output_tree(output_dir: Path) -> None:
    cascade = output_dir / "cascade"
    for path in [
        cascade,
        cascade / "rays",
        cascade / "powheg",
        cascade / "gbw_iim_reweighting",
        output_dir / "plots" / "science",
        output_dir / "dashboard",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_failure_report(
    path: Path,
    *,
    step: FinalStep,
    returncode: int,
    reason: str,
    stdout: str,
) -> None:
    expected = [
        {
            "path": str(item),
            "exists": item.exists(),
            "size_bytes": item.stat().st_size if item.exists() else 0,
        }
        for item in step.required_outputs
    ]
    tail = "\n".join(stdout.splitlines()[-80:])
    lines = [
        "# HADROS Final Pipeline Failure",
        "",
        f"- step: `{step.name}`",
        f"- returncode: `{returncode}`",
        f"- reason: `{reason}`",
        f"- command: `{command_text(step.command)}`",
        "",
        "## Required Outputs",
        "",
        "| path | exists | size_bytes |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| `{row['path']}` | {row['exists']} | {row['size_bytes']} |" for row in expected)
    lines.extend(["", "## Command Output Tail", "", "```text", tail, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_steps(steps: list[FinalStep], log_path: Path) -> int:
    lines: list[str] = []
    failure_path = log_path.parent / "final_pipeline_failure.md"
    if failure_path.exists():
        failure_path.unlink()
    for step in steps:
        command_line = f"$ {command_text(step.command)}"
        lines.append(command_line)
        print(command_line, flush=True)
        proc = subprocess.Popen(
            step.command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        stdout_parts: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_parts.append(line)
            print(line, end="", flush=True)
        returncode = proc.wait()
        stdout = "".join(stdout_parts)
        lines.append(stdout)
        if returncode != 0:
            if step.continue_on_geant4_partial and returncode == 2 and outputs_ready(step.required_outputs):
                lines.append(f"[warn] {step.name}: partial return code 2 with required outputs present")
                print(lines[-1], flush=True)
            else:
                lines.append(f"[fail] {step.name}: return code {returncode}")
                print(lines[-1], flush=True)
                log_path.write_text("\n".join(lines), encoding="utf-8")
                write_failure_report(
                    failure_path,
                    step=step,
                    returncode=returncode,
                    reason="command returned non-zero",
                    stdout=stdout,
                )
                return returncode
        if not outputs_ready(step.required_outputs):
            lines.append(f"[fail] {step.name}: missing required output")
            print(lines[-1], flush=True)
            log_path.write_text("\n".join(lines), encoding="utf-8")
            write_failure_report(
                failure_path,
                step=step,
                returncode=3,
                reason="missing required output",
                stdout=stdout,
            )
            return 3
    lines.append("[ok] final pipeline completed")
    print(lines[-1], flush=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    cascade = output_dir / "cascade"
    prepare_output_tree(output_dir)
    steps = build_steps(config, args.config)
    write_plan(cascade / "final_pipeline_plan.md", steps)
    (cascade / "final_pipeline_config_resolved.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"final_pipeline_plan={cascade / 'final_pipeline_plan.md'}")
        return 0
    return run_steps(steps, cascade / "final_pipeline.log")


if __name__ == "__main__":
    raise SystemExit(main())
