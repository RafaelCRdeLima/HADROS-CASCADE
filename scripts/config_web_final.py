#!/usr/bin/env python3
"""Clean final HADROS scientific config web.

This module exposes only the final scientific production chain and writes a compact JSON consumed by
scripts/run_hadros_final_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "presets" / "config_web" / "final_config.json"


def field(
    section: str,
    key: str,
    label: str,
    default: Any,
    *,
    kind: str = "text",
    options: list[str] | None = None,
    visibility: str = "NORMAL",
    description: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "section": section,
        "key": key,
        "label": label,
        "default": default,
        "kind": kind,
        "visibility": visibility,
        "description": description,
    }
    if options is not None:
        out["options"] = options
    return out


def schema() -> list[dict[str, Any]]:
    return [
        {
            "tab": "Run",
            "fields": [
                field("run", "run_name", "Run name", "Run_Final_PaperFigures"),
                field("run", "output_dir", "Output directory", "output/Run_Final_PaperFigures"),
                field(
                    "run",
                    "physics_mode",
                    "Physics mode",
                    "uhe_particles_camera",
                    kind="select",
                    options=["uhe_dis_only", "uhe_cascade", "uhe_particles_camera", "mev_torus"],
                ),
            ],
            "actions": ["run", "dry_run", "open_dashboard"],
        },
        {
            "tab": "Black Hole and Camera",
            "fields": [
                field("black_hole_camera", "black_hole_mass_msun", "Black hole mass (M☉)", 2.0, kind="number"),
                field("black_hole_camera", "spin", "Spin a", 0.8, kind="number"),
                field("black_hole_camera", "observer_inclination_deg", "Observer inclination θ (°)", 70.0, kind="number"),
                field("black_hole_camera", "field_of_view_deg", "Field of view (°)", 60.0, kind="number"),
                field("black_hole_camera", "resolution", "Resolution (px per side)", 8, kind="number"),
                field("black_hole_camera", "observer_radius_rg", "Observer radius (rg)", 80.0, kind="number", visibility="EXPERT"),
                field("black_hole_camera", "ray_max_radius_rg", "Ray max radius (rg)", 120.0, kind="number", visibility="EXPERT"),
                field("black_hole_camera", "ray_step", "Ray step (rg)", 0.05, kind="number", visibility="EXPERT"),
                field(
                    "particle_ray_association_camera",
                    "association_mode",
                    "Particle-ray association mode",
                    "spatial_plus_direction",
                    kind="select",
                    options=["spatial_only", "spatial_plus_direction", "full_transport"],
                    visibility="EXPERT",
                    description="Controls particle/ray matching: spatial_only uses position only; spatial_plus_direction also requires particle momentum alignment; full_transport is reserved and fails explicitly.",
                ),
                field(
                    "particle_ray_association_camera",
                    "spatial_tolerance_rg",
                    "Particle-ray spatial tolerance (rg)",
                    1.0,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum global position distance from a sampled Kerr ray for particle-ray association.",
                ),
                field(
                    "particle_ray_association_camera",
                    "angular_tolerance_deg",
                    "Particle-ray angular tolerance (deg)",
                    1.0,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum angle between secondary-particle momentum and the local Kerr ray direction when association_mode is spatial_plus_direction.",
                ),
                field(
                    "particle_ray_association_camera",
                    "camera_naming_mode",
                    "Particle-ray association output naming",
                    "both",
                    kind="select",
                    options=["both", "semantic", "legacy"],
                    visibility="EXPERT",
                    description="Controls camera output filenames. both writes semantic particle_ray_association_camera files plus legacy observed_particles_by_pixel compatibility files; semantic writes only semantic names; legacy writes only compatibility names.",
                ),
            ],
        },
        {
            "tab": "Photon Escape Classifier",
            "fields": [
                field(
                    "photon_escape_classifier",
                    "enable_photon_observer_camera",
                    "Enable photon escape classifier",
                    False,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="Runs Phase 1 photon-only escape classification after GEANT4. This does not create pixels, images, detector products, or observed-energy redshift.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_observer_mode",
                    "Photon observer mode",
                    "escape_classifier",
                    kind="select",
                    options=["escape_classifier", "observer_sphere_hits", "observer_camera_projection"],
                    visibility="EXPERT",
                    description="escape_classifier runs Phase 1 only. observer_sphere_hits runs Phase 1 plus Phase 2 crossing records. observer_camera_projection adds Phase 3 ideal pinhole pixel projection. No detector products, images, aperture acceptance, or observed-energy redshift are produced.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_observer_frame",
                    "Photon observer frame",
                    "ZAMO",
                    kind="select",
                    options=["ZAMO"],
                    visibility="EXPERT",
                    description="Observer local frame for Phase 1. Only ZAMO is implemented.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_null_norm_tolerance",
                    "Photon null norm tolerance",
                    1.0e-8,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum allowed initial null-momentum norm error g^{mu nu} p_mu p_nu.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_invariant_tolerance",
                    "Photon invariant tolerance",
                    1.0e-6,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum allowed drift in null norm, Killing energy, and Lz diagnostics.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_horizon_crossing_tolerance_rg",
                    "Photon horizon crossing tolerance (rg)",
                    1.0e-6,
                    kind="number",
                    visibility="EXPERT",
                    description="Numerical tolerance for classifying captured_by_black_hole only when a photon crosses from outside r_plus to r_plus plus this tolerance.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_observer_crossing_tolerance_rg",
                    "Photon observer crossing tolerance (rg)",
                    1.0e-8,
                    kind="number",
                    visibility="EXPERT",
                    description="Numerical tolerance for refining the observer-sphere crossing state with a fractional RK geodesic substep.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_fail_on_invariant_violation",
                    "Fail on invariant violation",
                    True,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="If enabled, photons with invariant drift above tolerance are classified as integration_failed_invariant_violation.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_max_geodesic_steps",
                    "Photon max geodesic steps",
                    200000,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum forward Kerr null-geodesic RK4 steps per photon.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_geodesic_step_rg",
                    "Photon geodesic step (rg)",
                    0.05,
                    kind="number",
                    visibility="EXPERT",
                    description="Forward null-geodesic RK4 step size in gravitational radii.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_min_energy_gev",
                    "Photon min energy (GeV)",
                    0.0,
                    kind="number",
                    visibility="EXPERT",
                    description="Minimum escaped photon input energy to classify.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_output_mode",
                    "Photon output mode",
                    "summary_only",
                    kind="select",
                    options=["summary_only", "arrivals"],
                    visibility="EXPERT",
                    description="Phase 1 output depth. No camera pixels or images are produced.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_redshift_mode",
                    "Photon redshift mode",
                    "disabled",
                    kind="select",
                    options=["disabled", "validated_zamo"],
                    visibility="EXPERT",
                    description="disabled stops at geometric camera outputs. validated_zamo runs Phase 4 and emits observed photon energy only after ZAMO redshift validation.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_redshift_emitter_frame",
                    "Photon redshift emitter frame",
                    "ZAMO",
                    kind="select",
                    options=["ZAMO"],
                    visibility="EXPERT",
                    description="Local frame used for emitted photon energy in Phase 4 redshift validation. Only ZAMO is implemented.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_redshift_observer_frame",
                    "Photon redshift observer frame",
                    "ZAMO",
                    kind="select",
                    options=["ZAMO"],
                    visibility="EXPERT",
                    description="Local observer frame used at the observer-sphere crossing in Phase 4. Only ZAMO is implemented.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_redshift_energy_tolerance",
                    "Photon redshift energy tolerance",
                    1.0e-6,
                    kind="number",
                    visibility="EXPERT",
                    description="Maximum relative mismatch between E_emit = -p_mu u_ZAMO^mu and input_energy_gev in validated_zamo mode.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_redshift_fail_on_invalid",
                    "Fail on invalid redshift",
                    True,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="If enabled, Phase 4 fails when any projected photon lacks validated ZAMO redshift.",
                ),
                field(
                    "photon_escape_classifier",
                    "enable_photon_validation_gate",
                    "Enable photon validation gate",
                    True,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="Runs the mandatory lightweight physics validation gate after validated_zamo photon redshift. Disable only for debug/smoke runs that intentionally skip scientific validation.",
                ),
                field(
                    "photon_escape_classifier",
                    "enable_photon_observer_science_products",
                    "Enable photon observer science products",
                    False,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="Builds ideal photon observer maps and spectra after validated_zamo redshift and a passing photon validation gate. No detector response, aperture, instrument response, or photon absorption is applied.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_observer_science_require_validation",
                    "Require validation for photon science products",
                    True,
                    kind="checkbox",
                    visibility="EXPERT",
                    description="Requires photon validation physics_status=PASS before writing photon observer science products. Keep enabled for scientific runs.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_projection_mode",
                    "Photon camera projection mode",
                    "gnomonic_pinhole",
                    kind="select",
                    options=["gnomonic_pinhole"],
                    visibility="EXPERT",
                    description="Phase 3 tangent-plane projection from observer-sphere crossings to ideal camera coordinates.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_fov_deg",
                    "Photon camera FOV (deg)",
                    60.0,
                    kind="number",
                    visibility="EXPERT",
                    description="Phase 3 projection field of view. Keep equal to the main camera field_of_view_deg unless intentionally overriding through config_web_final.py.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_fov_definition",
                    "Photon camera FOV definition",
                    "square_half_angle",
                    kind="select",
                    options=["square_half_angle"],
                    visibility="EXPERT",
                    description="Uses the same angular half extent in camera x and y, matching the existing Kerr camera convention.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_resolution_mode",
                    "Photon camera resolution mode",
                    "reuse_main_camera",
                    kind="select",
                    options=["reuse_main_camera"],
                    visibility="EXPERT",
                    description="Phase 3 reuses the main camera_nx/camera_ny resolution.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_center_theta_source",
                    "Photon camera theta source",
                    "observer_inclination_deg",
                    kind="select",
                    options=["observer_inclination_deg"],
                    visibility="EXPERT",
                    description="Phase 3 optical center theta follows the main observer_inclination_deg.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_center_phi_rad",
                    "Photon camera center phi (rad)",
                    0.0,
                    kind="number",
                    visibility="EXPERT",
                    description="Phase 3 optical center azimuth in Boyer-Lindquist coordinates.",
                ),
                field(
                    "photon_escape_classifier",
                    "photon_camera_clipping_mode",
                    "Photon camera clipping mode",
                    "keep_outside_fov",
                    kind="select",
                    options=["keep_outside_fov"],
                    visibility="EXPERT",
                    description="Preserves outside-FOV Phase 3 rows with null pixels for diagnostics.",
                ),
            ],
        },
        {
            "tab": "Photon Observer Diagnostics",
            "fields": [],
        },
        {
            "tab": "Torus Model",
            "fields": [
                field("torus", "torus_model", "Torus model", "uribe_ndaf", kind="select", options=["analytic_torus", "uribe_ndaf"]),
                field("torus", "density_scale", "Density scale", 1.0, kind="number"),
                field("torus", "funnel_enabled", "Funnel enabled", True, kind="checkbox"),
                field("torus", "vertical_closure", "Vertical closure", "gaussian", visibility="EXPERT"),
                field("torus", "ambient_density", "Ambient density", 1.0, kind="number", visibility="EXPERT"),
                field("torus", "rho_floor", "Rho floor", 0.0, kind="number", visibility="EXPERT"),
                field("torus", "interpolation", "Interpolation", "loglog", visibility="EXPERT"),
            ],
        },
        {
            "tab": "UHE DIS",
            "fields": [
                field("uhe_dis", "source_model", "Source model", "funnel_wall", kind="select", options=["funnel_wall", "axial_point"]),
                field("uhe_dis", "energy_gev", "Energy (GeV)", 1.0e9, kind="number"),
                field("uhe_dis", "dis_model", "DIS model", "both", kind="select", options=["gbw", "iim", "both"]),
                field("uhe_dis", "spectral_bins", "Spectral bins", 1, kind="number", visibility="EXPERT"),
                field("uhe_dis", "source_geometry", "Source geometry", "axial_point", visibility="EXPERT"),
                field("uhe_dis", "sigma_table", "Sigma table", "data/sigma/sigma_nuN_CC_GBW.dat", visibility="EXPERT"),
                field("uhe_dis", "powheg_executable", "POWHEG executable", "", visibility="EXPERT"),
                field("uhe_dis", "pythia8_config", "PYTHIA8 config", "", visibility="EXPERT"),
                field("uhe_dis", "n_events", "Events", 10, kind="number", visibility="EXPERT"),
                field("uhe_dis", "seed", "Seed", 12345, kind="number", visibility="EXPERT"),
            ],
        },
        {
            "tab": "Outputs",
            "fields": [
                field("outputs", "science_plots", "Generate standard science plots", True, kind="checkbox"),
                field("outputs", "dashboard", "Generate dashboard", True, kind="checkbox"),
                field("outputs", "validation_plots", "Validation plots", True, kind="checkbox", visibility="EXPERT"),
                field("outputs", "diagnostic_plots", "Diagnostic plots", False, kind="checkbox", visibility="EXPERT"),
            ],
        },
    ]


def defaults() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for tab in schema():
        for item in tab["fields"]:
            values.setdefault(item["section"], {})[item["key"]] = item["default"]
    return values


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for section, section_values in updates.items():
        if isinstance(section_values, dict):
            out.setdefault(section, {}).update(section_values)
        else:
            out[section] = section_values
    return out


def load_values(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return defaults()
    return deep_update(defaults(), json.loads(path.read_text(encoding="utf-8")))


def final_pipeline_config(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mode = str(values["run"]["physics_mode"])
    camera = values["black_hole_camera"]
    uhe = values["uhe_dis"]
    outputs = values["outputs"]
    association = values["particle_ray_association_camera"]
    photon_escape = values["photon_escape_classifier"]
    return {
        "run": dict(values["run"]),
        "black_hole": {
            "black_hole_mass_msun": camera["black_hole_mass_msun"],
            "spin": camera["spin"],
        },
        "camera": dict(camera),
        "uhe_dis": dict(uhe),
        "outputs": dict(outputs),
        "photon_escape_classifier": dict(photon_escape),
        "run_name": values["run"]["run_name"],
        "output_dir": values["run"]["output_dir"],
        "physics_mode": mode,
        "produce_uhe_collision_particles": mode in {"uhe_cascade", "uhe_particles_camera"},
        "run_mev_torus_neutrinos": mode == "mev_torus",
        "black_hole_mass_msun": camera["black_hole_mass_msun"],
        "spin": camera["spin"],
        "camera_theta_deg": camera["observer_inclination_deg"],
        "camera_fov_deg": camera["field_of_view_deg"],
        "camera_nx": camera["resolution"],
        "camera_ny": camera["resolution"],
        "camera_r_obs_rg": camera["observer_radius_rg"],
        "camera_r_max_rg": camera["ray_max_radius_rg"],
        "camera_step": camera["ray_step"],
        "particle_ray_association_camera": dict(association),
        "association_mode": association["association_mode"],
        "spatial_tolerance_rg": association["spatial_tolerance_rg"],
        "angular_tolerance_deg": association["angular_tolerance_deg"],
        "camera_naming_mode": association["camera_naming_mode"],
        "enable_photon_observer_camera": photon_escape["enable_photon_observer_camera"],
        "photon_observer_mode": photon_escape["photon_observer_mode"],
        "photon_observer_frame": photon_escape["photon_observer_frame"],
        "photon_null_norm_tolerance": photon_escape["photon_null_norm_tolerance"],
        "photon_invariant_tolerance": photon_escape["photon_invariant_tolerance"],
        "photon_horizon_crossing_tolerance_rg": photon_escape["photon_horizon_crossing_tolerance_rg"],
        "photon_observer_crossing_tolerance_rg": photon_escape["photon_observer_crossing_tolerance_rg"],
        "photon_fail_on_invariant_violation": photon_escape["photon_fail_on_invariant_violation"],
        "photon_max_geodesic_steps": photon_escape["photon_max_geodesic_steps"],
        "photon_geodesic_step_rg": photon_escape["photon_geodesic_step_rg"],
        "photon_min_energy_gev": photon_escape["photon_min_energy_gev"],
        "photon_camera_output_mode": photon_escape["photon_camera_output_mode"],
        "photon_redshift_mode": photon_escape["photon_redshift_mode"],
        "photon_redshift_emitter_frame": photon_escape["photon_redshift_emitter_frame"],
        "photon_redshift_observer_frame": photon_escape["photon_redshift_observer_frame"],
        "photon_redshift_energy_tolerance": photon_escape["photon_redshift_energy_tolerance"],
        "photon_redshift_fail_on_invalid": photon_escape["photon_redshift_fail_on_invalid"],
        "enable_photon_validation_gate": photon_escape["enable_photon_validation_gate"],
        "enable_photon_observer_science_products": photon_escape["enable_photon_observer_science_products"],
        "photon_observer_science_require_validation": photon_escape["photon_observer_science_require_validation"],
        "photon_camera_projection_mode": photon_escape["photon_camera_projection_mode"],
        "photon_camera_fov_deg": photon_escape["photon_camera_fov_deg"],
        "photon_camera_fov_definition": photon_escape["photon_camera_fov_definition"],
        "photon_camera_resolution_mode": photon_escape["photon_camera_resolution_mode"],
        "photon_camera_center_theta_source": photon_escape["photon_camera_center_theta_source"],
        "photon_camera_center_phi_rad": photon_escape["photon_camera_center_phi_rad"],
        "photon_camera_clipping_mode": photon_escape["photon_camera_clipping_mode"],
        "torus": values["torus"],
        "source_model": uhe["source_model"],
        "neutrino_energy_gev": uhe["energy_gev"],
        "dis_model": uhe["dis_model"],
        "spectral_bins": uhe["spectral_bins"],
        "sigma_table_path": uhe["sigma_table"],
        "powheg_executable_path": uhe["powheg_executable"],
        "pythia8_config_path": uhe["pythia8_config"],
        "n_events": uhe["n_events"],
        "seed": uhe["seed"],
        "generate_standard_scientific_plots": outputs["science_plots"],
        "generate_dashboard": outputs["dashboard"],
        "validation_plots": outputs["validation_plots"],
        "diagnostic_plots": outputs["diagnostic_plots"],
    }


def write_values(path: Path, values: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_html(values: dict[str, dict[str, Any]], config_path: Path) -> str:
    payload = json.dumps({"schema": schema(), "values": values, "config": str(config_path)})
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HADROS Final Config</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f4f5f2; color: #1e2124; }}
    header {{ padding: 14px 24px; background: #1a2e24; color: white; display: flex; align-items: center; gap: 16px; }}
    header h1 {{ margin: 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 0 20px 40px; }}

    /* Tab navigation */
    .tab-nav {{ display: flex; gap: 2px; padding: 0; margin: 0; border-bottom: 2px solid #c8cfc4; background: #eceee9; }}
    .tab-btn {{
      padding: 10px 18px; border: none; background: transparent; cursor: pointer;
      font-size: 0.88rem; font-weight: 500; color: #556; border-bottom: 3px solid transparent;
      margin-bottom: -2px; transition: color 0.15s;
    }}
    .tab-btn:hover {{ color: #1a2e24; background: #e0e3dc; }}
    .tab-btn.active {{ color: #1a2e24; border-bottom-color: #1a2e24; background: white; }}
    .tab-btn.preview-btn {{ color: #5a3a00; }}
    .tab-btn.preview-btn.active {{ border-bottom-color: #8c5800; color: #8c5800; }}

    /* Tab panels */
    .tab-panel {{ display: none; padding: 20px 0; }}
    .tab-panel.active {{ display: block; }}

    /* Fields */
    label {{ display: grid; grid-template-columns: 240px 1fr; gap: 10px; align-items: center; margin: 7px 0; }}
    label span {{ font-size: 0.88rem; color: #344; }}
    input[type=text], input[type=number], select {{
      padding: 6px 8px; border: 1px solid #b0b8a8; border-radius: 4px;
      font-size: 0.88rem; background: white; max-width: 200px;
    }}
    input[type=checkbox] {{ width: 16px; height: 16px; }}

    /* Buttons */
    .actions {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid #d0d4cc; }}
    button.action {{
      margin-right: 8px; padding: 8px 14px; border: 1px solid #1a2e24;
      background: #1a2e24; color: white; border-radius: 4px; cursor: pointer; font-size: 0.88rem;
    }}
    button.action:hover {{ background: #243d2e; }}
    button.action.running {{ animation: blink 0.9s ease-in-out infinite; background: #4a6e54; border-color: #4a6e54; }}
    button.action:disabled {{ opacity: 0.7; cursor: progress; }}
    @keyframes blink {{ 0%, 100% {{ filter: brightness(1); }} 50% {{ filter: brightness(1.6); }} }}
    pre {{ white-space: pre-wrap; background: #101612; color: #d8e8d0; padding: 14px; min-height: 100px; border-radius: 4px; margin-top: 14px; font-size: 0.82rem; }}

    /* Preview tab layout */
    .preview-settings {{ max-width: 520px; }}
    .prev-row {{ margin-bottom: 12px; display: grid; grid-template-columns: 180px 1fr; gap: 8px; align-items: center; }}
    .prev-row label {{ font-size: 0.88rem; color: #445; }}
    .prev-row select, .prev-row input[type=text] {{ padding: 5px 8px; border: 1px solid #b0b8a8; border-radius: 4px; font-size: 0.88rem; background: white; }}
    .launch-btn {{
      margin-top: 14px; padding: 9px 20px; border: 1px solid #8c5800;
      background: #8c5800; color: white; border-radius: 4px; cursor: pointer; font-size: 0.9rem;
    }}
    .launch-btn:hover {{ background: #a06800; }}
    .launch-btn.blinking {{ animation: blink 0.9s ease-in-out infinite; }}
    .load-camera-btn {{
      margin-top: 8px; margin-left: 10px; padding: 9px 16px; border: 1px solid #4a6e54;
      background: #4a6e54; color: white; border-radius: 4px; cursor: pointer; font-size: 0.9rem;
    }}
    .load-camera-btn:hover {{ background: #2a5e3a; }}
    .preview-info {{ margin-top: 16px; padding: 12px; background: #f0f4ee; border-radius: 4px; font-size: 0.82rem; color: #445; line-height: 1.6; }}
    .preview-log {{ white-space: pre-wrap; background: #101612; color: #d8e8d0; padding: 12px; min-height: 80px; border-radius: 4px; margin-top: 12px; font-size: 0.8rem; }}
  </style>
</head>
<body>
<header>
  <h1>HADROS — Final Scientific Run</h1>
</header>
<main>
  <nav class="tab-nav" id="tab-nav"></nav>
  <div id="panels"></div>
  <div class="actions" id="actions" style="display:none">
    <button class="action" onclick="saveDry()">Dry run / Save</button>
    <button class="action" id="run-button" onclick="runPipeline()">Run pipeline</button>
    <button class="action" onclick="window.open('/dashboard','_blank')">Open dashboard</button>
    <pre id="log"></pre>
  </div>
</main>
<script>
const state = {payload};

// ── Tab navigation ────────────────────────────────────────────
let activeTab = 0;
function showTab(i) {{
  activeTab = i;
  document.querySelectorAll('.tab-btn').forEach((b, j) => b.classList.toggle('active', i === j));
  document.querySelectorAll('.tab-panel').forEach((p, j) => p.classList.toggle('active', i === j));
  const isPreview = (i === state.schema.length);
  document.getElementById('actions').style.display = isPreview ? 'none' : '';
}}

// ── Input helper ──────────────────────────────────────────────
function inputFor(f, value) {{
  if (f.kind === 'select') {{
    return '<select data-section="' + f.section + '" data-key="' + f.key + '">' +
      f.options.map(o => '<option value="' + o + '"' + (String(value) === String(o) ? ' selected' : '') + '>' + o + '</option>').join('') +
      '</select>';
  }}
  if (f.kind === 'checkbox') {{
    return '<input type="checkbox" data-section="' + f.section + '" data-key="' + f.key + '"' + (value ? ' checked' : '') + '>';
  }}
  return '<input type="' + (f.kind === 'number' ? 'number' : 'text') + '" data-section="' + f.section + '" data-key="' + f.key + '" value="' + value + '">';
}}

// ── Collect form values ───────────────────────────────────────
function collect() {{
  const values = JSON.parse(JSON.stringify(state.values));
  document.querySelectorAll('[data-section]').forEach(el => {{
    const s = el.dataset.section, k = el.dataset.key;
    if (s && k) values[s][k] = el.type === 'checkbox' ? el.checked : el.value;
  }});
  return values;
}}

// ── API calls ─────────────────────────────────────────────────
async function saveDry() {{
  const res = await fetch('/api/save', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(collect())}});
  document.querySelector('#log').textContent = await res.text();
}}

async function runPipeline() {{
  const button = document.querySelector('#run-button');
  const log = document.querySelector('#log');
  button.classList.add('running');
  button.disabled = true;
  log.textContent = 'Starting final pipeline...\\n';
  try {{
    const res = await fetch('/api/run', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(collect())}});
    if (!res.body) {{ log.textContent += await res.text(); return; }}
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {{
      const chunk = await reader.read();
      if (chunk.done) break;
      log.textContent += decoder.decode(chunk.value, {{stream: true}});
      log.scrollTop = log.scrollHeight;
    }}
  }} catch (err) {{
    log.textContent += '\\nUI stream error: ' + err + '\\n';
  }} finally {{
    button.disabled = false;
    button.classList.remove('running');
  }}
}}

// ── Preview: real Kerr geodesic camera preview ───────────────
let previewPollTimer = null;
let lastCameraMtime = 0;

function previewLog(msg) {{
  const el = document.getElementById('preview-log');
  if (el) {{ el.textContent = msg; el.scrollTop = el.scrollHeight; }}
}}

async function applyLastCamera() {{
  previewLog('Loading last saved camera config...');
  try {{
    const res = await fetch('/api/apply-last-camera', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(collect()),
    }});
    const data = await res.json();
    if (!data.exists) {{
      previewLog('No configs/cameras/last_camera.json found yet.\\nOpen Camera Preview, adjust the view, and press Q or S to save.');
      return;
    }}
    // Apply mapped values to the Black Hole and Camera form
    const mapped = data.mapped || {{}};
    if (mapped.black_hole_camera) {{
      Object.entries(mapped.black_hole_camera).forEach(function([k, v]) {{
        document.querySelectorAll('[data-section="black_hole_camera"]').forEach(function(el) {{
          if (el.dataset.key === k) el.value = v;
        }});
        state.values.black_hole_camera[k] = v;
      }});
    }}
    previewLog('Camera applied: ' + JSON.stringify(data.camera, null, 2));
    // Switch to BH tab
    const bhIdx = state.schema.findIndex(function(t) {{ return t.tab === 'Black Hole and Camera'; }});
    if (bhIdx >= 0) showTab(bhIdx);
    fetch('/api/save', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(collect())}});
  }} catch (err) {{
    previewLog('Error loading camera: ' + err);
  }}
}}

function startPreviewPolling(baseMtime) {{
  if (previewPollTimer) return;
  previewPollTimer = setInterval(async function() {{
    try {{
      const res = await fetch('/api/last-camera');
      const data = await res.json();
      if (data.exists && data.mtime > baseMtime + 0.5) {{
        clearInterval(previewPollTimer);
        previewPollTimer = null;
        previewLog('Camera saved by preview window — loading...');
        await applyLastCamera();
      }}
    }} catch (_) {{}}
  }}, 1200);
  setTimeout(function() {{
    if (previewPollTimer) {{ clearInterval(previewPollTimer); previewPollTimer = null; }}
  }}, 10 * 60 * 1000);
}}

async function launchCameraPreview() {{
  const btn = document.getElementById('launch-preview-btn');
  if (btn) {{ btn.classList.add('blinking'); setTimeout(function() {{ btn.classList.remove('blinking'); }}, 3000); }}
  const mode    = document.getElementById('prev-mode')    ? document.getElementById('prev-mode').value    : 'celestial_sphere';
  const quality = document.getElementById('prev-quality') ? document.getElementById('prev-quality').value : 'medium';
  const nx      = document.getElementById('prev-nx')      ? document.getElementById('prev-nx').value      : '256';
  const ny      = document.getElementById('prev-ny')      ? document.getElementById('prev-ny').value      : '144';
  const sky     = document.getElementById('prev-sky')     ? document.getElementById('prev-sky').value     : '';
  previewLog('Launching geodesic camera preview (' + nx + 'x' + ny + ', ' + quality + ', ' + mode + ')...\\n\\nIf no native window appears, run in a terminal:\\nmake geodesic_preview PREVIEW_NAV_MODE=' + mode + ' PREVIEW_QUALITY=' + quality + ' PREVIEW_NX=' + nx + ' PREVIEW_NY=' + ny + '\\n\\nAdjust the view and press Q or S to save the camera config.');
  try {{
    const current = await fetch('/api/last-camera');
    const cur = await current.json();
    lastCameraMtime = cur.mtime || 0;
  }} catch (_) {{ lastCameraMtime = 0; }}
  startPreviewPolling(lastCameraMtime);
  try {{
    const res = await fetch('/api/run-camera-preview', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ values: collect(), previewMode: mode, previewQuality: quality, previewNx: nx, previewNy: ny, previewSky: sky }}),
    }});
    const data = await res.json();
    previewLog((previewLog.textContent || '') + '\\n' + (data.output || ''));
  }} catch (err) {{
    previewLog('Launch error: ' + err);
  }}
}}

// ── Photon observer diagnostics ──────────────────────────────
function photonDiagLog(msg) {{
  const el = document.getElementById('photon-diagnostics-log');
  if (el) {{ el.textContent = msg; el.scrollTop = el.scrollHeight; }}
}}

async function generatePhotonDiagnostics() {{
  photonDiagLog('Generating photon observer diagnostic plots...\\nDiagnostic only: ideal photon observer camera, no detector response.\\n');
  try {{
    const res = await fetch('/api/generate-photon-diagnostics', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(collect()),
    }});
    photonDiagLog(await res.text());
  }} catch (err) {{
    photonDiagLog('Photon diagnostic generation failed: ' + err);
  }}
}}

function openPhotonDiagnosticFolder() {{
  window.open('/photon-diagnostics/folder', '_blank');
}}

function openPhotonDiagnosticFile(name) {{
  window.open('/photon-diagnostics/file/' + encodeURIComponent(name), '_blank');
}}

// ── Render all tabs ───────────────────────────────────────────
function render() {{
  const nav = document.getElementById('tab-nav');
  const panels = document.getElementById('panels');

  // Tab buttons
  nav.innerHTML = state.schema.map(function(tab, i) {{
    return '<button class="tab-btn' + (i === 0 ? ' active' : '') + '" onclick="showTab(' + i + ')">' + tab.tab + '</button>';
  }}).join('') + '<button class="tab-btn preview-btn" onclick="showTab(' + state.schema.length + ')">Preview</button>';

  // Regular tab panels
  let html = state.schema.map(function(tab, i) {{
    const active = i === 0 ? ' active' : '';
    if (tab.tab === 'Photon Observer Diagnostics') {{
      return '<div class="tab-panel' + active + '" data-tab="' + i + '">' +
        '<h2 style="margin:0 0 4px;font-size:1rem;color:#1a2e24">Photon Observer Diagnostics</h2>' +
        '<p style="color:#556;font-size:0.86rem;line-height:1.55;max-width:820px">' +
          '<strong>Diagnostic only.</strong> ideal photon observer camera, no detector response. ' +
          'These buttons use the current official run configuration and require <code>photon_observer_camera_redshift.csv</code>. ' +
          'They do not run GEANT4, POWHEG/PYTHIA, the full pipeline, dashboards, or paper-ready figures. ' +
          'They are separate from <code>particle_ray_association_camera</code>.' +
        '</p>' +
        '<div class="actions" style="margin-top:12px">' +
          '<button class="action" onclick="generatePhotonDiagnostics()">Generate photon diagnostic plots</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFolder()">Open photon diagnostic output folder</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_counts_map.png\\')">Open photon_diagnostic_counts_map.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_input_energy_map.png\\')">Open photon_diagnostic_input_energy_map.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_observed_energy_map.png\\')">Open photon_diagnostic_observed_energy_map.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_mean_redshift_map.png\\')">Open photon_diagnostic_mean_redshift_map.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_input_vs_observed_energy.png\\')">Open photon_diagnostic_input_vs_observed_energy.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_redshift_histogram.png\\')">Open photon_diagnostic_redshift_histogram.png</button>' +
          '<button class="action" onclick="openPhotonDiagnosticFile(\\'photon_diagnostic_morphology_summary.md\\')">Open photon_diagnostic_morphology_summary.md</button>' +
        '</div>' +
        '<pre id="photon-diagnostics-log">Photon observer diagnostic output will appear here.</pre>' +
      '</div>';
    }}
    return '<div class="tab-panel' + active + '" data-tab="' + i + '"><h2 style="margin:0 0 14px;font-size:1rem;color:#1a2e24">' + tab.tab + '</h2>' +
      tab.fields.map(function(f) {{
        return '<label><span>' + f.label + (f.visibility === 'EXPERT' ? ' <em style="color:#aaa;font-size:0.78rem">(expert)</em>' : '') + '</span>' + inputFor(f, state.values[f.section][f.key]) + '</label>';
      }}).join('') +
      '</div>';
  }}).join('');

  // Preview panel
  html += '<div class="tab-panel" data-tab="' + state.schema.length + '">' +
    '<h2 style="margin:0 0 4px;font-size:1rem;color:#5a3a00">Black Hole Geodesic Camera Preview</h2>' +
    '<p style="color:#668;font-size:0.82rem;margin:0 0 18px">Launches the real Kerr null-geodesic renderer. If GLFW is available, a native window opens — adjust the camera interactively and press Q or S to save. Then click <strong>Load Saved Camera</strong> to apply to the pipeline.</p>' +
    '<div class="preview-settings">' +
      '<div class="prev-row"><label>Resolution</label><select id="prev-nx"><option value="128">128×72 (fast)</option><option value="256" selected>256×144</option><option value="512">512×288</option><option value="1024">1024×576</option></select></div>' +
      '<div class="prev-row"><label>Display mode</label><select id="prev-mode"><option value="celestial_sphere" selected>Celestial sphere</option><option value="disk_intersection">Disk intersection</option><option value="shadow_only">Shadow only</option><option value="combined">Combined</option></select></div>' +
      '<div class="prev-row"><label>Quality</label><select id="prev-quality"><option value="fast">Fast</option><option value="medium" selected>Medium</option><option value="high">High</option></select></div>' +
      '<div class="prev-row"><label>Sky texture (optional path)</label><input type="text" id="prev-sky" placeholder="assets/sky/eso0932a.ppm" style="max-width:300px"></div>' +
    '</div>' +
    '<div style="margin-top:16px">' +
      '<button class="launch-btn" id="launch-preview-btn" onclick="launchCameraPreview()">Launch Camera Preview</button>' +
      '<button class="load-camera-btn" onclick="applyLastCamera()">Load Saved Camera</button>' +
    '</div>' +
    '<div class="preview-info">' +
      'Current camera: a=' + state.values.black_hole_camera.spin + '  θ=' + state.values.black_hole_camera.observer_inclination_deg + '°  FoV=' + state.values.black_hole_camera.field_of_view_deg + '°  r_obs=' + state.values.black_hole_camera.observer_radius_rg + ' r_g<br>' +
      'Adjust in the native window, press <strong>Q</strong> or <strong>S</strong> to save camera config, then click <strong>Load Saved Camera</strong>.' +
    '</div>' +
    '<pre class="preview-log" id="preview-log">Preview output will appear here after launch.</pre>' +
  '</div>';

  panels.innerHTML = html;
  document.getElementById('actions').style.display = '';
}}

render();
</script>
</body>
</html>
"""


def read_last_camera() -> dict[str, Any]:
    path = ROOT / "configs" / "cameras" / "last_camera.json"
    if not path.exists():
        return {"exists": False, "path": str(path), "mtime": 0.0, "camera": {}, "mapped": {}}
    camera = json.loads(path.read_text(encoding="utf-8"))
    mapped: dict[str, dict[str, str]] = {}

    def put(key: str, value: Any) -> None:
        if value is None:
            return
        mapped.setdefault("black_hole_camera", {})[key] = str(value)

    put("spin", camera.get("spin"))
    put("observer_radius_rg", camera.get("observer_distance_rg"))
    put("observer_inclination_deg", camera.get("inclination_deg"))
    put("field_of_view_deg", camera.get("fov_deg"))
    put("ray_max_radius_rg", camera.get("r_max_rg"))
    put("ray_step", camera.get("integration_step"))
    return {"exists": True, "path": str(path), "mtime": path.stat().st_mtime, "camera": camera, "mapped": mapped}


def apply_last_camera_to_values(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = read_last_camera()
    if not payload["exists"]:
        return payload
    for section, section_values in payload["mapped"].items():
        values.setdefault(section, {}).update(section_values)
    return payload


def photon_diagnostic_paths(values: dict[str, dict[str, Any]]) -> dict[str, Path]:
    run_dir = Path(final_pipeline_config(values)["output_dir"])
    cascade = run_dir / "cascade"
    return {
        "cascade": cascade,
        "redshift_csv": cascade / "photon_observer_camera_redshift.csv",
        "diagnostics": cascade / "photon_observer_diagnostics",
    }


def photon_diagnostic_filenames() -> list[str]:
    return [
        "photon_diagnostic_counts_map.png",
        "photon_diagnostic_input_energy_map.png",
        "photon_diagnostic_observed_energy_map.png",
        "photon_diagnostic_mean_redshift_map.png",
        "photon_diagnostic_valid_photon_density_map.png",
        "photon_diagnostic_mean_observed_energy_map.png",
        "photon_diagnostic_input_vs_observed_energy.png",
        "photon_diagnostic_redshift_histogram.png",
        "photon_diagnostic_morphology_summary.md",
        "photon_diagnostic_summary.md",
    ]


def generate_photon_diagnostics(values: dict[str, dict[str, Any]]) -> tuple[int, str]:
    paths = photon_diagnostic_paths(values)
    redshift_csv = paths["redshift_csv"]
    diagnostics = paths["diagnostics"]
    if not redshift_csv.exists():
        return (
            2,
            "Missing photon_observer_camera_redshift.csv.\n"
            f"Expected: {redshift_csv}\n"
            "Run photon_observer_mode=observer_camera_projection with "
            "photon_redshift_mode=validated_zamo before generating diagnostics.\n",
        )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "science" / "build_photon_observer_diagnostic_plots.py"),
        "--input",
        str(redshift_csv),
        "--output-dir",
        str(diagnostics),
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, "$ " + " ".join(command) + "\n" + proc.stdout + f"\nreturncode={proc.returncode}\n"


def render_photon_diagnostic_folder(values: dict[str, dict[str, Any]]) -> str:
    paths = photon_diagnostic_paths(values)
    diagnostics = paths["diagnostics"]
    if not paths["redshift_csv"].exists():
        return (
            "<!doctype html><meta charset=\"utf-8\">"
            "<title>Photon Observer Diagnostics Missing Input</title>"
            "<body style=\"font-family:system-ui,sans-serif;max-width:900px;margin:32px auto\">"
            "<h1>Photon Observer Diagnostics</h1>"
            "<p><strong>Diagnostic only.</strong> ideal photon observer camera, no detector response.</p>"
            "<p style=\"color:#8a3b00\"><strong>Missing photon_observer_camera_redshift.csv.</strong></p>"
            f"<p>Expected: <code>{paths['redshift_csv']}</code></p>"
            "<p>Run photon_observer_mode=observer_camera_projection with "
            "photon_redshift_mode=validated_zamo before opening diagnostics.</p>"
            "</body>"
        )
    rows = []
    for name in photon_diagnostic_filenames():
        path = diagnostics / name
        status = "present" if path.exists() and path.stat().st_size > 0 else "missing"
        rows.append(
            f"<li><a href=\"/photon-diagnostics/file/{name}\">{name}</a> "
            f"<span style=\"color:#667\">{status}</span></li>"
        )
    return (
        "<!doctype html><meta charset=\"utf-8\">"
        "<title>Photon Observer Diagnostics</title>"
        "<body style=\"font-family:system-ui,sans-serif;max-width:900px;margin:32px auto\">"
        "<h1>Photon Observer Diagnostics</h1>"
        "<p><strong>Diagnostic only.</strong> ideal photon observer camera, no detector response.</p>"
        f"<p>Folder: <code>{diagnostics}</code></p>"
        f"<p>Input CSV: <code>{paths['redshift_csv']}</code></p>"
        "<ul>" + "\n".join(rows) + "</ul>"
        "</body>"
    )


def photon_diagnostic_file(values: dict[str, dict[str, Any]], filename: str) -> Path | None:
    if not photon_diagnostic_paths(values)["redshift_csv"].exists():
        return None
    safe_name = Path(filename).name
    if safe_name not in photon_diagnostic_filenames():
        return None
    path = photon_diagnostic_paths(values)["diagnostics"] / safe_name
    if not path.exists() or not path.is_file():
        return None
    return path


def launch_camera_preview(
    values: dict[str, dict[str, Any]],
    preview_mode: str = "celestial_sphere",
    preview_quality: str = "medium",
    preview_nx: str = "256",
    preview_ny: str = "144",
    preview_sky: str = "",
) -> tuple[int, str]:
    cam = values.get("black_hole_camera", {})
    spin = str(cam.get("spin", "0.8")).strip() or "0.8"
    theta = str(cam.get("observer_inclination_deg", "70.0")).strip() or "70.0"
    fov = str(cam.get("field_of_view_deg", "60.0")).strip() or "60.0"
    r_obs = str(cam.get("observer_radius_rg", "80.0")).strip() or "80.0"
    r_max = str(cam.get("ray_max_radius_rg", "120.0")).strip() or "120.0"

    allowed_modes = {"celestial_sphere", "disk_intersection", "shadow_only", "combined"}
    if preview_mode not in allowed_modes:
        preview_mode = "celestial_sphere"
    if preview_quality not in {"fast", "medium", "high"}:
        preview_quality = "medium"

    try:
        nx = str(max(32, int(preview_nx)))
        ny = str(max(18, int(preview_nx) * 9 // 16)) if preview_ny == preview_nx else str(max(18, int(preview_ny)))
    except (ValueError, ZeroDivisionError):
        nx, ny = "256", "144"

    preview_output_dir = ROOT / "output" / "camera_preview"
    preview_output_dir.mkdir(parents=True, exist_ok=True)

    make_args = [
        "make", "geodesic_preview",
        f"ASPIN={spin}",
        f"CAM_THETA_DEG={theta}",
        f"CAM_FOV_DEG={fov}",
        f"CAM_R_OBS_RG={r_obs}",
        f"PREVIEW_R_MAX_RG={r_max}",
        f"PREVIEW_NAV_MODE={preview_mode}",
        f"PREVIEW_QUALITY={preview_quality}",
        f"PREVIEW_NX={nx}",
        f"PREVIEW_NY={ny}",
        f"PREVIEW_OUTPUT_DIR={preview_output_dir.as_posix()}",
    ]
    if preview_sky.strip():
        make_args.append(f"SKY_TEXTURE={preview_sky.strip()}")

    log_path = preview_output_dir / "camera_preview.log"
    env = {**__import__("os").environ, "HADROS_PREVIEW_OUTPUT_DIR": preview_output_dir.as_posix()}
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(make_args) + "\n")
        subprocess.Popen(make_args, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True, cwd=ROOT, env=env)

    output = (
        "$ " + " ".join(make_args) + "\n\n"
        "Geodesic camera preview launched as a background process.\n"
        "If GLFW/OpenGL is available, a native Kerr geodesic window will open.\n"
        "If no window appears, the headless renderer saves a PPM to:\n"
        f"  {preview_output_dir}/geodesic_preview.ppm\n\n"
        f"Log: {log_path}\n"
        f"Camera: a={spin}, θ={theta}°, FoV={fov}°, r_obs={r_obs} r_g\n"
        f"Resolution: {nx}×{ny}, quality: {preview_quality}, mode: {preview_mode}\n\n"
        "Adjust the camera in the native window and press Q or S to save.\n"
        "Then click 'Load Saved Camera' to apply it to the pipeline config."
    )
    return 0, output


class Handler(BaseHTTPRequestHandler):
    config_path = DEFAULT_CONFIG

    def log_message(self, fmt: str, *args: Any) -> None:  # suppress server logs
        pass

    def _send(self, code: int, text: str, content_type: str = "text/plain") -> None:
        payload = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, code: int, payload: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_run_stream(self, command: list[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        def write(text: str) -> None:
            self.wfile.write(text.encode("utf-8"))
            self.wfile.flush()

        write("$ " + " ".join(command) + "\n")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                write(line)
            returncode = proc.wait()
            write(f"\nreturncode={returncode}\n")
        except BrokenPipeError:
            proc.terminate()

    def _send_json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, indent=2), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        values = load_values(self.config_path)
        if self.path == "/":
            self._send(200, render_html(values, self.config_path), "text/html")
        elif self.path == "/api/state":
            self._send(200, json.dumps({"schema": schema(), "values": values}, indent=2), "application/json")
        elif self.path == "/api/last-camera":
            self._send_json(200, read_last_camera())
        elif self.path == "/dashboard":
            dashboard = Path(final_pipeline_config(values)["output_dir"]) / "dashboard" / "index.html"
            self._send(200, dashboard.read_text(encoding="utf-8") if dashboard.exists() else "Dashboard not generated yet.", "text/html")
        elif self.path == "/photon-diagnostics/folder":
            self._send(200, render_photon_diagnostic_folder(values), "text/html")
        elif self.path.startswith("/photon-diagnostics/file/"):
            filename = unquote(self.path.removeprefix("/photon-diagnostics/file/"))
            path = photon_diagnostic_file(values, filename)
            if path is None:
                self._send(404, "Photon diagnostic file not found. Generate photon diagnostic plots first.")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".md":
                content_type = "text/plain"
            self._send_bytes(200, path.read_bytes(), content_type)
        else:
            self._send(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/run-camera-preview":
            values = deep_update(defaults(), body.get("values", body))
            _code, output = launch_camera_preview(
                values,
                preview_mode=str(body.get("previewMode", "celestial_sphere")),
                preview_quality=str(body.get("previewQuality", "medium")),
                preview_nx=str(body.get("previewNx", "256")),
                preview_ny=str(body.get("previewNy", "144")),
                preview_sky=str(body.get("previewSky", "")),
            )
            self._send_json(200, {"returncode": _code, "output": output})
            return

        if self.path == "/api/apply-last-camera":
            values = deep_update(defaults(), body.get("values", body))
            payload = apply_last_camera_to_values(values)
            if payload["exists"]:
                write_values(self.config_path, values)
            self._send_json(200, payload)
            return

        if self.path == "/api/generate-photon-diagnostics":
            values = deep_update(defaults(), body)
            write_values(self.config_path, values)
            pipeline_config = self.config_path.with_name("final_pipeline_config.json")
            pipeline_config.write_text(json.dumps(final_pipeline_config(values), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            code, output = generate_photon_diagnostics(values)
            self._send(200 if code == 0 else 400, output)
            return

        values = deep_update(defaults(), body)
        write_values(self.config_path, values)
        pipeline_config = self.config_path.with_name("final_pipeline_config.json")
        pipeline_config.write_text(json.dumps(final_pipeline_config(values), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.path == "/api/save":
            self._send(200, f"wrote {pipeline_config}\n")
            return
        if self.path == "/api/run":
            self._send_run_stream([sys.executable, str(ROOT / "scripts" / "run_hadros_final_pipeline.py"), str(pipeline_config)])
            return
        self._send(404, "not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--write-default-pipeline-config", type=Path)
    args = parser.parse_args()
    if args.print_schema:
        print(json.dumps(schema(), indent=2, sort_keys=True))
        return 0
    if args.write_default_pipeline_config is not None:
        args.write_default_pipeline_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_default_pipeline_config.write_text(
            json.dumps(final_pipeline_config(defaults()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.write_default_pipeline_config}")
        return 0
    Handler.config_path = args.config
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving HADROS final config at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
