#!/usr/bin/env python3
"""Orchestrate optional HADROS-CASCADE diagnostics from a config-web JSON file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LIMITATIONS = [
    "HADROS-CASCADE diagnostics are experimental.",
    "Particle-channel images are energy-proxy diagnostics, not physical luminosities.",
    "GEANT4 is used only in local homogeneous boxes.",
    "PYTHIA proxy does not replace GBW/IIM DIS physics.",
    "Massive geodesics are not implemented.",
    "Only massless/ultrarelativistic packets are propagated with effective null geodesics.",
]

EXPECTED_OUTPUTS = [
    "hadros_backward_camera_particle_summary.md",
    "hadros_backward_camera_particle_channels.npz",
    "observed_particles_by_pixel.csv",
    "observed_particles_by_pixel.jsonl",
    "observed_particles_by_pixel_summary.md",
    "observed_particle_pdg_histogram.csv",
    "observed_particle_channel_histogram.csv",
    "plots/hadros_backward_rgb.png",
    "plots/hadros_backward_gamma.png",
    "plots/hadros_backward_electromagnetic.png",
    "plots/observed_particles_by_pdg.png",
    "plots/observed_particles_by_channel.png",
    "plots/observed_gamma_map.png",
    "plots/observed_neutrino_map.png",
    "plots/observed_hadronic_map.png",
    "plots/observed_electromagnetic_map.png",
    "plots/observed_rgb_channels.png",
    "plots/hadros_backward_tau.png",
]


@dataclass
class Step:
    name: str
    command: list[str]
    reason: str
    optional_dependency: str = ""
    enabled: bool = True


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("output_dir", "output/config_web_cascade_run")
    data.setdefault("enable_cascade_diagnostics", False)
    return data


def bool_value(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_available(name: str) -> bool:
    return shutil.which(name) is not None


def command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def already_inside_micromamba_env(config: dict[str, Any]) -> bool:
    target = str(config.get("micromamba_env_name", "hadros-cascade")).strip() or "hadros-cascade"
    active_names = {
        str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip(),
        str(os.environ.get("MAMBA_DEFAULT_ENV", "")).strip(),
    }
    if target in active_names:
        return True
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    return bool(conda_prefix) and Path(conda_prefix).name == target


def cascade_prefix(config: dict[str, Any]) -> list[str]:
    if not bool_value(config, "use_micromamba_env", False):
        return []
    if bool_value(config, "no_micromamba", False) or already_inside_micromamba_env(config):
        return []
    executable = str(config.get("micromamba_executable", "micromamba")).strip() or "micromamba"
    env_name = str(config.get("micromamba_env_name", "hadros-cascade")).strip() or "hadros-cascade"
    return [executable, "run", "-n", env_name]


def prefixed(config: dict[str, Any], command: list[str]) -> list[str]:
    return [*cascade_prefix(config), *command]


def normalize_geant4_mode(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    mode = str(normalized.get("geant4_mode", "")).strip()
    if mode not in {"proxy", "real_safe", "real_direct"}:
        transport = str(normalized.get("geant4_transport_mode", "proxy"))
        safety = str(normalized.get("geant4_safety_mode", "strict"))
        one_particle = bool_value(normalized, "geant4_one_particle_per_run", False)
        if transport == "proxy":
            mode = "proxy"
        elif safety == "strict" and one_particle:
            mode = "real_safe"
        else:
            mode = "real_direct"
    if mode == "proxy":
        normalized["geant4_transport_mode"] = "proxy"
        normalized["geant4_safety_mode"] = "off"
        normalized["geant4_one_particle_per_run"] = False
        normalized.setdefault("uhe_transport_policy", "error")
    elif mode == "real_safe":
        normalized["geant4_transport_mode"] = "geant4"
        normalized["geant4_safety_mode"] = "strict"
        normalized["geant4_one_particle_per_run"] = True
        normalized.setdefault("geant4_execution_mode", "resumable_batches")
        normalized.setdefault("geant4_batch_target_energy_fraction", None)
        normalized.setdefault("geant4_batch_prioritize_energy_desc", True)
        normalized["uhe_transport_policy"] = "skip_to_escaped"
        normalized.setdefault("geant4_hadron_max_kinetic_gev", 1.0e5)
        normalized.setdefault("geant4_lepton_max_kinetic_gev", 1.0e5)
        normalized.setdefault("geant4_photon_max_kinetic_gev", 1.0e5)
    else:
        normalized["geant4_transport_mode"] = "geant4"
        normalized["geant4_safety_mode"] = "off"
        normalized["geant4_one_particle_per_run"] = False
        normalized.setdefault("uhe_transport_policy", "error")
    normalized["geant4_mode"] = mode
    return normalized


def write_run_manifest(output_dir: Path, config: dict[str, Any], steps: list[Step], warnings: list[str], *, dry_run: bool, returncode: int | None = None) -> None:
    generated: list[str] = []
    not_generated: list[str] = []
    for rel in EXPECTED_OUTPUTS:
        path = output_dir / rel
        (generated if path.exists() else not_generated).append(rel)
    required_modules = list(config.get("required_modules", []))
    modules_executed = []
    if bool_value(config, "run_uhe_dis", bool_value(config, "run_uhe_particle_cascade", False)):
        modules_executed.append("UHE DIS")
    if bool_value(config, "run_uhe_particle_cascade", bool_value(config, "enable_cascade_diagnostics", False)):
        modules_executed.append("particle cascade")
    if bool_value(config, "run_mev_torus_neutrinos", False):
        modules_executed.append("MeV torus neutrinos")
    payload = {
        "dry_run": dry_run,
        "returncode": returncode,
        "produce_uhe_collision_particles": bool_value(config, "produce_uhe_collision_particles", bool_value(config, "run_uhe_particle_cascade", False)),
        "physical_mode": "UHE_DIS_PLUS_PARTICLE_PRODUCTION" if bool_value(config, "produce_uhe_collision_particles", bool_value(config, "run_uhe_particle_cascade", False)) else "UHE_DIS_ONLY",
        "particle_production": bool_value(config, "produce_uhe_collision_particles", bool_value(config, "run_uhe_particle_cascade", False)),
        "observed_particles": config.get("observed_particles", []),
        "observed_particle_filter": config.get("observed_particle_filter", "all"),
        "observed_pdg_filter": config.get("observed_pdg_filter", ""),
        "observed_channel_filter": config.get("observed_channel_filter", ""),
        "observed_energy_mode": config.get("observed_energy_mode", "monochromatic"),
        "observed_energy_min": config.get("observed_energy_min"),
        "observed_energy_max": config.get("observed_energy_max"),
        "observed_momentum_mode": config.get("observed_momentum_mode", "integrated"),
        "camera_parameters": {
            "CAM_NX": config.get("camera_nx"),
            "CAM_NY": config.get("camera_ny"),
            "CAM_FOV_DEG": config.get("camera_fov_deg"),
            "CAM_THETA_DEG": config.get("camera_theta_deg"),
            "CAM_PHI_DEG": config.get("camera_phi_deg", config.get("phi_obs")),
            "CAM_R_OBS_RG": config.get("camera_r_obs_rg"),
            "CAM_R_MAX_RG": config.get("camera_r_max_rg"),
            "CAM_STEP": config.get("camera_step"),
        },
        "required_modules": required_modules,
        "modules_executed": modules_executed,
        "run_uhe_dis": bool_value(config, "run_uhe_dis", bool_value(config, "run_uhe_particle_cascade", False)),
        "run_uhe_particle_cascade": bool_value(config, "run_uhe_particle_cascade", bool_value(config, "enable_cascade_diagnostics", False)),
        "run_mev_torus_neutrinos": bool_value(config, "run_mev_torus_neutrinos", False),
        "particle_image_mode": config.get("particle_image_mode", "real_hadros_backward_kerr"),
        "camera_backend_requested": config.get("camera_backend", "auto"),
        "camera_backend_effective": "unknown_until_run",
        "cascade_backend": config.get("cascade_backend", "none"),
        "geant4_mode": config.get("geant4_mode", "proxy"),
        "dis_model": config.get("dis_model", "GBW"),
        "sigma_table_path": config.get("sigma_table_path", "data/sigma/sigma_nuN_CC_GBW.dat"),
        "outputs_generated": generated,
        "outputs_not_generated": not_generated,
        "warnings": warnings,
        "blocked_reason": "",
        "planned_steps": [step.name for step in steps if step.enabled],
        "is_physical_luminosity": False,
        "is_flux_observable": False,
        "observed_particles_by_pixel_generated": (output_dir / "observed_particles_by_pixel.csv").exists(),
        "pdg_preserved_to_camera": True,
        "particle_tracking_status": "PIXEL_PARTICLE_TRACKING_PARTIAL" if (output_dir / "observed_particles_by_pixel.csv").exists() or bool_value(config, "generate_particle_channel_images", True) else "PIXEL_PARTICLE_TRACKING_NOT_AVAILABLE",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HADROS-CASCADE Run Manifest",
        "",
        f"- observed_particles: `{payload['observed_particles']}`",
        f"- observed_particle_filter: `{payload['observed_particle_filter']}`",
        f"- observed_pdg_filter: `{payload['observed_pdg_filter']}`",
        f"- observed_channel_filter: `{payload['observed_channel_filter']}`",
        f"- observed_energy_mode: `{payload['observed_energy_mode']}`",
        f"- observed_energy_min: `{payload['observed_energy_min']}`",
        f"- observed_energy_max: `{payload['observed_energy_max']}`",
        f"- observed_momentum_mode: `{payload['observed_momentum_mode']}`",
        f"- required_modules: `{payload['required_modules']}`",
        f"- modules_executed: `{payload['modules_executed']}`",
        f"- run_uhe_dis: `{payload['run_uhe_dis']}`",
        f"- run_uhe_particle_cascade: `{payload['run_uhe_particle_cascade']}`",
        f"- run_mev_torus_neutrinos: `{payload['run_mev_torus_neutrinos']}`",
        f"- particle_image_mode: `{payload['particle_image_mode']}`",
        f"- camera_backend_requested: `{payload['camera_backend_requested']}`",
        f"- camera_backend_effective: `{payload['camera_backend_effective']}`",
        f"- cascade_backend: `{payload['cascade_backend']}`",
        f"- geant4_mode: `{payload['geant4_mode']}`",
        f"- dis_model: `{payload['dis_model']}`",
        f"- sigma_table_path: `{payload['sigma_table_path']}`",
        f"- is_physical_luminosity: `{payload['is_physical_luminosity']}`",
        f"- is_flux_observable: `{payload['is_flux_observable']}`",
        f"- observed_particles_by_pixel_generated: `{payload['observed_particles_by_pixel_generated']}`",
        f"- pdg_preserved_to_camera: `{payload['pdg_preserved_to_camera']}`",
        f"- particle_tracking_status: `{payload['particle_tracking_status']}`",
        "",
        "## Planned Steps",
        "",
        *([f"- `{item}`" for item in payload["planned_steps"]] or ["- None."]),
        "",
        "## Outputs Generated",
        "",
        *([f"- `{item}`" for item in generated] or ["- None detected yet."]),
        "",
        "## Outputs Not Generated",
        "",
        *([f"- `{item}`" for item in not_generated] or ["- None."]),
        "",
        "## Warnings",
        "",
        *([f"- {item}" for item in warnings] or ["- None."]),
    ]
    (output_dir / "run_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_steps(config: dict[str, Any], config_path: Path) -> tuple[list[Step], list[str]]:
    config = normalize_geant4_mode(config)
    output_dir = Path(config["output_dir"])
    warnings: list[str] = []
    steps: list[Step] = []
    if bool_value(config, "produce_uhe_collision_particles", False):
        return build_source_driven_particle_steps(config, config_path)
    observed = str(config.get("observed_particle_channel", "")).strip()
    if observed == "none":
        warnings.append("No particle-ray association channel selected.")
        return steps, warnings
    required_modules = set(config.get("required_modules", []))
    if "particle cascade" in required_modules and not bool_value(config, "enable_cascade_diagnostics", False):
        warnings.append("Particle-ray association requires cascade but cascade module is unavailable or disabled.")
        return steps, warnings
    if "MeV torus neutrinos" in required_modules and not bool_value(config, "run_mev_torus_neutrinos", False):
        warnings.append("Particle-ray association requires MeV torus neutrinos but MeV module is disabled.")
        return steps, warnings
    if not bool_value(config, "enable_cascade_diagnostics", False):
        warnings.append("Cascade diagnostics disabled by config.")
        return steps, warnings
    particle_image_mode = str(config.get("particle_image_mode", "real_hadros_backward_kerr"))
    if particle_image_mode == "real_hadros_backward_kerr":
        warnings.append("Particle-channel imaging requires the HADROS particle-ray association camera. Directional screen projection, legacy forward packet projection, hybrid packet screen, local_response_proxy images, and packet auto-frame are diagnostic-only routes.")
    elif particle_image_mode == "directional_screen_projection_debug_only":
        warnings.append("Directional screen projection requested explicitly as debug-only. It is not a scientific particle-ray association camera.")
    if str(config.get("packet_propagation_backend", "proxy_straight_line")) == "real_kerr_geodesic":
        warnings.append("This uses real Kerr null geodesic trajectories, but the channel images are still weighted-energy proxy maps, not physical luminosities. Redshift/radiative transfer are not fully calibrated unless explicitly enabled.")
    if bool_value(config, "auto_frame_particle_packets", False):
        if str(config.get("packet_propagation_backend", "proxy_straight_line")) != "real_kerr_geodesic":
            warnings.append("auto_frame_particle_packets requested but disabled because packet_propagation_backend is not real_kerr_geodesic.")
        elif str(config.get("particle_camera_mode", "exit_sky")) != "hybrid_packet_screen":
            warnings.append("auto_frame_particle_packets requested but disabled because particle_camera_mode is not hybrid_packet_screen.")
        else:
            warnings.append("Auto-frame chooses a diagnostic camera axis to capture propagated packets. It is not a physical observer selection.")

    event_generator = str(config.get("event_generator", "none"))
    cascade_backend = str(config.get("cascade_backend", "none"))
    if event_generator == "pythia_proxy":
        warnings.append("Current PYTHIA proxy uses e+e- -> hadrons and is for infrastructure/debug only. It is not a physical nuN DIS event generator.")
    response_mode = str(config.get("local_response_table_mode", "use_existing"))
    secondaries = output_dir / ("pythia_secondaries.jsonl" if event_generator == "pythia_proxy" else "secondaries.jsonl")
    prefix = cascade_prefix(config)
    pythia_available = env_available("pythia8-config") or bool(prefix)
    geant4_available = env_available("geant4-config") or bool(prefix)

    steps.append(Step(
        "prepare_analytic_inputs",
        prefixed(config, [
            sys.executable,
            "scripts/cascade/run_analytic_cascade_demo.py",
            "--output-dir",
            str(output_dir),
            "--n-events",
            str(int(config.get("n_events", 32))),
            "--energy-gev",
            str(float(config.get("neutrino_energy_gev", 1.0e4))),
            "--seed",
            str(int(config.get("seed", 12345))),
        ]),
        "Generate/reuse interaction_points.jsonl, primary_events.jsonl, and analytic audit products.",
        enabled=event_generator in {"none", "analytic", "pythia_proxy"} or cascade_backend == "analytic",
    ))

    if event_generator == "pythia_proxy":
        if pythia_available:
            steps.append(Step(
                "pythia_proxy_secondaries",
                prefixed(config, [
                    sys.executable,
                    "scripts/cascade/run_pythia_proxy_demo.py",
                    "--output-dir",
                    str(output_dir),
                    "--n-events",
                    str(int(config.get("n_events", 32))),
                    "--energy-gev",
                    str(float(config.get("neutrino_energy_gev", 1.0e4))),
                    "--seed",
                    str(int(config.get("seed", 12345))),
                ]),
                "Generate pythia_secondaries.jsonl with optional PYTHIA proxy/shower plumbing.",
                optional_dependency="PYTHIA",
            ))
        else:
            warnings.append("PYTHIA unavailable: pythia_proxy will be skipped unless the hadros-cascade environment is active.")

    if cascade_backend == "geant4_local_box":
        if geant4_available:
            geant4_transport_mode = str(config.get("geant4_transport_mode", "proxy"))
            geant4_safety_mode = str(config.get("geant4_safety_mode", "strict" if geant4_transport_mode == "geant4" else "off"))
            geant4_one_particle = bool_value(config, "geant4_one_particle_per_run", geant4_transport_mode == "geant4")
            geant4_execution_mode = str(config.get("geant4_execution_mode", "resumable_batches" if config.get("geant4_mode") == "real_safe" else "direct_grouped"))
            uhe_transport_policy = str(config.get("uhe_transport_policy", "skip_to_escaped" if config.get("geant4_mode") == "real_safe" else "error"))
            hadron_max = float(config.get("geant4_hadron_max_kinetic_gev", 1.0e5))
            lepton_max = float(config.get("geant4_lepton_max_kinetic_gev", 1.0e5))
            photon_max = float(config.get("geant4_photon_max_kinetic_gev", 1.0e5))
            target_fraction = config.get("geant4_batch_target_energy_fraction", None)
            try:
                target_fraction_value = float(target_fraction) if target_fraction not in {None, ""} else 0.0
            except (TypeError, ValueError):
                target_fraction_value = 0.0
            if geant4_transport_mode == "geant4" and geant4_execution_mode == "resumable_batches":
                steps.append(Step(
                    "geant4_real_resumable_batches",
                    prefixed(config, [
                        sys.executable,
                        "scripts/cascade/run_geant4_real_resumable_batches.py",
                        "--secondaries",
                        str(secondaries),
                        "--output-dir",
                        str(output_dir),
                        "--interaction-points",
                        str(output_dir / "interaction_points.jsonl"),
                        "--batch-mode",
                        str(config.get("geant4_batch_mode", "one_particle_per_process")),
                        "--workers",
                        str(int(config.get("geant4_batch_workers", 1))),
                        "--geant4-safety-mode",
                        geant4_safety_mode,
                        "--uhe-transport-policy",
                        uhe_transport_policy,
                        "--geant4-hadron-max-kinetic-gev",
                        f"{hadron_max:.17g}",
                        "--geant4-lepton-max-kinetic-gev",
                        f"{lepton_max:.17g}",
                        "--geant4-photon-max-kinetic-gev",
                        f"{photon_max:.17g}",
                        *(["--target-processed-energy-fraction", f"{target_fraction_value:.17g}"] if target_fraction_value > 0.0 else []),
                        *(["--prioritize-energy-desc"] if bool_value(config, "geant4_batch_prioritize_energy_desc", False) else []),
                        *(["--allow-partial-exit-zero"] if bool_value(config, "allow_partial_cascade_products", False) else []),
                    ]),
                    "Run real GEANT4 through slow resumable batches. This is the validated route for rich PYTHIA lists.",
                    optional_dependency="GEANT4",
                ))
            else:
                steps.append(Step(
                    "geant4_local_box",
                    prefixed(config, [
                        sys.executable,
                        "scripts/cascade/run_geant4_local_box_demo.py",
                        "--output-dir",
                        str(output_dir),
                        "--transport-mode",
                        geant4_transport_mode,
                        "--geant4-safety-mode",
                        geant4_safety_mode,
                        "--uhe-transport-policy",
                        uhe_transport_policy,
                        "--geant4-hadron-max-kinetic-gev",
                        f"{hadron_max:.17g}",
                        "--geant4-lepton-max-kinetic-gev",
                        f"{lepton_max:.17g}",
                        "--geant4-photon-max-kinetic-gev",
                        f"{photon_max:.17g}",
                        *(["--geant4-one-particle-per-run"] if geant4_one_particle else []),
                        "--reuse-existing",
                    ]),
                    "Run optional GEANT4 local homogeneous-box diagnostics.",
                    optional_dependency="GEANT4",
                ))
            steps.append(Step(
                "summarize_uhe_transport_policy",
                prefixed(config, [
                    sys.executable,
                    "scripts/cascade/summarize_uhe_transport_policy.py",
                    "--output-dir",
                    str(output_dir),
                ]),
                "Summarize UHE particles skipped from GEANT4 transport and passed to escaping packets.",
            ))
        else:
            warnings.append("GEANT4 unavailable: geant4_local_box real transport will be skipped; existing response tables may still be used.")

    if response_mode == "build_quick":
        steps.append(Step(
            "build_local_response_table_quick",
            prefixed(config, [sys.executable, "scripts/cascade/build_local_response_table.py", "--quick", "--output-dir", str(output_dir)]),
            "Build a small optional local response table.",
        ))
    elif response_mode == "build_from_secondaries":
        steps.append(Step(
            "build_local_response_table_from_secondaries",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/build_local_response_table.py",
                "--from-secondaries",
                str(secondaries),
                "--output-dir",
                str(output_dir),
            ]),
            "Build an expanded response table from observed secondary species.",
        ))
    elif response_mode == "refine_existing":
        steps.append(Step(
            "refine_local_response_table",
            prefixed(config, [sys.executable, "scripts/cascade/refine_local_response_table.py", "--output-dir", str(output_dir)]),
            "Refine an existing local response table based on coverage diagnostics.",
        ))

    if response_mode != "none":
        table_name = {
            "build_from_secondaries": "local_response_table_expanded.csv",
            "refine_existing": "local_response_table_refined.csv",
        }.get(response_mode, "local_response_table.csv")
        steps.append(Step(
            "apply_local_response",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/apply_local_response_to_events.py",
                "--secondaries",
                str(secondaries),
                "--table",
                str(output_dir / table_name),
                "--output-dir",
                str(output_dir),
            ]),
            "Apply an existing local response table to event secondaries.",
        ))
        steps.append(Step(
            "build_deposition_proxy",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/build_deposition_emissivity_proxy.py",
                "--output-dir",
                str(output_dir),
                "--input",
                str(output_dir / "response_weighted_deposition.csv"),
            ]),
            "Build deposition emissivity proxy from weighted local deposition.",
        ))

    if bool_value(config, "build_escaping_packets", True):
        steps.append(Step(
            "build_escaping_packets",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/build_escaping_particle_packets.py",
                "--input",
                str(output_dir / "geant4_escaped_particles.jsonl" if cascade_backend == "geant4_local_box" else secondaries),
                "--output-dir",
                str(output_dir),
                "--interaction-points",
                str(output_dir / "interaction_points.jsonl"),
                *(["--require-physical-interaction-points"] if bool_value(config, "require_physical_interaction_points", cascade_backend == "geant4_local_box") else []),
            ]),
            "Compress escaping energy into effective EscapingParticlePackets.",
        ))
        steps.append(Step(
            "audit_packet_origin",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/audit_packet_origin.py",
                "--output-dir",
                str(output_dir),
            ]),
            "Audit packet/intermediate positions and flag missing or default-like origins.",
        ))
    if bool_value(config, "classify_packets", True):
        steps.append(Step(
            "classify_escaping_packets",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/classify_escaping_packets.py",
                "--input",
                str(output_dir / "escaping_particle_packets.jsonl"),
                "--output-dir",
                str(output_dir),
            ]),
            "Classify packets for effective null propagation eligibility.",
        ))
    if bool_value(config, "propagate_null_packets", True):
        steps.append(Step(
            "propagate_null_packets",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/propagate_kerr_null_packets.py",
                "--packets",
                str(output_dir / "escaping_particle_packets.jsonl"),
                "--classification",
                str(output_dir / "escaping_packet_classification.csv"),
                "--straight-line",
                str(output_dir / "null_propagated_packets.csv"),
                "--output-dir",
                str(output_dir),
                "--packet-propagation-backend",
                str(config.get("packet_propagation_backend", "proxy_straight_line")),
                "--kerr-init-mode",
                str(config.get("kerr_init_mode", "zamo_tetrad")),
                *(["--normalize-null-momentum"] if bool_value(config, "normalize_null_momentum", True) else []),
            ]),
            "Run experimental null/Kerr packet propagation diagnostics.",
        ))
        steps.append(Step(
            "scan_kerr_packet_observers",
            prefixed(config, [
                sys.executable,
                "scripts/cascade/scan_kerr_packet_observers.py",
                "--packets",
                str(output_dir / "escaping_particle_packets.jsonl"),
                "--classification",
                str(output_dir / "escaping_packet_classification.csv"),
                "--straight-scan",
                str(output_dir / "packet_observer_scan.csv"),
                "--output-dir",
                str(output_dir),
                "--cones-deg",
                str(config.get("cone_deg", 30.0)),
                "--kerr-init-mode",
                str(config.get("kerr_init_mode", "zamo_tetrad")),
            ]),
            "Scan observer overlap for null-compatible escaping packets.",
        ))

    auto_frame_enabled = (
        bool_value(config, "auto_frame_particle_packets", False)
        and str(config.get("packet_propagation_backend", "proxy_straight_line")) == "real_kerr_geodesic"
        and str(config.get("particle_camera_mode", "exit_sky")) == "hybrid_packet_screen"
    )
    if auto_frame_enabled:
        steps.append(Step(
            "auto_frame_particle_packets",
            prefixed(config, [
                sys.executable,
                "scripts/science/diagnose_hybrid_camera_packet_overlap.py",
                "--real-kerr",
                str(output_dir / "real_kerr_propagated_packets.csv"),
                "--packets",
                str(output_dir / "escaping_particle_packets.jsonl"),
                "--classification",
                str(output_dir / "escaping_packet_classification.csv"),
                "--particle-channel-images",
                str(output_dir / "particle_channel_images.npz"),
                "--output-dir",
                str(output_dir),
                "--camera-theta-deg",
                str(config.get("camera_theta_deg", config.get("theta_obs", 0.0)) or 0.0),
                "--camera-phi-deg",
                str(config.get("camera_phi_deg", config.get("phi_obs", 0.0)) or 0.0),
                "--camera-fov-deg",
                str(config.get("camera_fov_deg", config.get("cone_deg", 30.0)) or config.get("cone_deg", 30.0)),
                "--camera-r-obs-rg",
                str(config.get("camera_r_obs_rg", "")),
                "--image-width",
                str(int(config.get("camera_nx", 64) or 64)),
                "--image-height",
                str(int(config.get("camera_ny", config.get("camera_nx", 64)) or 64)),
                "--auto-frame-capture-fraction",
                str(config.get("auto_frame_capture_fraction", 0.99)),
                "--auto-frame-fov-margin-deg",
                str(config.get("auto_frame_fov_margin_deg", 5.0)),
            ]),
            "Compute diagnostic auto-frame for propagated real-Kerr packets. Not a physical observer selection.",
        ))

    if bool_value(config, "generate_particle_channel_images", True):
        if particle_image_mode == "real_hadros_backward_kerr":
            steps.append(Step(
                "real_hadros_backward_kerr_particle_camera",
                prefixed(config, [
                    sys.executable,
                    "scripts/science/run_real_kerr_particle_camera.py",
                    "--output-dir",
                    str(output_dir),
                    "--camera-nx",
                    str(int(config.get("camera_nx", 64) or 64)),
                    "--camera-ny",
                    str(int(config.get("camera_ny", config.get("camera_nx", 64)) or 64)),
                    "--camera-fov-deg",
                    str(config.get("camera_fov_deg", config.get("cone_deg", 60.0)) or 60.0),
                    "--camera-theta-deg",
                    str(config.get("camera_theta_deg", config.get("theta_obs", 70.0)) or 70.0),
                    "--camera-r-obs-rg",
                    str(config.get("camera_r_obs_rg", 80.0) or 80.0),
                    "--camera-r-max-rg",
                    str(config.get("camera_r_max_rg", 120.0) or 120.0),
                    "--camera-step",
                    str(config.get("camera_step", 0.02) or 0.02),
                    "--aspin",
                    str(config.get("spin", config.get("ASPIN", 0.8)) or 0.8),
                ]),
                "Run the HADROS particle-ray association camera. Fails clearly if physical particle positions are unavailable.",
            ))
        elif particle_image_mode == "hadros_backward_camera":
            steps.append(Step(
                "hadros_backward_camera_particle_images",
                prefixed(config, [
                    sys.executable,
                    "scripts/science/run_hadros_backward_camera_particle_image.py",
                    "--output-dir",
                    str(output_dir),
                    "--camera-backend",
                    str(config.get("camera_backend", "auto")),
                    "--camera-nx",
                    str(int(config.get("camera_nx", 64) or 64)),
                    "--camera-ny",
                    str(int(config.get("camera_ny", config.get("camera_nx", 64)) or 64)),
                    "--camera-fov-deg",
                    str(config.get("camera_fov_deg", config.get("cone_deg", 60.0)) or 60.0),
                    "--camera-theta-deg",
                    str(config.get("camera_theta_deg", config.get("theta_obs", 70.0)) or 70.0),
                    "--camera-phi-deg",
                    str(config.get("camera_phi_deg", config.get("phi_obs", 0.0)) or 0.0),
                    "--camera-r-obs-rg",
                    str(config.get("camera_r_obs_rg", 80.0) or 80.0),
                    "--camera-r-max-rg",
                    str(config.get("camera_r_max_rg", 120.0) or 120.0),
                    "--camera-step",
                    str(config.get("camera_step", 0.02) or 0.02),
                    "--energy-gev",
                    str(float(config.get("neutrino_energy_gev", 1.0e9))),
                    "--sigma-table",
                    str(config.get("sigma_table_path", "data/sigma/sigma_nuN_CC_GBW.dat")),
                    "--observed-particle-filter",
                    str(config.get("observed_particle_filter", "all")),
                    "--observed-pdg-filter",
                    str(config.get("observed_pdg_filter", "")),
                    "--observed-channel-filter",
                    str(config.get("observed_channel_filter", "")),
                ]),
                "Generate camera-selected particle-channel diagnostics through the real HADROS backward camera engine. No forward packet projection is used.",
            ))
        else:
            command = prefixed(config, [
                sys.executable,
                "scripts/cascade/build_particle_channel_images.py",
                "--output-dir",
                str(output_dir),
                "--packets",
                str(output_dir / "escaping_particle_packets.jsonl"),
                "--classification",
                str(output_dir / "escaping_packet_classification.csv"),
                "--kerr-scan",
                str(output_dir / "kerr_packet_observer_scan.csv"),
                "--straight-scan",
                str(output_dir / "packet_observer_scan.csv"),
                "--packet-propagation-backend",
                str(config.get("packet_propagation_backend", "proxy_straight_line")),
                "--observer-mode",
                str(config.get("observer_mode", "best_cone")),
                "--cone-deg",
                str(config.get("cone_deg", 30.0)),
                "--camera-source",
                str(config.get("camera_source", "cascade_defaults")),
                "--image-width",
                str(int(config.get("camera_nx", 64) or 64)),
                "--image-height",
                str(int(config.get("camera_ny", config.get("camera_nx", 64)) or 64)),
                "--camera-fov-deg",
                str(config.get("camera_fov_deg", config.get("cone_deg", 30.0)) or config.get("cone_deg", 30.0)),
                "--camera-theta-deg",
                str(config.get("camera_theta_deg", config.get("theta_obs", 0.0)) or 0.0),
                "--camera-phi-deg",
                str(config.get("camera_phi_deg", config.get("phi_obs", 0.0)) or 0.0),
                "--camera-r-obs-rg",
                str(config.get("camera_r_obs_rg", "")),
                "--camera-r-max-rg",
                str(config.get("camera_r_max_rg", "")),
                "--camera-step",
                str(config.get("camera_step", "")),
                "--particle-camera-mode",
                str(config.get("particle_camera_mode", "exit_sky")),
            ])
            if auto_frame_enabled:
                command.extend([
                    "--auto-frame-json",
                    str(output_dir / "hybrid_camera_autoframe.json"),
                    "--auto-frame-capture-fraction",
                    str(config.get("auto_frame_capture_fraction", 0.99)),
                ])
            if config.get("observer_mode") == "manual":
                command.extend(["--theta-deg", str(config.get("theta_obs", 0.0)), "--phi-deg", str(config.get("phi_obs", 0.0))])
            steps.append(Step("particle_channel_images_legacy_diagnostic", command, "Generate legacy packet-projection diagnostic images. Deprecated for scientific imaging."))
            steps.append(Step(
                "particle_channel_image_audit",
                prefixed(config, [
                    sys.executable,
                    "scripts/cascade/build_particle_channel_image_audit.py",
                    "--output-dir",
                    str(output_dir),
                    "--channel-csv",
                    str(output_dir / "particle_channel_images.csv"),
                    "--channel-summary",
                    str(output_dir / "particle_channel_images_summary.md"),
                ]),
                "Build legacy particle-channel image audit and quality report.",
            ))

    steps.insert(0, Step(
        "check_environment",
        [
            sys.executable,
            "scripts/cascade/check_cascade_environment.py",
            "--output-dir",
            str(output_dir),
            *(["--use-micromamba-env", "--micromamba-env-name", str(config.get("micromamba_env_name", "hadros-cascade")), "--micromamba-executable", str(config.get("micromamba_executable", "micromamba"))] if bool_value(config, "use_micromamba_env", False) else []),
        ],
        "Check optional PYTHIA/GEANT4/HDF5/Python dependency status.",
    ))
    steps.insert(1, Step(
        "record_config",
        prefixed(config, [sys.executable, "-m", "json.tool", str(config_path)]),
        "Validate and pretty-print the config-web cascade JSON.",
    ))
    return steps, warnings


def _event_record_inputs(config: dict[str, Any]) -> list[str]:
    raw = config.get("powheg_pythia_event_record_inputs", "")
    items: list[str] = []
    if isinstance(raw, list):
        items.extend(str(item).strip() for item in raw if str(item).strip())
    else:
        for token in str(raw).replace(",", ";").split(";"):
            if token.strip():
                items.append(token.strip())
    for key, interaction in [
        ("powheg_pythia_cc_event_record", "CC"),
        ("powheg_pythia_nc_event_record", "NC"),
        ("cc_event_record_dump", "CC"),
        ("nc_event_record_dump", "NC"),
    ]:
        value = str(config.get(key, "")).strip()
        if value:
            items.append(value if ":" in value else f"{interaction}:{value}")
    return items


def build_source_driven_particle_steps(config: dict[str, Any], config_path: Path) -> tuple[list[Step], list[str]]:
    output_dir = Path(config["output_dir"])
    run_dir = output_dir.parent if output_dir.name == "cascade" else output_dir
    warnings = [
        "Source-driven particle-production mode enabled. Proxy, legacy packet projection, and particle-to-screen fallbacks are disabled.",
        "MISSING_PARTICLE_POSITION is checked only after GEANT4/ZAMO and the particle-ray association camera have run.",
    ]
    prefix = cascade_prefix(config)
    event_inputs = _event_record_inputs(config)
    ray_cache = output_dir / "rays" / "kerr_geodesics_e2e.bin"
    steps: list[Step] = [
        Step(
            "check_environment",
            [
                sys.executable,
                "scripts/cascade/check_cascade_environment.py",
                "--output-dir",
                str(output_dir),
                *(["--use-micromamba-env", "--micromamba-env-name", str(config.get("micromamba_env_name", "hadros-cascade")), "--micromamba-executable", str(config.get("micromamba_executable", "micromamba"))] if bool_value(config, "use_micromamba_env", False) else []),
            ],
            "Check optional PYTHIA/GEANT4/HDF5/Python dependency status.",
        ),
        Step(
            "record_config",
            [sys.executable, "-m", "json.tool", str(config_path)],
            "Validate and pretty-print the config-web cascade JSON.",
        ),
    ]
    steps.extend(
        [
            Step(
                "prepare_run_local_ray_cache_dir",
                [sys.executable, "-c", f"from pathlib import Path; Path({str(ray_cache.parent)!r}).mkdir(parents=True, exist_ok=True)"],
                "Prepare run-local UHE Kerr ray-cache directory.",
            ),
            Step(
                "hadros_uhe_kerr_rays",
                [
                    "./compute_kerr_geodesics",
                    str(config.get("spin", config.get("ASPIN", 0.8)) or 0.8),
                    str(config.get("camera_r_obs_rg", 80.0) or 80.0),
                    str(config.get("camera_theta_deg", config.get("theta_obs", 70.0)) or 70.0),
                    str(config.get("camera_fov_deg", config.get("cone_deg", 60.0)) or 60.0),
                    str(int(config.get("camera_nx", 8) or 8)),
                    str(int(config.get("camera_ny", config.get("camera_nx", 8)) or 8)),
                    str(config.get("camera_r_max_rg", 120.0) or 120.0),
                    str(config.get("camera_step", 0.05) or 0.05),
                    str(ray_cache),
                ],
                "Generate run-local HADROS UHE Kerr ray samples.",
            ),
            Step(
                "powheg_pythia_event_records",
                [
                    sys.executable,
                    "scripts/science/run_powheg_pythia_event_records.py",
                    "--output-dir",
                    str(output_dir),
                    "--run-name",
                    str(config.get("run_name", "config_web_run")),
                    "--n-events",
                    str(int(config.get("n_events", 10) or 10)),
                    "--interaction-points",
                    str(output_dir / "interaction_points_ray_linked.jsonl"),
                    "--dis-mode",
                    str(config.get("dis_model", "both")).lower(),
                    "--mode",
                    str(config.get("powheg_interaction_mode", "both")).lower(),
                    "--seed",
                    str(int(config.get("seed", 12345) or 12345)),
                    *(["--pwhg-main", str(config.get("powheg_executable_path", "")).strip()] if str(config.get("powheg_executable_path", "")).strip() else []),
                    *(["--pythia8-config", str(config.get("pythia8_config_path", "")).strip()] if str(config.get("pythia8_config_path", "")).strip() else []),
                    *(["--lhe-file", str(config.get("lhe_file_path", "")).strip()] if str(config.get("lhe_file_path", "")).strip() else []),
                    *(["--reuse-existing-lhe"] if bool_value(config, "reuse_existing_lhe", False) else []),
                    *[part for spec in event_inputs for part in ("--event-record-input", spec)],
                ],
                "Generate or convert run-local real POWHEG/PYTHIA event records to HADROS particle records. Blocks here if no versioned real event-record generator/input is available.",
            ),
            Step(
                "interaction_points_initial_global_positions",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/sample_powheg_global_interaction_points.py",
                        "--input",
                        str(output_dir / "hadros_particle_events.jsonl"),
                        "--output",
                        str(output_dir / "interaction_points.jsonl"),
                        "--summary",
                        str(output_dir / "interaction_points_summary.md"),
                        "--config",
                        str(config_path),
                    ],
                ),
                "Create initial event interaction-position records consumed by GEANT4/ZAMO. The incoming geodesic link is applied later by build_uhe_ray_event_link.py.",
            ),
            Step(
                "geant4_real_safe_zamo_positions",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/run_powheg_pythia_geant4_resumable.py",
                        "--input",
                        str(output_dir / "hadros_particle_events.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--interaction-points",
                        str(output_dir / "interaction_points.jsonl"),
                        "--workers",
                        str(int(config.get("geant4_batch_workers", 1) or 1)),
                        "--geant4-app",
                        "build/cascade_geant4_local_box",
                    ],
                ),
                "Run GEANT4_LOCAL_BOX_REAL_SAFE and ZAMO local-to-global position propagation.",
            ),
            Step(
                "real_hadros_backward_kerr_particle_camera",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/run_real_kerr_particle_camera.py",
                        "--input",
                        str(output_dir / "geant4_ready_particles.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--camera-nx",
                        str(int(config.get("camera_nx", 8) or 8)),
                        "--camera-ny",
                        str(int(config.get("camera_ny", config.get("camera_nx", 8)) or 8)),
                        "--camera-fov-deg",
                        str(config.get("camera_fov_deg", 60.0) or 60.0),
                        "--camera-theta-deg",
                        str(config.get("camera_theta_deg", 70.0) or 70.0),
                        "--camera-r-obs-rg",
                        str(config.get("camera_r_obs_rg", 80.0) or 80.0),
                        "--camera-r-max-rg",
                        str(config.get("camera_r_max_rg", 120.0) or 120.0),
                        "--camera-step",
                        str(config.get("camera_step", 0.05) or 0.05),
                        "--aspin",
                        str(config.get("spin", config.get("ASPIN", 0.8)) or 0.8),
                        "--skip-build",
                    ],
                ),
                "Run the HADROS particle-ray association camera after GEANT4/ZAMO positions exist.",
            ),
            Step(
                "incoming_uhe_ray_event_link",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/build_uhe_ray_event_link.py",
                        "--geodesic-cache",
                        str(ray_cache),
                        "--interaction-points",
                        str(output_dir / "interaction_points.jsonl"),
                        "--particles",
                        str(output_dir / "hadros_particle_events.jsonl"),
                        "--ready",
                        str(output_dir / "geant4_ready_particles.jsonl"),
                        "--observed-csv",
                        str(output_dir / "observed_particles_by_pixel.csv"),
                        "--observed-jsonl",
                        str(output_dir / "observed_particles_by_pixel.jsonl"),
                        "--output-dir",
                        str(output_dir),
                        "--config",
                        str(config_path),
                    ],
                ),
                "Attach incoming_ray_id, ray_sample_index, and incoming-geodesic column to event, GEANT4, and observed rows.",
            ),
            Step(
                "gbw_iim_incoming_geodesic_reweighting",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/build_gbw_iim_real_kerr_reweighting.py",
                        "--particles",
                        str(output_dir / "hadros_particle_events.jsonl"),
                        "--interaction-points",
                        str(output_dir / "interaction_points_ray_linked.jsonl"),
                        "--ready",
                        str(output_dir / "geant4_ready_particles.jsonl"),
                        "--observed",
                        str(output_dir / "observed_particles_by_pixel.csv"),
                        "--output-dir",
                        str(output_dir / "gbw_iim_reweighting"),
                        "--camera-output-dir",
                        str(output_dir),
                        "--config",
                        str(config_path),
                    ],
                ),
                "Apply GBW/IIM weights using the incoming geodesic column and write gbw_iim_camera_summary.csv in the run-local cascade directory.",
            ),
            Step(
                "paper_ready_science_plots",
                prefixed(
                    config,
                    [
                        sys.executable,
                        "scripts/science/build_scientific_plot_bundle.py",
                        "--run-dir",
                        str(run_dir),
                        "--dis-model",
                        "both",
                        "--production-plots",
                        "--geant4-plots",
                        "--observed-kerr-plots",
                        "--gbw-iim-plots",
                    ],
                ),
                "Build the 10 paper-ready science figures from run-local products only.",
            ),
            Step(
                "paper_ready_dashboard",
                prefixed(config, [sys.executable, "scripts/build_run_plot_dashboard.py", "--run-dir", str(run_dir)]),
                "Build the run-local dashboard with paper-ready science figures first.",
            ),
        ]
    )
    return steps, warnings


def write_plan(path: Path, config: dict[str, Any], steps: list[Step], warnings: list[str]) -> None:
    lines = [
        "# HADROS-CASCADE Config-Web Execution Plan",
        "",
        (
            "Source-driven particle-production chain. Not physical luminosity."
            if bool_value(config, "produce_uhe_collision_particles", False)
            else "Diagnostic cascade only. Not physical luminosity."
        ),
        (
            "Proxy and particle-to-screen fallbacks disabled."
            if bool_value(config, "produce_uhe_collision_particles", False)
            else "Diagnostic proxy only. Not physical luminosity."
        ),
        "",
        "## What did the camera observe?",
        "",
        f"- particle/channel: `{config.get('observed_particle_channel', config.get('observed_particles', []))}`",
        f"- energy mode: `{config.get('observed_energy_mode', 'monochromatic')}`",
        f"- energy range: `{config.get('observed_energy_min')}` - `{config.get('observed_energy_max')}`",
        f"- momentum mode: `{config.get('observed_momentum_mode', 'integrated')}`",
        f"- required modules: `{config.get('required_modules', [])}`",
        "",
        f"- output_dir: `{config.get('output_dir')}`",
        f"- enable_cascade_diagnostics: `{config.get('enable_cascade_diagnostics')}`",
        f"- event_generator: `{config.get('event_generator')}`",
        f"- cascade_backend: `{config.get('cascade_backend')}`",
        f"- geant4_mode: `{config.get('geant4_mode')}`",
        f"- geant4_transport_mode: `{config.get('geant4_transport_mode')}`",
        f"- geant4_safety_mode: `{config.get('geant4_safety_mode')}`",
        f"- geant4_one_particle_per_run: `{bool_value(config, 'geant4_one_particle_per_run', False)}`",
        f"- geant4_execution_mode: `{config.get('geant4_execution_mode', 'resumable_batches')}`",
        f"- geant4_batch_target_energy_fraction: `{config.get('geant4_batch_target_energy_fraction', '')}`",
        f"- geant4_batch_prioritize_energy_desc: `{config.get('geant4_batch_prioritize_energy_desc', False)}`",
        f"- uhe_transport_policy: `{config.get('uhe_transport_policy')}`",
        f"- geant4_hadron_max_kinetic_gev: `{config.get('geant4_hadron_max_kinetic_gev', 1.0e5)}`",
        f"- geant4_lepton_max_kinetic_gev: `{config.get('geant4_lepton_max_kinetic_gev', 1.0e5)}`",
        f"- geant4_photon_max_kinetic_gev: `{config.get('geant4_photon_max_kinetic_gev', 1.0e5)}`",
        f"- local_response_table_mode: `{config.get('local_response_table_mode')}`",
        f"- kerr_init_mode: `{config.get('kerr_init_mode')}`",
        f"- particle_image_mode: `{config.get('particle_image_mode', 'real_hadros_backward_kerr')}`",
        f"- hadros_camera_backend: `{config.get('camera_backend', 'auto')}`",
        f"- sigma_table_path: `{config.get('sigma_table_path', 'data/sigma/sigma_nuN_CC_GBW.dat')}`",
        f"- packet_propagation_backend: `{config.get('packet_propagation_backend', 'proxy_straight_line')}`",
        f"- particle_camera_mode: `{config.get('particle_camera_mode', 'exit_sky')}`",
        f"- auto_frame_particle_packets: `{bool_value(config, 'auto_frame_particle_packets', False)}`",
        f"- auto_frame_capture_fraction: `{config.get('auto_frame_capture_fraction', '')}`",
        f"- auto_frame_fov_margin_deg: `{config.get('auto_frame_fov_margin_deg', '')}`",
        f"- particle_camera_source: `{config.get('camera_source', 'cascade_defaults')}`",
        f"- particle_camera_nx: `{config.get('camera_nx', '')}`",
        f"- particle_camera_ny: `{config.get('camera_ny', '')}`",
        f"- particle_camera_fov_deg: `{config.get('camera_fov_deg', '')}`",
        f"- particle_camera_theta_deg: `{config.get('camera_theta_deg', '')}`",
        f"- particle_camera_phi_deg: `{config.get('camera_phi_deg', '')}`",
        f"- particle_camera_r_obs_rg: `{config.get('camera_r_obs_rg', '')}`",
        f"- particle_camera_r_max_rg: `{config.get('camera_r_max_rg', '')}`",
        f"- particle_camera_step: `{config.get('camera_step', '')}`",
        f"- use_micromamba_env: `{bool_value(config, 'use_micromamba_env', False)}`",
        f"- micromamba_env_name: `{config.get('micromamba_env_name', 'hadros-cascade')}`",
        f"- micromamba_executable: `{config.get('micromamba_executable', 'micromamba')}`",
        f"- cascade_command_prefix: `{' '.join(cascade_prefix(config))}`",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- No warnings."])
    lines.extend(["", "## Steps", ""])
    if not steps:
        lines.append("- No steps enabled.")
    for idx, step in enumerate(steps, start=1):
        status = "enabled" if step.enabled else "disabled"
        dep = f" Optional dependency: {step.optional_dependency}." if step.optional_dependency else ""
        lines.extend([
            f"### {idx}. {step.name}",
            "",
            f"- status: `{status}`",
            f"- reason: {step.reason}{dep}",
            "",
            "```bash",
            command_text(step.command),
            "```",
            "",
        ])
    lines.extend(["## Expected UI Outputs", ""])
    for rel in EXPECTED_OUTPUTS:
        lines.append(f"- `{Path(config.get('output_dir', 'output/config_web_cascade_run')) / rel}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_steps(steps: list[Step]) -> tuple[int, str]:
    output: list[str] = []
    for step in steps:
        if not step.enabled:
            output.append(f"[skip] {step.name}")
            continue
        output.append(f"$ {command_text(step.command)}")
        proc = subprocess.run(step.command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output.append(proc.stdout)
        if proc.returncode != 0:
            output.append(f"[fail] {step.name}: return code {proc.returncode}")
            return proc.returncode, "\n".join(output)
    return 0, "\n".join(output)


def write_command_trace(path: Path, steps: list[Step], config: dict[str, Any]) -> None:
    lines = [
        "# Config-Web E2E Command Trace",
        "",
        f"- use_micromamba_env: `{bool_value(config, 'use_micromamba_env', False)}`",
        f"- no_micromamba: `{bool_value(config, 'no_micromamba', False)}`",
        f"- already_inside_target_env: `{already_inside_micromamba_env(config)}`",
        f"- effective_micromamba_prefix: `{command_text(cascade_prefix(config)) or 'none'}`",
        "",
        "| step | enabled | command |",
        "|---|---:|---|",
    ]
    for step in steps:
        lines.append(f"| `{step.name}` | `{int(step.enabled)}` | `{command_text(step.command)}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-micromamba", action="store_true", help="Run subprocesses directly instead of wrapping them in micromamba run.")
    args = parser.parse_args()
    config = normalize_geant4_mode(load_config(args.config))
    if args.no_micromamba:
        config["no_micromamba"] = True
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, warnings = build_steps(config, args.config)
    plan_path = output_dir / "cascade_execution_plan.md"
    write_command_trace(output_dir / "e2e_command_trace.md", steps, config)
    write_plan(plan_path, config, steps, warnings)
    if str(config.get("observed_particle_channel", "")).strip() == "none":
        write_run_manifest(output_dir, config, steps, warnings, dry_run=args.dry_run, returncode=2)
        print("No observed particle/channel selected.")
        return 2
    if args.dry_run:
        write_run_manifest(output_dir, config, steps, warnings, dry_run=True, returncode=0)
        print(f"dry_run_plan={plan_path}")
        print(plan_path.read_text(encoding="utf-8"))
        return 0
    returncode, text = run_steps(steps)
    log_path = output_dir / "cascade_execution.log"
    log_path.write_text(text, encoding="utf-8")
    write_run_manifest(output_dir, config, steps, warnings, dry_run=False, returncode=returncode)
    print(text)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
