#!/usr/bin/env python3
"""Lightweight tests for Photon Escape Classifier Phase 1."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


final_pipeline = load_module("run_hadros_final_pipeline_photon_escape", "scripts/run_hadros_final_pipeline.py")
phase2_hits = load_module("build_photon_observer_sphere_hits", "scripts/science/build_photon_observer_sphere_hits.py")


def compile_backend(tmp: Path) -> Path:
    binary = tmp / "compute_kerr_photon_escape_classifier"
    cmd = [
        "g++",
        "-std=c++17",
        "-Iinclude",
        "apps/compute_kerr_photon_escape_classifier.cpp",
        "src/photon_escape_classifier.cpp",
        "src/cascade/kerr_local_tetrad.cpp",
        "src/kerr_metric.cpp",
        "src/kerr_geodesic.cpp",
        "-o",
        str(binary),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return binary


def classifier_command(
    tmp: Path,
    binary: Path,
    rows: list[dict[str, object]],
    *,
    invariant_tolerance: str = "1e-6",
    horizon_tolerance: str = "1e-6",
    fail_on_invariant: str = "true",
) -> tuple[list[str], Path, Path, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_path = tmp / "geant4_ready_particles.jsonl"
    output_jsonl = tmp / "photon_escape_classifier.jsonl"
    summary_csv = tmp / "photon_escape_summary.csv"
    summary_md = tmp / "photon_escape_summary.md"
    provenance = tmp / "photon_escape_provenance.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/science/run_kerr_photon_escape_classifier.py",
        "--input", str(input_path),
        "--output-jsonl", str(output_jsonl),
        "--summary-csv", str(summary_csv),
        "--summary-md", str(summary_md),
        "--provenance", str(provenance),
        "--backend", str(binary),
        "--spin", "0.0",
        "--observer-radius-rg", "20.0",
        "--max-radius-rg", "40.0",
        "--photon-geodesic-step-rg", "0.02",
        "--photon-max-geodesic-steps", "10000",
        "--photon-null-norm-tolerance", "1e-8",
        "--photon-invariant-tolerance", invariant_tolerance,
        "--photon-horizon-crossing-tolerance-rg", horizon_tolerance,
        "--photon-fail-on-invariant-violation", fail_on_invariant,
        "--photon-min-energy-gev", "0.0",
        "--photon-observer-frame", "ZAMO",
    ]
    return cmd, output_jsonl, summary_csv, provenance


def run_classifier(
    tmp: Path,
    binary: Path,
    rows: list[dict[str, object]],
    *,
    invariant_tolerance: str = "1e-6",
    horizon_tolerance: str = "1e-6",
    fail_on_invariant: str = "true",
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    cmd, output_jsonl, summary_csv, provenance = classifier_command(
        tmp,
        binary,
        rows,
        invariant_tolerance=invariant_tolerance,
        horizon_tolerance=horizon_tolerance,
        fail_on_invariant=fail_on_invariant,
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    out_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return out_rows, summary, prov


def photon_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": 1,
        "particle_id": 1,
        "pdg": 22,
        "energy_gev": 10.0,
        "global_exit_r_rg": 10.0,
        "global_exit_theta_rad": 1.5707963267948966,
        "global_exit_phi_rad": 0.0,
        "global_position_status": "GLOBAL_POSITION_VALID_ZAMO_TETRAD",
        "global_momentum_status": "GLOBAL_MOMENTUM_ZAMO_SPATIAL_TRIAD",
        "momentum_input_mode": "zamo_tetrad",
        "px": 0.0,
        "py": 0.0,
        "pz": 1.0,
    }
    row.update(updates)
    return row


def test_pdg_filtering(binary: Path, tmp: Path) -> None:
    rows = [photon_row(), photon_row(event_id=2, particle_id=2, pdg=11)]
    out_rows, summary, _ = run_classifier(tmp, binary, rows)
    if len(out_rows) != 1:
        raise AssertionError(f"expected one photon output row, got {out_rows}")
    if summary["n_input_particles"] != "2" or summary["n_photons"] != "1" or summary["n_non_photons"] != "1":
        raise AssertionError(f"bad filtering summary: {summary}")


def test_radial_outward_reaches_observer(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row()])
    if out_rows[0]["classification"] != "reaches_observer_sphere":
        raise AssertionError(f"outward photon did not reach observer: {out_rows[0]}")
    if out_rows[0]["observer_crossing_interpolated"] is not True:
        raise AssertionError(f"observer crossing was not interpolated: {out_rows[0]}")
    if abs(float(out_rows[0]["observer_crossing_r_rg"]) - 20.0) > 1.0e-12:
        raise AssertionError(f"bad observer crossing radius: {out_rows[0]}")


def test_radial_inward_captured(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row(pz=-1.0)], fail_on_invariant="false")
    if out_rows[0]["classification"] != "captured_by_black_hole":
        raise AssertionError(f"inward photon was not captured: {out_rows[0]}")
    if int(out_rows[0]["geodesic_steps"]) <= 0:
        raise AssertionError(f"capture did not require an integrated horizon crossing: {out_rows[0]}")


def test_ambiguous_generic_momentum_rejected(binary: Path, tmp: Path) -> None:
    row = photon_row()
    row.pop("momentum_input_mode")
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["classification"] != "integration_failed_ambiguous_momentum_input":
        raise AssertionError(f"ambiguous px/py/pz was not rejected: {out_rows[0]}")


def test_global_boyer_lindquist_mode_accepted(binary: Path, tmp: Path) -> None:
    row = photon_row(momentum_input_mode="global_boyer_lindquist")
    row.pop("px")
    row.pop("py")
    row.pop("pz")
    row.update({"global_px": 1.0, "global_py": 0.0, "global_pz": 0.0})
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["momentum_input_mode"] != "global_boyer_lindquist":
        raise AssertionError(f"momentum_input_mode was not preserved: {out_rows[0]}")
    if out_rows[0]["classification"] != "reaches_observer_sphere":
        raise AssertionError(f"global_boyer_lindquist mode was not accepted: {out_rows[0]}")


def test_invalid_null_momentum(binary: Path, tmp: Path) -> None:
    row = photon_row(momentum_input_mode="covariant_p_mu", p_t=-1.0, p_r=0.0, p_theta=0.0, p_phi=0.0)
    out_rows, _, _ = run_classifier(tmp, binary, [row])
    if out_rows[0]["classification"] != "integration_failed_invalid_null_momentum":
        raise AssertionError(f"invalid null momentum was not rejected: {out_rows[0]}")


def test_invariant_violation_can_fail(binary: Path, tmp: Path) -> None:
    out_rows, _, _ = run_classifier(tmp, binary, [photon_row()], invariant_tolerance="1e-30")
    if out_rows[0]["classification"] != "integration_failed_invariant_violation":
        raise AssertionError(f"invariant violation did not fail: {out_rows[0]}")


def test_negative_tolerances_rejected(binary: Path, tmp: Path) -> None:
    cmd, _, _, _ = classifier_command(tmp, binary, [photon_row()], invariant_tolerance="-1.0")
    completed = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        raise AssertionError("negative photon_invariant_tolerance was accepted")
    if "Invalid photon escape classifier configuration" not in completed.stderr:
        raise AssertionError(f"negative tolerance did not fail clearly: {completed.stderr}")


def test_config_web_contains_all_parameters() -> None:
    text = (ROOT / "scripts/config_web_final.py").read_text(encoding="utf-8")
    for needle in [
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
        "observer_sphere_hits",
    ]:
        if needle not in text:
            raise AssertionError(f"config_web_final.py missing {needle}")


def test_pipeline_passes_all_parameters(tmp: Path) -> None:
    config = {
        "run_name": "PhotonPipelinePass",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 3,
        "camera_ny": 3,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "escape_classifier",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "disabled_until_validated",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    step = next(item for item in steps if item.name == "photon_escape_classifier")
    command = step.command
    expected_pairs = {
        "--photon-max-geodesic-steps": "1234",
        "--photon-geodesic-step-rg": "0.03",
        "--photon-min-energy-gev": "2.0",
        "--photon-horizon-crossing-tolerance-rg": "1e-07",
        "--photon-observer-frame": "ZAMO",
        "--photon-fail-on-invariant-violation": "true",
    }
    for flag, value in expected_pairs.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise AssertionError(f"pipeline did not pass {flag}={value}: {command}")
    science_config = final_pipeline.config_for_interaction_scripts(config, tmp / "run" / "cascade")
    provenance = json.loads(science_config.read_text(encoding="utf-8"))["provenance"]
    if provenance.get("photon_camera_physical_interpretation") != "photon_escape_classifier":
        raise AssertionError(f"pipeline provenance missing photon interpretation: {provenance}")
    if provenance.get("photon_projected_to_pixels") is not False:
        raise AssertionError(f"pipeline provenance must record projected_to_pixels=false: {provenance}")


def test_pipeline_runs_phase1_then_phase2_for_observer_sphere_hits(tmp: Path) -> None:
    config = {
        "run_name": "PhotonSphereHitsPipeline",
        "output_dir": str(tmp / "run"),
        "physics_mode": "uhe_cascade",
        "black_hole_mass_msun": 3.0,
        "spin": 0.0,
        "camera_nx": 3,
        "camera_ny": 3,
        "camera_fov_deg": 60.0,
        "camera_theta_deg": 70.0,
        "camera_r_obs_rg": 20.0,
        "camera_r_max_rg": 40.0,
        "camera_step": 0.05,
        "neutrino_energy_gev": 1.0e9,
        "n_events": 1,
        "seed": 1,
        "generate_standard_scientific_plots": False,
        "generate_dashboard": False,
        "enable_photon_observer_camera": True,
        "photon_observer_mode": "observer_sphere_hits",
        "photon_observer_frame": "ZAMO",
        "photon_null_norm_tolerance": 1.0e-8,
        "photon_invariant_tolerance": 1.0e-6,
        "photon_horizon_crossing_tolerance_rg": 1.0e-7,
        "photon_fail_on_invariant_violation": True,
        "photon_max_geodesic_steps": 1234,
        "photon_geodesic_step_rg": 0.03,
        "photon_min_energy_gev": 2.0,
        "photon_camera_output_mode": "summary_only",
        "photon_redshift_mode": "disabled_until_validated",
    }
    steps = final_pipeline.build_steps(config, ROOT / "presets/config_web/final_pipeline_config.json")
    names = [step.name for step in steps]
    if "photon_escape_classifier" not in names or "photon_observer_sphere_hit_map" not in names:
        raise AssertionError(f"pipeline did not include Phase 1 and Phase 2: {names}")
    if names.index("photon_escape_classifier") > names.index("photon_observer_sphere_hit_map"):
        raise AssertionError(f"Phase 2 was not scheduled after Phase 1: {names}")
    phase2 = next(step for step in steps if step.name == "photon_observer_sphere_hit_map")
    command_text = " ".join(phase2.command)
    for expected in [
        "photon_escape_classifier.jsonl",
        "photon_observer_sphere_hits.jsonl",
        "photon_observer_sphere_summary.csv",
        "photon_observer_sphere_provenance.json",
    ]:
        if expected not in command_text:
            raise AssertionError(f"Phase 2 command missing {expected}: {phase2.command}")
    science_config = final_pipeline.config_for_interaction_scripts(config, tmp / "run" / "cascade")
    provenance = json.loads(science_config.read_text(encoding="utf-8"))["provenance"]
    if provenance.get("photon_observer_sphere_hit_map_enabled_effective") is not True:
        raise AssertionError(f"pipeline provenance missing Phase 2 enabled flag: {provenance}")
    if provenance.get("photon_observer_sphere_hits_camera_aperture") is not False:
        raise AssertionError(f"pipeline provenance must record no camera aperture hit: {provenance}")


def test_wrapper_has_no_physical_defaults() -> None:
    text = (ROOT / "scripts/science/run_kerr_photon_escape_classifier.py").read_text(encoding="utf-8")
    forbidden = [
        'parser.add_argument("--spin", default=',
        'parser.add_argument("--observer-radius-rg", default=',
        'parser.add_argument("--max-radius-rg", default=',
        'parser.add_argument("--photon-geodesic-step-rg", default=',
        'parser.add_argument("--photon-max-geodesic-steps", default=',
        'parser.add_argument("--photon-null-norm-tolerance", default=',
        'parser.add_argument("--photon-invariant-tolerance", default=',
        'parser.add_argument("--photon-horizon-crossing-tolerance-rg", default=',
        'parser.add_argument("--photon-min-energy-gev", default=',
        'default="ZAMO"',
        'default="true"',
    ]
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"hidden wrapper physical default remains: {needle}")


def test_pipeline_has_no_photon_physical_defaults() -> None:
    text = (ROOT / "scripts/run_hadros_final_pipeline.py").read_text(encoding="utf-8")
    forbidden = [
        'config.get("photon_observer_mode"',
        'config.get("photon_observer_frame"',
        'config.get("photon_null_norm_tolerance"',
        'config.get("photon_invariant_tolerance"',
        'config.get("photon_horizon_crossing_tolerance_rg"',
        'config.get("photon_fail_on_invariant_violation"',
        'config.get("photon_max_geodesic_steps"',
        'config.get("photon_geodesic_step_rg"',
        'config.get("photon_min_energy_gev"',
        'config.get("photon_camera_output_mode"',
        'config.get("photon_redshift_mode"',
        'photon.get("photon_observer_mode"',
        'photon.get("photon_observer_frame"',
        'photon.get("photon_null_norm_tolerance"',
        'photon.get("photon_invariant_tolerance"',
        'photon.get("photon_horizon_crossing_tolerance_rg"',
        'photon.get("photon_fail_on_invariant_violation"',
        'photon.get("photon_max_geodesic_steps"',
        'photon.get("photon_geodesic_step_rg"',
        'photon.get("photon_min_energy_gev"',
        'photon.get("photon_camera_output_mode"',
        'photon.get("photon_redshift_mode"',
    ]
    for needle in forbidden:
        if needle in text:
            raise AssertionError(f"hidden pipeline photon default remains: {needle}")


def test_provenance_contains_phase1_limitations(binary: Path, tmp: Path) -> None:
    _, _, prov = run_classifier(tmp, binary, [photon_row()])
    expected = {
        "camera_physical_interpretation": "photon_escape_classifier",
        "camera_is_full_observational_transport": False,
        "projected_to_pixels": False,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise AssertionError(f"provenance[{key!r}]={prov.get(key)!r}, expected {value!r}")
    if prov.get("momentum_input_mode") != "per_record_required":
        raise AssertionError(f"provenance missing momentum_input_mode policy: {prov}")
    if float(prov.get("photon_horizon_crossing_tolerance_rg", -1.0)) < 0.0:
        raise AssertionError(f"provenance missing horizon tolerance: {prov}")


def phase1_hit_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": 7,
        "particle_id": 3,
        "pdg": 22,
        "input_energy_gev": 10.0,
        "classification": "reaches_observer_sphere",
        "observer_crossing_r_rg": 20.0,
        "observer_crossing_theta_rad": 1.2,
        "observer_crossing_phi_rad": 0.4,
        "observer_crossing_interpolated": True,
        "geodesic_steps": 42,
        "E_killing_initial": 0.9,
        "E_killing_final": 0.900000001,
        "Lz_initial": 0.1,
        "Lz_final": 0.100000001,
        "null_norm_max_abs_error": 1.0e-12,
        "relative_E_error": 1.0e-9,
        "relative_Lz_error": 1.0e-8,
        "momentum_input_mode": "zamo_tetrad",
    }
    row.update(updates)
    return row


def run_phase2_hits(tmp: Path, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object], str]:
    tmp.mkdir(parents=True, exist_ok=True)
    input_path = tmp / "photon_escape_classifier.jsonl"
    output_jsonl = tmp / "photon_observer_sphere_hits.jsonl"
    summary_csv = tmp / "photon_observer_sphere_summary.csv"
    summary_md = tmp / "photon_observer_sphere_summary.md"
    provenance = tmp / "photon_observer_sphere_provenance.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    args = type("Args", (), {})()
    args.input = input_path
    args.output_jsonl = output_jsonl
    args.summary_csv = summary_csv
    args.summary_md = summary_md
    args.provenance = provenance
    phase1_rows = phase2_hits.read_jsonl(args.input)
    hits = [phase2_hits.hit_from_row(row) for row in phase1_rows if row.get("classification") == "reaches_observer_sphere"]
    summary = phase2_hits.build_summary(phase1_rows, hits)
    phase2_hits.write_jsonl(output_jsonl, hits)
    phase2_hits.write_summary_csv(summary_csv, summary)
    phase2_hits.write_summary_md(summary_md, summary)
    phase2_hits.write_provenance(provenance, args, summary)
    out_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_row = next(csv.DictReader(handle))
    prov = json.loads(provenance.read_text(encoding="utf-8"))
    return out_rows, summary_row, prov, summary_md.read_text(encoding="utf-8")


def test_phase2_excludes_non_observer_sphere_classifications(tmp: Path) -> None:
    rows = [
        phase1_hit_row(event_id=1),
        phase1_hit_row(event_id=2, classification="captured_by_black_hole"),
        phase1_hit_row(event_id=3, classification="escapes_but_misses_observer"),
    ]
    hits, summary, _, _ = run_phase2_hits(tmp, rows)
    if [hit["event_id"] for hit in hits] != [1]:
        raise AssertionError(f"Phase 2 did not filter classifications correctly: {hits}")
    if summary["n_input_particles"] != "3" or summary["n_reached_observer_sphere"] != "1":
        raise AssertionError(f"bad Phase 2 filtering summary: {summary}")


def test_phase2_uses_interpolated_crossing_coordinates(tmp: Path) -> None:
    row = phase1_hit_row(observer_crossing_theta_rad=1.2345, observer_crossing_phi_rad=2.3456, geodesic_steps=77)
    hits, _, _, _ = run_phase2_hits(tmp, [row])
    hit = hits[0]
    if hit["observer_crossing_interpolated"] is not True:
        raise AssertionError(f"Phase 2 lost interpolation flag: {hit}")
    if hit["observer_crossing_theta_rad"] != 1.2345 or hit["observer_crossing_phi_rad"] != 2.3456:
        raise AssertionError(f"Phase 2 did not preserve crossing coordinates: {hit}")
    if hit["crossing_step_index"] != 77:
        raise AssertionError(f"Phase 2 did not preserve crossing step index: {hit}")


def test_phase2_summary_accumulates_reached_energy(tmp: Path) -> None:
    rows = [
        phase1_hit_row(event_id=1, input_energy_gev=10.0),
        phase1_hit_row(event_id=2, input_energy_gev=5.0, observer_crossing_theta_rad=1.4, observer_crossing_phi_rad=0.8),
        phase1_hit_row(event_id=3, input_energy_gev=100.0, classification="captured_by_black_hole"),
    ]
    _, summary, _, _ = run_phase2_hits(tmp, rows)
    if summary["n_reached_observer_sphere"] != "2":
        raise AssertionError(f"bad Phase 2 reached count: {summary}")
    if abs(float(summary["total_input_energy_reached_observer_sphere_gev"]) - 15.0) > 1.0e-12:
        raise AssertionError(f"bad Phase 2 total energy: {summary}")
    if abs(float(summary["mean_input_energy_reached_observer_sphere_gev"]) - 7.5) > 1.0e-12:
        raise AssertionError(f"bad Phase 2 mean energy: {summary}")


def test_phase2_outputs_avoid_pixel_detector_and_observed_energy_fields(tmp: Path) -> None:
    hits, _, _, summary_md = run_phase2_hits(tmp, [phase1_hit_row()])
    serialized_hits = json.dumps(hits, sort_keys=True)
    for forbidden in ["pixel_x", "pixel_y", "observed_energy_gev", "detector", "aperture"]:
        if forbidden in serialized_hits:
            raise AssertionError(f"Phase 2 hit output contains forbidden field {forbidden}: {serialized_hits}")
    for forbidden in ["pixel_x", "pixel_y", "observed_energy_gev"]:
        if forbidden in summary_md:
            raise AssertionError(f"Phase 2 summary contains forbidden field {forbidden}: {summary_md}")


def test_phase2_provenance_records_limitations(tmp: Path) -> None:
    _, _, prov, _ = run_phase2_hits(tmp, [phase1_hit_row()])
    expected = {
        "phase": "photon_observer_sphere_hit_map",
        "photon_observer_mode": "observer_sphere_hits",
        "projected_to_pixels": False,
        "hits_camera_aperture": False,
        "observer_sphere_crossing_is_detection": False,
        "observed_energy_available": False,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise AssertionError(f"Phase 2 provenance[{key!r}]={prov.get(key)!r}, expected {value!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_escape_tests_") as tmp_name:
        base = Path(tmp_name)
        binary = compile_backend(base)
        tests = [
            lambda: test_pdg_filtering(binary, base / "filter"),
            lambda: test_radial_outward_reaches_observer(binary, base / "outward"),
            lambda: test_radial_inward_captured(binary, base / "inward"),
            lambda: test_ambiguous_generic_momentum_rejected(binary, base / "ambiguous"),
            lambda: test_global_boyer_lindquist_mode_accepted(binary, base / "global_bl"),
            lambda: test_invalid_null_momentum(binary, base / "invalid_null"),
            lambda: test_invariant_violation_can_fail(binary, base / "invariant"),
            lambda: test_negative_tolerances_rejected(binary, base / "negative_tolerance"),
            test_config_web_contains_all_parameters,
            lambda: test_pipeline_passes_all_parameters(base / "pipeline"),
            lambda: test_pipeline_runs_phase1_then_phase2_for_observer_sphere_hits(base / "pipeline_phase2"),
            test_wrapper_has_no_physical_defaults,
            test_pipeline_has_no_photon_physical_defaults,
            lambda: test_provenance_contains_phase1_limitations(binary, base / "provenance"),
            lambda: test_phase2_excludes_non_observer_sphere_classifications(base / "phase2_filter"),
            lambda: test_phase2_uses_interpolated_crossing_coordinates(base / "phase2_crossing"),
            lambda: test_phase2_summary_accumulates_reached_energy(base / "phase2_summary"),
            lambda: test_phase2_outputs_avoid_pixel_detector_and_observed_energy_fields(base / "phase2_forbidden"),
            lambda: test_phase2_provenance_records_limitations(base / "phase2_provenance"),
        ]
        for test in tests:
            test()
            name = getattr(test, "__name__", "lambda_test")
            print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
