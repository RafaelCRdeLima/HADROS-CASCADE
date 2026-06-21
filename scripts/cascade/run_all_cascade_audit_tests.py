#!/usr/bin/env python3
"""Run the HADROS-CASCADE audit test suite through Phase 2.5."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


TESTS = [
    {
        "name": "phase0_cpp_analytic_and_optional_stubs",
        "phase": "0",
        "command": ["make", "cascade-tests"],
        "requires": "C++ compiler",
    },
    {
        "name": "phase0_5_analytic_pipeline",
        "phase": "0.5",
        "command": ["make", "cascade-pipeline-test"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase1_5_particle_agnostic_schema",
        "phase": "1.5",
        "command": ["make", "cascade-particle-schema-test"],
        "requires": "Python",
    },
    {
        "name": "phase2_5_kinetic_energy_conversion",
        "phase": "2.5",
        "command": [sys.executable, "tests/cascade/test_geant4_kinetic_energy_conversion.py"],
        "requires": "Python",
    },
    {
        "name": "phase1_pythia_proxy_optional",
        "phase": "1",
        "command": [sys.executable, "tests/cascade/test_pythia_proxy_optional.py"],
        "requires": "PYTHIA optional",
    },
    {
        "name": "phase2_geant4_local_box_optional",
        "phase": "2",
        "command": [sys.executable, "tests/cascade/test_geant4_local_box_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "geant4_real_crash_debug_optional",
        "phase": "2 debug",
        "command": [sys.executable, "tests/cascade/test_geant4_real_crash_debug.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "geant4_safety_filter_optional",
        "phase": "2 debug",
        "command": [sys.executable, "tests/cascade/test_geant4_safety_filter.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase2_geant4_smoke_optional",
        "phase": "2 diagnostic",
        "command": [sys.executable, "tests/cascade/test_geant4_smoke_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase2_1_geant4_robustness_quick_optional",
        "phase": "2.1",
        "command": [sys.executable, "tests/cascade/test_geant4_robustness_quick_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase2_2_geant4_material_benchmarks_optional",
        "phase": "2.2",
        "command": [sys.executable, "tests/cascade/test_geant4_material_benchmarks_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase2_3_geant4_scaling_quick_optional",
        "phase": "2.3",
        "command": [sys.executable, "tests/cascade/test_geant4_scaling_quick_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase2_4_pythia_to_geant4_chain_optional",
        "phase": "2.4",
        "command": [sys.executable, "tests/cascade/test_pythia_to_geant4_chain_optional.py"],
        "requires": "PYTHIA and GEANT4 optional",
    },
    {
        "name": "phase2_5_pythia_to_geant4_energy_convention_optional",
        "phase": "2.5",
        "command": [sys.executable, "tests/cascade/test_pythia_to_geant4_energy_convention_optional.py"],
        "requires": "PYTHIA and GEANT4 optional",
    },
    {
        "name": "phase3_local_response_table_optional",
        "phase": "3",
        "command": [sys.executable, "tests/cascade/test_local_response_table_optional.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase3_1_local_response_reader",
        "phase": "3.1",
        "command": [sys.executable, "tests/cascade/test_local_response_table_reader.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase3_2_apply_local_response_to_events",
        "phase": "3.2",
        "command": [sys.executable, "tests/cascade/test_apply_local_response_to_events.py"],
        "requires": "Python",
    },
    {
        "name": "phase3_3_response_table_coverage_optional",
        "phase": "3.3",
        "command": [sys.executable, "tests/cascade/test_response_table_coverage_optional.py"],
        "requires": "Python, GEANT4 optional",
    },
    {
        "name": "phase3_4_local_response_quality",
        "phase": "3.4",
        "command": [sys.executable, "tests/cascade/test_local_response_table_quality.py"],
        "requires": "Python",
    },
    {
        "name": "phase3_4_local_response_refinement_optional",
        "phase": "3.4",
        "command": [sys.executable, "tests/cascade/test_local_response_refinement_optional.py"],
        "requires": "Python, GEANT4 optional",
    },
    {
        "name": "phase4_deposition_emissivity_proxy",
        "phase": "4",
        "command": [sys.executable, "tests/cascade/test_deposition_emissivity_proxy.py"],
        "requires": "Python",
    },
    {
        "name": "phase4_1_deposition_emissivity_export",
        "phase": "4.1",
        "command": [sys.executable, "tests/cascade/test_export_deposition_emissivity_proxy.py"],
        "requires": "Python, h5py",
    },
    {
        "name": "phase4_2_deposition_emissivity_field_reader",
        "phase": "4.2",
        "command": [sys.executable, "tests/cascade/test_deposition_emissivity_field_reader.py"],
        "requires": "C++ compiler, HDF5 optional",
    },
    {
        "name": "phase4_3_deposition_proxy_camera_mode",
        "phase": "4.3",
        "command": [sys.executable, "tests/cascade/test_deposition_proxy_camera_mode.py"],
        "requires": "C++ compiler, HDF5 optional",
    },
    {
        "name": "phase4_4_deposition_camera_overlap_diagnostic",
        "phase": "4.4",
        "command": [sys.executable, "tests/cascade/test_deposition_camera_overlap_diagnostic.py"],
        "requires": "Python, h5py",
    },
    {
        "name": "phase5_escaping_particle_packets",
        "phase": "5",
        "command": [sys.executable, "tests/cascade/test_escaping_particle_packets.py"],
        "requires": "Python",
    },
    {
        "name": "phase5_1_escaping_packet_classification",
        "phase": "5.1",
        "command": [sys.executable, "tests/cascade/test_escaping_packet_classification.py"],
        "requires": "Python",
    },
    {
        "name": "phase5_2_null_packet_propagation",
        "phase": "5.2",
        "command": [sys.executable, "tests/cascade/test_null_packet_propagation.py"],
        "requires": "Python",
    },
    {
        "name": "phase5_3_packet_observer_overlap",
        "phase": "5.3",
        "command": [sys.executable, "tests/cascade/test_packet_observer_overlap_diagnostic.py"],
        "requires": "Python",
    },
    {
        "name": "phase5_4_packet_observer_scan",
        "phase": "5.4",
        "command": [sys.executable, "tests/cascade/test_packet_observer_scan.py"],
        "requires": "Python",
    },
    {
        "name": "phase5_5_phase5_audit_summary",
        "phase": "5.5",
        "command": [sys.executable, "tests/cascade/test_phase5_audit_summary.py"],
        "requires": "Python",
    },
    {
        "name": "phase6_0_kerr_null_packet_propagation",
        "phase": "6.0",
        "command": [sys.executable, "tests/cascade/test_kerr_null_packet_propagation.py"],
        "requires": "Python",
    },
    {
        "name": "phase6_1_kerr_tetrad_initialization",
        "phase": "6.1",
        "command": [sys.executable, "tests/cascade/test_kerr_tetrad_initialization.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase6_1_kerr_null_packet_zamo_mode",
        "phase": "6.1",
        "command": [sys.executable, "tests/cascade/test_kerr_null_packet_zamo_mode.py"],
        "requires": "Python",
    },
    {
        "name": "phase6_2_kerr_packet_observer_scan",
        "phase": "6.2",
        "command": [sys.executable, "tests/cascade/test_kerr_packet_observer_scan.py"],
        "requires": "Python",
    },
    {
        "name": "phase7_0_particle_channel_images",
        "phase": "7.0",
        "command": [sys.executable, "tests/cascade/test_particle_channel_images.py"],
        "requires": "Python",
    },
    {
        "name": "phase7_1_particle_channel_image_audit",
        "phase": "7.1",
        "command": [sys.executable, "tests/cascade/test_particle_channel_image_audit.py"],
        "requires": "Python",
    },
    {
        "name": "phase9_1_real_kerr_packet_app",
        "phase": "9.1",
        "command": [sys.executable, "tests/cascade/test_real_kerr_packet_app.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase9_1_packet_image_backend_routing",
        "phase": "9.1",
        "command": [sys.executable, "tests/cascade/test_packet_image_backend_routing.py"],
        "requires": "Python",
    },
    {
        "name": "phase9_1_proxy_vs_real_kerr_images",
        "phase": "9.1",
        "command": [sys.executable, "tests/cascade/test_proxy_vs_real_kerr_packet_images.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase9_1_real_kerr_packet_physics_audit",
        "phase": "9.1",
        "command": [sys.executable, "tests/cascade/test_real_kerr_packet_physics_audit.py"],
        "requires": "Python",
    },
    {
        "name": "phase9_1_no_hidden_physical_labels",
        "phase": "9.1",
        "command": [sys.executable, "tests/cascade/test_no_hidden_physical_labels.py"],
        "requires": "Python",
    },
    {
        "name": "phase10_scientific_status_report",
        "phase": "10.0",
        "command": [sys.executable, "tests/cascade/test_scientific_status_report.py"],
        "requires": "Python",
    },
    {
        "name": "phase11_gbw_iim_real_kerr_study",
        "phase": "11.0",
        "command": [sys.executable, "tests/science/test_gbw_iim_real_kerr_study.py"],
        "requires": "Python",
    },
    {
        "name": "phase11_1_dis_model_trace_audit",
        "phase": "11.1",
        "command": [sys.executable, "tests/science/test_dis_model_trace_audit.py"],
        "requires": "Python",
    },
    {
        "name": "phase11_2_dis_event_weights",
        "phase": "11.2",
        "command": [sys.executable, "tests/science/test_dis_event_weights.py"],
        "requires": "Python",
    },
    {
        "name": "phase11_2_dis_weighted_packets",
        "phase": "11.2",
        "command": [sys.executable, "tests/science/test_dis_weighted_packet_propagation.py"],
        "requires": "Python",
    },
    {
        "name": "phase11_2_dis_weighted_gbw_iim",
        "phase": "11.2",
        "command": [sys.executable, "tests/science/test_dis_weighted_gbw_iim_comparison.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase11_3_dis_reweighting_physics_validation",
        "phase": "11.3",
        "command": [sys.executable, "tests/science/test_dis_reweighting_physics_validation.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_cascade_config",
        "phase": "config-web",
        "command": [sys.executable, "tests/cascade/test_config_web_cascade_config.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_cascade_pipeline_dryrun",
        "phase": "config-web",
        "command": [sys.executable, "tests/cascade/test_config_web_cascade_pipeline_dryrun.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_cascade_micromamba_env",
        "phase": "config-web",
        "command": [sys.executable, "tests/cascade/test_config_web_cascade_micromamba_env.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_geant4_real_safe_mode",
        "phase": "config-web",
        "command": [sys.executable, "tests/cascade/test_config_web_geant4_real_safe_mode.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_geant4_modes",
        "phase": "config-web",
        "command": [sys.executable, "tests/cascade/test_config_web_geant4_modes.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_run_tab_physical_modes",
        "phase": "config-web",
        "command": [sys.executable, "tests/config_web/test_run_tab_physical_modes.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_run_manifest_channels",
        "phase": "config-web",
        "command": [sys.executable, "tests/config_web/test_run_manifest_channels.py"],
        "requires": "Python",
    },
    {
        "name": "config_web_dashboard_channel_sections",
        "phase": "config-web",
        "command": [sys.executable, "tests/config_web/test_dashboard_channel_sections.py"],
        "requires": "Python",
    },
    {
        "name": "phase12_3_config_web_particle_autoframe",
        "phase": "12.3",
        "command": [sys.executable, "tests/science/test_config_web_particle_autoframe.py"],
        "requires": "Python",
    },
    {
        "name": "phase13_0_backward_camera_design",
        "phase": "13.0",
        "command": [sys.executable, "tests/science/test_backward_camera_particle_design.py"],
        "requires": "Python",
    },
    {
        "name": "phase13_1_backward_camera_particle_image",
        "phase": "13.1",
        "command": [sys.executable, "tests/science/test_backward_camera_particle_image.py"],
        "requires": "C++ compiler, Python, matplotlib",
    },
    {
        "name": "phase13_1_backward_camera_geometry_validation",
        "phase": "13.1",
        "command": [sys.executable, "tests/science/test_backward_camera_geometry_validation.py"],
        "requires": "C++ compiler, Python, matplotlib",
    },
    {
        "name": "phase13_1_backward_camera_axisymmetry",
        "phase": "13.1",
        "command": [sys.executable, "tests/science/test_backward_camera_axisymmetry.py"],
        "requires": "C++ compiler, Python, matplotlib",
    },
    {
        "name": "hard_reset_hadros_backward_camera_route",
        "phase": "camera-hard-reset",
        "command": [sys.executable, "tests/science/test_hadros_backward_camera_hard_reset.py"],
        "requires": "Python",
    },
    {
        "name": "phase13_2_hadros_backward_camera_cuda_validation",
        "phase": "13.2",
        "command": [sys.executable, "tests/science/test_hadros_backward_camera_cuda_validation.py"],
        "requires": "CUDA optional, Python",
    },
    {
        "name": "stream_image_argument_order",
        "phase": "stream-image",
        "command": [sys.executable, "tests/science/test_stream_image_argument_order.py"],
        "requires": "C++ compiler, Python",
    },
    {
        "name": "phase8_geant4_real_safe_audit",
        "phase": "8.0",
        "command": [sys.executable, "tests/cascade/test_geant4_real_safe_audit.py"],
        "requires": "Python, matplotlib",
    },
    {
        "name": "phase8_1_uhe_transport_policy",
        "phase": "8.1",
        "command": [sys.executable, "tests/cascade/test_uhe_transport_policy.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase8_1_escaping_packets_include_unsupported_uhe",
        "phase": "8.1",
        "command": [sys.executable, "tests/cascade/test_escaping_packets_include_unsupported_uhe.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_1_config_web_uhe_policy",
        "phase": "8.1",
        "command": [sys.executable, "tests/cascade/test_config_web_uhe_policy.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_2_geant4_resumable_batches",
        "phase": "8.2",
        "command": [sys.executable, "tests/cascade/test_geant4_resumable_batches.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "phase8_2_config_web_geant4_batch_mode",
        "phase": "8.2",
        "command": [sys.executable, "tests/cascade/test_config_web_geant4_batch_mode.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_2_geant4_batch_energy_target",
        "phase": "8.2",
        "command": [sys.executable, "tests/cascade/test_geant4_batch_energy_target.py"],
        "requires": "GEANT4 optional",
    },
    {
        "name": "kerr_packet_failed_integration_debug",
        "phase": "8.2-debug",
        "command": [sys.executable, "tests/cascade/test_kerr_packet_failed_integration_debug.py"],
        "requires": "Python",
    },
    {
        "name": "kerr_null_momentum_normalization",
        "phase": "8.2-debug",
        "command": [sys.executable, "tests/cascade/test_kerr_null_momentum_normalization.py"],
        "requires": "Python",
    },
    {
        "name": "packet_origin_propagation",
        "phase": "origin-safety",
        "command": [sys.executable, "tests/cascade/test_packet_origin_propagation.py"],
        "requires": "Python",
    },
    {
        "name": "kerr_skips_missing_or_inside_horizon",
        "phase": "origin-safety",
        "command": [sys.executable, "tests/cascade/test_kerr_skips_missing_or_inside_horizon.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_3_packet_origin_validation_audit",
        "phase": "8.3",
        "command": [sys.executable, "tests/cascade/test_packet_origin_validation_audit.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_4_energy_fraction_convergence",
        "phase": "8.4",
        "command": [sys.executable, "tests/cascade/test_energy_fraction_convergence.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_5_stratified_channel_convergence",
        "phase": "8.5",
        "command": [sys.executable, "tests/cascade/test_stratified_channel_convergence.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_6_angular_packetization",
        "phase": "8.6",
        "command": [sys.executable, "tests/cascade/test_angular_packetization.py"],
        "requires": "Python",
    },
    {
        "name": "phase8_6_packetization_mode_comparison",
        "phase": "8.6",
        "command": [sys.executable, "tests/cascade/test_packetization_mode_comparison.py"],
        "requires": "Python",
    },
    {
        "name": "packet_raytracing_path_audit",
        "phase": "ray-path-audit",
        "command": [sys.executable, "tests/cascade/test_packet_raytracing_path_audit.py"],
        "requires": "Python",
    },
    {
        "name": "phase9_0_packet_real_kerr_vs_straight",
        "phase": "9.0",
        "command": [sys.executable, "tests/cascade/test_packet_real_kerr_vs_straight.py"],
        "requires": "C++ compiler, Python",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def classify(returncode: int, stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode == 0 and "skipped:" in combined:
        return "SKIP"
    if returncode == 0:
        return "PASS"
    return "FAIL"


def tail(text: str, max_lines: int = 18) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def run_test(root: Path, item: dict[str, object], timeout: float) -> dict[str, object]:
    command = list(item["command"])  # type: ignore[index]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = -999
        stderr = (stderr + f"\nTIMEOUT after {timeout} s").strip()
    else:
        elapsed = time.monotonic() - started

    status = "TIMEOUT" if returncode == -999 else classify(returncode, stdout, stderr)
    return {
        "name": item["name"],
        "phase": item["phase"],
        "requires": item["requires"],
        "command": shell_join(command),
        "status": status,
        "returncode": returncode,
        "elapsed_s": elapsed,
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
    }


def write_report(path: Path, results: list[dict[str, object]], timeout: float) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    counts = {status: sum(row["status"] == status for row in results) for status in ["PASS", "SKIP", "FAIL", "TIMEOUT"]}

    lines = [
        "# HADROS-CASCADE Audit Test Summary",
        "",
        f"- generated_utc: `{now}`",
        f"- timeout_per_test_s: `{timeout:g}`",
        f"- total: `{len(results)}`",
        f"- PASS: `{counts['PASS']}`",
        f"- SKIP: `{counts['SKIP']}`",
        f"- FAIL: `{counts['FAIL']}`",
        f"- TIMEOUT: `{counts['TIMEOUT']}`",
        "",
        "| Phase | Test | Requires | Status | Return code | Runtime [s] |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['phase']} | `{row['name']}` | {row['requires']} | "
            f"{row['status']} | `{row['returncode']}` | {float(row['elapsed_s']):.2f} |"
        )

    lines.extend(["", "## Commands", ""])
    for row in results:
        lines.append(f"- `{row['name']}`: `{row['command']}`")

    lines.extend(["", "## Failure/Skip Details", ""])
    for row in results:
        if row["status"] not in {"FAIL", "TIMEOUT", "SKIP"}:
            continue
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- status: `{row['status']}`")
        lines.append(f"- returncode: `{row['returncode']}`")
        if row["stdout_tail"]:
            lines.append("")
            lines.append("stdout tail:")
            lines.append("")
            lines.append("```text")
            lines.append(str(row["stdout_tail"]))
            lines.append("```")
        if row["stderr_tail"]:
            lines.append("")
            lines.append("stderr tail:")
            lines.append("")
            lines.append("```text")
            lines.append(str(row["stderr_tail"]))
            lines.append("```")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cascade/cascade_audit_test_summary.md"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output

    results: list[dict[str, object]] = []
    for item in TESTS:
        print(f"[audit] {item['phase']} {item['name']}")
        result = run_test(root, item, args.timeout)
        results.append(result)
        print(f"[audit] -> {result['status']} ({float(result['elapsed_s']):.2f} s)")
        if result["status"] in {"FAIL", "TIMEOUT"} and not args.keep_going:
            break

    write_report(output, results, args.timeout)
    print(f"audit_summary={output}")

    return 1 if any(row["status"] in {"FAIL", "TIMEOUT"} for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
