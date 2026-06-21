#!/usr/bin/env python3
"""Audit whether escaping packets use real HADROS Kerr geodesics or a proxy path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


AUDIT_ROWS = [
    {
        "component": "packet python propagator",
        "file": "scripts/cascade/propagate_kerr_null_packets.py",
        "function_or_class": "propagate_one",
        "role": "Propagates selected escaping packets for Phase 6.x diagnostics.",
        "uses_hadros_null_geodesic_integrator": "no",
        "uses_kerr_metric": "yes",
        "uses_straight_line": "yes",
        "status": "PROXY_STRAIGHT_LINE",
        "notes": "Uses ray_sphere_distance, hits_horizon, and observer_pixel. ZAMO tetrad diagnostics compute p^mu/null norm, but p^mu is not passed to KerrGeodesic.",
    },
    {
        "component": "packet C++ propagator",
        "file": "src/cascade/packet_kerr_null_propagator.cpp",
        "function_or_class": "PacketKerrNullPropagator::propagate",
        "role": "C++ packet propagation backend connected in Phase 9.0.",
        "uses_hadros_null_geodesic_integrator": "yes",
        "uses_kerr_metric": "yes",
        "uses_straight_line": "no",
        "status": "REAL_HADROS_KERR_GEODESIC",
        "notes": "Initializes a ZAMO null packet and advances GeodesicState with KerrGeodesic::step_adaptive.",
    },
    {
        "component": "packet C++ interface",
        "file": "include/hadros/cascade/packet_kerr_null_propagator.hpp",
        "function_or_class": "PacketKerrNullPropagator",
        "role": "Configuration/result API for packet propagation.",
        "uses_hadros_null_geodesic_integrator": "no",
        "uses_kerr_metric": "no",
        "uses_straight_line": "no",
        "status": "API_ONLY",
        "notes": "Configuration/result API for the C++ real-Kerr packet backend. Python channel images do not use it yet.",
    },
    {
        "component": "HADROS camera geodesic integrator",
        "file": "src/kerr_camera.cpp",
        "function_or_class": "KerrCamera::trace_pixel",
        "role": "Main HADROS camera ray tracer.",
        "uses_hadros_null_geodesic_integrator": "yes",
        "uses_kerr_metric": "yes",
        "uses_straight_line": "no",
        "status": "REAL_HADROS_GEODESIC",
        "notes": "Builds a GeodesicState and calls KerrGeodesic::step_adaptive in a loop.",
    },
    {
        "component": "HADROS null geodesic stepper",
        "file": "include/kerr_geodesic.hpp",
        "function_or_class": "KerrGeodesic::step_adaptive",
        "role": "Hamiltonian null-geodesic integrator used by HADROS camera/cache paths.",
        "uses_hadros_null_geodesic_integrator": "yes",
        "uses_kerr_metric": "yes",
        "uses_straight_line": "no",
        "status": "REAL_HADROS_GEODESIC",
        "notes": "Not called by current packet propagation scripts/classes.",
    },
    {
        "component": "deposition proxy camera",
        "file": "apps/compute_deposition_proxy_camera.cpp",
        "function_or_class": "KerrCamera::trace_pixel",
        "role": "Camera for deposition emissivity proxy field.",
        "uses_hadros_null_geodesic_integrator": "yes",
        "uses_kerr_metric": "yes",
        "uses_straight_line": "no",
        "status": "REAL_HADROS_GEODESIC_FOR_DEPOSITION_FIELD",
        "notes": "This is a different Phase 4.x path; it does not propagate escaping packets.",
    },
    {
        "component": "particle-channel image builder",
        "file": "scripts/cascade/build_particle_channel_images.py",
        "function_or_class": "project_direction",
        "role": "Maps packet directions to diagnostic image pixels.",
        "uses_hadros_null_geodesic_integrator": "no",
        "uses_kerr_metric": "no",
        "uses_straight_line": "yes",
        "status": "ANGULAR_PROXY_IMAGE",
        "notes": "Projects normalized packet momentum into an observer cone using camera_basis/project_direction.",
    },
    {
        "component": "config-web cascade orchestration",
        "file": "scripts/config_web.py",
        "function_or_class": "cascade diagnostics pipeline",
        "role": "Exposes kerr_init_mode and packet diagnostic switches.",
        "uses_hadros_null_geodesic_integrator": "no",
        "uses_kerr_metric": "indirect",
        "uses_straight_line": "yes",
        "status": "CONFIGURES_PROXY_PACKET_PATH",
        "notes": "The zamo_tetrad option reaches propagate_kerr_null_packets.py, not KerrCamera/KerrGeodesic.",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def run_packet_case(tmp: Path, packets: list[dict[str, Any]], label: str) -> list[dict[str, str]]:
    packets_path = tmp / f"{label}_packets.jsonl"
    classes_path = tmp / f"{label}_classes.csv"
    out_dir = tmp / label
    out_dir.mkdir()
    write_jsonl(packets_path, packets)
    classes_path.write_text(
        "event_id,pdg_id,classification,energy_gev,weighted_energy_gev\n"
        + "\n".join(
            f"{row['event_id']},{row['pdg_id']},MASSLESS_NULL,{row['energy_gev']},{row.get('weighted_energy_gev', row['energy_gev'])}"
            for row in packets
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/cascade/propagate_kerr_null_packets.py"),
            "--packets",
            str(packets_path),
            "--classification",
            str(classes_path),
            "--output-dir",
            str(out_dir),
            "--kerr-init-mode",
            "zamo_tetrad",
            "--observer-axis",
            "+z",
            "--fov-deg",
            "120",
            "--nx",
            "16",
            "--ny",
            "16",
            "--domain-radius",
            "100",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return read_csv(out_dir / "kerr_null_propagated_packets_zamo.csv")


def physical_comparison_rows() -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="hadros-packet-raypath-") as tmp_name:
        tmp = Path(tmp_name)
        same_origin = [
            {
                "event_id": 1, "pdg_id": 22, "energy_gev": 1.0, "weighted_energy_gev": 1.0,
                "px_gev": 0.0, "py_gev": 0.0, "pz_gev": 1.0,
                "x": 10.0, "y": 0.0, "z": 0.0, "r": 10.0, "theta": math.pi / 2, "phi": 0.0,
                "origin_status": "SYNTHETIC_TEST_POSITION",
            },
            {
                "event_id": 2, "pdg_id": 22, "energy_gev": 1.0, "weighted_energy_gev": 1.0,
                "px_gev": 0.4, "py_gev": 0.0, "pz_gev": 1.0,
                "x": 10.0, "y": 0.0, "z": 0.0, "r": 10.0, "theta": math.pi / 2, "phi": 0.0,
                "origin_status": "SYNTHETIC_TEST_POSITION",
            },
        ]
        same_direction = [
            {
                "event_id": 3, "pdg_id": 22, "energy_gev": 1.0, "weighted_energy_gev": 1.0,
                "px_gev": 0.0, "py_gev": 0.0, "pz_gev": 1.0,
                "x": 10.0, "y": 0.0, "z": 0.0, "r": 10.0, "theta": math.pi / 2, "phi": 0.0,
                "origin_status": "SYNTHETIC_TEST_POSITION",
            },
            {
                "event_id": 4, "pdg_id": 22, "energy_gev": 1.0, "weighted_energy_gev": 1.0,
                "px_gev": 0.0, "py_gev": 0.0, "pz_gev": 1.0,
                "x": 20.0, "y": 0.0, "z": 0.0, "r": 20.0, "theta": math.pi / 2, "phi": 0.0,
                "origin_status": "SYNTHETIC_TEST_POSITION",
            },
        ]
        rows = []
        for label, packet_rows in [("same_origin_different_directions", same_origin), ("same_direction_different_origins", same_direction)]:
            for row in run_packet_case(tmp, packet_rows, label):
                rows.append({
                    "case": label,
                    "backend": "current_zamo_tetrad_packet_path",
                    "event_id": row["event_id"],
                    "pdg_id": row["pdg_id"],
                    "initial_x": row["x"],
                    "initial_y": row["y"],
                    "initial_z": row["z"],
                    "dir_x": row["dir_x"],
                    "dir_y": row["dir_y"],
                    "dir_z": row["dir_z"],
                    "final_status": row["final_status"],
                    "observer_pixel_i": row["observer_pixel_i"],
                    "observer_pixel_j": row["observer_pixel_j"],
                    "path_length": row["path_length"],
                    "redshift_factor": row["redshift_factor"],
                    "uses_real_hadros_geodesic": "no",
                    "interpretation": "Straight angular proxy; pixel depends on direction projection, not Kerr-integrated bending.",
                })
        rows.append({
            "case": "real_hadros_integrator",
            "backend": "KerrCamera/KerrGeodesic",
            "event_id": "",
            "pdg_id": "",
            "initial_x": "",
            "initial_y": "",
            "initial_z": "",
            "dir_x": "",
            "dir_y": "",
            "dir_z": "",
            "final_status": "NOT_CONNECTED_TO_PACKET_PIPELINE",
            "observer_pixel_i": "",
            "observer_pixel_j": "",
            "path_length": "",
            "redshift_factor": "",
            "uses_real_hadros_geodesic": "yes_but_not_for_packets",
            "interpretation": "Available for camera/deposition paths, not invoked by packet propagation.",
        })
        return rows


def write_markdown_report(path: Path) -> None:
    lines = [
        "# Packet Raytracing Path Audit",
        "",
        "Conclusion: `REAL_HADROS_KERR_GEODESIC` is available in the C++ packet backend; `PROXY_STRAIGHT_LINE` remains the Python/channel-image path.",
        "",
        "Phase 9.0 connects `PacketKerrNullPropagator::propagate` to the real HADROS null-geodesic stepper.",
        "The existing Python particle-channel image path is still produced by Kerr/ZAMO-initialized diagnostics followed by straight-line/angular projection.",
        "",
        "## Direct Answers",
        "",
        "- Does the C++ packet backend call the original HADROS null Kerr integrator? **Yes.** It calls `KerrGeodesic::step_adaptive`.",
        "- Do the current Python particle-channel images call the original HADROS null Kerr integrator? **No.** They still use `propagate_kerr_null_packets.py::propagate_one` plus angular projection.",
        "- Exact real packet function: `src/cascade/packet_kerr_null_propagator.cpp::PacketKerrNullPropagator::propagate`.",
        "- Exact proxy packet function: `scripts/cascade/propagate_kerr_null_packets.py::propagate_one`.",
        "- Where does real packet geodesic integration occur? In `PacketKerrNullPropagator::propagate`, after ZAMO initialization, through `KerrGeodesic::step_adaptive`.",
        "- How are Phase 9 observer coordinates determined? By the final Boyer-Lindquist angles at the escape/domain radius.",
        "- Is redshift calculated for Phase 9 packets? **Not yet.** `redshift_factor` remains `1.0` in the minimal validation.",
        "- Does Kerr curvature affect Phase 9 packet trajectories? **Yes.** The validation shows same-direction packets from different origins have identical straight-line angular coordinates but different real-Kerr observer coordinates.",
        "- Does spin enter dynamically? **Yes in the C++ backend**, through `KerrMetric` and `KerrGeodesic`. **No in the legacy Python proxy trajectory**, except horizon/tetrad diagnostics.",
        "- Does `zamo_tetrad` initialize a null `p^mu` and is it integrated? **Yes in the C++ backend.** The Python proxy still does not integrate that `p^mu`.",
        "",
        "## Real HADROS Geodesic Path Exists",
        "",
        "The real HADROS camera ray path is `KerrCamera::trace_pixel`, which builds a `GeodesicState` and calls `KerrGeodesic::step_adaptive` in `src/kerr_camera.cpp`.",
        "Phase 9.0 also exercises this camera path in `apps/packet_real_kerr_vs_straight.cpp` and separately advances packet initial conditions with `KerrGeodesic::step_adaptive`.",
        "",
        "## Remaining TODO",
        "",
        "Route the production Python/config-web particle-channel image pipeline through the C++ real-Kerr packet backend or a Python binding/app wrapper.",
        "A physical observer/camera criterion and packet redshift still need to be implemented before calling the resulting images physical observables.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Packet Straight vs Kerr-Geodesic Comparison",
        "",
        "The current packet `zamo_tetrad` path is compared against the available but not connected HADROS Kerr integrator.",
        "",
        "| case | backend | event | status | pixel | uses real HADROS geodesic | interpretation |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        pixel = f"({row['observer_pixel_i']},{row['observer_pixel_j']})" if row["observer_pixel_i"] != "" else ""
        lines.append(
            f"| {row['case']} | {row['backend']} | {row['event_id']} | {row['final_status']} | {pixel} | "
            f"{row['uses_real_hadros_geodesic']} | {row['interpretation']} |"
        )
    lines.extend([
        "",
        "Same-origin/different-direction packets can land in different pixels, but this comes from angular projection.",
        "Same-direction/different-origin packets share the same angular pixel in the current proxy unless status/domain intersections differ.",
        "No evidence of Kerr bending is produced because the packet path does not call the Kerr geodesic stepper.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    args = parser.parse_args()
    fields = [
        "component", "file", "function_or_class", "role", "uses_hadros_null_geodesic_integrator",
        "uses_kerr_metric", "uses_straight_line", "status", "notes",
    ]
    write_csv(args.output_dir / "packet_raytracing_path_audit.csv", AUDIT_ROWS, fields)
    comparison_rows = physical_comparison_rows()
    comparison_fields = [
        "case", "backend", "event_id", "pdg_id", "initial_x", "initial_y", "initial_z",
        "dir_x", "dir_y", "dir_z", "final_status", "observer_pixel_i", "observer_pixel_j",
        "path_length", "redshift_factor", "uses_real_hadros_geodesic", "interpretation",
    ]
    write_csv(args.output_dir / "packet_straight_vs_kerr_geodesic_comparison.csv", comparison_rows, comparison_fields)
    write_markdown_report(ROOT / "docs/external_generators/PACKET_RAYTRACING_PATH_AUDIT.md")
    write_comparison_report(args.output_dir / "packet_straight_vs_kerr_geodesic_comparison.md", comparison_rows)
    print(json.dumps({"conclusion": "REAL_HADROS_KERR_GEODESIC_AVAILABLE_WITH_PROXY_PYTHON_PATH_REMAINING", "audit_csv": str(args.output_dir / "packet_raytracing_path_audit.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
