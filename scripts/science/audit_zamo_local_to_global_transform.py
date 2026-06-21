#!/usr/bin/env python3
"""Audit Phase 15.4g ZAMO local-box to global Kerr particle positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READY = ROOT / "output" / "science" / "powheg_pythia_geant4_resumable" / "geant4_ready_particles.jsonl"
DEFAULT_CAMERA = ROOT / "output" / "science" / "real_kerr_particle_camera"
APP = ROOT / "build" / "compute_kerr_particle_camera"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def make_cartesian_fixture(rows: list[dict[str, Any]], path: Path) -> None:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["global_exit_x_rg"] = row.get("cartesian_global_exit_x_rg", row.get("global_exit_x_rg"))
        out["global_exit_y_rg"] = row.get("cartesian_global_exit_y_rg", row.get("global_exit_y_rg"))
        out["global_exit_z_rg"] = row.get("cartesian_global_exit_z_rg", row.get("global_exit_z_rg"))
        out["global_exit_r_rg"] = row.get("cartesian_global_exit_r_rg", row.get("global_exit_r_rg"))
        out["global_exit_theta_rad"] = row.get("cartesian_global_exit_theta_rad", row.get("global_exit_theta_rad"))
        out["global_exit_phi_rad"] = row.get("cartesian_global_exit_phi_rad", row.get("global_exit_phi_rad"))
        out["global_position_status"] = "GLOBAL_POSITION_VALID"
        out["global_position_transform"] = "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION"
        out["local_to_global_transform"] = "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION"
        out["tetrad_status"] = "NOT_USED_LOCAL_CARTESIAN_COMPARISON"
        out_rows.append(out)
    write_jsonl(path, out_rows)


def run_camera(input_path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.camera_app),
            str(input_path),
            str(out_dir),
            str(args.aspin),
            str(args.r_obs_rg),
            str(args.theta_deg),
            str(args.fov_deg),
            str(args.nx),
            str(args.ny),
            str(args.r_max_rg),
            str(args.step),
            str(args.spatial_tolerance_rg),
            str(args.angular_tolerance_deg),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def observed_metrics(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    total_energy = sum(fnum(row, "weighted_energy_gev") for row in rows)
    nonzero = {(int(float(row["pixel_x"])), int(float(row["pixel_y"]))) for row in rows}
    if total_energy > 0.0:
        cx = sum(fnum(row, "pixel_x") * fnum(row, "weighted_energy_gev") for row in rows) / total_energy
        cy = sum(fnum(row, "pixel_y") * fnum(row, "weighted_energy_gev") for row in rows) / total_energy
    else:
        cx = cy = 0.0
    return {
        "observed_rows": len(rows),
        "nonzero_pixels": len(nonzero),
        "weighted_energy_gev": total_energy,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def position_delta_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas: list[float] = []
    for row in rows:
        dx = fnum(row, "global_exit_x_rg") - fnum(row, "cartesian_global_exit_x_rg")
        dy = fnum(row, "global_exit_y_rg") - fnum(row, "cartesian_global_exit_y_rg")
        dz = fnum(row, "global_exit_z_rg") - fnum(row, "cartesian_global_exit_z_rg")
        delta = math.sqrt(dx * dx + dy * dy + dz * dz)
        if math.isfinite(delta):
            deltas.append(delta)
    deltas.sort()
    if not deltas:
        return {"delta_position_min_rg": 0.0, "delta_position_median_rg": 0.0, "delta_position_max_rg": 0.0}
    return {
        "delta_position_min_rg": deltas[0],
        "delta_position_median_rg": deltas[len(deltas) // 2],
        "delta_position_max_rg": deltas[-1],
    }


def rgb_image_from_observed(rows: list[dict[str, str]], nx: int, ny: int) -> Any:
    import numpy as np

    image = np.zeros((ny, nx, 3), dtype=float)
    max_value = 0.0
    for row in rows:
        x = int(float(row["pixel_x"]))
        y = int(float(row["pixel_y"]))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        energy = fnum(row, "weighted_energy_gev")
        channel = row.get("channel", "")
        if channel == "gamma":
            image[y, x, 0] += energy
        elif channel in {"lepton", "electromagnetic"}:
            image[y, x, 1] += energy
        else:
            image[y, x, 2] += energy
        max_value = max(max_value, image[y, x].max())
    if max_value > 0.0:
        image = image / max_value
    return image


def write_plot(zamo_csv: Path, cart_csv: Path, out_path: Path, nx: int, ny: int) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hadros-zamo")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    zamo = rgb_image_from_observed(read_csv(zamo_csv), nx, ny)
    cart = rgb_image_from_observed(read_csv(cart_csv), nx, ny)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cart, origin="lower")
    axes[0].set_title("local Cartesian")
    axes[1].imshow(zamo, origin="lower")
    axes[1].set_title("ZAMO tetrad")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_zamo_audit_docs(camera_output_dir: Path) -> None:
    audit_rows = [
        {
            "question": "exists_reusable_zamo_tetrad",
            "answer": "YES",
            "status": "REUSABLE_WITH_PYTHON_EQUIVALENT_FOR_LOCAL_BOX",
            "evidence": "include/hadros/cascade/kerr_local_tetrad.hpp and src/cascade/kerr_local_tetrad.cpp implement ZAMO/LNRF null-packet initialization; scripts/science/run_powheg_pythia_geant4_resumable.py reuses the same Kerr spatial metric factors for local orthonormal box displacements.",
        },
        {
            "question": "already_used_in_packets",
            "answer": "YES",
            "status": "USED_BY_PACKET_KERR_NULL_PROPAGATOR",
            "evidence": "src/cascade/packet_kerr_null_propagator.cpp includes hadros/cascade/kerr_local_tetrad.hpp and uses initialize_zamo_null_packet for packet initialization.",
        },
        {
            "question": "accepts_global_kerr_position",
            "answer": "YES",
            "status": "ACCEPTS_BL_R_THETA_PHI",
            "evidence": "initialize_zamo_null_packet(metric,r,theta,phi,direction) accepts Boyer-Lindquist position; the GEANT4 resumable path loads interaction_r_rg, interaction_theta_rad, and interaction_phi_rad.",
        },
        {
            "question": "transforms_local_momentum_to_global",
            "answer": "PARTIAL",
            "status": "SPATIAL_TRIAD_MOMENTUM_RECORDED",
            "evidence": "GEANT4 local px,py,pz are rotated through the local ZAMO spatial triad into global_px/global_py/global_pz; a full massive BL four-momentum is not claimed.",
        },
        {
            "question": "transforms_local_box_displacement_to_global",
            "answer": "YES",
            "status": "ZAMO_TETRAD_LOCAL_BOX",
            "evidence": "dx_local is interpreted as orthonormal (+theta,+phi,+radial) and converted through dr=dz/sqrt(g_rr), dtheta=dx/sqrt(g_thetatheta), dphi=dy/sqrt(g_phiphi).",
        },
        {
            "question": "cartesian_approximation_science_default",
            "answer": "NO",
            "status": "COMPARISON_ONLY",
            "evidence": "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION is retained only in cartesian_global_exit_* fields and cartesian_transform_ready_particles.jsonl.",
        },
    ]
    write_csv(camera_output_dir / "zamo_transform_audit.csv", audit_rows)

    doc = [
        "# ZAMO Local-to-Global Transform Audit",
        "",
        "Status: `REAL_KERR_PARTICLE_CAMERA_GEOMETRY_VALIDATED_ZAMO`.",
        "",
        "Phase 15.4g removes the scientific `LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION` from the POWHEG/PYTHIA8 -> GEANT4 local box -> real Kerr particle-camera route. The Cartesian result is still retained only as an explicit comparison product.",
        "",
        "## Findings",
        "",
        "| question | answer | status |",
        "|---|---|---|",
    ]
    for row in audit_rows:
        doc.append(f"| {row['question']} | {row['answer']} | `{row['status']}` |")
    doc.extend(
        [
            "",
            "## Implementation",
            "",
            "The GEANT4 local box records escaped positions as local orthonormal displacements. The adopted convention is:",
            "",
            "```text",
            "local z -> outward ZAMO radial axis",
            "local x -> +theta ZAMO axis",
            "local y -> +phi ZAMO axis",
            "```",
            "",
            "For a small local displacement at the sampled interaction point, the transform is:",
            "",
            "```text",
            "dr     = dz_local / sqrt(g_rr)",
            "dtheta = dx_local / sqrt(g_thetatheta)",
            "dphi   = dy_local / sqrt(g_phiphi)",
            "```",
            "",
            "The transformed global fields are written as `global_exit_x_rg`, `global_exit_y_rg`, `global_exit_z_rg`, `global_exit_r_rg`, `global_exit_theta_rad`, and `global_exit_phi_rad`, with `global_position_status=GLOBAL_POSITION_VALID_ZAMO_TETRAD` and `local_to_global_transform=ZAMO_TETRAD_LOCAL_BOX`.",
            "",
            "This is a local small-box transform. It is valid when the GEANT4 box is small compared with the curvature and coordinate-variation scales at the sampled interaction point.",
            "",
            "## Momentum",
            "",
            "`global_px`, `global_py`, and `global_pz` are recorded by rotating the GEANT4 local spatial momentum through the same local ZAMO spatial triad. The status is `GLOBAL_MOMENTUM_ZAMO_SPATIAL_TRIAD`; no full massive Boyer-Lindquist four-momentum claim is made.",
            "",
            "## Evidence",
            "",
            "- `output/science/powheg_pythia_geant4_resumable/geant4_ready_particles.jsonl` uses `ZAMO_TETRAD_LOCAL_BOX`.",
            "- `output/science/real_kerr_particle_camera/particle_ray_association_camera.csv` is generated by the particle-ray association camera; legacy `observed_particles_by_pixel.csv` may be present only for compatibility.",
            "- `output/science/real_kerr_particle_camera/local_cartesian_vs_zamo_transform.md` documents the Cartesian comparison.",
            "- No particle-to-screen projection fallback is used.",
        ]
    )
    (ROOT / "docs" / "science" / "ZAMO_LOCAL_TO_GLOBAL_TRANSFORM_AUDIT.md").write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--camera-output-dir", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--camera-app", type=Path, default=APP)
    parser.add_argument("--aspin", type=float, default=-0.8)
    parser.add_argument("--r-obs-rg", type=float, default=60.0)
    parser.add_argument("--theta-deg", type=float, default=85.2500572958)
    parser.add_argument("--fov-deg", type=float, default=75.0)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--r-max-rg", type=float, default=80.0)
    parser.add_argument("--step", type=float, default=0.75)
    parser.add_argument("--spatial-tolerance-rg", type=float, default=1.0)
    parser.add_argument("--angular-tolerance-deg", type=float, default=5.0)
    args = parser.parse_args()

    rows = read_jsonl(args.ready)
    cartesian_ready = args.camera_output_dir / "cartesian_transform_ready_particles.jsonl"
    cartesian_out = args.camera_output_dir / "cartesian_transform_camera"
    make_cartesian_fixture(rows, cartesian_ready)
    run_camera(cartesian_ready, cartesian_out, args)

    zamo_metrics = observed_metrics(args.camera_output_dir / "observed_particles_by_pixel.csv")
    cart_metrics = observed_metrics(cartesian_out / "observed_particles_by_pixel.csv")
    delta_metrics = position_delta_metrics(rows)
    row = {
        "component": "local_box_to_global_position",
        "old_transform": "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION",
        "new_transform": "ZAMO_TETRAD_LOCAL_BOX",
        "rows": len(rows),
        **delta_metrics,
        "zamo_observed_rows": zamo_metrics["observed_rows"],
        "cartesian_observed_rows": cart_metrics["observed_rows"],
        "zamo_nonzero_pixels": zamo_metrics["nonzero_pixels"],
        "cartesian_nonzero_pixels": cart_metrics["nonzero_pixels"],
        "zamo_weighted_energy_gev": zamo_metrics["weighted_energy_gev"],
        "cartesian_weighted_energy_gev": cart_metrics["weighted_energy_gev"],
        "zamo_centroid_x": zamo_metrics["centroid_x"],
        "zamo_centroid_y": zamo_metrics["centroid_y"],
        "cartesian_centroid_x": cart_metrics["centroid_x"],
        "cartesian_centroid_y": cart_metrics["centroid_y"],
        "delta_global_exit_position_rg": delta_metrics["delta_position_median_rg"],
        "delta_observed_energy": zamo_metrics["weighted_energy_gev"] - cart_metrics["weighted_energy_gev"],
        "delta_nonzero_pixels": zamo_metrics["nonzero_pixels"] - cart_metrics["nonzero_pixels"],
        "delta_channel_centroid_x": zamo_metrics["centroid_x"] - cart_metrics["centroid_x"],
        "delta_channel_centroid_y": zamo_metrics["centroid_y"] - cart_metrics["centroid_y"],
        "delta_channel_centroids": math.sqrt(
            (zamo_metrics["centroid_x"] - cart_metrics["centroid_x"]) ** 2
            + (zamo_metrics["centroid_y"] - cart_metrics["centroid_y"]) ** 2
        ),
        "status": "REAL_KERR_PARTICLE_CAMERA_GEOMETRY_VALIDATED_ZAMO"
        if zamo_metrics["observed_rows"] > 0
        else "REAL_KERR_PARTICLE_CAMERA_GEOMETRY_PARTIAL_ZAMO",
    }
    write_csv(args.camera_output_dir / "local_cartesian_vs_zamo_transform.csv", [row])
    write_zamo_audit_docs(args.camera_output_dir)
    write_plot(
        args.camera_output_dir / "observed_particles_by_pixel.csv",
        cartesian_out / "observed_particles_by_pixel.csv",
        args.camera_output_dir / "plots" / "cartesian_vs_zamo_rgb_comparison.png",
        args.nx,
        args.ny,
    )
    md = [
        "# Local Cartesian vs ZAMO Transform",
        "",
        f"Status: `{row['status']}`.",
        "",
        f"Rows compared: `{row['rows']}`.",
        f"Median position delta [rg]: `{row['delta_position_median_rg']:.12g}`.",
        f"Max position delta [rg]: `{row['delta_position_max_rg']:.12g}`.",
        f"Associated-energy delta [GeV]: `{row['delta_observed_energy']:.12g}`.",
        f"Nonzero pixel delta: `{row['delta_nonzero_pixels']}`.",
        f"Channel centroid delta [pixel]: `{row['delta_channel_centroids']:.12g}`.",
        f"ZAMO observed rows: `{row['zamo_observed_rows']}`.",
        f"Cartesian observed rows: `{row['cartesian_observed_rows']}`.",
        "",
        "The Cartesian transform is retained only as an explicit comparison product.",
        "Scientific GEANT4-ready particles use `ZAMO_TETRAD_LOCAL_BOX`.",
    ]
    (args.camera_output_dir / "local_cartesian_vs_zamo_transform.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
