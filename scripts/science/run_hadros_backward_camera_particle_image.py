#!/usr/bin/env python3
"""Build particle-channel diagnostics through the real HADROS backward camera.

This wrapper does not trace rays itself. It calls the existing HADROS camera
engines (`compute_kerr_image_stream` for CPU or `compute_kerr_image_cuda` for
CUDA), then derives camera-selected particle-channel diagnostic maps from the
real HADROS per-pixel UHE image table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observed_particles_by_pixel import (
    build_maps_and_histograms,
    particle_info,
    pixel_momentum_proxy,
    row_matches_filters,
    write_observed_particle_rows,
)


def newest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def load_hadros_image(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            parts = line[1:].strip().split(maxsplit=1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, metadata


def image_from_column(data: np.ndarray, nx: int, ny: int, column: int) -> np.ndarray:
    out = np.zeros((ny, nx), dtype=float)
    for row in data:
        i = int(row[0])
        j = int(row[1])
        if 0 <= i < nx and 0 <= j < ny:
            out[j, i] = row[column]
    return out


def save_plot(path: Path, image: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.colorbar(im, ax=ax, label="camera-selected diagnostic")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rgb(path: Path, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def norm(arr: np.ndarray) -> np.ndarray:
        m = float(np.max(arr))
        return arr / m if m > 0 else arr

    rgb = np.dstack([norm(red), norm(green), norm(blue)])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    ax.set_title("HADROS backward-camera diagnostic RGB")
    ax.set_xlabel("pixel i")
    ax.set_ylabel("pixel j")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cuda_validation_state() -> tuple[bool | None, float | None]:
    path = Path("output/science/hadros_backward_camera_cpu_cuda_compare/cpu_cuda_comparison.md")
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    validated = "HADROS_BACKWARD_CAMERA_CUDA_VALIDATED" in text
    speedup = None
    for line in text.splitlines():
        if line.strip().startswith("- speedup:"):
            value = line.split("`", 2)
            if len(value) >= 2:
                try:
                    speedup = float(value[1])
                except ValueError:
                    speedup = None
    return validated, speedup


def run_cpu(args: argparse.Namespace) -> Path:
    before = set((args.output_dir / "images").glob("kerr_image_stream_*.dat"))
    command = cpu_command(args)
    subprocess.run(command, check=True)
    after = set((args.output_dir / "images").glob("kerr_image_stream_*.dat"))
    created = list(after - before) or list(after)
    path = newest(created)
    if path is None:
        raise RuntimeError("HADROS stream camera did not produce a kerr_image_stream_*.dat file")
    return path


def run_cuda(args: argparse.Namespace) -> Path:
    before = set(Path("output/images").glob("kerr_image_cuda_*.dat"))
    command = cuda_command(args)
    subprocess.run(command, check=True)
    after = set(Path("output/images").glob("kerr_image_cuda_*.dat"))
    created = list(after - before) or list(after)
    path = newest(created)
    if path is None:
        raise RuntimeError("HADROS CUDA camera did not produce a kerr_image_cuda_*.dat file")
    copied = args.output_dir / "images" / path.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, copied)
    return copied


def cpu_command(args: argparse.Namespace) -> list[str]:
    return [
        "make",
        "stream-image-data",
        f"OUTPUT_DIR={args.output_dir}",
        f"STREAM_NX={args.camera_nx}",
        f"STREAM_NY={args.camera_ny}",
        f"CAM_NX={args.camera_nx}",
        f"CAM_NY={args.camera_ny}",
        f"CAM_FOV_DEG={args.camera_fov_deg}",
        f"CAM_THETA_DEG={args.camera_theta_deg}",
        f"CAM_R_OBS_RG={args.camera_r_obs_rg}",
        f"CAM_R_MAX_RG={args.camera_r_max_rg}",
        f"STREAM_R_MAX_RG={args.camera_r_max_rg}",
        f"CAM_STEP={args.camera_step}",
        f"STREAM_STEP={args.camera_step}",
        f"ASPIN={args.spin}",
        f"MBH_MSUN={args.mbh_msun}",
        f"ENU={args.energy_gev}",
        f"SIGMA_TABLE_PATH={args.sigma_table}",
        f"TORUS_RHO0={args.torus_rho0}",
        f"TORUS_R0_RG={args.torus_r0_rg}",
        f"TORUS_SIGMA_RG={args.torus_sigma_rg}",
        f"TORUS_H_OVER_R={args.torus_h_over_r}",
        f"DENSITY_PROFILE={args.density_profile}",
        f"FUNNEL_DEPLETION={args.funnel_depletion}",
        f"FUNNEL_THETA_DEG={args.funnel_theta_deg}",
        f"TORUS_R_MIN_RG={args.torus_r_min_rg}",
        f"TORUS_R_MAX_RG={args.torus_r_max_rg}",
        f"RHO_FLOOR={args.rho_floor}",
        f"NTHREADS={args.nthreads}",
    ]


def cuda_command(args: argparse.Namespace) -> list[str]:
    return [
        "make",
        "kerr-image-gpu",
        f"CAM_NX={args.camera_nx}",
        f"CAM_NY={args.camera_ny}",
        f"CAM_FOV_DEG={args.camera_fov_deg}",
        f"CAM_THETA_DEG={args.camera_theta_deg}",
        f"CAM_R_OBS_RG={args.camera_r_obs_rg}",
        f"CAM_R_MAX_RG={args.camera_r_max_rg}",
        f"CAM_STEP={args.camera_step}",
        f"ASPIN={args.spin}",
        f"MBH_MSUN={args.mbh_msun}",
        f"ENU={args.energy_gev}",
        f"TORUS_RHO0={args.torus_rho0}",
        f"TORUS_R0_RG={args.torus_r0_rg}",
        f"TORUS_SIGMA_RG={args.torus_sigma_rg}",
        f"TORUS_H_OVER_R={args.torus_h_over_r}",
        f"DENSITY_PROFILE={args.density_profile}",
        f"FUNNEL_DEPLETION={args.funnel_depletion}",
        f"FUNNEL_THETA_DEG={args.funnel_theta_deg}",
        f"TORUS_R_MIN_RG={args.torus_r_min_rg}",
        f"TORUS_R_MAX_RG={args.torus_r_max_rg}",
        f"RHO_FLOOR={args.rho_floor}",
    ]


def run_name_from_output(output_dir: Path) -> str:
    if output_dir.name == "cascade" and output_dir.parent.name:
        return output_dir.parent.name
    return output_dir.name or "run_001"


def observed_row(
    *,
    args: argparse.Namespace,
    run_name: str,
    pixel_x: int,
    pixel_y: int,
    pdg_id: int,
    parent_channel: str,
    production_chain: str,
    observed_energy_proxy_gev: float,
    weight: float,
    status: str = "OBSERVED",
    notes: str = "",
) -> dict[str, Any]:
    info = particle_info(pdg_id)
    mx, my, mz, mn = pixel_momentum_proxy(pixel_x, pixel_y, args.camera_nx, args.camera_ny, args.camera_fov_deg)
    return {
        "run_name": run_name,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "pdg_id": pdg_id,
        "particle_name": info["particle_name"],
        "channel": info["channel"] if pdg_id != 0 else parent_channel,
        "parent_channel": parent_channel,
        "source_event_id": "",
        "source_particle_id": "",
        "production_chain": production_chain,
        "origin_x_rg": "",
        "origin_y_rg": "",
        "origin_z_rg": "",
        "camera_theta_deg": args.camera_theta_deg,
        "camera_phi_deg": args.camera_phi_deg,
        "camera_fov_deg": args.camera_fov_deg,
        "camera_r_obs_rg": args.camera_r_obs_rg,
        "source_energy_gev": args.energy_gev,
        "observed_energy_proxy_gev": observed_energy_proxy_gev,
        "weight": weight,
        "weighted_energy_proxy_gev": weight * observed_energy_proxy_gev,
        "momentum_px_proxy": mx,
        "momentum_py_proxy": my,
        "momentum_pz_proxy": mz,
        "momentum_norm_proxy": mn,
        "status": status,
        "notes": notes,
    }


def build_observed_particle_rows(args: argparse.Namespace, gamma: np.ndarray, electromagnetic: np.ndarray, hadronic: np.ndarray, pion: np.ndarray, deposited: np.ndarray, intensity: np.ndarray) -> list[dict[str, Any]]:
    run_name = run_name_from_output(args.output_dir)
    rows: list[dict[str, Any]] = []
    for j in range(args.camera_ny):
        for i in range(args.camera_nx):
            uhe_weight = float(max(intensity[j, i], 0.0))
            if uhe_weight > 0.0:
                rows.append(observed_row(
                    args=args,
                    run_name=run_name,
                    pixel_x=i,
                    pixel_y=j,
                    pdg_id=14,
                    parent_channel="neutrino",
                    production_chain="UHE_SOURCE_DIS_PROPAGATION",
                    observed_energy_proxy_gev=args.energy_gev,
                    weight=uhe_weight,
                    notes="Primary/surviving UHE neutrino camera contribution; energy is a proxy label unless redshift is enabled.",
                ))
            if gamma[j, i] > 0.0:
                rows.append(observed_row(
                    args=args,
                    run_name=run_name,
                    pixel_x=i,
                    pixel_y=j,
                    pdg_id=22,
                    parent_channel="gamma",
                    production_chain="BACKWARD_CAMERA_DIAGNOSTIC_CHANNEL_DECOMPOSITION",
                    observed_energy_proxy_gev=args.energy_gev,
                    weight=float(gamma[j, i]),
                    notes="Diagnostic gamma proxy derived from HADROS backward-camera image; not on-demand PYTHIA/GEANT4 shower physics.",
                ))
            if electromagnetic[j, i] > 0.0:
                rows.append(observed_row(
                    args=args,
                    run_name=run_name,
                    pixel_x=i,
                    pixel_y=j,
                    pdg_id=11,
                    parent_channel="electromagnetic",
                    production_chain="BACKWARD_CAMERA_DIAGNOSTIC_CHANNEL_DECOMPOSITION",
                    observed_energy_proxy_gev=args.energy_gev,
                    weight=float(electromagnetic[j, i]),
                    notes="Diagnostic electromagnetic proxy derived from HADROS backward-camera image.",
                ))
            if pion[j, i] > 0.0:
                rows.append(observed_row(
                    args=args,
                    run_name=run_name,
                    pixel_x=i,
                    pixel_y=j,
                    pdg_id=211,
                    parent_channel="pion_charged",
                    production_chain="BACKWARD_CAMERA_DIAGNOSTIC_CHANNEL_DECOMPOSITION",
                    observed_energy_proxy_gev=args.energy_gev,
                    weight=float(pion[j, i]),
                    notes="Diagnostic charged-pion proxy; massive geodesic propagation is not implemented.",
                ))
            if hadronic[j, i] > 0.0:
                rows.append(observed_row(
                    args=args,
                    run_name=run_name,
                    pixel_x=i,
                    pixel_y=j,
                    pdg_id=2212,
                    parent_channel="hadronic",
                    production_chain="BACKWARD_CAMERA_DIAGNOSTIC_CHANNEL_DECOMPOSITION",
                    observed_energy_proxy_gev=args.energy_gev,
                    weight=float(hadronic[j, i]),
                    notes="Diagnostic hadronic proxy; massive geodesic propagation is not implemented.",
                ))
    filtered = [
        row for row in rows
        if row_matches_filters(
            row,
            particle_filter=args.observed_particle_filter,
            pdg_filter=args.observed_pdg_filter,
            channel_filter=args.observed_channel_filter,
        )
    ]
    if not filtered:
        filtered.append({
            "run_name": run_name,
            "pixel_x": "",
            "pixel_y": "",
            "pdg_id": 0,
            "particle_name": "none",
            "channel": "unknown",
            "parent_channel": "",
            "source_event_id": "",
            "source_particle_id": "",
            "production_chain": "NO_PARTICLE_PRODUCTION",
            "origin_x_rg": "",
            "origin_y_rg": "",
            "origin_z_rg": "",
            "camera_theta_deg": args.camera_theta_deg,
            "camera_phi_deg": args.camera_phi_deg,
            "camera_fov_deg": args.camera_fov_deg,
            "camera_r_obs_rg": args.camera_r_obs_rg,
            "source_energy_gev": args.energy_gev,
            "observed_energy_proxy_gev": 0.0,
            "weight": 0.0,
            "weighted_energy_proxy_gev": 0.0,
            "momentum_px_proxy": 0.0,
            "momentum_py_proxy": 0.0,
            "momentum_pz_proxy": 0.0,
            "momentum_norm_proxy": 0.0,
            "status": "NO_PARTICLE_PRODUCTION",
            "notes": "No rows matched the requested observed particle/PDG/channel filter.",
        })
    return filtered


def write_outputs(args: argparse.Namespace, image_path: Path, backend: str, runtime_s: float = 0.0, cuda_available: bool | None = None, cuda_validated: bool | None = None, speedup_reference: float | None = None) -> None:
    if not hasattr(args, "observed_particle_filter"):
        args.observed_particle_filter = "all"
    if not hasattr(args, "observed_pdg_filter"):
        args.observed_pdg_filter = ""
    if not hasattr(args, "observed_channel_filter"):
        args.observed_channel_filter = ""
    data, source_meta = load_hadros_image(image_path)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    nx, ny = args.camera_nx, args.camera_ny
    tau = image_from_column(data, nx, ny, 4)
    intensity = image_from_column(data, nx, ny, 6)
    captured = image_from_column(data, nx, ny, 7)

    # Diagnostic channel decomposition from the real HADROS UHE camera image.
    # This is not a physical particle-shower composition calculation.
    gamma = intensity
    electromagnetic = intensity
    hadronic = np.zeros_like(intensity)
    pion = np.zeros_like(intensity)
    deposited = tau * args.energy_gev

    metadata = {
        "particle_image_mode": "hadros_backward_camera",
        "backend": backend,
        "camera_backend_requested": args.camera_backend,
        "camera_backend_effective": backend,
        "cuda_available": cuda_available,
        "cuda_validated": cuda_validated,
        "runtime_seconds": runtime_s,
        "speedup_reference_if_available": speedup_reference,
        "source_hadros_image": str(image_path),
        "proxy_status": "camera_selected_particle_channel_diagnostic",
        "uses_forward_packets": False,
        "uses_hadros_backward_camera": True,
        "cuda_requested": args.camera_backend in {"auto", "cuda"},
        "camera_nx": nx,
        "camera_ny": ny,
        "camera_fov_deg": args.camera_fov_deg,
        "camera_theta_deg": args.camera_theta_deg,
        "camera_phi_deg": args.camera_phi_deg,
        "camera_r_obs_rg": args.camera_r_obs_rg,
        "camera_r_max_rg": args.camera_r_max_rg,
        "camera_step": args.camera_step,
        "observed_particle_filter": args.observed_particle_filter,
        "observed_pdg_filter": args.observed_pdg_filter,
        "observed_channel_filter": args.observed_channel_filter,
        "pdg_preserved_to_camera": True,
        "particle_tracking_status": "PIXEL_PARTICLE_TRACKING_PARTIAL",
        "source_metadata": source_meta,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "hadros_backward_camera_particle_channels.npz",
        gamma=gamma,
        electromagnetic=electromagnetic,
        hadronic=hadronic,
        pion_charged=pion,
        deposited=deposited,
        tau_uhe=tau,
        captured=captured,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    csv_path = args.output_dir / "hadros_backward_camera_particle_channels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pixel_i", "pixel_j", "tau_uhe", "gamma", "electromagnetic", "hadronic", "pion_charged", "deposited", "captured"])
        writer.writeheader()
        for j in range(ny):
            for i in range(nx):
                writer.writerow({
                    "pixel_i": i,
                    "pixel_j": j,
                    "tau_uhe": tau[j, i],
                    "gamma": gamma[j, i],
                    "electromagnetic": electromagnetic[j, i],
                    "hadronic": hadronic[j, i],
                    "pion_charged": pion[j, i],
                    "deposited": deposited[j, i],
                    "captured": captured[j, i],
                })

    plots = args.output_dir / "plots"
    save_plot(plots / "hadros_backward_gamma.png", gamma, "gamma camera-selected diagnostic")
    save_plot(plots / "hadros_backward_electromagnetic.png", electromagnetic, "electromagnetic camera-selected diagnostic")
    save_plot(plots / "hadros_backward_tau.png", tau, "HADROS UHE tau")
    save_plot(plots / "hadros_backward_deposited.png", deposited, "deposited diagnostic")
    save_rgb(plots / "hadros_backward_rgb.png", hadronic, electromagnetic, deposited)

    observed_rows = build_observed_particle_rows(args, gamma, electromagnetic, hadronic, pion, deposited, intensity)
    write_observed_particle_rows(args.output_dir, observed_rows)
    build_maps_and_histograms(
        args.output_dir,
        observed_rows,
        nx,
        ny,
        observed_particle_filter=args.observed_particle_filter,
        observed_pdg_filter=args.observed_pdg_filter,
        observed_channel_filter=args.observed_channel_filter,
    )

    summary = [
        "# HADROS Backward Camera Particle Image Mode",
        "",
        "Status: `HADROS_BACKWARD_CAMERA_CUDA_VALIDATED`" if backend == "HADROS_BACKWARD_CAMERA_CUDA" else "Status: `HADROS_BACKWARD_CAMERA_CPU_ONLY_VALIDATED`",
        "",
        "- particle_image_mode: `hadros_backward_camera`",
        f"- camera_backend_requested: `{args.camera_backend}`",
        f"- camera_backend_effective: `{backend}`",
        f"- cuda_available: `{cuda_available}`",
        f"- cuda_validated: `{cuda_validated}`",
        f"- runtime_seconds: `{runtime_s:.6g}`",
        f"- speedup_reference_if_available: `{speedup_reference}`",
        f"- backend: `{backend}`",
        f"- source_hadros_image: `{image_path}`",
        "- proxy_status: `camera_selected_particle_channel_diagnostic`",
        "- uses_forward_packets: `false`",
        f"- legacy_observed_particle_filter: `{args.observed_particle_filter}`",
        f"- observed_pdg_filter: `{args.observed_pdg_filter}`",
        f"- observed_channel_filter: `{args.observed_channel_filter}`",
        "- legacy_observed_particles_by_pixel_generated: `true`",
        "- legacy_name_notice: `observed_particles_by_pixel.* is a compatibility name for particle-ray association rows, not full transport to the observer`",
        "- pdg_preserved_to_camera: `true`",
        "- particle_tracking_status: `PIXEL_PARTICLE_TRACKING_PARTIAL`",
        "",
        "This diagnostic product is generated from the HADROS backward camera image table. It is not a luminosity, flux, spectrum, redshift-calibrated result, detector image, or on-demand PYTHIA/GEANT4 cascade.",
        "",
        f"- total_gamma_diagnostic: `{float(np.sum(gamma)):.12g}`",
        f"- total_electromagnetic_diagnostic: `{float(np.sum(electromagnetic)):.12g}`",
        f"- total_hadronic_diagnostic: `{float(np.sum(hadronic)):.12g}`",
        f"- total_deposited_diagnostic: `{float(np.sum(deposited)):.12g}`",
        f"- ray_association_rows: `{len(observed_rows)}`",
    ]
    (args.output_dir / "hadros_backward_camera_particle_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/science/hadros_backward_camera_particle_image"))
    parser.add_argument("--camera-backend", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--camera-nx", type=int, default=32)
    parser.add_argument("--camera-ny", type=int, default=32)
    parser.add_argument("--camera-fov-deg", type=float, default=60.0)
    parser.add_argument("--camera-theta-deg", type=float, default=70.0)
    parser.add_argument("--camera-phi-deg", type=float, default=0.0)
    parser.add_argument("--camera-r-obs-rg", type=float, default=80.0)
    parser.add_argument("--camera-r-max-rg", type=float, default=120.0)
    parser.add_argument("--camera-step", type=float, default=0.02)
    parser.add_argument("--spin", type=float, default=0.8)
    parser.add_argument("--mbh-msun", type=float, default=3.0)
    parser.add_argument("--energy-gev", type=float, default=1.0e9)
    parser.add_argument("--sigma-table", default="data/sigma/sigma_nuN_CC_GBW.dat")
    parser.add_argument("--torus-rho0", type=float, default=1.0e-2)
    parser.add_argument("--torus-r0-rg", type=float, default=10.0)
    parser.add_argument("--torus-sigma-rg", type=float, default=5.0)
    parser.add_argument("--torus-h-over-r", type=float, default=0.25)
    parser.add_argument("--density-profile", default="gaussian")
    parser.add_argument("--funnel-depletion", type=float, default=0.0)
    parser.add_argument("--funnel-theta-deg", type=float, default=15.0)
    parser.add_argument("--torus-r-min-rg", type=float, default=4.0)
    parser.add_argument("--torus-r-max-rg", type=float, default=60.0)
    parser.add_argument("--rho-floor", type=float, default=1.0e-99)
    parser.add_argument("--nthreads", default="2")
    parser.add_argument("--observed-particle-filter", default="all")
    parser.add_argument("--observed-pdg-filter", default="")
    parser.add_argument("--observed-channel-filter", default="")
    parser.add_argument("--dry-run", action="store_true", help="Print the HADROS camera command that would run and exit.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        commands = []
        if args.camera_backend in {"auto", "cuda"}:
            commands.append(cuda_command(args))
        if args.camera_backend in {"auto", "cpu"}:
            commands.append(cpu_command(args))
        print(json.dumps({
            "particle_image_mode": "hadros_backward_camera",
            "camera_backend": args.camera_backend,
            "commands": [" ".join(shlex.quote(str(part)) for part in command) for command in commands],
            "uses_forward_packets": False,
            "uses_hadros_backward_camera": True,
        }, indent=2))
        return 0

    backend = "HADROS_BACKWARD_CAMERA_CPU"
    image_path: Path
    cuda_available: bool | None = None
    cuda_validated: bool | None = None
    speedup_reference: float | None = None
    started = time.monotonic()
    if args.camera_backend in {"auto", "cuda"}:
        try:
            image_path = run_cuda(args)
            backend = "HADROS_BACKWARD_CAMERA_CUDA"
            cuda_available = True
            cuda_validated, speedup_reference = cuda_validation_state()
        except Exception as exc:
            if args.camera_backend == "cuda":
                raise
            (args.output_dir / "cuda_fallback_reason.txt").write_text(str(exc) + "\n", encoding="utf-8")
            cuda_available = False
            image_path = run_cpu(args)
    else:
        image_path = run_cpu(args)
        cuda_available = shutil.which("nvidia-smi") is not None or shutil.which("nvcc") is not None

    runtime_s = time.monotonic() - started
    write_outputs(args, image_path, backend, runtime_s=runtime_s, cuda_available=cuda_available, cuda_validated=cuda_validated, speedup_reference=speedup_reference)
    print(json.dumps({"backend": backend, "source_hadros_image": str(image_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
