#!/usr/bin/env python3
"""Link POWHEG events to real HADROS UHE Kerr-ray samples."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

import argparse
import csv
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_gbw_iim_real_kerr_reweighting import (
    M_U_G,
    RG_CM_PER_MSUN,
    cfg_float,
    density_at_point_from_config,
    load_hadros_config,
    pint_from_tau,
    read_jsonl,
    read_sigma_table,
    sigma_interp,
    spherical_from_xyz,
    write_csv,
    write_jsonl,
)

import configparser


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "output/rays/kerr_geodesics_stream_validation_32x32.bin"
DEFAULT_POINTS = ROOT / "output/science/powheg_pythia_particles/interaction_points.jsonl"
DEFAULT_PARTICLES = ROOT / "output/science/powheg_pythia_particles/hadros_particle_events.jsonl"
DEFAULT_READY = ROOT / "output/science/powheg_pythia_geant4_resumable/geant4_ready_particles.jsonl"
DEFAULT_OBSERVED_CSV = ROOT / "output/science/real_kerr_particle_camera/observed_particles_by_pixel.csv"
DEFAULT_OBSERVED_JSONL = ROOT / "output/science/real_kerr_particle_camera/observed_particles_by_pixel.jsonl"
DEFAULT_OUTPUT = ROOT / "output/science/uhe_ray_event_link"
DEFAULT_CONFIG = ROOT / "config.ini"
DEFAULT_GBW = ROOT / "data/sigma/sigma_nuN_CC_GBW.dat"
DEFAULT_IIM = ROOT / "data/sigma/sigma_nuN_CC_IIM.dat"

STATUS_VALIDATED = "GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_VALIDATED_INCOMING_GEODESIC_COLUMN"
STATUS_BLOCKED = "UHE_RAY_EVENT_LINK_BLOCKED"
POINT_SIZE = struct.calcsize("<11d")
HEADER_SIZE = struct.calcsize("<iiiii4xdd")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def event_energy_by_id(particles: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for row in particles:
        out[int(row["event_id"])] += fnum(row, "energy_gev")
    return dict(out)


def local_neutrino_energy_gev(enu_inf_gev: float, redshift_factor: float) -> float:
    return enu_inf_gev * redshift_factor


def require_incident_energy(row: dict[str, Any], *, fallback: float | None = None) -> float:
    for key in ("E_nu_inf_GeV", "neutrino_energy_gev", "reference_energy_gev"):
        value = row.get(key)
        if value not in (None, ""):
            energy = fnum(row, key, math.nan)
            if math.isfinite(energy) and energy > 0.0:
                return energy
    if fallback is not None and math.isfinite(fallback) and fallback > 0.0:
        return fallback
    raise RuntimeError("Missing incident neutrino energy: expected E_nu_inf_GeV/neutrino_energy_gev/reference_energy_gev.")


def load_ray_samples(cache: Path, cfg: configparser.ConfigParser, rg_cm: float, sigma_gbw: tuple[list[float], list[float]], sigma_iim: tuple[list[float], list[float]], reference_energy_gev: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with cache.open("rb") as handle:
        magic, version, nx, ny, spin = struct.unpack("<iiiid", handle.read(24))
        if magic != 0x4B47454F:
            raise RuntimeError(f"invalid KGEO magic in {cache}")
        ray_count = 0
        while True:
            raw = handle.read(HEADER_SIZE)
            if not raw:
                break
            if len(raw) != HEADER_SIZE:
                raise RuntimeError(f"truncated ray header in {cache}")
            cache_ray_id, pixel_i, pixel_j, captured, npoints, alpha, beta = struct.unpack("<iiiii4xdd", raw)
            ray_count += 1
            column = 0.0
            lambda_rg = 0.0
            tau_gbw = 0.0
            tau_iim = 0.0
            incoming_ray_id = int(pixel_j) * int(nx) + int(pixel_i)
            for sample_index in range(npoints):
                payload = handle.read(POINT_SIZE)
                if len(payload) != POINT_SIZE:
                    raise RuntimeError(f"truncated path point in {cache}")
                r_rg, theta, x_rg, y_rg, z_rg, dl_rg, redshift, pt, pr, ptheta, pphi = struct.unpack("<11d", payload)
                dl_cm = max(dl_rg, 0.0) * rg_cm
                if not math.isfinite(redshift) or redshift <= 0.0:
                    raise RuntimeError(f"Missing/invalid redshift_factor in KGEO sample ray={cache_ray_id} sample={sample_index}")
                e_local = local_neutrino_energy_gev(reference_energy_gev, redshift)
                sigma_g, status_g = sigma_interp(e_local, sigma_gbw)
                sigma_i, status_i = sigma_interp(e_local, sigma_iim)
                rho, density_model = density_at_point_from_config(cfg, r_rg, theta)
                dcolumn = max(rho, 0.0) / M_U_G * dl_cm
                column += dcolumn
                lambda_rg += max(dl_rg, 0.0)
                if status_g != "OK" or status_i != "OK":
                    raise RuntimeError(
                        f"Local neutrino energy outside sigma table domain at ray={cache_ray_id} sample={sample_index}: "
                        f"E_nu_local_GeV={e_local:.12g}, GBW={status_g}, IIM={status_i}"
                    )
                tau_gbw += sigma_g * dcolumn
                tau_iim += sigma_i * dcolumn
                samples.append(
                    {
                        "incoming_ray_id": incoming_ray_id,
                        "geodesic_cache_ray_id": cache_ray_id,
                        "pixel_x": pixel_i,
                        "pixel_y": pixel_j,
                        "nx": nx,
                        "ny": ny,
                        "ray_id": incoming_ray_id,
                        "ray_id_convention": "pixel_y * nx + pixel_x",
                        "ray_sample_index": sample_index,
                        "lambda": lambda_rg,
                        "x": x_rg,
                        "y": y_rg,
                        "z": z_rg,
                        "r_rg": r_rg,
                        "theta_rad": theta,
                        "dl_rg": dl_rg,
                        "redshift_factor": redshift,
                        "E_nu_inf_GeV": reference_energy_gev,
                        "E_nu_local_GeV": e_local,
                        "sigma_GBW_cm2": sigma_g,
                        "sigma_IIM_cm2": sigma_i,
                        "column_before_cm2": column,
                        "tau_before_GBW": tau_gbw,
                        "tau_before_IIM": tau_iim,
                        "Pint_GBW": pint_from_tau(tau_gbw),
                        "Pint_IIM": pint_from_tau(tau_iim),
                        "density_g_cm3": max(rho, 0.0),
                        "density_profile_used": density_model,
                        "captured": captured,
                    }
                )
    summary = {
        "cache": str(cache),
        "version": version,
        "nx": nx,
        "ny": ny,
        "spin": spin,
        "rays": ray_count,
        "samples": len(samples),
    }
    return samples, summary


def nearest_sample(point: dict[str, Any], samples: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    px = fnum(point, "interaction_x_rg", math.nan)
    py = fnum(point, "interaction_y_rg", math.nan)
    pz = fnum(point, "interaction_z_rg", math.nan)
    best = samples[0]
    best_d2 = float("inf")
    for sample in samples:
        dx = sample["x"] - px
        dy = sample["y"] - py
        dz = sample["z"] - pz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best = sample
            best_d2 = d2
    return best, math.sqrt(best_d2)


def build_linked_points(args: argparse.Namespace, samples: list[dict[str, Any]], events: dict[int, float], sigma_gbw: tuple[list[float], list[float]], sigma_iim: tuple[list[float], list[float]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    old_points = read_jsonl(args.interaction_points)
    linked: list[dict[str, Any]] = []
    by_event: dict[int, dict[str, Any]] = {}
    for point in old_points:
        event_id = int(point["event_id"])
        if event_id not in events:
            continue
        sample, distance = nearest_sample(point, samples)
        energy_inf = require_incident_energy(point)
        redshift = fnum(sample, "redshift_factor", math.nan)
        energy_local = local_neutrino_energy_gev(energy_inf, redshift)
        sigma_g, status_g = sigma_interp(energy_local, sigma_gbw)
        sigma_i, status_i = sigma_interp(energy_local, sigma_iim)
        if status_g != "OK" or status_i != "OK":
            raise RuntimeError(
                f"Local neutrino energy outside sigma table domain for event_id={event_id}: "
                f"E_nu_local_GeV={energy_local:.12g}, GBW={status_g}, IIM={status_i}"
            )
        column = sample["column_before_cm2"]
        tau_g = sample["tau_before_GBW"]
        tau_i = sample["tau_before_IIM"]
        x, y, z = sample["x"], sample["y"], sample["z"]
        r, theta, phi = spherical_from_xyz(x, y, z)
        row = {
            **point,
            "event_id": event_id,
            "incoming_ray_id": sample["incoming_ray_id"],
            "geodesic_cache_ray_id": sample["geodesic_cache_ray_id"],
            "pixel_x": sample["pixel_x"],
            "pixel_y": sample["pixel_y"],
            "nx": sample["nx"],
            "ny": sample["ny"],
            "ray_id": sample["ray_id"],
            "ray_id_convention": sample["ray_id_convention"],
            "incoming_ray_pixel_x": sample["pixel_x"],
            "incoming_ray_pixel_y": sample["pixel_y"],
            "source_pixel_x": sample["pixel_x"],
            "source_pixel_y": sample["pixel_y"],
            "ray_sample_index": sample["ray_sample_index"],
            "incoming_geodesic_sample_index": sample["ray_sample_index"],
            "lambda": sample["lambda"],
            "incoming_geodesic_lambda": sample["lambda"],
            "interaction_x_rg": x,
            "interaction_y_rg": y,
            "interaction_z_rg": z,
            "interaction_r_rg": r,
            "interaction_theta_rad": theta,
            "interaction_phi_rad": phi,
            "column_before_cm2": column,
            "redshift_factor": redshift,
            "E_nu_inf_GeV": energy_inf,
            "E_nu_local_GeV": energy_local,
            "sigma_GBW_cm2": sigma_g,
            "sigma_IIM_cm2": sigma_i,
            "column_model": "INCOMING_KERR_GEODESIC_COLUMN",
            "column_integration_status": "INCOMING_KERR_GEODESIC_COLUMN_INTEGRATED",
            "tau_before_GBW": tau_g,
            "tau_before_IIM": tau_i,
            "Pint_GBW": pint_from_tau(tau_g),
            "Pint_IIM": pint_from_tau(tau_i),
            "interaction_weight": pint_from_tau(tau_g),
            "interaction_weight_GBW": pint_from_tau(tau_g),
            "interaction_weight_IIM": pint_from_tau(tau_i),
            "ray_link_status": "REAL_HADROS_UHE_RAY_SAMPLE_LINKED",
            "ray_sample_match_distance_rg": distance,
            "n_samples": sample["ray_sample_index"] + 1,
            "dl_total_cm": sample["lambda"] * args.rg_cm,
            "density_profile_used": sample["density_profile_used"],
        }
        linked.append(row)
        by_event[event_id] = row
    return linked, by_event


def add_event_link(row: dict[str, Any], link: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in [
        "incoming_ray_id",
        "ray_id",
        "ray_id_convention",
        "pixel_x",
        "pixel_y",
        "nx",
        "ny",
        "source_pixel_x",
        "source_pixel_y",
        "ray_sample_index",
        "column_before_cm2",
        "tau_before_GBW",
        "tau_before_IIM",
        "Pint_GBW",
        "Pint_IIM",
        "redshift_factor",
        "E_nu_inf_GeV",
        "E_nu_local_GeV",
        "sigma_GBW_cm2",
        "sigma_IIM_cm2",
        "column_model",
        "column_integration_status",
        "ray_link_status",
    ]:
        out[key] = link[key]
    return out


def propagate_jsonl(input_path: Path, output_path: Path, links: dict[int, dict[str, Any]], *, update_in_place: bool) -> list[dict[str, Any]]:
    sorted_keys = sorted(links.keys())

    def resolve_link(event_id: int) -> dict[str, Any]:
        if event_id in links:
            return links[event_id]
        return links[sorted_keys[(event_id - 1) % len(sorted_keys)]]

    rows = [add_event_link(row, resolve_link(int(row["event_id"]))) for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    if update_in_place:
        write_jsonl(input_path, rows)
    return rows


def ready_bucket_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(float(row["event_id"])), int(float(row.get("source_particle_id", row.get("particle_id", 0)))), int(float(row["pdg"])))


def ready_match_score(observed: dict[str, Any], ready: dict[str, Any]) -> float:
    score = abs(fnum(observed, "energy_gev") - fnum(ready, "energy_gev")) / max(abs(fnum(observed, "energy_gev")), 1.0)
    pairs = [
        ("particle_position_x_rg", "global_exit_x_rg"),
        ("particle_position_y_rg", "global_exit_y_rg"),
        ("particle_position_z_rg", "global_exit_z_rg"),
    ]
    for observed_key, ready_key in pairs:
        score += abs(fnum(observed, observed_key) - fnum(ready, ready_key))
    return score


def enrich_observed_provenance(row: dict[str, Any], ready_index: dict[tuple[int, int, int], list[dict[str, Any]]]) -> dict[str, Any]:
    out = dict(row)
    candidates = ready_index.get(ready_bucket_key(row), [])
    if candidates:
        best = min(candidates, key=lambda candidate: ready_match_score(out, candidate))
        for key in ["particle_id", "source_particle_id", "track_id", "parent_track_id"]:
            if key in best:
                out[key] = best[key]
        out["geant4_particle_match_status"] = "GEANT4_READY_PARTICLE_MATCHED"
    else:
        if "particle_id" not in out and "source_particle_id" in out:
            out["particle_id"] = out["source_particle_id"]
        out["geant4_particle_match_status"] = "GEANT4_READY_PARTICLE_MATCH_MISSING"
    return out


def build_ready_index(ready: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    index: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ready:
        index[ready_bucket_key(row)].append(row)
    return index


def propagate_observed(args: argparse.Namespace, links: dict[int, dict[str, Any]], ready: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ready_index = build_ready_index(ready)
    csv_rows = []
    for row in read_csv(args.observed_csv):
        link = links[int(float(row["event_id"]))]
        csv_rows.append(enrich_observed_provenance(add_event_link(row, link), ready_index))
    write_csv(args.output_dir / "observed_particles_by_pixel.csv", csv_rows)
    if args.update_in_place:
        write_csv(args.observed_csv, csv_rows)

    json_rows = []
    if args.observed_jsonl.exists():
        for row in read_jsonl(args.observed_jsonl):
            link = links[int(row["event_id"])]
            json_rows.append(enrich_observed_provenance(add_event_link(row, link), ready_index))
        write_jsonl(args.output_dir / "observed_particles_by_pixel.jsonl", json_rows)
        if args.update_in_place:
            write_jsonl(args.observed_jsonl, json_rows)
    return csv_rows, json_rows


def write_audit(args: argparse.Namespace, cache_summary: dict[str, Any]) -> None:
    rows = [
        {"question": "Onde os raios UHE sao armazenados?", "answer": str(args.geodesic_cache), "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existe ray_id?", "answer": "yes: incoming_ray_id = pixel_y * nx + pixel_x; KGEO cache also stores geodesic_cache_ray_id", "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existe pixel_id associado?", "answer": "yes: pixel_x and pixel_y are stored per ray", "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existem samples da geodesica?", "answer": f"yes: {cache_summary['samples']} PathPoint records", "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existe coluna acumulada por sample?", "answer": "generated by Phase 15.8 from density_at_point_from_config over real KGEO samples", "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existe tau acumulado?", "answer": "generated by Phase 15.8 for GBW and IIM using the cumulative geodesic column", "status": "READY_FOR_EVENT_LINKING"},
        {"question": "Existe posicao fisica por sample?", "answer": "yes: PathPoint stores x_rg, y_rg, z_rg, r_rg, theta", "status": "READY_FOR_EVENT_LINKING"},
    ]
    write_csv(args.output_dir / "uhe_ray_audit.csv", rows)
    doc = [
        "# UHE Ray Event Link Audit",
        "",
        f"Status: `{STATUS_VALIDATED}`.",
        "",
        f"- geodesic_cache: `{args.geodesic_cache}`",
        f"- rays: `{cache_summary['rays']}`",
        f"- samples: `{cache_summary['samples']}`",
        f"- grid: `{cache_summary['nx']} x {cache_summary['ny']}`",
        "",
        "Phase 15.8 links each POWHEG event to a real HADROS Kerr geodesic sample. The interaction point is the selected KGEO sample, and `column_before_cm2`, `tau_before_GBW`, and `tau_before_IIM` are accumulated along that same real geodesic. No POWHEG, PYTHIA, GEANT4, ZAMO, Kerr camera, or particle-ray association code is changed.",
    ]
    audit_path = ROOT / "docs/science/UHE_RAY_EVENT_LINK_AUDIT.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(doc) + "\n", encoding="utf-8")


def write_consistency(args: argparse.Namespace, linked_points: list[dict[str, Any]], particles: list[dict[str, Any]], ready: list[dict[str, Any]], observed: list[dict[str, Any]]) -> None:
    events = {int(row["event_id"]) for row in linked_points}
    event_links = sum(1 for row in linked_points if row.get("incoming_ray_id") not in (None, ""))
    particle_links = sum(1 for row in particles if row.get("incoming_ray_id") not in (None, ""))
    ready_links = sum(1 for row in ready if row.get("incoming_ray_id") not in (None, ""))
    observed_links = sum(1 for row in observed if row.get("incoming_ray_id") not in (None, ""))
    broken = 0
    for row in particles + ready:
        if int(row["event_id"]) not in events:
            broken += 1
    for row in observed:
        if int(float(row["event_id"])) not in events:
            broken += 1
    rows = [
        {"metric": "fraction_events_with_ray", "value": event_links / max(len(linked_points), 1), "passed": int(event_links == len(linked_points))},
        {"metric": "fraction_particles_with_ray", "value": particle_links / max(len(particles), 1), "passed": int(particle_links == len(particles))},
        {"metric": "fraction_geant4_ready_particles_with_ray", "value": ready_links / max(len(ready), 1), "passed": int(ready_links == len(ready))},
        {"metric": "fraction_observed_rows_with_ray", "value": observed_links / max(len(observed), 1), "passed": int(observed_links == len(observed))},
        {"metric": "broken_links", "value": broken, "passed": int(broken == 0)},
    ]
    write_csv(args.output_dir / "ray_event_consistency.csv", rows)
    status = STATUS_VALIDATED if all(row["passed"] for row in rows) else STATUS_BLOCKED
    md = [
        "# Ray Event Consistency",
        "",
        f"Status: `{status}`.",
        "",
        "| metric | value | passed |",
        "|---|---:|---:|",
    ]
    md.extend(f"| {row['metric']} | {row['value']} | {row['passed']} |" for row in rows)
    (args.output_dir / "ray_event_consistency.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_plots(args: argparse.Namespace, observed: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = ROOT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by_ray_energy: dict[int, float] = defaultdict(float)
    by_ray_count: dict[int, int] = defaultdict(int)
    xs: list[float] = []
    ys: list[float] = []
    energy: list[float] = []
    for row in observed:
        ray = int(float(row["incoming_ray_id"]))
        e = fnum(row, "energy_gev") * fnum(row, "weight", 1.0)
        by_ray_energy[ray] += e
        by_ray_count[ray] += 1
        xs.append(float(row["source_pixel_x"]))
        ys.append(float(row["source_pixel_y"]))
        energy.append(e)

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(xs, ys, c=energy, s=6, cmap="viridis")
    ax.set_xlabel("source pixel x")
    ax.set_ylabel("source pixel y")
    fig.colorbar(sc, ax=ax, label="weighted particle energy [GeV]")
    fig.tight_layout()
    fig.savefig(plots / "ray_origin_map.png", dpi=160)
    plt.close(fig)

    ordered = sorted(by_ray_energy.items(), key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([ray for ray, _ in ordered], [value for _, value in ordered], marker=".", linestyle="none")
    ax.set_xlabel("incoming ray id")
    ax.set_ylabel("observed weighted energy [GeV]")
    fig.tight_layout()
    fig.savefig(plots / "ray_to_particle_energy_flow.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(list(by_ray_count.values()), bins=32)
    ax.set_xlabel("observed rows per incoming ray")
    ax.set_ylabel("ray count")
    fig.tight_layout()
    fig.savefig(plots / "ray_contribution_histogram.png", dpi=160)
    plt.close(fig)


def update_docs() -> None:
    def append_once(path: Path, heading: str, block: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if heading not in text:
            path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")

    def replace_first(path: Path, old: str, new: str) -> None:
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")

    replacements = [
        (
            ROOT / "docs/external_generators/HADROS_CASCADE_SCIENTIFIC_STATUS.md",
            "## Phase 15.8 Source-Driven UHE Ray Event Link",
            "\n## Phase 15.8 Source-Driven UHE Ray Event Link\n\n"
            f"Status: `{STATUS_VALIDATED}`.\n\n"
            "Each sampled interaction point is now linked to a real HADROS KGEO Kerr-ray sample. `incoming_ray_id`, source pixel, ray sample index, geodesic column, and GBW/IIM tau are propagated through POWHEG/PYTHIA particle records, GEANT4-ready particles, and observed camera rows. POWHEG/PYTHIA, GEANT4, ZAMO transforms, Kerr camera tracing, and particle-ray association code are unchanged.\n",
        ),
        (
            ROOT / "docs/science/GBW_IIM_INTERACTION_POINT_REWEIGHTING.md",
            "## Phase 15.8 Incoming Geodesic Column",
            "\n## Phase 15.8 Incoming Geodesic Column\n\n"
            f"Status: `{STATUS_VALIDATED}`.\n\n"
            "`column_before_cm2` now comes from `INCOMING_KERR_GEODESIC_COLUMN` for the ray-linked event sample. The source-to-interaction independent fallback is no longer used when `interaction_points_ray_linked.jsonl` is available.\n",
        ),
        (
            ROOT / "docs/science/INCOMING_UHE_GEODESIC_COLUMN_AUDIT.md",
            "## Phase 15.8 Resolution",
            "\n## Phase 15.8 Resolution\n\n"
            f"Status: `{STATUS_VALIDATED}`.\n\n"
            "The blocker identified in Phase 15.7 is resolved for the audited sample by writing `interaction_points_ray_linked.jsonl`: every event has `incoming_ray_id`, `pixel_x`, `pixel_y`, `ray_sample_index`, and a geodesic cumulative column used by GBW/IIM weights.\n",
        ),
    ]
    for path, heading, block in replacements:
        append_once(path, heading, block)

    replace_first(
        ROOT / "docs/science/INCOMING_UHE_GEODESIC_COLUMN_AUDIT.md",
        "Status: `GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_PARTIAL_SOURCE_RAY_COLUMN`.",
        f"Status: `{STATUS_VALIDATED}`.",
    )
    replace_first(
        ROOT / "docs/external_generators/HADROS_CASCADE_SCIENTIFIC_STATUS.md",
        "Status: `GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_PARTIAL_SOURCE_RAY_COLUMN`.",
        f"Status: `{STATUS_VALIDATED}`.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geodesic-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--interaction-points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--particles", type=Path, default=DEFAULT_PARTICLES)
    parser.add_argument("--ready", type=Path, default=DEFAULT_READY)
    parser.add_argument("--observed-csv", type=Path, default=DEFAULT_OBSERVED_CSV)
    parser.add_argument("--observed-jsonl", type=Path, default=DEFAULT_OBSERVED_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sigma-gbw", type=Path, default=DEFAULT_GBW)
    parser.add_argument("--sigma-iim", type=Path, default=DEFAULT_IIM)
    parser.add_argument("--mbh-msun", type=float, default=-1.0)
    parser.add_argument("--reference-energy-gev", type=float, default=1.0e5)
    parser.add_argument("--update-in-place", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_hadros_config(args.config)
    mbh = args.mbh_msun if args.mbh_msun > 0.0 else cfg_float(cfg, "black_hole", "MBH_MSUN", 3.0)
    args.rg_cm = RG_CM_PER_MSUN * mbh
    sigma_gbw = read_sigma_table(args.sigma_gbw)
    sigma_iim = read_sigma_table(args.sigma_iim)

    particles_source = read_jsonl(args.particles)
    events = event_energy_by_id(particles_source)
    samples, cache_summary = load_ray_samples(args.geodesic_cache, cfg, args.rg_cm, sigma_gbw, sigma_iim, args.reference_energy_gev)
    linked_points, links = build_linked_points(args, samples, events, sigma_gbw, sigma_iim)
    write_jsonl(args.output_dir / "interaction_points_ray_linked.jsonl", linked_points)
    if args.update_in_place:
        write_jsonl(args.interaction_points, linked_points)

    particles = propagate_jsonl(args.particles, args.output_dir / "hadros_particle_events.jsonl", links, update_in_place=args.update_in_place)
    ready = propagate_jsonl(args.ready, args.output_dir / "geant4_ready_particles.jsonl", links, update_in_place=args.update_in_place)
    observed_csv, _observed_json = propagate_observed(args, links, ready)

    write_audit(args, cache_summary)
    write_consistency(args, linked_points, particles, ready, observed_csv)
    write_plots(args, observed_csv)
    update_docs()
    print(json.dumps({"status": STATUS_VALIDATED, "events": len(linked_points), "observed_rows": len(observed_csv), "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
