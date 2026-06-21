#!/usr/bin/env python3
"""Run real GEANT4 local-box transport in resumable conservative batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import configparser
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def event_id(row: dict[str, Any]) -> int:
    return int(row.get("event_id", 0) or 0)


def particle_energy(row: dict[str, Any]) -> float:
    return float(row.get("energy_gev", 0.0) or 0.0)


def job_energy(rows: list[dict[str, Any]]) -> float:
    return sum(particle_energy(row) for row in rows)


def build_app(root: Path) -> None:
    subprocess.run(["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"], cwd=root, check=True)


def rg_cm_from_mbh(mbh_msun: float) -> float:
    return 6.67430e-8 * mbh_msun * 1.98847e33 / (2.99792458e10 * 2.99792458e10)


def load_mbh_msun(root: Path) -> float:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(root / "config.ini")
    for section in parser.sections():
        if parser.has_option(section, "MBH_MSUN"):
            try:
                return float(parser.get(section, "MBH_MSUN"))
            except ValueError:
                pass
    return 2.0


def spherical_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0 or not math.isfinite(r):
        return 0.0, 0.0, 0.0
    return r, math.acos(max(-1.0, min(1.0, z / r))), math.atan2(y, x)


def load_interaction_positions(path: Path, rg_cm: float) -> dict[int, dict[str, Any]]:
    positions: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return positions
    for row in read_jsonl(path):
        event = int(row.get("event_id", row.get("primary", {}).get("event_id", 0)) or 0)
        point = row.get("point") if isinstance(row.get("point"), dict) else row
        if {"x", "y", "z"}.issubset(point):
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            z = float(point.get("z", 0.0))
        elif {"x_cm", "y_cm", "z_cm"}.issubset(point):
            x = float(point.get("x_cm", 0.0)) / rg_cm
            y = float(point.get("y_cm", 0.0)) / rg_cm
            z = float(point.get("z_cm", 0.0)) / rg_cm
        else:
            continue
        r, theta, phi = spherical_from_xyz(x, y, z)
        positions[event] = {
            "x": x, "y": y, "z": z, "r": r, "theta": theta, "phi": phi,
            "x_cm": point.get("x_cm", ""), "y_cm": point.get("y_cm", ""), "z_cm": point.get("z_cm", ""),
            "origin_status": "INTERACTION_POINT_POSITION",
            "region_label": point.get("region_label", point.get("region_class", "")),
        }
    return positions


def annotate_positions(rows: list[dict[str, Any]], positions: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        pos = positions.get(event_id(out))
        if pos:
            for key, value in pos.items():
                out[key] = value
        else:
            out.setdefault("origin_status", "MISSING_POSITION")
        annotated.append(out)
    return annotated


def make_jobs(rows: list[dict[str, Any]], mode: str, chunk_size: int) -> list[tuple[str, list[dict[str, Any]]]]:
    if mode == "one_particle_per_process":
        return [(f"job_{idx:06d}_event_{event_id(row)}_pdg_{int(row.get('pdg_id', row.get('pdg', 0)) or 0)}", [row]) for idx, row in enumerate(rows)]
    if mode == "one_event_per_process":
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(event_id(row), []).append(row)
        return [(f"job_{idx:06d}_event_{eid}", grouped[eid]) for idx, eid in enumerate(sorted(grouped))]
    if mode == "chunked_particles":
        jobs = []
        for start in range(0, len(rows), max(chunk_size, 1)):
            jobs.append((f"job_{len(jobs):06d}_chunk_{start}_{start + len(rows[start:start + chunk_size])}", rows[start:start + chunk_size]))
        return jobs
    raise ValueError(f"unknown batch mode: {mode}")


def status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def load_status(job_dir: Path) -> dict[str, Any] | None:
    path = status_path(job_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(job_dir: Path, status: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path(job_dir).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_returncode(code: int, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if code < 0:
        return "CRASH"
    if code == 0:
        return "PASS"
    return "FAIL"


def run_job(root: Path, job_name: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.output_dir / "geant4_batch_work" / job_name
    existing = load_status(work_dir)
    if existing and existing.get("status") == "PASS":
        return {**existing, "skipped_existing_pass": True}
    if existing and existing.get("status") in {"FAIL", "CRASH", "TIMEOUT"} and not args.retry_failed:
        return {**existing, "skipped_existing_failed": True}
    retries = int(existing.get("retries", 0)) + 1 if existing else 0
    if existing and retries > args.max_retries:
        return {**existing, "skipped_max_retries": True}

    input_path = work_dir / "input.jsonl"
    outputs_dir = work_dir / "outputs"
    write_jsonl(input_path, rows)
    cmd = [
        str(root / "build" / "cascade_geant4_local_box"),
        str(input_path),
        str(outputs_dir),
        f"{args.box_size_cm:.17g}",
        f"{args.density_g_cm3:.17g}",
        args.physics_list,
        args.material,
        "geant4",
        "--geant4-safety-mode",
        args.geant4_safety_mode,
        "--uhe-transport-policy",
        args.uhe_transport_policy,
        "--geant4-hadron-max-kinetic-gev",
        f"{args.geant4_hadron_max_kinetic_gev:.17g}",
        "--geant4-lepton-max-kinetic-gev",
        f"{args.geant4_lepton_max_kinetic_gev:.17g}",
        "--geant4-photon-max-kinetic-gev",
        f"{args.geant4_photon_max_kinetic_gev:.17g}",
        "--energy-convention",
        args.energy_convention,
    ]
    if args.batch_mode != "one_particle_per_process":
        cmd.append("--geant4-one-particle-per-run")
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout, check=False)
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    runtime = time.monotonic() - start
    (work_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (work_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    status = classify_returncode(returncode, timed_out)
    result = {
        "job": job_name,
        "status": status,
        "returncode": returncode,
        "runtime_s": runtime,
        "n_particles": len(rows),
        "input_energy_gev": sum(particle_energy(row) for row in rows),
        "event_ids": sorted({event_id(row) for row in rows}),
        "command": cmd,
        "retries": retries,
    }
    save_status(work_dir, result)
    return result


def select_jobs_for_target(
    output_dir: Path,
    all_jobs: list[tuple[str, list[dict[str, Any]]]],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[dict[str, Any]], float, float]:
    total_energy = job_energy([row for _, rows in all_jobs for row in rows])
    target_fraction = max(0.0, min(float(args.target_processed_energy_fraction or 0.0), 1.0))
    target_energy = target_fraction * total_energy
    existing_statuses: list[dict[str, Any]] = []
    achieved_energy = 0.0
    jobs_to_consider = list(all_jobs)
    if args.prioritize_energy_desc:
        jobs_to_consider.sort(key=lambda item: job_energy(item[1]), reverse=True)

    if target_fraction <= 0.0:
        jobs = jobs_to_consider
        if args.max_jobs > 0:
            jobs = jobs[: args.max_jobs]
        return jobs, existing_statuses, target_fraction, target_energy

    selected: list[tuple[str, list[dict[str, Any]]]] = []
    for job_name, rows in jobs_to_consider:
        status = load_status(output_dir / "geant4_batch_work" / job_name)
        if status and status.get("status") == "PASS":
            existing_statuses.append(status)
            achieved_energy += job_energy(rows)
            continue
        if status and status.get("status") in {"FAIL", "CRASH", "TIMEOUT"} and not args.retry_failed:
            existing_statuses.append(status)
            continue
        if achieved_energy + sum(job_energy(item[1]) for item in selected) >= target_energy:
            continue
        selected.append((job_name, rows))
        if args.max_jobs > 0 and len(selected) >= args.max_jobs:
            break

    return selected, existing_statuses, target_fraction, target_energy


def aggregate(output_dir: Path, jobs: list[tuple[str, list[dict[str, Any]]]], statuses: list[dict[str, Any]], positions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    status_by_job = {row["job"]: row for row in statuses}
    event_buckets: dict[int, dict[str, float]] = {}
    escaped: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    processed_energy = 0.0
    unprocessed_energy = 0.0
    for job_name, rows in jobs:
        status = status_by_job.get(job_name, {})
        if status.get("status") != "PASS":
            unprocessed_energy += sum(particle_energy(row) for row in rows)
            continue
        processed_energy += sum(particle_energy(row) for row in rows)
        job_outputs = output_dir / "geant4_batch_work" / job_name / "outputs"
        for budget_row in read_csv(job_outputs / "geant4_energy_budget.csv"):
            eid = int(float(budget_row.get("event_id", 0) or 0))
            bucket = event_buckets.setdefault(eid, {
                "event_id": float(eid),
                "input_energy_gev": 0.0,
                "deposited_energy_gev": 0.0,
                "escaped_energy_gev": 0.0,
                "invisible_energy_gev": 0.0,
                "untracked_energy_gev": 0.0,
                "unsupported_uhe_energy_gev": 0.0,
                "escaped_unsupported_uhe_energy_gev": 0.0,
                "n_unsupported_uhe_particles": 0.0,
                "escaped_particle_count": 0.0,
            })
            for key in bucket:
                if key != "event_id":
                    bucket[key] += f(budget_row, key)
        escaped.extend(annotate_positions(read_jsonl(job_outputs / "geant4_escaped_particles.jsonl"), positions))
        unsupported.extend(annotate_positions(read_jsonl(job_outputs / "geant4_unsupported_uhe_particles.jsonl"), positions))

    budget_rows = []
    for eid in sorted(event_buckets):
        row = event_buckets[eid]
        accounted = row["deposited_energy_gev"] + row["escaped_energy_gev"] + row["invisible_energy_gev"] + row["untracked_energy_gev"] + row["escaped_unsupported_uhe_energy_gev"]
        budget_rows.append({
            "event_id": int(eid),
            **{key: row[key] for key in row if key != "event_id"},
            "accounted_energy_gev": accounted,
            "closure_error_gev": accounted - row["input_energy_gev"],
            "energy_convention": "total",
            "uhe_transport_policy": "skip_to_escaped",
        })

    with (output_dir / "geant4_energy_budget.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "event_id", "input_energy_gev", "deposited_energy_gev", "escaped_energy_gev",
            "invisible_energy_gev", "untracked_energy_gev", "unsupported_uhe_energy_gev",
            "escaped_unsupported_uhe_energy_gev", "n_unsupported_uhe_particles",
            "accounted_energy_gev", "closure_error_gev", "escaped_particle_count",
            "energy_convention", "uhe_transport_policy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(budget_rows)
    with (output_dir / "geant4_cascade_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in budget_rows:
            payload = dict(row)
            payload["backend"] = "Geant4RealResumableBatchRunner"
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    write_jsonl(output_dir / "geant4_escaped_particles.jsonl", escaped)
    write_jsonl(output_dir / "geant4_unsupported_uhe_particles.jsonl", unsupported)

    n_pass = sum(1 for row in statuses if row.get("status") == "PASS")
    total_input = sum(sum(particle_energy(row) for row in rows) for _, rows in jobs)
    deposited = sum(float(row["deposited_energy_gev"]) for row in budget_rows)
    escaped_energy = sum(float(row["escaped_energy_gev"]) for row in budget_rows)
    invisible = sum(float(row["invisible_energy_gev"]) for row in budget_rows)
    untracked = sum(float(row["untracked_energy_gev"]) for row in budget_rows)
    escaped_unsupported = sum(float(row["escaped_unsupported_uhe_energy_gev"]) for row in budget_rows)
    processed_fraction = processed_energy / total_input if total_input > 0.0 else 0.0
    summary = {
        "status": "COMPLETE" if n_pass == len(jobs) else "PARTIAL",
        "jobs_total": len(jobs),
        "jobs_pass": n_pass,
        "jobs_failed": len(jobs) - n_pass,
        "jobs_run_this_invocation": len(statuses),
        "total_input_energy_gev": total_input,
        "processed_energy_gev": processed_energy,
        "processed_energy_fraction": processed_fraction,
        "unprocessed_energy_gev": unprocessed_energy,
        "deposited_energy_gev": deposited,
        "escaped_energy_gev": escaped_energy,
        "invisible_energy_gev": invisible,
        "untracked_energy_gev": untracked,
        "escaped_unsupported_uhe_energy_gev": escaped_unsupported,
        "closure_error_gev": deposited + escaped_energy + invisible + untracked + escaped_unsupported - processed_energy,
    }
    return summary


def write_status_csv(output_dir: Path, statuses: list[dict[str, Any]]) -> None:
    with (output_dir / "geant4_batch_status.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["job", "status", "returncode", "runtime_s", "n_particles", "input_energy_gev", "retries"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in statuses:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(output_dir: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# GEANT4 Real Resumable Batch Summary",
        "",
        "This is a slow, checkpointed real-GEANT4 execution route for PYTHIA-rich lists.",
        "It is operational infrastructure, not a new physics model.",
        "",
        f"- status: `{summary['status']}`",
        f"- batch_mode: `{args.batch_mode}`",
        f"- workers: `{args.workers}`",
        f"- jobs_total: `{summary['jobs_total']}`",
        f"- jobs_pass: `{summary['jobs_pass']}`",
        f"- jobs_failed: `{summary['jobs_failed']}`",
        f"- jobs_run_this_invocation: `{summary['jobs_run_this_invocation']}`",
        f"- target_processed_energy_fraction: `{summary.get('target_processed_energy_fraction', 0.0):.12g}`",
        f"- achieved_processed_energy_fraction: `{summary['processed_energy_fraction']:.12g}`",
        f"- processed_energy_gev: `{summary['processed_energy_gev']:.12g}`",
        f"- unprocessed_energy_gev: `{summary['unprocessed_energy_gev']:.12g}`",
        f"- deposited_energy_gev: `{summary['deposited_energy_gev']:.12g}`",
        f"- escaped_energy_gev: `{summary['escaped_energy_gev']:.12g}`",
        f"- invisible_energy_gev: `{summary['invisible_energy_gev']:.12g}`",
        f"- untracked_energy_gev: `{summary['untracked_energy_gev']:.12g}`",
        f"- escaped_unsupported_uhe_energy_gev: `{summary['escaped_unsupported_uhe_energy_gev']:.12g}`",
        f"- closure_error_gev: `{summary['closure_error_gev']:.12e}`",
    ]
    if summary["status"] != "COMPLETE":
        message = "PARTIAL: downstream final images should not be generated unless explicitly allowed."
        if summary["status"] == "PARTIAL_ENERGY_TARGET_REACHED":
            message = "PARTIAL_ENERGY_TARGET_REACHED: this is an explicitly energy-sampled partial run."
        lines.extend(["", message])
    (output_dir / "geant4_batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondaries", type=Path, default=Path("output/cascade/pythia_secondaries.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--batch-mode", choices=["one_particle_per_process", "one_event_per_process", "chunked_particles"], default="one_particle_per_process")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--target-processed-energy-fraction", type=float, default=0.0)
    parser.add_argument("--prioritize-energy-desc", action="store_true")
    parser.add_argument("--allow-partial-exit-zero", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--box-size-cm", type=float, default=100.0)
    parser.add_argument("--density-g-cm3", type=float, default=1.0)
    parser.add_argument("--physics-list", choices=["FTFP_BERT", "QGSP_BERT"], default="FTFP_BERT")
    parser.add_argument("--material", choices=["hydrogen", "water"], default="hydrogen")
    parser.add_argument("--geant4-safety-mode", choices=["off", "strict"], default="strict")
    parser.add_argument("--uhe-transport-policy", choices=["error", "skip_to_escaped", "split_energy_proxy"], default="skip_to_escaped")
    parser.add_argument("--geant4-hadron-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-lepton-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--geant4-photon-max-kinetic-gev", type=float, default=1.0e5)
    parser.add_argument("--energy-convention", choices=["total", "kinetic"], default="total")
    parser.add_argument("--interaction-points", type=Path, default=None)
    parser.add_argument("--mbh-msun", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    args.output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.secondaries = (root / args.secondaries).resolve() if not args.secondaries.is_absolute() else args.secondaries
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("geant4-config") is None:
        (args.output_dir / "geant4_batch_summary.md").write_text("SKIP: geant4-config not found.\n", encoding="utf-8")
        print("SKIP: geant4-config not found")
        return 0
    rows = read_jsonl(args.secondaries)
    if args.interaction_points is None:
        candidate = args.output_dir / "interaction_points.jsonl"
        if candidate.exists():
            args.interaction_points = candidate
    if args.interaction_points is not None and not args.interaction_points.is_absolute():
        args.interaction_points = (root / args.interaction_points).resolve()
    mbh = args.mbh_msun if args.mbh_msun is not None else load_mbh_msun(root)
    positions = load_interaction_positions(args.interaction_points, rg_cm_from_mbh(mbh)) if args.interaction_points else {}
    build_app(root)
    all_jobs = make_jobs(rows, args.batch_mode, args.chunk_size)
    jobs, statuses, target_fraction, target_energy = select_jobs_for_target(args.output_dir, all_jobs, args)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(run_job, root, name, job_rows, args): name for name, job_rows in jobs}
        for future in as_completed(future_map):
            statuses.append(future.result())
    statuses.sort(key=lambda row: str(row.get("job", "")))
    write_status_csv(args.output_dir, statuses)
    summary = aggregate(args.output_dir, all_jobs, statuses, positions)
    summary["target_processed_energy_fraction"] = target_fraction
    summary["target_processed_energy_gev"] = target_energy
    if (
        target_fraction > 0.0
        and summary["status"] != "COMPLETE"
        and summary["processed_energy_fraction"] + 1.0e-15 >= target_fraction
        and not any(row.get("status") in {"FAIL", "CRASH", "TIMEOUT"} for row in statuses)
    ):
        summary["status"] = "PARTIAL_ENERGY_TARGET_REACHED"
    write_summary(args.output_dir, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETE" or args.allow_partial_exit_zero else 2


if __name__ == "__main__":
    raise SystemExit(main())
