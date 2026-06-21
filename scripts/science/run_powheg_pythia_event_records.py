#!/usr/bin/env python3
"""Run-local POWHEG/PYTHIA event-record handoff for config-web E2E runs.

This wrapper intentionally does not synthesize particles and does not call the
debug PYTHIA e+e- proxy. It converts real POWHEG/PYTHIA event-record dumps when
they are present in the run directory. If no run-local physical event-record
input is available, it stops at the event-generation boundary with a clear
blocked report.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "science"))

import powheg_pythia_to_hadros_particles as converter  # noqa: E402


STATUS_READY = "POWHEG_PYTHIA_EVENT_RECORDS_RUN_LOCAL_READY"
STATUS_DEPENDENCY_MISSING = "POWHEG_PYTHIA_EVENT_RECORDS_BLOCKED_DEPENDENCY_MISSING"
STATUS_RUNTIME_READY = "POWHEG_PYTHIA_RUNTIME_READY"
STATUS_RUNTIME_PARTIAL_LHE_ONLY = "POWHEG_PYTHIA_RUNTIME_PARTIAL_LHE_ONLY"
STATUS_MISSING_PYTHIA8 = "POWHEG_PYTHIA_RUNTIME_BLOCKED_MISSING_PYTHIA8"
STATUS_MISSING_POWHEG = "POWHEG_PYTHIA_RUNTIME_BLOCKED_MISSING_POWHEG"
STATUS_MISSING_CARDS = "POWHEG_PYTHIA_RUNTIME_BLOCKED_MISSING_CARDS"
STATUS_LHE_MISSING_CARDS = "POWHEG_LHE_BLOCKED_MISSING_CARDS"

DEFAULT_PATTERNS = [
    ("CC", "event_records/cc_event_record.txt"),
    ("NC", "event_records/nc_event_record.txt"),
    ("CC", "powheg_pythia_event_records/cc_event_record.txt"),
    ("NC", "powheg_pythia_event_records/nc_event_record.txt"),
    ("CC", "powheg/cc_event_record.txt"),
    ("NC", "powheg/nc_event_record.txt"),
]

DEFAULT_LHE_PATTERNS = [
    ("CC", "powheg/pwgevents.lhe"),
    ("NC", "powheg/pwgevents.lhe"),
    ("CC", "powheg/cc/pwgevents.lhe"),
    ("NC", "powheg/nc/pwgevents.lhe"),
    ("CC", "powheg/CC/pwgevents.lhe"),
    ("NC", "powheg/NC/pwgevents.lhe"),
    ("CC", "lhe/cc/pwgevents.lhe"),
    ("NC", "lhe/nc/pwgevents.lhe"),
    ("CC", "cc/pwgevents.lhe"),
    ("NC", "nc/pwgevents.lhe"),
]

RAY_FIELDS = [
    "incoming_ray_id",
    "geodesic_cache_ray_id",
    "pixel_x",
    "pixel_y",
    "source_pixel_x",
    "source_pixel_y",
    "ray_sample_index",
    "incoming_geodesic_sample_index",
    "interaction_x_rg",
    "interaction_y_rg",
    "interaction_z_rg",
    "interaction_r_rg",
    "interaction_theta_rad",
    "interaction_phi_rad",
    "column_before_cm2",
    "column_model",
    "column_integration_status",
    "tau_before_GBW",
    "tau_before_IIM",
    "Pint_GBW",
    "Pint_IIM",
    "ray_link_status",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_input_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"event-record input must be CC:/path or NC:/path, got {spec!r}")
        interaction, path_text = spec.split(":", 1)
        interaction = interaction.strip().upper()
        if interaction not in {"CC", "NC"}:
            raise SystemExit(f"event-record interaction must be CC or NC, got {interaction!r}")
        out.append((interaction, Path(path_text).expanduser()))
    return out


def wanted_interactions(mode: str) -> set[str]:
    mode = str(mode or "both").strip().lower()
    if mode == "cc":
        return {"CC"}
    if mode == "nc":
        return {"NC"}
    return {"CC", "NC"}


def autodetect_executable(name: str, explicit: str = "") -> str:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.exists() else explicit
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates = [
        home / "micromamba" / "envs" / "hadros-cascade" / "bin" / name,
        home / "micromamba" / "pkgs",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            matches = sorted(candidate.glob(f"*/bin/{name}"))
            if matches:
                return str(matches[0])
    return ""


def pythia_env(pythia8_config: str) -> dict[str, str]:
    env = dict(os.environ)
    if pythia8_config:
        bin_dir = str(Path(pythia8_config).expanduser().parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        try:
            data_dir = subprocess.check_output([pythia8_config, "--datadir"], text=True).strip()
            xml_dir = Path(data_dir) / "xmldoc"
            env["PYTHIA8DATA"] = str(xml_dir if xml_dir.exists() else Path(data_dir))
        except Exception:
            pass
    return env


def discover_inputs(output_dir: Path, explicit: list[str]) -> list[tuple[str, Path]]:
    inputs = parse_input_specs(explicit)
    if inputs:
        return inputs
    found: list[tuple[str, Path]] = []
    for interaction, rel in DEFAULT_PATTERNS:
        path = output_dir / rel
        if path.exists() and path.stat().st_size > 0:
            found.append((interaction, path))
    return found


def discover_lhe_inputs(output_dir: Path, explicit: list[str], mode: str) -> list[tuple[str, Path]]:
    wanted = wanted_interactions(mode)
    inputs = [(interaction, path) for interaction, path in parse_input_specs(explicit) if interaction in wanted]
    if inputs:
        return inputs
    found: list[tuple[str, Path]] = []
    for interaction, rel in DEFAULT_LHE_PATTERNS:
        if interaction not in wanted:
            continue
        path = output_dir / rel
        if path.exists() and path.stat().st_size > 0:
            found.append((interaction, path))
    return found


def run_local_lhe(args: argparse.Namespace) -> list[tuple[str, Path]]:
    workdir = Path(args.powheg_workdir) if args.powheg_workdir else args.output_dir / "powheg"
    workdir.mkdir(parents=True, exist_ok=True)
    if args.lhe_file:
        source = Path(args.lhe_file).expanduser()
        if not source.exists() or source.stat().st_size == 0:
            raise SystemExit(f"--lhe-file does not exist or is empty: {source}")
        target = workdir / "pwgevents.lhe"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        interaction = "CC" if args.mode == "cc" else "NC" if args.mode == "nc" else "CC"
        return [(interaction, target)]
    existing = workdir / "pwgevents.lhe"
    if args.reuse_existing_lhe and existing.exists() and existing.stat().st_size > 0:
        interaction = "CC" if args.mode == "cc" else "NC" if args.mode == "nc" else "CC"
        return [(interaction, existing)]
    return []


def ray_links(path: Path) -> dict[int, dict[str, Any]]:
    links: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        try:
            links[int(row["event_id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return links


def enrich_rows(path: Path, links: dict[int, dict[str, Any]], *, csv_file: bool) -> int:
    rows: list[dict[str, Any]]
    if csv_file:
        rows = [dict(row) for row in read_csv(path)]
    else:
        rows = read_jsonl(path)
    if not rows or not links:
        return 0
    enriched = 0
    for row in rows:
        try:
            event_id = int(float(row["event_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        link = links.get(event_id)
        if not link:
            continue
        for field in RAY_FIELDS:
            if field in link:
                row[field] = link[field]
        enriched += 1
    if csv_file:
        write_csv(path, rows)
    else:
        write_jsonl(path, rows)
    return enriched


def write_dependency_blocked(args: argparse.Namespace, reason: str, lhe_inputs: list[tuple[str, Path]], status: str = STATUS_DEPENDENCY_MISSING) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tried = [str(args.output_dir / rel) for _interaction, rel in DEFAULT_LHE_PATTERNS]
    pythia8_config = autodetect_executable("pythia8-config", args.pythia8_config)
    pwhg_main = autodetect_executable("pwhg_main", args.pwhg_main)
    powheg_workdir = Path(args.powheg_workdir) if args.powheg_workdir else args.output_dir / "powheg"
    rows = [
        {
            "status": status,
            "reason": reason,
            "output_dir": str(args.output_dir),
            "run_name": args.run_name,
            "n_events": args.n_events,
            "mode": args.mode,
            "dis_mode": args.dis_mode,
            "lhe_inputs": ";".join(f"{kind}:{path}" for kind, path in lhe_inputs),
            "pythia8_config": pythia8_config,
            "pwhg_main": pwhg_main,
            "powheg_workdir": str(powheg_workdir),
        }
    ]
    write_csv(args.output_dir / "powheg_pythia_event_record_status.csv", rows)
    lines = [
        "# POWHEG/PYTHIA Event Records",
        "",
        f"Status: `{status}`.",
        "",
        reason,
        "",
        "No proxy, debug e+e- PYTHIA route, synthetic particle source, or particle-to-screen fallback was used.",
        "",
        "## Run-Local LHE Paths Checked",
        "",
    ]
    lines.extend(f"- `{path}`" for path in tried)
    lines.extend(
        [
            "",
            "## Required Dependencies",
            "",
            f"- `pythia8-config`: `{pythia8_config or 'missing'}`",
            f"- `pwhg_main`: `{pwhg_main or 'missing'}`",
            f"- `powheg_workdir`: `{powheg_workdir}`",
            "",
            "The versioned HADROS driver is `apps/powheg_pythia_hadros_driver.cpp` and consumes POWHEG LHE through PYTHIA8 `Beams:frameType = 4`.",
        ]
    )
    (args.output_dir / "powheg_pythia_event_records.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ready_report(args: argparse.Namespace, stats: dict[str, int], enriched: dict[str, int], inputs: list[tuple[str, Path]]) -> None:
    write_csv(
        args.output_dir / "powheg_pythia_event_record_status.csv",
        [
            {
                "status": STATUS_READY,
                "reason": "Run-local real POWHEG/PYTHIA event records were converted to HADROS particle records.",
                "output_dir": str(args.output_dir),
                "run_name": args.run_name,
                "n_events": args.n_events,
                "mode": args.mode,
                "dis_mode": args.dis_mode,
                "events": stats.get("events", 0),
                "particles": stats.get("particles", 0),
                "inputs": ";".join(f"{interaction}:{path}" for interaction, path in inputs),
            }
        ],
    )
    lines = [
        "# POWHEG/PYTHIA Event Records",
        "",
        f"Status: `{STATUS_READY}`.",
        "",
        "No proxy, debug e+e- PYTHIA route, synthetic particle source, or particle-to-screen fallback was used.",
        "",
        f"- run_name: `{args.run_name}`",
        f"- output_dir: `{args.output_dir}`",
        f"- requested_n_events: `{args.n_events}`",
        f"- mode: `{args.mode}`",
        f"- dis_mode: `{args.dis_mode}`",
        f"- events: `{stats.get('events', 0)}`",
        f"- final_particles: `{stats.get('particles', 0)}`",
        f"- generator_backend: `{converter.GENERATOR_BACKEND}`",
        f"- interaction_points: `{args.interaction_points}`",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{interaction}:{path}`" for interaction, path in inputs)
    lines.extend(["", "## Ray Provenance Enrichment", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in enriched.items())
    (args.output_dir / "powheg_pythia_event_records.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_driver(args: argparse.Namespace) -> Path:
    binary = ROOT / "build" / "powheg_pythia_hadros_driver"
    if binary.exists():
        return binary
    pythia8_config = autodetect_executable("pythia8-config", args.pythia8_config)
    if not pythia8_config:
        raise FileNotFoundError("pythia8-config")
    cxxflags = subprocess.check_output([pythia8_config, "--cxxflags"], text=True).split()
    ldflags = subprocess.check_output([pythia8_config, "--ldflags"], text=True).split()
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Iinclude",
        "-DHADROS_WITH_PYTHIA",
        *cxxflags,
        "apps/powheg_pythia_hadros_driver.cpp",
        "-o",
        str(binary),
        *ldflags,
    ]
    log_path = args.output_dir / "powheg_pythia_driver_build.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        proc = subprocess.run(command, cwd=ROOT, env=pythia_env(pythia8_config), stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"BUILD_RETURN_CODE={proc.returncode}\n")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return binary


def run_lhe_driver(args: argparse.Namespace, lhe_inputs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    binary = build_driver(args)
    event_dir = args.output_dir / "event_records"
    event_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, Path]] = []
    for offset, (interaction, lhe) in enumerate(lhe_inputs):
        lhe_path = lhe
        if lhe.is_absolute():
            try:
                lhe_path = lhe.relative_to(ROOT)
            except ValueError:
                lhe_path = lhe
        out_path = event_dir / f"{interaction.lower()}_event_record.txt"
        subprocess.run(
            [
                str(binary),
                "--lhe",
                str(lhe_path),
                "--output",
                str(out_path),
                "--interaction",
                interaction,
                "--seed",
                str(args.seed + offset),
                "--max-events",
                str(args.n_events),
            ],
            cwd=ROOT,
            env=pythia_env(autodetect_executable("pythia8-config", args.pythia8_config)),
            check=True,
        )
        outputs.append((interaction, out_path))
    return outputs


def maybe_run_powheg(args: argparse.Namespace, pwhg_main: str) -> tuple[str, list[tuple[str, Path]], str]:
    workdir = Path(args.powheg_workdir) if args.powheg_workdir else args.output_dir / "powheg"
    workdir.mkdir(parents=True, exist_ok=True)
    card = workdir / "powheg.input"
    if not card.exists():
        report = workdir / "powheg_lhe_generation.md"
        report.write_text(
            "\n".join(
                [
                    "# POWHEG LHE Generation",
                    "",
                    f"Status: `{STATUS_LHE_MISSING_CARDS}`.",
                    "",
                    "`pwhg_main` was found, but no versioned or run-local `powheg.input` card exists.",
                    "No placeholder card was invented.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return STATUS_MISSING_CARDS, [], "pwhg_main exists, but POWHEG DIS input cards are missing."
    log_path = workdir / "powheg_lhe_generation.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([pwhg_main], cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"POWHEG_RETURN_CODE={proc.returncode}\n")
    lhe = workdir / "pwgevents.lhe"
    if proc.returncode != 0 or not lhe.exists() or lhe.stat().st_size == 0:
        return STATUS_MISSING_CARDS, [], f"POWHEG did not produce a non-empty LHE file; see {log_path}."
    interaction = "CC" if args.mode == "cc" else "NC" if args.mode == "nc" else "CC"
    return STATUS_RUNTIME_READY, [(interaction, lhe)], ""


def generate_event_records(args: argparse.Namespace) -> tuple[str, list[tuple[str, Path]], str]:
    pythia8_config = autodetect_executable("pythia8-config", args.pythia8_config)
    pwhg_main = autodetect_executable("pwhg_main", args.pwhg_main)
    lhe_inputs = run_local_lhe(args) or discover_lhe_inputs(args.output_dir, args.lhe_input, args.mode)
    if not lhe_inputs and not pwhg_main:
        return (
            STATUS_MISSING_POWHEG,
            [],
            "No run-local POWHEG LHE file was found and `pwhg_main` is not available in PATH, so POWHEG DIS cannot be generated from scratch.",
        )
    if not lhe_inputs:
        status, lhe_inputs, reason = maybe_run_powheg(args, pwhg_main)
        if status != STATUS_RUNTIME_READY:
            return status, lhe_inputs, reason
    if not pythia8_config:
        return (
            STATUS_MISSING_PYTHIA8,
            lhe_inputs,
            "Run-local POWHEG LHE input exists, but `pythia8-config` is not available, so the versioned PYTHIA8 LHE event-record driver cannot be built.",
        )
    try:
        status = STATUS_RUNTIME_PARTIAL_LHE_ONLY if args.lhe_file or args.reuse_existing_lhe else STATUS_RUNTIME_READY
        return status, run_lhe_driver(args, lhe_inputs), ""
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return (
            STATUS_MISSING_PYTHIA8,
            [],
            f"Failed to build or execute the versioned POWHEG/PYTHIA HADROS driver: {exc}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="Run_E2E_PaperFigures")
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--interaction-points", type=Path, required=True)
    parser.add_argument("--dis-mode", choices=["both", "gbw", "iim", "GBW", "IIM"], default="both")
    parser.add_argument("--mode", choices=["cc", "nc", "both"], default="both")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--event-record-input", action="append", default=[], help="Debug/expert input: CC:/path or NC:/path to a real POWHEG/PYTHIA event-record dump.")
    parser.add_argument("--lhe-input", action="append", default=[], help="Expert input: CC:/path or NC:/path to a real POWHEG DIS LHE file.")
    parser.add_argument("--lhe-file", default="", help="Expert input: copy this real POWHEG LHE file to <output-dir>/powheg/pwgevents.lhe and run PYTHIA8.")
    parser.add_argument("--pwhg-main", default="", help="Expert input: path to POWHEG DIS pwhg_main.")
    parser.add_argument("--pythia8-config", default="", help="Expert input: path to pythia8-config.")
    parser.add_argument("--powheg-workdir", default="", help="Run-local POWHEG workdir; defaults to <output-dir>/powheg.")
    parser.add_argument("--reuse-existing-lhe", action="store_true", help="Reuse <powheg-workdir>/pwgevents.lhe when present.")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = discover_inputs(args.output_dir, args.event_record_input)
    if not inputs:
        generation_status, inputs, reason = generate_event_records(args)
        if not inputs:
            write_dependency_blocked(args, reason, inputs, generation_status)
            print(json.dumps({"status": generation_status, "reason": reason, "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))
            return 2
    resolved = [(interaction, path if path.is_absolute() else (ROOT / path).resolve()) for interaction, path in inputs]
    missing = [path for _interaction, path in resolved if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing event-record input(s): " + ", ".join(str(path) for path in missing))
    stats = converter.convert(resolved, args.output_dir)
    if stats.get("events", 0) <= 0 or stats.get("particles", 0) <= 0:
        write_dependency_blocked(
            args,
            "POWHEG/PYTHIA conversion produced zero events or zero final particles; refusing to mark material event records ready.",
            resolved,
            STATUS_MISSING_PYTHIA8,
        )
        print(json.dumps({"status": STATUS_MISSING_PYTHIA8, **stats, "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))
        return 2
    links = ray_links(args.interaction_points)
    enriched = {
        "hadros_particle_events.jsonl": enrich_rows(args.output_dir / "hadros_particle_events.jsonl", links, csv_file=False),
        "powheg_pythia_particles.jsonl": enrich_rows(args.output_dir / "powheg_pythia_particles.jsonl", links, csv_file=False),
        "powheg_pythia_particles.csv": enrich_rows(args.output_dir / "powheg_pythia_particles.csv", links, csv_file=True),
    }
    write_ready_report(args, stats, enriched, resolved)
    print(json.dumps({"status": STATUS_READY, **stats, "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
