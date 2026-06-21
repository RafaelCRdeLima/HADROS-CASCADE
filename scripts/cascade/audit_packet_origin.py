#!/usr/bin/env python3
"""Audit where escaping-packet positions enter the cascade chain."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rg_cm_from_mbh(mbh_msun: float) -> float:
    return 6.67430e-8 * mbh_msun * 1.98847e33 / (2.99792458e10 * 2.99792458e10)


def load_mbh_msun(config_path: Path) -> float:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path)
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


def extract_position(row: dict[str, Any], rg_cm: float) -> tuple[dict[str, Any], str, str]:
    point = row.get("point") if isinstance(row.get("point"), dict) else row
    fields = []
    for key in ["x", "y", "z", "r", "theta", "phi", "x_cm", "y_cm", "z_cm", "r_cm"]:
        if key in point:
            fields.append(key)
    if {"x", "y", "z"}.issubset(point):
        x = float(point.get("x", 0.0))
        y = float(point.get("y", 0.0))
        z = float(point.get("z", 0.0))
        r, theta, phi = spherical_from_xyz(x, y, z)
        status = "synthetic_test" if point.get("origin_status") == "SYNTHETIC_TEST_POSITION" else "real_or_inherited"
    elif {"x_cm", "y_cm", "z_cm"}.issubset(point):
        x = float(point.get("x_cm", 0.0)) / rg_cm
        y = float(point.get("y_cm", 0.0)) / rg_cm
        z = float(point.get("z_cm", 0.0)) / rg_cm
        r, theta, phi = spherical_from_xyz(x, y, z)
        status = "real_interaction_point"
    elif {"r", "theta", "phi"}.issubset(point):
        r = float(point.get("r", 0.0))
        theta = float(point.get("theta", 0.0))
        phi = float(point.get("phi", 0.0))
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        status = "spherical_inherited"
    elif {"r_cm"}.issubset(point):
        r = float(point.get("r_cm", 0.0)) / rg_cm
        theta = float(point.get("theta_rad", 0.0))
        phi = float(point.get("phi_rad", 0.0))
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        status = "real_interaction_point"
    else:
        x = y = z = r = theta = phi = math.nan
        status = "missing"
    origin_status = str(point.get("origin_status", ""))
    if origin_status == "MISSING_POSITION":
        status = "missing"
    if origin_status == "SYNTHETIC_TEST_POSITION":
        status = "synthetic_test"
    if r == 1.0 and theta == 0.0:
        status = "default_like_r1_theta0" if status != "synthetic_test" else status
    return (
        {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "theta": theta,
            "phi": phi,
        },
        ",".join(fields),
        status,
    )


def audit_file(label: str, path: Path, rg_cm: float, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    data = read_jsonl(path)
    for row in data[:max_rows]:
        point = row.get("point") if isinstance(row.get("point"), dict) else row
        pos, fields, status = extract_position(row, rg_cm)
        rows.append({
            "stage": label,
            "file": str(path),
            "file_exists": path.exists(),
            "event_id": point.get("event_id", row.get("event_id", "")),
            "pdg_id": point.get("pdg_id", point.get("pdg", "")),
            "position_fields": fields,
            "position_status": status,
            "origin_status": point.get("origin_status", ""),
            "x": pos["x"],
            "y": pos["y"],
            "z": pos["z"],
            "r": pos["r"],
            "theta": pos["theta"],
            "phi": pos["phi"],
        })
    if not data:
        rows.append({
            "stage": label,
            "file": str(path),
            "file_exists": path.exists(),
            "event_id": "",
            "pdg_id": "",
            "position_fields": "",
            "position_status": "missing_file_or_empty",
            "origin_status": "",
            "x": "",
            "y": "",
            "z": "",
            "r": "",
            "theta": "",
            "phi": "",
        })
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage", "file", "file_exists", "event_id", "pdg_id", "position_fields",
        "position_status", "origin_status", "x", "y", "z", "r", "theta", "phi",
    ]
    csv_path = output_dir / "packet_origin_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["position_status"])] = status_counts.get(str(row["position_status"]), 0) + 1
    lines = [
        "# Packet Origin Audit",
        "",
        "This report audits whether escaping-packet positions are real, inherited, synthetic test positions, or missing.",
        "Physical runs must not use propagable defaults such as `r=1, theta=0`.",
        "",
        "## Status Counts",
        "",
        "| position_status | rows |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Rows", "", "| stage | event_id | pdg_id | fields | status | origin_status | r | theta | phi |", "|---|---:|---:|---|---|---|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['event_id']} | {row['pdg_id']} | {row['position_fields']} | "
            f"{row['position_status']} | {row['origin_status']} | {row['r']} | {row['theta']} | {row['phi']} |"
        )
    (output_dir / "packet_origin_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--config", type=Path, default=Path("config.ini"))
    parser.add_argument("--mbh-msun", type=float, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=40)
    args = parser.parse_args()
    mbh = args.mbh_msun if args.mbh_msun is not None else load_mbh_msun(args.config)
    rg_cm = rg_cm_from_mbh(mbh)
    files = [
        ("interaction_points", args.output_dir / "interaction_points.jsonl"),
        ("primary_interactions", args.output_dir / "primary_interactions.jsonl"),
        ("pythia_secondaries", args.output_dir / "pythia_secondaries.jsonl"),
        ("geant4_escaped", args.output_dir / "geant4_escaped_particles.jsonl"),
        ("geant4_unsupported_uhe", args.output_dir / "geant4_unsupported_uhe_particles.jsonl"),
        ("escaping_packets", args.output_dir / "escaping_particle_packets.jsonl"),
    ]
    rows: list[dict[str, Any]] = []
    for label, path in files:
        rows.extend(audit_file(label, path, rg_cm, args.max_rows_per_file))
    write_outputs(args.output_dir, rows)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
