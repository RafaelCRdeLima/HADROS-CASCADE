#!/usr/bin/env python3
"""Tests for photon observer-camera diagnostic plot generation."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "build_photon_observer_diagnostic_plots.py"


FIELDNAMES = [
    "event_id",
    "particle_id",
    "pixel_x",
    "pixel_y",
    "inside_fov",
    "input_energy_gev",
    "observed_energy_gev",
    "redshift_factor",
    "redshift_status",
]


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] = FIELDNAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_script(input_csv: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_csv),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_diagnostic_plots_from_synthetic_csv() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_diag_") as tmp_name:
        tmp = Path(tmp_name)
        input_csv = tmp / "photon_observer_camera_redshift.csv"
        out = tmp / "diagnostics"
        write_rows(
            input_csv,
            [
                {
                    "event_id": 1,
                    "particle_id": 1,
                    "pixel_x": 0,
                    "pixel_y": 1,
                    "inside_fov": "true",
                    "input_energy_gev": 10.0,
                    "observed_energy_gev": 9.0,
                    "redshift_factor": 0.9,
                    "redshift_status": "valid",
                },
                {
                    "event_id": 1,
                    "particle_id": 2,
                    "pixel_x": "",
                    "pixel_y": "",
                    "inside_fov": "false",
                    "input_energy_gev": 8.0,
                    "observed_energy_gev": 7.2,
                    "redshift_factor": 0.9,
                    "redshift_status": "valid",
                },
                {
                    "event_id": 1,
                    "particle_id": 3,
                    "pixel_x": 0,
                    "pixel_y": 1,
                    "inside_fov": "true",
                    "input_energy_gev": 1000.0,
                    "observed_energy_gev": 1.0,
                    "redshift_factor": 0.001,
                    "redshift_status": "invalid_invariants",
                },
            ],
        )
        completed = run_script(input_csv, out)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        expected = [
            "photon_diagnostic_input_energy_map.png",
            "photon_diagnostic_observed_energy_map.png",
            "photon_diagnostic_counts_map.png",
            "photon_diagnostic_redshift_histogram.png",
            "photon_diagnostic_input_vs_observed_energy.png",
            "photon_diagnostic_summary.md",
        ]
        for name in expected:
            path = out / name
            if not path.exists() or path.stat().st_size <= 0:
                raise AssertionError(f"missing diagnostic product: {path}")
            if "paper" in name.lower() or "final" in name.lower():
                raise AssertionError(f"diagnostic product has forbidden name: {name}")
        summary = (out / "photon_diagnostic_summary.md").read_text(encoding="utf-8")
        if "not paper-ready" not in summary:
            raise AssertionError(summary)
        if "n_valid_redshift_rows: `2`" not in summary:
            raise AssertionError(summary)
        if "n_valid_inside_fov_rows: `1`" not in summary:
            raise AssertionError(summary)
        if "total_input_energy_inside_fov_gev: `10`" not in summary:
            raise AssertionError(summary)
        if "total_observed_energy_inside_fov_gev: `9`" not in summary:
            raise AssertionError(summary)


def test_missing_observed_energy_fails_clearly() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_diag_missing_") as tmp_name:
        tmp = Path(tmp_name)
        input_csv = tmp / "photon_observer_camera_redshift.csv"
        out = tmp / "diagnostics"
        fields = [field for field in FIELDNAMES if field != "observed_energy_gev"]
        write_rows(
            input_csv,
            [
                {
                    "event_id": 1,
                    "particle_id": 1,
                    "pixel_x": 0,
                    "pixel_y": 0,
                    "inside_fov": "true",
                    "input_energy_gev": 1.0,
                    "redshift_factor": 1.0,
                    "redshift_status": "valid",
                }
            ],
            fields,
        )
        completed = run_script(input_csv, out)
        if completed.returncode == 0:
            raise AssertionError("script accepted CSV without observed_energy_gev")
        if "observed_energy_gev" not in completed.stderr:
            raise AssertionError(completed.stderr)


if __name__ == "__main__":
    test_diagnostic_plots_from_synthetic_csv()
    test_missing_observed_energy_fails_clearly()
