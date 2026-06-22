#!/usr/bin/env python3
"""Tests for photon observer-camera diagnostic plot generation."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "science" / "build_photon_observer_diagnostic_plots.py"

from scripts.science import build_photon_observer_diagnostic_plots as diag


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


def run_script(input_csv: Path, output_dir: Path, *, diagnostic_recenter: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_csv),
        "--output-dir",
        str(output_dir),
    ]
    if diagnostic_recenter:
        command.append("--diagnostic-recenter")
    return subprocess.run(
        command,
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
            "photon_diagnostic_mean_redshift_map.png",
            "photon_diagnostic_valid_photon_density_map.png",
            "photon_diagnostic_mean_observed_energy_map.png",
            "photon_diagnostic_redshift_histogram.png",
            "photon_diagnostic_input_vs_observed_energy.png",
            "photon_diagnostic_summary.md",
            "photon_diagnostic_morphology_summary.md",
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
        if "n_pixels_active: `1`" not in summary:
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


def test_map_values_orientation_and_filters() -> None:
    rows = [
        {
            "pixel_x": "0",
            "pixel_y": "0",
            "inside_fov": "true",
            "input_energy_gev": "2",
            "observed_energy_gev": "1",
            "redshift_factor": "0.5",
            "redshift_status": "valid",
        },
        {
            "pixel_x": "1",
            "pixel_y": "1",
            "inside_fov": "true",
            "input_energy_gev": "4",
            "observed_energy_gev": "3",
            "redshift_factor": "0.75",
            "redshift_status": "valid",
        },
        {
            "pixel_x": "1",
            "pixel_y": "1",
            "inside_fov": "true",
            "input_energy_gev": "100",
            "observed_energy_gev": "100",
            "redshift_factor": "1",
            "redshift_status": "invalid_invariants",
        },
        {
            "pixel_x": "0",
            "pixel_y": "1",
            "inside_fov": "false",
            "input_energy_gev": "100",
            "observed_energy_gev": "100",
            "redshift_factor": "1",
            "redshift_status": "valid",
        },
    ]
    inside = diag.valid_inside_fov_rows(rows)
    maps = diag.build_maps(inside, nx=2, ny=2)
    if maps["counts"][0][0] != 1.0 or maps["counts"][1][1] != 1.0:
        raise AssertionError(f"bad count map or pixel_y orientation: {maps['counts']}")
    if maps["observed_energy"][0][0] != 1.0 or maps["observed_energy"][1][1] != 3.0:
        raise AssertionError(f"bad observed energy map: {maps['observed_energy']}")
    if maps["mean_redshift"][0][0] != 0.5 or maps["mean_redshift"][1][1] != 0.75:
        raise AssertionError(f"bad mean redshift map: {maps['mean_redshift']}")
    if maps["valid_density"] != maps["counts"]:
        raise AssertionError(f"valid density should match valid count map: {maps}")
    if any(value < 0.0 for value in diag.flatten_map(maps["observed_energy"])):
        raise AssertionError(f"negative observed energy map value: {maps['observed_energy']}")


def test_negative_energy_warning() -> None:
    rows = [
        {
            "pixel_x": "0",
            "pixel_y": "0",
            "inside_fov": "true",
            "input_energy_gev": "1",
            "observed_energy_gev": "-1",
            "redshift_factor": "1",
            "redshift_status": "valid",
        }
    ]
    inside = diag.valid_inside_fov_rows(rows)
    maps = diag.build_maps(inside, nx=1, ny=1)
    metrics = diag.morphology_metrics(maps, nx=1, ny=1, inside=inside, valid=diag.valid_rows(rows))
    if "negative_energy_values_present" not in metrics["warnings"]:
        raise AssertionError(f"negative energy warning missing: {metrics}")


def test_diagnostic_recenter_outputs_and_preserves_input() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_photon_diag_recenter_") as tmp_name:
        tmp = Path(tmp_name)
        input_csv = tmp / "photon_observer_camera_redshift.csv"
        out = tmp / "diagnostics"
        fields = FIELDNAMES + [
            "observer_crossing_theta_rad",
            "observer_crossing_phi_rad",
            "camera_x",
            "camera_y",
        ]
        rows = []
        # A compact cluster near negative camera_x / positive camera_y for the
        # default center, plus enough angular spread to deconcentrate after
        # the diagnostic recentering pass.
        for index, (theta, phi) in enumerate(
            [
                (1.0472, -0.42),
                (1.0520, -0.41),
                (1.0570, -0.40),
                (1.0620, -0.39),
                (1.0670, -0.38),
                (1.0720, -0.37),
                (1.0770, -0.36),
                (1.0820, -0.35),
                (1.0870, -0.34),
                (1.0920, -0.33),
                (1.0970, -0.32),
                (1.1020, -0.31),
            ],
            start=1,
        ):
            rows.append(
                {
                    "event_id": 1,
                    "particle_id": index,
                    "pixel_x": 0,
                    "pixel_y": 0,
                    "inside_fov": "true",
                    "input_energy_gev": 10.0,
                    "observed_energy_gev": 9.0,
                    "redshift_factor": 0.9,
                    "redshift_status": "valid",
                    "observer_crossing_theta_rad": theta,
                    "observer_crossing_phi_rad": phi,
                    "camera_x": -0.5,
                    "camera_y": 0.5,
                }
            )
        write_rows(input_csv, rows, fields)
        provenance = tmp / "photon_observer_camera_provenance.json"
        provenance.write_text(
            '{'
            '"camera_nx": 8, "camera_ny": 8, "photon_camera_fov_deg": 60.0, '
            '"photon_camera_center_theta_deg": 70.0, "photon_camera_center_phi_rad": 0.0'
            '}',
            encoding="utf-8",
        )
        before = input_csv.read_text(encoding="utf-8")
        completed = run_script(input_csv, out, diagnostic_recenter=True)
        after = input_csv.read_text(encoding="utf-8")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        if before != after:
            raise AssertionError("diagnostic recentering modified the input CSV")
        for name in [
            "photon_diagnostic_recentered_counts_map.png",
            "photon_diagnostic_recentered_observed_energy_map.png",
            "photon_diagnostic_recentered_mean_redshift_map.png",
            "photon_diagnostic_recentered_summary.md",
            "photon_diagnostic_recentered_projection_stats.csv",
        ]:
            path = out / name
            if not path.exists() or path.stat().st_size <= 0:
                raise AssertionError(f"missing recentered diagnostic product: {path}")
            if "paper" in name.lower() or "final" in name.lower():
                raise AssertionError(f"recentered diagnostic product has forbidden name: {name}")
        summary = (out / "photon_diagnostic_recentered_summary.md").read_text(encoding="utf-8")
        for needle in [
            "Diagnostic recentered projection only",
            "not the default camera pointing",
            "not paper-ready",
            "Official `photon_observer_camera.csv` and `photon_observer_camera_redshift.csv` are not modified",
        ]:
            if needle not in summary:
                raise AssertionError(summary)
        before_fraction = None
        after_fraction = None
        for line in summary.splitlines():
            if line.startswith("- brightest_pixel_fraction_before:"):
                before_fraction = float(line.split("`")[1])
            if line.startswith("- brightest_pixel_fraction_after:"):
                after_fraction = float(line.split("`")[1])
        if before_fraction is None or after_fraction is None:
            raise AssertionError(summary)
        if not after_fraction < before_fraction:
            raise AssertionError(summary)


def test_diagnostic_recenter_adds_no_config_parameters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    if "config_web_final" in source:
        raise AssertionError("diagnostic recentering should not introduce config-web parameters")


if __name__ == "__main__":
    test_diagnostic_plots_from_synthetic_csv()
    test_missing_observed_energy_fails_clearly()
    test_map_values_orientation_and_filters()
    test_negative_energy_warning()
    test_diagnostic_recenter_outputs_and_preserves_input()
    test_diagnostic_recenter_adds_no_config_parameters()
