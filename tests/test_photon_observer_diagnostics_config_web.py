#!/usr/bin/env python3
"""Tests for final-config-web photon observer diagnostics integration."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config_web_final


def values_for_output(output_dir: Path) -> dict[str, dict[str, object]]:
    values = config_web_final.defaults()
    values["run"]["output_dir"] = str(output_dir)
    return values


def test_photon_observer_diagnostics_tab_appears() -> None:
    tabs = [item["tab"] for item in config_web_final.schema()]
    if "Photon Observer Diagnostics" not in tabs:
        raise AssertionError(f"missing Photon Observer Diagnostics tab: {tabs}")


def test_photon_observer_diagnostics_buttons_exist() -> None:
    html = config_web_final.render_html(config_web_final.defaults(), config_web_final.DEFAULT_CONFIG)
    for needle in [
        "Generate photon diagnostic plots",
        "Open photon diagnostic output folder",
        "Open photon_diagnostic_counts_map.png",
        "Open photon_diagnostic_input_energy_map.png",
        "Open photon_diagnostic_observed_energy_map.png",
        "Open photon_diagnostic_mean_redshift_map.png",
        "Open photon_diagnostic_input_vs_observed_energy.png",
        "Open photon_diagnostic_redshift_histogram.png",
        "Open photon_diagnostic_morphology_summary.md",
        "Diagnostic only.",
        "ideal photon observer camera, no detector response",
    ]:
        if needle not in html:
            raise AssertionError(f"missing UI text/button: {needle}")


def test_photon_diagnostics_adds_no_physical_parameters() -> None:
    values = config_web_final.defaults()
    if "photon_observer_diagnostics" in values:
        raise AssertionError("diagnostics tab introduced a config parameter section")
    pipeline = config_web_final.final_pipeline_config(values)
    for key in pipeline:
        if key.startswith("photon_diagnostic"):
            raise AssertionError(f"diagnostic key leaked into physical pipeline config: {key}")


def test_missing_redshift_csv_fails_clearly() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_web_photon_diag_missing_") as tmp_name:
        values = values_for_output(Path(tmp_name) / "run")
        code, output = config_web_final.generate_photon_diagnostics(values)
        if code == 0:
            raise AssertionError("diagnostic generation succeeded without redshift CSV")
        if "Missing photon_observer_camera_redshift.csv" not in output:
            raise AssertionError(output)
        folder_html = config_web_final.render_photon_diagnostic_folder(values)
        if "Missing photon_observer_camera_redshift.csv" not in folder_html:
            raise AssertionError(folder_html)


def test_generate_command_uses_diagnostic_script() -> None:
    with tempfile.TemporaryDirectory(prefix="hadros_web_photon_diag_") as tmp_name:
        run_dir = Path(tmp_name) / "run"
        cascade = run_dir / "cascade"
        cascade.mkdir(parents=True)
        redshift = cascade / "photon_observer_camera_redshift.csv"
        redshift.write_text("redshift_status,inside_fov,pixel_x,pixel_y,input_energy_gev,observed_energy_gev,redshift_factor\n", encoding="utf-8")
        values = values_for_output(run_dir)
        calls: list[list[str]] = []
        original_run = config_web_final.subprocess.run

        def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout="ok\n")

        config_web_final.subprocess.run = fake_run  # type: ignore[assignment]
        try:
            code, output = config_web_final.generate_photon_diagnostics(values)
        finally:
            config_web_final.subprocess.run = original_run  # type: ignore[assignment]
        if code != 0:
            raise AssertionError(output)
        if not calls:
            raise AssertionError("diagnostic generation did not invoke subprocess.run")
        command_text = " ".join(calls[0])
        if "scripts/science/build_photon_observer_diagnostic_plots.py" not in command_text:
            raise AssertionError(f"wrong diagnostic command: {command_text}")
        if str(redshift) not in calls[0]:
            raise AssertionError(f"redshift CSV not passed to command: {calls[0]}")


if __name__ == "__main__":
    test_photon_observer_diagnostics_tab_appears()
    test_photon_observer_diagnostics_buttons_exist()
    test_photon_diagnostics_adds_no_physical_parameters()
    test_missing_redshift_csv_fails_clearly()
    test_generate_command_uses_diagnostic_script()
