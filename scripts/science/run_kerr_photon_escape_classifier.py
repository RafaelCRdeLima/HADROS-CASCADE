#!/usr/bin/env python3
"""Run HADROS photon escape classifier Phase 1."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--path-samples-jsonl", type=Path, required=True)
    parser.add_argument("--path-samples-summary-csv", type=Path, required=True)
    parser.add_argument("--path-samples-per-photon-summary-csv", type=Path, required=True)
    parser.add_argument("--path-samples-provenance", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--spin", required=True)
    parser.add_argument("--black-hole-mass-msun", required=True)
    parser.add_argument("--observer-radius-rg", required=True)
    parser.add_argument("--max-radius-rg", required=True)
    parser.add_argument("--photon-geodesic-step-rg", required=True)
    parser.add_argument("--photon-max-geodesic-steps", required=True)
    parser.add_argument("--photon-null-norm-tolerance", required=True)
    parser.add_argument("--photon-invariant-tolerance", required=True)
    parser.add_argument("--photon-horizon-crossing-tolerance-rg", required=True)
    parser.add_argument("--photon-observer-crossing-tolerance-rg", required=True)
    parser.add_argument("--photon-fail-on-invariant-violation", choices=["true", "false"], required=True)
    parser.add_argument("--photon-min-energy-gev", required=True)
    parser.add_argument("--photon-observer-frame", required=True)
    parser.add_argument("--enable-photon-path-sampling", choices=["true", "false"], required=True)
    parser.add_argument("--photon-path-sample-stride", required=True)
    parser.add_argument("--photon-path-sample-max-rows-per-photon", required=True)
    parser.add_argument("--photon-path-sampling-output-format", required=True)
    parser.add_argument("--photon-path-sampling-require-validation", choices=["true", "false"], required=True)
    parser.add_argument("--photon-opacity-mode", required=True)
    parser.add_argument("--photon-gamma-gamma-target-field-model", required=True)
    parser.add_argument("--photon-gamma-gamma-dilution-factor", required=True)
    parser.add_argument("--photon-gamma-gamma-energy-grid-min-gev", required=True)
    parser.add_argument("--photon-gamma-gamma-energy-grid-max-gev", required=True)
    parser.add_argument("--photon-gamma-gamma-temperature-grid-min-mev", required=True)
    parser.add_argument("--photon-gamma-gamma-temperature-grid-max-mev", required=True)
    parser.add_argument("--photon-gamma-gamma-n-energy-bins", required=True)
    parser.add_argument("--photon-gamma-gamma-n-temperature-bins", required=True)
    parser.add_argument("--photon-gamma-gamma-n-epsilon-quad", required=True)
    parser.add_argument("--photon-gamma-gamma-n-mu-quad", required=True)
    parser.add_argument("--photon-gamma-gamma-max-table-cells", required=True)
    parser.add_argument("--photon-gamma-gamma-max-steps-per-photon", required=True)
    parser.add_argument("--photon-gamma-gamma-step-stride", required=True)
    parser.add_argument("--photon-gamma-gamma-alpha-floor-cm-inv", required=True)
    parser.add_argument("--photon-gamma-gamma-table-direct-integral-tolerance", required=True)
    parser.add_argument("--photon-gamma-gamma-fail-on-invalid", choices=["true", "false"], required=True)
    parser.add_argument("--photon-gamma-gamma-requires-medium", choices=["true", "false"], required=True)
    parser.add_argument("--photon-medium-model", required=True)
    parser.add_argument("--photon-medium-torus-temperature-mev", required=True)
    parser.add_argument("--photon-medium-torus-fluid-frame", required=True)
    args = parser.parse_args()

    for path in [
        args.output_jsonl,
        args.summary_csv,
        args.summary_md,
        args.provenance,
        args.path_samples_jsonl,
        args.path_samples_summary_csv,
        args.path_samples_per_photon_summary_csv,
        args.path_samples_provenance,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.backend),
        "--input", str(args.input),
        "--output-jsonl", str(args.output_jsonl),
        "--summary-csv", str(args.summary_csv),
        "--summary-md", str(args.summary_md),
        "--provenance", str(args.provenance),
        "--path-samples-jsonl", str(args.path_samples_jsonl),
        "--path-samples-summary-csv", str(args.path_samples_summary_csv),
        "--path-samples-per-photon-summary-csv", str(args.path_samples_per_photon_summary_csv),
        "--path-samples-provenance", str(args.path_samples_provenance),
        "--spin", str(args.spin),
        "--black-hole-mass-msun", str(args.black_hole_mass_msun),
        "--observer-radius-rg", str(args.observer_radius_rg),
        "--max-radius-rg", str(args.max_radius_rg),
        "--photon-geodesic-step-rg", str(args.photon_geodesic_step_rg),
        "--photon-max-geodesic-steps", str(args.photon_max_geodesic_steps),
        "--photon-null-norm-tolerance", str(args.photon_null_norm_tolerance),
        "--photon-invariant-tolerance", str(args.photon_invariant_tolerance),
        "--photon-horizon-crossing-tolerance-rg", str(args.photon_horizon_crossing_tolerance_rg),
        "--photon-observer-crossing-tolerance-rg", str(args.photon_observer_crossing_tolerance_rg),
        "--photon-fail-on-invariant-violation", str(args.photon_fail_on_invariant_violation),
        "--photon-min-energy-gev", str(args.photon_min_energy_gev),
        "--photon-observer-frame", str(args.photon_observer_frame),
        "--enable-photon-path-sampling", str(args.enable_photon_path_sampling),
        "--photon-path-sample-stride", str(args.photon_path_sample_stride),
        "--photon-path-sample-max-rows-per-photon", str(args.photon_path_sample_max_rows_per_photon),
        "--photon-path-sampling-output-format", str(args.photon_path_sampling_output_format),
        "--photon-path-sampling-require-validation", str(args.photon_path_sampling_require_validation),
        "--photon-opacity-mode", str(args.photon_opacity_mode),
        "--photon-gamma-gamma-target-field-model", str(args.photon_gamma_gamma_target_field_model),
        "--photon-gamma-gamma-dilution-factor", str(args.photon_gamma_gamma_dilution_factor),
        "--photon-gamma-gamma-energy-grid-min-gev", str(args.photon_gamma_gamma_energy_grid_min_gev),
        "--photon-gamma-gamma-energy-grid-max-gev", str(args.photon_gamma_gamma_energy_grid_max_gev),
        "--photon-gamma-gamma-temperature-grid-min-mev", str(args.photon_gamma_gamma_temperature_grid_min_mev),
        "--photon-gamma-gamma-temperature-grid-max-mev", str(args.photon_gamma_gamma_temperature_grid_max_mev),
        "--photon-gamma-gamma-n-energy-bins", str(args.photon_gamma_gamma_n_energy_bins),
        "--photon-gamma-gamma-n-temperature-bins", str(args.photon_gamma_gamma_n_temperature_bins),
        "--photon-gamma-gamma-n-epsilon-quad", str(args.photon_gamma_gamma_n_epsilon_quad),
        "--photon-gamma-gamma-n-mu-quad", str(args.photon_gamma_gamma_n_mu_quad),
        "--photon-gamma-gamma-max-table-cells", str(args.photon_gamma_gamma_max_table_cells),
        "--photon-gamma-gamma-max-steps-per-photon", str(args.photon_gamma_gamma_max_steps_per_photon),
        "--photon-gamma-gamma-step-stride", str(args.photon_gamma_gamma_step_stride),
        "--photon-gamma-gamma-alpha-floor-cm-inv", str(args.photon_gamma_gamma_alpha_floor_cm_inv),
        "--photon-gamma-gamma-table-direct-integral-tolerance", str(args.photon_gamma_gamma_table_direct_integral_tolerance),
        "--photon-gamma-gamma-fail-on-invalid", str(args.photon_gamma_gamma_fail_on_invalid),
        "--photon-gamma-gamma-requires-medium", str(args.photon_gamma_gamma_requires_medium),
        "--photon-medium-model", str(args.photon_medium_model),
        "--photon-medium-torus-temperature-mev", str(args.photon_medium_torus_temperature_mev),
        "--photon-medium-torus-fluid-frame", str(args.photon_medium_torus_fluid_frame),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
