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
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--spin", required=True)
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
    args = parser.parse_args()

    for path in [args.output_jsonl, args.summary_csv, args.summary_md, args.provenance]:
        path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.backend),
        "--input", str(args.input),
        "--output-jsonl", str(args.output_jsonl),
        "--summary-csv", str(args.summary_csv),
        "--summary-md", str(args.summary_md),
        "--provenance", str(args.provenance),
        "--spin", str(args.spin),
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
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
