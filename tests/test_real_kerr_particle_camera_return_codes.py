#!/usr/bin/env python3
"""Lightweight checks for real Kerr particle camera wrapper return codes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "science" / "run_real_kerr_particle_camera.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run_real_kerr_particle_camera_for_test", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    wrapper = load_wrapper()
    cases = [
        ("REAL_HADROS_BACKWARD_KERR_CAMERA_VALIDATED", 0, 0),
        ("PARTICLE_RAY_ASSOCIATION_CAMERA_VALIDATED", 0, 0),
        ("REAL_HADROS_BACKWARD_KERR_CAMERA_PARTIAL_SAMPLED_INTERACTIONS", 0, 0),
        ("PARTICLE_RAY_ASSOCIATION_CAMERA_PARTIAL_SAMPLED_INTERACTIONS", 0, 0),
        ("REAL_HADROS_BACKWARD_KERR_CAMERA_VALIDATED", 1, 2),
        ("PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_ASSOCIATION_CRITERIA", 0, 2),
    ]
    for status, backend_returncode, expected in cases:
        actual = wrapper.camera_exit_code(status, backend_returncode)
        if actual != expected:
            raise AssertionError(
                f"camera_exit_code({status!r}, {backend_returncode})={actual}, expected {expected}"
            )
    print("PASS real Kerr particle camera return-code contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
