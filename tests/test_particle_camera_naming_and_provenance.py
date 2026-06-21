#!/usr/bin/env python3
"""Source-level checks for camera naming modes and physical provenance."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "apps" / "compute_kerr_particle_camera.cpp"
PIPELINE = ROOT / "scripts" / "run_hadros_final_pipeline.py"
WRAPPER = ROOT / "scripts" / "science" / "run_real_kerr_particle_camera.py"
CONFIG_WEB = ROOT / "scripts" / "config_web_final.py"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing expected source text: {needle}")


def main() -> int:
    cpp = CPP.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    config_web = CONFIG_WEB.read_text(encoding="utf-8")

    require(config_web, 'options=["both", "semantic", "legacy"]')
    require(config_web, '"camera_naming_mode"')
    require(config_web, '"both"')

    require(pipeline, "def effective_camera_naming_mode")
    require(pipeline, '"camera_physical_interpretation": "particle-ray association / cascade origin map"')
    require(pipeline, '"camera_is_full_observational_transport": False')
    require(pipeline, '"full_transport_available": False')
    require(pipeline, "--camera-naming-mode")
    require(pipeline, "camera_csv = cascade /")

    require(wrapper, "--camera-naming-mode")
    require(wrapper, "validation_path(args.output_dir, args.camera_naming_mode)")
    require(wrapper, "Legacy `observed_particles_by_pixel.*` names are compatibility outputs only")

    require(cpp, "CAMERA_NAMING_MODE")
    require(cpp, 'camera_naming_mode != "both"')
    require(cpp, "write_semantic_outputs")
    require(cpp, "write_legacy_outputs")
    require(cpp, 'association_csv.open(out_dir / "particle_ray_association_camera.csv")')
    require(cpp, 'observed_csv.open(out_dir / "observed_particles_by_pixel.csv")')
    require(cpp, "if (write_semantic_outputs)")
    require(cpp, "if (write_legacy_outputs)")
    require(cpp, "camera_physical_interpretation")
    require(cpp, "camera_is_full_observational_transport")
    require(cpp, "camera_limitation")
    require(cpp, "full_transport_available")

    print("PASS particle-ray association naming/provenance source checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
