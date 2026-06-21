#!/usr/bin/env python3
"""Source-level checks for the particle-ray association camera semantics."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "apps" / "compute_kerr_particle_camera.cpp"
WRAPPER = ROOT / "scripts" / "science" / "run_real_kerr_particle_camera.py"
PIPELINE = ROOT / "scripts" / "run_hadros_final_pipeline.py"
CONFIG_WEB = ROOT / "scripts" / "config_web_final.py"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing expected source text: {needle}")


def reject(text: str, needle: str) -> None:
    if needle in text:
        raise AssertionError(f"forbidden source text still present: {needle}")


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    config_web = CONFIG_WEB.read_text(encoding="utf-8")
    require(text, "PARTICLE_RAY_ASSOCIATION_CAMERA")
    require(text, "particle_ray_association_camera.csv")
    require(text, "direction_misalignment_deg(const ParticleRow& particle, const RaySample& sample)")
    require(text, "misalignment > angular_tolerance_deg")
    require(text, "rejected_missing_direction")
    require(text, "NaN means not calculated")
    require(text, "spatial_only_direction_not_calculated")
    require(text, "spatial_plus_direction_particle_momentum_vs_kerr_ray_to_observer")
    require(text, "full_transport is not implemented yet")
    require(text, "particle_pdg,particle_energy_gev,production_x_rg")
    require(text, "total_energy_gev,total_weighted_energy_gev,mean_energy_gev")
    require(wrapper, "--association-mode")
    require(wrapper, "full_transport is not implemented yet")
    require(pipeline, "effective_association_mode")
    require(pipeline, "--association-mode")
    require(pipeline, "full_transport is not implemented yet")
    require(config_web, "particle_ray_association_camera")
    require(config_web, "association_mode")
    require(config_web, "spatial_tolerance_rg")
    require(config_web, "angular_tolerance_deg")
    require(config_web, "camera_naming_mode")
    require(config_web, "description")
    reject(text, "<< distance << ',' << 0.0 << ','")
    reject(text, "argc > 11 ? std::stod(argv[11]) : 1.0")
    reject(text, "REAL_HADROS_BACKWARD_KERR_CAMERA_VALIDATED")
    print("PASS particle-ray association camera source checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
