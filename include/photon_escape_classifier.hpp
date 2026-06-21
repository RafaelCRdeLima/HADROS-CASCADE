#pragma once

#include <cstddef>
#include <string>

struct PhotonEscapeConfig {
    double spin = 0.0;
    double observer_radius_rg = 0.0;
    double max_radius_rg = 0.0;
    double geodesic_step_rg = 0.0;
    int max_geodesic_steps = 0;
    double photon_null_norm_tolerance = 0.0;
    double photon_invariant_tolerance = 0.0;
    double photon_horizon_crossing_tolerance_rg = 0.0;
    bool photon_fail_on_invariant_violation = true;
    double photon_min_energy_gev = 0.0;
    std::string observer_frame;
};

struct PhotonEscapeSummary {
    std::size_t n_input_particles = 0;
    std::size_t n_photons = 0;
    std::size_t n_non_photons = 0;
    std::size_t n_captured = 0;
    std::size_t n_reached_observer_sphere = 0;
    std::size_t n_missed = 0;
    std::size_t n_failed = 0;
    std::size_t n_failed_invalid_null_momentum = 0;
    std::size_t n_failed_invariant_violation = 0;
};

PhotonEscapeSummary run_photon_escape_classifier(
    const std::string& input_jsonl,
    const std::string& output_jsonl,
    const std::string& summary_csv,
    const std::string& summary_md,
    const std::string& provenance_json,
    const PhotonEscapeConfig& config
);
