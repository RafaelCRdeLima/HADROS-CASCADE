#pragma once

#include <cstddef>
#include <string>

struct PhotonEscapeConfig {
    double spin = 0.0;
    double black_hole_mass_msun = 1.0;
    double observer_radius_rg = 0.0;
    double max_radius_rg = 0.0;
    double geodesic_step_rg = 0.0;
    int max_geodesic_steps = 0;
    double photon_null_norm_tolerance = 0.0;
    double photon_invariant_tolerance = 0.0;
    double photon_horizon_crossing_tolerance_rg = 0.0;
    double photon_observer_crossing_tolerance_rg = 0.0;
    bool photon_fail_on_invariant_violation = true;
    double photon_min_energy_gev = 0.0;
    std::string observer_frame;
    bool enable_photon_path_sampling = false;
    int photon_path_sample_stride = 1;
    int photon_path_sample_max_rows_per_photon = 10000;
    std::string photon_path_sampling_output_format = "jsonl";
    bool photon_path_sampling_require_validation = true;
    std::string path_samples_jsonl;
    std::string path_samples_summary_csv;
    std::string path_samples_per_photon_summary_csv;
    std::string path_samples_provenance;
    std::string photon_opacity_mode = "disabled";
    std::string photon_gamma_gamma_target_field_model = "local_blackbody_isotropic";
    double photon_gamma_gamma_dilution_factor = 1.0;
    double photon_gamma_gamma_energy_grid_min_gev = 1.0e-6;
    double photon_gamma_gamma_energy_grid_max_gev = 1.0e6;
    double photon_gamma_gamma_temperature_grid_min_mev = 1.0e-3;
    double photon_gamma_gamma_temperature_grid_max_mev = 10.0;
    int photon_gamma_gamma_n_energy_bins = 64;
    int photon_gamma_gamma_n_temperature_bins = 64;
    int photon_gamma_gamma_n_epsilon_quad = 64;
    int photon_gamma_gamma_n_mu_quad = 32;
    int photon_gamma_gamma_max_table_cells = 4096;
    int photon_gamma_gamma_max_steps_per_photon = 200000;
    int photon_gamma_gamma_step_stride = 1;
    double photon_gamma_gamma_alpha_floor_cm_inv = 1.0e-300;
    double photon_gamma_gamma_table_direct_integral_tolerance = 0.2;
    bool photon_gamma_gamma_fail_on_invalid = true;
    bool photon_gamma_gamma_requires_medium = true;
    std::string photon_medium_model = "none";
    double photon_medium_torus_temperature_mev = 1.0;
    std::string photon_medium_torus_fluid_frame = "zamo";
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
