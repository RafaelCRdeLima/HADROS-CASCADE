#include "photon_escape_classifier.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv)
{
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("Expected --key value arguments");
        }
        args[key] = argv[++i];
    }
    return args;
}

std::string required(const std::map<std::string, std::string>& args, const std::string& key)
{
    auto it = args.find(key);
    if (it == args.end() || it->second.empty()) {
        throw std::runtime_error("Missing required argument: " + key);
    }
    return it->second;
}

double required_double(const std::map<std::string, std::string>& args, const std::string& key)
{
    return std::stod(required(args, key));
}

int required_int(const std::map<std::string, std::string>& args, const std::string& key)
{
    return std::stoi(required(args, key));
}

bool required_bool(const std::map<std::string, std::string>& args, const std::string& key)
{
    const std::string value = required(args, key);
    if (value == "true" || value == "1") {
        return true;
    }
    if (value == "false" || value == "0") {
        return false;
    }
    throw std::runtime_error("Expected boolean true/false for " + key);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto args = parse_args(argc, argv);
        PhotonEscapeConfig config;
        config.spin = required_double(args, "--spin");
        config.black_hole_mass_msun = required_double(args, "--black-hole-mass-msun");
        config.observer_radius_rg = required_double(args, "--observer-radius-rg");
        config.max_radius_rg = required_double(args, "--max-radius-rg");
        config.geodesic_step_rg = required_double(args, "--photon-geodesic-step-rg");
        config.max_geodesic_steps = required_int(args, "--photon-max-geodesic-steps");
        config.photon_null_norm_tolerance = required_double(args, "--photon-null-norm-tolerance");
        config.photon_invariant_tolerance = required_double(args, "--photon-invariant-tolerance");
        config.photon_horizon_crossing_tolerance_rg = required_double(args, "--photon-horizon-crossing-tolerance-rg");
        config.photon_observer_crossing_tolerance_rg = required_double(args, "--photon-observer-crossing-tolerance-rg");
        config.photon_fail_on_invariant_violation = required_bool(args, "--photon-fail-on-invariant-violation");
        config.photon_min_energy_gev = required_double(args, "--photon-min-energy-gev");
        config.observer_frame = required(args, "--photon-observer-frame");
        config.enable_photon_path_sampling = required_bool(args, "--enable-photon-path-sampling");
        config.photon_path_sample_stride = required_int(args, "--photon-path-sample-stride");
        config.photon_path_sample_max_rows_per_photon = required_int(args, "--photon-path-sample-max-rows-per-photon");
        config.photon_path_sampling_output_format = required(args, "--photon-path-sampling-output-format");
        config.photon_path_sampling_require_validation = required_bool(args, "--photon-path-sampling-require-validation");
        config.path_samples_jsonl = required(args, "--path-samples-jsonl");
        config.path_samples_summary_csv = required(args, "--path-samples-summary-csv");
        config.path_samples_per_photon_summary_csv = required(args, "--path-samples-per-photon-summary-csv");
        config.path_samples_provenance = required(args, "--path-samples-provenance");
        config.photon_opacity_mode = required(args, "--photon-opacity-mode");
        config.photon_gamma_gamma_target_field_model = required(args, "--photon-gamma-gamma-target-field-model");
        config.photon_gamma_gamma_dilution_factor = required_double(args, "--photon-gamma-gamma-dilution-factor");
        config.photon_gamma_gamma_energy_grid_min_gev = required_double(args, "--photon-gamma-gamma-energy-grid-min-gev");
        config.photon_gamma_gamma_energy_grid_max_gev = required_double(args, "--photon-gamma-gamma-energy-grid-max-gev");
        config.photon_gamma_gamma_temperature_grid_min_mev = required_double(args, "--photon-gamma-gamma-temperature-grid-min-mev");
        config.photon_gamma_gamma_temperature_grid_max_mev = required_double(args, "--photon-gamma-gamma-temperature-grid-max-mev");
        config.photon_gamma_gamma_n_energy_bins = required_int(args, "--photon-gamma-gamma-n-energy-bins");
        config.photon_gamma_gamma_n_temperature_bins = required_int(args, "--photon-gamma-gamma-n-temperature-bins");
        config.photon_gamma_gamma_n_epsilon_quad = required_int(args, "--photon-gamma-gamma-n-epsilon-quad");
        config.photon_gamma_gamma_n_mu_quad = required_int(args, "--photon-gamma-gamma-n-mu-quad");
        config.photon_gamma_gamma_max_table_cells = required_int(args, "--photon-gamma-gamma-max-table-cells");
        config.photon_gamma_gamma_max_steps_per_photon = required_int(args, "--photon-gamma-gamma-max-steps-per-photon");
        config.photon_gamma_gamma_step_stride = required_int(args, "--photon-gamma-gamma-step-stride");
        config.photon_gamma_gamma_alpha_floor_cm_inv = required_double(args, "--photon-gamma-gamma-alpha-floor-cm-inv");
        config.photon_gamma_gamma_table_direct_integral_tolerance = required_double(args, "--photon-gamma-gamma-table-direct-integral-tolerance");
        config.photon_gamma_gamma_fail_on_invalid = required_bool(args, "--photon-gamma-gamma-fail-on-invalid");
        config.photon_gamma_gamma_requires_medium = required_bool(args, "--photon-gamma-gamma-requires-medium");
        config.photon_medium_model = required(args, "--photon-medium-model");
        config.photon_medium_torus_temperature_mev = required_double(args, "--photon-medium-torus-temperature-mev");
        config.photon_medium_torus_fluid_frame = required(args, "--photon-medium-torus-fluid-frame");

        run_photon_escape_classifier(
            required(args, "--input"),
            required(args, "--output-jsonl"),
            required(args, "--summary-csv"),
            required(args, "--summary-md"),
            required(args, "--provenance"),
            config
        );
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << "compute_kerr_photon_escape_classifier: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
}
