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
