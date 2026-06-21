// DEPRECATED / DEBUG ONLY:
// This file is not part of the final scientific HADROS chain.
// Do not use for scientific production.
//
// The final camera product is REAL_HADROS_BACKWARD_KERR_CAMERA with
// observed_particles_by_pixel and incoming UHE ray provenance. This deposition
// proxy camera is retained only as legacy/debug reference.

#include "hadros/cascade/deposition_emissivity_field.hpp"
#include "kerr_camera.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string emissivity_mode = "default";
    std::string h5_path;
    std::string manifest_path;
    std::string output_dir = "output/cascade";
    int nx = 32;
    int ny = 32;
    double spin = 0.5;
    double r_obs = 80.0;
    double theta_obs_deg = 72.0;
    double fov_deg = 18.0;
    double r_max = 120.0;
    double step = 0.01;
    double low_coverage_threshold = 0.5;
    bool auto_frame_deposition_field = false;
};

void usage() {
    std::cerr
        << "Usage: compute_deposition_proxy_camera "
        << "--emissivity-mode default|deposition_proxy "
        << "[--deposition-emissivity-h5 FIELD.h5 --deposition-emissivity-manifest MANIFEST.json] "
        << "[--output-dir output/cascade] [--nx 32 --ny 32]\n";
}

double deg_to_rad(double degrees) {
    return degrees * 3.141592653589793238462643383279502884 / 180.0;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--emissivity-mode") {
            args.emissivity_mode = need_value(arg);
        } else if (arg == "--deposition-emissivity-h5") {
            args.h5_path = need_value(arg);
        } else if (arg == "--deposition-emissivity-manifest") {
            args.manifest_path = need_value(arg);
        } else if (arg == "--output-dir") {
            args.output_dir = need_value(arg);
        } else if (arg == "--nx") {
            args.nx = std::stoi(need_value(arg));
        } else if (arg == "--ny") {
            args.ny = std::stoi(need_value(arg));
        } else if (arg == "--spin") {
            args.spin = std::stod(need_value(arg));
        } else if (arg == "--r-obs") {
            args.r_obs = std::stod(need_value(arg));
        } else if (arg == "--theta-obs-deg") {
            args.theta_obs_deg = std::stod(need_value(arg));
        } else if (arg == "--fov-deg") {
            args.fov_deg = std::stod(need_value(arg));
        } else if (arg == "--r-max") {
            args.r_max = std::stod(need_value(arg));
        } else if (arg == "--step") {
            args.step = std::stod(need_value(arg));
        } else if (arg == "--low-coverage-threshold") {
            args.low_coverage_threshold = std::stod(need_value(arg));
        } else if (arg == "--auto-frame-deposition-field") {
            args.auto_frame_deposition_field = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (args.emissivity_mode != "default" && args.emissivity_mode != "deposition_proxy") {
        throw std::runtime_error("--emissivity-mode must be default or deposition_proxy");
    }
    if (args.emissivity_mode == "deposition_proxy" && (args.h5_path.empty() || args.manifest_path.empty())) {
        throw std::runtime_error("deposition_proxy mode requires --deposition-emissivity-h5 and --deposition-emissivity-manifest");
    }
    if (args.nx <= 0 || args.ny <= 0) {
        throw std::runtime_error("--nx and --ny must be positive");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::filesystem::create_directories(args.output_dir);

        hadros::cascade::DepositionEmissivityField field;
        const bool use_proxy = args.emissivity_mode == "deposition_proxy";
        if (use_proxy) {
            field.load_hdf5(args.h5_path, args.manifest_path);
        }

        double theta_obs = deg_to_rad(args.theta_obs_deg);
        // Diagnostic-only framing: aim the observer direction toward the
        // weighted deposition center in the exported Cartesian proxy grid.
        if (use_proxy && args.auto_frame_deposition_field) {
            // The current camera implementation fixes phi_obs=0. We can only
            // steer theta here without changing the main camera machinery.
            double weighted_z = 0.0;
            double weighted_r = 0.0;
            const auto& gx = field.grid_x();
            const auto& gy = field.grid_y();
            const auto& gz = field.grid_z();
            for (double x : gx) {
                for (double y : gy) {
                    for (double z : gz) {
                        const auto q = field.query_nearest(x, y, z);
                        if (q.deposited_energy <= 0.0) {
                            continue;
                        }
                        weighted_z += q.deposited_energy * z;
                        weighted_r += q.deposited_energy * std::sqrt(x * x + y * y + z * z);
                    }
                }
            }
            if (weighted_r > 0.0) {
                const double cos_theta = std::max(-1.0, std::min(1.0, weighted_z / weighted_r));
                theta_obs = std::acos(cos_theta);
            }
        }

        KerrCamera camera(
            args.spin,
            args.r_obs,
            theta_obs,
            args.fov_deg,
            args.nx,
            args.ny,
            args.r_max,
            args.step
        );

        const std::filesystem::path raw_path = std::filesystem::path(args.output_dir) / "deposition_proxy_camera_raw.csv";
        std::ofstream raw(raw_path);
        raw << "i,j,alpha,beta,I_proxy,ok_queries,out_of_range_queries,low_coverage_queries,mean_coverage,captured\n";

        long long total_ok = 0;
        long long total_out = 0;
        long long total_low = 0;
        double coverage_sum = 0.0;
        long long coverage_count = 0;
        double image_sum = 0.0;

        for (int i = 0; i < args.nx; ++i) {
            for (int j = 0; j < args.ny; ++j) {
                const RayPath ray = camera.trace_pixel(i, j);
                double intensity = 0.0;
                long long ok = 0;
                long long out = 0;
                long long low = 0;
                double pixel_coverage_sum = 0.0;
                long long pixel_coverage_count = 0;
                if (use_proxy) {
                    for (const auto& p : ray.points) {
                        const auto result = field.query_trilinear(p.x_rg, p.y_rg, p.z_rg);
                        if (result.valid) {
                            intensity += result.j_dep * p.dl_rg;
                            ok += 1;
                            pixel_coverage_sum += result.coverage;
                            pixel_coverage_count += 1;
                            if (result.coverage < args.low_coverage_threshold) {
                                low += 1;
                            }
                        } else if (result.status == "OUT_OF_RANGE") {
                            out += 1;
                        }
                    }
                }
                const double mean_cov = pixel_coverage_count > 0
                    ? pixel_coverage_sum / static_cast<double>(pixel_coverage_count)
                    : 0.0;
                total_ok += ok;
                total_out += out;
                total_low += low;
                coverage_sum += pixel_coverage_sum;
                coverage_count += pixel_coverage_count;
                image_sum += intensity;
                raw << i << "," << j << "," << ray.alpha_rg << "," << ray.beta_rg << ","
                    << intensity << "," << ok << "," << out << "," << low << ","
                    << mean_cov << "," << (ray.captured ? 1 : 0) << "\n";
            }
        }

        const double mean_coverage = coverage_count > 0
            ? coverage_sum / static_cast<double>(coverage_count)
            : 0.0;
        const std::filesystem::path stats_path = std::filesystem::path(args.output_dir) / "deposition_proxy_camera_stats.json";
        std::ofstream stats(stats_path);
        stats << "{\n";
        stats << "  \"emissivity_mode\": \"" << args.emissivity_mode << "\",\n";
        stats << "  \"h5_path\": \"" << args.h5_path << "\",\n";
        stats << "  \"manifest_path\": \"" << args.manifest_path << "\",\n";
        stats << "  \"nx\": " << args.nx << ",\n";
        stats << "  \"ny\": " << args.ny << ",\n";
        stats << "  \"ok_queries\": " << total_ok << ",\n";
        stats << "  \"out_of_range_queries\": " << total_out << ",\n";
        stats << "  \"low_coverage_queries\": " << total_low << ",\n";
        stats << "  \"mean_coverage\": " << mean_coverage << ",\n";
        stats << "  \"image_sum\": " << image_sum << ",\n";
        stats << "  \"auto_frame_deposition_field\": " << (args.auto_frame_deposition_field ? "true" : "false") << ",\n";
        stats << "  \"theta_obs_deg_effective\": " << (theta_obs * 180.0 / 3.141592653589793238462643383279502884) << ",\n";
        stats << "  \"total_weighted_deposited_energy_gev\": "
              << (use_proxy ? field.total_deposited_energy() : 0.0) << ",\n";
        stats << "  \"warning\": \"experimental emissivity proxy, not physical luminosity\"\n";
        stats << "}\n";

        std::cout << "raw_image=" << raw_path << "\n";
        std::cout << "stats=" << stats_path << "\n";
        return 0;
    } catch (const std::exception& exc) {
        usage();
        std::cerr << "compute_deposition_proxy_camera failed: " << exc.what() << "\n";
        return 1;
    }
}
