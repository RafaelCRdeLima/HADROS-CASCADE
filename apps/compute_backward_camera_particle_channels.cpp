#include "constants.hpp"
#include "kerr_camera.hpp"
#include "sigma_table.hpp"
#include "torus_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

struct Options {
    std::string output_csv = "output/science/backward_camera_particle_channels/backward_camera_particle_channels.csv";
    std::string output_summary = "output/science/backward_camera_particle_channels/backward_camera_particle_summary.md";
    std::string sigma_table = "data/sigma/sigma_nuN_CC_GBW.dat";
    std::string dis_model = "GBW";
    double energy_gev = 1.0e9;
    double spin = 0.8;
    double mbh_msun = 3.0;
    double camera_r_obs_rg = 80.0;
    double camera_theta_deg = 70.0;
    double camera_fov_deg = 60.0;
    int camera_nx = 32;
    int camera_ny = 32;
    double camera_r_max_rg = 120.0;
    double camera_step = 0.02;
    double torus_rho0 = 1.0e-2;
    double torus_r0_rg = 10.0;
    double torus_sigma_rg = 5.0;
    double torus_h_over_r = 0.25;
    std::string density_profile = "gaussian";
    double torus_radial_power = 2.0;
    double funnel_depletion = 0.0;
    double funnel_theta_deg = 15.0;
    double envelope_rho0 = 0.0;
    double envelope_alpha = 2.5;
    double torus_r_min_rg = 4.0;
    double torus_r_max_rg = 60.0;
    double rho_floor = 1.0e-99;
};

double to_double(const std::string& text)
{
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (!end || *end != '\0') {
        throw std::runtime_error("Expected floating-point value, got '" + text + "'");
    }
    return value;
}

int to_int(const std::string& text)
{
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0') {
        throw std::runtime_error("Expected integer value, got '" + text + "'");
    }
    return static_cast<int>(value);
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    std::map<std::string, std::string*> strings = {
        {"--output-csv", &opt.output_csv},
        {"--output-summary", &opt.output_summary},
        {"--sigma-table", &opt.sigma_table},
        {"--dis-model", &opt.dis_model},
        {"--density-profile", &opt.density_profile},
    };
    std::map<std::string, double*> doubles = {
        {"--energy-gev", &opt.energy_gev},
        {"--spin", &opt.spin},
        {"--mbh-msun", &opt.mbh_msun},
        {"--camera-r-obs-rg", &opt.camera_r_obs_rg},
        {"--camera-theta-deg", &opt.camera_theta_deg},
        {"--camera-fov-deg", &opt.camera_fov_deg},
        {"--camera-r-max-rg", &opt.camera_r_max_rg},
        {"--camera-step", &opt.camera_step},
        {"--torus-rho0", &opt.torus_rho0},
        {"--torus-r0-rg", &opt.torus_r0_rg},
        {"--torus-sigma-rg", &opt.torus_sigma_rg},
        {"--torus-h-over-r", &opt.torus_h_over_r},
        {"--torus-radial-power", &opt.torus_radial_power},
        {"--funnel-depletion", &opt.funnel_depletion},
        {"--funnel-theta-deg", &opt.funnel_theta_deg},
        {"--envelope-rho0", &opt.envelope_rho0},
        {"--envelope-alpha", &opt.envelope_alpha},
        {"--torus-r-min-rg", &opt.torus_r_min_rg},
        {"--torus-r-max-rg", &opt.torus_r_max_rg},
        {"--rho-floor", &opt.rho_floor},
    };
    std::map<std::string, int*> ints = {
        {"--camera-nx", &opt.camera_nx},
        {"--camera-ny", &opt.camera_ny},
    };

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "Usage: compute_backward_camera_particle_channels [options]\n"
                << "Produces backward-camera weighted-energy proxy channels.\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (strings.count(key)) {
            *strings[key] = value;
        } else if (doubles.count(key)) {
            *doubles[key] = to_double(value);
        } else if (ints.count(key)) {
            *ints[key] = to_int(value);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    opt.camera_nx = std::max(1, opt.camera_nx);
    opt.camera_ny = std::max(1, opt.camera_ny);
    opt.camera_step = std::max(1.0e-6, opt.camera_step);
    return opt;
}

struct PixelRow {
    int i = 0;
    int j = 0;
    int n_samples = 0;
    int n_inside_torus = 0;
    int n_inside_funnel = 0;
    int n_outside_torus = 0;
    double path_length_rg = 0.0;
    double path_inside_torus_rg = 0.0;
    double path_inside_funnel_rg = 0.0;
    double tau = 0.0;
    double pint = 0.0;
    double gamma = 0.0;
    double electromagnetic = 0.0;
    double hadronic = 0.0;
    double pion_charged = 0.0;
    double deposited = 0.0;
    double invisible = 0.0;
    double unsupported_uhe = 0.0;
};

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_args(argc, argv);
        SigmaTable sigma(opt.sigma_table);
        TorusProfile torus(
            opt.torus_rho0,
            opt.torus_r0_rg,
            opt.torus_sigma_rg,
            opt.torus_h_over_r,
            opt.density_profile,
            opt.torus_radial_power,
            opt.funnel_depletion,
            opt.funnel_theta_deg * pi / 180.0,
            opt.envelope_rho0,
            opt.envelope_alpha,
            opt.torus_r_min_rg,
            opt.torus_r_max_rg,
            opt.rho_floor
        );
        KerrCamera camera(
            opt.spin,
            opt.camera_r_obs_rg,
            opt.camera_theta_deg * pi / 180.0,
            opt.camera_fov_deg,
            opt.camera_nx,
            opt.camera_ny,
            opt.camera_r_max_rg,
            opt.camera_step
        );

        std::ofstream csv(opt.output_csv);
        if (!csv) {
            throw std::runtime_error("Could not open output CSV: " + opt.output_csv);
        }
        csv << std::setprecision(12);
        csv
            << "pixel_i,pixel_j,n_samples,n_inside_torus,n_outside_torus,"
            << "n_inside_funnel,path_length_rg,path_inside_torus_rg,path_inside_funnel_rg,tau,pint,"
            << "gamma_weighted_energy_proxy,electromagnetic_weighted_energy_proxy,"
            << "hadronic_weighted_energy_proxy,pion_charged_weighted_energy_proxy,"
            << "deposited_weighted_energy_proxy,invisible_weighted_energy_proxy,"
            << "unsupported_uhe_weighted_energy_proxy,proxy_status\n";

        double total_tau = 0.0;
        double total_gamma = 0.0;
        double total_em = 0.0;
        double total_had = 0.0;
        double total_pion = 0.0;
        double total_dep = 0.0;
        double total_inv = 0.0;
        double total_unsupported = 0.0;
        int pixels_with_torus = 0;
        int total_inside_samples = 0;
        int total_funnel_samples = 0;
        int total_samples = 0;

        const double rg_cm = constants::rg_cm(opt.mbh_msun);
        const double sigma_cm2 = sigma.sigma_cm2(std::min(std::max(opt.energy_gev, sigma.Emin()), sigma.Emax()));

        for (int j = 0; j < opt.camera_ny; ++j) {
            for (int i = 0; i < opt.camera_nx; ++i) {
                PixelRow row;
                row.i = i;
                row.j = j;
                camera.trace_pixel_stream(i, j, [&](const PathPoint& p, int) {
                    ++row.n_samples;
                    row.path_length_rg += p.dl_rg;
                    const bool inside = torus.in_torus(p.r_rg, p.theta);
                    const bool in_funnel =
                        p.r_rg >= opt.torus_r_min_rg &&
                        p.r_rg <= opt.torus_r_max_rg &&
                        (p.theta <= opt.funnel_theta_deg * pi / 180.0 ||
                         p.theta >= pi - opt.funnel_theta_deg * pi / 180.0);
                    if (in_funnel) {
                        ++row.n_inside_funnel;
                        row.path_inside_funnel_rg += p.dl_rg;
                    }
                    if (inside) {
                        ++row.n_inside_torus;
                        row.path_inside_torus_rg += p.dl_rg;
                        const double rho = torus.rho(p.r_rg, p.theta);
                        const double nb = rho / constants::m_u_g;
                        row.tau += nb * sigma_cm2 * p.dl_rg * rg_cm;
                    } else {
                        ++row.n_outside_torus;
                    }
                    return true;
                });
                row.pint = 1.0 - std::exp(-std::max(row.tau, 0.0));

                // Explicit local-response proxy fractions. These are not
                // luminosities, fluxes, spectra, or on-demand PYTHIA/GEANT4.
                const double base = opt.energy_gev * row.pint;
                row.gamma = 0.45 * base;
                row.electromagnetic = 0.60 * base;
                row.hadronic = 0.30 * base;
                row.pion_charged = 0.10 * base;
                row.deposited = 0.05 * base;
                row.invisible = 0.03 * base;
                row.unsupported_uhe = 0.02 * base;

                if (row.n_inside_torus > 0) {
                    ++pixels_with_torus;
                }
                total_samples += row.n_samples;
                total_inside_samples += row.n_inside_torus;
                total_funnel_samples += row.n_inside_funnel;
                total_tau += row.tau;
                total_gamma += row.gamma;
                total_em += row.electromagnetic;
                total_had += row.hadronic;
                total_pion += row.pion_charged;
                total_dep += row.deposited;
                total_inv += row.invisible;
                total_unsupported += row.unsupported_uhe;

                csv
                    << row.i << ','
                    << row.j << ','
                    << row.n_samples << ','
                    << row.n_inside_torus << ','
                    << row.n_outside_torus << ','
                    << row.n_inside_funnel << ','
                    << row.path_length_rg << ','
                    << row.path_inside_torus_rg << ','
                    << row.path_inside_funnel_rg << ','
                    << row.tau << ','
                    << row.pint << ','
                    << row.gamma << ','
                    << row.electromagnetic << ','
                    << row.hadronic << ','
                    << row.pion_charged << ','
                    << row.deposited << ','
                    << row.invisible << ','
                    << row.unsupported_uhe << ','
                    << "backward_camera_local_response_proxy\n";
            }
        }

        std::ofstream md(opt.output_summary);
        if (!md) {
            throw std::runtime_error("Could not open output summary: " + opt.output_summary);
        }
        md << "# Backward Camera Particle Channels\n\n"
           << "Status: `BACKWARD_CAMERA_LOCAL_RESPONSE_PROTOTYPE`\n\n"
           << "This product uses HADROS backward Kerr camera rays to select physical regions seen by each pixel. "
           << "The channel response is an explicit `backward_camera_local_response_proxy`; it is not luminosity, flux, "
           << "radiative transfer, redshift-calibrated energy, or on-demand PYTHIA/GEANT4.\n\n"
           << "- proxy_status: `backward_camera_local_response_proxy`\n"
           << "- dis_model: `" << opt.dis_model << "`\n"
           << "- sigma_table: `" << opt.sigma_table << "`\n"
           << "- energy_gev: `" << opt.energy_gev << "`\n"
           << "- camera_nx: `" << opt.camera_nx << "`\n"
           << "- camera_ny: `" << opt.camera_ny << "`\n"
           << "- camera_fov_deg: `" << opt.camera_fov_deg << "`\n"
           << "- camera_theta_deg: `" << opt.camera_theta_deg << "`\n"
           << "- camera_r_obs_rg: `" << opt.camera_r_obs_rg << "`\n"
           << "- camera_r_max_rg: `" << opt.camera_r_max_rg << "`\n"
           << "- camera_step: `" << opt.camera_step << "`\n"
           << "- pixels_with_torus_intersection: `" << pixels_with_torus << "`\n"
           << "- total_samples: `" << total_samples << "`\n"
           << "- total_inside_torus_samples: `" << total_inside_samples << "`\n"
           << "- total_inside_funnel_samples: `" << total_funnel_samples << "`\n"
           << "- total_tau_sum_over_pixels: `" << total_tau << "`\n"
           << "- total_gamma_weighted_energy_proxy: `" << total_gamma << "`\n"
           << "- total_electromagnetic_weighted_energy_proxy: `" << total_em << "`\n"
           << "- total_hadronic_weighted_energy_proxy: `" << total_had << "`\n"
           << "- total_pion_charged_weighted_energy_proxy: `" << total_pion << "`\n"
           << "- total_deposited_weighted_energy_proxy: `" << total_dep << "`\n"
           << "- total_invisible_weighted_energy_proxy: `" << total_inv << "`\n"
           << "- total_unsupported_uhe_weighted_energy_proxy: `" << total_unsupported << "`\n\n"
           << "Old forward packet products remain diagnostic forward-projected packet maps only.\n";

        std::cout << "Wrote " << opt.output_csv << "\n";
        std::cout << "Wrote " << opt.output_summary << "\n";
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
