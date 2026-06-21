#ifndef HADROS_WITH_GEANT4
#error "cascade_geant4_local_box requires HADROS_WITH_GEANT4=ON."
#endif

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "hadros/cascade/geant4_local_box_backend.hpp"
#include "hadros/cascade/jsonl_io.hpp"

namespace {

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " secondaries.jsonl output_dir box_size_cm density_g_cm3 physics_list material"
                 " [transport_mode] [primary_interactions.jsonl]"
                 " [--energy-convention total|kinetic]"
                 " [--geant4-safety-mode off|strict]"
                 " [--geant4-one-particle-per-run]"
                 " [--debug-single-particle]"
                 " [--uhe-transport-policy error|skip_to_escaped|split_energy_proxy]"
                 " [--geant4-hadron-max-kinetic-gev value]"
                 " [--geant4-lepton-max-kinetic-gev value]"
                 " [--geant4-photon-max-kinetic-gev value]"
                 " [--geant4-local-cm-per-rg value]\n";
}

std::map<std::uint64_t, std::vector<hadros::cascade::SecondaryParticle>>
group_by_event(const std::vector<hadros::cascade::SecondaryParticle>& particles) {
    std::map<std::uint64_t, std::vector<hadros::cascade::SecondaryParticle>> grouped;
    for (const auto& particle : particles) {
        grouped[particle.event_id].push_back(particle);
    }
    return grouped;
}

std::map<std::uint64_t, double> read_density_map(const std::string& path) {
    std::map<std::uint64_t, double> densities;
    if (path.empty()) {
        return densities;
    }
    const auto interactions = hadros::cascade::read_primary_interactions_jsonl(path);
    for (const auto& interaction : interactions) {
        densities[interaction.event_id] = interaction.point.density_g_cm3;
    }
    return densities;
}

bool supported_strict_pdg_app(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 22 || a == 11 || a == 13 || a == 211 || a == 321 ||
           a == 130 || a == 310 || a == 2212 || a == 2112;
}

double momentum2_gev2_app(const hadros::cascade::SecondaryParticle& particle) {
    return particle.px_gev * particle.px_gev +
           particle.py_gev * particle.py_gev +
           particle.pz_gev * particle.pz_gev;
}

std::string strict_drop_reason(const hadros::cascade::SecondaryParticle& particle,
                               const std::string& energy_convention) {
    const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
    if (hadros::cascade::is_neutrino_pdg(pdg)) {
        return "neutrino_invisible";
    }
    if (!supported_strict_pdg_app(pdg)) {
        return "unsupported_pdg";
    }
    if (!particle.stable) {
        return "unstable_secondary";
    }
    if (!std::isfinite(particle.energy_gev) || !std::isfinite(particle.px_gev) ||
        !std::isfinite(particle.py_gev) || !std::isfinite(particle.pz_gev) ||
        !std::isfinite(particle.mass_gev)) {
        return "nonfinite_kinematics";
    }
    if (particle.energy_gev < 0.0 || particle.mass_gev < 0.0) {
        return "negative_energy_or_mass";
    }
    if (momentum2_gev2_app(particle) <= 0.0) {
        return "zero_momentum";
    }
    if (energy_convention == "total" && particle.energy_gev + 1.0e-12 < particle.mass_gev) {
        return "total_energy_below_mass";
    }
    if (energy_convention == "kinetic" && particle.energy_gev < 0.0) {
        return "negative_kinetic_energy";
    }
    return "";
}

bool is_hadron_for_uhe_policy_app(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 211 || a == 321 || a == 130 || a == 310 ||
           a == 2212 || a == 2112 || a >= 1000;
}

bool is_lepton_for_uhe_policy_app(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 11 || a == 13 || a == 15;
}

double kinetic_energy_app(const hadros::cascade::SecondaryParticle& particle,
                          const std::string& energy_convention) {
    return energy_convention == "kinetic"
        ? particle.energy_gev
        : std::max(particle.energy_gev - particle.mass_gev, 0.0);
}

double uhe_threshold_app(const hadros::cascade::SecondaryParticle& particle,
                         const hadros::cascade::Geant4LocalBoxOptions& options) {
    const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
    const int a = pdg < 0 ? -pdg : pdg;
    if (a == 22) {
        return options.geant4_photon_max_kinetic_gev;
    }
    if (is_lepton_for_uhe_policy_app(pdg)) {
        return options.geant4_lepton_max_kinetic_gev;
    }
    if (is_hadron_for_uhe_policy_app(pdg)) {
        return options.geant4_hadron_max_kinetic_gev;
    }
    return options.geant4_hadron_max_kinetic_gev;
}

void write_safety_filter_report(const std::filesystem::path& path,
                                const std::vector<hadros::cascade::SecondaryParticle>& particles,
                                const std::string& safety_mode,
                                const std::string& energy_convention,
                                bool one_particle_per_run,
                                const hadros::cascade::Geant4LocalBoxOptions& options) {
    std::map<std::string, std::size_t> counts;
    std::map<std::string, double> energies;
    std::set<int> kept_pdgs;
    std::set<int> dropped_pdgs;
    std::size_t kept = 0;
    double kept_energy = 0.0;
    double dropped_energy = 0.0;
    for (const auto& particle : particles) {
        const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
        const auto reason = safety_mode == "strict" ? strict_drop_reason(particle, energy_convention) : "";
        if (reason.empty()) {
            ++kept;
            kept_energy += particle.energy_gev;
            kept_pdgs.insert(pdg);
        } else {
            counts[reason] += 1;
            energies[reason] += particle.energy_gev;
            dropped_energy += particle.energy_gev;
            dropped_pdgs.insert(pdg);
        }
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write " + path.string());
    }
    output << "# GEANT4 Safety Filter Report\n\n";
    output << "- safety_mode: `" << safety_mode << "`\n";
    output << "- energy_convention: `" << energy_convention << "`\n";
    output << "- geant4_one_particle_per_run: `" << (one_particle_per_run ? "true" : "false") << "`\n";
    output << "- uhe_transport_policy: `" << options.uhe_transport_policy << "`\n";
    output << "- geant4_hadron_max_kinetic_gev: `" << options.geant4_hadron_max_kinetic_gev << "`\n";
    output << "- geant4_lepton_max_kinetic_gev: `" << options.geant4_lepton_max_kinetic_gev << "`\n";
    output << "- geant4_photon_max_kinetic_gev: `" << options.geant4_photon_max_kinetic_gev << "`\n";
    output << "- total_particles: `" << particles.size() << "`\n";
    output << "- kept_particles: `" << kept << "`\n";
    output << "- dropped_particles: `" << (particles.size() - kept) << "`\n";
    output << "- kept_energy_gev: `" << std::setprecision(12) << kept_energy << "`\n";
    output << "- dropped_energy_gev: `" << std::setprecision(12) << dropped_energy << "`\n\n";
    output << "## Drop Reasons\n\n";
    output << "| Reason | Count | Energy [GeV] |\n";
    output << "|---|---:|---:|\n";
    for (const auto& [reason, count] : counts) {
        output << "| " << reason << " | " << count << " | " << energies[reason] << " |\n";
    }
    output << "\n## Kept PDGs\n\n";
    for (const auto pdg : kept_pdgs) {
        output << "- `" << pdg << "`\n";
    }
    output << "\n## Dropped PDGs\n\n";
    for (const auto pdg : dropped_pdgs) {
        output << "- `" << pdg << "`\n";
    }
}

void write_results_jsonl(const std::filesystem::path& path,
                         const std::vector<hadros::cascade::Geant4LocalBoxEventResult>& results,
                         const std::string& energy_convention) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write " + path.string());
    }
    output << std::setprecision(17);
    for (const auto& result : results) {
        output << "{\"event_id\":" << result.event_id
               << ",\"input_energy_gev\":" << result.input_energy_gev
               << ",\"deposited_energy_gev\":" << result.deposited_energy_gev
               << ",\"escaped_energy_gev\":" << result.escaped_energy_gev
               << ",\"invisible_energy_gev\":" << result.invisible_energy_gev
               << ",\"untracked_energy_gev\":" << result.untracked_energy_gev
               << ",\"unsupported_uhe_energy_gev\":" << result.unsupported_uhe_energy_gev
               << ",\"escaped_unsupported_uhe_energy_gev\":" << result.escaped_unsupported_uhe_energy_gev
               << ",\"n_unsupported_uhe_particles\":" << result.n_unsupported_uhe_particles
               << ",\"escaped_particle_count\":" << result.escaped_particles.size()
               << ",\"energy_convention\":\"" << energy_convention << "\""
               << ",\"uhe_transport_policy\":\"" << result.uhe_transport_policy << "\""
               << ",\"backend\":\"Geant4LocalBoxBackend\"}\n";
    }
}

void write_unsupported_uhe_jsonl(
    const std::filesystem::path& path,
    const std::vector<hadros::cascade::SecondaryParticle>& particles,
    const hadros::cascade::Geant4LocalBoxOptions& options) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write " + path.string());
    }
    output << std::setprecision(17);
    for (const auto& particle : particles) {
        const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
        const double kinetic = kinetic_energy_app(particle, options.energy_convention);
        const double threshold = uhe_threshold_app(particle, options);
        output << "{\"event_id\":" << particle.event_id
               << ",\"parent_event_id\":" << particle.parent_event_id
               << ",\"pdg\":" << particle.pdg
               << ",\"pdg_id\":" << pdg
               << ",\"energy_gev\":" << particle.energy_gev
               << ",\"kinetic_energy_gev\":" << kinetic
               << ",\"px_gev\":" << particle.px_gev
               << ",\"py_gev\":" << particle.py_gev
               << ",\"pz_gev\":" << particle.pz_gev
               << ",\"mass_gev\":" << particle.mass_gev
               << ",\"weight\":" << particle.weight
               << ",\"stable\":" << (particle.stable ? 1 : 0)
               << ",\"reason\":\"kinetic_energy_above_geant4_threshold\""
               << ",\"threshold_gev\":" << threshold
               << ",\"origin\":\"geant4_unsupported_uhe_policy\""
               << ",\"origin_backend\":\"geant4_unsupported_uhe_policy\"}\n";
    }
}

void write_energy_budget_csv(const std::filesystem::path& path,
                             const std::vector<hadros::cascade::Geant4LocalBoxEventResult>& results,
                             const std::string& energy_convention) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write " + path.string());
    }
    output << std::setprecision(17);
    output << "event_id,input_energy_gev,deposited_energy_gev,escaped_energy_gev,"
              "invisible_energy_gev,untracked_energy_gev,unsupported_uhe_energy_gev,"
              "escaped_unsupported_uhe_energy_gev,n_unsupported_uhe_particles,accounted_energy_gev,"
              "closure_error_gev,escaped_particle_count,energy_convention,uhe_transport_policy\n";
    for (const auto& result : results) {
        const double accounted = result.deposited_energy_gev + result.escaped_energy_gev +
            result.invisible_energy_gev + result.untracked_energy_gev +
            result.escaped_unsupported_uhe_energy_gev;
        output << result.event_id << ','
               << result.input_energy_gev << ','
               << result.deposited_energy_gev << ','
               << result.escaped_energy_gev << ','
               << result.invisible_energy_gev << ','
               << result.untracked_energy_gev << ','
               << result.unsupported_uhe_energy_gev << ','
               << result.escaped_unsupported_uhe_energy_gev << ','
               << result.n_unsupported_uhe_particles << ','
               << accounted << ','
               << (accounted - result.input_energy_gev) << ','
               << result.escaped_particles.size() << ','
               << energy_convention << ','
               << result.uhe_transport_policy << '\n';
    }
}

hadros::cascade::Geant4LocalBoxEventResult run_proxy_local_box_event(
    const std::vector<hadros::cascade::SecondaryParticle>& particles,
    const hadros::cascade::Geant4LocalBoxOptions& options) {
    using namespace hadros::cascade;

    Geant4LocalBoxEventResult result;
    if (!particles.empty()) {
        result.event_id = particles.front().event_id;
    }
    result.uhe_transport_policy = options.uhe_transport_policy;

    double transportable_energy = 0.0;
    for (const auto& particle : particles) {
        const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
        if (!std::isfinite(particle.energy_gev) || particle.energy_gev < 0.0) {
            throw std::runtime_error("GEANT4 local-box proxy received invalid input energy.");
        }
        result.input_energy_gev += particle.energy_gev;
        if (is_neutrino_pdg(pdg)) {
            result.invisible_energy_gev += particle.energy_gev;
        } else {
            transportable_energy += particle.energy_gev;
        }
    }

    const double areal_proxy = std::max(options.density_g_cm3, 0.0) * std::max(options.box_size_cm, 0.0);
    const double deposited_fraction = 1.0 - std::exp(-1.0e-2 * areal_proxy);
    result.deposited_energy_gev = transportable_energy * std::clamp(deposited_fraction, 0.0, 1.0);
    result.escaped_energy_gev = transportable_energy - result.deposited_energy_gev;

    for (const auto& particle : particles) {
        const int pdg = particle.pdg_id == 0 ? particle.pdg : particle.pdg_id;
        if (is_neutrino_pdg(pdg) || transportable_energy <= 0.0) {
            continue;
        }
        SecondaryParticle escaped = particle;
        escaped.energy_gev *= result.escaped_energy_gev / transportable_energy;
        escaped.px_gev *= result.escaped_energy_gev / transportable_energy;
        escaped.py_gev *= result.escaped_energy_gev / transportable_energy;
        escaped.pz_gev *= result.escaped_energy_gev / transportable_energy;
        escaped.origin = "geant4_local_box_proxy_escape";
        escaped.origin_backend = "Geant4LocalBoxBackendProxy";
        result.escaped_particles.push_back(escaped);
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace hadros::cascade;

    if (argc < 7) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const std::string secondaries_path = argv[1];
        const std::filesystem::path output_dir = argv[2];
        const double box_size_cm = std::stod(argv[3]);
        const double density_g_cm3 = std::stod(argv[4]);
        const std::string physics_list = argv[5];
        const std::string material = argv[6];
        std::string transport_mode = "geant4";
        std::string interactions_path;
        std::string energy_convention = "total";
        std::string geant4_safety_mode = "off";
        std::string uhe_transport_policy = "error";
        double geant4_hadron_max_kinetic_gev = 1.0e5;
        double geant4_lepton_max_kinetic_gev = 1.0e9;
        double geant4_photon_max_kinetic_gev = 1.0e9;
        double geant4_local_cm_per_rg = 1.0;
        bool geant4_one_particle_per_run = false;
        bool debug_single_particle = false;
        for (int i = 7; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--energy-convention") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--energy-convention requires total or kinetic.");
                }
                energy_convention = argv[++i];
            } else if (arg == "--geant4-safety-mode") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--geant4-safety-mode requires off or strict.");
                }
                geant4_safety_mode = argv[++i];
            } else if (arg == "--uhe-transport-policy") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--uhe-transport-policy requires error, skip_to_escaped, or split_energy_proxy.");
                }
                uhe_transport_policy = argv[++i];
            } else if (arg == "--geant4-hadron-max-kinetic-gev") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--geant4-hadron-max-kinetic-gev requires a value.");
                }
                geant4_hadron_max_kinetic_gev = std::stod(argv[++i]);
            } else if (arg == "--geant4-lepton-max-kinetic-gev") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--geant4-lepton-max-kinetic-gev requires a value.");
                }
                geant4_lepton_max_kinetic_gev = std::stod(argv[++i]);
            } else if (arg == "--geant4-photon-max-kinetic-gev") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--geant4-photon-max-kinetic-gev requires a value.");
                }
                geant4_photon_max_kinetic_gev = std::stod(argv[++i]);
            } else if (arg == "--geant4-local-cm-per-rg") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--geant4-local-cm-per-rg requires a value.");
                }
                geant4_local_cm_per_rg = std::stod(argv[++i]);
            } else if (arg == "--geant4-one-particle-per-run") {
                geant4_one_particle_per_run = true;
            } else if (arg == "--debug-single-particle") {
                debug_single_particle = true;
            } else if (arg == "geant4" || arg == "proxy") {
                transport_mode = arg;
            } else {
                interactions_path = arg;
            }
        }
        if (energy_convention != "total" && energy_convention != "kinetic") {
            throw std::runtime_error("Energy convention must be 'total' or 'kinetic'.");
        }
        if (geant4_safety_mode != "off" && geant4_safety_mode != "strict") {
            throw std::runtime_error("GEANT4 safety mode must be 'off' or 'strict'.");
        }
        if (uhe_transport_policy != "error" &&
            uhe_transport_policy != "skip_to_escaped" &&
            uhe_transport_policy != "split_energy_proxy") {
            throw std::runtime_error("UHE transport policy must be 'error', 'skip_to_escaped', or 'split_energy_proxy'.");
        }
        if (geant4_hadron_max_kinetic_gev <= 0.0 ||
            geant4_lepton_max_kinetic_gev <= 0.0 ||
            geant4_photon_max_kinetic_gev <= 0.0) {
            throw std::runtime_error("GEANT4 UHE kinetic thresholds must be positive.");
        }
        if (geant4_local_cm_per_rg <= 0.0 || !std::isfinite(geant4_local_cm_per_rg)) {
            throw std::runtime_error("GEANT4 local cm-per-rg scale must be finite and positive.");
        }

        std::filesystem::create_directories(output_dir);
        const auto particles = read_secondaries_jsonl(secondaries_path);
        Geant4LocalBoxOptions base_options;
        base_options.box_size_cm = box_size_cm;
        base_options.density_g_cm3 = density_g_cm3;
        base_options.physics_list = physics_list;
        base_options.material = material;
        base_options.energy_convention = energy_convention;
        base_options.safety_mode = geant4_safety_mode;
        base_options.one_particle_per_run = geant4_one_particle_per_run;
        base_options.debug_single_particle = debug_single_particle;
        base_options.uhe_transport_policy = uhe_transport_policy;
        base_options.geant4_hadron_max_kinetic_gev = geant4_hadron_max_kinetic_gev;
        base_options.geant4_lepton_max_kinetic_gev = geant4_lepton_max_kinetic_gev;
        base_options.geant4_photon_max_kinetic_gev = geant4_photon_max_kinetic_gev;
        base_options.geant4_local_cm_per_rg = geant4_local_cm_per_rg;
        write_safety_filter_report(output_dir / "geant4_safety_filter_report.md",
                                   particles,
                                   geant4_safety_mode,
                                   energy_convention,
                                   geant4_one_particle_per_run,
                                   base_options);
        const auto grouped = group_by_event(particles);
        const auto densities = read_density_map(interactions_path);

        std::vector<Geant4LocalBoxEventResult> results;
        std::vector<SecondaryParticle> escaped_all;
        std::vector<SecondaryParticle> unsupported_uhe_all;
        results.reserve(grouped.size());

        for (const auto& [event_id, event_particles] : grouped) {
            Geant4LocalBoxOptions options = base_options;
            const auto density_it = densities.find(event_id);
            if (density_it != densities.end() && density_it->second > 0.0 && std::isfinite(density_it->second)) {
                options.density_g_cm3 = density_it->second;
            }

            auto result = transport_mode == "proxy"
                ? run_proxy_local_box_event(event_particles, options)
                : run_geant4_local_box_event(event_particles, options);
            result.event_id = event_id;
            for (auto& escaped : result.escaped_particles) {
                escaped.event_id = event_id;
                escaped.parent_event_id = event_id;
            }
            for (auto& unsupported : result.unsupported_uhe_particles) {
                unsupported.event_id = event_id;
                unsupported.parent_event_id = event_id;
            }
            escaped_all.insert(escaped_all.end(), result.escaped_particles.begin(), result.escaped_particles.end());
            unsupported_uhe_all.insert(unsupported_uhe_all.end(),
                                       result.unsupported_uhe_particles.begin(),
                                       result.unsupported_uhe_particles.end());
            results.push_back(result);
        }

        write_results_jsonl(output_dir / "geant4_cascade_results.jsonl", results, energy_convention);
        write_energy_budget_csv(output_dir / "geant4_energy_budget.csv", results, energy_convention);
        write_secondaries_jsonl((output_dir / "geant4_escaped_particles.jsonl").string(), escaped_all);
        write_unsupported_uhe_jsonl(output_dir / "geant4_unsupported_uhe_particles.jsonl",
                                    unsupported_uhe_all,
                                    base_options);

        double input = 0.0;
        double deposited = 0.0;
        double escaped = 0.0;
        double invisible = 0.0;
        double unsupported_uhe = 0.0;
        double escaped_unsupported_uhe = 0.0;
        for (const auto& result : results) {
            input += result.input_energy_gev;
            deposited += result.deposited_energy_gev;
            escaped += result.escaped_energy_gev;
            invisible += result.invisible_energy_gev;
            unsupported_uhe += result.unsupported_uhe_energy_gev;
            escaped_unsupported_uhe += result.escaped_unsupported_uhe_energy_gev;
        }

        std::cout.setf(std::ios::scientific);
        std::cout.precision(12);
        std::cout << "geant4_local_box_events=" << results.size() << "\n";
        std::cout << "input_energy_gev=" << input << "\n";
        std::cout << "deposited_energy_gev=" << deposited << "\n";
        std::cout << "escaped_energy_gev=" << escaped << "\n";
        std::cout << "invisible_energy_gev=" << invisible << "\n";
        std::cout << "unsupported_uhe_energy_gev=" << unsupported_uhe << "\n";
        std::cout << "escaped_unsupported_uhe_energy_gev=" << escaped_unsupported_uhe << "\n";
        double untracked = 0.0;
        for (const auto& result : results) {
            untracked += result.untracked_energy_gev;
        }
        std::cout << "untracked_energy_gev=" << untracked << "\n";
        std::cout << "closure_error_gev=" << (deposited + escaped + invisible + untracked + escaped_unsupported_uhe - input) << "\n";
        std::cout << "energy_convention=" << energy_convention << "\n";
        std::cout << "geant4_safety_mode=" << geant4_safety_mode << "\n";
        std::cout << "geant4_one_particle_per_run=" << (geant4_one_particle_per_run ? "true" : "false") << "\n";
        std::cout << "debug_single_particle=" << (debug_single_particle ? "true" : "false") << "\n";
        std::cout << "uhe_transport_policy=" << uhe_transport_policy << "\n";
        std::cout << "geant4_hadron_max_kinetic_gev=" << geant4_hadron_max_kinetic_gev << "\n";
        std::cout << "geant4_lepton_max_kinetic_gev=" << geant4_lepton_max_kinetic_gev << "\n";
        std::cout << "geant4_photon_max_kinetic_gev=" << geant4_photon_max_kinetic_gev << "\n";
        std::cout << "NOTE: GEANT4 local homogeneous box only; not global collapsar transport.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << "cascade_geant4_local_box failed: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
}
