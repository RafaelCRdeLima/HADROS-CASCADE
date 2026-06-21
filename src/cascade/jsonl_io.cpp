#include "hadros/cascade/jsonl_io.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace hadros::cascade {
namespace {

std::ifstream open_input(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open input file: " + path);
    }
    return input;
}

std::ofstream open_output(const std::string& path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + path);
    }
    output << std::setprecision(17);
    return output;
}

double number_value(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto begin = line.find(needle);
    if (begin == std::string::npos) {
        throw std::runtime_error("JSONL line missing numeric key: " + key);
    }
    const auto value_begin = begin + needle.size();
    const auto value_end = line.find_first_of(",}", value_begin);
    return std::stod(line.substr(value_begin, value_end - value_begin));
}

bool has_key(const std::string& line, const std::string& key) {
    return line.find("\"" + key + "\":") != std::string::npos;
}

double number_value_or(const std::string& line, const std::string& key, double fallback) {
    return has_key(line, key) ? number_value(line, key) : fallback;
}

std::uint64_t uint_value(const std::string& line, const std::string& key) {
    return static_cast<std::uint64_t>(number_value(line, key));
}

std::uint64_t uint_value_or(const std::string& line, const std::string& key, std::uint64_t fallback) {
    return has_key(line, key) ? static_cast<std::uint64_t>(number_value(line, key)) : fallback;
}

int int_value(const std::string& line, const std::string& key) {
    return static_cast<int>(number_value(line, key));
}

int int_value_or(const std::string& line, const std::string& key, int fallback) {
    return has_key(line, key) ? static_cast<int>(number_value(line, key)) : fallback;
}

bool bool_value(const std::string& line, const std::string& key) {
    return number_value(line, key) != 0.0;
}

bool bool_value_or(const std::string& line, const std::string& key, bool fallback) {
    return has_key(line, key) ? number_value(line, key) != 0.0 : fallback;
}

std::string string_value(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto begin = line.find(needle);
    if (begin == std::string::npos) {
        throw std::runtime_error("JSONL line missing string key: " + key);
    }
    const auto value_begin = begin + needle.size();
    const auto value_end = line.find('"', value_begin);
    return line.substr(value_begin, value_end - value_begin);
}

std::string string_value_or(const std::string& line, const std::string& key, const std::string& fallback) {
    const std::string needle = "\"" + key + "\":\"";
    return line.find(needle) == std::string::npos ? fallback : string_value(line, key);
}

void write_optional_number(std::ostream& output,
                           const std::string& key,
                           double value) {
    if (std::isfinite(value)) {
        output << ",\"" << key << "\":" << value;
    }
}

void write_primary_particle_json(std::ostream& output, const PrimaryParticle& particle) {
    output << "\"event_id\":" << particle.event_id
           << ",\"pdg_id\":" << particle.pdg_id
           << ",\"energy_gev\":" << particle.energy_gev
           << ",\"px_gev\":" << particle.px_gev
           << ",\"py_gev\":" << particle.py_gev
           << ",\"pz_gev\":" << particle.pz_gev
           << ",\"mass_gev\":" << particle.mass_gev
           << ",\"weight\":" << particle.weight
           << ",\"seed\":" << particle.seed
           << ",\"particle_label\":\"" << particle.particle_label << "\"";
}

PrimaryParticle read_primary_particle_json(const std::string& line) {
    PrimaryParticle particle;
    particle.event_id = uint_value(line, "event_id");
    particle.pdg_id = int_value(line, "pdg_id");
    particle.energy_gev = number_value(line, "energy_gev");
    particle.px_gev = number_value_or(line, "px_gev", 0.0);
    particle.py_gev = number_value_or(line, "py_gev", 0.0);
    particle.pz_gev = number_value_or(line, "pz_gev", particle.energy_gev);
    particle.mass_gev = number_value_or(line, "mass_gev", 0.0);
    particle.weight = number_value(line, "weight");
    particle.seed = uint_value_or(line, "seed", particle.event_id);
    particle.particle_label = string_value_or(line, "particle_label", "unspecified");
    return particle;
}

void write_interaction_json(std::ostream& output, const InteractionPoint& point) {
    output << "\"event_id\":" << point.event_id
           << ",\"x_cm\":" << point.x_cm
           << ",\"y_cm\":" << point.y_cm
           << ",\"z_cm\":" << point.z_cm
           << ",\"r_cm\":" << point.r_cm
           << ",\"theta_rad\":" << point.theta_rad
           << ",\"phi_rad\":" << point.phi_rad
           << ",\"density_g_cm3\":" << point.density_g_cm3
           << ",\"temperature_mev\":" << point.temperature_mev
           << ",\"temperature_proxy\":" << point.temperature_proxy
           << ",\"composition_proxy\":" << point.composition_proxy
           << ",\"electron_fraction\":" << point.electron_fraction
           << ",\"column_before_cm2\":" << point.column_before_cm2
           << ",\"tau_before\":" << point.tau_before
           << ",\"weight\":" << point.weight
           << ",\"region_label\":\"" << point.region_label << "\""
           << ",\"region_class\":\"" << point.region_class << "\"";
}

InteractionPoint read_interaction_json(const std::string& line) {
    InteractionPoint point;
    point.event_id = uint_value(line, "event_id");
    point.x_cm = number_value(line, "x_cm");
    point.y_cm = number_value(line, "y_cm");
    point.z_cm = number_value(line, "z_cm");
    point.r_cm = number_value_or(line, "r_cm", 0.0);
    point.theta_rad = number_value_or(line, "theta_rad", 0.0);
    point.phi_rad = number_value_or(line, "phi_rad", 0.0);
    point.density_g_cm3 = number_value(line, "density_g_cm3");
    point.temperature_mev = number_value_or(line, "temperature_mev", 0.0);
    point.temperature_proxy = number_value_or(line, "temperature_proxy", point.temperature_mev);
    point.composition_proxy = number_value_or(line, "composition_proxy", 0.0);
    point.electron_fraction = number_value(line, "electron_fraction");
    point.column_before_cm2 = number_value(line, "column_before_cm2");
    point.tau_before = number_value(line, "tau_before");
    point.weight = number_value(line, "weight");
    point.region_label = string_value_or(line, "region_label", string_value_or(line, "region_class", "unspecified"));
    point.region_class = string_value(line, "region_class");
    return point;
}

}  // namespace

void write_interaction_points_jsonl(const std::string& path,
                                    const std::vector<InteractionPoint>& points) {
    auto output = open_output(path);
    for (const auto& point : points) {
        output << "{";
        write_interaction_json(output, point);
        output << "}\n";
    }
}

std::vector<InteractionPoint> read_interaction_points_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<InteractionPoint> points;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            points.push_back(read_interaction_json(line));
        }
    }
    return points;
}

void write_primary_particles_jsonl(const std::string& path,
                                   const std::vector<PrimaryParticle>& particles) {
    auto output = open_output(path);
    for (const auto& particle : particles) {
        output << "{";
        write_primary_particle_json(output, particle);
        output << "}\n";
    }
}

std::vector<PrimaryParticle> read_primary_particles_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<PrimaryParticle> particles;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            particles.push_back(read_primary_particle_json(line));
        }
    }
    return particles;
}

void write_primary_interactions_jsonl(const std::string& path,
                                      const std::vector<PrimaryInteractionEvent>& events) {
    auto output = open_output(path);
    for (const auto& event : events) {
        output << "{\"event_id\":" << event.event_id << ",\"primary\":{";
        write_primary_particle_json(output, event.primary);
        output << "},\"point\":{";
        write_interaction_json(output, event.point);
        output << "},\"interaction_model\":\"" << event.interaction_model
               << "\",\"backend_name\":\"" << event.backend_name
               << "\",\"x_bjorken\":" << event.x_bjorken
               << ",\"q2_gev2\":" << event.q2_gev2
               << ",\"y_inelasticity\":" << event.y_inelasticity
               << ",\"metadata\":\"" << event.metadata << "\"}\n";
    }
}

std::vector<PrimaryInteractionEvent> read_primary_interactions_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<PrimaryInteractionEvent> events;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        PrimaryInteractionEvent event;
        event.event_id = uint_value(line, "event_id");
        event.primary = read_primary_particle_json(line);
        event.point = read_interaction_json(line);
        event.interaction_model = string_value_or(line, "interaction_model", "unspecified");
        event.backend_name = string_value_or(line, "backend_name", "unspecified");
        event.x_bjorken = number_value_or(line, "x_bjorken", -1.0);
        event.q2_gev2 = number_value_or(line, "q2_gev2", -1.0);
        event.y_inelasticity = number_value_or(line, "y_inelasticity", -1.0);
        event.metadata = string_value_or(line, "metadata", "");
        events.push_back(event);
    }
    return events;
}

void write_primary_events_jsonl(const std::string& path,
                                const std::vector<PrimaryNeutrinoEvent>& events) {
    auto output = open_output(path);
    for (const auto& event : events) {
        output << "{\"event_id\":" << event.event_id
               << ",\"neutrino_pdg\":" << event.neutrino_pdg
               << ",\"energy_gev\":" << event.energy_gev
               << ",\"weight\":" << event.weight
               << ",\"charged_current\":" << (event.charged_current ? 1 : 0)
               << ",\"interaction\":{";
        write_interaction_json(output, event.interaction);
        output << "}}\n";
    }
}

std::vector<PrimaryNeutrinoEvent> read_primary_events_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<PrimaryNeutrinoEvent> events;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        PrimaryNeutrinoEvent event;
        event.event_id = uint_value(line, "event_id");
        event.neutrino_pdg = int_value(line, "neutrino_pdg");
        event.energy_gev = number_value(line, "energy_gev");
        event.weight = number_value(line, "weight");
        event.charged_current = bool_value(line, "charged_current");
        event.interaction = read_interaction_json(line);
        events.push_back(event);
    }
    return events;
}

void write_secondaries_jsonl(const std::string& path,
                             const std::vector<SecondaryParticle>& particles) {
    auto output = open_output(path);
    for (const auto& particle : particles) {
        output << "{\"event_id\":" << particle.event_id
               << ",\"parent_event_id\":" << particle.parent_event_id
               << ",\"pdg\":" << particle.pdg
               << ",\"pdg_id\":" << (particle.pdg_id == 0 ? particle.pdg : particle.pdg_id)
               << ",\"energy_gev\":" << particle.energy_gev
               << ",\"px_gev\":" << particle.px_gev
               << ",\"py_gev\":" << particle.py_gev
               << ",\"pz_gev\":" << particle.pz_gev
               << ",\"mass_gev\":" << particle.mass_gev
               << ",\"weight\":" << particle.weight
               << ",\"stable\":" << (particle.stable ? 1 : 0)
               << ",\"origin\":\"" << particle.origin
               << "\",\"origin_backend\":\"" << particle.origin_backend << "\"";
        write_optional_number(output, "interaction_x_rg", particle.interaction_x_rg);
        write_optional_number(output, "interaction_y_rg", particle.interaction_y_rg);
        write_optional_number(output, "interaction_z_rg", particle.interaction_z_rg);
        write_optional_number(output, "geant4_box_origin_x_rg", particle.geant4_box_origin_x_rg);
        write_optional_number(output, "geant4_box_origin_y_rg", particle.geant4_box_origin_y_rg);
        write_optional_number(output, "geant4_box_origin_z_rg", particle.geant4_box_origin_z_rg);
        write_optional_number(output, "exit_x_rg", particle.exit_x_rg);
        write_optional_number(output, "exit_y_rg", particle.exit_y_rg);
        write_optional_number(output, "exit_z_rg", particle.exit_z_rg);
        write_optional_number(output, "geant4_box_origin_x_cm", particle.geant4_box_origin_x_cm);
        write_optional_number(output, "geant4_box_origin_y_cm", particle.geant4_box_origin_y_cm);
        write_optional_number(output, "geant4_box_origin_z_cm", particle.geant4_box_origin_z_cm);
        write_optional_number(output, "geant4_local_exit_x_cm", particle.geant4_local_exit_x_cm);
        write_optional_number(output, "geant4_local_exit_y_cm", particle.geant4_local_exit_y_cm);
        write_optional_number(output, "geant4_local_exit_z_cm", particle.geant4_local_exit_z_cm);
        write_optional_number(output, "geant4_local_cm_per_rg", particle.geant4_local_cm_per_rg);
        output << ",\"position_status\":\"" << particle.position_status << "\"}\n";
    }
}

std::vector<SecondaryParticle> read_secondaries_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<SecondaryParticle> particles;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        SecondaryParticle particle;
        particle.event_id = uint_value(line, "event_id");
        particle.parent_event_id = uint_value_or(line, "parent_event_id", particle.event_id);
        particle.pdg = int_value_or(line, "pdg", int_value_or(line, "pdg_id", 0));
        particle.pdg_id = int_value_or(line, "pdg_id", particle.pdg);
        particle.energy_gev = number_value(line, "energy_gev");
        particle.px_gev = number_value(line, "px_gev");
        particle.py_gev = number_value(line, "py_gev");
        particle.pz_gev = number_value(line, "pz_gev");
        particle.mass_gev = number_value_or(line, "mass_gev", 0.0);
        particle.weight = number_value(line, "weight");
        particle.stable = bool_value_or(line, "stable", true);
        particle.origin = string_value_or(line, "origin", string_value_or(line, "origin_backend", "unspecified"));
        particle.origin_backend = string_value_or(line, "origin_backend", particle.origin);
        particle.interaction_x_rg = number_value_or(line, "interaction_x_rg", particle.interaction_x_rg);
        particle.interaction_y_rg = number_value_or(line, "interaction_y_rg", particle.interaction_y_rg);
        particle.interaction_z_rg = number_value_or(line, "interaction_z_rg", particle.interaction_z_rg);
        particle.geant4_box_origin_x_rg = number_value_or(line, "geant4_box_origin_x_rg", particle.geant4_box_origin_x_rg);
        particle.geant4_box_origin_y_rg = number_value_or(line, "geant4_box_origin_y_rg", particle.geant4_box_origin_y_rg);
        particle.geant4_box_origin_z_rg = number_value_or(line, "geant4_box_origin_z_rg", particle.geant4_box_origin_z_rg);
        particle.exit_x_rg = number_value_or(line, "exit_x_rg", particle.exit_x_rg);
        particle.exit_y_rg = number_value_or(line, "exit_y_rg", particle.exit_y_rg);
        particle.exit_z_rg = number_value_or(line, "exit_z_rg", particle.exit_z_rg);
        particle.geant4_box_origin_x_cm = number_value_or(line, "geant4_box_origin_x_cm", particle.geant4_box_origin_x_cm);
        particle.geant4_box_origin_y_cm = number_value_or(line, "geant4_box_origin_y_cm", particle.geant4_box_origin_y_cm);
        particle.geant4_box_origin_z_cm = number_value_or(line, "geant4_box_origin_z_cm", particle.geant4_box_origin_z_cm);
        particle.geant4_local_exit_x_cm = number_value_or(line, "geant4_local_exit_x_cm", particle.geant4_local_exit_x_cm);
        particle.geant4_local_exit_y_cm = number_value_or(line, "geant4_local_exit_y_cm", particle.geant4_local_exit_y_cm);
        particle.geant4_local_exit_z_cm = number_value_or(line, "geant4_local_exit_z_cm", particle.geant4_local_exit_z_cm);
        particle.geant4_local_cm_per_rg = number_value_or(line, "geant4_local_cm_per_rg", particle.geant4_local_cm_per_rg);
        particle.position_status = string_value_or(line, "position_status", particle.position_status);
        particles.push_back(particle);
    }
    return particles;
}

void write_cascade_results_jsonl(const std::string& path,
                                 const std::vector<CascadeResult>& results) {
    auto output = open_output(path);
    for (const auto& result : results) {
        output << "{\"event_id\":" << result.event_id
               << ",\"weight\":" << result.weight
               << ",\"deposited_em_gev\":" << result.deposited_em_gev
               << ",\"deposited_hadronic_gev\":" << result.deposited_hadronic_gev
               << ",\"escaped_muon_gev\":" << result.escaped_muon_gev
               << ",\"escaped_neutrino_gev\":" << result.escaped_neutrino_gev
               << ",\"deposited_total_gev\":" << result.deposited_energy_gev()
               << ",\"escaped_total_gev\":" << result.escaped_energy_gev()
               << ",\"escaped_particle_count\":" << result.escaped_particles.size()
               << "}\n";
    }
}

std::vector<CascadeResult> read_cascade_results_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<CascadeResult> results;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        CascadeResult result;
        result.event_id = uint_value(line, "event_id");
        result.weight = number_value(line, "weight");
        result.deposited_em_gev = number_value(line, "deposited_em_gev");
        result.deposited_hadronic_gev = number_value(line, "deposited_hadronic_gev");
        result.escaped_muon_gev = number_value(line, "escaped_muon_gev");
        result.escaped_neutrino_gev = number_value(line, "escaped_neutrino_gev");
        results.push_back(result);
    }
    return results;
}

void write_cascade_energy_budget_csv(const std::string& path,
                                     const std::vector<CascadeResult>& results) {
    auto output = open_output(path);
    output << "event_id,weight,deposited_em_gev,deposited_hadronic_gev,"
              "escaped_muon_gev,escaped_neutrino_gev,deposited_total_gev,"
              "escaped_total_gev,total_accounted_gev\n";
    for (const auto& result : results) {
        output << result.event_id << ','
               << result.weight << ','
               << result.deposited_em_gev << ','
               << result.deposited_hadronic_gev << ','
               << result.escaped_muon_gev << ','
               << result.escaped_neutrino_gev << ','
               << result.deposited_energy_gev() << ','
               << result.escaped_energy_gev() << ','
               << result.total_accounted_energy_gev() << '\n';
    }
}

void write_interaction_results_jsonl(const std::string& path,
                                     const std::vector<InteractionResult>& results) {
    auto output = open_output(path);
    for (const auto& result : results) {
        output << "{\"event_id\":" << result.event_id
               << ",\"input_energy_gev\":" << result.input_energy_gev
               << ",\"visible_energy_gev\":" << result.visible_energy_gev
               << ",\"invisible_energy_gev\":" << result.invisible_energy_gev
               << ",\"escaped_energy_gev\":" << result.escaped_energy_gev
               << ",\"deposited_energy_gev\":" << result.deposited_energy_gev
               << ",\"secondary_count\":" << result.secondaries.size()
               << ",\"metadata\":\"" << result.metadata << "\"}\n";
    }
}

std::vector<InteractionResult> read_interaction_results_jsonl(const std::string& path) {
    auto input = open_input(path);
    std::vector<InteractionResult> results;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        InteractionResult result;
        result.event_id = uint_value(line, "event_id");
        result.input_energy_gev = number_value(line, "input_energy_gev");
        result.visible_energy_gev = number_value(line, "visible_energy_gev");
        result.invisible_energy_gev = number_value(line, "invisible_energy_gev");
        result.escaped_energy_gev = number_value_or(line, "escaped_energy_gev", 0.0);
        result.deposited_energy_gev = number_value_or(line, "deposited_energy_gev", result.visible_energy_gev);
        result.metadata = string_value_or(line, "metadata", "");
        results.push_back(result);
    }
    return results;
}

}  // namespace hadros::cascade
