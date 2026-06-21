#include "hadros/cascade/local_response_table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace hadros::cascade {
namespace {

constexpr double kMatchTolerance = 1.0e-10;

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (const char ch : line) {
        if (ch == '"') {
            quoted = !quoted;
        } else if (ch == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(field);
    return fields;
}

double to_double(const std::map<std::string, std::string>& row, const std::string& key, double fallback = 0.0) {
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
        return fallback;
    }
    return std::stod(it->second);
}

int to_int(const std::map<std::string, std::string>& row, const std::string& key, int fallback = 0) {
    return static_cast<int>(std::lround(to_double(row, key, fallback)));
}

std::string to_string_value(const std::map<std::string, std::string>& row, const std::string& key) {
    const auto it = row.find(key);
    return it == row.end() ? std::string{} : it->second;
}

bool close(double a, double b) {
    return std::abs(a - b) <= kMatchTolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

bool discrete_match(const LocalResponseTable::Row& row, const LocalResponseQuery& query) {
    return row.pdg_id == query.pdg_id &&
           row.material == query.material &&
           close(row.box_size_cm, query.box_size_cm) &&
           row.physics_list == query.physics_list &&
           row.status == "PASS";
}

LocalResponseResult from_row(const LocalResponseTable::Row& row, const std::string& status, const std::string& mode) {
    LocalResponseResult result;
    result.valid = true;
    result.status = status;
    result.deposited_fraction = row.deposited_fraction;
    result.escaped_fraction = row.escaped_fraction;
    result.invisible_fraction = row.invisible_fraction;
    result.untracked_fraction = row.untracked_fraction;
    result.energy_closure_error = row.energy_closure_error;
    result.interpolation_mode = mode;
    return result;
}

LocalResponseResult status_result(const std::string& status) {
    LocalResponseResult result;
    result.valid = false;
    result.status = status;
    result.interpolation_mode = "none";
    return result;
}

double clamp_fraction(double value) {
    if (!std::isfinite(value)) {
        return value;
    }
    return std::clamp(value, 0.0, 1.0);
}

double bilinear(double f00, double f10, double f01, double f11, double tx, double ty) {
    return (1.0 - tx) * (1.0 - ty) * f00 +
           tx * (1.0 - ty) * f10 +
           (1.0 - tx) * ty * f01 +
           tx * ty * f11;
}

}  // namespace

bool LocalResponseTable::load_csv(const std::string& path) {
    rows_.clear();
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open local response table: " + path);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        return false;
    }
    const auto headers = split_csv_line(header_line);

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv_line(line);
        std::map<std::string, std::string> values;
        for (std::size_t i = 0; i < headers.size() && i < fields.size(); ++i) {
            values[headers[i]] = fields[i];
        }

        Row row;
        row.pdg_id = to_int(values, "pdg_id");
        row.energy_gev = to_double(values, "energy_gev");
        row.density_g_cm3 = to_double(values, "density_g_cm3");
        row.material = to_string_value(values, "material");
        row.box_size_cm = to_double(values, "box_size_cm");
        row.physics_list = to_string_value(values, "physics_list");
        row.deposited_fraction = to_double(values, "deposited_fraction");
        row.escaped_fraction = to_double(values, "escaped_fraction");
        row.invisible_fraction = to_double(values, "invisible_fraction");
        row.untracked_fraction = to_double(values, "untracked_fraction");
        row.energy_closure_error = to_double(values, "energy_closure_error");
        row.status = to_string_value(values, "status");
        rows_.push_back(row);
    }
    return !rows_.empty();
}

LocalResponseResult LocalResponseTable::query_nearest(const LocalResponseQuery& query) const {
    if (rows_.empty()) {
        return status_result("EMPTY_TABLE");
    }

    const bool has_pdg = std::any_of(rows_.begin(), rows_.end(), [&](const Row& row) { return row.pdg_id == query.pdg_id; });
    if (!has_pdg) {
        return status_result("MISSING_PDG");
    }
    const bool has_material = std::any_of(rows_.begin(), rows_.end(), [&](const Row& row) {
        return row.pdg_id == query.pdg_id && row.material == query.material;
    });
    if (!has_material) {
        return status_result("MISSING_MATERIAL");
    }
    const bool has_physics = std::any_of(rows_.begin(), rows_.end(), [&](const Row& row) {
        return row.pdg_id == query.pdg_id && row.material == query.material && row.physics_list == query.physics_list;
    });
    if (!has_physics) {
        return status_result("MISSING_PHYSICS_LIST");
    }

    const Row* best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto& row : rows_) {
        if (!discrete_match(row, query)) {
            continue;
        }
        const double le = std::log(row.energy_gev) - std::log(query.kinetic_energy_gev);
        const double lr = std::log(row.density_g_cm3) - std::log(query.density_g_cm3);
        const double distance = le * le + lr * lr;
        if (distance < best_distance) {
            best_distance = distance;
            best = &row;
        }
    }
    if (best == nullptr) {
        return status_result("OUT_OF_RANGE");
    }
    return from_row(*best, "OK_NEAREST", "nearest");
}

LocalResponseResult LocalResponseTable::query_interpolated(const LocalResponseQuery& query) const {
    if (rows_.empty()) {
        return status_result("EMPTY_TABLE");
    }
    if (query.kinetic_energy_gev <= 0.0 || query.density_g_cm3 <= 0.0) {
        return status_result("OUT_OF_RANGE");
    }

    const auto nearest_status = query_nearest(query);
    if (nearest_status.status == "MISSING_PDG" ||
        nearest_status.status == "MISSING_MATERIAL" ||
        nearest_status.status == "MISSING_PHYSICS_LIST" ||
        nearest_status.status == "EMPTY_TABLE") {
        return nearest_status;
    }

    std::vector<double> energies;
    std::vector<double> densities;
    for (const auto& row : rows_) {
        if (!discrete_match(row, query)) {
            continue;
        }
        energies.push_back(row.energy_gev);
        densities.push_back(row.density_g_cm3);
    }
    auto unique_sorted = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(), close), values.end());
        return values;
    };
    energies = unique_sorted(energies);
    densities = unique_sorted(densities);
    if (energies.empty() || densities.empty() ||
        query.kinetic_energy_gev < energies.front() || query.kinetic_energy_gev > energies.back() ||
        query.density_g_cm3 < densities.front() || query.density_g_cm3 > densities.back()) {
        return status_result("OUT_OF_RANGE");
    }

    auto bracket = [](const std::vector<double>& grid, double value) {
        auto upper = std::lower_bound(grid.begin(), grid.end(), value);
        if (upper == grid.end()) {
            return std::pair<double, double>{grid.back(), grid.back()};
        }
        if (close(*upper, value) || upper == grid.begin()) {
            return std::pair<double, double>{*upper, *upper};
        }
        return std::pair<double, double>{*(upper - 1), *upper};
    };
    const auto [e0, e1] = bracket(energies, query.kinetic_energy_gev);
    const auto [r0, r1] = bracket(densities, query.density_g_cm3);

    const auto find_corner = [&](double energy, double density) -> const Row* {
        for (const auto& row : rows_) {
            if (discrete_match(row, query) && close(row.energy_gev, energy) && close(row.density_g_cm3, density)) {
                return &row;
            }
        }
        return nullptr;
    };

    const Row* c00 = find_corner(e0, r0);
    const Row* c10 = find_corner(e1, r0);
    const Row* c01 = find_corner(e0, r1);
    const Row* c11 = find_corner(e1, r1);
    if (c00 == nullptr || c10 == nullptr || c01 == nullptr || c11 == nullptr) {
        return status_result("OUT_OF_RANGE");
    }

    if (close(e0, e1) && close(r0, r1)) {
        return from_row(*c00, "OK_INTERPOLATED", "exact_grid");
    }

    const double tx = close(e0, e1) ? 0.0 :
        (std::log(query.kinetic_energy_gev) - std::log(e0)) / (std::log(e1) - std::log(e0));
    const double ty = close(r0, r1) ? 0.0 :
        (std::log(query.density_g_cm3) - std::log(r0)) / (std::log(r1) - std::log(r0));

    LocalResponseResult result;
    result.valid = true;
    result.status = "OK_INTERPOLATED";
    result.interpolation_mode = "bilinear_log_energy_log_density";
    result.deposited_fraction = clamp_fraction(bilinear(c00->deposited_fraction, c10->deposited_fraction, c01->deposited_fraction, c11->deposited_fraction, tx, ty));
    result.escaped_fraction = clamp_fraction(bilinear(c00->escaped_fraction, c10->escaped_fraction, c01->escaped_fraction, c11->escaped_fraction, tx, ty));
    result.invisible_fraction = clamp_fraction(bilinear(c00->invisible_fraction, c10->invisible_fraction, c01->invisible_fraction, c11->invisible_fraction, tx, ty));
    result.untracked_fraction = clamp_fraction(bilinear(c00->untracked_fraction, c10->untracked_fraction, c01->untracked_fraction, c11->untracked_fraction, tx, ty));
    result.energy_closure_error = std::max({c00->energy_closure_error, c10->energy_closure_error, c01->energy_closure_error, c11->energy_closure_error});
    return result;
}

}  // namespace hadros::cascade
