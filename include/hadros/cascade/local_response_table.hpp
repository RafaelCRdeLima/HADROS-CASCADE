#pragma once

#include <string>
#include <vector>

namespace hadros::cascade {

struct LocalResponseQuery {
    int pdg_id = 0;
    double kinetic_energy_gev = 0.0;
    double density_g_cm3 = 0.0;
    std::string material;
    double box_size_cm = 0.0;
    std::string physics_list;
};

struct LocalResponseResult {
    bool valid = false;
    std::string status = "EMPTY_TABLE";
    double deposited_fraction = 0.0;
    double escaped_fraction = 0.0;
    double invisible_fraction = 0.0;
    double untracked_fraction = 0.0;
    double energy_closure_error = 0.0;
    std::string interpolation_mode = "none";
};

class LocalResponseTable {
public:
    struct Row {
        int pdg_id = 0;
        double energy_gev = 0.0;
        double density_g_cm3 = 0.0;
        std::string material;
        double box_size_cm = 0.0;
        std::string physics_list;
        double deposited_fraction = 0.0;
        double escaped_fraction = 0.0;
        double invisible_fraction = 0.0;
        double untracked_fraction = 0.0;
        double energy_closure_error = 0.0;
        std::string status;
    };

    bool load_csv(const std::string& path);
    LocalResponseResult query_nearest(const LocalResponseQuery& query) const;
    LocalResponseResult query_interpolated(const LocalResponseQuery& query) const;
    std::size_t size() const { return rows_.size(); }

private:
    std::vector<Row> rows_;
};

}  // namespace hadros::cascade
