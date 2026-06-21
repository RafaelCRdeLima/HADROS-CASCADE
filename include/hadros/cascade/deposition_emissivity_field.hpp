#pragma once

#include <string>
#include <vector>

namespace hadros::cascade {

struct DepositionEmissivityQuery {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string mode = "trilinear";
};

struct DepositionEmissivityResult {
    bool valid = false;
    std::string status = "EMPTY_FIELD";
    std::string interpolation_mode = "none";
    double j_dep = 0.0;
    double coverage = 0.0;
    double deposited_energy = 0.0;
    double escaped_energy = 0.0;
    double invisible_energy = 0.0;
    double untracked_energy = 0.0;
};

class DepositionEmissivityField {
public:
    bool load_hdf5(const std::string& hdf5_path, const std::string& manifest_path);

    DepositionEmissivityResult query(const DepositionEmissivityQuery& query) const;
    DepositionEmissivityResult query_nearest(double x, double y, double z) const;
    DepositionEmissivityResult query_trilinear(double x, double y, double z) const;

    bool empty() const { return j_dep_.empty(); }
    bool finite() const;
    bool shape_valid() const;

    double total_deposited_energy() const;
    double total_escaped_energy() const;
    double total_invisible_energy() const;
    double total_untracked_energy() const;

    const std::vector<double>& grid_x() const { return grid_x_; }
    const std::vector<double>& grid_y() const { return grid_y_; }
    const std::vector<double>& grid_z() const { return grid_z_; }
    const std::string& manifest_raw() const { return manifest_raw_; }
    const std::vector<std::string>& warning_labels() const { return warning_labels_; }

private:
    std::vector<double> grid_x_;
    std::vector<double> grid_y_;
    std::vector<double> grid_z_;
    std::vector<double> j_dep_;
    std::vector<double> deposited_;
    std::vector<double> escaped_;
    std::vector<double> invisible_;
    std::vector<double> untracked_;
    std::vector<double> coverage_;
    std::vector<double> event_count_;
    std::string manifest_raw_;
    std::vector<std::string> warning_labels_;
    std::size_t nx_ = 0;
    std::size_t ny_ = 0;
    std::size_t nz_ = 0;

    std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) const;
    bool in_range(double x, double y, double z) const;
};

}  // namespace hadros::cascade
