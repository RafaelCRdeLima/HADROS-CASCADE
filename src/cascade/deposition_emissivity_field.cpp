#include "hadros/cascade/deposition_emissivity_field.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

#ifdef HADROS_WITH_HDF5
#include <H5Cpp.h>
#endif

namespace hadros::cascade {
namespace {

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open manifest: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

std::vector<std::string> extract_warning_labels(const std::string& json) {
    std::vector<std::string> labels;
    const std::string key = "\"warning_labels\"";
    const auto key_pos = json.find(key);
    if (key_pos == std::string::npos) {
        return labels;
    }
    const auto open = json.find('[', key_pos);
    const auto close = json.find(']', open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        return labels;
    }
    std::size_t pos = open + 1;
    while (pos < close) {
        const auto q0 = json.find('"', pos);
        if (q0 == std::string::npos || q0 >= close) {
            break;
        }
        const auto q1 = json.find('"', q0 + 1);
        if (q1 == std::string::npos || q1 > close) {
            break;
        }
        labels.push_back(json.substr(q0 + 1, q1 - q0 - 1));
        pos = q1 + 1;
    }
    return labels;
}

bool finite_vector(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

double sum_vector(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

DepositionEmissivityResult status_result(const std::string& status) {
    DepositionEmissivityResult result;
    result.valid = false;
    result.status = status;
    return result;
}

std::size_t nearest_index(const std::vector<double>& grid, double value) {
    auto best = std::min_element(grid.begin(), grid.end(), [&](double a, double b) {
        return std::abs(a - value) < std::abs(b - value);
    });
    return static_cast<std::size_t>(std::distance(grid.begin(), best));
}

std::pair<std::size_t, std::size_t> bracket(const std::vector<double>& grid, double value) {
    auto upper = std::lower_bound(grid.begin(), grid.end(), value);
    if (upper == grid.end()) {
        return {grid.size() - 1, grid.size() - 1};
    }
    if (upper == grid.begin() || *upper == value) {
        const auto idx = static_cast<std::size_t>(std::distance(grid.begin(), upper));
        return {idx, idx};
    }
    const auto hi = static_cast<std::size_t>(std::distance(grid.begin(), upper));
    return {hi - 1, hi};
}

double lerp(double a, double b, double t) {
    return (1.0 - t) * a + t * b;
}

#ifdef HADROS_WITH_HDF5
std::vector<double> read_1d(const H5::H5File& file, const std::string& name) {
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace space = dataset.getSpace();
    if (space.getSimpleExtentNdims() != 1) {
        throw std::runtime_error("Expected 1D dataset: " + name);
    }
    hsize_t dims[1] = {0};
    space.getSimpleExtentDims(dims);
    std::vector<double> values(static_cast<std::size_t>(dims[0]));
    dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}

std::vector<double> read_3d(const H5::H5File& file, const std::string& name,
                            std::size_t& nx, std::size_t& ny, std::size_t& nz) {
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace space = dataset.getSpace();
    if (space.getSimpleExtentNdims() != 3) {
        throw std::runtime_error("Expected 3D dataset: " + name);
    }
    hsize_t dims[3] = {0, 0, 0};
    space.getSimpleExtentDims(dims);
    nx = static_cast<std::size_t>(dims[0]);
    ny = static_cast<std::size_t>(dims[1]);
    nz = static_cast<std::size_t>(dims[2]);
    std::vector<double> values(nx * ny * nz);
    dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}
#endif

}  // namespace

bool DepositionEmissivityField::load_hdf5(const std::string& hdf5_path, const std::string& manifest_path) {
#ifndef HADROS_WITH_HDF5
    (void)hdf5_path;
    (void)manifest_path;
    throw std::runtime_error("HDF5 support is not enabled. Rebuild with HADROS_WITH_HDF5=ON.");
#else
    H5::H5File file(hdf5_path, H5F_ACC_RDONLY);
    grid_x_ = read_1d(file, "grid_x");
    grid_y_ = read_1d(file, "grid_y");
    grid_z_ = read_1d(file, "grid_z");

    std::size_t nx = 0, ny = 0, nz = 0;
    j_dep_ = read_3d(file, "j_dep", nx, ny, nz);
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;

    std::size_t tx = 0, ty = 0, tz = 0;
    deposited_ = read_3d(file, "deposited_energy_grid", tx, ty, tz);
    escaped_ = read_3d(file, "escaped_energy_grid", tx, ty, tz);
    invisible_ = read_3d(file, "invisible_energy_grid", tx, ty, tz);
    untracked_ = read_3d(file, "untracked_energy_grid", tx, ty, tz);
    coverage_ = read_3d(file, "coverage_grid", tx, ty, tz);
    event_count_ = read_3d(file, "event_count", tx, ty, tz);

    manifest_raw_ = read_text(manifest_path);
    warning_labels_ = extract_warning_labels(manifest_raw_);

    if (!shape_valid()) {
        throw std::runtime_error("Deposition emissivity field shape validation failed.");
    }
    if (!finite()) {
        throw std::runtime_error("Deposition emissivity field contains NaN or inf.");
    }
    return true;
#endif
}

bool DepositionEmissivityField::shape_valid() const {
    const std::size_t n = nx_ * ny_ * nz_;
    return nx_ == grid_x_.size() && ny_ == grid_y_.size() && nz_ == grid_z_.size() &&
           !grid_x_.empty() && !grid_y_.empty() && !grid_z_.empty() &&
           j_dep_.size() == n && deposited_.size() == n && escaped_.size() == n &&
           invisible_.size() == n && untracked_.size() == n && coverage_.size() == n &&
           event_count_.size() == n;
}

bool DepositionEmissivityField::finite() const {
    return finite_vector(grid_x_) && finite_vector(grid_y_) && finite_vector(grid_z_) &&
           finite_vector(j_dep_) && finite_vector(deposited_) && finite_vector(escaped_) &&
           finite_vector(invisible_) && finite_vector(untracked_) && finite_vector(coverage_) &&
           finite_vector(event_count_);
}

double DepositionEmissivityField::total_deposited_energy() const { return sum_vector(deposited_); }
double DepositionEmissivityField::total_escaped_energy() const { return sum_vector(escaped_); }
double DepositionEmissivityField::total_invisible_energy() const { return sum_vector(invisible_); }
double DepositionEmissivityField::total_untracked_energy() const { return sum_vector(untracked_); }

std::size_t DepositionEmissivityField::index(std::size_t ix, std::size_t iy, std::size_t iz) const {
    return (ix * ny_ + iy) * nz_ + iz;
}

bool DepositionEmissivityField::in_range(double x, double y, double z) const {
    return !empty() &&
           x >= grid_x_.front() && x <= grid_x_.back() &&
           y >= grid_y_.front() && y <= grid_y_.back() &&
           z >= grid_z_.front() && z <= grid_z_.back();
}

DepositionEmissivityResult DepositionEmissivityField::query(const DepositionEmissivityQuery& query_value) const {
    if (query_value.mode == "nearest") {
        return query_nearest(query_value.x, query_value.y, query_value.z);
    }
    return query_trilinear(query_value.x, query_value.y, query_value.z);
}

DepositionEmissivityResult DepositionEmissivityField::query_nearest(double x, double y, double z) const {
    if (!shape_valid()) {
        return status_result("EMPTY_FIELD");
    }
    if (!in_range(x, y, z)) {
        return status_result("OUT_OF_RANGE");
    }
    const auto ix = nearest_index(grid_x_, x);
    const auto iy = nearest_index(grid_y_, y);
    const auto iz = nearest_index(grid_z_, z);
    const auto id = index(ix, iy, iz);
    DepositionEmissivityResult result;
    result.valid = true;
    result.status = "OK_NEAREST";
    result.interpolation_mode = "nearest";
    result.j_dep = j_dep_[id];
    result.coverage = coverage_[id];
    result.deposited_energy = deposited_[id];
    result.escaped_energy = escaped_[id];
    result.invisible_energy = invisible_[id];
    result.untracked_energy = untracked_[id];
    return result;
}

DepositionEmissivityResult DepositionEmissivityField::query_trilinear(double x, double y, double z) const {
    if (!shape_valid()) {
        return status_result("EMPTY_FIELD");
    }
    if (!in_range(x, y, z)) {
        return status_result("OUT_OF_RANGE");
    }
    const auto [x0, x1] = bracket(grid_x_, x);
    const auto [y0, y1] = bracket(grid_y_, y);
    const auto [z0, z1] = bracket(grid_z_, z);
    const double tx = x0 == x1 ? 0.0 : (x - grid_x_[x0]) / (grid_x_[x1] - grid_x_[x0]);
    const double ty = y0 == y1 ? 0.0 : (y - grid_y_[y0]) / (grid_y_[y1] - grid_y_[y0]);
    const double tz = z0 == z1 ? 0.0 : (z - grid_z_[z0]) / (grid_z_[z1] - grid_z_[z0]);

    const auto tri = [&](const std::vector<double>& values) {
        const double c000 = values[index(x0, y0, z0)];
        const double c100 = values[index(x1, y0, z0)];
        const double c010 = values[index(x0, y1, z0)];
        const double c110 = values[index(x1, y1, z0)];
        const double c001 = values[index(x0, y0, z1)];
        const double c101 = values[index(x1, y0, z1)];
        const double c011 = values[index(x0, y1, z1)];
        const double c111 = values[index(x1, y1, z1)];
        const double c00 = lerp(c000, c100, tx);
        const double c10 = lerp(c010, c110, tx);
        const double c01 = lerp(c001, c101, tx);
        const double c11 = lerp(c011, c111, tx);
        return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
    };

    DepositionEmissivityResult result;
    result.valid = true;
    result.status = "OK_INTERPOLATED";
    result.interpolation_mode = "trilinear";
    result.j_dep = tri(j_dep_);
    result.coverage = tri(coverage_);
    result.deposited_energy = tri(deposited_);
    result.escaped_energy = tri(escaped_);
    result.invisible_energy = tri(invisible_);
    result.untracked_energy = tri(untracked_);
    return result;
}

}  // namespace hadros::cascade
