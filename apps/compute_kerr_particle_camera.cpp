#include "kerr_camera.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct ParticleSchemaAudit {
    std::size_t rows = 0;
    std::size_t valid_backend_rows = 0;
    std::size_t rows_with_position = 0;
    std::size_t rows_with_global_position = 0;
    std::size_t rows_with_local_to_global_approximation = 0;
};

struct ParticleRow {
    std::uint64_t event_id = 0;
    std::uint64_t source_particle_id = 0;
    int pdg = 0;
    double energy_gev = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    bool has_momentum = false;
    double weight = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string interaction_type;
    std::string target_type;
    std::string generator_backend;
    std::string transport_backend;
    std::string global_position_status;
    std::string global_position_transform;
};

struct RaySample {
    int pixel_x = 0;
    int pixel_y = 0;
    std::size_t ray_id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double dir_to_observer_x = std::numeric_limits<double>::quiet_NaN();
    double dir_to_observer_y = std::numeric_limits<double>::quiet_NaN();
    double dir_to_observer_z = std::numeric_limits<double>::quiet_NaN();
    // Gravitational redshift factor at this point from backward ray-tracing.
    // E_obs = E_local / redshift_factor  (factor > 1 near BH → particle loses
    // energy escaping to the distant observer).
    double redshift_factor = 1.0;
};

struct HistogramAccumulator {
    double total_energy_gev = 0.0;
    double total_weighted_energy_gev = 0.0;
    std::size_t n_particles = 0;
    std::size_t n_observed = 0;
    std::map<std::size_t, bool> pixels;
};

struct CellKey {
    long long ix = 0;
    long long iy = 0;
    long long iz = 0;

    bool operator==(const CellKey& other) const
    {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& key) const
    {
        std::size_t h = static_cast<std::size_t>(key.ix * 73856093LL);
        h ^= static_cast<std::size_t>(key.iy * 19349663LL);
        h ^= static_cast<std::size_t>(key.iz * 83492791LL);
        return h;
    }
};

static CellKey cell_for(double x, double y, double z, double cell_size)
{
    return CellKey{
        static_cast<long long>(std::floor(x / cell_size)),
        static_cast<long long>(std::floor(y / cell_size)),
        static_cast<long long>(std::floor(z / cell_size))
    };
}

static bool contains_any_position_key(const std::string& line)
{
    static const char* keys[] = {
        "\"particle_position_x_rg\"",
        "\"position_x_rg\"",
        "\"origin_x_rg\"",
        "\"exit_x_rg\"",
        "\"x_rg\""
    };
    for (const char* key : keys) {
        if (line.find(key) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool contains_global_position(const std::string& line)
{
    return line.find("\"global_exit_x_rg\"") != std::string::npos &&
           line.find("\"global_exit_y_rg\"") != std::string::npos &&
           line.find("\"global_exit_z_rg\"") != std::string::npos &&
           (line.find("\"global_position_status\":\"GLOBAL_POSITION_VALID") != std::string::npos ||
            line.find("\"global_position_status\": \"GLOBAL_POSITION_VALID") != std::string::npos);
}

static ParticleSchemaAudit audit_particles(const fs::path& input)
{
    ParticleSchemaAudit audit;
    std::ifstream in(input);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        audit.rows += 1;
        if (line.find("\"origin_backend\": \"POWHEG_NUDIS_PYTHIA8_GEANT4_REAL_SAFE\"") != std::string::npos ||
            line.find("\"origin_backend\":\"POWHEG_NUDIS_PYTHIA8_GEANT4_REAL_SAFE\"") != std::string::npos) {
            audit.valid_backend_rows += 1;
        }
        if (contains_any_position_key(line)) {
            audit.rows_with_position += 1;
        }
        if (contains_global_position(line)) {
            audit.rows_with_global_position += 1;
        }
        if (line.find("\"global_position_transform\":\"LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION\"") != std::string::npos ||
            line.find("\"global_position_transform\": \"LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION\"") != std::string::npos) {
            audit.rows_with_local_to_global_approximation += 1;
        }
    }
    return audit;
}

static double number_value_or(const std::string& line, const std::string& key, double fallback)
{
    const std::string needle = "\"" + key + "\":";
    auto begin = line.find(needle);
    std::size_t value_begin = 0;
    if (begin == std::string::npos) {
        const std::string spaced = "\"" + key + "\": ";
        begin = line.find(spaced);
        if (begin == std::string::npos) {
            return fallback;
        }
        value_begin = begin + spaced.size();
    } else {
        value_begin = begin + needle.size();
    }
    const auto value_end = line.find_first_of(",}", value_begin);
    try {
        return std::stod(line.substr(value_begin, value_end - value_begin));
    } catch (...) {
        return fallback;
    }
}

static bool has_number_key(const std::string& line, const std::string& key)
{
    return line.find("\"" + key + "\":") != std::string::npos ||
           line.find("\"" + key + "\": ") != std::string::npos;
}

static std::string string_value_or(const std::string& line, const std::string& key, const std::string& fallback)
{
    const std::string needle = "\"" + key + "\":\"";
    auto begin = line.find(needle);
    std::size_t value_begin = 0;
    if (begin == std::string::npos) {
        const std::string spaced = "\"" + key + "\": \"";
        begin = line.find(spaced);
        if (begin == std::string::npos) {
            return fallback;
        }
        value_begin = begin + spaced.size();
    } else {
        value_begin = begin + needle.size();
    }
    const auto value_end = line.find('"', value_begin);
    if (value_end == std::string::npos) {
        return fallback;
    }
    return line.substr(value_begin, value_end - value_begin);
}

static std::vector<ParticleRow> read_global_particles(const fs::path& input)
{
    std::vector<ParticleRow> particles;
    std::ifstream in(input);
    std::string line;
    while (std::getline(in, line)) {
        if (!contains_global_position(line)) {
            continue;
        }
        ParticleRow p;
        p.event_id = static_cast<std::uint64_t>(number_value_or(line, "event_id", 0.0));
        p.source_particle_id = static_cast<std::uint64_t>(number_value_or(line, "source_particle_id", 0.0));
        p.pdg = static_cast<int>(number_value_or(line, "pdg", 0.0));
        p.energy_gev = number_value_or(line, "energy_gev", 0.0);
        p.px = number_value_or(line, "px", 0.0);
        p.py = number_value_or(line, "py", 0.0);
        p.pz = number_value_or(line, "pz", 0.0);
        p.has_momentum = has_number_key(line, "px") && has_number_key(line, "py") && has_number_key(line, "pz");
        p.weight = number_value_or(line, "weight", 1.0);
        p.x = number_value_or(line, "global_exit_x_rg", std::numeric_limits<double>::quiet_NaN());
        p.y = number_value_or(line, "global_exit_y_rg", std::numeric_limits<double>::quiet_NaN());
        p.z = number_value_or(line, "global_exit_z_rg", std::numeric_limits<double>::quiet_NaN());
        p.interaction_type = string_value_or(line, "interaction_type", "");
        p.target_type = string_value_or(line, "target_type", "");
        p.generator_backend = string_value_or(line, "generator_backend", "");
        p.transport_backend = string_value_or(line, "transport_backend", "");
        p.global_position_status = string_value_or(line, "global_position_status", "");
        p.global_position_transform = string_value_or(line, "global_position_transform", "");
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            particles.push_back(p);
        }
    }
    return particles;
}

static const char* particle_name(int pdg)
{
    const int a = pdg < 0 ? -pdg : pdg;
    if (pdg == 22) return "gamma";
    if (a == 11) return "electron";
    if (a == 13) return "muon";
    if (a == 12 || a == 14 || a == 16) return "neutrino";
    if (a == 211) return "pion";
    if (a == 321) return "kaon";
    if (a == 2212) return "proton";
    if (a == 2112) return "neutron";
    return "other";
}

static const char* channel_name(int pdg)
{
    const int a = pdg < 0 ? -pdg : pdg;
    if (pdg == 22) return "gamma";
    if (a == 11 || a == 13 || a == 15) return "electromagnetic";
    if (a == 12 || a == 14 || a == 16) return "neutrino";
    if (a == 211 || a == 321 || a == 2212 || a == 2112 || a == 130 || a == 310) return "hadronic";
    return "other";
}

static std::string observed_header()
{
    return
        "pixel_x,pixel_y,nx,ny,ray_id,event_id,source_particle_id,pdg,particle_name,channel,"
        "particle_pdg,particle_energy_gev,production_x_rg,production_y_rg,production_z_rg,"
        "energy_gev,weighted_energy_gev,energy_gev_local,redshift_factor,"
        "px,py,pz,weight,particle_position_x_rg,"
        "particle_position_y_rg,particle_position_z_rg,nearest_ray_distance_rg,spatial_distance_rg,"
        "direction_misalignment_deg,association_mode,direction_association_mode,association_status,interaction_type,target_type,"
        "generator_backend,transport_backend,camera_backend\n";
}

static bool finite3(double x, double y, double z)
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

static bool normalized(double& x, double& y, double& z)
{
    const double norm = std::sqrt(x*x + y*y + z*z);
    if (!std::isfinite(norm) || norm <= 0.0) {
        return false;
    }
    x /= norm;
    y /= norm;
    z /= norm;
    return true;
}

static double direction_misalignment_deg(const ParticleRow& particle, const RaySample& sample)
{
    if (!particle.has_momentum) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double px = particle.px;
    double py = particle.py;
    double pz = particle.pz;
    double rx = sample.dir_to_observer_x;
    double ry = sample.dir_to_observer_y;
    double rz = sample.dir_to_observer_z;
    if (!finite3(px, py, pz) || !finite3(rx, ry, rz)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!normalized(px, py, pz) || !normalized(rx, ry, rz)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double dot = std::clamp(px*rx + py*ry + pz*rz, -1.0, 1.0);
    constexpr double pi = 3.141592653589793238462643383279502884;
    return std::acos(dot) * 180.0 / pi;
}

static void write_observed_row(
    std::ofstream& out,
    const RaySample& best,
    int nx,
    int ny,
    const ParticleRow& particle,
    double energy_gev_observed,
    double weighted_energy,
    double g,
    double distance,
    double misalignment,
    const std::string& association_mode,
    const std::string& direction_mode)
{
    out
        << best.pixel_x << ',' << best.pixel_y << ',' << nx << ',' << ny << ',' << best.ray_id << ','
        << particle.event_id << ',' << particle.source_particle_id << ','
        << particle.pdg << ',' << particle_name(particle.pdg) << ','
        << channel_name(particle.pdg) << ','
        << particle.pdg << ',' << particle.energy_gev << ','
        << particle.x << ',' << particle.y << ',' << particle.z << ','
        << energy_gev_observed << ',' << weighted_energy << ','
        << particle.energy_gev << ',' << g << ','
        << particle.px << ',' << particle.py << ',' << particle.pz << ','
        << particle.weight << ','
        << particle.x << ',' << particle.y << ',' << particle.z << ','
        << distance << ',' << distance << ',' << misalignment << ','
        << association_mode << ','
        << direction_mode << ','
        << "GLOBAL_POSITION_KERR_RAY_SPATIAL_AND_ANGULAR_ASSOCIATION" << ','
        << particle.interaction_type << ',' << particle.target_type << ','
        << particle.generator_backend << ',' << particle.transport_backend << ','
        << "PARTICLE_RAY_ASSOCIATION_CAMERA" << '\n';
}

static void write_observed_json(
    std::ofstream& out,
    const RaySample& best,
    int nx,
    int ny,
    const ParticleRow& particle,
    double energy_gev_observed,
    double weighted_energy,
    double g,
    double distance,
    double misalignment,
    const std::string& association_mode,
    const std::string& direction_mode)
{
    out
        << "{\"pixel_x\":" << best.pixel_x
        << ",\"pixel_y\":" << best.pixel_y
        << ",\"nx\":" << nx
        << ",\"ny\":" << ny
        << ",\"ray_id\":" << best.ray_id
        << ",\"event_id\":" << particle.event_id
        << ",\"source_particle_id\":" << particle.source_particle_id
        << ",\"pdg\":" << particle.pdg
        << ",\"particle_pdg\":" << particle.pdg
        << ",\"particle_name\":\"" << particle_name(particle.pdg)
        << "\",\"channel\":\"" << channel_name(particle.pdg)
        << "\",\"particle_energy_gev\":" << particle.energy_gev
        << ",\"production_x_rg\":" << particle.x
        << ",\"production_y_rg\":" << particle.y
        << ",\"production_z_rg\":" << particle.z
        << ",\"energy_gev\":" << energy_gev_observed
        << ",\"weighted_energy_gev\":" << weighted_energy
        << ",\"energy_gev_local\":" << particle.energy_gev
        << ",\"redshift_factor\":" << g
        << ",\"particle_position_x_rg\":" << particle.x
        << ",\"particle_position_y_rg\":" << particle.y
        << ",\"particle_position_z_rg\":" << particle.z
        << ",\"nearest_ray_distance_rg\":" << distance
        << ",\"spatial_distance_rg\":" << distance
        << ",\"direction_misalignment_deg\":" << misalignment
        << ",\"association_mode\":\"" << association_mode
        << "\",\"direction_association_mode\":\"" << direction_mode
        << "\",\"association_status\":\"GLOBAL_POSITION_KERR_RAY_SPATIAL_AND_ANGULAR_ASSOCIATION\""
        << ",\"camera_backend\":\"PARTICLE_RAY_ASSOCIATION_CAMERA\"}\n";
}

int main(int argc, char** argv)
{
    if (argc < 15) {
        std::cerr
            << "Usage: compute_kerr_particle_camera INPUT_JSONL OUTPUT_DIR "
            << "ASPIN CAM_R_OBS_RG CAM_THETA_DEG CAM_FOV_DEG CAM_NX CAM_NY CAM_R_MAX_RG CAM_STEP "
            << "SPATIAL_TOLERANCE_RG ANGULAR_TOLERANCE_DEG ASSOCIATION_MODE CAMERA_NAMING_MODE\n";
        return 64;
    }

    const fs::path input = argv[1];
    const fs::path out_dir = argv[2];
    const double a_spin = std::stod(argv[3]);
    const double r_obs = std::stod(argv[4]);
    const double theta_deg = std::stod(argv[5]);
    const double fov_deg = std::stod(argv[6]);
    const int nx = std::stoi(argv[7]);
    const int ny = std::stoi(argv[8]);
    const double r_max = std::stod(argv[9]);
    const double step = std::stod(argv[10]);
    const double spatial_tolerance_rg = std::stod(argv[11]);
    const double angular_tolerance_deg = std::stod(argv[12]);
    const std::string association_mode = argv[13];
    const std::string camera_naming_mode = argv[14];
    if (association_mode == "full_transport") {
        std::cerr << "full_transport is not implemented yet\n";
        return 64;
    }
    if (association_mode != "spatial_only" && association_mode != "spatial_plus_direction") {
        std::cerr << "Invalid ASSOCIATION_MODE '" << association_mode
                  << "'. Use spatial_only, spatial_plus_direction, or full_transport.\n";
        return 64;
    }
    if (camera_naming_mode != "both" && camera_naming_mode != "semantic" && camera_naming_mode != "legacy") {
        std::cerr << "Invalid CAMERA_NAMING_MODE '" << camera_naming_mode
                  << "'. Use both, semantic, or legacy.\n";
        return 64;
    }
    const bool write_semantic_outputs = camera_naming_mode == "both" || camera_naming_mode == "semantic";
    const bool write_legacy_outputs = camera_naming_mode == "both" || camera_naming_mode == "legacy";

    fs::create_directories(out_dir);
    fs::create_directories(out_dir / "plots");

    const ParticleSchemaAudit particle_audit = audit_particles(input);

    constexpr double pi = 3.141592653589793238462643383279502884;
    KerrCamera camera(a_spin, r_obs, theta_deg * pi / 180.0, fov_deg, nx, ny, r_max, step);

    std::size_t trace_pixel_calls = 0;
    std::size_t ray_sample_count = 0;
    std::uint64_t ray_hash = 1469598103934665603ull;
    std::vector<RaySample> ray_samples;
    auto mix = [&](double value) {
        const long long scaled = static_cast<long long>(std::llround(value * 1000000.0));
        ray_hash ^= static_cast<std::uint64_t>(scaled);
        ray_hash *= 1099511628211ull;
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const RayPath ray = camera.trace_pixel(i, j);
            const std::size_t ray_id = static_cast<std::size_t>(j * nx + i);
            trace_pixel_calls += 1;
            ray_sample_count += ray.points.size();
            for (std::size_t idx = 0; idx < ray.points.size(); ++idx) {
                const auto& point = ray.points[idx];
                double dx = std::numeric_limits<double>::quiet_NaN();
                double dy = std::numeric_limits<double>::quiet_NaN();
                double dz = std::numeric_limits<double>::quiet_NaN();
                if (idx > 0) {
                    const auto& previous = ray.points[idx - 1];
                    dx = previous.x_rg - point.x_rg;
                    dy = previous.y_rg - point.y_rg;
                    dz = previous.z_rg - point.z_rg;
                    (void)normalized(dx, dy, dz);
                } else if (idx + 1 < ray.points.size()) {
                    const auto& next = ray.points[idx + 1];
                    dx = point.x_rg - next.x_rg;
                    dy = point.y_rg - next.y_rg;
                    dz = point.z_rg - next.z_rg;
                    (void)normalized(dx, dy, dz);
                }
                ray_samples.push_back(RaySample{i, j, ray_id, point.x_rg, point.y_rg, point.z_rg,
                    dx, dy, dz, std::max(point.redshift_factor, 1.0e-10)});
            }
            if (!ray.points.empty()) {
                const PathPoint& p = ray.points.back();
                mix(p.r_rg);
                mix(p.theta);
                mix(p.x_rg);
                mix(p.y_rg);
                mix(p.z_rg);
            }
        }
    }

    const std::vector<ParticleRow> global_particles = read_global_particles(input);
    std::unordered_map<CellKey, std::vector<const RaySample*>, CellKeyHash> ray_grid;
    ray_grid.reserve(ray_samples.size());
    for (const auto& sample : ray_samples) {
        ray_grid[cell_for(sample.x, sample.y, sample.z, spatial_tolerance_rg)].push_back(&sample);
    }
    std::size_t observed_rows = 0;
    std::size_t spatial_candidate_rows = 0;
    std::size_t rejected_missing_direction = 0;
    std::size_t rejected_angular_tolerance = 0;
    std::map<int, HistogramAccumulator> pdg_histograms;
    std::map<std::string, HistogramAccumulator> channel_histograms;

    std::ofstream observed_csv;
    std::ofstream observed_jsonl;
    std::ofstream association_csv;
    std::ofstream association_jsonl;
    if (write_legacy_outputs) {
        observed_csv.open(out_dir / "observed_particles_by_pixel.csv");
        observed_jsonl.open(out_dir / "observed_particles_by_pixel.jsonl");
        observed_csv << observed_header();
    }
    if (write_semantic_outputs) {
        association_csv.open(out_dir / "particle_ray_association_camera.csv");
        association_jsonl.open(out_dir / "particle_ray_association_camera.jsonl");
        association_csv << observed_header();
    }
    for (const auto& particle : global_particles) {
        double best_d2 = std::numeric_limits<double>::infinity();
        double best_misalignment = std::numeric_limits<double>::quiet_NaN();
        const RaySample* best = nullptr;
        const CellKey center = cell_for(particle.x, particle.y, particle.z, spatial_tolerance_rg);
        for (long long dx_cell = -1; dx_cell <= 1; ++dx_cell) {
            for (long long dy_cell = -1; dy_cell <= 1; ++dy_cell) {
                for (long long dz_cell = -1; dz_cell <= 1; ++dz_cell) {
                    const CellKey key{center.ix + dx_cell, center.iy + dy_cell, center.iz + dz_cell};
                    const auto it = ray_grid.find(key);
                    if (it == ray_grid.end()) {
                        continue;
                    }
                    for (const RaySample* sample : it->second) {
                        const double dx = sample->x - particle.x;
                        const double dy = sample->y - particle.y;
                        const double dz = sample->z - particle.z;
                        const double d2 = dx * dx + dy * dy + dz * dz;
                        if (std::sqrt(d2) > spatial_tolerance_rg) {
                            continue;
                        }
                        spatial_candidate_rows += 1;
                        double misalignment = std::numeric_limits<double>::quiet_NaN();
                        if (association_mode == "spatial_plus_direction") {
                            misalignment = direction_misalignment_deg(particle, *sample);
                            if (!std::isfinite(misalignment)) {
                                rejected_missing_direction += 1;
                                continue;
                            }
                            if (misalignment > angular_tolerance_deg) {
                                rejected_angular_tolerance += 1;
                                continue;
                            }
                        }
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_misalignment = misalignment;
                            best = sample;
                        }
                    }
                }
            }
        }
        const double distance = std::sqrt(best_d2);
        if (best == nullptr || !std::isfinite(distance) || distance > spatial_tolerance_rg) {
            continue;
        }
        observed_rows += 1;
        // Apply gravitational redshift: E_obs = E_local / redshift_factor.
        // redshift_factor > 1 near the BH (photon blue-shifts in backward tracing),
        // so the particle loses energy escaping to the distant observer.
        const double g = best->redshift_factor;
        const double energy_gev_observed = particle.energy_gev / g;
        const double weighted_energy = energy_gev_observed * particle.weight;
        const std::string direction_mode = association_mode == "spatial_only"
            ? "spatial_only_direction_not_calculated"
            : "spatial_plus_direction_particle_momentum_vs_kerr_ray_to_observer";
        if (write_legacy_outputs) {
            write_observed_row(observed_csv, *best, nx, ny, particle, energy_gev_observed, weighted_energy, g, distance, best_misalignment, association_mode, direction_mode);
            write_observed_json(observed_jsonl, *best, nx, ny, particle, energy_gev_observed, weighted_energy, g, distance, best_misalignment, association_mode, direction_mode);
        }
        if (write_semantic_outputs) {
            write_observed_row(association_csv, *best, nx, ny, particle, energy_gev_observed, weighted_energy, g, distance, best_misalignment, association_mode, direction_mode);
            write_observed_json(association_jsonl, *best, nx, ny, particle, energy_gev_observed, weighted_energy, g, distance, best_misalignment, association_mode, direction_mode);
        }

        const std::size_t pixel_key = static_cast<std::size_t>(best->pixel_y * nx + best->pixel_x);
        auto& pdg_hist = pdg_histograms[particle.pdg];
        pdg_hist.total_energy_gev += energy_gev_observed;
        pdg_hist.total_weighted_energy_gev += weighted_energy;
        pdg_hist.n_particles += 1;
        pdg_hist.n_observed += 1;
        pdg_hist.pixels[pixel_key] = true;
        auto& channel_hist = channel_histograms[channel_name(particle.pdg)];
        channel_hist.total_energy_gev += energy_gev_observed;
        channel_hist.total_weighted_energy_gev += weighted_energy;
        channel_hist.n_particles += 1;
        channel_hist.n_observed += 1;
        channel_hist.pixels[pixel_key] = true;
    }

    std::ofstream pdg_hist;
    std::ofstream pdg_assoc_hist;
    if (write_legacy_outputs) {
        pdg_hist.open(out_dir / "observed_particle_pdg_histogram.csv");
        pdg_hist << "pdg,particle_name,channel,total_energy_gev,total_weighted_energy_gev,mean_energy_gev,n_particles,n_observed,n_pixels\n";
    }
    if (write_semantic_outputs) {
        pdg_assoc_hist.open(out_dir / "particle_ray_association_pdg_histogram.csv");
        pdg_assoc_hist << "pdg,particle_name,channel,total_energy_gev,total_weighted_energy_gev,mean_energy_gev,n_particles,n_observed,n_pixels\n";
    }
    for (const auto& [pdg, hist] : pdg_histograms) {
        const double mean = hist.n_observed > 0 ? hist.total_energy_gev / static_cast<double>(hist.n_observed) : 0.0;
        if (write_legacy_outputs) {
            pdg_hist << pdg << ',' << particle_name(pdg) << ',' << channel_name(pdg) << ','
                     << hist.total_energy_gev << ',' << hist.total_weighted_energy_gev << ',' << mean << ','
                     << hist.n_particles << ',' << hist.n_observed << ',' << hist.pixels.size() << '\n';
        }
        if (write_semantic_outputs) {
            pdg_assoc_hist << pdg << ',' << particle_name(pdg) << ',' << channel_name(pdg) << ','
                           << hist.total_energy_gev << ',' << hist.total_weighted_energy_gev << ',' << mean << ','
                           << hist.n_particles << ',' << hist.n_observed << ',' << hist.pixels.size() << '\n';
        }
    }
    std::ofstream channel_hist;
    std::ofstream channel_assoc_hist;
    if (write_legacy_outputs) {
        channel_hist.open(out_dir / "observed_particle_channel_histogram.csv");
        channel_hist << "channel,total_energy_gev,total_weighted_energy_gev,mean_energy_gev,n_particles,n_observed,n_pixels\n";
    }
    if (write_semantic_outputs) {
        channel_assoc_hist.open(out_dir / "particle_ray_association_channel_histogram.csv");
        channel_assoc_hist << "channel,total_energy_gev,total_weighted_energy_gev,mean_energy_gev,n_particles,n_observed,n_pixels\n";
    }
    for (const auto& [channel, hist] : channel_histograms) {
        const double mean = hist.n_observed > 0 ? hist.total_energy_gev / static_cast<double>(hist.n_observed) : 0.0;
        if (write_legacy_outputs) {
            channel_hist << channel << ',' << hist.total_energy_gev << ',' << hist.total_weighted_energy_gev << ',' << mean << ','
                         << hist.n_particles << ',' << hist.n_observed << ',' << hist.pixels.size() << '\n';
        }
        if (write_semantic_outputs) {
            channel_assoc_hist << channel << ',' << hist.total_energy_gev << ',' << hist.total_weighted_energy_gev << ',' << mean << ','
                               << hist.n_particles << ',' << hist.n_observed << ',' << hist.pixels.size() << '\n';
        }
    }

    const bool missing_local_positions = particle_audit.rows > 0 && particle_audit.rows_with_position == 0;
    const bool missing_global_positions = particle_audit.rows > 0 && particle_audit.rows_with_global_position == 0;
    const bool local_to_global_approximation =
        particle_audit.rows_with_local_to_global_approximation > 0;
    std::string status;
    std::string blocked_reason;
    if (missing_global_positions && missing_local_positions) {
        status = "PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_GLOBAL_POSITION";
        blocked_reason = "MISSING_PARTICLE_POSITION";
    } else if (missing_global_positions) {
        status = "PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_GLOBAL_POSITION";
        blocked_reason = "MISSING_GLOBAL_KERR_INTERACTION_POSITION";
    } else if (observed_rows == 0) {
        status = "PARTICLE_RAY_ASSOCIATION_CAMERA_BLOCKED_BY_ASSOCIATION_CRITERIA";
        blocked_reason = spatial_candidate_rows == 0 ? "GLOBAL_POSITIONS_OUTSIDE_RAY_TOLERANCE_OR_FOV" : "NO_PARTICLES_PASS_ANGULAR_TOLERANCE";
    } else if (local_to_global_approximation) {
        status = "PARTICLE_RAY_ASSOCIATION_CAMERA_PARTIAL_SAMPLED_INTERACTIONS";
        blocked_reason = "LOCAL_CARTESIAN_BOX_TO_GLOBAL_APPROXIMATION";
    } else {
        status = "PARTICLE_RAY_ASSOCIATION_CAMERA_VALIDATED";
        blocked_reason = "";
    }

    std::ofstream validation;
    std::ofstream association_validation;
    const std::string validation_header =
        "status,input_particles,valid_backend_rows,rows_with_position,rows_with_global_position,"
        "rows_with_local_to_global_approximation,trace_pixel_calls,"
        "ray_sample_count,ray_hash,observed_rows,spatial_candidate_rows,rejected_missing_direction,"
        "rejected_angular_tolerance,association_mode,camera_naming_mode,spatial_tolerance_rg,"
        "angular_tolerance_deg,direction_misalignment_nan_semantics,camera_physical_interpretation,"
        "camera_is_full_observational_transport,camera_limitation,full_transport_available,camera_backend,blocked_reason\n";
    if (write_legacy_outputs) {
        validation.open(out_dir / "real_kerr_particle_camera_validation.csv");
        validation << validation_header;
    }
    if (write_semantic_outputs) {
        association_validation.open(out_dir / "particle_ray_association_camera_validation.csv");
        association_validation << validation_header;
    }
    const auto write_validation_row = [&](std::ofstream& out) {
        out
        << status << ','
        << particle_audit.rows << ','
        << particle_audit.valid_backend_rows << ','
        << particle_audit.rows_with_position << ','
        << particle_audit.rows_with_global_position << ','
        << particle_audit.rows_with_local_to_global_approximation << ','
        << trace_pixel_calls << ','
        << ray_sample_count << ','
        << ray_hash << ','
        << observed_rows << ','
        << spatial_candidate_rows << ','
        << rejected_missing_direction << ','
        << rejected_angular_tolerance << ','
        << association_mode << ','
        << camera_naming_mode << ','
        << spatial_tolerance_rg << ','
        << angular_tolerance_deg << ','
        << "NaN means direction misalignment was not calculable and is not accepted as aligned" << ','
        << "particle-ray association / cascade origin map" << ','
        << "false" << ','
        << "secondary particles are associated with Kerr rays by spatial/angular criteria; they are not propagated to the distant observer" << ','
        << "false" << ','
        << "PARTICLE_RAY_ASSOCIATION_CAMERA" << ','
        << blocked_reason << '\n';
    };
    if (write_legacy_outputs) {
        write_validation_row(validation);
    }
    if (write_semantic_outputs) {
        write_validation_row(association_validation);
    }

    std::ofstream summary;
    std::ofstream association_summary;
    if (write_legacy_outputs) {
        summary.open(out_dir / "observed_particles_by_pixel_summary.md");
    }
    if (write_semantic_outputs) {
        association_summary.open(out_dir / "particle_ray_association_camera_summary.md");
    }
    const std::string legacy_warning = write_legacy_outputs
        ? " Legacy `observed_particles_by_pixel.*` files are compatibility outputs only and do not imply full observation."
        : "";
    const std::string association_rule = association_mode == "spatial_only"
        ? "A row is accepted when the particle global position is within the spatial tolerance of a sampled Kerr ray. Direction is not calculated in this mode, so `direction_misalignment_deg = NaN` is diagnostic, not an alignment claim."
        : "A row is accepted only when the particle global position is within the spatial tolerance of a sampled Kerr ray and the particle momentum is within the angular tolerance of the local ray direction toward the observer.";
    auto write_summary = [&](std::ofstream& summary_out, const std::string& title) {
    summary_out
        << "# " << title << "\n\n"
        << "Status: `" << status << "`.\n\n"
        << "- particle_camera_backend: `particle_ray_association_camera`\n"
        << "- legacy_compatible_outputs: `" << (write_legacy_outputs ? "observed_particles_by_pixel.csv, observed_particles_by_pixel.jsonl" : "not written in this camera_naming_mode") << "`\n"
        << "- camera_backend: `PARTICLE_RAY_ASSOCIATION_CAMERA`\n"
        << "- input_particles: `" << particle_audit.rows << "`\n"
        << "- valid_backend_rows: `" << particle_audit.valid_backend_rows << "`\n"
        << "- rows_with_position: `" << particle_audit.rows_with_position << "`\n"
        << "- rows_with_global_position: `" << particle_audit.rows_with_global_position << "`\n"
        << "- rows_with_local_to_global_approximation: `" << particle_audit.rows_with_local_to_global_approximation << "`\n"
        << "- observed_rows: `" << observed_rows << "`\n"
        << "- trace_pixel_calls: `" << trace_pixel_calls << "`\n"
        << "- ray_sample_count: `" << ray_sample_count << "`\n"
        << "- association_mode: `" << association_mode << "`\n"
        << "- camera_naming_mode: `" << camera_naming_mode << "`\n"
        << "- camera_physical_interpretation: `particle-ray association / cascade origin map`\n"
        << "- camera_is_full_observational_transport: `false`\n"
        << "- full_transport_available: `false`\n"
        << "- spatial_tolerance_rg: `" << spatial_tolerance_rg << "`\n"
        << "- angular_tolerance_deg: `" << angular_tolerance_deg << "`\n"
        << "- rejected_missing_direction: `" << rejected_missing_direction << "`\n"
        << "- rejected_angular_tolerance: `" << rejected_angular_tolerance << "`\n"
        << "- direction_misalignment_deg_nan_semantics: `NaN means not calculated; it is not treated as perfect alignment.`\n"
        << "- blocked_reason: `" << blocked_reason << "`\n\n"
        << "The backend traces pixels through `KerrCamera::trace_pixel`, which advances rays with "
        << "`KerrGeodesic::step_adaptive`. This product is a particle-to-Kerr-ray association camera, "
        << "not a physical observer image. " << association_rule << " No full particle transport "
        << "to the observer is implemented here." << legacy_warning << "\n";
    };
    if (write_legacy_outputs) {
        write_summary(summary, "Particle Ray Association Camera");
    }
    if (write_semantic_outputs) {
        write_summary(association_summary, "Particle Ray Association Camera");
    }

    std::ofstream call_graph(out_dir / "real_kerr_camera_call_graph.md");
    call_graph
        << "# Particle Ray Association Camera Call Graph\n\n"
        << "Status: `" << status << "`.\n\n"
        << "```text\n"
        << "scripts/science/run_real_kerr_particle_camera.py\n"
        << "-> build/compute_kerr_particle_camera\n"
        << "-> KerrCamera camera(...)\n"
        << "-> for pixel in camera grid\n"
        << "-> KerrCamera::trace_pixel(i, j)\n"
        << "-> KerrGeodesic::step_adaptive(y)\n"
        << "-> ray samples\n"
        << "-> spatial + angular particle/ray association: " << (observed_rows > 0 ? "associated" : blocked_reason) << "\n"
        << "```\n\n"
        << "- trace_pixel_called: `true`\n"
        << "- step_adaptive_used: `true`\n"
        << "- trace_pixel_calls: `" << trace_pixel_calls << "`\n"
        << "- ray_sample_count: `" << ray_sample_count << "`\n";

    std::cout
        << "{\n"
        << "  \"status\": \"" << status << "\",\n"
        << "  \"trace_pixel_calls\": " << trace_pixel_calls << ",\n"
        << "  \"ray_sample_count\": " << ray_sample_count << ",\n"
        << "  \"rows_with_position\": " << particle_audit.rows_with_position << ",\n"
        << "  \"rows_with_global_position\": " << particle_audit.rows_with_global_position << ",\n"
        << "  \"observed_rows\": " << observed_rows << "\n"
        << "}\n";

    return (status == "PARTICLE_RAY_ASSOCIATION_CAMERA_VALIDATED" ||
            status == "PARTICLE_RAY_ASSOCIATION_CAMERA_PARTIAL_SAMPLED_INTERACTIONS")
        ? 0 : 2;
}
