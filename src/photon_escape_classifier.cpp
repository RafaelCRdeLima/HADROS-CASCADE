#include "photon_escape_classifier.hpp"

#include "hadros/cascade/kerr_local_tetrad.hpp"
#include "kerr_geodesic.hpp"
#include "kerr_metric.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double REL_EPS = 1.0e-300;

struct PhotonRecord {
    long long event_id = 0;
    long long particle_id = 0;
    int pdg = 0;
    double energy_gev = std::numeric_limits<double>::quiet_NaN();
    double r = std::numeric_limits<double>::quiet_NaN();
    double theta = std::numeric_limits<double>::quiet_NaN();
    double phi = std::numeric_limits<double>::quiet_NaN();
    double nr = std::numeric_limits<double>::quiet_NaN();
    double ntheta = std::numeric_limits<double>::quiet_NaN();
    double nphi = std::numeric_limits<double>::quiet_NaN();
    bool has_direct_pcov = false;
    bool has_ambiguous_generic_momentum_input = false;
    double pcov[4] = {0.0, 0.0, 0.0, 0.0};
    std::string momentum_input_mode;
    std::string global_position_status;
    std::string global_momentum_status;
};

struct PhotonResult {
    long long event_id = 0;
    long long particle_id = 0;
    int pdg = 22;
    double input_energy_gev = std::numeric_limits<double>::quiet_NaN();
    std::string classification = "integration_failed";
    double null_norm_initial = std::numeric_limits<double>::quiet_NaN();
    double null_norm_max_abs_error = std::numeric_limits<double>::quiet_NaN();
    double E_killing_initial = std::numeric_limits<double>::quiet_NaN();
    double E_killing_final = std::numeric_limits<double>::quiet_NaN();
    double Lz_initial = std::numeric_limits<double>::quiet_NaN();
    double Lz_final = std::numeric_limits<double>::quiet_NaN();
    double relative_E_error = std::numeric_limits<double>::quiet_NaN();
    double relative_Lz_error = std::numeric_limits<double>::quiet_NaN();
    std::string invariant_status = "not_evaluated";
    int geodesic_steps = 0;
    std::string momentum_input_mode;
    double initial_r_rg = std::numeric_limits<double>::quiet_NaN();
    double initial_theta_rad = std::numeric_limits<double>::quiet_NaN();
    double initial_phi_rad = std::numeric_limits<double>::quiet_NaN();
    double p_initial[4] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
    };
    double p_crossing[4] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
    };
    bool crossing_momentum_available = false;
    bool observer_crossing_interpolated = false;
    double observer_crossing_r_rg = std::numeric_limits<double>::quiet_NaN();
    double observer_crossing_theta_rad = std::numeric_limits<double>::quiet_NaN();
    double observer_crossing_phi_rad = std::numeric_limits<double>::quiet_NaN();
    std::string failure_reason;
};

bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

bool has_key(const std::string& line, const std::string& key)
{
    return line.find("\"" + key + "\"") != std::string::npos;
}

double number_value_or(const std::string& line, const std::string& key, double fallback)
{
    const std::string needle = "\"" + key + "\":";
    auto begin = line.find(needle);
    if (begin == std::string::npos) {
        return fallback;
    }
    begin += needle.size();
    const auto end = line.find_first_of(",}", begin);
    try {
        return std::stod(line.substr(begin, end - begin));
    } catch (...) {
        return fallback;
    }
}

long long int_value_or(const std::string& line, const std::string& key, long long fallback)
{
    const double value = number_value_or(line, key, std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(value)) {
        return fallback;
    }
    return static_cast<long long>(value);
}

std::string string_value_or(const std::string& line, const std::string& key, const std::string& fallback)
{
    const std::string needle = "\"" + key + "\":\"";
    auto begin = line.find(needle);
    if (begin == std::string::npos) {
        const std::string spaced = "\"" + key + "\": \"";
        begin = line.find(spaced);
        if (begin == std::string::npos) {
            return fallback;
        }
        begin += spaced.size();
    } else {
        begin += needle.size();
    }
    const auto end = line.find('"', begin);
    if (end == std::string::npos) {
        return fallback;
    }
    return line.substr(begin, end - begin);
}

double first_number(const std::string& line, std::initializer_list<const char*> keys)
{
    for (const char* key : keys) {
        const double value = number_value_or(line, key, std::numeric_limits<double>::quiet_NaN());
        if (std::isfinite(value)) {
            return value;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void spherical_from_xyz(double x, double y, double z, double& r, double& theta, double& phi)
{
    r = std::sqrt(x*x + y*y + z*z);
    if (!std::isfinite(r) || r <= 0.0) {
        theta = std::numeric_limits<double>::quiet_NaN();
        phi = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    theta = std::acos(std::clamp(z / r, -1.0, 1.0));
    phi = std::atan2(y, x);
}

PhotonRecord parse_record(const std::string& line)
{
    PhotonRecord record;
    record.event_id = int_value_or(line, "event_id", 0);
    record.particle_id = int_value_or(line, "particle_id", int_value_or(line, "source_particle_id", 0));
    record.pdg = static_cast<int>(int_value_or(line, "pdg", int_value_or(line, "pdg_id", 0)));
    record.energy_gev = first_number(line, {"energy_gev", "input_energy_gev"});
    record.global_position_status = string_value_or(line, "global_position_status", "");
    record.global_momentum_status = string_value_or(line, "global_momentum_status", "");
    record.momentum_input_mode = string_value_or(line, "momentum_input_mode", "");
    record.has_ambiguous_generic_momentum_input = record.momentum_input_mode.empty()
        && (has_key(line, "px") || has_key(line, "px_gev")
            || has_key(line, "py") || has_key(line, "py_gev")
            || has_key(line, "pz") || has_key(line, "pz_gev")
            || has_key(line, "n_r") || has_key(line, "nr") || has_key(line, "local_n_r")
            || has_key(line, "n_theta") || has_key(line, "ntheta") || has_key(line, "local_n_theta")
            || has_key(line, "n_phi") || has_key(line, "nphi") || has_key(line, "local_n_phi")
            || has_key(line, "p_t") || has_key(line, "initial_p_t")
            || has_key(line, "global_px") || has_key(line, "global_py") || has_key(line, "global_pz"));

    record.r = first_number(line, {"global_exit_r_rg", "r_rg", "initial_r_rg"});
    record.theta = first_number(line, {"global_exit_theta_rad", "theta_rad", "initial_theta_rad"});
    record.phi = first_number(line, {"global_exit_phi_rad", "phi_rad", "initial_phi_rad"});
    if (!std::isfinite(record.r) || !std::isfinite(record.theta) || !std::isfinite(record.phi)) {
        const double x = first_number(line, {"global_exit_x_rg", "x_rg", "initial_x_rg"});
        const double y = first_number(line, {"global_exit_y_rg", "y_rg", "initial_y_rg"});
        const double z = first_number(line, {"global_exit_z_rg", "z_rg", "initial_z_rg"});
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            spherical_from_xyz(x, y, z, record.r, record.theta, record.phi);
        }
    }

    if (record.momentum_input_mode == "covariant_p_mu") {
        if (!has_key(line, "p_t") && !has_key(line, "initial_p_t")) {
            return record;
        }
        record.has_direct_pcov = true;
        record.pcov[0] = first_number(line, {"p_t", "initial_p_t"});
        record.pcov[1] = first_number(line, {"p_r", "initial_p_r"});
        record.pcov[2] = first_number(line, {"p_theta", "initial_p_theta"});
        record.pcov[3] = first_number(line, {"p_phi", "initial_p_phi"});
    } else if (record.momentum_input_mode == "zamo_tetrad") {
        record.nr = first_number(line, {"n_zamo_r", "n_r", "nr", "local_n_r"});
        record.ntheta = first_number(line, {"n_zamo_theta", "n_theta", "ntheta", "local_n_theta"});
        record.nphi = first_number(line, {"n_zamo_phi", "n_phi", "nphi", "local_n_phi"});
        if (!std::isfinite(record.nr) || !std::isfinite(record.ntheta) || !std::isfinite(record.nphi)) {
            const double px = first_number(line, {"px", "px_gev"});
            const double py = first_number(line, {"py", "py_gev"});
            const double pz = first_number(line, {"pz", "pz_gev"});
            if (std::isfinite(px) && std::isfinite(py) && std::isfinite(pz)) {
                // GEANT4 local box convention used upstream:
                // x -> +theta, y -> +phi, z -> outward radial ZAMO axis.
                record.nr = pz;
                record.ntheta = px;
                record.nphi = py;
            }
        }
    } else if (record.momentum_input_mode == "global_boyer_lindquist") {
        const double gx = first_number(line, {"global_px"});
        const double gy = first_number(line, {"global_py"});
        const double gz = first_number(line, {"global_pz"});
        if (std::isfinite(gx) && std::isfinite(gy) && std::isfinite(gz)
            && std::isfinite(record.theta) && std::isfinite(record.phi)) {
            const double sin_t = std::sin(record.theta);
            const double cos_t = std::cos(record.theta);
            const double sin_p = std::sin(record.phi);
            const double cos_p = std::cos(record.phi);
            record.nr = gx * sin_t * cos_p + gy * sin_t * sin_p + gz * cos_t;
            record.ntheta = gx * cos_t * cos_p + gy * cos_t * sin_p - gz * sin_t;
            record.nphi = -gx * sin_p + gy * cos_p;
        }
    }
    return record;
}

void ensure_output(std::ofstream& stream, const std::string& path)
{
    if (!stream) {
        throw std::runtime_error("Could not open output file: " + path);
    }
    stream << std::setprecision(17);
}

double null_norm_from_pcov(const KerrMetric& metric, const GeodesicState& state)
{
    double ginv[4][4];
    metric.inverse_metric(state.r, state.theta, ginv);
    const double p[4] = {state.pt, state.pr, state.ptheta, state.pphi};
    double norm = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            norm += ginv[mu][nu] * p[mu] * p[nu];
        }
    }
    return norm;
}

bool finite_state(const GeodesicState& state)
{
    return std::isfinite(state.t) && std::isfinite(state.r)
        && std::isfinite(state.theta) && std::isfinite(state.phi)
        && std::isfinite(state.pt) && std::isfinite(state.pr)
        && std::isfinite(state.ptheta) && std::isfinite(state.pphi);
}

double relative_error(double initial, double final)
{
    return std::abs(final - initial) / std::max(std::abs(initial), REL_EPS);
}

void store_pcov(double out[4], const GeodesicState& state)
{
    out[0] = state.pt;
    out[1] = state.pr;
    out[2] = state.ptheta;
    out[3] = state.pphi;
}

void store_interpolated_pcov(double out[4], const GeodesicState& previous, const GeodesicState& current, double alpha)
{
    out[0] = previous.pt + alpha * (current.pt - previous.pt);
    out[1] = previous.pr + alpha * (current.pr - previous.pr);
    out[2] = previous.ptheta + alpha * (current.ptheta - previous.ptheta);
    out[3] = previous.pphi + alpha * (current.pphi - previous.pphi);
}

bool finite_pcov(const double p[4])
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && std::isfinite(p[3]);
}

bool valid_config(const PhotonEscapeConfig& config)
{
    return std::isfinite(config.spin)
        && std::isfinite(config.observer_radius_rg) && config.observer_radius_rg > 0.0
        && std::isfinite(config.max_radius_rg) && config.max_radius_rg > config.observer_radius_rg
        && std::isfinite(config.geodesic_step_rg) && config.geodesic_step_rg > 0.0
        && config.max_geodesic_steps > 0
        && std::isfinite(config.photon_null_norm_tolerance) && config.photon_null_norm_tolerance > 0.0
        && std::isfinite(config.photon_invariant_tolerance) && config.photon_invariant_tolerance > 0.0
        && std::isfinite(config.photon_horizon_crossing_tolerance_rg) && config.photon_horizon_crossing_tolerance_rg >= 0.0
        && std::isfinite(config.photon_min_energy_gev) && config.photon_min_energy_gev >= 0.0
        && config.observer_frame == "ZAMO";
}

PhotonResult fail_result(const PhotonRecord& record, const std::string& classification, const std::string& reason)
{
    PhotonResult result;
    result.event_id = record.event_id;
    result.particle_id = record.particle_id;
    result.pdg = record.pdg;
    result.input_energy_gev = record.energy_gev;
    result.momentum_input_mode = record.momentum_input_mode;
    result.classification = classification;
    result.failure_reason = reason;
    return result;
}

bool position_status_valid(const std::string& status)
{
    return starts_with(status, "GLOBAL_POSITION_VALID");
}

bool momentum_status_valid(const std::string& status)
{
    return starts_with(status, "GLOBAL_MOMENTUM")
        && status.find("NOT_AVAILABLE") == std::string::npos
        && status.find("INVALID") == std::string::npos;
}

bool momentum_input_mode_valid(const std::string& mode)
{
    return mode == "zamo_tetrad"
        || mode == "global_boyer_lindquist"
        || mode == "covariant_p_mu";
}

bool has_generic_momentum_without_mode(const PhotonRecord& record)
{
    return record.has_ambiguous_generic_momentum_input;
}

PhotonResult classify_photon(const PhotonRecord& record, const PhotonEscapeConfig& config)
{
    PhotonResult result;
    result.event_id = record.event_id;
    result.particle_id = record.particle_id;
    result.pdg = record.pdg;
    result.input_energy_gev = record.energy_gev;
    result.momentum_input_mode = record.momentum_input_mode;

    if (!std::isfinite(record.energy_gev) || record.energy_gev <= 0.0) {
        return fail_result(record, "integration_failed", "integration_failed_missing_photon_energy");
    }
    if (record.energy_gev < config.photon_min_energy_gev) {
        return fail_result(record, "integration_failed", "integration_failed_below_photon_min_energy");
    }
    if (!position_status_valid(record.global_position_status)) {
        return fail_result(record, "integration_failed", "integration_failed_unvalidated_global_position");
    }
    if (!momentum_status_valid(record.global_momentum_status)) {
        return fail_result(record, "integration_failed", "integration_failed_unvalidated_global_momentum");
    }
    if (!std::isfinite(record.r) || !std::isfinite(record.theta) || !std::isfinite(record.phi)) {
        return fail_result(record, "integration_failed", "integration_failed_missing_valid_global_position");
    }
    if (has_generic_momentum_without_mode(record)) {
        return fail_result(record, "integration_failed_ambiguous_momentum_input", "momentum_input_mode_required_for_generic_momentum_fields");
    }
    if (!momentum_input_mode_valid(record.momentum_input_mode)) {
        return fail_result(record, "integration_failed", "integration_failed_invalid_or_missing_momentum_input_mode");
    }

    KerrMetric metric(config.spin);
    const double horizon = metric.horizon_radius();
    const double horizon_crossing_radius = horizon + config.photon_horizon_crossing_tolerance_rg;
    if (record.r <= horizon_crossing_radius) {
        return fail_result(record, "integration_failed", "integration_failed_initial_position_at_or_inside_horizon");
    }

    GeodesicState state{};
    state.t = 0.0;
    state.r = record.r;
    state.theta = std::clamp(record.theta, 1.0e-12, PI - 1.0e-12);
    state.phi = record.phi;

    if (record.has_direct_pcov) {
        for (double component : record.pcov) {
            if (!std::isfinite(component)) {
                return fail_result(record, "integration_failed_invalid_null_momentum", "invalid_direct_covariant_momentum");
            }
        }
        state.pt = record.pcov[0];
        state.pr = record.pcov[1];
        state.ptheta = record.pcov[2];
        state.pphi = record.pcov[3];
    } else {
        const double direction_norm = std::sqrt(
            record.nr * record.nr + record.ntheta * record.ntheta + record.nphi * record.nphi
        );
        if (!std::isfinite(direction_norm) || direction_norm <= 0.0) {
            return fail_result(record, "integration_failed", "integration_failed_missing_valid_global_momentum");
        }
        hadros::cascade::LocalDirection direction{
            record.nr / direction_norm,
            record.ntheta / direction_norm,
            record.nphi / direction_norm,
        };
        auto init = hadros::cascade::initialize_zamo_null_packet(
            metric, state.r, state.theta, state.phi, direction
        );
        if (!init.valid) {
            return fail_result(record, "integration_failed", "integration_failed_zamo_tetrad_initialization_" + init.status);
        }
        state = init.state;
        state.pt *= record.energy_gev;
        state.pr *= record.energy_gev;
        state.ptheta *= record.energy_gev;
        state.pphi *= record.energy_gev;
    }

    result.null_norm_initial = null_norm_from_pcov(metric, state);
    result.null_norm_max_abs_error = std::abs(result.null_norm_initial);
    if (!std::isfinite(result.null_norm_initial)
        || std::abs(result.null_norm_initial) > config.photon_null_norm_tolerance) {
        result.classification = "integration_failed_invalid_null_momentum";
        result.failure_reason = "null_norm_initial_out_of_tolerance";
        return result;
    }

    result.initial_r_rg = state.r;
    result.initial_theta_rad = state.theta;
    result.initial_phi_rad = state.phi;
    store_pcov(result.p_initial, state);
    result.E_killing_initial = -state.pt;
    result.Lz_initial = state.pphi;

    KerrGeodesic geodesic(metric, config.geodesic_step_rg);
    bool reached = false;
    bool captured = false;
    bool missed = false;
    bool failed = false;

    if (state.r >= config.observer_radius_rg) {
        reached = true;
        result.observer_crossing_interpolated = false;
        result.observer_crossing_r_rg = state.r;
        result.observer_crossing_theta_rad = state.theta;
        result.observer_crossing_phi_rad = state.phi;
        store_pcov(result.p_crossing, state);
        result.crossing_momentum_available = finite_pcov(result.p_crossing);
    }

    for (int step = 0; !reached && !captured && !missed && !failed && step < config.max_geodesic_steps; ++step) {
        const GeodesicState previous_state = state;
        geodesic.step_rk4(state);
        result.geodesic_steps = step + 1;
        if (!finite_state(state)) {
            failed = true;
            result.failure_reason = "nonfinite_geodesic_state";
            break;
        }
        if (previous_state.r > horizon && state.r <= horizon_crossing_radius) {
            captured = true;
            break;
        }
        const double norm = null_norm_from_pcov(metric, state);
        if (std::isfinite(norm)) {
            result.null_norm_max_abs_error = std::max(result.null_norm_max_abs_error, std::abs(norm));
        }
        const double prev_delta = previous_state.r - config.observer_radius_rg;
        const double curr_delta = state.r - config.observer_radius_rg;
        if (prev_delta * curr_delta <= 0.0) {
            const double denom = state.r - previous_state.r;
            const double alpha = std::isfinite(denom) && std::abs(denom) > REL_EPS
                ? std::clamp((config.observer_radius_rg - previous_state.r) / denom, 0.0, 1.0)
                : 0.0;
            result.observer_crossing_interpolated = true;
            result.observer_crossing_r_rg = config.observer_radius_rg;
            result.observer_crossing_theta_rad = previous_state.theta + alpha * (state.theta - previous_state.theta);
            result.observer_crossing_phi_rad = previous_state.phi + alpha * (state.phi - previous_state.phi);
            store_interpolated_pcov(result.p_crossing, previous_state, state, alpha);
            result.crossing_momentum_available = finite_pcov(result.p_crossing);
            reached = true;
            break;
        }
        if (state.r > config.max_radius_rg) {
            missed = true;
            break;
        }
    }

    if (!reached && !captured && !missed && !failed) {
        failed = true;
        result.failure_reason = "max_geodesic_steps_exceeded";
    }

    result.E_killing_final = -state.pt;
    result.Lz_final = state.pphi;
    result.relative_E_error = relative_error(result.E_killing_initial, result.E_killing_final);
    result.relative_Lz_error = relative_error(result.Lz_initial, result.Lz_final);

    const bool invariant_violation =
        result.null_norm_max_abs_error > config.photon_invariant_tolerance
        || result.relative_E_error > config.photon_invariant_tolerance
        || result.relative_Lz_error > config.photon_invariant_tolerance;

    if (invariant_violation) {
        result.invariant_status = config.photon_fail_on_invariant_violation
            ? "failed_invariant_violation"
            : "warning_invariant_violation";
        if (config.photon_fail_on_invariant_violation) {
            result.classification = "integration_failed_invariant_violation";
            result.failure_reason = "invariant_violation";
            return result;
        }
    } else {
        result.invariant_status = "pass";
    }

    if (reached) {
        result.classification = "reaches_observer_sphere";
    } else if (captured) {
        result.classification = "captured_by_black_hole";
    } else if (missed) {
        result.classification = "escapes_but_misses_observer";
    } else {
        result.classification = "integration_failed";
        if (result.failure_reason.empty()) {
            result.failure_reason = "integration_failed";
        }
    }
    return result;
}

void write_json_field(std::ostream& out, const std::string& key, const std::string& value, bool comma = true)
{
    if (comma) {
        out << ",";
    }
    out << "\"" << key << "\":\"";
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\';
        }
        out << ch;
    }
    out << "\"";
}

void write_json_number(std::ostream& out, const std::string& key, double value)
{
    out << ",\"" << key << "\":";
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

void write_result(std::ostream& out, const PhotonResult& result)
{
    out << "{\"event_id\":" << result.event_id
        << ",\"particle_id\":" << result.particle_id
        << ",\"pdg\":" << result.pdg;
    write_json_number(out, "input_energy_gev", result.input_energy_gev);
    write_json_field(out, "classification", result.classification);
    write_json_number(out, "null_norm_initial", result.null_norm_initial);
    write_json_number(out, "null_norm_max_abs_error", result.null_norm_max_abs_error);
    write_json_number(out, "E_killing_initial", result.E_killing_initial);
    write_json_number(out, "E_killing_final", result.E_killing_final);
    write_json_number(out, "Lz_initial", result.Lz_initial);
    write_json_number(out, "Lz_final", result.Lz_final);
    write_json_number(out, "relative_E_error", result.relative_E_error);
    write_json_number(out, "relative_Lz_error", result.relative_Lz_error);
    write_json_field(out, "invariant_status", result.invariant_status);
    write_json_field(out, "momentum_input_mode", result.momentum_input_mode);
    write_json_number(out, "initial_r_rg", result.initial_r_rg);
    write_json_number(out, "initial_theta_rad", result.initial_theta_rad);
    write_json_number(out, "initial_phi_rad", result.initial_phi_rad);
    write_json_number(out, "p_t_initial", result.p_initial[0]);
    write_json_number(out, "p_r_initial", result.p_initial[1]);
    write_json_number(out, "p_theta_initial", result.p_initial[2]);
    write_json_number(out, "p_phi_initial", result.p_initial[3]);
    out << ",\"crossing_momentum_available\":" << (result.crossing_momentum_available ? "true" : "false");
    if (result.crossing_momentum_available) {
        write_json_number(out, "p_t_crossing", result.p_crossing[0]);
        write_json_number(out, "p_r_crossing", result.p_crossing[1]);
        write_json_number(out, "p_theta_crossing", result.p_crossing[2]);
        write_json_number(out, "p_phi_crossing", result.p_crossing[3]);
    }
    out << ",\"geodesic_steps\":" << result.geodesic_steps;
    out << ",\"observer_crossing_interpolated\":" << (result.observer_crossing_interpolated ? "true" : "false");
    write_json_number(out, "observer_crossing_r_rg", result.observer_crossing_r_rg);
    write_json_number(out, "observer_crossing_theta_rad", result.observer_crossing_theta_rad);
    write_json_number(out, "observer_crossing_phi_rad", result.observer_crossing_phi_rad);
    write_json_field(out, "failure_reason", result.failure_reason);
    out << "}\n";
}

void update_summary(PhotonEscapeSummary& summary, const PhotonResult& result)
{
    if (result.classification == "captured_by_black_hole") {
        ++summary.n_captured;
    } else if (result.classification == "reaches_observer_sphere") {
        ++summary.n_reached_observer_sphere;
    } else if (result.classification == "escapes_but_misses_observer") {
        ++summary.n_missed;
    } else {
        ++summary.n_failed;
        if (result.classification == "integration_failed_invalid_null_momentum") {
            ++summary.n_failed_invalid_null_momentum;
        }
        if (result.classification == "integration_failed_invariant_violation") {
            ++summary.n_failed_invariant_violation;
        }
    }
}

void write_summary_csv(const std::string& path, const PhotonEscapeSummary& summary)
{
    std::ofstream out(path);
    ensure_output(out, path);
    out << "n_input_particles,n_photons,n_non_photons,n_captured,"
           "n_reached_observer_sphere,n_missed,n_failed,"
           "n_failed_invalid_null_momentum,n_failed_invariant_violation,"
           "total_particles_seen,total_photons_seen,total_non_photons_ignored\n";
    out << summary.n_input_particles << ","
        << summary.n_photons << ","
        << summary.n_non_photons << ","
        << summary.n_captured << ","
        << summary.n_reached_observer_sphere << ","
        << summary.n_missed << ","
        << summary.n_failed << ","
        << summary.n_failed_invalid_null_momentum << ","
        << summary.n_failed_invariant_violation << ","
        << summary.n_input_particles << ","
        << summary.n_photons << ","
        << summary.n_non_photons << "\n";
}

void write_summary_md(const std::string& path, const PhotonEscapeSummary& summary)
{
    std::ofstream out(path);
    ensure_output(out, path);
    out << "# Photon Escape Classifier Summary\n\n"
        << "- camera_physical_interpretation: `photon_escape_classifier`\n"
        << "- camera_is_full_observational_transport: `false`\n"
        << "- projected_to_pixels: `false`\n"
        << "- observer_sphere_crossing_is_detection: `false`\n"
        << "- photon_only: `true`\n"
        << "- charged_particle_transport_enabled: `false`\n\n"
        << "| metric | value |\n|---|---:|\n"
        << "| n_input_particles | " << summary.n_input_particles << " |\n"
        << "| n_photons | " << summary.n_photons << " |\n"
        << "| n_non_photons | " << summary.n_non_photons << " |\n"
        << "| total_particles_seen | " << summary.n_input_particles << " |\n"
        << "| total_photons_seen | " << summary.n_photons << " |\n"
        << "| total_non_photons_ignored | " << summary.n_non_photons << " |\n"
        << "| n_captured | " << summary.n_captured << " |\n"
        << "| n_reached_observer_sphere | " << summary.n_reached_observer_sphere << " |\n"
        << "| n_missed | " << summary.n_missed << " |\n"
        << "| n_failed | " << summary.n_failed << " |\n";
}

void write_provenance(const std::string& path, const PhotonEscapeConfig& config, const PhotonEscapeSummary& summary)
{
    std::ofstream out(path);
    ensure_output(out, path);
    out << "{\n"
        << "  \"camera_physical_interpretation\":\"photon_escape_classifier\",\n"
        << "  \"camera_is_full_observational_transport\":false,\n"
        << "  \"projected_to_pixels\":false,\n"
        << "  \"observer_sphere_crossing_is_detection\":false,\n"
        << "  \"photon_only\":true,\n"
        << "  \"charged_particle_transport_enabled\":false,\n"
        << "  \"photon_observer_frame\":\"" << config.observer_frame << "\",\n"
        << "  \"effective_photon_observer_radius_rg\":" << config.observer_radius_rg << ",\n"
        << "  \"photon_null_norm_tolerance\":" << config.photon_null_norm_tolerance << ",\n"
        << "  \"photon_invariant_tolerance\":" << config.photon_invariant_tolerance << ",\n"
        << "  \"photon_horizon_crossing_tolerance_rg\":" << config.photon_horizon_crossing_tolerance_rg << ",\n"
        << "  \"photon_fail_on_invariant_violation\":" << (config.photon_fail_on_invariant_violation ? "true" : "false") << ",\n"
        << "  \"photon_min_energy_gev\":" << config.photon_min_energy_gev << ",\n"
        << "  \"photon_geodesic_step_rg\":" << config.geodesic_step_rg << ",\n"
        << "  \"photon_max_geodesic_steps\":" << config.max_geodesic_steps << ",\n"
        << "  \"momentum_input_mode\":\"per_record_required\",\n"
        << "  \"momentum_input_mode_allowed_values\":[\"zamo_tetrad\",\"global_boyer_lindquist\",\"covariant_p_mu\"],\n"
        << "  \"observer_crossing_interpolation\":\"linear in integration step; classification only, not detection\",\n"
        << "  \"crossing_momentum_interpolation\":\"linear_between_geodesic_steps\",\n"
        << "  \"horizon_capture_definition\":\"captured_by_black_hole requires crossing from r > r_plus to r <= r_plus + photon_horizon_crossing_tolerance_rg\",\n"
        << "  \"invariant_violation_policy\":\"photon_fail_on_invariant_violation=true always classifies invariant drift as integration_failed_invariant_violation\",\n"
        << "  \"n_input_particles\":" << summary.n_input_particles << ",\n"
        << "  \"n_photons\":" << summary.n_photons << ",\n"
        << "  \"n_non_photons\":" << summary.n_non_photons << ",\n"
        << "  \"total_particles_seen\":" << summary.n_input_particles << ",\n"
        << "  \"total_photons_seen\":" << summary.n_photons << ",\n"
        << "  \"total_non_photons_ignored\":" << summary.n_non_photons << ",\n"
        << "  \"limitation\":\"Phase 1 classifies photon geodesic destinations only; no detector, pixels, images, or observed-energy redshift are produced.\"\n"
        << "}\n";
}

}  // namespace

PhotonEscapeSummary run_photon_escape_classifier(
    const std::string& input_jsonl,
    const std::string& output_jsonl,
    const std::string& summary_csv,
    const std::string& summary_md,
    const std::string& provenance_json,
    const PhotonEscapeConfig& config
)
{
    if (!valid_config(config)) {
        throw std::runtime_error("Invalid photon escape classifier configuration; check config_web_final.py values");
    }

    std::ifstream input(input_jsonl);
    if (!input) {
        throw std::runtime_error("Could not open input file: " + input_jsonl);
    }
    std::ofstream output(output_jsonl);
    ensure_output(output, output_jsonl);

    PhotonEscapeSummary summary;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        ++summary.n_input_particles;
        PhotonRecord record = parse_record(line);
        if (record.pdg != 22) {
            ++summary.n_non_photons;
            continue;
        }
        ++summary.n_photons;
        PhotonResult result = classify_photon(record, config);
        update_summary(summary, result);
        write_result(output, result);
    }

    write_summary_csv(summary_csv, summary);
    write_summary_md(summary_md, summary);
    write_provenance(provenance_json, config, summary);
    return summary;
}
