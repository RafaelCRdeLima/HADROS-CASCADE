#include "torus_profile.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;
}

TorusProfile::TorusProfile(
    double rho0_gcm3,
    double r0_rg,
    double sigma_r,
    double H_over_R,
    std::string profile_name,
    double radial_power,
    double funnel_depletion,
    double funnel_theta_rad,
    double envelope_rho0_gcm3,
    double envelope_alpha,
    double r_min_rg,
    double r_max_rg,
    double rho_floor_gcm3
)
    : rho0_(rho0_gcm3),
      r0_(r0_rg),
      sigma_r_(sigma_r),
      H_over_R_(H_over_R),
      profile_name_(std::move(profile_name)),
      profile_kind_(parse_profile_kind(profile_name_)),
      radial_power_(radial_power),
      funnel_depletion_(std::clamp(funnel_depletion, 0.0, 1.0)),
      funnel_theta_rad_(std::max(funnel_theta_rad, 1.0e-6)),
      envelope_rho0_(std::max(envelope_rho0_gcm3, 0.0)),
      envelope_alpha_(envelope_alpha),
      r_min_(r_min_rg),
      r_max_(std::max(r_max_rg, r_min_rg)),
      rho_floor_(std::max(rho_floor_gcm3, 0.0))
{
}

DensityProfileKind TorusProfile::parse_profile_kind(
    const std::string& profile_name
)
{
    if (profile_name == "gaussian" ||
        profile_name == "torus" ||
        profile_name == "gaussian_torus") {
        return DensityProfileKind::Gaussian;
    }

    if (profile_name == "powerlaw" ||
        profile_name == "powerlaw_disk") {
        return DensityProfileKind::PowerLaw;
    }

    if (profile_name == "funnel" ||
        profile_name == "gaussian_funnel") {
        return DensityProfileKind::GaussianFunnel;
    }

    if (profile_name == "powerlaw_funnel") {
        return DensityProfileKind::PowerLawFunnel;
    }

    if (profile_name == "gaussian_envelope") {
        return DensityProfileKind::GaussianEnvelope;
    }

    if (profile_name == "powerlaw_envelope") {
        return DensityProfileKind::PowerLawEnvelope;
    }

    if (profile_name == "powerlaw_funnel_envelope") {
        return DensityProfileKind::PowerLawFunnelEnvelope;
    }

    if (profile_name == "collapsar_ndaf_like" ||
        profile_name == "collapsar_ndaf") {
        return DensityProfileKind::CollapsarNdafLike;
    }

    throw std::runtime_error(
        "Unknown density profile '" + profile_name +
        "'. Use gaussian, powerlaw, gaussian_funnel, powerlaw_funnel, "
        "gaussian_envelope, powerlaw_envelope, powerlaw_funnel_envelope, "
        "or collapsar_ndaf_like."
    );
}

const std::string& TorusProfile::profile_name() const
{
    return profile_name_;
}

double TorusProfile::rho0_gcm3() const
{
    return rho0_;
}

double TorusProfile::r0_rg() const
{
    return r0_;
}

double TorusProfile::r_min_rg() const
{
    return r_min_;
}

double TorusProfile::r_max_rg() const
{
    return r_max_;
}

bool TorusProfile::in_torus(double r_rg, double theta) const
{
    (void)theta;

    return r_rg >= r_min_ && r_rg <= r_max_;
}

bool TorusProfile::uses_powerlaw_disk() const
{
    return
        profile_kind_ == DensityProfileKind::PowerLaw ||
        profile_kind_ == DensityProfileKind::PowerLawFunnel ||
        profile_kind_ == DensityProfileKind::PowerLawEnvelope ||
        profile_kind_ == DensityProfileKind::PowerLawFunnelEnvelope;
}

bool TorusProfile::uses_funnel() const
{
    return
        profile_kind_ == DensityProfileKind::GaussianFunnel ||
        profile_kind_ == DensityProfileKind::PowerLawFunnel ||
        profile_kind_ == DensityProfileKind::PowerLawFunnelEnvelope;
}

bool TorusProfile::uses_envelope() const
{
    return
        profile_kind_ == DensityProfileKind::GaussianEnvelope ||
        profile_kind_ == DensityProfileKind::PowerLawEnvelope ||
        profile_kind_ == DensityProfileKind::PowerLawFunnelEnvelope;
}

double TorusProfile::collapsar_ndaf_shape(double r_rg, double theta) const
{
    if (!in_torus(r_rg, theta) || r_rg <= 0.0 || r0_ <= 0.0) {
        return 0.0;
    }

    const double mu = std::cos(theta);
    const double vertical =
        std::exp(-std::pow(mu / std::max(H_over_R_, 1.0e-6), 2.0));

    const double inner_taper =
        1.0 / (1.0 + std::exp(-(r_rg - r_min_) / std::max(0.35 * sigma_r_, 1.0e-6)));

    const double outer_taper =
        1.0 / (1.0 + std::exp((r_rg - r_max_) / std::max(0.60 * sigma_r_, 1.0e-6)));

    const double inner_core =
        0.45 * std::exp(-std::pow((r_rg - 0.75 * r0_) / std::max(0.75 * sigma_r_, 1.0e-6), 2.0));

    const double radial =
        std::pow(std::max(r_rg / r0_, 1.0e-300), -std::max(radial_power_, 0.0));

    const double floor_tail =
        0.08 * std::exp(-std::pow((r_rg - r0_) / std::max(1.8 * sigma_r_, 1.0e-6), 2.0));

    return (radial + inner_core + floor_tail)
        * vertical
        * inner_taper
        * outer_taper;
}

double TorusProfile::gaussian_shape(double r_rg, double theta) const
{
    const double delta = theta - 0.5 * PI;

    // Use configurable r_min_, r_max_ and an H_over_R_-derived vertical cutoff
    // (clip beyond 4σ where the Gaussian is < exp(-16) ≈ 1e-7).
    const double delta_cut = std::min(PI * 0.5, 4.0 * std::max(H_over_R_, 1.0e-6));

    if (r_rg <= r_min_ || r_rg >= r_max_ || std::abs(delta) >= delta_cut) {
        return 0.0;
    }

    const double radial =
        std::exp(-std::pow((r_rg - r0_) / sigma_r_, 2.0));

    const double vertical =
        std::exp(-std::pow(delta / H_over_R_, 2.0));

    return radial * vertical;
}

double TorusProfile::powerlaw_shape(double r_rg, double theta) const
{
    if (!in_torus(r_rg, theta) || r_rg <= 0.0 || r0_ <= 0.0) {
        return 0.0;
    }

    const double vertical =
        std::exp(-std::pow(std::cos(theta) / H_over_R_, 2.0));

    const double inner_taper =
        1.0 - std::exp(-std::pow((r_rg - r_min_) / sigma_r_, 2.0));

    const double outer_taper =
        std::exp(-std::pow(r_rg / r_max_, 4.0));

    const double radial =
        std::pow(r_rg / r0_, -radial_power_);

    return radial * vertical * inner_taper * outer_taper;
}

double TorusProfile::funnel_factor(double theta) const
{
    if (!uses_funnel()) {
        return 1.0;
    }

    const double north =
        std::exp(-std::pow(theta / funnel_theta_rad_, 2.0));

    const double south =
        std::exp(-std::pow((PI - theta) / funnel_theta_rad_, 2.0));

    return std::clamp(
        1.0 - funnel_depletion_ * (north + south),
        0.0,
        1.0
    );
}

double TorusProfile::envelope_rho(double r_rg) const
{
    if (!uses_envelope() ||
        envelope_rho0_ <= 0.0 ||
        r_rg < r0_ ||
        r_rg > r_max_) {
        return 0.0;
    }

    return envelope_rho0_ * std::pow(r_rg / r0_, -envelope_alpha_);
}

double TorusProfile::disk_shape(double r_rg, double theta) const
{
    if (profile_kind_ == DensityProfileKind::CollapsarNdafLike) {
        return collapsar_ndaf_shape(r_rg, theta);
    }

    return uses_powerlaw_disk()
        ? powerlaw_shape(r_rg, theta)
        : gaussian_shape(r_rg, theta);
}

double TorusProfile::rho(double r_rg, double theta) const
{
    return std::max(raw_rho(r_rg, theta), rho_floor_);
}

double TorusProfile::raw_rho(double r_rg, double theta) const
{
    const double disk =
        rho0_ * disk_shape(r_rg, theta) * funnel_factor(theta);

    return disk + envelope_rho(r_rg);
}

double TorusProfile::temperature_MeV(double r_rg, double theta) const
{
    const double shape =
        std::clamp(rho(r_rg, theta) / std::max(rho0_, 1.0e-300), 0.0, 1.0);

    double T =
        6.0 * std::pow(shape, 0.2);

    return std::max(T, 1.0e-10);
}

double TorusProfile::Ye(double r_rg, double theta) const
{
    if (rho(r_rg, theta) <= 0.0) {
        return 0.0;
    }
    (void)theta;

    return 0.2 + 0.1 * std::exp(-std::pow((r_rg - r0_) / 4.0, 2.0));
}
