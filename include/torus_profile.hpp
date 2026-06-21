#ifndef TORUS_PROFILE_HPP
#define TORUS_PROFILE_HPP

#include <string>

enum class DensityProfileKind {
    Gaussian,
    PowerLaw,
    GaussianFunnel,
    PowerLawFunnel,
    GaussianEnvelope,
    PowerLawEnvelope,
    PowerLawFunnelEnvelope,
    CollapsarNdafLike
};

class TorusProfile {
public:
    TorusProfile(
        double rho0_gcm3 = 1.0e11,
        double r0_rg     = 10.0,
        double sigma_r   = 5.0,
        double H_over_R  = 0.30,
        std::string profile_name = "gaussian",
        double radial_power = 2.0,
        double funnel_depletion = 0.0,
        double funnel_theta_rad = 0.25,
        double envelope_rho0_gcm3 = 0.0,
        double envelope_alpha = 2.5,
        double r_min_rg = 4.0,
        double r_max_rg = 60.0,
        double rho_floor_gcm3 = 1.0e-99
    );

    double rho(double r_rg, double theta) const;
    double raw_rho(double r_rg, double theta) const;
    double temperature_MeV(double r_rg, double theta) const;
    double Ye(double r_rg, double theta) const;
    bool in_torus(double r_rg, double theta) const;
    const std::string& profile_name() const;
    double rho0_gcm3() const;
    double r0_rg() const;
    double r_min_rg() const;
    double r_max_rg() const;

    static DensityProfileKind parse_profile_kind(
        const std::string& profile_name
    );

private:
    double disk_shape(double r_rg, double theta) const;
    double gaussian_shape(double r_rg, double theta) const;
    double powerlaw_shape(double r_rg, double theta) const;
    double funnel_factor(double theta) const;
    double envelope_rho(double r_rg) const;
    double collapsar_ndaf_shape(double r_rg, double theta) const;
    bool uses_powerlaw_disk() const;
    bool uses_funnel() const;
    bool uses_envelope() const;

    double rho0_;
    double r0_;
    double sigma_r_;
    double H_over_R_;
    std::string profile_name_;
    DensityProfileKind profile_kind_;
    double radial_power_;
    double funnel_depletion_;
    double funnel_theta_rad_;
    double envelope_rho0_;
    double envelope_alpha_;
    double r_min_;
    double r_max_;
    double rho_floor_;
};

#endif
