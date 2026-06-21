#include "radiative_transfer.hpp"
#include "constants.hpp"
#include "mev_neutrino_physics.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace radiative_transfer {

namespace {

mev_neutrino::MeVPhysicsParams make_mev_physics_params(const MeVThermalParams& mev)
{
    mev_neutrino::MeVPhysicsParams params;
    params.model = mev.model;
    params.model_name = mev.model_name;
    params.flavor = mev.flavor;
    params.flavor_name = mev.flavor_name;
    params.include_urca = mev.include_urca;
    params.include_pair = mev.include_pair;
    params.include_brems = mev.include_brems;
    params.include_absorption = mev.include_absorption;
    params.include_scattering = mev.include_scattering;
    params.use_degeneracy_correction = mev.use_degeneracy_correction;
    params.include_abs_n = mev.include_abs_n;
    params.include_abs_p = mev.include_abs_p;
    params.include_scat_n = mev.include_scat_n;
    params.include_scat_p = mev.include_scat_p;
    params.include_scat_e = mev.include_scat_e;
    params.norm = mev.norm;
    params.sigma_abs0_cm2 = mev.sigma_abs0_cm2;
    params.sigma_scat0_cm2 = mev.sigma_scat0_cm2;
    params.thermal_profile = mev.thermal_profile;
    params.thermal_profile_name = mev.thermal_profile_name;
    params.ye_profile = mev.ye_profile;
    params.ye_profile_name = mev.ye_profile_name;
    params.spectral_mode = mev.spectral_mode;
    params.spectral_mode_name = mev.spectral_mode_name;
    params.T0_MeV = mev.T0_MeV;
    params.T_floor_MeV = mev.T_floor_MeV;
    params.T_power = mev.T_power;
    params.Ye_torus = mev.Ye_torus;
    params.Ye_funnel = mev.Ye_funnel;
    params.Ye_envelope = mev.Ye_envelope;
    params.Ye_floor = mev.Ye_floor;
    params.Ye_ceil = mev.Ye_ceil;
    params.E_min_MeV = mev.E_min_MeV;
    params.E_max_MeV = mev.E_max_MeV;
    params.n_bins = mev.n_bins;
    return params;
}

double spectral_uhe(
    double Enu_local_GeV,
    const UHESourceParams& source,
    const UHESpectralParams& spectral
)
{
    return uhe_spectral_weight(Enu_local_GeV, source, spectral);
}

double bipolar_gaussian(double theta, double theta0, double width)
{
    if (width <= 0.0) {
        return 0.0;
    }

    const double north = theta - theta0;
    const double south = theta - (constants::pi - theta0);

    return
        std::exp(-(north / width) * (north / width))
        + std::exp(-(south / width) * (south / width));
}

}

UHESourceModel parse_uhe_source_model(const std::string& model_name)
{
    if (model_name == "inner_ring" ||
        model_name == "INNER_RING" ||
        model_name == "ring") {
        return UHESourceModel::InnerRing;
    }

    if (model_name == "funnel_wall" ||
        model_name == "FUNNEL_WALL") {
        return UHESourceModel::FunnelWall;
    }

    if (model_name == "jet_base" ||
        model_name == "JET_BASE") {
        return UHESourceModel::JetBase;
    }

    if (model_name == "shock_layer" ||
        model_name == "SHOCK_LAYER") {
        return UHESourceModel::ShockLayer;
    }

    if (model_name == "density_weighted" ||
        model_name == "DENSITY_WEIGHTED") {
        return UHESourceModel::DensityWeighted;
    }

    throw std::runtime_error(
        "Unknown UHE source model '" + model_name +
        "'. Use inner_ring, funnel_wall, jet_base, shock_layer, or density_weighted."
    );
}

const char* uhe_source_model_name(UHESourceModel model)
{
    switch (model) {
        case UHESourceModel::InnerRing:
            return "inner_ring";
        case UHESourceModel::FunnelWall:
            return "funnel_wall";
        case UHESourceModel::JetBase:
            return "jet_base";
        case UHESourceModel::ShockLayer:
            return "shock_layer";
        case UHESourceModel::DensityWeighted:
            return "density_weighted";
    }

    return "inner_ring";
}

UHESpectralModel parse_uhe_spectral_model(const std::string& model_name)
{
    if (model_name == "monochromatic" ||
        model_name == "MONOCHROMATIC" ||
        model_name == "mono") {
        return UHESpectralModel::Monochromatic;
    }

    if (model_name == "powerlaw" ||
        model_name == "POWERLAW" ||
        model_name == "power_law") {
        return UHESpectralModel::PowerLaw;
    }

    if (model_name == "powerlaw_cutoff" ||
        model_name == "POWERLAW_CUTOFF" ||
        model_name == "power_law_cutoff") {
        return UHESpectralModel::PowerLawCutoff;
    }

    throw std::runtime_error(
        "Unknown UHE spectral model '" + model_name +
        "'. Use monochromatic, powerlaw, or powerlaw_cutoff."
    );
}

const char* uhe_spectral_model_name(UHESpectralModel model)
{
    switch (model) {
        case UHESpectralModel::Monochromatic:
            return "monochromatic";
        case UHESpectralModel::PowerLaw:
            return "powerlaw";
        case UHESpectralModel::PowerLawCutoff:
            return "powerlaw_cutoff";
    }

    return "monochromatic";
}

double uhe_spectral_weight(
    double E_GeV,
    const UHESourceParams& source,
    const UHESpectralParams& spectral
)
{
    if (E_GeV <= 0.0) {
        return 0.0;
    }

    const double safe_E = std::max(E_GeV, 1.0e-300);

    if (spectral.model == UHESpectralModel::Monochromatic) {
        // Backward-compatible single-energy behavior: preserve the legacy
        // source spectral factor used by previous image calculations.
        if (source.emax_GeV <= 0.0) {
            return 0.0;
        }
        return std::pow(safe_E, -source.powerlaw)
            * std::exp(-safe_E / source.emax_GeV);
    }

    if (spectral.model == UHESpectralModel::PowerLaw) {
        return std::pow(safe_E, -spectral.gamma);
    }

    if (spectral.model == UHESpectralModel::PowerLawCutoff) {
        if (spectral.ecut_GeV <= 0.0) {
            return 0.0;
        }
        return std::pow(safe_E, -spectral.gamma)
            * std::exp(-safe_E / spectral.ecut_GeV);
    }

    return 0.0;
}

double uhe_source_spatial_weight(
    double r_rg,
    double theta,
    const TorusProfile& torus,
    const UHESourceParams& source
)
{
    if (r_rg <= 0.0 || source.sigma_r_rg <= 0.0 ||
        source.theta_width_rad <= 0.0) {
        return 0.0;
    }

    if (source.model == UHESourceModel::InnerRing) {
        const double delta_r =
            (r_rg - source.r_center_rg) / source.sigma_r_rg;

        const double delta_theta =
            (theta - 0.5 * constants::pi) / source.theta_width_rad;

        return std::exp(-delta_r * delta_r)
            * std::exp(-delta_theta * delta_theta);
    }

    if (source.model == UHESourceModel::FunnelWall) {
        const double delta_r =
            (r_rg - source.r_center_rg) / source.sigma_r_rg;

        return std::exp(-delta_r * delta_r)
            * bipolar_gaussian(
                theta,
                source.funnel_theta_rad,
                source.theta_width_rad
            );
    }

    if (source.model == UHESourceModel::JetBase) {
        const double radial =
            std::exp(-std::pow(r_rg / source.r_center_rg, 2.0));

        const double polar =
            bipolar_gaussian(theta, 0.0, source.theta_width_rad);

        return radial * polar;
    }

    if (source.model == UHESourceModel::DensityWeighted) {
        const double rho =
            torus.rho(r_rg, theta);

        const double rho_norm =
            std::max(
                source.rho_ref_gcm3 > 0.0
                ? source.rho_ref_gcm3
                : torus.rho(source.r_center_rg, 0.5 * constants::pi),
                1.0e-300
            );

        const double density_weight =
            std::pow(std::max(rho / rho_norm, 0.0), source.density_power_q);

        const double radial_weight =
            std::pow(
                std::max(r_rg / std::max(source.r_center_rg, 1.0e-300), 1.0e-300),
                -source.radial_power_s
            );

        return std::clamp(
            density_weight * radial_weight,
            std::max(source.cutoff_min, 0.0),
            std::max(source.cutoff_max, std::max(source.cutoff_min, 0.0))
        );
    }

    if (source.model == UHESourceModel::ShockLayer) {
        const double dr =
            std::max(source.gradient_dr_rg, 1.0e-4);

        const double dtheta =
            std::max(source.gradient_dtheta_rad, 1.0e-5);

        const double r_minus =
            std::max(r_rg - dr, 1.0e-6);

        const double theta_minus =
            std::clamp(theta - dtheta, 0.0, constants::pi);

        const double theta_plus =
            std::clamp(theta + dtheta, 0.0, constants::pi);

        const double rho_r_plus =
            torus.rho(r_rg + dr, theta);

        const double rho_r_minus =
            torus.rho(r_minus, theta);

        const double rho_t_plus =
            torus.rho(r_rg, theta_plus);

        const double rho_t_minus =
            torus.rho(r_rg, theta_minus);

        const double rho_norm =
            std::max(torus.rho(source.r_center_rg, 0.5 * constants::pi), 1.0e-300);

        const double d_r =
            (rho_r_plus - rho_r_minus) / (2.0 * dr);

        const double d_theta =
            (rho_t_plus - rho_t_minus) / (2.0 * dtheta * std::max(r_rg, 1.0e-6));

        const double gradient =
            std::sqrt(d_r * d_r + d_theta * d_theta) / rho_norm;

        const double radial_taper =
            std::exp(-std::pow(r_rg / std::max(source.r_center_rg, 1.0e-300), 2.0));

        return std::clamp(
            gradient * radial_taper,
            std::max(source.cutoff_min, 0.0),
            std::max(source.cutoff_max, std::max(source.cutoff_min, 0.0))
        );
    }

    return 0.0;
}

double emissivity_collapsar_ring(
    double r_rg,
    double theta,
    double Enu_local_GeV,
    const UHESourceParams& source,
    const UHESpectralParams& spectral
)
{
    if (Enu_local_GeV <= 0.0 ||
        source.r_center_rg <= 0.0 ||
        source.sigma_r_rg <= 0.0 ||
        source.theta_width_rad <= 0.0 ||
        source.emax_GeV <= 0.0) {
        return 0.0;
    }

    const double delta_r =
        (r_rg - source.r_center_rg) / source.sigma_r_rg;

    const double delta_theta =
        (theta - 0.5 * constants::pi) / source.theta_width_rad;

    const double radial_profile =
        std::exp(-delta_r * delta_r);

    const double vertical_profile =
        std::exp(-delta_theta * delta_theta);

    const double spectral_profile =
        spectral_uhe(Enu_local_GeV, source, spectral);

    return source.norm * radial_profile * vertical_profile * spectral_profile;
}

double emissivity_uhe(
    double r_rg,
    double theta,
    double Enu_local_GeV,
    const TorusProfile& torus,
    const UHESourceParams& source,
    const UHESpectralParams& spectral
)
{
    if (Enu_local_GeV <= 0.0) {
        return 0.0;
    }

    if (source.model == UHESourceModel::InnerRing) {
        return emissivity_collapsar_ring(
            r_rg,
            theta,
            Enu_local_GeV,
            source,
            spectral
        );
    }

    return source.norm
        * uhe_source_spatial_weight(r_rg, theta, torus, source)
        * spectral_uhe(Enu_local_GeV, source, spectral);
}

double emissivity_mev_thermal(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double Enu_local_MeV,
    const MeVThermalParams& mev
)
{
    const mev_neutrino::MeVPhysicsParams params =
        make_mev_physics_params(mev);

    return mev_neutrino::mev_total_emissivity(
        rho_gcm3,
        T_MeV,
        Ye,
        Enu_local_MeV,
        params
    );
}

double opacity_mev_absorption_cm_inv(
    double rho_gcm3,
    double Ye,
    double Enu_local_MeV,
    const MeVThermalParams& mev
)
{
    mev_neutrino::MeVPhysicsParams params =
        make_mev_physics_params(mev);

    if (mev.model == mev_neutrino::MeVModel::Toy) {
        params.flavor = mev_neutrino::MeVFlavor::AntiNuE;
    }

    return mev_neutrino::mev_opacity_absorption_cm_inv(
        rho_gcm3,
        Ye,
        Enu_local_MeV,
        params
    );
}

double opacity_mev_scattering_cm_inv(
    double rho_gcm3,
    double Ye,
    double Enu_local_MeV,
    const MeVThermalParams& mev
)
{
    const mev_neutrino::MeVPhysicsParams params =
        make_mev_physics_params(mev);

    return mev_neutrino::mev_opacity_scattering_cm_inv(
        rho_gcm3,
        Ye,
        Enu_local_MeV,
        params
    );
}

RTResult integrate_kerr_ray(
    const RayPath& ray,
    double Enu_obs_GeV,
    double M_bh_msun,
    const TorusProfile& torus,
    const SigmaTable& sigma,
    UHECircularSourceParams source,
    MeVThermalParams mev,
    UHESpectralParams spectral,
    bool compute_mev
)
{
    RTResult result;

    const double rg_cm =
        constants::rg_cm(M_bh_msun);

    /*
        Horizon radius in units of r_g = GM/c^2.

        For Kerr:
            r_+ / r_g = 1 + sqrt(1 - a^2)

        Here I assume that ray.a_bh stores the dimensionless spin.
        If your RayPath does not have a_bh, replace this line by the
        same value used in the geodesic calculation, for example:

            const double a_bh = 0.95;
    */
    const double a_bh =
        std::clamp(ray.a_bh, 0.0, 0.999999);

    const double r_h =
        1.0 + std::sqrt(1.0 - a_bh * a_bh);

    double tau = 0.0;
    double Iobs = 0.0;
    double tau_mev = 0.0;
    double Iobs_mev = 0.0;
    double leakage_sum = 0.0;
    double leakage_weight = 0.0;
    double r_neutrinosphere = -1.0;
    const mev_neutrino::MeVPhysicsParams mev_physics =
        make_mev_physics_params(mev);

    for (const auto& p : ray.points) {

        if (p.r_rg < r_h + 1.0e-3) {
            continue;
        }

        const double g =
            1.0 / std::max(p.redshift_factor, 1.0e-300);

        const double Enu_local_GeV =
            Enu_obs_GeV / std::max(g, 1.0e-300);

        const double dl_cm =
            p.dl_rg * rg_cm;

        const double rho =
            torus.rho(p.r_rg, p.theta);

        if (compute_mev) {
            const double Enu_local_MeV =
                mev.Enu_obs_MeV / std::max(g, 1.0e-300);

            const double T_MeV =
                mev_neutrino::mev_temperature_profile_MeV(
                    p.r_rg,
                    p.theta,
                    rho,
                    torus.rho0_gcm3(),
                    torus.r0_rg(),
                    torus.r_min_rg(),
                    torus.r_max_rg(),
                    mev_physics
                );

            const double Ye =
                mev_neutrino::mev_ye_profile(
                    p.r_rg,
                    p.theta,
                    rho,
                    torus.rho0_gcm3(),
                    torus.r0_rg(),
                    torus.r_min_rg(),
                    torus.r_max_rg(),
                    mev_physics
                );

            double kappa_mev = 0.0;
            double j_mev = 0.0;

            if (mev_physics.spectral_mode == mev_neutrino::MeVSpectralMode::FermiDiracBand) {
                const int n_bins = std::max(mev_physics.n_bins, 1);
                const double e_min = std::max(mev_physics.E_min_MeV, 1.0e-6);
                const double e_max = std::max(mev_physics.E_max_MeV, e_min);
                const double dE = (e_max - e_min) / static_cast<double>(n_bins);
                double weight_sum = 0.0;
                double kappa_weighted = 0.0;

                for (int ib = 0; ib < n_bins; ++ib) {
                    const double E_mid =
                        e_min + (static_cast<double>(ib) + 0.5) * dE;
                    const double E_local =
                        E_mid / std::max(g, 1.0e-300);
                    const double w =
                        mev_neutrino::mev_fermi_dirac_weight(E_local, T_MeV)
                        * dE;

                    if (w <= 0.0 || !std::isfinite(w)) {
                        continue;
                    }

                    const double local_kappa =
                        mev_neutrino::mev_total_opacity_cm_inv(
                            rho,
                            Ye,
                            E_local,
                            mev_physics
                        );

                    j_mev +=
                        mev_neutrino::mev_total_emissivity(
                            rho,
                            T_MeV,
                            Ye,
                            E_local,
                            mev_physics
                        ) * dE;
                    kappa_weighted += w * local_kappa;
                    weight_sum += w;
                }

                kappa_mev =
                    weight_sum > 0.0 ? kappa_weighted / weight_sum : 0.0;
            } else {
                kappa_mev =
                    mev_neutrino::mev_total_opacity_cm_inv(
                        rho,
                        Ye,
                        Enu_local_MeV,
                        mev_physics
                    );

                j_mev =
                    mev_neutrino::mev_total_emissivity(
                        rho,
                        T_MeV,
                        Ye,
                        Enu_local_MeV,
                        mev_physics
                    );
            }

            const double dtau_mev =
                kappa_mev * dl_cm;

            if (kappa_mev > 0.0 && dtau_mev > 1.0e-12) {
                const double attenuation =
                    std::exp(-std::min(dtau_mev, 700.0));

                Iobs_mev =
                    Iobs_mev * attenuation
                    + std::pow(g, 3.0)
                    * (j_mev / kappa_mev)
                    * (1.0 - attenuation);
            } else {
                Iobs_mev +=
                    std::pow(g, 3.0)
                    * j_mev
                    * dl_cm;
            }

            leakage_sum += j_mev * dl_cm;
            leakage_weight += j_mev * dl_cm;

            tau_mev += dtau_mev;

            if (r_neutrinosphere < 0.0 &&
                tau_mev >= mev.neutrinosphere_tau) {
                r_neutrinosphere = p.r_rg;
            }
        }

        const double j =
            emissivity_uhe(
                p.r_rg,
                p.theta,
                Enu_local_GeV,
                torus,
                source,
                spectral
            );

        Iobs +=
            std::pow(g, 3.0)
            * j
            * std::exp(-tau)
            * dl_cm;

        if (rho <= 0.0 ||
            Enu_local_GeV < sigma.Emin() ||
            Enu_local_GeV > sigma.Emax()) {
            continue;
        }

        const double sigma_cm2 =
            sigma.sigma_cm2(Enu_local_GeV);

        const double nb =
            rho / constants::m_u_g;

        const double dtau =
            nb * sigma_cm2 * dl_cm;

        tau += dtau;
    }

    result.tau = tau;
    result.P_surv = std::exp(-tau);
    result.I_obs = Iobs;
    result.tau_mev = tau_mev;
    result.P_surv_mev = std::exp(-tau_mev);
    result.I_obs_mev = Iobs_mev;
    result.r_neutrinosphere_rg = r_neutrinosphere;
    result.leakage_factor =
        leakage_weight > 0.0 ? leakage_sum / leakage_weight : 1.0;

    return result;
}

KerrRTAccumulator::KerrRTAccumulator(
    double a_spin,
    double Enu_obs_GeV,
    double M_bh_msun,
    const TorusProfile& torus,
    const SigmaTable& sigma,
    UHESourceParams source,
    MeVThermalParams mev,
    UHESpectralParams spectral,
    bool compute_mev
)
    : a_spin_(std::clamp(a_spin, 0.0, 0.999999)),
      Enu_obs_GeV_(Enu_obs_GeV),
      rg_cm_(constants::rg_cm(M_bh_msun)),
      torus_(torus),
      sigma_(sigma),
      source_(source),
      mev_(mev),
      spectral_(spectral),
      compute_mev_(compute_mev),
      r_h_(1.0 + std::sqrt(1.0 - a_spin_ * a_spin_))
{
}

void KerrRTAccumulator::add_point(const PathPoint& p)
{
    if (p.r_rg < r_h_ + 1.0e-3) {
        return;
    }

    const double g =
        1.0 / std::max(p.redshift_factor, 1.0e-300);

    const double Enu_local_GeV =
        Enu_obs_GeV_ / std::max(g, 1.0e-300);

    const double dl_cm =
        p.dl_rg * rg_cm_;

    const double rho =
        torus_.rho(p.r_rg, p.theta);

    const mev_neutrino::MeVPhysicsParams mev_physics =
        make_mev_physics_params(mev_);

    if (compute_mev_ && mev_active_) {
        const double Enu_local_MeV =
            mev_.Enu_obs_MeV / std::max(g, 1.0e-300);

        const double T_MeV =
            mev_neutrino::mev_temperature_profile_MeV(
                p.r_rg,
                p.theta,
                rho,
                torus_.rho0_gcm3(),
                torus_.r0_rg(),
                torus_.r_min_rg(),
                torus_.r_max_rg(),
                mev_physics
            );

        const double Ye =
            mev_neutrino::mev_ye_profile(
                p.r_rg,
                p.theta,
                rho,
                torus_.rho0_gcm3(),
                torus_.r0_rg(),
                torus_.r_min_rg(),
                torus_.r_max_rg(),
                mev_physics
            );

        double kappa_mev = 0.0;
        double j_mev = 0.0;

        if (mev_physics.spectral_mode == mev_neutrino::MeVSpectralMode::FermiDiracBand) {
            const int n_bins = std::max(mev_physics.n_bins, 1);
            const double e_min = std::max(mev_physics.E_min_MeV, 1.0e-6);
            const double e_max = std::max(mev_physics.E_max_MeV, e_min);
            const double dE = (e_max - e_min) / static_cast<double>(n_bins);
            double weight_sum = 0.0;
            double kappa_weighted = 0.0;
            for (int ib = 0; ib < n_bins; ++ib) {
                const double E_mid = e_min + (static_cast<double>(ib) + 0.5) * dE;
                const double E_local = E_mid / std::max(g, 1.0e-300);
                const double w = mev_neutrino::mev_fermi_dirac_weight(E_local, T_MeV) * dE;
                if (w <= 0.0 || !std::isfinite(w)) continue;
                const double local_kappa =
                    mev_neutrino::mev_total_opacity_cm_inv(rho, Ye, E_local, mev_physics);
                j_mev += mev_neutrino::mev_total_emissivity(rho, T_MeV, Ye, E_local, mev_physics) * dE;
                kappa_weighted += w * local_kappa;
                weight_sum += w;
            }
            kappa_mev = weight_sum > 0.0 ? kappa_weighted / weight_sum : 0.0;
        } else {
            kappa_mev = mev_neutrino::mev_total_opacity_cm_inv(rho, Ye, Enu_local_MeV, mev_physics);
            j_mev = mev_neutrino::mev_total_emissivity(rho, T_MeV, Ye, Enu_local_MeV, mev_physics);
        }

        const double dtau_mev = kappa_mev * dl_cm;
        if (kappa_mev > 0.0 && dtau_mev > 1.0e-12) {
            const double attenuation = std::exp(-std::min(dtau_mev, 700.0));
            Iobs_mev_ = Iobs_mev_ * attenuation
                + std::pow(g, 3.0) * (j_mev / kappa_mev) * (1.0 - attenuation);
        } else {
            Iobs_mev_ += std::pow(g, 3.0) * j_mev * dl_cm;
        }
        leakage_sum_ += j_mev * dl_cm;
        leakage_weight_ += j_mev * dl_cm;
        tau_mev_ += dtau_mev;
        if (r_neutrinosphere_ < 0.0 && tau_mev_ >= mev_.neutrinosphere_tau) {
            r_neutrinosphere_ = p.r_rg;
        }
    }

    if (!uhe_active_) {
        return;
    }

    const double j =
        emissivity_uhe(p.r_rg, p.theta, Enu_local_GeV, torus_, source_, spectral_);

    Iobs_ += std::pow(g, 3.0) * j * std::exp(-tau_) * dl_cm;

    if (rho <= 0.0 ||
        Enu_local_GeV < sigma_.Emin() ||
        Enu_local_GeV > sigma_.Emax()) {
        return;
    }

    const double sigma_cm2 = sigma_.sigma_cm2(Enu_local_GeV);
    const double nb = rho / constants::m_u_g;
    tau_ += nb * sigma_cm2 * dl_cm;
}

void KerrRTAccumulator::set_uhe_active(bool active)
{
    uhe_active_ = active;
}

void KerrRTAccumulator::set_mev_active(bool active)
{
    mev_active_ = active;
}

bool KerrRTAccumulator::uhe_active() const
{
    return uhe_active_;
}

bool KerrRTAccumulator::mev_active() const
{
    return compute_mev_ && mev_active_;
}

RTResult KerrRTAccumulator::result() const
{
    RTResult result;
    result.tau = tau_;
    result.P_surv = std::exp(-tau_);
    result.I_obs = Iobs_;
    result.tau_mev = tau_mev_;
    result.P_surv_mev = std::exp(-tau_mev_);
    result.I_obs_mev = Iobs_mev_;
    result.r_neutrinosphere_rg = r_neutrinosphere_;
    result.leakage_factor =
        leakage_weight_ > 0.0 ? leakage_sum_ / leakage_weight_ : 1.0;
    return result;
}

KerrSpectralRTAccumulator::KerrSpectralRTAccumulator(
    double a_spin,
    double Enu_obs_GeV,
    double M_bh_msun,
    const TorusProfile& torus,
    const SigmaTable& sigma,
    UHESourceParams source,
    MeVThermalParams mev,
    UHESpectralParams spectral
)
{
    if (spectral.model == UHESpectralModel::Monochromatic) {
        dE_.push_back(1.0);
        weights_.push_back(1.0);
        accumulators_.push_back(std::make_unique<KerrRTAccumulator>(
            a_spin,
            Enu_obs_GeV,
            M_bh_msun,
            torus,
            sigma,
            source,
            mev,
            spectral,
            true
        ));
        return;
    }

    const int n_bins = std::max(spectral.n_bins, 1);
    const double e_min = std::max(spectral.e_min_GeV, sigma.Emin());
    const double e_max = std::min(spectral.e_max_GeV, sigma.Emax());

    if (e_min <= 0.0 || e_max <= e_min) {
        return;
    }

    const double log_min = std::log(e_min);
    const double log_max = std::log(e_max);
    const double dlog = (log_max - log_min) / static_cast<double>(n_bins);
    bool have_mev_reference = false;

    for (int i = 0; i < n_bins; ++i) {
        const double log_left = log_min + dlog * static_cast<double>(i);
        const double log_right = log_left + dlog;
        const double E_mid = std::exp(0.5 * (log_left + log_right));
        const double dE = std::exp(log_right) - std::exp(log_left);
        const double w = uhe_spectral_weight(E_mid, source, spectral) * dE;

        if (w <= 0.0 || !std::isfinite(w)) {
            continue;
        }

        dE_.push_back(dE);
        weights_.push_back(w);
        accumulators_.push_back(std::make_unique<KerrRTAccumulator>(
            a_spin,
            E_mid,
            M_bh_msun,
            torus,
            sigma,
            source,
            mev,
            spectral,
            !have_mev_reference
        ));
        have_mev_reference = true;
    }
}

void KerrSpectralRTAccumulator::add_point(const PathPoint& p)
{
    for (const auto& acc : accumulators_) {
        acc->add_point(p);
    }
}

void KerrSpectralRTAccumulator::set_uhe_active(bool active)
{
    for (const auto& acc : accumulators_) {
        acc->set_uhe_active(active);
    }
}

void KerrSpectralRTAccumulator::set_mev_active(bool active)
{
    for (const auto& acc : accumulators_) {
        acc->set_mev_active(active);
    }
}

bool KerrSpectralRTAccumulator::uhe_active() const
{
    for (const auto& acc : accumulators_) {
        if (acc->uhe_active()) {
            return true;
        }
    }
    return false;
}

bool KerrSpectralRTAccumulator::mev_active() const
{
    for (const auto& acc : accumulators_) {
        if (acc->mev_active()) {
            return true;
        }
    }
    return false;
}

RTResult KerrSpectralRTAccumulator::result() const
{
    RTResult result;

    if (accumulators_.empty()) {
        return result;
    }

    if (accumulators_.size() == 1 && weights_.size() == 1 && dE_.size() == 1 && weights_[0] == 1.0 && dE_[0] == 1.0) {
        return accumulators_[0]->result();
    }

    double weight_sum = 0.0;
    double tau_sum = 0.0;
    double psurv_sum = 0.0;
    double intensity_sum = 0.0;

    for (std::size_t i = 0; i < accumulators_.size(); ++i) {
        const RTResult rt = accumulators_[i]->result();
        weight_sum += weights_[i];
        tau_sum += weights_[i] * rt.tau;
        psurv_sum += weights_[i] * rt.P_surv;
        intensity_sum += rt.I_obs * dE_[i];
    }

    if (weight_sum > 0.0) {
        result.tau = tau_sum / weight_sum;
        result.P_surv = psurv_sum / weight_sum;
        result.I_obs = intensity_sum;
    }

    const RTResult mev_reference = accumulators_[0]->result();
    result.tau_mev = mev_reference.tau_mev;
    result.P_surv_mev = mev_reference.P_surv_mev;
    result.I_obs_mev = mev_reference.I_obs_mev;
    result.r_neutrinosphere_rg = mev_reference.r_neutrinosphere_rg;
    result.leakage_factor = mev_reference.leakage_factor;
    return result;
}

RTResult integrate_kerr_ray_spectral(
    const RayPath& ray,
    double Enu_obs_GeV,
    double M_bh_msun,
    const TorusProfile& torus,
    const SigmaTable& sigma,
    UHECircularSourceParams source,
    MeVThermalParams mev,
    UHESpectralParams spectral
)
{
    if (spectral.model == UHESpectralModel::Monochromatic) {
        return integrate_kerr_ray(
            ray,
            Enu_obs_GeV,
            M_bh_msun,
            torus,
            sigma,
            source,
            mev,
            spectral,
            true
        );
    }

    const int n_bins = std::max(spectral.n_bins, 1);
    const double e_min = std::max(spectral.e_min_GeV, sigma.Emin());
    const double e_max = std::min(spectral.e_max_GeV, sigma.Emax());

    if (e_min <= 0.0 || e_max <= e_min) {
        return RTResult{};
    }

    double weight_sum = 0.0;
    double tau_sum = 0.0;
    double psurv_sum = 0.0;
    double intensity_sum = 0.0;
    RTResult mev_reference;
    bool have_reference = false;

    const double log_min = std::log(e_min);
    const double log_max = std::log(e_max);
    const double dlog = (log_max - log_min) / static_cast<double>(n_bins);

    for (int i = 0; i < n_bins; ++i) {
        const double log_left = log_min + dlog * static_cast<double>(i);
        const double log_right = log_left + dlog;
        const double E_mid = std::exp(0.5 * (log_left + log_right));
        const double dE = std::exp(log_right) - std::exp(log_left);
        const double w = uhe_spectral_weight(E_mid, source, spectral) * dE;

        if (w <= 0.0 || !std::isfinite(w)) {
            continue;
        }

        const RTResult rt = integrate_kerr_ray(
            ray,
            E_mid,
            M_bh_msun,
            torus,
            sigma,
            source,
            mev,
            spectral,
            !have_reference
        );

        weight_sum += w;
        tau_sum += w * rt.tau;
        psurv_sum += w * rt.P_surv;
        intensity_sum += rt.I_obs * dE;

        if (!have_reference) {
            mev_reference = rt;
            have_reference = true;
        }
    }

    RTResult result;

    if (weight_sum > 0.0) {
        result.tau = tau_sum / weight_sum;
        result.P_surv = psurv_sum / weight_sum;
        result.I_obs = intensity_sum;
    }

    if (have_reference) {
        result.tau_mev = mev_reference.tau_mev;
        result.P_surv_mev = mev_reference.P_surv_mev;
        result.I_obs_mev = mev_reference.I_obs_mev;
        result.r_neutrinosphere_rg = mev_reference.r_neutrinosphere_rg;
        result.leakage_factor = mev_reference.leakage_factor;
    }

    return result;
}

}
