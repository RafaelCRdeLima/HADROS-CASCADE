#include "mev_neutrino_physics.hpp"
#include "constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mev_neutrino {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double HBAR_C_MEV_CM = 1.973269804e-11;
constexpr double M_E_MEV = 0.510998950;

double clamp_ye(double Ye)
{
    return std::clamp(Ye, 0.0, 1.0);
}

double safe_pow(double x, double p)
{
    return std::pow(std::max(x, 0.0), p);
}

double density_shape(double rho_gcm3, double rho0_gcm3)
{
    return std::clamp(
        rho_gcm3 / std::max(rho0_gcm3, 1.0e-300),
        0.0,
        1.0
    );
}

}

MeVModel parse_mev_model(const std::string& model_name)
{
    if (model_name == "toy" || model_name == "TOY") {
        return MeVModel::Toy;
    }

    if (model_name == "physical" ||
        model_name == "PHYSICAL" ||
        model_name == "phys") {
        return MeVModel::Physical;
    }

    throw std::runtime_error(
        "Unknown MEV_MODEL '" + model_name + "'. Use toy or physical."
    );
}

const char* mev_model_name(MeVModel model)
{
    switch (model) {
        case MeVModel::Toy:
            return "toy";
        case MeVModel::Physical:
            return "physical";
    }

    return "physical";
}

MeVFlavor parse_mev_flavor(const std::string& flavor_name)
{
    if (flavor_name == "nu_e" ||
        flavor_name == "nue" ||
        flavor_name == "NU_E") {
        return MeVFlavor::NuE;
    }

    if (flavor_name == "anti_nu_e" ||
        flavor_name == "antinue" ||
        flavor_name == "anti-nu-e" ||
        flavor_name == "ANTI_NU_E") {
        return MeVFlavor::AntiNuE;
    }

    if (flavor_name == "nu_x" ||
        flavor_name == "nux" ||
        flavor_name == "NU_X") {
        return MeVFlavor::NuX;
    }

    throw std::runtime_error(
        "Unknown MEV_FLAVOR '" + flavor_name +
        "'. Use nu_e, anti_nu_e, or nu_x."
    );
}

const char* mev_flavor_name(MeVFlavor flavor)
{
    switch (flavor) {
        case MeVFlavor::NuE:
            return "nu_e";
        case MeVFlavor::AntiNuE:
            return "anti_nu_e";
        case MeVFlavor::NuX:
            return "nu_x";
    }

    return "anti_nu_e";
}

MeVThermalProfile parse_mev_thermal_profile(const std::string& profile_name)
{
    if (profile_name == "constant" || profile_name == "CONSTANT") {
        return MeVThermalProfile::Constant;
    }
    if (profile_name == "inner_hot_torus" || profile_name == "INNER_HOT_TORUS") {
        return MeVThermalProfile::InnerHotTorus;
    }
    if (profile_name == "radial_powerlaw" || profile_name == "RADIAL_POWERLAW") {
        return MeVThermalProfile::RadialPowerLaw;
    }
    if (profile_name == "torus_plus_cool_envelope" ||
        profile_name == "TORUS_PLUS_COOL_ENVELOPE") {
        return MeVThermalProfile::TorusPlusCoolEnvelope;
    }
    if (profile_name == "collapsar_inner_hot" ||
        profile_name == "COLLAPSAR_INNER_HOT") {
        return MeVThermalProfile::CollapsarInnerHot;
    }
    throw std::runtime_error(
        "Unknown MEV_THERMAL_PROFILE '" + profile_name +
        "'. Use constant, inner_hot_torus, radial_powerlaw, "
        "torus_plus_cool_envelope, or collapsar_inner_hot."
    );
}

const char* mev_thermal_profile_name(MeVThermalProfile profile)
{
    switch (profile) {
        case MeVThermalProfile::Constant:
            return "constant";
        case MeVThermalProfile::InnerHotTorus:
            return "inner_hot_torus";
        case MeVThermalProfile::RadialPowerLaw:
            return "radial_powerlaw";
        case MeVThermalProfile::TorusPlusCoolEnvelope:
            return "torus_plus_cool_envelope";
        case MeVThermalProfile::CollapsarInnerHot:
            return "collapsar_inner_hot";
    }
    return "inner_hot_torus";
}

MeVYeProfile parse_mev_ye_profile(const std::string& profile_name)
{
    if (profile_name == "constant" || profile_name == "CONSTANT") {
        return MeVYeProfile::Constant;
    }
    if (profile_name == "neutron_rich_torus" || profile_name == "NEUTRON_RICH_TORUS") {
        return MeVYeProfile::NeutronRichTorus;
    }
    if (profile_name == "funnel_proton_rich" || profile_name == "FUNNEL_PROTON_RICH") {
        return MeVYeProfile::FunnelProtonRich;
    }
    if (profile_name == "torus_envelope_contrast" ||
        profile_name == "TORUS_ENVELOPE_CONTRAST") {
        return MeVYeProfile::TorusEnvelopeContrast;
    }
    if (profile_name == "collapsar_neutron_rich" ||
        profile_name == "COLLAPSAR_NEUTRON_RICH") {
        return MeVYeProfile::CollapsarNeutronRich;
    }
    throw std::runtime_error(
        "Unknown MEV_YE_PROFILE '" + profile_name +
        "'. Use constant, neutron_rich_torus, funnel_proton_rich, "
        "torus_envelope_contrast, or collapsar_neutron_rich."
    );
}

const char* mev_ye_profile_name(MeVYeProfile profile)
{
    switch (profile) {
        case MeVYeProfile::Constant:
            return "constant";
        case MeVYeProfile::NeutronRichTorus:
            return "neutron_rich_torus";
        case MeVYeProfile::FunnelProtonRich:
            return "funnel_proton_rich";
        case MeVYeProfile::TorusEnvelopeContrast:
            return "torus_envelope_contrast";
        case MeVYeProfile::CollapsarNeutronRich:
            return "collapsar_neutron_rich";
    }
    return "neutron_rich_torus";
}

MeVSpectralMode parse_mev_spectral_mode(const std::string& mode_name)
{
    if (mode_name == "monochromatic" ||
        mode_name == "MONOCHROMATIC" ||
        mode_name == "mono") {
        return MeVSpectralMode::Monochromatic;
    }
    if (mode_name == "fermi_dirac_band" ||
        mode_name == "FERMI_DIRAC_BAND" ||
        mode_name == "fd_band") {
        return MeVSpectralMode::FermiDiracBand;
    }
    throw std::runtime_error(
        "Unknown MEV_SPECTRAL_MODE '" + mode_name +
        "'. Use monochromatic or fermi_dirac_band."
    );
}

const char* mev_spectral_mode_name(MeVSpectralMode mode)
{
    switch (mode) {
        case MeVSpectralMode::Monochromatic:
            return "monochromatic";
        case MeVSpectralMode::FermiDiracBand:
            return "fermi_dirac_band";
    }
    return "monochromatic";
}

double mev_temperature_profile_MeV(
    double r_rg,
    double theta,
    double rho_gcm3,
    double rho0_gcm3,
    double r0_rg,
    double r_min_rg,
    double r_max_rg,
    const MeVPhysicsParams& params
)
{
    const double T0 = std::max(params.T0_MeV, 1.0e-12);
    const double floor = std::max(params.T_floor_MeV, 1.0e-12);
    const double shape = density_shape(rho_gcm3, rho0_gcm3);
    const double radial =
        std::pow(
            std::max(r_rg / std::max(r0_rg, 1.0e-300), 1.0e-300),
            -params.T_power
        );
    const double equatorial =
        std::exp(-std::pow((theta - 0.5 * PI) / 0.45, 2.0));
    const bool outside_torus = r_rg > r_max_rg || r_rg < r_min_rg;

    double T = T0;
    if (params.thermal_profile == MeVThermalProfile::Constant) {
        T = T0;
    } else if (params.thermal_profile == MeVThermalProfile::InnerHotTorus) {
        T = floor + T0 * std::pow(shape, std::max(params.T_power, 0.0));
    } else if (params.thermal_profile == MeVThermalProfile::RadialPowerLaw) {
        T = floor + T0 * radial * (0.35 + 0.65 * equatorial);
    } else if (params.thermal_profile == MeVThermalProfile::TorusPlusCoolEnvelope) {
        T = floor + T0 * std::pow(shape, std::max(params.T_power, 0.0));
        if (outside_torus || shape < 1.0e-3) {
            T = floor + 0.15 * T0 * radial;
        }
    } else if (params.thermal_profile == MeVThermalProfile::CollapsarInnerHot) {
        const double inner_hot =
            std::exp(-std::pow((r_rg - r_min_rg) / std::max(1.15 * r0_rg, 1.0e-6), 2.0));
        const double radial_cooling =
            std::pow(
                1.0 + std::max(r_rg - r_min_rg, 0.0) / std::max(r0_rg, 1.0e-6),
                -0.75
            );
        T = floor
            + T0
                * (0.30 + 0.70 * std::pow(shape, 0.18))
                * (0.45 + 0.55 * equatorial)
                * (0.55 + 0.45 * inner_hot)
                * radial_cooling;
    }

    return std::max(T, floor);
}

double mev_ye_profile(
    double r_rg,
    double theta,
    double rho_gcm3,
    double rho0_gcm3,
    double r0_rg,
    double r_min_rg,
    double r_max_rg,
    const MeVPhysicsParams& params
)
{
    const double shape = density_shape(rho_gcm3, rho0_gcm3);
    const double polar =
        std::exp(-std::pow(theta / 0.35, 2.0))
        + std::exp(-std::pow((PI - theta) / 0.35, 2.0));
    const bool envelope_like = r_rg > r0_rg && (r_rg > r_max_rg || shape < 1.0e-2);

    double Ye = params.Ye_torus;
    if (params.ye_profile == MeVYeProfile::Constant) {
        Ye = params.Ye_torus;
    } else if (params.ye_profile == MeVYeProfile::NeutronRichTorus) {
        Ye = params.Ye_torus + 0.05 * std::exp(-std::pow((r_rg - r0_rg) / 4.0, 2.0));
    } else if (params.ye_profile == MeVYeProfile::FunnelProtonRich) {
        Ye = params.Ye_torus * (1.0 - std::clamp(polar, 0.0, 1.0))
            + params.Ye_funnel * std::clamp(polar, 0.0, 1.0);
    } else if (params.ye_profile == MeVYeProfile::TorusEnvelopeContrast) {
        Ye = envelope_like ? params.Ye_envelope : params.Ye_torus;
        if (r_rg < r_min_rg && shape < 1.0e-2) {
            Ye = params.Ye_funnel;
        }
    } else if (params.ye_profile == MeVYeProfile::CollapsarNeutronRich) {
        const double polar_weight = std::clamp(polar, 0.0, 1.0);
        const double dense_weight = std::pow(shape, 0.25);
        const double radial_neutronization =
            std::exp(-std::pow((r_rg - r0_rg) / std::max(1.2 * r0_rg, 1.0e-6), 2.0));
        const double torus_ye =
            params.Ye_torus
            - 0.05 * dense_weight * radial_neutronization;
        Ye = torus_ye * (1.0 - polar_weight)
            + params.Ye_funnel * polar_weight;
    }

    return std::clamp(
        Ye,
        std::max(params.Ye_floor, 1.0e-6),
        std::min(params.Ye_ceil, 1.0 - 1.0e-6)
    );
}

double mev_thermal_spectral_shape(
    double E_MeV,
    double T_MeV
)
{
    if (E_MeV <= 0.0 || T_MeV <= 0.0) {
        return 0.0;
    }

    const double x = E_MeV / T_MeV;

    if (x > 120.0) {
        return 0.0;
    }

    const double T3 = std::max(T_MeV * T_MeV * T_MeV, 1.0e-300);

    // Dimensionless thermal spectral shape used by the approximate local
    // spectral emissivity model. This is a controlled Fermi-Dirac-like proxy,
    // not a transport solution with degeneracy, blocking, or chemical
    // potentials.
    return E_MeV * E_MeV / (std::exp(x) + 1.0) / T3;
}

double mev_fermi_dirac_weight(
    double E_MeV,
    double T_MeV
)
{
    if (E_MeV <= 0.0 || T_MeV <= 0.0) {
        return 0.0;
    }
    const double x = E_MeV / T_MeV;
    if (x > 120.0) {
        return 0.0;
    }
    return E_MeV * E_MeV / (std::exp(x) + 1.0);
}

double electron_number_density_cm3(
    double rho_gcm3,
    double Ye
)
{
    if (rho_gcm3 <= 0.0) {
        return 0.0;
    }

    return rho_gcm3 * clamp_ye(Ye) / constants::m_u_g;
}

double electron_fermi_momentum_MeV(
    double rho_gcm3,
    double Ye
)
{
    const double ne = electron_number_density_cm3(rho_gcm3, Ye);
    if (ne <= 0.0) {
        return 0.0;
    }

    // Zero-temperature relativistic Fermi momentum proxy:
    // p_F = hbar c (3 pi^2 n_e)^(1/3). This diagnostic ignores finite-
    // temperature corrections to the electron chemical potential.
    return HBAR_C_MEV_CM * std::cbrt(3.0 * PI * PI * ne);
}

double electron_chemical_potential_MeV(
    double rho_gcm3,
    double Ye,
    double
)
{
    const double pF = electron_fermi_momentum_MeV(rho_gcm3, Ye);
    return std::sqrt(pF * pF + M_E_MEV * M_E_MEV);
}

double electron_degeneracy_eta(
    double rho_gcm3,
    double Ye,
    double T_MeV
)
{
    if (T_MeV <= 0.0) {
        return 0.0;
    }

    return electron_chemical_potential_MeV(rho_gcm3, Ye, T_MeV) / T_MeV;
}

double mev_urca_degeneracy_correction(
    double rho_gcm3,
    double Ye,
    double T_MeV,
    MeVFlavor flavor
)
{
    if (flavor == MeVFlavor::NuX) {
        return 1.0;
    }

    const double eta = electron_degeneracy_eta(rho_gcm3, Ye, T_MeV);
    if (!std::isfinite(eta)) {
        return 1.0;
    }

    // Bounded diagnostic correction. It is intentionally conservative because
    // the chemical potential uses a zero-temperature approximation and omits
    // blocking/threshold physics. Electron-rich degenerate zones modestly
    // enhance nu_e-like capture and suppress anti_nu_e-like capture.
    const double shift = 0.25 * std::tanh((eta - 1.0) / 4.0);
    const double factor =
        flavor == MeVFlavor::NuE ? (1.0 + shift) : (1.0 - shift);

    return std::clamp(factor, 0.5, 1.5);
}

double mev_emissivity_urca(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    MeVFlavor flavor
)
{
    if (rho_gcm3 <= 0.0 || T_MeV <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    if (flavor == MeVFlavor::NuX) {
        return 0.0;
    }

    const double ye = clamp_ye(Ye);
    const double rho10 = rho_gcm3 / 1.0e10;
    const double T10 = T_MeV / 10.0;
    const double flavor_factor =
        flavor == MeVFlavor::NuE ? ye : (1.0 - ye);

    // Approximate local spectral emissivity model for charged-current
    // beta/URCA-like processes. The returned value is a physically motivated
    // emissivity proxy per MeV with arbitrary normalization; it is not a
    // calibrated luminosity. Scaling: rho*T^6 with Ye or (1-Ye) flavor factors.
    // Validity limitations: no electron/nucleon degeneracy, blocking,
    // threshold, weak-magnetism, or detailed beta-equilibrium corrections.
    return 1.0e30
        * safe_pow(rho10, 1.0)
        * safe_pow(T10, 6.0)
        * flavor_factor
        * mev_thermal_spectral_shape(E_MeV, T_MeV);
}

double mev_emissivity_pair(
    double,
    double T_MeV,
    double,
    double E_MeV,
    MeVFlavor flavor
)
{
    if (T_MeV <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double T10 = T_MeV / 10.0;
    const double flavor_factor =
        flavor == MeVFlavor::NuX ? 1.0 : 0.7;

    // Approximate local spectral emissivity model for e- e+ annihilation.
    // Returned units are the same emissivity proxy per MeV as the URCA branch.
    // Scaling: T^9. The coefficient is a diagnostic normalization, not a
    // calibrated pair-annihilation rate.
    return 3.0e28
        * safe_pow(T10, 9.0)
        * flavor_factor
        * mev_thermal_spectral_shape(E_MeV, T_MeV);
}

double mev_emissivity_brems(
    double rho_gcm3,
    double T_MeV,
    double,
    double E_MeV,
    MeVFlavor flavor
)
{
    if (rho_gcm3 <= 0.0 || T_MeV <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double rho10 = rho_gcm3 / 1.0e10;
    const double T10 = T_MeV / 10.0;
    const double flavor_factor =
        flavor == MeVFlavor::NuX ? 1.0 : 0.8;

    // Approximate local spectral emissivity model for nucleon-nucleon
    // bremsstrahlung. Returned units are an emissivity proxy per MeV.
    // Scaling: rho^2*T^5. The coefficient is not calibrated to a detailed
    // microphysical rate.
    return 1.0e27
        * safe_pow(rho10, 2.0)
        * safe_pow(T10, 5.0)
        * flavor_factor
        * mev_thermal_spectral_shape(E_MeV, T_MeV);
}

double mev_total_emissivity(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (params.model == MeVModel::Toy) {
        if (rho_gcm3 <= 0.0 || T_MeV <= 0.0 || E_MeV <= 0.0) {
            return 0.0;
        }

        // Legacy toy branch retained for backward comparisons. This is a
        // phenomenological emissivity proxy, not the canonical physical CPU
        // MeV model.
        const double nb = rho_gcm3 / constants::m_u_g;
        const double neutron_fraction = std::clamp(1.0 - Ye, 0.0, 1.0);
        const double proton_fraction = std::clamp(Ye, 0.0, 1.0);
        const double charged_current_weight =
            neutron_fraction + 0.5 * proton_fraction;
        const double thermal_spectrum =
            E_MeV * E_MeV / (std::exp(std::min(E_MeV / T_MeV, 120.0)) + 1.0);
        const double capture_like =
            nb * charged_current_weight * std::pow(T_MeV, 6.0);
        const double pair_like =
            0.05 * nb * std::pow(T_MeV, 9.0);

        return params.norm * (capture_like + pair_like) * thermal_spectrum;
    }

    double total = 0.0;

    if (params.include_urca) {
        double urca = mev_emissivity_urca(
            rho_gcm3,
            T_MeV,
            Ye,
            E_MeV,
            params.flavor
        );
        if (params.use_degeneracy_correction) {
            urca *= mev_urca_degeneracy_correction(
                rho_gcm3,
                Ye,
                T_MeV,
                params.flavor
            );
        }
        total += urca;
    }

    if (params.include_pair) {
        total += mev_emissivity_pair(
            rho_gcm3,
            T_MeV,
            Ye,
            E_MeV,
            params.flavor
        );
    }

    if (params.include_brems) {
        total += mev_emissivity_brems(
            rho_gcm3,
            T_MeV,
            Ye,
            E_MeV,
            params.flavor
        );
    }

    return params.norm * total;
}

double mev_opacity_absorption_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (params.model == MeVModel::Toy) {
        if (!params.include_absorption || rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
            return 0.0;
        }
        const double nb = rho_gcm3 / constants::m_u_g;
        const double neutron_fraction = std::clamp(1.0 - Ye, 0.0, 1.0);
        const double proton_fraction = std::clamp(Ye, 0.0, 1.0);
        const double target_fraction =
            neutron_fraction + 0.25 * proton_fraction;

        return nb
            * params.sigma_abs0_cm2
            * E_MeV * E_MeV
            * target_fraction;
    }

    return mev_opacity_absorption_neutron_cm_inv(rho_gcm3, Ye, E_MeV, params)
        + mev_opacity_absorption_proton_cm_inv(rho_gcm3, Ye, E_MeV, params);
}

double mev_opacity_absorption_neutron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (!params.include_absorption || !params.include_abs_n ||
        params.flavor != MeVFlavor::NuE ||
        rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double nb = rho_gcm3 / constants::m_u_g;
    const double neutron_fraction = 1.0 - clamp_ye(Ye);

    // nu_e + n -> p + e-. Approximate charged-current absorption opacity in
    // cm^-1 with sigma proportional to E_MeV^2.
    return nb * neutron_fraction * params.sigma_abs0_cm2 * E_MeV * E_MeV;
}

double mev_opacity_absorption_proton_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (!params.include_absorption || !params.include_abs_p ||
        params.flavor != MeVFlavor::AntiNuE ||
        rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double nb = rho_gcm3 / constants::m_u_g;
    const double proton_fraction = clamp_ye(Ye);

    // anti_nu_e + p -> n + e+. Approximate charged-current absorption opacity
    // in cm^-1 with sigma proportional to E_MeV^2.
    return nb * proton_fraction * params.sigma_abs0_cm2 * E_MeV * E_MeV;
}

double mev_opacity_scattering_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (params.model == MeVModel::Toy) {
        if (!params.include_scattering || rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
            return 0.0;
        }
        const double nb = rho_gcm3 / constants::m_u_g;
        const double composition_factor =
            1.0 + 0.5 * std::clamp(Ye, 0.0, 1.0);

        return nb
            * params.sigma_scat0_cm2
            * E_MeV * E_MeV
            * composition_factor;
    }

    return mev_opacity_scattering_neutron_cm_inv(rho_gcm3, Ye, E_MeV, params)
        + mev_opacity_scattering_proton_cm_inv(rho_gcm3, Ye, E_MeV, params)
        + mev_opacity_scattering_electron_cm_inv(rho_gcm3, Ye, E_MeV, params);
}

double mev_opacity_scattering_neutron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (!params.include_scattering || !params.include_scat_n ||
        rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double nb = rho_gcm3 / constants::m_u_g;
    const double neutron_fraction = 1.0 - clamp_ye(Ye);

    // Neutral-current scattering on neutrons. Coefficient is a documented
    // low-energy proxy with sigma proportional to E_MeV^2.
    return nb * neutron_fraction * params.sigma_scat0_cm2 * E_MeV * E_MeV;
}

double mev_opacity_scattering_proton_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (!params.include_scattering || !params.include_scat_p ||
        rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double nb = rho_gcm3 / constants::m_u_g;
    const double proton_fraction = clamp_ye(Ye);

    // Neutral-current scattering on protons. The 0.5 factor preserves the
    // previous composition scaling while exposing the proton contribution.
    return nb * proton_fraction * 0.5 * params.sigma_scat0_cm2 * E_MeV * E_MeV;
}

double mev_opacity_scattering_electron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    if (!params.include_scattering || !params.include_scat_e ||
        rho_gcm3 <= 0.0 || E_MeV <= 0.0) {
        return 0.0;
    }

    const double ne = electron_number_density_cm3(rho_gcm3, Ye);

    // Approximate neutral-current scattering on electrons/positrons. Positrons
    // are not solved explicitly; this is a diagnostic electron-density proxy.
    return ne * params.sigma_scat_e0_cm2 * E_MeV * E_MeV;
}

double mev_total_opacity_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
)
{
    return mev_opacity_absorption_cm_inv(rho_gcm3, Ye, E_MeV, params)
        + mev_opacity_scattering_cm_inv(rho_gcm3, Ye, E_MeV, params);
}

}
