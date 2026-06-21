#ifndef MEV_NEUTRINO_PHYSICS_HPP
#define MEV_NEUTRINO_PHYSICS_HPP

#include <string>

namespace mev_neutrino {

enum class MeVModel {
    Toy,
    Physical
};

enum class MeVFlavor {
    NuE,
    AntiNuE,
    NuX
};

enum class MeVThermalProfile {
    Constant,
    InnerHotTorus,
    RadialPowerLaw,
    TorusPlusCoolEnvelope,
    CollapsarInnerHot
};

enum class MeVYeProfile {
    Constant,
    NeutronRichTorus,
    FunnelProtonRich,
    TorusEnvelopeContrast,
    CollapsarNeutronRich
};

enum class MeVSpectralMode {
    Monochromatic,
    FermiDiracBand
};

struct MeVPhysicsParams {
    MeVModel model = MeVModel::Physical;
    std::string model_name = "physical";
    MeVFlavor flavor = MeVFlavor::AntiNuE;
    std::string flavor_name = "anti_nu_e";
    bool include_urca = true;
    bool include_pair = true;
    bool include_brems = true;
    bool include_absorption = true;
    bool include_scattering = true;
    bool use_degeneracy_correction = false;
    bool include_abs_n = true;
    bool include_abs_p = true;
    bool include_scat_n = true;
    bool include_scat_p = true;
    bool include_scat_e = true;
    double norm = 1.0;
    double sigma_abs0_cm2 = 9.6e-44;
    double sigma_scat0_cm2 = 1.7e-44;
    double sigma_scat_e0_cm2 = 3.4e-45;
    MeVThermalProfile thermal_profile = MeVThermalProfile::InnerHotTorus;
    std::string thermal_profile_name = "inner_hot_torus";
    MeVYeProfile ye_profile = MeVYeProfile::NeutronRichTorus;
    std::string ye_profile_name = "neutron_rich_torus";
    MeVSpectralMode spectral_mode = MeVSpectralMode::Monochromatic;
    std::string spectral_mode_name = "monochromatic";
    double T0_MeV = 6.0;
    double T_floor_MeV = 0.1;
    double T_power = 0.2;
    double Ye_torus = 0.25;
    double Ye_funnel = 0.55;
    double Ye_envelope = 0.45;
    double Ye_floor = 0.01;
    double Ye_ceil = 0.60;
    double E_min_MeV = 3.0;
    double E_max_MeV = 50.0;
    int n_bins = 8;
};

MeVModel parse_mev_model(const std::string& model_name);
const char* mev_model_name(MeVModel model);

MeVFlavor parse_mev_flavor(const std::string& flavor_name);
const char* mev_flavor_name(MeVFlavor flavor);

MeVThermalProfile parse_mev_thermal_profile(const std::string& profile_name);
const char* mev_thermal_profile_name(MeVThermalProfile profile);

MeVYeProfile parse_mev_ye_profile(const std::string& profile_name);
const char* mev_ye_profile_name(MeVYeProfile profile);

MeVSpectralMode parse_mev_spectral_mode(const std::string& mode_name);
const char* mev_spectral_mode_name(MeVSpectralMode mode);

double mev_temperature_profile_MeV(
    double r_rg,
    double theta,
    double rho_gcm3,
    double rho0_gcm3,
    double r0_rg,
    double r_min_rg,
    double r_max_rg,
    const MeVPhysicsParams& params
);

double mev_ye_profile(
    double r_rg,
    double theta,
    double rho_gcm3,
    double rho0_gcm3,
    double r0_rg,
    double r_min_rg,
    double r_max_rg,
    const MeVPhysicsParams& params
);

double mev_thermal_spectral_shape(
    double E_MeV,
    double T_MeV
);

double mev_fermi_dirac_weight(
    double E_MeV,
    double T_MeV
);

double electron_number_density_cm3(
    double rho_gcm3,
    double Ye
);

double electron_fermi_momentum_MeV(
    double rho_gcm3,
    double Ye
);

double electron_chemical_potential_MeV(
    double rho_gcm3,
    double Ye,
    double T_MeV
);

double electron_degeneracy_eta(
    double rho_gcm3,
    double Ye,
    double T_MeV
);

double mev_urca_degeneracy_correction(
    double rho_gcm3,
    double Ye,
    double T_MeV,
    MeVFlavor flavor
);

double mev_emissivity_urca(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    MeVFlavor flavor
);

double mev_emissivity_pair(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    MeVFlavor flavor
);

double mev_emissivity_brems(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    MeVFlavor flavor
);

double mev_total_emissivity(
    double rho_gcm3,
    double T_MeV,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_absorption_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_absorption_neutron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_absorption_proton_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_scattering_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_scattering_neutron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_scattering_proton_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_opacity_scattering_electron_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

double mev_total_opacity_cm_inv(
    double rho_gcm3,
    double Ye,
    double E_MeV,
    const MeVPhysicsParams& params
);

}

#endif
