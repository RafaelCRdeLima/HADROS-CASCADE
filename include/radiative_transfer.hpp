#ifndef RADIATIVE_TRANSFER_HPP
#define RADIATIVE_TRANSFER_HPP

#include "ray.hpp"
#include "torus_profile.hpp"
#include "sigma_table.hpp"
#include "mev_neutrino_physics.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * @brief Integrated radiative-transfer observables accumulated along one ray.
 *
 * The structure stores UHE DIS attenuation products and diagnostic MeV
 * quantities computed by the stream-mode accumulators.
 */
struct RTResult {
    double tau = 0.0;
    double P_surv = 1.0;
    double I_obs = 0.0;

    double tau_mev = 0.0;
    double P_surv_mev = 1.0;
    double I_obs_mev = 0.0;
    double r_neutrinosphere_rg = -1.0;
    double leakage_factor = 1.0;
};

enum class UHESpectralModel {
    Monochromatic,
    PowerLaw,
    PowerLawCutoff
};

/**
 * @brief Parameters describing the UHE source spectrum used for weighting.
 */
struct UHESpectralParams {
    UHESpectralModel model = UHESpectralModel::Monochromatic;
    std::string model_name = "monochromatic";
    double gamma = 2.0;
    double ecut_GeV = 1.0e12;
    double e_min_GeV = 1.0e5;
    double e_max_GeV = 1.0e12;
    int n_bins = 8;
};

enum class UHESourceModel {
    InnerRing,
    FunnelWall,
    JetBase,
    ShockLayer,
    DensityWeighted
};

/**
 * @brief Spatial and spectral controls for phenomenological UHE source models.
 */
struct UHESourceParams {
    UHESourceModel model = UHESourceModel::InnerRing;
    std::string model_name = "inner_ring";
    double r_center_rg = 3.5;
    double sigma_r_rg = 1.0;
    double theta_width_rad = 8.0 * 3.141592653589793238462643383279502884 / 180.0;
    double powerlaw = 2.0;
    double emax_GeV = 1.0e12;
    double norm = 1.0;
    double funnel_theta_rad = 20.0 * 3.141592653589793238462643383279502884 / 180.0;
    double density_power_q = 1.0;
    double radial_power_s = 2.0;
    double rho_ref_gcm3 = -1.0;
    double cutoff_min = 0.0;
    double cutoff_max = 1.0e2;
    double gradient_dr_rg = 0.1;
    double gradient_dtheta_rad = 1.0 * 3.141592653589793238462643383279502884 / 180.0;
};

using UHECircularSourceParams = UHESourceParams;

/**
 * @brief Diagnostic MeV neutrino transport parameters for shared backgrounds.
 */
struct MeVThermalParams {
    double Enu_obs_MeV = 10.0;
    double norm = 1.0;
    double sigma_abs0_cm2 = 9.6e-44;
    double sigma_scat0_cm2 = 1.7e-44;
    double neutrinosphere_tau = 2.0 / 3.0;
    mev_neutrino::MeVModel model = mev_neutrino::MeVModel::Physical;
    std::string model_name = "physical";
    mev_neutrino::MeVFlavor flavor = mev_neutrino::MeVFlavor::AntiNuE;
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
    mev_neutrino::MeVThermalProfile thermal_profile = mev_neutrino::MeVThermalProfile::InnerHotTorus;
    std::string thermal_profile_name = "inner_hot_torus";
    mev_neutrino::MeVYeProfile ye_profile = mev_neutrino::MeVYeProfile::NeutronRichTorus;
    std::string ye_profile_name = "neutron_rich_torus";
    mev_neutrino::MeVSpectralMode spectral_mode = mev_neutrino::MeVSpectralMode::Monochromatic;
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

namespace radiative_transfer {

    /**
     * @brief Incremental Kerr-ray radiative-transfer accumulator.
     *
     * Path samples are fed one at a time, allowing stream-mode image generation
     * without storing full geodesic caches for every production run.
     */
    class KerrRTAccumulator {
    public:
        /**
         * @brief Construct an accumulator for one geodesic ray.
         * @param a_spin Dimensionless Kerr spin.
         * @param Enu_obs_GeV Observed UHE neutrino energy in GeV.
         * @param M_bh_msun Black-hole mass in solar masses.
         * @param torus Semi-analytic background density/source profile.
         * @param sigma DIS neutrino--nucleon cross-section table.
         * @param source UHE source-emissivity prescription.
         * @param mev Diagnostic MeV transport controls.
         * @param spectral UHE spectral weighting controls.
         * @param compute_mev Whether to accumulate MeV diagnostics.
         */
        KerrRTAccumulator(
            double a_spin,
            double Enu_obs_GeV,
            double M_bh_msun,
            const TorusProfile& torus,
            const SigmaTable& sigma,
            UHESourceParams source = UHESourceParams{},
            MeVThermalParams mev = MeVThermalParams{},
            UHESpectralParams spectral = UHESpectralParams{},
            bool compute_mev = true
        );

        /**
         * @brief Add one sampled point from the geodesic path.
         * @param p Path sample in geometric and physical coordinates.
         */
        void add_point(const PathPoint& p);

        /**
         * @brief Return final optical-depth, survival, and intensity products.
         * @return Accumulated radiative-transfer result.
         */
        RTResult result() const;
        void set_uhe_active(bool active);
        void set_mev_active(bool active);
        bool uhe_active() const;
        bool mev_active() const;

    private:
        double a_spin_;
        double Enu_obs_GeV_;
        double rg_cm_;
        const TorusProfile& torus_;
        const SigmaTable& sigma_;
        UHESourceParams source_;
        MeVThermalParams mev_;
        UHESpectralParams spectral_;
        bool compute_mev_;
        bool uhe_active_ = true;
        bool mev_active_ = true;
        double r_h_;
        double tau_ = 0.0;
        double Iobs_ = 0.0;
        double tau_mev_ = 0.0;
        double Iobs_mev_ = 0.0;
        double leakage_sum_ = 0.0;
        double leakage_weight_ = 0.0;
        double r_neutrinosphere_ = -1.0;
    };

    /**
     * @brief Multi-energy wrapper around KerrRTAccumulator for spectral runs.
     */
    class KerrSpectralRTAccumulator {
    public:
        KerrSpectralRTAccumulator(
            double a_spin,
            double Enu_obs_GeV,
            double M_bh_msun,
            const TorusProfile& torus,
            const SigmaTable& sigma,
            UHESourceParams source = UHESourceParams{},
            MeVThermalParams mev = MeVThermalParams{},
            UHESpectralParams spectral = UHESpectralParams{}
        );

        void add_point(const PathPoint& p);
        RTResult result() const;
        void set_uhe_active(bool active);
        void set_mev_active(bool active);
        bool uhe_active() const;
        bool mev_active() const;

    private:
        std::vector<double> dE_;
        std::vector<double> weights_;
        std::vector<std::unique_ptr<KerrRTAccumulator>> accumulators_;
    };

    /**
     * @brief Integrate UHE/MeV transfer along a complete Kerr ray path.
     * @param ray Stored ray path to integrate.
     * @param Enu_obs_GeV Observed UHE neutrino energy in GeV.
     * @param M_bh_msun Black-hole mass in solar masses.
     * @param torus Background torus profile.
     * @param sigma DIS cross-section table.
     * @param source UHE source prescription.
     * @param mev MeV diagnostic parameters.
     * @param spectral UHE spectral model parameters.
     * @param compute_mev Whether to compute MeV diagnostics.
     * @return Integrated transfer observables.
     */
    RTResult integrate_kerr_ray(
        const RayPath& ray,
        double Enu_obs_GeV,
        double M_bh_msun,
        const TorusProfile& torus,
        const SigmaTable& sigma,
        UHESourceParams source = UHESourceParams{},
        MeVThermalParams mev = MeVThermalParams{},
        UHESpectralParams spectral = UHESpectralParams{},
        bool compute_mev = true
    );

    RTResult integrate_kerr_ray_spectral(
        const RayPath& ray,
        double Enu_obs_GeV,
        double M_bh_msun,
        const TorusProfile& torus,
        const SigmaTable& sigma,
        UHESourceParams source = UHESourceParams{},
        MeVThermalParams mev = MeVThermalParams{},
        UHESpectralParams spectral = UHESpectralParams{}
    );

    double emissivity_collapsar_ring(
        double r_rg,
        double theta,
        double Enu_local_GeV,
        const UHESourceParams& source,
        const UHESpectralParams& spectral = UHESpectralParams{}
    );

    double emissivity_uhe(
        double r_rg,
        double theta,
        double Enu_local_GeV,
        const TorusProfile& torus,
        const UHESourceParams& source,
        const UHESpectralParams& spectral = UHESpectralParams{}
    );

    UHESpectralModel parse_uhe_spectral_model(const std::string& model_name);

    const char* uhe_spectral_model_name(UHESpectralModel model);

    double uhe_spectral_weight(
        double E_GeV,
        const UHESourceParams& source,
        const UHESpectralParams& spectral
    );

    UHESourceModel parse_uhe_source_model(const std::string& model_name);

    const char* uhe_source_model_name(UHESourceModel model);

    double uhe_source_spatial_weight(
        double r_rg,
        double theta,
        const TorusProfile& torus,
        const UHESourceParams& source
    );

    double emissivity_mev_thermal(
        double rho_gcm3,
        double T_MeV,
        double Ye,
        double Enu_local_MeV,
        const MeVThermalParams& mev
    );

    double opacity_mev_absorption_cm_inv(
        double rho_gcm3,
        double Ye,
        double Enu_local_MeV,
        const MeVThermalParams& mev
    );

    double opacity_mev_scattering_cm_inv(
        double rho_gcm3,
        double Ye,
        double Enu_local_MeV,
        const MeVThermalParams& mev
    );

}

#endif
