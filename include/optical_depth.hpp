#ifndef OPTICAL_DEPTH_HPP
#define OPTICAL_DEPTH_HPP

#include "sigma_table.hpp"
#include "torus_profile.hpp"
#include "ray.hpp"


namespace optical_depth {

    /**
     * @brief Compute optical depth along a straight reference ray.
     * @param impact_parameter_rg Impact parameter in gravitational radii.
     * @param Enu_GeV Neutrino energy in GeV.
     * @param xmax_rg Half-length of the integration domain in gravitational radii.
     * @param dx_rg Step size in gravitational radii.
     * @param M_bh_msun Black-hole mass in solar masses.
     * @param torus Semi-analytic density profile.
     * @param sigma DIS cross-section table.
     * @return Charged-current DIS optical depth.
     */
    double tau_straight_ray(
        double impact_parameter_rg,
        double Enu_GeV,
        double xmax_rg,
        double dx_rg,
        double M_bh_msun,
        const TorusProfile& torus,
        const SigmaTable& sigma
    );

    /**
     * @brief Compute optical depth along a sampled geodesic ray.
     * @param ray Geodesic path samples.
     * @param Enu_inf_GeV Neutrino energy at infinity in GeV.
     * @param torus Semi-analytic density profile.
     * @param sigma DIS cross-section table.
     * @return Charged-current DIS optical depth.
     */
    double tau_along_ray(
        const RayPath& ray,
        double Enu_inf_GeV,
        const TorusProfile& torus,
        const SigmaTable& sigma
    );

    /**
     * @brief Convert optical depth to survival probability.
     * @param tau Optical depth.
     * @return exp(-tau) with implementation-defined numerical safeguards.
     */
    double survival_probability(double tau);

}

#endif
