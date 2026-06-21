#include "optical_depth.hpp"
#include "constants.hpp"

#include <cmath>

namespace optical_depth {

double tau_straight_ray(
    double impact_parameter_rg,
    double Enu_GeV,
    double xmax_rg,
    double dx_rg,
    double M_bh_msun,
    const TorusProfile& torus,
    const SigmaTable& sigma
)
{
    const double rg_cm = constants::rg_cm(M_bh_msun);

    const double sigma_cm2 = sigma.sigma_cm2(Enu_GeV);

    double tau = 0.0;

    for (double x = -xmax_rg; x <= xmax_rg; x += dx_rg) {

        const double z = impact_parameter_rg;

        const double r = std::sqrt(x*x + z*z);

        if (r <= 1.0e-10) {
            continue;
        }

        const double theta = std::acos(z / r);

        const double rho = torus.rho(r, theta);

        const double nb =
            rho / constants::m_u_g;

        const double dl_cm =
            dx_rg * rg_cm;

        tau += nb * sigma_cm2 * dl_cm;
    }

    return tau;
}

double survival_probability(double tau)
{
    return std::exp(-tau);
}


double tau_along_ray(
    const RayPath& ray,
    double Enu_inf_GeV,
    const TorusProfile& torus,
    const SigmaTable& sigma
)
{
    double tau = 0.0;

    for (const auto& p : ray.points) {

        const double rho = torus.rho(p.r_rg, p.theta);

        const double nb =
            rho / constants::m_u_g;

        const double Enu_local_GeV =
            Enu_inf_GeV * p.redshift_factor;

        if (Enu_local_GeV < sigma.Emin() ||
            Enu_local_GeV > sigma.Emax()) {
            continue;
        }

        const double sigma_cm2 =
            sigma.sigma_cm2(Enu_local_GeV);

        // temporarily use M = 3 Msun
        const double dl_cm =
            p.dl_rg * constants::rg_cm(3.0);

        tau += nb * sigma_cm2 * dl_cm;
    }

    return tau;
}

}