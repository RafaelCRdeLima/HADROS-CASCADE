#include "schwarzschild_raytracer.hpp"

#include <cmath>
#include <algorithm>

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;
}

SchwarzschildRayTracer::SchwarzschildRayTracer(
    double r_obs_rg,
    double r_max_rg,
    double r_horizon_rg,
    double dlambda
)
    : r_obs_(r_obs_rg),
      r_max_(r_max_rg),
      r_horizon_(r_horizon_rg),
      dlambda_(dlambda)
{
}

double SchwarzschildRayTracer::radial_acceleration(double r, double b)
{
    return b*b * (1.0/(r*r*r) - 3.0/(r*r*r*r));
}

double SchwarzschildRayTracer::redshift_factor(double r)
{
    if (r <= 2.0) return 1.0e30;

    return 1.0 / std::sqrt(1.0 - 2.0/r);
}

RayPath SchwarzschildRayTracer::trace_ray(
    double alpha_rg,
    double beta_rg,
    int pixel_i,
    int pixel_j
) const
{
    RayPath ray;

    ray.pixel_i = pixel_i;
    ray.pixel_j = pixel_j;
    ray.alpha_rg = alpha_rg;
    ray.beta_rg = beta_rg;

    const double b = std::sqrt(alpha_rg*alpha_rg + beta_rg*beta_rg);

    ray.impact_parameter_rg = b;

    double r = std::sqrt(r_obs_*r_obs_ + b*b);
    double phi = std::atan2(b, r_obs_);

    const double V =
        b*b * (1.0 - 2.0/r) / (r*r);

    double pr = -std::sqrt(std::max(1.0 - V, 0.0));

    double previous_x = r * std::cos(phi);
    double previous_z = r * std::sin(phi);

    const int max_steps = 200000;

    for (int n = 0; n < max_steps; ++n) {

        if (r <= r_horizon_ + 1.0e-4) {
            ray.captured = true;
            break;
        }

        if (r > r_max_ && n > 10) {
            break;
        }

        const double x = r * std::cos(phi);
        const double z_abs = r * std::sin(phi);

        const double sign_beta = (beta_rg >= 0.0) ? 1.0 : -1.0;
        const double z = sign_beta * z_abs;

        const double dx = x - previous_x;
        const double dz = z - previous_z;

        const double dl_coord = std::sqrt(dx*dx + dz*dz);

        const double metric_factor =
            1.0 / std::sqrt(std::max(1.0 - 2.0/r, 1.0e-12));

        PathPoint p;

        p.r_rg = r;
        p.theta = std::acos(std::clamp(z / r, -1.0, 1.0));
        p.x_rg = x;
        p.z_rg = z;
        p.dl_rg = dl_coord * metric_factor;
        p.redshift_factor = redshift_factor(r);

        ray.points.push_back(p);

        previous_x = x;
        previous_z = z;

        auto rhs_r = [](double pr_) {
            return pr_;
        };

        auto rhs_phi = [](double r_, double b_) {
            return b_ / (r_*r_);
        };

        auto rhs_pr = [](double r_, double b_) {
            return radial_acceleration(r_, b_);
        };

        const double h = dlambda_;

        const double k1_r = rhs_r(pr);
        const double k1_phi = rhs_phi(r, b);
        const double k1_pr = rhs_pr(r, b);

        const double k2_r = rhs_r(pr + 0.5*h*k1_pr);
        const double k2_phi = rhs_phi(r + 0.5*h*k1_r, b);
        const double k2_pr = rhs_pr(r + 0.5*h*k1_r, b);

        const double k3_r = rhs_r(pr + 0.5*h*k2_pr);
        const double k3_phi = rhs_phi(r + 0.5*h*k2_r, b);
        const double k3_pr = rhs_pr(r + 0.5*h*k2_r, b);

        const double k4_r = rhs_r(pr + h*k3_pr);
        const double k4_phi = rhs_phi(r + h*k3_r, b);
        const double k4_pr = rhs_pr(r + h*k3_r, b);

        r += h * (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r) / 6.0;
        phi += h * (k1_phi + 2.0*k2_phi + 2.0*k3_phi + k4_phi) / 6.0;
        pr += h * (k1_pr + 2.0*k2_pr + 2.0*k3_pr + k4_pr) / 6.0;
    }

    return ray;
}