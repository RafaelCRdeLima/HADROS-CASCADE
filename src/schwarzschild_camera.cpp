#include "schwarzschild_camera.hpp"

#include <cmath>
#include <algorithm>

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;

    double deg_to_rad(double x)
    {
        return x * PI / 180.0;
    }
}

SchwarzschildCamera::SchwarzschildCamera(
    double r_obs_rg,
    double theta_obs,
    double fov_deg,
    int nx,
    int ny,
    double r_max_rg,
    double r_horizon_rg,
    double dlambda
)
    : r_obs_(r_obs_rg),
      theta_obs_(theta_obs),
      fov_rad_(deg_to_rad(fov_deg)),
      nx_(nx),
      ny_(ny),
      r_max_(r_max_rg),
      r_horizon_(r_horizon_rg),
      dlambda_(dlambda)
{
}

int SchwarzschildCamera::nx() const
{
    return nx_;
}

int SchwarzschildCamera::ny() const
{
    return ny_;
}

double SchwarzschildCamera::redshift_factor(double r)
{
    if (r <= 2.0) return 1.0e30;
    return 1.0 / std::sqrt(1.0 - 2.0 / r);
}

RayPath SchwarzschildCamera::trace_pixel(int i, int j) const
{
    const double u =
        (2.0 * (i + 0.5) / nx_ - 1.0) * std::tan(0.5 * fov_rad_);

    const double v =
        (2.0 * (j + 0.5) / ny_ - 1.0) * std::tan(0.5 * fov_rad_);

    const double norm = std::sqrt(1.0 + u*u + v*v);

    const double n_r   = -1.0 / norm;
    const double n_phi =  v   / norm;

    return integrate_ray(n_r, n_phi, i, j, u, v);
}

RayPath SchwarzschildCamera::integrate_ray(
    double n_r,
    double n_phi,
    int i,
    int j,
    double alpha,
    double beta
) const
{
    RayPath ray;

    ray.pixel_i = i;
    ray.pixel_j = j;
    ray.alpha_rg = alpha;
    ray.beta_rg  = beta;

    double r = r_obs_;
    double phi = 0.0;

    const double f_obs = 1.0 - 2.0 / r_obs_;

    const double L = r_obs_ * n_phi / std::sqrt(f_obs);

    double pr = n_r;

    ray.impact_parameter_rg = std::abs(L);

    const int max_steps = 200000;

    double x_prev = r * std::cos(phi);
    double z_prev = r * std::sin(phi);

    for (int step = 0; step < max_steps; ++step) {

        if (r <= r_horizon_ + 1.0e-4) {
            ray.captured = true;
            break;
        }

        if (r >= r_max_ && step > 10) {
            break;
        }

        const double x = r * std::cos(phi);
        const double z = r * std::sin(phi);

        const double dx = x - x_prev;
        const double dz = z - z_prev;

        const double dl_coord = std::sqrt(dx*dx + dz*dz);

        const double metric_factor =
            1.0 / std::sqrt(std::max(1.0 - 2.0 / r, 1.0e-12));

        PathPoint p;
        p.r_rg = r;
        p.theta = std::acos(std::clamp(z / r, -1.0, 1.0));
        p.x_rg = x;
        p.z_rg = z;
        p.dl_rg = dl_coord * metric_factor;
        p.redshift_factor = redshift_factor(r);

        ray.points.push_back(p);

        x_prev = x;
        z_prev = z;

        auto rhs = [](double rr, double prr, double LL) {
            struct Deriv {
                double dr;
                double dphi;
                double dpr;
            };

            Deriv d;

            d.dr = prr;
            d.dphi = LL / (rr * rr);

            d.dpr =
                LL * LL *
                (
                    1.0 / (rr*rr*rr)
                    - 3.0 / (rr*rr*rr*rr)
                );

            return d;
        };

        const double h = dlambda_;

        auto k1 = rhs(r, pr, L);

        auto k2 = rhs(
            r + 0.5*h*k1.dr,
            pr + 0.5*h*k1.dpr,
            L
        );

        auto k3 = rhs(
            r + 0.5*h*k2.dr,
            pr + 0.5*h*k2.dpr,
            L
        );

        auto k4 = rhs(
            r + h*k3.dr,
            pr + h*k3.dpr,
            L
        );

        r += h * (k1.dr + 2.0*k2.dr + 2.0*k3.dr + k4.dr) / 6.0;

        phi += h * (k1.dphi + 2.0*k2.dphi + 2.0*k3.dphi + k4.dphi) / 6.0;

        pr += h * (k1.dpr + 2.0*k2.dpr + 2.0*k3.dpr + k4.dpr) / 6.0;
    }

    return ray;
}