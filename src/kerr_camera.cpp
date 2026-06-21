#include "kerr_camera.hpp"

#include <cmath>
#include <algorithm>

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;

    double deg_to_rad(double x)
    {
        return x * PI / 180.0;
    }

    double zamo_energy(
        const KerrMetric& metric,
        double r,
        double theta,
        double p_t,
        double p_phi
    )
    {
        const double alpha =
            metric.lapse(r, theta);

        const double omega =
            metric.omega_frame_drag(r, theta);

        return -(p_t + omega * p_phi) / alpha;
    }

    double wrapped_delta_phi(double phi_new, double phi_old)
    {
        double dphi = phi_new - phi_old;

        while (dphi > PI) {
            dphi -= 2.0 * PI;
        }

        while (dphi < -PI) {
            dphi += 2.0 * PI;
        }

        return dphi;
    }

    double zamo_spatial_interval_rg(
        const KerrMetric& metric,
        const GeodesicState& current,
        const GeodesicState& previous
    )
    {
        const double r_mid =
            0.5 * (current.r + previous.r);

        const double theta_mid =
            0.5 * (current.theta + previous.theta);

        double g[4][4];
        metric.metric(r_mid, theta_mid, g);

        const double dr =
            current.r - previous.r;

        const double dtheta =
            current.theta - previous.theta;

        const double dphi =
            wrapped_delta_phi(current.phi, previous.phi);

        const double dl2 =
            g[1][1] * dr * dr
            + g[2][2] * dtheta * dtheta
            + g[3][3] * dphi * dphi;

        return std::sqrt(std::max(dl2, 0.0));
    }
}

KerrCamera::KerrCamera(
    double a_spin,
    double r_obs_rg,
    double theta_obs,
    double fov_deg,
    int nx,
    int ny,
    double r_max_rg,
    double h,
    KerrDerivativeMode derivative_mode
)
    : metric_(a_spin),
      geodesic_(metric_, h, 1.0e-6, derivative_mode),
      r_obs_(r_obs_rg),
      theta_obs_(theta_obs),
      fov_rad_(deg_to_rad(fov_deg)),
      nx_(nx),
      ny_(ny),
      r_max_(r_max_rg)
{
}

int KerrCamera::nx() const
{
    return nx_;
}

int KerrCamera::ny() const
{
    return ny_;
}

GeodesicState KerrCamera::initial_state(int i, int j) const
{
    const double u =
        (2.0 * (i + 0.5) / nx_ - 1.0) * std::tan(0.5 * fov_rad_);

    const double v =
        (2.0 * (j + 0.5) / ny_ - 1.0) * std::tan(0.5 * fov_rad_);

    const double norm = std::sqrt(1.0 + u*u + v*v);

    // Local orthonormal camera direction.
    // Backward ray: mostly inward radial.
    const double n_r     = -1.0 / norm;
    const double n_theta =  v   / norm;
    const double n_phi   =  u   / norm;

    double g[4][4];
    metric_.metric(r_obs_, theta_obs_, g);

    const double alpha =
        metric_.lapse(r_obs_, theta_obs_);

    const double omega =
        metric_.omega_frame_drag(r_obs_, theta_obs_);

    const double grr =
        g[1][1];

    const double gthth =
        g[2][2];

    const double gphph =
        g[3][3];

    // ZAMO/LNRF tetrad approximation in Boyer-Lindquist components.
    const double p_t_contra =
        1.0 / alpha;

    const double p_r_contra =
        n_r / std::sqrt(grr);

    const double p_theta_contra =
        n_theta / std::sqrt(gthth);

    const double p_phi_contra =
        n_phi / std::sqrt(gphph)
        + omega * p_t_contra;

    const double p_contra[4] = {
        p_t_contra,
        p_r_contra,
        p_theta_contra,
        p_phi_contra
    };

    double p_cov[4] = {0.0, 0.0, 0.0, 0.0};

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            p_cov[mu] += g[mu][nu] * p_contra[nu];
        }
    }

    GeodesicState y;

    y.t = 0.0;
    y.r = r_obs_;
    y.theta = theta_obs_;
    y.phi = 0.0;

    y.pt = p_cov[0];
    y.pr = p_cov[1];
    y.ptheta = p_cov[2];
    y.pphi = p_cov[3];

    return y;
}

RayPath KerrCamera::trace_pixel(int i, int j) const
{
    RayPath ray;

    ray.a_bh = metric_.a;
    ray.pixel_i = i;
    ray.pixel_j = j;

    ray.alpha_rg =
        (2.0 * (i + 0.5) / nx_ - 1.0) * std::tan(0.5 * fov_rad_);

    ray.beta_rg =
        (2.0 * (j + 0.5) / ny_ - 1.0) * std::tan(0.5 * fov_rad_);

    GeodesicState y =
        initial_state(i, j);

    GeodesicState y_prev = y;

    const double energy_obs =
        zamo_energy(
            metric_,
            r_obs_,
            theta_obs_,
            y.pt,
            y.pphi
        );

    const double r_h =
        metric_.horizon_radius();

    const double r_stop =
        r_h + 1.0e-3;

    const int max_steps = 200000;

    for (int step = 0; step < max_steps; ++step) {

        if (y.r <= r_stop) {
            ray.captured = true;
            break;
        }

        if (y.r >= r_max_ && step > 10) {
            break;
        }


        const double x =
            y.r * std::sin(y.theta) * std::cos(y.phi);

        const double ycart =
            y.r * std::sin(y.theta) * std::sin(y.phi);

        const double z =
            y.r * std::cos(y.theta);

        PathPoint p;

        p.r_rg = y.r;
        p.theta = y.theta;
        p.x_rg = x;
        p.y_rg = ycart;
        p.z_rg = z;

        if (step == 0) {
            p.dl_rg = 0.0;
        } else {
            p.dl_rg =
                zamo_spatial_interval_rg(
                    metric_,
                    y,
                    y_prev
                );
        }

        p.redshift_factor =
            zamo_energy(
                metric_,
                y.r,
                y.theta,
                y.pt,
                y.pphi
            ) / std::max(energy_obs, 1.0e-300);
        p.pt = y.pt;
        p.pr = y.pr;
        p.ptheta = y.ptheta;
        p.pphi = y.pphi;

        ray.points.push_back(p);

        (void)ycart;

        y_prev = y;

        geodesic_.step_adaptive(y);
    }

    return ray;
}

bool KerrCamera::trace_pixel_stream(
    int i,
    int j,
    const std::function<bool(const PathPoint&, int)>& visit
) const
{
    GeodesicState y =
        initial_state(i, j);

    GeodesicState y_prev = y;

    const double energy_obs =
        zamo_energy(
            metric_,
            r_obs_,
            theta_obs_,
            y.pt,
            y.pphi
        );

    const double r_h =
        metric_.horizon_radius();

    const double r_stop =
        r_h + 1.0e-3;

    const int max_steps = 200000;

    for (int step = 0; step < max_steps; ++step) {

        if (y.r <= r_stop) {
            return true;
        }

        if (y.r >= r_max_ && step > 10) {
            return false;
        }

        PathPoint p;
        p.r_rg = y.r;
        p.theta = y.theta;
        p.x_rg = y.r * std::sin(y.theta) * std::cos(y.phi);
        p.y_rg = y.r * std::sin(y.theta) * std::sin(y.phi);
        p.z_rg = y.r * std::cos(y.theta);
        p.dl_rg = step == 0
            ? 0.0
            : zamo_spatial_interval_rg(metric_, y, y_prev);
        p.redshift_factor =
            zamo_energy(
                metric_,
                y.r,
                y.theta,
                y.pt,
                y.pphi
            ) / std::max(energy_obs, 1.0e-300);
        p.pt = y.pt;
        p.pr = y.pr;
        p.ptheta = y.ptheta;
        p.pphi = y.pphi;

        if (!visit(p, step)) {
            return false;
        }

        y_prev = y;
        geodesic_.step_adaptive(y);
    }

    return false;
}
