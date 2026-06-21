#ifndef SCHWARZSCHILD_CAMERA_HPP
#define SCHWARZSCHILD_CAMERA_HPP

#include "ray.hpp"

class SchwarzschildCamera {
public:
    SchwarzschildCamera(
        double r_obs_rg = 80.0,
        double theta_obs = 1.5707963267948966,
        double fov_deg = 20.0,
        int nx = 21,
        int ny = 21,
        double r_max_rg = 120.0,
        double r_horizon_rg = 2.0,
        double dlambda = 0.02
    );

    RayPath trace_pixel(int i, int j) const;

    int nx() const;
    int ny() const;

private:
    double r_obs_;
    double theta_obs_;
    double fov_rad_;
    int nx_;
    int ny_;
    double r_max_;
    double r_horizon_;
    double dlambda_;

    RayPath integrate_ray(
        double n_r,
        double n_theta,
        int i,
        int j,
        double alpha,
        double beta
    ) const;

    static double redshift_factor(double r);
};

#endif