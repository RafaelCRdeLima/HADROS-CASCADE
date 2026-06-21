#ifndef SCHWARZSCHILD_RAYTRACER_HPP
#define SCHWARZSCHILD_RAYTRACER_HPP

#include "ray.hpp"

class SchwarzschildRayTracer {
public:
    SchwarzschildRayTracer(
        double r_obs_rg = 100.0,
        double r_max_rg = 150.0,
        double r_horizon_rg = 2.0,
        double dlambda = 0.02
    );

    RayPath trace_ray(
        double alpha_rg,
        double beta_rg,
        int pixel_i = 0,
        int pixel_j = 0
    ) const;

private:
    double r_obs_;
    double r_max_;
    double r_horizon_;
    double dlambda_;

    static double radial_acceleration(double r, double b);
    static double redshift_factor(double r);
};

#endif