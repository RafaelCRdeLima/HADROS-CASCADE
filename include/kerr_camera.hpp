#ifndef KERR_CAMERA_HPP
#define KERR_CAMERA_HPP

#include "kerr_metric.hpp"
#include "kerr_geodesic.hpp"
#include "ray.hpp"

#include <functional>

class KerrCamera {
public:
    KerrCamera(
        double a_spin,
        double r_obs_rg,
        double theta_obs,
        double fov_deg,
        int nx,
        int ny,
        double r_max_rg,
        double h,
        KerrDerivativeMode derivative_mode = KerrDerivativeMode::Environment
    );

    RayPath trace_pixel(int i, int j) const;
    bool trace_pixel_stream(
        int i,
        int j,
        const std::function<bool(const PathPoint&, int)>& visit
    ) const;

    int nx() const;
    int ny() const;

private:
    KerrMetric metric_;
    KerrGeodesic geodesic_;

    double r_obs_;
    double theta_obs_;
    double fov_rad_;
    int nx_;
    int ny_;
    double r_max_;

    GeodesicState initial_state(int i, int j) const;
};

#endif
