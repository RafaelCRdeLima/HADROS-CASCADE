#ifndef RAY_HPP
#define RAY_HPP

#include <vector>

struct PathPoint {
    double r_rg = 0.0;
    double theta = 0.0;
    double x_rg = 0.0;
    double y_rg = 0.0;
    double z_rg = 0.0;
    double dl_rg = 0.0;
    double redshift_factor = 1.0;

    double pt = 0.0;
    double pr = 0.0;
    double ptheta = 0.0;
    double pphi = 0.0;
};

struct RayPath {
    int pixel_i = 0;
    int pixel_j = 0;

    double alpha_rg = 0.0;
    double beta_rg  = 0.0;
    double impact_parameter_rg = 0.0;

    double a_bh = 0.95;

    bool captured = false;

    std::vector<PathPoint> points;
};

#endif
