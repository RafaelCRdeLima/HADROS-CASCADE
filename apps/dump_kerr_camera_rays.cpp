#include "kerr_camera.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

int main(int argc, char* argv[])
{
    double a_spin = 0.9;
    double r_obs_rg = 60.0;
    double theta_obs_deg = 74.0;
    double fov_deg = 45.0;
    int nx = 100;
    int ny = 100;
    double r_max_rg = 120.0;
    double h = 0.001;

    if (argc > 1) {
        a_spin = std::atof(argv[1]);
    }
    if (argc > 2) {
        r_obs_rg = std::atof(argv[2]);
    }
    if (argc > 3) {
        theta_obs_deg = std::atof(argv[3]);
    }
    if (argc > 4) {
        fov_deg = std::atof(argv[4]);
    }
    if (argc > 5) {
        nx = std::atoi(argv[5]);
    }
    if (argc > 6) {
        ny = std::atoi(argv[6]);
    }
    if (argc > 7) {
        r_max_rg = std::atof(argv[7]);
    }
    if (argc > 8) {
        h = std::atof(argv[8]);
    }

    KerrCamera camera(
        a_spin,
        r_obs_rg,
        M_PI * theta_obs_deg / 180.0,
        fov_deg,
        nx,
        ny,
        r_max_rg,
        h
    );

    std::ofstream out("output/rays/kerr_camera_rays.dat");

    out << "# ray_id pixel_i pixel_j alpha beta "
    << "x_rg y_rg z_rg r_rg theta dl_rg redshift captured\n";

    int ray_id = 0;

    for (int i = 0; i < camera.nx(); ++i) {
        for (int j = 0; j < camera.ny(); ++j) {

            RayPath ray = camera.trace_pixel(i, j);

            for (const auto& p : ray.points) {
                out << std::scientific << std::setprecision(8)
                     << ray_id << " "
                     << ray.pixel_i << " "
                     << ray.pixel_j << " "
                     << ray.alpha_rg << " "
                     << ray.beta_rg << " "
                     << p.x_rg << " "
                     << p.y_rg << " "
                     << p.z_rg << " "
                     << p.r_rg << " "
                     << p.theta << " "
                     << p.dl_rg << " "
                     << p.redshift_factor << " "
                     << ray.captured << "\n";
            }

            out << "\n";
            ++ray_id;
        }
    }

    std::cout << "Saved: output/rays/kerr_camera_rays.dat\n";

    return 0;
}
