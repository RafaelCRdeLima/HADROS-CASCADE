#include "kerr_camera.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

struct BinaryRayHeader {
    std::int32_t ray_id;
    std::int32_t pixel_i;
    std::int32_t pixel_j;
    std::int32_t captured;
    std::int32_t npoints;
    double alpha_rg;
    double beta_rg;
};

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
    std::string output_path = "output/rays/kerr_geodesics.bin";

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
    if (argc > 9) {
        output_path = argv[9];
    }

#ifdef _OPENMP
    std::cout
        << "OpenMP max threads = "
        << omp_get_max_threads()
        << "\n";
#endif

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

    const int camera_nx = camera.nx();
    const int camera_ny = camera.ny();

    std::ofstream out(
        output_path,
        std::ios::binary
    );

    if (!out) {
        std::cerr << "Could not open " << output_path << "\n";
        return 1;
    }

    const std::int32_t magic = 0x4B47454F; // "KGEO"
    const std::int32_t version = 1;
    const std::int32_t nx_i32 = static_cast<std::int32_t>(camera_nx);
    const std::int32_t ny_i32 = static_cast<std::int32_t>(camera_ny);

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&nx_i32), sizeof(nx_i32));
    out.write(reinterpret_cast<const char*>(&ny_i32), sizeof(ny_i32));
    out.write(reinterpret_cast<const char*>(&a_spin), sizeof(a_spin));

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < camera_nx; ++i) {
        for (int j = 0; j < camera_ny; ++j) {

            const int ray_id = j * camera_nx + i;

            RayPath ray = camera.trace_pixel(i, j);

            BinaryRayHeader header;
            header.ray_id = static_cast<std::int32_t>(ray_id);
            header.pixel_i = static_cast<std::int32_t>(ray.pixel_i);
            header.pixel_j = static_cast<std::int32_t>(ray.pixel_j);
            header.captured = ray.captured ? 1 : 0;
            header.npoints = static_cast<std::int32_t>(ray.points.size());
            header.alpha_rg = ray.alpha_rg;
            header.beta_rg = ray.beta_rg;

#pragma omp critical
            {
                out.write(
                    reinterpret_cast<const char*>(&header),
                    sizeof(header)
                );

                if (!ray.points.empty()) {
                    out.write(
                        reinterpret_cast<const char*>(ray.points.data()),
                        static_cast<std::streamsize>(
                            ray.points.size() * sizeof(PathPoint)
                        )
                    );
                }
            }
        }
    }

    std::cout << "Camera grid: " << camera_nx << " x " << camera_ny << "\n";
    std::cout << "Saved: " << output_path << "\n";

    return 0;
}
