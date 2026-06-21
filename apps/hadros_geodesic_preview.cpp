#include "kerr_camera.hpp"
#include "ray.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HADROS_GEODESIC_PREVIEW_GLFW
#include <GLFW/glfw3.h>
#endif

namespace fs = std::filesystem;

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

fs::path preview_output_dir()
{
    const char* env = std::getenv("HADROS_PREVIEW_OUTPUT_DIR");
    if (env && *env) return fs::path(env);
    return fs::path("output/camera_preview");
}

std::string timestamp_iso();
fs::path write_ppm(const std::vector<struct Rgb>& pixels, int nx, int ny);

struct PreviewConfig {
    double spin = 0.0001;
    double observer_distance_rg = 60.0;
    double inclination_deg = 80.0;
    double azimuth_deg = 0.0;
    double fov_deg = 25.0;
    double r_max_rg = 120.0;
    double integration_step = 0.05;
    int max_steps = 200000;
    int nx = 64;
    int ny = 64;
    int tile_size = 16;
    std::string display_mode = "combined";
    std::string sky_texture_path = "assets/sky/eso0932a.ppm";
    std::string quality = "medium";
    bool linear_filter = true;
    bool display_blur = false;
};

struct Rgb {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};
static_assert(sizeof(Rgb) == 3, "Rgb must match packed PPM RGB bytes");

double clamp(double value, double lo, double hi)
{
    return std::max(lo, std::min(hi, value));
}

struct SkyTexture {
    int width = 0;
    int height = 0;
    std::vector<Rgb> pixels;

    bool loaded() const
    {
        return width > 0 && height > 0 && pixels.size() == static_cast<std::size_t>(width * height);
    }

    Rgb sample(double lon, double theta) const
    {
        if (!loaded()) {
            return {20, 25, 35};
        }
        lon = std::fmod(lon, 2.0 * PI);
        if (lon < 0.0) lon += 2.0 * PI;
        theta = clamp(theta, 0.0, PI);

        const double u = lon / (2.0 * PI);
        const double v = theta / PI;
        const double x = u * static_cast<double>(width - 1);
        const double y = v * static_cast<double>(height - 1);
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = (x0 + 1) % width;
        const int y1 = std::min(height - 1, y0 + 1);
        const double tx = x - static_cast<double>(x0);
        const double ty = y - static_cast<double>(y0);

        auto at = [&](int xi, int yi) -> const Rgb& {
            return pixels[static_cast<std::size_t>(yi * width + xi)];
        };
        auto mix = [&](std::uint8_t a, std::uint8_t b, double t) -> double {
            return (1.0 - t) * static_cast<double>(a) + t * static_cast<double>(b);
        };
        auto mixd = [&](double a, double b, double t) -> double {
            return (1.0 - t) * a + t * b;
        };
        const Rgb& c00 = at(x0, y0);
        const Rgb& c10 = at(x1, y0);
        const Rgb& c01 = at(x0, y1);
        const Rgb& c11 = at(x1, y1);
        const double r0 = mix(c00.r, c10.r, tx);
        const double r1 = mix(c01.r, c11.r, tx);
        const double g0 = mix(c00.g, c10.g, tx);
        const double g1 = mix(c01.g, c11.g, tx);
        const double b0 = mix(c00.b, c10.b, tx);
        const double b1 = mix(c01.b, c11.b, tx);
        return {
            static_cast<std::uint8_t>(clamp(mixd(r0, r1, ty), 0.0, 255.0)),
            static_cast<std::uint8_t>(clamp(mixd(g0, g1, ty), 0.0, 255.0)),
            static_cast<std::uint8_t>(clamp(mixd(b0, b1, ty), 0.0, 255.0))
        };
    }
};

struct QualitySettings {
    std::vector<int> levels;
    double min_step = 0.08;
    double r_max_rg = 120.0;
    int nominal_max_steps = 200000;
};

struct RenderStats {
    std::uint64_t version = 0;
    int width = 0;
    int height = 0;
    int rays = 0;
    int stale_discards = 0;
    double render_seconds = 0.0;
    double fps = 0.0;
    double integration_step = 0.0;
    double r_max_rg = 0.0;
    int nominal_max_steps = 0;
    std::string quality;
    bool stale = false;
};

struct SharedFrame {
    std::mutex mutex;
    std::vector<Rgb> pixels;
    int width = 0;
    int height = 0;
    bool dirty = false;
    bool complete = false;
};

int env_int(const char* key, int fallback)
{
    const char* raw = std::getenv(key);
    if (!raw) return fallback;
    try {
        return std::max(1, std::stoi(raw));
    } catch (...) {
        return fallback;
    }
}

std::string env_string(const char* key, const std::string& fallback)
{
    const char* raw = std::getenv(key);
    return raw && *raw ? std::string(raw) : fallback;
}

QualitySettings quality_settings(const PreviewConfig& c)
{
    QualitySettings q;
    if (c.quality == "fast") {
        q.levels = {32, 64, 128};
        q.min_step = 2.50;
        q.r_max_rg = std::min(c.r_max_rg, 65.0);
        q.nominal_max_steps = 25000;
    } else if (c.quality == "high") {
        q.levels = {32, 64, 128, 256, 512};
        q.min_step = 0.40;
        q.r_max_rg = std::min(std::max(c.r_max_rg, 140.0), 180.0);
        q.nominal_max_steps = 200000;
    } else {
        q.levels = {32, 64, 128, 256};
        q.min_step = 1.20;
        q.r_max_rg = std::min(std::max(c.r_max_rg, 90.0), 120.0);
        q.nominal_max_steps = 70000;
    }
    return q;
}

void append_performance_log(const RenderStats& stats)
{
    const fs::path out_dir = preview_output_dir();
    fs::create_directories(out_dir);
    std::ofstream out(out_dir / "performance_log.txt", std::ios::app);
    out << timestamp_iso()
        << " version=" << stats.version
        << " quality=" << stats.quality
        << " resolution=" << stats.width << "x" << stats.height
        << " rays=" << stats.rays
        << " render_seconds=" << std::fixed << std::setprecision(4) << stats.render_seconds
        << " fps=" << std::setprecision(2) << stats.fps
        << " max_steps=" << stats.nominal_max_steps
        << " step=" << stats.integration_step
        << " r_max=" << stats.r_max_rg
        << " stale=" << (stats.stale ? "yes" : "no")
        << " stale_discards=" << stats.stale_discards
        << "\n";
}

void skip_ppm_ws_and_comments(std::istream& in)
{
    while (true) {
        in >> std::ws;
        if (in.peek() != '#') break;
        std::string ignored;
        std::getline(in, ignored);
    }
}

SkyTexture load_ppm_texture(const std::string& path)
{
    SkyTexture texture;
    if (path.empty()) {
        return texture;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Sky texture not found: " << path << ". Using procedural celestial grid.\n";
        return texture;
    }

    std::string magic;
    in >> magic;
    if (magic != "P6") {
        std::cerr << "Sky texture must be binary PPM P6: " << path << ". Using procedural celestial grid.\n";
        return texture;
    }
    skip_ppm_ws_and_comments(in);
    in >> texture.width;
    skip_ppm_ws_and_comments(in);
    in >> texture.height;
    skip_ppm_ws_and_comments(in);
    int max_value = 0;
    in >> max_value;
    in.get();
    if (!in || texture.width <= 0 || texture.height <= 0 || max_value != 255) {
        std::cerr << "Invalid PPM sky texture: " << path << ". Using procedural celestial grid.\n";
        texture = SkyTexture{};
        return texture;
    }
    texture.pixels.resize(static_cast<std::size_t>(texture.width * texture.height));
    in.read(reinterpret_cast<char*>(texture.pixels.data()), static_cast<std::streamsize>(texture.pixels.size() * 3));
    if (!in) {
        std::cerr << "Could not read all pixels from sky texture: " << path << ". Using procedural celestial grid.\n";
        texture = SkyTexture{};
        return texture;
    }
    std::cout << "Loaded sky texture: " << path << " (" << texture.width << "x" << texture.height << ")\n";
    return texture;
}

std::string timestamp_compact()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

std::string timestamp_iso()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

std::string json_payload(const PreviewConfig& c, const std::string& created_at)
{
    std::ostringstream out;
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"camera_name\": \"hadros_geodesic_preview\",\n";
    out << "  \"created_at\": \"" << created_at << "\",\n";
    out << "  \"observer_distance_rg\": " << c.observer_distance_rg << ",\n";
    out << "  \"inclination_deg\": " << c.inclination_deg << ",\n";
    out << "  \"azimuth_deg\": " << c.azimuth_deg << ",\n";
    out << "  \"fov_deg\": " << c.fov_deg << ",\n";
    out << "  \"spin\": " << c.spin << ",\n";
    out << "  \"r_max_rg\": " << c.r_max_rg << ",\n";
    out << "  \"integration_step\": " << c.integration_step << ",\n";
    out << "  \"max_steps\": " << c.max_steps << ",\n";
    out << "  \"target\": [0, 0, 0],\n";
    out << "  \"preview_resolution\": [" << c.nx << ", " << c.ny << "],\n";
    out << "  \"display_mode\": \"" << c.display_mode << "\",\n";
    out << "  \"sky_texture_path\": \"" << c.sky_texture_path << "\",\n";
    out << "  \"preview_quality\": \"" << c.quality << "\",\n";
    out << "  \"notes\": \"saved from HADROS geodesic camera preview; low-resolution Kerr null-geodesic tracing, not full radiative transfer\"\n";
    out << "}\n";
    return out.str();
}

fs::path save_camera(const PreviewConfig& c)
{
    fs::create_directories("configs/cameras");
    const std::string created_at = timestamp_iso();
    const fs::path camera_path = fs::path("configs/cameras") / ("camera_" + timestamp_compact() + ".json");
    const fs::path last_json = fs::path("configs/cameras") / "last_camera.json";
    const fs::path last_md = fs::path("configs/cameras") / "last_camera.md";
    const std::string payload = json_payload(c, created_at);

    {
        std::ofstream out(camera_path);
        out << payload;
    }
    {
        std::ofstream out(last_json);
        out << payload;
    }
    {
        std::ofstream out(last_md);
        out << "# HADROS Geodesic Camera Preview\n\n";
        out << "Low-resolution Kerr null-geodesic preview for camera framing. This is not full radiative transfer.\n\n";
        out << "- JSON: `" << camera_path.generic_string() << "`\n";
        out << "- created_at: `" << created_at << "`\n";
        out << "- observer_distance_rg: `" << c.observer_distance_rg << "`\n";
        out << "- inclination_deg: `" << c.inclination_deg << "`\n";
        out << "- azimuth_deg: `" << c.azimuth_deg << "`\n";
        out << "- fov_deg: `" << c.fov_deg << "`\n";
        out << "- spin: `" << c.spin << "`\n";
        out << "- r_max_rg: `" << c.r_max_rg << "`\n";
        out << "- integration_step: `" << c.integration_step << "`\n";
        out << "- display_mode: `" << c.display_mode << "`\n\n";
        out << "- sky_texture_path: `" << c.sky_texture_path << "`\n\n";
        out << "- preview_quality: `" << c.quality << "`\n\n";
        out << "```bash\n";
        out << "make image-from-small-cache CAMERA_CONFIG=configs/cameras/last_camera.json\n";
        out << "```\n";
    }

    std::cout << "Saved camera: " << camera_path << "\n";
    std::cout << "Updated: " << last_json << "\n";
    std::cout << "Updated: " << last_md << "\n";
    return camera_path;
}

bool crosses_disk(const RayPath& ray)
{
    if (ray.points.size() < 2) return false;
    for (std::size_t i = 1; i < ray.points.size(); ++i) {
        const PathPoint& a = ray.points[i - 1];
        const PathPoint& b = ray.points[i];
        if ((a.z_rg <= 0.0 && b.z_rg >= 0.0) || (a.z_rg >= 0.0 && b.z_rg <= 0.0)) {
            const double r_cyl = std::sqrt(b.x_rg * b.x_rg + b.y_rg * b.y_rg);
            if (r_cyl >= 4.0 && r_cyl <= 30.0) {
                return true;
            }
        }
    }
    return false;
}

Rgb celestial_color(const RayPath& ray, const PreviewConfig& c, const std::shared_ptr<const SkyTexture>& sky)
{
    if (ray.points.empty()) {
        return {20, 25, 35};
    }
    const PathPoint& p = ray.points.back();
    double lon = std::atan2(p.y_rg, p.x_rg) + c.azimuth_deg * PI / 180.0;
    if (lon < 0.0) lon += 2.0 * PI;
    if (sky && sky->loaded()) {
        return sky->sample(lon, p.theta);
    }
    const double lat = PI / 2.0 - p.theta;

    const int lon_band = static_cast<int>(std::floor(lon / (PI / 12.0)));
    const int lat_band = static_cast<int>(std::floor((lat + PI / 2.0) / (PI / 12.0)));
    const bool checker = ((lon_band + lat_band) % 2) == 0;
    const bool grid = std::abs(std::sin(12.0 * lon)) < 0.055 || std::abs(std::sin(12.0 * (lat + PI / 2.0))) < 0.055;

    if (grid) {
        return {245, 248, 255};
    }
    if (std::abs(lat) < 0.035) {
        return {240, 110, 65};
    }
    if (std::abs(lon) < 0.035 || std::abs(lon - PI / 2.0) < 0.035 || std::abs(lon - PI) < 0.035) {
        return {90, 170, 255};
    }
    return checker ? Rgb{55, 92, 150} : Rgb{15, 34, 78};
}

Rgb color_ray(const RayPath& ray, const PreviewConfig& c, const std::shared_ptr<const SkyTexture>& sky)
{
    if (ray.captured || ray.points.empty()) {
        return {0, 0, 0};
    }
    const bool disk = crosses_disk(ray);
    if (c.display_mode == "shadow_only") {
        return {210, 220, 235};
    }
    if (disk && c.display_mode != "celestial_sphere") {
        return {230, 130, 35};
    }
    if (c.display_mode == "disk_intersection") {
        return {18, 23, 34};
    }
    return celestial_color(ray, c, sky);
}

std::vector<Rgb> render_frame(const PreviewConfig& c, const std::shared_ptr<const SkyTexture>& sky)
{
    const QualitySettings q = quality_settings(c);
    const double step = std::max(std::max(1.0e-4, c.integration_step), q.min_step);
    const double r_max = std::max(c.observer_distance_rg + 1.0, q.r_max_rg);
    std::cout << "Rendering geodesic preview "
              << c.nx << "x" << c.ny
              << " spin=" << c.spin
              << " inc=" << c.inclination_deg
              << " fov=" << c.fov_deg
              << " mode=" << c.display_mode
              << " quality=" << c.quality
              << " step=" << step
              << " sky=" << (sky && sky->loaded() ? c.sky_texture_path : "procedural")
              << "\n";

    std::vector<Rgb> pixels(static_cast<std::size_t>(c.nx) * static_cast<std::size_t>(c.ny));
    const double theta = clamp(c.inclination_deg, 0.001, 179.999) * PI / 180.0;
    KerrCamera camera(
        clamp(c.spin, 0.0, 0.999),
        std::max(3.0, c.observer_distance_rg),
        theta,
        clamp(c.fov_deg, 1.0, 160.0),
        c.nx,
        c.ny,
        r_max,
        step
    );

#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < c.ny; ++j) {
        for (int i = 0; i < c.nx; ++i) {
            RayPath ray = camera.trace_pixel(i, j);
            pixels[static_cast<std::size_t>((c.ny - 1 - j) * c.nx + i)] = color_ray(ray, c, sky);
        }
    }
    return pixels;
}

RenderStats render_stage(
    const PreviewConfig& base,
    int resolution,
    std::uint64_t version,
    const std::shared_ptr<const SkyTexture>& sky,
    std::atomic<std::uint64_t>& current_version,
    SharedFrame& shared,
    std::vector<Rgb>& completed_pixels
)
{
    PreviewConfig c = base;
    c.nx = resolution;
    c.ny = resolution;
    const QualitySettings q = quality_settings(c);
    const double step = std::max(std::max(1.0e-4, c.integration_step), q.min_step);
    const double r_max = std::max(c.observer_distance_rg + 1.0, q.r_max_rg);
    const auto start = std::chrono::steady_clock::now();
    RenderStats stats;
    stats.version = version;
    stats.width = c.nx;
    stats.height = c.ny;
    stats.rays = c.nx * c.ny;
    stats.integration_step = step;
    stats.r_max_rg = r_max;
    stats.nominal_max_steps = q.nominal_max_steps;
    stats.quality = c.quality;

    std::cout << "Progressive stage " << c.nx << "x" << c.ny
              << " quality=" << c.quality
              << " step=" << step
              << " r_max=" << r_max
              << "\n" << std::flush;

    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.width = c.nx;
        shared.height = c.ny;
        shared.pixels.assign(static_cast<std::size_t>(c.nx) * static_cast<std::size_t>(c.ny), Rgb{12, 17, 28});
        shared.dirty = true;
        shared.complete = false;
    }

    const double theta = clamp(c.inclination_deg, 0.001, 179.999) * PI / 180.0;
    KerrCamera camera(
        clamp(c.spin, 0.0, 0.999),
        std::max(3.0, c.observer_distance_rg),
        theta,
        clamp(c.fov_deg, 1.0, 160.0),
        c.nx,
        c.ny,
        r_max,
        step
    );

    std::vector<Rgb> pixels(static_cast<std::size_t>(c.nx) * static_cast<std::size_t>(c.ny), Rgb{12, 17, 28});
    const int tile = std::max(4, c.tile_size);
    for (int y0 = 0; y0 < c.ny; y0 += tile) {
        for (int x0 = 0; x0 < c.nx; x0 += tile) {
            if (current_version.load(std::memory_order_relaxed) != version) {
                stats.stale = true;
                stats.stale_discards = 1;
                return stats;
            }
            const int y1 = std::min(c.ny, y0 + tile);
            const int x1 = std::min(c.nx, x0 + tile);
            for (int j = y0; j < y1; ++j) {
                for (int i = x0; i < x1; ++i) {
                    if (current_version.load(std::memory_order_relaxed) != version) {
                        stats.stale = true;
                        stats.stale_discards = 1;
                        return stats;
                    }
                    RayPath ray = camera.trace_pixel(i, j);
                    pixels[static_cast<std::size_t>((c.ny - 1 - j) * c.nx + i)] = color_ray(ray, c, sky);
                }
            }
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                if (shared.width == c.nx && shared.height == c.ny) {
                    for (int j = y0; j < y1; ++j) {
                        for (int i = x0; i < x1; ++i) {
                            const std::size_t idx = static_cast<std::size_t>((c.ny - 1 - j) * c.nx + i);
                            shared.pixels[idx] = pixels[idx];
                        }
                    }
                    shared.dirty = true;
                }
            }
        }
    }

    const auto end = std::chrono::steady_clock::now();
    stats.render_seconds = std::chrono::duration<double>(end - start).count();
    stats.fps = stats.render_seconds > 0.0 ? 1.0 / stats.render_seconds : 0.0;
    completed_pixels = pixels;
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.pixels = pixels;
        shared.dirty = true;
        shared.complete = true;
    }
    append_performance_log(stats);
    return stats;
}

RenderStats render_progressive(
    const PreviewConfig& base,
    std::uint64_t version,
    const std::shared_ptr<const SkyTexture>& sky,
    std::atomic<std::uint64_t>& current_version,
    SharedFrame& shared,
    std::vector<Rgb>& final_pixels
)
{
    RenderStats last;
    const QualitySettings q = quality_settings(base);
    for (std::size_t level = 0; level < q.levels.size(); ++level) {
        if (current_version.load(std::memory_order_relaxed) != version) {
            last.stale = true;
            last.stale_discards = 1;
            append_performance_log(last);
            return last;
        }
        std::vector<Rgb> completed;
        last = render_stage(base, q.levels[level], version, sky, current_version, shared, completed);
        if (last.stale) {
            append_performance_log(last);
            return last;
        }
        final_pixels = std::move(completed);
        if (level + 1 < q.levels.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(110));
        }
    }
    return last;
}

int run_headless_preview(PreviewConfig c, const std::shared_ptr<const SkyTexture>& sky)
{
    std::cout << "Running headless progressive geodesic preview.\n";
    std::atomic<std::uint64_t> version{1};
    SharedFrame shared;
    std::vector<Rgb> final_pixels;
    RenderStats stats = render_progressive(c, 1, sky, version, shared, final_pixels);
    if (final_pixels.empty() || stats.stale) {
        std::vector<Rgb> pixels = render_frame(c, sky);
        write_ppm(pixels, c.nx, c.ny);
        save_camera(c);
        return 0;
    }
    write_ppm(final_pixels, stats.width, stats.height);
    c.nx = stats.width;
    c.ny = stats.height;
    c.integration_step = stats.integration_step;
    c.r_max_rg = stats.r_max_rg;
    c.max_steps = stats.nominal_max_steps;
    save_camera(c);
    return 0;
}

fs::path write_ppm(const std::vector<Rgb>& pixels, int nx, int ny)
{
    const fs::path out_dir = preview_output_dir();
    fs::create_directories(out_dir);
    const fs::path path = out_dir / "geodesic_preview.ppm";
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << nx << " " << ny << "\n255\n";
    for (const Rgb& p : pixels) {
        out.put(static_cast<char>(p.r));
        out.put(static_cast<char>(p.g));
        out.put(static_cast<char>(p.b));
    }
    std::cout << "Saved preview image: " << path << "\n";
    return path;
}

PreviewConfig default_config()
{
    PreviewConfig c;
    c.nx = env_int("PREVIEW_NX", 64);
    c.ny = env_int("PREVIEW_NY", 64);
    c.quality = env_string("PREVIEW_QUALITY", "medium");
    const std::string filter = env_string("PREVIEW_FILTER", "linear");
    c.linear_filter = filter != "nearest";
    return c;
}

void parse_args(int argc, char* argv[], PreviewConfig& c, bool& headless)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--nx") {
            c.nx = std::max(1, std::atoi(next()));
        } else if (arg == "--ny") {
            c.ny = std::max(1, std::atoi(next()));
        } else if (arg == "--spin") {
            c.spin = std::atof(next());
        } else if (arg == "--inclination") {
            c.inclination_deg = std::atof(next());
        } else if (arg == "--azimuth") {
            c.azimuth_deg = std::atof(next());
        } else if (arg == "--fov") {
            c.fov_deg = std::atof(next());
        } else if (arg == "--r-obs") {
            c.observer_distance_rg = std::atof(next());
        } else if (arg == "--r-max") {
            c.r_max_rg = std::atof(next());
        } else if (arg == "--step") {
            c.integration_step = std::atof(next());
        } else if (arg == "--mode") {
            c.display_mode = next();
        } else if (arg == "--sky") {
            c.sky_texture_path = next();
        } else if (arg == "--quality") {
            c.quality = next();
        } else if (arg == "--filter") {
            const std::string filter = next();
            c.linear_filter = filter != "nearest";
        } else if (arg == "--blur") {
            c.display_blur = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: hadros_geodesic_preview [--headless] [--nx N] [--ny N]\n"
                << "       [--spin a] [--inclination deg] [--azimuth deg]\n"
                << "       [--fov deg] [--r-obs rg] [--r-max rg] [--step h]\n"
                << "       [--sky assets/sky/eso0932a.ppm]\n"
                << "       [--quality fast|medium|high] [--filter linear|nearest] [--blur]\n"
                << "       [--mode celestial_sphere|disk_intersection|shadow_only|combined]\n";
            std::exit(0);
        }
    }
    if (c.quality != "fast" && c.quality != "medium" && c.quality != "high") {
        std::cerr << "Unknown preview quality '" << c.quality << "'; using medium.\n";
        c.quality = "medium";
    }
}

#ifdef HADROS_GEODESIC_PREVIEW_GLFW
PreviewConfig* g_config = nullptr;
std::vector<Rgb>* g_pixels = nullptr;
std::atomic<std::uint64_t> g_camera_version{1};
bool g_needs_render = true;
bool g_dragging = false;
double g_last_x = 0.0;
double g_last_y = 0.0;

void request_render()
{
    ++g_camera_version;
    g_needs_render = true;
}

void key_callback(GLFWwindow* window, int key, int, int action, int)
{
    if ((action != GLFW_PRESS && action != GLFW_REPEAT) || !g_config) return;
    PreviewConfig& c = *g_config;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
        save_camera(c);
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_S || key == GLFW_KEY_ENTER) {
        save_camera(c);
    } else if (key == GLFW_KEY_R) {
        request_render();
    } else if (key == GLFW_KEY_UP) {
        c.inclination_deg = clamp(c.inclination_deg - 2.0, 0.001, 179.999);
        request_render();
    } else if (key == GLFW_KEY_DOWN) {
        c.inclination_deg = clamp(c.inclination_deg + 2.0, 0.001, 179.999);
        request_render();
    } else if (key == GLFW_KEY_LEFT) {
        c.azimuth_deg -= 4.0;
        request_render();
    } else if (key == GLFW_KEY_RIGHT) {
        c.azimuth_deg += 4.0;
        request_render();
    } else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
        c.observer_distance_rg = clamp(c.observer_distance_rg - 2.0, 4.0, 1000.0);
        request_render();
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
        c.observer_distance_rg = clamp(c.observer_distance_rg + 2.0, 4.0, 1000.0);
        request_render();
    } else if (key == GLFW_KEY_LEFT_BRACKET) {
        c.fov_deg = clamp(c.fov_deg - 2.0, 1.0, 160.0);
        request_render();
    } else if (key == GLFW_KEY_RIGHT_BRACKET) {
        c.fov_deg = clamp(c.fov_deg + 2.0, 1.0, 160.0);
        request_render();
    } else if (key == GLFW_KEY_A) {
        c.spin = clamp(c.spin - 0.02, 0.0, 0.999);
        request_render();
    } else if (key == GLFW_KEY_D) {
        c.spin = clamp(c.spin + 0.02, 0.0, 0.999);
        request_render();
    } else if (key == GLFW_KEY_1) {
        c.display_mode = "celestial_sphere";
        request_render();
    } else if (key == GLFW_KEY_2) {
        c.display_mode = "disk_intersection";
        request_render();
    } else if (key == GLFW_KEY_3) {
        c.display_mode = "shadow_only";
        request_render();
    } else if (key == GLFW_KEY_4) {
        c.display_mode = "combined";
        request_render();
    } else if (key == GLFW_KEY_5) {
        c.quality = "fast";
        request_render();
    } else if (key == GLFW_KEY_6) {
        c.quality = "medium";
        request_render();
    } else if (key == GLFW_KEY_7) {
        c.quality = "high";
        request_render();
    } else if (key == GLFW_KEY_F) {
        c.linear_filter = !c.linear_filter;
        request_render();
    } else if (key == GLFW_KEY_B) {
        c.display_blur = !c.display_blur;
        request_render();
    }
}

void cursor_callback(GLFWwindow*, double x, double y)
{
    if (!g_config) return;
    if (g_dragging) {
        g_config->azimuth_deg += 0.25 * (x - g_last_x);
        g_config->inclination_deg = clamp(g_config->inclination_deg + 0.25 * (y - g_last_y), 0.001, 179.999);
        request_render();
    }
    g_last_x = x;
    g_last_y = y;
}

void mouse_callback(GLFWwindow*, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_dragging = action == GLFW_PRESS;
    }
}

void scroll_callback(GLFWwindow*, double, double y)
{
    if (!g_config) return;
    g_config->fov_deg = clamp(g_config->fov_deg - y * 2.0, 1.0, 160.0);
    request_render();
}

std::vector<Rgb> loading_pixels(int nx, int ny)
{
    std::vector<Rgb> pixels(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const bool grid = (i % 8 == 0) || (j % 8 == 0);
            const bool diagonal = ((i + j) % 16) < 8;
            pixels[static_cast<std::size_t>(j * nx + i)] = grid
                ? Rgb{62, 74, 94}
                : (diagonal ? Rgb{28, 36, 54} : Rgb{16, 22, 34});
        }
    }
    return pixels;
}

std::vector<Rgb> blur_for_display(const std::vector<Rgb>& pixels, int nx, int ny)
{
    if (pixels.empty()) return pixels;
    std::vector<Rgb> out(pixels.size());
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int count = 0;
            int r = 0;
            int g = 0;
            int b = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::min(nx - 1, std::max(0, x + dx));
                    const int yy = std::min(ny - 1, std::max(0, y + dy));
                    const Rgb& p = pixels[static_cast<std::size_t>(yy * nx + xx)];
                    r += p.r;
                    g += p.g;
                    b += p.b;
                    ++count;
                }
            }
            out[static_cast<std::size_t>(y * nx + x)] = {
                static_cast<std::uint8_t>(r / count),
                static_cast<std::uint8_t>(g / count),
                static_cast<std::uint8_t>(b / count)
            };
        }
    }
    return out;
}

void upload_texture(GLuint texture, const std::vector<Rgb>& pixels, int nx, int ny, bool linear_filter)
{
    if (pixels.empty() || nx <= 0 || ny <= 0) return;
    glBindTexture(GL_TEXTURE_2D, texture);
    const GLint filter = linear_filter ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        nx,
        ny,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        pixels.data()
    );
}

void draw_texture_scaled(GLFWwindow* window, GLuint texture, int nx, int ny)
{
    int fb_width = 0;
    int fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    glViewport(0, 0, fb_width, fb_height);

    const float scale_x = static_cast<float>(fb_width) / static_cast<float>(std::max(1, nx));
    const float scale_y = static_cast<float>(fb_height) / static_cast<float>(std::max(1, ny));
    const float scale = std::max(1.0f, std::min(scale_x, scale_y));
    const float draw_width = scale * static_cast<float>(nx);
    const float draw_height = scale * static_cast<float>(ny);
    const float x0 = 0.5f * (static_cast<float>(fb_width) - draw_width);
    const float y0 = 0.5f * (static_cast<float>(fb_height) - draw_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(fb_width), 0.0, static_cast<double>(fb_height), -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x0, y0);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x0 + draw_width, y0);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(x0 + draw_width, y0 + draw_height);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(x0, y0 + draw_height);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

int run_window(PreviewConfig& c, std::shared_ptr<const SkyTexture> sky)
{
    if (!glfwInit()) {
        std::cerr << "GLFW could not initialize. Falling back to headless geodesic preview.\n";
        return run_headless_preview(c, sky);
    }
    GLFWwindow* window = glfwCreateWindow(
        std::max(512, c.nx * 3),
        std::max(512, c.ny * 3),
        "HADROS geodesic camera preview",
        nullptr,
        nullptr
    );
    if (!window) {
        glfwTerminate();
        std::cerr << "GLFW window creation failed. Falling back to headless geodesic preview.\n";
        return run_headless_preview(c, sky);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    g_config = &c;
    std::vector<Rgb> pixels = loading_pixels(c.nx, c.ny);
    g_pixels = &pixels;
    SharedFrame shared;
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.width = c.nx;
        shared.height = c.ny;
        shared.pixels = pixels;
        shared.dirty = true;
    }
    GLuint texture = 0;
    glGenTextures(1, &texture);
    upload_texture(texture, pixels, c.nx, c.ny, c.linear_filter);
    int texture_width = c.nx;
    int texture_height = c.ny;
    std::future<RenderStats> render_future;
    bool render_active = false;
    RenderStats last_stats;
    int stale_discards = 0;
    std::vector<Rgb> final_pixels;

    std::cout << "HADROS geodesic camera preview: low-resolution Kerr null-geodesic tracing.\n";
    std::cout << "Controls: R render, S save, arrows/mouse orbit, +/- distance, [] FOV, A/D spin, 1-4 modes, 5-7 quality, F filter, B blur, Q quit.\n";

    while (!glfwWindowShouldClose(window)) {
        if (g_needs_render && !render_active) {
            PreviewConfig render_config = c;
            const std::uint64_t version = g_camera_version.load(std::memory_order_relaxed);
            final_pixels.clear();
            render_future = std::async(std::launch::async, [render_config, sky, version, &shared, &final_pixels]() {
                return render_progressive(render_config, version, sky, g_camera_version, shared, final_pixels);
            });
            render_active = true;
            g_needs_render = false;
        }
        if (render_active && render_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            last_stats = render_future.get();
            if (last_stats.stale) {
                ++stale_discards;
            } else if (!final_pixels.empty()) {
                write_ppm(final_pixels, last_stats.width, last_stats.height);
            }
            render_active = false;
            if (g_needs_render) {
                continue;
            }
        }
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            if (shared.dirty && !shared.pixels.empty()) {
                std::vector<Rgb> display_pixels = c.display_blur
                    ? blur_for_display(shared.pixels, shared.width, shared.height)
                    : shared.pixels;
                upload_texture(texture, display_pixels, shared.width, shared.height, c.linear_filter);
                texture_width = shared.width;
                texture_height = shared.height;
                shared.dirty = false;
            }
        }
        std::ostringstream title;
        title << "HADROS geodesic preview | r=" << c.observer_distance_rg
              << " inc=" << c.inclination_deg
              << " fov=" << c.fov_deg
              << " a=" << c.spin
              << " mode=" << c.display_mode
              << " q=" << c.quality
              << " res=" << texture_width << "x" << texture_height
              << " step=" << (last_stats.integration_step > 0.0 ? last_stats.integration_step : quality_settings(c).min_step)
              << " rays=" << texture_width * texture_height
              << " t=" << std::fixed << std::setprecision(2) << last_stats.render_seconds << "s"
              << " fps=" << std::setprecision(1) << last_stats.fps
              << " stale=" << stale_discards
              << (c.linear_filter ? " linear" : " nearest")
              << (c.display_blur ? " blur" : "")
              << (render_active ? " | rendering..." : " | S save R rerender");
        glfwSetWindowTitle(window, title.str().c_str());
        glClearColor(0.02f, 0.025f, 0.035f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        draw_texture_scaled(window, texture, texture_width, texture_height);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    if (render_active) {
        ++g_camera_version;
        render_future.wait();
    }
    save_camera(c);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
#endif
} // namespace

int main(int argc, char* argv[])
{
    PreviewConfig c = default_config();
    std::cout << "HADROS Geodesic Camera Preview\n";
    std::cout << "Uses low-resolution Kerr null geodesics for camera framing; not full DIS/MeV/radiative transfer.\n";

    bool headless = std::getenv("HADROS_PREVIEW_HEADLESS") != nullptr;
    parse_args(argc, argv, c, headless);
    const auto sky = std::make_shared<const SkyTexture>(load_ppm_texture(c.sky_texture_path));

    if (headless) {
        return run_headless_preview(c, sky);
    }

#ifdef HADROS_GEODESIC_PREVIEW_GLFW
    return run_window(c, sky);
#else
    std::cout << "GLFW/OpenGL support was not compiled in. Running headless geodesic preview.\n";
    std::cout << "Install libglfw3-dev and mesa/OpenGL development packages for the interactive window.\n";
    return run_headless_preview(c, sky);
#endif
}
