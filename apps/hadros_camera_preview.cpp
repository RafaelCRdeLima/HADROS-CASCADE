#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef HADROS_CAMERA_PREVIEW_GLFW
#include <GLFW/glfw3.h>
#endif

namespace fs = std::filesystem;

constexpr double PI = 3.141592653589793238462643383279502884;

struct CameraConfig {
    std::string camera_name = "hadros_camera_preview";
    std::string created_at;
    double observer_distance_rg = 60.0;
    double inclination_deg = 80.0;
    double azimuth_deg = 0.0;
    double fov_deg = 25.0;
    double spin = 0.0001;
    int preview_nx = 256;
    int preview_ny = 256;
};

static std::string timestamp()
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

static std::string iso_time()
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

static std::string json_for(const CameraConfig& c)
{
    std::ostringstream out;
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"camera_name\": \"" << c.camera_name << "\",\n";
    out << "  \"created_at\": \"" << c.created_at << "\",\n";
    out << "  \"observer_distance_rg\": " << c.observer_distance_rg << ",\n";
    out << "  \"inclination_deg\": " << c.inclination_deg << ",\n";
    out << "  \"azimuth_deg\": " << c.azimuth_deg << ",\n";
    out << "  \"fov_deg\": " << c.fov_deg << ",\n";
    out << "  \"spin\": " << c.spin << ",\n";
    out << "  \"target\": [0, 0, 0],\n";
    out << "  \"preview_resolution\": [" << c.preview_nx << ", " << c.preview_ny << "],\n";
    out << "  \"notes\": \"saved from HADROS camera preview\"\n";
    out << "}\n";
    return out.str();
}

static std::string markdown_for(const CameraConfig& c, const fs::path& json_path)
{
    std::ostringstream out;
    out << "# HADROS Camera Preview\n\n";
    out << "Geometric camera preview -- not a physical radiative-transfer render.\n\n";
    out << "- JSON: `" << json_path.generic_string() << "`\n";
    out << "- created_at: `" << c.created_at << "`\n";
    out << "- observer_distance_rg: `" << c.observer_distance_rg << "`\n";
    out << "- inclination_deg: `" << c.inclination_deg << "`\n";
    out << "- azimuth_deg: `" << c.azimuth_deg << "`\n";
    out << "- fov_deg: `" << c.fov_deg << "`\n";
    out << "- spin: `" << c.spin << "`\n";
    out << "- preview_resolution: `" << c.preview_nx << " x " << c.preview_ny << "`\n\n";
    out << "Production command example:\n\n";
    out << "```bash\n";
    out << "make image-from-small-cache CAMERA_CONFIG=configs/cameras/last_camera.json\n";
    out << "```\n";
    return out.str();
}

static fs::path save_camera(CameraConfig& c)
{
    c.created_at = iso_time();
    fs::create_directories("configs/cameras");
    const fs::path json_path = fs::path("configs/cameras") / ("camera_" + timestamp() + ".json");
    const fs::path last_json = fs::path("configs/cameras") / "last_camera.json";
    const fs::path last_md = fs::path("configs/cameras") / "last_camera.md";

    const std::string payload = json_for(c);
    {
        std::ofstream out(json_path);
        out << payload;
    }
    {
        std::ofstream out(last_json);
        out << payload;
    }
    {
        std::ofstream out(last_md);
        out << markdown_for(c, json_path);
    }

    std::cout << "Saved camera: " << json_path << "\n";
    std::cout << "Updated: " << last_json << "\n";
    std::cout << "Updated: " << last_md << "\n";
    return json_path;
}

static int env_int(const char* key, int fallback)
{
    const char* raw = std::getenv(key);
    if (!raw) {
        return fallback;
    }
    try {
        return std::max(1, std::stoi(raw));
    } catch (...) {
        return fallback;
    }
}

#ifdef HADROS_CAMERA_PREVIEW_GLFW
static double clamp(double v, double lo, double hi)
{
    return std::max(lo, std::min(hi, v));
}

static CameraConfig* g_camera = nullptr;
static bool g_dragging = false;
static double g_last_x = 0.0;
static double g_last_y = 0.0;

static void draw_circle(double radius, int segments, double r, double g, double b)
{
    glColor3d(r, g, b);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
        const double a = 2.0 * PI * static_cast<double>(i) / static_cast<double>(segments);
        glVertex2d(radius * std::cos(a), radius * std::sin(a));
    }
    glEnd();
}

static void draw_disk_proxy()
{
    glColor4d(0.0, 0.55, 0.38, 0.35);
    glBegin(GL_QUADS);
    glVertex2d(-0.82, -0.10);
    glVertex2d(0.82, -0.10);
    glVertex2d(0.82, 0.10);
    glVertex2d(-0.82, 0.10);
    glEnd();
    draw_circle(0.82, 96, 0.0, 0.42, 0.30);
    draw_circle(0.45, 96, 0.0, 0.42, 0.30);
}

static void draw_scene(const CameraConfig& c)
{
    glClearColor(0.965f, 0.976f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.25, 1.25, -1.25, 1.25, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3d(0.80, 0.86, 0.94);
    glBegin(GL_LINES);
    glVertex2d(-1.12, 0.0);
    glVertex2d(1.12, 0.0);
    glVertex2d(0.0, -1.12);
    glVertex2d(0.0, 1.12);
    glEnd();

    draw_disk_proxy();

    const double funnel = 0.42;
    glColor3d(0.10, 0.38, 1.0);
    glBegin(GL_LINES);
    glVertex2d(0.0, 0.12);
    glVertex2d(-funnel, 1.08);
    glVertex2d(0.0, 0.12);
    glVertex2d(funnel, 1.08);
    glVertex2d(0.0, -0.12);
    glVertex2d(-funnel, -1.08);
    glVertex2d(0.0, -0.12);
    glVertex2d(funnel, -1.08);
    glEnd();

    glColor3d(0.03, 0.05, 0.09);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(0.0, 0.0);
    for (int i = 0; i <= 96; ++i) {
        const double a = 2.0 * PI * static_cast<double>(i) / 96.0;
        glVertex2d(0.11 * std::cos(a), 0.11 * std::sin(a));
    }
    glEnd();

    const double az = c.azimuth_deg * PI / 180.0;
    const double inc = c.inclination_deg * PI / 180.0;
    const double cam_r = 1.02;
    const double cam_x = cam_r * std::sin(inc) * std::cos(az);
    const double cam_y = cam_r * std::cos(inc);
    glColor3d(0.65, 0.30, 0.0);
    glBegin(GL_LINES);
    glVertex2d(cam_x, cam_y);
    glVertex2d(0.0, 0.0);
    const double half = c.fov_deg * PI / 360.0;
    glVertex2d(cam_x, cam_y);
    glVertex2d(cam_x - 0.25 * std::sin(az + half), cam_y - 0.25 * std::cos(az + half));
    glVertex2d(cam_x, cam_y);
    glVertex2d(cam_x - 0.25 * std::sin(az - half), cam_y - 0.25 * std::cos(az - half));
    glEnd();
}

static void key_callback(GLFWwindow* window, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }
    CameraConfig& c = *g_camera;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_S) {
        save_camera(c);
    } else if (key == GLFW_KEY_UP) {
        c.inclination_deg = clamp(c.inclination_deg - 1.0, 0.0, 180.0);
    } else if (key == GLFW_KEY_DOWN) {
        c.inclination_deg = clamp(c.inclination_deg + 1.0, 0.0, 180.0);
    } else if (key == GLFW_KEY_LEFT) {
        c.azimuth_deg -= 2.0;
    } else if (key == GLFW_KEY_RIGHT) {
        c.azimuth_deg += 2.0;
    } else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
        c.observer_distance_rg = clamp(c.observer_distance_rg - 1.0, 5.0, 1000.0);
    } else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
        c.observer_distance_rg = clamp(c.observer_distance_rg + 1.0, 5.0, 1000.0);
    } else if (key == GLFW_KEY_LEFT_BRACKET) {
        c.fov_deg = clamp(c.fov_deg - 1.0, 1.0, 160.0);
    } else if (key == GLFW_KEY_RIGHT_BRACKET) {
        c.fov_deg = clamp(c.fov_deg + 1.0, 1.0, 160.0);
    } else if (key == GLFW_KEY_A) {
        c.spin = clamp(c.spin - 0.01, 0.0, 0.999);
    } else if (key == GLFW_KEY_D) {
        c.spin = clamp(c.spin + 0.01, 0.0, 0.999);
    }
}

static void cursor_callback(GLFWwindow*, double x, double y)
{
    if (!g_dragging || !g_camera) {
        g_last_x = x;
        g_last_y = y;
        return;
    }
    g_camera->azimuth_deg += 0.25 * (x - g_last_x);
    g_camera->inclination_deg = clamp(g_camera->inclination_deg + 0.25 * (y - g_last_y), 0.0, 180.0);
    g_last_x = x;
    g_last_y = y;
}

static void mouse_button_callback(GLFWwindow*, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_dragging = action == GLFW_PRESS;
    }
}

static void scroll_callback(GLFWwindow*, double, double yoffset)
{
    if (g_camera) {
        g_camera->observer_distance_rg = clamp(g_camera->observer_distance_rg - 2.0 * yoffset, 5.0, 1000.0);
    }
}

static int run_glfw_preview(CameraConfig& camera)
{
    if (!glfwInit()) {
        std::cerr << "GLFW could not initialize. Falling back to default camera save.\n";
        save_camera(camera);
        return 0;
    }

    GLFWwindow* window = glfwCreateWindow(
        960,
        720,
        "HADROS Geometric camera preview -- not a physical radiative-transfer render",
        nullptr,
        nullptr
    );
    if (!window) {
        glfwTerminate();
        std::cerr << "GLFW window creation failed. Falling back to default camera save.\n";
        save_camera(camera);
        return 0;
    }

    g_camera = &camera;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    std::cout << "Geometric camera preview -- not a physical radiative-transfer render.\n";
    std::cout << "Controls: drag=orbit, scroll=zoom, arrows=inclination/azimuth, +/-=distance, []=FOV, A/D=spin, S/Enter=save, Q/Esc=quit.\n";

    while (!glfwWindowShouldClose(window)) {
        std::ostringstream title;
        title << "HADROS camera preview | r=" << std::fixed << std::setprecision(1)
              << camera.observer_distance_rg
              << " rg inc=" << camera.inclination_deg
              << " az=" << camera.azimuth_deg
              << " fov=" << camera.fov_deg
              << " spin=" << std::setprecision(3) << camera.spin
              << " | S saves";
        glfwSetWindowTitle(window, title.str().c_str());
        draw_scene(camera);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
#endif

int main()
{
    CameraConfig camera;
    camera.preview_nx = env_int("PREVIEW_NX", 256);
    camera.preview_ny = env_int("PREVIEW_NY", 256);

#ifdef HADROS_CAMERA_PREVIEW_GLFW
    return run_glfw_preview(camera);
#else
    std::cout << "Geometric camera preview -- not a physical radiative-transfer render.\n";
    std::cout << "OpenGL/GLFW preview support was not compiled in. Saving a default camera config instead.\n";
    std::cout << "Install GLFW/OpenGL development packages and rebuild with make build_preview_camera for the interactive window.\n";
    save_camera(camera);
    return 0;
#endif
}
