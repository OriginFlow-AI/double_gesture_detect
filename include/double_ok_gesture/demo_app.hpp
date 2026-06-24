#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

#include "double_ok_gesture/camera.hpp"
#include "double_ok_gesture/runtime_pipeline.hpp"

namespace double_ok_gesture {

struct DemoArgs {
    CameraConfig camera;
    std::filesystem::path config = "configs/default.json";
    std::optional<std::filesystem::path> model;
    std::optional<double> threshold;
    bool capture_gate = false;
    bool require_glasses_pose = false;
    std::optional<std::filesystem::path> glasses_pose;
    bool headless = false;
    bool list_cameras = false;
    double status_interval = 1.0;
    double target_fps = 25.0;
    int dashboard_width = 1440;
    int dashboard_height = 810;
    bool fullscreen = false;
    std::filesystem::path screenshot_dir = "reports/live";
    std::optional<std::filesystem::path> capture_output_dir;
    std::optional<double> capture_cooldown_sec;
    bool disable_auto_capture = false;
    int max_frames = 0;
    LandmarkBackend landmark_backend = LandmarkBackend::Rknn;
    std::optional<std::filesystem::path> landmarks_json;
};

DemoArgs parse_demo_args(int argc, char** argv);
RuntimeOptions demo_runtime_options(const DemoArgs& args);
ProcessFrameOptions demo_process_frame_options(const DemoArgs& args);
std::optional<std::string> backend_unavailable_message(LandmarkBackend backend);

int write_demo_camera_list(std::ostream& out);
int run_demo_headless(const DemoArgs& args, std::ostream& out, std::ostream& err);

}  // namespace double_ok_gesture
