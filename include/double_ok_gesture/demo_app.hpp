#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "double_ok_gesture/camera.hpp"
#include "double_ok_gesture/runtime_pipeline.hpp"

namespace double_ok_gesture {

struct DemoArgs {
    CameraConfig camera;
    std::filesystem::path config = "configs/default.json";
    std::optional<std::filesystem::path> palm_model;
    std::optional<std::filesystem::path> hand_model;
    std::optional<double> threshold;
    bool capture_gate = false;
    bool require_glasses_pose = false;
    std::optional<std::filesystem::path> glasses_pose;
    double target_fps = 25.0;
    int dashboard_width = 1440;
    int dashboard_height = 810;
    bool fullscreen = false;
    std::filesystem::path screenshot_dir = "reports/live";
    std::optional<std::filesystem::path> capture_output_dir;
    std::optional<double> capture_cooldown_sec;
    bool disable_auto_capture = false;
    std::string log_level = "INFO";
    int max_frames = 0;
    int detection_skip_frames = 1;  // 跳帧检测：每 N 帧做一次推理
    bool loop_file = false;         // 文件输入时循环播放
    LandmarkBackend landmark_backend = default_landmark_backend();
    std::optional<std::filesystem::path> landmarks_json;
    bool right_half = false;
};

DemoArgs parse_demo_args(int argc, char** argv);
RuntimeOptions demo_runtime_options(const DemoArgs& args);
ProcessFrameOptions demo_process_frame_options(const DemoArgs& args);

}  // namespace double_ok_gesture
