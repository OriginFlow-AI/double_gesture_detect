#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "double_ok_gesture/runtime_pipeline.hpp"

class QApplication;

namespace double_ok_gesture {

struct QtDashboardOptions {
    int width = 1440;
    int height = 810;
    bool fullscreen = false;
    std::filesystem::path screenshot_dir = "reports/live";
    int max_frames = 0;
    double target_fps = 25.0;
    bool capture_gate = false;
    std::optional<std::filesystem::path> glasses_pose;
    LandmarkBackend landmark_backend = default_landmark_backend();
    std::optional<std::filesystem::path> palm_model_path;
    std::optional<std::filesystem::path> hand_model_path;
    bool right_half = false;
    int crop_x = 0;
    int crop_y = 0;
    int crop_width = 0;
    int crop_height = 0;
};

class QtDashboard {
public:
    QtDashboard(QApplication& application, RuntimeBundle& runtime, QtDashboardOptions options);
    ~QtDashboard();

    QtDashboard(const QtDashboard&) = delete;
    QtDashboard& operator=(const QtDashboard&) = delete;

    int run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace double_ok_gesture
