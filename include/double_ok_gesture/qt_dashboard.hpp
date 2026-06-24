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
    LandmarkBackend landmark_backend = LandmarkBackend::Rknn;
    std::optional<std::filesystem::path> model_path;
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
