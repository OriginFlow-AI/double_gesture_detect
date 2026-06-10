#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "double_ok_gesture/capture_gate.hpp"

namespace double_ok_gesture {

struct RecognizerConfig {
    int max_num_hands = 2;
    double ok_threshold = 0.68;
    int stable_window = 5;
    int stable_min_positive = 3;
    double min_detection_confidence = 0.55;
    double min_tracking_confidence = 0.55;
};

struct CameraConfig {
    std::string source = "/dev/video0";
    int width = 1280;
    int height = 720;
    double fps = 30.0;
    std::string fourcc = "MJPG";
    int open_retries = 5;
    double retry_delay_sec = 0.5;
    int warmup_reads = 5;
    int read_failure_limit = 5;
};

struct DataCaptureConfig {
    bool enabled = true;
    std::filesystem::path output_dir = "data/raw/rv1126_gate";
    double cooldown_sec = 1.0;
};

struct RuntimeConfig {
    RecognizerConfig recognizer;
    CaptureGateConfig capture_gate;
    DataCaptureConfig data_capture;
};

RuntimeConfig load_runtime_config(const std::filesystem::path& path);
RuntimeConfig load_runtime_config_or_default(const std::optional<std::filesystem::path>& path);
void apply_threshold_override(RuntimeConfig& config, std::optional<double> threshold);
void require_glasses_pose(RuntimeConfig& config);

}  // namespace double_ok_gesture
