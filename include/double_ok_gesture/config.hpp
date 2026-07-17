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
};

struct OnnxHandConfig {
    std::filesystem::path palm_model_path =
        "models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx";
    std::filesystem::path hand_model_path =
        "models/opencv_zoo/handpose_estimation_mediapipe_2023feb_opencv46.onnx";
    double palm_detection_threshold = 0.55;
    double hand_presence_threshold = 0.80;
    double palm_nms_threshold = 0.30;
    bool input_mirrored = false;
};

struct RknnHandConfig {
    std::filesystem::path palm_model_path =
        "models/rk3588/palm_detection_mediapipe_2023feb_fp16.rknn";
    std::filesystem::path hand_model_path =
        "models/rk3588/handpose_estimation_mediapipe_2023feb_fp16.rknn";
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
    std::filesystem::path output_dir = "data/raw/captures";
    double cooldown_sec = 1.0;
};

struct RuntimeConfig {
    RecognizerConfig recognizer;
    OnnxHandConfig onnx_hand;
    RknnHandConfig rknn_hand;
    CaptureGateConfig capture_gate;
    DataCaptureConfig data_capture;
};

RuntimeConfig load_runtime_config(const std::filesystem::path& path);
void validate_runtime_config(const RuntimeConfig& config);
void apply_threshold_override(RuntimeConfig& config, std::optional<double> threshold);
void require_glasses_pose(RuntimeConfig& config);

}  // namespace double_ok_gesture
