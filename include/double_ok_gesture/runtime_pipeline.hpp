#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "double_ok_gesture/camera.hpp"
#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/landmark_provider.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/runtime.hpp"

namespace double_ok_gesture {

enum class LandmarkBackend {
    Rknn,
    MediaPipe,
    LandmarksJson,
    OpenCVDebug,
    None,
};

struct RuntimeOptions {
    CameraConfig camera;
    std::filesystem::path config_path = "configs/default.json";
    std::optional<std::filesystem::path> model_path;
    std::optional<double> threshold;
    bool require_glasses_pose = false;
    std::optional<std::filesystem::path> capture_output_dir;
    std::optional<double> capture_cooldown_sec;
    bool disable_auto_capture = false;
    LandmarkBackend landmark_backend = LandmarkBackend::Rknn;
    std::optional<std::filesystem::path> landmarks_json_path;
};

struct RuntimeBundle {
    RuntimeConfig config;
    OKHandClassifier classifier;
    DoubleOKRecognizer recognizer;
    std::unique_ptr<HandLandmarkProvider> landmark_provider;
    CameraStream camera;
    RuntimeMetrics metrics;
    LandmarkBackend landmark_backend = LandmarkBackend::Rknn;
};

struct ProcessFrameOptions {
    bool capture_gate = false;
    std::optional<GlassesPose> glasses_pose;
};

struct RuntimeFrameResult {
    DoubleOKResult result;
    std::optional<CaptureGateDecision> decision;
    double started = 0.0;
};

LandmarkBackend landmark_backend_from_string(const std::string& value);
const char* landmark_backend_value(LandmarkBackend backend);
bool landmark_backend_available_in_current_build(LandmarkBackend backend);

RuntimeBundle make_runtime(const RuntimeOptions& options);
std::vector<DetectedHand> detect_hands(const RuntimeBundle& runtime, const cv::Mat& frame);
RuntimeFrameResult process_runtime_frame(
    RuntimeBundle& runtime,
    const cv::Mat& frame,
    const ProcessFrameOptions& options = {});

}  // namespace double_ok_gesture
