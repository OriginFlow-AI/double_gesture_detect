#include "double_ok_gesture/runtime_pipeline.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace double_ok_gesture {

LandmarkBackend landmark_backend_from_string(const std::string& value) {
    if (value == "rknn") {
        return LandmarkBackend::Rknn;
    }
    if (value == "mediapipe") {
        return LandmarkBackend::MediaPipe;
    }
    if (value == "opencv-heuristic") {
        return LandmarkBackend::OpenCVDebug;
    }
    if (value == "none") {
        return LandmarkBackend::None;
    }
    throw std::invalid_argument("--landmark-backend must be one of: rknn, mediapipe, opencv-heuristic, none");
}

const char* landmark_backend_value(LandmarkBackend backend) {
    switch (backend) {
        case LandmarkBackend::Rknn:
            return "rknn";
        case LandmarkBackend::MediaPipe:
            return "mediapipe";
        case LandmarkBackend::OpenCVDebug:
            return "opencv-heuristic";
        case LandmarkBackend::None:
            return "none";
    }
    return "unknown";
}

bool landmark_backend_available_in_current_build(LandmarkBackend backend) {
    return backend == LandmarkBackend::OpenCVDebug || backend == LandmarkBackend::None;
}

std::unique_ptr<HandLandmarkProvider> make_landmark_provider(LandmarkBackend backend, const RuntimeConfig& config) {
    if (backend == LandmarkBackend::OpenCVDebug) {
        return std::make_unique<OpenCVDebugLandmarkProvider>(HandDetectorConfig{
            config.recognizer.max_num_hands,
            0.006,
            3.0,
        });
    }
    return std::make_unique<NullHandLandmarkProvider>(landmark_backend_value(backend), backend == LandmarkBackend::None);
}

RuntimeBundle make_runtime(const RuntimeOptions& options) {
    auto runtime_config = load_runtime_config(options.config_path);
    apply_threshold_override(runtime_config, options.threshold);
    if (options.require_glasses_pose) {
        require_glasses_pose(runtime_config);
    }
    if (options.capture_output_dir) {
        runtime_config.data_capture.output_dir = *options.capture_output_dir;
    }
    if (options.capture_cooldown_sec) {
        runtime_config.data_capture.cooldown_sec = *options.capture_cooldown_sec;
    }
    if (options.disable_auto_capture) {
        runtime_config.data_capture.enabled = false;
    }
    if (!std::isfinite(runtime_config.data_capture.cooldown_sec) || runtime_config.data_capture.cooldown_sec < 0.0) {
        throw std::invalid_argument("--capture-cooldown must be finite and non-negative");
    }

    OKHandClassifier classifier = options.model_path
                                      ? OKHandClassifier(*options.model_path, runtime_config.recognizer.ok_threshold)
                                      : OKHandClassifier(runtime_config.recognizer.ok_threshold);
    DoubleOKRecognizer recognizer(
        classifier,
        static_cast<std::size_t>(runtime_config.recognizer.stable_window),
        static_cast<std::size_t>(runtime_config.recognizer.stable_min_positive));

    auto landmark_provider = make_landmark_provider(options.landmark_backend, runtime_config);

    return {
        runtime_config,
        classifier,
        std::move(recognizer),
        std::move(landmark_provider),
        open_camera(options.camera),
        RuntimeMetrics(),
        options.landmark_backend,
    };
}

std::vector<DetectedHand> detect_hands(const RuntimeBundle& runtime, const cv::Mat& frame) {
    if (runtime.landmark_provider) {
        return runtime.landmark_provider->detect(frame);
    }
    return {};
}

RuntimeFrameResult process_runtime_frame(
    RuntimeBundle& runtime,
    const cv::Mat& frame,
    const ProcessFrameOptions& options) {
    const double started = monotonic_seconds();
    const auto detected_hands = detect_hands(runtime, frame);
    auto result = runtime.recognizer.process_hands(detected_hands);
    std::optional<CaptureGateDecision> decision;
    if (options.capture_gate) {
        decision = evaluate_capture_gate(result, runtime.config.capture_gate, options.glasses_pose);
    }
    return {std::move(result), std::move(decision), started};
}

}  // namespace double_ok_gesture
