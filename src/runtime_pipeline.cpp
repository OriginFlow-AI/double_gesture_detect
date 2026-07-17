#include "double_ok_gesture/runtime_pipeline.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace double_ok_gesture {

LandmarkBackend landmark_backend_from_string(const std::string& value) {
    if (value == "onnx" || value == "mediapipe-onnx") {
        return LandmarkBackend::Onnx;
    }
    if (value == "landmarks-json") {
        return LandmarkBackend::LandmarksJson;
    }
    if (value == "none") {
        return LandmarkBackend::None;
    }
    throw std::invalid_argument(
        "--landmark-backend must be one of: onnx, landmarks-json, none");
}

const char* landmark_backend_value(LandmarkBackend backend) {
    switch (backend) {
        case LandmarkBackend::Onnx:
            return "onnx";
        case LandmarkBackend::LandmarksJson:
            return "landmarks-json";
        case LandmarkBackend::None:
            return "none";
    }
    return "unknown";
}

bool landmark_backend_available_in_current_build(LandmarkBackend backend) {
    return backend == LandmarkBackend::Onnx ||
           backend == LandmarkBackend::LandmarksJson ||
           backend == LandmarkBackend::None;
}

std::unique_ptr<HandLandmarkProvider> make_landmark_provider(const RuntimeOptions& options, const RuntimeConfig& config) {
    const LandmarkBackend backend = options.landmark_backend;
    if (backend == LandmarkBackend::Onnx) {
        return std::make_unique<MediaPipeOnnxHandLandmarkProvider>(
            MediaPipeOnnxPipelineConfig{
                options.palm_model_path.value_or(
                    config.onnx_hand.palm_model_path),
                options.hand_model_path.value_or(
                    config.onnx_hand.hand_model_path),
                config.recognizer.max_num_hands,
                config.onnx_hand.palm_detection_threshold,
                config.onnx_hand.hand_presence_threshold,
                config.onnx_hand.palm_nms_threshold,
                config.onnx_hand.input_mirrored,
            });
    }
    if (backend == LandmarkBackend::LandmarksJson) {
        if (!options.landmarks_json_path) {
            throw std::invalid_argument("--landmarks-json is required with --landmark-backend landmarks-json");
        }
        return std::make_unique<JsonHandLandmarkProvider>(*options.landmarks_json_path);
    }
    return std::make_unique<NullHandLandmarkProvider>(
        landmark_backend_value(backend), backend == LandmarkBackend::None);
}

RuntimeBundle make_runtime(const RuntimeOptions& options) {
    configure_logging(options.log_level);
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
    validate_runtime_config(runtime_config);

    OKHandClassifier classifier(runtime_config.recognizer.ok_threshold);
    DoubleOKRecognizer recognizer(
        classifier,
        static_cast<std::size_t>(runtime_config.recognizer.stable_window),
        static_cast<std::size_t>(runtime_config.recognizer.stable_min_positive));

    auto landmark_provider = make_landmark_provider(options, runtime_config);
    auto camera = open_camera(options.camera);

    log_message(
        LogLevel::Info,
        std::string("runtime initialized: backend=") +
            landmark_backend_value(options.landmark_backend) +
            ", config=" + options.config_path.string());

    return {
        runtime_config,
        classifier,
        std::move(recognizer),
        std::move(landmark_provider),
        std::move(camera),
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
    const double inference_started = monotonic_seconds();
    const auto detected_hands = detect_hands(runtime, frame);
    const double inference_ms =
        (monotonic_seconds() - inference_started) * 1000.0;
    auto result = runtime.recognizer.process_hands(detected_hands);
    std::optional<CaptureGateDecision> decision;
    if (options.capture_gate) {
        decision = evaluate_capture_gate(result, runtime.config.capture_gate, options.glasses_pose);
    }
    return {
        std::move(result), std::move(decision), started, inference_ms};
}

}  // namespace double_ok_gesture
