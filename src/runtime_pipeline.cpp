#include "double_ok_gesture/runtime_pipeline.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "double_ok_gesture/model_contract.hpp"

namespace double_ok_gesture {

LandmarkBackend landmark_backend_from_string(const std::string& value) {
    if (value == "yolov8-onnx") {
        return LandmarkBackend::Onnx;
    }
    if (value == "rknn" || value == "yolov8-rknn") {
        return LandmarkBackend::Rknn;
    }
    if (value == "mediapipe") {
        return LandmarkBackend::MediaPipe;
    }
    if (value == "landmarks-json") {
        return LandmarkBackend::LandmarksJson;
    }
    if (value == "opencv-heuristic") {
        return LandmarkBackend::OpenCVDebug;
    }
    if (value == "none") {
        return LandmarkBackend::None;
    }
    throw std::invalid_argument("--landmark-backend must be one of: yolov8-onnx, yolov8-rknn (rknn alias), mediapipe, landmarks-json, opencv-heuristic, none");
}

const char* landmark_backend_value(LandmarkBackend backend) {
    switch (backend) {
        case LandmarkBackend::Onnx:
            return "yolov8-onnx";
        case LandmarkBackend::Rknn:
            return "yolov8-rknn";
        case LandmarkBackend::MediaPipe:
            return "mediapipe";
        case LandmarkBackend::LandmarksJson:
            return "landmarks-json";
        case LandmarkBackend::OpenCVDebug:
            return "opencv-heuristic";
        case LandmarkBackend::None:
            return "none";
    }
    return "unknown";
}

bool landmark_backend_available_in_current_build(LandmarkBackend backend) {
    if (backend == LandmarkBackend::Onnx) {
        return true;
    }
    if (backend == LandmarkBackend::Rknn) {
#if DOUBLE_OK_ENABLE_RKNN
        return true;
#else
        return false;
#endif
    }
    return backend == LandmarkBackend::MediaPipe ||
           backend == LandmarkBackend::LandmarksJson ||
           backend == LandmarkBackend::OpenCVDebug ||
           backend == LandmarkBackend::None;
}

std::unique_ptr<HandLandmarkProvider> make_landmark_provider(const RuntimeOptions& options, const RuntimeConfig& config) {
    const LandmarkBackend backend = options.landmark_backend;
    if (backend == LandmarkBackend::Onnx) {
        if (!options.pose_model_path || options.pose_model_path->empty()) {
            throw std::invalid_argument(
                "yolov8-onnx requires --pose-model /path/to/hand_pose.onnx");
        }
        return std::make_unique<YoloV8OnnxHandLandmarkProvider>(
            YoloV8OnnxPipelineConfig{
                *options.pose_model_path,
                config.yolov8_pose.input_size,
                config.recognizer.max_num_hands,
                config.yolov8_pose.min_detection_confidence,
                config.yolov8_pose.min_keypoint_visibility,
                config.yolov8_pose.nms_iou_threshold,
                static_cast<std::size_t>(
                    config.yolov8_pose.min_reliable_keypoints),
            });
    }
    if (backend == LandmarkBackend::Rknn) {
#if !DOUBLE_OK_ENABLE_RKNN
        throw std::runtime_error(
            "yolov8-rknn backend is unavailable: rebuild with "
            "DOUBLE_OK_ENABLE_RKNN=ON and an RK3588 RKNN Runtime");
#elif !defined(__aarch64__)
        throw std::runtime_error(
            "yolov8-rknn backend requires an RK3588 AArch64 process");
#else
        const std::filesystem::path model_path =
            options.pose_model_path.value_or(config.yolov8_pose.model_path);
        const std::filesystem::path manifest_path =
            options.pose_manifest_path.value_or(
                config.yolov8_pose.manifest_path);
        (void)validate_yolov8_pose_model_contract(
            model_path, manifest_path, config.yolov8_pose.input_size);
        return std::make_unique<YoloV8RknnHandLandmarkProvider>(
            YoloV8RknnPipelineConfig{
                model_path,
                config.yolov8_pose.input_size,
                config.recognizer.max_num_hands,
                config.yolov8_pose.min_detection_confidence,
                config.yolov8_pose.min_keypoint_visibility,
                config.yolov8_pose.nms_iou_threshold,
                static_cast<std::size_t>(
                    config.yolov8_pose.min_reliable_keypoints),
            });
#endif
    }
    if (backend == LandmarkBackend::LandmarksJson) {
        if (!options.landmarks_json_path) {
            throw std::invalid_argument("--landmarks-json is required with --landmark-backend landmarks-json");
        }
        return std::make_unique<JsonHandLandmarkProvider>(*options.landmarks_json_path);
    }
    if (backend == LandmarkBackend::OpenCVDebug) {
        return std::make_unique<OpenCVDebugLandmarkProvider>(HandDetectorConfig{
            config.recognizer.max_num_hands,
            0.006,
            3.0,
        });
    }
    if (backend == LandmarkBackend::MediaPipe) {
        return std::make_unique<MediaPipePythonLandmarkProvider>();
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

    const bool production_backend =
        options.landmark_backend == LandmarkBackend::Rknn;
    const bool pose_backend = production_backend ||
                              options.landmark_backend == LandmarkBackend::Onnx;
    std::optional<HandAttributeClassifier> attribute_classifier;
    OKHandClassifier classifier(runtime_config.recognizer.ok_threshold);
    if (pose_backend) {
        const std::filesystem::path attribute_model =
            options.model_path.value_or(
                runtime_config.hand_attribute.model_path);
        if (attribute_model.empty() && production_backend) {
            throw std::runtime_error(
                "yolov8-rknn requires a trained Left/Right + OK hand "
                "attribute model; set --model or attribute_model_path");
        }
        if (!attribute_model.empty()) {
            attribute_classifier.emplace(
                attribute_model,
                runtime_config.recognizer.handedness_confidence_threshold,
                runtime_config.recognizer.ok_threshold,
                runtime_config.recognizer.input_mirrored);
        } else {
            log_message(
                LogLevel::Warning,
                "yolov8-onnx has no hand attribute model: using the "
                "experimental geometry OK scorer; handedness remains Unknown");
        }
    } else if (options.model_path) {
        // Preserve 0612 compatibility for explicitly selected test/debug
        // backends and their historical OK-only text classifier.
        classifier = OKHandClassifier(
            *options.model_path,
            runtime_config.recognizer.ok_threshold);
    }
    DoubleOKRecognizer recognizer(
        classifier,
        static_cast<std::size_t>(runtime_config.recognizer.stable_window),
        static_cast<std::size_t>(runtime_config.recognizer.stable_min_positive),
        std::move(attribute_classifier));

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
