#include "double_ok_gesture/landmark_provider.hpp"

#include "double_ok_gesture/yolov8_pose_postprocess.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace double_ok_gesture {
namespace {

void require_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and in [0,1]");
    }
}

YoloV8OnnxPipelineConfig validate_config(YoloV8OnnxPipelineConfig config) {
    if (config.model.empty()) {
        throw std::invalid_argument("YOLOv8 ONNX model path must not be empty");
    }
    std::string extension = config.model.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension != ".onnx") {
        throw std::invalid_argument(
            "YOLOv8 ONNX model must have a .onnx extension: " +
            config.model.string());
    }
    if (!std::filesystem::is_regular_file(config.model)) {
        throw std::runtime_error(
            "YOLOv8 ONNX model not found: " + config.model.string());
    }
    if (std::filesystem::file_size(config.model) == 0) {
        throw std::runtime_error(
            "YOLOv8 ONNX model is empty: " + config.model.string());
    }
    if (config.input_size < 32 || config.input_size % 32 != 0) {
        throw std::invalid_argument(
            "YOLOv8 ONNX input size must be at least 32 and divisible by 32");
    }
    if (config.max_num_hands < 1 || config.max_num_hands > 2) {
        throw std::invalid_argument("YOLOv8 ONNX max_num_hands must be 1 or 2");
    }
    require_probability(
        config.min_detection_confidence, "min_detection_confidence");
    require_probability(
        config.min_keypoint_visibility, "min_keypoint_visibility");
    require_probability(config.nms_iou_threshold, "nms_iou_threshold");
    if (config.min_reliable_keypoints < 1 ||
        config.min_reliable_keypoints > yolov8_pose::kHandKeypointCount) {
        throw std::invalid_argument(
            "min_reliable_keypoints must be in [1,21]");
    }
    return config;
}

cv::Mat prepare_letterboxed_bgr(
    const cv::Mat& frame_bgr,
    const yolov8_pose::LetterboxTransform& transform) {
    const int resized_width = static_cast<int>(
        std::lround(transform.scale_x * transform.source_width));
    const int resized_height = static_cast<int>(
        std::lround(transform.scale_y * transform.source_height));
    const int pad_left = static_cast<int>(std::lround(transform.pad_left));
    const int pad_top = static_cast<int>(std::lround(transform.pad_top));
    if (resized_width <= 0 || resized_height <= 0 || pad_left < 0 ||
        pad_top < 0 || pad_left + resized_width > transform.input_width ||
        pad_top + resized_height > transform.input_height) {
        throw std::runtime_error("Invalid YOLOv8 ONNX letterbox geometry");
    }

    cv::Mat resized;
    cv::resize(
        frame_bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
        cv::INTER_LINEAR);
    cv::Mat letterboxed(
        transform.input_height, transform.input_width, CV_8UC3,
        cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(
        pad_left, pad_top, resized_width, resized_height)));
    return letterboxed;
}

std::vector<std::size_t> tensor_shape(const cv::Mat& tensor) {
    std::vector<std::size_t> shape;
    shape.reserve(static_cast<std::size_t>(tensor.dims));
    for (int index = 0; index < tensor.dims; ++index) {
        if (tensor.size[index] <= 0) {
            throw std::runtime_error(
                "YOLOv8 ONNX output contains a non-positive dimension");
        }
        shape.push_back(static_cast<std::size_t>(tensor.size[index]));
    }
    return shape;
}

std::string output_names_text(const std::vector<cv::String>& names) {
    std::ostringstream output;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << names[index];
    }
    return output.str();
}

}  // namespace

struct YoloV8OnnxHandLandmarkProvider::Impl {
    explicit Impl(YoloV8OnnxPipelineConfig pipeline_config)
        : config(validate_config(std::move(pipeline_config))) {
        try {
            net = cv::dnn::readNetFromONNX(config.model.string());
            if (net.empty()) {
                throw std::runtime_error("OpenCV DNN returned an empty network");
            }
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            output_names = net.getUnconnectedOutLayersNames();
        } catch (const cv::Exception& error) {
            throw std::runtime_error(
                "Unable to load YOLOv8 ONNX model '" + config.model.string() +
                "' with OpenCV DNN: " + error.what());
        }
        if (output_names.size() != 1) {
            throw std::runtime_error(
                "YOLOv8 Pose ONNX model must expose exactly one output; got " +
                std::to_string(output_names.size()) + " (" +
                output_names_text(output_names) + ")");
        }
    }

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) {
        if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
            throw std::invalid_argument(
                "YOLOv8 ONNX provider requires a non-empty CV_8UC3 BGR frame");
        }

        const auto letterbox = yolov8_pose::make_letterbox_transform(
            frame_bgr.cols, frame_bgr.rows, config.input_size,
            config.input_size);
        const cv::Mat letterboxed =
            prepare_letterboxed_bgr(frame_bgr, letterbox);
        const cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0,
            cv::Size(config.input_size, config.input_size), cv::Scalar(), true,
            false, CV_32F);

        std::vector<cv::Mat> outputs;
        try {
            net.setInput(blob);
            net.forward(outputs, output_names);
        } catch (const cv::Exception& error) {
            throw std::runtime_error(
                "YOLOv8 ONNX inference failed for '" +
                config.model.string() + "': " + error.what());
        }
        if (outputs.size() != 1 || outputs[0].empty()) {
            throw std::runtime_error(
                "YOLOv8 Pose ONNX inference did not return its single output");
        }
        cv::Mat output = outputs[0];
        if (output.depth() != CV_32F) {
            throw std::runtime_error(
                "YOLOv8 Pose ONNX output must contain float32 values");
        }
        if (!output.isContinuous()) {
            output = output.clone();
        }

        yolov8_pose::DecodeOptions options;
        options.input_width = config.input_size;
        options.input_height = config.input_size;
        options.min_score = config.min_detection_confidence;
        options.nms_iou_threshold = config.nms_iou_threshold;
        options.min_keypoint_visibility = config.min_keypoint_visibility;
        options.min_reliable_keypoints = config.min_reliable_keypoints;
        options.max_hands = static_cast<std::size_t>(config.max_num_hands);
        options.clip_to_source = true;
        const std::vector<yolov8_pose::HandPose> poses =
            yolov8_pose::decode_flat_pose(
                yolov8_pose::FloatTensorView{
                    output.ptr<float>(), output.total(), tensor_shape(output)},
                letterbox, options);

        std::vector<DetectedHand> detected;
        detected.reserve(poses.size());
        for (const yolov8_pose::HandPose& pose : poses) {
            const std::size_t reliable = static_cast<std::size_t>(
                std::count_if(
                    pose.keypoints.begin(), pose.keypoints.end(),
                    [](const yolov8_pose::Keypoint& point) {
                        return point.reliable;
                    }));
            if (reliable < config.min_reliable_keypoints) {
                continue;
            }

            Landmarks landmarks{};
            Landmarks metric_landmarks{};
            LandmarkConfidences confidences{};
            for (std::size_t index = 0; index < landmarks.size(); ++index) {
                const yolov8_pose::Keypoint& point = pose.keypoints[index];
                landmarks[index] = {
                    point.x / static_cast<double>(frame_bgr.cols),
                    point.y / static_cast<double>(frame_bgr.rows),
                    0.0,
                };
                metric_landmarks[index] = {point.x, point.y, 0.0};
                confidences[index] = point.visibility;
            }
            validate_landmarks(landmarks);
            const bool gesture_landmarks_reliable = std::all_of(
                kGeometryRequiredLandmarkIndices.begin(),
                kGeometryRequiredLandmarkIndices.end(),
                [&](int index) {
                    return pose.keypoints[static_cast<std::size_t>(index)]
                        .reliable;
                });
            detected.push_back(DetectedHand{
                landmarks,
                "Unknown",
                std::nullopt,
                false,
                confidences,
                HandBoundingBox{
                    pose.box.xmin / static_cast<double>(frame_bgr.cols),
                    pose.box.ymin / static_cast<double>(frame_bgr.rows),
                    pose.box.xmax / static_cast<double>(frame_bgr.cols),
                    pose.box.ymax / static_cast<double>(frame_bgr.rows),
                    pose.score,
                },
                metric_landmarks,
                gesture_landmarks_reliable,
            });
        }
        return detected;
    }

    YoloV8OnnxPipelineConfig config;
    cv::dnn::Net net;
    std::vector<cv::String> output_names;
};

YoloV8OnnxHandLandmarkProvider::YoloV8OnnxHandLandmarkProvider(
    YoloV8OnnxPipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

YoloV8OnnxHandLandmarkProvider::~YoloV8OnnxHandLandmarkProvider() = default;

LandmarkProviderInfo YoloV8OnnxHandLandmarkProvider::info() const {
    return {"yolov8-onnx", true, false};
}

std::vector<DetectedHand> YoloV8OnnxHandLandmarkProvider::detect(
    const cv::Mat& frame_bgr) {
    return impl_->detect(frame_bgr);
}

}  // namespace double_ok_gesture
