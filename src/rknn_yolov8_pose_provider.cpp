#include "double_ok_gesture/landmark_provider.hpp"

#include "double_ok_gesture/yolov8_pose_postprocess.hpp"

#include <rknn_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace double_ok_gesture {
namespace {

using yolov8_pose::FeatureMapLayout;
using yolov8_pose::FeatureMapView;
using yolov8_pose::FloatTensorView;
using yolov8_pose::KeypointTensorLayout;
using yolov8_pose::KeypointTensorView;

using TensorShape = std::vector<std::size_t>;

std::runtime_error rknn_error(
    const std::string& operation,
    int status,
    const std::filesystem::path& model_path) {
    std::ostringstream message;
    message << operation << " failed for '" << model_path.string()
            << "' (RKNN status " << status << ')';
    return std::runtime_error(message.str());
}

std::vector<std::uint8_t> read_model(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open YOLOv8 RKNN model: " + path.string());
    }
    const std::streampos end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error("YOLOv8 RKNN model is empty: " + path.string());
    }
    const auto size = static_cast<std::uintmax_t>(end);
    if (size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("YOLOv8 RKNN model exceeds rknn_init size limit: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error("Unable to read complete YOLOv8 RKNN model: " + path.string());
    }
    return bytes;
}

TensorShape tensor_shape(const rknn_tensor_attr& attr) {
    if (attr.n_dims == 0 || attr.n_dims > RKNN_MAX_DIMS) {
        return {};
    }
    TensorShape shape;
    shape.reserve(attr.n_dims);
    for (std::uint32_t index = 0; index < attr.n_dims; ++index) {
        shape.push_back(attr.dims[index]);
    }
    return shape;
}

std::string shape_text(const TensorShape& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            output << ',';
        }
        output << shape[index];
    }
    output << ']';
    return output.str();
}

std::string tensor_name(const rknn_tensor_attr& attr) {
    const char* end = std::find(attr.name, attr.name + sizeof(attr.name), '\0');
    return std::string(attr.name, end);
}

bool supported_runtime_tensor_type(rknn_tensor_type type) {
    return type == RKNN_TENSOR_FLOAT32 || type == RKNN_TENSOR_FLOAT16 ||
           type == RKNN_TENSOR_INT8 || type == RKNN_TENSOR_UINT8 ||
           type == RKNN_TENSOR_INT16;
}

void log_tensor_contract(
    const std::filesystem::path& path,
    const char* kind,
    const rknn_tensor_attr& attr) {
    std::clog << "YOLOv8 RKNN tensor: model=" << path.string() << ' ' << kind << '['
              << attr.index << "] name=" << tensor_name(attr)
              << " shape=" << shape_text(tensor_shape(attr))
              << " fmt=" << static_cast<int>(attr.fmt)
              << " type=" << static_cast<int>(attr.type)
              << " qnt_type=" << static_cast<int>(attr.qnt_type)
              << " zp=" << attr.zp << " scale=" << attr.scale << '\n';
}

void require_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and in [0,1]");
    }
}

YoloV8RknnPipelineConfig validate_config(YoloV8RknnPipelineConfig config) {
    if (config.model.empty()) {
        throw std::invalid_argument("YOLOv8 RKNN model path must not be empty");
    }
    if (config.input_size < 32 || config.input_size % 32 != 0) {
        throw std::invalid_argument("YOLOv8 input size must be at least 32 and divisible by 32");
    }
    if (config.max_num_hands < 1 || config.max_num_hands > 2) {
        throw std::invalid_argument("YOLOv8 max_num_hands must be 1 or 2");
    }
    require_probability(config.min_detection_confidence, "min_detection_confidence");
    require_probability(config.min_keypoint_visibility, "min_keypoint_visibility");
    require_probability(config.nms_iou_threshold, "nms_iou_threshold");
    if (config.min_reliable_keypoints > yolov8_pose::kHandKeypointCount) {
        throw std::invalid_argument("min_reliable_keypoints must be at most 21");
    }
    return config;
}

struct FeatureOutputContract {
    std::size_t output_index = 0;
    TensorShape shape;
    FeatureMapLayout layout = FeatureMapLayout::kNchw;
    std::size_t grid_height = 0;
    std::size_t grid_width = 0;
};

struct KeypointOutputContract {
    std::size_t output_index = 0;
    TensorShape shape;
    KeypointTensorLayout layout = KeypointTensorLayout::kKeypointDimensionCandidates;
};

std::optional<FeatureOutputContract> feature_contract(
    std::size_t output_index,
    const TensorShape& shape) {
    if (shape.size() != 4 || shape[0] != 1) {
        return std::nullopt;
    }
    if (shape[1] == yolov8_pose::kDetectionChannelCount && shape[2] > 0 && shape[3] > 0) {
        return FeatureOutputContract{
            output_index,
            shape,
            FeatureMapLayout::kNchw,
            shape[2],
            shape[3],
        };
    }
    if (shape[3] == yolov8_pose::kDetectionChannelCount && shape[1] > 0 && shape[2] > 0) {
        return FeatureOutputContract{
            output_index,
            shape,
            FeatureMapLayout::kNhwc,
            shape[1],
            shape[2],
        };
    }
    return std::nullopt;
}

std::optional<KeypointOutputContract> keypoint_contract(
    std::size_t output_index,
    const TensorShape& shape,
    std::size_t candidate_count) {
    if (shape == TensorShape{1, yolov8_pose::kHandKeypointCount,
                             yolov8_pose::kKeypointDimensionCount, candidate_count}) {
        return KeypointOutputContract{
            output_index,
            shape,
            KeypointTensorLayout::kKeypointDimensionCandidates,
        };
    }
    if (shape == TensorShape{1, candidate_count, yolov8_pose::kHandKeypointCount,
                             yolov8_pose::kKeypointDimensionCount}) {
        return KeypointOutputContract{
            output_index,
            shape,
            KeypointTensorLayout::kCandidatesKeypointDimension,
        };
    }
    if (shape == TensorShape{1, yolov8_pose::kHandKeypointCount *
                                    yolov8_pose::kKeypointDimensionCount,
                             candidate_count}) {
        return KeypointOutputContract{
            output_index,
            shape,
            KeypointTensorLayout::kChannelsCandidates,
        };
    }
    if (shape == TensorShape{1, candidate_count,
                             yolov8_pose::kHandKeypointCount *
                                 yolov8_pose::kKeypointDimensionCount}) {
        return KeypointOutputContract{
            output_index,
            shape,
            KeypointTensorLayout::kCandidatesChannels,
        };
    }
    return std::nullopt;
}

class RknnYoloV8PoseModel {
public:
    RknnYoloV8PoseModel(std::filesystem::path path, int input_size)
        : path_(std::move(path)), input_size_(input_size) {
        const std::vector<std::uint8_t> model = read_model(path_);
        const int status = rknn_init(
            &context_,
            const_cast<std::uint8_t*>(model.data()),
            static_cast<std::uint32_t>(model.size()),
            0,
            nullptr);
        if (status != RKNN_SUCC) {
            context_ = 0;
            throw rknn_error("rknn_init", status, path_);
        }
        try {
            query_and_validate_contract();
        } catch (...) {
            rknn_destroy(context_);
            context_ = 0;
            throw;
        }
    }

    ~RknnYoloV8PoseModel() {
        if (context_ != 0) {
            rknn_destroy(context_);
        }
    }

    RknnYoloV8PoseModel(const RknnYoloV8PoseModel&) = delete;
    RknnYoloV8PoseModel& operator=(const RknnYoloV8PoseModel&) = delete;

    std::vector<std::vector<float>> infer(const cv::Mat& rgb) {
        if (rgb.empty() || rgb.type() != CV_8UC3 || rgb.rows != input_size_ ||
            rgb.cols != input_size_) {
            throw std::invalid_argument("YOLOv8 RKNN input must be fixed-size CV_8UC3 RGB");
        }
        const cv::Mat contiguous = rgb.isContinuous() ? rgb : rgb.clone();
        const std::size_t input_bytes = contiguous.total() * contiguous.elemSize();
        if (input_bytes > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("YOLOv8 RKNN input exceeds API size limit");
        }

        rknn_input input{};
        input.index = 0;
        input.buf = const_cast<unsigned char*>(contiguous.ptr<unsigned char>());
        input.size = static_cast<std::uint32_t>(input_bytes);
        input.pass_through = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;

        int status = rknn_inputs_set(context_, 1, &input);
        if (status != RKNN_SUCC) {
            throw rknn_error("rknn_inputs_set", status, path_);
        }
        status = rknn_run(context_, nullptr);
        if (status != RKNN_SUCC) {
            throw rknn_error("rknn_run", status, path_);
        }

        std::vector<rknn_output> outputs(output_attrs_.size());
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            outputs[index].index = static_cast<std::uint32_t>(index);
            outputs[index].want_float = 1;
            outputs[index].is_prealloc = 0;
        }
        status = rknn_outputs_get(
            context_, static_cast<std::uint32_t>(outputs.size()), outputs.data(), nullptr);
        if (status != RKNN_SUCC) {
            throw rknn_error("rknn_outputs_get", status, path_);
        }

        struct OutputReleaseGuard {
            rknn_context context;
            std::vector<rknn_output>& outputs;
            ~OutputReleaseGuard() {
                rknn_outputs_release(
                    context, static_cast<std::uint32_t>(outputs.size()), outputs.data());
            }
        } release{context_, outputs};

        std::vector<std::vector<float>> copied;
        copied.reserve(outputs.size());
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const std::size_t values = output_attrs_[index].n_elems;
            const std::size_t bytes = values * sizeof(float);
            if (!outputs[index].buf || outputs[index].size < bytes) {
                throw std::runtime_error(
                    "RKNN returned undersized float output " + std::to_string(index) +
                    " for '" + path_.string() + "'");
            }
            const auto* begin = static_cast<const float*>(outputs[index].buf);
            copied.emplace_back(begin, begin + values);
        }
        return copied;
    }

    const std::array<FeatureOutputContract, 3>& features() const {
        return features_;
    }

    const KeypointOutputContract& keypoints() const {
        return keypoints_;
    }

private:
    void query_and_validate_contract() {
        rknn_input_output_num counts{};
        int status = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts));
        if (status != RKNN_SUCC) {
            throw rknn_error("rknn_query(RKNN_QUERY_IN_OUT_NUM)", status, path_);
        }
        if (counts.n_input != 1 || counts.n_output != 4) {
            std::ostringstream message;
            message << "Unexpected YOLOv8 RKOPT tensor count in '" << path_.string()
                    << "': expected 1 input and 4 outputs, got " << counts.n_input
                    << " and " << counts.n_output;
            throw std::runtime_error(message.str());
        }

        rknn_tensor_attr input_attr{};
        input_attr.index = 0;
        status = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
        if (status != RKNN_SUCC) {
            throw rknn_error("rknn_query(RKNN_QUERY_INPUT_ATTR)", status, path_);
        }
        const TensorShape input_shape = tensor_shape(input_attr);
        const TensorShape expected_nhwc{
            1,
            static_cast<std::size_t>(input_size_),
            static_cast<std::size_t>(input_size_),
            3,
        };
        const TensorShape expected_nchw{
            1,
            3,
            static_cast<std::size_t>(input_size_),
            static_cast<std::size_t>(input_size_),
        };
        const std::size_t expected_values =
            static_cast<std::size_t>(input_size_) * input_size_ * 3;
        if ((input_shape != expected_nhwc && input_shape != expected_nchw) ||
            input_attr.n_elems != expected_values ||
            !supported_runtime_tensor_type(input_attr.type) ||
            (input_attr.fmt != RKNN_TENSOR_NHWC &&
             input_attr.fmt != RKNN_TENSOR_NCHW)) {
            throw std::runtime_error(
                "Unexpected YOLOv8 RKNN input contract: " + shape_text(input_shape));
        }
        log_tensor_contract(path_, "input", input_attr);

        output_attrs_.resize(counts.n_output);
        std::vector<FeatureOutputContract> detected_features;
        std::vector<std::pair<std::size_t, TensorShape>> other_outputs;
        for (std::uint32_t index = 0; index < counts.n_output; ++index) {
            rknn_tensor_attr attr{};
            attr.index = index;
            status = rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (status != RKNN_SUCC) {
                throw rknn_error("rknn_query(RKNN_QUERY_OUTPUT_ATTR)", status, path_);
            }
            if (!supported_runtime_tensor_type(attr.type) || attr.n_elems == 0) {
                throw std::runtime_error(
                    "Unsupported YOLOv8 RKNN output type at index " + std::to_string(index));
            }
            output_attrs_[index] = attr;
            log_tensor_contract(path_, "output", attr);
            const TensorShape shape = tensor_shape(attr);
            if (const auto feature = feature_contract(index, shape)) {
                detected_features.push_back(*feature);
            } else {
                other_outputs.emplace_back(index, shape);
            }
        }
        if (detected_features.size() != 3 || other_outputs.size() != 1) {
            throw std::runtime_error(
                "YOLOv8 RKOPT outputs must contain three 65-channel feature maps and one keypoint tensor");
        }

        std::sort(
            detected_features.begin(),
            detected_features.end(),
            [](const FeatureOutputContract& lhs, const FeatureOutputContract& rhs) {
                return lhs.grid_height * lhs.grid_width > rhs.grid_height * rhs.grid_width;
            });
        for (std::size_t index = 0; index < features_.size(); ++index) {
            features_[index] = detected_features[index];
        }

        const std::size_t candidate_count = yolov8_pose::candidate_count_for_input(
            input_size_, input_size_);
        const auto keypoint = keypoint_contract(
            other_outputs[0].first, other_outputs[0].second, candidate_count);
        if (!keypoint) {
            throw std::runtime_error(
                "Unexpected YOLOv8 RKOPT keypoint shape " + shape_text(other_outputs[0].second) +
                "; expected 21x3x" + std::to_string(candidate_count));
        }
        keypoints_ = *keypoint;

        std::array<FeatureMapView, 3> views{};
        for (std::size_t index = 0; index < views.size(); ++index) {
            views[index].layout = features_[index].layout;
            views[index].tensor.shape = features_[index].shape;
            views[index].tensor.value_count = output_attrs_[features_[index].output_index].n_elems;
        }
        (void)yolov8_pose::validate_three_scale_contract(
            views, input_size_, input_size_);
    }

    std::filesystem::path path_;
    int input_size_ = 0;
    rknn_context context_ = 0;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::array<FeatureOutputContract, 3> features_{};
    KeypointOutputContract keypoints_{};
};

cv::Mat prepare_letterboxed_rgb(
    const cv::Mat& frame_bgr,
    const yolov8_pose::LetterboxTransform& transform) {
    if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
        throw std::invalid_argument("YOLOv8 provider requires a non-empty CV_8UC3 BGR frame");
    }
    const int resized_width = static_cast<int>(std::lround(
        transform.scale_x * static_cast<double>(transform.source_width)));
    const int resized_height = static_cast<int>(std::lround(
        transform.scale_y * static_cast<double>(transform.source_height)));
    const int pad_left = static_cast<int>(std::lround(transform.pad_left));
    const int pad_top = static_cast<int>(std::lround(transform.pad_top));
    if (resized_width <= 0 || resized_height <= 0 || pad_left < 0 || pad_top < 0 ||
        pad_left + resized_width > transform.input_width ||
        pad_top + resized_height > transform.input_height) {
        throw std::runtime_error("Invalid YOLOv8 letterbox geometry");
    }

    cv::Mat resized;
    cv::resize(
        frame_bgr,
        resized,
        cv::Size(resized_width, resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::Mat canvas(
        transform.input_height,
        transform.input_width,
        CV_8UC3,
        cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_left, pad_top, resized_width, resized_height)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

}  // namespace

struct YoloV8RknnHandLandmarkProvider::Impl {
    explicit Impl(YoloV8RknnPipelineConfig requested)
        : config(validate_config(std::move(requested))),
          model(config.model, config.input_size) {
        std::clog
            << "YOLOv8 RKNN provider uses 2D landmarks: visibility is retained separately, "
               "z=0, handedness=Unknown\n";
    }

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) {
        const auto letterbox = yolov8_pose::make_letterbox_transform(
            frame_bgr.cols,
            frame_bgr.rows,
            config.input_size,
            config.input_size);
        const cv::Mat rgb = prepare_letterboxed_rgb(frame_bgr, letterbox);
        const std::vector<std::vector<float>> outputs = model.infer(rgb);

        std::array<FeatureMapView, 3> features{};
        const auto& feature_contracts = model.features();
        for (std::size_t index = 0; index < features.size(); ++index) {
            const FeatureOutputContract& contract = feature_contracts[index];
            const std::vector<float>& values = outputs[contract.output_index];
            features[index] = FeatureMapView{
                FloatTensorView{values.data(), values.size(), contract.shape},
                contract.layout,
            };
        }
        const KeypointOutputContract& keypoint_contract_value = model.keypoints();
        const std::vector<float>& keypoint_values = outputs[keypoint_contract_value.output_index];
        const KeypointTensorView keypoints{
            FloatTensorView{
                keypoint_values.data(),
                keypoint_values.size(),
                keypoint_contract_value.shape,
            },
            keypoint_contract_value.layout,
            yolov8_pose::KeypointCoordinateEncoding::kDecodedPixelsAndVisibility,
        };

        yolov8_pose::DecodeOptions options;
        options.input_width = config.input_size;
        options.input_height = config.input_size;
        options.min_score = config.min_detection_confidence;
        options.nms_iou_threshold = config.nms_iou_threshold;
        options.min_keypoint_visibility = config.min_keypoint_visibility;
        options.max_hands = static_cast<std::size_t>(config.max_num_hands);
        options.clip_to_source = true;
        const std::vector<yolov8_pose::HandPose> poses =
            yolov8_pose::decode_three_scale_pose(features, keypoints, letterbox, options);

        std::vector<DetectedHand> detected;
        detected.reserve(poses.size());
        for (const yolov8_pose::HandPose& pose : poses) {
            const std::size_t reliable = static_cast<std::size_t>(std::count_if(
                pose.keypoints.begin(),
                pose.keypoints.end(),
                [](const yolov8_pose::Keypoint& point) { return point.reliable; }));
            if (reliable < config.min_reliable_keypoints) {
                continue;
            }

            Landmarks landmarks{};
            LandmarkConfidences confidences{};
            for (std::size_t index = 0; index < landmarks.size(); ++index) {
                const yolov8_pose::Keypoint& point = pose.keypoints[index];
                landmarks[index] = {
                    point.x / static_cast<double>(frame_bgr.cols),
                    point.y / static_cast<double>(frame_bgr.rows),
                    0.0,
                };
                confidences[index] = point.visibility;
            }
            validate_landmarks(landmarks);
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
            });
        }
        return detected;
    }

    YoloV8RknnPipelineConfig config;
    RknnYoloV8PoseModel model;
};

YoloV8RknnHandLandmarkProvider::YoloV8RknnHandLandmarkProvider(
    YoloV8RknnPipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

YoloV8RknnHandLandmarkProvider::~YoloV8RknnHandLandmarkProvider() = default;

LandmarkProviderInfo YoloV8RknnHandLandmarkProvider::info() const {
    return {"yolov8-rknn", true, false};
}

std::vector<DetectedHand> YoloV8RknnHandLandmarkProvider::detect(
    const cv::Mat& frame_bgr) {
    return impl_->detect(frame_bgr);
}

}  // namespace double_ok_gesture
