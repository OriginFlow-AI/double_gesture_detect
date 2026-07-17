#include "double_ok_gesture/landmark_provider.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace double_ok_gesture {
namespace {

constexpr int kPalmInputSize = 192;
constexpr int kHandInputSize = 224;
constexpr std::size_t kPalmAnchorCount = 2016;
constexpr std::size_t kPalmValueCount = 18;
constexpr std::size_t kHandLandmarkValueCount = 63;

struct PalmDetection {
    cv::Rect2d box;
    std::array<cv::Point2d, 7> landmarks{};
    double score = 0.0;
};

struct CropResult {
    cv::Mat image;
    std::array<cv::Point2d, 2> box{};
    cv::Point2d bias;
};

struct HandInputTransform {
    cv::Mat input;
    std::array<cv::Point2d, 2> rotated_palm_box{};
    double angle_degrees = 0.0;
    cv::Matx23d rotation;
    cv::Point2d pad_bias;
};

class InvalidPalmCrop : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and in [0,1]");
    }
}

void require_onnx_model(
    const std::filesystem::path& path,
    const char* name) {
    if (path.empty() || path.extension() != ".onnx") {
        throw std::invalid_argument(
            std::string(name) + " must point to an .onnx file");
    }
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) == 0) {
        throw std::runtime_error(
            std::string(name) + " not found or empty: " + path.string());
    }
}

MediaPipeOnnxPipelineConfig validate_config(
    MediaPipeOnnxPipelineConfig config) {
    require_onnx_model(config.palm_model, "palm ONNX model");
    require_onnx_model(config.hand_model, "hand landmark ONNX model");
    if (config.max_num_hands < 1 || config.max_num_hands > 2) {
        throw std::invalid_argument("max_num_hands must be 1 or 2");
    }
    require_probability(
        config.palm_detection_threshold, "palm_detection_threshold");
    require_probability(
        config.hand_presence_threshold, "hand_presence_threshold");
    require_probability(config.palm_nms_threshold, "palm_nms_threshold");
    return config;
}

cv::dnn::Net load_network(
    const std::filesystem::path& path,
    const char* name) {
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(path.string());
        if (net.empty()) {
            throw std::runtime_error("OpenCV DNN returned an empty network");
        }
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return net;
    } catch (const cv::Exception& error) {
        throw std::runtime_error(
            std::string("Unable to load ") + name + " '" + path.string() +
            "' with OpenCV DNN: " + error.what());
    }
}

cv::Mat nhwc_float_input(const cv::Mat& rgb) {
    if (rgb.empty() || rgb.type() != CV_8UC3 || !rgb.isContinuous()) {
        throw std::invalid_argument(
            "ONNX preprocessing requires a continuous CV_8UC3 image");
    }
    cv::Mat values;
    rgb.convertTo(values, CV_32FC3, 1.0 / 255.0);
    const int shape[] = {1, values.rows, values.cols, 3};
    return values.reshape(1, 4, shape);
}

std::vector<cv::Mat> forward(
    cv::dnn::Net& net,
    const std::vector<cv::String>& names,
    const cv::Mat& input,
    const std::string& label) {
    try {
        net.setInput(input);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, names);
        return outputs;
    } catch (const cv::Exception& error) {
        throw std::runtime_error(label + " ONNX inference failed: " + error.what());
    }
}

std::vector<cv::Point2d> make_palm_anchors() {
    std::vector<cv::Point2d> anchors;
    anchors.reserve(kPalmAnchorCount);
    const auto append_grid = [&](int side, int anchors_per_cell) {
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                const cv::Point2d center(
                    (static_cast<double>(x) + 0.5) /
                        static_cast<double>(side),
                    (static_cast<double>(y) + 0.5) /
                        static_cast<double>(side));
                for (int index = 0; index < anchors_per_cell; ++index) {
                    anchors.push_back(center);
                }
            }
        }
    };
    append_grid(24, 2);
    append_grid(12, 6);
    if (anchors.size() != kPalmAnchorCount) {
        throw std::logic_error("invalid MediaPipe palm anchor count");
    }
    return anchors;
}

double sigmoid(double value) {
    value = std::clamp(value, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(-value));
}

double intersection_over_union(
    const cv::Rect2d& lhs,
    const cv::Rect2d& rhs) {
    const double x1 = std::max(lhs.x, rhs.x);
    const double y1 = std::max(lhs.y, rhs.y);
    const double x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const double y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const double intersection =
        std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double total = lhs.area() + rhs.area() - intersection;
    return total > 0.0 ? intersection / total : 0.0;
}

std::vector<PalmDetection> suppress_overlapping_palms(
    std::vector<PalmDetection> candidates,
    double iou_threshold,
    int maximum) {
    std::sort(
        candidates.begin(), candidates.end(),
        [](const PalmDetection& lhs, const PalmDetection& rhs) {
            return lhs.score > rhs.score;
        });
    std::vector<PalmDetection> selected;
    for (const PalmDetection& candidate : candidates) {
        const bool overlaps = std::any_of(
            selected.begin(), selected.end(),
            [&](const PalmDetection& kept) {
                return intersection_over_union(candidate.box, kept.box) >
                       iou_threshold;
            });
        if (!overlaps) {
            selected.push_back(candidate);
            if (selected.size() >= static_cast<std::size_t>(maximum)) {
                break;
            }
        }
    }
    return selected;
}

const cv::Mat& palm_regression_output(
    const std::vector<cv::Mat>& outputs) {
    for (const cv::Mat& output : outputs) {
        if (output.total() == kPalmAnchorCount * kPalmValueCount) {
            return output;
        }
    }
    throw std::runtime_error(
        "palm ONNX model must output [1,2016,18] regressions");
}

const cv::Mat& palm_score_output(const std::vector<cv::Mat>& outputs) {
    for (const cv::Mat& output : outputs) {
        if (output.total() == kPalmAnchorCount) {
            return output;
        }
    }
    throw std::runtime_error(
        "palm ONNX model must output [1,2016,1] scores");
}

std::vector<PalmDetection> detect_palms(
    cv::dnn::Net& net,
    const std::vector<cv::String>& output_names,
    const std::vector<cv::Point2d>& anchors,
    const cv::Mat& frame_bgr,
    const MediaPipeOnnxPipelineConfig& config) {
    const double resize_scale = std::min(
        static_cast<double>(kPalmInputSize) / frame_bgr.cols,
        static_cast<double>(kPalmInputSize) / frame_bgr.rows);
    const int resized_width = std::max(
        1, static_cast<int>(frame_bgr.cols * resize_scale));
    const int resized_height = std::max(
        1, static_cast<int>(frame_bgr.rows * resize_scale));
    cv::Mat resized;
    cv::resize(
        frame_bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
        cv::INTER_LINEAR);
    const int pad_width = kPalmInputSize - resized_width;
    const int pad_height = kPalmInputSize - resized_height;
    const int left = pad_width / 2;
    const int top = pad_height / 2;
    cv::Mat square;
    cv::copyMakeBorder(
        resized, square, top, pad_height - top, left, pad_width - left,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::cvtColor(square, square, cv::COLOR_BGR2RGB);

    const std::vector<cv::Mat> outputs = forward(
        net, output_names, nhwc_float_input(square), "palm detection");
    cv::Mat regression = palm_regression_output(outputs);
    cv::Mat scores = palm_score_output(outputs);
    if (regression.depth() != CV_32F || scores.depth() != CV_32F) {
        throw std::runtime_error("palm ONNX outputs must be float32");
    }
    if (!regression.isContinuous()) {
        regression = regression.clone();
    }
    if (!scores.isContinuous()) {
        scores = scores.clone();
    }

    const double image_scale =
        static_cast<double>(std::max(frame_bgr.cols, frame_bgr.rows));
    const cv::Point2d pad_bias(
        static_cast<int>(static_cast<double>(left) / resize_scale),
        static_cast<int>(static_cast<double>(top) / resize_scale));
    const float* regression_values = regression.ptr<float>();
    const float* score_values = scores.ptr<float>();
    std::vector<PalmDetection> candidates;
    candidates.reserve(32);
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        const double score = sigmoid(score_values[index]);
        if (score < config.palm_detection_threshold) {
            continue;
        }
        const float* values =
            regression_values + index * kPalmValueCount;
        const cv::Point2d center(
            (values[0] / kPalmInputSize + anchors[index].x) * image_scale -
                pad_bias.x,
            (values[1] / kPalmInputSize + anchors[index].y) * image_scale -
                pad_bias.y);
        const cv::Size2d size(
            std::max(0.0, values[2] / kPalmInputSize * image_scale),
            std::max(0.0, values[3] / kPalmInputSize * image_scale));
        if (size.width <= 0.0 || size.height <= 0.0) {
            continue;
        }
        PalmDetection palm;
        palm.box = {
            center.x - size.width / 2.0,
            center.y - size.height / 2.0,
            size.width,
            size.height,
        };
        for (std::size_t point = 0; point < palm.landmarks.size(); ++point) {
            palm.landmarks[point] = {
                (values[4 + point * 2] / kPalmInputSize +
                 anchors[index].x) * image_scale - pad_bias.x,
                (values[5 + point * 2] / kPalmInputSize +
                 anchors[index].y) * image_scale - pad_bias.y,
            };
        }
        palm.score = score;
        candidates.push_back(palm);
    }
    return suppress_overlapping_palms(
        std::move(candidates), config.palm_nms_threshold,
        config.max_num_hands);
}

CropResult crop_and_pad(
    const cv::Mat& image,
    std::array<cv::Point2d, 2> box,
    bool for_rotation) {
    const cv::Point2d size = box[1] - box[0];
    const cv::Point2d shift = for_rotation
        ? cv::Point2d(0.0, 0.0)
        : cv::Point2d(0.0, -0.4 * size.y);
    box[0] += shift;
    box[1] += shift;
    const cv::Point2d center = (box[0] + box[1]) * 0.5;
    const cv::Point2d shifted_size = box[1] - box[0];
    const double enlarge = for_rotation ? 4.0 : 3.0;
    const cv::Point2d half_size = shifted_size * (enlarge * 0.5);

    int x1 = std::clamp(
        static_cast<int>(center.x - half_size.x), 0, image.cols);
    int y1 = std::clamp(
        static_cast<int>(center.y - half_size.y), 0, image.rows);
    int x2 = std::clamp(
        static_cast<int>(center.x + half_size.x), 0, image.cols);
    int y2 = std::clamp(
        static_cast<int>(center.y + half_size.y), 0, image.rows);
    if (x2 <= x1 || y2 <= y1) {
        throw InvalidPalmCrop("palm detection produced an empty hand crop");
    }

    cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    const int side = std::max(
        1,
        for_rotation
            ? static_cast<int>(std::hypot(crop.rows, crop.cols))
            : std::max(crop.rows, crop.cols));
    const int pad_width = side - crop.cols;
    const int pad_height = side - crop.rows;
    const int left = pad_width / 2;
    const int top = pad_height / 2;
    cv::Mat padded;
    cv::copyMakeBorder(
        crop, padded, top, pad_height - top, left, pad_width - left,
        cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0, 0, 0));
    return {
        padded,
        {cv::Point2d(x1, y1), cv::Point2d(x2, y2)},
        cv::Point2d(x1 - left, y1 - top),
    };
}

cv::Point2d transform_point(
    const cv::Point2d& point,
    const cv::Matx23d& matrix) {
    return {
        point.x * matrix(0, 0) + point.y * matrix(0, 1) + matrix(0, 2),
        point.x * matrix(1, 0) + point.y * matrix(1, 1) + matrix(1, 2),
    };
}

HandInputTransform prepare_hand_input(
    const cv::Mat& frame_bgr,
    const PalmDetection& palm) {
    std::array<cv::Point2d, 2> palm_box = {
        cv::Point2d(palm.box.x, palm.box.y),
        cv::Point2d(
            palm.box.x + palm.box.width, palm.box.y + palm.box.height),
    };
    CropResult first = crop_and_pad(frame_bgr, palm_box, true);
    cv::cvtColor(first.image, first.image, cv::COLOR_BGR2RGB);

    std::array<cv::Point2d, 2> local_box = first.box;
    local_box[0] -= first.bias;
    local_box[1] -= first.bias;
    std::array<cv::Point2d, 7> local_landmarks = palm.landmarks;
    for (cv::Point2d& point : local_landmarks) {
        point -= first.bias;
    }
    const cv::Point2d& base = local_landmarks[0];
    const cv::Point2d& middle_finger_base = local_landmarks[2];
    double radians =
        M_PI / 2.0 -
        std::atan2(
            -(middle_finger_base.y - base.y),
            middle_finger_base.x - base.x);
    radians -= 2.0 * M_PI * std::floor((radians + M_PI) / (2.0 * M_PI));
    const double angle = radians * 180.0 / M_PI;
    const cv::Point2d center = (local_box[0] + local_box[1]) * 0.5;
    const cv::Mat rotation_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Matx23d rotation;
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotation(row, column) = rotation_mat.at<double>(row, column);
        }
    }
    cv::Mat rotated;
    cv::warpAffine(
        first.image, rotated, rotation, first.image.size(), cv::INTER_LINEAR,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Point2d minimum(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    cv::Point2d maximum(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    for (const cv::Point2d& point : local_landmarks) {
        const cv::Point2d transformed = transform_point(point, rotation);
        minimum.x = std::min(minimum.x, transformed.x);
        minimum.y = std::min(minimum.y, transformed.y);
        maximum.x = std::max(maximum.x, transformed.x);
        maximum.y = std::max(maximum.y, transformed.y);
    }
    CropResult second = crop_and_pad(rotated, {minimum, maximum}, false);
    cv::Mat resized;
    cv::resize(
        second.image, resized, cv::Size(kHandInputSize, kHandInputSize), 0.0,
        0.0, cv::INTER_AREA);
    return {
        nhwc_float_input(resized),
        second.box,
        angle,
        rotation,
        first.bias,
    };
}

void validate_hand_outputs(const std::vector<cv::Mat>& outputs) {
    if (outputs.size() != 4 ||
        outputs[0].total() != kHandLandmarkValueCount ||
        outputs[1].total() != 1 || outputs[2].total() != 1 ||
        outputs[3].total() != kHandLandmarkValueCount) {
        throw std::runtime_error(
            "hand landmark ONNX model must output [63], [1], [1], [63]");
    }
    if (std::any_of(
            outputs.begin(), outputs.end(),
            [](const cv::Mat& output) {
                return output.empty() || output.depth() != CV_32F;
            })) {
        throw std::runtime_error("hand landmark ONNX outputs must be float32");
    }
}

std::optional<DetectedHand> estimate_hand(
    cv::dnn::Net& net,
    const std::vector<cv::String>& output_names,
    const cv::Mat& frame_bgr,
    const PalmDetection& palm,
    const MediaPipeOnnxPipelineConfig& config) {
    const HandInputTransform transform = prepare_hand_input(frame_bgr, palm);
    std::vector<cv::Mat> outputs = forward(
        net, output_names, transform.input, "hand landmark");
    validate_hand_outputs(outputs);
    for (cv::Mat& output : outputs) {
        if (!output.isContinuous()) {
            output = output.clone();
        }
    }
    const double presence = outputs[1].ptr<float>()[0];
    if (!std::isfinite(presence) ||
        presence < config.hand_presence_threshold) {
        return std::nullopt;
    }
    const double handedness_value = outputs[2].ptr<float>()[0];
    if (!std::isfinite(handedness_value)) {
        throw std::runtime_error("handedness ONNX output is not finite");
    }

    const cv::Point2d palm_size =
        transform.rotated_palm_box[1] - transform.rotated_palm_box[0];
    const double scale =
        std::max(palm_size.x, palm_size.y) / kHandInputSize;
    const double radians = transform.angle_degrees * M_PI / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);
    const cv::Matx22d coordinate_rotation(
        cosine, sine,
        -sine, cosine);

    const cv::Matx22d inverse_rotation(
        transform.rotation(0, 0), transform.rotation(1, 0),
        transform.rotation(0, 1), transform.rotation(1, 1));
    const cv::Vec2d translation(
        transform.rotation(0, 2), transform.rotation(1, 2));
    const cv::Vec2d inverse_translation(
        -(inverse_rotation(0, 0) * translation[0] +
          inverse_rotation(0, 1) * translation[1]),
        -(inverse_rotation(1, 0) * translation[0] +
          inverse_rotation(1, 1) * translation[1]));
    const cv::Point2d rotated_center =
        (transform.rotated_palm_box[0] + transform.rotated_palm_box[1]) * 0.5;
    const cv::Point2d original_center(
        rotated_center.x * inverse_rotation(0, 0) +
            rotated_center.y * inverse_rotation(0, 1) +
            inverse_translation[0],
        rotated_center.x * inverse_rotation(1, 0) +
            rotated_center.y * inverse_rotation(1, 1) +
            inverse_translation[1]);

    const float* raw = outputs[0].ptr<float>();
    Landmarks normalized{};
    Landmarks metric{};
    cv::Point2d minimum(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    cv::Point2d maximum(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    for (std::size_t index = 0; index < normalized.size(); ++index) {
        const double x =
            (static_cast<double>(raw[index * 3]) - kHandInputSize / 2.0) *
            scale;
        const double y =
            (static_cast<double>(raw[index * 3 + 1]) -
             kHandInputSize / 2.0) * scale;
        const double z = static_cast<double>(raw[index * 3 + 2]) * scale;
        const cv::Point2d rotated(
            x * coordinate_rotation(0, 0) +
                y * coordinate_rotation(1, 0),
            x * coordinate_rotation(0, 1) +
                y * coordinate_rotation(1, 1));
        const cv::Point2d screen =
            rotated + original_center + transform.pad_bias;
        metric[index] = {screen.x, screen.y, z};
        normalized[index] = {
            screen.x / frame_bgr.cols,
            screen.y / frame_bgr.rows,
            z / std::max(frame_bgr.cols, frame_bgr.rows),
        };
        minimum.x = std::min(minimum.x, screen.x);
        minimum.y = std::min(minimum.y, screen.y);
        maximum.x = std::max(maximum.x, screen.x);
        maximum.y = std::max(maximum.y, screen.y);
    }
    validate_landmarks(normalized);

    cv::Point2d hand_size = maximum - minimum;
    cv::Point2d box_center = (minimum + maximum) * 0.5;
    box_center.y -= 0.1 * hand_size.y;
    const cv::Point2d half = hand_size * (1.65 * 0.5);
    const double xmin = std::clamp(box_center.x - half.x, 0.0,
                                   static_cast<double>(frame_bgr.cols));
    const double ymin = std::clamp(box_center.y - half.y, 0.0,
                                   static_cast<double>(frame_bgr.rows));
    const double xmax = std::clamp(box_center.x + half.x, 0.0,
                                   static_cast<double>(frame_bgr.cols));
    const double ymax = std::clamp(box_center.y + half.y, 0.0,
                                   static_cast<double>(frame_bgr.rows));

    const double right_probability = std::clamp(handedness_value, 0.0, 1.0);
    bool is_right = right_probability >= 0.5;
    if (!config.input_mirrored) {
        is_right = !is_right;
    }
    const double handedness_confidence =
        std::max(right_probability, 1.0 - right_probability);
    const std::string handedness = handedness_confidence >= 0.65
        ? (is_right ? "Right" : "Left")
        : "Unknown";
    LandmarkConfidences confidences{};
    confidences.fill(presence);

    return DetectedHand{
        normalized,
        handedness,
        std::nullopt,
        false,
        confidences,
        HandBoundingBox{
            xmin / frame_bgr.cols,
            ymin / frame_bgr.rows,
            xmax / frame_bgr.cols,
            ymax / frame_bgr.rows,
            std::min(palm.score, presence),
        },
        metric,
        true,
        handedness_confidence,
    };
}

}  // namespace

struct MediaPipeOnnxHandLandmarkProvider::Impl {
    explicit Impl(MediaPipeOnnxPipelineConfig pipeline_config)
        : config(validate_config(std::move(pipeline_config))),
          palm_net(load_network(config.palm_model, "palm ONNX model")),
          hand_net(load_network(config.hand_model, "hand landmark ONNX model")),
          palm_output_names(palm_net.getUnconnectedOutLayersNames()),
          hand_output_names(hand_net.getUnconnectedOutLayersNames()),
          anchors(make_palm_anchors()) {
        const cv::Mat palm_probe(
            4, std::array<int, 4>{1, kPalmInputSize, kPalmInputSize, 3}.data(),
            CV_32F, cv::Scalar(0));
        const auto palm_outputs = forward(
            palm_net, palm_output_names, palm_probe, "palm model probe");
        (void)palm_regression_output(palm_outputs);
        (void)palm_score_output(palm_outputs);

        const cv::Mat hand_probe(
            4, std::array<int, 4>{1, kHandInputSize, kHandInputSize, 3}.data(),
            CV_32F, cv::Scalar(0));
        validate_hand_outputs(forward(
            hand_net, hand_output_names, hand_probe, "hand model probe"));
    }

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) {
        if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
            throw std::invalid_argument(
                "ONNX provider requires a non-empty CV_8UC3 BGR frame");
        }
        const auto palms = detect_palms(
            palm_net, palm_output_names, anchors, frame_bgr, config);
        std::vector<DetectedHand> hands;
        hands.reserve(palms.size());
        for (const PalmDetection& palm : palms) {
            try {
                if (auto hand = estimate_hand(
                        hand_net, hand_output_names, frame_bgr, palm,
                        config)) {
                    hands.push_back(std::move(*hand));
                }
            } catch (const InvalidPalmCrop&) {
                // A palm touching the image boundary can collapse its rotated
                // crop. Skip only that detection; the other hand remains usable.
            }
        }
        std::sort(
            hands.begin(), hands.end(),
            [](const DetectedHand& lhs, const DetectedHand& rhs) {
                return lhs.box && rhs.box
                    ? lhs.box->xmin < rhs.box->xmin
                    : lhs.landmarks[WRIST].x < rhs.landmarks[WRIST].x;
            });
        return hands;
    }

    MediaPipeOnnxPipelineConfig config;
    cv::dnn::Net palm_net;
    cv::dnn::Net hand_net;
    std::vector<cv::String> palm_output_names;
    std::vector<cv::String> hand_output_names;
    std::vector<cv::Point2d> anchors;
};

MediaPipeOnnxHandLandmarkProvider::MediaPipeOnnxHandLandmarkProvider(
    MediaPipeOnnxPipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

MediaPipeOnnxHandLandmarkProvider::~MediaPipeOnnxHandLandmarkProvider() = default;

LandmarkProviderInfo MediaPipeOnnxHandLandmarkProvider::info() const {
    return {"mediapipe-onnx-fp32", true, false};
}

std::vector<DetectedHand> MediaPipeOnnxHandLandmarkProvider::detect(
    const cv::Mat& frame_bgr) {
    return impl_->detect(frame_bgr);
}

}  // namespace double_ok_gesture
