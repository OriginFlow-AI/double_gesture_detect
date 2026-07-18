#include "double_ok_gesture/landmark_provider.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef DOUBLE_OK_HAS_RKNN
#include <rknn_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "double_ok_gesture/runtime.hpp"

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

void require_model(
    const std::filesystem::path& path,
    const std::string& name,
    const char* extension) {
    if (path.empty() || path.extension() != extension) {
        throw std::invalid_argument(
            name + " must point to a " + extension + " file");
    }
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) == 0) {
        throw std::runtime_error(
            name + " not found or empty: " + path.string());
    }
}

MediaPipeOnnxPipelineConfig validate_config(
    MediaPipeOnnxPipelineConfig config,
    const char* extension,
    const char* backend) {
    require_model(
        config.palm_model,
        std::string("palm ") + backend + " model",
        extension);
    require_model(
        config.hand_model,
        std::string("hand landmark ") + backend + " model",
        extension);
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

class NetworkRunner {
public:
    virtual ~NetworkRunner() = default;
    virtual std::vector<cv::Mat> forward(const cv::Mat& rgb_u8) = 0;
};

class OpenCvNetworkRunner final : public NetworkRunner {
public:
    OpenCvNetworkRunner(
        std::filesystem::path path,
        std::vector<cv::String> output_names,
        std::string label)
        : output_names_(std::move(output_names)),
          label_(std::move(label)) {
        try {
            net_ = cv::dnn::readNetFromONNX(path.string());
            if (net_.empty()) {
                throw std::runtime_error(
                    "OpenCV DNN returned an empty network");
            }
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (const cv::Exception& error) {
            throw std::runtime_error(
                "Unable to load " + label_ + " '" + path.string() +
                "' with OpenCV DNN: " + error.what());
        }
    }

    std::vector<cv::Mat> forward(const cv::Mat& rgb_u8) override {
        try {
            net_.setInput(nhwc_float_input(rgb_u8));
            std::vector<cv::Mat> outputs;
            net_.forward(outputs, output_names_);
            return outputs;
        } catch (const cv::Exception& error) {
            throw std::runtime_error(
                label_ + " ONNX inference failed: " + error.what());
        }
    }

private:
    cv::dnn::Net net_;
    std::vector<cv::String> output_names_;
    std::string label_;
};

#ifdef DOUBLE_OK_HAS_RKNN

std::vector<unsigned char> load_binary_file(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open RKNN model: " + path.string());
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 ||
        static_cast<unsigned long long>(size) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "RKNN model has an invalid file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!input.read(
            reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Unable to read RKNN model: " + path.string());
    }
    return data;
}

void require_rknn_success(int status, const std::string& operation) {
    if (status != RKNN_SUCC) {
        throw std::runtime_error(
            operation + " failed with RKNN status " +
            std::to_string(status));
    }
}

class RknnNetworkRunner final : public NetworkRunner {
public:
    RknnNetworkRunner(
        std::filesystem::path path,
        int input_size,
        std::vector<std::string> output_names,
        std::vector<std::size_t> output_element_counts,
        std::string label)
        : model_data_(load_binary_file(path)),
          input_size_(input_size),
          label_(std::move(label)) {
        if (output_names.size() != output_element_counts.size()) {
            throw std::logic_error("invalid RKNN output contract");
        }
        require_rknn_success(
            rknn_init(
                &context_,
                model_data_.data(),
                static_cast<std::uint32_t>(model_data_.size()),
                0,
                nullptr),
            "rknn_init(" + label_ + ")");
        initialized_ = true;
        try {
            require_rknn_success(
                rknn_set_core_mask(context_, RKNN_NPU_CORE_AUTO),
                "rknn_set_core_mask(" + label_ + ")");
            initialize_contract(
                std::move(output_names),
                std::move(output_element_counts));
        } catch (...) {
            rknn_destroy(context_);
            initialized_ = false;
            throw;
        }
    }

    ~RknnNetworkRunner() override {
        if (initialized_) {
            rknn_destroy(context_);
        }
    }

    std::vector<cv::Mat> forward(const cv::Mat& rgb_u8) override {
        if (rgb_u8.empty() || rgb_u8.type() != CV_8UC3 ||
            rgb_u8.rows != input_size_ || rgb_u8.cols != input_size_ ||
            !rgb_u8.isContinuous()) {
            throw std::invalid_argument(
                label_ + " RKNN input must be a continuous square CV_8UC3 RGB image");
        }
        rknn_input input{};
        input.index = 0;
        input.buf = const_cast<unsigned char*>(rgb_u8.ptr<unsigned char>());
        input.size = static_cast<std::uint32_t>(rgb_u8.total() * rgb_u8.elemSize());
        input.pass_through = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        require_rknn_success(
            rknn_inputs_set(context_, 1, &input),
            "rknn_inputs_set(" + label_ + ")");
        require_rknn_success(
            rknn_run(context_, nullptr),
            "rknn_run(" + label_ + ")");

        std::vector<rknn_output> raw_outputs(output_indices_.size());
        for (std::size_t index = 0; index < raw_outputs.size(); ++index) {
            raw_outputs[index].index = output_indices_[index];
            raw_outputs[index].want_float = 1;
            raw_outputs[index].is_prealloc = 0;
        }
        require_rknn_success(
            rknn_outputs_get(
                context_,
                static_cast<std::uint32_t>(raw_outputs.size()),
                raw_outputs.data(),
                nullptr),
            "rknn_outputs_get(" + label_ + ")");
        bool outputs_released = false;
        try {
            std::vector<cv::Mat> outputs;
            outputs.reserve(raw_outputs.size());
            for (std::size_t index = 0; index < raw_outputs.size(); ++index) {
                if (!raw_outputs[index].buf) {
                    throw std::runtime_error(
                        label_ + " RKNN returned a null output buffer");
                }
                const std::size_t required_bytes =
                    output_element_counts_[index] * sizeof(float);
                if (raw_outputs[index].size < required_bytes) {
                    throw std::runtime_error(
                        label_ + " RKNN returned a truncated output buffer");
                }
                cv::Mat output(
                    1,
                    static_cast<int>(output_element_counts_[index]),
                    CV_32F);
                std::memcpy(
                    output.ptr<float>(),
                    raw_outputs[index].buf,
                    required_bytes);
                outputs.push_back(std::move(output));
            }
            const int release_status = rknn_outputs_release(
                context_,
                static_cast<std::uint32_t>(raw_outputs.size()),
                raw_outputs.data());
            outputs_released = true;
            require_rknn_success(
                release_status,
                "rknn_outputs_release(" + label_ + ")");
            return outputs;
        } catch (...) {
            if (!outputs_released) {
                (void)rknn_outputs_release(
                    context_,
                    static_cast<std::uint32_t>(raw_outputs.size()),
                    raw_outputs.data());
            }
            throw;
        }
    }

private:
    void initialize_contract(
        std::vector<std::string> expected_names,
        std::vector<std::size_t> expected_counts) {
        rknn_input_output_num io_count{};
        require_rknn_success(
            rknn_query(
                context_, RKNN_QUERY_IN_OUT_NUM, &io_count, sizeof(io_count)),
            "rknn_query(io count, " + label_ + ")");
        if (io_count.n_input != 1 ||
            io_count.n_output != expected_names.size()) {
            throw std::runtime_error(
                label_ + " RKNN model has an unexpected input/output count");
        }

        rknn_tensor_attr input_attr{};
        input_attr.index = 0;
        require_rknn_success(
            rknn_query(
                context_,
                RKNN_QUERY_INPUT_ATTR,
                &input_attr,
                sizeof(input_attr)),
            "rknn_query(input, " + label_ + ")");
        const std::size_t expected_input_elements =
            static_cast<std::size_t>(input_size_) * input_size_ * 3;
        const bool expected_nchw =
            input_attr.fmt == RKNN_TENSOR_NCHW && input_attr.n_dims == 4 &&
            input_attr.dims[0] == 1 && input_attr.dims[1] == 3 &&
            input_attr.dims[2] == static_cast<std::uint32_t>(input_size_) &&
            input_attr.dims[3] == static_cast<std::uint32_t>(input_size_);
        const bool expected_nhwc =
            input_attr.fmt == RKNN_TENSOR_NHWC && input_attr.n_dims == 4 &&
            input_attr.dims[0] == 1 &&
            input_attr.dims[1] == static_cast<std::uint32_t>(input_size_) &&
            input_attr.dims[2] == static_cast<std::uint32_t>(input_size_) &&
            input_attr.dims[3] == 3;
        if (std::string(input_attr.name) != "input_1" ||
            input_attr.n_elems != expected_input_elements ||
            (!expected_nchw && !expected_nhwc)) {
            throw std::runtime_error(
                label_ + " RKNN model has an unexpected input shape");
        }

        std::vector<rknn_tensor_attr> attributes(io_count.n_output);
        for (std::uint32_t index = 0; index < io_count.n_output; ++index) {
            attributes[index].index = index;
            require_rknn_success(
                rknn_query(
                    context_,
                    RKNN_QUERY_OUTPUT_ATTR,
                    &attributes[index],
                    sizeof(attributes[index])),
                "rknn_query(output, " + label_ + ")");
        }

        for (std::size_t expected = 0; expected < expected_names.size(); ++expected) {
            const auto found = std::find_if(
                attributes.begin(),
                attributes.end(),
                [&](const rknn_tensor_attr& attribute) {
                    return expected_names[expected] == attribute.name;
                });
            if (found == attributes.end()) {
                throw std::runtime_error(
                    label_ + " RKNN model is missing output '" +
                    expected_names[expected] + "'");
            }
            if (found->n_elems != expected_counts[expected]) {
                throw std::runtime_error(
                    label_ + " RKNN output '" + expected_names[expected] +
                    "' has an unexpected element count");
            }
            output_indices_.push_back(found->index);
            output_element_counts_.push_back(expected_counts[expected]);
        }

        rknn_sdk_version version{};
        require_rknn_success(
            rknn_query(
                context_,
                RKNN_QUERY_SDK_VERSION,
                &version,
                sizeof(version)),
            "rknn_query(version, " + label_ + ")");
        log_message(
            LogLevel::Info,
            label_ + " RKNN ready: api=" + version.api_version +
                ", driver=" + version.drv_version);
    }

    std::vector<unsigned char> model_data_;
    int input_size_ = 0;
    std::string label_;
    rknn_context context_{};
    bool initialized_ = false;
    std::vector<std::uint32_t> output_indices_;
    std::vector<std::size_t> output_element_counts_;
};

#endif

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
        "palm model must output [1,2016,18] regressions");
}

const cv::Mat& palm_score_output(const std::vector<cv::Mat>& outputs) {
    for (const cv::Mat& output : outputs) {
        if (output.total() == kPalmAnchorCount) {
            return output;
        }
    }
    throw std::runtime_error(
        "palm model must output [1,2016,1] scores");
}

std::vector<PalmDetection> detect_palms(
    NetworkRunner& runner,
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

    const std::vector<cv::Mat> outputs = runner.forward(square);
    cv::Mat regression = palm_regression_output(outputs);
    cv::Mat scores = palm_score_output(outputs);
    if (regression.depth() != CV_32F || scores.depth() != CV_32F) {
        throw std::runtime_error("palm model outputs must be float32");
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
        resized,
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
            "hand landmark model must output [63], [1], [1], [63]");
    }
    if (std::any_of(
            outputs.begin(), outputs.end(),
            [](const cv::Mat& output) {
                return output.empty() || output.depth() != CV_32F;
            })) {
        throw std::runtime_error("hand landmark outputs must be float32");
    }
}

struct HandResultTiming {
    double prepare_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
};

std::optional<DetectedHand> estimate_hand_with_timing(
    NetworkRunner& runner,
    const cv::Mat& frame_bgr,
    const PalmDetection& palm,
    const MediaPipeOnnxPipelineConfig& config,
    HandResultTiming& timing) {
    const double prepare_start = monotonic_seconds();
    const HandInputTransform transform = prepare_hand_input(frame_bgr, palm);
    timing.prepare_ms = (monotonic_seconds() - prepare_start) * 1000.0;

    const double infer_start = monotonic_seconds();
    std::vector<cv::Mat> outputs = runner.forward(transform.input);
    timing.infer_ms = (monotonic_seconds() - infer_start) * 1000.0;
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
        throw std::runtime_error("handedness model output is not finite");
    }

    const double post_start = monotonic_seconds();
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
    timing.postprocess_ms = (monotonic_seconds() - post_start) * 1000.0;

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

class MediaPipePipeline {
public:
    MediaPipePipeline(
        MediaPipeOnnxPipelineConfig pipeline_config,
        std::unique_ptr<NetworkRunner> palm_runner,
        std::unique_ptr<NetworkRunner> hand_runner)
        : config(std::move(pipeline_config)),
          palm_runner(std::move(palm_runner)),
          hand_runner(std::move(hand_runner)),
          anchors(make_palm_anchors()) {
        const cv::Mat palm_probe(
            kPalmInputSize,
            kPalmInputSize,
            CV_8UC3,
            cv::Scalar(0, 0, 0));
        const auto palm_outputs = this->palm_runner->forward(palm_probe);
        (void)palm_regression_output(palm_outputs);
        (void)palm_score_output(palm_outputs);

        const cv::Mat hand_probe(
            kHandInputSize,
            kHandInputSize,
            CV_8UC3,
            cv::Scalar(0, 0, 0));
        validate_hand_outputs(this->hand_runner->forward(hand_probe));
    }

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) {
        if (frame_bgr.empty() || frame_bgr.type() != CV_8UC3) {
            throw std::invalid_argument(
                "MediaPipe provider requires a non-empty CV_8UC3 BGR frame");
        }
        const double palm_start = monotonic_seconds();
        const auto palms = detect_palms(
            *palm_runner, anchors, frame_bgr, config);
        const double palm_ms = (monotonic_seconds() - palm_start) * 1000.0;

        const double hand_start = monotonic_seconds();
        double hand_prepare_ms = 0.0;
        double hand_infer_ms = 0.0;
        double hand_postprocess_ms = 0.0;
        std::vector<DetectedHand> hands;
        hands.reserve(palms.size());
        for (const PalmDetection& palm : palms) {
            try {
                HandResultTiming timing;
                if (auto hand = estimate_hand_with_timing(
                        *hand_runner, frame_bgr, palm, config, timing)) {
                    hand_prepare_ms += timing.prepare_ms;
                    hand_infer_ms += timing.infer_ms;
                    hand_postprocess_ms += timing.postprocess_ms;
                    hands.push_back(std::move(*hand));
                }
            } catch (const InvalidPalmCrop&) {
                // A palm touching the image boundary can collapse its rotated
                // crop. Skip only that detection; the other hand remains usable.
            }
        }
        const double hand_ms = (monotonic_seconds() - hand_start) * 1000.0;

        log_message(LogLevel::Debug,
            "palm_npu=" + std::to_string(palm_ms) + "ms, hand_prepare=" + std::to_string(hand_prepare_ms) + "ms, hand_npu=" + std::to_string(hand_infer_ms) + "ms, hand_post=" + std::to_string(hand_postprocess_ms) + "ms, palms=" + std::to_string(palms.size()) + ", hands=" + std::to_string(hands.size()));

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
    std::unique_ptr<NetworkRunner> palm_runner;
    std::unique_ptr<NetworkRunner> hand_runner;
    std::vector<cv::Point2d> anchors;
};

struct MediaPipeOnnxHandLandmarkProvider::Impl {
    explicit Impl(MediaPipeOnnxPipelineConfig pipeline_config)
        : config(validate_config(
              std::move(pipeline_config), ".onnx", "ONNX")),
          pipeline(
              config,
              std::make_unique<OpenCvNetworkRunner>(
                  config.palm_model,
                  std::vector<cv::String>{"Identity", "Identity_1"},
                  "palm detection"),
              std::make_unique<OpenCvNetworkRunner>(
                  config.hand_model,
                  std::vector<cv::String>{
                      "Identity", "Identity_1", "Identity_2", "Identity_3"},
                  "hand landmark")) {}

    MediaPipeOnnxPipelineConfig config;
    MediaPipePipeline pipeline;
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
    return impl_->pipeline.detect(frame_bgr);
}

#ifdef DOUBLE_OK_HAS_RKNN

struct MediaPipeRknnHandLandmarkProvider::Impl {
    explicit Impl(MediaPipeRknnPipelineConfig pipeline_config)
        : config(validate_config(
              std::move(pipeline_config), ".rknn", "RKNN")),
          pipeline(
              config,
              std::make_unique<RknnNetworkRunner>(
                  config.palm_model,
                  kPalmInputSize,
                  std::vector<std::string>{"Identity", "Identity_1"},
                  std::vector<std::size_t>{
                      kPalmAnchorCount * kPalmValueCount,
                      kPalmAnchorCount},
                  "palm detection"),
              std::make_unique<RknnNetworkRunner>(
                  config.hand_model,
                  kHandInputSize,
                  std::vector<std::string>{
                      "Identity", "Identity_1", "Identity_2", "Identity_3"},
                  std::vector<std::size_t>{
                      kHandLandmarkValueCount,
                      1,
                      1,
                      kHandLandmarkValueCount},
                  "hand landmark")) {}

    MediaPipeRknnPipelineConfig config;
    MediaPipePipeline pipeline;
};

#else

struct MediaPipeRknnHandLandmarkProvider::Impl {};

#endif

MediaPipeRknnHandLandmarkProvider::MediaPipeRknnHandLandmarkProvider(
    MediaPipeRknnPipelineConfig config)
#ifdef DOUBLE_OK_HAS_RKNN
    : impl_(std::make_unique<Impl>(std::move(config))) {}
#else
    : impl_(nullptr) {
    (void)config;
    throw std::runtime_error(
        "RKNN backend is not available in this build; rebuild with "
        "DOUBLE_OK_REQUIRE_RKNN=ON");
}
#endif

MediaPipeRknnHandLandmarkProvider::~MediaPipeRknnHandLandmarkProvider() = default;

LandmarkProviderInfo MediaPipeRknnHandLandmarkProvider::info() const {
#ifdef DOUBLE_OK_HAS_RKNN
    return {"mediapipe-rknn-fp16-npu", true, false};
#else
    return {"mediapipe-rknn-fp16-npu", false, false};
#endif
}

std::vector<DetectedHand> MediaPipeRknnHandLandmarkProvider::detect(
    const cv::Mat& frame_bgr) {
#ifdef DOUBLE_OK_HAS_RKNN
    return impl_->pipeline.detect(frame_bgr);
#else
    (void)frame_bgr;
    throw std::runtime_error("RKNN backend is unavailable");
#endif
}

}  // namespace double_ok_gesture
