#include "double_ok_gesture/capture_writer.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include "double_ok_gesture/runtime.hpp"

namespace double_ok_gesture {
namespace {

std::string timestamp_name(const std::string& prefix, int index, const std::string& extension) {
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return prefix + "_" + std::to_string(millis) + "_" + std::to_string(index) + extension;
}

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

}  // namespace

CaptureWriter::CaptureWriter(DataCaptureConfig config)
    : config_(std::move(config)) {
    if (!std::isfinite(config_.cooldown_sec) || config_.cooldown_sec < 0.0) {
        throw std::invalid_argument(
            "capture cooldown must be finite and non-negative");
    }
    if (config_.enabled && config_.output_dir.empty()) {
        throw std::invalid_argument(
            "capture output directory must not be empty when enabled");
    }
}

std::optional<std::filesystem::path> CaptureWriter::maybe_save(
    const cv::Mat& frame,
    const DoubleOKResult& result,
    const CaptureGateDecision& decision,
    const std::string& backend) {
    if (!config_.enabled || !decision.ready || frame.empty()) {
        return std::nullopt;
    }

    const double now = monotonic_seconds();
    if (last_saved_time_ >= 0.0 && now - last_saved_time_ < config_.cooldown_sec) {
        return std::nullopt;
    }

    std::filesystem::create_directories(config_.output_dir);
    const int index = saved_count_ + 1;
    const auto image_path = config_.output_dir / timestamp_name("double_ok_centered", index, ".jpg");
    const std::filesystem::path metadata_path = image_path.string() + ".json";
    const auto temporary_image = config_.output_dir /
        ("." + image_path.stem().string() + ".tmp.jpg");
    const auto temporary_metadata = config_.output_dir /
        ("." + metadata_path.filename().string() + ".tmp");
    bool image_committed = false;
    try {
        if (!cv::imwrite(temporary_image.string(), frame)) {
            throw std::runtime_error(
                "Failed to write capture frame: " + image_path.string());
        }

        std::ofstream meta(temporary_metadata);
        if (!meta) {
            throw std::runtime_error(
                "Failed to write capture metadata: " +
                metadata_path.string());
        }
        meta << std::setprecision(17);
        meta << "{\n";
        meta << "  \"schema\": \"double_ok_capture_v2\",\n";
        meta << "  \"image\": \"" << json_escape(image_path.filename().string()) << "\",\n";
        meta << "  \"backend\": \"" << json_escape(backend) << "\",\n";
        meta << "  \"label_source\": \"runtime_prediction\",\n";
        meta << "  \"reason\": \"" << gate_reason_value(decision.reason) << "\",\n";
        meta << "  \"double_ok\": " << (decision.double_ok ? "true" : "false") << ",\n";
        meta << "  \"hands_centered\": " << (decision.hands_centered ? "true" : "false") << ",\n";
        meta << "  \"hand_count\": " << decision.hand_count << ",\n";
        meta << "  \"ok_count\": " << result.ok_count << ",\n";
        meta << "  \"hands\": [";
        for (std::size_t hand_index = 0; hand_index < result.hands.size(); ++hand_index) {
            const HandPrediction& hand = result.hands[hand_index];
            meta << (hand_index == 0 ? "\n" : ",\n");
            meta << "    {\"handedness\": \"" << json_escape(hand.handedness)
                 << "\", \"handedness_confidence\": " << hand.handedness_confidence
                 << ", \"ok_score\": " << hand.ok_score
                 << ", \"is_ok\": " << (hand.is_ok ? "true" : "false")
                 << ", \"landmarks\": [";
            for (std::size_t point_index = 0; point_index < hand.landmarks.size(); ++point_index) {
                const Point3& point = hand.landmarks[point_index];
                meta << (point_index == 0 ? "" : ",") << '[' << point.x
                     << ',' << point.y << ',' << point.z << ']';
            }
            meta << "], \"visibility\": ";
            if (hand.landmark_confidences) {
                meta << '[';
                for (std::size_t point_index = 0;
                     point_index < hand.landmark_confidences->size();
                     ++point_index) {
                    meta << (point_index == 0 ? "" : ",")
                         << (*hand.landmark_confidences)[point_index];
                }
                meta << ']';
            } else {
                meta << "null";
            }
            meta << ", \"box\": ";
            if (hand.box) {
                meta << "{\"xmin\": " << hand.box->xmin
                     << ", \"ymin\": " << hand.box->ymin
                     << ", \"xmax\": " << hand.box->xmax
                     << ", \"ymax\": " << hand.box->ymax
                     << ", \"detection_score\": "
                     << hand.box->detection_score << '}';
            } else {
                meta << "null";
            }
            meta << '}';
        }
        if (!result.hands.empty()) {
            meta << '\n';
        }
        meta << "  ]\n";
        meta << "}\n";
        meta.close();
        if (!meta) {
            throw std::runtime_error(
                "Failed to finish capture metadata: " +
                metadata_path.string());
        }

        std::filesystem::rename(temporary_image, image_path);
        image_committed = true;
        std::filesystem::rename(temporary_metadata, metadata_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_image, ignored);
        std::filesystem::remove(temporary_metadata, ignored);
        if (image_committed) {
            std::filesystem::remove(image_path, ignored);
        }
        throw;
    }

    saved_count_ = index;
    last_saved_time_ = now;
    log_message(LogLevel::Info, "capture saved: " + image_path.string());
    return image_path;
}

int CaptureWriter::saved_count() const {
    return saved_count_;
}

}  // namespace double_ok_gesture
