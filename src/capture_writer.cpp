#include "double_ok_gesture/capture_writer.hpp"

#include <chrono>
#include <fstream>
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

CaptureWriter::CaptureWriter(DataCaptureConfig config) : config_(std::move(config)) {}

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
    const int index = ++saved_count_;
    const auto image_path = config_.output_dir / timestamp_name("double_ok_centered", index, ".jpg");
    if (!cv::imwrite(image_path.string(), frame)) {
        throw std::runtime_error("Failed to write capture frame: " + image_path.string());
    }

    std::ofstream meta(image_path.string() + ".json");
    if (!meta) {
        throw std::runtime_error("Failed to write capture metadata: " + image_path.string() + ".json");
    }
    meta << "{\n";
    meta << "  \"image\": \"" << json_escape(image_path.filename().string()) << "\",\n";
    meta << "  \"backend\": \"" << json_escape(backend) << "\",\n";
    meta << "  \"reason\": \"" << gate_reason_value(decision.reason) << "\",\n";
    meta << "  \"double_ok\": " << (decision.double_ok ? "true" : "false") << ",\n";
    meta << "  \"hands_centered\": " << (decision.hands_centered ? "true" : "false") << ",\n";
    meta << "  \"hand_count\": " << decision.hand_count << ",\n";
    meta << "  \"ok_count\": " << result.ok_count << "\n";
    meta << "}\n";

    last_saved_time_ = now;
    return image_path;
}

int CaptureWriter::saved_count() const {
    return saved_count_;
}

}  // namespace double_ok_gesture
