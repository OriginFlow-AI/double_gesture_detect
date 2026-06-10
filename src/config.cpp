#include "double_ok_gesture/config.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open config file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::optional<double> number_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    return std::stod(match[1].str());
}

std::optional<bool> bool_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    return match[1].str() == "true";
}

std::optional<std::string> string_value(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    return match[1].str();
}

void set_if_present(int& target, const std::string& text, const std::string& key) {
    if (auto value = number_value(text, key)) {
        target = static_cast<int>(*value);
    }
}

void set_if_present(double& target, const std::string& text, const std::string& key) {
    if (auto value = number_value(text, key)) {
        target = *value;
    }
}

void set_if_present(bool& target, const std::string& text, const std::string& key) {
    if (auto value = bool_value(text, key)) {
        target = *value;
    }
}

void set_if_present(std::filesystem::path& target, const std::string& text, const std::string& key) {
    if (auto value = string_value(text, key)) {
        target = *value;
    }
}

}  // namespace

RuntimeConfig load_runtime_config(const std::filesystem::path& path) {
    RuntimeConfig config;
    const std::string text = read_text(path);

    set_if_present(config.recognizer.max_num_hands, text, "max_num_hands");
    set_if_present(config.recognizer.ok_threshold, text, "ok_threshold");
    set_if_present(config.recognizer.stable_window, text, "stable_window");
    set_if_present(config.recognizer.stable_min_positive, text, "stable_min_positive");
    set_if_present(config.recognizer.min_detection_confidence, text, "min_detection_confidence");
    set_if_present(config.recognizer.min_tracking_confidence, text, "min_tracking_confidence");

    set_if_present(config.capture_gate.require_glasses_pose, text, "require_glasses_pose");
    set_if_present(config.capture_gate.pitch_min, text, "pitch_min");
    set_if_present(config.capture_gate.pitch_max, text, "pitch_max");
    set_if_present(config.capture_gate.roll_min, text, "roll_min");
    set_if_present(config.capture_gate.roll_max, text, "roll_max");
    set_if_present(config.capture_gate.yaw_min, text, "yaw_min");
    set_if_present(config.capture_gate.yaw_max, text, "yaw_max");
    set_if_present(config.capture_gate.frame_margin, text, "frame_margin");
    set_if_present(config.capture_gate.center_x_min, text, "center_x_min");
    set_if_present(config.capture_gate.center_x_max, text, "center_x_max");
    set_if_present(config.capture_gate.center_y_min, text, "center_y_min");
    set_if_present(config.capture_gate.center_y_max, text, "center_y_max");
    set_if_present(config.capture_gate.min_hand_separation, text, "min_hand_separation");
    set_if_present(config.capture_gate.use_stable_double_ok, text, "use_stable_double_ok");
    set_if_present(config.capture_gate.require_double_ok, text, "require_double_ok");
    config.capture_gate.validate();

    set_if_present(config.data_capture.enabled, text, "enabled");
    set_if_present(config.data_capture.output_dir, text, "output_dir");
    set_if_present(config.data_capture.cooldown_sec, text, "cooldown_sec");
    if (!std::isfinite(config.data_capture.cooldown_sec) || config.data_capture.cooldown_sec < 0.0) {
        throw std::invalid_argument("data_capture.cooldown_sec must be finite and non-negative");
    }
    return config;
}

RuntimeConfig load_runtime_config_or_default(const std::optional<std::filesystem::path>& path) {
    if (!path || path->empty()) {
        return RuntimeConfig{};
    }
    return load_runtime_config(*path);
}

void apply_threshold_override(RuntimeConfig& config, std::optional<double> threshold) {
    if (threshold) {
        config.recognizer.ok_threshold = *threshold;
    }
}

void require_glasses_pose(RuntimeConfig& config) {
    config.capture_gate.require_glasses_pose = true;
}

}  // namespace double_ok_gesture
