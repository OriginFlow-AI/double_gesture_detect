#include "double_ok_gesture/config.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/json.hpp"

namespace double_ok_gesture {
namespace {

std::string qualified_name(
    const std::string& section,
    const std::string& key) {
    return section.empty() ? key : section + "." + key;
}

double number_member(
    const Json& object,
    const std::string& section,
    const std::string& key) {
    const Json* value = object.get(key);
    if (!value || !value->is_number()) {
        throw std::runtime_error(
            "Config '" + qualified_name(section, key) +
            "' must be a number");
    }
    return value->as_number();
}

void set_if_present(
    int& target,
    const Json& object,
    const std::string& section,
    const std::string& key) {
    if (!object.get(key)) {
        return;
    }
    const double value = number_member(object, section, key);
    if (!std::isfinite(value) || std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Config '" + qualified_name(section, key) +
            "' must be an integer");
    }
    target = static_cast<int>(value);
}

void set_if_present(
    double& target,
    const Json& object,
    const std::string& section,
    const std::string& key) {
    if (object.get(key)) {
        target = number_member(object, section, key);
    }
}

void set_if_present(
    bool& target,
    const Json& object,
    const std::string& section,
    const std::string& key) {
    const Json* value = object.get(key);
    if (!value) {
        return;
    }
    if (!value->is_bool()) {
        throw std::runtime_error(
            "Config '" + qualified_name(section, key) +
            "' must be a boolean");
    }
    target = value->as_bool();
}

void set_if_present(
    std::filesystem::path& target,
    const Json& object,
    const std::string& section,
    const std::string& key) {
    const Json* value = object.get(key);
    if (!value) {
        return;
    }
    if (!value->is_string()) {
        throw std::runtime_error(
            "Config '" + qualified_name(section, key) +
            "' must be a string");
    }
    target = value->as_string();
}

const Json* object_section(const Json& root, const std::string& key) {
    const Json* section = root.get(key);
    if (!section) {
        return nullptr;
    }
    if (!section->is_object()) {
        throw std::runtime_error(
            "Config '" + key + "' must be an object");
    }
    return section;
}

void reject_unknown_members(
    const Json& object,
    const std::set<std::string>& allowed,
    const std::string& section) {
    for (const auto& [key, unused] : object.as_object()) {
        (void)unused;
        if (!allowed.contains(key)) {
            throw std::runtime_error(
                "Unknown config field '" + qualified_name(section, key) +
                "'");
        }
    }
}

void require_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and in [0,1]");
    }
}

}  // namespace

void validate_runtime_config(const RuntimeConfig& config) {
    if (config.recognizer.max_num_hands < 1 ||
        config.recognizer.max_num_hands > 2) {
        throw std::invalid_argument("max_num_hands must be 1 or 2");
    }
    require_probability(config.recognizer.ok_threshold, "ok_threshold");
    if (config.recognizer.stable_window < 1) {
        throw std::invalid_argument("stable_window must be at least 1");
    }
    if (config.recognizer.stable_min_positive < 1 ||
        config.recognizer.stable_min_positive >
            config.recognizer.stable_window) {
        throw std::invalid_argument(
            "stable_min_positive must be between 1 and stable_window");
    }
    require_probability(
        config.onnx_hand.palm_detection_threshold,
        "palm_detection_threshold");
    require_probability(
        config.onnx_hand.hand_presence_threshold,
        "hand_presence_threshold");
    require_probability(
        config.onnx_hand.palm_nms_threshold,
        "palm_nms_threshold");
    if (config.onnx_hand.palm_model_path.empty() ||
        config.onnx_hand.hand_model_path.empty()) {
        throw std::invalid_argument(
            "palm_model_path and hand_model_path must not be empty");
    }
    if (config.rknn_hand.palm_model_path.empty() ||
        config.rknn_hand.hand_model_path.empty()) {
        throw std::invalid_argument(
            "rknn_palm_model_path and rknn_hand_model_path must not be empty");
    }

    config.capture_gate.validate();
    if (!std::isfinite(config.data_capture.cooldown_sec) ||
        config.data_capture.cooldown_sec < 0.0) {
        throw std::invalid_argument(
            "data_capture.cooldown_sec must be finite and non-negative");
    }
    if (config.data_capture.enabled && config.data_capture.output_dir.empty()) {
        throw std::invalid_argument(
            "data_capture.output_dir must not be empty when capture is enabled");
    }
}

RuntimeConfig load_runtime_config(const std::filesystem::path& path) {
    RuntimeConfig config;
    const Json root = load_json(path);
    if (!root.is_object()) {
        throw std::runtime_error("Runtime config root must be a JSON object");
    }
    reject_unknown_members(
        root,
        {
            "max_num_hands",
            "ok_threshold",
            "stable_window",
            "stable_min_positive",
            "input_mirrored",
            "palm_model_path",
            "hand_model_path",
            "rknn_palm_model_path",
            "rknn_hand_model_path",
            "palm_detection_threshold",
            "hand_presence_threshold",
            "palm_nms_threshold",
            "capture_gate",
            "data_capture",
        },
        "");

    set_if_present(config.recognizer.max_num_hands, root, "", "max_num_hands");
    set_if_present(config.recognizer.ok_threshold, root, "", "ok_threshold");
    set_if_present(config.recognizer.stable_window, root, "", "stable_window");
    set_if_present(
        config.recognizer.stable_min_positive,
        root,
        "",
        "stable_min_positive");
    set_if_present(
        config.onnx_hand.input_mirrored,
        root,
        "",
        "input_mirrored");

    set_if_present(
        config.onnx_hand.palm_model_path, root, "", "palm_model_path");
    set_if_present(
        config.onnx_hand.hand_model_path, root, "", "hand_model_path");
    set_if_present(
        config.rknn_hand.palm_model_path,
        root,
        "",
        "rknn_palm_model_path");
    set_if_present(
        config.rknn_hand.hand_model_path,
        root,
        "",
        "rknn_hand_model_path");
    set_if_present(
        config.onnx_hand.palm_detection_threshold,
        root,
        "",
        "palm_detection_threshold");
    set_if_present(
        config.onnx_hand.hand_presence_threshold,
        root,
        "",
        "hand_presence_threshold");
    set_if_present(
        config.onnx_hand.palm_nms_threshold,
        root,
        "",
        "palm_nms_threshold");

    if (const Json* gate = object_section(root, "capture_gate")) {
        reject_unknown_members(
            *gate,
            {
                "require_glasses_pose",
                "pitch_min",
                "pitch_max",
                "roll_min",
                "roll_max",
                "yaw_min",
                "yaw_max",
                "frame_margin",
                "center_x_min",
                "center_x_max",
                "center_y_min",
                "center_y_max",
                "min_hand_separation",
                "use_stable_double_ok",
                "require_double_ok",
            },
            "capture_gate");
        set_if_present(
            config.capture_gate.require_glasses_pose,
            *gate,
            "capture_gate",
            "require_glasses_pose");
        set_if_present(config.capture_gate.pitch_min, *gate, "capture_gate", "pitch_min");
        set_if_present(config.capture_gate.pitch_max, *gate, "capture_gate", "pitch_max");
        set_if_present(config.capture_gate.roll_min, *gate, "capture_gate", "roll_min");
        set_if_present(config.capture_gate.roll_max, *gate, "capture_gate", "roll_max");
        set_if_present(config.capture_gate.yaw_min, *gate, "capture_gate", "yaw_min");
        set_if_present(config.capture_gate.yaw_max, *gate, "capture_gate", "yaw_max");
        set_if_present(config.capture_gate.frame_margin, *gate, "capture_gate", "frame_margin");
        set_if_present(config.capture_gate.center_x_min, *gate, "capture_gate", "center_x_min");
        set_if_present(config.capture_gate.center_x_max, *gate, "capture_gate", "center_x_max");
        set_if_present(config.capture_gate.center_y_min, *gate, "capture_gate", "center_y_min");
        set_if_present(config.capture_gate.center_y_max, *gate, "capture_gate", "center_y_max");
        set_if_present(
            config.capture_gate.min_hand_separation,
            *gate,
            "capture_gate",
            "min_hand_separation");
        set_if_present(
            config.capture_gate.use_stable_double_ok,
            *gate,
            "capture_gate",
            "use_stable_double_ok");
        set_if_present(
            config.capture_gate.require_double_ok,
            *gate,
            "capture_gate",
            "require_double_ok");
    }

    if (const Json* capture = object_section(root, "data_capture")) {
        reject_unknown_members(
            *capture,
            {"enabled", "output_dir", "cooldown_sec"},
            "data_capture");
        set_if_present(
            config.data_capture.enabled,
            *capture,
            "data_capture",
            "enabled");
        set_if_present(
            config.data_capture.output_dir,
            *capture,
            "data_capture",
            "output_dir");
        set_if_present(
            config.data_capture.cooldown_sec,
            *capture,
            "data_capture",
            "cooldown_sec");
    }

    validate_runtime_config(config);
    return config;
}

void apply_threshold_override(
    RuntimeConfig& config,
    std::optional<double> threshold) {
    if (threshold) {
        config.recognizer.ok_threshold = *threshold;
        require_probability(config.recognizer.ok_threshold, "ok_threshold");
    }
}

void require_glasses_pose(RuntimeConfig& config) {
    config.capture_gate.require_glasses_pose = true;
}

}  // namespace double_ok_gesture
