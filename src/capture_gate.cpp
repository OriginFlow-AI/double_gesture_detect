#include "double_ok_gesture/capture_gate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

void validate_range(const std::string& name, double minimum, double maximum) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
        throw std::invalid_argument(name + "_min and " + name + "_max must be finite and ordered");
    }
}

void validate_normalized_range(const std::string& name, double minimum, double maximum) {
    validate_range(name, minimum, maximum);
    if (!(0.0 <= minimum && minimum < maximum && maximum <= 1.0)) {
        throw std::invalid_argument(name + "_min and " + name + "_max must satisfy 0 <= min < max <= 1");
    }
}

std::optional<double> find_number(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    const double value = std::stod(match[1].str());
    if (!std::isfinite(value)) {
        throw std::runtime_error("pose values must be finite");
    }
    return value;
}

std::pair<bool, GateReason> check_glasses_pose(
    const std::optional<GlassesPose>& glasses_pose,
    const CaptureGateConfig& config) {
    if (!config.require_glasses_pose) {
        return {true, GateReason::Ready};
    }
    if (!glasses_pose || !glasses_pose->pitch || !glasses_pose->roll || !glasses_pose->yaw) {
        return {false, GateReason::GlassesPoseMissing};
    }
    const bool ok = *glasses_pose->pitch >= config.pitch_min && *glasses_pose->pitch <= config.pitch_max &&
                    *glasses_pose->roll >= config.roll_min && *glasses_pose->roll <= config.roll_max &&
                    *glasses_pose->yaw >= config.yaw_min && *glasses_pose->yaw <= config.yaw_max;
    return {ok, ok ? GateReason::Ready : GateReason::GlassesPoseBad};
}

bool is_hand_visible(const Landmarks& points, const CaptureGateConfig& config) {
    const double margin = config.frame_margin;
    return std::all_of(points.begin(), points.end(), [margin](const Point3& point) {
        return point.x >= margin && point.x <= 1.0 - margin && point.y >= margin && point.y <= 1.0 - margin;
    });
}

Point2 center_of(const Landmarks& points) {
    Point2 center;
    for (const Point3& point : points) {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= static_cast<double>(points.size());
    center.y /= static_cast<double>(points.size());
    return center;
}

bool is_point_centered(const Point2& point, const CaptureGateConfig& config) {
    return point.x >= config.center_x_min && point.x <= config.center_x_max && point.y >= config.center_y_min &&
           point.y <= config.center_y_max;
}

double distance(const Point2& lhs, const Point2& rhs) {
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

CaptureGateDecision decision(
    GateReason reason,
    const CaptureGateConfig& config,
    bool glasses_pose_ok,
    bool hands_centered,
    bool hands_visible,
    bool hands_separated,
    bool double_ok,
    int hand_count,
    std::optional<bool> gesture_ok = std::nullopt) {
    const bool final_gesture_ok = gesture_ok.value_or(config.require_double_ok ? double_ok : true);
    return {
        reason == GateReason::Ready,
        reason,
        gate_prompt(reason),
        glasses_pose_ok,
        hands_centered,
        hands_visible,
        hands_separated,
        double_ok,
        final_gesture_ok,
        hand_count,
    };
}

double monotonic_seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

}  // namespace

void CaptureGateConfig::validate() const {
    validate_range("pitch", pitch_min, pitch_max);
    validate_range("roll", roll_min, roll_max);
    validate_range("yaw", yaw_min, yaw_max);
    validate_normalized_range("center_x", center_x_min, center_x_max);
    validate_normalized_range("center_y", center_y_min, center_y_max);
    if (!std::isfinite(frame_margin) || frame_margin < 0.0 || frame_margin >= 0.5) {
        throw std::invalid_argument("frame_margin must be finite and in [0.0, 0.5)");
    }
    if (!std::isfinite(min_hand_separation) || min_hand_separation < 0.0) {
        throw std::invalid_argument("min_hand_separation must be finite and non-negative");
    }
}

const char* gate_reason_value(GateReason reason) {
    switch (reason) {
        case GateReason::Ready:
            return "ready";
        case GateReason::GlassesPoseMissing:
            return "glasses_pose_missing";
        case GateReason::GlassesPoseBad:
            return "glasses_pose_bad";
        case GateReason::NeedTwoHands:
            return "need_two_hands";
        case GateReason::HandsOutOfFrame:
            return "hands_out_of_frame";
        case GateReason::HandsNotCentered:
            return "hands_not_centered";
        case GateReason::HandsTooClose:
            return "hands_too_close";
        case GateReason::NeedDoubleOK:
            return "need_double_ok";
        case GateReason::AvoidDoubleOK:
            return "avoid_double_ok";
    }
    return "unknown";
}

const char* gate_prompt(GateReason reason) {
    switch (reason) {
        case GateReason::Ready:
            return "条件满足，开始采集";
        case GateReason::GlassesPoseMissing:
            return "等待眼镜姿态数据";
        case GateReason::GlassesPoseBad:
            return "请调整眼镜角度，保持视野正对双手";
        case GateReason::NeedTwoHands:
            return "请把双手放入相机画面";
        case GateReason::HandsOutOfFrame:
            return "请把双手完整放入画面";
        case GateReason::HandsNotCentered:
            return "请把双手移到画面中心";
        case GateReason::HandsTooClose:
            return "请将双手分开一些";
        case GateReason::NeedDoubleOK:
            return "请双手分开并做出 OK 手势";
        case GateReason::AvoidDoubleOK:
            return "负样本采集中，请不要同时做双手 OK";
    }
    return "未知状态";
}

GateReason gate_reason_from_string(const std::string& value) {
    const std::array<GateReason, 9> reasons = {
        GateReason::Ready,
        GateReason::GlassesPoseMissing,
        GateReason::GlassesPoseBad,
        GateReason::NeedTwoHands,
        GateReason::HandsOutOfFrame,
        GateReason::HandsNotCentered,
        GateReason::HandsTooClose,
        GateReason::NeedDoubleOK,
        GateReason::AvoidDoubleOK,
    };
    for (GateReason reason : reasons) {
        if (value == gate_reason_value(reason)) {
            return reason;
        }
    }
    throw std::invalid_argument("Unsupported gate reason: " + value);
}

StereoGateMode stereo_gate_mode_from_string(const std::string& value) {
    if (value == "left") {
        return StereoGateMode::Left;
    }
    if (value == "both") {
        return StereoGateMode::Both;
    }
    throw std::invalid_argument("Unsupported stereo gate mode: " + value);
}

std::optional<GlassesPose> load_glasses_pose(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path)) {
        return std::nullopt;
    }
    try {
        std::ifstream in(path);
        if (!in) {
            return std::nullopt;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return GlassesPose{
            find_number(text, "pitch"),
            find_number(text, "roll"),
            find_number(text, "yaw"),
        };
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

CaptureGateDecision evaluate_capture_gate(
    const DoubleOKResult& result,
    const CaptureGateConfig& config,
    const std::optional<GlassesPose>& glasses_pose) {
    config.validate();
    const auto [pose_ok, pose_reason] = check_glasses_pose(glasses_pose, config);
    const int hand_count = static_cast<int>(result.hands.size());
    bool hands_visible = false;
    bool hands_centered = false;
    bool hands_separated = false;
    const bool double_ok = result.double_ok && (config.use_stable_double_ok ? result.stable_double_ok : true);

    if (!pose_ok) {
        return decision(pose_reason, config, pose_ok, hands_centered, hands_visible, hands_separated, double_ok, hand_count);
    }
    if (hand_count < 2) {
        return decision(
            GateReason::NeedTwoHands,
            config,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count);
    }

    const Landmarks& left = result.hands[0].landmarks;
    const Landmarks& right = result.hands[1].landmarks;
    validate_landmarks(left);
    validate_landmarks(right);

    hands_visible = is_hand_visible(left, config) && is_hand_visible(right, config);
    if (!hands_visible) {
        return decision(
            GateReason::HandsOutOfFrame,
            config,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count);
    }

    const Point2 left_center = center_of(left);
    const Point2 right_center = center_of(right);
    hands_centered = is_point_centered(left_center, config) && is_point_centered(right_center, config);
    if (!hands_centered) {
        return decision(
            GateReason::HandsNotCentered,
            config,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count);
    }

    hands_separated = distance(left_center, right_center) >= config.min_hand_separation;
    if (!hands_separated) {
        return decision(
            GateReason::HandsTooClose,
            config,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count);
    }

    if (config.require_double_ok && !double_ok) {
        return decision(
            GateReason::NeedDoubleOK,
            config,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count);
    }

    return decision(
        GateReason::Ready,
        config,
        pose_ok,
        hands_centered,
        hands_visible,
        hands_separated,
        double_ok,
        hand_count);
}

CaptureGateDecision evaluate_labeled_capture_gate(
    const DoubleOKResult& result,
    const std::string& label,
    const CaptureGateConfig& config,
    const std::optional<GlassesPose>& glasses_pose) {
    if (label == "double_ok") {
        return evaluate_capture_gate(result, config, glasses_pose);
    }
    if (label != "not_double_ok") {
        throw std::invalid_argument("Unsupported capture label: " + label);
    }

    CaptureGateConfig geometry_config = config;
    geometry_config.require_double_ok = false;
    CaptureGateDecision geometry_decision = evaluate_capture_gate(result, geometry_config, glasses_pose);
    if (!geometry_decision.ready) {
        geometry_decision.gesture_ok = !result.double_ok;
        return geometry_decision;
    }
    if (result.double_ok) {
        return decision(
            GateReason::AvoidDoubleOK,
            geometry_config,
            geometry_decision.glasses_pose_ok,
            geometry_decision.hands_centered,
            geometry_decision.hands_visible,
            geometry_decision.hands_separated,
            result.double_ok,
            geometry_decision.hand_count,
            false);
    }
    return decision(
        GateReason::Ready,
        geometry_config,
        geometry_decision.glasses_pose_ok,
        geometry_decision.hands_centered,
        geometry_decision.hands_visible,
        geometry_decision.hands_separated,
        result.double_ok,
        geometry_decision.hand_count,
        true);
}

StereoCaptureGateDecision evaluate_stereo_capture_gate(
    const DoubleOKResult& left_result,
    const std::optional<DoubleOKResult>& right_result,
    const CaptureGateConfig& config,
    const std::optional<GlassesPose>& glasses_pose,
    StereoGateMode mode) {
    const CaptureGateDecision left_decision = evaluate_capture_gate(left_result, config, glasses_pose);
    if (mode == StereoGateMode::Left) {
        return {left_decision.ready, left_decision.reason, left_decision.prompt, left_decision, std::nullopt};
    }
    if (!right_result) {
        throw std::invalid_argument("right_result is required when stereo gate mode is both");
    }

    const CaptureGateDecision right_decision = evaluate_capture_gate(*right_result, config, glasses_pose);
    if (left_decision.ready && right_decision.ready) {
        return {true, GateReason::Ready, gate_prompt(GateReason::Ready), left_decision, right_decision};
    }
    if (!left_decision.ready && !right_decision.ready) {
        const std::string prompt = left_decision.reason == right_decision.reason
                                       ? "双目: " + left_decision.prompt
                                       : "左眼: " + left_decision.prompt + "; 右眼: " + right_decision.prompt;
        return {false, left_decision.reason, prompt, left_decision, right_decision};
    }
    const bool left_blocked = !left_decision.ready;
    const CaptureGateDecision blocked = left_blocked ? left_decision : right_decision;
    const std::string view = left_blocked ? "左眼" : "右眼";
    return {false, blocked.reason, view + ": " + blocked.prompt, left_decision, right_decision};
}

PromptSpeaker::PromptSpeaker(bool enabled, double min_interval_sec)
    : enabled_(enabled), min_interval_sec_(min_interval_sec) {
    if (!std::isfinite(min_interval_sec_) || min_interval_sec_ < 0.0) {
        throw std::invalid_argument("min_interval_sec must be finite and non-negative");
    }
}

void PromptSpeaker::emit(const std::string& prompt) {
    const double now = monotonic_seconds();
    if (prompt == last_prompt_ && now - last_time_ < min_interval_sec_) {
        return;
    }
    last_prompt_ = prompt;
    last_time_ = now;
    std::cout << prompt << '\n';
    if (enabled_) {
        std::cerr << "voice prompts are not spawned by the C++ build; prompt=" << prompt << '\n';
    }
}

}  // namespace double_ok_gesture
