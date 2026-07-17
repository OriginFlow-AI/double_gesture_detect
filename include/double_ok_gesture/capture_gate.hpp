#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "double_ok_gesture/recognizer.hpp"

namespace double_ok_gesture {

enum class GateReason {
    Ready,
    GlassesPoseMissing,
    GlassesPoseBad,
    NeedTwoHands,
    HandsOutOfFrame,
    HandsNotCentered,
    HandsTooClose,
    NeedDoubleOK,
};

struct GlassesPose {
    std::optional<double> pitch;
    std::optional<double> roll;
    std::optional<double> yaw;
};

struct CaptureGateConfig {
    bool require_glasses_pose = false;
    double pitch_min = -20.0;
    double pitch_max = 20.0;
    double roll_min = -12.0;
    double roll_max = 12.0;
    double yaw_min = -25.0;
    double yaw_max = 25.0;
    double frame_margin = 0.03;
    double center_x_min = 0.20;
    double center_x_max = 0.80;
    double center_y_min = 0.18;
    double center_y_max = 0.82;
    double min_hand_separation = 0.16;
    bool use_stable_double_ok = true;
    bool require_double_ok = true;

    void validate() const;
};

struct CaptureGateDecision {
    bool ready = false;
    GateReason reason = GateReason::NeedTwoHands;
    std::string prompt;
    bool glasses_pose_ok = false;
    bool hands_centered = false;
    bool hands_visible = false;
    bool hands_separated = false;
    bool double_ok = false;
    bool gesture_ok = false;
    int hand_count = 0;
};

const char* gate_reason_value(GateReason reason);
const char* gate_prompt(GateReason reason);

std::optional<GlassesPose> load_glasses_pose(const std::filesystem::path& path);
CaptureGateDecision evaluate_capture_gate(
    const DoubleOKResult& result,
    const CaptureGateConfig& config = CaptureGateConfig(),
    const std::optional<GlassesPose>& glasses_pose = std::nullopt);
}  // namespace double_ok_gesture
