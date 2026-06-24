#pragma once

#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/runtime.hpp"

namespace double_ok_gesture {

void draw_hand_tracking(cv::Mat& frame_bgr, const DoubleOKResult& result);
void draw_capture_guides(cv::Mat& frame_bgr, const CaptureGateDecision& decision, const CaptureGateConfig& config);
cv::Mat render_dashboard(
    const cv::Mat& camera_frame,
    const DoubleOKResult& result,
    const std::optional<CaptureGateDecision>& decision,
    const RuntimeSnapshot& snapshot,
    const std::string& camera_label,
    double target_fps,
    const std::string& model_label,
    int width,
    int height);

}  // namespace double_ok_gesture
