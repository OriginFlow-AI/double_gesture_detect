#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <opencv2/core/mat.hpp>

#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/recognizer.hpp"

namespace double_ok_gesture {

class CaptureWriter {
public:
    explicit CaptureWriter(DataCaptureConfig config = {});

    std::optional<std::filesystem::path> maybe_save(
        const cv::Mat& frame,
        const DoubleOKResult& result,
        const CaptureGateDecision& decision,
        const std::string& backend);

    int saved_count() const;

private:
    DataCaptureConfig config_;
    double last_saved_time_ = -1.0;
    int saved_count_ = 0;
};

}  // namespace double_ok_gesture
