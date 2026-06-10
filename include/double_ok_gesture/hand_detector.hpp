#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/recognizer.hpp"

namespace double_ok_gesture {

struct HandDetectorConfig {
    int max_num_hands = 2;
    double min_area_ratio = 0.006;
    double skin_mask_blur = 3.0;
};

class OpenCVHandDetector {
public:
    explicit OpenCVHandDetector(HandDetectorConfig config = {});

    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) const;

private:
    HandDetectorConfig config_;
};

}  // namespace double_ok_gesture
