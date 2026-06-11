#include "double_ok_gesture/landmark_provider.hpp"

#include <utility>

namespace double_ok_gesture {

NullHandLandmarkProvider::NullHandLandmarkProvider(std::string name, bool available)
    : info_{std::move(name), available, false} {}

LandmarkProviderInfo NullHandLandmarkProvider::info() const {
    return info_;
}

std::vector<DetectedHand> NullHandLandmarkProvider::detect(const cv::Mat&) {
    return {};
}

OpenCVDebugLandmarkProvider::OpenCVDebugLandmarkProvider(HandDetectorConfig config) : detector_(config) {}

LandmarkProviderInfo OpenCVDebugLandmarkProvider::info() const {
    return {"opencv-heuristic", true, true};
}

std::vector<DetectedHand> OpenCVDebugLandmarkProvider::detect(const cv::Mat& frame_bgr) {
    return detector_.detect(frame_bgr);
}

}  // namespace double_ok_gesture
