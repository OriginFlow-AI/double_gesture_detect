#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "double_ok_gesture/hand_detector.hpp"
#include "double_ok_gesture/recognizer.hpp"

namespace double_ok_gesture {

struct LandmarkProviderInfo {
    std::string name;
    bool available = false;
    bool estimated_landmarks = false;
};

class HandLandmarkProvider {
public:
    virtual ~HandLandmarkProvider() = default;
    virtual LandmarkProviderInfo info() const = 0;
    virtual std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) = 0;
};

class NullHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit NullHandLandmarkProvider(std::string name, bool available = false);

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    LandmarkProviderInfo info_;
};

class OpenCVDebugLandmarkProvider : public HandLandmarkProvider {
public:
    explicit OpenCVDebugLandmarkProvider(HandDetectorConfig config = {});

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    OpenCVHandDetector detector_;
};

class JsonHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit JsonHandLandmarkProvider(std::filesystem::path path);

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    std::filesystem::path path_;
};

class MediaPipePythonLandmarkProvider : public HandLandmarkProvider {
public:
    MediaPipePythonLandmarkProvider();
    MediaPipePythonLandmarkProvider(std::filesystem::path python_path, std::filesystem::path script_path);
    ~MediaPipePythonLandmarkProvider() override;

    MediaPipePythonLandmarkProvider(const MediaPipePythonLandmarkProvider&) = delete;
    MediaPipePythonLandmarkProvider& operator=(const MediaPipePythonLandmarkProvider&) = delete;

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace double_ok_gesture
