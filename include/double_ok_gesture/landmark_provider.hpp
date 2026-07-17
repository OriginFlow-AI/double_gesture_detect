#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

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

struct MediaPipeOnnxPipelineConfig {
    std::filesystem::path palm_model;
    std::filesystem::path hand_model;
    int max_num_hands = 2;
    double palm_detection_threshold = 0.55;
    double hand_presence_threshold = 0.80;
    double palm_nms_threshold = 0.30;
    bool input_mirrored = false;
};

using MediaPipeRknnPipelineConfig = MediaPipeOnnxPipelineConfig;

class MediaPipeOnnxHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit MediaPipeOnnxHandLandmarkProvider(
        MediaPipeOnnxPipelineConfig config);
    ~MediaPipeOnnxHandLandmarkProvider() override;

    MediaPipeOnnxHandLandmarkProvider(
        const MediaPipeOnnxHandLandmarkProvider&) = delete;
    MediaPipeOnnxHandLandmarkProvider& operator=(
        const MediaPipeOnnxHandLandmarkProvider&) = delete;

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MediaPipeRknnHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit MediaPipeRknnHandLandmarkProvider(
        MediaPipeRknnPipelineConfig config);
    ~MediaPipeRknnHandLandmarkProvider() override;

    MediaPipeRknnHandLandmarkProvider(
        const MediaPipeRknnHandLandmarkProvider&) = delete;
    MediaPipeRknnHandLandmarkProvider& operator=(
        const MediaPipeRknnHandLandmarkProvider&) = delete;

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class JsonHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit JsonHandLandmarkProvider(std::filesystem::path path);

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    std::filesystem::path path_;
};

}  // namespace double_ok_gesture
