#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "double_ok_gesture/hand_detector.hpp"
#include "double_ok_gesture/recognizer.hpp"

#ifndef DOUBLE_OK_ENABLE_RKNN
#define DOUBLE_OK_ENABLE_RKNN 0
#endif

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

struct YoloV8OnnxPipelineConfig {
    std::filesystem::path model;
    int input_size = 640;
    int max_num_hands = 2;
    double min_detection_confidence = 0.55;
    double min_keypoint_visibility = 0.55;
    double nms_iou_threshold = 0.45;
    std::size_t min_reliable_keypoints = 8;
};

class YoloV8OnnxHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit YoloV8OnnxHandLandmarkProvider(
        YoloV8OnnxPipelineConfig config);
    ~YoloV8OnnxHandLandmarkProvider() override;

    YoloV8OnnxHandLandmarkProvider(
        const YoloV8OnnxHandLandmarkProvider&) = delete;
    YoloV8OnnxHandLandmarkProvider& operator=(
        const YoloV8OnnxHandLandmarkProvider&) = delete;

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#if DOUBLE_OK_ENABLE_RKNN
struct YoloV8RknnPipelineConfig {
    std::filesystem::path model;
    int input_size = 640;
    int max_num_hands = 2;
    double min_detection_confidence = 0.55;
    double min_keypoint_visibility = 0.55;
    double nms_iou_threshold = 0.45;
    std::size_t min_reliable_keypoints = 8;
};

class YoloV8RknnHandLandmarkProvider : public HandLandmarkProvider {
public:
    explicit YoloV8RknnHandLandmarkProvider(
        YoloV8RknnPipelineConfig config);
    ~YoloV8RknnHandLandmarkProvider() override;

    YoloV8RknnHandLandmarkProvider(
        const YoloV8RknnHandLandmarkProvider&) = delete;
    YoloV8RknnHandLandmarkProvider& operator=(
        const YoloV8RknnHandLandmarkProvider&) = delete;

    LandmarkProviderInfo info() const override;
    std::vector<DetectedHand> detect(const cv::Mat& frame_bgr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

class OpenCVDebugLandmarkProvider : public HandLandmarkProvider {
public:
    explicit OpenCVDebugLandmarkProvider(
        const HandDetectorConfig& config = {});

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
