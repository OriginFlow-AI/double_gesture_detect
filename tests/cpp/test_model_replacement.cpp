#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/landmark_provider.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/runtime_pipeline.hpp"

namespace {

#define EXPECT_TRUE(expression)                                                \
    do {                                                                       \
        if (!(expression)) {                                                   \
            throw std::runtime_error(                                          \
                std::string("EXPECT_TRUE failed: ") + #expression);           \
        }                                                                      \
    } while (false)

#define EXPECT_EQ(lhs, rhs)                                                    \
    do {                                                                       \
        if (!((lhs) == (rhs))) {                                               \
            throw std::runtime_error(                                          \
                std::string("EXPECT_EQ failed: ") + #lhs + " != " + #rhs);   \
        }                                                                      \
    } while (false)

std::filesystem::path source_path(const std::string& relative) {
    return std::filesystem::path(DOUBLE_OK_SOURCE_DIR) / relative;
}

void test_onnx_is_the_default_and_only_inference_backend() {
    double_ok_gesture::RuntimeOptions options;
    EXPECT_EQ(
        options.landmark_backend,
        double_ok_gesture::LandmarkBackend::Onnx);
    EXPECT_EQ(
        double_ok_gesture::landmark_backend_from_string("onnx"),
        double_ok_gesture::LandmarkBackend::Onnx);
    EXPECT_EQ(
        std::string(double_ok_gesture::landmark_backend_value(
            double_ok_gesture::LandmarkBackend::Onnx)),
        std::string("onnx"));
    EXPECT_TRUE(double_ok_gesture::landmark_backend_available_in_current_build(
        double_ok_gesture::LandmarkBackend::Onnx));

    bool rejected = false;
    try {
        (void)double_ok_gesture::landmark_backend_from_string("unsupported");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    EXPECT_TRUE(rejected);
}

void test_default_config_selects_fp32_mediapipe_models() {
    const auto config = double_ok_gesture::load_runtime_config(
        source_path("configs/default.json"));
    EXPECT_EQ(
        config.onnx_hand.palm_model_path.string(),
        std::string(
            "models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx"));
    EXPECT_EQ(
        config.onnx_hand.hand_model_path.string(),
        std::string(
            "models/opencv_zoo/handpose_estimation_mediapipe_2023feb_opencv46.onnx"));
    EXPECT_TRUE(config.onnx_hand.hand_presence_threshold >= 0.8);
    EXPECT_TRUE(!config.onnx_hand.input_mirrored);
}

void test_models_load_with_opencv_and_black_frame_has_no_hands() {
    const auto palm = source_path(
        "models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx");
    const auto hand = source_path(
        "models/opencv_zoo/handpose_estimation_mediapipe_2023feb_opencv46.onnx");
    EXPECT_TRUE(std::filesystem::file_size(palm) > 3'000'000U);
    EXPECT_TRUE(std::filesystem::file_size(hand) > 3'000'000U);

    double_ok_gesture::MediaPipeOnnxHandLandmarkProvider provider({
        palm,
        hand,
        2,
        0.55,
        0.80,
        0.30,
        false,
    });
    EXPECT_EQ(provider.info().name, std::string("mediapipe-onnx-fp32"));
    const cv::Mat black(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_TRUE(provider.detect(black).empty());
}

void test_recognizer_preserves_model_metadata() {
    double_ok_gesture::DetectedHand detected;
    detected.handedness = "Left";
    detected.ok_score = 0.95;
    detected.handedness_confidence = 0.91;
    detected.box = double_ok_gesture::HandBoundingBox{
        0.1, 0.2, 0.4, 0.8, 0.88};

    double_ok_gesture::DoubleOKRecognizer recognizer;
    const auto result = recognizer.process_hands({detected});
    EXPECT_EQ(result.hands.size(), 1U);
    EXPECT_EQ(result.hands[0].handedness, std::string("Left"));
    EXPECT_TRUE(result.hands[0].is_ok);
    EXPECT_TRUE(result.hands[0].box.has_value());
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"onnx_is_the_default_and_only_inference_backend",
         test_onnx_is_the_default_and_only_inference_backend},
        {"default_config_selects_fp32_mediapipe_models",
         test_default_config_selects_fp32_mediapipe_models},
        {"models_load_with_opencv_and_black_frame_has_no_hands",
         test_models_load_with_opencv_and_black_frame_has_no_hands},
        {"recognizer_preserves_model_metadata",
         test_recognizer_preserves_model_metadata},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
