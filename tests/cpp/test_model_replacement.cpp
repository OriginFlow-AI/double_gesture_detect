#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/capture_gate.hpp"
#include "double_ok_gesture/hand_attribute_classifier.hpp"
#include "double_ok_gesture/landmark_provider.hpp"
#include "double_ok_gesture/model_contract.hpp"
#include "double_ok_gesture/recognizer.hpp"
#include "double_ok_gesture/runtime_pipeline.hpp"

namespace {

#define EXPECT_TRUE(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      throw std::runtime_error(std::string("EXPECT_TRUE failed: ") + #expr);   \
    }                                                                          \
  } while (false)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(lhs, rhs)                                                    \
  do {                                                                         \
    const auto lhs_value = (lhs);                                              \
    const auto rhs_value = (rhs);                                              \
    if (!(lhs_value == rhs_value)) {                                           \
      throw std::runtime_error(std::string("EXPECT_EQ failed: ") + #lhs +      \
                               " != " + #rhs);                                 \
    }                                                                          \
  } while (false)

#define EXPECT_NEAR(lhs, rhs, eps)                                             \
  do {                                                                         \
    if (std::abs((lhs) - (rhs)) > (eps)) {                                     \
      throw std::runtime_error(std::string("EXPECT_NEAR failed: ") + #lhs +    \
                               " != " + #rhs);                                 \
    }                                                                          \
  } while (false)

double_ok_gesture::Landmarks hand_landmarks(double direction) {
  double_ok_gesture::Landmarks points{};
  points[0] = {0.5, 0.8, 0.0};
  for (std::size_t index = 1; index < points.size(); ++index) {
    const double column = static_cast<double>((index - 1) % 4 + 1);
    const double row = static_cast<double>((index - 1) / 4 + 1);
    points[index] = {
        0.5 + direction * (0.02 * column + 0.004 * row),
        0.8 - 0.045 * row - 0.003 * column,
        0.0,
    };
  }
  return points;
}

double_ok_gesture::LandmarkConfidences full_visibility() {
  double_ok_gesture::LandmarkConfidences visibility{};
  visibility.fill(0.9);
  return visibility;
}

double_ok_gesture::HandAttributeModelArtifact attribute_artifact(
    double handedness_weight,
    double handedness_intercept,
    double ok_intercept) {
  double_ok_gesture::HandAttributeModelArtifact artifact;
  artifact.mean.assign(double_ok_gesture::kHandAttributeFeatureCount, 0.0);
  artifact.scale.assign(double_ok_gesture::kHandAttributeFeatureCount, 1.0);
  artifact.handedness_coef.assign(
      double_ok_gesture::kHandAttributeFeatureCount, 0.0);
  artifact.ok_coef.assign(
      double_ok_gesture::kHandAttributeFeatureCount, 0.0);
  // x of thumb CMC. The fixture makes it positive for Right and negative for
  // Left after wrist/scale normalization.
  artifact.handedness_coef[3] = handedness_weight;
  artifact.handedness_intercept = handedness_intercept;
  artifact.ok_intercept = ok_intercept;
  return artifact;
}

double_ok_gesture::DetectedHand detected_hand(double direction) {
  return {
      hand_landmarks(direction),
      "Unknown",
      std::nullopt,
      false,
      full_visibility(),
      double_ok_gesture::HandBoundingBox{
          0.2 + 0.3 * (direction > 0.0),
          0.2,
          0.45 + 0.3 * (direction > 0.0),
          0.75,
          0.9,
      },
  };
}

double_ok_gesture::HandPrediction gate_hand(
    double center_x, double center_y, bool is_ok = true) {
  double_ok_gesture::Landmarks points{};
  for (std::size_t index = 0; index < points.size(); ++index) {
    const double offset = -0.01 + 0.02 * static_cast<double>(index) /
                                      static_cast<double>(points.size() - 1);
    points[index] = {center_x + offset, center_y, 0.0};
  }
  return {
      "Unknown", is_ok ? 0.9 : 0.1, is_ok, points, false,
      std::nullopt, std::nullopt, 0.0};
}

double_ok_gesture::DoubleOKResult gate_result(
    std::vector<double_ok_gesture::HandPrediction> hands,
    bool stable = true) {
  const int ok_count = static_cast<int>(std::count_if(
      hands.begin(), hands.end(),
      [](const auto& hand) { return hand.is_ok; }));
  return {std::move(hands), ok_count == 2, stable, ok_count};
}

template <typename Function>
std::string thrown_message(Function function) {
  try {
    function();
  } catch (const std::exception& error) {
    return error.what();
  }
  throw std::runtime_error("expected an exception");
}

std::filesystem::path write_attribute_model_fixture() {
  const auto path = std::filesystem::temp_directory_path() /
                    "double_ok_hand_attribute_fixture.json";
  std::ofstream output(path);
  output << "{\"schema\":\"double_ok_hand_attribute_v1\","
            "\"feature_count\":63,\"mean\":[";
  for (std::size_t index = 0;
       index < double_ok_gesture::kHandAttributeFeatureCount;
       ++index) {
    output << (index == 0 ? "" : ",") << "0";
  }
  output << "],\"scale\":[";
  for (std::size_t index = 0;
       index < double_ok_gesture::kHandAttributeFeatureCount;
       ++index) {
    output << (index == 0 ? "" : ",") << "1";
  }
  output << "],\"handedness_coef\":[";
  for (std::size_t index = 0;
       index < double_ok_gesture::kHandAttributeFeatureCount;
       ++index) {
    output << (index == 0 ? "" : ",") << (index == 3 ? "20" : "0");
  }
  output << "],\"handedness_intercept\":0,\"ok_coef\":[";
  for (std::size_t index = 0;
       index < double_ok_gesture::kHandAttributeFeatureCount;
       ++index) {
    output << (index == 0 ? "" : ",") << "0";
  }
  output << "],\"ok_intercept\":10}";
  return path;
}

void test_attribute_features_use_visibility_not_z() {
  auto points = hand_landmarks(1.0);
  const auto visibility = full_visibility();
  const auto original = double_ok_gesture::hand_attribute_features(
      points, visibility, false);
  for (auto& point : points) {
    point.z = 999.0;
  }
  const auto changed_z = double_ok_gesture::hand_attribute_features(
      points, visibility, false);
  EXPECT_TRUE(original == changed_z);
  EXPECT_NEAR(original[2], 0.9, 1e-12);
  EXPECT_NEAR(original[5], 0.9, 1e-12);
}

void test_mirror_setting_unmirrors_x_only() {
  const auto points = hand_landmarks(1.0);
  const auto visibility = full_visibility();
  const auto normal = double_ok_gesture::hand_attribute_features(
      points, visibility, false);
  const auto mirrored = double_ok_gesture::hand_attribute_features(
      points, visibility, true);
  for (std::size_t index = 0; index < 21; ++index) {
    EXPECT_NEAR(normal[index * 3], -mirrored[index * 3], 1e-12);
    EXPECT_NEAR(normal[index * 3 + 1], mirrored[index * 3 + 1], 1e-12);
    EXPECT_NEAR(normal[index * 3 + 2], mirrored[index * 3 + 2], 1e-12);
  }
}

void test_attribute_classifier_left_right_ok_and_unknown() {
  const auto visibility = full_visibility();
  const double_ok_gesture::HandAttributeClassifier classifier(
      attribute_artifact(20.0, 0.0, 10.0), 0.65, 0.68, false);
  const auto right = classifier.predict(hand_landmarks(1.0), visibility);
  const auto left = classifier.predict(hand_landmarks(-1.0), visibility);
  EXPECT_EQ(right.handedness, std::string("Right"));
  EXPECT_EQ(left.handedness, std::string("Left"));
  EXPECT_TRUE(right.is_ok);
  EXPECT_TRUE(left.is_ok);
  EXPECT_TRUE(right.ok_score > 0.99);

  const double_ok_gesture::HandAttributeClassifier uncertain(
      attribute_artifact(0.0, 0.0, -10.0), 0.65, 0.68, false);
  const auto unknown = uncertain.predict(hand_landmarks(1.0), visibility);
  EXPECT_EQ(unknown.handedness, std::string("Unknown"));
  EXPECT_FALSE(unknown.is_ok);
}

void test_double_ok_requires_exact_left_and_right() {
  double_ok_gesture::HandAttributeClassifier attributes(
      attribute_artifact(20.0, 0.0, 10.0), 0.65, 0.68, false);
  double_ok_gesture::DoubleOKRecognizer recognizer(
      double_ok_gesture::OKHandClassifier(0.68),
      3,
      2,
      std::move(attributes));

  const std::vector<double_ok_gesture::DetectedHand> pair = {
      detected_hand(-1.0), detected_hand(1.0)};
  auto first = recognizer.process_hands(pair);
  EXPECT_TRUE(first.double_ok);
  EXPECT_FALSE(first.stable_double_ok);
  auto second = recognizer.process_hands(pair);
  EXPECT_TRUE(second.double_ok);
  EXPECT_TRUE(second.stable_double_ok);

  recognizer.reset();
  const auto duplicate = recognizer.process_hands(
      {detected_hand(1.0), detected_hand(1.0)});
  EXPECT_FALSE(duplicate.double_ok);
  EXPECT_FALSE(duplicate.stable_double_ok);
}

void test_capture_gate_all_block_reasons() {
  using double_ok_gesture::CaptureGateConfig;
  using double_ok_gesture::GateReason;
  using double_ok_gesture::GlassesPose;
  using double_ok_gesture::evaluate_capture_gate;

  const auto valid = gate_result({gate_hand(0.35, 0.5), gate_hand(0.65, 0.5)});
  CaptureGateConfig pose_config;
  pose_config.require_glasses_pose = true;
  EXPECT_EQ(
      evaluate_capture_gate(valid, pose_config).reason,
      GateReason::GlassesPoseMissing);
  EXPECT_EQ(
      evaluate_capture_gate(
          valid, pose_config, GlassesPose{30.0, 0.0, 0.0}).reason,
      GateReason::GlassesPoseBad);
  EXPECT_EQ(
      evaluate_capture_gate(gate_result({gate_hand(0.5, 0.5)})).reason,
      GateReason::NeedTwoHands);
  EXPECT_EQ(
      evaluate_capture_gate(
          gate_result({gate_hand(0.02, 0.5), gate_hand(0.65, 0.5)})).reason,
      GateReason::HandsOutOfFrame);
  EXPECT_EQ(
      evaluate_capture_gate(
          gate_result({gate_hand(0.10, 0.5), gate_hand(0.65, 0.5)})).reason,
      GateReason::HandsNotCentered);
  EXPECT_EQ(
      evaluate_capture_gate(
          gate_result({gate_hand(0.46, 0.5), gate_hand(0.54, 0.5)})).reason,
      GateReason::HandsTooClose);
  EXPECT_EQ(
      evaluate_capture_gate(gate_result(
          {gate_hand(0.35, 0.5), gate_hand(0.65, 0.5, false)})).reason,
      GateReason::NeedDoubleOK);
  EXPECT_EQ(evaluate_capture_gate(valid).reason, GateReason::Ready);
  EXPECT_EQ(
      double_ok_gesture::evaluate_labeled_capture_gate(
          valid, "not_double_ok").reason,
      GateReason::AvoidDoubleOK);
}

void test_attribute_path_rejects_missing_visibility() {
  double_ok_gesture::HandAttributeClassifier attributes(
      attribute_artifact(20.0, 0.0, 10.0));
  double_ok_gesture::DoubleOKRecognizer recognizer(
      double_ok_gesture::OKHandClassifier(), 5, 3, std::move(attributes));
  double_ok_gesture::DetectedHand hand;
  hand.landmarks = hand_landmarks(1.0);
  const std::string message = thrown_message(
      [&]() { (void)recognizer.process_hands({hand}); });
  EXPECT_TRUE(message.find("visibility") != std::string::npos);
}

void test_json_missing_handedness_is_unknown() {
  const auto path = std::filesystem::temp_directory_path() /
                    "double_ok_json_unknown_hand.json";
  {
    std::ofstream output(path);
    output << "{\"hands\":[{\"landmarks\":[";
    for (int index = 0; index < 21; ++index) {
      if (index != 0) {
        output << ',';
      }
      output << '[' << 0.1 + index * 0.01 << ','
             << 0.2 + index * 0.01 << ",0]";
    }
    output << "]}]}";
  }
  double_ok_gesture::JsonHandLandmarkProvider provider(path);
  const auto hands = provider.detect(cv::Mat{});
  EXPECT_EQ(hands.size(), 1U);
  EXPECT_EQ(hands[0].handedness, std::string("Unknown"));
}

void test_production_backend_has_no_fallback() {
  double_ok_gesture::RuntimeOptions options;
  options.landmark_backend = double_ok_gesture::LandmarkBackend::Rknn;
  options.config_path =
      std::filesystem::path(DOUBLE_OK_SOURCE_DIR) / "configs/default.json";
  const std::string missing_attribute =
      thrown_message([&]() { (void)double_ok_gesture::make_runtime(options); });
  EXPECT_TRUE(
      missing_attribute.find("attribute model") != std::string::npos);

  options.model_path = write_attribute_model_fixture();
  const std::string unavailable =
      thrown_message([&]() { (void)double_ok_gesture::make_runtime(options); });
  EXPECT_TRUE(
      unavailable.find("unavailable") != std::string::npos ||
      unavailable.find("AArch64") != std::string::npos);
  EXPECT_EQ(
      double_ok_gesture::landmark_backend_from_string("yolov8-rknn"),
      double_ok_gesture::LandmarkBackend::Rknn);
  EXPECT_EQ(
      std::string(double_ok_gesture::landmark_backend_value(
          double_ok_gesture::LandmarkBackend::Rknn)),
      std::string("yolov8-rknn"));
}

void test_attribute_model_fixture_loads_and_config_parses_mirror() {
  const auto model_path = write_attribute_model_fixture();
  const auto artifact =
      double_ok_gesture::load_hand_attribute_model(model_path);
  EXPECT_EQ(
      artifact.handedness_coef.size(),
      double_ok_gesture::kHandAttributeFeatureCount);

  const auto config_path = std::filesystem::temp_directory_path() /
                           "double_ok_mirrored_config.json";
  {
    std::ofstream output(config_path);
    output << "{\"input_mirrored\":true,"
              "\"handedness_confidence_threshold\":0.7}";
  }
  const auto config = double_ok_gesture::load_runtime_config(config_path);
  EXPECT_TRUE(config.recognizer.input_mirrored);
  EXPECT_NEAR(
      config.recognizer.handedness_confidence_threshold, 0.7, 1e-12);
}

void test_pose_manifest_and_sha_match_real_artifact() {
  const auto root = std::filesystem::path(DOUBLE_OK_SOURCE_DIR);
  const auto info = double_ok_gesture::validate_yolov8_pose_model_contract(
      root / "models/rk3588/hand_pose_640_fp.rknn",
      root / "models/rk3588/hand_pose_640_fp.rknn.manifest.json",
      640);
  EXPECT_EQ(
      info.model_kind, std::string("yolov8n_pose_hand_21_rkopt"));
  EXPECT_EQ(info.target_platform, std::string("rk3588"));
  EXPECT_EQ(
      info.sha256,
      std::string(
          "835be29087828e7992b7ae8c5b3fc9410b4c5f11afd31d9ff062aabf8f8ff25f"));
  EXPECT_FALSE(info.board_validation_completed);

  const std::string missing = thrown_message([&]() {
    (void)double_ok_gesture::validate_yolov8_pose_model_contract(
        root / "models/rk3588/not-present.rknn",
        root / "models/rk3588/hand_pose_640_fp.rknn.manifest.json",
        640);
  });
  EXPECT_TRUE(missing.find("not found") != std::string::npos);

  const auto mismatched_manifest =
      std::filesystem::temp_directory_path() /
      "double_ok_mismatched_pose_manifest.json";
  {
    std::ifstream input(
        root / "models/rk3588/hand_pose_640_fp.rknn.manifest.json");
    const std::string original{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::string changed = original;
    const auto sha_position = changed.find(info.sha256);
    EXPECT_TRUE(sha_position != std::string::npos);
    changed[sha_position] = changed[sha_position] == '0' ? '1' : '0';
    std::ofstream output(mismatched_manifest);
    output << changed;
  }
  const std::string mismatch = thrown_message([&]() {
    (void)double_ok_gesture::validate_yolov8_pose_model_contract(
        root / "models/rk3588/hand_pose_640_fp.rknn",
        mismatched_manifest,
        640);
  });
  EXPECT_TRUE(mismatch.find("SHA-256") != std::string::npos);
}

void test_missing_attribute_model_fails_clearly() {
  const auto missing = std::filesystem::temp_directory_path() /
                       "missing-hand-attribute-model.json";
  const std::string message = thrown_message(
      [&]() { (void)double_ok_gesture::load_hand_attribute_model(missing); });
  EXPECT_TRUE(message.find("not found") != std::string::npos);
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, void (*)()>> tests = {
      {"attribute_features_use_visibility_not_z",
       test_attribute_features_use_visibility_not_z},
      {"mirror_setting_unmirrors_x_only",
       test_mirror_setting_unmirrors_x_only},
      {"attribute_classifier_left_right_ok_and_unknown",
       test_attribute_classifier_left_right_ok_and_unknown},
      {"double_ok_requires_exact_left_and_right",
       test_double_ok_requires_exact_left_and_right},
      {"capture_gate_all_block_reasons",
       test_capture_gate_all_block_reasons},
      {"attribute_path_rejects_missing_visibility",
       test_attribute_path_rejects_missing_visibility},
      {"json_missing_handedness_is_unknown",
       test_json_missing_handedness_is_unknown},
      {"production_backend_has_no_fallback",
       test_production_backend_has_no_fallback},
      {"attribute_model_fixture_loads_and_config_parses_mirror",
       test_attribute_model_fixture_loads_and_config_parses_mirror},
      {"pose_manifest_and_sha_match_real_artifact",
       test_pose_manifest_and_sha_match_real_artifact},
      {"missing_attribute_model_fails_clearly",
       test_missing_attribute_model_fails_clearly},
  };

  int failed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  return failed == 0 ? 0 : 1;
}
