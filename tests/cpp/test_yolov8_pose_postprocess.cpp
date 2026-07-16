#include "double_ok_gesture/yolov8_pose_postprocess.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pose = double_ok_gesture::yolov8_pose;

namespace {

#define EXPECT_TRUE(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      throw std::runtime_error(std::string("EXPECT_TRUE failed: ") + #expr);   \
    }                                                                          \
  } while (false)

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

pose::FeatureMapView nchw_view(const std::vector<float> &values,
                               std::size_t height, std::size_t width) {
  return {{values.data(),
           values.size(),
           {1, pose::kDetectionChannelCount, height, width}},
          pose::FeatureMapLayout::kNchw};
}

pose::FeatureMapView nhwc_view(const std::vector<float> &values,
                               std::size_t height, std::size_t width) {
  return {{values.data(),
           values.size(),
           {1, height, width, pose::kDetectionChannelCount}},
          pose::FeatureMapLayout::kNhwc};
}

void set_nchw(std::vector<float> &values, std::size_t height, std::size_t width,
              std::size_t channel, std::size_t y, std::size_t x, float value) {
  values[(channel * height + y) * width + x] = value;
}

void set_nhwc(std::vector<float> &values, std::size_t width,
              std::size_t channel, std::size_t y, std::size_t x, float value) {
  values[(y * width + x) * pose::kDetectionChannelCount + channel] = value;
}

void set_candidate_dfl_nchw(std::vector<float> &values, std::size_t height,
                            std::size_t width, std::size_t y, std::size_t x,
                            std::size_t bin) {
  for (std::size_t side = 0; side < 4; ++side) {
    for (std::size_t index = 0; index < pose::kDflBinCount; ++index) {
      set_nchw(values, height, width, side * pose::kDflBinCount + index, y, x,
               index == bin ? 100.0F : -100.0F);
    }
  }
}

void set_candidate_dfl_nhwc(std::vector<float> &values, std::size_t width,
                            std::size_t y, std::size_t x, std::size_t bin) {
  for (std::size_t side = 0; side < 4; ++side) {
    for (std::size_t index = 0; index < pose::kDflBinCount; ++index) {
      set_nhwc(values, width, side * pose::kDflBinCount + index, y, x,
               index == bin ? 100.0F : -100.0F);
    }
  }
}

std::array<std::vector<float>, 3> make_nchw_224_features() {
  return {
      std::vector<float>(pose::kDetectionChannelCount * 28 * 28, -100.0F),
      std::vector<float>(pose::kDetectionChannelCount * 14 * 14, -100.0F),
      std::vector<float>(pose::kDetectionChannelCount * 7 * 7, -100.0F),
  };
}

std::array<pose::FeatureMapView, 3>
views_for_nchw_224(const std::array<std::vector<float>, 3> &features) {
  return {nchw_view(features[0], 28, 28), nchw_view(features[1], 14, 14),
          nchw_view(features[2], 7, 7)};
}

void set_flat_pose_value(std::vector<float> &values,
                         std::size_t candidate_count,
                         bool channels_first,
                         std::size_t candidate,
                         std::size_t channel,
                         float value) {
  const std::size_t index = channels_first
                                ? channel * candidate_count + candidate
                                : candidate * pose::kFlatPoseChannelCount +
                                      channel;
  values[index] = value;
}

void set_flat_pose_candidate(std::vector<float> &values,
                             std::size_t candidate_count,
                             bool channels_first,
                             std::size_t candidate,
                             float center_x,
                             float center_y,
                             float width,
                             float height,
                             float score) {
  set_flat_pose_value(
      values, candidate_count, channels_first, candidate, 0, center_x);
  set_flat_pose_value(
      values, candidate_count, channels_first, candidate, 1, center_y);
  set_flat_pose_value(
      values, candidate_count, channels_first, candidate, 2, width);
  set_flat_pose_value(
      values, candidate_count, channels_first, candidate, 3, height);
  set_flat_pose_value(
      values, candidate_count, channels_first, candidate, 4, score);
  for (std::size_t keypoint = 0; keypoint < pose::kHandKeypointCount;
       ++keypoint) {
    const std::size_t offset = 5 + keypoint * 3;
    set_flat_pose_value(
        values, candidate_count, channels_first, candidate, offset,
        60.0F + static_cast<float>(keypoint));
    set_flat_pose_value(
        values, candidate_count, channels_first, candidate, offset + 1,
        80.0F + static_cast<float>(keypoint));
    set_flat_pose_value(
        values, candidate_count, channels_first, candidate, offset + 2,
        keypoint == 1 ? 0.2F : 0.9F);
  }
}

void test_dynamic_candidate_counts_cover_224_and_640() {
  EXPECT_EQ(pose::candidate_count_for_input(224, 224), 1029U);
  EXPECT_EQ(pose::candidate_count_for_input(640, 640), 8400U);
  EXPECT_EQ(pose::candidate_count_for_input(640, 384), 5040U);

  bool threw = false;
  try {
    (void)pose::candidate_count_for_input(225, 224);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

void test_letterbox_inverse_mapping_uses_actual_resize_geometry() {
  const pose::LetterboxTransform transform =
      pose::make_letterbox_transform(1280, 720, 640, 640);
  EXPECT_NEAR(transform.scale_x, 0.5, 1e-12);
  EXPECT_NEAR(transform.scale_y, 0.5, 1e-12);
  EXPECT_NEAR(transform.pad_left, 0.0, 1e-12);
  EXPECT_NEAR(transform.pad_top, 140.0, 1e-12);

  const pose::Point2 point =
      pose::inverse_letterbox_point({320.0, 320.0}, transform);
  EXPECT_NEAR(point.x, 640.0, 1e-12);
  EXPECT_NEAR(point.y, 360.0, 1e-12);

  const pose::Box box = pose::inverse_letterbox_box(
      {-20.0, 100.0, 660.0, 600.0}, transform, true);
  EXPECT_NEAR(box.xmin, 0.0, 1e-12);
  EXPECT_NEAR(box.ymin, 0.0, 1e-12);
  EXPECT_NEAR(box.xmax, 1280.0, 1e-12);
  EXPECT_NEAR(box.ymax, 720.0, 1e-12);
}

void test_decodes_flat_onnx_output_layouts_and_maps_21_points() {
  constexpr std::size_t kCandidates = 1029;
  constexpr std::size_t kCandidate = 10;
  const pose::LetterboxTransform transform =
      pose::make_letterbox_transform(448, 224, 224, 224);
  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  options.min_score = 0.5;
  options.min_keypoint_visibility = 0.5;

  std::vector<pose::HandPose> decoded_by_layout[2];
  for (std::size_t layout = 0; layout < 2; ++layout) {
    const bool channels_first = layout == 0;
    std::vector<float> values(
        pose::kFlatPoseChannelCount * kCandidates, 0.0F);
    set_flat_pose_candidate(
        values, kCandidates, channels_first, kCandidate, 112.0F, 112.0F,
        80.0F, 40.0F, 0.9F);
    const std::vector<std::size_t> shape =
        channels_first
            ? std::vector<std::size_t>{
                  1, pose::kFlatPoseChannelCount, kCandidates}
            : std::vector<std::size_t>{
                  1, kCandidates, pose::kFlatPoseChannelCount};
    decoded_by_layout[layout] = pose::decode_flat_pose(
        {values.data(), values.size(), shape}, transform, options);
  }

  for (const auto &decoded : decoded_by_layout) {
    EXPECT_EQ(decoded.size(), 1U);
    EXPECT_EQ(decoded[0].candidate_index, kCandidate);
    EXPECT_NEAR(decoded[0].score, 0.9, 1e-6);
    EXPECT_NEAR(decoded[0].box.xmin, 144.0, 1e-12);
    EXPECT_NEAR(decoded[0].box.ymin, 72.0, 1e-12);
    EXPECT_NEAR(decoded[0].box.xmax, 304.0, 1e-12);
    EXPECT_NEAR(decoded[0].box.ymax, 152.0, 1e-12);
    EXPECT_NEAR(decoded[0].keypoints[0].x, 120.0, 1e-12);
    EXPECT_NEAR(decoded[0].keypoints[0].y, 48.0, 1e-12);
    EXPECT_NEAR(decoded[0].keypoints[20].x, 160.0, 1e-12);
    EXPECT_NEAR(decoded[0].keypoints[20].y, 88.0, 1e-12);
    EXPECT_TRUE(decoded[0].keypoints[0].reliable);
    EXPECT_TRUE(!decoded[0].keypoints[1].reliable);
  }
}

void test_flat_onnx_output_applies_nms_and_rejects_wrong_shape() {
  constexpr std::size_t kCandidates = 1029;
  std::vector<float> values(
      pose::kFlatPoseChannelCount * kCandidates, 0.0F);
  set_flat_pose_candidate(
      values, kCandidates, true, 10, 112.0F, 112.0F, 80.0F, 40.0F,
      0.9F);
  set_flat_pose_candidate(
      values, kCandidates, true, 11, 113.0F, 112.0F, 80.0F, 40.0F,
      0.8F);
  set_flat_pose_candidate(
      values, kCandidates, true, 100, 30.0F, 100.0F, 20.0F, 20.0F,
      0.7F);
  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  options.min_score = 0.5;
  options.max_hands = 2;
  const auto output = pose::decode_flat_pose(
      {values.data(), values.size(),
       {1, pose::kFlatPoseChannelCount, kCandidates}},
      pose::make_letterbox_transform(224, 224, 224, 224), options);
  EXPECT_EQ(output.size(), 2U);
  EXPECT_EQ(output[0].candidate_index, 10U);
  EXPECT_EQ(output[1].candidate_index, 100U);

  bool wrong_shape_threw = false;
  try {
    (void)pose::decode_flat_pose(
        {values.data(), values.size(),
         {1, pose::kFlatPoseChannelCount - 1, kCandidates}},
        pose::make_letterbox_transform(224, 224, 224, 224), options);
  } catch (const std::invalid_argument &) {
    wrong_shape_threw = true;
  }
  EXPECT_TRUE(wrong_shape_threw);
}

void test_low_reliability_candidate_cannot_displace_two_valid_hands() {
  constexpr std::size_t kCandidates = 1029;
  std::vector<float> values(
      pose::kFlatPoseChannelCount * kCandidates, 0.0F);
  set_flat_pose_candidate(
      values, kCandidates, true, 10, 30.0F, 40.0F, 20.0F, 20.0F,
      0.95F);
  set_flat_pose_candidate(
      values, kCandidates, true, 100, 100.0F, 40.0F, 20.0F, 20.0F,
      0.90F);
  set_flat_pose_candidate(
      values, kCandidates, true, 200, 180.0F, 40.0F, 20.0F, 20.0F,
      0.85F);
  for (std::size_t keypoint = 0; keypoint < pose::kHandKeypointCount;
       ++keypoint) {
    set_flat_pose_value(
        values, kCandidates, true, 10, 5 + keypoint * 3 + 2, 0.1F);
  }

  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  options.min_score = 0.5;
  options.min_keypoint_visibility = 0.5;
  options.min_reliable_keypoints = 8;
  options.max_hands = 2;
  const auto output = pose::decode_flat_pose(
      {values.data(), values.size(),
       {1, pose::kFlatPoseChannelCount, kCandidates}},
      pose::make_letterbox_transform(224, 224, 224, 224), options);
  EXPECT_EQ(output.size(), 2U);
  EXPECT_EQ(output[0].candidate_index, 100U);
  EXPECT_EQ(output[1].candidate_index, 200U);
}

void test_dfl_softmax_integral_is_stable() {
  std::array<float, pose::kDflBinCount> logits{};
  logits.fill(-100.0F);
  logits[2] = 100.0F;
  EXPECT_NEAR(pose::decode_dfl_expectation(logits.data(), logits.size()), 2.0,
              1e-12);

  logits[4] = std::numeric_limits<float>::quiet_NaN();
  bool threw = false;
  try {
    (void)pose::decode_dfl_expectation(logits.data(), logits.size());
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

void test_decodes_rkopt_nchw_dfl_and_21_visibility_values() {
  auto feature_values = make_nchw_224_features();
  constexpr std::size_t kGridY = 3;
  constexpr std::size_t kGridX = 4;
  constexpr std::size_t kCandidate = kGridY * 28 + kGridX;
  set_candidate_dfl_nchw(feature_values[0], 28, 28, kGridY, kGridX, 2);
  set_nchw(feature_values[0], 28, 28, pose::kDflChannelCount, kGridY, kGridX,
           static_cast<float>(std::log(9.0)));
  const auto feature_views = views_for_nchw_224(feature_values);

  constexpr std::size_t kCandidates = 1029;
  std::vector<float> keypoint_values(
      pose::kHandKeypointCount * pose::kKeypointDimensionCount * kCandidates,
      0.0F);
  for (std::size_t keypoint = 0; keypoint < pose::kHandKeypointCount;
       ++keypoint) {
    const std::size_t channel = keypoint * 3;
    keypoint_values[(channel + 0) * kCandidates + kCandidate] =
        static_cast<float>(20.0 + keypoint);
    keypoint_values[(channel + 1) * kCandidates + kCandidate] =
        static_cast<float>(30.0 + keypoint);
    keypoint_values[(channel + 2) * kCandidates + kCandidate] =
        keypoint == 1 ? 0.2F : 0.9F;
  }
  const pose::KeypointTensorView keypoints{
      {keypoint_values.data(), keypoint_values.size(), {1, 21, 3, 1029}},
      pose::KeypointTensorLayout::kKeypointDimensionCandidates,
      pose::KeypointCoordinateEncoding::kDecodedPixelsAndVisibility};
  const pose::LetterboxTransform identity =
      pose::make_letterbox_transform(224, 224, 224, 224);
  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  options.min_score = 0.5;
  options.min_keypoint_visibility = 0.5;

  const std::vector<pose::HandPose> output = pose::decode_three_scale_pose(
      feature_views, keypoints, identity, options);
  EXPECT_EQ(output.size(), 1U);
  EXPECT_EQ(output[0].candidate_index, kCandidate);
  EXPECT_NEAR(output[0].score, 0.9, 1e-6);
  EXPECT_NEAR(output[0].box.xmin, 20.0, 1e-12);
  EXPECT_NEAR(output[0].box.ymin, 12.0, 1e-12);
  EXPECT_NEAR(output[0].box.xmax, 52.0, 1e-12);
  EXPECT_NEAR(output[0].box.ymax, 44.0, 1e-12);
  EXPECT_NEAR(output[0].keypoints[0].x, 20.0, 1e-12);
  EXPECT_NEAR(output[0].keypoints[0].y, 30.0, 1e-12);
  EXPECT_NEAR(output[0].keypoints[0].visibility, 0.9, 1e-6);
  EXPECT_TRUE(output[0].keypoints[0].reliable);
  EXPECT_NEAR(output[0].keypoints[1].visibility, 0.2, 1e-6);
  EXPECT_TRUE(!output[0].keypoints[1].reliable);
  for (const auto &keypoint : output[0].keypoints) {
    EXPECT_TRUE(keypoint.x >= 0.0 && keypoint.x <= 224.0);
    EXPECT_TRUE(keypoint.y >= 0.0 && keypoint.y <= 224.0);
    EXPECT_TRUE(keypoint.visibility >= 0.0 && keypoint.visibility <= 1.0);
  }
}

void test_decodes_raw_keypoint_logits_and_nhwc_feature_map() {
  std::vector<float> p3(pose::kDetectionChannelCount * 28 * 28, -100.0F);
  std::vector<float> p4(pose::kDetectionChannelCount * 14 * 14, -100.0F);
  std::vector<float> p5(pose::kDetectionChannelCount * 7 * 7, -100.0F);
  constexpr std::size_t kGridY = 3;
  constexpr std::size_t kGridX = 4;
  constexpr std::size_t kCandidate = kGridY * 28 + kGridX;
  set_candidate_dfl_nhwc(p3, 28, kGridY, kGridX, 1);
  set_nhwc(p3, 28, pose::kDflChannelCount, kGridY, kGridX, 10.0F);
  const std::array<pose::FeatureMapView, 3> features = {
      nhwc_view(p3, 28, 28), nhwc_view(p4, 14, 14), nhwc_view(p5, 7, 7)};

  std::vector<float> raw_keypoints(1029 * 63, 0.0F);
  for (std::size_t keypoint = 0; keypoint < pose::kHandKeypointCount;
       ++keypoint) {
    const std::size_t offset = kCandidate * 63 + keypoint * 3;
    raw_keypoints[offset + 0] = 0.25F;
    raw_keypoints[offset + 1] = 0.5F;
    raw_keypoints[offset + 2] = 0.0F;
  }
  const pose::KeypointTensorView keypoints{
      {raw_keypoints.data(), raw_keypoints.size(), {1, 1029, 63}},
      pose::KeypointTensorLayout::kCandidatesChannels,
      pose::KeypointCoordinateEncoding::kRawGridLogits};
  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  options.min_score = 0.5;
  options.min_keypoint_visibility = 0.5;
  const auto output = pose::decode_three_scale_pose(
      features, keypoints, pose::make_letterbox_transform(224, 224, 224, 224),
      options);
  EXPECT_EQ(output.size(), 1U);
  EXPECT_NEAR(output[0].keypoints[0].x, 36.0, 1e-12);
  EXPECT_NEAR(output[0].keypoints[0].y, 32.0, 1e-12);
  EXPECT_NEAR(output[0].keypoints[0].visibility, 0.5, 1e-12);
  EXPECT_TRUE(output[0].keypoints[0].reliable);
}

void test_nms_is_deterministic_and_never_returns_more_than_two_hands() {
  pose::HandPose first;
  first.box = {0.0, 0.0, 10.0, 10.0};
  first.score = 0.9;
  first.candidate_index = 4;
  pose::HandPose overlapping;
  overlapping.box = {1.0, 1.0, 11.0, 11.0};
  overlapping.score = 0.8;
  overlapping.candidate_index = 2;
  pose::HandPose second;
  second.box = {20.0, 0.0, 30.0, 10.0};
  second.score = 0.7;
  second.candidate_index = 1;
  pose::HandPose third;
  third.box = {40.0, 0.0, 50.0, 10.0};
  third.score = 0.6;
  third.candidate_index = 0;

  EXPECT_NEAR(pose::box_iou(first.box, overlapping.box), 81.0 / 119.0, 1e-12);
  const auto output =
      pose::non_max_suppression({third, overlapping, second, first}, 0.5, 2);
  EXPECT_EQ(output.size(), 2U);
  EXPECT_EQ(output[0].candidate_index, 4U);
  EXPECT_EQ(output[1].candidate_index, 1U);
}

void test_rejects_illegal_feature_and_keypoint_shapes() {
  const std::array<pose::FeatureMapView, 3> shape_only_features = {
      pose::FeatureMapView{{nullptr,
                            pose::kDetectionChannelCount * 28 * 28,
                            {1, pose::kDetectionChannelCount, 28, 28}},
                           pose::FeatureMapLayout::kNchw},
      pose::FeatureMapView{{nullptr,
                            pose::kDetectionChannelCount * 14 * 14,
                            {1, pose::kDetectionChannelCount, 14, 14}},
                           pose::FeatureMapLayout::kNchw},
      pose::FeatureMapView{{nullptr,
                            pose::kDetectionChannelCount * 7 * 7,
                            {1, pose::kDetectionChannelCount, 7, 7}},
                           pose::FeatureMapLayout::kNchw},
  };
  const auto shape_only_contract =
      pose::validate_three_scale_contract(shape_only_features, 224, 224);
  EXPECT_EQ(shape_only_contract.total_candidate_count, 1029U);

  std::vector<float> bad_channel_values(64 * 28 * 28, 0.0F);
  const pose::FeatureMapView bad_channels{
      {bad_channel_values.data(), bad_channel_values.size(), {1, 64, 28, 28}},
      pose::FeatureMapLayout::kNchw};
  bool bad_channel_threw = false;
  try {
    (void)pose::validate_three_scale_contract(
        {bad_channels, bad_channels, bad_channels}, 224, 224);
  } catch (const std::invalid_argument &) {
    bad_channel_threw = true;
  }
  EXPECT_TRUE(bad_channel_threw);

  auto feature_values = make_nchw_224_features();
  const auto feature_views = views_for_nchw_224(feature_values);
  bool duplicate_stride_threw = false;
  try {
    (void)pose::validate_three_scale_contract(
        {feature_views[0], feature_views[0], feature_views[2]}, 224, 224);
  } catch (const std::invalid_argument &) {
    duplicate_stride_threw = true;
  }
  EXPECT_TRUE(duplicate_stride_threw);

  bool permuted_scale_threw = false;
  try {
    (void)pose::validate_three_scale_contract(
        {feature_views[1], feature_views[0], feature_views[2]}, 224, 224);
  } catch (const std::invalid_argument &) {
    permuted_scale_threw = true;
  }
  EXPECT_TRUE(permuted_scale_threw);

  std::vector<float> short_keypoints(21 * 3 * 1028, 0.0F);
  const pose::KeypointTensorView wrong_candidate_axis{
      {short_keypoints.data(), short_keypoints.size(), {1, 21, 3, 1028}},
      pose::KeypointTensorLayout::kKeypointDimensionCandidates,
      pose::KeypointCoordinateEncoding::kDecodedPixelsAndVisibility};
  pose::DecodeOptions options;
  options.input_width = 224;
  options.input_height = 224;
  bool keypoint_shape_threw = false;
  try {
    (void)pose::decode_three_scale_pose(
        feature_views, wrong_candidate_axis,
        pose::make_letterbox_transform(224, 224, 224, 224), options);
  } catch (const std::invalid_argument &) {
    keypoint_shape_threw = true;
  }
  EXPECT_TRUE(keypoint_shape_threw);

  feature_values[2][0] = std::numeric_limits<float>::infinity();
  const auto nonfinite_views = views_for_nchw_224(feature_values);
  std::vector<float> valid_keypoints(21 * 3 * 1029, 0.0F);
  const pose::KeypointTensorView valid_keypoint_view{
      {valid_keypoints.data(), valid_keypoints.size(), {1, 21, 3, 1029}},
      pose::KeypointTensorLayout::kKeypointDimensionCandidates,
      pose::KeypointCoordinateEncoding::kDecodedPixelsAndVisibility};
  bool null_data_threw = false;
  try {
    (void)pose::decode_three_scale_pose(
        shape_only_features, valid_keypoint_view,
        pose::make_letterbox_transform(224, 224, 224, 224), options);
  } catch (const std::invalid_argument &) {
    null_data_threw = true;
  }
  EXPECT_TRUE(null_data_threw);

  bool nonfinite_threw = false;
  try {
    (void)pose::decode_three_scale_pose(
        nonfinite_views, valid_keypoint_view,
        pose::make_letterbox_transform(224, 224, 224, 224), options);
  } catch (const std::invalid_argument &) {
    nonfinite_threw = true;
  }
  EXPECT_TRUE(nonfinite_threw);
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, void (*)()>> tests = {
      {"dynamic_candidate_counts_cover_224_and_640",
       test_dynamic_candidate_counts_cover_224_and_640},
      {"letterbox_inverse_mapping_uses_actual_resize_geometry",
       test_letterbox_inverse_mapping_uses_actual_resize_geometry},
      {"decodes_flat_onnx_output_layouts_and_maps_21_points",
       test_decodes_flat_onnx_output_layouts_and_maps_21_points},
      {"flat_onnx_output_applies_nms_and_rejects_wrong_shape",
       test_flat_onnx_output_applies_nms_and_rejects_wrong_shape},
      {"low_reliability_candidate_cannot_displace_two_valid_hands",
       test_low_reliability_candidate_cannot_displace_two_valid_hands},
      {"dfl_softmax_integral_is_stable", test_dfl_softmax_integral_is_stable},
      {"decodes_rkopt_nchw_dfl_and_21_visibility_values",
       test_decodes_rkopt_nchw_dfl_and_21_visibility_values},
      {"decodes_raw_keypoint_logits_and_nhwc_feature_map",
       test_decodes_raw_keypoint_logits_and_nhwc_feature_map},
      {"nms_is_deterministic_and_never_returns_more_than_two_hands",
       test_nms_is_deterministic_and_never_returns_more_than_two_hands},
      {"rejects_illegal_feature_and_keypoint_shapes",
       test_rejects_illegal_feature_and_keypoint_shapes},
  };

  int failed = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      ++failed;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  return failed == 0 ? 0 : 1;
}
