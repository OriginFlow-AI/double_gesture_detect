#include "double_ok_gesture/yolov8_pose_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace double_ok_gesture {
namespace yolov8_pose {
namespace {

bool finite(double value) { return std::isfinite(value); }

void require_positive_dimensions(int width, int height, const char *name) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(std::string(name) +
                                " dimensions must be positive");
  }
}

void require_probability(double value, const char *name) {
  if (!finite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and in [0,1]");
  }
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char *name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(std::string(name) + " size overflows size_t");
  }
  return lhs * rhs;
}

std::size_t shape_value_count(const std::vector<std::size_t> &shape,
                              const char *name) {
  if (shape.empty()) {
    throw std::invalid_argument(std::string(name) + " shape must not be empty");
  }
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    if (dimension == 0) {
      throw std::invalid_argument(std::string(name) +
                                  " shape contains a zero dimension");
    }
    count = checked_multiply(count, dimension, name);
  }
  return count;
}

void validate_tensor_metadata(const FloatTensorView &tensor, const char *name) {
  const std::size_t shape_count = shape_value_count(tensor.shape, name);
  if (shape_count != tensor.value_count) {
    throw std::invalid_argument(std::string(name) +
                                " value count does not match shape");
  }
}

void validate_tensor_view(const FloatTensorView &tensor, const char *name) {
  validate_tensor_metadata(tensor, name);
  if (tensor.data == nullptr) {
    throw std::invalid_argument(std::string(name) + " data must not be null");
  }
}

void validate_finite_values(const FloatTensorView &tensor, const char *name) {
  for (std::size_t index = 0; index < tensor.value_count; ++index) {
    if (!std::isfinite(tensor.data[index])) {
      throw std::invalid_argument(std::string(name) +
                                  " contains a non-finite value");
    }
  }
}

struct FeatureDimensions {
  std::size_t height = 0;
  std::size_t width = 0;
};

FeatureDimensions feature_dimensions(const FeatureMapView &feature,
                                     const char *name) {
  validate_tensor_metadata(feature.tensor, name);
  if (feature.tensor.shape.size() != 4 || feature.tensor.shape[0] != 1) {
    throw std::invalid_argument(std::string(name) +
                                " must be a rank-4, batch-1 tensor");
  }
  if (feature.layout == FeatureMapLayout::kNchw) {
    if (feature.tensor.shape[1] != kDetectionChannelCount) {
      throw std::invalid_argument(std::string(name) +
                                  " NCHW channel count must be 65");
    }
    return {feature.tensor.shape[2], feature.tensor.shape[3]};
  }
  if (feature.tensor.shape[3] != kDetectionChannelCount) {
    throw std::invalid_argument(std::string(name) +
                                " NHWC channel count must be 65");
  }
  return {feature.tensor.shape[1], feature.tensor.shape[2]};
}

float feature_value(const FeatureMapView &feature,
                    const FeatureDimensions &dimensions, std::size_t channel,
                    std::size_t y, std::size_t x) {
  if (feature.layout == FeatureMapLayout::kNchw) {
    return feature.tensor
        .data[(channel * dimensions.height + y) * dimensions.width + x];
  }
  return feature.tensor
      .data[(y * dimensions.width + x) * kDetectionChannelCount + channel];
}

std::size_t keypoint_candidate_count(const KeypointTensorView &keypoints) {
  const auto &shape = keypoints.tensor.shape;
  validate_tensor_view(keypoints.tensor, "keypoint tensor");
  switch (keypoints.layout) {
  case KeypointTensorLayout::kKeypointDimensionCandidates:
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != kHandKeypointCount ||
        shape[2] != kKeypointDimensionCount) {
      throw std::invalid_argument(
          "keypoint tensor must have shape [1,21,3,candidates]");
    }
    return shape[3];
  case KeypointTensorLayout::kCandidatesKeypointDimension:
    if (shape.size() != 4 || shape[0] != 1 || shape[2] != kHandKeypointCount ||
        shape[3] != kKeypointDimensionCount) {
      throw std::invalid_argument(
          "keypoint tensor must have shape [1,candidates,21,3]");
    }
    return shape[1];
  case KeypointTensorLayout::kChannelsCandidates:
    if (shape.size() != 3 || shape[0] != 1 ||
        shape[1] != kHandKeypointCount * kKeypointDimensionCount) {
      throw std::invalid_argument(
          "keypoint tensor must have shape [1,63,candidates]");
    }
    return shape[2];
  case KeypointTensorLayout::kCandidatesChannels:
    if (shape.size() != 3 || shape[0] != 1 ||
        shape[2] != kHandKeypointCount * kKeypointDimensionCount) {
      throw std::invalid_argument(
          "keypoint tensor must have shape [1,candidates,63]");
    }
    return shape[1];
  }
  throw std::invalid_argument("unsupported keypoint tensor layout");
}

float keypoint_value(const KeypointTensorView &keypoints,
                     std::size_t candidate_count, std::size_t candidate,
                     std::size_t keypoint, std::size_t dimension) {
  const std::size_t channel = keypoint * kKeypointDimensionCount + dimension;
  switch (keypoints.layout) {
  case KeypointTensorLayout::kKeypointDimensionCandidates:
  case KeypointTensorLayout::kChannelsCandidates:
    return keypoints.tensor.data[channel * candidate_count + candidate];
  case KeypointTensorLayout::kCandidatesKeypointDimension:
  case KeypointTensorLayout::kCandidatesChannels:
    return keypoints.tensor
        .data[candidate * (kHandKeypointCount * kKeypointDimensionCount) +
              channel];
  }
  throw std::invalid_argument("unsupported keypoint tensor layout");
}

double sigmoid(double value) {
  if (value >= 0.0) {
    const double exponential = std::exp(-value);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

void validate_letterbox(const LetterboxTransform &transform) {
  require_positive_dimensions(transform.source_width, transform.source_height,
                              "letterbox source");
  require_positive_dimensions(transform.input_width, transform.input_height,
                              "letterbox input");
  if (!finite(transform.scale_x) || !finite(transform.scale_y) ||
      transform.scale_x <= 0.0 || transform.scale_y <= 0.0 ||
      !finite(transform.pad_left) || !finite(transform.pad_top) ||
      transform.pad_left < 0.0 || transform.pad_top < 0.0) {
    throw std::invalid_argument(
        "letterbox transform contains invalid geometry");
  }
  const double right_edge =
      transform.pad_left + transform.scale_x * transform.source_width;
  const double bottom_edge =
      transform.pad_top + transform.scale_y * transform.source_height;
  constexpr double kRoundingTolerance = 1.01;
  if (right_edge > transform.input_width + kRoundingTolerance ||
      bottom_edge > transform.input_height + kRoundingTolerance) {
    throw std::invalid_argument("letterbox content exceeds model input bounds");
  }
}

double clamp_coordinate(double value, double upper_bound) {
  return std::clamp(value, 0.0, upper_bound);
}

void validate_box(const Box &box, const char *name) {
  if (!finite(box.xmin) || !finite(box.ymin) || !finite(box.xmax) ||
      !finite(box.ymax) || box.xmax < box.xmin || box.ymax < box.ymin) {
    throw std::invalid_argument(std::string(name) + " contains invalid bounds");
  }
}

void validate_options(const DecodeOptions &options) {
  require_positive_dimensions(options.input_width, options.input_height,
                              "decoder input");
  require_probability(options.min_score, "min_score");
  require_probability(options.nms_iou_threshold, "nms_iou_threshold");
  require_probability(options.min_keypoint_visibility,
                      "min_keypoint_visibility");
  if (options.min_reliable_keypoints > kHandKeypointCount) {
    throw std::invalid_argument(
        "min_reliable_keypoints must be between 0 and 21");
  }
  if (options.max_hands < 1 || options.max_hands > 2) {
    throw std::invalid_argument("max_hands must be 1 or 2");
  }
}

struct CandidateLocation {
  std::size_t grid_x = 0;
  std::size_t grid_y = 0;
  int stride = 0;
};

enum class FlatPoseLayout {
  kChannelsCandidates,
  kCandidatesChannels,
};

struct FlatPoseContract {
  FlatPoseLayout layout = FlatPoseLayout::kChannelsCandidates;
  std::size_t candidate_count = 0;
};

FlatPoseContract flat_pose_contract(const FloatTensorView &output,
                                    const DecodeOptions &options) {
  validate_tensor_view(output, "YOLOv8 ONNX output");
  if (output.shape.size() != 3 || output.shape[0] != 1) {
    throw std::invalid_argument(
        "YOLOv8 ONNX output must be a rank-3, batch-1 tensor");
  }

  FlatPoseContract contract;
  if (output.shape[1] == kFlatPoseChannelCount &&
      output.shape[2] != kFlatPoseChannelCount) {
    contract = {FlatPoseLayout::kChannelsCandidates, output.shape[2]};
  } else if (output.shape[2] == kFlatPoseChannelCount &&
             output.shape[1] != kFlatPoseChannelCount) {
    contract = {FlatPoseLayout::kCandidatesChannels, output.shape[1]};
  } else {
    throw std::invalid_argument(
        "YOLOv8 ONNX output must have shape [1,68,candidates] or "
        "[1,candidates,68] for one hand class and 21 keypoints");
  }

  const std::size_t expected_candidates =
      candidate_count_for_input(options.input_width, options.input_height);
  if (contract.candidate_count != expected_candidates) {
    throw std::invalid_argument(
        "YOLOv8 ONNX output candidate count does not match pose input size");
  }
  return contract;
}

float flat_pose_value(const FloatTensorView &output,
                      const FlatPoseContract &contract,
                      std::size_t candidate,
                      std::size_t channel) {
  if (contract.layout == FlatPoseLayout::kChannelsCandidates) {
    return output.data[channel * contract.candidate_count + candidate];
  }
  return output.data[candidate * kFlatPoseChannelCount + channel];
}

double output_probability(float value, const char *name) {
  constexpr double kProbabilityTolerance = 1e-5;
  if (!std::isfinite(value) || value < -kProbabilityTolerance ||
      value > 1.0 + kProbabilityTolerance) {
    throw std::invalid_argument(std::string(name) +
                                " must be a probability in [0,1]");
  }
  return std::clamp(static_cast<double>(value), 0.0, 1.0);
}

Keypoint decode_keypoint(const KeypointTensorView &keypoint_tensor,
                         std::size_t candidate_count,
                         std::size_t candidate_index,
                         std::size_t keypoint_index,
                         const CandidateLocation &location,
                         double minimum_visibility) {
  const double raw_x = keypoint_value(keypoint_tensor, candidate_count,
                                      candidate_index, keypoint_index, 0);
  const double raw_y = keypoint_value(keypoint_tensor, candidate_count,
                                      candidate_index, keypoint_index, 1);
  const double raw_visibility = keypoint_value(
      keypoint_tensor, candidate_count, candidate_index, keypoint_index, 2);
  if (!finite(raw_x) || !finite(raw_y) || !finite(raw_visibility)) {
    throw std::invalid_argument(
        "selected keypoint contains a non-finite value");
  }

  Keypoint output;
  if (keypoint_tensor.encoding == KeypointCoordinateEncoding::kRawGridLogits) {
    output.x =
        (raw_x * 2.0 + static_cast<double>(location.grid_x)) * location.stride;
    output.y =
        (raw_y * 2.0 + static_cast<double>(location.grid_y)) * location.stride;
    output.visibility = sigmoid(raw_visibility);
  } else {
    constexpr double kProbabilityTolerance = 1e-5;
    if (raw_visibility < -kProbabilityTolerance ||
        raw_visibility > 1.0 + kProbabilityTolerance) {
      throw std::invalid_argument(
          "decoded keypoint visibility is outside [0,1]");
    }
    output.x = raw_x;
    output.y = raw_y;
    output.visibility = std::clamp(raw_visibility, 0.0, 1.0);
  }
  output.reliable = output.visibility >= minimum_visibility;
  return output;
}

} // namespace

std::size_t candidate_count_for_input(int input_width, int input_height) {
  require_positive_dimensions(input_width, input_height, "YOLOv8 input");
  std::size_t total = 0;
  for (const int stride : kYoloV8PoseStrides) {
    if (input_width % stride != 0 || input_height % stride != 0) {
      throw std::invalid_argument(
          "YOLOv8 input dimensions must be divisible by 8, 16, and 32");
    }
    const auto width = static_cast<std::size_t>(input_width / stride);
    const auto height = static_cast<std::size_t>(input_height / stride);
    const std::size_t scale_count =
        checked_multiply(width, height, "YOLOv8 candidate");
    if (total > std::numeric_limits<std::size_t>::max() - scale_count) {
      throw std::overflow_error("YOLOv8 candidate count overflows size_t");
    }
    total += scale_count;
  }
  return total;
}

ThreeScaleContract
validate_three_scale_contract(const std::array<FeatureMapView, 3> &features,
                              int input_width, int input_height) {
  require_positive_dimensions(input_width, input_height, "YOLOv8 input");
  ThreeScaleContract contract;

  for (std::size_t index = 0; index < features.size(); ++index) {
    const std::string name = "feature map " + std::to_string(index);
    const FeatureDimensions dimensions =
        feature_dimensions(features[index], name.c_str());
    if (static_cast<std::size_t>(input_width) % dimensions.width != 0 ||
        static_cast<std::size_t>(input_height) % dimensions.height != 0) {
      throw std::invalid_argument(name +
                                  " grid does not evenly divide input shape");
    }
    const int stride_x = input_width / static_cast<int>(dimensions.width);
    const int stride_y = input_height / static_cast<int>(dimensions.height);
    if (stride_x != stride_y) {
      throw std::invalid_argument(name + " has unequal x/y stride");
    }
    if (stride_x != kYoloV8PoseStrides[index]) {
      throw std::invalid_argument(
          "three-scale tensors must be ordered P3/P4/P5 with strides "
          "8/16/32 to match the keypoint candidate axis");
    }
    contract.strides[index] = stride_x;
    contract.candidate_counts[index] = checked_multiply(
        dimensions.width, dimensions.height, "feature map candidate");
    if (contract.total_candidate_count >
        std::numeric_limits<std::size_t>::max() -
            contract.candidate_counts[index]) {
      throw std::overflow_error("feature map candidate count overflows size_t");
    }
    contract.total_candidate_count += contract.candidate_counts[index];
  }

  if (contract.total_candidate_count !=
      candidate_count_for_input(input_width, input_height)) {
    throw std::invalid_argument(
        "three-scale tensor candidate count does not match input shape");
  }
  return contract;
}

LetterboxTransform make_letterbox_transform(int source_width, int source_height,
                                            int input_width, int input_height) {
  require_positive_dimensions(source_width, source_height, "source image");
  require_positive_dimensions(input_width, input_height, "model input");
  const double scale =
      std::min(static_cast<double>(input_width) / source_width,
               static_cast<double>(input_height) / source_height);
  const int resized_width =
      std::max(1, static_cast<int>(std::round(source_width * scale)));
  const int resized_height =
      std::max(1, static_cast<int>(std::round(source_height * scale)));
  const int pad_left = (input_width - resized_width) / 2;
  const int pad_top = (input_height - resized_height) / 2;
  LetterboxTransform transform{
      source_width,
      source_height,
      input_width,
      input_height,
      static_cast<double>(resized_width) / source_width,
      static_cast<double>(resized_height) / source_height,
      static_cast<double>(pad_left),
      static_cast<double>(pad_top),
  };
  validate_letterbox(transform);
  return transform;
}

Point2 inverse_letterbox_point(const Point2 &point,
                               const LetterboxTransform &transform,
                               bool clip_to_source) {
  validate_letterbox(transform);
  if (!finite(point.x) || !finite(point.y)) {
    throw std::invalid_argument("point contains a non-finite coordinate");
  }
  Point2 output{(point.x - transform.pad_left) / transform.scale_x,
                (point.y - transform.pad_top) / transform.scale_y};
  if (clip_to_source) {
    output.x = clamp_coordinate(output.x, transform.source_width);
    output.y = clamp_coordinate(output.y, transform.source_height);
  }
  return output;
}

Box inverse_letterbox_box(const Box &box, const LetterboxTransform &transform,
                          bool clip_to_source) {
  validate_box(box, "box");
  const Point2 minimum =
      inverse_letterbox_point({box.xmin, box.ymin}, transform, clip_to_source);
  const Point2 maximum =
      inverse_letterbox_point({box.xmax, box.ymax}, transform, clip_to_source);
  return {minimum.x, minimum.y, maximum.x, maximum.y};
}

double decode_dfl_expectation(const float *logits, std::size_t bin_count) {
  if (logits == nullptr) {
    throw std::invalid_argument("DFL logits must not be null");
  }
  if (bin_count < 2) {
    throw std::invalid_argument("DFL requires at least two bins");
  }
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < bin_count; ++index) {
    if (!std::isfinite(logits[index])) {
      throw std::invalid_argument("DFL logits must all be finite");
    }
    maximum = std::max(maximum, static_cast<double>(logits[index]));
  }
  double normalizer = 0.0;
  double weighted_sum = 0.0;
  for (std::size_t index = 0; index < bin_count; ++index) {
    const double weight =
        std::exp(static_cast<double>(logits[index]) - maximum);
    normalizer += weight;
    weighted_sum += weight * index;
  }
  if (!finite(normalizer) || normalizer <= 0.0 || !finite(weighted_sum)) {
    throw std::invalid_argument("DFL softmax could not be normalized");
  }
  return weighted_sum / normalizer;
}

double box_iou(const Box &lhs, const Box &rhs) {
  validate_box(lhs, "left box");
  validate_box(rhs, "right box");
  const double intersection_width = std::max(
      0.0, std::min(lhs.xmax, rhs.xmax) - std::max(lhs.xmin, rhs.xmin));
  const double intersection_height = std::max(
      0.0, std::min(lhs.ymax, rhs.ymax) - std::max(lhs.ymin, rhs.ymin));
  const double intersection = intersection_width * intersection_height;
  const double lhs_area = (lhs.xmax - lhs.xmin) * (lhs.ymax - lhs.ymin);
  const double rhs_area = (rhs.xmax - rhs.xmin) * (rhs.ymax - rhs.ymin);
  const double union_area = lhs_area + rhs_area - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

std::vector<HandPose> non_max_suppression(std::vector<HandPose> candidates,
                                          double iou_threshold,
                                          std::size_t max_hands) {
  require_probability(iou_threshold, "NMS IoU threshold");
  if (max_hands < 1 || max_hands > 2) {
    throw std::invalid_argument("NMS max_hands must be 1 or 2");
  }
  for (const HandPose &candidate : candidates) {
    validate_box(candidate.box, "NMS candidate box");
    require_probability(candidate.score, "NMS candidate score");
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const HandPose &lhs, const HandPose &rhs) {
                     if (lhs.score != rhs.score) {
                       return lhs.score > rhs.score;
                     }
                     return lhs.candidate_index < rhs.candidate_index;
                   });

  std::vector<HandPose> retained;
  retained.reserve(std::min(max_hands, candidates.size()));
  for (HandPose &candidate : candidates) {
    bool suppressed = false;
    for (const HandPose &prior : retained) {
      if (box_iou(candidate.box, prior.box) > iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      retained.push_back(std::move(candidate));
      if (retained.size() == max_hands) {
        break;
      }
    }
  }
  return retained;
}

std::vector<HandPose>
decode_flat_pose(const FloatTensorView &output,
                 const LetterboxTransform &letterbox,
                 const DecodeOptions &options) {
  validate_options(options);
  validate_letterbox(letterbox);
  if (letterbox.input_width != options.input_width ||
      letterbox.input_height != options.input_height) {
    throw std::invalid_argument(
        "letterbox input dimensions do not match decoder options");
  }
  const FlatPoseContract contract = flat_pose_contract(output, options);
  validate_finite_values(output, "YOLOv8 ONNX output");

  std::vector<HandPose> candidates;
  for (std::size_t candidate = 0; candidate < contract.candidate_count;
       ++candidate) {
    const double score = output_probability(
        flat_pose_value(output, contract, candidate, kFlatBoxChannelCount),
        "YOLOv8 ONNX class confidence");
    if (score < options.min_score) {
      continue;
    }

    const double center_x = flat_pose_value(output, contract, candidate, 0);
    const double center_y = flat_pose_value(output, contract, candidate, 1);
    const double width = flat_pose_value(output, contract, candidate, 2);
    const double height = flat_pose_value(output, contract, candidate, 3);
    if (!finite(center_x) || !finite(center_y) || !finite(width) ||
        !finite(height) || width < 0.0 || height < 0.0) {
      throw std::invalid_argument(
          "YOLOv8 ONNX xywh box contains invalid values");
    }

    HandPose pose;
    pose.box = {center_x - width * 0.5, center_y - height * 0.5,
                center_x + width * 0.5, center_y + height * 0.5};
    pose.score = score;
    pose.candidate_index = candidate;
    for (std::size_t keypoint = 0; keypoint < kHandKeypointCount;
         ++keypoint) {
      const std::size_t offset =
          kFlatBoxChannelCount + kClassCount +
          keypoint * kKeypointDimensionCount;
      Keypoint &point = pose.keypoints[keypoint];
      point.x = flat_pose_value(output, contract, candidate, offset);
      point.y = flat_pose_value(output, contract, candidate, offset + 1);
      point.visibility = output_probability(
          flat_pose_value(output, contract, candidate, offset + 2),
          "YOLOv8 ONNX keypoint visibility");
      point.reliable =
          point.visibility >= options.min_keypoint_visibility;
    }
    const std::size_t reliable = static_cast<std::size_t>(std::count_if(
        pose.keypoints.begin(), pose.keypoints.end(),
        [](const Keypoint &point) { return point.reliable; }));
    if (reliable < options.min_reliable_keypoints) {
      continue;
    }
    candidates.push_back(std::move(pose));
  }

  std::vector<HandPose> retained = non_max_suppression(
      std::move(candidates), options.nms_iou_threshold, options.max_hands);
  for (HandPose &pose : retained) {
    pose.box =
        inverse_letterbox_box(pose.box, letterbox, options.clip_to_source);
    for (Keypoint &keypoint : pose.keypoints) {
      const Point2 mapped = inverse_letterbox_point(
          {keypoint.x, keypoint.y}, letterbox, options.clip_to_source);
      keypoint.x = mapped.x;
      keypoint.y = mapped.y;
    }
  }
  return retained;
}

std::vector<HandPose>
decode_three_scale_pose(const std::array<FeatureMapView, 3> &features,
                        const KeypointTensorView &keypoints,
                        const LetterboxTransform &letterbox,
                        const DecodeOptions &options) {
  validate_options(options);
  validate_letterbox(letterbox);
  if (letterbox.input_width != options.input_width ||
      letterbox.input_height != options.input_height) {
    throw std::invalid_argument(
        "letterbox input dimensions do not match decoder options");
  }
  const ThreeScaleContract contract = validate_three_scale_contract(
      features, options.input_width, options.input_height);
  const std::size_t keypoint_candidates = keypoint_candidate_count(keypoints);
  if (keypoint_candidates != contract.total_candidate_count) {
    throw std::invalid_argument(
        "keypoint candidate axis does not match detection feature maps");
  }
  for (std::size_t index = 0; index < features.size(); ++index) {
    const std::string name = "feature map " + std::to_string(index);
    validate_tensor_view(features[index].tensor, name.c_str());
    validate_finite_values(features[index].tensor, name.c_str());
  }
  validate_finite_values(keypoints.tensor, "keypoint tensor");

  std::vector<HandPose> candidates;
  std::size_t candidate_offset = 0;
  for (std::size_t scale_index = 0; scale_index < features.size();
       ++scale_index) {
    const FeatureMapView &feature = features[scale_index];
    const FeatureDimensions dimensions =
        feature_dimensions(feature, "feature map");
    const int stride = contract.strides[scale_index];
    for (std::size_t grid_y = 0; grid_y < dimensions.height; ++grid_y) {
      for (std::size_t grid_x = 0; grid_x < dimensions.width; ++grid_x) {
        const double score = sigmoid(feature_value(
            feature, dimensions, kDflChannelCount, grid_y, grid_x));
        if (score < options.min_score) {
          continue;
        }

        std::array<float, kDflChannelCount> dfl_logits{};
        for (std::size_t channel = 0; channel < kDflChannelCount; ++channel) {
          dfl_logits[channel] =
              feature_value(feature, dimensions, channel, grid_y, grid_x);
        }
        const double left =
            decode_dfl_expectation(dfl_logits.data(), kDflBinCount);
        const double top = decode_dfl_expectation(
            dfl_logits.data() + kDflBinCount, kDflBinCount);
        const double right = decode_dfl_expectation(
            dfl_logits.data() + 2 * kDflBinCount, kDflBinCount);
        const double bottom = decode_dfl_expectation(
            dfl_logits.data() + 3 * kDflBinCount, kDflBinCount);
        const double anchor_x = static_cast<double>(grid_x) + 0.5;
        const double anchor_y = static_cast<double>(grid_y) + 0.5;

        HandPose pose;
        pose.box = {(anchor_x - left) * stride, (anchor_y - top) * stride,
                    (anchor_x + right) * stride, (anchor_y + bottom) * stride};
        pose.score = score;
        pose.candidate_index =
            candidate_offset + grid_y * dimensions.width + grid_x;
        const CandidateLocation location{grid_x, grid_y, stride};
        for (std::size_t keypoint = 0; keypoint < kHandKeypointCount;
             ++keypoint) {
          pose.keypoints[keypoint] = decode_keypoint(
              keypoints, keypoint_candidates, pose.candidate_index, keypoint,
              location, options.min_keypoint_visibility);
        }
        const std::size_t reliable = static_cast<std::size_t>(std::count_if(
            pose.keypoints.begin(), pose.keypoints.end(),
            [](const Keypoint &point) { return point.reliable; }));
        if (reliable < options.min_reliable_keypoints) {
          continue;
        }
        candidates.push_back(std::move(pose));
      }
    }
    candidate_offset += contract.candidate_counts[scale_index];
  }

  std::vector<HandPose> retained = non_max_suppression(
      std::move(candidates), options.nms_iou_threshold, options.max_hands);
  for (HandPose &pose : retained) {
    pose.box =
        inverse_letterbox_box(pose.box, letterbox, options.clip_to_source);
    for (Keypoint &keypoint : pose.keypoints) {
      const Point2 mapped = inverse_letterbox_point(
          {keypoint.x, keypoint.y}, letterbox, options.clip_to_source);
      keypoint.x = mapped.x;
      keypoint.y = mapped.y;
    }
  }
  return retained;
}

} // namespace yolov8_pose
} // namespace double_ok_gesture
