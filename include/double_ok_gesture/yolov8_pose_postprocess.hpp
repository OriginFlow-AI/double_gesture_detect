#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace double_ok_gesture {
namespace yolov8_pose {

inline constexpr std::size_t kHandKeypointCount = 21;
inline constexpr std::size_t kKeypointDimensionCount = 3;
inline constexpr std::size_t kDflBinCount = 16;
inline constexpr std::size_t kDflChannelCount = 4 * kDflBinCount;
inline constexpr std::size_t kClassCount = 1;
inline constexpr std::size_t kDetectionChannelCount =
    kDflChannelCount + kClassCount;
inline constexpr std::size_t kFlatBoxChannelCount = 4;
inline constexpr std::size_t kFlatPoseChannelCount =
    kFlatBoxChannelCount + kClassCount +
    kHandKeypointCount * kKeypointDimensionCount;
inline constexpr std::array<int, 3> kYoloV8PoseStrides = {8, 16, 32};

// A non-owning tensor view. The caller (normally the RKNN provider) keeps the
// backing storage alive for the duration of decode_three_scale_pose().
struct FloatTensorView {
  const float *data = nullptr;
  std::size_t value_count = 0;
  std::vector<std::size_t> shape;
};

enum class FeatureMapLayout {
  // [1, 65, grid_h, grid_w]
  kNchw,
  // [1, grid_h, grid_w, 65]
  kNhwc,
};

struct FeatureMapView {
  FloatTensorView tensor;
  FeatureMapLayout layout = FeatureMapLayout::kNchw;
};

enum class KeypointTensorLayout {
  // Rockchip RKOPT Pose output: [1, 21, 3, candidates].
  kKeypointDimensionCandidates,
  // [1, candidates, 21, 3].
  kCandidatesKeypointDimension,
  // [1, 63, candidates], with channels ordered x,y,visibility per point.
  kChannelsCandidates,
  // [1, candidates, 63], with channels ordered x,y,visibility per point.
  kCandidatesChannels,
};

enum class KeypointCoordinateEncoding {
  // RKOPT output after kpts_decode(): x/y are model-input pixels and the third
  // value is a probability.
  kDecodedPixelsAndVisibility,
  // Raw YOLOv8 Pose head output: x/y require anchor-grid decoding and the
  // third value is a visibility logit.
  kRawGridLogits,
};

struct KeypointTensorView {
  FloatTensorView tensor;
  KeypointTensorLayout layout =
      KeypointTensorLayout::kKeypointDimensionCandidates;
  KeypointCoordinateEncoding encoding =
      KeypointCoordinateEncoding::kDecodedPixelsAndVisibility;
};

struct LetterboxTransform {
  int source_width = 0;
  int source_height = 0;
  int input_width = 0;
  int input_height = 0;
  double scale_x = 0.0;
  double scale_y = 0.0;
  double pad_left = 0.0;
  double pad_top = 0.0;
};

struct Point2 {
  double x = 0.0;
  double y = 0.0;
};

struct Box {
  double xmin = 0.0;
  double ymin = 0.0;
  double xmax = 0.0;
  double ymax = 0.0;
};

struct Keypoint {
  double x = 0.0;
  double y = 0.0;
  // This is YOLO visibility/confidence, not MediaPipe z/depth.
  double visibility = 0.0;
  // Low-visibility points retain their coordinates for diagnostics, but the
  // provider can exclude them from gesture features using this flag.
  bool reliable = false;
};

struct HandPose {
  Box box;
  double score = 0.0;
  std::array<Keypoint, kHandKeypointCount> keypoints{};
  // Index in the concatenated P3/P4/P5 candidate axis. It selects the matching
  // entry in the separate RKOPT keypoint tensor.
  std::size_t candidate_index = 0;
};

struct DecodeOptions {
  int input_width = 640;
  int input_height = 640;
  double min_score = 0.25;
  double nms_iou_threshold = 0.45;
  double min_keypoint_visibility = 0.5;
  std::size_t max_hands = 2;
  bool clip_to_source = true;
};

struct ThreeScaleContract {
  std::array<int, 3> strides{};
  std::array<std::size_t, 3> candidate_counts{};
  std::size_t total_candidate_count = 0;
};

// Returns P3+P4+P5 candidates and rejects dimensions that cannot have exact
// strides 8/16/32. Examples: 224 -> 1029, 640 -> 8400.
std::size_t candidate_count_for_input(int input_width, int input_height);

// Validates batch/channel/grid dimensions and requires the RKOPT concatenation
// order P3/P4/P5 (strides 8/16/32). The separate keypoint candidate axis uses
// that fixed order, so accepting a permutation would attach points to boxes
// from another scale.
ThreeScaleContract
validate_three_scale_contract(const std::array<FeatureMapView, 3> &features,
                              int input_width, int input_height);

// Reproduces centered, aspect-ratio-preserving letterbox geometry. Integer
// resized dimensions are recorded through independent x/y scales so inverse
// mapping agrees with the actual resize operation even after rounding.
LetterboxTransform make_letterbox_transform(int source_width, int source_height,
                                            int input_width, int input_height);

Point2 inverse_letterbox_point(const Point2 &point,
                               const LetterboxTransform &transform,
                               bool clip_to_source = true);
Box inverse_letterbox_box(const Box &box, const LetterboxTransform &transform,
                          bool clip_to_source = true);

// Stable softmax integral used by YOLOv8's 16-bin Distribution Focal Loss
// head. Throws when a logit is non-finite.
double decode_dfl_expectation(const float *logits, std::size_t bin_count);

double box_iou(const Box &lhs, const Box &rhs);

// Deterministic, class-agnostic NMS for this single-class hand model. At most
// two hands are accepted by contract.
std::vector<HandPose> non_max_suppression(std::vector<HandPose> candidates,
                                          double iou_threshold,
                                          std::size_t max_hands);

// Decodes the standard single-output Ultralytics YOLOv8 Pose ONNX tensor.
// The accepted layouts are [1,68,candidates] and [1,candidates,68], where
// each candidate is xywh + one hand-class confidence + 21 x/y/visibility
// triplets. Coordinates are already decoded in model-input pixels and both
// confidence fields are probabilities, as produced by a normal Ultralytics
// ONNX export without end-to-end NMS.
std::vector<HandPose>
decode_flat_pose(const FloatTensorView &output,
                 const LetterboxTransform &letterbox,
                 const DecodeOptions &options = {});

// Decodes Rockchip-friendly YOLOv8 Pose outputs: three raw 65-channel
// detection feature maps plus one 21x3 keypoint tensor. DFL boxes and class
// logits are decoded on CPU, NMS is applied in model-input coordinates, and
// retained boxes/keypoints are inverse-letterboxed to source-image pixels.
std::vector<HandPose>
decode_three_scale_pose(const std::array<FeatureMapView, 3> &features,
                        const KeypointTensorView &keypoints,
                        const LetterboxTransform &letterbox,
                        const DecodeOptions &options = {});

} // namespace yolov8_pose
} // namespace double_ok_gesture
