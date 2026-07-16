#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "double_ok_gesture/features.hpp"

namespace double_ok_gesture {

inline constexpr std::size_t kHandAttributeFeatureCount = 21 * 3;
using LandmarkConfidences = std::array<double, 21>;
using HandAttributeFeatures =
    std::array<double, kHandAttributeFeatureCount>;

struct HandAttributeModelArtifact {
    std::vector<double> mean;
    std::vector<double> scale;
    std::vector<double> handedness_coef;
    double handedness_intercept = 0.0;
    std::vector<double> ok_coef;
    double ok_intercept = 0.0;
};

struct HandAttributePrediction {
    std::string handedness = "Unknown";
    double handedness_confidence = 0.0;
    double ok_score = 0.0;
    bool is_ok = false;
};

// Produces wrist-centred, scale-normalized x/y coordinates followed by the
// YOLO keypoint visibility for each of the 21 MediaPipe-order points. z is
// deliberately excluded: YOLOv8-Pose's third component is visibility, not
// depth. input_mirrored unmirrors x before classification so output labels
// remain anatomical Left/Right.
HandAttributeFeatures hand_attribute_features(
    const Landmarks& landmarks,
    const LandmarkConfidences& visibility,
    bool input_mirrored = false);

HandAttributeModelArtifact load_hand_attribute_model(
    const std::filesystem::path& path);

class HandAttributeClassifier {
public:
    HandAttributeClassifier(
        const std::filesystem::path& model_path,
        double handedness_confidence_threshold = 0.65,
        double ok_threshold = 0.68,
        bool input_mirrored = false);
    HandAttributeClassifier(
        HandAttributeModelArtifact artifact,
        double handedness_confidence_threshold = 0.65,
        double ok_threshold = 0.68,
        bool input_mirrored = false);

    HandAttributePrediction predict(
        const Landmarks& landmarks,
        const LandmarkConfidences& visibility) const;

    double handedness_confidence_threshold() const;
    double ok_threshold() const;
    bool input_mirrored() const;

private:
    HandAttributeModelArtifact artifact_;
    double handedness_confidence_threshold_ = 0.65;
    double ok_threshold_ = 0.68;
    bool input_mirrored_ = false;
};

}  // namespace double_ok_gesture
