#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/hand_attribute_classifier.hpp"
#include "double_ok_gesture/model_io.hpp"

namespace double_ok_gesture {

struct HandBoundingBox {
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
    double detection_score = 0.0;
};

struct HandPrediction {
    std::string handedness = "Unknown";
    double ok_score = 0.0;
    bool is_ok = false;
    Landmarks landmarks{};
    bool landmarks_estimated = false;
    std::optional<LandmarkConfidences> landmark_confidences = std::nullopt;
    std::optional<HandBoundingBox> box = std::nullopt;
    double handedness_confidence = 0.0;
};

struct DoubleOKResult {
    std::vector<HandPrediction> hands;
    bool double_ok = false;
    bool stable_double_ok = false;
    int ok_count = 0;
};

struct DetectedHand {
    Landmarks landmarks{};
    std::string handedness = "Unknown";
    std::optional<double> ok_score;
    bool landmarks_estimated = false;
    std::optional<LandmarkConfidences> landmark_confidences = std::nullopt;
    std::optional<HandBoundingBox> box = std::nullopt;
    // Display/capture coordinates are normalized independently by image
    // width and height. Geometry classification instead needs equal x/y
    // units, normally source-image pixels, to preserve angles and distances.
    std::optional<Landmarks> metric_landmarks = std::nullopt;
    // Pose backends set this false when the distal points needed by the
    // fallback geometry rule do not meet their configured visibility limit.
    bool gesture_landmarks_reliable = true;
};

class OKHandClassifier {
public:
    explicit OKHandClassifier(double threshold = 0.68);
    OKHandClassifier(const std::filesystem::path& model_path, double threshold = 0.68);
    OKHandClassifier(LinearModelArtifact artifact, double threshold = 0.68);

    double score(const Landmarks& landmarks, const std::string& handedness = "") const;
    HandPrediction predict(const Landmarks& landmarks, const std::string& handedness = "") const;
    HandPrediction predict_with_score(
        const Landmarks& landmarks,
        const std::string& handedness,
        double ok_score,
        bool landmarks_estimated = false) const;
    bool uses_model() const;
    double threshold() const;

private:
    std::optional<LinearModelArtifact> artifact_;
    double threshold_ = 0.68;
};

class DoubleOKRecognizer {
public:
    DoubleOKRecognizer(
        OKHandClassifier classifier = OKHandClassifier(),
        std::size_t stable_window = 5,
        std::size_t stable_min_positive = 3,
        std::optional<HandAttributeClassifier> attribute_classifier =
            std::nullopt);

    DoubleOKResult process_hands(const std::vector<DetectedHand>& hands);
    void reset();

private:
    OKHandClassifier classifier_;
    std::size_t stable_min_positive_ = 3;
    std::deque<bool> history_;
    std::size_t history_limit_ = 5;
    std::optional<HandAttributeClassifier> attribute_classifier_;
};

double bounded_score(double score);

}  // namespace double_ok_gesture
