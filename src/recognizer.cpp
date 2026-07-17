#include "double_ok_gesture/recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

void validate_threshold(double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("threshold must be finite and in [0.0, 1.0]");
    }
}

}  // namespace

OKHandClassifier::OKHandClassifier(double threshold) : threshold_(threshold) {
    validate_threshold(threshold_);
}

double OKHandClassifier::score(const Landmarks& landmarks, const std::string& handedness) const {
    return rule_ok_score(landmarks, handedness);
}

HandPrediction OKHandClassifier::predict(const Landmarks& landmarks, const std::string& handedness) const {
    const double ok = score(landmarks, handedness);
    return predict_with_score(landmarks, handedness, ok, false);
}

HandPrediction OKHandClassifier::predict_with_score(
    const Landmarks& landmarks,
    const std::string& handedness,
    double ok_score,
    bool landmarks_estimated) const {
    const double ok = bounded_score(ok_score);
    return {
        handedness.empty() ? "Unknown" : handedness,
        ok,
        ok >= threshold_,
        landmarks,
        landmarks_estimated,
        std::nullopt,
        std::nullopt,
        0.0,
    };
}

double OKHandClassifier::threshold() const {
    return threshold_;
}

DoubleOKRecognizer::DoubleOKRecognizer(
    OKHandClassifier classifier,
    std::size_t stable_window,
    std::size_t stable_min_positive)
    : classifier_(std::move(classifier)), stable_min_positive_(stable_min_positive),
      history_limit_(stable_window) {
    if (stable_window < 1) {
        throw std::invalid_argument("stable_window must be at least 1");
    }
    if (stable_min_positive < 1 || stable_min_positive > stable_window) {
        throw std::invalid_argument("stable_min_positive must be between 1 and stable_window");
    }
}

DoubleOKResult DoubleOKRecognizer::process_hands(const std::vector<DetectedHand>& hands) {
    DoubleOKResult result;
    result.hands.reserve(hands.size());
    for (const DetectedHand& hand : hands) {
        if (hand.ok_score) {
            HandPrediction prediction = classifier_.predict_with_score(
                hand.landmarks,
                hand.handedness,
                *hand.ok_score,
                hand.landmarks_estimated);
            prediction.landmark_confidences = hand.landmark_confidences;
            prediction.box = hand.box;
            prediction.handedness_confidence =
                hand.handedness_confidence;
            result.hands.push_back(std::move(prediction));
        } else {
            const Landmarks& classifier_landmarks =
                hand.metric_landmarks
                    ? *hand.metric_landmarks
                    : hand.landmarks;
            HandPrediction prediction =
                !hand.gesture_landmarks_reliable
                    ? classifier_.predict_with_score(
                          classifier_landmarks,
                          hand.handedness,
                          0.0,
                          hand.landmarks_estimated)
                    : classifier_.predict(
                          classifier_landmarks, hand.handedness);
            // metric_landmarks are classification-only. Preserve normalized
            // coordinates for overlays and capture-gate geometry.
            prediction.landmarks = hand.landmarks;
            prediction.landmarks_estimated = hand.landmarks_estimated;
            prediction.landmark_confidences = hand.landmark_confidences;
            prediction.box = hand.box;
            prediction.handedness_confidence = hand.handedness_confidence;
            result.hands.push_back(prediction);
        }
    }
    result.ok_count = static_cast<int>(std::count_if(result.hands.begin(), result.hands.end(), [](const auto& hand) {
        return hand.is_ok;
    }));
    result.double_ok = result.ok_count >= 2;

    history_.push_back(result.double_ok);
    while (history_.size() > history_limit_) {
        history_.pop_front();
    }
    const int positives = static_cast<int>(std::count(history_.begin(), history_.end(), true));
    result.stable_double_ok = positives >= static_cast<int>(stable_min_positive_);
    return result;
}

void DoubleOKRecognizer::reset() {
    history_.clear();
}

double bounded_score(double score) {
    if (!std::isfinite(score)) {
        throw std::runtime_error("Classifier produced a non-finite score");
    }
    return std::clamp(score, 0.0, 1.0);
}

}  // namespace double_ok_gesture
