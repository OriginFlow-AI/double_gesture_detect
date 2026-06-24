#include "double_ok_gesture/recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

double sigmoid(double value) {
    value = std::clamp(value, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(-value));
}

void validate_threshold(double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("threshold must be finite and in [0.0, 1.0]");
    }
}

void validate_artifact_schema(const LinearModelArtifact& artifact) {
    if (!artifact.feature_columns.empty() && artifact.feature_columns != feature_names()) {
        throw std::runtime_error("Model feature schema does not match the runtime feature schema");
    }
    const std::size_t feature_count = feature_names().size();
    if (artifact.coef.size() != feature_count || artifact.mean.size() != feature_count ||
        artifact.scale.size() != feature_count) {
        throw std::runtime_error("Model vector lengths do not match the runtime feature schema");
    }
}

}  // namespace

OKHandClassifier::OKHandClassifier(double threshold) : threshold_(threshold) {
    validate_threshold(threshold_);
}

OKHandClassifier::OKHandClassifier(const std::filesystem::path& model_path, double threshold)
    : OKHandClassifier(load_model_artifact(model_path), threshold) {}

OKHandClassifier::OKHandClassifier(LinearModelArtifact artifact, double threshold)
    : artifact_(std::move(artifact)), threshold_(threshold) {
    validate_threshold(threshold_);
    validate_artifact_schema(*artifact_);
}

double OKHandClassifier::score(const Landmarks& landmarks, const std::string& handedness) const {
    if (!artifact_) {
        return rule_ok_score(landmarks, handedness);
    }

    const std::vector<double> vector = feature_vector(landmarks, handedness);
    const LinearModelArtifact& artifact = *artifact_;
    double raw = artifact.intercept;
    for (std::size_t i = 0; i < vector.size(); ++i) {
        const double scale = std::abs(artifact.scale[i]) < 1e-12 ? 1.0 : artifact.scale[i];
        raw += ((vector[i] - artifact.mean[i]) / scale) * artifact.coef[i];
    }
    return bounded_score(sigmoid(raw));
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
    };
}

bool OKHandClassifier::uses_model() const {
    return artifact_.has_value();
}

double OKHandClassifier::threshold() const {
    return threshold_;
}

DoubleOKRecognizer::DoubleOKRecognizer(
    OKHandClassifier classifier,
    std::size_t stable_window,
    std::size_t stable_min_positive)
    : classifier_(std::move(classifier)), stable_min_positive_(stable_min_positive), history_limit_(stable_window) {
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
            result.hands.push_back(
                classifier_.predict_with_score(
                    hand.landmarks,
                    hand.handedness,
                    *hand.ok_score,
                    hand.landmarks_estimated));
        } else {
            HandPrediction prediction = classifier_.predict(hand.landmarks, hand.handedness);
            prediction.landmarks_estimated = hand.landmarks_estimated;
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
