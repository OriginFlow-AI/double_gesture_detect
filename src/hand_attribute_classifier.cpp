#include "double_ok_gesture/hand_attribute_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "double_ok_gesture/json.hpp"

namespace double_ok_gesture {
namespace {

const Json& required_member(const Json& object, const char* name) {
    const Json* value = object.get(name);
    if (!value) {
        throw std::runtime_error(
            std::string("hand attribute model is missing '") + name + "'");
    }
    return *value;
}

std::vector<double> number_vector(
    const Json& object,
    const char* name,
    std::size_t expected_size) {
    const auto& values = required_member(object, name).as_array();
    if (values.size() != expected_size) {
        throw std::runtime_error(
            std::string("hand attribute model '") + name +
            "' must contain exactly " + std::to_string(expected_size) +
            " values");
    }
    std::vector<double> result;
    result.reserve(values.size());
    for (const Json& value : values) {
        const double number = value.as_number();
        if (!std::isfinite(number)) {
            throw std::runtime_error(
                std::string("hand attribute model '") + name +
                "' contains a non-finite value");
        }
        result.push_back(number);
    }
    return result;
}

double finite_number(const Json& object, const char* name) {
    const double value = required_member(object, name).as_number();
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string("hand attribute model '") + name +
            "' must be finite");
    }
    return value;
}

void validate_probability(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and in [0,1]");
    }
}

void validate_artifact(const HandAttributeModelArtifact& artifact) {
    const auto expected = kHandAttributeFeatureCount;
    if (artifact.mean.size() != expected || artifact.scale.size() != expected ||
        artifact.handedness_coef.size() != expected ||
        artifact.ok_coef.size() != expected) {
        throw std::runtime_error(
            "hand attribute model vectors must match the 63-value "
            "x,y,visibility contract");
    }
    for (const double value : artifact.scale) {
        if (!std::isfinite(value) || std::abs(value) < 1e-12) {
            throw std::runtime_error(
                "hand attribute model scale values must be finite and non-zero");
        }
    }
}

double sigmoid(double value) {
    value = std::clamp(value, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(-value));
}

double linear_probability(
    const HandAttributeFeatures& features,
    const HandAttributeModelArtifact& artifact,
    const std::vector<double>& coefficients,
    double intercept) {
    double raw = intercept;
    for (std::size_t index = 0; index < features.size(); ++index) {
        raw += ((features[index] - artifact.mean[index]) /
                artifact.scale[index]) * coefficients[index];
    }
    return sigmoid(raw);
}

}  // namespace

HandAttributeFeatures hand_attribute_features(
    const Landmarks& landmarks,
    const LandmarkConfidences& visibility,
    bool input_mirrored) {
    validate_landmarks(landmarks);
    for (const double confidence : visibility) {
        validate_probability(confidence, "keypoint visibility");
    }

    const Point3 wrist = landmarks[WRIST];
    double scale = 0.0;
    for (const Point3& point : landmarks) {
        const double dx = point.x - wrist.x;
        const double dy = point.y - wrist.y;
        scale = std::max(scale, std::sqrt(dx * dx + dy * dy));
    }
    if (!std::isfinite(scale) || scale < 1e-6) {
        throw std::invalid_argument(
            "hand attribute features require non-degenerate 2D landmarks");
    }

    HandAttributeFeatures features{};
    for (std::size_t index = 0; index < landmarks.size(); ++index) {
        double normalized_x = (landmarks[index].x - wrist.x) / scale;
        if (input_mirrored) {
            normalized_x *= -1.0;
        }
        features[index * 3] = normalized_x;
        features[index * 3 + 1] =
            (landmarks[index].y - wrist.y) / scale;
        features[index * 3 + 2] = visibility[index];
    }
    return features;
}

HandAttributeModelArtifact load_hand_attribute_model(
    const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "hand attribute model not found: " + path.string());
    }
    const Json root = load_json(path);
    (void)root.as_object();
    if (required_member(root, "schema").as_string() !=
        "double_ok_hand_attribute_v1") {
        throw std::runtime_error(
            "unsupported hand attribute model schema: " + path.string());
    }
    const double feature_count =
        required_member(root, "feature_count").as_number();
    if (!std::isfinite(feature_count) ||
        feature_count != static_cast<double>(kHandAttributeFeatureCount)) {
        throw std::runtime_error(
            "hand attribute model feature_count must be 63");
    }

    HandAttributeModelArtifact artifact;
    artifact.mean = number_vector(root, "mean", kHandAttributeFeatureCount);
    artifact.scale = number_vector(root, "scale", kHandAttributeFeatureCount);
    artifact.handedness_coef =
        number_vector(root, "handedness_coef", kHandAttributeFeatureCount);
    artifact.handedness_intercept =
        finite_number(root, "handedness_intercept");
    artifact.ok_coef =
        number_vector(root, "ok_coef", kHandAttributeFeatureCount);
    artifact.ok_intercept = finite_number(root, "ok_intercept");
    validate_artifact(artifact);
    return artifact;
}

HandAttributeClassifier::HandAttributeClassifier(
    const std::filesystem::path& model_path,
    double handedness_confidence_threshold,
    double ok_threshold,
    bool input_mirrored)
    : HandAttributeClassifier(
          load_hand_attribute_model(model_path),
          handedness_confidence_threshold,
          ok_threshold,
          input_mirrored) {}

HandAttributeClassifier::HandAttributeClassifier(
    HandAttributeModelArtifact artifact,
    double handedness_confidence_threshold,
    double ok_threshold,
    bool input_mirrored)
    : artifact_(std::move(artifact)),
      handedness_confidence_threshold_(handedness_confidence_threshold),
      ok_threshold_(ok_threshold),
      input_mirrored_(input_mirrored) {
    validate_artifact(artifact_);
    validate_probability(
        handedness_confidence_threshold_,
        "handedness confidence threshold");
    if (handedness_confidence_threshold_ < 0.5) {
        throw std::invalid_argument(
            "handedness confidence threshold must be at least 0.5");
    }
    validate_probability(ok_threshold_, "OK threshold");
}

HandAttributePrediction HandAttributeClassifier::predict(
    const Landmarks& landmarks,
    const LandmarkConfidences& visibility) const {
    const HandAttributeFeatures features = hand_attribute_features(
        landmarks, visibility, input_mirrored_);
    const double right_probability = linear_probability(
        features,
        artifact_,
        artifact_.handedness_coef,
        artifact_.handedness_intercept);
    const double confidence =
        std::max(right_probability, 1.0 - right_probability);
    std::string handedness = "Unknown";
    if (confidence >= handedness_confidence_threshold_) {
        handedness = right_probability >= 0.5 ? "Right" : "Left";
    }
    const double ok_score = linear_probability(
        features, artifact_, artifact_.ok_coef, artifact_.ok_intercept);
    return {
        handedness,
        confidence,
        ok_score,
        ok_score >= ok_threshold_,
    };
}

double HandAttributeClassifier::handedness_confidence_threshold() const {
    return handedness_confidence_threshold_;
}

double HandAttributeClassifier::ok_threshold() const {
    return ok_threshold_;
}

bool HandAttributeClassifier::input_mirrored() const {
    return input_mirrored_;
}

}  // namespace double_ok_gesture
