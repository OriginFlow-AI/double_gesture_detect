#include "double_ok_gesture/landmark_provider.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/json.hpp"

namespace double_ok_gesture {
namespace {

Point3 parse_point(const Json& value) {
    if (value.is_array()) {
        const auto& coordinates = value.as_array();
        if (coordinates.size() < 2 || coordinates.size() > 3) {
            throw std::runtime_error(
                "landmark point must contain [x,y] or [x,y,z]");
        }
        return {
            coordinates[0].as_number(),
            coordinates[1].as_number(),
            coordinates.size() == 3 ? coordinates[2].as_number() : 0.0,
        };
    }
    if (value.is_object()) {
        const Json* x = value.get("x");
        const Json* y = value.get("y");
        const Json* z = value.get("z");
        if (!x || !y) {
            throw std::runtime_error(
                "landmark point object must contain x and y");
        }
        return {
            x->as_number(), y->as_number(), z ? z->as_number() : 0.0};
    }
    throw std::runtime_error("landmark point must be an array or object");
}

Landmarks parse_landmarks(const Json& value) {
    const auto& points = value.as_array();
    if (points.size() != 21) {
        throw std::runtime_error(
            "landmarks-json hand must contain exactly 21 points");
    }
    Landmarks landmarks{};
    for (std::size_t index = 0; index < landmarks.size(); ++index) {
        landmarks[index] = parse_point(points[index]);
    }
    validate_landmarks(landmarks);
    return landmarks;
}

DetectedHand parse_hand(const Json& value) {
    if (value.is_array()) {
        return {parse_landmarks(value), "Unknown", std::nullopt, false};
    }
    if (!value.is_object()) {
        throw std::runtime_error(
            "landmarks-json hand must be an object or 21-point array");
    }
    const Json* landmarks = value.get("landmarks");
    if (!landmarks) {
        landmarks = value.get("hand_landmarks");
    }
    if (!landmarks) {
        throw std::runtime_error(
            "landmarks-json hand object must contain landmarks");
    }
    const Json* handedness = value.get("handedness");
    const Json* score = value.get("ok_score");
    if (!score) {
        score = value.get("score");
    }
    return {
        parse_landmarks(*landmarks),
        handedness ? handedness->as_string() : "Unknown",
        score ? std::optional<double>(score->as_number()) : std::nullopt,
        false,
    };
}

std::vector<DetectedHand> parse_hands_json(const Json& root) {
    const Json* hands_value = &root;
    if (root.is_object()) {
        hands_value = root.get("hands");
        if (!hands_value) {
            throw std::runtime_error(
                "landmarks-json root object must contain hands");
        }
    }
    const auto& hands = hands_value->as_array();
    std::vector<DetectedHand> detected;
    detected.reserve(hands.size());
    for (const Json& hand : hands) {
        detected.push_back(parse_hand(hand));
    }
    return detected;
}

}  // namespace

NullHandLandmarkProvider::NullHandLandmarkProvider(
    std::string name,
    bool available)
    : info_{std::move(name), available, false} {}

LandmarkProviderInfo NullHandLandmarkProvider::info() const {
    return info_;
}

std::vector<DetectedHand> NullHandLandmarkProvider::detect(const cv::Mat&) {
    return {};
}

JsonHandLandmarkProvider::JsonHandLandmarkProvider(
    std::filesystem::path path)
    : path_(std::move(path)) {}

LandmarkProviderInfo JsonHandLandmarkProvider::info() const {
    return {"landmarks-json", true, false};
}

std::vector<DetectedHand> JsonHandLandmarkProvider::detect(const cv::Mat&) {
    return parse_hands_json(load_json(path_));
}

}  // namespace double_ok_gesture
