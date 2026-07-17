#include "double_ok_gesture/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>

namespace double_ok_gesture {
namespace {

using FingerChain = std::array<int, 4>;

const std::unordered_map<std::string, FingerChain> kFingerChains = {
    {"thumb", {THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP}},
    {"index", {INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP}},
    {"middle", {MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP}},
    {"ring", {RING_MCP, RING_PIP, RING_DIP, RING_TIP}},
    {"pinky", {PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP}},
};

Point3 operator-(const Point3& lhs, const Point3& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Point3& operator-=(Point3& lhs, const Point3& rhs) {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

Point3& operator/=(Point3& lhs, double scale) {
    lhs.x /= scale;
    lhs.y /= scale;
    lhs.z /= scale;
    return lhs;
}

double norm3(const Point3& point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

double norm2(const Point3& point) {
    return std::sqrt(point.x * point.x + point.y * point.y);
}

double distance(const Landmarks& points, int a, int b) {
    return norm3(points[static_cast<std::size_t>(a)] - points[static_cast<std::size_t>(b)]);
}

double sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-value));
}

double smoothstep(double lower, double upper, double value) {
    const double normalized = std::clamp((value - lower) / (upper - lower), 0.0, 1.0);
    return normalized * normalized * (3.0 - 2.0 * normalized);
}

double angle(const Landmarks& points, int a, int b, int c) {
    const Point3 ab = points[static_cast<std::size_t>(a)] - points[static_cast<std::size_t>(b)];
    const Point3 cb = points[static_cast<std::size_t>(c)] - points[static_cast<std::size_t>(b)];
    const double denom = norm3(ab) * norm3(cb);
    if (denom < 1e-6) {
        return 0.0;
    }
    const double dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
    const double cosine = std::clamp(dot / denom, -1.0, 1.0);
    return std::acos(cosine) / M_PI;
}

bool starts_with_right(std::string handedness) {
    std::transform(handedness.begin(), handedness.end(), handedness.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return handedness.rfind("right", 0) == 0;
}

}  // namespace

void validate_landmarks(const Landmarks& landmarks) {
    for (const Point3& point : landmarks) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            throw std::invalid_argument("Landmarks must contain only finite values");
        }
    }
}

Landmarks normalize_landmarks(const Landmarks& landmarks, const std::string& handedness) {
    validate_landmarks(landmarks);
    Landmarks points = landmarks;
    const Point3 wrist = points[WRIST];
    for (Point3& point : points) {
        point -= wrist;
    }

    double scale = norm3(points[MIDDLE_MCP]);
    scale = std::max(scale, norm3(points[INDEX_MCP] - points[PINKY_MCP]));
    double max_xy = 0.0;
    for (const Point3& point : points) {
        max_xy = std::max(max_xy, norm2(point));
    }
    scale = std::max(scale, max_xy);
    if (scale < 1e-6 || !std::isfinite(scale)) {
        scale = 1.0;
    }

    for (Point3& point : points) {
        point /= scale;
        if (starts_with_right(handedness)) {
            point.x *= -1.0;
        }
    }
    return points;
}

double finger_extension(const Landmarks& normalized_landmarks, const std::string& finger) {
    const auto chain = kFingerChains.find(finger);
    if (chain == kFingerChains.end()) {
        throw std::invalid_argument("Unsupported finger name: " + finger);
    }
    const auto [mcp, pip, dip, tip] = chain->second;
    const double tip_gain = distance(normalized_landmarks, tip, WRIST) - distance(normalized_landmarks, pip, WRIST);
    const double straightness =
        (angle(normalized_landmarks, mcp, pip, dip) + angle(normalized_landmarks, pip, dip, tip)) * 0.5;
    return std::clamp(0.65 * sigmoid(8.0 * tip_gain) + 0.35 * straightness, 0.0, 1.0);
}

GeometryScores geometry_scores(const Landmarks& landmarks, const std::string& handedness) {
    validate_landmarks(landmarks);
    const double palm_extent = std::max(
        distance(landmarks, WRIST, MIDDLE_MCP),
        distance(landmarks, INDEX_MCP, PINKY_MCP));
    if (palm_extent < 1e-6) {
        // A collapsed pose contains no usable hand geometry.  In particular,
        // coincident tips must not look like a perfect pinch.
        return {};
    }

    const Landmarks points = normalize_landmarks(landmarks, handedness);
    const double pinch = distance(points, THUMB_TIP, INDEX_TIP);
    // Pose estimators are least accurate where the thumb and index finger
    // occlude one another.  Use a wider, continuous tolerance than the old
    // endpoint Gaussian, while retaining the historical hand-length scale.
    const double pinch_quality = std::exp(-std::pow(pinch / 0.75, 2.0));

    const double middle_ext = finger_extension(points, "middle");
    const double ring_ext = finger_extension(points, "ring");
    const double pinky_ext = finger_extension(points, "pinky");
    const double open_mean = (middle_ext + ring_ext + pinky_ext) / 3.0;
    const double index_ext = finger_extension(points, "index");

    // OK is a conjunction: the pinch alone is insufficient when the other
    // three fingers are folded.  This removes the old formula's >= 0.70
    // lower bound for any pose whose two tips happened to overlap.
    const double open_quality = smoothstep(0.45, 0.80, open_mean);
    // A true OK ring also bends the index finger. This independent negative
    // constraint keeps an open palm with noisy/overlapping tip predictions
    // well away from the decision threshold.
    const double index_bend_quality =
        1.0 - smoothstep(0.70, 0.82, index_ext);
    const double shape_quality =
        open_quality * (0.50 + 0.50 * index_bend_quality);
    const double ok_score = std::clamp(
        pinch_quality * (0.25 + 0.75 * shape_quality),
        0.0,
        1.0);

    return {pinch, middle_ext, ring_ext, pinky_ext, open_mean, ok_score};
}

double rule_ok_score(const Landmarks& landmarks, const std::string& handedness) {
    return geometry_scores(landmarks, handedness).ok_score;
}

}  // namespace double_ok_gesture
