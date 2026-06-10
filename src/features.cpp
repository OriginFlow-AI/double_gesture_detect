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

const std::array<std::string, 5> kFingerOrder = {"thumb", "index", "middle", "ring", "pinky"};

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
    const Landmarks points = normalize_landmarks(landmarks, handedness);
    const double pinch = distance(points, THUMB_TIP, INDEX_TIP);
    const double pinch_score = std::exp(-std::pow(pinch / 0.28, 2.0));

    const double middle_ext = finger_extension(points, "middle");
    const double ring_ext = finger_extension(points, "ring");
    const double pinky_ext = finger_extension(points, "pinky");
    const double open_mean = (middle_ext + ring_ext + pinky_ext) / 3.0;

    const double index_thumb_mcp_gap = distance(points, THUMB_MCP, INDEX_MCP);
    const double circle_gap_score = std::exp(-std::pow(pinch / std::max(index_thumb_mcp_gap, 0.15), 2.0));
    const double ok_score = std::clamp(0.58 * pinch_score + 0.30 * open_mean + 0.12 * circle_gap_score, 0.0, 1.0);

    return {pinch, middle_ext, ring_ext, pinky_ext, open_mean, ok_score};
}

double rule_ok_score(const Landmarks& landmarks, const std::string& handedness) {
    return geometry_scores(landmarks, handedness).ok_score;
}

std::vector<double> feature_vector(const Landmarks& landmarks, const std::string& handedness) {
    const Landmarks points = normalize_landmarks(landmarks, handedness);
    std::vector<double> features;
    features.reserve(96);

    for (const Point3& point : points) {
        features.push_back(point.x);
        features.push_back(point.y);
        features.push_back(point.z);
    }

    const std::array<double, 7> distances = {
        distance(points, THUMB_TIP, INDEX_TIP),
        distance(points, THUMB_TIP, MIDDLE_TIP),
        distance(points, INDEX_TIP, MIDDLE_TIP),
        distance(points, INDEX_TIP, RING_TIP),
        distance(points, INDEX_TIP, PINKY_TIP),
        distance(points, THUMB_MCP, INDEX_MCP),
        distance(points, INDEX_MCP, PINKY_MCP),
    };
    features.insert(features.end(), distances.begin(), distances.end());

    const std::array<double, 5> tip_distances = {
        distance(points, THUMB_TIP, WRIST),
        distance(points, INDEX_TIP, WRIST),
        distance(points, MIDDLE_TIP, WRIST),
        distance(points, RING_TIP, WRIST),
        distance(points, PINKY_TIP, WRIST),
    };
    features.insert(features.end(), tip_distances.begin(), tip_distances.end());

    for (const std::string& finger : kFingerOrder) {
        features.push_back(finger_extension(points, finger));
    }
    for (const std::string& finger : kFingerOrder) {
        const auto chain = kFingerChains.at(finger);
        features.push_back(angle(points, chain[0], chain[1], chain[2]));
        features.push_back(angle(points, chain[1], chain[2], chain[3]));
    }

    const GeometryScores geom = geometry_scores(points);
    features.push_back(geom.pinch);
    features.push_back(geom.middle_extension);
    features.push_back(geom.ring_extension);
    features.push_back(geom.pinky_extension);
    features.push_back(geom.open_finger_mean);
    features.push_back(geom.ok_score);
    return features;
}

std::vector<std::string> feature_names() {
    std::vector<std::string> names;
    names.reserve(96);
    for (int i = 0; i < 21; ++i) {
        names.push_back("lm_" + std::to_string(i) + "_x");
        names.push_back("lm_" + std::to_string(i) + "_y");
        names.push_back("lm_" + std::to_string(i) + "_z");
    }
    const int extra_count = static_cast<int>(feature_vector(Landmarks{}).size()) - static_cast<int>(names.size());
    for (int i = 0; i < extra_count; ++i) {
        names.push_back("geom_" + std::to_string(i));
    }
    return names;
}

}  // namespace double_ok_gesture
