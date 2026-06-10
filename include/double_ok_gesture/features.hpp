#pragma once

#include <array>
#include <string>
#include <vector>

namespace double_ok_gesture {

struct Point3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using Landmarks = std::array<Point3, 21>;

enum LandmarkIndex {
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_MCP = 5,
    INDEX_PIP = 6,
    INDEX_DIP = 7,
    INDEX_TIP = 8,
    MIDDLE_MCP = 9,
    MIDDLE_PIP = 10,
    MIDDLE_DIP = 11,
    MIDDLE_TIP = 12,
    RING_MCP = 13,
    RING_PIP = 14,
    RING_DIP = 15,
    RING_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20,
};

struct GeometryScores {
    double pinch = 0.0;
    double middle_extension = 0.0;
    double ring_extension = 0.0;
    double pinky_extension = 0.0;
    double open_finger_mean = 0.0;
    double ok_score = 0.0;
};

void validate_landmarks(const Landmarks& landmarks);
Landmarks normalize_landmarks(const Landmarks& landmarks, const std::string& handedness = "");
double finger_extension(const Landmarks& normalized_landmarks, const std::string& finger);
GeometryScores geometry_scores(const Landmarks& landmarks, const std::string& handedness = "");
double rule_ok_score(const Landmarks& landmarks, const std::string& handedness = "");
std::vector<double> feature_vector(const Landmarks& landmarks, const std::string& handedness = "");
std::vector<std::string> feature_names();

}  // namespace double_ok_gesture
