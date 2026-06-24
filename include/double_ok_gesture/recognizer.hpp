#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/model_io.hpp"

namespace double_ok_gesture {

struct HandPrediction {
    std::string handedness = "Unknown";
    double ok_score = 0.0;
    bool is_ok = false;
    Landmarks landmarks{};
    bool landmarks_estimated = false;
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
        std::size_t stable_min_positive = 3);

    DoubleOKResult process_hands(const std::vector<DetectedHand>& hands);
    void reset();

private:
    OKHandClassifier classifier_;
    std::size_t stable_min_positive_ = 3;
    std::deque<bool> history_;
    std::size_t history_limit_ = 5;
};

double bounded_score(double score);

}  // namespace double_ok_gesture
