#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "double_ok_gesture/model_io.hpp"

namespace double_ok_gesture {

struct FeatureDataset {
    std::vector<std::vector<double>> x;
    std::vector<int> y;
    std::vector<std::string> splits;
    std::vector<std::string> feature_columns;
};

struct SplitDataset {
    std::vector<std::vector<double>> x_train;
    std::vector<std::vector<double>> x_test;
    std::vector<int> y_train;
    std::vector<int> y_test;
};

struct BinaryMetrics {
    int matrix[2][2] = {{0, 0}, {0, 0}};
};

FeatureDataset load_feature_csv(const std::filesystem::path& path);
SplitDataset split_data(const FeatureDataset& dataset, unsigned int random_state);
LinearModelArtifact train_logistic_regression(
    const std::vector<std::vector<double>>& x,
    const std::vector<int>& y,
    const std::vector<std::string>& feature_columns,
    int max_iter = 160,
    double learning_rate = 0.25,
    double l2 = 1e-4);
std::vector<int> predict_labels(const LinearModelArtifact& artifact, const std::vector<std::vector<double>>& x);
std::vector<double> predict_scores(const LinearModelArtifact& artifact, const std::vector<std::vector<double>>& x);
BinaryMetrics confusion_matrix_2x2(const std::vector<int>& y_true, const std::vector<int>& y_pred);
std::string format_metrics(const BinaryMetrics& metrics);

}  // namespace double_ok_gesture
