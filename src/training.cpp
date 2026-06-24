#include "double_ok_gesture/training.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace double_ok_gesture {
namespace {

std::vector<std::string> parse_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(field);
    return fields;
}

bool has_values(const std::vector<std::string>& row) {
    return std::any_of(row.begin(), row.end(), [](const std::string& value) {
        return !value.empty();
    });
}

std::vector<int> find_indices_for_label(const std::vector<int>& y, int label) {
    std::vector<int> indices;
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (y[i] == label) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

bool contains_both_labels(const std::vector<int>& y) {
    bool has_zero = false;
    bool has_one = false;
    for (int value : y) {
        has_zero = has_zero || value == 0;
        has_one = has_one || value == 1;
    }
    return has_zero && has_one;
}

std::vector<int> subset_labels(const std::vector<int>& y, const std::vector<int>& indices) {
    std::vector<int> result;
    result.reserve(indices.size());
    for (int index : indices) {
        result.push_back(y[static_cast<std::size_t>(index)]);
    }
    return result;
}

std::vector<std::vector<double>> subset_rows(const std::vector<std::vector<double>>& x, const std::vector<int>& indices) {
    std::vector<std::vector<double>> result;
    result.reserve(indices.size());
    for (int index : indices) {
        result.push_back(x[static_cast<std::size_t>(index)]);
    }
    return result;
}

std::pair<std::vector<int>, std::vector<int>> stratified_split_indices(
    const std::vector<int>& y,
    unsigned int random_state,
    double test_fraction = 0.2) {
    if (y.size() < 2) {
        throw std::invalid_argument("At least two target rows are required for a train/test split");
    }

    std::mt19937 generator(random_state);
    std::vector<int> train_indices;
    std::vector<int> test_indices;
    for (int label : {0, 1}) {
        std::vector<int> label_indices = find_indices_for_label(y, label);
        if (label_indices.size() < 2) {
            throw std::invalid_argument("Each label needs at least two rows for a stratified split");
        }
        std::shuffle(label_indices.begin(), label_indices.end(), generator);
        std::size_t test_count = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(label_indices.size() * test_fraction)));
        test_count = std::min(test_count, label_indices.size() - 1);
        test_indices.insert(test_indices.end(), label_indices.begin(), label_indices.begin() + static_cast<long>(test_count));
        train_indices.insert(train_indices.end(), label_indices.begin() + static_cast<long>(test_count), label_indices.end());
    }
    std::shuffle(train_indices.begin(), train_indices.end(), generator);
    std::shuffle(test_indices.begin(), test_indices.end(), generator);
    return {train_indices, test_indices};
}

double sigmoid(double value) {
    value = std::clamp(value, -40.0, 40.0);
    return 1.0 / (1.0 + std::exp(-value));
}

void validate_training_data(const std::vector<std::vector<double>>& x, const std::vector<int>& y) {
    if (x.size() < 2 || x.size() != y.size() || x.front().empty()) {
        throw std::invalid_argument("Training features must have shape (n >= 2, features >= 1)");
    }
    const std::size_t feature_count = x.front().size();
    for (const auto& row : x) {
        if (row.size() != feature_count) {
            throw std::invalid_argument("Training rows must have a stable feature count");
        }
        for (double value : row) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Training data must contain only finite values");
            }
        }
    }
    if (!contains_both_labels(y)) {
        throw std::invalid_argument("Training targets must contain both binary labels 0 and 1");
    }
}

}  // namespace

FeatureDataset load_feature_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open feature CSV: " + path.string());
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("No rows found in " + path.string());
    }
    const std::vector<std::string> header = parse_csv_row(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }
    if (!columns.contains("target")) {
        throw std::runtime_error("Missing required target column in " + path.string());
    }

    std::vector<std::size_t> feature_indices;
    std::vector<std::string> feature_columns;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i].rfind("lm_", 0) == 0 || header[i].rfind("geom_", 0) == 0) {
            feature_indices.push_back(i);
            feature_columns.push_back(header[i]);
        }
    }
    if (feature_indices.empty()) {
        throw std::runtime_error("No feature columns found in " + path.string());
    }

    FeatureDataset dataset;
    dataset.feature_columns = std::move(feature_columns);
    const std::size_t target_index = columns.at("target");
    const std::optional<std::size_t> split_index =
        columns.contains("split") ? std::optional<std::size_t>(columns.at("split")) : std::nullopt;

    int csv_line = 1;
    while (std::getline(in, line)) {
        ++csv_line;
        const std::vector<std::string> row = parse_csv_row(line);
        if (!has_values(row)) {
            continue;
        }
        try {
            std::vector<double> values;
            values.reserve(feature_indices.size());
            for (std::size_t index : feature_indices) {
                if (index >= row.size()) {
                    throw std::runtime_error("short row");
                }
                const double value = std::stod(row[index]);
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite");
                }
                values.push_back(value);
            }
            if (target_index >= row.size()) {
                throw std::runtime_error("missing target");
            }
            const int target = std::stoi(row[target_index]);
            if (target != 0 && target != 1) {
                throw std::runtime_error("invalid target");
            }
            dataset.x.push_back(std::move(values));
            dataset.y.push_back(target);
            dataset.splits.push_back(split_index && *split_index < row.size() && !row[*split_index].empty() ? row[*split_index] : "unknown");
        } catch (const std::exception& exc) {
            throw std::runtime_error("Invalid feature row at CSV line " + std::to_string(csv_line) + ": " + exc.what());
        }
    }

    if (dataset.x.empty()) {
        throw std::runtime_error("No rows found in " + path.string());
    }
    return dataset;
}

SplitDataset split_data(const FeatureDataset& dataset, unsigned int random_state) {
    std::vector<int> train_indices;
    std::vector<int> val_indices;
    for (std::size_t i = 0; i < dataset.splits.size(); ++i) {
        if (dataset.splits[i] == "train") {
            train_indices.push_back(static_cast<int>(i));
        } else if (dataset.splits[i] == "val") {
            val_indices.push_back(static_cast<int>(i));
        }
    }
    if (!train_indices.empty() && !val_indices.empty() && contains_both_labels(subset_labels(dataset.y, train_indices)) &&
        contains_both_labels(subset_labels(dataset.y, val_indices))) {
        return {
            subset_rows(dataset.x, train_indices),
            subset_rows(dataset.x, val_indices),
            subset_labels(dataset.y, train_indices),
            subset_labels(dataset.y, val_indices),
        };
    }

    if (!train_indices.empty()) {
        const std::vector<int> train_y = subset_labels(dataset.y, train_indices);
        const auto [inner_train, inner_test] = stratified_split_indices(train_y, random_state);
        std::vector<int> train_absolute;
        std::vector<int> test_absolute;
        for (int index : inner_train) {
            train_absolute.push_back(train_indices[static_cast<std::size_t>(index)]);
        }
        for (int index : inner_test) {
            test_absolute.push_back(train_indices[static_cast<std::size_t>(index)]);
        }
        return {
            subset_rows(dataset.x, train_absolute),
            subset_rows(dataset.x, test_absolute),
            subset_labels(dataset.y, train_absolute),
            subset_labels(dataset.y, test_absolute),
        };
    }

    const auto [train, test] = stratified_split_indices(dataset.y, random_state);
    return {subset_rows(dataset.x, train), subset_rows(dataset.x, test), subset_labels(dataset.y, train), subset_labels(dataset.y, test)};
}

LinearModelArtifact train_logistic_regression(
    const std::vector<std::vector<double>>& x,
    const std::vector<int>& y,
    const std::vector<std::string>& feature_columns,
    int max_iter,
    double learning_rate,
    double l2) {
    if (max_iter < 1) {
        throw std::invalid_argument("max_iter must be at least 1");
    }
    validate_training_data(x, y);
    const std::size_t rows = x.size();
    const std::size_t cols = x.front().size();

    LinearModelArtifact artifact;
    artifact.feature_columns = feature_columns;
    artifact.mean.assign(cols, 0.0);
    artifact.scale.assign(cols, 0.0);
    artifact.coef.assign(cols, 0.0);

    for (const auto& row : x) {
        for (std::size_t col = 0; col < cols; ++col) {
            artifact.mean[col] += row[col];
        }
    }
    for (double& value : artifact.mean) {
        value /= static_cast<double>(rows);
    }
    for (const auto& row : x) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double diff = row[col] - artifact.mean[col];
            artifact.scale[col] += diff * diff;
        }
    }
    for (double& value : artifact.scale) {
        value = std::sqrt(value / static_cast<double>(rows));
        if (value < 1e-6) {
            value = 1.0;
        }
    }

    const double positives = std::max(1.0, static_cast<double>(std::accumulate(y.begin(), y.end(), 0)));
    const double negatives = std::max(1.0, static_cast<double>(rows) - positives);
    std::vector<double> sample_weight(rows, 1.0);
    for (std::size_t row = 0; row < rows; ++row) {
        sample_weight[row] = y[row] == 1 ? static_cast<double>(rows) / (2.0 * positives) : static_cast<double>(rows) / (2.0 * negatives);
    }
    const double weight_sum = std::accumulate(sample_weight.begin(), sample_weight.end(), 0.0);
    const double prior = std::clamp(positives / static_cast<double>(rows), 1e-4, 1.0 - 1e-4);
    artifact.intercept = std::log(prior / (1.0 - prior));

    for (int step = 0; step < max_iter; ++step) {
        std::vector<double> grad(cols, 0.0);
        double intercept_grad = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            double raw = artifact.intercept;
            for (std::size_t col = 0; col < cols; ++col) {
                raw += ((x[row][col] - artifact.mean[col]) / artifact.scale[col]) * artifact.coef[col];
            }
            const double error = (sigmoid(raw) - static_cast<double>(y[row])) * sample_weight[row];
            intercept_grad += error;
            for (std::size_t col = 0; col < cols; ++col) {
                grad[col] += ((x[row][col] - artifact.mean[col]) / artifact.scale[col]) * error;
            }
        }

        const double rate = learning_rate / std::sqrt(1.0 + static_cast<double>(step) * 0.05);
        for (std::size_t col = 0; col < cols; ++col) {
            grad[col] = grad[col] / weight_sum + l2 * artifact.coef[col];
            artifact.coef[col] -= rate * grad[col];
        }
        artifact.intercept -= rate * (intercept_grad / weight_sum);
    }
    return artifact;
}

std::vector<double> predict_scores(const LinearModelArtifact& artifact, const std::vector<std::vector<double>>& x) {
    std::vector<double> scores;
    scores.reserve(x.size());
    for (const auto& row : x) {
        if (row.size() != artifact.coef.size()) {
            throw std::invalid_argument("Prediction data has the wrong feature count");
        }
        double raw = artifact.intercept;
        for (std::size_t col = 0; col < row.size(); ++col) {
            const double scale = std::abs(artifact.scale[col]) < 1e-12 ? 1.0 : artifact.scale[col];
            raw += ((row[col] - artifact.mean[col]) / scale) * artifact.coef[col];
        }
        scores.push_back(sigmoid(raw));
    }
    return scores;
}

std::vector<int> predict_labels(const LinearModelArtifact& artifact, const std::vector<std::vector<double>>& x) {
    std::vector<int> labels;
    const std::vector<double> scores = predict_scores(artifact, x);
    labels.reserve(scores.size());
    for (double score : scores) {
        labels.push_back(score >= 0.5 ? 1 : 0);
    }
    return labels;
}

BinaryMetrics confusion_matrix_2x2(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }
    BinaryMetrics metrics;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] < 0 || y_true[i] > 1 || y_pred[i] < 0 || y_pred[i] > 1) {
            throw std::invalid_argument("confusion_matrix_2x2 only supports binary labels 0 and 1");
        }
        metrics.matrix[y_true[i]][y_pred[i]] += 1;
    }
    return metrics;
}

std::string format_metrics(const BinaryMetrics& metrics) {
    std::ostringstream out;
    out << "Confusion matrix:\n";
    out << "[[" << metrics.matrix[0][0] << ", " << metrics.matrix[0][1] << "],\n";
    out << " [" << metrics.matrix[1][0] << ", " << metrics.matrix[1][1] << "]]\n\n";
    out << "label precision recall f1 support\n";
    for (int label : {0, 1}) {
        const char* name = label == 0 ? "not_ok" : "ok";
        const double tp = static_cast<double>(metrics.matrix[label][label]);
        const double fp = static_cast<double>(metrics.matrix[0][label] + metrics.matrix[1][label]) - tp;
        const double fn = static_cast<double>(metrics.matrix[label][0] + metrics.matrix[label][1]) - tp;
        const int support = metrics.matrix[label][0] + metrics.matrix[label][1];
        const double precision = tp / std::max(tp + fp, 1.0);
        const double recall = tp / std::max(tp + fn, 1.0);
        const double f1 = 2.0 * precision * recall / std::max(precision + recall, 1e-12);
        out << std::left << std::setw(6) << name << ' ' << std::fixed << std::setprecision(4) << precision << ' ' << recall
            << ' ' << f1 << ' ' << support << '\n';
    }
    return out.str();
}

}  // namespace double_ok_gesture
