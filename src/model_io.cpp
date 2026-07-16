#include "double_ok_gesture/model_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace double_ok_gesture {
namespace {

std::vector<double> read_doubles(std::istringstream& stream, std::size_t count, const std::string& field) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!(stream >> value)) {
            throw std::runtime_error("Invalid or incomplete model field: " + field);
        }
        values.push_back(value);
    }
    return values;
}

void write_doubles(std::ofstream& out, const std::string& key, const std::vector<double>& values) {
    out << key;
    out << std::setprecision(17);
    for (double value : values) {
        out << ' ' << value;
    }
    out << '\n';
}

void require_line_end(std::istringstream& stream, const std::string& field) {
    std::string extra;
    if (stream >> extra) {
        throw std::runtime_error(
            "Unexpected trailing data in model field: " + field);
    }
}

void validate_artifact(const LinearModelArtifact& artifact) {
    if (artifact.coef.empty()) {
        throw std::invalid_argument("Model coefficients must not be empty");
    }
    if (artifact.mean.size() != artifact.coef.size() ||
        artifact.scale.size() != artifact.coef.size()) {
        throw std::invalid_argument(
            "Model normalization statistics must match coefficient count");
    }
    if (!artifact.feature_columns.empty() &&
        artifact.feature_columns.size() != artifact.coef.size()) {
        throw std::invalid_argument(
            "Model feature columns must match coefficient count");
    }
    if (artifact.model_type.empty() ||
        artifact.model_type.find_first_of(" \t\r\n") != std::string::npos) {
        throw std::invalid_argument(
            "Model type must be a non-empty token");
    }
    if (!std::isfinite(artifact.intercept)) {
        throw std::invalid_argument("Model intercept must be finite");
    }
    const auto finite = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
    };
    if (!finite(artifact.mean) || !finite(artifact.scale) ||
        !finite(artifact.coef)) {
        throw std::invalid_argument("Model vectors must contain finite values");
    }
    for (const std::string& name : artifact.feature_columns) {
        if (name.empty() ||
            name.find_first_of(" \t\r\n") != std::string::npos) {
            throw std::invalid_argument(
                "Model feature names must be non-empty tokens");
        }
    }
}

}  // namespace

bool has_model(const LinearModelArtifact& artifact) {
    return !artifact.coef.empty();
}

LinearModelArtifact load_model_artifact(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open model file: " + path.string());
    }

    std::string header;
    std::getline(in, header);
    if (header != "double_ok_model_v1") {
        throw std::runtime_error("Unsupported model format: " + path.string());
    }

    LinearModelArtifact artifact;
    std::size_t feature_count = 0;
    std::set<std::string> fields;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream stream(line);
        std::string key;
        stream >> key;
        if (!fields.insert(key).second) {
            throw std::runtime_error("Duplicate model field: " + key);
        }
        if (key == "model_type") {
            if (!(stream >> artifact.model_type)) {
                throw std::runtime_error("Invalid model field: " + key);
            }
            require_line_end(stream, key);
        } else if (key == "feature_count") {
            if (!(stream >> feature_count) || feature_count == 0) {
                throw std::runtime_error("Invalid model field: " + key);
            }
            require_line_end(stream, key);
        } else if (key == "intercept") {
            if (!(stream >> artifact.intercept)) {
                throw std::runtime_error("Invalid model field: " + key);
            }
            require_line_end(stream, key);
        } else if (key == "mean") {
            artifact.mean = read_doubles(stream, feature_count, key);
            require_line_end(stream, key);
        } else if (key == "scale") {
            artifact.scale = read_doubles(stream, feature_count, key);
            require_line_end(stream, key);
        } else if (key == "coef") {
            artifact.coef = read_doubles(stream, feature_count, key);
            require_line_end(stream, key);
        } else if (key == "feature_columns") {
            artifact.feature_columns.clear();
            artifact.feature_columns.reserve(feature_count);
            for (std::size_t i = 0; i < feature_count; ++i) {
                std::string name;
                if (!(stream >> name)) {
                    throw std::runtime_error("Invalid or incomplete model feature columns");
                }
                artifact.feature_columns.push_back(name);
            }
            require_line_end(stream, key);
        } else {
            throw std::runtime_error("Unknown model field: " + key);
        }
    }

    if (!in.eof()) {
        throw std::runtime_error("Failed while reading model file: " + path.string());
    }

    if (!fields.contains("model_type") || !fields.contains("feature_count") ||
        !fields.contains("intercept") || !fields.contains("mean") ||
        !fields.contains("scale") || !fields.contains("coef") ||
        artifact.coef.size() != feature_count) {
        throw std::runtime_error("Model file is missing coefficients: " + path.string());
    }
    if (artifact.mean.size() != feature_count || artifact.scale.size() != feature_count) {
        throw std::runtime_error("Model file is missing normalization statistics: " + path.string());
    }
    if (!artifact.feature_columns.empty() && artifact.feature_columns.size() != feature_count) {
        throw std::runtime_error("Model feature schema is incomplete: " + path.string());
    }
    try {
        validate_artifact(artifact);
    } catch (const std::invalid_argument& error) {
        throw std::runtime_error(
            "Invalid model file '" + path.string() + "': " + error.what());
    }
    return artifact;
}

void save_model_artifact(const std::filesystem::path& path, const LinearModelArtifact& artifact) {
    validate_artifact(artifact);
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument("Model output path must name a file");
    }
    const std::filesystem::path directory = path.parent_path();
    if (!directory.empty()) {
        std::filesystem::create_directories(directory);
    }
    const std::filesystem::path temporary =
        directory / ("." + path.filename().string() + ".tmp");

    try {
        std::ofstream out(temporary);
        if (!out) {
            throw std::runtime_error("Cannot write temporary model file: " + temporary.string());
        }
        out << "double_ok_model_v1\n";
        out << "model_type " << artifact.model_type << '\n';
        out << "feature_count " << artifact.coef.size() << '\n';
        out << std::setprecision(17) << "intercept " << artifact.intercept << '\n';
        write_doubles(out, "mean", artifact.mean);
        write_doubles(out, "scale", artifact.scale);
        write_doubles(out, "coef", artifact.coef);
        if (!artifact.feature_columns.empty()) {
            out << "feature_columns";
            for (const std::string& name : artifact.feature_columns) {
                out << ' ' << name;
            }
            out << '\n';
        }
        out.close();
        if (!out) {
            throw std::runtime_error(
                "Cannot finish temporary model file: " + temporary.string());
        }

        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

}  // namespace double_ok_gesture
