#include "double_ok_gesture/model_io.hpp"

#include <fstream>
#include <iomanip>
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
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream stream(line);
        std::string key;
        stream >> key;
        if (key == "model_type") {
            stream >> artifact.model_type;
        } else if (key == "feature_count") {
            stream >> feature_count;
        } else if (key == "intercept") {
            stream >> artifact.intercept;
        } else if (key == "mean") {
            artifact.mean = read_doubles(stream, feature_count, key);
        } else if (key == "scale") {
            artifact.scale = read_doubles(stream, feature_count, key);
        } else if (key == "coef") {
            artifact.coef = read_doubles(stream, feature_count, key);
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
        }
    }

    if (feature_count == 0 || artifact.coef.size() != feature_count) {
        throw std::runtime_error("Model file is missing coefficients: " + path.string());
    }
    if (artifact.mean.size() != feature_count || artifact.scale.size() != feature_count) {
        throw std::runtime_error("Model file is missing normalization statistics: " + path.string());
    }
    if (!artifact.feature_columns.empty() && artifact.feature_columns.size() != feature_count) {
        throw std::runtime_error("Model feature schema is incomplete: " + path.string());
    }
    return artifact;
}

void save_model_artifact(const std::filesystem::path& path, const LinearModelArtifact& artifact) {
    if (artifact.coef.empty()) {
        throw std::invalid_argument("Cannot save an empty model artifact");
    }
    if (artifact.mean.size() != artifact.coef.size() || artifact.scale.size() != artifact.coef.size()) {
        throw std::invalid_argument("Model normalization statistics must match coefficient count");
    }
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.parent_path() / ("." + path.filename().string() + ".tmp");

    {
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
        out << "feature_columns";
        for (const std::string& name : artifact.feature_columns) {
            out << ' ' << name;
        }
        out << '\n';
    }

    std::filesystem::rename(temporary, path);
}

}  // namespace double_ok_gesture
