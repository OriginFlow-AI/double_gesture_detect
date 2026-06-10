#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace double_ok_gesture {

struct LinearModelArtifact {
    std::string model_type = "numpy_logreg";
    std::vector<std::string> feature_columns;
    std::vector<double> mean;
    std::vector<double> scale;
    std::vector<double> coef;
    double intercept = 0.0;
};

bool has_model(const LinearModelArtifact& artifact);
LinearModelArtifact load_model_artifact(const std::filesystem::path& path);
void save_model_artifact(const std::filesystem::path& path, const LinearModelArtifact& artifact);

}  // namespace double_ok_gesture
