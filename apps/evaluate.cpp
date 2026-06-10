#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/model_io.hpp"
#include "double_ok_gesture/training.hpp"

namespace {

struct Args {
    std::filesystem::path input = "data/processed/hagrid_ok_features.csv";
    std::filesystem::path model = "models/ok_hand_numpy_logreg.txt";
    std::string split = "auto";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + key);
            }
            return argv[++i];
        };
        if (key == "--input") {
            args.input = next();
        } else if (key == "--model") {
            args.model = next();
        } else if (key == "--split") {
            args.split = next();
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    return args;
}

std::string choose_split(const std::vector<std::string>& splits, const std::string& requested) {
    if (requested != "auto") {
        return requested;
    }
    const auto has = [&](const std::string& name) {
        return std::find(splits.begin(), splits.end(), name) != splits.end();
    };
    if (has("test")) {
        return "test";
    }
    if (has("val")) {
        return "val";
    }
    return "all";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto artifact = double_ok_gesture::load_model_artifact(args.model);
        const auto dataset = double_ok_gesture::load_feature_csv(args.input);
        if (!artifact.feature_columns.empty() && artifact.feature_columns != dataset.feature_columns) {
            throw std::runtime_error("Model feature schema does not match the evaluation CSV");
        }

        const std::string split = choose_split(dataset.splits, args.split);
        std::vector<std::vector<double>> x;
        std::vector<int> y;
        for (std::size_t i = 0; i < dataset.x.size(); ++i) {
            if (split == "all" || dataset.splits[i] == split) {
                x.push_back(dataset.x[i]);
                y.push_back(dataset.y[i]);
            }
        }
        if (x.empty()) {
            throw std::runtime_error("Requested split is not present in the CSV: " + split);
        }
        std::cout << "Evaluating split: " << split << " (" << y.size() << " rows)\n";
        const auto pred = double_ok_gesture::predict_labels(artifact, x);
        std::cout << double_ok_gesture::format_metrics(double_ok_gesture::confusion_matrix_2x2(y, pred));
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
