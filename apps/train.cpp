#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/model_io.hpp"
#include "double_ok_gesture/training.hpp"

namespace {

struct Args {
    std::filesystem::path input = "data/processed/hagrid_ok_features.csv";
    std::filesystem::path output = "models/ok_hand_numpy_logreg.txt";
    int max_iter = 160;
    unsigned int random_state = 42;
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
        } else if (key == "--output") {
            args.output = next();
        } else if (key == "--max-iter") {
            args.max_iter = std::stoi(next());
        } else if (key == "--random-state") {
            args.random_state = static_cast<unsigned int>(std::stoul(next()));
        } else if (key == "--model") {
            const std::string model = next();
            if (model != "numpy_logreg" && model != "logreg") {
                throw std::invalid_argument("C++ build supports --model numpy_logreg/logreg");
            }
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const auto dataset = double_ok_gesture::load_feature_csv(args.input);
        const auto split = double_ok_gesture::split_data(dataset, args.random_state);
        auto model = double_ok_gesture::train_logistic_regression(
            split.x_train,
            split.y_train,
            dataset.feature_columns,
            args.max_iter);
        const auto pred = double_ok_gesture::predict_labels(model, split.x_test);
        std::cout << double_ok_gesture::format_metrics(double_ok_gesture::confusion_matrix_2x2(split.y_test, pred));
        double_ok_gesture::save_model_artifact(args.output, model);
        std::cout << "Saved model to " << args.output << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
