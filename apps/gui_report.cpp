#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "double_ok_gesture/training.hpp"

int main(int argc, char** argv) {
    try {
        std::filesystem::path csv = "data/processed/hagrid_ok_features.csv";
        std::filesystem::path output = "reports/gui/index.html";
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("Missing value for " + key);
                }
                return argv[++i];
            };
            if (key == "--csv") {
                csv = next();
            } else if (key == "--output") {
                output = next();
            } else if (key == "--config" || key == "--model") {
                (void)next();
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }
        std::size_t rows = 0;
        std::size_t positives = 0;
        std::size_t negatives = 0;
        std::size_t features = 0;
        if (std::filesystem::exists(csv)) {
            const auto dataset = double_ok_gesture::load_feature_csv(csv);
            rows = dataset.y.size();
            features = dataset.feature_columns.size();
            for (int y : dataset.y) {
                positives += y == 1 ? 1 : 0;
                negatives += y == 0 ? 1 : 0;
            }
        }
        std::filesystem::create_directories(output.parent_path());
        std::ofstream out(output);
        out << "<!doctype html><meta charset=\"utf-8\"><title>Double OK C++ Report</title>";
        out << "<style>body{font-family:Arial,sans-serif;margin:32px;color:#1f2937}"
               "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}</style>";
        out << "<h1>Double OK C++ Report</h1><table>";
        out << "<tr><th>CSV</th><td>" << csv << "</td></tr>";
        out << "<tr><th>Rows</th><td>" << rows << "</td></tr>";
        out << "<tr><th>Positive</th><td>" << positives << "</td></tr>";
        out << "<tr><th>Negative</th><td>" << negatives << "</td></tr>";
        out << "<tr><th>Feature columns</th><td>" << features << "</td></tr>";
        out << "</table>";
        std::cout << output << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
