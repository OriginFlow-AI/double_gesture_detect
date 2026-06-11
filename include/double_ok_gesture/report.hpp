#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

#include "double_ok_gesture/config.hpp"

namespace double_ok_gesture {

struct FileSummary {
    std::filesystem::path path;
    bool exists = false;
    std::uintmax_t size_bytes = 0;
    std::string size_label = "missing";
};

struct CsvSummary {
    FileSummary file;
    std::size_t row_count = 0;
    std::size_t positive_count = 0;
    std::size_t negative_count = 0;
    std::size_t feature_count = 0;
    std::size_t column_count = 0;
    std::map<std::string, std::size_t> split_counts;
    std::map<std::string, std::size_t> gesture_counts;
};

struct GuiReportRequest {
    std::filesystem::path config = "configs/default.json";
    std::filesystem::path csv = "data/processed/hagrid_ok_features.csv";
    std::filesystem::path model = "models/ok_hand_numpy_logreg.pkl";
    std::filesystem::path output = "reports/gui/index.html";
};

FileSummary summarize_file(const std::filesystem::path& path);
CsvSummary scan_feature_csv(const std::filesystem::path& path);
std::string render_gui_report(
    const std::filesystem::path& config_path,
    const RuntimeConfig& config,
    const CsvSummary& csv,
    const FileSummary& model);
void write_gui_report(const GuiReportRequest& request);

}  // namespace double_ok_gesture
