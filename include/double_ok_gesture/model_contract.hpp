#pragma once

#include <filesystem>
#include <string>

namespace double_ok_gesture {

struct YoloV8PoseManifestInfo {
    std::string model_kind;
    std::string target_platform;
    std::string dtype;
    std::string sha256;
    int input_size = 0;
    bool board_validation_completed = false;
};

std::string sha256_file(const std::filesystem::path& path);

YoloV8PoseManifestInfo validate_yolov8_pose_model_contract(
    const std::filesystem::path& model_path,
    const std::filesystem::path& manifest_path,
    int expected_input_size);

}  // namespace double_ok_gesture
