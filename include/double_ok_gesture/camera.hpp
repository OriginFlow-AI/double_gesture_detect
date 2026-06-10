#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "double_ok_gesture/config.hpp"

namespace double_ok_gesture {

struct VideoDevice {
    std::string path;
    std::string name;
};

struct CameraInfo {
    std::string source;
    std::string backend;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    std::string fourcc;
};

class CameraStream {
public:
    CameraStream(cv::VideoCapture capture, CameraConfig settings, CameraInfo info, cv::Mat pending_frame);
    std::optional<cv::Mat> read();
    void close();
    const CameraInfo& info() const;

private:
    cv::VideoCapture capture_;
    CameraConfig settings_;
    CameraInfo info_;
    cv::Mat pending_frame_;
    int consecutive_failures_ = 0;
};

std::string normalize_camera_source_text(const std::string& source);
std::vector<VideoDevice> list_video_devices(const std::filesystem::path& sys_class_path = "/sys/class/video4linux");
std::string format_video_devices(const std::vector<VideoDevice>& devices);
std::string format_video_devices();
CameraStream open_camera(const CameraConfig& settings);

}  // namespace double_ok_gesture
