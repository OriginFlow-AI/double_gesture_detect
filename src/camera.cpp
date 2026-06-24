#include "double_ok_gesture/camera.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace double_ok_gesture {
namespace {

bool is_decimal(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
}

std::string read_device_name(const std::filesystem::path& path) {
    std::ifstream in(path / "name");
    if (!in) {
        return "unknown";
    }
    std::string name;
    std::getline(in, name);
    return name.empty() ? "unknown" : name;
}

int video_index(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    if (name.rfind("video", 0) != 0) {
        return std::numeric_limits<int>::max();
    }
    const std::string suffix = name.substr(5);
    return is_decimal(suffix) ? std::stoi(suffix) : std::numeric_limits<int>::max();
}

cv::VideoCapture create_capture(const std::string& source) {
    if (is_decimal(source)) {
        return cv::VideoCapture(std::stoi(source), cv::CAP_V4L2);
    }
    return cv::VideoCapture(source, cv::CAP_V4L2);
}

void configure_capture(cv::VideoCapture& capture, const CameraConfig& settings) {
    if (settings.fourcc.size() != 4) {
        throw std::invalid_argument("camera fourcc must contain exactly four characters");
    }
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(
                                      settings.fourcc[0],
                                      settings.fourcc[1],
                                      settings.fourcc[2],
                                      settings.fourcc[3]));
    capture.set(cv::CAP_PROP_FRAME_WIDTH, settings.width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, settings.height);
    capture.set(cv::CAP_PROP_FPS, settings.fps);
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
}

std::optional<cv::Mat> read_warmup_frame(cv::VideoCapture& capture, int warmup_reads) {
    for (int i = 0; i < warmup_reads; ++i) {
        cv::Mat frame;
        if (capture.read(frame) && !frame.empty()) {
            return frame;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return std::nullopt;
}

std::string fourcc_text(int value) {
    std::string text;
    for (int i = 0; i < 4; ++i) {
        const char ch = static_cast<char>((value >> (8 * i)) & 0xFF);
        if (ch != '\0' && ch != ' ') {
            text.push_back(ch);
        }
    }
    return text;
}

CameraInfo camera_info(cv::VideoCapture& capture, const CameraConfig& settings) {
    CameraInfo info;
    info.source = settings.source;
    info.backend = capture.getBackendName();
    info.width = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
    info.height = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    info.fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(info.fps) || info.fps <= 0.0) {
        info.fps = settings.fps;
    }
    info.fourcc = fourcc_text(static_cast<int>(capture.get(cv::CAP_PROP_FOURCC)));
    if (info.fourcc.empty()) {
        info.fourcc = settings.fourcc;
    }
    return info;
}

}  // namespace

CameraStream::CameraStream(cv::VideoCapture capture, CameraConfig settings, CameraInfo info, cv::Mat pending_frame)
    : capture_(std::move(capture)), settings_(std::move(settings)), info_(std::move(info)),
      pending_frame_(std::move(pending_frame)) {}

std::optional<cv::Mat> CameraStream::read() {
    if (!pending_frame_.empty()) {
        cv::Mat frame = pending_frame_;
        pending_frame_.release();
        return frame;
    }

    cv::Mat frame;
    if (capture_.read(frame) && !frame.empty()) {
        consecutive_failures_ = 0;
        return frame;
    }

    ++consecutive_failures_;
    if (consecutive_failures_ >= settings_.read_failure_limit) {
        throw std::runtime_error(
            "Camera " + settings_.source + " failed to read " + std::to_string(consecutive_failures_) +
            " consecutive frames");
    }
    return std::nullopt;
}

void CameraStream::close() {
    capture_.release();
}

const CameraInfo& CameraStream::info() const {
    return info_;
}

std::string normalize_camera_source_text(const std::string& source) {
    return is_decimal(source) ? std::to_string(std::stoi(source)) : source;
}

std::vector<VideoDevice> list_video_devices(const std::filesystem::path& sys_class_path) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::is_directory(sys_class_path)) {
        return {};
    }
    for (const auto& entry : std::filesystem::directory_iterator(sys_class_path)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("video", 0) == 0) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs) {
        const int left_index = video_index(lhs);
        const int right_index = video_index(rhs);
        return left_index == right_index ? lhs.filename().string() < rhs.filename().string() : left_index < right_index;
    });

    std::vector<VideoDevice> devices;
    devices.reserve(paths.size());
    for (const auto& path : paths) {
        devices.push_back({"/dev/" + path.filename().string(), read_device_name(path)});
    }
    return devices;
}

std::string format_video_devices(const std::vector<VideoDevice>& devices) {
    if (devices.empty()) {
        return "No /dev/video* devices found";
    }
    std::ostringstream out;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (i > 0) {
            out << '\n';
        }
        out << devices[i].path << ": " << devices[i].name;
    }
    return out.str();
}

std::string format_video_devices() {
    return format_video_devices(list_video_devices());
}

CameraStream open_camera(const CameraConfig& settings) {
    if (settings.source.empty()) {
        throw std::invalid_argument("camera source must not be empty");
    }
    if (settings.width < 1 || settings.height < 1) {
        throw std::invalid_argument("camera width and height must be positive");
    }
    if (!std::isfinite(settings.fps) || settings.fps <= 0.0) {
        throw std::invalid_argument("camera fps must be finite and positive");
    }
    if (settings.open_retries < 1 || settings.warmup_reads < 1 || settings.read_failure_limit < 1) {
        throw std::invalid_argument("camera retry and read limits must be positive");
    }

    std::string last_error = "camera did not open";
    for (int attempt = 1; attempt <= settings.open_retries; ++attempt) {
        cv::VideoCapture capture = create_capture(settings.source);
        if (!capture.isOpened()) {
            last_error = "backend could not open the device";
        } else {
            configure_capture(capture, settings);
            if (auto frame = read_warmup_frame(capture, settings.warmup_reads)) {
                return CameraStream(std::move(capture), settings, camera_info(capture, settings), *frame);
            }
            last_error = "device opened but did not return a frame";
        }
        capture.release();
        if (attempt < settings.open_retries) {
            std::this_thread::sleep_for(std::chrono::duration<double>(settings.retry_delay_sec));
        }
    }

    throw std::runtime_error(
        "Cannot open camera " + settings.source + " after " + std::to_string(settings.open_retries) +
        " attempts: " + last_error + "\nAvailable devices:\n" + format_video_devices());
}

}  // namespace double_ok_gesture
